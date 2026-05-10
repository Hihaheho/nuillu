use std::future::Future;
use std::pin::pin;
use std::sync::Arc;
use std::time::Duration;

use chrono::{DateTime, Utc};
use futures::stream::{FuturesUnordered, StreamExt};
use nuillu_blackboard::{ActivationRatio, IdentityMemoryRecord};
use nuillu_module::{
    ActivateCx, AgentRuntimeControl, AllocatedModule, AllocatedModules, ModuleBatch,
    ModuleRunStatus, ports::Clock,
};
use nuillu_types::{ModuleId, ModuleInstanceId};
use thiserror::Error;
use tokio::task::{JoinHandle, spawn_local};
use tokio::time::{Instant, sleep_until};
use tracing::{Instrument as _, instrument::WithSubscriber as _};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AgentEventLoopConfig {
    pub idle_threshold: Duration,
    pub activate_retries: u8,
}

#[derive(Debug, Error)]
pub enum SchedulerError {
    #[error("module {owner} {phase} failed: {message}")]
    ModuleTaskFailed {
        owner: ModuleInstanceId,
        phase: &'static str,
        message: String,
    },
    #[error("module task panicked: {message}")]
    ModuleTaskPanicked { message: String },
}

/// Run the agent event loop until `shutdown` resolves.
///
/// The event loop owns activation policy. It starts `next_batch` only for
/// active module replicas, then runs `activate(&batch)` with retry while
/// preserving the immutable batch.
///
/// Awaited inside `tokio::task::LocalSet::run_until` (or `spawn_local`
/// on a `LocalSet`) so spawned tasks have an executor.
pub async fn run(
    modules: AllocatedModules,
    config: AgentEventLoopConfig,
    shutdown: impl Future<Output = ()>,
) -> Result<(), SchedulerError> {
    let (runtime, modules) = modules.into_parts();
    let owners = modules
        .iter()
        .map(|module| module.owner().clone())
        .collect::<Vec<_>>();
    let mut states = modules
        .into_iter()
        .map(|module| ModuleState::Stored {
            module,
            next_batch_throttle: None,
        })
        .collect::<Vec<_>>();
    let mut active = vec![false; states.len()];
    let mut tasks: FuturesUnordered<JoinHandle<TaskMessage>> = FuturesUnordered::new();
    let subscriber = tracing::dispatcher::get_default(Clone::clone);
    let parent = tracing::Span::current();

    refresh_active_and_schedule(
        &runtime,
        &owners,
        &mut active,
        &mut states,
        &mut tasks,
        config,
        &parent,
        &subscriber,
    )
    .await;

    let mut shutdown = pin!(shutdown);
    let mut idle_since: Option<Instant> = None;
    let mut idle_marker_sent = false;

    loop {
        let idle_now = is_idle(&active, &states);
        if idle_now {
            idle_since.get_or_insert_with(Instant::now);
        } else {
            idle_since = None;
            idle_marker_sent = false;
        }
        let idle_deadline = if idle_now && !idle_marker_sent {
            idle_since.map(|since| since + config.idle_threshold)
        } else {
            None
        };

        tokio::select! {
            biased;
            _ = shutdown.as_mut() => {
                for handle in tasks.iter() {
                    handle.abort();
                }
                while tasks.next().await.is_some() {}
                return Ok(());
            },
            joined = tasks.next(), if !tasks.is_empty() => {
                match joined {
                    Some(Ok(message)) => {
                        handle_task_message(
                            message,
                            &runtime,
                            &owners,
                            &mut states,
                            &mut tasks,
                            config,
                            &parent,
                            &subscriber,
                        ).await?;
                        refresh_active_and_schedule(
                            &runtime,
                            &owners,
                            &mut active,
                            &mut states,
                            &mut tasks,
                            config,
                            &parent,
                            &subscriber,
                        )
                        .await;
                    }
                    Some(Err(e)) => {
                        let message = e.to_string();
                        tracing::error!(error = ?e, "module task panicked");
                        return Err(SchedulerError::ModuleTaskPanicked { message });
                    }
                    None => {}
                }
            },
            _ = async {
                if let Some(deadline) = idle_deadline {
                    sleep_until(deadline).await;
                } else {
                    std::future::pending::<()>().await;
                }
            } => {
                let idle_for = idle_since
                    .map(|since| Instant::now().saturating_duration_since(since))
                    .unwrap_or(config.idle_threshold);
                runtime.record_agentic_deadlock_marker(idle_for).await;
                idle_marker_sent = true;
            }
        }
    }
}

enum ModuleState {
    Stored {
        module: AllocatedModule,
        next_batch_throttle: Option<NextBatchThrottle>,
    },
    CoolingDown,
    Awaiting,
    PendingBatch {
        module: AllocatedModule,
        batch: ModuleBatch,
    },
    Activating,
}

enum TaskMessage {
    BatchCooldownExpired {
        index: usize,
        module: AllocatedModule,
        delayed_for: Duration,
    },
    NextBatch {
        index: usize,
        module: AllocatedModule,
        result: Result<ModuleBatch, String>,
    },
    Activate {
        index: usize,
        module: AllocatedModule,
        activation_elapsed: Duration,
        result: Result<(), String>,
    },
}

struct NextBatchThrottle {
    not_before: DateTime<Utc>,
    activation_threshold: ActivationRatio,
}

async fn refresh_active_and_schedule(
    runtime: &AgentRuntimeControl,
    owners: &[ModuleInstanceId],
    active: &mut [bool],
    states: &mut [ModuleState],
    tasks: &mut FuturesUnordered<JoinHandle<TaskMessage>>,
    config: AgentEventLoopConfig,
    parent: &tracing::Span,
    subscriber: &tracing::Dispatch,
) {
    for (index, owner) in owners.iter().enumerate() {
        active[index] = runtime.is_active(owner).await;
    }

    for index in 0..states.len() {
        match &states[index] {
            ModuleState::Stored { .. } if active[index] => {
                runtime
                    .record_module_status(owners[index].clone(), ModuleRunStatus::AwaitingBatch)
                    .await;
                let ModuleState::Stored {
                    module,
                    next_batch_throttle,
                } = std::mem::replace(&mut states[index], ModuleState::Awaiting)
                else {
                    unreachable!("module state changed while scheduling next batch");
                };
                let now = runtime.clock().now();
                if let Some(throttle) =
                    next_batch_throttle.filter(|throttle| throttle.not_before > now)
                {
                    if let Some(activation_increase) = runtime
                        .activation_increase_waiter(&owners[index], throttle.activation_threshold)
                        .await
                    {
                        states[index] = ModuleState::CoolingDown;
                        spawn_batch_cooldown(
                            tasks,
                            index,
                            module,
                            throttle.not_before,
                            runtime.clock(),
                            activation_increase,
                            parent,
                            subscriber,
                        );
                    } else {
                        spawn_next_batch(tasks, index, module, parent, subscriber);
                    }
                } else {
                    spawn_next_batch(tasks, index, module, parent, subscriber);
                }
            }
            ModuleState::Stored { .. } => {
                runtime
                    .record_module_status(owners[index].clone(), ModuleRunStatus::Inactive)
                    .await;
            }
            ModuleState::CoolingDown | ModuleState::Awaiting => {
                let status = if active[index] {
                    ModuleRunStatus::AwaitingBatch
                } else {
                    ModuleRunStatus::Inactive
                };
                runtime
                    .record_module_status(owners[index].clone(), status)
                    .await;
            }
            ModuleState::PendingBatch { .. } if active[index] => {
                runtime
                    .record_module_status(owners[index].clone(), ModuleRunStatus::Activating)
                    .await;
                let ModuleState::PendingBatch { module, batch } =
                    std::mem::replace(&mut states[index], ModuleState::Activating)
                else {
                    unreachable!("module state changed while scheduling activation");
                };
                let catalog = runtime.module_catalog();
                let identity_memories = runtime.identity_memories().await;
                spawn_activate(
                    tasks,
                    index,
                    module,
                    batch,
                    config,
                    runtime.clone(),
                    catalog,
                    identity_memories,
                    parent,
                    subscriber,
                );
            }
            ModuleState::PendingBatch { .. } => {
                runtime
                    .record_module_status(owners[index].clone(), ModuleRunStatus::PendingBatch)
                    .await;
            }
            ModuleState::Activating => {
                runtime
                    .record_module_status(owners[index].clone(), ModuleRunStatus::Activating)
                    .await;
            }
        }
    }
}

async fn handle_task_message(
    message: TaskMessage,
    runtime: &AgentRuntimeControl,
    owners: &[ModuleInstanceId],
    states: &mut [ModuleState],
    tasks: &mut FuturesUnordered<JoinHandle<TaskMessage>>,
    config: AgentEventLoopConfig,
    parent: &tracing::Span,
    subscriber: &tracing::Dispatch,
) -> Result<(), SchedulerError> {
    match message {
        TaskMessage::BatchCooldownExpired {
            index,
            module,
            delayed_for,
        } => {
            runtime
                .record_module_batch_throttled(owners[index].clone(), delayed_for)
                .await;
            states[index] = ModuleState::Stored {
                module,
                next_batch_throttle: None,
            };
            Ok(())
        }
        TaskMessage::NextBatch {
            index,
            module,
            result,
        } => match result {
            Ok(batch) => {
                if runtime.is_active(&owners[index]).await {
                    runtime
                        .record_module_status(owners[index].clone(), ModuleRunStatus::Activating)
                        .await;
                    states[index] = ModuleState::Activating;
                    let catalog = runtime.module_catalog();
                    let identity_memories = runtime.identity_memories().await;
                    spawn_activate(
                        tasks,
                        index,
                        module,
                        batch,
                        config,
                        runtime.clone(),
                        catalog,
                        identity_memories,
                        parent,
                        subscriber,
                    );
                } else {
                    runtime
                        .record_module_status(owners[index].clone(), ModuleRunStatus::PendingBatch)
                        .await;
                    states[index] = ModuleState::PendingBatch { module, batch };
                }
                Ok(())
            }
            Err(message) => {
                runtime
                    .record_module_status(
                        owners[index].clone(),
                        ModuleRunStatus::Failed {
                            phase: "next_batch".to_string(),
                            message: message.clone(),
                        },
                    )
                    .await;
                Err(SchedulerError::ModuleTaskFailed {
                    owner: owners[index].clone(),
                    phase: "next_batch",
                    message,
                })
            }
        },
        TaskMessage::Activate {
            index,
            module,
            activation_elapsed,
            result,
        } => match result {
            Ok(()) => {
                let now = runtime.clock().now();
                let next_batch_throttle = runtime
                    .module_batch_throttle_baseline(&owners[index])
                    .await
                    .and_then(|(interval, activation_threshold)| {
                        let remaining = interval.saturating_sub(activation_elapsed);
                        if remaining.is_zero() {
                            None
                        } else {
                            chrono::Duration::from_std(remaining)
                                .ok()
                                .map(|d| NextBatchThrottle {
                                    not_before: now + d,
                                    activation_threshold,
                                })
                        }
                    });
                states[index] = ModuleState::Stored {
                    module,
                    next_batch_throttle,
                };
                Ok(())
            }
            Err(message) => {
                runtime
                    .record_module_status(
                        owners[index].clone(),
                        ModuleRunStatus::Failed {
                            phase: "activate".to_string(),
                            message: message.clone(),
                        },
                    )
                    .await;
                Err(SchedulerError::ModuleTaskFailed {
                    owner: owners[index].clone(),
                    phase: "activate",
                    message,
                })
            }
        },
    }
}

fn spawn_next_batch(
    tasks: &mut FuturesUnordered<JoinHandle<TaskMessage>>,
    index: usize,
    mut module: AllocatedModule,
    parent: &tracing::Span,
    subscriber: &tracing::Dispatch,
) {
    tasks.push(spawn_local(
        async move {
            let result = module
                .next_batch()
                .await
                .map_err(|error| format!("{error:#}"));
            TaskMessage::NextBatch {
                index,
                module,
                result,
            }
        }
        .instrument(parent.clone())
        .with_subscriber(subscriber.clone()),
    ));
}

fn spawn_batch_cooldown(
    tasks: &mut FuturesUnordered<JoinHandle<TaskMessage>>,
    index: usize,
    module: AllocatedModule,
    deadline: DateTime<Utc>,
    clock: Arc<dyn Clock>,
    activation_increase: tokio::sync::oneshot::Receiver<()>,
    parent: &tracing::Span,
    subscriber: &tracing::Dispatch,
) {
    tasks.push(spawn_local(
        async move {
            let started = clock.now();
            tokio::select! {
                _ = clock.sleep_until(deadline) => {}
                _ = activation_increase => {}
            }
            let delayed_for = (clock.now() - started).to_std().unwrap_or(Duration::ZERO);
            TaskMessage::BatchCooldownExpired {
                index,
                module,
                delayed_for,
            }
        }
        .instrument(parent.clone())
        .with_subscriber(subscriber.clone()),
    ));
}

fn spawn_activate(
    tasks: &mut FuturesUnordered<JoinHandle<TaskMessage>>,
    index: usize,
    module: AllocatedModule,
    batch: ModuleBatch,
    config: AgentEventLoopConfig,
    runtime: AgentRuntimeControl,
    catalog: Vec<(ModuleId, &'static str)>,
    identity_memories: Vec<IdentityMemoryRecord>,
    parent: &tracing::Span,
    subscriber: &tracing::Dispatch,
) {
    tasks.push(spawn_local(
        async move {
            let activation_started = Instant::now();
            let (module, result) = activate_with_retries(
                module,
                &runtime,
                &catalog,
                &identity_memories,
                &batch,
                config.activate_retries,
            )
            .await;
            let activation_elapsed = activation_started.elapsed();
            TaskMessage::Activate {
                index,
                module,
                activation_elapsed,
                result,
            }
        }
        .instrument(parent.clone())
        .with_subscriber(subscriber.clone()),
    ));
}

async fn activate_with_retries(
    mut module: AllocatedModule,
    runtime: &AgentRuntimeControl,
    catalog: &[(ModuleId, &'static str)],
    identity_memories: &[IdentityMemoryRecord],
    batch: &ModuleBatch,
    activate_retries: u8,
) -> (AllocatedModule, Result<(), String>) {
    let cx = ActivateCx::new(
        catalog,
        identity_memories,
        runtime.session_compaction_lutum(),
    );
    let mut retries = 0_u8;
    loop {
        match module.activate(&cx, batch).await {
            Ok(()) => return (module, Ok(())),
            Err(error) if retries < activate_retries => {
                retries = retries.saturating_add(1);
                tracing::warn!(
                    owner = %module.owner(),
                    retries,
                    max_retries = activate_retries,
                    error = %error,
                    "module activation failed; retrying"
                );
            }
            Err(error) => return (module, Err(format!("{error:#}"))),
        }
    }
}

fn is_idle(active: &[bool], states: &[ModuleState]) -> bool {
    let mut active_count = 0_usize;
    for (is_active, state) in active.iter().copied().zip(states) {
        if !is_active {
            continue;
        }
        active_count += 1;
        if !matches!(state, ModuleState::Awaiting) {
            return false;
        }
    }
    active_count > 0
}

#[cfg(test)]
mod tests {
    use super::{AgentEventLoopConfig, SchedulerError};

    use std::cell::{Cell, RefCell};
    use std::collections::VecDeque;
    use std::rc::Rc;
    use std::time::Duration;

    use async_trait::async_trait;
    use nuillu_blackboard::{
        ActivationRatio, Blackboard, BlackboardCommand, Bpm, IdentityMemoryRecord, ModuleConfig,
        ModuleRunStatus, ResourceAllocation, linear_ratio_fn,
    };
    use nuillu_module::{
        CognitionLogUpdated, CognitionLogUpdatedInbox, CognitionWriter, Memo, Module,
        ModuleRegistry, QueryInbox, QueryRequest,
    };
    use nuillu_types::{MemoryContent, MemoryIndex, ModelTier, ModuleId, builtin};
    use tokio::sync::oneshot;
    use tokio::task::LocalSet;

    use crate::testing::{test_caps, test_caps_with_real_clock};

    fn test_config() -> AgentEventLoopConfig {
        AgentEventLoopConfig {
            idle_threshold: std::time::Duration::from_millis(50),
            activate_retries: 2,
        }
    }

    fn echo_id() -> ModuleId {
        ModuleId::new("echo").unwrap()
    }

    struct EchoModule {
        query_inbox: QueryInbox,
        memo: Memo,
        on_done: Option<oneshot::Sender<()>>,
    }

    #[async_trait(?Send)]
    impl Module for EchoModule {
        type Batch = QueryRequest;

        fn id() -> &'static str {
            "echo"
        }

        fn role_description() -> &'static str {
            "test stub"
        }

        async fn next_batch(&mut self) -> anyhow::Result<Self::Batch> {
            Ok(self.query_inbox.next_item().await?.body)
        }

        async fn activate(
            &mut self,
            _cx: &nuillu_module::ActivateCx<'_>,
            batch: &Self::Batch,
        ) -> anyhow::Result<()> {
            self.memo.write(format!("echoed {}", batch.question)).await;
            if let Some(tx) = self.on_done.take() {
                let _ = tx.send(());
            }
            Ok(())
        }
    }

    struct QueryBatchRecorder {
        query_inbox: QueryInbox,
        batches: Rc<RefCell<Vec<Vec<String>>>>,
        first_done: Option<oneshot::Sender<()>>,
        second_done: Option<oneshot::Sender<()>>,
    }

    #[async_trait(?Send)]
    impl Module for QueryBatchRecorder {
        type Batch = Vec<String>;

        fn id() -> &'static str {
            "batch-recorder"
        }

        fn role_description() -> &'static str {
            "test stub"
        }

        async fn next_batch(&mut self) -> anyhow::Result<Self::Batch> {
            let first = self.query_inbox.next_item().await?.body.question;
            let mut batch = vec![first];
            batch.extend(
                self.query_inbox
                    .take_ready_items()?
                    .items
                    .into_iter()
                    .map(|envelope| envelope.body.question),
            );
            Ok(batch)
        }

        async fn activate(
            &mut self,
            _cx: &nuillu_module::ActivateCx<'_>,
            batch: &Self::Batch,
        ) -> anyhow::Result<()> {
            self.batches.borrow_mut().push(batch.clone());
            match self.batches.borrow().len() {
                1 => {
                    if let Some(done) = self.first_done.take() {
                        let _ = done.send(());
                    }
                }
                2 => {
                    if let Some(done) = self.second_done.take() {
                        let _ = done.send(());
                    }
                }
                _ => {}
            }
            Ok(())
        }
    }

    struct TimedQueryBatchRecorder {
        query_inbox: QueryInbox,
        batches: Rc<RefCell<Vec<Vec<String>>>>,
        activation_delays: Rc<RefCell<VecDeque<Duration>>>,
        first_done: Option<oneshot::Sender<()>>,
        second_done: Option<oneshot::Sender<()>>,
    }

    #[async_trait(?Send)]
    impl Module for TimedQueryBatchRecorder {
        type Batch = Vec<String>;

        fn id() -> &'static str {
            "timed-batch-recorder"
        }

        fn role_description() -> &'static str {
            "test stub"
        }

        async fn next_batch(&mut self) -> anyhow::Result<Self::Batch> {
            let first = self.query_inbox.next_item().await?.body.question;
            let mut batch = vec![first];
            batch.extend(
                self.query_inbox
                    .take_ready_items()?
                    .items
                    .into_iter()
                    .map(|envelope| envelope.body.question),
            );
            Ok(batch)
        }

        async fn activate(
            &mut self,
            _cx: &nuillu_module::ActivateCx<'_>,
            batch: &Self::Batch,
        ) -> anyhow::Result<()> {
            let activation_delay = self
                .activation_delays
                .borrow_mut()
                .pop_front()
                .unwrap_or_default();
            if !activation_delay.is_zero() {
                tokio::time::sleep(activation_delay).await;
            }
            self.batches.borrow_mut().push(batch.clone());
            match self.batches.borrow().len() {
                1 => {
                    if let Some(done) = self.first_done.take() {
                        let _ = done.send(());
                    }
                }
                2 => {
                    if let Some(done) = self.second_done.take() {
                        let _ = done.send(());
                    }
                }
                _ => {}
            }
            Ok(())
        }
    }

    struct CognitionGateStub {
        writer: CognitionWriter,
        on_done: Option<oneshot::Sender<()>>,
    }

    #[async_trait(?Send)]
    impl Module for CognitionGateStub {
        type Batch = ();

        fn id() -> &'static str {
            "cognition-gate"
        }

        fn role_description() -> &'static str {
            "test stub"
        }

        async fn next_batch(&mut self) -> anyhow::Result<Self::Batch> {
            if self.on_done.is_some() {
                Ok(())
            } else {
                std::future::pending().await
            }
        }

        async fn activate(
            &mut self,
            _cx: &nuillu_module::ActivateCx<'_>,
            _batch: &Self::Batch,
        ) -> anyhow::Result<()> {
            self.writer.append("novel-event").await;
            if let Some(tx) = self.on_done.take() {
                let _ = tx.send(());
            }
            Ok(())
        }
    }

    struct RetryStub {
        memo: Memo,
        attempts: Rc<Cell<u8>>,
        on_done: Option<oneshot::Sender<()>>,
        batch_sent: bool,
    }

    #[async_trait(?Send)]
    impl Module for RetryStub {
        type Batch = String;

        fn id() -> &'static str {
            "retry"
        }

        fn role_description() -> &'static str {
            "test stub"
        }

        async fn next_batch(&mut self) -> anyhow::Result<Self::Batch> {
            if self.batch_sent {
                std::future::pending().await
            } else {
                self.batch_sent = true;
                Ok("stable-batch".into())
            }
        }

        async fn activate(
            &mut self,
            _cx: &nuillu_module::ActivateCx<'_>,
            batch: &Self::Batch,
        ) -> anyhow::Result<()> {
            let attempt = self.attempts.get().saturating_add(1);
            self.attempts.set(attempt);
            if attempt < 3 {
                anyhow::bail!("transient activation failure");
            }
            self.memo
                .write(format!("attempt {attempt} handled {batch}"))
                .await;
            if let Some(tx) = self.on_done.take() {
                let _ = tx.send(());
            }
            Ok(())
        }
    }

    struct AlwaysFailStub {
        batch_sent: bool,
    }

    #[async_trait(?Send)]
    impl Module for AlwaysFailStub {
        type Batch = ();

        fn id() -> &'static str {
            "always-fail"
        }

        fn role_description() -> &'static str {
            "test stub"
        }

        async fn next_batch(&mut self) -> anyhow::Result<Self::Batch> {
            if self.batch_sent {
                std::future::pending().await
            } else {
                self.batch_sent = true;
                Ok(())
            }
        }

        async fn activate(
            &mut self,
            _cx: &nuillu_module::ActivateCx<'_>,
            _batch: &Self::Batch,
        ) -> anyhow::Result<()> {
            anyhow::bail!("permanent activation failure")
        }
    }

    struct IdentityCxObserver {
        seen: Rc<RefCell<Vec<(String, String)>>>,
        on_done: Option<oneshot::Sender<()>>,
        batch_sent: bool,
    }

    #[async_trait(?Send)]
    impl Module for IdentityCxObserver {
        type Batch = ();

        fn id() -> &'static str {
            "identity-cx-observer"
        }

        fn role_description() -> &'static str {
            "test stub"
        }

        async fn next_batch(&mut self) -> anyhow::Result<Self::Batch> {
            if self.batch_sent {
                std::future::pending().await
            } else {
                self.batch_sent = true;
                Ok(())
            }
        }

        async fn activate(
            &mut self,
            cx: &nuillu_module::ActivateCx<'_>,
            _batch: &Self::Batch,
        ) -> anyhow::Result<()> {
            *self.seen.borrow_mut() = cx
                .identity_memories()
                .iter()
                .map(|record| {
                    (
                        record.index.as_str().to_owned(),
                        record.content.as_str().to_owned(),
                    )
                })
                .collect();
            if let Some(tx) = self.on_done.take() {
                let _ = tx.send(());
            }
            Ok(())
        }
    }

    struct DeadlockObserver {
        updates: CognitionLogUpdatedInbox,
        memo: Memo,
        on_done: Option<oneshot::Sender<()>>,
    }

    #[async_trait(?Send)]
    impl Module for DeadlockObserver {
        type Batch = CognitionLogUpdated;

        fn id() -> &'static str {
            "deadlock-observer"
        }

        fn role_description() -> &'static str {
            "test stub"
        }

        async fn next_batch(&mut self) -> anyhow::Result<Self::Batch> {
            Ok(self.updates.next_item().await?.body)
        }

        async fn activate(
            &mut self,
            _cx: &nuillu_module::ActivateCx<'_>,
            batch: &Self::Batch,
        ) -> anyhow::Result<()> {
            assert_eq!(batch, &CognitionLogUpdated::AgenticDeadlockMarker);
            self.memo.write("observed deadlock marker").await;
            if let Some(tx) = self.on_done.take() {
                let _ = tx.send(());
            }
            Ok(())
        }
    }

    struct HangingBatchStub;

    #[async_trait(?Send)]
    impl Module for HangingBatchStub {
        type Batch = ();

        fn id() -> &'static str {
            "status-awaiting"
        }

        fn role_description() -> &'static str {
            "test stub"
        }

        async fn next_batch(&mut self) -> anyhow::Result<Self::Batch> {
            std::future::pending().await
        }

        async fn activate(
            &mut self,
            _cx: &nuillu_module::ActivateCx<'_>,
            _batch: &Self::Batch,
        ) -> anyhow::Result<()> {
            unreachable!("hanging batch stub never produces a batch")
        }
    }

    #[allow(dead_code)]
    struct DelayedBatchStub {
        release: Option<oneshot::Receiver<()>>,
        activated: Rc<Cell<bool>>,
    }

    #[async_trait(?Send)]
    impl Module for DelayedBatchStub {
        type Batch = ();

        fn id() -> &'static str {
            "status-pending"
        }

        fn role_description() -> &'static str {
            "test stub"
        }

        async fn next_batch(&mut self) -> anyhow::Result<Self::Batch> {
            if let Some(release) = self.release.take() {
                let _ = release.await;
                Ok(())
            } else {
                std::future::pending().await
            }
        }

        async fn activate(
            &mut self,
            _cx: &nuillu_module::ActivateCx<'_>,
            _batch: &Self::Batch,
        ) -> anyhow::Result<()> {
            self.activated.set(true);
            Ok(())
        }
    }

    struct BlockingActivateStub {
        entered: Option<oneshot::Sender<()>>,
        release: Option<oneshot::Receiver<()>>,
        batch_sent: bool,
    }

    #[async_trait(?Send)]
    impl Module for BlockingActivateStub {
        type Batch = ();

        fn id() -> &'static str {
            "status-activating"
        }

        fn role_description() -> &'static str {
            "test stub"
        }

        async fn next_batch(&mut self) -> anyhow::Result<Self::Batch> {
            if self.batch_sent {
                std::future::pending().await
            } else {
                self.batch_sent = true;
                Ok(())
            }
        }

        async fn activate(
            &mut self,
            _cx: &nuillu_module::ActivateCx<'_>,
            _batch: &Self::Batch,
        ) -> anyhow::Result<()> {
            if let Some(entered) = self.entered.take() {
                let _ = entered.send(());
            }
            if let Some(release) = self.release.take() {
                let _ = release.await;
            }
            Ok(())
        }
    }

    async fn wait_for_status(
        blackboard: &Blackboard,
        owner: &nuillu_types::ModuleInstanceId,
        expected: ModuleRunStatus,
    ) {
        for _ in 0..50 {
            let status = blackboard
                .read(|bb| bb.module_status_for_instance(owner).cloned())
                .await;
            if status.as_ref() == Some(&expected) {
                return;
            }
            tokio::task::yield_now().await;
            tokio::time::sleep(std::time::Duration::from_millis(1)).await;
        }
        let status = blackboard
            .read(|bb| bb.module_status_for_instance(owner).cloned())
            .await;
        panic!("expected status {expected:?}, got {status:?}");
    }

    #[tokio::test(flavor = "current_thread", start_paused = false)]
    async fn typed_query_fanout_reaches_active_module() {
        let local = LocalSet::new();
        local
            .run_until(async {
                let mut alloc = ResourceAllocation::default();
                alloc.set(
                    echo_id(),
                    ModuleConfig {
                        tier: ModelTier::Default,
                        ..Default::default()
                    },
                );
                alloc.set_activation(echo_id(), ActivationRatio::ONE);

                let blackboard = Blackboard::with_allocation(alloc);
                let caps = test_caps(blackboard.clone());
                let (done_tx, done_rx) = oneshot::channel();
                let done_tx = Rc::new(RefCell::new(Some(done_tx)));
                let modules = ModuleRegistry::new()
                    .register(
                        0..=0,
                        Bpm::from_f64(60.0)..=Bpm::from_f64(60.0),
                        linear_ratio_fn,
                        {
                            let done_tx = Rc::clone(&done_tx);
                            move |caps| EchoModule {
                                query_inbox: caps.query_inbox(),
                                memo: caps.memo(),
                                on_done: done_tx.borrow_mut().take(),
                            }
                        },
                    )
                    .unwrap()
                    .build(&caps)
                    .await
                    .unwrap();
                let mailbox = caps.internal_harness_io().query_mailbox();

                super::run(modules, test_config(), async move {
                    mailbox
                        .publish(QueryRequest::new("ping"))
                        .await
                        .expect("query should route to echo");
                    let _ = done_rx.await;
                    for _ in 0..4 {
                        tokio::task::yield_now().await;
                    }
                })
                .await
                .expect("scheduler returned err");

                blackboard
                    .read(|bb| {
                        assert!(
                            bb.recent_memo_logs().iter().any(|record| {
                                record.owner.module == echo_id() && record.content == "echoed ping"
                            }),
                            "echo memo log missing"
                        );
                        assert_eq!(
                            bb.cognition_log().len(),
                            0,
                            "echo does not hold CognitionWriter",
                        );
                    })
                    .await;
            })
            .await;
    }

    #[tokio::test(flavor = "current_thread", start_paused = false)]
    async fn scheduler_passes_identity_memories_into_activate_context() {
        let local = LocalSet::new();
        local
            .run_until(async {
                let module_id = ModuleId::new(IdentityCxObserver::id()).unwrap();
                let mut alloc = ResourceAllocation::default();
                alloc.set_activation(module_id.clone(), ActivationRatio::ONE);

                let blackboard = Blackboard::with_allocation(alloc);
                let caps = test_caps(blackboard.clone());
                let seen = Rc::new(RefCell::new(Vec::new()));
                let (done_tx, done_rx) = oneshot::channel();
                let done_tx = Rc::new(RefCell::new(Some(done_tx)));
                let modules = ModuleRegistry::new()
                    .register(
                        0..=0,
                        Bpm::from_f64(60.0)..=Bpm::from_f64(60.0),
                        linear_ratio_fn,
                        {
                            let seen = Rc::clone(&seen);
                            let done_tx = Rc::clone(&done_tx);
                            move |_caps| IdentityCxObserver {
                                seen: Rc::clone(&seen),
                                on_done: done_tx.borrow_mut().take(),
                                batch_sent: false,
                            }
                        },
                    )
                    .unwrap()
                    .build(&caps)
                    .await
                    .unwrap();
                blackboard
                    .apply(BlackboardCommand::SetIdentityMemories(vec![
                        IdentityMemoryRecord {
                            index: MemoryIndex::new("identity-1"),
                            content: MemoryContent::new("The agent is named Nuillu."),
                        },
                    ]))
                    .await;

                super::run(modules, test_config(), async move {
                    let _ = done_rx.await;
                    for _ in 0..4 {
                        tokio::task::yield_now().await;
                    }
                })
                .await
                .expect("scheduler returned err");

                assert_eq!(
                    seen.borrow().as_slice(),
                    &[(
                        "identity-1".to_string(),
                        "The agent is named Nuillu.".to_string()
                    )]
                );
            })
            .await;
    }

    #[tokio::test(flavor = "current_thread", start_paused = false)]
    async fn runtime_batch_throttle_delays_next_batch_start_and_coalesces_ready_work() {
        let local = LocalSet::new();
        local
            .run_until(async {
                let module_id = ModuleId::new(QueryBatchRecorder::id()).unwrap();
                let mut alloc = ResourceAllocation::default();
                alloc.set(
                    module_id.clone(),
                    ModuleConfig {
                        tier: ModelTier::Default,
                        ..Default::default()
                    },
                );
                alloc.set_activation(module_id.clone(), ActivationRatio::ONE);

                let blackboard = Blackboard::with_allocation(alloc);
                // Use real wall-clock so the test can observe the 30ms cooldown
                // actually delaying the second batch.
                let caps = test_caps_with_real_clock(blackboard);
                let batches = Rc::new(RefCell::new(Vec::<Vec<String>>::new()));
                let (first_tx, first_rx) = oneshot::channel();
                let (second_tx, second_rx) = oneshot::channel();
                let first_tx = Rc::new(RefCell::new(Some(first_tx)));
                let second_tx = Rc::new(RefCell::new(Some(second_tx)));
                // 2000 BPM = 30ms cooldown per batch under linear_ratio_fn at
                // activation_ratio=1.0.
                let modules = ModuleRegistry::new()
                    .register(
                        0..=0,
                        Bpm::from_f64(2000.0)..=Bpm::from_f64(2000.0),
                        linear_ratio_fn,
                        {
                            let batches = Rc::clone(&batches);
                            let first_tx = Rc::clone(&first_tx);
                            let second_tx = Rc::clone(&second_tx);
                            move |caps| QueryBatchRecorder {
                                query_inbox: caps.query_inbox(),
                                batches: Rc::clone(&batches),
                                first_done: first_tx.borrow_mut().take(),
                                second_done: second_tx.borrow_mut().take(),
                            }
                        },
                    )
                    .unwrap()
                    .build(&caps)
                    .await
                    .unwrap();
                let mailbox = caps.internal_harness_io().query_mailbox();

                super::run(modules, test_config(), async move {
                    mailbox
                        .publish(QueryRequest::new("first"))
                        .await
                        .expect("first query should route");
                    let _ = first_rx.await;

                    let started = tokio::time::Instant::now();
                    mailbox
                        .publish(QueryRequest::new("second"))
                        .await
                        .expect("second query should route");
                    mailbox
                        .publish(QueryRequest::new("third"))
                        .await
                        .expect("third query should route");
                    let _ = second_rx.await;
                    assert!(started.elapsed() >= std::time::Duration::from_millis(20));
                })
                .await
                .expect("scheduler returned err");

                assert_eq!(
                    batches.borrow().as_slice(),
                    &[
                        vec!["first".to_string()],
                        vec!["second".to_string(), "third".to_string()]
                    ]
                );
            })
            .await;
    }

    #[tokio::test(flavor = "current_thread", start_paused = false)]
    async fn runtime_batch_throttle_subtracts_activation_elapsed_from_sleep() {
        let local = LocalSet::new();
        local
            .run_until(async {
                let module_id = ModuleId::new(TimedQueryBatchRecorder::id()).unwrap();
                let mut alloc = ResourceAllocation::default();
                alloc.set(
                    module_id.clone(),
                    ModuleConfig {
                        tier: ModelTier::Default,
                        ..Default::default()
                    },
                );
                alloc.set_activation(module_id.clone(), ActivationRatio::ONE);

                let blackboard = Blackboard::with_allocation(alloc);
                let caps = test_caps_with_real_clock(blackboard);
                let batches = Rc::new(RefCell::new(Vec::<Vec<String>>::new()));
                let activation_delays = Rc::new(RefCell::new(VecDeque::from([
                    Duration::from_millis(80),
                    Duration::ZERO,
                ])));
                let (first_tx, first_rx) = oneshot::channel();
                let (second_tx, second_rx) = oneshot::channel();
                let first_tx = Rc::new(RefCell::new(Some(first_tx)));
                let second_tx = Rc::new(RefCell::new(Some(second_tx)));
                // 300 BPM = 200ms target period between activation starts.
                // The first activation takes ~80ms, so remaining cooldown
                // should be ~120ms rather than a full 200ms.
                let modules = ModuleRegistry::new()
                    .register(
                        0..=0,
                        Bpm::from_f64(300.0)..=Bpm::from_f64(300.0),
                        linear_ratio_fn,
                        {
                            let batches = Rc::clone(&batches);
                            let activation_delays = Rc::clone(&activation_delays);
                            let first_tx = Rc::clone(&first_tx);
                            let second_tx = Rc::clone(&second_tx);
                            move |caps| TimedQueryBatchRecorder {
                                query_inbox: caps.query_inbox(),
                                batches: Rc::clone(&batches),
                                activation_delays: Rc::clone(&activation_delays),
                                first_done: first_tx.borrow_mut().take(),
                                second_done: second_tx.borrow_mut().take(),
                            }
                        },
                    )
                    .unwrap()
                    .build(&caps)
                    .await
                    .unwrap();
                let mailbox = caps.internal_harness_io().query_mailbox();

                super::run(modules, test_config(), async move {
                    mailbox
                        .publish(QueryRequest::new("first"))
                        .await
                        .expect("first query should route");
                    let _ = first_rx.await;

                    let started = tokio::time::Instant::now();
                    mailbox
                        .publish(QueryRequest::new("second"))
                        .await
                        .expect("second query should route");
                    let _ = second_rx.await;
                    let elapsed = started.elapsed();
                    assert!(elapsed >= Duration::from_millis(90), "{elapsed:?}");
                    assert!(elapsed < Duration::from_millis(190), "{elapsed:?}");
                })
                .await
                .expect("scheduler returned err");

                assert_eq!(
                    batches.borrow().as_slice(),
                    &[vec!["first".to_string()], vec!["second".to_string()]]
                );
            })
            .await;
    }

    #[tokio::test(flavor = "current_thread", start_paused = false)]
    async fn runtime_batch_throttle_skips_sleep_when_activation_exceeds_period() {
        let local = LocalSet::new();
        local
            .run_until(async {
                let module_id = ModuleId::new(TimedQueryBatchRecorder::id()).unwrap();
                let mut alloc = ResourceAllocation::default();
                alloc.set(
                    module_id.clone(),
                    ModuleConfig {
                        tier: ModelTier::Default,
                        ..Default::default()
                    },
                );
                alloc.set_activation(module_id.clone(), ActivationRatio::ONE);

                let blackboard = Blackboard::with_allocation(alloc);
                let caps = test_caps_with_real_clock(blackboard);
                let batches = Rc::new(RefCell::new(Vec::<Vec<String>>::new()));
                let activation_delays = Rc::new(RefCell::new(VecDeque::from([
                    Duration::from_millis(160),
                    Duration::ZERO,
                ])));
                let (first_tx, first_rx) = oneshot::channel();
                let (second_tx, second_rx) = oneshot::channel();
                let first_tx = Rc::new(RefCell::new(Some(first_tx)));
                let second_tx = Rc::new(RefCell::new(Some(second_tx)));
                // 500 BPM = 120ms target period. Since the first activation
                // takes longer than that, no extra cooldown should be added.
                let modules = ModuleRegistry::new()
                    .register(
                        0..=0,
                        Bpm::from_f64(500.0)..=Bpm::from_f64(500.0),
                        linear_ratio_fn,
                        {
                            let batches = Rc::clone(&batches);
                            let activation_delays = Rc::clone(&activation_delays);
                            let first_tx = Rc::clone(&first_tx);
                            let second_tx = Rc::clone(&second_tx);
                            move |caps| TimedQueryBatchRecorder {
                                query_inbox: caps.query_inbox(),
                                batches: Rc::clone(&batches),
                                activation_delays: Rc::clone(&activation_delays),
                                first_done: first_tx.borrow_mut().take(),
                                second_done: second_tx.borrow_mut().take(),
                            }
                        },
                    )
                    .unwrap()
                    .build(&caps)
                    .await
                    .unwrap();
                let mailbox = caps.internal_harness_io().query_mailbox();

                super::run(modules, test_config(), async move {
                    mailbox
                        .publish(QueryRequest::new("first"))
                        .await
                        .expect("first query should route");
                    let _ = first_rx.await;

                    let started = tokio::time::Instant::now();
                    mailbox
                        .publish(QueryRequest::new("second"))
                        .await
                        .expect("second query should route");
                    let _ = second_rx.await;
                    assert!(started.elapsed() < Duration::from_millis(80));
                })
                .await
                .expect("scheduler returned err");

                assert_eq!(
                    batches.borrow().as_slice(),
                    &[vec!["first".to_string()], vec!["second".to_string()]]
                );
            })
            .await;
    }

    #[tokio::test(flavor = "current_thread", start_paused = false)]
    async fn runtime_batch_throttle_wakes_when_activation_increases() {
        let local = LocalSet::new();
        local
            .run_until(async {
                let module_id = ModuleId::new(QueryBatchRecorder::id()).unwrap();
                let mut alloc = ResourceAllocation::default();
                alloc.set(
                    module_id.clone(),
                    ModuleConfig {
                        tier: ModelTier::Default,
                        ..Default::default()
                    },
                );
                alloc.set_activation(module_id.clone(), ActivationRatio::from_f64(0.5));

                let blackboard = Blackboard::with_allocation(alloc);
                let caps = test_caps_with_real_clock(blackboard.clone());
                let batches = Rc::new(RefCell::new(Vec::<Vec<String>>::new()));
                let (first_tx, first_rx) = oneshot::channel();
                let (second_tx, second_rx) = oneshot::channel();
                let first_tx = Rc::new(RefCell::new(Some(first_tx)));
                let second_tx = Rc::new(RefCell::new(Some(second_tx)));
                // Fixed 120 BPM = 500ms period. The test bumps activation
                // during cooldown and expects the second batch before deadline.
                let modules = ModuleRegistry::new()
                    .register(
                        0..=0,
                        Bpm::from_f64(120.0)..=Bpm::from_f64(120.0),
                        linear_ratio_fn,
                        {
                            let batches = Rc::clone(&batches);
                            let first_tx = Rc::clone(&first_tx);
                            let second_tx = Rc::clone(&second_tx);
                            move |caps| QueryBatchRecorder {
                                query_inbox: caps.query_inbox(),
                                batches: Rc::clone(&batches),
                                first_done: first_tx.borrow_mut().take(),
                                second_done: second_tx.borrow_mut().take(),
                            }
                        },
                    )
                    .unwrap()
                    .build(&caps)
                    .await
                    .unwrap();
                let mailbox = caps.internal_harness_io().query_mailbox();

                super::run(modules, test_config(), async move {
                    mailbox
                        .publish(QueryRequest::new("first"))
                        .await
                        .expect("first query should route");
                    let _ = first_rx.await;

                    mailbox
                        .publish(QueryRequest::new("second"))
                        .await
                        .expect("second query should route");
                    tokio::time::sleep(Duration::from_millis(50)).await;

                    let mut raised = ResourceAllocation::default();
                    raised.set(
                        module_id.clone(),
                        ModuleConfig {
                            tier: ModelTier::Default,
                            ..Default::default()
                        },
                    );
                    raised.set_activation(module_id, ActivationRatio::ONE);
                    let started = tokio::time::Instant::now();
                    blackboard
                        .apply(BlackboardCommand::SetAllocation(raised))
                        .await;

                    let _ = second_rx.await;
                    assert!(started.elapsed() < Duration::from_millis(120));
                })
                .await
                .expect("scheduler returned err");

                assert_eq!(
                    batches.borrow().as_slice(),
                    &[vec!["first".to_string()], vec!["second".to_string()]]
                );
            })
            .await;
    }

    #[tokio::test(flavor = "current_thread", start_paused = false)]
    async fn scheduler_records_awaiting_batch_status() {
        let local = LocalSet::new();
        local
            .run_until(async {
                let module_id = ModuleId::new(HangingBatchStub::id()).unwrap();
                let owner = nuillu_types::ModuleInstanceId::new(
                    module_id.clone(),
                    nuillu_types::ReplicaIndex::ZERO,
                );
                let mut alloc = ResourceAllocation::default();
                alloc.set(
                    module_id.clone(),
                    ModuleConfig {
                        tier: ModelTier::Default,
                        ..Default::default()
                    },
                );
                alloc.set_activation(module_id.clone(), ActivationRatio::ONE);

                let blackboard = Blackboard::with_allocation(alloc);
                let caps = test_caps(blackboard.clone());
                let modules = ModuleRegistry::new()
                    .register(
                        0..=0,
                        Bpm::from_f64(60.0)..=Bpm::from_f64(60.0),
                        linear_ratio_fn,
                        |_| HangingBatchStub,
                    )
                    .unwrap()
                    .build(&caps)
                    .await
                    .unwrap();

                super::run(modules, test_config(), async {
                    wait_for_status(&blackboard, &owner, ModuleRunStatus::AwaitingBatch).await;
                })
                .await
                .expect("scheduler returned err");

                assert_eq!(
                    blackboard
                        .read(|bb| bb.module_status_for_instance(&owner).cloned())
                        .await,
                    Some(ModuleRunStatus::AwaitingBatch)
                );
            })
            .await;
    }

    // The "PendingBatch when replica deactivates mid-flight" scenario is no
    // longer reachable: registered modules always have replicas_range.min >= 1
    // active replicas, so a controller cannot deactivate them via the
    // allocation alone. The DelayedBatchStub itself is retained for future use.

    #[tokio::test(flavor = "current_thread", start_paused = false)]
    async fn scheduler_records_activating_status() {
        let local = LocalSet::new();
        local
            .run_until(async {
                let module_id = ModuleId::new(BlockingActivateStub::id()).unwrap();
                let owner = nuillu_types::ModuleInstanceId::new(
                    module_id.clone(),
                    nuillu_types::ReplicaIndex::ZERO,
                );
                let mut alloc = ResourceAllocation::default();
                alloc.set(
                    module_id.clone(),
                    ModuleConfig {
                        tier: ModelTier::Default,
                        ..Default::default()
                    },
                );
                alloc.set_activation(module_id.clone(), ActivationRatio::ONE);

                let blackboard = Blackboard::with_allocation(alloc);
                let caps = test_caps(blackboard.clone());
                let (entered_tx, entered_rx) = oneshot::channel();
                let entered_tx = Rc::new(RefCell::new(Some(entered_tx)));
                let (_release_tx, release_rx) = oneshot::channel();
                let release_rx = Rc::new(RefCell::new(Some(release_rx)));
                let modules = ModuleRegistry::new()
                    .register(
                        0..=0,
                        Bpm::from_f64(60.0)..=Bpm::from_f64(60.0),
                        linear_ratio_fn,
                        {
                            let entered_tx = Rc::clone(&entered_tx);
                            let release_rx = Rc::clone(&release_rx);
                            move |_| BlockingActivateStub {
                                entered: entered_tx.borrow_mut().take(),
                                release: release_rx.borrow_mut().take(),
                                batch_sent: false,
                            }
                        },
                    )
                    .unwrap()
                    .build(&caps)
                    .await
                    .unwrap();

                super::run(modules, test_config(), async {
                    let _ = entered_rx.await;
                    wait_for_status(&blackboard, &owner, ModuleRunStatus::Activating).await;
                })
                .await
                .expect("scheduler returned err");

                assert_eq!(
                    blackboard
                        .read(|bb| bb.module_status_for_instance(&owner).cloned())
                        .await,
                    Some(ModuleRunStatus::Activating)
                );
            })
            .await;
    }

    #[tokio::test(flavor = "current_thread", start_paused = false)]
    async fn cognition_writer_capability_appends_to_log() {
        let local = LocalSet::new();
        local
            .run_until(async {
                let mut alloc = ResourceAllocation::default();
                alloc.set(
                    builtin::cognition_gate(),
                    ModuleConfig {
                        tier: ModelTier::Default,
                        ..Default::default()
                    },
                );
                alloc.set_activation(builtin::cognition_gate(), ActivationRatio::ONE);

                let blackboard = Blackboard::with_allocation(alloc);
                let caps = test_caps(blackboard.clone());
                let (done_tx, done_rx) = oneshot::channel();
                let done_tx = Rc::new(RefCell::new(Some(done_tx)));
                let modules = ModuleRegistry::new()
                    .register(
                        0..=0,
                        Bpm::from_f64(60.0)..=Bpm::from_f64(60.0),
                        linear_ratio_fn,
                        {
                            let done_tx = Rc::clone(&done_tx);
                            move |caps| CognitionGateStub {
                                writer: caps.cognition_writer(),
                                on_done: done_tx.borrow_mut().take(),
                            }
                        },
                    )
                    .unwrap()
                    .build(&caps)
                    .await
                    .unwrap();

                super::run(modules, test_config(), async move {
                    let _ = done_rx.await;
                })
                .await
                .expect("scheduler returned err");

                blackboard
                    .read(|bb| {
                        assert_eq!(bb.cognition_log().len(), 1);
                        assert_eq!(bb.cognition_log().entries()[0].text, "novel-event");
                    })
                    .await;
            })
            .await;
    }

    #[tokio::test(flavor = "current_thread", start_paused = false)]
    async fn activation_retries_reuse_the_same_batch() {
        let local = LocalSet::new();
        local
            .run_until(async {
                let retry_id = ModuleId::new(RetryStub::id()).unwrap();
                let mut alloc = ResourceAllocation::default();
                alloc.set(
                    retry_id.clone(),
                    ModuleConfig {
                        tier: ModelTier::Default,
                        ..Default::default()
                    },
                );
                alloc.set_activation(retry_id.clone(), ActivationRatio::ONE);

                let blackboard = Blackboard::with_allocation(alloc);
                let caps = test_caps(blackboard.clone());
                let attempts = Rc::new(Cell::new(0));
                let (done_tx, done_rx) = oneshot::channel();
                let done_tx = Rc::new(RefCell::new(Some(done_tx)));
                let modules = ModuleRegistry::new()
                    .register(
                        0..=0,
                        Bpm::from_f64(60.0)..=Bpm::from_f64(60.0),
                        linear_ratio_fn,
                        {
                            let attempts = Rc::clone(&attempts);
                            let done_tx = Rc::clone(&done_tx);
                            move |caps| RetryStub {
                                memo: caps.memo(),
                                attempts: Rc::clone(&attempts),
                                on_done: done_tx.borrow_mut().take(),
                                batch_sent: false,
                            }
                        },
                    )
                    .unwrap()
                    .build(&caps)
                    .await
                    .unwrap();

                super::run(
                    modules,
                    AgentEventLoopConfig {
                        idle_threshold: std::time::Duration::from_millis(50),
                        activate_retries: 2,
                    },
                    async move {
                        let _ = done_rx.await;
                    },
                )
                .await
                .expect("scheduler returned err");

                assert_eq!(attempts.get(), 3);
                let retry_logs = blackboard.read(|bb| bb.recent_memo_logs()).await;
                assert!(retry_logs.iter().any(|record| {
                    record.owner.module == retry_id
                        && record.content == "attempt 3 handled stable-batch"
                }));
            })
            .await;
    }

    #[tokio::test(flavor = "current_thread", start_paused = false)]
    async fn activation_retry_exhaustion_fails_runtime() {
        let local = LocalSet::new();
        local
            .run_until(async {
                let fail_id = ModuleId::new(AlwaysFailStub::id()).unwrap();
                let mut alloc = ResourceAllocation::default();
                alloc.set(
                    fail_id.clone(),
                    ModuleConfig {
                        tier: ModelTier::Default,
                        ..Default::default()
                    },
                );
                alloc.set_activation(fail_id.clone(), ActivationRatio::ONE);

                let blackboard = Blackboard::with_allocation(alloc);
                let caps = test_caps(blackboard.clone());
                let modules = ModuleRegistry::new()
                    .register(
                        0..=0,
                        Bpm::from_f64(60.0)..=Bpm::from_f64(60.0),
                        linear_ratio_fn,
                        |_| AlwaysFailStub { batch_sent: false },
                    )
                    .unwrap()
                    .build(&caps)
                    .await
                    .unwrap();

                let err = super::run(
                    modules,
                    AgentEventLoopConfig {
                        idle_threshold: std::time::Duration::from_millis(50),
                        activate_retries: 1,
                    },
                    std::future::pending::<()>(),
                )
                .await
                .unwrap_err();

                assert!(matches!(
                    err,
                    SchedulerError::ModuleTaskFailed { owner, phase: "activate", message }
                        if owner.module == fail_id
                            && message.contains("permanent activation failure")
                ));
                let owner =
                    nuillu_types::ModuleInstanceId::new(fail_id, nuillu_types::ReplicaIndex::ZERO);
                assert!(matches!(
                    blackboard
                        .read(|bb| bb.module_status_for_instance(&owner).cloned())
                        .await,
                    Some(ModuleRunStatus::Failed { phase, message })
                        if phase == "activate" && message.contains("permanent activation failure")
                ));
            })
            .await;
    }

    // The "inactive replica is not started" scenario is no longer reachable:
    // registered modules always have replicas_range.min >= 1 active replicas
    // by construction.

    #[tokio::test(flavor = "current_thread", start_paused = false)]
    async fn idle_deadlock_records_marker_and_publishes_cognition_log_update() {
        let local = LocalSet::new();
        local
            .run_until(async {
                let observer_id = ModuleId::new(DeadlockObserver::id()).unwrap();
                let mut alloc = ResourceAllocation::default();
                alloc.set(
                    observer_id.clone(),
                    ModuleConfig {
                        tier: ModelTier::Default,
                        ..Default::default()
                    },
                );
                alloc.set_activation(observer_id.clone(), ActivationRatio::ONE);

                let blackboard = Blackboard::with_allocation(alloc);
                let caps = test_caps(blackboard.clone());
                let (done_tx, done_rx) = oneshot::channel();
                let done_tx = Rc::new(RefCell::new(Some(done_tx)));
                let modules = ModuleRegistry::new()
                    .register(
                        0..=0,
                        Bpm::from_f64(60.0)..=Bpm::from_f64(60.0),
                        linear_ratio_fn,
                        {
                            let done_tx = Rc::clone(&done_tx);
                            move |caps| DeadlockObserver {
                                updates: caps.cognition_log_updated_inbox(),
                                memo: caps.memo(),
                                on_done: done_tx.borrow_mut().take(),
                            }
                        },
                    )
                    .unwrap()
                    .build(&caps)
                    .await
                    .unwrap();

                super::run(
                    modules,
                    AgentEventLoopConfig {
                        idle_threshold: std::time::Duration::from_millis(10),
                        activate_retries: 0,
                    },
                    async move {
                        let _ = done_rx.await;
                    },
                )
                .await
                .expect("scheduler returned err");

                blackboard
                    .read(|bb| {
                        assert!(bb.agentic_deadlock_marker().is_some());
                        assert_eq!(bb.cognition_log().len(), 0);
                    })
                    .await;
                let observer_logs = blackboard.read(|bb| bb.recent_memo_logs()).await;
                assert!(observer_logs.iter().any(|record| {
                    record.owner.module == observer_id
                        && record.content == "observed deadlock marker"
                }));
            })
            .await;
    }
}
