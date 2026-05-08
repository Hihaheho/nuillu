use std::future::Future;
use std::pin::pin;
use std::time::Duration;

use futures::stream::{FuturesUnordered, StreamExt};
use nuillu_module::{AgentRuntimeControl, AllocatedModule, AllocatedModules, ModuleBatch};
use nuillu_types::ModuleInstanceId;
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
        .map(ModuleState::Stored)
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
                            &owners,
                            &active,
                            &mut states,
                            &mut tasks,
                            config,
                            &parent,
                            &subscriber,
                        )?;
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
    Stored(AllocatedModule),
    Awaiting,
    PendingBatch {
        module: AllocatedModule,
        batch: ModuleBatch,
    },
    Activating,
}

enum TaskMessage {
    NextBatch {
        index: usize,
        module: AllocatedModule,
        result: Result<ModuleBatch, String>,
    },
    Activate {
        index: usize,
        module: AllocatedModule,
        result: Result<(), String>,
    },
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
            ModuleState::Stored(_) if active[index] => {
                let ModuleState::Stored(module) =
                    std::mem::replace(&mut states[index], ModuleState::Awaiting)
                else {
                    unreachable!("module state changed while scheduling next batch");
                };
                spawn_next_batch(tasks, index, module, parent, subscriber);
            }
            ModuleState::PendingBatch { .. } if active[index] => {
                let ModuleState::PendingBatch { module, batch } =
                    std::mem::replace(&mut states[index], ModuleState::Activating)
                else {
                    unreachable!("module state changed while scheduling activation");
                };
                spawn_activate(tasks, index, module, batch, config, parent, subscriber);
            }
            _ => {}
        }
    }
}

fn handle_task_message(
    message: TaskMessage,
    owners: &[ModuleInstanceId],
    active: &[bool],
    states: &mut [ModuleState],
    tasks: &mut FuturesUnordered<JoinHandle<TaskMessage>>,
    config: AgentEventLoopConfig,
    parent: &tracing::Span,
    subscriber: &tracing::Dispatch,
) -> Result<(), SchedulerError> {
    match message {
        TaskMessage::NextBatch {
            index,
            module,
            result,
        } => match result {
            Ok(batch) => {
                if active[index] {
                    states[index] = ModuleState::Activating;
                    spawn_activate(tasks, index, module, batch, config, parent, subscriber);
                } else {
                    states[index] = ModuleState::PendingBatch { module, batch };
                }
                Ok(())
            }
            Err(message) => Err(SchedulerError::ModuleTaskFailed {
                owner: owners[index].clone(),
                phase: "next_batch",
                message,
            }),
        },
        TaskMessage::Activate {
            index,
            module,
            result,
        } => match result {
            Ok(()) => {
                states[index] = ModuleState::Stored(module);
                Ok(())
            }
            Err(message) => Err(SchedulerError::ModuleTaskFailed {
                owner: owners[index].clone(),
                phase: "activate",
                message,
            }),
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

fn spawn_activate(
    tasks: &mut FuturesUnordered<JoinHandle<TaskMessage>>,
    index: usize,
    module: AllocatedModule,
    batch: ModuleBatch,
    config: AgentEventLoopConfig,
    parent: &tracing::Span,
    subscriber: &tracing::Dispatch,
) {
    tasks.push(spawn_local(
        async move {
            let (module, result) =
                activate_with_retries(module, &batch, config.activate_retries).await;
            TaskMessage::Activate {
                index,
                module,
                result,
            }
        }
        .instrument(parent.clone())
        .with_subscriber(subscriber.clone()),
    ));
}

async fn activate_with_retries(
    mut module: AllocatedModule,
    batch: &ModuleBatch,
    activate_retries: u8,
) -> (AllocatedModule, Result<(), String>) {
    let mut retries = 0_u8;
    loop {
        match module.activate(batch).await {
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
    use std::rc::Rc;

    use async_trait::async_trait;
    use nuillu_blackboard::{ActivationRatio, Blackboard, ModuleConfig, ResourceAllocation};
    use nuillu_module::{
        AttentionStreamUpdated, AttentionStreamUpdatedInbox, AttentionWriter, Memo, Module,
        ModuleRegistry, QueryInbox, QueryRequest,
    };
    use nuillu_types::{ModelTier, ModuleId, builtin};
    use tokio::sync::oneshot;
    use tokio::task::LocalSet;

    use crate::testing::test_caps;

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

        async fn next_batch(&mut self) -> anyhow::Result<Self::Batch> {
            Ok(self.query_inbox.next_item().await?.body)
        }

        async fn activate(&mut self, batch: &Self::Batch) -> anyhow::Result<()> {
            self.memo.write(format!("echoed {}", batch.question)).await;
            if let Some(tx) = self.on_done.take() {
                let _ = tx.send(());
            }
            Ok(())
        }
    }

    struct AttentionGateStub {
        writer: AttentionWriter,
        on_done: Option<oneshot::Sender<()>>,
    }

    #[async_trait(?Send)]
    impl Module for AttentionGateStub {
        type Batch = ();

        async fn next_batch(&mut self) -> anyhow::Result<Self::Batch> {
            if self.on_done.is_some() {
                Ok(())
            } else {
                std::future::pending().await
            }
        }

        async fn activate(&mut self, _batch: &Self::Batch) -> anyhow::Result<()> {
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

        async fn next_batch(&mut self) -> anyhow::Result<Self::Batch> {
            if self.batch_sent {
                std::future::pending().await
            } else {
                self.batch_sent = true;
                Ok("stable-batch".into())
            }
        }

        async fn activate(&mut self, batch: &Self::Batch) -> anyhow::Result<()> {
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

        async fn next_batch(&mut self) -> anyhow::Result<Self::Batch> {
            if self.batch_sent {
                std::future::pending().await
            } else {
                self.batch_sent = true;
                Ok(())
            }
        }

        async fn activate(&mut self, _batch: &Self::Batch) -> anyhow::Result<()> {
            anyhow::bail!("permanent activation failure")
        }
    }

    struct DeadlockObserver {
        updates: AttentionStreamUpdatedInbox,
        memo: Memo,
        on_done: Option<oneshot::Sender<()>>,
    }

    #[async_trait(?Send)]
    impl Module for DeadlockObserver {
        type Batch = AttentionStreamUpdated;

        async fn next_batch(&mut self) -> anyhow::Result<Self::Batch> {
            Ok(self.updates.next_item().await?.body)
        }

        async fn activate(&mut self, batch: &Self::Batch) -> anyhow::Result<()> {
            assert_eq!(batch, &AttentionStreamUpdated::AgenticDeadlockMarker);
            self.memo.write("observed deadlock marker").await;
            if let Some(tx) = self.on_done.take() {
                let _ = tx.send(());
            }
            Ok(())
        }
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
                        activation_ratio: nuillu_blackboard::ActivationRatio::ONE,
                        tier: ModelTier::Default,
                        ..Default::default()
                    },
                );

                let blackboard = Blackboard::with_allocation(alloc);
                let caps = test_caps(blackboard.clone());
                let (done_tx, done_rx) = oneshot::channel();
                let done_tx = Rc::new(RefCell::new(Some(done_tx)));
                let modules = ModuleRegistry::new()
                    .register(echo_id(), 0..=1, {
                        let done_tx = Rc::clone(&done_tx);
                        move |caps| EchoModule {
                            query_inbox: caps.query_inbox(),
                            memo: caps.memo(),
                            on_done: done_tx.borrow_mut().take(),
                        }
                    })
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
                        let echo_memo = bb.memo(&echo_id()).expect("echo memo missing");
                        assert_eq!(echo_memo, "echoed ping");
                        assert_eq!(
                            bb.attention_stream().len(),
                            0,
                            "echo does not hold AttentionWriter",
                        );
                    })
                    .await;
            })
            .await;
    }

    #[tokio::test(flavor = "current_thread", start_paused = false)]
    async fn attention_writer_capability_appends_to_stream() {
        let local = LocalSet::new();
        local
            .run_until(async {
                let mut alloc = ResourceAllocation::default();
                alloc.set(
                    builtin::attention_gate(),
                    ModuleConfig {
                        activation_ratio: nuillu_blackboard::ActivationRatio::ONE,
                        tier: ModelTier::Default,
                        ..Default::default()
                    },
                );

                let blackboard = Blackboard::with_allocation(alloc);
                let caps = test_caps(blackboard.clone());
                let (done_tx, done_rx) = oneshot::channel();
                let done_tx = Rc::new(RefCell::new(Some(done_tx)));
                let modules = ModuleRegistry::new()
                    .register(builtin::attention_gate(), 0..=1, {
                        let done_tx = Rc::clone(&done_tx);
                        move |caps| AttentionGateStub {
                            writer: caps.attention_writer(),
                            on_done: done_tx.borrow_mut().take(),
                        }
                    })
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
                        assert_eq!(bb.attention_stream().len(), 1);
                        assert_eq!(bb.attention_stream().entries()[0].text, "novel-event");
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
                let retry_id = ModuleId::new("retry").unwrap();
                let mut alloc = ResourceAllocation::default();
                alloc.set(
                    retry_id.clone(),
                    ModuleConfig {
                        activation_ratio: ActivationRatio::ONE,
                        tier: ModelTier::Default,
                        ..Default::default()
                    },
                );

                let blackboard = Blackboard::with_allocation(alloc);
                let caps = test_caps(blackboard.clone());
                let attempts = Rc::new(Cell::new(0));
                let (done_tx, done_rx) = oneshot::channel();
                let done_tx = Rc::new(RefCell::new(Some(done_tx)));
                let modules = ModuleRegistry::new()
                    .register(retry_id.clone(), 0..=1, {
                        let attempts = Rc::clone(&attempts);
                        let done_tx = Rc::clone(&done_tx);
                        move |caps| RetryStub {
                            memo: caps.memo(),
                            attempts: Rc::clone(&attempts),
                            on_done: done_tx.borrow_mut().take(),
                            batch_sent: false,
                        }
                    })
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
                assert_eq!(
                    blackboard.memo(&retry_id).await.as_deref(),
                    Some("attempt 3 handled stable-batch")
                );
            })
            .await;
    }

    #[tokio::test(flavor = "current_thread", start_paused = false)]
    async fn activation_retry_exhaustion_fails_runtime() {
        let local = LocalSet::new();
        local
            .run_until(async {
                let fail_id = ModuleId::new("always-fail").unwrap();
                let mut alloc = ResourceAllocation::default();
                alloc.set(
                    fail_id.clone(),
                    ModuleConfig {
                        activation_ratio: ActivationRatio::ONE,
                        tier: ModelTier::Default,
                        ..Default::default()
                    },
                );

                let caps = test_caps(Blackboard::with_allocation(alloc));
                let modules = ModuleRegistry::new()
                    .register(fail_id.clone(), 0..=1, |_| AlwaysFailStub {
                        batch_sent: false,
                    })
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
            })
            .await;
    }

    #[tokio::test(flavor = "current_thread", start_paused = false)]
    async fn inactive_replica_is_not_started() {
        let local = LocalSet::new();
        local
            .run_until(async {
                let inactive_id = ModuleId::new("inactive").unwrap();
                let mut alloc = ResourceAllocation::default();
                alloc.set(
                    inactive_id.clone(),
                    ModuleConfig {
                        activation_ratio: ActivationRatio::ZERO,
                        tier: ModelTier::Default,
                        ..Default::default()
                    },
                );

                let blackboard = Blackboard::with_allocation(alloc);
                let caps = test_caps(blackboard.clone());
                let (done_tx, mut done_rx) = oneshot::channel::<()>();
                let done_tx = Rc::new(RefCell::new(Some(done_tx)));
                let modules = ModuleRegistry::new()
                    .register(inactive_id.clone(), 0..=1, {
                        let done_tx = Rc::clone(&done_tx);
                        move |caps| RetryStub {
                            memo: caps.memo(),
                            attempts: Rc::new(Cell::new(0)),
                            on_done: done_tx.borrow_mut().take(),
                            batch_sent: false,
                        }
                    })
                    .unwrap()
                    .build(&caps)
                    .await
                    .unwrap();

                super::run(modules, test_config(), async {
                    for _ in 0..4 {
                        tokio::task::yield_now().await;
                    }
                })
                .await
                .expect("scheduler returned err");

                assert!(done_rx.try_recv().is_err());
                assert!(blackboard.memo(&inactive_id).await.is_none());
            })
            .await;
    }

    #[tokio::test(flavor = "current_thread", start_paused = false)]
    async fn idle_deadlock_records_marker_and_publishes_attention_update() {
        let local = LocalSet::new();
        local
            .run_until(async {
                let observer_id = ModuleId::new("deadlock-observer").unwrap();
                let mut alloc = ResourceAllocation::default();
                alloc.set(
                    observer_id.clone(),
                    ModuleConfig {
                        activation_ratio: ActivationRatio::ONE,
                        tier: ModelTier::Default,
                        ..Default::default()
                    },
                );

                let blackboard = Blackboard::with_allocation(alloc);
                let caps = test_caps(blackboard.clone());
                let (done_tx, done_rx) = oneshot::channel();
                let done_tx = Rc::new(RefCell::new(Some(done_tx)));
                let modules = ModuleRegistry::new()
                    .register(observer_id.clone(), 0..=1, {
                        let done_tx = Rc::clone(&done_tx);
                        move |caps| DeadlockObserver {
                            updates: caps.attention_stream_updated_inbox(),
                            memo: caps.memo(),
                            on_done: done_tx.borrow_mut().take(),
                        }
                    })
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
                        assert_eq!(bb.attention_stream().len(), 0);
                    })
                    .await;
                assert_eq!(
                    blackboard.memo(&observer_id).await.as_deref(),
                    Some("observed deadlock marker")
                );
            })
            .await;
    }
}
