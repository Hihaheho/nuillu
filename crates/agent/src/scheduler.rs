use std::collections::HashMap;
use std::future::Future;
use std::pin::pin;
use std::sync::Arc;
use std::time::Duration;

use chrono::{DateTime, Utc};
use futures::stream::{FuturesUnordered, StreamExt};
use nuillu_blackboard::{ActivationRatio, IdentityMemoryRecord};
use nuillu_module::{
    ActivateCx, ActivationGateVote, AgentRuntimeControl, AllocatedModule, AllocatedModules,
    ModuleBatch, ModuleDependencies, ModuleRunStatus, ports::Clock,
};
use nuillu_types::{ModuleId, ModuleInstanceId};
use thiserror::Error;
use tokio::task::{JoinHandle, spawn_local};
use tokio::time::{Instant, sleep_until};
use tracing::{Instrument as _, instrument::WithSubscriber as _};

use crate::kicks::{Kick, KickHandle, KickInbox};

/// Silent window before nooping pending dependency kicks when no natural wake arrives.
const KICK_SILENT_TIMEOUT: Duration = Duration::from_secs(1);

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
    let (runtime, modules, dependencies) = modules.into_parts_with_dependencies();
    let owners = modules
        .iter()
        .map(|module| module.owner().clone())
        .collect::<Vec<_>>();
    let mut kick_inboxes: Vec<Option<KickInbox>> = Vec::with_capacity(owners.len());
    let mut kick_targets_by_role = HashMap::<ModuleId, Vec<KickTarget>>::new();
    for owner in &owners {
        let (inbox, handle) = KickInbox::new();
        kick_inboxes.push(Some(inbox));
        kick_targets_by_role
            .entry(owner.module.clone())
            .or_default()
            .push(KickTarget {
                owner: owner.clone(),
                handle,
            });
    }
    let deps_resolver = DepsResolver {
        runtime: runtime.clone(),
        dependencies: Arc::new(dependencies),
        kick_targets_by_role: Arc::new(kick_targets_by_role),
    };
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
        &mut kick_inboxes,
        &deps_resolver,
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
                            &mut kick_inboxes,
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
                            &mut kick_inboxes,
                            &deps_resolver,
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
        gate_approved: bool,
        pending_kicks: Vec<Kick>,
    },
    PendingActivationGate,
    Activating,
}

enum TaskMessage {
    BatchCooldownExpired {
        index: usize,
        module: AllocatedModule,
        kick_inbox: KickInbox,
        delayed_for: Duration,
    },
    NextBatch {
        index: usize,
        module: AllocatedModule,
        kick_inbox: KickInbox,
        result: Result<(ModuleBatch, Vec<Kick>), String>,
    },
    Activate {
        index: usize,
        module: AllocatedModule,
        kick_inbox: KickInbox,
        pending_kicks: Vec<Kick>,
        activation_elapsed: Duration,
        result: Result<(), String>,
    },
    ActivationGate {
        index: usize,
        module: AllocatedModule,
        kick_inbox: KickInbox,
        batch: ModuleBatch,
        pending_kicks: Vec<Kick>,
        outcome: ActivationGateOutcome,
    },
}

struct NextBatchThrottle {
    not_before: DateTime<Utc>,
    activation_threshold: ActivationRatio,
}

#[derive(Clone)]
struct KickTarget {
    owner: ModuleInstanceId,
    handle: KickHandle,
}

#[derive(Clone)]
struct DepsResolver {
    runtime: AgentRuntimeControl,
    dependencies: Arc<ModuleDependencies>,
    kick_targets_by_role: Arc<HashMap<ModuleId, Vec<KickTarget>>>,
}

impl DepsResolver {
    async fn resolve_active_now(&self, dependent: &ModuleId) -> Vec<KickHandle> {
        let mut handles = Vec::new();
        for dep_id in self.dependencies.deps_of(dependent) {
            let Some(targets) = self.kick_targets_by_role.get(dep_id) else {
                continue;
            };
            for target in targets {
                if self.runtime.is_active(&target.owner).await {
                    handles.push(target.handle.clone());
                }
            }
        }
        handles
    }
}

#[allow(clippy::too_many_arguments)]
async fn refresh_active_and_schedule(
    runtime: &AgentRuntimeControl,
    owners: &[ModuleInstanceId],
    active: &mut [bool],
    states: &mut [ModuleState],
    tasks: &mut FuturesUnordered<JoinHandle<TaskMessage>>,
    kick_inboxes: &mut [Option<KickInbox>],
    deps_resolver: &DepsResolver,
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
                let kick_inbox = kick_inboxes[index]
                    .take()
                    .expect("kick inbox available for stored module");
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
                            kick_inbox,
                            throttle.not_before,
                            runtime.clock(),
                            activation_increase,
                            parent,
                            subscriber,
                        );
                    } else {
                        spawn_next_batch(
                            tasks,
                            index,
                            module,
                            kick_inbox,
                            deps_resolver.clone(),
                            parent,
                            subscriber,
                        );
                    }
                } else {
                    spawn_next_batch(
                        tasks,
                        index,
                        module,
                        kick_inbox,
                        deps_resolver.clone(),
                        parent,
                        subscriber,
                    );
                }
            }
            ModuleState::Stored { .. } => {
                if let Some(kick_inbox) = &mut kick_inboxes[index] {
                    notify_pending_and_ready(Vec::new(), kick_inbox);
                }
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
                let ModuleState::PendingBatch {
                    module,
                    batch,
                    gate_approved,
                    pending_kicks,
                } = std::mem::replace(&mut states[index], ModuleState::Activating)
                else {
                    unreachable!("module state changed while scheduling activation");
                };
                let kick_inbox = kick_inboxes[index]
                    .take()
                    .expect("kick inbox available for pending-batch module");
                if gate_approved {
                    runtime
                        .record_module_status(owners[index].clone(), ModuleRunStatus::Activating)
                        .await;
                    let catalog = runtime.module_catalog();
                    let identity_memories = runtime.identity_memories().await;
                    spawn_activate(
                        tasks,
                        index,
                        module,
                        kick_inbox,
                        pending_kicks,
                        batch,
                        config,
                        runtime.clone(),
                        catalog,
                        identity_memories,
                        parent,
                        subscriber,
                    );
                } else {
                    let scheduled = spawn_activation_gate_or_activate(
                        runtime,
                        tasks,
                        index,
                        owners[index].clone(),
                        module,
                        kick_inbox,
                        pending_kicks,
                        batch,
                        config,
                        parent,
                        subscriber,
                    )
                    .await;
                    states[index] = scheduled.module_state();
                }
            }
            ModuleState::PendingBatch { .. } => {
                if let Some(kick_inbox) = &mut kick_inboxes[index] {
                    notify_pending_and_ready(Vec::new(), kick_inbox);
                }
                runtime
                    .record_module_status(owners[index].clone(), ModuleRunStatus::PendingBatch)
                    .await;
            }
            ModuleState::PendingActivationGate => {
                let status = if active[index] {
                    ModuleRunStatus::PendingActivationGate
                } else {
                    ModuleRunStatus::Inactive
                };
                runtime
                    .record_module_status(owners[index].clone(), status)
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

#[allow(clippy::too_many_arguments)]
async fn handle_task_message(
    message: TaskMessage,
    runtime: &AgentRuntimeControl,
    owners: &[ModuleInstanceId],
    states: &mut [ModuleState],
    tasks: &mut FuturesUnordered<JoinHandle<TaskMessage>>,
    kick_inboxes: &mut [Option<KickInbox>],
    config: AgentEventLoopConfig,
    parent: &tracing::Span,
    subscriber: &tracing::Dispatch,
) -> Result<(), SchedulerError> {
    match message {
        TaskMessage::BatchCooldownExpired {
            index,
            module,
            kick_inbox,
            delayed_for,
        } => {
            runtime
                .record_module_batch_throttled(owners[index].clone(), delayed_for)
                .await;
            kick_inboxes[index] = Some(kick_inbox);
            states[index] = ModuleState::Stored {
                module,
                next_batch_throttle: None,
            };
            Ok(())
        }
        TaskMessage::NextBatch {
            index,
            module,
            mut kick_inbox,
            result,
        } => match result {
            Ok((batch, pending_kicks)) => {
                if runtime.is_active(&owners[index]).await {
                    let scheduled = spawn_activation_gate_or_activate(
                        runtime,
                        tasks,
                        index,
                        owners[index].clone(),
                        module,
                        kick_inbox,
                        pending_kicks,
                        batch,
                        config,
                        parent,
                        subscriber,
                    )
                    .await;
                    states[index] = scheduled.module_state();
                } else {
                    runtime
                        .record_module_status(owners[index].clone(), ModuleRunStatus::PendingBatch)
                        .await;
                    notify_pending_and_ready(pending_kicks, &mut kick_inbox);
                    kick_inboxes[index] = Some(kick_inbox);
                    states[index] = ModuleState::PendingBatch {
                        module,
                        batch,
                        gate_approved: false,
                        pending_kicks: Vec::new(),
                    };
                }
                Ok(())
            }
            Err(message) => {
                notify_pending_and_ready(Vec::new(), &mut kick_inbox);
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
            mut kick_inbox,
            pending_kicks,
            activation_elapsed,
            result,
        } => match result {
            Ok(()) => {
                notify_pending_and_ready(pending_kicks, &mut kick_inbox);
                kick_inboxes[index] = Some(kick_inbox);
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
                notify_pending_and_ready(pending_kicks, &mut kick_inbox);
                kick_inboxes[index] = Some(kick_inbox);
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
        TaskMessage::ActivationGate {
            index,
            module,
            mut kick_inbox,
            batch,
            pending_kicks,
            outcome,
        } => {
            if outcome.allowed() {
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
                        kick_inbox,
                        pending_kicks,
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
                    notify_pending_and_ready(pending_kicks, &mut kick_inbox);
                    kick_inboxes[index] = Some(kick_inbox);
                    states[index] = ModuleState::PendingBatch {
                        module,
                        batch,
                        gate_approved: true,
                        pending_kicks: Vec::new(),
                    };
                }
            } else {
                notify_pending_and_ready(pending_kicks, &mut kick_inbox);
                kick_inboxes[index] = Some(kick_inbox);
                states[index] = ModuleState::Stored {
                    module,
                    next_batch_throttle: None,
                };
            }
            Ok(())
        }
    }
}

fn notify_pending_and_ready(mut pending_kicks: Vec<Kick>, kick_inbox: &mut KickInbox) {
    pending_kicks.extend(kick_inbox.drain_ready());
    for kick in pending_kicks {
        kick.notify_finish();
    }
}

#[allow(clippy::too_many_arguments)]
async fn spawn_activation_gate_or_activate(
    runtime: &AgentRuntimeControl,
    tasks: &mut FuturesUnordered<JoinHandle<TaskMessage>>,
    index: usize,
    owner: ModuleInstanceId,
    module: AllocatedModule,
    kick_inbox: KickInbox,
    pending_kicks: Vec<Kick>,
    batch: ModuleBatch,
    config: AgentEventLoopConfig,
    parent: &tracing::Span,
    subscriber: &tracing::Dispatch,
) -> ActivationScheduling {
    let gate_requests = runtime
        .activation_gate_requests(&owner, batch.clone())
        .await;
    if gate_requests.is_empty() {
        runtime
            .record_module_status(owner, ModuleRunStatus::Activating)
            .await;
        let catalog = runtime.module_catalog();
        let identity_memories = runtime.identity_memories().await;
        spawn_activate(
            tasks,
            index,
            module,
            kick_inbox,
            pending_kicks,
            batch,
            config,
            runtime.clone(),
            catalog,
            identity_memories,
            parent,
            subscriber,
        );
        ActivationScheduling::Activating
    } else {
        runtime
            .record_module_status(owner, ModuleRunStatus::PendingActivationGate)
            .await;
        spawn_activation_gate_wait(
            tasks,
            index,
            module,
            kick_inbox,
            pending_kicks,
            batch,
            gate_requests,
            parent,
            subscriber,
        );
        ActivationScheduling::PendingActivationGate
    }
}

enum ActivationScheduling {
    Activating,
    PendingActivationGate,
}

impl ActivationScheduling {
    fn module_state(self) -> ModuleState {
        match self {
            Self::Activating => ModuleState::Activating,
            Self::PendingActivationGate => ModuleState::PendingActivationGate,
        }
    }
}

fn spawn_next_batch(
    tasks: &mut FuturesUnordered<JoinHandle<TaskMessage>>,
    index: usize,
    mut module: AllocatedModule,
    mut kick_inbox: KickInbox,
    deps_resolver: DepsResolver,
    parent: &tracing::Span,
    subscriber: &tracing::Dispatch,
) {
    tasks.push(spawn_local(
        async move {
            let (batch_result, mut pending_kicks) = {
                let mut pending_kicks = Vec::new();
                let mut kick_inbox_closed = false;
                let mut next_batch = pin!(module.next_batch());

                loop {
                    let timeout_enabled = !pending_kicks.is_empty();
                    tokio::select! {
                        biased;
                        result = &mut next_batch => break (result, pending_kicks),
                        maybe_kick = kick_inbox.next(), if !kick_inbox_closed => {
                            if let Some(kick) = maybe_kick {
                                pending_kicks.push(kick);
                            } else {
                                kick_inbox_closed = true;
                            }
                        }
                        _ = tokio::time::sleep(KICK_SILENT_TIMEOUT), if timeout_enabled => {
                            for kick in pending_kicks.drain(..) {
                                kick.notify_finish();
                            }
                        }
                    }
                }
            };

            let result = match batch_result {
                Ok(batch) => {
                    let dep_handles = deps_resolver
                        .resolve_active_now(&module.owner().module)
                        .await;
                    let completions = dep_handles
                        .into_iter()
                        .map(|handle| handle.send(module.owner().clone()))
                        .collect::<Vec<_>>();
                    for completion in completions {
                        let _ = completion.await;
                    }
                    pending_kicks.extend(kick_inbox.drain_ready());
                    Ok((batch, pending_kicks))
                }
                Err(error) => {
                    pending_kicks.extend(kick_inbox.drain_ready());
                    for kick in pending_kicks {
                        kick.notify_finish();
                    }
                    Err(format!("{error:#}"))
                }
            };

            TaskMessage::NextBatch {
                index,
                module,
                kick_inbox,
                result,
            }
        }
        .instrument(parent.clone())
        .with_subscriber(subscriber.clone()),
    ));
}

#[allow(clippy::too_many_arguments)]
fn spawn_batch_cooldown(
    tasks: &mut FuturesUnordered<JoinHandle<TaskMessage>>,
    index: usize,
    module: AllocatedModule,
    kick_inbox: KickInbox,
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
                kick_inbox,
                delayed_for,
            }
        }
        .instrument(parent.clone())
        .with_subscriber(subscriber.clone()),
    ));
}

#[allow(clippy::too_many_arguments)]
fn spawn_activation_gate_wait(
    tasks: &mut FuturesUnordered<JoinHandle<TaskMessage>>,
    index: usize,
    module: AllocatedModule,
    kick_inbox: KickInbox,
    pending_kicks: Vec<Kick>,
    batch: ModuleBatch,
    requests: Vec<tokio::sync::oneshot::Receiver<ActivationGateVote>>,
    parent: &tracing::Span,
    subscriber: &tracing::Dispatch,
) {
    tasks.push(spawn_local(
        async move {
            let outcome = collect_activation_gate_votes(requests).await;
            TaskMessage::ActivationGate {
                index,
                module,
                kick_inbox,
                batch,
                pending_kicks,
                outcome,
            }
        }
        .instrument(parent.clone())
        .with_subscriber(subscriber.clone()),
    ));
}

#[allow(clippy::too_many_arguments)]
fn spawn_activate(
    tasks: &mut FuturesUnordered<JoinHandle<TaskMessage>>,
    index: usize,
    module: AllocatedModule,
    kick_inbox: KickInbox,
    pending_kicks: Vec<Kick>,
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
                kick_inbox,
                pending_kicks,
                activation_elapsed,
                result,
            }
        }
        .instrument(parent.clone())
        .with_subscriber(subscriber.clone()),
    ));
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ActivationGateOutcome {
    allow: usize,
    suppress: usize,
}

impl ActivationGateOutcome {
    fn allowed(self) -> bool {
        if self.allow == 0 && self.suppress == 0 {
            return true;
        }
        self.allow > self.suppress
    }
}

async fn collect_activation_gate_votes(
    requests: Vec<tokio::sync::oneshot::Receiver<ActivationGateVote>>,
) -> ActivationGateOutcome {
    let mut allow = 0;
    let mut suppress = 0;
    for request in requests {
        match request.await {
            Ok(ActivationGateVote::Allow) => allow += 1,
            Ok(ActivationGateVote::Suppress) => suppress += 1,
            Err(_) => {}
        }
    }
    ActivationGateOutcome { allow, suppress }
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
        runtime.clock().now(),
    );
    let mut retries = 0_u8;
    loop {
        let owner = module.owner().clone();
        let activation_span = tracing::info_span!(
            target: "lutum",
            "module_activate",
            lutum.capture = true,
            owner = %owner,
            module = %owner.module,
            replica = owner.replica.get(),
            retry = retries,
        );
        match module
            .activate(&cx, batch)
            .instrument(activation_span)
            .await
        {
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
    use std::sync::Arc;
    use std::time::Duration;

    use async_trait::async_trait;
    use nuillu_blackboard::{
        ActivationRatio, Blackboard, BlackboardCommand, Bpm, IdentityMemoryRecord, ModuleConfig,
        ModuleRunStatus, ResourceAllocation, linear_ratio_fn,
    };
    use nuillu_module::{
        ActivationGate, ActivationGateEvent, ActivationGateVote, AttentionControlRequest,
        AttentionControlRequestInbox, CognitionLogUpdated, CognitionLogUpdatedInbox,
        CognitionWriter, Memo, Module, ModuleRegistry,
    };
    use nuillu_types::{MemoryContent, MemoryIndex, ModuleId, builtin};
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

    fn request_question(request: &AttentionControlRequest) -> &str {
        match request {
            AttentionControlRequest::Query { question, .. } => question,
            _ => "",
        }
    }

    fn request_question_owned(request: AttentionControlRequest) -> String {
        match request {
            AttentionControlRequest::Query { question, .. } => question,
            _ => String::new(),
        }
    }

    struct EchoModule {
        attention_control_inbox: AttentionControlRequestInbox,
        memo: Memo,
        on_done: Option<oneshot::Sender<()>>,
    }

    #[async_trait(?Send)]
    impl Module for EchoModule {
        type Batch = AttentionControlRequest;

        fn id() -> &'static str {
            "echo"
        }

        fn role_description() -> &'static str {
            "test stub"
        }

        async fn next_batch(&mut self) -> anyhow::Result<Self::Batch> {
            Ok(self.attention_control_inbox.next_item().await?.body)
        }

        async fn activate(
            &mut self,
            _cx: &nuillu_module::ActivateCx<'_>,
            batch: &Self::Batch,
        ) -> anyhow::Result<()> {
            self.memo
                .write(format!("echoed {}", request_question(batch)))
                .await;
            if let Some(tx) = self.on_done.take() {
                let _ = tx.send(());
            }
            Ok(())
        }
    }

    struct QueryBatchRecorder {
        attention_control_inbox: AttentionControlRequestInbox,
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
            let first =
                request_question_owned(self.attention_control_inbox.next_item().await?.body);
            let mut batch = vec![first];
            batch.extend(
                self.attention_control_inbox
                    .take_ready_items()?
                    .items
                    .into_iter()
                    .map(|envelope| request_question_owned(envelope.body)),
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

    struct GatedEchoModule {
        attention_control_inbox: AttentionControlRequestInbox,
        memo: Memo,
        activations: Rc<Cell<u8>>,
        on_done: Option<oneshot::Sender<()>>,
    }

    #[async_trait(?Send)]
    impl Module for GatedEchoModule {
        type Batch = AttentionControlRequest;

        fn id() -> &'static str {
            "gated-echo"
        }

        fn role_description() -> &'static str {
            "test gated target"
        }

        async fn next_batch(&mut self) -> anyhow::Result<Self::Batch> {
            Ok(self.attention_control_inbox.next_item().await?.body)
        }

        async fn activate(
            &mut self,
            _cx: &nuillu_module::ActivateCx<'_>,
            batch: &Self::Batch,
        ) -> anyhow::Result<()> {
            self.activations
                .set(self.activations.get().saturating_add(1));
            self.memo
                .write(format!("gated echo handled {}", request_question(batch)))
                .await;
            if let Some(done) = self.on_done.take() {
                let _ = done.send(());
            }
            Ok(())
        }
    }

    macro_rules! activation_gate_stub {
        ($name:ident, $id:literal) => {
            struct $name {
                gate: ActivationGate<GatedEchoModule>,
                votes: Rc<RefCell<VecDeque<Option<ActivationGateVote>>>>,
                on_seen: Option<oneshot::Sender<()>>,
            }

            #[async_trait(?Send)]
            impl Module for $name {
                type Batch = ActivationGateEvent<GatedEchoModule>;

                fn id() -> &'static str {
                    $id
                }

                fn role_description() -> &'static str {
                    "test activation gate"
                }

                async fn next_batch(&mut self) -> anyhow::Result<Self::Batch> {
                    Ok(self.gate.next_event().await?)
                }

                async fn activate(
                    &mut self,
                    _cx: &nuillu_module::ActivateCx<'_>,
                    batch: &Self::Batch,
                ) -> anyhow::Result<()> {
                    if let Some(vote) = self.votes.borrow_mut().pop_front().flatten() {
                        batch.respond(vote);
                    }
                    if let Some(done) = self.on_seen.take() {
                        let _ = done.send(());
                    }
                    Ok(())
                }
            }
        };
    }

    activation_gate_stub!(PrimaryGateStub, "primary-gate");
    activation_gate_stub!(SecondaryGateStub, "secondary-gate");

    struct TimedQueryBatchRecorder {
        attention_control_inbox: AttentionControlRequestInbox,
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
            let first =
                request_question_owned(self.attention_control_inbox.next_item().await?.body);
            let mut batch = vec![first];
            batch.extend(
                self.attention_control_inbox
                    .take_ready_items()?
                    .items
                    .into_iter()
                    .map(|envelope| request_question_owned(envelope.body)),
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

    struct PendingDependencyModule {
        release: Option<oneshot::Receiver<()>>,
        activations: Rc<Cell<u8>>,
        on_done: Option<oneshot::Sender<()>>,
    }

    #[async_trait(?Send)]
    impl Module for PendingDependencyModule {
        type Batch = ();

        fn id() -> &'static str {
            "pending-dependency"
        }

        fn role_description() -> &'static str {
            "test dependency"
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
            self.activations
                .set(self.activations.get().saturating_add(1));
            if let Some(done) = self.on_done.take() {
                let _ = done.send(());
            }
            Ok(())
        }
    }

    struct ImmediateDependentModule {
        batch_sent: bool,
        on_done: Option<oneshot::Sender<()>>,
    }

    #[async_trait(?Send)]
    impl Module for ImmediateDependentModule {
        type Batch = ();

        fn id() -> &'static str {
            "immediate-dependent"
        }

        fn role_description() -> &'static str {
            "test dependent"
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
            if let Some(done) = self.on_done.take() {
                let _ = done.send(());
            }
            Ok(())
        }
    }

    struct ReleasedDependentModule {
        release: Option<oneshot::Receiver<()>>,
        on_done: Option<oneshot::Sender<()>>,
    }

    #[async_trait(?Send)]
    impl Module for ReleasedDependentModule {
        type Batch = ();

        fn id() -> &'static str {
            "released-dependent"
        }

        fn role_description() -> &'static str {
            "test dependent"
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
            if let Some(done) = self.on_done.take() {
                let _ = done.send(());
            }
            Ok(())
        }
    }

    struct GatedKickTargetModule {
        batch_sent: bool,
        activated: Rc<Cell<bool>>,
    }

    #[async_trait(?Send)]
    impl Module for GatedKickTargetModule {
        type Batch = ();

        fn id() -> &'static str {
            "gated-kick-target"
        }

        fn role_description() -> &'static str {
            "test gated dependency"
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
            self.activated.set(true);
            Ok(())
        }
    }

    struct BlockingAllowGateModule {
        gate: ActivationGate<GatedKickTargetModule>,
        seen: Option<oneshot::Sender<()>>,
        allow: Option<oneshot::Receiver<()>>,
    }

    #[async_trait(?Send)]
    impl Module for BlockingAllowGateModule {
        type Batch = ActivationGateEvent<GatedKickTargetModule>;

        fn id() -> &'static str {
            "blocking-allow-gate"
        }

        fn role_description() -> &'static str {
            "test blocking activation gate"
        }

        async fn next_batch(&mut self) -> anyhow::Result<Self::Batch> {
            Ok(self.gate.next_event().await?)
        }

        async fn activate(
            &mut self,
            _cx: &nuillu_module::ActivateCx<'_>,
            batch: &Self::Batch,
        ) -> anyhow::Result<()> {
            if let Some(seen) = self.seen.take() {
                let _ = seen.send(());
            }
            if let Some(allow) = self.allow.take() {
                let _ = allow.await;
            }
            batch.respond(ActivationGateVote::Allow);
            Ok(())
        }
    }

    struct GateKickDependentModule {
        release: Option<oneshot::Receiver<()>>,
        collected: Option<oneshot::Sender<()>>,
        on_done: Option<oneshot::Sender<()>>,
    }

    #[async_trait(?Send)]
    impl Module for GateKickDependentModule {
        type Batch = ();

        fn id() -> &'static str {
            "gate-kick-dependent"
        }

        fn role_description() -> &'static str {
            "test dependent"
        }

        async fn next_batch(&mut self) -> anyhow::Result<Self::Batch> {
            if let Some(release) = self.release.take() {
                let _ = release.await;
                if let Some(collected) = self.collected.take() {
                    let _ = collected.send(());
                }
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
            if let Some(done) = self.on_done.take() {
                let _ = done.send(());
            }
            Ok(())
        }
    }

    struct FailingBatchModule {
        release: Option<oneshot::Receiver<()>>,
    }

    #[async_trait(?Send)]
    impl Module for FailingBatchModule {
        type Batch = ();

        fn id() -> &'static str {
            "failing-batch"
        }

        fn role_description() -> &'static str {
            "test failing batch"
        }

        async fn next_batch(&mut self) -> anyhow::Result<Self::Batch> {
            if let Some(release) = self.release.take() {
                let _ = release.await;
            }
            anyhow::bail!("planned next_batch failure")
        }

        async fn activate(
            &mut self,
            _cx: &nuillu_module::ActivateCx<'_>,
            _batch: &Self::Batch,
        ) -> anyhow::Result<()> {
            unreachable!("failing batch never activates")
        }
    }

    struct FailingActivateAfterReleaseModule {
        entered: Option<oneshot::Sender<()>>,
        release: Option<oneshot::Receiver<()>>,
        batch_sent: bool,
    }

    #[async_trait(?Send)]
    impl Module for FailingActivateAfterReleaseModule {
        type Batch = ();

        fn id() -> &'static str {
            "failing-activate-after-release"
        }

        fn role_description() -> &'static str {
            "test failing activation"
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
            anyhow::bail!("planned activation failure")
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
                alloc.set(echo_id(), ModuleConfig::default());
                alloc.set_activation(echo_id(), ActivationRatio::ONE);

                let blackboard = Blackboard::with_allocation(alloc);
                let caps = test_caps(blackboard.clone());
                let (done_tx, done_rx) = oneshot::channel();
                let done_tx = Rc::new(RefCell::new(Some(done_tx)));
                let modules = ModuleRegistry::new()
                    .register(
                        0..=1,
                        Bpm::from_f64(60.0)..=Bpm::from_f64(60.0),
                        linear_ratio_fn,
                        {
                            let done_tx = Rc::clone(&done_tx);
                            move |caps| EchoModule {
                                attention_control_inbox: caps.attention_control_inbox(),
                                memo: caps.memo(),
                                on_done: done_tx.borrow_mut().take(),
                            }
                        },
                    )
                    .unwrap()
                    .build(&caps)
                    .await
                    .unwrap();
                let mailbox = caps.internal_harness_io().attention_control_mailbox();

                super::run(modules, test_config(), async move {
                    mailbox
                        .publish(AttentionControlRequest::query("ping"))
                        .await
                        .expect("attention request should route to echo");
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

    async fn run_gated_echo_case(
        primary_votes: Vec<Option<ActivationGateVote>>,
        secondary_votes: Vec<Option<ActivationGateVote>>,
        expect_target_activation: bool,
    ) -> (u8, Vec<nuillu_blackboard::MemoLogRecord>) {
        let target_id = ModuleId::new(GatedEchoModule::id()).unwrap();
        let primary_id = ModuleId::new(PrimaryGateStub::id()).unwrap();
        let secondary_id = ModuleId::new(SecondaryGateStub::id()).unwrap();
        let mut alloc = ResourceAllocation::default();
        alloc.set(target_id.clone(), ModuleConfig::default());
        alloc.set_activation(target_id.clone(), ActivationRatio::ONE);
        if !primary_votes.is_empty() {
            alloc.set(primary_id.clone(), ModuleConfig::default());
            alloc.set_activation(primary_id.clone(), ActivationRatio::ONE);
        }
        if !secondary_votes.is_empty() {
            alloc.set(secondary_id.clone(), ModuleConfig::default());
            alloc.set_activation(secondary_id.clone(), ActivationRatio::ONE);
        }

        let blackboard = Blackboard::with_allocation(alloc);
        let caps = test_caps(blackboard.clone());
        let activations = Rc::new(Cell::new(0_u8));
        let (target_tx, target_rx) = oneshot::channel();
        let target_tx = Rc::new(RefCell::new(Some(target_tx)));

        let primary_votes = Rc::new(RefCell::new(VecDeque::from(primary_votes)));
        let secondary_votes = Rc::new(RefCell::new(VecDeque::from(secondary_votes)));
        let mut primary_seen_rx = Vec::new();
        let mut primary_seen_tx = VecDeque::new();
        for _ in 0..primary_votes.borrow().len() {
            let (tx, rx) = oneshot::channel();
            primary_seen_tx.push_back(tx);
            primary_seen_rx.push(rx);
        }
        let mut secondary_seen_rx = Vec::new();
        let mut secondary_seen_tx = VecDeque::new();
        for _ in 0..secondary_votes.borrow().len() {
            let (tx, rx) = oneshot::channel();
            secondary_seen_tx.push_back(tx);
            secondary_seen_rx.push(rx);
        }
        let primary_seen_tx = Rc::new(RefCell::new(primary_seen_tx));
        let secondary_seen_tx = Rc::new(RefCell::new(secondary_seen_tx));

        let mut registry = ModuleRegistry::new()
            .register(
                1..=1,
                Bpm::from_f64(60.0)..=Bpm::from_f64(60.0),
                linear_ratio_fn,
                {
                    let activations = Rc::clone(&activations);
                    let target_tx = Rc::clone(&target_tx);
                    move |caps| GatedEchoModule {
                        attention_control_inbox: caps.attention_control_inbox(),
                        memo: caps.memo(),
                        activations: Rc::clone(&activations),
                        on_done: target_tx.borrow_mut().take(),
                    }
                },
            )
            .unwrap();

        if !primary_votes.borrow().is_empty() {
            registry = registry
                .register(
                    0..=u8::try_from(primary_votes.borrow().len()).unwrap(),
                    Bpm::from_f64(60.0)..=Bpm::from_f64(60.0),
                    linear_ratio_fn,
                    {
                        let primary_votes = Rc::clone(&primary_votes);
                        let primary_seen_tx = Rc::clone(&primary_seen_tx);
                        move |caps| PrimaryGateStub {
                            gate: caps.activation_gate_for::<GatedEchoModule>(),
                            votes: Rc::clone(&primary_votes),
                            on_seen: primary_seen_tx.borrow_mut().pop_front(),
                        }
                    },
                )
                .unwrap();
        }

        if !secondary_votes.borrow().is_empty() {
            registry = registry
                .register(
                    0..=u8::try_from(secondary_votes.borrow().len()).unwrap(),
                    Bpm::from_f64(60.0)..=Bpm::from_f64(60.0),
                    linear_ratio_fn,
                    {
                        let secondary_votes = Rc::clone(&secondary_votes);
                        let secondary_seen_tx = Rc::clone(&secondary_seen_tx);
                        move |caps| SecondaryGateStub {
                            gate: caps.activation_gate_for::<GatedEchoModule>(),
                            votes: Rc::clone(&secondary_votes),
                            on_seen: secondary_seen_tx.borrow_mut().pop_front(),
                        }
                    },
                )
                .unwrap();
        }

        let modules = registry.build(&caps).await.unwrap();
        let mailbox = caps.internal_harness_io().attention_control_mailbox();

        super::run(modules, test_config(), async move {
            mailbox
                .publish(AttentionControlRequest::query("gated"))
                .await
                .expect("attention request should route to gated echo");
            if expect_target_activation {
                let _ = target_rx.await;
            } else {
                for rx in primary_seen_rx {
                    let _ = rx.await;
                }
                for rx in secondary_seen_rx {
                    let _ = rx.await;
                }
            }
            for _ in 0..4 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("scheduler returned err");

        let memos = blackboard.read(|bb| bb.recent_memo_logs()).await;
        (activations.get(), memos)
    }

    #[tokio::test(flavor = "current_thread", start_paused = false)]
    async fn activation_gate_zero_votes_allows_activation() {
        let local = LocalSet::new();
        local
            .run_until(async {
                let (activations, memos) = run_gated_echo_case(Vec::new(), Vec::new(), true).await;

                assert_eq!(activations, 1);
                assert!(
                    memos
                        .iter()
                        .any(|record| record.content == "gated echo handled gated")
                );
            })
            .await;
    }

    #[tokio::test(flavor = "current_thread", start_paused = false)]
    async fn activation_gate_suppresses_single_suppress_vote() {
        let local = LocalSet::new();
        local
            .run_until(async {
                let (activations, memos) = run_gated_echo_case(
                    vec![Some(ActivationGateVote::Suppress)],
                    Vec::new(),
                    false,
                )
                .await;

                assert_eq!(activations, 0);
                assert!(
                    !memos
                        .iter()
                        .any(|record| record.content == "gated echo handled gated")
                );
            })
            .await;
    }

    #[tokio::test(flavor = "current_thread", start_paused = false)]
    async fn activation_gate_tie_suppresses_activation() {
        let local = LocalSet::new();
        local
            .run_until(async {
                let (activations, _) = run_gated_echo_case(
                    vec![Some(ActivationGateVote::Allow)],
                    vec![Some(ActivationGateVote::Suppress)],
                    false,
                )
                .await;

                assert_eq!(activations, 0);
            })
            .await;
    }

    #[tokio::test(flavor = "current_thread", start_paused = false)]
    async fn activation_gate_majority_allow_and_closed_abstain_allow_activation() {
        let local = LocalSet::new();
        local
            .run_until(async {
                let (majority_activations, _) = run_gated_echo_case(
                    vec![
                        Some(ActivationGateVote::Allow),
                        Some(ActivationGateVote::Allow),
                    ],
                    vec![Some(ActivationGateVote::Suppress)],
                    true,
                )
                .await;
                assert_eq!(majority_activations, 1);

                let (closed_activations, _) =
                    run_gated_echo_case(vec![None], Vec::new(), true).await;
                assert_eq!(closed_activations, 1);
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
                        0..=1,
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
                            occurred_at: None,
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
                alloc.set(module_id.clone(), ModuleConfig::default());
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
                        1..=1,
                        Bpm::from_f64(2000.0)..=Bpm::from_f64(2000.0),
                        linear_ratio_fn,
                        {
                            let batches = Rc::clone(&batches);
                            let first_tx = Rc::clone(&first_tx);
                            let second_tx = Rc::clone(&second_tx);
                            move |caps| QueryBatchRecorder {
                                attention_control_inbox: caps.attention_control_inbox(),
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
                let mailbox = caps.internal_harness_io().attention_control_mailbox();

                super::run(modules, test_config(), async move {
                    mailbox
                        .publish(AttentionControlRequest::query("first"))
                        .await
                        .expect("first attention request should route");
                    let _ = first_rx.await;

                    let started = tokio::time::Instant::now();
                    mailbox
                        .publish(AttentionControlRequest::query("second"))
                        .await
                        .expect("second attention request should route");
                    mailbox
                        .publish(AttentionControlRequest::query("third"))
                        .await
                        .expect("third attention request should route");
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
                alloc.set(module_id.clone(), ModuleConfig::default());
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
                        1..=1,
                        Bpm::from_f64(300.0)..=Bpm::from_f64(300.0),
                        linear_ratio_fn,
                        {
                            let batches = Rc::clone(&batches);
                            let activation_delays = Rc::clone(&activation_delays);
                            let first_tx = Rc::clone(&first_tx);
                            let second_tx = Rc::clone(&second_tx);
                            move |caps| TimedQueryBatchRecorder {
                                attention_control_inbox: caps.attention_control_inbox(),
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
                let mailbox = caps.internal_harness_io().attention_control_mailbox();

                super::run(modules, test_config(), async move {
                    mailbox
                        .publish(AttentionControlRequest::query("first"))
                        .await
                        .expect("first attention request should route");
                    let _ = first_rx.await;

                    let started = tokio::time::Instant::now();
                    mailbox
                        .publish(AttentionControlRequest::query("second"))
                        .await
                        .expect("second attention request should route");
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
                alloc.set(module_id.clone(), ModuleConfig::default());
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
                        1..=1,
                        Bpm::from_f64(500.0)..=Bpm::from_f64(500.0),
                        linear_ratio_fn,
                        {
                            let batches = Rc::clone(&batches);
                            let activation_delays = Rc::clone(&activation_delays);
                            let first_tx = Rc::clone(&first_tx);
                            let second_tx = Rc::clone(&second_tx);
                            move |caps| TimedQueryBatchRecorder {
                                attention_control_inbox: caps.attention_control_inbox(),
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
                let mailbox = caps.internal_harness_io().attention_control_mailbox();

                super::run(modules, test_config(), async move {
                    mailbox
                        .publish(AttentionControlRequest::query("first"))
                        .await
                        .expect("first attention request should route");
                    let _ = first_rx.await;

                    let started = tokio::time::Instant::now();
                    mailbox
                        .publish(AttentionControlRequest::query("second"))
                        .await
                        .expect("second attention request should route");
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
                alloc.set(module_id.clone(), ModuleConfig::default());
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
                        1..=1,
                        Bpm::from_f64(120.0)..=Bpm::from_f64(120.0),
                        linear_ratio_fn,
                        {
                            let batches = Rc::clone(&batches);
                            let first_tx = Rc::clone(&first_tx);
                            let second_tx = Rc::clone(&second_tx);
                            move |caps| QueryBatchRecorder {
                                attention_control_inbox: caps.attention_control_inbox(),
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
                let mailbox = caps.internal_harness_io().attention_control_mailbox();

                super::run(modules, test_config(), async move {
                    mailbox
                        .publish(AttentionControlRequest::query("first"))
                        .await
                        .expect("first attention request should route");
                    let _ = first_rx.await;

                    mailbox
                        .publish(AttentionControlRequest::query("second"))
                        .await
                        .expect("second attention request should route");
                    tokio::time::sleep(Duration::from_millis(50)).await;

                    let mut raised = ResourceAllocation::default();
                    raised.set(module_id.clone(), ModuleConfig::default());
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
                alloc.set(module_id.clone(), ModuleConfig::default());
                alloc.set_activation(module_id.clone(), ActivationRatio::ONE);

                let blackboard = Blackboard::with_allocation(alloc);
                let caps = test_caps(blackboard.clone());
                let modules = ModuleRegistry::new()
                    .register(
                        1..=1,
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

    // The DelayedBatchStub is retained for future inactive/pending transition
    // coverage.

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
                alloc.set(module_id.clone(), ModuleConfig::default());
                alloc.set_activation(module_id.clone(), ActivationRatio::ONE);

                let blackboard = Blackboard::with_allocation(alloc);
                let caps = test_caps(blackboard.clone());
                let (entered_tx, entered_rx) = oneshot::channel();
                let entered_tx = Rc::new(RefCell::new(Some(entered_tx)));
                let (_release_tx, release_rx) = oneshot::channel();
                let release_rx = Rc::new(RefCell::new(Some(release_rx)));
                let modules = ModuleRegistry::new()
                    .register(
                        0..=1,
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
                alloc.set(builtin::cognition_gate(), ModuleConfig::default());
                alloc.set_activation(builtin::cognition_gate(), ActivationRatio::ONE);

                let blackboard = Blackboard::with_allocation(alloc);
                let caps = test_caps(blackboard.clone());
                let (done_tx, done_rx) = oneshot::channel();
                let done_tx = Rc::new(RefCell::new(Some(done_tx)));
                let modules = ModuleRegistry::new()
                    .register(
                        0..=1,
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
                alloc.set(retry_id.clone(), ModuleConfig::default());
                alloc.set_activation(retry_id.clone(), ActivationRatio::ONE);

                let blackboard = Blackboard::with_allocation(alloc);
                let caps = test_caps(blackboard.clone());
                let attempts = Rc::new(Cell::new(0));
                let (done_tx, done_rx) = oneshot::channel();
                let done_tx = Rc::new(RefCell::new(Some(done_tx)));
                let modules = ModuleRegistry::new()
                    .register(
                        1..=1,
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
                alloc.set(fail_id.clone(), ModuleConfig::default());
                alloc.set_activation(fail_id.clone(), ActivationRatio::ONE);

                let blackboard = Blackboard::with_allocation(alloc);
                let caps = test_caps(blackboard.clone());
                let modules = ModuleRegistry::new()
                    .register(
                        1..=1,
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

    // Inactive replicas keep state and queued inbox messages, but the scheduler
    // does not start semantic work for them until allocation makes them active.

    #[tokio::test(flavor = "current_thread", start_paused = true)]
    async fn kick_silent_timeout_noops_kick_without_canceling_dependency_next_batch() {
        let local = LocalSet::new();
        local
            .run_until(async {
                let dependency_id = ModuleId::new(PendingDependencyModule::id()).unwrap();
                let dependent_id = ModuleId::new(ImmediateDependentModule::id()).unwrap();
                let mut alloc = ResourceAllocation::default();
                alloc.set(dependency_id.clone(), ModuleConfig::default());
                alloc.set_activation(dependency_id.clone(), ActivationRatio::ONE);
                alloc.set(dependent_id.clone(), ModuleConfig::default());
                alloc.set_activation(dependent_id.clone(), ActivationRatio::ONE);

                let blackboard = Blackboard::with_allocation(alloc);
                let caps = test_caps(blackboard);
                let dependency_activations = Rc::new(Cell::new(0_u8));
                let observed_dependency_activations = Rc::clone(&dependency_activations);
                let (dependency_release_tx, dependency_release_rx) = oneshot::channel();
                let dependency_release_rx = Rc::new(RefCell::new(Some(dependency_release_rx)));
                let (dependency_done_tx, dependency_done_rx) = oneshot::channel();
                let dependency_done_tx = Rc::new(RefCell::new(Some(dependency_done_tx)));
                let (dependent_done_tx, dependent_done_rx) = oneshot::channel();
                let dependent_done_tx = Rc::new(RefCell::new(Some(dependent_done_tx)));

                let modules = ModuleRegistry::new()
                    .register(
                        1..=1,
                        Bpm::from_f64(60.0)..=Bpm::from_f64(60.0),
                        linear_ratio_fn,
                        {
                            let dependency_activations = Rc::clone(&dependency_activations);
                            let dependency_release_rx = Rc::clone(&dependency_release_rx);
                            let dependency_done_tx = Rc::clone(&dependency_done_tx);
                            move |_| PendingDependencyModule {
                                release: dependency_release_rx.borrow_mut().take(),
                                activations: Rc::clone(&dependency_activations),
                                on_done: dependency_done_tx.borrow_mut().take(),
                            }
                        },
                    )
                    .unwrap()
                    .register(
                        1..=1,
                        Bpm::from_f64(60.0)..=Bpm::from_f64(60.0),
                        linear_ratio_fn,
                        {
                            let dependent_done_tx = Rc::clone(&dependent_done_tx);
                            move |_| ImmediateDependentModule {
                                batch_sent: false,
                                on_done: dependent_done_tx.borrow_mut().take(),
                            }
                        },
                    )
                    .unwrap()
                    .depends_on(dependent_id, dependency_id)
                    .build(&caps)
                    .await
                    .unwrap();

                super::run(modules, test_config(), async move {
                    for _ in 0..4 {
                        tokio::task::yield_now().await;
                    }
                    tokio::time::advance(Duration::from_secs(1)).await;
                    let _ = dependent_done_rx.await;
                    assert_eq!(observed_dependency_activations.get(), 0);
                    let _ = dependency_release_tx.send(());
                    let _ = dependency_done_rx.await;
                })
                .await
                .expect("scheduler returned err");

                assert_eq!(dependency_activations.get(), 1);
            })
            .await;
    }

    #[tokio::test(flavor = "current_thread", start_paused = false)]
    async fn dependency_active_at_spawn_but_inactive_at_kick_send_is_skipped() {
        let local = LocalSet::new();
        local
            .run_until(async {
                let dependency_id = ModuleId::new(PendingDependencyModule::id()).unwrap();
                let dependent_id = ModuleId::new(ReleasedDependentModule::id()).unwrap();
                let mut alloc = ResourceAllocation::default();
                alloc.set(dependency_id.clone(), ModuleConfig::default());
                alloc.set_activation(dependency_id.clone(), ActivationRatio::ONE);
                alloc.set(dependent_id.clone(), ModuleConfig::default());
                alloc.set_activation(dependent_id.clone(), ActivationRatio::ONE);

                let blackboard = Blackboard::with_allocation(alloc);
                let caps = test_caps(blackboard.clone());
                let dependency_activations = Rc::new(Cell::new(0_u8));
                let (_dependency_release_tx, dependency_release_rx) = oneshot::channel();
                let dependency_release_rx = Rc::new(RefCell::new(Some(dependency_release_rx)));
                let (dependent_release_tx, dependent_release_rx) = oneshot::channel();
                let dependent_release_rx = Rc::new(RefCell::new(Some(dependent_release_rx)));
                let (dependent_done_tx, dependent_done_rx) = oneshot::channel();
                let dependent_done_tx = Rc::new(RefCell::new(Some(dependent_done_tx)));

                let modules = ModuleRegistry::new()
                    .register(
                        0..=1,
                        Bpm::from_f64(60.0)..=Bpm::from_f64(60.0),
                        linear_ratio_fn,
                        {
                            let dependency_activations = Rc::clone(&dependency_activations);
                            let dependency_release_rx = Rc::clone(&dependency_release_rx);
                            move |_| PendingDependencyModule {
                                release: dependency_release_rx.borrow_mut().take(),
                                activations: Rc::clone(&dependency_activations),
                                on_done: None,
                            }
                        },
                    )
                    .unwrap()
                    .register(
                        1..=1,
                        Bpm::from_f64(60.0)..=Bpm::from_f64(60.0),
                        linear_ratio_fn,
                        {
                            let dependent_release_rx = Rc::clone(&dependent_release_rx);
                            let dependent_done_tx = Rc::clone(&dependent_done_tx);
                            move |_| ReleasedDependentModule {
                                release: dependent_release_rx.borrow_mut().take(),
                                on_done: dependent_done_tx.borrow_mut().take(),
                            }
                        },
                    )
                    .unwrap()
                    .depends_on(dependent_id.clone(), dependency_id.clone())
                    .build(&caps)
                    .await
                    .unwrap();

                super::run(modules, test_config(), async move {
                    for _ in 0..4 {
                        tokio::task::yield_now().await;
                    }
                    let mut lowered = ResourceAllocation::default();
                    lowered.set(dependency_id, ModuleConfig::default());
                    lowered.set_activation(
                        ModuleId::new(PendingDependencyModule::id()).unwrap(),
                        ActivationRatio::ZERO,
                    );
                    lowered.set(dependent_id, ModuleConfig::default());
                    lowered.set_activation(
                        ModuleId::new(ReleasedDependentModule::id()).unwrap(),
                        ActivationRatio::ONE,
                    );
                    blackboard
                        .apply(BlackboardCommand::SetAllocation(lowered))
                        .await;
                    let _ = dependent_release_tx.send(());
                    tokio::time::timeout(Duration::from_millis(100), dependent_done_rx)
                        .await
                        .expect("dependent should not wait for an inactive dependency")
                        .expect("dependent activation sender dropped");
                })
                .await
                .expect("scheduler returned err");

                assert_eq!(dependency_activations.get(), 0);
            })
            .await;
    }

    #[tokio::test(flavor = "current_thread", start_paused = false)]
    async fn gate_allow_after_target_inactive_noops_pending_dependency_kicks() {
        let local = LocalSet::new();
        local
            .run_until(async {
                let target_id = ModuleId::new(GatedKickTargetModule::id()).unwrap();
                let gate_id = ModuleId::new(BlockingAllowGateModule::id()).unwrap();
                let dependent_id = ModuleId::new(GateKickDependentModule::id()).unwrap();
                let mut alloc = ResourceAllocation::default();
                alloc.set(target_id.clone(), ModuleConfig::default());
                alloc.set_activation(target_id.clone(), ActivationRatio::ONE);
                alloc.set(gate_id.clone(), ModuleConfig::default());
                alloc.set_activation(gate_id.clone(), ActivationRatio::ONE);
                alloc.set(dependent_id.clone(), ModuleConfig::default());
                alloc.set_activation(dependent_id.clone(), ActivationRatio::ONE);

                let blackboard = Blackboard::with_allocation(alloc);
                let caps = test_caps(blackboard.clone());
                let target_activated = Rc::new(Cell::new(false));
                let observed_target_activated = Rc::clone(&target_activated);
                let (gate_seen_tx, gate_seen_rx) = oneshot::channel();
                let gate_seen_tx = Rc::new(RefCell::new(Some(gate_seen_tx)));
                let (gate_allow_tx, gate_allow_rx) = oneshot::channel();
                let gate_allow_rx = Rc::new(RefCell::new(Some(gate_allow_rx)));
                let (dependent_release_tx, dependent_release_rx) = oneshot::channel();
                let dependent_release_rx = Rc::new(RefCell::new(Some(dependent_release_rx)));
                let (dependent_collected_tx, dependent_collected_rx) = oneshot::channel();
                let dependent_collected_tx = Rc::new(RefCell::new(Some(dependent_collected_tx)));
                let (dependent_done_tx, dependent_done_rx) = oneshot::channel();
                let dependent_done_tx = Rc::new(RefCell::new(Some(dependent_done_tx)));

                let modules = ModuleRegistry::new()
                    .register(
                        0..=1,
                        Bpm::from_f64(60.0)..=Bpm::from_f64(60.0),
                        linear_ratio_fn,
                        {
                            let target_activated = Rc::clone(&target_activated);
                            move |_| GatedKickTargetModule {
                                batch_sent: false,
                                activated: Rc::clone(&target_activated),
                            }
                        },
                    )
                    .unwrap()
                    .register(
                        1..=1,
                        Bpm::from_f64(60.0)..=Bpm::from_f64(60.0),
                        linear_ratio_fn,
                        {
                            let gate_seen_tx = Rc::clone(&gate_seen_tx);
                            let gate_allow_rx = Rc::clone(&gate_allow_rx);
                            move |caps| BlockingAllowGateModule {
                                gate: caps.activation_gate_for::<GatedKickTargetModule>(),
                                seen: gate_seen_tx.borrow_mut().take(),
                                allow: gate_allow_rx.borrow_mut().take(),
                            }
                        },
                    )
                    .unwrap()
                    .register(
                        1..=1,
                        Bpm::from_f64(60.0)..=Bpm::from_f64(60.0),
                        linear_ratio_fn,
                        {
                            let dependent_release_rx = Rc::clone(&dependent_release_rx);
                            let dependent_collected_tx = Rc::clone(&dependent_collected_tx);
                            let dependent_done_tx = Rc::clone(&dependent_done_tx);
                            move |_| GateKickDependentModule {
                                release: dependent_release_rx.borrow_mut().take(),
                                collected: dependent_collected_tx.borrow_mut().take(),
                                on_done: dependent_done_tx.borrow_mut().take(),
                            }
                        },
                    )
                    .unwrap()
                    .depends_on(dependent_id.clone(), target_id.clone())
                    .build(&caps)
                    .await
                    .unwrap();

                super::run(modules, test_config(), async move {
                    let _ = gate_seen_rx.await;
                    let _ = dependent_release_tx.send(());
                    let _ = dependent_collected_rx.await;
                    for _ in 0..4 {
                        tokio::task::yield_now().await;
                    }

                    let mut lowered = ResourceAllocation::default();
                    lowered.set(target_id, ModuleConfig::default());
                    lowered.set_activation(
                        ModuleId::new(GatedKickTargetModule::id()).unwrap(),
                        ActivationRatio::ZERO,
                    );
                    lowered.set(gate_id, ModuleConfig::default());
                    lowered.set_activation(
                        ModuleId::new(BlockingAllowGateModule::id()).unwrap(),
                        ActivationRatio::ONE,
                    );
                    lowered.set(dependent_id, ModuleConfig::default());
                    lowered.set_activation(
                        ModuleId::new(GateKickDependentModule::id()).unwrap(),
                        ActivationRatio::ONE,
                    );
                    blackboard
                        .apply(BlackboardCommand::SetAllocation(lowered))
                        .await;

                    let _ = gate_allow_tx.send(());
                    tokio::time::timeout(Duration::from_millis(100), dependent_done_rx)
                        .await
                        .expect("dependent kick should be nooped when the gate target is inactive")
                        .expect("dependent activation sender dropped");
                    assert!(!observed_target_activated.get());
                })
                .await
                .expect("scheduler returned err");

                assert!(!target_activated.get());
            })
            .await;
    }

    #[tokio::test(flavor = "current_thread", start_paused = false)]
    async fn next_batch_error_noops_pending_kicks_before_reporting_error() {
        use futures::StreamExt as _;

        let local = LocalSet::new();
        local
            .run_until(async {
                let (release_tx, release_rx) = oneshot::channel();
                let release_rx = Rc::new(RefCell::new(Some(release_rx)));
                let caps = test_caps(Blackboard::default());
                let modules = ModuleRegistry::new()
                    .register(
                        1..=1,
                        Bpm::from_f64(60.0)..=Bpm::from_f64(60.0),
                        linear_ratio_fn,
                        {
                            let release_rx = Rc::clone(&release_rx);
                            move |_| FailingBatchModule {
                                release: release_rx.borrow_mut().take(),
                            }
                        },
                    )
                    .unwrap()
                    .build(&caps)
                    .await
                    .unwrap();
                let (runtime, mut modules, dependencies) = modules.into_parts_with_dependencies();
                let module = modules.pop().unwrap();
                let sender = module.owner().clone();
                let (kick_inbox, kick_handle) = crate::kicks::KickInbox::new();
                let completion = kick_handle.send(sender);
                let deps_resolver = super::DepsResolver {
                    runtime,
                    dependencies: Arc::new(dependencies),
                    kick_targets_by_role: Arc::new(std::collections::HashMap::new()),
                };
                let mut tasks = futures::stream::FuturesUnordered::new();
                let subscriber = tracing::dispatcher::get_default(Clone::clone);
                let parent = tracing::Span::current();

                super::spawn_next_batch(
                    &mut tasks,
                    0,
                    module,
                    kick_inbox,
                    deps_resolver,
                    &parent,
                    &subscriber,
                );
                tokio::task::yield_now().await;
                let _ = release_tx.send(());

                let message = tasks
                    .next()
                    .await
                    .expect("next_batch task should finish")
                    .expect("next_batch task should not panic");
                match message {
                    super::TaskMessage::NextBatch { result, .. } => {
                        let error = match result {
                            Ok(_) => panic!("next_batch should fail"),
                            Err(error) => error,
                        };
                        assert!(error.contains("planned next_batch failure"));
                    }
                    _ => panic!("expected next_batch task message"),
                }
                completion
                    .await
                    .expect("pending kick should be completed on next_batch error");
            })
            .await;
    }

    #[tokio::test(flavor = "current_thread", start_paused = false)]
    async fn dependency_pull_drains_kicks_that_arrive_while_waiting_for_dependency() {
        use futures::StreamExt as _;

        let local = LocalSet::new();
        local
            .run_until(async {
                let dependency_id = ModuleId::new(PendingDependencyModule::id()).unwrap();
                let dependent_id = ModuleId::new(ImmediateDependentModule::id()).unwrap();
                let mut alloc = ResourceAllocation::default();
                alloc.set(dependency_id.clone(), ModuleConfig::default());
                alloc.set_activation(dependency_id.clone(), ActivationRatio::ONE);
                alloc.set(dependent_id.clone(), ModuleConfig::default());
                alloc.set_activation(dependent_id.clone(), ActivationRatio::ONE);

                let blackboard = Blackboard::with_allocation(alloc);
                let caps = test_caps(blackboard);
                let modules = ModuleRegistry::new()
                    .register(
                        1..=1,
                        Bpm::from_f64(60.0)..=Bpm::from_f64(60.0),
                        linear_ratio_fn,
                        |_| PendingDependencyModule {
                            release: None,
                            activations: Rc::new(Cell::new(0)),
                            on_done: None,
                        },
                    )
                    .unwrap()
                    .register(
                        1..=1,
                        Bpm::from_f64(60.0)..=Bpm::from_f64(60.0),
                        linear_ratio_fn,
                        |_| ImmediateDependentModule {
                            batch_sent: false,
                            on_done: None,
                        },
                    )
                    .unwrap()
                    .depends_on(dependent_id.clone(), dependency_id.clone())
                    .build(&caps)
                    .await
                    .unwrap();
                let (runtime, mut modules, dependencies) = modules.into_parts_with_dependencies();
                let dependent_index = modules
                    .iter()
                    .position(|module| module.owner().module == dependent_id)
                    .unwrap();
                let dependent = modules.remove(dependent_index);
                let dependent_owner = dependent.owner().clone();
                let dependency_owner = nuillu_types::ModuleInstanceId::new(
                    dependency_id,
                    nuillu_types::ReplicaIndex::ZERO,
                );
                let (dependent_kick_inbox, dependent_kick_handle) = crate::kicks::KickInbox::new();
                let (mut dependency_kick_inbox, dependency_kick_handle) =
                    crate::kicks::KickInbox::new();
                let mut kick_targets = std::collections::HashMap::new();
                kick_targets.insert(
                    dependency_owner.module.clone(),
                    vec![super::KickTarget {
                        owner: dependency_owner.clone(),
                        handle: dependency_kick_handle,
                    }],
                );
                let deps_resolver = super::DepsResolver {
                    runtime,
                    dependencies: Arc::new(dependencies),
                    kick_targets_by_role: Arc::new(kick_targets),
                };
                let mut tasks = futures::stream::FuturesUnordered::new();
                let subscriber = tracing::dispatcher::get_default(Clone::clone);
                let parent = tracing::Span::current();

                super::spawn_next_batch(
                    &mut tasks,
                    0,
                    dependent,
                    dependent_kick_inbox,
                    deps_resolver,
                    &parent,
                    &subscriber,
                );
                let dependency_kick = dependency_kick_inbox
                    .next()
                    .await
                    .expect("dependent should kick its dependency");
                let dependent_completion = dependent_kick_handle.send(dependency_owner);
                dependency_kick.notify_finish();

                let message = tasks
                    .next()
                    .await
                    .expect("next_batch task should finish")
                    .expect("next_batch task should not panic");
                match message {
                    super::TaskMessage::NextBatch {
                        result: Ok((_batch, pending_kicks)),
                        ..
                    } => {
                        assert_eq!(pending_kicks.len(), 1);
                        for kick in pending_kicks {
                            kick.notify_finish();
                        }
                    }
                    _ => panic!("expected successful next_batch with drained pending kick"),
                }
                dependent_completion
                    .await
                    .expect("kick arriving during dependency pull should be carried forward");
                assert_eq!(dependent_owner.module, dependent_id);
            })
            .await;
    }

    #[tokio::test(flavor = "current_thread", start_paused = false)]
    async fn activation_failure_noops_ready_kicks_before_reporting_error() {
        use futures::StreamExt as _;

        let local = LocalSet::new();
        local
            .run_until(async {
                let module_id = ModuleId::new(FailingActivateAfterReleaseModule::id()).unwrap();
                let mut alloc = ResourceAllocation::default();
                alloc.set(module_id.clone(), ModuleConfig::default());
                alloc.set_activation(module_id.clone(), ActivationRatio::ONE);

                let blackboard = Blackboard::with_allocation(alloc);
                let caps = test_caps(blackboard);
                let (entered_tx, entered_rx) = oneshot::channel();
                let entered_tx = Rc::new(RefCell::new(Some(entered_tx)));
                let (release_tx, release_rx) = oneshot::channel();
                let release_rx = Rc::new(RefCell::new(Some(release_rx)));
                let modules = ModuleRegistry::new()
                    .register(
                        1..=1,
                        Bpm::from_f64(60.0)..=Bpm::from_f64(60.0),
                        linear_ratio_fn,
                        {
                            let entered_tx = Rc::clone(&entered_tx);
                            let release_rx = Rc::clone(&release_rx);
                            move |_| FailingActivateAfterReleaseModule {
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
                let (runtime, mut modules, _) = modules.into_parts_with_dependencies();
                let mut module = modules.pop().unwrap();
                let owner = module.owner().clone();
                let batch = module.next_batch().await.unwrap();
                let (kick_inbox, kick_handle) = crate::kicks::KickInbox::new();
                let mut tasks = futures::stream::FuturesUnordered::new();
                let subscriber = tracing::dispatcher::get_default(Clone::clone);
                let parent = tracing::Span::current();

                super::spawn_activate(
                    &mut tasks,
                    0,
                    module,
                    kick_inbox,
                    Vec::new(),
                    batch,
                    test_config(),
                    runtime.clone(),
                    runtime.module_catalog(),
                    runtime.identity_memories().await,
                    &parent,
                    &subscriber,
                );
                let _ = entered_rx.await;
                let completion = kick_handle.send(owner.clone());
                let _ = release_tx.send(());

                let message = tasks
                    .next()
                    .await
                    .expect("activation task should finish")
                    .expect("activation task should not panic");
                let mut states = vec![super::ModuleState::Activating];
                let mut kick_inboxes = Vec::new();
                kick_inboxes.push(None);
                let mut followup_tasks = futures::stream::FuturesUnordered::new();
                let err = super::handle_task_message(
                    message,
                    &runtime,
                    std::slice::from_ref(&owner),
                    &mut states,
                    &mut followup_tasks,
                    &mut kick_inboxes,
                    test_config(),
                    &parent,
                    &subscriber,
                )
                .await
                .expect_err("activation failure should be reported");
                assert!(matches!(
                    err,
                    SchedulerError::ModuleTaskFailed { phase: "activate", message, .. }
                        if message.contains("planned activation failure")
                ));
                completion
                    .await
                    .expect("ready kick should be completed before activation error escapes");
            })
            .await;
    }

    #[tokio::test(flavor = "current_thread", start_paused = true)]
    async fn kick_silent_timeout_debounces_when_more_kicks_arrive() {
        use futures::StreamExt as _;

        let local = LocalSet::new();
        local
            .run_until(async {
                let module_id = ModuleId::new(HangingBatchStub::id()).unwrap();
                let mut alloc = ResourceAllocation::default();
                alloc.set(module_id.clone(), ModuleConfig::default());
                alloc.set_activation(module_id.clone(), ActivationRatio::ONE);

                let blackboard = Blackboard::with_allocation(alloc);
                let caps = test_caps(blackboard);
                let modules = ModuleRegistry::new()
                    .register(
                        1..=1,
                        Bpm::from_f64(60.0)..=Bpm::from_f64(60.0),
                        linear_ratio_fn,
                        |_| HangingBatchStub,
                    )
                    .unwrap()
                    .build(&caps)
                    .await
                    .unwrap();
                let (runtime, mut modules, dependencies) = modules.into_parts_with_dependencies();
                let module = modules.pop().unwrap();
                let owner = module.owner().clone();
                let (kick_inbox, kick_handle) = crate::kicks::KickInbox::new();
                let deps_resolver = super::DepsResolver {
                    runtime,
                    dependencies: Arc::new(dependencies),
                    kick_targets_by_role: Arc::new(std::collections::HashMap::new()),
                };
                let mut tasks = futures::stream::FuturesUnordered::new();
                let subscriber = tracing::dispatcher::get_default(Clone::clone);
                let parent = tracing::Span::current();

                super::spawn_next_batch(
                    &mut tasks,
                    0,
                    module,
                    kick_inbox,
                    deps_resolver,
                    &parent,
                    &subscriber,
                );
                let mut first = kick_handle.send(owner.clone());
                tokio::task::yield_now().await;
                tokio::time::advance(Duration::from_millis(500)).await;
                let mut second = kick_handle.send(owner);
                tokio::task::yield_now().await;
                tokio::time::advance(Duration::from_millis(600)).await;
                tokio::task::yield_now().await;

                assert!(matches!(
                    first.try_recv(),
                    Err(tokio::sync::oneshot::error::TryRecvError::Empty)
                ));
                assert!(matches!(
                    second.try_recv(),
                    Err(tokio::sync::oneshot::error::TryRecvError::Empty)
                ));

                tokio::time::advance(Duration::from_millis(401)).await;
                tokio::task::yield_now().await;
                first.await.expect("first kick should be nooped");
                second.await.expect("second kick should be nooped");

                for task in tasks.iter() {
                    task.abort();
                }
                while tasks.next().await.is_some() {}
            })
            .await;
    }

    #[tokio::test(flavor = "current_thread", start_paused = false)]
    async fn idle_deadlock_records_marker_and_publishes_cognition_log_update() {
        let local = LocalSet::new();
        local
            .run_until(async {
                let observer_id = ModuleId::new(DeadlockObserver::id()).unwrap();
                let mut alloc = ResourceAllocation::default();
                alloc.set(observer_id.clone(), ModuleConfig::default());
                alloc.set_activation(observer_id.clone(), ActivationRatio::ONE);

                let blackboard = Blackboard::with_allocation(alloc);
                let caps = test_caps(blackboard.clone());
                let (done_tx, done_rx) = oneshot::channel();
                let done_tx = Rc::new(RefCell::new(Some(done_tx)));
                let modules = ModuleRegistry::new()
                    .register(
                        1..=1,
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
