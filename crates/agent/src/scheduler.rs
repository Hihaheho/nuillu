use std::collections::HashMap;
use std::future::Future;
use std::pin::pin;
use std::sync::Arc;
use std::time::Duration;

use chrono::{DateTime, Utc};
use futures::stream::{FuturesUnordered, StreamExt};
use nuillu_blackboard::{
    ActivationRatio, CorePolicyRecord, IdentityMemoryRecord, ZeroReplicaWindowPolicy,
};
use nuillu_module::{
    ActivateCx, ActivationGateVote, AgentRuntimeControl, AllocatedModule, AllocatedModules,
    LlmRequestMetadata, LlmRequestSource, ModuleBatch, ModuleDependencies, ModuleRunStatus,
    ports::Clock,
};
use nuillu_types::{ModelTier, ModuleId, ModuleInstanceId, ReplicaIndex, builtin};
use thiserror::Error;
use tokio::sync::oneshot;
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
    let mut kick_handles = Vec::with_capacity(owners.len());
    let mut target_indexes_by_role = HashMap::<ModuleId, Vec<usize>>::new();
    let mut replica_zero_index_by_role = HashMap::<ModuleId, usize>::new();
    for (index, owner) in owners.iter().enumerate() {
        let (inbox, handle) = KickInbox::new();
        kick_inboxes.push(Some(inbox));
        kick_handles.push(handle);
        if owner.replica == ReplicaIndex::ZERO {
            replica_zero_index_by_role.insert(owner.module.clone(), index);
        }
        target_indexes_by_role
            .entry(owner.module.clone())
            .or_default()
            .push(index);
    }
    let dependency_targets = DependencyTargets {
        dependencies: Arc::new(dependencies),
        target_indexes_by_role: Arc::new(target_indexes_by_role),
    };
    let mut states = modules
        .into_iter()
        .map(|module| ModuleState::Stored {
            module,
            next_batch_throttle: None,
        })
        .collect::<Vec<_>>();
    let mut active = vec![false; states.len()];
    let mut zero_windows = ZeroReplicaWindows::new(runtime.zero_replica_window_policies().await);
    let mut zero_window_wakers = std::iter::repeat_with(|| None)
        .take(owners.len())
        .collect::<Vec<_>>();
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
        &mut zero_window_wakers,
        &mut zero_windows,
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
                abort_tasks(&mut tasks).await;
                return Ok(());
            },
            joined = tasks.next(), if !tasks.is_empty() => {
                match joined {
                    Some(Ok(message)) => {
                        if let Err(error) = handle_task_message(
                            message,
                            &runtime,
                            &owners,
                            &mut states,
                            &mut tasks,
                            &mut kick_inboxes,
                            &mut zero_window_wakers,
                            &kick_handles,
                            &dependency_targets,
                            &replica_zero_index_by_role,
                            &mut zero_windows,
                            config,
                            &parent,
                            &subscriber,
                        ).await {
                            abort_tasks(&mut tasks).await;
                            return Err(error);
                        }
                        refresh_active_and_schedule(
                            &runtime,
                            &owners,
                            &mut active,
                            &mut states,
                            &mut tasks,
                            &mut kick_inboxes,
                            &mut zero_window_wakers,
                            &mut zero_windows,
                            config,
                            &parent,
                            &subscriber,
                        )
                        .await;
                    }
                    Some(Err(e)) => {
                        let message = e.to_string();
                        tracing::error!(error = ?e, "module task panicked");
                        abort_tasks(&mut tasks).await;
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

async fn abort_tasks(tasks: &mut FuturesUnordered<JoinHandle<TaskMessage>>) {
    for handle in tasks.iter() {
        handle.abort();
    }
    while tasks.next().await.is_some() {}
}

enum ModuleState {
    Stored {
        module: AllocatedModule,
        next_batch_throttle: Option<NextBatchThrottle>,
    },
    WaitingForActivation,
    CoolingDown,
    Awaiting,
    PendingBatch {
        module: AllocatedModule,
        batch: ModuleBatch,
        gate_approved: bool,
        pending_kicks: Vec<Kick>,
    },
    PendingDependencyFlush,
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
    DependencyFlush {
        index: usize,
        module: AllocatedModule,
        kick_inbox: KickInbox,
        batch: ModuleBatch,
        pending_kicks: Vec<Kick>,
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
    ActivationWaitReady {
        index: usize,
        module: AllocatedModule,
        kick_inbox: KickInbox,
        next_batch_throttle: Option<NextBatchThrottle>,
    },
}

struct NextBatchThrottle {
    not_before: DateTime<Utc>,
    activation_threshold: ActivationRatio,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ZeroReplicaPermit {
    Open,
    InFlight,
}

#[derive(Debug, Clone)]
struct ZeroReplicaWindowState {
    controller_activation_period: u32,
    controller_activations: u32,
    permit: Option<ZeroReplicaPermit>,
}

impl ZeroReplicaWindowState {
    fn new(policy: ZeroReplicaWindowPolicy) -> Option<Self> {
        Some(Self {
            controller_activation_period: policy.controller_activation_period()?,
            controller_activations: 0,
            permit: None,
        })
    }

    fn reset(&mut self) {
        self.controller_activations = 0;
        self.permit = None;
    }

    fn record_zero_controller_activation(&mut self) -> bool {
        if self.permit.is_some() {
            return false;
        }
        self.controller_activations = self.controller_activations.saturating_add(1);
        if self.controller_activations >= self.controller_activation_period {
            self.controller_activations = 0;
            self.permit = Some(ZeroReplicaPermit::Open);
            true
        } else {
            false
        }
    }
}

#[derive(Debug, Clone)]
struct ZeroReplicaWindows {
    states: HashMap<ModuleId, ZeroReplicaWindowState>,
}

impl ZeroReplicaWindows {
    fn new(policies: HashMap<ModuleId, ZeroReplicaWindowPolicy>) -> Self {
        let states = policies
            .into_iter()
            .filter_map(|(module, policy)| {
                ZeroReplicaWindowState::new(policy).map(|state| (module, state))
            })
            .collect();
        Self { states }
    }

    fn allows(&self, owner: &ModuleInstanceId) -> bool {
        owner.replica == ReplicaIndex::ZERO
            && self
                .states
                .get(&owner.module)
                .is_some_and(|state| state.permit.is_some())
    }

    fn mark_batch_accepted(&mut self, owner: &ModuleInstanceId) {
        if owner.replica != ReplicaIndex::ZERO {
            return;
        }
        if let Some(state) = self.states.get_mut(&owner.module)
            && matches!(state.permit, Some(ZeroReplicaPermit::Open))
        {
            state.permit = Some(ZeroReplicaPermit::InFlight);
        }
    }

    fn finish(&mut self, owner: &ModuleInstanceId) {
        if owner.replica != ReplicaIndex::ZERO {
            return;
        }
        if let Some(state) = self.states.get_mut(&owner.module)
            && matches!(state.permit, Some(ZeroReplicaPermit::InFlight))
        {
            state.permit = None;
        }
    }

    fn reset_allocation_active(&mut self, owners: &[ModuleInstanceId], active: &[bool]) {
        for (owner, active) in owners.iter().zip(active.iter().copied()) {
            if active && let Some(state) = self.states.get_mut(&owner.module) {
                state.reset();
            }
        }
    }

    async fn record_controller_activation(
        &mut self,
        runtime: &AgentRuntimeControl,
    ) -> Vec<ModuleId> {
        let modules = self.states.keys().cloned().collect::<Vec<_>>();
        let mut opened = Vec::new();
        for module in modules {
            let active_replicas = runtime.active_replicas(&module).await;
            let Some(state) = self.states.get_mut(&module) else {
                continue;
            };
            if active_replicas > 0 {
                state.reset();
                continue;
            }
            if state.record_zero_controller_activation() {
                opened.push(module);
            }
        }
        opened
    }
}

#[derive(Clone)]
struct DependencyTargets {
    dependencies: Arc<ModuleDependencies>,
    target_indexes_by_role: Arc<HashMap<ModuleId, Vec<usize>>>,
}

impl DependencyTargets {
    fn target_indexes(&self, dependent: &ModuleId) -> Vec<usize> {
        let mut indexes = Vec::new();
        for dep_id in self.dependencies.deps_of(dependent) {
            let Some(targets) = self.target_indexes_by_role.get(dep_id) else {
                continue;
            };
            indexes.extend(targets.iter().copied());
        }
        indexes
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
    zero_window_wakers: &mut [Option<oneshot::Sender<()>>],
    zero_windows: &mut ZeroReplicaWindows,
    config: AgentEventLoopConfig,
    parent: &tracing::Span,
    subscriber: &tracing::Dispatch,
) {
    let mut allocation_active = vec![false; owners.len()];
    for (index, owner) in owners.iter().enumerate() {
        allocation_active[index] = runtime.is_active(owner).await;
    }
    zero_windows.reset_allocation_active(owners, &allocation_active);
    for (index, owner) in owners.iter().enumerate() {
        active[index] = allocation_active[index] || zero_windows.allows(owner);
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
                        spawn_next_batch(tasks, index, module, kick_inbox, parent, subscriber);
                    }
                } else {
                    spawn_next_batch(tasks, index, module, kick_inbox, parent, subscriber);
                }
            }
            ModuleState::Stored { .. } => {
                runtime
                    .record_module_status(owners[index].clone(), ModuleRunStatus::Inactive)
                    .await;
                let ModuleState::Stored {
                    module,
                    next_batch_throttle,
                } = std::mem::replace(&mut states[index], ModuleState::WaitingForActivation)
                else {
                    unreachable!("module state changed while scheduling activation wait");
                };
                let kick_inbox = kick_inboxes[index]
                    .take()
                    .expect("kick inbox available for inactive stored module");
                let activation_waiter = runtime.activation_waiter(&owners[index]).await;
                let (zero_window_waker, zero_window_waiter) = oneshot::channel();
                zero_window_wakers[index] = Some(zero_window_waker);
                spawn_activation_wait(
                    tasks,
                    index,
                    module,
                    kick_inbox,
                    next_batch_throttle,
                    activation_waiter,
                    zero_window_waiter,
                    parent,
                    subscriber,
                );
            }
            ModuleState::WaitingForActivation => {
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
                zero_windows.mark_batch_accepted(&owners[index]);
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
                    let core_policies = runtime.core_policies().await;
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
                        core_policies,
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
            ModuleState::PendingDependencyFlush => {
                let status = if active[index] {
                    ModuleRunStatus::PendingBatch
                } else {
                    ModuleRunStatus::Inactive
                };
                runtime
                    .record_module_status(owners[index].clone(), status)
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
    zero_window_wakers: &mut [Option<oneshot::Sender<()>>],
    kick_handles: &[KickHandle],
    dependency_targets: &DependencyTargets,
    replica_zero_index_by_role: &HashMap<ModuleId, usize>,
    zero_windows: &mut ZeroReplicaWindows,
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
                if scheduling_active(runtime, zero_windows, &owners[index]).await {
                    zero_windows.mark_batch_accepted(&owners[index]);
                    let scheduled = spawn_dependency_flush_or_activate(
                        runtime,
                        tasks,
                        index,
                        owners[index].clone(),
                        module,
                        kick_inbox,
                        pending_kicks,
                        batch,
                        states,
                        kick_handles,
                        dependency_targets,
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
        TaskMessage::DependencyFlush {
            index,
            module,
            mut kick_inbox,
            batch,
            mut pending_kicks,
        } => {
            pending_kicks.extend(kick_inbox.drain_ready());
            if scheduling_active(runtime, zero_windows, &owners[index]).await {
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
                zero_windows.finish(&owners[index]);
                if owners[index].module == builtin::attention_controller() {
                    let opened = zero_windows.record_controller_activation(runtime).await;
                    wake_zero_window_modules(
                        opened,
                        replica_zero_index_by_role,
                        zero_window_wakers,
                    );
                }
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
                if scheduling_active(runtime, zero_windows, &owners[index]).await {
                    runtime
                        .record_module_status(owners[index].clone(), ModuleRunStatus::Activating)
                        .await;
                    states[index] = ModuleState::Activating;
                    let catalog = runtime.module_catalog();
                    let identity_memories = runtime.identity_memories().await;
                    let core_policies = runtime.core_policies().await;
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
                        core_policies,
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
                zero_windows.finish(&owners[index]);
                states[index] = ModuleState::Stored {
                    module,
                    next_batch_throttle: None,
                };
            }
            Ok(())
        }
        TaskMessage::ActivationWaitReady {
            index,
            module,
            kick_inbox,
            next_batch_throttle,
        } => {
            zero_window_wakers[index] = None;
            kick_inboxes[index] = Some(kick_inbox);
            states[index] = ModuleState::Stored {
                module,
                next_batch_throttle,
            };
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

async fn scheduling_active(
    runtime: &AgentRuntimeControl,
    zero_windows: &mut ZeroReplicaWindows,
    owner: &ModuleInstanceId,
) -> bool {
    if runtime.is_active(owner).await {
        zero_windows.reset_allocation_active(std::slice::from_ref(owner), &[true]);
        true
    } else {
        zero_windows.allows(owner)
    }
}

fn wake_zero_window_modules(
    modules: Vec<ModuleId>,
    replica_zero_index_by_role: &HashMap<ModuleId, usize>,
    zero_window_wakers: &mut [Option<oneshot::Sender<()>>],
) {
    for module in modules {
        let Some(index) = replica_zero_index_by_role.get(&module).copied() else {
            continue;
        };
        if let Some(waker) = zero_window_wakers[index].take() {
            let _ = waker.send(());
        }
    }
}

#[allow(clippy::too_many_arguments)]
async fn spawn_dependency_flush_or_activate(
    runtime: &AgentRuntimeControl,
    tasks: &mut FuturesUnordered<JoinHandle<TaskMessage>>,
    index: usize,
    owner: ModuleInstanceId,
    module: AllocatedModule,
    kick_inbox: KickInbox,
    pending_kicks: Vec<Kick>,
    batch: ModuleBatch,
    states: &mut [ModuleState],
    kick_handles: &[KickHandle],
    dependency_targets: &DependencyTargets,
    config: AgentEventLoopConfig,
    parent: &tracing::Span,
    subscriber: &tracing::Dispatch,
) -> ActivationScheduling {
    let completions = collect_dependency_flush_completions(
        index,
        &owner,
        states,
        kick_handles,
        dependency_targets,
    );
    if completions.is_empty() {
        spawn_activation_gate_or_activate(
            runtime,
            tasks,
            index,
            owner,
            module,
            kick_inbox,
            pending_kicks,
            batch,
            config,
            parent,
            subscriber,
        )
        .await
    } else {
        runtime
            .record_module_status(owner, ModuleRunStatus::PendingBatch)
            .await;
        spawn_dependency_flush_wait(
            tasks,
            index,
            module,
            kick_inbox,
            pending_kicks,
            batch,
            completions,
            parent,
            subscriber,
        );
        ActivationScheduling::PendingDependencyFlush
    }
}

fn collect_dependency_flush_completions(
    dependent_index: usize,
    sender: &ModuleInstanceId,
    states: &mut [ModuleState],
    kick_handles: &[KickHandle],
    dependency_targets: &DependencyTargets,
) -> Vec<tokio::sync::oneshot::Receiver<()>> {
    let mut completions = Vec::new();
    for target_index in dependency_targets.target_indexes(&sender.module) {
        if target_index == dependent_index {
            continue;
        }
        match &mut states[target_index] {
            ModuleState::CoolingDown
            | ModuleState::Awaiting
            | ModuleState::PendingDependencyFlush
            | ModuleState::PendingActivationGate
            | ModuleState::Activating => {
                completions.push(kick_handles[target_index].send(sender.clone()))
            }
            ModuleState::PendingBatch { pending_kicks, .. } => {
                let (kick, completion) = Kick::new(sender.clone());
                pending_kicks.push(kick);
                completions.push(completion);
            }
            ModuleState::Stored { .. } | ModuleState::WaitingForActivation => {}
        }
    }
    completions
}

#[allow(clippy::too_many_arguments)]
fn spawn_dependency_flush_wait(
    tasks: &mut FuturesUnordered<JoinHandle<TaskMessage>>,
    index: usize,
    module: AllocatedModule,
    mut kick_inbox: KickInbox,
    mut pending_kicks: Vec<Kick>,
    batch: ModuleBatch,
    completions: Vec<tokio::sync::oneshot::Receiver<()>>,
    parent: &tracing::Span,
    subscriber: &tracing::Dispatch,
) {
    tasks.push(spawn_local(
        async move {
            // Dependency activation can include LLM work, so this waits for
            // kick completion/drop rather than applying a wall-clock timeout.
            for completion in completions {
                let _ = completion.await;
            }
            pending_kicks.extend(kick_inbox.drain_ready());
            TaskMessage::DependencyFlush {
                index,
                module,
                kick_inbox,
                batch,
                pending_kicks,
            }
        }
        .instrument(parent.clone())
        .with_subscriber(subscriber.clone()),
    ));
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
        let core_policies = runtime.core_policies().await;
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
            core_policies,
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
    PendingDependencyFlush,
    PendingActivationGate,
}

impl ActivationScheduling {
    fn module_state(self) -> ModuleState {
        match self {
            Self::Activating => ModuleState::Activating,
            Self::PendingDependencyFlush => ModuleState::PendingDependencyFlush,
            Self::PendingActivationGate => ModuleState::PendingActivationGate,
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn spawn_activation_wait(
    tasks: &mut FuturesUnordered<JoinHandle<TaskMessage>>,
    index: usize,
    module: AllocatedModule,
    mut kick_inbox: KickInbox,
    next_batch_throttle: Option<NextBatchThrottle>,
    activation_waiter: Option<tokio::sync::oneshot::Receiver<()>>,
    zero_window_waiter: tokio::sync::oneshot::Receiver<()>,
    parent: &tracing::Span,
    subscriber: &tracing::Dispatch,
) {
    tasks.push(spawn_local(
        async move {
            let mut activation = pin!(async move {
                if let Some(waiter) = activation_waiter {
                    let _ = waiter.await;
                }
            });
            let mut zero_window = pin!(zero_window_waiter);
            let mut kick_inbox_closed = false;
            loop {
                tokio::select! {
                    biased;
                    _ = &mut activation => break,
                    _ = &mut zero_window => break,
                    maybe_kick = kick_inbox.next(), if !kick_inbox_closed => {
                        if let Some(kick) = maybe_kick {
                            kick.notify_finish();
                        } else {
                            kick_inbox_closed = true;
                        }
                    }
                }
            }
            notify_pending_and_ready(Vec::new(), &mut kick_inbox);
            TaskMessage::ActivationWaitReady {
                index,
                module,
                kick_inbox,
                next_batch_throttle,
            }
        }
        .instrument(parent.clone())
        .with_subscriber(subscriber.clone()),
    ));
}

fn spawn_next_batch(
    tasks: &mut FuturesUnordered<JoinHandle<TaskMessage>>,
    index: usize,
    mut module: AllocatedModule,
    mut kick_inbox: KickInbox,
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
    core_policies: Vec<CorePolicyRecord>,
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
                &core_policies,
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
    core_policies: &[CorePolicyRecord],
    batch: &ModuleBatch,
    activate_retries: u8,
) -> (AllocatedModule, Result<(), String>) {
    let module_owner = module.owner().clone();
    let cx = ActivateCx::new(
        catalog,
        identity_memories,
        core_policies,
        runtime
            .session_compaction_lutum()
            .clone()
            .with_extension(LlmRequestMetadata {
                owner: module_owner,
                tier: ModelTier::Cheap,
                source: LlmRequestSource::SessionCompaction,
            }),
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
    use std::collections::{HashMap, VecDeque};
    use std::rc::Rc;
    use std::sync::Arc;
    use std::time::Duration;

    use async_trait::async_trait;
    use nuillu_blackboard::{
        ActivationRatio, Blackboard, BlackboardCommand, Bpm, IdentityMemoryRecord, ModuleConfig,
        ModulePolicy, ModuleRunStatus, ResourceAllocation, ZeroReplicaWindowPolicy,
        linear_ratio_fn,
    };
    use nuillu_module::{
        ActivationGate, ActivationGateEvent, ActivationGateVote, AttentionControlRequest,
        AttentionControlRequestInbox, CognitionLogUpdated, CognitionLogUpdatedInbox,
        CognitionWriter, LlmRequestMetadata, LlmRequestSource, Memo, Module, ModuleDependencies,
        ModuleRegistry,
    };
    use nuillu_types::{
        MemoryContent, MemoryIndex, ModelTier, ModuleId, ReplicaCapRange, ReplicaIndex, builtin,
    };
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

    fn compaction_observer_id() -> ModuleId {
        ModuleId::new("compaction-observer").unwrap()
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

    struct ControllerTickModule {
        attention_control_inbox: AttentionControlRequestInbox,
        activations: Rc<Cell<u32>>,
    }

    #[async_trait(?Send)]
    impl Module for ControllerTickModule {
        type Batch = AttentionControlRequest;

        fn id() -> &'static str {
            "attention-controller"
        }

        fn role_description() -> &'static str {
            "test attention controller"
        }

        async fn next_batch(&mut self) -> anyhow::Result<Self::Batch> {
            Ok(self.attention_control_inbox.next_item().await?.body)
        }

        async fn activate(
            &mut self,
            _cx: &nuillu_module::ActivateCx<'_>,
            _batch: &Self::Batch,
        ) -> anyhow::Result<()> {
            self.activations
                .set(self.activations.get().saturating_add(1));
            Ok(())
        }
    }

    macro_rules! zero_window_target {
        ($name:ident, $id:literal) => {
            struct $name {
                attention_control_inbox: AttentionControlRequestInbox,
                activations: Rc<Cell<u32>>,
                batches: Rc<RefCell<Vec<String>>>,
            }

            #[async_trait(?Send)]
            impl Module for $name {
                type Batch = AttentionControlRequest;

                fn id() -> &'static str {
                    $id
                }

                fn role_description() -> &'static str {
                    "test zero-replica target"
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
                    self.batches
                        .borrow_mut()
                        .push(request_question(batch).to_owned());
                    Ok(())
                }
            }
        };
    }

    zero_window_target!(ZeroWindowTarget, "zero-window-target");
    zero_window_target!(ZeroWindowTargetA, "zero-window-target-a");
    zero_window_target!(ZeroWindowTargetB, "zero-window-target-b");
    zero_window_target!(
        HardDisabledZeroWindowTarget,
        "hard-disabled-zero-window-target"
    );

    struct CompactionMetadataObserver {
        attention_control_inbox: AttentionControlRequestInbox,
        on_seen: Option<oneshot::Sender<Option<LlmRequestMetadata>>>,
    }

    #[async_trait(?Send)]
    impl Module for CompactionMetadataObserver {
        type Batch = AttentionControlRequest;

        fn id() -> &'static str {
            "compaction-observer"
        }

        fn role_description() -> &'static str {
            "test stub"
        }

        async fn next_batch(&mut self) -> anyhow::Result<Self::Batch> {
            Ok(self.attention_control_inbox.next_item().await?.body)
        }

        async fn activate(
            &mut self,
            cx: &nuillu_module::ActivateCx<'_>,
            _batch: &Self::Batch,
        ) -> anyhow::Result<()> {
            if let Some(tx) = self.on_seen.take() {
                let metadata = cx
                    .session_compaction_lutum()
                    .default_extensions()
                    .get::<LlmRequestMetadata>()
                    .cloned();
                let _ = tx.send(metadata);
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

    macro_rules! silent_dependency_stub {
        ($name:ident, $id:literal) => {
            struct $name;

            #[async_trait(?Send)]
            impl Module for $name {
                type Batch = ();

                fn id() -> &'static str {
                    $id
                }

                fn role_description() -> &'static str {
                    "test silent dependency"
                }

                async fn next_batch(&mut self) -> anyhow::Result<Self::Batch> {
                    std::future::pending().await
                }

                async fn activate(
                    &mut self,
                    _cx: &nuillu_module::ActivateCx<'_>,
                    _batch: &Self::Batch,
                ) -> anyhow::Result<()> {
                    unreachable!("silent dependency never produces a batch")
                }
            }
        };
    }

    silent_dependency_stub!(SilentDependencyA, "silent-dependency-a");
    silent_dependency_stub!(SilentDependencyB, "silent-dependency-b");
    silent_dependency_stub!(SilentDependencyC, "silent-dependency-c");

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

    fn assert_oneshot_pending<T>(receiver: &mut oneshot::Receiver<T>, message: &str) {
        assert!(
            matches!(
                receiver.try_recv(),
                Err(tokio::sync::oneshot::error::TryRecvError::Empty)
            ),
            "{message}"
        );
    }

    async fn wait_for_cell_count(cell: &Rc<Cell<u32>>, expected: u32) {
        for _ in 0..100 {
            if cell.get() >= expected {
                return;
            }
            tokio::task::yield_now().await;
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
        panic!("expected count >= {expected}, got {}", cell.get());
    }

    fn set_activation(
        allocation: &mut ResourceAllocation,
        module: ModuleId,
        ratio: ActivationRatio,
    ) {
        allocation.set(module.clone(), ModuleConfig::default());
        allocation.set_activation(module, ratio);
    }

    fn fast_bpm() -> std::ops::RangeInclusive<Bpm> {
        Bpm::from_f64(60_000.0)..=Bpm::from_f64(60_000.0)
    }

    fn test_policy(
        replicas_range: std::ops::RangeInclusive<u8>,
        rate_limit_range: std::ops::RangeInclusive<Bpm>,
    ) -> ModulePolicy {
        ModulePolicy::new(
            ReplicaCapRange::new(*replicas_range.start(), *replicas_range.end()).unwrap(),
            rate_limit_range,
            linear_ratio_fn,
        )
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
                        test_policy(0..=1, Bpm::from_f64(60.0)..=Bpm::from_f64(60.0)),
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

    #[tokio::test(flavor = "current_thread", start_paused = false)]
    async fn activation_context_stamps_session_compaction_lutum_owner() {
        let local = LocalSet::new();
        local
            .run_until(async {
                let mut alloc = ResourceAllocation::default();
                alloc.set(compaction_observer_id(), ModuleConfig::default());
                alloc.set_activation(compaction_observer_id(), ActivationRatio::ONE);

                let blackboard = Blackboard::with_allocation(alloc);
                let caps = test_caps(blackboard);
                let (seen_tx, seen_rx) = oneshot::channel();
                let seen_tx = Rc::new(RefCell::new(Some(seen_tx)));
                let modules = ModuleRegistry::new()
                    .register(
                        test_policy(0..=1, Bpm::from_f64(60.0)..=Bpm::from_f64(60.0)),
                        {
                            let seen_tx = Rc::clone(&seen_tx);
                            move |caps| CompactionMetadataObserver {
                                attention_control_inbox: caps.attention_control_inbox(),
                                on_seen: seen_tx.borrow_mut().take(),
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
                        .expect("attention request should route to observer");
                    let metadata = seen_rx.await.expect("metadata observed");
                    assert_eq!(
                        metadata,
                        Some(LlmRequestMetadata {
                            owner: nuillu_types::ModuleInstanceId::new(
                                compaction_observer_id(),
                                ReplicaIndex::ZERO,
                            ),
                            tier: ModelTier::Cheap,
                            source: LlmRequestSource::SessionCompaction,
                        })
                    );
                    for _ in 0..4 {
                        tokio::task::yield_now().await;
                    }
                })
                .await
                .expect("scheduler returned err");
            })
            .await;
    }

    #[tokio::test(flavor = "current_thread", start_paused = false)]
    async fn zero_replica_window_runs_queued_work_after_default_controller_period() {
        let local = LocalSet::new();
        local
            .run_until(async {
                let controller_id = builtin::attention_controller();
                let target_id = ModuleId::new(ZeroWindowTarget::id()).unwrap();
                let mut alloc = ResourceAllocation::default();
                set_activation(&mut alloc, controller_id.clone(), ActivationRatio::ONE);
                set_activation(&mut alloc, target_id.clone(), ActivationRatio::ZERO);

                let blackboard = Blackboard::with_allocation(alloc);
                let caps = test_caps(blackboard.clone());
                let controller_activations = Rc::new(Cell::new(0_u32));
                let target_activations = Rc::new(Cell::new(0_u32));
                let target_batches = Rc::new(RefCell::new(Vec::<String>::new()));
                let modules = ModuleRegistry::new()
                    .register(test_policy(1..=1, fast_bpm()), {
                        let controller_activations = Rc::clone(&controller_activations);
                        move |caps| ControllerTickModule {
                            attention_control_inbox: caps.attention_control_inbox(),
                            activations: Rc::clone(&controller_activations),
                        }
                    })
                    .unwrap()
                    .register(test_policy(0..=1, fast_bpm()), {
                        let target_activations = Rc::clone(&target_activations);
                        let target_batches = Rc::clone(&target_batches);
                        move |caps| ZeroWindowTarget {
                            attention_control_inbox: caps.attention_control_inbox(),
                            activations: Rc::clone(&target_activations),
                            batches: Rc::clone(&target_batches),
                        }
                    })
                    .unwrap()
                    .build(&caps)
                    .await
                    .unwrap();
                let mailbox = caps.internal_harness_io().attention_control_mailbox();

                super::run(modules, test_config(), async move {
                    for i in 1..=2 {
                        mailbox
                            .publish(AttentionControlRequest::query(format!("tick-{i}")))
                            .await
                            .expect("controller tick should route");
                        wait_for_cell_count(&controller_activations, i).await;
                        for _ in 0..4 {
                            tokio::task::yield_now().await;
                        }
                        assert_eq!(target_activations.get(), 0);
                    }

                    mailbox
                        .publish(AttentionControlRequest::query("tick-3"))
                        .await
                        .expect("third controller tick should route");
                    wait_for_cell_count(&controller_activations, 3).await;
                    wait_for_cell_count(&target_activations, 1).await;

                    assert_eq!(
                        blackboard
                            .read(|bb| bb.allocation().active_replicas(&target_id))
                            .await,
                        0
                    );
                })
                .await
                .expect("scheduler returned err");

                assert_eq!(target_batches.borrow().len(), 1);
            })
            .await;
    }

    #[tokio::test(flavor = "current_thread", start_paused = false)]
    async fn zero_replica_window_period_is_configurable_per_module() {
        let local = LocalSet::new();
        local
            .run_until(async {
                let controller_id = builtin::attention_controller();
                let target_a_id = ModuleId::new(ZeroWindowTargetA::id()).unwrap();
                let target_b_id = ModuleId::new(ZeroWindowTargetB::id()).unwrap();
                let mut alloc = ResourceAllocation::default();
                set_activation(&mut alloc, controller_id, ActivationRatio::ONE);
                set_activation(&mut alloc, target_a_id, ActivationRatio::ZERO);
                set_activation(&mut alloc, target_b_id, ActivationRatio::ZERO);

                let blackboard = Blackboard::with_allocation(alloc);
                let caps = test_caps(blackboard);
                let controller_activations = Rc::new(Cell::new(0_u32));
                let target_a_activations = Rc::new(Cell::new(0_u32));
                let target_b_activations = Rc::new(Cell::new(0_u32));
                let target_a_batches = Rc::new(RefCell::new(Vec::<String>::new()));
                let target_b_batches = Rc::new(RefCell::new(Vec::<String>::new()));
                let modules = ModuleRegistry::new()
                    .register(test_policy(1..=1, fast_bpm()), {
                        let controller_activations = Rc::clone(&controller_activations);
                        move |caps| ControllerTickModule {
                            attention_control_inbox: caps.attention_control_inbox(),
                            activations: Rc::clone(&controller_activations),
                        }
                    })
                    .unwrap()
                    .register(
                        {
                            let mut policy = test_policy(0..=1, fast_bpm());
                            policy.zero_replica_window =
                                ZeroReplicaWindowPolicy::EveryControllerActivations(2);
                            policy
                        },
                        {
                            let target_a_activations = Rc::clone(&target_a_activations);
                            let target_a_batches = Rc::clone(&target_a_batches);
                            move |caps| ZeroWindowTargetA {
                                attention_control_inbox: caps.attention_control_inbox(),
                                activations: Rc::clone(&target_a_activations),
                                batches: Rc::clone(&target_a_batches),
                            }
                        },
                    )
                    .unwrap()
                    .register(
                        {
                            let mut policy = test_policy(0..=1, fast_bpm());
                            policy.zero_replica_window =
                                ZeroReplicaWindowPolicy::EveryControllerActivations(4);
                            policy
                        },
                        {
                            let target_b_activations = Rc::clone(&target_b_activations);
                            let target_b_batches = Rc::clone(&target_b_batches);
                            move |caps| ZeroWindowTargetB {
                                attention_control_inbox: caps.attention_control_inbox(),
                                activations: Rc::clone(&target_b_activations),
                                batches: Rc::clone(&target_b_batches),
                            }
                        },
                    )
                    .unwrap()
                    .build(&caps)
                    .await
                    .unwrap();
                let mailbox = caps.internal_harness_io().attention_control_mailbox();

                super::run(modules, test_config(), async move {
                    for i in 1..=2 {
                        mailbox
                            .publish(AttentionControlRequest::query(format!("period-{i}")))
                            .await
                            .expect("controller tick should route");
                        wait_for_cell_count(&controller_activations, i).await;
                    }
                    wait_for_cell_count(&target_a_activations, 1).await;
                    assert_eq!(target_b_activations.get(), 0);

                    for i in 3..=4 {
                        mailbox
                            .publish(AttentionControlRequest::query(format!("period-{i}")))
                            .await
                            .expect("controller tick should route");
                        wait_for_cell_count(&controller_activations, i).await;
                    }
                    wait_for_cell_count(&target_a_activations, 2).await;
                    wait_for_cell_count(&target_b_activations, 1).await;
                })
                .await
                .expect("scheduler returned err");

                assert_eq!(target_a_batches.borrow().len(), 2);
                assert_eq!(target_b_batches.borrow().len(), 1);
            })
            .await;
    }

    #[tokio::test(flavor = "current_thread", start_paused = false)]
    async fn zero_replica_window_counter_resets_when_module_becomes_active() {
        let local = LocalSet::new();
        local
            .run_until(async {
                let controller_id = builtin::attention_controller();
                let target_id = ModuleId::new(ZeroWindowTarget::id()).unwrap();
                let mut alloc = ResourceAllocation::default();
                set_activation(&mut alloc, controller_id.clone(), ActivationRatio::ONE);
                set_activation(&mut alloc, target_id.clone(), ActivationRatio::ZERO);

                let blackboard = Blackboard::with_allocation(alloc);
                let caps = test_caps(blackboard.clone());
                let controller_activations = Rc::new(Cell::new(0_u32));
                let target_activations = Rc::new(Cell::new(0_u32));
                let target_batches = Rc::new(RefCell::new(Vec::<String>::new()));
                let modules = ModuleRegistry::new()
                    .register(test_policy(1..=1, fast_bpm()), {
                        let controller_activations = Rc::clone(&controller_activations);
                        move |caps| ControllerTickModule {
                            attention_control_inbox: caps.attention_control_inbox(),
                            activations: Rc::clone(&controller_activations),
                        }
                    })
                    .unwrap()
                    .register(test_policy(0..=1, fast_bpm()), {
                        let target_activations = Rc::clone(&target_activations);
                        let target_batches = Rc::clone(&target_batches);
                        move |caps| ZeroWindowTarget {
                            attention_control_inbox: caps.attention_control_inbox(),
                            activations: Rc::clone(&target_activations),
                            batches: Rc::clone(&target_batches),
                        }
                    })
                    .unwrap()
                    .build(&caps)
                    .await
                    .unwrap();
                let mailbox = caps.internal_harness_io().attention_control_mailbox();

                super::run(modules, test_config(), async move {
                    for i in 1..=2 {
                        mailbox
                            .publish(AttentionControlRequest::query(format!("pre-{i}")))
                            .await
                            .expect("controller tick should route");
                        wait_for_cell_count(&controller_activations, i).await;
                    }
                    assert_eq!(target_activations.get(), 0);

                    let mut raised = ResourceAllocation::default();
                    set_activation(&mut raised, controller_id.clone(), ActivationRatio::ONE);
                    set_activation(&mut raised, target_id.clone(), ActivationRatio::ONE);
                    blackboard
                        .apply(BlackboardCommand::SetAllocation(raised))
                        .await;
                    wait_for_cell_count(&target_activations, 1).await;

                    let mut lowered = ResourceAllocation::default();
                    set_activation(&mut lowered, controller_id, ActivationRatio::ONE);
                    set_activation(&mut lowered, target_id, ActivationRatio::ZERO);
                    blackboard
                        .apply(BlackboardCommand::SetAllocation(lowered))
                        .await;
                    let post_reset_baseline = target_activations.get();

                    for i in 3..=4 {
                        mailbox
                            .publish(AttentionControlRequest::query(format!("post-{i}")))
                            .await
                            .expect("controller tick should route");
                        wait_for_cell_count(&controller_activations, i).await;
                    }
                    for _ in 0..4 {
                        tokio::task::yield_now().await;
                    }
                    assert_eq!(target_activations.get(), post_reset_baseline);

                    mailbox
                        .publish(AttentionControlRequest::query("post-5"))
                        .await
                        .expect("third post-reset tick should route");
                    wait_for_cell_count(&controller_activations, 5).await;
                    wait_for_cell_count(&target_activations, post_reset_baseline + 1).await;
                })
                .await
                .expect("scheduler returned err");

                assert!(target_batches.borrow().len() >= 2);
            })
            .await;
    }

    #[tokio::test(flavor = "current_thread", start_paused = false)]
    async fn zero_replica_window_is_one_shot_until_next_full_period() {
        let local = LocalSet::new();
        local
            .run_until(async {
                let controller_id = builtin::attention_controller();
                let target_id = ModuleId::new(ZeroWindowTarget::id()).unwrap();
                let mut alloc = ResourceAllocation::default();
                set_activation(&mut alloc, controller_id, ActivationRatio::ONE);
                set_activation(&mut alloc, target_id, ActivationRatio::ZERO);

                let blackboard = Blackboard::with_allocation(alloc);
                let caps = test_caps(blackboard);
                let controller_activations = Rc::new(Cell::new(0_u32));
                let target_activations = Rc::new(Cell::new(0_u32));
                let target_batches = Rc::new(RefCell::new(Vec::<String>::new()));
                let modules = ModuleRegistry::new()
                    .register(test_policy(1..=1, fast_bpm()), {
                        let controller_activations = Rc::clone(&controller_activations);
                        move |caps| ControllerTickModule {
                            attention_control_inbox: caps.attention_control_inbox(),
                            activations: Rc::clone(&controller_activations),
                        }
                    })
                    .unwrap()
                    .register(
                        {
                            let mut policy = test_policy(0..=1, fast_bpm());
                            policy.zero_replica_window =
                                ZeroReplicaWindowPolicy::EveryControllerActivations(2);
                            policy
                        },
                        {
                            let target_activations = Rc::clone(&target_activations);
                            let target_batches = Rc::clone(&target_batches);
                            move |caps| ZeroWindowTarget {
                                attention_control_inbox: caps.attention_control_inbox(),
                                activations: Rc::clone(&target_activations),
                                batches: Rc::clone(&target_batches),
                            }
                        },
                    )
                    .unwrap()
                    .build(&caps)
                    .await
                    .unwrap();
                let mailbox = caps.internal_harness_io().attention_control_mailbox();

                super::run(modules, test_config(), async move {
                    for i in 1..=2 {
                        mailbox
                            .publish(AttentionControlRequest::query(format!("one-shot-{i}")))
                            .await
                            .expect("controller tick should route");
                        wait_for_cell_count(&controller_activations, i).await;
                    }
                    wait_for_cell_count(&target_activations, 1).await;

                    mailbox
                        .publish(AttentionControlRequest::query("one-shot-3"))
                        .await
                        .expect("next queued message should not immediately re-open");
                    wait_for_cell_count(&controller_activations, 3).await;
                    for _ in 0..4 {
                        tokio::task::yield_now().await;
                    }
                    assert_eq!(target_activations.get(), 1);

                    mailbox
                        .publish(AttentionControlRequest::query("one-shot-4"))
                        .await
                        .expect("second period tick should re-open");
                    wait_for_cell_count(&controller_activations, 4).await;
                    wait_for_cell_count(&target_activations, 2).await;
                })
                .await
                .expect("scheduler returned err");

                assert_eq!(target_batches.borrow().len(), 2);
            })
            .await;
    }

    #[tokio::test(flavor = "current_thread", start_paused = false)]
    async fn zero_replica_window_policy_can_disable_module() {
        let local = LocalSet::new();
        local
            .run_until(async {
                let controller_id = builtin::attention_controller();
                let target_id = ModuleId::new(ZeroWindowTarget::id()).unwrap();
                let mut alloc = ResourceAllocation::default();
                set_activation(&mut alloc, controller_id, ActivationRatio::ONE);
                set_activation(&mut alloc, target_id, ActivationRatio::ZERO);

                let blackboard = Blackboard::with_allocation(alloc);
                let caps = test_caps(blackboard);
                let controller_activations = Rc::new(Cell::new(0_u32));
                let target_activations = Rc::new(Cell::new(0_u32));
                let target_batches = Rc::new(RefCell::new(Vec::<String>::new()));
                let modules = ModuleRegistry::new()
                    .register(test_policy(1..=1, fast_bpm()), {
                        let controller_activations = Rc::clone(&controller_activations);
                        move |caps| ControllerTickModule {
                            attention_control_inbox: caps.attention_control_inbox(),
                            activations: Rc::clone(&controller_activations),
                        }
                    })
                    .unwrap()
                    .register(
                        {
                            let mut policy = test_policy(0..=1, fast_bpm());
                            policy.zero_replica_window = ZeroReplicaWindowPolicy::Disabled;
                            policy
                        },
                        {
                            let target_activations = Rc::clone(&target_activations);
                            let target_batches = Rc::clone(&target_batches);
                            move |caps| ZeroWindowTarget {
                                attention_control_inbox: caps.attention_control_inbox(),
                                activations: Rc::clone(&target_activations),
                                batches: Rc::clone(&target_batches),
                            }
                        },
                    )
                    .unwrap()
                    .build(&caps)
                    .await
                    .unwrap();
                let mailbox = caps.internal_harness_io().attention_control_mailbox();

                super::run(modules, test_config(), async move {
                    for i in 1..=4 {
                        mailbox
                            .publish(AttentionControlRequest::query(format!("disabled-{i}")))
                            .await
                            .expect("controller tick should route");
                        wait_for_cell_count(&controller_activations, i).await;
                    }
                    for _ in 0..8 {
                        tokio::task::yield_now().await;
                    }
                    assert_eq!(target_activations.get(), 0);
                })
                .await
                .expect("scheduler returned err");

                assert!(target_batches.borrow().is_empty());
            })
            .await;
    }

    #[tokio::test(flavor = "current_thread", start_paused = false)]
    async fn zero_replica_window_skips_hard_disabled_modules() {
        let local = LocalSet::new();
        local
            .run_until(async {
                let controller_id = builtin::attention_controller();
                let target_id = ModuleId::new(HardDisabledZeroWindowTarget::id()).unwrap();
                let mut alloc = ResourceAllocation::default();
                set_activation(&mut alloc, controller_id, ActivationRatio::ONE);
                set_activation(&mut alloc, target_id, ActivationRatio::ZERO);

                let blackboard = Blackboard::with_allocation(alloc);
                let caps = test_caps(blackboard);
                let controller_activations = Rc::new(Cell::new(0_u32));
                let target_activations = Rc::new(Cell::new(0_u32));
                let target_batches = Rc::new(RefCell::new(Vec::<String>::new()));
                let modules = ModuleRegistry::new()
                    .register(test_policy(1..=1, fast_bpm()), {
                        let controller_activations = Rc::clone(&controller_activations);
                        move |caps| ControllerTickModule {
                            attention_control_inbox: caps.attention_control_inbox(),
                            activations: Rc::clone(&controller_activations),
                        }
                    })
                    .unwrap()
                    .register(test_policy(0..=0, fast_bpm()), {
                        let target_activations = Rc::clone(&target_activations);
                        let target_batches = Rc::clone(&target_batches);
                        move |caps| HardDisabledZeroWindowTarget {
                            attention_control_inbox: caps.attention_control_inbox(),
                            activations: Rc::clone(&target_activations),
                            batches: Rc::clone(&target_batches),
                        }
                    })
                    .unwrap()
                    .build(&caps)
                    .await
                    .unwrap();
                let mailbox = caps.internal_harness_io().attention_control_mailbox();

                super::run(modules, test_config(), async move {
                    for i in 1..=4 {
                        mailbox
                            .publish(AttentionControlRequest::query(format!("hard-{i}")))
                            .await
                            .expect("controller tick should route");
                        wait_for_cell_count(&controller_activations, i).await;
                    }
                    for _ in 0..8 {
                        tokio::task::yield_now().await;
                    }
                    assert_eq!(target_activations.get(), 0);
                })
                .await
                .expect("scheduler returned err");

                assert!(target_batches.borrow().is_empty());
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
                test_policy(1..=1, Bpm::from_f64(60.0)..=Bpm::from_f64(60.0)),
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
                    test_policy(
                        0..=u8::try_from(primary_votes.borrow().len()).unwrap(),
                        Bpm::from_f64(60.0)..=Bpm::from_f64(60.0),
                    ),
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
                    test_policy(
                        0..=u8::try_from(secondary_votes.borrow().len()).unwrap(),
                        Bpm::from_f64(60.0)..=Bpm::from_f64(60.0),
                    ),
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
                        test_policy(0..=1, Bpm::from_f64(60.0)..=Bpm::from_f64(60.0)),
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
                        test_policy(1..=1, Bpm::from_f64(2000.0)..=Bpm::from_f64(2000.0)),
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
                        test_policy(1..=1, Bpm::from_f64(300.0)..=Bpm::from_f64(300.0)),
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
                        test_policy(1..=1, Bpm::from_f64(500.0)..=Bpm::from_f64(500.0)),
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
                        test_policy(1..=1, Bpm::from_f64(120.0)..=Bpm::from_f64(120.0)),
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
                        test_policy(1..=1, Bpm::from_f64(60.0)..=Bpm::from_f64(60.0)),
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
                        test_policy(0..=1, Bpm::from_f64(60.0)..=Bpm::from_f64(60.0)),
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
                        test_policy(0..=1, Bpm::from_f64(60.0)..=Bpm::from_f64(60.0)),
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
                        test_policy(1..=1, Bpm::from_f64(60.0)..=Bpm::from_f64(60.0)),
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
                        test_policy(1..=1, Bpm::from_f64(60.0)..=Bpm::from_f64(60.0)),
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

    #[tokio::test(flavor = "current_thread", start_paused = false)]
    async fn inactive_stored_module_runs_queued_work_after_activation() {
        let local = LocalSet::new();
        local
            .run_until(async {
                let module_id = echo_id();
                let mut alloc = ResourceAllocation::default();
                alloc.set(module_id.clone(), ModuleConfig::default());
                alloc.set_activation(module_id.clone(), ActivationRatio::ZERO);

                let blackboard = Blackboard::with_allocation(alloc);
                let caps = test_caps(blackboard.clone());
                let (done_tx, mut done_rx) = oneshot::channel();
                let done_tx = Rc::new(RefCell::new(Some(done_tx)));
                let modules = ModuleRegistry::new()
                    .register(
                        test_policy(0..=1, Bpm::from_f64(60.0)..=Bpm::from_f64(60.0)),
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

                super::run(modules, test_config(), async {
                    mailbox
                        .publish(AttentionControlRequest::query("later"))
                        .await
                        .expect("inactive replica zero should queue the request");
                    for _ in 0..4 {
                        tokio::task::yield_now().await;
                    }
                    assert_oneshot_pending(
                        &mut done_rx,
                        "inactive module should not process queued work before activation",
                    );

                    let mut raised = ResourceAllocation::default();
                    raised.set(module_id.clone(), ModuleConfig::default());
                    raised.set_activation(module_id.clone(), ActivationRatio::ONE);
                    blackboard
                        .apply(BlackboardCommand::SetAllocation(raised))
                        .await;

                    tokio::time::timeout(Duration::from_millis(100), &mut done_rx)
                        .await
                        .expect("activation should wake the stored module")
                        .expect("done sender dropped");
                })
                .await
                .expect("scheduler returned err");

                let owner = nuillu_types::ModuleInstanceId::new(module_id, ReplicaIndex::ZERO);
                let memos = blackboard.read(|bb| bb.recent_memo_logs()).await;
                assert!(
                    memos.iter().any(|record| {
                        record.owner == owner && record.content == "echoed later"
                    })
                );
            })
            .await;
    }

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
                        test_policy(1..=1, Bpm::from_f64(60.0)..=Bpm::from_f64(60.0)),
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
                        test_policy(1..=1, Bpm::from_f64(60.0)..=Bpm::from_f64(60.0)),
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

    #[tokio::test(flavor = "current_thread", start_paused = true)]
    async fn active_awaiting_dependencies_apply_one_silent_window_backpressure() {
        let local = LocalSet::new();
        local
            .run_until(async {
                let dep_a_id = ModuleId::new(SilentDependencyA::id()).unwrap();
                let dep_b_id = ModuleId::new(SilentDependencyB::id()).unwrap();
                let dep_c_id = ModuleId::new(SilentDependencyC::id()).unwrap();
                let dependent_id = ModuleId::new(ReleasedDependentModule::id()).unwrap();
                let mut alloc = ResourceAllocation::default();
                for id in [&dep_a_id, &dep_b_id, &dep_c_id, &dependent_id] {
                    alloc.set(id.clone(), ModuleConfig::default());
                    alloc.set_activation(id.clone(), ActivationRatio::ONE);
                }

                let caps = test_caps(Blackboard::with_allocation(alloc));
                let (dependent_release_tx, dependent_release_rx) = oneshot::channel();
                let dependent_release_rx = Rc::new(RefCell::new(Some(dependent_release_rx)));
                let (dependent_done_tx, mut dependent_done_rx) = oneshot::channel();
                let dependent_done_tx = Rc::new(RefCell::new(Some(dependent_done_tx)));
                let modules = ModuleRegistry::new()
                    .register(
                        test_policy(1..=1, Bpm::from_f64(60.0)..=Bpm::from_f64(60.0)),
                        |_| SilentDependencyA,
                    )
                    .unwrap()
                    .register(
                        test_policy(1..=1, Bpm::from_f64(60.0)..=Bpm::from_f64(60.0)),
                        |_| SilentDependencyB,
                    )
                    .unwrap()
                    .register(
                        test_policy(1..=1, Bpm::from_f64(60.0)..=Bpm::from_f64(60.0)),
                        |_| SilentDependencyC,
                    )
                    .unwrap()
                    .register(
                        test_policy(1..=1, Bpm::from_f64(60.0)..=Bpm::from_f64(60.0)),
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
                    .depends_on(dependent_id.clone(), dep_a_id)
                    .depends_on(dependent_id.clone(), dep_b_id)
                    .depends_on(dependent_id, dep_c_id)
                    .build(&caps)
                    .await
                    .unwrap();

                super::run(modules, test_config(), async move {
                    for _ in 0..4 {
                        tokio::task::yield_now().await;
                    }
                    let _ = dependent_release_tx.send(());
                    for _ in 0..8 {
                        tokio::task::yield_now().await;
                    }
                    assert_oneshot_pending(
                        &mut dependent_done_rx,
                        "dependent should wait for active dependency kick completions",
                    );

                    tokio::time::advance(Duration::from_millis(999)).await;
                    for _ in 0..8 {
                        tokio::task::yield_now().await;
                    }
                    assert_oneshot_pending(
                        &mut dependent_done_rx,
                        "dependency silent-window backpressure should not complete early",
                    );

                    tokio::time::advance(Duration::from_millis(1)).await;
                    for _ in 0..50 {
                        match dependent_done_rx.try_recv() {
                            Ok(()) => return,
                            Err(tokio::sync::oneshot::error::TryRecvError::Empty) => {
                                tokio::task::yield_now().await;
                            }
                            Err(tokio::sync::oneshot::error::TryRecvError::Closed) => {
                                panic!("dependent activation sender dropped");
                            }
                        }
                    }
                    panic!("dependent should activate after one shared silent window");
                })
                .await
                .expect("scheduler returned err");
            })
            .await;
    }

    #[tokio::test(flavor = "current_thread", start_paused = false)]
    async fn stored_dependency_targets_are_skipped_by_state_based_filter() {
        use futures::FutureExt as _;

        let local = LocalSet::new();
        local
            .run_until(async {
                let dependency_id = ModuleId::new(SilentDependencyA::id()).unwrap();
                let dependent_id = ModuleId::new(ImmediateDependentModule::id()).unwrap();
                let mut alloc = ResourceAllocation::default();
                alloc.set(dependency_id.clone(), ModuleConfig::default());
                alloc.set_activation(dependency_id.clone(), ActivationRatio::ONE);
                alloc.set(dependent_id.clone(), ModuleConfig::default());
                alloc.set_activation(dependent_id.clone(), ActivationRatio::ONE);

                let caps = test_caps(Blackboard::with_allocation(alloc));
                let modules = ModuleRegistry::new()
                    .register(
                        test_policy(1..=1, Bpm::from_f64(60.0)..=Bpm::from_f64(60.0)),
                        |_| SilentDependencyA,
                    )
                    .unwrap()
                    .register(
                        test_policy(1..=1, Bpm::from_f64(60.0)..=Bpm::from_f64(60.0)),
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
                let (_, modules, dependencies) = modules.into_parts_with_dependencies();
                let owners = modules
                    .iter()
                    .map(|module| module.owner().clone())
                    .collect::<Vec<_>>();
                let dependency_index = owners
                    .iter()
                    .position(|owner| owner.module == dependency_id)
                    .unwrap();
                let dependent_index = owners
                    .iter()
                    .position(|owner| owner.module == dependent_id)
                    .unwrap();

                let mut states = Vec::with_capacity(owners.len());
                let mut kick_handles = Vec::with_capacity(owners.len());
                let mut dependency_kick_inbox = None;
                for (index, module) in modules.into_iter().enumerate() {
                    let (kick_inbox, kick_handle) = crate::kicks::KickInbox::new();
                    kick_handles.push(kick_handle);
                    if index == dependency_index {
                        dependency_kick_inbox = Some(kick_inbox);
                        states.push(super::ModuleState::Stored {
                            module,
                            next_batch_throttle: None,
                        });
                    } else {
                        states.push(super::ModuleState::Awaiting);
                    }
                }

                let mut target_indexes_by_role = HashMap::new();
                target_indexes_by_role.insert(dependency_id, vec![dependency_index]);
                let dependency_targets = super::DependencyTargets {
                    dependencies: Arc::new(dependencies),
                    target_indexes_by_role: Arc::new(target_indexes_by_role),
                };
                let completions = super::collect_dependency_flush_completions(
                    dependent_index,
                    &owners[dependent_index],
                    &mut states,
                    &kick_handles,
                    &dependency_targets,
                );

                assert!(
                    completions.is_empty(),
                    "stored dependency target should reproduce the zero-completion fast path"
                );
                assert!(
                    dependency_kick_inbox
                        .as_mut()
                        .unwrap()
                        .next()
                        .now_or_never()
                        .is_none(),
                    "stored dependency target should not receive a kick"
                );
            })
            .await;
    }

    #[tokio::test(flavor = "current_thread", start_paused = true)]
    async fn dependency_kick_silent_timeout_prevents_deadlock_when_target_goes_inactive() {
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
                let (dependent_done_tx, mut dependent_done_rx) = oneshot::channel();
                let dependent_done_tx = Rc::new(RefCell::new(Some(dependent_done_tx)));

                let modules = ModuleRegistry::new()
                    .register(
                        test_policy(0..=1, Bpm::from_f64(60.0)..=Bpm::from_f64(60.0)),
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
                        test_policy(1..=1, Bpm::from_f64(60.0)..=Bpm::from_f64(60.0)),
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
                    for _ in 0..4 {
                        tokio::task::yield_now().await;
                    }
                    tokio::time::advance(Duration::from_secs(1)).await;
                    for _ in 0..4 {
                        tokio::task::yield_now().await;
                    }
                    dependent_done_rx
                        .try_recv()
                        .expect("dependency kick silent timeout should release the dependent");
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
                        test_policy(0..=1, Bpm::from_f64(60.0)..=Bpm::from_f64(60.0)),
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
                        test_policy(1..=1, Bpm::from_f64(60.0)..=Bpm::from_f64(60.0)),
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
                        test_policy(1..=1, Bpm::from_f64(60.0)..=Bpm::from_f64(60.0)),
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
                        test_policy(1..=1, Bpm::from_f64(60.0)..=Bpm::from_f64(60.0)),
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
                let (_, mut modules, _) = modules.into_parts_with_dependencies();
                let module = modules.pop().unwrap();
                let sender = module.owner().clone();
                let (kick_inbox, kick_handle) = crate::kicks::KickInbox::new();
                let completion = kick_handle.send(sender);
                let mut tasks = futures::stream::FuturesUnordered::new();
                let subscriber = tracing::dispatcher::get_default(Clone::clone);
                let parent = tracing::Span::current();

                super::spawn_next_batch(&mut tasks, 0, module, kick_inbox, &parent, &subscriber);
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
    async fn dependency_flush_drains_kicks_that_arrive_while_waiting_for_dependency() {
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
                        test_policy(1..=1, Bpm::from_f64(60.0)..=Bpm::from_f64(60.0)),
                        |_| PendingDependencyModule {
                            release: None,
                            activations: Rc::new(Cell::new(0)),
                            on_done: None,
                        },
                    )
                    .unwrap()
                    .register(
                        test_policy(1..=1, Bpm::from_f64(60.0)..=Bpm::from_f64(60.0)),
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
                let (_, mut modules, dependencies) = modules.into_parts_with_dependencies();
                let dependent_index = modules
                    .iter()
                    .position(|module| module.owner().module == dependent_id)
                    .unwrap();
                let mut dependent = modules.remove(dependent_index);
                let dependent_owner = dependent.owner().clone();
                let batch = dependent.next_batch().await.unwrap();
                let dependency_owner = nuillu_types::ModuleInstanceId::new(
                    dependency_id,
                    nuillu_types::ReplicaIndex::ZERO,
                );
                let (dependent_kick_inbox, dependent_kick_handle) = crate::kicks::KickInbox::new();
                let (mut dependency_kick_inbox, dependency_kick_handle) =
                    crate::kicks::KickInbox::new();
                let mut target_indexes_by_role = HashMap::new();
                target_indexes_by_role.insert(dependency_owner.module.clone(), vec![1]);
                let dependency_targets = super::DependencyTargets {
                    dependencies: Arc::new(dependencies),
                    target_indexes_by_role: Arc::new(target_indexes_by_role),
                };
                let mut states = vec![super::ModuleState::Awaiting, super::ModuleState::Awaiting];
                let kick_handles = vec![dependent_kick_handle.clone(), dependency_kick_handle];
                let completions = super::collect_dependency_flush_completions(
                    0,
                    &dependent_owner,
                    &mut states,
                    &kick_handles,
                    &dependency_targets,
                );
                assert_eq!(completions.len(), 1);
                let mut tasks = futures::stream::FuturesUnordered::new();
                let subscriber = tracing::dispatcher::get_default(Clone::clone);
                let parent = tracing::Span::current();

                super::spawn_dependency_flush_wait(
                    &mut tasks,
                    0,
                    dependent,
                    dependent_kick_inbox,
                    Vec::new(),
                    batch,
                    completions,
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
                    .expect("dependency flush task should finish")
                    .expect("dependency flush task should not panic");
                match message {
                    super::TaskMessage::DependencyFlush { pending_kicks, .. } => {
                        assert_eq!(pending_kicks.len(), 1);
                        for kick in pending_kicks {
                            kick.notify_finish();
                        }
                    }
                    _ => panic!("expected dependency flush task message"),
                }
                dependent_completion
                    .await
                    .expect("kick arriving during dependency flush should be carried forward");
                assert_eq!(dependent_owner.module, dependent_id);
            })
            .await;
    }

    #[tokio::test(flavor = "current_thread", start_paused = false)]
    async fn dependency_flush_continues_when_kick_completion_is_dropped() {
        use futures::StreamExt as _;

        let local = LocalSet::new();
        local
            .run_until(async {
                let module_id = ModuleId::new(ImmediateDependentModule::id()).unwrap();
                let mut alloc = ResourceAllocation::default();
                alloc.set(module_id.clone(), ModuleConfig::default());
                alloc.set_activation(module_id, ActivationRatio::ONE);

                let caps = test_caps(Blackboard::with_allocation(alloc));
                let modules = ModuleRegistry::new()
                    .register(
                        test_policy(1..=1, Bpm::from_f64(60.0)..=Bpm::from_f64(60.0)),
                        |_| ImmediateDependentModule {
                            batch_sent: false,
                            on_done: None,
                        },
                    )
                    .unwrap()
                    .build(&caps)
                    .await
                    .unwrap();
                let (_, mut modules, _) = modules.into_parts_with_dependencies();
                let mut module = modules.pop().unwrap();
                let owner = module.owner().clone();
                let batch = module.next_batch().await.unwrap();
                let (kick_inbox, _) = crate::kicks::KickInbox::new();
                let (kick, completion) = crate::kicks::Kick::new(owner);
                drop(kick);

                let mut tasks = futures::stream::FuturesUnordered::new();
                let subscriber = tracing::dispatcher::get_default(Clone::clone);
                let parent = tracing::Span::current();
                super::spawn_dependency_flush_wait(
                    &mut tasks,
                    0,
                    module,
                    kick_inbox,
                    Vec::new(),
                    batch,
                    vec![completion],
                    &parent,
                    &subscriber,
                );

                let message = tasks
                    .next()
                    .await
                    .expect("dependency flush task should finish")
                    .expect("dependency flush task should not panic");
                match message {
                    super::TaskMessage::DependencyFlush { pending_kicks, .. } => {
                        assert!(pending_kicks.is_empty());
                    }
                    _ => panic!("expected dependency flush task message"),
                }
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
                        test_policy(1..=1, Bpm::from_f64(60.0)..=Bpm::from_f64(60.0)),
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
                    runtime.core_policies().await,
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
                let dependency_targets = super::DependencyTargets {
                    dependencies: Arc::new(ModuleDependencies::default()),
                    target_indexes_by_role: Arc::new(HashMap::new()),
                };
                let mut followup_tasks = futures::stream::FuturesUnordered::new();
                let mut zero_window_wakers = vec![None];
                let mut zero_windows = super::ZeroReplicaWindows::new(HashMap::new());
                let err = super::handle_task_message(
                    message,
                    &runtime,
                    std::slice::from_ref(&owner),
                    &mut states,
                    &mut followup_tasks,
                    &mut kick_inboxes,
                    &mut zero_window_wakers,
                    &[],
                    &dependency_targets,
                    &HashMap::new(),
                    &mut zero_windows,
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
                        test_policy(1..=1, Bpm::from_f64(60.0)..=Bpm::from_f64(60.0)),
                        |_| HangingBatchStub,
                    )
                    .unwrap()
                    .build(&caps)
                    .await
                    .unwrap();
                let (_, mut modules, _) = modules.into_parts_with_dependencies();
                let module = modules.pop().unwrap();
                let owner = module.owner().clone();
                let (kick_inbox, kick_handle) = crate::kicks::KickInbox::new();
                let mut tasks = futures::stream::FuturesUnordered::new();
                let subscriber = tracing::dispatcher::get_default(Clone::clone);
                let parent = tracing::Span::current();

                super::spawn_next_batch(&mut tasks, 0, module, kick_inbox, &parent, &subscriber);
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
                        test_policy(1..=1, Bpm::from_f64(60.0)..=Bpm::from_f64(60.0)),
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
