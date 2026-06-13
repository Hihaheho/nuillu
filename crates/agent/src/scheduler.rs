use std::collections::{HashMap, VecDeque};
use std::future::Future;
use std::pin::pin;
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use chrono::{DateTime, Duration as ChronoDuration, Utc};
use futures::stream::{FuturesUnordered, StreamExt};
use nuillu_blackboard::{
    ActivationRatio, CorePolicyRecord, IdentityMemoryRecord, ZeroReplicaWindowPolicy,
};
use nuillu_module::{
    ActivateCx, ActivationGateVote, AgentRuntimeControl, AllocatedModule, AllocatedModules,
    LlmBatchDebug, LlmRequestMetadata, LlmRequestSource, ModuleBatch, ModuleDependencies,
    ModuleRunStatus, SelfWakePermitClaim, SessionCompactionRuntime, WakeClaim, ports::Clock,
    with_activation_llm_request_metadata,
};
use nuillu_types::{ModelTier, ModuleId, ModuleInstanceId, ReplicaIndex, builtin};
use thiserror::Error;
use tokio::sync::{oneshot, watch};
use tokio::task::{JoinHandle, spawn_local};
use tokio::time::{Instant, sleep, sleep_until};
use tracing::{Instrument as _, instrument::WithSubscriber as _};

use crate::kicks::{Kick, KickHandle, KickInbox};

const DEPENDENCY_SETTLE_MAX_WAVES: u8 = 2;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AgentEventLoopConfig {
    pub idle_threshold: Duration,
    pub max_activation_attempts: u8,
    pub dependency_idle_timeout: Duration,
    pub dependency_hard_timeout: Duration,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModuleSessionReset {
    pub owner: ModuleInstanceId,
    pub deleted_sessions: u64,
}

#[derive(Debug, Error)]
pub enum ModuleSessionResetError {
    #[error("module {owner} is not registered in this runtime")]
    UnknownOwner { owner: ModuleInstanceId },
    #[error("agent runtime is not accepting reset requests")]
    RuntimeUnavailable,
    #[error("failed to reset module {owner} session history: {message}")]
    Failed {
        owner: ModuleInstanceId,
        message: String,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AgentRunMode {
    Running,
    Paused,
}

#[derive(Clone)]
pub struct AgentRunController {
    mode: watch::Sender<AgentRunMode>,
    running: Arc<AtomicBool>,
    requests: Arc<Mutex<VecDeque<AgentControlRequest>>>,
}

impl AgentRunController {
    pub fn new() -> (Self, AgentRunControl) {
        let (sender, receiver) = watch::channel(AgentRunMode::Running);
        let running = Arc::new(AtomicBool::new(true));
        let requests = Arc::new(Mutex::new(VecDeque::new()));
        (
            Self {
                mode: sender,
                running: Arc::clone(&running),
                requests: Arc::clone(&requests),
            },
            AgentRunControl {
                mode: Some(receiver),
                running: Some(running),
                requests,
            },
        )
    }

    pub fn pause(&self) {
        self.running.store(false, Ordering::SeqCst);
        let _ = self.mode.send(AgentRunMode::Paused);
    }

    pub fn resume(&self) {
        self.running.store(true, Ordering::SeqCst);
        let _ = self.mode.send(AgentRunMode::Running);
    }

    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::SeqCst)
    }

    pub async fn reset_module_session_history(
        &self,
        owner: ModuleInstanceId,
    ) -> Result<ModuleSessionReset, ModuleSessionResetError> {
        let (response, receiver) = oneshot::channel();
        {
            let Ok(mut requests) = self.requests.lock() else {
                return Err(ModuleSessionResetError::RuntimeUnavailable);
            };
            requests.push_back(AgentControlRequest::ResetModuleSessionHistory { owner, response });
        }
        let mode = if self.is_running() {
            AgentRunMode::Running
        } else {
            AgentRunMode::Paused
        };
        let _ = self.mode.send(mode);
        receiver
            .await
            .unwrap_or(Err(ModuleSessionResetError::RuntimeUnavailable))
    }
}

#[derive(Clone)]
pub struct AgentRunControl {
    mode: Option<watch::Receiver<AgentRunMode>>,
    running: Option<Arc<AtomicBool>>,
    requests: Arc<Mutex<VecDeque<AgentControlRequest>>>,
}

impl AgentRunControl {
    pub fn always_running() -> Self {
        Self {
            mode: None,
            running: None,
            requests: Arc::new(Mutex::new(VecDeque::new())),
        }
    }

    fn is_running(&self) -> bool {
        self.running
            .as_ref()
            .is_none_or(|running| running.load(Ordering::SeqCst))
    }

    async fn changed(&mut self) {
        let Some(mode) = &mut self.mode else {
            std::future::pending::<()>().await;
            return;
        };
        if mode.changed().await.is_err() {
            std::future::pending::<()>().await;
        }
    }

    fn drain_requests(&self) -> Vec<AgentControlRequest> {
        let Ok(mut requests) = self.requests.lock() else {
            return Vec::new();
        };
        requests.drain(..).collect()
    }
}

enum AgentControlRequest {
    ResetModuleSessionHistory {
        owner: ModuleInstanceId,
        response: ModuleSessionResetResponder,
    },
}

type ModuleSessionResetResponder =
    oneshot::Sender<Result<ModuleSessionReset, ModuleSessionResetError>>;

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
    run_controlled(modules, config, AgentRunControl::always_running(), shutdown).await
}

/// Run the agent event loop with an external Run/Stop control.
///
/// Pausing prevents new module activations from being scheduled. Already
/// running module tasks continue to completion, so in-flight LLM calls are not
/// aborted by a pause.
pub async fn run_controlled(
    modules: AllocatedModules,
    config: AgentEventLoopConfig,
    control: AgentRunControl,
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
    let mut pending_session_resets = std::iter::repeat_with(VecDeque::new)
        .take(states.len())
        .collect::<Vec<_>>();
    let mut consecutive_failures = vec![0_u32; states.len()];
    let mut active = vec![false; states.len()];
    let mut zero_windows = ZeroReplicaWindows::new(runtime.zero_replica_window_policies().await);
    let mut zero_window_wakers = std::iter::repeat_with(|| None)
        .take(owners.len())
        .collect::<Vec<_>>();
    let mut tasks: FuturesUnordered<JoinHandle<TaskMessage>> = FuturesUnordered::new();
    let subscriber = tracing::dispatcher::get_default(Clone::clone);
    let parent = tracing::Span::current();

    let mut control = control;
    if control.is_running() {
        refresh_active_and_schedule(
            &runtime,
            &owners,
            &mut active,
            &mut states,
            &mut tasks,
            &mut kick_inboxes,
            &mut zero_window_wakers,
            &kick_handles,
            &dependency_targets,
            &mut zero_windows,
            config,
            &parent,
            &subscriber,
        )
        .await;
    }

    let mut shutdown = pin!(shutdown);
    let mut idle_since: Option<Instant> = None;
    let mut idle_marker_sent = false;
    let mut observed_wake_sequence = runtime.wake_change_sequence();

    loop {
        let running = control.is_running();
        if !running {
            idle_since = None;
            idle_marker_sent = false;
        }
        let idle_now = running && is_idle(&active, &states);
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
                fail_pending_session_resets(&mut pending_session_resets);
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
                            &mut consecutive_failures,
                            control.is_running(),
                            config,
                            &parent,
                            &subscriber,
                        ).await {
                            abort_tasks(&mut tasks).await;
                            fail_pending_session_resets(&mut pending_session_resets);
                            return Err(error);
                        }
                        apply_pending_session_resets(
                            &runtime,
                            &owners,
                            &mut states,
                            &mut kick_inboxes,
                            &mut consecutive_failures,
                            &mut pending_session_resets,
                        )
                        .await;
                        if control.is_running() {
                            refresh_active_and_schedule(
                                &runtime,
                                &owners,
                                &mut active,
                                &mut states,
                                &mut tasks,
                                &mut kick_inboxes,
                                &mut zero_window_wakers,
                                &kick_handles,
                                &dependency_targets,
                                &mut zero_windows,
                                config,
                                &parent,
                                &subscriber,
                            )
                            .await;
                        }
                    }
                    Some(Err(e)) => {
                        let message = e.to_string();
                        tracing::error!(error = ?e, "module task panicked");
                        abort_tasks(&mut tasks).await;
                        fail_pending_session_resets(&mut pending_session_resets);
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
            },
            _ = control.changed() => {
                handle_control_requests(
                    &control,
                    &owners,
                    &mut pending_session_resets,
                    &mut zero_window_wakers,
                );
                apply_pending_session_resets(
                    &runtime,
                    &owners,
                    &mut states,
                    &mut kick_inboxes,
                    &mut consecutive_failures,
                    &mut pending_session_resets,
                )
                .await;
                if control.is_running() {
                    refresh_active_and_schedule(
                        &runtime,
                        &owners,
                        &mut active,
                        &mut states,
                        &mut tasks,
                        &mut kick_inboxes,
                        &mut zero_window_wakers,
                        &kick_handles,
                        &dependency_targets,
                        &mut zero_windows,
                        config,
                        &parent,
                        &subscriber,
                    )
                    .await;
                }
            },
            _ = runtime.wake_changed_since(observed_wake_sequence), if running => {
                observed_wake_sequence = runtime.wake_change_sequence();
                refresh_active_and_schedule(
                    &runtime,
                    &owners,
                    &mut active,
                    &mut states,
                    &mut tasks,
                    &mut kick_inboxes,
                    &mut zero_window_wakers,
                    &kick_handles,
                    &dependency_targets,
                    &mut zero_windows,
                    config,
                    &parent,
                    &subscriber,
                )
                .await;
            },
        }
    }
}

async fn abort_tasks(tasks: &mut FuturesUnordered<JoinHandle<TaskMessage>>) {
    for handle in tasks.iter() {
        handle.abort();
    }
    while tasks.next().await.is_some() {}
}

fn handle_control_requests(
    control: &AgentRunControl,
    owners: &[ModuleInstanceId],
    pending_session_resets: &mut [VecDeque<ModuleSessionResetResponder>],
    zero_window_wakers: &mut [Option<oneshot::Sender<()>>],
) {
    for request in control.drain_requests() {
        match request {
            AgentControlRequest::ResetModuleSessionHistory { owner, response } => {
                let Some(index) = owners.iter().position(|candidate| candidate == &owner) else {
                    let _ = response.send(Err(ModuleSessionResetError::UnknownOwner { owner }));
                    continue;
                };
                pending_session_resets[index].push_back(response);
                if let Some(waker) = zero_window_wakers[index].take() {
                    let _ = waker.send(());
                }
            }
        }
    }
}

async fn apply_pending_session_resets(
    runtime: &AgentRuntimeControl,
    owners: &[ModuleInstanceId],
    states: &mut [ModuleState],
    kick_inboxes: &mut [Option<KickInbox>],
    consecutive_failures: &mut [u32],
    pending_session_resets: &mut [VecDeque<ModuleSessionResetResponder>],
) {
    for index in 0..pending_session_resets.len() {
        while let Some(response) = pending_session_resets[index].pop_front() {
            let Some(result) = try_reset_module_session(
                runtime,
                owners,
                states,
                kick_inboxes,
                consecutive_failures,
                index,
            )
            .await
            else {
                pending_session_resets[index].push_front(response);
                break;
            };
            let _ = response.send(result);
        }
    }
}

async fn try_reset_module_session(
    runtime: &AgentRuntimeControl,
    owners: &[ModuleInstanceId],
    states: &mut [ModuleState],
    kick_inboxes: &mut [Option<KickInbox>],
    consecutive_failures: &mut [u32],
    index: usize,
) -> Option<Result<ModuleSessionReset, ModuleSessionResetError>> {
    let owner = owners[index].clone();
    let state = std::mem::replace(&mut states[index], ModuleState::Awaiting);
    match state {
        ModuleState::Stored { mut module, .. } => {
            let result = reset_allocated_module(runtime, &owner, &mut module).await;
            states[index] = ModuleState::Stored {
                module,
                next_batch_throttle: None,
            };
            consecutive_failures[index] = 0;
            Some(result)
        }
        ModuleState::PendingBatch {
            mut module,
            pending_kicks,
            ..
        } => {
            if let Some(kick_inbox) = &mut kick_inboxes[index] {
                notify_pending_and_ready(pending_kicks, kick_inbox);
            }
            let result = reset_allocated_module(runtime, &owner, &mut module).await;
            states[index] = ModuleState::Stored {
                module,
                next_batch_throttle: None,
            };
            consecutive_failures[index] = 0;
            Some(result)
        }
        ModuleState::FailedUntilActivation { mut module, .. } => {
            let result = reset_allocated_module(runtime, &owner, &mut module).await;
            states[index] = ModuleState::Stored {
                module,
                next_batch_throttle: None,
            };
            consecutive_failures[index] = 0;
            Some(result)
        }
        other => {
            states[index] = other;
            None
        }
    }
}

async fn reset_allocated_module(
    runtime: &AgentRuntimeControl,
    owner: &ModuleInstanceId,
    module: &mut AllocatedModule,
) -> Result<ModuleSessionReset, ModuleSessionResetError> {
    let deleted_sessions = runtime
        .delete_module_sessions(owner)
        .await
        .map_err(|error| ModuleSessionResetError::Failed {
            owner: owner.clone(),
            message: error.to_string(),
        })?;
    module
        .restart()
        .await
        .map_err(|error| ModuleSessionResetError::Failed {
            owner: owner.clone(),
            message: error.to_string(),
        })?;
    Ok(ModuleSessionReset {
        owner: owner.clone(),
        deleted_sessions,
    })
}

fn fail_pending_session_resets(
    pending_session_resets: &mut [VecDeque<ModuleSessionResetResponder>],
) {
    for pending in pending_session_resets {
        for response in pending.drain(..) {
            let _ = response.send(Err(ModuleSessionResetError::RuntimeUnavailable));
        }
    }
}

enum ModuleState {
    Stored {
        module: AllocatedModule,
        next_batch_throttle: Option<NextBatchThrottle>,
    },
    WaitingForActivation,
    Throttling,
    Awaiting,
    PendingBatch {
        module: AllocatedModule,
        batch: ModuleBatch,
        gate_approved: bool,
        self_wake_activation_permit: bool,
        pending_kicks: Vec<Kick>,
    },
    PendingDependencySettle,
    PendingActivationGate,
    Activating,
    FailedUntilActivation {
        module: AllocatedModule,
        phase: String,
        message: String,
        consecutive_failures: u32,
    },
    WaitingAfterFailure {
        phase: String,
        message: String,
        consecutive_failures: u32,
    },
}

enum TaskMessage {
    BatchThrottleExpired {
        index: usize,
        module: AllocatedModule,
        kick_inbox: KickInbox,
        delayed_for: Duration,
        next_batch_throttle: Option<NextBatchThrottle>,
    },
    NextBatch {
        index: usize,
        module: AllocatedModule,
        kick_inbox: KickInbox,
        wake_claim: Option<WakeClaim>,
        self_wake_permit_claim: Option<SelfWakePermitClaim>,
        result: Result<(ModuleBatch, Vec<Kick>), String>,
    },
    DependencySettled {
        index: usize,
        module: AllocatedModule,
        kick_inbox: KickInbox,
        pending_kicks: Vec<Kick>,
        wake_claim: Option<WakeClaim>,
        remaining_waves: u8,
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
        self_wake_activation_permit: bool,
        outcome: ActivationGateOutcome,
    },
    ActivationWaitReady {
        index: usize,
        module: AllocatedModule,
        kick_inbox: KickInbox,
        next_batch_throttle: Option<NextBatchThrottle>,
    },
    FailureWaitReady {
        index: usize,
        module: AllocatedModule,
        kick_inbox: KickInbox,
    },
}

#[derive(Debug, Clone)]
struct NextBatchThrottle {
    baseline: DateTime<Utc>,
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

    fn sync(&mut self, policies: HashMap<ModuleId, ZeroReplicaWindowPolicy>) {
        self.states
            .retain(|module, _| policies.contains_key(module));
        for (module, policy) in policies {
            if let Some(period) = policy.controller_activation_period() {
                if let Some(state) = self.states.get_mut(&module) {
                    state.controller_activation_period = period;
                } else if let Some(state) = ZeroReplicaWindowState::new(policy) {
                    self.states.insert(module, state);
                }
            }
        }
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
        self.sync(runtime.zero_replica_window_policies().await);
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

    fn has_dependencies(&self, dependent: &ModuleId) -> bool {
        !self.dependencies.deps_of(dependent).is_empty()
    }
}

struct DependencyWait {
    owner: ModuleInstanceId,
    completion: DependencyWaitCompletion,
}

enum DependencyWaitCompletion {
    Kick(oneshot::Receiver<()>),
    Inactive {
        runtime: AgentRuntimeControl,
        kick_handle: KickHandle,
        sender: ModuleInstanceId,
        activation_waiter: Option<oneshot::Receiver<()>>,
        idle_timeout: Duration,
    },
}

impl DependencyWait {
    fn kick(owner: ModuleInstanceId, completion: oneshot::Receiver<()>) -> Self {
        Self {
            owner,
            completion: DependencyWaitCompletion::Kick(completion),
        }
    }

    fn inactive(
        owner: ModuleInstanceId,
        runtime: AgentRuntimeControl,
        kick_handle: KickHandle,
        sender: ModuleInstanceId,
        activation_waiter: Option<oneshot::Receiver<()>>,
        idle_timeout: Duration,
    ) -> Self {
        Self {
            owner,
            completion: DependencyWaitCompletion::Inactive {
                runtime,
                kick_handle,
                sender,
                activation_waiter,
                idle_timeout,
            },
        }
    }

    async fn wait(self) -> ModuleInstanceId {
        let owner = self.owner;
        match self.completion {
            DependencyWaitCompletion::Kick(completion) => {
                let _ = completion.await;
            }
            DependencyWaitCompletion::Inactive {
                runtime,
                kick_handle,
                sender,
                activation_waiter,
                idle_timeout,
            } => {
                wait_for_inactive_dependency(
                    owner.clone(),
                    runtime,
                    kick_handle,
                    sender,
                    activation_waiter,
                    idle_timeout,
                )
                .await;
            }
        }
        owner
    }
}

async fn wait_for_inactive_dependency(
    owner: ModuleInstanceId,
    runtime: AgentRuntimeControl,
    kick_handle: KickHandle,
    sender: ModuleInstanceId,
    activation_waiter: Option<oneshot::Receiver<()>>,
    idle_timeout: Duration,
) {
    let mut activation = pin!(async move {
        if let Some(waiter) = activation_waiter {
            let _ = waiter.await;
        }
    });
    let mut idle = pin!(sleep(idle_timeout));

    tokio::select! {
        biased;
        _ = &mut activation => {
            let completion = kick_handle.send(sender);
            let _ = completion.await;
        }
        _ = &mut idle => {
            if runtime.is_active(&owner).await {
                let completion = kick_handle.send(sender.clone());
                let _ = completion.await;
            }
        }
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
    kick_handles: &[KickHandle],
    dependency_targets: &DependencyTargets,
    zero_windows: &mut ZeroReplicaWindows,
    config: AgentEventLoopConfig,
    parent: &tracing::Span,
    subscriber: &tracing::Dispatch,
) {
    zero_windows.sync(runtime.zero_replica_window_policies().await);
    let mut allocation_active = vec![false; owners.len()];
    for (index, owner) in owners.iter().enumerate() {
        allocation_active[index] = runtime.is_active(owner).await;
    }
    zero_windows.reset_allocation_active(owners, &allocation_active);
    for (index, owner) in owners.iter().enumerate() {
        let self_wake_permit_active = runtime.has_pending_self_wake_permit(owner)
            && !runtime.is_forced_disabled(&owner.module).await;
        active[index] =
            allocation_active[index] || zero_windows.allows(owner) || self_wake_permit_active;
    }

    for index in 0..states.len() {
        let state_self_wake_activation_active =
            matches!(
                &states[index],
                ModuleState::PendingBatch {
                    self_wake_activation_permit: true,
                    ..
                }
            ) && !runtime.is_forced_disabled(&owners[index].module).await;
        let state_active = active[index] || state_self_wake_activation_active;
        match &states[index] {
            ModuleState::Stored { .. } if state_active => {
                runtime
                    .record_module_status(owners[index].clone(), ModuleRunStatus::AwaitingBatch)
                    .await;
                let pending_wake = runtime.has_pending_wake(&owners[index]);
                let has_pending_self_wake_permit =
                    runtime.has_pending_self_wake_permit(&owners[index]);
                let has_dependencies = dependency_targets.has_dependencies(&owners[index].module);
                if has_dependencies && !pending_wake && !has_pending_self_wake_permit {
                    continue;
                }
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
                let wake_claim = pending_wake
                    .then(|| runtime.claim_wake(&owners[index]))
                    .flatten();
                if let Some(throttle) =
                    current_next_batch_throttle(runtime, &owners[index], next_batch_throttle, now)
                        .await
                {
                    if let Some(activation_increase) = runtime
                        .activation_increase_waiter(&owners[index], throttle.activation_threshold)
                        .await
                    {
                        states[index] = ModuleState::Throttling;
                        let allocation_change = runtime.allocation_change_waiter().await;
                        spawn_batch_throttle(
                            tasks,
                            index,
                            module,
                            kick_inbox,
                            throttle,
                            runtime.clock(),
                            activation_increase,
                            allocation_change,
                            parent,
                            subscriber,
                        );
                    } else {
                        let scheduled = schedule_dependency_settle_or_next_batch(
                            runtime,
                            tasks,
                            index,
                            owners[index].clone(),
                            module,
                            kick_inbox,
                            Vec::new(),
                            wake_claim,
                            DEPENDENCY_SETTLE_MAX_WAVES,
                            states,
                            kick_inboxes,
                            kick_handles,
                            dependency_targets,
                            owners,
                            zero_windows,
                            config,
                            parent,
                            subscriber,
                        )
                        .await;
                        states[index] = scheduled.module_state();
                    }
                } else {
                    let scheduled = schedule_dependency_settle_or_next_batch(
                        runtime,
                        tasks,
                        index,
                        owners[index].clone(),
                        module,
                        kick_inbox,
                        Vec::new(),
                        wake_claim,
                        DEPENDENCY_SETTLE_MAX_WAVES,
                        states,
                        kick_inboxes,
                        kick_handles,
                        dependency_targets,
                        owners,
                        zero_windows,
                        config,
                        parent,
                        subscriber,
                    )
                    .await;
                    states[index] = scheduled.module_state();
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
            ModuleState::Throttling | ModuleState::Awaiting => {
                let status = if state_active {
                    ModuleRunStatus::AwaitingBatch
                } else {
                    ModuleRunStatus::Inactive
                };
                runtime
                    .record_module_status(owners[index].clone(), status)
                    .await;
            }
            ModuleState::PendingBatch { .. } if state_active => {
                zero_windows.mark_batch_accepted(&owners[index]);
                let ModuleState::PendingBatch {
                    module,
                    batch,
                    gate_approved,
                    self_wake_activation_permit,
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
                    runtime.record_module_batch_ready(owners[index].clone(), &batch);
                    let peer_contexts = runtime.peer_contexts();
                    let allocation_hints = runtime.allocation_hints();
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
                        peer_contexts,
                        allocation_hints,
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
                        self_wake_activation_permit,
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
            ModuleState::PendingDependencySettle => {
                let status = if state_active {
                    ModuleRunStatus::PendingBatch
                } else {
                    ModuleRunStatus::Inactive
                };
                runtime
                    .record_module_status(owners[index].clone(), status)
                    .await;
            }
            ModuleState::PendingActivationGate => {
                let status = if state_active {
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
            ModuleState::FailedUntilActivation { .. } => {
                let ModuleState::FailedUntilActivation {
                    module,
                    phase,
                    message,
                    consecutive_failures,
                } = std::mem::replace(&mut states[index], ModuleState::Awaiting)
                else {
                    unreachable!("module state changed while scheduling failure wait");
                };
                runtime
                    .record_module_status(
                        owners[index].clone(),
                        ModuleRunStatus::Failed {
                            phase: phase.clone(),
                            message: message.clone(),
                        },
                    )
                    .await;
                states[index] = ModuleState::WaitingAfterFailure {
                    phase: phase.clone(),
                    message: message.clone(),
                    consecutive_failures,
                };
                let kick_inbox = kick_inboxes[index]
                    .take()
                    .expect("kick inbox available for failed module");
                let activation_waiter = runtime.activation_waiter(&owners[index]).await;
                let allocation_change_waiter = runtime.allocation_change_waiter().await;
                let retry_after = failed_retry_after(runtime, &owners[index]).await;
                let (zero_window_waker, zero_window_waiter) = oneshot::channel();
                zero_window_wakers[index] = Some(zero_window_waker);
                spawn_failure_wait(
                    tasks,
                    index,
                    module,
                    kick_inbox,
                    activation_waiter,
                    allocation_change_waiter,
                    zero_window_waiter,
                    retry_after,
                    runtime.clock(),
                    parent,
                    subscriber,
                );
            }
            ModuleState::WaitingAfterFailure {
                phase,
                message,
                consecutive_failures,
            } => {
                runtime
                    .record_module_status(
                        owners[index].clone(),
                        ModuleRunStatus::Failed {
                            phase: phase.clone(),
                            message: message.clone(),
                        },
                    )
                    .await;
                let _ = consecutive_failures;
            }
        }
    }
}

async fn failed_retry_after(
    runtime: &AgentRuntimeControl,
    owner: &ModuleInstanceId,
) -> Option<DateTime<Utc>> {
    let (bpm, _) = runtime.module_batch_throttle_baseline(owner).await?;
    chrono::Duration::from_std(bpm.period())
        .ok()
        .map(|duration| runtime.clock().now() + duration)
}

async fn current_next_batch_throttle(
    runtime: &AgentRuntimeControl,
    owner: &ModuleInstanceId,
    throttle: Option<NextBatchThrottle>,
    now: DateTime<Utc>,
) -> Option<NextBatchThrottle> {
    let throttle = throttle?;
    let (bpm, activation_threshold) = runtime.module_batch_throttle_baseline(owner).await?;
    let not_before = throttle.baseline + ChronoDuration::from_std(bpm.period()).ok()?;
    (not_before > now).then_some(NextBatchThrottle {
        baseline: throttle.baseline,
        not_before,
        activation_threshold,
    })
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
    consecutive_failures: &mut [u32],
    scheduling_enabled: bool,
    config: AgentEventLoopConfig,
    parent: &tracing::Span,
    subscriber: &tracing::Dispatch,
) -> Result<(), SchedulerError> {
    match message {
        TaskMessage::BatchThrottleExpired {
            index,
            module,
            kick_inbox,
            delayed_for,
            next_batch_throttle,
        } => {
            runtime.record_module_batch_throttled(owners[index].clone(), delayed_for);
            kick_inboxes[index] = Some(kick_inbox);
            states[index] = ModuleState::Stored {
                module,
                next_batch_throttle,
            };
            Ok(())
        }
        TaskMessage::NextBatch {
            index,
            module,
            mut kick_inbox,
            wake_claim,
            self_wake_permit_claim,
            result,
        } => match result {
            Ok((batch, pending_kicks)) => {
                let has_self_wake_permit_claim = self_wake_permit_claim.is_some();
                if let Some(claim) = wake_claim {
                    runtime.complete_wake_claim(claim);
                }
                if let Some(claim) = self_wake_permit_claim {
                    runtime.complete_self_wake_permit_claim(claim);
                }
                if scheduling_enabled
                    && scheduling_active(
                        runtime,
                        zero_windows,
                        &owners[index],
                        has_self_wake_permit_claim,
                    )
                    .await
                {
                    zero_windows.mark_batch_accepted(&owners[index]);
                    let scheduled = spawn_activation_gate_or_activate(
                        runtime,
                        tasks,
                        index,
                        owners[index].clone(),
                        module,
                        kick_inbox,
                        pending_kicks,
                        batch,
                        has_self_wake_permit_claim,
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
                        self_wake_activation_permit: false,
                        pending_kicks: Vec::new(),
                    };
                }
                Ok(())
            }
            Err(message) => {
                if let Some(claim) = wake_claim {
                    runtime.complete_wake_claim(claim);
                }
                if let Some(claim) = self_wake_permit_claim {
                    runtime.complete_self_wake_permit_claim(claim);
                }
                notify_pending_and_ready(Vec::new(), &mut kick_inbox);
                kick_inboxes[index] = Some(kick_inbox);
                park_failed_module(
                    runtime,
                    &owners[index],
                    states,
                    consecutive_failures,
                    index,
                    module,
                    "next_batch",
                    message,
                )
                .await;
                Ok(())
            }
        },
        TaskMessage::DependencySettled {
            index,
            module,
            mut kick_inbox,
            mut pending_kicks,
            wake_claim,
            remaining_waves,
        } => {
            pending_kicks.extend(kick_inbox.drain_ready());
            if scheduling_enabled
                && scheduling_active(runtime, zero_windows, &owners[index], false).await
            {
                let scheduled = schedule_dependency_settle_or_next_batch(
                    runtime,
                    tasks,
                    index,
                    owners[index].clone(),
                    module,
                    kick_inbox,
                    pending_kicks,
                    wake_claim,
                    remaining_waves,
                    states,
                    kick_inboxes,
                    kick_handles,
                    dependency_targets,
                    owners,
                    zero_windows,
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
                states[index] = ModuleState::Stored {
                    module,
                    next_batch_throttle: None,
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
        } => {
            runtime.record_module_activation_completed(
                owners[index].clone(),
                activation_elapsed,
                result.is_ok(),
            );
            match result {
                Ok(()) => {
                    consecutive_failures[index] = 0;
                    notify_pending_and_ready(pending_kicks, &mut kick_inbox);
                    kick_inboxes[index] = Some(kick_inbox);
                    let now = runtime.clock().now();
                    let next_batch_throttle = runtime
                        .module_batch_throttle_baseline(&owners[index])
                        .await
                        .and_then(|(bpm, activation_threshold)| {
                            let remaining = bpm.sleep_after_turn(activation_elapsed);
                            if remaining.is_zero() {
                                None
                            } else {
                                chrono::Duration::from_std(remaining).ok().map(|d| {
                                    NextBatchThrottle {
                                        baseline: now
                                            - ChronoDuration::from_std(activation_elapsed)
                                                .unwrap_or_default(),
                                        not_before: now + d,
                                        activation_threshold,
                                    }
                                })
                            }
                        });
                    states[index] = ModuleState::Stored {
                        module,
                        next_batch_throttle,
                    };
                    zero_windows.finish(&owners[index]);
                    if scheduling_enabled && owners[index].module == builtin::allocation() {
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
                    zero_windows.finish(&owners[index]);
                    park_failed_module(
                        runtime,
                        &owners[index],
                        states,
                        consecutive_failures,
                        index,
                        module,
                        "activate",
                        message,
                    )
                    .await;
                    Ok(())
                }
            }
        }
        TaskMessage::ActivationGate {
            index,
            module,
            mut kick_inbox,
            batch,
            pending_kicks,
            self_wake_activation_permit,
            outcome,
        } => {
            if outcome.allowed() {
                if scheduling_enabled
                    && scheduling_active(
                        runtime,
                        zero_windows,
                        &owners[index],
                        self_wake_activation_permit,
                    )
                    .await
                {
                    runtime
                        .record_module_status(owners[index].clone(), ModuleRunStatus::Activating)
                        .await;
                    runtime.record_module_batch_ready(owners[index].clone(), &batch);
                    states[index] = ModuleState::Activating;
                    let peer_contexts = runtime.peer_contexts();
                    let allocation_hints = runtime.allocation_hints();
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
                        peer_contexts,
                        allocation_hints,
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
                        self_wake_activation_permit: false,
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
        TaskMessage::FailureWaitReady {
            index,
            module,
            kick_inbox,
        } => {
            zero_window_wakers[index] = None;
            kick_inboxes[index] = Some(kick_inbox);
            states[index] = ModuleState::Stored {
                module,
                next_batch_throttle: None,
            };
            Ok(())
        }
    }
}

#[allow(clippy::too_many_arguments)]
async fn park_failed_module(
    runtime: &AgentRuntimeControl,
    owner: &ModuleInstanceId,
    states: &mut [ModuleState],
    consecutive_failures: &mut [u32],
    index: usize,
    module: AllocatedModule,
    phase: &'static str,
    message: String,
) {
    let failures = record_module_failure(
        runtime,
        owner,
        consecutive_failures,
        index,
        phase,
        message.clone(),
    )
    .await;
    states[index] = ModuleState::FailedUntilActivation {
        module,
        phase: phase.to_string(),
        message,
        consecutive_failures: failures,
    };
}

async fn record_module_failure(
    runtime: &AgentRuntimeControl,
    owner: &ModuleInstanceId,
    consecutive_failures: &mut [u32],
    index: usize,
    phase: &'static str,
    message: String,
) -> u32 {
    let failures = consecutive_failures[index].saturating_add(1);
    consecutive_failures[index] = failures;
    runtime
        .record_module_status(
            owner.clone(),
            ModuleRunStatus::Failed {
                phase: phase.to_string(),
                message: message.clone(),
            },
        )
        .await;
    runtime.record_module_task_failed(owner.clone(), phase, message);
    failures
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
    has_self_wake_permit_claim: bool,
) -> bool {
    if runtime.is_forced_disabled(&owner.module).await {
        return false;
    }
    if has_self_wake_permit_claim || runtime.has_pending_self_wake_permit(owner) {
        return true;
    }
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
async fn schedule_dependency_settle_or_next_batch(
    runtime: &AgentRuntimeControl,
    tasks: &mut FuturesUnordered<JoinHandle<TaskMessage>>,
    index: usize,
    owner: ModuleInstanceId,
    module: AllocatedModule,
    kick_inbox: KickInbox,
    pending_kicks: Vec<Kick>,
    wake_claim: Option<WakeClaim>,
    remaining_waves: u8,
    states: &mut [ModuleState],
    kick_inboxes: &mut [Option<KickInbox>],
    kick_handles: &[KickHandle],
    dependency_targets: &DependencyTargets,
    owners: &[ModuleInstanceId],
    zero_windows: &mut ZeroReplicaWindows,
    config: AgentEventLoopConfig,
    parent: &tracing::Span,
    subscriber: &tracing::Dispatch,
) -> PreBatchScheduling {
    let completions = if remaining_waves == 0 {
        Vec::new()
    } else {
        collect_dependency_settle_completions(
            runtime,
            tasks,
            index,
            &owner,
            states,
            kick_inboxes,
            kick_handles,
            dependency_targets,
            owners,
            zero_windows,
            config,
            parent,
            subscriber,
        )
        .await
    };
    if completions.is_empty() {
        spawn_next_batch(
            runtime.clone(),
            tasks,
            index,
            owner,
            module,
            kick_inbox,
            pending_kicks,
            wake_claim,
            config.dependency_idle_timeout,
            parent,
            subscriber,
        );
        PreBatchScheduling::Awaiting
    } else {
        runtime
            .record_module_status(owner.clone(), ModuleRunStatus::PendingBatch)
            .await;
        spawn_dependency_settle_wait(
            tasks,
            index,
            owner,
            module,
            kick_inbox,
            pending_kicks,
            wake_claim,
            completions,
            remaining_waves.saturating_sub(1),
            config.dependency_hard_timeout,
            parent,
            subscriber,
        );
        PreBatchScheduling::PendingDependencySettle
    }
}

#[allow(clippy::too_many_arguments)]
async fn collect_dependency_settle_completions(
    runtime: &AgentRuntimeControl,
    tasks: &mut FuturesUnordered<JoinHandle<TaskMessage>>,
    dependent_index: usize,
    sender: &ModuleInstanceId,
    states: &mut [ModuleState],
    kick_inboxes: &mut [Option<KickInbox>],
    kick_handles: &[KickHandle],
    dependency_targets: &DependencyTargets,
    owners: &[ModuleInstanceId],
    zero_windows: &mut ZeroReplicaWindows,
    config: AgentEventLoopConfig,
    parent: &tracing::Span,
    subscriber: &tracing::Dispatch,
) -> Vec<DependencyWait> {
    let mut completions = Vec::new();
    for target_index in dependency_targets.target_indexes(&sender.module) {
        if target_index == dependent_index {
            continue;
        }
        let target_owner = owners[target_index].clone();
        let target_has_wake = runtime.has_pending_wake(&target_owner);
        if !scheduling_active(runtime, zero_windows, &target_owner, false).await {
            if target_has_wake {
                let activation_waiter = runtime.activation_waiter(&target_owner).await;
                completions.push(DependencyWait::inactive(
                    target_owner,
                    runtime.clone(),
                    kick_handles[target_index].clone(),
                    sender.clone(),
                    activation_waiter,
                    config.dependency_idle_timeout,
                ));
            }
            continue;
        }

        match &mut states[target_index] {
            ModuleState::Stored { .. } if target_has_wake => {
                let completion = schedule_stored_dependency_for_wake(
                    runtime,
                    tasks,
                    target_index,
                    target_owner,
                    sender.clone(),
                    states,
                    kick_inboxes,
                    kick_handles,
                    config,
                    parent,
                    subscriber,
                )
                .await;
                completions.push(DependencyWait::kick(
                    owners[target_index].clone(),
                    completion,
                ));
            }
            ModuleState::Throttling
            | ModuleState::PendingDependencySettle
            | ModuleState::PendingActivationGate
            | ModuleState::Activating => completions.push(DependencyWait::kick(
                target_owner,
                kick_handles[target_index].send(sender.clone()),
            )),
            ModuleState::Awaiting if target_has_wake => completions.push(DependencyWait::kick(
                target_owner,
                kick_handles[target_index].send(sender.clone()),
            )),
            ModuleState::PendingBatch { pending_kicks, .. } => {
                let (kick, completion) = Kick::new(sender.clone());
                pending_kicks.push(kick);
                completions.push(DependencyWait::kick(target_owner, completion));
            }
            ModuleState::Stored { .. }
            | ModuleState::Awaiting
            | ModuleState::WaitingForActivation
            | ModuleState::FailedUntilActivation { .. }
            | ModuleState::WaitingAfterFailure { .. } => {}
        }
    }
    completions
}

#[allow(clippy::too_many_arguments)]
async fn schedule_stored_dependency_for_wake(
    runtime: &AgentRuntimeControl,
    tasks: &mut FuturesUnordered<JoinHandle<TaskMessage>>,
    target_index: usize,
    target_owner: ModuleInstanceId,
    sender: ModuleInstanceId,
    states: &mut [ModuleState],
    kick_inboxes: &mut [Option<KickInbox>],
    kick_handles: &[KickHandle],
    config: AgentEventLoopConfig,
    parent: &tracing::Span,
    subscriber: &tracing::Dispatch,
) -> oneshot::Receiver<()> {
    let ModuleState::Stored {
        module,
        next_batch_throttle,
    } = std::mem::replace(&mut states[target_index], ModuleState::Awaiting)
    else {
        unreachable!("dependency state changed while scheduling stored dependency");
    };
    let kick_inbox = kick_inboxes[target_index]
        .take()
        .expect("kick inbox available for stored dependency");
    let now = runtime.clock().now();
    if let Some(throttle) =
        current_next_batch_throttle(runtime, &target_owner, next_batch_throttle, now).await
    {
        if let Some(activation_increase) = runtime
            .activation_increase_waiter(&target_owner, throttle.activation_threshold)
            .await
        {
            states[target_index] = ModuleState::Throttling;
            let allocation_change = runtime.allocation_change_waiter().await;
            spawn_batch_throttle(
                tasks,
                target_index,
                module,
                kick_inbox,
                throttle,
                runtime.clock(),
                activation_increase,
                allocation_change,
                parent,
                subscriber,
            );
            return kick_handles[target_index].send(sender);
        }
    }

    let wake_claim = runtime.claim_wake(&target_owner);
    let (kick, completion) = Kick::new(sender);
    spawn_next_batch(
        runtime.clone(),
        tasks,
        target_index,
        target_owner,
        module,
        kick_inbox,
        vec![kick],
        wake_claim,
        config.dependency_idle_timeout,
        parent,
        subscriber,
    );
    completion
}

#[allow(clippy::too_many_arguments)]
fn spawn_dependency_settle_wait(
    tasks: &mut FuturesUnordered<JoinHandle<TaskMessage>>,
    index: usize,
    owner: ModuleInstanceId,
    module: AllocatedModule,
    mut kick_inbox: KickInbox,
    mut pending_kicks: Vec<Kick>,
    wake_claim: Option<WakeClaim>,
    completions: Vec<DependencyWait>,
    remaining_waves: u8,
    hard_timeout: Duration,
    parent: &tracing::Span,
    subscriber: &tracing::Dispatch,
) {
    tasks.push(spawn_local(
        async move {
            let started = Instant::now();
            let mut remaining = completions
                .iter()
                .map(|completion| completion.owner.clone())
                .collect::<Vec<_>>();
            let mut completions = completions
                .into_iter()
                .map(DependencyWait::wait)
                .collect::<FuturesUnordered<_>>();
            let hard_deadline = sleep(hard_timeout);
            tokio::pin!(hard_deadline);

            while !completions.is_empty() {
                tokio::select! {
                    biased;
                    _ = &mut hard_deadline => {
                        tracing::warn!(
                            dependent = %owner,
                            remaining_dependencies = ?remaining,
                            elapsed_ms = started.elapsed().as_millis(),
                            timeout_ms = hard_timeout.as_millis(),
                            reason = "dependency_settle_hard_timeout",
                            "dependency settle hard timeout; starting dependent without remaining dependencies"
                        );
                        break;
                    }
                    completed = completions.next() => {
                        if let Some(completed) = completed {
                            remaining.retain(|owner| owner != &completed);
                        } else {
                            break;
                        }
                    }
                }
            }
            pending_kicks.extend(kick_inbox.drain_ready());
            TaskMessage::DependencySettled {
                index,
                module,
                kick_inbox,
                pending_kicks,
                wake_claim,
                remaining_waves,
            }
        }
        .instrument(parent.clone())
        .with_subscriber(subscriber.clone()),
    ));
}

enum PreBatchScheduling {
    Awaiting,
    PendingDependencySettle,
}

impl PreBatchScheduling {
    fn module_state(self) -> ModuleState {
        match self {
            Self::Awaiting => ModuleState::Awaiting,
            Self::PendingDependencySettle => ModuleState::PendingDependencySettle,
        }
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
    self_wake_activation_permit: bool,
    config: AgentEventLoopConfig,
    parent: &tracing::Span,
    subscriber: &tracing::Dispatch,
) -> ActivationScheduling {
    let gate_requests = runtime
        .activation_gate_requests(&owner, batch.clone())
        .await;
    if gate_requests.is_empty() {
        runtime
            .record_module_status(owner.clone(), ModuleRunStatus::Activating)
            .await;
        runtime.record_module_batch_ready(owner.clone(), &batch);
        let peer_contexts = runtime.peer_contexts();
        let allocation_hints = runtime.allocation_hints();
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
            peer_contexts,
            allocation_hints,
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
            self_wake_activation_permit,
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

#[allow(clippy::too_many_arguments)]
fn spawn_failure_wait(
    tasks: &mut FuturesUnordered<JoinHandle<TaskMessage>>,
    index: usize,
    module: AllocatedModule,
    mut kick_inbox: KickInbox,
    activation_waiter: Option<tokio::sync::oneshot::Receiver<()>>,
    allocation_change_waiter: tokio::sync::oneshot::Receiver<()>,
    zero_window_waiter: tokio::sync::oneshot::Receiver<()>,
    retry_after: Option<DateTime<Utc>>,
    clock: Rc<dyn Clock>,
    parent: &tracing::Span,
    subscriber: &tracing::Dispatch,
) {
    tasks.push(spawn_local(
        async move {
            let mut activation = pin!(async move {
                if let Some(waiter) = activation_waiter {
                    let _ = waiter.await;
                } else {
                    std::future::pending::<()>().await;
                }
            });
            let mut allocation_change = pin!(allocation_change_waiter);
            let mut zero_window = pin!(zero_window_waiter);
            let mut retry_deadline = pin!(async move {
                if let Some(deadline) = retry_after {
                    clock.sleep_until(deadline).await;
                } else {
                    std::future::pending::<()>().await;
                }
            });
            let mut kick_inbox_closed = false;
            loop {
                tokio::select! {
                    biased;
                    _ = &mut activation => break,
                    _ = &mut allocation_change => break,
                    _ = &mut zero_window => break,
                    _ = &mut retry_deadline => break,
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
            TaskMessage::FailureWaitReady {
                index,
                module,
                kick_inbox,
            }
        }
        .instrument(parent.clone())
        .with_subscriber(subscriber.clone()),
    ));
}

fn spawn_next_batch(
    runtime: AgentRuntimeControl,
    tasks: &mut FuturesUnordered<JoinHandle<TaskMessage>>,
    index: usize,
    owner: ModuleInstanceId,
    mut module: AllocatedModule,
    mut kick_inbox: KickInbox,
    initial_pending_kicks: Vec<Kick>,
    wake_claim: Option<WakeClaim>,
    dependency_idle_timeout: Duration,
    parent: &tracing::Span,
    subscriber: &tracing::Dispatch,
) {
    tasks.push(spawn_local(
        async move {
            let (batch_result, mut pending_kicks, wake_claim, self_wake_permit_claim) = {
                let mut pending_kicks = initial_pending_kicks;
                let mut wake_claim = wake_claim;
                let mut self_wake_permit_claim = runtime.claim_self_wake_permit(&owner);
                let mut observed_wake_sequence = runtime.wake_change_sequence();
                let mut kick_inbox_closed = false;
                let mut next_batch = pin!(module.next_batch());

                loop {
                    let timeout_enabled = !pending_kicks.is_empty()
                        && wake_claim.is_none()
                        && !runtime.has_pending_wake(&owner);
                    tokio::select! {
                        biased;
                        _ = runtime.wake_changed_since(observed_wake_sequence), if wake_claim.is_none() || self_wake_permit_claim.is_none() => {
                            observed_wake_sequence = runtime.wake_change_sequence();
                            if wake_claim.is_none() {
                                wake_claim = runtime.claim_wake(&owner);
                            }
                            if self_wake_permit_claim.is_none() {
                                self_wake_permit_claim = runtime.claim_self_wake_permit(&owner);
                            }
                        }
                        result = &mut next_batch => break (
                            result,
                            pending_kicks,
                            wake_claim,
                            self_wake_permit_claim,
                        ),
                        maybe_kick = kick_inbox.next(), if !kick_inbox_closed => {
                            if let Some(kick) = maybe_kick {
                                pending_kicks.push(kick);
                            } else {
                                kick_inbox_closed = true;
                            }
                        }
                        _ = sleep(dependency_idle_timeout), if timeout_enabled => {
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
                wake_claim,
                self_wake_permit_claim,
                result,
            }
        }
        .instrument(parent.clone())
        .with_subscriber(subscriber.clone()),
    ));
}

#[allow(clippy::too_many_arguments)]
fn spawn_batch_throttle(
    tasks: &mut FuturesUnordered<JoinHandle<TaskMessage>>,
    index: usize,
    module: AllocatedModule,
    kick_inbox: KickInbox,
    throttle: NextBatchThrottle,
    clock: Rc<dyn Clock>,
    activation_increase: tokio::sync::oneshot::Receiver<()>,
    allocation_change: tokio::sync::oneshot::Receiver<()>,
    parent: &tracing::Span,
    subscriber: &tracing::Dispatch,
) {
    tasks.push(spawn_local(
        async move {
            let started = clock.now();
            let deadline = throttle.not_before;
            let mut next_batch_throttle = None;
            tokio::select! {
                biased;
                _ = activation_increase => {},
                _ = allocation_change => {
                    next_batch_throttle = Some(throttle);
                },
                _ = clock.sleep_until(deadline) => {},
            }
            let delayed_for = (clock.now() - started).to_std().unwrap_or(Duration::ZERO);
            TaskMessage::BatchThrottleExpired {
                index,
                module,
                kick_inbox,
                delayed_for,
                next_batch_throttle,
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
    self_wake_activation_permit: bool,
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
                self_wake_activation_permit,
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
    peer_contexts: Vec<(ModuleId, &'static str)>,
    allocation_hints: Vec<(ModuleId, &'static str)>,
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
                &peer_contexts,
                &allocation_hints,
                &identity_memories,
                &core_policies,
                &batch,
                config.max_activation_attempts,
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
    peer_contexts: &[(ModuleId, &'static str)],
    allocation_hints: &[(ModuleId, &'static str)],
    identity_memories: &[IdentityMemoryRecord],
    core_policies: &[CorePolicyRecord],
    batch: &ModuleBatch,
    max_activation_attempts: u8,
) -> (AllocatedModule, Result<(), String>) {
    let module_owner = module.owner().clone();
    let module_tier = runtime.tier_for(&module_owner).await;
    let max_attempts = u32::from(max_activation_attempts.max(1));
    let mut activation_attempt = 1_u32;
    loop {
        let owner = module.owner().clone();
        let cx = runtime.with_session_checkpoint_runtime(
            ActivateCx::new(
                peer_contexts,
                allocation_hints,
                identity_memories,
                core_policies,
                SessionCompactionRuntime::new(
                    runtime
                        .session_compaction_handle()
                        .lutum
                        .clone()
                        .with_extension(LlmRequestMetadata {
                            owner: module_owner.clone(),
                            tier: ModelTier::Cheap,
                            source: LlmRequestSource::SessionCompaction,
                            session_key: None,
                            activation_attempt: Some(activation_attempt),
                            batch: Some(LlmBatchDebug::from_batch(batch)),
                        }),
                    runtime.session_compaction_handle().concurrency.clone(),
                    module_tier,
                    runtime.session_compaction_policy(),
                ),
                runtime.clock().now(),
            ),
            owner.clone(),
        );
        let activation_span = tracing::info_span!(
            target: "lutum",
            "module_activate",
            lutum.capture = true,
            owner = %owner,
            module = %owner.module,
            replica = owner.replica.get(),
            activation_attempt,
        );
        let activation_result = with_activation_llm_request_metadata(
            activation_attempt,
            batch,
            module.activate(&cx, batch).instrument(activation_span),
        )
        .await;
        match activation_result {
            Ok(()) => return (module, Ok(())),
            Err(error) => {
                let message = format!("{error:#}");
                runtime.record_module_activation_attempt_failed(
                    module.owner().clone(),
                    activation_attempt,
                    max_attempts,
                    message.clone(),
                );
                if activation_attempt >= max_attempts {
                    return (module, Err(message));
                }
                tracing::warn!(
                    owner = %module.owner(),
                    activation_attempt,
                    max_attempts,
                    error = %message,
                    "module activation failed; retrying"
                );
                activation_attempt = activation_attempt.saturating_add(1);
            }
        }
    }
}

fn is_idle(active: &[bool], states: &[ModuleState]) -> bool {
    let mut active_count = 0_usize;
    for (is_active, state) in active.iter().copied().zip(states) {
        if matches!(
            state,
            ModuleState::FailedUntilActivation { .. } | ModuleState::WaitingAfterFailure { .. }
        ) {
            continue;
        }
        if !is_active {
            continue;
        }
        active_count += 1;
        if !matches!(state, ModuleState::Awaiting | ModuleState::Stored { .. }) {
            return false;
        }
    }
    active_count > 0
}

#[cfg(test)]
mod tests {
    use super::AgentEventLoopConfig;

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
        AttentionControlRequestInbox, CognitionLogReader, CognitionLogUpdated,
        CognitionLogUpdatedInbox, CognitionWriter, LlmAccess, LlmBatchDebug, LlmRequestMetadata,
        LlmRequestSource, Memo, MemoUpdated, MemoUpdatedInbox, Module, ModuleCapabilityFactory,
        ModuleDependencies, ModuleRegistry, ModuleRegistryError, PersistedSessionSnapshot,
        RuntimeEvent, RuntimeEventSink, RuntimePolicy, SelfWake, SessionCompactionPolicy,
        SessionKey, SessionStore,
    };
    use nuillu_types::{
        MemoryContent, MemoryIndex, ModelTier, ModuleId, ModuleInstanceId, ReplicaCapRange,
        ReplicaIndex, builtin,
    };
    use tokio::sync::oneshot;
    use tokio::task::LocalSet;

    use crate::testing::{
        test_caps, test_caps_with_event_sink, test_caps_with_policy, test_caps_with_real_clock,
        test_caps_with_session_store,
    };

    #[derive(Clone, Default)]
    struct RecordingRuntimeEventSink {
        events: Rc<RefCell<Vec<RuntimeEvent>>>,
    }

    impl RecordingRuntimeEventSink {
        fn events(&self) -> Vec<RuntimeEvent> {
            self.events.borrow().clone()
        }
    }

    impl RuntimeEventSink for RecordingRuntimeEventSink {
        fn on_event(&self, event: RuntimeEvent) -> Result<(), nuillu_module::ports::PortError> {
            self.events.borrow_mut().push(event);
            Ok(())
        }
    }

    #[derive(Clone, Default)]
    struct DeletingSessionStore {
        deleted: Rc<RefCell<Vec<ModuleInstanceId>>>,
    }

    impl DeletingSessionStore {
        fn deleted(&self) -> Vec<ModuleInstanceId> {
            self.deleted.borrow().clone()
        }
    }

    #[async_trait(?Send)]
    impl SessionStore for DeletingSessionStore {
        async fn load(
            &self,
            _owner: &ModuleInstanceId,
            _key: &SessionKey,
        ) -> Result<Option<PersistedSessionSnapshot>, nuillu_module::ports::PortError> {
            Ok(None)
        }

        async fn save(
            &self,
            _owner: &ModuleInstanceId,
            _key: &SessionKey,
            _snapshot: &PersistedSessionSnapshot,
        ) -> Result<(), nuillu_module::ports::PortError> {
            Ok(())
        }

        async fn delete_owner(
            &self,
            owner: &ModuleInstanceId,
        ) -> Result<u64, nuillu_module::ports::PortError> {
            self.deleted.borrow_mut().push(owner.clone());
            Ok(2)
        }
    }

    struct RestartProbeModule;

    #[async_trait(?Send)]
    impl Module for RestartProbeModule {
        type Batch = ();

        fn id() -> &'static str {
            "restart-probe"
        }

        fn peer_context() -> Option<&'static str> {
            None
        }

        async fn next_batch(&mut self) -> anyhow::Result<Self::Batch> {
            std::future::pending().await
        }

        async fn activate(
            &mut self,
            _cx: &nuillu_module::ActivateCx<'_>,
            _batch: &Self::Batch,
        ) -> anyhow::Result<()> {
            Ok(())
        }
    }

    trait TestModuleRegistryExt {
        fn register_sync<M, F>(
            self,
            policy: ModulePolicy,
            builder: F,
        ) -> Result<ModuleRegistry, ModuleRegistryError>
        where
            M: Module + 'static,
            F: Fn(ModuleCapabilityFactory) -> M + 'static;
    }

    impl TestModuleRegistryExt for ModuleRegistry {
        fn register_sync<M, F>(
            self,
            policy: ModulePolicy,
            builder: F,
        ) -> Result<ModuleRegistry, ModuleRegistryError>
        where
            M: Module + 'static,
            F: Fn(ModuleCapabilityFactory) -> M + 'static,
        {
            self.register(policy, move |caps| {
                std::future::ready(Ok::<M, ModuleRegistryError>(builder(caps)))
            })
        }
    }

    fn test_config() -> AgentEventLoopConfig {
        AgentEventLoopConfig {
            idle_threshold: std::time::Duration::from_millis(50),
            max_activation_attempts: 3,
            dependency_idle_timeout: std::time::Duration::from_secs(2),
            dependency_hard_timeout: std::time::Duration::from_secs(10),
        }
    }

    fn echo_id() -> ModuleId {
        ModuleId::new("echo").unwrap()
    }

    fn compaction_observer_id() -> ModuleId {
        ModuleId::new("compaction-observer").unwrap()
    }

    fn llm_metadata_observer_id() -> ModuleId {
        ModuleId::new("llm-metadata-observer").unwrap()
    }

    fn request_question(request: &AttentionControlRequest) -> &str {
        request.as_str()
    }

    fn request_question_owned(request: AttentionControlRequest) -> String {
        request.into_inner()
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

        fn peer_context() -> Option<&'static str> {
            Some("test stub")
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
            "allocation"
        }

        fn peer_context() -> Option<&'static str> {
            Some("test attention controller")
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

                fn peer_context() -> Option<&'static str> {
                    Some("test zero-replica target")
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
        on_seen: Option<oneshot::Sender<CompactionObservation>>,
    }

    #[derive(Debug, Eq, PartialEq)]
    struct CompactionObservation {
        metadata: Option<LlmRequestMetadata>,
        module_tier: ModelTier,
        threshold: u64,
    }

    #[async_trait(?Send)]
    impl Module for CompactionMetadataObserver {
        type Batch = AttentionControlRequest;

        fn id() -> &'static str {
            "compaction-observer"
        }

        fn peer_context() -> Option<&'static str> {
            Some("test stub")
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
                let session_compaction = cx.session_compaction();
                let metadata = session_compaction
                    .lutum()
                    .default_extensions()
                    .get::<LlmRequestMetadata>()
                    .cloned();
                let _ = tx.send(CompactionObservation {
                    metadata,
                    module_tier: session_compaction.module_tier(),
                    threshold: session_compaction.input_token_threshold(),
                });
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

    struct LlmMetadataRetryObserver {
        attention_control_inbox: AttentionControlRequestInbox,
        llm: LlmAccess,
        observations: Rc<RefCell<Vec<LlmMetadataObservation>>>,
        attempts: Rc<Cell<u8>>,
        on_done: Option<oneshot::Sender<()>>,
    }

    #[derive(Clone, Debug, PartialEq, Eq)]
    struct LlmMetadataObservation {
        module_turn: Option<LlmRequestMetadata>,
        session_compaction: Option<LlmRequestMetadata>,
    }

    #[async_trait(?Send)]
    impl Module for LlmMetadataRetryObserver {
        type Batch = AttentionControlRequest;

        fn id() -> &'static str {
            "llm-metadata-observer"
        }

        fn peer_context() -> Option<&'static str> {
            Some("test stub")
        }

        async fn next_batch(&mut self) -> anyhow::Result<Self::Batch> {
            Ok(self.attention_control_inbox.next_item().await?.body)
        }

        async fn activate(
            &mut self,
            cx: &nuillu_module::ActivateCx<'_>,
            _batch: &Self::Batch,
        ) -> anyhow::Result<()> {
            let module_turn = self
                .llm
                .lutum()
                .await
                .default_extensions()
                .get::<LlmRequestMetadata>()
                .cloned();
            let session_compaction = cx
                .session_compaction()
                .lutum()
                .default_extensions()
                .get::<LlmRequestMetadata>()
                .cloned();
            self.observations.borrow_mut().push(LlmMetadataObservation {
                module_turn,
                session_compaction,
            });
            let attempt = self.attempts.get().saturating_add(1);
            self.attempts.set(attempt);
            if attempt < 2 {
                anyhow::bail!("retry metadata probe");
            }
            if let Some(done) = self.on_done.take() {
                let _ = done.send(());
            }
            Ok(())
        }
    }

    #[async_trait(?Send)]
    impl Module for QueryBatchRecorder {
        type Batch = Vec<String>;

        fn id() -> &'static str {
            "batch-recorder"
        }

        fn peer_context() -> Option<&'static str> {
            Some("test stub")
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

        fn peer_context() -> Option<&'static str> {
            Some("test gated target")
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

                fn peer_context() -> Option<&'static str> {
                    Some("test activation gate")
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

        fn peer_context() -> Option<&'static str> {
            Some("test stub")
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

        fn peer_context() -> Option<&'static str> {
            Some("test stub")
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

        fn peer_context() -> Option<&'static str> {
            Some("test stub")
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

    struct FailsFiveAttemptsThenSucceeds {
        memo: Memo,
        attempts: Rc<Cell<u8>>,
        on_done: Option<oneshot::Sender<()>>,
        completed: bool,
    }

    #[async_trait(?Send)]
    impl Module for FailsFiveAttemptsThenSucceeds {
        type Batch = ();

        fn id() -> &'static str {
            "fails-five-attempts-then-succeeds"
        }

        fn peer_context() -> Option<&'static str> {
            Some("test parked activation")
        }

        async fn next_batch(&mut self) -> anyhow::Result<Self::Batch> {
            if self.completed {
                std::future::pending().await
            } else {
                Ok(())
            }
        }

        async fn activate(
            &mut self,
            _cx: &nuillu_module::ActivateCx<'_>,
            _batch: &Self::Batch,
        ) -> anyhow::Result<()> {
            let attempt = self.attempts.get().saturating_add(1);
            self.attempts.set(attempt);
            if attempt <= 5 {
                anyhow::bail!("activation attempt {attempt} failed");
            }
            self.completed = true;
            self.memo.write("fresh opportunity activated").await;
            if let Some(done) = self.on_done.take() {
                let _ = done.send(());
            }
            Ok(())
        }
    }

    struct FailsFirstBatchThenSucceeds {
        on_done: Option<oneshot::Sender<()>>,
        failed_once: bool,
        completed: bool,
    }

    #[async_trait(?Send)]
    impl Module for FailsFirstBatchThenSucceeds {
        type Batch = ();

        fn id() -> &'static str {
            "fails-first-batch-then-succeeds"
        }

        fn peer_context() -> Option<&'static str> {
            Some("test parked next_batch")
        }

        async fn next_batch(&mut self) -> anyhow::Result<Self::Batch> {
            if !self.failed_once {
                self.failed_once = true;
                anyhow::bail!("first next_batch fails");
            }
            if self.completed {
                std::future::pending().await
            } else {
                Ok(())
            }
        }

        async fn activate(
            &mut self,
            _cx: &nuillu_module::ActivateCx<'_>,
            _batch: &Self::Batch,
        ) -> anyhow::Result<()> {
            self.completed = true;
            if let Some(done) = self.on_done.take() {
                let _ = done.send(());
            }
            Ok(())
        }
    }

    struct StopAfterOneFailureDependency {
        batch_sent: bool,
    }

    #[async_trait(?Send)]
    impl Module for StopAfterOneFailureDependency {
        type Batch = ();

        fn id() -> &'static str {
            "stop-after-one-failure-dependency"
        }

        fn peer_context() -> Option<&'static str> {
            Some("test stopped dependency")
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
            anyhow::bail!("dependency stops on first failure")
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

        fn peer_context() -> Option<&'static str> {
            Some("test stub")
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

        fn peer_context() -> Option<&'static str> {
            Some("test stub")
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

        fn peer_context() -> Option<&'static str> {
            Some("test stub")
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

                fn peer_context() -> Option<&'static str> {
                    Some("test silent dependency")
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

        fn peer_context() -> Option<&'static str> {
            Some("test stub")
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

    struct PauseBlockingModule {
        entered: Option<oneshot::Sender<()>>,
        release: Option<oneshot::Receiver<()>>,
        done: Option<oneshot::Sender<()>>,
        batch_sent: bool,
    }

    #[async_trait(?Send)]
    impl Module for PauseBlockingModule {
        type Batch = ();

        fn id() -> &'static str {
            "pause-blocking"
        }

        fn peer_context() -> Option<&'static str> {
            Some("test pause blocking module")
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
            if let Some(done) = self.done.take() {
                let _ = done.send(());
            }
            Ok(())
        }
    }

    struct BlockingAllocationModule {
        attention_control_inbox: AttentionControlRequestInbox,
        entered: Option<oneshot::Sender<()>>,
        release: Option<oneshot::Receiver<()>>,
        done: Option<oneshot::Sender<()>>,
        batch_sent: bool,
    }

    #[async_trait(?Send)]
    impl Module for BlockingAllocationModule {
        type Batch = AttentionControlRequest;

        fn id() -> &'static str {
            "allocation"
        }

        fn peer_context() -> Option<&'static str> {
            Some("test blocking allocation")
        }

        async fn next_batch(&mut self) -> anyhow::Result<Self::Batch> {
            if self.batch_sent {
                std::future::pending().await
            } else {
                self.batch_sent = true;
                Ok(self.attention_control_inbox.next_item().await?.body)
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
            if let Some(done) = self.done.take() {
                let _ = done.send(());
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

        fn peer_context() -> Option<&'static str> {
            Some("test dependency")
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

        fn peer_context() -> Option<&'static str> {
            Some("test dependent")
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

        fn peer_context() -> Option<&'static str> {
            Some("test gated dependency")
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

        fn peer_context() -> Option<&'static str> {
            Some("test blocking activation gate")
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

    struct AttentionCognitionGateStub {
        input: AttentionControlRequestInbox,
        writer: CognitionWriter,
    }

    #[async_trait(?Send)]
    impl Module for AttentionCognitionGateStub {
        type Batch = String;

        fn id() -> &'static str {
            "cognition-gate"
        }

        fn peer_context() -> Option<&'static str> {
            Some("test cognition gate")
        }

        async fn next_batch(&mut self) -> anyhow::Result<Self::Batch> {
            Ok(self.input.next_item().await?.body.into_inner())
        }

        async fn activate(
            &mut self,
            _cx: &nuillu_module::ActivateCx<'_>,
            batch: &Self::Batch,
        ) -> anyhow::Result<()> {
            self.writer.append(format!("cognition:{batch}")).await;
            Ok(())
        }
    }

    struct QueryMemoryMemoStub {
        input: AttentionControlRequestInbox,
        memo: Memo,
    }

    #[async_trait(?Send)]
    impl Module for QueryMemoryMemoStub {
        type Batch = String;

        fn id() -> &'static str {
            "query-memory"
        }

        fn peer_context() -> Option<&'static str> {
            Some("test query memory")
        }

        async fn next_batch(&mut self) -> anyhow::Result<Self::Batch> {
            Ok(self.input.next_item().await?.body.into_inner())
        }

        async fn activate(
            &mut self,
            _cx: &nuillu_module::ActivateCx<'_>,
            batch: &Self::Batch,
        ) -> anyhow::Result<()> {
            self.memo.write(format!("retrieved:{batch}")).await;
            Ok(())
        }
    }

    struct MemoCognitionGateStub {
        updates: MemoUpdatedInbox,
        writer: CognitionWriter,
    }

    #[async_trait(?Send)]
    impl Module for MemoCognitionGateStub {
        type Batch = MemoUpdated;

        fn id() -> &'static str {
            "cognition-gate"
        }

        fn peer_context() -> Option<&'static str> {
            Some("test memo cognition gate")
        }

        async fn next_batch(&mut self) -> anyhow::Result<Self::Batch> {
            Ok(self.updates.next_item().await?.body)
        }

        async fn activate(
            &mut self,
            _cx: &nuillu_module::ActivateCx<'_>,
            batch: &Self::Batch,
        ) -> anyhow::Result<()> {
            self.writer
                .append(format!("memo-cognition:{}", batch.owner.module))
                .await;
            Ok(())
        }
    }

    struct SpeakCognitionSnapshotStub {
        input: AttentionControlRequestInbox,
        cognition: CognitionLogReader,
        observed: Rc<RefCell<Vec<Vec<String>>>>,
        on_done: Option<oneshot::Sender<()>>,
    }

    #[async_trait(?Send)]
    impl Module for SpeakCognitionSnapshotStub {
        type Batch = Vec<String>;

        fn id() -> &'static str {
            "speak"
        }

        fn peer_context() -> Option<&'static str> {
            Some("test speak")
        }

        async fn next_batch(&mut self) -> anyhow::Result<Self::Batch> {
            let _ = self.input.next_item().await?;
            Ok(self
                .cognition
                .unread_events()
                .await
                .into_iter()
                .map(|record| record.entry.text)
                .collect())
        }

        async fn activate(
            &mut self,
            _cx: &nuillu_module::ActivateCx<'_>,
            batch: &Self::Batch,
        ) -> anyhow::Result<()> {
            self.observed.borrow_mut().push(batch.clone());
            if let Some(done) = self.on_done.take() {
                let _ = done.send(());
            }
            Ok(())
        }
    }

    struct BlockingWakeClaimModule {
        input: AttentionControlRequestInbox,
        first_drained: Option<oneshot::Sender<()>>,
        first_release: Option<oneshot::Receiver<()>>,
        batches: Rc<RefCell<Vec<String>>>,
        on_two: Option<oneshot::Sender<()>>,
    }

    #[async_trait(?Send)]
    impl Module for BlockingWakeClaimModule {
        type Batch = String;

        fn id() -> &'static str {
            "wake-claim-recorder"
        }

        fn peer_context() -> Option<&'static str> {
            Some("test wake claim")
        }

        async fn next_batch(&mut self) -> anyhow::Result<Self::Batch> {
            let body = self.input.next_item().await?.body.into_inner();
            if body == "first" {
                if let Some(drained) = self.first_drained.take() {
                    let _ = drained.send(());
                }
                if let Some(release) = self.first_release.take() {
                    let _ = release.await;
                }
            }
            Ok(body)
        }

        async fn activate(
            &mut self,
            _cx: &nuillu_module::ActivateCx<'_>,
            batch: &Self::Batch,
        ) -> anyhow::Result<()> {
            self.batches.borrow_mut().push(batch.clone());
            if self.batches.borrow().len() >= 2
                && let Some(done) = self.on_two.take()
            {
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

        fn peer_context() -> Option<&'static str> {
            Some("test failing batch")
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

        fn peer_context() -> Option<&'static str> {
            Some("test failing activation")
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

    #[tokio::test]
    async fn session_reset_restarts_stored_module_and_deletes_owner_sessions() {
        let store = DeletingSessionStore::default();
        let caps = test_caps_with_session_store(Blackboard::default(), Rc::new(store.clone()));
        let builds = Rc::new(Cell::new(0_u32));
        let observed_builds = Rc::clone(&builds);
        let allocated = ModuleRegistry::new()
            .register_sync::<RestartProbeModule, _>(test_policy(1..=1, fast_bpm()), move |_| {
                observed_builds.set(observed_builds.get().saturating_add(1));
                RestartProbeModule
            })
            .unwrap()
            .build(&caps)
            .await
            .unwrap();
        let (runtime, mut modules) = allocated.into_parts();
        let owner = ModuleInstanceId::new(
            ModuleId::new(RestartProbeModule::id()).unwrap(),
            ReplicaIndex::ZERO,
        );
        let mut states = vec![super::ModuleState::Stored {
            module: modules.remove(0),
            next_batch_throttle: None,
        }];
        let mut kick_inboxes = vec![None];
        let mut failures = vec![0_u32];
        let (response, mut result) = oneshot::channel();
        let mut pending = vec![VecDeque::from([response])];

        super::apply_pending_session_resets(
            &runtime,
            std::slice::from_ref(&owner),
            &mut states,
            &mut kick_inboxes,
            &mut failures,
            &mut pending,
        )
        .await;

        let reset = result.try_recv().expect("reset response sent").unwrap();
        assert_eq!(reset.owner, owner);
        assert_eq!(reset.deleted_sessions, 2);
        assert_eq!(store.deleted(), vec![reset.owner]);
        assert_eq!(builds.get(), 2);
        assert!(pending[0].is_empty());
        assert!(matches!(states[0], super::ModuleState::Stored { .. }));
    }

    #[tokio::test]
    async fn session_reset_waits_until_activating_module_returns() {
        let store = DeletingSessionStore::default();
        let caps = test_caps_with_session_store(Blackboard::default(), Rc::new(store.clone()));
        let builds = Rc::new(Cell::new(0_u32));
        let observed_builds = Rc::clone(&builds);
        let allocated = ModuleRegistry::new()
            .register_sync::<RestartProbeModule, _>(test_policy(1..=1, fast_bpm()), move |_| {
                observed_builds.set(observed_builds.get().saturating_add(1));
                RestartProbeModule
            })
            .unwrap()
            .build(&caps)
            .await
            .unwrap();
        let (runtime, mut modules) = allocated.into_parts();
        let owner = ModuleInstanceId::new(
            ModuleId::new(RestartProbeModule::id()).unwrap(),
            ReplicaIndex::ZERO,
        );
        let module = modules.remove(0);
        let mut states = vec![super::ModuleState::Activating];
        let mut kick_inboxes = vec![None];
        let mut failures = vec![0_u32];
        let (response, mut result) = oneshot::channel();
        let mut pending = vec![VecDeque::from([response])];

        super::apply_pending_session_resets(
            &runtime,
            std::slice::from_ref(&owner),
            &mut states,
            &mut kick_inboxes,
            &mut failures,
            &mut pending,
        )
        .await;

        assert!(matches!(
            result.try_recv(),
            Err(tokio::sync::oneshot::error::TryRecvError::Empty)
        ));
        assert_eq!(pending[0].len(), 1);
        assert_eq!(store.deleted(), Vec::<ModuleInstanceId>::new());

        states[0] = super::ModuleState::Stored {
            module,
            next_batch_throttle: None,
        };
        super::apply_pending_session_resets(
            &runtime,
            std::slice::from_ref(&owner),
            &mut states,
            &mut kick_inboxes,
            &mut failures,
            &mut pending,
        )
        .await;

        let reset = result.try_recv().expect("reset response sent").unwrap();
        assert_eq!(reset.owner, owner);
        assert_eq!(store.deleted(), vec![reset.owner]);
        assert_eq!(builds.get(), 2);
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
                    .register_sync(
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
                        .publish(AttentionControlRequest::new("ping"))
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
    async fn run_control_pauses_queued_work_until_resumed() {
        let local = LocalSet::new();
        local
            .run_until(async {
                let mut alloc = ResourceAllocation::default();
                alloc.set(echo_id(), ModuleConfig::default());
                alloc.set_activation(echo_id(), ActivationRatio::ONE);

                let blackboard = Blackboard::with_allocation(alloc);
                let caps = test_caps(blackboard);
                let (done_tx, mut done_rx) = oneshot::channel();
                let done_tx = Rc::new(RefCell::new(Some(done_tx)));
                let modules = ModuleRegistry::new()
                    .register_sync(
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
                let (controller, control) = super::AgentRunController::new();
                controller.pause();

                super::run_controlled(modules, test_config(), control, async move {
                    mailbox
                        .publish(AttentionControlRequest::new("paused ping"))
                        .await
                        .expect("attention request should queue");
                    for _ in 0..4 {
                        tokio::task::yield_now().await;
                    }
                    assert_oneshot_pending(
                        &mut done_rx,
                        "paused scheduler should not activate queued work",
                    );

                    controller.resume();
                    done_rx
                        .await
                        .expect("queued work should activate after resume");
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
    async fn run_control_does_not_abort_in_flight_activation() {
        let local = LocalSet::new();
        local
            .run_until(async {
                let module_id = ModuleId::new("pause-blocking").unwrap();
                let mut alloc = ResourceAllocation::default();
                alloc.set(module_id.clone(), ModuleConfig::default());
                alloc.set_activation(module_id, ActivationRatio::ONE);

                let blackboard = Blackboard::with_allocation(alloc);
                let caps = test_caps(blackboard);
                let (entered_tx, entered_rx) = oneshot::channel();
                let (release_tx, release_rx) = oneshot::channel();
                let (done_tx, mut done_rx) = oneshot::channel();
                let entered_tx = Rc::new(RefCell::new(Some(entered_tx)));
                let release_rx = Rc::new(RefCell::new(Some(release_rx)));
                let done_tx = Rc::new(RefCell::new(Some(done_tx)));
                let modules = ModuleRegistry::new()
                    .register_sync(test_policy(0..=1, fast_bpm()), {
                        let entered_tx = Rc::clone(&entered_tx);
                        let release_rx = Rc::clone(&release_rx);
                        let done_tx = Rc::clone(&done_tx);
                        move |_caps| PauseBlockingModule {
                            entered: entered_tx.borrow_mut().take(),
                            release: release_rx.borrow_mut().take(),
                            done: done_tx.borrow_mut().take(),
                            batch_sent: false,
                        }
                    })
                    .unwrap()
                    .build(&caps)
                    .await
                    .unwrap();
                let (controller, control) = super::AgentRunController::new();

                super::run_controlled(modules, test_config(), control, async move {
                    entered_rx.await.expect("activation should start");
                    controller.pause();
                    for _ in 0..4 {
                        tokio::task::yield_now().await;
                    }
                    assert_oneshot_pending(
                        &mut done_rx,
                        "blocking activation should still be in flight while paused",
                    );

                    release_tx
                        .send(())
                        .expect("activation release receiver should remain alive");
                    done_rx
                        .await
                        .expect("in-flight activation should complete while paused");
                })
                .await
                .expect("scheduler returned err");
            })
            .await;
    }

    #[tokio::test(flavor = "current_thread", start_paused = false)]
    async fn paused_allocation_completion_does_not_open_zero_replica_window() {
        let local = LocalSet::new();
        local
            .run_until(async {
                let target_id = ModuleId::new("zero-window-target").unwrap();
                let mut alloc = ResourceAllocation::default();
                alloc.set(builtin::allocation(), ModuleConfig::default());
                alloc.set_activation(builtin::allocation(), ActivationRatio::ONE);
                alloc.set(target_id.clone(), ModuleConfig::default());
                alloc.set_activation(target_id, ActivationRatio::ZERO);

                let blackboard = Blackboard::with_allocation(alloc);
                let caps = test_caps(blackboard);
                let (entered_tx, entered_rx) = oneshot::channel();
                let (release_tx, release_rx) = oneshot::channel();
                let (done_tx, done_rx) = oneshot::channel();
                let entered_tx = Rc::new(RefCell::new(Some(entered_tx)));
                let release_rx = Rc::new(RefCell::new(Some(release_rx)));
                let done_tx = Rc::new(RefCell::new(Some(done_tx)));
                let target_activations = Rc::new(Cell::new(0_u32));
                let target_batches = Rc::new(RefCell::new(Vec::new()));
                let mut target_policy = test_policy(0..=1, fast_bpm());
                target_policy.zero_replica_window =
                    ZeroReplicaWindowPolicy::EveryControllerActivations(1);
                let modules = ModuleRegistry::new()
                    .register_sync(test_policy(0..=1, fast_bpm()), {
                        let entered_tx = Rc::clone(&entered_tx);
                        let release_rx = Rc::clone(&release_rx);
                        let done_tx = Rc::clone(&done_tx);
                        move |caps| BlockingAllocationModule {
                            attention_control_inbox: caps.attention_control_inbox(),
                            entered: entered_tx.borrow_mut().take(),
                            release: release_rx.borrow_mut().take(),
                            done: done_tx.borrow_mut().take(),
                            batch_sent: false,
                        }
                    })
                    .unwrap()
                    .register_sync(target_policy, {
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
                let (controller, control) = super::AgentRunController::new();

                super::run_controlled(modules, test_config(), control, async move {
                    mailbox
                        .publish(AttentionControlRequest::new("allocation tick"))
                        .await
                        .expect("attention request should route to allocation");
                    entered_rx
                        .await
                        .expect("allocation activation should start");
                    controller.pause();
                    release_tx
                        .send(())
                        .expect("allocation release receiver should remain alive");
                    done_rx
                        .await
                        .expect("allocation activation should complete");
                    for _ in 0..4 {
                        tokio::task::yield_now().await;
                    }

                    controller.resume();
                    tokio::time::sleep(Duration::from_millis(20)).await;
                    assert_eq!(
                        target_activations.get(),
                        0,
                        "paused allocation completion must not open a zero-replica window"
                    );
                    assert!(target_batches.borrow().is_empty());
                })
                .await
                .expect("scheduler returned err");
            })
            .await;
    }

    #[tokio::test(flavor = "current_thread", start_paused = false)]
    async fn activation_context_uses_module_tier_threshold_and_compaction_lutum_metadata() {
        let local = LocalSet::new();
        local
            .run_until(async {
                let mut alloc = ResourceAllocation::default();
                alloc.set(compaction_observer_id(), ModuleConfig::default());
                alloc.set_model_override(compaction_observer_id(), ModelTier::Premium);
                alloc.set_activation(compaction_observer_id(), ActivationRatio::ONE);

                let blackboard = Blackboard::with_allocation(alloc);
                let caps = test_caps_with_policy(
                    blackboard,
                    RuntimePolicy {
                        session_compaction: SessionCompactionPolicy::new(11, 22, 33),
                        ..RuntimePolicy::default()
                    },
                );
                let (seen_tx, seen_rx) = oneshot::channel();
                let seen_tx = Rc::new(RefCell::new(Some(seen_tx)));
                let modules = ModuleRegistry::new()
                    .register_sync(
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
                        .publish(AttentionControlRequest::new("ping"))
                        .await
                        .expect("attention request should route to observer");
                    let observation = seen_rx.await.expect("metadata observed");
                    assert_eq!(
                        observation.metadata,
                        Some(LlmRequestMetadata {
                            owner: nuillu_types::ModuleInstanceId::new(
                                compaction_observer_id(),
                                ReplicaIndex::ZERO,
                            ),
                            tier: ModelTier::Cheap,
                            source: LlmRequestSource::SessionCompaction,
                            session_key: None,
                            activation_attempt: Some(1),
                            batch: Some(nuillu_module::LlmBatchDebug {
                                batch_type: std::any::type_name::<AttentionControlRequest>()
                                    .to_string(),
                                batch_debug: "\"ping\"".to_string(),
                            }),
                        })
                    );
                    assert_eq!(observation.module_tier, ModelTier::Premium);
                    assert_eq!(observation.threshold, 33);
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
    async fn llm_metadata_includes_activation_attempt_and_batch_for_module_and_compaction() {
        let local = LocalSet::new();
        local
            .run_until(async {
                let mut alloc = ResourceAllocation::default();
                alloc.set(llm_metadata_observer_id(), ModuleConfig::default());
                alloc.set_model_override(llm_metadata_observer_id(), ModelTier::Premium);
                alloc.set_activation(llm_metadata_observer_id(), ActivationRatio::ONE);

                let blackboard = Blackboard::with_allocation(alloc);
                let caps = test_caps_with_policy(
                    blackboard,
                    RuntimePolicy {
                        session_compaction: SessionCompactionPolicy::new(11, 22, 33),
                        ..RuntimePolicy::default()
                    },
                );
                let observations = Rc::new(RefCell::new(Vec::new()));
                let attempts = Rc::new(Cell::new(0_u8));
                let (done_tx, done_rx) = oneshot::channel();
                let done_tx = Rc::new(RefCell::new(Some(done_tx)));
                let modules = ModuleRegistry::new()
                    .register_sync(
                        test_policy(0..=1, Bpm::from_f64(60.0)..=Bpm::from_f64(60.0)),
                        {
                            let observations = Rc::clone(&observations);
                            let attempts = Rc::clone(&attempts);
                            let done_tx = Rc::clone(&done_tx);
                            move |caps| LlmMetadataRetryObserver {
                                attention_control_inbox: caps.attention_control_inbox(),
                                llm: caps.llm_access(),
                                observations: Rc::clone(&observations),
                                attempts: Rc::clone(&attempts),
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
                        .publish(AttentionControlRequest::new("ping"))
                        .await
                        .expect("attention request should route to observer");
                    done_rx.await.expect("second activation should finish");
                    for _ in 0..4 {
                        tokio::task::yield_now().await;
                    }
                })
                .await
                .expect("scheduler returned err");

                let expected_batch = LlmBatchDebug {
                    batch_type: std::any::type_name::<AttentionControlRequest>().to_string(),
                    batch_debug: "\"ping\"".to_string(),
                };
                let owner = ModuleInstanceId::new(llm_metadata_observer_id(), ReplicaIndex::ZERO);
                let observations = observations.borrow();
                assert_eq!(observations.len(), 2);
                for (index, observation) in observations.iter().enumerate() {
                    let attempt = u32::try_from(index + 1).unwrap();
                    assert_eq!(
                        observation.module_turn,
                        Some(LlmRequestMetadata {
                            owner: owner.clone(),
                            tier: ModelTier::Premium,
                            source: LlmRequestSource::ModuleTurn,
                            session_key: None,
                            activation_attempt: Some(attempt),
                            batch: Some(expected_batch.clone()),
                        })
                    );
                    assert_eq!(
                        observation.session_compaction,
                        Some(LlmRequestMetadata {
                            owner: owner.clone(),
                            tier: ModelTier::Cheap,
                            source: LlmRequestSource::SessionCompaction,
                            session_key: None,
                            activation_attempt: Some(attempt),
                            batch: Some(expected_batch.clone()),
                        })
                    );
                }
            })
            .await;
    }

    #[tokio::test(flavor = "current_thread", start_paused = false)]
    async fn zero_replica_window_runs_queued_work_after_default_controller_period() {
        let local = LocalSet::new();
        local
            .run_until(async {
                let controller_id = builtin::allocation();
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
                    .register_sync(test_policy(1..=1, fast_bpm()), {
                        let controller_activations = Rc::clone(&controller_activations);
                        move |caps| ControllerTickModule {
                            attention_control_inbox: caps.attention_control_inbox(),
                            activations: Rc::clone(&controller_activations),
                        }
                    })
                    .unwrap()
                    .register_sync(test_policy(0..=1, fast_bpm()), {
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
                            .publish(AttentionControlRequest::new(format!("tick-{i}")))
                            .await
                            .expect("controller tick should route");
                        wait_for_cell_count(&controller_activations, i).await;
                        for _ in 0..4 {
                            tokio::task::yield_now().await;
                        }
                        assert_eq!(target_activations.get(), 0);
                    }

                    mailbox
                        .publish(AttentionControlRequest::new("tick-3"))
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
                let controller_id = builtin::allocation();
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
                    .register_sync(test_policy(1..=1, fast_bpm()), {
                        let controller_activations = Rc::clone(&controller_activations);
                        move |caps| ControllerTickModule {
                            attention_control_inbox: caps.attention_control_inbox(),
                            activations: Rc::clone(&controller_activations),
                        }
                    })
                    .unwrap()
                    .register_sync(
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
                    .register_sync(
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
                            .publish(AttentionControlRequest::new(format!("period-{i}")))
                            .await
                            .expect("controller tick should route");
                        wait_for_cell_count(&controller_activations, i).await;
                    }
                    wait_for_cell_count(&target_a_activations, 1).await;
                    assert_eq!(target_b_activations.get(), 0);

                    for i in 3..=4 {
                        mailbox
                            .publish(AttentionControlRequest::new(format!("period-{i}")))
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
                let controller_id = builtin::allocation();
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
                    .register_sync(test_policy(1..=1, fast_bpm()), {
                        let controller_activations = Rc::clone(&controller_activations);
                        move |caps| ControllerTickModule {
                            attention_control_inbox: caps.attention_control_inbox(),
                            activations: Rc::clone(&controller_activations),
                        }
                    })
                    .unwrap()
                    .register_sync(test_policy(0..=1, fast_bpm()), {
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
                            .publish(AttentionControlRequest::new(format!("pre-{i}")))
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
                            .publish(AttentionControlRequest::new(format!("post-{i}")))
                            .await
                            .expect("controller tick should route");
                        wait_for_cell_count(&controller_activations, i).await;
                    }
                    for _ in 0..4 {
                        tokio::task::yield_now().await;
                    }
                    assert_eq!(target_activations.get(), post_reset_baseline);

                    mailbox
                        .publish(AttentionControlRequest::new("post-5"))
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
                let controller_id = builtin::allocation();
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
                    .register_sync(test_policy(1..=1, fast_bpm()), {
                        let controller_activations = Rc::clone(&controller_activations);
                        move |caps| ControllerTickModule {
                            attention_control_inbox: caps.attention_control_inbox(),
                            activations: Rc::clone(&controller_activations),
                        }
                    })
                    .unwrap()
                    .register_sync(
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
                            .publish(AttentionControlRequest::new(format!("one-shot-{i}")))
                            .await
                            .expect("controller tick should route");
                        wait_for_cell_count(&controller_activations, i).await;
                    }
                    wait_for_cell_count(&target_activations, 1).await;

                    mailbox
                        .publish(AttentionControlRequest::new("one-shot-3"))
                        .await
                        .expect("next queued message should not immediately re-open");
                    wait_for_cell_count(&controller_activations, 3).await;
                    for _ in 0..4 {
                        tokio::task::yield_now().await;
                    }
                    assert_eq!(target_activations.get(), 1);

                    mailbox
                        .publish(AttentionControlRequest::new("one-shot-4"))
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
                let controller_id = builtin::allocation();
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
                    .register_sync(test_policy(1..=1, fast_bpm()), {
                        let controller_activations = Rc::clone(&controller_activations);
                        move |caps| ControllerTickModule {
                            attention_control_inbox: caps.attention_control_inbox(),
                            activations: Rc::clone(&controller_activations),
                        }
                    })
                    .unwrap()
                    .register_sync(
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
                            .publish(AttentionControlRequest::new(format!("disabled-{i}")))
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
                let controller_id = builtin::allocation();
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
                    .register_sync(test_policy(1..=1, fast_bpm()), {
                        let controller_activations = Rc::clone(&controller_activations);
                        move |caps| ControllerTickModule {
                            attention_control_inbox: caps.attention_control_inbox(),
                            activations: Rc::clone(&controller_activations),
                        }
                    })
                    .unwrap()
                    .register_sync(test_policy(0..=0, fast_bpm()), {
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
                            .publish(AttentionControlRequest::new(format!("hard-{i}")))
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
            .register_sync(
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
                .register_sync(
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
                .register_sync(
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
                .publish(AttentionControlRequest::new("gated"))
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
                    .register_sync(
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
                // Use real wall-clock so the test can observe the 30ms period
                // actually delaying the second batch.
                let caps = test_caps_with_real_clock(blackboard);
                let batches = Rc::new(RefCell::new(Vec::<Vec<String>>::new()));
                let (first_tx, first_rx) = oneshot::channel();
                let (second_tx, second_rx) = oneshot::channel();
                let first_tx = Rc::new(RefCell::new(Some(first_tx)));
                let second_tx = Rc::new(RefCell::new(Some(second_tx)));
                // 2000 BPM = 30ms period per batch under linear_ratio_fn at
                // activation_ratio=1.0.
                let modules = ModuleRegistry::new()
                    .register_sync(
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
                        .publish(AttentionControlRequest::new("first"))
                        .await
                        .expect("first attention request should route");
                    let _ = first_rx.await;

                    let started = tokio::time::Instant::now();
                    mailbox
                        .publish(AttentionControlRequest::new("second"))
                        .await
                        .expect("second attention request should route");
                    mailbox
                        .publish(AttentionControlRequest::new("third"))
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
                // The first activation takes ~80ms, so remaining sleep
                // should be ~120ms rather than a full 200ms.
                let modules = ModuleRegistry::new()
                    .register_sync(
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
                        .publish(AttentionControlRequest::new("first"))
                        .await
                        .expect("first attention request should route");
                    let _ = first_rx.await;

                    let started = tokio::time::Instant::now();
                    mailbox
                        .publish(AttentionControlRequest::new("second"))
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
                // takes longer than that, no extra sleep should be added.
                let modules = ModuleRegistry::new()
                    .register_sync(
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
                        .publish(AttentionControlRequest::new("first"))
                        .await
                        .expect("first attention request should route");
                    let _ = first_rx.await;

                    let started = tokio::time::Instant::now();
                    mailbox
                        .publish(AttentionControlRequest::new("second"))
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
                // during throttling and expects the second batch before deadline.
                let modules = ModuleRegistry::new()
                    .register_sync(
                        test_policy(1..=1, Bpm::from_f64(60.0)..=Bpm::from_f64(60.0)),
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
                        .publish(AttentionControlRequest::new("first"))
                        .await
                        .expect("first attention request should route");
                    let _ = first_rx.await;

                    mailbox
                        .publish(AttentionControlRequest::new("second"))
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
                    .register_sync(
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
                    .register_sync(
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
                    .register_sync(
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
                let event_sink = Rc::new(RecordingRuntimeEventSink::default());
                let caps = test_caps_with_event_sink(blackboard.clone(), event_sink.clone());
                let attempts = Rc::new(Cell::new(0));
                let (done_tx, done_rx) = oneshot::channel();
                let done_tx = Rc::new(RefCell::new(Some(done_tx)));
                let modules = ModuleRegistry::new()
                    .register_sync(
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
                        max_activation_attempts: 3,
                        dependency_idle_timeout: std::time::Duration::from_secs(2),
                        dependency_hard_timeout: std::time::Duration::from_secs(10),
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
                let attempt_failures = event_sink
                    .events()
                    .into_iter()
                    .filter_map(|event| match event {
                        RuntimeEvent::ModuleActivationAttemptFailed {
                            owner,
                            activation_attempt,
                            max_attempts,
                            message,
                            ..
                        } if owner.module == retry_id => {
                            Some((activation_attempt, max_attempts, message))
                        }
                        _ => None,
                    })
                    .collect::<Vec<_>>();
                assert_eq!(
                    attempt_failures,
                    vec![
                        (1, 3, "transient activation failure".to_string()),
                        (2, 3, "transient activation failure".to_string()),
                    ]
                );
            })
            .await;
    }

    #[tokio::test(flavor = "current_thread", start_paused = false)]
    async fn dependency_appends_cognition_before_dependent_builds_batch() {
        let local = LocalSet::new();
        local
            .run_until(async {
                let cognition_id = ModuleId::new(AttentionCognitionGateStub::id()).unwrap();
                let speak_id = ModuleId::new(SpeakCognitionSnapshotStub::id()).unwrap();
                let mut alloc = ResourceAllocation::default();
                for module in [cognition_id.clone(), speak_id.clone()] {
                    alloc.set(module.clone(), ModuleConfig::default());
                    alloc.set_activation(module, ActivationRatio::ONE);
                }

                let blackboard = Blackboard::with_allocation(alloc);
                let caps = test_caps(blackboard);
                let observed = Rc::new(RefCell::new(Vec::<Vec<String>>::new()));
                let (done_tx, done_rx) = oneshot::channel();
                let done_tx = Rc::new(RefCell::new(Some(done_tx)));
                let modules = ModuleRegistry::new()
                    .register_sync(
                        test_policy(1..=1, Bpm::from_f64(60.0)..=Bpm::from_f64(60.0)),
                        {
                            let observed = Rc::clone(&observed);
                            let done_tx = Rc::clone(&done_tx);
                            move |caps| SpeakCognitionSnapshotStub {
                                input: caps.attention_control_inbox(),
                                cognition: caps.cognition_log_reader(),
                                observed: Rc::clone(&observed),
                                on_done: done_tx.borrow_mut().take(),
                            }
                        },
                    )
                    .unwrap()
                    .register_sync(
                        test_policy(1..=1, Bpm::from_f64(60.0)..=Bpm::from_f64(60.0)),
                        |caps| AttentionCognitionGateStub {
                            input: caps.attention_control_inbox(),
                            writer: caps.cognition_writer(),
                        },
                    )
                    .unwrap()
                    .depends_on(speak_id.clone(), cognition_id.clone())
                    .build(&caps)
                    .await
                    .unwrap();
                let mailbox = caps.internal_harness_io().attention_control_mailbox();
                let delivered = mailbox
                    .publish(AttentionControlRequest::new("bright-water"))
                    .await
                    .expect("attention request should wake both modules");
                assert_eq!(delivered, 2);

                super::run(
                    modules,
                    AgentEventLoopConfig {
                        dependency_idle_timeout: Duration::from_millis(10),
                        ..test_config()
                    },
                    async move {
                        tokio::time::timeout(Duration::from_millis(500), done_rx)
                            .await
                            .expect("speak should activate after dependency settles")
                            .expect("done sender dropped");
                    },
                )
                .await
                .expect("scheduler returned err");

                assert_eq!(
                    observed.borrow().as_slice(),
                    &[vec!["cognition:bright-water".to_owned()]]
                );
            })
            .await;
    }

    #[tokio::test(flavor = "current_thread", start_paused = false)]
    async fn dependency_chain_settles_two_waves_before_dependent_builds_batch() {
        let local = LocalSet::new();
        local
            .run_until(async {
                let speak_id = ModuleId::new(SpeakCognitionSnapshotStub::id()).unwrap();
                let query_id = ModuleId::new(QueryMemoryMemoStub::id()).unwrap();
                let cognition_id = ModuleId::new(MemoCognitionGateStub::id()).unwrap();
                let mut alloc = ResourceAllocation::default();
                for module in [speak_id.clone(), query_id.clone(), cognition_id.clone()] {
                    alloc.set(module.clone(), ModuleConfig::default());
                    alloc.set_activation(module, ActivationRatio::ONE);
                }

                let blackboard = Blackboard::with_allocation(alloc);
                let caps = test_caps(blackboard);
                let observed = Rc::new(RefCell::new(Vec::<Vec<String>>::new()));
                let (done_tx, done_rx) = oneshot::channel();
                let done_tx = Rc::new(RefCell::new(Some(done_tx)));
                let modules = ModuleRegistry::new()
                    .register_sync(
                        test_policy(1..=1, Bpm::from_f64(60.0)..=Bpm::from_f64(60.0)),
                        {
                            let observed = Rc::clone(&observed);
                            let done_tx = Rc::clone(&done_tx);
                            move |caps| SpeakCognitionSnapshotStub {
                                input: caps.attention_control_inbox(),
                                cognition: caps.cognition_log_reader(),
                                observed: Rc::clone(&observed),
                                on_done: done_tx.borrow_mut().take(),
                            }
                        },
                    )
                    .unwrap()
                    .register_sync(
                        test_policy(1..=1, Bpm::from_f64(60.0)..=Bpm::from_f64(60.0)),
                        |caps| QueryMemoryMemoStub {
                            input: caps.attention_control_inbox(),
                            memo: caps.memo(),
                        },
                    )
                    .unwrap()
                    .register_sync(
                        test_policy(1..=1, Bpm::from_f64(60.0)..=Bpm::from_f64(60.0)),
                        |caps| MemoCognitionGateStub {
                            updates: caps.memo_updated_inbox(),
                            writer: caps.cognition_writer(),
                        },
                    )
                    .unwrap()
                    .depends_on(speak_id.clone(), query_id)
                    .depends_on(speak_id, cognition_id)
                    .build(&caps)
                    .await
                    .unwrap();
                let mailbox = caps.internal_harness_io().attention_control_mailbox();
                let delivered = mailbox
                    .publish(AttentionControlRequest::new("rule-request"))
                    .await
                    .expect("attention request should wake speak and query-memory");
                assert_eq!(delivered, 2);

                super::run(
                    modules,
                    AgentEventLoopConfig {
                        dependency_idle_timeout: Duration::from_millis(10),
                        ..test_config()
                    },
                    async move {
                        tokio::time::timeout(Duration::from_millis(500), done_rx)
                            .await
                            .expect("speak should activate after two settle waves")
                            .expect("done sender dropped");
                    },
                )
                .await
                .expect("scheduler returned err");

                assert_eq!(
                    observed.borrow().as_slice(),
                    &[vec!["memo-cognition:query-memory".to_owned()]]
                );
            })
            .await;
    }

    #[tokio::test(flavor = "current_thread", start_paused = false)]
    async fn wake_delivered_while_claim_is_in_flight_remains_pending() {
        let local = LocalSet::new();
        local
            .run_until(async {
                let module_id = ModuleId::new(BlockingWakeClaimModule::id()).unwrap();
                let mut alloc = ResourceAllocation::default();
                alloc.set(module_id.clone(), ModuleConfig::default());
                alloc.set_activation(module_id, ActivationRatio::ONE);

                let blackboard = Blackboard::with_allocation(alloc);
                let caps = test_caps(blackboard);
                let (first_drained_tx, first_drained_rx) = oneshot::channel();
                let (first_release_tx, first_release_rx) = oneshot::channel();
                let (done_tx, done_rx) = oneshot::channel();
                let batches = Rc::new(RefCell::new(Vec::<String>::new()));
                let modules = ModuleRegistry::new()
                    .register_sync(
                        test_policy(1..=1, Bpm::from_f64(60.0)..=Bpm::from_f64(60.0)),
                        {
                            let batches = Rc::clone(&batches);
                            let first_drained_tx = Rc::new(RefCell::new(Some(first_drained_tx)));
                            let first_release_rx = Rc::new(RefCell::new(Some(first_release_rx)));
                            let done_tx = Rc::new(RefCell::new(Some(done_tx)));
                            move |caps| BlockingWakeClaimModule {
                                input: caps.attention_control_inbox(),
                                first_drained: first_drained_tx.borrow_mut().take(),
                                first_release: first_release_rx.borrow_mut().take(),
                                batches: Rc::clone(&batches),
                                on_two: done_tx.borrow_mut().take(),
                            }
                        },
                    )
                    .unwrap()
                    .build(&caps)
                    .await
                    .unwrap();
                let mailbox = caps.internal_harness_io().attention_control_mailbox();
                mailbox
                    .publish(AttentionControlRequest::new("first"))
                    .await
                    .expect("first wake should enqueue");

                super::run(modules, test_config(), async move {
                    first_drained_rx
                        .await
                        .expect("first batch should drain first wake");
                    mailbox
                        .publish(AttentionControlRequest::new("second"))
                        .await
                        .expect("second wake should remain pending after first claim");
                    let _ = first_release_tx.send(());
                    tokio::time::timeout(Duration::from_millis(500), done_rx)
                        .await
                        .expect("module should run a second batch for the second wake")
                        .expect("done sender dropped");
                })
                .await
                .expect("scheduler returned err");

                assert_eq!(batches.borrow().as_slice(), &["first", "second"]);
            })
            .await;
    }

    #[tokio::test(flavor = "current_thread", start_paused = false)]
    async fn activation_exhaustion_parks_until_next_activation_opportunity() {
        let local = LocalSet::new();
        local
            .run_until(async {
                let fail_id = ModuleId::new(FailsFiveAttemptsThenSucceeds::id()).unwrap();
                let owner =
                    nuillu_types::ModuleInstanceId::new(fail_id.clone(), ReplicaIndex::ZERO);
                let mut alloc = ResourceAllocation::default();
                alloc.set(fail_id.clone(), ModuleConfig::default());
                alloc.set_activation(fail_id.clone(), ActivationRatio::from_f64(0.5));

                let blackboard = Blackboard::with_allocation(alloc);
                let caps = test_caps_with_real_clock(blackboard.clone());
                let attempts = Rc::new(Cell::new(0_u8));
                let (done_tx, done_rx) = oneshot::channel();
                let done_tx = Rc::new(RefCell::new(Some(done_tx)));
                let modules = ModuleRegistry::new()
                    .register_sync(
                        test_policy(1..=1, Bpm::from_f64(60.0)..=Bpm::from_f64(60.0)),
                        {
                            let attempts = Rc::clone(&attempts);
                            let done_tx = Rc::clone(&done_tx);
                            move |caps| FailsFiveAttemptsThenSucceeds {
                                memo: caps.memo(),
                                attempts: Rc::clone(&attempts),
                                on_done: done_tx.borrow_mut().take(),
                                completed: false,
                            }
                        },
                    )
                    .unwrap()
                    .build(&caps)
                    .await
                    .unwrap();

                let shutdown_blackboard = blackboard.clone();
                let shutdown_owner = owner.clone();
                let shutdown_attempts = Rc::clone(&attempts);
                super::run(
                    modules,
                    AgentEventLoopConfig {
                        idle_threshold: std::time::Duration::from_millis(50),
                        max_activation_attempts: 5,
                        dependency_idle_timeout: std::time::Duration::from_secs(2),
                        dependency_hard_timeout: std::time::Duration::from_secs(10),
                    },
                    async move {
                        loop {
                            let failed = shutdown_blackboard
                                .read(|bb| {
                                    matches!(
                                        bb.module_status_for_instance(&shutdown_owner),
                                        Some(ModuleRunStatus::Failed { phase, message })
                                            if phase == "activate"
                                                && message.contains("activation attempt 5 failed")
                                    )
                                })
                                .await;
                            if failed {
                                break;
                            }
                            tokio::task::yield_now().await;
                            tokio::time::sleep(std::time::Duration::from_millis(1)).await;
                        }
                        assert_eq!(shutdown_attempts.get(), 5);
                        tokio::time::sleep(std::time::Duration::from_millis(30)).await;
                        assert_eq!(
                            shutdown_attempts.get(),
                            5,
                            "failed module retried before a fresh activation opportunity"
                        );

                        let mut raised = ResourceAllocation::default();
                        raised.set(fail_id.clone(), ModuleConfig::default());
                        raised.set_activation(fail_id, ActivationRatio::from_f64(0.75));
                        shutdown_blackboard
                            .apply(BlackboardCommand::SetAllocation(raised))
                            .await;
                        let _ = done_rx.await;
                    },
                )
                .await
                .expect("scheduler should keep running while failed module is parked");

                assert_eq!(attempts.get(), 6);
                let logs = blackboard.read(|bb| bb.recent_memo_logs()).await;
                assert!(logs.iter().any(|record| {
                    record.owner == owner && record.content == "fresh opportunity activated"
                }));
            })
            .await;
    }

    #[tokio::test(flavor = "current_thread", start_paused = false)]
    async fn activation_failure_parks_without_restarting_module() {
        let local = LocalSet::new();
        local
            .run_until(async {
                let module_id = ModuleId::new(FailsFiveAttemptsThenSucceeds::id()).unwrap();
                let owner = ModuleInstanceId::new(module_id.clone(), ReplicaIndex::ZERO);
                let mut alloc = ResourceAllocation::default();
                alloc.set(module_id.clone(), ModuleConfig::default());
                alloc.set_activation(module_id.clone(), ActivationRatio::from_f64(0.5));

                let blackboard = Blackboard::with_allocation(alloc);
                let caps = test_caps_with_real_clock(blackboard.clone());
                let constructions = Rc::new(Cell::new(0_u32));
                let attempts = Rc::new(Cell::new(0_u8));
                let (done_tx, done_rx) = oneshot::channel();
                let done_tx = Rc::new(RefCell::new(Some(done_tx)));
                let modules = ModuleRegistry::new()
                    .register_sync(
                        test_policy(1..=1, Bpm::from_f64(60.0)..=Bpm::from_f64(60.0)),
                        {
                            let constructions = Rc::clone(&constructions);
                            let attempts = Rc::clone(&attempts);
                            let done_tx = Rc::clone(&done_tx);
                            move |caps| {
                                let construction = constructions.get();
                                constructions.set(construction.saturating_add(1));
                                FailsFiveAttemptsThenSucceeds {
                                    memo: caps.memo(),
                                    attempts: Rc::clone(&attempts),
                                    on_done: done_tx.borrow_mut().take(),
                                    completed: false,
                                }
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
                        max_activation_attempts: 5,
                        dependency_idle_timeout: std::time::Duration::from_secs(2),
                        dependency_hard_timeout: std::time::Duration::from_secs(10),
                    },
                    async move {
                        wait_for_status(
                            &blackboard,
                            &owner,
                            ModuleRunStatus::Failed {
                                phase: "activate".to_string(),
                                message: "activation attempt 5 failed".to_string(),
                            },
                        )
                        .await;
                        let mut raised = ResourceAllocation::default();
                        raised.set(module_id.clone(), ModuleConfig::default());
                        raised.set_activation(module_id, ActivationRatio::from_f64(0.75));
                        blackboard
                            .apply(BlackboardCommand::SetAllocation(raised))
                            .await;
                        let _ = done_rx.await;
                    },
                )
                .await
                .expect("scheduler should continue through parked activation failure");

                assert_eq!(constructions.get(), 1);
                assert_eq!(attempts.get(), 6);
            })
            .await;
    }

    #[tokio::test(flavor = "current_thread", start_paused = false)]
    async fn next_batch_failure_parks_module_and_then_succeeds() {
        let local = LocalSet::new();
        local
            .run_until(async {
                let module_id = ModuleId::new(FailsFirstBatchThenSucceeds::id()).unwrap();
                let owner = ModuleInstanceId::new(module_id.clone(), ReplicaIndex::ZERO);
                let mut alloc = ResourceAllocation::default();
                alloc.set(module_id.clone(), ModuleConfig::default());
                alloc.set_activation(module_id.clone(), ActivationRatio::from_f64(0.5));

                let blackboard = Blackboard::with_allocation(alloc);
                let caps = test_caps_with_real_clock(blackboard.clone());
                let constructions = Rc::new(Cell::new(0_u32));
                let (done_tx, done_rx) = oneshot::channel();
                let done_tx = Rc::new(RefCell::new(Some(done_tx)));
                let modules = ModuleRegistry::new()
                    .register_sync(
                        test_policy(1..=1, Bpm::from_f64(120.0)..=Bpm::from_f64(120.0)),
                        {
                            let constructions = Rc::clone(&constructions);
                            let done_tx = Rc::clone(&done_tx);
                            move |_| {
                                let construction = constructions.get();
                                constructions.set(construction.saturating_add(1));
                                FailsFirstBatchThenSucceeds {
                                    on_done: done_tx.borrow_mut().take(),
                                    failed_once: false,
                                    completed: false,
                                }
                            }
                        },
                    )
                    .unwrap()
                    .build(&caps)
                    .await
                    .unwrap();

                let shutdown_blackboard = blackboard.clone();
                let shutdown_owner = owner.clone();
                let shutdown_module_id = module_id.clone();
                super::run(
                    modules,
                    AgentEventLoopConfig {
                        idle_threshold: std::time::Duration::from_millis(50),
                        max_activation_attempts: 1,
                        dependency_idle_timeout: std::time::Duration::from_secs(2),
                        dependency_hard_timeout: std::time::Duration::from_secs(10),
                    },
                    async move {
                        wait_for_status(
                            &shutdown_blackboard,
                            &shutdown_owner,
                            ModuleRunStatus::Failed {
                                phase: "next_batch".to_string(),
                                message: "first next_batch fails".to_string(),
                            },
                        )
                        .await;
                        let mut raised = ResourceAllocation::default();
                        raised.set(shutdown_module_id.clone(), ModuleConfig::default());
                        raised.set_activation(shutdown_module_id, ActivationRatio::from_f64(0.75));
                        shutdown_blackboard
                            .apply(BlackboardCommand::SetAllocation(raised))
                            .await;
                        let _ = done_rx.await;
                    },
                )
                .await
                .expect("scheduler should continue after parked next_batch failure");

                assert_eq!(constructions.get(), 1);
            })
            .await;
    }

    #[tokio::test(flavor = "current_thread", start_paused = false)]
    async fn failed_dependency_does_not_block_dependent_activation() {
        let local = LocalSet::new();
        local
            .run_until(async {
                let dependency_id = ModuleId::new(StopAfterOneFailureDependency::id()).unwrap();
                let dependent_id = ModuleId::new(EchoModule::id()).unwrap();
                let dependency_owner =
                    ModuleInstanceId::new(dependency_id.clone(), ReplicaIndex::ZERO);
                let mut alloc = ResourceAllocation::default();
                for module in [dependency_id.clone(), dependent_id.clone()] {
                    alloc.set(module.clone(), ModuleConfig::default());
                    alloc.set_activation(module, ActivationRatio::ONE);
                }

                let blackboard = Blackboard::with_allocation(alloc);
                let caps = test_caps_with_real_clock(blackboard.clone());
                let (done_tx, done_rx) = oneshot::channel();
                let done_tx = Rc::new(RefCell::new(Some(done_tx)));
                let modules = ModuleRegistry::new()
                    .register_sync(
                        test_policy(1..=1, Bpm::from_f64(60.0)..=Bpm::from_f64(60.0)),
                        |_| StopAfterOneFailureDependency { batch_sent: false },
                    )
                    .unwrap()
                    .register_sync(
                        test_policy(1..=1, Bpm::from_f64(60.0)..=Bpm::from_f64(60.0)),
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
                    .depends_on(dependent_id, dependency_id)
                    .build(&caps)
                    .await
                    .unwrap();
                let mailbox = caps.internal_harness_io().attention_control_mailbox();

                let shutdown_blackboard = blackboard.clone();
                super::run(
                    modules,
                    AgentEventLoopConfig {
                        idle_threshold: std::time::Duration::from_millis(50),
                        max_activation_attempts: 1,
                        dependency_idle_timeout: std::time::Duration::from_secs(2),
                        dependency_hard_timeout: std::time::Duration::from_secs(10),
                    },
                    async move {
                        loop {
                            let failed = shutdown_blackboard
                                .read(|bb| {
                                    matches!(
                                        bb.module_status_for_instance(&dependency_owner),
                                        Some(ModuleRunStatus::Failed { phase, message })
                                            if phase == "activate"
                                                && message.contains(
                                                    "dependency stops on first failure"
                                                )
                                    )
                                })
                                .await;
                            if failed {
                                break;
                            }
                            tokio::task::yield_now().await;
                            tokio::time::sleep(std::time::Duration::from_millis(1)).await;
                        }
                        mailbox
                            .publish(AttentionControlRequest::new("wake-dependent"))
                            .await
                            .expect("attention request should wake dependent");
                        let _ = done_rx.await;
                    },
                )
                .await
                .expect("failed dependency should not block dependent activation");
            })
            .await;
    }

    // Inactive replicas keep state and queued inbox messages, but the scheduler
    // does not start semantic work for them until allocation makes them active.

    #[tokio::test(flavor = "current_thread", start_paused = false)]
    async fn allocation_change_alone_does_not_create_module_batch() {
        let local = LocalSet::new();
        local
            .run_until(async {
                let module_id = echo_id();
                let mut alloc = ResourceAllocation::default();
                alloc.set(module_id.clone(), ModuleConfig::default());
                alloc.set_activation(module_id.clone(), ActivationRatio::ONE);

                let blackboard = Blackboard::with_allocation(alloc);
                let caps = test_caps(blackboard.clone());
                let (done_tx, mut done_rx) = oneshot::channel();
                let done_tx = Rc::new(RefCell::new(Some(done_tx)));
                let modules = ModuleRegistry::new()
                    .register_sync(
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

                super::run(modules, test_config(), async {
                    for _ in 0..4 {
                        tokio::task::yield_now().await;
                    }

                    let mut changed = ResourceAllocation::default();
                    changed.set(
                        module_id.clone(),
                        ModuleConfig {
                            guidance: "new durable context".into(),
                        },
                    );
                    changed.set_activation(module_id.clone(), ActivationRatio::ONE);
                    blackboard
                        .apply(BlackboardCommand::SetAllocation(changed))
                        .await;
                    tokio::time::sleep(Duration::from_millis(30)).await;
                    assert_oneshot_pending(
                        &mut done_rx,
                        "allocation changes should not synthesize module work",
                    );
                })
                .await
                .expect("scheduler returned err");

                let memos = blackboard.read(|bb| bb.recent_memo_logs()).await;
                assert!(memos.is_empty());
            })
            .await;
    }

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
                    .register_sync(
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
                        .publish(AttentionControlRequest::new("later"))
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

    #[tokio::test(flavor = "current_thread", start_paused = false)]
    async fn self_wake_activates_inactive_owner_once() {
        let local = LocalSet::new();
        local
            .run_until(async {
                let module_id = ModuleId::new(ImmediateDependentModule::id()).unwrap();
                let mut alloc = ResourceAllocation::default();
                alloc.set(module_id.clone(), ModuleConfig::default());
                alloc.set_activation(module_id, ActivationRatio::ZERO);

                let caps = test_caps(Blackboard::with_allocation(alloc));
                let (done_tx, done_rx) = oneshot::channel();
                let done_tx = Rc::new(RefCell::new(Some(done_tx)));
                let self_wake = Rc::new(RefCell::new(None::<SelfWake>));
                let modules = ModuleRegistry::new()
                    .register_sync(
                        test_policy(0..=1, Bpm::from_f64(60.0)..=Bpm::from_f64(60.0)),
                        {
                            let done_tx = Rc::clone(&done_tx);
                            let self_wake = Rc::clone(&self_wake);
                            move |caps| {
                                *self_wake.borrow_mut() = Some(caps.self_wake());
                                ImmediateDependentModule {
                                    batch_sent: false,
                                    on_done: done_tx.borrow_mut().take(),
                                }
                            }
                        },
                    )
                    .unwrap()
                    .build(&caps)
                    .await
                    .unwrap();

                let wake = self_wake
                    .borrow()
                    .as_ref()
                    .expect("module constructor should expose self wake")
                    .clone();
                wake.wake();

                super::run(modules, test_config(), async move {
                    tokio::time::timeout(Duration::from_millis(500), done_rx)
                        .await
                        .expect("self-wake should activate the inactive owner")
                        .expect("done sender dropped");
                })
                .await
                .expect("scheduler returned err");
            })
            .await;
    }

    #[tokio::test(flavor = "current_thread", start_paused = false)]
    async fn self_wake_does_not_override_hard_disabled_owner() {
        let local = LocalSet::new();
        local
            .run_until(async {
                let module_id = ModuleId::new(ImmediateDependentModule::id()).unwrap();
                let mut alloc = ResourceAllocation::default();
                alloc.set(module_id.clone(), ModuleConfig::default());
                alloc.set_activation(module_id.clone(), ActivationRatio::ZERO);
                let blackboard = Blackboard::with_allocation(alloc);
                blackboard
                    .apply(BlackboardCommand::SetModuleForcedDisabled {
                        module: module_id.clone(),
                        disabled: true,
                    })
                    .await;

                let caps = test_caps(blackboard);
                let (done_tx, mut done_rx) = oneshot::channel();
                let done_tx = Rc::new(RefCell::new(Some(done_tx)));
                let self_wake = Rc::new(RefCell::new(None::<SelfWake>));
                let modules = ModuleRegistry::new()
                    .register_sync(
                        test_policy(0..=1, Bpm::from_f64(60.0)..=Bpm::from_f64(60.0)),
                        {
                            let done_tx = Rc::clone(&done_tx);
                            let self_wake = Rc::clone(&self_wake);
                            move |caps| {
                                *self_wake.borrow_mut() = Some(caps.self_wake());
                                ImmediateDependentModule {
                                    batch_sent: false,
                                    on_done: done_tx.borrow_mut().take(),
                                }
                            }
                        },
                    )
                    .unwrap()
                    .build(&caps)
                    .await
                    .unwrap();

                let wake = self_wake
                    .borrow()
                    .as_ref()
                    .expect("module constructor should expose self wake")
                    .clone();
                wake.wake();

                super::run(modules, test_config(), async move {
                    for _ in 0..8 {
                        tokio::task::yield_now().await;
                    }
                    assert_oneshot_pending(
                        &mut done_rx,
                        "hard-disabled self-wake should not activate the owner",
                    );
                })
                .await
                .expect("scheduler returned err");
            })
            .await;
    }

    #[tokio::test(flavor = "current_thread", start_paused = true)]
    async fn dependent_wake_does_not_wait_for_dependency_next_batch_without_dependency_wake() {
        let local = LocalSet::new();
        local
            .run_until(async {
                let dependency_id = ModuleId::new(PendingDependencyModule::id()).unwrap();
                let dependent_id = ModuleId::new(EchoModule::id()).unwrap();
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
                let (dependent_done_tx, mut dependent_done_rx) = oneshot::channel();
                let dependent_done_tx = Rc::new(RefCell::new(Some(dependent_done_tx)));

                let modules = ModuleRegistry::new()
                    .register_sync(
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
                    .register_sync(
                        test_policy(1..=1, Bpm::from_f64(60.0)..=Bpm::from_f64(60.0)),
                        {
                            let dependent_done_tx = Rc::clone(&dependent_done_tx);
                            move |caps| EchoModule {
                                attention_control_inbox: caps.attention_control_inbox(),
                                memo: caps.memo(),
                                on_done: dependent_done_tx.borrow_mut().take(),
                            }
                        },
                    )
                    .unwrap()
                    .depends_on(dependent_id, dependency_id)
                    .build(&caps)
                    .await
                    .unwrap();
                let mailbox = caps.internal_harness_io().attention_control_mailbox();

                super::run(modules, test_config(), async move {
                    mailbox
                        .publish(AttentionControlRequest::new("dependent-only"))
                        .await
                        .expect("attention request should wake dependent");
                    for _ in 0..4 {
                        tokio::task::yield_now().await;
                    }
                    let mut dependent_finished = false;
                    for _ in 0..20 {
                        match dependent_done_rx.try_recv() {
                            Ok(()) => {
                                dependent_finished = true;
                                break;
                            }
                            Err(tokio::sync::oneshot::error::TryRecvError::Empty) => {
                                tokio::task::yield_now().await;
                            }
                            Err(tokio::sync::oneshot::error::TryRecvError::Closed) => {
                                panic!("dependent activation sender dropped");
                            }
                        }
                    }
                    assert!(
                        dependent_finished,
                        "dependent should activate without waiting for dependency next_batch"
                    );
                    assert_eq!(
                        observed_dependency_activations.get(),
                        0,
                        "dependency should still be blocked in next_batch"
                    );
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
    async fn dependent_wake_skips_active_awaiting_dependencies_without_wakes() {
        let local = LocalSet::new();
        local
            .run_until(async {
                let dep_a_id = ModuleId::new(SilentDependencyA::id()).unwrap();
                let dep_b_id = ModuleId::new(SilentDependencyB::id()).unwrap();
                let dep_c_id = ModuleId::new(SilentDependencyC::id()).unwrap();
                let dependent_id = ModuleId::new(EchoModule::id()).unwrap();
                let mut alloc = ResourceAllocation::default();
                for id in [&dep_a_id, &dep_b_id, &dep_c_id, &dependent_id] {
                    alloc.set(id.clone(), ModuleConfig::default());
                    alloc.set_activation(id.clone(), ActivationRatio::ONE);
                }

                let blackboard = Blackboard::with_allocation(alloc);
                let caps = test_caps(blackboard);
                let (dependent_done_tx, dependent_done_rx) = oneshot::channel();
                let dependent_done_tx = Rc::new(RefCell::new(Some(dependent_done_tx)));
                let modules = ModuleRegistry::new()
                    .register_sync(
                        test_policy(1..=1, Bpm::from_f64(60.0)..=Bpm::from_f64(60.0)),
                        |_| SilentDependencyA,
                    )
                    .unwrap()
                    .register_sync(
                        test_policy(1..=1, Bpm::from_f64(60.0)..=Bpm::from_f64(60.0)),
                        |_| SilentDependencyB,
                    )
                    .unwrap()
                    .register_sync(
                        test_policy(1..=1, Bpm::from_f64(60.0)..=Bpm::from_f64(60.0)),
                        |_| SilentDependencyC,
                    )
                    .unwrap()
                    .register_sync(
                        test_policy(1..=1, Bpm::from_f64(60.0)..=Bpm::from_f64(60.0)),
                        {
                            let dependent_done_tx = Rc::clone(&dependent_done_tx);
                            move |caps| EchoModule {
                                attention_control_inbox: caps.attention_control_inbox(),
                                memo: caps.memo(),
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
                let mailbox = caps.internal_harness_io().attention_control_mailbox();

                super::run(modules, test_config(), async move {
                    mailbox
                        .publish(AttentionControlRequest::new("wake-dependent"))
                        .await
                        .expect("attention request should wake dependent");
                    tokio::time::timeout(Duration::from_millis(100), dependent_done_rx)
                        .await
                        .expect("dependent wake should not wait for idle dependencies")
                        .expect("dependent activation sender dropped");
                })
                .await
                .expect("scheduler returned err");
            })
            .await;
    }

    #[tokio::test(flavor = "current_thread", start_paused = false)]
    async fn stored_dependency_targets_are_skipped_by_state_based_filter() {
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
                    .register_sync(
                        test_policy(1..=1, Bpm::from_f64(60.0)..=Bpm::from_f64(60.0)),
                        |_| SilentDependencyA,
                    )
                    .unwrap()
                    .register_sync(
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
                let (runtime, modules, dependencies) = modules.into_parts_with_dependencies();
                let mut zero_windows =
                    super::ZeroReplicaWindows::new(runtime.zero_replica_window_policies().await);
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
                let mut kick_inboxes = Vec::with_capacity(owners.len());
                for (index, module) in modules.into_iter().enumerate() {
                    let (kick_inbox, kick_handle) = crate::kicks::KickInbox::new();
                    kick_handles.push(kick_handle);
                    if index == dependency_index {
                        kick_inboxes.push(Some(kick_inbox));
                        states.push(super::ModuleState::Stored {
                            module,
                            next_batch_throttle: None,
                        });
                    } else {
                        kick_inboxes.push(Some(kick_inbox));
                        states.push(super::ModuleState::Awaiting);
                    }
                }

                let mut target_indexes_by_role = HashMap::new();
                target_indexes_by_role.insert(dependency_id, vec![dependency_index]);
                let dependency_targets = super::DependencyTargets {
                    dependencies: Arc::new(dependencies),
                    target_indexes_by_role: Arc::new(target_indexes_by_role),
                };
                let mut tasks = futures::stream::FuturesUnordered::new();
                let subscriber = tracing::dispatcher::get_default(Clone::clone);
                let parent = tracing::Span::current();
                let completions = super::collect_dependency_settle_completions(
                    &runtime,
                    &mut tasks,
                    dependent_index,
                    &owners[dependent_index],
                    &mut states,
                    &mut kick_inboxes,
                    &kick_handles,
                    &dependency_targets,
                    &owners,
                    &mut zero_windows,
                    test_config(),
                    &parent,
                    &subscriber,
                )
                .await;

                assert!(
                    completions.is_empty(),
                    "stored dependency target should reproduce the zero-completion fast path"
                );
            })
            .await;
    }

    #[tokio::test(flavor = "current_thread", start_paused = false)]
    async fn inactive_dependency_targets_wait_for_idle_timeout_without_state_kick() {
        let local = LocalSet::new();
        local
            .run_until(async {
                let dependency_id = ModuleId::new(SilentDependencyA::id()).unwrap();
                let dependent_id = ModuleId::new(ImmediateDependentModule::id()).unwrap();
                let mut alloc = ResourceAllocation::default();
                alloc.set(dependency_id.clone(), ModuleConfig::default());
                alloc.set_activation(dependency_id.clone(), ActivationRatio::ZERO);
                alloc.set(dependent_id.clone(), ModuleConfig::default());
                alloc.set_activation(dependent_id.clone(), ActivationRatio::ONE);

                let caps = test_caps(Blackboard::with_allocation(alloc));
                let modules = ModuleRegistry::new()
                    .register_sync(
                        test_policy(0..=1, Bpm::from_f64(60.0)..=Bpm::from_f64(60.0)),
                        |_| SilentDependencyA,
                    )
                    .unwrap()
                    .register_sync(
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
                let (runtime, modules, dependencies) = modules.into_parts_with_dependencies();
                let mut zero_windows =
                    super::ZeroReplicaWindows::new(runtime.zero_replica_window_policies().await);
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
                assert!(
                    !runtime.is_active(&owners[dependency_index]).await,
                    "test setup must keep the dependency outside the active scheduling set"
                );

                let mut states = Vec::with_capacity(owners.len());
                let mut kick_handles = Vec::with_capacity(owners.len());
                let mut kick_inboxes = Vec::with_capacity(owners.len());
                for _owner in &owners {
                    let (kick_inbox, kick_handle) = crate::kicks::KickInbox::new();
                    kick_handles.push(kick_handle);
                    kick_inboxes.push(Some(kick_inbox));
                    states.push(super::ModuleState::Awaiting);
                }

                let mut target_indexes_by_role = HashMap::new();
                target_indexes_by_role.insert(dependency_id, vec![dependency_index]);
                let dependency_targets = super::DependencyTargets {
                    dependencies: Arc::new(dependencies),
                    target_indexes_by_role: Arc::new(target_indexes_by_role),
                };
                let mut tasks = futures::stream::FuturesUnordered::new();
                let subscriber = tracing::dispatcher::get_default(Clone::clone);
                let parent = tracing::Span::current();
                let completions = super::collect_dependency_settle_completions(
                    &runtime,
                    &mut tasks,
                    dependent_index,
                    &owners[dependent_index],
                    &mut states,
                    &mut kick_inboxes,
                    &kick_handles,
                    &dependency_targets,
                    &owners,
                    &mut zero_windows,
                    test_config(),
                    &parent,
                    &subscriber,
                )
                .await;

                assert!(
                    completions.is_empty(),
                    "inactive dependency with no pending wake should not delay dependent"
                );
            })
            .await;
    }

    #[tokio::test(flavor = "current_thread", start_paused = true)]
    async fn inactive_dependency_with_pending_wake_uses_idle_timeout_before_dependent_batch() {
        let local = LocalSet::new();
        local
            .run_until(async {
                let dependency_id = ModuleId::new(ControllerTickModule::id()).unwrap();
                let dependent_id = ModuleId::new(EchoModule::id()).unwrap();
                let mut alloc = ResourceAllocation::default();
                alloc.set(dependency_id.clone(), ModuleConfig::default());
                alloc.set_activation(dependency_id.clone(), ActivationRatio::ZERO);
                alloc.set(dependent_id.clone(), ModuleConfig::default());
                alloc.set_activation(dependent_id.clone(), ActivationRatio::ONE);

                let blackboard = Blackboard::with_allocation(alloc);
                let caps = test_caps(blackboard);
                let dependency_activations = Rc::new(Cell::new(0_u32));
                let (dependent_done_tx, mut dependent_done_rx) = oneshot::channel();
                let dependent_done_tx = Rc::new(RefCell::new(Some(dependent_done_tx)));

                let modules = ModuleRegistry::new()
                    .register_sync(
                        test_policy(0..=1, Bpm::from_f64(60.0)..=Bpm::from_f64(60.0)),
                        {
                            let dependency_activations = Rc::clone(&dependency_activations);
                            move |caps| ControllerTickModule {
                                attention_control_inbox: caps.attention_control_inbox(),
                                activations: Rc::clone(&dependency_activations),
                            }
                        },
                    )
                    .unwrap()
                    .register_sync(
                        test_policy(1..=1, Bpm::from_f64(60.0)..=Bpm::from_f64(60.0)),
                        {
                            let dependent_done_tx = Rc::clone(&dependent_done_tx);
                            move |caps| EchoModule {
                                attention_control_inbox: caps.attention_control_inbox(),
                                memo: caps.memo(),
                                on_done: dependent_done_tx.borrow_mut().take(),
                            }
                        },
                    )
                    .unwrap()
                    .depends_on(dependent_id.clone(), dependency_id.clone())
                    .build(&caps)
                    .await
                    .unwrap();
                let mailbox = caps.internal_harness_io().attention_control_mailbox();
                let delivered = mailbox
                    .publish(AttentionControlRequest::new("wake-both"))
                    .await
                    .expect("attention request should wake inactive dependency fallback");
                assert_eq!(delivered, 2);

                super::run(
                    modules,
                    AgentEventLoopConfig {
                        dependency_idle_timeout: Duration::from_millis(10),
                        ..test_config()
                    },
                    async move {
                        for _ in 0..4 {
                            tokio::task::yield_now().await;
                        }
                        assert_oneshot_pending(
                            &mut dependent_done_rx,
                            "dependent should wait briefly for inactive dependency with a wake",
                        );

                        tokio::time::advance(Duration::from_millis(9)).await;
                        for _ in 0..4 {
                            tokio::task::yield_now().await;
                        }
                        assert_oneshot_pending(
                            &mut dependent_done_rx,
                            "inactive dependency wait should use the configured idle timeout",
                        );

                        for _ in 0..100 {
                            tokio::time::advance(Duration::from_millis(1)).await;
                            tokio::task::yield_now().await;
                            match dependent_done_rx.try_recv() {
                                Ok(()) => return,
                                Err(tokio::sync::oneshot::error::TryRecvError::Empty) => {
                                    continue;
                                }
                                Err(tokio::sync::oneshot::error::TryRecvError::Closed) => {
                                    panic!("dependent activation sender dropped");
                                }
                            }
                        }
                        panic!("dependent should activate after inactive dependency idle timeout");
                    },
                )
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
                let dependent_id = ModuleId::new(EchoModule::id()).unwrap();
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
                let (dependent_done_tx, dependent_done_rx) = oneshot::channel();
                let dependent_done_tx = Rc::new(RefCell::new(Some(dependent_done_tx)));

                let modules = ModuleRegistry::new()
                    .register_sync(
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
                    .register_sync(
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
                    .register_sync(
                        test_policy(1..=1, Bpm::from_f64(60.0)..=Bpm::from_f64(60.0)),
                        {
                            let dependent_done_tx = Rc::clone(&dependent_done_tx);
                            move |caps| EchoModule {
                                attention_control_inbox: caps.attention_control_inbox(),
                                memo: caps.memo(),
                                on_done: dependent_done_tx.borrow_mut().take(),
                            }
                        },
                    )
                    .unwrap()
                    .depends_on(dependent_id.clone(), target_id.clone())
                    .build(&caps)
                    .await
                    .unwrap();
                let mailbox = caps.internal_harness_io().attention_control_mailbox();

                super::run(modules, test_config(), async move {
                    let _ = gate_seen_rx.await;
                    mailbox
                        .publish(AttentionControlRequest::new("dependent-wake"))
                        .await
                        .expect("attention request should wake dependent");
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
                        ModuleId::new(EchoModule::id()).unwrap(),
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
                    .register_sync(
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
                let (runtime, mut modules, _) = modules.into_parts_with_dependencies();
                let module = modules.pop().unwrap();
                let sender = module.owner().clone();
                let (kick_inbox, kick_handle) = crate::kicks::KickInbox::new();
                let completion = kick_handle.send(sender.clone());
                let mut tasks = futures::stream::FuturesUnordered::new();
                let subscriber = tracing::dispatcher::get_default(Clone::clone);
                let parent = tracing::Span::current();

                super::spawn_next_batch(
                    runtime,
                    &mut tasks,
                    0,
                    sender.clone(),
                    module,
                    kick_inbox,
                    Vec::new(),
                    None,
                    Duration::from_secs(2),
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
    async fn next_batch_error_consumes_self_wake_permit() {
        use futures::StreamExt as _;

        let local = LocalSet::new();
        local
            .run_until(async {
                let module_id = ModuleId::new(FailingBatchModule::id()).unwrap();
                let mut alloc = ResourceAllocation::default();
                alloc.set(module_id.clone(), ModuleConfig::default());
                alloc.set_activation(module_id, ActivationRatio::ONE);

                let (release_tx, release_rx) = oneshot::channel();
                let release_rx = Rc::new(RefCell::new(Some(release_rx)));
                let self_wake = Rc::new(RefCell::new(None::<SelfWake>));
                let caps = test_caps(Blackboard::with_allocation(alloc));
                let modules = ModuleRegistry::new()
                    .register_sync(
                        test_policy(1..=1, Bpm::from_f64(60.0)..=Bpm::from_f64(60.0)),
                        {
                            let release_rx = Rc::clone(&release_rx);
                            let self_wake = Rc::clone(&self_wake);
                            move |caps| {
                                *self_wake.borrow_mut() = Some(caps.self_wake());
                                FailingBatchModule {
                                    release: release_rx.borrow_mut().take(),
                                }
                            }
                        },
                    )
                    .unwrap()
                    .build(&caps)
                    .await
                    .unwrap();
                let (runtime, mut modules, _) = modules.into_parts_with_dependencies();
                let module = modules.pop().unwrap();
                let owner = module.owner().clone();
                let wake = self_wake
                    .borrow()
                    .as_ref()
                    .expect("module constructor should expose self wake")
                    .clone();
                wake.wake();
                assert!(runtime.has_pending_self_wake_permit(&owner));

                let (kick_inbox, _) = crate::kicks::KickInbox::new();
                let mut tasks = futures::stream::FuturesUnordered::new();
                let subscriber = tracing::dispatcher::get_default(Clone::clone);
                let parent = tracing::Span::current();
                super::spawn_next_batch(
                    runtime.clone(),
                    &mut tasks,
                    0,
                    owner.clone(),
                    module,
                    kick_inbox,
                    Vec::new(),
                    None,
                    Duration::from_secs(2),
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

                let owners = vec![owner.clone()];
                let mut states = vec![super::ModuleState::Awaiting];
                let mut kick_inboxes = vec![None];
                let mut zero_window_wakers = vec![None];
                let dependency_targets = super::DependencyTargets {
                    dependencies: Arc::new(ModuleDependencies::default()),
                    target_indexes_by_role: Arc::new(HashMap::new()),
                };
                let mut zero_windows = super::ZeroReplicaWindows::new(HashMap::new());
                let mut consecutive_failures = vec![0];
                super::handle_task_message(
                    message,
                    &runtime,
                    &owners,
                    &mut states,
                    &mut tasks,
                    &mut kick_inboxes,
                    &mut zero_window_wakers,
                    &[],
                    &dependency_targets,
                    &HashMap::new(),
                    &mut zero_windows,
                    &mut consecutive_failures,
                    true,
                    test_config(),
                    &parent,
                    &subscriber,
                )
                .await
                .expect("next_batch failure should be handled");

                assert!(!runtime.has_pending_self_wake_permit(&owner));
                assert!(matches!(
                    &states[0],
                    super::ModuleState::FailedUntilActivation { phase, message, .. }
                        if phase == "next_batch" && message.contains("planned next_batch failure")
                ));
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
                    .register_sync(
                        test_policy(1..=1, Bpm::from_f64(60.0)..=Bpm::from_f64(60.0)),
                        |_| PendingDependencyModule {
                            release: None,
                            activations: Rc::new(Cell::new(0)),
                            on_done: None,
                        },
                    )
                    .unwrap()
                    .register_sync(
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
                let (runtime, mut modules, dependencies) = modules.into_parts_with_dependencies();
                let mut zero_windows =
                    super::ZeroReplicaWindows::new(runtime.zero_replica_window_policies().await);
                let owners = modules
                    .iter()
                    .map(|module| module.owner().clone())
                    .collect::<Vec<_>>();
                let dependent_index = modules
                    .iter()
                    .position(|module| module.owner().module == dependent_id)
                    .unwrap();
                let dependency_index = modules
                    .iter()
                    .position(|module| module.owner().module == dependency_id)
                    .unwrap();
                let dependent = modules.remove(dependent_index);
                let dependent_owner = dependent.owner().clone();
                let dependency_owner = nuillu_types::ModuleInstanceId::new(
                    dependency_id,
                    nuillu_types::ReplicaIndex::ZERO,
                );
                let mut dependent_kick_inbox = None;
                let mut dependent_kick_handle = None;
                let mut dependency_kick_inbox = None;
                let mut kick_handles = Vec::with_capacity(owners.len());
                let mut kick_inboxes = Vec::with_capacity(owners.len());
                for index in 0..owners.len() {
                    let (kick_inbox, kick_handle) = crate::kicks::KickInbox::new();
                    if index == dependent_index {
                        dependent_kick_inbox = Some(kick_inbox);
                        dependent_kick_handle = Some(kick_handle.clone());
                        kick_inboxes.push(None);
                    } else if index == dependency_index {
                        dependency_kick_inbox = Some(kick_inbox);
                        kick_inboxes.push(None);
                    } else {
                        kick_inboxes.push(Some(kick_inbox));
                    }
                    kick_handles.push(kick_handle);
                }
                let mut target_indexes_by_role = HashMap::new();
                target_indexes_by_role
                    .insert(dependency_owner.module.clone(), vec![dependency_index]);
                let dependency_targets = super::DependencyTargets {
                    dependencies: Arc::new(dependencies),
                    target_indexes_by_role: Arc::new(target_indexes_by_role),
                };
                let mut states = (0..owners.len())
                    .map(|_| super::ModuleState::Awaiting)
                    .collect::<Vec<_>>();
                states[dependency_index] = super::ModuleState::Activating;
                let mut tasks = futures::stream::FuturesUnordered::new();
                let subscriber = tracing::dispatcher::get_default(Clone::clone);
                let parent = tracing::Span::current();
                let completions = super::collect_dependency_settle_completions(
                    &runtime,
                    &mut tasks,
                    dependent_index,
                    &dependent_owner,
                    &mut states,
                    &mut kick_inboxes,
                    &kick_handles,
                    &dependency_targets,
                    &owners,
                    &mut zero_windows,
                    test_config(),
                    &parent,
                    &subscriber,
                )
                .await;
                assert_eq!(completions.len(), 1);

                super::spawn_dependency_settle_wait(
                    &mut tasks,
                    dependent_index,
                    dependent_owner.clone(),
                    dependent,
                    dependent_kick_inbox.unwrap(),
                    Vec::new(),
                    None,
                    completions,
                    0,
                    Duration::from_secs(10),
                    &parent,
                    &subscriber,
                );
                let dependency_kick = dependency_kick_inbox
                    .as_mut()
                    .unwrap()
                    .next()
                    .await
                    .expect("dependent should kick its dependency");
                let dependent_completion = dependent_kick_handle.unwrap().send(dependency_owner);
                dependency_kick.notify_finish();

                let message = tasks
                    .next()
                    .await
                    .expect("dependency flush task should finish")
                    .expect("dependency flush task should not panic");
                match message {
                    super::TaskMessage::DependencySettled { pending_kicks, .. } => {
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
                    .register_sync(
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
                let module = modules.pop().unwrap();
                let owner = module.owner().clone();
                let (kick_inbox, _) = crate::kicks::KickInbox::new();
                let (kick, completion) = crate::kicks::Kick::new(owner.clone());
                drop(kick);

                let mut tasks = futures::stream::FuturesUnordered::new();
                let subscriber = tracing::dispatcher::get_default(Clone::clone);
                let parent = tracing::Span::current();
                super::spawn_dependency_settle_wait(
                    &mut tasks,
                    0,
                    owner.clone(),
                    module,
                    kick_inbox,
                    Vec::new(),
                    None,
                    vec![super::DependencyWait::kick(owner, completion)],
                    0,
                    Duration::from_secs(10),
                    &parent,
                    &subscriber,
                );

                let message = tasks
                    .next()
                    .await
                    .expect("dependency flush task should finish")
                    .expect("dependency flush task should not panic");
                match message {
                    super::TaskMessage::DependencySettled { pending_kicks, .. } => {
                        assert!(pending_kicks.is_empty());
                    }
                    _ => panic!("expected dependency flush task message"),
                }
            })
            .await;
    }

    #[tokio::test(flavor = "current_thread", start_paused = true)]
    async fn throttled_dependency_waits_for_hard_timeout_not_idle_timeout() {
        use futures::{FutureExt as _, StreamExt as _};

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
                    .register_sync(
                        test_policy(1..=1, Bpm::from_f64(60.0)..=Bpm::from_f64(60.0)),
                        |_| SilentDependencyA,
                    )
                    .unwrap()
                    .register_sync(
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
                let (runtime, mut modules, dependencies) = modules.into_parts_with_dependencies();
                let mut zero_windows =
                    super::ZeroReplicaWindows::new(runtime.zero_replica_window_policies().await);
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
                let dependent = modules.remove(dependent_index);
                let dependent_owner = dependent.owner().clone();

                let mut dependent_kick_inbox = None;
                let mut _dependency_kick_inbox = None;
                let mut kick_handles = Vec::with_capacity(owners.len());
                let mut kick_inboxes = Vec::with_capacity(owners.len());
                for index in 0..owners.len() {
                    let (kick_inbox, kick_handle) = crate::kicks::KickInbox::new();
                    if index == dependent_index {
                        dependent_kick_inbox = Some(kick_inbox);
                        kick_inboxes.push(None);
                    } else if index == dependency_index {
                        _dependency_kick_inbox = Some(kick_inbox);
                        kick_inboxes.push(None);
                    } else {
                        kick_inboxes.push(Some(kick_inbox));
                    }
                    kick_handles.push(kick_handle);
                }
                let mut target_indexes_by_role = HashMap::new();
                target_indexes_by_role.insert(dependency_id, vec![dependency_index]);
                let dependency_targets = super::DependencyTargets {
                    dependencies: Arc::new(dependencies),
                    target_indexes_by_role: Arc::new(target_indexes_by_role),
                };
                let mut states = (0..owners.len())
                    .map(|_| super::ModuleState::Awaiting)
                    .collect::<Vec<_>>();
                states[dependency_index] = super::ModuleState::Throttling;
                let mut tasks = futures::stream::FuturesUnordered::new();
                let subscriber = tracing::dispatcher::get_default(Clone::clone);
                let parent = tracing::Span::current();
                let completions = super::collect_dependency_settle_completions(
                    &runtime,
                    &mut tasks,
                    dependent_index,
                    &dependent_owner,
                    &mut states,
                    &mut kick_inboxes,
                    &kick_handles,
                    &dependency_targets,
                    &owners,
                    &mut zero_windows,
                    test_config(),
                    &parent,
                    &subscriber,
                )
                .await;
                assert_eq!(completions.len(), 1);

                super::spawn_dependency_settle_wait(
                    &mut tasks,
                    dependent_index,
                    dependent_owner,
                    dependent,
                    dependent_kick_inbox.unwrap(),
                    Vec::new(),
                    None,
                    completions,
                    0,
                    Duration::from_secs(3),
                    &parent,
                    &subscriber,
                );

                tokio::time::advance(Duration::from_secs(2)).await;
                tokio::task::yield_now().await;
                assert!(
                    tasks.next().now_or_never().is_none(),
                    "throttled dependency should not be released by idle timeout"
                );

                tokio::time::advance(Duration::from_secs(1)).await;
                let message = tasks
                    .next()
                    .await
                    .expect("dependency flush task should finish at hard timeout")
                    .expect("dependency flush task should not panic");
                assert!(matches!(
                    message,
                    super::TaskMessage::DependencySettled { .. }
                ));
            })
            .await;
    }

    #[tokio::test(flavor = "current_thread", start_paused = true)]
    async fn dependency_flush_waits_for_active_work_past_idle_timeout() {
        use futures::{FutureExt as _, StreamExt as _};

        let local = LocalSet::new();
        local
            .run_until(async {
                let module_id = ModuleId::new(ImmediateDependentModule::id()).unwrap();
                let mut alloc = ResourceAllocation::default();
                alloc.set(module_id.clone(), ModuleConfig::default());
                alloc.set_activation(module_id, ActivationRatio::ONE);

                let caps = test_caps(Blackboard::with_allocation(alloc));
                let modules = ModuleRegistry::new()
                    .register_sync(
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
                let module = modules.pop().unwrap();
                let owner = module.owner().clone();
                let dependency_owner = ModuleInstanceId::new(
                    ModuleId::new("active-work").unwrap(),
                    ReplicaIndex::ZERO,
                );
                let (kick, completion) = crate::kicks::Kick::new(dependency_owner.clone());
                let (kick_inbox, _) = crate::kicks::KickInbox::new();

                let mut tasks = futures::stream::FuturesUnordered::new();
                let subscriber = tracing::dispatcher::get_default(Clone::clone);
                let parent = tracing::Span::current();
                super::spawn_dependency_settle_wait(
                    &mut tasks,
                    0,
                    owner,
                    module,
                    kick_inbox,
                    Vec::new(),
                    None,
                    vec![super::DependencyWait::kick(dependency_owner, completion)],
                    0,
                    Duration::from_secs(10),
                    &parent,
                    &subscriber,
                );

                tokio::time::advance(Duration::from_secs(2)).await;
                tokio::task::yield_now().await;
                assert!(
                    tasks.next().now_or_never().is_none(),
                    "active dependency work should not be released by idle timeout"
                );

                kick.notify_finish();
                let message = tasks
                    .next()
                    .await
                    .expect("dependency flush task should finish after dependency completion")
                    .expect("dependency flush task should not panic");
                assert!(matches!(
                    message,
                    super::TaskMessage::DependencySettled { .. }
                ));
            })
            .await;
    }

    #[tokio::test(flavor = "current_thread", start_paused = false)]
    async fn activation_failure_noops_ready_kicks_before_parking() {
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
                    .register_sync(
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
                    runtime.peer_contexts(),
                    runtime.allocation_hints(),
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
                let mut consecutive_failures = vec![0_u32];
                super::handle_task_message(
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
                    &mut consecutive_failures,
                    true,
                    test_config(),
                    &parent,
                    &subscriber,
                )
                .await
                .expect("activation failure should be handled by module parking");
                assert_eq!(consecutive_failures[0], 1);
                assert!(matches!(
                    &states[0],
                    super::ModuleState::FailedUntilActivation { phase, message, .. }
                        if phase == "activate" && message.contains("planned activation failure")
                ));
                completion
                    .await
                    .expect("ready kick should be completed before activation parking");
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
                    .register_sync(
                        test_policy(1..=1, Bpm::from_f64(60.0)..=Bpm::from_f64(60.0)),
                        |_| HangingBatchStub,
                    )
                    .unwrap()
                    .build(&caps)
                    .await
                    .unwrap();
                let (runtime, mut modules, _) = modules.into_parts_with_dependencies();
                let module = modules.pop().unwrap();
                let owner = module.owner().clone();
                let (kick_inbox, kick_handle) = crate::kicks::KickInbox::new();
                let mut tasks = futures::stream::FuturesUnordered::new();
                let subscriber = tracing::dispatcher::get_default(Clone::clone);
                let parent = tracing::Span::current();

                super::spawn_next_batch(
                    runtime,
                    &mut tasks,
                    0,
                    owner.clone(),
                    module,
                    kick_inbox,
                    Vec::new(),
                    None,
                    Duration::from_secs(2),
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
                    .register_sync(
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
                        max_activation_attempts: 1,
                        dependency_idle_timeout: std::time::Duration::from_secs(2),
                        dependency_hard_timeout: std::time::Duration::from_secs(10),
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
