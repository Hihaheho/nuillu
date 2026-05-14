use std::cell::Cell;
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::rc::Rc;
use std::sync::Arc;
use std::time::Duration;

use lutum::Lutum;
use nuillu_blackboard::{
    ActivationRatio, AgenticDeadlockMarker, Blackboard, BlackboardCommand, ModulePolicy,
    ModuleRunStatus, ZeroReplicaWindowPolicy,
};
use nuillu_types::{ModuleId, ModuleInstanceId, ReplicaCapRange, ReplicaIndex};

use crate::activation_gate::ActivationGateHub;
use crate::channels::{Topic, TopicPolicy};
use crate::llm::LlmConcurrencyLimiter;
use crate::ports::{Clock, CognitionLogRepository};
use crate::rate_limit::{RateLimiter, RuntimePolicy, TopicKind};
use crate::runtime_events::{NoopRuntimeEventSink, RuntimeEventEmitter, RuntimeEventSink};
use crate::scene::{SceneReader, SceneRegistry};
use crate::r#trait::ErasedModule;
use crate::{
    AllocationReader, AllocationUpdated, AllocationUpdatedInbox, AllocationUpdatedMailbox,
    AllocationWriter, AttentionControlRequest, AttentionControlRequestInbox,
    AttentionControlRequestMailbox, BlackboardReader, CognitionLogReader, CognitionLogUpdated,
    CognitionLogUpdatedInbox, CognitionLogUpdatedMailbox, CognitionWriter, LlmAccess, LutumTiers,
    Memo, MemoUpdated, MemoUpdatedInbox, MemoUpdatedMailbox, Module, ModuleBatch,
    ModuleStatusReader, SensoryInput, SensoryInputInbox, SensoryInputMailbox, TimeDivision,
    TopicInbox, TopicMailbox, TypedMemo, VitalReader, VitalUpdated, VitalUpdatedInbox,
    VitalUpdatedMailbox, VitalWriter,
};

/// Provides [capabilities](crate) at agent boot.
///
/// Owner-stamped capabilities carry a hidden [`ModuleInstanceId`]. The root
/// provider set is a boot object; ordinary module constructors should receive
/// [`ModuleCapabilityFactory`] so they cannot choose another owner.
#[derive(Clone)]
pub struct CapabilityProviders {
    inner: Arc<CapabilityProvidersInner>,
}

struct CapabilityProvidersInner {
    blackboard: Blackboard,
    attention_control_requests: Topic<AttentionControlRequest>,
    cognition_log_updates: Topic<CognitionLogUpdated>,
    allocation_updates: Topic<AllocationUpdated>,
    vital_updates: Topic<VitalUpdated>,
    memo_updates: Topic<MemoUpdated>,
    sensory_input_topic: Topic<SensoryInput>,
    activation_gates: ActivationGateHub,
    cognition_log_port: Arc<dyn CognitionLogRepository>,
    clock: Arc<dyn Clock>,
    time_division: TimeDivision,
    tiers: LutumTiers,
    runtime_events: RuntimeEventEmitter,
    rate_limiter: RateLimiter,
    llm_concurrency_limiter: LlmConcurrencyLimiter,
    runtime_policy: RuntimePolicy,
    scene: SceneRegistry,
}

/// Required external services for the root capability provider set.
#[derive(Clone)]
pub struct CapabilityProviderPorts {
    pub blackboard: Blackboard,
    pub cognition_log_port: Arc<dyn CognitionLogRepository>,
    pub clock: Arc<dyn Clock>,
    pub tiers: LutumTiers,
}

/// Runtime policy and observation hooks layered on top of the boot ports.
#[derive(Clone)]
pub struct CapabilityProviderRuntime {
    pub event_sink: Arc<dyn RuntimeEventSink>,
    pub policy: RuntimePolicy,
}

impl Default for CapabilityProviderRuntime {
    fn default() -> Self {
        Self {
            event_sink: Arc::new(NoopRuntimeEventSink),
            policy: RuntimePolicy::default(),
        }
    }
}

/// Full root provider boot config.
#[derive(Clone)]
pub struct CapabilityProviderConfig {
    pub ports: CapabilityProviderPorts,
    pub runtime: CapabilityProviderRuntime,
}

impl From<CapabilityProviderPorts> for CapabilityProviderConfig {
    fn from(ports: CapabilityProviderPorts) -> Self {
        Self {
            ports,
            runtime: CapabilityProviderRuntime::default(),
        }
    }
}

impl CapabilityProviders {
    pub fn new(config: impl Into<CapabilityProviderConfig>) -> Self {
        let CapabilityProviderConfig { ports, runtime } = config.into();
        let CapabilityProviderPorts {
            blackboard,
            cognition_log_port,
            clock,
            tiers,
        } = ports;
        let CapabilityProviderRuntime { event_sink, policy } = runtime;
        let runtime_events = RuntimeEventEmitter::new(event_sink);
        let rate_limiter = RateLimiter::new(policy.rate_limits.clone());
        let llm_concurrency_limiter = LlmConcurrencyLimiter::new(policy.max_concurrent_llm_calls);
        Self {
            inner: Arc::new(CapabilityProvidersInner {
                attention_control_requests: Topic::new(
                    blackboard.clone(),
                    TopicPolicy::RoleLoadBalanced,
                    TopicKind::AttentionControlRequest,
                    rate_limiter.clone(),
                    runtime_events.clone(),
                ),
                cognition_log_updates: Topic::new(
                    blackboard.clone(),
                    TopicPolicy::Fanout,
                    TopicKind::CognitionLogUpdated,
                    rate_limiter.clone(),
                    runtime_events.clone(),
                ),
                allocation_updates: Topic::new(
                    blackboard.clone(),
                    TopicPolicy::Fanout,
                    TopicKind::AllocationUpdated,
                    rate_limiter.clone(),
                    runtime_events.clone(),
                ),
                vital_updates: Topic::new(
                    blackboard.clone(),
                    TopicPolicy::Fanout,
                    TopicKind::VitalUpdated,
                    rate_limiter.clone(),
                    runtime_events.clone(),
                ),
                memo_updates: Topic::new(
                    blackboard.clone(),
                    TopicPolicy::Fanout,
                    TopicKind::MemoUpdated,
                    rate_limiter.clone(),
                    runtime_events.clone(),
                ),
                sensory_input_topic: Topic::new(
                    blackboard.clone(),
                    TopicPolicy::RoleLoadBalanced,
                    TopicKind::SensoryInput,
                    rate_limiter.clone(),
                    runtime_events.clone(),
                ),
                activation_gates: ActivationGateHub::new(blackboard.clone()),
                blackboard,
                cognition_log_port,
                clock,
                time_division: TimeDivision::default(),
                tiers,
                runtime_events,
                rate_limiter,
                llm_concurrency_limiter,
                runtime_policy: policy,
                scene: SceneRegistry::empty(),
            }),
        }
    }

    /// Access the scene registry for host-driven participant updates.
    ///
    /// The host (eval harness, game runtime) calls `scene().set(...)` to
    /// declare which participants are currently in earshot. The agent only
    /// reads the registry, never mutates it.
    pub fn scene(&self) -> &SceneRegistry {
        &self.inner.scene
    }

    pub(crate) fn scoped(&self, owner: ModuleInstanceId) -> ModuleCapabilityFactory {
        ModuleCapabilityFactory {
            owner,
            root: self.clone(),
            memo_issued: Rc::new(Cell::new(false)),
        }
    }

    pub(crate) async fn set_module_policies(&self, policies: Vec<(ModuleId, ModulePolicy)>) {
        self.inner
            .blackboard
            .apply(BlackboardCommand::SetModulePolicies { policies })
            .await;
    }

    pub(crate) async fn set_module_replica_capacities(&self, capacities: Vec<(ModuleId, u8)>) {
        self.inner
            .blackboard
            .apply(BlackboardCommand::SetModuleReplicaCapacities { capacities })
            .await;
    }

    pub(crate) fn set_module_catalog(&self, catalog: Vec<(ModuleId, &'static str)>) {
        self.inner.blackboard.set_module_catalog(catalog);
    }

    pub(crate) async fn apply_runtime_policy(&self) {
        self.inner
            .blackboard
            .apply(BlackboardCommand::SetAllocationLimits(
                self.inner.runtime_policy.allocation_limits,
            ))
            .await;
        self.inner
            .blackboard
            .apply(BlackboardCommand::SetMemoRetentionPerOwner(
                self.inner.runtime_policy.memo_retained_per_owner,
            ))
            .await;
    }

    pub(crate) fn runtime_control(&self) -> AgentRuntimeControl {
        let owner = ModuleInstanceId::new(
            ModuleId::new("agent-event-loop").expect("agent event loop id is valid"),
            ReplicaIndex::ZERO,
        );
        AgentRuntimeControl {
            blackboard: self.inner.blackboard.clone(),
            cognition_log_updates: CognitionLogUpdatedMailbox::new(
                owner,
                self.inner.cognition_log_updates.clone(),
            ),
            clock: self.inner.clock.clone(),
            session_compaction_lutum: self.inner.tiers.cheap.clone(),
            runtime_events: self.inner.runtime_events.clone(),
            activation_gates: self.inner.activation_gates.clone(),
        }
    }

    pub fn blackboard_reader(&self) -> BlackboardReader {
        BlackboardReader::new(self.inner.blackboard.clone())
    }

    pub fn cognition_log_reader(&self) -> CognitionLogReader {
        CognitionLogReader::new(self.inner.blackboard.clone())
    }

    pub fn allocation_reader(&self) -> AllocationReader {
        AllocationReader::new(self.inner.blackboard.clone())
    }

    pub fn module_status_reader(&self) -> ModuleStatusReader {
        ModuleStatusReader::new(self.inner.blackboard.clone())
    }

    pub fn clock(&self) -> Arc<dyn Clock> {
        self.inner.clock.clone()
    }

    pub fn time_division(&self) -> TimeDivision {
        self.inner.time_division.clone()
    }

    pub fn host_io(&self) -> HostIo {
        HostIo {
            owner: ModuleInstanceId::new(
                ModuleId::new("host").expect("host module id is valid"),
                ReplicaIndex::ZERO,
            ),
            root: self.clone(),
        }
    }

    pub fn internal_harness_io(&self) -> InternalHarnessIo {
        InternalHarnessIo {
            owner: ModuleInstanceId::new(
                ModuleId::new("eval-harness").expect("eval-harness module id is valid"),
                ReplicaIndex::ZERO,
            ),
            root: self.clone(),
        }
    }
}

#[derive(Clone)]
pub struct AgentRuntimeControl {
    blackboard: Blackboard,
    cognition_log_updates: CognitionLogUpdatedMailbox,
    clock: Arc<dyn Clock>,
    session_compaction_lutum: Lutum,
    runtime_events: RuntimeEventEmitter,
    activation_gates: ActivationGateHub,
}

impl AgentRuntimeControl {
    pub async fn is_active(&self, owner: &ModuleInstanceId) -> bool {
        self.blackboard
            .read(|bb| bb.allocation().is_replica_active(owner))
            .await
    }

    pub fn clock(&self) -> Arc<dyn Clock> {
        self.clock.clone()
    }

    pub fn session_compaction_lutum(&self) -> &Lutum {
        &self.session_compaction_lutum
    }

    /// Snapshot of the registered-module catalog. Cheap synchronous read; the
    /// scheduler turns this into an [`ActivateCx`] for each `activate` call.
    pub fn module_catalog(&self) -> Vec<(ModuleId, &'static str)> {
        self.blackboard.module_catalog().to_vec()
    }

    pub async fn identity_memories(&self) -> Vec<nuillu_blackboard::IdentityMemoryRecord> {
        self.blackboard
            .read(|bb| bb.identity_memories().to_vec())
            .await
    }

    pub async fn core_policies(&self) -> Vec<nuillu_blackboard::CorePolicyRecord> {
        self.blackboard.read(|bb| bb.core_policies().to_vec()).await
    }

    pub async fn record_module_status(&self, owner: ModuleInstanceId, status: ModuleRunStatus) {
        self.blackboard
            .apply(BlackboardCommand::SetModuleRunStatus { owner, status })
            .await;
    }

    pub async fn module_batch_min_interval(&self, owner: &ModuleInstanceId) -> Option<Duration> {
        self.blackboard
            .read(|bb| bb.allocation().cooldown_for(&owner.module))
            .await
    }

    pub async fn module_batch_throttle_baseline(
        &self,
        owner: &ModuleInstanceId,
    ) -> Option<(Duration, ActivationRatio)> {
        self.blackboard
            .read(|bb| {
                let allocation = bb.allocation();
                allocation
                    .cooldown_for(&owner.module)
                    .map(|interval| (interval, allocation.activation_for(&owner.module)))
            })
            .await
    }

    pub async fn active_replicas(&self, module: &ModuleId) -> u8 {
        self.blackboard
            .read(|bb| bb.allocation().active_replicas(module))
            .await
    }

    pub async fn zero_replica_window_policies(&self) -> HashMap<ModuleId, ZeroReplicaWindowPolicy> {
        self.blackboard
            .read(|bb| {
                bb.module_policies()
                    .iter()
                    .filter_map(|(module, policy)| {
                        (policy.replicas_range.max > 0
                            && policy
                                .zero_replica_window
                                .controller_activation_period()
                                .is_some())
                        .then_some((module.clone(), policy.zero_replica_window))
                    })
                    .collect()
            })
            .await
    }

    pub async fn activation_increase_waiter(
        &self,
        owner: &ModuleInstanceId,
        threshold: ActivationRatio,
    ) -> Option<tokio::sync::oneshot::Receiver<()>> {
        self.blackboard
            .activation_increase_waiter(owner.module.clone(), threshold)
            .await
    }

    pub async fn activation_waiter(
        &self,
        owner: &ModuleInstanceId,
    ) -> Option<tokio::sync::oneshot::Receiver<()>> {
        self.blackboard.activation_waiter(owner.clone()).await
    }

    pub async fn allocation_change_waiter(&self) -> tokio::sync::oneshot::Receiver<()> {
        self.blackboard.allocation_change_waiter().await
    }

    pub async fn record_module_batch_throttled(
        &self,
        owner: ModuleInstanceId,
        delayed_for: Duration,
    ) {
        self.runtime_events
            .module_batch_throttled(owner, delayed_for)
            .await;
    }

    pub async fn record_module_batch_ready(&self, owner: ModuleInstanceId, batch: &ModuleBatch) {
        self.runtime_events.module_batch_ready(owner, batch).await;
    }

    pub async fn record_agentic_deadlock_marker(&self, idle_for: Duration) {
        self.blackboard
            .apply(BlackboardCommand::RecordAgenticDeadlockMarker(
                AgenticDeadlockMarker {
                    at: self.clock.now(),
                    idle_for,
                },
            ))
            .await;

        if self
            .cognition_log_updates
            .publish(CognitionLogUpdated::AgenticDeadlockMarker)
            .await
            .is_err()
        {
            tracing::trace!("agentic deadlock cognition-log update had no active subscribers");
        }
    }

    pub async fn activation_gate_requests(
        &self,
        target: &ModuleInstanceId,
        batch: ModuleBatch,
    ) -> Vec<tokio::sync::oneshot::Receiver<crate::ActivationGateVote>> {
        self.activation_gates.dispatch(target, batch).await
    }
}

#[derive(Clone)]
pub struct HostIo {
    owner: ModuleInstanceId,
    root: CapabilityProviders,
}

impl HostIo {
    pub fn sensory_input_mailbox(&self) -> SensoryInputMailbox {
        TopicMailbox::new(
            self.owner.clone(),
            self.root.inner.sensory_input_topic.clone(),
        )
    }

    pub fn allocation_updated_mailbox(&self) -> AllocationUpdatedMailbox {
        TopicMailbox::new(
            self.owner.clone(),
            self.root.inner.allocation_updates.clone(),
        )
    }
}

#[derive(Clone)]
pub struct InternalHarnessIo {
    owner: ModuleInstanceId,
    root: CapabilityProviders,
}

impl InternalHarnessIo {
    pub fn attention_control_mailbox(&self) -> AttentionControlRequestMailbox {
        TopicMailbox::new(
            self.owner.clone(),
            self.root.inner.attention_control_requests.clone(),
        )
    }

    pub fn cognition_log_updated_mailbox(&self) -> CognitionLogUpdatedMailbox {
        TopicMailbox::new(
            self.owner.clone(),
            self.root.inner.cognition_log_updates.clone(),
        )
    }

    pub fn allocation_updated_mailbox(&self) -> AllocationUpdatedMailbox {
        TopicMailbox::new(
            self.owner.clone(),
            self.root.inner.allocation_updates.clone(),
        )
    }

    pub fn vital_updated_mailbox(&self) -> VitalUpdatedMailbox {
        TopicMailbox::new(self.owner.clone(), self.root.inner.vital_updates.clone())
    }

    pub fn memo_updated_mailbox(&self) -> MemoUpdatedMailbox {
        TopicMailbox::new(self.owner.clone(), self.root.inner.memo_updates.clone())
    }
}

#[derive(Clone)]
pub struct ModuleCapabilityFactory {
    owner: ModuleInstanceId,
    root: CapabilityProviders,
    // Memo is the only single-issued capability: typed memo safety relies on
    // one payload type per module owner.
    memo_issued: Rc<Cell<bool>>,
}

impl ModuleCapabilityFactory {
    /// The owner this factory dispenses capabilities for. Capability handles
    /// returned by this factory are stamped with this id.
    pub fn owner(&self) -> &ModuleInstanceId {
        &self.owner
    }

    pub fn attention_control_mailbox(&self) -> AttentionControlRequestMailbox {
        TopicMailbox::new(
            self.owner.clone(),
            self.root.inner.attention_control_requests.clone(),
        )
    }

    pub fn attention_control_inbox(&self) -> AttentionControlRequestInbox {
        TopicInbox::new(
            self.owner.clone(),
            self.root.inner.attention_control_requests.clone(),
        )
    }

    pub fn cognition_log_updated_inbox(&self) -> CognitionLogUpdatedInbox {
        TopicInbox::new_excluding_self(
            self.owner.clone(),
            self.root.inner.cognition_log_updates.clone(),
        )
    }

    pub fn allocation_updated_inbox(&self) -> AllocationUpdatedInbox {
        TopicInbox::new(
            self.owner.clone(),
            self.root.inner.allocation_updates.clone(),
        )
    }

    pub fn vital_updated_inbox(&self) -> VitalUpdatedInbox {
        TopicInbox::new_excluding_self(self.owner.clone(), self.root.inner.vital_updates.clone())
    }

    pub fn memo_updated_inbox(&self) -> MemoUpdatedInbox {
        TopicInbox::new_excluding_self(self.owner.clone(), self.root.inner.memo_updates.clone())
    }

    pub fn sensory_input_mailbox(&self) -> SensoryInputMailbox {
        TopicMailbox::new(
            self.owner.clone(),
            self.root.inner.sensory_input_topic.clone(),
        )
    }

    pub fn sensory_input_inbox(&self) -> SensoryInputInbox {
        TopicInbox::new(
            self.owner.clone(),
            self.root.inner.sensory_input_topic.clone(),
        )
    }

    pub fn activation_gate_for<M: Module + 'static>(&self) -> crate::ActivationGate<M> {
        self.root
            .inner
            .activation_gates
            .subscribe::<M>(self.owner.clone())
    }

    fn claim_memo(&self) {
        assert!(
            !self.memo_issued.replace(true),
            "module requested multiple memo capabilities; choose exactly one of memo() or typed_memo::<T>()"
        );
    }

    pub fn memo(&self) -> Memo {
        self.claim_memo();
        Memo::new(
            self.owner.clone(),
            self.root.inner.blackboard.clone(),
            TopicMailbox::new(self.owner.clone(), self.root.inner.memo_updates.clone()),
            self.root.inner.clock.clone(),
            self.root.inner.runtime_events.clone(),
        )
    }

    pub fn typed_memo<T: 'static>(&self) -> TypedMemo<T> {
        self.claim_memo();
        TypedMemo::new(
            self.owner.clone(),
            self.root.inner.blackboard.clone(),
            TopicMailbox::new(self.owner.clone(), self.root.inner.memo_updates.clone()),
            self.root.inner.clock.clone(),
            self.root.inner.runtime_events.clone(),
        )
    }

    pub fn llm_access(&self) -> LlmAccess {
        LlmAccess::new(
            self.owner.clone(),
            self.root.inner.tiers.clone(),
            self.root.inner.blackboard.clone(),
            self.root.inner.runtime_events.clone(),
            self.root.inner.rate_limiter.clone(),
            self.root.inner.llm_concurrency_limiter.clone(),
        )
    }

    pub fn blackboard_reader(&self) -> BlackboardReader {
        self.root.blackboard_reader()
    }

    /// Raw [`Blackboard`] handle. Domain crates use this to build their own
    /// owner-stamped capability handles outside of `nuillu-module`.
    pub fn blackboard(&self) -> Blackboard {
        self.root.inner.blackboard.clone()
    }

    pub fn cognition_log_reader(&self) -> CognitionLogReader {
        self.root.cognition_log_reader()
    }

    pub fn allocation_reader(&self) -> AllocationReader {
        self.root.allocation_reader()
    }

    pub fn vital_reader(&self) -> VitalReader {
        VitalReader::new(self.root.inner.blackboard.clone())
    }

    pub fn module_status_reader(&self) -> ModuleStatusReader {
        self.root.module_status_reader()
    }

    pub fn cognition_writer(&self) -> CognitionWriter {
        CognitionWriter::new(
            self.owner.clone(),
            self.root.inner.blackboard.clone(),
            self.root.inner.cognition_log_port.clone(),
            CognitionLogUpdatedMailbox::new(
                self.owner.clone(),
                self.root.inner.cognition_log_updates.clone(),
            ),
            self.root.inner.clock.clone(),
        )
    }

    pub fn allocation_writer(
        &self,
        allowed_drive_modules: Vec<ModuleId>,
        allowed_cap_modules: Vec<ModuleId>,
    ) -> AllocationWriter {
        AllocationWriter::new(
            self.owner.clone(),
            self.root.inner.blackboard.clone(),
            AllocationUpdatedMailbox::new(
                self.owner.clone(),
                self.root.inner.allocation_updates.clone(),
            ),
            allowed_drive_modules,
            allowed_cap_modules,
        )
    }

    pub fn vital_writer(&self) -> VitalWriter {
        VitalWriter::new(
            self.owner.clone(),
            self.root.inner.blackboard.clone(),
            VitalUpdatedMailbox::new(self.owner.clone(), self.root.inner.vital_updates.clone()),
            self.root.inner.clock.clone(),
        )
    }

    pub fn scene_reader(&self) -> SceneReader {
        SceneReader::new(self.root.inner.scene.clone())
    }

    pub fn clock(&self) -> Arc<dyn Clock> {
        self.root.clock()
    }

    pub fn time_division(&self) -> TimeDivision {
        self.root.time_division()
    }
}

pub struct AllocatedModule {
    owner: ModuleInstanceId,
    module: Box<dyn ErasedModule>,
}

impl AllocatedModule {
    fn new(owner: ModuleInstanceId, module: Box<dyn ErasedModule>) -> Self {
        Self { owner, module }
    }

    pub fn owner(&self) -> &ModuleInstanceId {
        &self.owner
    }

    pub async fn next_batch(&mut self) -> anyhow::Result<ModuleBatch> {
        self.module.next_batch().await
    }

    pub async fn activate(
        &mut self,
        cx: &crate::ActivateCx<'_>,
        batch: &ModuleBatch,
    ) -> anyhow::Result<()> {
        self.module.activate(cx, batch).await
    }
}

pub struct AllocatedModules {
    runtime: AgentRuntimeControl,
    modules: Vec<AllocatedModule>,
    dependencies: ModuleDependencies,
}

impl AllocatedModules {
    fn new(
        runtime: AgentRuntimeControl,
        modules: Vec<AllocatedModule>,
        dependencies: ModuleDependencies,
    ) -> Self {
        Self {
            runtime,
            modules,
            dependencies,
        }
    }

    pub fn len(&self) -> usize {
        self.modules.len()
    }

    pub fn is_empty(&self) -> bool {
        self.modules.is_empty()
    }

    pub fn dependencies(&self) -> &ModuleDependencies {
        &self.dependencies
    }

    pub fn into_parts(self) -> (AgentRuntimeControl, Vec<AllocatedModule>) {
        (self.runtime, self.modules)
    }

    pub fn into_parts_with_dependencies(
        self,
    ) -> (
        AgentRuntimeControl,
        Vec<AllocatedModule>,
        ModuleDependencies,
    ) {
        (self.runtime, self.modules, self.dependencies)
    }
}

/// Per-module dependency map keyed by role, not replica.
#[derive(Debug, Default, Clone)]
pub struct ModuleDependencies {
    deps_of: HashMap<ModuleId, Vec<ModuleId>>,
    dependents_of: HashMap<ModuleId, Vec<ModuleId>>,
}

impl ModuleDependencies {
    pub fn deps_of(&self, module: &ModuleId) -> &[ModuleId] {
        self.deps_of.get(module).map(Vec::as_slice).unwrap_or(&[])
    }

    pub fn dependents_of(&self, module: &ModuleId) -> &[ModuleId] {
        self.dependents_of
            .get(module)
            .map(Vec::as_slice)
            .unwrap_or(&[])
    }
}

pub struct ModuleRegistry {
    registrations: Vec<ModuleRegistration>,
    dependencies: Vec<(ModuleId, ModuleId)>,
}

impl fmt::Debug for ModuleRegistry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ModuleRegistry")
            .field("registrations", &self.registrations)
            .field("dependencies", &self.dependencies)
            .finish()
    }
}

struct ModuleRegistration {
    module: ModuleId,
    role_description: &'static str,
    policy: ModulePolicy,
    replica_capacity: u8,
    builder: Box<dyn Fn(ModuleCapabilityFactory) -> Box<dyn ErasedModule>>,
}

impl fmt::Debug for ModuleRegistration {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ModuleRegistration")
            .field("module", &self.module)
            .field("role_description", &self.role_description)
            .field("policy", &self.policy)
            .field("replica_capacity", &self.replica_capacity)
            .finish_non_exhaustive()
    }
}

/// Builds one module replica from its replica-scoped capability factory.
///
/// Closures that return a concrete `Module` implement this automatically, so
/// registration call sites do not need to allocate erased module wrappers.
pub trait ModuleRegisterer: Fn(ModuleCapabilityFactory) -> Self::Module {
    type Module: crate::Module + 'static;
}

impl<F, M> ModuleRegisterer for F
where
    F: Fn(ModuleCapabilityFactory) -> M,
    M: Module + 'static,
{
    type Module = M;
}

impl ModuleRegistry {
    pub fn new() -> Self {
        Self {
            registrations: Vec::new(),
            dependencies: Vec::new(),
        }
    }

    /// Declare that `dependent` should wait for active `dependency` replicas to flush before
    /// activation. Both roles must be registered and the dependency graph must be acyclic.
    pub fn depends_on(mut self, dependent: ModuleId, dependency: ModuleId) -> Self {
        self.dependencies.push((dependent, dependency));
        self
    }

    /// Register a module type with its boot-time policy. The module's identity
    /// comes from [`Module::id`] / [`Module::role_description`].
    pub fn register<B>(
        mut self,
        policy: ModulePolicy,
        builder: B,
    ) -> Result<Self, ModuleRegistryError>
    where
        B: ModuleRegisterer + 'static,
    {
        let module = ModuleId::new(<B::Module as Module>::id())?;
        let role_description = <B::Module as Module>::role_description();
        if self
            .registrations
            .iter()
            .any(|registration| registration.module == module)
        {
            return Err(ModuleRegistryError::DuplicateModule { module });
        }
        let replica_capacity = policy.max_active_replicas();
        self.registrations.push(ModuleRegistration {
            module,
            role_description,
            policy,
            replica_capacity,
            builder: Box::new(move |caps| Box::new(builder(caps))),
        });
        Ok(self)
    }

    pub fn register_with_replica_capacity<B>(
        mut self,
        policy: ModulePolicy,
        replica_capacity: u8,
        builder: B,
    ) -> Result<Self, ModuleRegistryError>
    where
        B: ModuleRegisterer + 'static,
    {
        let module = ModuleId::new(<B::Module as Module>::id())?;
        let role_description = <B::Module as Module>::role_description();
        if self
            .registrations
            .iter()
            .any(|registration| registration.module == module)
        {
            return Err(ModuleRegistryError::DuplicateModule { module });
        }
        if replica_capacity > ReplicaCapRange::V1_MAX {
            return Err(ModuleRegistryError::ReplicaCapacityAboveV1Max {
                module,
                capacity: replica_capacity,
            });
        }
        let policy_capacity = policy.max_active_replicas();
        if replica_capacity < policy_capacity {
            return Err(ModuleRegistryError::ReplicaCapacityBelowPolicyMax {
                module,
                capacity: replica_capacity,
                policy_capacity,
            });
        }
        self.registrations.push(ModuleRegistration {
            module,
            role_description,
            policy,
            replica_capacity,
            builder: Box::new(move |caps| Box::new(builder(caps))),
        });
        Ok(self)
    }

    pub async fn build(
        &self,
        caps: &CapabilityProviders,
    ) -> Result<AllocatedModules, ModuleRegistryError> {
        let dependencies = self.compile_dependencies()?;

        caps.apply_runtime_policy().await;
        caps.set_module_policies(
            self.registrations
                .iter()
                .map(|registration| (registration.module.clone(), registration.policy.clone()))
                .collect(),
        )
        .await;
        caps.set_module_replica_capacities(
            self.registrations
                .iter()
                .map(|registration| (registration.module.clone(), registration.replica_capacity))
                .collect(),
        )
        .await;
        // Install the post-boot module catalog before any module is constructed
        // so module constructors can read peers from `caps.module_catalog()`
        // synchronously when they assemble their system prompts.
        caps.set_module_catalog(
            self.registrations
                .iter()
                .map(|registration| (registration.module.clone(), registration.role_description))
                .collect(),
        );

        let mut modules = Vec::new();
        for registration in &self.registrations {
            // Build every possible replica up to the registered max, with a
            // replica-0 floor so inactive modules can retain queued messages.
            let total_replicas = registration.replica_capacity;
            for replica in 0..total_replicas {
                let owner =
                    ModuleInstanceId::new(registration.module.clone(), ReplicaIndex::new(replica));
                let scoped = caps.scoped(owner.clone());
                modules.push(AllocatedModule::new(owner, (registration.builder)(scoped)));
            }
        }
        Ok(AllocatedModules::new(
            caps.runtime_control(),
            modules,
            dependencies,
        ))
    }

    fn compile_dependencies(&self) -> Result<ModuleDependencies, ModuleRegistryError> {
        let registered = self
            .registrations
            .iter()
            .map(|registration| &registration.module)
            .collect::<HashSet<_>>();
        let mut deps_of = HashMap::<ModuleId, Vec<ModuleId>>::new();
        let mut dependents_of = HashMap::<ModuleId, Vec<ModuleId>>::new();

        for (dependent, dependency) in &self.dependencies {
            if !registered.contains(dependent) {
                return Err(ModuleRegistryError::UnknownDependent {
                    dependent: dependent.clone(),
                });
            }
            if !registered.contains(dependency) {
                return Err(ModuleRegistryError::UnknownDependency {
                    dependency: dependency.clone(),
                });
            }
            if dependent == dependency {
                return Err(ModuleRegistryError::DependencyCycle {
                    cycle: vec![dependent.clone()],
                });
            }

            let deps = deps_of.entry(dependent.clone()).or_default();
            if !deps.contains(dependency) {
                deps.push(dependency.clone());
            }
            let dependents = dependents_of.entry(dependency.clone()).or_default();
            if !dependents.contains(dependent) {
                dependents.push(dependent.clone());
            }
        }

        let mut visiting = HashSet::<ModuleId>::new();
        let mut visited = HashSet::<ModuleId>::new();
        for module in registered {
            if visited.contains(module) {
                continue;
            }
            let mut stack = Vec::new();
            dfs_check_dependencies(
                module.clone(),
                &deps_of,
                &mut visiting,
                &mut visited,
                &mut stack,
            )?;
        }

        Ok(ModuleDependencies {
            deps_of,
            dependents_of,
        })
    }
}

fn dfs_check_dependencies(
    node: ModuleId,
    deps_of: &HashMap<ModuleId, Vec<ModuleId>>,
    visiting: &mut HashSet<ModuleId>,
    visited: &mut HashSet<ModuleId>,
    stack: &mut Vec<ModuleId>,
) -> Result<(), ModuleRegistryError> {
    if visited.contains(&node) {
        return Ok(());
    }
    if !visiting.insert(node.clone()) {
        let cycle_start = stack.iter().position(|module| module == &node).unwrap_or(0);
        let mut cycle = stack[cycle_start..].to_vec();
        cycle.push(node);
        return Err(ModuleRegistryError::DependencyCycle { cycle });
    }

    stack.push(node.clone());
    if let Some(deps) = deps_of.get(&node) {
        for dep in deps {
            dfs_check_dependencies(dep.clone(), deps_of, visiting, visited, stack)?;
        }
    }
    stack.pop();
    visiting.remove(&node);
    visited.insert(node);
    Ok(())
}

impl Default for ModuleRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ModuleRegistryError {
    #[error(transparent)]
    ModuleId(#[from] nuillu_types::ModuleIdParseError),
    #[error("module {module} is already registered")]
    DuplicateModule { module: ModuleId },
    #[error("module {module} replica capacity {capacity} exceeds v1 limit")]
    ReplicaCapacityAboveV1Max { module: ModuleId, capacity: u8 },
    #[error(
        "module {module} replica capacity {capacity} is below policy capacity {policy_capacity}"
    )]
    ReplicaCapacityBelowPolicyMax {
        module: ModuleId,
        capacity: u8,
        policy_capacity: u8,
    },
    #[error("dependent {dependent} declared in depends_on() but not registered")]
    UnknownDependent { dependent: ModuleId },
    #[error("dependency {dependency} declared in depends_on() but not registered")]
    UnknownDependency { dependency: ModuleId },
    #[error(
        "module dependency cycle detected: {}",
        cycle.iter().map(ModuleId::as_str).collect::<Vec<_>>().join(" -> ")
    )]
    DependencyCycle { cycle: Vec<ModuleId> },
}

#[cfg(test)]
mod tests {
    use super::*;

    use async_trait::async_trait;
    use chrono::{DateTime, Utc};
    use nuillu_blackboard::{
        ActivationRatio, Blackboard, BlackboardCommand, CognitionLogEntry, ModuleConfig,
        ResourceAllocation,
    };
    use nuillu_types::{ReplicaCapRange, builtin};

    use crate::ports::{CognitionLogRepository, PortError, SystemClock};
    use crate::test_support::{scoped, test_caps};

    fn test_policy(replicas_range: std::ops::RangeInclusive<u8>) -> ModulePolicy {
        ModulePolicy::new(
            ReplicaCapRange::new(*replicas_range.start(), *replicas_range.end()).unwrap(),
            nuillu_blackboard::Bpm::from_f64(60.0)..=nuillu_blackboard::Bpm::from_f64(60.0),
            nuillu_blackboard::linear_ratio_fn,
        )
    }

    #[derive(Clone, Default)]
    struct RecordingCognitionLogRepository {
        records: Arc<std::sync::Mutex<Vec<(ModuleInstanceId, CognitionLogEntry)>>>,
    }

    impl RecordingCognitionLogRepository {
        fn records(&self) -> Vec<(ModuleInstanceId, CognitionLogEntry)> {
            self.records.lock().expect("records mutex poisoned").clone()
        }
    }

    #[async_trait(?Send)]
    impl CognitionLogRepository for RecordingCognitionLogRepository {
        async fn append(
            &self,
            source: ModuleInstanceId,
            entry: CognitionLogEntry,
        ) -> Result<(), PortError> {
            self.records
                .lock()
                .expect("records mutex poisoned")
                .push((source, entry));
            Ok(())
        }

        async fn since(
            &self,
            source: &ModuleInstanceId,
            from: DateTime<Utc>,
        ) -> Result<Vec<CognitionLogEntry>, PortError> {
            Ok(self
                .records
                .lock()
                .expect("records mutex poisoned")
                .iter()
                .filter(|(record_source, entry)| record_source == source && entry.at >= from)
                .map(|(_, entry)| entry.clone())
                .collect())
        }
    }

    fn test_caps_with_cognition_repo(
        blackboard: Blackboard,
        cognition_log_port: Arc<dyn CognitionLogRepository>,
    ) -> CapabilityProviders {
        let adapter = Arc::new(lutum::MockLlmAdapter::new());
        let budget = lutum::SharedPoolBudgetManager::new(lutum::SharedPoolBudgetOptions::default());
        let lutum = lutum::Lutum::new(adapter, budget);
        CapabilityProviders::new(CapabilityProviderPorts {
            blackboard,
            cognition_log_port,
            clock: Arc::new(SystemClock),
            tiers: LutumTiers {
                cheap: lutum.clone(),
                default: lutum.clone(),
                premium: lutum,
            },
        })
    }

    struct NoopModule;

    #[async_trait(?Send)]
    impl Module for NoopModule {
        type Batch = ();

        fn id() -> &'static str {
            "noop"
        }

        fn role_description() -> &'static str {
            "test stub"
        }

        async fn next_batch(&mut self) -> anyhow::Result<Self::Batch> {
            Ok(())
        }

        async fn activate(
            &mut self,
            _cx: &crate::ActivateCx<'_>,
            _batch: &Self::Batch,
        ) -> anyhow::Result<()> {
            Ok(())
        }
    }

    fn noop_builder(_: ModuleCapabilityFactory) -> NoopModule {
        NoopModule
    }

    #[test]
    fn register_rejects_duplicate_module_ids() {
        let registry = ModuleRegistry::new()
            .register(test_policy(0..=0), noop_builder)
            .unwrap();

        let err = registry
            .register(test_policy(0..=0), noop_builder)
            .unwrap_err();

        let expected = nuillu_types::ModuleId::new(NoopModule::id()).unwrap();
        assert!(matches!(
            err,
            ModuleRegistryError::DuplicateModule { module } if module == expected
        ));
    }

    #[tokio::test]
    async fn register_with_replica_capacity_builds_hard_cap_but_keeps_soft_policy() {
        let blackboard = Blackboard::default();
        let caps = test_caps(blackboard.clone());
        let allocated = ModuleRegistry::new()
            .register_with_replica_capacity(test_policy(0..=1), 2, noop_builder)
            .unwrap()
            .build(&caps)
            .await
            .unwrap();

        assert_eq!(allocated.len(), 2);
        let module = nuillu_types::ModuleId::new(NoopModule::id()).unwrap();
        let (soft_max, capacity) = blackboard
            .read(|bb| {
                (
                    bb.module_policies()
                        .get(&module)
                        .unwrap()
                        .replicas_range
                        .max,
                    bb.module_replica_capacity(&module).unwrap(),
                )
            })
            .await;
        assert_eq!(soft_max, 1);
        assert_eq!(capacity, 2);
    }

    #[tokio::test]
    async fn capabilities_are_non_exclusive() {
        let caps = test_caps(Blackboard::default());
        let cognition_gate = scoped(&caps, builtin::cognition_gate(), 0);
        let controller = scoped(&caps, builtin::attention_controller(), 0);
        let _w1 = cognition_gate.cognition_writer();
        let _w2 = cognition_gate.cognition_writer();
        let _a1 = controller.allocation_writer(vec![builtin::cognition_gate()], Vec::new());
        let _a2 = controller.allocation_writer(vec![builtin::cognition_gate()], Vec::new());
    }

    #[tokio::test]
    async fn cognition_writer_appends_persists_publishes_and_owner_stamps() {
        let blackboard = Blackboard::default();
        let repo = RecordingCognitionLogRepository::default();
        let caps = test_caps_with_cognition_repo(blackboard.clone(), Arc::new(repo.clone()));
        let cognition_gate = scoped(&caps, builtin::cognition_gate(), 1);
        let subscriber = scoped(&caps, builtin::predict(), 0);
        let owner = ModuleInstanceId::new(builtin::cognition_gate(), ReplicaIndex::new(1));
        let mut updates = subscriber.cognition_log_updated_inbox();

        cognition_gate
            .cognition_writer()
            .append("food boundary changed")
            .await;

        let entries = blackboard
            .read(|bb| bb.cognition_log().entries().to_vec())
            .await;
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].text, "food boundary changed");

        let records = repo.records();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].0, owner);
        assert_eq!(records[0].1.text, "food boundary changed");

        let update = updates.next_item().await.unwrap();
        assert_eq!(update.sender, owner);
        assert_eq!(
            update.body,
            CognitionLogUpdated::EntryAppended {
                source: owner.clone()
            }
        );
    }

    #[tokio::test]
    async fn memo_updated_inbox_filters_self_writes() {
        let caps = test_caps(Blackboard::default());
        let cognition_gate = scoped(&caps, builtin::cognition_gate(), 0);
        let sensory = scoped(&caps, builtin::sensory(), 0);
        let mut inbox = cognition_gate.memo_updated_inbox();

        cognition_gate.memo().write("own memo").await;
        sensory.memo().write("sensory memo").await;

        let event = inbox.next_item().await.unwrap();
        assert_eq!(event.sender.module, builtin::sensory());
        assert_eq!(event.body.owner.module, builtin::sensory());
        assert!(inbox.take_ready_items().unwrap().items.is_empty());
    }

    #[tokio::test]
    async fn typed_memo_writes_plaintext_publishes_and_keeps_typed_payload() {
        #[derive(Debug, Clone, PartialEq, Eq)]
        struct TestMemoPayload {
            value: String,
        }

        let blackboard = Blackboard::default();
        let caps = test_caps(blackboard.clone());
        let query_vector = scoped(&caps, builtin::query_vector(), 0);
        let cognition_gate = scoped(&caps, builtin::cognition_gate(), 0);
        let owner = ModuleInstanceId::new(builtin::query_vector(), ReplicaIndex::ZERO);
        let mut inbox = cognition_gate.memo_updated_inbox();

        query_vector
            .typed_memo::<TestMemoPayload>()
            .write(
                TestMemoPayload {
                    value: "typed".into(),
                },
                "plain",
            )
            .await;

        let event = inbox.next_item().await.unwrap();
        assert_eq!(event.sender, owner);
        assert_eq!(event.body.owner, owner);
        assert_eq!(event.body.index, 0);

        let plaintext = blackboard.read(|bb| bb.recent_memo_logs()).await;
        assert_eq!(plaintext.len(), 1);
        assert_eq!(plaintext[0].content, "plain");

        let typed = blackboard.typed_memo_logs::<TestMemoPayload>(&owner).await;
        assert_eq!(typed.len(), 1);
        assert_eq!(typed[0].content, "plain");
        assert_eq!(
            typed[0].data(),
            &TestMemoPayload {
                value: "typed".into()
            }
        );
    }

    #[test]
    #[should_panic(
        expected = "module requested multiple memo capabilities; choose exactly one of memo() or typed_memo::<T>()"
    )]
    fn memo_then_typed_memo_panics() {
        let caps = test_caps(Blackboard::default());
        let query_vector = scoped(&caps, builtin::query_vector(), 0);

        let _plain = query_vector.memo();
        let _typed = query_vector.typed_memo::<u8>();
    }

    #[test]
    #[should_panic(
        expected = "module requested multiple memo capabilities; choose exactly one of memo() or typed_memo::<T>()"
    )]
    fn typed_memo_then_typed_memo_panics() {
        let caps = test_caps(Blackboard::default());
        let query_vector = scoped(&caps, builtin::query_vector(), 0);

        let _first = query_vector.typed_memo::<u8>();
        let _second = query_vector.typed_memo::<u16>();
    }

    #[tokio::test]
    async fn cognition_log_updated_inbox_filters_self_writes() {
        let caps = test_caps(Blackboard::default());
        let attention_schema = scoped(&caps, builtin::attention_schema(), 0);
        let cognition_gate = scoped(&caps, builtin::cognition_gate(), 0);
        let mut inbox = attention_schema.cognition_log_updated_inbox();

        attention_schema
            .cognition_writer()
            .append("own attention experience")
            .await;
        cognition_gate
            .cognition_writer()
            .append("promoted external evidence")
            .await;

        let event = inbox.next_item().await.unwrap();
        assert_eq!(event.sender.module, builtin::cognition_gate());
        assert_eq!(
            event.body,
            CognitionLogUpdated::EntryAppended {
                source: ModuleInstanceId::new(builtin::cognition_gate(), ReplicaIndex::ZERO)
            }
        );
        assert!(inbox.take_ready_items().unwrap().items.is_empty());
    }

    #[tokio::test]
    async fn allocation_writer_publishes_guidance_changes_once() {
        let blackboard = Blackboard::default();
        blackboard
            .apply(BlackboardCommand::SetModulePolicies {
                policies: vec![
                    (
                        builtin::attention_controller(),
                        nuillu_blackboard::ModulePolicy::new(
                            ReplicaCapRange::new(1, 1).unwrap(),
                            nuillu_blackboard::Bpm::from_f64(60.0)
                                ..=nuillu_blackboard::Bpm::from_f64(60.0),
                            nuillu_blackboard::linear_ratio_fn,
                        ),
                    ),
                    (
                        builtin::cognition_gate(),
                        nuillu_blackboard::ModulePolicy::new(
                            ReplicaCapRange::new(0, 1).unwrap(),
                            nuillu_blackboard::Bpm::from_f64(60.0)
                                ..=nuillu_blackboard::Bpm::from_f64(60.0),
                            nuillu_blackboard::linear_ratio_fn,
                        ),
                    ),
                ],
            })
            .await;
        let caps = test_caps(blackboard);
        let controller = scoped(&caps, builtin::attention_controller(), 0);
        let cognition_gate = scoped(&caps, builtin::cognition_gate(), 0);
        let writer = controller.allocation_writer(vec![builtin::cognition_gate()], Vec::new());
        let mut inbox = cognition_gate.allocation_updated_inbox();

        let mut proposal = ResourceAllocation::default();
        proposal.set(
            builtin::cognition_gate(),
            ModuleConfig {
                guidance: "promote current sensory memo into attention".into(),
            },
        );
        proposal.set_activation(builtin::cognition_gate(), ActivationRatio::ONE);

        writer.set(proposal.clone()).await;
        let event = inbox.next_item().await.unwrap();
        assert_eq!(event.sender.module, builtin::attention_controller());
        assert_eq!(event.body, crate::AllocationUpdated);

        writer.set(proposal).await;
        assert!(inbox.take_ready_items().unwrap().items.is_empty());
    }

    #[tokio::test]
    async fn cognition_log_updates_do_not_wake_controller_memo_inbox() {
        let caps = test_caps(Blackboard::default());
        let controller = scoped(&caps, builtin::attention_controller(), 0);
        let cognition_gate = scoped(&caps, builtin::cognition_gate(), 0);
        let mut memo_updates = controller.memo_updated_inbox();

        cognition_gate
            .cognition_writer()
            .append("user question needs a summary")
            .await;

        assert!(memo_updates.take_ready_items().unwrap().items.is_empty());
    }

    #[tokio::test]
    async fn speak_completion_memo_wakes_controller() {
        let caps = test_caps(Blackboard::default());
        let controller = scoped(&caps, builtin::attention_controller(), 0);
        let speak = scoped(&caps, builtin::speak(), 0);
        let mut memo_updates = controller.memo_updated_inbox();

        speak.memo().write("utterance completed").await;

        let event = memo_updates.next_item().await.unwrap();
        assert_eq!(event.sender.module, builtin::speak());
        assert_eq!(event.body.owner.module, builtin::speak());
    }
}
