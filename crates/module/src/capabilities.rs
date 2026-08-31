use std::cell::{Cell, RefCell};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::fmt;
use std::future::{Future, IntoFuture};
use std::pin::Pin;
use std::rc::Rc;
use std::sync::{Arc, OnceLock};
use std::time::Duration;

use lutum::{Lutum, Session};
use nuillu_blackboard::{
    ActivationRatio, AgenticDeadlockMarker, Blackboard, BlackboardCommand, Bpm, ModulePolicy,
    ModuleRunStatus, RegisteredModulePolicy, RegisteredSubsystemPolicy, SubsystemPolicy,
    SubsystemReplicaRange, ZeroReplicaWindowPolicy,
};
use nuillu_types::{
    ModelTier, ModuleActivationId, ModuleGroupId, ModuleId, ModuleInstanceId, ReplicaCapRange,
    ReplicaIndex, ScopeId, ScopedModuleId, SubsystemId,
};

use crate::activation_gate::ActivationGateHub;
use crate::channels::{Topic, TopicPolicy, WakeClaim, WakeRegistry};
use crate::ports::{Clock, CognitionLogRepository, PortError, Timer, TokioTimer};
use crate::readers::RoleReaderCursors;
use crate::runtime_events::{NoopRuntimeEventSink, RuntimeEventEmitter, RuntimeEventSink};
use crate::runtime_policy::RuntimePolicy;
use crate::scene::{SceneReader, SceneRegistry};
use crate::session::{
    NoopSessionStore, SessionAutoCompaction, SessionKey, SessionStore,
    attach_persistent_session_metadata,
};
use crate::tiers::{LlmTierHandle, LutumTiers};
use crate::r#trait::ErasedModule;
use crate::{
    ActionAffordanceReader, ActionAffordanceRegistry, ActionAffordanceWriter,
    ActionAffordancesUpdated, ActionAffordancesUpdatedInbox, ActionAffordancesUpdatedMailbox,
    AllocationReader, AllocationStore, AllocationWriter, AttentionControlRequest,
    AttentionControlRequestInbox, AttentionControlRequestMailbox, BlackboardReader,
    CognitionLogEvictedInbox, CognitionLogEvictedMailbox, CognitionLogReader, CognitionLogUpdated,
    CognitionLogUpdatedInbox, CognitionLogUpdatedMailbox, CognitionWriter, ExternalActionExecutor,
    ExternalActionInvoker, InteroceptionRuntimePolicy, InteroceptiveReader, InteroceptiveUpdated,
    InteroceptiveUpdatedInbox, InteroceptiveUpdatedMailbox, InteroceptiveWriter, LlmAccess, Memo,
    MemoLogEvictedInbox, MemoLogEvictedMailbox, MemoLogRepository, MemoSubscription, MemoUpdated,
    MemoUpdatedInbox, MemoUpdatedMailbox, MemoryMetadataReader, Module, ModuleBatch,
    ModuleStatusReader, NoopAllocationStore, NoopExternalActionExecutor, NoopMemoLogRepository,
    PersistedMemoLogEntry, ScopeLabels, SensoryInput, SensoryInputInbox, SensoryInputMailbox,
    SessionCompactionPolicy, StaticModule, SubsystemAllocationReader, SubsystemAllocationWriter,
    TimeDivision, TopicInbox, TopicMailbox, TypedMemo,
};

/// Immutable boot-time description of one registered module role.
#[derive(Debug, Clone)]
pub struct ModuleRegistrationSpec {
    scope: ScopeId,
    module: ModuleId,
    peer_context: Option<Arc<str>>,
    policy: ModulePolicy,
    replica_capacity: u8,
    initial_activation: ActivationRatio,
    groups: BTreeSet<ModuleGroupId>,
    dependencies: Vec<ModuleId>,
    activation_barrier: Option<ModuleActivationBarrierSpec>,
    memo_subscription: MemoSubscription,
}

#[derive(Debug, Clone)]
struct ModuleActivationBarrierSpec {
    prerequisites: Vec<ModuleId>,
    timeout: Option<Duration>,
}

/// Immutable boot-time description of an immediate child subsystem mount.
#[derive(Debug, Clone)]
pub struct SubsystemRegistrationSpec {
    pub parent_scope: ScopeId,
    pub subsystem: SubsystemId,
    pub policy: SubsystemPolicy,
    pub initial_activation: ActivationRatio,
    pub label: Option<Arc<str>>,
    pub allocation_description: Arc<str>,
    pub activation_table: Arc<[ActivationRatio]>,
}

impl SubsystemRegistrationSpec {
    pub fn new(
        parent_scope: ScopeId,
        subsystem: SubsystemId,
        policy: SubsystemPolicy,
        initial_activation: ActivationRatio,
        allocation_description: impl Into<Arc<str>>,
    ) -> Self {
        Self {
            parent_scope,
            subsystem,
            policy,
            initial_activation,
            label: None,
            allocation_description: allocation_description.into(),
            activation_table: vec![
                ActivationRatio::ONE,
                ActivationRatio::from_f64(0.85),
                ActivationRatio::from_f64(0.70),
                ActivationRatio::from_f64(0.50),
                ActivationRatio::from_f64(0.30),
                ActivationRatio::ZERO,
            ]
            .into(),
        }
    }

    pub fn with_label(mut self, label: impl Into<Arc<str>>) -> Self {
        self.label = Some(label.into());
        self
    }

    pub fn with_allocation_description(mut self, description: impl Into<Arc<str>>) -> Self {
        self.allocation_description = description.into();
        self
    }

    pub fn with_activation_table(
        mut self,
        table: impl IntoIterator<Item = ActivationRatio>,
    ) -> Self {
        self.activation_table = table.into_iter().collect::<Vec<_>>().into();
        self
    }
}

impl ModuleRegistrationSpec {
    pub fn new(
        module: ModuleId,
        policy: ModulePolicy,
        initial_activation: ActivationRatio,
    ) -> Self {
        let replica_capacity = policy.max_active_replicas();
        Self {
            scope: ScopeId::root(),
            module,
            peer_context: None,
            policy,
            replica_capacity,
            initial_activation,
            groups: BTreeSet::new(),
            dependencies: Vec::new(),
            activation_barrier: None,
            memo_subscription: MemoSubscription::All,
        }
    }

    pub fn for_static<M: StaticModule>(
        policy: ModulePolicy,
        initial_activation: ActivationRatio,
    ) -> Result<Self, nuillu_types::ModuleIdParseError> {
        let mut spec = Self::new(ModuleId::new(M::id())?, policy, initial_activation);
        if let Some(context) = M::peer_context() {
            spec.peer_context = Some(Arc::from(context));
        }
        Ok(spec)
    }

    pub fn with_peer_context(mut self, peer_context: impl Into<Arc<str>>) -> Self {
        self.peer_context = Some(peer_context.into());
        self
    }

    pub fn in_scope(mut self, scope: ScopeId) -> Self {
        self.scope = scope;
        self
    }

    pub fn scope(&self) -> &ScopeId {
        &self.scope
    }

    pub fn scoped_module(&self) -> ScopedModuleId {
        ScopedModuleId::new(self.scope.clone(), self.module.clone())
    }

    pub fn without_peer_context(mut self) -> Self {
        self.peer_context = None;
        self
    }

    pub fn with_replica_capacity(mut self, replica_capacity: u8) -> Self {
        self.replica_capacity = replica_capacity;
        self
    }

    pub fn in_group(mut self, group: ModuleGroupId) -> Self {
        self.groups.insert(group);
        self
    }

    pub fn depends_on(mut self, dependency: ModuleId) -> Self {
        if !self.dependencies.contains(&dependency) {
            self.dependencies.push(dependency);
        }
        self
    }

    /// Require every prerequisite role to complete successfully after this
    /// replica's previous successful activation before its next activation.
    pub fn with_activation_barrier(
        mut self,
        prerequisites: impl IntoIterator<Item = ModuleId>,
        timeout: Option<Duration>,
    ) -> Self {
        self.activation_barrier = Some(ModuleActivationBarrierSpec {
            prerequisites: prerequisites.into_iter().collect(),
            timeout,
        });
        self
    }

    pub fn with_memo_sources(mut self, sources: impl IntoIterator<Item = ModuleId>) -> Self {
        self.memo_subscription = MemoSubscription::only(sources);
        self
    }

    pub fn module(&self) -> &ModuleId {
        &self.module
    }

    pub fn peer_context(&self) -> Option<&str> {
        self.peer_context.as_deref()
    }

    pub fn policy(&self) -> &ModulePolicy {
        &self.policy
    }

    pub fn replica_capacity(&self) -> u8 {
        self.replica_capacity
    }

    pub fn initial_activation(&self) -> ActivationRatio {
        self.initial_activation
    }

    pub fn groups(&self) -> &BTreeSet<ModuleGroupId> {
        &self.groups
    }

    pub fn dependencies(&self) -> &[ModuleId] {
        &self.dependencies
    }

    pub fn memo_subscription(&self) -> &MemoSubscription {
        &self.memo_subscription
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ModuleCatalogEntry {
    scope: ScopeId,
    module: ModuleId,
    peer_context: Option<Arc<str>>,
    groups: BTreeSet<ModuleGroupId>,
    replica_range: ReplicaCapRange,
    replica_capacity: u8,
    initial_activation: ActivationRatio,
    bpm_range_bits: (u64, u64),
    activation_curve_bits: [(u64, u64); 3],
    zero_replica_window: ZeroReplicaWindowPolicy,
    memo_subscription: MemoSubscription,
}

/// Immutable catalog of the module roles registered in one agent environment.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ModuleCatalog {
    entries: Arc<[ModuleCatalogEntry]>,
    dependency_edges: Arc<[(ScopedModuleId, ScopedModuleId)]>,
    subsystems: Arc<[SubsystemCatalogEntry]>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct SubsystemCatalogEntry {
    parent_scope: ScopeId,
    subsystem: SubsystemId,
    label: Option<Arc<str>>,
    allocation_description: Arc<str>,
    activation_table: Arc<[ActivationRatio]>,
    replica_range: SubsystemReplicaRange,
    replica_capacity: u8,
    initial_activation: ActivationRatio,
    projection_curve_bits: [u64; 3],
}

impl ModuleCatalog {
    fn from_registrations(
        registrations: &[ModuleRegistration],
        dependencies: &[(ScopedModuleId, ScopedModuleId)],
        subsystems: &[SubsystemRegistrationSpec],
    ) -> Self {
        let entries = registrations
            .iter()
            .map(|registration| {
                let policy = &registration.spec.policy;
                let curve_at = |activation| {
                    let (replicas, rate) = policy.project(nuillu_blackboard::ActivationInput::new(
                        activation,
                        ActivationRatio::ONE,
                    ));
                    (replicas.as_f64().to_bits(), rate.as_f64().to_bits())
                };
                ModuleCatalogEntry {
                    scope: registration.spec.scope.clone(),
                    module: registration.spec.module.clone(),
                    peer_context: registration.spec.peer_context.clone(),
                    groups: registration.spec.groups.clone(),
                    replica_range: policy.replicas_range,
                    replica_capacity: registration.spec.replica_capacity,
                    initial_activation: registration.spec.initial_activation,
                    bpm_range_bits: (
                        policy.rate_limit_range.start().as_f64().to_bits(),
                        policy.rate_limit_range.end().as_f64().to_bits(),
                    ),
                    activation_curve_bits: [
                        curve_at(ActivationRatio::ZERO),
                        curve_at(ActivationRatio::from_f64(0.5)),
                        curve_at(ActivationRatio::ONE),
                    ],
                    zero_replica_window: policy.zero_replica_window,
                    memo_subscription: registration.spec.memo_subscription.clone(),
                }
            })
            .collect::<Vec<_>>();
        Self {
            entries: entries.into(),
            dependency_edges: dependencies.to_vec().into(),
            subsystems: subsystems
                .iter()
                .map(|spec| SubsystemCatalogEntry {
                    parent_scope: spec.parent_scope.clone(),
                    subsystem: spec.subsystem.clone(),
                    label: spec.label.clone(),
                    allocation_description: spec.allocation_description.clone(),
                    activation_table: spec.activation_table.clone(),
                    replica_range: spec.policy.replicas_range,
                    replica_capacity: spec.policy.replica_capacity,
                    initial_activation: spec.initial_activation,
                    projection_curve_bits: [
                        ActivationRatio::ZERO,
                        ActivationRatio::from_f64(0.5),
                        ActivationRatio::ONE,
                    ]
                    .map(|activation| {
                        spec.policy
                            .replica_projection
                            .project(nuillu_blackboard::ActivationInput::new(
                                activation,
                                ActivationRatio::ONE,
                            ))
                            .as_f64()
                            .to_bits()
                    }),
                })
                .collect::<Vec<_>>()
                .into(),
        }
    }

    fn members_in_scope(&self, scope: &ScopeId, group: &ModuleGroupId) -> Vec<ModuleId> {
        self.entries
            .iter()
            .filter(|entry| &entry.scope == scope && entry.groups.contains(group))
            .map(|entry| entry.module.clone())
            .collect()
    }

    fn contains_in_scope(&self, scope: &ScopeId, module: &ModuleId) -> bool {
        self.entries
            .iter()
            .any(|entry| &entry.scope == scope && &entry.module == module)
    }

    fn subsystems_in_scope(&self, scope: &ScopeId) -> Vec<SubsystemCatalogItem> {
        self.subsystems
            .iter()
            .filter(|entry| &entry.parent_scope == scope)
            .map(|entry| SubsystemCatalogItem {
                subsystem: entry.subsystem.clone(),
                label: entry.label.clone(),
                allocation_description: entry.allocation_description.clone(),
                activation_table: entry.activation_table.clone(),
                replica_range: entry.replica_range,
                replica_capacity: entry.replica_capacity,
                initial_activation: entry.initial_activation,
            })
            .collect()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SubsystemCatalogItem {
    pub subsystem: SubsystemId,
    pub label: Option<Arc<str>>,
    pub allocation_description: Arc<str>,
    pub activation_table: Arc<[ActivationRatio]>,
    pub replica_range: SubsystemReplicaRange,
    pub replica_capacity: u8,
    pub initial_activation: ActivationRatio,
}

pub struct ModuleCatalogView<'a> {
    catalog: &'a ModuleCatalog,
    scope: &'a ScopeId,
}

pub struct SubsystemCatalogView<'a> {
    catalog: &'a ModuleCatalog,
    scope: &'a ScopeId,
}

impl SubsystemCatalogView<'_> {
    pub fn children(&self) -> Vec<SubsystemCatalogItem> {
        self.catalog.subsystems_in_scope(self.scope)
    }
}

impl ModuleCatalogView<'_> {
    pub fn members(&self, group: &ModuleGroupId) -> Vec<ModuleId> {
        self.catalog.members_in_scope(self.scope, group)
    }

    pub fn contains(&self, module: &ModuleId) -> bool {
        self.catalog.contains_in_scope(self.scope, module)
    }

    pub fn child_subsystems(&self) -> Vec<SubsystemCatalogItem> {
        self.catalog.subsystems_in_scope(self.scope)
    }
}

/// Provides [capabilities](crate) at agent boot.
///
/// Owner-stamped capabilities carry a hidden [`ModuleInstanceId`]. The root
/// provider set is a boot object; ordinary module constructors should receive
/// [`ModuleCapabilityFactory`] so they cannot choose another owner.
#[derive(Clone)]
pub struct CapabilityProviders {
    inner: Rc<CapabilityProvidersInner>,
}

struct CapabilityProvidersInner {
    blackboard: Blackboard,
    wakes: WakeRegistry,
    self_wake_permits: SelfWakePermitRegistry,
    attention_control_requests: Topic<AttentionControlRequest>,
    cognition_log_updates: Topic<CognitionLogUpdated>,
    cognition_log_evictions: Topic<nuillu_blackboard::CognitionLogEntryRecord>,
    interoception_updates: Topic<InteroceptiveUpdated>,
    action_affordance_updates: Topic<ActionAffordancesUpdated>,
    memo_updates: Topic<MemoUpdated>,
    memo_log_evictions: Topic<nuillu_blackboard::MemoLogRecord>,
    sensory_input_topic: Topic<SensoryInput>,
    role_reader_cursors: RoleReaderCursors,
    activation_gates: ActivationGateHub,
    cognition_log_port: Rc<dyn CognitionLogRepository>,
    clock: Rc<dyn Clock>,
    timer: Rc<dyn Timer>,
    time_division: TimeDivision,
    tiers: LutumTiers,
    runtime_events: RuntimeEventEmitter,
    runtime_policy: RuntimePolicy,
    scene: SceneRegistry,
    action_affordances: ActionAffordanceRegistry,
    external_action_executor: Rc<dyn ExternalActionExecutor>,
    session_store: Rc<dyn SessionStore>,
    allocation_store: Rc<dyn AllocationStore>,
    memo_log_repository: Rc<dyn MemoLogRepository>,
    module_catalog: OnceLock<ModuleCatalog>,
    scope_labels: Rc<RefCell<Rc<ScopeLabels>>>,
}

/// Owner-stamped handle for requesting another scheduler pass for the holder.
#[derive(Clone)]
pub struct SelfWake {
    owner: ModuleInstanceId,
    permits: SelfWakePermitRegistry,
}

impl SelfWake {
    fn new(owner: ModuleInstanceId, permits: SelfWakePermitRegistry) -> Self {
        Self { owner, permits }
    }

    /// Mark this module owner as having pending work.
    pub fn wake(&self) {
        self.permits.issue(&self.owner);
    }
}

/// Claim for one owner-stamped self-wake scheduling opportunity.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SelfWakePermitClaim {
    owner: ModuleInstanceId,
    delivered_through: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WakeChangeSequence {
    wake: u64,
    self_wake_permit: u64,
}

#[derive(Clone)]
struct SelfWakePermitRegistry {
    inner: Rc<RefCell<SelfWakePermitRegistryInner>>,
    notify: Rc<tokio::sync::Notify>,
}

#[derive(Default)]
struct SelfWakePermitRegistryInner {
    delivered_by_owner: HashMap<ModuleInstanceId, u64>,
    completed_by_owner: HashMap<ModuleInstanceId, u64>,
    change_sequence: u64,
}

impl Default for SelfWakePermitRegistry {
    fn default() -> Self {
        Self {
            inner: Rc::new(RefCell::new(SelfWakePermitRegistryInner::default())),
            notify: Rc::new(tokio::sync::Notify::new()),
        }
    }
}

impl SelfWakePermitRegistry {
    fn issue(&self, owner: &ModuleInstanceId) {
        {
            let mut inner = self.inner.borrow_mut();
            let next = inner
                .delivered_by_owner
                .get(owner)
                .copied()
                .unwrap_or_default()
                .saturating_add(1);
            inner.delivered_by_owner.insert(owner.clone(), next);
            inner.change_sequence = inner.change_sequence.saturating_add(1);
        }
        self.notify.notify_waiters();
    }

    fn claim(&self, owner: &ModuleInstanceId) -> Option<SelfWakePermitClaim> {
        let inner = self.inner.borrow();
        let delivered = inner
            .delivered_by_owner
            .get(owner)
            .copied()
            .unwrap_or_default();
        let completed = inner
            .completed_by_owner
            .get(owner)
            .copied()
            .unwrap_or_default();
        (delivered > completed).then(|| SelfWakePermitClaim {
            owner: owner.clone(),
            delivered_through: delivered,
        })
    }

    fn complete(&self, claim: SelfWakePermitClaim) {
        let mut inner = self.inner.borrow_mut();
        let delivered = inner
            .delivered_by_owner
            .get(&claim.owner)
            .copied()
            .unwrap_or_default();
        let completed = claim.delivered_through.min(delivered);
        inner
            .completed_by_owner
            .entry(claim.owner)
            .and_modify(|current| *current = (*current).max(completed))
            .or_insert(completed);
    }

    fn has_pending(&self, owner: &ModuleInstanceId) -> bool {
        let inner = self.inner.borrow();
        let delivered = inner
            .delivered_by_owner
            .get(owner)
            .copied()
            .unwrap_or_default();
        let completed = inner
            .completed_by_owner
            .get(owner)
            .copied()
            .unwrap_or_default();
        delivered > completed
    }

    fn change_sequence(&self) -> u64 {
        self.inner.borrow().change_sequence
    }

    async fn changed_since(&self, observed: u64) {
        loop {
            let notified = self.notify.notified();
            if self.change_sequence() > observed {
                return;
            }
            notified.await;
        }
    }
}

/// Required external services for the root capability provider set.
#[derive(Clone)]
pub struct CapabilityProviderPorts {
    pub blackboard: Blackboard,
    pub cognition_log_port: Rc<dyn CognitionLogRepository>,
    pub clock: Rc<dyn Clock>,
    pub tiers: LutumTiers,
}

/// Runtime policy and observation hooks layered on top of the boot ports.
#[derive(Clone)]
pub struct CapabilityProviderRuntime {
    pub timer: Rc<dyn Timer>,
    pub event_sink: Rc<dyn RuntimeEventSink>,
    pub policy: RuntimePolicy,
    pub session_store: Rc<dyn SessionStore>,
    pub allocation_store: Rc<dyn AllocationStore>,
    pub memo_log_repository: Rc<dyn MemoLogRepository>,
    pub external_action_executor: Rc<dyn ExternalActionExecutor>,
}

impl Default for CapabilityProviderRuntime {
    fn default() -> Self {
        Self {
            timer: Rc::new(TokioTimer::new()),
            event_sink: Rc::new(NoopRuntimeEventSink),
            policy: RuntimePolicy::default(),
            session_store: Rc::new(NoopSessionStore),
            allocation_store: Rc::new(NoopAllocationStore),
            memo_log_repository: Rc::new(NoopMemoLogRepository),
            external_action_executor: Rc::new(NoopExternalActionExecutor),
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
        let CapabilityProviderRuntime {
            timer,
            event_sink,
            policy,
            session_store,
            allocation_store,
            memo_log_repository,
            external_action_executor,
        } = runtime;
        let runtime_events = RuntimeEventEmitter::new(event_sink);
        let wakes = WakeRegistry::default();
        let self_wake_permits = SelfWakePermitRegistry::default();
        let role_reader_cursors = RoleReaderCursors::default();
        Self {
            inner: Rc::new(CapabilityProvidersInner {
                wakes: wakes.clone(),
                self_wake_permits: self_wake_permits.clone(),
                attention_control_requests: Topic::new(
                    blackboard.clone(),
                    wakes.clone(),
                    TopicPolicy::RoleLoadBalanced,
                ),
                cognition_log_updates: Topic::new(
                    blackboard.clone(),
                    wakes.clone(),
                    TopicPolicy::Fanout,
                ),
                cognition_log_evictions: Topic::new(
                    blackboard.clone(),
                    wakes.clone(),
                    TopicPolicy::Fanout,
                ),
                interoception_updates: Topic::new(
                    blackboard.clone(),
                    wakes.clone(),
                    TopicPolicy::Fanout,
                ),
                action_affordance_updates: Topic::new(
                    blackboard.clone(),
                    wakes.clone(),
                    TopicPolicy::Fanout,
                ),
                memo_updates: Topic::new(blackboard.clone(), wakes.clone(), TopicPolicy::Fanout),
                memo_log_evictions: Topic::new(
                    blackboard.clone(),
                    wakes.clone(),
                    TopicPolicy::Fanout,
                ),
                sensory_input_topic: Topic::new(
                    blackboard.clone(),
                    wakes,
                    TopicPolicy::RoleLoadBalanced,
                ),
                role_reader_cursors,
                activation_gates: ActivationGateHub::new(blackboard.clone()),
                blackboard,
                cognition_log_port,
                clock,
                timer,
                time_division: TimeDivision::default(),
                tiers,
                runtime_events,
                runtime_policy: policy,
                scene: SceneRegistry::empty(),
                action_affordances: ActionAffordanceRegistry::default(),
                external_action_executor,
                session_store,
                allocation_store,
                memo_log_repository,
                module_catalog: OnceLock::new(),
                scope_labels: Rc::new(RefCell::new(Rc::new(ScopeLabels::default()))),
            }),
        }
    }

    /// Install boot-time subsystem display metadata before the agent starts.
    pub fn set_scope_labels(&self, labels: ScopeLabels) {
        *self.inner.scope_labels.borrow_mut() = Rc::new(labels);
    }

    /// Access the scene registry for host-driven participant updates.
    ///
    /// The host (eval harness, game runtime) calls `scene().set(...)` to
    /// declare which participants are currently in earshot. The agent only
    /// reads the registry, never mutates it.
    pub fn scene(&self) -> &SceneRegistry {
        &self.inner.scene
    }

    /// Test-only unscoped factory. Production wiring goes through
    /// [`Self::scoped_with_memo_subscription`] so a module never silently
    /// loses the memo scope its registration declared.
    #[cfg(test)]
    pub(crate) fn scoped(&self, owner: ModuleInstanceId) -> ModuleCapabilityFactory {
        self.scoped_with_memo_subscription(owner, MemoSubscription::All)
    }

    fn scoped_with_memo_subscription(
        &self,
        owner: ModuleInstanceId,
        memo_subscription: MemoSubscription,
    ) -> ModuleCapabilityFactory {
        let blackboard = self.inner.blackboard.scoped(owner.scope.clone());
        ModuleCapabilityFactory {
            owner,
            root: self.clone(),
            blackboard,
            memo_issued: Rc::new(Cell::new(false)),
            outer_memo_issued: Rc::new(Cell::new(false)),
            memo_subscription,
        }
    }

    pub(crate) fn set_module_contexts(
        &self,
        scope: ScopeId,
        peer_contexts: Vec<(ModuleId, Arc<str>)>,
    ) {
        self.inner
            .blackboard
            .scoped(scope)
            .set_module_contexts(peer_contexts);
    }

    fn install_module_catalog(&self, catalog: ModuleCatalog) -> Result<(), ModuleRegistryError> {
        if let Some(installed) = self.inner.module_catalog.get() {
            if installed == &catalog {
                return Ok(());
            }
            return Err(ModuleRegistryError::CatalogChangedAfterBoot);
        }
        self.inner
            .module_catalog
            .set(catalog)
            .map_err(|_| ModuleRegistryError::CatalogChangedAfterBoot)
    }

    async fn set_registered_modules(
        &self,
        scope: ScopeId,
        registrations: Vec<RegisteredModulePolicy>,
    ) {
        self.inner
            .blackboard
            .scoped(scope)
            .apply(BlackboardCommand::SetRegisteredModules { registrations })
            .await;
    }

    async fn set_registered_subsystems(
        &self,
        scope: ScopeId,
        registrations: Vec<RegisteredSubsystemPolicy>,
    ) {
        self.inner
            .blackboard
            .scoped(scope)
            .apply(BlackboardCommand::SetRegisteredSubsystems { registrations })
            .await;
    }

    pub(crate) async fn apply_runtime_policy(&self, scope: ScopeId) {
        let blackboard = self.inner.blackboard.scoped(scope);
        blackboard
            .apply(BlackboardCommand::SetAllocationLimits(
                self.inner.runtime_policy.allocation_limits,
            ))
            .await;
        blackboard
            .apply(BlackboardCommand::SetMemoRetentionPerOwner(
                self.inner.runtime_policy.memo_retained_per_owner,
            ))
            .await;
        blackboard
            .apply(BlackboardCommand::SetCognitionLogRetentionEntries(
                self.inner.runtime_policy.cognition_log_retained_entries,
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
            wakes: self.inner.wakes.clone(),
            self_wake_permits: self.inner.self_wake_permits.clone(),
            cognition_log_updates: CognitionLogUpdatedMailbox::new(
                owner,
                self.inner.cognition_log_updates.clone(),
            ),
            clock: self.inner.clock.clone(),
            session_compaction: self.inner.tiers.cheap.clone(),
            session_compaction_policy: self.inner.runtime_policy.session_compaction,
            runtime_events: self.inner.runtime_events.clone(),
            activation_gates: self.inner.activation_gates.clone(),
            session_store: self.inner.session_store.clone(),
            scope_labels: self.inner.scope_labels.clone(),
        }
    }

    pub fn blackboard_reader(&self) -> BlackboardReader {
        BlackboardReader::new(self.inner.blackboard.clone())
    }

    pub fn memory_metadata_reader(&self) -> MemoryMetadataReader {
        MemoryMetadataReader::new(self.inner.blackboard.clone())
    }

    pub fn cognition_log_reader(&self) -> CognitionLogReader {
        CognitionLogReader::new(self.inner.blackboard.clone())
    }

    pub fn allocation_reader(&self) -> AllocationReader {
        AllocationReader::new(self.inner.blackboard.clone())
    }

    pub async fn restore_allocation_snapshots(&self) -> Result<usize, PortError> {
        let snapshots = self.inner.allocation_store.load_all().await?;
        let count = snapshots.len();
        for snapshot in snapshots {
            snapshot.validate_version()?;
            let blackboard = self.inner.blackboard.scoped(snapshot.owner.scope.clone());
            blackboard
                .apply(BlackboardCommand::RecordAllocationEffects {
                    writer: snapshot.owner,
                    targets: snapshot.targets,
                    suppressions: snapshot.suppressions,
                })
                .await;
        }
        Ok(count)
    }

    pub async fn restore_cognition_log_entries(&self) -> Result<usize, PortError> {
        let records = self
            .inner
            .cognition_log_port
            .recent(self.inner.runtime_policy.cognition_log_retained_entries)
            .await?;
        let mut restorable_scopes = BTreeSet::new();
        for record in &records {
            let scope = record.scope.clone();
            let blackboard = self.inner.blackboard.scoped(scope.clone());
            if blackboard.read(|bb| bb.cognition_log().is_empty()).await {
                restorable_scopes.insert(scope);
            }
        }
        let mut count = 0;
        for record in records {
            if restorable_scopes.contains(&record.scope) {
                self.inner
                    .blackboard
                    .scoped(record.scope)
                    .apply(BlackboardCommand::AppendCognitionLog {
                        source: record.source,
                        entry: record.entry,
                    })
                    .await;
                count += 1;
            }
        }
        Ok(count)
    }

    pub async fn restore_memo_log_entries(&self) -> Result<usize, PortError> {
        let retained_per_owner = self
            .inner
            .blackboard
            .read(|bb| bb.memo_retained_per_owner())
            .await;
        let records = self
            .inner
            .memo_log_repository
            .recent_per_owner(retained_per_owner)
            .await?;
        let count = records.len();
        for PersistedMemoLogEntry {
            scope,
            record,
            payload,
        } in records
        {
            let blackboard = self.inner.blackboard.scoped(scope);
            blackboard.restore_memo_log_entry(record, payload).await;
        }
        Ok(count)
    }

    pub fn module_status_reader(&self) -> ModuleStatusReader {
        ModuleStatusReader::new(self.inner.blackboard.clone())
    }

    pub fn clock(&self) -> Rc<dyn Clock> {
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
    wakes: WakeRegistry,
    self_wake_permits: SelfWakePermitRegistry,
    cognition_log_updates: CognitionLogUpdatedMailbox,
    clock: Rc<dyn Clock>,
    session_compaction: LlmTierHandle,
    session_compaction_policy: SessionCompactionPolicy,
    runtime_events: RuntimeEventEmitter,
    activation_gates: ActivationGateHub,
    session_store: Rc<dyn SessionStore>,
    scope_labels: Rc<RefCell<Rc<ScopeLabels>>>,
}

impl AgentRuntimeControl {
    pub fn has_pending_wake(&self, owner: &ModuleInstanceId) -> bool {
        self.wakes.has_pending_wake(owner)
    }

    pub fn has_pending_self_wake_permit(&self, owner: &ModuleInstanceId) -> bool {
        self.self_wake_permits.has_pending(owner)
    }

    pub fn claim_self_wake_permit(&self, owner: &ModuleInstanceId) -> Option<SelfWakePermitClaim> {
        self.self_wake_permits.claim(owner)
    }

    pub fn complete_self_wake_permit_claim(&self, claim: SelfWakePermitClaim) {
        self.self_wake_permits.complete(claim);
    }

    pub fn claim_wake(&self, owner: &ModuleInstanceId) -> Option<WakeClaim> {
        self.wakes.claim_wake(owner)
    }

    pub fn complete_wake_claim(&self, claim: WakeClaim) {
        self.wakes.complete_wake_claim(claim);
    }

    pub fn wake_change_sequence(&self) -> WakeChangeSequence {
        WakeChangeSequence {
            wake: self.wakes.change_sequence(),
            self_wake_permit: self.self_wake_permits.change_sequence(),
        }
    }

    pub async fn wake_changed_since(&self, observed: WakeChangeSequence) {
        tokio::select! {
            _ = self.wakes.changed_since(observed.wake) => {},
            _ = self.self_wake_permits.changed_since(observed.self_wake_permit) => {},
        }
    }

    pub async fn is_active(&self, owner: &ModuleInstanceId) -> bool {
        let scope = self.blackboard.scope_activation_state(&owner.scope).await;
        scope.active
            && self
                .blackboard
                .effective_module_allocation(&owner.scope)
                .await
                .is_replica_active(owner)
    }

    pub async fn is_forced_disabled(&self, owner: &ModuleInstanceId) -> bool {
        let globally_disabled = self
            .blackboard
            .read(|bb| bb.forced_disabled_modules().contains(&owner.module))
            .await;
        let scope_inactive = !self
            .blackboard
            .scope_activation_state(&owner.scope)
            .await
            .active;
        globally_disabled
            || scope_inactive
            || self
                .blackboard
                .scoped(owner.scope.clone())
                .read(|bb| bb.forced_disabled_modules().contains(&owner.module))
                .await
    }

    pub fn clock(&self) -> Rc<dyn Clock> {
        self.clock.clone()
    }

    pub fn session_compaction_handle(&self) -> &LlmTierHandle {
        &self.session_compaction
    }

    pub fn session_compaction_lutum(&self) -> &Lutum {
        &self.session_compaction.lutum
    }

    pub fn session_compaction_policy(&self) -> SessionCompactionPolicy {
        self.session_compaction_policy
    }

    /// Snapshot of the registered-module peer-context catalog. Cheap
    /// synchronous read; the scheduler turns this into an [`ActivateCx`] for
    /// each `activate` call.
    pub fn peer_contexts(&self, owner: &ModuleInstanceId) -> Vec<(ModuleId, Arc<str>)> {
        self.blackboard
            .scoped(owner.scope.clone())
            .peer_contexts()
            .to_vec()
    }

    pub async fn identity_memories(
        &self,
        owner: &ModuleInstanceId,
    ) -> Vec<nuillu_blackboard::IdentityMemoryRecord> {
        self.blackboard
            .scoped(owner.scope.clone())
            .read(|bb| bb.identity_memories().to_vec())
            .await
    }

    pub async fn core_policies(&self) -> Vec<nuillu_blackboard::CorePolicyRecord> {
        self.blackboard.read(|bb| bb.core_policies().to_vec()).await
    }

    pub async fn record_module_status(&self, owner: ModuleInstanceId, status: ModuleRunStatus) {
        self.blackboard
            .scoped(owner.scope.clone())
            .apply(BlackboardCommand::SetModuleRunStatus { owner, status })
            .await;
    }

    pub async fn module_batch_throttle_baseline(&self, owner: &ModuleInstanceId) -> Option<Bpm> {
        let allocation = self
            .blackboard
            .effective_module_allocation(&owner.scope)
            .await;
        allocation.bpm_for(&owner.module)
    }

    pub async fn active_replicas(&self, module: &ScopedModuleId) -> u8 {
        if !self
            .blackboard
            .scope_activation_state(&module.scope)
            .await
            .active
        {
            return 0;
        }
        self.blackboard
            .effective_module_allocation(&module.scope)
            .await
            .active_replicas(&module.module)
    }

    pub async fn zero_replica_window_policies(
        &self,
    ) -> HashMap<ScopedModuleId, ZeroReplicaWindowPolicy> {
        let mut policies = HashMap::new();
        for blackboard in self.blackboard.all_scopes() {
            let scope = blackboard.scope().clone();
            policies.extend(
                blackboard
                    .read(|bb| {
                        bb.module_policies()
                            .iter()
                            .filter_map(|(module, policy)| {
                                (policy.replicas_range.max > 0
                                    && policy
                                        .zero_replica_window
                                        .controller_activation_period()
                                        .is_some())
                                .then_some((
                                    ScopedModuleId::new(scope.clone(), module.clone()),
                                    policy.zero_replica_window,
                                ))
                            })
                            .collect::<Vec<_>>()
                    })
                    .await,
            );
        }
        policies
    }

    pub async fn activation_waiter(
        &self,
        owner: &ModuleInstanceId,
    ) -> Option<tokio::sync::oneshot::Receiver<()>> {
        if !self
            .blackboard
            .scope_activation_state(&owner.scope)
            .await
            .active
        {
            return Some(self.blackboard.allocation_change_waiter().await);
        }
        self.blackboard
            .scoped(owner.scope.clone())
            .activation_waiter(owner.clone())
            .await
    }

    pub async fn allocation_change_waiter(&self) -> tokio::sync::oneshot::Receiver<()> {
        self.blackboard.allocation_change_waiter().await
    }

    pub fn record_module_batch_throttled(&self, owner: ModuleInstanceId, delayed_for: Duration) {
        self.runtime_events
            .module_batch_throttled(owner, delayed_for);
    }

    pub fn next_module_activation_id(&self) -> ModuleActivationId {
        self.runtime_events.next_module_activation_id()
    }

    pub fn record_module_batch_ready(
        &self,
        activation_id: ModuleActivationId,
        activation_attempt: u32,
        owner: ModuleInstanceId,
        batch: &ModuleBatch,
    ) {
        self.runtime_events
            .module_batch_ready(activation_id, activation_attempt, owner, batch);
    }

    pub fn with_session_checkpoint_runtime<'a>(
        &self,
        cx: crate::ActivateCx<'a>,
        owner: ModuleInstanceId,
    ) -> crate::ActivateCx<'a> {
        cx.with_scope_labels(self.scope_labels.borrow().clone())
            .with_session_checkpoint_runtime(
                self.session_store.clone(),
                self.runtime_events.clone(),
                owner,
            )
    }

    pub async fn delete_module_sessions(&self, owner: &ModuleInstanceId) -> Result<u64, PortError> {
        self.session_store.delete_owner(owner).await
    }

    pub fn record_module_activation_completed(
        &self,
        activation_id: ModuleActivationId,
        owner: ModuleInstanceId,
        duration: Duration,
        succeeded: bool,
    ) {
        self.runtime_events
            .module_activation_completed(activation_id, owner, duration, succeeded);
    }

    pub fn record_module_warning(&self, owner: ModuleInstanceId, message: String) {
        self.runtime_events.module_warning(owner, message);
    }

    pub fn record_module_activation_attempt_failed(
        &self,
        activation_id: ModuleActivationId,
        owner: ModuleInstanceId,
        activation_attempt: u32,
        max_attempts: u32,
        message: impl Into<String>,
    ) {
        self.runtime_events.module_activation_attempt_failed(
            activation_id,
            owner,
            activation_attempt,
            max_attempts,
            message.into(),
        );
    }

    pub fn record_module_task_failed(
        &self,
        owner: ModuleInstanceId,
        phase: impl Into<String>,
        message: impl Into<String>,
    ) {
        self.runtime_events
            .module_task_failed(owner, phase.into(), message.into());
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

    pub fn action_affordance_writer(&self) -> ActionAffordanceWriter {
        ActionAffordanceWriter::new(
            self.root.inner.action_affordances.clone(),
            ActionAffordancesUpdatedMailbox::new(
                self.owner.clone(),
                self.root.inner.action_affordance_updates.clone(),
            ),
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

    pub fn cognition_log_evicted_mailbox(&self) -> CognitionLogEvictedMailbox {
        TopicMailbox::new(
            self.owner.clone(),
            self.root.inner.cognition_log_evictions.clone(),
        )
    }

    pub fn interoception_updated_mailbox(&self) -> InteroceptiveUpdatedMailbox {
        TopicMailbox::new(
            self.owner.clone(),
            self.root.inner.interoception_updates.clone(),
        )
    }

    pub fn memo_updated_mailbox(&self) -> MemoUpdatedMailbox {
        TopicMailbox::new(self.owner.clone(), self.root.inner.memo_updates.clone())
    }

    pub fn memo_log_evicted_mailbox(&self) -> MemoLogEvictedMailbox {
        TopicMailbox::new(
            self.owner.clone(),
            self.root.inner.memo_log_evictions.clone(),
        )
    }
}

#[derive(Clone)]
pub struct ModuleCapabilityFactory {
    owner: ModuleInstanceId,
    root: CapabilityProviders,
    blackboard: Blackboard,
    // Memo is the only single-issued capability: typed memo safety relies on
    // one payload type per module owner.
    memo_issued: Rc<Cell<bool>>,
    outer_memo_issued: Rc<Cell<bool>>,
    memo_subscription: MemoSubscription,
}

impl ModuleCapabilityFactory {
    /// The owner this factory dispenses capabilities for. Capability handles
    /// returned by this factory are stamped with this id.
    pub fn owner(&self) -> &ModuleInstanceId {
        &self.owner
    }

    /// The immutable catalog compiled from all registrations before any
    /// module constructor runs.
    pub fn module_catalog(&self) -> ModuleCatalogView<'_> {
        let catalog = self
            .root
            .inner
            .module_catalog
            .get()
            .expect("module catalog is installed before module construction");
        ModuleCatalogView {
            catalog,
            scope: &self.owner.scope,
        }
    }

    pub fn subsystem_catalog(&self) -> SubsystemCatalogView<'_> {
        SubsystemCatalogView {
            catalog: self
                .root
                .inner
                .module_catalog
                .get()
                .expect("module catalog installed before module construction"),
            scope: &self.owner.scope,
        }
    }

    pub fn self_wake(&self) -> SelfWake {
        SelfWake::new(
            self.owner.clone(),
            self.root.inner.self_wake_permits.clone(),
        )
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
        let role_reader_cursors = self.root.inner.role_reader_cursors.clone();
        let scope = self.owner.scope.clone();
        TopicInbox::new_excluding_self_with_round_robin_hook(
            self.owner.clone(),
            self.root.inner.cognition_log_updates.clone(),
            Some(Rc::new(move |role| {
                role_reader_cursors.enable_cognition_round_robin(
                    &ScopedModuleId::new(scope.clone(), role.clone()),
                    &scope,
                );
            })),
        )
    }

    pub fn outer_cognition_log_updated_inbox(&self) -> Option<CognitionLogUpdatedInbox> {
        let outer_scope = self.owner.scope.parent()?;
        let role_reader_cursors = self.root.inner.role_reader_cursors.clone();
        let owner_role = self.owner.scoped_module();
        let cursor_scope = outer_scope.clone();
        Some(TopicInbox::new_excluding_self_in_scope(
            self.owner.clone(),
            outer_scope,
            self.root.inner.cognition_log_updates.clone(),
            Some(Rc::new(move |_role| {
                role_reader_cursors.enable_cognition_round_robin(&owner_role, &cursor_scope);
            })),
        ))
    }

    pub fn cognition_log_evicted_inbox(&self) -> CognitionLogEvictedInbox {
        TopicInbox::new_excluding_self(
            self.owner.clone(),
            self.root.inner.cognition_log_evictions.clone(),
        )
    }

    pub fn interoception_updated_inbox(&self) -> InteroceptiveUpdatedInbox {
        TopicInbox::new_excluding_self(
            self.owner.clone(),
            self.root.inner.interoception_updates.clone(),
        )
    }

    pub fn action_affordances_updated_inbox(&self) -> ActionAffordancesUpdatedInbox {
        TopicInbox::new_excluding_self(
            self.owner.clone(),
            self.root.inner.action_affordance_updates.clone(),
        )
    }

    pub fn memo_updated_inbox(&self) -> MemoUpdatedInbox {
        let role_reader_cursors = self.root.inner.role_reader_cursors.clone();
        let scope = self.owner.scope.clone();
        TopicInbox::new_excluding_self_with_round_robin_hook_and_sources(
            self.owner.clone(),
            self.root.inner.memo_updates.clone(),
            Some(Rc::new(move |role| {
                role_reader_cursors
                    .enable_memo_round_robin(&ScopedModuleId::new(scope.clone(), role.clone()));
            })),
            self.memo_subscription.clone(),
        )
    }

    pub fn memo_log_evicted_inbox(&self) -> MemoLogEvictedInbox {
        TopicInbox::new_excluding_self(
            self.owner.clone(),
            self.root.inner.memo_log_evictions.clone(),
        )
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

    pub fn activation_gate_for<M: Module + 'static>(
        &self,
        target: ModuleId,
    ) -> crate::ActivationGate<M> {
        self.root
            .inner
            .activation_gates
            .subscribe::<M>(self.owner.clone(), target)
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
            self.owner.scope.clone(),
            self.blackboard.clone(),
            self.root.inner.memo_log_repository.clone(),
            TopicMailbox::new(self.owner.clone(), self.root.inner.memo_updates.clone()),
            TopicMailbox::new(
                self.owner.clone(),
                self.root.inner.memo_log_evictions.clone(),
            ),
            self.root.inner.clock.clone(),
            self.root.inner.runtime_events.clone(),
        )
    }

    pub fn typed_memo<T>(&self) -> TypedMemo<T>
    where
        T: serde::Serialize + serde::de::DeserializeOwned + 'static,
    {
        self.claim_memo();
        TypedMemo::new(
            self.owner.clone(),
            self.owner.scope.clone(),
            self.blackboard.clone(),
            self.root.inner.memo_log_repository.clone(),
            TopicMailbox::new(self.owner.clone(), self.root.inner.memo_updates.clone()),
            TopicMailbox::new(
                self.owner.clone(),
                self.root.inner.memo_log_evictions.clone(),
            ),
            self.root.inner.clock.clone(),
            self.root.inner.runtime_events.clone(),
        )
    }

    fn claim_outer_memo(&self) {
        assert!(
            !self.outer_memo_issued.replace(true),
            "module requested multiple outer memo capabilities; choose exactly one of outer_memo() or outer_typed_memo::<T>()"
        );
    }

    /// Plaintext memo written into the immediate parent scope while retaining
    /// the holder's owner stamp. Returns `None` for a root-scoped module.
    pub fn outer_memo(&self) -> Option<Memo> {
        let outer_scope = self.owner.scope.parent()?;
        self.claim_outer_memo();
        Some(Memo::new(
            self.owner.clone(),
            outer_scope.clone(),
            self.root.inner.blackboard.scoped(outer_scope.clone()),
            self.root.inner.memo_log_repository.clone(),
            TopicMailbox::new_in_scope(
                self.owner.clone(),
                outer_scope.clone(),
                self.root.inner.memo_updates.clone(),
            ),
            TopicMailbox::new_in_scope(
                self.owner.clone(),
                outer_scope,
                self.root.inner.memo_log_evictions.clone(),
            ),
            self.root.inner.clock.clone(),
            self.root.inner.runtime_events.clone(),
        ))
    }

    /// Typed memo written into the immediate parent scope while retaining the
    /// holder's owner stamp. Returns `None` for a root-scoped module.
    pub fn outer_typed_memo<T>(&self) -> Option<TypedMemo<T>>
    where
        T: serde::Serialize + serde::de::DeserializeOwned + 'static,
    {
        let outer_scope = self.owner.scope.parent()?;
        self.claim_outer_memo();
        Some(TypedMemo::new(
            self.owner.clone(),
            outer_scope.clone(),
            self.root.inner.blackboard.scoped(outer_scope.clone()),
            self.root.inner.memo_log_repository.clone(),
            TopicMailbox::new_in_scope(
                self.owner.clone(),
                outer_scope.clone(),
                self.root.inner.memo_updates.clone(),
            ),
            TopicMailbox::new_in_scope(
                self.owner.clone(),
                outer_scope,
                self.root.inner.memo_log_evictions.clone(),
            ),
            self.root.inner.clock.clone(),
            self.root.inner.runtime_events.clone(),
        ))
    }

    pub fn llm(&self, key: impl Into<String>) -> LlmCapabilityRequest {
        LlmCapabilityRequest {
            owner: self.owner.clone(),
            root: self.root.clone(),
            key: key.into(),
            tier: ModelTier::Default,
        }
    }

    pub fn session(&self, key: impl Into<String>) -> SessionCapabilityRequest {
        SessionCapabilityRequest {
            owner: self.owner.clone(),
            root: self.root.clone(),
            key: key.into(),
            tier: ModelTier::Default,
            auto_compaction: None,
        }
    }

    pub fn blackboard_reader(&self) -> BlackboardReader {
        BlackboardReader::new_for_owner_with_role_cursors(
            self.blackboard.clone(),
            self.owner.clone(),
            self.root.inner.role_reader_cursors.clone(),
            self.memo_subscription.clone(),
        )
    }

    pub fn memory_metadata_reader(&self) -> MemoryMetadataReader {
        MemoryMetadataReader::new(self.blackboard.clone())
    }

    /// Raw [`Blackboard`] handle. Domain crates use this to build their own
    /// owner-stamped capability handles outside of `nuillu-module`.
    pub fn blackboard(&self) -> Blackboard {
        self.blackboard.clone()
    }

    pub fn cognition_log_reader(&self) -> CognitionLogReader {
        CognitionLogReader::new_for_owner_with_role_cursors(
            self.blackboard.clone(),
            self.owner.clone(),
            self.root.inner.role_reader_cursors.clone(),
        )
    }

    pub fn outer_cognition_log_reader(&self) -> Option<CognitionLogReader> {
        let outer_scope = self.owner.scope.parent()?;
        Some(CognitionLogReader::new_for_owner_with_role_cursors(
            self.root.inner.blackboard.scoped(outer_scope),
            self.owner.clone(),
            self.root.inner.role_reader_cursors.clone(),
        ))
    }

    pub fn allocation_reader(&self) -> AllocationReader {
        AllocationReader::new(self.blackboard.clone())
    }

    pub fn interoception_reader(&self) -> InteroceptiveReader {
        InteroceptiveReader::new(self.blackboard.clone())
    }

    pub fn module_status_reader(&self) -> ModuleStatusReader {
        ModuleStatusReader::new(self.blackboard.clone())
    }

    pub fn cognition_writer(&self) -> CognitionWriter {
        CognitionWriter::new(
            self.owner.clone(),
            self.blackboard.clone(),
            self.root.inner.cognition_log_port.clone(),
            CognitionLogUpdatedMailbox::new(
                self.owner.clone(),
                self.root.inner.cognition_log_updates.clone(),
            ),
            CognitionLogEvictedMailbox::new(
                self.owner.clone(),
                self.root.inner.cognition_log_evictions.clone(),
            ),
            self.root.inner.clock.clone(),
        )
    }

    pub fn outer_cognition_writer(&self) -> Option<CognitionWriter> {
        let outer_scope = self.owner.scope.parent()?;
        Some(CognitionWriter::new_in_scope(
            self.owner.clone(),
            outer_scope.clone(),
            self.root.inner.blackboard.scoped(outer_scope.clone()),
            self.root.inner.cognition_log_port.clone(),
            CognitionLogUpdatedMailbox::new_in_scope(
                self.owner.clone(),
                outer_scope.clone(),
                self.root.inner.cognition_log_updates.clone(),
            ),
            CognitionLogEvictedMailbox::new_in_scope(
                self.owner.clone(),
                outer_scope,
                self.root.inner.cognition_log_evictions.clone(),
            ),
            self.root.inner.clock.clone(),
        ))
    }

    pub fn allocation_writer(
        &self,
        allowed_target_modules: Vec<ModuleId>,
        allowed_suppression_modules: Vec<ModuleId>,
    ) -> AllocationWriter {
        AllocationWriter::new(
            self.owner.clone(),
            self.blackboard.clone(),
            allowed_target_modules,
            allowed_suppression_modules,
            self.root.inner.runtime_policy.allocation_effects.clone(),
            self.root.inner.allocation_store.clone(),
        )
    }

    pub fn subsystem_allocation_reader(&self) -> SubsystemAllocationReader {
        SubsystemAllocationReader::new(self.blackboard.clone())
    }

    pub fn subsystem_allocation_writer(
        &self,
        activation_tables: Vec<(SubsystemId, Vec<ActivationRatio>)>,
    ) -> SubsystemAllocationWriter {
        SubsystemAllocationWriter::new(
            self.owner.clone(),
            self.blackboard.clone(),
            activation_tables,
        )
    }

    pub fn interoception_policy(&self) -> InteroceptionRuntimePolicy {
        self.root.inner.runtime_policy.interoception.clone()
    }

    pub fn interoception_writer(&self) -> InteroceptiveWriter {
        InteroceptiveWriter::new(
            self.owner.clone(),
            self.blackboard.clone(),
            InteroceptiveUpdatedMailbox::new(
                self.owner.clone(),
                self.root.inner.interoception_updates.clone(),
            ),
            self.root.inner.clock.clone(),
        )
    }

    pub fn action_affordance_reader(&self) -> ActionAffordanceReader {
        ActionAffordanceReader::new(self.root.inner.action_affordances.clone())
    }

    pub fn external_action_invoker(&self) -> ExternalActionInvoker {
        ExternalActionInvoker::new(
            self.owner.clone(),
            self.root.inner.external_action_executor.clone(),
        )
    }

    pub fn scene_reader(&self) -> SceneReader {
        SceneReader::new(self.root.inner.scene.clone())
    }

    pub fn clock(&self) -> Rc<dyn Clock> {
        self.root.clock()
    }

    pub fn timer(&self) -> Rc<dyn Timer> {
        self.root.inner.timer.clone()
    }

    pub fn time_division(&self) -> TimeDivision {
        self.root.time_division()
    }
}

pub struct LlmCapabilityRequest {
    owner: ModuleInstanceId,
    root: CapabilityProviders,
    key: String,
    tier: ModelTier,
}

impl LlmCapabilityRequest {
    pub fn with_tier(mut self, tier: ModelTier) -> Self {
        self.tier = tier;
        self
    }
}

impl From<LlmCapabilityRequest> for LlmAccess {
    fn from(request: LlmCapabilityRequest) -> Self {
        LlmAccess::new(
            request.owner,
            request.key,
            request.tier,
            request.root.inner.tiers.clone(),
            request.root.inner.runtime_events.clone(),
        )
    }
}

pub struct SessionCapabilityRequest {
    owner: ModuleInstanceId,
    root: CapabilityProviders,
    key: String,
    tier: ModelTier,
    auto_compaction: Option<SessionAutoCompaction>,
}

impl SessionCapabilityRequest {
    pub fn with_tier(mut self, tier: ModelTier) -> Self {
        self.tier = tier;
        self
    }

    pub fn with_auto_compaction(mut self, auto_compaction: SessionAutoCompaction) -> Self {
        self.auto_compaction = Some(auto_compaction);
        self
    }
}

impl IntoFuture for SessionCapabilityRequest {
    type Output = Result<Session, ModuleRegistryError>;
    type IntoFuture = Pin<Box<dyn Future<Output = Self::Output>>>;

    fn into_future(self) -> Self::IntoFuture {
        Box::pin(async move {
            let key = SessionKey::new(self.key).map_err(|source| {
                ModuleRegistryError::SessionAcquire {
                    owner: self.owner.clone(),
                    source,
                }
            })?;
            let snapshot = self
                .root
                .inner
                .session_store
                .load(&self.owner, &key)
                .await
                .map_err(|source| ModuleRegistryError::SessionRestore {
                    owner: self.owner.clone(),
                    key: key.clone(),
                    source,
                })?;
            let restored = snapshot.is_some();
            let mut session = snapshot
                .map(crate::PersistedSessionSnapshot::into_session)
                .unwrap_or_else(Session::new);
            let reasoning = self.root.inner.tiers.pick_handle(self.tier).reasoning;
            attach_persistent_session_metadata(
                &mut session,
                self.owner,
                key,
                self.auto_compaction,
                restored,
                reasoning,
            );
            Ok(session)
        })
    }
}

type ErasedModuleBuildFuture =
    Pin<Box<dyn Future<Output = Result<Box<dyn ErasedModule>, ModuleRegistryError>>>>;
type ErasedModuleBuilder = Rc<dyn Fn(ModuleCapabilityFactory) -> ErasedModuleBuildFuture>;

pub struct AllocatedModule {
    owner: ModuleInstanceId,
    caps: CapabilityProviders,
    builder: ErasedModuleBuilder,
    // Kept so a restart rebuilds the module with the boot-time memo scope
    // instead of silently widening it back to every module role.
    memo_subscription: MemoSubscription,
    module: Box<dyn ErasedModule>,
}

impl AllocatedModule {
    fn new(
        owner: ModuleInstanceId,
        caps: CapabilityProviders,
        builder: ErasedModuleBuilder,
        memo_subscription: MemoSubscription,
        module: Box<dyn ErasedModule>,
    ) -> Self {
        Self {
            owner,
            caps,
            builder,
            memo_subscription,
            module,
        }
    }

    pub fn owner(&self) -> &ModuleInstanceId {
        &self.owner
    }

    pub async fn restart(&mut self) -> Result<(), ModuleRegistryError> {
        let scoped = self
            .caps
            .scoped_with_memo_subscription(self.owner.clone(), self.memo_subscription.clone());
        self.module = (self.builder)(scoped).await?;
        Ok(())
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
    deps_of: HashMap<ScopedModuleId, Vec<ScopedModuleId>>,
    dependents_of: HashMap<ScopedModuleId, Vec<ScopedModuleId>>,
    activation_barriers: HashMap<ScopedModuleId, ActivationBarrier>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ActivationBarrier {
    prerequisites: Vec<ScopedModuleId>,
    timeout: Option<Duration>,
}

impl ActivationBarrier {
    pub fn prerequisites(&self) -> &[ScopedModuleId] {
        &self.prerequisites
    }

    pub fn timeout(&self) -> Option<Duration> {
        self.timeout
    }
}

impl ModuleDependencies {
    pub fn deps_of(&self, module: &ScopedModuleId) -> &[ScopedModuleId] {
        self.deps_of.get(module).map(Vec::as_slice).unwrap_or(&[])
    }

    pub fn dependents_of(&self, module: &ScopedModuleId) -> &[ScopedModuleId] {
        self.dependents_of
            .get(module)
            .map(Vec::as_slice)
            .unwrap_or(&[])
    }

    pub fn activation_barrier_for(&self, module: &ScopedModuleId) -> Option<&ActivationBarrier> {
        self.activation_barriers.get(module)
    }
}

pub struct ModuleRegistry {
    registrations: Vec<ModuleRegistration>,
    dependencies: Vec<(ScopedModuleId, ScopedModuleId)>,
    activation_barriers: Vec<(ScopedModuleId, Vec<ScopedModuleId>, Option<Duration>)>,
    registration_scope: ScopeId,
    subsystems: Vec<SubsystemRegistrationSpec>,
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
    spec: ModuleRegistrationSpec,
    builder: ErasedModuleBuilder,
}

impl fmt::Debug for ModuleRegistration {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ModuleRegistration")
            .field("spec", &self.spec)
            .finish_non_exhaustive()
    }
}

/// Builds one module replica from its replica-scoped capability factory.
///
/// Registration builders are async so boot-time capability acquisition can
/// perform eager I/O, such as loading persistent module sessions.
pub trait ModuleRegisterer: Fn(ModuleCapabilityFactory) -> Self::Future {
    type Module: crate::Module + 'static;
    type Future: Future<Output = Result<Self::Module, ModuleRegistryError>> + 'static;
}

impl<F, Fut, M> ModuleRegisterer for F
where
    F: Fn(ModuleCapabilityFactory) -> Fut,
    Fut: Future<Output = Result<M, ModuleRegistryError>> + 'static,
    M: Module + 'static,
{
    type Module = M;
    type Future = Fut;
}

impl ModuleRegistry {
    pub fn new() -> Self {
        Self {
            registrations: Vec::new(),
            dependencies: Vec::new(),
            activation_barriers: Vec::new(),
            registration_scope: ScopeId::root(),
            subsystems: Vec::new(),
        }
    }

    /// Set the scope applied to subsequently registered specs. Hosts use this
    /// while expanding reusable module wiring across subsystem instances.
    pub fn with_registration_scope(mut self, scope: ScopeId) -> Self {
        self.registration_scope = scope;
        self
    }

    /// Register an immediate child subsystem mount as a first-class resource
    /// allocation target.
    pub fn with_subsystem(mut self, spec: SubsystemRegistrationSpec) -> Self {
        self.subsystems.push(spec);
        self
    }

    /// Declare that `dependent` should wait for active `dependency` replicas to flush before
    /// activation. Both roles must be registered and the dependency graph must be acyclic.
    pub fn depends_on(mut self, dependent: ModuleId, dependency: ModuleId) -> Self {
        self.dependencies.push((
            ScopedModuleId::new(ScopeId::root(), dependent),
            ScopedModuleId::new(ScopeId::root(), dependency),
        ));
        self
    }

    pub fn scoped_depends_on(
        mut self,
        scope: ScopeId,
        dependent: ModuleId,
        dependency: ModuleId,
    ) -> Self {
        self.dependencies.push((
            ScopedModuleId::new(scope.clone(), dependent),
            ScopedModuleId::new(scope, dependency),
        ));
        self
    }

    pub fn activation_barrier(
        self,
        dependent: ModuleId,
        prerequisites: impl IntoIterator<Item = ModuleId>,
        timeout: Option<Duration>,
    ) -> Self {
        self.scoped_activation_barrier(ScopeId::root(), dependent, prerequisites, timeout)
    }

    pub fn scoped_activation_barrier(
        mut self,
        scope: ScopeId,
        dependent: ModuleId,
        prerequisites: impl IntoIterator<Item = ModuleId>,
        timeout: Option<Duration>,
    ) -> Self {
        self.activation_barriers.push((
            ScopedModuleId::new(scope.clone(), dependent),
            prerequisites
                .into_iter()
                .map(|module| ScopedModuleId::new(scope.clone(), module))
                .collect(),
            timeout,
        ));
        self
    }

    /// Remove a registered module role and any dependency edges touching it.
    ///
    /// Removing an absent module is a no-op. This is intended for host boot
    /// configuration that starts from a common registry and subtracts modules.
    pub fn remove_module(self, module: ModuleId) -> Self {
        self.remove_modules([module])
    }

    /// Remove registered module roles and any dependency edges touching them.
    pub fn remove_modules<I>(mut self, modules: I) -> Self
    where
        I: IntoIterator<Item = ModuleId>,
    {
        let removed = modules.into_iter().collect::<HashSet<_>>();
        if removed.is_empty() {
            return self;
        }

        self.registrations
            .retain(|registration| !removed.contains(&registration.spec.module));
        self.dependencies.retain(|(dependent, dependency)| {
            !removed.contains(&dependent.module) && !removed.contains(&dependency.module)
        });
        self.activation_barriers
            .retain_mut(|(dependent, prerequisites, _)| {
                if removed.contains(&dependent.module) {
                    return false;
                }
                prerequisites.retain(|prerequisite| !removed.contains(&prerequisite.module));
                !prerequisites.is_empty()
            });
        self
    }

    /// Register a module implementation under the explicit boot-time role in
    /// `spec`. The same concrete module type may be registered under multiple
    /// distinct ids.
    pub fn register<B>(
        mut self,
        mut spec: ModuleRegistrationSpec,
        builder: B,
    ) -> Result<Self, ModuleRegistryError>
    where
        B: ModuleRegisterer + 'static,
    {
        if spec.scope.is_root() && !self.registration_scope.is_root() {
            spec.scope = self.registration_scope.clone();
        }
        let module = spec.module.clone();
        let scoped_module = spec.scoped_module();
        if self
            .registrations
            .iter()
            .any(|registration| registration.spec.scoped_module() == scoped_module)
        {
            return Err(ModuleRegistryError::DuplicateModule {
                module: scoped_module,
            });
        }
        let replica_capacity = spec.replica_capacity;
        if replica_capacity > ReplicaCapRange::V1_MAX {
            return Err(ModuleRegistryError::ReplicaCapacityAboveV1Max {
                module,
                capacity: replica_capacity,
            });
        }
        let policy_capacity = spec.policy.max_active_replicas();
        if replica_capacity < policy_capacity {
            return Err(ModuleRegistryError::ReplicaCapacityBelowPolicyMax {
                module,
                capacity: replica_capacity,
                policy_capacity,
            });
        }
        self.dependencies
            .extend(spec.dependencies.iter().cloned().map(|dependency| {
                (
                    spec.scoped_module(),
                    ScopedModuleId::new(spec.scope.clone(), dependency),
                )
            }));
        if let Some(barrier) = &spec.activation_barrier {
            self.activation_barriers.push((
                spec.scoped_module(),
                barrier
                    .prerequisites
                    .iter()
                    .cloned()
                    .map(|module| ScopedModuleId::new(spec.scope.clone(), module))
                    .collect(),
                barrier.timeout,
            ));
        }
        self.registrations.push(ModuleRegistration {
            spec,
            builder: Rc::new(move |caps| {
                let future = builder(caps);
                Box::pin(async move {
                    future
                        .await
                        .map(|module| Box::new(module) as Box<dyn ErasedModule>)
                })
            }),
        });
        Ok(self)
    }

    pub async fn build(
        &self,
        caps: &CapabilityProviders,
    ) -> Result<AllocatedModules, ModuleRegistryError> {
        self.validate_memo_subscriptions()?;
        let dependencies = self.compile_dependencies()?;
        self.validate_subsystems()?;
        let catalog = ModuleCatalog::from_registrations(
            &self.registrations,
            &self.dependencies,
            &self.subsystems,
        );
        caps.install_module_catalog(catalog)?;

        let mut registrations_by_scope = BTreeMap::<ScopeId, Vec<&ModuleRegistration>>::new();
        for registration in &self.registrations {
            registrations_by_scope
                .entry(registration.spec.scope.clone())
                .or_default()
                .push(registration);
        }
        // Root resources exist even when the registry is empty (for example in
        // persistence-only tests and tools), so their retention policy must
        // always be installed before restoring persisted entries.
        caps.apply_runtime_policy(ScopeId::root()).await;
        let mut subsystems_by_parent = BTreeMap::<ScopeId, Vec<RegisteredSubsystemPolicy>>::new();
        for subsystem in &self.subsystems {
            subsystems_by_parent
                .entry(subsystem.parent_scope.clone())
                .or_default()
                .push(RegisteredSubsystemPolicy {
                    subsystem: subsystem.subsystem.clone(),
                    policy: subsystem.policy.clone(),
                    initial_activation: subsystem.initial_activation,
                });
        }
        for (scope, registrations) in subsystems_by_parent {
            caps.set_registered_subsystems(scope, registrations).await;
        }
        for (scope, registrations) in &registrations_by_scope {
            if !scope.is_root() {
                caps.apply_runtime_policy(scope.clone()).await;
            }
            caps.set_registered_modules(
                scope.clone(),
                registrations
                    .iter()
                    .map(|registration| RegisteredModulePolicy {
                        module: registration.spec.module.clone(),
                        policy: registration.spec.policy.clone(),
                        replica_capacity: registration.spec.replica_capacity,
                        initial_activation: registration.spec.initial_activation,
                    })
                    .collect(),
            )
            .await;
        }
        caps.restore_memo_log_entries()
            .await
            .map_err(ModuleRegistryError::MemoLogRestore)?;
        caps.restore_allocation_snapshots()
            .await
            .map_err(ModuleRegistryError::AllocationRestore)?;
        caps.restore_cognition_log_entries()
            .await
            .map_err(ModuleRegistryError::CognitionLogRestore)?;
        // Install the post-boot module catalogs before any module is constructed
        // so module constructors can read peers from `caps.peer_contexts()`
        // synchronously when they assemble their system prompts.
        for (scope, registrations) in &registrations_by_scope {
            caps.set_module_contexts(
                scope.clone(),
                registrations
                    .iter()
                    .filter_map(|registration| {
                        registration
                            .spec
                            .peer_context
                            .clone()
                            .map(|context| (registration.spec.module.clone(), context))
                    })
                    .collect(),
            );
        }
        let mut modules = Vec::new();
        for registration in &self.registrations {
            // Build every possible replica up to the registered max, with a
            // replica-0 floor so inactive modules can retain queued messages.
            let total_replicas = registration.spec.replica_capacity;
            for replica in 0..total_replicas {
                let owner = ModuleInstanceId::in_scope(
                    registration.spec.scope.clone(),
                    registration.spec.module.clone(),
                    ReplicaIndex::new(replica),
                );
                let scoped = caps.scoped_with_memo_subscription(
                    owner.clone(),
                    registration.spec.memo_subscription.clone(),
                );
                modules.push(AllocatedModule::new(
                    owner,
                    caps.clone(),
                    Rc::clone(&registration.builder),
                    registration.spec.memo_subscription.clone(),
                    (registration.builder)(scoped).await?,
                ));
            }
        }
        Ok(AllocatedModules::new(
            caps.runtime_control(),
            modules,
            dependencies,
        ))
    }

    fn validate_subsystems(&self) -> Result<(), ModuleRegistryError> {
        let mut mounted = HashSet::new();
        for spec in &self.subsystems {
            if !mounted.insert((spec.parent_scope.clone(), spec.subsystem.clone())) {
                return Err(ModuleRegistryError::DuplicateSubsystemMount {
                    parent: spec.parent_scope.clone(),
                    subsystem: spec.subsystem.clone(),
                });
            }
            if spec.policy.replica_capacity == 0 {
                return Err(ModuleRegistryError::SubsystemReplicaCapacityZero {
                    subsystem: spec.subsystem.clone(),
                });
            }
            if spec.allocation_description.trim().is_empty() {
                return Err(ModuleRegistryError::EmptySubsystemAllocationDescription {
                    subsystem: spec.subsystem.clone(),
                });
            }
            if spec.policy.replica_capacity < spec.policy.replicas_range.max {
                return Err(
                    ModuleRegistryError::SubsystemReplicaCapacityBelowPolicyMax {
                        subsystem: spec.subsystem.clone(),
                        capacity: spec.policy.replica_capacity,
                        policy_capacity: spec.policy.replicas_range.max,
                    },
                );
            }
        }
        Ok(())
    }

    fn compile_dependencies(&self) -> Result<ModuleDependencies, ModuleRegistryError> {
        let registered = self
            .registrations
            .iter()
            .map(|registration| registration.spec.scoped_module())
            .collect::<HashSet<_>>();
        let mut deps_of = HashMap::<ScopedModuleId, Vec<ScopedModuleId>>::new();
        let mut dependents_of = HashMap::<ScopedModuleId, Vec<ScopedModuleId>>::new();
        let mut activation_barriers = HashMap::<ScopedModuleId, ActivationBarrier>::new();

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

        let mut cycle_edges = deps_of.clone();
        for (dependent, prerequisites, timeout) in &self.activation_barriers {
            if !registered.contains(dependent) {
                return Err(ModuleRegistryError::UnknownBarrierDependent {
                    dependent: dependent.clone(),
                });
            }
            if prerequisites.is_empty() {
                return Err(ModuleRegistryError::EmptyActivationBarrier {
                    dependent: dependent.clone(),
                });
            }
            if timeout.is_some_and(|timeout| timeout.is_zero()) {
                return Err(ModuleRegistryError::ZeroActivationBarrierTimeout {
                    dependent: dependent.clone(),
                });
            }
            let mut seen = HashSet::new();
            for prerequisite in prerequisites {
                if !registered.contains(prerequisite) {
                    return Err(ModuleRegistryError::UnknownBarrierPrerequisite {
                        dependent: dependent.clone(),
                        prerequisite: prerequisite.clone(),
                    });
                }
                if dependent == prerequisite {
                    return Err(ModuleRegistryError::DependencyCycle {
                        cycle: vec![dependent.clone()],
                    });
                }
                if !seen.insert(prerequisite.clone()) {
                    return Err(ModuleRegistryError::DuplicateBarrierPrerequisite {
                        dependent: dependent.clone(),
                        prerequisite: prerequisite.clone(),
                    });
                }
                let edges = cycle_edges.entry(dependent.clone()).or_default();
                if !edges.contains(prerequisite) {
                    edges.push(prerequisite.clone());
                }
            }
            if activation_barriers
                .insert(
                    dependent.clone(),
                    ActivationBarrier {
                        prerequisites: prerequisites.clone(),
                        timeout: *timeout,
                    },
                )
                .is_some()
            {
                return Err(ModuleRegistryError::DuplicateActivationBarrier {
                    dependent: dependent.clone(),
                });
            }
        }

        let mut visiting = HashSet::<ScopedModuleId>::new();
        let mut visited = HashSet::<ScopedModuleId>::new();
        for module in registered {
            if visited.contains(&module) {
                continue;
            }
            let mut stack = Vec::new();
            dfs_check_dependencies(
                module.clone(),
                &cycle_edges,
                &mut visiting,
                &mut visited,
                &mut stack,
            )?;
        }

        Ok(ModuleDependencies {
            deps_of,
            dependents_of,
            activation_barriers,
        })
    }

    fn validate_memo_subscriptions(&self) -> Result<(), ModuleRegistryError> {
        let registered = self
            .registrations
            .iter()
            .map(|registration| registration.spec.scoped_module())
            .collect::<HashSet<_>>();
        for registration in &self.registrations {
            let Some(sources) = registration.spec.memo_subscription.sources() else {
                continue;
            };
            for source in sources {
                let scoped_source =
                    ScopedModuleId::new(registration.spec.scope.clone(), source.clone());
                if !registered.contains(&scoped_source) {
                    return Err(ModuleRegistryError::UnknownMemoSource {
                        subscriber: registration.spec.module.clone(),
                        memo_source: source.clone(),
                    });
                }
            }
        }
        Ok(())
    }
}

fn dfs_check_dependencies(
    node: ScopedModuleId,
    deps_of: &HashMap<ScopedModuleId, Vec<ScopedModuleId>>,
    visiting: &mut HashSet<ScopedModuleId>,
    visited: &mut HashSet<ScopedModuleId>,
    stack: &mut Vec<ScopedModuleId>,
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
    DuplicateModule { module: ScopedModuleId },
    #[error(
        "module catalog changed after this agent environment booted; create a new environment to apply registration changes"
    )]
    CatalogChangedAfterBoot,
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
    UnknownDependent { dependent: ScopedModuleId },
    #[error("dependency {dependency} declared in depends_on() but not registered")]
    UnknownDependency { dependency: ScopedModuleId },
    #[error("activation barrier dependent {dependent} is not registered")]
    UnknownBarrierDependent { dependent: ScopedModuleId },
    #[error("activation barrier prerequisite {prerequisite} for {dependent} is not registered")]
    UnknownBarrierPrerequisite {
        dependent: ScopedModuleId,
        prerequisite: ScopedModuleId,
    },
    #[error("activation barrier for {dependent} must declare at least one prerequisite")]
    EmptyActivationBarrier { dependent: ScopedModuleId },
    #[error("activation barrier for {dependent} must have a timeout greater than zero")]
    ZeroActivationBarrierTimeout { dependent: ScopedModuleId },
    #[error("activation barrier for {dependent} is declared more than once")]
    DuplicateActivationBarrier { dependent: ScopedModuleId },
    #[error(
        "activation barrier prerequisite {prerequisite} is declared more than once for {dependent}"
    )]
    DuplicateBarrierPrerequisite {
        dependent: ScopedModuleId,
        prerequisite: ScopedModuleId,
    },
    #[error("module {subscriber} subscribes to memos from unregistered module {memo_source}")]
    UnknownMemoSource {
        subscriber: ModuleId,
        memo_source: ModuleId,
    },
    #[error("subsystem {subsystem} is mounted more than once under {parent}")]
    DuplicateSubsystemMount {
        parent: ScopeId,
        subsystem: SubsystemId,
    },
    #[error("subsystem {subsystem} replica capacity must be greater than zero")]
    SubsystemReplicaCapacityZero { subsystem: SubsystemId },
    #[error("subsystem {subsystem} allocation description must not be empty")]
    EmptySubsystemAllocationDescription { subsystem: SubsystemId },
    #[error(
        "subsystem {subsystem} replica capacity {capacity} is below policy capacity {policy_capacity}"
    )]
    SubsystemReplicaCapacityBelowPolicyMax {
        subsystem: SubsystemId,
        capacity: u8,
        policy_capacity: u8,
    },
    #[error("module {owner} requires an outer scope")]
    MissingOuterScope { owner: ModuleInstanceId },
    #[error(
        "module dependency cycle detected: {}",
        cycle.iter().map(ToString::to_string).collect::<Vec<_>>().join(" -> ")
    )]
    DependencyCycle { cycle: Vec<ScopedModuleId> },
    #[error("failed to acquire session capability for {owner}: {source}")]
    SessionAcquire {
        owner: ModuleInstanceId,
        source: PortError,
    },
    #[error("failed to restore session {owner}/{key}: {source}")]
    SessionRestore {
        owner: ModuleInstanceId,
        key: SessionKey,
        source: PortError,
    },
    #[error("failed to restore persisted allocation snapshots: {0}")]
    AllocationRestore(PortError),
    #[error("failed to restore persisted memo log entries: {0}")]
    MemoLogRestore(PortError),
    #[error("failed to restore persisted cognition log entries: {0}")]
    CognitionLogRestore(PortError),
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::cell::RefCell;
    use std::sync::Arc;

    use async_trait::async_trait;
    use chrono::{DateTime, Utc};
    use nuillu_blackboard::{
        ActivationRatio, AllocationCommand, AllocationEffectLevel, Blackboard, BlackboardCommand,
        CognitionLogEntry, CognitionLogOrigin, MemoLogPayload, MemoLogRecord, ResourceAllocation,
    };
    use nuillu_types::{ModuleId, ReplicaCapRange, SubsystemId, SubsystemInstanceId, builtin};

    use crate::allocation_persistence::PersistedAllocationSnapshot;
    use crate::ports::{
        CognitionLogCursor, CognitionLogRepository, PersistedCognitionLogEntry,
        PersistedCognitionLogPageEntry, PortError, SystemClock,
    };
    use crate::runtime_events::{RuntimeEvent, RuntimeEventEmitter, RuntimeEventSink};
    use crate::session::{
        NoopSessionStore, PersistedSessionSnapshot, SessionAutoCompaction, SessionKey,
        persistent_session_metadata,
    };
    use crate::session_compaction::{
        SessionCompactionConfig, SessionCompactionPolicy, SessionCompactionProtectedPrefix,
    };
    use crate::test_support::{scoped, test_caps};
    use lutum::{FinishReason, MockLlmAdapter, MockTextScenario, RawTextTurnEvent};

    fn test_policy(replicas_range: std::ops::RangeInclusive<u8>) -> ModulePolicy {
        ModulePolicy::new(
            ReplicaCapRange::new(*replicas_range.start(), *replicas_range.end()).unwrap(),
            nuillu_blackboard::Bpm::from_f64(60.0)..=nuillu_blackboard::Bpm::from_f64(60.0),
            nuillu_blackboard::linear_ratio_fn,
        )
    }

    #[derive(Clone, Default)]
    struct RecordingCognitionLogRepository {
        records: Arc<std::sync::Mutex<Vec<(ScopeId, ModuleInstanceId, CognitionLogEntry)>>>,
    }

    impl RecordingCognitionLogRepository {
        fn with_records(records: Vec<(ModuleInstanceId, CognitionLogEntry)>) -> Self {
            Self {
                records: Arc::new(std::sync::Mutex::new(
                    records
                        .into_iter()
                        .map(|(source, entry)| (source.scope.clone(), source, entry))
                        .collect(),
                )),
            }
        }

        fn records(&self) -> Vec<(ScopeId, ModuleInstanceId, CognitionLogEntry)> {
            self.records.lock().expect("records mutex poisoned").clone()
        }
    }

    #[derive(Clone, Default)]
    struct RecordingRuntimeEventSink {
        events: Rc<RefCell<Vec<RuntimeEvent>>>,
    }

    impl RecordingRuntimeEventSink {
        fn events(&self) -> Vec<RuntimeEvent> {
            self.events.borrow().clone()
        }
    }

    #[async_trait(?Send)]
    impl RuntimeEventSink for RecordingRuntimeEventSink {
        fn on_event(&self, event: RuntimeEvent) -> Result<(), PortError> {
            self.events.borrow_mut().push(event);
            Ok(())
        }
    }

    #[derive(Clone, Default)]
    struct RecordingSessionStore {
        saves: Rc<RefCell<Vec<(ModuleInstanceId, SessionKey, PersistedSessionSnapshot)>>>,
    }

    impl RecordingSessionStore {
        fn saves(&self) -> Vec<(ModuleInstanceId, SessionKey, PersistedSessionSnapshot)> {
            self.saves.borrow().clone()
        }
    }

    #[async_trait(?Send)]
    impl SessionStore for RecordingSessionStore {
        async fn load(
            &self,
            _owner: &ModuleInstanceId,
            _key: &SessionKey,
        ) -> Result<Option<PersistedSessionSnapshot>, PortError> {
            Ok(None)
        }

        async fn save(
            &self,
            owner: &ModuleInstanceId,
            key: &SessionKey,
            snapshot: &PersistedSessionSnapshot,
        ) -> Result<(), PortError> {
            self.saves
                .borrow_mut()
                .push((owner.clone(), key.clone(), snapshot.clone()));
            Ok(())
        }

        async fn delete_owner(&self, owner: &ModuleInstanceId) -> Result<u64, PortError> {
            let before = self.saves.borrow().len();
            self.saves
                .borrow_mut()
                .retain(|(saved_owner, _, _)| saved_owner != owner);
            Ok((before - self.saves.borrow().len()) as u64)
        }
    }

    #[derive(Clone, Default)]
    struct RecordingAllocationStore {
        snapshots: Rc<RefCell<Vec<PersistedAllocationSnapshot>>>,
        saves: Rc<RefCell<Vec<PersistedAllocationSnapshot>>>,
    }

    impl RecordingAllocationStore {
        fn with_snapshots(snapshots: Vec<PersistedAllocationSnapshot>) -> Self {
            Self {
                snapshots: Rc::new(RefCell::new(snapshots)),
                saves: Rc::new(RefCell::new(Vec::new())),
            }
        }

        fn saves(&self) -> Vec<PersistedAllocationSnapshot> {
            self.saves.borrow().clone()
        }
    }

    #[async_trait(?Send)]
    impl crate::AllocationStore for RecordingAllocationStore {
        async fn load_all(&self) -> Result<Vec<PersistedAllocationSnapshot>, PortError> {
            Ok(self.snapshots.borrow().clone())
        }

        async fn save(&self, snapshot: &PersistedAllocationSnapshot) -> Result<(), PortError> {
            self.saves.borrow_mut().push(snapshot.clone());
            Ok(())
        }
    }

    #[derive(Clone, Default)]
    struct RecordingMemoLogRepository {
        records: Rc<RefCell<Vec<PersistedMemoLogEntry>>>,
        appends: Rc<RefCell<Vec<PersistedMemoLogEntry>>>,
    }

    impl RecordingMemoLogRepository {
        fn with_records(records: Vec<PersistedMemoLogEntry>) -> Self {
            Self {
                records: Rc::new(RefCell::new(records)),
                appends: Rc::new(RefCell::new(Vec::new())),
            }
        }

        fn appends(&self) -> Vec<PersistedMemoLogEntry> {
            self.appends.borrow().clone()
        }
    }

    #[async_trait(?Send)]
    impl MemoLogRepository for RecordingMemoLogRepository {
        async fn append(&self, entry: &PersistedMemoLogEntry) -> Result<(), PortError> {
            self.appends.borrow_mut().push(entry.clone());
            Ok(())
        }

        async fn recent_per_owner(
            &self,
            retained_per_owner: usize,
        ) -> Result<Vec<PersistedMemoLogEntry>, PortError> {
            if retained_per_owner == 0 {
                return Ok(Vec::new());
            }
            let mut grouped =
                std::collections::BTreeMap::<String, Vec<PersistedMemoLogEntry>>::new();
            for entry in self.records.borrow().iter().cloned() {
                grouped
                    .entry(format!("{}\n{}", entry.scope, entry.record.owner))
                    .or_default()
                    .push(entry);
            }
            let mut records = Vec::new();
            for group in grouped.values_mut() {
                group.sort_by_key(|entry| entry.record.index);
                let keep_from = group.len().saturating_sub(retained_per_owner);
                records.extend(group[keep_from..].iter().cloned());
            }
            records.sort_by(|left, right| {
                left.scope
                    .cmp(&right.scope)
                    .then_with(|| {
                        left.record
                            .owner
                            .module
                            .as_str()
                            .cmp(right.record.owner.module.as_str())
                    })
                    .then_with(|| left.record.owner.replica.cmp(&right.record.owner.replica))
                    .then_with(|| left.record.index.cmp(&right.record.index))
            });
            Ok(records)
        }
    }

    #[async_trait(?Send)]
    impl CognitionLogRepository for RecordingCognitionLogRepository {
        async fn append(
            &self,
            scope: ScopeId,
            source: ModuleInstanceId,
            entry: CognitionLogEntry,
        ) -> Result<(), PortError> {
            self.records
                .lock()
                .expect("records mutex poisoned")
                .push((scope, source, entry));
            Ok(())
        }

        async fn since(
            &self,
            scope: &ScopeId,
            source: &ModuleInstanceId,
            from: DateTime<Utc>,
        ) -> Result<Vec<CognitionLogEntry>, PortError> {
            Ok(self
                .records
                .lock()
                .expect("records mutex poisoned")
                .iter()
                .filter(|(record_scope, record_source, entry)| {
                    record_scope == scope && record_source == source && entry.at >= from
                })
                .map(|(_, _, entry)| entry.clone())
                .collect())
        }

        async fn recent(&self, limit: usize) -> Result<Vec<PersistedCognitionLogEntry>, PortError> {
            if limit == 0 {
                return Ok(Vec::new());
            }
            let mut records = self
                .records
                .lock()
                .expect("records mutex poisoned")
                .iter()
                .rev()
                .take(limit)
                .map(|(scope, source, entry)| PersistedCognitionLogEntry {
                    scope: scope.clone(),
                    source: source.clone(),
                    entry: entry.clone(),
                })
                .collect::<Vec<_>>();
            records.reverse();
            Ok(records)
        }

        async fn page(
            &self,
            cursor: CognitionLogCursor,
            limit: usize,
        ) -> Result<Vec<PersistedCognitionLogPageEntry>, PortError> {
            Ok(self
                .records
                .lock()
                .expect("records mutex poisoned")
                .iter()
                .enumerate()
                .rev()
                .filter(|(index, _)| {
                    let id = i64::try_from(*index).unwrap_or(i64::MAX);
                    match cursor {
                        CognitionLogCursor::Newest => true,
                        CognitionLogCursor::Older { before_id } => id < before_id,
                        CognitionLogCursor::Newer { after_id } => id > after_id,
                    }
                })
                .take(limit)
                .map(
                    |(index, (scope, source, entry))| PersistedCognitionLogPageEntry {
                        id: i64::try_from(index).unwrap_or(i64::MAX),
                        scope: scope.clone(),
                        source: source.clone(),
                        entry: entry.clone(),
                    },
                )
                .collect())
        }
    }

    fn test_caps_with_cognition_repo(
        blackboard: Blackboard,
        cognition_log_port: Rc<dyn CognitionLogRepository>,
    ) -> CapabilityProviders {
        test_caps_with_cognition_repo_and_runtime(
            blackboard,
            cognition_log_port,
            CapabilityProviderRuntime::default(),
        )
    }

    fn test_caps_with_cognition_repo_and_runtime(
        blackboard: Blackboard,
        cognition_log_port: Rc<dyn CognitionLogRepository>,
        runtime: CapabilityProviderRuntime,
    ) -> CapabilityProviders {
        let adapter = Arc::new(lutum::MockLlmAdapter::new());
        let budget = lutum::SharedPoolBudgetManager::new(lutum::SharedPoolBudgetOptions::default());
        let lutum = lutum::Lutum::new(adapter, budget);
        CapabilityProviders::new(CapabilityProviderConfig {
            ports: CapabilityProviderPorts {
                blackboard,
                cognition_log_port,
                clock: Rc::new(SystemClock),
                tiers: LutumTiers::from_shared_lutum(lutum),
            },
            runtime,
        })
    }

    fn test_caps_with_session_store(
        blackboard: Blackboard,
        session_store: Rc<dyn SessionStore>,
    ) -> CapabilityProviders {
        let adapter = Arc::new(lutum::MockLlmAdapter::new());
        let budget = lutum::SharedPoolBudgetManager::new(lutum::SharedPoolBudgetOptions::default());
        let lutum = lutum::Lutum::new(adapter, budget);
        CapabilityProviders::new(CapabilityProviderConfig {
            ports: CapabilityProviderPorts {
                blackboard,
                cognition_log_port: Rc::new(crate::ports::NoopCognitionLogRepository),
                clock: Rc::new(SystemClock),
                tiers: LutumTiers::from_shared_lutum(lutum),
            },
            runtime: CapabilityProviderRuntime {
                session_store,
                ..CapabilityProviderRuntime::default()
            },
        })
    }

    fn test_caps_with_allocation_store(
        blackboard: Blackboard,
        allocation_store: Rc<dyn crate::AllocationStore>,
    ) -> CapabilityProviders {
        let adapter = Arc::new(lutum::MockLlmAdapter::new());
        let budget = lutum::SharedPoolBudgetManager::new(lutum::SharedPoolBudgetOptions::default());
        let lutum = lutum::Lutum::new(adapter, budget);
        CapabilityProviders::new(CapabilityProviderConfig {
            ports: CapabilityProviderPorts {
                blackboard,
                cognition_log_port: Rc::new(crate::ports::NoopCognitionLogRepository),
                clock: Rc::new(SystemClock),
                tiers: LutumTiers::from_shared_lutum(lutum),
            },
            runtime: CapabilityProviderRuntime {
                allocation_store,
                ..CapabilityProviderRuntime::default()
            },
        })
    }

    fn test_caps_with_memo_log_repository(
        blackboard: Blackboard,
        memo_log_repository: Rc<dyn MemoLogRepository>,
        policy: RuntimePolicy,
    ) -> CapabilityProviders {
        let adapter = Arc::new(lutum::MockLlmAdapter::new());
        let budget = lutum::SharedPoolBudgetManager::new(lutum::SharedPoolBudgetOptions::default());
        let lutum = lutum::Lutum::new(adapter, budget);
        CapabilityProviders::new(CapabilityProviderConfig {
            ports: CapabilityProviderPorts {
                blackboard,
                cognition_log_port: Rc::new(crate::ports::NoopCognitionLogRepository),
                clock: Rc::new(SystemClock),
                tiers: LutumTiers::from_shared_lutum(lutum),
            },
            runtime: CapabilityProviderRuntime {
                policy,
                memo_log_repository,
                ..CapabilityProviderRuntime::default()
            },
        })
    }

    struct NoopModule;

    #[async_trait(?Send)]
    impl StaticModule for NoopModule {
        fn id() -> &'static str {
            "noop"
        }

        fn peer_context() -> Option<&'static str> {
            Some("test stub")
        }
    }

    #[async_trait(?Send)]
    impl Module for NoopModule {
        type Batch = ();

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

    async fn noop_builder(_: ModuleCapabilityFactory) -> Result<NoopModule, ModuleRegistryError> {
        Ok(NoopModule)
    }

    struct NoPeerContextModule;

    #[async_trait(?Send)]
    impl StaticModule for NoPeerContextModule {
        fn id() -> &'static str {
            "no-peer-context"
        }

        fn peer_context() -> Option<&'static str> {
            None
        }
    }

    #[async_trait(?Send)]
    impl Module for NoPeerContextModule {
        type Batch = ();

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

    async fn no_peer_context_builder(
        _: ModuleCapabilityFactory,
    ) -> Result<NoPeerContextModule, ModuleRegistryError> {
        Ok(NoPeerContextModule)
    }

    fn static_spec<M: StaticModule>(policy: ModulePolicy) -> ModuleRegistrationSpec {
        ModuleRegistrationSpec::for_static::<M>(policy, ActivationRatio::ZERO).unwrap()
    }

    #[test]
    fn register_rejects_duplicate_module_ids() {
        let registry = ModuleRegistry::new()
            .register(static_spec::<NoopModule>(test_policy(0..=0)), noop_builder)
            .unwrap();

        let err = registry
            .register(static_spec::<NoopModule>(test_policy(0..=0)), noop_builder)
            .unwrap_err();

        let expected = ScopedModuleId::new(
            ScopeId::root(),
            nuillu_types::ModuleId::new(NoopModule::id()).unwrap(),
        );
        assert!(matches!(
            err,
            ModuleRegistryError::DuplicateModule { module } if module == expected
        ));
    }

    #[tokio::test]
    async fn child_owned_outer_cognition_inbox_observes_the_immediate_parent_scope() {
        let caps = test_caps(Blackboard::default());
        let child_scope = ScopeId::root().child(SubsystemInstanceId::new(
            SubsystemId::new("arm").unwrap(),
            ReplicaIndex::ZERO,
        ));
        let gate_owner =
            ModuleInstanceId::in_scope(child_scope, builtin::subsystem_gate(), ReplicaIndex::ZERO);
        let mut outer_inbox = caps
            .scoped(gate_owner)
            .outer_cognition_log_updated_inbox()
            .unwrap()
            .broadcast();
        let root_source = ModuleInstanceId::new(builtin::cognition_gate(), ReplicaIndex::ZERO);

        caps.scoped(root_source.clone())
            .cognition_writer()
            .append("outer cognition")
            .await;

        assert_eq!(outer_inbox.next_item().await.unwrap().sender, root_source);
    }

    #[tokio::test]
    async fn registry_allows_the_same_role_in_distinct_subsystem_scopes() {
        let blackboard = Blackboard::default();
        let cognition_repo = RecordingCognitionLogRepository::default();
        let caps =
            test_caps_with_cognition_repo(blackboard.clone(), Rc::new(cognition_repo.clone()));
        let subsystem = SubsystemId::new("arm").unwrap();
        let left_scope = ScopeId::root().child(SubsystemInstanceId::new(
            subsystem.clone(),
            ReplicaIndex::ZERO,
        ));
        let right_scope =
            ScopeId::root().child(SubsystemInstanceId::new(subsystem, ReplicaIndex::new(1)));
        let role = ModuleId::new(NoopModule::id()).unwrap();
        let left_outer = Rc::new(Cell::new(false));
        let right_outer = Rc::new(Cell::new(false));
        let left_seen = Rc::clone(&left_outer);
        let right_seen = Rc::clone(&right_outer);

        let registry = ModuleRegistry::new()
            .with_registration_scope(left_scope.clone())
            .register(
                static_spec::<NoopModule>(test_policy(0..=1)),
                move |caps: ModuleCapabilityFactory| {
                    let left_seen = Rc::clone(&left_seen);
                    async move {
                        let outer = caps.outer_cognition_writer();
                        left_seen.set(outer.is_some());
                        outer.unwrap().append("forwarded from left").await;
                        Ok(NoopModule)
                    }
                },
            )
            .unwrap()
            .with_registration_scope(right_scope.clone())
            .register(
                static_spec::<NoopModule>(test_policy(0..=1)),
                move |caps: ModuleCapabilityFactory| {
                    let right_seen = Rc::clone(&right_seen);
                    async move {
                        right_seen.set(caps.outer_cognition_writer().is_some());
                        Ok(NoopModule)
                    }
                },
            )
            .unwrap();

        let allocated = registry.build(&caps).await.unwrap();
        let owners = allocated
            .modules
            .iter()
            .map(|module| module.owner().clone())
            .collect::<HashSet<_>>();
        assert_eq!(
            owners,
            HashSet::from([
                ModuleInstanceId::in_scope(left_scope.clone(), role.clone(), ReplicaIndex::ZERO,),
                ModuleInstanceId::in_scope(right_scope.clone(), role.clone(), ReplicaIndex::ZERO,),
            ])
        );
        assert!(left_outer.get());
        assert!(right_outer.get());
        let persisted = cognition_repo.records();
        assert_eq!(persisted.len(), 1);
        assert!(persisted[0].0.is_root());
        assert_eq!(persisted[0].1.scope, left_scope);
        assert_eq!(
            persisted[0].2.origin,
            CognitionLogOrigin::direct(ModuleInstanceId::in_scope(
                left_scope.clone(),
                role.clone(),
                ReplicaIndex::ZERO,
            ))
        );
        assert_eq!(
            blackboard
                .read(|bb| {
                    bb.cognition_log()
                        .entries()
                        .iter()
                        .map(|entry| entry.text.clone())
                        .collect::<Vec<_>>()
                })
                .await,
            vec!["forwarded from left".to_owned()]
        );
        assert!(blackboard.read(|bb| bb.module_policies().is_empty()).await);
        assert!(
            blackboard
                .scoped(left_scope)
                .read(|bb| bb.module_policies().contains_key(&role))
                .await
        );
        assert!(
            blackboard
                .scoped(right_scope)
                .read(|bb| bb.module_policies().contains_key(&role))
                .await
        );
    }

    #[tokio::test]
    async fn registry_build_rejects_unregistered_memo_source() {
        let subscriber = ModuleId::new("memo-subscriber").unwrap();
        let missing = ModuleId::new("missing-source").unwrap();
        let caps = test_caps(Blackboard::default());
        let result = ModuleRegistry::new()
            .register(
                ModuleRegistrationSpec::new(
                    subscriber.clone(),
                    test_policy(0..=0),
                    ActivationRatio::ZERO,
                )
                .with_memo_sources([missing.clone()]),
                noop_builder,
            )
            .unwrap()
            .build(&caps)
            .await;
        let error = match result {
            Ok(_) => panic!("unregistered memo source should fail registry build"),
            Err(error) => error,
        };

        assert!(matches!(
            error,
            ModuleRegistryError::UnknownMemoSource {
                subscriber: actual_subscriber,
                memo_source,
            } if actual_subscriber == subscriber && memo_source == missing
        ));
    }

    #[tokio::test]
    async fn restart_rebuilds_module_with_registered_memo_subscription() {
        let caps = test_caps(Blackboard::default());
        let subscriber = ModuleId::new("memo-subscriber").unwrap();
        let inboxes: Rc<RefCell<Vec<MemoUpdatedInbox>>> = Rc::default();
        let captured = inboxes.clone();
        let (_runtime, mut modules) = ModuleRegistry::new()
            .register(
                ModuleRegistrationSpec::new(
                    subscriber.clone(),
                    test_policy(0..=0),
                    ActivationRatio::ZERO,
                )
                .with_memo_sources([builtin::query_memory()]),
                move |factory: ModuleCapabilityFactory| {
                    let captured = captured.clone();
                    async move {
                        captured.borrow_mut().push(factory.memo_updated_inbox());
                        Ok(NoopModule)
                    }
                },
            )
            .unwrap()
            .register(
                ModuleRegistrationSpec::new(
                    builtin::query_memory(),
                    test_policy(0..=0),
                    ActivationRatio::ZERO,
                ),
                noop_builder,
            )
            .unwrap()
            .register(
                ModuleRegistrationSpec::new(
                    builtin::sensory(),
                    test_policy(0..=0),
                    ActivationRatio::ZERO,
                ),
                noop_builder,
            )
            .unwrap()
            .build(&caps)
            .await
            .unwrap()
            .into_parts();

        modules
            .iter_mut()
            .find(|module| module.owner().module == subscriber)
            .expect("subscriber replica should be allocated")
            .restart()
            .await
            .unwrap();

        let mut inbox = inboxes
            .borrow_mut()
            .pop()
            .expect("restart should rebuild the module through its capability factory");
        scoped(&caps, builtin::sensory(), 0)
            .memo()
            .write("external context")
            .await;
        assert!(inbox.take_ready_items().unwrap().items.is_empty());

        scoped(&caps, builtin::query_memory(), 0)
            .memo()
            .write("identity evidence")
            .await;
        let event = inbox.next_item().await.unwrap();
        assert_eq!(event.body.owner.module, builtin::query_memory());
    }

    #[tokio::test]
    async fn register_with_replica_capacity_builds_hard_cap_but_keeps_soft_policy() {
        let blackboard = Blackboard::default();
        let caps = test_caps(blackboard.clone());
        let allocated = ModuleRegistry::new()
            .register(
                static_spec::<NoopModule>(test_policy(0..=1)).with_replica_capacity(2),
                noop_builder,
            )
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
    async fn build_installs_peer_context_catalog() {
        let blackboard = Blackboard::default();
        let caps = test_caps(blackboard.clone());
        ModuleRegistry::new()
            .register(static_spec::<NoopModule>(test_policy(0..=0)), noop_builder)
            .unwrap()
            .register(
                static_spec::<NoPeerContextModule>(test_policy(0..=0)),
                no_peer_context_builder,
            )
            .unwrap()
            .build(&caps)
            .await
            .unwrap();

        let noop = nuillu_types::ModuleId::new(NoopModule::id()).unwrap();

        assert_eq!(
            blackboard.peer_contexts().to_vec(),
            vec![(noop, Arc::from("test stub"))]
        );
    }

    #[tokio::test]
    async fn build_installs_subsystem_allocation_metadata_in_catalog() {
        let blackboard = Blackboard::default();
        let caps = test_caps(blackboard);
        let subsystem = SubsystemId::new("arm").unwrap();
        let catalog_seen = Rc::new(RefCell::new(Vec::new()));
        let seen = Rc::clone(&catalog_seen);
        ModuleRegistry::new()
            .with_subsystem(
                SubsystemRegistrationSpec::new(
                    ScopeId::root(),
                    subsystem.clone(),
                    SubsystemPolicy::new(
                        SubsystemReplicaRange::new(0, 2).unwrap(),
                        2,
                        nuillu_blackboard::ReplicaProjection::Linear,
                    ),
                    ActivationRatio::from_f64(0.5),
                    "Reaching and grasping",
                )
                .with_label("Arm")
                .with_activation_table([
                    ActivationRatio::ONE,
                    ActivationRatio::from_f64(0.4),
                    ActivationRatio::ZERO,
                ]),
            )
            .register(static_spec::<NoopModule>(test_policy(0..=0)), move |caps| {
                let seen = Rc::clone(&seen);
                async move {
                    *seen.borrow_mut() = caps.subsystem_catalog().children();
                    Ok(NoopModule)
                }
            })
            .unwrap()
            .build(&caps)
            .await
            .unwrap();

        let seen = catalog_seen.borrow();
        assert_eq!(seen.len(), 1);
        assert_eq!(seen[0].subsystem, subsystem);
        assert_eq!(seen[0].label.as_deref(), Some("Arm"));
        assert_eq!(
            seen[0].allocation_description.as_ref(),
            "Reaching and grasping"
        );
        assert_eq!(
            seen[0].activation_table.as_ref(),
            &[
                ActivationRatio::ONE,
                ActivationRatio::from_f64(0.4),
                ActivationRatio::ZERO,
            ]
        );
    }

    #[tokio::test]
    async fn registry_rejects_zero_capacity_subsystem_mount() {
        let caps = test_caps(Blackboard::default());
        let subsystem = SubsystemId::new("arm").unwrap();
        let result = ModuleRegistry::new()
            .with_subsystem(SubsystemRegistrationSpec::new(
                ScopeId::root(),
                subsystem.clone(),
                SubsystemPolicy::new(
                    SubsystemReplicaRange::new(0, 0).unwrap(),
                    0,
                    nuillu_blackboard::ReplicaProjection::Linear,
                ),
                ActivationRatio::ZERO,
                "Test arm subsystem",
            ))
            .build(&caps)
            .await;

        assert!(matches!(
            result,
            Err(ModuleRegistryError::SubsystemReplicaCapacityZero {
                subsystem: rejected
            }) if rejected == subsystem
        ));
    }

    #[tokio::test]
    async fn registry_rejects_empty_subsystem_allocation_description() {
        let caps = test_caps(Blackboard::default());
        let subsystem = SubsystemId::new("arm").unwrap();
        let result = ModuleRegistry::new()
            .with_subsystem(SubsystemRegistrationSpec::new(
                ScopeId::root(),
                subsystem.clone(),
                SubsystemPolicy::new(
                    SubsystemReplicaRange::new(0, 1).unwrap(),
                    1,
                    nuillu_blackboard::ReplicaProjection::Linear,
                ),
                ActivationRatio::ZERO,
                "  ",
            ))
            .build(&caps)
            .await;

        assert!(matches!(
            result,
            Err(ModuleRegistryError::EmptySubsystemAllocationDescription {
                subsystem: rejected
            }) if rejected == subsystem
        ));
    }

    #[tokio::test]
    async fn dynamic_registrations_reuse_one_module_type_with_distinct_roles() {
        let blackboard = Blackboard::default();
        let caps = test_caps(blackboard.clone());
        let alpha = ModuleId::new("mcp-alpha").unwrap();
        let beta = ModuleId::new("mcp-beta").unwrap();
        let group = ModuleGroupId::new("external-tools").unwrap();
        let built_owners = Rc::new(RefCell::new(Vec::new()));
        let catalog_seen_during_build = Rc::new(RefCell::new(Vec::new()));

        let alpha_owners = Rc::clone(&built_owners);
        let alpha_catalog = Rc::clone(&catalog_seen_during_build);
        let alpha_group = group.clone();
        let beta_owners = Rc::clone(&built_owners);
        let allocated = ModuleRegistry::new()
            .register(
                ModuleRegistrationSpec::new(
                    alpha.clone(),
                    test_policy(0..=1),
                    ActivationRatio::ONE,
                )
                .with_peer_context(Arc::<str>::from("alpha tools"))
                .in_group(group.clone()),
                move |caps| {
                    let alpha_owners = Rc::clone(&alpha_owners);
                    let alpha_catalog = Rc::clone(&alpha_catalog);
                    let alpha_group = alpha_group.clone();
                    async move {
                        *alpha_catalog.borrow_mut() = caps.module_catalog().members(&alpha_group);
                        alpha_owners.borrow_mut().push(caps.owner().clone());
                        Ok(NoopModule)
                    }
                },
            )
            .unwrap()
            .register(
                ModuleRegistrationSpec::new(
                    beta.clone(),
                    test_policy(0..=1),
                    ActivationRatio::from_f64(0.5),
                )
                .with_peer_context(Arc::<str>::from("beta tools"))
                .in_group(group.clone())
                .depends_on(alpha.clone()),
                move |caps| {
                    let beta_owners = Rc::clone(&beta_owners);
                    async move {
                        beta_owners.borrow_mut().push(caps.owner().clone());
                        Ok(NoopModule)
                    }
                },
            )
            .unwrap()
            .build(&caps)
            .await
            .unwrap();

        assert_eq!(allocated.len(), 2);
        assert_eq!(
            built_owners.borrow().as_slice(),
            &[
                ModuleInstanceId::new(alpha.clone(), ReplicaIndex::ZERO),
                ModuleInstanceId::new(beta.clone(), ReplicaIndex::ZERO),
            ]
        );
        assert_eq!(
            allocated
                .dependencies()
                .deps_of(&ScopedModuleId::new(ScopeId::root(), beta.clone())),
            std::slice::from_ref(&ScopedModuleId::new(ScopeId::root(), alpha.clone()))
        );
        assert_eq!(
            catalog_seen_during_build.borrow().as_slice(),
            &[alpha.clone(), beta.clone()]
        );
        let (alpha_activation, beta_activation) = blackboard
            .read(|bb| {
                (
                    bb.allocation().activation_for(&alpha),
                    bb.allocation().activation_for(&beta),
                )
            })
            .await;
        assert_eq!(alpha_activation, ActivationRatio::ONE);
        assert_eq!(beta_activation, ActivationRatio::from_f64(0.5));
        assert_eq!(
            caps.scoped(ModuleInstanceId::new(alpha, ReplicaIndex::ZERO))
                .module_catalog()
                .members(&group),
            vec![ModuleId::new("mcp-alpha").unwrap(), beta]
        );
    }

    #[tokio::test]
    async fn registry_compiles_activation_barrier_separately_from_settle_dependencies() {
        let caps = test_caps(Blackboard::default());
        let prerequisite = ModuleId::new("barrier-prerequisite").unwrap();
        let dependent = ModuleId::new("barrier-dependent").unwrap();
        let allocated = ModuleRegistry::new()
            .register(
                ModuleRegistrationSpec::new(
                    prerequisite.clone(),
                    test_policy(0..=1),
                    ActivationRatio::ONE,
                ),
                noop_builder,
            )
            .unwrap()
            .register(
                ModuleRegistrationSpec::new(
                    dependent.clone(),
                    test_policy(0..=1),
                    ActivationRatio::ONE,
                )
                .with_activation_barrier([prerequisite.clone()], Some(Duration::from_secs(7))),
                noop_builder,
            )
            .unwrap()
            .build(&caps)
            .await
            .unwrap();

        let dependent = ScopedModuleId::new(ScopeId::root(), dependent);
        let prerequisite = ScopedModuleId::new(ScopeId::root(), prerequisite);
        assert!(allocated.dependencies().deps_of(&dependent).is_empty());
        let barrier = allocated
            .dependencies()
            .activation_barrier_for(&dependent)
            .unwrap();
        assert_eq!(barrier.prerequisites(), std::slice::from_ref(&prerequisite));
        assert_eq!(barrier.timeout(), Some(Duration::from_secs(7)));
    }

    #[tokio::test]
    async fn registry_rejects_cycle_across_dependency_and_activation_barrier_edges() {
        let caps = test_caps(Blackboard::default());
        let alpha = ModuleId::new("barrier-cycle-alpha").unwrap();
        let beta = ModuleId::new("barrier-cycle-beta").unwrap();
        let result = ModuleRegistry::new()
            .register(
                ModuleRegistrationSpec::new(
                    alpha.clone(),
                    test_policy(0..=1),
                    ActivationRatio::ONE,
                )
                .depends_on(beta.clone()),
                noop_builder,
            )
            .unwrap()
            .register(
                ModuleRegistrationSpec::new(beta, test_policy(0..=1), ActivationRatio::ONE)
                    .with_activation_barrier([alpha], None),
                noop_builder,
            )
            .unwrap()
            .build(&caps)
            .await;

        assert!(matches!(
            result,
            Err(ModuleRegistryError::DependencyCycle { .. })
        ));
    }

    #[tokio::test]
    async fn registry_build_preserves_existing_base_activation() {
        let module = ModuleId::new("mcp-existing").unwrap();
        let mut base = ResourceAllocation::default();
        base.set_activation(module.clone(), ActivationRatio::from_f64(0.25));
        let blackboard = Blackboard::with_allocation(base);
        let caps = test_caps(blackboard.clone());

        ModuleRegistry::new()
            .register(
                ModuleRegistrationSpec::new(
                    module.clone(),
                    test_policy(0..=1),
                    ActivationRatio::ONE,
                ),
                noop_builder,
            )
            .unwrap()
            .build(&caps)
            .await
            .unwrap();

        assert_eq!(
            blackboard
                .read(|bb| bb.allocation().activation_for(&module))
                .await,
            ActivationRatio::from_f64(0.25)
        );
    }

    #[tokio::test]
    async fn environment_rejects_a_different_catalog_after_boot() {
        let module = ModuleId::new("mcp-stable").unwrap();
        let blackboard = Blackboard::default();
        let caps = test_caps(blackboard.clone());
        let original =
            ModuleRegistrationSpec::new(module.clone(), test_policy(0..=1), ActivationRatio::ONE)
                .with_peer_context(Arc::<str>::from("stable tools"));

        ModuleRegistry::new()
            .register(original, noop_builder)
            .unwrap()
            .build(&caps)
            .await
            .unwrap();
        let changed = ModuleRegistrationSpec::new(module, test_policy(0..=1), ActivationRatio::ONE)
            .with_peer_context(Arc::<str>::from("changed tools"));

        let result = ModuleRegistry::new()
            .register(changed, noop_builder)
            .unwrap()
            .build(&caps)
            .await;
        let error = match result {
            Ok(_) => panic!("changed catalog unexpectedly rebuilt"),
            Err(error) => error,
        };

        assert!(matches!(
            error,
            ModuleRegistryError::CatalogChangedAfterBoot
        ));
        assert_eq!(
            blackboard.peer_contexts(),
            &[(
                ModuleId::new("mcp-stable").unwrap(),
                Arc::<str>::from("stable tools")
            )]
        );
    }

    #[tokio::test]
    async fn environment_allows_rebuilding_the_same_catalog() {
        let module = ModuleId::new("mcp-rebuild").unwrap();
        let blackboard = Blackboard::default();
        let caps = test_caps(blackboard);

        for _ in 0..2 {
            let spec = ModuleRegistrationSpec::new(
                module.clone(),
                test_policy(0..=1),
                ActivationRatio::ONE,
            )
            .with_peer_context(Arc::<str>::from("rebuild tools"));
            let allocated = ModuleRegistry::new()
                .register(spec, noop_builder)
                .unwrap()
                .build(&caps)
                .await
                .unwrap();
            assert_eq!(allocated.len(), 1);
        }
    }

    #[tokio::test]
    async fn remove_module_omits_build_policy_and_context_catalog() {
        let blackboard = Blackboard::default();
        let caps = test_caps(blackboard.clone());
        let allocated = ModuleRegistry::new()
            .register(static_spec::<NoopModule>(test_policy(0..=1)), noop_builder)
            .unwrap()
            .register(
                static_spec::<NoPeerContextModule>(test_policy(0..=1)),
                no_peer_context_builder,
            )
            .unwrap()
            .remove_module(nuillu_types::ModuleId::new(NoopModule::id()).unwrap())
            .build(&caps)
            .await
            .unwrap();

        let noop = nuillu_types::ModuleId::new(NoopModule::id()).unwrap();
        let no_peer_context = nuillu_types::ModuleId::new(NoPeerContextModule::id()).unwrap();

        assert_eq!(allocated.len(), 1);
        let has_noop_policy = blackboard
            .read(|bb| bb.module_policies().contains_key(&noop))
            .await;
        assert!(!has_noop_policy);
        let has_allocation_only_policy = blackboard
            .read(|bb| bb.module_policies().contains_key(&no_peer_context))
            .await;
        assert!(has_allocation_only_policy);
        assert_eq!(blackboard.peer_contexts().to_vec(), Vec::new());
    }

    #[tokio::test]
    async fn remove_module_prunes_dependency_edges() {
        let blackboard = Blackboard::default();
        let caps = test_caps(blackboard);
        let dependent = nuillu_types::ModuleId::new(NoopModule::id()).unwrap();
        let dependency = nuillu_types::ModuleId::new(NoPeerContextModule::id()).unwrap();
        let allocated = ModuleRegistry::new()
            .register(static_spec::<NoopModule>(test_policy(0..=1)), noop_builder)
            .unwrap()
            .register(
                static_spec::<NoPeerContextModule>(test_policy(0..=1)),
                no_peer_context_builder,
            )
            .unwrap()
            .depends_on(dependent.clone(), dependency.clone())
            .remove_modules([dependency.clone()])
            .build(&caps)
            .await
            .unwrap();

        assert_eq!(allocated.len(), 1);
        assert_eq!(
            allocated
                .dependencies()
                .deps_of(&ScopedModuleId::new(ScopeId::root(), dependent)),
            &[]
        );
        assert_eq!(
            allocated
                .dependencies()
                .dependents_of(&ScopedModuleId::new(ScopeId::root(), dependency)),
            &[]
        );
    }

    #[tokio::test]
    async fn capabilities_are_non_exclusive() {
        let caps = test_caps(Blackboard::default());
        let cognition_gate = scoped(&caps, builtin::cognition_gate(), 0);
        let controller = scoped(&caps, builtin::allocation(), 0);
        let _w1 = cognition_gate.cognition_writer();
        let _w2 = cognition_gate.cognition_writer();
        let _a1 = controller.allocation_writer(vec![builtin::cognition_gate()], Vec::new());
        let _a2 = controller.allocation_writer(vec![builtin::cognition_gate()], Vec::new());
        let _wake1 = cognition_gate.self_wake();
        let _wake2 = cognition_gate.self_wake();
    }

    #[tokio::test]
    async fn self_wake_marks_only_its_owner_pending() {
        let caps = test_caps(Blackboard::default());
        let speak_owner = ModuleInstanceId::new(builtin::speak(), ReplicaIndex::ZERO);
        let memory_owner = ModuleInstanceId::new(builtin::memory(), ReplicaIndex::ZERO);
        let wake = caps.scoped(speak_owner.clone()).self_wake();
        let runtime = caps.runtime_control();

        assert!(!runtime.has_pending_wake(&speak_owner));
        assert!(!runtime.has_pending_self_wake_permit(&speak_owner));
        assert!(!runtime.has_pending_wake(&memory_owner));
        assert!(!runtime.has_pending_self_wake_permit(&memory_owner));

        wake.wake();
        wake.wake();

        assert!(!runtime.has_pending_wake(&speak_owner));
        assert!(runtime.has_pending_self_wake_permit(&speak_owner));
        assert!(!runtime.has_pending_wake(&memory_owner));
        assert!(!runtime.has_pending_self_wake_permit(&memory_owner));
        let claim = runtime
            .claim_self_wake_permit(&speak_owner)
            .expect("self wake should create a permit claim");
        runtime.complete_self_wake_permit_claim(claim);
        assert!(!runtime.has_pending_wake(&speak_owner));
        assert!(!runtime.has_pending_self_wake_permit(&speak_owner));
    }

    #[tokio::test]
    async fn owned_session_checkpoint_saves_each_time() {
        let store = RecordingSessionStore::default();
        let caps = test_caps_with_session_store(Blackboard::default(), Rc::new(store.clone()));
        let owner = ModuleInstanceId::new(builtin::memory(), ReplicaIndex::ZERO);
        let mut session = caps
            .scoped(owner.clone())
            .session("main")
            .await
            .expect("session acquisition should succeed");
        session.push_user("remember this");

        let adapter = Arc::new(lutum::MockLlmAdapter::new());
        let budget = lutum::SharedPoolBudgetManager::new(lutum::SharedPoolBudgetOptions::default());
        let lutum = lutum::Lutum::new(adapter, budget);
        let compaction = crate::SessionCompactionRuntime::new(
            lutum,
            crate::LlmConcurrencyLimiter::new(None),
            ModelTier::Cheap,
            SessionCompactionPolicy::default(),
        );
        let runtime = caps.runtime_control();
        let cx = runtime.with_session_checkpoint_runtime(
            crate::ActivateCx::new(&[], &[], &[], compaction, Rc::new(SystemClock)),
            owner.clone(),
        );

        cx.compact_and_save(&mut session, lutum::Usage::zero())
            .await
            .unwrap();
        let saves = store.saves();
        assert_eq!(saves.len(), 1);
        assert_eq!(saves[0].0, owner);
        assert_eq!(saves[0].1, SessionKey::new("main").unwrap());
        assert_eq!(saves[0].2.items.len(), 1);

        cx.compact_and_save(&mut session, lutum::Usage::zero())
            .await
            .unwrap();
        assert_eq!(store.saves().len(), 2);
    }

    #[tokio::test]
    async fn compact_and_save_restores_metadata_after_session_compaction() {
        let store = RecordingSessionStore::default();
        let caps = test_caps_with_session_store(Blackboard::default(), Rc::new(store.clone()));
        let owner = ModuleInstanceId::new(builtin::memory(), ReplicaIndex::ZERO);
        let mut session = caps
            .scoped(owner.clone())
            .session("main")
            .with_auto_compaction(SessionAutoCompaction::new(
                SessionCompactionConfig::default(),
                SessionCompactionProtectedPrefix::LeadingSystem,
                "Compacted session:",
                "Preserve test facts.",
            ))
            .await
            .expect("session acquisition should succeed");
        session.push_system("SYSTEM PROMPT");
        for index in 0..5 {
            session.push_user(format!("history-{index}"));
        }

        let adapter = Arc::new(
            MockLlmAdapter::new().with_text_scenario(MockTextScenario::events(vec![
                Ok(RawTextTurnEvent::Started {
                    request_id: Some("compact".into()),
                    model: "mock".into(),
                }),
                Ok(RawTextTurnEvent::TextDelta {
                    delta: "history summarized".into(),
                }),
                Ok(RawTextTurnEvent::Completed {
                    request_id: Some("compact".into()),
                    finish_reason: FinishReason::Stop,
                    usage: lutum::Usage::zero(),
                }),
            ])),
        );
        let budget = lutum::SharedPoolBudgetManager::new(lutum::SharedPoolBudgetOptions::default());
        let lutum = lutum::Lutum::new(adapter, budget);
        let compaction = crate::SessionCompactionRuntime::new(
            lutum,
            crate::LlmConcurrencyLimiter::new(None),
            ModelTier::Cheap,
            SessionCompactionPolicy::new(1, 1, 1),
        );
        let runtime = caps.runtime_control();
        let cx = runtime.with_session_checkpoint_runtime(
            crate::ActivateCx::new(&[], &[], &[], compaction, Rc::new(SystemClock)),
            owner.clone(),
        );

        cx.compact_and_save(
            &mut session,
            lutum::Usage {
                input_tokens: 2,
                ..lutum::Usage::zero()
            },
        )
        .await
        .expect("first checkpoint after compaction should succeed");
        assert!(
            persistent_session_metadata(&session).is_some(),
            "session metadata should survive compaction"
        );

        cx.compact_and_save(&mut session, lutum::Usage::zero())
            .await
            .expect("second checkpoint should succeed after metadata restore");
    }

    #[test]
    fn activate_cx_warn_emits_module_warning() {
        let sink = Rc::new(RecordingRuntimeEventSink::default());
        let owner = ModuleInstanceId::new(builtin::memory(), ReplicaIndex::ZERO);
        let cx = crate::ActivateCx::new(
            &[],
            &[],
            &[],
            crate::SessionCompactionRuntime::new(
                lutum::Lutum::new(
                    Arc::new(lutum::MockLlmAdapter::new()),
                    lutum::SharedPoolBudgetManager::new(lutum::SharedPoolBudgetOptions::default()),
                ),
                crate::LlmConcurrencyLimiter::new(None),
                ModelTier::Cheap,
                SessionCompactionPolicy::default(),
            ),
            Rc::new(SystemClock),
        )
        .with_session_checkpoint_runtime(
            Rc::new(NoopSessionStore),
            RuntimeEventEmitter::new(sink.clone()),
            owner.clone(),
        );

        cx.warn("decision attempt failed: no tool call");

        assert_eq!(sink.events().len(), 1);
        assert_eq!(
            sink.events()[0],
            RuntimeEvent::ModuleWarning {
                sequence: 0,
                owner,
                message: "decision attempt failed: no tool call".to_owned(),
            }
        );
    }

    #[tokio::test]
    async fn cognition_writer_appends_persists_publishes_and_owner_stamps() {
        let blackboard = Blackboard::default();
        let repo = RecordingCognitionLogRepository::default();
        let caps = test_caps_with_cognition_repo(blackboard.clone(), Rc::new(repo.clone()));
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
        assert!(records[0].0.is_root());
        assert_eq!(records[0].1, owner);
        assert_eq!(records[0].2.text, "food boundary changed");

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
        let reader = cognition_gate.blackboard_reader();

        cognition_gate.memo().write("own memo").await;
        sensory.memo().write("sensory memo").await;

        let event = inbox.next_item().await.unwrap();
        assert_eq!(event.sender.module, builtin::sensory());
        assert_eq!(event.body.owner.module, builtin::sensory());
        assert!(inbox.take_ready_items().unwrap().items.is_empty());

        let unread = reader.unread_memo_logs().await;
        assert_eq!(
            unread
                .iter()
                .map(|record| (record.owner.clone(), record.content.as_str()))
                .collect::<Vec<_>>(),
            vec![(event.sender, "sensory memo")]
        );
        assert!(reader.unread_memo_logs().await.is_empty());
    }

    #[tokio::test]
    async fn memo_updated_inbox_filters_registered_source_roles() {
        let caps = test_caps(Blackboard::default());
        let self_model_owner = ModuleInstanceId::new(builtin::self_model(), ReplicaIndex::ZERO);
        let filtered = caps.scoped_with_memo_subscription(
            self_model_owner,
            MemoSubscription::only([builtin::query_memory()]),
        );
        let sensory = scoped(&caps, builtin::sensory(), 0);
        let query_memory = scoped(&caps, builtin::query_memory(), 0);
        let mut inbox = filtered.memo_updated_inbox();

        sensory.memo().write("external context").await;
        assert!(inbox.take_ready_items().unwrap().items.is_empty());

        query_memory.memo().write("identity evidence").await;
        let event = inbox.next_item().await.unwrap();
        assert_eq!(event.body.owner.module, builtin::query_memory());
    }

    #[tokio::test]
    async fn memo_updated_inbox_can_round_robin_across_active_replicas() {
        let mut allocation = ResourceAllocation::default();
        allocation.set_activation(builtin::predict(), ActivationRatio::ONE);
        let blackboard = Blackboard::with_allocation(allocation);
        blackboard
            .apply(BlackboardCommand::SetModulePolicies {
                policies: vec![(builtin::predict(), test_policy(0..=2))],
            })
            .await;
        let caps = test_caps(blackboard);
        let sensory = scoped(&caps, builtin::sensory(), 0);
        let mut predict_0 = scoped(&caps, builtin::predict(), 0)
            .memo_updated_inbox()
            .round_robin();
        let mut predict_1 = scoped(&caps, builtin::predict(), 1)
            .memo_updated_inbox()
            .round_robin();
        let memo = sensory.memo();

        memo.write("first").await;
        memo.write("second").await;

        assert_eq!(predict_0.next_item().await.unwrap().body.index, 0);
        assert_eq!(predict_1.next_item().await.unwrap().body.index, 1);
        assert!(predict_0.take_ready_items().unwrap().items.is_empty());
        assert!(predict_1.take_ready_items().unwrap().items.is_empty());
    }

    #[tokio::test]
    async fn memo_updated_inbox_broadcast_remains_the_default() {
        let mut allocation = ResourceAllocation::default();
        allocation.set_activation(builtin::predict(), ActivationRatio::ONE);
        let blackboard = Blackboard::with_allocation(allocation);
        blackboard
            .apply(BlackboardCommand::SetModulePolicies {
                policies: vec![(builtin::predict(), test_policy(0..=2))],
            })
            .await;
        let caps = test_caps(blackboard);
        let sensory = scoped(&caps, builtin::sensory(), 0);
        let mut predict_0 = scoped(&caps, builtin::predict(), 0).memo_updated_inbox();
        let mut predict_1 = scoped(&caps, builtin::predict(), 1)
            .memo_updated_inbox()
            .broadcast();

        sensory.memo().write("shared").await;

        assert_eq!(predict_0.next_item().await.unwrap().body.index, 0);
        assert_eq!(predict_1.next_item().await.unwrap().body.index, 0);
    }

    #[tokio::test]
    async fn coalesced_inbox_keeps_one_unread_activation_signal() {
        let caps = test_caps(Blackboard::default());
        let sensory = scoped(&caps, builtin::sensory(), 0);
        let mut predict = scoped(&caps, builtin::predict(), 0)
            .memo_updated_inbox()
            .coalesce();
        let memo = sensory.memo();

        memo.write("first").await;
        memo.write("second").await;
        memo.write("third").await;

        let ready = predict.take_ready_items().unwrap();
        assert_eq!(ready.items.len(), 1);
        assert_eq!(ready.items[0].body.index, 0);

        memo.write("after drain").await;
        assert_eq!(predict.next_item().await.unwrap().body.index, 3);
    }

    #[tokio::test]
    async fn round_robin_memo_inboxes_share_unread_cursor_by_role() {
        let mut allocation = ResourceAllocation::default();
        allocation.set_activation(builtin::predict(), ActivationRatio::ONE);
        let blackboard = Blackboard::with_allocation(allocation);
        blackboard
            .apply(BlackboardCommand::SetModulePolicies {
                policies: vec![(builtin::predict(), test_policy(0..=2))],
            })
            .await;
        let caps = test_caps(blackboard);
        let predict_0 = scoped(&caps, builtin::predict(), 0);
        let predict_1 = scoped(&caps, builtin::predict(), 1);
        // Resolve readers before configuring the inboxes to prove that cursor
        // selection is independent of constructor argument order.
        let reader_0 = predict_0.blackboard_reader();
        let reader_1 = predict_1.blackboard_reader();
        let mut inbox_0 = predict_0.memo_updated_inbox().round_robin();
        let mut inbox_1 = predict_1.memo_updated_inbox().round_robin();
        let sensory = scoped(&caps, builtin::sensory(), 0);
        let memo = sensory.memo();

        memo.write("first").await;
        assert_eq!(inbox_0.next_item().await.unwrap().body.index, 0);
        assert_eq!(reader_0.unread_memo_logs().await[0].content, "first");

        memo.write("second").await;
        assert_eq!(inbox_1.next_item().await.unwrap().body.index, 1);
        let unread = reader_1.unread_memo_logs().await;
        assert_eq!(unread.len(), 1);
        assert_eq!(unread[0].content, "second");
        assert!(reader_0.unread_memo_logs().await.is_empty());
    }

    #[tokio::test]
    async fn round_robin_cognition_inboxes_share_unread_cursor_by_role() {
        let mut allocation = ResourceAllocation::default();
        allocation.set_activation(builtin::predict(), ActivationRatio::ONE);
        let blackboard = Blackboard::with_allocation(allocation);
        blackboard
            .apply(BlackboardCommand::SetModulePolicies {
                policies: vec![(builtin::predict(), test_policy(0..=2))],
            })
            .await;
        let caps = test_caps(blackboard);
        let predict_0 = scoped(&caps, builtin::predict(), 0);
        let predict_1 = scoped(&caps, builtin::predict(), 1);
        let reader_0 = predict_0.cognition_log_reader();
        let reader_1 = predict_1.cognition_log_reader();
        let mut inbox_0 = predict_0.cognition_log_updated_inbox().round_robin();
        let mut inbox_1 = predict_1.cognition_log_updated_inbox().round_robin();
        let cognition_gate = scoped(&caps, builtin::cognition_gate(), 0);
        let cognition = cognition_gate.cognition_writer();

        cognition.append("first").await;
        assert!(matches!(
            inbox_0.next_item().await.unwrap().body,
            CognitionLogUpdated::EntryAppended { .. }
        ));
        assert_eq!(reader_0.unread_events().await[0].entry.text, "first");

        cognition.append("second").await;
        assert!(matches!(
            inbox_1.next_item().await.unwrap().body,
            CognitionLogUpdated::EntryAppended { .. }
        ));
        let unread = reader_1.unread_events().await;
        assert_eq!(unread.len(), 1);
        assert_eq!(unread[0].entry.text, "second");
        assert!(reader_0.unread_events().await.is_empty());
    }

    #[tokio::test]
    async fn typed_memo_writes_plaintext_publishes_and_keeps_typed_payload() {
        #[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
        struct TestMemoPayload {
            value: String,
        }

        let blackboard = Blackboard::default();
        let caps = test_caps(blackboard.clone());
        let query_memory = scoped(&caps, builtin::query_memory(), 0);
        let cognition_gate = scoped(&caps, builtin::cognition_gate(), 0);
        let owner = ModuleInstanceId::new(builtin::query_memory(), ReplicaIndex::ZERO);
        let mut inbox = cognition_gate.memo_updated_inbox();

        query_memory
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

    #[tokio::test]
    async fn outer_typed_memo_targets_parent_without_changing_owner_stamp() {
        #[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
        struct BridgeState {
            direction: String,
        }

        let blackboard = Blackboard::default();
        let repo = RecordingMemoLogRepository::default();
        let caps = test_caps_with_memo_log_repository(
            blackboard.clone(),
            Rc::new(repo.clone()),
            RuntimePolicy::default(),
        );
        let child_scope = ScopeId::root().child(SubsystemInstanceId::new(
            SubsystemId::new("arm").unwrap(),
            ReplicaIndex::ZERO,
        ));
        let owner = ModuleInstanceId::in_scope(
            child_scope.clone(),
            builtin::subsystem_gate(),
            ReplicaIndex::ZERO,
        );
        let gate = caps.scoped(owner.clone());
        let outer = gate.outer_typed_memo::<BridgeState>().unwrap();
        let inner = gate.typed_memo::<BridgeState>();

        inner
            .write_cognitive(
                BridgeState {
                    direction: "in".to_owned(),
                },
                "parent cognition",
            )
            .await;
        outer
            .write_cognitive(
                BridgeState {
                    direction: "out".to_owned(),
                },
                "child cognition",
            )
            .await;

        assert_eq!(
            blackboard
                .scoped(child_scope.clone())
                .read(|bb| bb.recent_memo_logs())
                .await
                .into_iter()
                .map(|record| (record.owner, record.content, record.cognitive))
                .collect::<Vec<_>>(),
            vec![(owner.clone(), "parent cognition".to_owned(), true)]
        );
        assert_eq!(
            blackboard
                .scoped(ScopeId::root())
                .read(|bb| bb.recent_memo_logs())
                .await
                .into_iter()
                .map(|record| (record.owner, record.content, record.cognitive))
                .collect::<Vec<_>>(),
            vec![(owner, "child cognition".to_owned(), true)]
        );
        assert_eq!(
            repo.appends()
                .into_iter()
                .map(|entry| entry.scope)
                .collect::<Vec<_>>(),
            vec![child_scope, ScopeId::root()]
        );
    }

    #[tokio::test]
    async fn memo_write_persists_plain_payload() {
        let blackboard = Blackboard::default();
        let repo = RecordingMemoLogRepository::default();
        let caps = test_caps_with_memo_log_repository(
            blackboard,
            Rc::new(repo.clone()),
            RuntimePolicy::default(),
        );
        let sensory = scoped(&caps, builtin::sensory(), 0);
        let owner = ModuleInstanceId::new(builtin::sensory(), ReplicaIndex::ZERO);

        sensory.memo().write_cognitive("salient memo").await;

        let appends = repo.appends();
        assert_eq!(
            appends,
            vec![PersistedMemoLogEntry {
                scope: ScopeId::root(),
                record: MemoLogRecord {
                    owner,
                    index: 0,
                    written_at: appends[0].record.written_at,
                    content: "salient memo".to_owned(),
                    cognitive: true,
                },
                payload: MemoLogPayload::Plain,
            }]
        );
    }

    #[tokio::test]
    async fn typed_memo_write_persists_json_payload() {
        #[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
        struct TestMemoPayload {
            value: String,
        }

        let blackboard = Blackboard::default();
        let repo = RecordingMemoLogRepository::default();
        let caps = test_caps_with_memo_log_repository(
            blackboard,
            Rc::new(repo.clone()),
            RuntimePolicy::default(),
        );
        let query_memory = scoped(&caps, builtin::query_memory(), 0);
        let owner = ModuleInstanceId::new(builtin::query_memory(), ReplicaIndex::ZERO);

        query_memory
            .typed_memo::<TestMemoPayload>()
            .write_cognitive(
                TestMemoPayload {
                    value: "typed".into(),
                },
                "typed memo",
            )
            .await;

        let appends = repo.appends();
        assert_eq!(appends.len(), 1);
        assert_eq!(appends[0].scope, ScopeId::root());
        assert_eq!(appends[0].record.owner, owner);
        assert_eq!(appends[0].record.index, 0);
        assert!(appends[0].record.cognitive);
        assert_eq!(appends[0].record.content, "typed memo");
        assert_eq!(
            appends[0].payload,
            MemoLogPayload::Typed {
                type_name: std::any::type_name::<TestMemoPayload>().to_owned(),
                json: serde_json::json!({ "value": "typed" }),
                forwarded_cognition: None,
            }
        );
    }

    #[tokio::test]
    async fn registry_build_restores_recent_memo_log_entries() {
        let blackboard = Blackboard::default();
        let owner = ModuleInstanceId::new(builtin::sensory(), ReplicaIndex::ZERO);
        let repo = RecordingMemoLogRepository::with_records(vec![
            PersistedMemoLogEntry {
                scope: ScopeId::root(),
                record: MemoLogRecord {
                    owner: owner.clone(),
                    index: 0,
                    written_at: Utc::now() - chrono::Duration::seconds(2),
                    content: "old memo".to_owned(),
                    cognitive: false,
                },
                payload: MemoLogPayload::Plain,
            },
            PersistedMemoLogEntry {
                scope: ScopeId::root(),
                record: MemoLogRecord {
                    owner: owner.clone(),
                    index: 1,
                    written_at: Utc::now() - chrono::Duration::seconds(1),
                    content: "kept memo".to_owned(),
                    cognitive: true,
                },
                payload: MemoLogPayload::Plain,
            },
        ]);
        let caps = test_caps_with_memo_log_repository(
            blackboard.clone(),
            Rc::new(repo),
            RuntimePolicy {
                memo_retained_per_owner: 1,
                ..RuntimePolicy::default()
            },
        );

        ModuleRegistry::new().build(&caps).await.unwrap();

        let memos = blackboard.read(|bb| bb.recent_memo_logs()).await;
        assert_eq!(memos.len(), 1);
        assert_eq!(memos[0].index, 1);
        assert_eq!(memos[0].content, "kept memo");
        assert!(memos[0].cognitive);

        let next = blackboard
            .update_memo(owner, "after restore".to_owned(), Utc::now())
            .await;
        assert_eq!(next.index, 2);
    }

    #[tokio::test]
    async fn memo_log_evicted_inbox_receives_evicted_records() {
        let blackboard = Blackboard::default();
        blackboard
            .apply(BlackboardCommand::SetMemoRetentionPerOwner(1))
            .await;
        let caps = test_caps(blackboard);
        let sensory = scoped(&caps, builtin::sensory(), 0);
        let policy = scoped(&caps, builtin::policy(), 0);
        let memo = sensory.memo();
        let mut inbox = policy.memo_log_evicted_inbox();

        memo.write("first").await;
        memo.write("second").await;

        let event = inbox.next_item().await.unwrap();
        assert_eq!(event.sender.module, builtin::sensory());
        assert_eq!(event.body.owner.module, builtin::sensory());
        assert_eq!(event.body.index, 0);
        assert_eq!(event.body.content, "first");
    }

    #[tokio::test]
    async fn cognition_log_evicted_inbox_receives_evicted_records() {
        let blackboard = Blackboard::default();
        blackboard
            .apply(BlackboardCommand::SetCognitionLogRetentionEntries(1))
            .await;
        let caps = test_caps(blackboard);
        let cognition_gate = scoped(&caps, builtin::cognition_gate(), 0);
        let memory = scoped(&caps, builtin::memory(), 0);
        let writer = cognition_gate.cognition_writer();
        let mut inbox = memory.cognition_log_evicted_inbox();

        writer.append("first cognition").await;
        writer.append("second cognition").await;

        let event = inbox.next_item().await.unwrap();
        assert_eq!(event.sender.module, builtin::cognition_gate());
        assert_eq!(event.body.source.module, builtin::cognition_gate());
        assert_eq!(event.body.index, 0);
        assert_eq!(event.body.entry.text, "first cognition");
    }

    #[test]
    #[should_panic(
        expected = "module requested multiple memo capabilities; choose exactly one of memo() or typed_memo::<T>()"
    )]
    fn memo_then_typed_memo_panics() {
        let caps = test_caps(Blackboard::default());
        let query_memory = scoped(&caps, builtin::query_memory(), 0);

        let _plain = query_memory.memo();
        let _typed = query_memory.typed_memo::<u8>();
    }

    #[test]
    #[should_panic(
        expected = "module requested multiple memo capabilities; choose exactly one of memo() or typed_memo::<T>()"
    )]
    fn typed_memo_then_typed_memo_panics() {
        let caps = test_caps(Blackboard::default());
        let query_memory = scoped(&caps, builtin::query_memory(), 0);

        let _first = query_memory.typed_memo::<u8>();
        let _second = query_memory.typed_memo::<u16>();
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
    async fn allocation_writer_records_activation_changes() {
        let blackboard = Blackboard::default();
        blackboard
            .apply(BlackboardCommand::SetModulePolicies {
                policies: vec![
                    (
                        builtin::allocation(),
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
        let store = RecordingAllocationStore::default();
        let caps = test_caps_with_allocation_store(blackboard.clone(), Rc::new(store.clone()));
        let controller = scoped(&caps, builtin::allocation(), 0);
        let writer = controller.allocation_writer(vec![builtin::cognition_gate()], Vec::new());

        let commands = vec![AllocationCommand::target(
            builtin::cognition_gate(),
            AllocationEffectLevel::Max,
        )];

        writer.submit(commands.clone()).await.unwrap();

        let allocation = blackboard.read(|bb| bb.allocation().clone()).await;
        assert_eq!(
            allocation.activation_for(&builtin::cognition_gate()),
            ActivationRatio::ONE
        );

        let mut unchanged = blackboard.allocation_change_waiter().await;
        writer.submit(commands).await.unwrap();
        assert!(
            unchanged.try_recv().is_err(),
            "an identical proposal should not notify allocation change waiters"
        );
        assert_eq!(store.saves().len(), 1);

        let changed = blackboard.allocation_change_waiter().await;
        writer
            .submit([AllocationCommand::target(
                builtin::cognition_gate(),
                AllocationEffectLevel::High,
            )])
            .await
            .unwrap();
        assert_eq!(changed.await, Ok(()));
        assert_eq!(store.saves().len(), 2);
        assert_eq!(
            blackboard
                .read(|bb| {
                    bb.allocation_proposals()
                        .get(&ModuleInstanceId::new(
                            builtin::allocation(),
                            ReplicaIndex::ZERO,
                        ))
                        .cloned()
                })
                .await
                .expect("allocation proposal should be recorded")
                .activation_for(&builtin::cognition_gate()),
            ActivationRatio::from_f64(0.85)
        );
    }

    #[tokio::test]
    async fn subsystem_writer_uses_each_mounts_activation_table_and_zeros_omissions() {
        let blackboard = Blackboard::default();
        let arm = SubsystemId::new("arm").unwrap();
        let eye = SubsystemId::new("eye").unwrap();
        let policy = SubsystemPolicy::new(
            SubsystemReplicaRange::new(0, 1).unwrap(),
            1,
            nuillu_blackboard::ReplicaProjection::Linear,
        );
        blackboard
            .apply(BlackboardCommand::SetRegisteredSubsystems {
                registrations: vec![
                    RegisteredSubsystemPolicy {
                        subsystem: arm.clone(),
                        policy: policy.clone(),
                        initial_activation: ActivationRatio::ZERO,
                    },
                    RegisteredSubsystemPolicy {
                        subsystem: eye.clone(),
                        policy,
                        initial_activation: ActivationRatio::ZERO,
                    },
                ],
            })
            .await;
        let caps = test_caps(blackboard.clone());
        let writer =
            scoped(&caps, builtin::subsystem_allocation(), 0).subsystem_allocation_writer(vec![
                (
                    arm.clone(),
                    vec![ActivationRatio::ONE, ActivationRatio::from_f64(0.8)],
                ),
                (
                    eye.clone(),
                    vec![ActivationRatio::ONE, ActivationRatio::from_f64(0.4)],
                ),
            ]);

        writer
            .submit([
                nuillu_blackboard::SubsystemAllocationCommand::target(
                    arm.clone(),
                    AllocationEffectLevel::High,
                ),
                nuillu_blackboard::SubsystemAllocationCommand::target(
                    eye.clone(),
                    AllocationEffectLevel::High,
                ),
            ])
            .await
            .unwrap();
        assert_eq!(
            blackboard
                .read(|bb| bb.subsystem_allocation().activation_for(&arm))
                .await,
            ActivationRatio::from_f64(0.8)
        );
        assert_eq!(
            blackboard
                .read(|bb| bb.subsystem_allocation().activation_for(&eye))
                .await,
            ActivationRatio::from_f64(0.4)
        );

        writer
            .submit([nuillu_blackboard::SubsystemAllocationCommand::target(
                arm.clone(),
                AllocationEffectLevel::Max,
            )])
            .await
            .unwrap();
        assert_eq!(
            blackboard
                .read(|bb| bb.subsystem_allocation().activation_for(&eye))
                .await,
            ActivationRatio::ZERO
        );
    }

    #[tokio::test]
    async fn allocation_writer_persists_owner_scoped_snapshot() {
        let blackboard = Blackboard::default();
        let store = RecordingAllocationStore::default();
        let caps = test_caps_with_allocation_store(blackboard, Rc::new(store.clone()));
        let owner = ModuleInstanceId::new(builtin::allocation(), ReplicaIndex::ZERO);
        let controller = caps.scoped(owner.clone());
        let writer =
            controller.allocation_writer(vec![builtin::cognition_gate()], vec![builtin::speak()]);

        writer
            .submit([
                AllocationCommand::target(builtin::cognition_gate(), AllocationEffectLevel::Max),
                AllocationCommand::suppression(builtin::speak(), AllocationEffectLevel::High),
            ])
            .await
            .unwrap();

        let mut targets = ResourceAllocation::default();
        targets.set_activation(builtin::cognition_gate(), ActivationRatio::ONE);
        let mut suppressions = ResourceAllocation::default();
        suppressions.set_activation(builtin::speak(), ActivationRatio::from_f64(0.10));

        assert_eq!(
            store.saves(),
            vec![PersistedAllocationSnapshot::new(
                owner,
                targets,
                suppressions
            )]
        );
    }

    #[tokio::test]
    async fn registry_build_restores_persisted_allocation_snapshots() {
        let owner =
            ModuleInstanceId::new(ModuleId::new(NoopModule::id()).unwrap(), ReplicaIndex::ZERO);
        let target = ModuleId::new(NoPeerContextModule::id()).unwrap();
        let mut targets = ResourceAllocation::default();
        targets.set_activation(target.clone(), ActivationRatio::ONE);
        let snapshot =
            PersistedAllocationSnapshot::new(owner.clone(), targets, ResourceAllocation::default());
        let store = RecordingAllocationStore::with_snapshots(vec![snapshot]);
        let mut base = ResourceAllocation::default();
        base.set_activation(owner.module.clone(), ActivationRatio::ONE);
        base.set_activation(target.clone(), ActivationRatio::ZERO);
        let blackboard = Blackboard::with_allocation(base);
        let caps = test_caps_with_allocation_store(blackboard.clone(), Rc::new(store));

        ModuleRegistry::new()
            .register(static_spec::<NoopModule>(test_policy(0..=1)), noop_builder)
            .unwrap()
            .register(
                static_spec::<NoPeerContextModule>(test_policy(0..=1)),
                no_peer_context_builder,
            )
            .unwrap()
            .build(&caps)
            .await
            .unwrap();

        let allocation = blackboard.read(|bb| bb.allocation().clone()).await;
        assert_eq!(allocation.activation_for(&target), ActivationRatio::ONE);
    }

    #[tokio::test]
    async fn registry_build_restores_recent_cognition_log_entries() {
        let blackboard = Blackboard::default();
        let owner_a = ModuleInstanceId::new(builtin::cognition_gate(), ReplicaIndex::ZERO);
        let owner_b = ModuleInstanceId::new(builtin::attention_schema(), ReplicaIndex::ZERO);
        let now = Utc::now();
        let repo = RecordingCognitionLogRepository::with_records(vec![
            (
                owner_a.clone(),
                CognitionLogEntry {
                    at: now - chrono::Duration::seconds(3),
                    text: "old cognition".to_owned(),
                    origin: CognitionLogOrigin::direct(owner_a.clone()),
                },
            ),
            (
                owner_b.clone(),
                CognitionLogEntry {
                    at: now - chrono::Duration::seconds(2),
                    text: "recent cognition".to_owned(),
                    origin: CognitionLogOrigin::direct(owner_b),
                },
            ),
            (
                owner_a.clone(),
                CognitionLogEntry {
                    at: now - chrono::Duration::seconds(1),
                    text: "newest cognition".to_owned(),
                    origin: CognitionLogOrigin::direct(owner_a),
                },
            ),
        ]);
        let caps = test_caps_with_cognition_repo_and_runtime(
            blackboard.clone(),
            Rc::new(repo),
            CapabilityProviderRuntime {
                policy: RuntimePolicy {
                    cognition_log_retained_entries: 2,
                    ..RuntimePolicy::default()
                },
                ..CapabilityProviderRuntime::default()
            },
        );

        ModuleRegistry::new()
            .register(static_spec::<NoopModule>(test_policy(0..=1)), noop_builder)
            .unwrap()
            .build(&caps)
            .await
            .unwrap();

        let entries = blackboard
            .read(|bb| bb.cognition_log().entries().to_vec())
            .await;
        assert_eq!(
            entries
                .iter()
                .map(|entry| entry.text.as_str())
                .collect::<Vec<_>>(),
            vec!["recent cognition", "newest cognition"]
        );
    }

    #[tokio::test]
    async fn registry_build_skips_cognition_restore_when_blackboard_is_not_empty() {
        let blackboard = Blackboard::default();
        let owner = ModuleInstanceId::new(builtin::cognition_gate(), ReplicaIndex::ZERO);
        let now = Utc::now();
        blackboard
            .apply(BlackboardCommand::AppendCognitionLog {
                source: owner.clone(),
                entry: CognitionLogEntry {
                    at: now,
                    text: "seeded cognition".to_owned(),
                    origin: CognitionLogOrigin::direct(owner.clone()),
                },
            })
            .await;
        let repo = RecordingCognitionLogRepository::with_records(vec![(
            owner.clone(),
            CognitionLogEntry {
                at: now - chrono::Duration::seconds(1),
                text: "persisted cognition".to_owned(),
                origin: CognitionLogOrigin::direct(owner.clone()),
            },
        )]);
        let caps = test_caps_with_cognition_repo(blackboard.clone(), Rc::new(repo));

        ModuleRegistry::new()
            .register(static_spec::<NoopModule>(test_policy(0..=1)), noop_builder)
            .unwrap()
            .build(&caps)
            .await
            .unwrap();

        let entries = blackboard
            .read(|bb| bb.cognition_log().entries().to_vec())
            .await;
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].text, "seeded cognition");
    }

    #[tokio::test]
    async fn allocation_writer_applies_only_allowed_command_kinds() {
        let mut base = ResourceAllocation::default();
        base.set_activation(builtin::allocation(), ActivationRatio::ONE);
        base.set_activation(builtin::cognition_gate(), ActivationRatio::ONE);
        base.set_activation(builtin::speak(), ActivationRatio::ZERO);
        let blackboard = Blackboard::with_allocation(base);
        blackboard
            .apply(BlackboardCommand::SetModulePolicies {
                policies: vec![
                    (
                        builtin::allocation(),
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
                    (
                        builtin::speak(),
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
        let caps = test_caps(blackboard.clone());
        let controller = scoped(&caps, builtin::allocation(), 0);
        let writer =
            controller.allocation_writer(vec![builtin::speak()], vec![builtin::cognition_gate()]);

        writer
            .submit([
                AllocationCommand::target(builtin::speak(), AllocationEffectLevel::Max),
                AllocationCommand::target(builtin::cognition_gate(), AllocationEffectLevel::Max),
                AllocationCommand::suppression(
                    builtin::cognition_gate(),
                    AllocationEffectLevel::High,
                ),
                AllocationCommand::suppression(builtin::speak(), AllocationEffectLevel::Max),
            ])
            .await
            .unwrap();

        let allocation = blackboard.read(|bb| bb.allocation().clone()).await;
        assert_eq!(
            allocation.activation_for(&builtin::speak()),
            ActivationRatio::ONE
        );
        assert_eq!(
            allocation.activation_for(&builtin::cognition_gate()),
            ActivationRatio::from_f64(0.10)
        );
    }

    #[tokio::test]
    async fn cognition_log_updates_do_not_wake_controller_memo_inbox() {
        let caps = test_caps(Blackboard::default());
        let controller = scoped(&caps, builtin::allocation(), 0);
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
        let controller = scoped(&caps, builtin::allocation(), 0);
        let speak = scoped(&caps, builtin::speak(), 0);
        let mut memo_updates = controller.memo_updated_inbox();

        speak.memo().write("utterance completed").await;

        let event = memo_updates.next_item().await.unwrap();
        assert_eq!(event.sender.module, builtin::speak());
        assert_eq!(event.body.owner.module, builtin::speak());
    }
}
