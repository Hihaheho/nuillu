use std::cell::Cell;
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::future::{Future, IntoFuture};
use std::pin::Pin;
use std::rc::Rc;
use std::time::Duration;

use lutum::{Lutum, Session};
use nuillu_blackboard::{
    ActivationRatio, AgenticDeadlockMarker, Blackboard, BlackboardCommand, ModulePolicy,
    ModuleRunStatus, ZeroReplicaWindowPolicy,
};
use nuillu_types::{ModelTier, ModuleId, ModuleInstanceId, ReplicaCapRange, ReplicaIndex};

use crate::activation_gate::ActivationGateHub;
use crate::channels::{Topic, TopicPolicy};
use crate::ports::{Clock, CognitionLogRepository, PortError};
use crate::rate_limit::{RateLimiter, RuntimePolicy, TopicKind};
use crate::runtime_events::{NoopRuntimeEventSink, RuntimeEventEmitter, RuntimeEventSink};
use crate::scene::{SceneReader, SceneRegistry};
use crate::session::{
    NoopSessionStore, SessionAutoCompaction, SessionKey, SessionStore,
    attach_persistent_session_metadata,
};
use crate::tiers::{LlmTierHandle, LutumTiers};
use crate::r#trait::ErasedModule;
use crate::{
    AllocationReader, AllocationWriter, AttentionControlRequest, AttentionControlRequestInbox,
    AttentionControlRequestMailbox, BlackboardReader, CognitionLogEvictedInbox,
    CognitionLogEvictedMailbox, CognitionLogReader, CognitionLogUpdated, CognitionLogUpdatedInbox,
    CognitionLogUpdatedMailbox, CognitionWriter, InteroceptionRuntimePolicy, InteroceptiveReader,
    InteroceptiveUpdated, InteroceptiveUpdatedInbox, InteroceptiveUpdatedMailbox,
    InteroceptiveWriter, LlmAccess, Memo, MemoLogEvictedInbox, MemoLogEvictedMailbox, MemoUpdated,
    MemoUpdatedInbox, MemoUpdatedMailbox, MemoryMetadataReader, Module, ModuleBatch,
    ModuleStatusReader, SensoryInput, SensoryInputInbox, SensoryInputMailbox,
    SessionCompactionPolicy, TimeDivision, TopicInbox, TopicMailbox, TypedMemo,
};

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
    attention_control_requests: Topic<AttentionControlRequest>,
    cognition_log_updates: Topic<CognitionLogUpdated>,
    cognition_log_evictions: Topic<nuillu_blackboard::CognitionLogEntryRecord>,
    interoception_updates: Topic<InteroceptiveUpdated>,
    memo_updates: Topic<MemoUpdated>,
    memo_log_evictions: Topic<nuillu_blackboard::MemoLogRecord>,
    sensory_input_topic: Topic<SensoryInput>,
    activation_gates: ActivationGateHub,
    cognition_log_port: Rc<dyn CognitionLogRepository>,
    clock: Rc<dyn Clock>,
    time_division: TimeDivision,
    tiers: LutumTiers,
    runtime_events: RuntimeEventEmitter,
    rate_limiter: RateLimiter,
    runtime_policy: RuntimePolicy,
    scene: SceneRegistry,
    session_store: Rc<dyn SessionStore>,
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
    pub event_sink: Rc<dyn RuntimeEventSink>,
    pub policy: RuntimePolicy,
    pub session_store: Rc<dyn SessionStore>,
}

impl Default for CapabilityProviderRuntime {
    fn default() -> Self {
        Self {
            event_sink: Rc::new(NoopRuntimeEventSink),
            policy: RuntimePolicy::default(),
            session_store: Rc::new(NoopSessionStore),
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
            event_sink,
            policy,
            session_store,
        } = runtime;
        let runtime_events = RuntimeEventEmitter::new(event_sink);
        let rate_limiter = RateLimiter::new(policy.rate_limits.clone());
        Self {
            inner: Rc::new(CapabilityProvidersInner {
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
                cognition_log_evictions: Topic::new(
                    blackboard.clone(),
                    TopicPolicy::Fanout,
                    TopicKind::CognitionLogEvicted,
                    rate_limiter.clone(),
                    runtime_events.clone(),
                ),
                interoception_updates: Topic::new(
                    blackboard.clone(),
                    TopicPolicy::Fanout,
                    TopicKind::InteroceptiveUpdated,
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
                memo_log_evictions: Topic::new(
                    blackboard.clone(),
                    TopicPolicy::Fanout,
                    TopicKind::MemoLogEvicted,
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
                runtime_policy: policy,
                scene: SceneRegistry::empty(),
                session_store,
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

    pub(crate) fn set_module_contexts(
        &self,
        peer_contexts: Vec<(ModuleId, &'static str)>,
        allocation_hints: Vec<(ModuleId, &'static str)>,
    ) {
        self.inner
            .blackboard
            .set_module_contexts(peer_contexts, allocation_hints);
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
        self.inner
            .blackboard
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
    cognition_log_updates: CognitionLogUpdatedMailbox,
    clock: Rc<dyn Clock>,
    session_compaction: LlmTierHandle,
    session_compaction_policy: SessionCompactionPolicy,
    runtime_events: RuntimeEventEmitter,
    activation_gates: ActivationGateHub,
    session_store: Rc<dyn SessionStore>,
}

impl AgentRuntimeControl {
    pub async fn is_active(&self, owner: &ModuleInstanceId) -> bool {
        self.blackboard
            .read(|bb| bb.allocation().is_replica_active(owner))
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

    pub async fn tier_for(&self, owner: &ModuleInstanceId) -> ModelTier {
        self.blackboard
            .read(|bb| bb.allocation().tier_for(&owner.module))
            .await
    }

    /// Snapshot of the registered-module peer-context catalog. Cheap
    /// synchronous read; the scheduler turns this into an [`ActivateCx`] for
    /// each `activate` call.
    pub fn peer_contexts(&self) -> Vec<(ModuleId, &'static str)> {
        self.blackboard.peer_contexts().to_vec()
    }

    /// Snapshot of the registered-module allocation-hint catalog. Cheap
    /// synchronous read; the scheduler turns this into an [`ActivateCx`] for
    /// each `activate` call.
    pub fn allocation_hints(&self) -> Vec<(ModuleId, &'static str)> {
        self.blackboard.allocation_hints().to_vec()
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

    pub fn record_module_batch_throttled(&self, owner: ModuleInstanceId, delayed_for: Duration) {
        self.runtime_events
            .module_batch_throttled(owner, delayed_for);
    }

    pub fn record_module_batch_ready(&self, owner: ModuleInstanceId, batch: &ModuleBatch) {
        self.runtime_events.module_batch_ready(owner, batch);
    }

    pub fn with_session_checkpoint_runtime<'a>(
        &self,
        cx: crate::ActivateCx<'a>,
        owner: ModuleInstanceId,
    ) -> crate::ActivateCx<'a> {
        cx.with_session_checkpoint_runtime(
            self.session_store.clone(),
            self.runtime_events.clone(),
            owner,
        )
    }

    pub fn record_module_activation_completed(
        &self,
        owner: ModuleInstanceId,
        duration: Duration,
        succeeded: bool,
    ) {
        self.runtime_events
            .module_activation_completed(owner, duration, succeeded);
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

    pub fn record_module_restarted(
        &self,
        owner: ModuleInstanceId,
        consecutive_failures: u32,
        failure_limit: u32,
    ) {
        self.runtime_events
            .module_restarted(owner, consecutive_failures, failure_limit);
    }

    pub fn record_module_stopped(
        &self,
        owner: ModuleInstanceId,
        phase: impl Into<String>,
        message: impl Into<String>,
        consecutive_failures: u32,
    ) {
        self.runtime_events.module_stopped(
            owner,
            phase.into(),
            message.into(),
            consecutive_failures,
        );
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

    pub fn memo_updated_inbox(&self) -> MemoUpdatedInbox {
        TopicInbox::new_excluding_self(self.owner.clone(), self.root.inner.memo_updates.clone())
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
            TopicMailbox::new(
                self.owner.clone(),
                self.root.inner.memo_log_evictions.clone(),
            ),
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
            TopicMailbox::new(
                self.owner.clone(),
                self.root.inner.memo_log_evictions.clone(),
            ),
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
        )
    }

    pub fn session(&self, key: impl Into<String>) -> SessionCapabilityRequest {
        SessionCapabilityRequest {
            owner: self.owner.clone(),
            root: self.root.clone(),
            key: key.into(),
            auto_compaction: None,
        }
    }

    pub fn blackboard_reader(&self) -> BlackboardReader {
        self.root.blackboard_reader()
    }

    pub fn memory_metadata_reader(&self) -> MemoryMetadataReader {
        self.root.memory_metadata_reader()
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

    pub fn interoception_reader(&self) -> InteroceptiveReader {
        InteroceptiveReader::new(self.root.inner.blackboard.clone())
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
            CognitionLogEvictedMailbox::new(
                self.owner.clone(),
                self.root.inner.cognition_log_evictions.clone(),
            ),
            self.root.inner.clock.clone(),
        )
    }

    pub fn allocation_writer(
        &self,
        allowed_target_modules: Vec<ModuleId>,
        allowed_suppression_modules: Vec<ModuleId>,
    ) -> AllocationWriter {
        AllocationWriter::new(
            self.owner.clone(),
            self.root.inner.blackboard.clone(),
            allowed_target_modules,
            allowed_suppression_modules,
            self.root.inner.runtime_policy.allocation_effects.clone(),
        )
    }

    pub fn interoception_policy(&self) -> InteroceptionRuntimePolicy {
        self.root.inner.runtime_policy.interoception.clone()
    }

    pub fn interoception_writer(&self) -> InteroceptiveWriter {
        InteroceptiveWriter::new(
            self.owner.clone(),
            self.root.inner.blackboard.clone(),
            InteroceptiveUpdatedMailbox::new(
                self.owner.clone(),
                self.root.inner.interoception_updates.clone(),
            ),
            self.root.inner.clock.clone(),
        )
    }

    pub fn scene_reader(&self) -> SceneReader {
        SceneReader::new(self.root.inner.scene.clone())
    }

    pub fn clock(&self) -> Rc<dyn Clock> {
        self.root.clock()
    }

    pub fn time_division(&self) -> TimeDivision {
        self.root.time_division()
    }
}

pub struct SessionCapabilityRequest {
    owner: ModuleInstanceId,
    root: CapabilityProviders,
    key: String,
    auto_compaction: Option<SessionAutoCompaction>,
}

impl SessionCapabilityRequest {
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
            attach_persistent_session_metadata(
                &mut session,
                self.owner,
                key,
                self.auto_compaction,
                restored,
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
    module: Box<dyn ErasedModule>,
}

impl AllocatedModule {
    fn new(
        owner: ModuleInstanceId,
        caps: CapabilityProviders,
        builder: ErasedModuleBuilder,
        module: Box<dyn ErasedModule>,
    ) -> Self {
        Self {
            owner,
            caps,
            builder,
            module,
        }
    }

    pub fn owner(&self) -> &ModuleInstanceId {
        &self.owner
    }

    pub async fn restart(&mut self) -> Result<(), ModuleRegistryError> {
        let scoped = self.caps.scoped(self.owner.clone());
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
    peer_context: Option<&'static str>,
    allocation_hint: Option<&'static str>,
    policy: ModulePolicy,
    replica_capacity: u8,
    builder: ErasedModuleBuilder,
}

impl fmt::Debug for ModuleRegistration {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ModuleRegistration")
            .field("module", &self.module)
            .field("peer_context", &self.peer_context)
            .field("allocation_hint", &self.allocation_hint)
            .field("policy", &self.policy)
            .field("replica_capacity", &self.replica_capacity)
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
        }
    }

    /// Declare that `dependent` should wait for active `dependency` replicas to flush before
    /// activation. Both roles must be registered and the dependency graph must be acyclic.
    pub fn depends_on(mut self, dependent: ModuleId, dependency: ModuleId) -> Self {
        self.dependencies.push((dependent, dependency));
        self
    }

    /// Register a module type with its boot-time policy. The module's identity
    /// and prompt/allocation catalogs come from [`Module::id`],
    /// [`Module::peer_context`], and [`Module::allocation_hint`].
    pub fn register<B>(
        mut self,
        policy: ModulePolicy,
        builder: B,
    ) -> Result<Self, ModuleRegistryError>
    where
        B: ModuleRegisterer + 'static,
    {
        let module = ModuleId::new(<B::Module as Module>::id())?;
        let peer_context = <B::Module as Module>::peer_context();
        let allocation_hint = <B::Module as Module>::allocation_hint();
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
            peer_context,
            allocation_hint,
            policy,
            replica_capacity,
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
        let peer_context = <B::Module as Module>::peer_context();
        let allocation_hint = <B::Module as Module>::allocation_hint();
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
            peer_context,
            allocation_hint,
            policy,
            replica_capacity,
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
        // Install the post-boot module catalogs before any module is constructed
        // so module constructors can read peers from `caps.peer_contexts()`
        // synchronously when they assemble their system prompts.
        caps.set_module_contexts(
            self.registrations
                .iter()
                .filter_map(|registration| {
                    registration
                        .peer_context
                        .map(|context| (registration.module.clone(), context))
                })
                .collect(),
            self.registrations
                .iter()
                .filter_map(|registration| {
                    registration
                        .allocation_hint
                        .map(|hint| (registration.module.clone(), hint))
                })
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
                modules.push(AllocatedModule::new(
                    owner,
                    caps.clone(),
                    Rc::clone(&registration.builder),
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
        CognitionLogEntry, ResourceAllocation,
    };
    use nuillu_types::{ReplicaCapRange, builtin};

    use crate::ports::{CognitionLogRepository, PortError, SystemClock};
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
        records: Arc<std::sync::Mutex<Vec<(ModuleInstanceId, CognitionLogEntry)>>>,
    }

    impl RecordingCognitionLogRepository {
        fn records(&self) -> Vec<(ModuleInstanceId, CognitionLogEntry)> {
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
        cognition_log_port: Rc<dyn CognitionLogRepository>,
    ) -> CapabilityProviders {
        let adapter = Arc::new(lutum::MockLlmAdapter::new());
        let budget = lutum::SharedPoolBudgetManager::new(lutum::SharedPoolBudgetOptions::default());
        let lutum = lutum::Lutum::new(adapter, budget);
        CapabilityProviders::new(CapabilityProviderPorts {
            blackboard,
            cognition_log_port,
            clock: Rc::new(SystemClock),
            tiers: LutumTiers::from_shared_lutum(lutum),
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

    struct NoopModule;

    #[async_trait(?Send)]
    impl Module for NoopModule {
        type Batch = ();

        fn id() -> &'static str {
            "noop"
        }

        fn peer_context() -> Option<&'static str> {
            Some("test stub")
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

    async fn noop_builder(_: ModuleCapabilityFactory) -> Result<NoopModule, ModuleRegistryError> {
        Ok(NoopModule)
    }

    struct AllocationHintOnlyModule;

    #[async_trait(?Send)]
    impl Module for AllocationHintOnlyModule {
        type Batch = ();

        fn id() -> &'static str {
            "allocation-hint-only"
        }

        fn peer_context() -> Option<&'static str> {
            None
        }

        fn allocation_hint() -> Option<&'static str> {
            Some("test allocation hint")
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

    async fn allocation_hint_only_builder(
        _: ModuleCapabilityFactory,
    ) -> Result<AllocationHintOnlyModule, ModuleRegistryError> {
        Ok(AllocationHintOnlyModule)
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
    async fn build_installs_separate_peer_and_allocation_catalogs() {
        let blackboard = Blackboard::default();
        let caps = test_caps(blackboard.clone());
        ModuleRegistry::new()
            .register(test_policy(0..=0), noop_builder)
            .unwrap()
            .register(test_policy(0..=0), allocation_hint_only_builder)
            .unwrap()
            .build(&caps)
            .await
            .unwrap();

        let noop = nuillu_types::ModuleId::new(NoopModule::id()).unwrap();
        let allocation_only = nuillu_types::ModuleId::new(AllocationHintOnlyModule::id()).unwrap();

        assert_eq!(
            blackboard.peer_contexts().to_vec(),
            vec![(noop, "test stub")]
        );
        assert_eq!(
            blackboard.allocation_hints().to_vec(),
            vec![(allocation_only, "test allocation hint")]
        );
    }

    #[tokio::test]
    async fn capabilities_are_non_exclusive() {
        let caps = test_caps(Blackboard::default());
        let cognition_gate = scoped(&caps, builtin::cognition_gate(), 0);
        let controller = scoped(&caps, builtin::allocation_controller(), 0);
        let _w1 = cognition_gate.cognition_writer();
        let _w2 = cognition_gate.cognition_writer();
        let _a1 = controller.allocation_writer(vec![builtin::cognition_gate()], Vec::new());
        let _a2 = controller.allocation_writer(vec![builtin::cognition_gate()], Vec::new());
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
            crate::ActivateCx::new(&[], &[], &[], &[], compaction, Utc::now()),
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
                "Summarize.",
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
            crate::ActivateCx::new(&[], &[], &[], &[], compaction, Utc::now()),
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
            Utc::now(),
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
    async fn allocation_writer_records_guidance_changes() {
        let blackboard = Blackboard::default();
        blackboard
            .apply(BlackboardCommand::SetModulePolicies {
                policies: vec![
                    (
                        builtin::allocation_controller(),
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
        let caps = test_caps(blackboard.clone());
        let controller = scoped(&caps, builtin::allocation_controller(), 0);
        let writer = controller.allocation_writer(vec![builtin::cognition_gate()], Vec::new());

        let commands = vec![AllocationCommand::target(
            builtin::cognition_gate(),
            AllocationEffectLevel::Max,
            Some("promote current sensory memo into attention".into()),
        )];

        writer.submit(commands).await;

        let allocation = blackboard.read(|bb| bb.allocation().clone()).await;
        assert_eq!(
            allocation.for_module(&builtin::cognition_gate()).guidance,
            "promote current sensory memo into attention"
        );
    }

    #[tokio::test]
    async fn allocation_writer_applies_only_allowed_command_kinds() {
        let mut base = ResourceAllocation::default();
        base.set_activation(builtin::allocation_controller(), ActivationRatio::ONE);
        base.set_activation(builtin::cognition_gate(), ActivationRatio::ONE);
        base.set_activation(builtin::speak(), ActivationRatio::ZERO);
        let blackboard = Blackboard::with_allocation(base);
        blackboard
            .apply(BlackboardCommand::SetModulePolicies {
                policies: vec![
                    (
                        builtin::allocation_controller(),
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
        let controller = scoped(&caps, builtin::allocation_controller(), 0);
        let writer =
            controller.allocation_writer(vec![builtin::speak()], vec![builtin::cognition_gate()]);

        writer
            .submit([
                AllocationCommand::target(
                    builtin::speak(),
                    AllocationEffectLevel::Max,
                    Some("speak if attention is ready".into()),
                ),
                AllocationCommand::target(
                    builtin::cognition_gate(),
                    AllocationEffectLevel::Max,
                    Some("disallowed target".into()),
                ),
                AllocationCommand::suppression(
                    builtin::cognition_gate(),
                    AllocationEffectLevel::High,
                ),
                AllocationCommand::suppression(builtin::speak(), AllocationEffectLevel::Max),
            ])
            .await;

        let allocation = blackboard.read(|bb| bb.allocation().clone()).await;
        assert_eq!(
            allocation.activation_for(&builtin::speak()),
            ActivationRatio::ONE
        );
        assert_eq!(
            allocation.for_module(&builtin::speak()).guidance,
            "speak if attention is ready"
        );
        assert_eq!(
            allocation.activation_for(&builtin::cognition_gate()),
            ActivationRatio::from_f64(0.10)
        );
        assert_ne!(
            allocation.for_module(&builtin::cognition_gate()).guidance,
            "disallowed target"
        );
    }

    #[tokio::test]
    async fn cognition_log_updates_do_not_wake_controller_memo_inbox() {
        let caps = test_caps(Blackboard::default());
        let controller = scoped(&caps, builtin::allocation_controller(), 0);
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
        let controller = scoped(&caps, builtin::allocation_controller(), 0);
        let speak = scoped(&caps, builtin::speak(), 0);
        let mut memo_updates = controller.memo_updated_inbox();

        speak.memo().write("utterance completed").await;

        let event = memo_updates.next_item().await.unwrap();
        assert_eq!(event.sender.module, builtin::speak());
        assert_eq!(event.body.owner.module, builtin::speak());
    }
}
