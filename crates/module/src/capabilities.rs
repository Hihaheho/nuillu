use std::fmt;
use std::ops::RangeInclusive;
use std::sync::Arc;
use std::time::Duration;

use nuillu_blackboard::{
    ActivationRatioFn, AgenticDeadlockMarker, Blackboard, BlackboardCommand, Bpm, ModulePolicy,
    ModuleRunStatus,
};
use nuillu_types::{
    ModuleId, ModuleInstanceId, ReplicaCapRange, ReplicaCapRangeError, ReplicaIndex,
};

use crate::channels::{Topic, TopicPolicy};
use crate::ports::{AttentionRepository, Clock, FileSearchProvider, MemoryStore, UtteranceSink};
use crate::rate_limit::{RateLimiter, RuntimePolicy, TopicKind};
use crate::runtime_events::{NoopRuntimeEventSink, RuntimeEventEmitter, RuntimeEventSink};
use crate::r#trait::ErasedModule;
use crate::utterance::UtteranceWriter;
use crate::{
    AllocationReader, AllocationUpdated, AllocationUpdatedInbox, AllocationUpdatedMailbox,
    AllocationWriter, AttentionReader, AttentionStreamUpdated, AttentionStreamUpdatedInbox,
    AttentionStreamUpdatedMailbox, AttentionWriter, BlackboardReader, FileSearcher, LlmAccess,
    LutumTiers, Memo, MemoUpdated, MemoUpdatedInbox, MemoryCompactor, MemoryContentReader,
    MemoryRequest, MemoryRequestInbox, MemoryRequestMailbox, MemoryWriter, Module, ModuleBatch,
    ModuleStatusReader, QueryInbox, QueryMailbox, QueryRequest, SelfModelInbox, SelfModelMailbox,
    SelfModelRequest, SensoryDetailRequest, SensoryDetailRequestInbox, SensoryDetailRequestMailbox,
    SensoryInput, SensoryInputInbox, SensoryInputMailbox, SpeakInbox, SpeakMailbox, SpeakRequest,
    TimeDivision, TopicInbox, TopicMailbox, VectorMemorySearcher,
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
    query_topic: Topic<QueryRequest>,
    self_model_topic: Topic<SelfModelRequest>,
    sensory_detail_topic: Topic<SensoryDetailRequest>,
    speak_topic: Topic<SpeakRequest>,
    memory_request_topic: Topic<MemoryRequest>,
    attention_updates: Topic<AttentionStreamUpdated>,
    allocation_updates: Topic<AllocationUpdated>,
    memo_updates: Topic<MemoUpdated>,
    sensory_input_topic: Topic<SensoryInput>,
    attention_port: Arc<dyn AttentionRepository>,
    primary_memory_store: Arc<dyn MemoryStore>,
    memory_replicas: Vec<Arc<dyn MemoryStore>>,
    file_search: Arc<dyn FileSearchProvider>,
    utterance_sink: Arc<dyn UtteranceSink>,
    clock: Arc<dyn Clock>,
    time_division: TimeDivision,
    tiers: LutumTiers,
    runtime_events: RuntimeEventEmitter,
    rate_limiter: RateLimiter,
    runtime_policy: RuntimePolicy,
}

impl CapabilityProviders {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        blackboard: Blackboard,
        attention_port: Arc<dyn AttentionRepository>,
        primary_memory_store: Arc<dyn MemoryStore>,
        memory_replicas: Vec<Arc<dyn MemoryStore>>,
        file_search: Arc<dyn FileSearchProvider>,
        utterance_sink: Arc<dyn UtteranceSink>,
        clock: Arc<dyn Clock>,
        tiers: LutumTiers,
    ) -> Self {
        Self::new_with_runtime_events(
            blackboard,
            attention_port,
            primary_memory_store,
            memory_replicas,
            file_search,
            utterance_sink,
            clock,
            tiers,
            Arc::new(NoopRuntimeEventSink),
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new_with_runtime_events(
        blackboard: Blackboard,
        attention_port: Arc<dyn AttentionRepository>,
        primary_memory_store: Arc<dyn MemoryStore>,
        memory_replicas: Vec<Arc<dyn MemoryStore>>,
        file_search: Arc<dyn FileSearchProvider>,
        utterance_sink: Arc<dyn UtteranceSink>,
        clock: Arc<dyn Clock>,
        tiers: LutumTiers,
        runtime_event_sink: Arc<dyn RuntimeEventSink>,
    ) -> Self {
        Self::new_with_runtime_policy(
            blackboard,
            attention_port,
            primary_memory_store,
            memory_replicas,
            file_search,
            utterance_sink,
            clock,
            tiers,
            runtime_event_sink,
            RuntimePolicy::default(),
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new_with_runtime_policy(
        blackboard: Blackboard,
        attention_port: Arc<dyn AttentionRepository>,
        primary_memory_store: Arc<dyn MemoryStore>,
        memory_replicas: Vec<Arc<dyn MemoryStore>>,
        file_search: Arc<dyn FileSearchProvider>,
        utterance_sink: Arc<dyn UtteranceSink>,
        clock: Arc<dyn Clock>,
        tiers: LutumTiers,
        runtime_event_sink: Arc<dyn RuntimeEventSink>,
        policy: RuntimePolicy,
    ) -> Self {
        let runtime_events = RuntimeEventEmitter::new(runtime_event_sink);
        let rate_limiter = RateLimiter::new(policy.rate_limits.clone());
        Self {
            inner: Arc::new(CapabilityProvidersInner {
                query_topic: Topic::new(
                    blackboard.clone(),
                    TopicPolicy::RoleLoadBalanced,
                    TopicKind::Query,
                    rate_limiter.clone(),
                    runtime_events.clone(),
                ),
                self_model_topic: Topic::new(
                    blackboard.clone(),
                    TopicPolicy::RoleLoadBalanced,
                    TopicKind::SelfModel,
                    rate_limiter.clone(),
                    runtime_events.clone(),
                ),
                sensory_detail_topic: Topic::new(
                    blackboard.clone(),
                    TopicPolicy::RoleLoadBalanced,
                    TopicKind::SensoryDetailRequest,
                    rate_limiter.clone(),
                    runtime_events.clone(),
                ),
                speak_topic: Topic::new(
                    blackboard.clone(),
                    TopicPolicy::RoleLoadBalanced,
                    TopicKind::Speak,
                    rate_limiter.clone(),
                    runtime_events.clone(),
                ),
                memory_request_topic: Topic::new(
                    blackboard.clone(),
                    TopicPolicy::RoleLoadBalanced,
                    TopicKind::MemoryRequest,
                    rate_limiter.clone(),
                    runtime_events.clone(),
                ),
                attention_updates: Topic::new(
                    blackboard.clone(),
                    TopicPolicy::Fanout,
                    TopicKind::AttentionStreamUpdated,
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
                blackboard,
                attention_port,
                primary_memory_store,
                memory_replicas,
                file_search,
                utterance_sink,
                clock,
                time_division: TimeDivision::default(),
                tiers,
                runtime_events,
                rate_limiter,
                runtime_policy: policy,
            }),
        }
    }

    pub(crate) fn scoped(&self, owner: ModuleInstanceId) -> ModuleCapabilityFactory {
        ModuleCapabilityFactory {
            owner,
            root: self.clone(),
        }
    }

    pub(crate) async fn set_module_policies(&self, policies: Vec<(ModuleId, ModulePolicy)>) {
        self.inner
            .blackboard
            .apply(BlackboardCommand::SetModulePolicies { policies })
            .await;
    }

    pub(crate) fn set_module_catalog(&self, catalog: Vec<(ModuleId, &'static str)>) {
        self.inner.blackboard.set_module_catalog(catalog);
    }

    /// Boot-time access to the post-boot module catalog (lives outside the
    /// async lock so it can be read synchronously). The scheduler uses this
    /// to construct the per-activate `ActivateCx`. Modules should not read
    /// this directly: agent-global state is delivered to modules through the
    /// [`ActivateCx`] passed to `activate`, not through capability handles.
    pub(crate) fn module_catalog(&self) -> &[(ModuleId, &'static str)] {
        self.inner.blackboard.module_catalog()
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
            attention_updates: AttentionStreamUpdatedMailbox::new(
                owner,
                self.inner.attention_updates.clone(),
            ),
            clock: self.inner.clock.clone(),
            runtime_events: self.inner.runtime_events.clone(),
        }
    }

    pub fn blackboard_reader(&self) -> BlackboardReader {
        BlackboardReader::new(self.inner.blackboard.clone())
    }

    pub fn attention_reader(&self) -> AttentionReader {
        AttentionReader::new(self.inner.blackboard.clone())
    }

    pub fn allocation_reader(&self) -> AllocationReader {
        AllocationReader::new(self.inner.blackboard.clone())
    }

    pub fn module_status_reader(&self) -> ModuleStatusReader {
        ModuleStatusReader::new(self.inner.blackboard.clone())
    }

    pub fn vector_memory_searcher(&self) -> VectorMemorySearcher {
        VectorMemorySearcher::new(
            self.inner.primary_memory_store.clone(),
            self.inner.blackboard.clone(),
            self.inner.clock.clone(),
        )
    }

    pub fn memory_content_reader(&self) -> MemoryContentReader {
        MemoryContentReader::new(self.inner.primary_memory_store.clone())
    }

    pub fn memory_writer(&self) -> MemoryWriter {
        MemoryWriter::new(
            self.inner.primary_memory_store.clone(),
            self.inner.memory_replicas.clone(),
            self.inner.blackboard.clone(),
            self.inner.clock.clone(),
        )
    }

    pub fn memory_compactor(&self) -> MemoryCompactor {
        MemoryCompactor::new(
            self.inner.primary_memory_store.clone(),
            self.inner.memory_replicas.clone(),
            self.inner.blackboard.clone(),
            self.inner.clock.clone(),
        )
    }

    pub fn file_searcher(&self) -> FileSearcher {
        FileSearcher::new(self.inner.file_search.clone())
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
    attention_updates: AttentionStreamUpdatedMailbox,
    clock: Arc<dyn Clock>,
    runtime_events: RuntimeEventEmitter,
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

    /// Snapshot of the registered-module catalog. Cheap synchronous read; the
    /// scheduler turns this into an [`ActivateCx`] for each `activate` call.
    pub fn module_catalog(&self) -> Vec<(ModuleId, &'static str)> {
        self.blackboard.module_catalog().to_vec()
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

    pub async fn record_module_batch_throttled(
        &self,
        owner: ModuleInstanceId,
        delayed_for: Duration,
    ) {
        self.runtime_events
            .module_batch_throttled(owner, delayed_for)
            .await;
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
            .attention_updates
            .publish(AttentionStreamUpdated::AgenticDeadlockMarker)
            .await
            .is_err()
        {
            tracing::trace!("agentic deadlock attention update had no active subscribers");
        }
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
    pub fn query_mailbox(&self) -> QueryMailbox {
        TopicMailbox::new(self.owner.clone(), self.root.inner.query_topic.clone())
    }

    pub fn self_model_mailbox(&self) -> SelfModelMailbox {
        TopicMailbox::new(self.owner.clone(), self.root.inner.self_model_topic.clone())
    }

    pub fn sensory_detail_mailbox(&self) -> SensoryDetailRequestMailbox {
        TopicMailbox::new(
            self.owner.clone(),
            self.root.inner.sensory_detail_topic.clone(),
        )
    }

    pub fn attention_stream_updated_mailbox(&self) -> AttentionStreamUpdatedMailbox {
        TopicMailbox::new(
            self.owner.clone(),
            self.root.inner.attention_updates.clone(),
        )
    }
}

#[derive(Clone)]
pub struct ModuleCapabilityFactory {
    owner: ModuleInstanceId,
    root: CapabilityProviders,
}

impl ModuleCapabilityFactory {
    /// The owner this factory dispenses capabilities for. Capability handles
    /// returned by this factory are stamped with this id.
    pub fn owner(&self) -> &ModuleInstanceId {
        &self.owner
    }

    pub fn query_mailbox(&self) -> QueryMailbox {
        TopicMailbox::new(self.owner.clone(), self.root.inner.query_topic.clone())
    }

    pub fn query_inbox(&self) -> QueryInbox {
        TopicInbox::new(self.owner.clone(), self.root.inner.query_topic.clone())
    }

    pub fn self_model_mailbox(&self) -> SelfModelMailbox {
        TopicMailbox::new(self.owner.clone(), self.root.inner.self_model_topic.clone())
    }

    pub fn self_model_inbox(&self) -> SelfModelInbox {
        TopicInbox::new(self.owner.clone(), self.root.inner.self_model_topic.clone())
    }

    pub fn sensory_detail_mailbox(&self) -> SensoryDetailRequestMailbox {
        TopicMailbox::new(
            self.owner.clone(),
            self.root.inner.sensory_detail_topic.clone(),
        )
    }

    pub fn sensory_detail_inbox(&self) -> SensoryDetailRequestInbox {
        TopicInbox::new(
            self.owner.clone(),
            self.root.inner.sensory_detail_topic.clone(),
        )
    }

    pub fn speak_mailbox(&self) -> SpeakMailbox {
        TopicMailbox::new(self.owner.clone(), self.root.inner.speak_topic.clone())
    }

    pub fn speak_inbox(&self) -> SpeakInbox {
        TopicInbox::new(self.owner.clone(), self.root.inner.speak_topic.clone())
    }

    pub fn memory_request_mailbox(&self) -> MemoryRequestMailbox {
        TopicMailbox::new(
            self.owner.clone(),
            self.root.inner.memory_request_topic.clone(),
        )
    }

    pub fn memory_request_inbox(&self) -> MemoryRequestInbox {
        TopicInbox::new(
            self.owner.clone(),
            self.root.inner.memory_request_topic.clone(),
        )
    }

    pub fn attention_stream_updated_inbox(&self) -> AttentionStreamUpdatedInbox {
        TopicInbox::new(
            self.owner.clone(),
            self.root.inner.attention_updates.clone(),
        )
    }

    pub fn allocation_updated_inbox(&self) -> AllocationUpdatedInbox {
        TopicInbox::new(
            self.owner.clone(),
            self.root.inner.allocation_updates.clone(),
        )
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

    pub fn memo(&self) -> Memo {
        Memo::new(
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
        )
    }

    pub fn blackboard_reader(&self) -> BlackboardReader {
        self.root.blackboard_reader()
    }

    pub fn attention_reader(&self) -> AttentionReader {
        self.root.attention_reader()
    }

    pub fn allocation_reader(&self) -> AllocationReader {
        self.root.allocation_reader()
    }

    pub fn module_status_reader(&self) -> ModuleStatusReader {
        self.root.module_status_reader()
    }

    pub fn vector_memory_searcher(&self) -> VectorMemorySearcher {
        self.root.vector_memory_searcher()
    }

    pub fn memory_content_reader(&self) -> MemoryContentReader {
        self.root.memory_content_reader()
    }

    pub fn memory_writer(&self) -> MemoryWriter {
        self.root.memory_writer()
    }

    pub fn memory_compactor(&self) -> MemoryCompactor {
        self.root.memory_compactor()
    }

    pub fn file_searcher(&self) -> FileSearcher {
        self.root.file_searcher()
    }

    pub fn attention_writer(&self) -> AttentionWriter {
        AttentionWriter::new(
            self.owner.clone(),
            self.root.inner.blackboard.clone(),
            self.root.inner.attention_port.clone(),
            AttentionStreamUpdatedMailbox::new(
                self.owner.clone(),
                self.root.inner.attention_updates.clone(),
            ),
            self.root.inner.clock.clone(),
        )
    }

    pub fn allocation_writer(&self) -> AllocationWriter {
        AllocationWriter::new(
            self.owner.clone(),
            self.root.inner.blackboard.clone(),
            AllocationUpdatedMailbox::new(
                self.owner.clone(),
                self.root.inner.allocation_updates.clone(),
            ),
        )
    }

    pub fn utterance_writer(&self) -> UtteranceWriter {
        UtteranceWriter::new(
            self.owner.clone(),
            self.root.inner.blackboard.clone(),
            self.root.inner.utterance_sink.clone(),
            self.root.inner.clock.clone(),
        )
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
}

impl AllocatedModules {
    fn new(runtime: AgentRuntimeControl, modules: Vec<AllocatedModule>) -> Self {
        Self { runtime, modules }
    }

    pub fn len(&self) -> usize {
        self.modules.len()
    }

    pub fn is_empty(&self) -> bool {
        self.modules.is_empty()
    }

    pub fn into_parts(self) -> (AgentRuntimeControl, Vec<AllocatedModule>) {
        (self.runtime, self.modules)
    }
}

pub struct ModuleRegistry {
    registrations: Vec<ModuleRegistration>,
}

impl fmt::Debug for ModuleRegistry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ModuleRegistry")
            .field("registrations", &self.registrations)
            .finish()
    }
}

struct ModuleRegistration {
    module: ModuleId,
    role_description: &'static str,
    policy: ModulePolicy,
    builder: Box<dyn Fn(ModuleCapabilityFactory) -> Box<dyn ErasedModule>>,
}

impl fmt::Debug for ModuleRegistration {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ModuleRegistration")
            .field("module", &self.module)
            .field("role_description", &self.role_description)
            .field("policy", &self.policy)
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
        }
    }

    /// Register a module type with its base-active replica range, BPM tempo
    /// range, and per-module activation-ratio mapping. The module's identity
    /// comes from [`Module::id`] / [`Module::role_description`].
    pub fn register<B>(
        mut self,
        replicas_range: RangeInclusive<u8>,
        rate_limit_range: RangeInclusive<Bpm>,
        activation_ratio_fn: ActivationRatioFn,
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
        // The supplied range counts *additional* replicas above the always-on
        // base of 1; total active replicas = additional + 1.
        let range = ReplicaCapRange::new(*replicas_range.start(), *replicas_range.end())?;
        let policy = ModulePolicy::new(range, rate_limit_range, activation_ratio_fn);
        self.registrations.push(ModuleRegistration {
            module,
            role_description,
            policy,
            builder: Box::new(move |caps| Box::new(builder(caps))),
        });
        Ok(self)
    }

    pub async fn build(
        &self,
        caps: &CapabilityProviders,
    ) -> Result<AllocatedModules, ModuleRegistryError> {
        caps.apply_runtime_policy().await;
        caps.set_module_policies(
            self.registrations
                .iter()
                .map(|registration| (registration.module.clone(), registration.policy.clone()))
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
            // Build every possible replica up to (additional max + base 1);
            // allocation and the agent event loop decide which are active.
            let total_replicas = registration.policy.max_active_replicas();
            for replica in 0..total_replicas {
                let owner =
                    ModuleInstanceId::new(registration.module.clone(), ReplicaIndex::new(replica));
                let scoped = caps.scoped(owner.clone());
                modules.push(AllocatedModule::new(owner, (registration.builder)(scoped)));
            }
        }
        Ok(AllocatedModules::new(caps.runtime_control(), modules))
    }
}

impl Default for ModuleRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ModuleRegistryError {
    #[error(transparent)]
    ReplicaCapRange(#[from] ReplicaCapRangeError),
    #[error(transparent)]
    ModuleId(#[from] nuillu_types::ModuleIdParseError),
    #[error("module {module} is already registered")]
    DuplicateModule { module: ModuleId },
}

#[cfg(test)]
mod tests {
    use super::*;

    use async_trait::async_trait;
    use nuillu_blackboard::{
        ActivationRatio, Blackboard, BlackboardCommand, ModuleConfig, ResourceAllocation,
        UtteranceProgress,
    };
    use nuillu_types::{ModelTier, ReplicaCapRange, builtin};

    use crate::test_support::{scoped, test_caps};

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
            .register(
                0..=0,
                nuillu_blackboard::Bpm::from_f64(60.0)..=nuillu_blackboard::Bpm::from_f64(60.0),
                nuillu_blackboard::linear_ratio_fn,
                noop_builder,
            )
            .unwrap();

        let err = registry
            .register(
                0..=0,
                nuillu_blackboard::Bpm::from_f64(60.0)..=nuillu_blackboard::Bpm::from_f64(60.0),
                nuillu_blackboard::linear_ratio_fn,
                noop_builder,
            )
            .unwrap_err();

        let expected = nuillu_types::ModuleId::new(NoopModule::id()).unwrap();
        assert!(matches!(
            err,
            ModuleRegistryError::DuplicateModule { module } if module == expected
        ));
    }

    #[tokio::test]
    async fn capabilities_are_non_exclusive() {
        let caps = test_caps(Blackboard::default());
        let attention_gate = scoped(&caps, builtin::attention_gate(), 0);
        let controller = scoped(&caps, builtin::attention_controller(), 0);
        let _w1 = attention_gate.attention_writer();
        let _w2 = attention_gate.attention_writer();
        let _a1 = controller.allocation_writer();
        let _a2 = controller.allocation_writer();
    }

    #[tokio::test]
    async fn memo_updated_inbox_filters_self_writes() {
        let caps = test_caps(Blackboard::default());
        let attention_gate = scoped(&caps, builtin::attention_gate(), 0);
        let sensory = scoped(&caps, builtin::sensory(), 0);
        let mut inbox = attention_gate.memo_updated_inbox();

        attention_gate.memo().write("own memo").await;
        sensory.memo().write("sensory memo").await;

        let event = inbox.next_item().await.unwrap();
        assert_eq!(event.sender.module, builtin::sensory());
        assert_eq!(event.body.owner.module, builtin::sensory());
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
                            ReplicaCapRange::new(0, 0).unwrap(),
                            nuillu_blackboard::Bpm::from_f64(60.0)
                                ..=nuillu_blackboard::Bpm::from_f64(60.0),
                            nuillu_blackboard::linear_ratio_fn,
                        ),
                    ),
                    (
                        builtin::attention_gate(),
                        nuillu_blackboard::ModulePolicy::new(
                            ReplicaCapRange::new(0, 0).unwrap(),
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
        let attention_gate = scoped(&caps, builtin::attention_gate(), 0);
        let writer = controller.allocation_writer();
        let mut inbox = attention_gate.allocation_updated_inbox();

        let mut proposal = ResourceAllocation::default();
        proposal.set(
            builtin::attention_gate(),
            ModuleConfig {
                guidance: "promote current sensory memo into attention".into(),
                tier: ModelTier::Default,
            },
        );
        proposal.set_activation(builtin::attention_gate(), ActivationRatio::ONE);

        writer.set(proposal.clone()).await;
        let event = inbox.next_item().await.unwrap();
        assert_eq!(event.sender.module, builtin::attention_controller());
        assert_eq!(event.body, crate::AllocationUpdated);

        writer.set(proposal).await;
        assert!(inbox.take_ready_items().unwrap().items.is_empty());
    }

    #[tokio::test]
    async fn attention_stream_updates_do_not_wake_controller_memo_inbox() {
        let caps = test_caps(Blackboard::default());
        let controller = scoped(&caps, builtin::attention_controller(), 0);
        let attention_gate = scoped(&caps, builtin::attention_gate(), 0);
        let mut memo_updates = controller.memo_updated_inbox();

        attention_gate
            .attention_writer()
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

    #[tokio::test]
    async fn utterance_writer_owner_stamps_stream_progress() {
        let blackboard = Blackboard::default();
        let caps = test_caps(blackboard.clone());
        let speak = scoped(&caps, builtin::speak(), 0);

        speak
            .utterance_writer()
            .record_progress(UtteranceProgress::streaming(
                7,
                2,
                "Koro, wait",
                "answer Koro",
                "peer needs response",
            ))
            .await;

        let records = blackboard.read(|bb| bb.utterance_progress_records()).await;
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].owner.module, builtin::speak());
        assert_eq!(
            records[0].progress,
            UtteranceProgress::streaming(7, 2, "Koro, wait", "answer Koro", "peer needs response",)
        );
    }
}
