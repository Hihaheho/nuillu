use std::fmt;
use std::ops::RangeInclusive;
use std::sync::Arc;

use nuillu_blackboard::{Blackboard, BlackboardCommand};
use nuillu_types::{
    ModuleId, ModuleInstanceId, ReplicaCapRange, ReplicaCapRangeError, ReplicaIndex,
};

use crate::activation::ActivationGate;
use crate::channels::{Topic, TopicPolicy};
use crate::periodic::PeriodicRegistry;
use crate::ports::{AttentionRepository, Clock, FileSearchProvider, MemoryStore, UtteranceSink};
use crate::utterance::UtteranceWriter;
use crate::{
    AllocationReader, AllocationWriter, AttentionReader, AttentionStreamUpdated,
    AttentionStreamUpdatedInbox, AttentionStreamUpdatedMailbox, AttentionWriter, BlackboardReader,
    FileSearcher, LlmAccess, LutumTiers, Memo, MemoryCompactor, MemoryContentReader, MemoryRequest,
    MemoryRequestInbox, MemoryRequestMailbox, MemoryWriter, Module, PeriodicActivation,
    PeriodicInbox, QueryInbox, QueryMailbox, QueryRequest, SelfModelInbox, SelfModelMailbox,
    SelfModelRequest, SensoryInput, SensoryInputInbox, SensoryInputMailbox, TimeDivision,
    TopicInbox, TopicMailbox, VectorMemorySearcher,
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
    periodic_registry: Arc<PeriodicRegistry>,
    query_topic: Topic<QueryRequest>,
    self_model_topic: Topic<SelfModelRequest>,
    memory_request_topic: Topic<MemoryRequest>,
    attention_updates: Topic<AttentionStreamUpdated>,
    sensory_input_topic: Topic<SensoryInput>,
    attention_port: Arc<dyn AttentionRepository>,
    primary_memory_store: Arc<dyn MemoryStore>,
    memory_replicas: Vec<Arc<dyn MemoryStore>>,
    file_search: Arc<dyn FileSearchProvider>,
    utterance_sink: Arc<dyn UtteranceSink>,
    clock: Arc<dyn Clock>,
    time_division: TimeDivision,
    tiers: LutumTiers,
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
        Self {
            inner: Arc::new(CapabilityProvidersInner {
                periodic_registry: Arc::new(PeriodicRegistry::new()),
                query_topic: Topic::new(blackboard.clone(), TopicPolicy::RoleLoadBalanced),
                self_model_topic: Topic::new(blackboard.clone(), TopicPolicy::RoleLoadBalanced),
                memory_request_topic: Topic::new(blackboard.clone(), TopicPolicy::RoleLoadBalanced),
                attention_updates: Topic::new(blackboard.clone(), TopicPolicy::Fanout),
                sensory_input_topic: Topic::new(blackboard.clone(), TopicPolicy::RoleLoadBalanced),
                blackboard,
                attention_port,
                primary_memory_store,
                memory_replicas,
                file_search,
                utterance_sink,
                clock,
                time_division: TimeDivision::default(),
                tiers,
            }),
        }
    }

    pub fn periodic_activation(&self) -> PeriodicActivation {
        PeriodicActivation::new(
            self.inner.blackboard.clone(),
            self.inner.periodic_registry.clone(),
        )
    }

    fn scoped(&self, owner: ModuleInstanceId) -> ModuleCapabilityFactory {
        ModuleCapabilityFactory {
            owner,
            root: self.clone(),
        }
    }

    async fn set_replica_caps(&self, caps: Vec<(ModuleId, ReplicaCapRange)>) {
        self.inner
            .blackboard
            .apply(BlackboardCommand::SetReplicaCaps { caps })
            .await;
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
}

#[derive(Clone)]
pub struct ModuleCapabilityFactory {
    owner: ModuleInstanceId,
    root: CapabilityProviders,
}

impl ModuleCapabilityFactory {
    pub fn activation_gate(&self) -> ActivationGate {
        ActivationGate::new(self.owner.clone(), self.root.inner.blackboard.clone())
    }

    pub fn periodic_inbox(&self) -> PeriodicInbox {
        self.root
            .inner
            .periodic_registry
            .register(self.owner.clone(), 64)
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
        Memo::new(self.owner.clone(), self.root.inner.blackboard.clone())
    }

    pub fn llm_access(&self) -> LlmAccess {
        LlmAccess::new(
            self.owner.clone(),
            self.root.inner.tiers.clone(),
            self.root.inner.blackboard.clone(),
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
        AllocationWriter::new(self.owner.clone(), self.root.inner.blackboard.clone())
    }

    pub fn utterance_writer(&self) -> UtteranceWriter {
        UtteranceWriter::new(
            self.owner.clone(),
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

pub struct AllocatedModules {
    modules: Vec<Box<dyn Module>>,
}

impl AllocatedModules {
    fn new(modules: Vec<Box<dyn Module>>) -> Self {
        Self { modules }
    }

    pub fn len(&self) -> usize {
        self.modules.len()
    }

    pub fn is_empty(&self) -> bool {
        self.modules.is_empty()
    }

    pub fn into_modules(self) -> impl Iterator<Item = Box<dyn Module>> {
        self.modules.into_iter()
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
    cap_range: ReplicaCapRange,
    builder: Box<dyn Fn(ModuleCapabilityFactory) -> Box<dyn Module>>,
}

impl fmt::Debug for ModuleRegistration {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ModuleRegistration")
            .field("module", &self.module)
            .field("cap_range", &self.cap_range)
            .finish_non_exhaustive()
    }
}

/// Builds one module replica from its replica-scoped capability factory.
///
/// Closures that return a concrete `Module` implement this automatically, so
/// registration call sites do not need to allocate `Box<dyn Module>`.
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

    pub fn register(
        mut self,
        module: ModuleId,
        cap_range: RangeInclusive<u8>,
        builder: impl ModuleRegisterer + 'static,
    ) -> Result<Self, ModuleRegistryError> {
        if self
            .registrations
            .iter()
            .any(|registration| registration.module == module)
        {
            return Err(ModuleRegistryError::DuplicateModule { module });
        }
        let range = ReplicaCapRange::new(*cap_range.start(), *cap_range.end())?;
        self.registrations.push(ModuleRegistration {
            module,
            cap_range: range,
            builder: Box::new(move |caps| Box::new(builder(caps))),
        });
        Ok(self)
    }

    pub async fn build(
        &self,
        caps: &CapabilityProviders,
    ) -> Result<AllocatedModules, ModuleRegistryError> {
        caps.set_replica_caps(
            self.registrations
                .iter()
                .map(|registration| (registration.module.clone(), registration.cap_range))
                .collect(),
        )
        .await;

        let mut modules = Vec::new();
        for registration in &self.registrations {
            // Build every possible replica up to the registered max; allocation
            // and `ActivationGate` decide which replicas are active at runtime.
            for replica in 0..registration.cap_range.max {
                let scoped = caps.scoped(ModuleInstanceId::new(
                    registration.module.clone(),
                    ReplicaIndex::new(replica),
                ));
                modules.push((registration.builder)(scoped));
            }
        }
        Ok(AllocatedModules::new(modules))
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
    #[error("module {module} is already registered")]
    DuplicateModule { module: ModuleId },
}

#[cfg(test)]
mod tests {
    use super::*;

    use async_trait::async_trait;
    use nuillu_types::builtin;

    struct NoopModule;

    #[async_trait(?Send)]
    impl Module for NoopModule {
        async fn run(&mut self) {}
    }

    fn noop_builder(_: ModuleCapabilityFactory) -> NoopModule {
        NoopModule
    }

    #[test]
    fn register_rejects_duplicate_module_ids() {
        let registry = ModuleRegistry::new()
            .register(builtin::summarize(), 0..=1, noop_builder)
            .unwrap();

        let err = registry
            .register(builtin::summarize(), 0..=1, noop_builder)
            .unwrap_err();

        assert!(matches!(
            err,
            ModuleRegistryError::DuplicateModule { module } if module == builtin::summarize()
        ));
    }
}
