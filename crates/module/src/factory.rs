use std::sync::Arc;

use nuillu_blackboard::Blackboard;
use nuillu_types::ModuleId;

use crate::channels::Topic;
use crate::periodic::PeriodicRegistry;
use crate::ports::{AttentionRepository, Clock, FileSearchProvider, MemoryStore, UtteranceSink};
use crate::utterance::UtteranceWriter;
use crate::{
    AllocationReader, AllocationWriter, AttentionReader, AttentionStreamUpdated,
    AttentionStreamUpdatedInbox, AttentionStreamUpdatedMailbox, AttentionWriter, BlackboardReader,
    FileSearcher, LlmAccess, LutumTiers, Memo, MemoryCompactor, MemoryContentReader, MemoryRequest,
    MemoryRequestInbox, MemoryRequestMailbox, MemoryWriter, PeriodicActivation, PeriodicInbox,
    QueryInbox, QueryMailbox, QueryRequest, SelfModelInbox, SelfModelMailbox, SelfModelRequest,
    SensoryInput, SensoryInputInbox, SensoryInputMailbox, TopicInbox, TopicMailbox,
    VectorMemorySearcher,
};

/// Builds and dispenses [capabilities](crate) at agent boot.
///
/// Boot flow:
/// 1. `CapabilityFactory::new(...)` — bind the shared blackboard, ports,
///    and `LutumTiers`.
/// 2. Issue the capabilities each module should hold. Requesting
///    [`periodic_inbox_for`](Self::periodic_inbox_for) also registers
///    the module for elapsed-tick periodic activation.
///
/// Owner-stamped capabilities (`Memo`, typed mailboxes, `PeriodicInbox`,
/// `LlmAccess`, `AttentionWriter`) bake in the `ModuleId` at construction.
///
/// All capabilities are non-exclusive: every issuer call returns a fresh
/// handle and the factory does not enforce uniqueness. Single-writer
/// roles (summarize → attention stream, attention-controller → allocation)
/// are upheld by boot-time wiring, not by factory enforcement.
pub struct CapabilityFactory {
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
    tiers: LutumTiers,
}

impl CapabilityFactory {
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
            blackboard,
            periodic_registry: Arc::new(PeriodicRegistry::new()),
            query_topic: Topic::new(),
            self_model_topic: Topic::new(),
            memory_request_topic: Topic::new(),
            attention_updates: Topic::new(),
            sensory_input_topic: Topic::new(),
            attention_port,
            primary_memory_store,
            memory_replicas,
            file_search,
            utterance_sink,
            clock,
            tiers,
        }
    }

    pub fn periodic_activation(&self) -> PeriodicActivation {
        PeriodicActivation::new(self.blackboard.clone(), self.periodic_registry.clone())
    }

    pub fn periodic_inbox_for(&self, owner: ModuleId) -> PeriodicInbox {
        self.periodic_registry.register(owner, 64)
    }

    pub fn query_mailbox(&self, owner: ModuleId) -> QueryMailbox {
        TopicMailbox::new(owner, self.query_topic.clone())
    }

    pub fn query_inbox_for(&self, owner: ModuleId) -> QueryInbox {
        TopicInbox::new(owner, self.query_topic.clone())
    }

    pub fn self_model_mailbox(&self, owner: ModuleId) -> SelfModelMailbox {
        TopicMailbox::new(owner, self.self_model_topic.clone())
    }

    pub fn self_model_inbox_for(&self, owner: ModuleId) -> SelfModelInbox {
        TopicInbox::new(owner, self.self_model_topic.clone())
    }

    pub fn memory_request_mailbox(&self, owner: ModuleId) -> MemoryRequestMailbox {
        TopicMailbox::new(owner, self.memory_request_topic.clone())
    }

    pub fn memory_request_inbox_for(&self, owner: ModuleId) -> MemoryRequestInbox {
        TopicInbox::new(owner, self.memory_request_topic.clone())
    }

    pub fn attention_stream_updated_inbox_for(
        &self,
        owner: ModuleId,
    ) -> AttentionStreamUpdatedInbox {
        TopicInbox::new(owner, self.attention_updates.clone())
    }

    pub fn sensory_input_mailbox(&self, owner: ModuleId) -> SensoryInputMailbox {
        TopicMailbox::new(owner, self.sensory_input_topic.clone())
    }

    pub fn sensory_input_inbox_for(&self, owner: ModuleId) -> SensoryInputInbox {
        TopicInbox::new(owner, self.sensory_input_topic.clone())
    }

    pub fn memo(&self, owner: ModuleId) -> Memo {
        Memo::new(owner, self.blackboard.clone())
    }

    pub fn llm_access(&self, owner: ModuleId) -> LlmAccess {
        LlmAccess::new(owner, self.tiers.clone(), self.blackboard.clone())
    }

    pub fn blackboard_reader(&self) -> BlackboardReader {
        BlackboardReader::new(self.blackboard.clone())
    }

    pub fn attention_reader(&self) -> AttentionReader {
        AttentionReader::new(self.blackboard.clone())
    }

    pub fn allocation_reader(&self) -> AllocationReader {
        AllocationReader::new(self.blackboard.clone())
    }

    pub fn vector_memory_searcher(&self) -> VectorMemorySearcher {
        VectorMemorySearcher::new(self.primary_memory_store.clone(), self.blackboard.clone())
    }

    pub fn memory_content_reader(&self) -> MemoryContentReader {
        MemoryContentReader::new(self.primary_memory_store.clone())
    }

    pub fn memory_writer(&self) -> MemoryWriter {
        MemoryWriter::new(
            self.primary_memory_store.clone(),
            self.memory_replicas.clone(),
            self.blackboard.clone(),
        )
    }

    pub fn memory_compactor(&self) -> MemoryCompactor {
        MemoryCompactor::new(
            self.primary_memory_store.clone(),
            self.memory_replicas.clone(),
            self.blackboard.clone(),
        )
    }

    pub fn file_searcher(&self) -> FileSearcher {
        FileSearcher::new(self.file_search.clone())
    }

    pub fn attention_writer(&self, owner: ModuleId) -> AttentionWriter {
        AttentionWriter::new(
            owner.clone(),
            self.blackboard.clone(),
            self.attention_port.clone(),
            AttentionStreamUpdatedMailbox::new(owner, self.attention_updates.clone()),
            self.clock.clone(),
        )
    }

    pub fn allocation_writer(&self) -> AllocationWriter {
        AllocationWriter::new(self.blackboard.clone())
    }

    pub fn utterance_writer(&self, owner: ModuleId) -> UtteranceWriter {
        UtteranceWriter::new(owner, self.utterance_sink.clone(), self.clock.clone())
    }

    pub fn clock(&self) -> Arc<dyn Clock> {
        self.clock.clone()
    }
}
