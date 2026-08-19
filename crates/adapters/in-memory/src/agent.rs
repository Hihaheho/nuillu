use std::rc::Rc;
use std::sync::Arc;

use nuillu_memory::MemoryStore;
use nuillu_module::ports::{CognitionLogRepository, Embedder, PortError};
use nuillu_module::{AllocationStore, MemoLogRepository, SessionStore};
use nuillu_reward::PolicyStore;
use nuillu_storage::{
    AgentStore, AmbientSensorySnapshotStore, EmbeddingProfile, ExternalActionEventStore,
    LlmTranscriptStore, OneShotSensoryInputStore, UtteranceEventStore,
};

use crate::{
    InMemoryAllocationStore, InMemoryAmbientSensorySnapshotStore, InMemoryCognitionLogRepository,
    InMemoryExternalActionEventStore, InMemoryLlmTranscriptStore, InMemoryMemoLogRepository,
    InMemoryMemoryStore, InMemoryOneShotSensoryInputStore, InMemoryPolicyStore,
    InMemorySessionStore, InMemoryUtteranceEventStore,
};

pub struct InMemoryAgentStore {
    memory: Rc<InMemoryMemoryStore>,
    policy: Rc<InMemoryPolicyStore>,
    session: Rc<InMemorySessionStore>,
    allocation: Rc<InMemoryAllocationStore>,
    cognition_log: Rc<InMemoryCognitionLogRepository>,
    memo_log: Rc<InMemoryMemoLogRepository>,
    llm_transcript: Arc<InMemoryLlmTranscriptStore>,
    one_shot_sensory_input: Rc<InMemoryOneShotSensoryInputStore>,
    ambient_sensory_snapshot: Rc<InMemoryAmbientSensorySnapshotStore>,
    utterance_event: Rc<InMemoryUtteranceEventStore>,
    external_action_event: Rc<InMemoryExternalActionEventStore>,
}

impl InMemoryAgentStore {
    pub fn new(
        memory_profile: EmbeddingProfile,
        memory_embedder: Rc<dyn Embedder>,
        policy_profile: EmbeddingProfile,
        policy_embedder: Rc<dyn Embedder>,
    ) -> Result<Self, PortError> {
        Ok(Self {
            memory: Rc::new(InMemoryMemoryStore::new(memory_profile, memory_embedder)?),
            policy: Rc::new(InMemoryPolicyStore::new(policy_profile, policy_embedder)?),
            session: Rc::new(InMemorySessionStore::new()),
            allocation: Rc::new(InMemoryAllocationStore::new()),
            cognition_log: Rc::new(InMemoryCognitionLogRepository::new()),
            memo_log: Rc::new(InMemoryMemoLogRepository::new()),
            llm_transcript: Arc::new(InMemoryLlmTranscriptStore::new()),
            one_shot_sensory_input: Rc::new(InMemoryOneShotSensoryInputStore::new()),
            ambient_sensory_snapshot: Rc::new(InMemoryAmbientSensorySnapshotStore::new()),
            utterance_event: Rc::new(InMemoryUtteranceEventStore::new()),
            external_action_event: Rc::new(InMemoryExternalActionEventStore::new()),
        })
    }
}

impl AgentStore for InMemoryAgentStore {
    fn memory_store(&self) -> Rc<dyn MemoryStore> {
        self.memory.clone()
    }

    fn policy_store(&self) -> Rc<dyn PolicyStore> {
        self.policy.clone()
    }

    fn session_store(&self) -> Rc<dyn SessionStore> {
        self.session.clone()
    }

    fn allocation_store(&self) -> Rc<dyn AllocationStore> {
        self.allocation.clone()
    }

    fn cognition_log_repository(&self) -> Rc<dyn CognitionLogRepository> {
        self.cognition_log.clone()
    }

    fn memo_log_repository(&self) -> Rc<dyn MemoLogRepository> {
        self.memo_log.clone()
    }

    fn llm_transcript_store(&self) -> Arc<dyn LlmTranscriptStore> {
        self.llm_transcript.clone()
    }

    fn one_shot_sensory_input_store(&self) -> Rc<dyn OneShotSensoryInputStore> {
        self.one_shot_sensory_input.clone()
    }

    fn ambient_sensory_snapshot_store(&self) -> Rc<dyn AmbientSensorySnapshotStore> {
        self.ambient_sensory_snapshot.clone()
    }

    fn utterance_event_store(&self) -> Rc<dyn UtteranceEventStore> {
        self.utterance_event.clone()
    }

    fn external_action_event_store(&self) -> Rc<dyn ExternalActionEventStore> {
        self.external_action_event.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embedding::TestEmbedder;
    use nuillu_storage::NewOneShotSensoryInput;

    #[tokio::test(flavor = "current_thread")]
    async fn aggregate_returns_shared_trait_object_stores() {
        let profile = EmbeddingProfile::default_for_dimensions(3);
        let store = InMemoryAgentStore::new(
            profile.clone(),
            Rc::new(TestEmbedder),
            profile,
            Rc::new(TestEmbedder),
        )
        .unwrap();

        store
            .one_shot_sensory_input_store()
            .append(NewOneShotSensoryInput {
                server_session_id: "session".into(),
                modality: "vision".into(),
                direction: None,
                content: "hello".into(),
                observed_at_ms: 1,
            })
            .await
            .unwrap();

        assert_eq!(
            store
                .one_shot_sensory_input_store()
                .recent(1)
                .await
                .unwrap()[0]
                .content,
            "hello"
        );
    }
}
