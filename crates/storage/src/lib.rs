//! Backend-neutral storage ports and shared storage data types.

use std::rc::Rc;
use std::sync::Arc;

use async_trait::async_trait;
use nuillu_memory::MemoryStore;
use nuillu_module::ports::{CognitionLogRepository, PortError};
use nuillu_module::{AllocationStore, AmbientSensoryEntry, MemoLogRepository, SessionStore};
use nuillu_reward::PolicyStore;
use nuillu_types::ModuleInstanceId;
use sha2::{Digest, Sha256};

pub const MAX_EMBEDDING_DIMENSIONS: usize = 65_536;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EmbeddingProfile {
    pub name: String,
    pub version: String,
    pub dimensions: usize,
}

impl EmbeddingProfile {
    pub fn new(name: impl Into<String>, version: impl Into<String>, dimensions: usize) -> Self {
        Self {
            name: name.into(),
            version: version.into(),
            dimensions,
        }
    }

    pub fn default_for_dimensions(dimensions: usize) -> Self {
        Self::new("default", "v1", dimensions)
    }

    pub fn profile_id(&self) -> String {
        let mut hasher = Sha256::new();
        hasher.update(self.name.as_bytes());
        hasher.update([0]);
        hasher.update(self.version.as_bytes());
        hasher.update([0]);
        hasher.update(self.dimensions.to_string().as_bytes());
        let digest = hasher.finalize();
        let prefix = digest[..8]
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>();
        format!("p{prefix}")
    }

    pub fn validate(&self) -> Result<(), PortError> {
        if self.name.trim().is_empty() {
            return Err(PortError::InvalidInput(
                "embedding profile name must not be empty".into(),
            ));
        }
        if self.version.trim().is_empty() {
            return Err(PortError::InvalidInput(
                "embedding profile version must not be empty".into(),
            ));
        }
        if self.dimensions == 0 {
            return Err(PortError::InvalidInput(
                "embedding profile dimensions must be greater than zero".into(),
            ));
        }
        if self.dimensions > MAX_EMBEDDING_DIMENSIONS {
            return Err(PortError::InvalidInput(format!(
                "embedding profile dimensions must be <= {MAX_EMBEDDING_DIMENSIONS}, got {}",
                self.dimensions
            )));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct NewLlmTranscriptTurn {
    pub server_session_id: String,
    pub turn_id: String,
    pub owner: String,
    pub owner_module: String,
    pub owner_replica: u8,
    pub tier: String,
    pub source: String,
    pub session_key: Option<String>,
    pub operation: String,
    pub started_at_ms: i64,
    pub completed_at_ms: i64,
    pub trace_json: serde_json::Value,
}

#[derive(Clone, Debug, PartialEq)]
pub struct LlmTranscriptTurnRecord {
    pub id: i64,
    pub server_session_id: String,
    pub turn_id: String,
    pub owner: String,
    pub owner_module: String,
    pub owner_replica: u8,
    pub tier: String,
    pub source: String,
    pub session_key: Option<String>,
    pub operation: String,
    pub started_at_ms: i64,
    pub completed_at_ms: i64,
    pub trace_json: serde_json::Value,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NewOneShotSensoryInput {
    pub server_session_id: String,
    pub modality: String,
    pub direction: Option<String>,
    pub content: String,
    pub observed_at_ms: i64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OneShotSensoryInputRecord {
    pub id: i64,
    pub server_session_id: String,
    pub modality: String,
    pub direction: Option<String>,
    pub content: String,
    pub observed_at_ms: i64,
    pub created_at_ms: i64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NewAmbientSensorySnapshot {
    pub server_session_id: String,
    pub entries: Vec<AmbientSensoryEntry>,
    pub observed_at_ms: i64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AmbientSensorySnapshotRecord {
    pub id: i64,
    pub server_session_id: String,
    pub entries: Vec<AmbientSensoryEntry>,
    pub observed_at_ms: i64,
    pub created_at_ms: i64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UtteranceEventKind {
    Delta,
    Completed,
    Aborted,
}

impl UtteranceEventKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Delta => "delta",
            Self::Completed => "completed",
            Self::Aborted => "aborted",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NewUtteranceEvent {
    pub server_session_id: String,
    pub event_kind: UtteranceEventKind,
    pub sender: ModuleInstanceId,
    pub target: String,
    pub generation_id: u64,
    pub sequence: u32,
    pub content: String,
    pub reason: Option<String>,
    pub occurred_at_ms: i64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UtteranceEventRecord {
    pub id: i64,
    pub server_session_id: String,
    pub event_kind: UtteranceEventKind,
    pub sender: ModuleInstanceId,
    pub target: String,
    pub generation_id: u64,
    pub sequence: u32,
    pub content: String,
    pub reason: Option<String>,
    pub occurred_at_ms: i64,
    pub created_at_ms: i64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExternalActionEventStatus {
    Pending,
    Completed,
}

impl ExternalActionEventStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Pending => "pending",
            Self::Completed => "completed",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct NewExternalActionEvent {
    pub server_session_id: String,
    pub invocation_id: String,
    pub invoked_by: ModuleInstanceId,
    pub action_id: String,
    pub arguments: serde_json::Value,
    pub requested_at_ms: i64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExternalActionEventRecord {
    pub id: i64,
    pub server_session_id: String,
    pub invocation_id: String,
    pub invoked_by: ModuleInstanceId,
    pub action_id: String,
    pub arguments: serde_json::Value,
    pub status: ExternalActionEventStatus,
    pub accepted: Option<bool>,
    pub message: Option<String>,
    pub requested_at_ms: i64,
    pub completed_at_ms: Option<i64>,
    pub created_at_ms: i64,
    pub updated_at_ms: i64,
}

#[async_trait]
pub trait LlmTranscriptStore: Send + Sync {
    async fn insert_completed_turn(&self, turn: NewLlmTranscriptTurn) -> Result<(), PortError>;
    async fn completed_turns_page(
        &self,
        offset: usize,
        limit: usize,
    ) -> Result<Vec<LlmTranscriptTurnRecord>, PortError>;
    async fn prune_completed_turns(&self, keep: usize) -> Result<(), PortError>;

    async fn recent_completed_turns(
        &self,
        limit: usize,
    ) -> Result<Vec<LlmTranscriptTurnRecord>, PortError> {
        self.completed_turns_page(0, limit).await
    }
}

#[async_trait(?Send)]
pub trait OneShotSensoryInputStore {
    async fn append(
        &self,
        input: NewOneShotSensoryInput,
    ) -> Result<OneShotSensoryInputRecord, PortError>;
    async fn recent_page(
        &self,
        offset: usize,
        limit: usize,
    ) -> Result<Vec<OneShotSensoryInputRecord>, PortError>;

    async fn recent(&self, limit: usize) -> Result<Vec<OneShotSensoryInputRecord>, PortError> {
        self.recent_page(0, limit).await
    }
}

#[async_trait(?Send)]
pub trait AmbientSensorySnapshotStore {
    async fn append(
        &self,
        snapshot: NewAmbientSensorySnapshot,
    ) -> Result<AmbientSensorySnapshotRecord, PortError>;
    async fn recent_page(
        &self,
        offset: usize,
        limit: usize,
    ) -> Result<Vec<AmbientSensorySnapshotRecord>, PortError>;

    async fn recent(&self, limit: usize) -> Result<Vec<AmbientSensorySnapshotRecord>, PortError> {
        self.recent_page(0, limit).await
    }
}

#[async_trait(?Send)]
pub trait UtteranceEventStore {
    async fn append(&self, event: NewUtteranceEvent) -> Result<UtteranceEventRecord, PortError>;
    async fn recent_page(
        &self,
        offset: usize,
        limit: usize,
    ) -> Result<Vec<UtteranceEventRecord>, PortError>;

    async fn recent(&self, limit: usize) -> Result<Vec<UtteranceEventRecord>, PortError> {
        self.recent_page(0, limit).await
    }
}

#[async_trait(?Send)]
pub trait ExternalActionEventStore {
    async fn append_pending(
        &self,
        event: NewExternalActionEvent,
    ) -> Result<ExternalActionEventRecord, PortError>;
    async fn complete(
        &self,
        id: i64,
        accepted: bool,
        message: String,
        completed_at_ms: i64,
    ) -> Result<ExternalActionEventRecord, PortError>;
    async fn recent_page(
        &self,
        offset: usize,
        limit: usize,
    ) -> Result<Vec<ExternalActionEventRecord>, PortError>;

    async fn recent(&self, limit: usize) -> Result<Vec<ExternalActionEventRecord>, PortError> {
        self.recent_page(0, limit).await
    }
}

pub trait AgentStore {
    fn memory_store(&self) -> Rc<dyn MemoryStore>;
    fn policy_store(&self) -> Rc<dyn PolicyStore>;
    fn session_store(&self) -> Rc<dyn SessionStore>;
    fn allocation_store(&self) -> Rc<dyn AllocationStore>;
    fn cognition_log_repository(&self) -> Rc<dyn CognitionLogRepository>;
    fn memo_log_repository(&self) -> Rc<dyn MemoLogRepository>;
    fn llm_transcript_store(&self) -> Arc<dyn LlmTranscriptStore>;
    fn one_shot_sensory_input_store(&self) -> Rc<dyn OneShotSensoryInputStore>;
    fn ambient_sensory_snapshot_store(&self) -> Rc<dyn AmbientSensorySnapshotStore>;
    fn utterance_event_store(&self) -> Rc<dyn UtteranceEventStore>;
    fn external_action_event_store(&self) -> Rc<dyn ExternalActionEventStore>;
}
