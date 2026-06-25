use async_trait::async_trait;
use nuillu_blackboard::{MemoLogPayload, MemoLogRecord};

use crate::ports::PortError;

#[derive(Debug, Clone, PartialEq)]
pub struct PersistedMemoLogEntry {
    pub record: MemoLogRecord,
    pub payload: MemoLogPayload,
}

/// Append-only persistence for retained module memo logs.
#[async_trait(?Send)]
pub trait MemoLogRepository {
    async fn append(&self, entry: &PersistedMemoLogEntry) -> Result<(), PortError>;

    async fn recent_per_owner(
        &self,
        retained_per_owner: usize,
    ) -> Result<Vec<PersistedMemoLogEntry>, PortError>;
}

#[derive(Debug, Default)]
pub struct NoopMemoLogRepository;

#[async_trait(?Send)]
impl MemoLogRepository for NoopMemoLogRepository {
    async fn append(&self, _entry: &PersistedMemoLogEntry) -> Result<(), PortError> {
        Ok(())
    }

    async fn recent_per_owner(
        &self,
        _retained_per_owner: usize,
    ) -> Result<Vec<PersistedMemoLogEntry>, PortError> {
        Ok(Vec::new())
    }
}
