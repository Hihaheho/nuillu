//! In-memory adapters for local development and tests.

use std::sync::Mutex;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use nuillu_blackboard::CognitionLogEntry;
use nuillu_module::ports::{CognitionLogRepository, PortError};
use nuillu_types::ModuleInstanceId;

#[derive(Debug, Default)]
pub struct InMemoryCognitionLogRepository {
    events: Mutex<Vec<(ModuleInstanceId, CognitionLogEntry)>>,
}

impl InMemoryCognitionLogRepository {
    pub fn new() -> Self {
        Self::default()
    }
}

#[async_trait(?Send)]
impl CognitionLogRepository for InMemoryCognitionLogRepository {
    async fn append(
        &self,
        source: ModuleInstanceId,
        entry: CognitionLogEntry,
    ) -> Result<(), PortError> {
        self.events
            .lock()
            .map_err(|_| PortError::Backend("cognition log repository lock poisoned".into()))?
            .push((source, entry));
        Ok(())
    }

    async fn since(
        &self,
        source: &ModuleInstanceId,
        from: DateTime<Utc>,
    ) -> Result<Vec<CognitionLogEntry>, PortError> {
        let events = self
            .events
            .lock()
            .map_err(|_| PortError::Backend("cognition log repository lock poisoned".into()))?;
        Ok(events
            .iter()
            .filter(|(owner, entry)| owner == source && entry.at >= from)
            .map(|(_, entry)| entry.clone())
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nuillu_types::{ReplicaIndex, builtin};

    #[tokio::test]
    async fn cognition_log_repo_filters_by_time() {
        let repo = InMemoryCognitionLogRepository::new();
        let stream = ModuleInstanceId::new(builtin::cognition_gate(), ReplicaIndex::ZERO);
        let old = Utc::now();
        repo.append(
            stream.clone(),
            CognitionLogEntry {
                at: old,
                text: "old".into(),
            },
        )
        .await
        .unwrap();
        let cutoff = Utc::now();
        repo.append(
            stream.clone(),
            CognitionLogEntry {
                at: cutoff,
                text: "new".into(),
            },
        )
        .await
        .unwrap();

        let events = repo.since(&stream, cutoff).await.unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].text, "new");
    }
}
