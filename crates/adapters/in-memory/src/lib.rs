//! In-memory adapters for local development and tests.

use std::sync::Mutex;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use nuillu_blackboard::AttentionStreamEvent;
use nuillu_module::ports::{AttentionRepository, PortError};
use nuillu_types::ModuleInstanceId;

#[derive(Debug, Default)]
pub struct InMemoryAttentionRepository {
    events: Mutex<Vec<(ModuleInstanceId, AttentionStreamEvent)>>,
}

impl InMemoryAttentionRepository {
    pub fn new() -> Self {
        Self::default()
    }
}

#[async_trait(?Send)]
impl AttentionRepository for InMemoryAttentionRepository {
    async fn append(
        &self,
        stream: ModuleInstanceId,
        event: AttentionStreamEvent,
    ) -> Result<(), PortError> {
        self.events
            .lock()
            .map_err(|_| PortError::Backend("attention repository lock poisoned".into()))?
            .push((stream, event));
        Ok(())
    }

    async fn since(
        &self,
        stream: &ModuleInstanceId,
        from: DateTime<Utc>,
    ) -> Result<Vec<AttentionStreamEvent>, PortError> {
        let events = self
            .events
            .lock()
            .map_err(|_| PortError::Backend("attention repository lock poisoned".into()))?;
        Ok(events
            .iter()
            .filter(|(owner, event)| owner == stream && event.at >= from)
            .map(|(_, event)| event.clone())
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nuillu_types::{ReplicaIndex, builtin};

    #[tokio::test]
    async fn attention_repo_filters_by_time() {
        let repo = InMemoryAttentionRepository::new();
        let stream = ModuleInstanceId::new(builtin::summarize(), ReplicaIndex::ZERO);
        let old = Utc::now();
        repo.append(
            stream.clone(),
            AttentionStreamEvent {
                at: old,
                text: "old".into(),
            },
        )
        .await
        .unwrap();
        let cutoff = Utc::now();
        repo.append(
            stream.clone(),
            AttentionStreamEvent {
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
