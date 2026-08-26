//! In-memory adapters for local development and tests.

mod agent;
mod embedding;
mod events;
mod memory;
mod persistence;
mod policy;

pub use agent::InMemoryAgentStore;
pub use events::{
    InMemoryAmbientSensorySnapshotStore, InMemoryExternalActionEventStore,
    InMemoryLlmTranscriptStore, InMemoryOneShotSensoryInputStore, InMemoryUtteranceEventStore,
};
pub use memory::InMemoryMemoryStore;
pub use persistence::{InMemoryAllocationStore, InMemoryMemoLogRepository, InMemorySessionStore};
pub use policy::InMemoryPolicyStore;

use std::sync::Mutex;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use nuillu_blackboard::CognitionLogEntry;
use nuillu_module::ports::{
    CognitionLogCursor, CognitionLogRepository, PersistedCognitionLogEntry,
    PersistedCognitionLogPageEntry, PortError,
};
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

    async fn recent(&self, limit: usize) -> Result<Vec<PersistedCognitionLogEntry>, PortError> {
        if limit == 0 {
            return Ok(Vec::new());
        }
        let events = self
            .events
            .lock()
            .map_err(|_| PortError::Backend("cognition log repository lock poisoned".into()))?;
        let mut records = events
            .iter()
            .rev()
            .take(limit)
            .map(|(source, entry)| PersistedCognitionLogEntry {
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
        let events = self
            .events
            .lock()
            .map_err(|_| PortError::Backend("cognition log repository lock poisoned".into()))?;
        // Appends never remove earlier events, so the insertion index is a
        // stable identity and doubles as the keyset anchor.
        Ok(events
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
            .map(|(index, (source, entry))| PersistedCognitionLogPageEntry {
                id: i64::try_from(index).unwrap_or(i64::MAX),
                source: source.clone(),
                entry: entry.clone(),
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nuillu_blackboard::CognitionLogOrigin;
    use nuillu_types::{ReplicaIndex, builtin};

    #[tokio::test(flavor = "current_thread")]
    async fn cognition_log_repo_filters_by_time() {
        let repo = InMemoryCognitionLogRepository::new();
        let stream = ModuleInstanceId::new(builtin::cognition_gate(), ReplicaIndex::ZERO);
        let old = Utc::now();
        repo.append(
            stream.clone(),
            CognitionLogEntry {
                at: old,
                text: "old".into(),
                origin: CognitionLogOrigin::direct(stream.clone()),
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
                origin: CognitionLogOrigin::direct(stream.clone()),
            },
        )
        .await
        .unwrap();

        let events = repo.since(&stream, cutoff).await.unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].text, "new");
    }

    #[tokio::test(flavor = "current_thread")]
    async fn cognition_log_repo_pages_newest_first_around_a_cursor() {
        let repo = InMemoryCognitionLogRepository::new();
        let stream = ModuleInstanceId::new(builtin::cognition_gate(), ReplicaIndex::ZERO);
        for text in ["first", "second", "third"] {
            repo.append(
                stream.clone(),
                CognitionLogEntry {
                    at: Utc::now(),
                    text: text.into(),
                    origin: CognitionLogOrigin::direct(stream.clone()),
                },
            )
            .await
            .unwrap();
        }

        let newest = repo.page(CognitionLogCursor::Newest, 2).await.unwrap();
        assert_eq!(page_texts(&newest), vec!["third", "second"]);
        assert!(newest[0].id > newest[1].id);

        let older = repo
            .page(
                CognitionLogCursor::Older {
                    before_id: newest[1].id,
                },
                2,
            )
            .await
            .unwrap();
        assert_eq!(page_texts(&older), vec!["first"]);

        let newer = repo
            .page(
                CognitionLogCursor::Newer {
                    after_id: newest[1].id,
                },
                8,
            )
            .await
            .unwrap();
        assert_eq!(page_texts(&newer), vec!["third"]);
    }

    fn page_texts(page: &[PersistedCognitionLogPageEntry]) -> Vec<&str> {
        page.iter()
            .map(|record| record.entry.text.as_str())
            .collect()
    }
}
