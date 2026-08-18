use std::collections::HashMap;
use std::sync::Mutex;

use async_trait::async_trait;
use nuillu_module::ports::PortError;
use nuillu_module::{
    AllocationStore, MemoLogRepository, PersistedAllocationSnapshot, PersistedMemoLogEntry,
    PersistedSessionSnapshot, SessionKey, SessionStore,
};
use nuillu_types::ModuleInstanceId;

fn lock_error(name: &str) -> PortError {
    PortError::Backend(format!("{name} lock poisoned"))
}

#[derive(Debug, Default)]
pub struct InMemoryMemoLogRepository {
    entries: Mutex<Vec<PersistedMemoLogEntry>>,
}

impl InMemoryMemoLogRepository {
    pub fn new() -> Self {
        Self::default()
    }
}

#[async_trait(?Send)]
impl MemoLogRepository for InMemoryMemoLogRepository {
    async fn append(&self, entry: &PersistedMemoLogEntry) -> Result<(), PortError> {
        self.entries
            .lock()
            .map_err(|_| lock_error("memo log repository"))?
            .push(entry.clone());
        Ok(())
    }

    async fn recent_per_owner(
        &self,
        retained_per_owner: usize,
    ) -> Result<Vec<PersistedMemoLogEntry>, PortError> {
        if retained_per_owner == 0 {
            return Ok(Vec::new());
        }
        let entries = self
            .entries
            .lock()
            .map_err(|_| lock_error("memo log repository"))?;
        let mut by_owner = HashMap::<ModuleInstanceId, Vec<PersistedMemoLogEntry>>::new();
        for entry in entries.iter().cloned() {
            by_owner
                .entry(entry.record.owner.clone())
                .or_default()
                .push(entry);
        }
        let mut recent = Vec::new();
        for owner_entries in by_owner.values_mut() {
            owner_entries.sort_by_key(|entry| entry.record.index);
            let start = owner_entries.len().saturating_sub(retained_per_owner);
            recent.extend(owner_entries.drain(start..));
        }
        recent.sort_by(|left, right| {
            left.record
                .owner
                .module
                .as_str()
                .cmp(right.record.owner.module.as_str())
                .then_with(|| {
                    left.record
                        .owner
                        .replica
                        .get()
                        .cmp(&right.record.owner.replica.get())
                })
                .then_with(|| left.record.index.cmp(&right.record.index))
        });
        Ok(recent)
    }
}

#[derive(Debug, Default)]
pub struct InMemoryAllocationStore {
    snapshots: Mutex<HashMap<ModuleInstanceId, PersistedAllocationSnapshot>>,
}

impl InMemoryAllocationStore {
    pub fn new() -> Self {
        Self::default()
    }
}

#[async_trait(?Send)]
impl AllocationStore for InMemoryAllocationStore {
    async fn load_all(&self) -> Result<Vec<PersistedAllocationSnapshot>, PortError> {
        let snapshots = self
            .snapshots
            .lock()
            .map_err(|_| lock_error("allocation store"))?;
        let mut snapshots = snapshots.values().cloned().collect::<Vec<_>>();
        snapshots.sort_by(|left, right| {
            left.owner
                .module
                .as_str()
                .cmp(right.owner.module.as_str())
                .then_with(|| left.owner.replica.get().cmp(&right.owner.replica.get()))
        });
        Ok(snapshots)
    }

    async fn save(&self, snapshot: &PersistedAllocationSnapshot) -> Result<(), PortError> {
        snapshot.validate_version()?;
        self.snapshots
            .lock()
            .map_err(|_| lock_error("allocation store"))?
            .insert(snapshot.owner.clone(), snapshot.clone());
        Ok(())
    }
}

#[derive(Debug, Default)]
pub struct InMemorySessionStore {
    snapshots: Mutex<HashMap<(ModuleInstanceId, SessionKey), PersistedSessionSnapshot>>,
}

impl InMemorySessionStore {
    pub fn new() -> Self {
        Self::default()
    }
}

#[async_trait(?Send)]
impl SessionStore for InMemorySessionStore {
    async fn load(
        &self,
        owner: &ModuleInstanceId,
        key: &SessionKey,
    ) -> Result<Option<PersistedSessionSnapshot>, PortError> {
        Ok(self
            .snapshots
            .lock()
            .map_err(|_| lock_error("session store"))?
            .get(&(owner.clone(), key.clone()))
            .cloned())
    }

    async fn save(
        &self,
        owner: &ModuleInstanceId,
        key: &SessionKey,
        snapshot: &PersistedSessionSnapshot,
    ) -> Result<(), PortError> {
        self.snapshots
            .lock()
            .map_err(|_| lock_error("session store"))?
            .insert((owner.clone(), key.clone()), snapshot.clone());
        Ok(())
    }

    async fn delete_owner(&self, owner: &ModuleInstanceId) -> Result<u64, PortError> {
        let mut snapshots = self
            .snapshots
            .lock()
            .map_err(|_| lock_error("session store"))?;
        let before = snapshots.len();
        snapshots.retain(|(entry_owner, _), _| entry_owner != owner);
        Ok((before - snapshots.len()) as u64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{TimeZone as _, Utc};
    use nuillu_blackboard::{MemoLogPayload, MemoLogRecord, ResourceAllocation};
    use nuillu_types::{ReplicaIndex, builtin};

    fn owner(replica: u8) -> ModuleInstanceId {
        ModuleInstanceId::new(builtin::memory(), ReplicaIndex::new(replica))
    }

    #[tokio::test(flavor = "current_thread")]
    async fn memo_log_retains_recent_entries_per_owner() {
        let store = InMemoryMemoLogRepository::new();
        for (owner, index) in [(owner(1), 1), (owner(0), 1), (owner(0), 2)] {
            store
                .append(&PersistedMemoLogEntry {
                    record: MemoLogRecord {
                        owner,
                        index,
                        written_at: Utc.with_ymd_and_hms(2026, 1, 1, 0, 0, 0).unwrap(),
                        content: index.to_string(),
                        cognitive: false,
                    },
                    payload: MemoLogPayload::Plain,
                })
                .await
                .unwrap();
        }

        let entries = store.recent_per_owner(1).await.unwrap();
        assert_eq!(
            entries
                .iter()
                .map(|entry| (entry.record.owner.replica.get(), entry.record.index))
                .collect::<Vec<_>>(),
            vec![(0, 2), (1, 1)]
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn allocation_and_session_saves_replace_existing_values() {
        let allocations = InMemoryAllocationStore::new();
        let allocation_owner = owner(0);
        let snapshot = PersistedAllocationSnapshot::new(
            allocation_owner.clone(),
            ResourceAllocation::default(),
            ResourceAllocation::default(),
        );
        allocations.save(&snapshot).await.unwrap();
        assert_eq!(allocations.load_all().await.unwrap(), vec![snapshot]);

        let sessions = InMemorySessionStore::new();
        let key = SessionKey::new("main").unwrap();
        let snapshot = PersistedSessionSnapshot {
            version: 1,
            items: Vec::new(),
        };
        sessions
            .save(&allocation_owner, &key, &snapshot)
            .await
            .unwrap();
        assert_eq!(
            sessions
                .load(&allocation_owner, &key)
                .await
                .unwrap()
                .unwrap()
                .version,
            1
        );
        assert_eq!(sessions.delete_owner(&allocation_owner).await.unwrap(), 1);
        assert!(
            sessions
                .load(&allocation_owner, &key)
                .await
                .unwrap()
                .is_none()
        );
    }
}
