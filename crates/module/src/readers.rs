//! Read-only views over agent state.
//!
//! Each reader exposes only the slice of the blackboard the design
//! permits the holding module to see. The compile-time signal is the
//! constructor signature: a module that takes only [`CognitionLogReader`]
//! cannot read the non-cognitive blackboard, regardless of what its
//! `run` body tries.

use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::{Arc, Mutex};

use nuillu_blackboard::{
    Blackboard, BlackboardInner, CognitionLog, CognitionLogEntryRecord, InteroceptiveState,
    MemoLogRecord, MemoryMetadata, ModuleRunStatus, ModuleRunStatusRecord, ResourceAllocation,
};
use nuillu_types::{MemoryIndex, ModuleId, ModuleInstanceId};

type MemoCursor = HashMap<ModuleInstanceId, u64>;
type SharedMemoCursor = Rc<tokio::sync::Mutex<MemoCursor>>;
type SharedCognitionCursor = Rc<tokio::sync::Mutex<Option<u64>>>;

/// Role-scoped unread cursors enabled by round-robin update inboxes.
///
/// A role is registered synchronously while its module replicas are built.
/// Readers resolve the shared cursor lazily so constructor argument order does
/// not affect whether the cursor is shared.
#[derive(Clone, Default)]
pub(crate) struct RoleReaderCursors {
    memo: Rc<RefCell<HashMap<ModuleId, SharedMemoCursor>>>,
    cognition: Rc<RefCell<HashMap<ModuleId, SharedCognitionCursor>>>,
}

impl RoleReaderCursors {
    pub(crate) fn enable_memo_round_robin(&self, role: &ModuleId) {
        self.memo.borrow_mut().entry(role.clone()).or_default();
    }

    pub(crate) fn enable_cognition_round_robin(&self, role: &ModuleId) {
        self.cognition.borrow_mut().entry(role.clone()).or_default();
    }

    fn memo_for(&self, role: &ModuleId) -> Option<SharedMemoCursor> {
        self.memo.borrow().get(role).cloned()
    }

    fn cognition_for(&self, role: &ModuleId) -> Option<SharedCognitionCursor> {
        self.cognition.borrow().get(role).cloned()
    }
}

/// Read-only access to the entire blackboard (memos + memory metadata).
///
/// Held by modules that legitimately need a wide view (cognition-gate,
/// query, memory, memory-compaction, memory-association, and the attention
/// controller.
#[derive(Clone)]
pub struct BlackboardReader {
    blackboard: Blackboard,
    last_seen_memo_indices: Arc<Mutex<MemoCursor>>,
    role_cursors: Option<(ModuleId, RoleReaderCursors)>,
}

impl BlackboardReader {
    pub(crate) fn new(blackboard: Blackboard) -> Self {
        Self {
            blackboard,
            last_seen_memo_indices: Arc::new(Mutex::new(HashMap::new())),
            role_cursors: None,
        }
    }

    pub(crate) fn new_for_role(
        blackboard: Blackboard,
        role: ModuleId,
        role_cursors: RoleReaderCursors,
    ) -> Self {
        Self {
            blackboard,
            last_seen_memo_indices: Arc::new(Mutex::new(HashMap::new())),
            role_cursors: Some((role, role_cursors)),
        }
    }

    /// Apply `f` to a borrowed snapshot. The read lock is held for the
    /// duration of `f`; do not await inside it.
    pub async fn read<R>(&self, f: impl FnOnce(&BlackboardInner) -> R) -> R {
        self.blackboard.read(f).await
    }

    pub async fn recent_memo_logs(&self) -> Vec<MemoLogRecord> {
        self.blackboard.read(|bb| bb.recent_memo_logs()).await
    }

    pub async fn unread_memo_logs(&self) -> Vec<MemoLogRecord> {
        self.unread_memo_logs_matching(|_| true).await
    }

    pub async fn unread_cognitive_memo_logs(&self) -> Vec<MemoLogRecord> {
        self.unread_memo_logs_matching(|record| record.cognitive)
            .await
    }

    async fn unread_memo_logs_matching(
        &self,
        include: impl Fn(&MemoLogRecord) -> bool,
    ) -> Vec<MemoLogRecord> {
        let shared_cursor = self
            .role_cursors
            .as_ref()
            .and_then(|(role, cursors)| cursors.memo_for(role));
        let records = if let Some(shared_cursor) = shared_cursor {
            let mut cursor = shared_cursor.lock().await;
            let records = self
                .blackboard
                .read(|bb| bb.unread_memo_logs(&cursor))
                .await;
            advance_memo_cursor(&mut cursor, &records);
            records
        } else {
            let last_seen = self
                .last_seen_memo_indices
                .lock()
                .expect("memo reader cursor poisoned")
                .clone();
            let records = self
                .blackboard
                .read(|bb| bb.unread_memo_logs(&last_seen))
                .await;
            let mut cursor = self
                .last_seen_memo_indices
                .lock()
                .expect("memo reader cursor poisoned");
            advance_memo_cursor(&mut cursor, &records);
            records
        };
        records.into_iter().filter(include).collect()
    }
}

fn advance_memo_cursor(cursor: &mut MemoCursor, records: &[MemoLogRecord]) {
    for record in records {
        cursor
            .entry(record.owner.clone())
            .and_modify(|index| *index = (*index).max(record.index))
            .or_insert(record.index);
    }
}

/// Read-only access to memory metadata without exposing memo or cognition-log
/// state.
#[derive(Clone)]
pub struct MemoryMetadataReader {
    blackboard: Blackboard,
}

impl MemoryMetadataReader {
    pub(crate) fn new(blackboard: Blackboard) -> Self {
        Self { blackboard }
    }

    pub async fn read<R>(&self, f: impl FnOnce(&HashMap<MemoryIndex, MemoryMetadata>) -> R) -> R {
        self.blackboard.read(|bb| f(bb.memory_metadata())).await
    }

    pub async fn snapshot(&self) -> HashMap<MemoryIndex, MemoryMetadata> {
        self.blackboard
            .read(|bb| bb.memory_metadata().clone())
            .await
    }
}

/// Read-only access to the cognition log. The holder
/// cannot see memos, memory metadata, or allocation through this
/// capability.
#[derive(Clone)]
pub struct CognitionLogReader {
    blackboard: Blackboard,
    owner: Option<ModuleInstanceId>,
    last_seen_cognition_index: Rc<Cell<Option<u64>>>,
    role_cursors: Option<(ModuleId, RoleReaderCursors)>,
}

impl CognitionLogReader {
    pub(crate) fn new(blackboard: Blackboard) -> Self {
        Self {
            blackboard,
            owner: None,
            last_seen_cognition_index: Rc::new(Cell::new(None)),
            role_cursors: None,
        }
    }

    #[cfg(test)]
    pub(crate) fn new_for_owner(blackboard: Blackboard, owner: ModuleInstanceId) -> Self {
        Self {
            blackboard,
            owner: Some(owner),
            last_seen_cognition_index: Rc::new(Cell::new(None)),
            role_cursors: None,
        }
    }

    pub(crate) fn new_for_owner_with_role_cursors(
        blackboard: Blackboard,
        owner: ModuleInstanceId,
        role_cursors: RoleReaderCursors,
    ) -> Self {
        Self {
            blackboard,
            owner: Some(owner.clone()),
            last_seen_cognition_index: Rc::new(Cell::new(None)),
            role_cursors: Some((owner.module, role_cursors)),
        }
    }

    pub async fn read<R>(&self, f: impl FnOnce(&CognitionLog) -> R) -> R {
        self.blackboard
            .read(|bb| {
                let log = if let Some(owner) = &self.owner {
                    bb.cognition_log_excluding_owner(owner)
                } else {
                    bb.cognition_log()
                };
                f(&log)
            })
            .await
    }

    pub async fn snapshot(&self) -> nuillu_blackboard::CognitionLogSet {
        self.blackboard
            .read(|bb| {
                if let Some(owner) = &self.owner {
                    bb.cognition_log_set_excluding_owner(owner)
                } else {
                    bb.cognition_log_set()
                }
            })
            .await
    }

    pub async fn peek_unread_events(&self) -> Vec<CognitionLogEntryRecord> {
        let shared_cursor = self
            .role_cursors
            .as_ref()
            .and_then(|(role, cursors)| cursors.cognition_for(role));
        let last_seen = if let Some(shared_cursor) = shared_cursor {
            *shared_cursor.lock().await
        } else {
            self.last_seen_cognition_index.get()
        };
        let records = self
            .blackboard
            .read(|bb| bb.unread_cognition_log_entries(last_seen))
            .await;
        self.filter_records(records)
    }

    pub async fn unread_events(&self) -> Vec<CognitionLogEntryRecord> {
        let shared_cursor = self
            .role_cursors
            .as_ref()
            .and_then(|(role, cursors)| cursors.cognition_for(role));
        let records = if let Some(shared_cursor) = shared_cursor {
            let mut cursor = shared_cursor.lock().await;
            let records = self
                .blackboard
                .read(|bb| bb.unread_cognition_log_entries(*cursor))
                .await;
            if let Some(index) = records.last().map(|record| record.index) {
                *cursor = Some(index);
            }
            records
        } else {
            let last_seen = self.last_seen_cognition_index.get();
            let records = self
                .blackboard
                .read(|bb| bb.unread_cognition_log_entries(last_seen))
                .await;
            if let Some(index) = records.last().map(|record| record.index) {
                self.last_seen_cognition_index.set(Some(index));
            }
            records
        };
        self.filter_records(records)
    }

    fn filter_records(
        &self,
        records: Vec<CognitionLogEntryRecord>,
    ) -> Vec<CognitionLogEntryRecord> {
        let Some(owner) = &self.owner else {
            return records;
        };
        records
            .into_iter()
            .filter(|record| record.source != *owner && record.entry.origin.owner != *owner)
            .collect()
    }
}

/// Read-only access to the resource-allocation snapshot. Modules may inspect
/// activation priority and derived scheduling state, but only holders of
/// `AllocationWriter` can change it.
#[derive(Clone)]
pub struct AllocationReader {
    blackboard: Blackboard,
}

impl AllocationReader {
    pub(crate) fn new(blackboard: Blackboard) -> Self {
        Self { blackboard }
    }

    pub async fn read<R>(&self, f: impl FnOnce(&ResourceAllocation) -> R) -> R {
        self.blackboard.read(|bb| f(bb.allocation())).await
    }

    pub async fn snapshot(&self) -> ResourceAllocation {
        self.blackboard.read(|bb| bb.allocation().clone()).await
    }

    pub async fn registered_module_ids(&self) -> Vec<nuillu_types::ModuleId> {
        self.blackboard
            .read(|bb| {
                let mut ids = bb.module_policies().keys().cloned().collect::<Vec<_>>();
                ids.sort_by(|a, b| a.as_str().cmp(b.as_str()));
                ids
            })
            .await
    }
}

#[derive(Clone)]
pub struct InteroceptiveReader {
    blackboard: Blackboard,
}

impl InteroceptiveReader {
    pub(crate) fn new(blackboard: Blackboard) -> Self {
        Self { blackboard }
    }

    pub async fn snapshot(&self) -> InteroceptiveState {
        self.blackboard.read(|bb| bb.interoception().clone()).await
    }

    pub async fn read<R>(&self, f: impl FnOnce(&InteroceptiveState) -> R) -> R {
        self.blackboard.read(|bb| f(bb.interoception())).await
    }
}

/// Read-only access to scheduler-owned module lifecycle status.
#[derive(Clone)]
pub struct ModuleStatusReader {
    blackboard: Blackboard,
}

impl ModuleStatusReader {
    pub(crate) fn new(blackboard: Blackboard) -> Self {
        Self { blackboard }
    }

    pub async fn status_for_instance(&self, owner: &ModuleInstanceId) -> ModuleRunStatus {
        self.blackboard
            .read(|bb| {
                bb.module_status_for_instance(owner)
                    .cloned()
                    .unwrap_or_default()
            })
            .await
    }

    pub async fn records(&self) -> Vec<ModuleRunStatusRecord> {
        self.blackboard.read(|bb| bb.module_status_records()).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use chrono::{TimeZone, Utc};
    use nuillu_blackboard::{BlackboardCommand, CognitionLogEntry, CognitionLogOrigin};
    use nuillu_types::{ReplicaIndex, builtin};

    #[tokio::test]
    async fn module_status_reader_exposes_scheduler_owned_status() {
        let blackboard = Blackboard::default();
        let owner = ModuleInstanceId::new(builtin::speak(), ReplicaIndex::ZERO);
        blackboard
            .apply(BlackboardCommand::SetModuleRunStatus {
                owner: owner.clone(),
                status: ModuleRunStatus::Activating,
            })
            .await;
        let reader = ModuleStatusReader::new(blackboard);

        assert_eq!(
            reader.status_for_instance(&owner).await,
            ModuleRunStatus::Activating
        );
        assert_eq!(
            reader
                .status_for_instance(&ModuleInstanceId::new(
                    builtin::query_memory(),
                    ReplicaIndex::ZERO,
                ))
                .await,
            ModuleRunStatus::Inactive
        );
    }

    #[tokio::test]
    async fn unread_memo_logs_advance_per_reader_handle() {
        let blackboard = Blackboard::default();
        let owner = ModuleInstanceId::new(builtin::sensory(), ReplicaIndex::ZERO);
        let reader_a = BlackboardReader::new(blackboard.clone());
        let reader_b = BlackboardReader::new(blackboard.clone());

        blackboard
            .update_memo(
                owner.clone(),
                "first".into(),
                Utc.timestamp_opt(0, 0).unwrap(),
            )
            .await;

        let a_first = reader_a.unread_memo_logs().await;
        assert_eq!(
            a_first
                .iter()
                .map(|record| (record.owner.clone(), record.index, record.content.as_str()))
                .collect::<Vec<_>>(),
            vec![(owner.clone(), 0, "first")]
        );
        assert!(reader_a.unread_memo_logs().await.is_empty());

        let b_first = reader_b.unread_memo_logs().await;
        assert_eq!(b_first.len(), 1);
        assert_eq!(b_first[0].index, 0);

        blackboard
            .update_memo(owner, "second".into(), Utc.timestamp_opt(1, 0).unwrap())
            .await;

        let a_second = reader_a.unread_memo_logs().await;
        assert_eq!(
            a_second
                .iter()
                .map(|record| (record.index, record.content.as_str()))
                .collect::<Vec<_>>(),
            vec![(1, "second")]
        );
        let b_second = reader_b.unread_memo_logs().await;
        assert_eq!(
            b_second
                .iter()
                .map(|record| (record.index, record.content.as_str()))
                .collect::<Vec<_>>(),
            vec![(1, "second")]
        );
    }

    #[tokio::test]
    async fn unread_cognitive_memo_logs_filters_and_advances_reader_cursor() {
        let blackboard = Blackboard::default();
        let owner = ModuleInstanceId::new(builtin::sensory(), ReplicaIndex::ZERO);
        let reader = BlackboardReader::new(blackboard.clone());

        blackboard
            .update_memo(
                owner.clone(),
                "non-cognitive".into(),
                Utc.timestamp_opt(0, 0).unwrap(),
            )
            .await;
        blackboard
            .update_cognitive_memo(
                owner.clone(),
                "cognitive".into(),
                Utc.timestamp_opt(1, 0).unwrap(),
            )
            .await;

        let cognitive = reader.unread_cognitive_memo_logs().await;
        assert_eq!(
            cognitive
                .iter()
                .map(|record| (record.index, record.content.as_str(), record.cognitive))
                .collect::<Vec<_>>(),
            vec![(1, "cognitive", true)]
        );
        assert!(reader.unread_memo_logs().await.is_empty());
    }

    #[tokio::test]
    async fn unread_cognition_log_entries_advance_per_reader_cursor() {
        let blackboard = Blackboard::default();
        let stream = ModuleInstanceId::new(builtin::cognition_gate(), ReplicaIndex::ZERO);
        let reader_a = CognitionLogReader::new(blackboard.clone());
        let reader_a_clone = reader_a.clone();
        let reader_b = CognitionLogReader::new(blackboard.clone());

        blackboard
            .apply(BlackboardCommand::AppendCognitionLog {
                source: stream.clone(),
                entry: CognitionLogEntry {
                    at: Utc.timestamp_opt(0, 0).unwrap(),
                    text: "first".into(),
                    origin: CognitionLogOrigin::direct(stream.clone()),
                },
            })
            .await;

        let a_first = reader_a.unread_events().await;
        assert_eq!(
            a_first
                .iter()
                .map(|record| (
                    record.index,
                    record.source.clone(),
                    record.entry.text.as_str()
                ))
                .collect::<Vec<_>>(),
            vec![(0, stream.clone(), "first")]
        );
        assert!(reader_a_clone.unread_events().await.is_empty());

        let b_first = reader_b.unread_events().await;
        assert_eq!(
            b_first
                .iter()
                .map(|record| (record.index, record.entry.text.as_str()))
                .collect::<Vec<_>>(),
            vec![(0, "first")]
        );

        blackboard
            .apply(BlackboardCommand::AppendCognitionLog {
                source: stream.clone(),
                entry: CognitionLogEntry {
                    at: Utc.timestamp_opt(1, 0).unwrap(),
                    text: "second".into(),
                    origin: CognitionLogOrigin::direct(stream),
                },
            })
            .await;

        let a_second = reader_a_clone.unread_events().await;
        assert_eq!(
            a_second
                .iter()
                .map(|record| (record.index, record.entry.text.as_str()))
                .collect::<Vec<_>>(),
            vec![(1, "second")]
        );
        let b_second = reader_b.unread_events().await;
        assert_eq!(
            b_second
                .iter()
                .map(|record| (record.index, record.entry.text.as_str()))
                .collect::<Vec<_>>(),
            vec![(1, "second")]
        );
    }

    #[tokio::test]
    async fn scoped_cognition_log_reader_filters_self_source_and_origin() {
        let blackboard = Blackboard::default();
        let self_owner = ModuleInstanceId::new(builtin::interpreter(), ReplicaIndex::ZERO);
        let gate = ModuleInstanceId::new(builtin::cognition_gate(), ReplicaIndex::ZERO);
        let sensory = ModuleInstanceId::new(builtin::sensory(), ReplicaIndex::ZERO);
        let reader = CognitionLogReader::new_for_owner(blackboard.clone(), self_owner.clone());

        blackboard
            .apply(BlackboardCommand::AppendCognitionLog {
                source: self_owner.clone(),
                entry: CognitionLogEntry {
                    at: Utc.timestamp_opt(0, 0).unwrap(),
                    text: "direct self".into(),
                    origin: CognitionLogOrigin::direct(self_owner.clone()),
                },
            })
            .await;
        blackboard
            .apply(BlackboardCommand::AppendCognitionLog {
                source: gate.clone(),
                entry: CognitionLogEntry {
                    at: Utc.timestamp_opt(1, 0).unwrap(),
                    text: "promoted self memo".into(),
                    origin: CognitionLogOrigin::memo(self_owner.clone(), 7),
                },
            })
            .await;
        blackboard
            .apply(BlackboardCommand::AppendCognitionLog {
                source: gate.clone(),
                entry: CognitionLogEntry {
                    at: Utc.timestamp_opt(2, 0).unwrap(),
                    text: "promoted sensory memo".into(),
                    origin: CognitionLogOrigin::memo(sensory.clone(), 1),
                },
            })
            .await;

        let unread = reader.unread_events().await;
        assert_eq!(
            unread
                .iter()
                .map(|record| (record.index, record.entry.text.as_str()))
                .collect::<Vec<_>>(),
            vec![(2, "promoted sensory memo")]
        );
        assert!(reader.unread_events().await.is_empty());

        let snapshot = reader.snapshot().await;
        assert_eq!(snapshot.logs().len(), 1);
        assert_eq!(snapshot.logs()[0].source, gate);
        assert_eq!(snapshot.logs()[0].entries[0].text, "promoted sensory memo");
    }

    #[tokio::test]
    async fn peek_unread_cognition_log_entries_does_not_advance_reader_cursor() {
        let blackboard = Blackboard::default();
        let stream = ModuleInstanceId::new(builtin::cognition_gate(), ReplicaIndex::ZERO);
        let reader = CognitionLogReader::new(blackboard.clone());

        blackboard
            .apply(BlackboardCommand::AppendCognitionLog {
                source: stream.clone(),
                entry: CognitionLogEntry {
                    at: Utc.timestamp_opt(0, 0).unwrap(),
                    text: "first".into(),
                    origin: CognitionLogOrigin::direct(stream.clone()),
                },
            })
            .await;

        let peeked = reader.peek_unread_events().await;
        assert_eq!(
            peeked
                .iter()
                .map(|record| (record.index, record.entry.text.as_str()))
                .collect::<Vec<_>>(),
            vec![(0, "first")]
        );
        assert_eq!(
            reader
                .peek_unread_events()
                .await
                .iter()
                .map(|record| (record.index, record.entry.text.as_str()))
                .collect::<Vec<_>>(),
            vec![(0, "first")]
        );
        assert_eq!(
            reader
                .unread_events()
                .await
                .iter()
                .map(|record| (
                    record.index,
                    record.source.clone(),
                    record.entry.text.as_str()
                ))
                .collect::<Vec<_>>(),
            vec![(0, stream, "first")]
        );
        assert!(reader.unread_events().await.is_empty());
    }
}
