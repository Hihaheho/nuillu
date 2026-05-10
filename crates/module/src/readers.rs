//! Read-only views over agent state.
//!
//! Each reader exposes only the slice of the blackboard the design
//! permits the holding module to see. The compile-time signal is the
//! constructor signature: a module that takes only [`CognitionLogReader`]
//! cannot read the non-cognitive blackboard, regardless of what its
//! `run` body tries.

use std::cell::Cell;
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::{Arc, Mutex};

use nuillu_blackboard::{
    Blackboard, BlackboardInner, CognitionLog, CognitionLogEntryRecord, MemoLogRecord,
    ModuleRunStatus, ModuleRunStatusRecord, ResourceAllocation,
};
use nuillu_types::ModuleInstanceId;

/// Read-only access to the entire blackboard (memos + memory metadata).
///
/// Held by modules that legitimately need a wide view (cognition-gate,
/// query, memory, memory-compaction, and the attention controller.
#[derive(Clone)]
pub struct BlackboardReader {
    blackboard: Blackboard,
    last_seen_memo_indices: Arc<Mutex<HashMap<ModuleInstanceId, u64>>>,
}

impl BlackboardReader {
    pub(crate) fn new(blackboard: Blackboard) -> Self {
        Self {
            blackboard,
            last_seen_memo_indices: Arc::new(Mutex::new(HashMap::new())),
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
        let last_seen = self
            .last_seen_memo_indices
            .lock()
            .expect("memo reader cursor poisoned")
            .clone();
        let records = self
            .blackboard
            .read(|bb| bb.unread_memo_logs(&last_seen))
            .await;
        if !records.is_empty() {
            let mut cursor = self
                .last_seen_memo_indices
                .lock()
                .expect("memo reader cursor poisoned");
            for record in &records {
                cursor
                    .entry(record.owner.clone())
                    .and_modify(|index| *index = (*index).max(record.index))
                    .or_insert(record.index);
            }
        }
        records
    }
}

/// Read-only access to the cognition log. The holder
/// cannot see memos, memory metadata, or allocation through this
/// capability.
#[derive(Clone)]
pub struct CognitionLogReader {
    blackboard: Blackboard,
    last_seen_cognition_index: Rc<Cell<Option<u64>>>,
}

impl CognitionLogReader {
    pub(crate) fn new(blackboard: Blackboard) -> Self {
        Self {
            blackboard,
            last_seen_cognition_index: Rc::new(Cell::new(None)),
        }
    }

    pub async fn read<R>(&self, f: impl FnOnce(&CognitionLog) -> R) -> R {
        self.blackboard
            .read(|bb| {
                let log = bb.cognition_log();
                f(&log)
            })
            .await
    }

    pub async fn snapshot(&self) -> nuillu_blackboard::CognitionLogSet {
        self.blackboard.read(|bb| bb.cognition_log_set()).await
    }

    pub async fn unread_events(&self) -> Vec<CognitionLogEntryRecord> {
        let last_seen = self.last_seen_cognition_index.get();
        let records = self
            .blackboard
            .read(|bb| bb.unread_cognition_log_entries(last_seen))
            .await;
        if let Some(index) = records.last().map(|record| record.index) {
            self.last_seen_cognition_index.set(Some(index));
        }
        records
    }
}

/// Read-only access to the resource-allocation snapshot. Modules may inspect
/// allocation guidance for themselves and for other modules, but only holders
/// of `AllocationWriter` can change it.
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

    pub async fn controller_schema_json(&self) -> serde_json::Value {
        let ids = self.registered_module_ids().await;
        let module_ids = ids.iter().map(|id| id.as_str()).collect::<Vec<_>>();

        let patch_items = if module_ids.is_empty() {
            serde_json::Value::Bool(false)
        } else {
            serde_json::json!({
                "type": "object",
                "additionalProperties": false,
                "properties": {
                    "module_id": {
                        "enum": module_ids,
                    },
                    "activation_ratio": {
                        "type": "number",
                        "description": "Runtime clamps activation_ratio to 0.0..=1.0 before deriving active replicas.",
                    },
                    "guidance": {
                        "type": "string",
                    },
                    "tier": {
                        "enum": ["Cheap", "Default", "Premium"],
                    },
                },
                "required": [
                    "module_id",
                    "activation_ratio",
                    "guidance",
                    "tier",
                ],
            })
        };

        serde_json::json!({
            "type": "object",
            "additionalProperties": false,
            "properties": {
                "memo": {
                    "type": "string",
                },
                "allocations": {
                    "type": "array",
                    "items": patch_items,
                },
            },
            "required": ["memo", "allocations"],
        })
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

    pub async fn snapshot_json(&self) -> serde_json::Value {
        self.blackboard.read(|bb| bb.module_statuses()).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use chrono::{TimeZone, Utc};
    use nuillu_blackboard::{
        BlackboardCommand, Bpm, CognitionLogEntry, ModulePolicy, linear_ratio_fn,
    };
    use nuillu_types::{ReplicaCapRange, ReplicaIndex, builtin};

    fn test_policy(range: ReplicaCapRange) -> ModulePolicy {
        ModulePolicy::new(
            range,
            Bpm::from_f64(1.0)..=Bpm::from_f64(60.0),
            linear_ratio_fn,
        )
    }

    #[tokio::test]
    async fn controller_schema_enumerates_registered_modules_with_cap_ranges() {
        let blackboard = Blackboard::default();
        blackboard
            .apply(BlackboardCommand::SetModulePolicies {
                policies: vec![
                    (
                        builtin::query_vector(),
                        test_policy(ReplicaCapRange::new(0, 2).unwrap()),
                    ),
                    (
                        builtin::speak(),
                        test_policy(ReplicaCapRange::new(0, 0).unwrap()),
                    ),
                ],
            })
            .await;
        let reader = AllocationReader::new(blackboard);

        let schema = reader.controller_schema_json().await;
        assert_eq!(
            schema,
            serde_json::json!({
                "type": "object",
                "additionalProperties": false,
                "properties": {
                    "memo": {
                        "type": "string",
                    },
                    "allocations": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": false,
                            "properties": {
                                "module_id": {
                                    "enum": ["query-vector", "speak"],
                                },
                                "activation_ratio": {
                                    "type": "number",
                                    "description": "Runtime clamps activation_ratio to 0.0..=1.0 before deriving active replicas.",
                                },
                                "guidance": {
                                    "type": "string",
                                },
                                "tier": {
                                    "enum": ["Cheap", "Default", "Premium"],
                                },
                            },
                            "required": [
                                "module_id",
                                "activation_ratio",
                                "guidance",
                                "tier",
                            ],
                        },
                    },
                },
                "required": ["memo", "allocations"],
            })
        );
    }

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
                    builtin::query_vector(),
                    ReplicaIndex::ZERO,
                ))
                .await,
            ModuleRunStatus::Inactive
        );
        assert_eq!(
            reader.snapshot_json().await,
            serde_json::json!({
                "speak": {
                    "state": "activating"
                }
            })
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
                source: stream,
                entry: CognitionLogEntry {
                    at: Utc.timestamp_opt(1, 0).unwrap(),
                    text: "second".into(),
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
}
