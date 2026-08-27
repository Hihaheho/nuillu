use std::collections::{HashSet, VecDeque};

use anyhow::Result;
use async_trait::async_trait;
use nuillu_module::{
    CognitionLogEntryRecord, CognitionLogReader, CognitionLogUpdatedInbox, Module, TypedMemo,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest as _, Sha256};

const MAX_SEEN_FINGERPRINTS: usize = 1_024;

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SubsystemGateMemo {
    seen: VecDeque<String>,
}

#[derive(Debug, Default)]
pub struct SubsystemGateBatch {
    inner: Vec<CognitionLogEntryRecord>,
    outer: Vec<CognitionLogEntryRecord>,
}

/// Deterministic cognition-to-memo bridge between one scope and its parent.
///
/// Outer cognition becomes cognitive memo input in the inner scope. Inner
/// cognition becomes cognitive memo input in the outer scope. Cognition-gate
/// remains the sole admission boundary on both sides.
pub struct SubsystemGateModule {
    inner_updates: CognitionLogUpdatedInbox,
    inner_reader: CognitionLogReader,
    inner_memo: TypedMemo<SubsystemGateMemo>,
    outer_updates: CognitionLogUpdatedInbox,
    outer_reader: CognitionLogReader,
    outer_memo: TypedMemo<SubsystemGateMemo>,
    state: Option<SubsystemGateMemo>,
}

impl SubsystemGateModule {
    pub fn new(
        inner_updates: CognitionLogUpdatedInbox,
        inner_reader: CognitionLogReader,
        inner_memo: TypedMemo<SubsystemGateMemo>,
        outer_updates: CognitionLogUpdatedInbox,
        outer_reader: CognitionLogReader,
        outer_memo: TypedMemo<SubsystemGateMemo>,
    ) -> Self {
        Self {
            inner_updates,
            inner_reader,
            inner_memo,
            outer_updates,
            outer_reader,
            outer_memo,
            state: None,
        }
    }

    async fn next_gate_batch(&mut self) -> Result<SubsystemGateBatch> {
        tokio::select! {
            update = self.inner_updates.next_item() => {
                let _ = update?;
            }
            update = self.outer_updates.next_item() => {
                let _ = update?;
            }
        }
        let _ = self.inner_updates.take_ready_items()?;
        let _ = self.outer_updates.take_ready_items()?;
        Ok(SubsystemGateBatch {
            inner: self.inner_reader.unread_events().await,
            outer: self.outer_reader.unread_events().await,
        })
    }

    async fn ensure_state_loaded(&mut self) {
        if self.state.is_none() {
            let inner = latest_state(&self.inner_memo).await;
            let outer = latest_state(&self.outer_memo).await;
            self.state = Some(merge_states(inner, outer));
        }
    }

    async fn bridge_batch(&mut self, batch: &SubsystemGateBatch) -> Result<()> {
        self.ensure_state_loaded().await;

        // Parent cognition is offered to the cognition-gate in this scope.
        bridge_records(
            &batch.outer,
            self.state.as_mut().expect("bridge state is initialized"),
            &self.inner_memo,
        )
        .await?;

        // Child cognition is offered to the cognition-gate in the parent scope.
        bridge_records(
            &batch.inner,
            self.state.as_mut().expect("bridge state is initialized"),
            &self.outer_memo,
        )
        .await?;
        Ok(())
    }
}

async fn latest_state(memo: &TypedMemo<SubsystemGateMemo>) -> SubsystemGateMemo {
    memo.recent_logs()
        .await
        .last()
        .map(|record| record.data().clone())
        .unwrap_or_default()
}

fn merge_states(left: SubsystemGateMemo, right: SubsystemGateMemo) -> SubsystemGateMemo {
    let mut merged = SubsystemGateMemo::default();
    let mut seen = HashSet::new();
    for fingerprint in left.seen.into_iter().chain(right.seen) {
        if seen.insert(fingerprint.clone()) {
            merged.seen.push_back(fingerprint);
            while merged.seen.len() > MAX_SEEN_FINGERPRINTS {
                merged.seen.pop_front();
            }
        }
    }
    merged
}

async fn bridge_records(
    records: &[CognitionLogEntryRecord],
    state: &mut SubsystemGateMemo,
    target: &TypedMemo<SubsystemGateMemo>,
) -> Result<()> {
    for record in take_unseen(records, state)? {
        target
            .write_forwarded_cognitive(state.clone(), record.entry.clone())
            .await;
    }
    Ok(())
}

fn take_unseen<'a>(
    records: &'a [CognitionLogEntryRecord],
    state: &mut SubsystemGateMemo,
) -> Result<Vec<&'a CognitionLogEntryRecord>> {
    let mut seen = state.seen.iter().cloned().collect::<HashSet<_>>();
    let mut unseen = Vec::new();
    for record in records {
        let fingerprint = fingerprint(&record.entry)?;
        if !seen.insert(fingerprint.clone()) {
            continue;
        }
        state.seen.push_back(fingerprint);
        while state.seen.len() > MAX_SEEN_FINGERPRINTS {
            state.seen.pop_front();
        }
        unseen.push(record);
    }
    Ok(unseen)
}

fn fingerprint(entry: &nuillu_blackboard::CognitionLogEntry) -> Result<String> {
    let encoded = serde_json::to_vec(entry)?;
    Ok(format!("{:x}", Sha256::digest(encoded)))
}

#[async_trait(?Send)]
impl nuillu_module::StaticModule for SubsystemGateModule {
    fn id() -> &'static str {
        "subsystem-gate"
    }

    fn peer_context() -> Option<&'static str> {
        Some(
            "Passes cognition across the local boundary as cognitive memo input for the cognition gate on the other side.",
        )
    }
}

#[async_trait(?Send)]
impl Module for SubsystemGateModule {
    type Batch = SubsystemGateBatch;

    async fn next_batch(&mut self) -> Result<Self::Batch> {
        self.next_gate_batch().await
    }

    async fn activate(
        &mut self,
        _cx: &nuillu_module::ActivateCx<'_>,
        batch: &Self::Batch,
    ) -> Result<()> {
        self.bridge_batch(batch).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{TimeZone as _, Utc};
    use nuillu_blackboard::{CognitionLogEntry, CognitionLogOrigin};
    use nuillu_types::{ModuleInstanceId, ReplicaIndex, builtin};

    #[test]
    fn fingerprint_is_stable_for_the_same_provenance() {
        let owner = ModuleInstanceId::new(builtin::cognition_gate(), ReplicaIndex::ZERO);
        let entry = CognitionLogEntry {
            at: Utc.timestamp_opt(1_700_000_000, 0).unwrap(),
            text: "shared cognition".to_owned(),
            origin: CognitionLogOrigin::direct(owner),
        };
        assert_eq!(fingerprint(&entry).unwrap(), fingerprint(&entry).unwrap());
    }

    #[test]
    fn bridge_state_forwards_each_cognition_record_once() {
        let owner = ModuleInstanceId::new(builtin::cognition_gate(), ReplicaIndex::ZERO);
        let entry = CognitionLogEntry {
            at: Utc.timestamp_opt(1_700_000_000, 0).unwrap(),
            text: "shared cognition".to_owned(),
            origin: CognitionLogOrigin::direct(owner.clone()),
        };
        let records = vec![CognitionLogEntryRecord {
            index: 0,
            source: owner,
            entry,
        }];
        let mut state = SubsystemGateMemo::default();

        assert_eq!(
            take_unseen(&records, &mut state).unwrap(),
            vec![&records[0]]
        );
        let returned_through_other_scope = vec![CognitionLogEntryRecord {
            index: 42,
            source: ModuleInstanceId::new(builtin::subsystem_gate(), ReplicaIndex::ZERO),
            entry: records[0].entry.clone(),
        }];
        assert!(
            take_unseen(&returned_through_other_scope, &mut state)
                .unwrap()
                .is_empty(),
            "the shared lineage fingerprint must stop a bridged entry returning in the opposite direction"
        );
    }

    #[test]
    fn restored_direction_states_merge_into_one_lineage_set() {
        let left = SubsystemGateMemo {
            seen: VecDeque::from(["left".to_owned(), "shared".to_owned()]),
        };
        let right = SubsystemGateMemo {
            seen: VecDeque::from(["shared".to_owned(), "right".to_owned()]),
        };

        assert_eq!(
            merge_states(left, right).seen,
            VecDeque::from(["left".to_owned(), "shared".to_owned(), "right".to_owned()])
        );
    }
}
