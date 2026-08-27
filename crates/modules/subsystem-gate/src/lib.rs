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
    inner_state: Option<SubsystemGateMemo>,
    outer_state: Option<SubsystemGateMemo>,
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
            inner_state: None,
            outer_state: None,
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
        if self.inner_state.is_none() {
            self.inner_state = Some(latest_state(&self.inner_memo).await);
        }
        if self.outer_state.is_none() {
            self.outer_state = Some(latest_state(&self.outer_memo).await);
        }
    }

    async fn bridge_batch(&mut self, batch: &SubsystemGateBatch) -> Result<()> {
        self.ensure_state_loaded().await;

        // Parent cognition is offered to the cognition-gate in this scope.
        bridge_records(
            &batch.outer,
            self.inner_state
                .as_mut()
                .expect("inner bridge state is initialized"),
            &self.inner_memo,
        )
        .await?;

        // Child cognition is offered to the cognition-gate in the parent scope.
        bridge_records(
            &batch.inner,
            self.outer_state
                .as_mut()
                .expect("outer bridge state is initialized"),
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

async fn bridge_records(
    records: &[CognitionLogEntryRecord],
    state: &mut SubsystemGateMemo,
    target: &TypedMemo<SubsystemGateMemo>,
) -> Result<()> {
    for record in take_unseen(records, state)? {
        target
            .write_cognitive(state.clone(), record.entry.text.clone())
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
        assert!(take_unseen(&records, &mut state).unwrap().is_empty());
    }
}
