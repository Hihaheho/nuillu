use std::collections::HashMap;
use std::sync::Arc;

use chrono::{DateTime, Utc};
use nuillu_blackboard::{Blackboard, BlackboardCommand, CorePolicyRecord, PolicyMetaPatch};
use nuillu_types::{ModuleInstanceId, PolicyIndex, PolicyRank, SignedUnitF32, UnitF32, builtin};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::ports::Clock;
use crate::ports::{
    IndexedPolicy, NewPolicy, PolicyQuery, PolicyRecord, PolicySearchHit, PolicyStore, PortError,
};

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
pub struct PolicyWindowKey {
    pub owner: ModuleInstanceId,
    pub memo_index: u64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct PolicyRetrievalMemo {
    pub request_context: String,
    pub query_text: String,
    pub hits: Vec<PolicyRetrievalHit>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct PolicyRetrievalHit {
    pub policy_index: PolicyIndex,
    pub similarity: f32,
    pub rank: PolicyRank,
    pub trigger: String,
    pub behavior: String,
    pub expected_reward: f32,
    pub confidence: f32,
    pub value: f32,
    pub reward_tokens: u32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ValueEstimateMemo {
    pub retrieval_window: PolicyWindowKey,
    pub predictions: Vec<ValueEstimatePrediction>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ValueEstimatePrediction {
    pub policy_index: PolicyIndex,
    pub predicted_expected_reward: f32,
    pub confidence_hint: f32,
    pub rationale: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ObservedReward {
    pub external: f32,
    pub task: f32,
    pub social: f32,
    pub cost: f32,
    pub risk: f32,
    pub novelty: f32,
}

impl ObservedReward {
    pub fn scalar_default(&self) -> f32 {
        (self.external + self.task + self.social + self.novelty - self.cost - self.risk)
            .clamp(-1.0, 1.0)
    }
}

#[derive(Clone)]
pub struct PolicyWriter {
    primary_store: Arc<dyn PolicyStore>,
    replicas: Vec<Arc<dyn PolicyStore>>,
    blackboard: Blackboard,
}

impl PolicyWriter {
    pub(crate) fn new(
        primary_store: Arc<dyn PolicyStore>,
        replicas: Vec<Arc<dyn PolicyStore>>,
        blackboard: Blackboard,
    ) -> Self {
        Self {
            primary_store,
            replicas,
            blackboard,
        }
    }

    pub async fn insert(
        &self,
        trigger: String,
        behavior: String,
        decay_secs: i64,
    ) -> Result<PolicyIndex, PortError> {
        let new = NewPolicy {
            trigger: trigger.clone(),
            behavior: behavior.clone(),
            rank: PolicyRank::Tentative,
            expected_reward: SignedUnitF32::ZERO,
            confidence: UnitF32::ZERO,
            value: SignedUnitF32::ZERO,
            reward_tokens: 0,
            decay_remaining_secs: decay_secs,
        };
        let index = self.primary_store.insert(new).await?;
        let indexed = IndexedPolicy {
            index: index.clone(),
            trigger,
            behavior,
            rank: PolicyRank::Tentative,
            expected_reward: SignedUnitF32::ZERO,
            confidence: UnitF32::ZERO,
            value: SignedUnitF32::ZERO,
            reward_tokens: 0,
            decay_remaining_secs: decay_secs,
        };
        fanout_policy_put(&self.replicas, indexed).await;
        self.blackboard
            .apply(BlackboardCommand::UpsertPolicyMetadata {
                index: index.clone(),
                rank_if_new: PolicyRank::Tentative,
                decay_if_new_secs: decay_secs,
                patch: PolicyMetaPatch {
                    rank: Some(PolicyRank::Tentative),
                    expected_reward: Some(SignedUnitF32::ZERO),
                    confidence: Some(UnitF32::ZERO),
                    value: Some(SignedUnitF32::ZERO),
                    reward_tokens: Some(0),
                    decay_remaining_secs: Some(decay_secs),
                    ..Default::default()
                },
            })
            .await;
        Ok(index)
    }
}

#[derive(Clone)]
pub struct PolicySearcher {
    primary_store: Arc<dyn PolicyStore>,
    blackboard: Blackboard,
    clock: Arc<dyn Clock>,
}

impl PolicySearcher {
    pub(crate) fn new(
        primary_store: Arc<dyn PolicyStore>,
        blackboard: Blackboard,
        clock: Arc<dyn Clock>,
    ) -> Self {
        Self {
            primary_store,
            blackboard,
            clock,
        }
    }

    pub async fn search(
        &self,
        trigger: &str,
        limit: usize,
    ) -> Result<Vec<PolicySearchHit>, PortError> {
        let hits = self
            .primary_store
            .search(&PolicyQuery {
                trigger: trigger.to_owned(),
                limit,
            })
            .await?;
        let now = self.clock.now();
        for hit in &hits {
            self.blackboard
                .apply(BlackboardCommand::UpsertPolicyMetadata {
                    index: hit.policy.index.clone(),
                    rank_if_new: hit.policy.rank,
                    decay_if_new_secs: hit.policy.decay_remaining_secs,
                    patch: PolicyMetaPatch {
                        rank: Some(hit.policy.rank),
                        expected_reward: Some(hit.policy.expected_reward),
                        confidence: Some(hit.policy.confidence),
                        value: Some(hit.policy.value),
                        reward_tokens: Some(hit.policy.reward_tokens),
                        decay_remaining_secs: Some(hit.policy.decay_remaining_secs),
                        record_access_at: Some(now),
                        ..Default::default()
                    },
                })
                .await;
        }
        Ok(hits)
    }
}

#[derive(Clone)]
pub struct PolicyValueUpdater {
    primary_store: Arc<dyn PolicyStore>,
    replicas: Vec<Arc<dyn PolicyStore>>,
    blackboard: Blackboard,
    clock: Arc<dyn Clock>,
}

impl PolicyValueUpdater {
    pub(crate) fn new(
        primary_store: Arc<dyn PolicyStore>,
        replicas: Vec<Arc<dyn PolicyStore>>,
        blackboard: Blackboard,
        clock: Arc<dyn Clock>,
    ) -> Self {
        Self {
            primary_store,
            replicas,
            blackboard,
            clock,
        }
    }

    pub async fn reinforce(
        &self,
        index: &PolicyIndex,
        value_delta: f32,
        reward_tokens_delta: u32,
        expected_reward_delta: f32,
        confidence_delta: f32,
    ) -> Result<PolicyRecord, PortError> {
        let record = self
            .primary_store
            .reinforce(
                index,
                value_delta,
                reward_tokens_delta,
                expected_reward_delta,
                confidence_delta,
            )
            .await?;
        fanout_policy_put(&self.replicas, indexed_policy(&record)).await;
        self.blackboard
            .apply(BlackboardCommand::UpsertPolicyMetadata {
                index: record.index.clone(),
                rank_if_new: record.rank,
                decay_if_new_secs: record.decay_remaining_secs,
                patch: PolicyMetaPatch {
                    rank: Some(record.rank),
                    expected_reward: Some(record.expected_reward),
                    confidence: Some(record.confidence),
                    value: Some(record.value),
                    reward_tokens: Some(record.reward_tokens),
                    decay_remaining_secs: Some(record.decay_remaining_secs),
                    reinforced_at: Some(self.clock.now()),
                    ..Default::default()
                },
            })
            .await;
        Ok(record)
    }
}

#[derive(Clone)]
pub struct PolicyWindowReader {
    blackboard: Blackboard,
}

impl PolicyWindowReader {
    pub(crate) fn new(blackboard: Blackboard) -> Self {
        Self { blackboard }
    }

    pub async fn retrieval_windows(&self) -> Vec<TypedPolicyRetrievalWindow> {
        let owners = self.owners_for_module(builtin::query_policy()).await;
        let mut records = Vec::new();
        for owner in owners {
            for record in self
                .blackboard
                .typed_memo_logs::<PolicyRetrievalMemo>(&owner)
                .await
            {
                let payload = record.data().clone();
                records.push(TypedPolicyRetrievalWindow {
                    key: PolicyWindowKey {
                        owner: record.owner.clone(),
                        memo_index: record.index,
                    },
                    written_at: record.written_at,
                    plaintext: record.content,
                    payload,
                });
            }
        }
        records.sort_by(|a, b| {
            a.written_at
                .cmp(&b.written_at)
                .then(a.key.memo_index.cmp(&b.key.memo_index))
        });
        records
    }

    pub async fn value_estimates(&self) -> Vec<TypedValueEstimateWindow> {
        let owners = self.owners_for_module(builtin::value_estimator()).await;
        let mut records = Vec::new();
        for owner in owners {
            for record in self
                .blackboard
                .typed_memo_logs::<ValueEstimateMemo>(&owner)
                .await
            {
                let payload = record.data().clone();
                records.push(TypedValueEstimateWindow {
                    key: PolicyWindowKey {
                        owner: record.owner.clone(),
                        memo_index: record.index,
                    },
                    written_at: record.written_at,
                    plaintext: record.content,
                    payload,
                });
            }
        }
        records.sort_by(|a, b| {
            a.written_at
                .cmp(&b.written_at)
                .then(a.key.memo_index.cmp(&b.key.memo_index))
        });
        records
    }

    async fn owners_for_module(&self, module: nuillu_types::ModuleId) -> Vec<ModuleInstanceId> {
        let mut owners = self
            .blackboard
            .read(|bb| {
                bb.recent_memo_logs()
                    .into_iter()
                    .filter(|record| record.owner.module == module)
                    .map(|record| record.owner)
                    .collect::<Vec<_>>()
            })
            .await;
        owners.sort_by(|a, b| {
            a.module
                .as_str()
                .cmp(b.module.as_str())
                .then_with(|| a.replica.cmp(&b.replica))
        });
        owners.dedup();
        owners
    }

    pub async fn retrieval_by_key(&self, key: &PolicyWindowKey) -> Option<PolicyRetrievalMemo> {
        self.blackboard
            .typed_memo_logs::<PolicyRetrievalMemo>(&key.owner)
            .await
            .into_iter()
            .find(|record| record.index == key.memo_index)
            .map(|record| record.data().clone())
    }
}

#[derive(Clone, Debug)]
pub struct TypedPolicyRetrievalWindow {
    pub key: PolicyWindowKey,
    pub written_at: DateTime<Utc>,
    pub plaintext: String,
    pub payload: PolicyRetrievalMemo,
}

#[derive(Clone, Debug)]
pub struct TypedValueEstimateWindow {
    pub key: PolicyWindowKey,
    pub written_at: DateTime<Utc>,
    pub plaintext: String,
    pub payload: ValueEstimateMemo,
}

pub(crate) fn core_policy_record(record: PolicyRecord) -> CorePolicyRecord {
    CorePolicyRecord {
        index: record.index,
        trigger: record.trigger,
        behavior: record.behavior,
    }
}

fn indexed_policy(record: &PolicyRecord) -> IndexedPolicy {
    IndexedPolicy {
        index: record.index.clone(),
        trigger: record.trigger.clone(),
        behavior: record.behavior.clone(),
        rank: record.rank,
        expected_reward: record.expected_reward,
        confidence: record.confidence,
        value: record.value,
        reward_tokens: record.reward_tokens,
        decay_remaining_secs: record.decay_remaining_secs,
    }
}

async fn fanout_policy_put(replicas: &[Arc<dyn PolicyStore>], indexed: IndexedPolicy) {
    let replica_writes = replicas.iter().enumerate().map(|(replica, store)| {
        let indexed = indexed.clone();
        async move {
            if let Err(error) = store.put(indexed).await {
                tracing::warn!(replica, ?error, "secondary policy write failed");
            }
        }
    });
    futures::future::join_all(replica_writes).await;
}

#[allow(dead_code)]
fn _assert_hash_map_send_free(_: HashMap<PolicyIndex, PolicyRecord>) {}

#[cfg(test)]
mod tests {
    use super::*;

    use std::sync::Mutex;

    use async_trait::async_trait;
    use nuillu_blackboard::Blackboard;
    use nuillu_types::ReplicaIndex;

    use crate::ports::SystemClock;
    use crate::test_support::{scoped, test_caps};

    #[derive(Default)]
    struct RecordingPolicyStore {
        state: Mutex<RecordingPolicyState>,
    }

    #[derive(Default)]
    struct RecordingPolicyState {
        records: HashMap<String, PolicyRecord>,
        inserts: usize,
        puts: usize,
        reinforces: usize,
    }

    impl RecordingPolicyStore {
        fn seed(&self, record: PolicyRecord) {
            self.state
                .lock()
                .expect("policy store mutex poisoned")
                .records
                .insert(record.index.as_str().to_owned(), record);
        }

        fn counts(&self) -> (usize, usize, usize) {
            let state = self.state.lock().expect("policy store mutex poisoned");
            (state.inserts, state.puts, state.reinforces)
        }

        fn snapshot(&self, index: &PolicyIndex) -> Option<PolicyRecord> {
            self.state
                .lock()
                .expect("policy store mutex poisoned")
                .records
                .get(index.as_str())
                .cloned()
        }
    }

    #[async_trait(?Send)]
    impl PolicyStore for RecordingPolicyStore {
        async fn insert(&self, policy: NewPolicy) -> Result<PolicyIndex, PortError> {
            let mut state = self.state.lock().expect("policy store mutex poisoned");
            state.inserts += 1;
            let index = PolicyIndex::new(format!("policy-{}", state.inserts));
            state.records.insert(
                index.as_str().to_owned(),
                PolicyRecord {
                    index: index.clone(),
                    trigger: policy.trigger,
                    behavior: policy.behavior,
                    rank: policy.rank,
                    expected_reward: policy.expected_reward,
                    confidence: policy.confidence,
                    value: policy.value,
                    reward_tokens: policy.reward_tokens,
                    decay_remaining_secs: policy.decay_remaining_secs,
                },
            );
            Ok(index)
        }

        async fn put(&self, policy: IndexedPolicy) -> Result<(), PortError> {
            let mut state = self.state.lock().expect("policy store mutex poisoned");
            state.puts += 1;
            state.records.insert(
                policy.index.as_str().to_owned(),
                PolicyRecord {
                    index: policy.index,
                    trigger: policy.trigger,
                    behavior: policy.behavior,
                    rank: policy.rank,
                    expected_reward: policy.expected_reward,
                    confidence: policy.confidence,
                    value: policy.value,
                    reward_tokens: policy.reward_tokens,
                    decay_remaining_secs: policy.decay_remaining_secs,
                },
            );
            Ok(())
        }

        async fn get(&self, index: &PolicyIndex) -> Result<Option<PolicyRecord>, PortError> {
            Ok(self.snapshot(index))
        }

        async fn list_by_rank(&self, rank: PolicyRank) -> Result<Vec<PolicyRecord>, PortError> {
            Ok(self
                .state
                .lock()
                .expect("policy store mutex poisoned")
                .records
                .values()
                .filter(|record| record.rank == rank)
                .cloned()
                .collect())
        }

        async fn search(&self, _q: &PolicyQuery) -> Result<Vec<PolicySearchHit>, PortError> {
            Ok(Vec::new())
        }

        async fn reinforce(
            &self,
            index: &PolicyIndex,
            value_delta: f32,
            reward_tokens_delta: u32,
            expected_reward_delta: f32,
            confidence_delta: f32,
        ) -> Result<PolicyRecord, PortError> {
            let mut state = self.state.lock().expect("policy store mutex poisoned");
            state.reinforces += 1;
            let record = state
                .records
                .get_mut(index.as_str())
                .ok_or_else(|| PortError::NotFound(index.as_str().to_owned()))?;
            record.value = SignedUnitF32::clamp(record.value.get() + value_delta);
            record.expected_reward =
                SignedUnitF32::clamp(record.expected_reward.get() + expected_reward_delta);
            record.confidence = UnitF32::clamp(record.confidence.get() + confidence_delta);
            record.reward_tokens = record.reward_tokens.saturating_add(reward_tokens_delta);
            Ok(record.clone())
        }

        async fn delete(&self, index: &PolicyIndex) -> Result<(), PortError> {
            self.state
                .lock()
                .expect("policy store mutex poisoned")
                .records
                .remove(index.as_str());
            Ok(())
        }
    }

    fn test_record(index: &str) -> PolicyRecord {
        PolicyRecord {
            index: PolicyIndex::new(index),
            trigger: "alpha trigger".into(),
            behavior: "alpha behavior".into(),
            rank: PolicyRank::Tentative,
            expected_reward: SignedUnitF32::ZERO,
            confidence: UnitF32::ZERO,
            value: SignedUnitF32::ZERO,
            reward_tokens: 0,
            decay_remaining_secs: 60,
        }
    }

    #[tokio::test]
    async fn policy_writer_only_uses_insert_and_mirrors_metadata() {
        let blackboard = Blackboard::default();
        let primary = Arc::new(RecordingPolicyStore::default());
        let replica = Arc::new(RecordingPolicyStore::default());
        let writer = PolicyWriter::new(primary.clone(), vec![replica.clone()], blackboard.clone());

        let index = writer
            .insert("alpha trigger".into(), "alpha behavior".into(), 60)
            .await
            .unwrap();

        assert_eq!(primary.counts(), (1, 0, 0));
        assert_eq!(replica.counts(), (0, 1, 0));
        assert_eq!(
            primary.snapshot(&index).unwrap().rank,
            PolicyRank::Tentative
        );
        let mirrored = blackboard
            .read(|bb| bb.policy_metadata().get(&index).cloned())
            .await
            .unwrap();
        assert_eq!(mirrored.rank, PolicyRank::Tentative);
        assert_eq!(mirrored.reward_tokens, 0);
    }

    #[tokio::test]
    async fn policy_value_updater_reinforces_and_mirrors_returned_record() {
        let blackboard = Blackboard::default();
        let primary = Arc::new(RecordingPolicyStore::default());
        let replica = Arc::new(RecordingPolicyStore::default());
        let index = PolicyIndex::new("p1");
        primary.seed(test_record(index.as_str()));
        let updater = PolicyValueUpdater::new(
            primary.clone(),
            vec![replica.clone()],
            blackboard.clone(),
            Arc::new(SystemClock),
        );

        let record = updater.reinforce(&index, 0.5, 1, 0.25, 0.75).await.unwrap();

        assert_eq!(primary.counts(), (0, 0, 1));
        assert_eq!(replica.counts(), (0, 1, 0));
        assert_eq!(record.value.get(), 0.5);
        assert_eq!(record.expected_reward.get(), 0.25);
        assert_eq!(record.confidence.get(), 0.75);
        assert_eq!(record.reward_tokens, 1);
        let mirrored = blackboard
            .read(|bb| bb.policy_metadata().get(&index).cloned())
            .await
            .unwrap();
        assert_eq!(mirrored.value.get(), 0.5);
        assert_eq!(mirrored.expected_reward.get(), 0.25);
        assert_eq!(mirrored.confidence.get(), 0.75);
        assert_eq!(mirrored.reward_tokens, 1);
        assert!(mirrored.last_reinforced_at.is_some());
    }

    #[tokio::test]
    async fn policy_window_reader_reads_only_typed_policy_windows() {
        let blackboard = Blackboard::default();
        let caps = test_caps(blackboard.clone());
        let query_policy = scoped(&caps, builtin::query_policy(), 0);
        let value_estimator = scoped(&caps, builtin::value_estimator(), 0);
        let query_vector = scoped(&caps, builtin::query_vector(), 0);
        let retrieval_owner = ModuleInstanceId::new(builtin::query_policy(), ReplicaIndex::ZERO);
        let retrieval_payload = PolicyRetrievalMemo {
            request_context: "context".into(),
            query_text: "alpha".into(),
            hits: vec![PolicyRetrievalHit {
                policy_index: PolicyIndex::new("p1"),
                similarity: 0.9,
                rank: PolicyRank::Tentative,
                trigger: "alpha".into(),
                behavior: "do alpha".into(),
                expected_reward: 0.1,
                confidence: 0.2,
                value: 0.3,
                reward_tokens: 1,
            }],
        };

        query_policy
            .typed_memo::<PolicyRetrievalMemo>()
            .write(retrieval_payload.clone(), "policy plaintext")
            .await;
        query_vector
            .typed_memo::<PolicyRetrievalMemo>()
            .write(retrieval_payload, "wrong owner")
            .await;
        value_estimator
            .typed_memo::<ValueEstimateMemo>()
            .write(
                ValueEstimateMemo {
                    retrieval_window: PolicyWindowKey {
                        owner: retrieval_owner,
                        memo_index: 0,
                    },
                    predictions: Vec::new(),
                },
                "value plaintext",
            )
            .await;

        let reader = PolicyWindowReader::new(blackboard);
        let retrievals = reader.retrieval_windows().await;
        let values = reader.value_estimates().await;

        assert_eq!(retrievals.len(), 1);
        assert_eq!(retrievals[0].plaintext, "policy plaintext");
        assert_eq!(retrievals[0].payload.query_text, "alpha");
        assert_eq!(values.len(), 1);
        assert_eq!(values[0].plaintext, "value plaintext");
    }
}
