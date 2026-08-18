use std::collections::HashMap;
use std::rc::Rc;
use std::sync::Mutex;

use async_trait::async_trait;
use nuillu_module::ports::{Embedder, PortError};
use nuillu_reward::{
    IndexedPolicy, NewPolicy, PolicyQuery, PolicyRecord, PolicySearchHit, PolicyStore,
};
use nuillu_types::{PolicyIndex, PolicyRank, SignedUnitF32, UnitF32};
use uuid::Uuid;

#[derive(Debug, Default)]
struct PolicyState {
    records: HashMap<PolicyIndex, StoredPolicy>,
    next_sequence: u64,
}

#[derive(Debug, Clone)]
struct StoredPolicy {
    record: PolicyRecord,
    embedding: Vec<f32>,
    sequence: u64,
}

/// Embedding-backed policy store that keeps records and vectors in process memory.
///
/// Every trigger write and every search with a nonzero limit calls the configured
/// [`Embedder`]. No lexical-search fallback is used.
pub struct InMemoryPolicyStore {
    embedder: Rc<dyn Embedder>,
    dimensions: usize,
    state: Mutex<PolicyState>,
}

impl std::fmt::Debug for InMemoryPolicyStore {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("InMemoryPolicyStore")
            .field("dimensions", &self.dimensions)
            .field("state", &self.state)
            .finish_non_exhaustive()
    }
}

impl InMemoryPolicyStore {
    pub fn new(embedder: Rc<dyn Embedder>) -> Self {
        Self {
            dimensions: embedder.dimensions(),
            embedder,
            state: Mutex::new(PolicyState::default()),
        }
    }

    async fn embed(&self, text: &str) -> Result<Vec<f32>, PortError> {
        let embedding = self.embedder.embed(text).await?;
        crate::embedding::validate("policy", self.dimensions, embedding)
    }

    fn put_locked(state: &mut PolicyState, policy: IndexedPolicy, embedding: Vec<f32>) {
        let sequence = state.records.get(&policy.index).map_or_else(
            || {
                let sequence = state.next_sequence;
                state.next_sequence = state.next_sequence.saturating_add(1);
                sequence
            },
            |stored| stored.sequence,
        );
        let record = PolicyRecord {
            index: policy.index.clone(),
            trigger: policy.trigger,
            behavior: policy.behavior,
            rank: policy.rank,
            expected_reward: policy.expected_reward,
            confidence: policy.confidence,
            value: policy.value,
            reward_tokens: policy.reward_tokens,
            decay_remaining_secs: policy.decay_remaining_secs,
        };
        state.records.insert(
            policy.index,
            StoredPolicy {
                record,
                embedding,
                sequence,
            },
        );
    }
}

#[async_trait(?Send)]
impl PolicyStore for InMemoryPolicyStore {
    async fn insert(&self, policy: NewPolicy) -> Result<PolicyIndex, PortError> {
        let index = PolicyIndex::new(Uuid::now_v7().to_string());
        self.put(IndexedPolicy {
            index: index.clone(),
            trigger: policy.trigger,
            behavior: policy.behavior,
            rank: policy.rank,
            expected_reward: policy.expected_reward,
            confidence: policy.confidence,
            value: policy.value,
            reward_tokens: policy.reward_tokens,
            decay_remaining_secs: policy.decay_remaining_secs,
        })
        .await?;
        Ok(index)
    }

    async fn put(&self, policy: IndexedPolicy) -> Result<(), PortError> {
        let embedding = self.embed(&policy.trigger).await?;
        let mut state = self
            .state
            .lock()
            .map_err(|_| PortError::Backend("policy store lock poisoned".into()))?;
        Self::put_locked(&mut state, policy, embedding);
        Ok(())
    }

    async fn get(&self, index: &PolicyIndex) -> Result<Option<PolicyRecord>, PortError> {
        Ok(self
            .state
            .lock()
            .map_err(|_| PortError::Backend("policy store lock poisoned".into()))?
            .records
            .get(index)
            .map(|stored| stored.record.clone()))
    }

    async fn list_by_rank(&self, rank: PolicyRank) -> Result<Vec<PolicyRecord>, PortError> {
        let state = self
            .state
            .lock()
            .map_err(|_| PortError::Backend("policy store lock poisoned".into()))?;
        let mut records = state
            .records
            .values()
            .filter(|stored| stored.record.rank == rank)
            .map(|stored| stored.record.clone())
            .collect::<Vec<_>>();
        records.sort_by(|left, right| left.index.as_str().cmp(right.index.as_str()));
        Ok(records)
    }

    async fn search(&self, q: &PolicyQuery) -> Result<Vec<PolicySearchHit>, PortError> {
        if q.limit == 0 {
            return Ok(Vec::new());
        }
        let query_embedding = self.embed(&q.trigger).await?;
        let state = self
            .state
            .lock()
            .map_err(|_| PortError::Backend("policy store lock poisoned".into()))?;
        let mut matches = state
            .records
            .values()
            .filter(|stored| {
                stored.record.rank == PolicyRank::Core || stored.record.decay_remaining_secs > 0
            })
            .map(|stored| {
                (
                    crate::embedding::cosine_similarity(&query_embedding, &stored.embedding),
                    stored.sequence,
                    stored.record.clone(),
                )
            })
            .collect::<Vec<_>>();
        matches.sort_by(|left, right| {
            right
                .0
                .total_cmp(&left.0)
                .then_with(|| left.1.cmp(&right.1))
        });
        Ok(matches
            .into_iter()
            .take(q.limit)
            .map(|(similarity, _, policy)| PolicySearchHit { policy, similarity })
            .collect())
    }

    async fn reinforce(
        &self,
        index: &PolicyIndex,
        value_delta: f32,
        reward_tokens_delta: u32,
        expected_reward_delta: f32,
        confidence_delta: f32,
    ) -> Result<PolicyRecord, PortError> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| PortError::Backend("policy store lock poisoned".into()))?;
        let stored = state
            .records
            .get_mut(index)
            .ok_or_else(|| PortError::NotFound(index.to_string()))?;
        let record = &mut stored.record;
        record.value = SignedUnitF32::clamp(record.value.get() + value_delta);
        record.expected_reward =
            SignedUnitF32::clamp(record.expected_reward.get() + expected_reward_delta);
        record.confidence = UnitF32::clamp(record.confidence.get() + confidence_delta);
        record.reward_tokens = record.reward_tokens.saturating_add(reward_tokens_delta);
        record.rank = rank_after_reinforcement(
            record.rank,
            record.value,
            record.confidence,
            record.reward_tokens,
        );
        Ok(record.clone())
    }

    async fn delete(&self, index: &PolicyIndex) -> Result<(), PortError> {
        self.state
            .lock()
            .map_err(|_| PortError::Backend("policy store lock poisoned".into()))?
            .records
            .remove(index);
        Ok(())
    }
}

fn rank_after_reinforcement(
    current: PolicyRank,
    value: SignedUnitF32,
    confidence: UnitF32,
    reward_tokens: u32,
) -> PolicyRank {
    if current == PolicyRank::Core {
        PolicyRank::Core
    } else if reward_tokens >= 16 && value.get() >= 0.7 && confidence.get() >= 0.7 {
        PolicyRank::Habit
    } else if reward_tokens >= 8 && value.get() >= 0.45 && confidence.get() >= 0.5 {
        PolicyRank::Established
    } else if reward_tokens >= 2 && value.get() >= 0.2 {
        PolicyRank::Provisional
    } else {
        PolicyRank::Tentative
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embedding::TestEmbedder;

    fn indexed(index: &str, trigger: &str, rank: PolicyRank) -> IndexedPolicy {
        IndexedPolicy {
            index: PolicyIndex::new(index),
            trigger: trigger.into(),
            behavior: format!("do {trigger}"),
            rank,
            expected_reward: SignedUnitF32::ZERO,
            confidence: UnitF32::ZERO,
            value: SignedUnitF32::ZERO,
            reward_tokens: 0,
            decay_remaining_secs: 60,
        }
    }

    #[tokio::test(flavor = "current_thread")]
    async fn search_filters_expired_and_orders_by_similarity() {
        let store = InMemoryPolicyStore::new(Rc::new(TestEmbedder));
        store
            .put(indexed("alpha", "alpha launch", PolicyRank::Tentative))
            .await
            .unwrap();
        store
            .put(indexed("beta", "beta review", PolicyRank::Core))
            .await
            .unwrap();
        store
            .put(IndexedPolicy {
                decay_remaining_secs: 0,
                ..indexed("expired", "alpha launch", PolicyRank::Established)
            })
            .await
            .unwrap();

        let hits = store
            .search(&PolicyQuery {
                trigger: "alpha launch".into(),
                limit: 8,
            })
            .await
            .unwrap();
        assert_eq!(
            hits.iter()
                .map(|hit| hit.policy.index.as_str())
                .collect::<Vec<_>>(),
            vec!["alpha", "beta"]
        );
        assert_eq!(hits[0].similarity, 1.0);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn reinforce_clamps_values_and_promotes_rank() {
        let store = InMemoryPolicyStore::new(Rc::new(TestEmbedder));
        store
            .put(IndexedPolicy {
                confidence: UnitF32::clamp(0.9),
                value: SignedUnitF32::clamp(0.6),
                reward_tokens: 15,
                ..indexed("policy", "trigger", PolicyRank::Tentative)
            })
            .await
            .unwrap();

        let record = store
            .reinforce(&PolicyIndex::new("policy"), 1.0, 1, 2.0, 1.0)
            .await
            .unwrap();
        assert_eq!(record.value, SignedUnitF32::clamp(1.0));
        assert_eq!(record.expected_reward, SignedUnitF32::clamp(1.0));
        assert_eq!(record.confidence, UnitF32::ONE);
        assert_eq!(record.reward_tokens, 16);
        assert_eq!(record.rank, PolicyRank::Habit);
    }
}
