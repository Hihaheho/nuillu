use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;
use nuillu_blackboard::{Blackboard, BlackboardCommand, PolicyMetaPatch};
use nuillu_module::ports::{Clock, PortError};
use nuillu_module::{
    AllocationReader, AllocationUpdatedInbox, BlackboardReader, CognitionLogUpdatedInbox,
    LlmAccess, Module, TypedMemo,
};
use nuillu_types::{ModuleId, PolicyIndex, PolicyRank};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::{PolicyQuery, PolicySearchHit, PolicyStore};

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

/// Trigger-similarity search over the primary policy store with diagnostic
/// access metadata mirrored to the blackboard.
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
        let hits = hits
            .into_iter()
            .filter(|hit| {
                hit.policy.rank == PolicyRank::Core || hit.policy.decay_remaining_secs > 0
            })
            .collect::<Vec<_>>();
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

pub struct QueryPolicyModule {
    owner: ModuleId,
    allocation_updates: AllocationUpdatedInbox,
    cognition_updates: CognitionLogUpdatedInbox,
    allocation: AllocationReader,
    blackboard: BlackboardReader,
    policy_searcher: PolicySearcher,
    memo: TypedMemo<PolicyRetrievalMemo>,
    _llm: LlmAccess,
}

impl QueryPolicyModule {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        allocation_updates: AllocationUpdatedInbox,
        cognition_updates: CognitionLogUpdatedInbox,
        allocation: AllocationReader,
        blackboard: BlackboardReader,
        policy_searcher: PolicySearcher,
        memo: TypedMemo<PolicyRetrievalMemo>,
        llm: LlmAccess,
    ) -> Self {
        Self {
            owner: ModuleId::new(<Self as Module>::id()).expect("query-policy id is valid"),
            allocation_updates,
            cognition_updates,
            allocation,
            blackboard,
            policy_searcher,
            memo,
            _llm: llm,
        }
    }

    async fn build_query(&self) -> (String, String) {
        let guidance = self
            .allocation
            .snapshot()
            .await
            .for_module(&self.owner)
            .guidance
            .trim()
            .to_owned();
        if !guidance.is_empty() {
            return (guidance.clone(), guidance);
        }

        let latest = self
            .blackboard
            .read(|bb| {
                bb.cognition_log()
                    .entries()
                    .last()
                    .map(|entry| entry.text.clone())
                    .unwrap_or_else(|| "current cognition context".to_owned())
            })
            .await;
        (format!("Find policies applicable to: {latest}"), latest)
    }

    async fn activate(&mut self) -> Result<()> {
        let (query_text, request_context) = self.build_query().await;
        let hits = self.policy_searcher.search(&query_text, 8).await?;
        if hits.is_empty() {
            return Ok(());
        }

        let hits = hits
            .into_iter()
            .map(|hit| PolicyRetrievalHit {
                policy_index: hit.policy.index,
                similarity: hit.similarity,
                rank: hit.policy.rank,
                trigger: hit.policy.trigger,
                behavior: hit.policy.behavior,
                expected_reward: hit.policy.expected_reward.get(),
                confidence: hit.policy.confidence.get(),
                value: hit.policy.value.get(),
                reward_tokens: hit.policy.reward_tokens,
            })
            .collect::<Vec<_>>();
        let payload = PolicyRetrievalMemo {
            request_context,
            query_text,
            hits,
        };
        let plaintext = render_retrieval_memo(&payload);
        self.memo.write(payload, plaintext).await;
        Ok(())
    }
}

fn render_retrieval_memo(memo: &PolicyRetrievalMemo) -> String {
    let mut output = format!(
        "Policy retrieval window\nRequest context: {}\nQuery text: {}",
        memo.request_context.trim(),
        memo.query_text.trim(),
    );
    for hit in &memo.hits {
        output.push_str("\nPolicy hit [");
        output.push_str(hit.policy_index.as_str());
        output.push_str("]\nTrigger: ");
        output.push_str(hit.trigger.trim());
        output.push_str("\nBehavior: ");
        output.push_str(hit.behavior.trim());
        output.push_str(&format!(
            "\nRank: {:?}; similarity: {:.3}; expected_reward: {:.3}; confidence: {:.3}; value: {:.3}; reward_tokens: {}",
            hit.rank, hit.similarity, hit.expected_reward, hit.confidence, hit.value, hit.reward_tokens
        ));
    }
    output
}

#[async_trait(?Send)]
impl Module for QueryPolicyModule {
    type Batch = ();

    fn id() -> &'static str {
        "query-policy"
    }

    fn role_description() -> &'static str {
        "Retrieves policy records by trigger-only similarity and writes typed policy retrieval windows."
    }

    async fn next_batch(&mut self) -> Result<Self::Batch> {
        tokio::select! {
            result = self.allocation_updates.next_item() => {
                result?;
            }
            result = self.cognition_updates.next_item() => {
                result?;
            }
        }
        Ok(())
    }

    async fn activate(
        &mut self,
        _cx: &nuillu_module::ActivateCx<'_>,
        _batch: &Self::Batch,
    ) -> Result<()> {
        QueryPolicyModule::activate(self).await
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use async_trait::async_trait;
    use nuillu_blackboard::Blackboard;
    use nuillu_module::ports::{PortError, SystemClock};
    use nuillu_types::{SignedUnitF32, UnitF32};

    use super::*;
    use crate::{IndexedPolicy, NewPolicy, PolicyRecord};

    #[derive(Default)]
    struct RecordingPolicyStore {
        records: Mutex<Vec<PolicyRecord>>,
    }

    impl RecordingPolicyStore {
        fn seed(&self, record: PolicyRecord) {
            self.records
                .lock()
                .expect("policy store mutex poisoned")
                .push(record);
        }
    }

    #[async_trait(?Send)]
    impl PolicyStore for RecordingPolicyStore {
        async fn insert(&self, _policy: NewPolicy) -> Result<PolicyIndex, PortError> {
            Ok(PolicyIndex::new("unused"))
        }

        async fn put(&self, _policy: IndexedPolicy) -> Result<(), PortError> {
            Ok(())
        }

        async fn get(&self, _index: &PolicyIndex) -> Result<Option<PolicyRecord>, PortError> {
            Ok(None)
        }

        async fn list_by_rank(&self, _rank: PolicyRank) -> Result<Vec<PolicyRecord>, PortError> {
            Ok(Vec::new())
        }

        async fn search(&self, _q: &PolicyQuery) -> Result<Vec<PolicySearchHit>, PortError> {
            let mut records = self
                .records
                .lock()
                .expect("policy store mutex poisoned")
                .clone();
            records.sort_by(|left, right| left.index.as_str().cmp(right.index.as_str()));
            Ok(records
                .into_iter()
                .map(|policy| PolicySearchHit {
                    policy,
                    similarity: 1.0,
                })
                .collect())
        }

        async fn reinforce(
            &self,
            _index: &PolicyIndex,
            _value_delta: f32,
            _reward_tokens_delta: u32,
            _expected_reward_delta: f32,
            _confidence_delta: f32,
        ) -> Result<PolicyRecord, PortError> {
            Err(PortError::Backend("unused".into()))
        }

        async fn delete(&self, _index: &PolicyIndex) -> Result<(), PortError> {
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
    async fn policy_searcher_filters_expired_non_core_records() {
        let blackboard = Blackboard::default();
        let primary = Arc::new(RecordingPolicyStore::default());
        primary.seed(PolicyRecord {
            index: PolicyIndex::new("expired"),
            decay_remaining_secs: 0,
            ..test_record("expired")
        });
        primary.seed(PolicyRecord {
            index: PolicyIndex::new("active"),
            decay_remaining_secs: 60,
            ..test_record("active")
        });
        primary.seed(PolicyRecord {
            index: PolicyIndex::new("core"),
            rank: PolicyRank::Core,
            decay_remaining_secs: 0,
            ..test_record("core")
        });
        let searcher = PolicySearcher::new(primary, blackboard.clone(), Arc::new(SystemClock));

        let hits = searcher.search("alpha", 8).await.unwrap();

        assert_eq!(
            hits.iter()
                .map(|hit| hit.policy.index.as_str())
                .collect::<Vec<_>>(),
            vec!["active", "core"]
        );
        let metadata = blackboard.read(|bb| bb.policy_metadata().clone()).await;
        assert!(metadata.contains_key(&PolicyIndex::new("active")));
        assert!(metadata.contains_key(&PolicyIndex::new("core")));
        assert!(!metadata.contains_key(&PolicyIndex::new("expired")));
    }
}
