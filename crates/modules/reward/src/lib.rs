//! Reward / policy domain crate.
//!
//! Owns the [`PolicyStore`] port, the four policy-domain capability handles
//! ([`PolicyWriter`], [`PolicySearcher`], [`PolicyValueUpdater`],
//! [`PolicyWindowReader`]), and the four modules that operate on policy:
//! [`PolicyModule`] (proposes), [`QueryPolicyModule`] (retrieves),
//! [`ValueEstimatorModule`] (predicts), and [`RewardModule`] (reinforces).
//!
//! Hosts build a [`PolicyCapabilities`] provider once at boot to bundle the
//! store + blackboard + clock, then pass it to registration closures and
//! call [`PolicyCapabilities::bootstrap_core_policies`] before
//! [`ModuleRegistry::build`] runs.
//!
//! [`ModuleRegistry::build`]: nuillu_module::ModuleRegistry::build

use std::sync::Arc;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use nuillu_blackboard::{Blackboard, BlackboardCommand, CorePolicyRecord};
use nuillu_module::ports::{Clock, PortError};
use nuillu_types::{ModuleInstanceId, PolicyIndex, PolicyRank, SignedUnitF32, UnitF32, builtin};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

mod estimation;
mod policy;
mod query;
mod reward;

pub use estimation::{ValueEstimateMemo, ValueEstimatePrediction, ValueEstimatorModule};
pub use policy::{PolicyCandidate, PolicyFormationDecision, PolicyModule, PolicyWriter};
pub use query::{PolicyRetrievalHit, PolicyRetrievalMemo, PolicySearcher, QueryPolicyModule};
pub use reward::{
    NovelPolicyRequest, ObservedReward, PolicyCredit, PolicyValueUpdater, RewardAssessment,
    RewardMemo, RewardModule,
};

// ---------------------------------------------------------------------------
// Policy store port + types

/// Content-store for policy entries. Metadata (rank, value, confidence,
/// reward-token counts, decay, diagnostic access) is mirrored on the
/// blackboard; this trait owns durable trigger/behavior content and
/// adapter-local trigger embedding/search state.
#[async_trait(?Send)]
pub trait PolicyStore {
    async fn insert(&self, policy: NewPolicy) -> Result<PolicyIndex, PortError>;
    async fn put(&self, policy: IndexedPolicy) -> Result<(), PortError>;
    async fn get(&self, index: &PolicyIndex) -> Result<Option<PolicyRecord>, PortError>;
    async fn list_by_rank(&self, rank: PolicyRank) -> Result<Vec<PolicyRecord>, PortError>;
    async fn search(&self, q: &PolicyQuery) -> Result<Vec<PolicySearchHit>, PortError>;
    async fn reinforce(
        &self,
        index: &PolicyIndex,
        value_delta: f32,
        reward_tokens_delta: u32,
        expected_reward_delta: f32,
        confidence_delta: f32,
    ) -> Result<PolicyRecord, PortError>;
    async fn delete(&self, index: &PolicyIndex) -> Result<(), PortError>;
}

#[derive(Debug, Clone)]
pub struct NewPolicy {
    pub trigger: String,
    pub behavior: String,
    pub rank: PolicyRank,
    pub expected_reward: SignedUnitF32,
    pub confidence: UnitF32,
    pub value: SignedUnitF32,
    pub reward_tokens: u32,
    pub decay_remaining_secs: i64,
}

#[derive(Debug, Clone)]
pub struct IndexedPolicy {
    pub index: PolicyIndex,
    pub trigger: String,
    pub behavior: String,
    pub rank: PolicyRank,
    pub expected_reward: SignedUnitF32,
    pub confidence: UnitF32,
    pub value: SignedUnitF32,
    pub reward_tokens: u32,
    pub decay_remaining_secs: i64,
}

#[derive(Debug, Clone)]
pub struct PolicyRecord {
    pub index: PolicyIndex,
    pub trigger: String,
    pub behavior: String,
    pub rank: PolicyRank,
    pub expected_reward: SignedUnitF32,
    pub confidence: UnitF32,
    pub value: SignedUnitF32,
    pub reward_tokens: u32,
    pub decay_remaining_secs: i64,
}

#[derive(Debug, Clone)]
pub struct PolicySearchHit {
    pub policy: PolicyRecord,
    pub similarity: f32,
}

#[derive(Debug, Clone)]
pub struct PolicyQuery {
    pub trigger: String,
    pub limit: usize,
}

/// Policy store that accepts every write, returns empty reads, and assigns a
/// unique synthetic [`PolicyIndex`] to each insert.
#[derive(Debug, Default)]
pub struct NoopPolicyStore;

#[async_trait(?Send)]
impl PolicyStore for NoopPolicyStore {
    async fn insert(&self, _policy: NewPolicy) -> Result<PolicyIndex, PortError> {
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let id = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Ok(PolicyIndex::new(format!("noop-policy-{id}")))
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
        Ok(PolicyRecord {
            index: index.clone(),
            trigger: String::new(),
            behavior: String::new(),
            rank: PolicyRank::Tentative,
            expected_reward: SignedUnitF32::clamp(expected_reward_delta),
            confidence: UnitF32::clamp(confidence_delta),
            value: SignedUnitF32::clamp(value_delta),
            reward_tokens: reward_tokens_delta,
            decay_remaining_secs: 0,
        })
    }

    async fn delete(&self, _index: &PolicyIndex) -> Result<(), PortError> {
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Cross-cutting capability + provider

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
pub struct PolicyWindowKey {
    pub owner: ModuleInstanceId,
    pub memo_index: u64,
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

/// Reads typed policy windows out of the blackboard memo log. Spans both
/// query-policy and value-estimator memos because reward, value-estimator
/// and downstream consumers all need cross-module memo aggregation.
#[derive(Clone)]
pub struct PolicyWindowReader {
    blackboard: Blackboard,
}

impl PolicyWindowReader {
    pub fn new(blackboard: Blackboard) -> Self {
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

pub(crate) async fn fanout_policy_put(replicas: &[Arc<dyn PolicyStore>], indexed: IndexedPolicy) {
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

pub(crate) fn core_policy_record(record: PolicyRecord) -> CorePolicyRecord {
    CorePolicyRecord {
        index: record.index,
        trigger: record.trigger,
        behavior: record.behavior,
    }
}

/// Domain-scoped capability provider for the reward / policy subsystem.
///
/// Bundles the primary policy store, optional replica stores, the
/// blackboard, and the clock. Constructs the four policy-domain capability
/// handles on demand and seeds boot-time core policies.
#[derive(Clone)]
pub struct PolicyCapabilities {
    primary_store: Arc<dyn PolicyStore>,
    replicas: Vec<Arc<dyn PolicyStore>>,
    blackboard: Blackboard,
    clock: Arc<dyn Clock>,
}

impl PolicyCapabilities {
    pub fn new(
        blackboard: Blackboard,
        clock: Arc<dyn Clock>,
        primary_store: Arc<dyn PolicyStore>,
        replicas: Vec<Arc<dyn PolicyStore>>,
    ) -> Self {
        Self {
            primary_store,
            replicas,
            blackboard,
            clock,
        }
    }

    pub fn writer(&self) -> PolicyWriter {
        PolicyWriter::new(
            self.primary_store.clone(),
            self.replicas.clone(),
            self.blackboard.clone(),
        )
    }

    pub fn searcher(&self) -> PolicySearcher {
        PolicySearcher::new(
            self.primary_store.clone(),
            self.blackboard.clone(),
            self.clock.clone(),
        )
    }

    pub fn value_updater(&self) -> PolicyValueUpdater {
        PolicyValueUpdater::new(
            self.primary_store.clone(),
            self.replicas.clone(),
            self.blackboard.clone(),
            self.clock.clone(),
        )
    }

    pub fn window_reader(&self) -> PolicyWindowReader {
        PolicyWindowReader::new(self.blackboard.clone())
    }

    /// Seed `Core` rank policies onto the blackboard from the primary
    /// store. Hosts call this before `ModuleRegistry::build`.
    pub async fn bootstrap_core_policies(&self) -> Result<(), PortError> {
        let mut records = self
            .primary_store
            .list_by_rank(PolicyRank::Core)
            .await?
            .into_iter()
            .map(core_policy_record)
            .collect::<Vec<_>>();
        records.sort_by(|a, b| a.index.as_str().cmp(b.index.as_str()));
        self.blackboard
            .apply(BlackboardCommand::SetCorePolicies(records))
            .await;
        Ok(())
    }
}
