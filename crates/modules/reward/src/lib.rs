//! Reward / policy domain crate.
//!
//! Owns the [`PolicyStore`] port, the policy-domain capability handles, and
//! the modules that operate on policy: [`PolicyModule`] (read-only
//! advisor/proposer), [`RewardModule`] (consolidates and reinforces), and
//! [`PolicyCompactionModule`] (conservative duplicate cleanup).
//!
//! Hosts build a [`PolicyCapabilities`] provider once at boot to bundle the
//! store + blackboard + clock, then pass it to registration closures and
//! call [`PolicyCapabilities::bootstrap_core_policies`] before
//! [`ModuleRegistry::build`] runs.
//!
//! [`ModuleRegistry::build`]: nuillu_module::ModuleRegistry::build

use std::cell::RefCell;
use std::collections::{HashMap, VecDeque};
use std::rc::Rc;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use futures::StreamExt;
use futures::channel::mpsc;
use nuillu_blackboard::{Blackboard, BlackboardCommand, CorePolicyRecord};
use nuillu_module::ports::{Clock, PortError};
use nuillu_types::{ModuleInstanceId, PolicyIndex, PolicyRank, SignedUnitF32, UnitF32};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

mod compaction;
mod policy;
mod reward;

pub use compaction::{
    CompactDuplicatePoliciesArgs, CompactDuplicatePoliciesOutput, GetPoliciesArgs,
    GetPoliciesOutput, PolicyCompactionModule, PolicyCompactionTools, PolicyCompactionToolsCall,
    PolicyCompactionToolsSelector, PolicyContentView, SearchPoliciesArgs, SearchPoliciesOutput,
};
pub use policy::{
    ExistingPolicyConsideration, PolicyConsideration, PolicyConsiderationDecision,
    PolicyConsiderationPayload, PolicyConsiderationSource, PolicyConsiderationWriter, PolicyModule,
    SyntheticPolicyConsideration,
};
pub use reward::{
    InsertPolicyCandidateDecision, ObservedReward, PolicyCandidateCredit, PolicyRewardSkip,
    PolicyRewardUpdate, PolicyUpserter, ReinforceExistingPolicyDecision, RewardAssessment,
    RewardMemo, RewardModule, SyntheticPolicyDedupDecision,
};

// ---------------------------------------------------------------------------
// Policy store port + types

/// Content-store for policy entries. Metadata (rank, value, confidence,
/// reward-token counts, decay, reward reinforcement counts) is mirrored on
/// the blackboard; this trait owns durable trigger/behavior content and
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

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
pub struct PolicyConsiderationKey {
    pub owner: ModuleInstanceId,
    pub memo_index: u64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PolicyConsiderationEvicted {
    pub key: PolicyConsiderationKey,
    pub written_at: DateTime<Utc>,
    pub payload: PolicyConsiderationPayload,
}

#[derive(Clone)]
struct PolicyConsiderationLog {
    inner: Rc<RefCell<PolicyConsiderationLogInner>>,
}

#[derive(Default)]
struct PolicyConsiderationLogInner {
    retained_per_owner: usize,
    next_indices: HashMap<ModuleInstanceId, u64>,
    records: HashMap<ModuleInstanceId, VecDeque<PolicyConsiderationLogRecord>>,
    subscribers: Vec<mpsc::UnboundedSender<PolicyConsiderationEvicted>>,
}

#[derive(Clone, Debug, PartialEq)]
struct PolicyConsiderationLogRecord {
    key: PolicyConsiderationKey,
    written_at: DateTime<Utc>,
    payload: PolicyConsiderationPayload,
}

impl PolicyConsiderationLog {
    fn new(retained_per_owner: usize) -> Self {
        Self {
            inner: Rc::new(RefCell::new(PolicyConsiderationLogInner {
                retained_per_owner: retained_per_owner.max(1),
                ..PolicyConsiderationLogInner::default()
            })),
        }
    }

    fn append(
        &self,
        owner: ModuleInstanceId,
        payload: PolicyConsiderationPayload,
        written_at: DateTime<Utc>,
    ) -> PolicyConsiderationLogRecord {
        let mut inner = self.inner.borrow_mut();
        let index = inner.next_indices.entry(owner.clone()).or_default();
        let record = PolicyConsiderationLogRecord {
            key: PolicyConsiderationKey {
                owner: owner.clone(),
                memo_index: *index,
            },
            written_at,
            payload,
        };
        *index = (*index).saturating_add(1);

        let retained = inner.retained_per_owner.max(1);
        let records = inner.records.entry(owner).or_default();
        records.push_back(record.clone());
        let mut evicted = Vec::new();
        while records.len() > retained {
            if let Some(record) = records.pop_front() {
                evicted.push(PolicyConsiderationEvicted {
                    key: record.key,
                    written_at: record.written_at,
                    payload: record.payload,
                });
            }
        }
        drop(inner);

        for evicted in evicted {
            let delivered = self.publish(evicted);
            if delivered == 0 {
                tracing::trace!("policy-consideration eviction had no active subscribers");
            }
        }
        record
    }

    fn subscribe(&self) -> PolicyConsiderationEvictedInbox {
        let (sender, receiver) = mpsc::unbounded();
        self.inner.borrow_mut().subscribers.push(sender);
        PolicyConsiderationEvictedInbox { receiver }
    }

    fn publish(&self, evicted: PolicyConsiderationEvicted) -> usize {
        let mut inner = self.inner.borrow_mut();
        let subscribers = &mut inner.subscribers;
        subscribers.retain(|sender| !sender.is_closed());
        let mut delivered = 0;
        for sender in subscribers.iter() {
            if sender.unbounded_send(evicted.clone()).is_ok() {
                delivered += 1;
            }
        }
        delivered
    }
}

pub struct PolicyConsiderationEvictedInbox {
    receiver: mpsc::UnboundedReceiver<PolicyConsiderationEvicted>,
}

impl PolicyConsiderationEvictedInbox {
    pub async fn next_item(&mut self) -> Option<PolicyConsiderationEvicted> {
        self.receiver.next().await
    }

    pub fn take_ready_items(&mut self) -> Vec<PolicyConsiderationEvicted> {
        let mut items = Vec::new();
        loop {
            match self.receiver.try_recv() {
                Ok(item) => items.push(item),
                Err(mpsc::TryRecvError::Empty) | Err(mpsc::TryRecvError::Closed) => return items,
            }
        }
    }
}

/// Trigger-similarity search over the primary policy store.
#[derive(Clone)]
pub struct PolicySearcher {
    primary_store: Rc<dyn PolicyStore>,
}

impl PolicySearcher {
    pub(crate) fn new(primary_store: Rc<dyn PolicyStore>) -> Self {
        Self { primary_store }
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
        Ok(hits)
    }
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

pub(crate) async fn fanout_policy_put(replicas: &[Rc<dyn PolicyStore>], indexed: IndexedPolicy) {
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

pub(crate) async fn fanout_policy_delete(replicas: &[Rc<dyn PolicyStore>], index: &PolicyIndex) {
    let replica_deletes = replicas
        .iter()
        .enumerate()
        .map(|(replica, store)| async move {
            if let Err(error) = store.delete(index).await {
                tracing::warn!(replica, ?error, "secondary policy delete failed");
            }
        });
    futures::future::join_all(replica_deletes).await;
}

/// Conservative duplicate-removal capability for policy maintenance.
#[derive(Clone)]
pub struct PolicyCompactor {
    primary_store: Rc<dyn PolicyStore>,
    replicas: Vec<Rc<dyn PolicyStore>>,
    blackboard: Blackboard,
}

impl PolicyCompactor {
    pub(crate) fn new(
        primary_store: Rc<dyn PolicyStore>,
        replicas: Vec<Rc<dyn PolicyStore>>,
        blackboard: Blackboard,
    ) -> Self {
        Self {
            primary_store,
            replicas,
            blackboard,
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
        Ok(filter_live_policy_hits(hits))
    }

    pub async fn get(&self, index: &PolicyIndex) -> Result<Option<PolicyRecord>, PortError> {
        self.primary_store.get(index).await
    }

    pub async fn get_many(&self, indexes: &[PolicyIndex]) -> Result<Vec<PolicyRecord>, PortError> {
        let mut out = Vec::new();
        for index in indexes {
            if let Some(record) = self.get(index).await? {
                out.push(record);
            }
        }
        Ok(out)
    }

    pub async fn list_compaction_candidates(&self) -> Result<Vec<PolicyRecord>, PortError> {
        let mut out = Vec::new();
        for rank in [
            PolicyRank::Tentative,
            PolicyRank::Provisional,
            PolicyRank::Established,
            PolicyRank::Habit,
            PolicyRank::Core,
        ] {
            out.extend(
                self.primary_store
                    .list_by_rank(rank)
                    .await?
                    .into_iter()
                    .filter(|record| {
                        record.rank == PolicyRank::Core || record.decay_remaining_secs > 0
                    }),
            );
        }
        out.sort_by(|left, right| left.index.as_str().cmp(right.index.as_str()));
        Ok(out)
    }

    pub async fn compact_duplicates(
        &self,
        canonical: &PolicyIndex,
        duplicates: &[PolicyIndex],
    ) -> Result<PolicyCompactionResult, PortError> {
        let canonical_record = self
            .primary_store
            .get(canonical)
            .await?
            .ok_or_else(|| PortError::NotFound(canonical.as_str().to_owned()))?;
        let mut deleted = Vec::new();
        let mut skipped = Vec::new();
        let mut seen = std::collections::HashSet::new();

        for duplicate in duplicates {
            if !seen.insert(duplicate.clone()) {
                skipped.push(PolicyCompactionSkipped {
                    index: duplicate.clone(),
                    reason: "duplicate index was listed more than once".to_owned(),
                });
                continue;
            }
            if duplicate == canonical {
                skipped.push(PolicyCompactionSkipped {
                    index: duplicate.clone(),
                    reason: "duplicate index matched the canonical policy".to_owned(),
                });
                continue;
            }
            let Some(record) = self.primary_store.get(duplicate).await? else {
                skipped.push(PolicyCompactionSkipped {
                    index: duplicate.clone(),
                    reason: "policy was not found".to_owned(),
                });
                continue;
            };
            if record.rank == PolicyRank::Core {
                skipped.push(PolicyCompactionSkipped {
                    index: duplicate.clone(),
                    reason: "core policies cannot be deleted by policy-compaction".to_owned(),
                });
                continue;
            }
            if policy_rank_strength(record.rank) > policy_rank_strength(canonical_record.rank) {
                skipped.push(PolicyCompactionSkipped {
                    index: duplicate.clone(),
                    reason: "duplicate policy outranks the canonical policy".to_owned(),
                });
                continue;
            }
            self.primary_store.delete(duplicate).await?;
            fanout_policy_delete(&self.replicas, duplicate).await;
            self.blackboard
                .apply(BlackboardCommand::RemovePolicyMetadata {
                    index: duplicate.clone(),
                })
                .await;
            deleted.push(duplicate.clone());
        }

        Ok(PolicyCompactionResult {
            canonical: canonical_record,
            deleted,
            skipped,
        })
    }
}

fn policy_rank_strength(rank: PolicyRank) -> u8 {
    match rank {
        PolicyRank::Tentative => 0,
        PolicyRank::Provisional => 1,
        PolicyRank::Established => 2,
        PolicyRank::Habit => 3,
        PolicyRank::Core => 4,
    }
}

fn filter_live_policy_hits(hits: Vec<PolicySearchHit>) -> Vec<PolicySearchHit> {
    hits.into_iter()
        .filter(|hit| hit.policy.rank == PolicyRank::Core || hit.policy.decay_remaining_secs > 0)
        .collect()
}

#[derive(Clone, Debug)]
pub struct PolicyCompactionResult {
    pub canonical: PolicyRecord,
    pub deleted: Vec<PolicyIndex>,
    pub skipped: Vec<PolicyCompactionSkipped>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PolicyCompactionSkipped {
    pub index: PolicyIndex,
    pub reason: String,
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
    primary_store: Rc<dyn PolicyStore>,
    replicas: Vec<Rc<dyn PolicyStore>>,
    blackboard: Blackboard,
    clock: Rc<dyn Clock>,
    considerations: PolicyConsiderationLog,
}

impl PolicyCapabilities {
    const DEFAULT_CONSIDERATION_RETAINED_PER_OWNER: usize = 8;

    pub fn new(
        blackboard: Blackboard,
        clock: Rc<dyn Clock>,
        primary_store: Rc<dyn PolicyStore>,
        replicas: Vec<Rc<dyn PolicyStore>>,
    ) -> Self {
        Self::new_with_consideration_retention(
            blackboard,
            clock,
            primary_store,
            replicas,
            Self::DEFAULT_CONSIDERATION_RETAINED_PER_OWNER,
        )
    }

    pub fn new_with_consideration_retention(
        blackboard: Blackboard,
        clock: Rc<dyn Clock>,
        primary_store: Rc<dyn PolicyStore>,
        replicas: Vec<Rc<dyn PolicyStore>>,
        consideration_retained_per_owner: usize,
    ) -> Self {
        Self {
            primary_store,
            replicas,
            blackboard,
            clock,
            considerations: PolicyConsiderationLog::new(consideration_retained_per_owner),
        }
    }

    pub fn searcher(&self) -> PolicySearcher {
        PolicySearcher::new(self.primary_store.clone())
    }

    pub fn consideration_writer(&self, owner: ModuleInstanceId) -> PolicyConsiderationWriter {
        PolicyConsiderationWriter::new(owner, self.considerations.clone(), self.clock.clone())
    }

    pub fn consideration_evicted_inbox(&self) -> PolicyConsiderationEvictedInbox {
        self.considerations.subscribe()
    }

    pub fn upserter(&self) -> PolicyUpserter {
        PolicyUpserter::new(
            self.primary_store.clone(),
            self.replicas.clone(),
            self.blackboard.clone(),
            self.clock.clone(),
        )
    }

    pub fn compactor(&self) -> PolicyCompactor {
        PolicyCompactor::new(
            self.primary_store.clone(),
            self.replicas.clone(),
            self.blackboard.clone(),
        )
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
