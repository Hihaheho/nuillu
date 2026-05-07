use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::sync::Mutex;

use nuillu_types::{
    MemoryIndex, ModelTier, ModuleId, ModuleInstanceId, ReplicaCapRange, ReplicaIndex, TokenBudget,
    builtin,
};
use tokio::sync::{RwLock, oneshot};

use crate::{
    AttentionStream, AttentionStreamRecord, AttentionStreamSet, BlackboardCommand, MemoryMetadata,
    ResourceAllocation,
};

/// The non-cognitive blackboard plus the cognitive surface and its
/// allocation snapshot. This is a cheap cloneable handle; locking is an
/// implementation detail hidden behind its methods.
#[derive(Debug, Clone)]
pub struct Blackboard {
    inner: Arc<RwLock<BlackboardInner>>,
    activation_waiters: Arc<Mutex<Vec<ActivationWaiter>>>,
}

/// Inner blackboard state. Public so read closures in other crates can
/// inspect it, but its fields are private and mutations stay behind
/// [`BlackboardCommand`].
#[derive(Debug, Default)]
pub struct BlackboardInner {
    memos: HashMap<ModuleInstanceId, String>,
    attention_streams: HashMap<ModuleInstanceId, AttentionStream>,
    memory_metadata: HashMap<MemoryIndex, MemoryMetadata>,
    base_allocation: ResourceAllocation,
    allocation: ResourceAllocation,
    allocation_proposals: HashMap<ModuleInstanceId, ResourceAllocation>,
    replica_caps: HashMap<ModuleId, ReplicaCapRange>,
}

struct ActivationWaiter {
    owner: ModuleInstanceId,
    sender: oneshot::Sender<()>,
}

impl std::fmt::Debug for ActivationWaiter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ActivationWaiter")
            .field("owner", &self.owner)
            .finish_non_exhaustive()
    }
}

impl Blackboard {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(RwLock::new(BlackboardInner::default())),
            activation_waiters: Arc::new(Mutex::new(Vec::new())),
        }
    }

    pub fn with_allocation(allocation: ResourceAllocation) -> Self {
        Self {
            inner: Arc::new(RwLock::new(BlackboardInner {
                base_allocation: allocation.clone(),
                allocation,
                ..BlackboardInner::default()
            })),
            activation_waiters: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Apply `f` to a borrowed snapshot. The read lock is held for the
    /// duration of `f`; do not await inside it.
    pub async fn read<R>(&self, f: impl FnOnce(&BlackboardInner) -> R) -> R {
        let guard = self.inner.read().await;
        f(&guard)
    }

    /// Apply one command under the blackboard write lock.
    pub async fn apply(&self, cmd: BlackboardCommand) {
        let mut guard = self.inner.write().await;
        guard.apply(cmd);
        self.notify_active_waiters(&guard.allocation);
    }

    pub async fn memo(&self, id: &ModuleId) -> Option<String> {
        self.read(|bb| bb.memo(id).map(String::from)).await
    }

    pub async fn memo_for_instance(&self, id: &ModuleInstanceId) -> Option<String> {
        self.read(|bb| bb.memo_for_instance(id).map(String::from))
            .await
    }

    /// Register a one-shot notification for the next time `owner` is active.
    ///
    /// Returns `None` when the owner is already active in the current
    /// allocation. The inactive check and waiter registration share the
    /// blackboard lock ordering used by [`apply`](Self::apply), so allocation
    /// updates cannot be missed between the check and registration.
    pub async fn activation_waiter(
        &self,
        owner: ModuleInstanceId,
    ) -> Option<oneshot::Receiver<()>> {
        let guard = self.inner.read().await;
        if guard.is_instance_active(&owner) {
            return None;
        }

        let mut waiters = self
            .activation_waiters
            .lock()
            .expect("activation waiters poisoned");
        waiters.retain(|waiter| !waiter.sender.is_closed());
        let (sender, receiver) = oneshot::channel();
        waiters.push(ActivationWaiter { owner, sender });
        Some(receiver)
    }

    fn notify_active_waiters(&self, allocation: &ResourceAllocation) {
        let mut waiters = self
            .activation_waiters
            .lock()
            .expect("activation waiters poisoned");
        let mut pending = Vec::with_capacity(waiters.len());
        for waiter in waiters.drain(..) {
            if allocation.is_replica_active(&waiter.owner) {
                let _ = waiter.sender.send(());
            } else if !waiter.sender.is_closed() {
                pending.push(waiter);
            }
        }
        *waiters = pending;
    }
}

impl Default for Blackboard {
    fn default() -> Self {
        Self::new()
    }
}

impl BlackboardInner {
    pub fn memo(&self, id: &ModuleId) -> Option<&str> {
        if let Some((_, memo)) = self
            .memos
            .iter()
            .find(|(owner, _)| &owner.module == id && owner.replica == ReplicaIndex::ZERO)
        {
            return Some(memo.as_str());
        }
        let mut matching = self
            .memos
            .iter()
            .filter(|(owner, _)| &owner.module == id)
            .map(|(_, memo)| memo.as_str());
        let first = matching.next()?;
        if matching.next().is_none() {
            Some(first)
        } else {
            None
        }
    }

    pub fn memo_for_instance(&self, id: &ModuleInstanceId) -> Option<&str> {
        self.memos.get(id).map(String::as_str)
    }

    pub fn memos(&self) -> serde_json::Value {
        let mut grouped = std::collections::BTreeMap::<String, Vec<(u8, String)>>::new();
        for (owner, memo) in &self.memos {
            grouped
                .entry(owner.module.as_str().to_owned())
                .or_default()
                .push((owner.replica.get(), memo.clone()));
        }

        let mut object = serde_json::Map::new();
        for (module, mut entries) in grouped {
            entries.sort_by_key(|(replica, _)| *replica);
            if entries.len() == 1 {
                object.insert(module, serde_json::Value::String(entries.remove(0).1));
            } else {
                object.insert(
                    module,
                    serde_json::Value::Array(
                        entries
                            .into_iter()
                            .map(|(replica, memo)| {
                                serde_json::json!({
                                    "replica": replica,
                                    "memo": memo,
                                })
                            })
                            .collect(),
                    ),
                );
            }
        }
        serde_json::Value::Object(object)
    }

    pub fn attention_stream(&self) -> AttentionStream {
        let mut entries = self
            .attention_streams
            .values()
            .flat_map(|stream| stream.entries().iter().cloned())
            .collect::<Vec<_>>();
        entries.sort_by_key(|event| event.at);
        let mut stream = AttentionStream::default();
        for event in entries {
            stream.append(event);
        }
        stream
    }

    pub fn attention_stream_set(&self) -> AttentionStreamSet {
        let mut records = self
            .attention_streams
            .iter()
            .map(|(owner, stream)| AttentionStreamRecord {
                stream: owner.clone(),
                entries: stream.entries().to_vec(),
            })
            .collect::<Vec<_>>();
        records.sort_by(|a, b| {
            a.stream
                .module
                .as_str()
                .cmp(b.stream.module.as_str())
                .then_with(|| a.stream.replica.cmp(&b.stream.replica))
        });
        AttentionStreamSet::new(records)
    }

    pub fn memory_metadata(&self) -> &HashMap<MemoryIndex, MemoryMetadata> {
        &self.memory_metadata
    }

    pub fn allocation(&self) -> &ResourceAllocation {
        &self.allocation
    }

    pub fn base_allocation(&self) -> &ResourceAllocation {
        &self.base_allocation
    }

    pub fn allocation_proposals(&self) -> &HashMap<ModuleInstanceId, ResourceAllocation> {
        &self.allocation_proposals
    }

    pub fn replica_caps(&self) -> &HashMap<ModuleId, ReplicaCapRange> {
        &self.replica_caps
    }

    fn is_instance_active(&self, owner: &ModuleInstanceId) -> bool {
        self.allocation.is_replica_active(owner)
    }

    /// Apply one command. Mutations are localised to the matching arm so
    /// adding a variant is a compile error in any consumer that pattern-
    /// matches.
    fn apply(&mut self, cmd: BlackboardCommand) {
        match cmd {
            BlackboardCommand::UpdateMemo { owner, memo } => {
                self.memos.insert(owner, memo);
            }
            BlackboardCommand::AppendAttentionStream { stream, event } => {
                self.attention_streams
                    .entry(stream)
                    .or_default()
                    .append(event);
            }
            BlackboardCommand::UpsertMemoryMetadata {
                index,
                rank_if_new,
                decay_if_new_secs,
                now,
                patch,
            } => {
                let entry = self
                    .memory_metadata
                    .entry(index.clone())
                    .or_insert_with(|| {
                        MemoryMetadata::new_at(index, rank_if_new, decay_if_new_secs, now)
                    });
                patch.apply_at(entry, now);
            }
            BlackboardCommand::RemoveMemoryMetadata { index } => {
                self.memory_metadata.remove(&index);
            }
            BlackboardCommand::SetAllocation(alloc) => {
                self.base_allocation = alloc;
                self.recompute_effective_allocation();
            }
            BlackboardCommand::SetReplicaCaps { caps } => {
                for (module, range) in caps {
                    self.replica_caps.insert(module, range);
                }
                self.recompute_effective_allocation();
            }
            BlackboardCommand::RecordAllocationProposal {
                controller,
                proposal,
            } => {
                self.allocation_proposals.insert(controller, proposal);
                self.recompute_effective_allocation();
            }
        }
    }

    fn recompute_effective_allocation(&mut self) {
        let active_controller_replicas = self
            .allocation
            .for_module(&builtin::attention_controller())
            .replicas;
        let active_proposals = self
            .allocation_proposals
            .iter()
            .filter(|(owner, _)| {
                owner.module == builtin::attention_controller()
                    && owner.replica.get() < active_controller_replicas
            })
            .map(|(_, proposal)| proposal)
            .collect::<Vec<_>>();

        if active_proposals.is_empty() {
            self.allocation = self.base_allocation.clone().clamped(&self.replica_caps);
            return;
        }

        let module_ids: HashSet<ModuleId> = self
            .replica_caps
            .keys()
            .cloned()
            .chain(self.base_allocation.iter().map(|(id, _)| id.clone()))
            .collect();

        let mut effective = ResourceAllocation::default();
        for id in module_ids {
            let configs = active_proposals
                .iter()
                .map(|proposal| {
                    let mut cfg = proposal
                        .get(&id)
                        .cloned()
                        .unwrap_or_else(|| self.base_allocation.for_module(&id));
                    if let Some(range) = self.replica_caps.get(&id) {
                        cfg.replicas = range.clamp(cfg.replicas);
                    }
                    cfg
                })
                .collect::<Vec<_>>();
            let count = configs.len() as u32;
            let replica_sum = configs.iter().map(|cfg| cfg.replicas as u32).sum::<u32>();
            let tier_sum = configs
                .iter()
                .map(|cfg| tier_to_ordinal(cfg.tier))
                .sum::<u32>();
            let budget_sum = configs.iter().map(|cfg| cfg.context_budget.0).sum::<u32>();

            let none_periods = configs.iter().filter(|cfg| cfg.period.is_none()).count();
            let positive_periods = configs
                .iter()
                .filter_map(|cfg| cfg.period)
                .filter(|period| !period.is_zero())
                .collect::<Vec<_>>();
            // Ties keep the positive average; only a strict majority of `None`
            // proposals turns the module back into message-only activation.
            let period = if none_periods * 2 > configs.len() || positive_periods.is_empty() {
                None
            } else {
                let millis = positive_periods
                    .iter()
                    .map(|period| period.as_millis())
                    .sum::<u128>()
                    / positive_periods.len() as u128;
                Some(std::time::Duration::from_millis(millis as u64))
            };

            let mut cfg = crate::ModuleConfig {
                replicas: rounded_div(replica_sum, count) as u8,
                tier: ordinal_to_tier(rounded_div(tier_sum, count)),
                period,
                context_budget: TokenBudget::new(rounded_div(budget_sum, count)),
            };
            if let Some(range) = self.replica_caps.get(&id) {
                cfg.replicas = range.clamp(cfg.replicas);
            }
            effective.set(id, cfg);
        }

        self.allocation = effective.clamped(&self.replica_caps);
    }
}

fn rounded_div(sum: u32, count: u32) -> u32 {
    if count == 0 {
        return 0;
    }
    (sum + count / 2) / count
}

fn tier_to_ordinal(tier: ModelTier) -> u32 {
    match tier {
        ModelTier::Cheap => 0,
        ModelTier::Default => 1,
        ModelTier::Premium => 2,
    }
}

fn ordinal_to_tier(ordinal: u32) -> ModelTier {
    match ordinal {
        0 => ModelTier::Cheap,
        1 => ModelTier::Default,
        2 => ModelTier::Premium,
        other => unreachable!("model tier ordinal {other} is outside the known tier range"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    use nuillu_types::{ModelTier, ModuleId, ReplicaIndex, TokenBudget, builtin};

    #[tokio::test]
    async fn memo_round_trip() {
        let bb = Blackboard::new();
        let id = builtin::summarize();
        bb.apply(BlackboardCommand::UpdateMemo {
            owner: ModuleInstanceId::new(id.clone(), ReplicaIndex::ZERO),
            memo: "noted".into(),
        })
        .await;
        assert_eq!(bb.memo(&id).await.as_deref(), Some("noted"));
    }

    #[tokio::test]
    async fn allocation_proposals_clamp_each_proposal_before_averaging() {
        let mut base = ResourceAllocation::default();
        base.set(
            builtin::attention_controller(),
            crate::ModuleConfig {
                replicas: 2,
                ..Default::default()
            },
        );
        let bb = Blackboard::with_allocation(base);
        bb.apply(BlackboardCommand::SetReplicaCaps {
            caps: vec![
                (
                    builtin::attention_controller(),
                    ReplicaCapRange { min: 1, max: 2 },
                ),
                (builtin::query_vector(), ReplicaCapRange { min: 0, max: 3 }),
                (builtin::speak(), ReplicaCapRange { min: 0, max: 1 }),
            ],
        })
        .await;

        let mut proposal_a = ResourceAllocation::default();
        proposal_a.set(
            builtin::query_vector(),
            crate::ModuleConfig {
                replicas: 1,
                tier: ModelTier::Cheap,
                period: Some(Duration::from_millis(10)),
                context_budget: TokenBudget::new(1000),
            },
        );
        proposal_a.set(
            builtin::speak(),
            crate::ModuleConfig {
                replicas: 0,
                ..Default::default()
            },
        );

        let mut proposal_b = ResourceAllocation::default();
        proposal_b.set(
            builtin::query_vector(),
            crate::ModuleConfig {
                replicas: 5,
                tier: ModelTier::Premium,
                period: Some(Duration::from_millis(30)),
                context_budget: TokenBudget::new(3000),
            },
        );
        proposal_b.set(
            builtin::speak(),
            crate::ModuleConfig {
                replicas: 1,
                ..Default::default()
            },
        );

        bb.apply(BlackboardCommand::RecordAllocationProposal {
            controller: ModuleInstanceId::new(
                builtin::attention_controller(),
                ReplicaIndex::new(0),
            ),
            proposal: proposal_a,
        })
        .await;
        bb.apply(BlackboardCommand::RecordAllocationProposal {
            controller: ModuleInstanceId::new(
                builtin::attention_controller(),
                ReplicaIndex::new(1),
            ),
            proposal: proposal_b,
        })
        .await;

        let effective = bb.read(|bb| bb.allocation().clone()).await;
        let query = effective.for_module(&builtin::query_vector());
        assert_eq!(query.replicas, 2);
        assert_eq!(query.tier, ModelTier::Default);
        assert_eq!(query.period, Some(Duration::from_millis(20)));
        assert_eq!(query.context_budget, TokenBudget::new(2000));
        assert_eq!(effective.for_module(&builtin::speak()).replicas, 1);
    }

    #[tokio::test]
    async fn allocation_ignores_proposals_from_inactive_controller_replicas() {
        let mut base = ResourceAllocation::default();
        base.set(
            builtin::attention_controller(),
            crate::ModuleConfig {
                replicas: 1,
                ..Default::default()
            },
        );
        base.set(
            builtin::speak(),
            crate::ModuleConfig {
                replicas: 0,
                ..Default::default()
            },
        );

        let bb = Blackboard::with_allocation(base);
        bb.apply(BlackboardCommand::SetReplicaCaps {
            caps: vec![
                (
                    builtin::attention_controller(),
                    ReplicaCapRange { min: 1, max: 2 },
                ),
                (builtin::speak(), ReplicaCapRange { min: 0, max: 1 }),
            ],
        })
        .await;

        let mut active = ResourceAllocation::default();
        active.set(
            builtin::speak(),
            crate::ModuleConfig {
                replicas: 0,
                ..Default::default()
            },
        );

        let mut inactive = ResourceAllocation::default();
        inactive.set(
            builtin::speak(),
            crate::ModuleConfig {
                replicas: 1,
                ..Default::default()
            },
        );

        bb.apply(BlackboardCommand::RecordAllocationProposal {
            controller: ModuleInstanceId::new(
                builtin::attention_controller(),
                ReplicaIndex::new(0),
            ),
            proposal: active,
        })
        .await;
        bb.apply(BlackboardCommand::RecordAllocationProposal {
            controller: ModuleInstanceId::new(
                builtin::attention_controller(),
                ReplicaIndex::new(1),
            ),
            proposal: inactive,
        })
        .await;

        let effective = bb.read(|bb| bb.allocation().clone()).await;
        assert_eq!(effective.for_module(&builtin::speak()).replicas, 0);
    }

    #[tokio::test]
    async fn allocation_proposals_do_not_add_unregistered_modules() {
        let mut base = ResourceAllocation::default();
        base.set(
            builtin::attention_controller(),
            crate::ModuleConfig {
                replicas: 1,
                ..Default::default()
            },
        );
        let bb = Blackboard::with_allocation(base);
        bb.apply(BlackboardCommand::SetReplicaCaps {
            caps: vec![(
                builtin::attention_controller(),
                ReplicaCapRange { min: 1, max: 1 },
            )],
        })
        .await;

        let unknown = ModuleId::new("invented-module").unwrap();
        let mut proposal = ResourceAllocation::default();
        proposal.set(
            unknown.clone(),
            crate::ModuleConfig {
                replicas: 1,
                tier: ModelTier::Premium,
                period: Some(Duration::from_millis(10)),
                context_budget: TokenBudget::new(1234),
            },
        );

        bb.apply(BlackboardCommand::RecordAllocationProposal {
            controller: ModuleInstanceId::new(builtin::attention_controller(), ReplicaIndex::ZERO),
            proposal,
        })
        .await;

        let effective = bb.read(|bb| bb.allocation().clone()).await;
        assert!(effective.get(&unknown).is_none());
    }
}
