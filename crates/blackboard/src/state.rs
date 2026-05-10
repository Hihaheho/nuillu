use std::any::Any;
use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt;
use std::marker::PhantomData;
use std::sync::{Arc, Mutex, OnceLock};

use chrono::{DateTime, Utc};
use nuillu_types::{MemoryIndex, ModuleId, ModuleInstanceId, builtin};
use tokio::sync::{RwLock, oneshot};

use crate::{
    ActivationRatio, AgenticDeadlockMarker, AllocationLimits, BlackboardCommand, CognitionLog,
    CognitionLogEntryRecord, CognitionLogRecord, CognitionLogSet, IdentityMemoryRecord,
    MemoryMetadata, ModulePolicy, ResourceAllocation,
};

const DEFAULT_MEMO_RETAINED_PER_OWNER: usize = 8;

/// The non-cognitive blackboard plus the cognitive surface and its
/// allocation snapshot. This is a cheap cloneable handle; locking is an
/// implementation detail hidden behind its methods.
///
/// `module_catalog` lives outside the inner lock as a write-once
/// `OnceLock<Vec<(ModuleId, &'static str)>>`: it is populated by
/// `ModuleRegistry::build` before any module is constructed and never
/// changes afterwards, so module constructors can read it synchronously
/// without taking the async lock.
#[derive(Debug, Clone)]
pub struct Blackboard {
    inner: Arc<RwLock<BlackboardInner>>,
    activation_waiters: Arc<Mutex<Vec<ActivationWaiter>>>,
    activation_increase_waiters: Arc<Mutex<Vec<ActivationIncreaseWaiter>>>,
    module_catalog: Arc<OnceLock<Vec<(ModuleId, &'static str)>>>,
}

/// Inner blackboard state. Public so read closures in other crates can
/// inspect it, but its fields are private and mutations stay behind
/// [`BlackboardCommand`].
#[derive(Debug)]
pub struct BlackboardInner {
    memos: HashMap<ModuleInstanceId, VecDeque<MemoLogEntry>>,
    memo_next_indices: HashMap<ModuleInstanceId, u64>,
    memo_retained_per_owner: usize,
    module_statuses: HashMap<ModuleInstanceId, ModuleRunStatus>,
    utterance_progresses: HashMap<ModuleInstanceId, UtteranceProgress>,
    cognition_logs: HashMap<ModuleInstanceId, CognitionLog>,
    cognition_entry_log: Vec<CognitionLogEntryRecord>,
    cognition_next_index: u64,
    agentic_deadlock_marker: Option<AgenticDeadlockMarker>,
    memory_metadata: HashMap<MemoryIndex, MemoryMetadata>,
    identity_memories: Vec<IdentityMemoryRecord>,
    base_allocation: ResourceAllocation,
    allocation: ResourceAllocation,
    allocation_proposals: HashMap<ModuleInstanceId, ResourceAllocation>,
    module_policies: HashMap<ModuleId, ModulePolicy>,
    allocation_limits: AllocationLimits,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct MemoLogRecord {
    pub owner: ModuleInstanceId,
    pub index: u64,
    pub written_at: DateTime<Utc>,
    pub content: String,
}

#[derive(Clone)]
pub struct TypedMemoLogRecord<T> {
    pub owner: ModuleInstanceId,
    pub index: u64,
    pub written_at: DateTime<Utc>,
    pub content: String,
    payload: Arc<dyn Any>,
    _marker: PhantomData<fn() -> T>,
}

impl<T: 'static> TypedMemoLogRecord<T> {
    pub fn data(&self) -> &T {
        self.payload.downcast_ref::<T>().expect(
            "typed memo payload type mismatch: entries for this owner must be written through one TypedMemo<T> payload type",
        )
    }
}

struct MemoLogEntry {
    record: MemoLogRecord,
    payload: Arc<dyn Any>,
}

impl fmt::Debug for MemoLogEntry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MemoLogEntry")
            .field("record", &self.record)
            .finish_non_exhaustive()
    }
}

impl MemoLogEntry {
    fn new(record: MemoLogRecord, payload: Arc<dyn Any>) -> Self {
        Self { record, payload }
    }

    fn record(&self) -> MemoLogRecord {
        self.record.clone()
    }

    fn typed_record<T: 'static>(&self) -> TypedMemoLogRecord<T> {
        TypedMemoLogRecord {
            owner: self.record.owner.clone(),
            index: self.record.index,
            written_at: self.record.written_at,
            content: self.record.content.clone(),
            payload: Arc::clone(&self.payload),
            _marker: PhantomData,
        }
    }
}

#[derive(Debug, Default, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum ModuleRunStatus {
    #[default]
    Inactive,
    AwaitingBatch,
    PendingBatch,
    Activating,
    Failed {
        phase: String,
        message: String,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModuleRunStatusRecord {
    pub owner: ModuleInstanceId,
    pub status: ModuleRunStatus,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UtteranceProgressState {
    Streaming,
    Completed,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct UtteranceProgress {
    pub state: UtteranceProgressState,
    pub generation_id: u64,
    pub sequence: u32,
    pub target: String,
    pub partial_utterance: String,
    pub generation_hint: String,
    pub rationale: String,
}

impl UtteranceProgress {
    pub fn streaming(
        generation_id: u64,
        sequence: u32,
        target: impl Into<String>,
        partial_utterance: impl Into<String>,
        generation_hint: impl Into<String>,
        rationale: impl Into<String>,
    ) -> Self {
        Self {
            state: UtteranceProgressState::Streaming,
            generation_id,
            sequence,
            target: target.into(),
            partial_utterance: partial_utterance.into(),
            generation_hint: generation_hint.into(),
            rationale: rationale.into(),
        }
    }

    pub fn completed(
        generation_id: u64,
        sequence: u32,
        target: impl Into<String>,
        utterance: impl Into<String>,
        generation_hint: impl Into<String>,
        rationale: impl Into<String>,
    ) -> Self {
        Self {
            state: UtteranceProgressState::Completed,
            generation_id,
            sequence,
            target: target.into(),
            partial_utterance: utterance.into(),
            generation_hint: generation_hint.into(),
            rationale: rationale.into(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UtteranceProgressRecord {
    pub owner: ModuleInstanceId,
    pub progress: UtteranceProgress,
}

struct ActivationWaiter {
    owner: ModuleInstanceId,
    sender: oneshot::Sender<()>,
}

struct ActivationIncreaseWaiter {
    module: ModuleId,
    threshold: ActivationRatio,
    sender: oneshot::Sender<()>,
}

impl std::fmt::Debug for ActivationWaiter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ActivationWaiter")
            .field("owner", &self.owner)
            .finish_non_exhaustive()
    }
}

impl std::fmt::Debug for ActivationIncreaseWaiter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ActivationIncreaseWaiter")
            .field("module", &self.module)
            .field("threshold", &self.threshold)
            .finish_non_exhaustive()
    }
}

impl Blackboard {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(RwLock::new(BlackboardInner::default())),
            activation_waiters: Arc::new(Mutex::new(Vec::new())),
            activation_increase_waiters: Arc::new(Mutex::new(Vec::new())),
            module_catalog: Arc::new(OnceLock::new()),
        }
    }

    pub fn with_allocation(allocation: ResourceAllocation) -> Self {
        let mut inner = BlackboardInner {
            base_allocation: allocation,
            ..BlackboardInner::default()
        };
        inner.recompute_effective_allocation();
        Self {
            inner: Arc::new(RwLock::new(inner)),
            activation_waiters: Arc::new(Mutex::new(Vec::new())),
            activation_increase_waiters: Arc::new(Mutex::new(Vec::new())),
            module_catalog: Arc::new(OnceLock::new()),
        }
    }

    /// Install the registered-module catalog. Idempotent on first call;
    /// subsequent calls are silently ignored to keep the post-boot snapshot
    /// stable for prompt caching.
    pub fn set_module_catalog(&self, catalog: Vec<(ModuleId, &'static str)>) {
        let _ = self.module_catalog.set(catalog);
    }

    /// Read the registered-module catalog. Returns an empty slice before the
    /// catalog is installed, which lets tests that skip registry boot still
    /// build modules; the system prompt simply omits the peer list in that
    /// case.
    pub fn module_catalog(&self) -> &[(ModuleId, &'static str)] {
        self.module_catalog
            .get()
            .map(|v| v.as_slice())
            .unwrap_or(&[])
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
        self.notify_activation_increase_waiters(&guard.allocation);
    }

    pub async fn update_memo(
        &self,
        owner: ModuleInstanceId,
        memo: String,
        written_at: DateTime<Utc>,
    ) -> MemoLogRecord {
        self.update_typed_memo(owner, memo, (), written_at).await
    }

    pub async fn update_typed_memo<T: 'static>(
        &self,
        owner: ModuleInstanceId,
        memo: String,
        payload: T,
        written_at: DateTime<Utc>,
    ) -> MemoLogRecord {
        let mut guard = self.inner.write().await;
        guard.append_memo(owner, memo, Arc::new(payload), written_at)
    }

    pub async fn typed_memo_logs<T: 'static>(
        &self,
        owner: &ModuleInstanceId,
    ) -> Vec<TypedMemoLogRecord<T>> {
        self.read(|bb| bb.typed_memo_logs(owner)).await
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

    /// Register a one-shot notification for the next time `module`'s effective
    /// activation ratio strictly exceeds `threshold`.
    pub async fn activation_increase_waiter(
        &self,
        module: ModuleId,
        threshold: ActivationRatio,
    ) -> Option<oneshot::Receiver<()>> {
        let guard = self.inner.read().await;
        if guard.allocation.activation_for(&module) > threshold {
            return None;
        }

        let mut waiters = self
            .activation_increase_waiters
            .lock()
            .expect("activation increase waiters poisoned");
        waiters.retain(|waiter| !waiter.sender.is_closed());
        let (sender, receiver) = oneshot::channel();
        waiters.push(ActivationIncreaseWaiter {
            module,
            threshold,
            sender,
        });
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

    fn notify_activation_increase_waiters(&self, allocation: &ResourceAllocation) {
        let mut waiters = self
            .activation_increase_waiters
            .lock()
            .expect("activation increase waiters poisoned");
        let mut pending = Vec::with_capacity(waiters.len());
        for waiter in waiters.drain(..) {
            if allocation.activation_for(&waiter.module) > waiter.threshold {
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

impl Default for BlackboardInner {
    fn default() -> Self {
        Self {
            memos: HashMap::new(),
            memo_next_indices: HashMap::new(),
            memo_retained_per_owner: DEFAULT_MEMO_RETAINED_PER_OWNER,
            module_statuses: HashMap::new(),
            utterance_progresses: HashMap::new(),
            cognition_logs: HashMap::new(),
            cognition_entry_log: Vec::new(),
            cognition_next_index: 0,
            agentic_deadlock_marker: None,
            memory_metadata: HashMap::new(),
            identity_memories: Vec::new(),
            base_allocation: ResourceAllocation::default(),
            allocation: ResourceAllocation::default(),
            allocation_proposals: HashMap::new(),
            module_policies: HashMap::new(),
            allocation_limits: AllocationLimits::default(),
        }
    }
}

impl BlackboardInner {
    pub fn recent_memo_logs(&self) -> Vec<MemoLogRecord> {
        let mut records = self
            .memos
            .values()
            .flat_map(|records| records.iter().map(MemoLogEntry::record))
            .collect::<Vec<_>>();
        sort_memo_logs(&mut records);
        records
    }

    pub fn unread_memo_logs(
        &self,
        last_seen_indices: &HashMap<ModuleInstanceId, u64>,
    ) -> Vec<MemoLogRecord> {
        let mut records = self
            .memos
            .iter()
            .flat_map(|(owner, records)| {
                let last_seen = last_seen_indices.get(owner).copied();
                records.iter().filter(move |entry| {
                    last_seen.is_none_or(|last_seen| entry.record.index > last_seen)
                })
            })
            .map(MemoLogEntry::record)
            .collect::<Vec<_>>();
        sort_memo_logs(&mut records);
        records
    }

    pub fn typed_memo_logs<T: 'static>(
        &self,
        owner: &ModuleInstanceId,
    ) -> Vec<TypedMemoLogRecord<T>> {
        self.memos
            .get(owner)
            .into_iter()
            .flat_map(|records| records.iter().map(MemoLogEntry::typed_record))
            .collect()
    }

    pub fn module_status_for_instance(&self, id: &ModuleInstanceId) -> Option<&ModuleRunStatus> {
        self.module_statuses.get(id)
    }

    pub fn module_status_records(&self) -> Vec<ModuleRunStatusRecord> {
        let mut records = self
            .module_statuses
            .iter()
            .map(|(owner, status)| ModuleRunStatusRecord {
                owner: owner.clone(),
                status: status.clone(),
            })
            .collect::<Vec<_>>();
        records.sort_by(|a, b| {
            a.owner
                .module
                .as_str()
                .cmp(b.owner.module.as_str())
                .then_with(|| a.owner.replica.cmp(&b.owner.replica))
        });
        records
    }

    pub fn module_statuses(&self) -> serde_json::Value {
        let mut object = serde_json::Map::new();
        for record in self.module_status_records() {
            object.insert(record.owner.to_string(), serde_json::json!(record.status));
        }
        serde_json::Value::Object(object)
    }

    pub fn utterance_progress_for_instance(
        &self,
        id: &ModuleInstanceId,
    ) -> Option<&UtteranceProgress> {
        self.utterance_progresses.get(id)
    }

    pub fn utterance_progress_records(&self) -> Vec<UtteranceProgressRecord> {
        let mut records = self
            .utterance_progresses
            .iter()
            .map(|(owner, progress)| UtteranceProgressRecord {
                owner: owner.clone(),
                progress: progress.clone(),
            })
            .collect::<Vec<_>>();
        records.sort_by(|a, b| {
            a.owner
                .module
                .as_str()
                .cmp(b.owner.module.as_str())
                .then_with(|| a.owner.replica.cmp(&b.owner.replica))
        });
        records
    }

    pub fn utterance_progresses(&self) -> serde_json::Value {
        let mut object = serde_json::Map::new();
        for record in self.utterance_progress_records() {
            object.insert(record.owner.to_string(), serde_json::json!(record.progress));
        }
        serde_json::Value::Object(object)
    }

    pub fn memo_logs(&self) -> serde_json::Value {
        let mut grouped = std::collections::BTreeMap::<String, Vec<serde_json::Value>>::new();
        for record in self.recent_memo_logs() {
            grouped
                .entry(record.owner.module.as_str().to_owned())
                .or_default()
                .push(serde_json::json!({
                    "replica": record.owner.replica.get(),
                    "index": record.index,
                    "written_at": record.written_at,
                    "content": record.content,
                }));
        }
        serde_json::json!(grouped)
    }

    pub fn cognition_log(&self) -> CognitionLog {
        let mut entries = self
            .cognition_logs
            .values()
            .flat_map(|log| log.entries().iter().cloned())
            .collect::<Vec<_>>();
        entries.sort_by_key(|entry| entry.at);
        let mut log = CognitionLog::default();
        for entry in entries {
            log.append(entry);
        }
        log
    }

    pub fn unread_cognition_log_entries(
        &self,
        last_seen_index: Option<u64>,
    ) -> Vec<CognitionLogEntryRecord> {
        self.cognition_entry_log
            .iter()
            .filter(|record| last_seen_index.is_none_or(|index| record.index > index))
            .cloned()
            .collect()
    }

    pub fn cognition_log_set(&self) -> CognitionLogSet {
        let mut records = self
            .cognition_logs
            .iter()
            .map(|(owner, log)| CognitionLogRecord {
                source: owner.clone(),
                entries: log.entries().to_vec(),
            })
            .collect::<Vec<_>>();
        records.sort_by(|a, b| {
            a.source
                .module
                .as_str()
                .cmp(b.source.module.as_str())
                .then_with(|| a.source.replica.cmp(&b.source.replica))
        });
        CognitionLogSet::new(records, self.agentic_deadlock_marker.clone())
    }

    pub fn agentic_deadlock_marker(&self) -> Option<&AgenticDeadlockMarker> {
        self.agentic_deadlock_marker.as_ref()
    }

    pub fn memory_metadata(&self) -> &HashMap<MemoryIndex, MemoryMetadata> {
        &self.memory_metadata
    }

    pub fn identity_memories(&self) -> &[IdentityMemoryRecord] {
        &self.identity_memories
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

    pub fn module_policies(&self) -> &HashMap<ModuleId, ModulePolicy> {
        &self.module_policies
    }

    pub fn allocation_limits(&self) -> AllocationLimits {
        self.allocation_limits
    }

    pub fn memo_retained_per_owner(&self) -> usize {
        self.memo_retained_per_owner
    }

    fn is_instance_active(&self, owner: &ModuleInstanceId) -> bool {
        self.allocation.is_replica_active(owner)
    }

    /// Apply one command. Mutations are localised to the matching arm so
    /// adding a variant is a compile error in any consumer that pattern-
    /// matches.
    fn apply(&mut self, cmd: BlackboardCommand) {
        match cmd {
            BlackboardCommand::UpdateMemo {
                owner,
                memo,
                written_at,
            } => {
                self.append_memo(owner, memo, Arc::new(()), written_at);
            }
            BlackboardCommand::SetModuleRunStatus { owner, status } => {
                self.module_statuses.insert(owner, status);
            }
            BlackboardCommand::SetUtteranceProgress { owner, progress } => {
                self.utterance_progresses.insert(owner, progress);
            }
            BlackboardCommand::AppendCognitionLog { source, entry } => {
                self.cognition_entry_log.push(CognitionLogEntryRecord {
                    index: self.cognition_next_index,
                    source: source.clone(),
                    entry: entry.clone(),
                });
                self.cognition_next_index = self.cognition_next_index.saturating_add(1);
                self.cognition_logs.entry(source).or_default().append(entry);
            }
            BlackboardCommand::RecordAgenticDeadlockMarker(marker) => {
                self.agentic_deadlock_marker = Some(marker);
            }
            BlackboardCommand::UpsertMemoryMetadata {
                index,
                rank_if_new,
                occurred_at_if_new,
                decay_if_new_secs,
                now,
                patch,
            } => {
                let entry = self
                    .memory_metadata
                    .entry(index.clone())
                    .or_insert_with(|| {
                        MemoryMetadata::new_at(
                            index,
                            rank_if_new,
                            occurred_at_if_new,
                            decay_if_new_secs,
                            now,
                        )
                    });
                patch.apply_at(entry, now);
            }
            BlackboardCommand::RemoveMemoryMetadata { index } => {
                self.memory_metadata.remove(&index);
            }
            BlackboardCommand::SetIdentityMemories(records) => {
                self.identity_memories = records;
            }
            BlackboardCommand::SetAllocation(alloc) => {
                self.base_allocation = alloc;
                self.recompute_effective_allocation();
            }
            BlackboardCommand::SetModulePolicies { policies } => {
                for (module, policy) in policies {
                    self.module_policies.insert(module, policy);
                }
                self.recompute_effective_allocation();
            }
            BlackboardCommand::SetAllocationLimits(limits) => {
                self.allocation_limits = limits;
                self.recompute_effective_allocation();
            }
            BlackboardCommand::SetMemoRetentionPerOwner(retained) => {
                self.memo_retained_per_owner = retained.max(1);
                self.truncate_memos_to_retention();
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

    fn append_memo(
        &mut self,
        owner: ModuleInstanceId,
        content: String,
        payload: Arc<dyn Any>,
        written_at: DateTime<Utc>,
    ) -> MemoLogRecord {
        let index = self.memo_next_indices.entry(owner.clone()).or_default();
        let record = MemoLogRecord {
            owner: owner.clone(),
            index: *index,
            written_at,
            content,
        };
        *index = (*index).saturating_add(1);

        let retained = self.memo_retained_per_owner.max(1);
        let records = self.memos.entry(owner).or_default();
        records.push_back(MemoLogEntry::new(record.clone(), payload));
        while records.len() > retained {
            records.pop_front();
        }
        record
    }

    fn truncate_memos_to_retention(&mut self) {
        let retained = self.memo_retained_per_owner.max(1);
        for records in self.memos.values_mut() {
            while records.len() > retained {
                records.pop_front();
            }
        }
    }

    fn recompute_effective_allocation(&mut self) {
        let active_controller_replicas = self
            .allocation
            .active_replicas(&builtin::attention_controller());
        let mut active_proposals = self
            .allocation_proposals
            .iter()
            .filter(|(owner, _)| {
                owner.module == builtin::attention_controller()
                    && owner.replica.get() < active_controller_replicas
            })
            .collect::<Vec<_>>();
        active_proposals.sort_by_key(|(owner, _)| owner.replica);

        if active_proposals.is_empty() {
            self.allocation = self
                .base_allocation
                .clone()
                .derived(&self.module_policies)
                .limited(self.allocation_limits);
            return;
        }

        let module_ids: HashSet<ModuleId> = self
            .module_policies
            .keys()
            .cloned()
            .chain(self.base_allocation.iter().map(|(id, _)| id.clone()))
            .chain(
                self.base_allocation
                    .iter_activation()
                    .map(|(id, _)| id.clone()),
            )
            .collect();

        let mut effective = ResourceAllocation::default();
        // Carry host-set tier and activation-table state through unchanged.
        effective.set_activation_table(self.base_allocation.activation_table().to_vec());
        for (id, tier) in self.base_allocation.iter_model_override() {
            effective.set_model_override(id.clone(), tier);
        }
        for id in module_ids {
            let count = active_proposals.len() as u32;
            let activation_sum = active_proposals
                .iter()
                .map(|(_, proposal)| {
                    let ratio = if proposal.get(&id).is_some() {
                        proposal.activation_for(&id)
                    } else {
                        self.base_allocation.activation_for(&id)
                    };
                    u32::from(ratio.raw())
                })
                .sum::<u32>();
            let guidance =
                combine_guidance(active_proposals.iter().filter_map(|(owner, proposal)| {
                    proposal
                        .get(&id)
                        .map(|cfg| (owner.to_string(), cfg.guidance.trim().to_owned()))
                }));

            effective.set(id.clone(), crate::ModuleConfig { guidance });
            effective.set_activation(
                id,
                crate::ActivationRatio::from_raw(rounded_div(activation_sum, count) as u16),
            );
        }

        self.allocation = effective
            .derived(&self.module_policies)
            .limited(self.allocation_limits);
    }
}

fn sort_memo_logs(records: &mut [MemoLogRecord]) {
    records.sort_by(|a, b| {
        a.owner
            .module
            .as_str()
            .cmp(b.owner.module.as_str())
            .then_with(|| a.owner.replica.cmp(&b.owner.replica))
            .then_with(|| a.index.cmp(&b.index))
    });
}

fn rounded_div(sum: u32, count: u32) -> u32 {
    if count == 0 {
        return 0;
    }
    (sum + count / 2) / count
}

fn combine_guidance(items: impl IntoIterator<Item = (String, String)>) -> String {
    let mut items = items
        .into_iter()
        .filter(|(_, guidance)| !guidance.is_empty())
        .collect::<Vec<_>>();
    if items.len() == 1 {
        return items.remove(0).1;
    }
    items
        .into_iter()
        .map(|(owner, guidance)| format!("{owner}: {guidance}"))
        .collect::<Vec<_>>()
        .join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    use chrono::TimeZone;
    use nuillu_types::{
        MemoryContent, MemoryIndex, ModelTier, ModuleId, ReplicaCapRange, ReplicaIndex, builtin,
    };

    fn memo_time(seconds: i64) -> DateTime<Utc> {
        Utc.timestamp_opt(seconds, 0).unwrap()
    }

    #[tokio::test]
    async fn memo_round_trip() {
        let bb = Blackboard::new();
        let id = builtin::cognition_gate();
        let owner = ModuleInstanceId::new(id, ReplicaIndex::ZERO);
        bb.apply(BlackboardCommand::UpdateMemo {
            owner: owner.clone(),
            memo: "noted".into(),
            written_at: memo_time(0),
        })
        .await;
        let logs = bb.read(|bb| bb.recent_memo_logs()).await;
        assert_eq!(
            logs,
            vec![MemoLogRecord {
                owner,
                index: 0,
                written_at: memo_time(0),
                content: "noted".into(),
            }]
        );
    }

    #[tokio::test]
    async fn typed_memo_round_trip_keeps_plaintext_view() {
        #[derive(Debug, Clone, PartialEq, Eq)]
        struct TestPayload {
            value: String,
        }

        let bb = Blackboard::new();
        let owner = ModuleInstanceId::new(builtin::query_vector(), ReplicaIndex::ZERO);
        bb.update_typed_memo(
            owner.clone(),
            "plain memo".into(),
            TestPayload {
                value: "structured".into(),
            },
            memo_time(0),
        )
        .await;

        let logs = bb.read(|bb| bb.recent_memo_logs()).await;
        assert_eq!(
            logs,
            vec![MemoLogRecord {
                owner: owner.clone(),
                index: 0,
                written_at: memo_time(0),
                content: "plain memo".into(),
            }]
        );

        let typed_logs = bb.typed_memo_logs::<TestPayload>(&owner).await;
        assert_eq!(typed_logs.len(), 1);
        assert_eq!(typed_logs[0].content, "plain memo");
        assert_eq!(
            typed_logs[0].data(),
            &TestPayload {
                value: "structured".into()
            }
        );

        let json = bb.read(|bb| bb.memo_logs()).await;
        assert_eq!(
            json,
            serde_json::json!({
                "query-vector": [{
                    "replica": 0,
                    "index": 0,
                    "written_at": memo_time(0),
                    "content": "plain memo",
                }]
            })
        );
    }

    #[tokio::test]
    async fn identity_memories_replace_boot_snapshot() {
        let bb = Blackboard::new();
        let first = IdentityMemoryRecord {
            index: MemoryIndex::new("identity-1"),
            content: MemoryContent::new("first identity"),
            occurred_at: None,
        };
        let second = IdentityMemoryRecord {
            index: MemoryIndex::new("identity-2"),
            content: MemoryContent::new("second identity"),
            occurred_at: None,
        };

        bb.apply(BlackboardCommand::SetIdentityMemories(vec![first.clone()]))
            .await;
        bb.apply(BlackboardCommand::SetIdentityMemories(vec![second.clone()]))
            .await;

        let records = bb.read(|bb| bb.identity_memories().to_vec()).await;
        assert_eq!(records, vec![second]);
    }

    #[tokio::test]
    async fn memo_queue_assigns_per_owner_indexes_and_retains_latest_items() {
        let bb = Blackboard::new();
        let owner = ModuleInstanceId::new(builtin::sensory(), ReplicaIndex::ZERO);
        bb.apply(BlackboardCommand::SetMemoRetentionPerOwner(2))
            .await;

        let first = bb
            .update_memo(owner.clone(), "first".into(), memo_time(1))
            .await;
        let second = bb
            .update_memo(owner.clone(), "second".into(), memo_time(2))
            .await;
        let third = bb
            .update_memo(owner.clone(), "third".into(), memo_time(3))
            .await;

        assert_eq!(first.index, 0);
        assert_eq!(second.index, 1);
        assert_eq!(third.index, 2);

        let logs = bb.read(|bb| bb.recent_memo_logs()).await;
        let owner_logs = logs
            .into_iter()
            .filter(|record| record.owner == owner)
            .collect::<Vec<_>>();
        assert_eq!(
            owner_logs
                .iter()
                .map(|record| (record.index, record.content.as_str()))
                .collect::<Vec<_>>(),
            vec![(1, "second"), (2, "third")]
        );
    }

    #[tokio::test]
    async fn allocation_proposals_average_ratio_before_active_replica_derivation() {
        let mut base = ResourceAllocation::default();
        base.set_activation(builtin::attention_controller(), crate::ActivationRatio::ONE);
        let bb = Blackboard::with_allocation(base);
        bb.apply(BlackboardCommand::SetModulePolicies {
            policies: vec![
                (
                    builtin::attention_controller(),
                    test_policy(ReplicaCapRange::new(1, 2).unwrap()),
                ),
                (
                    builtin::query_vector(),
                    test_policy(ReplicaCapRange::new(0, 2).unwrap()),
                ),
                (
                    builtin::speak(),
                    test_policy(ReplicaCapRange::new(0, 1).unwrap()),
                ),
            ],
        })
        .await;

        let mut proposal_a = ResourceAllocation::default();
        proposal_a.set(
            builtin::query_vector(),
            crate::ModuleConfig {
                guidance: "query cheaply".into(),
            },
        );
        proposal_a.set_activation(
            builtin::query_vector(),
            crate::ActivationRatio::from_f64(1.0 / 3.0),
        );
        proposal_a.set(
            builtin::speak(),
            crate::ModuleConfig {
                guidance: "wait".into(),
            },
        );
        proposal_a.set_activation(builtin::speak(), crate::ActivationRatio::ZERO);

        let mut proposal_b = ResourceAllocation::default();
        proposal_b.set(
            builtin::query_vector(),
            crate::ModuleConfig {
                guidance: "query deeply".into(),
            },
        );
        proposal_b.set_activation(builtin::query_vector(), crate::ActivationRatio::ONE);
        proposal_b.set(
            builtin::speak(),
            crate::ModuleConfig {
                guidance: "respond if attention is ready".into(),
            },
        );
        proposal_b.set_activation(builtin::speak(), crate::ActivationRatio::ONE);

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
        let query_activation = effective.activation_for(&builtin::query_vector());
        assert_eq!(effective.active_replicas(&builtin::query_vector()), 2);
        assert!((query_activation.as_f64() - 0.6667).abs() < 0.001);
        assert_eq!(
            effective.tier_for(&builtin::query_vector()),
            ModelTier::Default
        );
        assert_eq!(
            query.guidance,
            "attention-controller: query cheaply\nattention-controller[1]: query deeply"
        );
        assert_eq!(effective.active_replicas(&builtin::speak()), 1);
    }

    #[tokio::test]
    async fn allocation_ignores_proposals_from_inactive_controller_replicas() {
        let mut base = ResourceAllocation::default();
        base.set_activation(
            builtin::attention_controller(),
            crate::ActivationRatio::from_f64(0.5),
        );
        base.set_activation(builtin::speak(), crate::ActivationRatio::ZERO);

        let bb = Blackboard::with_allocation(base);
        bb.apply(BlackboardCommand::SetModulePolicies {
            policies: vec![
                (
                    builtin::attention_controller(),
                    test_policy(ReplicaCapRange::new(1, 2).unwrap()),
                ),
                (
                    builtin::speak(),
                    test_policy(ReplicaCapRange::new(0, 0).unwrap()),
                ),
            ],
        })
        .await;

        let mut active = ResourceAllocation::default();
        active.set_activation(builtin::speak(), crate::ActivationRatio::ZERO);

        let mut inactive = ResourceAllocation::default();
        inactive.set_activation(builtin::speak(), crate::ActivationRatio::ONE);

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
        // Inactive controller's "ONE" activation is filtered out, only the
        // active controller's ZERO is applied.
        assert_eq!(effective.active_replicas(&builtin::speak()), 0);
        assert_eq!(
            effective.activation_for(&builtin::speak()),
            crate::ActivationRatio::ZERO
        );
    }

    #[tokio::test]
    async fn allocation_proposals_do_not_add_unregistered_modules() {
        let mut base = ResourceAllocation::default();
        base.set_activation(builtin::attention_controller(), crate::ActivationRatio::ONE);
        let bb = Blackboard::with_allocation(base);
        bb.apply(BlackboardCommand::SetModulePolicies {
            policies: vec![(
                builtin::attention_controller(),
                test_policy(ReplicaCapRange::new(1, 1).unwrap()),
            )],
        })
        .await;

        let unknown = ModuleId::new("invented-module").unwrap();
        let mut proposal = ResourceAllocation::default();
        proposal.set(
            unknown.clone(),
            crate::ModuleConfig {
                guidance: "ignore me".into(),
            },
        );
        proposal.set_activation(unknown.clone(), crate::ActivationRatio::ONE);

        bb.apply(BlackboardCommand::RecordAllocationProposal {
            controller: ModuleInstanceId::new(builtin::attention_controller(), ReplicaIndex::ZERO),
            proposal,
        })
        .await;

        let effective = bb.read(|bb| bb.allocation().clone()).await;
        assert!(effective.get(&unknown).is_none());
    }

    #[tokio::test]
    async fn activation_increase_waiter_fires_only_on_strict_effective_increase() {
        let module = builtin::query_vector();
        let threshold = crate::ActivationRatio::from_f64(0.5);
        let mut base = ResourceAllocation::default();
        base.set_activation(module.clone(), crate::ActivationRatio::from_f64(0.25));
        let bb = Blackboard::with_allocation(base);

        let mut waiter = bb
            .activation_increase_waiter(module.clone(), threshold)
            .await
            .expect("activation has not increased past threshold yet");

        let mut same_threshold = ResourceAllocation::default();
        same_threshold.set_activation(module.clone(), threshold);
        bb.apply(BlackboardCommand::SetAllocation(same_threshold))
            .await;
        assert!(
            waiter.try_recv().is_err(),
            "equal threshold should not fire"
        );

        let mut lower = ResourceAllocation::default();
        lower.set_activation(module.clone(), crate::ActivationRatio::from_f64(0.4));
        bb.apply(BlackboardCommand::SetAllocation(lower)).await;
        assert!(waiter.try_recv().is_err(), "lower value should not fire");

        let mut higher = ResourceAllocation::default();
        higher.set_activation(module, crate::ActivationRatio::from_f64(0.6));
        bb.apply(BlackboardCommand::SetAllocation(higher)).await;
        assert_eq!(waiter.await, Ok(()));
    }

    #[tokio::test]
    async fn activation_increase_waiter_returns_none_when_already_above_threshold() {
        let module = builtin::query_vector();
        let mut base = ResourceAllocation::default();
        base.set_activation(module.clone(), crate::ActivationRatio::from_f64(0.75));
        let bb = Blackboard::with_allocation(base);

        let waiter = bb
            .activation_increase_waiter(module, crate::ActivationRatio::from_f64(0.5))
            .await;

        assert!(waiter.is_none());
    }

    fn test_policy(range: ReplicaCapRange) -> crate::ModulePolicy {
        crate::ModulePolicy::new(
            range,
            crate::Bpm::from_f64(1.0)..=crate::Bpm::from_f64(60.0),
            crate::linear_ratio_fn,
        )
    }
}
