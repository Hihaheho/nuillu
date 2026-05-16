use std::any::Any;
use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt;
use std::marker::PhantomData;
use std::rc::Rc;
use std::sync::{Arc, Mutex, OnceLock};

use chrono::{DateTime, Utc};
use nuillu_types::{MemoryIndex, ModuleId, ModuleInstanceId, PolicyIndex};
use tokio::sync::{RwLock, oneshot};

use crate::{
    ActivationRatio, AgenticDeadlockMarker, AllocationLimits, BlackboardCommand, CognitionLog,
    CognitionLogEntryRecord, CognitionLogRecord, CognitionLogSet, CorePolicyRecord,
    IdentityMemoryRecord, InteroceptiveState, MemoryMetadata, ModulePolicy, PolicyMetadata,
    ResourceAllocation,
};

const DEFAULT_MEMO_RETAINED_PER_OWNER: usize = 8;
const DEFAULT_COGNITION_LOG_RETAINED_ENTRIES: usize = 16;

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
    inner: Rc<RwLock<BlackboardInner>>,
    activation_waiters: Arc<Mutex<Vec<ActivationWaiter>>>,
    activation_increase_waiters: Arc<Mutex<Vec<ActivationIncreaseWaiter>>>,
    allocation_change_waiters: Arc<Mutex<Vec<oneshot::Sender<()>>>>,
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
    cognition_log_retained_entries: usize,
    interoception: InteroceptiveState,
    agentic_deadlock_marker: Option<AgenticDeadlockMarker>,
    memory_metadata: HashMap<MemoryIndex, MemoryMetadata>,
    identity_memories: Vec<IdentityMemoryRecord>,
    policy_metadata: HashMap<PolicyIndex, PolicyMetadata>,
    core_policies: Vec<CorePolicyRecord>,
    base_allocation: ResourceAllocation,
    allocation: ResourceAllocation,
    allocation_proposals: HashMap<ModuleInstanceId, ResourceAllocation>,
    allocation_caps: HashMap<ModuleInstanceId, ResourceAllocation>,
    forced_disabled_modules: HashSet<ModuleId>,
    module_policies: HashMap<ModuleId, ModulePolicy>,
    module_replica_capacities: HashMap<ModuleId, u8>,
    allocation_limits: AllocationLimits,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct MemoLogRecord {
    pub owner: ModuleInstanceId,
    pub index: u64,
    pub written_at: DateTime<Utc>,
    pub content: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MemoAppendResult {
    pub record: MemoLogRecord,
    pub evicted: Vec<MemoLogRecord>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CognitionLogAppendResult {
    pub record: CognitionLogEntryRecord,
    pub evicted: Vec<CognitionLogEntryRecord>,
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
    PendingActivationGate,
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
}

impl UtteranceProgress {
    pub fn streaming(
        generation_id: u64,
        sequence: u32,
        target: impl Into<String>,
        partial_utterance: impl Into<String>,
    ) -> Self {
        Self {
            state: UtteranceProgressState::Streaming,
            generation_id,
            sequence,
            target: target.into(),
            partial_utterance: partial_utterance.into(),
        }
    }

    pub fn completed(
        generation_id: u64,
        sequence: u32,
        target: impl Into<String>,
        utterance: impl Into<String>,
    ) -> Self {
        Self {
            state: UtteranceProgressState::Completed,
            generation_id,
            sequence,
            target: target.into(),
            partial_utterance: utterance.into(),
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
            inner: Rc::new(RwLock::new(BlackboardInner::default())),
            activation_waiters: Arc::new(Mutex::new(Vec::new())),
            activation_increase_waiters: Arc::new(Mutex::new(Vec::new())),
            allocation_change_waiters: Arc::new(Mutex::new(Vec::new())),
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
            inner: Rc::new(RwLock::new(inner)),
            activation_waiters: Arc::new(Mutex::new(Vec::new())),
            activation_increase_waiters: Arc::new(Mutex::new(Vec::new())),
            allocation_change_waiters: Arc::new(Mutex::new(Vec::new())),
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
        let before = guard.allocation.clone();
        guard.apply(cmd);
        let allocation_changed = before != guard.allocation;
        self.notify_active_waiters(&guard.allocation);
        self.notify_activation_increase_waiters(&guard.allocation);
        if allocation_changed {
            self.notify_allocation_change_waiters();
        }
    }

    pub async fn update_memo(
        &self,
        owner: ModuleInstanceId,
        memo: String,
        written_at: DateTime<Utc>,
    ) -> MemoLogRecord {
        self.update_memo_with_evictions(owner, memo, written_at)
            .await
            .record
    }

    pub async fn update_memo_with_evictions(
        &self,
        owner: ModuleInstanceId,
        memo: String,
        written_at: DateTime<Utc>,
    ) -> MemoAppendResult {
        self.update_typed_memo_with_evictions(owner, memo, (), written_at)
            .await
    }

    pub async fn update_typed_memo<T: 'static>(
        &self,
        owner: ModuleInstanceId,
        memo: String,
        payload: T,
        written_at: DateTime<Utc>,
    ) -> MemoLogRecord {
        self.update_typed_memo_with_evictions(owner, memo, payload, written_at)
            .await
            .record
    }

    pub async fn update_typed_memo_with_evictions<T: 'static>(
        &self,
        owner: ModuleInstanceId,
        memo: String,
        payload: T,
        written_at: DateTime<Utc>,
    ) -> MemoAppendResult {
        let mut guard = self.inner.write().await;
        guard.append_memo(owner, memo, Arc::new(payload), written_at)
    }

    pub async fn append_cognition_log(
        &self,
        source: ModuleInstanceId,
        entry: crate::CognitionLogEntry,
    ) -> CognitionLogAppendResult {
        let mut guard = self.inner.write().await;
        guard.append_cognition_log(source, entry)
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

    pub async fn allocation_change_waiter(&self) -> oneshot::Receiver<()> {
        let mut waiters = self
            .allocation_change_waiters
            .lock()
            .expect("allocation change waiters poisoned");
        waiters.retain(|sender| !sender.is_closed());
        let (sender, receiver) = oneshot::channel();
        waiters.push(sender);
        receiver
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

    fn notify_allocation_change_waiters(&self) {
        let mut waiters = self
            .allocation_change_waiters
            .lock()
            .expect("allocation change waiters poisoned");
        for waiter in waiters.drain(..) {
            let _ = waiter.send(());
        }
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
            cognition_log_retained_entries: DEFAULT_COGNITION_LOG_RETAINED_ENTRIES,
            interoception: InteroceptiveState::default(),
            agentic_deadlock_marker: None,
            memory_metadata: HashMap::new(),
            identity_memories: Vec::new(),
            policy_metadata: HashMap::new(),
            core_policies: Vec::new(),
            base_allocation: ResourceAllocation::default(),
            allocation: ResourceAllocation::default(),
            allocation_proposals: HashMap::new(),
            allocation_caps: HashMap::new(),
            forced_disabled_modules: HashSet::new(),
            module_policies: HashMap::new(),
            module_replica_capacities: HashMap::new(),
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

    pub fn cognition_log_retained_entries(&self) -> usize {
        self.cognition_log_retained_entries
    }

    pub fn memory_metadata(&self) -> &HashMap<MemoryIndex, MemoryMetadata> {
        &self.memory_metadata
    }

    pub fn identity_memories(&self) -> &[IdentityMemoryRecord] {
        &self.identity_memories
    }

    pub fn policy_metadata(&self) -> &HashMap<PolicyIndex, PolicyMetadata> {
        &self.policy_metadata
    }

    pub fn core_policies(&self) -> &[CorePolicyRecord] {
        &self.core_policies
    }

    pub fn interoception(&self) -> &InteroceptiveState {
        &self.interoception
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

    pub fn allocation_caps(&self) -> &HashMap<ModuleInstanceId, ResourceAllocation> {
        &self.allocation_caps
    }

    pub fn forced_disabled_modules(&self) -> &HashSet<ModuleId> {
        &self.forced_disabled_modules
    }

    pub fn module_policies(&self) -> &HashMap<ModuleId, ModulePolicy> {
        &self.module_policies
    }

    pub fn module_replica_capacity(&self, module: &ModuleId) -> Option<u8> {
        self.module_replica_capacities
            .get(module)
            .copied()
            .or_else(|| {
                self.module_policies
                    .get(module)
                    .map(ModulePolicy::max_active_replicas)
            })
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
                self.append_cognition_log(source, entry);
            }
            BlackboardCommand::UpdateInteroceptive { patch, now } => {
                self.interoception.apply_patch(patch, now);
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
            BlackboardCommand::UpsertPolicyMetadata {
                index,
                rank_if_new,
                decay_if_new_secs,
                patch,
            } => {
                let entry = self
                    .policy_metadata
                    .entry(index.clone())
                    .or_insert_with(|| {
                        PolicyMetadata::new_at(index, rank_if_new, decay_if_new_secs)
                    });
                patch.apply(entry);
            }
            BlackboardCommand::RemovePolicyMetadata { index } => {
                self.policy_metadata.remove(&index);
            }
            BlackboardCommand::SetCorePolicies(records) => {
                self.core_policies = records;
            }
            BlackboardCommand::SetAllocation(alloc) => {
                self.base_allocation = alloc;
                self.recompute_effective_allocation();
            }
            BlackboardCommand::SetModulePolicies { policies } => {
                for (module, policy) in policies {
                    self.module_replica_capacities
                        .entry(module.clone())
                        .or_insert_with(|| policy.max_active_replicas());
                    self.module_policies.insert(module, policy);
                }
                self.recompute_effective_allocation();
            }
            BlackboardCommand::SetModuleReplicaCapacities { capacities } => {
                for (module, capacity) in capacities {
                    self.module_replica_capacities
                        .insert(module, capacity.max(1));
                }
            }
            BlackboardCommand::SetAllocationLimits(limits) => {
                self.allocation_limits = limits;
                self.recompute_effective_allocation();
            }
            BlackboardCommand::SetMemoRetentionPerOwner(retained) => {
                self.memo_retained_per_owner = retained.max(1);
                self.truncate_memos_to_retention();
            }
            BlackboardCommand::SetCognitionLogRetentionEntries(retained) => {
                self.cognition_log_retained_entries = retained.max(1);
                self.truncate_cognition_log_to_retention();
            }
            BlackboardCommand::SetModuleForcedDisabled { module, disabled } => {
                if disabled {
                    self.forced_disabled_modules.insert(module);
                } else {
                    self.forced_disabled_modules.remove(&module);
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
            BlackboardCommand::RecordAllocationCap { controller, cap } => {
                self.allocation_caps.insert(controller, cap);
                self.recompute_effective_allocation();
            }
            BlackboardCommand::RecordAllocationEffects {
                writer,
                targets,
                suppressions,
            } => {
                self.allocation_proposals.insert(writer.clone(), targets);
                self.allocation_caps.insert(writer, suppressions);
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
    ) -> MemoAppendResult {
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
        let mut evicted = Vec::new();
        while records.len() > retained {
            if let Some(entry) = records.pop_front() {
                evicted.push(entry.record());
            }
        }
        MemoAppendResult { record, evicted }
    }

    fn truncate_memos_to_retention(&mut self) -> Vec<MemoLogRecord> {
        let retained = self.memo_retained_per_owner.max(1);
        let mut evicted = Vec::new();
        for records in self.memos.values_mut() {
            while records.len() > retained {
                if let Some(entry) = records.pop_front() {
                    evicted.push(entry.record());
                }
            }
        }
        sort_memo_logs(&mut evicted);
        evicted
    }

    fn append_cognition_log(
        &mut self,
        source: ModuleInstanceId,
        entry: crate::CognitionLogEntry,
    ) -> CognitionLogAppendResult {
        let record = CognitionLogEntryRecord {
            index: self.cognition_next_index,
            source: source.clone(),
            entry: entry.clone(),
        };
        self.cognition_next_index = self.cognition_next_index.saturating_add(1);
        self.cognition_entry_log.push(record.clone());
        self.cognition_logs.entry(source).or_default().append(entry);
        let evicted = self.truncate_cognition_log_to_retention();
        CognitionLogAppendResult { record, evicted }
    }

    fn truncate_cognition_log_to_retention(&mut self) -> Vec<CognitionLogEntryRecord> {
        let retained = self.cognition_log_retained_entries.max(1);
        let mut evicted = Vec::new();
        while self.cognition_entry_log.len() > retained {
            let record = self.cognition_entry_log.remove(0);
            let remove_source = if let Some(log) = self.cognition_logs.get_mut(&record.source) {
                log.remove_oldest();
                log.is_empty()
            } else {
                false
            };
            if remove_source {
                self.cognition_logs.remove(&record.source);
            }
            evicted.push(record);
        }
        evicted
    }

    fn recompute_effective_allocation(&mut self) {
        let mut active_proposals = self.active_allocation_maps(&self.allocation_proposals);
        active_proposals.sort_by(|(left, _), (right, _)| {
            left.module
                .as_str()
                .cmp(right.module.as_str())
                .then_with(|| left.replica.cmp(&right.replica))
        });
        let mut active_caps = self.active_allocation_maps(&self.allocation_caps);
        active_caps.sort_by(|(left, _), (right, _)| {
            left.module
                .as_str()
                .cmp(right.module.as_str())
                .then_with(|| left.replica.cmp(&right.replica))
        });

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
            let module_proposals = active_proposals
                .iter()
                .filter(|(_, proposal)| proposal.has_module_opinion(&id))
                .collect::<Vec<_>>();
            if module_proposals.is_empty() {
                effective.set(id.clone(), self.base_allocation.for_module(&id));
                effective.set_activation(id.clone(), self.base_allocation.activation_for(&id));
                continue;
            }

            let count = module_proposals.len() as u32;
            let activation_sum = module_proposals
                .iter()
                .map(|(_, proposal)| {
                    if proposal.has_activation(&id) {
                        proposal.activation_for(&id)
                    } else {
                        self.base_allocation.activation_for(&id)
                    }
                })
                .map(|ratio| u32::from(ratio.raw()))
                .sum::<u32>();

            let guidance = if module_proposals
                .iter()
                .any(|(_, proposal)| proposal.get(&id).is_some())
            {
                combine_guidance(module_proposals.iter().filter_map(|(owner, proposal)| {
                    proposal
                        .get(&id)
                        .map(|cfg| (owner.to_string(), cfg.guidance.trim().to_owned()))
                }))
            } else {
                self.base_allocation.for_module(&id).guidance
            };

            effective.set(id.clone(), crate::ModuleConfig { guidance });
            effective.set_activation(
                id,
                crate::ActivationRatio::from_raw(rounded_div(activation_sum, count) as u16),
            );
        }

        let capped_ids = effective
            .iter_activation()
            .map(|(id, _)| id.clone())
            .collect::<Vec<_>>();
        for id in capped_ids {
            let multipliers = active_caps
                .iter()
                .filter(|(_, cap)| cap.has_activation(&id))
                .map(|(_, cap)| cap.activation_for(&id))
                .collect::<Vec<_>>();
            for multiplier in multipliers {
                effective.multiply_activation(id.clone(), multiplier);
            }
        }

        self.allocation = effective
            .derived(&self.module_policies)
            .limited(self.allocation_limits)
            .force_disable_modules(&self.forced_disabled_modules);
    }

    fn active_allocation_maps<'a>(
        &self,
        maps: &'a HashMap<ModuleInstanceId, ResourceAllocation>,
    ) -> Vec<(&'a ModuleInstanceId, &'a ResourceAllocation)> {
        maps.iter()
            .filter(|(owner, _)| self.allocation.is_replica_active(owner))
            .collect()
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
        let owner = ModuleInstanceId::new(builtin::query_memory(), ReplicaIndex::ZERO);
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
                "query-memory": [{
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
    async fn memo_append_result_reports_evicted_entries() {
        let bb = Blackboard::new();
        let owner = ModuleInstanceId::new(builtin::sensory(), ReplicaIndex::ZERO);
        bb.apply(BlackboardCommand::SetMemoRetentionPerOwner(1))
            .await;

        let first = bb
            .update_memo(owner.clone(), "first".into(), memo_time(1))
            .await;
        assert_eq!(first.index, 0);

        let second = bb
            .update_typed_memo_with_evictions(owner.clone(), "second".into(), (), memo_time(2))
            .await;
        assert_eq!(second.record.index, 1);
        assert_eq!(
            second.evicted,
            vec![MemoLogRecord {
                owner,
                index: 0,
                written_at: memo_time(1),
                content: "first".into(),
            }]
        );
    }

    #[tokio::test]
    async fn cognition_log_retains_latest_entries_and_reports_evictions() {
        let bb = Blackboard::new();
        let cognition_gate = ModuleInstanceId::new(builtin::cognition_gate(), ReplicaIndex::ZERO);
        let attention_schema =
            ModuleInstanceId::new(builtin::attention_schema(), ReplicaIndex::ZERO);
        bb.apply(BlackboardCommand::SetCognitionLogRetentionEntries(2))
            .await;

        let first = bb
            .append_cognition_log(
                cognition_gate.clone(),
                crate::CognitionLogEntry {
                    at: memo_time(1),
                    text: "first".into(),
                },
            )
            .await;
        assert_eq!(first.record.index, 0);
        assert!(first.evicted.is_empty());

        let second = bb
            .append_cognition_log(
                attention_schema.clone(),
                crate::CognitionLogEntry {
                    at: memo_time(2),
                    text: "second".into(),
                },
            )
            .await;
        assert_eq!(second.record.index, 1);
        assert!(second.evicted.is_empty());

        let third = bb
            .append_cognition_log(
                cognition_gate.clone(),
                crate::CognitionLogEntry {
                    at: memo_time(3),
                    text: "third".into(),
                },
            )
            .await;
        assert_eq!(third.record.index, 2);
        assert_eq!(
            third.evicted,
            vec![CognitionLogEntryRecord {
                index: 0,
                source: cognition_gate.clone(),
                entry: crate::CognitionLogEntry {
                    at: memo_time(1),
                    text: "first".into(),
                },
            }]
        );

        let retained = bb
            .read(|bb| bb.unread_cognition_log_entries(None))
            .await
            .into_iter()
            .map(|record| (record.index, record.source, record.entry.text))
            .collect::<Vec<_>>();
        assert_eq!(
            retained,
            vec![
                (1, attention_schema, "second".to_owned()),
                (2, cognition_gate, "third".to_owned()),
            ]
        );
    }

    #[tokio::test]
    async fn allocation_proposals_average_ratio_before_active_replica_derivation() {
        let mut base = ResourceAllocation::default();
        base.set_activation(
            builtin::allocation_controller(),
            crate::ActivationRatio::ONE,
        );
        let bb = Blackboard::with_allocation(base);
        bb.apply(BlackboardCommand::SetModulePolicies {
            policies: vec![
                (
                    builtin::allocation_controller(),
                    test_policy(ReplicaCapRange::new(1, 2).unwrap()),
                ),
                (
                    builtin::query_memory(),
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
            builtin::query_memory(),
            crate::ModuleConfig {
                guidance: "query cheaply".into(),
            },
        );
        proposal_a.set_activation(
            builtin::query_memory(),
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
            builtin::query_memory(),
            crate::ModuleConfig {
                guidance: "query deeply".into(),
            },
        );
        proposal_b.set_activation(builtin::query_memory(), crate::ActivationRatio::ONE);
        proposal_b.set(
            builtin::speak(),
            crate::ModuleConfig {
                guidance: "respond if attention is ready".into(),
            },
        );
        proposal_b.set_activation(builtin::speak(), crate::ActivationRatio::ONE);

        bb.apply(BlackboardCommand::RecordAllocationProposal {
            controller: ModuleInstanceId::new(
                builtin::allocation_controller(),
                ReplicaIndex::new(0),
            ),
            proposal: proposal_a,
        })
        .await;
        bb.apply(BlackboardCommand::RecordAllocationProposal {
            controller: ModuleInstanceId::new(
                builtin::allocation_controller(),
                ReplicaIndex::new(1),
            ),
            proposal: proposal_b,
        })
        .await;

        let effective = bb.read(|bb| bb.allocation().clone()).await;
        let query = effective.for_module(&builtin::query_memory());
        let query_activation = effective.activation_for(&builtin::query_memory());
        assert_eq!(effective.active_replicas(&builtin::query_memory()), 2);
        assert!((query_activation.as_f64() - 0.6667).abs() < 0.001);
        assert_eq!(
            effective.tier_for(&builtin::query_memory()),
            ModelTier::Default
        );
        assert_eq!(
            query.guidance,
            "allocation-controller: query cheaply\nallocation-controller[1]: query deeply"
        );
        assert_eq!(effective.active_replicas(&builtin::speak()), 1);
    }

    #[tokio::test]
    async fn forced_disabled_module_overrides_derived_active_replicas() {
        let mut base = ResourceAllocation::default();
        base.set_activation(builtin::sensory(), crate::ActivationRatio::ONE);
        let bb = Blackboard::with_allocation(base);
        bb.apply(BlackboardCommand::SetModulePolicies {
            policies: vec![(
                builtin::sensory(),
                test_policy(ReplicaCapRange::new(1, 1).unwrap()),
            )],
        })
        .await;

        assert_eq!(
            bb.read(|bb| bb.allocation().active_replicas(&builtin::sensory()))
                .await,
            1
        );

        bb.apply(BlackboardCommand::SetModuleForcedDisabled {
            module: builtin::sensory(),
            disabled: true,
        })
        .await;

        let disabled = bb
            .read(|bb| {
                (
                    bb.allocation().active_replicas(&builtin::sensory()),
                    bb.allocation().activation_for(&builtin::sensory()),
                    bb.forced_disabled_modules().contains(&builtin::sensory()),
                )
            })
            .await;
        assert_eq!(disabled.0, 0);
        assert_eq!(disabled.1, crate::ActivationRatio::ONE);
        assert!(disabled.2);

        bb.apply(BlackboardCommand::SetModuleForcedDisabled {
            module: builtin::sensory(),
            disabled: false,
        })
        .await;

        let reenabled = bb
            .read(|bb| {
                (
                    bb.allocation().active_replicas(&builtin::sensory()),
                    bb.forced_disabled_modules().contains(&builtin::sensory()),
                )
            })
            .await;
        assert_eq!(reenabled, (1, false));
    }

    #[tokio::test]
    async fn allocation_ignores_proposals_from_inactive_controller_replicas() {
        let mut base = ResourceAllocation::default();
        base.set_activation(
            builtin::allocation_controller(),
            crate::ActivationRatio::from_f64(0.5),
        );
        base.set_activation(builtin::speak(), crate::ActivationRatio::ZERO);

        let bb = Blackboard::with_allocation(base);
        bb.apply(BlackboardCommand::SetModulePolicies {
            policies: vec![
                (
                    builtin::allocation_controller(),
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
                builtin::allocation_controller(),
                ReplicaIndex::new(0),
            ),
            proposal: active,
        })
        .await;
        bb.apply(BlackboardCommand::RecordAllocationProposal {
            controller: ModuleInstanceId::new(
                builtin::allocation_controller(),
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
        base.set_activation(
            builtin::allocation_controller(),
            crate::ActivationRatio::ONE,
        );
        let bb = Blackboard::with_allocation(base);
        bb.apply(BlackboardCommand::SetModulePolicies {
            policies: vec![(
                builtin::allocation_controller(),
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
            controller: ModuleInstanceId::new(builtin::allocation_controller(), ReplicaIndex::ZERO),
            proposal,
        })
        .await;

        let effective = bb.read(|bb| bb.allocation().clone()).await;
        assert!(effective.get(&unknown).is_none());
    }

    #[tokio::test]
    async fn allocation_caps_clamp_active_drive_without_disabling_replica() {
        let mut base = ResourceAllocation::default();
        base.set_activation(
            builtin::allocation_controller(),
            crate::ActivationRatio::ONE,
        );
        base.set_activation(
            builtin::homeostatic_controller(),
            crate::ActivationRatio::ONE,
        );
        base.set_activation(builtin::speak(), crate::ActivationRatio::ZERO);

        let bb = Blackboard::with_allocation(base);
        bb.apply(BlackboardCommand::SetModulePolicies {
            policies: vec![
                (
                    builtin::allocation_controller(),
                    test_policy(ReplicaCapRange::new(1, 1).unwrap()),
                ),
                (
                    builtin::homeostatic_controller(),
                    test_policy(ReplicaCapRange::new(1, 1).unwrap()),
                ),
                (
                    builtin::speak(),
                    test_policy(ReplicaCapRange::new(0, 1).unwrap()),
                ),
            ],
        })
        .await;

        let mut drive = ResourceAllocation::default();
        drive.set_activation(builtin::speak(), crate::ActivationRatio::ONE);
        bb.apply(BlackboardCommand::RecordAllocationProposal {
            controller: ModuleInstanceId::new(builtin::allocation_controller(), ReplicaIndex::ZERO),
            proposal: drive,
        })
        .await;

        let mut cap = ResourceAllocation::default();
        cap.set_activation(builtin::speak(), crate::ActivationRatio::from_f64(0.15));
        bb.apply(BlackboardCommand::RecordAllocationCap {
            controller: ModuleInstanceId::new(
                builtin::homeostatic_controller(),
                ReplicaIndex::ZERO,
            ),
            cap,
        })
        .await;

        let effective = bb.read(|bb| bb.allocation().clone()).await;
        assert_eq!(
            effective.activation_for(&builtin::speak()),
            crate::ActivationRatio::from_f64(0.15)
        );
        assert_eq!(effective.active_replicas(&builtin::speak()), 1);

        let mut wake_cap = ResourceAllocation::default();
        wake_cap.set_activation(builtin::speak(), crate::ActivationRatio::ONE);
        bb.apply(BlackboardCommand::RecordAllocationCap {
            controller: ModuleInstanceId::new(
                builtin::homeostatic_controller(),
                ReplicaIndex::ZERO,
            ),
            cap: wake_cap,
        })
        .await;

        let effective = bb.read(|bb| bb.allocation().clone()).await;
        assert_eq!(
            effective.activation_for(&builtin::speak()),
            crate::ActivationRatio::ONE
        );
    }

    #[tokio::test]
    async fn allocation_effects_average_targets_multiply_suppressions_and_derive_bpm() {
        let mut base = ResourceAllocation::default();
        base.set_activation(
            builtin::allocation_controller(),
            crate::ActivationRatio::ONE,
        );
        base.set_activation(builtin::interoception(), crate::ActivationRatio::ONE);
        base.set_activation(
            builtin::homeostatic_controller(),
            crate::ActivationRatio::ONE,
        );
        base.set_activation(builtin::speak(), crate::ActivationRatio::ZERO);

        let bb = Blackboard::with_allocation(base);
        bb.apply(BlackboardCommand::SetModulePolicies {
            policies: vec![
                (
                    builtin::allocation_controller(),
                    test_policy(ReplicaCapRange::new(1, 1).unwrap()),
                ),
                (
                    builtin::interoception(),
                    test_policy(ReplicaCapRange::new(1, 1).unwrap()),
                ),
                (
                    builtin::homeostatic_controller(),
                    test_policy(ReplicaCapRange::new(1, 1).unwrap()),
                ),
                (
                    builtin::speak(),
                    crate::ModulePolicy::new(
                        ReplicaCapRange::new(0, 2).unwrap(),
                        crate::Bpm::from_f64(1.0)..=crate::Bpm::from_f64(101.0),
                        crate::linear_ratio_fn,
                    ),
                ),
            ],
        })
        .await;

        let mut target_a = ResourceAllocation::default();
        target_a.set(
            builtin::speak(),
            crate::ModuleConfig {
                guidance: "target A".into(),
            },
        );
        target_a.set_activation(builtin::speak(), crate::ActivationRatio::from_f64(0.2));
        bb.apply(BlackboardCommand::RecordAllocationEffects {
            writer: ModuleInstanceId::new(builtin::allocation_controller(), ReplicaIndex::ZERO),
            targets: target_a,
            suppressions: ResourceAllocation::default(),
        })
        .await;

        let mut target_b = ResourceAllocation::default();
        target_b.set(
            builtin::speak(),
            crate::ModuleConfig {
                guidance: "target B".into(),
            },
        );
        target_b.set_activation(builtin::speak(), crate::ActivationRatio::from_f64(0.8));
        let mut suppression_b = ResourceAllocation::default();
        suppression_b.set_activation(builtin::speak(), crate::ActivationRatio::from_f64(0.5));
        bb.apply(BlackboardCommand::RecordAllocationEffects {
            writer: ModuleInstanceId::new(builtin::interoception(), ReplicaIndex::ZERO),
            targets: target_b,
            suppressions: suppression_b,
        })
        .await;

        let mut suppression_c = ResourceAllocation::default();
        suppression_c.set_activation(builtin::speak(), crate::ActivationRatio::from_f64(0.25));
        bb.apply(BlackboardCommand::RecordAllocationEffects {
            writer: ModuleInstanceId::new(builtin::homeostatic_controller(), ReplicaIndex::ZERO),
            targets: ResourceAllocation::default(),
            suppressions: suppression_c,
        })
        .await;

        let effective = bb.read(|bb| bb.allocation().clone()).await;
        assert_eq!(
            effective.activation_for(&builtin::speak()),
            crate::ActivationRatio::from_f64(0.0625)
        );
        assert_eq!(effective.active_replicas(&builtin::speak()), 1);
        assert_eq!(
            effective.cooldown_for(&builtin::speak()),
            Some(crate::Bpm::from_f64(7.25).cooldown())
        );
        assert_eq!(
            effective.for_module(&builtin::speak()).guidance,
            "allocation-controller: target A\ninteroception: target B"
        );
    }

    #[tokio::test]
    async fn activation_increase_waiter_fires_only_on_strict_effective_increase() {
        let module = builtin::query_memory();
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
        let module = builtin::query_memory();
        let mut base = ResourceAllocation::default();
        base.set_activation(module.clone(), crate::ActivationRatio::from_f64(0.75));
        let bb = Blackboard::with_allocation(base);

        let waiter = bb
            .activation_increase_waiter(module, crate::ActivationRatio::from_f64(0.5))
            .await;

        assert!(waiter.is_none());
    }

    #[tokio::test]
    async fn allocation_change_waiter_fires_when_policy_changes_derived_cooldown() {
        let module = builtin::query_memory();
        let mut base = ResourceAllocation::default();
        base.set(module.clone(), crate::ModuleConfig::default());
        base.set_activation(module.clone(), crate::ActivationRatio::ONE);
        let bb = Blackboard::with_allocation(base);
        bb.apply(BlackboardCommand::SetModulePolicies {
            policies: vec![(
                module.clone(),
                crate::ModulePolicy::new(
                    ReplicaCapRange::new(0, 1).unwrap(),
                    crate::Bpm::from_f64(1.0)..=crate::Bpm::from_f64(1.0),
                    crate::linear_ratio_fn,
                ),
            )],
        })
        .await;

        let waiter = bb.allocation_change_waiter().await;
        bb.apply(BlackboardCommand::SetModulePolicies {
            policies: vec![(
                module,
                crate::ModulePolicy::new(
                    ReplicaCapRange::new(0, 1).unwrap(),
                    crate::Bpm::from_f64(60.0)..=crate::Bpm::from_f64(60.0),
                    crate::linear_ratio_fn,
                ),
            )],
        })
        .await;

        assert_eq!(waiter.await, Ok(()));
    }

    fn test_policy(range: ReplicaCapRange) -> crate::ModulePolicy {
        crate::ModulePolicy::new(
            range,
            crate::Bpm::from_f64(1.0)..=crate::Bpm::from_f64(60.0),
            crate::linear_ratio_fn,
        )
    }
}
