use chrono::{DateTime, Utc};
use nuillu_types::{MemoryIndex, MemoryRank, ModuleId, ModuleInstanceId, PolicyIndex, PolicyRank};

use crate::{
    AgenticDeadlockMarker, AllocationLimits, CognitionLogEntry, CorePolicyRecord,
    IdentityMemoryRecord, MemoryMetaPatch, ModulePolicy, ModuleRunStatus, PolicyMetaPatch,
    ResourceAllocation, UtteranceProgress, VitalPatch,
};

/// Internal blackboard mutation. Constructed only by the agent's
/// capability layer (each capability is responsible for one variant) and
/// applied via [`Blackboard::apply`](crate::Blackboard::apply).
///
/// Modules never see this enum — they hold capability handles whose
/// methods build and apply the appropriate command. That is what enforces
/// the design invariants at the type level: a module without
/// [`CognitionWriter`](nuillu_module::CognitionWriter) cannot construct
/// `AppendCognitionLog`, etc.
#[derive(Debug, Clone)]
pub enum BlackboardCommand {
    UpdateMemo {
        owner: ModuleInstanceId,
        memo: String,
        written_at: DateTime<Utc>,
    },
    SetModuleRunStatus {
        owner: ModuleInstanceId,
        status: ModuleRunStatus,
    },
    SetUtteranceProgress {
        owner: ModuleInstanceId,
        progress: UtteranceProgress,
    },
    AppendCognitionLog {
        source: ModuleInstanceId,
        entry: CognitionLogEntry,
    },
    UpdateVital {
        patch: VitalPatch,
        now: DateTime<Utc>,
    },
    RecordAgenticDeadlockMarker(AgenticDeadlockMarker),
    UpsertMemoryMetadata {
        index: MemoryIndex,
        rank_if_new: MemoryRank,
        occurred_at_if_new: Option<DateTime<Utc>>,
        decay_if_new_secs: i64,
        now: DateTime<Utc>,
        patch: MemoryMetaPatch,
    },
    RemoveMemoryMetadata {
        index: MemoryIndex,
    },
    SetIdentityMemories(Vec<IdentityMemoryRecord>),
    UpsertPolicyMetadata {
        index: PolicyIndex,
        rank_if_new: PolicyRank,
        decay_if_new_secs: i64,
        patch: PolicyMetaPatch,
    },
    RemovePolicyMetadata {
        index: PolicyIndex,
    },
    SetCorePolicies(Vec<CorePolicyRecord>),
    SetAllocation(ResourceAllocation),
    SetModulePolicies {
        policies: Vec<(ModuleId, ModulePolicy)>,
    },
    SetAllocationLimits(AllocationLimits),
    SetMemoRetentionPerOwner(usize),
    RecordAllocationProposal {
        controller: ModuleInstanceId,
        proposal: ResourceAllocation,
    },
    RecordAllocationCap {
        controller: ModuleInstanceId,
        cap: ResourceAllocation,
    },
}
