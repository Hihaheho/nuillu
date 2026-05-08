use chrono::{DateTime, Utc};
use nuillu_types::{MemoryIndex, MemoryRank, ModuleId, ModuleInstanceId, ReplicaCapRange};

use crate::{
    AgenticDeadlockMarker, AllocationLimits, AttentionStreamEvent, MemoryMetaPatch,
    ModuleRunStatus, ResourceAllocation, UtteranceProgress,
};

/// Internal blackboard mutation. Constructed only by the agent's
/// capability layer (each capability is responsible for one variant) and
/// applied via [`Blackboard::apply`](crate::Blackboard::apply).
///
/// Modules never see this enum — they hold capability handles whose
/// methods build and apply the appropriate command. That is what enforces
/// the design invariants at the type level: a module without
/// [`AttentionWriter`](nuillu_module::AttentionWriter) cannot construct
/// `AppendAttentionStream`, etc.
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
    AppendAttentionStream {
        stream: ModuleInstanceId,
        event: AttentionStreamEvent,
    },
    RecordAgenticDeadlockMarker(AgenticDeadlockMarker),
    UpsertMemoryMetadata {
        index: MemoryIndex,
        rank_if_new: MemoryRank,
        decay_if_new_secs: i64,
        now: DateTime<Utc>,
        patch: MemoryMetaPatch,
    },
    RemoveMemoryMetadata {
        index: MemoryIndex,
    },
    SetAllocation(ResourceAllocation),
    SetReplicaCaps {
        caps: Vec<(ModuleId, ReplicaCapRange)>,
    },
    SetAllocationLimits(AllocationLimits),
    SetMemoRetentionPerOwner(usize),
    RecordAllocationProposal {
        controller: ModuleInstanceId,
        proposal: ResourceAllocation,
    },
}
