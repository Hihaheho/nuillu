use nuillu_types::{MemoryIndex, MemoryRank, ModuleId};

use crate::{AttentionStreamEvent, MemoryMetaPatch, ResourceAllocation};

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
        module: ModuleId,
        memo: String,
    },
    AppendAttentionStream(AttentionStreamEvent),
    UpsertMemoryMetadata {
        index: MemoryIndex,
        rank_if_new: MemoryRank,
        decay_if_new_secs: i64,
        patch: MemoryMetaPatch,
    },
    RemoveMemoryMetadata {
        index: MemoryIndex,
    },
    SetAllocation(ResourceAllocation),
}
