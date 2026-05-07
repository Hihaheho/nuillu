use nuillu_blackboard::{Blackboard, BlackboardCommand, ResourceAllocation};
use nuillu_types::ModuleInstanceId;

/// Replace the agent's resource-allocation snapshot.
///
/// The factory does not enforce uniqueness: capabilities are non-exclusive.
/// By boot-time wiring convention only the attention controller receives
/// this handle, but multiple writers are structurally permitted.
pub struct AllocationWriter {
    owner: ModuleInstanceId,
    blackboard: Blackboard,
}

impl AllocationWriter {
    pub(crate) fn new(owner: ModuleInstanceId, blackboard: Blackboard) -> Self {
        Self { owner, blackboard }
    }

    /// Record this controller replica's proposal. The runtime derives the
    /// effective allocation from active controller proposals.
    pub async fn set(&self, allocation: ResourceAllocation) {
        self.blackboard
            .apply(BlackboardCommand::RecordAllocationProposal {
                controller: self.owner.clone(),
                proposal: allocation,
            })
            .await;
    }
}
