use nuillu_blackboard::{Blackboard, BlackboardCommand, ResourceAllocation};
use nuillu_types::ModuleInstanceId;

use crate::{AllocationUpdated, AllocationUpdatedMailbox};

/// Replace the agent's resource-allocation snapshot.
///
/// Capability issuers do not enforce uniqueness: capabilities are non-exclusive.
/// By boot-time wiring convention only the attention controller receives
/// this handle, but multiple writers are structurally permitted.
pub struct AllocationWriter {
    owner: ModuleInstanceId,
    blackboard: Blackboard,
    updates: AllocationUpdatedMailbox,
}

impl AllocationWriter {
    pub(crate) fn new(
        owner: ModuleInstanceId,
        blackboard: Blackboard,
        updates: AllocationUpdatedMailbox,
    ) -> Self {
        Self {
            owner,
            blackboard,
            updates,
        }
    }

    /// Record this controller replica's proposal. The runtime derives the
    /// effective allocation from active controller proposals.
    pub async fn set(&self, allocation: ResourceAllocation) {
        let before = self.blackboard.read(|bb| bb.allocation().clone()).await;
        self.blackboard
            .apply(BlackboardCommand::RecordAllocationProposal {
                controller: self.owner.clone(),
                proposal: allocation,
            })
            .await;
        let after = self.blackboard.read(|bb| bb.allocation().clone()).await;
        if before != after && self.updates.publish(AllocationUpdated).await.is_err() {
            tracing::trace!("allocation update had no active subscribers");
        }
    }
}
