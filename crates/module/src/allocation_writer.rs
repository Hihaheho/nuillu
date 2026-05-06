use nuillu_blackboard::{Blackboard, BlackboardCommand, ResourceAllocation};

/// Replace the agent's resource-allocation snapshot.
///
/// The factory does not enforce uniqueness: capabilities are non-exclusive.
/// By boot-time wiring convention only the attention controller receives
/// this handle, but multiple writers are structurally permitted.
pub struct AllocationWriter {
    blackboard: Blackboard,
}

impl AllocationWriter {
    pub(crate) fn new(blackboard: Blackboard) -> Self {
        Self { blackboard }
    }

    pub async fn set(&self, allocation: ResourceAllocation) {
        self.blackboard
            .apply(BlackboardCommand::SetAllocation(allocation))
            .await;
    }
}
