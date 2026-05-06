use nuillu_blackboard::{Blackboard, BlackboardCommand};
use nuillu_types::ModuleId;

/// Read-write handle for the activating module's own memo slot.
///
/// The owner identity is stamped at construction time by
/// [`CapabilityFactory`](crate::CapabilityFactory); the module cannot
/// change it. A module that does not hold a `Memo` has no memo slot
/// allocated on the blackboard at all (slot creation is lazy and only
/// `Memo::write` ever inserts one).
#[derive(Clone)]
pub struct Memo {
    owner: ModuleId,
    blackboard: Blackboard,
}

impl Memo {
    pub(crate) fn new(owner: ModuleId, blackboard: Blackboard) -> Self {
        Self { owner, blackboard }
    }

    pub fn owner(&self) -> &ModuleId {
        &self.owner
    }

    /// Replace the owner module's memo. Allocates the slot on first call.
    pub async fn write(&self, memo: impl Into<String>) {
        self.blackboard
            .apply(BlackboardCommand::UpdateMemo {
                module: self.owner.clone(),
                memo: memo.into(),
            })
            .await;
    }

    /// Read the owner module's own memo, or `None` if it has never been
    /// written.
    pub async fn read(&self) -> Option<String> {
        self.blackboard.memo(&self.owner).await
    }
}
