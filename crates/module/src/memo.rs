use nuillu_blackboard::{Blackboard, BlackboardCommand};
use nuillu_types::ModuleInstanceId;

use crate::runtime_events::RuntimeEventEmitter;
use crate::{MemoUpdated, MemoUpdatedMailbox};

/// Read-write handle for the activating module's own memo slot.
///
/// The owner identity is stamped at construction time by
/// [`ModuleCapabilityFactory`](crate::ModuleCapabilityFactory); the module cannot
/// change it. A module that does not hold a `Memo` has no memo slot
/// allocated on the blackboard at all (slot creation is lazy and only
/// `Memo::write` ever inserts one).
#[derive(Clone)]
pub struct Memo {
    owner: ModuleInstanceId,
    blackboard: Blackboard,
    updates: MemoUpdatedMailbox,
    events: RuntimeEventEmitter,
}

impl Memo {
    pub(crate) fn new(
        owner: ModuleInstanceId,
        blackboard: Blackboard,
        updates: MemoUpdatedMailbox,
        events: RuntimeEventEmitter,
    ) -> Self {
        Self {
            owner,
            blackboard,
            updates,
            events,
        }
    }

    /// Replace the owner module's memo. Allocates the slot on first call.
    pub async fn write(&self, memo: impl Into<String>) {
        let memo = memo.into();
        let char_count = memo.chars().count();
        self.blackboard
            .apply(BlackboardCommand::UpdateMemo {
                owner: self.owner.clone(),
                memo,
            })
            .await;
        self.events
            .memo_updated(self.owner.clone(), char_count)
            .await;
        if self
            .updates
            .publish(MemoUpdated {
                owner: self.owner.clone(),
            })
            .await
            .is_err()
        {
            tracing::trace!("memo update had no active subscribers");
        }
    }

    /// Read the owner module's own memo, or `None` if it has never been
    /// written.
    pub async fn read(&self) -> Option<String> {
        self.blackboard.memo_for_instance(&self.owner).await
    }
}
