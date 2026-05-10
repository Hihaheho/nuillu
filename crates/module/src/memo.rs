use std::sync::Arc;

use nuillu_blackboard::Blackboard;
use nuillu_types::ModuleInstanceId;

use crate::ports::Clock;
use crate::runtime_events::RuntimeEventEmitter;
use crate::{MemoUpdated, MemoUpdatedMailbox};

/// Write handle for the activating module's own memo log.
///
/// The owner identity is stamped at construction time by
/// [`ModuleCapabilityFactory`](crate::ModuleCapabilityFactory); the module cannot
/// change it. A module that does not hold a `Memo` has no memo log allocated
/// on the blackboard at all (queue creation is lazy and only `Memo::write`
/// ever inserts one).
#[derive(Clone)]
pub struct Memo {
    owner: ModuleInstanceId,
    blackboard: Blackboard,
    updates: MemoUpdatedMailbox,
    clock: Arc<dyn Clock>,
    events: RuntimeEventEmitter,
}

impl Memo {
    pub(crate) fn new(
        owner: ModuleInstanceId,
        blackboard: Blackboard,
        updates: MemoUpdatedMailbox,
        clock: Arc<dyn Clock>,
        events: RuntimeEventEmitter,
    ) -> Self {
        Self {
            owner,
            blackboard,
            updates,
            clock,
            events,
        }
    }

    /// Append a new owner memo item. Allocates the queue on first call.
    pub async fn write(&self, memo: impl Into<String>) {
        let memo = memo.into();
        let char_count = memo.chars().count();
        let record = self
            .blackboard
            .update_memo(self.owner.clone(), memo, self.clock.now())
            .await;
        self.events
            .memo_updated(self.owner.clone(), char_count)
            .await;
        if self
            .updates
            .publish(MemoUpdated {
                owner: self.owner.clone(),
                index: record.index,
            })
            .await
            .is_err()
        {
            tracing::trace!("memo update had no active subscribers");
        }
    }
}
