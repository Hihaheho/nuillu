use std::rc::Rc;

use nuillu_blackboard::{Blackboard, BlackboardCommand, InteroceptivePatch};
use nuillu_types::ModuleInstanceId;

use crate::ports::Clock;
use crate::{InteroceptiveUpdated, InteroceptiveUpdatedMailbox};

pub struct InteroceptiveWriter {
    owner: ModuleInstanceId,
    blackboard: Blackboard,
    updates: InteroceptiveUpdatedMailbox,
    clock: Rc<dyn Clock>,
}

impl InteroceptiveWriter {
    pub(crate) fn new(
        owner: ModuleInstanceId,
        blackboard: Blackboard,
        updates: InteroceptiveUpdatedMailbox,
        clock: Rc<dyn Clock>,
    ) -> Self {
        Self {
            owner,
            blackboard,
            updates,
            clock,
        }
    }

    pub async fn update(&self, patch: InteroceptivePatch) {
        self.blackboard
            .apply(BlackboardCommand::UpdateInteroceptive {
                patch,
                now: self.clock.now(),
            })
            .await;

        if self.updates.publish(InteroceptiveUpdated).await.is_err() {
            tracing::trace!(
                owner = %self.owner,
                "interoception update had no active subscribers"
            );
        }
    }
}
