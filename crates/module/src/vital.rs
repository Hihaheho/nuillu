use std::rc::Rc;

use nuillu_blackboard::{Blackboard, BlackboardCommand, VitalPatch};
use nuillu_types::ModuleInstanceId;

use crate::ports::Clock;
use crate::{VitalUpdated, VitalUpdatedMailbox};

pub struct VitalWriter {
    owner: ModuleInstanceId,
    blackboard: Blackboard,
    updates: VitalUpdatedMailbox,
    clock: Rc<dyn Clock>,
}

impl VitalWriter {
    pub(crate) fn new(
        owner: ModuleInstanceId,
        blackboard: Blackboard,
        updates: VitalUpdatedMailbox,
        clock: Rc<dyn Clock>,
    ) -> Self {
        Self {
            owner,
            blackboard,
            updates,
            clock,
        }
    }

    pub async fn update(&self, patch: VitalPatch) {
        self.blackboard
            .apply(BlackboardCommand::UpdateVital {
                patch,
                now: self.clock.now(),
            })
            .await;

        if self.updates.publish(VitalUpdated).await.is_err() {
            tracing::trace!(
                owner = %self.owner,
                "vital update had no active subscribers"
            );
        }
    }
}
