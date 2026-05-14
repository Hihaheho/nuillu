use std::rc::Rc;

use nuillu_blackboard::{Blackboard, BlackboardCommand, CognitionLogEntry};
use nuillu_types::ModuleInstanceId;

use crate::ports::{Clock, CognitionLogRepository};
use crate::{CognitionLogUpdated, CognitionLogUpdatedMailbox};

/// Append-only access to the cognition log.
///
/// Capability issuers do not enforce uniqueness: capabilities are non-exclusive.
/// Boot-time wiring decides which modules receive this handle.
pub struct CognitionWriter {
    owner: ModuleInstanceId,
    blackboard: Blackboard,
    cognition_log_port: Rc<dyn CognitionLogRepository>,
    updates: CognitionLogUpdatedMailbox,
    clock: Rc<dyn Clock>,
}

impl CognitionWriter {
    pub(crate) fn new(
        owner: ModuleInstanceId,
        blackboard: Blackboard,
        cognition_log_port: Rc<dyn CognitionLogRepository>,
        updates: CognitionLogUpdatedMailbox,
        clock: Rc<dyn Clock>,
    ) -> Self {
        Self {
            owner,
            blackboard,
            cognition_log_port,
            updates,
            clock,
        }
    }

    /// Append an event to the cognition log, persist it via the port,
    /// and notify cognition-log consumers.
    ///
    /// Each step is best-effort — failures are traced and dropped so a
    /// slow port or a missing controller does not stall the cognitive
    /// pipeline.
    pub async fn append(&self, text: impl Into<String>) {
        let entry = CognitionLogEntry {
            at: self.clock.now(),
            text: text.into(),
        };

        self.blackboard
            .apply(BlackboardCommand::AppendCognitionLog {
                source: self.owner.clone(),
                entry: entry.clone(),
            })
            .await;

        if let Err(e) = self
            .cognition_log_port
            .append(self.owner.clone(), entry.clone())
            .await
        {
            tracing::warn!(error = ?e, "cognition log port append failed");
        }

        if self
            .updates
            .publish(CognitionLogUpdated::EntryAppended {
                source: self.owner.clone(),
            })
            .await
            .is_err()
        {
            tracing::trace!("cognition-log update had no active subscribers",);
        }
    }
}
