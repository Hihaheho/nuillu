use std::sync::Arc;

use nuillu_blackboard::{AttentionStreamEvent, Blackboard, BlackboardCommand};
use nuillu_types::ModuleId;

use crate::ports::{AttentionRepository, Clock};
use crate::{AttentionStreamUpdated, AttentionStreamUpdatedMailbox};

/// Append-only access to the cognitive attention stream.
///
/// The factory does not enforce uniqueness: capabilities are non-exclusive.
/// By boot-time wiring convention only the summarize module receives this
/// handle, but multiple writers are structurally permitted.
pub struct AttentionWriter {
    owner: ModuleId,
    blackboard: Blackboard,
    attention_port: Arc<dyn AttentionRepository>,
    updates: AttentionStreamUpdatedMailbox,
    clock: Arc<dyn Clock>,
}

impl AttentionWriter {
    pub(crate) fn new(
        owner: ModuleId,
        blackboard: Blackboard,
        attention_port: Arc<dyn AttentionRepository>,
        updates: AttentionStreamUpdatedMailbox,
        clock: Arc<dyn Clock>,
    ) -> Self {
        Self {
            owner,
            blackboard,
            attention_port,
            updates,
            clock,
        }
    }

    pub fn owner(&self) -> &ModuleId {
        &self.owner
    }

    /// Append an event to the attention stream, persist it via the port,
    /// and notify the attention controller.
    ///
    /// Each step is best-effort — failures are traced and dropped so a
    /// slow port or a missing controller does not stall the cognitive
    /// pipeline.
    pub async fn append(&self, text: impl Into<String>) {
        let event = AttentionStreamEvent {
            at: self.clock.now(),
            text: text.into(),
        };

        self.blackboard
            .apply(BlackboardCommand::AppendAttentionStream(event.clone()))
            .await;

        if let Err(e) = self.attention_port.append(event.clone()).await {
            tracing::warn!(error = ?e, "attention port append failed");
        }

        if self.updates.publish(AttentionStreamUpdated).is_err() {
            tracing::trace!("attention-stream update had no active subscribers",);
        }
    }
}
