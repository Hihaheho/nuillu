use std::sync::Arc;

use nuillu_types::ModuleId;

use crate::ports::{Clock, Utterance, UtteranceSink};

/// Write capability for emitting user-visible utterances.
///
/// Stamps `emitted_at` from the injected [`Clock`] so the timestamp is
/// testable without wall-clock dependence. Best-effort: failures are logged
/// and dropped so a slow sink does not stall the speak module.
pub struct UtteranceWriter {
    owner: ModuleId,
    sink: Arc<dyn UtteranceSink>,
    clock: Arc<dyn Clock>,
}

impl UtteranceWriter {
    pub(crate) fn new(
        owner: ModuleId,
        sink: Arc<dyn UtteranceSink>,
        clock: Arc<dyn Clock>,
    ) -> Self {
        Self { owner, sink, clock }
    }

    pub fn owner(&self) -> &ModuleId {
        &self.owner
    }

    pub async fn emit(&self, text: impl Into<String>) {
        let utterance = Utterance {
            text: text.into(),
            emitted_at: self.clock.now(),
        };
        if let Err(e) = self.sink.append(utterance).await {
            tracing::warn!(error = ?e, "utterance sink append failed");
        }
    }
}
