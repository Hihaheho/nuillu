use std::cell::Cell;
use std::rc::Rc;
use std::sync::Arc;

use nuillu_blackboard::{Blackboard, BlackboardCommand, UtteranceProgress};
use nuillu_types::ModuleInstanceId;

use crate::ports::{Clock, Utterance, UtteranceDelta, UtteranceSink};

/// Write capability for emitting user-visible utterances.
///
/// Stamps `emitted_at` from the injected [`Clock`] so the timestamp is
/// testable without wall-clock dependence. Best-effort: failures are logged
/// and dropped so a slow sink does not stall the speak module.
pub struct UtteranceWriter {
    owner: ModuleInstanceId,
    blackboard: Blackboard,
    sink: Arc<dyn UtteranceSink>,
    clock: Arc<dyn Clock>,
    next_generation: Rc<Cell<u64>>,
}

impl UtteranceWriter {
    pub(crate) fn new(
        owner: ModuleInstanceId,
        blackboard: Blackboard,
        sink: Arc<dyn UtteranceSink>,
        clock: Arc<dyn Clock>,
    ) -> Self {
        Self {
            owner,
            blackboard,
            sink,
            clock,
            next_generation: Rc::new(Cell::new(0)),
        }
    }

    pub fn next_generation_id(&self) -> u64 {
        let id = self.next_generation.get();
        self.next_generation.set(id.wrapping_add(1));
        id
    }

    pub async fn emit(&self, text: impl Into<String>) {
        let utterance = Utterance {
            sender: self.owner.clone(),
            text: text.into(),
            emitted_at: self.clock.now(),
        };
        if let Err(e) = self.sink.on_complete(utterance).await {
            tracing::warn!(error = ?e, "utterance sink complete failed");
        }
    }

    pub async fn emit_delta(&self, generation_id: u64, sequence: u32, delta: impl Into<String>) {
        let delta = UtteranceDelta {
            sender: self.owner.clone(),
            generation_id,
            sequence,
            delta: delta.into(),
        };
        if let Err(e) = self.sink.on_delta(delta).await {
            tracing::warn!(error = ?e, "utterance sink delta failed");
        }
    }

    pub async fn record_progress(&self, progress: UtteranceProgress) {
        self.blackboard
            .apply(BlackboardCommand::SetUtteranceProgress {
                owner: self.owner.clone(),
                progress,
            })
            .await;
    }
}
