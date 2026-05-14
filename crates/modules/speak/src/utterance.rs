//! `UtteranceSink` port + the `UtteranceWriter` capability handle.
//!
//! Only the speak module emits user-visible utterances, so the trait and the
//! owner-stamped writer that wraps it live in this crate. Adapters provide
//! concrete implementations of `UtteranceSink` and the host injects them into
//! the speak registration closure.

use std::cell::Cell;
use std::rc::Rc;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use nuillu_blackboard::{Blackboard, BlackboardCommand, UtteranceProgress};
use nuillu_module::ports::{Clock, PortError};
use nuillu_types::ModuleInstanceId;

/// A single user-visible utterance emitted by the speak module.
pub struct Utterance {
    pub sender: ModuleInstanceId,
    pub target: String,
    pub text: String,
    pub emitted_at: DateTime<Utc>,
}

/// A streaming chunk emitted while a user-visible utterance is being generated.
///
/// A generation may be interrupted and resumed. Resumed chunks keep the same
/// `generation_id` and continue the `sequence`; sinks should append them to
/// the chunks they already accepted instead of discarding partial text.
pub struct UtteranceDelta {
    pub sender: ModuleInstanceId,
    pub target: String,
    pub generation_id: u64,
    pub sequence: u32,
    pub delta: String,
}

/// Append-only sink for utterances. Adapters provide concrete implementations.
#[async_trait(?Send)]
pub trait UtteranceSink {
    async fn on_complete(&self, utterance: Utterance) -> Result<(), PortError>;

    async fn on_delta(&self, _delta: UtteranceDelta) -> Result<(), PortError> {
        Ok(())
    }
}

/// Utterance sink that drops every event.
#[derive(Debug, Default)]
pub struct NoopUtteranceSink;

#[async_trait(?Send)]
impl UtteranceSink for NoopUtteranceSink {
    async fn on_complete(&self, _utterance: Utterance) -> Result<(), PortError> {
        Ok(())
    }
}

fn normalized_target(target: impl Into<String>) -> Option<String> {
    let target = target.into().trim().to_owned();
    (!target.is_empty()).then_some(target)
}

/// Write capability for emitting user-visible utterances.
///
/// Stamps `emitted_at` from the injected [`Clock`] so the timestamp is
/// testable without wall-clock dependence. Best-effort: failures are logged
/// and dropped so a slow sink does not stall the speak module.
pub struct UtteranceWriter {
    owner: ModuleInstanceId,
    blackboard: Blackboard,
    sink: Rc<dyn UtteranceSink>,
    clock: Rc<dyn Clock>,
    next_generation: Rc<Cell<u64>>,
}

impl UtteranceWriter {
    pub fn new(
        owner: ModuleInstanceId,
        blackboard: Blackboard,
        sink: Rc<dyn UtteranceSink>,
        clock: Rc<dyn Clock>,
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

    pub async fn emit(&self, target: impl Into<String>, text: impl Into<String>) {
        let Some(target) = normalized_target(target) else {
            tracing::warn!("utterance sink complete skipped empty target");
            return;
        };
        let utterance = Utterance {
            sender: self.owner.clone(),
            target,
            text: text.into(),
            emitted_at: self.clock.now(),
        };
        if let Err(e) = self.sink.on_complete(utterance).await {
            tracing::warn!(error = ?e, "utterance sink complete failed");
        }
    }

    pub async fn emit_delta(
        &self,
        target: impl Into<String>,
        generation_id: u64,
        sequence: u32,
        delta: impl Into<String>,
    ) {
        let Some(target) = normalized_target(target) else {
            tracing::warn!("utterance sink delta skipped empty target");
            return;
        };
        let delta = UtteranceDelta {
            sender: self.owner.clone(),
            target,
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
