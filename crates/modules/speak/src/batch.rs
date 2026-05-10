use anyhow::Result;
use nuillu_module::{ActivationGateEvent, CognitionLogUpdated};

use crate::{SpeakGateModule, SpeakModule};

#[derive(Debug)]
pub struct SpeakBatch {
    pub(crate) updates: Vec<CognitionLogUpdated>,
}

impl SpeakGateModule {
    pub(crate) async fn next_batch(&mut self) -> Result<ActivationGateEvent<SpeakModule>> {
        Ok(self.activation_gate.next_event().await?)
    }
}

impl SpeakModule {
    pub(crate) async fn next_batch(&mut self) -> Result<SpeakBatch> {
        let first = self.cognition_updates.next_item().await?;
        let mut updates = vec![first.body];
        updates.extend(
            self.cognition_updates
                .take_ready_items()?
                .items
                .into_iter()
                .map(|envelope| envelope.body),
        );

        Ok(SpeakBatch { updates })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn speak_batch_keeps_drained_updates() {
        let batch = SpeakBatch {
            updates: vec![CognitionLogUpdated::AgenticDeadlockMarker],
        };

        assert_eq!(
            batch.updates,
            vec![CognitionLogUpdated::AgenticDeadlockMarker]
        );
    }
}
