use anyhow::Result;

use crate::InterpreterModule;

#[derive(Debug, Default)]
pub struct InterpreterBatch {
    pub(crate) cognition_log: Vec<nuillu_module::CognitionLogEntryRecord>,
}

impl InterpreterBatch {
    pub(crate) fn has_updates(&self) -> bool {
        !self.cognition_log.is_empty()
    }
}

impl InterpreterModule {
    pub(crate) async fn next_batch(&mut self) -> Result<InterpreterBatch> {
        let _ = self.cognition_updates.next_item().await?;
        let _ = self.cognition_updates.take_ready_items()?;
        Ok(InterpreterBatch {
            cognition_log: self.cognition_log.unread_events().await,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn batch_has_updates_when_cognition_delta_is_present() {
        let empty = InterpreterBatch::default();
        let with_cognition = InterpreterBatch {
            cognition_log: vec![nuillu_module::CognitionLogEntryRecord {
                index: 0,
                source: nuillu_types::ModuleInstanceId::new(
                    nuillu_types::builtin::cognition_gate(),
                    nuillu_types::ReplicaIndex::ZERO,
                ),
                entry: nuillu_blackboard::CognitionLogEntry {
                    at: chrono::Utc::now(),
                    text: "Alice asks for an interesting story.".into(),
                },
            }],
        };

        assert!(!empty.has_updates());
        assert!(with_cognition.has_updates());
    }
}
