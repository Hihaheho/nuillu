use anyhow::Result;

use crate::AttentionSchemaModule;

#[derive(Debug, Default)]
pub struct NextBatch {
    pub(crate) memo_logs: Vec<nuillu_module::MemoLogRecord>,
    pub(crate) cognition_log: Vec<nuillu_module::CognitionLogEntryRecord>,
}

impl NextBatch {
    pub(crate) fn has_updates(&self) -> bool {
        !self.memo_logs.is_empty() || !self.cognition_log.is_empty()
    }
}

impl AttentionSchemaModule {
    pub(crate) async fn next_batch(&mut self) -> Result<NextBatch> {
        self.await_first_batch().await?;
        self.collect_ready_events_into_batch()?;
        Ok(NextBatch {
            memo_logs: self.blackboard.unread_cognitive_memo_logs().await,
            cognition_log: self.cognition_log.unread_events().await,
        })
    }

    async fn await_first_batch(&mut self) -> Result<()> {
        tokio::select! {
            update = self.memo_updates.next_item() => {
                let _ = update?;
            }
            update = self.cognition_updates.next_item() => {
                let _ = update?;
            }
        }
        Ok(())
    }

    fn collect_ready_events_into_batch(&mut self) -> Result<()> {
        let _ = self.memo_updates.take_ready_items()?;
        let _ = self.cognition_updates.take_ready_items()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn batch_has_updates_when_any_delta_is_present() {
        let empty = NextBatch::default();
        let source = nuillu_types::ModuleInstanceId::new(
            nuillu_types::builtin::cognition_gate(),
            nuillu_types::ReplicaIndex::ZERO,
        );
        let with_cognition = NextBatch {
            memo_logs: Vec::new(),
            cognition_log: vec![nuillu_module::CognitionLogEntryRecord {
                index: 0,
                source: source.clone(),
                entry: nuillu_blackboard::CognitionLogEntry {
                    at: chrono::Utc::now(),
                    text: "attention shifted".into(),
                    origin: nuillu_blackboard::CognitionLogOrigin::direct(source),
                },
            }],
        };

        assert!(!empty.has_updates());
        assert!(with_cognition.has_updates());
    }
}
