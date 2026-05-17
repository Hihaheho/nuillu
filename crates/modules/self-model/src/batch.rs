use anyhow::Result;

use crate::SelfModelModule;

#[derive(Debug, Default)]
pub struct NextBatch {
    pub(crate) cognition_updated: bool,
}

impl NextBatch {
    fn cognition_updated() -> Self {
        Self {
            cognition_updated: true,
        }
    }
}

impl SelfModelModule {
    pub(crate) async fn next_batch(&mut self) -> Result<NextBatch> {
        let mut batch = self.await_first_batch().await?;
        self.collect_ready_events_into_batch(&mut batch)?;
        Ok(batch)
    }

    async fn await_first_batch(&mut self) -> Result<NextBatch> {
        let _ = self.cognition_updates.next_item().await?;
        Ok(NextBatch::cognition_updated())
    }

    fn collect_ready_events_into_batch(&mut self, batch: &mut NextBatch) -> Result<()> {
        if !self.cognition_updates.take_ready_items()?.items.is_empty() {
            batch.cognition_updated = true;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cognition_update_batch_marks_self_model_work() {
        let batch = NextBatch::cognition_updated();

        assert!(batch.cognition_updated);
    }
}
