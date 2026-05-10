use anyhow::Result;

use crate::SelfModelModule;

#[derive(Debug, Default)]
pub struct NextBatch {
    pub(crate) allocation_updated: bool,
}

impl NextBatch {
    fn allocation_updated() -> Self {
        Self {
            allocation_updated: true,
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
        let _ = self.allocation_updates.next_item().await?;
        Ok(NextBatch::allocation_updated())
    }

    fn collect_ready_events_into_batch(&mut self, batch: &mut NextBatch) -> Result<()> {
        if !self.allocation_updates.take_ready_items()?.items.is_empty() {
            batch.allocation_updated = true;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allocation_update_batch_marks_guidance_work() {
        let batch = NextBatch::allocation_updated();

        assert!(batch.allocation_updated);
    }
}
