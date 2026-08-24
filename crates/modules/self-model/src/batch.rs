use anyhow::Result;

use crate::SelfModelModule;

#[derive(Debug, Default)]
pub struct NextBatch {
    pub(crate) memo_updated: bool,
}

impl NextBatch {
    fn memo_updated() -> Self {
        Self { memo_updated: true }
    }
}

impl SelfModelModule {
    pub(crate) async fn next_batch(&mut self) -> Result<NextBatch> {
        let mut batch = self.await_first_batch().await?;
        self.collect_ready_events_into_batch(&mut batch)?;
        Ok(batch)
    }

    async fn await_first_batch(&mut self) -> Result<NextBatch> {
        let _ = self.memo_updates.next_item().await?;
        Ok(NextBatch::memo_updated())
    }

    fn collect_ready_events_into_batch(&mut self, batch: &mut NextBatch) -> Result<()> {
        if !self.memo_updates.take_ready_items()?.items.is_empty() {
            batch.memo_updated = true;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn memo_update_batch_marks_self_model_work() {
        let batch = NextBatch::memo_updated();

        assert!(batch.memo_updated);
    }
}
