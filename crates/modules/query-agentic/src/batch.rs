use anyhow::Result;

use crate::QueryAgenticModule;

#[derive(Debug, Default)]
pub struct NextBatch {
    pub(crate) guidance: bool,
}

impl NextBatch {
    fn guidance() -> Self {
        Self { guidance: true }
    }

    fn mark_guidance(&mut self) {
        self.guidance = true;
    }
}

impl QueryAgenticModule {
    pub(crate) async fn next_batch(&mut self) -> Result<NextBatch> {
        let mut batch = self.await_first_batch().await?;
        self.collect_ready_events_into_batch(&mut batch)?;
        Ok(batch)
    }

    async fn await_first_batch(&mut self) -> Result<NextBatch> {
        let batch = tokio::select! {
            update = self.allocation_updates.next_item() => {
                let _ = update?;
                NextBatch::guidance()
            }
        };
        Ok(batch)
    }

    fn collect_ready_events_into_batch(&mut self, batch: &mut NextBatch) -> Result<()> {
        if !self.allocation_updates.take_ready_items()?.items.is_empty() {
            batch.mark_guidance();
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn guidance_flag_can_be_marked() {
        let mut batch = NextBatch::default();
        batch.mark_guidance();

        assert!(batch.guidance);
    }
}
