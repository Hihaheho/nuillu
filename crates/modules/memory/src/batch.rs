use anyhow::Result;

use crate::MemoryModule;

#[derive(Debug, Default)]
pub struct NextBatch {
    pub(crate) allocation_updated: bool,
    pub(crate) cognition_updated: bool,
}

impl NextBatch {
    fn allocation_update() -> Self {
        Self {
            allocation_updated: true,
            cognition_updated: false,
        }
    }

    fn cognition_log_update() -> Self {
        Self {
            allocation_updated: false,
            cognition_updated: true,
        }
    }

    fn mark_cognition_updated(&mut self) {
        self.cognition_updated = true;
    }

    fn mark_allocation_updated(&mut self) {
        self.allocation_updated = true;
    }
}

impl MemoryModule {
    pub(crate) async fn next_batch(&mut self) -> Result<NextBatch> {
        let mut batch = self.await_first_batch().await?;
        self.collect_ready_events_into_batch(&mut batch)?;
        Ok(batch)
    }

    async fn await_first_batch(&mut self) -> Result<NextBatch> {
        let batch = tokio::select! {
            update = self.allocation_updates.next_item() => {
                let _ = update?;
                NextBatch::allocation_update()
            }
            update = self.cognition_updates.next_item() => {
                let _ = update?;
                NextBatch::cognition_log_update()
            }
        };
        Ok(batch)
    }

    fn collect_ready_events_into_batch(&mut self, batch: &mut NextBatch) -> Result<()> {
        if !self.allocation_updates.take_ready_items()?.items.is_empty() {
            batch.mark_allocation_updated();
        }

        if !self.cognition_updates.take_ready_items()?.items.is_empty() {
            batch.mark_cognition_updated();
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cognition_log_update_flag_can_share_batch_with_allocation_update() {
        let mut batch = NextBatch::allocation_update();
        batch.mark_cognition_updated();

        assert!(batch.allocation_updated);
        assert!(batch.cognition_updated);
    }
}
