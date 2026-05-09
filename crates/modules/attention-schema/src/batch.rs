use anyhow::Result;

use crate::AttentionSchemaModule;

#[derive(Debug, Default)]
pub struct NextBatch {
    pub(crate) update_model: bool,
}

impl NextBatch {
    fn model_update() -> Self {
        Self { update_model: true }
    }

    fn mark_model_update(&mut self) {
        self.update_model = true;
    }
}

impl AttentionSchemaModule {
    pub(crate) async fn next_batch(&mut self) -> Result<NextBatch> {
        let mut batch = self.await_first_batch().await?;
        self.collect_ready_events_into_batch(&mut batch)?;
        Ok(batch)
    }

    async fn await_first_batch(&mut self) -> Result<NextBatch> {
        tokio::select! {
            update = self.memo_updates.next_item() => {
                let _ = update?;
            }
            update = self.allocation_updates.next_item() => {
                let _ = update?;
            }
            update = self.cognition_updates.next_item() => {
                let _ = update?;
            }
        }
        Ok(NextBatch::model_update())
    }

    fn collect_ready_events_into_batch(&mut self, batch: &mut NextBatch) -> Result<()> {
        if !self.memo_updates.take_ready_items()?.items.is_empty() {
            batch.mark_model_update();
        }
        if !self.allocation_updates.take_ready_items()?.items.is_empty() {
            batch.mark_model_update();
        }
        if !self.cognition_updates.take_ready_items()?.items.is_empty() {
            batch.mark_model_update();
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn multiple_model_inputs_collapse_to_one_model_update() {
        let mut batch = NextBatch::model_update();
        batch.mark_model_update();

        assert!(batch.update_model);
    }
}
