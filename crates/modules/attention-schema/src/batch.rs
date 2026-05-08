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
        self.updates.next_item().await?;
        Ok(NextBatch::model_update())
    }

    fn collect_ready_events_into_batch(&mut self, batch: &mut NextBatch) -> Result<()> {
        if !self.updates.take_ready_items()?.items.is_empty() {
            batch.mark_model_update();
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn multiple_attention_updates_collapse_to_one_model_update() {
        let mut batch = NextBatch::model_update();
        batch.mark_model_update();

        assert!(batch.update_model);
    }
}
