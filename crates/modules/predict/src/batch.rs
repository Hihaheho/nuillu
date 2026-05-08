use anyhow::Result;

use crate::PredictModule;

impl PredictModule {
    pub(crate) async fn next_batch(&mut self) -> Result<()> {
        let _ = self.updates.next_item().await?;
        let _ = self.updates.take_ready_items()?;
        Ok(())
    }
}
