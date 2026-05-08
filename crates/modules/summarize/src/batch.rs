use anyhow::Result;

use crate::SummarizeModule;

impl SummarizeModule {
    pub(crate) async fn next_batch(&mut self) -> Result<()> {
        self.updates.next_item().await?;
        let _ = self.updates.take_ready_items()?;
        self.gate.block().await;
        let _ = self.updates.take_ready_items()?;
        Ok(())
    }
}
