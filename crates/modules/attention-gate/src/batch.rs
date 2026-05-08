use anyhow::Result;

use crate::AttentionGateModule;

impl AttentionGateModule {
    pub(crate) async fn next_batch(&mut self) -> Result<()> {
        self.updates.next_item().await?;
        let _ = self.updates.take_ready_items()?;
        Ok(())
    }
}
