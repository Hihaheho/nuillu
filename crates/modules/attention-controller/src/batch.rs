use anyhow::Result;

use crate::AttentionControllerModule;

impl AttentionControllerModule {
    pub(crate) async fn next_batch(&mut self) -> Result<()> {
        if self.initial_batch_pending {
            self.initial_batch_pending = false;
            return Ok(());
        }
        let _ = self.updates.next_item().await?;
        let _ = self.updates.take_ready_items()?;
        Ok(())
    }
}
