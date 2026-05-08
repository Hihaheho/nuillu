use anyhow::Result;

use crate::MemoryCompactionModule;

impl MemoryCompactionModule {
    pub(crate) async fn next_batch(&mut self) -> Result<()> {
        let _ = self.allocation_updates.next_item().await?;
        let _ = self.allocation_updates.take_ready_items()?;
        Ok(())
    }
}
