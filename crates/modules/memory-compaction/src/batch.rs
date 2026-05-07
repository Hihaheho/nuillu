use anyhow::Result;

use crate::MemoryCompactionModule;

impl MemoryCompactionModule {
    pub(crate) async fn next_batch(&mut self) -> Result<()> {
        self.periodic.next_tick().await?;
        let _ = self.periodic.take_ready_ticks()?;
        self.gate.block().await;
        let _ = self.periodic.take_ready_ticks()?;
        Ok(())
    }
}
