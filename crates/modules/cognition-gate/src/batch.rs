use anyhow::Result;

use crate::CognitionGateModule;

impl CognitionGateModule {
    pub(crate) async fn next_batch(&mut self) -> Result<()> {
        let _ = self.memo_updates.next_item().await?;
        let _ = self.memo_updates.take_ready_items()?;
        Ok(())
    }
}
