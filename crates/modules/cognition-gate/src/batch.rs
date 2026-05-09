use anyhow::Result;

use crate::CognitionGateModule;

impl CognitionGateModule {
    pub(crate) async fn next_batch(&mut self) -> Result<()> {
        tokio::select! {
            update = self.memo_updates.next_item() => {
                let _ = update?;
            }
            update = self.allocation_updates.next_item() => {
                let _ = update?;
            }
        }
        let _ = self.memo_updates.take_ready_items()?;
        let _ = self.allocation_updates.take_ready_items()?;
        Ok(())
    }
}
