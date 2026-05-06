use anyhow::Result;

use crate::PredictModule;

impl PredictModule {
    pub(crate) async fn next_batch(&mut self) -> Result<()> {
        tokio::select! {
            tick = self.periodic.next_tick() => {
                tick?;
            }
            update = self.updates.next_item() => {
                let _ = update?;
            }
        }

        let _ = self.updates.take_ready_items()?;
        let _ = self.periodic.take_ready_ticks()?;
        Ok(())
    }
}
