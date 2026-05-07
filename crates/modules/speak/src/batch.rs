use anyhow::Result;

use crate::SpeakModule;

impl SpeakModule {
    pub(crate) async fn next_batch(&mut self) -> Result<()> {
        if let Some(periodic) = self.periodic.as_mut() {
            tokio::select! {
                tick = periodic.next_tick() => {
                    tick?;
                }
                update = self.updates.next_item() => {
                    let _ = update?;
                }
            }

            let _ = self.updates.take_ready_items()?;
            let _ = periodic.take_ready_ticks()?;
            self.gate.block().await;
            let _ = self.updates.take_ready_items()?;
            let _ = periodic.take_ready_ticks()?;
        } else {
            let _ = self.updates.next_item().await?;
            let _ = self.updates.take_ready_items()?;
            self.gate.block().await;
            let _ = self.updates.take_ready_items()?;
        }

        Ok(())
    }
}
