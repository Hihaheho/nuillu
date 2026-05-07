use anyhow::Result;

use crate::AttentionControllerModule;

impl AttentionControllerModule {
    pub(crate) async fn next_batch(&mut self) -> Result<()> {
        let updates = &mut self.updates;
        if let Some(periodic) = self.periodic.as_mut() {
            tokio::select! {
                tick = periodic.next_tick() => {
                    tick?;
                }
                update = updates.next_item() => {
                    let _ = update?;
                }
            }

            let _ = updates.take_ready_items()?;
            let _ = periodic.take_ready_ticks()?;
            self.gate.block().await;
            let _ = updates.take_ready_items()?;
            let _ = periodic.take_ready_ticks()?;
        } else {
            let _ = updates.next_item().await?;
            let _ = updates.take_ready_items()?;
            self.gate.block().await;
            let _ = updates.take_ready_items()?;
        }

        Ok(())
    }
}
