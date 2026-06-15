use anyhow::Result;
use nuillu_blackboard::MemoLogRecord;

use crate::CognitionGateModule;

#[derive(Debug, Default)]
pub struct NextBatch {
    pub(crate) memo_logs: Vec<MemoLogRecord>,
}

impl CognitionGateModule {
    pub(crate) async fn next_batch(&mut self) -> Result<NextBatch> {
        loop {
            let _ = self.memo_updates.next_item().await?;
            let _ = self.memo_updates.take_ready_items()?;
            let memo_logs = self.blackboard.unread_cognitive_memo_logs().await;
            if !memo_logs.is_empty() {
                return Ok(NextBatch { memo_logs });
            }
        }
    }
}
