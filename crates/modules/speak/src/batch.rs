use anyhow::Result;
use nuillu_module::SpeakRequest;

use crate::{SpeakGateModule, SpeakModule};

#[derive(Debug)]
pub struct NextBatch {
    pub(crate) request: SpeakRequest,
}

impl SpeakGateModule {
    pub(crate) async fn next_batch(&mut self) -> Result<()> {
        let _ = self.cognition_updates.next_item().await?;
        let _ = self.cognition_updates.take_ready_items()?;
        Ok(())
    }
}

impl SpeakModule {
    pub(crate) async fn next_batch(&mut self) -> Result<NextBatch> {
        let first = self.requests.next_item().await?;
        let mut request = first.body;
        for ready in self.requests.take_ready_items()?.items {
            request = ready.body;
        }

        Ok(NextBatch { request })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn latest_ready_request_wins() {
        let mut batch = NextBatch {
            request: SpeakRequest::new("Koro", "first", "old"),
        };
        batch.request = SpeakRequest::new("Pibi", "second", "new");

        assert_eq!(batch.request.target, "Pibi");
        assert_eq!(batch.request.generation_hint, "second");
        assert_eq!(batch.request.rationale, "new");
    }
}
