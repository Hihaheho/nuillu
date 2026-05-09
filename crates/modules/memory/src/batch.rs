use anyhow::Result;
use nuillu_module::MemoryRequest;

use crate::MemoryModule;

#[derive(Debug, Default)]
pub struct NextBatch {
    pub(crate) requests: Vec<MemoryRequest>,
    pub(crate) cognition_updated: bool,
}

impl NextBatch {
    fn cognition_log_update() -> Self {
        Self {
            requests: Vec::new(),
            cognition_updated: true,
        }
    }

    fn request(request: MemoryRequest) -> Option<Self> {
        Some(Self {
            requests: vec![Self::accepted_request(request)?],
            cognition_updated: false,
        })
    }

    fn mark_cognition_updated(&mut self) {
        self.cognition_updated = true;
    }

    fn accepted_request(request: MemoryRequest) -> Option<MemoryRequest> {
        if request.content.trim().is_empty() {
            tracing::warn!("memory request ignored because content is empty");
            return None;
        }
        Some(request)
    }

    fn push_request(&mut self, request: MemoryRequest) {
        if let Some(request) = Self::accepted_request(request) {
            self.requests.push(request);
        }
    }
}

impl MemoryModule {
    pub(crate) async fn next_batch(&mut self) -> Result<NextBatch> {
        let mut batch = self.await_first_batch().await?;
        self.collect_ready_events_into_batch(&mut batch)?;
        Ok(batch)
    }

    async fn await_first_batch(&mut self) -> Result<NextBatch> {
        loop {
            let batch = tokio::select! {
                update = self.cognition_updates.next_item() => {
                    let _ = update?;
                    NextBatch::cognition_log_update()
                }
                request = self.requests.next_item() => {
                    let envelope = request?;
                    let Some(batch) = NextBatch::request(envelope.body) else {
                        continue;
                    };
                    batch
                }
            };
            return Ok(batch);
        }
    }

    fn collect_ready_events_into_batch(&mut self, batch: &mut NextBatch) -> Result<()> {
        let ready_requests = self.requests.take_ready_items()?;
        for envelope in ready_requests.items {
            batch.push_request(envelope.body);
        }

        if !self.cognition_updates.take_ready_items()?.items.is_empty() {
            batch.mark_cognition_updated();
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use nuillu_module::{MemoryImportance, MemoryRequest};

    use super::*;

    fn request(content: &str) -> MemoryRequest {
        MemoryRequest {
            content: content.into(),
            importance: MemoryImportance::Normal,
            reason: "test".into(),
        }
    }

    #[test]
    fn empty_memory_requests_are_not_actionable_without_cognition_log_work() {
        assert!(NextBatch::request(request("  ")).is_none());
    }

    #[test]
    fn request_batch_preserves_candidate_order() {
        let mut batch = NextBatch::request(request("first")).unwrap();
        batch.push_request(request("second"));

        let contents = batch
            .requests
            .into_iter()
            .map(|request| request.content)
            .collect::<Vec<_>>();
        assert_eq!(contents, vec!["first", "second"]);
    }

    #[test]
    fn cognition_log_update_flag_survives_empty_request_filtering() {
        let mut batch = NextBatch::cognition_log_update();
        batch.push_request(request(""));
        batch.mark_cognition_updated();

        assert!(batch.cognition_updated);
        assert!(batch.requests.is_empty());
    }
}
