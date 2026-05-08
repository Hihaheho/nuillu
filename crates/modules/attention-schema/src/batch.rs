use anyhow::Result;
use nuillu_module::SelfModelRequest;

use crate::AttentionSchemaModule;

#[derive(Debug, Default)]
pub(crate) struct NextBatch {
    pub(crate) update_model: bool,
    pub(crate) requests: Vec<SelfModelRequest>,
}

impl NextBatch {
    fn model_update() -> Self {
        Self {
            update_model: true,
            requests: Vec::new(),
        }
    }

    fn request(request: SelfModelRequest) -> Self {
        Self {
            update_model: false,
            requests: vec![request],
        }
    }

    fn mark_model_update(&mut self) {
        self.update_model = true;
    }

    fn extend_requests(&mut self, requests: impl IntoIterator<Item = SelfModelRequest>) {
        self.requests.extend(requests);
    }
}

impl AttentionSchemaModule {
    pub(crate) async fn next_batch(&mut self) -> Result<NextBatch> {
        let mut batch = self.await_first_batch().await?;
        self.collect_ready_events_into_batch(&mut batch)?;
        self.gate.block().await;
        self.collect_ready_events_into_batch(&mut batch)?;
        Ok(batch)
    }

    async fn await_first_batch(&mut self) -> Result<NextBatch> {
        let batch = tokio::select! {
            update = self.allocation_updates.next_item() => {
                let _ = update?;
                NextBatch::model_update()
            }
            request = self.self_model.next_item() => {
                let envelope = request?;
                NextBatch::request(envelope.body)
            }
        };
        Ok(batch)
    }

    fn collect_ready_events_into_batch(&mut self, batch: &mut NextBatch) -> Result<()> {
        let ready_requests = self.self_model.take_ready_items()?;
        batch.extend_requests(
            ready_requests
                .items
                .into_iter()
                .map(|envelope| envelope.body),
        );

        if !self.allocation_updates.take_ready_items()?.items.is_empty() {
            batch.mark_model_update();
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_update_and_requests_can_share_batch() {
        let mut batch = NextBatch::request(SelfModelRequest::new("what are you attending to?"));
        batch.mark_model_update();
        batch.extend_requests([SelfModelRequest::new("what changed?")]);

        assert!(batch.update_model);
        assert_eq!(batch.requests.len(), 2);
    }

    #[test]
    fn requests_preserve_receive_order() {
        let mut batch = NextBatch::request(SelfModelRequest::new("first"));
        batch.extend_requests([SelfModelRequest::new("second")]);

        let questions = batch
            .requests
            .into_iter()
            .map(|request| request.question)
            .collect::<Vec<_>>();
        assert_eq!(questions, vec!["first", "second"]);
    }
}
