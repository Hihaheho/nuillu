use anyhow::Result;
use nuillu_module::SelfModelRequest;

use crate::SelfModelModule;

#[derive(Debug)]
pub struct NextBatch {
    pub(crate) requests: Vec<SelfModelRequest>,
}

impl NextBatch {
    fn request(request: SelfModelRequest) -> Self {
        Self {
            requests: vec![request],
        }
    }

    fn extend_requests(&mut self, requests: impl IntoIterator<Item = SelfModelRequest>) {
        self.requests.extend(requests);
    }
}

impl SelfModelModule {
    pub(crate) async fn next_batch(&mut self) -> Result<NextBatch> {
        let mut batch = self.await_first_batch().await?;
        self.collect_ready_events_into_batch(&mut batch)?;
        Ok(batch)
    }

    async fn await_first_batch(&mut self) -> Result<NextBatch> {
        let envelope = self.requests.next_item().await?;
        Ok(NextBatch::request(envelope.body))
    }

    fn collect_ready_events_into_batch(&mut self, batch: &mut NextBatch) -> Result<()> {
        let ready_requests = self.requests.take_ready_items()?;
        batch.extend_requests(
            ready_requests
                .items
                .into_iter()
                .map(|envelope| envelope.body),
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
