use anyhow::Result;
use nuillu_module::QueryRequest;

use crate::QueryAgenticModule;

#[derive(Debug, Default)]
pub(crate) struct NextBatch {
    pub(crate) queries: Vec<QueryRequest>,
    pub(crate) guidance: bool,
}

impl NextBatch {
    fn guidance() -> Self {
        Self {
            queries: Vec::new(),
            guidance: true,
        }
    }

    fn query(request: QueryRequest) -> Self {
        Self {
            queries: vec![request],
            guidance: false,
        }
    }

    fn mark_guidance(&mut self) {
        self.guidance = true;
    }

    fn extend_queries(&mut self, requests: impl IntoIterator<Item = QueryRequest>) {
        self.queries.extend(requests);
    }
}

impl QueryAgenticModule {
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
                NextBatch::guidance()
            }
            request = self.query.next_item() => {
                let envelope = request?;
                NextBatch::query(envelope.body)
            }
        };
        Ok(batch)
    }

    fn collect_ready_events_into_batch(&mut self, batch: &mut NextBatch) -> Result<()> {
        let ready_queries = self.query.take_ready_items()?;
        batch.extend_queries(
            ready_queries
                .items
                .into_iter()
                .map(|envelope| envelope.body),
        );

        if !self.allocation_updates.take_ready_items()?.items.is_empty() {
            batch.mark_guidance();
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn query_batch_preserves_request_order() {
        let mut batch = NextBatch::query(QueryRequest::new("first"));
        batch.extend_queries([QueryRequest::new("second")]);

        let questions = batch
            .queries
            .iter()
            .map(|request| request.question.as_str())
            .collect::<Vec<_>>();
        assert_eq!(questions, vec!["first", "second"]);
    }

    #[test]
    fn guidance_flag_can_share_batch_with_queries() {
        let mut batch = NextBatch::query(QueryRequest::new("question"));
        batch.mark_guidance();

        assert!(batch.guidance);
        assert_eq!(batch.queries.len(), 1);
    }
}
