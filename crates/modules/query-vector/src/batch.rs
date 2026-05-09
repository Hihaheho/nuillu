use anyhow::Result;
use nuillu_module::QueryRequest;

use crate::QueryVectorModule;

#[derive(Debug, Default)]
pub struct NextBatch {
    pub(crate) queries: Vec<QueryRequest>,
    pub(crate) cognition_updated: bool,
}

impl NextBatch {
    fn cognition_updated() -> Self {
        Self {
            queries: Vec::new(),
            cognition_updated: true,
        }
    }

    fn query(request: QueryRequest) -> Self {
        Self {
            queries: vec![request],
            cognition_updated: false,
        }
    }

    fn mark_cognition_updated(&mut self) {
        self.cognition_updated = true;
    }

    fn extend_queries(&mut self, requests: impl IntoIterator<Item = QueryRequest>) {
        self.queries.extend(requests);
    }
}

impl QueryVectorModule {
    pub(crate) async fn next_batch(&mut self) -> Result<NextBatch> {
        let mut batch = self.await_first_batch().await?;
        self.collect_ready_events_into_batch(&mut batch)?;
        Ok(batch)
    }

    async fn await_first_batch(&mut self) -> Result<NextBatch> {
        let batch = tokio::select! {
            update = self.cognition_updates.next_item() => {
                let _ = update?;
                NextBatch::cognition_updated()
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

        if !self.cognition_updates.take_ready_items()?.items.is_empty() {
            batch.mark_cognition_updated();
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
    fn cognition_log_update_flag_can_share_batch_with_queries() {
        let mut batch = NextBatch::query(QueryRequest::new("question"));
        batch.mark_cognition_updated();

        assert!(batch.cognition_updated);
        assert_eq!(batch.queries.len(), 1);
    }
}
