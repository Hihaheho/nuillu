use anyhow::Result;
use nuillu_module::QueryRequest;

use crate::QueryVectorModule;

#[derive(Debug, Default)]
pub(crate) struct NextBatch {
    pub(crate) queries: Vec<QueryRequest>,
    pub(crate) periodic: bool,
}

impl NextBatch {
    fn periodic() -> Self {
        Self {
            queries: Vec::new(),
            periodic: true,
        }
    }

    fn query(request: QueryRequest) -> Self {
        Self {
            queries: vec![request],
            periodic: false,
        }
    }

    fn mark_periodic(&mut self) {
        self.periodic = true;
    }

    fn extend_queries(&mut self, requests: impl IntoIterator<Item = QueryRequest>) {
        self.queries.extend(requests);
    }
}

impl QueryVectorModule {
    pub(crate) async fn next_batch(&mut self) -> Result<NextBatch> {
        let mut batch = self.await_first_batch().await?;
        self.collect_ready_events_into_batch(&mut batch)?;
        self.gate.block().await;
        self.collect_ready_events_into_batch(&mut batch)?;
        Ok(batch)
    }

    async fn await_first_batch(&mut self) -> Result<NextBatch> {
        let batch = tokio::select! {
            tick = self.periodic.next_tick() => {
                tick?;
                NextBatch::periodic()
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

        if self.periodic.take_ready_ticks()? > 0 {
            batch.mark_periodic();
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
    fn periodic_flag_can_share_batch_with_queries() {
        let mut batch = NextBatch::query(QueryRequest::new("question"));
        batch.mark_periodic();

        assert!(batch.periodic);
        assert_eq!(batch.queries.len(), 1);
    }
}
