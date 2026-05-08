use anyhow::{Context, Result};
use async_trait::async_trait;
use lutum::{Session, StructuredTurnOutcome};
use nuillu_module::{
    AllocationReader, AllocationUpdatedInbox, AttentionReader, LlmAccess, Memo, Module,
    SelfModelInbox, SelfModelRequest,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

mod batch;
pub use batch::NextBatch as AttentionSchemaBatch;

const MODEL_PROMPT: &str = r#"You are the attention-schema module.
Maintain a compact first-person model of what the agent is currently attending to. The model is
a simplified self-model, not a controller or ground truth allocator. Return only raw JSON for the
structured self-model; do not wrap it in Markdown or code fences."#;

const ANSWER_PROMPT: &str = r#"Answer from the attention schema in the first person.
Use only the current attention stream and self-model. Be concise and do not claim access to hidden
non-cognitive state. Return only raw JSON for the structured answer; do not wrap it in Markdown or
code fences."#;

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct SelfModel {
    pub memo: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct SelfReport {
    pub answer: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct SelfReportAnswer {
    pub question: String,
    pub answer: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct SelfReportBatch {
    pub answers: Vec<SelfReportAnswer>,
}

pub struct AttentionSchemaModule {
    self_model: SelfModelInbox,
    allocation_updates: AllocationUpdatedInbox,
    attention: AttentionReader,
    allocation: AllocationReader,
    memo: Memo,
    llm: LlmAccess,
}

impl AttentionSchemaModule {
    pub fn new(
        self_model: SelfModelInbox,
        allocation_updates: AllocationUpdatedInbox,
        attention: AttentionReader,
        allocation: AllocationReader,
        memo: Memo,
        llm: LlmAccess,
    ) -> Self {
        Self {
            self_model,
            allocation_updates,
            attention,
            allocation,
            memo,
            llm,
        }
    }

    #[tracing::instrument(skip_all, err(Debug, level = "warn"))]
    async fn update_model(&self) -> Result<()> {
        let attention = self
            .attention
            .read(|stream| stream.entries().to_vec())
            .await;
        let allocation = self.allocation.snapshot().await;
        let lutum = self.llm.lutum().await;
        let mut session = Session::new(lutum);
        session.push_system(MODEL_PROMPT);
        session.push_user(
            serde_json::json!({
                "attention_stream": attention,
                "allocation": allocation,
            })
            .to_string(),
        );

        let result = session
            .structured_turn::<SelfModel>()
            .collect()
            .await
            .context("attention-schema model update failed")?;

        let StructuredTurnOutcome::Structured(model) = result.semantic else {
            anyhow::bail!("attention-schema model update refused");
        };

        self.memo.write(model.memo).await;
        Ok(())
    }

    #[tracing::instrument(skip_all, err(Debug, level = "warn"))]
    async fn answer_batch(&self, requests: Vec<SelfModelRequest>) -> Result<()> {
        let attention = self
            .attention
            .read(|stream| stream.entries().to_vec())
            .await;
        let model = self.memo.read().await.unwrap_or_default();
        let allocation = self.allocation.snapshot().await;
        let questions = requests
            .into_iter()
            .map(|request| request.question)
            .collect::<Vec<_>>();
        let lutum = self.llm.lutum().await;
        let mut session = Session::new(lutum);
        session.push_system(ANSWER_PROMPT);
        session.push_user(
            serde_json::json!({
                "questions": questions,
                "attention_stream": attention,
                "allocation": allocation,
                "self_model": model,
            })
            .to_string(),
        );

        let result = session
            .structured_turn::<SelfReportBatch>()
            .collect()
            .await
            .context("attention-schema self-report failed")?;

        let StructuredTurnOutcome::Structured(report) = result.semantic else {
            anyhow::bail!("attention-schema self-report refused");
        };

        let serialized = serde_json::to_string(&report).context("serialize self-report batch")?;
        self.memo.write(serialized).await;
        Ok(())
    }
}

#[async_trait(?Send)]
impl Module for AttentionSchemaModule {
    type Batch = AttentionSchemaBatch;

    async fn next_batch(&mut self) -> Result<Self::Batch> {
        AttentionSchemaModule::next_batch(self).await
    }

    async fn activate(&mut self, batch: &Self::Batch) -> Result<()> {
        if batch.update_model {
            self.update_model().await?;
        }
        if !batch.requests.is_empty() {
            self.answer_batch(batch.requests.clone()).await?;
        }
        Ok(())
    }
}
