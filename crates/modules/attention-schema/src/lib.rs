use anyhow::{Context, Result};
use async_trait::async_trait;
use lutum::{Session, StructuredTurnOutcome};
use nuillu_module::{AttentionReader, AttentionStreamUpdatedInbox, LlmAccess, Memo, Module};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

mod batch;
pub use batch::NextBatch as AttentionSchemaBatch;

const MODEL_PROMPT: &str = r#"You are the attention-schema module.
Maintain a compact model of the current attentional relation and process. Use only the current
attention stream. Model what is attended, whether attention is stable or contested, what competing
targets are present, and what the current attention makes claimable as awareness. This is an
attention schema, not a self-model, controller, or ground truth allocator. Return only raw JSON for
the structured attention-schema memo; do not wrap it in Markdown or code fences."#;

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct AttentionSchemaMemo {
    pub attended_target: String,
    pub attentional_relation: String,
    pub stability: String,
    pub competing_targets: Vec<String>,
    pub claimable_awareness: String,
    pub rationale: String,
}

pub struct AttentionSchemaModule {
    updates: AttentionStreamUpdatedInbox,
    attention: AttentionReader,
    memo: Memo,
    llm: LlmAccess,
}

impl AttentionSchemaModule {
    pub fn new(
        updates: AttentionStreamUpdatedInbox,
        attention: AttentionReader,
        memo: Memo,
        llm: LlmAccess,
    ) -> Self {
        Self {
            updates,
            attention,
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
        let lutum = self.llm.lutum().await;
        let mut session = Session::new(lutum);
        session.push_system(MODEL_PROMPT);
        session.push_user(
            serde_json::json!({
                "attention_stream": attention,
            })
            .to_string(),
        );

        let result = session
            .structured_turn::<AttentionSchemaMemo>()
            .collect()
            .await
            .context("attention-schema model update failed")?;

        let StructuredTurnOutcome::Structured(model) = result.semantic else {
            anyhow::bail!("attention-schema model update refused");
        };

        let serialized =
            serde_json::to_string(&model).context("serialize attention-schema memo")?;
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
        Ok(())
    }
}
