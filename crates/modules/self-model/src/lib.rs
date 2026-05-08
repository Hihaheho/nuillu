use anyhow::{Context, Result};
use async_trait::async_trait;
use lutum::{Session, StructuredTurnOutcome};
use nuillu_module::{BlackboardReader, LlmAccess, Memo, Module, SelfModelInbox, SelfModelRequest};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

mod batch;
pub use batch::NextBatch as SelfModelBatch;

const SYSTEM_PROMPT: &str = r#"You are the self-model module.
Maintain a current first-person self-description from module memos, especially the attention-schema
memo, and self-related facts that query or memory modules have surfaced in their memos.
Stable self-knowledge may be present in retrieved memory memos, but do not claim direct access to
raw hidden memories or the attention stream unless those facts are present in the provided memo
context. Answer explicit self-model questions from this current self model.
Return only raw JSON for the structured object; do not wrap it in Markdown or code fences."#;

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct SelfModelAnswer {
    pub question: String,
    pub answer: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct SelfModelReport {
    pub memo: String,
    pub answers: Vec<SelfModelAnswer>,
}

pub struct SelfModelModule {
    requests: SelfModelInbox,
    blackboard: BlackboardReader,
    memo: Memo,
    llm: LlmAccess,
}

impl SelfModelModule {
    pub fn new(
        requests: SelfModelInbox,
        blackboard: BlackboardReader,
        memo: Memo,
        llm: LlmAccess,
    ) -> Self {
        Self {
            requests,
            blackboard,
            memo,
            llm,
        }
    }

    #[tracing::instrument(skip_all, err(Debug, level = "warn"))]
    async fn answer_batch(&self, requests: Vec<SelfModelRequest>) -> Result<()> {
        let context = self
            .blackboard
            .read(|bb| {
                serde_json::json!({
                    "memos": bb.memos(),
                })
            })
            .await;
        let previous_self_model = self.memo.read().await.unwrap_or_default();
        let questions = requests
            .into_iter()
            .map(|request| request.question)
            .collect::<Vec<_>>();

        let lutum = self.llm.lutum().await;
        let mut session = Session::new(lutum);
        session.push_system(SYSTEM_PROMPT);
        session.push_user(
            serde_json::json!({
                "questions": questions,
                "memo_context": context,
                "previous_self_model": previous_self_model,
            })
            .to_string(),
        );

        let result = session
            .structured_turn::<SelfModelReport>()
            .collect()
            .await
            .context("self-model structured turn failed")?;

        let StructuredTurnOutcome::Structured(report) = result.semantic else {
            anyhow::bail!("self-model structured turn refused");
        };

        let serialized = serde_json::to_string(&report).context("serialize self-model report")?;
        self.memo.write(serialized).await;
        Ok(())
    }
}

#[async_trait(?Send)]
impl Module for SelfModelModule {
    type Batch = SelfModelBatch;

    async fn next_batch(&mut self) -> Result<Self::Batch> {
        SelfModelModule::next_batch(self).await
    }

    async fn activate(&mut self, batch: &Self::Batch) -> Result<()> {
        if !batch.requests.is_empty() {
            self.answer_batch(batch.requests.clone()).await?;
        }
        Ok(())
    }
}
