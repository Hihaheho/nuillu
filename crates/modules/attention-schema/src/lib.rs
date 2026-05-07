use anyhow::{Context, Result};
use async_trait::async_trait;
use lutum::{Session, StructuredTurnOutcome};
use nuillu_module::{
    ActivationGate, AttentionReader, LlmAccess, Memo, Module, PeriodicInbox, SelfModelInbox,
    SelfModelRequest,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

mod batch;

const MODEL_PROMPT: &str = r#"You are the attention-schema module.
Maintain a compact first-person model of what the agent is currently attending to. The model is
a simplified self-model, not a controller or ground truth allocator."#;

const ANSWER_PROMPT: &str = r#"Answer from the attention schema in the first person.
Use only the current attention stream and self-model. Be concise and do not claim access to hidden
non-cognitive state."#;

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
    periodic: PeriodicInbox,
    gate: ActivationGate,
    attention: AttentionReader,
    memo: Memo,
    llm: LlmAccess,
}

impl AttentionSchemaModule {
    pub fn new(
        self_model: SelfModelInbox,
        periodic: PeriodicInbox,
        gate: ActivationGate,
        attention: AttentionReader,
        memo: Memo,
        llm: LlmAccess,
    ) -> Self {
        Self {
            self_model,
            periodic,
            gate,
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
        session.push_user(serde_json::json!({ "attention_stream": attention }).to_string());

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

    async fn run_loop(&mut self) -> Result<()> {
        loop {
            let batch = self.next_batch().await?;
            if batch.update_model {
                let _ = self.update_model().await;
            }
            if !batch.requests.is_empty() {
                let _ = self.answer_batch(batch.requests).await;
            }
        }
    }
}

#[async_trait(?Send)]
impl Module for AttentionSchemaModule {
    async fn run(&mut self) {
        if let Err(error) = self.run_loop().await {
            tracing::debug!(?error, "attention-schema module loop stopped");
        }
    }
}
