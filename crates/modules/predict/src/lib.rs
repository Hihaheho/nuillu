use anyhow::{Context, Result};
use async_trait::async_trait;
use lutum::{Session, StructuredTurnOutcome};
use nuillu_module::{
    AttentionReader, AttentionStreamUpdatedInbox, BlackboardReader, LlmAccess, Memo, Module,
    PeriodicInbox,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

mod batch;

const SYSTEM_PROMPT: &str = r#"You are the predict module.
Maintain forward predictions about the current attention targets.
Generate predictions only; do not assess whether earlier predictions were correct and do not
request memory writes. Keep predictions concise, grounded in the current attention stream and
blackboard context."#;

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct PredictionMemo {
    pub predictions: Vec<PredictionEntry>,
    pub rationale: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct PredictionEntry {
    pub subject: String,
    pub predicted_state: String,
    pub validity_horizon: String,
    pub rationale: String,
}

pub struct PredictModule {
    updates: AttentionStreamUpdatedInbox,
    periodic: PeriodicInbox,
    attention: AttentionReader,
    blackboard: BlackboardReader,
    memo: Memo,
    llm: LlmAccess,
}

impl PredictModule {
    pub fn new(
        updates: AttentionStreamUpdatedInbox,
        periodic: PeriodicInbox,
        attention: AttentionReader,
        blackboard: BlackboardReader,
        memo: Memo,
        llm: LlmAccess,
    ) -> Self {
        Self {
            updates,
            periodic,
            attention,
            blackboard,
            memo,
            llm,
        }
    }

    #[tracing::instrument(skip_all, err(Debug, level = "warn"))]
    async fn activate(&self) -> Result<()> {
        let attention = self
            .attention
            .read(|stream| stream.entries().to_vec())
            .await;
        let context = self
            .blackboard
            .read(|bb| {
                serde_json::json!({
                    "memos": bb.memos(),
                    "memory_metadata": bb.memory_metadata(),
                })
            })
            .await;

        let lutum = self.llm.lutum().await;
        let mut session = Session::new(lutum);
        session.push_system(SYSTEM_PROMPT);
        session.push_user(
            serde_json::json!({
                "attention_stream": attention,
                "blackboard_context": context,
            })
            .to_string(),
        );

        let result = session
            .structured_turn::<PredictionMemo>()
            .collect()
            .await
            .context("predict structured turn failed")?;

        let StructuredTurnOutcome::Structured(prediction) = result.semantic else {
            anyhow::bail!("predict structured turn refused");
        };

        let serialized = serde_json::to_string(&prediction).context("serialize predict memo")?;
        self.memo.write(serialized).await;
        Ok(())
    }

    async fn run_loop(&mut self) -> Result<()> {
        loop {
            self.next_batch().await?;
            let _ = self.activate().await;
        }
    }
}

#[async_trait(?Send)]
impl Module for PredictModule {
    async fn run(&mut self) {
        if let Err(error) = self.run_loop().await {
            tracing::debug!(?error, "predict module loop stopped");
        }
    }
}
