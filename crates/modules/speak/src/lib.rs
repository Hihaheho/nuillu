use anyhow::{Context, Result};
use async_trait::async_trait;
use lutum::{Session, StructuredTurnOutcome};
use nuillu_module::{
    AttentionReader, AttentionStreamUpdatedInbox, BlackboardReader, LlmAccess, Memo, Module,
    UtteranceWriter,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

mod batch;

const SYSTEM_PROMPT: &str = r#"You are the speak module.
Given the current cognitive attention stream and all module memos, decide whether the agent
should emit a user-visible utterance right now. Only respond when there is something
genuinely worth saying — avoid repetition and filler. If you do respond, make the utterance
concise and directly relevant to the most attended content.
Do not query memory, write the attention stream, or change resource allocation."#;

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
struct SpeakDecision {
    should_respond: bool,
    utterance: Option<String>,
    rationale: String,
}

pub struct SpeakModule {
    updates: AttentionStreamUpdatedInbox,
    attention: AttentionReader,
    blackboard: BlackboardReader,
    memo: Memo,
    utterance: UtteranceWriter,
    llm: LlmAccess,
}

impl SpeakModule {
    pub fn new(
        updates: AttentionStreamUpdatedInbox,
        attention: AttentionReader,
        blackboard: BlackboardReader,
        memo: Memo,
        utterance: UtteranceWriter,
        llm: LlmAccess,
    ) -> Self {
        Self {
            updates,
            attention,
            blackboard,
            memo,
            utterance,
            llm,
        }
    }

    #[tracing::instrument(skip_all, err(Debug, level = "warn"))]
    async fn activate(&self) -> Result<()> {
        let attention = self
            .attention
            .read(|stream| stream.entries().to_vec())
            .await;
        let memos = self.blackboard.read(|bb| bb.memos().clone()).await;

        let lutum = self.llm.lutum().await;
        let mut session = Session::new(lutum);
        session.push_system(SYSTEM_PROMPT);
        session.push_user(
            serde_json::json!({
                "attention_stream": attention,
                "memos": memos,
            })
            .to_string(),
        );

        let result = session
            .structured_turn::<SpeakDecision>()
            .collect()
            .await
            .context("speak structured turn failed")?;

        let StructuredTurnOutcome::Structured(decision) = result.semantic else {
            anyhow::bail!("speak structured turn refused");
        };

        self.memo.write(decision.rationale).await;

        if decision.should_respond
            && let Some(text) = decision.utterance
            && !text.trim().is_empty()
        {
            self.utterance.emit(text).await;
        }
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
impl Module for SpeakModule {
    async fn run(&mut self) {
        if let Err(error) = self.run_loop().await {
            tracing::debug!(?error, "speak module loop stopped");
        }
    }
}
