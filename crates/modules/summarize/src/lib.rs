use anyhow::{Context, Result};
use async_trait::async_trait;
use lutum::{Session, StructuredTurnOutcome};
use nuillu_module::{
    ActivationGate, AllocationReader, AllocationUpdatedInbox, AttentionWriter, BlackboardReader,
    LlmAccess, Module, TimeDivision,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

mod batch;

const SYSTEM_PROMPT: &str = r#"You are the summarize module.
Read the non-cognitive blackboard snapshot and allocation guidance, then decide whether anything
should enter the cognitive attention stream. Append only concise, novel, currently relevant events.
When promoting sensory memo content, convert detailed observation ages to one of the provided
time-division tags before writing attention text. Return only raw JSON for the structured object;
do not wrap it in Markdown or code fences."#;

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct SummaryDecision {
    pub append_attention: bool,
    pub attention_text: Option<String>,
}

pub struct SummarizeModule {
    updates: AllocationUpdatedInbox,
    gate: ActivationGate,
    blackboard: BlackboardReader,
    allocation: AllocationReader,
    attention: AttentionWriter,
    time_division: TimeDivision,
    llm: LlmAccess,
}

impl SummarizeModule {
    pub fn new(
        updates: AllocationUpdatedInbox,
        gate: ActivationGate,
        blackboard: BlackboardReader,
        allocation: AllocationReader,
        attention: AttentionWriter,
        time_division: TimeDivision,
        llm: LlmAccess,
    ) -> Self {
        Self {
            updates,
            gate,
            blackboard,
            allocation,
            attention,
            time_division,
            llm,
        }
    }

    #[tracing::instrument(skip_all, err(Debug, level = "warn"))]
    async fn activate(&self) -> Result<()> {
        let snapshot = self
            .blackboard
            .read(|bb| {
                serde_json::json!({
                    "memos": bb.memos(),
                    "attention_stream": bb.attention_stream().entries(),
                    "memory_metadata": bb.memory_metadata(),
                    "time_division": self.time_division.as_prompt_json(),
                })
            })
            .await;
        let allocation = self.allocation.snapshot().await;

        let lutum = self.llm.lutum().await;
        let mut session = Session::new(lutum);
        session.push_system(SYSTEM_PROMPT);
        session.push_user(
            serde_json::json!({
                "blackboard": snapshot,
                "allocation": allocation,
            })
            .to_string(),
        );

        let result = session
            .structured_turn::<SummaryDecision>()
            .collect()
            .await
            .context("summarize structured turn failed")?;

        let StructuredTurnOutcome::Structured(decision) = result.semantic else {
            anyhow::bail!("summarize structured turn refused");
        };

        if decision.append_attention
            && let Some(text) = decision.attention_text
            && !text.trim().is_empty()
        {
            self.attention.append(text).await;
        }
        Ok(())
    }

    async fn run_loop(&mut self) -> Result<()> {
        loop {
            self.next_batch().await?;
            self.activate().await?;
        }
    }
}

#[async_trait(?Send)]
impl Module for SummarizeModule {
    async fn run(&mut self) {
        if let Err(error) = self.run_loop().await {
            panic!("summarize module failed: {error:#}");
        }
    }
}
