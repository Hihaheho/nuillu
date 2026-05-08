use anyhow::{Context, Result};
use async_trait::async_trait;
use lutum::{Session, StructuredTurnOutcome};
use nuillu_module::{
    AllocationReader, AllocationUpdatedInbox, AttentionWriter, BlackboardReader, LlmAccess, Module,
    TimeDivision,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

mod batch;

const SYSTEM_PROMPT: &str = r#"You are the attention-gate module.
Read the non-cognitive blackboard snapshot and allocation guidance, then decide whether anything
should enter the cognitive attention stream. Append only concise, novel, currently relevant events.
When promoting sensory memo content, convert detailed observation ages to one of the provided
time-division tags before writing attention text.
If allocation guidance asks for speech evidence promotion and a query, self-model, sensory, or
other memo contains the requested fact, promote that fact into attention in plain speech-ready form.
Include the retrieved fact and the immediate attended question or peer situation. Do not promote
generic advice, speculation, hidden module mechanics, or facts not present in memos.
Return only raw JSON for the structured object;
do not wrap it in Markdown or code fences."#;

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct AttentionGateDecision {
    pub append_attention: bool,
    pub attention_text: Option<String>,
}

pub struct AttentionGateModule {
    updates: AllocationUpdatedInbox,
    blackboard: BlackboardReader,
    allocation: AllocationReader,
    attention: AttentionWriter,
    time_division: TimeDivision,
    llm: LlmAccess,
}

impl AttentionGateModule {
    pub fn new(
        updates: AllocationUpdatedInbox,
        blackboard: BlackboardReader,
        allocation: AllocationReader,
        attention: AttentionWriter,
        time_division: TimeDivision,
        llm: LlmAccess,
    ) -> Self {
        Self {
            updates,
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
            .structured_turn::<AttentionGateDecision>()
            .collect()
            .await
            .context("attention-gate structured turn failed")?;

        let StructuredTurnOutcome::Structured(decision) = result.semantic else {
            anyhow::bail!("attention-gate structured turn refused");
        };

        if decision.append_attention
            && let Some(text) = decision.attention_text
            && !text.trim().is_empty()
        {
            self.attention.append(text).await;
        }
        Ok(())
    }
}

#[async_trait(?Send)]
impl Module for AttentionGateModule {
    type Batch = ();

    async fn next_batch(&mut self) -> Result<Self::Batch> {
        AttentionGateModule::next_batch(self).await
    }

    async fn activate(&mut self, _batch: &Self::Batch) -> Result<()> {
        AttentionGateModule::activate(self).await
    }
}
