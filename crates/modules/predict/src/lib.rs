use anyhow::{Context, Result};
use async_trait::async_trait;
use lutum::{Session, StructuredTurnOutcome};
use nuillu_module::{
    AllocationReader, BlackboardReader, CognitionLogReader, CognitionLogUpdatedInbox, LlmAccess,
    Memo, Module,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

mod batch;

const SYSTEM_PROMPT: &str = r#"You are the predict module.
Maintain forward predictions about the current cognition-log targets.
Generate predictions only; do not assess whether earlier predictions were correct and do not
request memory writes. Keep predictions concise, grounded in the current cognition log and
blackboard context. Return only raw JSON for the structured object; do not wrap it in Markdown or
code fences."#;

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
    owner: nuillu_module::ModuleId,
    updates: CognitionLogUpdatedInbox,
    cognition_log: CognitionLogReader,
    allocation: AllocationReader,
    blackboard: BlackboardReader,
    memo: Memo,
    llm: LlmAccess,
    system_prompt: std::sync::OnceLock<String>,
}

impl PredictModule {
    pub fn new(
        updates: CognitionLogUpdatedInbox,
        cognition_log: CognitionLogReader,
        allocation: AllocationReader,
        blackboard: BlackboardReader,
        memo: Memo,
        llm: LlmAccess,
    ) -> Self {
        Self {
            owner: nuillu_module::ModuleId::new(<Self as Module>::id())
                .expect("predict id is valid"),
            updates,
            cognition_log,
            allocation,
            blackboard,
            memo,
            llm,
            system_prompt: std::sync::OnceLock::new(),
        }
    }

    fn system_prompt(&self, cx: &nuillu_module::ActivateCx<'_>) -> &str {
        self.system_prompt.get_or_init(|| {
            nuillu_module::format_system_prompt(
                SYSTEM_PROMPT,
                cx.modules(),
                &self.owner,
                cx.identity_memories(),
            )
        })
    }

    #[tracing::instrument(skip_all, err(Debug, level = "warn"))]
    async fn activate(&self, cx: &nuillu_module::ActivateCx<'_>) -> Result<()> {
        let cognition_log = self.cognition_log.read(|log| log.entries().to_vec()).await;
        let context = self
            .blackboard
            .read(|bb| {
                serde_json::json!({
                    "memos": bb.memos(),
                    "memory_metadata": bb.memory_metadata(),
                })
            })
            .await;
        let allocation = self.allocation.snapshot().await;

        let mut session = Session::new();
        session.push_system(self.system_prompt(cx));
        session.push_user(
            serde_json::json!({
                "cognition_log": cognition_log,
                "blackboard_context": context,
                "allocation": allocation,
            })
            .to_string(),
        );

        let result = session
            .structured_turn::<PredictionMemo>(&self.llm.lutum().await)
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
}

#[async_trait(?Send)]
impl Module for PredictModule {
    type Batch = ();

    fn id() -> &'static str {
        "predict"
    }

    fn role_description() -> &'static str {
        "Generates forward predictions about current cognition-log subjects on each cognition-log update and writes them to its memo."
    }

    async fn next_batch(&mut self) -> Result<Self::Batch> {
        PredictModule::next_batch(self).await
    }

    async fn activate(
        &mut self,
        cx: &nuillu_module::ActivateCx<'_>,
        _batch: &Self::Batch,
    ) -> Result<()> {
        PredictModule::activate(self, cx).await
    }
}
