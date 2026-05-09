use anyhow::{Context, Result};
use async_trait::async_trait;
use lutum::{Session, StructuredTurnOutcome};
use nuillu_module::{
    AllocationReader, AllocationUpdatedInbox, BlackboardReader, CognitionWriter, LlmAccess,
    MemoUpdatedInbox, Module, TimeDivision,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

mod batch;

const SYSTEM_PROMPT: &str = r#"You are the cognition-gate module.
Read the indexed memo-log snapshot and allocation guidance, then decide whether anything should
enter the cognition log. Append only concise, novel, currently relevant events needed by cognitive work.
Treat unread_memo_logs as newly written module output. Treat recent_memo_logs as older context.
Do not promote a fact solely because it appears in recent_memo_logs if it has already been promoted
or is not directly relevant now.
When promoting sensory memo content, convert detailed observation ages to one of the provided
time-division tags before writing cognition-log text.
If allocation guidance asks for speech evidence promotion and a query, self-model, sensory, or
other memo contains the requested fact, promote that fact into the cognition log in plain speech-ready form.
Include the retrieved fact and the immediate cognition-log question or peer situation. Do not promote
generic advice, speculation, hidden module mechanics, or facts not present in memos.
Return only raw JSON for the structured object;
do not wrap it in Markdown or code fences."#;

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct CognitionGateDecision {
    pub append_cognition: bool,
    pub cognition_text: Option<String>,
}

pub struct CognitionGateModule {
    owner: nuillu_types::ModuleId,
    memo_updates: MemoUpdatedInbox,
    allocation_updates: AllocationUpdatedInbox,
    blackboard: BlackboardReader,
    allocation: AllocationReader,
    cognition: CognitionWriter,
    time_division: TimeDivision,
    llm: LlmAccess,
    system_prompt: std::sync::OnceLock<String>,
}

impl CognitionGateModule {
    pub fn new(
        memo_updates: MemoUpdatedInbox,
        allocation_updates: AllocationUpdatedInbox,
        blackboard: BlackboardReader,
        allocation: AllocationReader,
        cognition: CognitionWriter,
        time_division: TimeDivision,
        llm: LlmAccess,
    ) -> Self {
        Self {
            owner: nuillu_types::ModuleId::new(<Self as Module>::id())
                .expect("cognition-gate id is valid"),
            memo_updates,
            allocation_updates,
            blackboard,
            allocation,
            cognition,
            time_division,
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
        let snapshot = self.blackboard.unread_memo_logs().await;
        let context = self
            .blackboard
            .read(|bb| {
                serde_json::json!({
                    "latest_memos": bb.memos(),
                    "recent_memo_logs": bb.recent_memo_logs(),
                    "cognition_log": bb.cognition_log().entries(),
                    "memory_metadata": bb.memory_metadata(),
                    "time_division": self.time_division.as_prompt_json(),
                })
            })
            .await;
        let allocation = self.allocation.snapshot().await;

        let mut session = Session::new();
        session.push_system(self.system_prompt(cx));
        session.push_user(
            serde_json::json!({
                "unread_memo_logs": snapshot,
                "blackboard": context,
                "allocation": allocation,
            })
            .to_string(),
        );

        let result = session
            .structured_turn::<CognitionGateDecision>(&self.llm.lutum().await)
            .collect()
            .await
            .context("cognition-gate structured turn failed")?;

        let StructuredTurnOutcome::Structured(decision) = result.semantic else {
            anyhow::bail!("cognition-gate structured turn refused");
        };

        if decision.append_cognition
            && let Some(text) = decision.cognition_text
            && !text.trim().is_empty()
        {
            self.cognition.append(text).await;
        }
        Ok(())
    }
}

#[async_trait(?Send)]
impl Module for CognitionGateModule {
    type Batch = ();

    fn id() -> &'static str {
        "cognition-gate"
    }

    fn role_description() -> &'static str {
        "Filters blackboard memos into the cognition log: appends cognitively relevant, novel, changed, or controller-requested events when promotion is warranted."
    }

    async fn next_batch(&mut self) -> Result<Self::Batch> {
        CognitionGateModule::next_batch(self).await
    }

    async fn activate(
        &mut self,
        cx: &nuillu_module::ActivateCx<'_>,
        _batch: &Self::Batch,
    ) -> Result<()> {
        CognitionGateModule::activate(self, cx).await
    }
}
