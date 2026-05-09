use anyhow::{Context, Result};
use async_trait::async_trait;
use lutum::{Session, TextStepOutcomeWithTools, ToolResult};
use nuillu_module::{
    AllocationReader, AllocationUpdatedInbox, BlackboardReader, CognitionLogReader,
    CognitionLogUpdatedInbox, CognitionWriter, LlmAccess, MemoUpdatedInbox, Module,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

mod batch;
pub use batch::NextBatch as AttentionSchemaBatch;

const MODEL_PROMPT: &str = r#"You are the attention-schema module.
Maintain the current state of attention from the global memo surface, current resource allocation,
and cognition log. The global memo is the aggregate of per-module memos; the cognition log is
admitted cognitive evidence.

Assume the agent is a non-physical experiencer: it can direct attention to any target, physical or
non-physical, and it can freely control that attention. When a current attention state should become
part of cognitive processing, call append_attention_experience. Use the tool only for a new,
claimable, cognitively useful experience of attention. If there is no new attention experience to
add, finish without calling tools.

The plaintext field is the exact cognition-log entry to append. Write it as subjective experience:
use "I" as the subject whenever possible, use an experiential verb, and describe the attention as an
active first-person experience. Do not add extra explanation that would become decision noise. Do
not mention mechanical internals such as modules, memos, allocation, tools, prompts, schemas,
blackboards, logs, or implementation details in the appended text."#;

#[lutum::tool_input(
    name = "append_attention_experience",
    output = AppendAttentionExperienceOutput
)]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct AppendAttentionExperienceArgs {
    pub plaintext: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct AppendAttentionExperienceOutput {
    pub appended: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema, lutum::Toolset)]
pub enum AttentionSchemaTools {
    AppendAttentionExperience(AppendAttentionExperienceArgs),
}

pub struct AttentionSchemaModule {
    owner: nuillu_types::ModuleId,
    memo_updates: MemoUpdatedInbox,
    allocation_updates: AllocationUpdatedInbox,
    cognition_updates: CognitionLogUpdatedInbox,
    blackboard: BlackboardReader,
    allocation: AllocationReader,
    cognition_log: CognitionLogReader,
    cognition: CognitionWriter,
    llm: LlmAccess,
    session: Session,
    model_prompt: std::sync::OnceLock<String>,
}

impl AttentionSchemaModule {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        memo_updates: MemoUpdatedInbox,
        allocation_updates: AllocationUpdatedInbox,
        cognition_updates: CognitionLogUpdatedInbox,
        blackboard: BlackboardReader,
        allocation: AllocationReader,
        cognition_log: CognitionLogReader,
        cognition: CognitionWriter,
        llm: LlmAccess,
    ) -> Self {
        Self {
            owner: nuillu_types::ModuleId::new(<Self as Module>::id())
                .expect("attention-schema id is valid"),
            memo_updates,
            allocation_updates,
            cognition_updates,
            blackboard,
            allocation,
            cognition_log,
            cognition,
            llm,
            session: Session::new(),
            model_prompt: std::sync::OnceLock::new(),
        }
    }

    fn model_prompt(&self, cx: &nuillu_module::ActivateCx<'_>) -> &str {
        self.model_prompt.get_or_init(|| {
            nuillu_module::format_system_prompt(
                MODEL_PROMPT,
                cx.modules(),
                &self.owner,
                cx.identity_memories(),
            )
        })
    }

    #[tracing::instrument(skip_all, err(Debug, level = "warn"))]
    async fn update_model(&mut self, cx: &nuillu_module::ActivateCx<'_>) -> Result<()> {
        let unread_memo_logs = self.blackboard.unread_memo_logs().await;
        let unread_cognition_log = self.cognition_log.unread_events().await;
        let cognition_log = self.cognition_log.snapshot().await;
        let memo_surface = self
            .blackboard
            .read(|bb| {
                let mut latest_memos = bb.memos();
                if let Some(object) = latest_memos.as_object_mut() {
                    object.remove(self.owner.as_str());
                }
                serde_json::json!({
                    "latest_memos": latest_memos,
                    "recent_memo_logs": bb.recent_memo_logs(),
                })
            })
            .await;
        let allocation = self.allocation.snapshot().await;

        let prompt = self.model_prompt(cx).to_owned();
        self.session.push_ephemeral_system(prompt);
        self.session.push_ephemeral_user(
            serde_json::json!({
                "unread_memo_logs": unread_memo_logs,
                "unread_cognition_log": unread_cognition_log,
                "cognition_log": cognition_log,
                "memo_surface": memo_surface,
                "allocation": allocation,
            })
            .to_string(),
        );

        let outcome = self
            .session
            .text_turn(&self.llm.lutum().await)
            .tools::<AttentionSchemaTools>()
            .available_tools([AttentionSchemaToolsSelector::AppendAttentionExperience])
            .collect()
            .await
            .context("attention-schema attention experience turn failed")?;

        let TextStepOutcomeWithTools::NeedsTools(round) = outcome else {
            return Ok(());
        };

        let mut results: Vec<ToolResult> = Vec::new();
        for call in round.tool_calls.iter().cloned() {
            let AttentionSchemaToolsCall::AppendAttentionExperience(call) = call;
            let output = self
                .append_attention_experience(call.input.clone())
                .await
                .context("run append_attention_experience tool")?;
            results.push(
                call.complete(output)
                    .context("complete append_attention_experience tool call")?,
            );
        }
        round
            .commit(&mut self.session, results)
            .context("commit attention-schema tool round")?;
        Ok(())
    }

    async fn append_attention_experience(
        &self,
        args: AppendAttentionExperienceArgs,
    ) -> Result<AppendAttentionExperienceOutput> {
        let plaintext = args.plaintext.trim();
        if plaintext.is_empty() {
            return Ok(AppendAttentionExperienceOutput { appended: false });
        }
        self.cognition.append(plaintext.to_owned()).await;
        Ok(AppendAttentionExperienceOutput { appended: true })
    }
}

#[async_trait(?Send)]
impl Module for AttentionSchemaModule {
    type Batch = AttentionSchemaBatch;

    fn id() -> &'static str {
        "attention-schema"
    }

    fn role_description() -> &'static str {
        "Writes first-person attention experience entries to the cognition log when memo, allocation, and cognition-log context warrant a new claimable attention state."
    }

    async fn next_batch(&mut self) -> Result<Self::Batch> {
        AttentionSchemaModule::next_batch(self).await
    }

    async fn activate(
        &mut self,
        cx: &nuillu_module::ActivateCx<'_>,
        batch: &Self::Batch,
    ) -> Result<()> {
        if batch.update_model {
            self.update_model(cx).await?;
        }
        Ok(())
    }
}
