use anyhow::{Context, Result};
use async_trait::async_trait;
use lutum::{Session, TextStepOutcomeWithTools, ToolResult};
use nuillu_module::{
    AllocationReader, BlackboardReader, CognitionLogReader, CognitionLogUpdatedInbox,
    CognitionWriter, LlmAccess, LlmContextWindow, MemoUpdatedInbox, Module, SessionAutoCompaction,
    SessionCompactionConfig, SessionCompactionProtectedPrefix, ensure_persistent_session_seeded,
    format_current_attention_guidance, push_formatted_cognition_log_batch,
    push_formatted_memo_log_batch,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

mod batch;
pub use batch::NextBatch as AttentionSchemaBatch;

const MODEL_PROMPT: &str = r#"You are the attention-schema module.
Maintain the current state of attention from memo-log history, current resource allocation, and
cognition log. Memo logs are durable module output; the cognition log is admitted cognitive evidence.

Assume the agent is a non-physical experiencer: it can direct attention to any target, physical or
non-physical, and it can freely control that attention.

Use exactly one tool per activation:
- append_attention_experience when a new, claimable, cognitively useful attention state should
  become part of cognitive processing.
- leave_attention_unchanged when there is no new claimable, cognitively useful attention
  experience to add.

Do not use final assistant text as an output channel.

The plaintext field is the exact cognition-log entry to append. Write it as subjective experience:
use "I" as the subject whenever possible, use an experiential verb, and describe the attention as an
active first-person experience. Do not add extra explanation that would become decision noise. Do
not mention mechanical internals such as modules, memos, allocation, tools, prompts, schemas,
blackboards, logs, or implementation details in the appended text."#;

const COMPACTED_ATTENTION_SCHEMA_SESSION_PREFIX: &str =
    "Compacted attention-schema session history:";
const MEMO_CONTEXT_WINDOW: LlmContextWindow = LlmContextWindow::new(8, 1_200, 4_800);
const COGNITION_CONTEXT_WINDOW: LlmContextWindow = LlmContextWindow::new(12, 600, 4_800);
const TOOL_TURN_MAX_OUTPUT_TOKENS: u32 = 768;
const SESSION_COMPACTION_FOCUS: &str = r#"Preserve memo-log facts, attention-state interpretations,
prior appended first-person attention experiences, rejected candidates, allocation guidance, and
cognition-log context needed for future attention updates."#;

pub fn session_auto_compaction() -> SessionAutoCompaction {
    SessionAutoCompaction::new(
        SessionCompactionConfig::default(),
        SessionCompactionProtectedPrefix::LeadingSystemAndIdentitySeed,
        COMPACTED_ATTENTION_SCHEMA_SESSION_PREFIX,
        SESSION_COMPACTION_FOCUS,
    )
}

fn format_attention_schema_context(
    allocation: &nuillu_module::ResourceAllocation,
) -> Option<String> {
    format_current_attention_guidance(allocation).map(|attention| {
        format!("Attention-schema context for updating the attention model:\n\n{attention}")
    })
}

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

#[lutum::tool_input(
    name = "leave_attention_unchanged",
    output = LeaveAttentionUnchangedOutput
)]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct LeaveAttentionUnchangedArgs {
    pub reason: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct LeaveAttentionUnchangedOutput {
    pub unchanged: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema, lutum::Toolset)]
pub enum AttentionSchemaTools {
    AppendAttentionExperience(AppendAttentionExperienceArgs),
    LeaveAttentionUnchanged(LeaveAttentionUnchangedArgs),
}

pub struct AttentionSchemaModule {
    memo_updates: MemoUpdatedInbox,
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
        cognition_updates: CognitionLogUpdatedInbox,
        blackboard: BlackboardReader,
        allocation: AllocationReader,
        cognition_log: CognitionLogReader,
        cognition: CognitionWriter,
        llm: LlmAccess,
        session: Session,
    ) -> Self {
        Self {
            memo_updates,
            cognition_updates,
            blackboard,
            allocation,
            cognition_log,
            cognition,
            llm,
            session,
            model_prompt: std::sync::OnceLock::new(),
        }
    }

    fn ensure_session_seeded(&mut self, cx: &nuillu_module::ActivateCx<'_>) {
        let model_prompt = self.model_prompt(cx).to_owned();
        ensure_persistent_session_seeded(
            &mut self.session,
            model_prompt,
            cx.identity_memories(),
            cx.now(),
        );
    }

    fn model_prompt(&self, cx: &nuillu_module::ActivateCx<'_>) -> &str {
        self.model_prompt.get_or_init(|| {
            nuillu_module::format_identity_system_prompt(
                MODEL_PROMPT,
                cx.identity_memories(),
                cx.core_policies(),
                cx.now(),
            )
        })
    }

    #[tracing::instrument(skip_all, err(Debug, level = "warn"))]
    async fn update_model(&mut self, cx: &nuillu_module::ActivateCx<'_>) -> Result<()> {
        self.ensure_session_seeded(cx);

        let unread_memo_logs = self.blackboard.unread_memo_logs().await;
        let unread_cognition_log = self.cognition_log.unread_events().await;
        let allocation = self.allocation.snapshot().await;

        let lutum = self.llm.lutum().await;
        let outcome = {
            push_formatted_memo_log_batch(
                &mut self.session,
                &unread_memo_logs,
                cx.now(),
                MEMO_CONTEXT_WINDOW,
            );
            push_formatted_cognition_log_batch(
                &mut self.session,
                &unread_cognition_log,
                cx.now(),
                COGNITION_CONTEXT_WINDOW,
            );
            if let Some(context) = format_attention_schema_context(&allocation) {
                self.session.push_ephemeral_system(context);
            }
            self.session
                .push_ephemeral_developer("Update the attention schema from the new notes above.");
            self.session
                .text_turn()
                .tools::<AttentionSchemaTools>()
                .available_tools([
                    AttentionSchemaToolsSelector::AppendAttentionExperience,
                    AttentionSchemaToolsSelector::LeaveAttentionUnchanged,
                ])
                .require_any_tool()
                .max_output_tokens(TOOL_TURN_MAX_OUTPUT_TOKENS)
                .collect(&lutum)
                .await
                .context("attention-schema attention experience turn failed")?
        };

        let round = match outcome {
            TextStepOutcomeWithTools::Finished(result) => {
                cx.compact_and_save(&mut self.session, result.usage).await?;
                anyhow::bail!("attention-schema finished without required tool call");
            }
            TextStepOutcomeWithTools::FinishedNoOutput(result) => {
                cx.compact_and_save(&mut self.session, result.usage).await?;
                anyhow::bail!("attention-schema finished without required tool call");
            }
            TextStepOutcomeWithTools::NeedsTools(round) => round,
        };
        let usage = round.usage;

        let mut results: Vec<ToolResult> = Vec::new();
        nuillu_module::emit_trace_tool_calls(&round.tool_calls);
        for call in round.tool_calls.iter().cloned() {
            match call {
                AttentionSchemaToolsCall::AppendAttentionExperience(call) => {
                    let output = self
                        .append_attention_experience(call.input.clone())
                        .await
                        .context("run append_attention_experience tool")?;
                    results.push(
                        call.complete(output)
                            .context("complete append_attention_experience tool call")?,
                    );
                }
                AttentionSchemaToolsCall::LeaveAttentionUnchanged(call) => {
                    let output = self.leave_attention_unchanged(call.input.clone());
                    results.push(
                        call.complete(output)
                            .context("complete leave_attention_unchanged tool call")?,
                    );
                }
            }
        }
        round
            .commit(&mut self.session, results)
            .context("commit attention-schema tool round")?;
        cx.compact_and_save(&mut self.session, usage).await?;
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

    fn leave_attention_unchanged(
        &self,
        _args: LeaveAttentionUnchangedArgs,
    ) -> LeaveAttentionUnchangedOutput {
        LeaveAttentionUnchangedOutput { unchanged: true }
    }
}

#[async_trait(?Send)]
impl Module for AttentionSchemaModule {
    type Batch = AttentionSchemaBatch;

    fn id() -> &'static str {
        "attention-schema"
    }

    fn peer_context() -> Option<&'static str> {
        Some("Attention-schema forms a first-person model of what is currently held in attention.")
    }

    fn allocation_hint() -> Option<&'static str> {
        Some(
            "Raise attention-schema when the current brain state suggests a meaningful shift in attention or a new first-person attention experience. Keep it low when attention has not meaningfully changed.",
        )
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
