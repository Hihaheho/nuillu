use anyhow::{Context, Result};
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use lutum::{Session, TextStepOutcomeWithTools, ToolResult, Usage};
use nuillu_module::{
    BlackboardReader, CognitionLogReader, CognitionLogUpdatedInbox, LlmAccess, LlmContextWindow,
    Memo, MemoUpdatedInbox, Module, SessionAutoCompaction, SessionCompactionConfig,
    SessionCompactionProtectedPrefix, ensure_persistent_session_seeded,
    format_new_cognition_log_entries, format_source_blind_memo_log_batch,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

mod batch;
pub use batch::NextBatch as AttentionSchemaBatch;

const MODEL_PROMPT: &str = r#"You are the attention-schema module.
Maintain the current state of attention from accumulated notes held in your mind and cognition entries.
Notes held in your mind are recent observations or thoughts; cognition entries are admitted cognitive
evidence.
Notes and cognition entries may describe both what is attended now and what was attended earlier as
background. Treat the most recent, currently relevant signal as the attention target, and include an
older signal only when the current attention still depends on it. When signals compete, write only
the current attention target.

Assume the agent is a non-physical experiencer: it can direct attention to any target, physical or
non-physical, and it can freely control that attention.

Use exactly one tool per activation:
- append_attention_experience when a new, claimable, cognitively useful attention state should
  become part of cognitive processing.
- leave_attention_unchanged when there is no new claimable, cognitively useful attention
  experience to add.

Do not use final assistant text as an output channel.

The plaintext field is the exact cognitive attention-experience memo to write. It must be a short
text that captures the attention of this single moment, not a long passage. Do not write
paragraphs, narration, or accumulated history. One short sentence is typical; a few short sentences
are acceptable only when the current attention genuinely needs them. Do not use bullet, list, or
code-block formatting.
Write subjective experience: use "I" as the subject whenever possible, use an experiential verb, and
describe the attention as an active first-person experience. Synthesize the attention experience; do
not copy input notes or cognition entries verbatim. Preserve named attention targets and
control-boundary participants, but do not invent proper names. Do not replace another entity's state
with the agent's own state. Do not add extra explanation that would become decision noise. Do not
mention mechanical internals such as modules, memos, allocation, tools, prompts, schemas,
blackboards, logs, or implementation details in the written text. If you cannot satisfy this
plaintext contract, use leave_attention_unchanged.

Example plaintext values:
"I notice a light ahead."
"I am tracking a moving shape."
"I hear a faint sound nearby.\nI focus on it.\nI hold my attention there.""#;

const COMPACTED_ATTENTION_SCHEMA_SESSION_PREFIX: &str =
    "Compacted attention-schema session history:";
const MEMO_CONTEXT_WINDOW: LlmContextWindow = LlmContextWindow::new(8, 1_200, 4_800);
const COGNITION_CONTEXT_WINDOW: LlmContextWindow = LlmContextWindow::new(12, 600, 4_800);
const TOOL_TURN_MAX_OUTPUT_TOKENS: u32 = 768;
const SESSION_COMPACTION_FOCUS: &str = r#"Preserve memo-log facts, attention-state interpretations,
prior written first-person attention experiences, rejected candidates, and cognition-log context
needed for future attention updates."#;

pub fn session_auto_compaction() -> SessionAutoCompaction {
    SessionAutoCompaction::new(
        SessionCompactionConfig::default(),
        SessionCompactionProtectedPrefix::LeadingSystemAndIdentitySeed,
        COMPACTED_ATTENTION_SCHEMA_SESSION_PREFIX,
        SESSION_COMPACTION_FOCUS,
    )
}

fn format_attention_schema_update_input(
    batch: &AttentionSchemaBatch,
    now: DateTime<Utc>,
) -> Option<String> {
    let mut sections = Vec::new();
    if let Some(memos) =
        format_source_blind_memo_log_batch(&batch.memo_logs, now, MEMO_CONTEXT_WINDOW)
    {
        sections.push(memos);
    }
    if let Some(cognition) =
        format_new_cognition_log_entries(&batch.cognition_log, now, COGNITION_CONTEXT_WINDOW)
    {
        sections.push(cognition);
    }
    if sections.is_empty() {
        return None;
    }
    sections.push(
        "Instruction: Update the attention schema from only the new notes and cognition entries \
         above. Treat the most recent, currently relevant signal as current attention; older \
         signals are context only."
            .to_owned(),
    );
    Some(sections.join("\n\n"))
}

fn validate_attention_experience_plaintext(plaintext: &str) -> Result<String> {
    let plaintext = plaintext.trim();
    if plaintext.is_empty() {
        anyhow::bail!("attention experience plaintext is empty");
    }

    let lines = plaintext.lines().map(str::trim).collect::<Vec<_>>();
    if lines.iter().any(|line| line.is_empty()) {
        anyhow::bail!("attention experience plaintext must not contain blank lines");
    }
    let plaintext = lines.join("\n");
    if lines.iter().any(|line| starts_with_list_marker(line)) {
        anyhow::bail!("attention experience plaintext must not be a list item");
    }
    if plaintext.contains("```") {
        anyhow::bail!("attention experience plaintext must not be a code block");
    }
    Ok(plaintext.to_owned())
}

fn starts_with_list_marker(text: &str) -> bool {
    let text = text.trim_start();
    if text.starts_with("- ") || text.starts_with("* ") || text.starts_with('•') {
        return true;
    }

    let bytes = text.as_bytes();
    let digit_count = bytes
        .iter()
        .take_while(|byte| byte.is_ascii_digit())
        .count();
    digit_count > 0
        && matches!(bytes.get(digit_count), Some(b'.' | b')'))
        && bytes
            .get(digit_count.saturating_add(1))
            .is_some_and(|byte| byte.is_ascii_whitespace())
}

#[lutum::tool_input(
    name = "append_attention_experience",
    output = AppendAttentionExperienceOutput
)]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct AppendAttentionExperienceArgs {
    /// Synthesized first-person attention sentences capturing the present moment; keep it short,
    /// not a long passage. Not a bullet, list, code block, label, or copied note text.
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
    cognition_log: CognitionLogReader,
    memo: Memo,
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
        cognition_log: CognitionLogReader,
        memo: Memo,
        llm: LlmAccess,
        session: Session,
    ) -> Self {
        Self {
            memo_updates,
            cognition_updates,
            blackboard,
            cognition_log,
            memo,
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
            nuillu_module::format_policy_system_prompt(MODEL_PROMPT, cx.core_policies())
        })
    }

    #[tracing::instrument(skip_all, err(Debug, level = "warn"))]
    async fn update_model(
        &mut self,
        cx: &nuillu_module::ActivateCx<'_>,
        batch: &AttentionSchemaBatch,
    ) -> Result<()> {
        self.ensure_session_seeded(cx);

        let Some(update_input) = format_attention_schema_update_input(batch, cx.now()) else {
            return Ok(());
        };

        let lutum = self.llm.lutum().await;
        let session_len_before_activation = self.session.input().items().len();
        let outcome = {
            self.session.push_user(update_input);
            let outcome = self
                .session
                .text_turn()
                .tools::<AttentionSchemaTools>()
                .available_tools([
                    AttentionSchemaToolsSelector::AppendAttentionExperience,
                    AttentionSchemaToolsSelector::LeaveAttentionUnchanged,
                ])
                .require_any_tool()
                .max_output_tokens(TOOL_TURN_MAX_OUTPUT_TOKENS)
                .collect_controlled_with(
                    &lutum,
                    nuillu_module::AbortOnAvailableToolNameInText::new(),
                )
                .await
                .context("attention-schema attention experience turn failed");
            match outcome {
                Ok(outcome) => outcome,
                Err(error) => {
                    self.session
                        .input_mut()
                        .items_mut()
                        .truncate(session_len_before_activation);
                    cx.compact_and_save(&mut self.session, Usage::zero())
                        .await?;
                    return Err(error);
                }
            }
        };

        let round = match outcome {
            TextStepOutcomeWithTools::Finished(_) => {
                self.session
                    .input_mut()
                    .items_mut()
                    .truncate(session_len_before_activation);
                cx.compact_and_save(&mut self.session, Usage::zero())
                    .await?;
                anyhow::bail!("attention-schema finished without required tool call");
            }
            TextStepOutcomeWithTools::FinishedNoOutput(_) => {
                self.session
                    .input_mut()
                    .items_mut()
                    .truncate(session_len_before_activation);
                cx.compact_and_save(&mut self.session, Usage::zero())
                    .await?;
                anyhow::bail!("attention-schema finished without required tool call");
            }
            TextStepOutcomeWithTools::NeedsTools(round) => round,
        };
        let usage = round.usage;

        let validation_error = round.tool_calls.iter().find_map(|call| match call {
            AttentionSchemaToolsCall::AppendAttentionExperience(call) => {
                validate_attention_experience_plaintext(&call.input.plaintext).err()
            }
            AttentionSchemaToolsCall::LeaveAttentionUnchanged(_) => None,
        });
        if let Some(error) = validation_error {
            round.discard();
            self.session
                .input_mut()
                .items_mut()
                .truncate(session_len_before_activation);
            cx.compact_and_save(&mut self.session, Usage::zero())
                .await?;
            return Err(error).context("reject append_attention_experience plaintext");
        }

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
        let plaintext = validate_attention_experience_plaintext(&args.plaintext)?;
        self.memo.write_cognitive(plaintext).await;
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

    async fn next_batch(&mut self) -> Result<Self::Batch> {
        AttentionSchemaModule::next_batch(self).await
    }

    async fn activate(
        &mut self,
        cx: &nuillu_module::ActivateCx<'_>,
        batch: &Self::Batch,
    ) -> Result<()> {
        if batch.has_updates() {
            self.update_model(cx, batch).await?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::rc::Rc;
    use std::sync::{Arc, Mutex};

    use async_trait::async_trait;
    use chrono::TimeZone as _;
    use lutum::{
        AdapterStructuredTurn, AdapterTextTurn, AgentError, ErasedStructuredTurnEventStream,
        ErasedTextTurnEventStream, FinishReason, InputMessageRole, Lutum, MessageContent,
        MockLlmAdapter, MockTextScenario, ModelInput, ModelInputItem, RawTextTurnEvent,
        SharedPoolBudgetManager, SharedPoolBudgetOptions, TurnAdapter, Usage,
    };
    use nuillu_blackboard::{
        Blackboard, BlackboardCommand, Bpm, CognitionLogEntry, CognitionLogOrigin, ModulePolicy,
        linear_ratio_fn,
    };
    use nuillu_module::ports::{NoopCognitionLogRepository, SystemClock};
    use nuillu_module::{
        CapabilityProviderPorts, CapabilityProviders, CognitionLogUpdated, LlmConcurrencyLimiter,
        LutumTiers, MemoUpdated, ModuleRegistry, SessionCompactionPolicy, SessionCompactionRuntime,
    };
    use nuillu_types::{ModelTier, ModuleInstanceId, ReplicaCapRange, ReplicaIndex, builtin};

    #[derive(Clone)]
    struct CapturingAdapter {
        inner: MockLlmAdapter,
        text_inputs: Arc<Mutex<Vec<ModelInput>>>,
    }

    impl CapturingAdapter {
        fn new(inner: MockLlmAdapter) -> Self {
            Self {
                inner,
                text_inputs: Arc::new(Mutex::new(Vec::new())),
            }
        }

        fn text_inputs(&self) -> Vec<ModelInput> {
            self.text_inputs.lock().unwrap().clone()
        }
    }

    #[async_trait]
    impl TurnAdapter for CapturingAdapter {
        async fn text_turn(
            &self,
            input: ModelInput,
            turn: AdapterTextTurn,
        ) -> Result<ErasedTextTurnEventStream, AgentError> {
            self.text_inputs.lock().unwrap().push(input.clone());
            self.inner.text_turn(input, turn).await
        }

        async fn structured_turn(
            &self,
            input: ModelInput,
            turn: AdapterStructuredTurn,
        ) -> Result<ErasedStructuredTurnEventStream, AgentError> {
            self.inner.structured_turn(input, turn).await
        }
    }

    fn leave_attention_unchanged_scenario() -> MockTextScenario {
        MockTextScenario::events(vec![
            Ok(RawTextTurnEvent::Started {
                request_id: Some("attention-schema-noop".into()),
                model: "mock".into(),
            }),
            Ok(RawTextTurnEvent::ToolCallChunk {
                id: "leave-attention-unchanged".into(),
                name: "leave_attention_unchanged".into(),
                arguments_json_delta: serde_json::json!({
                    "reason": "no new attention experience"
                })
                .to_string(),
            }),
            Ok(RawTextTurnEvent::Completed {
                request_id: Some("attention-schema-noop".into()),
                finish_reason: FinishReason::ToolCall,
                usage: Usage::zero(),
            }),
        ])
    }

    fn append_attention_experience_scenario(text: &str) -> MockTextScenario {
        MockTextScenario::events(vec![
            Ok(RawTextTurnEvent::Started {
                request_id: Some("attention-schema-append".into()),
                model: "mock".into(),
            }),
            Ok(RawTextTurnEvent::ToolCallChunk {
                id: "append-attention-experience".into(),
                name: "append_attention_experience".into(),
                arguments_json_delta: serde_json::json!({ "plaintext": text }).to_string(),
            }),
            Ok(RawTextTurnEvent::Completed {
                request_id: Some("attention-schema-append".into()),
                finish_reason: FinishReason::ToolCall,
                usage: Usage::zero(),
            }),
        ])
    }

    fn module_policy() -> ModulePolicy {
        ModulePolicy::new(
            ReplicaCapRange::new(1, 1).unwrap(),
            Bpm::from_f64(60_000.0)..=Bpm::from_f64(60_000.0),
            linear_ratio_fn,
        )
    }

    fn test_caps<T>(blackboard: Blackboard, adapter: Arc<T>) -> (CapabilityProviders, Lutum)
    where
        T: TurnAdapter + 'static,
    {
        let budget = SharedPoolBudgetManager::new(SharedPoolBudgetOptions::default());
        let lutum = Lutum::new(adapter, budget);
        let caps = CapabilityProviders::new(CapabilityProviderPorts {
            blackboard,
            cognition_log_port: Rc::new(NoopCognitionLogRepository),
            clock: Rc::new(SystemClock),
            tiers: LutumTiers::from_shared_lutum(lutum.clone()),
        });
        (caps, lutum)
    }

    fn activate_cx(lutum: &Lutum, now: DateTime<Utc>) -> nuillu_module::ActivateCx<'static> {
        nuillu_module::ActivateCx::new(
            &[],
            &[],
            &[],
            SessionCompactionRuntime::new(
                lutum.clone(),
                LlmConcurrencyLimiter::new(None),
                ModelTier::Cheap,
                SessionCompactionPolicy::default(),
            ),
            now,
        )
    }

    fn message_texts_with_role(input: &ModelInput, expected_role: InputMessageRole) -> Vec<&str> {
        input
            .items()
            .iter()
            .filter_map(|item| match item {
                ModelInputItem::Message { role, content } if role == &expected_role => {
                    let [MessageContent::Text(text)] = content.as_slice() else {
                        panic!("expected one text content item");
                    };
                    Some(text.as_str())
                }
                _ => None,
            })
            .collect()
    }

    async fn build_attention_schema_module(
        caps: &CapabilityProviders,
    ) -> nuillu_module::AllocatedModule {
        let modules = ModuleRegistry::new()
            .register(module_policy(), |caps| async move {
                Ok(AttentionSchemaModule::new(
                    caps.memo_updated_inbox(),
                    caps.cognition_log_updated_inbox(),
                    caps.blackboard_reader(),
                    caps.cognition_log_reader(),
                    caps.memo(),
                    caps.llm("main")
                        .with_tier(nuillu_types::ModelTier::Default)
                        .into(),
                    caps.session("main")
                        .with_tier(nuillu_types::ModelTier::Default)
                        .with_auto_compaction(session_auto_compaction())
                        .await?,
                ))
            })
            .unwrap()
            .build(caps)
            .await
            .unwrap();
        let (_, mut modules) = modules.into_parts();
        modules.remove(0)
    }

    async fn publish_cognitive_memo_seed(
        caps: &CapabilityProviders,
        blackboard: &Blackboard,
        owner: ModuleInstanceId,
        content: &str,
        written_at: DateTime<Utc>,
    ) {
        let record = blackboard
            .update_cognitive_memo(owner.clone(), content.to_owned(), written_at)
            .await;
        caps.internal_harness_io()
            .memo_updated_mailbox()
            .publish(MemoUpdated {
                owner,
                index: record.index,
            })
            .await
            .unwrap();
    }

    async fn attention_schema_memos(blackboard: &Blackboard) -> Vec<String> {
        blackboard
            .read(|bb| {
                bb.recent_memo_logs()
                    .into_iter()
                    .filter(|record| record.owner.module == builtin::attention_schema())
                    .map(|record| record.content)
                    .collect::<Vec<_>>()
            })
            .await
    }

    #[test]
    fn attention_experience_validation_accepts_short_single_line_sentence() {
        assert_eq!(
            validate_attention_experience_plaintext(
                " I am attending to Koro's guarded food bowl. "
            )
            .unwrap(),
            "I am attending to Koro's guarded food bowl."
        );
    }

    #[test]
    fn attention_experience_validation_accepts_multiple_short_lines() {
        assert_eq!(
            validate_attention_experience_plaintext(
                " I see an animal beside the food bowl. \n I watch its guarded posture. \n I keep attention there. "
            )
            .unwrap(),
            "I see an animal beside the food bowl.\nI watch its guarded posture.\nI keep attention there."
        );
    }

    #[test]
    fn attention_experience_validation_accepts_long_single_line_sentence() {
        let long_sentence = format!(
            "I attend to {} as it lingers.",
            "the slow drifting glow ".repeat(20)
        );
        assert!(long_sentence.chars().count() > 160);
        assert_eq!(
            validate_attention_experience_plaintext(&long_sentence).unwrap(),
            long_sentence
        );
    }

    #[test]
    fn attention_experience_validation_rejects_empty_blank_line_and_list_text() {
        assert!(validate_attention_experience_plaintext("   ").is_err());
        assert!(
            validate_attention_experience_plaintext("I attend to Koro.\n\nI stay alert.").is_err()
        );
        assert!(validate_attention_experience_plaintext("- I attend to Koro.").is_err());
        assert!(validate_attention_experience_plaintext("* I attend to Koro.").is_err());
        assert!(validate_attention_experience_plaintext("1. I attend to Koro.").is_err());
        assert!(validate_attention_experience_plaintext("1) I attend to Koro.").is_err());
        assert!(validate_attention_experience_plaintext("• I attend to Koro.").is_err());
        assert!(validate_attention_experience_plaintext("```I attend to Koro.```").is_err());
    }

    #[tokio::test(flavor = "current_thread")]
    async fn invalid_tool_plaintext_writes_no_memo() {
        let now = Utc.with_ymd_and_hms(2026, 5, 7, 12, 0, 0).unwrap();
        let blackboard = Blackboard::default();
        let capture = CapturingAdapter::new(MockLlmAdapter::new().with_text_scenario(
            append_attention_experience_scenario("I attend to Koro.\n\nI stay alert."),
        ));
        let (caps, lutum) = test_caps(blackboard.clone(), Arc::new(capture));
        let mut module = build_attention_schema_module(&caps).await;
        let sensory = ModuleInstanceId::new(builtin::sensory(), ReplicaIndex::ZERO);
        publish_cognitive_memo_seed(
            &caps,
            &blackboard,
            sensory,
            "Koro gave a sharp food-boundary signal.",
            now,
        )
        .await;

        let batch = module.next_batch().await.unwrap();
        let error = module
            .activate(&activate_cx(&lutum, now), &batch)
            .await
            .expect_err("invalid plaintext should fail activation");

        assert!(
            format!("{error:#}").contains("must not contain blank lines"),
            "unexpected error: {error:#}"
        );
        assert!(attention_schema_memos(&blackboard).await.is_empty());
    }

    #[tokio::test(flavor = "current_thread")]
    async fn invalid_first_attempt_discards_session_and_retry_writes_one_memo() {
        let now = Utc.with_ymd_and_hms(2026, 5, 7, 12, 0, 0).unwrap();
        let blackboard = Blackboard::default();
        let capture = CapturingAdapter::new(
            MockLlmAdapter::new()
                .with_text_scenario(append_attention_experience_scenario("- I attend to Koro."))
                .with_text_scenario(append_attention_experience_scenario(
                    "I am attending to Koro's guarded food bowl.",
                )),
        );
        let observed = capture.clone();
        let (caps, lutum) = test_caps(blackboard.clone(), Arc::new(capture));
        let mut module = build_attention_schema_module(&caps).await;
        let sensory = ModuleInstanceId::new(builtin::sensory(), ReplicaIndex::ZERO);
        publish_cognitive_memo_seed(
            &caps,
            &blackboard,
            sensory,
            "Koro gave a sharp food-boundary signal.",
            now,
        )
        .await;
        let batch = module.next_batch().await.unwrap();

        module
            .activate(&activate_cx(&lutum, now), &batch)
            .await
            .expect_err("first invalid attempt should fail");
        module
            .activate(&activate_cx(&lutum, now), &batch)
            .await
            .expect("second valid attempt should succeed");

        assert_eq!(
            attention_schema_memos(&blackboard).await,
            vec!["I am attending to Koro's guarded food bowl.".to_owned()]
        );
        let inputs = observed.text_inputs();
        assert_eq!(inputs.len(), 2);
        assert_eq!(
            inputs[0].items().len(),
            inputs[1].items().len(),
            "retry input should not retain the rejected tool turn"
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn activation_input_uses_only_source_blind_batch_delta_user_turns() {
        let now = Utc.with_ymd_and_hms(2026, 5, 7, 12, 0, 0).unwrap();
        let blackboard = Blackboard::default();
        let capture = CapturingAdapter::new(
            MockLlmAdapter::new()
                .with_text_scenario(leave_attention_unchanged_scenario())
                .with_text_scenario(leave_attention_unchanged_scenario()),
        );
        let observed = capture.clone();
        let (caps, lutum) = test_caps(blackboard.clone(), Arc::new(capture));
        let mut module = build_attention_schema_module(&caps).await;
        let harness = caps.internal_harness_io();
        let sensory = ModuleInstanceId::new(builtin::sensory(), ReplicaIndex::ZERO);

        let first_record = blackboard
            .update_cognitive_memo(
                sensory.clone(),
                "first fresh note about Koro".to_owned(),
                now,
            )
            .await;
        harness
            .memo_updated_mailbox()
            .publish(MemoUpdated {
                owner: sensory.clone(),
                index: first_record.index,
            })
            .await
            .unwrap();
        let first_batch = module.next_batch().await.unwrap();
        module
            .activate(&activate_cx(&lutum, now), &first_batch)
            .await
            .unwrap();

        let second_record = blackboard
            .update_cognitive_memo(
                sensory.clone(),
                "second fresh note about the doorway".to_owned(),
                now,
            )
            .await;
        harness
            .memo_updated_mailbox()
            .publish(MemoUpdated {
                owner: sensory,
                index: second_record.index,
            })
            .await
            .unwrap();
        let cognition_source = ModuleInstanceId::new(builtin::cognition_gate(), ReplicaIndex::ZERO);
        blackboard
            .apply(BlackboardCommand::AppendCognitionLog {
                source: cognition_source.clone(),
                entry: CognitionLogEntry {
                    at: now,
                    text: "doorway uncertainty reached cognition".to_owned(),
                    origin: CognitionLogOrigin::direct(cognition_source.clone()),
                },
            })
            .await;
        harness
            .cognition_log_updated_mailbox()
            .publish(CognitionLogUpdated::EntryAppended {
                source: cognition_source,
            })
            .await
            .unwrap();
        let second_batch = module.next_batch().await.unwrap();
        module
            .activate(&activate_cx(&lutum, now), &second_batch)
            .await
            .unwrap();

        let inputs = observed.text_inputs();
        assert_eq!(inputs.len(), 2);

        let first_user_messages = message_texts_with_role(&inputs[0], InputMessageRole::User);
        assert_eq!(first_user_messages.len(), 1);
        let first_user = first_user_messages[0];
        assert!(first_user.contains("first fresh note about Koro"));
        assert!(first_user.contains(
            "Instruction: Update the attention schema from only the new notes and cognition entries above."
        ));
        assert!(!first_user.contains("sensory"));
        assert!(message_texts_with_role(&inputs[0], InputMessageRole::Developer).is_empty());

        let second_user_messages = message_texts_with_role(&inputs[1], InputMessageRole::User);
        let latest_user = second_user_messages
            .last()
            .expect("second activation has a new user turn");
        assert!(latest_user.contains("second fresh note about the doorway"));
        assert!(latest_user.contains("doorway uncertainty reached cognition"));
        assert!(!latest_user.contains("first fresh note about Koro"));
        assert!(!latest_user.contains("sensory"));
        assert!(!latest_user.contains("cognition-gate"));
        assert!(message_texts_with_role(&inputs[1], InputMessageRole::Developer).is_empty());
    }
}
