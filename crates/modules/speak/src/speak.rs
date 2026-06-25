use std::borrow::Cow;
use std::rc::Rc;

use anyhow::{Context, Result};
use async_trait::async_trait;
#[cfg(test)]
use chrono::SecondsFormat;
use chrono::{DateTime, Utc};
use lutum::{Session, TextStepOutcomeWithTools, Usage};
use nuillu_blackboard::CognitionLogEntryRecord;
#[cfg(test)]
use nuillu_blackboard::CognitionLogRecord;
#[cfg(test)]
use nuillu_module::format_bounded_cognition_log_batch;
use nuillu_module::{
    AttentionControlRequest, AttentionControlRequestMailbox, CognitionLogReader,
    CognitionLogUpdated, CognitionLogUpdatedInbox, LlmAccess, LlmContextWindow, Memo, Module,
    SceneReader, SessionAutoCompaction, SessionCompactionConfig, SessionCompactionProtectedPrefix,
    UtteranceProgress, ensure_persistent_session_seeded, format_new_cognition_log_entries,
    ports::Clock,
};
use nuillu_types::builtin;
use schemars::{JsonSchema, Schema, SchemaGenerator};
use serde::{Deserialize, Serialize};

use crate::utterance::UtteranceWriter;

const SPEECH_PLANNING_PROMPT: &str = r#"Plan and emit outward speech from the current cognition log and scene target hints.
Use exactly one available tool.
Call prepare_speech only when new cognition supports a new outward utterance. target is the person or group who should hear it.
For direct speech heard from a named speaker, target that speaker. Use everyone only when the cognition explicitly calls for group or broadcast speech.
Call decline_speech_now when no new outward utterance is appropriate now. If speak should not be prioritized again until new cognition arrives, include inhibit_reason.
Predictions, expected dialogue flow, and my own previous speech are not new outward speech motivation by themselves.
Do not plan speech that repeats content the listener has already heard.
speech_content is the exact listener-visible utterance text, not a directive, summary, rationale, or future plan.
Write speech_content as one short in-world utterance that can be safely preempted. The brain may receive new sensory input or recall a new memory one second later.
Prefer one sentence. If one sentence would be long, stop at a natural clause boundary such as a comma or Japanese comma.
Do not pack multiple topics, long explanations, or future continuation plans into one speech_content.
If the cognition log contains an explicit language request, write speech_content itself in that language.
Do not wrap speech_content in quotation marks.
Do not summarize the request or say what the user wants.
Do not output introspection, narration, analysis, implementation mechanics, lookup, reasoning, prompts, rubrics, judges, or evaluation mechanics.
Nui is my own name; do not treat my name as the listener.
Do not write the target's future reply, expression, feeling, action, or narration.
Do not invent policy, actions, identity, memory, visible evidence, unknown-state evidence, or other facts not supported by the provided context."#;

const FRESH_PLANNING_TURN_DEVELOPER_INSTRUCTION: &str = "Use exactly one tool: prepare_speech with exact listener-visible speech_content for a new grounded outward utterance, or decline_speech_now when no new outward utterance is appropriate now.";

const COGNITION_CONTEXT_WINDOW: LlmContextWindow = LlmContextWindow::new(12, 600, 4_800);
const SPEECH_PLANNING_TURN_MAX_OUTPUT_TOKENS: u32 = 1024;
const COMPACTED_SPEAK_PLANNING_SESSION_PREFIX: &str = "Compacted speak planning session history:";
const PLANNING_SESSION_COMPACTION_FOCUS: &str = r#"Preserve prior speech target decisions,
selected targets, rejected/no-speech decisions, completed outward utterances, and cognition-log
context needed for future speak planning."#;

pub fn planning_session_auto_compaction() -> SessionAutoCompaction {
    SessionAutoCompaction::new(
        SessionCompactionConfig::default(),
        SessionCompactionProtectedPrefix::LeadingSystemAndIdentitySeed,
        COMPACTED_SPEAK_PLANNING_SESSION_PREFIX,
        PLANNING_SESSION_COMPACTION_FOCUS,
    )
}

tokio::task_local! {
    /// JSON Schema for `PrepareSpeechArgs.target`. `.scope`d around each
    /// planning turn so the LLM sees the current non-empty target contract.
    static SPEECH_TARGET_SCHEMA: Schema;
}

fn freeform_speech_target_schema() -> Schema {
    Schema::try_from(serde_json::json!({
        "type": "string",
        "minLength": 1,
    }))
    .expect("speech target schema must be a JSON object")
}

/// Wire-format string for a concrete speech addressee. Stored as `String` so
/// existing serialization and downstream `Utterance.target` are unchanged.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
struct SpeechTarget(String);

impl<S: Into<String>> From<S> for SpeechTarget {
    fn from(value: S) -> Self {
        Self(value.into())
    }
}

impl SpeechTarget {
    fn as_str(&self) -> &str {
        self.0.as_str()
    }
}

impl JsonSchema for SpeechTarget {
    fn inline_schema() -> bool {
        true
    }

    fn schema_name() -> Cow<'static, str> {
        "SpeechTarget".into()
    }

    fn schema_id() -> Cow<'static, str> {
        "nuillu_speak::SpeechTarget.dynamic".into()
    }

    fn json_schema(_generator: &mut SchemaGenerator) -> Schema {
        SPEECH_TARGET_SCHEMA
            .try_with(Clone::clone)
            .unwrap_or_else(|_| freeform_speech_target_schema())
    }
}

#[lutum::tool_input(name = "prepare_speech", output = PrepareSpeechOutput)]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
/// Call when proceeding to an outward utterance.
struct PrepareSpeechArgs {
    /// The concrete addressee or audience who should hear the utterance.
    target: SpeechTarget,
    /// Exact listener-visible utterance text. This is not a directive.
    speech_content: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
struct PrepareSpeechOutput {
    accepted: bool,
}

#[lutum::tool_input(name = "decline_speech_now", output = DeclineSpeechNowOutput)]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
/// Call only when a concrete blocker makes outward speech inappropriate or
/// impossible now.
struct DeclineSpeechNowArgs {
    /// Why outward speech is not appropriate right now.
    blocking_reason: String,
    /// Optional one-shot reason that speak should not be prioritized again
    /// until new cognition arrives.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    inhibit_reason: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
struct DeclineSpeechNowOutput {
    accepted: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema, lutum::Toolset)]
enum SpeakTools {
    PrepareSpeech(PrepareSpeechArgs),
    DeclineSpeechNow(DeclineSpeechNowArgs),
}

#[derive(Clone, Debug)]
struct PlannedSpeech {
    target: String,
    speech_content: String,
}

#[derive(Clone, Debug)]
enum FreshSpeechPlan {
    Prepare(PlannedSpeech),
    Decline {
        blocking_reason: String,
        inhibit_reason: Option<String>,
    },
    None,
}

#[derive(Debug)]
pub struct SpeakBatch {
    pub(crate) updates: Vec<CognitionLogUpdated>,
    pub(crate) cognition_entries: Vec<CognitionLogEntryRecord>,
}

fn render_completed_utterance_memo(target: &str, text: &str) -> String {
    format!("I said to {}:\n{}", target.trim(), text.trim())
}

fn render_declined_speech_memo(reason: &str) -> String {
    format!("I am staying silent for now. Reason:\n{}", reason.trim())
}

fn render_completed_utterance_planning_record(target: &str, text: &str) -> String {
    format!(
        "Completed outward utterance to {}:\n{}",
        target.trim(),
        text.trim(),
    )
}

fn format_planning_input(cognition_context: &str, target_hints: &[String]) -> String {
    let mut out = cognition_context.trim().to_owned();
    if !target_hints.is_empty() {
        out.push_str("\n\nPreferred visible speech targets (not exhaustive): ");
        out.push_str(&target_hints.join(", "));
        out.push_str("\nUse another concrete non-empty target when cognition supports it.");
    }
    out
}

fn trimmed_optional(value: Option<&str>) -> Option<&str> {
    value.map(str::trim).filter(|value| !value.is_empty())
}

fn target_hints_from_schema(schema: &Schema) -> Vec<String> {
    let Ok(value) = serde_json::to_value(schema) else {
        return Vec::new();
    };
    let Some(values) = value.get("enum").and_then(serde_json::Value::as_array) else {
        return Vec::new();
    };
    values
        .iter()
        .filter_map(serde_json::Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
        .collect()
}

fn is_non_empty_target(target: &str) -> bool {
    !target.trim().is_empty()
}

fn is_non_empty_speech_content(text: &str) -> bool {
    !text.trim().is_empty()
}

#[cfg(test)]
fn cognition_context_fallback(now: DateTime<Utc>) -> String {
    format!(
        "What you are currently thinking at {}:\n- none",
        now.to_rfc3339_opts(SecondsFormat::Secs, true)
    )
}

fn speech_cognition_context_from_entries(
    records: &[CognitionLogEntryRecord],
    now: DateTime<Utc>,
) -> Option<String> {
    let records = speech_planning_cognition_records(records);
    format_new_cognition_log_entries(&records, now, COGNITION_CONTEXT_WINDOW)
}

fn speech_planning_cognition_records(
    records: &[CognitionLogEntryRecord],
) -> Vec<CognitionLogEntryRecord> {
    records
        .iter()
        .filter(|record| is_speech_planning_cognition_record(record))
        .cloned()
        .collect()
}

fn is_speech_planning_cognition_record(record: &CognitionLogEntryRecord) -> bool {
    let origin = &record.entry.origin.owner.module;
    origin == &builtin::query_memory()
        || origin == &builtin::attention_schema()
        || origin == &builtin::interpreter()
        || origin == &builtin::sensory()
        || origin == &builtin::predict()
}

#[cfg(test)]
fn append_idle_context(mut cognition_context: String, idle_for_secs: Option<u64>) -> String {
    if let Some(seconds) = idle_for_secs {
        cognition_context.push_str(&format!("\n- I have been idle for {seconds} seconds."));
    }
    cognition_context
}

#[cfg(test)]
fn cognition_entry_records(logs: &[CognitionLogRecord]) -> Vec<CognitionLogEntryRecord> {
    let mut records = Vec::new();
    for log in logs
        .iter()
        .filter(|record| record.source.module != builtin::dreaming())
    {
        for entry in &log.entries {
            records.push(CognitionLogEntryRecord {
                index: records.len() as u64,
                source: log.source.clone(),
                entry: entry.clone(),
            });
        }
    }
    records
}

fn push_planning_context(session: &mut Session, cognition_context: &str, target_hints: &[String]) {
    session.push_user(format_planning_input(cognition_context, target_hints));
    session.push_ephemeral_developer(FRESH_PLANNING_TURN_DEVELOPER_INSTRUCTION);
}

pub struct SpeakModule {
    cognition_updates: CognitionLogUpdatedInbox,
    cognition_log: CognitionLogReader,
    attention_control: AttentionControlRequestMailbox,
    memo: Memo,
    utterance: UtteranceWriter,
    planning_llm: LlmAccess,
    scene: SceneReader,
    clock: Rc<dyn Clock>,
    planning_session: Session,
    plan_prompt: std::sync::OnceLock<String>,
}

pub struct SpeakModuleParts {
    pub cognition_updates: CognitionLogUpdatedInbox,
    pub cognition_log: CognitionLogReader,
    pub attention_control: AttentionControlRequestMailbox,
    pub memo: Memo,
    pub utterance: UtteranceWriter,
    pub planning_llm: LlmAccess,
    pub scene: SceneReader,
    pub clock: Rc<dyn Clock>,
    pub planning_session: Session,
}

impl SpeakModule {
    pub fn new(parts: SpeakModuleParts) -> Self {
        let SpeakModuleParts {
            cognition_updates,
            cognition_log,
            attention_control,
            memo,
            utterance,
            planning_llm,
            scene,
            clock,
            planning_session,
        } = parts;

        Self {
            cognition_updates,
            cognition_log,
            attention_control,
            memo,
            utterance,
            planning_llm,
            scene,
            clock,
            planning_session,
            plan_prompt: std::sync::OnceLock::new(),
        }
    }

    fn plan_prompt(&self, cx: &nuillu_module::ActivateCx<'_>) -> &str {
        self.plan_prompt.get_or_init(|| {
            nuillu_module::format_policy_system_prompt(SPEECH_PLANNING_PROMPT, cx.core_policies())
        })
    }

    fn ensure_planning_session_seeded(&mut self, cx: &nuillu_module::ActivateCx<'_>) {
        let system_prompt = self.plan_prompt(cx).to_owned();
        ensure_persistent_session_seeded(
            &mut self.planning_session,
            system_prompt,
            cx.identity_memories(),
            cx.now(),
        );
    }

    #[tracing::instrument(skip_all, err(Debug, level = "warn"))]
    async fn activate(
        &mut self,
        cx: &nuillu_module::ActivateCx<'_>,
        batch: &SpeakBatch,
    ) -> Result<()> {
        let _update_count = batch.updates.len();
        let now = self.clock.now();
        let Some(planning_context) =
            speech_cognition_context_from_entries(&batch.cognition_entries, now)
        else {
            return Ok(());
        };

        let plan = match self.plan_speech(cx, &planning_context).await? {
            FreshSpeechPlan::Prepare(plan) => plan,
            FreshSpeechPlan::Decline {
                blocking_reason,
                inhibit_reason,
            } => {
                self.record_declined_speech(&blocking_reason, inhibit_reason.as_deref())
                    .await;
                return Ok(());
            }
            FreshSpeechPlan::None => return Ok(()),
        };

        self.record_completed_speech(cx, &plan).await?;
        Ok(())
    }

    #[cfg(test)]
    async fn speech_cognition_context(&self, now: DateTime<Utc>) -> String {
        let snapshot = self.cognition_log.snapshot().await;
        let records = cognition_entry_records(snapshot.logs());
        let cognition_context =
            format_bounded_cognition_log_batch(&records, now, COGNITION_CONTEXT_WINDOW)
                .unwrap_or_else(|| cognition_context_fallback(now));
        let idle_for_secs = snapshot
            .agentic_deadlock_marker()
            .map(|marker| marker.idle_for.as_secs());
        append_idle_context(cognition_context, idle_for_secs)
    }

    async fn plan_speech(
        &mut self,
        cx: &nuillu_module::ActivateCx<'_>,
        cognition_context: &str,
    ) -> Result<FreshSpeechPlan> {
        self.ensure_planning_session_seeded(cx);
        let scene_target_schema = self.scene.target_schema();
        let target_hints = target_hints_from_schema(&scene_target_schema);
        let mut turn_session = self.planning_session.clone();
        push_planning_context(&mut turn_session, cognition_context, &target_hints);

        let lutum = self.planning_llm.lutum().await;
        let target_schema = freeform_speech_target_schema();
        let outcome = SPEECH_TARGET_SCHEMA
            .scope(target_schema, async {
                turn_session
                    .text_turn()
                    .tools::<SpeakTools>()
                    .available_tools([
                        SpeakToolsSelector::PrepareSpeech,
                        SpeakToolsSelector::DeclineSpeechNow,
                    ])
                    .require_any_tool()
                    .max_output_tokens(SPEECH_PLANNING_TURN_MAX_OUTPUT_TOKENS)
                    .collect_controlled_with(
                        &lutum,
                        nuillu_module::AbortOnAvailableToolNameInText::new(),
                    )
                    .await
                    .context("speak planning turn failed")
            })
            .await?;

        match outcome {
            TextStepOutcomeWithTools::Finished(_result) => {
                let detail = "model finished with assistant output but no tool call \
                    (require_any_tool should have prevented this outcome)";
                cx.warn(format!("speak planning failed: {detail}"));
                anyhow::bail!("speak planning finished without required tool call: {detail}");
            }
            TextStepOutcomeWithTools::FinishedNoOutput(_result) => {
                let detail = "model finished with no output and no tool call \
                    (require_any_tool should have prevented this outcome)";
                cx.warn(format!("speak planning failed: {detail}"));
                anyhow::bail!("speak planning finished without required tool call: {detail}");
            }
            TextStepOutcomeWithTools::NeedsTools(round) => {
                let usage = round.usage;
                nuillu_module::emit_trace_tool_calls(&round.tool_calls);
                if round.tool_calls.is_empty() {
                    let detail = "model returned NeedsTools outcome with empty tool_calls; \
                        expected prepare_speech or decline_speech_now";
                    cx.warn(format!("speak planning failed: {detail}"));
                    anyhow::bail!("speak planning finished without required tool call: {detail}");
                }
                let mut plan = FreshSpeechPlan::None;
                let mut results = Vec::new();
                for call in round.tool_calls.iter().cloned() {
                    match call {
                        SpeakToolsCall::PrepareSpeech(call) => {
                            let target = call.input.target.as_str().trim().to_owned();
                            let speech_content = call.input.speech_content.trim().to_owned();
                            let accepted = matches!(plan, FreshSpeechPlan::None)
                                && is_non_empty_target(&target)
                                && is_non_empty_speech_content(&speech_content);
                            if accepted {
                                plan = FreshSpeechPlan::Prepare(PlannedSpeech {
                                    target,
                                    speech_content,
                                });
                            }
                            results.push(
                                call.complete(PrepareSpeechOutput { accepted })
                                    .context("complete prepare_speech tool call")?,
                            );
                        }
                        SpeakToolsCall::DeclineSpeechNow(call) => {
                            let accepted = matches!(plan, FreshSpeechPlan::None);
                            if accepted {
                                plan = FreshSpeechPlan::Decline {
                                    blocking_reason: call.input.blocking_reason.clone(),
                                    inhibit_reason: call.input.inhibit_reason.clone(),
                                };
                            }
                            results.push(
                                call.complete(DeclineSpeechNowOutput { accepted })
                                    .context("complete decline_speech_now tool call")?,
                            );
                        }
                    }
                }
                round
                    .commit(&mut turn_session, results)
                    .context("commit speak planning tool round")?;
                cx.compact_and_save(&mut turn_session, usage).await?;
                self.planning_session = turn_session;
                Ok(plan)
            }
        }
    }

    async fn record_completed_speech(
        &mut self,
        cx: &nuillu_module::ActivateCx<'_>,
        plan: &PlannedSpeech,
    ) -> Result<()> {
        let generation_id = self.utterance.next_generation_id();
        self.memo
            .write_cognitive(render_completed_utterance_memo(
                &plan.target,
                &plan.speech_content,
            ))
            .await;
        self.utterance
            .record_progress(UtteranceProgress::completed(
                generation_id,
                0,
                plan.target.clone(),
                plan.speech_content.clone(),
            ))
            .await;
        self.utterance
            .emit(
                plan.target.clone(),
                generation_id,
                plan.speech_content.clone(),
            )
            .await;
        self.ensure_planning_session_seeded(cx);
        self.planning_session
            .push_system(render_completed_utterance_planning_record(
                &plan.target,
                &plan.speech_content,
            ));
        cx.compact_and_save(&mut self.planning_session, Usage::zero())
            .await?;
        Ok(())
    }

    async fn record_declined_speech(&self, reason: &str, inhibit_reason: Option<&str>) {
        self.memo.write(render_declined_speech_memo(reason)).await;
        let Some(inhibit_reason) = trimmed_optional(inhibit_reason) else {
            return;
        };
        if let Err(error) = self
            .attention_control
            .publish(AttentionControlRequest::inhibit(inhibit_reason.to_owned()))
            .await
        {
            tracing::warn!(
                target: "nuillu_speak::attention_control",
                ?error,
                "failed to publish speak inhibition reason"
            );
        }
    }
}

#[async_trait(?Send)]
impl Module for SpeakModule {
    type Batch = SpeakBatch;

    fn id() -> &'static str {
        "speak"
    }

    fn peer_context() -> Option<&'static str> {
        Some("Speak is the outward expression path for admitted cognition.")
    }

    async fn next_batch(&mut self) -> Result<Self::Batch> {
        SpeakModule::next_batch(self).await
    }

    async fn activate(
        &mut self,
        cx: &nuillu_module::ActivateCx<'_>,
        batch: &Self::Batch,
    ) -> Result<()> {
        SpeakModule::activate(self, cx, batch).await
    }
}

impl SpeakModule {
    pub(crate) async fn next_batch(&mut self) -> Result<SpeakBatch> {
        let first = self.cognition_updates.next_item().await?;
        let mut updates = vec![first.body];
        updates.extend(
            self.cognition_updates
                .take_ready_items()?
                .items
                .into_iter()
                .map(|envelope| envelope.body),
        );
        let cognition_entries = self.cognition_log.unread_events().await;

        Ok(SpeakBatch {
            updates,
            cognition_entries,
        })
    }
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::rc::Rc;
    use std::sync::{Arc, Mutex};

    use async_trait::async_trait;
    use lutum::{
        AdapterStructuredTurn, AdapterTextTurn, AgentError, ErasedStructuredTurnEventStream,
        ErasedTextTurnEventStream, FinishReason, InputMessageRole, MaxOutputTokens, MessageContent,
        MockLlmAdapter, MockTextScenario, ModelInput, ModelInputItem, RawTextTurnEvent,
        SharedPoolBudgetManager, SharedPoolBudgetOptions, TurnAdapter, Usage,
    };
    use nuillu_blackboard::{
        ActivationRatio, Blackboard, BlackboardCommand, CognitionLogEntry, CognitionLogOrigin,
        IdentityMemoryRecord, ResourceAllocation,
    };
    use nuillu_module::ports::{Clock, NoopCognitionLogRepository, PortError, SystemClock};
    use nuillu_module::{
        CapabilityProviderPorts, CapabilityProviders, CognitionLogUpdated, LutumTiers,
        ModuleRegistry, Participant,
    };
    use nuillu_types::{MemoryContent, MemoryIndex, ModuleInstanceId, ReplicaIndex, builtin};

    use super::*;
    use crate::test_support::*;
    use crate::utterance::{Utterance, UtteranceSink};

    #[derive(Clone)]
    struct CapturingAdapter<T> {
        inner: Arc<T>,
        text_turns: Arc<Mutex<Vec<AdapterTextTurn>>>,
        text_inputs: Arc<Mutex<Vec<ModelInput>>>,
    }

    impl<T> CapturingAdapter<T> {
        fn new(inner: T) -> Self {
            Self {
                inner: Arc::new(inner),
                text_turns: Arc::new(Mutex::new(Vec::new())),
                text_inputs: Arc::new(Mutex::new(Vec::new())),
            }
        }

        fn text_turns(&self) -> Vec<AdapterTextTurn> {
            self.text_turns.lock().unwrap().clone()
        }

        fn text_inputs(&self) -> Vec<ModelInput> {
            self.text_inputs.lock().unwrap().clone()
        }
    }

    #[async_trait]
    impl<T> TurnAdapter for CapturingAdapter<T>
    where
        T: TurnAdapter,
    {
        async fn text_turn(
            &self,
            input: ModelInput,
            turn: AdapterTextTurn,
        ) -> Result<ErasedTextTurnEventStream, AgentError> {
            self.text_turns.lock().unwrap().push(turn.clone());
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

    struct CapturingCompleteSink {
        completed: Rc<RefCell<Vec<(String, u64, String)>>>,
    }

    #[async_trait(?Send)]
    impl UtteranceSink for CapturingCompleteSink {
        async fn on_complete(&self, utterance: Utterance) -> Result<(), PortError> {
            self.completed.borrow_mut().push((
                utterance.target,
                utterance.generation_id,
                utterance.text,
            ));
            Ok(())
        }
    }

    fn prepare_speech_scenario(target: &str, speech_content: &str) -> MockTextScenario {
        let arguments_json = serde_json::json!({
            "target": target,
            "speech_content": speech_content
        })
        .to_string();
        MockTextScenario::events(vec![
            Ok(RawTextTurnEvent::Started {
                request_id: Some("prepare-speech".into()),
                model: "mock".into(),
            }),
            Ok(RawTextTurnEvent::ToolCallChunk {
                id: "call-prepare-speech".into(),
                name: "prepare_speech".into(),
                arguments_json_delta: arguments_json,
            }),
            Ok(RawTextTurnEvent::Completed {
                request_id: Some("prepare-speech".into()),
                finish_reason: FinishReason::ToolCall,
                usage: Usage::zero(),
            }),
        ])
    }

    fn decline_speech_now_scenario(blocking_reason: &str) -> MockTextScenario {
        decline_speech_now_scenario_with_inhibit(blocking_reason, None)
    }

    fn decline_speech_now_scenario_with_inhibit(
        blocking_reason: &str,
        inhibit_reason: Option<&str>,
    ) -> MockTextScenario {
        let arguments_json = serde_json::json!({
            "blocking_reason": blocking_reason,
            "inhibit_reason": inhibit_reason
        })
        .to_string();
        MockTextScenario::events(vec![
            Ok(RawTextTurnEvent::Started {
                request_id: Some("decline-speech-now".into()),
                model: "mock".into(),
            }),
            Ok(RawTextTurnEvent::ToolCallChunk {
                id: "call-decline-speech-now".into(),
                name: "decline_speech_now".into(),
                arguments_json_delta: arguments_json,
            }),
            Ok(RawTextTurnEvent::Completed {
                request_id: Some("decline-speech-now".into()),
                finish_reason: FinishReason::ToolCall,
                usage: Usage::zero(),
            }),
        ])
    }

    fn finish_without_planning_tool_scenario() -> MockTextScenario {
        MockTextScenario::events(vec![
            Ok(RawTextTurnEvent::Started {
                request_id: Some("prepare-speech".into()),
                model: "mock".into(),
            }),
            Ok(RawTextTurnEvent::Completed {
                request_id: Some("prepare-speech".into()),
                finish_reason: FinishReason::Stop,
                usage: Usage::zero(),
            }),
        ])
    }

    fn test_caps_with_turn_adapter<T>(
        blackboard: Blackboard,
        adapter: Arc<T>,
    ) -> CapabilityProviders
    where
        T: TurnAdapter,
    {
        let budget = SharedPoolBudgetManager::new(SharedPoolBudgetOptions::default());
        let lutum = lutum::Lutum::new(adapter, budget);
        CapabilityProviders::new(CapabilityProviderPorts {
            blackboard,
            cognition_log_port: Rc::new(NoopCognitionLogRepository),
            clock: Rc::new(SystemClock),
            tiers: LutumTiers::from_shared_lutum(lutum),
        })
    }

    async fn speak_module_with_turn_adapter<T>(
        blackboard: Blackboard,
        adapter: Arc<T>,
        sink: Rc<dyn UtteranceSink>,
    ) -> (SpeakModule, CapabilityProviders)
    where
        T: TurnAdapter,
    {
        let (module, caps, _attention_control) =
            speak_module_with_turn_adapter_and_attention_control(blackboard, adapter, sink).await;
        (module, caps)
    }

    async fn speak_module_with_turn_adapter_and_attention_control<T>(
        blackboard: Blackboard,
        adapter: Arc<T>,
        sink: Rc<dyn UtteranceSink>,
    ) -> (
        SpeakModule,
        CapabilityProviders,
        nuillu_module::AttentionControlRequestInbox,
    )
    where
        T: TurnAdapter,
    {
        let caps = test_caps_with_turn_adapter(blackboard, adapter);
        let module_cell = Rc::new(RefCell::new(None));
        let module_sink = Rc::clone(&module_cell);
        let attention_control_cell = Rc::new(RefCell::new(None));
        let attention_control_sink = Rc::clone(&attention_control_cell);
        let utterance_sink_for_closure = sink.clone();

        let _modules = ModuleRegistry::new()
            .register(test_policy(), move |caps| {
                let module_sink = Rc::clone(&module_sink);
                let attention_control_sink = Rc::clone(&attention_control_sink);
                let utterance_sink_for_closure = utterance_sink_for_closure.clone();
                async move {
                    *attention_control_sink.borrow_mut() = Some(caps.attention_control_inbox());
                    *module_sink.borrow_mut() = Some(SpeakModule::new(SpeakModuleParts {
                        cognition_updates: caps.cognition_log_updated_inbox(),
                        cognition_log: caps.cognition_log_reader(),
                        attention_control: caps.attention_control_mailbox(),
                        memo: caps.memo(),
                        utterance: UtteranceWriter::new(
                            caps.owner().clone(),
                            caps.blackboard(),
                            utterance_sink_for_closure.clone(),
                            caps.clock(),
                        ),
                        planning_llm: caps
                            .llm("planning")
                            .with_tier(nuillu_types::ModelTier::Premium)
                            .into(),
                        scene: caps.scene_reader(),
                        clock: caps.clock(),
                        planning_session: caps
                            .session("planning")
                            .with_tier(nuillu_types::ModelTier::Premium)
                            .with_auto_compaction(planning_session_auto_compaction())
                            .await?,
                    }));
                    Ok(SpeakStub)
                }
            })
            .unwrap()
            .build(&caps)
            .await
            .unwrap();

        let module = module_cell.borrow_mut().take().unwrap();
        let attention_control = attention_control_cell.borrow_mut().take().unwrap();
        (module, caps, attention_control)
    }

    async fn publish_cognition_update(
        blackboard: &Blackboard,
        caps: &CapabilityProviders,
        at: chrono::DateTime<chrono::Utc>,
        text: impl Into<String>,
    ) {
        let source = ModuleInstanceId::new(builtin::cognition_gate(), ReplicaIndex::ZERO);
        let origin = ModuleInstanceId::new(builtin::sensory(), ReplicaIndex::ZERO);
        blackboard
            .apply(BlackboardCommand::AppendCognitionLog {
                source: source.clone(),
                entry: CognitionLogEntry {
                    at,
                    text: text.into(),
                    origin: CognitionLogOrigin::memo(origin, 0),
                },
            })
            .await;
        caps.internal_harness_io()
            .cognition_log_updated_mailbox()
            .publish(CognitionLogUpdated::EntryAppended { source })
            .await
            .unwrap();
    }

    async fn test_activate_cx(
        module: &SpeakModule,
        now: chrono::DateTime<chrono::Utc>,
    ) -> nuillu_module::ActivateCx<'static> {
        let compaction_lutum = module.planning_llm.lutum().await;
        nuillu_module::ActivateCx::new(
            &[],
            &[],
            &[],
            nuillu_module::SessionCompactionRuntime::new(
                compaction_lutum.lutum().clone(),
                nuillu_module::LlmConcurrencyLimiter::new(None),
                nuillu_types::ModelTier::Cheap,
                nuillu_module::SessionCompactionPolicy::default(),
            ),
            now,
        )
    }

    async fn try_activate_once_with_adapter(
        adapter: MockLlmAdapter,
        participants: impl IntoIterator<Item = Participant>,
        cognition: impl Into<String>,
    ) -> Result<(Blackboard, Rc<RefCell<Vec<(String, String)>>>)> {
        let mut allocation = ResourceAllocation::default();
        allocation.set_activation(builtin::speak(), ActivationRatio::ONE);
        let blackboard = Blackboard::with_allocation(allocation);
        let completed = Rc::new(RefCell::new(Vec::new()));
        let sink: Rc<dyn crate::utterance::UtteranceSink> = Rc::new(CapturingUtteranceSink {
            completed: Rc::clone(&completed),
            done: RefCell::new(None),
        });
        let (mut module, caps) =
            speak_module_with_turn_adapter(blackboard.clone(), Arc::new(adapter), sink).await;
        caps.scene().set(participants);
        let now = SystemClock.now();
        publish_cognition_update(&blackboard, &caps, now, cognition).await;
        let batch = module.next_batch().await.unwrap();
        let cx = test_activate_cx(&module, now).await;
        SpeakModule::activate(&mut module, &cx, &batch).await?;
        Ok((blackboard, completed))
    }

    async fn activate_once_with_adapter(
        adapter: MockLlmAdapter,
        participants: impl IntoIterator<Item = Participant>,
        cognition: impl Into<String>,
    ) -> (Blackboard, Rc<RefCell<Vec<(String, String)>>>) {
        try_activate_once_with_adapter(adapter, participants, cognition)
            .await
            .unwrap()
    }

    async fn speak_memo_count(blackboard: &Blackboard) -> usize {
        speak_memos(blackboard).await.len()
    }

    async fn speak_memos(blackboard: &Blackboard) -> Vec<String> {
        blackboard
            .read(|bb| {
                bb.recent_memo_logs()
                    .into_iter()
                    .filter(|record| record.owner.module == builtin::speak())
                    .map(|record| record.content)
                    .collect()
            })
            .await
    }

    async fn speak_progress_exists(blackboard: &Blackboard) -> bool {
        let speak_owner = ModuleInstanceId::new(builtin::speak(), ReplicaIndex::ZERO);
        blackboard
            .read(|bb| bb.utterance_progress_for_instance(&speak_owner).is_some())
            .await
    }

    fn session_input_text(session: &Session) -> String {
        model_input_text(session.input())
    }

    fn model_input_text(input: &ModelInput) -> String {
        let mut out = String::new();
        for item in input.items() {
            match item {
                ModelInputItem::Message { content, .. } => {
                    for content in content.as_slice() {
                        if let MessageContent::Text(text) = content {
                            out.push_str(text);
                            out.push('\n');
                        }
                    }
                }
                ModelInputItem::Assistant(lutum::AssistantInputItem::Text(text)) => {
                    out.push_str(text);
                    out.push('\n');
                }
                _ => {}
            }
        }
        out
    }

    fn session_message_texts(session: &Session, expected_role: InputMessageRole) -> Vec<String> {
        session
            .input()
            .items()
            .iter()
            .filter_map(|item| {
                let ModelInputItem::Message { role, content } = item else {
                    return None;
                };
                if role != &expected_role {
                    return None;
                }
                let text = content
                    .as_slice()
                    .iter()
                    .filter_map(|content| match content {
                        MessageContent::Text(text) => Some(text.as_str()),
                        _ => None,
                    })
                    .collect::<Vec<_>>()
                    .join("\n");
                Some(text)
            })
            .collect()
    }

    #[tokio::test(flavor = "current_thread")]
    async fn speech_cognition_context_uses_shared_formatter() {
        let blackboard = Blackboard::with_allocation(ResourceAllocation::default());
        let sink: Rc<dyn crate::utterance::UtteranceSink> = Rc::new(CapturingUtteranceSink {
            completed: Rc::new(RefCell::new(Vec::new())),
            done: RefCell::new(None),
        });
        let (module, _caps) = speak_module_with_turn_adapter(
            blackboard.clone(),
            Arc::new(MockLlmAdapter::new()),
            sink,
        )
        .await;
        let now = SystemClock.now();
        let source = ModuleInstanceId::new(builtin::cognition_gate(), ReplicaIndex::ZERO);
        let origin = ModuleInstanceId::new(builtin::sensory(), ReplicaIndex::ZERO);
        blackboard
            .apply(BlackboardCommand::AppendCognitionLog {
                source: source.clone(),
                entry: CognitionLogEntry {
                    at: now - chrono::Duration::minutes(4),
                    text: "Koro asks Nuillu for help.".into(),
                    origin: CognitionLogOrigin::memo(origin, 0),
                },
            })
            .await;

        let context = module.speech_cognition_context(now).await;

        assert!(context.contains("What you are currently thinking at "));
        assert!(context.contains("About 4 minutes ago: Koro asks Nuillu for help."));
    }

    #[test]
    fn speech_cognition_context_uses_allowed_origin_modules_only() {
        let now = SystemClock.now();
        let cognition_gate = ModuleInstanceId::new(builtin::cognition_gate(), ReplicaIndex::ZERO);
        let sensory = ModuleInstanceId::new(builtin::sensory(), ReplicaIndex::ZERO);
        let query_memory = ModuleInstanceId::new(builtin::query_memory(), ReplicaIndex::ZERO);
        let interpreter = ModuleInstanceId::new(builtin::interpreter(), ReplicaIndex::ZERO);
        let predict = ModuleInstanceId::new(builtin::predict(), ReplicaIndex::ZERO);
        let attention_schema =
            ModuleInstanceId::new(builtin::attention_schema(), ReplicaIndex::ZERO);
        let memory = ModuleInstanceId::new(builtin::memory(), ReplicaIndex::ZERO);
        let speak = ModuleInstanceId::new(builtin::speak(), ReplicaIndex::ZERO);
        let records = vec![
            CognitionLogEntryRecord {
                index: 0,
                source: cognition_gate.clone(),
                entry: CognitionLogEntry {
                    at: now,
                    text: "sensory evidence".into(),
                    origin: CognitionLogOrigin::memo(sensory, 0),
                },
            },
            CognitionLogEntryRecord {
                index: 1,
                source: cognition_gate.clone(),
                entry: CognitionLogEntry {
                    at: now,
                    text: "query-memory evidence".into(),
                    origin: CognitionLogOrigin::memo(query_memory, 0),
                },
            },
            CognitionLogEntryRecord {
                index: 2,
                source: cognition_gate.clone(),
                entry: CognitionLogEntry {
                    at: now,
                    text: "interpreter evidence".into(),
                    origin: CognitionLogOrigin::memo(interpreter, 0),
                },
            },
            CognitionLogEntryRecord {
                index: 3,
                source: cognition_gate.clone(),
                entry: CognitionLogEntry {
                    at: now,
                    text: "predict evidence".into(),
                    origin: CognitionLogOrigin::memo(predict, 0),
                },
            },
            CognitionLogEntryRecord {
                index: 4,
                source: cognition_gate.clone(),
                entry: CognitionLogEntry {
                    at: now,
                    text: "attention-schema evidence".into(),
                    origin: CognitionLogOrigin::memo(attention_schema, 0),
                },
            },
            CognitionLogEntryRecord {
                index: 5,
                source: cognition_gate.clone(),
                entry: CognitionLogEntry {
                    at: now,
                    text: "memory evidence should not be included".into(),
                    origin: CognitionLogOrigin::memo(memory, 0),
                },
            },
            CognitionLogEntryRecord {
                index: 6,
                source: cognition_gate,
                entry: CognitionLogEntry {
                    at: now,
                    text: "speak evidence should not be included".into(),
                    origin: CognitionLogOrigin::memo(speak, 0),
                },
            },
        ];

        let context = speech_cognition_context_from_entries(&records, now).unwrap();

        assert!(context.contains("sensory evidence"));
        assert!(context.contains("query-memory evidence"));
        assert!(context.contains("interpreter evidence"));
        assert!(context.contains("predict evidence"));
        assert!(context.contains("attention-schema evidence"));
        assert!(!context.contains("memory evidence should not be included"));
        assert!(!context.contains("speak evidence should not be included"));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn next_batch_captures_only_unread_cognition_entries() {
        let blackboard = Blackboard::with_allocation(ResourceAllocation::default());
        let sink: Rc<dyn crate::utterance::UtteranceSink> = Rc::new(CapturingUtteranceSink {
            completed: Rc::new(RefCell::new(Vec::new())),
            done: RefCell::new(None),
        });
        let (mut module, caps) = speak_module_with_turn_adapter(
            blackboard.clone(),
            Arc::new(MockLlmAdapter::new()),
            sink,
        )
        .await;
        let now = SystemClock.now();

        publish_cognition_update(&blackboard, &caps, now, "first cognition").await;
        let first = module.next_batch().await.unwrap();
        assert_eq!(
            first
                .cognition_entries
                .iter()
                .map(|record| record.entry.text.as_str())
                .collect::<Vec<_>>(),
            vec!["first cognition"]
        );

        publish_cognition_update(
            &blackboard,
            &caps,
            now + chrono::Duration::seconds(1),
            "second cognition",
        )
        .await;
        let second = module.next_batch().await.unwrap();
        assert_eq!(
            second
                .cognition_entries
                .iter()
                .map(|record| record.entry.text.as_str())
                .collect::<Vec<_>>(),
            vec!["second cognition"]
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn planning_context_uses_new_entries_for_each_batch() {
        let adapter = MockLlmAdapter::new()
            .with_text_scenario(decline_speech_now_scenario("not enough to say yet"))
            .with_text_scenario(decline_speech_now_scenario("still not enough to say"));
        let mut allocation = ResourceAllocation::default();
        allocation.set_activation(builtin::speak(), ActivationRatio::ONE);
        let blackboard = Blackboard::with_allocation(allocation);
        let sink: Rc<dyn crate::utterance::UtteranceSink> = Rc::new(CapturingUtteranceSink {
            completed: Rc::new(RefCell::new(Vec::new())),
            done: RefCell::new(None),
        });
        let (mut module, caps) =
            speak_module_with_turn_adapter(blackboard.clone(), Arc::new(adapter), sink).await;
        caps.scene().set([Participant::new("Koro")]);
        let now = SystemClock.now();

        publish_cognition_update(&blackboard, &caps, now, "first cognition").await;
        let first_batch = module.next_batch().await.unwrap();
        let cx = test_activate_cx(&module, now).await;
        SpeakModule::activate(&mut module, &cx, &first_batch)
            .await
            .unwrap();

        publish_cognition_update(
            &blackboard,
            &caps,
            now + chrono::Duration::seconds(1),
            "second cognition",
        )
        .await;
        let second_batch = module.next_batch().await.unwrap();
        let cx = test_activate_cx(&module, now + chrono::Duration::seconds(1)).await;
        SpeakModule::activate(&mut module, &cx, &second_batch)
            .await
            .unwrap();

        let contexts = session_message_texts(&module.planning_session, InputMessageRole::User)
            .into_iter()
            .filter(|text| text.starts_with("New thoughts available to you at "))
            .collect::<Vec<_>>();
        assert_eq!(contexts.len(), 2);
        assert!(contexts[0].contains("first cognition"));
        assert!(!contexts[0].contains("second cognition"));
        assert!(contexts[1].contains("second cognition"));
        assert!(!contexts[1].contains("first cognition"));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn empty_unread_batch_does_not_plan_from_snapshot() {
        let blackboard = Blackboard::with_allocation(ResourceAllocation::default());
        let sink: Rc<dyn crate::utterance::UtteranceSink> = Rc::new(CapturingUtteranceSink {
            completed: Rc::new(RefCell::new(Vec::new())),
            done: RefCell::new(None),
        });
        let (mut module, caps) = speak_module_with_turn_adapter(
            blackboard.clone(),
            Arc::new(MockLlmAdapter::new()),
            sink,
        )
        .await;
        publish_cognition_update(
            &blackboard,
            &caps,
            SystemClock.now(),
            "old cognition that should not be replanned",
        )
        .await;
        let old_batch = module.next_batch().await.unwrap();
        assert_eq!(old_batch.cognition_entries.len(), 1);

        caps.internal_harness_io()
            .cognition_log_updated_mailbox()
            .publish(CognitionLogUpdated::AgenticDeadlockMarker)
            .await
            .unwrap();

        let batch = module.next_batch().await.unwrap();
        assert!(batch.cognition_entries.is_empty());
        let cx = test_activate_cx(&module, SystemClock.now()).await;
        SpeakModule::activate(&mut module, &cx, &batch)
            .await
            .unwrap();

        assert!(module.planning_session.input().items().is_empty());
        assert_eq!(speak_memo_count(&blackboard).await, 0);
        assert!(!speak_progress_exists(&blackboard).await);
    }

    #[test]
    fn planning_prompt_defines_direct_short_speech_content_boundary() {
        let prompt = nuillu_module::format_policy_system_prompt(SPEECH_PLANNING_PROMPT, &[]);

        assert!(prompt.contains("speech_content is the exact listener-visible utterance text"));
        assert!(prompt.contains("not a directive, summary, rationale, or future plan"));
        assert!(prompt.contains("one short in-world utterance"));
        assert!(prompt.contains("The brain may receive new sensory input"));
        assert!(prompt.contains("one second later"));
        assert!(prompt.contains("comma or Japanese comma"));
        assert!(prompt.contains("Do not wrap speech_content in quotation marks."));
        assert!(prompt.contains("Do not summarize the request or say what the user wants."));
        assert!(prompt.contains("Do not output introspection, narration, analysis"));
        assert!(prompt.contains("implementation mechanics"));
        assert!(prompt.contains("prompts, rubrics, judges, or evaluation mechanics"));
        assert!(prompt.contains("write speech_content itself in that language"));
        assert!(!prompt.contains("Already emitted"));
        assert!(!prompt.contains("continue_speech"));
        assert!(!prompt.contains("redirect_speech"));
        assert!(!prompt.contains("append_directive"));
        assert!(!prompt.contains("utterance_directive"));
        assert!(!prompt.contains("Planner Directive"));
        assert!(!prompt.contains("Append Directive"));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn prepare_speech_emits_speech_content_exactly_and_records_completion() {
        let adapter = MockLlmAdapter::new()
            .with_text_scenario(prepare_speech_scenario("Koro", "Koro, stay close."));
        let capture = CapturingAdapter::new(adapter);
        let observed = capture.clone();
        let mut allocation = ResourceAllocation::default();
        allocation.set_activation(builtin::speak(), ActivationRatio::ONE);
        let blackboard = Blackboard::with_allocation(allocation);
        let completed = Rc::new(RefCell::new(Vec::new()));
        let sink: Rc<dyn crate::utterance::UtteranceSink> = Rc::new(CapturingCompleteSink {
            completed: Rc::clone(&completed),
        });
        let (mut module, caps) =
            speak_module_with_turn_adapter(blackboard.clone(), Arc::new(capture), sink).await;
        caps.scene().set([Participant::new("Koro")]);
        let source = ModuleInstanceId::new(builtin::cognition_gate(), ReplicaIndex::ZERO);
        let origin = ModuleInstanceId::new(builtin::sensory(), ReplicaIndex::ZERO);
        blackboard
            .apply(BlackboardCommand::AppendCognitionLog {
                source: source.clone(),
                entry: CognitionLogEntry {
                    at: SystemClock.now(),
                    text: "Koro asks Nuillu to help them stay safe.".into(),
                    origin: CognitionLogOrigin::memo(origin, 0),
                },
            })
            .await;
        caps.internal_harness_io()
            .cognition_log_updated_mailbox()
            .publish(CognitionLogUpdated::EntryAppended { source })
            .await
            .unwrap();

        let batch = module.next_batch().await.unwrap();
        let catalog = Vec::new();
        let identity_memories = vec![IdentityMemoryRecord {
            index: MemoryIndex::new("identity-1"),
            content: MemoryContent::new("Nui is a small blue frog."),
            occurred_at: None,
        }];
        let compaction_lutum = module.planning_llm.lutum().await;
        let clock = SystemClock;
        let cx = nuillu_module::ActivateCx::new(
            &catalog,
            &identity_memories,
            &[],
            nuillu_module::SessionCompactionRuntime::new(
                compaction_lutum.lutum().clone(),
                nuillu_module::LlmConcurrencyLimiter::new(None),
                nuillu_types::ModelTier::Cheap,
                nuillu_module::SessionCompactionPolicy::default(),
            ),
            clock.now(),
        );
        SpeakModule::activate(&mut module, &cx, &batch)
            .await
            .unwrap();

        assert_eq!(
            completed.borrow().as_slice(),
            &[("Koro".to_string(), 0, "Koro, stay close.".to_string())]
        );
        let speak_owner = ModuleInstanceId::new(builtin::speak(), ReplicaIndex::ZERO);
        let progress = blackboard
            .read(|bb| bb.utterance_progress_for_instance(&speak_owner).cloned())
            .await
            .unwrap();
        assert_eq!(progress.target, "Koro");
        assert_eq!(progress.generation_id, 0);
        assert_eq!(progress.sequence, 0);
        assert_eq!(
            progress.state,
            nuillu_blackboard::UtteranceProgressState::Completed
        );
        assert_eq!(progress.partial_utterance, "Koro, stay close.");

        let memos = speak_memos(&blackboard).await;
        assert_eq!(memos.len(), 1);
        assert_eq!(memos[0], "I said to Koro:\nKoro, stay close.");

        let planning_text = session_input_text(&module.planning_session);
        assert_eq!(planning_text.matches("Your identity:").count(), 1);
        assert!(!planning_text.contains("Identity memory loaded at agent startup"));
        assert!(planning_text.contains("Koro asks Nuillu to help them stay safe."));
        assert!(planning_text.contains("speech_content"));
        assert!(planning_text.contains("Completed outward utterance to Koro:\nKoro, stay close."));
        assert!(!planning_text.contains("Already emitted"));
        assert!(!planning_text.contains("generation"));
        assert!(!planning_text.contains("utterance_directive"));

        let inputs = observed.text_inputs();
        assert_eq!(inputs.len(), 1);
        let planning_input = model_input_text(&inputs[0]);
        assert!(planning_input.contains("speech_content"));
        assert!(planning_input.contains("Koro asks Nuillu to help them stay safe."));
        assert!(!planning_input.contains("Already emitted"));
        assert!(!planning_input.contains("Planner Directive"));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn language_request_is_handled_inside_speech_content() {
        let adapter = MockLlmAdapter::new()
            .with_text_scenario(prepare_speech_scenario("Ryo", "もちろん、日本語で話すよ。"));

        let (_blackboard, completed) = activate_once_with_adapter(
            adapter,
            [Participant::new("Ryo")],
            "Ryo says, \"日本語でお願い\".",
        )
        .await;

        assert_eq!(
            completed.borrow().as_slice(),
            &[("Ryo".to_string(), "もちろん、日本語で話すよ。".to_string())]
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn speak_sets_max_output_tokens_for_planning_only() {
        let adapter = MockLlmAdapter::new()
            .with_text_scenario(prepare_speech_scenario("Koro", "Koro, stay close."));
        let capture = CapturingAdapter::new(adapter);
        let observed = capture.clone();
        let mut allocation = ResourceAllocation::default();
        allocation.set_activation(builtin::speak(), ActivationRatio::ONE);
        let blackboard = Blackboard::with_allocation(allocation);
        let sink: Rc<dyn crate::utterance::UtteranceSink> = Rc::new(CapturingUtteranceSink {
            completed: Rc::new(RefCell::new(Vec::new())),
            done: RefCell::new(None),
        });
        let (mut module, caps) =
            speak_module_with_turn_adapter(blackboard.clone(), Arc::new(capture), sink).await;
        caps.scene().set([Participant::new("Koro")]);
        let now = SystemClock.now();
        publish_cognition_update(
            &blackboard,
            &caps,
            now,
            "Koro asks Nuillu to help them stay safe.",
        )
        .await;

        let batch = module.next_batch().await.unwrap();
        let cx = test_activate_cx(&module, now).await;
        SpeakModule::activate(&mut module, &cx, &batch)
            .await
            .unwrap();

        let turns = observed.text_turns();
        assert_eq!(turns.len(), 1);
        assert_eq!(
            turns[0].config.generation.max_output_tokens,
            Some(MaxOutputTokens::new(SPEECH_PLANNING_TURN_MAX_OUTPUT_TOKENS))
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn speak_stays_silent_when_decline_speech_now_is_called() {
        let adapter = MockLlmAdapter::new().with_text_scenario(decline_speech_now_scenario(
            "no supported listener-facing content",
        ));

        let (blackboard, completed) = activate_once_with_adapter(
            adapter,
            [Participant::new("Koro")],
            "Koro asks whether Nui should say anything.",
        )
        .await;

        assert!(completed.borrow().is_empty());
        let memos = speak_memos(&blackboard).await;
        assert_eq!(memos.len(), 1);
        assert!(memos[0].contains("I am staying silent for now. Reason:"));
        assert!(memos[0].contains("no supported listener-facing content"));
        assert!(!speak_progress_exists(&blackboard).await);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn repeated_greeting_decline_publishes_attention_inhibit_reason() {
        let inhibit_reason =
            "No new listener-facing cognition since the previous greeting; avoid repeating it.";
        let adapter =
            MockLlmAdapter::new().with_text_scenario(decline_speech_now_scenario_with_inhibit(
                "Ryo repeated the same greeting and there is no new content to say.",
                Some(inhibit_reason),
            ));
        let mut allocation = ResourceAllocation::default();
        allocation.set_activation(builtin::speak(), ActivationRatio::ONE);
        let blackboard = Blackboard::with_allocation(allocation);
        let completed = Rc::new(RefCell::new(Vec::new()));
        let sink: Rc<dyn crate::utterance::UtteranceSink> = Rc::new(CapturingUtteranceSink {
            completed: Rc::clone(&completed),
            done: RefCell::new(None),
        });
        let (mut module, caps, mut attention_control) =
            speak_module_with_turn_adapter_and_attention_control(
                blackboard.clone(),
                Arc::new(adapter),
                sink,
            )
            .await;
        caps.scene().set([Participant::new("Ryo")]);
        let now = SystemClock.now();
        publish_cognition_update(&blackboard, &caps, now, "Ryo says, \"こんにちは！\".").await;

        let batch = module.next_batch().await.unwrap();
        let cx = test_activate_cx(&module, now).await;
        SpeakModule::activate(&mut module, &cx, &batch)
            .await
            .unwrap();

        assert!(completed.borrow().is_empty());
        let envelope = attention_control
            .next_item()
            .await
            .expect("speak inhibit request should be published");
        assert_eq!(envelope.sender.module, builtin::speak());
        assert_eq!(
            envelope.body.kind(),
            nuillu_module::AttentionControlRequestKind::Inhibit
        );
        assert_eq!(envelope.body.as_str(), inhibit_reason);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn speak_planning_rejects_finishing_without_required_tool() {
        let adapter =
            MockLlmAdapter::new().with_text_scenario(finish_without_planning_tool_scenario());

        let error = try_activate_once_with_adapter(
            adapter,
            [Participant::new("Koro")],
            "Koro asks whether Nui should say anything.",
        )
        .await
        .unwrap_err();

        let error = format!("{error:#}");
        assert!(
            error.contains("required tool call") || error.contains("without required tool call"),
            "unexpected error: {error}"
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn failed_planning_turn_does_not_persist_planning_context() {
        let adapter =
            MockLlmAdapter::new().with_text_scenario(finish_without_planning_tool_scenario());
        let mut allocation = ResourceAllocation::default();
        allocation.set_activation(builtin::speak(), ActivationRatio::ONE);
        let blackboard = Blackboard::with_allocation(allocation);
        let sink: Rc<dyn crate::utterance::UtteranceSink> = Rc::new(CapturingUtteranceSink {
            completed: Rc::new(RefCell::new(Vec::new())),
            done: RefCell::new(None),
        });
        let (mut module, caps) =
            speak_module_with_turn_adapter(blackboard.clone(), Arc::new(adapter), sink).await;
        caps.scene().set([Participant::new("Koro")]);
        let now = SystemClock.now();
        publish_cognition_update(
            &blackboard,
            &caps,
            now,
            "Koro asks whether Nui should say anything.",
        )
        .await;

        let batch = module.next_batch().await.unwrap();
        let cx = test_activate_cx(&module, now).await;
        SpeakModule::activate(&mut module, &cx, &batch)
            .await
            .expect_err("planning without a required tool should fail");

        let planning_users =
            session_message_texts(&module.planning_session, InputMessageRole::User);
        assert!(planning_users.is_empty());
        assert!(!session_input_text(&module.planning_session).contains("Koro asks"));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn speak_stays_silent_for_empty_planned_target() {
        let adapter = MockLlmAdapter::new()
            .with_text_scenario(prepare_speech_scenario("   ", "Koro, stay close."));

        let (blackboard, completed) = activate_once_with_adapter(
            adapter,
            [Participant::new("Koro")],
            "Koro asks whether Nui should say anything.",
        )
        .await;

        assert!(completed.borrow().is_empty());
        assert_eq!(speak_memo_count(&blackboard).await, 0);
        assert!(!speak_progress_exists(&blackboard).await);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn speak_stays_silent_for_empty_speech_content() {
        let adapter =
            MockLlmAdapter::new().with_text_scenario(prepare_speech_scenario("Koro", "   "));

        let (blackboard, completed) = activate_once_with_adapter(
            adapter,
            [Participant::new("Koro")],
            "Koro asks whether Nui should say anything.",
        )
        .await;

        assert!(completed.borrow().is_empty());
        assert_eq!(speak_memo_count(&blackboard).await, 0);
        assert!(!speak_progress_exists(&blackboard).await);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn speak_emits_for_non_scene_planned_target() {
        let adapter = MockLlmAdapter::new().with_text_scenario(prepare_speech_scenario(
            "OffstageVoice",
            "Stop that, please.",
        ));

        let (blackboard, completed) = activate_once_with_adapter(
            adapter,
            [Participant::new("Pibi")],
            "An offstage voice is causing trouble and should be addressed.",
        )
        .await;

        assert_eq!(
            completed.borrow().as_slice(),
            &[(
                "OffstageVoice".to_string(),
                "Stop that, please.".to_string()
            )]
        );
        let speak_owner = ModuleInstanceId::new(builtin::speak(), ReplicaIndex::ZERO);
        let progress = blackboard
            .read(|bb| bb.utterance_progress_for_instance(&speak_owner).cloned())
            .await
            .unwrap();
        assert_eq!(progress.target, "OffstageVoice");
        assert_eq!(
            progress.state,
            nuillu_blackboard::UtteranceProgressState::Completed
        );
    }

    #[test]
    fn speak_batch_keeps_drained_updates() {
        let batch = SpeakBatch {
            updates: vec![CognitionLogUpdated::AgenticDeadlockMarker],
            cognition_entries: Vec::new(),
        };

        assert_eq!(
            batch.updates,
            vec![CognitionLogUpdated::AgenticDeadlockMarker]
        );
    }
}
