use std::borrow::Cow;
use std::future::Future;
use std::pin::Pin;
use std::rc::Rc;

use anyhow::{Context, Result};
use async_trait::async_trait;
use chrono::{DateTime, SecondsFormat, Utc};
use futures::StreamExt;
use lutum::{
    ModelInput, ModelInputItem, Session, TextStepOutcomeWithTools, TextTurnEvent, ToolResult, Usage,
};
use nuillu_blackboard::{CognitionLogEntryRecord, CognitionLogRecord};
use nuillu_module::{
    CognitionLogReader, CognitionLogUpdated, CognitionLogUpdatedInbox, LlmAccess, LlmContextWindow,
    Memo, Module, SceneReader, SessionAutoCompaction, SessionCompactionConfig,
    SessionCompactionProtectedPrefix, UtteranceProgress, ensure_persistent_session_seeded,
    format_bounded_cognition_log_batch, ports::Clock,
};

use crate::utterance::UtteranceWriter;
use nuillu_types::builtin;
use schemars::{JsonSchema, Schema, SchemaGenerator};
use serde::{Deserialize, Serialize};

const SPEECH_PLANNING_PROMPT: &str = r#"Plan outward speech after allocation has raised the speak module.
Use only the cognition log and the available target schema. Do not casually re-decide whether speak should have been allocated.
Call prepare_speech exactly once when a grounded outward utterance can be prepared. Choose the participant whose question, request, warning, or need should be answered. Do not choose a participant merely because they are the topic, threat, object of advice, or quoted speaker. Use "everyone" only for explicit group/broadcast speech.
Call decline_speech_now only when a concrete blocker makes speech inappropriate or impossible despite allocation, such as no allowed target, no cognition-supported listener-facing content, a policy or consent conflict, or fresh evidence that invalidates speaking now. Put that blocker in blocking_reason.
Put the speech-facing transformation of the cognition log in speech_content. It is the information that should survive into speech, with perspective, deixis, and addressee adjusted for outward utterance.
speech_content is not hidden reasoning, not a rubric, and not a generic summary. It should contain the load-bearing fact, answer, warning, advice, visible absence, or unknown-state evidence that the listener needs.
For questions or requests, transform the relevant cognition into an answer. Preserve answer polarity: yes/no/unknown must remain visible when supported by the cognition log.
For self-directed cognition, transform only the listener-relevant implication into outward substance. Keep first person only when the speaker is reporting perception, knowledge, uncertainty, consent, or shared action that directly answers the listener.
Do not invent policy, actions, or facts not supported by the cognition log. If the cognition log only supports a limited warning or uncertainty, keep speech_content limited.
For unknown evidence, make speech_content say unknown and include the concrete visible absence or missing evidence."#;

const PLANNING_TURN_DEVELOPER_INSTRUCTION: &str = "Speak has already been allocated. Use exactly one tool: prepare_speech when listener-facing substance can be produced, or decline_speech_now only for a concrete blocker that makes speech inappropriate or impossible now.";
const PLANNING_TURN_FINAL_REMINDER: &str = nuillu_module::REQUIRED_FUNCTION_CALL_REMINDER;

const GENERATION_PROMPT: &str = r#"Render the supplied substance as one concise in-world utterance to the named listener.
The substance is already transformed for outward speech. Render that transformed information; do not redo listener selection or add a new plan.
Preserve its answer polarity, addressee-facing perspective, direct warnings, advice, uncertainty, and visible-absence evidence.
Use the cognition log only to keep wording grounded. Do not add facts, turn an answer back into a question, turn listener-facing substance into a self-directed note, weaken direct content into vague caution, or merely restate the situation.
Do not mention implementation mechanics, lookup, reasoning, prompts, rubrics, or evaluation mechanics."#;

const PARTIAL_CONTINUATION_PROMPT: &str = "Continue the partial utterance from where it stopped.";

const INTERRUPT_TURN_DEVELOPER_INSTRUCTION: &str = "A speech is already streaming. Use exactly one tool: continue_speech if the current speech remains appropriate, interrupt_speech if new cognition requires a replacement utterance now, or stop_speech if the current speech should stop and no replacement is warranted.";
const COGNITION_CONTEXT_WINDOW: LlmContextWindow = LlmContextWindow::new(12, 600, 4_800);
const SPEECH_PLANNING_TURN_MAX_OUTPUT_TOKENS: u32 = 768;
const SPEECH_GENERATION_MAX_OUTPUT_TOKENS: u32 = 256;
const COMPACTED_SPEAK_PLANNING_SESSION_PREFIX: &str = "Compacted speak planning session history:";
const COMPACTED_SPEAK_GENERATION_SESSION_PREFIX: &str =
    "Compacted speak generation session history:";
const PLANNING_SESSION_COMPACTION_FOCUS: &str = r#"Preserve prior speech target decisions,
selected targets, rejected/no-speech decisions, interruption decisions, and cognition-log context
needed for future speak planning."#;
const GENERATION_SESSION_COMPACTION_FOCUS: &str = r#"Preserve completed outward utterances,
their addressees, speech substance, and cognition-log context needed for future utterance rendering."#;

pub fn planning_session_auto_compaction() -> SessionAutoCompaction {
    SessionAutoCompaction::new(
        SessionCompactionConfig::default(),
        SessionCompactionProtectedPrefix::LeadingSystemAndIdentitySeed,
        COMPACTED_SPEAK_PLANNING_SESSION_PREFIX,
        PLANNING_SESSION_COMPACTION_FOCUS,
    )
}

pub fn generation_session_auto_compaction() -> SessionAutoCompaction {
    SessionAutoCompaction::new(
        SessionCompactionConfig::default(),
        SessionCompactionProtectedPrefix::LeadingSystemAndIdentitySeed,
        COMPACTED_SPEAK_GENERATION_SESSION_PREFIX,
        GENERATION_SESSION_COMPACTION_FOCUS,
    )
}

tokio::task_local! {
    /// JSON Schema for `PrepareSpeechArgs.target` derived from the live
    /// `SceneReader`. `.scope`d around each planning turn so the LLM sees the
    /// current host-constrained target enum.
    static SPEECH_TARGET_SCHEMA: Schema;
}

fn fallback_speech_target_schema() -> Schema {
    Schema::try_from(serde_json::json!({ "type": "string" }))
        .expect("fallback speech target schema must be a JSON object")
}

/// Wire-format string with a JSON Schema dynamically constrained to the
/// current scene's targets. Stored as `String` so existing serialization,
/// downstream `Utterance.target` are unchanged.
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
            .unwrap_or_else(|_| fallback_speech_target_schema())
    }
}

#[lutum::tool_input(name = "prepare_speech", output = PrepareSpeechOutput)]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
/// Call when allocation should proceed to an outward utterance. The input
/// carries cognition-log information transformed for speech, not hidden
/// analysis.
struct PrepareSpeechArgs {
    /// The participant who should hear the utterance.
    target: SpeechTarget,
    /// Speech-facing information to render for `target`, with perspective and
    /// addressee adjusted from the cognition log.
    speech_content: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
struct PrepareSpeechOutput {
    accepted: bool,
}

#[lutum::tool_input(name = "decline_speech_now", output = DeclineSpeechNowOutput)]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
/// Call only when a concrete blocker makes outward speech inappropriate or
/// impossible despite allocation.
struct DeclineSpeechNowArgs {
    blocking_reason: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
struct DeclineSpeechNowOutput {
    accepted: bool,
}

#[lutum::tool_input(name = "continue_speech", output = ContinueSpeechOutput)]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
/// Call when new cognition does not require interrupting the current speech.
struct ContinueSpeechArgs {
    rationale: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
struct ContinueSpeechOutput {
    accepted: bool,
}

#[lutum::tool_input(name = "interrupt_speech", output = InterruptSpeechOutput)]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
/// Call when current speech should be interrupted and replaced immediately.
struct InterruptSpeechArgs {
    target: SpeechTarget,
    speech_content: String,
    rationale: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
struct InterruptSpeechOutput {
    accepted: bool,
}

#[lutum::tool_input(name = "stop_speech", output = StopSpeechOutput)]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
/// Call when current speech should stop and no replacement utterance is warranted.
struct StopSpeechArgs {
    rationale: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
struct StopSpeechOutput {
    accepted: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema, lutum::Toolset)]
enum SpeakTools {
    PrepareSpeech(PrepareSpeechArgs),
    DeclineSpeechNow(DeclineSpeechNowArgs),
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema, lutum::Toolset)]
enum SpeechInterruptTools {
    Continue(ContinueSpeechArgs),
    Interrupt(InterruptSpeechArgs),
    Stop(StopSpeechArgs),
}

#[derive(Clone, Debug)]
struct PlannedSpeech {
    args: PrepareSpeechArgs,
    target: String,
}

#[derive(Debug)]
pub struct SpeakBatch {
    pub(crate) updates: Vec<CognitionLogUpdated>,
}

struct GenerationDraft {
    generation_id: u64,
    sequence: u32,
    accumulated: String,
    target: String,
}

impl GenerationDraft {
    fn new(generation_id: u64, target: impl Into<String>) -> GenerationDraft {
        GenerationDraft {
            generation_id,
            sequence: 0,
            accumulated: String::new(),
            target: target.into(),
        }
    }

    fn push_delta(&mut self, delta: &str) -> u32 {
        let sequence = self.sequence;
        self.accumulated.push_str(delta);
        self.sequence = self.sequence.wrapping_add(1);
        sequence
    }
}

fn render_completed_utterance_memo(draft: &GenerationDraft, text: &str) -> String {
    format!(
        "Completed utterance to {}:\n{}",
        draft.target.trim(),
        text.trim(),
    )
}

fn format_generation_input(
    cognition_context: &str,
    args: &PrepareSpeechArgs,
    draft: &GenerationDraft,
) -> String {
    format!(
        "{}\n\nSpeak to: {}\n\nSubstance to express:\n{}",
        cognition_context.trim(),
        draft.target.trim(),
        args.speech_content.trim()
    )
}

fn is_target_allowed_by_schema(schema: &Schema, target: &str) -> bool {
    if target.trim().is_empty() {
        return false;
    }
    let Ok(value) = serde_json::to_value(schema) else {
        return false;
    };
    let Some(values) = value.get("enum").and_then(serde_json::Value::as_array) else {
        return false;
    };
    values.iter().any(|value| value.as_str() == Some(target))
}

fn push_generation_context(
    session: &mut Session,
    cognition_context: &str,
    args: &PrepareSpeechArgs,
    draft: &GenerationDraft,
) {
    session.push_user(format_generation_input(cognition_context, args, draft));
    if !draft.accumulated.is_empty() {
        session.push_ephemeral_system(PARTIAL_CONTINUATION_PROMPT);
        session.push_ephemeral_assistant_text(draft.accumulated.clone());
    }
}

fn cognition_context_fallback(now: DateTime<Utc>) -> String {
    format!(
        "Current cognition log at {}:\n- none",
        now.to_rfc3339_opts(SecondsFormat::Secs, true)
    )
}

fn append_idle_context(mut cognition_context: String, idle_for_secs: Option<u64>) -> String {
    if let Some(seconds) = idle_for_secs {
        cognition_context.push_str(&format!("\n- I have been idle for {seconds} seconds."));
    }
    cognition_context
}

fn cognition_entry_records(logs: &[CognitionLogRecord]) -> Vec<CognitionLogEntryRecord> {
    let mut records = Vec::new();
    for log in logs
        .iter()
        .filter(|record| record.source.module != builtin::memory_recombination())
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

fn push_planning_context(session: &mut Session, cognition_context: &str) {
    session.push_user(cognition_context.trim().to_owned());
    session.push_ephemeral_developer(PLANNING_TURN_DEVELOPER_INSTRUCTION);
    session.push_ephemeral_user(PLANNING_TURN_FINAL_REMINDER);
}

fn push_interrupt_planning_context(
    session: &mut Session,
    cognition_context_at_start: &str,
    current_args: &PrepareSpeechArgs,
    draft: &GenerationDraft,
    new_cognition_context: &str,
) {
    let mut out = format!(
        "Cognition log at the start of the current speech attempt:\n{}",
        cognition_context_at_start.trim()
    );
    out.push_str("\n\nCurrent speech attempt:");
    out.push_str(&format!("\n- Target: {}", draft.target.trim()));
    out.push_str(&format!(
        "\n- Planned substance: {}",
        current_args.speech_content.trim()
    ));
    out.push_str(&format!(
        "\n- Partial utterance already emitted: {}",
        if draft.accumulated.trim().is_empty() {
            "(none)"
        } else {
            draft.accumulated.trim()
        }
    ));
    out.push_str("\n\nNew cognition entries since speech started:");
    out.push('\n');
    out.push_str(new_cognition_context.trim());
    session.push_user(out);
    session.push_ephemeral_developer(INTERRUPT_TURN_DEVELOPER_INSTRUCTION);
    session.push_ephemeral_user(nuillu_module::REQUIRED_FUNCTION_CALL_REMINDER);
}

#[derive(Debug)]
enum GenerationStreamOutcome {
    Completed,
    Retry,
    Interrupted(GenerationInterruption),
}

#[derive(Debug)]
enum GenerationInterruption {
    Planned(PlannedSpeech),
    NeedsFreshPlan,
    Stop,
}

#[derive(Debug)]
enum InterruptDecision {
    Continue,
    Interrupt(PlannedSpeech),
    Stop,
}

struct InterruptDecisionResult {
    decision: InterruptDecision,
    input: ModelInput,
    usage: Usage,
}

struct ActiveGenerationInterruptContext<'a> {
    cognition_context_at_start: &'a str,
    args: &'a PrepareSpeechArgs,
    draft: &'a GenerationDraft,
    stream_started_at: DateTime<Utc>,
}

type InterruptDecisionFuture =
    Pin<Box<dyn Future<Output = Result<InterruptDecisionResult>> + 'static>>;

async fn poll_pending_interrupt_decision(
    pending: &mut Option<InterruptDecisionFuture>,
) -> Result<InterruptDecisionResult> {
    pending
        .as_mut()
        .expect("pending interrupt decision is only polled when present")
        .as_mut()
        .await
}

pub struct SpeakModule {
    cognition_updates: CognitionLogUpdatedInbox,
    cognition_log: CognitionLogReader,
    memo: Memo,
    utterance: UtteranceWriter,
    llm: LlmAccess,
    scene: SceneReader,
    clock: Rc<dyn Clock>,
    planning_session: Session,
    generation_session: Session,
    plan_prompt: std::sync::OnceLock<String>,
    generation_prompt: std::sync::OnceLock<String>,
}

pub struct SpeakModuleParts {
    pub cognition_updates: CognitionLogUpdatedInbox,
    pub cognition_log: CognitionLogReader,
    pub memo: Memo,
    pub utterance: UtteranceWriter,
    pub llm: LlmAccess,
    pub scene: SceneReader,
    pub clock: Rc<dyn Clock>,
    pub planning_session: Session,
    pub generation_session: Session,
}

impl SpeakModule {
    pub fn new(parts: SpeakModuleParts) -> Self {
        let SpeakModuleParts {
            cognition_updates,
            cognition_log,
            memo,
            utterance,
            llm,
            scene,
            clock,
            planning_session,
            generation_session,
        } = parts;

        Self {
            cognition_updates,
            cognition_log,
            memo,
            utterance,
            llm,
            scene,
            clock,
            planning_session,
            generation_session,
            plan_prompt: std::sync::OnceLock::new(),
            generation_prompt: std::sync::OnceLock::new(),
        }
    }
}

impl SpeakModule {
    fn plan_prompt(&self, cx: &nuillu_module::ActivateCx<'_>) -> &str {
        self.plan_prompt.get_or_init(|| {
            nuillu_module::format_identity_system_prompt(
                SPEECH_PLANNING_PROMPT,
                cx.identity_memories(),
                cx.core_policies(),
                cx.now(),
            )
        })
    }

    fn generation_prompt(&self, cx: &nuillu_module::ActivateCx<'_>) -> &str {
        self.generation_prompt.get_or_init(|| {
            nuillu_module::format_identity_system_prompt(
                GENERATION_PROMPT,
                cx.identity_memories(),
                cx.core_policies(),
                cx.now(),
            )
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

    fn ensure_generation_session_seeded(&mut self, cx: &nuillu_module::ActivateCx<'_>) {
        let system_prompt = self.generation_prompt(cx).to_owned();
        ensure_persistent_session_seeded(
            &mut self.generation_session,
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
        let mut cognition_context = self.speech_cognition_context(self.clock.now()).await;
        let Some(mut plan) = self.plan_speech(cx, &cognition_context).await? else {
            return Ok(());
        };
        let mut draft = GenerationDraft::new(self.utterance.next_generation_id(), &plan.target);

        loop {
            self.record_streaming_progress(&draft).await;
            match self
                .stream_generation(cx, cognition_context.clone(), &plan.args, &mut draft)
                .await?
            {
                GenerationStreamOutcome::Completed => return Ok(()),
                GenerationStreamOutcome::Retry => {
                    cognition_context = self.speech_cognition_context(self.clock.now()).await;
                }
                GenerationStreamOutcome::Interrupted(interruption) => match interruption {
                    GenerationInterruption::Planned(new_plan) => {
                        cognition_context = self.speech_cognition_context(self.clock.now()).await;
                        plan = new_plan;
                        draft =
                            GenerationDraft::new(self.utterance.next_generation_id(), &plan.target);
                    }
                    GenerationInterruption::NeedsFreshPlan => {
                        cognition_context = self.speech_cognition_context(self.clock.now()).await;
                        let Some(new_plan) = self.plan_speech(cx, &cognition_context).await? else {
                            return Ok(());
                        };
                        plan = new_plan;
                        draft =
                            GenerationDraft::new(self.utterance.next_generation_id(), &plan.target);
                    }
                    GenerationInterruption::Stop => return Ok(()),
                },
            }
        }
    }

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
    ) -> Result<Option<PlannedSpeech>> {
        self.ensure_planning_session_seeded(cx);
        push_planning_context(&mut self.planning_session, cognition_context);

        let lutum = self.llm.lutum().await;
        let target_schema = self.scene.target_schema();
        let validation_schema = target_schema.clone();
        let outcome = SPEECH_TARGET_SCHEMA
            .scope(target_schema, async {
                self.planning_session
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
            TextStepOutcomeWithTools::Finished(result) => {
                cx.compact_and_save(&mut self.planning_session, result.usage)
                    .await?;
                let detail = "model finished with assistant output but no tool call \
                    (require_any_tool should have prevented this outcome)";
                cx.warn(format!("speak planning failed: {detail}"));
                anyhow::bail!("speak planning finished without required tool call: {detail}");
            }
            TextStepOutcomeWithTools::FinishedNoOutput(result) => {
                cx.compact_and_save(&mut self.planning_session, result.usage)
                    .await?;
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
                let mut selected = None;
                let mut declined = false;
                let mut results = Vec::new();
                for call in round.tool_calls.iter().cloned() {
                    match call {
                        SpeakToolsCall::PrepareSpeech(call) => {
                            let target = call.input.target.as_str().trim().to_owned();
                            let accepted = selected.is_none()
                                && !declined
                                && is_target_allowed_by_schema(&validation_schema, &target);
                            if accepted {
                                selected = Some(PlannedSpeech {
                                    args: call.input.clone(),
                                    target,
                                });
                            }
                            results.push(
                                call.complete(PrepareSpeechOutput { accepted })
                                    .context("complete prepare_speech tool call")?,
                            );
                        }
                        SpeakToolsCall::DeclineSpeechNow(call) => {
                            let accepted = selected.is_none() && !declined;
                            if accepted {
                                declined = true;
                            }
                            results.push(
                                call.complete(DeclineSpeechNowOutput { accepted })
                                    .context("complete decline_speech_now tool call")?,
                            );
                        }
                    }
                }
                round
                    .commit(&mut self.planning_session, results)
                    .context("commit speak planning tool round")?;
                cx.compact_and_save(&mut self.planning_session, usage)
                    .await?;
                Ok(selected)
            }
        }
    }

    async fn stream_generation(
        &mut self,
        cx: &nuillu_module::ActivateCx<'_>,
        cognition_context: String,
        args: &PrepareSpeechArgs,
        draft: &mut GenerationDraft,
    ) -> Result<GenerationStreamOutcome> {
        let stream_started_at = self.clock.now();
        let cognition_context_at_start = cognition_context.clone();

        self.ensure_generation_session_seeded(cx);
        let mut turn_session = Session::from_input(self.generation_session.input().clone());
        push_generation_context(&mut turn_session, &cognition_context, args, draft);

        let lutum = self.llm.lutum().await;
        let mut stream = turn_session
            .text_turn()
            .max_output_tokens(SPEECH_GENERATION_MAX_OUTPUT_TOKENS)
            .stream(&lutum)
            .await
            .context("speak generation stream failed")?;
        let mut pending_decision: Option<InterruptDecisionFuture> = None;
        let mut buffered_entries = Vec::new();

        loop {
            tokio::select! {
                biased;

                decision = poll_pending_interrupt_decision(&mut pending_decision), if pending_decision.is_some() => {
                    pending_decision = None;
                    let decision = decision?;
                    self.merge_planning_session(decision.input);
                    cx.compact_and_save(&mut self.planning_session, decision.usage).await?;
                    match decision.decision {
                        InterruptDecision::Continue => {
                            let next_entries = std::mem::take(&mut buffered_entries);
                            if !next_entries.is_empty() {
                                pending_decision = Some(self.start_interrupt_decision(
                                    cx,
                                    &cognition_context_at_start,
                                    args,
                                    draft,
                                    next_entries,
                                ));
                            }
                        }
                        InterruptDecision::Interrupt(plan) => {
                            self.record_aborted_progress(draft).await;
                            if buffered_entries.is_empty() {
                                return Ok(GenerationStreamOutcome::Interrupted(
                                    GenerationInterruption::Planned(plan),
                                ));
                            }
                            return Ok(GenerationStreamOutcome::Interrupted(
                                GenerationInterruption::NeedsFreshPlan,
                            ));
                        }
                        InterruptDecision::Stop => {
                            self.record_aborted_progress(draft).await;
                            return Ok(GenerationStreamOutcome::Interrupted(
                                GenerationInterruption::Stop,
                            ));
                        }
                    }
                }
                event = stream.next() => {
                    match event {
                        Some(Ok(TextTurnEvent::TextDelta { delta })) => {
                            let sequence = draft.push_delta(&delta);
                            self.utterance
                                .emit_delta(
                                    draft.target.clone(),
                                    draft.generation_id,
                                    sequence,
                                    delta,
                                )
                                .await;
                            self.record_streaming_progress(draft).await;
                        }
                        Some(Ok(TextTurnEvent::WillRetry { .. })) => {
                            return Ok(GenerationStreamOutcome::Retry);
                        }
                        Some(Ok(TextTurnEvent::Completed { committed_turn, usage, .. })) => {
                            if let Some(outcome) = self
                                .resolve_interrupts_before_completion(
                                    cx,
                                    &mut pending_decision,
                                    &mut buffered_entries,
                                    ActiveGenerationInterruptContext {
                                        cognition_context_at_start: &cognition_context_at_start,
                                        args,
                                        draft,
                                        stream_started_at,
                                    },
                                )
                                .await?
                            {
                                return Ok(outcome);
                            }
                            turn_session.input_mut().push(ModelInputItem::turn(committed_turn));
                            *self.generation_session.input_mut() = turn_session.into_input();
                            cx.compact_and_save(&mut self.generation_session, usage).await?;
                            let text = draft.accumulated.trim().to_owned();
                            self.memo.write(render_completed_utterance_memo(draft, &text)).await;
                            self.utterance
                                .record_progress(UtteranceProgress::completed(
                                    draft.generation_id,
                                    draft.sequence,
                                    draft.target.clone(),
                                    text.clone(),
                                ))
                                .await;
                            if !text.is_empty() {
                                self.utterance.emit(draft.target.clone(), text).await;
                            }
                            return Ok(GenerationStreamOutcome::Completed);
                        }
                        None => {
                            if let Some(outcome) = self
                                .resolve_interrupts_before_completion(
                                    cx,
                                    &mut pending_decision,
                                    &mut buffered_entries,
                                    ActiveGenerationInterruptContext {
                                        cognition_context_at_start: &cognition_context_at_start,
                                        args,
                                        draft,
                                        stream_started_at,
                                    },
                                )
                                .await?
                            {
                                return Ok(outcome);
                            }
                            let text = draft.accumulated.trim().to_owned();
                            self.memo.write(render_completed_utterance_memo(draft, &text)).await;
                            self.utterance
                                .record_progress(UtteranceProgress::completed(
                                    draft.generation_id,
                                    draft.sequence,
                                    draft.target.clone(),
                                    text.clone(),
                                ))
                                .await;
                            if !text.is_empty() {
                                self.utterance.emit(draft.target.clone(), text).await;
                            }
                            return Ok(GenerationStreamOutcome::Completed);
                        }
                        Some(Ok(_)) => {}
                        Some(Err(error)) => return Err(error).context("speak generation stream event failed"),
                    }
                }
                update = self.cognition_updates.next_item() => {
                    let _ = update.context("speak interrupt watch lost cognition update")?;
                    let _ = self.cognition_updates.take_ready_items()
                        .context("speak interrupt watch failed to drain cognition updates")?;

                    let new_entries = self
                        .new_cognition_entries_since(stream_started_at)
                        .await;
                    if new_entries.is_empty() {
                        continue;
                    }

                    if pending_decision.is_some() {
                        buffered_entries = new_entries;
                    } else {
                        pending_decision = Some(self.start_interrupt_decision(
                            cx,
                            &cognition_context_at_start,
                            args,
                            draft,
                            new_entries,
                        ));
                    }
                }
            }
        }
    }

    async fn resolve_interrupts_before_completion(
        &mut self,
        cx: &nuillu_module::ActivateCx<'_>,
        pending_decision: &mut Option<InterruptDecisionFuture>,
        buffered_entries: &mut Vec<CognitionLogEntryRecord>,
        active: ActiveGenerationInterruptContext<'_>,
    ) -> Result<Option<GenerationStreamOutcome>> {
        while pending_decision.is_some() {
            tokio::select! {
                biased;

                decision = poll_pending_interrupt_decision(pending_decision) => {
                    *pending_decision = None;
                    let decision = decision?;
                    self.merge_planning_session(decision.input);
                    cx.compact_and_save(&mut self.planning_session, decision.usage).await?;
                    match decision.decision {
                        InterruptDecision::Continue => {
                            let next_entries = std::mem::take(buffered_entries);
                            if !next_entries.is_empty() {
                                *pending_decision = Some(self.start_interrupt_decision(
                                    cx,
                                    active.cognition_context_at_start,
                                    active.args,
                                    active.draft,
                                    next_entries,
                                ));
                            }
                        }
                        InterruptDecision::Interrupt(plan) => {
                            self.record_aborted_progress(active.draft).await;
                            if buffered_entries.is_empty() {
                                return Ok(Some(GenerationStreamOutcome::Interrupted(
                                    GenerationInterruption::Planned(plan),
                                )));
                            }
                            return Ok(Some(GenerationStreamOutcome::Interrupted(
                                GenerationInterruption::NeedsFreshPlan,
                            )));
                        }
                        InterruptDecision::Stop => {
                            self.record_aborted_progress(active.draft).await;
                            return Ok(Some(GenerationStreamOutcome::Interrupted(
                                GenerationInterruption::Stop,
                            )));
                        }
                    }
                }
                update = self.cognition_updates.next_item() => {
                    let _ = update.context("speak interrupt completion watch lost cognition update")?;
                    let _ = self.cognition_updates.take_ready_items()
                        .context("speak interrupt completion watch failed to drain cognition updates")?;
                    let new_entries = self
                        .new_cognition_entries_since(active.stream_started_at)
                        .await;
                    if !new_entries.is_empty() {
                        *buffered_entries = new_entries;
                    }
                }
            }
        }

        Ok(None)
    }

    async fn new_cognition_entries_since(
        &self,
        threshold: DateTime<Utc>,
    ) -> Vec<CognitionLogEntryRecord> {
        let snapshot = self.cognition_log.snapshot().await;
        cognition_entry_records(snapshot.logs())
            .into_iter()
            .filter(|record| record.entry.at > threshold)
            .collect()
    }

    fn start_interrupt_decision(
        &mut self,
        cx: &nuillu_module::ActivateCx<'_>,
        cognition_context_at_start: &str,
        current_args: &PrepareSpeechArgs,
        draft: &GenerationDraft,
        new_entries: Vec<CognitionLogEntryRecord>,
    ) -> InterruptDecisionFuture {
        self.ensure_planning_session_seeded(cx);
        let llm = self.llm.clone();
        let target_schema = self.scene.target_schema();
        let validation_schema = target_schema.clone();
        let now = self.clock.now();
        let new_cognition_context =
            format_bounded_cognition_log_batch(&new_entries, now, COGNITION_CONTEXT_WINDOW)
                .unwrap_or_else(|| cognition_context_fallback(now));
        let mut session = Session::from_input(self.planning_session.input().clone());
        push_interrupt_planning_context(
            &mut session,
            cognition_context_at_start,
            current_args,
            draft,
            &new_cognition_context,
        );
        Box::pin(Self::decide_interrupt_owned(
            llm,
            target_schema,
            validation_schema,
            session,
        ))
    }

    async fn decide_interrupt_owned(
        llm: LlmAccess,
        target_schema: Schema,
        validation_schema: Schema,
        mut session: Session,
    ) -> Result<InterruptDecisionResult> {
        let lutum = llm.lutum().await;
        let outcome = SPEECH_TARGET_SCHEMA
            .scope(target_schema, async {
                session
                    .text_turn()
                    .tools::<SpeechInterruptTools>()
                    .available_tools([
                        SpeechInterruptToolsSelector::Continue,
                        SpeechInterruptToolsSelector::Interrupt,
                        SpeechInterruptToolsSelector::Stop,
                    ])
                    .require_any_tool()
                    .max_output_tokens(SPEECH_PLANNING_TURN_MAX_OUTPUT_TOKENS)
                    .collect_controlled_with(
                        &lutum,
                        nuillu_module::AbortOnAvailableToolNameInText::new(),
                    )
                    .await
                    .context("speak interrupt-planning turn failed")
            })
            .await?;

        match outcome {
            TextStepOutcomeWithTools::Finished(result) => Ok(InterruptDecisionResult {
                decision: InterruptDecision::Continue,
                input: session.into_input(),
                usage: result.usage,
            }),
            TextStepOutcomeWithTools::FinishedNoOutput(result) => Ok(InterruptDecisionResult {
                decision: InterruptDecision::Continue,
                input: session.into_input(),
                usage: result.usage,
            }),
            TextStepOutcomeWithTools::NeedsTools(round) => {
                let usage = round.usage;
                nuillu_module::emit_trace_tool_calls(&round.tool_calls);
                let mut decision = None;
                let mut results: Vec<ToolResult> = Vec::new();

                for call in round.tool_calls.iter().cloned() {
                    match call {
                        SpeechInterruptToolsCall::Continue(call) => {
                            let accepted = decision.is_none();
                            if accepted {
                                decision = Some(InterruptDecision::Continue);
                            }
                            results.push(
                                call.complete(ContinueSpeechOutput { accepted })
                                    .context("complete continue_speech tool call")?,
                            );
                        }
                        SpeechInterruptToolsCall::Interrupt(call) => {
                            let target = call.input.target.as_str().trim().to_owned();
                            let accepted = decision.is_none()
                                && is_target_allowed_by_schema(&validation_schema, &target);
                            if accepted {
                                decision = Some(InterruptDecision::Interrupt(PlannedSpeech {
                                    args: PrepareSpeechArgs {
                                        target: call.input.target.clone(),
                                        speech_content: call.input.speech_content.clone(),
                                    },
                                    target,
                                }));
                            } else if decision.is_none() {
                                decision = Some(InterruptDecision::Continue);
                            }
                            results.push(
                                call.complete(InterruptSpeechOutput { accepted })
                                    .context("complete interrupt_speech tool call")?,
                            );
                        }
                        SpeechInterruptToolsCall::Stop(call) => {
                            let accepted = decision.is_none();
                            if accepted {
                                decision = Some(InterruptDecision::Stop);
                            }
                            results.push(
                                call.complete(StopSpeechOutput { accepted })
                                    .context("complete stop_speech tool call")?,
                            );
                        }
                    }
                }

                round
                    .commit(&mut session, results)
                    .context("commit speak interrupt-planning tool round")?;
                Ok(InterruptDecisionResult {
                    decision: decision.unwrap_or(InterruptDecision::Continue),
                    input: session.into_input(),
                    usage,
                })
            }
        }
    }

    fn merge_planning_session(&mut self, input: ModelInput) {
        *self.planning_session.input_mut() = input;
    }

    async fn record_streaming_progress(&self, draft: &GenerationDraft) {
        self.utterance
            .record_progress(UtteranceProgress::streaming(
                draft.generation_id,
                draft.sequence,
                draft.target.clone(),
                draft.accumulated.clone(),
            ))
            .await;
    }

    async fn record_aborted_progress(&self, draft: &GenerationDraft) {
        self.utterance
            .record_progress(UtteranceProgress::aborted(
                draft.generation_id,
                draft.sequence,
                draft.target.clone(),
                draft.accumulated.clone(),
            ))
            .await;
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

    fn allocation_hint() -> Option<&'static str> {
        Some(
            "Raise speak when current cognition, or fresh evidence that should be admitted now, is grounded enough for an outward answer, address, warning, or expression of intent. For direct questions or advice requests to the agent, keep speak available alongside any registered evidence retrieval or admission path; those paths support an answer, not replace outward speech. Keep it low when evidence is unsettled, no addressee or focus is clear, or silence is better.",
        )
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

        Ok(SpeakBatch { updates })
    }
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::collections::VecDeque;
    use std::future::poll_fn;
    use std::pin::Pin;
    use std::rc::Rc;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Arc, Mutex};
    use std::task::{Context, Poll, Waker};
    use std::time::Duration;

    use futures::{Stream, stream};
    use lutum::{
        AdapterStructuredTurn, AdapterTextTurn, AgentError, AssistantInputItem, AssistantTurnItem,
        AssistantTurnView, ErasedStructuredTurnEventStream, ErasedTextTurnEvent,
        ErasedTextTurnEventStream, FinishReason, InputMessageRole, MaxOutputTokens, MessageContent,
        MockLlmAdapter, MockTextScenario, ModelInput, ModelInputItem, RawJson, RawTextTurnEvent,
        SharedPoolBudgetManager, SharedPoolBudgetOptions, ToolCallId, ToolName, TurnAdapter, Usage,
    };
    use nuillu_blackboard::{
        ActivationRatio, Blackboard, BlackboardCommand, CognitionLogEntry, ModuleConfig,
        ResourceAllocation,
    };
    use nuillu_module::ports::{Clock, NoopCognitionLogRepository, PortError, SystemClock};
    use nuillu_module::{
        CapabilityProviderPorts, CapabilityProviders, CognitionLogUpdated, LutumTiers,
        ModuleRegistry, Participant,
    };
    use nuillu_types::{ModuleInstanceId, ReplicaIndex, builtin};

    use super::*;
    use crate::test_support::*;
    use crate::utterance::{Utterance, UtteranceDelta, UtteranceSink};

    #[derive(Default)]
    struct PollGate {
        released: AtomicBool,
        waker: Mutex<Option<Waker>>,
    }

    impl PollGate {
        fn release(&self) {
            self.released.store(true, Ordering::SeqCst);
            if let Some(waker) = self.waker.lock().unwrap().take() {
                waker.wake();
            }
        }

        fn poll(&self, cx: &mut Context<'_>) -> Poll<()> {
            if self.released.load(Ordering::SeqCst) {
                Poll::Ready(())
            } else {
                *self.waker.lock().unwrap() = Some(cx.waker().clone());
                Poll::Pending
            }
        }

        async fn wait(self: Arc<Self>) {
            poll_fn(|cx| self.poll(cx)).await;
        }
    }

    #[derive(Clone)]
    struct CapturingAdapter<T> {
        inner: Arc<T>,
        text_turns: Arc<Mutex<Vec<AdapterTextTurn>>>,
    }

    impl<T> CapturingAdapter<T> {
        fn new(inner: T) -> Self {
            Self {
                inner: Arc::new(inner),
                text_turns: Arc::new(Mutex::new(Vec::new())),
            }
        }

        fn text_turns(&self) -> Vec<AdapterTextTurn> {
            self.text_turns.lock().unwrap().clone()
        }
    }

    #[async_trait::async_trait]
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

    enum GatedTextState {
        Started,
        WaitingForDelta,
        Completed,
        Done,
    }

    struct GatedTextStream {
        delta_gate: Arc<PollGate>,
        state: GatedTextState,
        delta: String,
    }

    impl GatedTextStream {
        fn new(delta_gate: Arc<PollGate>, delta: impl Into<String>) -> Self {
            Self {
                delta_gate,
                state: GatedTextState::Started,
                delta: delta.into(),
            }
        }
    }

    impl Stream for GatedTextStream {
        type Item = Result<ErasedTextTurnEvent, AgentError>;

        fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
            let this = self.get_mut();
            match this.state {
                GatedTextState::Started => {
                    this.state = GatedTextState::WaitingForDelta;
                    Poll::Ready(Some(Ok(ErasedTextTurnEvent::Started {
                        request_id: Some("speak-text".into()),
                        model: "mock".into(),
                    })))
                }
                GatedTextState::WaitingForDelta => {
                    if this.delta_gate.poll(cx).is_pending() {
                        return Poll::Pending;
                    }
                    this.state = GatedTextState::Completed;
                    Poll::Ready(Some(Ok(ErasedTextTurnEvent::TextDelta {
                        delta: this.delta.clone(),
                    })))
                }
                GatedTextState::Completed => {
                    this.state = GatedTextState::Done;
                    Poll::Ready(Some(Ok(ErasedTextTurnEvent::Completed {
                        request_id: Some("speak-text".into()),
                        finish_reason: FinishReason::Stop,
                        usage: Usage::zero(),
                        committed_turn: Arc::new(AssistantTurnView::from_items(&[
                            AssistantTurnItem::Text(this.delta.clone()),
                        ])),
                    })))
                }
                GatedTextState::Done => Poll::Ready(None),
            }
        }
    }

    enum GatedTurnScript {
        Generation {
            delta_gate: Arc<PollGate>,
            started: Option<tokio::sync::oneshot::Sender<()>>,
            delta: String,
        },
        Tool {
            release_gate: Arc<PollGate>,
            started: Option<tokio::sync::oneshot::Sender<()>>,
            request_id: &'static str,
            call_id: &'static str,
            name: &'static str,
            arguments_json: String,
        },
    }

    struct GatedTextAdapter {
        scripts: Mutex<VecDeque<GatedTurnScript>>,
    }

    impl GatedTurnScript {
        fn generation(
            delta_gate: Arc<PollGate>,
            started: Option<tokio::sync::oneshot::Sender<()>>,
            delta: impl Into<String>,
        ) -> Self {
            Self::Generation {
                delta_gate,
                started,
                delta: delta.into(),
            }
        }

        fn tool(
            release_gate: Arc<PollGate>,
            started: Option<tokio::sync::oneshot::Sender<()>>,
            request_id: &'static str,
            call_id: &'static str,
            name: &'static str,
            arguments_json: String,
        ) -> Self {
            Self::Tool {
                release_gate,
                started,
                request_id,
                call_id,
                name,
                arguments_json,
            }
        }
    }

    impl GatedTextAdapter {
        fn new(scripts: impl IntoIterator<Item = GatedTurnScript>) -> Self {
            Self {
                scripts: Mutex::new(scripts.into_iter().collect()),
            }
        }
    }

    #[async_trait::async_trait]
    impl TurnAdapter for GatedTextAdapter {
        async fn text_turn(
            &self,
            _input: ModelInput,
            _turn: AdapterTextTurn,
        ) -> Result<ErasedTextTurnEventStream, AgentError> {
            let script = self
                .scripts
                .lock()
                .unwrap()
                .pop_front()
                .expect("missing gated text turn script");
            match script {
                GatedTurnScript::Generation {
                    delta_gate,
                    started,
                    delta,
                } => {
                    if let Some(started) = started {
                        let _ = started.send(());
                    }
                    Ok(Box::pin(GatedTextStream::new(delta_gate, delta)))
                }
                GatedTurnScript::Tool {
                    release_gate,
                    started,
                    request_id,
                    call_id,
                    name,
                    arguments_json,
                } => {
                    if let Some(started) = started {
                        let _ = started.send(());
                    }
                    release_gate.wait().await;
                    Ok(tool_call_stream(request_id, call_id, name, arguments_json))
                }
            }
        }

        async fn structured_turn(
            &self,
            _input: ModelInput,
            _turn: AdapterStructuredTurn,
        ) -> Result<ErasedStructuredTurnEventStream, AgentError> {
            panic!("gated text adapter does not serve structured turns")
        }
    }

    fn released_gate() -> Arc<PollGate> {
        let gate = Arc::new(PollGate::default());
        gate.release();
        gate
    }

    fn tool_call_stream(
        request_id: &'static str,
        call_id: &'static str,
        name: &'static str,
        arguments_json: String,
    ) -> ErasedTextTurnEventStream {
        let arguments = RawJson::parse(arguments_json.clone()).unwrap();
        Box::pin(stream::iter(vec![
            Ok(ErasedTextTurnEvent::Started {
                request_id: Some(request_id.into()),
                model: "mock".into(),
            }),
            Ok(ErasedTextTurnEvent::ToolCallChunk {
                id: call_id.into(),
                name: name.into(),
                arguments_json_delta: arguments_json,
            }),
            Ok(ErasedTextTurnEvent::ToolCallReady(
                lutum::ToolMetadata::new(
                    ToolCallId::new(call_id),
                    ToolName::new(name),
                    arguments.clone(),
                ),
            )),
            Ok(ErasedTextTurnEvent::Completed {
                request_id: Some(request_id.into()),
                finish_reason: FinishReason::ToolCall,
                usage: Usage::zero(),
                committed_turn: Arc::new(AssistantTurnView::from_items(&[
                    AssistantTurnItem::ToolCall {
                        id: ToolCallId::new(call_id),
                        name: ToolName::new(name),
                        arguments,
                    },
                ])),
            }),
        ]))
    }

    struct InterruptToolArgs {
        target: &'static str,
        speech_content: &'static str,
    }

    fn interrupt_tool_script(
        release_gate: Arc<PollGate>,
        started: Option<tokio::sync::oneshot::Sender<()>>,
        args: InterruptToolArgs,
    ) -> GatedTurnScript {
        GatedTurnScript::tool(
            release_gate,
            started,
            "interrupt-speech",
            "call-interrupt-speech",
            "interrupt_speech",
            serde_json::json!({
                "target": args.target,
                "speech_content": args.speech_content,
                "rationale": "test interruption"
            })
            .to_string(),
        )
    }

    fn continue_tool_script(
        release_gate: Arc<PollGate>,
        started: Option<tokio::sync::oneshot::Sender<()>>,
    ) -> GatedTurnScript {
        GatedTurnScript::tool(
            release_gate,
            started,
            "continue-speech",
            "call-continue-speech",
            "continue_speech",
            serde_json::json!({
                "rationale": "test continuation"
            })
            .to_string(),
        )
    }

    fn stop_tool_script(
        release_gate: Arc<PollGate>,
        started: Option<tokio::sync::oneshot::Sender<()>>,
    ) -> GatedTurnScript {
        GatedTurnScript::tool(
            release_gate,
            started,
            "stop-speech",
            "call-stop-speech",
            "stop_speech",
            serde_json::json!({
                "rationale": "test stop"
            })
            .to_string(),
        )
    }

    fn prepare_speech_tool_script(
        target: &'static str,
        speech_content: &'static str,
    ) -> GatedTurnScript {
        GatedTurnScript::tool(
            released_gate(),
            None,
            "prepare-speech",
            "call-prepare-speech",
            "prepare_speech",
            serde_json::json!({
                "target": target,
                "speech_content": speech_content
            })
            .to_string(),
        )
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
        let arguments_json = serde_json::json!({
            "blocking_reason": blocking_reason
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

    fn test_prepare_speech_args(target: &str) -> PrepareSpeechArgs {
        PrepareSpeechArgs {
            target: SpeechTarget::from(target),
            speech_content: "Tell Koro to stay close because Koro asks for help.".into(),
        }
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

    type CapturedDelta = (String, u64, u32, String);
    type CapturedDeltas = Rc<RefCell<Vec<CapturedDelta>>>;
    type CapturedDeltaSender = tokio::sync::oneshot::Sender<CapturedDelta>;

    struct CapturingDeltaSink {
        deltas: CapturedDeltas,
        first_delta: RefCell<Option<CapturedDeltaSender>>,
    }

    #[async_trait(?Send)]
    impl UtteranceSink for CapturingDeltaSink {
        async fn on_complete(&self, _utterance: Utterance) -> Result<(), PortError> {
            Ok(())
        }

        async fn on_delta(&self, delta: UtteranceDelta) -> Result<(), PortError> {
            let captured = (
                delta.target,
                delta.generation_id,
                delta.sequence,
                delta.delta,
            );
            self.deltas.borrow_mut().push(captured.clone());
            if let Some(first_delta) = self.first_delta.borrow_mut().take() {
                let _ = first_delta.send(captured);
            }
            Ok(())
        }
    }

    async fn speak_module_with_turn_adapter<T>(
        blackboard: Blackboard,
        adapter: Arc<T>,
        sink: Rc<dyn UtteranceSink>,
    ) -> (SpeakModule, CapabilityProviders)
    where
        T: TurnAdapter,
    {
        let caps = test_caps_with_turn_adapter(blackboard, adapter);
        let module_cell = Rc::new(RefCell::new(None));
        let module_sink = Rc::clone(&module_cell);
        let utterance_sink_for_closure = sink.clone();

        let _modules = ModuleRegistry::new()
            .register(test_policy(), move |caps| {
                let module_sink = Rc::clone(&module_sink);
                let utterance_sink_for_closure = utterance_sink_for_closure.clone();
                async move {
                    *module_sink.borrow_mut() = Some(SpeakModule::new(SpeakModuleParts {
                        cognition_updates: caps.cognition_log_updated_inbox(),
                        cognition_log: caps.cognition_log_reader(),
                        memo: caps.memo(),
                        utterance: UtteranceWriter::new(
                            caps.owner().clone(),
                            caps.blackboard(),
                            utterance_sink_for_closure.clone(),
                            caps.clock(),
                        ),
                        llm: caps.llm_access(),
                        scene: caps.scene_reader(),
                        clock: caps.clock(),
                        planning_session: caps
                            .session("planning")
                            .with_auto_compaction(planning_session_auto_compaction())
                            .await?,
                        generation_session: caps
                            .session("generation")
                            .with_auto_compaction(generation_session_auto_compaction())
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
        (module, caps)
    }

    async fn publish_cognition_update(
        blackboard: &Blackboard,
        caps: &CapabilityProviders,
        at: chrono::DateTime<chrono::Utc>,
        text: impl Into<String>,
    ) {
        let source = ModuleInstanceId::new(builtin::cognition_gate(), ReplicaIndex::ZERO);
        blackboard
            .apply(BlackboardCommand::AppendCognitionLog {
                source: source.clone(),
                entry: CognitionLogEntry {
                    at,
                    text: text.into(),
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
        let compaction_lutum = module.llm.lutum().await;
        nuillu_module::ActivateCx::new(
            &[],
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
        allocation.set(builtin::speak(), ModuleConfig::default());
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
        blackboard
            .read(|bb| {
                bb.recent_memo_logs()
                    .into_iter()
                    .filter(|record| record.owner.module == builtin::speak())
                    .count()
            })
            .await
    }

    async fn speak_progress_exists(blackboard: &Blackboard) -> bool {
        let speak_owner = ModuleInstanceId::new(builtin::speak(), ReplicaIndex::ZERO);
        blackboard
            .read(|bb| bb.utterance_progress_for_instance(&speak_owner).is_some())
            .await
    }

    fn session_turn_texts(session: &Session) -> Vec<String> {
        session
            .list_turns()
            .filter_map(|turn| {
                turn.item_at(0)
                    .and_then(|item| item.as_text())
                    .map(str::to_owned)
            })
            .collect()
    }

    fn session_input_text(session: &Session) -> String {
        let mut out = String::new();
        for item in session.input().items() {
            match item {
                ModelInputItem::Message { content, .. } => {
                    for content in content.as_slice() {
                        if let MessageContent::Text(text) = content {
                            out.push_str(text);
                            out.push('\n');
                        }
                    }
                }
                ModelInputItem::Assistant(AssistantInputItem::Text(text)) => {
                    out.push_str(text);
                    out.push('\n');
                }
                _ => {}
            }
        }
        out
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
        blackboard
            .apply(BlackboardCommand::AppendCognitionLog {
                source,
                entry: CognitionLogEntry {
                    at: now - chrono::Duration::minutes(4),
                    text: "Koro asks Nuillu for help.".into(),
                },
            })
            .await;

        let context = module.speech_cognition_context(now).await;

        assert!(context.contains("Current cognition log at "));
        assert!(context.contains("About 4 minutes ago: Koro asks Nuillu for help."));
    }

    #[test]
    fn fresh_generation_omits_assistant_prefill() {
        let draft = GenerationDraft::new(7, "Koro");
        let args = test_prepare_speech_args("Koro");
        let mut session = Session::new();

        push_generation_context(
            &mut session,
            "Current cognition log at 2026-05-11T06:23:00Z:\n- none",
            &args,
            &draft,
        );
        let items = session.input().items();

        assert_eq!(draft.generation_id, 7);
        assert_eq!(draft.sequence, 0);
        assert_eq!(items.len(), 1);
        let ModelInputItem::Message {
            role: InputMessageRole::User,
            content,
        } = &items[0]
        else {
            panic!("expected user generation context");
        };
        let [MessageContent::Text(text)] = content.as_slice() else {
            panic!("expected one text content item");
        };
        assert!(text.contains("Current cognition log at 2026-05-11T06:23:00Z:\n- none"));
        assert!(text.contains("Speak to: Koro"));
        assert!(text.contains("Substance to express:"));
        assert!(text.contains("Tell Koro to stay close because Koro asks for help."));
        assert!(!text.contains("prepare_speech"));
        assert!(!text.contains("tool call"));
        assert!(!text.contains("speech_content"));
        assert!(!text.contains("target:"));
        assert!(!text.contains("rationale"));
        assert!(!text.contains("allocation"));
        assert!(!text.contains("partial_utterance"));
    }

    #[test]
    fn unknown_evidence_plan_renders_uncertainty_and_visible_absence() {
        let draft = GenerationDraft::new(8, "Pibi");
        let args = PrepareSpeechArgs {
            target: SpeechTarget::from("Pibi"),
            speech_content: "Tell Pibi I do not know whether dinner is ready or where it is, because I see no food or person nearby.".into(),
        };

        let text = format_generation_input("- Pibi asks about dinner.", &args, &draft);

        assert!(text.contains("I do not know whether dinner is ready"));
        assert!(text.contains("I see no food or person nearby."));
    }

    #[test]
    fn speech_prompts_are_slim_and_avoid_peer_catalog() {
        let prompt = nuillu_module::format_identity_system_prompt(
            GENERATION_PROMPT,
            &[],
            &[],
            SystemClock.now(),
        );

        assert!(prompt.len() < 1100);
        assert!(!prompt.contains("prepare_speech"));
        assert!(!prompt.contains("tool call"));
        assert!(!prompt.contains("speech_content"));
        assert!(!prompt.contains("You are part of a cognitive system"));
        assert!(!prompt.contains("- cognition-gate:"));
        assert!(!prompt.contains("- query-memory:"));
        assert!(!SPEECH_PLANNING_PROMPT.contains("\"self\""));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn speak_selects_target_from_cognition_log_before_streaming() {
        let adapter = MockLlmAdapter::new()
            .with_text_scenario(prepare_speech_scenario("Koro", "Tell Koro to stay close."))
            .with_text_scenario(generation_text_scenario("Koro, stay close."));
        let mut allocation = ResourceAllocation::default();
        allocation.set(builtin::speak(), ModuleConfig::default());
        allocation.set_activation(builtin::speak(), ActivationRatio::ONE);
        let blackboard = Blackboard::with_allocation(allocation);
        let completed = Rc::new(RefCell::new(Vec::new()));
        let (done_tx, done_rx) = tokio::sync::oneshot::channel();
        let sink: Rc<dyn crate::utterance::UtteranceSink> = Rc::new(CapturingUtteranceSink {
            completed: Rc::clone(&completed),
            done: RefCell::new(Some(done_tx)),
        });
        let caps = test_caps_with_adapter(blackboard.clone(), adapter);
        caps.scene().set([Participant::new("Koro")]);
        let module_cell = Rc::new(RefCell::new(None));
        let module_sink = Rc::clone(&module_cell);
        let utterance_sink_for_closure = sink.clone();

        let _modules = ModuleRegistry::new()
            .register(test_policy(), move |caps| {
                let module_sink = Rc::clone(&module_sink);
                let utterance_sink_for_closure = utterance_sink_for_closure.clone();
                async move {
                    *module_sink.borrow_mut() = Some(SpeakModule::new(SpeakModuleParts {
                        cognition_updates: caps.cognition_log_updated_inbox(),
                        cognition_log: caps.cognition_log_reader(),
                        memo: caps.memo(),
                        utterance: UtteranceWriter::new(
                            caps.owner().clone(),
                            caps.blackboard(),
                            utterance_sink_for_closure.clone(),
                            caps.clock(),
                        ),
                        llm: caps.llm_access(),
                        scene: caps.scene_reader(),
                        clock: caps.clock(),
                        planning_session: caps
                            .session("planning")
                            .with_auto_compaction(planning_session_auto_compaction())
                            .await?,
                        generation_session: caps
                            .session("generation")
                            .with_auto_compaction(generation_session_auto_compaction())
                            .await?,
                    }));
                    Ok(SpeakStub)
                }
            })
            .unwrap()
            .build(&caps)
            .await
            .unwrap();
        let mut module = module_cell.borrow_mut().take().unwrap();
        let source = ModuleInstanceId::new(builtin::cognition_gate(), ReplicaIndex::ZERO);
        blackboard
            .apply(BlackboardCommand::AppendCognitionLog {
                source: source.clone(),
                entry: CognitionLogEntry {
                    at: SystemClock.now(),
                    text: "Koro asks Nuillu to help them stay safe.".into(),
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
        let identity_memories = Vec::new();
        let compaction_lutum = module.llm.lutum().await;
        let clock = SystemClock;
        let cx = nuillu_module::ActivateCx::new(
            &catalog,
            &[],
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
        let _ = done_rx.await;

        assert_eq!(
            completed.borrow().as_slice(),
            &[("Koro".to_string(), "Koro, stay close.".to_string())]
        );
        let speak_owner = ModuleInstanceId::new(builtin::speak(), ReplicaIndex::ZERO);
        let progress = blackboard
            .read(|bb| bb.utterance_progress_for_instance(&speak_owner).cloned())
            .await
            .unwrap();
        assert_eq!(progress.target, "Koro");
        assert_eq!(progress.partial_utterance, "Koro, stay close.");
        assert_eq!(module.planning_session.list_turns().count(), 1);
        assert_eq!(
            session_turn_texts(&module.generation_session),
            vec!["Koro, stay close.".to_string()]
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn speak_sets_max_output_tokens_for_planning_and_generation() {
        let adapter = MockLlmAdapter::new()
            .with_text_scenario(prepare_speech_scenario("Koro", "Tell Koro to stay close."))
            .with_text_scenario(generation_text_scenario("Koro, stay close."));
        let capture = CapturingAdapter::new(adapter);
        let observed = capture.clone();
        let mut allocation = ResourceAllocation::default();
        allocation.set(builtin::speak(), ModuleConfig::default());
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
        assert_eq!(turns.len(), 2);
        assert_eq!(
            turns[0].config.generation.max_output_tokens,
            Some(MaxOutputTokens::new(SPEECH_PLANNING_TURN_MAX_OUTPUT_TOKENS))
        );
        assert_eq!(
            turns[1].config.generation.max_output_tokens,
            Some(MaxOutputTokens::new(SPEECH_GENERATION_MAX_OUTPUT_TOKENS))
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
        assert_eq!(speak_memo_count(&blackboard).await, 0);
        assert!(!speak_progress_exists(&blackboard).await);
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
    async fn speak_stays_silent_for_empty_planned_target() {
        let adapter = MockLlmAdapter::new()
            .with_text_scenario(prepare_speech_scenario("", "Tell Koro to stay close."));

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
    async fn speak_stays_silent_for_planned_target_outside_scene_schema() {
        let adapter = MockLlmAdapter::new()
            .with_text_scenario(prepare_speech_scenario("Koro", "Tell Pibi to stay close."));

        let (blackboard, completed) = activate_once_with_adapter(
            adapter,
            [Participant::new("Pibi")],
            "Pibi asks whether Nui should say anything.",
        )
        .await;

        assert!(completed.borrow().is_empty());
        assert_eq!(speak_memo_count(&blackboard).await, 0);
        assert!(!speak_progress_exists(&blackboard).await);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn speak_stream_drains_delta_while_interrupt_decision_is_pending() {
        let text_delta_gate = Arc::new(PollGate::default());
        let decision_release_gate = Arc::new(PollGate::default());
        let (text_started_tx, text_started_rx) = tokio::sync::oneshot::channel();
        let (decision_started_tx, decision_started_rx) = tokio::sync::oneshot::channel();
        let adapter = Arc::new(GatedTextAdapter::new([
            GatedTurnScript::generation(
                text_delta_gate.clone(),
                Some(text_started_tx),
                "Koro, stay close.",
            ),
            continue_tool_script(decision_release_gate.clone(), Some(decision_started_tx)),
        ]));
        let mut allocation = ResourceAllocation::default();
        allocation.set(builtin::speak(), ModuleConfig::default());
        allocation.set_activation(builtin::speak(), ActivationRatio::ONE);
        let blackboard = Blackboard::with_allocation(allocation);
        let deltas = Rc::new(RefCell::new(Vec::new()));
        let (delta_tx, delta_rx) = tokio::sync::oneshot::channel();
        let sink: Rc<dyn UtteranceSink> = Rc::new(CapturingDeltaSink {
            deltas: Rc::clone(&deltas),
            first_delta: RefCell::new(Some(delta_tx)),
        });
        let (mut module, caps) =
            speak_module_with_turn_adapter(blackboard.clone(), adapter, sink).await;
        let clock = SystemClock;
        let now = clock.now();
        let cx = test_activate_cx(&module, now).await;
        let mut draft = GenerationDraft::new(0, "Koro");
        let args = test_prepare_speech_args("Koro");
        let stream = module.stream_generation(&cx, "- initial awareness".into(), &args, &mut draft);
        tokio::pin!(stream);

        tokio::select! {
            result = &mut stream => {
                let _ = result;
                panic!("stream ended before text generation started");
            }
            result = text_started_rx => {
                result.expect("text generation should start");
            }
        }

        publish_cognition_update(
            &blackboard,
            &caps,
            now + chrono::Duration::milliseconds(1),
            "Koro notices a new hazard while speech is in progress.",
        )
        .await;

        tokio::select! {
            result = &mut stream => {
                let _ = result;
                panic!("stream ended before interrupt decision started");
            }
            result = decision_started_rx => {
                result.expect("interrupt decision should start");
            }
        }

        text_delta_gate.release();
        let delta = tokio::select! {
            result = &mut stream => {
                let _ = result;
                deltas
                    .borrow()
                    .first()
                    .cloned()
                    .expect("stream ended before emitting the gated text delta")
            }
            result = tokio::time::timeout(Duration::from_millis(100), delta_rx) => {
                result
                    .expect("text delta should be emitted while interrupt decision is pending")
                    .expect("delta sink should send the first delta")
            }
        };

        assert_eq!(
            delta,
            ("Koro".to_string(), 0, 0, "Koro, stay close.".to_string())
        );
        assert_eq!(
            deltas.borrow().as_slice(),
            &[("Koro".to_string(), 0, 0, "Koro, stay close.".to_string())]
        );
        decision_release_gate.release();
    }

    #[tokio::test(flavor = "current_thread")]
    async fn speak_rechecks_buffered_cognition_after_continue_decision() {
        let text_delta_gate = Arc::new(PollGate::default());
        let first_decision_gate = Arc::new(PollGate::default());
        let second_decision_gate = Arc::new(PollGate::default());
        let (text_started_tx, text_started_rx) = tokio::sync::oneshot::channel();
        let (first_decision_started_tx, first_decision_started_rx) =
            tokio::sync::oneshot::channel();
        let (second_decision_started_tx, second_decision_started_rx) =
            tokio::sync::oneshot::channel();
        let adapter = Arc::new(GatedTextAdapter::new([
            GatedTurnScript::generation(
                text_delta_gate,
                Some(text_started_tx),
                "Koro, stay close.",
            ),
            continue_tool_script(first_decision_gate.clone(), Some(first_decision_started_tx)),
            interrupt_tool_script(
                second_decision_gate.clone(),
                Some(second_decision_started_tx),
                InterruptToolArgs {
                    target: "Koro",
                    speech_content: "Tell Koro to duck now.",
                },
            ),
        ]));
        let mut allocation = ResourceAllocation::default();
        allocation.set(builtin::speak(), ModuleConfig::default());
        allocation.set_activation(builtin::speak(), ActivationRatio::ONE);
        let blackboard = Blackboard::with_allocation(allocation);
        let deltas = Rc::new(RefCell::new(Vec::new()));
        let (delta_tx, _delta_rx) = tokio::sync::oneshot::channel();
        let sink: Rc<dyn UtteranceSink> = Rc::new(CapturingDeltaSink {
            deltas,
            first_delta: RefCell::new(Some(delta_tx)),
        });
        let (mut module, caps) =
            speak_module_with_turn_adapter(blackboard.clone(), adapter, sink).await;
        caps.scene().set([Participant::new("Koro")]);

        let clock = SystemClock;
        let now = clock.now();
        let cx = test_activate_cx(&module, now).await;
        let mut draft = GenerationDraft::new(0, "Koro");
        let args = test_prepare_speech_args("Koro");

        {
            let stream =
                module.stream_generation(&cx, "- initial awareness".into(), &args, &mut draft);
            tokio::pin!(stream);

            tokio::select! {
                result = &mut stream => {
                    let _ = result;
                    panic!("stream ended before text generation started");
                }
                result = text_started_rx => {
                    result.expect("text generation should start");
                }
            }

            publish_cognition_update(
                &blackboard,
                &caps,
                now + chrono::Duration::milliseconds(1),
                "Koro notices the first new hazard.",
            )
            .await;

            tokio::select! {
                result = &mut stream => {
                    let _ = result;
                    panic!("stream ended before first interrupt decision started");
                }
                result = first_decision_started_rx => {
                    result.expect("first interrupt decision should start");
                }
            }

            publish_cognition_update(
                &blackboard,
                &caps,
                now + chrono::Duration::milliseconds(2),
                "Koro notices a second, more urgent hazard.",
            )
            .await;
            first_decision_gate.release();

            tokio::select! {
                result = &mut stream => {
                    let _ = result;
                    panic!("stream ended before buffered cognition triggered another decision");
                }
                result = tokio::time::timeout(Duration::from_millis(100), second_decision_started_rx) => {
                    result
                        .expect("buffered cognition should start another decision")
                        .expect("second interrupt decision should start");
                }
            }

            second_decision_gate.release();
            let outcome = tokio::time::timeout(Duration::from_millis(100), &mut stream)
                .await
                .expect("second decision should finish the stream")
                .unwrap();
            assert!(
                matches!(
                    outcome,
                    GenerationStreamOutcome::Interrupted(GenerationInterruption::Planned(_))
                ),
                "unexpected outcome: {outcome:?}"
            );
        }

        assert_eq!(module.planning_session.list_turns().count(), 2);
        let planning_text = session_input_text(&module.planning_session);
        assert!(!planning_text.contains(INTERRUPT_TURN_DEVELOPER_INSTRUCTION));
        assert!(!planning_text.contains(nuillu_module::REQUIRED_FUNCTION_CALL_REMINDER));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn speak_interrupt_decision_wins_over_ready_text_delta() {
        let text_delta_gate = Arc::new(PollGate::default());
        let decision_release_gate = Arc::new(PollGate::default());
        let (text_started_tx, text_started_rx) = tokio::sync::oneshot::channel();
        let (decision_started_tx, decision_started_rx) = tokio::sync::oneshot::channel();
        let adapter = Arc::new(GatedTextAdapter::new([
            GatedTurnScript::generation(
                text_delta_gate.clone(),
                Some(text_started_tx),
                "Koro, stay close.",
            ),
            interrupt_tool_script(
                decision_release_gate.clone(),
                Some(decision_started_tx),
                InterruptToolArgs {
                    target: "Koro",
                    speech_content: "Tell Koro to stop immediately.",
                },
            ),
        ]));
        let mut allocation = ResourceAllocation::default();
        allocation.set(builtin::speak(), ModuleConfig::default());
        allocation.set_activation(builtin::speak(), ActivationRatio::ONE);
        let blackboard = Blackboard::with_allocation(allocation);
        let deltas = Rc::new(RefCell::new(Vec::new()));
        let (delta_tx, _delta_rx) = tokio::sync::oneshot::channel();
        let sink: Rc<dyn UtteranceSink> = Rc::new(CapturingDeltaSink {
            deltas: Rc::clone(&deltas),
            first_delta: RefCell::new(Some(delta_tx)),
        });
        let (mut module, caps) =
            speak_module_with_turn_adapter(blackboard.clone(), adapter, sink).await;
        caps.scene().set([Participant::new("Koro")]);

        let clock = SystemClock;
        let now = clock.now();
        let cx = test_activate_cx(&module, now).await;
        let mut draft = GenerationDraft::new(0, "Koro");
        let args = test_prepare_speech_args("Koro");
        let stream = module.stream_generation(&cx, "- initial awareness".into(), &args, &mut draft);
        tokio::pin!(stream);

        tokio::select! {
            result = &mut stream => {
                let _ = result;
                panic!("stream ended before text generation started");
            }
            result = text_started_rx => {
                result.expect("text generation should start");
            }
        }

        publish_cognition_update(
            &blackboard,
            &caps,
            now + chrono::Duration::milliseconds(1),
            "Koro notices speech should be interrupted immediately.",
        )
        .await;

        tokio::select! {
            result = &mut stream => {
                let _ = result;
                panic!("stream ended before interrupt decision started");
            }
            result = decision_started_rx => {
                result.expect("interrupt decision should start");
            }
        }

        text_delta_gate.release();
        decision_release_gate.release();
        let outcome = tokio::time::timeout(Duration::from_millis(100), &mut stream)
            .await
            .expect("ready interrupt decision should finish the stream")
            .unwrap();

        assert!(
            matches!(
                outcome,
                GenerationStreamOutcome::Interrupted(GenerationInterruption::Planned(_))
            ),
            "unexpected outcome: {outcome:?}"
        );
        assert!(deltas.borrow().is_empty());
    }

    #[tokio::test(flavor = "current_thread")]
    async fn speak_interruption_replans_and_generates_inside_one_activation() {
        let first_generation_gate = Arc::new(PollGate::default());
        let interrupt_gate = Arc::new(PollGate::default());
        let replacement_gate = released_gate();
        let (generation_started_tx, generation_started_rx) = tokio::sync::oneshot::channel();
        let (interrupt_started_tx, interrupt_started_rx) = tokio::sync::oneshot::channel();
        let adapter = Arc::new(GatedTextAdapter::new([
            prepare_speech_tool_script("Koro", "Tell Koro to stay close."),
            GatedTurnScript::generation(
                first_generation_gate,
                Some(generation_started_tx),
                "Koro, stay close.",
            ),
            interrupt_tool_script(
                interrupt_gate.clone(),
                Some(interrupt_started_tx),
                InterruptToolArgs {
                    target: "Pibi",
                    speech_content: "Tell Pibi to duck now.",
                },
            ),
            GatedTurnScript::generation(replacement_gate, None, "Pibi, duck now."),
        ]));
        let mut allocation = ResourceAllocation::default();
        allocation.set(builtin::speak(), ModuleConfig::default());
        allocation.set_activation(builtin::speak(), ActivationRatio::ONE);
        let blackboard = Blackboard::with_allocation(allocation);
        let completed = Rc::new(RefCell::new(Vec::new()));
        let sink: Rc<dyn crate::utterance::UtteranceSink> = Rc::new(CapturingUtteranceSink {
            completed: Rc::clone(&completed),
            done: RefCell::new(None),
        });
        let (mut module, caps) =
            speak_module_with_turn_adapter(blackboard.clone(), adapter, sink).await;
        caps.scene()
            .set([Participant::new("Koro"), Participant::new("Pibi")]);
        let clock = SystemClock;
        let now = clock.now();
        publish_cognition_update(
            &blackboard,
            &caps,
            now,
            "Koro asks Nuillu to help them stay safe.",
        )
        .await;
        let batch = module.next_batch().await.unwrap();
        let cx = test_activate_cx(&module, now).await;
        {
            let activation = SpeakModule::activate(&mut module, &cx, &batch);
            tokio::pin!(activation);

            tokio::select! {
                result = &mut activation => {
                    let _ = result;
                    panic!("activation ended before first generation started");
                }
                result = generation_started_rx => {
                    result.expect("first generation should start");
                }
            }

            publish_cognition_update(
                &blackboard,
                &caps,
                now + chrono::Duration::milliseconds(1),
                "Pibi is in immediate danger and needs a warning.",
            )
            .await;

            tokio::select! {
                result = &mut activation => {
                    let _ = result;
                    panic!("activation ended before interrupt planning started");
                }
                result = interrupt_started_rx => {
                    result.expect("interrupt planning should start");
                }
            }

            interrupt_gate.release();
            tokio::time::timeout(Duration::from_millis(100), &mut activation)
                .await
                .expect("activation should finish after replacement generation")
                .unwrap();
        }

        assert_eq!(
            completed.borrow().as_slice(),
            &[("Pibi".to_string(), "Pibi, duck now.".to_string())]
        );
        let generation_text = session_input_text(&module.generation_session);
        assert!(!generation_text.contains("Tell Koro to stay close."));
        assert!(generation_text.contains("Tell Pibi to duck now."));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn speak_stop_interruption_records_aborted_progress() {
        let text_delta_gate = Arc::new(PollGate::default());
        let decision_release_gate = Arc::new(PollGate::default());
        let (text_started_tx, text_started_rx) = tokio::sync::oneshot::channel();
        let (decision_started_tx, decision_started_rx) = tokio::sync::oneshot::channel();
        let adapter = Arc::new(GatedTextAdapter::new([
            GatedTurnScript::generation(
                text_delta_gate,
                Some(text_started_tx),
                "Koro, stay close.",
            ),
            stop_tool_script(decision_release_gate.clone(), Some(decision_started_tx)),
        ]));
        let mut allocation = ResourceAllocation::default();
        allocation.set(builtin::speak(), ModuleConfig::default());
        allocation.set_activation(builtin::speak(), ActivationRatio::ONE);
        let blackboard = Blackboard::with_allocation(allocation);
        let completed = Rc::new(RefCell::new(Vec::new()));
        let sink: Rc<dyn crate::utterance::UtteranceSink> = Rc::new(CapturingUtteranceSink {
            completed: Rc::clone(&completed),
            done: RefCell::new(None),
        });
        let (mut module, caps) =
            speak_module_with_turn_adapter(blackboard.clone(), adapter, sink).await;
        let clock = SystemClock;
        let now = clock.now();
        let cx = test_activate_cx(&module, now).await;
        let mut draft = GenerationDraft::new(0, "Koro");
        let args = test_prepare_speech_args("Koro");
        let stream = module.stream_generation(&cx, "- initial awareness".into(), &args, &mut draft);
        tokio::pin!(stream);

        tokio::select! {
            result = &mut stream => {
                let _ = result;
                panic!("stream ended before text generation started");
            }
            result = text_started_rx => {
                result.expect("text generation should start");
            }
        }

        publish_cognition_update(
            &blackboard,
            &caps,
            now + chrono::Duration::milliseconds(1),
            "Koro no longer needs the warning.",
        )
        .await;

        tokio::select! {
            result = &mut stream => {
                let _ = result;
                panic!("stream ended before stop decision started");
            }
            result = decision_started_rx => {
                result.expect("stop decision should start");
            }
        }

        decision_release_gate.release();
        let outcome = tokio::time::timeout(Duration::from_millis(100), &mut stream)
            .await
            .expect("stop decision should finish the stream")
            .unwrap();
        assert!(matches!(
            outcome,
            GenerationStreamOutcome::Interrupted(GenerationInterruption::Stop)
        ));
        assert!(completed.borrow().is_empty());

        let speak_owner = ModuleInstanceId::new(builtin::speak(), ReplicaIndex::ZERO);
        let progress = blackboard
            .read(|bb| bb.utterance_progress_for_instance(&speak_owner).cloned())
            .await
            .unwrap();
        assert_eq!(
            progress.state,
            nuillu_blackboard::UtteranceProgressState::Aborted
        );
    }

    #[test]
    fn resumed_generation_keeps_id_sequence_and_pushes_assistant_prefill() {
        let mut draft = GenerationDraft::new(11, "Koro");
        let args = test_prepare_speech_args("Koro");
        let mut session = Session::new();

        assert_eq!(draft.push_delta("hello "), 0);
        assert_eq!(draft.push_delta("world"), 1);
        push_generation_context(
            &mut session,
            "Current cognition log at 2026-05-11T06:23:00Z:\n- none",
            &args,
            &draft,
        );
        let items = session.input().items();

        assert_eq!(draft.generation_id, 11);
        assert_eq!(draft.sequence, 2);
        assert_eq!(draft.accumulated, "hello world");
        assert_eq!(items.len(), 3);
        assert!(matches!(
            &items[0],
            ModelInputItem::Message {
                role: InputMessageRole::User,
                ..
            }
        ));
        assert!(matches!(
            &items[1],
            ModelInputItem::Message {
                role: InputMessageRole::System,
                ..
            }
        ));
        let ModelInputItem::Assistant(AssistantInputItem::Text(text)) = &items[2] else {
            panic!("expected assistant prefill");
        };
        assert_eq!(text, "hello world");
    }

    #[test]
    fn speak_batch_keeps_drained_updates() {
        let batch = SpeakBatch {
            updates: vec![CognitionLogUpdated::AgenticDeadlockMarker],
        };

        assert_eq!(
            batch.updates,
            vec![CognitionLogUpdated::AgenticDeadlockMarker]
        );
    }
}
