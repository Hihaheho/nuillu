use std::borrow::Cow;
use std::collections::HashSet;
use std::rc::Rc;
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result};
use async_trait::async_trait;
use chrono::{DateTime, SecondsFormat, Utc};
use lutum::{
    AgentError, AssistantTurnItem, AssistantTurnView, CollectError, EventHandler, FinishReason,
    HandlerContext, HandlerDirective, HandlerResult, ModelInputItem, Session,
    TextStepOutcomeWithTools, TextTurnEvent, TextTurnReductionError, TextTurnState, Usage,
};
use nuillu_blackboard::CognitionLogEntryRecord;
#[cfg(test)]
use nuillu_blackboard::CognitionLogRecord;
#[cfg(test)]
use nuillu_module::format_bounded_cognition_log_batch;
use nuillu_module::{
    CognitionLogReader, CognitionLogUpdated, CognitionLogUpdatedInbox, LlmAccess, LlmContextWindow,
    Memo, Module, SceneReader, SelfWake, SessionAutoCompaction, SessionCompactionConfig,
    SessionCompactionProtectedPrefix, UtteranceProgress, ensure_persistent_session_seeded,
    format_new_cognition_log_entries, ports::Clock,
};

use crate::utterance::UtteranceWriter;
#[cfg(test)]
use nuillu_types::builtin;
use schemars::{JsonSchema, Schema, SchemaGenerator};
use serde::{Deserialize, Serialize};

const SPEECH_PLANNING_PROMPT: &str = r#"Plan outward speech from the current cognition log and current scene target hints.
Use exactly one available tool.
When no speech is in progress, call prepare_speech exactly once when a grounded outward utterance can be prepared. Choose the participant whose question, request, warning, or need should be answered. Scene target hints are preferred but not exhaustive; use another concrete non-empty addressee when the cognition log supports it. Do not choose a participant merely because they are the topic, threat, object of advice, or quoted speaker. Use "everyone" only for explicit group/broadcast speech.
When no speech is in progress, call decline_speech_now only when a concrete blocker makes speech inappropriate or impossible now, such as no concrete addressee, no cognition-supported listener-facing content, a policy or consent conflict, or fresh evidence that invalidates speaking now. Put that blocker in blocking_reason.
When speech is already in progress, treat already emitted text as immutable. Call continue_speech to preserve the current target and continue coherently from the partial utterance. Call interrupt_and_redirect_speech only for urgent or safety-priority cognition that must interrupt the current listener and redirect immediately. Call abort_speech only when the partial utterance should stop without completion.
Put the speech-facing transformation of the cognition log in speech_content. It is the information that should survive into speech, with perspective, deixis, and addressee adjusted for outward utterance.
If the cognition log contains an explicit listener language request, such as asking for Japanese, include the requested language in the language field and transform speech_content for that language.
speech_content is not hidden reasoning, not a rubric, and not a generic summary. It should contain the load-bearing fact, answer, warning, advice, visible absence, or unknown-state evidence that the listener needs.
For questions or requests, transform the relevant cognition into an answer. Preserve answer polarity: yes/no/unknown must remain visible when supported by the cognition log.
For self-directed cognition, transform only the listener-relevant implication into outward substance. Keep first person only when the speaker is reporting perception, knowledge, uncertainty, consent, or shared action that directly answers the listener.
Do not invent policy, actions, or facts not supported by the cognition log. If the cognition log only supports a limited warning or uncertainty, keep speech_content limited.
For unknown evidence, make speech_content say unknown and include the concrete visible absence or missing evidence."#;

const FRESH_PLANNING_TURN_DEVELOPER_INSTRUCTION: &str = "Use exactly one tool: prepare_speech when listener-facing substance can be produced, or decline_speech_now only for a concrete blocker that makes speech inappropriate or impossible now.";
const ACTIVE_PLANNING_TURN_DEVELOPER_INSTRUCTION: &str = "Use exactly one tool: continue_speech for ordinary continuation of the current partial utterance, interrupt_and_redirect_speech only for urgent or safety-priority interruption, or abort_speech only when the partial should stop without completion.";

const GENERATION_PROMPT: &str = r#"Render the supplied substance as one concise in-world utterance to the named listener.
The substance is already transformed for outward speech. Render that transformed information; do not redo listener selection or add a new plan.
If a language is supplied, render the utterance in that language.
Preserve its answer polarity, addressee-facing perspective, direct warnings, advice, uncertainty, and visible-absence evidence.
Use the cognition log only to keep wording grounded. Do not add facts, turn an answer back into a question, turn listener-facing substance into a self-directed note, weaken direct content into vague caution, or merely restate the situation.
Do not mention implementation mechanics, lookup, reasoning, prompts, rubrics, or evaluation mechanics."#;

const COGNITION_CONTEXT_WINDOW: LlmContextWindow = LlmContextWindow::new(12, 600, 4_800);
const SPEECH_PLANNING_TURN_MAX_OUTPUT_TOKENS: u32 = 1024;
const SPEECH_GENERATION_TEXT_DELTA_SLICE_LIMIT: usize = 8;
const THINK_OPEN_TAG: &str = "<think>";
const THINK_CLOSE_TAG: &str = "</think>";
const COMPACTED_SPEAK_PLANNING_SESSION_PREFIX: &str = "Compacted speak planning session history:";
const COMPACTED_SPEAK_GENERATION_SESSION_PREFIX: &str =
    "Compacted speak generation session history:";
const PLANNING_SESSION_COMPACTION_FOCUS: &str = r#"Preserve prior speech target decisions,
selected targets, rejected/no-speech decisions, completed outward utterances, in-progress
speech continuations, and cognition-log context needed for future speak planning."#;
const GENERATION_SESSION_COMPACTION_FOCUS: &str = r#"Preserve completed outward utterances
and their addressees. Do not preserve per-turn generation request context unless it is part of
a completed utterance."#;

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
/// Call when proceeding to an outward utterance. The input carries cognition-log
/// information transformed for speech, not hidden analysis.
struct PrepareSpeechArgs {
    /// The concrete addressee who should hear the utterance.
    target: SpeechTarget,
    /// Requested output language when the cognition log contains an explicit
    /// listener language request.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    language: Option<String>,
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
/// impossible now.
struct DeclineSpeechNowArgs {
    blocking_reason: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
struct DeclineSpeechNowOutput {
    accepted: bool,
}

#[lutum::tool_input(name = "continue_speech", output = ContinueSpeechOutput)]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
/// Call when an in-progress utterance should continue to the same target.
struct ContinueSpeechArgs {
    /// Speech-facing information to render as a continuation of the already
    /// emitted partial utterance.
    speech_content: String,
    /// Requested output language when the continuation should preserve or apply
    /// an explicit listener language request.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    language: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
struct ContinueSpeechOutput {
    accepted: bool,
}

#[lutum::tool_input(name = "interrupt_and_redirect_speech", output = InterruptAndRedirectSpeechOutput)]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
/// Call only when urgent or safety-priority cognition must interrupt the current
/// partial utterance and redirect outward speech to another allowed target.
struct InterruptAndRedirectSpeechArgs {
    /// The participant who should immediately hear the redirected utterance.
    target: SpeechTarget,
    /// Requested output language when the redirected speech should preserve or
    /// apply an explicit listener language request.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    language: Option<String>,
    /// Speech-facing information to render for `target`.
    speech_content: String,
    /// Why the in-progress utterance must be interrupted now.
    interrupt_reason: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
struct InterruptAndRedirectSpeechOutput {
    accepted: bool,
}

#[lutum::tool_input(name = "abort_speech", output = AbortSpeechOutput)]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
/// Call only when the partial utterance should stop without completion.
struct AbortSpeechArgs {
    /// Why the in-progress utterance should stop without completion.
    interrupt_reason: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
struct AbortSpeechOutput {
    accepted: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema, lutum::Toolset)]
enum SpeakTools {
    PrepareSpeech(PrepareSpeechArgs),
    DeclineSpeechNow(DeclineSpeechNowArgs),
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema, lutum::Toolset)]
enum ActiveSpeakTools {
    ContinueSpeech(ContinueSpeechArgs),
    InterruptAndRedirectSpeech(InterruptAndRedirectSpeechArgs),
    AbortSpeech(AbortSpeechArgs),
}

#[derive(Clone, Debug)]
struct PlannedSpeech {
    args: PrepareSpeechArgs,
    target: String,
}

#[derive(Clone, Debug)]
enum FreshSpeechPlan {
    Prepare(PlannedSpeech),
    Decline { blocking_reason: String },
    None,
}

#[derive(Clone, Debug)]
enum ActiveSpeechPlan {
    Continue(PlannedSpeech),
    Redirect {
        plan: PlannedSpeech,
        interrupt_reason: String,
    },
    Abort {
        interrupt_reason: String,
    },
}

#[derive(Debug)]
pub struct SpeakBatch {
    pub(crate) updates: Vec<CognitionLogUpdated>,
    pub(crate) cognition_entries: Vec<CognitionLogEntryRecord>,
    pub(crate) continue_active_speech: bool,
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

struct ActiveSpeech {
    args: PrepareSpeechArgs,
    draft: GenerationDraft,
}

#[derive(Default)]
struct GenerationDeltaBuffer {
    deltas: Vec<String>,
    non_empty_text_delta_count: usize,
    think_filter: ThinkTagFilter,
}

impl GenerationDeltaBuffer {
    fn push(&mut self, delta: String) -> bool {
        let Some(delta) = self.think_filter.push(&delta) else {
            return self.non_empty_text_delta_count >= SPEECH_GENERATION_TEXT_DELTA_SLICE_LIMIT;
        };
        self.push_visible(delta)
    }

    fn finish(&mut self) {
        if let Some(delta) = self.think_filter.finish() {
            self.push_visible(delta);
        }
    }

    fn push_visible(&mut self, delta: String) -> bool {
        if !delta.is_empty() {
            self.non_empty_text_delta_count += 1;
        }
        self.deltas.push(delta);
        self.non_empty_text_delta_count >= SPEECH_GENERATION_TEXT_DELTA_SLICE_LIMIT
    }
}

#[derive(Default)]
struct ThinkTagFilter {
    pending: String,
    inside_think: bool,
}

impl ThinkTagFilter {
    fn push(&mut self, delta: &str) -> Option<String> {
        self.pending.push_str(delta);
        self.drain(false)
    }

    fn finish(&mut self) -> Option<String> {
        self.drain(true)
    }

    fn drain(&mut self, flush: bool) -> Option<String> {
        let mut out = String::new();
        loop {
            if self.inside_think {
                let Some(end) = self.pending.find(THINK_CLOSE_TAG) else {
                    self.keep_possible_tag_suffix(THINK_CLOSE_TAG, flush);
                    break;
                };
                self.pending.drain(..end + THINK_CLOSE_TAG.len());
                self.inside_think = false;
                continue;
            }

            let Some(start) = self.pending.find(THINK_OPEN_TAG) else {
                let keep = if flush {
                    0
                } else {
                    possible_tag_suffix_len(&self.pending, THINK_OPEN_TAG)
                };
                let emit_len = self.pending.len() - keep;
                out.push_str(&self.pending[..emit_len]);
                self.pending.drain(..emit_len);
                break;
            };

            out.push_str(&self.pending[..start]);
            self.pending.drain(..start + THINK_OPEN_TAG.len());
            self.inside_think = true;
        }

        (!out.is_empty()).then_some(out)
    }

    fn keep_possible_tag_suffix(&mut self, tag: &str, flush: bool) {
        if flush {
            self.pending.clear();
            return;
        }
        let keep = possible_tag_suffix_len(&self.pending, tag);
        if keep == 0 {
            self.pending.clear();
            return;
        }
        let suffix = self.pending[self.pending.len() - keep..].to_owned();
        self.pending = suffix;
    }
}

fn possible_tag_suffix_len(text: &str, tag: &str) -> usize {
    let max = tag.len().min(text.len());
    for len in (1..=max).rev() {
        let start = text.len() - len;
        if text.is_char_boundary(start) && tag.starts_with(&text[start..]) {
            return len;
        }
    }
    0
}

struct GenerationDeltaCollector {
    buffered_deltas: Arc<Mutex<GenerationDeltaBuffer>>,
}

#[async_trait]
impl EventHandler<TextTurnEvent, TextTurnState> for GenerationDeltaCollector {
    async fn on_event(
        &mut self,
        event: &TextTurnEvent,
        _cx: &HandlerContext<TextTurnState>,
    ) -> HandlerResult {
        if let TextTurnEvent::TextDelta { delta } = event {
            let mut buffered = self
                .buffered_deltas
                .lock()
                .expect("generation delta buffer mutex poisoned");
            if buffered.push(delta.clone()) {
                return Ok(HandlerDirective::Stop);
            }
        }
        Ok(HandlerDirective::Continue)
    }
}

fn render_completed_utterance_memo(draft: &GenerationDraft, text: &str) -> String {
    format!(
        "Completed utterance to {}:\n{}",
        draft.target.trim(),
        text.trim(),
    )
}

fn render_in_progress_utterance_memo(args: &PrepareSpeechArgs, draft: &GenerationDraft) -> String {
    format!(
        "Utterance in progress to {}. Continuation is pending.\nPlanned substance:\n{}\n\nAlready emitted:\n{}",
        draft.target.trim(),
        args.speech_content.trim(),
        if draft.accumulated.trim().is_empty() {
            "(none)"
        } else {
            draft.accumulated.trim()
        },
    )
}

fn render_aborted_utterance_memo(draft: &GenerationDraft, reason: &str) -> String {
    format!(
        "Aborted utterance to {}. Reason:\n{}\n\nAlready emitted:\n{}",
        draft.target.trim(),
        reason.trim(),
        if draft.accumulated.trim().is_empty() {
            "(none)"
        } else {
            draft.accumulated.trim()
        },
    )
}

fn render_declined_speech_memo(reason: &str) -> String {
    format!("Declined outward speech. Reason:\n{}", reason.trim())
}

fn render_completed_utterance_planning_record(draft: &GenerationDraft, text: &str) -> String {
    format!(
        "Completed outward utterance to {}:\n{}",
        draft.target.trim(),
        text.trim(),
    )
}

fn render_aborted_utterance_planning_record(draft: &GenerationDraft, reason: &str) -> String {
    format!(
        "Aborted outward speech to {}. Reason:\n{}\n\nAlready emitted:\n{}",
        draft.target.trim(),
        reason.trim(),
        if draft.accumulated.trim().is_empty() {
            "(none)"
        } else {
            draft.accumulated.trim()
        },
    )
}

fn format_generation_input(
    cognition_context: &str,
    args: &PrepareSpeechArgs,
    draft: &GenerationDraft,
) -> String {
    let mut out = format!(
        "{}\n\nSpeak to: {}",
        cognition_context.trim(),
        draft.target.trim(),
    );
    if let Some(language) = trimmed_optional(args.language.as_deref()) {
        out.push_str(&format!("\nLanguage: {language}"));
    }
    out.push_str(&format!(
        "\n\nSubstance to express:\n{}",
        args.speech_content.trim()
    ));
    out
}

fn format_planning_input(
    cognition_context: &str,
    active: Option<&ActiveSpeech>,
    target_hints: &[String],
) -> String {
    let mut out = cognition_context.trim().to_owned();
    if !target_hints.is_empty() {
        out.push_str("\n\nPreferred visible speech targets (not exhaustive): ");
        out.push_str(&target_hints.join(", "));
        out.push_str("\nUse another concrete non-empty target when cognition supports it.");
    }
    if let Some(active) = active {
        out.push_str("\n\nCurrent outward speech in progress:");
        out.push_str(&format!("\n- Target: {}", active.draft.target.trim()));
        if let Some(language) = trimmed_optional(active.args.language.as_deref()) {
            out.push_str(&format!("\n- Language: {language}"));
        }
        out.push_str(&format!(
            "\n- Planned substance: {}",
            active.args.speech_content.trim()
        ));
        out.push_str(&format!(
            "\n- Already emitted: {}",
            if active.draft.accumulated.trim().is_empty() {
                "(none)"
            } else {
                active.draft.accumulated.trim()
            }
        ));
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
    if target.trim().is_empty() {
        return false;
    }
    true
}

fn push_generation_context(
    session: &mut Session,
    cognition_context: &str,
    args: &PrepareSpeechArgs,
    draft: &GenerationDraft,
) {
    session.push_ephemeral_user(format_generation_input(cognition_context, args, draft));
    if !draft.accumulated.is_empty() {
        session.push_ephemeral_assistant_text(draft.accumulated.clone());
    }
}

fn push_completed_generation_turn(session: &mut Session, text: &str) {
    session.input_mut().push(ModelInputItem::turn(Arc::new(
        AssistantTurnView::from_items(&[AssistantTurnItem::Text(text.to_owned())]),
    )));
}

#[cfg(test)]
fn cognition_context_fallback(now: DateTime<Utc>) -> String {
    format!(
        "Current cognition log at {}:\n- none",
        now.to_rfc3339_opts(SecondsFormat::Secs, true)
    )
}

fn speech_cognition_context_from_entries(
    records: &[CognitionLogEntryRecord],
    now: DateTime<Utc>,
) -> Option<String> {
    format_new_cognition_log_entries(records, now, COGNITION_CONTEXT_WINDOW)
}

fn active_speech_cognition_context_from_entries(
    records: &[CognitionLogEntryRecord],
    now: DateTime<Utc>,
) -> String {
    speech_cognition_context_from_entries(records, now).unwrap_or_else(|| {
        format!(
            "New cognition entries at {}:\n- none since the previous speech slice",
            now.to_rfc3339_opts(SecondsFormat::Secs, true)
        )
    })
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

fn push_planning_context(
    session: &mut Session,
    cognition_context: &str,
    active: Option<&ActiveSpeech>,
    target_hints: &[String],
) {
    session.push_user(format_planning_input(
        cognition_context,
        active,
        target_hints,
    ));
    let instruction = if active.is_some() {
        ACTIVE_PLANNING_TURN_DEVELOPER_INSTRUCTION
    } else {
        FRESH_PLANNING_TURN_DEVELOPER_INSTRUCTION
    };
    session.push_ephemeral_developer(instruction);
}

#[derive(Debug)]
enum GenerationStreamOutcome {
    Completed,
    LengthLimited,
    NoVisibleOutput,
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
    generation_reflected_cognition_indices: HashSet<u64>,
    self_wake: SelfWake,
    active_speech: Option<ActiveSpeech>,
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
    pub self_wake: SelfWake,
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
            self_wake,
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
            generation_reflected_cognition_indices: HashSet::new(),
            self_wake,
            active_speech: None,
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
        let now = self.clock.now();
        let (cognition_context, plan, mut draft) = if let Some(active) = self.active_speech.take() {
            let cognition_context =
                active_speech_cognition_context_from_entries(&batch.cognition_entries, now);
            let should_continue_without_planning = batch.continue_active_speech
                && batch.updates.is_empty()
                && batch.cognition_entries.is_empty();
            if should_continue_without_planning {
                let plan = PlannedSpeech {
                    args: active.args,
                    target: active.draft.target.clone(),
                };
                (cognition_context, plan, active.draft)
            } else {
                let Some(active_plan) = self
                    .plan_active_speech(cx, &cognition_context, &active)
                    .await?
                else {
                    self.mark_generation_cognition_entries_reflected(&batch.cognition_entries);
                    self.active_speech = Some(active);
                    self.self_wake.wake();
                    return Ok(());
                };
                match active_plan {
                    ActiveSpeechPlan::Continue(plan) => (cognition_context, plan, active.draft),
                    ActiveSpeechPlan::Redirect {
                        plan,
                        interrupt_reason,
                    } => {
                        self.record_aborted_generation(cx, &active.draft, &interrupt_reason)
                            .await?;
                        let draft =
                            GenerationDraft::new(self.utterance.next_generation_id(), &plan.target);
                        (cognition_context, plan, draft)
                    }
                    ActiveSpeechPlan::Abort { interrupt_reason } => {
                        self.record_aborted_generation(cx, &active.draft, &interrupt_reason)
                            .await?;
                        self.mark_generation_cognition_entries_reflected(&batch.cognition_entries);
                        return Ok(());
                    }
                }
            }
        } else {
            let Some(cognition_context) =
                speech_cognition_context_from_entries(&batch.cognition_entries, now)
            else {
                return Ok(());
            };
            let plan = match self.plan_speech(cx, &cognition_context).await? {
                FreshSpeechPlan::Prepare(plan) => plan,
                FreshSpeechPlan::Decline { blocking_reason } => {
                    self.record_declined_speech(&blocking_reason).await;
                    self.mark_generation_cognition_entries_reflected(&batch.cognition_entries);
                    return Ok(());
                }
                FreshSpeechPlan::None => {
                    self.mark_generation_cognition_entries_reflected(&batch.cognition_entries);
                    return Ok(());
                }
            };
            let draft = GenerationDraft::new(self.utterance.next_generation_id(), &plan.target);
            (cognition_context, plan, draft)
        };

        match self
            .collect_generation_slice(cx, cognition_context.clone(), &plan.args, &mut draft)
            .await?
        {
            GenerationStreamOutcome::Completed => {
                self.mark_generation_cognition_entries_reflected(&batch.cognition_entries);
            }
            GenerationStreamOutcome::LengthLimited => {
                self.mark_generation_cognition_entries_reflected(&batch.cognition_entries);
                self.record_in_progress_generation(&plan.args, &draft).await;
                self.active_speech = Some(ActiveSpeech {
                    args: plan.args.clone(),
                    draft,
                });
                self.self_wake.wake();
            }
            GenerationStreamOutcome::NoVisibleOutput => {
                self.mark_generation_cognition_entries_reflected(&batch.cognition_entries);
            }
        }
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
        push_planning_context(
            &mut self.planning_session,
            cognition_context,
            None,
            &target_hints,
        );

        let lutum = self.llm.lutum().await;
        let target_schema = freeform_speech_target_schema();
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
                let mut plan = FreshSpeechPlan::None;
                let mut results = Vec::new();
                for call in round.tool_calls.iter().cloned() {
                    match call {
                        SpeakToolsCall::PrepareSpeech(call) => {
                            let target = call.input.target.as_str().trim().to_owned();
                            let accepted = matches!(plan, FreshSpeechPlan::None)
                                && is_non_empty_target(&target);
                            if accepted {
                                plan = FreshSpeechPlan::Prepare(PlannedSpeech {
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
                            let accepted = matches!(plan, FreshSpeechPlan::None);
                            if accepted {
                                plan = FreshSpeechPlan::Decline {
                                    blocking_reason: call.input.blocking_reason.clone(),
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
                    .commit(&mut self.planning_session, results)
                    .context("commit speak planning tool round")?;
                cx.compact_and_save(&mut self.planning_session, usage)
                    .await?;
                Ok(plan)
            }
        }
    }

    async fn plan_active_speech(
        &mut self,
        cx: &nuillu_module::ActivateCx<'_>,
        cognition_context: &str,
        active: &ActiveSpeech,
    ) -> Result<Option<ActiveSpeechPlan>> {
        self.ensure_planning_session_seeded(cx);
        let scene_target_schema = self.scene.target_schema();
        let target_hints = target_hints_from_schema(&scene_target_schema);
        push_planning_context(
            &mut self.planning_session,
            cognition_context,
            Some(active),
            &target_hints,
        );

        let lutum = self.llm.lutum().await;
        let target_schema = freeform_speech_target_schema();
        let outcome = SPEECH_TARGET_SCHEMA
            .scope(target_schema, async {
                self.planning_session
                    .text_turn()
                    .tools::<ActiveSpeakTools>()
                    .available_tools([
                        ActiveSpeakToolsSelector::ContinueSpeech,
                        ActiveSpeakToolsSelector::InterruptAndRedirectSpeech,
                        ActiveSpeakToolsSelector::AbortSpeech,
                    ])
                    .require_any_tool()
                    .max_output_tokens(SPEECH_PLANNING_TURN_MAX_OUTPUT_TOKENS)
                    .collect_controlled_with(
                        &lutum,
                        nuillu_module::AbortOnAvailableToolNameInText::new(),
                    )
                    .await
                    .context("active speak planning turn failed")
            })
            .await?;

        match outcome {
            TextStepOutcomeWithTools::Finished(result) => {
                cx.compact_and_save(&mut self.planning_session, result.usage)
                    .await?;
                let detail = "model finished with assistant output but no tool call \
                    (require_any_tool should have prevented this outcome)";
                cx.warn(format!("active speak planning failed: {detail}"));
                anyhow::bail!(
                    "active speak planning finished without required tool call: {detail}"
                );
            }
            TextStepOutcomeWithTools::FinishedNoOutput(result) => {
                cx.compact_and_save(&mut self.planning_session, result.usage)
                    .await?;
                let detail = "model finished with no output and no tool call \
                    (require_any_tool should have prevented this outcome)";
                cx.warn(format!("active speak planning failed: {detail}"));
                anyhow::bail!(
                    "active speak planning finished without required tool call: {detail}"
                );
            }
            TextStepOutcomeWithTools::NeedsTools(round) => {
                let usage = round.usage;
                nuillu_module::emit_trace_tool_calls(&round.tool_calls);
                if round.tool_calls.is_empty() {
                    let detail = "model returned NeedsTools outcome with empty tool_calls; \
                        expected continue_speech, interrupt_and_redirect_speech, or abort_speech";
                    cx.warn(format!("active speak planning failed: {detail}"));
                    anyhow::bail!(
                        "active speak planning finished without required tool call: {detail}"
                    );
                }
                let mut selected = None;
                let mut results = Vec::new();
                for call in round.tool_calls.iter().cloned() {
                    match call {
                        ActiveSpeakToolsCall::ContinueSpeech(call) => {
                            let accepted = selected.is_none();
                            if accepted {
                                let target = active.draft.target.trim().to_owned();
                                selected = Some(ActiveSpeechPlan::Continue(PlannedSpeech {
                                    args: PrepareSpeechArgs {
                                        target: SpeechTarget::from(target.clone()),
                                        language: call
                                            .input
                                            .language
                                            .clone()
                                            .or_else(|| active.args.language.clone()),
                                        speech_content: call.input.speech_content.clone(),
                                    },
                                    target,
                                }));
                            }
                            results.push(
                                call.complete(ContinueSpeechOutput { accepted })
                                    .context("complete continue_speech tool call")?,
                            );
                        }
                        ActiveSpeakToolsCall::InterruptAndRedirectSpeech(call) => {
                            let target = call.input.target.as_str().trim().to_owned();
                            let accepted = selected.is_none() && is_non_empty_target(&target);
                            if accepted {
                                selected = Some(ActiveSpeechPlan::Redirect {
                                    plan: PlannedSpeech {
                                        args: PrepareSpeechArgs {
                                            target: call.input.target.clone(),
                                            language: call.input.language.clone(),
                                            speech_content: call.input.speech_content.clone(),
                                        },
                                        target,
                                    },
                                    interrupt_reason: call.input.interrupt_reason.clone(),
                                });
                            }
                            results.push(
                                call.complete(InterruptAndRedirectSpeechOutput { accepted })
                                    .context("complete interrupt_and_redirect_speech tool call")?,
                            );
                        }
                        ActiveSpeakToolsCall::AbortSpeech(call) => {
                            let accepted = selected.is_none();
                            if accepted {
                                selected = Some(ActiveSpeechPlan::Abort {
                                    interrupt_reason: call.input.interrupt_reason.clone(),
                                });
                            }
                            results.push(
                                call.complete(AbortSpeechOutput { accepted })
                                    .context("complete abort_speech tool call")?,
                            );
                        }
                    }
                }
                round
                    .commit(&mut self.planning_session, results)
                    .context("commit active speak planning tool round")?;
                cx.compact_and_save(&mut self.planning_session, usage)
                    .await?;
                Ok(selected)
            }
        }
    }

    async fn collect_generation_slice(
        &mut self,
        cx: &nuillu_module::ActivateCx<'_>,
        cognition_context: String,
        args: &PrepareSpeechArgs,
        draft: &mut GenerationDraft,
    ) -> Result<GenerationStreamOutcome> {
        self.ensure_generation_session_seeded(cx);
        let mut turn_session = self.generation_session.clone();
        push_generation_context(&mut turn_session, &cognition_context, args, draft);

        let buffered_deltas = Arc::new(Mutex::new(GenerationDeltaBuffer::default()));
        let lutum = self.llm.lutum().await;
        let result = turn_session
            .text_turn()
            .on_event(GenerationDeltaCollector {
                buffered_deltas: Arc::clone(&buffered_deltas),
            })
            .collect_staged(&lutum)
            .await;

        self.emit_buffered_generation_deltas(draft, buffered_deltas)
            .await;

        match result {
            Ok(staged) => {
                let usage = staged.usage;
                let finish_reason = staged.finish_reason.clone();
                if finish_reason == FinishReason::Length {
                    staged.turn.discard();
                    cx.compact_and_save(&mut self.generation_session, usage)
                        .await?;
                    return Ok(if draft.accumulated.trim().is_empty() {
                        GenerationStreamOutcome::NoVisibleOutput
                    } else {
                        GenerationStreamOutcome::LengthLimited
                    });
                }
                staged.turn.discard();
                self.finish_completed_generation_slice(cx, turn_session, usage, draft)
                    .await
            }
            Err(CollectError::Reduction {
                source: TextTurnReductionError::OutputLimitExceeded(limit),
                partial,
            }) => {
                let usage = partial.usage.unwrap_or(limit.usage);
                cx.compact_and_save(&mut self.generation_session, usage)
                    .await?;
                Ok(if draft.accumulated.trim().is_empty() {
                    GenerationStreamOutcome::NoVisibleOutput
                } else {
                    GenerationStreamOutcome::LengthLimited
                })
            }
            Err(CollectError::Stopped { partial }) => {
                let usage = partial.usage.unwrap_or_else(Usage::zero);
                cx.compact_and_save(&mut self.generation_session, usage)
                    .await?;
                Ok(if draft.accumulated.trim().is_empty() {
                    GenerationStreamOutcome::NoVisibleOutput
                } else {
                    GenerationStreamOutcome::LengthLimited
                })
            }
            Err(CollectError::UnexpectedEof { partial }) => {
                let usage = partial.usage.unwrap_or_else(Usage::zero);
                self.finish_completed_generation_slice(cx, turn_session, usage, draft)
                    .await
            }
            Err(CollectError::Execution { source, partial })
            | Err(CollectError::Handler { source, partial }) => {
                let usage = partial.usage.unwrap_or_else(Usage::zero);
                self.handle_generation_stream_error(cx, draft, usage, source)
                    .await
            }
            Err(error) => Err(error).context("speak generation turn failed"),
        }
    }

    async fn finish_completed_generation_slice(
        &mut self,
        cx: &nuillu_module::ActivateCx<'_>,
        turn_session: Session,
        usage: Usage,
        draft: &GenerationDraft,
    ) -> Result<GenerationStreamOutcome> {
        let text = draft.accumulated.trim().to_owned();
        if text.is_empty() {
            cx.compact_and_save(&mut self.generation_session, usage)
                .await?;
            return Ok(GenerationStreamOutcome::NoVisibleOutput);
        }

        self.generation_session = turn_session;
        push_completed_generation_turn(&mut self.generation_session, &text);
        cx.compact_and_save(&mut self.generation_session, usage)
            .await?;
        self.record_completed_generation(cx, draft, &text).await?;
        Ok(GenerationStreamOutcome::Completed)
    }

    async fn handle_generation_stream_error(
        &mut self,
        cx: &nuillu_module::ActivateCx<'_>,
        draft: &GenerationDraft,
        usage: Usage,
        source: AgentError,
    ) -> Result<GenerationStreamOutcome> {
        if draft.accumulated.trim().is_empty() {
            return Err(source).context("speak generation turn failed");
        }

        cx.warn(format!(
            "speak generation interrupted after partial output; keeping active speech: {source}"
        ));
        cx.compact_and_save(&mut self.generation_session, usage)
            .await?;
        Ok(GenerationStreamOutcome::LengthLimited)
    }

    fn mark_generation_cognition_entries_reflected(&mut self, entries: &[CognitionLogEntryRecord]) {
        for record in entries {
            self.generation_reflected_cognition_indices
                .insert(record.index);
        }
    }

    async fn record_completed_generation(
        &mut self,
        cx: &nuillu_module::ActivateCx<'_>,
        draft: &GenerationDraft,
        text: &str,
    ) -> Result<()> {
        self.memo
            .write(render_completed_utterance_memo(draft, text))
            .await;
        self.utterance
            .record_progress(UtteranceProgress::completed(
                draft.generation_id,
                draft.sequence,
                draft.target.clone(),
                text.to_owned(),
            ))
            .await;
        if text.is_empty() {
            return Ok(());
        }

        self.utterance
            .emit(draft.target.clone(), draft.generation_id, text.to_owned())
            .await;
        self.ensure_planning_session_seeded(cx);
        self.planning_session
            .push_system(render_completed_utterance_planning_record(draft, text));
        cx.compact_and_save(&mut self.planning_session, Usage::zero())
            .await?;
        Ok(())
    }

    async fn record_in_progress_generation(
        &mut self,
        args: &PrepareSpeechArgs,
        draft: &GenerationDraft,
    ) {
        self.memo
            .write(render_in_progress_utterance_memo(args, draft))
            .await;
    }

    async fn record_aborted_generation(
        &mut self,
        cx: &nuillu_module::ActivateCx<'_>,
        draft: &GenerationDraft,
        reason: &str,
    ) -> Result<()> {
        self.memo
            .write(render_aborted_utterance_memo(draft, reason))
            .await;
        self.utterance
            .record_progress(UtteranceProgress::aborted(
                draft.generation_id,
                draft.sequence,
                draft.target.clone(),
                draft.accumulated.clone(),
            ))
            .await;
        self.utterance
            .abort(
                draft.target.clone(),
                draft.generation_id,
                draft.sequence,
                draft.accumulated.clone(),
                reason.to_owned(),
            )
            .await;
        self.ensure_planning_session_seeded(cx);
        self.planning_session
            .push_system(render_aborted_utterance_planning_record(draft, reason));
        cx.compact_and_save(&mut self.planning_session, Usage::zero())
            .await?;
        Ok(())
    }

    async fn record_declined_speech(&self, reason: &str) {
        self.memo.write(render_declined_speech_memo(reason)).await;
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

    async fn emit_buffered_generation_deltas(
        &self,
        draft: &mut GenerationDraft,
        buffered_deltas: Arc<Mutex<GenerationDeltaBuffer>>,
    ) {
        let deltas = {
            let mut buffered = buffered_deltas
                .lock()
                .expect("generation delta buffer mutex poisoned");
            buffered.finish();
            std::mem::take(&mut buffered.deltas)
        };
        for delta in deltas {
            let sequence = draft.push_delta(&delta);
            self.utterance
                .emit_delta(draft.target.clone(), draft.generation_id, sequence, delta)
                .await;
            self.record_streaming_progress(draft).await;
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
        if self.active_speech.is_some() {
            let updates = self
                .cognition_updates
                .take_ready_items()?
                .items
                .into_iter()
                .map(|envelope| envelope.body)
                .collect();
            let cognition_entries = self.cognition_log.unread_events().await;
            return Ok(SpeakBatch {
                updates,
                cognition_entries,
                continue_active_speech: true,
            });
        }

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
            continue_active_speech: false,
        })
    }
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::rc::Rc;
    use std::sync::{Arc, Mutex};
    use std::time::Duration;

    use lutum::{
        AdapterStructuredTurn, AdapterTextTurn, AgentError, AssistantInputItem,
        ErasedStructuredTurnEventStream, ErasedTextTurnEventStream, FinishReason, InputMessageRole,
        MaxOutputTokens, MessageContent, MockError, MockLlmAdapter, MockTextScenario, ModelInput,
        ModelInputItem, RawTextTurnEvent, SharedPoolBudgetManager, SharedPoolBudgetOptions,
        TurnAdapter, Usage,
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

    fn prepare_speech_scenario(target: &str, speech_content: &str) -> MockTextScenario {
        prepare_speech_scenario_with_language(target, None, speech_content)
    }

    fn prepare_speech_scenario_with_language(
        target: &str,
        language: Option<&str>,
        speech_content: &str,
    ) -> MockTextScenario {
        let arguments_json = serde_json::json!({
            "target": target,
            "language": language,
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

    fn continue_speech_scenario(speech_content: &str) -> MockTextScenario {
        let arguments_json = serde_json::json!({
            "speech_content": speech_content
        })
        .to_string();
        MockTextScenario::events(vec![
            Ok(RawTextTurnEvent::Started {
                request_id: Some("continue-speech".into()),
                model: "mock".into(),
            }),
            Ok(RawTextTurnEvent::ToolCallChunk {
                id: "call-continue-speech".into(),
                name: "continue_speech".into(),
                arguments_json_delta: arguments_json,
            }),
            Ok(RawTextTurnEvent::Completed {
                request_id: Some("continue-speech".into()),
                finish_reason: FinishReason::ToolCall,
                usage: Usage::zero(),
            }),
        ])
    }

    fn interrupt_and_redirect_speech_scenario(
        target: &str,
        speech_content: &str,
        interrupt_reason: &str,
    ) -> MockTextScenario {
        let arguments_json = serde_json::json!({
            "target": target,
            "speech_content": speech_content,
            "interrupt_reason": interrupt_reason
        })
        .to_string();
        MockTextScenario::events(vec![
            Ok(RawTextTurnEvent::Started {
                request_id: Some("interrupt-and-redirect-speech".into()),
                model: "mock".into(),
            }),
            Ok(RawTextTurnEvent::ToolCallChunk {
                id: "call-interrupt-and-redirect-speech".into(),
                name: "interrupt_and_redirect_speech".into(),
                arguments_json_delta: arguments_json,
            }),
            Ok(RawTextTurnEvent::Completed {
                request_id: Some("interrupt-and-redirect-speech".into()),
                finish_reason: FinishReason::ToolCall,
                usage: Usage::zero(),
            }),
        ])
    }

    fn abort_speech_scenario(interrupt_reason: &str) -> MockTextScenario {
        let arguments_json = serde_json::json!({
            "interrupt_reason": interrupt_reason
        })
        .to_string();
        MockTextScenario::events(vec![
            Ok(RawTextTurnEvent::Started {
                request_id: Some("abort-speech".into()),
                model: "mock".into(),
            }),
            Ok(RawTextTurnEvent::ToolCallChunk {
                id: "call-abort-speech".into(),
                name: "abort_speech".into(),
                arguments_json_delta: arguments_json,
            }),
            Ok(RawTextTurnEvent::Completed {
                request_id: Some("abort-speech".into()),
                finish_reason: FinishReason::ToolCall,
                usage: Usage::zero(),
            }),
        ])
    }

    fn generation_text_deltas_scenario_with_finish_reason(
        deltas: &[&str],
        finish_reason: FinishReason,
    ) -> MockTextScenario {
        let mut events = vec![Ok(RawTextTurnEvent::Started {
            request_id: Some("speak-text".into()),
            model: "mock".into(),
        })];
        events.extend(deltas.iter().map(|delta| {
            Ok(RawTextTurnEvent::TextDelta {
                delta: (*delta).into(),
            })
        }));
        events.push(Ok(RawTextTurnEvent::Completed {
            request_id: Some("speak-text".into()),
            finish_reason,
            usage: Usage::zero(),
        }));
        MockTextScenario::events(events)
    }

    fn generation_text_sliced_scenario(deltas: &[&str]) -> MockTextScenario {
        assert!(deltas.iter().all(|delta| !delta.is_empty()));
        generation_text_deltas_scenario_with_finish_reason(deltas, FinishReason::Stop)
    }

    fn generation_reasoning_then_text_scenario(
        reasoning_deltas: &[&str],
        text_deltas: &[&str],
        finish_reason: FinishReason,
    ) -> MockTextScenario {
        let mut events = vec![Ok(RawTextTurnEvent::Started {
            request_id: Some("speak-text".into()),
            model: "mock".into(),
        })];
        events.extend(reasoning_deltas.iter().map(|delta| {
            Ok(RawTextTurnEvent::ReasoningDelta {
                delta: (*delta).into(),
            })
        }));
        events.extend(text_deltas.iter().map(|delta| {
            Ok(RawTextTurnEvent::TextDelta {
                delta: (*delta).into(),
            })
        }));
        events.push(Ok(RawTextTurnEvent::Completed {
            request_id: Some("speak-text".into()),
            finish_reason,
            usage: Usage::zero(),
        }));
        MockTextScenario::events(events)
    }

    fn generation_text_then_error_scenario(deltas: &[&str]) -> MockTextScenario {
        let mut events = vec![Ok(RawTextTurnEvent::Started {
            request_id: Some("speak-text".into()),
            model: "mock".into(),
        })];
        events.extend(deltas.iter().map(|delta| {
            Ok(RawTextTurnEvent::TextDelta {
                delta: (*delta).into(),
            })
        }));
        events.push(Err(MockError::Synthetic {
            message: "stream closed before terminal event".into(),
        }));
        MockTextScenario::events(events)
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
            language: None,
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
                        self_wake: caps.self_wake(),
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
                ModelInputItem::Assistant(AssistantInputItem::Text(text)) => {
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
    async fn planning_and_generation_context_use_new_entries_for_each_batch() {
        let adapter = MockLlmAdapter::new()
            .with_text_scenario(decline_speech_now_scenario("not enough to say yet"))
            .with_text_scenario(decline_speech_now_scenario("still not enough to say"));
        let mut allocation = ResourceAllocation::default();
        allocation.set(builtin::speak(), ModuleConfig::default());
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
            .filter(|text| text.starts_with("New cognition entries at "))
            .collect::<Vec<_>>();
        assert_eq!(contexts.len(), 2);
        assert!(contexts[0].contains("first cognition"));
        assert!(!contexts[0].contains("second cognition"));
        assert!(contexts[1].contains("second cognition"));
        assert!(!contexts[1].contains("first cognition"));
        assert!(!contexts[0].contains("Current cognition log at"));
        assert!(!contexts[1].contains("Current cognition log at"));

        let generation_contexts =
            session_message_texts(&module.generation_session, InputMessageRole::User)
                .into_iter()
                .filter(|text| text.starts_with("New cognition entries at "))
                .collect::<Vec<_>>();
        assert!(
            generation_contexts.is_empty(),
            "generation request context is per-turn ephemeral, not durable session history"
        );
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

    #[tokio::test(flavor = "current_thread")]
    async fn language_request_flows_from_planning_to_generation_input() {
        let adapter = MockLlmAdapter::new()
            .with_text_scenario(prepare_speech_scenario_with_language(
                "Ryo",
                Some("Japanese"),
                "日本語で短く返事する。",
            ))
            .with_text_scenario(generation_text_scenario("もちろん、日本語で話すよ。"));
        let capture = CapturingAdapter::new(adapter);
        let observed = capture.clone();
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
            speak_module_with_turn_adapter(blackboard.clone(), Arc::new(capture), sink).await;
        caps.scene().set([Participant::new("Ryo")]);
        let now = SystemClock.now();
        publish_cognition_update(&blackboard, &caps, now, "Ryo says, \"日本語でお願い\".").await;

        let batch = module.next_batch().await.unwrap();
        let cx = test_activate_cx(&module, now).await;
        SpeakModule::activate(&mut module, &cx, &batch)
            .await
            .unwrap();

        assert_eq!(
            completed.borrow().as_slice(),
            &[("Ryo".to_string(), "もちろん、日本語で話すよ。".to_string())]
        );
        let inputs = observed.text_inputs();
        assert_eq!(inputs.len(), 2);
        let generation_text = model_input_text(&inputs[1]);
        assert!(generation_text.contains("Speak to: Ryo"));
        assert!(generation_text.contains("Language: Japanese"));
        assert!(generation_text.contains("Substance to express:\n日本語で短く返事する。"));
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
    fn generation_input_includes_planned_language() {
        let draft = GenerationDraft::new(8, "Ryo");
        let args = PrepareSpeechArgs {
            target: SpeechTarget::from("Ryo"),
            language: Some("Japanese".into()),
            speech_content: "日本語で短く返事する。".into(),
        };

        let text = format_generation_input("- Ryo says, \"日本語でお願い\".", &args, &draft);

        assert!(text.contains("Speak to: Ryo"));
        assert!(text.contains("Language: Japanese"));
        assert!(text.contains("Substance to express:\n日本語で短く返事する。"));
    }

    #[test]
    fn unknown_evidence_plan_renders_uncertainty_and_visible_absence() {
        let draft = GenerationDraft::new(8, "Pibi");
        let args = PrepareSpeechArgs {
            target: SpeechTarget::from("Pibi"),
            language: None,
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
        assert!(!SPEECH_PLANNING_PROMPT.contains("greet"));
        assert!(!SPEECH_PLANNING_PROMPT.contains("Hi"));
        assert!(!SPEECH_PLANNING_PROMPT.contains("allocation"));
        assert!(!SPEECH_PLANNING_PROMPT.contains("allocated"));
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
                        self_wake: caps.self_wake(),
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
        let planning_text = session_input_text(&module.planning_session);
        assert!(planning_text.contains("Koro asks Nuillu to help them stay safe."));
        assert!(planning_text.contains("Completed outward utterance to Koro:\nKoro, stay close."));
        let generation_text = session_input_text(&module.generation_session);
        assert!(!generation_text.contains("New cognition entries at "));
        assert!(!generation_text.contains("Koro asks Nuillu to help them stay safe."));
        assert_eq!(
            session_turn_texts(&module.generation_session),
            vec!["Koro, stay close.".to_string()]
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn speak_sets_max_output_tokens_for_planning_only() {
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
            turns[1].config.generation.max_output_tokens, None,
            "speak generation is sliced by visible text deltas so reasoning tokens do not consume the slice budget"
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn generation_collect_emits_buffered_text_deltas() {
        let adapter = MockLlmAdapter::new().with_text_scenario(
            generation_text_deltas_scenario_with_finish_reason(
                &["Koro", ", stay close."],
                FinishReason::Stop,
            ),
        );
        let mut allocation = ResourceAllocation::default();
        allocation.set(builtin::speak(), ModuleConfig::default());
        allocation.set_activation(builtin::speak(), ActivationRatio::ONE);
        let blackboard = Blackboard::with_allocation(allocation);
        let deltas = Rc::new(RefCell::new(Vec::new()));
        let sink: Rc<dyn UtteranceSink> = Rc::new(CapturingDeltaSink {
            deltas: Rc::clone(&deltas),
            first_delta: RefCell::new(None),
        });
        let (mut module, _caps) =
            speak_module_with_turn_adapter(blackboard, Arc::new(adapter), sink).await;
        let now = SystemClock.now();
        let cx = test_activate_cx(&module, now).await;
        let mut draft = GenerationDraft::new(0, "Koro");
        let args = test_prepare_speech_args("Koro");

        let outcome = module
            .collect_generation_slice(&cx, "- initial awareness".into(), &args, &mut draft)
            .await
            .unwrap();

        assert!(matches!(outcome, GenerationStreamOutcome::Completed));
        assert_eq!(
            deltas.borrow().as_slice(),
            &[
                ("Koro".to_string(), 0, 0, "Koro".to_string()),
                ("Koro".to_string(), 0, 1, ", stay close.".to_string()),
            ]
        );
        assert_eq!(draft.accumulated, "Koro, stay close.");
    }

    #[tokio::test(flavor = "current_thread")]
    async fn generation_slice_limit_ignores_reasoning_deltas() {
        let adapter =
            MockLlmAdapter::new().with_text_scenario(generation_reasoning_then_text_scenario(
                &[
                    "think-0", "think-1", "think-2", "think-3", "think-4", "think-5", "think-6",
                    "think-7", "think-8", "think-9",
                ],
                &["Koro", ", stay close."],
                FinishReason::Stop,
            ));
        let mut allocation = ResourceAllocation::default();
        allocation.set(builtin::speak(), ModuleConfig::default());
        allocation.set_activation(builtin::speak(), ActivationRatio::ONE);
        let blackboard = Blackboard::with_allocation(allocation);
        let deltas = Rc::new(RefCell::new(Vec::new()));
        let sink: Rc<dyn UtteranceSink> = Rc::new(CapturingDeltaSink {
            deltas: Rc::clone(&deltas),
            first_delta: RefCell::new(None),
        });
        let (mut module, _caps) =
            speak_module_with_turn_adapter(blackboard, Arc::new(adapter), sink).await;
        let now = SystemClock.now();
        let cx = test_activate_cx(&module, now).await;
        let mut draft = GenerationDraft::new(0, "Koro");
        let args = test_prepare_speech_args("Koro");

        let outcome = module
            .collect_generation_slice(&cx, "- initial awareness".into(), &args, &mut draft)
            .await
            .unwrap();

        assert!(matches!(outcome, GenerationStreamOutcome::Completed));
        assert_eq!(draft.accumulated, "Koro, stay close.");
        assert_eq!(
            deltas.borrow().as_slice(),
            &[
                ("Koro".to_string(), 0, 0, "Koro".to_string()),
                ("Koro".to_string(), 0, 1, ", stay close.".to_string()),
            ]
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn generation_think_tag_text_is_not_emitted_or_persisted() {
        let adapter = MockLlmAdapter::new().with_text_scenario(
            generation_text_deltas_scenario_with_finish_reason(
                &[
                    "<thi",
                    "nk>hidden reasoning",
                    "</thi",
                    "nk>Koro",
                    ", stay close.",
                ],
                FinishReason::Stop,
            ),
        );
        let mut allocation = ResourceAllocation::default();
        allocation.set(builtin::speak(), ModuleConfig::default());
        allocation.set_activation(builtin::speak(), ActivationRatio::ONE);
        let blackboard = Blackboard::with_allocation(allocation);
        let deltas = Rc::new(RefCell::new(Vec::new()));
        let sink: Rc<dyn UtteranceSink> = Rc::new(CapturingDeltaSink {
            deltas: Rc::clone(&deltas),
            first_delta: RefCell::new(None),
        });
        let (mut module, _caps) =
            speak_module_with_turn_adapter(blackboard, Arc::new(adapter), sink).await;
        let now = SystemClock.now();
        let cx = test_activate_cx(&module, now).await;
        let mut draft = GenerationDraft::new(0, "Koro");
        let args = test_prepare_speech_args("Koro");

        let outcome = module
            .collect_generation_slice(&cx, "- initial awareness".into(), &args, &mut draft)
            .await
            .unwrap();

        assert!(matches!(outcome, GenerationStreamOutcome::Completed));
        assert_eq!(draft.accumulated, "Koro, stay close.");
        assert_eq!(
            deltas.borrow().as_slice(),
            &[
                ("Koro".to_string(), 0, 0, "Koro".to_string()),
                ("Koro".to_string(), 0, 1, ", stay close.".to_string()),
            ]
        );
        assert_eq!(
            session_turn_texts(&module.generation_session),
            vec!["Koro, stay close.".to_string()]
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn generation_reasoning_only_completion_does_not_persist_empty_turn() {
        let adapter =
            MockLlmAdapter::new().with_text_scenario(generation_reasoning_then_text_scenario(
                &["thinking about the utterance"],
                &[],
                FinishReason::Stop,
            ));
        let mut allocation = ResourceAllocation::default();
        allocation.set(builtin::speak(), ModuleConfig::default());
        allocation.set_activation(builtin::speak(), ActivationRatio::ONE);
        let blackboard = Blackboard::with_allocation(allocation);
        let completed = Rc::new(RefCell::new(Vec::new()));
        let sink: Rc<dyn crate::utterance::UtteranceSink> = Rc::new(CapturingUtteranceSink {
            completed: Rc::clone(&completed),
            done: RefCell::new(None),
        });
        let (mut module, _caps) =
            speak_module_with_turn_adapter(blackboard.clone(), Arc::new(adapter), sink).await;
        let now = SystemClock.now();
        let cx = test_activate_cx(&module, now).await;
        let mut draft = GenerationDraft::new(0, "Koro");
        let args = test_prepare_speech_args("Koro");

        let outcome = module
            .collect_generation_slice(&cx, "- initial awareness".into(), &args, &mut draft)
            .await
            .unwrap();

        assert!(matches!(outcome, GenerationStreamOutcome::NoVisibleOutput));
        assert_eq!(draft.accumulated, "");
        assert!(completed.borrow().is_empty());
        assert_eq!(
            session_turn_texts(&module.generation_session),
            Vec::<String>::new()
        );
        assert!(speak_memos(&blackboard).await.is_empty());
        assert!(!speak_progress_exists(&blackboard).await);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn reasoning_only_generation_activation_does_not_leave_active_or_empty_progress() {
        let adapter = MockLlmAdapter::new()
            .with_text_scenario(prepare_speech_scenario("Koro", "Tell Koro to stay close."))
            .with_text_scenario(generation_reasoning_then_text_scenario(
                &["thinking about the utterance"],
                &[],
                FinishReason::Stop,
            ));
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

        assert!(completed.borrow().is_empty());
        assert!(module.active_speech.is_none());
        assert_eq!(
            session_turn_texts(&module.generation_session),
            Vec::<String>::new()
        );
        assert!(speak_memos(&blackboard).await.is_empty());
        assert!(!speak_progress_exists(&blackboard).await);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn generation_error_after_partial_keeps_active_speech_in_progress() {
        let adapter = MockLlmAdapter::new()
            .with_text_scenario(prepare_speech_scenario("Koro", "Tell Koro to stay close."))
            .with_text_scenario(generation_text_then_error_scenario(&["Koro", ", st"]));
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

        assert!(completed.borrow().is_empty());
        let active = module
            .active_speech
            .as_ref()
            .expect("partial output should remain active for continuation");
        assert_eq!(active.draft.accumulated, "Koro, st");
        assert_eq!(
            session_turn_texts(&module.generation_session),
            Vec::<String>::new()
        );
        let progress = blackboard
            .read(|bb| {
                bb.utterance_progress_for_instance(&ModuleInstanceId::new(
                    builtin::speak(),
                    ReplicaIndex::ZERO,
                ))
                .cloned()
            })
            .await
            .unwrap();
        assert_eq!(
            progress.state,
            nuillu_blackboard::UtteranceProgressState::Streaming
        );
        assert_eq!(progress.partial_utterance, "Koro, st");
        let memos = speak_memos(&blackboard).await;
        assert_eq!(memos.len(), 1);
        assert!(memos[0].contains("Utterance in progress to Koro"));
        assert!(memos[0].contains("Already emitted:\nKoro, st"));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn text_sliced_generation_records_streaming_without_complete_emit() {
        let adapter = MockLlmAdapter::new()
            .with_text_scenario(prepare_speech_scenario("Koro", "Tell Koro to stay close."))
            .with_text_scenario(generation_text_sliced_scenario(&[
                "K", "o", "r", "o", ",", " ", "s", "t",
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
            speak_module_with_turn_adapter(blackboard.clone(), Arc::new(adapter), sink).await;
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

        assert!(completed.borrow().is_empty());
        let memos = speak_memos(&blackboard).await;
        assert_eq!(memos.len(), 1);
        assert!(memos[0].contains("Utterance in progress to Koro"));
        assert!(memos[0].contains("Continuation is pending"));
        assert!(memos[0].contains("Already emitted:\nKoro, st"));
        assert!(!memos[0].contains("Completed utterance"));
        assert!(module.active_speech.is_some());
        let speak_owner = ModuleInstanceId::new(builtin::speak(), ReplicaIndex::ZERO);
        let progress = blackboard
            .read(|bb| bb.utterance_progress_for_instance(&speak_owner).cloned())
            .await
            .unwrap();
        assert_eq!(
            progress.state,
            nuillu_blackboard::UtteranceProgressState::Streaming
        );
        assert_eq!(progress.target, "Koro");
        assert_eq!(progress.partial_utterance, "Koro, st");

        let planning_text = session_input_text(&module.planning_session);
        assert_eq!(module.planning_session.list_turns().count(), 1);
        assert!(!planning_text.contains("Outward speech currently in progress to Koro"));
        assert!(!planning_text.contains("Already emitted:\nKoro, st"));
        assert!(!planning_text.contains("Completed outward utterance to Koro"));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn text_sliced_speech_continues_next_batch_without_new_cognition() {
        let adapter = MockLlmAdapter::new()
            .with_text_scenario(prepare_speech_scenario("Koro", "Tell Koro to stay close."))
            .with_text_scenario(generation_text_sliced_scenario(&[
                "K", "o", "r", "o", ",", " ", "s", "t",
            ]))
            .with_text_scenario(generation_text_scenario("ay close."));
        let capture = CapturingAdapter::new(adapter);
        let observed = capture.clone();
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
        let first_batch = module.next_batch().await.unwrap();
        let cx = test_activate_cx(&module, now).await;
        SpeakModule::activate(&mut module, &cx, &first_batch)
            .await
            .unwrap();

        let second_batch = tokio::time::timeout(Duration::from_millis(100), module.next_batch())
            .await
            .expect("active speech continuation should not wait for a cognition update")
            .unwrap();
        assert!(second_batch.continue_active_speech);
        assert!(second_batch.updates.is_empty());
        assert!(second_batch.cognition_entries.is_empty());
        let cx = test_activate_cx(&module, now + chrono::Duration::seconds(1)).await;
        SpeakModule::activate(&mut module, &cx, &second_batch)
            .await
            .unwrap();

        assert_eq!(
            completed.borrow().as_slice(),
            &[("Koro".to_string(), "Koro, stay close.".to_string())]
        );
        assert!(module.active_speech.is_none());
        let speak_owner = ModuleInstanceId::new(builtin::speak(), ReplicaIndex::ZERO);
        let progress = blackboard
            .read(|bb| bb.utterance_progress_for_instance(&speak_owner).cloned())
            .await
            .unwrap();
        assert_eq!(
            progress.state,
            nuillu_blackboard::UtteranceProgressState::Completed
        );
        assert_eq!(progress.partial_utterance, "Koro, stay close.");
        assert_eq!(
            session_turn_texts(&module.generation_session),
            vec!["Koro, stay close.".to_string()]
        );

        let inputs = observed.text_inputs();
        assert_eq!(inputs.len(), 3);
        let generation_input = &inputs[2];
        let items = generation_input.items();
        assert!(items.len() >= 2);
        let tail = &items[items.len() - 2..];
        let ModelInputItem::Message { role, content } = &tail[0] else {
            panic!("expected continuation generation user context");
        };
        assert_eq!(role, &InputMessageRole::User);
        let user_text = content
            .as_slice()
            .iter()
            .filter_map(|content| match content {
                MessageContent::Text(text) => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n");
        assert!(user_text.contains("- none since the previous speech slice"));
        assert!(user_text.contains("Substance to express:\nTell Koro to stay close."));
        let ModelInputItem::Assistant(AssistantInputItem::Text(text)) = &tail[1] else {
            panic!("expected assistant partial utterance prefill");
        };
        assert_eq!(text, "Koro, st");
        assert_eq!(
            items
                .iter()
                .filter(|item| {
                    matches!(item, ModelInputItem::Assistant(AssistantInputItem::Text(text)) if text == "Koro, st")
                })
                .count(),
            1,
            "partial utterance should appear only as the continuation prefill"
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn repeated_generation_only_slices_do_not_commit_in_progress_planning_records() {
        let adapter = MockLlmAdapter::new()
            .with_text_scenario(prepare_speech_scenario("Koro", "Tell Koro to stay close."))
            .with_text_scenario(generation_text_sliced_scenario(&[
                "K", "o", "r", "o", ",", " ", "s", "t",
            ]))
            .with_text_scenario(generation_text_sliced_scenario(&[
                "a", "y", " ", "c", "l", "o", "s", "e",
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
            speak_module_with_turn_adapter(blackboard.clone(), Arc::new(adapter), sink).await;
        caps.scene().set([Participant::new("Koro")]);
        let now = SystemClock.now();
        publish_cognition_update(
            &blackboard,
            &caps,
            now,
            "Koro asks Nuillu to help them stay safe.",
        )
        .await;

        let first_batch = module.next_batch().await.unwrap();
        let cx = test_activate_cx(&module, now).await;
        SpeakModule::activate(&mut module, &cx, &first_batch)
            .await
            .unwrap();
        let second_batch = tokio::time::timeout(Duration::from_millis(100), module.next_batch())
            .await
            .expect("active speech continuation should not wait for a cognition update")
            .unwrap();
        assert!(second_batch.continue_active_speech);
        assert!(second_batch.updates.is_empty());
        assert!(second_batch.cognition_entries.is_empty());
        let cx = test_activate_cx(&module, now + chrono::Duration::seconds(1)).await;
        SpeakModule::activate(&mut module, &cx, &second_batch)
            .await
            .unwrap();

        assert!(completed.borrow().is_empty());
        let active = module
            .active_speech
            .as_ref()
            .expect("generation should still be active after the second slice");
        assert_eq!(active.draft.accumulated, "Koro, stay close");
        assert_eq!(active.draft.sequence, 16);
        let speak_owner = ModuleInstanceId::new(builtin::speak(), ReplicaIndex::ZERO);
        let progress = blackboard
            .read(|bb| bb.utterance_progress_for_instance(&speak_owner).cloned())
            .await
            .unwrap();
        assert_eq!(
            progress.state,
            nuillu_blackboard::UtteranceProgressState::Streaming
        );
        assert_eq!(progress.partial_utterance, "Koro, stay close");

        let planning_text = session_input_text(&module.planning_session);
        assert_eq!(module.planning_session.list_turns().count(), 1);
        assert!(!planning_text.contains("Outward speech currently in progress to Koro"));
        assert!(!planning_text.contains("Already emitted:\nKoro, st"));
        assert!(!planning_text.contains("Already emitted:\nKoro, stay close"));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn active_speech_continue_preserves_target_and_prefills_partial_japanese_greeting() {
        let partial = "Aliceの挨拶には、親しみを";
        let adapter = MockLlmAdapter::new()
            .with_text_scenario(prepare_speech_scenario(
                "everyone",
                "Aliceの挨拶に親しみを込めて短く応える。",
            ))
            .with_text_scenario(generation_text_sliced_scenario(&[
                "Alice",
                "の",
                "挨",
                "拶",
                "に",
                "は",
                "、",
                "親しみを",
            ]))
            .with_text_scenario(continue_speech_scenario(
                "Aliceの挨拶に親しみを込めて短く応える。",
            ))
            .with_text_scenario(generation_text_scenario("込めて、こんにちは。"));
        let capture = CapturingAdapter::new(adapter);
        let observed = capture.clone();
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
            speak_module_with_turn_adapter(blackboard.clone(), Arc::new(capture), sink).await;
        caps.scene().set([Participant::new("Alice")]);
        let now = SystemClock.now();
        publish_cognition_update(&blackboard, &caps, now, "Alice greets Nuillu in Japanese.").await;
        let first_batch = module.next_batch().await.unwrap();
        let cx = test_activate_cx(&module, now).await;
        SpeakModule::activate(&mut module, &cx, &first_batch)
            .await
            .unwrap();

        publish_cognition_update(
            &blackboard,
            &caps,
            now + chrono::Duration::seconds(1),
            "Alice is still present and waiting for the greeting reply.",
        )
        .await;
        let second_batch = module.next_batch().await.unwrap();
        let cx = test_activate_cx(&module, now + chrono::Duration::seconds(1)).await;
        SpeakModule::activate(&mut module, &cx, &second_batch)
            .await
            .unwrap();

        assert_eq!(
            completed.borrow().as_slice(),
            &[(
                "everyone".to_string(),
                "Aliceの挨拶には、親しみを込めて、こんにちは。".to_string(),
            )]
        );
        let inputs = observed.text_inputs();
        assert_eq!(inputs.len(), 4);
        let generation_input = &inputs[3];
        let items = generation_input.items();
        assert!(items.len() >= 2);
        let tail = &items[items.len() - 2..];
        let ModelInputItem::Message { role, content } = &tail[0] else {
            panic!("expected continuation generation user context");
        };
        assert_eq!(role, &InputMessageRole::User);
        let user_text = content
            .as_slice()
            .iter()
            .filter_map(|content| match content {
                MessageContent::Text(text) => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n");
        assert!(user_text.contains("Speak to: everyone"));
        assert!(
            user_text.contains("Substance to express:\nAliceの挨拶に親しみを込めて短く応える。")
        );
        let ModelInputItem::Assistant(AssistantInputItem::Text(text)) = &tail[1] else {
            panic!("expected assistant partial utterance prefill");
        };
        assert_eq!(text, partial);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn text_sliced_speech_replans_with_late_cognition_and_prefills_generation() {
        let adapter = MockLlmAdapter::new()
            .with_text_scenario(prepare_speech_scenario("Koro", "Tell Koro to duck soon."))
            .with_text_scenario(generation_text_sliced_scenario(&[
                "K", "o", "r", "o", ",", " ", "d", "u",
            ]))
            .with_text_scenario(continue_speech_scenario("Tell Koro to duck now."))
            .with_text_scenario(generation_text_scenario("ck now."));
        let capture = CapturingAdapter::new(adapter);
        let observed = capture.clone();
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
        let first_batch = module.next_batch().await.unwrap();
        let cx = test_activate_cx(&module, now).await;
        SpeakModule::activate(&mut module, &cx, &first_batch)
            .await
            .unwrap();

        publish_cognition_update(
            &blackboard,
            &caps,
            now + chrono::Duration::seconds(1),
            "Koro sees a falling rock and now needs to duck.",
        )
        .await;
        let second_batch = module.next_batch().await.unwrap();
        let cx = test_activate_cx(&module, now + chrono::Duration::seconds(1)).await;
        SpeakModule::activate(&mut module, &cx, &second_batch)
            .await
            .unwrap();

        assert_eq!(
            completed.borrow().as_slice(),
            &[("Koro".to_string(), "Koro, duck now.".to_string())]
        );
        let planning_text = session_input_text(&module.planning_session);
        assert!(planning_text.contains("Already emitted: Koro, du"));
        assert!(planning_text.contains("Koro sees a falling rock"));

        let inputs = observed.text_inputs();
        assert_eq!(inputs.len(), 4);
        let generation_input = &inputs[3];
        let items = generation_input.items();
        assert!(items.len() >= 2);
        let tail = &items[items.len() - 2..];
        let ModelInputItem::Message { role, content } = &tail[0] else {
            panic!("expected continuation generation user plan");
        };
        assert_eq!(role, &InputMessageRole::User);
        let user_text = content
            .as_slice()
            .iter()
            .filter_map(|content| match content {
                MessageContent::Text(text) => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n");
        assert!(user_text.contains("Substance to express:\nTell Koro to duck now."));
        let ModelInputItem::Assistant(AssistantInputItem::Text(text)) = &tail[1] else {
            panic!("expected assistant partial utterance prefill");
        };
        assert_eq!(text, "Koro, du");
    }

    #[tokio::test(flavor = "current_thread")]
    async fn active_speech_retarget_drops_partial_and_generates_new_target_once() {
        let adapter = MockLlmAdapter::new()
            .with_text_scenario(prepare_speech_scenario("Koro", "Tell Koro to stay close."))
            .with_text_scenario(generation_text_sliced_scenario(&[
                "K", "o", "r", "o", ",", " ", "s", "t",
            ]))
            .with_text_scenario(interrupt_and_redirect_speech_scenario(
                "OffstageVoice",
                "Tell the offstage voice to stop causing trouble.",
                "The offstage voice is causing immediate trouble.",
            ))
            .with_text_scenario(generation_text_scenario(
                "Offstage voice, stop causing trouble.",
            ));
        let capture = CapturingAdapter::new(adapter);
        let observed = capture.clone();
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
        let first_batch = module.next_batch().await.unwrap();
        let cx = test_activate_cx(&module, now).await;
        SpeakModule::activate(&mut module, &cx, &first_batch)
            .await
            .unwrap();
        assert!(module.active_speech.is_some());

        publish_cognition_update(
            &blackboard,
            &caps,
            now + chrono::Duration::seconds(1),
            "An offstage voice is causing immediate trouble and needs a warning.",
        )
        .await;
        let second_batch = module.next_batch().await.unwrap();
        let cx = test_activate_cx(&module, now + chrono::Duration::seconds(1)).await;
        SpeakModule::activate(&mut module, &cx, &second_batch)
            .await
            .unwrap();

        assert_eq!(
            completed.borrow().as_slice(),
            &[(
                "OffstageVoice".to_string(),
                "Offstage voice, stop causing trouble.".to_string(),
            )]
        );
        assert!(module.active_speech.is_none());
        let speak_owner = ModuleInstanceId::new(builtin::speak(), ReplicaIndex::ZERO);
        let progress = blackboard
            .read(|bb| bb.utterance_progress_for_instance(&speak_owner).cloned())
            .await
            .unwrap();
        assert_eq!(
            progress.state,
            nuillu_blackboard::UtteranceProgressState::Completed
        );
        assert_eq!(progress.target, "OffstageVoice");
        assert_eq!(
            progress.partial_utterance,
            "Offstage voice, stop causing trouble."
        );
        assert_eq!(observed.text_turns().len(), 4);
        let planning_text = session_input_text(&module.planning_session);
        assert!(!planning_text.contains("Completed outward utterance to Koro:\nKoro, st"));
        assert!(planning_text.contains("Aborted outward speech to Koro"));
        assert!(planning_text.contains("Already emitted:\nKoro, st"));
        assert!(planning_text.contains("An offstage voice is causing immediate trouble"));
        let memos = speak_memos(&blackboard).await;
        assert!(
            memos
                .iter()
                .any(|memo| memo.contains("Aborted utterance to Koro"))
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn active_speech_decline_stops_partial_without_completion() {
        let adapter = MockLlmAdapter::new()
            .with_text_scenario(prepare_speech_scenario("Koro", "Tell Koro to stay close."))
            .with_text_scenario(generation_text_sliced_scenario(&[
                "K", "o", "r", "o", ",", " ", "s", "t",
            ]))
            .with_text_scenario(abort_speech_scenario("speech is no longer needed"));
        let capture = CapturingAdapter::new(adapter);
        let observed = capture.clone();
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
        let first_batch = module.next_batch().await.unwrap();
        let cx = test_activate_cx(&module, now).await;
        SpeakModule::activate(&mut module, &cx, &first_batch)
            .await
            .unwrap();

        publish_cognition_update(
            &blackboard,
            &caps,
            now + chrono::Duration::seconds(1),
            "Koro no longer needs the warning.",
        )
        .await;
        let second_batch = module.next_batch().await.unwrap();
        let cx = test_activate_cx(&module, now + chrono::Duration::seconds(1)).await;
        SpeakModule::activate(&mut module, &cx, &second_batch)
            .await
            .unwrap();

        assert!(completed.borrow().is_empty());
        assert!(module.active_speech.is_none());
        assert_eq!(observed.text_turns().len(), 3);
        let speak_owner = ModuleInstanceId::new(builtin::speak(), ReplicaIndex::ZERO);
        let progress = blackboard
            .read(|bb| bb.utterance_progress_for_instance(&speak_owner).cloned())
            .await
            .unwrap();
        assert_eq!(
            progress.state,
            nuillu_blackboard::UtteranceProgressState::Aborted
        );
        assert_eq!(progress.partial_utterance, "Koro, st");
        let memos = speak_memos(&blackboard).await;
        assert!(
            memos
                .iter()
                .any(|memo| memo.contains("Aborted utterance to Koro"))
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
        assert!(memos[0].contains("Declined outward speech. Reason:"));
        assert!(memos[0].contains("no supported listener-facing content"));
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
            .with_text_scenario(prepare_speech_scenario("   ", "Tell Koro to stay close."));

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
    async fn speak_generates_for_non_scene_planned_target() {
        let adapter = MockLlmAdapter::new()
            .with_text_scenario(prepare_speech_scenario(
                "OffstageVoice",
                "Tell the offstage voice to stop causing trouble.",
            ))
            .with_text_scenario(generation_text_scenario(
                "Offstage voice, stop causing trouble.",
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
                "Offstage voice, stop causing trouble.".to_string(),
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
        assert_eq!(items.len(), 2);
        assert!(matches!(
            &items[0],
            ModelInputItem::Message {
                role: InputMessageRole::User,
                ..
            }
        ));
        let ModelInputItem::Assistant(AssistantInputItem::Text(text)) = &items[1] else {
            panic!("expected assistant prefill");
        };
        assert_eq!(text, "hello world");
    }

    #[test]
    fn speak_batch_keeps_drained_updates() {
        let batch = SpeakBatch {
            updates: vec![CognitionLogUpdated::AgenticDeadlockMarker],
            cognition_entries: Vec::new(),
            continue_active_speech: false,
        };

        assert_eq!(
            batch.updates,
            vec![CognitionLogUpdated::AgenticDeadlockMarker]
        );
    }
}
