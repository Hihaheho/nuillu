use std::borrow::Cow;
use std::collections::HashSet;
use std::rc::Rc;
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result};
use async_trait::async_trait;
use chrono::{DateTime, SecondsFormat, Utc};
use futures::FutureExt as _;
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
    AttentionControlRequest, AttentionControlRequestMailbox, CognitionLogReader,
    CognitionLogUpdated, CognitionLogUpdatedInbox, LlmAccess, LlmContextWindow, Memo, Module,
    SceneReader, SelfWake, SessionAutoCompaction, SessionCompactionConfig,
    SessionCompactionProtectedPrefix, UtteranceProgress, compact_llm_context_text,
    ensure_persistent_session_seeded, format_new_cognition_log_entries, ports::Clock,
};

use crate::utterance::UtteranceWriter;
use nuillu_types::builtin;
use schemars::{JsonSchema, Schema, SchemaGenerator};
use serde::{Deserialize, Serialize};

const SPEECH_PLANNING_PROMPT: &str = r#"Plan outward speech from the current cognition log and scene target hints.
Use exactly one available tool.
When no speech is in progress, call prepare_speech only when new cognition supports a new outward utterance. target is the person or group who should hear it.
For direct speech heard from a named speaker, target that speaker. Use everyone only when the cognition explicitly calls for group or broadcast speech.
When no speech is in progress, call decline_speech_now when no new outward utterance is appropriate now. If speak should not be prioritized again until new cognition arrives, include inhibit_reason.
Already emitted text is visible to the listener. Treat it as something the listener has already heard.
When speech is already in progress, decide what text should be appended next.
If the next text should continue to the same listener, call continue_speech. Use it for ordinary continuation, correction, topic shift, recovery from repetition, or graceful ending.
In append_directive, give concrete instructions for the next appended text only: how it should connect from the already emitted text, what should now be conveyed, what should no longer be said if the current partial became stale or wrong, and any exact phrase that should not be repeated. Do not write a replacement utterance there.
Do not tell generation to ignore or restart after already visible text. If the partial is broken or repetitive, tell generation how to make the next words smooth it over for the listener.
Use redirect_speech only when the next words should be addressed to a different listener.
Predictions, expected dialogue flow, and my own previous speech are not new outward speech motivation by themselves.
Do not plan speech that repeats content the listener has already heard; only refer back to prior speech when it is needed to correct, finish, or smoothly bridge from it.
utterance_directive is the concrete instruction for rendering a fresh listener-facing utterance, grounded in identity memory or the cognition log. Nui is my own name; do not treat my name as the listener. Do not write the target's future reply, expression, feeling, action, or narration there.
If the cognition log contains an explicit language request, set language and shape utterance_directive or append_directive for that language.
Do not invent policy, actions, identity, memory, visible evidence, unknown-state evidence, or other facts not supported by the provided context."#;

const FRESH_PLANNING_TURN_DEVELOPER_INSTRUCTION: &str = "Use exactly one tool: prepare_speech for a new grounded outward utterance, or decline_speech_now when no new outward utterance is appropriate now.";
const ACTIVE_PLANNING_TURN_DEVELOPER_INSTRUCTION: &str = "Use exactly one tool: continue_speech when the next visible text should append to the same listener, or redirect_speech only when the next visible text should address a different listener. For continue_speech, fill append_directive with instructions for the next appended text only, not a replacement utterance.";

const GENERATION_PROMPT: &str = r#"Write one concise in-world utterance.
Use Recent context, Planner Directive, and Append Directive only; add no unsupported facts.
If a language is supplied, render the utterance in that language.
Do not address Me/Nui as the listener.
If Planner Directive is already a usable fresh utterance and no already emitted text is supplied, say that utterance directly.
If Append Directive and Already emitted are supplied, append after Already emitted only.
Output only the utterance text; do not wrap it in quotation marks.
If no appropriate fresh utterance should be completed now, output nothing; empty output stops speech and emits no user-visible utterance.
Do not summarize the request or say what the user wants.
Do not output introspection, narration, or analysis that is not appropriate to say aloud to the target.
Do not mention implementation mechanics, lookup, reasoning, prompts, rubrics, or evaluation mechanics.
Already emitted text is visible to the listener and cannot be erased. For correction, topic shift, recovery from repetition, or graceful ending, append listener-facing words that make the visible partial land naturally. Do not repeat any phrase the directive says to avoid.

Continuation example:
User:
Append the next visible text to the in-progress utterance for `Ryo`.
Language: Japanese
Append Directive:
The already emitted text says hello; finish the fun question.
The output is appended directly after Already emitted. Emit only the next new text. Do not repeat, restate, or include any already emitted text.

Already emitted:
こんにちは！Ryoさんは、何か楽しい
Assistant:
ことを準備していますか？"#;
const GENERATION_TURN_USER_PROMPT_PREFIX: &str = "Generate an utterance to";
const GENERATION_APPEND_USER_PROMPT_PREFIX: &str =
    "Append the next visible text to the in-progress utterance for";
const GENERATION_APPEND_INSTRUCTION: &str = "The output is appended directly after Already emitted. Emit only the next new text. Do not repeat, restate, or include any already emitted text.";

const COGNITION_CONTEXT_WINDOW: LlmContextWindow = LlmContextWindow::new(12, 600, 4_800);
const SPEECH_PLANNING_TURN_MAX_OUTPUT_TOKENS: u32 = 1024;
const SPEECH_GENERATION_TEXT_DELTA_SLICE_LIMIT: usize = 16;
const SPEECH_GENERATION_TEXT_DELTA_CHAR_LIMIT: usize = 20;
const SPEECH_GENERATION_SLICES_PER_PLAN: u8 = 3;
const ACTIVE_SPEECH_PLANNING_ALREADY_EMITTED_CHARS: usize = 600;
const THINK_OPEN_TAG: &str = "<think>";
const THINK_CLOSE_TAG: &str = "</think>";
const COMPACTED_SPEAK_PLANNING_SESSION_PREFIX: &str = "Compacted speak planning session history:";
const COMPACTED_SPEAK_GENERATION_SESSION_PREFIX: &str =
    "Compacted speak generation session history:";
const PLANNING_SESSION_COMPACTION_FOCUS: &str = r#"Preserve prior speech target decisions,
selected targets, rejected/no-speech decisions, completed outward utterances, in-progress
speech continuations, and cognition-log context needed for future speak planning."#;
const GENERATION_SESSION_COMPACTION_FOCUS: &str = r#"Preserve completed outward utterances,
their addressees, and any Planner Directive or Append Directive context needed to understand those utterances."#;

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
/// Call when proceeding to an outward utterance.
struct PrepareSpeechArgs {
    /// The concrete addressee or audience who should hear the utterance.
    target: SpeechTarget,
    /// Requested output language when the cognition log contains an explicit
    /// language request.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    language: Option<String>,
    /// What I should convey outwardly. Do not describe the target's future
    /// reply, expression, feeling, action, or narration here.
    utterance_directive: String,
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

#[lutum::tool_input(name = "continue_speech", output = ContinueSpeechOutput)]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
/// Call when an in-progress utterance should continue to the same target.
struct ContinueSpeechArgs {
    /// Concrete instructions for only the next visible text appended after the
    /// already emitted partial utterance. This is not a replacement utterance.
    append_directive: String,
    /// Requested output language when the continuation should preserve or apply
    /// an explicit language request.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    language: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
struct ContinueSpeechOutput {
    accepted: bool,
}

#[lutum::tool_input(name = "redirect_speech", output = RedirectSpeechOutput)]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
/// Call only when the next visible text should address a different target.
struct RedirectSpeechArgs {
    /// The participant who should immediately hear the redirected utterance.
    target: SpeechTarget,
    /// Requested output language when the redirected speech should preserve or
    /// apply an explicit language request.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    language: Option<String>,
    /// What I should convey outwardly to `target`.
    utterance_directive: String,
    /// Why the next visible text should address this different target.
    redirect_reason: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
struct RedirectSpeechOutput {
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
    RedirectSpeech(RedirectSpeechArgs),
}

#[derive(Clone, Debug)]
struct PlannedSpeech {
    args: PrepareSpeechArgs,
    target: String,
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

#[derive(Clone, Debug)]
enum ActiveSpeechPlan {
    Continue(PlannedSpeech),
    Redirect {
        plan: PlannedSpeech,
        redirect_reason: String,
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
    generation_slices_since_plan: u8,
}

#[derive(Default)]
struct GenerationDeltaBuffer {
    deltas: Vec<String>,
    visible_text_delta_count: usize,
    visible_text_char_count: usize,
    limit_reached: bool,
    think_filter: ThinkTagFilter,
}

impl GenerationDeltaBuffer {
    fn push(&mut self, delta: String) -> Option<Vec<String>> {
        if self.limit_reached {
            return None;
        }
        let Some(delta) = self.think_filter.push(&delta) else {
            return None;
        };
        self.push_visible(delta);
        self.drain_ready_batch()
    }

    fn finish(&mut self) -> Option<Vec<String>> {
        if self.limit_reached {
            return None;
        }
        if let Some(delta) = self.think_filter.finish() {
            self.push_visible(delta);
        }
        self.drain_pending_batch()
    }

    fn limit_reached(&self) -> bool {
        self.limit_reached
    }

    fn push_visible(&mut self, delta: String) {
        if delta.is_empty() {
            self.deltas.push(delta);
            return;
        }
        if self.visible_text_delta_count >= SPEECH_GENERATION_TEXT_DELTA_SLICE_LIMIT {
            self.limit_reached = true;
            return;
        }
        let remaining_chars =
            SPEECH_GENERATION_TEXT_DELTA_CHAR_LIMIT.saturating_sub(self.visible_text_char_count);
        let (delta, char_count, truncated) = truncate_generation_text_delta(delta, remaining_chars);
        self.visible_text_delta_count += 1;
        self.visible_text_char_count += char_count;
        self.deltas.push(delta);
        if truncated || self.visible_text_char_count >= SPEECH_GENERATION_TEXT_DELTA_CHAR_LIMIT {
            self.limit_reached = true;
        }
    }

    fn drain_ready_batch(&mut self) -> Option<Vec<String>> {
        if self.limit_reached
            || self.visible_text_delta_count >= SPEECH_GENERATION_TEXT_DELTA_SLICE_LIMIT
            || self.visible_text_char_count >= SPEECH_GENERATION_TEXT_DELTA_CHAR_LIMIT
        {
            self.limit_reached = true;
            return self.drain_pending_batch();
        }
        None
    }

    fn drain_pending_batch(&mut self) -> Option<Vec<String>> {
        if self.deltas.is_empty() {
            return None;
        }
        Some(std::mem::take(&mut self.deltas))
    }
}

fn truncate_generation_text_delta(delta: String, max_chars: usize) -> (String, usize, bool) {
    if max_chars == 0 {
        return (String::new(), 0, !delta.is_empty());
    };
    let Some((split_at, _)) = delta.char_indices().nth(max_chars) else {
        let char_count = delta.chars().count();
        return (delta, char_count, false);
    };
    (delta[..split_at].to_owned(), max_chars, true)
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
    flushed_deltas: tokio::sync::mpsc::UnboundedSender<Vec<String>>,
}

#[async_trait]
impl EventHandler<TextTurnEvent, TextTurnState> for GenerationDeltaCollector {
    async fn on_event(
        &mut self,
        event: &TextTurnEvent,
        _cx: &HandlerContext<TextTurnState>,
    ) -> HandlerResult {
        if let TextTurnEvent::TextDelta { delta } = event {
            let ready = {
                let mut buffered = self
                    .buffered_deltas
                    .lock()
                    .expect("generation delta buffer mutex poisoned");
                buffered.push(delta.clone())
            };
            if let Some(deltas) = ready {
                let _ = self.flushed_deltas.send(deltas);
                tokio::task::yield_now().await;
                return Ok(HandlerDirective::Stop);
            }
        }
        Ok(HandlerDirective::Continue)
    }
}

fn render_completed_utterance_memo(draft: &GenerationDraft, text: &str) -> String {
    format!("I said to {}:\n{}", draft.target.trim(), text.trim(),)
}

fn render_in_progress_utterance_memo(args: &PrepareSpeechArgs, draft: &GenerationDraft) -> String {
    format!(
        "I am speaking to {}.\nutterance_directive:\n{}\n\nAlready said:\n{}",
        draft.target.trim(),
        args.utterance_directive.trim(),
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
    format!("I am staying silent for now. Reason:\n{}", reason.trim())
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

fn format_generation_turn_user_prompt(args: &PrepareSpeechArgs, draft: &GenerationDraft) -> String {
    let active_append = !draft.accumulated.is_empty();
    let mut out = if active_append {
        format!(
            "{GENERATION_APPEND_USER_PROMPT_PREFIX} `{}`.",
            draft.target.trim()
        )
    } else {
        format!(
            "{GENERATION_TURN_USER_PROMPT_PREFIX} `{}`.",
            draft.target.trim()
        )
    };
    if let Some(language) = trimmed_optional(args.language.as_deref()) {
        out.push_str(&format!("\nLanguage: {language}"));
    }
    let directive_label = if active_append {
        "Append Directive"
    } else {
        "Planner Directive"
    };
    out.push_str(&format!(
        "\n{directive_label}:\n{}",
        args.utterance_directive.trim(),
    ));
    out
}

fn format_generation_llm_user_prompt(args: &PrepareSpeechArgs, draft: &GenerationDraft) -> String {
    let mut out = format_generation_turn_user_prompt(args, draft);
    if !draft.accumulated.is_empty() {
        out.push_str("\n\n");
        out.push_str(GENERATION_APPEND_INSTRUCTION);
        out.push_str("\n\nAlready emitted:\n");
        out.push_str(&draft.accumulated);
    }
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
            "\n- Planned utterance_directive: {}",
            active.args.utterance_directive.trim()
        ));
        out.push_str(&format!(
            "\n- Already emitted: {}",
            if active.draft.accumulated.trim().is_empty() {
                Cow::Borrowed("(none)")
            } else {
                Cow::Owned(compact_llm_context_text(
                    active.draft.accumulated.trim(),
                    ACTIVE_SPEECH_PLANNING_ALREADY_EMITTED_CHARS,
                ))
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

#[derive(Clone, Debug, PartialEq, Eq)]
struct GenerationTurnContext {
    base_user_prompt: String,
}

fn push_generation_context(
    session: &mut Session,
    args: &PrepareSpeechArgs,
    draft: &GenerationDraft,
) -> GenerationTurnContext {
    let base_user_prompt = format_generation_turn_user_prompt(args, draft);
    let llm_user_prompt = format_generation_llm_user_prompt(args, draft);
    session.push_ephemeral_user(llm_user_prompt);
    GenerationTurnContext { base_user_prompt }
}

fn push_completed_generation_turn(
    session: &mut Session,
    context: &GenerationTurnContext,
    text: &str,
) {
    session.push_user(context.base_user_prompt.clone());
    session.input_mut().push(ModelInputItem::turn(Arc::new(
        AssistantTurnView::from_items(&[AssistantTurnItem::Text(text.to_owned())]),
    )));
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

fn generation_cognition_context_from_entries(
    records: &[CognitionLogEntryRecord],
) -> Option<String> {
    let mut records = speech_planning_cognition_records(records);
    records.sort_by_key(|record| (record.entry.at, record.index));
    if records.is_empty() || COGNITION_CONTEXT_WINDOW.max_records == 0 {
        return None;
    }

    let omitted_for_record_limit = records
        .len()
        .saturating_sub(COGNITION_CONTEXT_WINDOW.max_records);
    if omitted_for_record_limit > 0 {
        tracing::warn!(
            target: "nuillu_speak::llm_context",
            context_kind = "generation_recent_context",
            original_records = records.len(),
            kept_records = COGNITION_CONTEXT_WINDOW.max_records,
            dropped_records = omitted_for_record_limit,
            "bounded LLM context dropped older records"
        );
    }
    let selected = &records[omitted_for_record_limit..];
    let mut lines = selected
        .iter()
        .filter_map(|record| {
            let text = compact_llm_context_text(
                &record.entry.text,
                COGNITION_CONTEXT_WINDOW.max_chars_per_record,
            );
            (!text.is_empty()).then(|| format!("- {text}"))
        })
        .collect::<Vec<_>>();
    if lines.is_empty() {
        return None;
    }

    while lines.join("\n").chars().count() > COGNITION_CONTEXT_WINDOW.max_total_chars
        && lines.len() > 1
    {
        lines.remove(0);
    }
    let context = compact_multiline_evidence(lines.join("\n"));
    (!context.is_empty()).then(|| format!("Recent context:\n{context}"))
}

fn compact_multiline_evidence(text: String) -> String {
    if text.chars().count() <= COGNITION_CONTEXT_WINDOW.max_total_chars {
        return text;
    }
    let mut out = text
        .chars()
        .take(COGNITION_CONTEXT_WINDOW.max_total_chars)
        .collect::<String>();
    if let Some((boundary, _)) = out.char_indices().rev().find(|(_, ch)| *ch == '\n') {
        out.truncate(boundary);
    }
    out.trim_end().to_owned()
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

fn active_speech_cognition_context_from_entries(
    records: &[CognitionLogEntryRecord],
    now: DateTime<Utc>,
) -> String {
    speech_cognition_context_from_entries(records, now).unwrap_or_else(|| {
        format!(
            "New thoughts available to you at {}:\n- none since the previous speech slice",
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
    attention_control: AttentionControlRequestMailbox,
    memo: Memo,
    utterance: UtteranceWriter,
    planning_llm: LlmAccess,
    generation_llm: LlmAccess,
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
    pub attention_control: AttentionControlRequestMailbox,
    pub memo: Memo,
    pub utterance: UtteranceWriter,
    pub planning_llm: LlmAccess,
    pub generation_llm: LlmAccess,
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
            attention_control,
            memo,
            utterance,
            planning_llm,
            generation_llm,
            scene,
            clock,
            planning_session,
            generation_session,
            self_wake,
        } = parts;

        Self {
            cognition_updates,
            cognition_log,
            attention_control,
            memo,
            utterance,
            planning_llm,
            generation_llm,
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
            nuillu_module::format_policy_system_prompt(SPEECH_PLANNING_PROMPT, cx.core_policies())
        })
    }

    fn generation_prompt(&self, cx: &nuillu_module::ActivateCx<'_>) -> &str {
        self.generation_prompt.get_or_init(|| {
            nuillu_module::format_policy_system_prompt(GENERATION_PROMPT, cx.core_policies())
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
        let recent_generation_context =
            generation_cognition_context_from_entries(&batch.cognition_entries);
        self.append_generation_cognition_context(cx, recent_generation_context.as_deref())
            .await?;
        let (plan, mut draft, generation_slices_since_plan) = if let Some(active) =
            self.active_speech.take()
        {
            let planning_context =
                active_speech_cognition_context_from_entries(&batch.cognition_entries, now);
            let should_continue_without_planning = batch.continue_active_speech
                && batch.updates.is_empty()
                && batch.cognition_entries.is_empty()
                && active.generation_slices_since_plan < SPEECH_GENERATION_SLICES_PER_PLAN;
            if should_continue_without_planning {
                let generation_slices_since_plan = active.generation_slices_since_plan;
                let plan = PlannedSpeech {
                    args: active.args,
                    target: active.draft.target.clone(),
                };
                (plan, active.draft, generation_slices_since_plan)
            } else {
                let Some(active_plan) = self
                    .plan_active_speech(cx, &planning_context, &active)
                    .await?
                else {
                    self.mark_generation_cognition_entries_reflected(&batch.cognition_entries);
                    self.active_speech = Some(ActiveSpeech {
                        generation_slices_since_plan: 0,
                        ..active
                    });
                    self.self_wake.wake();
                    return Ok(());
                };
                match active_plan {
                    ActiveSpeechPlan::Continue(plan) => (plan, active.draft, 0),
                    ActiveSpeechPlan::Redirect {
                        plan,
                        redirect_reason,
                    } => {
                        self.record_aborted_generation(cx, &active.draft, &redirect_reason)
                            .await?;
                        let draft =
                            GenerationDraft::new(self.utterance.next_generation_id(), &plan.target);
                        (plan, draft, 0)
                    }
                }
            }
        } else {
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
                    self.mark_generation_cognition_entries_reflected(&batch.cognition_entries);
                    return Ok(());
                }
                FreshSpeechPlan::None => {
                    self.mark_generation_cognition_entries_reflected(&batch.cognition_entries);
                    return Ok(());
                }
            };
            let draft = GenerationDraft::new(self.utterance.next_generation_id(), &plan.target);
            (plan, draft, 0)
        };

        match self
            .collect_generation_slice(
                cx,
                &plan.args,
                &mut draft,
                recent_generation_context.as_deref(),
            )
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
                    generation_slices_since_plan: generation_slices_since_plan.saturating_add(1),
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
        let mut turn_session = self.planning_session.clone();
        push_planning_context(&mut turn_session, cognition_context, None, &target_hints);

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

    async fn plan_active_speech(
        &mut self,
        cx: &nuillu_module::ActivateCx<'_>,
        cognition_context: &str,
        active: &ActiveSpeech,
    ) -> Result<Option<ActiveSpeechPlan>> {
        self.ensure_planning_session_seeded(cx);
        let scene_target_schema = self.scene.target_schema();
        let target_hints = target_hints_from_schema(&scene_target_schema);
        let mut turn_session = self.planning_session.clone();
        push_planning_context(
            &mut turn_session,
            cognition_context,
            Some(active),
            &target_hints,
        );

        let lutum = self.planning_llm.lutum().await;
        let target_schema = freeform_speech_target_schema();
        let outcome = SPEECH_TARGET_SCHEMA
            .scope(target_schema, async {
                turn_session
                    .text_turn()
                    .tools::<ActiveSpeakTools>()
                    .available_tools([
                        ActiveSpeakToolsSelector::ContinueSpeech,
                        ActiveSpeakToolsSelector::RedirectSpeech,
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
            TextStepOutcomeWithTools::Finished(_result) => {
                let detail = "model finished with assistant output but no tool call \
                    (require_any_tool should have prevented this outcome)";
                cx.warn(format!("active speak planning failed: {detail}"));
                anyhow::bail!(
                    "active speak planning finished without required tool call: {detail}"
                );
            }
            TextStepOutcomeWithTools::FinishedNoOutput(_result) => {
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
                        expected continue_speech or redirect_speech";
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
                                        utterance_directive: call.input.append_directive.clone(),
                                    },
                                    target,
                                }));
                            }
                            results.push(
                                call.complete(ContinueSpeechOutput { accepted })
                                    .context("complete continue_speech tool call")?,
                            );
                        }
                        ActiveSpeakToolsCall::RedirectSpeech(call) => {
                            let target = call.input.target.as_str().trim().to_owned();
                            let accepted = selected.is_none() && is_non_empty_target(&target);
                            if accepted {
                                selected = Some(ActiveSpeechPlan::Redirect {
                                    plan: PlannedSpeech {
                                        args: PrepareSpeechArgs {
                                            target: call.input.target.clone(),
                                            language: call.input.language.clone(),
                                            utterance_directive: call
                                                .input
                                                .utterance_directive
                                                .clone(),
                                        },
                                        target,
                                    },
                                    redirect_reason: call.input.redirect_reason.clone(),
                                });
                            }
                            results.push(
                                call.complete(RedirectSpeechOutput { accepted })
                                    .context("complete redirect_speech tool call")?,
                            );
                        }
                    }
                }
                round
                    .commit(&mut turn_session, results)
                    .context("commit active speak planning tool round")?;
                cx.compact_and_save(&mut turn_session, usage).await?;
                self.planning_session = turn_session;
                Ok(selected)
            }
        }
    }

    async fn append_generation_cognition_context(
        &mut self,
        cx: &nuillu_module::ActivateCx<'_>,
        context: Option<&str>,
    ) -> Result<()> {
        let Some(context) = context else {
            return Ok(());
        };
        self.ensure_generation_session_seeded(cx);
        self.generation_session.push_user(context);
        cx.compact_and_save(&mut self.generation_session, Usage::zero())
            .await?;
        Ok(())
    }

    async fn collect_generation_slice(
        &mut self,
        cx: &nuillu_module::ActivateCx<'_>,
        args: &PrepareSpeechArgs,
        draft: &mut GenerationDraft,
        recent_generation_context: Option<&str>,
    ) -> Result<GenerationStreamOutcome> {
        self.ensure_generation_session_seeded(cx);
        let active_append = !draft.accumulated.is_empty();
        let mut turn_session = if active_append {
            let mut session = Session::new();
            ensure_persistent_session_seeded(
                &mut session,
                self.generation_prompt(cx).to_owned(),
                cx.identity_memories(),
                cx.now(),
            );
            if let Some(context) = recent_generation_context {
                session.push_user(context.to_owned());
            }
            session
        } else {
            self.generation_session.clone()
        };
        let generation_context = push_generation_context(&mut turn_session, args, draft);

        let buffered_deltas = Arc::new(Mutex::new(GenerationDeltaBuffer::default()));
        let (flushed_deltas, mut flushed_delta_batches) =
            tokio::sync::mpsc::unbounded_channel::<Vec<String>>();
        let lutum = self.generation_llm.lutum().await;
        let result = {
            let collect = turn_session
                .text_turn()
                .on_event(GenerationDeltaCollector {
                    buffered_deltas: Arc::clone(&buffered_deltas),
                    flushed_deltas: flushed_deltas.clone(),
                })
                .collect_staged(&lutum)
                .fuse();
            futures::pin_mut!(collect);
            loop {
                futures::select_biased! {
                    batch = flushed_delta_batches.recv().fuse() => {
                        if let Some(batch) = batch {
                            self.emit_generation_delta_batch(draft, batch).await;
                        }
                    }
                    result = collect => break result,
                }
            }
        };

        let slice_limit_reached = self
            .emit_remaining_generation_deltas(
                draft,
                buffered_deltas,
                flushed_deltas,
                &mut flushed_delta_batches,
            )
            .await;

        match result {
            Ok(staged) => {
                let usage = staged.usage;
                let finish_reason = staged.finish_reason.clone();
                if slice_limit_reached || finish_reason == FinishReason::Length {
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
                self.finish_completed_generation_slice(
                    cx,
                    turn_session,
                    usage,
                    draft,
                    &generation_context,
                    active_append,
                )
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
                if slice_limit_reached {
                    cx.compact_and_save(&mut self.generation_session, usage)
                        .await?;
                    return Ok(if draft.accumulated.trim().is_empty() {
                        GenerationStreamOutcome::NoVisibleOutput
                    } else {
                        GenerationStreamOutcome::LengthLimited
                    });
                }
                self.finish_completed_generation_slice(
                    cx,
                    turn_session,
                    usage,
                    draft,
                    &generation_context,
                    active_append,
                )
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
        generation_context: &GenerationTurnContext,
        active_append: bool,
    ) -> Result<GenerationStreamOutcome> {
        let text = draft.accumulated.trim().to_owned();
        if text.is_empty() {
            cx.compact_and_save(&mut self.generation_session, usage)
                .await?;
            return Ok(GenerationStreamOutcome::NoVisibleOutput);
        }

        if !active_append {
            self.generation_session = turn_session;
        }
        push_completed_generation_turn(&mut self.generation_session, generation_context, &text);
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
            .write_cognitive(render_completed_utterance_memo(draft, text))
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
            .write_cognitive(render_in_progress_utterance_memo(args, draft))
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

    async fn emit_remaining_generation_deltas(
        &mut self,
        draft: &mut GenerationDraft,
        buffered_deltas: Arc<Mutex<GenerationDeltaBuffer>>,
        flushed_deltas: tokio::sync::mpsc::UnboundedSender<Vec<String>>,
        flushed_delta_batches: &mut tokio::sync::mpsc::UnboundedReceiver<Vec<String>>,
    ) -> bool {
        let (remaining, limit_reached) = {
            let mut buffered = buffered_deltas
                .lock()
                .expect("generation delta buffer mutex poisoned");
            let remaining = buffered.finish();
            (remaining, buffered.limit_reached())
        };
        if let Some(deltas) = remaining {
            let _ = flushed_deltas.send(deltas);
        }
        drop(flushed_deltas);
        while let Some(deltas) = flushed_delta_batches.recv().await {
            self.emit_generation_delta_batch(draft, deltas).await;
        }
        limit_reached
    }

    async fn emit_generation_delta_batch(
        &mut self,
        draft: &mut GenerationDraft,
        deltas: Vec<String>,
    ) {
        let mut emitted = false;
        for delta in deltas {
            emitted = true;
            let sequence = draft.push_delta(&delta);
            self.utterance
                .emit_delta(draft.target.clone(), draft.generation_id, sequence, delta)
                .await;
            self.record_streaming_progress(draft).await;
        }
        if emitted {
            tokio::task::yield_now().await;
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
    use std::pin::Pin;
    use std::rc::Rc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};
    use std::task::{Context as TaskContext, Poll};
    use std::time::Duration;

    use futures::Stream;
    use lutum::{
        AdapterStructuredTurn, AdapterTextTurn, AgentError, AssistantInputItem, AssistantTurnItem,
        AssistantTurnView, ErasedStructuredTurnEventStream, ErasedTextTurnEvent,
        ErasedTextTurnEventStream, FinishReason, InputMessageRole, MaxOutputTokens, MessageContent,
        MockError, MockLlmAdapter, MockTextScenario, ModelInput, ModelInputItem, RawTextTurnEvent,
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
    use crate::utterance::{Utterance, UtteranceDelta, UtteranceSink};

    #[test]
    fn generation_prompt_documents_empty_stop_and_context_pivot_rules() {
        assert!(GENERATION_PROMPT.contains(
            "If no appropriate fresh utterance should be completed now, output nothing; empty output stops speech and emits no user-visible utterance."
        ));
        assert!(
            GENERATION_PROMPT
                .contains("Already emitted text is visible to the listener and cannot be erased.")
        );
        assert!(
            GENERATION_PROMPT.contains(
                "append listener-facing words that make the visible partial land naturally"
            )
        );
    }

    struct TestSyncStream<S> {
        inner: Mutex<Pin<Box<S>>>,
    }

    impl<S> TestSyncStream<S> {
        fn new(stream: S) -> Self {
            Self {
                inner: Mutex::new(Box::pin(stream)),
            }
        }
    }

    impl<S> Stream for TestSyncStream<S>
    where
        S: Stream + Send + 'static,
    {
        type Item = S::Item;

        fn poll_next(self: Pin<&mut Self>, cx: &mut TaskContext<'_>) -> Poll<Option<Self::Item>> {
            let mut inner = self.inner.lock().expect("test stream mutex poisoned");
            inner.as_mut().poll_next(cx)
        }
    }

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

    #[derive(Clone)]
    struct DelayedGenerationAdapter {
        inner: MockLlmAdapter,
        text_turn_count: Arc<AtomicUsize>,
        release_completion: Arc<tokio::sync::Notify>,
    }

    impl DelayedGenerationAdapter {
        fn generation_only(release_completion: Arc<tokio::sync::Notify>) -> Self {
            Self {
                inner: MockLlmAdapter::new(),
                text_turn_count: Arc::new(AtomicUsize::new(1)),
                release_completion,
            }
        }
    }

    #[async_trait::async_trait]
    impl TurnAdapter for DelayedGenerationAdapter {
        async fn text_turn(
            &self,
            input: ModelInput,
            turn: AdapterTextTurn,
        ) -> Result<ErasedTextTurnEventStream, AgentError> {
            if self.text_turn_count.fetch_add(1, Ordering::SeqCst) == 0 {
                return self.inner.text_turn(input, turn).await;
            }

            Ok(delayed_generation_stream(Arc::clone(
                &self.release_completion,
            )))
        }

        async fn structured_turn(
            &self,
            input: ModelInput,
            turn: AdapterStructuredTurn,
        ) -> Result<ErasedStructuredTurnEventStream, AgentError> {
            self.inner.structured_turn(input, turn).await
        }
    }

    fn delayed_generation_stream(
        release_completion: Arc<tokio::sync::Notify>,
    ) -> ErasedTextTurnEventStream {
        enum State {
            Started,
            Delta(usize),
            Done,
        }

        const DELTAS: [&str; SPEECH_GENERATION_TEXT_DELTA_SLICE_LIMIT] =
            [
                "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P",
            ];

        let stream = futures::stream::unfold(State::Started, move |state| {
            let release_completion = Arc::clone(&release_completion);
            async move {
                match state {
                    State::Started => Some((
                        Ok(ErasedTextTurnEvent::Started {
                            request_id: Some("delayed-generation".to_string()),
                            model: "mock".to_string(),
                        }),
                        State::Delta(0),
                    )),
                    State::Delta(index) if index < DELTAS.len() => Some((
                        Ok(ErasedTextTurnEvent::TextDelta {
                            delta: DELTAS[index].to_string(),
                        }),
                        State::Delta(index + 1),
                    )),
                    State::Delta(_) => {
                        release_completion.notified().await;
                        Some((
                            Ok(ErasedTextTurnEvent::Completed {
                                request_id: Some("delayed-generation".to_string()),
                                finish_reason: FinishReason::Stop,
                                usage: Usage::zero(),
                                committed_turn: Arc::new(AssistantTurnView::from_items(&[
                                    AssistantTurnItem::Text("ABCDEFGHIJKLMNOP".to_string()),
                                ])),
                            }),
                            State::Done,
                        ))
                    }
                    State::Done => None,
                }
            }
        });
        Box::pin(TestSyncStream::new(stream)) as ErasedTextTurnEventStream
    }

    fn prepare_speech_scenario(target: &str, utterance_directive: &str) -> MockTextScenario {
        prepare_speech_scenario_with_language(target, None, utterance_directive)
    }

    fn prepare_speech_scenario_with_language(
        target: &str,
        language: Option<&str>,
        utterance_directive: &str,
    ) -> MockTextScenario {
        let arguments_json = serde_json::json!({
            "target": target,
            "language": language,
            "utterance_directive": utterance_directive
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

    fn continue_speech_scenario(append_directive: &str) -> MockTextScenario {
        let arguments_json = serde_json::json!({
            "append_directive": append_directive
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

    fn redirect_speech_scenario(
        target: &str,
        utterance_directive: &str,
        redirect_reason: &str,
    ) -> MockTextScenario {
        let arguments_json = serde_json::json!({
            "target": target,
            "utterance_directive": utterance_directive,
            "redirect_reason": redirect_reason
        })
        .to_string();
        MockTextScenario::events(vec![
            Ok(RawTextTurnEvent::Started {
                request_id: Some("redirect-speech".into()),
                model: "mock".into(),
            }),
            Ok(RawTextTurnEvent::ToolCallChunk {
                id: "call-redirect-speech".into(),
                name: "redirect_speech".into(),
                arguments_json_delta: arguments_json,
            }),
            Ok(RawTextTurnEvent::Completed {
                request_id: Some("redirect-speech".into()),
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

    fn generation_text_length_limited_scenario(text: &str) -> MockTextScenario {
        generation_text_deltas_scenario_with_finish_reason(&[text], FinishReason::Length)
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
            utterance_directive: "Tell Koro to stay close because Koro asks for help.".into(),
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

    #[derive(Debug, PartialEq, Eq)]
    enum CapturedUtteranceEvent {
        Delta(String),
        Complete(String),
    }

    struct CapturingUtteranceEventSink {
        events: Rc<RefCell<Vec<CapturedUtteranceEvent>>>,
    }

    #[async_trait(?Send)]
    impl UtteranceSink for CapturingUtteranceEventSink {
        async fn on_complete(&self, utterance: Utterance) -> Result<(), PortError> {
            self.events
                .borrow_mut()
                .push(CapturedUtteranceEvent::Complete(utterance.text));
            Ok(())
        }

        async fn on_delta(&self, delta: UtteranceDelta) -> Result<(), PortError> {
            self.events
                .borrow_mut()
                .push(CapturedUtteranceEvent::Delta(delta.delta));
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
                        generation_llm: caps
                            .llm("generation")
                            .with_tier(nuillu_types::ModelTier::Default)
                            .into(),
                        scene: caps.scene_reader(),
                        clock: caps.clock(),
                        self_wake: caps.self_wake(),
                        planning_session: caps
                            .session("planning")
                            .with_tier(nuillu_types::ModelTier::Premium)
                            .with_auto_compaction(planning_session_auto_compaction())
                            .await?,
                        generation_session: caps
                            .session("generation")
                            .with_tier(nuillu_types::ModelTier::Default)
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

    fn model_input_message_text(
        item: &ModelInputItem,
        expected_role: InputMessageRole,
        label: &str,
    ) -> String {
        let ModelInputItem::Message { role, content } = item else {
            panic!("expected {label}");
        };
        assert_eq!(role, &expected_role);
        content
            .as_slice()
            .iter()
            .filter_map(|content| match content {
                MessageContent::Text(text) => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n")
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

        let recent_context = generation_cognition_context_from_entries(&records).unwrap();
        assert!(recent_context.starts_with("Recent context:\n- "));
        assert!(recent_context.contains("sensory evidence"));
        assert!(recent_context.contains("query-memory evidence"));
        assert!(recent_context.contains("interpreter evidence"));
        assert!(recent_context.contains("predict evidence"));
        assert!(recent_context.contains("attention-schema evidence"));
        assert!(!recent_context.contains("New thoughts available to you"));
        assert!(!recent_context.contains("memory evidence should not be included"));
        assert!(!recent_context.contains("speak evidence should not be included"));
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
        assert!(!contexts[0].contains("What you are currently thinking at"));
        assert!(!contexts[1].contains("What you are currently thinking at"));

        let generation_users =
            session_message_texts(&module.generation_session, InputMessageRole::User);
        let generation_contexts = generation_users
            .iter()
            .filter(|text| text.starts_with("Recent context:"))
            .collect::<Vec<_>>();
        assert_eq!(generation_contexts.len(), 2);
        assert!(generation_contexts[0].contains("first cognition"));
        assert!(!generation_contexts[0].contains("second cognition"));
        assert!(generation_contexts[1].contains("second cognition"));
        assert!(!generation_contexts[1].contains("first cognition"));
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
        assert!(generation_text.contains("Generate an utterance to `Ryo`."));
        assert!(generation_text.contains("Language: Japanese"));
        assert!(generation_text.contains("Recent context:\n- Ryo says, \"日本語でお願い\"."));
        assert!(generation_text.contains("Ryo says, \"日本語でお願い\""));
        assert!(!generation_text.contains("Facts for speech:"));
        assert!(generation_text.contains("Planner Directive:\n日本語で短く返事する。"));
        assert!(!generation_text.contains("<think>"));
        assert!(!generation_text.contains("</think>"));
        assert!(!generation_text.contains("Context cue:"));
        assert!(!generation_text.contains("Speak to:"));
        assert!(!generation_text.contains("Substance to express:"));
        assert!(!generation_text.contains("New thoughts available to you at "));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn utterance_directive_flows_to_generation_input_without_facts() {
        let adapter = MockLlmAdapter::new()
            .with_text_scenario(prepare_speech_scenario(
                "Ryo",
                "Return the greeting warmly and acknowledge Ryo.",
            ))
            .with_text_scenario(generation_text_scenario("こんにちは、Ryo。"));
        let capture = CapturingAdapter::new(adapter);
        let observed = capture.clone();
        let mut allocation = ResourceAllocation::default();
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
        publish_cognition_update(&blackboard, &caps, now, "Ryo says, \"こんにちは Nui\".").await;

        let batch = module.next_batch().await.unwrap();
        let cx = test_activate_cx(&module, now).await;
        SpeakModule::activate(&mut module, &cx, &batch)
            .await
            .unwrap();

        assert_eq!(
            completed.borrow().as_slice(),
            &[("Ryo".to_string(), "こんにちは、Ryo。".to_string())]
        );
        let inputs = observed.text_inputs();
        assert_eq!(inputs.len(), 2);
        let generation_text = model_input_text(&inputs[1]);
        assert!(generation_text.contains("Recent context:\n- Ryo says, \"こんにちは Nui\"."));
        assert!(generation_text.contains("Ryo says, \"こんにちは Nui\""));
        assert!(!generation_text.contains("Facts for speech:"));
        assert!(
            generation_text
                .contains("Planner Directive:\nReturn the greeting warmly and acknowledge Ryo.")
        );
        assert!(!generation_text.contains("<think>"));
        assert!(!generation_text.contains("</think>"));
        assert!(!generation_text.contains("Context cue:"));
        assert!(!generation_text.contains("Ryo replies"));
        assert!(!generation_text.contains("Ryo smiles"));
    }

    #[test]
    fn fresh_generation_uses_single_user_prompt() {
        let draft = GenerationDraft::new(7, "Koro");
        let args = test_prepare_speech_args("Koro");
        let mut session = Session::new();

        push_generation_context(&mut session, &args, &draft);
        let items = session.input().items();

        assert_eq!(draft.generation_id, 7);
        assert_eq!(draft.sequence, 0);
        assert_eq!(items.len(), 1);
        let text =
            model_input_message_text(&items[0], InputMessageRole::User, "user generation context");
        assert!(!text.contains("Recent context:"));
        assert!(!text.contains("What you are currently thinking at"));
        assert!(!text.contains("New thoughts available to you at"));
        assert!(text.contains("Generate an utterance to `Koro`."));
        assert!(!text.contains("Facts for speech:"));
        assert!(text.contains("Planner Directive:"));
        assert!(text.contains("Tell Koro to stay close because Koro asks for help."));
        assert!(!text.contains("<think>"));
        assert!(!text.contains("</think>"));
        assert!(!text.contains("Me:"));
        assert!(!text.contains("Context cue:"));
        assert!(!text.contains("Speak to:"));
        assert!(!text.contains("Substance to express:"));
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
            utterance_directive: "日本語で短く返事する。".into(),
        };

        let text = format_generation_turn_user_prompt(&args, &draft);

        assert!(!text.contains("Recent context:"));
        assert!(text.contains("Generate an utterance to `Ryo`."));
        assert!(text.contains("Language: Japanese"));
        assert!(!text.contains("Facts for speech:"));
        assert!(text.contains("Planner Directive:\n日本語で短く返事する。"));
        assert!(!text.contains("<think>"));
        assert!(!text.contains("</think>"));
        assert!(!text.contains("Me:"));
        assert!(!text.contains("Speak to:"));
        assert!(!text.contains("Substance to express:"));
        assert!(!text.contains("New thoughts available to you at"));
    }

    #[test]
    fn unknown_evidence_plan_renders_uncertainty_and_visible_absence() {
        let draft = GenerationDraft::new(8, "Pibi");
        let args = PrepareSpeechArgs {
            target: SpeechTarget::from("Pibi"),
            language: None,
            utterance_directive: "Tell Pibi I do not know whether dinner is ready or where it is, because I see no food or person nearby.".into(),
        };

        let text = format_generation_turn_user_prompt(&args, &draft);

        assert!(!text.contains("Recent context:"));
        assert!(!text.contains("Facts for speech:"));
        assert!(text.contains("I do not know whether dinner is ready"));
        assert!(text.contains("I see no food or person nearby."));
    }

    #[test]
    fn speech_prompts_are_slim_and_avoid_peer_catalog() {
        let prompt = nuillu_module::format_policy_system_prompt(GENERATION_PROMPT, &[]);

        assert!(prompt.len() < 1800);
        assert!(prompt.contains("Do not summarize the request or say what the user wants."));
        assert!(prompt.contains("not appropriate to say aloud to the target"));
        assert!(prompt.contains("introspection, narration, or analysis"));
        assert!(prompt.contains(
            "If Planner Directive is already a usable fresh utterance and no already emitted text is supplied"
        ));
        assert!(prompt.contains("The output is appended directly after Already emitted."));
        assert!(prompt.contains("Emit only the next new text"));
        assert!(prompt.contains("Do not repeat, restate, or include any already emitted text."));
        assert!(
            prompt.contains("Append the next visible text to the in-progress utterance for `Ryo`.")
        );
        assert!(prompt.contains("Append Directive:\nThe already emitted text says hello"));
        assert!(prompt.contains(
            "Already emitted:\nこんにちは！Ryoさんは、何か楽しい\nAssistant:\nことを準備していますか？"
        ));
        assert!(!prompt.contains("Assistant prefill:"));
        assert!(!prompt.contains("Assistant continuation:"));
        assert!(!prompt.contains("Identity memory loaded at agent startup"));
        assert!(!prompt.contains("prepare_speech"));
        assert!(!prompt.contains("tool call"));
        assert!(!prompt.contains("speech_content"));
        assert!(!prompt.contains("You are part of a cognitive system"));
        assert!(!prompt.contains("- cognition-gate:"));
        assert!(!prompt.contains("- query-memory:"));
        assert!(SPEECH_PLANNING_PROMPT.contains("Already emitted text is visible to the listener"));
        assert!(SPEECH_PLANNING_PROMPT.contains("call continue_speech"));
        assert!(
            SPEECH_PLANNING_PROMPT.contains("correction, topic shift, recovery from repetition")
        );
        assert!(SPEECH_PLANNING_PROMPT.contains("In append_directive"));
        assert!(SPEECH_PLANNING_PROMPT.contains("Do not write a replacement utterance there."));
        assert!(
            SPEECH_PLANNING_PROMPT
                .contains("Do not plan speech that repeats content the listener has already heard")
        );
        assert!(SPEECH_PLANNING_PROMPT.contains("Use redirect_speech only when the next words should be addressed to a different listener."));
        assert!(!SPEECH_PLANNING_PROMPT.contains("abort_speech"));
        assert!(!SPEECH_PLANNING_PROMPT.contains("\"self\""));
        assert!(!SPEECH_PLANNING_PROMPT.contains("greet"));
        assert!(!SPEECH_PLANNING_PROMPT.contains("Hi"));
        assert!(!SPEECH_PLANNING_PROMPT.contains("allocation"));
        assert!(!SPEECH_PLANNING_PROMPT.contains("allocated"));
    }

    #[test]
    fn active_planning_input_truncates_long_partial_utterance() {
        let long_partial =
            "The user wants me to generate a Japanese utterance for Ryo. ".repeat(80);
        let active = ActiveSpeech {
            args: test_prepare_speech_args("Ryo"),
            draft: GenerationDraft {
                generation_id: 3,
                sequence: 9,
                accumulated: long_partial.clone(),
                target: "Ryo".into(),
            },
            generation_slices_since_plan: SPEECH_GENERATION_SLICES_PER_PLAN,
        };

        let input = format_planning_input("New thoughts available:\n- none", Some(&active), &[]);

        assert!(input.contains("Already emitted: The user wants me"));
        assert!(!input.contains(&long_partial));
        assert!(
            input.len() < 1_200,
            "active planning input should not carry the full runaway partial"
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn active_planner_directs_visible_repetition_recovery_same_target() {
        let directive = "The listener has already seen \"夢の中の隠れた庭には、\" repeat. Append a short Japanese phrase that smooths over the repetition, then pivot to asking Ryo what the funniest thing the pink elephant did was. Do not repeat \"夢の中の隠れた庭には、\".";
        let adapter = MockLlmAdapter::new().with_text_scenario(continue_speech_scenario(directive));
        let mut allocation = ResourceAllocation::default();
        allocation.set_activation(builtin::speak(), ActivationRatio::ONE);
        let blackboard = Blackboard::with_allocation(allocation);
        let sink: Rc<dyn crate::utterance::UtteranceSink> = Rc::new(CapturingUtteranceSink {
            completed: Rc::new(RefCell::new(Vec::new())),
            done: RefCell::new(None),
        });
        let (mut module, caps) =
            speak_module_with_turn_adapter(blackboard, Arc::new(adapter), sink).await;
        caps.scene().set([Participant::new("Ryo")]);
        let active = ActiveSpeech {
            args: PrepareSpeechArgs {
                target: SpeechTarget::from("Ryo"),
                language: Some("ja".into()),
                utterance_directive: "夢の中の隠れた庭の話をする。".into(),
            },
            draft: GenerationDraft {
                generation_id: 5,
                sequence: 24,
                accumulated: "夢の中の隠れた庭には、夢の中の隠れた庭には、".into(),
                target: "Ryo".into(),
            },
            generation_slices_since_plan: SPEECH_GENERATION_SLICES_PER_PLAN,
        };
        let cx = test_activate_cx(&module, SystemClock.now()).await;
        let plan = module
            .plan_active_speech(
                &cx,
                "New thoughts available to you now:\n- none since the previous speech slice",
                &active,
            )
            .await
            .unwrap()
            .expect("active planner should select a same-target continuation");

        let ActiveSpeechPlan::Continue(plan) = plan else {
            panic!("repeated visible partial should be repaired through continue_speech");
        };
        assert_eq!(plan.target, "Ryo");
        assert_eq!(plan.args.language.as_deref(), Some("ja"));
        assert_eq!(plan.args.utterance_directive, directive);
        let planning_text = session_input_text(&module.planning_session);
        assert!(
            planning_text.contains("Already emitted: 夢の中の隠れた庭には、夢の中の隠れた庭には、")
        );
        assert!(planning_text.contains("Already emitted text is visible to the listener."));
        assert!(planning_text.contains("Use redirect_speech only when the next words should be addressed to a different listener."));
        assert!(!planning_text.contains("abort_speech"));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn active_planner_directs_visible_name_recall_correction() {
        let directive = "Continue naturally in Japanese from the already emitted uncertainty. Say that the name just came back to me, then give the remembered name Luma. Do not keep saying I cannot remember it.";
        let adapter = MockLlmAdapter::new().with_text_scenario(continue_speech_scenario(directive));
        let mut allocation = ResourceAllocation::default();
        allocation.set_activation(builtin::speak(), ActivationRatio::ONE);
        let blackboard = Blackboard::with_allocation(allocation);
        let sink: Rc<dyn crate::utterance::UtteranceSink> = Rc::new(CapturingUtteranceSink {
            completed: Rc::new(RefCell::new(Vec::new())),
            done: RefCell::new(None),
        });
        let (mut module, caps) =
            speak_module_with_turn_adapter(blackboard, Arc::new(adapter), sink).await;
        caps.scene().set([Participant::new("Ryo")]);
        let active = ActiveSpeech {
            args: PrepareSpeechArgs {
                target: SpeechTarget::from("Ryo"),
                language: Some("ja".into()),
                utterance_directive: "名前が思い出せないことを伝える。".into(),
            },
            draft: GenerationDraft {
                generation_id: 6,
                sequence: 9,
                accumulated: "名前が思い出せないんだけど、".into(),
                target: "Ryo".into(),
            },
            generation_slices_since_plan: 1,
        };
        let cx = test_activate_cx(&module, SystemClock.now()).await;
        let plan = module
            .plan_active_speech(
                &cx,
                "New thoughts available to you now:\n- Just now: The remembered name is Luma.",
                &active,
            )
            .await
            .unwrap()
            .expect("active planner should select a same-target correction");

        let ActiveSpeechPlan::Continue(plan) = plan else {
            panic!("same-target name recall should be handled by continue_speech");
        };
        assert_eq!(plan.target, "Ryo");
        assert_eq!(plan.args.language.as_deref(), Some("ja"));
        assert_eq!(plan.args.utterance_directive, directive);
        let planning_text = session_input_text(&module.planning_session);
        assert!(planning_text.contains("Already emitted: 名前が思い出せないんだけど、"));
        assert!(planning_text.contains("The remembered name is Luma."));
        assert!(planning_text.contains(
            "what should no longer be said if the current partial became stale or wrong"
        ));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn active_planner_uses_append_directive_to_bridge_broken_partial_to_new_cognition() {
        let directive = "The visible partial is broken and repetitive. Append a short Japanese bridge that smooths it over, then ask Ryo what Nuillu is, grounded in Ryo saying they develop Nuillu. Do not repeat \"もしよければ、最近行った展示会\" or \"Nuilluとは、どんなものなん\".";
        let adapter = MockLlmAdapter::new().with_text_scenario(continue_speech_scenario(directive));
        let mut allocation = ResourceAllocation::default();
        allocation.set_activation(builtin::speak(), ActivationRatio::ONE);
        let blackboard = Blackboard::with_allocation(allocation);
        let sink: Rc<dyn crate::utterance::UtteranceSink> = Rc::new(CapturingUtteranceSink {
            completed: Rc::new(RefCell::new(Vec::new())),
            done: RefCell::new(None),
        });
        let (mut module, caps) =
            speak_module_with_turn_adapter(blackboard, Arc::new(adapter), sink).await;
        caps.scene().set([Participant::new("Ryo")]);
        let active = ActiveSpeech {
            args: PrepareSpeechArgs {
                target: SpeechTarget::from("Ryo"),
                language: Some("ja".into()),
                utterance_directive: "最近あった楽しいことを聞く。".into(),
            },
            draft: GenerationDraft {
                generation_id: 9,
                sequence: 19,
                accumulated: "何か最近あった楽しいことや、気になるもしよければ、最近行った展示会もしよければ、最近行った展示会Nuilluとは、どんなものなん".into(),
                target: "Ryo".into(),
            },
            generation_slices_since_plan: SPEECH_GENERATION_SLICES_PER_PLAN,
        };
        let cx = test_activate_cx(&module, SystemClock.now()).await;
        let plan = module
            .plan_active_speech(
                &cx,
                "New thoughts available to you now:\n- Ryo says, \"Nuilluの開発をしているよ\".",
                &active,
            )
            .await
            .unwrap()
            .expect("active planner should continue same-target speech with an append directive");

        let ActiveSpeechPlan::Continue(plan) = plan else {
            panic!("same-target broken partial should be bridged through continue_speech");
        };
        assert_eq!(plan.target, "Ryo");
        assert_eq!(plan.args.language.as_deref(), Some("ja"));
        assert_eq!(plan.args.utterance_directive, directive);
        let planning_text = session_input_text(&module.planning_session);
        assert!(planning_text.contains("append_directive"));
        assert!(planning_text.contains("Do not write a replacement utterance there."));
        assert!(planning_text.contains("Already emitted: 何か最近あった楽しいことや"));
        assert!(planning_text.contains("Ryo says, \"Nuilluの開発をしているよ\"."));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn speak_selects_target_from_cognition_log_before_streaming() {
        let adapter = MockLlmAdapter::new()
            .with_text_scenario(prepare_speech_scenario("Koro", "Tell Koro to stay close."))
            .with_text_scenario(generation_text_scenario("Koro, stay close."));
        let capture = CapturingAdapter::new(adapter);
        let observed = capture.clone();
        let mut allocation = ResourceAllocation::default();
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
        assert_eq!(planning_text.matches("Your identity:").count(), 1);
        assert!(!planning_text.contains("Identity memory loaded at agent startup"));
        assert!(planning_text.contains("Koro asks Nuillu to help them stay safe."));
        assert!(planning_text.contains("Completed outward utterance to Koro:\nKoro, stay close."));
        let generation_text = session_input_text(&module.generation_session);
        assert_eq!(generation_text.matches("Your identity:").count(), 1);
        assert!(!generation_text.contains("Identity memory loaded at agent startup"));
        assert!(!generation_text.contains("New thoughts available to you at "));
        assert!(
            generation_text.contains("Recent context:\n- Koro asks Nuillu to help them stay safe.")
        );
        assert!(generation_text.contains("Generate an utterance to `Koro`."));
        assert!(generation_text.contains("Koro asks Nuillu to help them stay safe."));
        assert!(generation_text.contains("Planner Directive:\nTell Koro to stay close."));
        assert!(!generation_text.contains("<think>"));
        assert!(!generation_text.contains("</think>"));
        let inputs = observed.text_inputs();
        assert_eq!(inputs.len(), 2);
        let generation_input = model_input_text(&inputs[1]);
        assert!(
            generation_input
                .contains("Recent context:\n- Koro asks Nuillu to help them stay safe.")
        );
        assert!(!generation_input.contains("New thoughts available to you at "));
        assert!(!generation_input.contains("What you are currently thinking at "));
        assert!(generation_input.contains("Generate an utterance to `Koro`."));
        assert!(!generation_input.contains("<think>"));
        assert!(!generation_input.contains("</think>"));
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

    #[test]
    fn generation_delta_buffer_seals_after_slice_limit() {
        let mut buffer = GenerationDeltaBuffer::default();
        let first_fifteen = [
            "K", "o", "r", "o", ",", " ", "s", "t", "a", "y", " ", "c", "l", "o", "s",
        ];

        for delta in first_fifteen {
            assert_eq!(buffer.push(delta.to_string()), None);
            assert!(!buffer.limit_reached());
        }

        assert_eq!(
            buffer.push("t".to_string()),
            Some(
                [
                    "K", "o", "r", "o", ",", " ", "s", "t", "a", "y", " ", "c", "l", "o", "s",
                    "t",
                ]
                    .into_iter()
                    .map(str::to_string)
                    .collect()
            )
        );
        assert!(buffer.limit_reached());
        assert_eq!(buffer.push(" ignored".to_string()), None);
        assert_eq!(buffer.finish(), None);
    }

    #[test]
    fn generation_delta_buffer_truncates_long_visible_delta() {
        let mut buffer = GenerationDeltaBuffer::default();

        assert_eq!(
            buffer.push("あいうえおかきくけこさしすせそたちつてと追加".to_string()),
            Some(vec!["あいうえおかきくけこさしすせそたちつてと".to_string()])
        );
        assert!(buffer.limit_reached());
        assert_eq!(buffer.push(" ignored".to_string()), None);
        assert_eq!(buffer.finish(), None);
    }

    #[test]
    fn generation_delta_buffer_seals_on_accumulated_char_limit() {
        let mut buffer = GenerationDeltaBuffer::default();

        assert_eq!(buffer.push("123456789".to_string()), None);
        assert!(!buffer.limit_reached());
        assert_eq!(
            buffer.push("abcdefghijklmno".to_string()),
            Some(vec!["123456789".to_string(), "abcdefghijk".to_string()])
        );
        assert!(buffer.limit_reached());
        assert_eq!(buffer.push(" ignored".to_string()), None);
        assert_eq!(buffer.finish(), None);
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
            .collect_generation_slice(&cx, &args, &mut draft, None)
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
    async fn active_generation_prompt_uses_append_directive_without_stale_history() {
        let adapter = MockLlmAdapter::new().with_text_scenario(generation_text_scenario("ay."));
        let capture = CapturingAdapter::new(adapter);
        let observed = capture.clone();
        let mut allocation = ResourceAllocation::default();
        allocation.set_activation(builtin::speak(), ActivationRatio::ONE);
        let blackboard = Blackboard::with_allocation(allocation);
        let deltas = Rc::new(RefCell::new(Vec::new()));
        let sink: Rc<dyn UtteranceSink> = Rc::new(CapturingDeltaSink {
            deltas: Rc::clone(&deltas),
            first_delta: RefCell::new(None),
        });
        let (mut module, _caps) =
            speak_module_with_turn_adapter(blackboard, Arc::new(capture), sink).await;
        module
            .generation_session
            .push_user("Recent context:\n- stale old generation history".to_string());
        let now = SystemClock.now();
        let cx = test_activate_cx(&module, now).await;
        let mut draft = GenerationDraft {
            generation_id: 4,
            sequence: 8,
            accumulated: "Koro, st".into(),
            target: "Koro".into(),
        };
        let args = PrepareSpeechArgs {
            target: SpeechTarget::from("Koro"),
            language: None,
            utterance_directive: "Append only the rest of the phrase so Koro hears stay.".into(),
        };

        let outcome = module
            .collect_generation_slice(
                &cx,
                &args,
                &mut draft,
                Some("Recent context:\n- Koro is waiting for the warning to finish."),
            )
            .await
            .unwrap();

        assert!(matches!(outcome, GenerationStreamOutcome::Completed));
        assert_eq!(draft.accumulated, "Koro, stay.");
        let inputs = observed.text_inputs();
        assert_eq!(inputs.len(), 1);
        let full_input = model_input_text(&inputs[0]);
        assert!(
            full_input.contains("Recent context:\n- Koro is waiting for the warning to finish.")
        );
        assert!(!full_input.contains("stale old generation history"));
        let user_text = model_input_message_text(
            inputs[0].items().last().expect("active generation prompt"),
            InputMessageRole::User,
            "active append generation prompt",
        );
        assert!(
            user_text
                .contains("Append the next visible text to the in-progress utterance for `Koro`.")
        );
        assert!(
            user_text.contains(
                "Append Directive:\nAppend only the rest of the phrase so Koro hears stay."
            )
        );
        assert!(!user_text.contains("Planner Directive:"));
        assert!(!user_text.contains("Generate an utterance to `Koro`."));
        assert!(user_text.contains("The output is appended directly after Already emitted."));
        assert!(user_text.ends_with("Already emitted:\nKoro, st"));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn exact_duplicate_generation_delta_is_emitted_and_advances_draft() {
        let blackboard = Blackboard::with_allocation(ResourceAllocation::default());
        let deltas = Rc::new(RefCell::new(Vec::new()));
        let sink: Rc<dyn UtteranceSink> = Rc::new(CapturingDeltaSink {
            deltas: Rc::clone(&deltas),
            first_delta: RefCell::new(None),
        });
        let (mut module, _caps) =
            speak_module_with_turn_adapter(blackboard, Arc::new(MockLlmAdapter::new()), sink).await;
        let mut draft = GenerationDraft::new(7, "Ryo");

        module
            .emit_generation_delta_batch(&mut draft, vec!["最近".to_string(), "最近".to_string()])
            .await;

        assert_eq!(
            deltas.borrow().as_slice(),
            &[
                ("Ryo".to_string(), 7, 0, "最近".to_string()),
                ("Ryo".to_string(), 7, 1, "最近".to_string()),
            ]
        );
        assert_eq!(draft.sequence, 2);
        assert_eq!(draft.accumulated, "最近最近");
    }

    #[tokio::test(flavor = "current_thread")]
    async fn generation_delta_repetitions_are_not_dropped() {
        let blackboard = Blackboard::with_allocation(ResourceAllocation::default());
        let deltas = Rc::new(RefCell::new(Vec::new()));
        let sink: Rc<dyn UtteranceSink> = Rc::new(CapturingDeltaSink {
            deltas: Rc::clone(&deltas),
            first_delta: RefCell::new(None),
        });
        let (mut module, _caps) =
            speak_module_with_turn_adapter(blackboard, Arc::new(MockLlmAdapter::new()), sink).await;
        let mut draft = GenerationDraft::new(7, "Ryo");

        module
            .emit_generation_delta_batch(
                &mut draft,
                vec![
                    "最近".to_string(),
                    "最近".to_string(),
                    "最近あった".to_string(),
                    " 最近".to_string(),
                    "Nuilluとは".to_string(),
                    "Nuillu".to_string(),
                ],
            )
            .await;

        assert_eq!(
            deltas.borrow().as_slice(),
            &[
                ("Ryo".to_string(), 7, 0, "最近".to_string()),
                ("Ryo".to_string(), 7, 1, "最近".to_string()),
                ("Ryo".to_string(), 7, 2, "最近あった".to_string()),
                ("Ryo".to_string(), 7, 3, " 最近".to_string()),
                ("Ryo".to_string(), 7, 4, "Nuilluとは".to_string()),
                ("Ryo".to_string(), 7, 5, "Nuillu".to_string()),
            ]
        );
        assert_eq!(draft.sequence, 6);
        assert_eq!(draft.accumulated, "最近最近最近あった 最近NuilluとはNuillu");
    }

    #[tokio::test(flavor = "current_thread")]
    async fn repeated_japanese_name_chunks_are_preserved() {
        let blackboard = Blackboard::with_allocation(ResourceAllocation::default());
        let deltas = Rc::new(RefCell::new(Vec::new()));
        let sink: Rc<dyn UtteranceSink> = Rc::new(CapturingDeltaSink {
            deltas: Rc::clone(&deltas),
            first_delta: RefCell::new(None),
        });
        let (mut module, _caps) =
            speak_module_with_turn_adapter(blackboard, Arc::new(MockLlmAdapter::new()), sink).await;
        let mut draft = GenerationDraft::new(7, "Ryo");

        module
            .emit_generation_delta_batch(
                &mut draft,
                vec![
                    "リ".to_string(),
                    "ョ".to_string(),
                    "ウ".to_string(),
                    "さん".to_string(),
                    "、".to_string(),
                    "リ".to_string(),
                    "ョ".to_string(),
                    "ウ".to_string(),
                    "さん".to_string(),
                ],
            )
            .await;

        assert_eq!(draft.sequence, 9);
        assert_eq!(draft.accumulated, "リョウさん、リョウさん");

        assert_eq!(
            deltas.borrow().as_slice(),
            &[
                ("Ryo".to_string(), 7, 0, "リ".to_string()),
                ("Ryo".to_string(), 7, 1, "ョ".to_string()),
                ("Ryo".to_string(), 7, 2, "ウ".to_string()),
                ("Ryo".to_string(), 7, 3, "さん".to_string()),
                ("Ryo".to_string(), 7, 4, "、".to_string()),
                ("Ryo".to_string(), 7, 5, "リ".to_string()),
                ("Ryo".to_string(), 7, 6, "ョ".to_string()),
                ("Ryo".to_string(), 7, 7, "ウ".to_string()),
                ("Ryo".to_string(), 7, 8, "さん".to_string()),
            ]
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn generation_slice_limit_flushes_before_turn_completion() {
        let release_completion = Arc::new(tokio::sync::Notify::new());
        let adapter = DelayedGenerationAdapter::generation_only(Arc::clone(&release_completion));
        let mut allocation = ResourceAllocation::default();
        allocation.set_activation(builtin::speak(), ActivationRatio::ONE);
        let blackboard = Blackboard::with_allocation(allocation);
        let deltas = Rc::new(RefCell::new(Vec::new()));
        let (first_delta_tx, first_delta_rx) = tokio::sync::oneshot::channel();
        let sink: Rc<dyn UtteranceSink> = Rc::new(CapturingDeltaSink {
            deltas: Rc::clone(&deltas),
            first_delta: RefCell::new(Some(first_delta_tx)),
        });
        let (mut module, _caps) =
            speak_module_with_turn_adapter(blackboard, Arc::new(adapter), sink).await;
        let now = SystemClock.now();
        let cx = test_activate_cx(&module, now).await;
        let mut draft = GenerationDraft::new(0, "Koro");
        let args = test_prepare_speech_args("Koro");
        let outcome = {
            let collect = module
                .collect_generation_slice(&cx, &args, &mut draft, None)
                .fuse();
            let first_delta = first_delta_rx.fuse();
            let timeout = tokio::time::sleep(Duration::from_millis(100)).fuse();
            futures::pin_mut!(collect);
            futures::pin_mut!(first_delta);
            futures::pin_mut!(timeout);

            let captured = futures::select_biased! {
                captured = first_delta => captured.expect("first generation delta should be captured"),
                result = collect => panic!("generation returned before any delta was emitted: {result:?}"),
                _ = timeout => panic!("generation did not flush the 16-delta slice before completion"),
            };

            assert_eq!(captured, ("Koro".to_string(), 0, 0, "A".to_string()));
            assert_eq!(
                deltas.borrow().as_slice(),
                &[
                    ("Koro".to_string(), 0, 0, "A".to_string()),
                    ("Koro".to_string(), 0, 1, "B".to_string()),
                    ("Koro".to_string(), 0, 2, "C".to_string()),
                    ("Koro".to_string(), 0, 3, "D".to_string()),
                    ("Koro".to_string(), 0, 4, "E".to_string()),
                    ("Koro".to_string(), 0, 5, "F".to_string()),
                    ("Koro".to_string(), 0, 6, "G".to_string()),
                    ("Koro".to_string(), 0, 7, "H".to_string()),
                    ("Koro".to_string(), 0, 8, "I".to_string()),
                    ("Koro".to_string(), 0, 9, "J".to_string()),
                    ("Koro".to_string(), 0, 10, "K".to_string()),
                    ("Koro".to_string(), 0, 11, "L".to_string()),
                    ("Koro".to_string(), 0, 12, "M".to_string()),
                    ("Koro".to_string(), 0, 13, "N".to_string()),
                    ("Koro".to_string(), 0, 14, "O".to_string()),
                    ("Koro".to_string(), 0, 15, "P".to_string()),
                ]
            );
            release_completion.notify_waiters();
            collect.await.unwrap()
        };
        assert!(matches!(outcome, GenerationStreamOutcome::LengthLimited));
        assert_eq!(draft.accumulated, "ABCDEFGHIJKLMNOP");
    }

    #[tokio::test(flavor = "current_thread")]
    async fn generation_slice_limit_ignores_text_deltas_after_cap() {
        let adapter = MockLlmAdapter::new().with_text_scenario(
            generation_text_deltas_scenario_with_finish_reason(
                &[
                    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O",
                    "P", "ignored",
                ],
                FinishReason::Stop,
            ),
        );
        let mut allocation = ResourceAllocation::default();
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
            .collect_generation_slice(&cx, &args, &mut draft, None)
            .await
            .unwrap();

        assert!(matches!(outcome, GenerationStreamOutcome::LengthLimited));
        assert_eq!(
            deltas.borrow().as_slice(),
            &[
                ("Koro".to_string(), 0, 0, "A".to_string()),
                ("Koro".to_string(), 0, 1, "B".to_string()),
                ("Koro".to_string(), 0, 2, "C".to_string()),
                ("Koro".to_string(), 0, 3, "D".to_string()),
                ("Koro".to_string(), 0, 4, "E".to_string()),
                ("Koro".to_string(), 0, 5, "F".to_string()),
                ("Koro".to_string(), 0, 6, "G".to_string()),
                ("Koro".to_string(), 0, 7, "H".to_string()),
                ("Koro".to_string(), 0, 8, "I".to_string()),
                ("Koro".to_string(), 0, 9, "J".to_string()),
                ("Koro".to_string(), 0, 10, "K".to_string()),
                ("Koro".to_string(), 0, 11, "L".to_string()),
                ("Koro".to_string(), 0, 12, "M".to_string()),
                ("Koro".to_string(), 0, 13, "N".to_string()),
                ("Koro".to_string(), 0, 14, "O".to_string()),
                ("Koro".to_string(), 0, 15, "P".to_string()),
            ]
        );
        assert_eq!(draft.accumulated, "ABCDEFGHIJKLMNOP");
    }

    #[tokio::test(flavor = "current_thread")]
    async fn generation_long_text_delta_is_truncated_and_kept_active() {
        let adapter = MockLlmAdapter::new().with_text_scenario(
            generation_text_deltas_scenario_with_finish_reason(
                &["あいうえおかきくけこさしすせそたちつてと追加", "ignored"],
                FinishReason::Stop,
            ),
        );
        let mut allocation = ResourceAllocation::default();
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
        let mut draft = GenerationDraft::new(0, "Ryo");
        let args = test_prepare_speech_args("Ryo");

        let outcome = module
            .collect_generation_slice(&cx, &args, &mut draft, None)
            .await
            .unwrap();

        assert!(matches!(outcome, GenerationStreamOutcome::LengthLimited));
        assert_eq!(
            deltas.borrow().as_slice(),
            &[(
                "Ryo".to_string(),
                0,
                0,
                "あいうえおかきくけこさしすせそたちつてと".to_string(),
            )]
        );
        assert_eq!(
            draft.accumulated,
            "あいうえおかきくけこさしすせそたちつてと"
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn generation_accumulated_char_limit_stops_before_slice_delta_limit() {
        let adapter = MockLlmAdapter::new().with_text_scenario(
            generation_text_deltas_scenario_with_finish_reason(
                &["1234567890", "abcdefghijklmno", "ignored"],
                FinishReason::Stop,
            ),
        );
        let mut allocation = ResourceAllocation::default();
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
        let mut draft = GenerationDraft::new(0, "Ryo");
        let args = test_prepare_speech_args("Ryo");

        let outcome = module
            .collect_generation_slice(&cx, &args, &mut draft, None)
            .await
            .unwrap();

        assert!(matches!(outcome, GenerationStreamOutcome::LengthLimited));
        assert_eq!(
            deltas.borrow().as_slice(),
            &[
                ("Ryo".to_string(), 0, 0, "1234567890".to_string()),
                ("Ryo".to_string(), 0, 1, "abcdefghij".to_string()),
            ]
        );
        assert_eq!(draft.accumulated, "1234567890abcdefghij");
    }

    #[tokio::test(flavor = "current_thread")]
    async fn short_generation_flushes_delta_before_complete_event() {
        let adapter = MockLlmAdapter::new().with_text_scenario(generation_text_scenario("hi!"));
        let mut allocation = ResourceAllocation::default();
        allocation.set_activation(builtin::speak(), ActivationRatio::ONE);
        let blackboard = Blackboard::with_allocation(allocation);
        let events = Rc::new(RefCell::new(Vec::new()));
        let sink: Rc<dyn UtteranceSink> = Rc::new(CapturingUtteranceEventSink {
            events: Rc::clone(&events),
        });
        let (mut module, _caps) =
            speak_module_with_turn_adapter(blackboard, Arc::new(adapter), sink).await;
        let now = SystemClock.now();
        let cx = test_activate_cx(&module, now).await;
        let mut draft = GenerationDraft::new(0, "Koro");
        let args = test_prepare_speech_args("Koro");

        let outcome = module
            .collect_generation_slice(&cx, &args, &mut draft, None)
            .await
            .unwrap();

        assert!(matches!(outcome, GenerationStreamOutcome::Completed));
        assert_eq!(draft.accumulated, "hi!");
        assert_eq!(
            events.borrow().as_slice(),
            &[
                CapturedUtteranceEvent::Delta("hi!".to_string()),
                CapturedUtteranceEvent::Complete("hi!".to_string()),
            ]
        );
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
            .collect_generation_slice(&cx, &args, &mut draft, None)
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
    async fn generation_model_think_tag_text_is_not_emitted_or_persisted_as_utterance() {
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
            .collect_generation_slice(&cx, &args, &mut draft, None)
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
        let generation_text = session_input_text(&module.generation_session);
        assert!(generation_text.contains("Generate an utterance to `Koro`."));
        assert!(!generation_text.contains("hidden reasoning"));
        assert!(!generation_text.contains("<think>"));
        assert!(!generation_text.contains("</think>"));
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
            .collect_generation_slice(&cx, &args, &mut draft, None)
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
        assert_eq!(active.generation_slices_since_plan, 1);
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
        assert!(memos[0].contains("I am speaking to Koro"));
        assert!(memos[0].contains("Already said:\nKoro, st"));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn text_sliced_generation_records_streaming_without_complete_emit() {
        let adapter = MockLlmAdapter::new()
            .with_text_scenario(prepare_speech_scenario("Koro", "Tell Koro to stay close."))
            .with_text_scenario(generation_text_length_limited_scenario("Koro, st"));
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
        assert!(memos[0].contains("I am speaking to Koro"));
        assert!(memos[0].contains("utterance_directive:\nTell Koro to stay close."));
        assert!(memos[0].contains("Already said:\nKoro, st"));
        assert!(!memos[0].contains("I said to Koro"));
        let active = module
            .active_speech
            .as_ref()
            .expect("sliced output should remain active for continuation");
        assert_eq!(active.generation_slices_since_plan, 1);
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
            .with_text_scenario(generation_text_length_limited_scenario("Koro, st"))
            .with_text_scenario(generation_text_scenario("ay close."));
        let capture = CapturingAdapter::new(adapter);
        let observed = capture.clone();
        let mut allocation = ResourceAllocation::default();
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
        let full_input_text = model_input_text(generation_input);
        assert!(
            !full_input_text
                .contains("Recent context:\n- Koro asks Nuillu to help them stay safe.")
        );
        assert!(!full_input_text.contains("none since the previous speech slice"));
        assert!(!full_input_text.contains("New thoughts available to you at "));
        let items = generation_input.items();
        assert!(!items.is_empty());
        let last = items
            .last()
            .expect("generation input should have a user prompt");
        let user_text = model_input_message_text(
            last,
            InputMessageRole::User,
            "continuation generation user context",
        );
        assert!(
            user_text
                .contains("Append the next visible text to the in-progress utterance for `Koro`.")
        );
        assert!(!user_text.contains("Recent context:"));
        assert!(user_text.contains("Append Directive:\nTell Koro to stay close."));
        assert!(user_text.contains("The output is appended directly after Already emitted."));
        assert!(user_text.contains("Emit only the next new text"));
        assert!(user_text.contains("Do not repeat, restate, or include any already emitted text."));
        assert!(user_text.ends_with("Already emitted:\nKoro, st"));
        assert!(!user_text.contains("Already said:"));
        assert!(!user_text.contains("Speak to:"));
        assert!(!user_text.contains("Substance to express:"));
        assert!(!user_text.contains("<think>"));
        assert!(!user_text.contains("</think>"));
        assert_eq!(
            items
                .iter()
                .filter(|item| {
                    matches!(item, ModelInputItem::Assistant(AssistantInputItem::Text(text)) if text == "Koro, st")
                })
                .count(),
            0,
            "partial utterance should not appear as an assistant prefill"
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn repeated_generation_only_slices_do_not_commit_in_progress_planning_records() {
        let adapter = MockLlmAdapter::new()
            .with_text_scenario(prepare_speech_scenario("Koro", "Tell Koro to stay close."))
            .with_text_scenario(generation_text_length_limited_scenario("Koro, st"))
            .with_text_scenario(generation_text_length_limited_scenario("ay close"));
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
        assert_eq!(active.draft.sequence, 2);
        assert_eq!(active.generation_slices_since_plan, 2);
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
    async fn generation_only_continuation_replans_after_three_slices() {
        let adapter = MockLlmAdapter::new()
            .with_text_scenario(prepare_speech_scenario("Koro", "Tell Koro to stay close."))
            .with_text_scenario(generation_text_length_limited_scenario("Koro, st"))
            .with_text_scenario(generation_text_length_limited_scenario("ay close"))
            .with_text_scenario(generation_text_length_limited_scenario(" now ple"))
            .with_text_scenario(continue_speech_scenario(
                "Tell Koro to stay close now please.",
            ))
            .with_text_scenario(generation_text_scenario("ase."));
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
            .expect("first continuation should not wait for cognition")
            .unwrap();
        assert!(second_batch.continue_active_speech);
        assert!(second_batch.updates.is_empty());
        assert!(second_batch.cognition_entries.is_empty());
        let cx = test_activate_cx(&module, now + chrono::Duration::seconds(1)).await;
        SpeakModule::activate(&mut module, &cx, &second_batch)
            .await
            .unwrap();
        assert_eq!(module.planning_session.list_turns().count(), 1);

        let third_batch = tokio::time::timeout(Duration::from_millis(100), module.next_batch())
            .await
            .expect("second continuation should not wait for cognition")
            .unwrap();
        assert!(third_batch.continue_active_speech);
        assert!(third_batch.updates.is_empty());
        assert!(third_batch.cognition_entries.is_empty());
        let cx = test_activate_cx(&module, now + chrono::Duration::seconds(2)).await;
        SpeakModule::activate(&mut module, &cx, &third_batch)
            .await
            .unwrap();

        let active = module
            .active_speech
            .as_ref()
            .expect("third sliced output should remain active");
        assert_eq!(active.generation_slices_since_plan, 3);
        assert_eq!(module.planning_session.list_turns().count(), 1);

        let fourth_batch = tokio::time::timeout(Duration::from_millis(100), module.next_batch())
            .await
            .expect("third continuation should still self-wake")
            .unwrap();
        assert!(fourth_batch.continue_active_speech);
        assert!(fourth_batch.updates.is_empty());
        assert!(fourth_batch.cognition_entries.is_empty());
        let cx = test_activate_cx(&module, now + chrono::Duration::seconds(3)).await;
        SpeakModule::activate(&mut module, &cx, &fourth_batch)
            .await
            .unwrap();

        assert_eq!(module.planning_session.list_turns().count(), 2);
        assert_eq!(
            completed.borrow().as_slice(),
            &[(
                "Koro".to_string(),
                "Koro, stay close now please.".to_string(),
            )]
        );
        assert!(module.active_speech.is_none());
        let planning_text = session_input_text(&module.planning_session);
        assert!(planning_text.contains("Already emitted: Koro, stay close now ple"));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn active_speech_continue_preserves_target_and_prefills_partial_japanese_greeting() {
        let partial = "Aliceの挨拶には、親しみを";
        let adapter = MockLlmAdapter::new()
            .with_text_scenario(prepare_speech_scenario(
                "everyone",
                "Aliceの挨拶に親しみを込めて短く応える。",
            ))
            .with_text_scenario(generation_text_length_limited_scenario(partial))
            .with_text_scenario(continue_speech_scenario(
                "Aliceの挨拶に親しみを込めて短く応える。",
            ))
            .with_text_scenario(generation_text_scenario("込めて、こんにちは。"));
        let capture = CapturingAdapter::new(adapter);
        let observed = capture.clone();
        let mut allocation = ResourceAllocation::default();
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
        let full_input_text = model_input_text(generation_input);
        assert!(
            full_input_text.contains("Alice is still present and waiting for the greeting reply.")
        );
        let items = generation_input.items();
        assert!(!items.is_empty());
        let last = items
            .last()
            .expect("generation input should have a user prompt");
        let user_text = model_input_message_text(
            last,
            InputMessageRole::User,
            "continuation generation user context",
        );
        assert!(
            user_text.contains(
                "Append the next visible text to the in-progress utterance for `everyone`."
            )
        );
        assert!(!user_text.contains("Alice is still present and waiting for the greeting reply."));
        assert!(user_text.contains("Append Directive:\nAliceの挨拶に親しみを込めて短く応える。"));
        assert!(user_text.contains("The output is appended directly after Already emitted."));
        assert!(user_text.contains("Emit only the next new text"));
        assert!(user_text.contains("Do not repeat, restate, or include any already emitted text."));
        assert!(user_text.ends_with(&format!("Already emitted:\n{partial}")));
        assert!(!user_text.contains("Already said:"));
        assert!(!user_text.contains("Speak to:"));
        assert!(!user_text.contains("Substance to express:"));
        assert!(!user_text.contains("<think>"));
        assert!(!user_text.contains("</think>"));
        assert_eq!(
            items
                .iter()
                .filter(|item| {
                    matches!(item, ModelInputItem::Assistant(AssistantInputItem::Text(text)) if text == partial)
                })
                .count(),
            0,
            "partial utterance should not appear as an assistant prefill"
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn text_sliced_speech_replans_with_late_cognition_and_prefills_generation() {
        let adapter = MockLlmAdapter::new()
            .with_text_scenario(prepare_speech_scenario("Koro", "Tell Koro to duck soon."))
            .with_text_scenario(generation_text_length_limited_scenario("Koro, du"))
            .with_text_scenario(continue_speech_scenario("Tell Koro to duck now."))
            .with_text_scenario(generation_text_scenario("ck now."));
        let capture = CapturingAdapter::new(adapter);
        let observed = capture.clone();
        let mut allocation = ResourceAllocation::default();
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
        let full_input_text = model_input_text(generation_input);
        assert!(full_input_text.contains("Koro sees a falling rock and now needs to duck."));
        assert!(!full_input_text.contains("Koro asks Nuillu to help them stay safe."));
        let items = generation_input.items();
        assert!(!items.is_empty());
        let last = items
            .last()
            .expect("generation input should have a user prompt");
        let user_text = model_input_message_text(
            last,
            InputMessageRole::User,
            "continuation generation user plan",
        );
        assert!(
            user_text
                .contains("Append the next visible text to the in-progress utterance for `Koro`.")
        );
        assert!(!user_text.contains("Koro sees a falling rock and now needs to duck."));
        assert!(user_text.contains("Append Directive:\nTell Koro to duck now."));
        assert!(user_text.contains("The output is appended directly after Already emitted."));
        assert!(user_text.contains("Emit only the next new text"));
        assert!(user_text.contains("Do not repeat, restate, or include any already emitted text."));
        assert!(user_text.ends_with("Already emitted:\nKoro, du"));
        assert!(!user_text.contains("Already said:"));
        assert!(!user_text.contains("Speak to:"));
        assert!(!user_text.contains("Substance to express:"));
        assert!(!user_text.contains("<think>"));
        assert!(!user_text.contains("</think>"));
        assert_eq!(
            items
                .iter()
                .filter(|item| {
                    matches!(item, ModelInputItem::Assistant(AssistantInputItem::Text(text)) if text == "Koro, du")
                })
                .count(),
            0,
            "partial utterance should not appear as an assistant prefill"
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn active_speech_retarget_drops_partial_and_generates_new_target_once() {
        let adapter = MockLlmAdapter::new()
            .with_text_scenario(prepare_speech_scenario("Koro", "Tell Koro to stay close."))
            .with_text_scenario(generation_text_length_limited_scenario("Koro, st"))
            .with_text_scenario(redirect_speech_scenario(
                "OffstageVoice",
                "Tell the offstage voice to stop causing trouble.",
                "The offstage voice is causing immediate trouble.",
            ))
            .with_text_scenario(generation_text_deltas_scenario_with_finish_reason(
                &["Stop that, please."],
                FinishReason::Stop,
            ));
        let capture = CapturingAdapter::new(adapter);
        let observed = capture.clone();
        let mut allocation = ResourceAllocation::default();
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
                "Stop that, please.".to_string()
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
        assert_eq!(progress.partial_utterance, "Stop that, please.");
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
    async fn active_speech_same_target_closes_visible_partial_with_continue() {
        let directive = "The already emitted text says \"Koro, ok\" to Koro. Continue to the same listener by finishing it as a brief graceful ending that Koro is safe now.";
        let adapter = MockLlmAdapter::new()
            .with_text_scenario(prepare_speech_scenario("Koro", "Tell Koro to stay close."))
            .with_text_scenario(generation_text_length_limited_scenario("Koro, ok"))
            .with_text_scenario(continue_speech_scenario(directive))
            .with_text_scenario(generation_text_scenario("ay."));
        let capture = CapturingAdapter::new(adapter);
        let observed = capture.clone();
        let mut allocation = ResourceAllocation::default();
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

        assert_eq!(
            completed.borrow().as_slice(),
            &[("Koro".to_string(), "Koro, okay.".to_string())]
        );
        assert!(module.active_speech.is_none());
        assert_eq!(observed.text_turns().len(), 4);
        let speak_owner = ModuleInstanceId::new(builtin::speak(), ReplicaIndex::ZERO);
        let progress = blackboard
            .read(|bb| bb.utterance_progress_for_instance(&speak_owner).cloned())
            .await
            .unwrap();
        assert_eq!(
            progress.state,
            nuillu_blackboard::UtteranceProgressState::Completed
        );
        assert_eq!(progress.partial_utterance, "Koro, okay.");
        let planning_text = session_input_text(&module.planning_session);
        assert!(planning_text.contains("Already emitted: Koro, ok"));
        assert!(planning_text.contains("Koro no longer needs the warning."));
        let inputs = observed.text_inputs();
        let generation_input = inputs.last().expect("continuation generation input");
        let user_text = model_input_message_text(
            generation_input
                .items()
                .last()
                .expect("generation user prompt"),
            InputMessageRole::User,
            "same-target graceful ending generation input",
        );
        assert!(user_text.contains(&format!("Append Directive:\n{directive}")));
        assert!(user_text.ends_with("Already emitted:\nKoro, ok"));
        let memos = speak_memos(&blackboard).await;
        assert!(
            !memos
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
            .with_text_scenario(generation_text_deltas_scenario_with_finish_reason(
                &["Stop that, please."],
                FinishReason::Stop,
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
    fn resumed_generation_keeps_id_sequence_and_uses_user_prompt_continuation() {
        let mut draft = GenerationDraft::new(11, "Koro");
        let args = test_prepare_speech_args("Koro");
        let mut session = Session::new();

        assert_eq!(draft.push_delta("hello "), 0);
        assert_eq!(draft.push_delta("world"), 1);
        push_generation_context(&mut session, &args, &draft);
        let items = session.input().items();

        assert_eq!(draft.generation_id, 11);
        assert_eq!(draft.sequence, 2);
        assert_eq!(draft.accumulated, "hello world");
        assert_eq!(items.len(), 1);
        let text =
            model_input_message_text(&items[0], InputMessageRole::User, "user generation context");
        assert!(
            text.contains("Append the next visible text to the in-progress utterance for `Koro`.")
        );
        assert!(
            text.contains("Append Directive:\nTell Koro to stay close because Koro asks for help.")
        );
        assert!(text.contains("The output is appended directly after Already emitted."));
        assert!(text.contains("Emit only the next new text"));
        assert!(text.contains("Do not repeat, restate, or include any already emitted text."));
        assert!(text.ends_with("Already emitted:\nhello world"));
        assert!(!text.contains("Already said:"));
        assert!(!text.contains("Output only the remaining continuation"));
        assert!(!text.contains("<think>"));
        assert!(!text.contains("</think>"));
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
