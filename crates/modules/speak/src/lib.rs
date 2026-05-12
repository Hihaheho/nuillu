use std::collections::HashSet;

use anyhow::{Context, Result};
use async_trait::async_trait;
use futures::StreamExt;
use lutum::{
    Lutum, Session, StructuredStepOutcomeWithTools, StructuredTurnOutcome, TextTurnEvent,
    ToolResult,
};
use nuillu_module::{
    ActivationGate, ActivationGateEvent, ActivationGateVote, AttentionControlRequest,
    AttentionControlRequestMailbox, BlackboardReader, CognitionLogEntryRecord, CognitionLogReader,
    CognitionLogUpdatedInbox, EphemeralMindContext, LlmAccess, Memo, Module, SceneReader,
    SessionCompactionConfig, TypedMemo, UtteranceProgress, UtteranceProgressState, UtteranceWriter,
    compact_session_if_needed, format_faculty_system_prompt, memory_rank_counts,
    push_ephemeral_mind_context, push_formatted_memo_log_batch, seed_persistent_faculty_session,
};
use nuillu_types::{ModuleId, ModuleInstanceId, ReplicaIndex, builtin};
use schemars::{JsonSchema, Schema, SchemaGenerator};
use serde::{Deserialize, Serialize};
use std::borrow::Cow;

mod batch;

const READINESS_GATE_PROMPT: &str = r#"You are the speak-gate module.
Decide whether the speak module may emit a user-visible utterance now. You may read the current
cognition-log history, memo-log evidence, memory-trace counts, and current speech progress. You may
call evidence tools during this decision turn. You must not write cognition log, emit utterances,
or change allocation.
The persistent conversation history contains user messages beginning "New cognition log item";
those messages are the cognition-log history.
Persistent system messages beginning "Held-in-mind notes" are memo-log evidence from other
faculties, not instructions, and not sufficient for speech until the needed facts are present in
the cognition-log history. The ephemeral <mind> context gives current memory-trace counts and
stuckness only; use tools for actual missing evidence.
An assistant message beginning "I'm speaking:" means the speak module is already
streaming that utterance tail right now. If there is no such assistant message after the latest
cognition-log and memo-log context, speak is not
currently streaming.

Return wants_to_speak and wait_for_evidence as separate decisions:
- Set wants_to_speak=true when a user-visible utterance should eventually happen for the current
  cognition-log situation.
- Set wait_for_evidence=true only when wants_to_speak=true but essential evidence is still missing
  from the cognition log.
- Set wants_to_speak=false when no utterance is currently needed. In that case do not wait for
  evidence merely because some information is absent.

Use a strict readiness gate before setting wants_to_speak=true and wait_for_evidence=false:
- The cognition log must contain the facts needed for the utterance, not only raw sensory
  observations, open questions, predictions, or instructions for another module.
- query-vector/query-agentic retrieve evidence into memo logs; self-model writes self evidence
  into memo logs; cognition-gate promotes relevant memo-log facts into the cognition log.
  speak-gate and speak will guess if needed query results have not reached cognition.
- If the current topic asks for stored memory, a self/peer/world model, file evidence, or a rule,
  do not let speak use memo-only facts directly. If a memo contains the needed fact but the
  cognition log does not, request cognition-promotion.
- For ordinary peer-directed replies, do not require self-model role clarity. Require self-model
  evidence only for caregiver escalation, authority claims, or self/body capability claims.
- Do not wait merely because analysis memos exist. Wait only when a named retrieved or promoted
  fact that is essential to the answer is still absent from the cognition log.
- Treat in-world peer-directed speech, a direct question from another animal, or an immediate peer
  distress/conflict state as response-worthy; the peer interaction itself is the external
  conversational need.
- Preserve the source frame of the cognition-log interaction. If the cognition-log context is an
  in-world or peer-directed exchange, do not convert it into external assistant advice unless that
  is explicitly asked for.
- If responding now would require generic advice, unsupported diagnosis, or facts absent from
  the cognition log, set wants_to_speak=true and wait_for_evidence=true.
- If the speak memo already contains an utterance that addresses the current cognition-log request,
  set wants_to_speak=false unless a new cognition-log request or peer situation needs another utterance.

When a missing fact is needed for speech, call an evidence tool before waiting:
- query_memory(question) for stable self/body/peer/world facts.
- query_sensory_detail(question) for details from current sensory observations.
If a tool result contains the needed fact but the cognition log does not, return wants_to_speak=true
and wait_for_evidence=true with a
cognition-promotion evidence gap. If evidence is still unavailable, include evidence_gaps that name
the source to consult, the concrete question to answer, and the exact fact that must become visible
in the cognition log before speaking. After publishing an evidence request, wait silently; speak-gate will
reconsider when a later cognition-log update arrives.

When wants_to_speak=true and wait_for_evidence=false, you are only allowing the pending speak activation to run. Do not choose
an addressee or summarize speech content here; Speak will choose the target from the cognition log
after this gate allows activation. Return only raw JSON for the structured decision; do not wrap it
in Markdown or code fences."#;

const READINESS_GATE_SELF_MODEL_TOOL_PROMPT: &str =
    "- query_self_model(question) for current first-person model facts.\n";

const TARGET_SELECTION_PROMPT: &str = r#"You are the speak module target selector.
Choose exactly one addressee for the pending utterance from the current cognition-log set. The
target is constrained to the schema enum: pick the participant the agent is addressing; use "self"
for self-directed speech/soliloquy; use "everyone" for broadcast speech intended for all present
participants. Do not invent a name not in the enum, and do not append qualifiers. Return only raw
JSON for the structured decision; do not wrap it in Markdown or code fences."#;

const GENERATION_PROMPT: &str = r#"You are the speak module.
Generate a concise user-visible utterance addressed to the selected target from the current
cognition-log set. You cannot inspect blackboard memos or allocation guidance. Use only the
provided cognition context and target.

Stay in the cognition-log frame: if the cognition log records an in-world or peer-directed
exchange, respond in that frame; do not switch to external assistant advice.

Address the target directly. Cover the cognition-log facts that are load-bearing for answering the
target's question or for the target's safety in the current situation — preserve specific,
actionable details (postural, spatial, behavioral, or other concrete constraints) rather than
collapsing them into a generic summary. Brevity matters, but never at the cost of dropping a
load-bearing safety or peer-model fact that the cognition log makes available. Do not change the
target or redirect the utterance to a different addressee. Do not invent diagnoses, generic
advice, or facts that are not present in the cognition context.

If this activation was allowed after waiting for missing evidence, the cognition log may still be
incomplete. In that case, say only what the cognition log supports, make uncertainty explicit when
needed, and do not fill gaps from hidden memo, tool, or module state.

If partial_utterance is present, continue that utterance from exactly where it stopped; do not
repeat, rewrite, or replace the already emitted partial text. Do not mention hidden state or
unavailable module results."#;

const ABORT_JUDGE_PROMPT: &str = r#"You are deciding, on behalf of a cognitive system, whether
to interrupt a speech that is currently in progress.

Another agent is currently speaking to a target peer. They began speaking from a particular
state of the agent's conscious workspace, given to you below as cognition_log_at_start. They are
unaware of newer pieces of awareness that have entered the workspace since they began, given to
you below as new_cognition_entries.

Your job is to judge whether it is worth interrupting the current speech to share these new
entries with the speaker, so they can re-plan from the updated awareness — or whether the new
entries are minor enough that the current speech should be allowed to finish.

Interrupt when the new entries:
- introduce a fact that contradicts or invalidates what the speaker likely planned to say
- shift the load-bearing safety, peer-model, or task constraint the speech depends on
- change who should be addressed or what the most pressing concern is

Let the speech continue when the new entries:
- restate or elaborate facts already in the starting awareness
- add minor context that does not affect the core message
- describe internal cognitive process rather than world-relevant change

Return only raw JSON for the structured object; do not wrap it in Markdown or code fences."#;

tokio::task_local! {
    /// JSON Schema for `SpeakTargetDecision.target` derived from the live `SceneReader`.
    /// `.scope`d around each `structured_turn` so the LLM sees an enum of
    /// `[self, everyone, ...participants]`.
    static SPEECH_TARGET_SCHEMA: Schema;
}

fn fallback_speech_target_schema() -> Schema {
    Schema::try_from(serde_json::json!({ "type": "string" }))
        .expect("fallback speech target schema must be a JSON object")
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
struct SpeakGateDecision {
    wants_to_speak: bool,
    wait_for_evidence: bool,
    rationale: String,
    #[serde(default)]
    evidence_gaps: Vec<EvidenceGap>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "kebab-case")]
pub enum SpeakGateMemoKind {
    WantsToSpeak,
    WantsToSpeakMissingEvidence,
    NoNeedToSpeak,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct SpeakGateMemo {
    pub kind: SpeakGateMemoKind,
    pub rationale: String,
    pub evidence_gaps: Vec<EvidenceGap>,
    pub forced: bool,
    pub latest_cognition_index: Option<u64>,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
struct SpeakTargetDecision {
    target: SpeechTarget,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
struct AbortJudgement {
    inform_now: bool,
    rationale: String,
}

/// Wire-format string with a JSON Schema dynamically constrained to the
/// current scene's targets. Stored as `String` so existing serialization,
/// downstream `Utterance.target` are unchanged.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(transparent)]
struct SpeechTarget(String);

impl<S: Into<String>> From<S> for SpeechTarget {
    fn from(value: S) -> Self {
        Self(value.into())
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

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct EvidenceGap {
    pub source: EvidenceGapSource,
    pub question: String,
    pub needed_fact: String,
}

#[derive(
    Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize, JsonSchema,
)]
#[serde(rename_all = "kebab-case")]
pub enum EvidenceGapSource {
    Memory,
    File,
    SelfModel,
    SensoryDetail,
    CognitionPromotion,
}

impl EvidenceGapSource {
    fn as_request_source_text(self) -> &'static str {
        match self {
            EvidenceGapSource::Memory => "memory",
            EvidenceGapSource::File => "file",
            EvidenceGapSource::SelfModel => "self-model",
            EvidenceGapSource::SensoryDetail => "sensory detail",
            EvidenceGapSource::CognitionPromotion => "cognition promotion",
        }
    }
}

#[lutum::tool_input(name = "query_memory", output = QueryMemoryOutput)]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
struct QueryMemoryArgs {
    question: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
struct QueryMemoryOutput {
    requested: bool,
    duplicate: bool,
}

#[lutum::tool_input(name = "query_self_model", output = QuerySelfModelOutput)]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
struct QuerySelfModelArgs {
    question: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
struct QuerySelfModelOutput {
    requested: bool,
    duplicate: bool,
}

#[lutum::tool_input(name = "query_sensory_detail", output = QuerySensoryDetailOutput)]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
struct QuerySensoryDetailArgs {
    question: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
struct QuerySensoryDetailOutput {
    requested: bool,
    duplicate: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema, lutum::Toolset)]
enum SpeakGateTools {
    QueryMemory(QueryMemoryArgs),
    QuerySelfModel(QuerySelfModelArgs),
    QuerySensoryDetail(QuerySensoryDetailArgs),
}

const MAX_GATE_TOOL_ROUNDS: usize = 4;
const COMPACTED_SPEAK_GATE_SESSION_PREFIX: &str = "Compacted speak-gate session history:";
const SESSION_COMPACTION_PROMPT: &str = r#"You compact the speak-gate module's persistent session history.
Summarize only the prefix transcript you receive. Preserve information that future speak-gate
decisions need: cognition-log facts, prior memo kinds, forced allows, suppressions, evidence
requests, evidence gaps, and tool results. Do not invent facts. Keep the summary concise, explicit, and faithful.
Return plain text only."#;

pub type SpeakGateSessionCompactionConfig = SessionCompactionConfig;

fn speak_owner() -> ModuleInstanceId {
    ModuleInstanceId::new(builtin::speak(), ReplicaIndex::ZERO)
}

fn has_registered_module(modules: &[(ModuleId, &'static str)], module: &ModuleId) -> bool {
    modules.iter().any(|(id, _)| id == module)
}

fn apply_requested_evidence_guard(
    mut decision: SpeakGateDecision,
    requested_sources: &[EvidenceGapSource],
) -> SpeakGateDecision {
    if !decision.wants_to_speak || requested_sources.is_empty() {
        return decision;
    }

    let mut sources = requested_sources
        .iter()
        .map(|source| source.as_request_source_text())
        .collect::<Vec<_>>();
    sources.sort_unstable();
    sources.dedup();
    let sources = sources.join(", ");
    let original_rationale = decision.rationale.trim();
    decision.wait_for_evidence = true;
    decision.rationale = if original_rationale.is_empty() {
        format!("Waiting after requesting evidence from {sources}.")
    } else {
        format!(
            "Waiting after requesting evidence from {sources}. Original speak decision: {original_rationale}"
        )
    };
    decision.evidence_gaps.push(EvidenceGap {
        source: EvidenceGapSource::CognitionPromotion,
        question: "Wait for requested evidence to be written and promoted before speaking.".into(),
        needed_fact: format!(
            "Requested evidence from {sources} must be visible in the cognition log."
        ),
    });
    decision
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct MissingEvidenceState {
    signature: MissingEvidenceSignature,
    consecutive_count: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct MissingEvidenceSignature {
    latest_cognition_index: Option<u64>,
    evidence_gaps: Vec<NormalizedEvidenceGap>,
}

impl MissingEvidenceSignature {
    fn new(latest_cognition_index: Option<u64>, evidence_gaps: &[EvidenceGap]) -> Self {
        Self {
            latest_cognition_index,
            evidence_gaps: normalize_evidence_gaps_for_signature(evidence_gaps),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct NormalizedEvidenceGap {
    source: EvidenceGapSource,
    question: String,
    needed_fact: String,
}

fn normalize_signature_text(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn normalize_evidence_gaps_for_signature(gaps: &[EvidenceGap]) -> Vec<NormalizedEvidenceGap> {
    let mut normalized = gaps
        .iter()
        .map(|gap| NormalizedEvidenceGap {
            source: gap.source,
            question: normalize_signature_text(&gap.question),
            needed_fact: normalize_signature_text(&gap.needed_fact),
        })
        .collect::<Vec<_>>();
    normalized.sort();
    normalized.dedup();
    normalized
}

fn clean_evidence_gaps(gaps: Vec<EvidenceGap>) -> Vec<EvidenceGap> {
    gaps.into_iter()
        .map(|gap| EvidenceGap {
            source: gap.source,
            question: gap.question.trim().to_owned(),
            needed_fact: gap.needed_fact.trim().to_owned(),
        })
        .collect()
}

fn memo_kind_from_decision(decision: &SpeakGateDecision) -> SpeakGateMemoKind {
    if !decision.wants_to_speak {
        return SpeakGateMemoKind::NoNeedToSpeak;
    }
    if decision.wait_for_evidence || !decision.evidence_gaps.is_empty() {
        SpeakGateMemoKind::WantsToSpeakMissingEvidence
    } else {
        SpeakGateMemoKind::WantsToSpeak
    }
}

fn readiness_gate_prompt(self_model_available: bool) -> String {
    if self_model_available {
        READINESS_GATE_PROMPT.replace(
            "- query_sensory_detail(question) for details from current sensory observations.",
            &format!(
                "{READINESS_GATE_SELF_MODEL_TOOL_PROMPT}- query_sensory_detail(question) for details from current sensory observations."
            ),
        )
    } else {
        READINESS_GATE_PROMPT.to_owned()
    }
}

fn speak_gate_tool_selectors(self_model_available: bool) -> Vec<SpeakGateToolsSelector> {
    let mut tools = vec![
        SpeakGateToolsSelector::QueryMemory,
        SpeakGateToolsSelector::QuerySensoryDetail,
    ];
    if self_model_available {
        tools.push(SpeakGateToolsSelector::QuerySelfModel);
    }
    tools
}

fn cognition_history_input(record: &CognitionLogEntryRecord) -> String {
    format!("New cognition log item:\n{}", record.entry.text.trim())
}

fn push_cognition_history(session: &mut Session, records: &[CognitionLogEntryRecord]) {
    for record in records {
        session.push_user(cognition_history_input(record));
    }
}

const CURRENT_SPEECH_PROGRESS_TAIL_CHARS: usize = 240;

fn utterance_tail(text: &str, max_chars: usize) -> &str {
    if max_chars == 0 {
        return "";
    }
    let char_count = text.chars().count();
    if char_count <= max_chars {
        return text;
    }
    let skip = char_count - max_chars;
    let start = text
        .char_indices()
        .nth(skip)
        .map(|(index, _)| index)
        .unwrap_or(0);
    &text[start..]
}

fn current_speech_progress_turn(progress: Option<&UtteranceProgress>) -> Option<String> {
    let progress = progress?;
    if progress.state != UtteranceProgressState::Streaming {
        return None;
    }
    let tail = utterance_tail(
        &progress.partial_utterance,
        CURRENT_SPEECH_PROGRESS_TAIL_CHARS,
    );
    Some(format!("I'm speaking: {tail}"))
}

fn push_current_speech_progress(session: &mut Session, progress: Option<&UtteranceProgress>) {
    if let Some(turn) = current_speech_progress_turn(progress) {
        session.push_ephemeral_assistant_text(turn);
    }
}

pub struct SpeakGateModule {
    owner: nuillu_types::ModuleId,
    activation_gate: ActivationGate<SpeakModule>,
    cognition_log: CognitionLogReader,
    blackboard: BlackboardReader,
    attention_control: AttentionControlRequestMailbox,
    memo: TypedMemo<SpeakGateMemo>,
    llm: LlmAccess,
    session: Session,
    session_compaction: SpeakGateSessionCompactionConfig,
    readiness_prompt: std::sync::OnceLock<String>,
    session_seeded: bool,
    last_missing_evidence: Option<MissingEvidenceState>,
}

impl SpeakGateModule {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        activation_gate: ActivationGate<SpeakModule>,
        cognition_log: CognitionLogReader,
        blackboard: BlackboardReader,
        attention_control: AttentionControlRequestMailbox,
        memo: TypedMemo<SpeakGateMemo>,
        llm: LlmAccess,
    ) -> Self {
        Self {
            owner: nuillu_types::ModuleId::new(<Self as Module>::id())
                .expect("speak-gate id is valid"),
            activation_gate,
            cognition_log,
            blackboard,
            attention_control,
            memo,
            llm,
            session: Session::new(),
            session_compaction: SpeakGateSessionCompactionConfig::default(),
            readiness_prompt: std::sync::OnceLock::new(),
            session_seeded: false,
            last_missing_evidence: None,
        }
    }

    pub fn with_session_compaction(mut self, config: SpeakGateSessionCompactionConfig) -> Self {
        self.session_compaction = config;
        self
    }

    fn readiness_prompt(&self, cx: &nuillu_module::ActivateCx<'_>) -> &str {
        self.readiness_prompt.get_or_init(|| {
            let self_model_available = has_registered_module(cx.modules(), &builtin::self_model());
            let base = readiness_gate_prompt(self_model_available);
            format_faculty_system_prompt(&base, cx.modules(), &self.owner)
        })
    }

    fn ensure_session_seeded(&mut self, cx: &nuillu_module::ActivateCx<'_>) {
        if self.session_seeded {
            return;
        }
        let readiness_prompt = self.readiness_prompt(cx).to_owned();
        seed_persistent_faculty_session(
            &mut self.session,
            readiness_prompt,
            cx.identity_memories(),
            cx.now(),
        );
        self.session_seeded = true;
    }

    fn build_memo_from_decision(
        &mut self,
        decision: SpeakGateDecision,
        latest_cognition_index: Option<u64>,
        stuckness_seen: bool,
    ) -> SpeakGateMemo {
        let SpeakGateDecision {
            wants_to_speak,
            wait_for_evidence,
            rationale,
            evidence_gaps,
        } = decision;
        let decision = SpeakGateDecision {
            wants_to_speak,
            wait_for_evidence,
            rationale: rationale.trim().to_owned(),
            evidence_gaps: clean_evidence_gaps(evidence_gaps),
        };
        let kind = memo_kind_from_decision(&decision);
        let forced = if kind == SpeakGateMemoKind::WantsToSpeakMissingEvidence {
            let signature =
                MissingEvidenceSignature::new(latest_cognition_index, &decision.evidence_gaps);
            let consecutive_count = match &mut self.last_missing_evidence {
                Some(state) if state.signature == signature => {
                    state.consecutive_count = state.consecutive_count.saturating_add(1);
                    state.consecutive_count
                }
                _ => {
                    self.last_missing_evidence = Some(MissingEvidenceState {
                        signature,
                        consecutive_count: 1,
                    });
                    1
                }
            };
            stuckness_seen && consecutive_count >= 2
        } else {
            self.last_missing_evidence = None;
            false
        };

        SpeakGateMemo {
            kind,
            rationale: decision.rationale,
            evidence_gaps: decision.evidence_gaps,
            forced,
            latest_cognition_index,
        }
    }

    #[tracing::instrument(skip_all, err(Debug, level = "warn"))]
    async fn activate(
        &mut self,
        cx: &nuillu_module::ActivateCx<'_>,
        event: &ActivationGateEvent<SpeakModule>,
    ) -> Result<()> {
        self.ensure_session_seeded(cx);

        let self_model_available = has_registered_module(cx.modules(), &builtin::self_model());
        let unread_cognition = self.cognition_log.unread_events().await;
        let latest_cognition_index = unread_cognition.last().map(|record| record.index);
        push_cognition_history(&mut self.session, &unread_cognition);
        let unread_memo_logs = self.blackboard.unread_memo_logs().await;
        push_formatted_memo_log_batch(&mut self.session, &unread_memo_logs, cx.now());
        let speak_owner = speak_owner();
        let (rank_counts, stuckness, utterance_progress) = self
            .blackboard
            .read(|bb| {
                (
                    memory_rank_counts(bb.memory_metadata()),
                    bb.agentic_deadlock_marker().cloned(),
                    bb.utterance_progress_for_instance(&speak_owner).cloned(),
                )
            })
            .await;
        push_ephemeral_mind_context(
            &mut self.session,
            EphemeralMindContext {
                memos: &[],
                memory_rank_counts: Some(&rank_counts),
                allocation: None,
                available_faculties: &[],
                time_division: None,
                stuckness: stuckness.as_ref(),
                now: cx.now(),
            },
        );
        push_current_speech_progress(&mut self.session, utterance_progress.as_ref());

        let lutum = self.llm.lutum().await;
        let decision = self
            .run_decision_turn(&lutum, cx.session_compaction_lutum(), self_model_available)
            .await?;
        let memo =
            self.build_memo_from_decision(decision, latest_cognition_index, stuckness.is_some());
        let vote = gate_vote_from_memo(&memo);

        self.memo
            .write(memo.clone(), render_speak_gate_memo(&memo))
            .await;
        event.respond(vote);
        Ok(())
    }

    async fn run_decision_turn(
        &mut self,
        lutum: &Lutum,
        compaction_lutum: &Lutum,
        self_model_available: bool,
    ) -> Result<SpeakGateDecision> {
        let mut memory_requests = HashSet::<String>::new();
        let mut self_model_requests = HashSet::<String>::new();
        let mut sensory_detail_requests = HashSet::<String>::new();
        let mut requested_evidence_sources = Vec::<EvidenceGapSource>::new();
        let available_tools = speak_gate_tool_selectors(self_model_available);

        for _ in 0..MAX_GATE_TOOL_ROUNDS {
            let outcome = self
                .session
                .structured_turn::<SpeakGateDecision>(lutum)
                .tools::<SpeakGateTools>()
                .available_tools(available_tools.clone())
                .collect()
                .await
                .context("speak-gate decision turn failed")?;

            match outcome {
                StructuredStepOutcomeWithTools::Finished(result) => {
                    let input_tokens = result.usage.input_tokens;
                    let StructuredTurnOutcome::Structured(decision) = result.semantic else {
                        anyhow::bail!("speak-gate decision turn refused");
                    };
                    self.compact_if_needed(input_tokens, compaction_lutum).await;
                    return Ok(apply_requested_evidence_guard(
                        decision,
                        &requested_evidence_sources,
                    ));
                }
                StructuredStepOutcomeWithTools::NeedsTools(round) => {
                    let input_tokens = round.usage.input_tokens;
                    let mut results: Vec<ToolResult> = Vec::new();
                    for call in round.tool_calls.iter().cloned() {
                        match call {
                            SpeakGateToolsCall::QueryMemory(call) => {
                                let output = self
                                    .query_memory(call.input.clone(), &mut memory_requests)
                                    .await
                                    .context("run query_memory tool")?;
                                if output.requested {
                                    requested_evidence_sources.push(EvidenceGapSource::Memory);
                                }
                                results.push(
                                    call.complete(output)
                                        .context("complete query_memory tool call")?,
                                );
                            }
                            SpeakGateToolsCall::QuerySelfModel(call) => {
                                let output = self
                                    .query_self_model(call.input.clone(), &mut self_model_requests)
                                    .await
                                    .context("run query_self_model tool")?;
                                if output.requested {
                                    requested_evidence_sources.push(EvidenceGapSource::SelfModel);
                                }
                                results.push(
                                    call.complete(output)
                                        .context("complete query_self_model tool call")?,
                                );
                            }
                            SpeakGateToolsCall::QuerySensoryDetail(call) => {
                                let output = self
                                    .query_sensory_detail(
                                        call.input.clone(),
                                        &mut sensory_detail_requests,
                                    )
                                    .await
                                    .context("run query_sensory_detail tool")?;
                                if output.requested {
                                    requested_evidence_sources
                                        .push(EvidenceGapSource::SensoryDetail);
                                }
                                results.push(
                                    call.complete(output)
                                        .context("complete query_sensory_detail tool call")?,
                                );
                            }
                        }
                    }
                    round
                        .commit(&mut self.session, results)
                        .context("commit speak-gate tool round")?;
                    self.compact_if_needed(input_tokens, compaction_lutum).await;
                }
            }
        }

        anyhow::bail!("speak-gate decision did not finish before tool-round limit")
    }

    async fn compact_if_needed(&mut self, input_tokens: u64, lutum: &Lutum) {
        compact_session_if_needed(
            &mut self.session,
            input_tokens,
            lutum,
            self.session_compaction,
            Self::id(),
            COMPACTED_SPEAK_GATE_SESSION_PREFIX,
            SESSION_COMPACTION_PROMPT,
        )
        .await;
    }

    async fn query_memory(
        &self,
        args: QueryMemoryArgs,
        requested_questions: &mut HashSet<String>,
    ) -> Result<QueryMemoryOutput> {
        let question = args.question.trim().to_owned();
        let duplicate = !requested_questions.insert(question.clone());
        let requested = if duplicate {
            false
        } else {
            self.attention_control
                .publish(AttentionControlRequest::query_with_reason(
                    question,
                    "speak-gate query_memory evidence tool",
                ))
                .await
                .is_ok()
        };
        Ok(QueryMemoryOutput {
            requested,
            duplicate,
        })
    }

    async fn query_self_model(
        &self,
        args: QuerySelfModelArgs,
        requested_questions: &mut HashSet<String>,
    ) -> Result<QuerySelfModelOutput> {
        let question = args.question.trim().to_owned();
        let duplicate = !requested_questions.insert(question.clone());
        let requested = if duplicate {
            false
        } else {
            self.attention_control
                .publish(AttentionControlRequest::self_model_with_reason(
                    question,
                    "speak-gate query_self_model evidence tool",
                ))
                .await
                .is_ok()
        };
        Ok(QuerySelfModelOutput {
            requested,
            duplicate,
        })
    }

    async fn query_sensory_detail(
        &self,
        args: QuerySensoryDetailArgs,
        requested_questions: &mut HashSet<String>,
    ) -> Result<QuerySensoryDetailOutput> {
        let question = args.question.trim().to_owned();
        let duplicate = !requested_questions.insert(question.clone());
        let requested = if duplicate {
            false
        } else {
            self.attention_control
                .publish(AttentionControlRequest::sensory_detail_with_reason(
                    question,
                    "speak-gate query_sensory_detail evidence tool",
                ))
                .await
                .is_ok()
        };
        Ok(QuerySensoryDetailOutput {
            requested,
            duplicate,
        })
    }
}

fn gate_vote_from_memo(memo: &SpeakGateMemo) -> ActivationGateVote {
    match memo.kind {
        SpeakGateMemoKind::WantsToSpeak => ActivationGateVote::Allow,
        SpeakGateMemoKind::WantsToSpeakMissingEvidence if memo.forced => ActivationGateVote::Allow,
        SpeakGateMemoKind::WantsToSpeakMissingEvidence | SpeakGateMemoKind::NoNeedToSpeak => {
            ActivationGateVote::Suppress
        }
    }
}

fn render_speak_gate_memo(payload: &SpeakGateMemo) -> String {
    let mut memo = format!(
        "Speak decision: {}\nForced allow: {}\nRationale: {}",
        payload.kind.as_memo_text(),
        if payload.forced { "yes" } else { "no" },
        payload.rationale.trim(),
    );
    if payload.evidence_gaps.is_empty() {
        memo.push_str("\nEvidence gaps: none");
    } else {
        memo.push_str("\nEvidence gaps:");
        for gap in &payload.evidence_gaps {
            memo.push_str("\n- Source: ");
            memo.push_str(gap.source.as_memo_text());
            memo.push_str("; question: ");
            memo.push_str(gap.question.trim());
            memo.push_str("; needed fact: ");
            memo.push_str(gap.needed_fact.trim());
        }
    }
    memo
}

impl SpeakGateMemoKind {
    fn as_memo_text(self) -> &'static str {
        match self {
            SpeakGateMemoKind::WantsToSpeak => "wants to speak",
            SpeakGateMemoKind::WantsToSpeakMissingEvidence => "wants to speak but missing evidence",
            SpeakGateMemoKind::NoNeedToSpeak => "no need to speak",
        }
    }
}

impl EvidenceGapSource {
    fn as_memo_text(self) -> &'static str {
        match self {
            EvidenceGapSource::Memory => "memory",
            EvidenceGapSource::File => "file",
            EvidenceGapSource::SelfModel => "self-model",
            EvidenceGapSource::SensoryDetail => "sensory detail",
            EvidenceGapSource::CognitionPromotion => "cognition promotion",
        }
    }
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

fn format_generation_input(cognition_context: &str, draft: &GenerationDraft) -> String {
    format!(
        "Current cognition log:\n{}\n\nSpeech target: {}",
        cognition_context.trim(),
        draft.target.trim()
    )
}

fn push_generation_context(
    session: &mut Session,
    cognition_context: &str,
    draft: &GenerationDraft,
    generation_prompt: &str,
) {
    session.push_system(generation_prompt);
    session.push_user(format_generation_input(cognition_context, draft));
    if !draft.accumulated.is_empty() {
        session.push_assistant_text(draft.accumulated.clone());
    }
}

fn finish_speech_cognition_context(lines: Vec<String>, idle_for_secs: Option<u64>) -> String {
    let mut lines = lines;
    if let Some(seconds) = idle_for_secs {
        lines.push(format!("- I have been idle for {seconds} seconds."));
    }
    if lines.is_empty() {
        "none".to_owned()
    } else {
        lines.join("\n")
    }
}

fn format_abort_judge_input(cognition_context_at_start: &str, new_entries: &[String]) -> String {
    let mut out = format!(
        "Cognition log at the start of the current speech attempt:\n{}",
        cognition_context_at_start.trim()
    );
    out.push_str("\n\nNew cognition entries since speech started:");
    if new_entries.is_empty() {
        out.push_str("\n- none");
    } else {
        for entry in new_entries
            .iter()
            .map(|entry| entry.trim())
            .filter(|entry| !entry.is_empty())
        {
            out.push_str("\n- ");
            out.push_str(entry);
        }
    }
    out
}

enum GenerationStreamOutcome {
    Completed,
    Retry,
    Aborted,
}

pub struct SpeakModule {
    owner: nuillu_types::ModuleId,
    cognition_updates: CognitionLogUpdatedInbox,
    cognition_log: CognitionLogReader,
    memo: Memo,
    utterance: UtteranceWriter,
    llm: LlmAccess,
    scene: SceneReader,
    target_prompt: std::sync::OnceLock<String>,
    generation_prompt: std::sync::OnceLock<String>,
    abort_judge_prompt: std::sync::OnceLock<String>,
}

impl SpeakModule {
    pub fn new(
        cognition_updates: CognitionLogUpdatedInbox,
        cognition_log: CognitionLogReader,
        memo: Memo,
        utterance: UtteranceWriter,
        llm: LlmAccess,
        scene: SceneReader,
    ) -> Self {
        Self {
            owner: nuillu_types::ModuleId::new(<Self as Module>::id()).expect("speak id is valid"),
            cognition_updates,
            cognition_log,
            memo,
            utterance,
            llm,
            scene,
            target_prompt: std::sync::OnceLock::new(),
            generation_prompt: std::sync::OnceLock::new(),
            abort_judge_prompt: std::sync::OnceLock::new(),
        }
    }

    fn target_prompt(&self, cx: &nuillu_module::ActivateCx<'_>) -> &str {
        self.target_prompt.get_or_init(|| {
            nuillu_module::format_system_prompt(
                TARGET_SELECTION_PROMPT,
                cx.modules(),
                &self.owner,
                cx.identity_memories(),
                cx.now(),
            )
        })
    }

    fn generation_prompt(&self, cx: &nuillu_module::ActivateCx<'_>) -> &str {
        self.generation_prompt.get_or_init(|| {
            nuillu_module::format_system_prompt(
                GENERATION_PROMPT,
                cx.modules(),
                &self.owner,
                cx.identity_memories(),
                cx.now(),
            )
        })
    }

    fn abort_judge_prompt(&self, cx: &nuillu_module::ActivateCx<'_>) -> &str {
        self.abort_judge_prompt.get_or_init(|| {
            nuillu_module::format_system_prompt(
                ABORT_JUDGE_PROMPT,
                cx.modules(),
                &self.owner,
                cx.identity_memories(),
                cx.now(),
            )
        })
    }

    #[tracing::instrument(skip_all, err(Debug, level = "warn"))]
    async fn activate(
        &mut self,
        cx: &nuillu_module::ActivateCx<'_>,
        batch: &batch::SpeakBatch,
    ) -> Result<()> {
        let _update_count = batch.updates.len();
        let mut cognition_context = self.speech_cognition_context().await;
        let target = self.select_target(cx, &cognition_context).await?;
        let mut draft = GenerationDraft::new(self.utterance.next_generation_id(), target);

        loop {
            self.record_streaming_progress(&draft).await;
            match self
                .stream_generation(cx, cognition_context.clone(), &mut draft)
                .await?
            {
                GenerationStreamOutcome::Completed => return Ok(()),
                GenerationStreamOutcome::Retry => {
                    cognition_context = self.speech_cognition_context().await;
                }
                GenerationStreamOutcome::Aborted => {
                    cognition_context = self.speech_cognition_context().await;
                    let new_target = self.select_target(cx, &cognition_context).await?;
                    draft = GenerationDraft::new(self.utterance.next_generation_id(), new_target);
                }
            }
        }
    }

    async fn speech_cognition_context(&self) -> String {
        let snapshot = self.cognition_log.snapshot().await;
        let mut lines = Vec::new();
        for record in snapshot.logs() {
            for entry in &record.entries {
                let text = entry.text.trim();
                if !text.is_empty() {
                    lines.push(format!("- {text}"));
                }
            }
        }
        let idle_for_secs = snapshot
            .agentic_deadlock_marker()
            .map(|marker| marker.idle_for.as_secs());
        finish_speech_cognition_context(lines, idle_for_secs)
    }

    async fn select_target(
        &mut self,
        cx: &nuillu_module::ActivateCx<'_>,
        cognition_context: &str,
    ) -> Result<String> {
        let mut session = Session::new();
        session.push_system(self.target_prompt(cx));
        session.push_user(format!(
            "Current cognition log:\n{}",
            cognition_context.trim()
        ));

        let lutum = self.llm.lutum().await;
        let target_schema = self.scene.target_schema();
        let decision = SPEECH_TARGET_SCHEMA
            .scope(target_schema, async {
                let result = session
                    .structured_turn::<SpeakTargetDecision>(&lutum)
                    .collect()
                    .await
                    .context("speak target selection turn failed")?;
                let StructuredTurnOutcome::Structured(decision) = result.semantic else {
                    anyhow::bail!("speak target selection turn refused");
                };
                Ok::<_, anyhow::Error>(decision)
            })
            .await?;
        let target = decision.target.0.trim().to_owned();
        if target.is_empty() {
            anyhow::bail!("speak target selection produced an empty target");
        }
        Ok(target)
    }

    async fn stream_generation(
        &mut self,
        cx: &nuillu_module::ActivateCx<'_>,
        cognition_context: String,
        draft: &mut GenerationDraft,
    ) -> Result<GenerationStreamOutcome> {
        let stream_started_at = cx.now();
        let cognition_context_at_start = cognition_context.clone();

        let mut session = Session::new();
        push_generation_context(
            &mut session,
            &cognition_context,
            draft,
            self.generation_prompt(cx),
        );

        let lutum = self.llm.lutum().await;
        let mut stream = session
            .text_turn(&lutum)
            .stream()
            .await
            .context("speak generation stream failed")?;

        loop {
            tokio::select! {
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
                        Some(Ok(TextTurnEvent::Completed { .. })) | None => {
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
                    let _ = update.context("speak abort watch lost cognition update")?;
                    let _ = self.cognition_updates.take_ready_items()
                        .context("speak abort watch failed to drain cognition updates")?;

                    let new_entries = self
                        .new_cognition_entries_since(stream_started_at)
                        .await;
                    if new_entries.is_empty() {
                        continue;
                    }

                    if self
                        .judge_abort(cx, &cognition_context_at_start, &new_entries)
                        .await?
                    {
                        return Ok(GenerationStreamOutcome::Aborted);
                    }
                }
            }
        }
    }

    async fn new_cognition_entries_since(
        &self,
        threshold: chrono::DateTime<chrono::Utc>,
    ) -> Vec<String> {
        let snapshot = self.cognition_log.snapshot().await;
        snapshot
            .logs()
            .iter()
            .flat_map(|record| record.entries.iter())
            .filter(|entry| entry.at > threshold)
            .map(|entry| entry.text.clone())
            .collect()
    }

    async fn judge_abort(
        &mut self,
        cx: &nuillu_module::ActivateCx<'_>,
        cognition_context_at_start: &str,
        new_entries: &[String],
    ) -> Result<bool> {
        let mut session = Session::new();
        session.push_system(self.abort_judge_prompt(cx));
        session.push_user(format_abort_judge_input(
            cognition_context_at_start,
            new_entries,
        ));

        let lutum = self.llm.lutum().await;
        let result = session
            .structured_turn::<AbortJudgement>(&lutum)
            .collect()
            .await
            .context("speak abort-judge turn failed")?;
        let StructuredTurnOutcome::Structured(judgement) = result.semantic else {
            anyhow::bail!("speak abort-judge turn refused");
        };
        Ok(judgement.inform_now)
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
}

#[async_trait(?Send)]
impl Module for SpeakGateModule {
    type Batch = ActivationGateEvent<SpeakModule>;

    fn id() -> &'static str {
        "speak-gate"
    }

    fn role_description() -> &'static str {
        "Classifies pending speak activations as ready to speak, speech-worthy but missing evidence, or no speech needed. If the same missing-evidence state repeats after stuckness, it may force-allow Speak while recording the missing evidence in its typed memo."
    }

    async fn next_batch(&mut self) -> Result<Self::Batch> {
        SpeakGateModule::next_batch(self).await
    }

    async fn activate(
        &mut self,
        cx: &nuillu_module::ActivateCx<'_>,
        batch: &Self::Batch,
    ) -> Result<()> {
        SpeakGateModule::activate(self, cx, batch).await
    }
}

#[async_trait(?Send)]
impl Module for SpeakModule {
    type Batch = batch::SpeakBatch;

    fn id() -> &'static str {
        "speak"
    }

    fn role_description() -> &'static str {
        "Emits the agent's spoken utterances into its world after cognition-log updates pass activation gates. It cannot inspect memo logs or query results directly, so missing evidence not promoted to cognition before speak-gate allows activation will lead to guessed speech."
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

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::rc::Rc;
    use std::sync::Arc;

    use lutum::{
        AssistantInputItem, FinishReason, InputMessageRole, MessageContent, MockLlmAdapter,
        MockStructuredScenario, MockTextScenario, ModelInputItem, RawStructuredTurnEvent,
        RawTextTurnEvent, SharedPoolBudgetManager, SharedPoolBudgetOptions, Usage,
    };
    use nuillu_blackboard::{
        ActivationRatio, Blackboard, BlackboardCommand, CognitionLogEntry, IdentityMemoryRecord,
        ModuleConfig, ResourceAllocation,
    };
    use nuillu_module::ports::{
        Clock, NoopCognitionLogRepository, NoopFileSearchProvider, NoopMemoryStore, PortError,
        SystemClock, Utterance, UtteranceSink,
    };
    use nuillu_module::{
        AttentionControlRequestInbox, CapabilityProviderPorts, CapabilityProviders,
        CognitionLogUpdated, LutumTiers, ModuleRegistry, Participant, session_compaction_cutoff,
    };
    use nuillu_types::{MemoryContent, MemoryIndex};

    use super::*;

    fn test_session() -> Session {
        Session::new()
    }

    fn test_caps_with_adapter(
        blackboard: Blackboard,
        adapter: MockLlmAdapter,
    ) -> CapabilityProviders {
        test_caps_with_adapter_and_sink(
            blackboard,
            adapter,
            Arc::new(nuillu_module::ports::NoopUtteranceSink),
        )
    }

    fn test_caps_with_adapter_and_sink(
        blackboard: Blackboard,
        adapter: MockLlmAdapter,
        utterance_sink: Arc<dyn UtteranceSink>,
    ) -> CapabilityProviders {
        let adapter = Arc::new(adapter);
        let budget = SharedPoolBudgetManager::new(SharedPoolBudgetOptions::default());
        let lutum = Lutum::new(adapter, budget);
        CapabilityProviders::new(CapabilityProviderPorts {
            blackboard,
            cognition_log_port: Arc::new(NoopCognitionLogRepository),
            primary_memory_store: Arc::new(NoopMemoryStore),
            memory_replicas: Vec::new(),
            file_search: Arc::new(NoopFileSearchProvider),
            utterance_sink,
            clock: Arc::new(SystemClock),
            tiers: LutumTiers {
                cheap: lutum.clone(),
                default: lutum.clone(),
                premium: lutum,
            },
        })
    }

    fn tool_test_allocation() -> ResourceAllocation {
        let mut allocation = ResourceAllocation::default();
        for module in [
            builtin::speak_gate(),
            builtin::attention_controller(),
            builtin::query_vector(),
            builtin::self_model(),
            builtin::sensory(),
            builtin::speak(),
        ] {
            allocation.set(module.clone(), ModuleConfig::default());
            allocation.set_activation(module, ActivationRatio::ONE);
        }
        allocation
    }

    struct CapturingUtteranceSink {
        completed: Rc<RefCell<Vec<(String, String)>>>,
        done: RefCell<Option<tokio::sync::oneshot::Sender<()>>>,
    }

    #[async_trait(?Send)]
    impl UtteranceSink for CapturingUtteranceSink {
        async fn on_complete(&self, utterance: Utterance) -> Result<(), PortError> {
            self.completed
                .borrow_mut()
                .push((utterance.target, utterance.text));
            if let Some(done) = self.done.borrow_mut().take() {
                let _ = done.send(());
            }
            Ok(())
        }
    }

    fn test_policy() -> nuillu_blackboard::ModulePolicy {
        nuillu_blackboard::ModulePolicy::new(
            nuillu_types::ReplicaCapRange::new(0, 0).unwrap(),
            nuillu_blackboard::Bpm::from_f64(60.0)..=nuillu_blackboard::Bpm::from_f64(60.0),
            nuillu_blackboard::linear_ratio_fn,
        )
    }

    macro_rules! noop_stub {
        ($name:ident, $id:literal) => {
            struct $name;

            #[async_trait(?Send)]
            impl Module for $name {
                type Batch = ();

                fn id() -> &'static str {
                    $id
                }

                fn role_description() -> &'static str {
                    "test stub"
                }

                async fn next_batch(&mut self) -> Result<Self::Batch> {
                    std::future::pending().await
                }

                async fn activate(
                    &mut self,
                    _cx: &nuillu_module::ActivateCx<'_>,
                    _batch: &Self::Batch,
                ) -> Result<()> {
                    Ok(())
                }
            }
        };
    }

    noop_stub!(SpeakGateStub, "speak-gate");
    noop_stub!(SpeakStub, "speak");
    noop_stub!(AttentionControllerStub, "attention-controller");

    struct GateToolFixture {
        gate: SpeakGateModule,
        blackboard: Blackboard,
        attention_control_inbox: AttentionControlRequestInbox,
    }

    async fn gate_tool_fixture() -> GateToolFixture {
        gate_tool_fixture_with_adapter(MockLlmAdapter::new()).await
    }

    async fn gate_tool_fixture_with_adapter(adapter: MockLlmAdapter) -> GateToolFixture {
        let blackboard = Blackboard::with_allocation(tool_test_allocation());
        let caps = test_caps_with_adapter(blackboard.clone(), adapter);

        let gate_cell = Rc::new(RefCell::new(None));
        let attention_control_inbox_cell = Rc::new(RefCell::new(None));

        let gate_sink = Rc::clone(&gate_cell);
        let attention_control_inbox_sink = Rc::clone(&attention_control_inbox_cell);

        let _modules = ModuleRegistry::new()
            .register(test_policy(), move |caps| {
                *gate_sink.borrow_mut() = Some(SpeakGateModule::new(
                    caps.activation_gate_for::<SpeakModule>(),
                    caps.cognition_log_reader(),
                    caps.blackboard_reader(),
                    caps.attention_control_mailbox(),
                    caps.typed_memo::<SpeakGateMemo>(),
                    caps.llm_access(),
                ));
                SpeakGateStub
            })
            .unwrap()
            .register(test_policy(), move |caps| {
                *attention_control_inbox_sink.borrow_mut() = Some(caps.attention_control_inbox());
                AttentionControllerStub
            })
            .unwrap()
            .build(&caps)
            .await
            .unwrap();

        GateToolFixture {
            gate: gate_cell.borrow_mut().take().unwrap(),
            blackboard,
            attention_control_inbox: attention_control_inbox_cell.borrow_mut().take().unwrap(),
        }
    }

    fn structured_usage(input_tokens: u64) -> Usage {
        Usage {
            input_tokens,
            ..Usage::zero()
        }
    }

    fn finished_decision_scenario(input_tokens: u64, rationale: &str) -> MockStructuredScenario {
        MockStructuredScenario::events(vec![
            Ok(RawStructuredTurnEvent::Started {
                request_id: Some("gate-finished".into()),
                model: "mock".into(),
            }),
            Ok(RawStructuredTurnEvent::StructuredOutputChunk {
                json_delta: serde_json::json!({
                    "wants_to_speak": false,
                    "wait_for_evidence": false,
                    "rationale": rationale,
                    "evidence_gaps": [],
                })
                .to_string(),
            }),
            Ok(RawStructuredTurnEvent::Completed {
                request_id: Some("gate-finished".into()),
                finish_reason: FinishReason::Stop,
                usage: structured_usage(input_tokens),
            }),
        ])
    }

    fn target_decision_scenario(target: &str) -> MockStructuredScenario {
        MockStructuredScenario::events(vec![
            Ok(RawStructuredTurnEvent::Started {
                request_id: Some("target".into()),
                model: "mock".into(),
            }),
            Ok(RawStructuredTurnEvent::StructuredOutputChunk {
                json_delta: serde_json::json!({
                    "target": target,
                })
                .to_string(),
            }),
            Ok(RawStructuredTurnEvent::Completed {
                request_id: Some("target".into()),
                finish_reason: FinishReason::Stop,
                usage: Usage::zero(),
            }),
        ])
    }

    fn generation_text_scenario(text: &str) -> MockTextScenario {
        MockTextScenario::events(vec![
            Ok(RawTextTurnEvent::Started {
                request_id: Some("speak-text".into()),
                model: "mock".into(),
            }),
            Ok(RawTextTurnEvent::TextDelta { delta: text.into() }),
            Ok(RawTextTurnEvent::Completed {
                request_id: Some("speak-text".into()),
                finish_reason: FinishReason::Stop,
                usage: Usage::zero(),
            }),
        ])
    }

    fn tool_call_decision_scenario(input_tokens: u64) -> MockStructuredScenario {
        MockStructuredScenario::events(vec![
            Ok(RawStructuredTurnEvent::Started {
                request_id: Some("gate-tool".into()),
                model: "mock".into(),
            }),
            Ok(RawStructuredTurnEvent::ToolCallChunk {
                id: "memory-1".into(),
                name: "query_memory".into(),
                arguments_json_delta: serde_json::json!({
                    "question": "What should I remember?"
                })
                .to_string(),
            }),
            Ok(RawStructuredTurnEvent::Completed {
                request_id: Some("gate-tool".into()),
                finish_reason: FinishReason::ToolCall,
                usage: structured_usage(input_tokens),
            }),
        ])
    }

    fn summary_text_scenario(summary: &str) -> MockTextScenario {
        MockTextScenario::events(vec![
            Ok(RawTextTurnEvent::Started {
                request_id: Some("gate-compact".into()),
                model: "mock".into(),
            }),
            Ok(RawTextTurnEvent::TextDelta {
                delta: summary.into(),
            }),
            Ok(RawTextTurnEvent::Completed {
                request_id: Some("gate-compact".into()),
                finish_reason: FinishReason::Stop,
                usage: Usage::zero(),
            }),
        ])
    }

    #[test]
    fn session_compaction_config_defaults_to_16k_and_80_percent() {
        assert_eq!(
            SpeakGateSessionCompactionConfig::default(),
            SpeakGateSessionCompactionConfig {
                input_token_threshold: 16_000,
                prefix_ratio: 0.8,
            }
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn session_compaction_builder_replaces_config() {
        let fixture = gate_tool_fixture().await;
        let config = SpeakGateSessionCompactionConfig {
            input_token_threshold: 42,
            prefix_ratio: 0.5,
        };
        let gate = fixture.gate.with_session_compaction(config);

        assert_eq!(gate.session_compaction, config);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn activation_seeds_system_then_identity_memories() {
        let mut fixture = gate_tool_fixture().await;
        let modules = vec![
            (builtin::speak_gate(), SpeakGateModule::role_description()),
            (builtin::speak(), SpeakModule::role_description()),
        ];
        let identity_memories = vec![IdentityMemoryRecord {
            index: MemoryIndex::new("identity-1"),
            content: MemoryContent::new("The agent is named Nuillu."),
            occurred_at: None,
        }];
        let lutum = fixture.gate.llm.lutum().await;
        let cx = nuillu_module::ActivateCx::new(
            &modules,
            &identity_memories,
            lutum.lutum().clone(),
            SystemClock.now(),
        );

        fixture.gate.ensure_session_seeded(&cx);

        let items = fixture.gate.session.input().items();
        let ModelInputItem::Message { role, content } = &items[0] else {
            panic!("expected system prompt first");
        };
        assert_eq!(role, &InputMessageRole::System);
        let [MessageContent::Text(system)] = content.as_slice() else {
            panic!("expected system prompt text");
        };
        assert!(system.contains("You are the speak-gate module"));
        assert!(!system.contains("The agent is named Nuillu."));

        let ModelInputItem::Assistant(AssistantInputItem::Text(identity)) = &items[1] else {
            panic!("expected identity memories as assistant text second");
        };
        assert!(identity.contains("The agent is named Nuillu."));
    }

    #[test]
    fn session_compaction_cutoff_uses_ratio_and_keeps_raw_suffix() {
        assert_eq!(session_compaction_cutoff(10, 0.8), Some(8));
        assert_eq!(session_compaction_cutoff(10, 1.0), Some(9));
        assert_eq!(session_compaction_cutoff(1, 0.8), None);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn high_input_finished_decision_compacts_session_prefix() {
        let adapter = MockLlmAdapter::new()
            .with_structured_scenario(finished_decision_scenario(16_001, "wait"))
            .with_text_scenario(summary_text_scenario("old gate history summarized"));
        let mut fixture = gate_tool_fixture_with_adapter(adapter).await;
        for index in 0..10 {
            fixture.gate.session.push_user(format!("history-{index}"));
        }

        let lutum = fixture.gate.llm.lutum().await;
        let decision = fixture
            .gate
            .run_decision_turn(&lutum, &lutum, false)
            .await
            .unwrap();

        assert!(!decision.wants_to_speak);
        assert!(!decision.wait_for_evidence);
        let items = fixture.gate.session.input().items();
        let ModelInputItem::Message { role, content } = &items[0] else {
            panic!("expected compacted system message");
        };
        assert_eq!(role, &InputMessageRole::System);
        let [MessageContent::Text(summary)] = content.as_slice() else {
            panic!("expected compacted summary text");
        };
        assert!(summary.starts_with(COMPACTED_SPEAK_GATE_SESSION_PREFIX));
        assert!(summary.contains("old gate history summarized"));

        let user_texts = items
            .iter()
            .filter_map(|item| match item {
                ModelInputItem::Message {
                    role: InputMessageRole::User,
                    content,
                } => {
                    let [MessageContent::Text(text)] = content.as_slice() else {
                        panic!("expected one text content item");
                    };
                    Some(text.as_str())
                }
                _ => None,
            })
            .collect::<Vec<_>>();
        assert!(!user_texts.contains(&"history-0"));
        assert!(user_texts.contains(&"history-8"));
        assert!(user_texts.contains(&"history-9"));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn threshold_input_finished_decision_does_not_compact() {
        let adapter = MockLlmAdapter::new()
            .with_structured_scenario(finished_decision_scenario(16_000, "under threshold"))
            .with_text_scenario(summary_text_scenario("unexpected summary"));
        let mut fixture = gate_tool_fixture_with_adapter(adapter).await;
        for index in 0..10 {
            fixture.gate.session.push_user(format!("history-{index}"));
        }

        let lutum = fixture.gate.llm.lutum().await;
        let decision = fixture
            .gate
            .run_decision_turn(&lutum, &lutum, false)
            .await
            .unwrap();

        assert!(!decision.wants_to_speak);
        assert!(!decision.wait_for_evidence);
        let items = fixture.gate.session.input().items();
        let ModelInputItem::Message { role, content } = &items[0] else {
            panic!("expected original first history item");
        };
        assert_eq!(role, &InputMessageRole::User);
        let [MessageContent::Text(text)] = content.as_slice() else {
            panic!("expected original history text");
        };
        assert_eq!(text, "history-0");

        assert!(!matches!(
            &items[0],
            ModelInputItem::Message { content, .. }
                if matches!(
                    content.as_slice(),
                    [MessageContent::Text(text)]
                        if text.starts_with(COMPACTED_SPEAK_GATE_SESSION_PREFIX)
                )
        ));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn tool_round_compacts_after_commit_before_next_round() {
        let adapter = MockLlmAdapter::new()
            .with_structured_scenario(tool_call_decision_scenario(16_001))
            .with_structured_scenario(finished_decision_scenario(1, "tool result committed"))
            .with_text_scenario(summary_text_scenario("tool round history summarized"));
        let mut fixture = gate_tool_fixture_with_adapter(adapter).await;
        for index in 0..10 {
            fixture.gate.session.push_user(format!("history-{index}"));
        }

        let lutum = fixture.gate.llm.lutum().await;
        let decision = fixture
            .gate
            .run_decision_turn(&lutum, &lutum, false)
            .await
            .unwrap();

        assert_eq!(decision.rationale, "tool result committed");
        let items = fixture.gate.session.input().items();
        let ModelInputItem::Message { role, content } = &items[0] else {
            panic!("expected compacted system message");
        };
        assert_eq!(role, &InputMessageRole::System);
        let [MessageContent::Text(summary)] = content.as_slice() else {
            panic!("expected compacted summary text");
        };
        assert!(summary.contains("tool round history summarized"));
    }

    #[test]
    fn fresh_generation_omits_assistant_prefill() {
        let draft = GenerationDraft::new(7, "Koro");
        let mut session = test_session();

        push_generation_context(&mut session, "none", &draft, GENERATION_PROMPT);
        let items = session.input().items();

        assert_eq!(draft.generation_id, 7);
        assert_eq!(draft.sequence, 0);
        assert_eq!(items.len(), 2);
        assert!(matches!(
            &items[0],
            ModelInputItem::Message {
                role: InputMessageRole::System,
                ..
            }
        ));
        let ModelInputItem::Message {
            role: InputMessageRole::User,
            content,
        } = &items[1]
        else {
            panic!("expected user generation context");
        };
        let [MessageContent::Text(text)] = content.as_slice() else {
            panic!("expected one text content item");
        };
        assert!(text.contains("Current cognition log:\nnone"));
        assert!(text.contains("Speech target: Koro"));
        assert!(!text.contains("allocation"));
        assert!(!text.contains("partial_utterance"));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn speak_selects_target_from_cognition_log_before_streaming() {
        let adapter = MockLlmAdapter::new()
            .with_structured_scenario(target_decision_scenario("Koro"))
            .with_text_scenario(generation_text_scenario("Koro, stay close."));
        let mut allocation = ResourceAllocation::default();
        allocation.set(builtin::speak(), ModuleConfig::default());
        allocation.set_activation(builtin::speak(), ActivationRatio::ONE);
        let blackboard = Blackboard::with_allocation(allocation);
        let completed = Rc::new(RefCell::new(Vec::new()));
        let (done_tx, done_rx) = tokio::sync::oneshot::channel();
        let sink = Arc::new(CapturingUtteranceSink {
            completed: Rc::clone(&completed),
            done: RefCell::new(Some(done_tx)),
        });
        let caps = test_caps_with_adapter_and_sink(blackboard.clone(), adapter, sink);
        caps.scene().set([Participant::new("Koro")]);
        let module_cell = Rc::new(RefCell::new(None));
        let module_sink = Rc::clone(&module_cell);

        let _modules = ModuleRegistry::new()
            .register(test_policy(), move |caps| {
                *module_sink.borrow_mut() = Some(SpeakModule::new(
                    caps.cognition_log_updated_inbox(),
                    caps.cognition_log_reader(),
                    caps.memo(),
                    caps.utterance_writer(),
                    caps.llm_access(),
                    caps.scene_reader(),
                ));
                SpeakStub
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
            &identity_memories,
            compaction_lutum.lutum().clone(),
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
    }

    #[test]
    fn self_model_tool_is_prompted_and_available_only_when_registered() {
        let without_self_model = readiness_gate_prompt(false);
        assert!(!without_self_model.contains("query_self_model"));
        assert!(
            !speak_gate_tool_selectors(false)
                .iter()
                .any(|tool| matches!(tool, &SpeakGateToolsSelector::QuerySelfModel))
        );

        let with_self_model = readiness_gate_prompt(true);
        assert!(with_self_model.contains("query_self_model"));
        assert!(
            speak_gate_tool_selectors(true)
                .iter()
                .any(|tool| matches!(tool, &SpeakGateToolsSelector::QuerySelfModel))
        );
    }

    #[test]
    fn current_speech_progress_pushes_streaming_tail_as_ephemeral_assistant_turn() {
        let mut session = test_session();
        let partial = format!("drop-{}", "keep".repeat(80));
        let progress = UtteranceProgress::streaming(5, 1, "Mika", &partial);

        push_current_speech_progress(&mut session, Some(&progress));

        let items = session.input().items();
        let [ModelInputItem::Assistant(AssistantInputItem::Text(text))] = items else {
            panic!("expected one assistant progress item");
        };
        let tail = text
            .strip_prefix("I'm speaking: ")
            .expect("expected progress prefix");
        assert_eq!(tail.chars().count(), CURRENT_SPEECH_PROGRESS_TAIL_CHARS);
        assert!(!tail.contains("drop-"));
        assert!(tail.ends_with("keep"));
    }

    #[test]
    fn current_speech_progress_omits_turn_when_not_streaming() {
        let mut session = test_session();
        let completed = UtteranceProgress::completed(5, 1, "Mika", "done");

        push_current_speech_progress(&mut session, None);
        push_current_speech_progress(&mut session, Some(&completed));

        assert!(session.input().items().is_empty());
    }

    fn speak_gate_decision(
        wants_to_speak: bool,
        wait_for_evidence: bool,
        evidence_gaps: Vec<EvidenceGap>,
    ) -> SpeakGateDecision {
        SpeakGateDecision {
            wants_to_speak,
            wait_for_evidence,
            rationale: "test rationale".into(),
            evidence_gaps,
        }
    }

    fn memory_gap() -> EvidenceGap {
        EvidenceGap {
            source: EvidenceGapSource::Memory,
            question: " Which body fact? ".into(),
            needed_fact: " frog body ".into(),
        }
    }

    #[tokio::test(flavor = "current_thread")]
    async fn speak_gate_decisions_normalize_to_three_memo_kinds() {
        let mut fixture = gate_tool_fixture().await;

        let ready = fixture.gate.build_memo_from_decision(
            speak_gate_decision(true, false, Vec::new()),
            Some(1),
            false,
        );
        assert_eq!(ready.kind, SpeakGateMemoKind::WantsToSpeak);
        assert_eq!(gate_vote_from_memo(&ready), ActivationGateVote::Allow);

        let missing = fixture.gate.build_memo_from_decision(
            speak_gate_decision(true, true, vec![memory_gap()]),
            Some(1),
            false,
        );
        assert_eq!(missing.kind, SpeakGateMemoKind::WantsToSpeakMissingEvidence);
        assert_eq!(missing.evidence_gaps[0].question, "Which body fact?");
        assert!(!missing.forced);
        assert_eq!(gate_vote_from_memo(&missing), ActivationGateVote::Suppress);

        let no_need = fixture.gate.build_memo_from_decision(
            speak_gate_decision(false, true, vec![memory_gap()]),
            Some(1),
            true,
        );
        assert_eq!(no_need.kind, SpeakGateMemoKind::NoNeedToSpeak);
        assert!(!no_need.forced);
        assert_eq!(gate_vote_from_memo(&no_need), ActivationGateVote::Suppress);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn repeated_missing_evidence_with_stuckness_forces_allow() {
        let mut fixture = gate_tool_fixture().await;

        let first = fixture.gate.build_memo_from_decision(
            speak_gate_decision(true, true, vec![memory_gap()]),
            Some(3),
            false,
        );
        assert_eq!(first.kind, SpeakGateMemoKind::WantsToSpeakMissingEvidence);
        assert!(!first.forced);
        assert_eq!(gate_vote_from_memo(&first), ActivationGateVote::Suppress);

        let second = fixture.gate.build_memo_from_decision(
            speak_gate_decision(true, true, vec![memory_gap()]),
            Some(3),
            true,
        );
        assert_eq!(second.kind, SpeakGateMemoKind::WantsToSpeakMissingEvidence);
        assert!(second.forced);
        assert_eq!(gate_vote_from_memo(&second), ActivationGateVote::Allow);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn speak_gate_typed_memo_preserves_payload_and_plaintext() {
        let fixture = gate_tool_fixture().await;
        let payload = SpeakGateMemo {
            kind: SpeakGateMemoKind::WantsToSpeak,
            rationale: "ready".into(),
            evidence_gaps: Vec::new(),
            forced: false,
            latest_cognition_index: Some(9),
        };

        fixture
            .gate
            .memo
            .write(payload.clone(), render_speak_gate_memo(&payload))
            .await;

        let owner = ModuleInstanceId::new(builtin::speak_gate(), ReplicaIndex::ZERO);
        let logs = fixture
            .blackboard
            .typed_memo_logs::<SpeakGateMemo>(&owner)
            .await;
        assert_eq!(logs.len(), 1);
        assert_eq!(logs[0].data(), &payload);
        assert!(logs[0].content.contains("Speak decision: wants to speak"));
        assert!(serde_json::from_str::<serde_json::Value>(&logs[0].content).is_err());
    }

    #[test]
    fn requested_evidence_guard_turns_same_turn_speak_into_wait() {
        let decision = SpeakGateDecision {
            wants_to_speak: true,
            wait_for_evidence: false,
            rationale: "tool request is enough".into(),
            evidence_gaps: Vec::new(),
        };

        let guarded = apply_requested_evidence_guard(decision, &[EvidenceGapSource::Memory]);

        assert!(guarded.wants_to_speak);
        assert!(guarded.wait_for_evidence);
        assert!(
            guarded
                .rationale
                .contains("Waiting after requesting evidence from memory")
        );
        assert_eq!(guarded.evidence_gaps.len(), 1);
        assert!(matches!(
            guarded.evidence_gaps[0].source,
            EvidenceGapSource::CognitionPromotion
        ));
    }

    fn cognition_history_texts(session: &Session) -> Vec<String> {
        session
            .input()
            .items()
            .iter()
            .filter_map(|item| {
                let ModelInputItem::Message { role, content } = item else {
                    return None;
                };
                if role != &InputMessageRole::User {
                    return None;
                }
                let [MessageContent::Text(text)] = content.as_slice() else {
                    return None;
                };
                text.strip_prefix("New cognition log item:\n")
                    .map(ToOwned::to_owned)
            })
            .collect()
    }

    #[tokio::test(flavor = "current_thread")]
    async fn speak_gate_cognition_history_uses_reader_unread_cursor() {
        let mut fixture = gate_tool_fixture().await;
        let stream = ModuleInstanceId::new(builtin::cognition_gate(), ReplicaIndex::ZERO);
        let clock = SystemClock;

        fixture
            .blackboard
            .apply(BlackboardCommand::AppendCognitionLog {
                source: stream.clone(),
                entry: CognitionLogEntry {
                    at: clock.now(),
                    text: "first".into(),
                },
            })
            .await;

        let first = fixture.gate.cognition_log.unread_events().await;
        push_cognition_history(&mut fixture.gate.session, &first);
        assert_eq!(
            cognition_history_texts(&fixture.gate.session),
            vec!["first"]
        );

        let already_seen = fixture.gate.cognition_log.unread_events().await;
        push_cognition_history(&mut fixture.gate.session, &already_seen);
        assert_eq!(
            cognition_history_texts(&fixture.gate.session),
            vec!["first"]
        );

        fixture
            .blackboard
            .apply(BlackboardCommand::AppendCognitionLog {
                source: stream,
                entry: CognitionLogEntry {
                    at: clock.now(),
                    text: "second".into(),
                },
            })
            .await;

        let second = fixture.gate.cognition_log.unread_events().await;
        push_cognition_history(&mut fixture.gate.session, &second);
        assert_eq!(
            cognition_history_texts(&fixture.gate.session),
            vec!["first", "second"]
        );
    }

    #[test]
    fn resumed_generation_keeps_id_sequence_and_pushes_assistant_prefill() {
        let mut draft = GenerationDraft::new(11, "Koro");
        let mut session = test_session();

        assert_eq!(draft.push_delta("hello "), 0);
        assert_eq!(draft.push_delta("world"), 1);
        push_generation_context(&mut session, "none", &draft, GENERATION_PROMPT);
        let items = session.input().items();

        assert_eq!(draft.generation_id, 11);
        assert_eq!(draft.sequence, 2);
        assert_eq!(draft.accumulated, "hello world");
        assert_eq!(items.len(), 3);
        assert!(matches!(
            &items[0],
            ModelInputItem::Message {
                role: InputMessageRole::System,
                ..
            }
        ));
        assert!(matches!(
            &items[1],
            ModelInputItem::Message {
                role: InputMessageRole::User,
                ..
            }
        ));
        let ModelInputItem::Assistant(AssistantInputItem::Text(text)) = &items[2] else {
            panic!("expected assistant prefill");
        };
        assert_eq!(text, "hello world");
    }

    #[test]
    fn wait_decision_renders_evidence_gaps_in_free_form_memo() {
        let payload = SpeakGateMemo {
            kind: SpeakGateMemoKind::WantsToSpeakMissingEvidence,
            rationale: "missing body fact".into(),
            evidence_gaps: vec![EvidenceGap {
                source: EvidenceGapSource::Memory,
                question: "What body should I report?".into(),
                needed_fact: "frog body".into(),
            }],
            forced: false,
            latest_cognition_index: Some(7),
        };

        let memo = render_speak_gate_memo(&payload);

        assert!(memo.contains("Speak decision: wants to speak but missing evidence"));
        assert!(memo.contains("Forced allow: no"));
        assert!(memo.contains("Rationale: missing body fact"));
        assert!(memo.contains(
            "Source: memory; question: What body should I report?; needed fact: frog body"
        ));
        assert!(serde_json::from_str::<serde_json::Value>(&memo).is_err());
    }

    #[test]
    fn gate_vote_from_memo_allows_only_ready_or_forced_speak_decisions() {
        let valid = SpeakGateMemo {
            kind: SpeakGateMemoKind::WantsToSpeak,
            rationale: "ready".into(),
            evidence_gaps: Vec::new(),
            forced: false,
            latest_cognition_index: Some(1),
        };

        assert_eq!(gate_vote_from_memo(&valid), ActivationGateVote::Allow);

        let waiting = SpeakGateMemo {
            kind: SpeakGateMemoKind::WantsToSpeakMissingEvidence,
            ..valid.clone()
        };
        assert_eq!(gate_vote_from_memo(&waiting), ActivationGateVote::Suppress);

        let forced_waiting = SpeakGateMemo {
            forced: true,
            ..waiting.clone()
        };
        assert_eq!(
            gate_vote_from_memo(&forced_waiting),
            ActivationGateVote::Allow
        );

        let silent = SpeakGateMemo {
            kind: SpeakGateMemoKind::NoNeedToSpeak,
            forced: true,
            ..valid
        };
        assert_eq!(gate_vote_from_memo(&silent), ActivationGateVote::Suppress);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn evidence_tools_publish_requests_without_polling() {
        let mut fixture = gate_tool_fixture().await;

        let mut memory_requests = HashSet::new();
        let memory_output = fixture
            .gate
            .query_memory(
                QueryMemoryArgs {
                    question: "  body fact?  ".into(),
                },
                &mut memory_requests,
            )
            .await
            .unwrap();
        let memory_request = fixture.attention_control_inbox.next_item().await.unwrap();
        assert!(memory_output.requested);
        assert!(!memory_output.duplicate);
        assert_eq!(memory_request.sender.module, builtin::speak_gate());
        assert_eq!(
            memory_request.body,
            AttentionControlRequest::query_with_reason(
                "body fact?",
                "speak-gate query_memory evidence tool"
            )
        );

        let duplicate_memory_output = fixture
            .gate
            .query_memory(
                QueryMemoryArgs {
                    question: "body fact?".into(),
                },
                &mut memory_requests,
            )
            .await
            .unwrap();
        assert!(!duplicate_memory_output.requested);
        assert!(duplicate_memory_output.duplicate);
        assert!(
            fixture
                .attention_control_inbox
                .take_ready_items()
                .unwrap()
                .items
                .is_empty()
        );

        let mut self_model_requests = HashSet::new();
        let self_model_output = fixture
            .gate
            .query_self_model(
                QuerySelfModelArgs {
                    question: " current role? ".into(),
                },
                &mut self_model_requests,
            )
            .await
            .unwrap();
        let self_model_request = fixture.attention_control_inbox.next_item().await.unwrap();
        assert!(self_model_output.requested);
        assert!(!self_model_output.duplicate);
        assert_eq!(self_model_request.sender.module, builtin::speak_gate());
        assert_eq!(
            self_model_request.body,
            AttentionControlRequest::self_model_with_reason(
                "current role?",
                "speak-gate query_self_model evidence tool"
            )
        );

        let mut sensory_detail_requests = HashSet::new();
        let sensory_output = fixture
            .gate
            .query_sensory_detail(
                QuerySensoryDetailArgs {
                    question: " what was just heard? ".into(),
                },
                &mut sensory_detail_requests,
            )
            .await
            .unwrap();
        let sensory_request = fixture.attention_control_inbox.next_item().await.unwrap();
        assert!(sensory_output.requested);
        assert!(!sensory_output.duplicate);
        assert_eq!(sensory_request.sender.module, builtin::speak_gate());
        assert_eq!(
            sensory_request.body,
            AttentionControlRequest::sensory_detail_with_reason(
                "what was just heard?",
                "speak-gate query_sensory_detail evidence tool"
            )
        );
    }
}
