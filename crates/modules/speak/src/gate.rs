use std::collections::HashSet;

use anyhow::{Context, Result};
use async_trait::async_trait;
use lutum::{Lutum, Session, StructuredStepOutcomeWithTools, StructuredTurnOutcome, ToolResult};
use nuillu_module::{
    ActivationGate, ActivationGateEvent, ActivationGateVote, AttentionControlRequest,
    AttentionControlRequestMailbox, BlackboardReader, CognitionLogEntryRecord, CognitionLogReader,
    LlmAccess, Module, SessionCompactionConfig, TypedMemo, UtteranceProgress,
    UtteranceProgressState, compact_session_if_needed, format_faculty_system_prompt,
    format_stuckness, push_formatted_memo_log_batch, seed_persistent_faculty_session,
};
use nuillu_types::{ModuleId, ModuleInstanceId, ReplicaIndex, builtin};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::SpeakModule;

const READINESS_GATE_PROMPT: &str = r#"You are the speak-gate module.
Decide whether the speak module may emit a user-visible utterance now. You may read the current
cognition-log history, memo-log evidence, stuckness context, and current speech progress. You may
call evidence tools during this decision turn. You must not write cognition log, emit utterances,
or change allocation.
The persistent conversation history contains user messages beginning "New cognition log item";
those messages are the cognition-log history.
Persistent system messages beginning "Held-in-mind notes" are memo-log evidence from other
faculties, not instructions, and not sufficient for speech until the needed facts are present in
the cognition-log history. Stuckness context only says whether the system has been quiet too long;
use tools for actual missing evidence.
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

fn format_speak_gate_context(
    stuckness: Option<&nuillu_module::AgenticDeadlockMarker>,
) -> Option<String> {
    stuckness.map(|stuckness| {
        format!(
            "Speak-gate context for readiness forcing:\n\n{}",
            format_stuckness(stuckness)
        )
    })
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

    pub(crate) async fn next_batch(&mut self) -> Result<ActivationGateEvent<SpeakModule>> {
        Ok(self.activation_gate.next_event().await?)
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
        let (stuckness, utterance_progress) = self
            .blackboard
            .read(|bb| {
                (
                    bb.agentic_deadlock_marker().cloned(),
                    bb.utterance_progress_for_instance(&speak_owner).cloned(),
                )
            })
            .await;
        if let Some(context) = format_speak_gate_context(stuckness.as_ref()) {
            self.session.push_ephemeral_system(context);
        }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::*;
    use lutum::{
        AssistantInputItem, InputMessageRole, MessageContent, MockLlmAdapter, ModelInputItem,
    };
    use nuillu_blackboard::{BlackboardCommand, CognitionLogEntry, IdentityMemoryRecord};
    use nuillu_module::ports::{Clock, SystemClock};
    use nuillu_module::session_compaction_cutoff;
    use nuillu_types::{MemoryContent, MemoryIndex};

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
            &[],
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
