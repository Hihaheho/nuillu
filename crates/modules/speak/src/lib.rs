use std::collections::HashSet;

use anyhow::{Context, Result};
use async_trait::async_trait;
use futures::StreamExt;
use lutum::{
    AssistantInputItem, InputMessageRole, Lutum, MessageContent, ModelInput, ModelInputItem,
    RawJson, Session, StructuredStepOutcomeWithTools, StructuredTurnOutcome, TextTurnEvent,
    ToolResult, TurnRole,
};
use nuillu_module::{
    ActivationGate, ActivationGateEvent, ActivationGateVote, AttentionControlRequest,
    AttentionControlRequestMailbox, BlackboardReader, CognitionLogEntryRecord, CognitionLogReader,
    CognitionLogUpdatedInbox, LlmAccess, Memo, Module, ModuleRunStatus, ModuleStatusReader,
    SceneReader, UtteranceProgress, UtteranceWriter, push_unread_memo_logs,
};
use nuillu_types::{ModuleId, ModuleInstanceId, ReplicaIndex, builtin};
use schemars::{JsonSchema, Schema, SchemaGenerator};
use serde::{Deserialize, Serialize};
use std::borrow::Cow;

mod batch;

const READINESS_GATE_PROMPT: &str = r#"You are the speak-gate module.
Decide whether the speak module may emit a user-visible utterance now. You may read the current
cognition-log set, blackboard memos, memory metadata, scheduler-owned module status,
and utterance progress. You may call evidence tools during this decision turn. You must not write
cognition log, emit utterances, or change allocation.
The persistent conversation history contains user messages named new_cognition_log_item; those
messages are the cognition-log history.

Speak is not currently streaming. Use a strict readiness gate before setting should_speak=true:
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
  the cognition log, wait silently.
- If the speak memo already contains an utterance that addresses the current cognition-log request,
  set should_speak=false unless a new cognition-log request or peer situation needs another utterance.

When a missing fact is needed for speech, call an evidence tool before waiting:
- query_memory(question) for stable self/body/peer/world facts.
- query_sensory_detail(question) for details from current sensory observations.
If a tool result contains the needed fact but the cognition log does not, return should_speak=false with an
cognition-promotion evidence gap. If evidence is still unavailable, include evidence_gaps that name
the source to consult, the concrete question to answer, and the exact fact that must become visible
in the cognition log before speaking. After publishing an evidence request, wait silently; speak-gate will
reconsider when a later cognition-log update arrives.

When should_speak=true, you are only allowing the pending speak activation to run. Do not choose
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
    should_speak: bool,
    rationale: String,
    #[serde(default)]
    evidence_gaps: Vec<EvidenceGap>,
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

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
struct EvidenceGap {
    source: EvidenceGapSource,
    question: String,
    needed_fact: String,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "kebab-case")]
enum EvidenceGapSource {
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
const DEFAULT_SESSION_COMPACTION_INPUT_TOKEN_THRESHOLD: u64 = 16_000;
const DEFAULT_SESSION_COMPACTION_PREFIX_RATIO: f64 = 0.8;
const COMPACTED_SPEAK_GATE_SESSION_PREFIX: &str = "Compacted speak-gate session history:";
const SESSION_COMPACTION_PROMPT: &str = r#"You compact the speak-gate module's persistent session history.
Summarize only the prefix transcript you receive. Preserve information that future speak-gate
decisions need: cognition-log facts, prior allow/suppress decisions, evidence requests, evidence
gaps, and tool results. Do not invent facts. Keep the summary concise, explicit, and faithful.
Return plain text only."#;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SpeakGateSessionCompactionConfig {
    pub input_token_threshold: u64,
    pub prefix_ratio: f64,
}

impl Default for SpeakGateSessionCompactionConfig {
    fn default() -> Self {
        Self {
            input_token_threshold: DEFAULT_SESSION_COMPACTION_INPUT_TOKEN_THRESHOLD,
            prefix_ratio: DEFAULT_SESSION_COMPACTION_PREFIX_RATIO,
        }
    }
}

fn speak_owner() -> ModuleInstanceId {
    ModuleInstanceId::new(builtin::speak(), ReplicaIndex::ZERO)
}

fn gate_prompt_for<'a>(
    readiness: &'a str,
    _status: &ModuleRunStatus,
    _progress: Option<&UtteranceProgress>,
) -> &'a str {
    readiness
}

fn has_registered_module(modules: &[(ModuleId, &'static str)], module: &ModuleId) -> bool {
    modules.iter().any(|(id, _)| id == module)
}

fn apply_requested_evidence_guard(
    mut decision: SpeakGateDecision,
    requested_sources: &[EvidenceGapSource],
) -> SpeakGateDecision {
    if !decision.should_speak || requested_sources.is_empty() {
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
    decision.should_speak = false;
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

fn cognition_history_input(record: &CognitionLogEntryRecord) -> serde_json::Value {
    serde_json::json!({
        "new_cognition_log_item": record,
    })
}

fn push_cognition_history(session: &mut Session, records: &[CognitionLogEntryRecord]) {
    for record in records {
        session.push_user(cognition_history_input(record).to_string());
    }
}

fn gate_ephemeral_input(
    cognition_context_json: serde_json::Value,
    blackboard_json: serde_json::Value,
    speak_status: ModuleRunStatus,
    utterance_progress: Option<UtteranceProgress>,
) -> serde_json::Value {
    serde_json::json!({
        "cognition_context": cognition_context_json,
        "blackboard": blackboard_json,
        "speak_module_status": speak_status,
        "current_utterance_progress": utterance_progress,
    })
}

fn session_compaction_cutoff(item_count: usize, prefix_ratio: f64) -> Option<usize> {
    if item_count < 2 {
        return None;
    }
    let ratio = if prefix_ratio.is_finite() {
        prefix_ratio.clamp(f64::EPSILON, 1.0)
    } else {
        DEFAULT_SESSION_COMPACTION_PREFIX_RATIO
    };
    let cutoff = ((item_count as f64) * ratio).floor() as usize;
    Some(cutoff.clamp(1, item_count.saturating_sub(1)))
}

fn input_message_role_text(role: InputMessageRole) -> &'static str {
    match role {
        InputMessageRole::System => "system",
        InputMessageRole::Developer => "developer",
        InputMessageRole::User => "user",
    }
}

fn turn_role_text(role: TurnRole) -> &'static str {
    match role {
        TurnRole::System => "system",
        TurnRole::Developer => "developer",
        TurnRole::User => "user",
        TurnRole::Assistant => "assistant",
    }
}

fn raw_json_value(raw: &RawJson) -> serde_json::Value {
    serde_json::from_str(raw.get()).unwrap_or_else(|_| serde_json::Value::String(raw.to_string()))
}

fn render_message_content_for_compaction(content: &MessageContent) -> serde_json::Value {
    match content {
        MessageContent::Text(text) => serde_json::json!({
            "type": "text",
            "text": text,
        }),
    }
}

fn render_assistant_input_for_compaction(item: &AssistantInputItem) -> serde_json::Value {
    match item {
        AssistantInputItem::Text(text) => serde_json::json!({
            "type": "text",
            "text": text,
        }),
        AssistantInputItem::Reasoning(text) => serde_json::json!({
            "type": "reasoning",
            "text": text,
        }),
        AssistantInputItem::Refusal(text) => serde_json::json!({
            "type": "refusal",
            "text": text,
        }),
    }
}

fn render_turn_item_for_compaction(item: &dyn lutum::ItemView) -> serde_json::Value {
    if let Some(text) = item.as_text() {
        return serde_json::json!({
            "type": "text",
            "text": text,
        });
    }
    if let Some(text) = item.as_reasoning() {
        return serde_json::json!({
            "type": "reasoning",
            "text": text,
        });
    }
    if let Some(text) = item.as_refusal() {
        return serde_json::json!({
            "type": "refusal",
            "text": text,
        });
    }
    if let Some(call) = item.as_tool_call() {
        return serde_json::json!({
            "type": "tool_call",
            "id": call.id.to_string(),
            "name": call.name.to_string(),
            "arguments": raw_json_value(call.arguments),
        });
    }
    if let Some(result) = item.as_tool_result() {
        return serde_json::json!({
            "type": "tool_result",
            "id": result.id.to_string(),
            "name": result.name.to_string(),
            "arguments": raw_json_value(result.arguments),
            "result": raw_json_value(result.result),
        });
    }
    serde_json::json!({
        "type": "unknown",
    })
}

fn render_session_item_for_compaction(index: usize, item: &ModelInputItem) -> serde_json::Value {
    match item {
        ModelInputItem::Message { role, content } => serde_json::json!({
            "index": index,
            "kind": "message",
            "role": input_message_role_text(*role),
            "content": content
                .as_slice()
                .iter()
                .map(render_message_content_for_compaction)
                .collect::<Vec<_>>(),
        }),
        ModelInputItem::Assistant(item) => serde_json::json!({
            "index": index,
            "kind": "assistant_input",
            "item": render_assistant_input_for_compaction(item),
        }),
        ModelInputItem::ToolResult(result) => serde_json::json!({
            "index": index,
            "kind": "tool_result",
            "id": result.id.to_string(),
            "name": result.name.to_string(),
            "arguments": raw_json_value(&result.arguments),
            "result": raw_json_value(&result.result),
        }),
        ModelInputItem::Turn(turn) => {
            let items = (0..turn.item_count())
                .filter_map(|item_index| turn.item_at(item_index))
                .map(render_turn_item_for_compaction)
                .collect::<Vec<_>>();
            serde_json::json!({
                "index": index,
                "kind": "turn",
                "role": turn_role_text(turn.role()),
                "items": items,
            })
        }
    }
}

fn render_session_items_for_compaction(items: &[ModelInputItem]) -> serde_json::Value {
    serde_json::Value::Array(
        items
            .iter()
            .enumerate()
            .map(|(index, item)| render_session_item_for_compaction(index, item))
            .collect(),
    )
}

pub struct SpeakGateModule {
    owner: nuillu_types::ModuleId,
    activation_gate: ActivationGate<SpeakModule>,
    cognition_log: CognitionLogReader,
    blackboard: BlackboardReader,
    module_status: ModuleStatusReader,
    attention_control: AttentionControlRequestMailbox,
    memo: Memo,
    llm: LlmAccess,
    session: Session,
    session_compaction: SpeakGateSessionCompactionConfig,
    readiness_prompt: std::sync::OnceLock<String>,
}

impl SpeakGateModule {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        activation_gate: ActivationGate<SpeakModule>,
        cognition_log: CognitionLogReader,
        blackboard: BlackboardReader,
        module_status: ModuleStatusReader,
        attention_control: AttentionControlRequestMailbox,
        memo: Memo,
        llm: LlmAccess,
    ) -> Self {
        Self {
            owner: nuillu_types::ModuleId::new(<Self as Module>::id())
                .expect("speak-gate id is valid"),
            activation_gate,
            cognition_log,
            blackboard,
            module_status,
            attention_control,
            memo,
            llm,
            session: Session::new(),
            session_compaction: SpeakGateSessionCompactionConfig::default(),
            readiness_prompt: std::sync::OnceLock::new(),
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
            nuillu_module::format_system_prompt(
                &base,
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
        event: &ActivationGateEvent<SpeakModule>,
    ) -> Result<()> {
        let self_model_available = has_registered_module(cx.modules(), &builtin::self_model());
        let unread_cognition = self.cognition_log.unread_events().await;
        push_cognition_history(&mut self.session, &unread_cognition);
        let unread_memo_logs = self.blackboard.unread_memo_logs().await;
        push_unread_memo_logs(&mut self.session, &unread_memo_logs);
        let cognition_snapshot = self.cognition_log.snapshot().await;
        let cognition_context_json = serde_json::json!({
            "agentic_deadlock_marker": cognition_snapshot.agentic_deadlock_marker(),
        });
        let speak_owner = speak_owner();
        let speak_status = self.module_status.status_for_instance(&speak_owner).await;
        let (blackboard_json, utterance_progress) = self
            .blackboard
            .read(|bb| {
                (
                    serde_json::json!({
                        "memory_metadata": bb.memory_metadata(),
                        "utterance_progresses": bb.utterance_progresses(),
                    }),
                    bb.utterance_progress_for_instance(&speak_owner).cloned(),
                )
            })
            .await;
        let gate_prompt = gate_prompt_for(
            self.readiness_prompt(cx),
            &speak_status,
            utterance_progress.as_ref(),
        )
        .to_owned();
        self.session.push_ephemeral_system(gate_prompt);
        self.session.push_ephemeral_user(
            gate_ephemeral_input(
                cognition_context_json,
                blackboard_json,
                speak_status,
                utterance_progress,
            )
            .to_string(),
        );

        let lutum = self.llm.lutum().await;
        let decision = self
            .run_decision_turn(&lutum, cx.session_compaction_lutum(), self_model_available)
            .await?;

        self.memo.write(render_speak_gate_memo(&decision)).await;
        event.respond(gate_vote_from_decision(&decision));
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
        if input_tokens <= self.session_compaction.input_token_threshold {
            return;
        }
        if let Err(error) = self.compact(lutum).await {
            tracing::warn!(
                input_tokens,
                threshold = self.session_compaction.input_token_threshold,
                error = ?error,
                "speak-gate session compaction failed"
            );
        }
    }

    async fn compact(&mut self, lutum: &Lutum) -> Result<()> {
        let items = self.session.input().items();
        let Some(cutoff) =
            session_compaction_cutoff(items.len(), self.session_compaction.prefix_ratio)
        else {
            return Ok(());
        };

        let prefix = items[..cutoff].to_vec();
        let suffix = items[cutoff..].to_vec();
        let transcript =
            serde_json::to_string_pretty(&render_session_items_for_compaction(&prefix))
                .context("render speak-gate session compaction transcript")?;

        let mut summary_session = Session::new();
        summary_session.push_system(SESSION_COMPACTION_PROMPT);
        summary_session.push_user(transcript);
        let summary = summary_session
            .text_turn(lutum)
            .collect()
            .await
            .context("summarize speak-gate session prefix")?
            .assistant_text();
        let summary = summary.trim();
        if summary.is_empty() {
            tracing::warn!("speak-gate session compaction produced an empty summary");
            return Ok(());
        }

        let mut compacted_items = Vec::with_capacity(suffix.len().saturating_add(1));
        compacted_items.push(ModelInputItem::text(
            InputMessageRole::System,
            format!("{COMPACTED_SPEAK_GATE_SESSION_PREFIX}\n{summary}"),
        ));
        compacted_items.extend(suffix);
        self.session = Session::from_input(ModelInput::from_items(compacted_items));
        Ok(())
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

fn gate_vote_from_decision(decision: &SpeakGateDecision) -> ActivationGateVote {
    if decision.should_speak {
        ActivationGateVote::Allow
    } else {
        ActivationGateVote::Suppress
    }
}

fn render_speak_gate_memo(decision: &SpeakGateDecision) -> String {
    let mut memo = format!(
        "Speak decision: {}\nRationale: {}",
        if decision.should_speak {
            "speak"
        } else {
            "wait silently"
        },
        decision.rationale.trim(),
    );
    if decision.evidence_gaps.is_empty() {
        memo.push_str("\nEvidence gaps: none");
    } else {
        memo.push_str("\nEvidence gaps:");
        for gap in &decision.evidence_gaps {
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

fn generation_input(
    cognition_log_json: serde_json::Value,
    draft: &GenerationDraft,
) -> serde_json::Value {
    serde_json::json!({
        "cognition_logs": cognition_log_json,
        "speech_target": {
            "target": draft.target.as_str(),
        },
    })
}

fn push_generation_context(
    session: &mut Session,
    cognition_log_json: serde_json::Value,
    draft: &GenerationDraft,
    generation_prompt: &str,
) {
    session.push_system(generation_prompt);
    session.push_user(generation_input(cognition_log_json, draft).to_string());
    if !draft.accumulated.is_empty() {
        session.push_assistant_text(draft.accumulated.clone());
    }
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
        let mut cognition_log_json = self.cognition_log.snapshot().await.compact_json();
        let target = self.select_target(cx, &cognition_log_json).await?;
        let mut draft = GenerationDraft::new(self.utterance.next_generation_id(), target);

        loop {
            self.record_streaming_progress(&draft).await;
            match self
                .stream_generation(cx, cognition_log_json, &mut draft)
                .await?
            {
                GenerationStreamOutcome::Completed => return Ok(()),
                GenerationStreamOutcome::Retry => {
                    cognition_log_json = self.cognition_log.snapshot().await.compact_json();
                }
                GenerationStreamOutcome::Aborted => {
                    cognition_log_json = self.cognition_log.snapshot().await.compact_json();
                    let new_target = self.select_target(cx, &cognition_log_json).await?;
                    draft = GenerationDraft::new(self.utterance.next_generation_id(), new_target);
                }
            }
        }
    }

    async fn select_target(
        &mut self,
        cx: &nuillu_module::ActivateCx<'_>,
        cognition_log_json: &serde_json::Value,
    ) -> Result<String> {
        let mut session = Session::new();
        session.push_system(self.target_prompt(cx));
        session.push_user(
            serde_json::json!({
                "cognition_logs": cognition_log_json,
            })
            .to_string(),
        );

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
        cognition_log_json: serde_json::Value,
        draft: &mut GenerationDraft,
    ) -> Result<GenerationStreamOutcome> {
        let stream_started_at = cx.now();
        let cognition_log_at_start_json = cognition_log_json.clone();

        let mut session = Session::new();
        push_generation_context(
            &mut session,
            cognition_log_json,
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
                        .judge_abort(cx, &cognition_log_at_start_json, &new_entries)
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
        cognition_log_at_start_json: &serde_json::Value,
        new_entries: &[String],
    ) -> Result<bool> {
        let mut session = Session::new();
        session.push_system(self.abort_judge_prompt(cx));
        session.push_user(
            serde_json::json!({
                "cognition_log_at_start": cognition_log_at_start_json,
                "new_cognition_entries": new_entries,
            })
            .to_string(),
        );

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
        "Decides whether pending speak activations may run from cognition-log evidence. If needed query, sensory-detail, or self-model results have not been promoted by cognition-gate, speaking becomes a guess; request evidence or suppress. Records waiting/evidence-gap notes in its memo."
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
        FinishReason, MockLlmAdapter, MockStructuredScenario, MockTextScenario,
        RawStructuredTurnEvent, RawTextTurnEvent, SharedPoolBudgetManager, SharedPoolBudgetOptions,
        Usage,
    };
    use nuillu_blackboard::{
        ActivationRatio, Blackboard, BlackboardCommand, CognitionLogEntry, ModuleConfig,
        ResourceAllocation, linear_ratio_fn,
    };
    use nuillu_module::ports::{
        Clock, NoopCognitionLogRepository, NoopFileSearchProvider, NoopMemoryStore, PortError,
        SystemClock, Utterance, UtteranceSink,
    };
    use nuillu_module::{
        AttentionControlRequestInbox, CapabilityProviderPorts, CapabilityProviders,
        CognitionLogUpdated, LutumTiers, ModuleRegistry, Participant,
    };

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

    fn test_bpm() -> std::ops::RangeInclusive<nuillu_blackboard::Bpm> {
        nuillu_blackboard::Bpm::from_f64(60.0)..=nuillu_blackboard::Bpm::from_f64(60.0)
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
            .register(0..=0, test_bpm(), linear_ratio_fn, move |caps| {
                *gate_sink.borrow_mut() = Some(SpeakGateModule::new(
                    caps.activation_gate_for::<SpeakModule>(),
                    caps.cognition_log_reader(),
                    caps.blackboard_reader(),
                    caps.module_status_reader(),
                    caps.attention_control_mailbox(),
                    caps.memo(),
                    caps.llm_access(),
                ));
                SpeakGateStub
            })
            .unwrap()
            .register(0..=0, test_bpm(), linear_ratio_fn, move |caps| {
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
                    "should_speak": false,
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

        assert!(!decision.should_speak);
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

        let rendered = render_session_items_for_compaction(items).to_string();
        assert!(!rendered.contains("history-0"));
        assert!(rendered.contains("history-8"));
        assert!(rendered.contains("history-9"));
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

        assert!(!decision.should_speak);
        let items = fixture.gate.session.input().items();
        let ModelInputItem::Message { role, content } = &items[0] else {
            panic!("expected original first history item");
        };
        assert_eq!(role, &InputMessageRole::User);
        let [MessageContent::Text(text)] = content.as_slice() else {
            panic!("expected original history text");
        };
        assert_eq!(text, "history-0");

        let rendered = render_session_items_for_compaction(items).to_string();
        assert!(!rendered.contains(COMPACTED_SPEAK_GATE_SESSION_PREFIX));
        assert!(!rendered.contains("unexpected summary"));
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

        push_generation_context(
            &mut session,
            serde_json::json!({"streams": []}),
            &draft,
            GENERATION_PROMPT,
        );
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
        let json: serde_json::Value = serde_json::from_str(text).unwrap();
        assert_eq!(json["speech_target"]["target"], "Koro");
        assert!(json.get("allocation").is_none());
        assert!(json.get("partial_utterance").is_none());
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
            .register(0..=0, test_bpm(), linear_ratio_fn, move |caps| {
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
    fn gate_ephemeral_input_includes_module_status_and_current_utterance_progress() {
        let progress = UtteranceProgress::streaming(5, 1, "Mika", "Mika,");

        let input = gate_ephemeral_input(
            serde_json::json!({"agentic_deadlock_marker": null}),
            serde_json::json!({"memos": {}}),
            ModuleRunStatus::Activating,
            Some(progress),
        );

        assert_eq!(
            input,
            serde_json::json!({
                "cognition_context": {"agentic_deadlock_marker": null},
                "blackboard": {"memos": {}},
                "speak_module_status": {"state": "activating"},
                "current_utterance_progress": {
                    "state": "streaming",
                    "generation_id": 5,
                    "sequence": 1,
                    "target": "Mika",
                    "partial_utterance": "Mika,"
                }
            })
        );
    }

    #[test]
    fn requested_evidence_guard_turns_same_turn_speak_into_wait() {
        let decision = SpeakGateDecision {
            should_speak: true,
            rationale: "tool request is enough".into(),
            evidence_gaps: Vec::new(),
        };

        let guarded = apply_requested_evidence_guard(decision, &[EvidenceGapSource::Memory]);

        assert!(!guarded.should_speak);
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
                let value: serde_json::Value = serde_json::from_str(text).ok()?;
                value
                    .get("new_cognition_log_item")?
                    .get("entry")?
                    .get("text")?
                    .as_str()
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
        push_generation_context(
            &mut session,
            serde_json::json!({"streams": []}),
            &draft,
            GENERATION_PROMPT,
        );
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
        let decision = SpeakGateDecision {
            should_speak: false,
            rationale: "missing body fact".into(),
            evidence_gaps: vec![EvidenceGap {
                source: EvidenceGapSource::Memory,
                question: "What body should I report?".into(),
                needed_fact: "frog body".into(),
            }],
        };

        let memo = render_speak_gate_memo(&decision);

        assert!(memo.contains("Speak decision: wait silently"));
        assert!(memo.contains("Rationale: missing body fact"));
        assert!(memo.contains(
            "Source: memory; question: What body should I report?; needed fact: frog body"
        ));
        assert!(serde_json::from_str::<serde_json::Value>(&memo).is_err());
    }

    #[test]
    fn gate_vote_from_decision_allows_only_speak_decisions() {
        let valid = SpeakGateDecision {
            should_speak: true,
            rationale: "ready".into(),
            evidence_gaps: Vec::new(),
        };

        assert_eq!(gate_vote_from_decision(&valid), ActivationGateVote::Allow);

        let waiting = SpeakGateDecision {
            should_speak: false,
            ..valid
        };
        assert_eq!(
            gate_vote_from_decision(&waiting),
            ActivationGateVote::Suppress
        );
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
