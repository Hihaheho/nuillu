use std::collections::HashSet;

use anyhow::{Context, Result};
use async_trait::async_trait;
use futures::StreamExt;
use lutum::{
    Lutum, Session, StructuredStepOutcomeWithTools, StructuredTurnOutcome, TextTurnEvent,
    ToolResult,
};
use nuillu_module::{
    BlackboardReader, CognitionLogEntryRecord, CognitionLogReader, CognitionLogUpdatedInbox,
    LlmAccess, Memo, Module, ModuleRunStatus, ModuleStatusReader, QueryMailbox, QueryRequest,
    SelfModelMailbox, SelfModelRequest, SensoryDetailRequest, SensoryDetailRequestMailbox,
    SpeakInbox, SpeakMailbox, SpeakRequest, UtteranceProgress, UtteranceProgressState,
    UtteranceWriter,
};
use nuillu_types::{ModuleId, ModuleInstanceId, ReplicaIndex, builtin};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

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

When should_speak=true, provide a speech_plan with a non-empty target and a concrete
generation_hint naming the cognition-log facts to use, the intended frame, and any constraints on style
or scope. The target is mandatory for every utterance: use the questioner when answering a question,
the peer being directly addressed for direct signals, or "self" for self-directed speech/soliloquy.
If you cannot choose a target and write a hint, should_speak must be false and speech_plan must be
null. Return only raw JSON for the structured decision; do not wrap it in Markdown or code fences."#;

const READINESS_GATE_SELF_MODEL_TOOL_PROMPT: &str =
    "- query_self_model(question) for current first-person model facts.\n";

const INTERRUPTION_GATE_PROMPT: &str = r#"You are the speak-gate module.
Speak is currently streaming a user-visible utterance. Decide whether the current stream must be
cancelled and replaced by publishing a new typed SpeakRequest. You may read the current cognitive
cognition-log set, blackboard memos, memory metadata, scheduler-owned module status, and
utterance progress. You may call evidence tools during this decision turn. You must not write
cognition log, emit utterances, or change allocation.
The persistent conversation history contains user messages named new_cognition_log_item; those
messages are the cognition-log history.

Use an interruption gate:
- Compare the new cognition log with the partial utterance and current generation hint.
- Set should_speak=true only if the new cognition-log input changes the required answer, addressee, safety,
  or grounding materially.
- Keep should_speak=false for minor updates, redundant evidence, or cognition-log input that can wait until
  the current utterance completes.
- If interruption is needed, write a generation_hint for the replacement utterance that accounts
  for both the new cognition-log input and the already spoken partial utterance.
- If interruption would require a missing fact, set should_speak=false and include evidence_gaps
  naming the source, concrete question, and exact fact that must become visible in the cognition log.

When should_speak=true, provide a speech_plan with a non-empty target and a concrete
generation_hint naming the cognition-log facts to use, what should replace or continue from the partial
utterance, and any constraints on style or scope. Preserve the same target as the current utterance
unless the new cognition-log input materially changes who must be addressed. Use "self" for self-directed
speech/soliloquy. If you cannot choose a target and write a hint, should_speak must be false and
speech_plan must be null. Return only raw JSON for the structured decision; do not wrap it in
Markdown or code fences."#;

const GENERATION_PROMPT: &str = r#"You are the speak module.
Generate concise user-visible text addressed to the SpeakRequest target from the current cognitive
cognition-log set and the typed SpeakRequest. You cannot inspect blackboard memos or allocation
guidance. Use only the provided cognition context, target, and generation_hint. Do not change the
target or redirect the utterance to a different addressee. Follow the generation_hint as the primary
contract for frame, style, and scope. Do not add generic advice, diagnosis, or facts that are not
present in the cognition context or generation_hint.
If partial_utterance is present, continue that
utterance from exactly where it stopped; do not repeat, rewrite, or replace the already emitted
partial text. Do not mention hidden state or unavailable module results."#;

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
struct SpeakGateDecision {
    should_speak: bool,
    rationale: String,
    speech_plan: Option<SpeechPlan>,
    #[serde(default)]
    evidence_gaps: Vec<EvidenceGap>,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
struct SpeechPlan {
    target: String,
    generation_hint: String,
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

#[lutum::tool_input(name = "query_memory", output = QueryMemoryOutput)]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
struct QueryMemoryArgs {
    question: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
struct QueryMemoryOutput {
    latest_memo: Option<String>,
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
    latest_memo: Option<String>,
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
    latest_memo: Option<String>,
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

fn speak_owner() -> ModuleInstanceId {
    ModuleInstanceId::new(builtin::speak(), ReplicaIndex::ZERO)
}

fn is_speak_streaming(status: &ModuleRunStatus, progress: Option<&UtteranceProgress>) -> bool {
    matches!(status, ModuleRunStatus::Activating)
        && matches!(
            progress.map(|progress| progress.state),
            Some(UtteranceProgressState::Streaming)
        )
}

fn gate_prompt_for<'a>(
    readiness: &'a str,
    interruption: &'a str,
    status: &ModuleRunStatus,
    progress: Option<&UtteranceProgress>,
) -> &'a str {
    if is_speak_streaming(status, progress) {
        interruption
    } else {
        readiness
    }
}

fn has_registered_module(modules: &[(ModuleId, &'static str)], module: &ModuleId) -> bool {
    modules.iter().any(|(id, _)| id == module)
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

pub struct SpeakGateModule {
    owner: nuillu_types::ModuleId,
    cognition_updates: CognitionLogUpdatedInbox,
    cognition_log: CognitionLogReader,
    blackboard: BlackboardReader,
    module_status: ModuleStatusReader,
    query: QueryMailbox,
    self_model: SelfModelMailbox,
    sensory_detail: SensoryDetailRequestMailbox,
    memo: Memo,
    speak: SpeakMailbox,
    llm: LlmAccess,
    session: Session,
    readiness_prompt: std::sync::OnceLock<String>,
    interruption_prompt: std::sync::OnceLock<String>,
}

impl SpeakGateModule {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        cognition_updates: CognitionLogUpdatedInbox,
        cognition_log: CognitionLogReader,
        blackboard: BlackboardReader,
        module_status: ModuleStatusReader,
        query: QueryMailbox,
        self_model: SelfModelMailbox,
        sensory_detail: SensoryDetailRequestMailbox,
        memo: Memo,
        speak: SpeakMailbox,
        llm: LlmAccess,
    ) -> Self {
        Self {
            owner: nuillu_types::ModuleId::new(<Self as Module>::id())
                .expect("speak-gate id is valid"),
            cognition_updates,
            cognition_log,
            blackboard,
            module_status,
            query,
            self_model,
            sensory_detail,
            memo,
            speak,
            llm,
            session: Session::new(),
            readiness_prompt: std::sync::OnceLock::new(),
            interruption_prompt: std::sync::OnceLock::new(),
        }
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
            )
        })
    }

    fn interruption_prompt(&self, cx: &nuillu_module::ActivateCx<'_>) -> &str {
        self.interruption_prompt.get_or_init(|| {
            nuillu_module::format_system_prompt(
                INTERRUPTION_GATE_PROMPT,
                cx.modules(),
                &self.owner,
                cx.identity_memories(),
            )
        })
    }

    #[tracing::instrument(skip_all, err(Debug, level = "warn"))]
    async fn activate(&mut self, cx: &nuillu_module::ActivateCx<'_>) -> Result<()> {
        let self_model_available = has_registered_module(cx.modules(), &builtin::self_model());
        let unread_cognition = self.cognition_log.unread_events().await;
        push_cognition_history(&mut self.session, &unread_cognition);
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
                        "memos": bb.memos(),
                        "memory_metadata": bb.memory_metadata(),
                        "utterance_progresses": bb.utterance_progresses(),
                    }),
                    bb.utterance_progress_for_instance(&speak_owner).cloned(),
                )
            })
            .await;
        let gate_prompt = gate_prompt_for(
            self.readiness_prompt(cx),
            self.interruption_prompt(cx),
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
        let decision = self.run_decision_turn(&lutum, self_model_available).await?;

        self.memo.write(render_speak_gate_memo(&decision)).await;

        if let Some(request) = speak_request_from_decision(&decision)
            && self.speak.publish(request).await.is_err()
        {
            tracing::trace!("speak request had no active subscribers");
        }
        Ok(())
    }

    async fn run_decision_turn(
        &mut self,
        lutum: &Lutum,
        self_model_available: bool,
    ) -> Result<SpeakGateDecision> {
        let mut memory_requests = HashSet::<String>::new();
        let mut self_model_requests = HashSet::<String>::new();
        let mut sensory_detail_requests = HashSet::<String>::new();
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
                    let StructuredTurnOutcome::Structured(decision) = result.semantic else {
                        anyhow::bail!("speak-gate decision turn refused");
                    };
                    return Ok(decision);
                }
                StructuredStepOutcomeWithTools::NeedsTools(round) => {
                    let mut results: Vec<ToolResult> = Vec::new();
                    for call in round.tool_calls.iter().cloned() {
                        match call {
                            SpeakGateToolsCall::QueryMemory(call) => {
                                let output = self
                                    .query_memory(call.input.clone(), &mut memory_requests)
                                    .await
                                    .context("run query_memory tool")?;
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
                }
            }
        }

        anyhow::bail!("speak-gate decision did not finish before tool-round limit")
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
            self.query
                .publish(QueryRequest::new(question))
                .await
                .is_ok()
        };
        Ok(QueryMemoryOutput {
            latest_memo: self.latest_module_memo(&builtin::query_vector()).await,
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
        let before = self.latest_module_memo(&builtin::self_model()).await;
        let requested = if duplicate {
            false
        } else {
            self.self_model
                .publish(SelfModelRequest::new(question))
                .await
                .is_ok()
        };
        Ok(QuerySelfModelOutput {
            latest_memo: before,
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
        let before = self.latest_module_memo(&builtin::sensory()).await;
        let requested = if duplicate {
            false
        } else {
            self.sensory_detail
                .publish(SensoryDetailRequest::new(question))
                .await
                .is_ok()
        };
        Ok(QuerySensoryDetailOutput {
            latest_memo: before,
            requested,
            duplicate,
        })
    }

    async fn latest_module_memo(&self, module: &ModuleId) -> Option<String> {
        self.blackboard.latest_memo_for_module(module).await
    }
}

fn speak_request_from_decision(decision: &SpeakGateDecision) -> Option<SpeakRequest> {
    if !decision.should_speak {
        return None;
    }
    let plan = decision.speech_plan.as_ref()?;
    if plan.generation_hint.trim().is_empty() {
        return None;
    }
    SpeakRequest::try_new(
        plan.target.as_str(),
        plan.generation_hint.clone(),
        decision.rationale.clone(),
    )
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
    if let Some(plan) = &decision.speech_plan {
        memo.push_str("\nSpeech target: ");
        memo.push_str(plan.target.trim());
        memo.push_str("\nGeneration hint: ");
        memo.push_str(plan.generation_hint.trim());
    } else {
        memo.push_str("\nSpeech plan: none");
    }
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
    generation_hint: String,
    rationale: String,
}

impl GenerationDraft {
    fn new(generation_id: u64, request: SpeakRequest) -> GenerationDraft {
        GenerationDraft {
            generation_id,
            sequence: 0,
            accumulated: String::new(),
            target: request.target,
            generation_hint: request.generation_hint,
            rationale: request.rationale,
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
        "Completed utterance to {}:\n{}\nRationale: {}",
        draft.target.trim(),
        text.trim(),
        draft.rationale.trim(),
    )
}

fn generation_input(
    cognition_log_json: serde_json::Value,
    draft: &GenerationDraft,
) -> serde_json::Value {
    serde_json::json!({
        "cognition_logs": cognition_log_json,
        "speak_request": {
            "target": draft.target.as_str(),
            "generation_hint": draft.generation_hint.as_str(),
            "rationale": draft.rationale.as_str(),
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
    Replaced(SpeakRequest),
}

pub struct SpeakModule {
    owner: nuillu_types::ModuleId,
    requests: SpeakInbox,
    cognition_log: CognitionLogReader,
    memo: Memo,
    utterance: UtteranceWriter,
    llm: LlmAccess,
    generation_prompt: std::sync::OnceLock<String>,
}

impl SpeakModule {
    pub fn new(
        requests: SpeakInbox,
        cognition_log: CognitionLogReader,
        memo: Memo,
        utterance: UtteranceWriter,
        llm: LlmAccess,
    ) -> Self {
        Self {
            owner: nuillu_types::ModuleId::new(<Self as Module>::id()).expect("speak id is valid"),
            requests,
            cognition_log,
            memo,
            utterance,
            llm,
            generation_prompt: std::sync::OnceLock::new(),
        }
    }

    fn generation_prompt(&self, cx: &nuillu_module::ActivateCx<'_>) -> &str {
        self.generation_prompt.get_or_init(|| {
            nuillu_module::format_system_prompt(
                GENERATION_PROMPT,
                cx.modules(),
                &self.owner,
                cx.identity_memories(),
            )
        })
    }

    #[tracing::instrument(skip_all, err(Debug, level = "warn"))]
    async fn activate(
        &mut self,
        cx: &nuillu_module::ActivateCx<'_>,
        request: SpeakRequest,
    ) -> Result<()> {
        let mut cognition_log_json = self.cognition_log.snapshot().await.compact_json();

        let mut draft = GenerationDraft::new(self.utterance.next_generation_id(), request);

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
                GenerationStreamOutcome::Replaced(request) => {
                    cognition_log_json = self.cognition_log.snapshot().await.compact_json();
                    draft = GenerationDraft::new(self.utterance.next_generation_id(), request);
                }
            }
        }
    }

    async fn stream_generation(
        &mut self,
        cx: &nuillu_module::ActivateCx<'_>,
        cognition_log_json: serde_json::Value,
        draft: &mut GenerationDraft,
    ) -> Result<GenerationStreamOutcome> {
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
                                    draft.generation_hint.clone(),
                                    draft.rationale.clone(),
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
                request = self.requests.next_item() => {
                    let envelope = request?;
                    let mut replacement = envelope.body;
                    for ready in self.requests.take_ready_items()?.items {
                        replacement = ready.body;
                    }
                    return Ok(GenerationStreamOutcome::Replaced(replacement));
                }
            }
        }
    }

    async fn record_streaming_progress(&self, draft: &GenerationDraft) {
        self.utterance
            .record_progress(UtteranceProgress::streaming(
                draft.generation_id,
                draft.sequence,
                draft.target.clone(),
                draft.accumulated.clone(),
                draft.generation_hint.clone(),
                draft.rationale.clone(),
            ))
            .await;
    }
}

#[async_trait(?Send)]
impl Module for SpeakGateModule {
    type Batch = ();

    fn id() -> &'static str {
        "speak-gate"
    }

    fn role_description() -> &'static str {
        "Decides whether the cognition log is ready for speech; sends SpeakRequest to speak when ready, otherwise records waiting/evidence-gap notes in its memo."
    }

    async fn next_batch(&mut self) -> Result<Self::Batch> {
        SpeakGateModule::next_batch(self).await
    }

    async fn activate(
        &mut self,
        cx: &nuillu_module::ActivateCx<'_>,
        _batch: &Self::Batch,
    ) -> Result<()> {
        SpeakGateModule::activate(self, cx).await
    }
}

#[async_trait(?Send)]
impl Module for SpeakModule {
    type Batch = batch::NextBatch;

    fn id() -> &'static str {
        "speak"
    }

    fn role_description() -> &'static str {
        "Emits user-visible utterances on SpeakRequest; streams output and records the completed utterance to its memo."
    }

    async fn next_batch(&mut self) -> Result<Self::Batch> {
        SpeakModule::next_batch(self).await
    }

    async fn activate(
        &mut self,
        cx: &nuillu_module::ActivateCx<'_>,
        batch: &Self::Batch,
    ) -> Result<()> {
        SpeakModule::activate(self, cx, batch.request.clone()).await
    }
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::rc::Rc;
    use std::sync::Arc;

    use lutum::{
        AssistantInputItem, InputMessageRole, Lutum, MessageContent, MockLlmAdapter,
        ModelInputItem, Session, SharedPoolBudgetManager, SharedPoolBudgetOptions,
    };
    use nuillu_blackboard::{
        ActivationRatio, Blackboard, BlackboardCommand, CognitionLogEntry, ModuleConfig,
        ResourceAllocation, linear_ratio_fn,
    };
    use nuillu_module::ports::{
        Clock, NoopCognitionLogRepository, NoopFileSearchProvider, NoopMemoryStore,
        NoopUtteranceSink, SystemClock,
    };
    use nuillu_module::{
        CapabilityProviders, LutumTiers, ModuleRegistry, QueryInbox, SelfModelInbox,
        SensoryDetailRequestInbox,
    };

    use super::*;

    fn test_session() -> Session {
        Session::new()
    }

    fn test_caps(blackboard: Blackboard) -> CapabilityProviders {
        let adapter = Arc::new(MockLlmAdapter::new());
        let budget = SharedPoolBudgetManager::new(SharedPoolBudgetOptions::default());
        let lutum = Lutum::new(adapter, budget);
        CapabilityProviders::new(
            blackboard,
            Arc::new(NoopCognitionLogRepository),
            Arc::new(NoopMemoryStore),
            Vec::new(),
            Arc::new(NoopFileSearchProvider),
            Arc::new(NoopUtteranceSink),
            Arc::new(SystemClock),
            LutumTiers {
                cheap: lutum.clone(),
                default: lutum.clone(),
                premium: lutum,
            },
        )
    }

    fn tool_test_allocation() -> ResourceAllocation {
        let mut allocation = ResourceAllocation::default();
        for module in [
            builtin::speak_gate(),
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
    noop_stub!(QueryVectorStub, "query-vector");
    noop_stub!(SelfModelStub, "self-model");
    noop_stub!(SensoryStub, "sensory");

    struct GateToolFixture {
        gate: SpeakGateModule,
        blackboard: Blackboard,
        query_inbox: QueryInbox,
        self_model_inbox: SelfModelInbox,
        sensory_detail_inbox: SensoryDetailRequestInbox,
        query_memo: Memo,
        self_model_memo: Memo,
        sensory_memo: Memo,
    }

    async fn gate_tool_fixture() -> GateToolFixture {
        let blackboard = Blackboard::with_allocation(tool_test_allocation());
        let caps = test_caps(blackboard.clone());

        let gate_cell = Rc::new(RefCell::new(None));
        let query_inbox_cell = Rc::new(RefCell::new(None));
        let self_model_inbox_cell = Rc::new(RefCell::new(None));
        let sensory_detail_inbox_cell = Rc::new(RefCell::new(None));
        let query_memo_cell = Rc::new(RefCell::new(None));
        let self_model_memo_cell = Rc::new(RefCell::new(None));
        let sensory_memo_cell = Rc::new(RefCell::new(None));

        let gate_sink = Rc::clone(&gate_cell);
        let query_inbox_sink = Rc::clone(&query_inbox_cell);
        let query_memo_sink = Rc::clone(&query_memo_cell);
        let self_model_inbox_sink = Rc::clone(&self_model_inbox_cell);
        let self_model_memo_sink = Rc::clone(&self_model_memo_cell);
        let sensory_detail_inbox_sink = Rc::clone(&sensory_detail_inbox_cell);
        let sensory_memo_sink = Rc::clone(&sensory_memo_cell);

        let _modules = ModuleRegistry::new()
            .register(0..=0, test_bpm(), linear_ratio_fn, move |caps| {
                *gate_sink.borrow_mut() = Some(SpeakGateModule::new(
                    caps.cognition_log_updated_inbox(),
                    caps.cognition_log_reader(),
                    caps.blackboard_reader(),
                    caps.module_status_reader(),
                    caps.query_mailbox(),
                    caps.self_model_mailbox(),
                    caps.sensory_detail_mailbox(),
                    caps.memo(),
                    caps.speak_mailbox(),
                    caps.llm_access(),
                ));
                SpeakGateStub
            })
            .unwrap()
            .register(0..=0, test_bpm(), linear_ratio_fn, move |caps| {
                *query_inbox_sink.borrow_mut() = Some(caps.query_inbox());
                *query_memo_sink.borrow_mut() = Some(caps.memo());
                QueryVectorStub
            })
            .unwrap()
            .register(0..=0, test_bpm(), linear_ratio_fn, move |caps| {
                *self_model_inbox_sink.borrow_mut() = Some(caps.self_model_inbox());
                *self_model_memo_sink.borrow_mut() = Some(caps.memo());
                SelfModelStub
            })
            .unwrap()
            .register(0..=0, test_bpm(), linear_ratio_fn, move |caps| {
                *sensory_detail_inbox_sink.borrow_mut() = Some(caps.sensory_detail_inbox());
                *sensory_memo_sink.borrow_mut() = Some(caps.memo());
                SensoryStub
            })
            .unwrap()
            .build(&caps)
            .await
            .unwrap();

        GateToolFixture {
            gate: gate_cell.borrow_mut().take().unwrap(),
            blackboard,
            query_inbox: query_inbox_cell.borrow_mut().take().unwrap(),
            self_model_inbox: self_model_inbox_cell.borrow_mut().take().unwrap(),
            sensory_detail_inbox: sensory_detail_inbox_cell.borrow_mut().take().unwrap(),
            query_memo: query_memo_cell.borrow_mut().take().unwrap(),
            self_model_memo: self_model_memo_cell.borrow_mut().take().unwrap(),
            sensory_memo: sensory_memo_cell.borrow_mut().take().unwrap(),
        }
    }

    #[test]
    fn fresh_generation_omits_assistant_prefill() {
        let draft = GenerationDraft::new(7, SpeakRequest::new("Koro", "be concise", "respond"));
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
        assert_eq!(json["speak_request"]["target"], "Koro");
        assert_eq!(json["speak_request"]["generation_hint"], "be concise");
        assert_eq!(json["speak_request"]["rationale"], "respond");
        assert!(json.get("allocation").is_none());
        assert!(json.get("partial_utterance").is_none());
    }

    #[test]
    fn gate_prompt_switches_only_for_active_streaming_speak() {
        let progress = UtteranceProgress::streaming(
            3,
            2,
            "Koro",
            "Koro, wait",
            "answer Koro",
            "peer request changed",
        );

        assert!(
            gate_prompt_for(
                READINESS_GATE_PROMPT,
                INTERRUPTION_GATE_PROMPT,
                &ModuleRunStatus::Inactive,
                None,
            )
            .contains("not currently streaming")
        );
        assert!(
            gate_prompt_for(
                READINESS_GATE_PROMPT,
                INTERRUPTION_GATE_PROMPT,
                &ModuleRunStatus::Activating,
                Some(&progress),
            )
            .contains("currently streaming")
        );
        assert!(
            gate_prompt_for(
                READINESS_GATE_PROMPT,
                INTERRUPTION_GATE_PROMPT,
                &ModuleRunStatus::AwaitingBatch,
                Some(&progress),
            )
            .contains("not currently streaming")
        );
        assert!(
            gate_prompt_for(
                READINESS_GATE_PROMPT,
                INTERRUPTION_GATE_PROMPT,
                &ModuleRunStatus::Activating,
                None,
            )
            .contains("not currently streaming")
        );
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
        let progress = UtteranceProgress::streaming(
            5,
            1,
            "Mika",
            "Mika,",
            "answer Mika calmly",
            "peer is stressed",
        );

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
                    "partial_utterance": "Mika,",
                    "generation_hint": "answer Mika calmly",
                    "rationale": "peer is stressed"
                }
            })
        );
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
        let mut draft = GenerationDraft::new(11, SpeakRequest::new("Koro", "continue", "respond"));
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
            speech_plan: None,
            evidence_gaps: vec![EvidenceGap {
                source: EvidenceGapSource::Memory,
                question: "What body should I report?".into(),
                needed_fact: "frog body".into(),
            }],
        };

        let memo = render_speak_gate_memo(&decision);

        assert!(memo.contains("Speak decision: wait silently"));
        assert!(memo.contains("Rationale: missing body fact"));
        assert!(memo.contains("Speech plan: none"));
        assert!(memo.contains(
            "Source: memory; question: What body should I report?; needed fact: frog body"
        ));
        assert!(serde_json::from_str::<serde_json::Value>(&memo).is_err());
    }

    #[test]
    fn speak_request_from_decision_requires_target_and_hint() {
        let valid = SpeakGateDecision {
            should_speak: true,
            rationale: "ready".into(),
            speech_plan: Some(SpeechPlan {
                target: " Koro ".into(),
                generation_hint: "answer Koro".into(),
            }),
            evidence_gaps: Vec::new(),
        };

        let request = speak_request_from_decision(&valid).unwrap();
        assert_eq!(request.target, "Koro");
        assert_eq!(request.generation_hint, "answer Koro");
        assert_eq!(request.rationale, "ready");

        let empty_target = SpeakGateDecision {
            speech_plan: Some(SpeechPlan {
                target: " ".into(),
                generation_hint: "answer".into(),
            }),
            ..valid.clone()
        };
        assert!(speak_request_from_decision(&empty_target).is_none());

        let empty_hint = SpeakGateDecision {
            speech_plan: Some(SpeechPlan {
                target: "Koro".into(),
                generation_hint: " ".into(),
            }),
            ..valid.clone()
        };
        assert!(speak_request_from_decision(&empty_hint).is_none());

        let waiting = SpeakGateDecision {
            should_speak: false,
            ..valid
        };
        assert!(speak_request_from_decision(&waiting).is_none());
    }

    #[tokio::test(flavor = "current_thread")]
    async fn evidence_tools_publish_requests_without_polling_and_return_current_memos() {
        let mut fixture = gate_tool_fixture().await;
        fixture.query_memo.write("cached memory fact").await;
        fixture
            .self_model_memo
            .write("cached self-model fact")
            .await;
        fixture.sensory_memo.write("cached sensory detail").await;

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
        let memory_request = fixture.query_inbox.next_item().await.unwrap();
        assert_eq!(
            memory_output.latest_memo.as_deref(),
            Some("cached memory fact")
        );
        assert!(memory_output.requested);
        assert!(!memory_output.duplicate);
        assert_eq!(memory_request.sender.module, builtin::speak_gate());
        assert_eq!(memory_request.body.question, "body fact?");

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
        assert_eq!(
            duplicate_memory_output.latest_memo.as_deref(),
            Some("cached memory fact")
        );
        assert!(!duplicate_memory_output.requested);
        assert!(duplicate_memory_output.duplicate);
        assert!(
            fixture
                .query_inbox
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
        let self_model_request = fixture.self_model_inbox.next_item().await.unwrap();
        assert_eq!(
            self_model_output.latest_memo.as_deref(),
            Some("cached self-model fact")
        );
        assert!(self_model_output.requested);
        assert!(!self_model_output.duplicate);
        assert_eq!(self_model_request.sender.module, builtin::speak_gate());
        assert_eq!(self_model_request.body.question, "current role?");

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
        let sensory_request = fixture.sensory_detail_inbox.next_item().await.unwrap();
        assert_eq!(
            sensory_output.latest_memo.as_deref(),
            Some("cached sensory detail")
        );
        assert!(sensory_output.requested);
        assert!(!sensory_output.duplicate);
        assert_eq!(sensory_request.sender.module, builtin::speak_gate());
        assert_eq!(sensory_request.body.question, "what was just heard?");
    }
}
