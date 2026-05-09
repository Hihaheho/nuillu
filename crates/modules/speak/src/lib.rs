use std::collections::HashSet;

use anyhow::{Context, Result};
use async_trait::async_trait;
use futures::StreamExt;
use lutum::{
    Lutum, Session, StructuredStepOutcomeWithTools, StructuredTurnOutcome, TextTurnEvent,
    ToolResult,
};
use nuillu_module::{
    AttentionLogRecord, AttentionReader, AttentionStreamUpdatedInbox, BlackboardReader, LlmAccess,
    Memo, Module, ModuleRunStatus, ModuleStatusReader, QueryMailbox, QueryRequest,
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
cognitive attention-stream set, blackboard memos, memory metadata, scheduler-owned module status,
and utterance progress. You may call evidence tools during this decision turn. You must not write
attention, emit utterances, or change allocation.
The persistent conversation history contains user messages named new_attention_stream_item; those
messages are the cognitive attention stream history.

Speak is not currently streaming. Use a strict readiness gate before setting should_speak=true:
- The attention stream must contain the facts needed for the utterance, not only raw sensory
  observations, open questions, predictions, or instructions for another module.
- If the current topic asks for stored memory, a self/peer/world model, file evidence, or a rule,
  do not let speak use memo-only facts directly. If a memo contains the needed fact but attention
  does not, request attention-promotion.
- For ordinary peer-directed replies, do not require self-model role clarity. Require self-model
  evidence only for caregiver escalation, authority claims, or self/body capability claims.
- Do not wait merely because analysis memos exist. Wait only when a named retrieved or promoted
  fact that is essential to the answer is still absent from attention.
- Treat in-world peer-directed speech, a direct question from another animal, or an immediate peer
  distress/conflict state as response-worthy; the peer interaction itself is the external
  conversational need.
- Preserve the source frame of the attended interaction. If the attended context is an in-world or
  peer-directed exchange, do not convert it into external assistant advice unless that is explicitly
  asked for.
- If responding now would require generic advice, unsupported diagnosis, or facts absent from
  attention, wait silently.
- If the speak memo already contains an utterance that addresses the current attended request, set
  should_speak=false unless a new attended request or peer situation needs another utterance.

When a missing fact is needed for speech, call an evidence tool before waiting:
- query_memory(question) for stable self/body/peer/world facts.
- query_self_model(question) for current first-person model facts.
- query_sensory_detail(question) for details from current sensory observations.
If a tool result contains the needed fact but attention does not, return should_speak=false with an
attention-promotion evidence gap. If evidence is still unavailable, include evidence_gaps that name
the source to consult, the concrete question to answer, and the exact fact that must become visible
in attention before speaking. After publishing an evidence request, wait silently; speak-gate will
reconsider when a later attention-stream update arrives.

When should_speak=true, provide a concrete generation_hint naming the attended facts to use, the
intended addressee/frame, and any constraints on style or scope. If you cannot write such a hint,
should_speak must be false. Return only raw JSON for the structured decision; do not wrap it in
Markdown or code fences."#;

const INTERRUPTION_GATE_PROMPT: &str = r#"You are the speak-gate module.
Speak is currently streaming a user-visible utterance. Decide whether the current stream must be
cancelled and replaced by publishing a new typed SpeakRequest. You may read the current cognitive
attention-stream set, blackboard memos, memory metadata, scheduler-owned module status, and
utterance progress. You may call evidence tools during this decision turn. You must not write
attention, emit utterances, or change allocation.
The persistent conversation history contains user messages named new_attention_stream_item; those
messages are the cognitive attention stream history.

Use an interruption gate:
- Compare the new attention stream with the partial utterance and current generation hint.
- Set should_speak=true only if the new attention changes the required answer, addressee, safety,
  or grounding materially.
- Keep should_speak=false for minor updates, redundant evidence, or attention that can wait until
  the current utterance completes.
- If interruption is needed, write a generation_hint for the replacement utterance that accounts
  for both the new attention and the already spoken partial utterance.
- If interruption would require a missing fact, set should_speak=false and include evidence_gaps
  naming the source, concrete question, and exact fact that must become visible in attention.

When should_speak=true, provide a concrete generation_hint naming the attended facts to use, the
intended addressee/frame, what should replace or continue from the partial utterance, and any
constraints on style or scope. If you cannot write such a hint, should_speak must be false. Return
only raw JSON for the structured decision; do not wrap it in Markdown or code fences."#;

const GENERATION_PROMPT: &str = r#"You are the speak module.
Generate concise user-visible text from the current cognitive attention-stream set and the typed
SpeakRequest. You cannot inspect blackboard memos or allocation guidance. Use only the provided
attention context and the SpeakRequest generation_hint. Follow the generation_hint as the primary
contract for addressee, frame, style, and scope. Do not add generic advice, diagnosis, or facts that
are not present in the attention context or generation_hint.
If partial_utterance is present, continue that
utterance from exactly where it stopped; do not repeat, rewrite, or replace the already emitted
partial text. Do not mention hidden state or unavailable module results."#;

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
struct SpeakGateDecision {
    should_speak: bool,
    rationale: String,
    generation_hint: Option<String>,
    #[serde(default)]
    evidence_gaps: Vec<EvidenceGap>,
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
    AttentionPromotion,
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

fn attention_history_input(record: &AttentionLogRecord) -> serde_json::Value {
    serde_json::json!({
        "new_attention_stream_item": record,
    })
}

fn push_attention_history(session: &mut Session, records: &[AttentionLogRecord]) {
    for record in records {
        session.push_user(attention_history_input(record).to_string());
    }
}

fn gate_ephemeral_input(
    attention_context_json: serde_json::Value,
    blackboard_json: serde_json::Value,
    speak_status: ModuleRunStatus,
    utterance_progress: Option<UtteranceProgress>,
) -> serde_json::Value {
    serde_json::json!({
        "attention_context": attention_context_json,
        "blackboard": blackboard_json,
        "speak_module_status": speak_status,
        "current_utterance_progress": utterance_progress,
    })
}

pub struct SpeakGateModule {
    owner: nuillu_types::ModuleId,
    attention_updates: AttentionStreamUpdatedInbox,
    attention: AttentionReader,
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
        attention_updates: AttentionStreamUpdatedInbox,
        attention: AttentionReader,
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
            attention_updates,
            attention,
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
            nuillu_module::format_system_prompt(READINESS_GATE_PROMPT, cx.modules(), &self.owner)
        })
    }

    fn interruption_prompt(&self, cx: &nuillu_module::ActivateCx<'_>) -> &str {
        self.interruption_prompt.get_or_init(|| {
            nuillu_module::format_system_prompt(INTERRUPTION_GATE_PROMPT, cx.modules(), &self.owner)
        })
    }

    #[tracing::instrument(skip_all, err(Debug, level = "warn"))]
    async fn activate(&mut self, cx: &nuillu_module::ActivateCx<'_>) -> Result<()> {
        let unread_attention = self.attention.unread_events().await;
        push_attention_history(&mut self.session, &unread_attention);
        let attention_snapshot = self.attention.snapshot().await;
        let attention_context_json = serde_json::json!({
            "agentic_deadlock_marker": attention_snapshot.agentic_deadlock_marker(),
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
                attention_context_json,
                blackboard_json,
                speak_status,
                utterance_progress,
            )
            .to_string(),
        );

        let lutum = self.llm.lutum().await;
        let decision = self.run_decision_turn(&lutum).await?;

        self.memo
            .write(serde_json::to_string(&decision).context("serialize speak-gate decision memo")?)
            .await;

        if decision.should_speak {
            if let Some(generation_hint) = decision.generation_hint
                && !generation_hint.trim().is_empty()
                && self
                    .speak
                    .publish(SpeakRequest::new(generation_hint, decision.rationale))
                    .await
                    .is_err()
            {
                tracing::trace!("speak request had no active subscribers");
            }
        }
        Ok(())
    }

    async fn run_decision_turn(&mut self, lutum: &Lutum) -> Result<SpeakGateDecision> {
        let mut memory_requests = HashSet::<String>::new();
        let mut self_model_requests = HashSet::<String>::new();
        let mut sensory_detail_requests = HashSet::<String>::new();

        for _ in 0..MAX_GATE_TOOL_ROUNDS {
            let outcome = self
                .session
                .structured_turn::<SpeakGateDecision>(lutum)
                .tools::<SpeakGateTools>()
                .available_tools([
                    SpeakGateToolsSelector::QueryMemory,
                    SpeakGateToolsSelector::QuerySelfModel,
                    SpeakGateToolsSelector::QuerySensoryDetail,
                ])
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

struct GenerationDraft {
    generation_id: u64,
    sequence: u32,
    accumulated: String,
    generation_hint: String,
    rationale: String,
}

impl GenerationDraft {
    fn new(generation_id: u64, request: SpeakRequest) -> GenerationDraft {
        GenerationDraft {
            generation_id,
            sequence: 0,
            accumulated: String::new(),
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

fn generation_input(
    attention_json: serde_json::Value,
    draft: &GenerationDraft,
) -> serde_json::Value {
    serde_json::json!({
        "attention_streams": attention_json,
        "speak_request": {
            "generation_hint": draft.generation_hint.as_str(),
            "rationale": draft.rationale.as_str(),
        },
    })
}

fn push_generation_context(
    session: &mut Session,
    attention_json: serde_json::Value,
    draft: &GenerationDraft,
    generation_prompt: &str,
) {
    session.push_system(generation_prompt);
    session.push_user(generation_input(attention_json, draft).to_string());
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
    attention: AttentionReader,
    memo: Memo,
    utterance: UtteranceWriter,
    llm: LlmAccess,
    generation_prompt: std::sync::OnceLock<String>,
}

impl SpeakModule {
    pub fn new(
        requests: SpeakInbox,
        attention: AttentionReader,
        memo: Memo,
        utterance: UtteranceWriter,
        llm: LlmAccess,
    ) -> Self {
        Self {
            owner: nuillu_types::ModuleId::new(<Self as Module>::id())
                .expect("speak id is valid"),
            requests,
            attention,
            memo,
            utterance,
            llm,
            generation_prompt: std::sync::OnceLock::new(),
        }
    }

    fn generation_prompt(&self, cx: &nuillu_module::ActivateCx<'_>) -> &str {
        self.generation_prompt.get_or_init(|| {
            nuillu_module::format_system_prompt(GENERATION_PROMPT, cx.modules(), &self.owner)
        })
    }

    #[tracing::instrument(skip_all, err(Debug, level = "warn"))]
    async fn activate(
        &mut self,
        cx: &nuillu_module::ActivateCx<'_>,
        request: SpeakRequest,
    ) -> Result<()> {
        let mut attention_json = self.attention.snapshot().await.compact_json();

        let mut draft = GenerationDraft::new(self.utterance.next_generation_id(), request);

        loop {
            self.record_streaming_progress(&draft).await;
            match self.stream_generation(cx, attention_json, &mut draft).await? {
                GenerationStreamOutcome::Completed => return Ok(()),
                GenerationStreamOutcome::Retry => {
                    attention_json = self.attention.snapshot().await.compact_json();
                }
                GenerationStreamOutcome::Replaced(request) => {
                    attention_json = self.attention.snapshot().await.compact_json();
                    draft = GenerationDraft::new(self.utterance.next_generation_id(), request);
                }
            }
        }
    }

    async fn stream_generation(
        &mut self,
        cx: &nuillu_module::ActivateCx<'_>,
        attention_json: serde_json::Value,
        draft: &mut GenerationDraft,
    ) -> Result<GenerationStreamOutcome> {
        let mut session = Session::new();
        push_generation_context(&mut session, attention_json, draft, self.generation_prompt(cx));

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
                                .emit_delta(draft.generation_id, sequence, delta)
                                .await;
                            self.record_streaming_progress(draft).await;
                        }
                        Some(Ok(TextTurnEvent::WillRetry { .. })) => {
                            return Ok(GenerationStreamOutcome::Retry);
                        }
                        Some(Ok(TextTurnEvent::Completed { .. })) | None => {
                            let text = draft.accumulated.trim().to_owned();
                            self.memo
                                .write(serde_json::json!({
                                    "utterance": text,
                                    "rationale": draft.rationale.as_str(),
                                }).to_string())
                                .await;
                            self.utterance
                                .record_progress(UtteranceProgress::completed(
                                    draft.generation_id,
                                    draft.sequence,
                                    text.clone(),
                                    draft.generation_hint.clone(),
                                    draft.rationale.clone(),
                                ))
                                .await;
                            if !text.is_empty() {
                                self.utterance.emit(text).await;
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
                draft.accumulated.clone(),
                draft.generation_hint.clone(),
                draft.rationale.clone(),
            ))
            .await;
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
        ActivationRatio, AttentionStreamEvent, Blackboard, BlackboardCommand, ModuleConfig,
        ResourceAllocation, linear_ratio_fn,
    };
    use nuillu_module::ports::{
        Clock, NoopAttentionRepository, NoopFileSearchProvider, NoopMemoryStore, NoopUtteranceSink,
        SystemClock,
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
            Arc::new(NoopAttentionRepository),
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
                    caps.attention_stream_updated_inbox(),
                    caps.attention_reader(),
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
        let draft = GenerationDraft::new(7, SpeakRequest::new("be concise", "respond"));
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
        assert_eq!(json["speak_request"]["generation_hint"], "be concise");
        assert_eq!(json["speak_request"]["rationale"], "respond");
        assert!(json.get("allocation").is_none());
        assert!(json.get("partial_utterance").is_none());
    }

    #[test]
    fn gate_prompt_switches_only_for_active_streaming_speak() {
        let progress =
            UtteranceProgress::streaming(3, 2, "Koro, wait", "answer Koro", "peer request changed");

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
    fn gate_ephemeral_input_includes_module_status_and_current_utterance_progress() {
        let progress =
            UtteranceProgress::streaming(5, 1, "Mika,", "answer Mika calmly", "peer is stressed");

        let input = gate_ephemeral_input(
            serde_json::json!({"agentic_deadlock_marker": null}),
            serde_json::json!({"memos": {}}),
            ModuleRunStatus::Activating,
            Some(progress),
        );

        assert_eq!(
            input,
            serde_json::json!({
                "attention_context": {"agentic_deadlock_marker": null},
                "blackboard": {"memos": {}},
                "speak_module_status": {"state": "activating"},
                "current_utterance_progress": {
                    "state": "streaming",
                    "generation_id": 5,
                    "sequence": 1,
                    "partial_utterance": "Mika,",
                    "generation_hint": "answer Mika calmly",
                    "rationale": "peer is stressed"
                }
            })
        );
    }

    fn attention_history_texts(session: &Session) -> Vec<String> {
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
                    .get("new_attention_stream_item")?
                    .get("event")?
                    .get("text")?
                    .as_str()
                    .map(ToOwned::to_owned)
            })
            .collect()
    }

    #[tokio::test(flavor = "current_thread")]
    async fn speak_gate_attention_history_uses_reader_unread_cursor() {
        let mut fixture = gate_tool_fixture().await;
        let stream = ModuleInstanceId::new(builtin::attention_gate(), ReplicaIndex::ZERO);
        let clock = SystemClock;

        fixture
            .blackboard
            .apply(BlackboardCommand::AppendAttentionStream {
                stream: stream.clone(),
                event: AttentionStreamEvent {
                    at: clock.now(),
                    text: "first".into(),
                },
            })
            .await;

        let first = fixture.gate.attention.unread_events().await;
        push_attention_history(&mut fixture.gate.session, &first);
        assert_eq!(
            attention_history_texts(&fixture.gate.session),
            vec!["first"]
        );

        let already_seen = fixture.gate.attention.unread_events().await;
        push_attention_history(&mut fixture.gate.session, &already_seen);
        assert_eq!(
            attention_history_texts(&fixture.gate.session),
            vec!["first"]
        );

        fixture
            .blackboard
            .apply(BlackboardCommand::AppendAttentionStream {
                stream,
                event: AttentionStreamEvent {
                    at: clock.now(),
                    text: "second".into(),
                },
            })
            .await;

        let second = fixture.gate.attention.unread_events().await;
        push_attention_history(&mut fixture.gate.session, &second);
        assert_eq!(
            attention_history_texts(&fixture.gate.session),
            vec!["first", "second"]
        );
    }

    #[test]
    fn resumed_generation_keeps_id_sequence_and_pushes_assistant_prefill() {
        let mut draft = GenerationDraft::new(11, SpeakRequest::new("continue", "respond"));
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
    fn wait_decision_serializes_evidence_gaps_for_memo() {
        let decision = SpeakGateDecision {
            should_speak: false,
            rationale: "missing body fact".into(),
            generation_hint: None,
            evidence_gaps: vec![EvidenceGap {
                source: EvidenceGapSource::Memory,
                question: "What body should I report?".into(),
                needed_fact: "frog body".into(),
            }],
        };

        let value = serde_json::to_value(decision).unwrap();

        assert_eq!(
            value,
            serde_json::json!({
                "should_speak": false,
                "rationale": "missing body fact",
                "generation_hint": null,
                "evidence_gaps": [
                    {
                        "source": "memory",
                        "question": "What body should I report?",
                        "needed_fact": "frog body"
                    }
                ]
            })
        );
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

#[async_trait(?Send)]
impl Module for SpeakGateModule {
    type Batch = ();

    fn id() -> &'static str {
        "speak-gate"
    }

    fn role_description() -> &'static str {
        "Decides whether attention is ready for speech; sends SpeakRequest to speak when ready, otherwise records waiting/evidence-gap notes in its memo."
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
