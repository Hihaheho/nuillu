use anyhow::{Context, Result};
use async_trait::async_trait;
use futures::StreamExt;
use lutum::{Session, StructuredTurnOutcome, TextTurnEvent};
use nuillu_module::{
    AttentionReader, AttentionStreamUpdatedInbox, BlackboardReader, LlmAccess, Memo, Module,
    ModuleRunStatus, ModuleStatusReader, SpeakInbox, SpeakMailbox, SpeakRequest, UtteranceProgress,
    UtteranceProgressState, UtteranceWriter,
};
use nuillu_types::{ModuleInstanceId, ReplicaIndex, builtin};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

mod batch;

const READINESS_GATE_PROMPT: &str = r#"You are the speak-gate module.
Decide whether the speak module may emit a user-visible utterance now. You may read the current
cognitive attention-stream set, blackboard memos, memory metadata, scheduler-owned module status,
and utterance progress. You must not write attention, publish query/self-model requests, emit
utterances, or change allocation.

Speak is not currently streaming. Use a strict readiness gate before setting should_speak=true:
- The attention stream must contain the facts needed for the utterance, not only raw sensory
  observations, open questions, predictions, or instructions for another module.
- If the current topic asks for stored memory, a self/peer/world model, file evidence, or a rule,
  do not let speak use memo-only facts directly. If a memo contains the needed fact but attention
  does not, request attention-promotion.
- For questions about the speaker's own action or body/capability (for example why I/Nui did
  something), wait for self/body-model evidence in attention unless attention already contains that
  evidence.
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

When should_speak=false because a missing fact is needed for speech, include evidence_gaps that
name the source to consult, the concrete question to answer, and the exact fact that must become
visible in attention before speaking. Use memory for stable self/body/peer/world facts, file for
local design or world-rule documents, self-model for a current first-person model, and
attention-promotion when a memo already contains the needed fact but attention does not.

When should_speak=true, provide a concrete generation_hint naming the attended facts to use, the
intended addressee/frame, and any constraints on style or scope. If you cannot write such a hint,
should_speak must be false. Return only raw JSON for the structured decision; do not wrap it in
Markdown or code fences."#;

const INTERRUPTION_GATE_PROMPT: &str = r#"You are the speak-gate module.
Speak is currently streaming a user-visible utterance. Decide whether the current stream must be
cancelled and replaced by publishing a new typed SpeakRequest. You may read the current cognitive
attention-stream set, blackboard memos, memory metadata, scheduler-owned module status, and
utterance progress. You must not write attention, publish query/self-model requests, emit
utterances, or change allocation.

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
    AttentionPromotion,
}

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

fn gate_prompt_for(status: &ModuleRunStatus, progress: Option<&UtteranceProgress>) -> &'static str {
    if is_speak_streaming(status, progress) {
        INTERRUPTION_GATE_PROMPT
    } else {
        READINESS_GATE_PROMPT
    }
}

fn gate_input(
    attention_json: serde_json::Value,
    blackboard_json: serde_json::Value,
    speak_status: ModuleRunStatus,
    utterance_progress: Option<UtteranceProgress>,
) -> serde_json::Value {
    serde_json::json!({
        "attention_streams": attention_json,
        "blackboard": blackboard_json,
        "speak_module_status": speak_status,
        "current_utterance_progress": utterance_progress,
    })
}

pub struct SpeakGateModule {
    attention_updates: AttentionStreamUpdatedInbox,
    attention: AttentionReader,
    blackboard: BlackboardReader,
    module_status: ModuleStatusReader,
    memo: Memo,
    speak: SpeakMailbox,
    llm: LlmAccess,
}

impl SpeakGateModule {
    pub fn new(
        attention_updates: AttentionStreamUpdatedInbox,
        attention: AttentionReader,
        blackboard: BlackboardReader,
        module_status: ModuleStatusReader,
        memo: Memo,
        speak: SpeakMailbox,
        llm: LlmAccess,
    ) -> Self {
        Self {
            attention_updates,
            attention,
            blackboard,
            module_status,
            memo,
            speak,
            llm,
        }
    }

    #[tracing::instrument(skip_all, err(Debug, level = "warn"))]
    async fn activate(&mut self) -> Result<()> {
        let attention_json = self.attention.snapshot().await.compact_json();
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
        let lutum = self.llm.lutum().await;
        let mut session = Session::new(lutum);
        session.push_system(gate_prompt_for(&speak_status, utterance_progress.as_ref()));
        session.push_user(
            gate_input(
                attention_json,
                blackboard_json,
                speak_status,
                utterance_progress,
            )
            .to_string(),
        );

        let result = session
            .structured_turn::<SpeakGateDecision>()
            .collect()
            .await
            .context("speak-gate decision turn failed")?;

        let StructuredTurnOutcome::Structured(decision) = result.semantic else {
            anyhow::bail!("speak-gate decision turn refused");
        };

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
) {
    session.push_system(GENERATION_PROMPT);
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
    requests: SpeakInbox,
    attention: AttentionReader,
    memo: Memo,
    utterance: UtteranceWriter,
    llm: LlmAccess,
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
            requests,
            attention,
            memo,
            utterance,
            llm,
        }
    }

    #[tracing::instrument(skip_all, err(Debug, level = "warn"))]
    async fn activate(&mut self, request: SpeakRequest) -> Result<()> {
        let mut attention_json = self.attention.snapshot().await.compact_json();

        let mut draft = GenerationDraft::new(self.utterance.next_generation_id(), request);

        loop {
            self.record_streaming_progress(&draft).await;
            match self.stream_generation(attention_json, &mut draft).await? {
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
        attention_json: serde_json::Value,
        draft: &mut GenerationDraft,
    ) -> Result<GenerationStreamOutcome> {
        let lutum = self.llm.lutum().await;
        let mut session = Session::new(lutum);
        push_generation_context(&mut session, attention_json, draft);

        let mut stream = session
            .text_turn()
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
    use std::sync::Arc;

    use lutum::{
        AssistantInputItem, InputMessageRole, MessageContent, MockLlmAdapter, ModelInputItem,
        Session, SharedPoolBudgetManager, SharedPoolBudgetOptions,
    };

    use super::*;

    fn test_session() -> Session {
        let adapter = MockLlmAdapter::new();
        let budget = SharedPoolBudgetManager::new(SharedPoolBudgetOptions::default());
        Session::new(lutum::Lutum::new(Arc::new(adapter), budget))
    }

    #[test]
    fn fresh_generation_omits_assistant_prefill() {
        let draft = GenerationDraft::new(7, SpeakRequest::new("be concise", "respond"));
        let mut session = test_session();

        push_generation_context(&mut session, serde_json::json!({"streams": []}), &draft);
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
            gate_prompt_for(&ModuleRunStatus::Inactive, None).contains("not currently streaming")
        );
        assert!(
            gate_prompt_for(&ModuleRunStatus::Activating, Some(&progress))
                .contains("currently streaming")
        );
        assert!(
            gate_prompt_for(&ModuleRunStatus::AwaitingBatch, Some(&progress))
                .contains("not currently streaming")
        );
        assert!(
            gate_prompt_for(&ModuleRunStatus::Activating, None).contains("not currently streaming")
        );
    }

    #[test]
    fn gate_input_includes_module_status_and_current_utterance_progress() {
        let progress =
            UtteranceProgress::streaming(5, 1, "Mika,", "answer Mika calmly", "peer is stressed");

        let input = gate_input(
            serde_json::json!({"streams": []}),
            serde_json::json!({"memos": {}}),
            ModuleRunStatus::Activating,
            Some(progress),
        );

        assert_eq!(
            input,
            serde_json::json!({
                "attention_streams": {"streams": []},
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

    #[test]
    fn resumed_generation_keeps_id_sequence_and_pushes_assistant_prefill() {
        let mut draft = GenerationDraft::new(11, SpeakRequest::new("continue", "respond"));
        let mut session = test_session();

        assert_eq!(draft.push_delta("hello "), 0);
        assert_eq!(draft.push_delta("world"), 1);
        push_generation_context(&mut session, serde_json::json!({"streams": []}), &draft);
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
}

#[async_trait(?Send)]
impl Module for SpeakGateModule {
    type Batch = ();

    async fn next_batch(&mut self) -> Result<Self::Batch> {
        SpeakGateModule::next_batch(self).await
    }

    async fn activate(&mut self, _batch: &Self::Batch) -> Result<()> {
        SpeakGateModule::activate(self).await
    }
}

#[async_trait(?Send)]
impl Module for SpeakModule {
    type Batch = batch::NextBatch;

    async fn next_batch(&mut self) -> Result<Self::Batch> {
        SpeakModule::next_batch(self).await
    }

    async fn activate(&mut self, batch: &Self::Batch) -> Result<()> {
        SpeakModule::activate(self, batch.request.clone()).await
    }
}
