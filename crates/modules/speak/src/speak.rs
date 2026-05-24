use std::borrow::Cow;
use std::future::Future;
use std::pin::Pin;

use anyhow::{Context, Result};
use async_trait::async_trait;
use futures::StreamExt;
use lutum::{Session, StructuredTurnOutcome, TextStepOutcomeWithTools, TextTurnEvent};
use nuillu_module::{
    CognitionLogReader, CognitionLogUpdated, CognitionLogUpdatedInbox, LlmAccess, Memo, Module,
    SceneReader, UtteranceProgress,
};

use crate::utterance::UtteranceWriter;
use nuillu_types::builtin;
use schemars::{JsonSchema, Schema, SchemaGenerator};
use serde::{Deserialize, Serialize};

const SHOULD_SPEAK_PROMPT: &str = r#"Decide whether the agent should speak now.
Use only the cognition log and the available target schema. If speech is not warranted, finish without calling a tool.
If speaking is warranted, call should_speak exactly once. Choose the participant whose question, request, warning, or need should be answered. Do not choose a participant merely because they are the topic, threat, object of advice, or quoted speaker. Use "everyone" only for explicit group/broadcast speech.
Put the speech-facing transformation of the cognition log in speech_content. It is the information that should survive into speech, with perspective, deixis, and addressee adjusted for outward utterance.
speech_content is not hidden reasoning, not a rubric, and not a generic summary. It should contain the load-bearing fact, answer, warning, advice, visible absence, or unknown-state evidence that the listener needs.
For questions or requests, transform the relevant cognition into an answer. Preserve answer polarity: yes/no/unknown must remain visible when supported by the cognition log.
For self-directed cognition, transform only the listener-relevant implication into outward substance. Keep first person only when the speaker is reporting perception, knowledge, uncertainty, consent, or shared action that directly answers the listener.
Do not invent policy, actions, or facts not supported by the cognition log. If the cognition log only supports a limited warning or uncertainty, keep speech_content limited.
For unknown evidence, make speech_content say unknown and include the concrete visible absence or missing evidence."#;

const GENERATION_PROMPT: &str = r#"Render the accepted should_speak tool call as one concise in-world utterance to the selected target.
speech_content is already the speech-facing transformation of the cognition log. Render that transformed information; do not redo target selection or add a new plan.
Preserve its answer polarity, addressee-facing perspective, direct warnings, advice, uncertainty, and visible-absence evidence.
Use the cognition log only to keep wording grounded. Do not add facts, turn an answer back into a question, turn listener-facing substance into a self-directed note, weaken direct content into vague caution, or merely restate the situation.
Do not mention tools, fields, lookup, reasoning, modules, memos, prompts, rubrics, or evaluation mechanics."#;

const PARTIAL_CONTINUATION_PROMPT: &str = "Continue the partial utterance from where it stopped.";

const ABORT_JUDGE_PROMPT: &str = r#"A speech is in progress. Set inform_now=true only if the new cognition entries contradict the speech, shift the safety/peer/task constraint, or change who should be addressed."#;

tokio::task_local! {
    /// JSON Schema for `ShouldSpeakArgs.target` derived from the live
    /// `SceneReader`. `.scope`d around each planning turn so the LLM sees the
    /// current host-constrained target enum.
    static SPEECH_TARGET_SCHEMA: Schema;
}

fn fallback_speech_target_schema() -> Schema {
    Schema::try_from(serde_json::json!({ "type": "string" }))
        .expect("fallback speech target schema must be a JSON object")
}
#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
struct AbortJudgement {
    inform_now: bool,
    rationale: String,
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

#[lutum::tool_input(name = "should_speak", output = ShouldSpeakOutput)]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
/// Call only when speech should be emitted. The input carries cognition-log
/// information transformed for outward speech, not hidden analysis.
struct ShouldSpeakArgs {
    /// The participant who should hear the utterance.
    target: SpeechTarget,
    /// Speech-facing information to render for `target`, with perspective and
    /// addressee adjusted from the cognition log.
    speech_content: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
struct ShouldSpeakOutput {
    accepted: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema, lutum::Toolset)]
enum SpeakTools {
    ShouldSpeak(ShouldSpeakArgs),
}

#[derive(Clone, Debug)]
struct PlannedSpeech {
    args: ShouldSpeakArgs,
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

fn render_should_speak_call(args: &ShouldSpeakArgs, accepted_target: &str) -> String {
    format!(
        "target: {}\nspeech_content: {}",
        accepted_target.trim(),
        args.speech_content.trim()
    )
}

fn format_generation_input(
    cognition_context: &str,
    args: &ShouldSpeakArgs,
    draft: &GenerationDraft,
) -> String {
    format!(
        "Current cognition log:\n{}\n\nSpeech target: {}\n\nAccepted should_speak tool call:\n{}",
        cognition_context.trim(),
        draft.target.trim(),
        render_should_speak_call(args, &draft.target)
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
    args: &ShouldSpeakArgs,
    draft: &GenerationDraft,
    generation_prompt: &str,
) {
    session.push_system(generation_prompt);
    session.push_user(format_generation_input(cognition_context, args, draft));
    if !draft.accumulated.is_empty() {
        session.push_system(PARTIAL_CONTINUATION_PROMPT);
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

type AbortJudgeFuture = Pin<Box<dyn Future<Output = Result<bool>> + 'static>>;

async fn poll_pending_abort_judge(pending: &mut Option<AbortJudgeFuture>) -> Result<bool> {
    pending
        .as_mut()
        .expect("pending abort judge is only polled when present")
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
    plan_prompt: std::sync::OnceLock<String>,
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
            cognition_updates,
            cognition_log,
            memo,
            utterance,
            llm,
            scene,
            plan_prompt: std::sync::OnceLock::new(),
            generation_prompt: std::sync::OnceLock::new(),
            abort_judge_prompt: std::sync::OnceLock::new(),
        }
    }

    fn plan_prompt(&self, cx: &nuillu_module::ActivateCx<'_>) -> &str {
        self.plan_prompt.get_or_init(|| {
            nuillu_module::format_identity_system_prompt(
                SHOULD_SPEAK_PROMPT,
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

    fn abort_judge_prompt(&self, cx: &nuillu_module::ActivateCx<'_>) -> &str {
        self.abort_judge_prompt.get_or_init(|| {
            nuillu_module::format_identity_system_prompt(
                ABORT_JUDGE_PROMPT,
                cx.identity_memories(),
                cx.core_policies(),
                cx.now(),
            )
        })
    }

    #[tracing::instrument(skip_all, err(Debug, level = "warn"))]
    async fn activate(
        &mut self,
        cx: &nuillu_module::ActivateCx<'_>,
        batch: &SpeakBatch,
    ) -> Result<()> {
        let _update_count = batch.updates.len();
        let mut cognition_context = self.speech_cognition_context().await;
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
                    cognition_context = self.speech_cognition_context().await;
                }
                GenerationStreamOutcome::Aborted => {
                    cognition_context = self.speech_cognition_context().await;
                    let Some(new_plan) = self.plan_speech(cx, &cognition_context).await? else {
                        return Ok(());
                    };
                    plan = new_plan;
                    draft = GenerationDraft::new(self.utterance.next_generation_id(), &plan.target);
                }
            }
        }
    }

    async fn speech_cognition_context(&self) -> String {
        let snapshot = self.cognition_log.snapshot().await;
        let mut lines = Vec::new();
        for record in snapshot.logs() {
            if record.source.module == builtin::memory_recombination() {
                continue;
            }
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

    async fn plan_speech(
        &mut self,
        cx: &nuillu_module::ActivateCx<'_>,
        cognition_context: &str,
    ) -> Result<Option<PlannedSpeech>> {
        let mut session = Session::new();
        session.push_system(self.plan_prompt(cx));
        session.push_user(format!(
            "Current cognition log:\n{}",
            cognition_context.trim()
        ));

        let lutum = self.llm.lutum().await;
        let target_schema = self.scene.target_schema();
        let validation_schema = target_schema.clone();
        let outcome = SPEECH_TARGET_SCHEMA
            .scope(target_schema, async {
                session
                    .text_turn(&lutum)
                    .tools::<SpeakTools>()
                    .available_tools([SpeakToolsSelector::ShouldSpeak])
                    .collect()
                    .await
                    .context("speak should_speak turn failed")
            })
            .await?;
        let TextStepOutcomeWithTools::NeedsTools(round) = outcome else {
            return Ok(None);
        };

        let mut selected = None;
        for call in round.tool_calls.iter().cloned() {
            let SpeakToolsCall::ShouldSpeak(call) = call;
            if selected.is_none() {
                selected = Some(call.input.clone());
            }
        }
        let Some(args) = selected else {
            return Ok(None);
        };
        let target = args.target.as_str();
        let target = target.trim().to_owned();
        if !is_target_allowed_by_schema(&validation_schema, &target) {
            return Ok(None);
        }
        Ok(Some(PlannedSpeech { args, target }))
    }

    async fn stream_generation(
        &mut self,
        cx: &nuillu_module::ActivateCx<'_>,
        cognition_context: String,
        args: &ShouldSpeakArgs,
        draft: &mut GenerationDraft,
    ) -> Result<GenerationStreamOutcome> {
        let stream_started_at = cx.now();
        let cognition_context_at_start = cognition_context.clone();

        let mut session = Session::new();
        push_generation_context(
            &mut session,
            &cognition_context,
            args,
            draft,
            self.generation_prompt(cx),
        );

        let lutum = self.llm.lutum().await;
        let mut stream = session
            .text_turn(&lutum)
            .stream()
            .await
            .context("speak generation stream failed")?;
        let mut pending_judge: Option<AbortJudgeFuture> = None;
        let mut buffered_entries = Vec::new();

        loop {
            tokio::select! {
                biased;

                verdict = poll_pending_abort_judge(&mut pending_judge), if pending_judge.is_some() => {
                    pending_judge = None;
                    if verdict? {
                        return Ok(GenerationStreamOutcome::Aborted);
                    }
                    let next_entries = std::mem::take(&mut buffered_entries);
                    if !next_entries.is_empty() {
                        pending_judge = Some(self.start_abort_judge(
                            cx,
                            &cognition_context_at_start,
                            next_entries,
                        ));
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

                    if pending_judge.is_some() {
                        buffered_entries = new_entries;
                    } else {
                        pending_judge = Some(self.start_abort_judge(
                            cx,
                            &cognition_context_at_start,
                            new_entries,
                        ));
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
            .filter(|record| record.source.module != builtin::memory_recombination())
            .flat_map(|record| record.entries.iter())
            .filter(|entry| entry.at > threshold)
            .map(|entry| entry.text.clone())
            .collect()
    }

    fn start_abort_judge(
        &self,
        cx: &nuillu_module::ActivateCx<'_>,
        cognition_context_at_start: &str,
        new_entries: Vec<String>,
    ) -> AbortJudgeFuture {
        let llm = self.llm.clone();
        let abort_judge_prompt = self.abort_judge_prompt(cx).to_owned();
        let cognition_context_at_start = cognition_context_at_start.to_owned();
        Box::pin(Self::judge_abort_owned(
            llm,
            abort_judge_prompt,
            cognition_context_at_start,
            new_entries,
        ))
    }

    async fn judge_abort_owned(
        llm: LlmAccess,
        abort_judge_prompt: String,
        cognition_context_at_start: String,
        new_entries: Vec<String>,
    ) -> Result<bool> {
        let mut session = Session::new();
        session.push_system(abort_judge_prompt);
        session.push_user(format_abort_judge_input(
            &cognition_context_at_start,
            &new_entries,
        ));

        let lutum = llm.lutum().await;
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
impl Module for SpeakModule {
    type Batch = SpeakBatch;

    fn id() -> &'static str {
        "speak"
    }

    fn role_description() -> &'static str {
        "Emits the agent's spoken utterances into its world when activated by allocation and when the cognition log supports speech. It calls an optional should_speak tool to choose an addressee and utterance focus; if no tool is called, it stays silent. It cannot inspect memo logs, allocation guidance, or query results directly."
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
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};
    use std::task::{Context, Poll, Waker};
    use std::time::Duration;

    use futures::{Stream, stream};
    use lutum::{
        AdapterStructuredTurn, AdapterTextTurn, AgentError, AssistantInputItem, AssistantTurnItem,
        AssistantTurnView, ErasedStructuredTurnEvent, ErasedStructuredTurnEventStream,
        ErasedTextTurnEvent, ErasedTextTurnEventStream, FinishReason, InputMessageRole,
        MessageContent, MockLlmAdapter, MockTextScenario, ModelInput, ModelInputItem, RawJson,
        RawTextTurnEvent, SharedPoolBudgetManager, SharedPoolBudgetOptions, TurnAdapter, Usage,
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

    struct GatedAbortAdapter {
        text_delta_gate: Arc<PollGate>,
        text_started: Mutex<Option<tokio::sync::oneshot::Sender<()>>>,
        judge_scripts: Mutex<VecDeque<JudgeScript>>,
        structured_calls: AtomicUsize,
    }

    struct JudgeScript {
        release_gate: Arc<PollGate>,
        started: Option<tokio::sync::oneshot::Sender<()>>,
        judge_inform_now: bool,
    }

    impl GatedAbortAdapter {
        fn new(
            text_delta_gate: Arc<PollGate>,
            text_started: tokio::sync::oneshot::Sender<()>,
            judge_scripts: impl IntoIterator<Item = JudgeScript>,
        ) -> Self {
            Self {
                text_delta_gate,
                text_started: Mutex::new(Some(text_started)),
                judge_scripts: Mutex::new(judge_scripts.into_iter().collect()),
                structured_calls: AtomicUsize::new(0),
            }
        }
    }

    impl JudgeScript {
        fn new(
            release_gate: Arc<PollGate>,
            started: tokio::sync::oneshot::Sender<()>,
            judge_inform_now: bool,
        ) -> Self {
            Self {
                release_gate,
                started: Some(started),
                judge_inform_now,
            }
        }
    }

    #[async_trait::async_trait]
    impl TurnAdapter for GatedAbortAdapter {
        async fn text_turn(
            &self,
            _input: ModelInput,
            _turn: AdapterTextTurn,
        ) -> Result<ErasedTextTurnEventStream, AgentError> {
            if let Some(text_started) = self.text_started.lock().unwrap().take() {
                let _ = text_started.send(());
            }
            Ok(Box::pin(GatedTextStream::new(
                self.text_delta_gate.clone(),
                "Koro, stay close.",
            )))
        }

        async fn structured_turn(
            &self,
            _input: ModelInput,
            _turn: AdapterStructuredTurn,
        ) -> Result<ErasedStructuredTurnEventStream, AgentError> {
            let call = self.structured_calls.fetch_add(1, Ordering::SeqCst);
            let script = self
                .judge_scripts
                .lock()
                .unwrap()
                .pop_front()
                .unwrap_or_else(|| panic!("missing abort judge script for call {call}"));
            if let Some(judge_started) = script.started {
                let _ = judge_started.send(());
            }
            script.release_gate.wait().await;
            Ok(abort_judge_stream(script.judge_inform_now))
        }
    }

    fn abort_judge_stream(inform_now: bool) -> ErasedStructuredTurnEventStream {
        let json = serde_json::json!({
            "inform_now": inform_now,
            "rationale": "test judgement"
        })
        .to_string();
        Box::pin(stream::iter(vec![
            Ok(ErasedStructuredTurnEvent::Started {
                request_id: Some("abort-judge".into()),
                model: "mock".into(),
            }),
            Ok(ErasedStructuredTurnEvent::StructuredOutputChunk {
                json_delta: json.clone(),
            }),
            Ok(ErasedStructuredTurnEvent::StructuredOutputReady(
                RawJson::parse(json.clone()).unwrap(),
            )),
            Ok(ErasedStructuredTurnEvent::Completed {
                request_id: Some("abort-judge".into()),
                finish_reason: FinishReason::Stop,
                usage: Usage::zero(),
                committed_turn: Arc::new(AssistantTurnView::from_items(&[
                    AssistantTurnItem::Text(json),
                ])),
            }),
        ]))
    }

    fn should_speak_scenario(target: &str, speech_content: &str) -> MockTextScenario {
        let arguments_json = serde_json::json!({
            "target": target,
            "speech_content": speech_content
        })
        .to_string();
        MockTextScenario::events(vec![
            Ok(RawTextTurnEvent::Started {
                request_id: Some("should-speak".into()),
                model: "mock".into(),
            }),
            Ok(RawTextTurnEvent::ToolCallChunk {
                id: "call-should-speak".into(),
                name: "should_speak".into(),
                arguments_json_delta: arguments_json,
            }),
            Ok(RawTextTurnEvent::Completed {
                request_id: Some("should-speak".into()),
                finish_reason: FinishReason::ToolCall,
                usage: Usage::zero(),
            }),
        ])
    }

    fn no_speech_scenario() -> MockTextScenario {
        MockTextScenario::events(vec![
            Ok(RawTextTurnEvent::Started {
                request_id: Some("should-speak".into()),
                model: "mock".into(),
            }),
            Ok(RawTextTurnEvent::Completed {
                request_id: Some("should-speak".into()),
                finish_reason: FinishReason::Stop,
                usage: Usage::zero(),
            }),
        ])
    }

    fn test_should_speak_args(target: &str) -> ShouldSpeakArgs {
        ShouldSpeakArgs {
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

    struct CapturingDeltaSink {
        deltas: Rc<RefCell<Vec<(String, u64, u32, String)>>>,
        first_delta: RefCell<Option<tokio::sync::oneshot::Sender<(String, u64, u32, String)>>>,
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
                *module_sink.borrow_mut() = Some(SpeakModule::new(
                    caps.cognition_log_updated_inbox(),
                    caps.cognition_log_reader(),
                    caps.memo(),
                    UtteranceWriter::new(
                        caps.owner().clone(),
                        caps.blackboard(),
                        utterance_sink_for_closure.clone(),
                        caps.clock(),
                    ),
                    caps.llm_access(),
                    caps.scene_reader(),
                ));
                SpeakStub
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
            nuillu_module::SessionCompactionRuntime::new(
                compaction_lutum.lutum().clone(),
                nuillu_module::LlmConcurrencyLimiter::new(None),
                nuillu_types::ModelTier::Cheap,
                nuillu_module::SessionCompactionPolicy::default(),
            ),
            now,
        )
    }

    async fn activate_once_with_adapter(
        adapter: MockLlmAdapter,
        participants: impl IntoIterator<Item = Participant>,
        cognition: impl Into<String>,
    ) -> (Blackboard, Rc<RefCell<Vec<(String, String)>>>) {
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
        SpeakModule::activate(&mut module, &cx, &batch)
            .await
            .unwrap();
        (blackboard, completed)
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

    #[test]
    fn fresh_generation_omits_assistant_prefill() {
        let draft = GenerationDraft::new(7, "Koro");
        let args = test_should_speak_args("Koro");
        let mut session = test_session();

        push_generation_context(&mut session, "none", &args, &draft, GENERATION_PROMPT);
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
        assert!(
            text.contains("speech_content: Tell Koro to stay close because Koro asks for help.")
        );
        assert!(!text.contains("rationale"));
        assert!(!text.contains("allocation"));
        assert!(!text.contains("partial_utterance"));
    }

    #[test]
    fn unknown_evidence_plan_renders_uncertainty_and_visible_absence() {
        let draft = GenerationDraft::new(8, "Pibi");
        let args = ShouldSpeakArgs {
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
        assert!(!prompt.contains("You are part of a cognitive system"));
        assert!(!prompt.contains("- cognition-gate:"));
        assert!(!prompt.contains("- query-memory:"));
        assert!(!SHOULD_SPEAK_PROMPT.contains("\"self\""));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn speak_selects_target_from_cognition_log_before_streaming() {
        let adapter = MockLlmAdapter::new()
            .with_text_scenario(should_speak_scenario("Koro", "Tell Koro to stay close."))
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
                *module_sink.borrow_mut() = Some(SpeakModule::new(
                    caps.cognition_log_updated_inbox(),
                    caps.cognition_log_reader(),
                    caps.memo(),
                    UtteranceWriter::new(
                        caps.owner().clone(),
                        caps.blackboard(),
                        utterance_sink_for_closure.clone(),
                        caps.clock(),
                    ),
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
    }

    #[tokio::test(flavor = "current_thread")]
    async fn speak_stays_silent_when_should_speak_tool_is_not_called() {
        let adapter = MockLlmAdapter::new().with_text_scenario(no_speech_scenario());

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
    async fn speak_stays_silent_for_empty_planned_target() {
        let adapter = MockLlmAdapter::new()
            .with_text_scenario(should_speak_scenario("", "Tell Koro to stay close."));

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
            .with_text_scenario(should_speak_scenario("Koro", "Tell Pibi to stay close."));

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
    async fn speak_stream_drains_delta_while_abort_judge_is_pending() {
        let text_delta_gate = Arc::new(PollGate::default());
        let judge_release_gate = Arc::new(PollGate::default());
        let (text_started_tx, text_started_rx) = tokio::sync::oneshot::channel();
        let (judge_started_tx, judge_started_rx) = tokio::sync::oneshot::channel();
        let adapter = Arc::new(GatedAbortAdapter::new(
            text_delta_gate.clone(),
            text_started_tx,
            [JudgeScript::new(
                judge_release_gate.clone(),
                judge_started_tx,
                false,
            )],
        ));
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
        let args = test_should_speak_args("Koro");
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
                panic!("stream ended before abort judge started");
            }
            result = judge_started_rx => {
                result.expect("abort judge should start");
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
                    .expect("text delta should be emitted while abort judge is pending")
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
        judge_release_gate.release();
    }

    #[tokio::test(flavor = "current_thread")]
    async fn speak_rejudges_buffered_cognition_after_false_abort_judge() {
        let text_delta_gate = Arc::new(PollGate::default());
        let first_judge_gate = Arc::new(PollGate::default());
        let second_judge_gate = Arc::new(PollGate::default());
        let (text_started_tx, text_started_rx) = tokio::sync::oneshot::channel();
        let (first_judge_started_tx, first_judge_started_rx) = tokio::sync::oneshot::channel();
        let (second_judge_started_tx, second_judge_started_rx) = tokio::sync::oneshot::channel();
        let adapter = Arc::new(GatedAbortAdapter::new(
            text_delta_gate,
            text_started_tx,
            [
                JudgeScript::new(first_judge_gate.clone(), first_judge_started_tx, false),
                JudgeScript::new(second_judge_gate.clone(), second_judge_started_tx, true),
            ],
        ));
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

        let clock = SystemClock;
        let now = clock.now();
        let cx = test_activate_cx(&module, now).await;
        let mut draft = GenerationDraft::new(0, "Koro");
        let args = test_should_speak_args("Koro");
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
            "Koro notices the first new hazard.",
        )
        .await;

        tokio::select! {
            result = &mut stream => {
                let _ = result;
                panic!("stream ended before first abort judge started");
            }
            result = first_judge_started_rx => {
                result.expect("first abort judge should start");
            }
        }

        publish_cognition_update(
            &blackboard,
            &caps,
            now + chrono::Duration::milliseconds(2),
            "Koro notices a second, more urgent hazard.",
        )
        .await;
        first_judge_gate.release();

        tokio::select! {
            result = &mut stream => {
                let _ = result;
                panic!("stream ended before buffered cognition triggered a second judge");
            }
            result = tokio::time::timeout(Duration::from_millis(100), second_judge_started_rx) => {
                result
                    .expect("buffered cognition should start a second judge")
                    .expect("second abort judge should start");
            }
        }

        second_judge_gate.release();
        let outcome = tokio::time::timeout(Duration::from_millis(100), &mut stream)
            .await
            .expect("second judge should finish the stream")
            .unwrap();
        assert!(matches!(outcome, GenerationStreamOutcome::Aborted));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn speak_abort_judge_true_wins_over_ready_text_delta() {
        let text_delta_gate = Arc::new(PollGate::default());
        let judge_release_gate = Arc::new(PollGate::default());
        let (text_started_tx, text_started_rx) = tokio::sync::oneshot::channel();
        let (judge_started_tx, judge_started_rx) = tokio::sync::oneshot::channel();
        let adapter = Arc::new(GatedAbortAdapter::new(
            text_delta_gate.clone(),
            text_started_tx,
            [JudgeScript::new(
                judge_release_gate.clone(),
                judge_started_tx,
                true,
            )],
        ));
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

        let clock = SystemClock;
        let now = clock.now();
        let cx = test_activate_cx(&module, now).await;
        let mut draft = GenerationDraft::new(0, "Koro");
        let args = test_should_speak_args("Koro");
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
                panic!("stream ended before abort judge started");
            }
            result = judge_started_rx => {
                result.expect("abort judge should start");
            }
        }

        text_delta_gate.release();
        judge_release_gate.release();
        let outcome = tokio::time::timeout(Duration::from_millis(100), &mut stream)
            .await
            .expect("ready abort judge should finish the stream")
            .unwrap();

        assert!(matches!(outcome, GenerationStreamOutcome::Aborted));
        assert!(deltas.borrow().is_empty());
    }

    #[test]
    fn resumed_generation_keeps_id_sequence_and_pushes_assistant_prefill() {
        let mut draft = GenerationDraft::new(11, "Koro");
        let args = test_should_speak_args("Koro");
        let mut session = test_session();

        assert_eq!(draft.push_delta("hello "), 0);
        assert_eq!(draft.push_delta("world"), 1);
        push_generation_context(&mut session, "none", &args, &draft, GENERATION_PROMPT);
        let items = session.input().items();

        assert_eq!(draft.generation_id, 11);
        assert_eq!(draft.sequence, 2);
        assert_eq!(draft.accumulated, "hello world");
        assert_eq!(items.len(), 4);
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
        assert!(matches!(
            &items[2],
            ModelInputItem::Message {
                role: InputMessageRole::System,
                ..
            }
        ));
        let ModelInputItem::Assistant(AssistantInputItem::Text(text)) = &items[3] else {
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
