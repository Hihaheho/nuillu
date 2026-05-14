use std::borrow::Cow;

use anyhow::{Context, Result};
use async_trait::async_trait;
use futures::StreamExt;
use lutum::{Session, StructuredTurnOutcome, TextTurnEvent};
use nuillu_module::{
    CognitionLogReader, CognitionLogUpdated, CognitionLogUpdatedInbox, LlmAccess, Memo, Module,
    SceneReader, UtteranceProgress,
};

use crate::utterance::UtteranceWriter;
use nuillu_types::builtin;
use schemars::{JsonSchema, Schema, SchemaGenerator};
use serde::{Deserialize, Serialize};

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
                cx.core_policies(),
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
                cx.core_policies(),
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
            .filter(|record| record.source.module != builtin::memory_recombination())
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
impl Module for SpeakModule {
    type Batch = SpeakBatch;

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
    use std::rc::Rc;

    use lutum::{
        AssistantInputItem, InputMessageRole, MessageContent, MockLlmAdapter, ModelInputItem,
    };
    use nuillu_blackboard::{
        ActivationRatio, Blackboard, BlackboardCommand, CognitionLogEntry, ModuleConfig,
        ResourceAllocation,
    };
    use nuillu_module::ports::{Clock, SystemClock};
    use nuillu_module::{CognitionLogUpdated, ModuleRegistry, Participant};
    use nuillu_types::{ModuleInstanceId, ReplicaIndex, builtin};

    use super::*;
    use crate::test_support::*;

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
