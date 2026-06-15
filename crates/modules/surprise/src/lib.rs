use anyhow::{Context, Result, anyhow};
use async_trait::async_trait;
use lutum::{Session, TextStepOutcomeWithTools, ToolResult};
use nuillu_module::{
    AttentionControlRequest, AttentionControlRequestMailbox, BlackboardReader, CognitionLogReader,
    CognitionLogUpdatedInbox, LlmAccess, LlmContextWindow, Memo, Module, SessionAutoCompaction,
    SessionCompactionConfig, SessionCompactionProtectedPrefix, ensure_persistent_session_seeded,
    push_formatted_cognition_log_batch, push_formatted_memo_log_batch,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

mod batch;

const SYSTEM_PROMPT: &str = r#"You are the surprise module.
Detect unexpected cognition-log entries. If predict memo log entries are present, frame the
assessment as divergence from pending predictions. If no predict memo log is present, judge novelty against recent
cognition-log history. Do not generate forward predictions.

Predict and other faculty notes arrive as system context. New cognition-log entries are appended
to the session above. Assess each batch using the developer instruction for this activation.

Use exactly one tool per activation:
- preserve_unexpected_event when new cognition diverges from a pending prediction, breaks recent
  continuity in a meaningful way, or is novel enough to deserve memory preservation.
- mark_expected_event when new cognition fits pending predictions or recent history and does not
  deserve preservation.

When predict memo-log entries are present, compare the new cognition against those pending
predictions before choosing. A direct contradiction of a predicted state or action is significant
surprise and should be preserved. Do not generate forward predictions. Do not use final assistant
text as an output channel."#;

const EXPECTED_SURPRISE_MEMO: &str =
    "Surprise assessment: expected\nNo memory preservation requested.";

const ACTIVATION_INPUT: &str = "Assess whether the new cognition is surprising relative to pending predictions and recent history.";

const COMPACTED_SURPRISE_SESSION_PREFIX: &str = "Compacted surprise session history:";
const MEMO_CONTEXT_WINDOW: LlmContextWindow = LlmContextWindow::new(8, 1_200, 4_800);
const COGNITION_CONTEXT_WINDOW: LlmContextWindow = LlmContextWindow::new(12, 600, 4_800);
const TOOL_TURN_MAX_OUTPUT_TOKENS: u32 = 512;
const SESSION_COMPACTION_FOCUS: &str = r#"Preserve prior surprise assessments, predict memo-log
facts, significant events, memory preservation requests, and cognition-log context needed for future
surprise checks."#;

pub fn session_auto_compaction() -> SessionAutoCompaction {
    SessionAutoCompaction::new(
        SessionCompactionConfig::default(),
        SessionCompactionProtectedPrefix::LeadingSystemAndIdentitySeed,
        COMPACTED_SURPRISE_SESSION_PREFIX,
        SESSION_COMPACTION_FOCUS,
    )
}

const HIGH_SURPRISE_THRESHOLD: f32 = 0.75;

#[lutum::tool_input(
    name = "preserve_unexpected_event",
    output = PreserveUnexpectedEventOutput
)]
#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct PreserveUnexpectedEventArgs {
    pub content: String,
    pub surprise: f32,
    pub reason: String,
}

impl PartialEq for PreserveUnexpectedEventArgs {
    fn eq(&self, other: &Self) -> bool {
        self.content == other.content
            && self.surprise.to_bits() == other.surprise.to_bits()
            && self.reason == other.reason
    }
}

impl Eq for PreserveUnexpectedEventArgs {}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct PreserveUnexpectedEventOutput {
    pub preserved: bool,
}

#[lutum::tool_input(name = "mark_expected_event", output = MarkExpectedEventOutput)]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct MarkExpectedEventArgs {
    pub reason: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct MarkExpectedEventOutput {
    pub recorded: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema, lutum::Toolset)]
pub enum SurpriseTools {
    PreserveUnexpectedEvent(PreserveUnexpectedEventArgs),
    MarkExpectedEvent(MarkExpectedEventArgs),
}

pub struct SurpriseModule {
    updates: CognitionLogUpdatedInbox,
    cognition_log: CognitionLogReader,
    blackboard: BlackboardReader,
    attention_control: AttentionControlRequestMailbox,
    memo: Memo,
    llm: LlmAccess,
    session: Session,
    system_prompt: std::sync::OnceLock<String>,
}

impl SurpriseModule {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        updates: CognitionLogUpdatedInbox,
        cognition_log: CognitionLogReader,
        blackboard: BlackboardReader,
        attention_control: AttentionControlRequestMailbox,
        memo: Memo,
        llm: LlmAccess,
        session: Session,
    ) -> Self {
        Self {
            updates,
            cognition_log,
            blackboard,
            attention_control,
            memo,
            llm,
            session,
            system_prompt: std::sync::OnceLock::new(),
        }
    }

    fn ensure_session_seeded(&mut self, cx: &nuillu_module::ActivateCx<'_>) {
        let system_prompt = self.system_prompt(cx).to_owned();
        ensure_persistent_session_seeded(
            &mut self.session,
            system_prompt,
            cx.identity_memories(),
            cx.now(),
        );
    }

    fn system_prompt(&self, cx: &nuillu_module::ActivateCx<'_>) -> &str {
        self.system_prompt.get_or_init(|| {
            nuillu_module::format_policy_system_prompt(SYSTEM_PROMPT, cx.core_policies())
        })
    }

    #[tracing::instrument(skip_all, err(Debug, level = "warn"))]
    async fn activate(&mut self, cx: &nuillu_module::ActivateCx<'_>) -> Result<()> {
        self.ensure_session_seeded(cx);

        let unread_cognition = self.cognition_log.unread_events().await;
        let unread_memo_logs = self.blackboard.unread_memo_logs().await;

        let lutum = self.llm.lutum().await;
        let outcome = {
            push_formatted_memo_log_batch(
                &mut self.session,
                &unread_memo_logs,
                cx.now(),
                MEMO_CONTEXT_WINDOW,
            );
            push_formatted_cognition_log_batch(
                &mut self.session,
                &unread_cognition,
                cx.now(),
                COGNITION_CONTEXT_WINDOW,
            );
            self.session.push_ephemeral_developer(ACTIVATION_INPUT);
            self.session
                .text_turn()
                .tools::<SurpriseTools>()
                .available_tools([
                    SurpriseToolsSelector::PreserveUnexpectedEvent,
                    SurpriseToolsSelector::MarkExpectedEvent,
                ])
                .require_any_tool()
                .max_output_tokens(TOOL_TURN_MAX_OUTPUT_TOKENS)
                .collect_controlled_with(
                    &lutum,
                    nuillu_module::AbortOnAvailableToolNameInText::new(),
                )
                .await
                .map_err(|error| {
                    if error
                        .to_string()
                        .contains("required tool call was not produced")
                    {
                        anyhow!("surprise finished without required tool call")
                    } else {
                        anyhow!(error).context("surprise assessment turn failed")
                    }
                })?
        };

        let round = match outcome {
            TextStepOutcomeWithTools::Finished(result) => {
                cx.compact_and_save(&mut self.session, result.usage).await?;
                anyhow::bail!("surprise finished without required tool call");
            }
            TextStepOutcomeWithTools::FinishedNoOutput(result) => {
                cx.compact_and_save(&mut self.session, result.usage).await?;
                anyhow::bail!("surprise finished without required tool call");
            }
            TextStepOutcomeWithTools::NeedsTools(round) => round,
        };
        let usage = round.usage;
        let mut memo: Option<String> = None;
        let mut memo_cognitive = false;
        let mut results: Vec<ToolResult> = Vec::new();
        nuillu_module::emit_trace_tool_calls(&round.tool_calls);
        // The LLM may return multiple tool calls; adopt the first valid decision only.
        for call in round.tool_calls.iter().cloned() {
            match call {
                SurpriseToolsCall::PreserveUnexpectedEvent(call) => {
                    let first_decision = memo.is_none();
                    if first_decision {
                        if let Some(args) = normalize_preserve_args(call.input.clone()) {
                            memo = Some(render_surprise_memo(&args));
                            memo_cognitive = true;
                        }
                    }
                    let output = if first_decision {
                        self.preserve_unexpected_event(call.input.clone()).await
                    } else {
                        PreserveUnexpectedEventOutput { preserved: false }
                    };
                    results.push(
                        call.complete(output)
                            .context("complete preserve_unexpected_event tool call")?,
                    );
                }
                SurpriseToolsCall::MarkExpectedEvent(call) => {
                    let first_decision = memo.is_none();
                    if first_decision {
                        memo = Some(render_expected_memo(&call.input));
                    }
                    results.push(
                        call.complete(MarkExpectedEventOutput {
                            recorded: first_decision,
                        })
                        .context("complete mark_expected_event tool call")?,
                    );
                }
            }
        }
        round
            .commit(&mut self.session, results)
            .context("commit surprise tool round")?;
        cx.compact_and_save(&mut self.session, usage).await?;

        let Some(memo) = memo else {
            anyhow::bail!("surprise finished without a valid tool decision");
        };
        if memo_cognitive {
            self.memo.write_cognitive(memo).await;
        } else {
            self.memo.write(memo).await;
        }
        Ok(())
    }

    async fn preserve_unexpected_event(
        &self,
        args: PreserveUnexpectedEventArgs,
    ) -> PreserveUnexpectedEventOutput {
        let Some(args) = normalize_preserve_args(args) else {
            return PreserveUnexpectedEventOutput { preserved: false };
        };
        let _ = self
            .attention_control
            .publish(AttentionControlRequest::new(
                render_memory_attention_request(&args),
            ))
            .await;
        PreserveUnexpectedEventOutput { preserved: true }
    }
}

fn normalize_preserve_args(
    args: PreserveUnexpectedEventArgs,
) -> Option<PreserveUnexpectedEventArgs> {
    let content = args.content.trim();
    if content.is_empty() {
        return None;
    }
    if !args.surprise.is_finite() {
        return None;
    }
    Some(PreserveUnexpectedEventArgs {
        content: content.to_owned(),
        surprise: args.surprise.clamp(0.0, 1.0),
        reason: args.reason.trim().to_owned(),
    })
}

fn memory_preservation_priority(surprise: f32) -> &'static str {
    if surprise >= HIGH_SURPRISE_THRESHOLD {
        "high-priority memory preservation"
    } else {
        "normal-priority memory preservation"
    }
}

fn render_memory_attention_request(args: &PreserveUnexpectedEventArgs) -> String {
    format!(
        "{}: {} Reason: {}",
        memory_preservation_priority(args.surprise),
        args.content,
        args.reason,
    )
}

fn render_surprise_memo(args: &PreserveUnexpectedEventArgs) -> String {
    format!(
        "Surprise assessment: unexpected\nSurprise: {:.2}\nMemory preservation request:\nContent: {}\nReason: {}",
        args.surprise, args.content, args.reason,
    )
}

fn render_expected_memo(args: &MarkExpectedEventArgs) -> String {
    let reason = args.reason.trim();
    if reason.is_empty() {
        return EXPECTED_SURPRISE_MEMO.to_owned();
    }
    format!("{EXPECTED_SURPRISE_MEMO}\nReason: {reason}")
}

#[async_trait(?Send)]
impl Module for SurpriseModule {
    type Batch = ();

    fn id() -> &'static str {
        "surprise"
    }

    fn peer_context() -> Option<&'static str> {
        Some(
            "Surprise detects when current cognition departs from expectation or recent continuity.",
        )
    }

    fn allocation_hint() -> Option<&'static str> {
        Some(
            "Raise surprise when new cognition may violate expectation, break recent continuity, or deserve preservation because of novelty or significance. Keep it low for routine expected updates or when no expectation is available.",
        )
    }

    async fn next_batch(&mut self) -> Result<Self::Batch> {
        SurpriseModule::next_batch(self).await
    }

    async fn activate(
        &mut self,
        cx: &nuillu_module::ActivateCx<'_>,
        _batch: &Self::Batch,
    ) -> Result<()> {
        SurpriseModule::activate(self, cx).await
    }
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::rc::Rc;
    use std::sync::{Arc, Mutex};

    use lutum::{
        AdapterStructuredTurn, AdapterTextTurn, AgentError, ErasedStructuredTurnEventStream,
        ErasedTextTurnEventStream, FinishReason, InputMessageRole, Lutum, MaxOutputTokens,
        MessageContent, MockLlmAdapter, MockTextScenario, ModelInput, ModelInputItem,
        RawTextTurnEvent, SharedPoolBudgetManager, SharedPoolBudgetOptions, TurnAdapter, Usage,
    };
    use nuillu_blackboard::{
        ActivationRatio, Blackboard, BlackboardCommand, Bpm, CognitionLogEntry, CognitionLogOrigin,
        ModuleConfig, ResourceAllocation, linear_ratio_fn,
    };
    use nuillu_module::ports::{Clock, NoopCognitionLogRepository, SystemClock};
    use nuillu_module::{
        CapabilityProviderPorts, CapabilityProviders, LutumTiers, ModuleRegistry,
        SessionCompactionPolicy, SessionCompactionRuntime,
    };
    use nuillu_types::{ModuleInstanceId, ReplicaIndex, builtin};

    use super::*;

    fn test_blackboard() -> Blackboard {
        let mut allocation = ResourceAllocation::default();
        allocation.set(builtin::surprise(), ModuleConfig::default());
        allocation.set_activation(builtin::surprise(), ActivationRatio::ONE);
        Blackboard::with_allocation(allocation)
    }

    fn test_caps<T>(blackboard: Blackboard, adapter: Arc<T>) -> CapabilityProviders
    where
        T: TurnAdapter,
    {
        let budget = SharedPoolBudgetManager::new(SharedPoolBudgetOptions::default());
        let lutum = Lutum::new(adapter, budget);
        CapabilityProviders::new(CapabilityProviderPorts {
            blackboard,
            cognition_log_port: Rc::new(NoopCognitionLogRepository),
            clock: Rc::new(SystemClock),
            tiers: LutumTiers::from_shared_lutum(lutum),
        })
    }

    #[derive(Clone)]
    struct CapturingAdapter {
        inner: MockLlmAdapter,
        text_inputs: Arc<Mutex<Vec<ModelInput>>>,
        text_turns: Arc<Mutex<Vec<AdapterTextTurn>>>,
    }

    impl CapturingAdapter {
        fn new(inner: MockLlmAdapter) -> Self {
            Self {
                inner,
                text_inputs: Arc::new(Mutex::new(Vec::new())),
                text_turns: Arc::new(Mutex::new(Vec::new())),
            }
        }

        fn text_inputs(&self) -> Vec<ModelInput> {
            self.text_inputs.lock().unwrap().clone()
        }

        fn text_turns(&self) -> Vec<AdapterTextTurn> {
            self.text_turns.lock().unwrap().clone()
        }
    }

    #[async_trait::async_trait]
    impl TurnAdapter for CapturingAdapter {
        async fn text_turn(
            &self,
            input: ModelInput,
            turn: AdapterTextTurn,
        ) -> Result<ErasedTextTurnEventStream, AgentError> {
            self.text_inputs.lock().unwrap().push(input.clone());
            self.text_turns.lock().unwrap().push(turn.clone());
            self.inner.text_turn(input, turn).await
        }

        async fn structured_turn(
            &self,
            input: lutum::ModelInput,
            turn: AdapterStructuredTurn,
        ) -> Result<ErasedStructuredTurnEventStream, AgentError> {
            self.inner.structured_turn(input, turn).await
        }
    }

    fn test_policy() -> nuillu_blackboard::ModulePolicy {
        nuillu_blackboard::ModulePolicy::new(
            nuillu_types::ReplicaCapRange::new(0, 0).unwrap(),
            Bpm::from_f64(60.0)..=Bpm::from_f64(60.0),
            linear_ratio_fn,
        )
    }

    macro_rules! noop_stub {
        ($name:ident, $id:literal) => {
            struct $name;

            #[async_trait::async_trait(?Send)]
            impl Module for $name {
                type Batch = ();

                fn id() -> &'static str {
                    $id
                }

                fn peer_context() -> Option<&'static str> {
                    Some("test stub")
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

    noop_stub!(SurpriseStub, "surprise");

    struct SurpriseFixture {
        module: SurpriseModule,
        blackboard: Blackboard,
        attention_requests: nuillu_module::AttentionControlRequestInbox,
    }

    async fn surprise_fixture<T>(adapter: Arc<T>) -> SurpriseFixture
    where
        T: TurnAdapter,
    {
        let blackboard = test_blackboard();
        let caps = test_caps(blackboard.clone(), adapter);
        let module_cell = Rc::new(RefCell::new(None));
        let attention_requests_cell = Rc::new(RefCell::new(None));
        let module_sink = Rc::clone(&module_cell);
        let attention_requests_sink = Rc::clone(&attention_requests_cell);

        ModuleRegistry::new()
            .register(test_policy(), move |caps| {
                let module_sink = Rc::clone(&module_sink);
                let attention_requests_sink = Rc::clone(&attention_requests_sink);
                async move {
                    *module_sink.borrow_mut() = Some(SurpriseModule::new(
                        caps.cognition_log_updated_inbox(),
                        caps.cognition_log_reader(),
                        caps.blackboard_reader(),
                        caps.attention_control_mailbox(),
                        caps.memo(),
                        caps.llm_access(),
                        caps.session("main")
                            .with_auto_compaction(session_auto_compaction())
                            .await?,
                    ));
                    *attention_requests_sink.borrow_mut() = Some(caps.attention_control_inbox());
                    Ok(SurpriseStub)
                }
            })
            .unwrap()
            .build(&caps)
            .await
            .unwrap();

        SurpriseFixture {
            module: module_cell.borrow_mut().take().unwrap(),
            blackboard,
            attention_requests: attention_requests_cell.borrow_mut().take().unwrap(),
        }
    }

    async fn seed_surprise_inputs(blackboard: &Blackboard) {
        let now = SystemClock.now();
        blackboard
            .apply(BlackboardCommand::UpdateMemo {
                owner: ModuleInstanceId::new(builtin::predict(), ReplicaIndex::ZERO),
                memo: "Forward prediction: Koro will continue calmly eating.".into(),
                written_at: now,
            })
            .await;
        let source = ModuleInstanceId::new(builtin::cognition_gate(), ReplicaIndex::ZERO);
        blackboard
            .apply(BlackboardCommand::AppendCognitionLog {
                source: source.clone(),
                entry: CognitionLogEntry {
                    at: now,
                    text: "Koro suddenly growls toward the doorway.".into(),
                    origin: CognitionLogOrigin::direct(source),
                },
            })
            .await;
    }

    fn text_usage(input_tokens: u64) -> Usage {
        Usage {
            input_tokens,
            ..Usage::zero()
        }
    }

    fn silent_scenario(input_tokens: u64) -> MockTextScenario {
        MockTextScenario::events(vec![
            Ok(RawTextTurnEvent::Started {
                request_id: Some("surprise-silent".into()),
                model: "mock".into(),
            }),
            Ok(RawTextTurnEvent::Completed {
                request_id: Some("surprise-silent".into()),
                finish_reason: FinishReason::Stop,
                usage: text_usage(input_tokens),
            }),
        ])
    }

    fn expected_scenario(args: &MarkExpectedEventArgs, input_tokens: u64) -> MockTextScenario {
        MockTextScenario::events(vec![
            Ok(RawTextTurnEvent::Started {
                request_id: Some("surprise-expected".into()),
                model: "mock".into(),
            }),
            Ok(RawTextTurnEvent::ToolCallChunk {
                id: "call-expected".into(),
                name: "mark_expected_event".into(),
                arguments_json_delta: serde_json::to_string(args)
                    .expect("tool args must serialize"),
            }),
            Ok(RawTextTurnEvent::Completed {
                request_id: Some("surprise-expected".into()),
                finish_reason: FinishReason::ToolCall,
                usage: text_usage(input_tokens),
            }),
        ])
    }

    fn preserve_scenario(
        args: &PreserveUnexpectedEventArgs,
        input_tokens: u64,
    ) -> MockTextScenario {
        MockTextScenario::events(vec![
            Ok(RawTextTurnEvent::Started {
                request_id: Some("surprise-preserve".into()),
                model: "mock".into(),
            }),
            Ok(RawTextTurnEvent::ToolCallChunk {
                id: "call-surprise".into(),
                name: "preserve_unexpected_event".into(),
                arguments_json_delta: serde_json::to_string(args)
                    .expect("tool args must serialize"),
            }),
            Ok(RawTextTurnEvent::Completed {
                request_id: Some("surprise-preserve".into()),
                finish_reason: FinishReason::ToolCall,
                usage: text_usage(input_tokens),
            }),
        ])
    }

    async fn activate_cx(module: &SurpriseModule) -> nuillu_module::ActivateCx<'static> {
        let lutum = module.llm.lutum().await;
        nuillu_module::ActivateCx::new(
            &[],
            &[],
            &[],
            &[],
            SessionCompactionRuntime::new(
                lutum.lutum().clone(),
                nuillu_module::LlmConcurrencyLimiter::new(None),
                nuillu_types::ModelTier::Cheap,
                SessionCompactionPolicy::default(),
            ),
            SystemClock.now(),
        )
    }

    async fn latest_surprise_memo(blackboard: &Blackboard) -> String {
        let owner = ModuleInstanceId::new(builtin::surprise(), ReplicaIndex::ZERO);
        blackboard
            .read(|bb| {
                bb.recent_memo_logs()
                    .into_iter()
                    .filter(|record| record.owner == owner)
                    .map(|record| record.content)
                    .last()
            })
            .await
            .expect("surprise memo should exist")
    }

    #[test]
    fn preserve_tool_renders_preservation_memo() {
        let memo = render_surprise_memo(&PreserveUnexpectedEventArgs {
            content: "Koro snapped up and growled toward the doorway.".into(),
            surprise: 0.9,
            reason: "Violated calm-eating prediction.".into(),
        });
        assert_eq!(
            memo,
            "Surprise assessment: unexpected\nSurprise: 0.90\nMemory preservation request:\nContent: Koro snapped up and growled toward the doorway.\nReason: Violated calm-eating prediction."
        );
    }

    #[test]
    fn expected_tool_renders_expected_memo() {
        assert_eq!(
            render_expected_memo(&MarkExpectedEventArgs {
                reason: "  Fits calm-eating prediction.  ".into(),
            }),
            "Surprise assessment: expected\nNo memory preservation requested.\nReason: Fits calm-eating prediction."
        );
        assert_eq!(
            render_expected_memo(&MarkExpectedEventArgs { reason: " ".into() }),
            EXPECTED_SURPRISE_MEMO
        );
    }

    #[test]
    fn normalize_preserve_args_clamps_surprise_and_rejects_empty_content() {
        assert!(
            normalize_preserve_args(PreserveUnexpectedEventArgs {
                content: "   ".into(),
                surprise: 0.5,
                reason: "x".into(),
            })
            .is_none()
        );
        assert_eq!(
            normalize_preserve_args(PreserveUnexpectedEventArgs {
                content: "event".into(),
                surprise: 1.5,
                reason: "  note  ".into(),
            }),
            Some(PreserveUnexpectedEventArgs {
                content: "event".into(),
                surprise: 1.0,
                reason: "note".into(),
            })
        );
    }

    #[test]
    fn memory_preservation_priority_uses_surprise_threshold() {
        assert_eq!(
            memory_preservation_priority(0.74),
            "normal-priority memory preservation"
        );
        assert_eq!(
            memory_preservation_priority(0.75),
            "high-priority memory preservation"
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn activate_writes_expected_memo_via_tool() {
        let args = MarkExpectedEventArgs {
            reason: "Fits pending prediction.".into(),
        };
        let adapter =
            Arc::new(MockLlmAdapter::new().with_text_scenario(expected_scenario(&args, 1)));
        let mut fixture = surprise_fixture(adapter).await;
        seed_surprise_inputs(&fixture.blackboard).await;

        let cx = activate_cx(&fixture.module).await;
        fixture.module.activate(&cx).await.unwrap();

        assert_eq!(
            latest_surprise_memo(&fixture.blackboard).await,
            "Surprise assessment: expected\nNo memory preservation requested.\nReason: Fits pending prediction."
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn activate_caps_tool_turn_output_tokens() {
        let args = MarkExpectedEventArgs {
            reason: "Fits pending prediction.".into(),
        };
        let capture = CapturingAdapter::new(
            MockLlmAdapter::new().with_text_scenario(expected_scenario(&args, 1)),
        );
        let observed = capture.clone();
        let mut fixture = surprise_fixture(Arc::new(capture)).await;
        seed_surprise_inputs(&fixture.blackboard).await;

        let cx = activate_cx(&fixture.module).await;
        fixture.module.activate(&cx).await.unwrap();

        let turns = observed.text_turns();
        assert_eq!(turns.len(), 1);
        assert_eq!(
            turns[0].config.generation.max_output_tokens,
            Some(MaxOutputTokens::new(TOOL_TURN_MAX_OUTPUT_TOKENS))
        );
        let inputs = observed.text_inputs();
        assert_eq!(inputs.len(), 1);
        assert!(inputs[0].items().iter().any(|item| matches!(
            item,
            ModelInputItem::Message {
                role: InputMessageRole::Developer,
                content,
            } if matches!(
                content.as_slice(),
                [MessageContent::Text(text)] if text == ACTIVATION_INPUT
            )
        )));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn activate_errors_when_no_tool_called() {
        let adapter = Arc::new(MockLlmAdapter::new().with_text_scenario(silent_scenario(1)));
        let mut fixture = surprise_fixture(adapter).await;
        seed_surprise_inputs(&fixture.blackboard).await;

        let cx = activate_cx(&fixture.module).await;
        let err = fixture.module.activate(&cx).await.unwrap_err();

        assert!(
            err.to_string()
                .contains("surprise finished without required tool call")
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn activate_preserves_unexpected_event_via_tool() {
        let args = PreserveUnexpectedEventArgs {
            content: "Koro snapped up and growled toward the doorway.".into(),
            surprise: 0.9,
            reason: "Violated calm-eating prediction.".into(),
        };
        let adapter =
            Arc::new(MockLlmAdapter::new().with_text_scenario(preserve_scenario(&args, 1)));
        let mut fixture = surprise_fixture(adapter).await;
        seed_surprise_inputs(&fixture.blackboard).await;

        let cx = activate_cx(&fixture.module).await;
        fixture.module.activate(&cx).await.unwrap();

        let memo = latest_surprise_memo(&fixture.blackboard).await;
        assert!(memo.contains("Surprise assessment: unexpected"));
        assert!(memo.contains("Memory preservation request:"));
        assert_eq!(
            fixture
                .attention_requests
                .take_ready_items()
                .expect("attention-control inbox should be readable")
                .items
                .len(),
            1
        );
    }
}
