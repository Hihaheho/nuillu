use anyhow::{Context, Result};
use async_trait::async_trait;
use lutum::{Session, TextStepOutcomeWithTools, ToolResult};
use nuillu_module::{
    AllocationReader, AttentionControlRequest, AttentionControlRequestMailbox, BlackboardReader,
    CognitionLogReader, CognitionLogUpdatedInbox, LlmAccess, Memo, Module, SessionCompactionConfig,
    SessionCompactionProtectedPrefix, compact_session_if_needed, format_current_attention_guidance,
    push_formatted_cognition_log_batch, push_formatted_memo_log_batch,
    seed_persistent_faculty_session,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

mod batch;

const SYSTEM_PROMPT: &str = r#"You are the surprise module.
Detect unexpected cognition-log entries. If predict memo log entries are present, frame the
assessment as divergence from pending predictions. If no predict memo log is present, judge novelty against recent
cognition-log history. Do not generate forward predictions.

Predict and other faculty notes arrive as system context. New cognition-log entries are appended
to the session above. Assess each batch using the user instruction for this activation.
If the cognition fits pending predictions or recent history, finish without calling tools.
If the cognition diverges enough that the event should be preserved in memory, call
preserve_unexpected_event exactly once with the assessment and preservation details."#;

const EXPECTED_SURPRISE_MEMO: &str =
    "Surprise assessment: expected\nNo memory preservation requested.";

const ACTIVATION_INPUT: &str = "Assess whether the new cognition is surprising relative to pending predictions and recent history.";

const COMPACTED_SURPRISE_SESSION_PREFIX: &str = "Compacted surprise session history:";
const SESSION_COMPACTION_PROMPT: &str = r#"You compact the surprise module's persistent session history.
Summarize only the prefix transcript you receive. Preserve prior surprise assessments, predict memo
log facts, significant events, memory preservation requests, and cognition-log context needed for
future surprise checks. Do not invent facts. Return plain text only."#;

fn format_surprise_context(allocation: &nuillu_module::ResourceAllocation) -> Option<String> {
    format_current_attention_guidance(allocation).map(|attention| {
        format!("Surprise context for assessing unexpected cognition:\n\n{attention}")
    })
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

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema, lutum::Toolset)]
pub enum SurpriseTools {
    PreserveUnexpectedEvent(PreserveUnexpectedEventArgs),
}

pub struct SurpriseModule {
    updates: CognitionLogUpdatedInbox,
    cognition_log: CognitionLogReader,
    allocation: AllocationReader,
    blackboard: BlackboardReader,
    attention_control: AttentionControlRequestMailbox,
    memo: Memo,
    llm: LlmAccess,
    session: Session,
    session_compaction: SessionCompactionConfig,
    system_prompt: std::sync::OnceLock<String>,
    session_seeded: bool,
}

impl SurpriseModule {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        updates: CognitionLogUpdatedInbox,
        cognition_log: CognitionLogReader,
        allocation: AllocationReader,
        blackboard: BlackboardReader,
        attention_control: AttentionControlRequestMailbox,
        memo: Memo,
        llm: LlmAccess,
    ) -> Self {
        Self {
            updates,
            cognition_log,
            allocation,
            blackboard,
            attention_control,
            memo,
            llm,
            session: Session::new(),
            session_compaction: SessionCompactionConfig::default(),
            system_prompt: std::sync::OnceLock::new(),
            session_seeded: false,
        }
    }

    fn ensure_session_seeded(&mut self, cx: &nuillu_module::ActivateCx<'_>) {
        if self.session_seeded {
            return;
        }
        let system_prompt = self.system_prompt(cx).to_owned();
        seed_persistent_faculty_session(
            &mut self.session,
            system_prompt,
            cx.identity_memories(),
            cx.now(),
        );
        self.session_seeded = true;
    }

    fn system_prompt(&self, cx: &nuillu_module::ActivateCx<'_>) -> &str {
        self.system_prompt.get_or_init(|| {
            nuillu_module::format_identity_system_prompt(
                SYSTEM_PROMPT,
                cx.identity_memories(),
                cx.core_policies(),
                cx.now(),
            )
        })
    }

    #[tracing::instrument(skip_all, err(Debug, level = "warn"))]
    async fn activate(&mut self, cx: &nuillu_module::ActivateCx<'_>) -> Result<()> {
        self.ensure_session_seeded(cx);

        let unread_cognition = self.cognition_log.unread_events().await;
        let unread_memo_logs = self.blackboard.unread_memo_logs().await;
        push_formatted_memo_log_batch(&mut self.session, &unread_memo_logs, cx.now());
        push_formatted_cognition_log_batch(&mut self.session, &unread_cognition, cx.now());
        let allocation = self.allocation.snapshot().await;

        if let Some(context) = format_surprise_context(&allocation) {
            self.session.push_ephemeral_system(context);
        }
        self.session.push_ephemeral_user(ACTIVATION_INPUT);

        let lutum = self.llm.lutum().await;
        let outcome = self
            .session
            .text_turn(&lutum)
            .tools::<SurpriseTools>()
            .available_tools([SurpriseToolsSelector::PreserveUnexpectedEvent])
            .collect()
            .await
            .context("surprise assessment turn failed")?;

        let mut memo: Option<String> = None;
        let input_tokens = match outcome {
            TextStepOutcomeWithTools::Finished(result) => result.usage.input_tokens,
            TextStepOutcomeWithTools::FinishedNoOutput(result) => result.usage.input_tokens,
            TextStepOutcomeWithTools::NeedsTools(round) => {
                let input_tokens = round.usage.input_tokens;
                let mut results: Vec<ToolResult> = Vec::new();
                // The LLM may return multiple tool calls; adopt the first memo only.
                for call in round.tool_calls.iter().cloned() {
                    let SurpriseToolsCall::PreserveUnexpectedEvent(call) = call;
                    if memo.is_none() {
                        if let Some(args) = normalize_preserve_args(call.input.clone()) {
                            memo = Some(render_surprise_memo(&args));
                        }
                    }
                    let output = self.preserve_unexpected_event(call.input.clone()).await;
                    results.push(
                        call.complete(output)
                            .context("complete preserve_unexpected_event tool call")?,
                    );
                }
                round
                    .commit(&mut self.session, results)
                    .context("commit surprise tool round")?;
                input_tokens
            }
        };
        compact_session_if_needed(
            &mut self.session,
            input_tokens,
            cx.session_compaction(),
            self.session_compaction,
            SessionCompactionProtectedPrefix::LeadingSystemAndIdentitySeed,
            Self::id(),
            COMPACTED_SURPRISE_SESSION_PREFIX,
            SESSION_COMPACTION_PROMPT,
        )
        .await;

        self.memo
            .write(memo.unwrap_or_else(|| EXPECTED_SURPRISE_MEMO.to_owned()))
            .await;
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
    use std::sync::Arc;

    use lutum::{
        FinishReason, Lutum, MockLlmAdapter, MockTextScenario, RawTextTurnEvent,
        SharedPoolBudgetManager, SharedPoolBudgetOptions, Usage,
    };
    use nuillu_blackboard::{
        ActivationRatio, Blackboard, BlackboardCommand, Bpm, CognitionLogEntry, ModuleConfig,
        ResourceAllocation, linear_ratio_fn,
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

    fn test_caps(blackboard: Blackboard, adapter: Arc<MockLlmAdapter>) -> CapabilityProviders {
        let budget = SharedPoolBudgetManager::new(SharedPoolBudgetOptions::default());
        let lutum = Lutum::new(adapter, budget);
        CapabilityProviders::new(CapabilityProviderPorts {
            blackboard,
            cognition_log_port: Rc::new(NoopCognitionLogRepository),
            clock: Rc::new(SystemClock),
            tiers: LutumTiers::from_shared_lutum(lutum),
        })
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

    async fn surprise_fixture(adapter: Arc<MockLlmAdapter>) -> SurpriseFixture {
        let blackboard = test_blackboard();
        let caps = test_caps(blackboard.clone(), adapter);
        let module_cell = Rc::new(RefCell::new(None));
        let attention_requests_cell = Rc::new(RefCell::new(None));
        let module_sink = Rc::clone(&module_cell);
        let attention_requests_sink = Rc::clone(&attention_requests_cell);

        ModuleRegistry::new()
            .register(test_policy(), move |caps| {
                *module_sink.borrow_mut() = Some(SurpriseModule::new(
                    caps.cognition_log_updated_inbox(),
                    caps.cognition_log_reader(),
                    caps.allocation_reader(),
                    caps.blackboard_reader(),
                    caps.attention_control_mailbox(),
                    caps.memo(),
                    caps.llm_access(),
                ));
                *attention_requests_sink.borrow_mut() = Some(caps.attention_control_inbox());
                SurpriseStub
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
        blackboard
            .apply(BlackboardCommand::AppendCognitionLog {
                source: ModuleInstanceId::new(builtin::cognition_gate(), ReplicaIndex::ZERO),
                entry: CognitionLogEntry {
                    at: now,
                    text: "Koro suddenly growls toward the doorway.".into(),
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
    async fn activate_writes_expected_memo_when_no_tool_called() {
        let adapter = Arc::new(MockLlmAdapter::new().with_text_scenario(silent_scenario(1)));
        let mut fixture = surprise_fixture(adapter).await;
        seed_surprise_inputs(&fixture.blackboard).await;

        let cx = activate_cx(&fixture.module).await;
        fixture.module.activate(&cx).await.unwrap();

        assert_eq!(
            latest_surprise_memo(&fixture.blackboard).await,
            EXPECTED_SURPRISE_MEMO
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
