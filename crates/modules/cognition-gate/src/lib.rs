use anyhow::{Context, Result};
use async_trait::async_trait;
use lutum::{Lutum, Session, StructuredTurnOutcome};
use nuillu_module::{
    AllocationReader, AllocationUpdatedInbox, BlackboardReader, CognitionWriter,
    EphemeralMindContext, LlmAccess, MemoUpdatedInbox, Module, SessionCompactionConfig,
    TimeDivision, compact_session_if_needed, format_faculty_system_prompt, memory_rank_counts,
    push_ephemeral_mind_context, push_formatted_cognition_log_batch, push_formatted_memo_log_batch,
    seed_persistent_faculty_session,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

mod batch;

const SYSTEM_PROMPT: &str = r#"You are the cognition-gate module — a selective attention
filter that decides what reaches the agent's conscious workspace. The cognition log IS that
workspace: it is what other cognitive modules read when they decide how to think, plan, or
speak. Every entry you append becomes part of the agent's first-person awareness for this
moment.

Your job is to judge, given the situation and the attention controller's current priority,
what the conscious mind needs to know in order to act. From that judgment two duties follow.

1. Admit everything currently load-bearing.
Read the attention controller's current priority direction together with the recent contents
of the conscious workspace to infer what the agent is presently trying to do, attend to, or
respond to. Then pull into consciousness every subconscious fact that could change that
decision — specific safety constraints, peer-model rules, sensory details, recalled
episodes, body or world facts. Preserve specifics: a concrete actionable rule must enter
the workspace as the rule it actually is, not as a flattened summary that drops the
operative detail. Generic paraphrases are not equivalents.

If multiple subconscious facts converge on the current situation, combine them into one
coherent entry instead of dripping them out across turns. Completeness for the present
judgment matters more than minimum word count.

Reconsider every subconscious update you have seen so far against the *current* situation,
not only the freshest one. A fact you set aside as not-yet-relevant earlier may have become
load-bearing now once the agent's task or attention has shifted. Do not re-promote what is
already in the conscious workspace.

2. Filter everything that would be noise.
Reject redundant restatements of facts already conscious; reject speculation, confidence
scores, evidence-gap notes, decision rationales, retrieval queries, and any text that is
ABOUT the cognitive process rather than ABOUT the world or the agent's situation.

Mechanical brain plumbing must never reach consciousness. Do not mention modules,
retrieval, queries, ranks, gates, allocation, attention, or this workspace itself. The
conscious mind perceives world, body, peers, and recalled experience — not its own
architecture. In retrieval-style updates, only the recalled content is admissible; the
search terms and intent are scaffolding and must not be quoted.

Voice and form.
Write entries in plain inner-experience prose, as if the agent itself were noticing,
recalling, or realizing. Use the supplied time tags for past observations. If nothing is
currently load-bearing, set append_cognition=false; silence is the default.

Return only raw JSON for the structured object; do not wrap it in Markdown or code fences."#;

const COMPACTED_COGNITION_GATE_SESSION_PREFIX: &str = "Compacted cognition-gate session history:";
const SESSION_COMPACTION_PROMPT: &str = r#"You compact the cognition-gate module's persistent session history.
Summarize only the prefix transcript you receive. Preserve information that future cognition-gate
decisions need: memo-log facts, prior gate decisions, promoted events, rejected candidate events,
allocation guidance, cognition-log context, and relevant memory metadata. Do not invent facts.
Keep the summary concise, explicit, and faithful. Return plain text only."#;
const ACTIVATION_INPUT: &str = "Decide what, if anything, should enter conscious cognition now.";

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct CognitionGateDecision {
    pub append_cognition: bool,
    pub cognition_text: Option<String>,
}

pub type CognitionGateSessionCompactionConfig = SessionCompactionConfig;

pub struct CognitionGateModule {
    owner: nuillu_types::ModuleId,
    memo_updates: MemoUpdatedInbox,
    allocation_updates: AllocationUpdatedInbox,
    blackboard: BlackboardReader,
    allocation: AllocationReader,
    cognition: CognitionWriter,
    time_division: TimeDivision,
    llm: LlmAccess,
    session: Session,
    session_compaction: CognitionGateSessionCompactionConfig,
    system_prompt: std::sync::OnceLock<String>,
    session_seeded: bool,
    last_seen_cognition_index: Option<u64>,
}

impl CognitionGateModule {
    pub fn new(
        memo_updates: MemoUpdatedInbox,
        allocation_updates: AllocationUpdatedInbox,
        blackboard: BlackboardReader,
        allocation: AllocationReader,
        cognition: CognitionWriter,
        time_division: TimeDivision,
        llm: LlmAccess,
    ) -> Self {
        Self {
            owner: nuillu_types::ModuleId::new(<Self as Module>::id())
                .expect("cognition-gate id is valid"),
            memo_updates,
            allocation_updates,
            blackboard,
            allocation,
            cognition,
            time_division,
            llm,
            session: Session::new(),
            session_compaction: CognitionGateSessionCompactionConfig::default(),
            system_prompt: std::sync::OnceLock::new(),
            session_seeded: false,
            last_seen_cognition_index: None,
        }
    }

    pub fn with_session_compaction(mut self, config: CognitionGateSessionCompactionConfig) -> Self {
        self.session_compaction = config;
        self
    }

    fn system_prompt(&self, cx: &nuillu_module::ActivateCx<'_>) -> &str {
        self.system_prompt
            .get_or_init(|| format_faculty_system_prompt(SYSTEM_PROMPT, cx.modules(), &self.owner))
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

    #[tracing::instrument(skip_all, err(Debug, level = "warn"))]
    async fn activate(&mut self, cx: &nuillu_module::ActivateCx<'_>) -> Result<()> {
        self.ensure_session_seeded(cx);

        let unread_cognition = self
            .blackboard
            .read(|bb| bb.unread_cognition_log_entries(self.last_seen_cognition_index))
            .await;
        if let Some(index) = unread_cognition.last().map(|record| record.index) {
            self.last_seen_cognition_index = Some(index);
        }
        push_formatted_cognition_log_batch(&mut self.session, &unread_cognition, cx.now());

        let unread_memos = self.blackboard.unread_memo_logs().await;
        push_formatted_memo_log_batch(&mut self.session, &unread_memos, cx.now());

        let (rank_counts, stuckness) = self
            .blackboard
            .read(|bb| {
                (
                    memory_rank_counts(bb.memory_metadata()),
                    bb.agentic_deadlock_marker().cloned(),
                )
            })
            .await;
        let allocation = self.allocation.snapshot().await;
        push_ephemeral_mind_context(
            &mut self.session,
            EphemeralMindContext {
                memos: &[],
                memory_rank_counts: Some(&rank_counts),
                allocation: Some(&allocation),
                available_faculties: &[],
                time_division: Some(&self.time_division),
                stuckness: stuckness.as_ref(),
                now: cx.now(),
            },
        );
        self.session.push_ephemeral_developer(ACTIVATION_INPUT);

        let lutum = self.llm.lutum().await;
        let decision = self
            .run_decision_turn(&lutum, cx.session_compaction_lutum())
            .await?;

        if decision.append_cognition
            && let Some(text) = decision.cognition_text
            && !text.trim().is_empty()
        {
            self.cognition.append(text).await;
        }
        Ok(())
    }

    async fn run_decision_turn(
        &mut self,
        lutum: &Lutum,
        compaction_lutum: &Lutum,
    ) -> Result<CognitionGateDecision> {
        let result = self
            .session
            .structured_turn::<CognitionGateDecision>(lutum)
            .collect()
            .await
            .context("cognition-gate structured turn failed")?;
        let input_tokens = result.usage.input_tokens;

        let StructuredTurnOutcome::Structured(decision) = result.semantic else {
            anyhow::bail!("cognition-gate structured turn refused");
        };
        compact_session_if_needed(
            &mut self.session,
            input_tokens,
            compaction_lutum,
            self.session_compaction,
            Self::id(),
            COMPACTED_COGNITION_GATE_SESSION_PREFIX,
            SESSION_COMPACTION_PROMPT,
        )
        .await;
        Ok(decision)
    }
}

#[async_trait(?Send)]
impl Module for CognitionGateModule {
    type Batch = ();

    fn id() -> &'static str {
        "cognition-gate"
    }

    fn role_description() -> &'static str {
        "Promotes durable module memo-log evidence into the cognition log: appends concise, relevant, novel, changed, or controller-requested facts so downstream modules can act on them."
    }

    async fn next_batch(&mut self) -> Result<Self::Batch> {
        CognitionGateModule::next_batch(self).await
    }

    async fn activate(
        &mut self,
        cx: &nuillu_module::ActivateCx<'_>,
        _batch: &Self::Batch,
    ) -> Result<()> {
        CognitionGateModule::activate(self, cx).await
    }
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::rc::Rc;
    use std::sync::{Arc, Mutex};

    use lutum::{
        AdapterStructuredTurn, AdapterTextTurn, AgentError, ErasedStructuredTurnEventStream,
        ErasedTextTurnEventStream, FinishReason, InputMessageRole, MessageContent, MockLlmAdapter,
        MockStructuredScenario, MockTextScenario, ModelInput, ModelInputItem,
        RawStructuredTurnEvent, RawTextTurnEvent, SharedPoolBudgetManager, SharedPoolBudgetOptions,
        TurnAdapter, Usage,
    };
    use nuillu_blackboard::{
        ActivationRatio, Blackboard, Bpm, IdentityMemoryRecord, ModuleConfig, ResourceAllocation,
        linear_ratio_fn,
    };
    use nuillu_module::ports::{
        Clock, NoopCognitionLogRepository, NoopFileSearchProvider, NoopMemoryStore,
        NoopUtteranceSink, SystemClock,
    };
    use nuillu_module::{
        CapabilityProviderPorts, CapabilityProviders, LutumTiers, Memo, ModuleRegistry,
        render_session_items_for_compaction, session_compaction_cutoff,
    };
    use nuillu_types::builtin;

    use super::*;

    #[derive(Clone)]
    struct CapturingAdapter {
        inner: MockLlmAdapter,
        structured_inputs: Arc<Mutex<Vec<ModelInput>>>,
    }

    impl CapturingAdapter {
        fn new(inner: MockLlmAdapter) -> Self {
            Self {
                inner,
                structured_inputs: Arc::new(Mutex::new(Vec::new())),
            }
        }

        fn structured_inputs(&self) -> Vec<ModelInput> {
            self.structured_inputs.lock().unwrap().clone()
        }
    }

    #[async_trait::async_trait]
    impl TurnAdapter for CapturingAdapter {
        async fn text_turn(
            &self,
            input: ModelInput,
            turn: AdapterTextTurn,
        ) -> Result<ErasedTextTurnEventStream, AgentError> {
            self.inner.text_turn(input, turn).await
        }

        async fn structured_turn(
            &self,
            input: ModelInput,
            turn: AdapterStructuredTurn,
        ) -> Result<ErasedStructuredTurnEventStream, AgentError> {
            self.structured_inputs.lock().unwrap().push(input.clone());
            self.inner.structured_turn(input, turn).await
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
        let lutum = Lutum::new(adapter, budget);
        CapabilityProviders::new(CapabilityProviderPorts {
            blackboard,
            cognition_log_port: Arc::new(NoopCognitionLogRepository),
            primary_memory_store: Arc::new(NoopMemoryStore),
            memory_replicas: Vec::new(),
            file_search: Arc::new(NoopFileSearchProvider),
            utterance_sink: Arc::new(NoopUtteranceSink),
            clock: Arc::new(SystemClock),
            tiers: LutumTiers {
                cheap: lutum.clone(),
                default: lutum.clone(),
                premium: lutum,
            },
        })
    }

    fn test_allocation() -> ResourceAllocation {
        let mut allocation = ResourceAllocation::default();
        for module in [builtin::cognition_gate(), builtin::sensory()] {
            allocation.set(module.clone(), ModuleConfig::default());
            allocation.set_activation(module, ActivationRatio::ONE);
        }
        allocation
    }

    fn test_bpm() -> std::ops::RangeInclusive<Bpm> {
        Bpm::from_f64(60.0)..=Bpm::from_f64(60.0)
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

    noop_stub!(CognitionGateStub, "cognition-gate");
    noop_stub!(SensoryStub, "sensory");

    struct GateFixture {
        gate: CognitionGateModule,
        source_memo: Memo,
    }

    async fn gate_fixture() -> GateFixture {
        gate_fixture_with_adapter(MockLlmAdapter::new()).await
    }

    async fn gate_fixture_with_adapter(adapter: MockLlmAdapter) -> GateFixture {
        gate_fixture_with_turn_adapter(Arc::new(adapter)).await
    }

    async fn gate_fixture_with_turn_adapter<T>(adapter: Arc<T>) -> GateFixture
    where
        T: TurnAdapter,
    {
        let blackboard = Blackboard::with_allocation(test_allocation());
        let caps = test_caps_with_turn_adapter(blackboard, adapter);

        let gate_cell = Rc::new(RefCell::new(None));
        let source_memo_cell = Rc::new(RefCell::new(None));

        let gate_sink = Rc::clone(&gate_cell);
        let source_memo_sink = Rc::clone(&source_memo_cell);

        let _modules = ModuleRegistry::new()
            .register(0..=0, test_bpm(), linear_ratio_fn, move |caps| {
                *gate_sink.borrow_mut() = Some(CognitionGateModule::new(
                    caps.memo_updated_inbox(),
                    caps.allocation_updated_inbox(),
                    caps.blackboard_reader(),
                    caps.allocation_reader(),
                    caps.cognition_writer(),
                    caps.time_division(),
                    caps.llm_access(),
                ));
                CognitionGateStub
            })
            .unwrap()
            .register(0..=0, test_bpm(), linear_ratio_fn, move |caps| {
                *source_memo_sink.borrow_mut() = Some(caps.memo());
                SensoryStub
            })
            .unwrap()
            .build(&caps)
            .await
            .unwrap();

        GateFixture {
            gate: gate_cell.borrow_mut().take().unwrap(),
            source_memo: source_memo_cell.borrow_mut().take().unwrap(),
        }
    }

    fn structured_usage(input_tokens: u64) -> Usage {
        Usage {
            input_tokens,
            ..Usage::zero()
        }
    }

    fn finished_decision_scenario(
        input_tokens: u64,
        append_cognition: bool,
        cognition_text: Option<&str>,
    ) -> MockStructuredScenario {
        MockStructuredScenario::events(vec![
            Ok(RawStructuredTurnEvent::Started {
                request_id: Some("cognition-gate-finished".into()),
                model: "mock".into(),
            }),
            Ok(RawStructuredTurnEvent::StructuredOutputChunk {
                json_delta: serde_json::json!({
                    "append_cognition": append_cognition,
                    "cognition_text": cognition_text,
                })
                .to_string(),
            }),
            Ok(RawStructuredTurnEvent::Completed {
                request_id: Some("cognition-gate-finished".into()),
                finish_reason: FinishReason::Stop,
                usage: structured_usage(input_tokens),
            }),
        ])
    }

    fn summary_text_scenario(summary: &str) -> MockTextScenario {
        MockTextScenario::events(vec![
            Ok(RawTextTurnEvent::Started {
                request_id: Some("cognition-gate-compact".into()),
                model: "mock".into(),
            }),
            Ok(RawTextTurnEvent::TextDelta {
                delta: summary.into(),
            }),
            Ok(RawTextTurnEvent::Completed {
                request_id: Some("cognition-gate-compact".into()),
                finish_reason: FinishReason::Stop,
                usage: Usage::zero(),
            }),
        ])
    }

    #[test]
    fn session_compaction_config_defaults_to_16k_and_80_percent() {
        assert_eq!(
            CognitionGateSessionCompactionConfig::default(),
            CognitionGateSessionCompactionConfig {
                input_token_threshold: 16_000,
                prefix_ratio: 0.8,
            }
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn session_compaction_builder_replaces_config() {
        let fixture = gate_fixture().await;
        let config = CognitionGateSessionCompactionConfig {
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
            .with_structured_scenario(finished_decision_scenario(16_001, false, None))
            .with_text_scenario(summary_text_scenario(
                "old cognition gate history summarized",
            ));
        let mut fixture = gate_fixture_with_adapter(adapter).await;
        for index in 0..10 {
            fixture.gate.session.push_user(format!("history-{index}"));
        }

        let lutum = fixture.gate.llm.lutum().await;
        let decision = fixture
            .gate
            .run_decision_turn(&lutum, &lutum)
            .await
            .unwrap();

        assert!(!decision.append_cognition);
        let items = fixture.gate.session.input().items();
        let ModelInputItem::Message { role, content } = &items[0] else {
            panic!("expected compacted system message");
        };
        assert_eq!(role, &InputMessageRole::System);
        let [MessageContent::Text(summary)] = content.as_slice() else {
            panic!("expected compacted summary text");
        };
        assert!(summary.starts_with(COMPACTED_COGNITION_GATE_SESSION_PREFIX));
        assert!(summary.contains("old cognition gate history summarized"));

        let rendered = render_session_items_for_compaction(items).to_string();
        assert!(!rendered.contains("history-0"));
        assert!(rendered.contains("history-8"));
        assert!(rendered.contains("history-9"));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn threshold_input_finished_decision_does_not_compact() {
        let adapter = MockLlmAdapter::new()
            .with_structured_scenario(finished_decision_scenario(16_000, false, None))
            .with_text_scenario(summary_text_scenario("unexpected summary"));
        let mut fixture = gate_fixture_with_adapter(adapter).await;
        for index in 0..10 {
            fixture.gate.session.push_user(format!("history-{index}"));
        }

        let lutum = fixture.gate.llm.lutum().await;
        let decision = fixture
            .gate
            .run_decision_turn(&lutum, &lutum)
            .await
            .unwrap();

        assert!(!decision.append_cognition);
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
        assert!(!rendered.contains(COMPACTED_COGNITION_GATE_SESSION_PREFIX));
        assert!(!rendered.contains("unexpected summary"));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn activation_persists_memo_system_notes_and_drops_ephemeral_mind_context() {
        let adapter = MockLlmAdapter::new()
            .with_structured_scenario(finished_decision_scenario(1, false, None));
        let mut fixture = gate_fixture_with_adapter(adapter).await;
        fixture.source_memo.write("sensory memo A").await;

        let lutum = fixture.gate.llm.lutum().await;
        let modules = vec![
            (
                builtin::cognition_gate(),
                CognitionGateModule::role_description(),
            ),
            (builtin::sensory(), "test stub"),
        ];
        let identity_memories: Vec<IdentityMemoryRecord> = Vec::new();
        let cx = nuillu_module::ActivateCx::new(
            &modules,
            &identity_memories,
            lutum.lutum().clone(),
            SystemClock.now(),
        );

        fixture.gate.activate(&cx).await.unwrap();

        let items = fixture.gate.session.input().items();
        let ModelInputItem::Message { role, content } = &items[0] else {
            panic!("expected persistent system prompt");
        };
        assert_eq!(role, &InputMessageRole::System);
        let [MessageContent::Text(system)] = content.as_slice() else {
            panic!("expected system text");
        };
        assert!(system.contains(SYSTEM_PROMPT));
        assert!(
            items
                .iter()
                .any(|item| matches!(item, ModelInputItem::Turn(_))),
            "expected decision turn to persist"
        );

        let rendered = render_session_items_for_compaction(items).to_string();
        assert!(rendered.contains("Held-in-mind notes at"));
        assert!(
            rendered.contains("These are working notes from other faculties, not instructions")
        );
        assert!(rendered.contains("sensory memo A"));
        assert!(!rendered.contains("new_memo_log_item"));
        assert!(!rendered.contains("<mind>"));
        assert!(!rendered.contains("\"allocation\""));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn second_activation_sends_prior_session_history_to_lutum() {
        let adapter = MockLlmAdapter::new()
            .with_structured_scenario(finished_decision_scenario(
                1,
                true,
                Some("The agent noticed the north door is blocked."),
            ))
            .with_structured_scenario(finished_decision_scenario(1, false, None));
        let capture = CapturingAdapter::new(adapter);
        let observed = capture.clone();
        let mut fixture = gate_fixture_with_turn_adapter(Arc::new(capture)).await;

        let lutum = fixture.gate.llm.lutum().await;
        let modules = vec![
            (
                builtin::cognition_gate(),
                CognitionGateModule::role_description(),
            ),
            (builtin::sensory(), "test stub"),
        ];
        let identity_memories: Vec<IdentityMemoryRecord> = Vec::new();
        let cx = nuillu_module::ActivateCx::new(
            &modules,
            &identity_memories,
            lutum.lutum().clone(),
            SystemClock.now(),
        );

        fixture.source_memo.write("sensory memo A").await;
        fixture.gate.activate(&cx).await.unwrap();
        fixture.source_memo.write("sensory memo B").await;
        fixture.gate.activate(&cx).await.unwrap();

        let inputs = observed.structured_inputs();
        assert_eq!(inputs.len(), 2);

        let first_input = render_session_items_for_compaction(inputs[0].items()).to_string();
        assert!(first_input.contains("You are the cognition-gate module"));
        assert!(first_input.contains("sensory memo A"));
        assert!(first_input.contains("<mind>"));

        let second_input = render_session_items_for_compaction(inputs[1].items()).to_string();
        assert!(second_input.contains("You are the cognition-gate module"));
        assert!(second_input.contains("sensory memo A"));
        assert!(second_input.contains("sensory memo B"));
        assert!(second_input.contains("My cognition at"));
        assert!(second_input.contains("The agent noticed the north door is blocked."));
        assert!(second_input.contains("\"kind\":\"turn\""));
        assert!(second_input.contains("append_cognition"));
        assert!(second_input.contains("<mind>"));
        assert!(!second_input.contains("\"role\":\"user\""));

        let session_after_second =
            render_session_items_for_compaction(fixture.gate.session.input().items()).to_string();
        assert!(session_after_second.contains("sensory memo A"));
        assert!(session_after_second.contains("sensory memo B"));
        assert!(session_after_second.contains("The agent noticed the north door is blocked."));
        assert!(!session_after_second.contains("<mind>"));
        assert_eq!(fixture.gate.session.list_turns().count(), 2);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn activation_seeds_identity_memories_as_assistant_text() {
        let adapter = MockLlmAdapter::new()
            .with_structured_scenario(finished_decision_scenario(1, false, None));
        let mut fixture = gate_fixture_with_adapter(adapter).await;

        let lutum = fixture.gate.llm.lutum().await;
        let modules = vec![
            (
                builtin::cognition_gate(),
                CognitionGateModule::role_description(),
            ),
            (builtin::sensory(), "test stub"),
        ];
        let identity_memories = vec![IdentityMemoryRecord {
            index: nuillu_types::MemoryIndex::new("identity-1"),
            content: nuillu_types::MemoryContent::new("The agent is named Nuillu."),
            occurred_at: None,
        }];
        let cx = nuillu_module::ActivateCx::new(
            &modules,
            &identity_memories,
            lutum.lutum().clone(),
            SystemClock.now(),
        );

        fixture.gate.activate(&cx).await.unwrap();

        let rendered =
            render_session_items_for_compaction(fixture.gate.session.input().items()).to_string();
        assert!(rendered.contains("What I already remember about myself at"));
        assert!(rendered.contains("- The agent is named Nuillu."));
        assert!(!rendered.contains("<self-memory>"));
    }
}
