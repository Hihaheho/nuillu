use anyhow::{Context, Result};
use async_trait::async_trait;
use lutum::{
    AssistantInputItem, InputMessageRole, Lutum, MessageContent, ModelInput, ModelInputItem,
    RawJson, Session, StructuredTurnOutcome, TurnRole,
};
use nuillu_module::{
    AllocationReader, AllocationUpdatedInbox, BlackboardReader, CognitionWriter, LlmAccess,
    MemoUpdatedInbox, Module, TimeDivision,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

mod batch;

const SYSTEM_PROMPT: &str = r#"You are the cognition-gate module.
Read new_memo_log_item messages in the persistent session plus the current blackboard and
allocation guidance, then decide whether anything should enter the cognition log. Append only
concise, novel, currently relevant events needed by cognitive work.
Treat new_memo_log_item messages as newly written module output. Treat recent_memo_logs as older context.
Do not promote a fact solely because it appears in recent_memo_logs if it has already been promoted
or is not directly relevant now.
When promoting sensory memo content, convert detailed observation ages to one of the provided
time-division tags before writing cognition-log text.
If allocation guidance asks for speech evidence promotion and a query, self-model, sensory, or
other memo contains the requested fact, promote that fact into the cognition log in plain speech-ready form.
Include the retrieved fact and the immediate cognition-log question or peer situation. Do not promote
generic advice, speculation, hidden module mechanics, or facts not present in memos.
Return only raw JSON for the structured object;
do not wrap it in Markdown or code fences."#;

const DEFAULT_SESSION_COMPACTION_INPUT_TOKEN_THRESHOLD: u64 = 16_000;
const DEFAULT_SESSION_COMPACTION_PREFIX_RATIO: f64 = 0.8;
const COMPACTED_COGNITION_GATE_SESSION_PREFIX: &str = "Compacted cognition-gate session history:";
const SESSION_COMPACTION_PROMPT: &str = r#"You compact the cognition-gate module's persistent session history.
Summarize only the prefix transcript you receive. Preserve information that future cognition-gate
decisions need: memo-log facts, prior gate decisions, promoted events, rejected candidate events,
allocation guidance, cognition-log context, and relevant memory metadata. Do not invent facts.
Keep the summary concise, explicit, and faithful. Return plain text only."#;

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct CognitionGateDecision {
    pub append_cognition: bool,
    pub cognition_text: Option<String>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CognitionGateSessionCompactionConfig {
    pub input_token_threshold: u64,
    pub prefix_ratio: f64,
}

impl Default for CognitionGateSessionCompactionConfig {
    fn default() -> Self {
        Self {
            input_token_threshold: DEFAULT_SESSION_COMPACTION_INPUT_TOKEN_THRESHOLD,
            prefix_ratio: DEFAULT_SESSION_COMPACTION_PREFIX_RATIO,
        }
    }
}

fn memo_log_history_input(record: &impl Serialize) -> serde_json::Value {
    serde_json::json!({
        "new_memo_log_item": record,
    })
}

fn push_memo_log_history<T: Serialize>(session: &mut Session, records: &[T]) {
    for record in records {
        session.push_user(memo_log_history_input(record).to_string());
    }
}

fn gate_ephemeral_input(
    blackboard_json: serde_json::Value,
    allocation_json: impl Serialize,
) -> serde_json::Value {
    serde_json::json!({
        "blackboard": blackboard_json,
        "allocation": allocation_json,
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
        }
    }

    pub fn with_session_compaction(mut self, config: CognitionGateSessionCompactionConfig) -> Self {
        self.session_compaction = config;
        self
    }

    fn system_prompt(&self, cx: &nuillu_module::ActivateCx<'_>) -> &str {
        self.system_prompt.get_or_init(|| {
            nuillu_module::format_system_prompt(
                SYSTEM_PROMPT,
                cx.modules(),
                &self.owner,
                cx.identity_memories(),
            )
        })
    }

    #[tracing::instrument(skip_all, err(Debug, level = "warn"))]
    async fn activate(&mut self, cx: &nuillu_module::ActivateCx<'_>) -> Result<()> {
        let unread_memo_logs = self.blackboard.unread_memo_logs().await;
        push_memo_log_history(&mut self.session, &unread_memo_logs);
        let context = self
            .blackboard
            .read(|bb| {
                serde_json::json!({
                    "cognition_log": bb.cognition_log().entries(),
                    "memory_metadata": bb.memory_metadata(),
                    "time_division": self.time_division.as_prompt_json(),
                })
            })
            .await;
        let allocation = self.allocation.snapshot().await;

        let system_prompt = self.system_prompt(cx).to_owned();
        self.session.push_ephemeral_system(system_prompt);
        self.session
            .push_ephemeral_user(gate_ephemeral_input(context, allocation).to_string());

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
        self.compact_if_needed(input_tokens, compaction_lutum).await;
        Ok(decision)
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
                "cognition-gate session compaction failed"
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
                .context("render cognition-gate session compaction transcript")?;

        let mut summary_session = Session::new();
        summary_session.push_system(SESSION_COMPACTION_PROMPT);
        summary_session.push_user(transcript);
        let summary = summary_session
            .text_turn(lutum)
            .collect()
            .await
            .context("summarize cognition-gate session prefix")?
            .assistant_text();
        let summary = summary.trim();
        if summary.is_empty() {
            tracing::warn!("cognition-gate session compaction produced an empty summary");
            return Ok(());
        }

        let mut compacted_items = Vec::with_capacity(suffix.len().saturating_add(1));
        compacted_items.push(ModelInputItem::text(
            InputMessageRole::System,
            format!("{COMPACTED_COGNITION_GATE_SESSION_PREFIX}\n{summary}"),
        ));
        compacted_items.extend(suffix);
        self.session = Session::from_input(ModelInput::from_items(compacted_items));
        Ok(())
    }
}

#[async_trait(?Send)]
impl Module for CognitionGateModule {
    type Batch = ();

    fn id() -> &'static str {
        "cognition-gate"
    }

    fn role_description() -> &'static str {
        "Filters blackboard memos into the cognition log: appends cognitively relevant, novel, changed, or controller-requested events when promotion is warranted."
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
    use std::sync::Arc;

    use lutum::{
        FinishReason, MockLlmAdapter, MockStructuredScenario, MockTextScenario,
        RawStructuredTurnEvent, RawTextTurnEvent, SharedPoolBudgetManager, SharedPoolBudgetOptions,
        Usage,
    };
    use nuillu_blackboard::{
        ActivationRatio, Blackboard, Bpm, IdentityMemoryRecord, ModuleConfig, ResourceAllocation,
        linear_ratio_fn,
    };
    use nuillu_module::ports::{
        NoopCognitionLogRepository, NoopFileSearchProvider, NoopMemoryStore, NoopUtteranceSink,
        SystemClock,
    };
    use nuillu_module::{
        CapabilityProviderPorts, CapabilityProviders, LutumTiers, Memo, ModuleRegistry,
    };
    use nuillu_types::builtin;

    use super::*;

    fn test_caps_with_adapter(
        blackboard: Blackboard,
        adapter: MockLlmAdapter,
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
        let blackboard = Blackboard::with_allocation(test_allocation());
        let caps = test_caps_with_adapter(blackboard, adapter);

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
    async fn activation_keeps_memo_history_but_drops_ephemeral_context() {
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
        let cx = nuillu_module::ActivateCx::new(&modules, &identity_memories, &lutum);

        fixture.gate.activate(&cx).await.unwrap();

        let items = fixture.gate.session.input().items();
        assert!(
            items.iter().any(|item| {
                let ModelInputItem::Message {
                    role: InputMessageRole::User,
                    content,
                } = item
                else {
                    return false;
                };
                content.as_slice().iter().any(|content| {
                    let MessageContent::Text(text) = content;
                    text.contains("new_memo_log_item") && text.contains("sensory memo A")
                })
            }),
            "expected memo-log history to persist"
        );
        assert!(
            items
                .iter()
                .any(|item| matches!(item, ModelInputItem::Turn(_))),
            "expected decision turn to persist"
        );

        let rendered = render_session_items_for_compaction(items).to_string();
        assert!(!rendered.contains("recent_memo_logs"));
        assert!(!rendered.contains("\"allocation\""));
    }
}
