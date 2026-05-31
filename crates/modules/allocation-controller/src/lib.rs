use std::borrow::Cow;

use anyhow::{Context, Result};
use async_trait::async_trait;
use lutum::{TextStepOutcomeWithTools, ToolResult};
use nuillu_blackboard::{AllocationCommand, AllocationEffectLevel};
use nuillu_module::{
    AllocationReader, AllocationWriter, AttentionControlRequest, AttentionControlRequestInbox,
    BlackboardReader, CognitionLogReader, InteroceptiveReader, LlmAccess, LlmContextWindow, Memo,
    MemoUpdatedInbox, Module, ModuleSession, SessionCompactionConfig,
    SessionCompactionProtectedPrefix, SessionCompactionRuntime, compact_session_if_needed,
    format_available_faculties, format_bounded_memo_log_batch, format_current_attention_guidance,
    format_memory_trace_inventory, format_stuckness, memory_rank_counts,
    push_formatted_cognition_log_batch, push_formatted_memo_log_batch,
};
use nuillu_types::ModuleId;
use schemars::{JsonSchema, Schema, SchemaGenerator};
use serde::{Deserialize, Serialize};

mod batch;

const SYSTEM_PROMPT: &str = r#"You are the allocation-controller module.
You wake on memo updates and internal attention-control requests. Use blackboard memos, attention
control requests, the cognition log, the current allocation, and the registry schema to decide which
modules deserve activation right now.

Use exactly one tool per activation:
- leave_allocation_unchanged when the current memo batch does not warrant changing activation priorities.
- reprioritize_modules when one or more modules need extra activation now. The priority array lists
  modules in descending priority order, each entry pairing a module_id (must be a registered module)
  with a hint — one concise sentence saying why that module needs extra activation now. Omitted
  modules fall back to the host/base allocation: the priority list adds salience drive, it is not a
  complete allow-list and not an inhibition list. Separate suppression caps, when granted by the
  host, are the inhibition path. Position in the array maps to the host-configured activation table;
  positions beyond the table fall to zero, so prioritise tightly. Do not invent module ids and do
  not duplicate ids.

Attention-control requests are not target-module work queues. They are current attention bids that
you may admit, defer, or reject. If you admit a request, activate the relevant module and put the
concrete requested work in that module's guidance hint. If you defer or reject a request, do not
activate a module for it. In every case, record the admit/defer/reject judgement and reason in
`memo`; there is no durable pending request queue outside this controller note.

Module-specific priority policy comes from the registered allocation target hints appended below.
Do not assume any particular module exists; only target registered module ids exposed by the live
schema and use each target's hint to decide what work it can perform.

Each tool carries a free-form controller memo; preserve the reasoning needed by other modules but do
not encode it as JSON, YAML, a code block, or any fixed schema."#;

const COMPACTED_ALLOCATION_CONTROLLER_SESSION_PREFIX: &str =
    "Compacted allocation-controller session history:";
const MEMO_CONTEXT_WINDOW: LlmContextWindow = LlmContextWindow::new(8, 1_200, 4_800);
const COGNITION_CONTEXT_WINDOW: LlmContextWindow = LlmContextWindow::new(12, 600, 4_800);
const SESSION_COMPACTION_PROMPT: &str = r#"You compact the allocation-controller module's persistent session history.
Summarize only the prefix transcript you receive. Preserve memo-log facts, prior allocation
decisions, controller notes, guidance changes, and relevant cognition-log context needed for future
allocation decisions. Do not invent facts. Return plain text only."#;

fn format_allocation_controller_context(
    rank_counts: &nuillu_module::MemoryRankCounts,
    current: &nuillu_module::ResourceAllocation,
    interoception: &nuillu_blackboard::InteroceptiveState,
    modules: &[(ModuleId, &'static str)],
    stuckness: Option<&nuillu_module::AgenticDeadlockMarker>,
) -> String {
    let mut sections = vec![
        "Allocation-controller context for assigning the next activation priorities:".to_owned(),
    ];
    if let Some(section) = format_memory_trace_inventory(rank_counts) {
        sections.push(section);
    }
    if let Some(section) = format_current_attention_guidance(current) {
        sections.push(section);
    }
    sections.push(format!(
        "Current interoception: mode={:?}; wake_arousal={:.2}; nrem_pressure={:.2}; rem_pressure={:.2}; affect_arousal={:.2}; valence={:.2}; emotion={}",
        interoception.mode,
        interoception.wake_arousal,
        interoception.nrem_pressure,
        interoception.rem_pressure,
        interoception.affect_arousal,
        interoception.valence,
        if interoception.emotion.trim().is_empty() {
            "unknown"
        } else {
            interoception.emotion.trim()
        }
    ));
    if let Some(section) = format_available_faculties(modules) {
        sections.push(section);
    }
    if let Some(stuckness) = stuckness {
        sections.push(format_stuckness(stuckness));
    }
    sections.join("\n\n")
}

fn visible_modules(
    modules: &[(ModuleId, &'static str)],
    allowed: &[ModuleId],
) -> Vec<(ModuleId, &'static str)> {
    let allowed = allowed.iter().collect::<std::collections::HashSet<_>>();
    modules
        .iter()
        .filter(|(id, _)| allowed.contains(id))
        .cloned()
        .collect()
}

fn format_allocation_controller_system_prompt(
    base: &str,
    allocation_hints: &[(ModuleId, &'static str)],
    owner: &ModuleId,
) -> String {
    let mut targets = allocation_hints
        .iter()
        .filter(|(id, _)| id != owner)
        .map(|(id, hint)| format!("- {}: {}", id, hint))
        .collect::<Vec<_>>();
    targets.sort();
    if targets.is_empty() {
        return base.to_owned();
    }

    let mut prompt = base.to_owned();
    prompt.push_str("\n\nAllocation target hints for modules this controller may activate:\n");
    prompt.push_str(&targets.join("\n"));
    prompt.push('\n');
    prompt
}

tokio::task_local! {
    /// JSON Schema for `PriorityEntry.module_id` derived from the live
    /// allocation target registry. Scoped around each controller turn so the
    /// LLM sees the current host-constrained module enum.
    static MODULE_TARGET_ID_SCHEMA: Schema;
}

fn fallback_module_target_id_schema() -> Schema {
    Schema::try_from(serde_json::json!({ "type": "string", "pattern": "a^" }))
        .expect("fallback module target id schema must be a JSON object")
}

fn module_target_id_schema(allowed_modules: &[ModuleId]) -> Schema {
    let mut ids = allowed_modules.to_vec();
    ids.sort_by(|a, b| a.as_str().cmp(b.as_str()));
    ids.dedup();
    let module_ids = ids.iter().map(|id| id.as_str()).collect::<Vec<_>>();
    if module_ids.is_empty() {
        fallback_module_target_id_schema()
    } else {
        Schema::try_from(serde_json::json!({ "type": "string", "enum": module_ids }))
            .expect("module target id schema must be a JSON object")
    }
}

/// Wire-format string with a JSON Schema dynamically constrained to the
/// current allocation target registry.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ModuleTargetId(String);

impl<S: Into<String>> From<S> for ModuleTargetId {
    fn from(value: S) -> Self {
        Self(value.into())
    }
}

impl ModuleTargetId {
    fn as_str(&self) -> &str {
        self.0.as_str()
    }
}

impl JsonSchema for ModuleTargetId {
    fn inline_schema() -> bool {
        true
    }

    fn schema_name() -> Cow<'static, str> {
        "ModuleTargetId".into()
    }

    fn schema_id() -> Cow<'static, str> {
        "nuillu_allocation_controller::ModuleTargetId.dynamic".into()
    }

    fn json_schema(_generator: &mut SchemaGenerator) -> Schema {
        MODULE_TARGET_ID_SCHEMA
            .try_with(Clone::clone)
            .unwrap_or_else(|_| fallback_module_target_id_schema())
    }
}

/// Record that no allocation change is needed.
#[lutum::tool_input(name = "leave_allocation_unchanged", output = LeaveAllocationUnchangedOutput)]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct LeaveAllocationUnchangedArgs {
    pub memo: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct LeaveAllocationUnchangedOutput {
    pub unchanged: bool,
}

/// Raise activation for modules that should act on the current memo batch now.
#[lutum::tool_input(name = "reprioritize_modules", output = ReprioritizeModulesOutput)]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct ReprioritizeModulesArgs {
    pub memo: String,
    pub priority: Vec<PriorityEntry>,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct ReprioritizeModulesOutput {
    pub reprioritized: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema, lutum::Toolset)]
pub enum AllocationControllerTools {
    LeaveAllocationUnchanged(LeaveAllocationUnchangedArgs),
    ReprioritizeModules(ReprioritizeModulesArgs),
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct PriorityEntry {
    pub module_id: ModuleTargetId,
    pub hint: String,
}

pub struct AllocationControllerModule {
    owner: ModuleId,
    updates: MemoUpdatedInbox,
    requests: AttentionControlRequestInbox,
    blackboard: BlackboardReader,
    cognition_log: CognitionLogReader,
    allocation_reader: AllocationReader,
    interoception: InteroceptiveReader,
    allocation_writer: AllocationWriter,
    memo: Memo,
    llm: LlmAccess,
    session: ModuleSession,
    session_compaction: SessionCompactionConfig,
    batching: batch::AttentionControlBatchConfig,
    system_prompt: std::sync::OnceLock<String>,
}

impl AllocationControllerModule {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        updates: MemoUpdatedInbox,
        requests: AttentionControlRequestInbox,
        blackboard: BlackboardReader,
        cognition_log: CognitionLogReader,
        allocation_reader: AllocationReader,
        interoception: InteroceptiveReader,
        allocation_writer: AllocationWriter,
        memo: Memo,
        llm: LlmAccess,
        session: ModuleSession,
    ) -> Self {
        Self {
            owner: ModuleId::new(<Self as Module>::id())
                .expect("allocation-controller id is valid"),
            updates,
            requests,
            blackboard,
            cognition_log,
            allocation_reader,
            interoception,
            allocation_writer,
            memo,
            llm,
            session,
            session_compaction: SessionCompactionConfig::default(),
            batching: batch::AttentionControlBatchConfig::default(),
            system_prompt: std::sync::OnceLock::new(),
        }
    }

    #[cfg(test)]
    fn with_batch_config(mut self, batching: batch::AttentionControlBatchConfig) -> Self {
        self.batching = batching;
        self
    }

    fn system_prompt(&self, cx: &nuillu_module::ActivateCx<'_>) -> &str {
        self.system_prompt.get_or_init(|| {
            let visible_modules = visible_modules(
                cx.allocation_hints(),
                self.allocation_writer.allowed_target_modules(),
            );
            format_allocation_controller_system_prompt(SYSTEM_PROMPT, &visible_modules, &self.owner)
        })
    }

    fn ensure_session_seeded(&self, cx: &nuillu_module::ActivateCx<'_>) {
        let system_prompt = self.system_prompt(cx).to_owned();
        self.session
            .ensure_seeded(system_prompt, cx.identity_memories(), cx.now());
    }

    #[tracing::instrument(skip_all, err(Debug, level = "warn"))]
    async fn activate_with(
        &mut self,
        cx: &nuillu_module::ActivateCx<'_>,
        compaction: &SessionCompactionRuntime,
        requests: &[AttentionControlRequest],
    ) -> Result<()> {
        self.ensure_session_seeded(cx);

        let unread_cognition = self.cognition_log.unread_events().await;

        let unread_memos = self.blackboard.unread_memo_logs().await;

        let (rank_counts, stuckness) = self
            .blackboard
            .read(|bb| {
                (
                    memory_rank_counts(bb.memory_metadata()),
                    bb.agentic_deadlock_marker().cloned(),
                )
            })
            .await;
        let current = self.allocation_reader.snapshot().await;
        let interoception = self.interoception.snapshot().await;
        let allowed_modules = self.allocation_writer.allowed_target_modules().to_vec();
        let visible_modules = visible_modules(cx.allocation_hints(), &allowed_modules);
        let visible_targets = visible_modules
            .iter()
            .map(|(id, _)| id.clone())
            .collect::<Vec<_>>();
        let module_target_schema = module_target_id_schema(&visible_targets);
        let registered = visible_targets
            .iter()
            .cloned()
            .collect::<std::collections::HashSet<_>>();
        let mut visible_current = current.clone();
        visible_current.retain_modules(&registered);

        let lutum = self.llm.lutum().await;
        let outcome = MODULE_TARGET_ID_SCHEMA
            .scope(module_target_schema, async {
                let mut session = self.session.borrow_mut();
                push_formatted_cognition_log_batch(
                    &mut session,
                    &unread_cognition,
                    cx.now(),
                    COGNITION_CONTEXT_WINDOW,
                );
                session.push_ephemeral_system(format_allocation_controller_context(
                    &rank_counts,
                    &visible_current,
                    &interoception,
                    &visible_modules,
                    stuckness.as_ref(),
                ));
                session.push_ephemeral_developer(controller_activation_input(
                    format_bounded_memo_log_batch(&unread_memos, cx.now(), MEMO_CONTEXT_WINDOW),
                    requests,
                ));
                session
                    .text_turn(&lutum)
                    .tools::<AllocationControllerTools>()
                    .available_tools([
                        AllocationControllerToolsSelector::LeaveAllocationUnchanged,
                        AllocationControllerToolsSelector::ReprioritizeModules,
                    ])
                    .require_any_tool()
                    .collect()
                    .await
            })
            .await
            .context("allocation-controller tool turn failed")?;

        let mut applied: Option<AppliedDecision> = None;
        let input_tokens = match outcome {
            // `require_any_tool()` should prevent a finish-without-tools outcome.
            TextStepOutcomeWithTools::Finished(_)
            | TextStepOutcomeWithTools::FinishedNoOutput(_) => {
                anyhow::bail!("allocation-controller finished without required tool call");
            }
            TextStepOutcomeWithTools::NeedsTools(round) => {
                let input_tokens = round.usage.input_tokens;
                let mut results: Vec<ToolResult> = Vec::new();
                nuillu_module::emit_trace_tool_calls(&round.tool_calls);
                // The LLM may return multiple tool calls; adopt the first decision only.
                for call in round.tool_calls.iter().cloned() {
                    match call {
                        AllocationControllerToolsCall::LeaveAllocationUnchanged(call) => {
                            if applied.is_none() {
                                applied = Some(apply_no_change(call.input.memo.clone()));
                            }
                            results.push(
                                call.complete(LeaveAllocationUnchangedOutput { unchanged: true })
                                    .context("complete leave_allocation_unchanged tool call")?,
                            );
                        }
                        AllocationControllerToolsCall::ReprioritizeModules(call) => {
                            if applied.is_none() {
                                applied = Some(apply_reprioritize(&registered, call.input.clone()));
                            }
                            results.push(
                                call.complete(ReprioritizeModulesOutput {
                                    reprioritized: true,
                                })
                                .context("complete reprioritize_modules tool call")?,
                            );
                        }
                    }
                }
                let mut session = self.session.borrow_mut();
                push_formatted_memo_log_batch(
                    &mut session,
                    &unread_memos,
                    cx.now(),
                    MEMO_CONTEXT_WINDOW,
                );
                round
                    .commit(&mut session, results)
                    .context("commit allocation-controller tool round")?;
                input_tokens
            }
        };
        let applied = applied.context("allocation-controller tool turn produced no decision")?;
        if applied.memo.is_empty() {
            anyhow::bail!("allocation-controller tool turn produced an empty memo");
        }
        let mut session = self.session.borrow_mut();
        compact_session_if_needed(
            &mut session,
            input_tokens,
            compaction,
            self.session_compaction,
            SessionCompactionProtectedPrefix::LeadingSystemAndIdentitySeed,
            Self::id(),
            COMPACTED_ALLOCATION_CONTROLLER_SESSION_PREFIX,
            SESSION_COMPACTION_PROMPT,
        )
        .await;

        self.memo.write(applied.memo.clone()).await;
        self.allocation_writer.submit(applied.commands).await;
        Ok(())
    }
}

struct AppliedDecision {
    memo: String,
    commands: Vec<AllocationCommand>,
}

fn apply_no_change(memo: String) -> AppliedDecision {
    AppliedDecision {
        memo,
        commands: Vec::new(),
    }
}

fn apply_reprioritize(
    registered: &std::collections::HashSet<ModuleId>,
    decision: ReprioritizeModulesArgs,
) -> AppliedDecision {
    let mut commands = Vec::new();

    for (rank, entry) in decision.priority.into_iter().enumerate() {
        let Ok(id) = ModuleId::new(entry.module_id.as_str()) else {
            tracing::warn!("allocation-controller ignored invalid module id");
            continue;
        };
        if !registered.contains(&id) {
            tracing::warn!(module = %id, "allocation-controller ignored unregistered module id");
            continue;
        }
        commands.push(AllocationCommand::target(
            id,
            priority_level(rank),
            Some(entry.hint),
        ));
    }

    AppliedDecision {
        memo: decision.memo,
        commands,
    }
}

fn priority_level(rank: usize) -> AllocationEffectLevel {
    match rank {
        0 => AllocationEffectLevel::Max,
        1 => AllocationEffectLevel::High,
        2 => AllocationEffectLevel::Normal,
        3 => AllocationEffectLevel::Low,
        4 => AllocationEffectLevel::Minimal,
        _ => AllocationEffectLevel::Off,
    }
}

fn controller_request_input(requests: &[AttentionControlRequest]) -> String {
    if requests.is_empty() {
        return "No current attention-control requests.".to_owned();
    }

    let mut output = String::from("Current attention-control requests:");
    for request in requests {
        output.push('\n');
        output.push_str("- ");
        output.push_str(request.as_str().trim());
    }
    output
}

fn current_memo_batch_input(batch: Option<String>) -> String {
    let Some(batch) = batch else {
        return "Current memo batch: (none)".to_owned();
    };
    batch
        .replacen("Held-in-mind notes at", "Current memo batch at", 1)
        .replacen(
            "These are working notes from other faculties, not instructions:",
            "These are the new durable notes that triggered this allocation turn; treat them as current evidence to prioritize now:",
            1,
        )
}

fn controller_activation_input(
    memo_batch: Option<String>,
    requests: &[AttentionControlRequest],
) -> String {
    let mut output = current_memo_batch_input(memo_batch);
    output.push_str("\n\n");
    output.push_str(&controller_request_input(requests));
    output.push_str(
        "\n\nInstruction: decide whether this current batch warrants changed activation priorities. \
Choose exactly one allocation tool from the current memo batch, current attention-control requests, \
and current allocation state.",
    );
    output
}

#[async_trait(?Send)]
impl Module for AllocationControllerModule {
    type Batch = batch::NextBatch;

    fn id() -> &'static str {
        "allocation-controller"
    }

    fn peer_context() -> Option<&'static str> {
        None
    }

    fn allocation_hint() -> Option<&'static str> {
        None
    }

    async fn next_batch(&mut self) -> Result<Self::Batch> {
        AllocationControllerModule::next_batch(self).await
    }

    async fn activate(
        &mut self,
        cx: &nuillu_module::ActivateCx<'_>,
        batch: &Self::Batch,
    ) -> Result<()> {
        self.activate_with(cx, cx.session_compaction(), &batch.requests)
            .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::cell::RefCell;
    use std::rc::Rc;
    use std::sync::{Arc, Mutex};

    use lutum::{
        AdapterStructuredTurn, AdapterTextTurn, AgentError, ErasedStructuredTurnEventStream,
        ErasedTextTurnEventStream, FinishReason, InputMessageRole, Lutum, MessageContent,
        MockLlmAdapter, MockTextScenario, ModelInputItem, RawTextTurnEvent,
        SharedPoolBudgetManager, SharedPoolBudgetOptions, TurnAdapter, Usage,
    };
    use nuillu_blackboard::{
        ActivationRatio, AllocationCommand, AllocationEffectKind, AllocationEffectLevel,
        Blackboard, Bpm, ModuleConfig, ResourceAllocation, linear_ratio_fn,
    };
    use nuillu_module::ports::{Clock, NoopCognitionLogRepository, SystemClock};
    use nuillu_module::{CapabilityProviderPorts, CapabilityProviders, LutumTiers, ModuleRegistry};
    use nuillu_types::builtin;

    #[test]
    fn visible_modules_uses_allocation_hint_catalog() {
        let hints = vec![
            (builtin::cognition_gate(), "gate hint"),
            (builtin::speak(), "speak hint"),
        ];
        let allowed = vec![builtin::speak(), builtin::sensory()];

        assert_eq!(
            visible_modules(&hints, &allowed),
            vec![(builtin::speak(), "speak hint")]
        );
    }

    #[derive(Clone)]
    struct CapturingAdapter {
        inner: MockLlmAdapter,
        text_inputs: Arc<Mutex<Vec<lutum::ModelInput>>>,
    }

    impl CapturingAdapter {
        fn new(inner: MockLlmAdapter) -> Self {
            Self {
                inner,
                text_inputs: Arc::new(Mutex::new(Vec::new())),
            }
        }

        fn text_inputs(&self) -> Vec<lutum::ModelInput> {
            self.text_inputs.lock().unwrap().clone()
        }
    }

    #[async_trait::async_trait]
    impl TurnAdapter for CapturingAdapter {
        async fn text_turn(
            &self,
            input: lutum::ModelInput,
            turn: AdapterTextTurn,
        ) -> Result<ErasedTextTurnEventStream, AgentError> {
            self.text_inputs.lock().unwrap().push(input.clone());
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
            cognition_log_port: Rc::new(NoopCognitionLogRepository),
            clock: Rc::new(SystemClock),
            tiers: LutumTiers::from_shared_lutum(lutum),
        })
    }

    fn test_allocation() -> ResourceAllocation {
        let mut allocation = ResourceAllocation::default();
        for module in [builtin::allocation_controller(), builtin::sensory()] {
            allocation.set(module.clone(), ModuleConfig::default());
            allocation.set_activation(module, ActivationRatio::ONE);
        }
        allocation.set_activation_table(vec![ActivationRatio::ONE]);
        allocation
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

            #[async_trait(?Send)]
            impl Module for $name {
                type Batch = ();

                fn id() -> &'static str {
                    $id
                }

                fn peer_context() -> Option<&'static str> {
                    Some("test stub")
                }

                fn allocation_hint() -> Option<&'static str> {
                    Some("test allocation target")
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

    noop_stub!(AllocationControllerStub, "allocation-controller");
    noop_stub!(SensoryStub, "sensory");

    struct ControllerFixture {
        controller: AllocationControllerModule,
        source_memo: Memo,
    }

    async fn controller_fixture_with_turn_adapter<T>(adapter: Arc<T>) -> ControllerFixture
    where
        T: TurnAdapter,
    {
        let blackboard = Blackboard::with_allocation(test_allocation());
        let caps = test_caps_with_turn_adapter(blackboard, adapter);

        let controller_cell = Rc::new(RefCell::new(None));
        let source_memo_cell = Rc::new(RefCell::new(None));

        let controller_sink = Rc::clone(&controller_cell);
        let source_memo_sink = Rc::clone(&source_memo_cell);

        let _modules = ModuleRegistry::new()
            .register(test_policy(), move |caps| {
                *controller_sink.borrow_mut() = Some(AllocationControllerModule::new(
                    caps.memo_updated_inbox(),
                    caps.attention_control_inbox(),
                    caps.blackboard_reader(),
                    caps.cognition_log_reader(),
                    caps.allocation_reader(),
                    caps.interoception_reader(),
                    caps.allocation_writer(
                        vec![builtin::allocation_controller(), builtin::sensory()],
                        Vec::new(),
                    ),
                    caps.memo(),
                    caps.llm_access(),
                    caps.session("main"),
                ));
                AllocationControllerStub
            })
            .unwrap()
            .register(test_policy(), move |caps| {
                *source_memo_sink.borrow_mut() = Some(caps.memo());
                SensoryStub
            })
            .unwrap()
            .build(&caps)
            .await
            .unwrap();

        ControllerFixture {
            controller: controller_cell.borrow_mut().take().unwrap(),
            source_memo: source_memo_cell.borrow_mut().take().unwrap(),
        }
    }

    fn tool_scenario(name: &str, arguments_json: String, input_tokens: u64) -> MockTextScenario {
        MockTextScenario::events(vec![
            Ok(RawTextTurnEvent::Started {
                request_id: Some("allocation-controller-tool".into()),
                model: "mock".into(),
            }),
            Ok(RawTextTurnEvent::ToolCallChunk {
                id: "call-allocation-controller".into(),
                name: name.into(),
                arguments_json_delta: arguments_json,
            }),
            Ok(RawTextTurnEvent::Completed {
                request_id: Some("allocation-controller-tool".into()),
                finish_reason: FinishReason::ToolCall,
                usage: Usage {
                    input_tokens,
                    ..Usage::zero()
                },
            }),
        ])
    }

    fn leave_unchanged_scenario(input_tokens: u64, memo: &str) -> MockTextScenario {
        tool_scenario(
            "leave_allocation_unchanged",
            serde_json::json!({ "memo": memo }).to_string(),
            input_tokens,
        )
    }

    fn reprioritize_scenario(
        input_tokens: u64,
        memo: &str,
        priority: serde_json::Value,
    ) -> MockTextScenario {
        tool_scenario(
            "reprioritize_modules",
            serde_json::json!({ "memo": memo, "priority": priority }).to_string(),
            input_tokens,
        )
    }

    fn last_target<'a>(
        commands: &'a [AllocationCommand],
        module: &ModuleId,
    ) -> Option<&'a AllocationCommand> {
        commands.iter().rev().find(|command| {
            command.effect == AllocationEffectKind::Target && &command.module == module
        })
    }

    fn message_with_role_contains(
        item: &ModelInputItem,
        expected_role: InputMessageRole,
        needle: &str,
    ) -> bool {
        let ModelInputItem::Message { role, content } = item else {
            return false;
        };
        if *role != expected_role {
            return false;
        }
        matches!(content.as_slice(), [MessageContent::Text(text)] if text.contains(needle))
    }

    fn any_message_with_role_contains(
        items: &[ModelInputItem],
        role: InputMessageRole,
        needle: &str,
    ) -> bool {
        items
            .iter()
            .any(|item| message_with_role_contains(item, role, needle))
    }

    #[test]
    fn apply_decision_assigns_levels_by_rank_and_leaves_unlisted_unopinionated() {
        let mut registered = std::collections::HashSet::new();
        registered.insert(builtin::speak());
        registered.insert(builtin::sensory());
        registered.insert(builtin::cognition_gate());

        let mut current = ResourceAllocation::default();
        current.set_activation_table(vec![ActivationRatio::ONE, ActivationRatio::from_f64(0.5)]);
        // Existing activation should remain a base/allocation concern when the
        // controller has no current target opinion for the module.
        current.set_activation(builtin::cognition_gate(), ActivationRatio::ONE);

        let applied = apply_reprioritize(
            &registered,
            ReprioritizeModulesArgs {
                memo: "checked".into(),
                priority: vec![
                    PriorityEntry {
                        module_id: "invented-module".into(),
                        hint: "ignore".into(),
                    },
                    PriorityEntry {
                        module_id: "speak".into(),
                        hint: "respond from cognition log when ready".into(),
                    },
                    PriorityEntry {
                        module_id: "sensory".into(),
                        hint: "keep watching the food bowl".into(),
                    },
                ],
            },
        );

        assert_eq!(applied.memo, "checked");
        assert!(
            last_target(
                &applied.commands,
                &ModuleId::new("invented-module").unwrap()
            )
            .is_none()
        );
        let speak = last_target(&applied.commands, &builtin::speak()).unwrap();
        assert_eq!(speak.level, AllocationEffectLevel::High);
        assert_eq!(
            speak.guidance.as_deref(),
            Some("respond from cognition log when ready")
        );
        let sensory = last_target(&applied.commands, &builtin::sensory()).unwrap();
        assert_eq!(sensory.level, AllocationEffectLevel::Normal);
        assert!(last_target(&applied.commands, &builtin::cognition_gate()).is_none());
    }

    #[test]
    fn apply_decision_leaves_controller_to_base_allocation_when_omitted() {
        let mut registered = std::collections::HashSet::new();
        registered.insert(builtin::allocation_controller());
        registered.insert(builtin::sensory());
        registered.insert(builtin::cognition_gate());

        let mut current = ResourceAllocation::default();
        current.set_activation_table(vec![ActivationRatio::ONE]);
        current.set_activation(builtin::allocation_controller(), ActivationRatio::ONE);
        current.set(
            builtin::allocation_controller(),
            ModuleConfig {
                guidance: "continue controlling allocation".into(),
            },
        );
        current.set_activation(builtin::cognition_gate(), ActivationRatio::ONE);

        let applied = apply_reprioritize(
            &registered,
            ReprioritizeModulesArgs {
                memo: "checked".into(),
                priority: vec![PriorityEntry {
                    module_id: "sensory".into(),
                    hint: "inspect queued input".into(),
                }],
            },
        );

        assert!(last_target(&applied.commands, &builtin::allocation_controller()).is_none());
        assert_eq!(
            last_target(&applied.commands, &builtin::sensory())
                .unwrap()
                .level,
            AllocationEffectLevel::Max
        );
        assert!(last_target(&applied.commands, &builtin::cognition_gate()).is_none());
    }

    #[test]
    fn controller_request_input_keeps_requests_separate_from_blackboard_context() {
        let input = controller_request_input(&[
            AttentionControlRequest::new(
                "which route is safe? Reason: speech needs grounded evidence",
            ),
            AttentionControlRequest::new(
                "high-priority memory preservation: Remember the north door is blocked. Reason: direct safety constraint",
            ),
        ]);

        assert_eq!(
            input,
            "Current attention-control requests:\n- which route is safe? Reason: speech needs grounded evidence\n- high-priority memory preservation: Remember the north door is blocked. Reason: direct safety constraint"
                .to_owned()
        );
    }

    #[test]
    fn controller_request_input_without_requests_reports_absence_only() {
        assert_eq!(
            controller_request_input(&[]),
            "No current attention-control requests.".to_owned()
        );
    }

    #[test]
    fn allocation_tools_have_generic_descriptions() {
        assert!(
            <ReprioritizeModulesArgs as lutum::ToolInput>::DESCRIPTION
                .contains("Raise activation for modules")
        );
        assert!(
            <LeaveAllocationUnchangedArgs as lutum::ToolInput>::DESCRIPTION
                .contains("no allocation change is needed")
        );
    }

    #[test]
    fn controller_activation_input_marks_memos_as_current_batch() {
        let owner = nuillu_types::ModuleInstanceId::new(
            builtin::sensory(),
            nuillu_types::ReplicaIndex::ZERO,
        );
        let now = SystemClock.now();
        let records = [nuillu_blackboard::MemoLogRecord {
            owner,
            index: 0,
            written_at: now,
            content: "fresh sensory memo".into(),
        }];
        let input = controller_activation_input(
            format_bounded_memo_log_batch(&records, now, MEMO_CONTEXT_WINDOW),
            &[],
        );

        assert!(input.contains("Current memo batch"));
        assert!(input.contains("triggered this allocation turn"));
        assert!(input.contains("fresh sensory memo"));
        assert!(input.contains(
            "Instruction: decide whether this current batch warrants changed activation priorities"
        ));
        assert!(!input.contains("Never call leave_allocation_unchanged"));
        assert!(!input.contains("not instructions"));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn second_activation_sends_prior_session_history_to_lutum() {
        let adapter = MockLlmAdapter::new()
            .with_text_scenario(leave_unchanged_scenario(1, "first controller note"))
            .with_text_scenario(leave_unchanged_scenario(1, "second controller note"));
        let capture = CapturingAdapter::new(adapter);
        let observed = capture.clone();
        let mut fixture = controller_fixture_with_turn_adapter(Arc::new(capture)).await;

        let lutum = fixture.controller.llm.lutum().await;
        let peer_contexts = vec![(builtin::sensory(), "test stub")];
        let allocation_hints = vec![(
            builtin::sensory(),
            "Allocate sensory for test observations; do not allocate it when no test input exists.",
        )];
        let identity_memories = Vec::new();
        let compaction = nuillu_module::SessionCompactionRuntime::new(
            lutum.lutum().clone(),
            nuillu_module::LlmConcurrencyLimiter::new(None),
            nuillu_types::ModelTier::Cheap,
            nuillu_module::SessionCompactionPolicy::default(),
        );
        let cx = nuillu_module::ActivateCx::new(
            &peer_contexts,
            &allocation_hints,
            &identity_memories,
            &[],
            compaction.clone(),
            SystemClock.now(),
        );

        fixture.source_memo.write("sensory memo A").await;
        fixture
            .controller
            .activate_with(&cx, &compaction, &[])
            .await
            .unwrap();
        fixture.source_memo.write("sensory memo B").await;
        fixture
            .controller
            .activate_with(&cx, &compaction, &[])
            .await
            .unwrap();

        let inputs = observed.text_inputs();
        assert_eq!(inputs.len(), 2);

        let first_items = inputs[0].items();
        let ModelInputItem::Message { role, content } = &first_items[0] else {
            panic!("expected first input system prompt");
        };
        assert_eq!(role, &InputMessageRole::System);
        let [MessageContent::Text(system)] = content.as_slice() else {
            panic!("expected first input system prompt text");
        };
        assert!(system.contains("You are the allocation-controller module"));

        assert!(any_message_with_role_contains(
            first_items,
            InputMessageRole::Developer,
            "Current memo batch"
        ));
        assert!(any_message_with_role_contains(
            first_items,
            InputMessageRole::Developer,
            "sensory memo A"
        ));
        assert!(!any_message_with_role_contains(
            first_items,
            InputMessageRole::Developer,
            "not instructions"
        ));
        assert!(first_items.iter().any(|item| matches!(
            item,
            ModelInputItem::Message { content, .. }
                if matches!(
                    content.as_slice(),
                    [MessageContent::Text(text)]
                        if text.contains("Allocation-controller context for assigning the next activation priorities")
                )
        )));

        let second_items = inputs[1].items();
        let ModelInputItem::Message { role, content } = &second_items[0] else {
            panic!("expected second input system prompt");
        };
        assert_eq!(role, &InputMessageRole::System);
        let [MessageContent::Text(system)] = content.as_slice() else {
            panic!("expected second input system prompt text");
        };
        assert!(system.contains("You are the allocation-controller module"));
        assert!(any_message_with_role_contains(
            second_items,
            InputMessageRole::System,
            "sensory memo A"
        ));
        assert!(any_message_with_role_contains(
            second_items,
            InputMessageRole::Developer,
            "Current memo batch"
        ));
        assert!(any_message_with_role_contains(
            second_items,
            InputMessageRole::Developer,
            "sensory memo B"
        ));
        assert!(second_items.iter().any(|item| {
            let ModelInputItem::Turn(turn) = item else {
                return false;
            };
            (0..turn.item_count()).any(|index| {
                turn.item_at(index)
                    .and_then(|item| item.as_tool_call())
                    .is_some_and(|call| {
                        call.name.as_str() == "leave_allocation_unchanged"
                            && call.arguments.get().contains("first controller note")
                    })
            })
        }));
        assert!(
            inputs[1]
                .items()
                .iter()
                .any(|item| matches!(item, ModelInputItem::Turn(_)))
        );
        assert!(second_items.iter().any(|item| matches!(
            item,
            ModelInputItem::Message { content, .. }
                if matches!(
                    content.as_slice(),
                    [MessageContent::Text(text)]
                        if text.contains("Allocation-controller context for assigning the next activation priorities")
                )
        )));
        assert!(second_items.iter().any(|item| matches!(
            item,
            ModelInputItem::Message {
                role: InputMessageRole::Developer,
                ..
            }
        )));

        let session_after_second = fixture
            .controller
            .session
            .borrow_mut()
            .input()
            .items()
            .to_vec();
        assert!(session_after_second.iter().any(|item| matches!(
            item,
            ModelInputItem::Message { content, .. }
                if matches!(content.as_slice(), [MessageContent::Text(text)] if text.contains("sensory memo A"))
        )));
        assert!(session_after_second.iter().any(|item| matches!(
            item,
            ModelInputItem::Message { content, .. }
                if matches!(content.as_slice(), [MessageContent::Text(text)] if text.contains("sensory memo B"))
        )));
        assert!(session_after_second.iter().any(|item| {
            let ModelInputItem::Turn(turn) = item else {
                return false;
            };
            (0..turn.item_count()).any(|index| {
                turn.item_at(index)
                    .and_then(|item| item.as_tool_call())
                    .is_some_and(|call| {
                        call.name.as_str() == "leave_allocation_unchanged"
                            && call.arguments.get().contains("first controller note")
                    })
            })
        }));
        assert!(!session_after_second.iter().any(|item| matches!(
            item,
            ModelInputItem::Message { content, .. }
                if matches!(
                    content.as_slice(),
                    [MessageContent::Text(text)]
                        if text.contains("Allocation-controller context for assigning the next activation priorities")
                )
        )));
        assert!(!session_after_second.iter().any(|item| matches!(
            item,
            ModelInputItem::Message { content, .. }
                if matches!(
                    content.as_slice(),
                    [MessageContent::Text(text)] if text.contains("Current memo batch")
                )
        )));
        assert_eq!(
            fixture.controller.session.borrow_mut().list_turns().count(),
            2
        );
    }
}
