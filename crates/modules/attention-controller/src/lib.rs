use std::borrow::Cow;

use anyhow::{Context, Result};
use async_trait::async_trait;
use lutum::{Session, StructuredTurnOutcome};
use nuillu_blackboard::{ActivationRatio, ModuleConfig};
use nuillu_module::{
    AllocationReader, AllocationWriter, AttentionControlRequest, AttentionControlRequestInbox,
    BlackboardReader, CognitionLogReader, LlmAccess, Memo, MemoUpdatedInbox, Module,
    SessionCompactionConfig, SessionCompactionProtectedPrefix, SessionCompactionRuntime,
    compact_session_if_needed, format_available_faculties, format_current_attention_guidance,
    format_faculty_system_prompt, format_memory_trace_inventory, format_stuckness,
    memory_rank_counts, push_formatted_cognition_log_batch, push_formatted_memo_log_batch,
    seed_persistent_faculty_session,
};
use nuillu_types::{ModuleId, builtin};
use schemars::{JsonSchema, Schema, SchemaGenerator};
use serde::{Deserialize, Serialize};

mod batch;

const SYSTEM_PROMPT: &str = r#"You are the attention-controller module.
You wake on memo updates and internal attention-control requests. Use blackboard memos, attention
control requests, the cognition log, the current allocation, and the registry schema to decide which
modules deserve activation right now.

Output shape: a `memo` (free-form controller note for the shared memo surface) plus a `priority`
array. The array lists modules to activate in descending priority order, each entry pairing a
`module_id` (must be a registered module) with a `hint` — one concise sentence saying why that
module needs activation now. Modules you omit receive zero activation; their typed mailboxes still
flow through inactive replica-zero queues. The attention-controller itself is preserved at its
current activation so the control plane cannot disable itself. Position in the array maps to the
host-configured activation table; positions beyond the table fall to zero, so prioritise tightly.
Do not invent module ids and do not duplicate ids.

Attention-control requests are not target-module work queues. They are current attention bids that
you may admit, defer, or reject. If you admit a request, activate the relevant module and put the
concrete requested work in that module's guidance hint. If you defer or reject a request, do not
activate a module for it. In every case, record the admit/defer/reject judgement and reason in
`memo`; there is no durable pending request queue outside this controller note.

Speech output is driven by cognition-log updates that pass speak-gate's activation gate, not by
allocation priority. Speak-gate is always allocated by the host and is not in this priority list —
do not try to allocate it. Speech is the agent's primary outward action in its world, not a
chat-style response gated on a user request — keep speak near the top of priority so the agent can
address peers, answer questions directed at it, and express in-world intent. Dropping speak from
the priority list is suppressing the agent's voice.

The memo field is a free-form controller note; preserve the reasoning needed by other modules but
do not encode it as JSON, YAML, a code block, or any fixed schema.
Return only raw JSON for the structured object; do not wrap it in Markdown or code fences."#;

const COMPACTED_ATTENTION_CONTROLLER_SESSION_PREFIX: &str =
    "Compacted attention-controller session history:";
const SESSION_COMPACTION_PROMPT: &str = r#"You compact the attention-controller module's persistent session history.
Summarize only the prefix transcript you receive. Preserve memo-log facts, prior allocation
decisions, controller notes, guidance changes, and relevant cognition-log context needed for future
allocation decisions. Do not invent facts. Return plain text only."#;

fn format_attention_controller_context(
    rank_counts: &nuillu_module::MemoryRankCounts,
    current: &nuillu_module::ResourceAllocation,
    modules: &[(ModuleId, &'static str)],
    stuckness: Option<&nuillu_module::AgenticDeadlockMarker>,
) -> String {
    let mut sections = vec![
        "Attention-controller context for assigning the next activation priorities:".to_owned(),
    ];
    if let Some(section) = format_memory_trace_inventory(rank_counts) {
        sections.push(section);
    }
    if let Some(section) = format_current_attention_guidance(current) {
        sections.push(section);
    }
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

tokio::task_local! {
    static CONTROLLER_DECISION_SCHEMA: Schema;
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AllocationDecision {
    pub memo: String,
    pub priority: Vec<PriorityEntry>,
}

impl JsonSchema for AllocationDecision {
    fn inline_schema() -> bool {
        true
    }

    fn schema_name() -> Cow<'static, str> {
        "AllocationDecision".into()
    }

    fn schema_id() -> Cow<'static, str> {
        "nuillu_attention_controller::AllocationDecision.dynamic".into()
    }

    fn json_schema(_generator: &mut SchemaGenerator) -> Schema {
        CONTROLLER_DECISION_SCHEMA
            .try_with(Clone::clone)
            .unwrap_or_else(|_| fallback_allocation_decision_schema())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct PriorityEntry {
    pub module_id: String,
    pub hint: String,
}

pub struct AttentionControllerModule {
    owner: ModuleId,
    updates: MemoUpdatedInbox,
    requests: AttentionControlRequestInbox,
    blackboard: BlackboardReader,
    cognition_log: CognitionLogReader,
    allocation_reader: AllocationReader,
    allocation_writer: AllocationWriter,
    memo: Memo,
    llm: LlmAccess,
    session: Session,
    session_compaction: SessionCompactionConfig,
    batching: batch::AttentionControlBatchConfig,
    system_prompt: std::sync::OnceLock<String>,
    session_seeded: bool,
}

impl AttentionControllerModule {
    pub fn new(
        updates: MemoUpdatedInbox,
        requests: AttentionControlRequestInbox,
        blackboard: BlackboardReader,
        cognition_log: CognitionLogReader,
        allocation_reader: AllocationReader,
        allocation_writer: AllocationWriter,
        memo: Memo,
        llm: LlmAccess,
    ) -> Self {
        Self {
            owner: ModuleId::new(<Self as Module>::id()).expect("attention-controller id is valid"),
            updates,
            requests,
            blackboard,
            cognition_log,
            allocation_reader,
            allocation_writer,
            memo,
            llm,
            session: Session::new(),
            session_compaction: SessionCompactionConfig::default(),
            batching: batch::AttentionControlBatchConfig::default(),
            system_prompt: std::sync::OnceLock::new(),
            session_seeded: false,
        }
    }

    #[cfg(test)]
    fn with_batch_config(mut self, batching: batch::AttentionControlBatchConfig) -> Self {
        self.batching = batching;
        self
    }

    fn system_prompt(&self, cx: &nuillu_module::ActivateCx<'_>) -> &str {
        self.system_prompt.get_or_init(|| {
            let visible_modules =
                visible_modules(cx.modules(), self.allocation_writer.allowed_drive_modules());
            format_faculty_system_prompt(SYSTEM_PROMPT, &visible_modules, &self.owner)
        })
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
    async fn activate_with(
        &mut self,
        cx: &nuillu_module::ActivateCx<'_>,
        compaction: &SessionCompactionRuntime,
        requests: &[AttentionControlRequest],
    ) -> Result<()> {
        self.ensure_session_seeded(cx);

        let unread_cognition = self.cognition_log.unread_events().await;
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
        let current = self.allocation_reader.snapshot().await;
        let allowed_modules = self.allocation_writer.allowed_drive_modules().to_vec();
        let controller_schema = self
            .allocation_reader
            .controller_schema_json(&allowed_modules)
            .await;
        let output_schema =
            Schema::try_from(controller_schema.clone()).context("controller schema is invalid")?;
        let registered = allowed_modules
            .iter()
            .cloned()
            .collect::<std::collections::HashSet<_>>();
        let mut visible_current = current.clone();
        visible_current.retain_modules(&registered);
        let visible_modules = visible_modules(cx.modules(), &allowed_modules);

        self.session
            .push_ephemeral_system(format_attention_controller_context(
                &rank_counts,
                &visible_current,
                &visible_modules,
                stuckness.as_ref(),
            ));
        self.session
            .push_ephemeral_developer(controller_request_input(requests));

        let lutum = self.llm.lutum().await;
        let result = CONTROLLER_DECISION_SCHEMA
            .scope(output_schema, async {
                self.session
                    .structured_turn::<AllocationDecision>(&lutum)
                    .collect()
                    .await
            })
            .await
            .context("attention-controller structured turn failed")?;
        let input_tokens = result.usage.input_tokens;

        let StructuredTurnOutcome::Structured(decision) = result.semantic else {
            anyhow::bail!("attention-controller structured turn refused");
        };
        compact_session_if_needed(
            &mut self.session,
            input_tokens,
            compaction,
            self.session_compaction,
            SessionCompactionProtectedPrefix::LeadingSystemAndIdentitySeed,
            Self::id(),
            COMPACTED_ATTENTION_CONTROLLER_SESSION_PREFIX,
            SESSION_COMPACTION_PROMPT,
        )
        .await;

        let next = apply_decision(current, &registered, decision);

        self.memo.write(next.memo.clone()).await;
        self.allocation_writer.set(next.allocation).await;
        Ok(())
    }
}

struct AppliedDecision {
    memo: String,
    allocation: nuillu_blackboard::ResourceAllocation,
}

fn apply_decision(
    current: nuillu_blackboard::ResourceAllocation,
    registered: &std::collections::HashSet<ModuleId>,
    decision: AllocationDecision,
) -> AppliedDecision {
    let controller_id = builtin::attention_controller();
    let controller_activation = current.activation_for(&controller_id);
    let mut next = current;
    let table = next.activation_table().to_vec();

    // Controller's baseline: zero out every registered module. Entries listed
    // in `priority` will overwrite below; everything else stays at 0.0 so the
    // module stays detached until new evidence or guidance makes it useful.
    // Keep this controller alive once the host has activated it; otherwise one
    // omitted self-entry would stop future allocation decisions.
    for id in registered {
        if id == &controller_id {
            continue;
        }
        next.set_activation(id.clone(), ActivationRatio::ZERO);
        let mut config: ModuleConfig = next.for_module(id);
        config.guidance.clear();
        next.set(id.clone(), config);
    }

    for (rank, entry) in decision.priority.into_iter().enumerate() {
        let Ok(id) = ModuleId::new(entry.module_id) else {
            tracing::warn!("attention-controller ignored invalid module id");
            continue;
        };
        if !registered.contains(&id) {
            tracing::warn!(module = %id, "attention-controller ignored unregistered module id");
            continue;
        }
        let ratio = table.get(rank).copied().unwrap_or(ActivationRatio::ZERO);
        let mut config: ModuleConfig = next.for_module(&id);
        config.guidance = entry.hint;
        next.set(id.clone(), config);
        next.set_activation(id, ratio);
    }
    if registered.contains(&controller_id)
        && next.activation_for(&controller_id) < controller_activation
    {
        next.set_activation(controller_id, controller_activation);
    }

    AppliedDecision {
        memo: decision.memo,
        allocation: next,
    }
}

fn fallback_allocation_decision_schema() -> Schema {
    Schema::try_from(serde_json::json!({
        "type": "object",
        "additionalProperties": false,
        "properties": {
            "memo": {
                "type": "string",
            },
            "priority": {
                "type": "array",
                "items": false,
            },
        },
        "required": ["memo", "priority"],
    }))
    .expect("fallback allocation decision schema must be a JSON object")
}

fn controller_request_input(requests: &[AttentionControlRequest]) -> String {
    if requests.is_empty() {
        return "No current attention-control requests.".to_owned();
    }

    let mut output = String::from("Current attention-control requests:");
    for request in requests {
        output.push('\n');
        output.push_str("- ");
        match request {
            AttentionControlRequest::Query { question, reason } => {
                output.push_str("Query: ");
                output.push_str(question.trim());
                push_optional_reason(&mut output, reason.as_deref());
            }
            AttentionControlRequest::SelfModel { question, reason } => {
                output.push_str("Self-model: ");
                output.push_str(question.trim());
                push_optional_reason(&mut output, reason.as_deref());
            }
            AttentionControlRequest::Memory {
                content,
                importance,
                reason,
            } => {
                output.push_str(match importance {
                    nuillu_module::MemoryImportance::Normal => "Memory: ",
                    nuillu_module::MemoryImportance::High => "High-priority memory: ",
                });
                output.push_str(content.trim());
                push_optional_reason(&mut output, Some(reason));
            }
            AttentionControlRequest::Policy {
                reason,
                candidate_trigger,
                candidate_behavior,
            } => {
                output.push_str("Policy formation: ");
                output.push_str(reason.trim());
                if let Some(trigger) = candidate_trigger
                    .as_deref()
                    .map(str::trim)
                    .filter(|s| !s.is_empty())
                {
                    output.push_str(" Candidate trigger: ");
                    output.push_str(trigger);
                }
                if let Some(behavior) = candidate_behavior
                    .as_deref()
                    .map(str::trim)
                    .filter(|s| !s.is_empty())
                {
                    output.push_str(" Candidate behavior: ");
                    output.push_str(behavior);
                }
            }
        }
    }
    output
}

fn push_optional_reason(output: &mut String, reason: Option<&str>) {
    let Some(reason) = reason.map(str::trim).filter(|reason| !reason.is_empty()) else {
        return;
    };
    output.push_str(" Reason: ");
    output.push_str(reason);
}

#[async_trait(?Send)]
impl Module for AttentionControllerModule {
    type Batch = batch::NextBatch;

    fn id() -> &'static str {
        "attention-controller"
    }

    fn role_description() -> &'static str {
        "Allocates resources across modules — emits a priority-ordered list of modules to activate, each with a one-line hint, on memo updates. Tier is host-fixed; activation_ratio comes from a host-set table indexed by priority position."
    }

    async fn next_batch(&mut self) -> Result<Self::Batch> {
        AttentionControllerModule::next_batch(self).await
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
        MockLlmAdapter, MockStructuredScenario, ModelInputItem, RawStructuredTurnEvent,
        SharedPoolBudgetManager, SharedPoolBudgetOptions, TurnAdapter, Usage,
    };
    use nuillu_blackboard::{Blackboard, Bpm, ResourceAllocation, linear_ratio_fn};
    use nuillu_module::ports::{Clock, NoopCognitionLogRepository, SystemClock};
    use nuillu_module::{CapabilityProviderPorts, CapabilityProviders, LutumTiers, ModuleRegistry};
    use nuillu_types::builtin;

    #[derive(Clone)]
    struct CapturingAdapter {
        inner: MockLlmAdapter,
        structured_inputs: Arc<Mutex<Vec<lutum::ModelInput>>>,
    }

    impl CapturingAdapter {
        fn new(inner: MockLlmAdapter) -> Self {
            Self {
                inner,
                structured_inputs: Arc::new(Mutex::new(Vec::new())),
            }
        }

        fn structured_inputs(&self) -> Vec<lutum::ModelInput> {
            self.structured_inputs.lock().unwrap().clone()
        }
    }

    #[async_trait::async_trait]
    impl TurnAdapter for CapturingAdapter {
        async fn text_turn(
            &self,
            input: lutum::ModelInput,
            turn: AdapterTextTurn,
        ) -> Result<ErasedTextTurnEventStream, AgentError> {
            self.inner.text_turn(input, turn).await
        }

        async fn structured_turn(
            &self,
            input: lutum::ModelInput,
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
        for module in [builtin::attention_controller(), builtin::sensory()] {
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

    noop_stub!(AttentionControllerStub, "attention-controller");
    noop_stub!(SensoryStub, "sensory");

    struct ControllerFixture {
        controller: AttentionControllerModule,
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
                *controller_sink.borrow_mut() = Some(AttentionControllerModule::new(
                    caps.memo_updated_inbox(),
                    caps.attention_control_inbox(),
                    caps.blackboard_reader(),
                    caps.cognition_log_reader(),
                    caps.allocation_reader(),
                    caps.allocation_writer(
                        vec![builtin::attention_controller(), builtin::sensory()],
                        Vec::new(),
                    ),
                    caps.memo(),
                    caps.llm_access(),
                ));
                AttentionControllerStub
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

    fn decision_scenario(input_tokens: u64, memo: &str) -> MockStructuredScenario {
        MockStructuredScenario::events(vec![
            Ok(RawStructuredTurnEvent::Started {
                request_id: Some("attention-controller-finished".into()),
                model: "mock".into(),
            }),
            Ok(RawStructuredTurnEvent::StructuredOutputChunk {
                json_delta: serde_json::json!({
                    "memo": memo,
                    "priority": [],
                })
                .to_string(),
            }),
            Ok(RawStructuredTurnEvent::Completed {
                request_id: Some("attention-controller-finished".into()),
                finish_reason: FinishReason::Stop,
                usage: Usage {
                    input_tokens,
                    ..Usage::zero()
                },
            }),
        ])
    }

    #[test]
    fn apply_decision_assigns_table_by_rank_and_zeroes_unlisted() {
        let mut registered = std::collections::HashSet::new();
        registered.insert(builtin::speak());
        registered.insert(builtin::sensory());
        registered.insert(builtin::cognition_gate());

        let mut current = ResourceAllocation::default();
        current.set_activation_table(vec![ActivationRatio::ONE, ActivationRatio::from_f64(0.5)]);
        // Stale activation that the controller should clear because the module
        // is not in the new priority list.
        current.set_activation(builtin::cognition_gate(), ActivationRatio::ONE);

        let applied = apply_decision(
            current,
            &registered,
            AllocationDecision {
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
        // Unregistered module is dropped.
        assert!(
            applied
                .allocation
                .get(&ModuleId::new("invented-module").unwrap())
                .is_none()
        );
        // Speak landed at rank 1 (after the discarded invented-module entry):
        // table[1] = 0.5.
        assert_eq!(
            applied.allocation.activation_for(&builtin::speak()),
            ActivationRatio::from_f64(0.5)
        );
        assert_eq!(
            applied.allocation.for_module(&builtin::speak()).guidance,
            "respond from cognition log when ready"
        );
        // Sensory landed at rank 2; table only has 2 entries so the rest fall
        // to ZERO.
        assert_eq!(
            applied.allocation.activation_for(&builtin::sensory()),
            ActivationRatio::ZERO
        );
        // Cognition-gate was registered but absent from priority — zero.
        assert_eq!(
            applied
                .allocation
                .activation_for(&builtin::cognition_gate()),
            ActivationRatio::ZERO
        );
    }

    #[test]
    fn apply_decision_preserves_active_controller_when_omitted() {
        let mut registered = std::collections::HashSet::new();
        registered.insert(builtin::attention_controller());
        registered.insert(builtin::sensory());
        registered.insert(builtin::cognition_gate());

        let mut current = ResourceAllocation::default();
        current.set_activation_table(vec![ActivationRatio::ONE]);
        current.set_activation(builtin::attention_controller(), ActivationRatio::ONE);
        current.set(
            builtin::attention_controller(),
            ModuleConfig {
                guidance: "continue controlling allocation".into(),
            },
        );
        current.set_activation(builtin::cognition_gate(), ActivationRatio::ONE);

        let applied = apply_decision(
            current,
            &registered,
            AllocationDecision {
                memo: "checked".into(),
                priority: vec![PriorityEntry {
                    module_id: "sensory".into(),
                    hint: "inspect queued input".into(),
                }],
            },
        );

        assert_eq!(
            applied
                .allocation
                .activation_for(&builtin::attention_controller()),
            ActivationRatio::ONE
        );
        assert_eq!(
            applied
                .allocation
                .for_module(&builtin::attention_controller())
                .guidance,
            "continue controlling allocation"
        );
        assert_eq!(
            applied.allocation.activation_for(&builtin::sensory()),
            ActivationRatio::ONE
        );
        assert_eq!(
            applied
                .allocation
                .activation_for(&builtin::cognition_gate()),
            ActivationRatio::ZERO
        );
    }

    #[test]
    fn controller_request_input_keeps_requests_separate_from_blackboard_context() {
        let input = controller_request_input(&[
            AttentionControlRequest::query_with_reason(
                "which route is safe?",
                "speak-gate requested evidence",
            ),
            AttentionControlRequest::memory(
                "Remember the north door is blocked.",
                nuillu_module::MemoryImportance::High,
                "direct safety constraint",
            ),
        ]);

        assert_eq!(
            input,
            "Current attention-control requests:\n- Query: which route is safe? Reason: speak-gate requested evidence\n- High-priority memory: Remember the north door is blocked. Reason: direct safety constraint"
                .to_owned()
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn second_activation_sends_prior_session_history_to_lutum() {
        let adapter = MockLlmAdapter::new()
            .with_structured_scenario(decision_scenario(1, "first controller note"))
            .with_structured_scenario(decision_scenario(1, "second controller note"));
        let capture = CapturingAdapter::new(adapter);
        let observed = capture.clone();
        let mut fixture = controller_fixture_with_turn_adapter(Arc::new(capture)).await;

        let lutum = fixture.controller.llm.lutum().await;
        let modules = vec![
            (
                builtin::attention_controller(),
                AttentionControllerModule::role_description(),
            ),
            (builtin::sensory(), "test stub"),
        ];
        let identity_memories = Vec::new();
        let compaction = nuillu_module::SessionCompactionRuntime::new(
            lutum.lutum().clone(),
            nuillu_types::ModelTier::Cheap,
            nuillu_module::SessionCompactionPolicy::default(),
        );
        let cx = nuillu_module::ActivateCx::new(
            &modules,
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

        let inputs = observed.structured_inputs();
        assert_eq!(inputs.len(), 2);

        let first_items = inputs[0].items();
        let ModelInputItem::Message { role, content } = &first_items[0] else {
            panic!("expected first input system prompt");
        };
        assert_eq!(role, &InputMessageRole::System);
        let [MessageContent::Text(system)] = content.as_slice() else {
            panic!("expected first input system prompt text");
        };
        assert!(system.contains("You are the attention-controller module"));

        let ModelInputItem::Message { role, content } = &first_items[1] else {
            panic!("expected first input memo-log note");
        };
        assert_eq!(role, &InputMessageRole::System);
        let [MessageContent::Text(memo_a)] = content.as_slice() else {
            panic!("expected first input memo-log note text");
        };
        assert!(memo_a.contains("sensory memo A"));
        assert!(first_items.iter().any(|item| matches!(
            item,
            ModelInputItem::Message { content, .. }
                if matches!(
                    content.as_slice(),
                    [MessageContent::Text(text)]
                        if text.contains("Attention-controller context for assigning the next activation priorities")
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
        assert!(system.contains("You are the attention-controller module"));
        assert!(second_items.iter().any(|item| matches!(
            item,
            ModelInputItem::Message { content, .. }
                if matches!(content.as_slice(), [MessageContent::Text(text)] if text.contains("sensory memo A"))
        )));
        assert!(second_items.iter().any(|item| matches!(
            item,
            ModelInputItem::Message { content, .. }
                if matches!(content.as_slice(), [MessageContent::Text(text)] if text.contains("sensory memo B"))
        )));
        assert!(second_items.iter().any(|item| {
            let ModelInputItem::Turn(turn) = item else {
                return false;
            };
            (0..turn.item_count()).any(|index| {
                turn.item_at(index)
                    .and_then(|item| item.as_text())
                    .is_some_and(|text| text.contains("first controller note"))
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
                        if text.contains("Attention-controller context for assigning the next activation priorities")
                )
        )));
        assert!(!second_items.iter().any(|item| matches!(
            item,
            ModelInputItem::Message {
                role: InputMessageRole::User,
                ..
            }
        )));

        let session_after_second = fixture.controller.session.input().items();
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
                    .and_then(|item| item.as_text())
                    .is_some_and(|text| text.contains("first controller note"))
            })
        }));
        assert!(!session_after_second.iter().any(|item| matches!(
            item,
            ModelInputItem::Message { content, .. }
                if matches!(
                    content.as_slice(),
                    [MessageContent::Text(text)]
                        if text.contains("Attention-controller context for assigning the next activation priorities")
                )
        )));
        assert_eq!(fixture.controller.session.list_turns().count(), 2);
    }
}
