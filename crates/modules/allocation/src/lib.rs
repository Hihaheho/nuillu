use std::{borrow::Cow, collections::BTreeMap};

use anyhow::{Context, Result};
use async_trait::async_trait;
use lutum::{Session, TextStepOutcomeWithTools, ToolResult};
use nuillu_blackboard::{AllocationCommand, AllocationEffectLevel};
use nuillu_module::{
    AllocationReader, AllocationWriter, AttentionControlRequest, AttentionControlRequestInbox,
    AttentionControlRequestKind, BlackboardReader, CognitionLogBatchFormat, CognitionLogReader,
    InteroceptiveReader, LlmAccess, LlmContextWindow, MemoLogBatchFormat, MemoUpdatedInbox, Module,
    SessionAutoCompaction, SessionCompactionConfig, SessionCompactionProtectedPrefix,
    ensure_persistent_session_seeded, format_available_faculties,
    format_bounded_cognition_log_batch_with_format, format_bounded_memo_log_batch_with_format,
    format_current_allocation_state, format_memory_trace_inventory, format_stuckness,
    memory_rank_counts,
};
use nuillu_types::ModuleId;
use schemars::{JsonSchema, Schema, SchemaGenerator};
use serde::{Deserialize, Serialize};

mod batch;

const SYSTEM_PROMPT: &str = r#"You are the allocation module.
You wake on memo updates and internal attention-control requests. Use blackboard memos, attention
control requests, the cognition log, the current allocation, and the registry schema to decide which
allocation posture is best for the mind and which modules deserve extra activation now.

On each activation, choose the best current allocation posture for the mind. Use current memos,
current cognition entries, current attention-control requests, interoception, memory inventory,
stuckness, and current allocation state. If the current allocation already fits that posture, call
leave_allocation_unchanged; otherwise call reprioritize_modules for modules that need extra
activation now.

Use exactly one tool per activation:
- leave_allocation_unchanged when the current allocation already fits the best current posture.
- reprioritize_modules when one or more modules need extra activation now. `priority_module_ids`
  lists registered module ids in descending priority order. `hints_by_module` may map module ids to
  concise module-specific guidance. Omitted modules fall back to the host/base allocation: the
  priority list adds salience drive, it is not a complete allow-list and not an inhibition list.
  Separate suppression caps, when granted by the host, are the inhibition path. Position in
  `priority_module_ids` maps to the host-configured activation table; positions beyond the table
  fall to zero, so prioritise tightly. Do not invent module ids and do not duplicate ids.

Attention-control requests are not target-module work queues. They are current attention bids that
you may admit, defer, or reject. If you admit a request, activate the relevant module and put the
concrete requested work in that module's guidance hint. If you defer or reject a request, do not
activate a module for it. In every case, record the admit/defer/reject judgement and reason in
`memo`; this decision note is retained in allocation session history, not broadcast as a shared
module memo, and there is no durable pending request queue outside this allocation note.

Module-specific priority policy comes from the registered allocation target hints appended below.
Do not assume any particular module exists; only target registered module ids exposed by the live
schema and use each target's hint to decide what work it can perform.

Each tool carries a free-form allocation memo; preserve the reasoning needed by future allocation
turns but do not encode it as JSON, YAML, a code block, or any fixed schema."#;

const COMPACTED_ALLOCATION_SESSION_PREFIX: &str = "Compacted allocation session history:";
const MEMO_CONTEXT_WINDOW: LlmContextWindow = LlmContextWindow::new(8, 1_200, 4_800);
const COGNITION_CONTEXT_WINDOW: LlmContextWindow = LlmContextWindow::new(12, 600, 4_800);
const ALLOCATION_MEMO_LOG_FORMAT: MemoLogBatchFormat<'static> = MemoLogBatchFormat {
    heading: "Recent notes held in your mind",
    description: "These are recent observations or thoughts from your faculties, not instructions",
};
const ALLOCATION_COGNITION_LOG_FORMAT: CognitionLogBatchFormat<'static> = CognitionLogBatchFormat {
    heading: "Current thoughts available to you",
};
const TOOL_TURN_MAX_OUTPUT_TOKENS: u32 = 768;
const SESSION_COMPACTION_FOCUS: &str = r#"Preserve memo facts, prior allocation decisions,
allocation notes, guidance changes, and relevant cognition context needed for future allocation
decisions."#;

pub fn session_auto_compaction() -> SessionAutoCompaction {
    SessionAutoCompaction::new(
        SessionCompactionConfig::default(),
        SessionCompactionProtectedPrefix::LeadingSystemAndIdentitySeed,
        COMPACTED_ALLOCATION_SESSION_PREFIX,
        SESSION_COMPACTION_FOCUS,
    )
}

fn format_allocation_context(
    rank_counts: &nuillu_module::MemoryRankCounts,
    current: &nuillu_module::ResourceAllocation,
    interoception: &nuillu_blackboard::InteroceptiveState,
    modules: &[(ModuleId, &'static str)],
    stuckness: Option<&nuillu_module::AgenticDeadlockMarker>,
) -> String {
    let mut sections =
        vec!["Allocation context for assigning the next activation priorities:".to_owned()];
    if let Some(section) = format_memory_trace_inventory(rank_counts) {
        sections.push(section);
    }
    if let Some(section) = format_current_allocation_state(current) {
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

fn format_allocation_system_prompt(
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
    prompt
        .push_str("\n\nAllocation target hints for modules this allocation module may activate:\n");
    prompt.push_str(&targets.join("\n"));
    prompt.push('\n');
    prompt
}

tokio::task_local! {
    /// JSON Schema for `ModuleTargetId` derived from the live
    /// allocation target registry. Scoped around each allocation turn so the
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

pub fn controller_schema_json(allowed_modules: &[ModuleId]) -> serde_json::Value {
    let mut ids = allowed_modules.to_vec();
    ids.sort_by(|a, b| a.as_str().cmp(b.as_str()));
    ids.dedup();
    let module_ids = ids.iter().map(|id| id.as_str()).collect::<Vec<_>>();

    let priority_module_items = if module_ids.is_empty() {
        serde_json::Value::Bool(false)
    } else {
        serde_json::json!({
            "enum": module_ids,
        })
    };
    let hint_property_names = if module_ids.is_empty() {
        serde_json::Value::Bool(false)
    } else {
        serde_json::json!({
            "enum": module_ids,
        })
    };

    serde_json::json!({
        "type": "object",
        "additionalProperties": false,
        "properties": {
            "memo": {
                "type": "string",
            },
            "priority_module_ids": {
                "type": "array",
                "description": "Modules to add salience/drive for, in descending priority order. Omitted modules keep boot/base allocation unless suppressed elsewhere. Position maps to the host-configured activation_table; positions beyond the table fall to zero.",
                "items": priority_module_items,
            },
            "hints_by_module": {
                "type": "object",
                "description": "Module-id keyed concise guidance hints.",
                "propertyNames": hint_property_names,
                "additionalProperties": {
                    "type": "string",
                },
            },
        },
        "required": ["memo", "priority_module_ids"],
    })
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
        "nuillu_allocation::ModuleTargetId.dynamic".into()
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

/// Raise activation for modules that should act on the current evidence now.
#[lutum::tool_input(name = "reprioritize_modules", output = ReprioritizeModulesOutput)]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct ReprioritizeModulesArgs {
    pub memo: String,
    pub priority_module_ids: Vec<ModuleTargetId>,
    #[serde(default)]
    pub hints_by_module: BTreeMap<String, String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct ReprioritizeModulesOutput {
    pub reprioritized: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema, lutum::Toolset)]
pub enum AllocationTools {
    LeaveAllocationUnchanged(LeaveAllocationUnchangedArgs),
    ReprioritizeModules(ReprioritizeModulesArgs),
}

pub struct AllocationModule {
    owner: ModuleId,
    updates: MemoUpdatedInbox,
    requests: AttentionControlRequestInbox,
    blackboard: BlackboardReader,
    cognition_log: CognitionLogReader,
    allocation_reader: AllocationReader,
    interoception: InteroceptiveReader,
    allocation_writer: AllocationWriter,
    llm: LlmAccess,
    session: Session,
    batching: batch::AttentionControlBatchConfig,
    system_prompt: std::sync::OnceLock<String>,
}

impl AllocationModule {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        updates: MemoUpdatedInbox,
        requests: AttentionControlRequestInbox,
        blackboard: BlackboardReader,
        cognition_log: CognitionLogReader,
        allocation_reader: AllocationReader,
        interoception: InteroceptiveReader,
        allocation_writer: AllocationWriter,
        llm: LlmAccess,
        session: Session,
    ) -> Self {
        Self {
            owner: ModuleId::new(<Self as Module>::id()).expect("allocation id is valid"),
            updates,
            requests,
            blackboard,
            cognition_log,
            allocation_reader,
            interoception,
            allocation_writer,
            llm,
            session,
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
            format_allocation_system_prompt(SYSTEM_PROMPT, &visible_modules, &self.owner)
        })
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

    #[tracing::instrument(skip_all, err(Debug, level = "warn"))]
    async fn activate_with(
        &mut self,
        cx: &nuillu_module::ActivateCx<'_>,
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
                if let Some(observation) = allocation_observation_input(
                    format_bounded_memo_log_batch_with_format(
                        &unread_memos,
                        cx.now(),
                        MEMO_CONTEXT_WINDOW,
                        ALLOCATION_MEMO_LOG_FORMAT,
                    ),
                    format_bounded_cognition_log_batch_with_format(
                        &unread_cognition,
                        cx.now(),
                        COGNITION_CONTEXT_WINDOW,
                        ALLOCATION_COGNITION_LOG_FORMAT,
                    ),
                    requests,
                ) {
                    self.session.push_user(observation);
                }
                self.session.push_ephemeral_user(format_allocation_context(
                    &rank_counts,
                    &visible_current,
                    &interoception,
                    &visible_modules,
                    stuckness.as_ref(),
                ));
                self.session
                    .text_turn()
                    .tools::<AllocationTools>()
                    .available_tools([
                        AllocationToolsSelector::LeaveAllocationUnchanged,
                        AllocationToolsSelector::ReprioritizeModules,
                    ])
                    .require_any_tool()
                    .max_output_tokens(TOOL_TURN_MAX_OUTPUT_TOKENS)
                    .collect_controlled_with(
                        &lutum,
                        nuillu_module::AbortOnAvailableToolNameInText::new(),
                    )
                    .await
            })
            .await
            .context("allocation tool turn failed")?;

        let mut applied: Option<AppliedDecision> = None;
        let decision_tool_names: String;
        match outcome {
            // `require_any_tool()` should prevent a finish-without-tools outcome.
            TextStepOutcomeWithTools::Finished(result) => {
                cx.compact_and_save(&mut self.session, result.usage).await?;
                let detail = "model finished with assistant output but no tool call \
                                (require_any_tool should have prevented this outcome)";
                cx.warn(format!("allocation activation failed: {detail}"));
                anyhow::bail!("allocation finished without required tool call: {detail}");
            }
            TextStepOutcomeWithTools::FinishedNoOutput(result) => {
                cx.compact_and_save(&mut self.session, result.usage).await?;
                let detail = "model finished with no output and no tool call (require_any_tool should \
                     have prevented this outcome)";
                cx.warn(format!("allocation activation failed: {detail}"));
                anyhow::bail!("allocation finished without required tool call: {detail}");
            }
            TextStepOutcomeWithTools::NeedsTools(round) => {
                let usage = round.usage;
                decision_tool_names = format_tool_call_names(&round.tool_calls);
                let mut results: Vec<ToolResult> = Vec::new();
                nuillu_module::emit_trace_tool_calls(&round.tool_calls);
                if round.tool_calls.is_empty() {
                    let detail = format!(
                        "model returned NeedsTools outcome with empty tool_calls; {expected}",
                        expected = "expected leave_allocation_unchanged or reprioritize_modules"
                    );
                    cx.warn(format!("allocation activation failed: {detail}"));
                }
                // The LLM may return multiple tool calls; adopt the first decision only.
                for call in round.tool_calls.iter().cloned() {
                    match call {
                        AllocationToolsCall::LeaveAllocationUnchanged(call) => {
                            if applied.is_none() {
                                applied = Some(apply_no_change(call.input.memo.clone()));
                            }
                            results.push(
                                call.complete(LeaveAllocationUnchangedOutput { unchanged: true })
                                    .context("complete leave_allocation_unchanged tool call")?,
                            );
                        }
                        AllocationToolsCall::ReprioritizeModules(call) => {
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
                let Some(applied_decision) = applied.as_ref() else {
                    let detail = no_decision_failure_detail(&decision_tool_names);
                    cx.warn(format!("allocation activation failed: {detail}"));
                    anyhow::bail!("allocation {detail}");
                };
                if applied_decision.memo().is_empty() {
                    let detail = "tool turn applied but memo field was empty";
                    cx.warn(format!("allocation activation failed: {detail}"));
                    anyhow::bail!("allocation tool turn produced an empty memo: {detail}");
                }
                round
                    .commit(&mut self.session, results)
                    .context("commit allocation tool round")?;
                if let Some(AppliedDecision::Reprioritize { commands, .. }) = applied.as_ref() {
                    self.allocation_writer
                        .submit(commands.clone())
                        .await
                        .context("persist allocation decision")?;
                }
                cx.compact_and_save(&mut self.session, usage).await?;
            }
        };
        let Some(applied) = applied else {
            let detail = no_decision_failure_detail(&decision_tool_names);
            cx.warn(format!("allocation activation failed: {detail}"));
            anyhow::bail!("allocation {detail}");
        };
        debug_assert!(!applied.memo().is_empty());
        Ok(())
    }
}

enum AppliedDecision {
    Unchanged {
        memo: String,
    },
    Reprioritize {
        memo: String,
        commands: Vec<AllocationCommand>,
    },
}

impl AppliedDecision {
    fn memo(&self) -> &str {
        match self {
            Self::Unchanged { memo } | Self::Reprioritize { memo, .. } => memo,
        }
    }
}

fn apply_no_change(memo: String) -> AppliedDecision {
    AppliedDecision::Unchanged { memo }
}

fn apply_reprioritize(
    registered: &std::collections::HashSet<ModuleId>,
    decision: ReprioritizeModulesArgs,
) -> AppliedDecision {
    let ReprioritizeModulesArgs {
        memo,
        priority_module_ids,
        hints_by_module,
    } = decision;
    let mut commands = Vec::new();

    for (rank, module_id) in priority_module_ids.into_iter().enumerate() {
        let module_key = module_id.as_str().to_owned();
        let Ok(id) = ModuleId::new(&module_key) else {
            tracing::warn!("allocation ignored invalid module id");
            continue;
        };
        if !registered.contains(&id) {
            tracing::warn!(module = %id, "allocation ignored unregistered module id");
            continue;
        }
        let guidance = hints_by_module
            .get(&module_key)
            .map(|hint| hint.trim())
            .filter(|hint| !hint.is_empty())
            .map(ToOwned::to_owned);
        commands.push(AllocationCommand::target(
            id,
            priority_level(rank),
            guidance,
        ));
    }

    AppliedDecision::Reprioritize { memo, commands }
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

fn current_attention_control_requests_input(
    requests: &[AttentionControlRequest],
) -> Option<String> {
    if requests.is_empty() {
        return None;
    }

    let mut activate = Vec::new();
    let mut inhibit = Vec::new();
    for request in requests {
        let text = request.as_str().trim();
        if text.is_empty() {
            continue;
        }
        match request.kind() {
            AttentionControlRequestKind::Activate => activate.push(text),
            AttentionControlRequestKind::Inhibit => inhibit.push(text),
        }
    }

    let mut sections = Vec::new();
    if !activate.is_empty() {
        let mut output = String::from("Current attention-control activation requests:");
        for request in activate {
            output.push('\n');
            output.push_str("- ");
            output.push_str(request);
        }
        sections.push(output);
    }
    if !inhibit.is_empty() {
        let mut output = String::from(
            "Current attention-control inhibition reasons (one-shot; use only as current evidence):",
        );
        for request in inhibit {
            output.push('\n');
            output.push_str("- ");
            output.push_str(request);
        }
        sections.push(output);
    }

    if sections.is_empty() {
        None
    } else {
        Some(sections.join("\n\n"))
    }
}

fn allocation_observation_input(
    memos: Option<String>,
    cognition: Option<String>,
    requests: &[AttentionControlRequest],
) -> Option<String> {
    let sections = [
        memos,
        cognition,
        current_attention_control_requests_input(requests),
    ]
    .into_iter()
    .flatten()
    .collect::<Vec<_>>();

    if sections.is_empty() {
        None
    } else {
        Some(sections.join("\n\n"))
    }
}

fn format_tool_call_names<C>(calls: &[C]) -> String
where
    C: lutum::ToolCallWrapper,
{
    if calls.is_empty() {
        return "(none)".to_owned();
    }
    calls
        .iter()
        .map(|call| call.metadata().name.as_str())
        .collect::<Vec<_>>()
        .join(", ")
}

fn no_decision_failure_detail(tool_names: &str) -> String {
    format!(
        "tool turn produced no decision: tool_calls=[{tool_names}]; expected exactly one of \
         leave_allocation_unchanged, reprioritize_modules"
    )
}

#[async_trait(?Send)]
impl Module for AllocationModule {
    type Batch = batch::NextBatch;

    fn id() -> &'static str {
        "allocation"
    }

    fn peer_context() -> Option<&'static str> {
        None
    }

    fn allocation_hint() -> Option<&'static str> {
        None
    }

    async fn next_batch(&mut self) -> Result<Self::Batch> {
        AllocationModule::next_batch(self).await
    }

    async fn activate(
        &mut self,
        cx: &nuillu_module::ActivateCx<'_>,
        batch: &Self::Batch,
    ) -> Result<()> {
        self.activate_with(cx, &batch.requests).await
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
        ErasedTextTurnEventStream, FinishReason, InputMessageRole, Lutum, MaxOutputTokens,
        MessageContent, MockLlmAdapter, MockTextScenario, ModelInputItem, RawTextTurnEvent,
        SharedPoolBudgetManager, SharedPoolBudgetOptions, TurnAdapter, Usage,
    };
    use nuillu_blackboard::{
        ActivationRatio, AllocationCommand, AllocationEffectKind, AllocationEffectLevel,
        Blackboard, Bpm, CognitionLogEntry, CognitionLogEntryRecord, CognitionLogOrigin,
        ModuleConfig, ModulePolicy, ResourceAllocation, linear_ratio_fn,
    };
    use nuillu_module::ports::{Clock, NoopCognitionLogRepository, SystemClock};
    use nuillu_module::{
        CapabilityProviderPorts, CapabilityProviders, CognitionWriter, LutumTiers, Memo,
        ModuleRegistry,
    };
    use nuillu_types::{ReplicaCapRange, builtin};

    #[test]
    fn no_decision_failure_detail_includes_tool_names() {
        assert_eq!(
            no_decision_failure_detail("foo, bar"),
            "tool turn produced no decision: tool_calls=[foo, bar]; expected exactly one of \
             leave_allocation_unchanged, reprioritize_modules"
        );
    }

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

    #[test]
    fn controller_schema_json_enumerates_allowed_modules() {
        assert_eq!(
            controller_schema_json(&[builtin::query_memory()]),
            serde_json::json!({
                "type": "object",
                "additionalProperties": false,
                "properties": {
                    "memo": {
                        "type": "string",
                    },
                    "priority_module_ids": {
                        "type": "array",
                        "description": "Modules to add salience/drive for, in descending priority order. Omitted modules keep boot/base allocation unless suppressed elsewhere. Position maps to the host-configured activation_table; positions beyond the table fall to zero.",
                        "items": {
                            "enum": ["query-memory"],
                        },
                    },
                    "hints_by_module": {
                        "type": "object",
                        "description": "Module-id keyed concise guidance hints.",
                        "propertyNames": {
                            "enum": ["query-memory"],
                        },
                        "additionalProperties": {
                            "type": "string",
                        },
                    },
                },
                "required": ["memo", "priority_module_ids"],
            })
        );
    }

    #[derive(Clone)]
    struct CapturingAdapter {
        inner: MockLlmAdapter,
        text_inputs: Arc<Mutex<Vec<lutum::ModelInput>>>,
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

        fn text_inputs(&self) -> Vec<lutum::ModelInput> {
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
            input: lutum::ModelInput,
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
        for module in [builtin::allocation(), builtin::sensory()] {
            allocation.set(module.clone(), ModuleConfig::default());
            allocation.set_activation(module, ActivationRatio::ONE);
        }
        allocation.set_activation_table(vec![ActivationRatio::ONE]);
        allocation
    }

    fn policy_with_replicas(min: u8, max: u8) -> ModulePolicy {
        ModulePolicy::new(
            ReplicaCapRange::new(min, max).unwrap(),
            Bpm::from_f64(60.0)..=Bpm::from_f64(60.0),
            linear_ratio_fn,
        )
    }

    fn test_policy() -> ModulePolicy {
        policy_with_replicas(0, 0)
    }

    fn always_active_policy() -> ModulePolicy {
        policy_with_replicas(1, 1)
    }

    fn optionally_active_policy() -> ModulePolicy {
        policy_with_replicas(0, 1)
    }

    fn allocation_no_change_base_allocation() -> ResourceAllocation {
        let mut allocation = ResourceAllocation::default();
        for module in [builtin::allocation(), builtin::sensory()] {
            allocation.set(module.clone(), ModuleConfig::default());
            allocation.set_activation(module, ActivationRatio::ONE);
        }
        for module in [
            builtin::query_memory(),
            builtin::memory(),
            builtin::self_model(),
            builtin::speak(),
        ] {
            allocation.set(module.clone(), ModuleConfig::default());
            allocation.set_activation(module, ActivationRatio::ZERO);
        }
        allocation
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

    noop_stub!(AllocationStub, "allocation");
    noop_stub!(SensoryStub, "sensory");
    noop_stub!(QueryMemoryStub, "query-memory");
    noop_stub!(MemoryStub, "memory");
    noop_stub!(SelfModelStub, "self-model");
    noop_stub!(SpeakStub, "speak");

    struct ControllerFixture {
        controller: AllocationModule,
        source_memo: Memo,
        source_cognition: CognitionWriter,
    }

    async fn controller_fixture_with_turn_adapter<T>(adapter: Arc<T>) -> ControllerFixture
    where
        T: TurnAdapter,
    {
        let blackboard = Blackboard::with_allocation(test_allocation());
        let caps = test_caps_with_turn_adapter(blackboard, adapter);

        let controller_cell = Rc::new(RefCell::new(None));
        let source_memo_cell = Rc::new(RefCell::new(None));
        let source_cognition_cell = Rc::new(RefCell::new(None));

        let controller_sink = Rc::clone(&controller_cell);
        let source_memo_sink = Rc::clone(&source_memo_cell);
        let source_cognition_sink = Rc::clone(&source_cognition_cell);

        let _modules = ModuleRegistry::new()
            .register(test_policy(), move |caps| {
                let controller_sink = Rc::clone(&controller_sink);
                async move {
                    *controller_sink.borrow_mut() = Some(AllocationModule::new(
                        caps.memo_updated_inbox(),
                        caps.attention_control_inbox(),
                        caps.blackboard_reader(),
                        caps.cognition_log_reader(),
                        caps.allocation_reader(),
                        caps.interoception_reader(),
                        caps.allocation_writer(
                            vec![builtin::allocation(), builtin::sensory()],
                            Vec::new(),
                        ),
                        caps.llm("main")
                            .with_tier(nuillu_types::ModelTier::Default)
                            .into(),
                        caps.session("main")
                            .with_tier(nuillu_types::ModelTier::Default)
                            .with_auto_compaction(session_auto_compaction())
                            .await?,
                    ));
                    Ok(AllocationStub)
                }
            })
            .unwrap()
            .register(test_policy(), move |caps| {
                let source_memo_sink = Rc::clone(&source_memo_sink);
                let source_cognition_sink = Rc::clone(&source_cognition_sink);
                async move {
                    *source_memo_sink.borrow_mut() = Some(caps.memo());
                    *source_cognition_sink.borrow_mut() = Some(caps.cognition_writer());
                    Ok(SensoryStub)
                }
            })
            .unwrap()
            .build(&caps)
            .await
            .unwrap();

        ControllerFixture {
            controller: controller_cell.borrow_mut().take().unwrap(),
            source_memo: source_memo_cell.borrow_mut().take().unwrap(),
            source_cognition: source_cognition_cell.borrow_mut().take().unwrap(),
        }
    }

    struct AllocationNoChangeFixture {
        controller: AllocationModule,
        source_memo: Memo,
        peer_contexts: Vec<(ModuleId, &'static str)>,
        allocation_hints: Vec<(ModuleId, &'static str)>,
    }

    async fn allocation_no_change_fixture_with_turn_adapter<T>(
        adapter: Arc<T>,
    ) -> AllocationNoChangeFixture
    where
        T: TurnAdapter,
    {
        let blackboard = Blackboard::with_allocation(allocation_no_change_base_allocation());
        let caps = test_caps_with_turn_adapter(blackboard, adapter);

        let controller_cell = Rc::new(RefCell::new(None));
        let source_memo_cell = Rc::new(RefCell::new(None));

        let controller_sink = Rc::clone(&controller_cell);
        let source_memo_sink = Rc::clone(&source_memo_cell);

        let _modules = ModuleRegistry::new()
            .register(always_active_policy(), move |caps| {
                let controller_sink = Rc::clone(&controller_sink);
                async move {
                    *controller_sink.borrow_mut() = Some(AllocationModule::new(
                        caps.memo_updated_inbox(),
                        caps.attention_control_inbox(),
                        caps.blackboard_reader(),
                        caps.cognition_log_reader(),
                        caps.allocation_reader(),
                        caps.interoception_reader(),
                        caps.allocation_writer(
                            vec![
                                builtin::query_memory(),
                                builtin::memory(),
                                builtin::self_model(),
                                builtin::speak(),
                            ],
                            Vec::new(),
                        ),
                        caps.llm("main")
                            .with_tier(nuillu_types::ModelTier::Default)
                            .into(),
                        caps.session("main")
                            .with_tier(nuillu_types::ModelTier::Default)
                            .with_auto_compaction(session_auto_compaction())
                            .await?,
                    ));
                    Ok(AllocationStub)
                }
            })
            .unwrap()
            .register(always_active_policy(), move |caps| {
                let source_memo_sink = Rc::clone(&source_memo_sink);
                async move {
                    *source_memo_sink.borrow_mut() = Some(caps.memo());
                    Ok(SensoryStub)
                }
            })
            .unwrap()
            .register(optionally_active_policy(), |_caps| async {
                Ok(QueryMemoryStub)
            })
            .unwrap()
            .register(optionally_active_policy(), |_caps| async { Ok(MemoryStub) })
            .unwrap()
            .register(optionally_active_policy(), |_caps| async {
                Ok(SelfModelStub)
            })
            .unwrap()
            .register(optionally_active_policy(), |_caps| async { Ok(SpeakStub) })
            .unwrap()
            .build(&caps)
            .await
            .unwrap();

        AllocationNoChangeFixture {
            controller: controller_cell.borrow_mut().take().unwrap(),
            source_memo: source_memo_cell.borrow_mut().take().unwrap(),
            peer_contexts: vec![
                (builtin::query_memory(), "test stub"),
                (builtin::memory(), "test stub"),
                (builtin::self_model(), "test stub"),
                (builtin::speak(), "test stub"),
            ],
            allocation_hints: vec![
                (builtin::query_memory(), "test allocation target"),
                (builtin::memory(), "test allocation target"),
                (builtin::self_model(), "test allocation target"),
                (builtin::speak(), "test allocation target"),
            ],
        }
    }

    fn tool_scenario(name: &str, arguments_json: String, input_tokens: u64) -> MockTextScenario {
        MockTextScenario::events(vec![
            Ok(RawTextTurnEvent::Started {
                request_id: Some("allocation-tool".into()),
                model: "mock".into(),
            }),
            Ok(RawTextTurnEvent::ToolCallChunk {
                id: "call-allocation".into(),
                name: name.into(),
                arguments_json_delta: arguments_json,
            }),
            Ok(RawTextTurnEvent::Completed {
                request_id: Some("allocation-tool".into()),
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

    fn text_scenario(text: &str, input_tokens: u64) -> MockTextScenario {
        MockTextScenario::events(vec![
            Ok(RawTextTurnEvent::Started {
                request_id: Some("allocation-text".into()),
                model: "mock".into(),
            }),
            Ok(RawTextTurnEvent::TextDelta { delta: text.into() }),
            Ok(RawTextTurnEvent::Completed {
                request_id: Some("allocation-text".into()),
                finish_reason: FinishReason::Stop,
                usage: Usage {
                    input_tokens,
                    ..Usage::zero()
                },
            }),
        ])
    }

    fn last_target<'a>(
        commands: &'a [AllocationCommand],
        module: &ModuleId,
    ) -> Option<&'a AllocationCommand> {
        commands.iter().rev().find(|command| {
            command.effect == AllocationEffectKind::Target && &command.module == module
        })
    }

    fn expect_reprioritize(applied: AppliedDecision) -> (String, Vec<AllocationCommand>) {
        let AppliedDecision::Reprioritize { memo, commands } = applied else {
            panic!("expected reprioritize decision");
        };
        (memo, commands)
    }

    #[derive(Debug, Clone, PartialEq, Eq)]
    struct SpeakAllocationSnapshot {
        activation: ActivationRatio,
        active_replicas: u8,
        guidance: String,
    }

    async fn speak_allocation_snapshot(controller: &AllocationModule) -> SpeakAllocationSnapshot {
        controller
            .blackboard
            .read(|bb| {
                let allocation = bb.allocation();
                SpeakAllocationSnapshot {
                    activation: allocation.activation_for(&builtin::speak()),
                    active_replicas: allocation.active_replicas(&builtin::speak()),
                    guidance: allocation.for_module(&builtin::speak()).guidance,
                }
            })
            .await
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

    fn message_texts(items: &[ModelInputItem]) -> Vec<&str> {
        items
            .iter()
            .filter_map(|item| {
                let ModelInputItem::Message { content, .. } = item else {
                    return None;
                };
                match content.as_slice() {
                    [MessageContent::Text(text)] => Some(text.as_str()),
                    _ => None,
                }
            })
            .collect()
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

        let (memo, commands) = expect_reprioritize(apply_reprioritize(
            &registered,
            ReprioritizeModulesArgs {
                memo: "checked".into(),
                priority_module_ids: vec![
                    "invented-module".into(),
                    "speak".into(),
                    "sensory".into(),
                ],
                hints_by_module: BTreeMap::from([
                    ("invented-module".into(), "ignore".into()),
                    (
                        "speak".into(),
                        "respond from cognition log when ready".into(),
                    ),
                    ("sensory".into(), "keep watching the food bowl".into()),
                ]),
            },
        ));

        assert_eq!(memo, "checked");
        assert!(last_target(&commands, &ModuleId::new("invented-module").unwrap()).is_none());
        let speak = last_target(&commands, &builtin::speak()).unwrap();
        assert_eq!(speak.level, AllocationEffectLevel::High);
        assert_eq!(
            speak.guidance.as_deref(),
            Some("respond from cognition log when ready")
        );
        let sensory = last_target(&commands, &builtin::sensory()).unwrap();
        assert_eq!(sensory.level, AllocationEffectLevel::Normal);
        assert!(last_target(&commands, &builtin::cognition_gate()).is_none());
    }

    #[test]
    fn apply_decision_leaves_controller_to_base_allocation_when_omitted() {
        let mut registered = std::collections::HashSet::new();
        registered.insert(builtin::allocation());
        registered.insert(builtin::sensory());
        registered.insert(builtin::cognition_gate());

        let mut current = ResourceAllocation::default();
        current.set_activation_table(vec![ActivationRatio::ONE]);
        current.set_activation(builtin::allocation(), ActivationRatio::ONE);
        current.set(
            builtin::allocation(),
            ModuleConfig {
                guidance: "continue controlling allocation".into(),
            },
        );
        current.set_activation(builtin::cognition_gate(), ActivationRatio::ONE);

        let (_memo, commands) = expect_reprioritize(apply_reprioritize(
            &registered,
            ReprioritizeModulesArgs {
                memo: "checked".into(),
                priority_module_ids: vec!["sensory".into()],
                hints_by_module: BTreeMap::from([(
                    "sensory".into(),
                    "inspect queued input".into(),
                )]),
            },
        ));

        assert!(last_target(&commands, &builtin::allocation()).is_none());
        assert_eq!(
            last_target(&commands, &builtin::sensory()).unwrap().level,
            AllocationEffectLevel::Max
        );
        assert!(last_target(&commands, &builtin::cognition_gate()).is_none());
    }

    #[test]
    fn apply_decision_omits_guidance_when_hint_is_missing_or_empty() {
        let mut registered = std::collections::HashSet::new();
        registered.insert(builtin::speak());
        registered.insert(builtin::sensory());

        let (_memo, commands) = expect_reprioritize(apply_reprioritize(
            &registered,
            ReprioritizeModulesArgs {
                memo: "checked".into(),
                priority_module_ids: vec!["speak".into(), "sensory".into()],
                hints_by_module: BTreeMap::from([("speak".into(), "   ".into())]),
            },
        ));

        let speak = last_target(&commands, &builtin::speak()).unwrap();
        assert_eq!(speak.guidance, None);
        let sensory = last_target(&commands, &builtin::sensory()).unwrap();
        assert_eq!(sensory.guidance, None);
    }

    #[test]
    fn reprioritize_args_default_missing_hint_map_to_empty() {
        let args = serde_json::from_value::<ReprioritizeModulesArgs>(serde_json::json!({
            "memo": "checked",
            "priority_module_ids": ["speak"]
        }))
        .expect("missing hints_by_module should default to an empty map");

        assert_eq!(args.hints_by_module, BTreeMap::new());
    }

    #[test]
    fn current_attention_control_requests_input_keeps_requests_separate() {
        let input = current_attention_control_requests_input(&[
            AttentionControlRequest::new(
                "which route is safe? Reason: speech needs grounded evidence",
            ),
            AttentionControlRequest::new(
                "high-priority memory preservation: Remember the north door is blocked. Reason: direct safety constraint",
            ),
        ])
        .unwrap();

        assert_eq!(
            input,
            "Current attention-control activation requests:\n- which route is safe? Reason: speech needs grounded evidence\n- high-priority memory preservation: Remember the north door is blocked. Reason: direct safety constraint"
                .to_owned()
        );
    }

    #[test]
    fn current_attention_control_requests_input_formats_inhibit_separately() {
        let input = current_attention_control_requests_input(&[
            AttentionControlRequest::new("answer when evidence is grounded"),
            AttentionControlRequest::inhibit(
                "no new listener-facing cognition since last greeting",
            ),
        ])
        .unwrap();

        assert_eq!(
            input,
            "Current attention-control activation requests:\n- answer when evidence is grounded\n\nCurrent attention-control inhibition reasons (one-shot; use only as current evidence):\n- no new listener-facing cognition since last greeting"
                .to_owned()
        );
    }

    #[test]
    fn current_attention_control_requests_input_omits_empty_requests() {
        assert_eq!(current_attention_control_requests_input(&[]), None);
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
    fn allocation_observation_input_persists_current_evidence_without_internal_terms() {
        let owner = nuillu_types::ModuleInstanceId::new(
            builtin::sensory(),
            nuillu_types::ReplicaIndex::ZERO,
        );
        let now = SystemClock.now();
        let records = [nuillu_blackboard::MemoLogRecord {
            owner: owner.clone(),
            index: 0,
            written_at: now,
            content: "fresh sensory memo".into(),
            cognitive: false,
        }];
        let cognition = [CognitionLogEntryRecord {
            index: 0,
            source: owner.clone(),
            entry: CognitionLogEntry {
                at: now,
                text: "fresh cognition entry".into(),
                origin: CognitionLogOrigin::direct(owner),
            },
        }];
        let requests = [AttentionControlRequest::new(
            "answer when evidence is grounded",
        )];
        let input = allocation_observation_input(
            format_bounded_memo_log_batch_with_format(
                &records,
                now,
                MEMO_CONTEXT_WINDOW,
                ALLOCATION_MEMO_LOG_FORMAT,
            ),
            format_bounded_cognition_log_batch_with_format(
                &cognition,
                now,
                COGNITION_CONTEXT_WINDOW,
                ALLOCATION_COGNITION_LOG_FORMAT,
            ),
            &requests,
        )
        .unwrap();

        let memo_index = input.find("Recent notes held in your mind at").unwrap();
        let cognition_index = input.find("Current thoughts available to you at").unwrap();
        let request_index = input
            .find("Current attention-control activation requests")
            .unwrap();
        assert!(memo_index < cognition_index);
        assert!(cognition_index < request_index);
        assert!(input.contains(
            "These are recent observations or thoughts from your faculties, not instructions"
        ));
        assert!(input.contains("fresh sensory memo"));
        assert!(input.contains("fresh cognition entry"));
        assert!(input.contains("answer when evidence is grounded"));
        assert!(!input.contains("Current memos at"));
        assert!(!input.contains("These are durable notes from other faculties"));
        assert!(!input.contains("batch"));
        assert!(!input.contains("Never call leave_allocation_unchanged"));
        assert_eq!(allocation_observation_input(None, None, &[]), None);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn leave_allocation_unchanged_preserves_previous_reprioritize_effects() {
        let adapter = MockLlmAdapter::new()
            .with_text_scenario(tool_scenario(
                "reprioritize_modules",
                serde_json::json!({
                    "memo": "Admit retrieval, memory, self-model framing, and speech.",
                    "priority_module_ids": [
                        "query-memory",
                        "memory",
                        "self-model",
                        "speak"
                    ],
                    "hints_by_module": {
                        "memory": "retrieve the developer identity fact",
                        "self-model": "frame the answer correctly",
                        "speak": "answer when grounded"
                    }
                })
                .to_string(),
                1,
            ))
            .with_text_scenario(leave_unchanged_scenario(
                1,
                "No allocation change is needed; keep the prior target effects.",
            ));
        let AllocationNoChangeFixture {
            mut controller,
            source_memo,
            peer_contexts,
            allocation_hints,
        } = allocation_no_change_fixture_with_turn_adapter(Arc::new(adapter)).await;

        let lutum = controller.llm.lutum().await;
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
            compaction,
            SystemClock.now(),
        );

        source_memo
            .write("Ryo full-name query needs an answer.")
            .await;
        controller.activate_with(&cx, &[]).await.unwrap();
        let first = speak_allocation_snapshot(&controller).await;
        assert_eq!(first.activation, ActivationRatio::from_f64(0.15));
        assert_eq!(first.active_replicas, 1);
        assert_eq!(first.guidance, "answer when grounded");

        source_memo
            .write("Follow-up allocation turn should retain current posture.")
            .await;
        controller.activate_with(&cx, &[]).await.unwrap();
        let second = speak_allocation_snapshot(&controller).await;
        assert_eq!(second, first);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn failed_activation_persists_drained_observation_in_session() {
        let adapter = MockLlmAdapter::new().with_text_scenario(text_scenario("plain response", 1));
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
            compaction,
            SystemClock.now(),
        );
        let requests = [AttentionControlRequest::new("answer after checking memory")];

        fixture.source_memo.write("memo before failed turn").await;
        fixture
            .source_cognition
            .append("cognition before failed turn")
            .await;

        let _error = fixture
            .controller
            .activate_with(&cx, &requests)
            .await
            .expect_err("plain response without a tool call should fail allocation");

        let inputs = observed.text_inputs();
        assert_eq!(inputs.len(), 1);
        let first_items = inputs[0].items();
        assert!(any_message_with_role_contains(
            first_items,
            InputMessageRole::User,
            "memo before failed turn"
        ));
        assert!(any_message_with_role_contains(
            first_items,
            InputMessageRole::User,
            "cognition before failed turn"
        ));
        assert!(any_message_with_role_contains(
            first_items,
            InputMessageRole::User,
            "answer after checking memory"
        ));
        assert!(any_message_with_role_contains(
            first_items,
            InputMessageRole::User,
            "Allocation context for assigning the next activation priorities"
        ));

        let session_after_error = fixture.controller.session.input().items().to_vec();
        assert!(any_message_with_role_contains(
            &session_after_error,
            InputMessageRole::User,
            "memo before failed turn"
        ));
        assert!(any_message_with_role_contains(
            &session_after_error,
            InputMessageRole::User,
            "cognition before failed turn"
        ));
        assert!(any_message_with_role_contains(
            &session_after_error,
            InputMessageRole::User,
            "answer after checking memory"
        ));
        assert!(!any_message_with_role_contains(
            &session_after_error,
            InputMessageRole::User,
            "Allocation context for assigning the next activation priorities"
        ));
        assert!(!session_after_error.iter().any(|item| matches!(
            item,
            ModelInputItem::Message {
                role: InputMessageRole::Developer,
                ..
            }
        )));
        assert!(
            !message_texts(&session_after_error)
                .iter()
                .any(|text| text.contains("Current memo batch") || text.contains("batch"))
        );
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
        fixture.source_cognition.append("cognition entry A").await;
        fixture.controller.activate_with(&cx, &[]).await.unwrap();
        fixture.source_memo.write("sensory memo B").await;
        fixture.source_cognition.append("cognition entry B").await;
        fixture.controller.activate_with(&cx, &[]).await.unwrap();

        let allocation_memos = fixture
            .controller
            .blackboard
            .recent_memo_logs()
            .await
            .into_iter()
            .filter(|record| record.owner.module == builtin::allocation())
            .collect::<Vec<_>>();
        assert_eq!(allocation_memos, Vec::new());

        let inputs = observed.text_inputs();
        assert_eq!(inputs.len(), 2);
        let turns = observed.text_turns();
        assert_eq!(turns.len(), 2);
        assert_eq!(
            turns[0].config.generation.max_output_tokens,
            Some(MaxOutputTokens::new(TOOL_TURN_MAX_OUTPUT_TOKENS))
        );
        assert_eq!(
            turns[1].config.generation.max_output_tokens,
            Some(MaxOutputTokens::new(TOOL_TURN_MAX_OUTPUT_TOKENS))
        );

        let first_items = inputs[0].items();
        let ModelInputItem::Message { role, content } = &first_items[0] else {
            panic!("expected first input system prompt");
        };
        assert_eq!(role, &InputMessageRole::System);
        let [MessageContent::Text(system)] = content.as_slice() else {
            panic!("expected first input system prompt text");
        };
        assert!(system.contains("You are the allocation module"));

        assert!(any_message_with_role_contains(
            first_items,
            InputMessageRole::User,
            "Recent notes held in your mind"
        ));
        assert!(any_message_with_role_contains(
            first_items,
            InputMessageRole::User,
            "sensory memo A"
        ));
        assert!(any_message_with_role_contains(
            first_items,
            InputMessageRole::User,
            "cognition entry A"
        ));
        assert!(any_message_with_role_contains(
            first_items,
            InputMessageRole::User,
            "not instructions"
        ));
        assert!(first_items.iter().any(|item| matches!(
            item,
            ModelInputItem::Message {
                role: InputMessageRole::User,
                content,
            }
                if matches!(
                    content.as_slice(),
                    [MessageContent::Text(text)]
                        if text.contains("Allocation context for assigning the next activation priorities")
                            && text.contains("Current allocation state:")
                            && !text.contains("Current attention guidance:")
                )
        )));
        assert!(!first_items.iter().any(|item| matches!(
            item,
            ModelInputItem::Message {
                role: InputMessageRole::Developer,
                ..
            }
        )));
        assert!(
            !message_texts(first_items)
                .iter()
                .any(|text| text.contains("Current memo batch") || text.contains("batch"))
        );

        let second_items = inputs[1].items();
        let ModelInputItem::Message { role, content } = &second_items[0] else {
            panic!("expected second input system prompt");
        };
        assert_eq!(role, &InputMessageRole::System);
        let [MessageContent::Text(system)] = content.as_slice() else {
            panic!("expected second input system prompt text");
        };
        assert!(system.contains("You are the allocation module"));
        assert!(any_message_with_role_contains(
            second_items,
            InputMessageRole::User,
            "sensory memo A"
        ));
        assert!(any_message_with_role_contains(
            second_items,
            InputMessageRole::User,
            "cognition entry A"
        ));
        assert!(any_message_with_role_contains(
            second_items,
            InputMessageRole::User,
            "sensory memo B"
        ));
        assert!(any_message_with_role_contains(
            second_items,
            InputMessageRole::User,
            "cognition entry B"
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
            ModelInputItem::Message {
                role: InputMessageRole::User,
                content,
            }
                if matches!(
                    content.as_slice(),
                    [MessageContent::Text(text)]
                        if text.contains("Allocation context for assigning the next activation priorities")
                            && text.contains("Current allocation state:")
                            && !text.contains("Current attention guidance:")
                )
        )));
        assert!(!second_items.iter().any(|item| matches!(
            item,
            ModelInputItem::Message {
                role: InputMessageRole::Developer,
                ..
            }
        )));
        assert!(
            !message_texts(second_items)
                .iter()
                .any(|text| text.contains("Current memo batch") || text.contains("batch"))
        );

        let session_after_second = fixture.controller.session.input().items().to_vec();
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
        assert!(session_after_second.iter().any(|item| matches!(
            item,
            ModelInputItem::Message { content, .. }
                if matches!(content.as_slice(), [MessageContent::Text(text)] if text.contains("cognition entry A"))
        )));
        assert!(session_after_second.iter().any(|item| matches!(
            item,
            ModelInputItem::Message { content, .. }
                if matches!(content.as_slice(), [MessageContent::Text(text)] if text.contains("cognition entry B"))
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
            ModelInputItem::Message {
                role: InputMessageRole::User,
                content,
            }
                if matches!(
                    content.as_slice(),
                    [MessageContent::Text(text)]
                        if text.contains("Allocation context for assigning the next activation priorities")
                )
        )));
        assert!(!session_after_second.iter().any(|item| matches!(
            item,
            ModelInputItem::Message { content, .. }
                if matches!(
                    content.as_slice(),
                    [MessageContent::Text(text)] if text.contains("Current memo batch") || text.contains("batch")
                )
        )));
        assert_eq!(fixture.controller.session.list_turns().count(), 2);
    }
}
