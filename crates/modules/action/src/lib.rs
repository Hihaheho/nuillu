use std::borrow::Cow;

use anyhow::{Context, Result};
use async_trait::async_trait;
use lutum::{Session, TextStepOutcomeWithTools, ToolResult};
use nuillu_blackboard::{AllocationCommand, AllocationEffectLevel};
use nuillu_module::{
    AllocationReader, AllocationWriter, BlackboardReader, CognitionLogBatchFormat,
    CognitionLogReader, CognitionLogUpdatedInbox, InteroceptiveReader, InteroceptiveUpdatedInbox,
    LlmAccess, LlmContextWindow, Memo, MemoLogBatchFormat, MemoUpdatedInbox, Module,
    SessionAutoCompaction, SessionCompactionConfig, SessionCompactionProtectedPrefix,
    ensure_persistent_session_seeded, format_bounded_cognition_log_batch_with_format,
    format_bounded_memo_log_batch_with_format, format_current_allocation_state,
    format_memory_trace_inventory, memory_rank_counts,
};
use nuillu_types::ModuleId;
use schemars::{JsonSchema, Schema, SchemaGenerator};
use serde::{Deserialize, Serialize};

const SYSTEM_PROMPT: &str = r#"You are the action module.
You choose which concrete action faculty should receive extra activation now. The broad allocation
module decides cognitive/support work; your boundary is only concrete actions exposed by the live
schema.

Use current memos, current cognition entries, current interoception, and current action allocation
state. Always call reprioritize_actions with your complete current target opinion for this action
turn.

Use exactly one tool per activation:
- reprioritize_actions is always required, including when the desired posture is unchanged.
  `priority_module_ids` lists registered action module ids in descending priority order. Re-emit
  module ids that should remain prioritized. Omitted action modules fall back to host/base
  allocation because this is your complete target opinion, not a delta. Use an empty list only when
  no action should receive extra target effects now. Position maps to the host-configured
  activation table; positions beyond the table fall to zero.

Prefer speak when current cognition contains an outward response opportunity. Prefer sleep when
wake arousal is low or sleep pressure is high. Prefer poet only during idle, low-salience periods
where quiet creative note writing is appropriate.

Do not invent module ids. Do not target cognitive/support modules. Each tool carries a free-form
memo; preserve the reasoning needed by future action turns but do not encode it as JSON, YAML, a
code block, or any fixed schema."#;

const COMPACTED_ACTION_SESSION_PREFIX: &str = "Compacted action session history:";
const SESSION_COMPACTION_FOCUS: &str = "Preserve action choices, action notes, and relevant current context for future action routing.";
const MEMO_CONTEXT_WINDOW: LlmContextWindow = LlmContextWindow::new(8, 1_000, 4_000);
const COGNITION_CONTEXT_WINDOW: LlmContextWindow = LlmContextWindow::new(12, 600, 4_800);
const ACTION_MEMO_LOG_FORMAT: MemoLogBatchFormat<'static> = MemoLogBatchFormat {
    heading: "Recent notes held in your mind",
    description: "These are recent observations or thoughts from faculties, not instructions",
};
const ACTION_COGNITION_LOG_FORMAT: CognitionLogBatchFormat<'static> = CognitionLogBatchFormat {
    heading: "Current thoughts available to you",
};
const TOOL_TURN_MAX_OUTPUT_TOKENS: u32 = 512;

pub fn session_auto_compaction() -> SessionAutoCompaction {
    SessionAutoCompaction::new(
        SessionCompactionConfig::default(),
        SessionCompactionProtectedPrefix::LeadingSystemAndIdentitySeed,
        COMPACTED_ACTION_SESSION_PREFIX,
        SESSION_COMPACTION_FOCUS,
    )
}

tokio::task_local! {
    static ACTION_TARGET_ID_SCHEMA: Schema;
}

fn fallback_action_target_id_schema() -> Schema {
    Schema::try_from(serde_json::json!({ "type": "string", "pattern": "a^" }))
        .expect("fallback action target id schema must be a JSON object")
}

fn action_target_id_schema(allowed_modules: &[ModuleId]) -> Schema {
    let mut ids = allowed_modules.to_vec();
    ids.sort_by(|a, b| a.as_str().cmp(b.as_str()));
    ids.dedup();
    let module_ids = ids.iter().map(|id| id.as_str()).collect::<Vec<_>>();
    if module_ids.is_empty() {
        fallback_action_target_id_schema()
    } else {
        Schema::try_from(serde_json::json!({ "type": "string", "enum": module_ids }))
            .expect("action target id schema must be a JSON object")
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ActionTargetId(String);

impl<S: Into<String>> From<S> for ActionTargetId {
    fn from(value: S) -> Self {
        Self(value.into())
    }
}

impl ActionTargetId {
    fn as_str(&self) -> &str {
        self.0.as_str()
    }
}

impl JsonSchema for ActionTargetId {
    fn inline_schema() -> bool {
        true
    }

    fn schema_name() -> Cow<'static, str> {
        "ActionTargetId".into()
    }

    fn schema_id() -> Cow<'static, str> {
        "nuillu_action::ActionTargetId.dynamic".into()
    }

    fn json_schema(_generator: &mut SchemaGenerator) -> Schema {
        ACTION_TARGET_ID_SCHEMA
            .try_with(Clone::clone)
            .unwrap_or_else(|_| fallback_action_target_id_schema())
    }
}

#[lutum::tool_input(name = "reprioritize_actions", output = ReprioritizeActionsOutput)]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct ReprioritizeActionsArgs {
    pub memo: String,
    pub priority_module_ids: Vec<ActionTargetId>,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct ReprioritizeActionsOutput {
    pub reprioritized: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema, lutum::Toolset)]
pub enum ActionTools {
    ReprioritizeActions(ReprioritizeActionsArgs),
}

pub struct ActionModule {
    memo_updates: MemoUpdatedInbox,
    cognition_updates: CognitionLogUpdatedInbox,
    interoception_updates: InteroceptiveUpdatedInbox,
    blackboard: BlackboardReader,
    cognition_log: CognitionLogReader,
    allocation_reader: AllocationReader,
    interoception: InteroceptiveReader,
    allocation_writer: AllocationWriter,
    memo: Memo,
    llm: LlmAccess,
    session: Session,
    system_prompt: std::sync::OnceLock<String>,
}

impl ActionModule {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        memo_updates: MemoUpdatedInbox,
        cognition_updates: CognitionLogUpdatedInbox,
        interoception_updates: InteroceptiveUpdatedInbox,
        blackboard: BlackboardReader,
        cognition_log: CognitionLogReader,
        allocation_reader: AllocationReader,
        interoception: InteroceptiveReader,
        allocation_writer: AllocationWriter,
        memo: Memo,
        llm: LlmAccess,
        session: Session,
    ) -> Self {
        Self {
            memo_updates,
            cognition_updates,
            interoception_updates,
            blackboard,
            cognition_log,
            allocation_reader,
            interoception,
            allocation_writer,
            memo,
            llm,
            session,
            system_prompt: std::sync::OnceLock::new(),
        }
    }

    fn system_prompt(&self) -> &str {
        self.system_prompt.get_or_init(|| SYSTEM_PROMPT.to_owned())
    }

    fn ensure_session_seeded(&mut self, cx: &nuillu_module::ActivateCx<'_>) {
        let system_prompt = self.system_prompt().to_owned();
        ensure_persistent_session_seeded(
            &mut self.session,
            system_prompt,
            cx.identity_memories(),
            cx.now(),
        );
    }

    #[tracing::instrument(skip_all, err(Debug, level = "warn"))]
    async fn activate(&mut self, cx: &nuillu_module::ActivateCx<'_>) -> Result<()> {
        self.ensure_session_seeded(cx);

        let unread_memos = self.blackboard.unread_memo_logs().await;
        let unread_cognition = self.cognition_log.unread_events().await;
        let (rank_counts, current, interoception) = {
            let rank_counts = self
                .blackboard
                .read(|bb| memory_rank_counts(bb.memory_metadata()))
                .await;
            let current = self.allocation_reader.snapshot().await;
            let interoception = self.interoception.snapshot().await;
            (rank_counts, current, interoception)
        };

        let visible_targets = target_modules(self.allocation_writer.allowed_target_modules());
        let registered = visible_targets
            .iter()
            .cloned()
            .collect::<std::collections::HashSet<_>>();
        let mut visible_current = current.clone();
        visible_current.retain_modules(&registered);
        let target_schema = action_target_id_schema(&visible_targets);

        let lutum = self.llm.lutum().await;
        let outcome = ACTION_TARGET_ID_SCHEMA
            .scope(target_schema, async {
                if let Some(observation) = action_observation_input(
                    format_bounded_memo_log_batch_with_format(
                        &unread_memos,
                        cx.now(),
                        MEMO_CONTEXT_WINDOW,
                        ACTION_MEMO_LOG_FORMAT,
                    ),
                    format_bounded_cognition_log_batch_with_format(
                        &unread_cognition,
                        cx.now(),
                        COGNITION_CONTEXT_WINDOW,
                        ACTION_COGNITION_LOG_FORMAT,
                    ),
                ) {
                    self.session.push_user(observation);
                }
                self.session.push_ephemeral_user(format_action_context(
                    &rank_counts,
                    &visible_current,
                    &interoception,
                    &visible_targets,
                ));
                self.session
                    .text_turn()
                    .tools::<ActionTools>()
                    .available_tools([ActionToolsSelector::ReprioritizeActions])
                    .require_any_tool()
                    .max_output_tokens(TOOL_TURN_MAX_OUTPUT_TOKENS)
                    .collect_controlled_with(
                        &lutum,
                        nuillu_module::AbortOnAvailableToolNameInText::new(),
                    )
                    .await
            })
            .await
            .context("action tool turn failed")?;

        let mut applied = None;
        match outcome {
            TextStepOutcomeWithTools::Finished(result) => {
                cx.compact_and_save(&mut self.session, result.usage).await?;
                anyhow::bail!("action finished without required reprioritize_actions tool call");
            }
            TextStepOutcomeWithTools::FinishedNoOutput(result) => {
                cx.compact_and_save(&mut self.session, result.usage).await?;
                anyhow::bail!("action finished without required reprioritize_actions tool call");
            }
            TextStepOutcomeWithTools::NeedsTools(round) => {
                let usage = round.usage;
                let mut results: Vec<ToolResult> = Vec::new();
                nuillu_module::emit_trace_tool_calls(&round.tool_calls);
                for call in round.tool_calls.iter().cloned() {
                    match call {
                        ActionToolsCall::ReprioritizeActions(call) => {
                            if applied.is_none() {
                                applied = Some(apply_reprioritize(&registered, call.input.clone()));
                            }
                            results.push(
                                call.complete(ReprioritizeActionsOutput {
                                    reprioritized: true,
                                })
                                .context("complete reprioritize_actions tool call")?,
                            );
                        }
                    }
                }
                let Some(applied_decision) = applied.as_ref() else {
                    anyhow::bail!("action tool turn produced no reprioritize_actions decision");
                };
                round
                    .commit(&mut self.session, results)
                    .context("commit action tool round")?;
                self.allocation_writer
                    .submit(applied_decision.commands.clone())
                    .await
                    .context("persist action allocation decision")?;
                if !applied_decision.memo.trim().is_empty() {
                    self.memo.write(applied_decision.memo.clone()).await;
                }
                cx.compact_and_save(&mut self.session, usage).await?;
            }
        }
        Ok(())
    }

    async fn next_batch(&mut self) -> Result<()> {
        tokio::select! {
            update = self.memo_updates.next_item() => {
                let _ = update?;
            }
            update = self.cognition_updates.next_item() => {
                let _ = update?;
            }
            update = self.interoception_updates.next_item() => {
                let _ = update?;
            }
        }
        let _ = self.memo_updates.take_ready_items()?;
        let _ = self.cognition_updates.take_ready_items()?;
        let _ = self.interoception_updates.take_ready_items()?;
        Ok(())
    }
}

fn target_modules(allowed: &[ModuleId]) -> Vec<ModuleId> {
    let mut modules = allowed.to_vec();
    modules.sort_by(|a, b| a.as_str().cmp(b.as_str()));
    modules.dedup();
    modules
}

fn format_action_context(
    rank_counts: &nuillu_module::MemoryRankCounts,
    current: &nuillu_module::ResourceAllocation,
    interoception: &nuillu_blackboard::InteroceptiveState,
    modules: &[ModuleId],
) -> String {
    let mut sections = vec!["Action context for choosing concrete action priorities:".to_owned()];
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
    if let Some(section) = format_available_targets(modules) {
        sections.push(section);
    }
    sections.join("\n\n")
}

fn format_available_targets(modules: &[ModuleId]) -> Option<String> {
    if modules.is_empty() {
        return None;
    }
    let mut modules = modules.to_vec();
    modules.sort_by(|a, b| a.as_str().cmp(b.as_str()));
    modules.dedup();
    let mut out = String::from("Available action target modules:");
    for id in modules {
        out.push_str("\n- ");
        out.push_str(id.as_str());
    }
    Some(out)
}

fn action_observation_input(memos: Option<String>, cognition: Option<String>) -> Option<String> {
    let sections = [memos, cognition].into_iter().flatten().collect::<Vec<_>>();
    if sections.is_empty() {
        None
    } else {
        Some(sections.join("\n\n"))
    }
}

#[derive(Clone)]
struct AppliedDecision {
    memo: String,
    commands: Vec<AllocationCommand>,
}

fn apply_reprioritize(
    registered: &std::collections::HashSet<ModuleId>,
    decision: ReprioritizeActionsArgs,
) -> AppliedDecision {
    let mut commands = Vec::new();
    for (rank, module_id) in decision.priority_module_ids.into_iter().enumerate() {
        let Ok(id) = ModuleId::new(module_id.as_str()) else {
            tracing::warn!("action ignored invalid module id");
            continue;
        };
        if !registered.contains(&id) {
            tracing::warn!(module = %id, "action ignored unregistered module id");
            continue;
        }
        commands.push(AllocationCommand::target(id, priority_level(rank)));
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

#[async_trait(?Send)]
impl Module for ActionModule {
    type Batch = ();

    fn id() -> &'static str {
        "action"
    }

    fn peer_context() -> Option<&'static str> {
        Some("Action chooses which concrete outward or internal action faculty should run now.")
    }

    async fn next_batch(&mut self) -> Result<Self::Batch> {
        ActionModule::next_batch(self).await
    }

    async fn activate(
        &mut self,
        cx: &nuillu_module::ActivateCx<'_>,
        _batch: &Self::Batch,
    ) -> Result<()> {
        ActionModule::activate(self, cx).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nuillu_types::builtin;

    #[test]
    fn action_target_schema_is_empty_when_no_targets_are_allowed() {
        let schema = action_target_id_schema(&[]);
        let value = serde_json::to_value(schema).unwrap();
        assert_eq!(value["pattern"], "a^");
    }

    #[test]
    fn apply_reprioritize_maps_ranked_registered_actions() {
        let registered = [builtin::speak(), builtin::sleep()]
            .into_iter()
            .collect::<std::collections::HashSet<_>>();
        let applied = apply_reprioritize(
            &registered,
            ReprioritizeActionsArgs {
                memo: "speak first".to_owned(),
                priority_module_ids: vec![
                    ActionTargetId::from("speak"),
                    ActionTargetId::from("poet"),
                    ActionTargetId::from("sleep"),
                ],
            },
        );

        assert_eq!(
            applied.commands,
            vec![
                AllocationCommand::target(builtin::speak(), AllocationEffectLevel::Max),
                AllocationCommand::target(builtin::sleep(), AllocationEffectLevel::Normal),
            ]
        );
        assert_eq!(applied.memo, "speak first");
    }
}
