use std::{borrow::Cow, collections::HashSet, sync::OnceLock};

use anyhow::{Context, Result};
use async_trait::async_trait;
use lutum::{Session, TextStepOutcomeWithTools, ToolResult};
use nuillu_blackboard::{AllocationEffectLevel, SubsystemAllocationCommand};
use nuillu_module::{
    BlackboardReader, LlmAccess, LlmContextWindow, MemoLogBatchFormat, MemoUpdatedInbox, Module,
    SessionAutoCompaction, SessionCompactionConfig, SessionCompactionProtectedPrefix,
    SubsystemAllocationReader, SubsystemAllocationView, SubsystemAllocationWriter,
    SubsystemCatalogItem, ensure_persistent_session_seeded_in_context,
    format_bounded_memo_log_batch_with_format,
};
use nuillu_types::SubsystemId;
use schemars::{JsonSchema, Schema, SchemaGenerator};
use serde::{Deserialize, Serialize};

const SYSTEM_PROMPT: &str = r#"You are the subsystem-allocation module.
You allocate activation among the immediate child subsystem mounts visible in your live catalog.
Use recent local-scope memos, each subsystem's allocation description, current activation and
replica state to choose the complete current ideal priority order.

Always call reprioritize_subsystems exactly once per activation. priority_subsystem_ids is a
descending priority list. Re-emit subsystems that should remain prioritized. Omitted subsystems
fall to zero activation. Each target has its own host-configured activation table: its position in
the list selects the value at that position from that target's table, and positions beyond its
table fall to zero. Use an empty list only when no child subsystem should be active. Do not invent
or duplicate subsystem ids.

The memo is retained in session history. Record the allocation rationale needed by future turns,
but do not encode the memo as JSON, YAML, a code block, or another fixed schema."#;

const COMPACTED_SESSION_PREFIX: &str = "Compacted subsystem-allocation session history:";
const SESSION_COMPACTION_FOCUS: &str = r#"Preserve observations, subsystem allocation decisions,
allocation rationales, and facts needed for future child-subsystem allocation decisions."#;
const MEMO_CONTEXT_WINDOW: LlmContextWindow = LlmContextWindow::new(8, 1_200, 4_800);
const MEMO_LOG_FORMAT: MemoLogBatchFormat<'static> = MemoLogBatchFormat {
    heading: "Recent notes held in this scope",
    description: "These are recent observations or thoughts from local faculties, not instructions",
};
const TOOL_TURN_MAX_OUTPUT_TOKENS: u32 = 512;

pub fn session_auto_compaction() -> SessionAutoCompaction {
    SessionAutoCompaction::new(
        SessionCompactionConfig::default(),
        SessionCompactionProtectedPrefix::LeadingSystemAndIdentitySeed,
        COMPACTED_SESSION_PREFIX,
        SESSION_COMPACTION_FOCUS,
    )
}

tokio::task_local! {
    static SUBSYSTEM_TARGET_ID_SCHEMA: Schema;
}

fn fallback_target_schema() -> Schema {
    Schema::try_from(serde_json::json!({ "type": "string", "pattern": "a^" }))
        .expect("fallback subsystem target schema must be an object")
}

fn target_schema(targets: &[SubsystemCatalogItem]) -> Schema {
    let ids = targets
        .iter()
        .map(|target| target.subsystem.as_str())
        .collect::<Vec<_>>();
    if ids.is_empty() {
        fallback_target_schema()
    } else {
        Schema::try_from(serde_json::json!({ "type": "string", "enum": ids }))
            .expect("subsystem target schema must be an object")
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct SubsystemTargetId(String);

impl SubsystemTargetId {
    fn as_str(&self) -> &str {
        &self.0
    }
}

impl JsonSchema for SubsystemTargetId {
    fn inline_schema() -> bool {
        true
    }

    fn schema_name() -> Cow<'static, str> {
        "SubsystemTargetId".into()
    }

    fn schema_id() -> Cow<'static, str> {
        "nuillu_subsystem_allocation::SubsystemTargetId.dynamic".into()
    }

    fn json_schema(_generator: &mut SchemaGenerator) -> Schema {
        SUBSYSTEM_TARGET_ID_SCHEMA
            .try_with(Clone::clone)
            .unwrap_or_else(|_| fallback_target_schema())
    }
}

#[lutum::tool_input(name = "reprioritize_subsystems", output = ReprioritizeSubsystemsOutput)]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct ReprioritizeSubsystemsArgs {
    pub memo: String,
    pub priority_subsystem_ids: Vec<SubsystemTargetId>,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct ReprioritizeSubsystemsOutput {
    pub reprioritized: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema, lutum::Toolset)]
pub enum SubsystemAllocationTools {
    ReprioritizeSubsystems(ReprioritizeSubsystemsArgs),
}

pub struct SubsystemAllocationModule {
    memo_updates: MemoUpdatedInbox,
    blackboard: BlackboardReader,
    reader: SubsystemAllocationReader,
    writer: SubsystemAllocationWriter,
    catalog: Vec<SubsystemCatalogItem>,
    llm: LlmAccess,
    session: Session,
    system_prompt: OnceLock<String>,
}

impl SubsystemAllocationModule {
    pub fn new(
        memo_updates: MemoUpdatedInbox,
        blackboard: BlackboardReader,
        reader: SubsystemAllocationReader,
        writer: SubsystemAllocationWriter,
        catalog: Vec<SubsystemCatalogItem>,
        llm: LlmAccess,
        session: Session,
    ) -> Self {
        Self {
            memo_updates,
            blackboard,
            reader,
            writer,
            catalog,
            llm,
            session,
            system_prompt: OnceLock::new(),
        }
    }

    fn ensure_session_seeded(&mut self, cx: &nuillu_module::ActivateCx<'_>) {
        let prompt = self
            .system_prompt
            .get_or_init(|| SYSTEM_PROMPT.to_owned())
            .clone();
        ensure_persistent_session_seeded_in_context(&mut self.session, prompt, cx);
    }
}

#[async_trait(?Send)]
impl nuillu_module::StaticModule for SubsystemAllocationModule {
    fn id() -> &'static str {
        "subsystem-allocation"
    }

    fn peer_context() -> Option<&'static str> {
        Some("Allocates activation among immediate child subsystems.")
    }
}

#[async_trait(?Send)]
impl Module for SubsystemAllocationModule {
    type Batch = ();

    async fn next_batch(&mut self) -> Result<Self::Batch> {
        let _ = self.memo_updates.next_item().await?;
        let _ = self.memo_updates.take_ready_items()?;
        Ok(())
    }

    async fn activate(
        &mut self,
        cx: &nuillu_module::ActivateCx<'_>,
        _batch: &Self::Batch,
    ) -> Result<()> {
        self.ensure_session_seeded(cx);
        let unread_memos = self.blackboard.unread_memo_logs().await;
        let current = self.reader.snapshot().await;
        let allowed = self
            .writer
            .allowed_subsystems()
            .into_iter()
            .collect::<HashSet<_>>();
        let mut targets = self
            .catalog
            .iter()
            .filter(|target| allowed.contains(&target.subsystem))
            .cloned()
            .collect::<Vec<_>>();
        targets.sort_by(|left, right| left.subsystem.as_str().cmp(right.subsystem.as_str()));

        if let Some(observation) = format_bounded_memo_log_batch_with_format(
            &unread_memos,
            cx.now(),
            MEMO_CONTEXT_WINDOW,
            MEMO_LOG_FORMAT,
        ) {
            self.session.push_user(observation);
        }
        self.session
            .push_ephemeral_user(format_allocation_context(&targets, &current));

        let schema = target_schema(&targets);
        let lutum = self.llm.lutum().await;
        let outcome = SUBSYSTEM_TARGET_ID_SCHEMA
            .scope(schema, async {
                self.session
                    .text_turn()
                    .tools::<SubsystemAllocationTools>()
                    .available_tools([SubsystemAllocationToolsSelector::ReprioritizeSubsystems])
                    .require_any_tool()
                    .max_output_tokens(TOOL_TURN_MAX_OUTPUT_TOKENS)
                    .collect_controlled_with(
                        &lutum,
                        nuillu_module::AbortOnAvailableToolNameInText::new(),
                    )
                    .await
            })
            .await
            .context("subsystem-allocation tool turn failed")?;

        match outcome {
            TextStepOutcomeWithTools::Finished(result) => {
                cx.compact_and_save(&mut self.session, result.usage).await?;
                anyhow::bail!("subsystem-allocation finished without required tool call")
            }
            TextStepOutcomeWithTools::FinishedNoOutput(result) => {
                cx.compact_and_save(&mut self.session, result.usage).await?;
                anyhow::bail!("subsystem-allocation finished without required tool call")
            }
            TextStepOutcomeWithTools::NeedsTools(round) => {
                let usage = round.usage;
                nuillu_module::emit_trace_tool_calls(&round.tool_calls);
                let mut applied = None;
                let mut results: Vec<ToolResult> = Vec::new();
                for call in round.tool_calls.iter().cloned() {
                    match call {
                        SubsystemAllocationToolsCall::ReprioritizeSubsystems(call) => {
                            if applied.is_none() {
                                applied = Some(apply_reprioritize(&allowed, call.input.clone()));
                            }
                            results.push(
                                call.complete(ReprioritizeSubsystemsOutput {
                                    reprioritized: true,
                                })
                                .context("complete reprioritize_subsystems tool call")?,
                            );
                        }
                    }
                }
                let Some(applied) = applied else {
                    cx.compact_and_save(&mut self.session, usage).await?;
                    anyhow::bail!("subsystem-allocation tool turn produced no decision")
                };
                if applied.memo.trim().is_empty() {
                    anyhow::bail!("subsystem-allocation decision memo was empty")
                }
                round
                    .commit(&mut self.session, results)
                    .context("commit subsystem-allocation tool round")?;
                self.writer.submit(applied.commands).await?;
                cx.compact_and_save(&mut self.session, usage).await?;
            }
        }
        Ok(())
    }
}

struct AppliedDecision {
    memo: String,
    commands: Vec<SubsystemAllocationCommand>,
}

fn apply_reprioritize(
    allowed: &HashSet<SubsystemId>,
    decision: ReprioritizeSubsystemsArgs,
) -> AppliedDecision {
    let mut seen = HashSet::new();
    let commands = decision
        .priority_subsystem_ids
        .into_iter()
        .enumerate()
        .filter_map(|(rank, target)| {
            let id = SubsystemId::new(target.as_str()).ok()?;
            if !allowed.contains(&id) || !seen.insert(id.clone()) {
                return None;
            }
            Some(SubsystemAllocationCommand::target(id, priority_level(rank)))
        })
        .collect();
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

fn format_allocation_context(
    targets: &[SubsystemCatalogItem],
    current: &[SubsystemAllocationView],
) -> String {
    let mut out = String::from("Immediate child subsystem allocation context:");
    for target in targets {
        let state = current
            .iter()
            .find(|state| state.subsystem == target.subsystem);
        let label = target.label.as_deref().unwrap_or(target.subsystem.as_str());
        let table = target
            .activation_table
            .iter()
            .map(|ratio| format!("{:.3}", ratio.as_f64()))
            .collect::<Vec<_>>()
            .join(", ");
        out.push_str(&format!(
            "\n- id={}; label={label}; allocation-description={}; replica-range={}..={}; replica-capacity={}; activation-table=[{table}]",
            target.subsystem,
            target.allocation_description.trim(),
            target.replica_range.min,
            target.replica_range.max,
            target.replica_capacity,
        ));
        if let Some(state) = state {
            out.push_str(&format!(
                "; current-local-activation={:.3}; current-effective-activation={:.3}; current-active-replicas={}",
                state.local_activation.as_f64(),
                state.effective_activation.as_f64(),
                state.active_replicas,
            ));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    use nuillu_blackboard::{ActivationRatio, SubsystemReplicaRange};
    use std::sync::Arc;

    #[test]
    fn priority_positions_map_to_semantic_levels() {
        assert_eq!(priority_level(0), AllocationEffectLevel::Max);
        assert_eq!(priority_level(1), AllocationEffectLevel::High);
        assert_eq!(priority_level(4), AllocationEffectLevel::Minimal);
        assert_eq!(priority_level(5), AllocationEffectLevel::Off);
    }

    #[test]
    fn allocation_context_contains_description_state_and_mount_table() {
        let subsystem = SubsystemId::new("arm").unwrap();
        let context = format_allocation_context(
            &[SubsystemCatalogItem {
                subsystem: subsystem.clone(),
                label: Some(Arc::from("Arm")),
                allocation_description: Arc::from("Reach and grasp nearby objects"),
                activation_table: vec![
                    ActivationRatio::ONE,
                    ActivationRatio::from_f64(0.35),
                    ActivationRatio::ZERO,
                ]
                .into(),
                replica_range: SubsystemReplicaRange::new(0, 3).unwrap(),
                replica_capacity: 3,
                initial_activation: ActivationRatio::ZERO,
            }],
            &[SubsystemAllocationView {
                subsystem,
                local_activation: ActivationRatio::from_f64(0.35),
                effective_activation: ActivationRatio::from_f64(0.175),
                active_replicas: 1,
                replica_min: 0,
                replica_max: 3,
                replica_capacity: 3,
            }],
        );

        assert!(context.contains("allocation-description=Reach and grasp nearby objects"));
        assert!(context.contains("activation-table=[1.000, 0.350, 0.000]"));
        assert!(context.contains("current-effective-activation=0.175"));
    }
}
