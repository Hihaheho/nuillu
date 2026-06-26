use anyhow::{Context, Result};
use async_trait::async_trait;
use lutum::{DynamicTool, RawJson, Session, TextStepOutcomeWithTools};
use nuillu_blackboard::{
    AllocationCommand, AllocationEffectLevel, InteroceptiveMode, InteroceptivePatch,
    InteroceptiveState,
};
use nuillu_module::{
    ActionAffordanceReader, ActionAffordancesUpdatedInbox, AllocationReader, AllocationWriter,
    BlackboardReader, CognitionLogBatchFormat, CognitionLogReader, CognitionLogUpdatedInbox,
    ExternalActionInvocationResult, ExternalActionInvoker, InteroceptiveReader,
    InteroceptiveUpdatedInbox, InteroceptiveWriter, LlmAccess, LlmContextWindow, Memo,
    MemoLogBatchFormat, MemoUpdatedInbox, Module, SessionAutoCompaction, SessionCompactionConfig,
    SessionCompactionProtectedPrefix, ensure_persistent_session_seeded,
    format_bounded_cognition_log_batch_with_format, format_bounded_memo_log_batch_with_format,
    format_current_allocation_state, format_memory_trace_inventory, memory_rank_counts,
};
use nuillu_types::builtin;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

const SYSTEM_PROMPT: &str = r#"You are the action module.
You decide and execute one concrete action tool for the current moment.

Your available tools are a mixed action surface:
- activate_speak: alloc-backed. Use it only when the agent should speak now. This submits your
  complete speak allocation effect for the turn.
- sleep: built-in. Use it only when the body is sleep-ready; it directly changes interoception so
  homeostasis can react through the interoceptive update.
- hold_actions: built-in. Use it when no concrete action should run now. This clears your previous
  speak allocation effect.
- dynamic external tools: host-owned affordances from the latest visualizer/runtime snapshot. Their
  descriptions tell you what they do and when to use them. External action results are execution
  acknowledgements only; do not treat the tool result as semantic observation. The host will publish
  real outcomes separately as sensory input or scene changes.

Use current memos, current cognition entries, current interoception, current action allocation
state, and the live affordance list. Use exactly one action tool per activation. Each static tool
carries a free-form memo; preserve the reasoning needed by future action turns but do not encode it
as JSON, YAML, a code block, or any fixed schema."#;

const COMPACTED_ACTION_SESSION_PREFIX: &str = "Compacted action session history:";
const SESSION_COMPACTION_FOCUS: &str = "Preserve action choices, action notes, and relevant current context for future action execution.";
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

pub const SLEEP_WAKE_AROUSAL_THRESHOLD: f32 = 0.25;
pub const SLEEP_NREM_PRESSURE_THRESHOLD: f32 = 0.80;
pub const SLEEP_REM_PRESSURE_THRESHOLD: f32 = 0.80;
pub const SLEEP_WAKE_AROUSAL: f32 = 0.05;
pub const SLEEP_NREM_PRESSURE: f32 = 0.85;
pub const SLEEP_REM_PRESSURE: f32 = 0.45;

pub fn session_auto_compaction() -> SessionAutoCompaction {
    SessionAutoCompaction::new(
        SessionCompactionConfig::default(),
        SessionCompactionProtectedPrefix::LeadingSystemAndIdentitySeed,
        COMPACTED_ACTION_SESSION_PREFIX,
        SESSION_COMPACTION_FOCUS,
    )
}

#[lutum::tool_input(name = "activate_speak", output = ActionToolOutput)]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct ActivateSpeakArgs {
    pub memo: String,
}

#[lutum::tool_input(name = "hold_actions", output = ActionToolOutput)]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct HoldActionsArgs {
    pub memo: String,
}

#[lutum::tool_input(name = "sleep", output = ActionToolOutput)]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct SleepActionArgs {
    pub memo: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct ActionToolOutput {
    pub accepted: bool,
    pub message: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema, lutum::Toolset)]
pub enum ActionTools {
    ActivateSpeak(ActivateSpeakArgs),
    HoldActions(HoldActionsArgs),
    Sleep(SleepActionArgs),
    #[dynamic]
    External(DynamicTool),
}

pub struct ActionModule {
    memo_updates: MemoUpdatedInbox,
    cognition_updates: CognitionLogUpdatedInbox,
    interoception_updates: InteroceptiveUpdatedInbox,
    action_affordance_updates: ActionAffordancesUpdatedInbox,
    blackboard: BlackboardReader,
    cognition_log: CognitionLogReader,
    allocation_reader: AllocationReader,
    interoception: InteroceptiveReader,
    action_affordances: ActionAffordanceReader,
    external_actions: ExternalActionInvoker,
    allocation_writer: AllocationWriter,
    interoception_writer: InteroceptiveWriter,
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
        action_affordance_updates: ActionAffordancesUpdatedInbox,
        blackboard: BlackboardReader,
        cognition_log: CognitionLogReader,
        allocation_reader: AllocationReader,
        interoception: InteroceptiveReader,
        action_affordances: ActionAffordanceReader,
        external_actions: ExternalActionInvoker,
        allocation_writer: AllocationWriter,
        interoception_writer: InteroceptiveWriter,
        memo: Memo,
        llm: LlmAccess,
        session: Session,
    ) -> Self {
        Self {
            memo_updates,
            cognition_updates,
            interoception_updates,
            action_affordance_updates,
            blackboard,
            cognition_log,
            allocation_reader,
            interoception,
            action_affordances,
            external_actions,
            allocation_writer,
            interoception_writer,
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
        let action_snapshot = self.action_affordances.snapshot();
        let dynamic_tools = action_snapshot
            .affordances
            .iter()
            .map(|affordance| {
                DynamicTool::new(
                    affordance.id.clone(),
                    affordance.tool_description(),
                    affordance.input_schema.clone(),
                )
            })
            .collect::<Vec<_>>();
        let (rank_counts, current, interoception) = {
            let rank_counts = self
                .blackboard
                .read(|bb| memory_rank_counts(bb.memory_metadata()))
                .await;
            let current = self.allocation_reader.snapshot().await;
            let interoception = self.interoception.snapshot().await;
            (rank_counts, current, interoception)
        };

        let mut visible_current = current.clone();
        visible_current.retain_modules(&[builtin::speak()].into_iter().collect());

        let lutum = self.llm.lutum().await;
        let outcome = {
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
                action_snapshot.version,
                &action_snapshot.affordances,
            ));
            self.session
                .text_turn()
                .tools::<ActionTools>()
                .with_dynamic_tools(dynamic_tools)
                .require_any_tool()
                .max_output_tokens(TOOL_TURN_MAX_OUTPUT_TOKENS)
                .collect_controlled_with(
                    &lutum,
                    nuillu_module::AbortOnAvailableToolNameInText::new(),
                )
                .await
        }
        .context("action tool turn failed")?;

        match outcome {
            TextStepOutcomeWithTools::Finished(result) => {
                cx.compact_and_save(&mut self.session, result.usage).await?;
                anyhow::bail!("action finished without required action tool call");
            }
            TextStepOutcomeWithTools::FinishedNoOutput(result) => {
                cx.compact_and_save(&mut self.session, result.usage).await?;
                anyhow::bail!("action finished without required action tool call");
            }
            TextStepOutcomeWithTools::NeedsTools(round) => {
                let usage = round.usage;
                let mut handled = Vec::new();
                let mut executed = false;
                nuillu_module::emit_trace_tool_calls(&round.tool_calls);
                for call in round.tool_calls.iter().cloned() {
                    match call {
                        ActionToolsCall::ActivateSpeak(call) => {
                            let input = call.input.clone();
                            let output = if executed {
                                duplicate_action_output()
                            } else {
                                executed = true;
                                self.activate_speak(input.memo).await?
                            };
                            handled.push(ActionToolsHandled::from(call.handled(output)));
                        }
                        ActionToolsCall::HoldActions(call) => {
                            let input = call.input.clone();
                            let output = if executed {
                                duplicate_action_output()
                            } else {
                                executed = true;
                                self.hold_actions(input.memo).await?
                            };
                            handled.push(ActionToolsHandled::from(call.handled(output)));
                        }
                        ActionToolsCall::Sleep(call) => {
                            let input = call.input.clone();
                            let output = if executed {
                                duplicate_action_output()
                            } else {
                                executed = true;
                                self.sleep(input.memo, &interoception).await?
                            };
                            handled.push(ActionToolsHandled::from(call.handled(output)));
                        }
                        ActionToolsCall::External(call) => {
                            let output = if executed {
                                duplicate_action_output()
                            } else {
                                executed = true;
                                self.invoke_external_action(
                                    call.name().to_owned(),
                                    call.arguments().deserialize().with_context(|| {
                                        format!("decode external action {} arguments", call.name())
                                    })?,
                                )
                                .await?
                            };
                            handled.push(ActionToolsHandled::from(
                                call.handled(RawJson::from_serializable(&output)?),
                            ));
                        }
                    }
                }
                if !executed {
                    anyhow::bail!("action tool turn produced no executable action");
                }
                round
                    .commit(&mut self.session, handled)
                    .context("commit action tool round")?;
                cx.compact_and_save(&mut self.session, usage).await?;
            }
        }
        Ok(())
    }

    async fn activate_speak(&mut self, memo: String) -> Result<ActionToolOutput> {
        self.allocation_writer
            .submit(vec![AllocationCommand::target(
                builtin::speak(),
                AllocationEffectLevel::Max,
            )])
            .await
            .context("persist action speak allocation")?;
        self.write_memo(memo).await;
        Ok(ActionToolOutput {
            accepted: true,
            message: "speak allocation activated".to_owned(),
        })
    }

    async fn hold_actions(&mut self, memo: String) -> Result<ActionToolOutput> {
        self.clear_speak_allocation().await?;
        self.write_memo(memo).await;
        Ok(ActionToolOutput {
            accepted: true,
            message: "action allocation cleared".to_owned(),
        })
    }

    async fn sleep(
        &mut self,
        memo: String,
        interoception: &InteroceptiveState,
    ) -> Result<ActionToolOutput> {
        self.clear_speak_allocation().await?;
        if !sleep_is_appropriate(interoception) {
            self.write_memo(memo).await;
            return Ok(ActionToolOutput {
                accepted: false,
                message: "sleep rejected because current interoception is not sleep-ready"
                    .to_owned(),
            });
        }
        self.interoception_writer
            .update(sleep_interoceptive_patch())
            .await;
        self.write_memo(memo).await;
        Ok(ActionToolOutput {
            accepted: true,
            message: "sleep interoception patch applied".to_owned(),
        })
    }

    async fn invoke_external_action(
        &mut self,
        action_id: String,
        arguments: serde_json::Value,
    ) -> Result<ActionToolOutput> {
        self.clear_speak_allocation().await?;
        let result = match self.external_actions.invoke(action_id, arguments).await {
            Ok(result) => result,
            Err(error) => ExternalActionInvocationResult {
                accepted: false,
                message: format!("external action failed: {error}"),
            },
        };
        Ok(action_result_to_tool_output(result))
    }

    async fn clear_speak_allocation(&self) -> Result<()> {
        self.allocation_writer
            .submit(Vec::<AllocationCommand>::new())
            .await
            .context("clear action speak allocation")
    }

    async fn write_memo(&self, memo: String) {
        if !memo.trim().is_empty() {
            self.memo.write(memo).await;
        }
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
            update = self.action_affordance_updates.next_item() => {
                let _ = update?;
            }
        }
        let _ = self.memo_updates.take_ready_items()?;
        let _ = self.cognition_updates.take_ready_items()?;
        let _ = self.interoception_updates.take_ready_items()?;
        let _ = self.action_affordance_updates.take_ready_items()?;
        Ok(())
    }
}

fn duplicate_action_output() -> ActionToolOutput {
    ActionToolOutput {
        accepted: false,
        message: "ignored because an action already executed this activation".to_owned(),
    }
}

fn action_result_to_tool_output(result: ExternalActionInvocationResult) -> ActionToolOutput {
    ActionToolOutput {
        accepted: result.accepted,
        message: result.message,
    }
}

fn sleep_is_appropriate(state: &InteroceptiveState) -> bool {
    matches!(state.mode, InteroceptiveMode::Wake)
        && (state.wake_arousal <= SLEEP_WAKE_AROUSAL_THRESHOLD
            || state.nrem_pressure >= SLEEP_NREM_PRESSURE_THRESHOLD
            || state.rem_pressure >= SLEEP_REM_PRESSURE_THRESHOLD)
}

fn sleep_interoceptive_patch() -> InteroceptivePatch {
    InteroceptivePatch {
        mode: Some(InteroceptiveMode::NremPressure),
        wake_arousal: Some(SLEEP_WAKE_AROUSAL),
        nrem_pressure: Some(SLEEP_NREM_PRESSURE),
        rem_pressure: Some(SLEEP_REM_PRESSURE),
        affect_arousal: None,
        valence: None,
        emotion: Some("sleeping".to_owned()),
    }
}

fn format_action_context(
    rank_counts: &nuillu_module::MemoryRankCounts,
    current: &nuillu_module::ResourceAllocation,
    interoception: &InteroceptiveState,
    action_affordance_version: u64,
    affordances: &[nuillu_module::ActionAffordance],
) -> String {
    let mut sections = vec!["Action context for executing one concrete action:".to_owned()];
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
    sections.push("Built-in actions:\n- activate_speak: use for an outward utterance; effect activates only the speak module.\n- sleep: use only when interoception is sleep-ready; effect patches interoception into sleep pressure.\n- hold_actions: use when no action should run; effect clears Action's previous speak allocation.".to_owned());
    sections.push(format_external_affordances(
        action_affordance_version,
        affordances,
    ));
    sections.join("\n\n")
}

fn format_external_affordances(
    version: u64,
    affordances: &[nuillu_module::ActionAffordance],
) -> String {
    if affordances.is_empty() {
        return format!("External action affordances (version {version}): none");
    }
    let mut out = format!("External action affordances (version {version}):");
    for affordance in affordances {
        out.push_str("\n- ");
        out.push_str(&affordance.id);
        if !affordance.label.trim().is_empty() {
            out.push_str(": ");
            out.push_str(affordance.label.trim());
        }
        if !affordance.use_when.trim().is_empty() {
            out.push_str("\n  use_when: ");
            out.push_str(affordance.use_when.trim());
        }
        if !affordance.effect.trim().is_empty() {
            out.push_str("\n  effect: ");
            out.push_str(affordance.effect.trim());
        }
    }
    out
}

fn action_observation_input(memos: Option<String>, cognition: Option<String>) -> Option<String> {
    let sections = [memos, cognition].into_iter().flatten().collect::<Vec<_>>();
    if sections.is_empty() {
        None
    } else {
        Some(sections.join("\n\n"))
    }
}

#[async_trait(?Send)]
impl Module for ActionModule {
    type Batch = ();

    fn id() -> &'static str {
        "action"
    }

    fn peer_context() -> Option<&'static str> {
        Some(
            "Action executes concrete action tools: speak allocation, built-in sleep, and host-owned external affordances.",
        )
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

    fn interoception(
        mode: InteroceptiveMode,
        wake_arousal: f32,
        nrem_pressure: f32,
        rem_pressure: f32,
    ) -> InteroceptiveState {
        InteroceptiveState {
            mode,
            wake_arousal,
            nrem_pressure,
            rem_pressure,
            affect_arousal: 0.1,
            valence: 0.0,
            emotion: "calm".to_owned(),
            ..InteroceptiveState::default()
        }
    }

    #[test]
    fn sleep_requires_wake_mode_and_sleep_ready_interoception() {
        assert!(sleep_is_appropriate(&interoception(
            InteroceptiveMode::Wake,
            SLEEP_WAKE_AROUSAL_THRESHOLD,
            0.1,
            0.1
        )));
        assert!(sleep_is_appropriate(&interoception(
            InteroceptiveMode::Wake,
            0.6,
            SLEEP_NREM_PRESSURE_THRESHOLD,
            0.1
        )));
        assert!(!sleep_is_appropriate(&interoception(
            InteroceptiveMode::Wake,
            0.6,
            0.1,
            0.1
        )));
        assert!(!sleep_is_appropriate(&interoception(
            InteroceptiveMode::NremPressure,
            0.0,
            1.0,
            0.0
        )));
    }

    #[test]
    fn sleep_patch_moves_interoception_to_nrem() {
        assert_eq!(
            sleep_interoceptive_patch(),
            InteroceptivePatch {
                mode: Some(InteroceptiveMode::NremPressure),
                wake_arousal: Some(SLEEP_WAKE_AROUSAL),
                nrem_pressure: Some(SLEEP_NREM_PRESSURE),
                rem_pressure: Some(SLEEP_REM_PRESSURE),
                affect_arousal: None,
                valence: None,
                emotion: Some("sleeping".to_owned()),
            }
        );
    }

    #[test]
    fn external_action_result_becomes_ack_only_tool_output() {
        assert_eq!(
            action_result_to_tool_output(ExternalActionInvocationResult {
                accepted: true,
                message: "accepted".to_owned(),
            }),
            ActionToolOutput {
                accepted: true,
                message: "accepted".to_owned(),
            }
        );
    }
}
