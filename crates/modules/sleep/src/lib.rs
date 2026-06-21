use std::time::Duration;

use anyhow::{Context, Result};
use async_trait::async_trait;
use lutum::{Session, TextStepOutcomeWithTools, ToolResult};
use nuillu_blackboard::{InteroceptiveMode, InteroceptivePatch, InteroceptiveState};
use nuillu_module::{
    InteroceptiveReader, InteroceptiveUpdatedInbox, InteroceptiveWriter, LlmAccess, Memo, Module,
    SessionAutoCompaction, SessionCompactionConfig, SessionCompactionProtectedPrefix,
    ensure_persistent_session_seeded,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

const SYSTEM_PROMPT: &str = r#"You are the sleep module.
When wake arousal is already low, decide whether the agent should sleep now. Sleep is an internal
body-state transition, not an outward utterance. Use the current interoception only. If sleeping is
appropriate, call decide_sleep with should_sleep=true. If it is not appropriate, call decide_sleep
with should_sleep=false and a short memo."#;

const COMPACTED_SLEEP_SESSION_PREFIX: &str = "Compacted sleep session history:";
const SESSION_COMPACTION_FOCUS: &str =
    "Preserve prior sleep decisions and the interoceptive rationale for future sleep choices.";
const PERIODIC_WAKEUP: Duration = Duration::from_secs(1);
pub const SLEEP_WAKE_AROUSAL_THRESHOLD: f32 = 0.25;
const SLEEP_WAKE_AROUSAL: f32 = 0.05;
const SLEEP_NREM_PRESSURE: f32 = 0.85;
const SLEEP_REM_PRESSURE: f32 = 0.45;

pub fn session_auto_compaction() -> SessionAutoCompaction {
    SessionAutoCompaction::new(
        SessionCompactionConfig::default(),
        SessionCompactionProtectedPrefix::LeadingSystemAndIdentitySeed,
        COMPACTED_SLEEP_SESSION_PREFIX,
        SESSION_COMPACTION_FOCUS,
    )
}

#[lutum::tool_input(name = "decide_sleep", output = DecideSleepOutput)]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct DecideSleepArgs {
    pub should_sleep: bool,
    pub memo: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct DecideSleepOutput {
    pub accepted: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema, lutum::Toolset)]
pub enum SleepTools {
    DecideSleep(DecideSleepArgs),
}

pub struct SleepModule {
    updates: InteroceptiveUpdatedInbox,
    interoception: InteroceptiveReader,
    writer: InteroceptiveWriter,
    memo: Memo,
    llm: LlmAccess,
    session: Session,
    system_prompt: std::sync::OnceLock<String>,
}

impl SleepModule {
    pub fn new(
        updates: InteroceptiveUpdatedInbox,
        interoception: InteroceptiveReader,
        writer: InteroceptiveWriter,
        memo: Memo,
        llm: LlmAccess,
        session: Session,
    ) -> Self {
        Self {
            updates,
            interoception,
            writer,
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
        let state = self.interoception.snapshot().await;
        if !should_ask_sleep(&state) {
            return Ok(());
        }

        self.ensure_session_seeded(cx);
        let lutum = self.llm.lutum().await;
        self.session.push_user(format_sleep_decision_input(&state));
        let outcome = self
            .session
            .text_turn()
            .tools::<SleepTools>()
            .available_tools([SleepToolsSelector::DecideSleep])
            .require_any_tool()
            .max_output_tokens(256)
            .collect_controlled_with(&lutum, nuillu_module::AbortOnAvailableToolNameInText::new())
            .await
            .context("sleep decision tool turn failed")?;

        let mut decision = None;
        match outcome {
            TextStepOutcomeWithTools::Finished(result) => {
                cx.compact_and_save(&mut self.session, result.usage).await?;
                anyhow::bail!("sleep finished without required decide_sleep tool call");
            }
            TextStepOutcomeWithTools::FinishedNoOutput(result) => {
                cx.compact_and_save(&mut self.session, result.usage).await?;
                anyhow::bail!("sleep finished without required decide_sleep tool call");
            }
            TextStepOutcomeWithTools::NeedsTools(round) => {
                let usage = round.usage;
                let mut results: Vec<ToolResult> = Vec::new();
                nuillu_module::emit_trace_tool_calls(&round.tool_calls);
                for call in round.tool_calls.iter().cloned() {
                    match call {
                        SleepToolsCall::DecideSleep(call) => {
                            if decision.is_none() {
                                decision = Some(call.input.clone());
                            }
                            results.push(
                                call.complete(DecideSleepOutput { accepted: true })
                                    .context("complete decide_sleep tool call")?,
                            );
                        }
                    }
                }
                round
                    .commit(&mut self.session, results)
                    .context("commit sleep tool round")?;
                cx.compact_and_save(&mut self.session, usage).await?;
            }
        }

        let Some(decision) = decision else {
            anyhow::bail!("sleep tool turn produced no decide_sleep decision");
        };
        if decision.should_sleep {
            self.writer.update(sleep_patch(&state)).await;
        }
        if !decision.memo.trim().is_empty() {
            self.memo.write(decision.memo).await;
        }
        Ok(())
    }

    async fn next_batch(&mut self) -> Result<()> {
        tokio::select! {
            update = self.updates.next_item() => {
                let _ = update?;
            }
            _ = tokio::time::sleep(PERIODIC_WAKEUP) => {}
        }
        let _ = self.updates.take_ready_items()?;
        Ok(())
    }
}

fn should_ask_sleep(state: &InteroceptiveState) -> bool {
    state.wake_arousal <= SLEEP_WAKE_AROUSAL_THRESHOLD
        && matches!(state.mode, InteroceptiveMode::Wake)
}

fn sleep_patch(state: &InteroceptiveState) -> InteroceptivePatch {
    InteroceptivePatch {
        mode: Some(InteroceptiveMode::NremPressure),
        wake_arousal: Some(SLEEP_WAKE_AROUSAL),
        nrem_pressure: Some(state.nrem_pressure.max(SLEEP_NREM_PRESSURE)),
        rem_pressure: Some(state.rem_pressure.max(SLEEP_REM_PRESSURE)),
        affect_arousal: None,
        valence: None,
        emotion: None,
    }
}

fn format_sleep_decision_input(state: &InteroceptiveState) -> String {
    format!(
        "Sleep decision request\n\nCurrent interoception:\n- mode: {:?}\n- wake_arousal: {:.2}\n- nrem_pressure: {:.2}\n- rem_pressure: {:.2}\n- affect_arousal: {:.2}\n- valence: {:.2}\n- emotion: {}\n\nCall decide_sleep exactly once.",
        state.mode,
        state.wake_arousal,
        state.nrem_pressure,
        state.rem_pressure,
        state.affect_arousal,
        state.valence,
        if state.emotion.trim().is_empty() {
            "(none)"
        } else {
            state.emotion.trim()
        }
    )
}

#[async_trait(?Send)]
impl Module for SleepModule {
    type Batch = ();

    fn id() -> &'static str {
        "sleep"
    }

    fn peer_context() -> Option<&'static str> {
        None
    }

    async fn next_batch(&mut self) -> Result<Self::Batch> {
        SleepModule::next_batch(self).await
    }

    async fn activate(
        &mut self,
        cx: &nuillu_module::ActivateCx<'_>,
        _batch: &Self::Batch,
    ) -> Result<()> {
        SleepModule::activate(self, cx).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sleep_threshold_requires_low_wake_mode() {
        assert!(should_ask_sleep(&InteroceptiveState {
            wake_arousal: 0.25,
            mode: InteroceptiveMode::Wake,
            ..InteroceptiveState::default()
        }));
        assert!(!should_ask_sleep(&InteroceptiveState {
            wake_arousal: 0.26,
            mode: InteroceptiveMode::Wake,
            ..InteroceptiveState::default()
        }));
        assert!(!should_ask_sleep(&InteroceptiveState {
            wake_arousal: 0.1,
            mode: InteroceptiveMode::NremPressure,
            ..InteroceptiveState::default()
        }));
    }

    #[test]
    fn sleep_patch_drops_wake_and_preserves_stronger_pressures() {
        let patch = sleep_patch(&InteroceptiveState {
            nrem_pressure: 0.9,
            rem_pressure: 0.2,
            ..InteroceptiveState::default()
        });

        assert_eq!(
            patch,
            InteroceptivePatch {
                mode: Some(InteroceptiveMode::NremPressure),
                wake_arousal: Some(0.05),
                nrem_pressure: Some(0.9),
                rem_pressure: Some(0.45),
                affect_arousal: None,
                valence: None,
                emotion: None,
            }
        );
    }
}
