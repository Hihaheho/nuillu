use std::time::Duration;

use anyhow::Result;
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use nuillu_blackboard::{AllocationCommand, AllocationEffectLevel, InteroceptiveState};
use nuillu_module::{AllocationWriter, InteroceptiveReader, InteroceptiveUpdatedInbox, Module};
use nuillu_types::{ModuleId, builtin};

const PERIODIC_WAKEUP: Duration = Duration::from_secs(1);
const NREM_ENTER: f32 = 0.7;
const NREM_EXIT: f32 = 0.3;
const REM_EXIT: f32 = 0.2;
const FORCE_WAKE: f32 = 0.7;
const MIN_PHASE_DURATION: Duration = Duration::from_secs(2);
const MAX_NREM_DURATION: Duration = Duration::from_secs(30);
const MAX_REM_DURATION: Duration = Duration::from_secs(20);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum HomeostaticPhase {
    Wake,
    Compacting,
    Recombining,
}

pub struct HomeostaticControllerModule {
    interoception_updates: InteroceptiveUpdatedInbox,
    interoception: InteroceptiveReader,
    allocation: AllocationWriter,
    phase: HomeostaticPhase,
    last_phase_entered_at: Option<DateTime<Utc>>,
}

impl HomeostaticControllerModule {
    pub fn new(
        interoception_updates: InteroceptiveUpdatedInbox,
        interoception: InteroceptiveReader,
        allocation: AllocationWriter,
    ) -> Self {
        Self {
            interoception_updates,
            interoception,
            allocation,
            phase: HomeostaticPhase::Wake,
            last_phase_entered_at: None,
        }
    }

    async fn activate(&mut self, cx: &nuillu_module::ActivateCx<'_>) -> Result<()> {
        let interoception = self.interoception.snapshot().await;
        let entered_at = self.last_phase_entered_at.unwrap_or_else(|| cx.now());
        let next = next_phase(self.phase, entered_at, cx.now(), &interoception);
        let should_emit = self.last_phase_entered_at.is_none() || next != self.phase;
        if next != self.phase {
            tracing::info!(from = ?self.phase, to = ?next, "homeostatic phase transition");
            self.phase = next;
            self.last_phase_entered_at = Some(cx.now());
        } else if self.last_phase_entered_at.is_none() {
            self.last_phase_entered_at = Some(cx.now());
        }
        if should_emit {
            self.emit_phase(next).await;
        }
        Ok(())
    }

    async fn emit_phase(&self, phase: HomeostaticPhase) {
        let mut commands = drive_commands(phase, self.allocation.allowed_target_modules());
        commands.extend(suppression_commands(
            phase,
            self.allocation.allowed_suppression_modules(),
        ));
        self.allocation.submit(commands).await;
    }

    async fn next_batch(&mut self) -> Result<()> {
        tokio::select! {
            update = self.interoception_updates.next_item() => {
                let _ = update?;
            }
            _ = tokio::time::sleep(PERIODIC_WAKEUP) => {}
        }
        let _ = self.interoception_updates.take_ready_items()?;
        Ok(())
    }
}

fn next_phase(
    current: HomeostaticPhase,
    entered_at: DateTime<Utc>,
    now: DateTime<Utc>,
    interoception: &InteroceptiveState,
) -> HomeostaticPhase {
    if interoception.wake_arousal >= FORCE_WAKE {
        return HomeostaticPhase::Wake;
    }
    let elapsed = (now - entered_at).to_std().unwrap_or(Duration::ZERO);
    match current {
        HomeostaticPhase::Wake => {
            if interoception.nrem_pressure >= NREM_ENTER
                || matches!(
                    interoception.mode,
                    nuillu_blackboard::InteroceptiveMode::NremPressure
                )
            {
                HomeostaticPhase::Compacting
            } else {
                HomeostaticPhase::Wake
            }
        }
        HomeostaticPhase::Compacting => {
            if elapsed >= MAX_NREM_DURATION
                || (elapsed >= MIN_PHASE_DURATION && interoception.nrem_pressure <= NREM_EXIT)
            {
                HomeostaticPhase::Recombining
            } else {
                HomeostaticPhase::Compacting
            }
        }
        HomeostaticPhase::Recombining => {
            if elapsed >= MAX_REM_DURATION
                || (elapsed >= MIN_PHASE_DURATION && interoception.rem_pressure <= REM_EXIT)
            {
                if sleep_persists(interoception) {
                    HomeostaticPhase::Compacting
                } else {
                    HomeostaticPhase::Wake
                }
            } else {
                HomeostaticPhase::Recombining
            }
        }
    }
}

fn sleep_persists(interoception: &InteroceptiveState) -> bool {
    interoception.wake_arousal < FORCE_WAKE
        && (interoception.nrem_pressure >= NREM_EXIT
            || matches!(
                interoception.mode,
                nuillu_blackboard::InteroceptiveMode::NremPressure
                    | nuillu_blackboard::InteroceptiveMode::RemPressure
            ))
}

fn drive_commands(phase: HomeostaticPhase, allowed: &[ModuleId]) -> Vec<AllocationCommand> {
    let mut commands = Vec::new();
    for id in allowed {
        let level = if *id == builtin::memory_compaction()
            || *id == builtin::memory_association()
            || *id == builtin::policy_compaction()
        {
            match phase {
                HomeostaticPhase::Compacting => AllocationEffectLevel::Low,
                HomeostaticPhase::Wake | HomeostaticPhase::Recombining => {
                    AllocationEffectLevel::Off
                }
            }
        } else if *id == builtin::memory_recombination() {
            match phase {
                HomeostaticPhase::Recombining => AllocationEffectLevel::Low,
                HomeostaticPhase::Wake | HomeostaticPhase::Compacting => AllocationEffectLevel::Off,
            }
        } else {
            continue;
        };
        commands.push(AllocationCommand::target(
            id.clone(),
            level,
            Some(phase_guidance(phase, id)),
        ));
    }
    commands
}

fn suppression_commands(phase: HomeostaticPhase, capped: &[ModuleId]) -> Vec<AllocationCommand> {
    let level = match phase {
        HomeostaticPhase::Wake => AllocationEffectLevel::Off,
        HomeostaticPhase::Compacting => AllocationEffectLevel::Max,
        HomeostaticPhase::Recombining => AllocationEffectLevel::High,
    };
    capped
        .iter()
        .cloned()
        .map(|id| AllocationCommand::suppression(id, level))
        .collect()
}

fn phase_guidance(phase: HomeostaticPhase, id: &ModuleId) -> String {
    match (phase, id.as_str()) {
        (HomeostaticPhase::Compacting, "memory-compaction") => {
            "NREM-like memory consolidation: merge redundant memories and reduce nrem pressure."
                .into()
        }
        (HomeostaticPhase::Compacting, "memory-association") => {
            "NREM-like memory association: write non-destructive reflection summaries and memory links where source memories should remain live."
                .into()
        }
        (HomeostaticPhase::Compacting, "policy-compaction") => {
            "NREM-like policy cleanup: conservatively remove redundant non-Core policy duplicates."
                .into()
        }
        (HomeostaticPhase::Recombining, "memory-recombination") => {
            "REM-like internal simulation: recombine recent cognition with memory without treating it as verified fact."
                .into()
        }
        _ => "Homeostatic controller keeps this sleep-phase module quiet now.".into(),
    }
}

#[async_trait(?Send)]
impl Module for HomeostaticControllerModule {
    type Batch = ();

    fn id() -> &'static str {
        "homeostatic-controller"
    }

    fn role_description() -> &'static str {
        "Autonomic finite-state controller that drives sleep-like memory compaction/recombination and caps action modules during sleep pressure without using an LLM."
    }

    async fn next_batch(&mut self) -> Result<Self::Batch> {
        HomeostaticControllerModule::next_batch(self).await
    }

    async fn activate(
        &mut self,
        cx: &nuillu_module::ActivateCx<'_>,
        _batch: &Self::Batch,
    ) -> Result<()> {
        HomeostaticControllerModule::activate(self, cx).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn at(secs: i64) -> DateTime<Utc> {
        DateTime::<Utc>::from_timestamp(secs, 0).unwrap()
    }

    #[test]
    fn phase_transitions_use_hysteresis_and_force_wake() {
        let interoception = InteroceptiveState {
            nrem_pressure: 0.8,
            ..InteroceptiveState::default()
        };
        assert_eq!(
            next_phase(HomeostaticPhase::Wake, at(0), at(0), &interoception),
            HomeostaticPhase::Compacting
        );

        let interoception = InteroceptiveState {
            nrem_pressure: 0.2,
            ..InteroceptiveState::default()
        };
        assert_eq!(
            next_phase(HomeostaticPhase::Compacting, at(0), at(1), &interoception),
            HomeostaticPhase::Compacting
        );
        assert_eq!(
            next_phase(HomeostaticPhase::Compacting, at(0), at(3), &interoception),
            HomeostaticPhase::Recombining
        );

        let interoception = InteroceptiveState {
            wake_arousal: 0.95,
            rem_pressure: 0.8,
            ..InteroceptiveState::default()
        };
        assert_eq!(
            next_phase(HomeostaticPhase::Recombining, at(0), at(3), &interoception),
            HomeostaticPhase::Wake
        );
    }

    #[test]
    fn phase_transitions_enter_and_cycle_sleep_from_interoceptive_mode() {
        let sleeping = InteroceptiveState {
            mode: nuillu_blackboard::InteroceptiveMode::NremPressure,
            wake_arousal: 0.1,
            nrem_pressure: 0.2,
            ..InteroceptiveState::default()
        };
        assert_eq!(
            next_phase(HomeostaticPhase::Wake, at(0), at(1), &sleeping),
            HomeostaticPhase::Compacting
        );

        let rem_done_but_sleeping = InteroceptiveState {
            mode: nuillu_blackboard::InteroceptiveMode::RemPressure,
            wake_arousal: 0.1,
            nrem_pressure: 0.4,
            rem_pressure: 0.1,
            ..InteroceptiveState::default()
        };
        assert_eq!(
            next_phase(
                HomeostaticPhase::Recombining,
                at(0),
                at(3),
                &rem_done_but_sleeping,
            ),
            HomeostaticPhase::Compacting
        );

        let awake = InteroceptiveState {
            mode: nuillu_blackboard::InteroceptiveMode::Wake,
            wake_arousal: 0.8,
            nrem_pressure: 1.0,
            rem_pressure: 1.0,
            ..InteroceptiveState::default()
        };
        assert_eq!(
            next_phase(HomeostaticPhase::Compacting, at(0), at(3), &awake),
            HomeostaticPhase::Wake
        );
    }

    #[test]
    fn allocation_commands_drive_memory_and_suppress_actions() {
        let drive = drive_commands(
            HomeostaticPhase::Compacting,
            &[
                builtin::memory_compaction(),
                builtin::memory_association(),
                builtin::policy_compaction(),
                builtin::memory_recombination(),
            ],
        );
        assert_eq!(
            target_level(&drive, &builtin::memory_compaction()),
            Some(AllocationEffectLevel::Low)
        );
        assert_eq!(
            target_level(&drive, &builtin::memory_association()),
            Some(AllocationEffectLevel::Low)
        );
        assert_eq!(
            target_level(&drive, &builtin::policy_compaction()),
            Some(AllocationEffectLevel::Low)
        );
        assert_eq!(
            target_level(&drive, &builtin::memory_recombination()),
            Some(AllocationEffectLevel::Off)
        );

        let drive = drive_commands(
            HomeostaticPhase::Recombining,
            &[
                builtin::memory_compaction(),
                builtin::memory_recombination(),
            ],
        );
        assert_eq!(
            target_level(&drive, &builtin::memory_compaction()),
            Some(AllocationEffectLevel::Off)
        );
        assert_eq!(
            target_level(&drive, &builtin::memory_recombination()),
            Some(AllocationEffectLevel::Low)
        );

        let suppressions = suppression_commands(
            HomeostaticPhase::Compacting,
            &[builtin::speak_gate(), builtin::speak()],
        );
        assert_eq!(
            suppression_level_for(&suppressions, &builtin::speak()),
            Some(AllocationEffectLevel::Max)
        );
    }

    fn target_level(
        commands: &[AllocationCommand],
        module: &ModuleId,
    ) -> Option<AllocationEffectLevel> {
        commands
            .iter()
            .find(|command| command.module == *module)
            .map(|command| command.level)
    }

    fn suppression_level_for(
        commands: &[AllocationCommand],
        module: &ModuleId,
    ) -> Option<AllocationEffectLevel> {
        commands
            .iter()
            .find(|command| command.module == *module)
            .map(|command| command.level)
    }
}
