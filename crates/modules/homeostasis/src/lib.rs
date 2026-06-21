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
    Dreaming,
}

pub struct HomeostasisModule {
    interoception_updates: InteroceptiveUpdatedInbox,
    interoception: InteroceptiveReader,
    allocation: AllocationWriter,
    phase: HomeostaticPhase,
    last_phase_entered_at: Option<DateTime<Utc>>,
    last_emitted_suppression: Option<AllocationEffectLevel>,
}

impl HomeostasisModule {
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
            last_emitted_suppression: None,
        }
    }

    async fn activate(&mut self, cx: &nuillu_module::ActivateCx<'_>) -> Result<()> {
        let interoception = self.interoception.snapshot().await;
        let entered_at = self.last_phase_entered_at.unwrap_or_else(|| cx.now());
        let next = next_phase(self.phase, entered_at, cx.now(), &interoception);
        let suppression = suppression_level(next, &interoception);
        let should_emit = self.last_phase_entered_at.is_none()
            || next != self.phase
            || self.last_emitted_suppression != Some(suppression);
        if next != self.phase {
            tracing::info!(from = ?self.phase, to = ?next, "homeostatic phase transition");
            self.phase = next;
            self.last_phase_entered_at = Some(cx.now());
        } else if self.last_phase_entered_at.is_none() {
            self.last_phase_entered_at = Some(cx.now());
        }
        if should_emit {
            self.emit_phase(next, suppression).await?;
            self.last_emitted_suppression = Some(suppression);
        }
        Ok(())
    }

    async fn emit_phase(
        &self,
        phase: HomeostaticPhase,
        suppression: AllocationEffectLevel,
    ) -> Result<()> {
        let mut commands = drive_commands(phase, self.allocation.allowed_target_modules());
        commands.extend(suppression_commands(
            suppression,
            self.allocation.allowed_suppression_modules(),
        ));
        self.allocation.submit(commands).await?;
        Ok(())
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
                HomeostaticPhase::Dreaming
            } else {
                HomeostaticPhase::Compacting
            }
        }
        HomeostaticPhase::Dreaming => {
            if elapsed >= MAX_REM_DURATION
                || (elapsed >= MIN_PHASE_DURATION && interoception.rem_pressure <= REM_EXIT)
            {
                if sleep_persists(interoception) {
                    HomeostaticPhase::Compacting
                } else {
                    HomeostaticPhase::Wake
                }
            } else {
                HomeostaticPhase::Dreaming
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
                HomeostaticPhase::Wake | HomeostaticPhase::Dreaming => AllocationEffectLevel::Off,
            }
        } else if *id == builtin::dreaming() {
            match phase {
                HomeostaticPhase::Dreaming => AllocationEffectLevel::Low,
                HomeostaticPhase::Wake | HomeostaticPhase::Compacting => AllocationEffectLevel::Off,
            }
        } else {
            continue;
        };
        commands.push(AllocationCommand::target(id.clone(), level));
    }
    commands
}

fn suppression_level(
    phase: HomeostaticPhase,
    interoception: &InteroceptiveState,
) -> AllocationEffectLevel {
    match phase {
        HomeostaticPhase::Wake => wake_suppression_level(interoception),
        HomeostaticPhase::Compacting => AllocationEffectLevel::Max,
        HomeostaticPhase::Dreaming => AllocationEffectLevel::High,
    }
}

fn wake_suppression_level(interoception: &InteroceptiveState) -> AllocationEffectLevel {
    if interoception.wake_arousal >= 0.70 {
        AllocationEffectLevel::Off
    } else if interoception.wake_arousal >= 0.45 {
        AllocationEffectLevel::Low
    } else if interoception.wake_arousal >= 0.25 {
        AllocationEffectLevel::Normal
    } else {
        AllocationEffectLevel::High
    }
}

fn suppression_commands(
    level: AllocationEffectLevel,
    capped: &[ModuleId],
) -> Vec<AllocationCommand> {
    capped
        .iter()
        .cloned()
        .map(|id| AllocationCommand::suppression(id, level))
        .collect()
}

#[async_trait(?Send)]
impl Module for HomeostasisModule {
    type Batch = ();

    fn id() -> &'static str {
        "homeostasis"
    }

    fn peer_context() -> Option<&'static str> {
        None
    }

    async fn next_batch(&mut self) -> Result<Self::Batch> {
        HomeostasisModule::next_batch(self).await
    }

    async fn activate(
        &mut self,
        cx: &nuillu_module::ActivateCx<'_>,
        _batch: &Self::Batch,
    ) -> Result<()> {
        HomeostasisModule::activate(self, cx).await
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
            HomeostaticPhase::Dreaming
        );

        let interoception = InteroceptiveState {
            wake_arousal: 0.95,
            rem_pressure: 0.8,
            ..InteroceptiveState::default()
        };
        assert_eq!(
            next_phase(HomeostaticPhase::Dreaming, at(0), at(3), &interoception),
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
                HomeostaticPhase::Dreaming,
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
                builtin::dreaming(),
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
            target_level(&drive, &builtin::dreaming()),
            Some(AllocationEffectLevel::Off)
        );

        let drive = drive_commands(
            HomeostaticPhase::Dreaming,
            &[builtin::memory_compaction(), builtin::dreaming()],
        );
        assert_eq!(
            target_level(&drive, &builtin::memory_compaction()),
            Some(AllocationEffectLevel::Off)
        );
        assert_eq!(
            target_level(&drive, &builtin::dreaming()),
            Some(AllocationEffectLevel::Low)
        );

        let suppressions = suppression_commands(AllocationEffectLevel::Max, &[builtin::speak()]);
        assert_eq!(
            suppression_level_for(&suppressions, &builtin::speak()),
            Some(AllocationEffectLevel::Max)
        );
    }

    #[test]
    fn wake_suppression_uses_wake_arousal_thresholds() {
        assert_eq!(
            suppression_level(
                HomeostaticPhase::Wake,
                &InteroceptiveState {
                    wake_arousal: 0.75,
                    ..InteroceptiveState::default()
                },
            ),
            AllocationEffectLevel::Off
        );
        assert_eq!(
            suppression_level(
                HomeostaticPhase::Wake,
                &InteroceptiveState {
                    wake_arousal: 0.50,
                    ..InteroceptiveState::default()
                },
            ),
            AllocationEffectLevel::Low
        );
        assert_eq!(
            suppression_level(
                HomeostaticPhase::Wake,
                &InteroceptiveState {
                    wake_arousal: 0.30,
                    ..InteroceptiveState::default()
                },
            ),
            AllocationEffectLevel::Normal
        );
        assert_eq!(
            suppression_level(
                HomeostaticPhase::Wake,
                &InteroceptiveState {
                    wake_arousal: 0.10,
                    ..InteroceptiveState::default()
                },
            ),
            AllocationEffectLevel::High
        );
    }

    #[test]
    fn sleep_phase_suppression_overrides_wake_arousal_thresholds() {
        let interoception = InteroceptiveState {
            wake_arousal: 0.90,
            ..InteroceptiveState::default()
        };

        assert_eq!(
            suppression_level(HomeostaticPhase::Compacting, &interoception),
            AllocationEffectLevel::Max
        );
        assert_eq!(
            suppression_level(HomeostaticPhase::Dreaming, &interoception),
            AllocationEffectLevel::High
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
