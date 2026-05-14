use std::time::Duration;

use anyhow::Result;
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use nuillu_blackboard::{ActivationRatio, InteroceptiveState, ModuleConfig, ResourceAllocation};
use nuillu_module::{AllocationWriter, InteroceptiveReader, InteroceptiveUpdatedInbox, Module};
use nuillu_types::{ModuleId, builtin};

const PERIODIC_WAKEUP: Duration = Duration::from_secs(1);
const NREM_ENTER: f32 = 0.7;
const NREM_EXIT: f32 = 0.3;
const REM_EXIT: f32 = 0.2;
const FORCE_WAKE: f32 = 0.9;
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
        let drive = drive_allocation(phase, self.allocation.allowed_drive_modules());
        self.allocation.set_drive(drive).await;
        let cap = cap_allocation(phase, self.allocation.allowed_cap_modules());
        self.allocation.set_cap(cap).await;
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
            if interoception.nrem_pressure >= NREM_ENTER {
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
                HomeostaticPhase::Wake
            } else {
                HomeostaticPhase::Recombining
            }
        }
    }
}

fn drive_allocation(phase: HomeostaticPhase, allowed: &[ModuleId]) -> ResourceAllocation {
    let mut allocation = ResourceAllocation::default();
    for id in allowed {
        let ratio = if *id == builtin::memory_compaction() || *id == builtin::memory_association() {
            match phase {
                HomeostaticPhase::Compacting => 1.0,
                HomeostaticPhase::Wake | HomeostaticPhase::Recombining => 0.0,
            }
        } else if *id == builtin::memory_recombination() {
            match phase {
                HomeostaticPhase::Recombining => 1.0,
                HomeostaticPhase::Wake | HomeostaticPhase::Compacting => 0.0,
            }
        } else {
            continue;
        };
        allocation.set(
            id.clone(),
            ModuleConfig {
                guidance: phase_guidance(phase, id),
            },
        );
        allocation.set_activation(id.clone(), ActivationRatio::from_f64(ratio));
    }
    allocation
}

fn cap_allocation(phase: HomeostaticPhase, capped: &[ModuleId]) -> ResourceAllocation {
    let ratio = match phase {
        HomeostaticPhase::Wake => 1.0,
        HomeostaticPhase::Compacting => 0.15,
        HomeostaticPhase::Recombining => 0.25,
    };
    let mut allocation = ResourceAllocation::default();
    for id in capped {
        allocation.set_activation(id.clone(), ActivationRatio::from_f64(ratio));
    }
    allocation
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
    fn allocations_drive_memory_and_cap_actions() {
        let drive = drive_allocation(
            HomeostaticPhase::Compacting,
            &[
                builtin::memory_compaction(),
                builtin::memory_association(),
                builtin::memory_recombination(),
            ],
        );
        assert_eq!(
            drive.activation_for(&builtin::memory_compaction()),
            ActivationRatio::ONE
        );
        assert_eq!(
            drive.activation_for(&builtin::memory_association()),
            ActivationRatio::ONE
        );
        assert_eq!(
            drive.activation_for(&builtin::memory_recombination()),
            ActivationRatio::ZERO
        );

        let cap = cap_allocation(
            HomeostaticPhase::Compacting,
            &[builtin::speak_gate(), builtin::speak()],
        );
        assert_eq!(
            cap.activation_for(&builtin::speak()),
            ActivationRatio::from_f64(0.15)
        );
    }
}
