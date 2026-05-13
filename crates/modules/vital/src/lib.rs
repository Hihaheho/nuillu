use std::time::Duration;

use anyhow::Result;
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use nuillu_blackboard::{VitalMode, VitalPatch, VitalState};
use nuillu_module::{
    AllocationUpdatedInbox, BlackboardReader, CognitionLogUpdatedInbox, Module, VitalWriter,
};
use nuillu_types::builtin;

const PERIODIC_WAKEUP: Duration = Duration::from_secs(1);
const NREM_FROM_COGNITION: f32 = 0.06;
const NREM_FROM_ELAPSED_SEC: f32 = 0.002;
const NREM_RELIEF_PER_REMEMBER_TOKEN: f32 = 0.25;
const REM_FROM_REMEMBER_TOKEN: f32 = 0.20;
const REM_RELIEF_PER_RECOMBINATION: f32 = 0.35;
const WAKE_AROUSAL_FROM_COGNITION: f32 = 0.03;
const WAKE_AROUSAL_DECAY_PER_SEC: f32 = 0.02;

pub struct VitalModule {
    cognition_updates: CognitionLogUpdatedInbox,
    allocation_updates: AllocationUpdatedInbox,
    blackboard: BlackboardReader,
    vital: VitalWriter,
    last_seen_cognition_index: Option<u64>,
    last_total_remember_tokens: Option<u32>,
}

impl VitalModule {
    pub fn new(
        cognition_updates: CognitionLogUpdatedInbox,
        allocation_updates: AllocationUpdatedInbox,
        blackboard: BlackboardReader,
        vital: VitalWriter,
    ) -> Self {
        Self {
            cognition_updates,
            allocation_updates,
            blackboard,
            vital,
            last_seen_cognition_index: None,
            last_total_remember_tokens: None,
        }
    }

    async fn activate(&mut self, cx: &nuillu_module::ActivateCx<'_>) -> Result<()> {
        let (current, unread, total_remember_tokens) = self
            .blackboard
            .read(|bb| {
                let unread = bb.unread_cognition_log_entries(self.last_seen_cognition_index);
                let total = bb
                    .memory_metadata()
                    .values()
                    .map(|metadata| metadata.remember_tokens)
                    .sum::<u32>();
                (bb.vital().clone(), unread, total)
            })
            .await;
        if let Some(index) = unread.last().map(|record| record.index) {
            self.last_seen_cognition_index = Some(index);
        }
        let previous_total = self
            .last_total_remember_tokens
            .replace(total_remember_tokens)
            .unwrap_or(total_remember_tokens);
        let remember_delta = total_remember_tokens.saturating_sub(previous_total);
        let recombination_entries = unread
            .iter()
            .filter(|record| record.source.module == builtin::memory_recombination())
            .count() as u32;
        let non_dream_entries = unread
            .iter()
            .filter(|record| record.source.module != builtin::memory_recombination())
            .count() as u32;

        let patch = next_vital_patch(
            &current,
            cx.now(),
            non_dream_entries,
            recombination_entries,
            remember_delta,
        );
        self.vital.update(patch).await;
        Ok(())
    }

    async fn next_batch(&mut self) -> Result<()> {
        tokio::select! {
            update = self.cognition_updates.next_item() => {
                let _ = update?;
            }
            update = self.allocation_updates.next_item() => {
                let _ = update?;
            }
            _ = tokio::time::sleep(PERIODIC_WAKEUP) => {}
        }
        let _ = self.cognition_updates.take_ready_items()?;
        let _ = self.allocation_updates.take_ready_items()?;
        Ok(())
    }
}

fn next_vital_patch(
    current: &VitalState,
    now: DateTime<Utc>,
    non_dream_entries: u32,
    recombination_entries: u32,
    remember_delta: u32,
) -> VitalPatch {
    let elapsed_secs = (now - current.last_updated)
        .to_std()
        .unwrap_or(Duration::ZERO)
        .as_secs_f32()
        .min(120.0);
    let nrem_pressure = current.nrem_pressure
        + non_dream_entries as f32 * NREM_FROM_COGNITION
        + elapsed_secs * NREM_FROM_ELAPSED_SEC
        - remember_delta as f32 * NREM_RELIEF_PER_REMEMBER_TOKEN;
    let rem_pressure = current.rem_pressure + remember_delta as f32 * REM_FROM_REMEMBER_TOKEN
        - recombination_entries as f32 * REM_RELIEF_PER_RECOMBINATION;
    let wake_arousal = current.wake_arousal
        + non_dream_entries as f32 * WAKE_AROUSAL_FROM_COGNITION
        - elapsed_secs * WAKE_AROUSAL_DECAY_PER_SEC;
    let nrem_pressure = clamp_unit(nrem_pressure);
    let rem_pressure = clamp_unit(rem_pressure);
    let wake_arousal = clamp_unit(wake_arousal);
    let mode = if wake_arousal >= 0.9 {
        VitalMode::Wake
    } else if nrem_pressure >= 0.7 {
        VitalMode::NremPressure
    } else if rem_pressure >= 0.4 && nrem_pressure <= 0.35 {
        VitalMode::RemPressure
    } else {
        VitalMode::Wake
    };

    VitalPatch {
        mode: Some(mode),
        wake_arousal: Some(wake_arousal),
        nrem_pressure: Some(nrem_pressure),
        rem_pressure: Some(rem_pressure),
    }
}

fn clamp_unit(value: f32) -> f32 {
    if value.is_nan() {
        0.0
    } else {
        value.clamp(0.0, 1.0)
    }
}

#[async_trait(?Send)]
impl Module for VitalModule {
    type Batch = ();

    fn id() -> &'static str {
        "vital"
    }

    fn role_description() -> &'static str {
        "Updates the agent's homeostatic vital state from cognition volume, elapsed time, memory compaction traces, and dream recombination traces without using an LLM."
    }

    async fn next_batch(&mut self) -> Result<Self::Batch> {
        VitalModule::next_batch(self).await
    }

    async fn activate(
        &mut self,
        cx: &nuillu_module::ActivateCx<'_>,
        _batch: &Self::Batch,
    ) -> Result<()> {
        VitalModule::activate(self, cx).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vital_pressure_cycles_from_cognition_compaction_and_recombination() {
        let now = DateTime::<Utc>::from_timestamp(10, 0).unwrap();
        let current = VitalState {
            last_updated: DateTime::<Utc>::from_timestamp(0, 0).unwrap(),
            ..VitalState::default()
        };

        let patch = next_vital_patch(&current, now, 12, 0, 0);
        assert_eq!(patch.mode, Some(VitalMode::NremPressure));
        assert!(patch.nrem_pressure.unwrap() >= 0.7);

        let compacted = VitalState {
            nrem_pressure: 0.8,
            rem_pressure: 0.0,
            last_updated: now,
            ..VitalState::default()
        };
        let patch = next_vital_patch(&compacted, now, 0, 0, 3);
        assert!(patch.nrem_pressure.unwrap() <= 0.3);
        assert!(patch.rem_pressure.unwrap() > 0.4);

        let recombined = VitalState {
            rem_pressure: 0.7,
            last_updated: now,
            ..VitalState::default()
        };
        let patch = next_vital_patch(&recombined, now, 0, 2, 0);
        assert!(patch.rem_pressure.unwrap() <= 0.1);
    }
}
