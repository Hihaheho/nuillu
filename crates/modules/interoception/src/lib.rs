use std::time::Duration;

use anyhow::{Context, Result};
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use lutum::{Session, StructuredTurnOutcome};
use nuillu_blackboard::{
    CognitionLogEntryRecord, InteroceptiveMode, InteroceptivePatch, InteroceptiveState,
    MemoLogRecord,
};
use nuillu_module::{
    AllocationUpdatedInbox, BlackboardReader, CognitionLogUpdatedInbox, InteroceptiveWriter,
    LlmAccess, MemoUpdatedInbox, Module,
};
use nuillu_types::builtin;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

const SYSTEM_PROMPT: &str = r#"You are the interoception module.
Maintain the agent's internal state from the recent cognitive workspace. Estimate affect as
current internal condition, not a moral judgment and not a memory fact.
Return affect_arousal in [0, 1], valence in [-1, 1], and one short untyped emotion phrase.
Use neutral values when evidence is weak. Do not mention implementation details."#;

const PERIODIC_WAKEUP: Duration = Duration::from_secs(1);
const NREM_FROM_COGNITION: f32 = 0.06;
const NREM_FROM_ELAPSED_SEC: f32 = 0.002;
const NREM_RELIEF_PER_REMEMBER_TOKEN: f32 = 0.25;
const REM_FROM_REMEMBER_TOKEN: f32 = 0.20;
const REM_RELIEF_PER_RECOMBINATION: f32 = 0.35;
const WAKE_AROUSAL_FROM_COGNITION: f32 = 0.03;
const WAKE_AROUSAL_DECAY_PER_SEC: f32 = 0.02;

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct AffectAssessment {
    pub affect_arousal: f32,
    pub valence: f32,
    pub emotion: String,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct InteroceptionBatch {
    affect_candidate: bool,
}

pub struct InteroceptionModule {
    memo_updates: MemoUpdatedInbox,
    cognition_updates: CognitionLogUpdatedInbox,
    allocation_updates: AllocationUpdatedInbox,
    blackboard: BlackboardReader,
    interoception: InteroceptiveWriter,
    llm: LlmAccess,
    session: Session,
    last_seen_cognition_index: Option<u64>,
    last_total_remember_tokens: Option<u32>,
}

impl InteroceptionModule {
    pub fn new(
        memo_updates: MemoUpdatedInbox,
        cognition_updates: CognitionLogUpdatedInbox,
        allocation_updates: AllocationUpdatedInbox,
        blackboard: BlackboardReader,
        interoception: InteroceptiveWriter,
        llm: LlmAccess,
    ) -> Self {
        Self {
            memo_updates,
            cognition_updates,
            allocation_updates,
            blackboard,
            interoception,
            llm,
            session: Session::new(),
            last_seen_cognition_index: None,
            last_total_remember_tokens: None,
        }
    }

    async fn activate(
        &mut self,
        cx: &nuillu_module::ActivateCx<'_>,
        batch: &InteroceptionBatch,
    ) -> Result<()> {
        let unread_memos = if batch.affect_candidate {
            self.blackboard.unread_memo_logs().await
        } else {
            Vec::new()
        };
        let (current, unread_cognition, total_remember_tokens, allocation) = self
            .blackboard
            .read(|bb| {
                let unread = bb.unread_cognition_log_entries(self.last_seen_cognition_index);
                let total = bb
                    .memory_metadata()
                    .values()
                    .map(|metadata| metadata.remember_tokens)
                    .sum::<u32>();
                (
                    bb.interoception().clone(),
                    unread,
                    total,
                    bb.allocation().clone(),
                )
            })
            .await;
        if let Some(index) = unread_cognition.last().map(|record| record.index) {
            self.last_seen_cognition_index = Some(index);
        }
        let previous_total = self
            .last_total_remember_tokens
            .replace(total_remember_tokens)
            .unwrap_or(total_remember_tokens);
        let remember_delta = total_remember_tokens.saturating_sub(previous_total);
        let recombination_entries = unread_cognition
            .iter()
            .filter(|record| record.source.module == builtin::memory_recombination())
            .count() as u32;
        let non_dream_entries = unread_cognition
            .iter()
            .filter(|record| record.source.module != builtin::memory_recombination())
            .count() as u32;

        let mut patch = next_interoception_patch(
            &current,
            cx.now(),
            non_dream_entries,
            recombination_entries,
            remember_delta,
        );
        if batch.affect_candidate && (!unread_memos.is_empty() || !unread_cognition.is_empty()) {
            let affect = self
                .estimate_affect(&current, &unread_memos, &unread_cognition, &allocation)
                .await?;
            merge_affect_patch(&mut patch, affect);
        }
        self.interoception.update(patch).await;
        Ok(())
    }

    async fn next_batch(&mut self) -> Result<InteroceptionBatch> {
        let mut affect_candidate = tokio::select! {
            update = self.memo_updates.next_item() => {
                let _ = update?;
                true
            }
            update = self.cognition_updates.next_item() => {
                let _ = update?;
                true
            }
            update = self.allocation_updates.next_item() => {
                let _ = update?;
                true
            }
            _ = tokio::time::sleep(PERIODIC_WAKEUP) => false,
        };
        affect_candidate |= !self.memo_updates.take_ready_items()?.items.is_empty();
        affect_candidate |= !self.cognition_updates.take_ready_items()?.items.is_empty();
        affect_candidate |= !self.allocation_updates.take_ready_items()?.items.is_empty();
        Ok(InteroceptionBatch { affect_candidate })
    }

    async fn estimate_affect(
        &mut self,
        current: &InteroceptiveState,
        unread_memos: &[MemoLogRecord],
        unread_cognition: &[CognitionLogEntryRecord],
        allocation: &nuillu_blackboard::ResourceAllocation,
    ) -> Result<AffectAssessment> {
        self.session.push_ephemeral_system(SYSTEM_PROMPT);
        self.session.push_ephemeral_user(format!(
            "Current interoceptive state:\n{}\n\nUnread memos:\n{}\n\nUnread cognition:\n{}\n\nCurrent allocation:\n{}",
            serde_json::to_string(current).unwrap_or_default(),
            serde_json::to_string(unread_memos).unwrap_or_default(),
            serde_json::to_string(unread_cognition).unwrap_or_default(),
            serde_json::to_string(allocation).unwrap_or_default(),
        ));
        let lutum = self.llm.lutum().await;
        let result = self
            .session
            .structured_turn::<AffectAssessment>(&lutum)
            .collect()
            .await
            .context("interoception structured turn failed")?;
        let StructuredTurnOutcome::Structured(assessment) = result.semantic else {
            anyhow::bail!("interoception structured turn refused");
        };
        Ok(normalize_affect_assessment(assessment))
    }
}

fn normalize_affect_assessment(assessment: AffectAssessment) -> AffectAssessment {
    AffectAssessment {
        affect_arousal: clamp_unit(assessment.affect_arousal),
        valence: clamp_signed_unit(assessment.valence),
        emotion: assessment.emotion.trim().to_owned(),
    }
}

fn merge_affect_patch(patch: &mut InteroceptivePatch, affect: AffectAssessment) {
    patch.affect_arousal = Some(affect.affect_arousal);
    patch.valence = Some(affect.valence);
    patch.emotion = Some(affect.emotion);
}

fn next_interoception_patch(
    current: &InteroceptiveState,
    now: DateTime<Utc>,
    non_dream_entries: u32,
    recombination_entries: u32,
    remember_delta: u32,
) -> InteroceptivePatch {
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
        InteroceptiveMode::Wake
    } else if nrem_pressure >= 0.7 {
        InteroceptiveMode::NremPressure
    } else if rem_pressure >= 0.4 && nrem_pressure <= 0.35 {
        InteroceptiveMode::RemPressure
    } else {
        InteroceptiveMode::Wake
    };

    InteroceptivePatch {
        mode: Some(mode),
        wake_arousal: Some(wake_arousal),
        nrem_pressure: Some(nrem_pressure),
        rem_pressure: Some(rem_pressure),
        affect_arousal: None,
        valence: None,
        emotion: None,
    }
}

fn clamp_unit(value: f32) -> f32 {
    if value.is_nan() {
        0.0
    } else {
        value.clamp(0.0, 1.0)
    }
}

fn clamp_signed_unit(value: f32) -> f32 {
    if value.is_nan() {
        0.0
    } else {
        value.clamp(-1.0, 1.0)
    }
}

#[async_trait(?Send)]
impl Module for InteroceptionModule {
    type Batch = InteroceptionBatch;

    fn id() -> &'static str {
        "interoception"
    }

    fn role_description() -> &'static str {
        "Updates the agent's interoceptive state: homeostatic sleep pressure, wake arousal, affect arousal, valence, and untyped emotion text."
    }

    async fn next_batch(&mut self) -> Result<Self::Batch> {
        InteroceptionModule::next_batch(self).await
    }

    async fn activate(
        &mut self,
        cx: &nuillu_module::ActivateCx<'_>,
        batch: &Self::Batch,
    ) -> Result<()> {
        InteroceptionModule::activate(self, cx, batch).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interoception_pressure_cycles_from_cognition_compaction_and_recombination() {
        let now = DateTime::<Utc>::from_timestamp(10, 0).unwrap();
        let current = InteroceptiveState {
            last_updated: DateTime::<Utc>::from_timestamp(0, 0).unwrap(),
            ..InteroceptiveState::default()
        };

        let patch = next_interoception_patch(&current, now, 12, 0, 0);
        assert_eq!(patch.mode, Some(InteroceptiveMode::NremPressure));
        assert!(patch.nrem_pressure.unwrap() >= 0.7);

        let compacted = InteroceptiveState {
            nrem_pressure: 0.8,
            rem_pressure: 0.0,
            last_updated: now,
            ..InteroceptiveState::default()
        };
        let patch = next_interoception_patch(&compacted, now, 0, 0, 3);
        assert!(patch.nrem_pressure.unwrap() <= 0.3);
        assert!(patch.rem_pressure.unwrap() > 0.4);

        let recombined = InteroceptiveState {
            rem_pressure: 0.7,
            last_updated: now,
            ..InteroceptiveState::default()
        };
        let patch = next_interoception_patch(&recombined, now, 0, 2, 0);
        assert!(patch.rem_pressure.unwrap() <= 0.1);
    }

    #[test]
    fn affect_assessment_is_normalized() {
        let normalized = normalize_affect_assessment(AffectAssessment {
            affect_arousal: f32::NAN,
            valence: -1.5,
            emotion: "  startled relief  ".to_owned(),
        });

        assert_eq!(normalized.affect_arousal, 0.0);
        assert_eq!(normalized.valence, -1.0);
        assert_eq!(normalized.emotion, "startled relief");
    }
}
