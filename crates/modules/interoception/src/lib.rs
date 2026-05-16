use std::time::Duration;

use anyhow::{Context, Result};
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use lutum::{Session, StructuredTurnOutcome};
use nuillu_blackboard::{
    AllocationCommand, AllocationEffectLevel, CognitionLogEntryRecord, InteroceptiveMode,
    InteroceptivePatch, InteroceptiveState, MemoLogRecord,
};
use nuillu_module::{
    AllocationUpdatedInbox, AllocationWriter, BlackboardReader, CognitionLogUpdatedInbox,
    InteroceptionRuntimePolicy, InteroceptiveWriter, LlmAccess, MemoUpdatedInbox, Module,
};
use nuillu_types::builtin;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

const SYSTEM_PROMPT: &str = r#"You are the interoception module.
Maintain the agent's internal state from the recent cognitive workspace. Estimate affect as
current internal condition, not a moral judgment and not a memory fact.
For arousal, return structured increase levels rather than numeric deltas. Use higher wake and
affect increase levels only when the unread sensory/cognitive evidence is salient enough to wake or
emotionally activate the agent. Return valence in [-1, 1] and one short untyped emotion phrase.
Use neutral values when evidence is weak. Do not mention implementation details."#;

const PERIODIC_WAKEUP: Duration = Duration::from_secs(1);
const NREM_FROM_COGNITION: f32 = 0.06;
const NREM_FROM_ELAPSED_SEC: f32 = 0.002;
const NREM_RELIEF_PER_REMEMBER_TOKEN: f32 = 0.25;
const REM_FROM_REMEMBER_TOKEN: f32 = 0.20;
const REM_RELIEF_PER_RECOMBINATION: f32 = 0.35;
const WAKE_AROUSAL_FROM_COGNITION: f32 = 0.03;
const WAKE_AROUSAL_DECAY_PER_SEC: f32 = 0.02;
const QUIET_WAKE_AROUSAL_DECAY_PER_SEC: f32 = 0.08;
const QUIET_AFFECT_AROUSAL_DECAY_PER_SEC: f32 = 0.08;
const WAKE_AROUSAL_WAKE_LEVEL: f32 = 0.70;
const WAKE_AROUSAL_SLEEP_LEVEL: f32 = 0.25;

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ArousalEffectLevel {
    Off,
    Minimal,
    Low,
    Normal,
    High,
    Max,
}

impl From<ArousalEffectLevel> for AllocationEffectLevel {
    fn from(level: ArousalEffectLevel) -> Self {
        match level {
            ArousalEffectLevel::Off => Self::Off,
            ArousalEffectLevel::Minimal => Self::Minimal,
            ArousalEffectLevel::Low => Self::Low,
            ArousalEffectLevel::Normal => Self::Normal,
            ArousalEffectLevel::High => Self::High,
            ArousalEffectLevel::Max => Self::Max,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct AffectAssessment {
    pub wake_arousal_increase: ArousalEffectLevel,
    pub affect_arousal_increase: ArousalEffectLevel,
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
    allocation_writer: AllocationWriter,
    interoception: InteroceptiveWriter,
    policy: InteroceptionRuntimePolicy,
    llm: LlmAccess,
    session: Session,
    last_seen_cognition_index: Option<u64>,
    last_total_remember_tokens: Option<u32>,
    last_activity_at: Option<DateTime<Utc>>,
}

impl InteroceptionModule {
    pub fn new(
        memo_updates: MemoUpdatedInbox,
        cognition_updates: CognitionLogUpdatedInbox,
        allocation_updates: AllocationUpdatedInbox,
        blackboard: BlackboardReader,
        allocation_writer: AllocationWriter,
        policy: InteroceptionRuntimePolicy,
        interoception: InteroceptiveWriter,
        llm: LlmAccess,
    ) -> Self {
        Self {
            memo_updates,
            cognition_updates,
            allocation_updates,
            blackboard,
            allocation_writer,
            policy,
            interoception,
            llm,
            session: Session::new(),
            last_seen_cognition_index: None,
            last_total_remember_tokens: None,
            last_activity_at: None,
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
        let sensory_activity = unread_memos
            .iter()
            .any(|record| record.owner.module == builtin::sensory());
        let cognitive_activity = non_dream_entries > 0;
        let activity = sensory_activity || cognitive_activity;
        let previous_activity_at = self.last_activity_at.unwrap_or_else(|| cx.now());
        let quiet_for = if activity {
            Duration::ZERO
        } else {
            (cx.now() - previous_activity_at)
                .to_std()
                .unwrap_or(Duration::ZERO)
        };
        if self.last_activity_at.is_none() || activity {
            self.last_activity_at = Some(cx.now());
        }

        let mut patch = next_interoception_patch(
            &current,
            cx.now(),
            non_dream_entries,
            recombination_entries,
            remember_delta,
            quiet_for,
            &self.policy,
        );
        if batch.affect_candidate && (!unread_memos.is_empty() || !unread_cognition.is_empty()) {
            let affect = self
                .estimate_affect(&current, &unread_memos, &unread_cognition, &allocation)
                .await?;
            merge_affect_patch(&mut patch, &current, affect, &self.policy);
        }
        let mut next_state = current.clone();
        next_state.apply_patch(patch.clone(), cx.now());
        self.interoception.update(patch).await;
        self.emit_suppression(&next_state).await;
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
        wake_arousal_increase: assessment.wake_arousal_increase,
        affect_arousal_increase: assessment.affect_arousal_increase,
        valence: clamp_signed_unit(assessment.valence),
        emotion: assessment.emotion.trim().to_owned(),
    }
}

fn merge_affect_patch(
    patch: &mut InteroceptivePatch,
    current: &InteroceptiveState,
    affect: AffectAssessment,
    policy: &InteroceptionRuntimePolicy,
) {
    let base_wake = patch.wake_arousal.unwrap_or(current.wake_arousal);
    let base_affect = patch.affect_arousal.unwrap_or(current.affect_arousal);
    let wake_arousal =
        clamp_unit(base_wake + policy.wake_increase_for(affect.wake_arousal_increase.into()));
    patch.wake_arousal = Some(wake_arousal);
    patch.affect_arousal = Some(clamp_unit(
        base_affect + policy.affect_increase_for(affect.affect_arousal_increase.into()),
    ));
    if wake_arousal >= WAKE_AROUSAL_WAKE_LEVEL {
        patch.mode = Some(InteroceptiveMode::Wake);
    }
    patch.valence = Some(affect.valence);
    patch.emotion = Some(affect.emotion);
}

fn next_interoception_patch(
    current: &InteroceptiveState,
    now: DateTime<Utc>,
    non_dream_entries: u32,
    recombination_entries: u32,
    remember_delta: u32,
    quiet_for: Duration,
    policy: &InteroceptionRuntimePolicy,
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
    let quiet_excess_secs = quiet_for
        .saturating_sub(policy.quiet_sleep_threshold)
        .as_secs_f32()
        .min(120.0);
    let wake_arousal = current.wake_arousal
        + non_dream_entries as f32 * WAKE_AROUSAL_FROM_COGNITION
        - elapsed_secs * WAKE_AROUSAL_DECAY_PER_SEC
        - quiet_excess_secs
            * QUIET_WAKE_AROUSAL_DECAY_PER_SEC
            * policy.wake_arousal_change_multiplier;
    let affect_arousal = current.affect_arousal
        - quiet_excess_secs
            * QUIET_AFFECT_AROUSAL_DECAY_PER_SEC
            * policy.affect_arousal_change_multiplier;
    let nrem_pressure = clamp_unit(nrem_pressure);
    let rem_pressure = clamp_unit(rem_pressure);
    let wake_arousal = clamp_unit(wake_arousal);
    let affect_arousal = clamp_unit(affect_arousal);
    let mode = if wake_arousal >= WAKE_AROUSAL_WAKE_LEVEL {
        InteroceptiveMode::Wake
    } else if nrem_pressure >= 0.7
        || (quiet_excess_secs > 0.0 && wake_arousal <= WAKE_AROUSAL_SLEEP_LEVEL)
    {
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
        affect_arousal: Some(affect_arousal),
        valence: None,
        emotion: None,
    }
}

impl InteroceptionModule {
    async fn emit_suppression(&self, state: &InteroceptiveState) {
        let level = suppression_level(state);
        let commands = self
            .allocation_writer
            .allowed_suppression_modules()
            .iter()
            .cloned()
            .map(|id| AllocationCommand::suppression(id, level))
            .collect::<Vec<_>>();
        self.allocation_writer.submit(commands).await;
    }
}

fn suppression_level(state: &InteroceptiveState) -> AllocationEffectLevel {
    if state.wake_arousal >= 0.70 {
        AllocationEffectLevel::Off
    } else if state.wake_arousal >= 0.45 {
        AllocationEffectLevel::Low
    } else if state.wake_arousal >= 0.25 {
        AllocationEffectLevel::Normal
    } else if matches!(
        state.mode,
        InteroceptiveMode::NremPressure | InteroceptiveMode::RemPressure
    ) {
        AllocationEffectLevel::Max
    } else {
        AllocationEffectLevel::High
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
        let policy = InteroceptionRuntimePolicy::default();
        let current = InteroceptiveState {
            last_updated: DateTime::<Utc>::from_timestamp(0, 0).unwrap(),
            ..InteroceptiveState::default()
        };

        let patch = next_interoception_patch(&current, now, 12, 0, 0, Duration::ZERO, &policy);
        assert_eq!(patch.mode, Some(InteroceptiveMode::NremPressure));
        assert!(patch.nrem_pressure.unwrap() >= 0.7);

        let compacted = InteroceptiveState {
            nrem_pressure: 0.8,
            rem_pressure: 0.0,
            last_updated: now,
            ..InteroceptiveState::default()
        };
        let patch = next_interoception_patch(&compacted, now, 0, 0, 3, Duration::ZERO, &policy);
        assert!(patch.nrem_pressure.unwrap() <= 0.3);
        assert!(patch.rem_pressure.unwrap() > 0.4);

        let recombined = InteroceptiveState {
            rem_pressure: 0.7,
            last_updated: now,
            ..InteroceptiveState::default()
        };
        let patch = next_interoception_patch(&recombined, now, 0, 2, 0, Duration::ZERO, &policy);
        assert!(patch.rem_pressure.unwrap() <= 0.1);
    }

    #[test]
    fn affect_assessment_is_normalized() {
        let normalized = normalize_affect_assessment(AffectAssessment {
            wake_arousal_increase: ArousalEffectLevel::High,
            affect_arousal_increase: ArousalEffectLevel::Low,
            valence: -1.5,
            emotion: "  startled relief  ".to_owned(),
        });

        assert!(matches!(
            normalized.wake_arousal_increase,
            ArousalEffectLevel::High
        ));
        assert_eq!(normalized.valence, -1.0);
        assert_eq!(normalized.emotion, "startled relief");
    }

    #[test]
    fn quiet_period_decays_wake_and_affect_arousal() {
        let now = DateTime::<Utc>::from_timestamp(40, 0).unwrap();
        let policy = InteroceptionRuntimePolicy {
            quiet_sleep_threshold: Duration::from_secs(5),
            wake_arousal_change_multiplier: 2.0,
            affect_arousal_change_multiplier: 2.0,
        };
        let current = InteroceptiveState {
            wake_arousal: 0.8,
            affect_arousal: 0.7,
            last_updated: DateTime::<Utc>::from_timestamp(39, 0).unwrap(),
            ..InteroceptiveState::default()
        };

        let patch =
            next_interoception_patch(&current, now, 0, 0, 0, Duration::from_secs(10), &policy);

        assert!(patch.wake_arousal.unwrap() < 0.1);
        assert!(patch.affect_arousal.unwrap() < 0.1);
        assert_eq!(patch.mode, Some(InteroceptiveMode::NremPressure));
    }

    #[test]
    fn salient_affect_assessment_raises_arousal_from_levels() {
        let policy = InteroceptionRuntimePolicy {
            wake_arousal_change_multiplier: 2.0,
            ..InteroceptionRuntimePolicy::default()
        };
        let current = InteroceptiveState {
            wake_arousal: 0.2,
            affect_arousal: 0.1,
            ..InteroceptiveState::default()
        };
        let mut patch = InteroceptivePatch {
            wake_arousal: Some(0.2),
            affect_arousal: Some(0.1),
            ..InteroceptivePatch::default()
        };

        merge_affect_patch(
            &mut patch,
            &current,
            AffectAssessment {
                wake_arousal_increase: ArousalEffectLevel::Max,
                affect_arousal_increase: ArousalEffectLevel::High,
                valence: 0.4,
                emotion: "alert".to_owned(),
            },
            &policy,
        );

        assert!(patch.wake_arousal.unwrap() > 0.7);
        assert!(patch.affect_arousal.unwrap() > 0.2);
        assert_eq!(patch.mode, Some(InteroceptiveMode::Wake));
        assert_eq!(patch.valence, Some(0.4));
        assert_eq!(patch.emotion.as_deref(), Some("alert"));
    }
}
