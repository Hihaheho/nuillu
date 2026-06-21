use std::time::Duration;

use anyhow::{Context, Result};
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use lutum::{Session, TextStepOutcomeWithTools, ToolResult, Usage};
use nuillu_blackboard::{
    AllocationEffectLevel, CognitionLogEntryRecord, InteroceptiveMode, InteroceptivePatch,
    InteroceptiveState, MemoLogRecord,
};
use nuillu_module::{
    BlackboardReader, CognitionLogUpdatedInbox, InteroceptionRuntimePolicy, InteroceptiveWriter,
    LlmAccess, LlmContextWindow, MemoUpdatedInbox, Module, SessionAutoCompaction,
    SessionCompactionConfig, SessionCompactionProtectedPrefix, ensure_persistent_session_seeded,
    format_bounded_cognition_log_batch, format_bounded_memo_log_batch, format_policy_system_prompt,
};
use nuillu_types::builtin;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

const SYSTEM_PROMPT: &str = r#"You are the interoception module.
Maintain the agent's internal state from the recent cognitive workspace. Estimate affect as
current internal condition, not a moral judgment and not a memory fact.
For arousal and valence, return structured semantic labels rather than numeric deltas. Use higher
wake and affect salience only when the unread sensory/cognitive evidence is enough to wake or
emotionally activate the agent. Return valence polarity/salience and one short untyped emotion phrase.
Use neutral values when evidence is weak. Do not mention implementation details."#;

const PERIODIC_WAKEUP: Duration = Duration::from_secs(1);
const NREM_FROM_COGNITION: f32 = 0.06;
const NREM_FROM_ELAPSED_SEC: f32 = 0.002;
const NREM_RELIEF_PER_REMEMBER_TOKEN: f32 = 0.25;
const REM_FROM_REMEMBER_TOKEN: f32 = 0.20;
const REM_RELIEF_PER_RECOMBINATION: f32 = 0.35;
const REM_DECAY_PER_SEC: f32 = 0.004;
const WAKE_AROUSAL_FROM_COGNITION: f32 = 0.03;
const WAKE_AROUSAL_DECAY_PER_SEC: f32 = 0.02;
const AFFECT_AROUSAL_DECAY_PER_SEC: f32 = 0.015;
const QUIET_WAKE_AROUSAL_DECAY_PER_SEC: f32 = 0.08;
const QUIET_AFFECT_AROUSAL_DECAY_PER_SEC: f32 = 0.08;
const VALENCE_RETURN_PER_SEC: f32 = 0.02;
const SETTLED_AFFECT_AROUSAL: f32 = 0.05;
const SETTLED_VALENCE: f32 = 0.05;
const WAKE_AROUSAL_WAKE_LEVEL: f32 = 0.70;
const WAKE_AROUSAL_SLEEP_LEVEL: f32 = 0.25;
const MEMO_CONTEXT_WINDOW: LlmContextWindow = LlmContextWindow::new(8, 800, 3_000);
const COGNITION_CONTEXT_WINDOW: LlmContextWindow = LlmContextWindow::new(8, 600, 3_000);
const COMPACTED_INTEROCEPTION_SESSION_PREFIX: &str = "Compacted interoception session history:";
const SESSION_COMPACTION_FOCUS: &str = r#"Preserve affect-state judgments, salient evidence, and
wake/sleep pressure rationale future interoception decisions need."#;

pub fn session_auto_compaction() -> SessionAutoCompaction {
    SessionAutoCompaction::new(
        SessionCompactionConfig::default(),
        SessionCompactionProtectedPrefix::LeadingSystemAndIdentitySeed,
        COMPACTED_INTEROCEPTION_SESSION_PREFIX,
        SESSION_COMPACTION_FOCUS,
    )
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
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

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ValencePolarity {
    Negative,
    Neutral,
    Positive,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ValenceEffectLevel {
    Off,
    Minimal,
    Low,
    Normal,
    High,
    Max,
}

impl From<ValenceEffectLevel> for AllocationEffectLevel {
    fn from(level: ValenceEffectLevel) -> Self {
        match level {
            ValenceEffectLevel::Off => Self::Off,
            ValenceEffectLevel::Minimal => Self::Minimal,
            ValenceEffectLevel::Low => Self::Low,
            ValenceEffectLevel::Normal => Self::Normal,
            ValenceEffectLevel::High => Self::High,
            ValenceEffectLevel::Max => Self::Max,
        }
    }
}

#[lutum::tool_input(name = "report_affect", output = ReportAffectOutput)]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct AffectAssessment {
    pub wake_salience: ArousalEffectLevel,
    pub affect_salience: ArousalEffectLevel,
    pub valence_polarity: ValencePolarity,
    pub valence_salience: ValenceEffectLevel,
    pub emotion: String,
}

#[derive(Clone, Debug, PartialEq)]
struct AffectAppraisal {
    wake_salience: ArousalEffectLevel,
    affect_salience: ArousalEffectLevel,
    valence: Option<ValenceAppraisal>,
    emotion: Option<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ValenceAppraisal {
    polarity: ValencePolarity,
    salience: ValenceEffectLevel,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ActivitySignals {
    sensory_activity: bool,
    non_dream_entries: u32,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct ReportAffectOutput {
    pub accepted: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema, lutum::Toolset)]
#[allow(clippy::large_enum_variant)]
pub enum InteroceptionTools {
    ReportAffect(AffectAssessment),
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct InteroceptionBatch {
    affect_candidate: bool,
}

pub struct InteroceptionModule {
    memo_updates: MemoUpdatedInbox,
    cognition_updates: CognitionLogUpdatedInbox,
    blackboard: BlackboardReader,
    interoception: InteroceptiveWriter,
    policy: InteroceptionRuntimePolicy,
    llm: LlmAccess,
    session: Session,
    system_prompt: std::sync::OnceLock<String>,
    last_seen_cognition_index: Option<u64>,
    last_total_remember_tokens: Option<u32>,
    last_activity_at: Option<DateTime<Utc>>,
}

impl InteroceptionModule {
    pub fn new(
        memo_updates: MemoUpdatedInbox,
        cognition_updates: CognitionLogUpdatedInbox,
        blackboard: BlackboardReader,
        policy: InteroceptionRuntimePolicy,
        interoception: InteroceptiveWriter,
        llm: LlmAccess,
        session: Session,
    ) -> Self {
        Self {
            memo_updates,
            cognition_updates,
            blackboard,
            policy,
            interoception,
            llm,
            session,
            system_prompt: std::sync::OnceLock::new(),
            last_seen_cognition_index: None,
            last_total_remember_tokens: None,
            last_activity_at: None,
        }
    }

    fn system_prompt(&self, cx: &nuillu_module::ActivateCx<'_>) -> &str {
        self.system_prompt
            .get_or_init(|| format_policy_system_prompt(SYSTEM_PROMPT, cx.core_policies()))
    }

    fn ensure_session_seeded(&mut self, cx: &nuillu_module::ActivateCx<'_>) {
        let system_prompt = self.system_prompt(cx).to_owned();
        ensure_persistent_session_seeded(
            &mut self.session,
            system_prompt,
            cx.identity_memories(),
            cx.now(),
        );
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
        let (current, unread_cognition, total_remember_tokens) = self
            .blackboard
            .read(|bb| {
                let unread = bb.unread_cognition_log_entries(self.last_seen_cognition_index);
                let total = bb
                    .memory_metadata()
                    .values()
                    .map(|metadata| metadata.remember_tokens)
                    .sum::<u32>();
                (bb.interoception().clone(), unread, total)
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
        let signals = ActivitySignals {
            sensory_activity,
            non_dream_entries,
        };
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
            let mut appraisal = self
                .estimate_affect(cx, &current, &unread_memos, &unread_cognition)
                .await?
                .map(AffectAppraisal::from)
                .unwrap_or_else(|| fallback_affect_appraisal(signals));
            apply_activity_arousal_floor(&mut appraisal, signals);
            merge_affect_patch(&mut patch, &current, appraisal, &self.policy);
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
            _ = tokio::time::sleep(PERIODIC_WAKEUP) => false,
        };
        affect_candidate |= !self.memo_updates.take_ready_items()?.items.is_empty();
        affect_candidate |= !self.cognition_updates.take_ready_items()?.items.is_empty();
        Ok(InteroceptionBatch { affect_candidate })
    }

    async fn estimate_affect(
        &mut self,
        cx: &nuillu_module::ActivateCx<'_>,
        current: &InteroceptiveState,
        unread_memos: &[MemoLogRecord],
        unread_cognition: &[CognitionLogEntryRecord],
    ) -> Result<Option<AffectAssessment>> {
        self.ensure_session_seeded(cx);
        let lutum = self.llm.lutum().await;
        let session_len_before_turn = self.session.input().items().len();
        self.session.push_user(format_affect_assessment_input(
            current,
            unread_memos,
            unread_cognition,
            cx.now(),
        ));

        let outcome = match self
            .session
            .text_turn()
            .tools::<InteroceptionTools>()
            .available_tools([InteroceptionToolsSelector::ReportAffect])
            .require_any_tool()
            .max_output_tokens(256)
            .collect_controlled_with(&lutum, nuillu_module::AbortOnAvailableToolNameInText::new())
            .await
        {
            Ok(outcome) => outcome,
            Err(error) => {
                tracing::warn!(error = %error, "interoception affect tool turn failed; using fallback");
                truncate_session_to(&mut self.session, session_len_before_turn);
                cx.compact_and_save(&mut self.session, Usage::zero())
                    .await?;
                return Ok(None);
            }
        };

        let assessment = match outcome {
            TextStepOutcomeWithTools::Finished(result) => {
                tracing::warn!(
                    input_tokens = result.usage.input_tokens,
                    "interoception affect turn finished without report_affect; using fallback"
                );
                truncate_session_to(&mut self.session, session_len_before_turn);
                cx.compact_and_save(&mut self.session, result.usage).await?;
                return Ok(None);
            }
            TextStepOutcomeWithTools::FinishedNoOutput(result) => {
                tracing::warn!(
                    input_tokens = result.usage.input_tokens,
                    "interoception affect turn finished without report_affect; using fallback"
                );
                truncate_session_to(&mut self.session, session_len_before_turn);
                cx.compact_and_save(&mut self.session, result.usage).await?;
                return Ok(None);
            }
            TextStepOutcomeWithTools::NeedsTools(round) => {
                let usage = round.usage;
                let mut selected = None;
                let mut results: Vec<ToolResult> = Vec::new();
                nuillu_module::emit_trace_tool_calls(&round.tool_calls);
                for call in round.tool_calls.iter().cloned() {
                    match call {
                        InteroceptionToolsCall::ReportAffect(call) => {
                            let accepted = selected.is_none();
                            if accepted {
                                selected = Some(call.input.clone());
                            }
                            results.push(
                                call.complete(ReportAffectOutput { accepted })
                                    .context("complete report_affect tool call")?,
                            );
                        }
                    }
                }
                let Some(assessment) = selected else {
                    tracing::warn!(
                        "interoception affect turn returned NeedsTools without report_affect; using fallback"
                    );
                    round.discard();
                    truncate_session_to(&mut self.session, session_len_before_turn);
                    cx.compact_and_save(&mut self.session, usage).await?;
                    return Ok(None);
                };
                if let Err(error) = round.commit(&mut self.session, results) {
                    tracing::warn!(
                        error = %error,
                        "interoception affect tool round commit failed; using fallback"
                    );
                    truncate_session_to(&mut self.session, session_len_before_turn);
                    cx.compact_and_save(&mut self.session, usage).await?;
                    return Ok(None);
                }
                cx.compact_and_save(&mut self.session, usage).await?;
                assessment
            }
        };
        Ok(Some(normalize_affect_assessment(assessment)))
    }
}

fn format_affect_assessment_input(
    current: &InteroceptiveState,
    unread_memos: &[MemoLogRecord],
    unread_cognition: &[CognitionLogEntryRecord],
    now: DateTime<Utc>,
) -> String {
    format!(
        "Interoception affect appraisal request\n\nCurrent interoceptive state:\n{}\n\nUnread memo evidence:\n{}\n\nUnread cognition evidence:\n{}\n\nInstruction:\nCall report_affect once. Use structured salience levels and valence polarity, not numeric arousal or valence deltas. Use off/no-change salience, neutral polarity, and preserve current emotion when evidence is weak.",
        format_interoceptive_state(current),
        format_bounded_memo_log_batch(unread_memos, now, MEMO_CONTEXT_WINDOW)
            .unwrap_or_else(|| "none".to_owned()),
        format_bounded_cognition_log_batch(unread_cognition, now, COGNITION_CONTEXT_WINDOW)
            .unwrap_or_else(|| "none".to_owned()),
    )
}

fn format_interoceptive_state(state: &InteroceptiveState) -> String {
    format!(
        "- mode: {:?}\n- wake_arousal: {:.2}\n- nrem_pressure: {:.2}\n- rem_pressure: {:.2}\n- affect_arousal: {:.2}\n- valence: {:.2}\n- emotion: {}\n- last_updated: {}",
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
        },
        state.last_updated.to_rfc3339(),
    )
}

fn truncate_session_to(session: &mut Session, len: usize) {
    session.input_mut().items_mut().truncate(len);
}

impl From<AffectAssessment> for AffectAppraisal {
    fn from(assessment: AffectAssessment) -> Self {
        Self {
            wake_salience: assessment.wake_salience,
            affect_salience: assessment.affect_salience,
            valence: Some(ValenceAppraisal {
                polarity: assessment.valence_polarity,
                salience: assessment.valence_salience,
            }),
            emotion: Some(assessment.emotion),
        }
    }
}

fn fallback_affect_appraisal(signals: ActivitySignals) -> AffectAppraisal {
    let (wake_salience, affect_salience) = activity_arousal_floor(signals);
    AffectAppraisal {
        wake_salience,
        affect_salience,
        valence: None,
        emotion: None,
    }
}

fn apply_activity_arousal_floor(appraisal: &mut AffectAppraisal, signals: ActivitySignals) {
    let (wake_floor, affect_floor) = activity_arousal_floor(signals);
    appraisal.wake_salience = max_arousal_level(appraisal.wake_salience, wake_floor);
    appraisal.affect_salience = max_arousal_level(appraisal.affect_salience, affect_floor);
}

fn activity_arousal_floor(signals: ActivitySignals) -> (ArousalEffectLevel, ArousalEffectLevel) {
    let wake_salience = if signals.sensory_activity || signals.non_dream_entries > 0 {
        ArousalEffectLevel::Low
    } else {
        ArousalEffectLevel::Off
    };
    let affect_salience = if signals.non_dream_entries >= 2 {
        ArousalEffectLevel::Low
    } else if signals.sensory_activity || signals.non_dream_entries > 0 {
        ArousalEffectLevel::Minimal
    } else {
        ArousalEffectLevel::Off
    };
    (wake_salience, affect_salience)
}

fn max_arousal_level(left: ArousalEffectLevel, right: ArousalEffectLevel) -> ArousalEffectLevel {
    if arousal_level_rank(left) >= arousal_level_rank(right) {
        left
    } else {
        right
    }
}

fn arousal_level_rank(level: ArousalEffectLevel) -> u8 {
    match level {
        ArousalEffectLevel::Off => 0,
        ArousalEffectLevel::Minimal => 1,
        ArousalEffectLevel::Low => 2,
        ArousalEffectLevel::Normal => 3,
        ArousalEffectLevel::High => 4,
        ArousalEffectLevel::Max => 5,
    }
}

fn normalize_affect_assessment(assessment: AffectAssessment) -> AffectAssessment {
    AffectAssessment {
        wake_salience: assessment.wake_salience,
        affect_salience: assessment.affect_salience,
        valence_polarity: assessment.valence_polarity,
        valence_salience: assessment.valence_salience,
        emotion: assessment.emotion.trim().to_owned(),
    }
}

fn merge_affect_patch(
    patch: &mut InteroceptivePatch,
    current: &InteroceptiveState,
    appraisal: AffectAppraisal,
    policy: &InteroceptionRuntimePolicy,
) {
    let base_wake = patch.wake_arousal.unwrap_or(current.wake_arousal);
    let base_affect = patch.affect_arousal.unwrap_or(current.affect_arousal);
    let wake_arousal =
        clamp_unit(base_wake + policy.wake_increase_for(appraisal.wake_salience.into()));
    patch.wake_arousal = Some(wake_arousal);
    patch.affect_arousal = Some(clamp_unit(
        base_affect + policy.affect_increase_for(appraisal.affect_salience.into()),
    ));
    if wake_arousal >= WAKE_AROUSAL_WAKE_LEVEL {
        patch.mode = Some(InteroceptiveMode::Wake);
    }
    if let Some(valence) = appraisal.valence {
        let base_valence = patch.valence.unwrap_or(current.valence);
        patch.valence = Some(clamp_signed_unit(
            base_valence + signed_valence_delta(valence, policy),
        ));
    }
    if let Some(emotion) = appraisal.emotion {
        patch.emotion = Some(emotion);
    }
}

fn signed_valence_delta(appraisal: ValenceAppraisal, policy: &InteroceptionRuntimePolicy) -> f32 {
    let magnitude = policy.valence_change_for(appraisal.salience.into());
    match appraisal.polarity {
        ValencePolarity::Negative => -magnitude,
        ValencePolarity::Neutral => 0.0,
        ValencePolarity::Positive => magnitude,
    }
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
        - recombination_entries as f32 * REM_RELIEF_PER_RECOMBINATION
        - elapsed_secs * REM_DECAY_PER_SEC;
    let quiet_excess_secs = quiet_for
        .saturating_sub(policy.quiet_sleep_threshold)
        .as_secs_f32()
        .min(120.0);
    let quiet_secs = quiet_for.as_secs_f32().min(120.0);
    let wake_arousal = current.wake_arousal
        + non_dream_entries as f32 * WAKE_AROUSAL_FROM_COGNITION
        - elapsed_secs * WAKE_AROUSAL_DECAY_PER_SEC
        - quiet_excess_secs
            * QUIET_WAKE_AROUSAL_DECAY_PER_SEC
            * policy.wake_arousal_change_multiplier;
    let affect_arousal = current.affect_arousal
        - if quiet_secs > 0.0 {
            elapsed_secs * AFFECT_AROUSAL_DECAY_PER_SEC * policy.affect_arousal_change_multiplier
        } else {
            0.0
        }
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
    let returned_valence = if quiet_secs > 0.0 {
        Some(return_toward_zero(
            current.valence,
            elapsed_secs * VALENCE_RETURN_PER_SEC,
        ))
    } else {
        None
    };
    let cleared_emotion = returned_valence.and_then(|valence| {
        if affect_arousal <= SETTLED_AFFECT_AROUSAL
            && valence.abs() <= SETTLED_VALENCE
            && !current.emotion.trim().is_empty()
        {
            Some(String::new())
        } else {
            None
        }
    });

    InteroceptivePatch {
        mode: Some(mode),
        wake_arousal: Some(wake_arousal),
        nrem_pressure: Some(nrem_pressure),
        rem_pressure: Some(rem_pressure),
        affect_arousal: Some(affect_arousal),
        valence: returned_valence,
        emotion: cleared_emotion,
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

fn return_toward_zero(value: f32, step: f32) -> f32 {
    if value.abs() <= step {
        0.0
    } else if value.is_sign_positive() {
        value - step
    } else {
        value + step
    }
}

#[async_trait(?Send)]
impl Module for InteroceptionModule {
    type Batch = InteroceptionBatch;

    fn id() -> &'static str {
        "interoception"
    }

    fn peer_context() -> Option<&'static str> {
        None
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
    use std::rc::Rc;
    use std::sync::{Arc, Mutex};

    use lutum::{
        AdapterStructuredTurn, AdapterTextTurn, AgentError, AssistantInputItem,
        ErasedStructuredTurnEventStream, ErasedTextTurnEventStream, FinishReason, Lutum,
        MessageContent, MockLlmAdapter, MockTextScenario, ModelInput, ModelInputItem,
        RawTextTurnEvent, SharedPoolBudgetManager, SharedPoolBudgetOptions, TurnAdapter, Usage,
    };
    use nuillu_blackboard::{Blackboard, Bpm, linear_ratio_fn};
    use nuillu_module::ports::{NoopCognitionLogRepository, SystemClock};
    use nuillu_module::{
        CapabilityProviderPorts, CapabilityProviders, LlmConcurrencyLimiter, LutumTiers,
        ModuleRegistry, SessionCompactionPolicy, SessionCompactionRuntime,
    };
    use nuillu_types::{ModelTier, ModuleInstanceId, ReplicaCapRange, ReplicaIndex};

    use super::*;

    #[derive(Clone)]
    struct CapturingTextAdapter {
        inner: MockLlmAdapter,
        text_inputs: Arc<Mutex<Vec<ModelInput>>>,
    }

    impl CapturingTextAdapter {
        fn new(inner: MockLlmAdapter) -> Self {
            Self {
                inner,
                text_inputs: Arc::new(Mutex::new(Vec::new())),
            }
        }

        fn text_inputs(&self) -> Vec<ModelInput> {
            self.text_inputs.lock().unwrap().clone()
        }
    }

    #[async_trait::async_trait]
    impl TurnAdapter for CapturingTextAdapter {
        async fn text_turn(
            &self,
            input: ModelInput,
            turn: AdapterTextTurn,
        ) -> Result<ErasedTextTurnEventStream, AgentError> {
            self.text_inputs.lock().unwrap().push(input.clone());
            self.inner.text_turn(input, turn).await
        }

        async fn structured_turn(
            &self,
            input: ModelInput,
            turn: AdapterStructuredTurn,
        ) -> Result<ErasedStructuredTurnEventStream, AgentError> {
            self.inner.structured_turn(input, turn).await
        }
    }

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
    fn rem_pressure_naturally_decays_without_recombination_output() {
        let now = DateTime::<Utc>::from_timestamp(10, 0).unwrap();
        let policy = InteroceptionRuntimePolicy::default();
        let current = InteroceptiveState {
            rem_pressure: 0.5,
            last_updated: DateTime::<Utc>::from_timestamp(0, 0).unwrap(),
            ..InteroceptiveState::default()
        };

        let patch = next_interoception_patch(&current, now, 0, 0, 0, Duration::ZERO, &policy);

        assert!(patch.rem_pressure.unwrap() < 0.5);
        assert!(patch.rem_pressure.unwrap() > 0.4);
    }

    #[test]
    fn affect_assessment_is_normalized() {
        let normalized = normalize_affect_assessment(AffectAssessment {
            wake_salience: ArousalEffectLevel::High,
            affect_salience: ArousalEffectLevel::Low,
            valence_polarity: ValencePolarity::Negative,
            valence_salience: ValenceEffectLevel::High,
            emotion: "  startled relief  ".to_owned(),
        });

        assert!(matches!(normalized.wake_salience, ArousalEffectLevel::High));
        assert_eq!(normalized.valence_polarity, ValencePolarity::Negative);
        assert_eq!(normalized.valence_salience, ValenceEffectLevel::High);
        assert_eq!(normalized.emotion, "startled relief");
    }

    #[test]
    fn activity_arousal_floor_overrides_under_called_salience_only() {
        let mut appraisal = AffectAppraisal {
            wake_salience: ArousalEffectLevel::Off,
            affect_salience: ArousalEffectLevel::Off,
            valence: Some(ValenceAppraisal {
                polarity: ValencePolarity::Negative,
                salience: ValenceEffectLevel::Normal,
            }),
            emotion: Some("uneasy".to_owned()),
        };

        apply_activity_arousal_floor(
            &mut appraisal,
            ActivitySignals {
                sensory_activity: true,
                non_dream_entries: 0,
            },
        );

        assert_eq!(appraisal.wake_salience, ArousalEffectLevel::Low);
        assert_eq!(appraisal.affect_salience, ArousalEffectLevel::Minimal);
        assert_eq!(
            appraisal.valence,
            Some(ValenceAppraisal {
                polarity: ValencePolarity::Negative,
                salience: ValenceEffectLevel::Normal,
            })
        );
        assert_eq!(appraisal.emotion.as_deref(), Some("uneasy"));
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
    fn quiet_period_returns_valence_to_neutral_and_clears_emotion() {
        let now = DateTime::<Utc>::from_timestamp(10, 0).unwrap();
        let policy = InteroceptionRuntimePolicy::default();
        let current = InteroceptiveState {
            affect_arousal: 0.1,
            valence: -0.1,
            emotion: "tense".to_owned(),
            last_updated: DateTime::<Utc>::from_timestamp(0, 0).unwrap(),
            ..InteroceptiveState::default()
        };

        let patch =
            next_interoception_patch(&current, now, 0, 0, 0, Duration::from_secs(10), &policy);

        assert_eq!(patch.affect_arousal, Some(0.0));
        assert_eq!(patch.valence, Some(0.0));
        assert_eq!(patch.emotion.as_deref(), Some(""));
    }

    #[test]
    fn salient_affect_assessment_raises_arousal_from_levels() {
        let policy = InteroceptionRuntimePolicy {
            wake_arousal_change_multiplier: 2.0,
            affect_arousal_change_multiplier: 2.0,
            ..InteroceptionRuntimePolicy::default()
        };
        let current = InteroceptiveState {
            wake_arousal: 0.2,
            affect_arousal: 0.1,
            valence: -0.2,
            ..InteroceptiveState::default()
        };
        let mut patch = InteroceptivePatch {
            wake_arousal: Some(0.2),
            affect_arousal: Some(0.1),
            valence: Some(-0.1),
            ..InteroceptivePatch::default()
        };

        merge_affect_patch(
            &mut patch,
            &current,
            AffectAppraisal {
                wake_salience: ArousalEffectLevel::Max,
                affect_salience: ArousalEffectLevel::High,
                valence: Some(ValenceAppraisal {
                    polarity: ValencePolarity::Positive,
                    salience: ValenceEffectLevel::Normal,
                }),
                emotion: Some("alert".to_owned()),
            },
            &policy,
        );

        assert!(patch.wake_arousal.unwrap() > 0.7);
        assert!(patch.affect_arousal.unwrap() > 0.2);
        assert_eq!(patch.mode, Some(InteroceptiveMode::Wake));
        assert!(patch.valence.unwrap() > 0.0);
        assert_eq!(patch.emotion.as_deref(), Some("alert"));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn successful_affect_tool_updates_affect_fields() {
        let adapter = MockLlmAdapter::new().with_text_scenario(report_affect_scenario(
            serde_json::json!({
                "wake_salience": "high",
                "affect_salience": "normal",
                "valence_polarity": "positive",
                "valence_salience": "normal",
                "emotion": "alert curiosity",
                "ignored_extra": "ok"
            })
            .to_string(),
        ));
        let blackboard = Blackboard::default();
        let (caps, lutum) = test_caps_with_adapter(blackboard.clone(), Arc::new(adapter));
        let mut module = build_interoception_module(&caps).await;

        let now = DateTime::<Utc>::from_timestamp(100, 0).unwrap();
        publish_interoception_wakeup(&caps, &blackboard, now).await;
        let cx = activate_cx(&lutum, now);
        let batch = module.next_batch().await.unwrap();
        module.activate(&cx, &batch).await.unwrap();

        let state = blackboard.read(|bb| bb.interoception().clone()).await;
        assert!(state.valence > 0.0);
        assert_eq!(state.emotion, "alert curiosity");
        assert!(state.affect_arousal > 0.0);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn successful_affect_tool_persists_readable_evidence_for_next_turn() {
        let adapter = MockLlmAdapter::new()
            .with_text_scenario(report_affect_scenario(
                serde_json::json!({
                    "wake_salience": "low",
                    "affect_salience": "low",
                    "valence_polarity": "positive",
                    "valence_salience": "low",
                    "emotion": "curious"
                })
                .to_string(),
            ))
            .with_text_scenario(report_affect_scenario(
                serde_json::json!({
                    "wake_salience": "minimal",
                    "affect_salience": "minimal",
                    "valence_polarity": "positive",
                    "valence_salience": "minimal",
                    "emotion": "settled"
                })
                .to_string(),
            ));
        let capture = CapturingTextAdapter::new(adapter);
        let observed = capture.clone();
        let blackboard = Blackboard::default();
        let (caps, lutum) = test_caps_with_adapter(blackboard.clone(), Arc::new(capture));
        let mut module = build_interoception_module(&caps).await;

        let now = DateTime::<Utc>::from_timestamp(100, 0).unwrap();
        publish_interoception_wakeup_with_content(&caps, &blackboard, "first affect evidence", now)
            .await;
        let cx = activate_cx(&lutum, now);
        let batch = module.next_batch().await.unwrap();
        module.activate(&cx, &batch).await.unwrap();

        publish_interoception_wakeup_with_content(
            &caps,
            &blackboard,
            "second affect evidence",
            now + chrono::Duration::seconds(1),
        )
        .await;
        let batch = module.next_batch().await.unwrap();
        module.activate(&cx, &batch).await.unwrap();

        let inputs = observed.text_inputs();
        assert_eq!(inputs.len(), 2);
        let second_text = all_input_text(&inputs[1]);
        assert!(second_text.contains("Interoception affect appraisal request"));
        assert!(second_text.contains("Current interoceptive state:"));
        assert!(second_text.contains("- wake_arousal:"));
        assert!(second_text.contains("Unread memo evidence:"));
        assert!(second_text.contains("Unread cognition evidence:"));
        assert!(second_text.contains("Instruction:"));
        assert!(second_text.contains("first affect evidence"));
        assert!(!second_text.contains("{\""));
        assert!(
            inputs[1]
                .items()
                .iter()
                .any(|item| matches!(item, ModelInputItem::ToolResult(_)))
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn affect_tool_failures_use_fallback_without_failing_activation() {
        for scenario in [
            no_affect_tool_scenario(),
            report_affect_scenario(
                serde_json::json!({
                    "wake_salience": 0.2,
                    "affect_salience": "normal",
                    "valence_polarity": "positive",
                    "valence_salience": "normal",
                    "emotion": "alert curiosity"
                })
                .to_string(),
            ),
        ] {
            let adapter = MockLlmAdapter::new().with_text_scenario(scenario);
            let blackboard = Blackboard::default();
            let (caps, lutum) = test_caps_with_adapter(blackboard.clone(), Arc::new(adapter));
            let mut module = build_interoception_module(&caps).await;

            let now = DateTime::<Utc>::from_timestamp(100, 0).unwrap();
            publish_interoception_wakeup(&caps, &blackboard, now).await;
            let cx = activate_cx(&lutum, now);
            let batch = module.next_batch().await.unwrap();
            module.activate(&cx, &batch).await.unwrap();

            let state = blackboard.read(|bb| bb.interoception().clone()).await;
            assert!(state.wake_arousal > 0.0);
            assert!(state.affect_arousal > 0.0);
            assert_eq!(state.valence, 0.0);
            assert_eq!(state.emotion, "");
        }
    }

    #[tokio::test(flavor = "current_thread")]
    async fn failed_affect_tool_turn_does_not_persist_evidence_prompt() {
        let adapter = MockLlmAdapter::new()
            .with_text_scenario(report_affect_scenario(
                serde_json::json!({
                    "wake_salience": 0.2,
                    "affect_salience": "normal",
                    "valence_polarity": "positive",
                    "valence_salience": "normal",
                    "emotion": "alert curiosity"
                })
                .to_string(),
            ))
            .with_text_scenario(report_affect_scenario(
                serde_json::json!({
                    "wake_salience": "minimal",
                    "affect_salience": "minimal",
                    "valence_polarity": "positive",
                    "valence_salience": "minimal",
                    "emotion": "settled"
                })
                .to_string(),
            ));
        let capture = CapturingTextAdapter::new(adapter);
        let observed = capture.clone();
        let blackboard = Blackboard::default();
        let (caps, lutum) = test_caps_with_adapter(blackboard.clone(), Arc::new(capture));
        let mut module = build_interoception_module(&caps).await;

        let now = DateTime::<Utc>::from_timestamp(100, 0).unwrap();
        publish_interoception_wakeup_with_content(
            &caps,
            &blackboard,
            "failed affect evidence",
            now,
        )
        .await;
        let cx = activate_cx(&lutum, now);
        let batch = module.next_batch().await.unwrap();
        module.activate(&cx, &batch).await.unwrap();

        publish_interoception_wakeup_with_content(
            &caps,
            &blackboard,
            "successful affect evidence",
            now + chrono::Duration::seconds(1),
        )
        .await;
        let batch = module.next_batch().await.unwrap();
        module.activate(&cx, &batch).await.unwrap();

        let inputs = observed.text_inputs();
        assert_eq!(inputs.len(), 2);
        let second_text = all_input_text(&inputs[1]);
        assert!(!second_text.contains("failed affect evidence"));
        assert!(second_text.contains("successful affect evidence"));
        assert!(
            !inputs[1]
                .items()
                .iter()
                .any(|item| matches!(item, ModelInputItem::ToolResult(_)))
        );
    }

    fn test_caps_with_adapter<T>(
        blackboard: Blackboard,
        adapter: Arc<T>,
    ) -> (CapabilityProviders, Lutum)
    where
        T: TurnAdapter + 'static,
    {
        let budget = SharedPoolBudgetManager::new(SharedPoolBudgetOptions::default());
        let lutum = Lutum::new(adapter, budget);
        let caps = CapabilityProviders::new(CapabilityProviderPorts {
            blackboard,
            cognition_log_port: Rc::new(NoopCognitionLogRepository),
            clock: Rc::new(SystemClock),
            tiers: LutumTiers::from_shared_lutum(lutum.clone()),
        });
        (caps, lutum)
    }

    async fn build_interoception_module(
        caps: &CapabilityProviders,
    ) -> nuillu_module::AllocatedModule {
        let modules = ModuleRegistry::new()
            .register(module_policy(), |caps| async move {
                Ok(InteroceptionModule::new(
                    caps.memo_updated_inbox(),
                    caps.cognition_log_updated_inbox(),
                    caps.blackboard_reader(),
                    caps.interoception_policy(),
                    caps.interoception_writer(),
                    caps.llm("main")
                        .with_tier(nuillu_types::ModelTier::Cheap)
                        .into(),
                    caps.session("main")
                        .with_tier(nuillu_types::ModelTier::Cheap)
                        .with_auto_compaction(session_auto_compaction())
                        .await?,
                ))
            })
            .unwrap()
            .build(caps)
            .await
            .unwrap();
        let (_, mut modules) = modules.into_parts();
        modules.remove(0)
    }

    fn module_policy() -> nuillu_blackboard::ModulePolicy {
        nuillu_blackboard::ModulePolicy::new(
            ReplicaCapRange::new(1, 1).unwrap(),
            Bpm::from_f64(60_000.0)..=Bpm::from_f64(60_000.0),
            linear_ratio_fn,
        )
    }

    fn activate_cx(
        lutum: &Lutum,
        now: chrono::DateTime<chrono::Utc>,
    ) -> nuillu_module::ActivateCx<'static> {
        nuillu_module::ActivateCx::new(
            &[],
            &[],
            &[],
            SessionCompactionRuntime::new(
                lutum.clone(),
                LlmConcurrencyLimiter::new(None),
                ModelTier::Cheap,
                SessionCompactionPolicy::default(),
            ),
            now,
        )
    }

    async fn publish_interoception_wakeup(
        caps: &CapabilityProviders,
        blackboard: &Blackboard,
        now: chrono::DateTime<chrono::Utc>,
    ) {
        publish_interoception_wakeup_with_content(caps, blackboard, "Alice asked for a story", now)
            .await;
    }

    async fn publish_interoception_wakeup_with_content(
        caps: &CapabilityProviders,
        blackboard: &Blackboard,
        content: impl Into<String>,
        now: chrono::DateTime<chrono::Utc>,
    ) {
        let owner = ModuleInstanceId::new(builtin::sensory(), ReplicaIndex::ZERO);
        let record = blackboard.update_memo(owner, content.into(), now).await;
        caps.internal_harness_io()
            .memo_updated_mailbox()
            .publish(nuillu_module::MemoUpdated {
                owner: record.owner,
                index: record.index,
            })
            .await
            .unwrap();
    }

    fn all_input_text(input: &ModelInput) -> String {
        let mut out = String::new();
        for item in input.items() {
            match item {
                ModelInputItem::Message { content, .. } => {
                    for content in content.as_slice() {
                        if let MessageContent::Text(text) = content {
                            out.push_str(text);
                            out.push('\n');
                        }
                    }
                }
                ModelInputItem::Assistant(AssistantInputItem::Text(text)) => {
                    out.push_str(text);
                    out.push('\n');
                }
                ModelInputItem::Turn(turn) => {
                    for index in 0..turn.item_count() {
                        let Some(item) = turn.item_at(index) else {
                            continue;
                        };
                        if let Some(text) = item.as_text() {
                            out.push_str(text);
                            out.push('\n');
                        }
                    }
                }
                _ => {}
            }
        }
        out
    }

    fn text_usage() -> Usage {
        Usage {
            input_tokens: 1,
            ..Usage::zero()
        }
    }

    fn report_affect_scenario(arguments_json: String) -> MockTextScenario {
        MockTextScenario::events(vec![
            Ok(RawTextTurnEvent::Started {
                request_id: Some("interoception-tool".into()),
                model: "mock".into(),
            }),
            Ok(RawTextTurnEvent::ToolCallChunk {
                id: "call-affect".into(),
                name: "report_affect".into(),
                arguments_json_delta: arguments_json,
            }),
            Ok(RawTextTurnEvent::Completed {
                request_id: Some("interoception-tool".into()),
                finish_reason: FinishReason::ToolCall,
                usage: text_usage(),
            }),
        ])
    }

    fn no_affect_tool_scenario() -> MockTextScenario {
        MockTextScenario::events(vec![
            Ok(RawTextTurnEvent::Started {
                request_id: Some("interoception-no-tool".into()),
                model: "mock".into(),
            }),
            Ok(RawTextTurnEvent::TextDelta {
                delta: "no change".into(),
            }),
            Ok(RawTextTurnEvent::Completed {
                request_id: Some("interoception-no-tool".into()),
                finish_reason: FinishReason::Stop,
                usage: text_usage(),
            }),
        ])
    }
}
