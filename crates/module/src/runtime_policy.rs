use std::time::Duration;

use nuillu_blackboard::{AllocationEffectLevel, AllocationEffectPolicy, AllocationLimits};

use crate::session_compaction::SessionCompactionPolicy;

#[derive(Debug, Clone, PartialEq)]
pub struct InteroceptionRuntimePolicy {
    pub quiet_sleep_threshold: Duration,
    pub wake_arousal_change_multiplier: f32,
    pub affect_arousal_change_multiplier: f32,
}

impl InteroceptionRuntimePolicy {
    pub fn wake_increase_for(&self, level: AllocationEffectLevel) -> f32 {
        level_unit(level) * 0.30 * self.wake_arousal_change_multiplier
    }

    pub fn affect_increase_for(&self, level: AllocationEffectLevel) -> f32 {
        level_unit(level) * 0.25 * self.affect_arousal_change_multiplier
    }

    pub fn valence_change_for(&self, level: AllocationEffectLevel) -> f32 {
        level_unit(level) * 0.25 * self.affect_arousal_change_multiplier
    }
}

impl Default for InteroceptionRuntimePolicy {
    fn default() -> Self {
        Self {
            quiet_sleep_threshold: Duration::from_secs(30),
            wake_arousal_change_multiplier: 1.0,
            affect_arousal_change_multiplier: 1.0,
        }
    }
}

fn level_unit(level: AllocationEffectLevel) -> f32 {
    match level {
        AllocationEffectLevel::Off => 0.0,
        AllocationEffectLevel::Minimal => 0.15,
        AllocationEffectLevel::Low => 0.35,
        AllocationEffectLevel::Normal => 0.60,
        AllocationEffectLevel::High => 0.85,
        AllocationEffectLevel::Max => 1.0,
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct RuntimePolicy {
    pub allocation_limits: AllocationLimits,
    pub allocation_effects: AllocationEffectPolicy,
    pub interoception: InteroceptionRuntimePolicy,
    pub memo_retained_per_owner: usize,
    pub cognition_log_retained_entries: usize,
    pub session_compaction: SessionCompactionPolicy,
}

impl Default for RuntimePolicy {
    fn default() -> Self {
        Self {
            allocation_limits: AllocationLimits::default(),
            allocation_effects: AllocationEffectPolicy::default(),
            interoception: InteroceptionRuntimePolicy::default(),
            memo_retained_per_owner: 8,
            cognition_log_retained_entries: 16,
            session_compaction: SessionCompactionPolicy::default(),
        }
    }
}
