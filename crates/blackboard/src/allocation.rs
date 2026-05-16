use std::collections::{BTreeMap, HashMap, HashSet};
use std::ops::RangeInclusive;
use std::time::Duration;

use nuillu_types::{ModelTier, ModuleId, ModuleInstanceId, ReplicaCapRange};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

const ACTIVATION_RATIO_SCALE: u16 = 10_000;

/// Fixed-point activation ratio in the inclusive `0.0..=1.0` range.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct ActivationRatio(u16);

impl ActivationRatio {
    pub const ZERO: Self = Self(0);
    pub const ONE: Self = Self(ACTIVATION_RATIO_SCALE);

    pub fn from_f64(value: f64) -> Self {
        if !value.is_finite() {
            return Self::ZERO;
        }
        let clamped = value.clamp(0.0, 1.0);
        Self((clamped * f64::from(ACTIVATION_RATIO_SCALE)).round() as u16)
    }

    pub fn as_f64(self) -> f64 {
        f64::from(self.0) / f64::from(ACTIVATION_RATIO_SCALE)
    }

    pub(crate) fn raw(self) -> u16 {
        self.0
    }

    pub(crate) fn from_raw(raw: u16) -> Self {
        Self(raw.min(ACTIVATION_RATIO_SCALE))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AllocationEffectLevel {
    Off,
    Minimal,
    Low,
    Normal,
    High,
    Max,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AllocationEffectKind {
    Target,
    Suppression,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AllocationCommand {
    pub effect: AllocationEffectKind,
    pub module: ModuleId,
    pub level: AllocationEffectLevel,
    pub guidance: Option<String>,
}

impl AllocationCommand {
    pub fn target(
        module: ModuleId,
        level: AllocationEffectLevel,
        guidance: impl Into<Option<String>>,
    ) -> Self {
        Self {
            effect: AllocationEffectKind::Target,
            module,
            level,
            guidance: guidance.into(),
        }
    }

    pub fn suppression(module: ModuleId, level: AllocationEffectLevel) -> Self {
        Self {
            effect: AllocationEffectKind::Suppression,
            module,
            level,
            guidance: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct AllocationEffectPolicy {
    target: BTreeMap<AllocationEffectLevel, ActivationRatio>,
    suppression_multiplier: BTreeMap<AllocationEffectLevel, ActivationRatio>,
}

impl AllocationEffectPolicy {
    pub fn target_ratio(&self, level: AllocationEffectLevel) -> ActivationRatio {
        self.target.get(&level).copied().unwrap_or_default()
    }

    pub fn suppression_multiplier(&self, level: AllocationEffectLevel) -> ActivationRatio {
        self.suppression_multiplier
            .get(&level)
            .copied()
            .unwrap_or(ActivationRatio::ONE)
    }
}

impl Default for AllocationEffectPolicy {
    fn default() -> Self {
        Self {
            target: BTreeMap::from([
                (AllocationEffectLevel::Off, ActivationRatio::ZERO),
                (
                    AllocationEffectLevel::Minimal,
                    ActivationRatio::from_f64(0.05),
                ),
                (AllocationEffectLevel::Low, ActivationRatio::from_f64(0.15)),
                (
                    AllocationEffectLevel::Normal,
                    ActivationRatio::from_f64(0.50),
                ),
                (AllocationEffectLevel::High, ActivationRatio::from_f64(0.85)),
                (AllocationEffectLevel::Max, ActivationRatio::ONE),
            ]),
            suppression_multiplier: BTreeMap::from([
                (AllocationEffectLevel::Off, ActivationRatio::ONE),
                (
                    AllocationEffectLevel::Minimal,
                    ActivationRatio::from_f64(0.75),
                ),
                (AllocationEffectLevel::Low, ActivationRatio::from_f64(0.50)),
                (
                    AllocationEffectLevel::Normal,
                    ActivationRatio::from_f64(0.25),
                ),
                (AllocationEffectLevel::High, ActivationRatio::from_f64(0.10)),
                (AllocationEffectLevel::Max, ActivationRatio::from_f64(0.03)),
            ]),
        }
    }
}

impl Default for ActivationRatio {
    fn default() -> Self {
        Self::ONE
    }
}

impl Serialize for ActivationRatio {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_f64(self.as_f64())
    }
}

impl<'de> Deserialize<'de> for ActivationRatio {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        f64::deserialize(deserializer).map(Self::from_f64)
    }
}

/// `0.0..=1.0` ratio of how many replicas to run, derived per-module from
/// [`ActivationRatio`] via the registered [`ActivationRatioFn`].
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
pub struct ReplicasRatio(f64);

impl ReplicasRatio {
    pub const ZERO: Self = Self(0.0);
    pub const ONE: Self = Self(1.0);

    pub fn from_f64(value: f64) -> Self {
        if !value.is_finite() {
            return Self::ZERO;
        }
        Self(value.clamp(0.0, 1.0))
    }

    pub fn as_f64(self) -> f64 {
        self.0
    }
}

/// `0.0..=1.0` ratio mapping into a per-module BPM range. Higher means more
/// frequent `next_batch` invocations (shorter cooldown between batches).
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
pub struct RateLimitRatio(f64);

impl RateLimitRatio {
    pub const ZERO: Self = Self(0.0);
    pub const ONE: Self = Self(1.0);

    pub fn from_f64(value: f64) -> Self {
        if !value.is_finite() {
            return Self::ZERO;
        }
        Self(value.clamp(0.0, 1.0))
    }

    pub fn as_f64(self) -> f64 {
        self.0
    }
}

/// Beats per minute — module-loop tempo for `next_batch` invocations.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct Bpm(f64);

impl Bpm {
    /// Floor for the BPM value. Anything `<=` this floor (including `0.0` and
    /// non-finite inputs) is treated as the floor so that `cooldown()` never
    /// produces a non-finite or out-of-range `Duration` and never panics.
    /// 0.001 BPM corresponds to one beat per 1000 minutes (~16.7h), which is
    /// effectively "never" but still a valid, finite cooldown.
    pub const MIN: Self = Self(0.001);

    pub fn from_f64(value: f64) -> Self {
        if !value.is_finite() || value <= Self::MIN.0 {
            return Self::MIN;
        }
        Self(value)
    }

    pub fn range(start: f64, end: f64) -> RangeInclusive<Self> {
        Self::from_f64(start)..=Self::from_f64(end)
    }

    pub fn as_f64(self) -> f64 {
        self.0
    }

    pub fn cooldown(self) -> Duration {
        // `Duration::from_secs_f64` panics on non-finite or out-of-range
        // inputs; saturate just in case `self.0` is somehow below `MIN`.
        let secs = 60.0 / self.0.max(Self::MIN.0);
        if !secs.is_finite() {
            return Duration::MAX;
        }
        Duration::try_from_secs_f64(secs).unwrap_or(Duration::MAX)
    }
}

/// Pure mapping each module declares at registration: how a single
/// controller-emitted activation knob splits into the replicas and rate-limit
/// dimensions. Pure `fn` (not closure) keeps the registry copyable and
/// stateless.
pub type ActivationRatioFn = fn(ActivationRatio) -> (ReplicasRatio, RateLimitRatio);

/// Both axes track the controller's activation linearly.
pub fn linear_ratio_fn(r: ActivationRatio) -> (ReplicasRatio, RateLimitRatio) {
    let v = r.as_f64();
    (ReplicasRatio::from_f64(v), RateLimitRatio::from_f64(v))
}

/// Activation scales replicas; rate limit is pinned at the maximum BPM.
pub fn replicas_only_ratio_fn(r: ActivationRatio) -> (ReplicasRatio, RateLimitRatio) {
    (ReplicasRatio::from_f64(r.as_f64()), RateLimitRatio::ONE)
}

/// Activation scales rate limit; replicas are pinned at the registered minimum.
pub fn rate_only_ratio_fn(r: ActivationRatio) -> (ReplicasRatio, RateLimitRatio) {
    (ReplicasRatio::ZERO, RateLimitRatio::from_f64(r.as_f64()))
}

/// Boot-time per-module policy: the registry stores one of these per
/// registered module and the blackboard reads them when deriving effective
/// allocation state.
#[derive(Debug, Clone)]
pub struct ModulePolicy {
    pub replicas_range: ReplicaCapRange,
    pub rate_limit_range: RangeInclusive<Bpm>,
    pub activation_ratio_fn: ActivationRatioFn,
    pub zero_replica_window: ZeroReplicaWindowPolicy,
}

impl ModulePolicy {
    pub fn new(
        replicas_range: ReplicaCapRange,
        rate_limit_range: RangeInclusive<Bpm>,
        activation_ratio_fn: ActivationRatioFn,
    ) -> Self {
        Self {
            replicas_range,
            rate_limit_range,
            activation_ratio_fn,
            zero_replica_window: ZeroReplicaWindowPolicy::default(),
        }
    }

    pub fn cooldown_for(&self, ratio: RateLimitRatio) -> Duration {
        let start = self.rate_limit_range.start().as_f64();
        let end = self.rate_limit_range.end().as_f64();
        // ratio = 1.0 picks the high end (max BPM, shortest cooldown);
        // ratio = 0.0 picks the low end (min BPM, longest cooldown).
        let bpm = start + (end - start) * ratio.as_f64();
        Bpm::from_f64(bpm).cooldown()
    }

    /// Total active replica count for a given `replicas_ratio`, clamped to the
    /// registered total replica range. A range with `min = 0` can be fully
    /// inactive.
    pub fn active_replicas_for(&self, ratio: ReplicasRatio) -> u8 {
        if self.replicas_range.max == 0 {
            return 0;
        }
        let requested = (ratio.as_f64() * f64::from(self.replicas_range.max)).ceil() as u8;
        self.replicas_range.clamp(requested)
    }

    /// Number of persistent replica instances to build for this module. Even a
    /// fully allocation-disabled `0..=0` module still gets replica 0 so typed
    /// messages can queue until boot wiring or a later policy makes it active.
    pub fn max_active_replicas(&self) -> u8 {
        self.replicas_range.max.max(1)
    }
}

/// Scheduler-owned low-activity recovery for roles with zero effective replicas.
///
/// This policy does not change [`ResourceAllocation`]. It only lets the agent
/// scheduler briefly run replica 0 after the role has remained allocation-zero
/// across a configured number of successful allocation-controller activations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ZeroReplicaWindowPolicy {
    Disabled,
    EveryControllerActivations(u32),
}

impl ZeroReplicaWindowPolicy {
    pub const DEFAULT_CONTROLLER_ACTIVATIONS: u32 = 3;

    pub fn controller_activation_period(self) -> Option<u32> {
        match self {
            Self::Disabled => None,
            Self::EveryControllerActivations(0) => None,
            Self::EveryControllerActivations(period) => Some(period),
        }
    }
}

impl Default for ZeroReplicaWindowPolicy {
    fn default() -> Self {
        Self::EveryControllerActivations(Self::DEFAULT_CONTROLLER_ACTIVATIONS)
    }
}

/// Per-module guidance the attention controller writes from each priority
/// entry's `hint`. The activation knob is stored separately on
/// [`ResourceAllocation`] (see `set_activation`); tier is host-fixed via
/// [`ResourceAllocation::set_model_override`].
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ModuleConfig {
    #[serde(default)]
    pub guidance: String,
}

/// Snapshot of the resource allocation across all modules.
///
/// Stores:
/// - `per_module`: controller-written guidance per module (from priority hints).
/// - `activation`: controller-derived `ActivationRatio` per module (mapped from
///   priority position via `activation_table`).
/// - `activation_table`: host-set ratio table; index = priority position.
/// - `model_override`: host-set tier per module; absent => `ModelTier::Default`.
/// - `active_replicas` / `cooldown`: derived state populated by `derived()`
///   when the blackboard knows the registered [`ModulePolicy`] per module.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ResourceAllocation {
    per_module: HashMap<ModuleId, ModuleConfig>,
    #[serde(default)]
    activation: HashMap<ModuleId, ActivationRatio>,
    #[serde(default)]
    activation_table: Vec<ActivationRatio>,
    #[serde(default)]
    model_override: BTreeMap<ModuleId, ModelTier>,
    #[serde(skip)]
    active_replicas: HashMap<ModuleId, u8>,
    #[serde(skip)]
    cooldown: HashMap<ModuleId, Duration>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct AllocationLimits {
    pub max_total_active_replicas: Option<u8>,
}

impl AllocationLimits {
    pub const fn unlimited() -> Self {
        Self {
            max_total_active_replicas: None,
        }
    }
}

impl Default for AllocationLimits {
    fn default() -> Self {
        Self {
            max_total_active_replicas: Some(10),
        }
    }
}

impl ResourceAllocation {
    pub fn for_module(&self, id: &ModuleId) -> ModuleConfig {
        self.per_module.get(id).cloned().unwrap_or_default()
    }

    pub fn get(&self, id: &ModuleId) -> Option<&ModuleConfig> {
        self.per_module.get(id)
    }

    pub fn has_activation(&self, id: &ModuleId) -> bool {
        self.activation.contains_key(id)
    }

    pub fn has_module_opinion(&self, id: &ModuleId) -> bool {
        self.per_module.contains_key(id) || self.activation.contains_key(id)
    }

    pub fn activation_for(&self, id: &ModuleId) -> ActivationRatio {
        self.activation.get(id).copied().unwrap_or_default()
    }

    pub fn cooldown_for(&self, id: &ModuleId) -> Option<Duration> {
        self.cooldown.get(id).copied()
    }

    pub fn active_replicas(&self, id: &ModuleId) -> u8 {
        self.active_replicas.get(id).copied().unwrap_or_default()
    }

    pub fn is_replica_active(&self, owner: &ModuleInstanceId) -> bool {
        owner.replica.get() < self.active_replicas(&owner.module)
    }

    /// Write guidance/tier for a module. Activation is set separately via
    /// [`set_activation`].
    pub fn set(&mut self, id: ModuleId, config: ModuleConfig) {
        self.per_module.insert(id, config);
    }

    /// Write the controller's activation knob for a module.
    pub fn set_activation(&mut self, id: ModuleId, ratio: ActivationRatio) {
        self.activation.insert(id, ratio);
    }

    pub fn multiply_activation(&mut self, id: ModuleId, multiplier: ActivationRatio) {
        let current = self.activation_for(&id);
        let product = u32::from(current.raw()) * u32::from(multiplier.raw());
        let rounded =
            (product + u32::from(ACTIVATION_RATIO_SCALE / 2)) / u32::from(ACTIVATION_RATIO_SCALE);
        self.activation
            .insert(id, ActivationRatio::from_raw(rounded as u16));
    }

    pub fn iter(&self) -> impl Iterator<Item = (&ModuleId, &ModuleConfig)> {
        self.per_module.iter()
    }

    pub fn iter_activation(&self) -> impl Iterator<Item = (&ModuleId, ActivationRatio)> {
        self.activation.iter().map(|(id, r)| (id, *r))
    }

    pub fn retain_modules(&mut self, allowed: &std::collections::HashSet<ModuleId>) {
        self.per_module.retain(|id, _| allowed.contains(id));
        self.activation.retain(|id, _| allowed.contains(id));
        self.model_override.retain(|id, _| allowed.contains(id));
        self.cooldown.retain(|id, _| allowed.contains(id));
        self.active_replicas.retain(|id, _| allowed.contains(id));
    }

    /// Host-set lookup table. Index = priority position; positions beyond the
    /// table fall to [`ActivationRatio::ZERO`].
    pub fn activation_table(&self) -> &[ActivationRatio] {
        &self.activation_table
    }

    pub fn set_activation_table(&mut self, table: Vec<ActivationRatio>) {
        self.activation_table = table;
    }

    /// Resolve a module's tier. Falls back to [`ModelTier::Default`] when the
    /// host did not register an override for the module.
    pub fn tier_for(&self, id: &ModuleId) -> ModelTier {
        self.model_override.get(id).copied().unwrap_or_default()
    }

    pub fn set_model_override(&mut self, id: ModuleId, tier: ModelTier) {
        self.model_override.insert(id, tier);
    }

    pub fn iter_model_override(&self) -> impl Iterator<Item = (&ModuleId, ModelTier)> {
        self.model_override.iter().map(|(id, tier)| (id, *tier))
    }

    /// Derive `active_replicas` and `cooldown` from the controller's activation
    /// knob and each registered module's [`ModulePolicy`]. Modules without a
    /// registered policy are left at zero active replicas (the unregistered
    /// fallback).
    pub fn derived(mut self, policies: &HashMap<ModuleId, ModulePolicy>) -> Self {
        self.active_replicas.clear();
        self.cooldown.clear();
        for (id, policy) in policies {
            let ratio = self.activation_for(id);
            let (replicas_ratio, rate_ratio) = (policy.activation_ratio_fn)(ratio);
            self.active_replicas
                .insert(id.clone(), policy.active_replicas_for(replicas_ratio));
            self.cooldown
                .insert(id.clone(), policy.cooldown_for(rate_ratio));
        }
        self
    }

    pub fn limited(mut self, limits: AllocationLimits) -> Self {
        if let Some(max_active) = limits.max_total_active_replicas {
            self.enforce_total_active_limit(max_active);
        }
        self
    }

    pub fn force_disable_modules(mut self, disabled: &HashSet<ModuleId>) -> Self {
        for id in disabled {
            self.active_replicas.insert(id.clone(), 0);
        }
        self
    }

    fn enforce_total_active_limit(&mut self, max_active: u8) {
        let mut active_modules = self
            .allocation_module_ids()
            .into_iter()
            .filter_map(|id| {
                let active = self.active_replicas(&id);
                let ratio = self.activation_for(&id);
                (active > 0).then_some((id, ratio, active))
            })
            .collect::<Vec<_>>();
        active_modules.sort_by(|(left_id, left_ratio, _), (right_id, right_ratio, _)| {
            right_ratio
                .cmp(left_ratio)
                .then_with(|| left_id.as_str().cmp(right_id.as_str()))
        });

        let mut kept = 0_u8;
        for (id, _ratio, active) in active_modules {
            let Some(next_kept) = kept.checked_add(active) else {
                self.active_replicas.insert(id, 0);
                continue;
            };
            if next_kept <= max_active {
                kept = next_kept;
            } else {
                self.active_replicas.insert(id, 0);
            }
        }
    }

    fn allocation_module_ids(&self) -> Vec<ModuleId> {
        let mut ids = self
            .per_module
            .keys()
            .cloned()
            .chain(self.activation.keys().cloned())
            .chain(self.active_replicas.keys().cloned())
            .collect::<HashSet<_>>()
            .into_iter()
            .collect::<Vec<_>>();
        ids.sort_by(|a, b| a.as_str().cmp(b.as_str()));
        ids
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn id(name: &str) -> ModuleId {
        ModuleId::new(name).unwrap()
    }

    fn linear_policy(min_extra: u8, max_extra: u8) -> ModulePolicy {
        ModulePolicy::new(
            ReplicaCapRange::new(min_extra, max_extra).unwrap(),
            Bpm::from_f64(1.0)..=Bpm::from_f64(60.0),
            linear_ratio_fn,
        )
    }

    fn set(allocation: &mut ResourceAllocation, module: &str, ratio: f64) {
        let module = id(module);
        allocation.set(module.clone(), ModuleConfig::default());
        allocation.set_activation(module, ActivationRatio::from_f64(ratio));
    }

    #[test]
    fn allocation_limits_deactivate_excess_active_by_ratio_then_lexical_id() {
        let mut allocation = ResourceAllocation::default();
        set(&mut allocation, "gamma", 0.7);
        set(&mut allocation, "alpha", 1.0);
        set(&mut allocation, "beta", 0.7);

        let mut policies = HashMap::new();
        policies.insert(id("alpha"), linear_policy(0, 1));
        policies.insert(id("beta"), linear_policy(0, 1));
        policies.insert(id("gamma"), linear_policy(0, 1));

        let limited = allocation.derived(&policies).limited(AllocationLimits {
            max_total_active_replicas: Some(2),
        });

        assert_eq!(limited.active_replicas(&id("alpha")), 1);
        assert_eq!(limited.active_replicas(&id("beta")), 1);
        assert_eq!(limited.active_replicas(&id("gamma")), 0);
    }

    #[test]
    fn allocation_limits_default_to_ten_active() {
        assert_eq!(
            AllocationLimits::default(),
            AllocationLimits {
                max_total_active_replicas: Some(10),
            }
        );
    }

    #[test]
    fn tier_for_resolves_via_model_override_with_default_fallback() {
        let mut allocation = ResourceAllocation::default();
        allocation.set_model_override(id("speak"), ModelTier::Premium);
        assert_eq!(allocation.tier_for(&id("speak")), ModelTier::Premium);
        assert_eq!(allocation.tier_for(&id("missing")), ModelTier::Default);
    }

    #[test]
    fn module_policy_active_replicas_uses_total_replica_range() {
        let disabled = linear_policy(0, 0);
        assert_eq!(disabled.active_replicas_for(ReplicasRatio::ZERO), 0);
        assert_eq!(disabled.active_replicas_for(ReplicasRatio::ONE), 0);

        let optional_one = linear_policy(0, 1);
        assert_eq!(optional_one.active_replicas_for(ReplicasRatio::ZERO), 0);
        assert_eq!(
            optional_one.active_replicas_for(ReplicasRatio::from_f64(0.5)),
            1
        );
        assert_eq!(optional_one.active_replicas_for(ReplicasRatio::ONE), 1);

        let one_to_two = linear_policy(1, 2);
        assert_eq!(one_to_two.active_replicas_for(ReplicasRatio::ZERO), 1);
        assert_eq!(
            one_to_two.active_replicas_for(ReplicasRatio::from_f64(0.5)),
            1
        );
        assert_eq!(one_to_two.active_replicas_for(ReplicasRatio::ONE), 2);
    }

    #[test]
    fn bpm_cooldown_is_inverse_of_rate() {
        // 60 BPM = 1 second per beat.
        assert_eq!(Bpm::from_f64(60.0).cooldown(), Duration::from_secs(1));
        // 120 BPM = 0.5 seconds per beat.
        assert_eq!(Bpm::from_f64(120.0).cooldown(), Duration::from_millis(500));
    }

    #[test]
    fn bpm_range_sanitizes_bounds() {
        let range = Bpm::range(0.0, 12.0);

        assert_eq!(*range.start(), Bpm::MIN);
        assert_eq!(range.end().as_f64(), 12.0);
    }

    #[test]
    fn bpm_handles_zero_negative_and_nan_without_panicking() {
        let from_zero = Bpm::from_f64(0.0);
        let from_negative = Bpm::from_f64(-1.0);
        let from_nan = Bpm::from_f64(f64::NAN);
        let from_inf = Bpm::from_f64(f64::INFINITY);
        assert_eq!(from_zero, Bpm::MIN);
        assert_eq!(from_negative, Bpm::MIN);
        assert_eq!(from_nan, Bpm::MIN);
        assert!(from_inf.as_f64().is_finite());
        // Should be a finite, non-zero Duration without panicking.
        assert!(from_zero.cooldown() > Duration::ZERO);
        assert!(from_zero.cooldown() < Duration::MAX);
    }
}
