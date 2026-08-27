use std::collections::{BTreeMap, HashMap, HashSet};
use std::ops::RangeInclusive;
use std::time::Duration;

use nuillu_types::{ModuleId, ModuleInstanceId, ReplicaCapRange, SubsystemId};
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

    /// Compose two hierarchy activation factors. The fixed-point product is
    /// rounded to the nearest representable ratio.
    pub fn multiplied(self, other: Self) -> Self {
        let product = u32::from(self.raw()) * u32::from(other.raw());
        let rounded =
            (product + u32::from(ACTIVATION_RATIO_SCALE / 2)) / u32::from(ACTIVATION_RATIO_SCALE);
        Self::from_raw(rounded as u16)
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
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SubsystemAllocationCommand {
    pub subsystem: SubsystemId,
    pub level: AllocationEffectLevel,
}

impl SubsystemAllocationCommand {
    pub fn target(subsystem: SubsystemId, level: AllocationEffectLevel) -> Self {
        Self { subsystem, level }
    }
}

impl AllocationCommand {
    pub fn target(module: ModuleId, level: AllocationEffectLevel) -> Self {
        Self {
            effect: AllocationEffectKind::Target,
            module,
            level,
        }
    }

    pub fn suppression(module: ModuleId, level: AllocationEffectLevel) -> Self {
        Self {
            effect: AllocationEffectKind::Suppression,
            module,
            level,
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
/// frequent `next_batch` invocations (shorter period between batches).
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
    /// non-finite inputs) is treated as the floor so that `period()` never
    /// produces a non-finite or out-of-range `Duration` and never panics.
    /// 0.001 BPM corresponds to one beat per 1000 minutes (~16.7h), which is
    /// effectively "never" but still a valid, finite period.
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

    pub fn period(self) -> Duration {
        // `Duration::from_secs_f64` panics on non-finite or out-of-range
        // inputs; saturate just in case `self.0` is somehow below `MIN`.
        let secs = 60.0 / self.0.max(Self::MIN.0);
        if !secs.is_finite() {
            return Duration::MAX;
        }
        Duration::try_from_secs_f64(secs).unwrap_or(Duration::MAX)
    }

    pub fn sleep_after_turn(self, turn_elapsed: Duration) -> Duration {
        self.period().saturating_sub(turn_elapsed)
    }
}

/// Pure mapping each module declares at registration: how a single
/// controller-emitted activation knob splits into the replicas and rate-limit
/// dimensions. Pure `fn` (not closure) keeps the registry copyable and
/// stateless.
pub type ActivationRatioFn = fn(ActivationRatio) -> (ReplicasRatio, RateLimitRatio);

/// Hierarchical activation values made available to Rust-registered custom
/// projections. Standard projections intentionally use only `effective`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActivationInput {
    pub local: ActivationRatio,
    pub scope: ActivationRatio,
    pub effective: ActivationRatio,
}

impl ActivationInput {
    pub fn new(local: ActivationRatio, scope: ActivationRatio) -> Self {
        Self {
            local,
            scope,
            effective: local.multiplied(scope),
        }
    }
}

pub type ReplicaProjectionFn = fn(ActivationInput) -> ReplicasRatio;
pub type RateProjectionFn = fn(ActivationInput) -> RateLimitRatio;

#[derive(Debug, Clone, Copy)]
pub enum ReplicaProjection {
    Linear,
    Threshold(ActivationRatio),
    Custom(ReplicaProjectionFn),
}

impl ReplicaProjection {
    pub fn project(self, input: ActivationInput) -> ReplicasRatio {
        match self {
            Self::Linear => ReplicasRatio::from_f64(input.effective.as_f64()),
            Self::Threshold(threshold) => {
                if input.effective < threshold {
                    ReplicasRatio::ZERO
                } else {
                    ReplicasRatio::ONE
                }
            }
            Self::Custom(project) => project(input),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum RateProjection {
    Linear,
    Threshold(ActivationRatio),
    Custom(RateProjectionFn),
}

impl RateProjection {
    pub fn project(self, input: ActivationInput) -> RateLimitRatio {
        match self {
            Self::Linear => RateLimitRatio::from_f64(input.effective.as_f64()),
            Self::Threshold(threshold) => {
                if input.effective < threshold {
                    RateLimitRatio::ZERO
                } else {
                    RateLimitRatio::ONE
                }
            }
            Self::Custom(project) => project(input),
        }
    }
}

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
    pub replica_projection: Option<ReplicaProjection>,
    pub rate_projection: Option<RateProjection>,
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
            replica_projection: None,
            rate_projection: None,
            zero_replica_window: ZeroReplicaWindowPolicy::default(),
        }
    }

    /// Register independent resource-axis projections. This is the preferred
    /// API for new wiring; `new` remains available for existing combined
    /// projection functions.
    pub fn with_projections(
        replicas_range: ReplicaCapRange,
        rate_limit_range: RangeInclusive<Bpm>,
        replica_projection: ReplicaProjection,
        rate_projection: RateProjection,
    ) -> Self {
        Self {
            replicas_range,
            rate_limit_range,
            activation_ratio_fn: linear_ratio_fn,
            replica_projection: Some(replica_projection),
            rate_projection: Some(rate_projection),
            zero_replica_window: ZeroReplicaWindowPolicy::default(),
        }
    }

    pub fn project(&self, input: ActivationInput) -> (ReplicasRatio, RateLimitRatio) {
        match (self.replica_projection, self.rate_projection) {
            (Some(replicas), Some(rate)) => (replicas.project(input), rate.project(input)),
            _ => (self.activation_ratio_fn)(input.effective),
        }
    }

    pub fn bpm_for(&self, ratio: RateLimitRatio) -> Bpm {
        let start = self.rate_limit_range.start().as_f64();
        let end = self.rate_limit_range.end().as_f64();
        // ratio = 1.0 picks the high end (max BPM, shortest period);
        // ratio = 0.0 picks the low end (min BPM, longest period).
        let bpm = start + (end - start) * ratio.as_f64();
        Bpm::from_f64(bpm)
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

/// Boot-time resource policy for one immediate child subsystem mount.
#[derive(Debug, Clone)]
pub struct SubsystemPolicy {
    pub replicas_range: SubsystemReplicaRange,
    pub replica_capacity: u8,
    pub replica_projection: ReplicaProjection,
}

impl SubsystemPolicy {
    pub fn new(
        replicas_range: SubsystemReplicaRange,
        replica_capacity: u8,
        replica_projection: ReplicaProjection,
    ) -> Self {
        Self {
            replicas_range,
            replica_capacity,
            replica_projection,
        }
    }

    pub fn active_replicas_for(&self, input: ActivationInput) -> u8 {
        let ratio = self.replica_projection.project(input);
        let requested = if self.replicas_range.max == 0 {
            0
        } else {
            (ratio.as_f64() * f64::from(self.replicas_range.max)).ceil() as u8
        };
        self.replicas_range
            .clamp(requested)
            .min(self.replica_capacity)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SubsystemReplicaRange {
    pub min: u8,
    pub max: u8,
}

impl SubsystemReplicaRange {
    pub fn new(min: u8, max: u8) -> Option<Self> {
        (min <= max).then_some(Self { min, max })
    }

    pub fn clamp(self, requested: u8) -> u8 {
        requested.clamp(self.min, self.max)
    }
}

/// Local allocation state for immediate child subsystem mounts in one scope.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct SubsystemAllocation {
    #[serde(default)]
    activation: HashMap<SubsystemId, ActivationRatio>,
}

impl SubsystemAllocation {
    pub fn has_activation(&self, subsystem: &SubsystemId) -> bool {
        self.activation.contains_key(subsystem)
    }

    pub fn activation_for(&self, subsystem: &SubsystemId) -> ActivationRatio {
        self.activation.get(subsystem).copied().unwrap_or_default()
    }

    pub fn set_activation(&mut self, subsystem: SubsystemId, ratio: ActivationRatio) {
        self.activation.insert(subsystem, ratio);
    }

    pub fn subsystem_ids(&self) -> Vec<SubsystemId> {
        let mut ids = self.activation.keys().cloned().collect::<Vec<_>>();
        ids.sort_by(|left, right| left.as_str().cmp(right.as_str()));
        ids
    }
}

/// Scheduler-owned low-activity recovery for roles with zero effective replicas.
///
/// This policy does not change [`ResourceAllocation`]. It only lets the agent
/// scheduler briefly run replica 0 after the role has remained allocation-zero
/// across a configured number of successful allocation activations.
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

/// Snapshot of the resource allocation across all modules.
///
/// Stores:
/// - `activation`: controller-derived `ActivationRatio` per module (mapped from
///   priority position via `activation_table`).
/// - `activation_table`: host-set ratio table; index = priority position.
/// - `active_replicas` / `bpm`: derived state populated by `derived()`
///   when the blackboard knows the registered [`ModulePolicy`] per module.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ResourceAllocation {
    #[serde(default)]
    activation: HashMap<ModuleId, ActivationRatio>,
    #[serde(default)]
    activation_table: Vec<ActivationRatio>,
    #[serde(skip)]
    active_replicas: HashMap<ModuleId, u8>,
    #[serde(skip)]
    bpm: HashMap<ModuleId, Bpm>,
    #[serde(skip)]
    effective_activation: HashMap<ModuleId, ActivationRatio>,
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
    pub fn has_activation(&self, id: &ModuleId) -> bool {
        self.activation.contains_key(id)
    }

    pub fn has_module_opinion(&self, id: &ModuleId) -> bool {
        self.activation.contains_key(id)
    }

    pub fn activation_for(&self, id: &ModuleId) -> ActivationRatio {
        self.activation.get(id).copied().unwrap_or_default()
    }

    pub fn bpm_for(&self, id: &ModuleId) -> Option<Bpm> {
        self.bpm.get(id).copied()
    }

    pub fn effective_activation_for(&self, id: &ModuleId) -> ActivationRatio {
        self.effective_activation
            .get(id)
            .copied()
            .unwrap_or_else(|| self.activation_for(id))
    }

    pub fn active_replicas(&self, id: &ModuleId) -> u8 {
        self.active_replicas.get(id).copied().unwrap_or_default()
    }

    pub fn is_replica_active(&self, owner: &ModuleInstanceId) -> bool {
        owner.replica.get() < self.active_replicas(&owner.module)
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

    pub fn iter_activation(&self) -> impl Iterator<Item = (&ModuleId, ActivationRatio)> {
        self.activation.iter().map(|(id, r)| (id, *r))
    }

    pub fn module_ids(&self) -> Vec<ModuleId> {
        self.allocation_module_ids()
    }

    pub fn retain_modules(&mut self, allowed: &std::collections::HashSet<ModuleId>) {
        self.activation.retain(|id, _| allowed.contains(id));
        self.bpm.retain(|id, _| allowed.contains(id));
        self.effective_activation
            .retain(|id, _| allowed.contains(id));
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

    /// Derive `active_replicas` and `bpm` from the controller's activation
    /// knob and each registered module's [`ModulePolicy`]. Modules without a
    /// registered policy are left at zero active replicas (the unregistered
    /// fallback).
    pub fn derived(self, policies: &HashMap<ModuleId, ModulePolicy>) -> Self {
        self.derive_with_scope(policies, ActivationRatio::ONE)
    }

    pub fn derived_in_scope(
        self,
        policies: &HashMap<ModuleId, ModulePolicy>,
        scope_activation: ActivationRatio,
    ) -> Self {
        self.derive_with_scope(policies, scope_activation)
    }

    fn derive_with_scope(
        mut self,
        policies: &HashMap<ModuleId, ModulePolicy>,
        scope_activation: ActivationRatio,
    ) -> Self {
        self.active_replicas.clear();
        self.bpm.clear();
        self.effective_activation.clear();
        for (id, policy) in policies {
            let input = ActivationInput::new(self.activation_for(id), scope_activation);
            let (replicas_ratio, rate_ratio) = policy.project(input);
            self.effective_activation
                .insert(id.clone(), input.effective);
            self.active_replicas
                .insert(id.clone(), policy.active_replicas_for(replicas_ratio));
            self.bpm.insert(id.clone(), policy.bpm_for(rate_ratio));
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
            .activation
            .keys()
            .cloned()
            .chain(self.active_replicas.keys().cloned())
            .chain(self.bpm.keys().cloned())
            .chain(self.effective_activation.keys().cloned())
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
        allocation.set_activation(module, ActivationRatio::from_f64(ratio));
    }

    #[test]
    fn hierarchical_linear_rate_projects_effective_ratio_into_bpm_range() {
        let module = id("worker");
        let policy = ModulePolicy::with_projections(
            ReplicaCapRange::new(0, 1).unwrap(),
            Bpm::range(1.0, 5.0),
            ReplicaProjection::Linear,
            RateProjection::Linear,
        );
        let mut allocation = ResourceAllocation::default();
        allocation.set_activation(module.clone(), ActivationRatio::ONE);
        let derived = allocation.derived_in_scope(
            &HashMap::from([(module.clone(), policy)]),
            ActivationRatio::from_f64(0.5),
        );

        assert_eq!(derived.effective_activation_for(&module).as_f64(), 0.5);
        assert_eq!(derived.bpm_for(&module).unwrap().as_f64(), 3.0);
    }

    #[test]
    fn threshold_replica_projection_respects_minimum() {
        let policy = SubsystemPolicy::new(
            SubsystemReplicaRange::new(1, 4).unwrap(),
            4,
            ReplicaProjection::Threshold(ActivationRatio::from_f64(0.3)),
        );
        assert_eq!(
            policy.active_replicas_for(ActivationInput::new(
                ActivationRatio::from_f64(0.2),
                ActivationRatio::ONE,
            )),
            1
        );
        assert_eq!(
            policy.active_replicas_for(ActivationInput::new(
                ActivationRatio::from_f64(0.3),
                ActivationRatio::ONE,
            )),
            4
        );
    }

    #[test]
    fn custom_projection_receives_local_scope_and_effective_activation() {
        fn custom(input: ActivationInput) -> ReplicasRatio {
            assert_eq!(input.local.as_f64(), 0.5);
            assert_eq!(input.scope.as_f64(), 0.8);
            assert_eq!(input.effective.as_f64(), 0.4);
            ReplicasRatio::ONE
        }
        let policy = SubsystemPolicy::new(
            SubsystemReplicaRange::new(0, 2).unwrap(),
            2,
            ReplicaProjection::Custom(custom),
        );
        assert_eq!(
            policy.active_replicas_for(ActivationInput::new(
                ActivationRatio::from_f64(0.5),
                ActivationRatio::from_f64(0.8),
            )),
            2
        );
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
    fn bpm_period_is_inverse_of_rate() {
        // 60 BPM = 1 second per beat.
        assert_eq!(Bpm::from_f64(60.0).period(), Duration::from_secs(1));
        // 120 BPM = 0.5 seconds per beat.
        assert_eq!(Bpm::from_f64(120.0).period(), Duration::from_millis(500));
        // 3 BPM = 20 seconds per beat.
        assert_eq!(Bpm::from_f64(3.0).period(), Duration::from_secs(20));
    }

    #[test]
    fn bpm_sleep_after_turn_subtracts_elapsed_from_period() {
        let bpm = Bpm::from_f64(3.0);

        assert_eq!(
            bpm.sleep_after_turn(Duration::from_secs(20)),
            Duration::ZERO
        );
        assert_eq!(
            bpm.sleep_after_turn(Duration::from_secs(10)),
            Duration::from_secs(10)
        );
        assert_eq!(
            bpm.sleep_after_turn(Duration::from_secs(25)),
            Duration::ZERO
        );
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
        assert!(from_zero.period() > Duration::ZERO);
        assert!(from_zero.period() < Duration::MAX);
    }
}
