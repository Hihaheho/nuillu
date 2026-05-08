use std::collections::{HashMap, HashSet};

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

    pub fn active_replicas(self, range: ReplicaCapRange) -> u8 {
        if range.max == 0 {
            return 0;
        }
        let requested =
            ((u32::from(self.0) * u32::from(range.max)) + u32::from(ACTIVATION_RATIO_SCALE) - 1)
                / u32::from(ACTIVATION_RATIO_SCALE);
        (requested as u8).clamp(range.min, range.max)
    }

    fn active_without_caps(self) -> u8 {
        u8::from(self.0 > 0)
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

/// Per-module knobs the attention controller writes to.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModuleConfig {
    #[serde(default)]
    pub activation_ratio: ActivationRatio,
    #[serde(default)]
    pub guidance: String,
    pub tier: ModelTier,
}

impl Default for ModuleConfig {
    fn default() -> Self {
        Self {
            activation_ratio: ActivationRatio::ONE,
            guidance: String::new(),
            tier: ModelTier::default(),
        }
    }
}

/// Snapshot of the resource allocation across all modules.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ResourceAllocation {
    per_module: HashMap<ModuleId, ModuleConfig>,
    #[serde(skip)]
    active_replicas: HashMap<ModuleId, u8>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct AllocationLimits {
    pub max_total_active_replicas: Option<u8>,
    pub max_premium_replicas: Option<u8>,
}

impl AllocationLimits {
    pub const fn unlimited() -> Self {
        Self {
            max_total_active_replicas: None,
            max_premium_replicas: None,
        }
    }
}

impl Default for AllocationLimits {
    fn default() -> Self {
        Self {
            max_total_active_replicas: Some(10),
            max_premium_replicas: Some(1),
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

    pub fn active_replicas(&self, id: &ModuleId) -> u8 {
        self.active_replicas
            .get(id)
            .copied()
            .unwrap_or_else(|| self.for_module(id).activation_ratio.active_without_caps())
    }

    pub fn is_replica_active(&self, owner: &ModuleInstanceId) -> bool {
        owner.replica.get() < self.active_replicas(&owner.module)
    }

    pub fn set(&mut self, id: ModuleId, config: ModuleConfig) {
        self.active_replicas
            .insert(id.clone(), config.activation_ratio.active_without_caps());
        self.per_module.insert(id, config);
    }

    pub fn iter(&self) -> impl Iterator<Item = (&ModuleId, &ModuleConfig)> {
        self.per_module.iter()
    }

    pub fn clamped(mut self, caps: &HashMap<ModuleId, ReplicaCapRange>) -> Self {
        for (id, range) in caps {
            let cfg = self.for_module(id);
            self.active_replicas
                .insert(id.clone(), cfg.activation_ratio.active_replicas(*range));
            self.per_module.insert(id.clone(), cfg);
        }
        self
    }

    pub fn limited(mut self, limits: AllocationLimits) -> Self {
        if let Some(max_premium) = limits.max_premium_replicas {
            self.enforce_premium_limit(max_premium);
        }
        if let Some(max_active) = limits.max_total_active_replicas {
            self.enforce_total_active_limit(max_active);
        }
        self
    }

    fn enforce_premium_limit(&mut self, max_premium: u8) {
        let mut active_premium = self
            .allocation_module_ids()
            .into_iter()
            .filter_map(|id| {
                let active = self.active_replicas(&id);
                let cfg = self.for_module(&id);
                (active > 0 && cfg.tier == ModelTier::Premium).then_some((
                    id,
                    cfg.activation_ratio,
                    active,
                ))
            })
            .collect::<Vec<_>>();
        active_premium.sort_by(|(left_id, left_ratio, _), (right_id, right_ratio, _)| {
            right_ratio
                .cmp(left_ratio)
                .then_with(|| left_id.as_str().cmp(right_id.as_str()))
        });

        let mut kept = 0_u8;
        for (id, _ratio, active) in active_premium {
            let Some(next_kept) = kept.checked_add(active) else {
                if let Some(cfg) = self.per_module.get_mut(&id) {
                    cfg.tier = ModelTier::Default;
                }
                continue;
            };
            if next_kept <= max_premium {
                kept = next_kept;
            } else if let Some(cfg) = self.per_module.get_mut(&id) {
                cfg.tier = ModelTier::Default;
            }
        }
    }

    fn enforce_total_active_limit(&mut self, max_active: u8) {
        let mut active_modules = self
            .allocation_module_ids()
            .into_iter()
            .filter_map(|id| {
                let active = self.active_replicas(&id);
                let cfg = self.for_module(&id);
                (active > 0).then_some((id, cfg.activation_ratio, active))
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

    fn set(allocation: &mut ResourceAllocation, module: &str, ratio: f64, tier: ModelTier) {
        allocation.set(
            id(module),
            ModuleConfig {
                activation_ratio: ActivationRatio::from_f64(ratio),
                guidance: String::new(),
                tier,
            },
        );
    }

    #[test]
    fn allocation_limits_downgrade_excess_premium_by_ratio_then_lexical_id() {
        let mut allocation = ResourceAllocation::default();
        set(&mut allocation, "beta", 1.0, ModelTier::Premium);
        set(&mut allocation, "alpha", 1.0, ModelTier::Premium);
        set(&mut allocation, "gamma", 0.8, ModelTier::Premium);

        let limited = allocation.limited(AllocationLimits {
            max_total_active_replicas: None,
            max_premium_replicas: Some(1),
        });

        assert_eq!(limited.for_module(&id("alpha")).tier, ModelTier::Premium);
        assert_eq!(limited.for_module(&id("beta")).tier, ModelTier::Default);
        assert_eq!(limited.for_module(&id("gamma")).tier, ModelTier::Default);
    }

    #[test]
    fn allocation_limits_deactivate_excess_active_by_ratio_then_lexical_id() {
        let mut allocation = ResourceAllocation::default();
        set(&mut allocation, "gamma", 0.7, ModelTier::Cheap);
        set(&mut allocation, "alpha", 1.0, ModelTier::Cheap);
        set(&mut allocation, "beta", 0.7, ModelTier::Cheap);

        let limited = allocation.limited(AllocationLimits {
            max_total_active_replicas: Some(2),
            max_premium_replicas: None,
        });

        assert_eq!(limited.active_replicas(&id("alpha")), 1);
        assert_eq!(limited.active_replicas(&id("beta")), 1);
        assert_eq!(limited.active_replicas(&id("gamma")), 0);
    }

    #[test]
    fn allocation_limits_default_to_ten_active_and_one_premium() {
        assert_eq!(
            AllocationLimits::default(),
            AllocationLimits {
                max_total_active_replicas: Some(10),
                max_premium_replicas: Some(1),
            }
        );
    }
}
