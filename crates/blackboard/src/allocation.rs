use std::collections::HashMap;

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
}
