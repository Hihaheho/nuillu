use std::collections::HashMap;
use std::time::Duration;

use nuillu_types::{ModelTier, ModuleId, ModuleInstanceId, ReplicaCapRange, TokenBudget};
use serde::{Deserialize, Serialize};

/// Per-module knobs the attention controller writes to.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModuleConfig {
    #[serde(default = "default_replicas")]
    pub replicas: u8,
    pub tier: ModelTier,
    /// Periodic activation period. `None` means message-only — the module
    /// runs only when its inbox receives an envelope.
    #[serde(with = "duration_opt")]
    pub period: Option<Duration>,
    pub context_budget: TokenBudget,
}

impl Default for ModuleConfig {
    fn default() -> Self {
        Self {
            replicas: default_replicas(),
            tier: ModelTier::default(),
            period: None,
            context_budget: TokenBudget::new(8192),
        }
    }
}

fn default_replicas() -> u8 {
    1
}

/// Snapshot of the resource allocation across all modules.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ResourceAllocation {
    per_module: HashMap<ModuleId, ModuleConfig>,
}

impl ResourceAllocation {
    pub fn for_module(&self, id: &ModuleId) -> ModuleConfig {
        self.per_module.get(id).cloned().unwrap_or_default()
    }

    pub fn get(&self, id: &ModuleId) -> Option<&ModuleConfig> {
        self.per_module.get(id)
    }

    pub fn active_replicas(&self, id: &ModuleId) -> u8 {
        self.for_module(id).replicas
    }

    pub fn is_replica_active(&self, owner: &ModuleInstanceId) -> bool {
        owner.replica.get() < self.active_replicas(&owner.module)
    }

    pub fn set(&mut self, id: ModuleId, config: ModuleConfig) {
        self.per_module.insert(id, config);
    }

    pub fn iter(&self) -> impl Iterator<Item = (&ModuleId, &ModuleConfig)> {
        self.per_module.iter()
    }

    pub fn clamped(mut self, caps: &HashMap<ModuleId, ReplicaCapRange>) -> Self {
        for (id, range) in caps {
            let mut cfg = self.for_module(id);
            cfg.replicas = range.clamp(cfg.replicas);
            self.set(id.clone(), cfg);
        }
        self
    }
}

mod duration_opt {
    use std::time::Duration;

    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub fn serialize<S: Serializer>(d: &Option<Duration>, s: S) -> Result<S::Ok, S::Error> {
        d.map(|d| d.as_millis() as u64).serialize(s)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Option<Duration>, D::Error> {
        let v = <Option<u64>>::deserialize(d)?;
        Ok(v.map(Duration::from_millis))
    }
}
