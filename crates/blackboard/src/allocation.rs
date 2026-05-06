use std::collections::HashMap;
use std::time::Duration;

use nuillu_types::{ModelTier, ModuleId, TokenBudget};
use serde::{Deserialize, Serialize};

/// Per-module knobs the attention controller writes to.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModuleConfig {
    pub enabled: bool,
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
            enabled: true,
            tier: ModelTier::default(),
            period: None,
            context_budget: TokenBudget::new(8192),
        }
    }
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

    pub fn is_enabled(&self, id: &ModuleId) -> bool {
        self.for_module(id).enabled
    }

    pub fn set(&mut self, id: ModuleId, config: ModuleConfig) {
        self.per_module.insert(id, config);
    }

    pub fn iter(&self) -> impl Iterator<Item = (&ModuleId, &ModuleConfig)> {
        self.per_module.iter()
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
