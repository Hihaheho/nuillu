use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// Coarse model tiers selected by the attention controller.
///
/// Each tier is bound to a concrete `lutum::Lutum` instance at boot, so
/// changing the underlying model is a startup-config concern, not a runtime
/// concern of modules.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize, JsonSchema)]
pub enum ModelTier {
    Cheap,
    #[default]
    Default,
    Premium,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct TokenBudget(pub u32);

impl TokenBudget {
    pub const fn new(tokens: u32) -> Self {
        Self(tokens)
    }
}
