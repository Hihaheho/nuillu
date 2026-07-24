use std::{
    collections::HashMap,
    num::NonZeroUsize,
    sync::{Arc, Mutex},
};

use lutum::Lutum;
use nuillu_types::ModelTier;

use crate::llm::LlmConcurrencyLimiter;

/// Shared runtime handle for one tier: Lutum plus the model-scoped concurrency limiter.
#[derive(Clone)]
pub struct LlmTierHandle {
    pub lutum: Lutum,
    pub concurrency: LlmConcurrencyLimiter,
    pub model_key: Arc<str>,
    pub reasoning: bool,
}

impl LlmTierHandle {
    pub fn new(
        lutum: Lutum,
        concurrency: LlmConcurrencyLimiter,
        model_key: impl Into<Arc<str>>,
        reasoning: bool,
    ) -> Self {
        Self {
            lutum,
            concurrency,
            model_key: model_key.into(),
            reasoning,
        }
    }
}

/// One [`LlmTierHandle`] per coarse model tier. Constructed once at boot and
/// shared by every [`LlmAccess`](crate::LlmAccess) capability handle.
#[derive(Clone)]
pub struct LutumTiers {
    pub cheap: LlmTierHandle,
    pub default: LlmTierHandle,
    pub premium: LlmTierHandle,
    pub image: LlmTierHandle,
}

impl LutumTiers {
    pub fn pick(&self, tier: ModelTier) -> Lutum {
        self.pick_handle(tier).lutum.clone()
    }

    pub fn pick_handle(&self, tier: ModelTier) -> &LlmTierHandle {
        match tier {
            ModelTier::Cheap => &self.cheap,
            ModelTier::Default => &self.default,
            ModelTier::Premium => &self.premium,
            ModelTier::Image => &self.image,
        }
    }

    pub fn from_shared_lutum(lutum: Lutum) -> Self {
        Self::from_shared_lutum_with_key(lutum, "test")
    }

    pub fn from_shared_lutum_with_key(lutum: Lutum, model_key: &str) -> Self {
        let key: Arc<str> = Arc::from(model_key);
        let handle = |lutum: Lutum| {
            LlmTierHandle::new(lutum, LlmConcurrencyLimiter::new(None), key.clone(), false)
        };
        Self {
            cheap: handle(lutum.clone()),
            default: handle(lutum.clone()),
            premium: handle(lutum.clone()),
            image: handle(lutum),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use lutum::{Lutum, MockLlmAdapter, SharedPoolBudgetManager, SharedPoolBudgetOptions};

    use super::*;

    #[test]
    fn image_tier_selects_image_handle() {
        let lutum = Lutum::new(
            Arc::new(MockLlmAdapter::new()),
            SharedPoolBudgetManager::new(SharedPoolBudgetOptions::default()),
        );
        let handle =
            |key| LlmTierHandle::new(lutum.clone(), LlmConcurrencyLimiter::new(None), key, false);
        let tiers = LutumTiers {
            cheap: handle("cheap"),
            default: handle("default"),
            premium: handle("premium"),
            image: handle("image"),
        };

        assert_eq!(
            tiers.pick_handle(ModelTier::Image).model_key.as_ref(),
            "image"
        );
    }
}

/// Builds shared semaphores keyed by model definition name.
#[derive(Clone, Default)]
pub struct LlmConcurrencyPool {
    inner: Arc<Mutex<HashMap<String, LlmConcurrencyLimiter>>>,
}

impl std::fmt::Debug for LlmConcurrencyPool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LlmConcurrencyPool").finish_non_exhaustive()
    }
}

impl LlmConcurrencyPool {
    pub fn limiter_for(
        &self,
        model_key: &str,
        max_concurrent_llm_calls: Option<NonZeroUsize>,
    ) -> LlmConcurrencyLimiter {
        let mut semaphores = self
            .inner
            .lock()
            .expect("LlmConcurrencyPool mutex poisoned");
        if let Some(existing) = semaphores.get(model_key)
            && existing.max_concurrent_calls() == max_concurrent_llm_calls
        {
            return existing.clone();
        }
        let limiter = LlmConcurrencyLimiter::new(max_concurrent_llm_calls);
        semaphores.insert(model_key.to_string(), limiter.clone());
        limiter
    }
}
