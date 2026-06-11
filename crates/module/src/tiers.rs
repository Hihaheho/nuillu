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
}

impl LlmTierHandle {
    pub fn new(
        lutum: Lutum,
        concurrency: LlmConcurrencyLimiter,
        model_key: impl Into<Arc<str>>,
    ) -> Self {
        Self {
            lutum,
            concurrency,
            model_key: model_key.into(),
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
        }
    }

    pub fn from_shared_lutum(lutum: Lutum) -> Self {
        Self::from_shared_lutum_with_key(lutum, "test")
    }

    pub fn from_shared_lutum_with_key(lutum: Lutum, model_key: &str) -> Self {
        let key: Arc<str> = Arc::from(model_key);
        let handle =
            |lutum: Lutum| LlmTierHandle::new(lutum, LlmConcurrencyLimiter::new(None), key.clone());
        Self {
            cheap: handle(lutum.clone()),
            default: handle(lutum.clone()),
            premium: handle(lutum),
        }
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
