//! Test fixtures shared by `#[cfg(test)] mod tests` blocks in this crate.

use std::sync::Arc;

use lutum::{Lutum, MockLlmAdapter, SharedPoolBudgetManager, SharedPoolBudgetOptions};
use nuillu_blackboard::Blackboard;
use nuillu_module::ports::{
    NoopAttentionRepository, NoopFileSearchProvider, NoopMemoryStore, NoopUtteranceSink,
    SystemClock,
};
use nuillu_module::{CapabilityProviders, LutumTiers};

pub(crate) fn test_caps(blackboard: Blackboard) -> CapabilityProviders {
    let adapter = Arc::new(MockLlmAdapter::new());
    let budget = SharedPoolBudgetManager::new(SharedPoolBudgetOptions::default());
    let lutum = Lutum::new(adapter, budget);
    CapabilityProviders::new(
        blackboard,
        Arc::new(NoopAttentionRepository),
        Arc::new(NoopMemoryStore),
        Vec::new(),
        Arc::new(NoopFileSearchProvider),
        Arc::new(NoopUtteranceSink),
        Arc::new(SystemClock),
        LutumTiers {
            cheap: lutum.clone(),
            default: lutum.clone(),
            premium: lutum,
        },
    )
}
