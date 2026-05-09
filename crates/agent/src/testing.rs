//! Test fixtures shared by `#[cfg(test)] mod tests` blocks in this crate.

use std::sync::Arc;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use lutum::{Lutum, MockLlmAdapter, SharedPoolBudgetManager, SharedPoolBudgetOptions};
use nuillu_blackboard::Blackboard;
use nuillu_module::ports::{
    Clock, NoopAttentionRepository, NoopFileSearchProvider, NoopMemoryStore, NoopUtteranceSink,
    SystemClock,
};
use nuillu_module::{CapabilityProviders, LutumTiers, RuntimePolicy};

/// Test clock whose `now()` is the wall clock but whose `sleep_until` returns
/// immediately. Cooldown deadlines and idle timers don't block test
/// wall-clock time.
struct InstantSleepClock;

#[async_trait(?Send)]
impl Clock for InstantSleepClock {
    fn now(&self) -> DateTime<Utc> {
        Utc::now()
    }

    async fn sleep_until(&self, _deadline: DateTime<Utc>) {}
}

pub(crate) fn test_caps(blackboard: Blackboard) -> CapabilityProviders {
    test_caps_inner(blackboard, RuntimePolicy::default(), Arc::new(InstantSleepClock))
}

pub(crate) fn test_caps_with_policy(
    blackboard: Blackboard,
    policy: RuntimePolicy,
) -> CapabilityProviders {
    test_caps_inner(blackboard, policy, Arc::new(InstantSleepClock))
}

/// Like `test_caps` but uses the real wall-clock for sleeps. Pick this when a
/// test specifically asserts that cooldown deadlines block long enough to
/// coalesce subsequent work.
pub(crate) fn test_caps_with_real_clock(blackboard: Blackboard) -> CapabilityProviders {
    test_caps_inner(blackboard, RuntimePolicy::default(), Arc::new(SystemClock))
}

fn test_caps_inner(
    blackboard: Blackboard,
    policy: RuntimePolicy,
    clock: Arc<dyn Clock>,
) -> CapabilityProviders {
    let adapter = Arc::new(MockLlmAdapter::new());
    let budget = SharedPoolBudgetManager::new(SharedPoolBudgetOptions::default());
    let lutum = Lutum::new(adapter, budget);
    CapabilityProviders::new_with_runtime_policy(
        blackboard,
        Arc::new(NoopAttentionRepository),
        Arc::new(NoopMemoryStore),
        Vec::new(),
        Arc::new(NoopFileSearchProvider),
        Arc::new(NoopUtteranceSink),
        clock,
        LutumTiers {
            cheap: lutum.clone(),
            default: lutum.clone(),
            premium: lutum,
        },
        Arc::new(nuillu_module::NoopRuntimeEventSink),
        policy,
    )
}
