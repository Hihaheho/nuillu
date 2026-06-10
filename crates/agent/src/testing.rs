//! Test fixtures shared by `#[cfg(test)] mod tests` blocks in this crate.

use std::rc::Rc;
use std::sync::Arc;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use lutum::{Lutum, MockLlmAdapter, SharedPoolBudgetManager, SharedPoolBudgetOptions};
use nuillu_blackboard::Blackboard;
use nuillu_module::ports::{Clock, NoopCognitionLogRepository, SystemClock};
use nuillu_module::{
    CapabilityProviderConfig, CapabilityProviderPorts, CapabilityProviderRuntime,
    CapabilityProviders, LutumTiers, RuntimeEventSink, RuntimePolicy,
};

/// Test clock whose `now()` is the wall clock but whose `sleep_until` returns
/// immediately. Period deadlines and idle timers don't block test
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
    test_caps_inner(
        blackboard,
        RuntimePolicy::default(),
        Rc::new(InstantSleepClock),
    )
}

/// Like `test_caps` but uses the real wall-clock for sleeps. Pick this when a
/// test specifically asserts that period deadlines block long enough to
/// coalesce subsequent work.
pub(crate) fn test_caps_with_real_clock(blackboard: Blackboard) -> CapabilityProviders {
    test_caps_inner(blackboard, RuntimePolicy::default(), Rc::new(SystemClock))
}

pub(crate) fn test_caps_with_policy(
    blackboard: Blackboard,
    policy: RuntimePolicy,
) -> CapabilityProviders {
    test_caps_inner(blackboard, policy, Rc::new(InstantSleepClock))
}

pub(crate) fn test_caps_with_event_sink(
    blackboard: Blackboard,
    event_sink: Rc<dyn RuntimeEventSink>,
) -> CapabilityProviders {
    test_caps_inner_with_runtime(
        blackboard,
        CapabilityProviderRuntime {
            event_sink,
            ..CapabilityProviderRuntime::default()
        },
        Rc::new(InstantSleepClock),
    )
}

fn test_caps_inner(
    blackboard: Blackboard,
    policy: RuntimePolicy,
    clock: Rc<dyn Clock>,
) -> CapabilityProviders {
    test_caps_inner_with_runtime(
        blackboard,
        CapabilityProviderRuntime {
            policy,
            ..CapabilityProviderRuntime::default()
        },
        clock,
    )
}

fn test_caps_inner_with_runtime(
    blackboard: Blackboard,
    runtime: CapabilityProviderRuntime,
    clock: Rc<dyn Clock>,
) -> CapabilityProviders {
    let adapter = Arc::new(MockLlmAdapter::new());
    let budget = SharedPoolBudgetManager::new(SharedPoolBudgetOptions::default());
    let lutum = Lutum::new(adapter, budget);
    CapabilityProviders::new(CapabilityProviderConfig {
        ports: CapabilityProviderPorts {
            blackboard,
            cognition_log_port: Rc::new(NoopCognitionLogRepository),
            clock,
            tiers: LutumTiers::from_shared_lutum(lutum),
        },
        runtime,
    })
}
