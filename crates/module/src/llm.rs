use std::num::NonZeroUsize;
use std::ops::Deref;
use std::sync::Arc;

use lutum::Lutum;
use nuillu_blackboard::Blackboard;
use nuillu_types::ModuleInstanceId;
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

use crate::LutumTiers;
use crate::rate_limit::{CapabilityKind, RateLimiter};
use crate::runtime_events::RuntimeEventEmitter;

/// Shared admission control for LLM turns.
#[derive(Clone)]
pub(crate) struct LlmConcurrencyLimiter {
    semaphore: Option<Arc<Semaphore>>,
}

impl LlmConcurrencyLimiter {
    pub(crate) fn new(max_concurrent_calls: Option<NonZeroUsize>) -> Self {
        Self {
            semaphore: max_concurrent_calls.map(|max| Arc::new(Semaphore::new(max.get()))),
        }
    }

    async fn acquire(&self) -> Option<OwnedSemaphorePermit> {
        let semaphore = self.semaphore.as_ref()?;
        Some(
            semaphore
                .clone()
                .acquire_owned()
                .await
                .expect("LLM concurrency semaphore is never closed"),
        )
    }
}

/// A [`Lutum`] handle plus any runtime admission permit held for its use.
///
/// Dropping this value releases the concurrent-call slot. It dereferences to
/// [`Lutum`] so module code can bind the value and pass `&lutum`.
pub struct LlmLease {
    lutum: Lutum,
    _permit: Option<OwnedSemaphorePermit>,
}

impl LlmLease {
    fn new(lutum: Lutum, permit: Option<OwnedSemaphorePermit>) -> Self {
        Self {
            lutum,
            _permit: permit,
        }
    }

    pub fn lutum(&self) -> &Lutum {
        &self.lutum
    }
}

impl Deref for LlmLease {
    type Target = Lutum;

    fn deref(&self) -> &Self::Target {
        &self.lutum
    }
}

/// LLM-access capability.
///
/// Owner-stamped: `lutum()` reads the current
/// [`ResourceAllocation`](nuillu_blackboard::ResourceAllocation) for the
/// owning module and returns a leased [`Lutum`] handle bound to the
/// allocation's tier.
///
/// Modules consume the returned `Lutum` directly — typically by passing
/// it into `lutum::Session`'s turn builders (e.g. `session.text_turn(&lutum)`)
/// for a single-turn or agent-loop activation. The capability deliberately stops at the `Lutum`
/// boundary; everything past it (Session shape, prompts, tools) is the
/// module's own concern.
#[derive(Clone)]
pub struct LlmAccess {
    owner: ModuleInstanceId,
    tiers: LutumTiers,
    blackboard: Blackboard,
    events: RuntimeEventEmitter,
    rate_limiter: RateLimiter,
    concurrency_limiter: LlmConcurrencyLimiter,
}

impl LlmAccess {
    pub(crate) fn new(
        owner: ModuleInstanceId,
        tiers: LutumTiers,
        blackboard: Blackboard,
        events: RuntimeEventEmitter,
        rate_limiter: RateLimiter,
        concurrency_limiter: LlmConcurrencyLimiter,
    ) -> Self {
        Self {
            owner,
            tiers,
            blackboard,
            events,
            rate_limiter,
            concurrency_limiter,
        }
    }

    /// A leased [`Lutum`] for the owner module's currently allocated tier.
    /// Tier resolution is per-call so allocation changes between
    /// activations take effect on the next call without re-issuing the
    /// capability.
    pub async fn lutum(&self) -> LlmLease {
        let outcome = self
            .rate_limiter
            .acquire(&self.owner, CapabilityKind::LlmCall)
            .await;
        if outcome.was_delayed() {
            self.events
                .rate_limit_delayed(
                    self.owner.clone(),
                    CapabilityKind::LlmCall,
                    outcome.delayed_for,
                )
                .await;
        }
        let permit = self.concurrency_limiter.acquire().await;

        let tier = self
            .blackboard
            .read(|bb| bb.allocation().tier_for(&self.owner.module))
            .await;
        self.events.llm_accessed(self.owner.clone(), tier).await;
        LlmLease::new(self.tiers.pick(tier), permit)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};
    use std::time::Duration;

    use async_trait::async_trait;
    use lutum::{Lutum, MockLlmAdapter, SharedPoolBudgetManager, SharedPoolBudgetOptions};
    use nuillu_blackboard::{Blackboard, ResourceAllocation};
    use nuillu_types::{ModelTier, ModuleInstanceId, ReplicaIndex, builtin};

    use crate::ports::PortError;
    use crate::rate_limit::{CapabilityKind, RateLimitConfig, RateLimitPolicy, RateLimiter};
    use crate::runtime_events::{RuntimeEvent, RuntimeEventEmitter, RuntimeEventSink};

    use super::{LlmAccess, LlmConcurrencyLimiter};

    #[derive(Debug, Default)]
    struct RecordingSink {
        events: Mutex<Vec<RuntimeEvent>>,
    }

    #[async_trait(?Send)]
    impl RuntimeEventSink for RecordingSink {
        async fn on_event(&self, event: RuntimeEvent) -> Result<(), PortError> {
            self.events.lock().expect("event lock poisoned").push(event);
            Ok(())
        }
    }

    #[tokio::test]
    async fn lutum_emits_one_runtime_event_per_acquisition() {
        let mut allocation = ResourceAllocation::default();
        allocation.set_model_override(builtin::cognition_gate(), ModelTier::Premium);
        let blackboard = Blackboard::with_allocation(allocation);
        let owner = ModuleInstanceId::new(builtin::cognition_gate(), ReplicaIndex::ZERO);
        let adapter = Arc::new(MockLlmAdapter::new());
        let budget = SharedPoolBudgetManager::new(SharedPoolBudgetOptions::default());
        let lutum = Lutum::new(adapter, budget);
        let tiers = crate::LutumTiers {
            cheap: lutum.clone(),
            default: lutum.clone(),
            premium: lutum,
        };
        let sink = Arc::new(RecordingSink::default());
        let events = RuntimeEventEmitter::new(sink.clone());
        let access = LlmAccess::new(
            owner.clone(),
            tiers,
            blackboard,
            events,
            RateLimiter::disabled(),
            LlmConcurrencyLimiter::new(None),
        );

        let _ = access.lutum().await;
        let _ = access.lutum().await;

        let actual = sink.events.lock().expect("event lock poisoned").clone();
        assert_eq!(
            actual,
            vec![
                RuntimeEvent::LlmAccessed {
                    sequence: 0,
                    call: 0,
                    owner: owner.clone(),
                    tier: ModelTier::Premium,
                },
                RuntimeEvent::LlmAccessed {
                    sequence: 1,
                    call: 1,
                    owner,
                    tier: ModelTier::Premium,
                },
            ]
        );
    }

    #[tokio::test]
    async fn lutum_waits_for_rate_limit_before_access_event() {
        let blackboard = Blackboard::default();
        let owner = ModuleInstanceId::new(builtin::cognition_gate(), ReplicaIndex::ZERO);
        let adapter = Arc::new(MockLlmAdapter::new());
        let budget = SharedPoolBudgetManager::new(SharedPoolBudgetOptions::default());
        let lutum = Lutum::new(adapter, budget);
        let tiers = crate::LutumTiers {
            cheap: lutum.clone(),
            default: lutum.clone(),
            premium: lutum,
        };
        let sink = Arc::new(RecordingSink::default());
        let events = RuntimeEventEmitter::new(sink.clone());
        let limiter = RateLimiter::new(
            RateLimitPolicy::for_module(
                owner.module.clone(),
                CapabilityKind::LlmCall,
                RateLimitConfig::new(Duration::from_millis(10), 100.0).unwrap(),
            )
            .unwrap(),
        );
        let access = LlmAccess::new(
            owner.clone(),
            tiers,
            blackboard,
            events,
            limiter,
            LlmConcurrencyLimiter::new(None),
        );

        let _ = access.lutum().await;
        let _ = access.lutum().await;

        let actual = sink.events.lock().expect("event lock poisoned").clone();
        assert_eq!(actual.len(), 3);
        assert!(matches!(
            actual[0],
            RuntimeEvent::LlmAccessed {
                sequence: 0,
                call: 0,
                ..
            }
        ));
        assert!(matches!(
            &actual[1],
            RuntimeEvent::RateLimitDelayed {
                sequence: 1,
                owner: delayed_owner,
                capability: CapabilityKind::LlmCall,
                delayed_for,
            } if delayed_owner == &owner && *delayed_for > Duration::ZERO
        ));
        assert!(matches!(
            actual[2],
            RuntimeEvent::LlmAccessed {
                sequence: 2,
                call: 1,
                ..
            }
        ));
    }

    #[tokio::test]
    async fn lutum_waits_for_concurrency_permit_until_prior_lease_is_dropped() {
        let blackboard = Blackboard::default();
        let owner = ModuleInstanceId::new(builtin::cognition_gate(), ReplicaIndex::ZERO);
        let adapter = Arc::new(MockLlmAdapter::new());
        let budget = SharedPoolBudgetManager::new(SharedPoolBudgetOptions::default());
        let lutum = Lutum::new(adapter, budget);
        let tiers = crate::LutumTiers {
            cheap: lutum.clone(),
            default: lutum.clone(),
            premium: lutum,
        };
        let sink = Arc::new(RecordingSink::default());
        let events = RuntimeEventEmitter::new(sink);
        let access = LlmAccess::new(
            owner,
            tiers,
            blackboard,
            events,
            RateLimiter::disabled(),
            LlmConcurrencyLimiter::new(Some(std::num::NonZeroUsize::new(1).unwrap())),
        );

        let first = access.lutum().await;
        let second = tokio::time::timeout(Duration::from_millis(5), access.lutum()).await;
        assert!(second.is_err());

        drop(first);
        let second = tokio::time::timeout(Duration::from_millis(50), access.lutum()).await;
        assert!(second.is_ok());
    }
}
