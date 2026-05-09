use lutum::Lutum;
use nuillu_blackboard::Blackboard;
use nuillu_types::ModuleInstanceId;

use crate::LutumTiers;
use crate::rate_limit::{CapabilityKind, RateLimiter};
use crate::runtime_events::RuntimeEventEmitter;

/// LLM-access capability.
///
/// Owner-stamped: `lutum()` reads the current
/// [`ResourceAllocation`](nuillu_blackboard::ResourceAllocation) for the
/// owning module and returns the [`Lutum`] handle bound to the
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
}

impl LlmAccess {
    pub(crate) fn new(
        owner: ModuleInstanceId,
        tiers: LutumTiers,
        blackboard: Blackboard,
        events: RuntimeEventEmitter,
        rate_limiter: RateLimiter,
    ) -> Self {
        Self {
            owner,
            tiers,
            blackboard,
            events,
            rate_limiter,
        }
    }

    /// The [`Lutum`] for the owner module's currently allocated tier.
    /// Tier resolution is per-call so allocation changes between
    /// activations take effect on the next call without re-issuing the
    /// capability.
    pub async fn lutum(&self) -> Lutum {
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

        let cfg = self
            .blackboard
            .read(|bb| bb.allocation().for_module(&self.owner.module))
            .await;
        self.events.llm_accessed(self.owner.clone(), cfg.tier).await;
        self.tiers.pick(cfg.tier)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};
    use std::time::Duration;

    use async_trait::async_trait;
    use lutum::{Lutum, MockLlmAdapter, SharedPoolBudgetManager, SharedPoolBudgetOptions};
    use nuillu_blackboard::{Blackboard, ModuleConfig, ResourceAllocation};
    use nuillu_types::{ModelTier, ModuleInstanceId, ReplicaIndex, builtin};

    use crate::ports::PortError;
    use crate::rate_limit::{CapabilityKind, RateLimitConfig, RateLimitPolicy, RateLimiter};
    use crate::runtime_events::{RuntimeEvent, RuntimeEventEmitter, RuntimeEventSink};

    use super::LlmAccess;

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
        allocation.set(
            builtin::cognition_gate(),
            ModuleConfig {
                tier: ModelTier::Premium,
                ..Default::default()
            },
        );
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
        let access = LlmAccess::new(owner.clone(), tiers, blackboard, events, limiter);

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
}
