use lutum::Lutum;
use nuillu_blackboard::Blackboard;
use nuillu_types::ModuleInstanceId;

use crate::LutumTiers;
use crate::runtime_events::RuntimeEventEmitter;

/// LLM-access capability.
///
/// Owner-stamped: `lutum()` reads the current
/// [`ResourceAllocation`](nuillu_blackboard::ResourceAllocation) for the
/// owning module and returns the [`Lutum`] handle bound to the
/// allocation's tier.
///
/// Modules consume the returned `Lutum` directly — typically by feeding
/// it into `lutum::Session::new(lutum)` for a single-turn or agent-loop
/// activation. The capability deliberately stops at the `Lutum`
/// boundary; everything past it (Session shape, prompts, tools) is the
/// module's own concern.
#[derive(Clone)]
pub struct LlmAccess {
    owner: ModuleInstanceId,
    tiers: LutumTiers,
    blackboard: Blackboard,
    events: RuntimeEventEmitter,
}

impl LlmAccess {
    pub(crate) fn new(
        owner: ModuleInstanceId,
        tiers: LutumTiers,
        blackboard: Blackboard,
        events: RuntimeEventEmitter,
    ) -> Self {
        Self {
            owner,
            tiers,
            blackboard,
            events,
        }
    }

    /// The [`Lutum`] for the owner module's currently allocated tier.
    /// Tier resolution is per-call so allocation changes between
    /// activations take effect on the next call without re-issuing the
    /// capability.
    pub async fn lutum(&self) -> Lutum {
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

    use async_trait::async_trait;
    use lutum::{Lutum, MockLlmAdapter, SharedPoolBudgetManager, SharedPoolBudgetOptions};
    use nuillu_blackboard::{Blackboard, ModuleConfig, ResourceAllocation};
    use nuillu_types::{ModelTier, ModuleInstanceId, ReplicaIndex, builtin};

    use crate::ports::PortError;
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
            builtin::summarize(),
            ModuleConfig {
                tier: ModelTier::Premium,
                ..Default::default()
            },
        );
        let blackboard = Blackboard::with_allocation(allocation);
        let owner = ModuleInstanceId::new(builtin::summarize(), ReplicaIndex::ZERO);
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
        let access = LlmAccess::new(owner.clone(), tiers, blackboard, events);

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
}
