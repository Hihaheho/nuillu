use nuillu_blackboard::Blackboard;
use nuillu_types::ModuleInstanceId;

/// Owner-scoped gate that parks a persistent replica while allocation excludes it.
///
/// The gate exposes only whether the holder instance may proceed. It does not
/// expose allocation details to module code.
#[derive(Clone)]
pub struct ActivationGate {
    owner: ModuleInstanceId,
    blackboard: Blackboard,
}

impl ActivationGate {
    pub(crate) fn new(owner: ModuleInstanceId, blackboard: Blackboard) -> Self {
        Self { owner, blackboard }
    }

    pub async fn block(&self) {
        while let Some(waiter) = self.blackboard.activation_waiter(self.owner.clone()).await {
            if waiter.await.is_err() {
                return;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::time::Duration;

    use nuillu_blackboard::{Blackboard, BlackboardCommand, ModuleConfig, ResourceAllocation};
    use nuillu_types::{ModelTier, ModuleId};
    use tokio::sync::Mutex;
    use tokio::task::LocalSet;

    use crate::test_support::{scoped, test_caps};

    fn ticker_id() -> ModuleId {
        ModuleId::new("ticker").unwrap()
    }

    async fn enable_ticker(blackboard: &Blackboard) {
        let mut alloc = ResourceAllocation::default();
        alloc.set(
            ticker_id(),
            ModuleConfig {
                replicas: 1,
                tier: ModelTier::Default,
                period: Some(Duration::from_millis(10)),
                ..Default::default()
            },
        );
        blackboard
            .apply(BlackboardCommand::SetAllocation(alloc))
            .await;
    }

    #[tokio::test]
    async fn activation_gate_blocks_until_replica_is_enabled() {
        let local = LocalSet::new();
        local
            .run_until(async {
                let mut alloc = ResourceAllocation::default();
                alloc.set(
                    ticker_id(),
                    ModuleConfig {
                        replicas: 0,
                        ..Default::default()
                    },
                );
                let blackboard = Blackboard::with_allocation(alloc);
                let caps = test_caps(blackboard.clone());
                let gate = scoped(&caps, ticker_id(), 0).activation_gate();
                let passed = Arc::new(Mutex::new(false));
                let passed_for_task = passed.clone();

                let handle = tokio::task::spawn_local(async move {
                    gate.block().await;
                    *passed_for_task.lock().await = true;
                });

                for _ in 0..3 {
                    tokio::task::yield_now().await;
                }
                assert!(!*passed.lock().await);

                enable_ticker(&blackboard).await;
                for _ in 0..10 {
                    if *passed.lock().await {
                        break;
                    }
                    tokio::task::yield_now().await;
                }

                assert!(*passed.lock().await);
                handle.await.unwrap();
            })
            .await;
    }
}
