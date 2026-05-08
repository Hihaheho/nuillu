use std::time::Duration;

use nuillu_module::PeriodicActivation;

/// Manually driven agent event loop handle.
///
/// Applications advance elapsed time explicitly through [`tick`](Self::tick).
/// The handle applies the current resource allocation and emits periodic
/// module activations through the module inbox registry.
pub struct AgentEventLoop {
    periodic: PeriodicActivation,
}

impl AgentEventLoop {
    pub fn new(periodic: PeriodicActivation) -> Self {
        Self { periodic }
    }

    pub async fn tick(&mut self, elapsed: Duration) {
        self.periodic.tick(elapsed).await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::sync::Arc;

    use async_trait::async_trait;
    use nuillu_blackboard::{Blackboard, BlackboardCommand, ModuleConfig, ResourceAllocation};
    use nuillu_module::{AllocatedModules, Module, ModuleRegistry, PeriodicInbox};
    use nuillu_types::{ModelTier, ModuleId};
    use tokio::sync::Mutex;
    use tokio::task::LocalSet;

    use crate::run;
    use crate::testing::test_caps;

    fn ticker_id() -> ModuleId {
        ModuleId::new("ticker").unwrap()
    }

    struct PeriodicCounter {
        periodic: PeriodicInbox,
        count: Arc<Mutex<u32>>,
    }

    #[async_trait(?Send)]
    impl Module for PeriodicCounter {
        async fn run(&mut self) {
            while self.periodic.next_tick().await.is_ok() {
                *self.count.lock().await += 1;
            }
        }
    }

    struct BatchingPeriodicCounter {
        periodic: PeriodicInbox,
        count: Arc<Mutex<u32>>,
    }

    #[async_trait(?Send)]
    impl Module for BatchingPeriodicCounter {
        async fn run(&mut self) {
            while self.periodic.next_tick().await.is_ok() {
                if self.periodic.take_ready_ticks().is_err() {
                    return;
                }
                *self.count.lock().await += 1;
            }
        }
    }

    async fn periodic_counter_setup(
        period: Duration,
        enabled: bool,
    ) -> (
        Blackboard,
        AllocatedModules,
        Arc<Mutex<u32>>,
        AgentEventLoop,
    ) {
        let mut alloc = ResourceAllocation::default();
        alloc.set(
            ticker_id(),
            ModuleConfig {
                replicas: u8::from(enabled),
                tier: ModelTier::Default,
                period: Some(period),
                ..Default::default()
            },
        );

        let blackboard = Blackboard::with_allocation(alloc);
        let caps = test_caps(blackboard.clone());
        let event_loop = AgentEventLoop::new(caps.periodic_activation());
        let count = Arc::new(Mutex::new(0));
        let modules = ModuleRegistry::new()
            .register(ticker_id(), 0..=1, {
                let count = count.clone();
                move |caps| PeriodicCounter {
                    periodic: caps.periodic_inbox(),
                    count: count.clone(),
                }
            })
            .unwrap()
            .build(&caps)
            .await
            .unwrap();

        (blackboard, modules, count, event_loop)
    }

    async fn set_counter_allocation(blackboard: &Blackboard, period: Duration, enabled: bool) {
        let mut alloc = ResourceAllocation::default();
        alloc.set(
            ticker_id(),
            ModuleConfig {
                replicas: u8::from(enabled),
                tier: ModelTier::Default,
                period: Some(period),
                ..Default::default()
            },
        );
        blackboard
            .apply(BlackboardCommand::SetAllocation(alloc))
            .await;
    }

    #[tokio::test(flavor = "current_thread", start_paused = false)]
    async fn periodic_tick_accumulates_until_period() {
        let local = LocalSet::new();
        local
            .run_until(async {
                let (blackboard, modules, count, mut event_loop) =
                    periodic_counter_setup(Duration::from_millis(10), true).await;

                run(modules, async move {
                    event_loop.tick(Duration::from_millis(6)).await;
                    tokio::task::yield_now().await;
                    assert_eq!(*count.lock().await, 0);

                    event_loop.tick(Duration::from_millis(4)).await;
                    tokio::task::yield_now().await;
                    assert_eq!(*count.lock().await, 1);
                })
                .await
                .expect("scheduler returned err");

                blackboard.read(|_| ()).await;
            })
            .await;
    }

    #[tokio::test(flavor = "current_thread", start_paused = false)]
    async fn periodic_tick_emits_at_most_once_and_carries_remainder() {
        let local = LocalSet::new();
        local
            .run_until(async {
                let (_blackboard, modules, count, mut event_loop) =
                    periodic_counter_setup(Duration::from_millis(10), true).await;

                run(modules, async move {
                    event_loop.tick(Duration::from_millis(25)).await;
                    tokio::task::yield_now().await;
                    assert_eq!(*count.lock().await, 1);

                    event_loop.tick(Duration::from_millis(5)).await;
                    tokio::task::yield_now().await;
                    assert_eq!(*count.lock().await, 2);
                })
                .await
                .expect("scheduler returned err");
            })
            .await;
    }

    #[tokio::test(flavor = "current_thread", start_paused = false)]
    async fn ready_periodic_ticks_can_be_collapsed_by_module_prelude() {
        let local = LocalSet::new();
        local
            .run_until(async {
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

                let blackboard = Blackboard::with_allocation(alloc);
                let caps = test_caps(blackboard);
                let mut event_loop = AgentEventLoop::new(caps.periodic_activation());
                let count = Arc::new(Mutex::new(0));
                let modules = ModuleRegistry::new()
                    .register(ticker_id(), 0..=1, {
                        let count = count.clone();
                        move |caps| BatchingPeriodicCounter {
                            periodic: caps.periodic_inbox(),
                            count: count.clone(),
                        }
                    })
                    .unwrap()
                    .build(&caps)
                    .await
                    .unwrap();

                run(modules, async move {
                    event_loop.tick(Duration::from_millis(10)).await;
                    event_loop.tick(Duration::from_millis(10)).await;
                    event_loop.tick(Duration::from_millis(10)).await;
                    tokio::task::yield_now().await;
                    assert_eq!(*count.lock().await, 1);
                })
                .await
                .expect("scheduler returned err");
            })
            .await;
    }

    #[tokio::test(flavor = "current_thread", start_paused = false)]
    async fn periodic_tick_clears_elapsed_while_disabled() {
        let local = LocalSet::new();
        local
            .run_until(async {
                let (blackboard, modules, count, mut event_loop) =
                    periodic_counter_setup(Duration::from_millis(10), false).await;

                run(modules, async move {
                    event_loop.tick(Duration::from_millis(50)).await;
                    tokio::task::yield_now().await;
                    assert_eq!(*count.lock().await, 0);

                    set_counter_allocation(&blackboard, Duration::from_millis(10), true).await;
                    event_loop.tick(Duration::from_millis(9)).await;
                    tokio::task::yield_now().await;
                    assert_eq!(*count.lock().await, 0);

                    event_loop.tick(Duration::from_millis(1)).await;
                    tokio::task::yield_now().await;
                    assert_eq!(*count.lock().await, 1);
                })
                .await
                .expect("scheduler returned err");
            })
            .await;
    }

    #[tokio::test(flavor = "current_thread", start_paused = false)]
    async fn periodic_tick_uses_latest_allocation_period() {
        let local = LocalSet::new();
        local
            .run_until(async {
                let (blackboard, modules, count, mut event_loop) =
                    periodic_counter_setup(Duration::from_millis(20), true).await;

                run(modules, async move {
                    event_loop.tick(Duration::from_millis(15)).await;
                    tokio::task::yield_now().await;
                    assert_eq!(*count.lock().await, 0);

                    set_counter_allocation(&blackboard, Duration::from_millis(10), true).await;
                    event_loop.tick(Duration::ZERO).await;
                    tokio::task::yield_now().await;
                    assert_eq!(*count.lock().await, 1);
                })
                .await
                .expect("scheduler returned err");
            })
            .await;
    }
}
