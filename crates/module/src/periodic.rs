use std::collections::HashMap;
use std::sync::Arc;
use std::sync::RwLock;
use std::time::Duration;

use nuillu_blackboard::Blackboard;
use nuillu_types::ModuleId;
use tokio::sync::mpsc;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PeriodicTick;

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum PeriodicRecvError {
    #[error("periodic inbox closed")]
    Closed,
}

/// Per-module periodic activation receiver.
///
/// Holding this capability is what makes a module eligible for allocation-
/// controlled periodic activations.
pub struct PeriodicInbox {
    owner: ModuleId,
    receiver: mpsc::Receiver<PeriodicTick>,
}

impl PeriodicInbox {
    pub(crate) fn new(owner: ModuleId, receiver: mpsc::Receiver<PeriodicTick>) -> Self {
        Self { owner, receiver }
    }

    pub fn owner(&self) -> &ModuleId {
        &self.owner
    }

    pub async fn next_tick(&mut self) -> Result<(), PeriodicRecvError> {
        self.receiver
            .recv()
            .await
            .map(|_| ())
            .ok_or(PeriodicRecvError::Closed)
    }

    pub fn take_ready_ticks(&mut self) -> Result<usize, PeriodicRecvError> {
        let mut count = 0;
        loop {
            match self.receiver.try_recv() {
                Ok(_) => count += 1,
                Err(mpsc::error::TryRecvError::Empty) => return Ok(count),
                Err(mpsc::error::TryRecvError::Disconnected) => {
                    return Err(PeriodicRecvError::Closed);
                }
            }
        }
    }
}

/// Registry of modules that were explicitly granted [`PeriodicInbox`].
pub struct PeriodicRegistry {
    senders: RwLock<HashMap<ModuleId, mpsc::Sender<PeriodicTick>>>,
}

impl PeriodicRegistry {
    pub(crate) fn new() -> Self {
        Self {
            senders: RwLock::new(HashMap::new()),
        }
    }

    pub(crate) fn register(&self, owner: ModuleId, capacity: usize) -> PeriodicInbox {
        let (tx, rx) = mpsc::channel(capacity);
        self.senders
            .write()
            .expect("PeriodicRegistry poisoned")
            .insert(owner.clone(), tx);
        PeriodicInbox::new(owner, rx)
    }

    fn module_ids(&self) -> Vec<ModuleId> {
        self.senders
            .read()
            .expect("PeriodicRegistry poisoned")
            .keys()
            .cloned()
            .collect()
    }

    fn try_send(&self, module: &ModuleId) -> Result<(), PeriodicTick> {
        let map = self.senders.read().expect("PeriodicRegistry poisoned");
        let Some(sender) = map.get(module) else {
            return Err(PeriodicTick);
        };
        sender
            .try_send(PeriodicTick)
            .map_err(|err| err.into_inner())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_id() -> ModuleId {
        ModuleId::new("test-module").unwrap()
    }

    #[test]
    fn take_ready_ticks_counts_ready_ticks_and_stops_on_empty() {
        let (tx, rx) = mpsc::channel(4);
        let mut inbox = PeriodicInbox::new(test_id(), rx);
        tx.try_send(PeriodicTick).unwrap();
        tx.try_send(PeriodicTick).unwrap();

        assert_eq!(inbox.take_ready_ticks(), Ok(2));
        assert_eq!(inbox.take_ready_ticks(), Ok(0));
    }

    #[tokio::test]
    async fn next_tick_distinguishes_ready_and_closed() {
        let (tx, rx) = mpsc::channel(1);
        let mut inbox = PeriodicInbox::new(test_id(), rx);
        tx.try_send(PeriodicTick).unwrap();

        assert_eq!(inbox.next_tick().await, Ok(()));
        drop(tx);
        assert_eq!(inbox.next_tick().await, Err(PeriodicRecvError::Closed));
    }
}

/// Allocation-aware periodic trigger driver.
///
/// The application or agent runtime advances this handle with elapsed
/// time. Each call reads the current resource allocation and delivers at
/// most one [`PeriodicTick`] per module holding a [`PeriodicInbox`].
pub struct PeriodicActivation {
    blackboard: Blackboard,
    registry: Arc<PeriodicRegistry>,
    elapsed_by_module: HashMap<ModuleId, Duration>,
}

impl PeriodicActivation {
    pub(crate) fn new(blackboard: Blackboard, registry: Arc<PeriodicRegistry>) -> Self {
        Self {
            blackboard,
            registry,
            elapsed_by_module: HashMap::new(),
        }
    }

    /// Advance periodic activation by `elapsed`.
    ///
    /// Modules without a positive period, or modules disabled in the
    /// current allocation, do not receive periodic triggers and have
    /// their accumulated elapsed time cleared.
    pub async fn tick(&mut self, elapsed: Duration) {
        let ids = self.registry.module_ids();
        let allocation = self.blackboard.read(|bb| bb.allocation().clone()).await;

        for id in ids {
            let cfg = allocation.for_module(&id);
            let Some(period) = cfg.period else {
                self.elapsed_by_module.remove(&id);
                continue;
            };
            if !cfg.enabled || period == Duration::ZERO {
                self.elapsed_by_module.remove(&id);
                continue;
            }

            let accumulated = self
                .elapsed_by_module
                .entry(id.clone())
                .and_modify(|d| *d += elapsed)
                .or_insert(elapsed);

            if *accumulated < period {
                continue;
            }

            match self.registry.try_send(&id) {
                Ok(()) => {
                    *accumulated -= period;
                }
                Err(PeriodicTick) => {
                    tracing::trace!(
                        module = %id,
                        "periodic inbox unavailable; elapsed retained for retry",
                    );
                }
            }
        }
    }
}
