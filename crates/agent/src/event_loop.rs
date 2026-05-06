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
