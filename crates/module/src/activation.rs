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
