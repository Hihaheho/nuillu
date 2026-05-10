use std::any::TypeId;
use std::cell::RefCell;
use std::marker::PhantomData;
use std::rc::Rc;
use std::sync::{Arc, Mutex};

use futures::StreamExt;
use futures::channel::mpsc;
use nuillu_blackboard::Blackboard;
use nuillu_types::{ModuleId, ModuleInstanceId};
use tokio::sync::oneshot;

use crate::{Module, ModuleBatch};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActivationGateVote {
    Allow,
    Suppress,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum ActivationGateRecvError {
    #[error("activation gate inbox closed")]
    Closed,
}

pub struct ActivationGateEvent<M: Module> {
    target: ModuleInstanceId,
    batch: Rc<M::Batch>,
    response: Rc<RefCell<Option<oneshot::Sender<ActivationGateVote>>>>,
}

impl<M: Module> std::fmt::Debug for ActivationGateEvent<M> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ActivationGateEvent")
            .field("target", &self.target)
            .field("has_response", &self.response.borrow().is_some())
            .finish_non_exhaustive()
    }
}

impl<M: Module> ActivationGateEvent<M> {
    pub fn target(&self) -> &ModuleInstanceId {
        &self.target
    }

    pub fn batch(&self) -> &M::Batch {
        self.batch.as_ref()
    }

    pub fn respond(&self, vote: ActivationGateVote) {
        if let Some(sender) = self.response.borrow_mut().take() {
            let _ = sender.send(vote);
        }
    }
}

struct ErasedActivationGateEvent {
    target: ModuleInstanceId,
    batch: ModuleBatch,
    response: oneshot::Sender<ActivationGateVote>,
}

pub struct ActivationGate<M: Module> {
    receiver: mpsc::UnboundedReceiver<ErasedActivationGateEvent>,
    _marker: PhantomData<fn() -> M>,
}

impl<M: Module> ActivationGate<M> {
    fn new(receiver: mpsc::UnboundedReceiver<ErasedActivationGateEvent>) -> Self {
        Self {
            receiver,
            _marker: PhantomData,
        }
    }

    pub async fn next_event(&mut self) -> Result<ActivationGateEvent<M>, ActivationGateRecvError> {
        while let Some(event) = self.receiver.next().await {
            if let Some(batch) = event.batch.downcast_rc::<M::Batch>() {
                return Ok(ActivationGateEvent {
                    target: event.target,
                    batch,
                    response: Rc::new(RefCell::new(Some(event.response))),
                });
            }
        }
        Err(ActivationGateRecvError::Closed)
    }
}

#[derive(Clone)]
pub(crate) struct ActivationGateHub {
    inner: Arc<Mutex<ActivationGateHubInner>>,
    blackboard: Blackboard,
}

impl ActivationGateHub {
    pub(crate) fn new(blackboard: Blackboard) -> Self {
        Self {
            inner: Arc::new(Mutex::new(ActivationGateHubInner::default())),
            blackboard,
        }
    }

    pub(crate) fn subscribe<M: Module + 'static>(
        &self,
        owner: ModuleInstanceId,
    ) -> ActivationGate<M> {
        let (sender, receiver) = mpsc::unbounded();
        let target_module = ModuleId::new(M::id()).expect("module id is valid");
        self.inner
            .lock()
            .expect("activation gate hub poisoned")
            .subscribers
            .push(ActivationGateSubscriber {
                owner,
                target_module,
                batch_type: TypeId::of::<M::Batch>(),
                sender,
            });
        ActivationGate::new(receiver)
    }

    pub(crate) async fn dispatch(
        &self,
        target: &ModuleInstanceId,
        batch: ModuleBatch,
    ) -> Vec<oneshot::Receiver<ActivationGateVote>> {
        let allocation = self.blackboard.read(|bb| bb.allocation().clone()).await;
        let batch_type = batch.type_id();
        let mut receivers = Vec::new();
        let mut inner = self.inner.lock().expect("activation gate hub poisoned");

        inner
            .subscribers
            .retain(|subscriber| !subscriber.sender.is_closed());

        for subscriber in inner.subscribers.iter().filter(|subscriber| {
            subscriber.target_module == target.module
                && subscriber.batch_type == batch_type
                && allocation.is_replica_active(&subscriber.owner)
        }) {
            let (response, receiver) = oneshot::channel();
            let event = ErasedActivationGateEvent {
                target: target.clone(),
                batch: batch.clone(),
                response,
            };
            if subscriber.sender.unbounded_send(event).is_ok() {
                receivers.push(receiver);
            }
        }

        receivers
    }
}

#[derive(Default)]
struct ActivationGateHubInner {
    subscribers: Vec<ActivationGateSubscriber>,
}

struct ActivationGateSubscriber {
    owner: ModuleInstanceId,
    target_module: ModuleId,
    batch_type: TypeId,
    sender: mpsc::UnboundedSender<ErasedActivationGateEvent>,
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use async_trait::async_trait;
    use nuillu_blackboard::{
        ActivationRatio, Blackboard, BlackboardCommand, Bpm, ModuleConfig, ModulePolicy,
        ResourceAllocation, linear_ratio_fn,
    };
    use nuillu_types::{ReplicaCapRange, ReplicaIndex};

    use super::*;
    use crate::test_support::{scoped, test_caps};
    use crate::{ActivateCx, Module};

    struct TargetModule;

    #[async_trait(?Send)]
    impl Module for TargetModule {
        type Batch = String;

        fn id() -> &'static str {
            "gate-target"
        }

        fn role_description() -> &'static str {
            "test target"
        }

        async fn next_batch(&mut self) -> anyhow::Result<Self::Batch> {
            Ok(String::new())
        }

        async fn activate(
            &mut self,
            _cx: &ActivateCx<'_>,
            _batch: &Self::Batch,
        ) -> anyhow::Result<()> {
            Ok(())
        }
    }

    fn gate_id() -> ModuleId {
        ModuleId::new("gate-observer").unwrap()
    }

    fn target_owner() -> ModuleInstanceId {
        ModuleInstanceId::new(
            ModuleId::new(TargetModule::id()).unwrap(),
            ReplicaIndex::ZERO,
        )
    }

    async fn install_gate_policy(blackboard: &Blackboard) {
        blackboard
            .apply(BlackboardCommand::SetModulePolicies {
                policies: vec![(
                    gate_id(),
                    ModulePolicy::new(
                        ReplicaCapRange::new(0, 2).unwrap(),
                        Bpm::from_f64(60.0)..=Bpm::from_f64(60.0),
                        linear_ratio_fn,
                    ),
                )],
            })
            .await;
    }

    #[tokio::test]
    async fn dispatches_only_to_active_gate_replicas() {
        let mut allocation = ResourceAllocation::default();
        allocation.set(gate_id(), ModuleConfig::default());
        allocation.set_activation(gate_id(), ActivationRatio::from_f64(0.5));
        let blackboard = Blackboard::with_allocation(allocation);
        install_gate_policy(&blackboard).await;
        let caps = test_caps(blackboard);
        let mut gate_0 = scoped(&caps, gate_id(), 0).activation_gate_for::<TargetModule>();
        let mut gate_1 = scoped(&caps, gate_id(), 1).activation_gate_for::<TargetModule>();

        let requests = caps
            .runtime_control()
            .activation_gate_requests(&target_owner(), ModuleBatch::new("candidate".to_string()))
            .await;

        assert_eq!(requests.len(), 1);
        let event = gate_0.next_event().await.unwrap();
        assert_eq!(event.target(), &target_owner());
        assert_eq!(event.batch(), "candidate");
        assert!(
            tokio::time::timeout(Duration::from_millis(1), gate_1.next_event())
                .await
                .is_err()
        );
    }

    #[tokio::test]
    async fn dispatch_skips_fully_disabled_gate_without_replica_zero_fallback() {
        let mut allocation = ResourceAllocation::default();
        allocation.set(gate_id(), ModuleConfig::default());
        allocation.set_activation(gate_id(), ActivationRatio::ZERO);
        let blackboard = Blackboard::with_allocation(allocation);
        install_gate_policy(&blackboard).await;
        let caps = test_caps(blackboard);
        let _gate_0 = scoped(&caps, gate_id(), 0).activation_gate_for::<TargetModule>();

        let requests = caps
            .runtime_control()
            .activation_gate_requests(&target_owner(), ModuleBatch::new("candidate".to_string()))
            .await;

        assert!(requests.is_empty());
    }
}
