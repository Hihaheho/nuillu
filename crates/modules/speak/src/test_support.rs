use std::cell::RefCell;
use std::rc::Rc;

use anyhow::Result;
use async_trait::async_trait;
use nuillu_module::Module;
use nuillu_module::ports::PortError;

use crate::utterance::{Utterance, UtteranceSink};

pub(crate) struct CapturingUtteranceSink {
    pub(crate) completed: Rc<RefCell<Vec<(String, String)>>>,
    pub(crate) done: RefCell<Option<tokio::sync::oneshot::Sender<()>>>,
}

#[async_trait(?Send)]
impl UtteranceSink for CapturingUtteranceSink {
    async fn on_complete(&self, utterance: Utterance) -> Result<(), PortError> {
        self.completed
            .borrow_mut()
            .push((utterance.target, utterance.text));
        if let Some(done) = self.done.borrow_mut().take() {
            let _ = done.send(());
        }
        Ok(())
    }
}

pub(crate) fn test_policy() -> nuillu_blackboard::ModulePolicy {
    nuillu_blackboard::ModulePolicy::new(
        nuillu_types::ReplicaCapRange::new(0, 0).unwrap(),
        nuillu_blackboard::Bpm::from_f64(60.0)..=nuillu_blackboard::Bpm::from_f64(60.0),
        nuillu_blackboard::linear_ratio_fn,
    )
}

macro_rules! noop_stub {
    ($name:ident, $id:literal) => {
        pub(crate) struct $name;

        #[async_trait(?Send)]
        impl Module for $name {
            type Batch = ();

            fn id() -> &'static str {
                $id
            }

            fn peer_context() -> Option<&'static str> {
                Some("test stub")
            }

            async fn next_batch(&mut self) -> Result<Self::Batch> {
                std::future::pending().await
            }

            async fn activate(
                &mut self,
                _cx: &nuillu_module::ActivateCx<'_>,
                _batch: &Self::Batch,
            ) -> Result<()> {
                Ok(())
            }
        }
    };
}

noop_stub!(SpeakStub, "speak");
