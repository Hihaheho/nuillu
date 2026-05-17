use std::cell::RefCell;
use std::rc::Rc;
use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;
use lutum::{
    FinishReason, Lutum, MockLlmAdapter, MockTextScenario, RawTextTurnEvent, Session,
    SharedPoolBudgetManager, SharedPoolBudgetOptions, Usage,
};
use nuillu_blackboard::Blackboard;
use nuillu_module::ports::{NoopCognitionLogRepository, PortError, SystemClock};
use nuillu_module::{CapabilityProviderPorts, CapabilityProviders, LutumTiers, Module};

use crate::utterance::{Utterance, UtteranceSink};

pub(crate) fn test_session() -> Session {
    Session::new()
}

pub(crate) fn test_caps_with_adapter(
    blackboard: Blackboard,
    adapter: MockLlmAdapter,
) -> CapabilityProviders {
    let adapter = Arc::new(adapter);
    let budget = SharedPoolBudgetManager::new(SharedPoolBudgetOptions::default());
    let lutum = Lutum::new(adapter, budget);
    CapabilityProviders::new(CapabilityProviderPorts {
        blackboard,
        cognition_log_port: Rc::new(NoopCognitionLogRepository),
        clock: Rc::new(SystemClock),
        tiers: LutumTiers {
            cheap: lutum.clone(),
            default: lutum.clone(),
            premium: lutum,
        },
    })
}

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

            fn role_description() -> &'static str {
                "test stub"
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

pub(crate) fn target_decision_scenario(target: &str) -> MockTextScenario {
    MockTextScenario::events(vec![
        Ok(RawTextTurnEvent::Started {
            request_id: Some("target".into()),
            model: "mock".into(),
        }),
        Ok(RawTextTurnEvent::ToolCallChunk {
            id: "call-target".into(),
            name: "speak_to".into(),
            arguments_json_delta: serde_json::json!({
                "target": target,
            })
            .to_string(),
        }),
        Ok(RawTextTurnEvent::Completed {
            request_id: Some("target".into()),
            finish_reason: FinishReason::ToolCall,
            usage: Usage::zero(),
        }),
    ])
}

pub(crate) fn no_target_decision_scenario(text: &str) -> MockTextScenario {
    MockTextScenario::events(vec![
        Ok(RawTextTurnEvent::Started {
            request_id: Some("target".into()),
            model: "mock".into(),
        }),
        Ok(RawTextTurnEvent::TextDelta { delta: text.into() }),
        Ok(RawTextTurnEvent::Completed {
            request_id: Some("target".into()),
            finish_reason: FinishReason::Stop,
            usage: Usage::zero(),
        }),
    ])
}

pub(crate) fn empty_target_decision_scenario() -> MockTextScenario {
    MockTextScenario::events(vec![
        Ok(RawTextTurnEvent::Started {
            request_id: Some("target".into()),
            model: "mock".into(),
        }),
        Ok(RawTextTurnEvent::Completed {
            request_id: Some("target".into()),
            finish_reason: FinishReason::Stop,
            usage: Usage::zero(),
        }),
    ])
}

pub(crate) fn generation_text_scenario(text: &str) -> MockTextScenario {
    MockTextScenario::events(vec![
        Ok(RawTextTurnEvent::Started {
            request_id: Some("speak-text".into()),
            model: "mock".into(),
        }),
        Ok(RawTextTurnEvent::TextDelta { delta: text.into() }),
        Ok(RawTextTurnEvent::Completed {
            request_id: Some("speak-text".into()),
            finish_reason: FinishReason::Stop,
            usage: Usage::zero(),
        }),
    ])
}
