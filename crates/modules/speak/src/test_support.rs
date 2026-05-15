use std::cell::RefCell;
use std::rc::Rc;
use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;
use lutum::{
    FinishReason, Lutum, MockLlmAdapter, MockStructuredScenario, MockTextScenario,
    RawStructuredTurnEvent, RawTextTurnEvent, Session, SharedPoolBudgetManager,
    SharedPoolBudgetOptions, Usage,
};
use nuillu_blackboard::{ActivationRatio, Blackboard, ModuleConfig, ResourceAllocation};
use nuillu_module::ports::{NoopCognitionLogRepository, PortError, SystemClock};
use nuillu_module::{
    AttentionControlRequestInbox, CapabilityProviderPorts, CapabilityProviders, LutumTiers, Module,
    ModuleRegistry,
};
use nuillu_types::builtin;

use crate::utterance::{Utterance, UtteranceSink};
use crate::{SpeakGateMemo, SpeakGateModule, SpeakModule};

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

pub(crate) fn tool_test_allocation() -> ResourceAllocation {
    let mut allocation = ResourceAllocation::default();
    for module in [
        builtin::speak_gate(),
        builtin::allocation_controller(),
        builtin::query_memory(),
        builtin::self_model(),
        builtin::sensory(),
        builtin::speak(),
    ] {
        allocation.set(module.clone(), ModuleConfig::default());
        allocation.set_activation(module, ActivationRatio::ONE);
    }
    allocation
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

noop_stub!(SpeakGateStub, "speak-gate");
noop_stub!(SpeakStub, "speak");
noop_stub!(AllocationControllerStub, "allocation-controller");

pub(crate) struct GateToolFixture {
    pub(crate) gate: SpeakGateModule,
    pub(crate) blackboard: Blackboard,
    pub(crate) attention_control_inbox: AttentionControlRequestInbox,
}

pub(crate) async fn gate_tool_fixture() -> GateToolFixture {
    gate_tool_fixture_with_adapter(MockLlmAdapter::new()).await
}

pub(crate) async fn gate_tool_fixture_with_adapter(adapter: MockLlmAdapter) -> GateToolFixture {
    let blackboard = Blackboard::with_allocation(tool_test_allocation());
    let caps = test_caps_with_adapter(blackboard.clone(), adapter);

    let gate_cell = Rc::new(RefCell::new(None));
    let attention_control_inbox_cell = Rc::new(RefCell::new(None));

    let gate_sink = Rc::clone(&gate_cell);
    let attention_control_inbox_sink = Rc::clone(&attention_control_inbox_cell);

    let _modules = ModuleRegistry::new()
        .register(test_policy(), move |caps| {
            *gate_sink.borrow_mut() = Some(SpeakGateModule::new(
                caps.activation_gate_for::<SpeakModule>(),
                caps.cognition_log_reader(),
                caps.blackboard_reader(),
                caps.attention_control_mailbox(),
                caps.typed_memo::<SpeakGateMemo>(),
                caps.llm_access(),
            ));
            SpeakGateStub
        })
        .unwrap()
        .register(test_policy(), move |caps| {
            *attention_control_inbox_sink.borrow_mut() = Some(caps.attention_control_inbox());
            AllocationControllerStub
        })
        .unwrap()
        .build(&caps)
        .await
        .unwrap();

    GateToolFixture {
        gate: gate_cell.borrow_mut().take().unwrap(),
        blackboard,
        attention_control_inbox: attention_control_inbox_cell.borrow_mut().take().unwrap(),
    }
}

fn structured_usage(input_tokens: u64) -> Usage {
    Usage {
        input_tokens,
        ..Usage::zero()
    }
}

pub(crate) fn finished_decision_scenario(
    input_tokens: u64,
    rationale: &str,
) -> MockStructuredScenario {
    MockStructuredScenario::events(vec![
        Ok(RawStructuredTurnEvent::Started {
            request_id: Some("gate-finished".into()),
            model: "mock".into(),
        }),
        Ok(RawStructuredTurnEvent::StructuredOutputChunk {
            json_delta: serde_json::json!({
                "wants_to_speak": false,
                "wait_for_evidence": false,
                "rationale": rationale,
                "evidence_gaps": [],
            })
            .to_string(),
        }),
        Ok(RawStructuredTurnEvent::Completed {
            request_id: Some("gate-finished".into()),
            finish_reason: FinishReason::Stop,
            usage: structured_usage(input_tokens),
        }),
    ])
}

pub(crate) fn target_decision_scenario(target: &str) -> MockStructuredScenario {
    MockStructuredScenario::events(vec![
        Ok(RawStructuredTurnEvent::Started {
            request_id: Some("target".into()),
            model: "mock".into(),
        }),
        Ok(RawStructuredTurnEvent::StructuredOutputChunk {
            json_delta: serde_json::json!({
                "target": target,
            })
            .to_string(),
        }),
        Ok(RawStructuredTurnEvent::Completed {
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

pub(crate) fn tool_call_decision_scenario(input_tokens: u64) -> MockStructuredScenario {
    MockStructuredScenario::events(vec![
        Ok(RawStructuredTurnEvent::Started {
            request_id: Some("gate-tool".into()),
            model: "mock".into(),
        }),
        Ok(RawStructuredTurnEvent::ToolCallChunk {
            id: "memory-1".into(),
            name: "query_memory".into(),
            arguments_json_delta: serde_json::json!({
                "question": "What should I remember?"
            })
            .to_string(),
        }),
        Ok(RawStructuredTurnEvent::Completed {
            request_id: Some("gate-tool".into()),
            finish_reason: FinishReason::ToolCall,
            usage: structured_usage(input_tokens),
        }),
    ])
}

pub(crate) fn summary_text_scenario(summary: &str) -> MockTextScenario {
    MockTextScenario::events(vec![
        Ok(RawTextTurnEvent::Started {
            request_id: Some("gate-compact".into()),
            model: "mock".into(),
        }),
        Ok(RawTextTurnEvent::TextDelta {
            delta: summary.into(),
        }),
        Ok(RawTextTurnEvent::Completed {
            request_id: Some("gate-compact".into()),
            finish_reason: FinishReason::Stop,
            usage: Usage::zero(),
        }),
    ])
}
