use std::collections::HashSet;

use anyhow::Result;
use async_trait::async_trait;
use nuillu_module::{
    AllocationReader, AllocationUpdatedInbox, BlackboardReader, CognitionLogReader,
    CognitionLogUpdatedInbox, LlmAccess, MemoUpdatedInbox, Module, TypedMemo,
};
use nuillu_types::PolicyIndex;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::{PolicyWindowKey, PolicyWindowReader};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ValueEstimateMemo {
    pub retrieval_window: PolicyWindowKey,
    pub predictions: Vec<ValueEstimatePrediction>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ValueEstimatePrediction {
    pub policy_index: PolicyIndex,
    pub predicted_expected_reward: f32,
    pub confidence_hint: f32,
    pub rationale: String,
}

pub struct ValueEstimatorModule {
    memo_updates: MemoUpdatedInbox,
    cognition_updates: CognitionLogUpdatedInbox,
    allocation_updates: AllocationUpdatedInbox,
    _blackboard: BlackboardReader,
    _cognition: CognitionLogReader,
    _allocation: AllocationReader,
    windows: PolicyWindowReader,
    memo: TypedMemo<ValueEstimateMemo>,
    _llm: LlmAccess,
    processed: HashSet<PolicyWindowKey>,
}

impl ValueEstimatorModule {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        memo_updates: MemoUpdatedInbox,
        cognition_updates: CognitionLogUpdatedInbox,
        allocation_updates: AllocationUpdatedInbox,
        blackboard: BlackboardReader,
        cognition: CognitionLogReader,
        allocation: AllocationReader,
        windows: PolicyWindowReader,
        memo: TypedMemo<ValueEstimateMemo>,
        llm: LlmAccess,
    ) -> Self {
        Self {
            memo_updates,
            cognition_updates,
            allocation_updates,
            _blackboard: blackboard,
            _cognition: cognition,
            _allocation: allocation,
            windows,
            memo,
            _llm: llm,
            processed: HashSet::new(),
        }
    }

    async fn activate(&mut self) -> Result<()> {
        let windows = self.windows.retrieval_windows().await;
        for window in windows {
            if !self.processed.insert(window.key.clone()) {
                continue;
            }
            let predictions = window
                .payload
                .hits
                .iter()
                .map(|hit| ValueEstimatePrediction {
                    policy_index: hit.policy_index.clone(),
                    predicted_expected_reward: hit.expected_reward,
                    confidence_hint: hit.confidence,
                    rationale: "v1 baseline: use stored expected_reward for this context".into(),
                })
                .collect::<Vec<_>>();
            if predictions.is_empty() {
                continue;
            }
            let payload = ValueEstimateMemo {
                retrieval_window: window.key,
                predictions,
            };
            let plaintext = render_value_estimate(&payload);
            self.memo.write(payload, plaintext).await;
        }
        Ok(())
    }
}

fn render_value_estimate(memo: &ValueEstimateMemo) -> String {
    let mut output = format!(
        "Policy value-estimate window for {}#{}",
        memo.retrieval_window.owner, memo.retrieval_window.memo_index
    );
    for prediction in &memo.predictions {
        output.push_str("\nPrediction [");
        output.push_str(prediction.policy_index.as_str());
        output.push_str(&format!(
            "]: expected_reward {:.3}; confidence_hint {:.3}; rationale: {}",
            prediction.predicted_expected_reward,
            prediction.confidence_hint,
            prediction.rationale.trim()
        ));
    }
    output
}

#[async_trait(?Send)]
impl Module for ValueEstimatorModule {
    type Batch = ();

    fn id() -> &'static str {
        "value-estimator"
    }

    fn role_description() -> &'static str {
        "Scores query-policy retrieval windows with per-policy expected reward predictions."
    }

    async fn next_batch(&mut self) -> Result<Self::Batch> {
        tokio::select! {
            result = self.memo_updates.next_item() => {
                result?;
            }
            result = self.cognition_updates.next_item() => {
                result?;
            }
            result = self.allocation_updates.next_item() => {
                result?;
            }
        }
        Ok(())
    }

    async fn activate(
        &mut self,
        _cx: &nuillu_module::ActivateCx<'_>,
        _batch: &Self::Batch,
    ) -> Result<()> {
        ValueEstimatorModule::activate(self).await
    }
}
