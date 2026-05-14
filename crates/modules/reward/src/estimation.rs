use std::collections::{HashMap, HashSet};

use anyhow::{Context, Result};
use async_trait::async_trait;
use lutum::{Session, StructuredTurnOutcome};
use nuillu_module::{
    AllocationReader, AllocationUpdatedInbox, BlackboardReader, CognitionLogReader,
    CognitionLogUpdatedInbox, InteroceptiveReader, LlmAccess, MemoUpdatedInbox, Module, TypedMemo,
    format_current_attention_guidance,
};
use nuillu_types::{ModuleId, PolicyIndex};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::{PolicyWindowKey, PolicyWindowReader, TypedPolicyRetrievalWindow};

const SYSTEM_PROMPT: &str = r#"You are the value-estimator module.
For each policy surfaced by query-policy, predict its expected reward in the current cognitive
context. Return a prediction for every provided policy index. The prediction is a scalar in
[-1, 1]. Do not mutate policy state, do not create policies, and do not invent policy indexes."#;

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct ValueEstimationDecision {
    pub predictions: Vec<ValueEstimationPrediction>,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct ValueEstimationPrediction {
    pub policy_index: PolicyIndex,
    pub predicted_expected_reward: f32,
    pub rationale: String,
}

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
    owner: ModuleId,
    memo_updates: MemoUpdatedInbox,
    cognition_updates: CognitionLogUpdatedInbox,
    allocation_updates: AllocationUpdatedInbox,
    blackboard: BlackboardReader,
    cognition: CognitionLogReader,
    allocation: AllocationReader,
    interoception: InteroceptiveReader,
    windows: PolicyWindowReader,
    memo: TypedMemo<ValueEstimateMemo>,
    llm: LlmAccess,
    session: Session,
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
        interoception: InteroceptiveReader,
        windows: PolicyWindowReader,
        memo: TypedMemo<ValueEstimateMemo>,
        llm: LlmAccess,
    ) -> Self {
        Self {
            owner: ModuleId::new(<Self as Module>::id()).expect("value-estimator id is valid"),
            memo_updates,
            cognition_updates,
            allocation_updates,
            blackboard,
            cognition,
            allocation,
            interoception,
            windows,
            memo,
            llm,
            session: Session::new(),
            processed: HashSet::new(),
        }
    }

    async fn activate(&mut self) -> Result<()> {
        let windows = self.windows.retrieval_windows().await;
        for window in windows {
            if !self.processed.insert(window.key.clone()) {
                continue;
            }
            let predictions = self.estimate_window(&window).await?;
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

    async fn estimate_window(
        &mut self,
        window: &TypedPolicyRetrievalWindow,
    ) -> Result<Vec<ValueEstimatePrediction>> {
        let cognition = self.cognition.snapshot().await;
        let memos = self.blackboard.recent_memo_logs().await;
        let allocation = self.allocation.snapshot().await;
        let interoception = self.interoception.snapshot().await;

        self.session.push_ephemeral_system(SYSTEM_PROMPT);
        if let Some(guidance) = format_current_attention_guidance(&allocation) {
            self.session.push_ephemeral_system(guidance);
        }
        self.session.push_ephemeral_system(format!(
            "Current interoception: affect_arousal={:.2}; valence={:.2}; emotion={}",
            interoception.affect_arousal,
            interoception.valence,
            if interoception.emotion.trim().is_empty() {
                "unknown"
            } else {
                interoception.emotion.trim()
            }
        ));
        self.session.push_ephemeral_user(format!(
            "Value-estimation request for {}:\nRetrieval window:\n{}\n\nRecent cognition:\n{}\n\nRecent memos:\n{}",
            self.owner,
            serde_json::to_string(&window.payload).unwrap_or_default(),
            serde_json::to_string(&cognition).unwrap_or_default(),
            serde_json::to_string(&memos).unwrap_or_default(),
        ));

        let lutum = self.llm.lutum().await;
        let result = self
            .session
            .structured_turn::<ValueEstimationDecision>(&lutum)
            .collect()
            .await
            .context("value-estimator structured turn failed")?;
        let StructuredTurnOutcome::Structured(decision) = result.semantic else {
            anyhow::bail!("value-estimator structured turn refused");
        };

        Ok(normalize_predictions(window, decision))
    }
}

fn normalize_predictions(
    window: &TypedPolicyRetrievalWindow,
    decision: ValueEstimationDecision,
) -> Vec<ValueEstimatePrediction> {
    let mut by_index = decision
        .predictions
        .into_iter()
        .map(|prediction| (prediction.policy_index.clone(), prediction))
        .collect::<HashMap<_, _>>();

    window
        .payload
        .hits
        .iter()
        .map(|hit| {
            let Some(prediction) = by_index.remove(&hit.policy_index) else {
                return ValueEstimatePrediction {
                    policy_index: hit.policy_index.clone(),
                    predicted_expected_reward: hit.expected_reward,
                    confidence_hint: hit.confidence,
                    rationale:
                        "fallback: no value-estimator prediction returned, used stored expected_reward"
                            .into(),
                };
            };
            ValueEstimatePrediction {
                policy_index: hit.policy_index.clone(),
                predicted_expected_reward: prediction.predicted_expected_reward.clamp(-1.0, 1.0),
                confidence_hint: hit.confidence,
                rationale: prediction.rationale,
            }
        })
        .collect()
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

#[cfg(test)]
mod tests {
    use chrono::Utc;
    use nuillu_types::{PolicyRank, ReplicaIndex, builtin};

    use super::*;
    use crate::{PolicyRetrievalHit, PolicyRetrievalMemo};

    #[test]
    fn normalize_predictions_filters_unknowns_and_fills_missing_hits() {
        let owner =
            nuillu_types::ModuleInstanceId::new(builtin::query_policy(), ReplicaIndex::ZERO);
        let window = TypedPolicyRetrievalWindow {
            key: PolicyWindowKey {
                owner: owner.clone(),
                memo_index: 0,
            },
            written_at: Utc::now(),
            plaintext: "policy plaintext".into(),
            payload: PolicyRetrievalMemo {
                request_context: "context".into(),
                query_text: "alpha".into(),
                hits: vec![
                    PolicyRetrievalHit {
                        policy_index: PolicyIndex::new("known"),
                        similarity: 0.9,
                        rank: PolicyRank::Tentative,
                        trigger: "alpha".into(),
                        behavior: "do alpha".into(),
                        expected_reward: 0.25,
                        confidence: 0.4,
                        value: 0.1,
                        reward_tokens: 0,
                    },
                    PolicyRetrievalHit {
                        policy_index: PolicyIndex::new("missing"),
                        similarity: 0.8,
                        rank: PolicyRank::Tentative,
                        trigger: "beta".into(),
                        behavior: "do beta".into(),
                        expected_reward: -0.2,
                        confidence: 0.3,
                        value: -0.1,
                        reward_tokens: 0,
                    },
                ],
            },
        };
        let decision = ValueEstimationDecision {
            predictions: vec![
                ValueEstimationPrediction {
                    policy_index: PolicyIndex::new("known"),
                    predicted_expected_reward: 4.0,
                    rationale: "strong context".into(),
                },
                ValueEstimationPrediction {
                    policy_index: PolicyIndex::new("unknown"),
                    predicted_expected_reward: -1.0,
                    rationale: "ignore".into(),
                },
            ],
        };

        let predictions = normalize_predictions(&window, decision);

        assert_eq!(
            predictions
                .iter()
                .map(|prediction| (
                    prediction.policy_index.as_str(),
                    prediction.predicted_expected_reward,
                    prediction.confidence_hint
                ))
                .collect::<Vec<_>>(),
            vec![("known", 1.0, 0.4), ("missing", -0.2, 0.3)]
        );
    }
}
