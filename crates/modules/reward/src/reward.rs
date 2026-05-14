use std::collections::HashSet;
use std::rc::Rc;

use anyhow::{Context, Result};
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use eure::FromEure;
use lutum::{Session, StructuredTurnOutcome};
use nuillu_blackboard::{Blackboard, BlackboardCommand, PolicyMetaPatch};
use nuillu_module::ports::{Clock, PortError};
use nuillu_module::{
    AllocationReader, AllocationUpdatedInbox, AttentionControlRequest,
    AttentionControlRequestMailbox, BlackboardReader, CognitionLogReader, CognitionLogUpdatedInbox,
    LlmAccess, MemoUpdatedInbox, Module, TypedMemo,
};
use nuillu_types::{ModuleId, PolicyIndex, builtin};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::{
    IndexedPolicy, PolicyRecord, PolicyRetrievalMemo, PolicyStore, PolicyWindowKey,
    PolicyWindowReader, ValueEstimateMemo, ValueEstimatePrediction, fanout_policy_put,
};

const SYSTEM_PROMPT: &str = r#"You are the reward module.
Assess outcomes after a policy retrieval/value-estimate window. Return structured reward channels
and per-policy credit values in [0, 1]. Credit only policies that plausibly affected the later
behavior or outcome. Do not invent policy indexes."#;
const DEFAULT_POLICY_REINFORCEMENT_CONFIG: &str =
    include_str!("../../../../configs/policy-reinforcement.eure");

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ObservedReward {
    pub external: f32,
    pub task: f32,
    pub social: f32,
    pub cost: f32,
    pub risk: f32,
    pub novelty: f32,
}

impl ObservedReward {
    pub fn scalar_default(&self) -> f32 {
        (self.external + self.task + self.social + self.novelty - self.cost - self.risk)
            .clamp(-1.0, 1.0)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct RewardAssessment {
    pub observed_reward: ObservedReward,
    pub policy_credits: Vec<PolicyCredit>,
    pub novel_policy_request: Option<NovelPolicyRequest>,
    pub rationale: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct PolicyCredit {
    pub policy_index: PolicyIndex,
    pub credit: f32,
    pub rationale: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct NovelPolicyRequest {
    pub reason: String,
    pub candidate_trigger: Option<String>,
    pub candidate_behavior: Option<String>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RewardMemo {
    pub value_estimate_window: PolicyWindowKey,
    pub observed_reward: ObservedReward,
    pub observed_scalar: f32,
    pub updates: Vec<PolicyRewardUpdate>,
    pub novel_policy_requested: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PolicyRewardUpdate {
    pub policy_index: PolicyIndex,
    pub credit: f32,
    pub compared_expected_reward: f32,
    pub td_error: f32,
    pub value_delta: f32,
    pub reward_tokens_delta: u32,
    pub expected_reward_delta: f32,
    pub confidence_delta: f32,
}

/// Atomically applies TD-style reinforcement deltas to the primary policy
/// store and mirrors metadata onto the blackboard.
#[derive(Clone)]
pub struct PolicyValueUpdater {
    primary_store: Rc<dyn PolicyStore>,
    replicas: Vec<Rc<dyn PolicyStore>>,
    blackboard: Blackboard,
    clock: Rc<dyn Clock>,
}

impl PolicyValueUpdater {
    pub(crate) fn new(
        primary_store: Rc<dyn PolicyStore>,
        replicas: Vec<Rc<dyn PolicyStore>>,
        blackboard: Blackboard,
        clock: Rc<dyn Clock>,
    ) -> Self {
        Self {
            primary_store,
            replicas,
            blackboard,
            clock,
        }
    }

    pub async fn reinforce(
        &self,
        index: &PolicyIndex,
        value_delta: f32,
        reward_tokens_delta: u32,
        expected_reward_delta: f32,
        confidence_delta: f32,
    ) -> Result<PolicyRecord, PortError> {
        let record = self
            .primary_store
            .reinforce(
                index,
                value_delta,
                reward_tokens_delta,
                expected_reward_delta,
                confidence_delta,
            )
            .await?;
        fanout_policy_put(&self.replicas, indexed_policy(&record)).await;
        self.blackboard
            .apply(BlackboardCommand::UpsertPolicyMetadata {
                index: record.index.clone(),
                rank_if_new: record.rank,
                decay_if_new_secs: record.decay_remaining_secs,
                patch: PolicyMetaPatch {
                    rank: Some(record.rank),
                    expected_reward: Some(record.expected_reward),
                    confidence: Some(record.confidence),
                    value: Some(record.value),
                    reward_tokens: Some(record.reward_tokens),
                    decay_remaining_secs: Some(record.decay_remaining_secs),
                    reinforced_at: Some(self.clock.now()),
                    ..Default::default()
                },
            })
            .await;
        Ok(record)
    }
}

fn indexed_policy(record: &PolicyRecord) -> IndexedPolicy {
    IndexedPolicy {
        index: record.index.clone(),
        trigger: record.trigger.clone(),
        behavior: record.behavior.clone(),
        rank: record.rank,
        expected_reward: record.expected_reward,
        confidence: record.confidence,
        value: record.value,
        reward_tokens: record.reward_tokens,
        decay_remaining_secs: record.decay_remaining_secs,
    }
}

#[derive(Clone, Copy)]
struct ReinforcementConfig {
    alpha: f32,
    beta: f32,
    confidence_gamma: f32,
    confidence_scale: f32,
    settle_band: f32,
    external_weight: f32,
    task_weight: f32,
    social_weight: f32,
    novelty_weight: f32,
    cost_weight: f32,
    risk_weight: f32,
}

impl Default for ReinforcementConfig {
    fn default() -> Self {
        Self::from_eure_str(DEFAULT_POLICY_REINFORCEMENT_CONFIG)
            .expect("bundled configs/policy-reinforcement.eure must be valid")
    }
}

impl ReinforcementConfig {
    fn from_eure_str(content: &str) -> Result<Self> {
        let parsed: ReinforcementConfigFile =
            eure::parse_content(content, "configs/policy-reinforcement.eure".into())
                .map_err(|message| anyhow::anyhow!(message))?;
        parsed.try_into_runtime()
    }

    fn observed_scalar(self, observed: &ObservedReward) -> f32 {
        (self.external_weight * observed.external
            + self.task_weight * observed.task
            + self.social_weight * observed.social
            + self.novelty_weight * observed.novelty
            - self.cost_weight * observed.cost
            - self.risk_weight * observed.risk)
            .clamp(-1.0, 1.0)
    }
}

#[derive(Debug, Clone, FromEure)]
#[eure(crate = ::eure::document, rename_all = "kebab-case")]
struct ReinforcementConfigFile {
    alpha: f64,
    beta: f64,
    confidence_gamma: f64,
    confidence_scale: f64,
    settle_band: f64,
    external_weight: f64,
    task_weight: f64,
    social_weight: f64,
    novelty_weight: f64,
    cost_weight: f64,
    risk_weight: f64,
    provisional_token_min: u32,
    provisional_value_min: f64,
    established_token_min: u32,
    established_value_min: f64,
    established_confidence_min: f64,
    habit_token_min: u32,
    habit_value_min: f64,
    habit_confidence_min: f64,
    tentative_decay_secs: i64,
    provisional_decay_secs: i64,
    established_decay_secs: i64,
    habit_decay_secs: i64,
}

impl ReinforcementConfigFile {
    fn try_into_runtime(self) -> Result<ReinforcementConfig> {
        validate_finite("alpha", self.alpha)?;
        validate_finite("beta", self.beta)?;
        validate_finite("confidence-gamma", self.confidence_gamma)?;
        validate_finite("confidence-scale", self.confidence_scale)?;
        validate_finite("settle-band", self.settle_band)?;
        validate_finite("external-weight", self.external_weight)?;
        validate_finite("task-weight", self.task_weight)?;
        validate_finite("social-weight", self.social_weight)?;
        validate_finite("novelty-weight", self.novelty_weight)?;
        validate_finite("cost-weight", self.cost_weight)?;
        validate_finite("risk-weight", self.risk_weight)?;
        validate_finite("provisional-value-min", self.provisional_value_min)?;
        validate_finite("established-value-min", self.established_value_min)?;
        validate_finite(
            "established-confidence-min",
            self.established_confidence_min,
        )?;
        validate_finite("habit-value-min", self.habit_value_min)?;
        validate_finite("habit-confidence-min", self.habit_confidence_min)?;
        let _ = (
            self.provisional_token_min,
            self.established_token_min,
            self.habit_token_min,
            self.tentative_decay_secs,
            self.provisional_decay_secs,
            self.established_decay_secs,
            self.habit_decay_secs,
        );
        Ok(ReinforcementConfig {
            alpha: self.alpha as f32,
            beta: self.beta as f32,
            confidence_gamma: self.confidence_gamma as f32,
            confidence_scale: self.confidence_scale as f32,
            settle_band: self.settle_band as f32,
            external_weight: self.external_weight as f32,
            task_weight: self.task_weight as f32,
            social_weight: self.social_weight as f32,
            novelty_weight: self.novelty_weight as f32,
            cost_weight: self.cost_weight as f32,
            risk_weight: self.risk_weight as f32,
        })
    }
}

pub struct RewardModule {
    cognition_updates: CognitionLogUpdatedInbox,
    memo_updates: MemoUpdatedInbox,
    allocation_updates: AllocationUpdatedInbox,
    blackboard: BlackboardReader,
    cognition: CognitionLogReader,
    allocation: AllocationReader,
    windows: PolicyWindowReader,
    updater: PolicyValueUpdater,
    attention_control: AttentionControlRequestMailbox,
    memo: TypedMemo<RewardMemo>,
    llm: LlmAccess,
    session: Session,
    settled_value_estimates: HashSet<PolicyWindowKey>,
    settled_retrievals: HashSet<PolicyWindowKey>,
    config: ReinforcementConfig,
}

impl RewardModule {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        cognition_updates: CognitionLogUpdatedInbox,
        memo_updates: MemoUpdatedInbox,
        allocation_updates: AllocationUpdatedInbox,
        blackboard: BlackboardReader,
        cognition: CognitionLogReader,
        allocation: AllocationReader,
        windows: PolicyWindowReader,
        updater: PolicyValueUpdater,
        attention_control: AttentionControlRequestMailbox,
        memo: TypedMemo<RewardMemo>,
        llm: LlmAccess,
    ) -> Self {
        Self {
            cognition_updates,
            memo_updates,
            allocation_updates,
            blackboard,
            cognition,
            allocation,
            windows,
            updater,
            attention_control,
            memo,
            llm,
            session: Session::new(),
            settled_value_estimates: HashSet::new(),
            settled_retrievals: HashSet::new(),
            config: ReinforcementConfig::default(),
        }
    }

    async fn activate(&mut self, cx: &nuillu_module::ActivateCx<'_>) -> Result<()> {
        self.settle_value_estimates().await?;
        if !module_registered(cx, &builtin::value_estimator()) {
            self.settle_retrieval_baselines().await?;
        }
        Ok(())
    }

    async fn settle_value_estimates(&mut self) -> Result<()> {
        let estimates = self.windows.value_estimates().await;
        for estimate in estimates {
            if !self.settled_value_estimates.insert(estimate.key.clone()) {
                continue;
            }
            let Some(retrieval) = self
                .windows
                .retrieval_by_key(&estimate.payload.retrieval_window)
                .await
            else {
                continue;
            };
            self.settle_estimate(estimate.key, estimate.payload, retrieval)
                .await?;
        }
        Ok(())
    }

    async fn settle_retrieval_baselines(&mut self) -> Result<()> {
        let retrievals = self.windows.retrieval_windows().await;
        for retrieval in retrievals {
            if !self.settled_retrievals.insert(retrieval.key.clone()) {
                continue;
            }
            let predictions = retrieval
                .payload
                .hits
                .iter()
                .map(|hit| ValueEstimatePrediction {
                    policy_index: hit.policy_index.clone(),
                    predicted_expected_reward: hit.expected_reward,
                    confidence_hint: hit.confidence,
                    rationale: "v1 fallback: value-estimator absent, use stored expected_reward"
                        .into(),
                })
                .collect::<Vec<_>>();
            if predictions.is_empty() {
                continue;
            }
            let estimate = ValueEstimateMemo {
                retrieval_window: retrieval.key.clone(),
                predictions,
            };
            self.settle_estimate(retrieval.key, estimate, retrieval.payload)
                .await?;
        }
        Ok(())
    }

    async fn settle_estimate(
        &mut self,
        estimate_key: PolicyWindowKey,
        estimate: ValueEstimateMemo,
        retrieval: PolicyRetrievalMemo,
    ) -> Result<()> {
        let assessment = self.assess(&estimate, &retrieval).await?;
        let observed_scalar = self.config.observed_scalar(&assessment.observed_reward);
        let mut updates = Vec::new();

        for prediction in &estimate.predictions {
            let Some(credit) = assessment
                .policy_credits
                .iter()
                .find(|credit| credit.policy_index == prediction.policy_index)
                .map(|credit| credit.credit.clamp(0.0, 1.0))
            else {
                continue;
            };
            if credit <= 0.0 {
                continue;
            }
            if !retrieval
                .hits
                .iter()
                .any(|hit| hit.policy_index == prediction.policy_index)
            {
                continue;
            };
            let td_error = observed_scalar - prediction.predicted_expected_reward;
            let expected_reward_delta = self.config.alpha * credit * td_error;
            let value_delta = self.config.beta * credit * td_error.clamp(-1.0, 1.0);
            let confidence_delta = confidence_delta(
                td_error.abs(),
                self.config.confidence_gamma,
                self.config.confidence_scale,
            ) * credit;
            let reward_tokens_delta = u32::from(td_error.abs() <= self.config.settle_band);
            self.updater
                .reinforce(
                    &prediction.policy_index,
                    value_delta,
                    reward_tokens_delta,
                    expected_reward_delta,
                    confidence_delta,
                )
                .await?;
            updates.push(PolicyRewardUpdate {
                policy_index: prediction.policy_index.clone(),
                credit,
                compared_expected_reward: prediction.predicted_expected_reward,
                td_error,
                value_delta,
                reward_tokens_delta,
                expected_reward_delta,
                confidence_delta,
            });
        }

        let novel_policy_requested = assessment.novel_policy_request.is_some();
        if let Some(request) = &assessment.novel_policy_request {
            let _ = self
                .attention_control
                .publish(AttentionControlRequest::policy(
                    request.reason.clone(),
                    request.candidate_trigger.clone(),
                    request.candidate_behavior.clone(),
                ))
                .await;
        }

        let payload = RewardMemo {
            value_estimate_window: estimate_key,
            observed_reward: assessment.observed_reward.clone(),
            observed_scalar,
            updates,
            novel_policy_requested,
        };
        let plaintext = render_reward_memo(&payload, &assessment);
        self.memo.write(payload, plaintext).await;
        Ok(())
    }

    async fn assess(
        &mut self,
        estimate: &ValueEstimateMemo,
        retrieval: &PolicyRetrievalMemo,
    ) -> Result<RewardAssessment> {
        let memos = self.blackboard.recent_memo_logs().await;
        let cognition = self.cognition.snapshot().await;
        let allocation = self.allocation.snapshot().await;
        self.session.push_ephemeral_system(SYSTEM_PROMPT);
        self.session.push_ephemeral_user(format!(
            "Retrieval window:\n{}\n\nValue estimate:\n{}\n\nRecent memos:\n{}\n\nCognition:\n{}\n\nAllocation:\n{}",
            serde_json::to_string(retrieval).unwrap_or_default(),
            serde_json::to_string(estimate).unwrap_or_default(),
            serde_json::to_string(&memos).unwrap_or_default(),
            serde_json::to_string(&cognition).unwrap_or_default(),
            serde_json::to_string(&allocation).unwrap_or_default(),
        ));
        let lutum = self.llm.lutum().await;
        let result = self
            .session
            .structured_turn::<RewardAssessment>(&lutum)
            .collect()
            .await
            .context("reward structured turn failed")?;
        let StructuredTurnOutcome::Structured(assessment) = result.semantic else {
            anyhow::bail!("reward structured turn refused");
        };
        Ok(assessment)
    }
}

fn confidence_delta(abs_td_error: f32, gamma: f32, scale: f32) -> f32 {
    if scale <= 0.0 {
        return 0.0;
    }
    gamma * (1.0 - abs_td_error / scale).max(0.0)
}

fn validate_finite(name: &str, value: f64) -> Result<()> {
    if value.is_finite() {
        Ok(())
    } else {
        anyhow::bail!("{name} must be finite")
    }
}

fn render_reward_memo(memo: &RewardMemo, assessment: &RewardAssessment) -> String {
    let mut output = format!(
        "Reward assessment for {}#{}\nObserved scalar: {:.3}\nObserved channels: external {:.3}, task {:.3}, social {:.3}, cost {:.3}, risk {:.3}, novelty {:.3}\nRationale: {}",
        memo.value_estimate_window.owner,
        memo.value_estimate_window.memo_index,
        memo.observed_scalar,
        assessment.observed_reward.external,
        assessment.observed_reward.task,
        assessment.observed_reward.social,
        assessment.observed_reward.cost,
        assessment.observed_reward.risk,
        assessment.observed_reward.novelty,
        assessment.rationale.trim(),
    );
    if memo.updates.is_empty() {
        output.push_str("\nPolicy updates: none");
    } else {
        output.push_str("\nPolicy updates:");
        for update in &memo.updates {
            output.push_str("\nPolicy [");
            output.push_str(update.policy_index.as_str());
            output.push_str(&format!(
                "]: credit {:.3}; expected_reward {:.3}; td_error {:.3}; value_delta {:.3}; expected_reward_delta {:.3}; confidence_delta {:.3}; reward_tokens_delta {}",
                update.credit,
                update.compared_expected_reward,
                update.td_error,
                update.value_delta,
                update.expected_reward_delta,
                update.confidence_delta,
                update.reward_tokens_delta,
            ));
        }
    }
    if let Some(request) = &assessment.novel_policy_request {
        output.push_str("\nNovel policy request: ");
        output.push_str(request.reason.trim());
    } else {
        output.push_str("\nNovel policy request: none");
    }
    output
}

#[async_trait(?Send)]
impl Module for RewardModule {
    type Batch = ();

    fn id() -> &'static str {
        "reward"
    }

    fn role_description() -> &'static str {
        "Aggregates observed reward, assigns per-policy credit, and updates policy value state."
    }

    async fn next_batch(&mut self) -> Result<Self::Batch> {
        tokio::select! {
            result = self.cognition_updates.next_item() => {
                result?;
            }
            result = self.memo_updates.next_item() => {
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
        cx: &nuillu_module::ActivateCx<'_>,
        _batch: &Self::Batch,
    ) -> Result<()> {
        RewardModule::activate(self, cx).await
    }
}

fn module_registered(cx: &nuillu_module::ActivateCx<'_>, id: &ModuleId) -> bool {
    cx.modules().iter().any(|(module, _)| module == id)
}

#[allow(dead_code)]
fn _suppress_unused(_: DateTime<Utc>) {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bundled_reinforcement_config_loads() {
        let config = ReinforcementConfig::default();

        assert_eq!(config.alpha, 0.2);
        assert_eq!(config.beta, 0.1);
        assert_eq!(config.settle_band, 0.2);
        assert_eq!(
            config.observed_scalar(&ObservedReward {
                external: 0.2,
                task: 0.3,
                social: 0.1,
                cost: 0.2,
                risk: 0.1,
                novelty: 0.4,
            }),
            0.7
        );
    }

    #[test]
    fn confidence_delta_rewards_accurate_predictions_only() {
        assert_eq!(confidence_delta(0.0, 0.05, 0.5), 0.05);
        assert_eq!(confidence_delta(0.5, 0.05, 0.5), 0.0);
        assert_eq!(confidence_delta(1.0, 0.05, 0.5), 0.0);
    }
}
