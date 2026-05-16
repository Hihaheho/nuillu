use std::collections::HashSet;
use std::rc::Rc;

use anyhow::{Context, Result};
use async_trait::async_trait;
use eure::FromEure;
use lutum::{Session, StructuredTurnOutcome};
use nuillu_blackboard::{Blackboard, BlackboardCommand, PolicyMetaPatch};
use nuillu_module::ports::{Clock, PortError};
use nuillu_module::{
    AllocationReader, BlackboardReader, CognitionLogReader, InteroceptiveReader, LlmAccess, Memo,
    Module,
};
use nuillu_types::{PolicyIndex, PolicyRank, SignedUnitF32, UnitF32};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::{
    IndexedPolicy, NewPolicy, PolicyConsiderationEvicted, PolicyConsiderationEvictedInbox,
    PolicyConsiderationKey, PolicyConsiderationSource, PolicyRecord, PolicySearchHit,
    PolicySearcher, PolicyStore, SyntheticPolicyConsideration, fanout_policy_put,
};

const SYSTEM_PROMPT: &str = r#"You are the reward module.
Assess outcomes after a policy-consideration memo has left the retained memo surface. Return
structured reward channels and per-candidate credit values in [0, 1]. Credit only candidates that
plausibly affected later behavior or outcome. Use candidate_id values exactly as provided."#;
const DEFAULT_POLICY_REINFORCEMENT_CONFIG: &str =
    include_str!("../../../../configs/policy-reinforcement.eure");
const DEFAULT_SYNTHETIC_POLICY_DECAY_SECS: i64 = 86_400;
const SYNTHETIC_DEDUP_SEARCH_LIMIT: usize = 8;
const SYNTHETIC_DEDUP_PROMPT: &str = r#"You are the reward module's policy deduplication boundary.
Decide whether a credited synthetic policy candidate is already covered by one of the provided
existing policies. Choose reinforce_existing_policy only when an existing policy's trigger and
behavior already cover the candidate without contradiction. Choose insert_policy_candidate when the
candidate is novel, contradictory, materially more specific, or otherwise not clearly covered.
Use only the existing policy indexes in the provided candidate set."#;

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
    pub candidate_credits: Vec<PolicyCandidateCredit>,
    pub rationale: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct PolicyCandidateCredit {
    pub candidate_id: String,
    pub credit: f32,
    pub rationale: String,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RewardMemo {
    pub policy_consideration: PolicyConsiderationKey,
    pub observed_reward: ObservedReward,
    pub observed_scalar: f32,
    pub updates: Vec<PolicyRewardUpdate>,
    pub skipped_updates: Vec<PolicyRewardSkip>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PolicyRewardUpdate {
    pub candidate_id: String,
    pub policy_index: PolicyIndex,
    pub inserted: bool,
    pub credit: f32,
    pub compared_expected_reward: f32,
    pub td_error: f32,
    pub value_delta: f32,
    pub reward_tokens_delta: u32,
    pub expected_reward_delta: f32,
    pub confidence_delta: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PolicyRewardSkip {
    pub candidate_id: String,
    pub credit: f32,
    pub reason: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "action", content = "data", rename_all = "snake_case")]
pub enum SyntheticPolicyDedupDecision {
    ReinforceExistingPolicy(ReinforceExistingPolicyDecision),
    InsertPolicyCandidate(InsertPolicyCandidateDecision),
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct ReinforceExistingPolicyDecision {
    pub index: PolicyIndex,
    pub reason: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct InsertPolicyCandidateDecision {
    pub reason: String,
}

/// Inserts synthetic policy candidates and applies reward updates to existing
/// or newly inserted policy records.
#[derive(Clone)]
pub struct PolicyUpserter {
    primary_store: Rc<dyn PolicyStore>,
    replicas: Vec<Rc<dyn PolicyStore>>,
    blackboard: Blackboard,
    clock: Rc<dyn Clock>,
}

impl PolicyUpserter {
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

    pub async fn insert_candidate(
        &self,
        candidate: &SyntheticPolicyConsideration,
        initial_expected_reward: f32,
        initial_confidence: f32,
        initial_value: f32,
        decay_secs: i64,
    ) -> Result<PolicyRecord, PortError> {
        let trigger = candidate.trigger.trim().to_owned();
        let behavior = candidate.behavior.trim().to_owned();
        let expected_reward = SignedUnitF32::clamp(initial_expected_reward);
        let confidence = UnitF32::clamp(initial_confidence);
        let value = SignedUnitF32::clamp(initial_value);
        let new = NewPolicy {
            trigger: trigger.clone(),
            behavior: behavior.clone(),
            rank: PolicyRank::Tentative,
            expected_reward,
            confidence,
            value,
            reward_tokens: 0,
            decay_remaining_secs: decay_secs,
        };
        let index = self.primary_store.insert(new).await?;
        let record = PolicyRecord {
            index,
            trigger,
            behavior,
            rank: PolicyRank::Tentative,
            expected_reward,
            confidence,
            value,
            reward_tokens: 0,
            decay_remaining_secs: decay_secs,
        };
        fanout_policy_put(&self.replicas, indexed_policy(&record)).await;
        self.mirror_metadata(&record, None).await;
        Ok(record)
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
        self.mirror_metadata(&record, Some(self.clock.now())).await;
        Ok(record)
    }

    async fn mirror_metadata(
        &self,
        record: &PolicyRecord,
        reinforced_at: Option<chrono::DateTime<chrono::Utc>>,
    ) {
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
                    reinforced_at,
                    ..Default::default()
                },
            })
            .await;
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
    policy_evictions: PolicyConsiderationEvictedInbox,
    blackboard: BlackboardReader,
    cognition: CognitionLogReader,
    allocation: AllocationReader,
    interoception: InteroceptiveReader,
    searcher: PolicySearcher,
    upserter: PolicyUpserter,
    memo: Memo,
    llm: LlmAccess,
    session: Session,
    config: ReinforcementConfig,
}

impl RewardModule {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        policy_evictions: PolicyConsiderationEvictedInbox,
        blackboard: BlackboardReader,
        cognition: CognitionLogReader,
        allocation: AllocationReader,
        interoception: InteroceptiveReader,
        searcher: PolicySearcher,
        upserter: PolicyUpserter,
        memo: Memo,
        llm: LlmAccess,
    ) -> Self {
        Self {
            policy_evictions,
            blackboard,
            cognition,
            allocation,
            interoception,
            searcher,
            upserter,
            memo,
            llm,
            session: Session::new(),
            config: ReinforcementConfig::default(),
        }
    }

    async fn settle(&mut self, evicted: PolicyConsiderationEvicted) -> Result<()> {
        let assessment = self.assess(&evicted).await?;
        let observed_scalar = self.config.observed_scalar(&assessment.observed_reward);
        let mut updates = Vec::new();
        let mut skipped_updates = Vec::new();

        for candidate in &evicted.payload.considerations {
            let Some(credit) = assessment
                .candidate_credits
                .iter()
                .find(|credit| credit.candidate_id == candidate.candidate_id)
                .map(|credit| credit.credit.clamp(0.0, 1.0))
            else {
                continue;
            };
            if credit <= 0.0 {
                continue;
            }

            let td_error = observed_scalar - candidate.predicted_expected_reward;
            let expected_reward_delta = self.config.alpha * credit * td_error;
            let value_delta = self.config.beta * credit * td_error.clamp(-1.0, 1.0);
            let confidence_delta = confidence_delta(
                td_error.abs(),
                self.config.confidence_gamma,
                self.config.confidence_scale,
            ) * credit;
            let reward_tokens_delta = u32::from(td_error.abs() <= self.config.settle_band);

            let Some((policy_index, inserted)) = (match &candidate.source {
                PolicyConsiderationSource::Existing(existing) => {
                    Some((existing.policy_index.clone(), false))
                }
                PolicyConsiderationSource::Synthetic(synthetic) => {
                    match self.resolve_synthetic_policy(candidate, synthetic).await? {
                        SyntheticPolicyResolution::Existing(index) => Some((index, false)),
                        SyntheticPolicyResolution::Insert => {
                            let record = self
                                .upserter
                                .insert_candidate(
                                    synthetic,
                                    candidate.predicted_expected_reward,
                                    candidate.confidence_hint,
                                    candidate.predicted_expected_reward,
                                    DEFAULT_SYNTHETIC_POLICY_DECAY_SECS,
                                )
                                .await?;
                            Some((record.index, true))
                        }
                        SyntheticPolicyResolution::Skip(reason) => {
                            skipped_updates.push(PolicyRewardSkip {
                                candidate_id: candidate.candidate_id.clone(),
                                credit,
                                reason,
                            });
                            None
                        }
                    }
                }
            }) else {
                continue;
            };

            self.upserter
                .reinforce(
                    &policy_index,
                    value_delta,
                    reward_tokens_delta,
                    expected_reward_delta,
                    confidence_delta,
                )
                .await?;
            updates.push(PolicyRewardUpdate {
                candidate_id: candidate.candidate_id.clone(),
                policy_index,
                inserted,
                credit,
                compared_expected_reward: candidate.predicted_expected_reward,
                td_error,
                value_delta,
                reward_tokens_delta,
                expected_reward_delta,
                confidence_delta,
            });
        }

        let payload = RewardMemo {
            policy_consideration: evicted.key,
            observed_reward: assessment.observed_reward.clone(),
            observed_scalar,
            updates,
            skipped_updates,
        };
        let plaintext = render_reward_memo(&payload, &assessment);
        self.memo.write(plaintext).await;
        Ok(())
    }

    async fn resolve_synthetic_policy(
        &self,
        candidate: &crate::PolicyConsideration,
        synthetic: &SyntheticPolicyConsideration,
    ) -> Result<SyntheticPolicyResolution> {
        let hits = self
            .searcher
            .search(&synthetic.trigger, SYNTHETIC_DEDUP_SEARCH_LIMIT)
            .await
            .context("search existing policies for synthetic deduplication")?;
        if hits.is_empty() {
            return Ok(SyntheticPolicyResolution::Insert);
        }

        let decision = self
            .assess_synthetic_dedup(candidate, synthetic, &hits)
            .await?;
        Ok(resolve_synthetic_dedup_decision(&hits, decision))
    }

    async fn assess_synthetic_dedup(
        &self,
        candidate: &crate::PolicyConsideration,
        synthetic: &SyntheticPolicyConsideration,
        hits: &[PolicySearchHit],
    ) -> Result<Option<SyntheticPolicyDedupDecision>> {
        let mut session = Session::new();
        session.push_system(SYNTHETIC_DEDUP_PROMPT);
        session.push_user(format!(
            "Synthetic policy candidate:\n{}\n\nExisting policy candidates:\n{}",
            serde_json::to_string(&SyntheticDedupCandidateInput {
                candidate_id: &candidate.candidate_id,
                trigger: &synthetic.trigger,
                behavior: &synthetic.behavior,
                predicted_expected_reward: candidate.predicted_expected_reward,
                confidence_hint: candidate.confidence_hint,
                advice: &candidate.advice,
                rationale: &candidate.rationale,
            })
            .expect("synthetic dedup candidate serialization should not fail"),
            serde_json::to_string(&render_policy_hit_inputs(hits))
                .expect("policy hit serialization should not fail"),
        ));
        let lutum = self.llm.lutum().await;
        let result = session
            .structured_turn::<SyntheticPolicyDedupDecision>(&lutum)
            .collect()
            .await
            .context("policy synthetic dedup structured turn failed")?;
        match result.semantic {
            StructuredTurnOutcome::Structured(decision) => Ok(Some(decision)),
            _ => Ok(None),
        }
    }

    async fn assess(&mut self, evicted: &PolicyConsiderationEvicted) -> Result<RewardAssessment> {
        let memos = self.blackboard.recent_memo_logs().await;
        let cognition = self.cognition.snapshot().await;
        let allocation = self.allocation.snapshot().await;
        let interoception = self.interoception.snapshot().await;
        self.session.push_ephemeral_system(SYSTEM_PROMPT);
        self.session.push_ephemeral_user(format!(
            "Evicted policy consideration:\n{}\n\nRecent memos:\n{}\n\nCognition:\n{}\n\nAllocation:\n{}\n\nInteroception:\n{}",
            serde_json::to_string(&evicted.payload)
                .expect("policy consideration serialization should not fail"),
            serde_json::to_string(&memos).expect("memo log serialization should not fail"),
            serde_json::to_string(&cognition).expect("cognition log serialization should not fail"),
            serde_json::to_string(&allocation).expect("allocation serialization should not fail"),
            serde_json::to_string(&interoception)
                .expect("interoception serialization should not fail"),
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

#[derive(Debug, PartialEq)]
enum SyntheticPolicyResolution {
    Existing(PolicyIndex),
    Insert,
    Skip(String),
}

fn resolve_synthetic_dedup_decision(
    hits: &[PolicySearchHit],
    decision: Option<SyntheticPolicyDedupDecision>,
) -> SyntheticPolicyResolution {
    let candidate_indexes = hits
        .iter()
        .map(|hit| hit.policy.index.clone())
        .collect::<HashSet<_>>();
    match decision {
        Some(SyntheticPolicyDedupDecision::ReinforceExistingPolicy(decision)) => {
            if candidate_indexes.contains(&decision.index) {
                SyntheticPolicyResolution::Existing(decision.index)
            } else {
                SyntheticPolicyResolution::Skip(format!(
                    "dedup decision selected policy [{}] outside the provided candidate set",
                    decision.index
                ))
            }
        }
        Some(SyntheticPolicyDedupDecision::InsertPolicyCandidate(_)) => {
            SyntheticPolicyResolution::Insert
        }
        None => SyntheticPolicyResolution::Skip(
            "dedup decision was refused while similar policy candidates existed".to_owned(),
        ),
    }
}

#[derive(Serialize)]
struct SyntheticDedupCandidateInput<'a> {
    candidate_id: &'a str,
    trigger: &'a str,
    behavior: &'a str,
    predicted_expected_reward: f32,
    confidence_hint: f32,
    advice: &'a str,
    rationale: &'a str,
}

#[derive(Serialize)]
struct PolicyHitInput<'a> {
    policy_index: &'a PolicyIndex,
    similarity: f32,
    rank: PolicyRank,
    trigger: &'a str,
    behavior: &'a str,
    expected_reward: f32,
    confidence: f32,
    value: f32,
    reward_tokens: u32,
}

fn render_policy_hit_inputs(hits: &[PolicySearchHit]) -> Vec<PolicyHitInput<'_>> {
    hits.iter()
        .map(|hit| PolicyHitInput {
            policy_index: &hit.policy.index,
            similarity: hit.similarity,
            rank: hit.policy.rank,
            trigger: &hit.policy.trigger,
            behavior: &hit.policy.behavior,
            expected_reward: hit.policy.expected_reward.get(),
            confidence: hit.policy.confidence.get(),
            value: hit.policy.value.get(),
            reward_tokens: hit.policy.reward_tokens,
        })
        .collect()
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
        memo.policy_consideration.owner,
        memo.policy_consideration.memo_index,
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
            output.push_str("\nCandidate [");
            output.push_str(update.candidate_id.as_str());
            output.push_str("] -> policy [");
            output.push_str(update.policy_index.as_str());
            output.push_str(&format!(
                "]: inserted {}; credit {:.3}; expected_reward {:.3}; td_error {:.3}; value_delta {:.3}; expected_reward_delta {:.3}; confidence_delta {:.3}; reward_tokens_delta {}",
                update.inserted,
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
    if !memo.skipped_updates.is_empty() {
        output.push_str("\nSkipped policy updates:");
        for skipped in &memo.skipped_updates {
            output.push_str("\nCandidate [");
            output.push_str(skipped.candidate_id.as_str());
            output.push_str(&format!(
                "]: credit {:.3}; reason: {}",
                skipped.credit,
                skipped.reason.trim(),
            ));
        }
    }
    output
}

#[async_trait(?Send)]
impl Module for RewardModule {
    type Batch = Vec<PolicyConsiderationEvicted>;

    fn id() -> &'static str {
        "reward"
    }

    fn role_description() -> &'static str {
        "Assesses evicted policy considerations and consolidates rewarded or failed policy candidates."
    }

    async fn next_batch(&mut self) -> Result<Self::Batch> {
        let Some(first) = self.policy_evictions.next_item().await else {
            anyhow::bail!("policy-consideration eviction inbox closed");
        };
        let mut batch = vec![first];
        batch.extend(self.policy_evictions.take_ready_items());
        Ok(batch)
    }

    async fn activate(
        &mut self,
        _cx: &nuillu_module::ActivateCx<'_>,
        batch: &Self::Batch,
    ) -> Result<()> {
        let mut first_error = None;
        for evicted in batch {
            if let Err(error) = RewardModule::settle(self, evicted.clone()).await {
                tracing::warn!(?error, "failed to settle policy consideration");
                if first_error.is_none() {
                    first_error = Some(error);
                }
            }
        }
        if let Some(error) = first_error {
            Err(error)
        } else {
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::rc::Rc;

    use async_trait::async_trait;
    use nuillu_module::ports::SystemClock;

    use super::*;
    use crate::{PolicyCapabilities, PolicyCompactor, PolicySearchHit};

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

    #[tokio::test(flavor = "current_thread")]
    async fn upserter_inserts_negative_synthetic_policy_and_reinforces_it() {
        let store = Rc::new(RecordingPolicyStore::default());
        let blackboard = Blackboard::default();
        let upserter = PolicyUpserter::new(
            store.clone(),
            Vec::new(),
            blackboard.clone(),
            Rc::new(SystemClock),
        );
        let candidate = SyntheticPolicyConsideration {
            trigger: "rushed answer after weak evidence".into(),
            behavior: "avoid answering until evidence is promoted".into(),
        };

        let inserted = upserter
            .insert_candidate(&candidate, -0.6, 0.35, -0.6, 123)
            .await
            .unwrap();
        upserter
            .reinforce(&inserted.index, -0.1, 1, -0.2, 0.03)
            .await
            .unwrap();

        assert_eq!(
            store
                .inserts
                .borrow()
                .iter()
                .map(|policy| (
                    policy.trigger.as_str(),
                    policy.behavior.as_str(),
                    policy.rank,
                    policy.expected_reward,
                    policy.confidence,
                    policy.value,
                    policy.reward_tokens,
                    policy.decay_remaining_secs,
                ))
                .collect::<Vec<_>>(),
            vec![(
                "rushed answer after weak evidence",
                "avoid answering until evidence is promoted",
                PolicyRank::Tentative,
                SignedUnitF32::clamp(-0.6),
                UnitF32::clamp(0.35),
                SignedUnitF32::clamp(-0.6),
                0,
                123,
            )]
        );
        assert_eq!(
            store.reinforces.borrow().as_slice(),
            &[(PolicyIndex::new("recording-policy-0"), -0.1, 1, -0.2, 0.03)]
        );
        let reinforcement_count = blackboard
            .read(|bb| {
                bb.policy_metadata()
                    .get(&PolicyIndex::new("recording-policy-0"))
                    .map(|metadata| metadata.reinforcement_count)
            })
            .await;
        assert_eq!(reinforcement_count, Some(1));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn policy_searcher_search_does_not_mutate_metadata() {
        let store = Rc::new(RecordingPolicyStore::with_records(vec![record(
            "existing-policy",
            "current trigger",
            "existing behavior",
            PolicyRank::Tentative,
        )]));
        let blackboard = Blackboard::default();
        let policy_caps = PolicyCapabilities::new(
            blackboard.clone(),
            Rc::new(SystemClock),
            store.clone(),
            Vec::new(),
        );

        let hits = policy_caps
            .searcher()
            .search("current trigger", 4)
            .await
            .unwrap();

        assert_eq!(
            hits.iter()
                .map(|hit| hit.policy.index.as_str())
                .collect::<Vec<_>>(),
            vec!["existing-policy"]
        );
        assert!(blackboard.read(|bb| bb.policy_metadata().is_empty()).await);
    }

    #[test]
    fn dedup_decision_can_select_only_provided_candidates() {
        let hits = vec![PolicySearchHit {
            policy: record(
                "existing-policy",
                "weak evidence",
                "ask for clarification",
                PolicyRank::Tentative,
            ),
            similarity: 0.9,
        }];

        assert_eq!(
            resolve_synthetic_dedup_decision(
                &hits,
                Some(SyntheticPolicyDedupDecision::ReinforceExistingPolicy(
                    ReinforceExistingPolicyDecision {
                        index: PolicyIndex::new("existing-policy"),
                        reason: "covered".into(),
                    },
                )),
            ),
            SyntheticPolicyResolution::Existing(PolicyIndex::new("existing-policy"))
        );
        assert!(matches!(
            resolve_synthetic_dedup_decision(
                &hits,
                Some(SyntheticPolicyDedupDecision::ReinforceExistingPolicy(
                    ReinforceExistingPolicyDecision {
                        index: PolicyIndex::new("outside-policy"),
                        reason: "invalid".into(),
                    },
                )),
            ),
            SyntheticPolicyResolution::Skip(reason)
                if reason.contains("outside the provided candidate set")
        ));
        assert_eq!(
            resolve_synthetic_dedup_decision(
                &hits,
                Some(SyntheticPolicyDedupDecision::InsertPolicyCandidate(
                    InsertPolicyCandidateDecision {
                        reason: "novel".into(),
                    },
                )),
            ),
            SyntheticPolicyResolution::Insert
        );
        assert!(matches!(
            resolve_synthetic_dedup_decision(&hits, None),
            SyntheticPolicyResolution::Skip(reason)
                if reason.contains("refused")
        ));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn policy_compactor_deletes_only_non_core_duplicates_and_removes_metadata() {
        let duplicate = PolicyIndex::new("duplicate-policy");
        let core_duplicate = PolicyIndex::new("core-duplicate");
        let higher_rank_duplicate = PolicyIndex::new("higher-rank-duplicate");
        let store = Rc::new(RecordingPolicyStore::with_records(vec![
            record(
                "canonical-policy",
                "weak evidence",
                "ask for clarification",
                PolicyRank::Established,
            ),
            record(
                duplicate.as_str(),
                "weak evidence duplicate",
                "ask for clarification",
                PolicyRank::Tentative,
            ),
            record(
                core_duplicate.as_str(),
                "core weak evidence",
                "ask for clarification",
                PolicyRank::Core,
            ),
            record(
                higher_rank_duplicate.as_str(),
                "habit weak evidence",
                "ask for clarification",
                PolicyRank::Habit,
            ),
        ]));
        let blackboard = Blackboard::default();
        blackboard
            .apply(BlackboardCommand::UpsertPolicyMetadata {
                index: duplicate.clone(),
                rank_if_new: PolicyRank::Tentative,
                decay_if_new_secs: 60,
                patch: PolicyMetaPatch::default(),
            })
            .await;
        let compactor = PolicyCompactor::new(store.clone(), Vec::new(), blackboard.clone());

        let result = compactor
            .compact_duplicates(
                &PolicyIndex::new("canonical-policy"),
                &[
                    duplicate.clone(),
                    duplicate.clone(),
                    core_duplicate.clone(),
                    higher_rank_duplicate.clone(),
                ],
            )
            .await
            .unwrap();

        assert_eq!(result.deleted, vec![duplicate.clone()]);
        assert_eq!(
            result
                .skipped
                .iter()
                .map(|skipped| (skipped.index.as_str(), skipped.reason.as_str()))
                .collect::<Vec<_>>(),
            vec![
                (
                    "duplicate-policy",
                    "duplicate index was listed more than once"
                ),
                (
                    "core-duplicate",
                    "core policies cannot be deleted by policy-compaction"
                ),
                (
                    "higher-rank-duplicate",
                    "duplicate policy outranks the canonical policy"
                )
            ]
        );
        assert_eq!(store.deletes.borrow().as_slice(), &[duplicate]);
        assert!(
            blackboard
                .read(|bb| !bb
                    .policy_metadata()
                    .contains_key(&PolicyIndex::new("duplicate-policy")))
                .await
        );
    }

    #[derive(Default)]
    struct RecordingPolicyStore {
        inserts: RefCell<Vec<NewPolicy>>,
        records: RefCell<Vec<PolicyRecord>>,
        reinforces: RefCell<Vec<(PolicyIndex, f32, u32, f32, f32)>>,
        deletes: RefCell<Vec<PolicyIndex>>,
    }

    impl RecordingPolicyStore {
        fn with_records(records: Vec<PolicyRecord>) -> Self {
            Self {
                records: RefCell::new(records),
                ..Self::default()
            }
        }
    }

    #[async_trait(?Send)]
    impl PolicyStore for RecordingPolicyStore {
        async fn insert(&self, policy: NewPolicy) -> std::result::Result<PolicyIndex, PortError> {
            let id = self.inserts.borrow().len();
            let index = PolicyIndex::new(format!("recording-policy-{id}"));
            self.records.borrow_mut().push(PolicyRecord {
                index: index.clone(),
                trigger: policy.trigger.clone(),
                behavior: policy.behavior.clone(),
                rank: policy.rank,
                expected_reward: policy.expected_reward,
                confidence: policy.confidence,
                value: policy.value,
                reward_tokens: policy.reward_tokens,
                decay_remaining_secs: policy.decay_remaining_secs,
            });
            self.inserts.borrow_mut().push(policy);
            Ok(index)
        }

        async fn put(&self, _policy: IndexedPolicy) -> std::result::Result<(), PortError> {
            Ok(())
        }

        async fn get(
            &self,
            index: &PolicyIndex,
        ) -> std::result::Result<Option<PolicyRecord>, PortError> {
            Ok(self
                .records
                .borrow()
                .iter()
                .find(|record| &record.index == index)
                .cloned())
        }

        async fn list_by_rank(
            &self,
            rank: PolicyRank,
        ) -> std::result::Result<Vec<PolicyRecord>, PortError> {
            Ok(self
                .records
                .borrow()
                .iter()
                .filter(|record| record.rank == rank)
                .cloned()
                .collect())
        }

        async fn search(
            &self,
            q: &crate::PolicyQuery,
        ) -> std::result::Result<Vec<crate::PolicySearchHit>, PortError> {
            Ok(self
                .records
                .borrow()
                .iter()
                .take(q.limit)
                .cloned()
                .map(|policy| PolicySearchHit {
                    policy,
                    similarity: 0.9,
                })
                .collect())
        }

        async fn reinforce(
            &self,
            index: &PolicyIndex,
            value_delta: f32,
            reward_tokens_delta: u32,
            expected_reward_delta: f32,
            confidence_delta: f32,
        ) -> std::result::Result<PolicyRecord, PortError> {
            self.reinforces.borrow_mut().push((
                index.clone(),
                value_delta,
                reward_tokens_delta,
                expected_reward_delta,
                confidence_delta,
            ));
            let existing = self
                .records
                .borrow()
                .iter()
                .find(|record| &record.index == index)
                .cloned();
            let mut record = existing.unwrap_or_else(|| PolicyRecord {
                index: index.clone(),
                trigger: "rushed answer after weak evidence".into(),
                behavior: "avoid answering until evidence is promoted".into(),
                rank: PolicyRank::Tentative,
                expected_reward: SignedUnitF32::clamp(-0.6),
                confidence: UnitF32::clamp(0.35),
                value: SignedUnitF32::clamp(-0.6),
                reward_tokens: 0,
                decay_remaining_secs: 123,
            });
            record.expected_reward =
                SignedUnitF32::clamp(record.expected_reward.get() + expected_reward_delta);
            record.confidence = UnitF32::clamp(record.confidence.get() + confidence_delta);
            record.value = SignedUnitF32::clamp(record.value.get() + value_delta);
            record.reward_tokens = record.reward_tokens.saturating_add(reward_tokens_delta);
            Ok(record)
        }

        async fn delete(&self, index: &PolicyIndex) -> std::result::Result<(), PortError> {
            self.deletes.borrow_mut().push(index.clone());
            self.records
                .borrow_mut()
                .retain(|record| &record.index != index);
            Ok(())
        }
    }

    fn record(index: &str, trigger: &str, behavior: &str, rank: PolicyRank) -> PolicyRecord {
        PolicyRecord {
            index: PolicyIndex::new(index),
            trigger: trigger.into(),
            behavior: behavior.into(),
            rank,
            expected_reward: SignedUnitF32::ZERO,
            confidence: UnitF32::clamp(0.5),
            value: SignedUnitF32::ZERO,
            reward_tokens: 0,
            decay_remaining_secs: 60,
        }
    }
}
