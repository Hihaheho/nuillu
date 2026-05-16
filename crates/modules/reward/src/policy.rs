use anyhow::{Context, Result};
use async_trait::async_trait;
use lutum::{Session, StructuredTurnOutcome};
use nuillu_module::ports::Clock;
use nuillu_module::{
    AllocationReader, AllocationUpdatedInbox, BlackboardReader, CognitionLogReader,
    CognitionLogUpdatedInbox, InteroceptiveReader, LlmAccess, Memo, MemoUpdatedInbox, Module,
    format_current_attention_guidance,
};
use nuillu_types::{ModuleId, ModuleInstanceId, PolicyIndex, PolicyRank};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::{PolicyConsiderationLog, PolicySearchHit, PolicySearcher};

const SYSTEM_PROMPT: &str = r#"You are the policy module.
Read recent memo and cognition context, retrieve applicable existing policies, and propose
policy-grounded advice for the current situation. You may also synthesize reusable candidate
policies when existing records do not cover the situation.

For low or negative predicted value, give caution or avoidance advice rather than recommending the
behavior. Do not mutate policy state. Do not claim a policy was persisted."#;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct PolicyConsiderationPayload {
    pub request_context: String,
    pub query_text: String,
    pub considerations: Vec<PolicyConsideration>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct PolicyConsideration {
    pub candidate_id: String,
    pub source: PolicyConsiderationSource,
    pub predicted_expected_reward: f32,
    pub confidence_hint: f32,
    pub advice: String,
    pub rationale: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "kind", content = "data", rename_all = "snake_case")]
pub enum PolicyConsiderationSource {
    Existing(ExistingPolicyConsideration),
    Synthetic(SyntheticPolicyConsideration),
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ExistingPolicyConsideration {
    pub policy_index: PolicyIndex,
    pub similarity: f32,
    pub rank: PolicyRank,
    pub trigger: String,
    pub behavior: String,
    pub stored_expected_reward: f32,
    pub stored_confidence: f32,
    pub stored_value: f32,
    pub reward_tokens: u32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct SyntheticPolicyConsideration {
    pub trigger: String,
    pub behavior: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct PolicyConsiderationDecision {
    pub existing: Vec<ExistingPolicyDecision>,
    pub synthetic: Vec<SyntheticPolicyDecision>,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct ExistingPolicyDecision {
    pub policy_index: PolicyIndex,
    pub predicted_expected_reward: f32,
    pub confidence_hint: f32,
    pub advice: String,
    pub rationale: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct SyntheticPolicyDecision {
    pub trigger: String,
    pub behavior: String,
    pub predicted_expected_reward: f32,
    pub confidence_hint: f32,
    pub advice: String,
    pub rationale: String,
}

#[derive(Clone)]
pub struct PolicyConsiderationWriter {
    owner: ModuleInstanceId,
    log: PolicyConsiderationLog,
    clock: std::rc::Rc<dyn Clock>,
}

impl PolicyConsiderationWriter {
    pub(crate) fn new(
        owner: ModuleInstanceId,
        log: PolicyConsiderationLog,
        clock: std::rc::Rc<dyn Clock>,
    ) -> Self {
        Self { owner, log, clock }
    }

    pub fn write(&self, payload: PolicyConsiderationPayload) {
        self.log
            .append(self.owner.clone(), payload, self.clock.now());
    }
}

pub struct PolicyModule {
    owner: ModuleId,
    memo_updates: MemoUpdatedInbox,
    cognition_updates: CognitionLogUpdatedInbox,
    allocation_updates: AllocationUpdatedInbox,
    blackboard: BlackboardReader,
    cognition: CognitionLogReader,
    allocation: AllocationReader,
    interoception: InteroceptiveReader,
    searcher: PolicySearcher,
    memo: Memo,
    writer: PolicyConsiderationWriter,
    llm: LlmAccess,
    session: Session,
}

impl PolicyModule {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        memo_updates: MemoUpdatedInbox,
        cognition_updates: CognitionLogUpdatedInbox,
        allocation_updates: AllocationUpdatedInbox,
        blackboard: BlackboardReader,
        cognition: CognitionLogReader,
        allocation: AllocationReader,
        interoception: InteroceptiveReader,
        searcher: PolicySearcher,
        memo: Memo,
        writer: PolicyConsiderationWriter,
        llm: LlmAccess,
    ) -> Self {
        Self {
            owner: ModuleId::new(<Self as Module>::id()).expect("policy id is valid"),
            memo_updates,
            cognition_updates,
            allocation_updates,
            blackboard,
            cognition,
            allocation,
            interoception,
            searcher,
            memo,
            writer,
            llm,
            session: Session::new(),
        }
    }

    async fn activate(&mut self) -> Result<()> {
        let allocation = self.allocation.snapshot().await;
        let query_text = self.build_query(&allocation).await;
        let hits = self.searcher.search(&query_text, 8).await?;
        let Some(decision) = self.assess(&query_text, &hits).await? else {
            return Ok(());
        };
        let considerations = normalize_decision(&hits, decision);
        if considerations.is_empty() {
            return Ok(());
        }
        let payload = PolicyConsiderationPayload {
            request_context: query_text.clone(),
            query_text,
            considerations,
        };
        let plaintext = render_policy_consideration(&payload);
        self.memo.write(plaintext).await;
        self.writer.write(payload);
        Ok(())
    }

    async fn build_query(&self, allocation: &nuillu_blackboard::ResourceAllocation) -> String {
        let guidance = allocation
            .for_module(&self.owner)
            .guidance
            .trim()
            .to_owned();
        if !guidance.is_empty() {
            return guidance;
        }
        self.blackboard
            .read(|bb| {
                bb.cognition_log()
                    .entries()
                    .last()
                    .map(|entry| entry.text.clone())
                    .or_else(|| {
                        bb.recent_memo_logs()
                            .last()
                            .map(|record| record.content.clone())
                    })
                    .unwrap_or_else(|| "current cognition context".to_owned())
            })
            .await
    }

    async fn assess(
        &mut self,
        query_text: &str,
        hits: &[PolicySearchHit],
    ) -> Result<Option<PolicyConsiderationDecision>> {
        let memos = self.blackboard.recent_memo_logs().await;
        let cognition = self.cognition.snapshot().await;
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
            "Policy consideration request for {}:\nQuery text:\n{}\n\nExisting policy hits:\n{}\n\nRecent memos:\n{}\n\nCognition:\n{}",
            self.owner,
            query_text,
            serde_json::to_string(&render_hit_inputs(hits))
                .expect("policy hit input serialization should not fail"),
            serde_json::to_string(&memos).expect("memo log serialization should not fail"),
            serde_json::to_string(&cognition).expect("cognition log serialization should not fail"),
        ));

        let lutum = self.llm.lutum().await;
        let result = self
            .session
            .structured_turn::<PolicyConsiderationDecision>(&lutum)
            .collect()
            .await
            .context("policy structured turn failed")?;
        let StructuredTurnOutcome::Structured(decision) = result.semantic else {
            tracing::debug!("policy structured turn refused; skipping activation");
            return Ok(None);
        };
        Ok(Some(decision))
    }
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

fn render_hit_inputs(hits: &[PolicySearchHit]) -> Vec<PolicyHitInput<'_>> {
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

fn normalize_decision(
    hits: &[PolicySearchHit],
    decision: PolicyConsiderationDecision,
) -> Vec<PolicyConsideration> {
    let mut existing = decision
        .existing
        .into_iter()
        .map(|decision| (decision.policy_index.clone(), decision))
        .collect::<std::collections::HashMap<_, _>>();
    let mut out = Vec::new();

    for hit in hits {
        let decision = existing.remove(&hit.policy.index);
        let (predicted_expected_reward, confidence_hint, advice, rationale) =
            if let Some(decision) = decision {
                (
                    decision.predicted_expected_reward,
                    decision.confidence_hint,
                    decision.advice,
                    decision.rationale,
                )
            } else {
                (
                    hit.policy.expected_reward.get(),
                    hit.policy.confidence.get(),
                    default_existing_advice(hit.policy.expected_reward.get()),
                    "fallback: no policy prediction returned, used stored expected_reward".into(),
                )
            };
        out.push(PolicyConsideration {
            candidate_id: format!("existing:{}", hit.policy.index),
            source: PolicyConsiderationSource::Existing(ExistingPolicyConsideration {
                policy_index: hit.policy.index.clone(),
                similarity: hit.similarity,
                rank: hit.policy.rank,
                trigger: hit.policy.trigger.clone(),
                behavior: hit.policy.behavior.clone(),
                stored_expected_reward: hit.policy.expected_reward.get(),
                stored_confidence: hit.policy.confidence.get(),
                stored_value: hit.policy.value.get(),
                reward_tokens: hit.policy.reward_tokens,
            }),
            predicted_expected_reward: predicted_expected_reward.clamp(-1.0, 1.0),
            confidence_hint: confidence_hint.clamp(0.0, 1.0),
            advice,
            rationale,
        });
    }

    for (index, candidate) in decision.synthetic.into_iter().enumerate() {
        let trigger = candidate.trigger.trim();
        let behavior = candidate.behavior.trim();
        if trigger.is_empty() || behavior.is_empty() {
            continue;
        }
        out.push(PolicyConsideration {
            candidate_id: format!("synthetic:{index}"),
            source: PolicyConsiderationSource::Synthetic(SyntheticPolicyConsideration {
                trigger: trigger.to_owned(),
                behavior: behavior.to_owned(),
            }),
            predicted_expected_reward: candidate.predicted_expected_reward.clamp(-1.0, 1.0),
            confidence_hint: candidate.confidence_hint.clamp(0.0, 1.0),
            advice: candidate.advice,
            rationale: candidate.rationale,
        });
    }
    out
}

fn default_existing_advice(expected_reward: f32) -> String {
    if expected_reward < 0.0 {
        "Treat this policy as cautionary in the current context.".into()
    } else {
        "Consider this policy if the current context still matches its trigger.".into()
    }
}

fn render_policy_consideration(memo: &PolicyConsiderationPayload) -> String {
    let mut output = format!(
        "Policy consideration\nRequest context: {}\nQuery text: {}",
        memo.request_context.trim(),
        memo.query_text.trim()
    );
    for consideration in &memo.considerations {
        output.push_str("\nCandidate [");
        output.push_str(consideration.candidate_id.as_str());
        output.push_str("]\n");
        match &consideration.source {
            PolicyConsiderationSource::Existing(existing) => {
                output.push_str("Existing policy: ");
                output.push_str(existing.policy_index.as_str());
                output.push_str("\nTrigger: ");
                output.push_str(existing.trigger.trim());
                output.push_str("\nBehavior: ");
                output.push_str(existing.behavior.trim());
            }
            PolicyConsiderationSource::Synthetic(synthetic) => {
                output.push_str("Synthetic policy candidate\nTrigger: ");
                output.push_str(synthetic.trigger.trim());
                output.push_str("\nBehavior: ");
                output.push_str(synthetic.behavior.trim());
            }
        }
        output.push_str(&format!(
            "\nPredicted expected_reward: {:.3}; confidence_hint: {:.3}\nAdvice: {}\nRationale: {}",
            consideration.predicted_expected_reward,
            consideration.confidence_hint,
            consideration.advice.trim(),
            consideration.rationale.trim(),
        ));
    }
    output
}

#[async_trait(?Send)]
impl Module for PolicyModule {
    type Batch = ();

    fn id() -> &'static str {
        "policy"
    }

    fn role_description() -> &'static str {
        "Reads policy records, synthesizes policy candidates, predicts context value, and writes policy advice."
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
        PolicyModule::activate(self).await
    }
}

#[cfg(test)]
mod tests {
    use std::rc::Rc;

    use nuillu_blackboard::Blackboard;
    use nuillu_module::ports::SystemClock;
    use nuillu_types::{ModuleInstanceId, ReplicaIndex, builtin};
    use nuillu_types::{SignedUnitF32, UnitF32};

    use super::*;
    use crate::PolicyRecord;

    fn hit(index: &str, expected_reward: f32) -> PolicySearchHit {
        PolicySearchHit {
            policy: PolicyRecord {
                index: PolicyIndex::new(index),
                trigger: "trigger".into(),
                behavior: "behavior".into(),
                rank: PolicyRank::Tentative,
                expected_reward: SignedUnitF32::clamp(expected_reward),
                confidence: UnitF32::clamp(0.4),
                value: SignedUnitF32::ZERO,
                reward_tokens: 0,
                decay_remaining_secs: 60,
            },
            similarity: 0.9,
        }
    }

    #[test]
    fn normalize_decision_keeps_hits_and_synthetic_candidates() {
        let considerations = normalize_decision(
            &[hit("p1", -0.2)],
            PolicyConsiderationDecision {
                existing: Vec::new(),
                synthetic: vec![SyntheticPolicyDecision {
                    trigger: "new situation".into(),
                    behavior: "new behavior".into(),
                    predicted_expected_reward: 2.0,
                    confidence_hint: 2.0,
                    advice: "try carefully".into(),
                    rationale: "novel pattern".into(),
                }],
            },
        );

        assert_eq!(
            considerations
                .iter()
                .map(|candidate| (
                    candidate.candidate_id.as_str(),
                    candidate.predicted_expected_reward,
                    candidate.confidence_hint
                ))
                .collect::<Vec<_>>(),
            vec![("existing:p1", -0.2, 0.4), ("synthetic:0", 1.0, 1.0)]
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn consideration_writer_publishes_custom_policy_evictions() {
        let blackboard = Blackboard::default();
        let clock: Rc<dyn nuillu_module::ports::Clock> = Rc::new(SystemClock);
        let policy_caps = crate::PolicyCapabilities::new_with_consideration_retention(
            blackboard.clone(),
            clock.clone(),
            Rc::new(crate::NoopPolicyStore),
            Vec::new(),
            1,
        );
        let owner = ModuleInstanceId::new(builtin::policy(), ReplicaIndex::ZERO);
        let writer = policy_caps.consideration_writer(owner);
        let mut inbox = policy_caps.consideration_evicted_inbox();
        let first = sample_consideration("first");
        let second = sample_consideration("second");

        writer.write(first.clone());
        writer.write(second);

        let evicted = inbox
            .next_item()
            .await
            .expect("custom eviction is delivered");
        assert_eq!(evicted.key.owner.module, builtin::policy());
        assert_eq!(evicted.key.owner.replica, ReplicaIndex::ZERO);
        assert_eq!(evicted.key.memo_index, 0);
        assert_eq!(evicted.payload, first);
    }

    fn sample_consideration(label: &str) -> PolicyConsiderationPayload {
        PolicyConsiderationPayload {
            request_context: label.into(),
            query_text: label.into(),
            considerations: vec![PolicyConsideration {
                candidate_id: format!("synthetic:{label}"),
                source: PolicyConsiderationSource::Synthetic(SyntheticPolicyConsideration {
                    trigger: format!("{label} trigger"),
                    behavior: format!("{label} behavior"),
                }),
                predicted_expected_reward: -0.25,
                confidence_hint: 0.5,
                advice: "avoid this pattern".into(),
                rationale: "failure candidate remains useful".into(),
            }],
        }
    }
}
