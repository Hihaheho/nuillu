use anyhow::{Context, Result};
use async_trait::async_trait;
use lutum::{Session, StructuredTurnOutcome};
use nuillu_blackboard::{CognitionLogEntryRecord, MemoLogRecord};
use nuillu_module::ports::Clock;
use nuillu_module::{
    AllocationReader, BlackboardReader, CognitionLogReader, CognitionLogUpdatedInbox,
    InteroceptiveReader, LlmAccess, LlmContextWindow, Memo, MemoUpdatedInbox, Module,
    SessionAutoCompaction, SessionCompactionConfig, SessionCompactionProtectedPrefix,
    compact_llm_context_text, ensure_persistent_session_seeded, format_bounded_cognition_log_batch,
    format_bounded_memo_log_batch, format_current_attention_guidance,
    format_identity_system_prompt,
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

const POLICY_SEARCH_LIMIT: usize = 5;
const POLICY_QUERY_CONTEXT_CHARS: usize = 1_000;
const POLICY_HIT_TEXT_CHARS: usize = 240;
const POLICY_MEMO_CONTEXT_WINDOW: LlmContextWindow = LlmContextWindow::new(6, 1_200, 4_200);
const POLICY_COGNITION_CONTEXT_WINDOW: LlmContextWindow = LlmContextWindow::new(8, 600, 3_000);
const COMPACTED_POLICY_SESSION_PREFIX: &str = "Compacted policy session history:";
const SESSION_COMPACTION_FOCUS: &str = r#"Preserve reusable policy advice, existing policy
judgments, synthetic policy candidates, cautions, and reward-relevant rationale future policy
decisions need."#;

pub fn policy_session_auto_compaction() -> SessionAutoCompaction {
    SessionAutoCompaction::new(
        SessionCompactionConfig::default(),
        SessionCompactionProtectedPrefix::LeadingSystemAndIdentitySeed,
        COMPACTED_POLICY_SESSION_PREFIX,
        SESSION_COMPACTION_FOCUS,
    )
}

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
    blackboard: BlackboardReader,
    cognition: CognitionLogReader,
    allocation: AllocationReader,
    interoception: InteroceptiveReader,
    searcher: PolicySearcher,
    memo: Memo,
    writer: PolicyConsiderationWriter,
    llm: LlmAccess,
    session: Session,
    system_prompt: std::sync::OnceLock<String>,
}

struct PolicyActivationContext {
    query_text: String,
    memos: Vec<MemoLogRecord>,
    cognition: Vec<CognitionLogEntryRecord>,
}

impl PolicyModule {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        memo_updates: MemoUpdatedInbox,
        cognition_updates: CognitionLogUpdatedInbox,
        blackboard: BlackboardReader,
        cognition: CognitionLogReader,
        allocation: AllocationReader,
        interoception: InteroceptiveReader,
        searcher: PolicySearcher,
        memo: Memo,
        writer: PolicyConsiderationWriter,
        llm: LlmAccess,
        session: Session,
    ) -> Self {
        Self {
            owner: ModuleId::new(<Self as Module>::id()).expect("policy id is valid"),
            memo_updates,
            cognition_updates,
            blackboard,
            cognition,
            allocation,
            interoception,
            searcher,
            memo,
            writer,
            llm,
            session,
            system_prompt: std::sync::OnceLock::new(),
        }
    }

    fn system_prompt(&self, cx: &nuillu_module::ActivateCx<'_>) -> &str {
        self.system_prompt.get_or_init(|| {
            format_identity_system_prompt(
                SYSTEM_PROMPT,
                cx.identity_memories(),
                cx.core_policies(),
                cx.now(),
            )
        })
    }

    fn ensure_session_seeded(&mut self, cx: &nuillu_module::ActivateCx<'_>) {
        let system_prompt = self.system_prompt(cx).to_owned();
        ensure_persistent_session_seeded(
            &mut self.session,
            system_prompt,
            cx.identity_memories(),
            cx.now(),
        );
    }

    async fn activate(&mut self, cx: &nuillu_module::ActivateCx<'_>) -> Result<()> {
        self.ensure_session_seeded(cx);
        let allocation = self.allocation.snapshot().await;
        let context = self.activation_context(&allocation).await;
        let hits = self
            .searcher
            .search(&context.query_text, POLICY_SEARCH_LIMIT)
            .await?;
        let Some(decision) = self.assess(cx, &context, &hits).await? else {
            return Ok(());
        };
        let considerations = normalize_decision(&hits, decision);
        if considerations.is_empty() {
            return Ok(());
        }
        let payload = PolicyConsiderationPayload {
            request_context: context.query_text.clone(),
            query_text: context.query_text.clone(),
            considerations,
        };
        let plaintext = render_policy_consideration(&payload);
        self.memo.write(plaintext).await;
        self.writer.write(payload);
        Ok(())
    }

    async fn activation_context(
        &self,
        allocation: &nuillu_blackboard::ResourceAllocation,
    ) -> PolicyActivationContext {
        let memos = self.blackboard.unread_memo_logs().await;
        let cognition = self.cognition.unread_events().await;
        let query_text = self.build_query(allocation, &memos, &cognition).await;
        PolicyActivationContext {
            query_text: compact_llm_context_text(&query_text, POLICY_QUERY_CONTEXT_CHARS),
            memos,
            cognition,
        }
    }

    async fn build_query(
        &self,
        allocation: &nuillu_blackboard::ResourceAllocation,
        memos: &[MemoLogRecord],
        cognition: &[CognitionLogEntryRecord],
    ) -> String {
        let guidance = allocation
            .for_module(&self.owner)
            .guidance
            .trim()
            .to_owned();
        if !guidance.is_empty() {
            return guidance;
        }
        if let Some(entry) = cognition.last() {
            return entry.entry.text.clone();
        }
        if let Some(record) = memos.last() {
            return record.content.clone();
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
        cx: &nuillu_module::ActivateCx<'_>,
        context: &PolicyActivationContext,
        hits: &[PolicySearchHit],
    ) -> Result<Option<PolicyConsiderationDecision>> {
        let allocation = self.allocation.snapshot().await;
        let interoception = self.interoception.snapshot().await;

        let lutum = self.llm.lutum().await;
        let result = {
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
                "Policy consideration request for {}:\nQuery text:\n{}\n\nExisting policy hits:\n{}\n\nCurrent memo evidence:\n{}\n\nCurrent cognition evidence:\n{}",
                self.owner,
                context.query_text,
                serde_json::to_string(&render_hit_inputs(hits))
                    .expect("policy hit input serialization should not fail"),
                format_bounded_memo_log_batch(
                    &context.memos,
                    cx.now(),
                    POLICY_MEMO_CONTEXT_WINDOW,
                )
                .unwrap_or_else(|| "none".to_owned()),
                format_bounded_cognition_log_batch(
                    &context.cognition,
                    cx.now(),
                    POLICY_COGNITION_CONTEXT_WINDOW,
                )
                .unwrap_or_else(|| "none".to_owned()),
            ));
            let result = self
                .session
                .structured_turn::<PolicyConsiderationDecision>()
                .max_output_tokens(1024)
                .collect(&lutum)
                .await
                .context("policy structured turn failed")?;
            let semantic = result.semantic;
            cx.compact_and_save(&mut self.session, result.usage).await?;
            semantic
        };
        let StructuredTurnOutcome::Structured(decision) = result else {
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
    trigger: String,
    behavior: String,
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
            trigger: compact_llm_context_text(&hit.policy.trigger, POLICY_HIT_TEXT_CHARS),
            behavior: compact_llm_context_text(&hit.policy.behavior, POLICY_HIT_TEXT_CHARS),
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

    fn peer_context() -> Option<&'static str> {
        None
    }

    fn allocation_hint() -> Option<&'static str> {
        Some(
            "Raise policy when current cognition needs behavioral guidance, a reusable response pattern, or value-based tradeoff judgment. Keep it low for plain recall, sensory filtering, memory maintenance, or already-settled action.",
        )
    }

    async fn next_batch(&mut self) -> Result<Self::Batch> {
        tokio::select! {
            result = self.memo_updates.next_item() => {
                result?;
            }
            result = self.cognition_updates.next_item() => {
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
        PolicyModule::activate(self, cx).await
    }
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::rc::Rc;
    use std::sync::{Arc, Mutex};

    use lutum::{
        AdapterStructuredTurn, AdapterTextTurn, AgentError, AssistantInputItem,
        ErasedStructuredTurnEventStream, ErasedTextTurnEventStream, FinishReason, InputMessageRole,
        Lutum, MessageContent, MockLlmAdapter, MockStructuredScenario, ModelInput, ModelInputItem,
        RawStructuredTurnEvent, SharedPoolBudgetManager, SharedPoolBudgetOptions, TurnAdapter,
        Usage,
    };
    use nuillu_blackboard::Blackboard;
    use nuillu_module::ports::SystemClock;
    use nuillu_module::{
        CapabilityProviderPorts, CapabilityProviders, LlmConcurrencyLimiter, LutumTiers,
        ModuleRegistry, SessionCompactionPolicy, SessionCompactionRuntime,
    };
    use nuillu_types::{ModelTier, ReplicaCapRange};
    use nuillu_types::{ModuleInstanceId, ReplicaIndex, builtin};
    use nuillu_types::{SignedUnitF32, UnitF32};

    use super::*;
    use crate::{
        IndexedPolicy, NewPolicy, PolicyCapabilities, PolicyQuery, PolicyRecord, PolicyStore,
    };

    #[derive(Clone)]
    struct CapturingStructuredAdapter {
        inner: MockLlmAdapter,
        structured_inputs: Arc<Mutex<Vec<ModelInput>>>,
    }

    impl CapturingStructuredAdapter {
        fn new(inner: MockLlmAdapter) -> Self {
            Self {
                inner,
                structured_inputs: Arc::new(Mutex::new(Vec::new())),
            }
        }

        fn structured_inputs(&self) -> Vec<ModelInput> {
            self.structured_inputs.lock().unwrap().clone()
        }
    }

    #[async_trait::async_trait]
    impl TurnAdapter for CapturingStructuredAdapter {
        async fn text_turn(
            &self,
            input: ModelInput,
            turn: AdapterTextTurn,
        ) -> Result<ErasedTextTurnEventStream, AgentError> {
            self.inner.text_turn(input, turn).await
        }

        async fn structured_turn(
            &self,
            input: ModelInput,
            turn: AdapterStructuredTurn,
        ) -> Result<ErasedStructuredTurnEventStream, AgentError> {
            self.structured_inputs.lock().unwrap().push(input.clone());
            self.inner.structured_turn(input, turn).await
        }
    }

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

    fn text_usage(input_tokens: u64) -> Usage {
        Usage {
            input_tokens,
            ..Usage::zero()
        }
    }

    fn policy_decision_scenario(input_tokens: u64) -> MockStructuredScenario {
        let json = serde_json::json!({
            "existing": [],
            "synthetic": [
                {
                    "trigger": "fresh bounded trigger",
                    "behavior": "fresh bounded behavior",
                    "predicted_expected_reward": 0.25,
                    "confidence_hint": 0.5,
                    "advice": "continue cautiously",
                    "rationale": "bounded current evidence is enough"
                }
            ]
        })
        .to_string();
        MockStructuredScenario::events(vec![
            Ok(RawStructuredTurnEvent::Started {
                request_id: Some("policy-assess".into()),
                model: "mock".into(),
            }),
            Ok(RawStructuredTurnEvent::StructuredOutputChunk { json_delta: json }),
            Ok(RawStructuredTurnEvent::Completed {
                request_id: Some("policy-assess".into()),
                finish_reason: FinishReason::Stop,
                usage: text_usage(input_tokens),
            }),
        ])
    }

    fn test_caps_with_adapter<T>(
        blackboard: Blackboard,
        adapter: Arc<T>,
    ) -> (CapabilityProviders, Lutum)
    where
        T: TurnAdapter + 'static,
    {
        let budget = SharedPoolBudgetManager::new(SharedPoolBudgetOptions::default());
        let lutum = Lutum::new(adapter, budget);
        let caps = CapabilityProviders::new(CapabilityProviderPorts {
            blackboard,
            cognition_log_port: Rc::new(nuillu_module::ports::NoopCognitionLogRepository),
            clock: Rc::new(SystemClock),
            tiers: LutumTiers::from_shared_lutum(lutum.clone()),
        });
        (caps, lutum)
    }

    fn compaction_runtime(lutum: &Lutum) -> SessionCompactionRuntime {
        SessionCompactionRuntime::new(
            lutum.clone(),
            LlmConcurrencyLimiter::new(None),
            ModelTier::Cheap,
            SessionCompactionPolicy::default(),
        )
    }

    fn module_policy() -> nuillu_blackboard::ModulePolicy {
        nuillu_blackboard::ModulePolicy::new(
            ReplicaCapRange::new(1, 1).unwrap(),
            nuillu_blackboard::Bpm::from_f64(60_000.0)..=nuillu_blackboard::Bpm::from_f64(60_000.0),
            nuillu_blackboard::linear_ratio_fn,
        )
    }

    async fn build_policy_module(
        caps: &CapabilityProviders,
        policy_caps: PolicyCapabilities,
    ) -> nuillu_module::AllocatedModule {
        let modules = ModuleRegistry::new()
            .register(module_policy(), move |caps| {
                let policy_caps = policy_caps.clone();
                async move {
                    Ok(PolicyModule::new(
                        caps.memo_updated_inbox(),
                        caps.cognition_log_updated_inbox(),
                        caps.blackboard_reader(),
                        caps.cognition_log_reader(),
                        caps.allocation_reader(),
                        caps.interoception_reader(),
                        policy_caps.searcher(),
                        caps.memo(),
                        policy_caps.consideration_writer(caps.owner().clone()),
                        caps.llm_access(),
                        caps.session("main")
                            .with_auto_compaction(policy_session_auto_compaction())
                            .await?,
                    ))
                }
            })
            .unwrap()
            .build(caps)
            .await
            .unwrap();
        let (_, mut modules) = modules.into_parts();
        modules.remove(0)
    }

    fn activate_cx(
        lutum: &Lutum,
        now: chrono::DateTime<chrono::Utc>,
    ) -> nuillu_module::ActivateCx<'static> {
        nuillu_module::ActivateCx::new(&[], &[], &[], &[], compaction_runtime(lutum), now)
    }

    fn all_input_text(input: &ModelInput) -> String {
        let mut out = String::new();
        for item in input.items() {
            match item {
                ModelInputItem::Message { content, .. } => {
                    for content in content.as_slice() {
                        if let MessageContent::Text(text) = content {
                            out.push_str(text);
                            out.push('\n');
                        }
                    }
                }
                ModelInputItem::Assistant(AssistantInputItem::Text(text)) => {
                    out.push_str(text);
                    out.push('\n');
                }
                _ => {}
            }
        }
        out
    }

    fn system_prompt_count(input: &ModelInput) -> usize {
        input
            .items()
            .iter()
            .filter(|item| {
                matches!(
                    item,
                    ModelInputItem::Message {
                        role: InputMessageRole::System,
                        content,
                    } if matches!(content.as_slice(), [MessageContent::Text(text)] if text.contains(SYSTEM_PROMPT))
                )
            })
            .count()
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
    async fn activation_bounds_policy_input_and_seeds_stable_prompt_once() {
        let adapter = MockLlmAdapter::new()
            .with_structured_scenario(policy_decision_scenario(1))
            .with_structured_scenario(policy_decision_scenario(1));
        let capture = CapturingStructuredAdapter::new(adapter);
        let observed = capture.clone();
        let blackboard = Blackboard::default();
        let (caps, lutum) = test_caps_with_adapter(blackboard.clone(), Arc::new(capture));
        let store = Rc::new(RecordingPolicyStore::with_records(
            (0..7)
                .map(|index| {
                    record(
                        &format!("policy-{index}"),
                        &format!("trigger {index} {}", "T".repeat(600)),
                        &format!("behavior {index} {}", "B".repeat(600)),
                    )
                })
                .collect(),
        ));
        let policy_caps = PolicyCapabilities::new(
            blackboard.clone(),
            Rc::new(SystemClock),
            store.clone(),
            Vec::new(),
        );
        let mut module = build_policy_module(&caps, policy_caps).await;
        let mut allocation = nuillu_blackboard::ResourceAllocation::default();
        allocation.set(
            builtin::policy(),
            nuillu_blackboard::ModuleConfig {
                guidance: "answer the current bounded policy query".into(),
            },
        );
        allocation.set_activation(builtin::policy(), nuillu_blackboard::ActivationRatio::ONE);
        blackboard
            .apply(nuillu_blackboard::BlackboardCommand::SetAllocation(
                allocation,
            ))
            .await;

        let now = chrono::Utc::now();
        let long_memo_tail = "M".repeat(1_200);
        let memo_owner = ModuleInstanceId::new(builtin::self_model(), ReplicaIndex::ZERO);
        let memo_mailbox = caps.internal_harness_io().memo_updated_mailbox();
        for index in 0..8 {
            let record = blackboard
                .update_memo(
                    memo_owner.clone(),
                    format!("memo {index} {long_memo_tail}"),
                    now + chrono::Duration::seconds(index),
                )
                .await;
            memo_mailbox
                .publish(nuillu_module::MemoUpdated {
                    owner: record.owner,
                    index: record.index,
                })
                .await
                .unwrap();
        }
        let cognition_owner = ModuleInstanceId::new(builtin::cognition_gate(), ReplicaIndex::ZERO);
        for index in 0..10 {
            blackboard
                .append_cognition_log(
                    cognition_owner.clone(),
                    nuillu_blackboard::CognitionLogEntry {
                        at: now + chrono::Duration::seconds(index),
                        text: format!("cognition {index} {}", "C".repeat(1_200)),
                    },
                )
                .await;
        }

        let cx = activate_cx(&lutum, now);
        let batch = module.next_batch().await.unwrap();
        module.activate(&cx, &batch).await.unwrap();

        let first = observed.structured_inputs();
        assert_eq!(first.len(), 1);
        let first_text = all_input_text(&first[0]);
        assert!(first_text.contains("answer the current bounded policy query"));
        assert!(first_text.contains("policy-0"));
        assert!(first_text.contains("policy-4"));
        assert!(!first_text.contains("policy-5"));
        assert!(!first_text.contains("not shown here"));
        assert!(!first_text.contains("omitted"));
        assert!(!first_text.contains("[truncated]"));
        assert!(!first_text.contains(&"M".repeat(800)));
        assert!(!first_text.contains(&"C".repeat(800)));
        assert_eq!(store.search_limits.borrow().as_slice(), &[5]);
        assert_eq!(system_prompt_count(&first[0]), 1);

        let record = blackboard
            .update_memo(
                memo_owner.clone(),
                "fresh second memo".to_owned(),
                now + chrono::Duration::seconds(20),
            )
            .await;
        memo_mailbox
            .publish(nuillu_module::MemoUpdated {
                owner: record.owner,
                index: record.index,
            })
            .await
            .unwrap();
        let batch = module.next_batch().await.unwrap();
        module.activate(&cx, &batch).await.unwrap();

        let inputs = observed.structured_inputs();
        assert_eq!(inputs.len(), 2);
        assert_eq!(system_prompt_count(&inputs[1]), 1);
        assert_eq!(store.search_limits.borrow().as_slice(), &[5, 5]);
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

    #[derive(Default)]
    struct RecordingPolicyStore {
        records: RefCell<Vec<PolicyRecord>>,
        search_limits: RefCell<Vec<usize>>,
    }

    impl RecordingPolicyStore {
        fn with_records(records: Vec<PolicyRecord>) -> Self {
            Self {
                records: RefCell::new(records),
                search_limits: RefCell::new(Vec::new()),
            }
        }
    }

    #[async_trait::async_trait(?Send)]
    impl PolicyStore for RecordingPolicyStore {
        async fn insert(
            &self,
            _policy: NewPolicy,
        ) -> std::result::Result<PolicyIndex, nuillu_module::ports::PortError> {
            Ok(PolicyIndex::new("recording-policy"))
        }

        async fn put(
            &self,
            _policy: IndexedPolicy,
        ) -> std::result::Result<(), nuillu_module::ports::PortError> {
            Ok(())
        }

        async fn get(
            &self,
            index: &PolicyIndex,
        ) -> std::result::Result<Option<PolicyRecord>, nuillu_module::ports::PortError> {
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
        ) -> std::result::Result<Vec<PolicyRecord>, nuillu_module::ports::PortError> {
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
            q: &PolicyQuery,
        ) -> std::result::Result<Vec<PolicySearchHit>, nuillu_module::ports::PortError> {
            self.search_limits.borrow_mut().push(q.limit);
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
            _value_delta: f32,
            _reward_tokens_delta: u32,
            _expected_reward_delta: f32,
            _confidence_delta: f32,
        ) -> std::result::Result<PolicyRecord, nuillu_module::ports::PortError> {
            Ok(self
                .records
                .borrow()
                .iter()
                .find(|record| &record.index == index)
                .cloned()
                .unwrap_or_else(|| record(index.as_str(), "trigger", "behavior")))
        }

        async fn delete(
            &self,
            _index: &PolicyIndex,
        ) -> std::result::Result<(), nuillu_module::ports::PortError> {
            Ok(())
        }
    }

    fn record(index: &str, trigger: &str, behavior: &str) -> PolicyRecord {
        PolicyRecord {
            index: PolicyIndex::new(index),
            trigger: trigger.into(),
            behavior: behavior.into(),
            rank: PolicyRank::Tentative,
            expected_reward: SignedUnitF32::ZERO,
            confidence: UnitF32::clamp(0.5),
            value: SignedUnitF32::ZERO,
            reward_tokens: 0,
            decay_remaining_secs: 60,
        }
    }
}
