use anyhow::{Context, Result};
use async_trait::async_trait;
use lutum::{
    ModelInput, Session, StructuredTurnOutcome, TextStepOutcomeWithTools, ToolResult, Usage,
};
use nuillu_blackboard::{CognitionLogEntryRecord, MemoLogRecord};
use nuillu_module::ports::Clock;
use nuillu_module::{
    AllocationReader, BlackboardReader, CognitionLogReader, CognitionLogUpdatedInbox,
    InteroceptiveReader, LlmAccess, LlmContextWindow, Memo, MemoUpdatedInbox, Module,
    SessionAutoCompaction, SessionCompactionConfig, SessionCompactionProtectedPrefix,
    compact_llm_context_text, ensure_persistent_session_seeded, format_bounded_cognition_log_batch,
    format_bounded_memo_log_batch, format_policy_system_prompt, format_system_seed,
};
use nuillu_types::{ModuleId, ModuleInstanceId, PolicyIndex, PolicyRank};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::{PolicyConsiderationLog, PolicySearchHit, PolicySearcher};

const SYSTEM_PROMPT: &str = r#"You are the policy module.
Read recent memo and cognition context, retrieve applicable existing policies, and propose
policy-grounded advice for the current situation. You may also synthesize reusable candidate
policies when existing records do not cover the situation.

Policy candidates must be policies: trigger-conditioned behavioral guidance. The trigger describes
the situation where the rule applies. The behavior describes what action should or should not be
taken. Advice must be derived from that behavior and must not invent factual, capability, identity,
or state claims.

For low or negative predicted value, give caution or avoidance advice rather than recommending the
behavior. Do not mutate policy state. Do not claim a policy was persisted."#;

const POLICY_CANDIDATE_EVALUATION_PROMPT: &str = r#"Evaluate a synthetic policy candidate before it is published.
Accept only if the candidate is a policy: trigger-conditioned behavioral guidance.
Reject candidates that introduce factual, capability, identity, or state claims as policy content,
unless those claims are directly supported by the supplied evidence.
Valid policy language includes should, must, and must not when it describes behavior.
Return accepted=false with a concise reason when behavior or advice is not actually policy."#;

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

#[lutum::tool_input(name = "propose_policy_candidate", output = ProposePolicyCandidateOutput)]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct ProposePolicyCandidateArgs {
    pub trigger: String,
    pub behavior: String,
    pub advice: String,
    pub expected_value: PolicyExpectedValue,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum PolicyExpectedValue {
    StrongNegative,
    Negative,
    Neutral,
    Positive,
    StrongPositive,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct ProposePolicyCandidateOutput {
    pub accepted: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
struct PolicyCandidateEvaluation {
    accepted: bool,
    reason: String,
}

#[derive(Serialize)]
struct PolicyCandidateEvaluationInput<'a> {
    query_text: &'a str,
    current_memo_evidence: String,
    current_cognition_evidence: String,
    candidate: PolicyCandidateEvaluationCandidate<'a>,
}

#[derive(Serialize)]
struct PolicyCandidateEvaluationCandidate<'a> {
    trigger: &'a str,
    behavior: &'a str,
    advice: &'a str,
    expected_value: PolicyExpectedValue,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema, lutum::Toolset)]
#[allow(clippy::large_enum_variant)]
pub enum PolicyTools {
    ProposePolicyCandidate(ProposePolicyCandidateArgs),
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
        self.system_prompt
            .get_or_init(|| format_policy_system_prompt(SYSTEM_PROMPT, cx.core_policies()))
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
        let considerations = self.considerations_for_context(cx, &context, &hits).await?;
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

    async fn considerations_for_context(
        &mut self,
        cx: &nuillu_module::ActivateCx<'_>,
        context: &PolicyActivationContext,
        hits: &[PolicySearchHit],
    ) -> Result<Vec<PolicyConsideration>> {
        let mut considerations = deterministic_existing_considerations(hits);
        considerations.extend(self.propose_synthetic_candidates(cx, context, hits).await?);
        Ok(considerations)
    }

    async fn propose_synthetic_candidates(
        &mut self,
        cx: &nuillu_module::ActivateCx<'_>,
        context: &PolicyActivationContext,
        hits: &[PolicySearchHit],
    ) -> Result<Vec<PolicyConsideration>> {
        let interoception = self.interoception.snapshot().await;

        self.ensure_session_seeded(cx);
        let lutum = self.llm.lutum().await;
        let session_len_before_turn = self.session.input().items().len();
        self.session.push_user(format_policy_candidate_request(
            &self.owner,
            context,
            hits,
            &interoception,
            cx.now(),
        ));

        let outcome = match self
            .session
            .text_turn()
            .tools::<PolicyTools>()
            .available_tools([PolicyToolsSelector::ProposePolicyCandidate])
            .max_output_tokens(512)
            .collect_controlled_with(&lutum, nuillu_module::AbortOnAvailableToolNameInText::new())
            .await
        {
            Ok(outcome) => outcome,
            Err(error) => {
                tracing::warn!(error = %error, "policy candidate tool turn failed; continuing with deterministic considerations");
                truncate_session_to(&mut self.session, session_len_before_turn);
                cx.compact_and_save(&mut self.session, Usage::zero())
                    .await?;
                return Ok(Vec::new());
            }
        };

        match outcome {
            TextStepOutcomeWithTools::Finished(result) => {
                cx.compact_and_save(&mut self.session, result.usage).await?;
                Ok(Vec::new())
            }
            TextStepOutcomeWithTools::FinishedNoOutput(result) => {
                truncate_session_to(&mut self.session, session_len_before_turn);
                cx.compact_and_save(&mut self.session, result.usage).await?;
                Ok(Vec::new())
            }
            TextStepOutcomeWithTools::NeedsTools(round) => {
                let usage = round.usage;
                nuillu_module::emit_trace_tool_calls(&round.tool_calls);
                let [PolicyToolsCall::ProposePolicyCandidate(call)] = round.tool_calls.as_slice()
                else {
                    round.discard();
                    truncate_session_to(&mut self.session, session_len_before_turn);
                    cx.compact_and_save(&mut self.session, usage).await?;
                    return Ok(Vec::new());
                };
                let Some(candidate) = normalize_proposed_candidate(call.input.clone()) else {
                    round.discard();
                    truncate_session_to(&mut self.session, session_len_before_turn);
                    cx.compact_and_save(&mut self.session, usage).await?;
                    return Ok(Vec::new());
                };
                if !self
                    .evaluate_synthetic_candidate(cx, context, &call.input)
                    .await
                {
                    round.discard();
                    truncate_session_to(&mut self.session, session_len_before_turn);
                    cx.compact_and_save(&mut self.session, usage).await?;
                    return Ok(Vec::new());
                }
                let results: Vec<ToolResult> = vec![
                    call.clone()
                        .complete(ProposePolicyCandidateOutput { accepted: true })
                        .context("complete propose_policy_candidate tool call")?,
                ];
                if let Err(error) = round.commit(&mut self.session, results) {
                    tracing::warn!(
                        error = %error,
                        "policy candidate tool round commit failed; not persisting turn"
                    );
                    truncate_session_to(&mut self.session, session_len_before_turn);
                    cx.compact_and_save(&mut self.session, usage).await?;
                    return Ok(Vec::new());
                }
                cx.compact_and_save(&mut self.session, usage).await?;
                Ok(vec![candidate])
            }
        }
    }

    async fn evaluate_synthetic_candidate(
        &self,
        cx: &nuillu_module::ActivateCx<'_>,
        context: &PolicyActivationContext,
        candidate: &ProposePolicyCandidateArgs,
    ) -> bool {
        let input = ModelInput::new()
            .system(format_system_seed(
                format_policy_system_prompt(POLICY_CANDIDATE_EVALUATION_PROMPT, cx.core_policies()),
                false,
                cx.identity_memories(),
                cx.now(),
            ))
            .user(format_policy_candidate_evaluation_request(
                context,
                candidate,
                cx.now(),
            ));
        let lutum = self.llm.lutum().await;
        let result = lutum
            .structured_turn::<PolicyCandidateEvaluation>(input)
            .max_output_tokens(256)
            .collect()
            .await;
        match result {
            Ok(result) => match result.semantic {
                StructuredTurnOutcome::Structured(evaluation) => {
                    if !evaluation.accepted {
                        tracing::warn!(
                            reason = %evaluation.reason,
                            "synthetic policy candidate rejected by contract evaluation"
                        );
                    }
                    evaluation.accepted
                }
                other => {
                    tracing::warn!(
                        outcome = ?other,
                        "synthetic policy candidate evaluation produced no structured acceptance"
                    );
                    false
                }
            },
            Err(error) => {
                tracing::warn!(
                    error = %error,
                    "synthetic policy candidate evaluation failed"
                );
                false
            }
        }
    }
}

fn format_policy_candidate_request(
    owner: &ModuleId,
    context: &PolicyActivationContext,
    hits: &[PolicySearchHit],
    interoception: &nuillu_blackboard::InteroceptiveState,
    now: chrono::DateTime<chrono::Utc>,
) -> String {
    format!(
        "Policy consideration request for {owner}\n\nQuery text:\n{}\n\nCurrent interoception:\n{}\n\nExisting policy hits already handled by runtime:\n{}\n\nCurrent memo evidence:\n{}\n\nCurrent cognition evidence:\n{}\n\nInstruction:\nOnly call propose_policy_candidate if the existing hits are insufficient and a reusable new trigger/behavior policy is needed.\nA policy candidate must be trigger-conditioned behavioral guidance. Put only the situation in trigger, only the action rule in behavior, and only behavior-derived guidance in advice. Do not introduce factual, capability, identity, or state claims as policy content.",
        context.query_text,
        format_policy_interoception_state(interoception),
        format_policy_hits(hits),
        format_bounded_memo_log_batch(&context.memos, now, POLICY_MEMO_CONTEXT_WINDOW)
            .unwrap_or_else(|| "none".to_owned()),
        format_bounded_cognition_log_batch(
            &context.cognition,
            now,
            POLICY_COGNITION_CONTEXT_WINDOW,
        )
        .unwrap_or_else(|| "none".to_owned()),
    )
}

fn format_policy_candidate_evaluation_request(
    context: &PolicyActivationContext,
    candidate: &ProposePolicyCandidateArgs,
    now: chrono::DateTime<chrono::Utc>,
) -> String {
    serde_json::to_string(&PolicyCandidateEvaluationInput {
        query_text: &context.query_text,
        current_memo_evidence: format_bounded_memo_log_batch(
            &context.memos,
            now,
            POLICY_MEMO_CONTEXT_WINDOW,
        )
        .unwrap_or_else(|| "none".to_owned()),
        current_cognition_evidence: format_bounded_cognition_log_batch(
            &context.cognition,
            now,
            POLICY_COGNITION_CONTEXT_WINDOW,
        )
        .unwrap_or_else(|| "none".to_owned()),
        candidate: PolicyCandidateEvaluationCandidate {
            trigger: candidate.trigger.trim(),
            behavior: candidate.behavior.trim(),
            advice: candidate.advice.trim(),
            expected_value: candidate.expected_value,
        },
    })
    .expect("policy candidate evaluation input serialization should not fail")
}

fn format_policy_interoception_state(state: &nuillu_blackboard::InteroceptiveState) -> String {
    format!(
        "- mode: {:?}\n- wake_arousal: {:.2}\n- nrem_pressure: {:.2}\n- rem_pressure: {:.2}\n- affect_arousal: {:.2}\n- valence: {:.2}\n- emotion: {}\n- last_updated: {}",
        state.mode,
        state.wake_arousal,
        state.nrem_pressure,
        state.rem_pressure,
        state.affect_arousal,
        state.valence,
        if state.emotion.trim().is_empty() {
            "(none)"
        } else {
            state.emotion.trim()
        },
        state.last_updated.to_rfc3339(),
    )
}

fn format_policy_hits(hits: &[PolicySearchHit]) -> String {
    if hits.is_empty() {
        return "none".to_owned();
    }
    hits.iter()
        .map(|hit| {
            format!(
                "- policy_index: {}\n  similarity: {:.3}\n  rank: {:?}\n  trigger: {}\n  behavior: {}\n  stored_expected_reward: {:.3}\n  stored_confidence: {:.3}\n  stored_value: {:.3}\n  reward_tokens: {}",
                hit.policy.index,
                hit.similarity,
                hit.policy.rank,
                compact_llm_context_text(&hit.policy.trigger, POLICY_HIT_TEXT_CHARS),
                compact_llm_context_text(&hit.policy.behavior, POLICY_HIT_TEXT_CHARS),
                hit.policy.expected_reward.get(),
                hit.policy.confidence.get(),
                hit.policy.value.get(),
                hit.policy.reward_tokens,
            )
        })
        .collect::<Vec<_>>()
        .join("\n\n")
}

fn truncate_session_to(session: &mut Session, len: usize) {
    session.input_mut().items_mut().truncate(len);
}

fn deterministic_existing_considerations(hits: &[PolicySearchHit]) -> Vec<PolicyConsideration> {
    hits.iter()
        .map(|hit| {
            let expected_reward = hit.policy.expected_reward.get();
            PolicyConsideration {
                candidate_id: format!("existing:{}", hit.policy.index),
                source: PolicyConsiderationSource::Existing(ExistingPolicyConsideration {
                    policy_index: hit.policy.index.clone(),
                    similarity: hit.similarity,
                    rank: hit.policy.rank,
                    trigger: hit.policy.trigger.clone(),
                    behavior: hit.policy.behavior.clone(),
                    stored_expected_reward: expected_reward,
                    stored_confidence: hit.policy.confidence.get(),
                    stored_value: hit.policy.value.get(),
                    reward_tokens: hit.policy.reward_tokens,
                }),
                predicted_expected_reward: expected_reward,
                confidence_hint: hit.policy.confidence.get(),
                advice: default_existing_advice(expected_reward),
                rationale:
                    "existing policy hit applied deterministically from stored policy statistics"
                        .into(),
            }
        })
        .collect()
}

fn normalize_proposed_candidate(args: ProposePolicyCandidateArgs) -> Option<PolicyConsideration> {
    let trigger = args.trigger.trim();
    let behavior = args.behavior.trim();
    let advice = args.advice.trim();
    if trigger.is_empty() || behavior.is_empty() || advice.is_empty() {
        return None;
    }
    Some(PolicyConsideration {
        candidate_id: "synthetic:0".into(),
        source: PolicyConsiderationSource::Synthetic(SyntheticPolicyConsideration {
            trigger: trigger.to_owned(),
            behavior: behavior.to_owned(),
        }),
        predicted_expected_reward: predicted_expected_reward_for(args.expected_value),
        confidence_hint: 0.5,
        advice: advice.to_owned(),
        rationale: "synthetic policy candidate proposed through propose_policy_candidate".into(),
    })
}

fn predicted_expected_reward_for(value: PolicyExpectedValue) -> f32 {
    match value {
        PolicyExpectedValue::StrongNegative => -0.75,
        PolicyExpectedValue::Negative => -0.35,
        PolicyExpectedValue::Neutral => 0.0,
        PolicyExpectedValue::Positive => 0.35,
        PolicyExpectedValue::StrongPositive => 0.75,
    }
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
        Lutum, MessageContent, MockLlmAdapter, MockStructuredScenario, MockTextScenario,
        ModelInput, ModelInputItem, RawStructuredTurnEvent, RawTextTurnEvent,
        SharedPoolBudgetManager, SharedPoolBudgetOptions, TurnAdapter, Usage,
    };
    use nuillu_blackboard::Blackboard;
    use nuillu_blackboard::IdentityMemoryRecord;
    use nuillu_module::ports::SystemClock;
    use nuillu_module::{
        CapabilityProviderPorts, CapabilityProviders, LlmConcurrencyLimiter, LutumTiers,
        ModuleRegistry, SessionCompactionPolicy, SessionCompactionRuntime,
    };
    use nuillu_types::{MemoryContent, MemoryIndex, ModelTier, ReplicaCapRange};
    use nuillu_types::{ModuleInstanceId, ReplicaIndex, builtin};
    use nuillu_types::{SignedUnitF32, UnitF32};

    use super::*;
    use crate::{
        IndexedPolicy, NewPolicy, PolicyCapabilities, PolicyQuery, PolicyRecord, PolicyStore,
    };

    #[derive(Clone)]
    struct CapturingTextAdapter {
        inner: MockLlmAdapter,
        text_inputs: Arc<Mutex<Vec<ModelInput>>>,
        structured_inputs: Arc<Mutex<Vec<ModelInput>>>,
    }

    impl CapturingTextAdapter {
        fn new(inner: MockLlmAdapter) -> Self {
            Self {
                inner,
                text_inputs: Arc::new(Mutex::new(Vec::new())),
                structured_inputs: Arc::new(Mutex::new(Vec::new())),
            }
        }

        fn text_inputs(&self) -> Vec<ModelInput> {
            self.text_inputs.lock().unwrap().clone()
        }

        fn structured_inputs(&self) -> Vec<ModelInput> {
            self.structured_inputs.lock().unwrap().clone()
        }
    }

    #[async_trait::async_trait]
    impl TurnAdapter for CapturingTextAdapter {
        async fn text_turn(
            &self,
            input: ModelInput,
            turn: AdapterTextTurn,
        ) -> Result<ErasedTextTurnEventStream, AgentError> {
            self.text_inputs.lock().unwrap().push(input.clone());
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

    fn proposed_candidate_scenario(input_tokens: u64) -> MockTextScenario {
        policy_tool_scenario(
            serde_json::json!({
                "trigger": "fresh bounded trigger",
                "behavior": "fresh bounded behavior",
                "advice": "continue cautiously",
                "expected_value": "positive"
            })
            .to_string(),
            input_tokens,
        )
    }

    fn rejected_candidate_scenario(input_tokens: u64) -> MockTextScenario {
        policy_tool_scenario(
            serde_json::json!({
                "trigger": "REJECTED_TRIGGER_MARKER",
                "behavior": "REJECTED_BEHAVIOR_MARKER",
                "advice": "REJECTED_ADVICE_MARKER",
                "expected_value": "neutral"
            })
            .to_string(),
            input_tokens,
        )
    }

    fn invalid_candidate_scenario(input_tokens: u64) -> MockTextScenario {
        policy_tool_scenario(
            serde_json::json!({
                "trigger": "",
                "behavior": "fresh bounded behavior",
                "advice": "continue cautiously",
                "expected_value": "positive"
            })
            .to_string(),
            input_tokens,
        )
    }

    fn no_tool_scenario(input_tokens: u64) -> MockTextScenario {
        MockTextScenario::events(vec![
            Ok(RawTextTurnEvent::Started {
                request_id: Some("policy-no-tool".into()),
                model: "mock".into(),
            }),
            Ok(RawTextTurnEvent::TextDelta {
                delta: "no new reusable candidate".into(),
            }),
            Ok(RawTextTurnEvent::Completed {
                request_id: Some("policy-no-tool".into()),
                finish_reason: FinishReason::Stop,
                usage: text_usage(input_tokens),
            }),
        ])
    }

    fn malformed_tool_scenario(input_tokens: u64) -> MockTextScenario {
        policy_tool_scenario(
            serde_json::json!({
                "trigger": "fresh bounded trigger",
                "behavior": "fresh bounded behavior",
                "advice": "continue cautiously",
                "expected_value": 0.7
            })
            .to_string(),
            input_tokens,
        )
    }

    fn policy_tool_scenario(arguments_json: String, input_tokens: u64) -> MockTextScenario {
        MockTextScenario::events(vec![
            Ok(RawTextTurnEvent::Started {
                request_id: Some("policy-tool".into()),
                model: "mock".into(),
            }),
            Ok(RawTextTurnEvent::ToolCallChunk {
                id: "call-policy".into(),
                name: "propose_policy_candidate".into(),
                arguments_json_delta: arguments_json,
            }),
            Ok(RawTextTurnEvent::Completed {
                request_id: Some("policy-tool".into()),
                finish_reason: FinishReason::ToolCall,
                usage: text_usage(input_tokens),
            }),
        ])
    }

    fn policy_candidate_evaluation_scenario(
        accepted: bool,
        reason: &str,
        input_tokens: u64,
    ) -> MockStructuredScenario {
        let json = serde_json::json!({
            "accepted": accepted,
            "reason": reason,
        })
        .to_string();
        MockStructuredScenario::events(vec![
            Ok(RawStructuredTurnEvent::Started {
                request_id: Some("policy-candidate-eval".into()),
                model: "mock".into(),
            }),
            Ok(RawStructuredTurnEvent::StructuredOutputChunk { json_delta: json }),
            Ok(RawStructuredTurnEvent::Completed {
                request_id: Some("policy-candidate-eval".into()),
                finish_reason: FinishReason::Stop,
                usage: text_usage(input_tokens),
            }),
        ])
    }

    fn policy_candidate_evaluation_refusal_scenario(input_tokens: u64) -> MockStructuredScenario {
        MockStructuredScenario::events(vec![
            Ok(RawStructuredTurnEvent::Started {
                request_id: Some("policy-candidate-eval-refusal".into()),
                model: "mock".into(),
            }),
            Ok(RawStructuredTurnEvent::RefusalDelta {
                delta: "cannot evaluate".into(),
            }),
            Ok(RawStructuredTurnEvent::Completed {
                request_id: Some("policy-candidate-eval-refusal".into()),
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

    fn activate_cx_with_identity<'a>(
        lutum: &Lutum,
        now: chrono::DateTime<chrono::Utc>,
        identity_memories: &'a [IdentityMemoryRecord],
    ) -> nuillu_module::ActivateCx<'a> {
        nuillu_module::ActivateCx::new(
            &[],
            &[],
            identity_memories,
            &[],
            compaction_runtime(lutum),
            now,
        )
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
                ModelInputItem::Turn(turn) => {
                    for index in 0..turn.item_count() {
                        let Some(item) = turn.item_at(index) else {
                            continue;
                        };
                        if let Some(text) = item.as_text() {
                            out.push_str(text);
                            out.push('\n');
                        }
                    }
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
    fn deterministic_existing_and_synthetic_candidate_normalization() {
        let mut considerations = deterministic_existing_considerations(&[hit("p1", -0.2)]);
        considerations.push(
            normalize_proposed_candidate(ProposePolicyCandidateArgs {
                trigger: "new situation".into(),
                behavior: "new behavior".into(),
                advice: "try carefully".into(),
                expected_value: PolicyExpectedValue::StrongPositive,
            })
            .unwrap(),
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
            vec![("existing:p1", -0.2, 0.4), ("synthetic:0", 0.75, 0.5)]
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn activation_bounds_policy_input_and_seeds_stable_prompt_once() {
        let adapter = MockLlmAdapter::new()
            .with_text_scenario(proposed_candidate_scenario(1))
            .with_structured_scenario(policy_candidate_evaluation_scenario(
                true,
                "behavioral guidance",
                1,
            ))
            .with_text_scenario(proposed_candidate_scenario(1))
            .with_structured_scenario(policy_candidate_evaluation_scenario(
                true,
                "behavioral guidance",
                1,
            ));
        let capture = CapturingTextAdapter::new(adapter);
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
                        origin: nuillu_blackboard::CognitionLogOrigin::direct(
                            cognition_owner.clone(),
                        ),
                    },
                )
                .await;
        }

        let cx = activate_cx(&lutum, now);
        let batch = module.next_batch().await.unwrap();
        module.activate(&cx, &batch).await.unwrap();

        let first = observed.text_inputs();
        assert_eq!(first.len(), 1);
        let first_text = all_input_text(&first[0]);
        assert!(first_text.contains("answer the current bounded policy query"));
        assert!(first_text.contains("Current interoception:"));
        assert!(first_text.contains("- affect_arousal:"));
        assert!(first_text.contains("Existing policy hits already handled by runtime:"));
        assert!(first_text.contains("- policy_index: policy-0"));
        assert!(first_text.contains("policy-0"));
        assert!(first_text.contains("policy-4"));
        assert!(!first_text.contains("policy-5"));
        assert!(!first_text.contains("\"policy_index\""));
        assert!(!first_text.contains("{\""));
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

        let inputs = observed.text_inputs();
        assert_eq!(inputs.len(), 2);
        assert_eq!(system_prompt_count(&inputs[1]), 1);
        assert_eq!(store.search_limits.borrow().as_slice(), &[5, 5]);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn activation_writes_existing_hit_when_model_calls_no_tool() {
        let adapter = MockLlmAdapter::new()
            .with_text_scenario(no_tool_scenario(1))
            .with_text_scenario(proposed_candidate_scenario(1))
            .with_structured_scenario(policy_candidate_evaluation_scenario(
                true,
                "behavioral guidance",
                1,
            ));
        let capture = CapturingTextAdapter::new(adapter);
        let observed = capture.clone();
        let blackboard = Blackboard::default();
        let (caps, lutum) = test_caps_with_adapter(blackboard.clone(), Arc::new(capture));
        let store = Rc::new(RecordingPolicyStore::with_records(vec![record(
            "p1",
            "when Alice asks a gentle question",
            "answer with calm curiosity",
        )]));
        let policy_caps = PolicyCapabilities::new(
            blackboard.clone(),
            Rc::new(SystemClock),
            store.clone(),
            Vec::new(),
        );
        let snapshots = policy_caps.clone();
        let mut module = build_policy_module(&caps, policy_caps).await;

        let now = chrono::Utc::now();
        publish_policy_wakeup_with_content(&caps, &blackboard, "first policy evidence", now).await;
        let cx = activate_cx(&lutum, now);
        let batch = module.next_batch().await.unwrap();
        module.activate(&cx, &batch).await.unwrap();

        let records = snapshots.consideration_snapshots();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].payload.considerations.len(), 1);
        let consideration = &records[0].payload.considerations[0];
        assert_eq!(consideration.candidate_id, "existing:p1");
        assert!(matches!(
            consideration.source,
            PolicyConsiderationSource::Existing(_)
        ));

        publish_policy_wakeup_with_content(
            &caps,
            &blackboard,
            "second policy evidence",
            now + chrono::Duration::seconds(1),
        )
        .await;
        let batch = module.next_batch().await.unwrap();
        module.activate(&cx, &batch).await.unwrap();

        let inputs = observed.text_inputs();
        assert_eq!(inputs.len(), 2);
        let second_text = all_input_text(&inputs[1]);
        assert!(second_text.contains("first policy evidence"));
        assert!(second_text.contains("no new reusable candidate"));
        assert!(second_text.contains("second policy evidence"));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn activation_writes_synthetic_candidate_after_contract_acceptance() {
        let adapter = MockLlmAdapter::new()
            .with_text_scenario(proposed_candidate_scenario(1))
            .with_structured_scenario(policy_candidate_evaluation_scenario(
                true,
                "behavioral guidance",
                1,
            ));
        let capture = CapturingTextAdapter::new(adapter);
        let observed = capture.clone();
        let blackboard = Blackboard::default();
        let (caps, lutum) = test_caps_with_adapter(blackboard.clone(), Arc::new(capture));
        let store = Rc::new(RecordingPolicyStore::default());
        let policy_caps =
            PolicyCapabilities::new(blackboard.clone(), Rc::new(SystemClock), store, Vec::new());
        let snapshots = policy_caps.clone();
        let mut module = build_policy_module(&caps, policy_caps).await;

        let now = chrono::Utc::now();
        publish_policy_wakeup_with_content(&caps, &blackboard, "fresh policy need", now).await;
        let cx = activate_cx(&lutum, now);
        let batch = module.next_batch().await.unwrap();
        module.activate(&cx, &batch).await.unwrap();

        let records = snapshots.consideration_snapshots();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].payload.considerations.len(), 1);
        let consideration = &records[0].payload.considerations[0];
        assert_eq!(consideration.candidate_id, "synthetic:0");
        assert_eq!(consideration.advice, "continue cautiously");
        let structured_inputs = observed.structured_inputs();
        assert_eq!(structured_inputs.len(), 1);
        let eval_text = all_input_text(&structured_inputs[0]);
        assert!(eval_text.contains("fresh bounded trigger"));
        assert!(eval_text.contains("fresh bounded behavior"));
        assert!(eval_text.contains("continue cautiously"));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn activation_rejects_synthetic_candidate_and_drops_turn_from_session() {
        // The contract evaluator is a mock with a fixed `accepted: false` verdict, so the
        // candidate/identity strings below are plumbing markers, not content the test judges.
        // This exercises the rejection-handling path: the evidence reaches the evaluation
        // request, the rejected candidate never lands in a memo, and the dropped turn does not
        // leak into the next activation's session.
        let adapter = MockLlmAdapter::new()
            .with_text_scenario(rejected_candidate_scenario(1))
            .with_structured_scenario(policy_candidate_evaluation_scenario(
                false,
                "fixed mock rejection verdict",
                1,
            ))
            .with_text_scenario(no_tool_scenario(1));
        let capture = CapturingTextAdapter::new(adapter);
        let observed = capture.clone();
        let blackboard = Blackboard::default();
        let (caps, lutum) = test_caps_with_adapter(blackboard.clone(), Arc::new(capture));
        let store = Rc::new(RecordingPolicyStore::default());
        let policy_caps =
            PolicyCapabilities::new(blackboard.clone(), Rc::new(SystemClock), store, Vec::new());
        let snapshots = policy_caps.clone();
        let mut module = build_policy_module(&caps, policy_caps).await;

        let now = chrono::Utc::now();
        publish_policy_wakeup_with_content(&caps, &blackboard, "REJECTED_QUERY_MARKER", now).await;
        let identity_memories = vec![IdentityMemoryRecord {
            index: MemoryIndex::new("identity-evidence-marker"),
            content: MemoryContent::new("IDENTITY_EVIDENCE_MARKER"),
            occurred_at: None,
        }];
        let cx = activate_cx_with_identity(&lutum, now, &identity_memories);
        let batch = module.next_batch().await.unwrap();
        module.activate(&cx, &batch).await.unwrap();

        assert!(snapshots.consideration_snapshots().is_empty());
        let structured_inputs = observed.structured_inputs();
        assert_eq!(structured_inputs.len(), 1);
        let eval_text = all_input_text(&structured_inputs[0]);
        assert!(eval_text.contains("What you already remember about yourself"));
        assert_eq!(
            eval_text
                .matches("What you already remember about yourself")
                .count(),
            1
        );
        assert!(!eval_text.contains("Identity memory loaded at agent startup"));
        assert!(eval_text.contains("IDENTITY_EVIDENCE_MARKER"));
        assert!(eval_text.contains("REJECTED_ADVICE_MARKER"));
        assert!(
            blackboard
                .read(|bb| bb.recent_memo_logs())
                .await
                .iter()
                .all(|record| !record.content.contains("REJECTED_ADVICE_MARKER"))
        );

        publish_policy_wakeup_with_content(
            &caps,
            &blackboard,
            "clean policy evidence",
            now + chrono::Duration::seconds(1),
        )
        .await;
        let batch = module.next_batch().await.unwrap();
        module.activate(&cx, &batch).await.unwrap();

        let inputs = observed.text_inputs();
        assert_eq!(inputs.len(), 2);
        let second_text = all_input_text(&inputs[1]);
        assert!(!second_text.contains("REJECTED_ADVICE_MARKER"));
        assert!(second_text.contains("clean policy evidence"));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn activation_drops_synthetic_candidate_when_contract_evaluation_refuses() {
        let adapter = MockLlmAdapter::new()
            .with_text_scenario(proposed_candidate_scenario(1))
            .with_structured_scenario(policy_candidate_evaluation_refusal_scenario(1));
        let blackboard = Blackboard::default();
        let (caps, lutum) = test_caps_with_adapter(blackboard.clone(), Arc::new(adapter));
        let store = Rc::new(RecordingPolicyStore::default());
        let policy_caps =
            PolicyCapabilities::new(blackboard.clone(), Rc::new(SystemClock), store, Vec::new());
        let snapshots = policy_caps.clone();
        let mut module = build_policy_module(&caps, policy_caps).await;

        let now = chrono::Utc::now();
        publish_policy_wakeup_with_content(&caps, &blackboard, "fresh policy need", now).await;
        let cx = activate_cx(&lutum, now);
        let batch = module.next_batch().await.unwrap();
        module.activate(&cx, &batch).await.unwrap();

        assert!(snapshots.consideration_snapshots().is_empty());
        assert!(
            blackboard
                .read(|bb| bb.recent_memo_logs())
                .await
                .iter()
                .all(|record| record.owner.module != builtin::policy())
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn activation_drops_invalid_or_malformed_synthetic_candidate_without_failure() {
        for scenario in [invalid_candidate_scenario(1), malformed_tool_scenario(1)] {
            let adapter = MockLlmAdapter::new().with_text_scenario(scenario);
            let blackboard = Blackboard::default();
            let (caps, lutum) = test_caps_with_adapter(blackboard.clone(), Arc::new(adapter));
            let store = Rc::new(RecordingPolicyStore::default());
            let policy_caps = PolicyCapabilities::new(
                blackboard.clone(),
                Rc::new(SystemClock),
                store.clone(),
                Vec::new(),
            );
            let snapshots = policy_caps.clone();
            let mut module = build_policy_module(&caps, policy_caps).await;

            let now = chrono::Utc::now();
            publish_policy_wakeup(&caps, &blackboard, now).await;
            let cx = activate_cx(&lutum, now);
            let batch = module.next_batch().await.unwrap();
            module.activate(&cx, &batch).await.unwrap();

            assert!(snapshots.consideration_snapshots().is_empty());
        }
    }

    #[tokio::test(flavor = "current_thread")]
    async fn activation_drops_malformed_candidate_turn_from_session() {
        let adapter = MockLlmAdapter::new()
            .with_text_scenario(malformed_tool_scenario(1))
            .with_text_scenario(no_tool_scenario(1));
        let capture = CapturingTextAdapter::new(adapter);
        let observed = capture.clone();
        let blackboard = Blackboard::default();
        let (caps, lutum) = test_caps_with_adapter(blackboard.clone(), Arc::new(capture));
        let store = Rc::new(RecordingPolicyStore::default());
        let policy_caps =
            PolicyCapabilities::new(blackboard.clone(), Rc::new(SystemClock), store, Vec::new());
        let mut module = build_policy_module(&caps, policy_caps).await;

        let now = chrono::Utc::now();
        publish_policy_wakeup_with_content(&caps, &blackboard, "malformed policy evidence", now)
            .await;
        let cx = activate_cx(&lutum, now);
        let batch = module.next_batch().await.unwrap();
        module.activate(&cx, &batch).await.unwrap();

        publish_policy_wakeup_with_content(
            &caps,
            &blackboard,
            "clean policy evidence",
            now + chrono::Duration::seconds(1),
        )
        .await;
        let batch = module.next_batch().await.unwrap();
        module.activate(&cx, &batch).await.unwrap();

        let inputs = observed.text_inputs();
        assert_eq!(inputs.len(), 2);
        let second_text = all_input_text(&inputs[1]);
        assert!(!second_text.contains("malformed policy evidence"));
        assert!(second_text.contains("clean policy evidence"));
        assert!(
            !inputs[1]
                .items()
                .iter()
                .any(|item| matches!(item, ModelInputItem::ToolResult(_)))
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

    async fn publish_policy_wakeup(
        caps: &CapabilityProviders,
        blackboard: &Blackboard,
        now: chrono::DateTime<chrono::Utc>,
    ) {
        publish_policy_wakeup_with_content(
            caps,
            blackboard,
            "policy-relevant current context",
            now,
        )
        .await;
    }

    async fn publish_policy_wakeup_with_content(
        caps: &CapabilityProviders,
        blackboard: &Blackboard,
        content: impl Into<String>,
        now: chrono::DateTime<chrono::Utc>,
    ) {
        let memo_owner = ModuleInstanceId::new(builtin::self_model(), ReplicaIndex::ZERO);
        let record = blackboard
            .update_memo(memo_owner, content.into(), now)
            .await;
        caps.internal_harness_io()
            .memo_updated_mailbox()
            .publish(nuillu_module::MemoUpdated {
                owner: record.owner,
                index: record.index,
            })
            .await
            .unwrap();
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
