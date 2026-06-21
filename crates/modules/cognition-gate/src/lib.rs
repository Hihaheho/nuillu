use std::collections::BTreeSet;

use anyhow::{Context, Result, anyhow};
use async_trait::async_trait;
use lutum::{Lutum, Session, TextStepOutcomeWithTools, ToolResult, Usage};
use nuillu_blackboard::MemoLogRecord;
use nuillu_module::{
    BlackboardReader, CognitionWriter, LlmAccess, LlmContextWindow, MemoUpdatedInbox, Module,
    SessionAutoCompaction, SessionCompactionConfig, SessionCompactionProtectedPrefix,
    compact_llm_context_text, ensure_persistent_session_seeded, push_formatted_cognition_log_batch,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

mod batch;
pub use batch::NextBatch as CognitionGateBatch;

const SYSTEM_PROMPT: &str = r#"Read the candidate facts and rank which should become part of this
agent's current awareness. Candidate order and labels are not priority signals.

Rank current, action-relevant facts highest: direct participant questions, requests,
warnings, safety constraints, fresh sensory or world facts, body facts, and memory
evidence needed to answer or act now. Prefer candidates that add information not
already present in recent awareness.

Rank stale greetings, redundant restatements, speech-planning status, process notes,
model reasoning, retrieval plumbing, confidence notes, evidence-gap notes, and decision
rationales lower when a concrete candidate exists. If every candidate is weak, choose
the most concrete world, body, memory, or participant-speech fact.

Use the available ranking tool once."#;

const COMPACTED_COGNITION_GATE_SESSION_PREFIX: &str = "Compacted prior ranking context:";
const COGNITION_CONTEXT_WINDOW: LlmContextWindow = LlmContextWindow::new(12, 600, 4_800);
const SESSION_COMPACTION_FOCUS: &str = r#"Preserve candidate facts, prior ranking decisions,
admitted events, rejected candidate events, recent awareness context, and relevant memory evidence
needed for future ranking decisions."#;
const NEW_CANDIDATE_HEADER: &str = "Candidate facts:";
const CANDIDATE_DECISION_INSTRUCTION: &str =
    "Rank the bracket labels. Use the available ranking tool now.";

const CANDIDATE_TEXT_CONTEXT_CHARS: usize = 1_200;

pub fn session_auto_compaction() -> SessionAutoCompaction {
    SessionAutoCompaction::new(
        SessionCompactionConfig::default(),
        SessionCompactionProtectedPrefix::LeadingSystemAndIdentitySeed,
        COMPACTED_COGNITION_GATE_SESSION_PREFIX,
        SESSION_COMPACTION_FOCUS,
    )
}

#[lutum::tool_input(
    name = "rank_awareness_candidates",
    output = RankAwarenessCandidatesOutput
)]
/// Rank candidate labels for the agent's current awareness.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct RankAwarenessCandidatesArgs {
    /// Bracket labels ordered from most important to least important, such as A, B, C.
    pub labels: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct RankAwarenessCandidatesOutput {
    pub accepted: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rejected_reason: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema, lutum::Toolset)]
pub enum CognitionGateRankingTools {
    RankAwarenessCandidates(RankAwarenessCandidatesArgs),
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct CognitionCandidate {
    id: String,
    record: MemoLogRecord,
}

pub type CognitionGateSessionCompactionConfig = SessionCompactionConfig;

pub struct CognitionGateModule {
    memo_updates: MemoUpdatedInbox,
    blackboard: BlackboardReader,
    cognition: CognitionWriter,
    llm: LlmAccess,
    session: Session,
    last_seen_cognition_index: Option<u64>,
    pending_unread_memos: Vec<MemoLogRecord>,
}

impl CognitionGateModule {
    pub fn new(
        memo_updates: MemoUpdatedInbox,
        blackboard: BlackboardReader,
        cognition: CognitionWriter,
        llm: LlmAccess,
        session: Session,
    ) -> Self {
        Self {
            memo_updates,
            blackboard,
            cognition,
            llm,
            session,
            last_seen_cognition_index: None,
            pending_unread_memos: Vec::new(),
        }
    }

    fn ensure_session_seeded(&mut self, cx: &nuillu_module::ActivateCx<'_>) {
        ensure_persistent_session_seeded(
            &mut self.session,
            SYSTEM_PROMPT,
            cx.identity_memories(),
            cx.now(),
        );
    }

    #[tracing::instrument(skip_all, err(Debug, level = "warn"))]
    async fn activate(
        &mut self,
        cx: &nuillu_module::ActivateCx<'_>,
        batch: &CognitionGateBatch,
    ) -> Result<()> {
        let self_owner = self.cognition.owner().clone();
        let (unread_cognition, latest_cognition_index) = self
            .blackboard
            .read(|bb| {
                let records = bb.cognition_log_entries_after_index(self.last_seen_cognition_index);
                let latest_index = records.last().map(|record| record.index);
                let filtered = records
                    .into_iter()
                    .filter(|record| {
                        record.source != self_owner && record.entry.origin.owner != self_owner
                    })
                    .collect::<Vec<_>>();
                (filtered, latest_index)
            })
            .await;
        for record in &batch.memo_logs {
            if !self
                .pending_unread_memos
                .iter()
                .any(|pending| pending.owner == record.owner && pending.index == record.index)
            {
                self.pending_unread_memos.push(record.clone());
            }
        }

        let candidates = clean_candidates(&self.pending_unread_memos);
        if candidates.is_empty() {
            if let Some(index) = latest_cognition_index {
                self.last_seen_cognition_index = Some(index);
            }
            self.pending_unread_memos.clear();
            return Ok(());
        }
        if candidates.len() <= 2 {
            let selected = candidates
                .iter()
                .map(|candidate| candidate.record.clone())
                .collect::<Vec<_>>();
            self.append_ranked_records(&selected).await?;
            if let Some(index) = latest_cognition_index {
                self.last_seen_cognition_index = Some(index);
            }
            self.pending_unread_memos.clear();
            return Ok(());
        }

        self.ensure_session_seeded(cx);
        let session_len_before_activation = self.session.input().items().len();

        push_formatted_cognition_log_batch(
            &mut self.session,
            &unread_cognition,
            cx.now(),
            COGNITION_CONTEXT_WINDOW,
        );
        self.session
            .push_user(format_candidate_selection_prompt(&candidates));

        let lutum = self.llm.lutum().await;
        let result = self.run_selection_turn(&lutum, cx, &candidates).await;
        if let Err(error) = result {
            self.session
                .input_mut()
                .items_mut()
                .truncate(session_len_before_activation);
            cx.compact_and_save(&mut self.session, Usage::zero())
                .await?;
            return Err(error);
        }
        if let Some(index) = latest_cognition_index {
            self.last_seen_cognition_index = Some(index);
        }
        self.pending_unread_memos.clear();
        Ok(())
    }

    async fn run_selection_turn(
        &mut self,
        lutum: &Lutum,
        cx: &nuillu_module::ActivateCx<'_>,
        candidates: &[CognitionCandidate],
    ) -> Result<()> {
        let outcome = self
            .session
            .text_turn()
            .tools::<CognitionGateRankingTools>()
            .available_tools([CognitionGateRankingToolsSelector::RankAwarenessCandidates])
            .require_any_tool()
            .max_output_tokens(768)
            .collect_controlled_with(lutum, nuillu_module::AbortOnAvailableToolNameInText::new())
            .await
            .map_err(|error| {
                if missing_required_tool_call(&error) {
                    anyhow!("cognition-gate finished without required tool call")
                } else {
                    anyhow!(error).context("cognition-gate decision turn failed")
                }
            })?;

        match outcome {
            // `require_any_tool()` should prevent a finish-without-tools outcome.
            TextStepOutcomeWithTools::Finished(_result) => {
                let detail = "model finished with assistant output but no tool call \
                                (require_any_tool should have prevented this outcome)";
                cx.warn(format!("cognition-gate activation failed: {detail}"));
                anyhow::bail!("cognition-gate finished without required tool call: {detail}");
            }
            TextStepOutcomeWithTools::FinishedNoOutput(_result) => {
                let detail = "model finished with no output and no tool call \
                                (require_any_tool should have prevented this outcome)";
                cx.warn(format!("cognition-gate activation failed: {detail}"));
                anyhow::bail!("cognition-gate finished without required tool call: {detail}");
            }
            TextStepOutcomeWithTools::NeedsTools(round) => {
                let usage = round.usage;
                let tool_names = format_tool_call_names(&round.tool_calls);
                nuillu_module::emit_trace_tool_calls(&round.tool_calls);
                if round.tool_calls.len() != 1 {
                    let detail = no_selection_failure_detail(&tool_names);
                    cx.warn(format!("cognition-gate activation failed: {detail}"));
                    anyhow::bail!("cognition-gate {detail}");
                }

                let call = round
                    .tool_calls
                    .first()
                    .expect("exactly one tool call exists")
                    .clone();
                let CognitionGateRankingToolsCall::RankAwarenessCandidates(call) = call;
                let selected_records = self
                    .resolve_ranked_candidates(&call.input, candidates)
                    .map_err(|error| {
                        cx.warn(format!("cognition-gate activation failed: {error}"));
                        error
                    })?;
                let results: Vec<ToolResult> = vec![
                    call.complete(RankAwarenessCandidatesOutput {
                        accepted: true,
                        rejected_reason: None,
                    })
                    .context("complete rank_awareness_candidates tool call")?,
                ];
                round
                    .commit(&mut self.session, results)
                    .context("commit cognition-gate tool round")?;
                self.append_ranked_records(&selected_records).await?;
                cx.compact_and_save(&mut self.session, usage).await?;
                Ok(())
            }
        }
    }

    fn resolve_ranked_candidates(
        &self,
        args: &RankAwarenessCandidatesArgs,
        candidates: &[CognitionCandidate],
    ) -> Result<Vec<MemoLogRecord>> {
        ranked_candidate_records(args, candidates)
    }

    async fn append_ranked_records(&self, records: &[MemoLogRecord]) -> Result<()> {
        if records.is_empty() {
            anyhow::bail!("no cognition record to append");
        }
        if records.len() > 2 {
            anyhow::bail!("too many cognition records");
        }
        for record in records {
            let text = record.content.trim();
            if text.is_empty() {
                anyhow::bail!("empty cognition text");
            }
            let mut trimmed = record.clone();
            trimmed.content = text.to_owned();
            self.cognition.append_from_memo(&trimmed).await;
        }
        Ok(())
    }
}

fn missing_required_tool_call(error: &impl std::fmt::Display) -> bool {
    error
        .to_string()
        .contains("required tool call was not produced")
}

fn no_selection_failure_detail(tool_names: &str) -> String {
    format!(
        "tool turn produced no single candidate ranking: tool_calls=[{tool_names}]; expected \
         rank_awareness_candidates"
    )
}

fn format_tool_call_names<C>(calls: &[C]) -> String
where
    C: lutum::ToolCallWrapper,
{
    if calls.is_empty() {
        return "(none)".to_owned();
    }
    calls
        .iter()
        .map(|call| call.metadata().name.as_str())
        .collect::<Vec<_>>()
        .join(", ")
}

fn clean_candidates(records: &[MemoLogRecord]) -> Vec<CognitionCandidate> {
    let mut seen = BTreeSet::new();
    let mut deduped_records = Vec::new();
    for record in records {
        let text = record.content.trim().to_owned();
        if text.is_empty() {
            continue;
        }
        if !seen.insert(text.clone()) {
            continue;
        }
        let mut record = record.clone();
        record.content = text;
        deduped_records.push(record);
    }

    records_to_letter_candidates(deduped_records)
}

fn format_candidate_selection_prompt(candidates: &[CognitionCandidate]) -> String {
    let mut out = NEW_CANDIDATE_HEADER.to_owned();
    for candidate in candidates {
        out.push('\n');
        out.push('[');
        out.push_str(&candidate.id);
        out.push_str("] ");
        out.push_str(&compact_llm_context_text(
            &candidate.record.content,
            CANDIDATE_TEXT_CONTEXT_CHARS,
        ));
    }
    out.push_str("\n\n");
    out.push_str(CANDIDATE_DECISION_INSTRUCTION);
    out
}

fn ranked_candidate_records(
    args: &RankAwarenessCandidatesArgs,
    candidates: &[CognitionCandidate],
) -> Result<Vec<MemoLogRecord>> {
    if args.labels.is_empty() {
        anyhow::bail!("rank_awareness_candidates rejected empty labels");
    }
    let mut seen = BTreeSet::new();
    let mut selected = Vec::new();
    for (index, id) in args.labels.iter().enumerate() {
        let field = format!("labels[{index}]");
        let id = normalize_candidate_id(id);
        if id.is_empty() {
            anyhow::bail!("rank_awareness_candidates rejected empty {field}");
        }
        if !seen.insert(id.clone()) {
            anyhow::bail!("rank_awareness_candidates rejected duplicate candidate IDs");
        }
        let record = candidate_record_by_id(candidates, &id, &field)?;
        if selected.len() < 2 {
            selected.push(record.clone());
        }
    }
    Ok(selected)
}

fn candidate_record_by_id<'a>(
    candidates: &'a [CognitionCandidate],
    id: &str,
    field: &str,
) -> Result<&'a MemoLogRecord> {
    if id.is_empty() {
        anyhow::bail!("rank_awareness_candidates rejected empty {field}");
    }
    candidates
        .iter()
        .find(|candidate| candidate.id == id)
        .map(|candidate| &candidate.record)
        .ok_or_else(|| {
            let valid_ids = candidates
                .iter()
                .map(|candidate| candidate.id.as_str())
                .collect::<Vec<_>>()
                .join(", ");
            anyhow!(
                "rank_awareness_candidates rejected {field}={id} outside candidate IDs [{valid_ids}]"
            )
        })
}

impl CognitionCandidate {
    fn new(id: String, record: MemoLogRecord) -> Self {
        Self { id, record }
    }
}

fn records_to_letter_candidates(records: Vec<MemoLogRecord>) -> Vec<CognitionCandidate> {
    records
        .into_iter()
        .enumerate()
        .map(|(index, record)| CognitionCandidate::new(candidate_letter_id(index), record))
        .collect()
}

fn candidate_letter_id(index: usize) -> String {
    let mut n = index;
    let mut chars = Vec::new();
    loop {
        chars.push(char::from(b'A' + (n % 26) as u8));
        n /= 26;
        if n == 0 {
            break;
        }
        n -= 1;
    }
    chars.iter().rev().collect()
}

fn normalize_candidate_id(value: &str) -> String {
    value
        .trim()
        .trim_start_matches('[')
        .trim_end_matches(']')
        .trim()
        .to_ascii_uppercase()
}

#[async_trait(?Send)]
impl Module for CognitionGateModule {
    type Batch = CognitionGateBatch;

    fn id() -> &'static str {
        "cognition-gate"
    }

    fn peer_context() -> Option<&'static str> {
        None
    }

    fn allocation_hint() -> Option<&'static str> {
        Some(
            "Raise cognition-gate when fresh candidates may deserve promotion into cognition. Keep it low for background noise or material that should remain outside conscious cognition.",
        )
    }

    async fn next_batch(&mut self) -> Result<Self::Batch> {
        CognitionGateModule::next_batch(self).await
    }

    async fn activate(
        &mut self,
        cx: &nuillu_module::ActivateCx<'_>,
        batch: &Self::Batch,
    ) -> Result<()> {
        CognitionGateModule::activate(self, cx, batch).await
    }
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::rc::Rc;
    use std::sync::{Arc, Mutex};
    use std::time::Duration;

    use lutum::{
        AdapterStructuredTurn, AdapterTextTurn, AdapterToolChoice, AgentError, AssistantInputItem,
        ErasedStructuredTurnEventStream, ErasedTextTurnEventStream, FinishReason, InputMessageRole,
        MessageContent, MockError, MockLlmAdapter, MockTextScenario, ModelInput, ModelInputItem,
        RawTextTurnEvent, SharedPoolBudgetManager, SharedPoolBudgetOptions, TurnAdapter, Usage,
    };
    use nuillu_blackboard::{
        ActivationRatio, Blackboard, Bpm, IdentityMemoryRecord, ModuleConfig, ResourceAllocation,
        linear_ratio_fn,
    };
    use nuillu_module::ports::{Clock, NoopCognitionLogRepository, SystemClock};
    use nuillu_module::{
        CapabilityProviderPorts, CapabilityProviders, LlmConcurrencyLimiter, LutumTiers, Memo,
        ModuleRegistry, SessionCompactionPolicy, SessionCompactionRuntime,
        session_compaction_cutoff,
    };
    use nuillu_types::{ModelTier, builtin};

    use super::*;

    #[derive(Clone)]
    struct CapturingAdapter {
        inner: MockLlmAdapter,
        text_inputs: Arc<Mutex<Vec<ModelInput>>>,
        text_turns: Arc<Mutex<Vec<AdapterTextTurn>>>,
    }

    impl CapturingAdapter {
        fn new(inner: MockLlmAdapter) -> Self {
            Self {
                inner,
                text_inputs: Arc::new(Mutex::new(Vec::new())),
                text_turns: Arc::new(Mutex::new(Vec::new())),
            }
        }

        fn text_inputs(&self) -> Vec<ModelInput> {
            self.text_inputs.lock().unwrap().clone()
        }

        fn text_turns(&self) -> Vec<AdapterTextTurn> {
            self.text_turns.lock().unwrap().clone()
        }
    }

    #[async_trait::async_trait]
    impl TurnAdapter for CapturingAdapter {
        async fn text_turn(
            &self,
            input: ModelInput,
            turn: AdapterTextTurn,
        ) -> Result<ErasedTextTurnEventStream, AgentError> {
            self.text_inputs.lock().unwrap().push(input.clone());
            self.text_turns.lock().unwrap().push(turn.clone());
            self.inner.text_turn(input, turn).await
        }

        async fn structured_turn(
            &self,
            input: ModelInput,
            turn: AdapterStructuredTurn,
        ) -> Result<ErasedStructuredTurnEventStream, AgentError> {
            self.inner.structured_turn(input, turn).await
        }
    }

    fn test_caps_with_turn_adapter<T>(
        blackboard: Blackboard,
        adapter: Arc<T>,
    ) -> CapabilityProviders
    where
        T: TurnAdapter,
    {
        let budget = SharedPoolBudgetManager::new(SharedPoolBudgetOptions::default());
        let lutum = Lutum::new(adapter, budget);
        CapabilityProviders::new(CapabilityProviderPorts {
            blackboard,
            cognition_log_port: Rc::new(NoopCognitionLogRepository),
            clock: Rc::new(SystemClock),
            tiers: LutumTiers::from_shared_lutum(lutum),
        })
    }

    fn test_allocation() -> ResourceAllocation {
        let mut allocation = ResourceAllocation::default();
        for module in [builtin::cognition_gate(), builtin::sensory()] {
            allocation.set(module.clone(), ModuleConfig::default());
            allocation.set_activation(module, ActivationRatio::ONE);
        }
        allocation
    }

    fn test_policy() -> nuillu_blackboard::ModulePolicy {
        nuillu_blackboard::ModulePolicy::new(
            nuillu_types::ReplicaCapRange::new(0, 0).unwrap(),
            Bpm::from_f64(60.0)..=Bpm::from_f64(60.0),
            linear_ratio_fn,
        )
    }

    macro_rules! noop_stub {
        ($name:ident, $id:literal) => {
            struct $name;

            #[async_trait(?Send)]
            impl Module for $name {
                type Batch = ();

                fn id() -> &'static str {
                    $id
                }

                fn peer_context() -> Option<&'static str> {
                    Some("test stub")
                }

                fn allocation_hint() -> Option<&'static str> {
                    Some("test allocation target")
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

    noop_stub!(CognitionGateStub, "cognition-gate");
    noop_stub!(SensoryStub, "sensory");

    struct GateFixture {
        gate: CognitionGateModule,
        source_memo: Memo,
        blackboard: Blackboard,
    }

    async fn gate_fixture_with_adapter(adapter: MockLlmAdapter) -> GateFixture {
        gate_fixture_with_turn_adapter(Arc::new(adapter)).await
    }

    async fn gate_fixture_with_turn_adapter<T>(adapter: Arc<T>) -> GateFixture
    where
        T: TurnAdapter,
    {
        let blackboard = Blackboard::with_allocation(test_allocation());
        let caps = test_caps_with_turn_adapter(blackboard.clone(), adapter);

        let gate_cell = Rc::new(RefCell::new(None));
        let source_memo_cell = Rc::new(RefCell::new(None));

        let gate_sink = Rc::clone(&gate_cell);
        let source_memo_sink = Rc::clone(&source_memo_cell);

        let _modules = ModuleRegistry::new()
            .register(test_policy(), move |caps| {
                let gate_sink = Rc::clone(&gate_sink);
                async move {
                    *gate_sink.borrow_mut() = Some(CognitionGateModule::new(
                        caps.memo_updated_inbox(),
                        caps.blackboard_reader(),
                        caps.cognition_writer(),
                        caps.llm("main")
                            .with_tier(nuillu_types::ModelTier::Default)
                            .into(),
                        caps.session("main")
                            .with_tier(nuillu_types::ModelTier::Default)
                            .with_auto_compaction(session_auto_compaction())
                            .await?,
                    ));
                    Ok(CognitionGateStub)
                }
            })
            .unwrap()
            .register(test_policy(), move |caps| {
                let source_memo_sink = Rc::clone(&source_memo_sink);
                async move {
                    *source_memo_sink.borrow_mut() = Some(caps.memo());
                    Ok(SensoryStub)
                }
            })
            .unwrap()
            .build(&caps)
            .await
            .unwrap();

        GateFixture {
            gate: gate_cell.borrow_mut().take().unwrap(),
            source_memo: source_memo_cell.borrow_mut().take().unwrap(),
            blackboard,
        }
    }

    fn text_usage(input_tokens: u64) -> Usage {
        Usage {
            input_tokens,
            ..Usage::zero()
        }
    }

    fn tool_scenario(name: &str, arguments_json: String, input_tokens: u64) -> MockTextScenario {
        MockTextScenario::events(vec![
            Ok(RawTextTurnEvent::Started {
                request_id: Some("cognition-gate-tool".into()),
                model: "mock".into(),
            }),
            Ok(RawTextTurnEvent::ToolCallChunk {
                id: "call-cognition-gate".into(),
                name: name.into(),
                arguments_json_delta: arguments_json,
            }),
            Ok(RawTextTurnEvent::Completed {
                request_id: Some("cognition-gate-tool".into()),
                finish_reason: FinishReason::ToolCall,
                usage: text_usage(input_tokens),
            }),
        ])
    }

    fn silent_scenario(input_tokens: u64) -> MockTextScenario {
        MockTextScenario::events(vec![
            Ok(RawTextTurnEvent::Started {
                request_id: Some("cognition-gate-silent".into()),
                model: "mock".into(),
            }),
            Ok(RawTextTurnEvent::Completed {
                request_id: Some("cognition-gate-silent".into()),
                finish_reason: FinishReason::Stop,
                usage: text_usage(input_tokens),
            }),
        ])
    }

    fn rank_awareness_candidates_scenario(
        labels: impl IntoIterator<Item = impl Into<String>>,
        input_tokens: u64,
    ) -> MockTextScenario {
        let labels = labels.into_iter().map(Into::into).collect::<Vec<String>>();
        let arguments = serde_json::json!({ "labels": labels });
        tool_scenario(
            "rank_awareness_candidates",
            arguments.to_string(),
            input_tokens,
        )
    }

    fn summary_text_scenario(summary: &str) -> MockTextScenario {
        MockTextScenario::events(vec![
            Ok(RawTextTurnEvent::Started {
                request_id: Some("cognition-gate-compact".into()),
                model: "mock".into(),
            }),
            Ok(RawTextTurnEvent::TextDelta {
                delta: summary.into(),
            }),
            Ok(RawTextTurnEvent::Completed {
                request_id: Some("cognition-gate-compact".into()),
                finish_reason: FinishReason::Stop,
                usage: Usage::zero(),
            }),
        ])
    }

    fn compaction_runtime(lutum: &Lutum) -> SessionCompactionRuntime {
        SessionCompactionRuntime::new(
            lutum.clone(),
            LlmConcurrencyLimiter::new(None),
            ModelTier::Cheap,
            SessionCompactionPolicy::default(),
        )
    }

    fn has_message_with_role_containing(
        items: &[ModelInputItem],
        expected_role: InputMessageRole,
        needle: &str,
    ) -> bool {
        items.iter().any(|item| matches!(
            item,
            ModelInputItem::Message { role, content }
                if role == &expected_role
                    && matches!(content.as_slice(), [MessageContent::Text(text)] if text.contains(needle))
        ))
    }

    fn message_texts_with_role(
        items: &[ModelInputItem],
        expected_role: InputMessageRole,
    ) -> Vec<&str> {
        items
            .iter()
            .filter_map(|item| match item {
                ModelInputItem::Message { role, content } if role == &expected_role => {
                    let [MessageContent::Text(text)] = content.as_slice() else {
                        panic!("expected one text content item");
                    };
                    Some(text.as_str())
                }
                _ => None,
            })
            .collect()
    }

    fn candidate_user_messages(items: &[ModelInputItem]) -> Vec<&str> {
        message_texts_with_role(items, InputMessageRole::User)
            .into_iter()
            .filter(|text| text.contains(NEW_CANDIDATE_HEADER))
            .collect()
    }

    fn latest_candidate_user_message(items: &[ModelInputItem]) -> &str {
        candidate_user_messages(items)
            .into_iter()
            .last()
            .expect("expected a candidate user message")
    }

    fn assert_candidate_decision_instruction(text: &str) {
        assert!(text.contains(CANDIDATE_DECISION_INSTRUCTION));
    }

    async fn next_gate_batch(fixture: &mut GateFixture) -> CognitionGateBatch {
        fixture.gate.next_batch().await.unwrap()
    }

    fn test_candidates(contents: &[&str]) -> Vec<CognitionCandidate> {
        test_candidates_for_module(builtin::sensory(), contents)
    }

    fn test_candidates_for_module(
        module: nuillu_types::ModuleId,
        contents: &[&str],
    ) -> Vec<CognitionCandidate> {
        let owner = nuillu_types::ModuleInstanceId::new(module, nuillu_types::ReplicaIndex::ZERO);
        let now = SystemClock.now();
        let records = contents
            .iter()
            .enumerate()
            .map(|(index, content)| MemoLogRecord {
                owner: owner.clone(),
                index: contents.len().saturating_sub(index + 1) as u64,
                written_at: now,
                content: (*content).to_owned(),
                cognitive: true,
            })
            .collect::<Vec<_>>();
        records_to_letter_candidates(records)
    }

    fn batch_from_candidates(candidates: &[CognitionCandidate]) -> CognitionGateBatch {
        CognitionGateBatch {
            memo_logs: candidates
                .iter()
                .map(|candidate| candidate.record.clone())
                .collect(),
        }
    }

    #[test]
    fn session_compaction_config_defaults_to_80_percent() {
        assert_eq!(
            CognitionGateSessionCompactionConfig::default(),
            CognitionGateSessionCompactionConfig { prefix_ratio: 0.8 }
        );
    }

    #[test]
    fn session_compaction_cutoff_uses_ratio_and_keeps_raw_suffix() {
        assert_eq!(session_compaction_cutoff(10, 0.8), Some(8));
        assert_eq!(session_compaction_cutoff(10, 1.0), Some(9));
        assert_eq!(session_compaction_cutoff(1, 0.8), None);
    }

    #[test]
    fn candidate_labels_are_letters_by_display_order() {
        let candidates = test_candidates(&[
            "candidate A",
            "candidate B",
            "candidate C",
            "candidate D",
            "candidate E",
        ]);

        assert_eq!(
            candidates
                .iter()
                .map(|candidate| candidate.id.as_str())
                .collect::<Vec<_>>(),
            vec!["A", "B", "C", "D", "E"]
        );
        assert_eq!(candidate_letter_id(25), "Z");
        assert_eq!(candidate_letter_id(26), "AA");
        assert_eq!(candidate_letter_id(27), "AB");
    }

    #[test]
    fn candidate_prompt_uses_letter_ids_not_numbers() {
        let candidates = test_candidates(&["candidate A", "candidate B", "candidate C"]);
        let prompt = format_candidate_selection_prompt(&candidates);

        for candidate in &candidates {
            assert!(prompt.contains(&format!("[{}] {}", candidate.id, candidate.record.content)));
        }
        assert!(!prompt.contains("1. candidate A"));
        assert!(!prompt.contains("2. candidate B"));
        assert!(!prompt.contains("3. candidate C"));
        assert!(prompt.contains("[A] candidate A"));
        assert!(prompt.contains("[B] candidate B"));
        assert!(prompt.contains("[C] candidate C"));
        assert!(!prompt.contains("candidate number"));
        assert!(!prompt.contains("rank_awareness_candidates"));
        assert!(!prompt.contains("candidate_ids"));
        assert!(!prompt.contains("top_candidate_ids"));
        assert!(!prompt.contains("one or two"));
    }

    #[test]
    fn system_prompt_is_rank_task_only() {
        assert!(SYSTEM_PROMPT.contains("agent's current awareness"));
        assert!(SYSTEM_PROMPT.contains("available ranking tool"));
        for forbidden in [
            "cognition-gate",
            "cognition log",
            "module",
            "other modules",
            "select_cognition_winner",
            "rank_top_cognition_candidates",
            "top_candidate_ids",
            "one or two",
            "champion",
            "winner",
            "race",
            "Mechanical brain plumbing",
            "conscious",
            "final assistant text",
            "free-form",
            "Do not write assistant text",
        ] {
            assert!(
                !SYSTEM_PROMPT.contains(forbidden),
                "rank prompt should not contain {forbidden:?}"
            );
        }
    }

    #[tokio::test(flavor = "current_thread")]
    async fn high_input_selection_compacts_session_prefix() {
        let candidates = test_candidates(&["candidate A"]);
        let adapter = MockLlmAdapter::new()
            .with_text_scenario(rank_awareness_candidates_scenario(
                [candidates[0].id.clone()],
                16_001,
            ))
            .with_text_scenario(summary_text_scenario(
                "old cognition gate history summarized",
            ));
        let mut fixture = gate_fixture_with_adapter(adapter).await;
        for index in 0..10 {
            fixture.gate.session.push_user(format!("history-{index}"));
        }

        let lutum = fixture.gate.llm.lutum().await;
        let identity_memories = Vec::new();
        let cx = nuillu_module::ActivateCx::new(
            &[],
            &[],
            &identity_memories,
            &[],
            compaction_runtime(&lutum),
            SystemClock.now(),
        );
        fixture
            .gate
            .run_selection_turn(&lutum, &cx, &candidates)
            .await
            .unwrap();
        let items = fixture.gate.session.input().items().to_vec();
        let summary = items
            .iter()
            .find_map(|item| match item {
                ModelInputItem::Assistant(AssistantInputItem::Text(summary))
                    if summary.starts_with(COMPACTED_COGNITION_GATE_SESSION_PREFIX) =>
                {
                    Some(summary)
                }
                _ => None,
            })
            .expect("expected compacted assistant message");
        assert!(summary.starts_with(COMPACTED_COGNITION_GATE_SESSION_PREFIX));
        assert!(summary.contains("old cognition gate history summarized"));

        let user_texts = items
            .iter()
            .filter_map(|item| match item {
                ModelInputItem::Message {
                    role: InputMessageRole::User,
                    content,
                } => {
                    let [MessageContent::Text(text)] = content.as_slice() else {
                        panic!("expected one text content item");
                    };
                    Some(text.as_str())
                }
                _ => None,
            })
            .collect::<Vec<_>>();
        assert!(!user_texts.contains(&"history-0"));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn threshold_input_selection_does_not_compact() {
        let candidates = test_candidates(&["candidate A"]);
        let adapter = MockLlmAdapter::new()
            .with_text_scenario(rank_awareness_candidates_scenario(
                [candidates[0].id.clone()],
                16_000,
            ))
            .with_text_scenario(summary_text_scenario("unexpected summary"));
        let mut fixture = gate_fixture_with_adapter(adapter).await;
        for index in 0..10 {
            fixture.gate.session.push_user(format!("history-{index}"));
        }

        let lutum = fixture.gate.llm.lutum().await;
        let identity_memories = Vec::new();
        let cx = nuillu_module::ActivateCx::new(
            &[],
            &[],
            &identity_memories,
            &[],
            compaction_runtime(&lutum),
            SystemClock.now(),
        );
        fixture
            .gate
            .run_selection_turn(&lutum, &cx, &candidates)
            .await
            .unwrap();
        let items = fixture.gate.session.input().items().to_vec();
        let ModelInputItem::Message { role, content } = &items[0] else {
            panic!("expected original first history item");
        };
        assert_eq!(role, &InputMessageRole::User);
        let [MessageContent::Text(text)] = content.as_slice() else {
            panic!("expected original history text");
        };
        assert_eq!(text, "history-0");

        assert!(!matches!(
            &items[0],
            ModelInputItem::Message { content, .. }
                if matches!(
                    content.as_slice(),
                    [MessageContent::Text(text)]
                        if text.starts_with(COMPACTED_COGNITION_GATE_SESSION_PREFIX)
                )
        ));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn activation_sends_three_or_more_candidates_as_persistent_user() {
        let candidates =
            test_candidates(&["sensory detail A", "sensory detail B", "sensory detail C"]);
        let adapter = MockLlmAdapter::new().with_text_scenario(rank_awareness_candidates_scenario(
            [candidates[1].id.clone()],
            1,
        ));
        let capture = CapturingAdapter::new(adapter);
        let observed = capture.clone();
        let mut fixture = gate_fixture_with_turn_adapter(Arc::new(capture)).await;

        let lutum = fixture.gate.llm.lutum().await;
        let peer_contexts = vec![(builtin::sensory(), "test stub")];
        let identity_memories: Vec<IdentityMemoryRecord> = Vec::new();
        let cx = nuillu_module::ActivateCx::new(
            &peer_contexts,
            &[],
            &identity_memories,
            &[],
            compaction_runtime(&lutum),
            SystemClock.now(),
        );

        let batch = batch_from_candidates(&candidates);
        fixture.gate.activate(&cx, &batch).await.unwrap();

        let inputs = observed.text_inputs();
        assert_eq!(inputs.len(), 1);
        let turns = observed.text_turns();
        assert_eq!(turns.len(), 1);
        assert_eq!(turns[0].config.tool_choice, AdapterToolChoice::Required);
        assert_eq!(turns[0].config.tools.len(), 1);
        assert_eq!(turns[0].config.tools[0].name, "rank_awareness_candidates");
        let request_items = inputs[0].items();
        let user_messages = message_texts_with_role(request_items, InputMessageRole::User);
        assert_eq!(user_messages.len(), 1);
        let candidate_input = user_messages[0];
        assert!(candidate_input.contains(NEW_CANDIDATE_HEADER));
        assert_candidate_decision_instruction(candidate_input);
        assert!(candidate_input.contains(&format!("[{}] sensory detail A", candidates[0].id)));
        assert!(candidate_input.contains("sensory detail A"));
        assert!(candidate_input.contains(&format!("[{}] sensory detail B", candidates[1].id)));
        assert!(candidate_input.contains("sensory detail B"));
        assert!(candidate_input.contains(&format!("[{}] sensory detail C", candidates[2].id)));
        assert!(candidate_input.contains("sensory detail C"));
        assert!(!candidate_input.contains("1. sensory detail A"));
        assert!(!candidate_input.contains("2. sensory detail B"));
        assert!(!candidate_input.contains("3. sensory detail C"));
        assert!(!candidate_input.contains("memo"));
        assert!(!candidate_input.contains("held-in-mind notes"));
        assert!(!candidate_input.contains("working notes"));
        assert!(!candidate_input.contains(
            "Cognition-gate context for deciding what should enter conscious cognition now"
        ));
        let developer_messages =
            message_texts_with_role(request_items, InputMessageRole::Developer);
        assert!(developer_messages.is_empty());
        assert!(!has_message_with_role_containing(
            request_items,
            InputMessageRole::System,
            "Cognition-gate context for deciding what should enter conscious cognition now"
        ));
        assert!(!has_message_with_role_containing(
            request_items,
            InputMessageRole::System,
            "Memory trace inventory"
        ));
        assert!(!has_message_with_role_containing(
            request_items,
            InputMessageRole::System,
            "Current attention guidance"
        ));
        assert!(!has_message_with_role_containing(
            request_items,
            InputMessageRole::System,
            "Sense of time"
        ));
        assert!(!has_message_with_role_containing(
            request_items,
            InputMessageRole::System,
            "sensory detail A"
        ));

        let items = fixture.gate.session.input().items().to_vec();
        let ModelInputItem::Message { role, content } = &items[0] else {
            panic!("expected persistent system prompt");
        };
        assert_eq!(role, &InputMessageRole::System);
        let [MessageContent::Text(system)] = content.as_slice() else {
            panic!("expected system text");
        };
        assert!(system.contains(SYSTEM_PROMPT));
        assert!(items.iter().any(|item| matches!(
            item,
            ModelInputItem::Message {
                role: InputMessageRole::User,
                content,
            } if matches!(
                content.as_slice(),
                [MessageContent::Text(text)]
                    if text.contains(NEW_CANDIDATE_HEADER)
                        && text.contains(CANDIDATE_DECISION_INSTRUCTION)
                        && text.contains("sensory detail A")
                        && text.contains("sensory detail B")
                        && text.contains("sensory detail C")
                        && !text.contains("memo")
                        && !text.contains("held-in-mind notes")
                        && !text.contains("working notes")
            )
        )));

        assert_eq!(
            fixture.gate.session.list_turns().count(),
            1,
            "explicit rank_awareness_candidates decision must persist a tool turn"
        );
        assert!(has_message_with_role_containing(
            &items,
            InputMessageRole::User,
            "sensory detail A"
        ));
        assert!(has_message_with_role_containing(
            &items,
            InputMessageRole::User,
            "sensory detail B"
        ));
        assert!(has_message_with_role_containing(
            &items,
            InputMessageRole::User,
            "sensory detail C"
        ));
        assert!(!has_message_with_role_containing(
            &items,
            InputMessageRole::System,
            "sensory detail A"
        ));
        assert!(!has_message_with_role_containing(
            &items,
            InputMessageRole::System,
            "sensory detail B"
        ));
        assert!(!has_message_with_role_containing(
            &items,
            InputMessageRole::System,
            "sensory detail C"
        ));

        let entries = fixture
            .blackboard
            .read(|bb| bb.cognition_log().entries().to_vec())
            .await;
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].text, "sensory detail B");
    }

    #[tokio::test(flavor = "current_thread")]
    async fn single_cognitive_memo_promotes_without_llm() {
        let capture = CapturingAdapter::new(MockLlmAdapter::new());
        let observed = capture.clone();
        let mut fixture = gate_fixture_with_turn_adapter(Arc::new(capture)).await;
        fixture
            .source_memo
            .write_cognitive("Peer said hello to me.")
            .await;

        let lutum = fixture.gate.llm.lutum().await;
        let identity_memories: Vec<IdentityMemoryRecord> = Vec::new();
        let cx = nuillu_module::ActivateCx::new(
            &[],
            &[],
            &identity_memories,
            &[],
            compaction_runtime(&lutum),
            SystemClock.now(),
        );

        let batch = next_gate_batch(&mut fixture).await;
        fixture.gate.activate(&cx, &batch).await.unwrap();

        assert!(observed.text_inputs().is_empty());
        let entries = fixture
            .blackboard
            .read(|bb| bb.cognition_log().entries().to_vec())
            .await;
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].text, "Peer said hello to me.");
    }

    #[tokio::test(flavor = "current_thread")]
    async fn two_cognitive_memos_promote_without_llm_as_separate_entries() {
        let capture = CapturingAdapter::new(MockLlmAdapter::new());
        let observed = capture.clone();
        let mut fixture = gate_fixture_with_turn_adapter(Arc::new(capture)).await;
        fixture
            .source_memo
            .write_cognitive("Ryo said, \"Nui, hello.\"")
            .await;
        fixture
            .source_memo
            .write_cognitive("Ryo asked what Nui is thinking.")
            .await;

        let lutum = fixture.gate.llm.lutum().await;
        let identity_memories: Vec<IdentityMemoryRecord> = Vec::new();
        let cx = nuillu_module::ActivateCx::new(
            &[],
            &[],
            &identity_memories,
            &[],
            compaction_runtime(&lutum),
            SystemClock.now(),
        );

        let batch = next_gate_batch(&mut fixture).await;
        fixture.gate.activate(&cx, &batch).await.unwrap();

        assert!(observed.text_inputs().is_empty());
        let entries = fixture
            .blackboard
            .read(|bb| bb.cognition_log().entries().to_vec())
            .await;
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].text, "Ryo said, \"Nui, hello.\"");
        assert_eq!(entries[1].text, "Ryo asked what Nui is thinking.");
    }

    #[tokio::test(flavor = "current_thread")]
    async fn empty_and_duplicate_candidates_are_cleaned_before_activation() {
        let capture = CapturingAdapter::new(MockLlmAdapter::new());
        let observed = capture.clone();
        let mut fixture = gate_fixture_with_turn_adapter(Arc::new(capture)).await;
        let sensory = nuillu_types::ModuleInstanceId::new(
            builtin::sensory(),
            nuillu_types::ReplicaIndex::ZERO,
        );
        let now = SystemClock.now();
        let empty = fixture
            .blackboard
            .update_cognitive_memo(sensory.clone(), "   ".into(), now)
            .await;
        let first = fixture
            .blackboard
            .update_cognitive_memo(sensory.clone(), "same candidate".into(), now)
            .await;
        let duplicate = fixture
            .blackboard
            .update_cognitive_memo(sensory, "same candidate".into(), now)
            .await;

        let lutum = fixture.gate.llm.lutum().await;
        let identity_memories: Vec<IdentityMemoryRecord> = Vec::new();
        let cx = nuillu_module::ActivateCx::new(
            &[],
            &[],
            &identity_memories,
            &[],
            compaction_runtime(&lutum),
            SystemClock.now(),
        );

        let batch = CognitionGateBatch {
            memo_logs: vec![empty, first, duplicate],
        };
        fixture.gate.activate(&cx, &batch).await.unwrap();

        assert!(observed.text_inputs().is_empty());
        let entries = fixture
            .blackboard
            .read(|bb| bb.cognition_log().entries().to_vec())
            .await;
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].text, "same candidate");
    }

    #[tokio::test(flavor = "current_thread")]
    async fn non_cognitive_memo_update_does_not_form_gate_batch() {
        let mut fixture = gate_fixture_with_adapter(MockLlmAdapter::new()).await;
        fixture.source_memo.write("internal bookkeeping").await;

        let timed_out = tokio::time::timeout(Duration::from_millis(10), fixture.gate.next_batch())
            .await
            .is_err();
        assert!(timed_out, "non-cognitive memo should not produce a batch");

        fixture
            .source_memo
            .write_cognitive("cognitive evidence")
            .await;
        let batch = next_gate_batch(&mut fixture).await;
        assert_eq!(batch.memo_logs.len(), 1);
        assert_eq!(batch.memo_logs[0].content, "cognitive evidence");
        assert!(batch.memo_logs[0].cognitive);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn silent_finish_without_tool_call_errors() {
        let adapter = MockLlmAdapter::new().with_text_scenario(silent_scenario(1));
        let mut fixture = gate_fixture_with_adapter(adapter).await;

        let lutum = fixture.gate.llm.lutum().await;
        let identity_memories = Vec::new();
        let cx = nuillu_module::ActivateCx::new(
            &[],
            &[],
            &identity_memories,
            &[],
            compaction_runtime(&lutum),
            SystemClock.now(),
        );
        let err = fixture
            .gate
            .run_selection_turn(&lutum, &cx, &test_candidates(&["candidate A"]))
            .await
            .unwrap_err();

        assert!(
            err.to_string().contains("required tool")
                || err.to_string().contains("decision turn failed")
        );
    }

    #[test]
    fn ranked_candidates_rejects_unknown_and_duplicate_ids() {
        let candidates = test_candidates(&["first", "second"]);
        let empty = ranked_candidate_records(
            &RankAwarenessCandidatesArgs { labels: Vec::new() },
            &candidates,
        )
        .unwrap_err();
        assert!(empty.to_string().contains("empty labels"));

        let unknown = ranked_candidate_records(
            &RankAwarenessCandidatesArgs {
                labels: vec!["UNKNOWNX".to_owned()],
            },
            &candidates,
        )
        .unwrap_err();
        assert!(unknown.to_string().contains("outside candidate IDs"));

        let no_candidates = ranked_candidate_records(
            &RankAwarenessCandidatesArgs {
                labels: vec![candidates[0].id.clone()],
            },
            &[],
        )
        .unwrap_err();
        assert!(no_candidates.to_string().contains("outside candidate IDs"));

        let duplicate = ranked_candidate_records(
            &RankAwarenessCandidatesArgs {
                labels: vec![candidates[0].id.clone(), format!("[{}]", candidates[0].id)],
            },
            &candidates,
        )
        .unwrap_err();
        assert!(duplicate.to_string().contains("duplicate"));
    }

    #[test]
    fn ranked_candidates_maps_first_two_ids_to_candidate_records() {
        let candidates = test_candidates(&["first", "second", "third"]);
        let selected = ranked_candidate_records(
            &RankAwarenessCandidatesArgs {
                labels: vec![
                    candidates[1].id.clone(),
                    candidates[0].id.clone(),
                    candidates[2].id.clone(),
                ],
            },
            &candidates,
        )
        .unwrap();
        assert_eq!(
            selected
                .iter()
                .map(|record| record.content.as_str())
                .collect::<Vec<_>>(),
            vec!["second", "first"]
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn ranked_top_appends_matching_memo_after_single_selection_turn() {
        let candidates =
            test_candidates(&["candidate A", "candidate B matters now", "candidate C"]);
        let adapter = MockLlmAdapter::new().with_text_scenario(rank_awareness_candidates_scenario(
            [candidates[1].id.clone()],
            1,
        ));
        let capture = CapturingAdapter::new(adapter);
        let observed = capture.clone();
        let mut fixture = gate_fixture_with_turn_adapter(Arc::new(capture)).await;

        let lutum = fixture.gate.llm.lutum().await;
        let identity_memories = Vec::new();
        let cx = nuillu_module::ActivateCx::new(
            &[],
            &[],
            &identity_memories,
            &[],
            compaction_runtime(&lutum),
            SystemClock.now(),
        );
        let batch = batch_from_candidates(&candidates);
        fixture.gate.activate(&cx, &batch).await.unwrap();

        assert_eq!(observed.text_inputs().len(), 1);
        let records = fixture
            .blackboard
            .read(|bb| bb.unread_cognition_log_entries(None))
            .await;
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].entry.text, "candidate B matters now");
        assert_eq!(records[0].entry.origin.owner.module, builtin::sensory());
        assert_eq!(records[0].entry.origin.memo_index, Some(1));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn selection_copies_memo_without_light_rewrite() {
        let candidates = test_candidates(&[
            "Ryo said, \"Nui, hello.\"",
            "Ryo asked Nui to say what Nui is thinking.",
            "background wind shifted.",
        ]);
        let adapter = MockLlmAdapter::new().with_text_scenario(rank_awareness_candidates_scenario(
            [candidates[1].id.clone()],
            1,
        ));
        let capture = CapturingAdapter::new(adapter);
        let observed = capture.clone();
        let mut fixture = gate_fixture_with_turn_adapter(Arc::new(capture)).await;

        let lutum = fixture.gate.llm.lutum().await;
        let identity_memories = Vec::new();
        let cx = nuillu_module::ActivateCx::new(
            &[],
            &[],
            &identity_memories,
            &[],
            compaction_runtime(&lutum),
            SystemClock.now(),
        );
        let batch = batch_from_candidates(&candidates);
        fixture.gate.activate(&cx, &batch).await.unwrap();

        assert_eq!(observed.text_inputs().len(), 1);

        let entries = fixture
            .blackboard
            .read(|bb| bb.cognition_log().entries().to_vec())
            .await;
        assert_eq!(entries.len(), 1);
        assert_eq!(
            entries[0].text,
            "Ryo asked Nui to say what Nui is thinking."
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn ranked_top_appends_second_selected_memo() {
        let candidates = test_candidates(&["candidate A", "candidate B", "candidate C"]);
        let adapter = MockLlmAdapter::new().with_text_scenario(rank_awareness_candidates_scenario(
            [candidates[1].id.clone(), candidates[0].id.clone()],
            1,
        ));
        let capture = CapturingAdapter::new(adapter);
        let observed = capture.clone();
        let mut fixture = gate_fixture_with_turn_adapter(Arc::new(capture)).await;

        let lutum = fixture.gate.llm.lutum().await;
        let identity_memories = Vec::new();
        let cx = nuillu_module::ActivateCx::new(
            &[],
            &[],
            &identity_memories,
            &[],
            compaction_runtime(&lutum),
            SystemClock.now(),
        );
        let batch = batch_from_candidates(&candidates);
        fixture.gate.activate(&cx, &batch).await.unwrap();
        assert!(fixture.gate.pending_unread_memos.is_empty());
        assert_eq!(observed.text_inputs().len(), 1);

        let entries = fixture
            .blackboard
            .read(|bb| bb.cognition_log().entries().to_vec())
            .await;
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].text, "candidate B");
        assert_eq!(entries[1].text, "candidate A");
    }

    #[tokio::test(flavor = "current_thread")]
    async fn selection_copies_memory_evidence_with_structured_provenance() {
        let memory_text = concat!(
            "Retrieved memory evidence:\n",
            "Ryo's full name is Ryo Hihaheho\n\n",
            r#"[stored affect: arousal 0.00, valence 0.00, emotion none] <today occurred-at="2026-06-20T04:40:47Z">"#,
            "\nHeard from Ryo that his full name is Ryo Hihaheho.\n",
            "</today>"
        );
        let candidates = test_candidates_for_module(
            builtin::query_memory(),
            &[
                "stale greeting already handled",
                memory_text,
                "I am speaking to Ryo about prior context.",
            ],
        );
        let adapter = MockLlmAdapter::new().with_text_scenario(rank_awareness_candidates_scenario(
            [candidates[1].id.clone()],
            1,
        ));
        let capture = CapturingAdapter::new(adapter);
        let observed = capture.clone();
        let mut fixture = gate_fixture_with_turn_adapter(Arc::new(capture)).await;

        let lutum = fixture.gate.llm.lutum().await;
        let identity_memories = Vec::new();
        let cx = nuillu_module::ActivateCx::new(
            &[],
            &[],
            &identity_memories,
            &[],
            compaction_runtime(&lutum),
            SystemClock.now(),
        );
        let batch = batch_from_candidates(&candidates);
        fixture.gate.activate(&cx, &batch).await.unwrap();

        assert_eq!(observed.text_inputs().len(), 1);

        let entries = fixture
            .blackboard
            .read(|bb| bb.cognition_log().entries().to_vec())
            .await;
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].text, memory_text);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn activation_rolls_back_failed_candidate_turn_and_retries_pending_candidates() {
        let candidates =
            test_candidates(&["sensory detail A", "sensory detail B", "sensory detail C"]);
        let adapter = MockLlmAdapter::new()
            .with_text_scenario(MockTextScenario::start_error(MockError::Synthetic {
                message: "synthetic failure".into(),
            }))
            .with_text_scenario(rank_awareness_candidates_scenario(
                [candidates[1].id.clone()],
                1,
            ));
        let capture = CapturingAdapter::new(adapter);
        let observed = capture.clone();
        let mut fixture = gate_fixture_with_turn_adapter(Arc::new(capture)).await;

        let lutum = fixture.gate.llm.lutum().await;
        let peer_contexts = vec![(builtin::sensory(), "test stub")];
        let identity_memories: Vec<IdentityMemoryRecord> = Vec::new();
        let cx = nuillu_module::ActivateCx::new(
            &peer_contexts,
            &[],
            &identity_memories,
            &[],
            compaction_runtime(&lutum),
            SystemClock.now(),
        );

        let batch = batch_from_candidates(&candidates);
        let err = fixture.gate.activate(&cx, &batch).await.unwrap_err();
        assert!(
            err.to_string().contains("decision turn failed")
                || err.to_string().contains("synthetic")
        );
        assert_eq!(fixture.gate.pending_unread_memos.len(), 3);

        let items = fixture.gate.session.input().items().to_vec();
        assert!(!has_message_with_role_containing(
            &items,
            InputMessageRole::System,
            "Earlier candidate context already considered"
        ));
        assert!(!has_message_with_role_containing(
            &items,
            InputMessageRole::System,
            "sensory detail A"
        ));
        assert!(!has_message_with_role_containing(
            &items,
            InputMessageRole::User,
            "sensory detail A"
        ));
        assert!(!has_message_with_role_containing(
            &items,
            InputMessageRole::User,
            "sensory detail B"
        ));
        assert!(!has_message_with_role_containing(
            &items,
            InputMessageRole::User,
            "sensory detail C"
        ));

        fixture.gate.activate(&cx, &batch).await.unwrap();
        assert!(fixture.gate.pending_unread_memos.is_empty());

        let inputs = observed.text_inputs();
        assert_eq!(inputs.len(), 2);
        let retried_candidates = latest_candidate_user_message(inputs[1].items());
        assert!(retried_candidates.contains(NEW_CANDIDATE_HEADER));
        assert_candidate_decision_instruction(retried_candidates);
        assert!(retried_candidates.contains("sensory detail A"));
        assert!(retried_candidates.contains("sensory detail B"));
        assert!(retried_candidates.contains("sensory detail C"));
        assert!(!retried_candidates.contains("memo"));
        assert!(!retried_candidates.contains("held-in-mind notes"));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn failed_activations_accumulate_pending_candidates_until_success() {
        let candidates = test_candidates(&[
            "sensory detail A",
            "sensory detail B",
            "sensory detail C",
            "sensory detail D",
        ]);
        let adapter = MockLlmAdapter::new()
            .with_text_scenario(MockTextScenario::start_error(MockError::Synthetic {
                message: "synthetic failure A".into(),
            }))
            .with_text_scenario(MockTextScenario::start_error(MockError::Synthetic {
                message: "synthetic failure B".into(),
            }))
            .with_text_scenario(rank_awareness_candidates_scenario(
                [candidates[2].id.clone()],
                1,
            ));
        let capture = CapturingAdapter::new(adapter);
        let observed = capture.clone();
        let mut fixture = gate_fixture_with_turn_adapter(Arc::new(capture)).await;

        let lutum = fixture.gate.llm.lutum().await;
        let peer_contexts = vec![(builtin::sensory(), "test stub")];
        let identity_memories: Vec<IdentityMemoryRecord> = Vec::new();
        let cx = nuillu_module::ActivateCx::new(
            &peer_contexts,
            &[],
            &identity_memories,
            &[],
            compaction_runtime(&lutum),
            SystemClock.now(),
        );

        let first_batch = batch_from_candidates(&candidates[..3]);
        fixture.gate.activate(&cx, &first_batch).await.unwrap_err();
        assert_eq!(fixture.gate.pending_unread_memos.len(), 3);

        let second_batch = batch_from_candidates(&candidates[3..4]);
        fixture.gate.activate(&cx, &second_batch).await.unwrap_err();
        assert_eq!(fixture.gate.pending_unread_memos.len(), 4);

        fixture.gate.activate(&cx, &second_batch).await.unwrap();
        assert!(fixture.gate.pending_unread_memos.is_empty());

        let inputs = observed.text_inputs();
        assert_eq!(inputs.len(), 3);
        let accumulated_candidates = latest_candidate_user_message(inputs[2].items());
        assert_candidate_decision_instruction(accumulated_candidates);
        assert!(accumulated_candidates.contains("sensory detail A"));
        assert!(accumulated_candidates.contains("sensory detail B"));
        assert!(accumulated_candidates.contains("sensory detail C"));
        assert!(accumulated_candidates.contains("sensory detail D"));

        let fresh = test_candidates(&["sensory detail E", "sensory detail F"]);
        let third_batch = batch_from_candidates(&fresh);
        fixture.gate.activate(&cx, &third_batch).await.unwrap();

        let inputs = observed.text_inputs();
        assert_eq!(inputs.len(), 3, "two fresh candidates should bypass LLM");
        let entries = fixture
            .blackboard
            .read(|bb| bb.cognition_log().entries().to_vec())
            .await;
        assert_eq!(entries.len(), 3);
        assert_eq!(entries[1].text, "sensory detail E");
        assert_eq!(entries[2].text, "sensory detail F");
    }

    #[tokio::test(flavor = "current_thread")]
    async fn second_activation_sends_prior_session_history_to_lutum() {
        let first_candidates = test_candidates(&[
            "The agent noticed the north door is blocked.",
            "sensory detail A2",
            "sensory detail A3",
        ]);
        let second_candidates =
            test_candidates(&["sensory detail B", "sensory detail B2", "sensory detail B3"]);
        let adapter = MockLlmAdapter::new()
            .with_text_scenario(rank_awareness_candidates_scenario(
                [first_candidates[0].id.clone()],
                1,
            ))
            .with_text_scenario(rank_awareness_candidates_scenario(
                [second_candidates[0].id.clone()],
                1,
            ));
        let capture = CapturingAdapter::new(adapter);
        let observed = capture.clone();
        let mut fixture = gate_fixture_with_turn_adapter(Arc::new(capture)).await;

        let lutum = fixture.gate.llm.lutum().await;
        let peer_contexts = vec![(builtin::sensory(), "test stub")];
        let identity_memories: Vec<IdentityMemoryRecord> = Vec::new();
        let cx = nuillu_module::ActivateCx::new(
            &peer_contexts,
            &[],
            &identity_memories,
            &[],
            compaction_runtime(&lutum),
            SystemClock.now(),
        );

        let first_batch = batch_from_candidates(&first_candidates);
        fixture.gate.activate(&cx, &first_batch).await.unwrap();
        let second_batch = batch_from_candidates(&second_candidates);
        fixture.gate.activate(&cx, &second_batch).await.unwrap();

        let inputs = observed.text_inputs();
        assert_eq!(inputs.len(), 2);

        let first_items = inputs[0].items();
        let ModelInputItem::Message { role, content } = &first_items[0] else {
            panic!("expected first input system prompt");
        };
        assert_eq!(role, &InputMessageRole::System);
        let [MessageContent::Text(system)] = content.as_slice() else {
            panic!("expected first input system prompt text");
        };
        assert!(system.contains(SYSTEM_PROMPT));
        assert!(!system.contains("Other modules"));
        assert!(!system.contains("test stub"));

        let first_user_messages = message_texts_with_role(first_items, InputMessageRole::User);
        assert_eq!(first_user_messages.len(), 1);
        assert!(first_user_messages[0].contains(NEW_CANDIDATE_HEADER));
        assert_candidate_decision_instruction(first_user_messages[0]);
        assert!(first_user_messages[0].contains("The agent noticed the north door is blocked."));
        assert!(!first_user_messages[0].contains("memo"));
        assert!(!first_user_messages[0].contains(
            "Cognition-gate context for deciding what should enter conscious cognition now"
        ));
        assert!(message_texts_with_role(first_items, InputMessageRole::Developer).is_empty());
        assert!(!has_message_with_role_containing(
            first_items,
            InputMessageRole::System,
            "Cognition-gate context for deciding what should enter conscious cognition now"
        ));
        assert!(!has_message_with_role_containing(
            first_items,
            InputMessageRole::System,
            "Memory trace inventory"
        ));
        assert!(!has_message_with_role_containing(
            first_items,
            InputMessageRole::System,
            "Current attention guidance"
        ));
        assert!(!has_message_with_role_containing(
            first_items,
            InputMessageRole::System,
            "Sense of time"
        ));
        assert!(!has_message_with_role_containing(
            first_items,
            InputMessageRole::System,
            "The agent noticed the north door is blocked."
        ));

        let second_items = inputs[1].items();
        let ModelInputItem::Message { role, content } = &second_items[0] else {
            panic!("expected second input system prompt");
        };
        assert_eq!(role, &InputMessageRole::System);
        let [MessageContent::Text(system)] = content.as_slice() else {
            panic!("expected second input system prompt text");
        };
        assert!(system.contains(SYSTEM_PROMPT));
        assert!(!system.contains("Other modules"));
        assert!(!system.contains("test stub"));
        assert!(!has_message_with_role_containing(
            second_items,
            InputMessageRole::System,
            "sensory detail A"
        ));
        let second_user_messages = message_texts_with_role(second_items, InputMessageRole::User);
        assert_eq!(second_user_messages.len(), 2);
        assert!(second_user_messages[0].contains(NEW_CANDIDATE_HEADER));
        assert_candidate_decision_instruction(second_user_messages[0]);
        assert!(second_user_messages[0].contains("sensory detail A"));
        assert!(!second_user_messages[1].contains("What you are currently thinking at"));
        assert!(second_user_messages[1].contains(NEW_CANDIDATE_HEADER));
        assert_candidate_decision_instruction(second_user_messages[1]);
        assert!(second_user_messages[1].contains("sensory detail B"));
        assert!(!second_user_messages[1].contains("memo"));
        assert!(!second_user_messages[1].contains(
            "Cognition-gate context for deciding what should enter conscious cognition now"
        ));
        assert!(message_texts_with_role(second_items, InputMessageRole::Developer).is_empty());
        assert!(!has_message_with_role_containing(
            second_items,
            InputMessageRole::System,
            "Cognition-gate context for deciding what should enter conscious cognition now"
        ));
        assert!(!has_message_with_role_containing(
            second_items,
            InputMessageRole::System,
            "Memory trace inventory"
        ));
        assert!(!has_message_with_role_containing(
            second_items,
            InputMessageRole::System,
            "Current attention guidance"
        ));
        assert!(!has_message_with_role_containing(
            second_items,
            InputMessageRole::System,
            "Sense of time"
        ));
        assert!(!has_message_with_role_containing(
            second_items,
            InputMessageRole::System,
            "sensory detail B"
        ));
        assert!(!second_items.iter().any(|item| matches!(
            item,
            ModelInputItem::Assistant(lutum::AssistantInputItem::Text(text))
                if text.contains("What you are currently thinking at")
        )));
        assert!(
            inputs[1]
                .items()
                .iter()
                .any(|item| matches!(item, ModelInputItem::Turn(_)))
        );
        assert!(second_items.iter().any(|item| {
            let ModelInputItem::Turn(turn) = item else {
                return false;
            };
            (0..turn.item_count()).any(|index| {
                turn.item_at(index)
                    .and_then(|item| item.as_tool_call())
                    .is_some_and(|call| call.name.as_str() == "rank_awareness_candidates")
            })
        }));

        let session_after_second = fixture.gate.session.input().items().to_vec();
        assert!(!has_message_with_role_containing(
            &session_after_second,
            InputMessageRole::System,
            "The agent noticed the north door is blocked."
        ));
        assert!(!has_message_with_role_containing(
            &session_after_second,
            InputMessageRole::System,
            "sensory detail B"
        ));
        assert!(has_message_with_role_containing(
            &session_after_second,
            InputMessageRole::User,
            "The agent noticed the north door is blocked."
        ));
        assert!(has_message_with_role_containing(
            &session_after_second,
            InputMessageRole::User,
            "sensory detail B"
        ));
        let session_user_messages =
            message_texts_with_role(&session_after_second, InputMessageRole::User);
        assert!(
            !session_user_messages
                .iter()
                .any(|text| text.contains("What you are currently thinking at"))
        );
        assert!(!session_after_second.iter().any(|item| matches!(
            item,
            ModelInputItem::Assistant(lutum::AssistantInputItem::Text(text))
                if text.contains("What you are currently thinking at")
        )));
        assert!(!session_after_second.iter().any(|item| matches!(
            item,
            ModelInputItem::Message { content, .. }
                if matches!(
                    content.as_slice(),
                    [MessageContent::Text(text)]
                        if text.contains("Cognition-gate context for deciding what should enter conscious cognition now")
                )
        )));
        assert_eq!(fixture.gate.session.list_turns().count(), 2);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn activation_seeds_identity_memories_in_system_prompt() {
        let candidates = test_candidates(&[
            "identity seed candidate A",
            "identity seed candidate B",
            "identity seed candidate C",
        ]);
        let adapter = MockLlmAdapter::new().with_text_scenario(rank_awareness_candidates_scenario(
            [candidates[1].id.clone()],
            1,
        ));
        let mut fixture = gate_fixture_with_adapter(adapter).await;

        let lutum = fixture.gate.llm.lutum().await;
        let peer_contexts = vec![(builtin::sensory(), "test stub")];
        let identity_memories = vec![IdentityMemoryRecord {
            index: nuillu_types::MemoryIndex::new("identity-1"),
            content: nuillu_types::MemoryContent::new("The agent is named Nuillu."),
            occurred_at: None,
        }];
        let cx = nuillu_module::ActivateCx::new(
            &peer_contexts,
            &[],
            &identity_memories,
            &[],
            compaction_runtime(&lutum),
            SystemClock.now(),
        );

        let batch = batch_from_candidates(&candidates);
        fixture.gate.activate(&cx, &batch).await.unwrap();

        let items = fixture.gate.session.input().items().to_vec();
        let ModelInputItem::Message {
            role: InputMessageRole::System,
            content,
        } = &items[0]
        else {
            panic!("expected leading system prompt");
        };
        let [MessageContent::Text(system)] = content.as_slice() else {
            panic!("expected system prompt text");
        };
        assert!(system.contains("Your identity:"));
        assert!(system.contains("- The agent is named Nuillu."));
        assert!(!system.contains("<self-memory>"));
    }
}
