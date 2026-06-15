use std::collections::BTreeSet;

use anyhow::{Context, Result, anyhow};
use async_trait::async_trait;
use lutum::{Lutum, Session, TextStepOutcomeWithTools, ToolResult, Usage};
use nuillu_blackboard::MemoLogRecord;
use nuillu_module::{
    BlackboardReader, CognitionWriter, LlmAccess, LlmContextWindow, MemoUpdatedInbox, Module,
    SessionAutoCompaction, SessionCompactionConfig, SessionCompactionProtectedPrefix,
    compact_llm_context_text, ensure_persistent_session_seeded, format_faculty_system_prompt,
    push_formatted_cognition_log_batch,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

mod batch;
pub use batch::NextBatch as CognitionGateBatch;

const SYSTEM_PROMPT: &str = r#"You are the cognition-gate module — a selective attention
race boundary for the agent's conscious workspace. The cognition log IS that workspace:
it is what other cognitive modules read when they decide how to think, plan, or speak.
Only the one or two candidate entries that win the current attention race may enter
conscious cognition.

Your job is to judge, given the recent conscious workspace and new cognition candidates,
which candidate facts the conscious mind needs to know in order to act now. Select only
current, load-bearing winners: specific safety constraints, peer-model rules, sensory
details, recalled episodes, body or world facts, direct participant speech, warnings,
questions, requests, and advice needs. A brief greeting or direct question can be a winner
when it changes what the agent should be aware of, say, do, or keep tracking now.

When a participant asks for help, asks a question, warns, or requests advice, preserve who
is asking and what they need answered. If the answer is about another participant, object,
place, or hazard, keep both the listener/requester and the topic so speech can answer the
right being.

There is no no-op decision at this boundary: always select a champion from the presented
candidates. As a soft tie-breaker, prefer candidates that add information not already
present in the current cognition log. Avoid redundant restatements, speculation,
confidence scores, evidence-gap notes, decision rationales, retrieval queries, and any
text that is ABOUT the cognitive process rather than ABOUT the world or the agent's
situation when a more concrete candidate exists. Speech-planning status such as "I am
speaking", "Intended message", or "Already said" is process-talk, not cognition text.
If every candidate is weak, choose the most concrete world, body, memory, or participant
speech fact and preserve it without plumbing.

Mechanical brain plumbing must never reach consciousness. Do not mention modules,
retrieval, queries, ranks, gates, allocation, attention, or this workspace itself. The
conscious mind perceives world, body, peers, and recalled experience — not its own
architecture. In retrieval-style updates, select only remembered content that changes the
current judgment, including the operative rule or fact; omit search terms, match status,
and evidence plumbing.

Voice and form.
Select winner text with select_cognition_winner exactly once; do not finish silently.
Copying a candidate is fine, but concise summarization or synthesis is also fine when it preserves the
load-bearing meaning, speaker/topic/request details, and operative facts without adding
unsupported facts. Reject only lossy summaries that drop those details or add facts not
grounded in the candidates. Do not use final assistant text as an output channel."#;

const COMPACTED_COGNITION_GATE_SESSION_PREFIX: &str = "Compacted cognition-gate session history:";
const COGNITION_CONTEXT_WINDOW: LlmContextWindow = LlmContextWindow::new(12, 600, 4_800);
const SESSION_COMPACTION_FOCUS: &str = r#"Preserve candidate facts, prior gate decisions, promoted
events, rejected candidate events, cognition context, and relevant memory evidence needed for
future cognition-gate decisions."#;
const NEW_CANDIDATE_HEADER: &str = "New cognition candidates:";
const CANDIDATE_DECISION_INSTRUCTION: &str = "Review the candidates above and call exactly one \
tool: select_cognition_winner with the one required champion text and, only if a second current \
winner is also load-bearing, optional_secondary. Do not write assistant text.";

const WINNER_TEXT_CONTEXT_CHARS: usize = 1_200;

pub fn session_auto_compaction() -> SessionAutoCompaction {
    SessionAutoCompaction::new(
        SessionCompactionConfig::default(),
        SessionCompactionProtectedPrefix::LeadingSystemAndIdentitySeed,
        COMPACTED_COGNITION_GATE_SESSION_PREFIX,
        SESSION_COMPACTION_FOCUS,
    )
}

#[lutum::tool_input(name = "select_cognition_winner", output = SelectCognitionWinnerOutput)]
/// Select the one or two candidate-derived texts that won the current cognition race.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct SelectCognitionWinnerArgs {
    /// Required winner text. It may copy, summarize, or synthesize candidates, but must preserve
    /// load-bearing meaning without adding unsupported facts.
    pub champion: String,
    /// Optional second winner text, only when a second candidate also deserves conscious bandwidth.
    pub optional_secondary: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct SelectCognitionWinnerOutput {
    pub accepted: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rejected_reason: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema, lutum::Toolset)]
pub enum CognitionWinnerSelectionTools {
    SelectCognitionWinner(SelectCognitionWinnerArgs),
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct CognitionCandidate {
    source: String,
    written_at: String,
    text: String,
}

pub type CognitionGateSessionCompactionConfig = SessionCompactionConfig;

pub struct CognitionGateModule {
    owner: nuillu_types::ModuleId,
    memo_updates: MemoUpdatedInbox,
    blackboard: BlackboardReader,
    cognition: CognitionWriter,
    llm: LlmAccess,
    session: Session,
    system_prompt: std::sync::OnceLock<String>,
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
            owner: nuillu_types::ModuleId::new(<Self as Module>::id())
                .expect("cognition-gate id is valid"),
            memo_updates,
            blackboard,
            cognition,
            llm,
            session,
            system_prompt: std::sync::OnceLock::new(),
            last_seen_cognition_index: None,
            pending_unread_memos: Vec::new(),
        }
    }

    fn system_prompt(&self, cx: &nuillu_module::ActivateCx<'_>) -> &str {
        self.system_prompt.get_or_init(|| {
            format_faculty_system_prompt(SYSTEM_PROMPT, cx.peer_contexts(), &self.owner)
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

    #[tracing::instrument(skip_all, err(Debug, level = "warn"))]
    async fn activate(
        &mut self,
        cx: &nuillu_module::ActivateCx<'_>,
        batch: &CognitionGateBatch,
    ) -> Result<()> {
        let unread_cognition = self
            .blackboard
            .read(|bb| bb.unread_cognition_log_entries(self.last_seen_cognition_index))
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
            if let Some(index) = unread_cognition.last().map(|record| record.index) {
                self.last_seen_cognition_index = Some(index);
            }
            self.pending_unread_memos.clear();
            return Ok(());
        }
        if candidates.len() <= 2 {
            self.append_winner_texts(candidate_texts(&candidates))
                .await?;
            if let Some(index) = unread_cognition.last().map(|record| record.index) {
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
        let result = self.run_selection_turn(&lutum, cx).await;
        if let Err(error) = result {
            self.session
                .input_mut()
                .items_mut()
                .truncate(session_len_before_activation);
            cx.compact_and_save(&mut self.session, Usage::zero())
                .await?;
            return Err(error);
        }
        if let Some(index) = unread_cognition.last().map(|record| record.index) {
            self.last_seen_cognition_index = Some(index);
        }
        self.pending_unread_memos.clear();
        Ok(())
    }

    async fn run_selection_turn(
        &mut self,
        lutum: &Lutum,
        cx: &nuillu_module::ActivateCx<'_>,
    ) -> Result<()> {
        let outcome = self
            .session
            .text_turn()
            .tools::<CognitionWinnerSelectionTools>()
            .available_tools([CognitionWinnerSelectionToolsSelector::SelectCognitionWinner])
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
                let CognitionWinnerSelectionToolsCall::SelectCognitionWinner(call) = call;
                let selected_texts =
                    self.resolve_selected_winners(&call.input)
                        .map_err(|error| {
                            cx.warn(format!("cognition-gate activation failed: {error}"));
                            error
                        })?;
                let results: Vec<ToolResult> = vec![
                    call.complete(SelectCognitionWinnerOutput {
                        accepted: true,
                        rejected_reason: None,
                    })
                    .context("complete select_cognition_winner tool call")?,
                ];
                round
                    .commit(&mut self.session, results)
                    .context("commit cognition-gate tool round")?;
                self.append_winner_texts(selected_texts).await?;
                cx.compact_and_save(&mut self.session, usage).await?;
                Ok(())
            }
        }
    }

    fn resolve_selected_winners(&self, args: &SelectCognitionWinnerArgs) -> Result<Vec<String>> {
        selected_winner_texts(args)
    }

    async fn append_winner_texts(&self, texts: Vec<String>) -> Result<()> {
        let entry = join_winner_texts(&texts)?;
        ensure_plain_cognition_text(&entry)?;
        self.cognition.append(entry).await;
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
        "tool turn produced no single winner selection: tool_calls=[{tool_names}]; expected \
         select_cognition_winner"
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
    let mut records = records
        .iter()
        .filter(|record| !record.content.trim().is_empty())
        .collect::<Vec<_>>();
    records.sort_by(|left, right| {
        left.written_at
            .cmp(&right.written_at)
            .then_with(|| left.owner.module.as_str().cmp(right.owner.module.as_str()))
            .then_with(|| left.owner.replica.cmp(&right.owner.replica))
            .then_with(|| left.index.cmp(&right.index))
    });

    let mut seen = BTreeSet::new();
    let mut candidates = Vec::new();
    for record in records {
        let text = record.content.trim().to_owned();
        if !seen.insert(text.clone()) {
            continue;
        }
        candidates.push(CognitionCandidate {
            source: record.owner.module.as_str().to_owned(),
            written_at: record.written_at.to_rfc3339(),
            text,
        });
    }
    candidates
}

fn candidate_texts(candidates: &[CognitionCandidate]) -> Vec<String> {
    candidates
        .iter()
        .map(|candidate| candidate.text.clone())
        .collect()
}

fn format_candidate_selection_prompt(candidates: &[CognitionCandidate]) -> String {
    let mut out = NEW_CANDIDATE_HEADER.to_owned();
    for (index, candidate) in candidates.iter().enumerate() {
        out.push_str("\n- candidate ");
        out.push_str(&(index + 1).to_string());
        out.push_str(" from ");
        out.push_str(&candidate.source);
        out.push_str(" at ");
        out.push_str(&candidate.written_at);
        out.push_str(": ");
        out.push_str(&compact_llm_context_text(
            &candidate.text,
            WINNER_TEXT_CONTEXT_CHARS,
        ));
    }
    out.push_str("\n\n");
    out.push_str(CANDIDATE_DECISION_INSTRUCTION);
    out
}

fn selected_winner_texts(args: &SelectCognitionWinnerArgs) -> Result<Vec<String>> {
    let champion = normalize_selected_text(&args.champion);
    if champion.is_empty() {
        anyhow::bail!("select_cognition_winner rejected empty champion");
    }
    ensure_plain_cognition_text(&champion)?;

    let mut selected = vec![champion];
    if let Some(secondary) = &args.optional_secondary {
        let secondary = normalize_selected_text(secondary);
        if secondary.is_empty() {
            anyhow::bail!("select_cognition_winner rejected empty optional_secondary");
        }
        ensure_plain_cognition_text(&secondary)?;
        if selected.iter().any(|text| text == &secondary) {
            anyhow::bail!("select_cognition_winner rejected duplicate winners");
        }
        selected.push(secondary);
    }
    Ok(selected)
}

fn normalize_selected_text(text: &str) -> String {
    text.trim().to_owned()
}

fn join_winner_texts(texts: &[String]) -> Result<String> {
    if texts.is_empty() {
        anyhow::bail!("no cognition winner text to append");
    }
    if texts.len() > 2 {
        anyhow::bail!("too many cognition winner texts");
    }
    let mut seen = BTreeSet::new();
    for text in texts {
        if text.trim().is_empty() {
            anyhow::bail!("empty cognition winner text");
        }
        if !seen.insert(text.trim()) {
            anyhow::bail!("duplicate cognition winner text");
        }
    }
    Ok(texts
        .iter()
        .map(|text| text.trim())
        .collect::<Vec<_>>()
        .join("\n"))
}

fn ensure_plain_cognition_text(text: &str) -> Result<()> {
    let text = text.trim();
    if contains_xml_like_tag(text) || looks_like_json_payload(text) {
        anyhow::bail!("cognition-gate rejected structured-output cognition text");
    }
    Ok(())
}

fn contains_xml_like_tag(text: &str) -> bool {
    let bytes = text.as_bytes();
    let mut index = 0;
    while let Some(offset) = text[index..].find('<') {
        let start = index + offset;
        let mut cursor = start + 1;
        if cursor >= bytes.len() {
            return false;
        }
        if matches!(bytes[cursor], b'!' | b'?') {
            if text[start..].contains('>') {
                return true;
            }
            index = start + 1;
            continue;
        }
        if bytes[cursor] == b'/' {
            cursor += 1;
        }
        if cursor >= bytes.len() || !is_xml_name_start(bytes[cursor]) {
            index = start + 1;
            continue;
        }
        cursor += 1;
        while cursor < bytes.len() && is_xml_name_char(bytes[cursor]) {
            cursor += 1;
        }
        while cursor < bytes.len() && !matches!(bytes[cursor], b'<' | b'\n' | b'\r' | b'>') {
            cursor += 1;
        }
        if cursor < bytes.len() && bytes[cursor] == b'>' {
            return true;
        }
        index = start + 1;
    }
    false
}

fn is_xml_name_start(byte: u8) -> bool {
    byte.is_ascii_alphabetic() || byte == b'_'
}

fn is_xml_name_char(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-' | b':' | b'.')
}

fn looks_like_json_payload(text: &str) -> bool {
    let trimmed = text.trim();
    let starts_like_json = matches!(trimmed.as_bytes().first(), Some(b'{' | b'['));
    if starts_like_json && serde_json::from_str::<serde_json::Value>(trimmed).is_ok() {
        return true;
    }
    if starts_like_json && contains_json_key_value_fragment(trimmed) {
        return true;
    }
    let normalized = trimmed.to_ascii_lowercase();
    contains_json_key_value_fragment(trimmed)
        || normalized.contains("\"select_cognition_winner\"")
        || normalized.contains("champion\":")
        || normalized.contains("optional_secondary\":")
        || normalized.contains("\"append_cognition\"")
        || normalized.contains("\"skip_cognition\"")
        || normalized.contains("cognition_text\":")
}

fn contains_json_key_value_fragment(text: &str) -> bool {
    let bytes = text.as_bytes();
    let mut index = 0;
    while index < bytes.len() {
        if bytes[index] != b'"' {
            index += 1;
            continue;
        }
        let key_start = index + 1;
        let Some(relative_key_end) = text[key_start..].find('"') else {
            return false;
        };
        let key_end = key_start + relative_key_end;
        let key = &text[key_start..key_end];
        if key.is_empty()
            || !key
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-'))
        {
            index = key_end + 1;
            continue;
        }
        let mut cursor = key_end + 1;
        while cursor < bytes.len() && bytes[cursor].is_ascii_whitespace() {
            cursor += 1;
        }
        if cursor < bytes.len() && bytes[cursor] == b':' {
            return true;
        }
        index = key_end + 1;
    }
    false
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
                        caps.llm_access(),
                        caps.session("main")
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

    fn select_cognition_winner_scenario(
        champion: &str,
        optional_secondary: Option<&str>,
        input_tokens: u64,
    ) -> MockTextScenario {
        let mut arguments = serde_json::json!({ "champion": champion });
        if let Some(secondary) = optional_secondary {
            arguments["optional_secondary"] = serde_json::json!(secondary);
        }
        tool_scenario(
            "select_cognition_winner",
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

    #[tokio::test(flavor = "current_thread")]
    async fn high_input_selection_compacts_session_prefix() {
        let adapter = MockLlmAdapter::new()
            .with_text_scenario(select_cognition_winner_scenario(
                "candidate A",
                None,
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
        fixture.gate.run_selection_turn(&lutum, &cx).await.unwrap();
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
        let adapter = MockLlmAdapter::new()
            .with_text_scenario(select_cognition_winner_scenario(
                "candidate A",
                None,
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
        fixture.gate.run_selection_turn(&lutum, &cx).await.unwrap();
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
        let adapter = MockLlmAdapter::new().with_text_scenario(select_cognition_winner_scenario(
            "sensory detail B",
            None,
            1,
        ));
        let capture = CapturingAdapter::new(adapter);
        let observed = capture.clone();
        let mut fixture = gate_fixture_with_turn_adapter(Arc::new(capture)).await;
        fixture
            .source_memo
            .write_cognitive("sensory detail A")
            .await;
        fixture
            .source_memo
            .write_cognitive("sensory detail B")
            .await;
        fixture
            .source_memo
            .write_cognitive("sensory detail C")
            .await;

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

        let batch = next_gate_batch(&mut fixture).await;
        fixture.gate.activate(&cx, &batch).await.unwrap();

        let inputs = observed.text_inputs();
        assert_eq!(inputs.len(), 1);
        let turns = observed.text_turns();
        assert_eq!(turns.len(), 1);
        assert_eq!(turns[0].config.tool_choice, AdapterToolChoice::Required);
        assert_eq!(turns[0].config.tools.len(), 1);
        assert_eq!(turns[0].config.tools[0].name, "select_cognition_winner");
        let request_items = inputs[0].items();
        let user_messages = message_texts_with_role(request_items, InputMessageRole::User);
        assert_eq!(user_messages.len(), 1);
        let candidate_input = user_messages[0];
        assert!(candidate_input.contains(NEW_CANDIDATE_HEADER));
        assert_candidate_decision_instruction(candidate_input);
        assert!(candidate_input.contains("- candidate 1"));
        assert!(candidate_input.contains("sensory detail A"));
        assert!(candidate_input.contains("- candidate 2"));
        assert!(candidate_input.contains("sensory detail B"));
        assert!(candidate_input.contains("- candidate 3"));
        assert!(candidate_input.contains("sensory detail C"));
        assert!(!candidate_input.contains("memo"));
        assert!(!candidate_input.contains("Held-in-mind notes"));
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
                        && !text.contains("Held-in-mind notes")
                        && !text.contains("working notes")
            )
        )));

        assert_eq!(
            fixture.gate.session.list_turns().count(),
            1,
            "explicit select_cognition_winner decision must persist a tool turn"
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
    async fn two_cognitive_memos_join_without_llm() {
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
        assert_eq!(entries.len(), 1);
        assert_eq!(
            entries[0].text,
            "Ryo said, \"Nui, hello.\"\nRyo asked what Nui is thinking."
        );
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
            .run_selection_turn(&lutum, &cx)
            .await
            .unwrap_err();

        assert!(
            err.to_string().contains("required tool")
                || err.to_string().contains("decision turn failed")
        );
    }

    #[test]
    fn selected_winner_rejects_empty_and_duplicate_text() {
        let empty = selected_winner_texts(&SelectCognitionWinnerArgs {
            champion: " ".into(),
            optional_secondary: None,
        })
        .unwrap_err();
        assert!(empty.to_string().contains("empty champion"));

        let duplicate = selected_winner_texts(&SelectCognitionWinnerArgs {
            champion: "same".into(),
            optional_secondary: Some(" same ".into()),
        })
        .unwrap_err();
        assert!(duplicate.to_string().contains("duplicate"));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn exact_copied_winner_appends_after_single_selection_turn() {
        let adapter = MockLlmAdapter::new().with_text_scenario(select_cognition_winner_scenario(
            "candidate B matters now",
            None,
            1,
        ));
        let capture = CapturingAdapter::new(adapter);
        let observed = capture.clone();
        let mut fixture = gate_fixture_with_turn_adapter(Arc::new(capture)).await;
        fixture.source_memo.write_cognitive("candidate A").await;
        fixture
            .source_memo
            .write_cognitive("candidate B matters now")
            .await;
        fixture.source_memo.write_cognitive("candidate C").await;

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
        let batch = next_gate_batch(&mut fixture).await;
        fixture.gate.activate(&cx, &batch).await.unwrap();

        assert_eq!(observed.text_inputs().len(), 1);
        let entries = fixture
            .blackboard
            .read(|bb| bb.cognition_log().entries().to_vec())
            .await;
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].text, "candidate B matters now");
    }

    #[tokio::test(flavor = "current_thread")]
    async fn lightly_varied_winner_appends_after_single_selection_turn() {
        let adapter = MockLlmAdapter::new().with_text_scenario(select_cognition_winner_scenario(
            "Ryo asked what Nui is thinking now.",
            None,
            1,
        ));
        let capture = CapturingAdapter::new(adapter);
        let observed = capture.clone();
        let mut fixture = gate_fixture_with_turn_adapter(Arc::new(capture)).await;
        fixture
            .source_memo
            .write_cognitive("Ryo said, \"Nui, hello.\"")
            .await;
        fixture
            .source_memo
            .write_cognitive("Ryo asked Nui to say what Nui is thinking.")
            .await;
        fixture
            .source_memo
            .write_cognitive("background wind shifted.")
            .await;

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
        let batch = next_gate_batch(&mut fixture).await;
        fixture.gate.activate(&cx, &batch).await.unwrap();

        assert_eq!(observed.text_inputs().len(), 1);

        let entries = fixture
            .blackboard
            .read(|bb| bb.cognition_log().entries().to_vec())
            .await;
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].text, "Ryo asked what Nui is thinking now.");
    }

    #[tokio::test(flavor = "current_thread")]
    async fn free_text_winner_appends_after_single_selection_turn() {
        let adapter = MockLlmAdapter::new().with_text_scenario(select_cognition_winner_scenario(
            "selection can lightly rewrite candidate meaning",
            None,
            1,
        ));
        let capture = CapturingAdapter::new(adapter);
        let observed = capture.clone();
        let mut fixture = gate_fixture_with_turn_adapter(Arc::new(capture)).await;
        fixture.source_memo.write_cognitive("candidate A").await;
        fixture.source_memo.write_cognitive("candidate B").await;
        fixture.source_memo.write_cognitive("candidate C").await;

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
        let batch = next_gate_batch(&mut fixture).await;
        fixture.gate.activate(&cx, &batch).await.unwrap();
        assert!(fixture.gate.pending_unread_memos.is_empty());
        assert_eq!(observed.text_inputs().len(), 1);

        let entries = fixture
            .blackboard
            .read(|bb| bb.cognition_log().entries().to_vec())
            .await;
        assert_eq!(entries.len(), 1);
        assert_eq!(
            entries[0].text,
            "selection can lightly rewrite candidate meaning"
        );
    }

    #[test]
    fn plain_cognition_text_allows_work_plan_words() {
        ensure_plain_cognition_text(
            "I need to determine what advice to give Pibi about Koro based on the remembered rules.",
        )
        .unwrap();
    }

    #[test]
    fn plain_cognition_text_rejects_structured_output_artifacts() {
        for text in [
            r#"<tool_call name="select_cognition_winner">{"champion":"Koro should stay in view."}</tool_call>"#,
            r#"{"champion":"Koro should stay in view."}"#,
            r#""champion":"Koro should stay in view.""#,
        ] {
            let err = ensure_plain_cognition_text(text).unwrap_err();
            assert!(err.to_string().contains("structured-output"));
        }
    }

    #[tokio::test(flavor = "current_thread")]
    async fn activation_rolls_back_failed_candidate_turn_and_retries_pending_candidates() {
        let adapter = MockLlmAdapter::new()
            .with_text_scenario(MockTextScenario::start_error(MockError::Synthetic {
                message: "synthetic failure".into(),
            }))
            .with_text_scenario(select_cognition_winner_scenario(
                "sensory detail B",
                None,
                1,
            ));
        let capture = CapturingAdapter::new(adapter);
        let observed = capture.clone();
        let mut fixture = gate_fixture_with_turn_adapter(Arc::new(capture)).await;
        fixture
            .source_memo
            .write_cognitive("sensory detail A")
            .await;
        fixture
            .source_memo
            .write_cognitive("sensory detail B")
            .await;
        fixture
            .source_memo
            .write_cognitive("sensory detail C")
            .await;

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

        let batch = next_gate_batch(&mut fixture).await;
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
        assert!(!retried_candidates.contains("Held-in-mind notes"));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn failed_activations_accumulate_pending_candidates_until_success() {
        let adapter = MockLlmAdapter::new()
            .with_text_scenario(MockTextScenario::start_error(MockError::Synthetic {
                message: "synthetic failure A".into(),
            }))
            .with_text_scenario(MockTextScenario::start_error(MockError::Synthetic {
                message: "synthetic failure B".into(),
            }))
            .with_text_scenario(select_cognition_winner_scenario(
                "sensory detail C",
                None,
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

        fixture
            .source_memo
            .write_cognitive("sensory detail A")
            .await;
        fixture
            .source_memo
            .write_cognitive("sensory detail B")
            .await;
        fixture
            .source_memo
            .write_cognitive("sensory detail C")
            .await;
        let first_batch = next_gate_batch(&mut fixture).await;
        fixture.gate.activate(&cx, &first_batch).await.unwrap_err();
        assert_eq!(fixture.gate.pending_unread_memos.len(), 3);

        fixture
            .source_memo
            .write_cognitive("sensory detail D")
            .await;
        let second_batch = next_gate_batch(&mut fixture).await;
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

        fixture
            .source_memo
            .write_cognitive("sensory detail E")
            .await;
        fixture
            .source_memo
            .write_cognitive("sensory detail F")
            .await;
        let third_batch = next_gate_batch(&mut fixture).await;
        fixture.gate.activate(&cx, &third_batch).await.unwrap();

        let inputs = observed.text_inputs();
        assert_eq!(inputs.len(), 3, "two fresh candidates should bypass LLM");
        let entries = fixture
            .blackboard
            .read(|bb| bb.cognition_log().entries().to_vec())
            .await;
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[1].text, "sensory detail E\nsensory detail F");
    }

    #[tokio::test(flavor = "current_thread")]
    async fn second_activation_sends_prior_session_history_to_lutum() {
        let adapter = MockLlmAdapter::new()
            .with_text_scenario(select_cognition_winner_scenario(
                "The agent noticed the north door is blocked.",
                None,
                1,
            ))
            .with_text_scenario(select_cognition_winner_scenario(
                "sensory detail B",
                None,
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

        fixture
            .source_memo
            .write_cognitive("The agent noticed the north door is blocked.")
            .await;
        fixture
            .source_memo
            .write_cognitive("sensory detail A2")
            .await;
        fixture
            .source_memo
            .write_cognitive("sensory detail A3")
            .await;
        let first_batch = next_gate_batch(&mut fixture).await;
        fixture.gate.activate(&cx, &first_batch).await.unwrap();
        fixture
            .source_memo
            .write_cognitive("sensory detail B")
            .await;
        fixture
            .source_memo
            .write_cognitive("sensory detail B2")
            .await;
        fixture
            .source_memo
            .write_cognitive("sensory detail B3")
            .await;
        let second_batch = next_gate_batch(&mut fixture).await;
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
        assert!(system.contains("You are the cognition-gate module"));

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
        assert!(system.contains("You are the cognition-gate module"));
        assert!(!has_message_with_role_containing(
            second_items,
            InputMessageRole::System,
            "sensory detail A"
        ));
        let second_user_messages = message_texts_with_role(second_items, InputMessageRole::User);
        assert_eq!(second_user_messages.len(), 3);
        assert!(second_user_messages[0].contains(NEW_CANDIDATE_HEADER));
        assert_candidate_decision_instruction(second_user_messages[0]);
        assert!(second_user_messages[0].contains("sensory detail A"));
        assert!(second_user_messages[1].contains("Current cognition log at"));
        assert!(second_user_messages[1].contains("The agent noticed the north door is blocked."));
        assert!(second_user_messages[2].contains(NEW_CANDIDATE_HEADER));
        assert_candidate_decision_instruction(second_user_messages[2]);
        assert!(second_user_messages[2].contains("sensory detail B"));
        assert!(!second_user_messages[2].contains("memo"));
        assert!(!second_user_messages[2].contains(
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
                if text.contains("Current cognition log at")
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
                    .is_some_and(|call| call.name.as_str() == "select_cognition_winner")
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
            session_user_messages
                .iter()
                .any(|text| text.contains("Current cognition log at")
                    && text.contains("The agent noticed the north door is blocked."))
        );
        assert!(!session_after_second.iter().any(|item| matches!(
            item,
            ModelInputItem::Assistant(lutum::AssistantInputItem::Text(text))
                if text.contains("Current cognition log at")
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
        let adapter = MockLlmAdapter::new().with_text_scenario(select_cognition_winner_scenario(
            "identity seed candidate B",
            None,
            1,
        ));
        let mut fixture = gate_fixture_with_adapter(adapter).await;
        let sensory = nuillu_types::ModuleInstanceId::new(
            builtin::sensory(),
            nuillu_types::ReplicaIndex::ZERO,
        );
        let now = SystemClock.now();
        let first = fixture
            .blackboard
            .update_cognitive_memo(sensory.clone(), "identity seed candidate A".into(), now)
            .await;
        let second = fixture
            .blackboard
            .update_cognitive_memo(sensory.clone(), "identity seed candidate B".into(), now)
            .await;
        let third = fixture
            .blackboard
            .update_cognitive_memo(sensory, "identity seed candidate C".into(), now)
            .await;

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

        let batch = CognitionGateBatch {
            memo_logs: vec![first, second, third],
        };
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
        assert!(system.contains("What I already remember about myself at"));
        assert!(system.contains("- The agent is named Nuillu."));
        assert!(!system.contains("<self-memory>"));
    }
}
