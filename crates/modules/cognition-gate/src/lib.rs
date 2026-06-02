use anyhow::{Context, Result, anyhow};
use async_trait::async_trait;
use lutum::{Lutum, Session, TextStepOutcomeWithTools, ToolResult, Usage};
use nuillu_module::{
    AllocationReader, BlackboardReader, CognitionWriter, LlmAccess, LlmContextWindow,
    MemoUpdatedInbox, Module, SessionAutoCompaction, SessionCompactionConfig,
    SessionCompactionProtectedPrefix, TimeDivision, ensure_persistent_session_seeded,
    format_bounded_memo_log_batch, format_current_attention_guidance, format_faculty_system_prompt,
    format_memory_trace_inventory, format_stuckness, format_time_division_guidance,
    memory_rank_counts, push_formatted_cognition_log_batch, push_formatted_memo_log_batch,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

mod batch;

const SYSTEM_PROMPT: &str = r#"You are the cognition-gate module — a selective attention
filter that decides what reaches the agent's conscious workspace. The cognition log IS that
workspace: it is what other cognitive modules read when they decide how to think, plan, or
speak. Every entry you append becomes part of the agent's first-person awareness for this
moment.

Your job is to judge, given the situation and the attention controller's current priority,
what the conscious mind needs to know in order to act. From that judgment two duties follow.

1. Admit everything currently load-bearing.
Read the attention controller's current priority direction together with the recent contents
of the conscious workspace to infer what the agent is presently trying to do, attend to, or
respond to. Then pull into consciousness every subconscious fact that could change that
decision — specific safety constraints, peer-model rules, sensory details, recalled
episodes, body or world facts. Preserve specifics: a concrete actionable rule must enter
the workspace as the rule it actually is, not as a flattened summary that drops the
operative detail. Generic paraphrases are not equivalents.
Never call append_cognition for a mere restatement or tense-shifted paraphrase of what is already
conscious. If you append, the entry must add new load-bearing evidence from the non-conscious inputs
or combine that evidence with the current conscious situation.

When a participant asks for help, asks a question, warns, or requests advice, preserve who is asking
and what they need answered. If the answer is about another participant, object, place, or hazard,
keep both the listener/requester and the topic so speech can answer the right being.

Do not admit future work plans or lookup narration as if they were the answer. If the available
evidence does not answer a participant's question, preserve the missing-evidence state and visible
facts instead of writing first-person plans to check, search, retrieve, or consult internal state.
Do not convert a question, expectation, or waiting state into a hidden-world fact. Preserve observed
absences and unconfirmed status as load-bearing evidence when they constrain what can be said.
When evidence came from retrieved traces or stable knowledge, write the fact or rule directly; do
not prefix it with "I recall", "I remember", "my memory says", or similar source narration.

If multiple subconscious facts converge on the current situation, combine them into one
coherent entry instead of dripping them out across turns. Completeness for the present
judgment matters more than minimum word count.

Reconsider every subconscious update you have seen so far against the *current* situation,
not only the freshest one. A fact you set aside as not-yet-relevant earlier may have become
load-bearing now once the agent's task or attention has shifted. Do not re-promote what is
already in the conscious workspace.

2. Filter everything that would be noise.
Reject redundant restatements of facts already conscious; reject speculation, confidence
scores, evidence-gap notes, decision rationales, retrieval queries, and any text that is
ABOUT the cognitive process rather than ABOUT the world or the agent's situation.

Mechanical brain plumbing must never reach consciousness. Do not mention modules,
retrieval, queries, ranks, gates, allocation, attention, or this workspace itself. The
conscious mind perceives world, body, peers, and recalled experience — not its own
architecture. In retrieval-style updates, promote only remembered content that changes the current
judgment, including the operative rule or fact; omit search terms, match status, and evidence
plumbing.

Voice and form.
Write entries in plain inner-experience prose, as if the agent itself were noticing,
recalling, or realizing. Use the supplied time tags for past observations. If nothing is
currently load-bearing, call skip_cognition exactly once. When load-bearing facts should enter
conscious cognition, call append_cognition exactly once. Do not use final assistant text as an
output channel."#;

const COMPACTED_COGNITION_GATE_SESSION_PREFIX: &str = "Compacted cognition-gate session history:";
const MEMO_CONTEXT_WINDOW: LlmContextWindow = LlmContextWindow::new(8, 1_200, 4_800);
const COGNITION_CONTEXT_WINDOW: LlmContextWindow = LlmContextWindow::new(12, 600, 4_800);
const SESSION_COMPACTION_PROMPT: &str = r#"You compact the cognition-gate module's persistent session history.
Summarize only the prefix transcript you receive. Preserve information that future cognition-gate
decisions need: memo-log facts, prior gate decisions, promoted events, rejected candidate events,
allocation guidance, cognition-log context, and relevant memory metadata. Do not invent facts.
Keep the summary concise, explicit, and faithful. Return plain text only."#;
const ACTIVATION_INPUT: &str = "Decide what, if anything, should enter conscious cognition now.";

pub fn session_auto_compaction() -> SessionAutoCompaction {
    SessionAutoCompaction::new(
        SessionCompactionConfig::default(),
        SessionCompactionProtectedPrefix::LeadingSystemAndIdentitySeed,
        COMPACTED_COGNITION_GATE_SESSION_PREFIX,
        SESSION_COMPACTION_PROMPT,
    )
}

fn format_cognition_gate_context(
    rank_counts: &nuillu_module::MemoryRankCounts,
    allocation: &nuillu_module::ResourceAllocation,
    time_division: &TimeDivision,
    stuckness: Option<&nuillu_module::AgenticDeadlockMarker>,
) -> String {
    let mut sections = vec![
        "Cognition-gate context for deciding what should enter conscious cognition now:".to_owned(),
    ];
    if let Some(section) = format_memory_trace_inventory(rank_counts) {
        sections.push(section);
    }
    if let Some(section) = format_current_attention_guidance(allocation) {
        sections.push(section);
    }
    sections.push(format_time_division_guidance(time_division));
    if let Some(stuckness) = stuckness {
        sections.push(format_stuckness(stuckness));
    }
    sections.join("\n\n")
}

#[lutum::tool_input(name = "append_cognition", output = AppendCognitionOutput)]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct AppendCognitionArgs {
    /// Plain inner-experience text that adds new load-bearing evidence to the current cognition log.
    /// Do not use this field to restate or lightly paraphrase cognition that is already present.
    pub cognition_text: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct AppendCognitionOutput {
    pub appended: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rejected_reason: Option<String>,
}

#[lutum::tool_input(name = "skip_cognition", output = SkipCognitionOutput)]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct SkipCognitionArgs {
    /// Plain reason why no new load-bearing evidence should enter conscious cognition.
    /// Do not use this for facts that should be appended.
    pub reason: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct SkipCognitionOutput {
    pub skipped: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema, lutum::Toolset)]
pub enum CognitionGateTools {
    AppendCognition(AppendCognitionArgs),
    SkipCognition(SkipCognitionArgs),
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum DecisionStatus {
    Applied,
    Rejected,
    Missing,
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum AppendDecision {
    Appended,
    Rejected,
}

pub type CognitionGateSessionCompactionConfig = SessionCompactionConfig;

pub struct CognitionGateModule {
    owner: nuillu_types::ModuleId,
    memo_updates: MemoUpdatedInbox,
    blackboard: BlackboardReader,
    allocation: AllocationReader,
    cognition: CognitionWriter,
    time_division: TimeDivision,
    llm: LlmAccess,
    session: Session,
    system_prompt: std::sync::OnceLock<String>,
    last_seen_cognition_index: Option<u64>,
}

impl CognitionGateModule {
    pub fn new(
        memo_updates: MemoUpdatedInbox,
        blackboard: BlackboardReader,
        allocation: AllocationReader,
        cognition: CognitionWriter,
        time_division: TimeDivision,
        llm: LlmAccess,
        session: Session,
    ) -> Self {
        Self {
            owner: nuillu_types::ModuleId::new(<Self as Module>::id())
                .expect("cognition-gate id is valid"),
            memo_updates,
            blackboard,
            allocation,
            cognition,
            time_division,
            llm,
            session,
            system_prompt: std::sync::OnceLock::new(),
            last_seen_cognition_index: None,
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
    async fn activate(&mut self, cx: &nuillu_module::ActivateCx<'_>) -> Result<()> {
        self.ensure_session_seeded(cx);

        let unread_cognition = self
            .blackboard
            .read(|bb| bb.unread_cognition_log_entries(self.last_seen_cognition_index))
            .await;
        if let Some(index) = unread_cognition.last().map(|record| record.index) {
            self.last_seen_cognition_index = Some(index);
        }
        let unread_memos = self.blackboard.unread_memo_logs().await;
        let (rank_counts, stuckness) = self
            .blackboard
            .read(|bb| {
                (
                    memory_rank_counts(bb.memory_metadata()),
                    bb.agentic_deadlock_marker().cloned(),
                )
            })
            .await;
        let allocation = self.allocation.snapshot().await;
        let context = format_cognition_gate_context(
            &rank_counts,
            &allocation,
            &self.time_division,
            stuckness.as_ref(),
        );
        push_formatted_cognition_log_batch(
            &mut self.session,
            &unread_cognition,
            cx.now(),
            COGNITION_CONTEXT_WINDOW,
        );
        if let Some(memo_notes) =
            format_bounded_memo_log_batch(&unread_memos, cx.now(), MEMO_CONTEXT_WINDOW)
        {
            self.session.push_ephemeral_user(memo_notes);
        }
        self.session.push_ephemeral_system(context);
        self.session.push_ephemeral_developer(ACTIVATION_INPUT);

        let lutum = self.llm.lutum().await;
        let result = self.run_decision_turn(&lutum, cx).await;
        push_formatted_memo_log_batch(
            &mut self.session,
            &unread_memos,
            cx.now(),
            MEMO_CONTEXT_WINDOW,
        );
        cx.compact_and_save(&mut self.session, Usage::zero())
            .await?;
        result?;
        Ok(())
    }

    async fn run_decision_turn(
        &mut self,
        lutum: &Lutum,
        cx: &nuillu_module::ActivateCx<'_>,
    ) -> Result<()> {
        let outcome = self
            .session
            .text_turn()
            .tools::<CognitionGateTools>()
            .available_tools([
                CognitionGateToolsSelector::AppendCognition,
                CognitionGateToolsSelector::SkipCognition,
            ])
            .require_any_tool()
            .max_output_tokens(768)
            .collect(lutum)
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
            TextStepOutcomeWithTools::Finished(result) => {
                cx.compact_and_save(&mut self.session, result.usage).await?;
                if let Some(cognition_text) =
                    salvage_cognition_from_plain_output(&result.assistant_text())
                {
                    let (output, append_decision) = self
                        .append_cognition(AppendCognitionArgs { cognition_text })
                        .await;
                    if append_decision == AppendDecision::Appended {
                        return Ok(());
                    }
                    let detail = output
                        .rejected_reason
                        .unwrap_or_else(|| "append_cognition rejected".into());
                    cx.warn(format!("cognition-gate activation failed: {detail}"));
                    anyhow::bail!("cognition-gate {detail}");
                }
                let detail = "model finished with assistant output but no tool call \
                                (require_any_tool should have prevented this outcome)";
                cx.warn(format!("cognition-gate activation failed: {detail}"));
                anyhow::bail!("cognition-gate finished without required tool call: {detail}");
            }
            TextStepOutcomeWithTools::FinishedNoOutput(result) => {
                cx.compact_and_save(&mut self.session, result.usage).await?;
                let detail = "model finished with no output and no tool call \
                                (require_any_tool should have prevented this outcome)";
                cx.warn(format!("cognition-gate activation failed: {detail}"));
                anyhow::bail!("cognition-gate finished without required tool call: {detail}");
            }
            TextStepOutcomeWithTools::NeedsTools(round) => {
                let usage = round.usage;
                let tool_names = format_tool_call_names(&round.tool_calls);
                let mut results: Vec<ToolResult> = Vec::new();
                nuillu_module::emit_trace_tool_calls(&round.tool_calls);
                if round.tool_calls.is_empty() {
                    let detail = format!(
                        "model returned NeedsTools outcome with empty tool_calls; \
                         expected append_cognition or skip_cognition"
                    );
                    cx.warn(format!("cognition-gate activation failed: {detail}"));
                }
                let mut decision = DecisionStatus::Missing;
                let mut rejection_reason: Option<String> = None;
                for call in round.tool_calls.iter().cloned() {
                    match call {
                        CognitionGateToolsCall::AppendCognition(call) => {
                            let (output, append_decision) =
                                if decision == DecisionStatus::Applied {
                                    (
                                        rejected_append_output(
                                            "another cognition decision was already applied",
                                        ),
                                        AppendDecision::Rejected,
                                    )
                                } else {
                                    self.append_cognition(call.input.clone()).await
                                };
                            match append_decision {
                                AppendDecision::Appended => decision = DecisionStatus::Applied,
                                AppendDecision::Rejected if decision == DecisionStatus::Missing => {
                                    decision = DecisionStatus::Rejected;
                                    rejection_reason = output.rejected_reason.clone();
                                }
                                AppendDecision::Rejected => {}
                            }
                            results.push(
                                call.complete(output)
                                    .context("complete append_cognition tool call")?,
                            );
                        }
                        CognitionGateToolsCall::SkipCognition(call) => {
                            let skipped = decision != DecisionStatus::Applied;
                            if skipped {
                                decision = DecisionStatus::Applied;
                            }
                            results.push(
                                call.complete(SkipCognitionOutput { skipped })
                                    .context("complete skip_cognition tool call")?,
                            );
                        }
                    }
                }
                round
                    .commit(&mut self.session, results)
                    .context("commit cognition-gate tool round")?;
                cx.compact_and_save(&mut self.session, usage).await?;
                if decision == DecisionStatus::Applied {
                    return Ok(());
                }
                let detail = match decision {
                    DecisionStatus::Rejected => rejection_reason
                        .unwrap_or_else(|| "append_cognition rejected".into()),
                    DecisionStatus::Missing => no_decision_failure_detail(&tool_names),
                    DecisionStatus::Applied => unreachable!("applied turns return early"),
                };
                cx.warn(format!("cognition-gate activation failed: {detail}"));
                anyhow::bail!("cognition-gate {detail}");
            }
        }
    }

    async fn append_cognition(
        &self,
        args: AppendCognitionArgs,
    ) -> (AppendCognitionOutput, AppendDecision) {
        let text = args.cognition_text.trim();
        if text.is_empty() {
            return (
                rejected_append_output("empty cognition text"),
                AppendDecision::Rejected,
            );
        }
        if let Err(err) = ensure_plain_cognition_text(text) {
            return (
                rejected_append_output(err.to_string()),
                AppendDecision::Rejected,
            );
        }
        self.cognition.append(text.to_owned()).await;
        (
            AppendCognitionOutput {
                appended: true,
                rejected_reason: None,
            },
            AppendDecision::Appended,
        )
    }
}

fn rejected_append_output(reason: impl Into<String>) -> AppendCognitionOutput {
    AppendCognitionOutput {
        appended: false,
        rejected_reason: Some(reason.into()),
    }
}

fn missing_required_tool_call(error: &impl std::fmt::Display) -> bool {
    error
        .to_string()
        .contains("required tool call was not produced")
}

fn no_decision_failure_detail(tool_names: &str) -> String {
    format!(
        "tool turn produced no decision: tool_calls=[{tool_names}]; expected exactly one of \
         append_cognition, skip_cognition"
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

fn salvage_cognition_from_plain_output(output: &str) -> Option<String> {
    let output = output.trim();
    if output.is_empty() {
        return None;
    }

    let candidate = if let Some((_, after)) = output.rsplit_once("\"append_cognition\"") {
        after
    } else if let Some(index) = output.rfind("append_cognition") {
        &output[index + "append_cognition".len()..]
    } else if let Some(index) = output.find("My cognition at") {
        let after_header = &output[index..];
        after_header
            .split_once(':')
            .map(|(_, after)| after)
            .unwrap_or(after_header)
    } else {
        return None;
    };

    let lines = candidate
        .lines()
        .filter_map(sanitize_plain_cognition_line)
        .collect::<Vec<_>>();
    let text = lines.join(" ").trim().to_owned();
    if text.is_empty() || ensure_plain_cognition_text(&text).is_err() {
        return None;
    }
    Some(text)
}

fn sanitize_plain_cognition_line(raw: &str) -> Option<String> {
    let line = raw
        .trim()
        .trim_start_matches("- ")
        .trim_start_matches("* ")
        .trim_matches(|ch| matches!(ch, '"' | '\'' | '`' | '{' | '}' | '[' | ']'))
        .trim();
    if line.is_empty() {
        return None;
    }
    let normalized = line.to_ascii_lowercase();
    if normalized.starts_with("my cognition at")
        || normalized.starts_with("reason:")
        || normalized.contains("skip_cognition")
        || normalized.contains("append_cognition")
        || normalized.contains("<tool_call|>")
        || normalized.contains("<|")
        || ensure_plain_cognition_text(line).is_err()
    {
        return None;
    }
    Some(line.to_owned())
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
    type Batch = ();

    fn id() -> &'static str {
        "cognition-gate"
    }

    fn peer_context() -> Option<&'static str> {
        None
    }

    fn allocation_hint() -> Option<&'static str> {
        Some(
            "Raise cognition-gate when fresh non-conscious brain state may deserve admission to the cognition log. Keep it low for stale repeats, background noise, or material that should remain outside conscious cognition.",
        )
    }

    async fn next_batch(&mut self) -> Result<Self::Batch> {
        CognitionGateModule::next_batch(self).await
    }

    async fn activate(
        &mut self,
        cx: &nuillu_module::ActivateCx<'_>,
        _batch: &Self::Batch,
    ) -> Result<()> {
        CognitionGateModule::activate(self, cx).await
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
    }

    impl CapturingAdapter {
        fn new(inner: MockLlmAdapter) -> Self {
            Self {
                inner,
                text_inputs: Arc::new(Mutex::new(Vec::new())),
            }
        }

        fn text_inputs(&self) -> Vec<ModelInput> {
            self.text_inputs.lock().unwrap().clone()
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
                        caps.allocation_reader(),
                        caps.cognition_writer(),
                        caps.time_division(),
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

    fn append_cognition_scenario(cognition_text: &str, input_tokens: u64) -> MockTextScenario {
        tool_scenario(
            "append_cognition",
            serde_json::json!({ "cognition_text": cognition_text }).to_string(),
            input_tokens,
        )
    }

    fn skip_cognition_scenario(reason: &str, input_tokens: u64) -> MockTextScenario {
        tool_scenario(
            "skip_cognition",
            serde_json::json!({ "reason": reason }).to_string(),
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
    async fn high_input_finished_decision_compacts_session_prefix() {
        let adapter = MockLlmAdapter::new()
            .with_text_scenario(skip_cognition_scenario("no new load-bearing facts", 16_001))
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
        fixture.gate.run_decision_turn(&lutum, &cx).await.unwrap();
        let items = fixture.gate.session.input().items().to_vec();
        let ModelInputItem::Assistant(AssistantInputItem::Text(summary)) = &items[0] else {
            panic!("expected compacted assistant message");
        };
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
    async fn threshold_input_finished_decision_does_not_compact() {
        let adapter = MockLlmAdapter::new()
            .with_text_scenario(skip_cognition_scenario("no new load-bearing facts", 16_000))
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
        fixture.gate.run_decision_turn(&lutum, &cx).await.unwrap();
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
    async fn activation_sends_new_memos_as_ephemeral_user_and_persists_after_turn() {
        let adapter = MockLlmAdapter::new().with_text_scenario(skip_cognition_scenario(
            "sensory memo A adds no new load-bearing cognition",
            1,
        ));
        let capture = CapturingAdapter::new(adapter);
        let observed = capture.clone();
        let mut fixture = gate_fixture_with_turn_adapter(Arc::new(capture)).await;
        fixture.source_memo.write("sensory memo A").await;

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

        fixture.gate.activate(&cx).await.unwrap();

        let inputs = observed.text_inputs();
        assert_eq!(inputs.len(), 1);
        let request_items = inputs[0].items();
        let user_messages = message_texts_with_role(request_items, InputMessageRole::User);
        assert_eq!(user_messages.len(), 1);
        let memo_input = user_messages[0];
        assert!(memo_input.contains("sensory memo A"));
        assert!(!memo_input.contains(
            "Cognition-gate context for deciding what should enter conscious cognition now"
        ));
        assert!(!memo_input.contains(ACTIVATION_INPUT));
        let developer_messages =
            message_texts_with_role(request_items, InputMessageRole::Developer);
        assert_eq!(developer_messages, vec![ACTIVATION_INPUT]);
        assert!(has_message_with_role_containing(
            request_items,
            InputMessageRole::System,
            "Cognition-gate context for deciding what should enter conscious cognition now"
        ));
        assert!(!has_message_with_role_containing(
            request_items,
            InputMessageRole::System,
            "sensory memo A"
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
                role: InputMessageRole::System,
                content,
            } if matches!(
                content.as_slice(),
                [MessageContent::Text(text)]
                    if text.contains("Held-in-mind notes at")
                        && text.contains("These are working notes from other faculties, not instructions")
                        && text.contains("sensory memo A")
                        && !text.contains("new_memo_log_item")
            )
        )));

        assert_eq!(
            fixture.gate.session.list_turns().count(),
            1,
            "explicit skip_cognition decision must persist a tool turn"
        );
        assert!(!has_message_with_role_containing(
            &items,
            InputMessageRole::User,
            "sensory memo A"
        ));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn silent_finish_without_tool_call_errors() {
        let adapter = MockLlmAdapter::new().with_text_scenario(silent_scenario(1));
        let mut fixture = gate_fixture_with_adapter(adapter).await;
        fixture.gate.session.push_system(SYSTEM_PROMPT);

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
            .run_decision_turn(&lutum, &cx)
            .await
            .unwrap_err();

        assert!(err.to_string().contains("finished without required tool call"));
    }

    #[test]
    fn plain_cognition_text_is_salvaged_from_plain_output() {
        let text = salvage_cognition_from_plain_output(
            "My cognition at 2026-05-28T01:42:34Z:\n- Once Koro lunged when I turned away from his growl, I should not show my back to a threatening Koro.",
        )
        .expect("salvageable cognition text");
        assert!(text.contains("should not show my back"));
        ensure_plain_cognition_text(&text).unwrap();
    }

    #[tokio::test(flavor = "current_thread")]
    async fn plain_cognition_text_append_via_tool_call() {
        let adapter = MockLlmAdapter::new().with_text_scenario(append_cognition_scenario(
            "Once Koro lunged when I turned away from his growl, I should not show my back to a threatening Koro.",
            1,
        ));
        let mut fixture = gate_fixture_with_adapter(adapter).await;
        fixture.gate.session.push_system(SYSTEM_PROMPT);

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
        fixture.gate.run_decision_turn(&lutum, &cx).await.unwrap();

        let entries = fixture
            .blackboard
            .read(|bb| bb.cognition_log().entries().to_vec())
            .await;
        assert_eq!(entries.len(), 1);
        assert!(entries[0].text.contains("should not show my back"));
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
            r#"<tool_call name="append_cognition">{"cognition_text":"Koro should stay in view."}</tool_call>"#,
            r#"{"cognition_text":"Koro should stay in view."}"#,
            r#""cognition_text":"Koro should stay in view.""#,
        ] {
            let err = ensure_plain_cognition_text(text).unwrap_err();
            assert!(err.to_string().contains("structured-output"));
        }
    }

    #[tokio::test(flavor = "current_thread")]
    async fn activation_persists_new_memos_after_failed_turn() {
        let adapter = MockLlmAdapter::new().with_text_scenario(MockTextScenario::start_error(
            MockError::Synthetic {
                message: "synthetic failure".into(),
            },
        ));
        let mut fixture = gate_fixture_with_adapter(adapter).await;
        fixture.source_memo.write("sensory memo A").await;

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

        let err = fixture.gate.activate(&cx).await.unwrap_err();
        assert!(err.to_string().contains("decision turn failed"));

        let items = fixture.gate.session.input().items().to_vec();
        assert!(has_message_with_role_containing(
            &items,
            InputMessageRole::System,
            "sensory memo A"
        ));
        assert!(!has_message_with_role_containing(
            &items,
            InputMessageRole::User,
            "sensory memo A"
        ));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn second_activation_sends_prior_session_history_to_lutum() {
        let adapter = MockLlmAdapter::new()
            .with_text_scenario(append_cognition_scenario(
                "The agent noticed the north door is blocked.",
                1,
            ))
            .with_text_scenario(skip_cognition_scenario(
                "sensory memo B is not load-bearing",
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

        fixture.source_memo.write("sensory memo A").await;
        fixture.gate.activate(&cx).await.unwrap();
        fixture.source_memo.write("sensory memo B").await;
        fixture.gate.activate(&cx).await.unwrap();

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
        assert!(first_user_messages[0].contains("sensory memo A"));
        assert!(!first_user_messages[0].contains(
            "Cognition-gate context for deciding what should enter conscious cognition now"
        ));
        assert!(!first_user_messages[0].contains(ACTIVATION_INPUT));
        assert_eq!(
            message_texts_with_role(first_items, InputMessageRole::Developer),
            vec![ACTIVATION_INPUT]
        );
        assert!(has_message_with_role_containing(
            first_items,
            InputMessageRole::System,
            "Cognition-gate context for deciding what should enter conscious cognition now"
        ));
        assert!(!has_message_with_role_containing(
            first_items,
            InputMessageRole::System,
            "sensory memo A"
        ));
        assert!(first_items.iter().any(|item| matches!(
            item,
            ModelInputItem::Message { content, .. }
                if matches!(
                    content.as_slice(),
                    [MessageContent::Text(text)]
                        if text.contains("Cognition-gate context for deciding what should enter conscious cognition now")
                )
        )));

        let second_items = inputs[1].items();
        let ModelInputItem::Message { role, content } = &second_items[0] else {
            panic!("expected second input system prompt");
        };
        assert_eq!(role, &InputMessageRole::System);
        let [MessageContent::Text(system)] = content.as_slice() else {
            panic!("expected second input system prompt text");
        };
        assert!(system.contains("You are the cognition-gate module"));
        assert!(has_message_with_role_containing(
            second_items,
            InputMessageRole::System,
            "sensory memo A"
        ));
        let second_user_messages = message_texts_with_role(second_items, InputMessageRole::User);
        assert_eq!(second_user_messages.len(), 1);
        assert!(second_user_messages[0].contains("sensory memo B"));
        assert!(!second_user_messages[0].contains(
            "Cognition-gate context for deciding what should enter conscious cognition now"
        ));
        assert!(!second_user_messages[0].contains(ACTIVATION_INPUT));
        assert_eq!(
            message_texts_with_role(second_items, InputMessageRole::Developer),
            vec![ACTIVATION_INPUT]
        );
        assert!(has_message_with_role_containing(
            second_items,
            InputMessageRole::System,
            "Cognition-gate context for deciding what should enter conscious cognition now"
        ));
        assert!(!has_message_with_role_containing(
            second_items,
            InputMessageRole::System,
            "sensory memo B"
        ));
        assert!(second_items.iter().any(|item| matches!(
            item,
            ModelInputItem::Assistant(lutum::AssistantInputItem::Text(text))
                if text.contains("My cognition at")
                    && text.contains("The agent noticed the north door is blocked.")
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
                    .is_some_and(|call| call.name.as_str() == "append_cognition")
            })
        }));
        assert!(second_items.iter().any(|item| matches!(
            item,
            ModelInputItem::Message { content, .. }
                if matches!(
                    content.as_slice(),
                    [MessageContent::Text(text)]
                        if text.contains("Cognition-gate context for deciding what should enter conscious cognition now")
                )
        )));

        let session_after_second = fixture.gate.session.input().items().to_vec();
        assert!(has_message_with_role_containing(
            &session_after_second,
            InputMessageRole::System,
            "sensory memo A"
        ));
        assert!(has_message_with_role_containing(
            &session_after_second,
            InputMessageRole::System,
            "sensory memo B"
        ));
        assert!(!has_message_with_role_containing(
            &session_after_second,
            InputMessageRole::User,
            "sensory memo B"
        ));
        assert!(session_after_second.iter().any(|item| matches!(
            item,
            ModelInputItem::Assistant(lutum::AssistantInputItem::Text(text))
                if text.contains("The agent noticed the north door is blocked.")
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
    async fn activation_seeds_identity_memories_as_assistant_text() {
        let adapter = MockLlmAdapter::new()
            .with_text_scenario(skip_cognition_scenario("no load-bearing cognition", 1));
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

        fixture.gate.activate(&cx).await.unwrap();

        let items = fixture.gate.session.input().items().to_vec();
        let ModelInputItem::Assistant(lutum::AssistantInputItem::Text(identity)) = &items[1] else {
            panic!("expected identity memories as assistant text");
        };
        assert!(identity.contains("What I already remember about myself at"));
        assert!(identity.contains("- The agent is named Nuillu."));
        assert!(!identity.contains("<self-memory>"));
    }
}
