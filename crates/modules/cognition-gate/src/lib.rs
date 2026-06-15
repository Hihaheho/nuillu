use anyhow::{Context, Result, anyhow};
use async_trait::async_trait;
use lutum::{Lutum, Session, TextStepOutcomeWithTools, ToolResult, Usage};
use nuillu_blackboard::MemoLogRecord;
use nuillu_module::{
    BlackboardReader, CognitionWriter, LlmAccess, LlmContextWindow, MemoUpdatedInbox, Module,
    SessionAutoCompaction, SessionCompactionConfig, SessionCompactionProtectedPrefix,
    ensure_persistent_session_seeded, format_bounded_memo_log_batch, format_faculty_system_prompt,
    push_formatted_cognition_log_batch,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

mod batch;
pub use batch::NextBatch as CognitionGateBatch;

const SYSTEM_PROMPT: &str = r#"You are the cognition-gate module — a selective attention
filter that decides what reaches the agent's conscious workspace. The cognition log IS that
workspace: it is what other cognitive modules read when they decide how to think, plan, or
speak. Every entry you promote becomes part of the agent's first-person awareness for this
moment.

Your job is to judge, given the recent conscious workspace and new cognition candidates,
what the conscious mind needs to know in order to act. From that judgment two duties follow.

1. Admit everything currently load-bearing.
Read the recent contents of the conscious workspace to infer what the agent is presently
trying to do, attend to, or respond to. Then pull into consciousness every subconscious
fact that could change that decision — specific safety constraints, peer-model rules,
sensory details, recalled episodes, body or world facts. Preserve specifics: a concrete
actionable rule must enter the workspace as the rule it actually is, not as a flattened
summary that drops the operative detail. Generic paraphrases are not equivalents.
New cognition candidates are newly reported occurrences, facts, or updates. Evaluate
each candidate against the current situation and prior conscious context. Use
promote_to_cognition when a candidate changes what the agent should be aware of, say, do, or
keep tracking now. Never call promote_to_cognition for a mere restatement or tense-shifted
paraphrase of what is already conscious. Use leave_cognition_unchanged only when promoting
the candidates would add no current situational value beyond what is already conscious.

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
recalling, or realizing. Use age tags supplied on candidates and cognition history for
past observations. When candidates should enter conscious cognition, call
promote_to_cognition exactly once. When cognition should stay unchanged, call
leave_cognition_unchanged exactly once. Do not use final assistant text as an output
channel."#;

const COMPACTED_COGNITION_GATE_SESSION_PREFIX: &str = "Compacted cognition-gate session history:";
const MEMO_CONTEXT_WINDOW: LlmContextWindow = LlmContextWindow::new(8, 1_200, 4_800);
const COGNITION_CONTEXT_WINDOW: LlmContextWindow = LlmContextWindow::new(12, 600, 4_800);
const SESSION_COMPACTION_FOCUS: &str = r#"Preserve candidate facts, prior gate decisions, promoted
events, rejected candidate events, cognition context, and relevant memory evidence needed for
future cognition-gate decisions."#;
const NEW_CANDIDATE_HEADER: &str = "New cognition candidates:";
const CANDIDATE_DECISION_INSTRUCTION: &str = "Review the new cognition candidates above and call \
exactly one tool: promote_to_cognition if any candidate should enter conscious cognition now, or \
leave_cognition_unchanged if none should. Do not write assistant text.";

pub fn session_auto_compaction() -> SessionAutoCompaction {
    SessionAutoCompaction::new(
        SessionCompactionConfig::default(),
        SessionCompactionProtectedPrefix::LeadingSystemAndIdentitySeed,
        COMPACTED_COGNITION_GATE_SESSION_PREFIX,
        SESSION_COMPACTION_FOCUS,
    )
}

fn retitle_candidate_batch(formatted: String, header: &str) -> Option<String> {
    let mut lines = formatted.lines();
    let _ = lines.next()?;

    let mut out = header.to_owned();
    let mut count = 0usize;
    for line in lines {
        let Some(line) = line.strip_prefix("- ") else {
            continue;
        };
        let Some((source_and_age, content)) = line.split_once(": ") else {
            continue;
        };
        let age = source_and_age
            .rsplit_once(", ")
            .map(|(_, age)| age)
            .unwrap_or(source_and_age);
        count += 1;
        out.push_str("\n- candidate ");
        out.push_str(&count.to_string());
        out.push_str(", ");
        out.push_str(age);
        out.push_str(": ");
        out.push_str(content);
    }

    if count == 0 {
        return None;
    }

    out.push_str("\n\n");
    out.push_str(CANDIDATE_DECISION_INSTRUCTION);
    Some(out)
}

#[lutum::tool_input(name = "promote_to_cognition", output = PromoteToCognitionOutput)]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct PromoteToCognitionArgs {
    /// Plain inner-experience prose to promote into current cognition.
    /// It must add current situational value rather than restating what is already conscious.
    pub promotion_text: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct PromoteToCognitionOutput {
    pub promoted: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rejected_reason: Option<String>,
}

#[lutum::tool_input(name = "leave_cognition_unchanged", output = LeaveCognitionUnchangedOutput)]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct LeaveCognitionUnchangedArgs {
    /// Plain reason why promoting the current candidates would add no current situational value.
    /// Do not use this for candidates that should enter conscious cognition.
    pub reason: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct LeaveCognitionUnchangedOutput {
    pub unchanged: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema, lutum::Toolset)]
pub enum CognitionGateTools {
    PromoteToCognition(PromoteToCognitionArgs),
    LeaveCognitionUnchanged(LeaveCognitionUnchangedArgs),
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum DecisionStatus {
    Applied,
    Rejected,
    Missing,
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum PromotionDecision {
    Promoted,
    Rejected,
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
        if self.pending_unread_memos.len() == 1 {
            let record = self
                .pending_unread_memos
                .first()
                .expect("single pending memo exists");
            let (output, decision) = self
                .promote_to_cognition(PromoteToCognitionArgs {
                    promotion_text: record.content.trim().to_owned(),
                })
                .await;
            if decision == PromotionDecision::Rejected {
                let detail = output
                    .rejected_reason
                    .unwrap_or_else(|| "direct promotion rejected".to_owned());
                anyhow::bail!("cognition-gate direct promotion rejected: {detail}");
            }
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
        if let Some(candidate_notes) =
            format_bounded_memo_log_batch(&self.pending_unread_memos, cx.now(), MEMO_CONTEXT_WINDOW)
                .and_then(|batch| retitle_candidate_batch(batch, NEW_CANDIDATE_HEADER))
        {
            self.session.push_user(candidate_notes);
        }

        let lutum = self.llm.lutum().await;
        let result = self.run_decision_turn(&lutum, cx).await;
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
                CognitionGateToolsSelector::PromoteToCognition,
                CognitionGateToolsSelector::LeaveCognitionUnchanged,
            ])
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
            TextStepOutcomeWithTools::Finished(result) => {
                let usage = result.usage;
                if let Some(cognition_text) =
                    salvage_cognition_from_plain_output(&result.assistant_text())
                {
                    let (output, promotion_decision) = self
                        .promote_to_cognition(PromoteToCognitionArgs {
                            promotion_text: cognition_text,
                        })
                        .await;
                    if promotion_decision == PromotionDecision::Promoted {
                        cx.compact_and_save(&mut self.session, usage).await?;
                        return Ok(());
                    }
                    let detail = output
                        .rejected_reason
                        .unwrap_or_else(|| "promote_to_cognition rejected".into());
                    cx.warn(format!("cognition-gate activation failed: {detail}"));
                    anyhow::bail!("cognition-gate {detail}");
                }
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
                let mut results: Vec<ToolResult> = Vec::new();
                nuillu_module::emit_trace_tool_calls(&round.tool_calls);
                if round.tool_calls.is_empty() {
                    let detail = format!(
                        "model returned NeedsTools outcome with empty tool_calls; \
                         expected promote_to_cognition or leave_cognition_unchanged"
                    );
                    cx.warn(format!("cognition-gate activation failed: {detail}"));
                }
                let mut decision = DecisionStatus::Missing;
                let mut rejection_reason: Option<String> = None;
                for call in round.tool_calls.iter().cloned() {
                    match call {
                        CognitionGateToolsCall::PromoteToCognition(call) => {
                            let (output, promotion_decision) =
                                if decision == DecisionStatus::Applied {
                                    (
                                        rejected_promotion_output(
                                            "another cognition decision was already applied",
                                        ),
                                        PromotionDecision::Rejected,
                                    )
                                } else {
                                    self.promote_to_cognition(call.input.clone()).await
                                };
                            match promotion_decision {
                                PromotionDecision::Promoted => decision = DecisionStatus::Applied,
                                PromotionDecision::Rejected
                                    if decision == DecisionStatus::Missing =>
                                {
                                    decision = DecisionStatus::Rejected;
                                    rejection_reason = output.rejected_reason.clone();
                                }
                                PromotionDecision::Rejected => {}
                            }
                            results.push(
                                call.complete(output)
                                    .context("complete promote_to_cognition tool call")?,
                            );
                        }
                        CognitionGateToolsCall::LeaveCognitionUnchanged(call) => {
                            let unchanged = decision != DecisionStatus::Applied;
                            if unchanged {
                                decision = DecisionStatus::Applied;
                            }
                            results.push(
                                call.complete(LeaveCognitionUnchangedOutput { unchanged })
                                    .context("complete leave_cognition_unchanged tool call")?,
                            );
                        }
                    }
                }
                round
                    .commit(&mut self.session, results)
                    .context("commit cognition-gate tool round")?;
                if decision == DecisionStatus::Applied {
                    cx.compact_and_save(&mut self.session, usage).await?;
                    return Ok(());
                }
                let detail = match decision {
                    DecisionStatus::Rejected => {
                        rejection_reason.unwrap_or_else(|| "promote_to_cognition rejected".into())
                    }
                    DecisionStatus::Missing => no_decision_failure_detail(&tool_names),
                    DecisionStatus::Applied => unreachable!("applied turns return early"),
                };
                cx.warn(format!("cognition-gate activation failed: {detail}"));
                anyhow::bail!("cognition-gate {detail}");
            }
        }
    }

    async fn promote_to_cognition(
        &self,
        args: PromoteToCognitionArgs,
    ) -> (PromoteToCognitionOutput, PromotionDecision) {
        let text = args.promotion_text.trim();
        if text.is_empty() {
            return (
                rejected_promotion_output("empty cognition text"),
                PromotionDecision::Rejected,
            );
        }
        if let Err(err) = ensure_plain_cognition_text(text) {
            return (
                rejected_promotion_output(err.to_string()),
                PromotionDecision::Rejected,
            );
        }
        self.cognition.append(text.to_owned()).await;
        (
            PromoteToCognitionOutput {
                promoted: true,
                rejected_reason: None,
            },
            PromotionDecision::Promoted,
        )
    }
}

fn rejected_promotion_output(reason: impl Into<String>) -> PromoteToCognitionOutput {
    PromoteToCognitionOutput {
        promoted: false,
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
         promote_to_cognition, leave_cognition_unchanged"
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

    let candidate = if let Some((_, after)) = output.rsplit_once("\"promote_to_cognition\"") {
        after
    } else if let Some(index) = output.rfind("promote_to_cognition") {
        &output[index + "promote_to_cognition".len()..]
    } else if let Some(after) = plain_cognition_header_suffix(output) {
        after
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
        || normalized.starts_with("current cognition log at")
        || normalized.starts_with("reason:")
        || normalized.contains("leave_cognition_unchanged")
        || normalized.contains("promote_to_cognition")
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

fn plain_cognition_header_suffix(output: &str) -> Option<&str> {
    ["Current cognition log at", "My cognition at"]
        .into_iter()
        .filter_map(|header| output.find(header).map(|index| &output[index..]))
        .min_by_key(|suffix| output.len() - suffix.len())
        .map(|after_header| {
            after_header
                .split_once(':')
                .map(|(_, after)| after)
                .unwrap_or(after_header)
        })
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
        || normalized.contains("\"promote_to_cognition\"")
        || normalized.contains("\"leave_cognition_unchanged\"")
        || normalized.contains("promotion_text\":")
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

    fn promote_to_cognition_scenario(promotion_text: &str, input_tokens: u64) -> MockTextScenario {
        tool_scenario(
            "promote_to_cognition",
            serde_json::json!({ "promotion_text": promotion_text }).to_string(),
            input_tokens,
        )
    }

    fn leave_cognition_unchanged_scenario(reason: &str, input_tokens: u64) -> MockTextScenario {
        tool_scenario(
            "leave_cognition_unchanged",
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
    async fn high_input_finished_decision_compacts_session_prefix() {
        let adapter = MockLlmAdapter::new()
            .with_text_scenario(leave_cognition_unchanged_scenario(
                "no new load-bearing facts",
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
            .with_text_scenario(leave_cognition_unchanged_scenario(
                "no new load-bearing facts",
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
    async fn activation_sends_new_candidates_as_persistent_user() {
        let adapter = MockLlmAdapter::new().with_text_scenario(leave_cognition_unchanged_scenario(
            "candidate A adds no current situational value",
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
                        && !text.contains("memo")
                        && !text.contains("Held-in-mind notes")
                        && !text.contains("working notes")
            )
        )));

        assert_eq!(
            fixture.gate.session.list_turns().count(),
            1,
            "explicit leave_cognition_unchanged decision must persist a tool turn"
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

        assert!(
            err.to_string()
                .contains("finished without required tool call")
        );
    }

    #[test]
    fn plain_cognition_text_is_salvaged_from_plain_output() {
        for header in [
            "Current cognition log at 2026-05-28T01:42:34Z:",
            "My cognition at 2026-05-28T01:42:34Z:",
        ] {
            let text = salvage_cognition_from_plain_output(&format!(
                "{header}\n- Once Koro lunged when I turned away from his growl, I should not show my back to a threatening Koro."
            ))
            .expect("salvageable cognition text");
            assert!(text.contains("should not show my back"));
            ensure_plain_cognition_text(&text).unwrap();
        }
    }

    #[tokio::test(flavor = "current_thread")]
    async fn plain_cognition_text_append_via_tool_call() {
        let adapter = MockLlmAdapter::new().with_text_scenario(promote_to_cognition_scenario(
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
            r#"<tool_call name="promote_to_cognition">{"promotion_text":"Koro should stay in view."}</tool_call>"#,
            r#"{"promotion_text":"Koro should stay in view."}"#,
            r#""promotion_text":"Koro should stay in view.""#,
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
            .with_text_scenario(leave_cognition_unchanged_scenario(
                "candidate A is not load-bearing",
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
        assert!(err.to_string().contains("decision turn failed"));
        assert_eq!(fixture.gate.pending_unread_memos.len(), 2);

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

        fixture.gate.activate(&cx, &batch).await.unwrap();
        assert!(fixture.gate.pending_unread_memos.is_empty());

        let inputs = observed.text_inputs();
        assert_eq!(inputs.len(), 2);
        let retried_candidates = latest_candidate_user_message(inputs[1].items());
        assert!(retried_candidates.contains(NEW_CANDIDATE_HEADER));
        assert_candidate_decision_instruction(retried_candidates);
        assert!(retried_candidates.contains("sensory detail A"));
        assert!(retried_candidates.contains("sensory detail B"));
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
            .with_text_scenario(leave_cognition_unchanged_scenario(
                "pending candidates are not load-bearing",
                1,
            ))
            .with_text_scenario(leave_cognition_unchanged_scenario(
                "candidate C is not load-bearing",
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
        let first_batch = next_gate_batch(&mut fixture).await;
        fixture.gate.activate(&cx, &first_batch).await.unwrap_err();
        assert_eq!(fixture.gate.pending_unread_memos.len(), 2);

        fixture
            .source_memo
            .write_cognitive("sensory detail C")
            .await;
        let second_batch = next_gate_batch(&mut fixture).await;
        fixture.gate.activate(&cx, &second_batch).await.unwrap_err();
        assert_eq!(fixture.gate.pending_unread_memos.len(), 3);

        fixture.gate.activate(&cx, &second_batch).await.unwrap();
        assert!(fixture.gate.pending_unread_memos.is_empty());

        let inputs = observed.text_inputs();
        assert_eq!(inputs.len(), 3);
        let accumulated_candidates = latest_candidate_user_message(inputs[2].items());
        assert_candidate_decision_instruction(accumulated_candidates);
        assert!(accumulated_candidates.contains("sensory detail A"));
        assert!(accumulated_candidates.contains("sensory detail B"));
        assert!(accumulated_candidates.contains("sensory detail C"));

        fixture
            .source_memo
            .write_cognitive("sensory detail D")
            .await;
        fixture
            .source_memo
            .write_cognitive("sensory detail E")
            .await;
        let third_batch = next_gate_batch(&mut fixture).await;
        fixture.gate.activate(&cx, &third_batch).await.unwrap();

        let inputs = observed.text_inputs();
        assert_eq!(inputs.len(), 4);
        let latest_candidates = latest_candidate_user_message(inputs[3].items());
        assert_candidate_decision_instruction(latest_candidates);
        assert!(latest_candidates.contains("sensory detail D"));
        assert!(latest_candidates.contains("sensory detail E"));
        assert!(!latest_candidates.contains("sensory detail A"));
        assert!(!latest_candidates.contains("sensory detail B"));
        assert!(!latest_candidates.contains("sensory detail C"));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn second_activation_sends_prior_session_history_to_lutum() {
        let adapter = MockLlmAdapter::new()
            .with_text_scenario(promote_to_cognition_scenario(
                "The agent noticed the north door is blocked.",
                1,
            ))
            .with_text_scenario(leave_cognition_unchanged_scenario(
                "candidate B is not load-bearing",
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
            .write_cognitive("sensory detail A2")
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
        assert!(first_user_messages[0].contains("sensory detail A"));
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
            "sensory detail A"
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
                    .is_some_and(|call| call.name.as_str() == "promote_to_cognition")
            })
        }));

        let session_after_second = fixture.gate.session.input().items().to_vec();
        assert!(!has_message_with_role_containing(
            &session_after_second,
            InputMessageRole::System,
            "sensory detail A"
        ));
        assert!(!has_message_with_role_containing(
            &session_after_second,
            InputMessageRole::System,
            "sensory detail B"
        ));
        assert!(has_message_with_role_containing(
            &session_after_second,
            InputMessageRole::User,
            "sensory detail A"
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
        let adapter = MockLlmAdapter::new().with_text_scenario(leave_cognition_unchanged_scenario(
            "no load-bearing cognition",
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
            .update_cognitive_memo(sensory, "identity seed candidate B".into(), now)
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
            memo_logs: vec![first, second],
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
