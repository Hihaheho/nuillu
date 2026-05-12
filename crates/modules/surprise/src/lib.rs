use anyhow::{Context, Result};
use async_trait::async_trait;
use lutum::{Session, StructuredTurnOutcome};
use nuillu_module::{
    AllocationReader, AttentionControlRequest, AttentionControlRequestMailbox, BlackboardReader,
    CognitionLogReader, CognitionLogUpdatedInbox, EphemeralMindContext, LlmAccess, Memo,
    MemoryImportance, Module, SessionCompactionConfig, compact_session_if_needed,
    push_ephemeral_mind_context, push_formatted_cognition_log_batch, push_formatted_memo_log_batch,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

mod batch;

const SYSTEM_PROMPT: &str = r#"You are the surprise module.
Detect unexpected cognition-log entries. If predict memo log entries are present, frame the
assessment as divergence from pending predictions. If no predict memo log is present, judge novelty against recent
cognition-log history. Do not generate forward predictions. Request memory only when the event is
significant enough to preserve. Return only raw JSON for the structured object; do not wrap it in
Markdown or code fences."#;

const COMPACTED_SURPRISE_SESSION_PREFIX: &str = "Compacted surprise session history:";
const SESSION_COMPACTION_PROMPT: &str = r#"You compact the surprise module's persistent session history.
Summarize only the prefix transcript you receive. Preserve prior surprise assessments, predict memo
log facts, significant events, memory preservation requests, and cognition-log context needed for
future surprise checks. Do not invent facts. Return plain text only."#;

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct SurpriseAssessment {
    pub assessment: String,
    pub surprise_level: SurpriseLevel,
    pub significant: bool,
    pub rationale: String,
    pub memory_request: Option<SurpriseMemoryRequest>,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub enum SurpriseLevel {
    None,
    Low,
    Moderate,
    High,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct SurpriseMemoryRequest {
    pub content: String,
    pub importance: MemoryImportance,
    pub reason: String,
}

pub struct SurpriseModule {
    owner: nuillu_types::ModuleId,
    updates: CognitionLogUpdatedInbox,
    cognition_log: CognitionLogReader,
    allocation: AllocationReader,
    blackboard: BlackboardReader,
    attention_control: AttentionControlRequestMailbox,
    memo: Memo,
    llm: LlmAccess,
    session: Session,
    session_compaction: SessionCompactionConfig,
    system_prompt: std::sync::OnceLock<String>,
}

impl SurpriseModule {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        updates: CognitionLogUpdatedInbox,
        cognition_log: CognitionLogReader,
        allocation: AllocationReader,
        blackboard: BlackboardReader,
        attention_control: AttentionControlRequestMailbox,
        memo: Memo,
        llm: LlmAccess,
    ) -> Self {
        Self {
            owner: nuillu_types::ModuleId::new(<Self as Module>::id())
                .expect("surprise id is valid"),
            updates,
            cognition_log,
            allocation,
            blackboard,
            attention_control,
            memo,
            llm,
            session: Session::new(),
            session_compaction: SessionCompactionConfig::default(),
            system_prompt: std::sync::OnceLock::new(),
        }
    }

    fn system_prompt(&self, cx: &nuillu_module::ActivateCx<'_>) -> &str {
        self.system_prompt.get_or_init(|| {
            nuillu_module::format_system_prompt(
                SYSTEM_PROMPT,
                cx.modules(),
                &self.owner,
                cx.identity_memories(),
                cx.now(),
            )
        })
    }

    #[tracing::instrument(skip_all, err(Debug, level = "warn"))]
    async fn activate(&mut self, cx: &nuillu_module::ActivateCx<'_>) -> Result<()> {
        let unread_cognition = self.cognition_log.unread_events().await;
        let unread_memo_logs = self.blackboard.unread_memo_logs().await;
        push_formatted_memo_log_batch(&mut self.session, &unread_memo_logs, cx.now());
        let allocation = self.allocation.snapshot().await;

        let system_prompt = self.system_prompt(cx).to_owned();
        self.session.push_ephemeral_system(system_prompt);
        push_formatted_cognition_log_batch(&mut self.session, &unread_cognition, cx.now());
        push_ephemeral_mind_context(
            &mut self.session,
            EphemeralMindContext {
                memos: &[],
                memory_rank_counts: None,
                allocation: Some(&allocation),
                available_faculties: &[],
                time_division: None,
                stuckness: None,
                now: cx.now(),
            },
        );
        self.session
            .push_ephemeral_developer("Assess whether the new cognition is surprising.");

        let lutum = self.llm.lutum().await;
        let result = self
            .session
            .structured_turn::<SurpriseAssessment>(&lutum)
            .collect()
            .await
            .context("surprise structured turn failed")?;
        let input_tokens = result.usage.input_tokens;

        let StructuredTurnOutcome::Structured(assessment) = result.semantic else {
            anyhow::bail!("surprise structured turn refused");
        };
        compact_session_if_needed(
            &mut self.session,
            input_tokens,
            cx.session_compaction_lutum(),
            self.session_compaction,
            Self::id(),
            COMPACTED_SURPRISE_SESSION_PREFIX,
            SESSION_COMPACTION_PROMPT,
        )
        .await;

        if assessment.significant
            && let Some(request) = &assessment.memory_request
            && !request.content.trim().is_empty()
        {
            let _ = self
                .attention_control
                .publish(AttentionControlRequest::memory(
                    request.content.trim(),
                    request.importance,
                    request.reason.trim(),
                ))
                .await;
        }

        self.memo.write(render_surprise_memo(&assessment)).await;
        Ok(())
    }
}

fn render_surprise_memo(assessment: &SurpriseAssessment) -> String {
    let mut memo = format!(
        "Surprise assessment: {}\nSurprise level: {:?}\nSignificant enough to preserve: {}\nRationale: {}",
        assessment.assessment.trim(),
        assessment.surprise_level,
        if assessment.significant { "yes" } else { "no" },
        assessment.rationale.trim(),
    );
    if let Some(request) = &assessment.memory_request {
        memo.push_str("\nMemory preservation request:");
        memo.push_str("\nContent: ");
        memo.push_str(request.content.trim());
        memo.push_str("\nImportance: ");
        memo.push_str(&format!("{:?}", request.importance));
        memo.push_str("\nReason: ");
        memo.push_str(request.reason.trim());
    } else {
        memo.push_str("\nMemory preservation request: none");
    }
    memo
}

#[async_trait(?Send)]
impl Module for SurpriseModule {
    type Batch = ();

    fn id() -> &'static str {
        "surprise"
    }

    fn role_description() -> &'static str {
        "Detects unexpected cognition-log entries by comparing new entries against predict's memo or recent cognition-log history; can request memory preservation."
    }

    async fn next_batch(&mut self) -> Result<Self::Batch> {
        SurpriseModule::next_batch(self).await
    }

    async fn activate(
        &mut self,
        cx: &nuillu_module::ActivateCx<'_>,
        _batch: &Self::Batch,
    ) -> Result<()> {
        SurpriseModule::activate(self, cx).await
    }
}
