use anyhow::{Context, Result};
use async_trait::async_trait;
use lutum::Session;
use nuillu_module::{
    AllocationReader, BlackboardReader, CognitionLogReader, CognitionLogUpdatedInbox,
    EphemeralMindContext, LlmAccess, Memo, Module, SessionCompactionConfig,
    compact_session_if_needed, memory_rank_counts, push_ephemeral_mind_context,
    push_formatted_cognition_log_batch, push_formatted_memo_log_batch,
};

mod batch;

const SYSTEM_PROMPT: &str = r#"You are the predict module.
Maintain forward predictions about the current cognition-log targets.
Generate predictions only; do not assess whether earlier predictions were correct and do not
request memory writes. Keep predictions concise, grounded in the current cognition log and
blackboard context.
Write the memo as free-form prose. For each useful prediction, preserve the subject, predicted
state, validity horizon, and rationale, but do not encode the memo as JSON, YAML, a code block, or
any fixed schema."#;

const COMPACTED_PREDICT_SESSION_PREFIX: &str = "Compacted predict session history:";
const SESSION_COMPACTION_PROMPT: &str = r#"You compact the predict module's persistent session history.
Summarize only the prefix transcript you receive. Preserve prior predictions, their subjects,
validity horizons, rationales, memo-log facts, and cognition-log context needed for future
prediction updates. Do not invent facts. Return plain text only."#;

pub struct PredictModule {
    owner: nuillu_module::ModuleId,
    updates: CognitionLogUpdatedInbox,
    cognition_log: CognitionLogReader,
    allocation: AllocationReader,
    blackboard: BlackboardReader,
    memo: Memo,
    llm: LlmAccess,
    session: Session,
    session_compaction: SessionCompactionConfig,
    system_prompt: std::sync::OnceLock<String>,
}

impl PredictModule {
    pub fn new(
        updates: CognitionLogUpdatedInbox,
        cognition_log: CognitionLogReader,
        allocation: AllocationReader,
        blackboard: BlackboardReader,
        memo: Memo,
        llm: LlmAccess,
    ) -> Self {
        Self {
            owner: nuillu_module::ModuleId::new(<Self as Module>::id())
                .expect("predict id is valid"),
            updates,
            cognition_log,
            allocation,
            blackboard,
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
        let rank_counts = self
            .blackboard
            .read(|bb| memory_rank_counts(bb.memory_metadata()))
            .await;
        let allocation = self.allocation.snapshot().await;

        let system_prompt = self.system_prompt(cx).to_owned();
        self.session.push_ephemeral_system(system_prompt);
        push_formatted_cognition_log_batch(&mut self.session, &unread_cognition, cx.now());
        push_ephemeral_mind_context(
            &mut self.session,
            EphemeralMindContext {
                memos: &[],
                memory_rank_counts: Some(&rank_counts),
                allocation: Some(&allocation),
                available_faculties: &[],
                time_division: None,
                stuckness: None,
                now: cx.now(),
            },
        );
        self.session
            .push_ephemeral_developer("Update forward predictions for the cognition above.");

        let lutum = self.llm.lutum().await;
        let result = self
            .session
            .text_turn(&lutum)
            .collect()
            .await
            .context("predict text turn failed")?;
        compact_session_if_needed(
            &mut self.session,
            result.usage.input_tokens,
            cx.session_compaction_lutum(),
            self.session_compaction,
            Self::id(),
            COMPACTED_PREDICT_SESSION_PREFIX,
            SESSION_COMPACTION_PROMPT,
        )
        .await;
        let memo = result.assistant_text();
        if !memo.trim().is_empty() {
            self.memo.write(memo).await;
        }
        Ok(())
    }
}

#[async_trait(?Send)]
impl Module for PredictModule {
    type Batch = ();

    fn id() -> &'static str {
        "predict"
    }

    fn role_description() -> &'static str {
        "Generates forward predictions about current cognition-log subjects on each cognition-log update and writes them to its memo."
    }

    async fn next_batch(&mut self) -> Result<Self::Batch> {
        PredictModule::next_batch(self).await
    }

    async fn activate(
        &mut self,
        cx: &nuillu_module::ActivateCx<'_>,
        _batch: &Self::Batch,
    ) -> Result<()> {
        PredictModule::activate(self, cx).await
    }
}
