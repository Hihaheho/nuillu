use anyhow::{Context, Result};
use async_trait::async_trait;
use lutum::Session;
use nuillu_module::{
    CognitionLogReader, CognitionLogUpdatedInbox, LlmAccess, LlmContextWindow, Memo, Module,
    SessionAutoCompaction, SessionCompactionConfig, SessionCompactionProtectedPrefix,
    ensure_persistent_session_seeded, format_identity_system_prompt,
    push_formatted_cognition_log_batch,
};

mod batch;

const SYSTEM_PROMPT: &str = r#"You are the prediction faculty.
Maintain forward predictions about the current cognition-log targets.
Generate predictions only; do not assess whether earlier predictions were correct and do not
request memory writes. Keep predictions concise, grounded in the current cognition log.
By default, predict near-future world, body, peer, object, and environment states for the subjects
that have entered conscious cognition: an apple may fall, a peer may speak, a doorway may remain
empty, a body posture may continue, and so on.
Corner case: when the cognition log explicitly makes the agent's own conscious attention, thought,
uncertainty, or expectation the target, predict the likely near-term flow of that conscious state.
This is still a cognition-level prediction, not a prediction about hidden implementation machinery.
Do not predict hidden implementation, scheduling, resource-control, storage, retrieval, grading, or
other machinery.
Write the memo as free-form prose. For each useful prediction, preserve the subject, predicted
state, validity horizon, and rationale, but do not encode the memo as JSON, YAML, a code block, or
any fixed schema."#;

const COMPACTED_PREDICT_SESSION_PREFIX: &str = "Compacted predict session history:";
const COGNITION_CONTEXT_WINDOW: LlmContextWindow = LlmContextWindow::new(12, 600, 4_800);
const SESSION_COMPACTION_FOCUS: &str = r#"Preserve prior predictions, their subjects, validity
horizons, rationales, and cognition-log context needed for future prediction updates."#;

pub fn session_auto_compaction() -> SessionAutoCompaction {
    SessionAutoCompaction::new(
        SessionCompactionConfig::default(),
        SessionCompactionProtectedPrefix::LeadingSystemAndIdentitySeed,
        COMPACTED_PREDICT_SESSION_PREFIX,
        SESSION_COMPACTION_FOCUS,
    )
}

pub struct PredictModule {
    updates: CognitionLogUpdatedInbox,
    cognition_log: CognitionLogReader,
    memo: Memo,
    llm: LlmAccess,
    session: Session,
    system_prompt: std::sync::OnceLock<String>,
}

impl PredictModule {
    pub fn new(
        updates: CognitionLogUpdatedInbox,
        cognition_log: CognitionLogReader,
        memo: Memo,
        llm: LlmAccess,
        session: Session,
    ) -> Self {
        Self {
            updates,
            cognition_log,
            memo,
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

    #[tracing::instrument(skip_all, err(Debug, level = "warn"))]
    async fn activate(&mut self, cx: &nuillu_module::ActivateCx<'_>) -> Result<()> {
        let unread_cognition = self.cognition_log.unread_events().await;

        self.ensure_session_seeded(cx);
        let lutum = self.llm.lutum().await;
        let memo = {
            push_formatted_cognition_log_batch(
                &mut self.session,
                &unread_cognition,
                cx.now(),
                COGNITION_CONTEXT_WINDOW,
            );
            self.session
                .push_ephemeral_developer("Update forward predictions for the cognition above.");
            let result = self
                .session
                .text_turn()
                .collect(&lutum)
                .await
                .context("predict text turn failed")?;
            cx.compact_and_save(&mut self.session, result.usage).await?;
            result.assistant_text()
        };
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

    fn peer_context() -> Option<&'static str> {
        Some(
            "Predict forms expectations about what may happen next from the current cognitive state.",
        )
    }

    fn allocation_hint() -> Option<&'static str> {
        Some(
            "Raise predict when current cognition needs an expectation about likely next states or what would count as surprising. Keep it low for static facts, resolved outcomes, or recall-only work.",
        )
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
