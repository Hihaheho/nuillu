use anyhow::{Context, Result};
use async_trait::async_trait;
use lutum::Session;
use nuillu_module::{
    BlackboardReader, CognitionLogReader, CognitionLogUpdatedInbox, LlmAccess, LlmContextWindow,
    Memo, Module, SessionAutoCompaction, SessionCompactionConfig, SessionCompactionProtectedPrefix,
    ensure_persistent_session_seeded, push_formatted_cognition_log_batch,
    push_formatted_memo_log_batch,
};
use nuillu_types::builtin;

mod batch;
pub use batch::NextBatch as SelfModelBatch;

const SYSTEM_PROMPT: &str = r#"Maintain an agent's current embodied and mental self-model.
You will receive working notes, first-person attention experiences, self-related remembered facts,
and a request for the next self-model memo. Integrate only evidence about the agent's own body or
form, abilities and limitations, interoceptive or affective condition, attention, intention,
uncertainty, agency, and other current mental state.
Use loaded identity memories and self-related notes as the agent's own embodied identity and
abilities. Write established self-state in the agent's first-person voice. Do not identify as the
underlying model, provider, runtime, or an outside observer of the agent. Stable self-knowledge may
be present in remembered facts, but do not claim direct access to raw hidden memories.
Do not recap dialogue, external events, another person's state, prior utterances, retrieved-memory
provenance, poems, or action history unless that evidence directly changes the agent's current body,
capability, affect, attention, intention, uncertainty, or agency. Integrate and replace superseded
self-state instead of appending a chronology or preserving every question and answer.
Write one concise free-form prose snapshot. Do not encode the memo as JSON, YAML, a code block, or
any fixed schema. Write nothing when there is no concrete change or clarification to embodied or
mental self-state."#;

const COMPACTED_SELF_MODEL_SESSION_PREFIX: &str = "Compacted self-model session history:";
const MEMO_CONTEXT_WINDOW: LlmContextWindow = LlmContextWindow::new(8, 1_200, 4_800);
const COGNITION_CONTEXT_WINDOW: LlmContextWindow = LlmContextWindow::new(8, 600, 3_000);
const SESSION_COMPACTION_FOCUS: &str = r#"Preserve the latest embodied identity, abilities and
limitations, interoceptive and affective condition, attention, intention, uncertainty, agency, and
corrections. Do not preserve dialogue or event chronology except where it directly changes that
current self-state."#;

pub fn session_auto_compaction() -> SessionAutoCompaction {
    SessionAutoCompaction::new(
        SessionCompactionConfig::default(),
        SessionCompactionProtectedPrefix::LeadingSystemAndIdentitySeed,
        COMPACTED_SELF_MODEL_SESSION_PREFIX,
        SESSION_COMPACTION_FOCUS,
    )
}

pub struct SelfModelModule {
    cognition_updates: CognitionLogUpdatedInbox,
    blackboard: BlackboardReader,
    cognition_log: CognitionLogReader,
    memo: Memo,
    llm: LlmAccess,
    session: Session,
    system_prompt: std::sync::OnceLock<String>,
}

impl SelfModelModule {
    pub fn new(
        cognition_updates: CognitionLogUpdatedInbox,
        blackboard: BlackboardReader,
        cognition_log: CognitionLogReader,
        memo: Memo,
        llm: LlmAccess,
        session: Session,
    ) -> Self {
        Self {
            cognition_updates,
            blackboard,
            cognition_log,
            memo,
            llm,
            session,
            system_prompt: std::sync::OnceLock::new(),
        }
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

    fn system_prompt(&self, cx: &nuillu_module::ActivateCx<'_>) -> &str {
        self.system_prompt.get_or_init(|| {
            nuillu_module::format_policy_system_prompt(SYSTEM_PROMPT, cx.core_policies())
        })
    }

    #[tracing::instrument(skip_all, err(Debug, level = "warn"))]
    async fn update_from_current_context(
        &mut self,
        cx: &nuillu_module::ActivateCx<'_>,
    ) -> Result<()> {
        let attention_schema_cognition = self
            .cognition_log
            .unread_events()
            .await
            .into_iter()
            .filter(|record| record.source.module == builtin::attention_schema())
            .collect::<Vec<_>>();
        let unread_memo_logs = self.blackboard.unread_memo_logs().await;
        if attention_schema_cognition.is_empty() && unread_memo_logs.is_empty() {
            return Ok(());
        }

        self.ensure_session_seeded(cx);
        let lutum = self.llm.lutum().await;
        let memo = {
            push_formatted_memo_log_batch(
                &mut self.session,
                &unread_memo_logs,
                cx.now(),
                MEMO_CONTEXT_WINDOW,
            );
            push_formatted_cognition_log_batch(
                &mut self.session,
                &attention_schema_cognition,
                cx.now(),
                COGNITION_CONTEXT_WINDOW,
            );
            self.session.push_user(
                "Update the concise embodied and mental self-state snapshot from current self-relevant working notes and attention-schema cognition. Do not recap the conversation. Write nothing if there is no concrete change or clarification to body, capability, affect, attention, intention, uncertainty, or agency.",
            );
            let result = self
                .session
                .text_turn()
                .collect(&lutum)
                .await
                .context("self-model text turn failed")?;
            cx.compact_and_save(&mut self.session, result.usage).await?;
            result.assistant_text()
        };
        if !memo.trim().is_empty() {
            self.memo.write_cognitive(memo).await;
        }
        Ok(())
    }
}

#[async_trait(?Send)]
impl nuillu_module::StaticModule for SelfModelModule {
    fn id() -> &'static str {
        "self-model"
    }

    fn peer_context() -> Option<&'static str> {
        Some(
            "Self-model forms the current first-person sense of identity, agency, intention, capability, and affective self-state.",
        )
    }
}

#[async_trait(?Send)]
impl Module for SelfModelModule {
    type Batch = SelfModelBatch;

    async fn next_batch(&mut self) -> Result<Self::Batch> {
        SelfModelModule::next_batch(self).await
    }

    async fn activate(
        &mut self,
        cx: &nuillu_module::ActivateCx<'_>,
        batch: &Self::Batch,
    ) -> Result<()> {
        if batch.cognition_updated {
            self.update_from_current_context(cx).await?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prompt_limits_self_model_to_embodied_and_mental_state() {
        assert!(SYSTEM_PROMPT.contains("embodied and mental self-model"));
        assert!(SYSTEM_PROMPT.contains("agent's first-person voice"));
        assert!(SYSTEM_PROMPT.contains("underlying model"));
        assert!(SYSTEM_PROMPT.contains("Do not recap dialogue"));
        assert!(SYSTEM_PROMPT.contains("instead of appending a chronology"));
        assert!(!SYSTEM_PROMPT.contains("You are the self-model module"));
        assert!(!SYSTEM_PROMPT.contains("allocation"));
    }
}
