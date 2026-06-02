use anyhow::{Context, Result};
use async_trait::async_trait;
use lutum::Session;
use nuillu_module::{
    AllocationReader, BlackboardReader, CognitionLogReader, CognitionLogUpdatedInbox, LlmAccess,
    LlmContextWindow, Memo, Module, SessionAutoCompaction, SessionCompactionConfig,
    SessionCompactionProtectedPrefix, ensure_persistent_session_seeded,
    push_formatted_cognition_log_batch, push_formatted_memo_log_batch,
};
use nuillu_types::builtin;

mod batch;
pub use batch::NextBatch as SelfModelBatch;

const SYSTEM_PROMPT: &str = r#"Update an agent's self-model memo from its recent history.
You will receive working notes, first-person attention experiences, self-related remembered facts,
and a request for the next self-model memo. Infer what this agent should next believe about itself.
Use loaded identity memories and self-related notes as the agent's own identity and abilities. Write
established self-facts in the agent's first-person voice. Do not identify as the underlying model,
provider, runtime, or an outside observer of the agent.
Stable self-knowledge may be present in remembered facts, but do not claim direct access to raw
hidden memories.
Write the memo as free-form prose. Preserve the current self-description and every explicit
question/answer, but do not encode the memo as JSON, YAML, a code block, or any fixed schema."#;

const COMPACTED_SELF_MODEL_SESSION_PREFIX: &str = "Compacted self-model session history:";
const MEMO_CONTEXT_WINDOW: LlmContextWindow = LlmContextWindow::new(8, 1_200, 4_800);
const COGNITION_CONTEXT_WINDOW: LlmContextWindow = LlmContextWindow::new(8, 600, 3_000);
const SESSION_COMPACTION_PROMPT: &str = r#"You compact the self-model module's persistent session history.
Summarize only the prefix transcript you receive. Preserve self-descriptions, self-model questions
and answers, memo-log facts about the agent, attention-schema first-person cognition, uncertainty,
and corrections. Do not invent facts. Return plain text only."#;

pub fn session_auto_compaction() -> SessionAutoCompaction {
    SessionAutoCompaction::new(
        SessionCompactionConfig::default(),
        SessionCompactionProtectedPrefix::LeadingSystemAndIdentitySeed,
        COMPACTED_SELF_MODEL_SESSION_PREFIX,
        SESSION_COMPACTION_PROMPT,
    )
}

pub struct SelfModelModule {
    owner: nuillu_module::ModuleId,
    cognition_updates: CognitionLogUpdatedInbox,
    allocation: AllocationReader,
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
        allocation: AllocationReader,
        blackboard: BlackboardReader,
        cognition_log: CognitionLogReader,
        memo: Memo,
        llm: LlmAccess,
        session: Session,
    ) -> Self {
        Self {
            owner: nuillu_module::ModuleId::new(<Self as Module>::id())
                .expect("self-model id is valid"),
            cognition_updates,
            allocation,
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
            nuillu_module::format_identity_system_prompt(
                SYSTEM_PROMPT,
                cx.identity_memories(),
                cx.core_policies(),
                cx.now(),
            )
        })
    }

    #[tracing::instrument(skip_all, err(Debug, level = "warn"))]
    async fn answer_from_guidance(&mut self, cx: &nuillu_module::ActivateCx<'_>) -> Result<()> {
        let guidance = self
            .allocation
            .snapshot()
            .await
            .for_module(&self.owner)
            .guidance;
        if guidance.trim().is_empty() {
            return Ok(());
        }
        let attention_schema_cognition = self
            .cognition_log
            .unread_events()
            .await
            .into_iter()
            .filter(|record| record.source.module == builtin::attention_schema())
            .collect::<Vec<_>>();
        self.ensure_session_seeded(cx);
        let unread_memo_logs = self.blackboard.unread_memo_logs().await;
        let lutum = self.llm.lutum().await;
        let memo = {
            push_formatted_memo_log_batch(
                &mut self.session,
                &unread_memo_logs,
                cx.now(),
                MEMO_CONTEXT_WINDOW,
            );
            self.session.push_user(format!(
                "Request for the next self-model memo:\n{}",
                guidance.trim()
            ));
            push_formatted_cognition_log_batch(
                &mut self.session,
                &attention_schema_cognition,
                cx.now(),
                COGNITION_CONTEXT_WINDOW,
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
            self.memo.write(memo).await;
        }
        Ok(())
    }
}

#[async_trait(?Send)]
impl Module for SelfModelModule {
    type Batch = SelfModelBatch;

    fn id() -> &'static str {
        "self-model"
    }

    fn peer_context() -> Option<&'static str> {
        Some(
            "Self-model forms the current first-person sense of identity, agency, intention, capability, and affective self-state.",
        )
    }

    fn allocation_hint() -> Option<&'static str> {
        Some(
            "Raise self-model when identity, agency, intention, capability, or felt self-state is at issue. Keep it low for plain external facts or responses that do not need self-understanding.",
        )
    }

    async fn next_batch(&mut self) -> Result<Self::Batch> {
        SelfModelModule::next_batch(self).await
    }

    async fn activate(
        &mut self,
        cx: &nuillu_module::ActivateCx<'_>,
        batch: &Self::Batch,
    ) -> Result<()> {
        if batch.cognition_updated {
            self.answer_from_guidance(cx).await?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prompt_frames_self_model_as_agent_history_task() {
        assert!(SYSTEM_PROMPT.contains("agent's self-model memo"));
        assert!(SYSTEM_PROMPT.contains("agent's first-person voice"));
        assert!(SYSTEM_PROMPT.contains("underlying model"));
        assert!(!SYSTEM_PROMPT.contains("You are the self-model module"));
        assert!(!SYSTEM_PROMPT.contains("allocation-controller"));
    }
}
