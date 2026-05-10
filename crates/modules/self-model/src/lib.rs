use anyhow::{Context, Result};
use async_trait::async_trait;
use lutum::Session;
use nuillu_module::{
    BlackboardReader, CognitionLogReader, LlmAccess, Memo, Module, SelfModelInbox,
    SelfModelRequest, SessionCompactionConfig, compact_session_if_needed, push_unread_memo_logs,
};
use nuillu_types::builtin;

mod batch;
pub use batch::NextBatch as SelfModelBatch;

const SYSTEM_PROMPT: &str = r#"You are the self-model module.
Maintain a current first-person self-description from module memos, first-person attention
experiences written by attention-schema to the cognition log, and self-related facts that query or
memory modules have surfaced in their memos.
Stable self-knowledge may be present in retrieved memory memos, but do not claim direct access to
raw hidden memories. Answer explicit self-model questions from this current self model.
Write the memo as free-form prose. Preserve the current self-description and every explicit
question/answer, but do not encode the memo as JSON, YAML, a code block, or any fixed schema."#;

const COMPACTED_SELF_MODEL_SESSION_PREFIX: &str = "Compacted self-model session history:";
const SESSION_COMPACTION_PROMPT: &str = r#"You compact the self-model module's persistent session history.
Summarize only the prefix transcript you receive. Preserve self-descriptions, self-model questions
and answers, memo-log facts about the agent, attention-schema first-person cognition, uncertainty,
and corrections. Do not invent facts. Return plain text only."#;

pub struct SelfModelModule {
    owner: nuillu_module::ModuleId,
    requests: SelfModelInbox,
    blackboard: BlackboardReader,
    cognition_log: CognitionLogReader,
    memo: Memo,
    llm: LlmAccess,
    session: Session,
    session_compaction: SessionCompactionConfig,
    system_prompt: std::sync::OnceLock<String>,
}

impl SelfModelModule {
    pub fn new(
        requests: SelfModelInbox,
        blackboard: BlackboardReader,
        cognition_log: CognitionLogReader,
        memo: Memo,
        llm: LlmAccess,
    ) -> Self {
        Self {
            owner: nuillu_module::ModuleId::new(<Self as Module>::id())
                .expect("self-model id is valid"),
            requests,
            blackboard,
            cognition_log,
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
    async fn answer_batch(
        &mut self,
        cx: &nuillu_module::ActivateCx<'_>,
        requests: Vec<SelfModelRequest>,
    ) -> Result<()> {
        let unread_memo_logs = self.blackboard.unread_memo_logs().await;
        push_unread_memo_logs(&mut self.session, &unread_memo_logs);
        let cognition_context = self
            .cognition_log
            .snapshot()
            .await
            .logs()
            .iter()
            .filter(|record| record.source.module == builtin::attention_schema())
            .cloned()
            .collect::<Vec<_>>();
        let questions = requests
            .into_iter()
            .map(|request| request.question)
            .collect::<Vec<_>>();

        let system_prompt = self.system_prompt(cx).to_owned();
        self.session.push_ephemeral_system(system_prompt);
        self.session.push_user(
            serde_json::json!({
                "questions": questions,
            })
            .to_string(),
        );
        self.session.push_ephemeral_user(
            serde_json::json!({
                "attention_schema_cognition_log": cognition_context,
            })
            .to_string(),
        );

        let lutum = self.llm.lutum().await;
        let result = self
            .session
            .text_turn(&lutum)
            .collect()
            .await
            .context("self-model text turn failed")?;
        compact_session_if_needed(
            &mut self.session,
            result.usage.input_tokens,
            cx.session_compaction_lutum(),
            self.session_compaction,
            Self::id(),
            COMPACTED_SELF_MODEL_SESSION_PREFIX,
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
impl Module for SelfModelModule {
    type Batch = SelfModelBatch;

    fn id() -> &'static str {
        "self-model"
    }

    fn role_description() -> &'static str {
        "Integrates attention-schema cognition-log entries, peer module memo logs, and retrieved self-knowledge into current first-person self evidence in its memo log; cognition-gate must promote useful self-model facts before speech uses them."
    }

    async fn next_batch(&mut self) -> Result<Self::Batch> {
        SelfModelModule::next_batch(self).await
    }

    async fn activate(
        &mut self,
        cx: &nuillu_module::ActivateCx<'_>,
        batch: &Self::Batch,
    ) -> Result<()> {
        if !batch.requests.is_empty() {
            self.answer_batch(cx, batch.requests.clone()).await?;
        }
        Ok(())
    }
}
