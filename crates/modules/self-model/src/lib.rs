use anyhow::{Context, Result};
use async_trait::async_trait;
use lutum::{Session, StructuredTurnOutcome};
use nuillu_module::{
    BlackboardReader, CognitionLogReader, LlmAccess, Memo, Module, SelfModelInbox, SelfModelRequest,
};
use nuillu_types::builtin;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

mod batch;
pub use batch::NextBatch as SelfModelBatch;

const SYSTEM_PROMPT: &str = r#"You are the self-model module.
Maintain a current first-person self-description from module memos, first-person attention
experiences written by attention-schema to the cognition log, and self-related facts that query or
memory modules have surfaced in their memos.
Stable self-knowledge may be present in retrieved memory memos, but do not claim direct access to
raw hidden memories. Answer explicit self-model questions from this current self model.
Return only raw JSON for the structured object; do not wrap it in Markdown or code fences."#;

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct SelfModelAnswer {
    pub question: String,
    pub answer: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct SelfModelReport {
    pub memo: String,
    pub answers: Vec<SelfModelAnswer>,
}

pub struct SelfModelModule {
    owner: nuillu_module::ModuleId,
    requests: SelfModelInbox,
    blackboard: BlackboardReader,
    cognition_log: CognitionLogReader,
    memo: Memo,
    llm: LlmAccess,
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
            )
        })
    }

    #[tracing::instrument(skip_all, err(Debug, level = "warn"))]
    async fn answer_batch(
        &self,
        cx: &nuillu_module::ActivateCx<'_>,
        requests: Vec<SelfModelRequest>,
    ) -> Result<()> {
        let context = self
            .blackboard
            .read(|bb| {
                serde_json::json!({
                    "memos": bb.memos(),
                })
            })
            .await;
        let cognition_context = self
            .cognition_log
            .snapshot()
            .await
            .logs()
            .iter()
            .filter(|record| record.source.module == builtin::attention_schema())
            .cloned()
            .collect::<Vec<_>>();
        let previous_self_model = self.memo.read().await.unwrap_or_default();
        let questions = requests
            .into_iter()
            .map(|request| request.question)
            .collect::<Vec<_>>();

        let mut session = Session::new();
        session.push_system(self.system_prompt(cx));
        session.push_user(
            serde_json::json!({
                "questions": questions,
                "memo_context": context,
                "attention_schema_cognition_log": cognition_context,
                "previous_self_model": previous_self_model,
            })
            .to_string(),
        );

        let result = session
            .structured_turn::<SelfModelReport>(&self.llm.lutum().await)
            .collect()
            .await
            .context("self-model structured turn failed")?;

        let StructuredTurnOutcome::Structured(report) = result.semantic else {
            anyhow::bail!("self-model structured turn refused");
        };

        let serialized = serde_json::to_string(&report).context("serialize self-model report")?;
        self.memo.write(serialized).await;
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
        "Integrates attention-schema cognition-log entries, peer module memos, and retrieved self-knowledge into a current first-person self-description; answers self-model requests."
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
