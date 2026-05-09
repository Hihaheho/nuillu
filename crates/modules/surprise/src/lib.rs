use anyhow::{Context, Result};
use async_trait::async_trait;
use lutum::{Session, StructuredTurnOutcome};
use nuillu_module::{
    AllocationReader, AttentionReader, AttentionStreamUpdatedInbox, BlackboardReader, LlmAccess,
    Memo, MemoryImportance, MemoryRequest, MemoryRequestMailbox, Module,
};
use nuillu_types::builtin;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

mod batch;

const SYSTEM_PROMPT: &str = r#"You are the surprise module.
Detect unexpected attention events. If a predict memo is present, frame the assessment as
divergence from pending predictions. If no predict memo is present, judge novelty against recent
attention history. Do not generate forward predictions. Request memory only when the event is
significant enough to preserve. Return only raw JSON for the structured object; do not wrap it in
Markdown or code fences."#;

const RECENT_ATTENTION_LIMIT: usize = 12;

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
    updates: AttentionStreamUpdatedInbox,
    attention: AttentionReader,
    allocation: AllocationReader,
    blackboard: BlackboardReader,
    memory_requests: MemoryRequestMailbox,
    memo: Memo,
    llm: LlmAccess,
    system_prompt: std::sync::OnceLock<String>,
}

impl SurpriseModule {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        updates: AttentionStreamUpdatedInbox,
        attention: AttentionReader,
        allocation: AllocationReader,
        blackboard: BlackboardReader,
        memory_requests: MemoryRequestMailbox,
        memo: Memo,
        llm: LlmAccess,
    ) -> Self {
        Self {
            owner: nuillu_types::ModuleId::new(<Self as Module>::id())
                .expect("surprise id is valid"),
            updates,
            attention,
            allocation,
            blackboard,
            memory_requests,
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
    async fn activate(&self, cx: &nuillu_module::ActivateCx<'_>) -> Result<()> {
        let recent_attention = self
            .attention
            .read(|stream| {
                let entries = stream.entries();
                let start = entries.len().saturating_sub(RECENT_ATTENTION_LIMIT);
                entries[start..].to_vec()
            })
            .await;
        let (predict_memo, memos) = self
            .blackboard
            .read(|bb| {
                (
                    bb.memo(&builtin::predict()).map(ToOwned::to_owned),
                    bb.memos().clone(),
                )
            })
            .await;
        let allocation = self.allocation.snapshot().await;

        let mut session = Session::new();
        session.push_system(self.system_prompt(cx));
        session.push_user(
            serde_json::json!({
                "recent_attention": recent_attention,
                "predict_memo": predict_memo,
                "memos": memos,
                "allocation": allocation,
            })
            .to_string(),
        );

        let result = session
            .structured_turn::<SurpriseAssessment>(&self.llm.lutum().await)
            .collect()
            .await
            .context("surprise structured turn failed")?;

        let StructuredTurnOutcome::Structured(assessment) = result.semantic else {
            anyhow::bail!("surprise structured turn refused");
        };

        if assessment.significant
            && let Some(request) = &assessment.memory_request
            && !request.content.trim().is_empty()
        {
            let _ = self
                .memory_requests
                .publish(MemoryRequest {
                    content: request.content.trim().to_owned(),
                    importance: request.importance,
                    reason: request.reason.trim().to_owned(),
                })
                .await;
        }

        let serialized = serde_json::to_string(&assessment).context("serialize surprise memo")?;
        self.memo.write(serialized).await;
        Ok(())
    }
}

#[async_trait(?Send)]
impl Module for SurpriseModule {
    type Batch = ();

    fn id() -> &'static str {
        "surprise"
    }

    fn role_description() -> &'static str {
        "Detects unexpected attention events by comparing new entries against predict's memo or recent attention; can request memory preservation."
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
