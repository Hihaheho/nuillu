use std::time::Duration;

use anyhow::{Context, Result};
use async_trait::async_trait;
use lutum::{Session, StructuredTurnOutcome};
use nuillu_blackboard::ModuleConfig;
use nuillu_module::{
    AllocationReader, AllocationWriter, AttentionReader, AttentionStreamUpdatedInbox, LlmAccess,
    Memo, Module, PeriodicInbox,
};
use nuillu_types::{ModelTier, ModuleId, TokenBudget};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

mod batch;

const SYSTEM_PROMPT: &str = r#"You are the attention-controller module.
You may only use the attention stream and current allocation. Return conservative patches for
module enablement, cadence, tier, and context budget. Do not invent module ids."#;

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct AllocationDecision {
    pub memo: String,
    pub patches: Vec<AllocationPatch>,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct AllocationPatch {
    pub module_id: String,
    pub enabled: Option<bool>,
    pub tier: Option<ModelTier>,
    pub period_ms: Option<u64>,
    pub message_only: bool,
    pub context_budget_tokens: Option<u32>,
}

pub struct AttentionControllerModule {
    updates: AttentionStreamUpdatedInbox,
    periodic: Option<PeriodicInbox>,
    attention: AttentionReader,
    allocation_reader: AllocationReader,
    allocation_writer: AllocationWriter,
    memo: Memo,
    llm: LlmAccess,
}

impl AttentionControllerModule {
    pub fn new(
        updates: AttentionStreamUpdatedInbox,
        periodic: Option<PeriodicInbox>,
        attention: AttentionReader,
        allocation_reader: AllocationReader,
        allocation_writer: AllocationWriter,
        memo: Memo,
        llm: LlmAccess,
    ) -> Self {
        Self {
            updates,
            periodic,
            attention,
            allocation_reader,
            allocation_writer,
            memo,
            llm,
        }
    }

    #[tracing::instrument(skip_all, err(Debug, level = "warn"))]
    async fn activate(&self) -> Result<()> {
        Self::activate_with(
            &self.attention,
            &self.allocation_reader,
            &self.allocation_writer,
            &self.memo,
            &self.llm,
        )
        .await
    }

    #[tracing::instrument(skip_all, err(Debug, level = "warn"))]
    async fn activate_with(
        attention_reader: &AttentionReader,
        allocation_reader: &AllocationReader,
        allocation_writer: &AllocationWriter,
        memo: &Memo,
        llm: &LlmAccess,
    ) -> Result<()> {
        let attention = attention_reader
            .read(|stream| stream.entries().to_vec())
            .await;
        let current = allocation_reader.snapshot().await;

        let lutum = llm.lutum().await;
        let mut session = Session::new(lutum);
        session.push_system(SYSTEM_PROMPT);
        session.push_user(
            serde_json::json!({
                "attention_stream": attention,
                "allocation": current,
            })
            .to_string(),
        );

        let result = session
            .structured_turn::<AllocationDecision>()
            .collect()
            .await
            .context("attention-controller structured turn failed")?;

        let StructuredTurnOutcome::Structured(decision) = result.semantic else {
            anyhow::bail!("attention-controller structured turn refused");
        };

        let mut next = current;
        for patch in decision.patches {
            let Ok(id) = ModuleId::new(patch.module_id) else {
                tracing::warn!("attention-controller ignored invalid module id");
                continue;
            };
            let mut config: ModuleConfig = next.for_module(&id);
            if let Some(enabled) = patch.enabled {
                config.enabled = enabled;
            }
            if let Some(tier) = patch.tier {
                config.tier = tier;
            }
            if patch.message_only {
                config.period = None;
            } else if let Some(ms) = patch.period_ms {
                config.period = Some(Duration::from_millis(ms));
            }
            if let Some(tokens) = patch.context_budget_tokens {
                config.context_budget = TokenBudget::new(tokens);
            }
            next.set(id, config);
        }

        memo.write(decision.memo).await;
        allocation_writer.set(next).await;
        Ok(())
    }

    async fn run_loop(&mut self) -> Result<()> {
        loop {
            self.next_batch().await?;
            let _ = self.activate().await;
        }
    }
}

#[async_trait(?Send)]
impl Module for AttentionControllerModule {
    async fn run(&mut self) {
        if let Err(error) = self.run_loop().await {
            tracing::debug!(?error, "attention-controller module loop stopped");
        }
    }
}
