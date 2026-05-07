use std::borrow::Cow;
use std::collections::HashMap;
use std::time::Duration;

use anyhow::{Context, Result};
use async_trait::async_trait;
use lutum::{Session, StructuredTurnOutcome};
use nuillu_blackboard::ModuleConfig;
use nuillu_module::{
    ActivationGate, AllocationReader, AllocationWriter, AttentionReader,
    AttentionStreamUpdatedInbox, LlmAccess, Memo, Module, PeriodicInbox,
};
use nuillu_types::{ModelTier, ModuleId, ReplicaCapRange, TokenBudget};
use schemars::{JsonSchema, Schema, SchemaGenerator};
use serde::{Deserialize, Serialize};

mod batch;

const SYSTEM_PROMPT: &str = r#"You are the attention-controller module.
You may only use the attention stream and current allocation. Return conservative patches for
module replica count, cadence, tier, and context budget. Do not invent module ids.
Every patch field must be present; use null for optional fields you are not changing."#;

tokio::task_local! {
    static CONTROLLER_DECISION_SCHEMA: Schema;
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AllocationDecision {
    pub memo: String,
    pub patches: Vec<AllocationPatch>,
}

impl JsonSchema for AllocationDecision {
    fn inline_schema() -> bool {
        true
    }

    fn schema_name() -> Cow<'static, str> {
        "AllocationDecision".into()
    }

    fn schema_id() -> Cow<'static, str> {
        "nuillu_attention_controller::AllocationDecision.dynamic".into()
    }

    fn json_schema(_generator: &mut SchemaGenerator) -> Schema {
        CONTROLLER_DECISION_SCHEMA
            .try_with(Clone::clone)
            .unwrap_or_else(|_| fallback_allocation_decision_schema())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct AllocationPatch {
    pub module_id: String,
    pub replicas: Option<u8>,
    pub tier: Option<ModelTier>,
    pub period_ms: Option<u64>,
    pub message_only: bool,
    pub context_budget_tokens: Option<u32>,
}

pub struct AttentionControllerModule {
    updates: AttentionStreamUpdatedInbox,
    periodic: Option<PeriodicInbox>,
    gate: ActivationGate,
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
        gate: ActivationGate,
        attention: AttentionReader,
        allocation_reader: AllocationReader,
        allocation_writer: AllocationWriter,
        memo: Memo,
        llm: LlmAccess,
    ) -> Self {
        Self {
            updates,
            periodic,
            gate,
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
        let controller_schema = allocation_reader.controller_schema_json().await;
        let output_schema =
            Schema::try_from(controller_schema.clone()).context("controller schema is invalid")?;
        let replica_caps = allocation_reader.replica_caps().await;

        let lutum = llm.lutum().await;
        let mut session = Session::new(lutum);
        session.push_system(SYSTEM_PROMPT);
        session.push_user(
            serde_json::json!({
                "attention_stream": attention,
                "allocation": current,
                "controller_schema": controller_schema,
            })
            .to_string(),
        );

        let result = CONTROLLER_DECISION_SCHEMA
            .scope(output_schema, async {
                session
                    .structured_turn::<AllocationDecision>()
                    .collect()
                    .await
            })
            .await
            .context("attention-controller structured turn failed")?;

        let StructuredTurnOutcome::Structured(decision) = result.semantic else {
            anyhow::bail!("attention-controller structured turn refused");
        };

        let next = apply_decision(current, &replica_caps, decision);

        memo.write(next.memo.clone()).await;
        allocation_writer.set(next.allocation).await;
        Ok(())
    }

    async fn run_loop(&mut self) -> Result<()> {
        loop {
            self.next_batch().await?;
            let _ = self.activate().await;
        }
    }
}

struct AppliedDecision {
    memo: String,
    allocation: nuillu_blackboard::ResourceAllocation,
}

fn apply_decision(
    current: nuillu_blackboard::ResourceAllocation,
    caps: &HashMap<ModuleId, ReplicaCapRange>,
    decision: AllocationDecision,
) -> AppliedDecision {
    let mut next = current;
    for patch in decision.patches {
        let Ok(id) = ModuleId::new(patch.module_id) else {
            tracing::warn!("attention-controller ignored invalid module id");
            continue;
        };
        let Some(range) = caps.get(&id) else {
            tracing::warn!(module = %id, "attention-controller ignored unregistered module id");
            continue;
        };
        let mut config: ModuleConfig = next.for_module(&id);
        if let Some(replicas) = patch.replicas {
            config.replicas = range.clamp(replicas);
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

    AppliedDecision {
        memo: decision.memo,
        allocation: next,
    }
}

fn fallback_allocation_decision_schema() -> Schema {
    Schema::try_from(serde_json::json!({
        "type": "object",
        "additionalProperties": false,
        "properties": {
            "memo": {
                "type": "string",
            },
            "patches": {
                "type": "array",
                "items": false,
            },
        },
        "required": ["memo", "patches"],
    }))
    .expect("fallback allocation decision schema must be a JSON object")
}

#[cfg(test)]
mod tests {
    use super::*;

    use nuillu_blackboard::ResourceAllocation;
    use nuillu_types::builtin;

    #[test]
    fn apply_decision_ignores_unregistered_modules_and_clamps_replicas() {
        let mut caps = HashMap::new();
        caps.insert(builtin::speak(), ReplicaCapRange { min: 0, max: 1 });

        let applied = apply_decision(
            ResourceAllocation::default(),
            &caps,
            AllocationDecision {
                memo: "checked".into(),
                patches: vec![
                    AllocationPatch {
                        module_id: "invented-module".into(),
                        replicas: Some(1),
                        tier: Some(ModelTier::Premium),
                        period_ms: Some(1),
                        message_only: false,
                        context_budget_tokens: Some(1),
                    },
                    AllocationPatch {
                        module_id: "speak".into(),
                        replicas: Some(9),
                        tier: None,
                        period_ms: None,
                        message_only: true,
                        context_budget_tokens: None,
                    },
                ],
            },
        );

        assert_eq!(applied.memo, "checked");
        assert!(
            applied
                .allocation
                .get(&ModuleId::new("invented-module").unwrap())
                .is_none()
        );
        let speak = applied.allocation.for_module(&builtin::speak());
        assert_eq!(speak.replicas, 1);
        assert_eq!(speak.period, None);
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
