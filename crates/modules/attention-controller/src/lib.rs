use std::borrow::Cow;
use std::collections::HashMap;

use anyhow::{Context, Result};
use async_trait::async_trait;
use lutum::{Session, StructuredTurnOutcome};
use nuillu_blackboard::{ActivationRatio, ModuleConfig};
use nuillu_module::{
    ActivationGate, AllocationReader, AllocationWriter, AttentionReader, BlackboardReader,
    LlmAccess, Memo, MemoUpdatedInbox, Module,
};
use nuillu_types::{ModelTier, ModuleId, ReplicaCapRange};
use schemars::{JsonSchema, Schema, SchemaGenerator};
use serde::{Deserialize, Serialize};

mod batch;

const SYSTEM_PROMPT: &str = r#"You are the attention-controller module.
You wake only on memo updates. Use blackboard memos, the attention stream, current allocation,
and registry schema to write guidance-based allocation for registered modules.
Return one allocation entry for every registered module. Use guidance to tell modules what work
should be done now and what other module work should be considered. Do not invent module ids.
Return only raw JSON for the structured object; do not wrap it in Markdown or code fences."#;

tokio::task_local! {
    static CONTROLLER_DECISION_SCHEMA: Schema;
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AllocationDecision {
    pub memo: String,
    pub allocations: Vec<AllocationEntry>,
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
pub struct AllocationEntry {
    pub module_id: String,
    pub activation_ratio: f64,
    pub guidance: String,
    pub tier: ModelTier,
}

pub struct AttentionControllerModule {
    updates: MemoUpdatedInbox,
    gate: ActivationGate,
    blackboard: BlackboardReader,
    attention: AttentionReader,
    allocation_reader: AllocationReader,
    allocation_writer: AllocationWriter,
    memo: Memo,
    llm: LlmAccess,
}

impl AttentionControllerModule {
    pub fn new(
        updates: MemoUpdatedInbox,
        gate: ActivationGate,
        blackboard: BlackboardReader,
        attention: AttentionReader,
        allocation_reader: AllocationReader,
        allocation_writer: AllocationWriter,
        memo: Memo,
        llm: LlmAccess,
    ) -> Self {
        Self {
            updates,
            gate,
            blackboard,
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
            &self.blackboard,
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
        blackboard_reader: &BlackboardReader,
        allocation_reader: &AllocationReader,
        allocation_writer: &AllocationWriter,
        memo: &Memo,
        llm: &LlmAccess,
    ) -> Result<()> {
        let attention = attention_reader
            .read(|stream| stream.entries().to_vec())
            .await;
        let blackboard = blackboard_reader
            .read(|bb| {
                serde_json::json!({
                    "memos": bb.memos(),
                    "memory_metadata": bb.memory_metadata(),
                })
            })
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
                "blackboard": blackboard,
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
            self.activate().await?;
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
    for entry in decision.allocations {
        let Ok(id) = ModuleId::new(entry.module_id) else {
            tracing::warn!("attention-controller ignored invalid module id");
            continue;
        };
        let Some(_range) = caps.get(&id) else {
            tracing::warn!(module = %id, "attention-controller ignored unregistered module id");
            continue;
        };
        let mut config: ModuleConfig = next.for_module(&id);
        config.activation_ratio = ActivationRatio::from_f64(entry.activation_ratio);
        config.guidance = entry.guidance;
        config.tier = entry.tier;
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
            "allocations": {
                "type": "array",
                "items": false,
            },
        },
        "required": ["memo", "allocations"],
    }))
    .expect("fallback allocation decision schema must be a JSON object")
}

#[cfg(test)]
mod tests {
    use super::*;

    use nuillu_blackboard::ResourceAllocation;
    use nuillu_types::builtin;

    #[test]
    fn apply_decision_ignores_unregistered_modules_and_clamps_activation_ratio() {
        let mut caps = HashMap::new();
        caps.insert(builtin::speak(), ReplicaCapRange { min: 0, max: 1 });

        let applied = apply_decision(
            ResourceAllocation::default(),
            &caps,
            AllocationDecision {
                memo: "checked".into(),
                allocations: vec![
                    AllocationEntry {
                        module_id: "invented-module".into(),
                        activation_ratio: 1.0,
                        guidance: "ignore".into(),
                        tier: ModelTier::Premium,
                    },
                    AllocationEntry {
                        module_id: "speak".into(),
                        activation_ratio: 9.0,
                        guidance: "respond from attention when ready".into(),
                        tier: ModelTier::Premium,
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
        assert_eq!(speak.activation_ratio, ActivationRatio::ONE);
        assert_eq!(speak.guidance, "respond from attention when ready");
        assert_eq!(speak.tier, ModelTier::Premium);
    }
}

#[async_trait(?Send)]
impl Module for AttentionControllerModule {
    async fn run(&mut self) {
        if let Err(error) = self.run_loop().await {
            panic!("attention-controller module failed: {error:#}");
        }
    }
}
