use std::borrow::Cow;

use anyhow::{Context, Result};
use async_trait::async_trait;
use lutum::{Session, StructuredTurnOutcome};
use nuillu_blackboard::{ActivationRatio, ModuleConfig};
use nuillu_module::{
    AllocationReader, AllocationWriter, AttentionReader, BlackboardReader, LlmAccess, Memo,
    MemoUpdatedInbox, Module,
};
use nuillu_types::{ModelTier, ModuleId};
use schemars::{JsonSchema, Schema, SchemaGenerator};
use serde::{Deserialize, Serialize};

mod batch;

const SYSTEM_PROMPT: &str = r#"You are the attention-controller module.
You wake on memo updates. Use blackboard memos, the attention stream, current allocation, and
registry schema to write guidance-based allocation for registered modules.
Return one allocation entry for every registered module. Use guidance to tell modules what work
should be done now and what other module work should be considered. Do not invent module ids.

Speech output is driven by a typed SpeakRequest from speak-gate to speak, not by allocation
guidance. Do not use speak guidance as a speech protocol. Keep speak and speak-gate active because
idle speak only waits on its inbox and does not call the LLM.
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
    owner: ModuleId,
    updates: MemoUpdatedInbox,
    blackboard: BlackboardReader,
    attention: AttentionReader,
    allocation_reader: AllocationReader,
    allocation_writer: AllocationWriter,
    memo: Memo,
    llm: LlmAccess,
    system_prompt: std::sync::OnceLock<String>,
}

impl AttentionControllerModule {
    pub fn new(
        updates: MemoUpdatedInbox,
        blackboard: BlackboardReader,
        attention: AttentionReader,
        allocation_reader: AllocationReader,
        allocation_writer: AllocationWriter,
        memo: Memo,
        llm: LlmAccess,
    ) -> Self {
        Self {
            owner: ModuleId::new(<Self as Module>::id())
                .expect("attention-controller id is valid"),
            updates,
            blackboard,
            attention,
            allocation_reader,
            allocation_writer,
            memo,
            llm,
            system_prompt: std::sync::OnceLock::new(),
        }
    }

    fn system_prompt(&self, cx: &nuillu_module::ActivateCx<'_>) -> &str {
        self.system_prompt.get_or_init(|| {
            nuillu_module::format_system_prompt(SYSTEM_PROMPT, cx.modules(), &self.owner)
        })
    }

    #[tracing::instrument(skip_all, err(Debug, level = "warn"))]
    async fn activate(&self, cx: &nuillu_module::ActivateCx<'_>) -> Result<()> {
        Self::activate_with(
            &self.attention,
            &self.blackboard,
            &self.allocation_reader,
            &self.allocation_writer,
            &self.memo,
            &self.llm,
            self.system_prompt(cx),
        )
        .await
    }

    #[tracing::instrument(skip_all, err(Debug, level = "warn"))]
    #[allow(clippy::too_many_arguments)]
    async fn activate_with(
        attention_reader: &AttentionReader,
        blackboard_reader: &BlackboardReader,
        allocation_reader: &AllocationReader,
        allocation_writer: &AllocationWriter,
        memo: &Memo,
        llm: &LlmAccess,
        system_prompt: &str,
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
        let registered = allocation_reader
            .registered_module_ids()
            .await
            .into_iter()
            .collect::<std::collections::HashSet<_>>();

        let mut session = Session::new();
        session.push_system(system_prompt);
        session.push_user(
            controller_input(
                serde_json::to_value(&attention).context("serialize attention stream")?,
                blackboard,
                serde_json::to_value(&current).context("serialize current allocation")?,
                controller_schema,
            )
            .to_string(),
        );

        let result = CONTROLLER_DECISION_SCHEMA
            .scope(output_schema, async {
                session
                    .structured_turn::<AllocationDecision>(&llm.lutum().await)
                    .collect()
                    .await
            })
            .await
            .context("attention-controller structured turn failed")?;

        let StructuredTurnOutcome::Structured(decision) = result.semantic else {
            anyhow::bail!("attention-controller structured turn refused");
        };

        let next = apply_decision(current, &registered, decision);

        memo.write(next.memo.clone()).await;
        allocation_writer.set(next.allocation).await;
        Ok(())
    }
}

struct AppliedDecision {
    memo: String,
    allocation: nuillu_blackboard::ResourceAllocation,
}

fn apply_decision(
    current: nuillu_blackboard::ResourceAllocation,
    registered: &std::collections::HashSet<ModuleId>,
    decision: AllocationDecision,
) -> AppliedDecision {
    let mut next = current;
    for entry in decision.allocations {
        let Ok(id) = ModuleId::new(entry.module_id) else {
            tracing::warn!("attention-controller ignored invalid module id");
            continue;
        };
        if !registered.contains(&id) {
            tracing::warn!(module = %id, "attention-controller ignored unregistered module id");
            continue;
        }
        let mut config: ModuleConfig = next.for_module(&id);
        config.guidance = entry.guidance;
        config.tier = entry.tier;
        next.set(id.clone(), config);
        next.set_activation(id, ActivationRatio::from_f64(entry.activation_ratio));
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

fn controller_input(
    attention: serde_json::Value,
    blackboard: serde_json::Value,
    allocation: serde_json::Value,
    controller_schema: serde_json::Value,
) -> serde_json::Value {
    serde_json::json!({
        "attention_stream": attention,
        "blackboard": blackboard,
        "allocation": allocation,
        "controller_schema": controller_schema,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    use nuillu_blackboard::ResourceAllocation;
    use nuillu_types::builtin;

    #[test]
    fn apply_decision_ignores_unregistered_modules_and_clamps_activation_ratio() {
        let mut registered = std::collections::HashSet::new();
        registered.insert(builtin::speak());

        let applied = apply_decision(
            ResourceAllocation::default(),
            &registered,
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
        assert_eq!(
            applied.allocation.activation_for(&builtin::speak()),
            ActivationRatio::ONE
        );
        assert_eq!(speak.guidance, "respond from attention when ready");
        assert_eq!(speak.tier, ModelTier::Premium);
    }

    #[test]
    fn controller_input_includes_blackboard_memos_without_special_protocol() {
        let input = controller_input(
            serde_json::json!([]),
            serde_json::json!({
                "memos": {
                    "speak-gate": "{\"should_speak\":false,\"rationale\":\"waiting for attended route facts\"}"
                }
            }),
            serde_json::json!({}),
            serde_json::json!({"schema": true}),
        );

        assert_eq!(
            input,
            serde_json::json!({
                "attention_stream": [],
                "blackboard": {
                    "memos": {
                        "speak-gate": "{\"should_speak\":false,\"rationale\":\"waiting for attended route facts\"}"
                    }
                },
                "allocation": {},
                "controller_schema": {"schema": true},
            })
        );
    }
}

#[async_trait(?Send)]
impl Module for AttentionControllerModule {
    type Batch = ();

    fn id() -> &'static str {
        "attention-controller"
    }

    fn role_description() -> &'static str {
        "Allocates resources across modules — writes activation, guidance, and tier per registered module on memo updates."
    }

    async fn next_batch(&mut self) -> Result<Self::Batch> {
        AttentionControllerModule::next_batch(self).await
    }

    async fn activate(
        &mut self,
        cx: &nuillu_module::ActivateCx<'_>,
        batch: &Self::Batch,
    ) -> Result<()> {
        let _ = batch;
        AttentionControllerModule::activate(self, cx).await
    }
}
