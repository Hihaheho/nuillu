use std::borrow::Cow;

use anyhow::{Context, Result};
use async_trait::async_trait;
use lutum::{Session, StructuredTurnOutcome};
use nuillu_blackboard::{ActivationRatio, ModuleConfig};
use nuillu_module::{
    AllocationReader, AllocationWriter, BlackboardReader, CognitionLogReader, LlmAccess, Memo,
    MemoUpdatedInbox, Module, SessionCompactionConfig, compact_session_if_needed,
    push_unread_memo_logs,
};
use nuillu_types::{ModelTier, ModuleId};
use schemars::{JsonSchema, Schema, SchemaGenerator};
use serde::{Deserialize, Serialize};

mod batch;

const SYSTEM_PROMPT: &str = r#"You are the attention-controller module.
You wake on memo updates. Use blackboard memos, the cognition log, current allocation, and
registry schema to write guidance-based allocation for registered modules.
Return one allocation entry for every registered module. Use guidance to tell modules what work
should be done now and what other module work should be considered. Do not invent module ids.
The memo field is a free-form controller note for the shared memo surface; preserve the reasoning
needed by other modules, but do not encode that memo text as JSON, YAML, a code block, or any fixed
schema.

Speech output is driven by a typed SpeakRequest from speak-gate to speak, not by allocation
guidance. Do not use speak guidance as a speech protocol. Speech is the agent's primary outward
action in its world, not a chat-style response gated on a user request — keep speak and speak-gate
fully active so the agent can address peers, answer questions directed at it, and express
in-world intent. Suppressing speak/speak-gate is suppressing the agent's voice.
Return only raw JSON for the structured object; do not wrap it in Markdown or code fences."#;

const COMPACTED_ATTENTION_CONTROLLER_SESSION_PREFIX: &str =
    "Compacted attention-controller session history:";
const SESSION_COMPACTION_PROMPT: &str = r#"You compact the attention-controller module's persistent session history.
Summarize only the prefix transcript you receive. Preserve memo-log facts, prior allocation
decisions, controller notes, guidance changes, and relevant cognition-log context needed for future
allocation decisions. Do not invent facts. Return plain text only."#;

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
    cognition_log: CognitionLogReader,
    allocation_reader: AllocationReader,
    allocation_writer: AllocationWriter,
    memo: Memo,
    llm: LlmAccess,
    session: Session,
    session_compaction: SessionCompactionConfig,
    system_prompt: std::sync::OnceLock<String>,
}

impl AttentionControllerModule {
    pub fn new(
        updates: MemoUpdatedInbox,
        blackboard: BlackboardReader,
        cognition_log: CognitionLogReader,
        allocation_reader: AllocationReader,
        allocation_writer: AllocationWriter,
        memo: Memo,
        llm: LlmAccess,
    ) -> Self {
        Self {
            owner: ModuleId::new(<Self as Module>::id()).expect("attention-controller id is valid"),
            updates,
            blackboard,
            cognition_log,
            allocation_reader,
            allocation_writer,
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
            )
        })
    }

    #[tracing::instrument(skip_all, err(Debug, level = "warn"))]
    async fn activate(&mut self, cx: &nuillu_module::ActivateCx<'_>) -> Result<()> {
        let system_prompt = self.system_prompt(cx).to_owned();
        self.activate_with(&system_prompt, cx.session_compaction_lutum())
            .await
    }

    #[tracing::instrument(skip_all, err(Debug, level = "warn"))]
    async fn activate_with(
        &mut self,
        system_prompt: &str,
        compaction_lutum: &lutum::Lutum,
    ) -> Result<()> {
        let cognition_log = self.cognition_log.read(|log| log.entries().to_vec()).await;
        let unread_memo_logs = self.blackboard.unread_memo_logs().await;
        push_unread_memo_logs(&mut self.session, &unread_memo_logs);
        let blackboard = self
            .blackboard
            .read(|bb| {
                serde_json::json!({
                    "memory_metadata": bb.memory_metadata(),
                })
            })
            .await;
        let current = self.allocation_reader.snapshot().await;
        let controller_schema = self.allocation_reader.controller_schema_json().await;
        let output_schema =
            Schema::try_from(controller_schema.clone()).context("controller schema is invalid")?;
        let registered = self
            .allocation_reader
            .registered_module_ids()
            .await
            .into_iter()
            .collect::<std::collections::HashSet<_>>();

        self.session.push_ephemeral_system(system_prompt);
        self.session.push_ephemeral_user(
            controller_input(
                serde_json::to_value(&cognition_log).context("serialize cognition log")?,
                blackboard,
                serde_json::to_value(&current).context("serialize current allocation")?,
                controller_schema,
            )
            .to_string(),
        );

        let result = CONTROLLER_DECISION_SCHEMA
            .scope(output_schema, async {
                self.session
                    .structured_turn::<AllocationDecision>(&self.llm.lutum().await)
                    .collect()
                    .await
            })
            .await
            .context("attention-controller structured turn failed")?;
        let input_tokens = result.usage.input_tokens;

        let StructuredTurnOutcome::Structured(decision) = result.semantic else {
            anyhow::bail!("attention-controller structured turn refused");
        };
        compact_session_if_needed(
            &mut self.session,
            input_tokens,
            compaction_lutum,
            self.session_compaction,
            Self::id(),
            COMPACTED_ATTENTION_CONTROLLER_SESSION_PREFIX,
            SESSION_COMPACTION_PROMPT,
        )
        .await;

        let next = apply_decision(current, &registered, decision);

        self.memo.write(next.memo.clone()).await;
        self.allocation_writer.set(next.allocation).await;
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
    cognition_log: serde_json::Value,
    blackboard: serde_json::Value,
    allocation: serde_json::Value,
    controller_schema: serde_json::Value,
) -> serde_json::Value {
    serde_json::json!({
        "cognition_log": cognition_log,
        "blackboard": blackboard,
        "allocation": allocation,
        "controller_schema": controller_schema,
    })
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
                        guidance: "respond from cognition log when ready".into(),
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
        assert_eq!(speak.guidance, "respond from cognition log when ready");
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
                "cognition_log": [],
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
