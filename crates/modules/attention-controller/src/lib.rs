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
use nuillu_types::ModuleId;
use schemars::{JsonSchema, Schema, SchemaGenerator};
use serde::{Deserialize, Serialize};

mod batch;

const SYSTEM_PROMPT: &str = r#"You are the attention-controller module.
You wake on memo updates. Use blackboard memos, the cognition log, the current allocation, and
the registry schema to decide which modules deserve activation right now.

Output shape: a `memo` (free-form controller note for the shared memo surface) plus a `priority`
array. The array lists modules to activate in descending priority order, each entry pairing a
`module_id` (must be a registered module) with a `hint` — one concise sentence saying why that
module needs activation now. Modules you omit receive zero activation; their typed mailboxes still
flow because each module keeps a minimum replica. Position in the array maps to the host-configured
activation table; positions beyond the table fall to zero, so prioritise tightly. Do not invent
module ids and do not duplicate ids.

Speech output is driven by a typed SpeakRequest from speak-gate to speak, not by allocation
priority. Speech is the agent's primary outward action in its world, not a chat-style response
gated on a user request — keep speak and speak-gate near the top of priority so the agent can
address peers, answer questions directed at it, and express in-world intent. Dropping speak or
speak-gate from the priority list is suppressing the agent's voice.

The memo field is a free-form controller note; preserve the reasoning needed by other modules but
do not encode it as JSON, YAML, a code block, or any fixed schema.
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
    pub priority: Vec<PriorityEntry>,
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
pub struct PriorityEntry {
    pub module_id: String,
    pub hint: String,
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
                cx.now(),
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
    let table = next.activation_table().to_vec();

    // Controller's baseline: zero out every registered module. Entries listed
    // in `priority` will overwrite below; everything else stays at 0.0 so the
    // module relies on its replicas-min floor only.
    for id in registered {
        next.set_activation(id.clone(), ActivationRatio::ZERO);
        let mut config: ModuleConfig = next.for_module(id);
        config.guidance.clear();
        next.set(id.clone(), config);
    }

    for (rank, entry) in decision.priority.into_iter().enumerate() {
        let Ok(id) = ModuleId::new(entry.module_id) else {
            tracing::warn!("attention-controller ignored invalid module id");
            continue;
        };
        if !registered.contains(&id) {
            tracing::warn!(module = %id, "attention-controller ignored unregistered module id");
            continue;
        }
        let ratio = table.get(rank).copied().unwrap_or(ActivationRatio::ZERO);
        let mut config: ModuleConfig = next.for_module(&id);
        config.guidance = entry.hint;
        next.set(id.clone(), config);
        next.set_activation(id, ratio);
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
            "priority": {
                "type": "array",
                "items": false,
            },
        },
        "required": ["memo", "priority"],
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
        "Allocates resources across modules — emits a priority-ordered list of modules to activate, each with a one-line hint, on memo updates. Tier is host-fixed; activation_ratio comes from a host-set table indexed by priority position."
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
    fn apply_decision_assigns_table_by_rank_and_zeroes_unlisted() {
        let mut registered = std::collections::HashSet::new();
        registered.insert(builtin::speak());
        registered.insert(builtin::sensory());
        registered.insert(builtin::cognition_gate());

        let mut current = ResourceAllocation::default();
        current.set_activation_table(vec![ActivationRatio::ONE, ActivationRatio::from_f64(0.5)]);
        // Stale activation that the controller should clear because the module
        // is not in the new priority list.
        current.set_activation(builtin::cognition_gate(), ActivationRatio::ONE);

        let applied = apply_decision(
            current,
            &registered,
            AllocationDecision {
                memo: "checked".into(),
                priority: vec![
                    PriorityEntry {
                        module_id: "invented-module".into(),
                        hint: "ignore".into(),
                    },
                    PriorityEntry {
                        module_id: "speak".into(),
                        hint: "respond from cognition log when ready".into(),
                    },
                    PriorityEntry {
                        module_id: "sensory".into(),
                        hint: "keep watching the food bowl".into(),
                    },
                ],
            },
        );

        assert_eq!(applied.memo, "checked");
        // Unregistered module is dropped.
        assert!(
            applied
                .allocation
                .get(&ModuleId::new("invented-module").unwrap())
                .is_none()
        );
        // Speak landed at rank 1 (after the discarded invented-module entry):
        // table[1] = 0.5.
        assert_eq!(
            applied.allocation.activation_for(&builtin::speak()),
            ActivationRatio::from_f64(0.5)
        );
        assert_eq!(
            applied.allocation.for_module(&builtin::speak()).guidance,
            "respond from cognition log when ready"
        );
        // Sensory landed at rank 2; table only has 2 entries so the rest fall
        // to ZERO.
        assert_eq!(
            applied.allocation.activation_for(&builtin::sensory()),
            ActivationRatio::ZERO
        );
        // Cognition-gate was registered but absent from priority — zero.
        assert_eq!(
            applied
                .allocation
                .activation_for(&builtin::cognition_gate()),
            ActivationRatio::ZERO
        );
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
