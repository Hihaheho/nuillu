use std::rc::Rc;

use anyhow::{Context, Result};
use async_trait::async_trait;
use lutum::{Session, StructuredTurnOutcome};
use nuillu_blackboard::{Blackboard, BlackboardCommand, PolicyMetaPatch};
use nuillu_module::ports::PortError;
use nuillu_module::{
    AllocationReader, AllocationUpdatedInbox, BlackboardReader, CognitionLogUpdatedInbox,
    InteroceptiveReader, LlmAccess, Module, format_current_attention_guidance,
    push_formatted_memo_log_batch,
};
use nuillu_types::{ModuleId, PolicyIndex, PolicyRank, SignedUnitF32, UnitF32};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::{IndexedPolicy, NewPolicy, PolicyStore, fanout_policy_put};

const SYSTEM_PROMPT: &str = r#"You are the policy module.
Preserve successful or distinctive behavior patterns as new tentative policies.
Only create policies when the trigger and behavior are concrete enough to reuse later.
Never rewrite existing policy records; refined behavior is a new policy."#;

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct PolicyFormationDecision {
    pub candidates: Vec<PolicyCandidate>,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct PolicyCandidate {
    pub trigger: String,
    pub behavior: String,
    pub reason: String,
}

/// Inserts new tentative policies, mirrors metadata onto the blackboard, and
/// fans out indexed writes to replica stores.
#[derive(Clone)]
pub struct PolicyWriter {
    primary_store: Rc<dyn PolicyStore>,
    replicas: Vec<Rc<dyn PolicyStore>>,
    blackboard: Blackboard,
}

impl PolicyWriter {
    pub(crate) fn new(
        primary_store: Rc<dyn PolicyStore>,
        replicas: Vec<Rc<dyn PolicyStore>>,
        blackboard: Blackboard,
    ) -> Self {
        Self {
            primary_store,
            replicas,
            blackboard,
        }
    }

    pub async fn insert(
        &self,
        trigger: String,
        behavior: String,
        decay_secs: i64,
    ) -> Result<PolicyIndex, PortError> {
        let new = NewPolicy {
            trigger: trigger.clone(),
            behavior: behavior.clone(),
            rank: PolicyRank::Tentative,
            expected_reward: SignedUnitF32::ZERO,
            confidence: UnitF32::ZERO,
            value: SignedUnitF32::ZERO,
            reward_tokens: 0,
            decay_remaining_secs: decay_secs,
        };
        let index = self.primary_store.insert(new).await?;
        let indexed = IndexedPolicy {
            index: index.clone(),
            trigger,
            behavior,
            rank: PolicyRank::Tentative,
            expected_reward: SignedUnitF32::ZERO,
            confidence: UnitF32::ZERO,
            value: SignedUnitF32::ZERO,
            reward_tokens: 0,
            decay_remaining_secs: decay_secs,
        };
        fanout_policy_put(&self.replicas, indexed).await;
        self.blackboard
            .apply(BlackboardCommand::UpsertPolicyMetadata {
                index: index.clone(),
                rank_if_new: PolicyRank::Tentative,
                decay_if_new_secs: decay_secs,
                patch: PolicyMetaPatch {
                    rank: Some(PolicyRank::Tentative),
                    expected_reward: Some(SignedUnitF32::ZERO),
                    confidence: Some(UnitF32::ZERO),
                    value: Some(SignedUnitF32::ZERO),
                    reward_tokens: Some(0),
                    decay_remaining_secs: Some(decay_secs),
                    ..Default::default()
                },
            })
            .await;
        Ok(index)
    }
}

pub struct PolicyModule {
    owner: ModuleId,
    cognition_updates: CognitionLogUpdatedInbox,
    allocation_updates: AllocationUpdatedInbox,
    blackboard: BlackboardReader,
    allocation: AllocationReader,
    interoception: InteroceptiveReader,
    writer: PolicyWriter,
    llm: LlmAccess,
    session: Session,
}

impl PolicyModule {
    pub fn new(
        cognition_updates: CognitionLogUpdatedInbox,
        allocation_updates: AllocationUpdatedInbox,
        blackboard: BlackboardReader,
        allocation: AllocationReader,
        interoception: InteroceptiveReader,
        writer: PolicyWriter,
        llm: LlmAccess,
    ) -> Self {
        Self {
            owner: ModuleId::new(<Self as Module>::id()).expect("policy id is valid"),
            cognition_updates,
            allocation_updates,
            blackboard,
            allocation,
            interoception,
            writer,
            llm,
            session: Session::new(),
        }
    }

    async fn activate(&mut self, _cx: &nuillu_module::ActivateCx<'_>) -> Result<()> {
        let memos = self.blackboard.unread_memo_logs().await;
        push_formatted_memo_log_batch(&mut self.session, &memos, _cx.now());
        let allocation = self.allocation.snapshot().await;
        let interoception = self.interoception.snapshot().await;
        self.session.push_ephemeral_system(SYSTEM_PROMPT);
        if let Some(guidance) = format_current_attention_guidance(&allocation) {
            self.session.push_ephemeral_system(guidance);
        }
        self.session.push_ephemeral_system(format!(
            "Current interoception: affect_arousal={:.2}; valence={:.2}; emotion={}",
            interoception.affect_arousal,
            interoception.valence,
            if interoception.emotion.trim().is_empty() {
                "unknown"
            } else {
                interoception.emotion.trim()
            }
        ));
        let cognition = self
            .blackboard
            .read(|bb| {
                bb.cognition_log()
                    .entries()
                    .iter()
                    .map(|entry| entry.text.clone())
                    .collect::<Vec<_>>()
                    .join("\n")
            })
            .await;
        self.session.push_ephemeral_user(format!(
            "Policy formation context for {}:\n{}",
            self.owner, cognition
        ));

        let lutum = self.llm.lutum().await;
        let result = self
            .session
            .structured_turn::<PolicyFormationDecision>(&lutum)
            .collect()
            .await
            .context("policy structured turn failed")?;
        let StructuredTurnOutcome::Structured(decision) = result.semantic else {
            return Ok(());
        };
        for candidate in decision.candidates {
            let trigger = candidate.trigger.trim();
            let behavior = candidate.behavior.trim();
            if trigger.is_empty() || behavior.is_empty() {
                continue;
            }
            self.writer
                .insert(trigger.to_owned(), behavior.to_owned(), 86_400)
                .await?;
        }
        Ok(())
    }
}

#[async_trait(?Send)]
impl Module for PolicyModule {
    type Batch = ();

    fn id() -> &'static str {
        "policy"
    }

    fn role_description() -> &'static str {
        "Creates new tentative trigger/behavior policies from successful or distinctive behavior patterns."
    }

    async fn next_batch(&mut self) -> Result<Self::Batch> {
        tokio::select! {
            result = self.cognition_updates.next_item() => {
                result?;
            }
            result = self.allocation_updates.next_item() => {
                result?;
            }
        }
        Ok(())
    }

    async fn activate(
        &mut self,
        cx: &nuillu_module::ActivateCx<'_>,
        _batch: &Self::Batch,
    ) -> Result<()> {
        PolicyModule::activate(self, cx).await
    }
}
