use anyhow::{Context, Result};
use async_trait::async_trait;
use lutum::{Session, StructuredTurnOutcome};
use nuillu_module::{
    AllocationReader, AllocationUpdatedInbox, BlackboardReader, CognitionLogUpdatedInbox,
    LlmAccess, Module, PolicyWriter, format_current_attention_guidance,
    push_formatted_memo_log_batch,
};
use nuillu_types::ModuleId;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

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

pub struct PolicyModule {
    owner: ModuleId,
    cognition_updates: CognitionLogUpdatedInbox,
    allocation_updates: AllocationUpdatedInbox,
    blackboard: BlackboardReader,
    allocation: AllocationReader,
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
        writer: PolicyWriter,
        llm: LlmAccess,
    ) -> Self {
        Self {
            owner: ModuleId::new(<Self as Module>::id()).expect("policy id is valid"),
            cognition_updates,
            allocation_updates,
            blackboard,
            allocation,
            writer,
            llm,
            session: Session::new(),
        }
    }

    async fn activate(&mut self, cx: &nuillu_module::ActivateCx<'_>) -> Result<()> {
        let memos = self.blackboard.unread_memo_logs().await;
        push_formatted_memo_log_batch(&mut self.session, &memos, cx.now());
        let allocation = self.allocation.snapshot().await;
        self.session.push_ephemeral_system(SYSTEM_PROMPT);
        if let Some(guidance) = format_current_attention_guidance(&allocation) {
            self.session.push_ephemeral_system(guidance);
        }
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
