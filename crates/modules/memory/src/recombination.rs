use anyhow::{Context, Result};
use async_trait::async_trait;
use lutum::{Session, TextStepOutcomeWithTools, ToolResult};
use nuillu_module::{
    AllocationReader, AllocationUpdatedInbox, BlackboardReader, CognitionWriter, LlmAccess, Module,
    render_memory_for_llm,
};
use nuillu_types::{MemoryRank, builtin};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::store::{MemoryRecord, MemoryRetriever, MemoryUsageTarget};

const SYSTEM_PROMPT: &str = r#"You are the memory-recombination module.
You run a REM-like internal dream simulation. Combine recent non-dream cognition with retrieved
memories, identity memory, and core policies. Produce one short associative counterfactual or
scenario that may help future planning or memory reweighting.

This is not a verified fact and not an outward reply. Do not repeat the logs faithfully. Do not
violate core policies. Keep the appended text under 200 Japanese characters or 120 English words.
You must call append_recombination exactly once if there is enough seed material; otherwise finish
without tools."#;

#[lutum::tool_input(name = "append_recombination", output = AppendRecombinationOutput)]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct AppendRecombinationArgs {
    pub text: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct AppendRecombinationOutput {
    pub appended: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema, lutum::Toolset)]
pub enum RecombinationTools {
    AppendRecombination(AppendRecombinationArgs),
}

pub struct MemoryRecombinationModule {
    owner: nuillu_types::ModuleId,
    allocation_updates: AllocationUpdatedInbox,
    allocation: AllocationReader,
    blackboard: BlackboardReader,
    memory: MemoryRetriever,
    cognition: CognitionWriter,
    llm: LlmAccess,
}

impl MemoryRecombinationModule {
    pub fn new(
        allocation_updates: AllocationUpdatedInbox,
        allocation: AllocationReader,
        blackboard: BlackboardReader,
        memory: MemoryRetriever,
        cognition: CognitionWriter,
        llm: LlmAccess,
    ) -> Self {
        Self {
            owner: nuillu_types::ModuleId::new(<Self as Module>::id())
                .expect("memory-recombination id is valid"),
            allocation_updates,
            allocation,
            blackboard,
            memory,
            cognition,
            llm,
        }
    }

    #[tracing::instrument(skip_all, err(Debug, level = "warn"))]
    async fn activate(&mut self, cx: &nuillu_module::ActivateCx<'_>) -> Result<()> {
        let allocation = self.allocation.snapshot().await;
        if allocation.activation_for(&self.owner) == nuillu_blackboard::ActivationRatio::ZERO {
            return Ok(());
        }

        let recent = self
            .blackboard
            .read(|bb| {
                let mut entries = bb
                    .unread_cognition_log_entries(None)
                    .into_iter()
                    .filter(|record| record.source.module != builtin::memory_recombination())
                    .collect::<Vec<_>>();
                entries.sort_by_key(|record| record.index);
                entries
                    .into_iter()
                    .rev()
                    .take(8)
                    .collect::<Vec<_>>()
                    .into_iter()
                    .rev()
                    .collect::<Vec<_>>()
            })
            .await;
        if recent.is_empty() {
            return Ok(());
        }
        let seed = recent
            .iter()
            .map(|record| record.entry.text.trim())
            .filter(|text| !text.is_empty())
            .collect::<Vec<_>>()
            .join("\n");
        let related = self
            .memory
            .search(&seed, 5)
            .await
            .context("search memory for recombination")?;
        let targets = related
            .iter()
            .map(MemoryUsageTarget::from)
            .collect::<Vec<_>>();
        self.memory.record_accesses(&targets).await;

        let mut session = Session::new();
        session.push_system(nuillu_module::format_system_prompt(
            SYSTEM_PROMPT,
            cx.modules(),
            &self.owner,
            cx.identity_memories(),
            cx.core_policies(),
            cx.now(),
        ));
        session.push_user(format_recombination_context(cx, &recent, &related));

        let lutum = self.llm.lutum().await;
        let outcome = session
            .text_turn(&lutum)
            .tools::<RecombinationTools>()
            .available_tools([RecombinationToolsSelector::AppendRecombination])
            .collect()
            .await
            .context("memory-recombination text turn failed")?;

        let TextStepOutcomeWithTools::NeedsTools(round) = outcome else {
            return Ok(());
        };
        let mut results = Vec::<ToolResult>::new();
        for call in round.tool_calls.iter().cloned() {
            let RecombinationToolsCall::AppendRecombination(call) = call;
            let output = self
                .append_recombination(call.input.clone())
                .await
                .context("run append_recombination tool")?;
            results.push(
                call.complete(output)
                    .context("complete append_recombination tool call")?,
            );
        }
        round
            .commit(&mut session, results)
            .context("commit memory-recombination tool round")?;
        Ok(())
    }

    async fn append_recombination(
        &self,
        args: AppendRecombinationArgs,
    ) -> Result<AppendRecombinationOutput> {
        let text = args.text.trim();
        if text.is_empty() {
            return Ok(AppendRecombinationOutput { appended: false });
        }
        self.cognition
            .append(format!(
                "Internal dream simulation, not a verified fact: {text}"
            ))
            .await;
        Ok(AppendRecombinationOutput { appended: true })
    }

    async fn next_batch(&mut self) -> Result<()> {
        let _ = self.allocation_updates.next_item().await?;
        let _ = self.allocation_updates.take_ready_items()?;
        Ok(())
    }
}

fn format_recombination_context(
    cx: &nuillu_module::ActivateCx<'_>,
    recent: &[nuillu_blackboard::CognitionLogEntryRecord],
    memories: &[MemoryRecord],
) -> String {
    let mut out = String::from("Recent non-dream cognition seeds:");
    for record in recent {
        let text = record.entry.text.trim();
        if !text.is_empty() {
            out.push_str("\n- ");
            out.push_str(text);
        }
    }

    out.push_str("\n\nRetrieved memory fragments:");
    if memories.is_empty() {
        out.push_str("\n- none");
    } else {
        for memory in memories {
            out.push_str("\n- [");
            out.push_str(memory.index.as_str());
            out.push_str("] rank=");
            out.push_str(memory_rank_label(memory.rank));
            out.push_str(": ");
            out.push_str(&render_memory_for_llm(
                memory.content.as_str(),
                memory.occurred_at,
                cx.now(),
            ));
        }
    }
    out
}

fn memory_rank_label(rank: MemoryRank) -> &'static str {
    match rank {
        MemoryRank::ShortTerm => "short-term",
        MemoryRank::MidTerm => "mid-term",
        MemoryRank::LongTerm => "long-term",
        MemoryRank::Permanent => "permanent",
        MemoryRank::Identity => "identity",
    }
}

#[async_trait(?Send)]
impl Module for MemoryRecombinationModule {
    type Batch = ();

    fn id() -> &'static str {
        "memory-recombination"
    }

    fn role_description() -> &'static str {
        "Runs REM-like internal memory recombination on allocation guidance and appends source-tagged dream simulations to cognition without treating them as verified facts."
    }

    async fn next_batch(&mut self) -> Result<Self::Batch> {
        MemoryRecombinationModule::next_batch(self).await
    }

    async fn activate(
        &mut self,
        cx: &nuillu_module::ActivateCx<'_>,
        _batch: &Self::Batch,
    ) -> Result<()> {
        MemoryRecombinationModule::activate(self, cx).await
    }
}
