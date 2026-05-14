use anyhow::{Context, Result};
use async_trait::async_trait;
use lutum::{Session, TextStepOutcomeWithTools, ToolResult};
use nuillu_module::{
    AllocationReader, AllocationUpdatedInbox, BlackboardReader, LlmAccess, Module,
};
use nuillu_types::{MemoryIndex, MemoryRank};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::common::{GetMemoriesOutput, MemoryMetadataContext, memory_record_to_view};
use crate::memory::{
    MemoryConceptInput, MemoryTagInput, memory_concept_from_input, memory_tag_from_input,
};
use crate::store::MemoryCompactor;

const SYSTEM_PROMPT: &str = r#"You are the memory-compaction module.
Inspect memory metadata, fetch source contents when useful, and perform destructive NREM-like
consolidation. Compaction output is one replacement summary memory; source memories are removed
from live retrieval after the replacement is written. Use merge_memories only when multiple source
memories are redundant enough that preserving them separately adds noise. Do not create
memory-to-memory links here; non-destructive relationship work belongs to memory-association. Do
not collapse evidence into a single current fact."#;

#[lutum::tool_input(name = "get_memories", output = GetMemoriesOutput)]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct GetMemoriesArgs {
    pub indexes: Vec<String>,
}

#[lutum::tool_input(name = "merge_memories", output = MergeMemoriesOutput)]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct MergeMemoriesArgs {
    pub source_indexes: Vec<String>,
    pub merged_content: String,
    pub merged_rank: MemoryRank,
    pub decay_secs: i64,
    pub concepts: Vec<MemoryConceptInput>,
    pub tags: Vec<MemoryTagInput>,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct MergeMemoriesOutput {
    pub merged_index: String,
    pub merged_sources: usize,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema, lutum::Toolset)]
pub enum CompactionTools {
    GetMemories(GetMemoriesArgs),
    MergeMemories(MergeMemoriesArgs),
}

pub struct MemoryCompactionModule {
    owner: nuillu_types::ModuleId,
    allocation_updates: AllocationUpdatedInbox,
    allocation: AllocationReader,
    blackboard: BlackboardReader,
    compactor: MemoryCompactor,
    llm: LlmAccess,
    system_prompt: std::sync::OnceLock<String>,
}

impl MemoryCompactionModule {
    pub fn new(
        allocation_updates: AllocationUpdatedInbox,
        allocation: AllocationReader,
        blackboard: BlackboardReader,
        compactor: MemoryCompactor,
        llm: LlmAccess,
    ) -> Self {
        Self {
            owner: nuillu_types::ModuleId::new(<Self as Module>::id())
                .expect("memory-compaction id is valid"),
            allocation_updates,
            allocation,
            blackboard,
            compactor,
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
                cx.core_policies(),
                cx.now(),
            )
        })
    }

    #[tracing::instrument(skip_all, err(Debug, level = "warn"))]
    async fn activate(&self, cx: &nuillu_module::ActivateCx<'_>) -> Result<()> {
        let memory_metadata = self
            .blackboard
            .read(|bb| {
                let mut metadata = bb
                    .memory_metadata()
                    .values()
                    .map(|item| MemoryMetadataContext {
                        index: item.index.to_string(),
                        rank: item.rank,
                        occurred_at: item
                            .occurred_at
                            .map(|at| at.to_rfc3339())
                            .unwrap_or_else(|| "unknown".to_owned()),
                        decay_remaining_secs: item.decay_remaining_secs,
                        access_count: item.access_count,
                    })
                    .collect::<Vec<_>>();
                metadata.sort_by(|left, right| left.index.cmp(&right.index));
                metadata
            })
            .await;
        let allocation = self.allocation.snapshot().await;
        let allocation_guidance = allocation
            .iter()
            .filter_map(|(id, config)| {
                let guidance = config.guidance.trim();
                (!guidance.is_empty()).then(|| (id.to_string(), guidance.to_owned()))
            })
            .collect::<Vec<_>>();

        let mut session = Session::new();
        session.push_system(self.system_prompt(cx));
        session.push_user(format_compaction_context(
            &memory_metadata,
            &allocation_guidance,
        ));

        for _ in 0..6 {
            let lutum = self.llm.lutum().await;
            let outcome = session
                .text_turn(&lutum)
                .tools::<CompactionTools>()
                .available_tools([
                    CompactionToolsSelector::GetMemories,
                    CompactionToolsSelector::MergeMemories,
                ])
                .collect()
                .await
                .context("memory-compaction text turn failed")?;

            match outcome {
                TextStepOutcomeWithTools::Finished(_) => return Ok(()),
                TextStepOutcomeWithTools::FinishedNoOutput(_) => return Ok(()),
                TextStepOutcomeWithTools::NeedsTools(round) => {
                    let mut results: Vec<ToolResult> = Vec::new();
                    for call in round.tool_calls.iter().cloned() {
                        match call {
                            CompactionToolsCall::GetMemories(call) => {
                                let output = self
                                    .get_memories(call.input.clone())
                                    .await
                                    .context("run get_memories tool")?;
                                let result = call
                                    .complete(output)
                                    .context("complete get_memories tool call")?;
                                results.push(result);
                            }
                            CompactionToolsCall::MergeMemories(call) => {
                                let output = self
                                    .merge_memories(call.input.clone())
                                    .await
                                    .context("run merge_memories tool")?;
                                let result = call
                                    .complete(output)
                                    .context("complete merge_memories tool call")?;
                                results.push(result);
                            }
                        }
                    }
                    round
                        .commit(&mut session, results)
                        .context("commit memory-compaction tool round")?;
                }
            }
        }
        Ok(())
    }

    async fn get_memories(&self, args: GetMemoriesArgs) -> Result<GetMemoriesOutput> {
        let indexes = args
            .indexes
            .into_iter()
            .map(MemoryIndex::new)
            .collect::<Vec<_>>();
        let records = self
            .compactor
            .get_many(&indexes)
            .await
            .context("get memories for compaction")?;
        Ok(GetMemoriesOutput {
            memories: records.into_iter().map(memory_record_to_view).collect(),
        })
    }

    async fn merge_memories(&self, args: MergeMemoriesArgs) -> Result<MergeMemoriesOutput> {
        let sources = args
            .source_indexes
            .iter()
            .cloned()
            .map(MemoryIndex::new)
            .collect::<Vec<_>>();
        let source_count = sources.len();
        let index = self
            .compactor
            .write_summary(
                &sources,
                args.merged_content,
                args.merged_rank,
                args.decay_secs,
                args.concepts
                    .into_iter()
                    .map(memory_concept_from_input)
                    .collect(),
                args.tags.into_iter().map(memory_tag_from_input).collect(),
            )
            .await
            .context("merge memories")?;
        Ok(MergeMemoriesOutput {
            merged_index: index.to_string(),
            merged_sources: source_count,
        })
    }

    async fn next_batch(&mut self) -> Result<()> {
        let _ = self.allocation_updates.next_item().await?;
        let _ = self.allocation_updates.take_ready_items()?;
        Ok(())
    }
}

fn format_compaction_context(
    memory_metadata: &[MemoryMetadataContext],
    allocation_guidance: &[(String, String)],
) -> String {
    let mut out = String::from("Memory compaction context.");
    out.push_str("\n\nMemory metadata candidates:");
    if memory_metadata.is_empty() {
        out.push_str("\n- none");
    } else {
        for item in memory_metadata {
            out.push_str(&format!(
                "\n- {}: rank={:?}; occurred_at={}; decay_remaining_secs={}; access_count={}",
                item.index,
                item.rank,
                item.occurred_at,
                item.decay_remaining_secs,
                item.access_count
            ));
        }
    }
    out.push_str("\n\nCurrent compaction guidance:");
    if allocation_guidance.is_empty() {
        out.push_str("\n- none");
    } else {
        for (id, guidance) in allocation_guidance {
            out.push_str(&format!("\n- {id}: {guidance}"));
        }
    }
    out
}

#[async_trait(?Send)]
impl Module for MemoryCompactionModule {
    type Batch = ();

    fn id() -> &'static str {
        "memory-compaction"
    }

    fn role_description() -> &'static str {
        "Destructively merges redundant memory entries and accumulates remember tokens; wakes on allocation guidance, never on raw memos."
    }

    async fn next_batch(&mut self) -> Result<Self::Batch> {
        MemoryCompactionModule::next_batch(self).await
    }

    async fn activate(
        &mut self,
        cx: &nuillu_module::ActivateCx<'_>,
        _batch: &Self::Batch,
    ) -> Result<()> {
        MemoryCompactionModule::activate(self, cx).await
    }
}
