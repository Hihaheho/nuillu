use anyhow::{Context, Result};
use async_trait::async_trait;
use lutum::{ModelInput, TextStepOutcomeWithTools, ToolResult};
use nuillu_module::{BlackboardReader, InteroceptiveUpdatedInbox, LlmAccess, Module};
use nuillu_types::{MemoryIndex, MemoryRank};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::common::{GetMemoriesOutput, MemoryMetadataContext, memory_record_to_view};
use crate::memory::{
    MemoryConceptInput, MemoryTagInput, SHORT_TERM_MEMORY_DECAY_SECS, memory_concept_from_input,
    memory_tag_from_input,
};
use crate::store::MemoryCompactor;

const SYSTEM_PROMPT: &str = r#"You are the memory-compaction module.
Inspect candidate memories, fetch source contents when useful, and consolidate memories whose
contents overlap enough that preserving them separately adds retrieval noise. Compatible variations
around the same subject, object, preference, or procedure can be merged when one concise summary
preserves the useful distinctions. Keep memories separate when they contain unrelated subjects,
conflicting evidence, or details that would be lost in a summary. Compaction output is one
replacement summary memory; source memories are removed from live retrieval after the replacement is
written. Do not create memory-to-memory links here. Do not collapse evidence into a single current
fact.

The candidate list is the maintenance work item. When multiple candidates are present, inspect their
contents with get_memories before deciding whether to merge or leave them unchanged.

Tool input rules: merge_memories takes source_indexes, merged_content, concepts, and tags only. The
runtime chooses the merged rank, decay, and storage metadata."#;

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
    #[serde(default)]
    pub concepts: Vec<MemoryConceptInput>,
    #[serde(default)]
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
    interoception_updates: InteroceptiveUpdatedInbox,
    blackboard: BlackboardReader,
    compactor: MemoryCompactor,
    llm: LlmAccess,
    system_prompt: std::sync::OnceLock<String>,
}

impl MemoryCompactionModule {
    pub fn new(
        interoception_updates: InteroceptiveUpdatedInbox,
        blackboard: BlackboardReader,
        compactor: MemoryCompactor,
        llm: LlmAccess,
    ) -> Self {
        Self {
            owner: nuillu_types::ModuleId::new(<Self as Module>::id())
                .expect("memory-compaction id is valid"),
            interoception_updates,
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
                cx.peer_contexts(),
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
                    })
                    .collect::<Vec<_>>();
                metadata.sort_by(|left, right| left.index.cmp(&right.index));
                metadata
            })
            .await;

        let mut input = ModelInput::new()
            .system(self.system_prompt(cx))
            .user(format_compaction_context(&memory_metadata))
            .user(nuillu_module::OPTIONAL_FUNCTION_CALL_REMINDER);

        for _ in 0..6 {
            let lutum = self.llm.lutum().await;
            let outcome = lutum
                .text_turn(input.clone())
                .tools::<CompactionTools>()
                .available_tools([
                    CompactionToolsSelector::GetMemories,
                    CompactionToolsSelector::MergeMemories,
                ])
                .collect_controlled_with(nuillu_module::AbortOnAvailableToolNameInText::new())
                .await
                .context("memory-compaction text turn failed")?;

            match outcome {
                TextStepOutcomeWithTools::Finished(_) => return Ok(()),
                TextStepOutcomeWithTools::FinishedNoOutput(_) => return Ok(()),
                TextStepOutcomeWithTools::NeedsTools(round) => {
                    let mut results: Vec<ToolResult> = Vec::new();
                    nuillu_module::emit_trace_tool_calls(&round.tool_calls);
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
                        .commit_into(&mut input, results)
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
        let (merged_rank, decay_secs) = self.summary_runtime_metadata(&sources).await;
        let index = self
            .compactor
            .write_summary(
                &sources,
                args.merged_content,
                merged_rank,
                decay_secs,
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

    async fn summary_runtime_metadata(&self, sources: &[MemoryIndex]) -> (MemoryRank, i64) {
        self.blackboard
            .read(|bb| {
                let metadata = bb.memory_metadata();
                let rank = sources
                    .iter()
                    .filter_map(|source| metadata.get(source).map(|item| item.rank))
                    .max_by_key(|rank| memory_rank_strength(*rank))
                    .unwrap_or(MemoryRank::ShortTerm);
                let decay_secs = sources
                    .iter()
                    .filter_map(|source| metadata.get(source).map(|item| item.decay_remaining_secs))
                    .max()
                    .unwrap_or(SHORT_TERM_MEMORY_DECAY_SECS);
                (rank, decay_secs)
            })
            .await
    }

    async fn next_batch(&mut self) -> Result<()> {
        let _ = self.interoception_updates.next_item().await?;
        let _ = self.interoception_updates.take_ready_items()?;
        Ok(())
    }
}

fn format_compaction_context(memory_metadata: &[MemoryMetadataContext]) -> String {
    let mut out = String::from("Memory compaction context.");
    out.push_str("\n\nMemory candidates:");
    if memory_metadata.is_empty() {
        out.push_str("\n- none");
    } else {
        for item in memory_metadata {
            out.push_str(&format!(
                "\n- {}: rank={:?}; occurred_at={}",
                item.index, item.rank, item.occurred_at
            ));
        }
    }
    out
}

fn memory_rank_strength(rank: MemoryRank) -> u8 {
    match rank {
        MemoryRank::ShortTerm => 0,
        MemoryRank::MidTerm => 1,
        MemoryRank::LongTerm => 2,
        MemoryRank::Permanent => 3,
        MemoryRank::Identity => 4,
    }
}

#[async_trait(?Send)]
impl Module for MemoryCompactionModule {
    type Batch = ();

    fn id() -> &'static str {
        "memory-compaction"
    }

    fn peer_context() -> Option<&'static str> {
        None
    }

    fn allocation_hint() -> Option<&'static str> {
        Some(
            "Raise memory-compaction during NREM-like maintenance when memories should be consolidated or redundant memory content should be reduced. Keep it low during active perception, speech, direct recall, or when there is no consolidation pressure.",
        )
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn merge_memories_schema_does_not_expose_runtime_metadata() {
        let schema = serde_json::to_value(schemars::schema_for!(MergeMemoriesArgs))
            .expect("merge memories schema should serialize");

        assert_eq!(schema.pointer("/properties/merged_rank"), None);
        assert_eq!(schema.pointer("/properties/decay_secs"), None);
        assert_eq!(schema.pointer("/properties/occurred_at"), None);
        assert_eq!(
            schema.pointer("/properties/concepts/items/type"),
            Some(&serde_json::json!("string"))
        );
        assert_eq!(
            schema.pointer("/properties/tags/items/type"),
            Some(&serde_json::json!("string"))
        );
    }

    #[test]
    fn compaction_context_exposes_only_candidate_identity() {
        let context = format_compaction_context(&[MemoryMetadataContext {
            index: "mem-1".to_owned(),
            rank: MemoryRank::ShortTerm,
            occurred_at: "2026-06-07T00:00:00Z".to_owned(),
        }]);

        assert_eq!(
            context,
            "Memory compaction context.\n\nMemory candidates:\n- mem-1: rank=ShortTerm; occurred_at=2026-06-07T00:00:00Z"
        );
        assert!(!context.contains("guidance"));
        assert!(!context.contains("decay_remaining_secs"));
        assert!(!context.contains("access_count"));
        assert!(!context.contains("reinforcement_count"));
    }
}
