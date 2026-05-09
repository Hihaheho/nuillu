use anyhow::{Context, Result};
use async_trait::async_trait;
use lutum::{Session, TextStepOutcomeWithTools, ToolResult};
use nuillu_module::{
    AllocationReader, AllocationUpdatedInbox, BlackboardReader, LlmAccess, MemoryCompactor, Module,
};
use nuillu_types::{MemoryIndex, MemoryRank};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

mod batch;

const SYSTEM_PROMPT: &str = r#"You are the memory-compaction module.
Inspect memory metadata, fetch source contents when useful, and merge redundant memories. Merging
should preserve the durable content while deleting source entries."#;

#[lutum::tool_input(name = "get_memories", output = GetMemoriesOutput)]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct GetMemoriesArgs {
    pub indexes: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct MemoryContentView {
    pub index: String,
    pub content: String,
    pub rank: MemoryRank,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct GetMemoriesOutput {
    pub memories: Vec<MemoryContentView>,
}

#[lutum::tool_input(name = "merge_memories", output = MergeMemoriesOutput)]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct MergeMemoriesArgs {
    pub source_indexes: Vec<String>,
    pub merged_content: String,
    pub merged_rank: MemoryRank,
    pub decay_secs: i64,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct MergeMemoriesOutput {
    pub merged_index: String,
    pub merged_sources: usize,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema, lutum::Toolset)]
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
            nuillu_module::format_system_prompt(SYSTEM_PROMPT, cx.modules(), &self.owner)
        })
    }

    #[tracing::instrument(skip_all, err(Debug, level = "warn"))]
    async fn activate(&self, cx: &nuillu_module::ActivateCx<'_>) -> Result<()> {
        let snapshot = self
            .blackboard
            .read(|bb| {
                serde_json::json!({
                    "attention_stream": bb.attention_stream().entries(),
                    "memory_metadata": bb.memory_metadata(),
                })
            })
            .await;
        let allocation = self.allocation.snapshot().await;

        let lutum = self.llm.lutum().await;
        let mut session = Session::new(lutum);
        session.push_system(self.system_prompt(cx));
        session.push_user(
            serde_json::json!({
                "blackboard": snapshot,
                "allocation": allocation,
            })
            .to_string(),
        );

        for _ in 0..6 {
            let outcome = session
                .text_turn()
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
            memories: records
                .into_iter()
                .map(|record| MemoryContentView {
                    index: record.index.to_string(),
                    content: record.content.as_str().to_owned(),
                    rank: record.rank,
                })
                .collect(),
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
            .merge(
                &sources,
                args.merged_content,
                args.merged_rank,
                args.decay_secs,
            )
            .await
            .context("merge memories")?;
        Ok(MergeMemoriesOutput {
            merged_index: index.to_string(),
            merged_sources: source_count,
        })
    }
}

#[async_trait(?Send)]
impl Module for MemoryCompactionModule {
    type Batch = ();

    fn id() -> &'static str {
        "memory-compaction"
    }

    fn role_description() -> &'static str {
        "Merges redundant memory entries and accumulates remember tokens; wakes on allocation guidance, never on raw memos."
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
