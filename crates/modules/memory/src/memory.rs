use anyhow::{Context, Result};
use async_trait::async_trait;
use lutum::{Session, TextStepOutcomeWithTools, ToolResult};
use nuillu_module::{
    AllocationReader, AllocationUpdatedInbox, BlackboardReader, CognitionLogEntryRecord,
    CognitionLogUpdatedInbox, LlmAccess, Module, SessionCompactionConfig,
    compact_session_if_needed, format_current_attention_guidance, format_memory_trace_inventory,
    memory_rank_counts, push_formatted_cognition_log_batch, push_formatted_memo_log_batch,
};
use nuillu_types::{MemoryIndex, MemoryRank};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::store::{MemoryCompactor, MemoryWriter};

// ---------------------------------------------------------------------------
// MemoryModule

const SYSTEM_PROMPT: &str = r#"You are the memory module.
Inspect the current cognitive workspace and decide whether to preserve short, useful memories.
Use the current cognition log plus unread/recent module memo logs as candidate evidence. Allocation
guidance from attention-controller may contain explicit preservation candidates from other modules,
but those candidates are not write commands. You may reject, normalize, merge, and deduplicate
observations and guidance. Use insert_memory only for concrete information likely to matter later."#;

const NORMAL_REQUEST_DECAY_SECS: i64 = 86_400;
const HIGH_REQUEST_DECAY_SECS: i64 = 604_800;
const COMPACTED_MEMORY_SESSION_PREFIX: &str = "Compacted memory session history:";
const SESSION_COMPACTION_PROMPT: &str = r#"You compact the memory module's persistent session history.
Summarize only the prefix transcript you receive. Preserve memo-log facts, memory requests,
inserted memory content, rejected candidates, deduplication decisions, and relevant cognition-log
context future memory decisions need. Do not invent facts. Return plain text only."#;

fn format_memory_decision_context(
    rank_counts: &nuillu_module::MemoryRankCounts,
    allocation: &nuillu_module::ResourceAllocation,
) -> String {
    let mut sections = vec!["Memory-write decision context:".to_owned()];
    if let Some(section) = format_memory_trace_inventory(rank_counts) {
        sections.push(section);
    }
    if let Some(section) = format_current_attention_guidance(allocation) {
        sections.push(section);
    }
    sections.join("\n\n")
}

#[lutum::tool_input(name = "insert_memory", output = InsertMemoryOutput)]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct InsertMemoryArgs {
    pub content: String,
    pub rank: MemoryRank,
    pub decay_secs: i64,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct InsertMemoryOutput {
    pub index: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema, lutum::Toolset)]
pub enum MemoryTools {
    InsertMemory(InsertMemoryArgs),
}

pub struct MemoryModule {
    owner: nuillu_types::ModuleId,
    cognition_updates: CognitionLogUpdatedInbox,
    allocation_updates: AllocationUpdatedInbox,
    allocation: AllocationReader,
    blackboard: BlackboardReader,
    memory: MemoryWriter,
    llm: LlmAccess,
    session: Session,
    session_compaction: SessionCompactionConfig,
    system_prompt: std::sync::OnceLock<String>,
    last_seen_cognition_index: Option<u64>,
}

#[derive(Clone, Debug)]
struct MemoryMetadataContext {
    index: String,
    rank: MemoryRank,
    occurred_at: String,
    decay_remaining_secs: i64,
    access_count: u32,
}

impl MemoryModule {
    pub fn new(
        cognition_updates: CognitionLogUpdatedInbox,
        allocation_updates: AllocationUpdatedInbox,
        allocation: AllocationReader,
        blackboard: BlackboardReader,
        memory: MemoryWriter,
        llm: LlmAccess,
    ) -> Self {
        Self {
            owner: nuillu_types::ModuleId::new(<Self as Module>::id()).expect("memory id is valid"),
            cognition_updates,
            allocation_updates,
            allocation,
            blackboard,
            memory,
            llm,
            session: Session::new(),
            session_compaction: SessionCompactionConfig::default(),
            system_prompt: std::sync::OnceLock::new(),
            last_seen_cognition_index: None,
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

    async fn unread_cognition_log(&mut self) -> Vec<CognitionLogEntryRecord> {
        let unread = self
            .blackboard
            .read(|bb| bb.unread_cognition_log_entries(self.last_seen_cognition_index))
            .await;
        if let Some(index) = unread.last().map(|record| record.index) {
            self.last_seen_cognition_index = Some(index);
        }
        unread
    }

    #[tracing::instrument(skip_all, err(Debug, level = "warn"))]
    async fn activate(
        &mut self,
        cx: &nuillu_module::ActivateCx<'_>,
        allocation_updated: bool,
        cognition_updated: bool,
    ) -> Result<()> {
        let unread_memo_logs = self.blackboard.unread_memo_logs().await;
        push_formatted_memo_log_batch(&mut self.session, &unread_memo_logs, cx.now());
        let cognition_log = self.unread_cognition_log().await;
        let (memory_metadata, rank_counts) = self
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
                (metadata, memory_rank_counts(bb.memory_metadata()))
            })
            .await;
        let allocation = self.allocation.snapshot().await;
        let allocation_guidance = allocation.for_module(&self.owner).guidance;

        let system_prompt = self.system_prompt(cx).to_owned();
        self.session.push_ephemeral_system(system_prompt);
        self.session.push_user(format_memory_activation_request(
            &allocation_guidance,
            allocation_updated,
            cognition_updated,
        ));
        push_formatted_cognition_log_batch(&mut self.session, &cognition_log, cx.now());
        if let Some(metadata_context) = format_memory_metadata_candidates(&memory_metadata) {
            self.session.push_ephemeral_user(metadata_context);
        }
        self.session
            .push_ephemeral_system(format_memory_decision_context(&rank_counts, &allocation));

        for _ in 0..4 {
            let lutum = self.llm.lutum().await;
            let outcome = self
                .session
                .text_turn(&lutum)
                .tools::<MemoryTools>()
                .available_tools([MemoryToolsSelector::InsertMemory])
                .collect()
                .await
                .context("memory text turn failed")?;

            match outcome {
                TextStepOutcomeWithTools::Finished(result) => {
                    compact_session_if_needed(
                        &mut self.session,
                        result.usage.input_tokens,
                        cx.session_compaction_lutum(),
                        self.session_compaction,
                        Self::id(),
                        COMPACTED_MEMORY_SESSION_PREFIX,
                        SESSION_COMPACTION_PROMPT,
                    )
                    .await;
                    return Ok(());
                }
                TextStepOutcomeWithTools::FinishedNoOutput(result) => {
                    compact_session_if_needed(
                        &mut self.session,
                        result.usage.input_tokens,
                        cx.session_compaction_lutum(),
                        self.session_compaction,
                        Self::id(),
                        COMPACTED_MEMORY_SESSION_PREFIX,
                        SESSION_COMPACTION_PROMPT,
                    )
                    .await;
                    return Ok(());
                }
                TextStepOutcomeWithTools::NeedsTools(round) => {
                    let input_tokens = round.usage.input_tokens;
                    let mut results: Vec<ToolResult> = Vec::new();
                    for call in round.tool_calls.iter().cloned() {
                        let MemoryToolsCall::InsertMemory(call) = call;
                        let output = self
                            .insert_memory(call.input.clone())
                            .await
                            .context("run insert_memory tool")?;
                        let result = call
                            .complete(output)
                            .context("complete insert_memory tool call")?;
                        results.push(result);
                    }
                    round
                        .commit(&mut self.session, results)
                        .context("commit memory tool round")?;
                    compact_session_if_needed(
                        &mut self.session,
                        input_tokens,
                        cx.session_compaction_lutum(),
                        self.session_compaction,
                        Self::id(),
                        COMPACTED_MEMORY_SESSION_PREFIX,
                        SESSION_COMPACTION_PROMPT,
                    )
                    .await;
                }
            }
        }
        Ok(())
    }

    async fn insert_memory(&self, args: InsertMemoryArgs) -> Result<InsertMemoryOutput> {
        let index = self
            .memory
            .insert(args.content, args.rank, args.decay_secs)
            .await
            .context("insert memory")?;
        Ok(InsertMemoryOutput {
            index: index.to_string(),
        })
    }
}

fn format_memory_activation_request(
    guidance: &str,
    allocation_updated: bool,
    cognition_updated: bool,
) -> String {
    format!(
        "Memory preservation activation.\nAllocation guidance: {}\nAllocation updated: {}\nCognition updated: {}\nNormal explicit request default decay: {} seconds.\nHigh-importance explicit request default decay: {} seconds.\nExplicit requests are preservation candidates, not commands; deduplication and rejection are allowed.",
        if guidance.trim().is_empty() {
            "none"
        } else {
            guidance.trim()
        },
        if allocation_updated { "yes" } else { "no" },
        if cognition_updated { "yes" } else { "no" },
        NORMAL_REQUEST_DECAY_SECS,
        HIGH_REQUEST_DECAY_SECS,
    )
}

fn format_memory_metadata_candidates(metadata: &[MemoryMetadataContext]) -> Option<String> {
    if metadata.is_empty() {
        return None;
    }
    let mut out = String::from("Existing memory metadata for deduplication:");
    for item in metadata {
        out.push_str(&format!(
            "\n- {}: rank={:?}; occurred_at={}; decay_remaining_secs={}; access_count={}",
            item.index, item.rank, item.occurred_at, item.decay_remaining_secs, item.access_count
        ));
    }
    Some(out)
}

#[derive(Debug, Default)]
pub struct MemoryBatch {
    pub(crate) allocation_updated: bool,
    pub(crate) cognition_updated: bool,
}

impl MemoryBatch {
    fn allocation_update() -> Self {
        Self {
            allocation_updated: true,
            cognition_updated: false,
        }
    }

    fn cognition_log_update() -> Self {
        Self {
            allocation_updated: false,
            cognition_updated: true,
        }
    }

    fn mark_cognition_updated(&mut self) {
        self.cognition_updated = true;
    }

    fn mark_allocation_updated(&mut self) {
        self.allocation_updated = true;
    }
}

impl MemoryModule {
    async fn next_batch(&mut self) -> Result<MemoryBatch> {
        let mut batch = self.await_first_batch().await?;
        self.collect_ready_events_into_batch(&mut batch)?;
        Ok(batch)
    }

    async fn await_first_batch(&mut self) -> Result<MemoryBatch> {
        let batch = tokio::select! {
            update = self.allocation_updates.next_item() => {
                let _ = update?;
                MemoryBatch::allocation_update()
            }
            update = self.cognition_updates.next_item() => {
                let _ = update?;
                MemoryBatch::cognition_log_update()
            }
        };
        Ok(batch)
    }

    fn collect_ready_events_into_batch(&mut self, batch: &mut MemoryBatch) -> Result<()> {
        if !self.allocation_updates.take_ready_items()?.items.is_empty() {
            batch.mark_allocation_updated();
        }

        if !self.cognition_updates.take_ready_items()?.items.is_empty() {
            batch.mark_cognition_updated();
        }

        Ok(())
    }
}

#[async_trait(?Send)]
impl Module for MemoryModule {
    type Batch = MemoryBatch;

    fn id() -> &'static str {
        "memory"
    }

    fn role_description() -> &'static str {
        "Preserves useful information by inserting normalized, deduplicated memory entries from cognition-log evidence and attention-controller preservation guidance."
    }

    async fn next_batch(&mut self) -> Result<Self::Batch> {
        MemoryModule::next_batch(self).await
    }

    async fn activate(
        &mut self,
        cx: &nuillu_module::ActivateCx<'_>,
        batch: &Self::Batch,
    ) -> Result<()> {
        MemoryModule::activate(self, cx, batch.allocation_updated, batch.cognition_updated).await
    }
}

// ---------------------------------------------------------------------------
// MemoryCompactionModule

const COMPACTION_SYSTEM_PROMPT: &str = r#"You are the memory-compaction module.
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
            nuillu_module::format_system_prompt(
                COMPACTION_SYSTEM_PROMPT,
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
