use std::collections::HashMap;
use std::collections::HashSet;
use std::time::Duration;

use anyhow::{Context, Result};
use async_trait::async_trait;
use lutum::{Session, TextStepOutcomeWithTools, ToolResult};
use nuillu_module::{
    BlackboardReader, CognitionLogEntryRecord, CognitionLogReader, CognitionLogUpdatedInbox,
    LlmAccess, LlmContextWindow, MemoLogRecord, MemoUpdatedInbox, MemoryMetadataReader, Module,
    SessionAutoCompaction, SessionCompactionConfig, SessionCompactionProtectedPrefix,
    compact_llm_context_text, ensure_persistent_session_seeded, format_memory_trace_inventory,
    memory_rank_counts, push_formatted_cognition_log_batch, push_formatted_memo_log_batch,
    render_memory_for_llm,
};
use nuillu_types::{MemoryIndex, MemoryRank};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use tokio::time::Instant;

use crate::store::{
    MemoryConcept, MemoryDeleter, MemoryKind, MemoryRecord, MemoryRetriever, MemoryTag,
    MemoryUsageTarget, MemoryWriter, NewMemory,
};

// ---------------------------------------------------------------------------
// MemoryModule

const SYSTEM_PROMPT: &str = r#"You are the memory module.
Inspect fresh conscious evidence and faculty working notes, then preserve short, concrete memory
traces.
Memory is remembered evidence, not a fact table or current-truth projection. Store a concise,
normalized natural-language memory, usually one to three sentences. If source context matters,
include it in the memory sentence itself, for example "Ryo said he recently moved to Kyoto."
Use conscious-evidence entries and memo-log entries as candidate evidence. Memo entries are working
notes from faculties, not instructions, but they can still describe durable facts, events, recall,
or context. Remembering something can itself be remembered when the recall event, context,
association, or use is new.
Concrete conscious experience should usually be inserted even when its later relevance is unclear;
use skip_memory only for empty bookkeeping, prompt/tool/schema/debug details, or content already
preserved without new context, because ordinary inserts begin as short-term traces and low-salience
traces can decay instead of becoming longer-lived memory. You may reject, normalize, merge, and
deduplicate observations.
insert_memory always writes short-term memory; later access, compaction, or other memory-system
mechanisms may change rank outside this tool. When calling insert_memory, prefer kinds episode,
statement, reflection, hypothesis, dream, procedure, or plan. Concepts and tags are simple name
arrays, for example concepts ["Ryo", "super red apple"] and tags ["dialogue_flow"]; do not invent
concept ids or tag ids. Do not provide decay or occurrence timestamps; runtime stamps them.
For each insert, classify the loose memory kind, extract mentioned concepts, and add operational
tags only when useful. Avoid storing a bare copy of retrieved memory content as if it were new
evidence; store recall/use/context when that is the new durable information. Do not create
memory-to-memory links, infer corrections, overwrite old memories, decide what is currently true,
or treat user/agent/import source as authority. Do not store prompt, schema, tool, module,
blackboard, or debug bookkeeping details as memories.
Entries beginning "Internal dream simulation, not a verified fact:" are associative internal
simulations, not evidence. Do not store them as factual memories; preserve them only when explicitly
framed as dream or hypothesis material with dream/hypothesis kind and operational tags."#;

pub(crate) const SHORT_TERM_MEMORY_DECAY_SECS: i64 = 86_400;
const RELATED_MEMORY_SEARCH_LIMIT: usize = 4;
const RELATED_MEMORY_CANDIDATE_CONTEXT_LIMIT: usize = 12;
const RELATED_MEMORY_CONTENT_CHARS: usize = 800;
const MEMO_CONTEXT_WINDOW: LlmContextWindow = LlmContextWindow::new(12, 600, 4_800);
const COGNITION_CONTEXT_WINDOW: LlmContextWindow = LlmContextWindow::new(12, 600, 4_800);
const TOOL_TURN_MAX_OUTPUT_TOKENS: u32 = 768;
const DEFAULT_MEMORY_BATCH_SILENT_WINDOW: Duration = Duration::from_millis(100);
const DEFAULT_MEMORY_BATCH_BUDGET: Duration = Duration::from_secs(1);
const COMPACTED_MEMORY_SESSION_PREFIX: &str = "Compacted memory session history:";
const SESSION_COMPACTION_FOCUS: &str = r#"Preserve memo facts, cognition-log facts, memory requests,
inserted memory content, skipped candidates, novelty decisions, and deduplication decisions future
memory decisions need."#;

pub fn session_auto_compaction() -> SessionAutoCompaction {
    SessionAutoCompaction::new(
        SessionCompactionConfig::default(),
        SessionCompactionProtectedPrefix::LeadingSystemAndIdentitySeed,
        COMPACTED_MEMORY_SESSION_PREFIX,
        SESSION_COMPACTION_FOCUS,
    )
}

fn format_memory_decision_context(rank_counts: &nuillu_module::MemoryRankCounts) -> String {
    let mut sections = vec!["Memory-write decision context:".to_owned()];
    if let Some(section) = format_memory_trace_inventory(rank_counts) {
        sections.push(section);
    }
    sections.join("\n\n")
}

#[lutum::tool_input(name = "insert_memory", output = InsertMemoryOutput)]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct InsertMemoryArgs {
    pub content: String,
    pub kind: MemoryKind,
    #[serde(default)]
    pub concepts: Vec<MemoryConceptInput>,
    #[serde(default)]
    pub tags: Vec<MemoryTagInput>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(transparent)]
pub struct MemoryConceptInput(pub String);

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(transparent)]
pub struct MemoryTagInput(pub String);

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct InsertMemoryOutput {
    pub index: String,
}

#[lutum::tool_input(name = "skip_memory", output = SkipMemoryOutput)]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct SkipMemoryArgs {
    pub reason: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct SkipMemoryOutput {
    pub skipped: bool,
    pub message: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema, lutum::Toolset)]
pub enum MemoryTools {
    InsertMemory(InsertMemoryArgs),
    SkipMemory(SkipMemoryArgs),
}

#[lutum::tool_input(name = "keep_new_memory", output = KeepNewMemoryOutput)]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct KeepNewMemoryArgs {
    pub reason: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct KeepNewMemoryOutput {
    pub kept: bool,
    pub index: MemoryIndex,
    pub message: String,
}

#[lutum::tool_input(
    name = "delete_redundant_memory",
    output = DeleteRedundantMemoryOutput
)]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct DeleteRedundantMemoryArgs {
    #[serde(default)]
    pub redundant_with: Vec<MemoryIndex>,
    pub reason: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct DeleteRedundantMemoryOutput {
    pub deleted: bool,
    pub index: MemoryIndex,
    pub reinforced_indexes: Vec<MemoryIndex>,
    pub message: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema, lutum::Toolset)]
pub enum MemoryNoveltyTools {
    KeepNewMemory(KeepNewMemoryArgs),
    DeleteRedundantMemory(DeleteRedundantMemoryArgs),
}

pub struct MemoryModule {
    owner: nuillu_types::ModuleId,
    memo_updates: MemoUpdatedInbox,
    cognition_updates: CognitionLogUpdatedInbox,
    blackboard: BlackboardReader,
    cognition_log: CognitionLogReader,
    memory_metadata: MemoryMetadataReader,
    memory: MemoryWriter,
    memory_deleter: MemoryDeleter,
    memory_retriever: MemoryRetriever,
    llm: LlmAccess,
    session: Session,
    system_prompt: std::sync::OnceLock<String>,
    batching: MemoryBatchConfig,
}

#[derive(Clone, Debug)]
struct RelatedMemoryCandidate {
    target: MemoryUsageTarget,
    content: String,
    stored_at: String,
    kind: MemoryKind,
}

impl MemoryModule {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        memo_updates: MemoUpdatedInbox,
        cognition_updates: CognitionLogUpdatedInbox,
        blackboard: BlackboardReader,
        cognition_log: CognitionLogReader,
        memory_metadata: MemoryMetadataReader,
        memory: MemoryWriter,
        memory_deleter: MemoryDeleter,
        memory_retriever: MemoryRetriever,
        llm: LlmAccess,
        session: Session,
    ) -> Self {
        Self {
            owner: nuillu_types::ModuleId::new(<Self as Module>::id()).expect("memory id is valid"),
            memo_updates,
            cognition_updates,
            blackboard,
            cognition_log,
            memory_metadata,
            memory,
            memory_deleter,
            memory_retriever,
            llm,
            session,
            system_prompt: std::sync::OnceLock::new(),
            batching: MemoryBatchConfig::default(),
        }
    }

    #[cfg(test)]
    fn with_batch_config(mut self, batching: MemoryBatchConfig) -> Self {
        self.batching = batching;
        self
    }

    fn system_prompt(&self, cx: &nuillu_module::ActivateCx<'_>) -> &str {
        self.system_prompt.get_or_init(|| {
            nuillu_module::format_system_prompt(
                SYSTEM_PROMPT,
                cx.peer_contexts(),
                &self.owner,
                cx.core_policies(),
            )
        })
    }

    fn ensure_session_seeded(&mut self, cx: &nuillu_module::ActivateCx<'_>) {
        let system_prompt = self.system_prompt(cx).to_owned();
        ensure_persistent_session_seeded(
            &mut self.session,
            system_prompt,
            cx.identity_memories(),
            cx.now(),
        );
    }

    #[tracing::instrument(skip_all, err(Debug, level = "warn"))]
    async fn activate(
        &mut self,
        cx: &nuillu_module::ActivateCx<'_>,
        batch: &MemoryBatch,
    ) -> Result<()> {
        if batch.is_empty() {
            return Ok(());
        }
        self.ensure_session_seeded(cx);
        push_formatted_memo_log_batch(
            &mut self.session,
            &batch.memo_logs,
            cx.now(),
            MEMO_CONTEXT_WINDOW,
        );
        push_formatted_cognition_log_batch(
            &mut self.session,
            &batch.cognition_log,
            cx.now(),
            COGNITION_CONTEXT_WINDOW,
        );
        let rank_counts = self.memory_metadata.read(memory_rank_counts).await;
        self.session.push_user(format_memory_activation_request(
            batch.memo_logs.len(),
            batch.cognition_log.len(),
        ));
        self.session
            .push_ephemeral_system(format_memory_decision_context(&rank_counts));

        let inserted = self.run_creation_turn(cx).await?;
        let inserted_this_activation = inserted
            .iter()
            .map(|record| record.index.clone())
            .collect::<HashSet<_>>();
        for record in &inserted {
            self.review_inserted_memory(cx, record, &inserted_this_activation)
                .await?;
        }
        Ok(())
    }

    async fn run_creation_turn(
        &mut self,
        cx: &nuillu_module::ActivateCx<'_>,
    ) -> Result<Vec<MemoryRecord>> {
        let lutum = self.llm.lutum().await;
        let outcome = self
            .session
            .text_turn()
            .tools::<MemoryTools>()
            .available_tools([
                MemoryToolsSelector::InsertMemory,
                MemoryToolsSelector::SkipMemory,
            ])
            .require_any_tool()
            .max_output_tokens(TOOL_TURN_MAX_OUTPUT_TOKENS)
            .collect_controlled_with(&lutum, nuillu_module::AbortOnAvailableToolNameInText::new())
            .await
            .context("memory creation text turn failed")?;

        let round = match outcome {
            TextStepOutcomeWithTools::Finished(result) => {
                cx.compact_and_save(&mut self.session, result.usage).await?;
                anyhow::bail!("memory creation finished without required tool call");
            }
            TextStepOutcomeWithTools::FinishedNoOutput(result) => {
                cx.compact_and_save(&mut self.session, result.usage).await?;
                anyhow::bail!("memory creation finished without required tool call");
            }
            TextStepOutcomeWithTools::NeedsTools(round) => round,
        };
        let usage = round.usage;

        let mut results: Vec<ToolResult> = Vec::new();
        let mut inserted = Vec::new();
        nuillu_module::emit_trace_tool_calls(&round.tool_calls);
        for call in round.tool_calls.iter().cloned() {
            match call {
                MemoryToolsCall::InsertMemory(call) => {
                    let (output, record) = self
                        .insert_memory(call.input.clone())
                        .await
                        .context("run insert_memory tool")?;
                    inserted.push(record);
                    results.push(
                        call.complete(output)
                            .context("complete insert_memory tool call")?,
                    );
                }
                MemoryToolsCall::SkipMemory(call) => {
                    let output = self.skip_memory(call.input.clone());
                    results.push(
                        call.complete(output)
                            .context("complete skip_memory tool call")?,
                    );
                }
            }
        }
        round
            .commit(&mut self.session, results)
            .context("commit memory creation tool round")?;
        cx.compact_and_save(&mut self.session, usage).await?;
        Ok(inserted)
    }

    async fn insert_memory(
        &self,
        args: InsertMemoryArgs,
    ) -> Result<(InsertMemoryOutput, MemoryRecord)> {
        let content = args.content.trim();
        if content.is_empty() {
            anyhow::bail!("insert_memory content was empty");
        }
        let record = self
            .memory
            .insert_entry_now(
                NewMemory {
                    content: nuillu_types::MemoryContent::new(content.to_owned()),
                    rank: MemoryRank::ShortTerm,
                    occurred_at: None,
                    kind: args.kind,
                    concepts: args
                        .concepts
                        .into_iter()
                        .map(memory_concept_from_input)
                        .collect(),
                    tags: args.tags.into_iter().map(memory_tag_from_input).collect(),
                    affect_arousal: 0.0,
                    valence: 0.0,
                    emotion: String::new(),
                },
                SHORT_TERM_MEMORY_DECAY_SECS,
            )
            .await
            .context("insert memory")?;
        let output = InsertMemoryOutput {
            index: record.index.to_string(),
        };
        Ok((output, record))
    }

    fn skip_memory(&self, _args: SkipMemoryArgs) -> SkipMemoryOutput {
        SkipMemoryOutput {
            skipped: true,
            message: "memory creation skipped for this activation".to_owned(),
        }
    }

    async fn review_inserted_memory(
        &mut self,
        cx: &nuillu_module::ActivateCx<'_>,
        new_memory: &MemoryRecord,
        inserted_this_activation: &HashSet<MemoryIndex>,
    ) -> Result<()> {
        let candidates = self
            .related_memory_candidates_for_new_memory(
                new_memory,
                cx.now(),
                inserted_this_activation,
            )
            .await?;
        if candidates.is_empty() {
            return Ok(());
        }
        let targets = candidates
            .iter()
            .map(|candidate| (candidate.target.index.clone(), candidate.target.clone()))
            .collect::<HashMap<_, _>>();
        self.session.push_user(format_novelty_review_request(
            new_memory,
            &candidates,
            cx.now(),
        ));

        let lutum = self.llm.lutum().await;
        let outcome = self
            .session
            .text_turn()
            .tools::<MemoryNoveltyTools>()
            .available_tools([
                MemoryNoveltyToolsSelector::KeepNewMemory,
                MemoryNoveltyToolsSelector::DeleteRedundantMemory,
            ])
            .require_any_tool()
            .max_output_tokens(TOOL_TURN_MAX_OUTPUT_TOKENS)
            .collect_controlled_with(&lutum, nuillu_module::AbortOnAvailableToolNameInText::new())
            .await
            .context("memory novelty text turn failed")?;

        let round = match outcome {
            TextStepOutcomeWithTools::Finished(result) => {
                cx.compact_and_save(&mut self.session, result.usage).await?;
                anyhow::bail!("memory novelty review finished without required tool call");
            }
            TextStepOutcomeWithTools::FinishedNoOutput(result) => {
                cx.compact_and_save(&mut self.session, result.usage).await?;
                anyhow::bail!("memory novelty review finished without required tool call");
            }
            TextStepOutcomeWithTools::NeedsTools(round) => round,
        };
        let usage = round.usage;

        let mut results: Vec<ToolResult> = Vec::new();
        nuillu_module::emit_trace_tool_calls(&round.tool_calls);
        for call in round.tool_calls.iter().cloned() {
            match call {
                MemoryNoveltyToolsCall::KeepNewMemory(call) => {
                    let output = self.keep_new_memory(new_memory, call.input.clone());
                    results.push(
                        call.complete(output)
                            .context("complete keep_new_memory tool call")?,
                    );
                }
                MemoryNoveltyToolsCall::DeleteRedundantMemory(call) => {
                    let output = self
                        .delete_redundant_memory(new_memory, call.input.clone(), &targets)
                        .await
                        .context("run delete_redundant_memory tool")?;
                    results.push(
                        call.complete(output)
                            .context("complete delete_redundant_memory tool call")?,
                    );
                }
            }
        }
        round
            .commit(&mut self.session, results)
            .context("commit memory novelty tool round")?;
        cx.compact_and_save(&mut self.session, usage).await?;
        Ok(())
    }

    fn keep_new_memory(
        &self,
        new_memory: &MemoryRecord,
        _args: KeepNewMemoryArgs,
    ) -> KeepNewMemoryOutput {
        KeepNewMemoryOutput {
            kept: true,
            index: new_memory.index.clone(),
            message: "new memory kept".to_owned(),
        }
    }

    async fn delete_redundant_memory(
        &self,
        new_memory: &MemoryRecord,
        args: DeleteRedundantMemoryArgs,
        targets: &HashMap<MemoryIndex, MemoryUsageTarget>,
    ) -> Result<DeleteRedundantMemoryOutput> {
        let reinforced_targets = args
            .redundant_with
            .iter()
            .filter_map(|index| targets.get(index).cloned())
            .collect::<Vec<_>>();
        for target in &reinforced_targets {
            self.memory_retriever.record_reinforcement(target).await;
        }
        self.memory_deleter
            .delete(&new_memory.index)
            .await
            .context("delete redundant new memory")?;
        Ok(DeleteRedundantMemoryOutput {
            deleted: true,
            index: new_memory.index.clone(),
            reinforced_indexes: reinforced_targets
                .into_iter()
                .map(|target| target.index)
                .collect(),
            message: "redundant new memory deleted".to_owned(),
        })
    }

    async fn related_memory_candidates_for_new_memory(
        &self,
        new_memory: &MemoryRecord,
        now: chrono::DateTime<chrono::Utc>,
        excluded_indexes: &HashSet<MemoryIndex>,
    ) -> Result<Vec<RelatedMemoryCandidate>> {
        let mut seen = HashSet::new();
        let mut candidates = Vec::new();
        let search_limit = RELATED_MEMORY_CANDIDATE_CONTEXT_LIMIT
            .saturating_add(excluded_indexes.len())
            .max(RELATED_MEMORY_SEARCH_LIMIT);
        let hits = self
            .memory_retriever
            .search(new_memory.content.as_str(), search_limit)
            .await
            .context("search related memory candidates")?;
        for hit in hits {
            if excluded_indexes.contains(&hit.index) || !seen.insert(hit.index.clone()) {
                continue;
            }
            candidates.push(RelatedMemoryCandidate {
                target: MemoryUsageTarget::from(&hit),
                content: render_memory_for_llm(hit.content.as_str(), hit.occurred_at, now),
                stored_at: hit.stored_at.to_rfc3339(),
                kind: hit.kind,
            });
            if candidates.len() >= RELATED_MEMORY_CANDIDATE_CONTEXT_LIMIT {
                break;
            }
        }
        let targets = candidates
            .iter()
            .map(|candidate| candidate.target.clone())
            .collect::<Vec<_>>();
        self.memory_retriever.record_accesses(&targets).await;
        Ok(candidates)
    }
}

pub(crate) fn memory_concept_from_input(input: MemoryConceptInput) -> MemoryConcept {
    MemoryConcept::new(input.0)
}

pub(crate) fn memory_tag_from_input(input: MemoryTagInput) -> MemoryTag {
    MemoryTag::operational(input.0)
}

fn format_memory_activation_request(memo_count: usize, cognition_count: usize) -> String {
    format!(
        "Memory preservation activation.\nNew memo-log entries: {}\nNew cognition-log entries: {}\nOrdinary memory writes begin as short-term traces; runtime stamps decay and occurrence time.\nUse insert_memory by default for concrete conscious experience. Use skip_memory only for empty bookkeeping, prompt/tool/schema/debug details, or content already preserved without new context.",
        memo_count, cognition_count,
    )
}

fn format_related_memory_candidates(candidates: &[RelatedMemoryCandidate]) -> Option<String> {
    if candidates.is_empty() {
        return None;
    }
    let mut out = String::from("Related existing memory candidates:");
    for candidate in candidates {
        out.push_str(&format!(
            "\n- [{}] rank={:?}; kind={:?}; occurred_at={}; stored_at={}\n  {}",
            candidate.target.index,
            candidate.target.rank,
            candidate.kind,
            candidate
                .target
                .occurred_at
                .map(|at| at.to_rfc3339())
                .unwrap_or_else(|| "unknown".to_owned()),
            candidate.stored_at,
            compact_llm_context_text(&candidate.content, RELATED_MEMORY_CONTENT_CHARS)
        ));
    }
    Some(out)
}

fn format_novelty_review_request(
    new_memory: &MemoryRecord,
    candidates: &[RelatedMemoryCandidate],
    now: chrono::DateTime<chrono::Utc>,
) -> String {
    let mut sections = vec![format!(
        "Novelty review for newly inserted memory.\nNew memory [{}] rank={:?}; kind={:?}; occurred_at={}; stored_at={}\n  {}",
        new_memory.index,
        new_memory.rank,
        new_memory.kind,
        new_memory
            .occurred_at
            .map(|at| at.to_rfc3339())
            .unwrap_or_else(|| "unknown".to_owned()),
        new_memory.stored_at.to_rfc3339(),
        compact_llm_context_text(
            &render_memory_for_llm(new_memory.content.as_str(), new_memory.occurred_at, now),
            RELATED_MEMORY_CONTENT_CHARS,
        )
    )];
    if let Some(related) = format_related_memory_candidates(candidates) {
        sections.push(related);
    }
    sections.push(
        "Instruction: Decide whether the new memory records information not already captured by the related existing memories. Keep it if it adds a new fact, event, contradiction, update, association, recall/use context, or other durable information. Delete it only if it merely restates or copies existing memory content without adding new context. Call exactly one novelty tool.".to_owned(),
    );
    sections.join("\n\n")
}

#[derive(Debug, Default)]
pub struct MemoryBatch {
    pub(crate) memo_logs: Vec<MemoLogRecord>,
    pub(crate) cognition_log: Vec<CognitionLogEntryRecord>,
}

impl MemoryBatch {
    fn is_empty(&self) -> bool {
        self.memo_logs.is_empty() && self.cognition_log.is_empty()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct MemoryBatchConfig {
    silent_window: Duration,
    budget: Duration,
}

impl Default for MemoryBatchConfig {
    fn default() -> Self {
        Self {
            silent_window: DEFAULT_MEMORY_BATCH_SILENT_WINDOW,
            budget: DEFAULT_MEMORY_BATCH_BUDGET,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct ReadyCounts {
    memo_updates: usize,
    cognition_updates: usize,
}

impl ReadyCounts {
    fn is_empty(self) -> bool {
        self.memo_updates == 0 && self.cognition_updates == 0
    }
}

impl MemoryModule {
    async fn next_batch(&mut self) -> Result<MemoryBatch> {
        self.await_first_update().await?;
        let _ = self.collect_ready_events()?;
        self.collect_update_burst().await?;
        Ok(MemoryBatch {
            memo_logs: self.blackboard.unread_memo_logs().await,
            cognition_log: self.cognition_log.unread_events().await,
        })
    }

    async fn await_first_update(&mut self) -> Result<()> {
        tokio::select! {
            update = self.memo_updates.next_item() => {
                let _ = update?;
            }
            update = self.cognition_updates.next_item() => {
                let _ = update?;
            }
        }
        Ok(())
    }

    async fn collect_update_burst(&mut self) -> Result<()> {
        let mut waited = Duration::ZERO;
        while waited < self.batching.budget {
            let remaining = self.batching.budget.saturating_sub(waited);
            let wait_for = std::cmp::min(self.batching.silent_window, remaining);
            if wait_for.is_zero() {
                break;
            }

            let started = Instant::now();
            tokio::select! {
                update = self.memo_updates.next_item() => {
                    let _ = update?;
                    waited += std::cmp::min(started.elapsed(), wait_for);
                    let _ = self.collect_ready_events()?;
                }
                update = self.cognition_updates.next_item() => {
                    let _ = update?;
                    waited += std::cmp::min(started.elapsed(), wait_for);
                    let _ = self.collect_ready_events()?;
                }
                _ = tokio::time::sleep(wait_for) => {
                    waited += wait_for;
                    let ready = self.collect_ready_events()?;
                    if ready.is_empty() {
                        break;
                    }
                }
            }
        }
        Ok(())
    }

    fn collect_ready_events(&mut self) -> Result<ReadyCounts> {
        let memo_updates = self.memo_updates.take_ready_items()?.items.len();
        let cognition_updates = self.cognition_updates.take_ready_items()?.items.len();
        Ok(ReadyCounts {
            memo_updates,
            cognition_updates,
        })
    }
}

#[async_trait(?Send)]
impl Module for MemoryModule {
    type Batch = MemoryBatch;

    fn id() -> &'static str {
        "memory"
    }

    fn peer_context() -> Option<&'static str> {
        Some("Memory preserves useful memo and cognition evidence for later recall.")
    }

    async fn next_batch(&mut self) -> Result<Self::Batch> {
        MemoryModule::next_batch(self).await
    }

    async fn activate(
        &mut self,
        cx: &nuillu_module::ActivateCx<'_>,
        batch: &Self::Batch,
    ) -> Result<()> {
        MemoryModule::activate(self, cx, batch).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::cell::RefCell;
    use std::collections::BTreeMap;
    use std::rc::Rc;
    use std::sync::{Arc, Mutex};

    use chrono::{TimeZone, Utc};
    use lutum::{
        AdapterStructuredTurn, AdapterTextTurn, AdapterToolChoice, AgentError,
        ErasedStructuredTurnEventStream, ErasedTextTurnEventStream, FinishReason, InputMessageRole,
        Lutum, MessageContent, MockLlmAdapter, MockTextScenario, ModelInput, ModelInputItem,
        RawTextTurnEvent, SharedPoolBudgetManager, SharedPoolBudgetOptions, TurnAdapter, Usage,
    };
    use nuillu_blackboard::{Blackboard, Bpm, ModulePolicy, linear_ratio_fn};
    use nuillu_module::ports::{NoopCognitionLogRepository, PortError, SystemClock};
    use nuillu_module::{
        CapabilityProviderPorts, CapabilityProviders, CognitionLogUpdated, LlmConcurrencyLimiter,
        LutumTiers, MemoUpdated, ModuleRegistry, SessionCompactionPolicy, SessionCompactionRuntime,
    };
    use nuillu_types::{ModelTier, ModuleInstanceId, ReplicaCapRange, ReplicaIndex, builtin};

    use crate::store::{
        IndexedMemory, LinkedMemoryQuery, LinkedMemoryRecord, MemoryLink, MemoryLinkRelation,
        MemoryQuery, MemoryStore, NewMemoryLink,
    };
    use crate::{MemoryCapabilities, NoopMemoryStore};

    type BatchRecorder = Rc<RefCell<Vec<(usize, usize)>>>;

    #[derive(Default)]
    struct SearchableMemoryStore {
        records: RefCell<BTreeMap<String, MemoryRecord>>,
    }

    impl SearchableMemoryStore {
        fn seed(&self, index: &str, content: &str, stored_at: chrono::DateTime<Utc>) {
            self.records.borrow_mut().insert(
                index.to_owned(),
                MemoryRecord {
                    index: MemoryIndex::new(index),
                    content: nuillu_types::MemoryContent::new(content),
                    rank: MemoryRank::ShortTerm,
                    occurred_at: Some(stored_at),
                    stored_at,
                    kind: MemoryKind::Statement,
                    concepts: Vec::new(),
                    tags: Vec::new(),
                    affect_arousal: 0.0,
                    valence: 0.0,
                    emotion: String::new(),
                },
            );
        }

        fn records(&self) -> Vec<MemoryRecord> {
            self.records.borrow().values().cloned().collect()
        }
    }

    #[async_trait(?Send)]
    impl MemoryStore for SearchableMemoryStore {
        async fn insert(
            &self,
            mem: NewMemory,
            stored_at: chrono::DateTime<Utc>,
        ) -> std::result::Result<MemoryRecord, PortError> {
            let index = MemoryIndex::new(format!("new-memory-{}", self.records.borrow().len()));
            let record = MemoryRecord {
                index: index.clone(),
                content: mem.content,
                rank: mem.rank,
                occurred_at: mem.occurred_at,
                stored_at,
                kind: mem.kind,
                concepts: mem.concepts,
                tags: mem.tags,
                affect_arousal: mem.affect_arousal,
                valence: mem.valence,
                emotion: mem.emotion,
            };
            self.records
                .borrow_mut()
                .insert(index.as_str().to_owned(), record.clone());
            Ok(record)
        }

        async fn put(&self, mem: IndexedMemory) -> std::result::Result<MemoryRecord, PortError> {
            let record = MemoryRecord {
                index: mem.index,
                content: mem.content,
                rank: mem.rank,
                occurred_at: mem.occurred_at,
                stored_at: mem.stored_at,
                kind: mem.kind,
                concepts: mem.concepts,
                tags: mem.tags,
                affect_arousal: mem.affect_arousal,
                valence: mem.valence,
                emotion: mem.emotion,
            };
            self.records
                .borrow_mut()
                .insert(record.index.as_str().to_owned(), record.clone());
            Ok(record)
        }

        async fn compact(
            &self,
            mem: NewMemory,
            _sources: &[MemoryIndex],
            stored_at: chrono::DateTime<Utc>,
        ) -> std::result::Result<MemoryRecord, PortError> {
            self.insert(mem, stored_at).await
        }

        async fn put_compacted(
            &self,
            mem: IndexedMemory,
            _sources: &[MemoryIndex],
        ) -> std::result::Result<MemoryRecord, PortError> {
            self.put(mem).await
        }

        async fn get(
            &self,
            index: &MemoryIndex,
        ) -> std::result::Result<Option<MemoryRecord>, PortError> {
            Ok(self.records.borrow().get(index.as_str()).cloned())
        }

        async fn list_by_rank(
            &self,
            rank: MemoryRank,
        ) -> std::result::Result<Vec<MemoryRecord>, PortError> {
            Ok(self
                .records
                .borrow()
                .values()
                .filter(|record| record.rank == rank)
                .cloned()
                .collect())
        }

        async fn search(
            &self,
            _q: &MemoryQuery,
        ) -> std::result::Result<Vec<MemoryRecord>, PortError> {
            Ok(self.records())
        }

        async fn linked(
            &self,
            _q: &LinkedMemoryQuery,
        ) -> std::result::Result<Vec<LinkedMemoryRecord>, PortError> {
            Ok(Vec::new())
        }

        async fn upsert_link(
            &self,
            link: NewMemoryLink,
            updated_at: chrono::DateTime<Utc>,
        ) -> std::result::Result<MemoryLink, PortError> {
            Ok(MemoryLink {
                from_memory: link.from_memory,
                to_memory: link.to_memory,
                relation: MemoryLinkRelation::Related,
                freeform_relation: link.freeform_relation,
                strength: link.strength,
                confidence: link.confidence,
                updated_at,
            })
        }

        async fn delete(&self, index: &MemoryIndex) -> std::result::Result<(), PortError> {
            self.records.borrow_mut().remove(index.as_str());
            Ok(())
        }
    }

    #[derive(Clone)]
    struct CapturingAdapter {
        inner: MockLlmAdapter,
        text_inputs: Arc<Mutex<Vec<ModelInput>>>,
        text_turns: Arc<Mutex<Vec<AdapterTextTurn>>>,
    }

    impl CapturingAdapter {
        fn new(inner: MockLlmAdapter) -> Self {
            Self {
                inner,
                text_inputs: Arc::new(Mutex::new(Vec::new())),
                text_turns: Arc::new(Mutex::new(Vec::new())),
            }
        }

        fn text_inputs(&self) -> Vec<ModelInput> {
            self.text_inputs.lock().unwrap().clone()
        }

        fn text_turns(&self) -> Vec<AdapterTextTurn> {
            self.text_turns.lock().unwrap().clone()
        }
    }

    #[async_trait]
    impl TurnAdapter for CapturingAdapter {
        async fn text_turn(
            &self,
            input: ModelInput,
            turn: AdapterTextTurn,
        ) -> Result<ErasedTextTurnEventStream, AgentError> {
            self.text_inputs.lock().unwrap().push(input.clone());
            self.text_turns.lock().unwrap().push(turn.clone());
            self.inner.text_turn(input, turn).await
        }

        async fn structured_turn(
            &self,
            input: ModelInput,
            turn: AdapterStructuredTurn,
        ) -> Result<ErasedStructuredTurnEventStream, AgentError> {
            self.inner.structured_turn(input, turn).await
        }
    }

    struct RecordingMemory {
        inner: MemoryModule,
        recorder: BatchRecorder,
    }

    #[async_trait(?Send)]
    impl Module for RecordingMemory {
        type Batch = MemoryBatch;

        fn id() -> &'static str {
            MemoryModule::id()
        }

        fn peer_context() -> Option<&'static str> {
            MemoryModule::peer_context()
        }

        async fn next_batch(&mut self) -> Result<Self::Batch> {
            let batch = self.inner.next_batch().await?;
            self.recorder
                .borrow_mut()
                .push((batch.memo_logs.len(), batch.cognition_log.len()));
            Ok(batch)
        }

        async fn activate(
            &mut self,
            _cx: &nuillu_module::ActivateCx<'_>,
            _batch: &Self::Batch,
        ) -> Result<()> {
            Ok(())
        }
    }

    fn test_caps() -> (Blackboard, CapabilityProviders, MemoryCapabilities, Lutum) {
        test_caps_with_adapter(Blackboard::default(), Arc::new(MockLlmAdapter::new()))
    }

    fn test_caps_with_adapter<T>(
        blackboard: Blackboard,
        adapter: Arc<T>,
    ) -> (Blackboard, CapabilityProviders, MemoryCapabilities, Lutum)
    where
        T: TurnAdapter + 'static,
    {
        test_caps_with_store_and_adapter(blackboard, Rc::new(NoopMemoryStore), adapter)
    }

    fn test_caps_with_store_and_adapter<T>(
        blackboard: Blackboard,
        primary_store: Rc<dyn MemoryStore>,
        adapter: Arc<T>,
    ) -> (Blackboard, CapabilityProviders, MemoryCapabilities, Lutum)
    where
        T: TurnAdapter + 'static,
    {
        let budget = SharedPoolBudgetManager::new(SharedPoolBudgetOptions::default());
        let lutum = Lutum::new(adapter, budget);
        let clock = Rc::new(SystemClock);
        let caps = CapabilityProviders::new(CapabilityProviderPorts {
            blackboard: blackboard.clone(),
            cognition_log_port: Rc::new(NoopCognitionLogRepository),
            clock: clock.clone(),
            tiers: LutumTiers::from_shared_lutum(lutum.clone()),
        });
        let memory_caps =
            MemoryCapabilities::new(blackboard.clone(), clock, primary_store, Vec::new());
        (blackboard, caps, memory_caps, lutum)
    }

    fn module_policy() -> ModulePolicy {
        ModulePolicy::new(
            ReplicaCapRange::new(1, 1).unwrap(),
            Bpm::from_f64(60_000.0)..=Bpm::from_f64(60_000.0),
            linear_ratio_fn,
        )
    }

    fn activate_cx(
        lutum: &Lutum,
        now: chrono::DateTime<Utc>,
    ) -> nuillu_module::ActivateCx<'static> {
        nuillu_module::ActivateCx::new(
            &[],
            &[],
            &[],
            SessionCompactionRuntime::new(
                lutum.clone(),
                LlmConcurrencyLimiter::new(None),
                ModelTier::Cheap,
                SessionCompactionPolicy::default(),
            ),
            now,
        )
    }

    fn message_texts_with_role(input: &ModelInput, expected_role: InputMessageRole) -> Vec<&str> {
        input
            .items()
            .iter()
            .filter_map(|item| match item {
                ModelInputItem::Message { role, content } if role == &expected_role => {
                    let [MessageContent::Text(text)] = content.as_slice() else {
                        panic!("expected one text content item");
                    };
                    Some(text.as_str())
                }
                _ => None,
            })
            .collect()
    }

    async fn build_recording_memory(
        caps: &CapabilityProviders,
        memory_caps: MemoryCapabilities,
        recorder: BatchRecorder,
        batching: MemoryBatchConfig,
    ) -> nuillu_module::AllocatedModule {
        let modules = ModuleRegistry::new()
            .register(
                ModulePolicy::new(
                    ReplicaCapRange::new(1, 1).unwrap(),
                    Bpm::from_f64(60_000.0)..=Bpm::from_f64(60_000.0),
                    linear_ratio_fn,
                ),
                move |caps| {
                    let memory_caps = memory_caps.clone();
                    let recorder = recorder.clone();
                    async move {
                        Ok(RecordingMemory {
                            inner: MemoryModule::new(
                                caps.memo_updated_inbox(),
                                caps.cognition_log_updated_inbox(),
                                caps.blackboard_reader(),
                                caps.cognition_log_reader(),
                                caps.memory_metadata_reader(),
                                memory_caps.writer(),
                                memory_caps.deleter(),
                                memory_caps.retriever(),
                                caps.llm("main")
                                    .with_tier(nuillu_types::ModelTier::Cheap)
                                    .into(),
                                caps.session("main")
                                    .with_tier(nuillu_types::ModelTier::Cheap)
                                    .with_auto_compaction(session_auto_compaction())
                                    .await?,
                            )
                            .with_batch_config(batching),
                            recorder,
                        })
                    }
                },
            )
            .unwrap()
            .build(caps)
            .await
            .unwrap();
        let (_, mut modules) = modules.into_parts();
        modules.remove(0)
    }

    async fn build_memory_module(
        caps: &CapabilityProviders,
        memory_caps: MemoryCapabilities,
    ) -> nuillu_module::AllocatedModule {
        let modules = ModuleRegistry::new()
            .register(module_policy(), move |caps| {
                let memory_caps = memory_caps.clone();
                async move {
                    Ok(MemoryModule::new(
                        caps.memo_updated_inbox(),
                        caps.cognition_log_updated_inbox(),
                        caps.blackboard_reader(),
                        caps.cognition_log_reader(),
                        caps.memory_metadata_reader(),
                        memory_caps.writer(),
                        memory_caps.deleter(),
                        memory_caps.retriever(),
                        caps.llm("main")
                            .with_tier(nuillu_types::ModelTier::Cheap)
                            .into(),
                        caps.session("main")
                            .with_tier(nuillu_types::ModelTier::Cheap)
                            .with_auto_compaction(session_auto_compaction())
                            .await?,
                    ))
                }
            })
            .unwrap()
            .build(caps)
            .await
            .unwrap();
        let (_, mut modules) = modules.into_parts();
        modules.remove(0)
    }

    fn cognition_entry(index: u64, content: &str) -> nuillu_blackboard::CognitionLogEntry {
        let source = ModuleInstanceId::new(builtin::cognition_gate(), ReplicaIndex::ZERO);
        nuillu_blackboard::CognitionLogEntry {
            at: Utc.timestamp_opt(index as i64, 0).unwrap(),
            text: content.to_owned(),
            origin: nuillu_blackboard::CognitionLogOrigin::direct(source),
        }
    }

    fn tool_scenario(name: &str, arguments_json: serde_json::Value) -> MockTextScenario {
        MockTextScenario::events(vec![
            Ok(RawTextTurnEvent::Started {
                request_id: Some(format!("{name}-request")),
                model: "mock".into(),
            }),
            Ok(RawTextTurnEvent::ToolCallChunk {
                id: format!("{name}-call").into(),
                name: name.to_owned().into(),
                arguments_json_delta: arguments_json.to_string(),
            }),
            Ok(RawTextTurnEvent::Completed {
                request_id: Some(format!("{name}-request")),
                finish_reason: FinishReason::ToolCall,
                usage: Usage::zero(),
            }),
        ])
    }

    #[test]
    fn insert_memory_args_accepts_name_arrays_without_rank() {
        let args: InsertMemoryArgs = serde_json::from_value(serde_json::json!({
            "concepts": ["Ryo", "super red apple"],
            "content": "I successfully responded to Ryo's greeting (\"Hi\") and began integrating the social exchange with the persistent visual context of the super red apple.",
            "kind": "episode",
            "tags": ["dialogue_flow", "successful_engagement", "context_binding"]
        }))
        .expect("compatible insert_memory payload should deserialize");

        assert_eq!(args.kind, MemoryKind::Episode);

        let concepts = args
            .concepts
            .into_iter()
            .map(memory_concept_from_input)
            .collect::<Vec<_>>();
        assert_eq!(
            concepts,
            vec![
                MemoryConcept {
                    label: "Ryo".to_owned(),
                    mention_text: None,
                    loose_type: None,
                    confidence: 1.0,
                },
                MemoryConcept {
                    label: "super red apple".to_owned(),
                    mention_text: None,
                    loose_type: None,
                    confidence: 1.0,
                },
            ]
        );

        let tags = args
            .tags
            .into_iter()
            .map(memory_tag_from_input)
            .collect::<Vec<_>>();
        assert_eq!(
            tags,
            vec![
                MemoryTag::operational("dialogue_flow"),
                MemoryTag::operational("successful_engagement"),
                MemoryTag::operational("context_binding"),
            ]
        );
    }

    #[test]
    fn skip_memory_args_accepts_reason_payload() {
        let args: SkipMemoryArgs = serde_json::from_value(serde_json::json!({
            "reason": "no durable information"
        }))
        .expect("compatible skip_memory payload should deserialize");

        assert_eq!(args.reason, "no durable information");

        let schema =
            serde_json::to_value(schemars::schema_for!(SkipMemoryArgs)).expect("schema serializes");
        assert_eq!(
            schema.pointer("/properties/reason/type"),
            Some(&serde_json::json!("string"))
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn next_batch_collects_memo_and_cognition_updates_with_idle_window() {
        let (blackboard, caps, memory_caps, _lutum) = test_caps();
        let recorder = Rc::new(RefCell::new(Vec::new()));
        let mut module = build_recording_memory(
            &caps,
            memory_caps,
            recorder.clone(),
            MemoryBatchConfig {
                silent_window: Duration::from_millis(10),
                budget: Duration::from_millis(50),
            },
        )
        .await;
        let harness = caps.internal_harness_io();
        let source = ModuleInstanceId::new(builtin::sensory(), ReplicaIndex::ZERO);
        let memo = blackboard
            .update_memo(
                source.clone(),
                "fresh memo".to_owned(),
                Utc.timestamp_opt(0, 0).unwrap(),
            )
            .await;
        harness
            .memo_updated_mailbox()
            .publish(MemoUpdated {
                owner: source.clone(),
                index: memo.index,
            })
            .await
            .expect("memory memo-update subscriber exists");

        let delayed_cognition = async {
            tokio::time::sleep(Duration::from_millis(2)).await;
            let source = ModuleInstanceId::new(builtin::cognition_gate(), ReplicaIndex::ZERO);
            blackboard
                .append_cognition_log(source.clone(), cognition_entry(1, "fresh cognition"))
                .await;
            harness
                .cognition_log_updated_mailbox()
                .publish(CognitionLogUpdated::EntryAppended { source })
                .await
                .expect("memory cognition-update subscriber exists");
        };
        let next_batch = module.next_batch();

        let (batch_result, _) = tokio::join!(next_batch, delayed_cognition);
        batch_result.expect("memory next batch succeeds");

        assert_eq!(recorder.borrow().as_slice(), &[(1, 1)]);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn activation_uses_update_evidence_and_requires_creation_tool() {
        let now = Utc.timestamp_opt(10, 0).unwrap();
        let adapter =
            CapturingAdapter::new(MockLlmAdapter::new().with_text_scenario(tool_scenario(
                "skip_memory",
                serde_json::json!({ "reason": "no durable information" }),
            )));
        let observed = adapter.clone();
        let blackboard = Blackboard::default();
        let (blackboard, caps, memory_caps, lutum) =
            test_caps_with_adapter(blackboard, Arc::new(adapter));
        let mut module = build_memory_module(&caps, memory_caps).await;
        let harness = caps.internal_harness_io();
        let sensory = ModuleInstanceId::new(builtin::sensory(), ReplicaIndex::ZERO);
        let memo = blackboard
            .update_memo(sensory.clone(), "fresh ordinary memo".to_owned(), now)
            .await;
        harness
            .memo_updated_mailbox()
            .publish(MemoUpdated {
                owner: sensory,
                index: memo.index,
            })
            .await
            .unwrap();
        let cognition_source = ModuleInstanceId::new(builtin::cognition_gate(), ReplicaIndex::ZERO);
        blackboard
            .append_cognition_log(
                cognition_source.clone(),
                nuillu_blackboard::CognitionLogEntry {
                    at: now,
                    text: "fresh cognition evidence".to_owned(),
                    origin: nuillu_blackboard::CognitionLogOrigin::direct(cognition_source.clone()),
                },
            )
            .await;
        harness
            .cognition_log_updated_mailbox()
            .publish(CognitionLogUpdated::EntryAppended {
                source: cognition_source,
            })
            .await
            .unwrap();

        let batch = module.next_batch().await.unwrap();
        module
            .activate(&activate_cx(&lutum, now), &batch)
            .await
            .unwrap();

        let turns = observed.text_turns();
        assert_eq!(turns.len(), 1);
        assert_eq!(turns[0].config.tool_choice, AdapterToolChoice::Required);
        let inputs = observed.text_inputs();
        let user_messages = message_texts_with_role(&inputs[0], InputMessageRole::User);
        let joined = user_messages.join("\n\n");
        assert!(joined.contains("fresh ordinary memo"));
        assert!(joined.contains("fresh cognition evidence"));
        assert!(joined.contains("New memo-log entries: 1"));
        assert!(joined.contains("New cognition-log entries: 1"));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn inserted_memory_with_empty_related_search_is_kept_without_novelty_turn() {
        let now = Utc.timestamp_opt(20, 0).unwrap();
        let adapter =
            CapturingAdapter::new(MockLlmAdapter::new().with_text_scenario(tool_scenario(
                "insert_memory",
                serde_json::json!({
                    "content": "Ryo mentioned a blue marker on the desk.",
                    "kind": "statement",
                    "concepts": ["Ryo", "blue marker"],
                    "tags": ["dialogue_flow"]
                }),
            )));
        let observed = adapter.clone();
        let blackboard = Blackboard::default();
        let (blackboard, caps, memory_caps, lutum) =
            test_caps_with_adapter(blackboard, Arc::new(adapter));
        let mut module = build_memory_module(&caps, memory_caps).await;
        let harness = caps.internal_harness_io();
        let sensory = ModuleInstanceId::new(builtin::sensory(), ReplicaIndex::ZERO);
        let memo = blackboard
            .update_memo(
                sensory.clone(),
                "Ryo mentioned a blue marker on the desk.".to_owned(),
                now,
            )
            .await;
        harness
            .memo_updated_mailbox()
            .publish(MemoUpdated {
                owner: sensory,
                index: memo.index,
            })
            .await
            .unwrap();

        let batch = module.next_batch().await.unwrap();
        module
            .activate(&activate_cx(&lutum, now), &batch)
            .await
            .unwrap();

        assert_eq!(observed.text_turns().len(), 1);
        let metadata = blackboard.read(|bb| bb.memory_metadata().clone()).await;
        assert_eq!(metadata.len(), 1);
        let (_, meta) = metadata.iter().next().unwrap();
        assert_eq!(meta.rank, MemoryRank::ShortTerm);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn redundant_novelty_decision_deletes_new_memory_and_reinforces_existing() {
        let now = Utc.timestamp_opt(30, 0).unwrap();
        let store = Rc::new(SearchableMemoryStore::default());
        store.seed("existing-memory", "Ryo likes coffee.", now);
        let adapter = CapturingAdapter::new(
            MockLlmAdapter::new()
                .with_text_scenario(tool_scenario(
                    "insert_memory",
                    serde_json::json!({
                        "content": "Ryo likes coffee.",
                        "kind": "statement",
                        "concepts": ["Ryo", "coffee"],
                        "tags": ["dialogue_flow"]
                    }),
                ))
                .with_text_scenario(tool_scenario(
                    "delete_redundant_memory",
                    serde_json::json!({
                        "redundant_with": ["existing-memory"],
                        "reason": "new memory only restates existing content"
                    }),
                )),
        );
        let observed = adapter.clone();
        let blackboard = Blackboard::default();
        let (blackboard, caps, memory_caps, lutum) =
            test_caps_with_store_and_adapter(blackboard, store.clone(), Arc::new(adapter));
        let mut module = build_memory_module(&caps, memory_caps).await;
        let harness = caps.internal_harness_io();
        let sensory = ModuleInstanceId::new(builtin::sensory(), ReplicaIndex::ZERO);
        let memo = blackboard
            .update_memo(sensory.clone(), "Ryo likes coffee.".to_owned(), now)
            .await;
        harness
            .memo_updated_mailbox()
            .publish(MemoUpdated {
                owner: sensory,
                index: memo.index,
            })
            .await
            .unwrap();

        let batch = module.next_batch().await.unwrap();
        module
            .activate(&activate_cx(&lutum, now), &batch)
            .await
            .unwrap();

        assert_eq!(observed.text_turns().len(), 2);
        let records = store.records();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].index.as_str(), "existing-memory");
        let metadata = blackboard.read(|bb| bb.memory_metadata().clone()).await;
        assert!(
            !metadata
                .keys()
                .any(|index| index.as_str() != "existing-memory")
        );
        let existing = metadata
            .get(&MemoryIndex::new("existing-memory"))
            .expect("existing memory metadata should be retained");
        assert_eq!(existing.reinforcement_count, 1);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn recall_context_memory_is_kept_even_when_recalled_fact_exists() {
        let now = Utc.timestamp_opt(40, 0).unwrap();
        let store = Rc::new(SearchableMemoryStore::default());
        store.seed("existing-memory", "Ryo likes coffee.", now);
        let adapter = CapturingAdapter::new(
            MockLlmAdapter::new()
                .with_text_scenario(tool_scenario(
                    "insert_memory",
                    serde_json::json!({
                        "content": "I recalled that Ryo likes coffee while choosing how to answer his drink question.",
                        "kind": "episode",
                        "concepts": ["Ryo", "coffee"],
                        "tags": ["recall_context", "dialogue_flow"]
                    }),
                ))
                .with_text_scenario(tool_scenario(
                    "keep_new_memory",
                    serde_json::json!({
                        "reason": "the recall/use context is new durable information"
                    }),
                )),
        );
        let observed = adapter.clone();
        let blackboard = Blackboard::default();
        let (blackboard, caps, memory_caps, lutum) =
            test_caps_with_store_and_adapter(blackboard, store.clone(), Arc::new(adapter));
        let mut module = build_memory_module(&caps, memory_caps).await;
        let harness = caps.internal_harness_io();
        let sensory = ModuleInstanceId::new(builtin::sensory(), ReplicaIndex::ZERO);
        let memo = blackboard
            .update_memo(
                sensory.clone(),
                "I recalled that Ryo likes coffee while choosing how to answer.".to_owned(),
                now,
            )
            .await;
        harness
            .memo_updated_mailbox()
            .publish(MemoUpdated {
                owner: sensory,
                index: memo.index,
            })
            .await
            .unwrap();

        let batch = module.next_batch().await.unwrap();
        module
            .activate(&activate_cx(&lutum, now), &batch)
            .await
            .unwrap();

        assert_eq!(observed.text_turns().len(), 2);
        let records = store.records();
        assert_eq!(records.len(), 2);
        assert!(records.iter().any(|record| {
            record
                .content
                .as_str()
                .contains("I recalled that Ryo likes coffee")
        }));
        let text_inputs = observed.text_inputs();
        let novelty_user_messages =
            message_texts_with_role(&text_inputs[1], InputMessageRole::User);
        assert!(
            novelty_user_messages
                .join("\n\n")
                .contains("recall/use context")
        );
    }

    #[test]
    fn insert_memory_schema_does_not_expose_runtime_metadata() {
        let schema = serde_json::to_value(schemars::schema_for!(InsertMemoryArgs))
            .expect("insert memory schema should serialize");

        assert_eq!(schema.pointer("/properties/rank"), None);
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
    fn novelty_tool_args_accept_expected_payloads() {
        let keep: KeepNewMemoryArgs = serde_json::from_value(serde_json::json!({
            "reason": "recall context is new"
        }))
        .expect("compatible keep_new_memory payload should deserialize");
        assert_eq!(keep.reason, "recall context is new");

        let delete: DeleteRedundantMemoryArgs = serde_json::from_value(serde_json::json!({
            "redundant_with": ["existing-memory"],
            "reason": "new memory only restates existing content"
        }))
        .expect("compatible delete_redundant_memory payload should deserialize");
        assert_eq!(
            delete.redundant_with,
            vec![MemoryIndex::new("existing-memory")]
        );

        let keep_schema = serde_json::to_value(schemars::schema_for!(KeepNewMemoryArgs))
            .expect("keep schema serializes");
        assert_eq!(
            keep_schema.pointer("/properties/reason/type"),
            Some(&serde_json::json!("string"))
        );

        let delete_schema = serde_json::to_value(schemars::schema_for!(DeleteRedundantMemoryArgs))
            .expect("delete schema serializes");
        assert_eq!(
            delete_schema.pointer("/properties/redundant_with/items/$ref"),
            Some(&serde_json::json!("#/$defs/MemoryIndex"))
        );
        assert_eq!(
            delete_schema.pointer("/$defs/MemoryIndex/type"),
            Some(&serde_json::json!("string"))
        );
        assert_eq!(
            delete_schema.pointer("/properties/reason/type"),
            Some(&serde_json::json!("string"))
        );
    }
}
