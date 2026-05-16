use std::collections::HashMap;
use std::collections::HashSet;
use std::time::Duration;

use anyhow::{Context, Result};
use async_trait::async_trait;
use lutum::{Session, TextStepOutcomeWithTools, ToolResult};
use nuillu_module::{
    AllocationReader, AllocationUpdatedInbox, CognitionLogEntryRecord, CognitionLogEvictedInbox,
    LlmAccess, MemoryMetadataReader, Module, SessionCompactionConfig,
    SessionCompactionProtectedPrefix, compact_session_if_needed, format_current_attention_guidance,
    format_memory_trace_inventory, memory_rank_counts, push_formatted_cognition_log_batch,
    render_memory_for_llm,
};
use nuillu_types::{MemoryIndex, MemoryRank};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use tokio::time::Instant;

use crate::store::{
    MemoryConcept, MemoryKind, MemoryRetriever, MemoryTag, MemoryUsageTarget, MemoryWriter,
    NewMemory,
};

// ---------------------------------------------------------------------------
// MemoryModule

const SYSTEM_PROMPT: &str = r#"You are the memory module.
Inspect evicted cognitive workspace evidence and decide whether to preserve short, useful memories.
Memory is remembered evidence, not a fact table or current-truth projection. Store a concise,
normalized natural-language memory, usually one to three sentences. If source context matters,
include it in the memory sentence itself, for example "Ryo said he recently moved to Kyoto."
Use evicted cognition-log entries as candidate evidence. Memo-log entries are non-conscious working
traces and are not valid direct memory evidence. Allocation guidance from allocation-controller may
contain explicit preservation candidates from other modules, but those candidates are prioritization
context rather than write commands. Use insert_memory only for concrete information likely to matter
later and grounded in cognition-log evidence. You may reject, normalize, merge, and deduplicate
observations and guidance.
When related existing memories are provided, decide whether the evicted evidence is already covered
by one of those memories. If it is, call reinforce_memory for that candidate instead of inserting a
duplicate. If the evidence adds new detail, contradicts, updates, or is not clearly covered by an
existing candidate, write a new short-term memory with insert_memory.
insert_memory always writes short-term memory; later access, compaction, or other memory-system
mechanisms may change rank outside this tool. When calling insert_memory, prefer kinds episode,
statement, reflection, hypothesis, dream, procedure, or plan. Concepts and tags are simple name
arrays, for example concepts ["Ryo", "super red apple"] and tags ["dialogue_flow"]; do not invent
concept ids or tag ids. Do not provide decay or occurrence timestamps; runtime stamps them.
For each insert, classify the loose memory kind, extract mentioned concepts, and add operational
tags only when useful. Do not create memory-to-memory links, infer corrections, overwrite old
memories, decide what is currently true, or treat user/agent/import source as authority.
Entries beginning "Internal dream simulation, not a verified fact:" are associative internal
simulations, not evidence. Do not store them as factual memories; preserve them only when explicitly
framed as dream or hypothesis material with dream/hypothesis kind and operational tags."#;

const SHORT_TERM_MEMORY_DECAY_SECS: i64 = 86_400;
const RELATED_MEMORY_SEARCH_LIMIT: usize = 4;
const DEFAULT_MEMORY_BATCH_SILENT_WINDOW: Duration = Duration::from_millis(100);
const DEFAULT_MEMORY_BATCH_BUDGET: Duration = Duration::from_secs(1);
const COMPACTED_MEMORY_SESSION_PREFIX: &str = "Compacted memory session history:";
const SESSION_COMPACTION_PROMPT: &str = r#"You compact the memory module's persistent session history.
Summarize only the prefix transcript you receive. Preserve cognition-log facts, memory requests,
inserted memory content, rejected candidates, and deduplication decisions future memory decisions
need. Do not invent facts. Return plain text only."#;

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

#[lutum::tool_input(name = "reinforce_memory", output = ReinforceMemoryOutput)]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct ReinforceMemoryArgs {
    pub index: MemoryIndex,
    pub reason: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct ReinforceMemoryOutput {
    pub reinforced: bool,
    pub index: MemoryIndex,
    pub message: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema, lutum::Toolset)]
pub enum MemoryTools {
    InsertMemory(InsertMemoryArgs),
    ReinforceMemory(ReinforceMemoryArgs),
}

pub struct MemoryModule {
    owner: nuillu_types::ModuleId,
    cognition_evictions: CognitionLogEvictedInbox,
    allocation_updates: AllocationUpdatedInbox,
    allocation: AllocationReader,
    memory_metadata: MemoryMetadataReader,
    memory: MemoryWriter,
    memory_retriever: MemoryRetriever,
    llm: LlmAccess,
    session: Session,
    session_compaction: SessionCompactionConfig,
    system_prompt: std::sync::OnceLock<String>,
    batching: MemoryBatchConfig,
}

#[derive(Clone, Debug)]
struct MemoryMetadataContext {
    index: String,
    rank: MemoryRank,
    occurred_at: String,
    decay_remaining_secs: i64,
    access_count: u32,
    use_count: u32,
    reinforcement_count: u32,
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
        cognition_evictions: CognitionLogEvictedInbox,
        allocation_updates: AllocationUpdatedInbox,
        allocation: AllocationReader,
        memory_metadata: MemoryMetadataReader,
        memory: MemoryWriter,
        memory_retriever: MemoryRetriever,
        llm: LlmAccess,
    ) -> Self {
        Self {
            owner: nuillu_types::ModuleId::new(<Self as Module>::id()).expect("memory id is valid"),
            cognition_evictions,
            allocation_updates,
            allocation,
            memory_metadata,
            memory,
            memory_retriever,
            llm,
            session: Session::new(),
            session_compaction: SessionCompactionConfig::default(),
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
                cx.modules(),
                &self.owner,
                cx.identity_memories(),
                cx.core_policies(),
                cx.now(),
            )
        })
    }

    #[tracing::instrument(skip_all, err(Debug, level = "warn"))]
    async fn activate(
        &mut self,
        cx: &nuillu_module::ActivateCx<'_>,
        batch: &MemoryBatch,
    ) -> Result<()> {
        push_formatted_cognition_log_batch(&mut self.session, &batch.cognition_log, cx.now());
        let (memory_metadata, rank_counts) = self
            .memory_metadata
            .read(|metadata_map| {
                let mut metadata = metadata_map
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
                        use_count: item.use_count,
                        reinforcement_count: item.reinforcement_count,
                    })
                    .collect::<Vec<_>>();
                metadata.sort_by(|left, right| left.index.cmp(&right.index));
                (metadata, memory_rank_counts(metadata_map))
            })
            .await;
        let allocation = self.allocation.snapshot().await;
        let allocation_guidance = allocation.for_module(&self.owner).guidance;
        let related_memories = self
            .related_memory_candidates(&batch.cognition_log, cx.now())
            .await?;
        let reinforcement_targets = related_memories
            .iter()
            .map(|candidate| (candidate.target.index.clone(), candidate.target.clone()))
            .collect::<HashMap<_, _>>();

        let system_prompt = self.system_prompt(cx).to_owned();
        self.session.push_ephemeral_system(system_prompt);
        self.session.push_user(format_memory_activation_request(
            &allocation_guidance,
            batch.allocation_updated,
            batch.cognition_log.len(),
        ));
        if let Some(metadata_context) = format_memory_metadata_candidates(&memory_metadata) {
            self.session.push_ephemeral_user(metadata_context);
        }
        if let Some(candidate_context) = format_related_memory_candidates(&related_memories) {
            self.session.push_ephemeral_user(candidate_context);
        }
        self.session
            .push_ephemeral_system(format_memory_decision_context(&rank_counts, &allocation));

        for _ in 0..4 {
            let lutum = self.llm.lutum().await;
            let outcome = self
                .session
                .text_turn(&lutum)
                .tools::<MemoryTools>()
                .available_tools([
                    MemoryToolsSelector::InsertMemory,
                    MemoryToolsSelector::ReinforceMemory,
                ])
                .collect()
                .await
                .context("memory text turn failed")?;

            match outcome {
                TextStepOutcomeWithTools::Finished(result) => {
                    compact_session_if_needed(
                        &mut self.session,
                        result.usage.input_tokens,
                        cx.session_compaction(),
                        self.session_compaction,
                        SessionCompactionProtectedPrefix::None,
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
                        cx.session_compaction(),
                        self.session_compaction,
                        SessionCompactionProtectedPrefix::None,
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
                        match call {
                            MemoryToolsCall::InsertMemory(call) => {
                                let output = self
                                    .insert_memory(call.input.clone())
                                    .await
                                    .context("run insert_memory tool")?;
                                let result = call
                                    .complete(output)
                                    .context("complete insert_memory tool call")?;
                                results.push(result);
                            }
                            MemoryToolsCall::ReinforceMemory(call) => {
                                let output = self
                                    .reinforce_memory(call.input.clone(), &reinforcement_targets)
                                    .await
                                    .context("run reinforce_memory tool")?;
                                let result = call
                                    .complete(output)
                                    .context("complete reinforce_memory tool call")?;
                                results.push(result);
                            }
                        }
                    }
                    round
                        .commit(&mut self.session, results)
                        .context("commit memory tool round")?;
                    compact_session_if_needed(
                        &mut self.session,
                        input_tokens,
                        cx.session_compaction(),
                        self.session_compaction,
                        SessionCompactionProtectedPrefix::None,
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
        let record = self
            .memory
            .insert_entry_now(
                NewMemory {
                    content: nuillu_types::MemoryContent::new(args.content),
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
        Ok(InsertMemoryOutput {
            index: record.index.to_string(),
        })
    }

    async fn reinforce_memory(
        &self,
        args: ReinforceMemoryArgs,
        targets: &HashMap<MemoryIndex, MemoryUsageTarget>,
    ) -> Result<ReinforceMemoryOutput> {
        let Some(target) = targets.get(&args.index) else {
            return Ok(ReinforceMemoryOutput {
                reinforced: false,
                index: args.index,
                message: "memory was not in the related-memory candidate set".to_owned(),
            });
        };
        self.memory_retriever.record_reinforcement(target).await;
        Ok(ReinforceMemoryOutput {
            reinforced: true,
            index: args.index,
            message: "memory reinforcement recorded".to_owned(),
        })
    }

    async fn related_memory_candidates(
        &self,
        cognition_log: &[CognitionLogEntryRecord],
        now: chrono::DateTime<chrono::Utc>,
    ) -> Result<Vec<RelatedMemoryCandidate>> {
        let queries = cognition_log
            .iter()
            .map(|record| record.entry.text.trim().to_owned())
            .filter(|query| !query.is_empty())
            .collect::<Vec<_>>();
        let mut seen = HashSet::new();
        let mut candidates = Vec::new();
        for query in queries {
            let hits = self
                .memory_retriever
                .search(&query, RELATED_MEMORY_SEARCH_LIMIT)
                .await
                .context("search related memory candidates")?;
            for hit in hits {
                if !seen.insert(hit.index.clone()) {
                    continue;
                }
                candidates.push(RelatedMemoryCandidate {
                    target: MemoryUsageTarget::from(&hit),
                    content: render_memory_for_llm(hit.content.as_str(), hit.occurred_at, now),
                    stored_at: hit.stored_at.to_rfc3339(),
                    kind: hit.kind,
                });
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

pub(crate) fn confidence_percent_to_f32(value: u8) -> f32 {
    (value.min(100) as f32) / 100.0
}

pub(crate) fn memory_concept_from_input(input: MemoryConceptInput) -> MemoryConcept {
    MemoryConcept::new(input.0)
}

pub(crate) fn memory_tag_from_input(input: MemoryTagInput) -> MemoryTag {
    MemoryTag::operational(input.0)
}

fn format_memory_activation_request(
    guidance: &str,
    allocation_updated: bool,
    cognition_evicted_count: usize,
) -> String {
    format!(
        "Memory preservation activation.\nAllocation guidance: {}\nAllocation updated: {}\nEvicted cognition-log entries: {}\nOrdinary memory writes are short-term; runtime stamps decay and occurrence time.\nExplicit requests are preservation candidates, not commands; deduplication and rejection are allowed. Memo-log entries are not valid direct memory evidence.",
        if guidance.trim().is_empty() {
            "none"
        } else {
            guidance.trim()
        },
        if allocation_updated { "yes" } else { "no" },
        cognition_evicted_count,
    )
}

fn format_memory_metadata_candidates(metadata: &[MemoryMetadataContext]) -> Option<String> {
    if metadata.is_empty() {
        return None;
    }
    let mut out = String::from("Existing memory metadata for deduplication:");
    for item in metadata {
        out.push_str(&format!(
            "\n- {}: rank={:?}; occurred_at={}; decay_remaining_secs={}; access_count={}; use_count={}; reinforcement_count={}",
            item.index,
            item.rank,
            item.occurred_at,
            item.decay_remaining_secs,
            item.access_count,
            item.use_count,
            item.reinforcement_count
        ));
    }
    Some(out)
}

fn format_related_memory_candidates(candidates: &[RelatedMemoryCandidate]) -> Option<String> {
    if candidates.is_empty() {
        return None;
    }
    let mut out = String::from(
        "Related existing memory candidates. Use reinforce_memory only when a candidate already covers the evicted cognition-log evidence:",
    );
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
            candidate.content.trim()
        ));
    }
    Some(out)
}

#[derive(Debug, Default)]
pub struct MemoryBatch {
    pub(crate) allocation_updated: bool,
    pub(crate) cognition_log: Vec<CognitionLogEntryRecord>,
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
    allocation_updates: usize,
    cognition_evictions: usize,
}

impl ReadyCounts {
    fn is_empty(self) -> bool {
        self.allocation_updates == 0 && self.cognition_evictions == 0
    }
}

impl MemoryBatch {
    fn allocation_update() -> Self {
        Self {
            allocation_updated: true,
            ..Self::default()
        }
    }

    fn cognition_log_eviction(record: CognitionLogEntryRecord) -> Self {
        Self {
            cognition_log: vec![record],
            ..Self::default()
        }
    }

    fn mark_allocation_updated(&mut self) {
        self.allocation_updated = true;
    }
}

impl MemoryModule {
    async fn next_batch(&mut self) -> Result<MemoryBatch> {
        let mut batch = self.await_first_batch().await?;
        let _ = self.collect_ready_events_into_batch(&mut batch)?;
        self.collect_eviction_burst(&mut batch).await?;
        Ok(batch)
    }

    async fn await_first_batch(&mut self) -> Result<MemoryBatch> {
        let batch = tokio::select! {
            update = self.allocation_updates.next_item() => {
                let _ = update?;
                MemoryBatch::allocation_update()
            }
            evicted = self.cognition_evictions.next_item() => {
                MemoryBatch::cognition_log_eviction(evicted?.body)
            }
        };
        Ok(batch)
    }

    async fn collect_eviction_burst(&mut self, batch: &mut MemoryBatch) -> Result<()> {
        let mut waited = Duration::ZERO;
        while waited < self.batching.budget {
            let remaining = self.batching.budget.saturating_sub(waited);
            let wait_for = std::cmp::min(self.batching.silent_window, remaining);
            if wait_for.is_zero() {
                break;
            }

            let started = Instant::now();
            tokio::select! {
                update = self.allocation_updates.next_item() => {
                    let _ = update?;
                    waited += std::cmp::min(started.elapsed(), wait_for);
                    let _ = self.collect_ready_events_into_batch(batch)?;
                }
                evicted = self.cognition_evictions.next_item() => {
                    batch.cognition_log.push(evicted?.body);
                    waited += std::cmp::min(started.elapsed(), wait_for);
                    let _ = self.collect_ready_events_into_batch(batch)?;
                }
                _ = tokio::time::sleep(wait_for) => {
                    waited += wait_for;
                    let ready = self.collect_ready_events_into_batch(batch)?;
                    if ready.is_empty() {
                        break;
                    }
                }
            }
        }
        Ok(())
    }

    fn collect_ready_events_into_batch(&mut self, batch: &mut MemoryBatch) -> Result<ReadyCounts> {
        let allocation_updates = self.allocation_updates.take_ready_items()?.items.len();
        if allocation_updates > 0 {
            batch.mark_allocation_updated();
        }

        let cognition_evictions = self.cognition_evictions.take_ready_items()?;
        let cognition_count = cognition_evictions.items.len();
        batch.cognition_log.extend(
            cognition_evictions
                .items
                .into_iter()
                .map(|envelope| envelope.body),
        );

        Ok(ReadyCounts {
            allocation_updates,
            cognition_evictions: cognition_count,
        })
    }
}

#[async_trait(?Send)]
impl Module for MemoryModule {
    type Batch = MemoryBatch;

    fn id() -> &'static str {
        "memory"
    }

    fn role_description() -> &'static str {
        "Preserves useful information by inserting normalized, deduplicated memory entries from evicted cognition-log evidence and allocation-controller preservation guidance."
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
    use std::rc::Rc;
    use std::sync::Arc;

    use chrono::{TimeZone, Utc};
    use lutum::{Lutum, MockLlmAdapter, SharedPoolBudgetManager, SharedPoolBudgetOptions};
    use nuillu_blackboard::{Blackboard, Bpm, ModulePolicy, linear_ratio_fn};
    use nuillu_module::ports::{NoopCognitionLogRepository, SystemClock};
    use nuillu_module::{CapabilityProviderPorts, CapabilityProviders, LutumTiers, ModuleRegistry};
    use nuillu_types::{ModuleInstanceId, ReplicaCapRange, ReplicaIndex, builtin};

    use crate::{MemoryCapabilities, NoopMemoryStore};

    type BatchRecorder = Rc<RefCell<Vec<(bool, usize)>>>;

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

        fn role_description() -> &'static str {
            MemoryModule::role_description()
        }

        async fn next_batch(&mut self) -> Result<Self::Batch> {
            let batch = self.inner.next_batch().await?;
            self.recorder
                .borrow_mut()
                .push((batch.allocation_updated, batch.cognition_log.len()));
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

    fn test_caps() -> (Blackboard, CapabilityProviders, MemoryCapabilities) {
        let blackboard = Blackboard::default();
        let adapter = Arc::new(MockLlmAdapter::new());
        let budget = SharedPoolBudgetManager::new(SharedPoolBudgetOptions::default());
        let lutum = Lutum::new(adapter, budget);
        let clock = Rc::new(SystemClock);
        let caps = CapabilityProviders::new(CapabilityProviderPorts {
            blackboard: blackboard.clone(),
            cognition_log_port: Rc::new(NoopCognitionLogRepository),
            clock: clock.clone(),
            tiers: LutumTiers {
                cheap: lutum.clone(),
                default: lutum.clone(),
                premium: lutum,
            },
        });
        let memory_caps = MemoryCapabilities::new(
            blackboard.clone(),
            clock,
            Rc::new(NoopMemoryStore),
            Vec::new(),
        );
        (blackboard, caps, memory_caps)
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
                move |caps| RecordingMemory {
                    inner: MemoryModule::new(
                        caps.cognition_log_evicted_inbox(),
                        caps.allocation_updated_inbox(),
                        caps.allocation_reader(),
                        caps.memory_metadata_reader(),
                        memory_caps.writer(),
                        memory_caps.retriever(),
                        caps.llm_access(),
                    )
                    .with_batch_config(batching),
                    recorder: recorder.clone(),
                },
            )
            .unwrap()
            .build(caps)
            .await
            .unwrap();
        let (_, mut modules) = modules.into_parts();
        modules.remove(0)
    }

    fn cognition_record(index: u64, content: &str) -> CognitionLogEntryRecord {
        CognitionLogEntryRecord {
            index,
            source: ModuleInstanceId::new(builtin::cognition_gate(), ReplicaIndex::ZERO),
            entry: nuillu_blackboard::CognitionLogEntry {
                at: Utc.timestamp_opt(index as i64, 0).unwrap(),
                text: content.to_owned(),
            },
        }
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

    #[tokio::test]
    async fn next_batch_collects_cognition_eviction_burst_with_idle_window() {
        let (_blackboard, caps, memory_caps) = test_caps();
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
        let cognition_evictions = harness.cognition_log_evicted_mailbox();

        cognition_evictions
            .publish(cognition_record(0, "first"))
            .await
            .expect("memory cognition-eviction subscriber exists");

        let delayed_evictions = async {
            tokio::time::sleep(Duration::from_millis(2)).await;
            cognition_evictions
                .publish(cognition_record(1, "second"))
                .await
                .expect("memory cognition-eviction subscriber exists");
        };
        let next_batch = module.next_batch();

        let (batch_result, _) = tokio::join!(next_batch, delayed_evictions);
        batch_result.expect("memory next batch succeeds");

        assert_eq!(recorder.borrow().as_slice(), &[(false, 2)]);
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
}
