use std::collections::{HashMap, HashSet};

use anyhow::{Context, Result};
use async_trait::async_trait;
use lutum::{
    IntoToolResult, ModelInput, RejectedToolCall, RejectedToolSource, StructuredTurnOutcome,
    TextStepOutcomeWithTools, ToolResult,
};
use nuillu_blackboard::MemoryMetadata;
use nuillu_module::{
    BlackboardReader, FixedTierLlmAccess, InteroceptiveUpdatedInbox, LlmAccess, Module,
};
use nuillu_types::{MemoryIndex, MemoryRank};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::common::{
    GetMemoriesOutput, MemoryContentView, MemoryMetadataContext, memory_record_to_view,
};
use crate::memory::{
    MemoryConceptInput, MemoryTagInput, SHORT_TERM_MEMORY_DECAY_SECS, memory_concept_from_input,
    memory_tag_from_input,
};
use crate::store::{MemoryCompactor, MemoryRecord};

const MIN_MERGE_SOURCES: usize = 2;
const MAX_MERGE_SOURCES: usize = 3;

const SYSTEM_PROMPT: &str = r#"You are the memory-compaction module.
Inspect candidate memories, fetch source contents when useful, and consolidate memories whose
contents overlap enough that preserving them separately adds retrieval noise. Compatible variations
around the same subject, object, preference, or procedure can be merged when one concise summary
preserves the useful distinctions. Keep memories separate when they contain unrelated subjects,
conflicting evidence, or details that would be lost in a summary. Compaction output is one
replacement summary memory; source memories are removed from live retrieval after the replacement is
written. Treat merge_memories as destructive for live memory: after it succeeds, the source entries
will not be available to ordinary recall or future compaction work. Do not create memory-to-memory
links here. Do not collapse evidence into a single current fact.

The candidate list is the complete maintenance work item and contains only live ShortTerm memories.
Never use indexes that are not in the candidate list. When multiple candidates are present, inspect
their contents with get_memories before deciding whether to merge or leave them unchanged. Merge at
most three source memories at a time.

Tool input rules: merge_memories takes source_indexes, merged_content, concepts, and tags only. The
runtime chooses the merged rank, decay, and storage metadata."#;

const AUDIT_PROMPT: &str = r#"You are the fresh pre-commit auditor for memory compaction.
Decide whether the proposed replacement summary is safe to write before the source memories leave
live retrieval. Approve only when all source memories are about the same concrete subject and the
summary preserves the useful concrete facts from every source. Reject if the summary is metadata-only,
too generic, loses source facts, merges unrelated subjects, collapses evidence into one current fact,
or if you are uncertain. Return structured output only."#;

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

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct CompactionAuditOutput {
    pub approved: bool,
    pub reason: String,
}

#[derive(Serialize)]
struct CompactionAuditInput {
    destructive_warning: &'static str,
    source_memories: Vec<MemoryContentView>,
    proposed_summary: String,
    proposed_concepts: Vec<MemoryConceptInput>,
    proposed_tags: Vec<MemoryTagInput>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema, lutum::Toolset)]
pub enum CompactionTools {
    GetMemories(GetMemoriesArgs),
    MergeMemories(MergeMemoriesArgs),
}

enum MergeDecision {
    Merged(MergeMemoriesOutput),
    Rejected(String),
}

pub struct MemoryCompactionModule {
    owner: nuillu_types::ModuleId,
    interoception_updates: InteroceptiveUpdatedInbox,
    blackboard: BlackboardReader,
    compactor: MemoryCompactor,
    llm: LlmAccess,
    audit_llm: FixedTierLlmAccess,
    system_prompt: std::sync::OnceLock<String>,
}

impl MemoryCompactionModule {
    pub fn new(
        interoception_updates: InteroceptiveUpdatedInbox,
        blackboard: BlackboardReader,
        compactor: MemoryCompactor,
        llm: LlmAccess,
        audit_llm: FixedTierLlmAccess,
    ) -> Self {
        Self {
            owner: nuillu_types::ModuleId::new(<Self as Module>::id())
                .expect("memory-compaction id is valid"),
            interoception_updates,
            blackboard,
            compactor,
            llm,
            audit_llm,
            system_prompt: std::sync::OnceLock::new(),
        }
    }

    fn system_prompt(&self, cx: &nuillu_module::ActivateCx<'_>) -> &str {
        self.system_prompt.get_or_init(|| {
            nuillu_module::format_system_seed(
                nuillu_module::format_system_prompt(
                    SYSTEM_PROMPT,
                    cx.peer_contexts(),
                    &self.owner,
                    cx.core_policies(),
                ),
                false,
                cx.identity_memories(),
                cx.now(),
            )
        })
    }

    #[tracing::instrument(skip_all, err(Debug, level = "warn"))]
    async fn activate(&self, cx: &nuillu_module::ActivateCx<'_>) -> Result<()> {
        let memory_metadata = self
            .blackboard
            .read(|bb| short_term_compaction_metadata(bb.memory_metadata()))
            .await;
        if memory_metadata.len() < MIN_MERGE_SOURCES {
            return Ok(());
        }
        let eligible_indexes = memory_metadata
            .iter()
            .map(|item| MemoryIndex::new(item.index.clone()))
            .collect::<HashSet<_>>();

        let mut input = ModelInput::new()
            .system(self.system_prompt(cx))
            .user(format_compaction_context(&memory_metadata));

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
                    let mut rejected = false;
                    nuillu_module::emit_trace_tool_calls(&round.tool_calls);
                    for call in round.tool_calls.iter().cloned() {
                        if rejected {
                            results.push(rejected_tool_result(
                                call,
                                "skipped because an earlier compaction tool call was rejected",
                            )?);
                            continue;
                        }
                        match call {
                            CompactionToolsCall::GetMemories(call) => {
                                let output = self
                                    .get_memories(call.input.clone(), &eligible_indexes)
                                    .await
                                    .context("run get_memories tool")?;
                                let result = call
                                    .complete(output)
                                    .context("complete get_memories tool call")?;
                                results.push(result);
                            }
                            CompactionToolsCall::MergeMemories(call) => {
                                match self
                                    .merge_memories(call.input.clone(), &eligible_indexes)
                                    .await
                                    .context("run merge_memories tool")?
                                {
                                    MergeDecision::Merged(output) => {
                                        let result = call
                                            .complete(output)
                                            .context("complete merge_memories tool call")?;
                                        results.push(result);
                                    }
                                    MergeDecision::Rejected(reason) => {
                                        results.push(rejected_tool_result(
                                            CompactionToolsCall::MergeMemories(call),
                                            reason,
                                        )?);
                                        rejected = true;
                                    }
                                }
                            }
                        }
                    }
                    round
                        .commit_into(&mut input, results)
                        .context("commit memory-compaction tool round")?;
                    if rejected {
                        return Ok(());
                    }
                }
            }
        }
        Ok(())
    }

    async fn get_memories(
        &self,
        args: GetMemoriesArgs,
        eligible_indexes: &HashSet<MemoryIndex>,
    ) -> Result<GetMemoriesOutput> {
        let indexes = args
            .indexes
            .into_iter()
            .map(MemoryIndex::new)
            .filter(|index| eligible_indexes.contains(index))
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

    async fn merge_memories(
        &self,
        args: MergeMemoriesArgs,
        eligible_indexes: &HashSet<MemoryIndex>,
    ) -> Result<MergeDecision> {
        let sources = args
            .source_indexes
            .iter()
            .cloned()
            .map(MemoryIndex::new)
            .collect::<Vec<_>>();
        let source_count = sources.len();
        if let Some(reason) = validate_merge_sources(&sources, eligible_indexes) {
            return Ok(MergeDecision::Rejected(reason));
        }
        let source_records = self
            .compactor
            .get_many(&sources)
            .await
            .context("get live source memories before compaction")?;
        if source_records.len() != sources.len() {
            return Ok(MergeDecision::Rejected(
                "merge_memories rejected: every source must still be live before compaction"
                    .to_owned(),
            ));
        }
        let live_indexes = source_records
            .iter()
            .map(|record| record.index.clone())
            .collect::<HashSet<_>>();
        if sources.iter().any(|source| !live_indexes.contains(source)) {
            return Ok(MergeDecision::Rejected(
                "merge_memories rejected: at least one source is missing from live retrieval"
                    .to_owned(),
            ));
        }
        let audit = self.audit_merge(&args, &source_records).await;
        match audit {
            Ok(decision) if decision.approved => {}
            Ok(decision) => {
                let reason = if decision.reason.trim().is_empty() {
                    "audit rejected the compaction proposal without a reason".to_owned()
                } else {
                    decision.reason
                };
                return Ok(MergeDecision::Rejected(format!(
                    "merge_memories rejected by pre-commit audit: {reason}"
                )));
            }
            Err(error) => {
                return Ok(MergeDecision::Rejected(format!(
                    "merge_memories rejected because pre-commit audit failed: {error:#}"
                )));
            }
        }
        let live_after_audit = self
            .compactor
            .get_many(&sources)
            .await
            .context("recheck live source memories after compaction audit")?;
        let live_after_audit_indexes = live_after_audit
            .iter()
            .map(|record| record.index.clone())
            .collect::<HashSet<_>>();
        if live_after_audit.len() != sources.len()
            || sources
                .iter()
                .any(|source| !live_after_audit_indexes.contains(source))
        {
            return Ok(MergeDecision::Rejected(
                "merge_memories rejected: every source must still be live after audit and before writing"
                    .to_owned(),
            ));
        }

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
        Ok(MergeDecision::Merged(MergeMemoriesOutput {
            merged_index: index.to_string(),
            merged_sources: source_count,
        }))
    }

    async fn audit_merge(
        &self,
        args: &MergeMemoriesArgs,
        source_records: &[MemoryRecord],
    ) -> Result<CompactionAuditOutput> {
        let audit_input = CompactionAuditInput {
            destructive_warning: "If approved, every source memory leaves live retrieval and future compaction candidates after the replacement summary is written.",
            source_memories: source_records
                .iter()
                .cloned()
                .map(memory_record_to_view)
                .collect(),
            proposed_summary: args.merged_content.clone(),
            proposed_concepts: args.concepts.clone(),
            proposed_tags: args.tags.clone(),
        };
        let input = ModelInput::new().system(AUDIT_PROMPT).user(
            serde_json::to_string_pretty(&audit_input)
                .expect("compaction audit input serialization should not fail"),
        );
        let lutum = self.audit_llm.lutum().await;
        let result = lutum
            .structured_turn::<CompactionAuditOutput>(input)
            .collect()
            .await
            .context("compaction audit structured turn failed")?;
        match result.semantic {
            StructuredTurnOutcome::Structured(decision) => Ok(decision),
            StructuredTurnOutcome::Refusal(reason) => Ok(CompactionAuditOutput {
                approved: false,
                reason: format!("audit refused: {reason}"),
            }),
        }
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

fn short_term_compaction_metadata(
    metadata: &HashMap<MemoryIndex, MemoryMetadata>,
) -> Vec<MemoryMetadataContext> {
    let mut metadata = metadata
        .values()
        .filter(|item| item.rank == MemoryRank::ShortTerm)
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
}

fn validate_merge_sources(
    sources: &[MemoryIndex],
    eligible_indexes: &HashSet<MemoryIndex>,
) -> Option<String> {
    if sources.len() < MIN_MERGE_SOURCES {
        return Some(format!(
            "merge_memories rejected: at least {MIN_MERGE_SOURCES} source memories are required"
        ));
    }
    if sources.len() > MAX_MERGE_SOURCES {
        return Some(format!(
            "merge_memories rejected: at most {MAX_MERGE_SOURCES} source memories may be compacted at once"
        ));
    }
    let unique_sources = sources.iter().collect::<HashSet<_>>();
    if unique_sources.len() != sources.len() {
        return Some("merge_memories rejected: source indexes must be unique".to_owned());
    }
    if let Some(source) = sources
        .iter()
        .find(|source| !eligible_indexes.contains(*source))
    {
        return Some(format!(
            "merge_memories rejected: source `{}` is not an eligible live ShortTerm compaction candidate",
            source.as_str()
        ));
    }
    None
}

fn rejected_tool_result(
    call: CompactionToolsCall,
    reason: impl Into<String>,
) -> Result<ToolResult> {
    RejectedToolCall::from_call(RejectedToolSource::Hook, call, reason)
        .into_tool_result()
        .context("create rejected compaction tool result")
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

    use std::cell::{Cell, RefCell};
    use std::rc::Rc;
    use std::sync::Arc;

    use chrono::{DateTime, TimeZone, Utc};
    use lutum::{
        FinishReason, Lutum, MockLlmAdapter, MockStructuredScenario, MockTextScenario,
        RawStructuredTurnEvent, RawTextTurnEvent, SharedPoolBudgetManager, SharedPoolBudgetOptions,
        Usage,
    };
    use nuillu_blackboard::{
        Blackboard, BlackboardCommand, Bpm, MemoryMetaPatch, MemoryMetadata, ModulePolicy,
        linear_ratio_fn,
    };
    use nuillu_module::ports::{NoopCognitionLogRepository, PortError, SystemClock};
    use nuillu_module::{
        ActivateCx, CapabilityProviderPorts, CapabilityProviders, InteroceptiveUpdated,
        LlmConcurrencyLimiter, LutumTiers, ModuleRegistry, SessionCompactionPolicy,
        SessionCompactionRuntime,
    };
    use nuillu_types::{MemoryContent, ModelTier, ReplicaCapRange};

    use crate::MemoryCapabilities;
    use crate::store::{
        IndexedMemory, LinkedMemoryQuery, LinkedMemoryRecord, MemoryLink, MemoryLinkRelation,
        MemoryQuery, MemoryStore, NewMemory, NewMemoryLink,
    };

    #[derive(Debug, Default)]
    struct RecordingMemoryStore {
        records: RefCell<HashMap<MemoryIndex, MemoryRecord>>,
        compactions: RefCell<Vec<(MemoryIndex, Vec<MemoryIndex>)>>,
        next_index: Cell<u64>,
    }

    #[async_trait(?Send)]
    impl MemoryStore for RecordingMemoryStore {
        async fn insert(
            &self,
            mem: NewMemory,
            stored_at: DateTime<Utc>,
        ) -> Result<MemoryRecord, PortError> {
            let id = self.next_index.get();
            self.next_index.set(id + 1);
            let record = MemoryRecord {
                index: MemoryIndex::new(format!("merged-{id}")),
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
                .insert(record.index.clone(), record.clone());
            Ok(record)
        }

        async fn put(&self, mem: IndexedMemory) -> Result<MemoryRecord, PortError> {
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
                .insert(record.index.clone(), record.clone());
            Ok(record)
        }

        async fn compact(
            &self,
            mem: NewMemory,
            sources: &[MemoryIndex],
            stored_at: DateTime<Utc>,
        ) -> Result<MemoryRecord, PortError> {
            let record = self.insert(mem, stored_at).await?;
            let mut records = self.records.borrow_mut();
            for source in sources {
                records.remove(source);
            }
            self.compactions
                .borrow_mut()
                .push((record.index.clone(), sources.to_vec()));
            Ok(record)
        }

        async fn put_compacted(
            &self,
            mem: IndexedMemory,
            sources: &[MemoryIndex],
        ) -> Result<MemoryRecord, PortError> {
            let record = self.put(mem).await?;
            let mut records = self.records.borrow_mut();
            for source in sources {
                records.remove(source);
            }
            self.compactions
                .borrow_mut()
                .push((record.index.clone(), sources.to_vec()));
            Ok(record)
        }

        async fn get(&self, index: &MemoryIndex) -> Result<Option<MemoryRecord>, PortError> {
            Ok(self.records.borrow().get(index).cloned())
        }

        async fn list_by_rank(&self, rank: MemoryRank) -> Result<Vec<MemoryRecord>, PortError> {
            Ok(self
                .records
                .borrow()
                .values()
                .filter(|record| record.rank == rank)
                .cloned()
                .collect())
        }

        async fn search(&self, _q: &MemoryQuery) -> Result<Vec<MemoryRecord>, PortError> {
            Ok(Vec::new())
        }

        async fn linked(
            &self,
            _q: &LinkedMemoryQuery,
        ) -> Result<Vec<LinkedMemoryRecord>, PortError> {
            Ok(Vec::new())
        }

        async fn upsert_link(
            &self,
            link: NewMemoryLink,
            updated_at: DateTime<Utc>,
        ) -> Result<MemoryLink, PortError> {
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

        async fn delete(&self, index: &MemoryIndex) -> Result<(), PortError> {
            self.records.borrow_mut().remove(index);
            Ok(())
        }
    }

    fn test_policy() -> ModulePolicy {
        ModulePolicy::new(
            ReplicaCapRange::new(1, 1).unwrap(),
            Bpm::from_f64(60_000.0)..=Bpm::from_f64(60_000.0),
            linear_ratio_fn,
        )
    }

    fn usage(input_tokens: u64) -> Usage {
        Usage {
            input_tokens,
            ..Usage::zero()
        }
    }

    fn text_scenario(text: &str) -> MockTextScenario {
        MockTextScenario::events(vec![
            Ok(RawTextTurnEvent::Started {
                request_id: Some("memory-compaction-text".into()),
                model: "mock".into(),
            }),
            Ok(RawTextTurnEvent::TextDelta { delta: text.into() }),
            Ok(RawTextTurnEvent::Completed {
                request_id: Some("memory-compaction-text".into()),
                finish_reason: FinishReason::Stop,
                usage: usage(1),
            }),
        ])
    }

    fn merge_tool_scenario(source_indexes: Vec<&str>, merged_content: &str) -> MockTextScenario {
        MockTextScenario::events(vec![
            Ok(RawTextTurnEvent::Started {
                request_id: Some("memory-compaction-merge".into()),
                model: "mock".into(),
            }),
            Ok(RawTextTurnEvent::ToolCallChunk {
                id: "call-merge".into(),
                name: "merge_memories".into(),
                arguments_json_delta: serde_json::json!({
                    "source_indexes": source_indexes,
                    "merged_content": merged_content,
                    "concepts": ["compacted subject"],
                    "tags": ["compaction"]
                })
                .to_string(),
            }),
            Ok(RawTextTurnEvent::Completed {
                request_id: Some("memory-compaction-merge".into()),
                finish_reason: FinishReason::ToolCall,
                usage: usage(1),
            }),
        ])
    }

    fn audit_scenario(approved: bool, reason: &str) -> MockStructuredScenario {
        let json = serde_json::json!({
            "approved": approved,
            "reason": reason,
        })
        .to_string();
        MockStructuredScenario::events(vec![
            Ok(RawStructuredTurnEvent::Started {
                request_id: Some("memory-compaction-audit".into()),
                model: "mock".into(),
            }),
            Ok(RawStructuredTurnEvent::StructuredOutputChunk { json_delta: json }),
            Ok(RawStructuredTurnEvent::Completed {
                request_id: Some("memory-compaction-audit".into()),
                finish_reason: FinishReason::Stop,
                usage: usage(1),
            }),
        ])
    }

    fn memory_record(
        index: &str,
        content: &str,
        rank: MemoryRank,
        now: DateTime<Utc>,
    ) -> MemoryRecord {
        MemoryRecord {
            index: MemoryIndex::new(index),
            content: MemoryContent::new(content),
            rank,
            occurred_at: Some(now),
            stored_at: now,
            kind: crate::store::MemoryKind::Statement,
            concepts: Vec::new(),
            tags: Vec::new(),
            affect_arousal: 0.0,
            valence: 0.0,
            emotion: String::new(),
        }
    }

    async fn seed_memory(
        blackboard: &Blackboard,
        store: &RecordingMemoryStore,
        index: &str,
        content: &str,
        rank: MemoryRank,
        now: DateTime<Utc>,
    ) -> MemoryIndex {
        let record = memory_record(index, content, rank, now);
        let index = record.index.clone();
        store.records.borrow_mut().insert(index.clone(), record);
        blackboard
            .apply(BlackboardCommand::UpsertMemoryMetadata {
                index: index.clone(),
                rank_if_new: rank,
                occurred_at_if_new: Some(now),
                decay_if_new_secs: SHORT_TERM_MEMORY_DECAY_SECS,
                now,
                patch: MemoryMetaPatch::default(),
            })
            .await;
        index
    }

    fn activate_cx(lutum: &Lutum, now: DateTime<Utc>) -> ActivateCx<'static> {
        ActivateCx::new(
            &[],
            &[],
            &[],
            &[],
            SessionCompactionRuntime::new(
                lutum.clone(),
                LlmConcurrencyLimiter::new(None),
                ModelTier::Default,
                SessionCompactionPolicy::default(),
            ),
            now,
        )
    }

    async fn run_compaction(
        adapter: MockLlmAdapter,
        blackboard: Blackboard,
        store: Rc<RecordingMemoryStore>,
        now: DateTime<Utc>,
    ) {
        let budget = SharedPoolBudgetManager::new(SharedPoolBudgetOptions::default());
        let lutum = Lutum::new(Arc::new(adapter), budget);
        let clock = Rc::new(SystemClock);
        let caps = CapabilityProviders::new(CapabilityProviderPorts {
            blackboard: blackboard.clone(),
            cognition_log_port: Rc::new(NoopCognitionLogRepository),
            clock: clock.clone(),
            tiers: LutumTiers::from_shared_lutum(lutum.clone()),
        });
        let memory_caps =
            MemoryCapabilities::new(blackboard.clone(), clock, store.clone(), Vec::new());
        let modules = ModuleRegistry::new()
            .register(test_policy(), move |caps| {
                let memory_caps = memory_caps.clone();
                async move {
                    Ok(MemoryCompactionModule::new(
                        caps.interoception_updated_inbox(),
                        caps.blackboard_reader(),
                        memory_caps.compactor(),
                        caps.llm_access(),
                        caps.default_tier_llm_access(),
                    ))
                }
            })
            .unwrap()
            .build(&caps)
            .await
            .unwrap();
        let (_, mut modules) = modules.into_parts();
        let mut module = modules.remove(0);
        caps.internal_harness_io()
            .interoception_updated_mailbox()
            .publish(InteroceptiveUpdated)
            .await
            .unwrap();
        let batch = module.next_batch().await.unwrap();
        module
            .activate(&activate_cx(&lutum, now), &batch)
            .await
            .unwrap();
    }

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

    #[test]
    fn compaction_metadata_includes_only_short_term_candidates() {
        let now = Utc.with_ymd_and_hms(2026, 6, 7, 0, 0, 0).unwrap();
        let mut metadata = HashMap::new();
        for (index, rank) in [
            ("identity-seed", MemoryRank::Identity),
            ("permanent-seed", MemoryRank::Permanent),
            ("long-term", MemoryRank::LongTerm),
            ("mid-term", MemoryRank::MidTerm),
            ("short-term-b", MemoryRank::ShortTerm),
            ("short-term-a", MemoryRank::ShortTerm),
        ] {
            metadata.insert(
                MemoryIndex::new(index),
                MemoryMetadata::new_at(MemoryIndex::new(index), rank, Some(now), 0, now),
            );
        }

        let context = short_term_compaction_metadata(&metadata);

        assert_eq!(
            context
                .into_iter()
                .map(|item| item.index)
                .collect::<Vec<_>>(),
            vec!["short-term-a", "short-term-b"]
        );
    }

    #[test]
    fn merge_source_validation_enforces_bounds_uniqueness_and_eligibility() {
        let first = MemoryIndex::new("first");
        let second = MemoryIndex::new("second");
        let third = MemoryIndex::new("third");
        let fourth = MemoryIndex::new("fourth");
        let eligible = [first.clone(), second.clone(), third.clone()]
            .into_iter()
            .collect::<HashSet<_>>();

        assert_eq!(
            validate_merge_sources(&[first.clone(), second.clone()], &eligible),
            None
        );
        assert!(
            validate_merge_sources(std::slice::from_ref(&first), &eligible)
                .unwrap()
                .contains("at least")
        );
        assert!(
            validate_merge_sources(
                &[first.clone(), second.clone(), third.clone(), fourth.clone()],
                &eligible,
            )
            .unwrap()
            .contains("at most")
        );
        assert!(
            validate_merge_sources(&[first.clone(), first.clone()], &eligible)
                .unwrap()
                .contains("unique")
        );
        assert!(
            validate_merge_sources(&[first, fourth], &eligible)
                .unwrap()
                .contains("eligible live ShortTerm")
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn audit_approval_writes_summary_and_hides_sources() {
        let now = Utc.with_ymd_and_hms(2026, 6, 7, 0, 0, 0).unwrap();
        let blackboard = Blackboard::default();
        let store = Rc::new(RecordingMemoryStore::default());
        let first = seed_memory(
            &blackboard,
            store.as_ref(),
            "short-a",
            "The red cup is on the left shelf.",
            MemoryRank::ShortTerm,
            now,
        )
        .await;
        let second = seed_memory(
            &blackboard,
            store.as_ref(),
            "short-b",
            "The red cup has a chipped handle.",
            MemoryRank::ShortTerm,
            now,
        )
        .await;
        let adapter = MockLlmAdapter::new()
            .with_text_scenario(merge_tool_scenario(
                vec!["short-a", "short-b"],
                "The red cup is on the left shelf and has a chipped handle.",
            ))
            .with_structured_scenario(audit_scenario(true, "preserves both concrete facts"))
            .with_text_scenario(text_scenario(""));

        run_compaction(adapter, blackboard.clone(), store.clone(), now).await;

        let compactions = store.compactions.borrow().clone();
        assert_eq!(
            compactions,
            vec![(
                MemoryIndex::new("merged-0"),
                vec![first.clone(), second.clone()]
            )]
        );
        assert!(store.get(&first).await.unwrap().is_none());
        assert!(store.get(&second).await.unwrap().is_none());
        let merged = store
            .get(&MemoryIndex::new("merged-0"))
            .await
            .unwrap()
            .unwrap();
        assert_eq!(
            merged.content.as_str(),
            "The red cup is on the left shelf and has a chipped handle."
        );
        blackboard
            .read(|bb| {
                let metadata = bb.memory_metadata();
                assert!(!metadata.contains_key(&first));
                assert!(!metadata.contains_key(&second));
                assert!(metadata.contains_key(&MemoryIndex::new("merged-0")));
            })
            .await;
    }

    #[tokio::test(flavor = "current_thread")]
    async fn audit_rejection_does_not_write_and_stops_activation() {
        let now = Utc.with_ymd_and_hms(2026, 6, 7, 0, 0, 0).unwrap();
        let blackboard = Blackboard::default();
        let store = Rc::new(RecordingMemoryStore::default());
        let first = seed_memory(
            &blackboard,
            store.as_ref(),
            "short-a",
            "The red cup is on the left shelf.",
            MemoryRank::ShortTerm,
            now,
        )
        .await;
        let second = seed_memory(
            &blackboard,
            store.as_ref(),
            "short-b",
            "The blue notebook is under the desk.",
            MemoryRank::ShortTerm,
            now,
        )
        .await;
        let adapter = MockLlmAdapter::new()
            .with_text_scenario(merge_tool_scenario(
                vec!["short-a", "short-b"],
                "There are objects in the room.",
            ))
            .with_structured_scenario(audit_scenario(
                false,
                "metadata-only summary loses concrete source facts",
            ));

        run_compaction(adapter, blackboard.clone(), store.clone(), now).await;

        assert!(store.compactions.borrow().is_empty());
        assert!(store.get(&first).await.unwrap().is_some());
        assert!(store.get(&second).await.unwrap().is_some());
        blackboard
            .read(|bb| {
                let metadata = bb.memory_metadata();
                assert!(metadata.contains_key(&first));
                assert!(metadata.contains_key(&second));
                assert_eq!(metadata.len(), 2);
            })
            .await;
    }

    #[tokio::test(flavor = "current_thread")]
    async fn missing_live_source_rejects_without_writing_or_auditing() {
        let now = Utc.with_ymd_and_hms(2026, 6, 7, 0, 0, 0).unwrap();
        let blackboard = Blackboard::default();
        let store = Rc::new(RecordingMemoryStore::default());
        let first = seed_memory(
            &blackboard,
            store.as_ref(),
            "short-a",
            "The red cup is on the left shelf.",
            MemoryRank::ShortTerm,
            now,
        )
        .await;
        let missing = MemoryIndex::new("short-b");
        blackboard
            .apply(BlackboardCommand::UpsertMemoryMetadata {
                index: missing.clone(),
                rank_if_new: MemoryRank::ShortTerm,
                occurred_at_if_new: Some(now),
                decay_if_new_secs: SHORT_TERM_MEMORY_DECAY_SECS,
                now,
                patch: MemoryMetaPatch::default(),
            })
            .await;
        let adapter = MockLlmAdapter::new().with_text_scenario(merge_tool_scenario(
            vec!["short-a", "short-b"],
            "The red cup has details.",
        ));

        run_compaction(adapter, blackboard.clone(), store.clone(), now).await;

        assert!(store.compactions.borrow().is_empty());
        assert!(store.get(&first).await.unwrap().is_some());
        blackboard
            .read(|bb| {
                let metadata = bb.memory_metadata();
                assert!(metadata.contains_key(&first));
                assert!(metadata.contains_key(&missing));
                assert_eq!(metadata.len(), 2);
            })
            .await;
    }

    #[tokio::test(flavor = "current_thread")]
    async fn identity_memory_is_never_candidate_and_cannot_be_merged_when_requested() {
        let now = Utc.with_ymd_and_hms(2026, 6, 7, 0, 0, 0).unwrap();
        let blackboard = Blackboard::default();
        let store = Rc::new(RecordingMemoryStore::default());
        let identity = seed_memory(
            &blackboard,
            store.as_ref(),
            "nui-identity-test",
            "Nui's identity anchor must never be compacted away.",
            MemoryRank::Identity,
            now,
        )
        .await;
        let first = seed_memory(
            &blackboard,
            store.as_ref(),
            "short-a",
            "The red cup is on the left shelf.",
            MemoryRank::ShortTerm,
            now,
        )
        .await;
        let second = seed_memory(
            &blackboard,
            store.as_ref(),
            "short-b",
            "The red cup has a chipped handle.",
            MemoryRank::ShortTerm,
            now,
        )
        .await;
        let adapter = MockLlmAdapter::new().with_text_scenario(merge_tool_scenario(
            vec!["nui-identity-test", "short-a"],
            "Nui remembers the red cup.",
        ));

        run_compaction(adapter, blackboard.clone(), store.clone(), now).await;

        assert!(store.compactions.borrow().is_empty());
        assert!(store.get(&identity).await.unwrap().is_some());
        assert!(store.get(&first).await.unwrap().is_some());
        assert!(store.get(&second).await.unwrap().is_some());
        blackboard
            .read(|bb| {
                let metadata = bb.memory_metadata();
                assert_eq!(metadata.get(&identity).unwrap().rank, MemoryRank::Identity);
                assert_eq!(metadata.get(&first).unwrap().rank, MemoryRank::ShortTerm);
                assert_eq!(metadata.get(&second).unwrap().rank, MemoryRank::ShortTerm);
            })
            .await;
    }
}
