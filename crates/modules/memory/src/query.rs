use std::collections::HashSet;

use anyhow::{Context, Result};
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use lutum::{Session, TextStepOutcomeWithTools, ToolResult};
use nuillu_module::{
    BlackboardReader, CognitionLogUpdatedInbox, LlmAccess, LlmContextWindow, Module,
    SessionAutoCompaction, SessionCompactionConfig, SessionCompactionProtectedPrefix, TypedMemo,
    ensure_persistent_session_seeded, format_memory_trace_inventory, memory_rank_counts,
    push_formatted_memo_log_batch, render_memory_for_llm,
};
use nuillu_types::{MemoryIndex, MemoryRank};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::store::{
    LinkedMemoryQuery, LinkedMemoryRecord, MemoryConcept, MemoryContentReader, MemoryKind,
    MemoryLink, MemoryLinkDirection, MemoryRetriever, MemoryTag, MemoryUsageTarget,
};

const SYSTEM_PROMPT: &str = r#"You are the query-memory module.
Retrieve memory evidence. Memory is not a fact table and retrieval must not decide current truth.
Use search_memory for ordinary flat memory search. Search hits report linked_neighbor_count;
when that count is greater than zero and linked context may help the question, call
fetch_linked_memories for that hit index before broadcasting results. Pass only the seed memory indexes;
the runtime applies default link direction, relation filter, and limit. Linked lookup is explicit; ordinary
search results are flat and do not include hidden bundles.
If the question contains allocation guidance or a speech evidence request, search for the concrete
requested facts, proper nouns, species/body/peer/world terms, route rules, and the needed_fact
phrases. Do not search for generic phrases such as "useful memory context" when a concrete guidance
question is available.
You may call search_memory multiple times in the same turn when the input contains multiple
distinct questions or evidence requests. Prefer multiple targeted searches in one turn over broad
generic searches or later follow-up turns.
If the requested facts are already covered by prior query-memory broadcasts, previous tool results,
or the cognition log, finish without calling tools; no new broadcast is needed for a no-op turn.
When retrieved evidence is useful, call broadcast_search_results exactly once. Put flat search hit
indexes in hit_indexes and fetch_linked_memories results in linked_hit_indexes. Search hits are
candidates, not automatic broadcast content. Always copy the exact literal index string from the tool
result; never use list positions or ordinal numbers as memory indexes.
You may summarize conflict or link structure only through retrieved tool evidence in the broadcast. Do
not produce user-facing answers, decide final truth, explain results from outside tool output, or
use a final answer as a data channel."#;

const COMPACTED_QUERY_MEMORY_SESSION_PREFIX: &str = "Compacted query-memory session history:";
const MEMO_CONTEXT_WINDOW: LlmContextWindow = LlmContextWindow::new(8, 1_200, 4_800);
const TOOL_TURN_MAX_OUTPUT_TOKENS: u32 = 768;
const SESSION_COMPACTION_FOCUS: &str = r#"Preserve broadcast facts, query requests, memory search
arguments, useful memory hits, rejected broad searches, and allocation/cognition context that future
retrieval should remember."#;
const RETRIEVAL_TURN_FINAL_REMINDER: &str = concat!(
    "Output instruction: call search_memory, fetch_linked_memories, or broadcast_search_results ",
    "only when retrieval or broadcast is needed. If no memory action is needed, finish without ",
    "assistant text. Do not write a plain answer.",
);
const TOOL_RESULT_CONTINUATION_PROMPT: &str = r#"Continue memory retrieval from the tool results above.
If a search hit may answer the request and has linked_neighbor_count greater than zero, call
fetch_linked_memories for the seed hit before broadcasting results.
If retrieved evidence is useful, call broadcast_search_results with the selected flat hit_indexes and
linked_hit_indexes. If the tool results do not contain useful evidence and no targeted search
remains, finish without assistant text. Use exact literal index strings from tool results, never
list positions."#;
const FINALIZE_RETRIEVAL_PROMPT: &str = r#"Finalization turn. Use broadcast_search_results or dispose_search_results.
Do not write prose, reasoning, markdown, or a plain answer.
Search and linked-memory fetch are closed. Use only existing tool results.
The search_memory and fetch_linked_memories results already in this session are fresh pending
retrieval evidence for this activation. Call broadcast_search_results when any pending retrieved
evidence should be published as module output. Call dispose_search_results only when no pending
retrieved evidence is useful enough to broadcast. Prior assistant prose is ignored by the runtime and
does not count as a successful broadcast. Never call dispose_search_results just because assistant
prose already summarized retrieved evidence. Use exact literal index strings from tool results,
never list positions."#;

pub fn query_session_auto_compaction() -> SessionAutoCompaction {
    SessionAutoCompaction::new(
        SessionCompactionConfig::default(),
        SessionCompactionProtectedPrefix::LeadingSystemAndIdentitySeed,
        COMPACTED_QUERY_MEMORY_SESSION_PREFIX,
        SESSION_COMPACTION_FOCUS,
    )
}

fn format_memory_context(rank_counts: &nuillu_module::MemoryRankCounts) -> String {
    let mut sections = vec!["Memory retrieval context:".to_owned()];
    if let Some(section) = format_memory_trace_inventory(rank_counts) {
        sections.push(section);
    }
    sections.join("\n\n")
}

#[lutum::tool_input(name = "search_memory", output = SearchMemoryOutput)]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct SearchMemoryArgs {
    pub query: String,
    pub limit: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct SearchMemoryOutput {
    pub hits: Vec<QueryMemoryHit>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct QueryMemoryHit {
    pub index: MemoryIndex,
    pub content: String,
    pub rank: MemoryRank,
    pub occurred_at: Option<DateTime<Utc>>,
    pub stored_at: DateTime<Utc>,
    pub kind: MemoryKind,
    pub concepts: Vec<MemoryConcept>,
    pub tags: Vec<MemoryTag>,
    pub affect_arousal: f32,
    pub valence: f32,
    pub emotion: String,
    pub linked_neighbor_count: usize,
}

const DEFAULT_LINKED_MEMORY_LIMIT: usize = 16;

#[lutum::tool_input(name = "fetch_linked_memories", output = FetchLinkedMemoriesOutput)]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct FetchLinkedMemoriesArgs {
    pub memory_indexes: Vec<MemoryIndex>,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct FetchLinkedMemoriesOutput {
    pub hits: Vec<QueryMemoryLinkedHit>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct QueryMemoryLinkedHit {
    pub index: MemoryIndex,
    pub content: String,
    pub rank: MemoryRank,
    pub occurred_at: Option<DateTime<Utc>>,
    pub stored_at: DateTime<Utc>,
    pub kind: MemoryKind,
    pub concepts: Vec<MemoryConcept>,
    pub tags: Vec<MemoryTag>,
    pub affect_arousal: f32,
    pub valence: f32,
    pub emotion: String,
    pub link: MemoryLink,
}

#[lutum::tool_input(name = "broadcast_search_results", output = BroadcastSearchResultsOutput)]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct BroadcastSearchResultsArgs {
    #[serde(default)]
    pub hit_indexes: Vec<MemoryIndex>,
    #[serde(default)]
    pub linked_hit_indexes: Vec<MemoryIndex>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct BroadcastSearchResultsOutput {
    pub broadcasted: bool,
    pub selected_hits: usize,
    pub selected_linked_hits: usize,
    pub used_indexes: Vec<MemoryIndex>,
    pub message: String,
}

#[lutum::tool_input(name = "dispose_search_results", output = DisposeSearchResultsOutput)]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct DisposeSearchResultsArgs {
    #[serde(default)]
    pub reason: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct DisposeSearchResultsOutput {
    pub disposed: bool,
    pub message: String,
}

#[derive(Clone, Debug, PartialEq)]
pub struct QueryMemoryMemo {
    pub requests: Vec<String>,
    pub searches: Vec<QueryMemoryMemoSearch>,
    pub hits: Vec<QueryMemoryMemoHit>,
    pub linked_hits: Vec<QueryMemoryMemoLinkedHit>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct QueryMemoryMemoSearch {
    pub query: String,
    pub limit: usize,
    pub hit_indices: Vec<MemoryIndex>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct QueryMemoryMemoHit {
    pub index: MemoryIndex,
    pub rank: MemoryRank,
    pub occurred_at: Option<DateTime<Utc>>,
    pub stored_at: DateTime<Utc>,
    pub kind: MemoryKind,
    pub affect_arousal: f32,
    pub valence: f32,
    pub emotion: String,
}

#[derive(Clone, Debug, PartialEq)]
pub struct QueryMemoryMemoLinkedHit {
    pub index: MemoryIndex,
    pub rank: MemoryRank,
    pub occurred_at: Option<DateTime<Utc>>,
    pub stored_at: DateTime<Utc>,
    pub kind: MemoryKind,
    pub affect_arousal: f32,
    pub valence: f32,
    pub emotion: String,
    pub link: MemoryLink,
}

#[derive(Clone, Debug, Default)]
struct QueryMemoryRetrieval {
    searches: Vec<QueryMemoryMemoSearch>,
    hits: Vec<QueryMemoryHit>,
    linked_hits: Vec<QueryMemoryLinkedHit>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema, lutum::Toolset)]
pub enum QueryMemoryTools {
    SearchMemory(SearchMemoryArgs),
    FetchLinkedMemories(FetchLinkedMemoriesArgs),
    BroadcastSearchResults(BroadcastSearchResultsArgs),
    DisposeSearchResults(DisposeSearchResultsArgs),
}

pub struct QueryMemoryModule {
    owner: nuillu_types::ModuleId,
    cognition_updates: CognitionLogUpdatedInbox,
    blackboard: BlackboardReader,
    memory: MemoryRetriever,
    linked_memory: MemoryContentReader,
    memo: TypedMemo<QueryMemoryMemo>,
    llm: LlmAccess,
    session: Session,
    system_prompt: std::sync::OnceLock<String>,
    pending_retrieval: QueryMemoryRetrieval,
}

impl QueryMemoryModule {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        cognition_updates: CognitionLogUpdatedInbox,
        blackboard: BlackboardReader,
        memory: MemoryRetriever,
        linked_memory: MemoryContentReader,
        memo: TypedMemo<QueryMemoryMemo>,
        llm: LlmAccess,
        session: Session,
    ) -> Self {
        Self {
            owner: nuillu_types::ModuleId::new(<Self as Module>::id())
                .expect("query-memory id is valid"),
            cognition_updates,
            blackboard,
            memory,
            linked_memory,
            memo,
            llm,
            session,
            system_prompt: std::sync::OnceLock::new(),
            pending_retrieval: QueryMemoryRetrieval::default(),
        }
    }

    fn system_prompt(&self, cx: &nuillu_module::ActivateCx<'_>) -> &str {
        self.system_prompt.get_or_init(|| {
            nuillu_module::format_faculty_system_prompt(
                SYSTEM_PROMPT,
                cx.peer_contexts(),
                &self.owner,
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
    async fn activate_cognition_update(
        &mut self,
        cx: &nuillu_module::ActivateCx<'_>,
    ) -> Result<()> {
        let owner = self.owner.clone();
        let question = self
            .blackboard
            .read(move |bb| {
                let guidance = bb.allocation().for_module(&owner).guidance;
                if !guidance.trim().is_empty() {
                    return guidance;
                }
                let entries = bb.cognition_log().entries().to_vec();
                let latest = entries
                    .last()
                    .map(|entry| entry.text.as_str())
                    .unwrap_or("current cognition log");
                format!("Find stable memory that clarifies this cognition-log context: {latest}")
            })
            .await;
        let questions = [question];
        self.search_with_memory(cx, &questions).await
    }

    async fn broadcast_search_results(
        &self,
        requests: &[String],
        retrieval: &QueryMemoryRetrieval,
        args: BroadcastSearchResultsArgs,
        already_broadcasted: &mut bool,
    ) -> Result<BroadcastSearchResultsOutput> {
        if *already_broadcasted {
            return Ok(BroadcastSearchResultsOutput {
                broadcasted: false,
                selected_hits: 0,
                selected_linked_hits: 0,
                used_indexes: Vec::new(),
                message: "search results were already broadcast for this activation".to_owned(),
            });
        }

        let selected_hits = select_hits(&retrieval.hits, &args.hit_indexes);
        let selected_linked_hits =
            select_linked_hits(&retrieval.linked_hits, &args.linked_hit_indexes);
        let hits = self.fresh_hits(&selected_hits).await;
        let linked_hits = self.fresh_linked_hits(&selected_linked_hits).await;
        let content = render_memo(requests, &retrieval.searches, &hits, &linked_hits);
        if content.is_empty() {
            return Ok(BroadcastSearchResultsOutput {
                broadcasted: false,
                selected_hits: hits.len(),
                selected_linked_hits: linked_hits.len(),
                used_indexes: Vec::new(),
                message: "no fresh selected evidence was available to broadcast".to_owned(),
            });
        }
        *already_broadcasted = true;
        let payload = QueryMemoryMemo {
            requests: requests.to_vec(),
            searches: retrieval.searches.clone(),
            hits: hits
                .iter()
                .map(|hit| QueryMemoryMemoHit {
                    index: hit.index.clone(),
                    rank: hit.rank,
                    occurred_at: hit.occurred_at,
                    stored_at: hit.stored_at,
                    kind: hit.kind,
                    affect_arousal: hit.affect_arousal,
                    valence: hit.valence,
                    emotion: hit.emotion.clone(),
                })
                .collect(),
            linked_hits: linked_hits
                .iter()
                .map(|hit| QueryMemoryMemoLinkedHit {
                    index: hit.index.clone(),
                    rank: hit.rank,
                    occurred_at: hit.occurred_at,
                    stored_at: hit.stored_at,
                    kind: hit.kind,
                    affect_arousal: hit.affect_arousal,
                    valence: hit.valence,
                    emotion: hit.emotion.clone(),
                    link: hit.link.clone(),
                })
                .collect(),
        };
        self.memo.write_cognitive(payload, content).await;
        let use_targets = usage_targets(&hits, &linked_hits);
        self.memory.record_uses(&use_targets).await;
        let mut seen_used = HashSet::new();
        let used_indexes = use_targets
            .into_iter()
            .filter_map(|target| {
                seen_used
                    .insert(target.index.clone())
                    .then_some(target.index)
            })
            .collect();
        Ok(BroadcastSearchResultsOutput {
            broadcasted: true,
            selected_hits: hits.len(),
            selected_linked_hits: linked_hits.len(),
            used_indexes,
            message: "search results broadcast".to_owned(),
        })
    }

    async fn fresh_hits(&self, hits: &[QueryMemoryHit]) -> Vec<QueryMemoryHit> {
        let mut seen = self
            .memo
            .recent_logs()
            .await
            .iter()
            .flat_map(|record| record.data().hits.iter().map(|hit| hit.index.clone()))
            .collect::<HashSet<_>>();
        hits.iter()
            .filter(|hit| seen.insert(hit.index.clone()))
            .cloned()
            .collect()
    }

    async fn fresh_linked_hits(&self, hits: &[QueryMemoryLinkedHit]) -> Vec<QueryMemoryLinkedHit> {
        let mut seen = self
            .memo
            .recent_logs()
            .await
            .iter()
            .flat_map(|record| {
                record
                    .data()
                    .hits
                    .iter()
                    .map(|hit| hit.index.clone())
                    .chain(
                        record
                            .data()
                            .linked_hits
                            .iter()
                            .map(|hit| hit.index.clone()),
                    )
            })
            .collect::<HashSet<_>>();
        hits.iter()
            .filter(|hit| seen.insert(hit.index.clone()))
            .cloned()
            .collect()
    }

    async fn search_with_memory(
        &mut self,
        cx: &nuillu_module::ActivateCx<'_>,
        questions: &[String],
    ) -> Result<()> {
        self.ensure_session_seeded(cx);
        let unread_memo_logs = self.blackboard.unread_memo_logs().await;
        let prior_query_memory_searches = self.prior_query_memory_searches().await;
        let rank_counts = self
            .blackboard
            .read(|bb| memory_rank_counts(bb.memory_metadata()))
            .await;
        push_formatted_memo_log_batch(
            &mut self.session,
            &unread_memo_logs,
            cx.now(),
            MEMO_CONTEXT_WINDOW,
        );
        self.session.push_user(format_memory_questions(questions));
        if let Some(prior) = prior_query_memory_searches {
            self.session.push_ephemeral_user(prior);
        }
        self.session
            .push_ephemeral_system(format_memory_context(&rank_counts));

        let mut retrieval = self.pending_retrieval.clone();
        let mut already_broadcasted = false;
        let mut clear_pending_retrieval = false;
        for _ in 0..4 {
            let lutum = self.llm.lutum().await;
            self.session
                .push_ephemeral_user(RETRIEVAL_TURN_FINAL_REMINDER);
            let outcome = self
                .session
                .text_turn()
                .tools::<QueryMemoryTools>()
                .available_tools([
                    QueryMemoryToolsSelector::SearchMemory,
                    QueryMemoryToolsSelector::FetchLinkedMemories,
                    QueryMemoryToolsSelector::BroadcastSearchResults,
                ])
                .max_output_tokens(TOOL_TURN_MAX_OUTPUT_TOKENS)
                .collect_controlled_with(
                    &lutum,
                    nuillu_module::AbortOnAvailableToolNameInText::new(),
                )
                .await
                .context("query-memory text turn failed")?;

            match outcome {
                TextStepOutcomeWithTools::Finished(result) => {
                    cx.compact_and_save(&mut self.session, result.usage).await?;
                    if !clear_pending_retrieval {
                        clear_pending_retrieval = self
                            .finalize_search_results(
                                cx,
                                questions,
                                &retrieval,
                                &mut already_broadcasted,
                            )
                            .await?;
                    }
                    self.pending_retrieval = if clear_pending_retrieval {
                        QueryMemoryRetrieval::default()
                    } else {
                        retrieval
                    };
                    return Ok(());
                }
                TextStepOutcomeWithTools::FinishedNoOutput(result) => {
                    cx.compact_and_save(&mut self.session, result.usage).await?;
                    if !clear_pending_retrieval {
                        clear_pending_retrieval = self
                            .finalize_search_results(
                                cx,
                                questions,
                                &retrieval,
                                &mut already_broadcasted,
                            )
                            .await?;
                    }
                    self.pending_retrieval = if clear_pending_retrieval {
                        QueryMemoryRetrieval::default()
                    } else {
                        retrieval
                    };
                    return Ok(());
                }
                TextStepOutcomeWithTools::NeedsTools(round) => {
                    let usage = round.usage;
                    nuillu_module::emit_trace_tool_calls(&round.tool_calls);
                    if round.tool_calls.is_empty() {
                        cx.compact_and_save(&mut self.session, usage).await?;
                        if !clear_pending_retrieval {
                            clear_pending_retrieval = self
                                .finalize_search_results(
                                    cx,
                                    questions,
                                    &retrieval,
                                    &mut already_broadcasted,
                                )
                                .await?;
                        }
                        self.pending_retrieval = if clear_pending_retrieval {
                            QueryMemoryRetrieval::default()
                        } else {
                            retrieval
                        };
                        return Ok(());
                    }
                    let mut tool_results: Vec<ToolResult> = Vec::new();
                    let mut broadcasted_search_results = false;
                    for call in round.tool_calls.iter().cloned() {
                        match call {
                            QueryMemoryToolsCall::SearchMemory(call) => {
                                let input = call.input.clone();
                                let output = self
                                    .search_memory(input.clone(), cx.now())
                                    .await
                                    .context("run search_memory tool")?;
                                retrieval.searches.push(QueryMemoryMemoSearch {
                                    query: input.query,
                                    limit: input.limit.clamp(1, 16),
                                    hit_indices: output
                                        .hits
                                        .iter()
                                        .map(|hit| hit.index.clone())
                                        .collect(),
                                });
                                retrieval.hits.extend(output.hits.clone());
                                tool_results.push(
                                    call.complete(output)
                                        .context("complete search_memory tool call")?,
                                );
                            }
                            QueryMemoryToolsCall::FetchLinkedMemories(call) => {
                                let output = self
                                    .fetch_linked_memories(call.input.clone(), cx.now())
                                    .await
                                    .context("run fetch_linked_memories tool")?;
                                retrieval.linked_hits.extend(output.hits.clone());
                                tool_results.push(
                                    call.complete(output)
                                        .context("complete fetch_linked_memories tool call")?,
                                );
                            }
                            QueryMemoryToolsCall::BroadcastSearchResults(call) => {
                                let output = self
                                    .broadcast_search_results(
                                        questions,
                                        &retrieval,
                                        call.input.clone(),
                                        &mut already_broadcasted,
                                    )
                                    .await
                                    .context("run broadcast_search_results tool")?;
                                broadcasted_search_results |= output.broadcasted;
                                tool_results.push(
                                    call.complete(output)
                                        .context("complete broadcast_search_results tool call")?,
                                );
                            }
                            QueryMemoryToolsCall::DisposeSearchResults(_) => {
                                unreachable!("retrieval loop excludes dispose_search_results")
                            }
                        }
                    }
                    round
                        .commit(&mut self.session, tool_results)
                        .context("commit query-memory tool round")?;
                    cx.compact_and_save(&mut self.session, usage).await?;
                    if broadcasted_search_results {
                        self.pending_retrieval = QueryMemoryRetrieval::default();
                        return Ok(());
                    }
                    self.pending_retrieval = retrieval.clone();
                    self.session
                        .push_ephemeral_user(TOOL_RESULT_CONTINUATION_PROMPT);
                }
            }
        }
        if !clear_pending_retrieval {
            clear_pending_retrieval = self
                .finalize_search_results(cx, questions, &retrieval, &mut already_broadcasted)
                .await?;
        }
        self.pending_retrieval = if clear_pending_retrieval {
            QueryMemoryRetrieval::default()
        } else {
            retrieval
        };
        Ok(())
    }

    async fn finalize_search_results(
        &mut self,
        cx: &nuillu_module::ActivateCx<'_>,
        questions: &[String],
        retrieval: &QueryMemoryRetrieval,
        already_broadcasted: &mut bool,
    ) -> Result<bool> {
        self.session.push_ephemeral_user(FINALIZE_RETRIEVAL_PROMPT);
        for _ in 0..2 {
            let lutum = self.llm.lutum().await;
            let outcome = self
                .session
                .text_turn()
                .tools::<QueryMemoryTools>()
                .available_tools([
                    QueryMemoryToolsSelector::BroadcastSearchResults,
                    QueryMemoryToolsSelector::DisposeSearchResults,
                ])
                .max_output_tokens(TOOL_TURN_MAX_OUTPUT_TOKENS)
                .collect_controlled_with(
                    &lutum,
                    nuillu_module::AbortOnAvailableToolNameInText::new(),
                )
                .await
                .context("query-memory finalization text turn failed")?;

            match outcome {
                TextStepOutcomeWithTools::Finished(result) => {
                    cx.compact_and_save(&mut self.session, result.usage).await?;
                    return Ok(false);
                }
                TextStepOutcomeWithTools::FinishedNoOutput(result) => {
                    cx.compact_and_save(&mut self.session, result.usage).await?;
                    return Ok(false);
                }
                TextStepOutcomeWithTools::NeedsTools(round) => {
                    let usage = round.usage;
                    nuillu_module::emit_trace_tool_calls(&round.tool_calls);
                    if round.tool_calls.is_empty() {
                        cx.compact_and_save(&mut self.session, usage).await?;
                        return Ok(false);
                    }

                    let mut tool_results: Vec<ToolResult> = Vec::new();
                    let mut broadcasted_search_results = false;
                    let mut disposed_search_results = false;
                    for call in round.tool_calls.iter().cloned() {
                        match call {
                            QueryMemoryToolsCall::BroadcastSearchResults(call) => {
                                let output = self
                                    .broadcast_search_results(
                                        questions,
                                        retrieval,
                                        call.input.clone(),
                                        already_broadcasted,
                                    )
                                    .await
                                    .context("run final broadcast_search_results tool")?;
                                broadcasted_search_results |= output.broadcasted;
                                tool_results.push(call.complete(output).context(
                                    "complete final broadcast_search_results tool call",
                                )?);
                            }
                            QueryMemoryToolsCall::DisposeSearchResults(call) => {
                                disposed_search_results = true;
                                tool_results.push(
                                    call.complete(DisposeSearchResultsOutput {
                                        disposed: true,
                                        message: "search results disposed for this activation"
                                            .to_owned(),
                                    })
                                    .context("complete dispose_search_results tool call")?,
                                );
                            }
                            QueryMemoryToolsCall::SearchMemory(_)
                            | QueryMemoryToolsCall::FetchLinkedMemories(_) => {
                                unreachable!("finalization toolset excludes retrieval tools")
                            }
                        }
                    }
                    round
                        .commit(&mut self.session, tool_results)
                        .context("commit query-memory finalization tool round")?;
                    cx.compact_and_save(&mut self.session, usage).await?;

                    if broadcasted_search_results || disposed_search_results {
                        return Ok(true);
                    }
                    self.session.push_ephemeral_user(FINALIZE_RETRIEVAL_PROMPT);
                }
            }
        }
        Ok(false)
    }

    async fn prior_query_memory_searches(&self) -> Option<String> {
        let records = self.memo.recent_logs().await;
        if records.is_empty() {
            return None;
        }
        let mut out = String::from("Recent query-memory searches already attempted:");
        for record in records {
            let data = record.data();
            out.push_str(&format!(
                "\n- broadcast {} at {}",
                record.index,
                record.written_at.to_rfc3339()
            ));
            for request in &data.requests {
                out.push_str(&format!("\n  request: {}", request.trim()));
            }
            for search in &data.searches {
                let hit_indices = if search.hit_indices.is_empty() {
                    "none".to_owned()
                } else {
                    search
                        .hit_indices
                        .iter()
                        .map(ToString::to_string)
                        .collect::<Vec<_>>()
                        .join(", ")
                };
                out.push_str(&format!(
                    "\n  search: {}; limit {}; hit indices: {}",
                    search.query.trim(),
                    search.limit,
                    hit_indices
                ));
            }
        }
        Some(out)
    }

    async fn search_memory(
        &self,
        args: SearchMemoryArgs,
        now: DateTime<Utc>,
    ) -> Result<SearchMemoryOutput> {
        let limit = args.limit.clamp(1, 16);
        let records = self
            .memory
            .search(&args.query, limit)
            .await
            .context("search memory")?;
        let targets = records
            .iter()
            .map(MemoryUsageTarget::from)
            .collect::<Vec<_>>();
        self.memory.record_accesses(&targets).await;
        let mut hits = Vec::with_capacity(records.len());
        for record in records {
            let linked_neighbor_count = self
                .linked_memory
                .linked(&LinkedMemoryQuery {
                    memory_indexes: vec![record.index.clone()],
                    relation_filter: Vec::new(),
                    direction: MemoryLinkDirection::Both,
                    offset: 0,
                    limit: DEFAULT_LINKED_MEMORY_LIMIT,
                })
                .await
                .context("count linked memory neighbors for search hit")?
                .len();
            hits.push(QueryMemoryHit {
                index: record.index,
                content: render_memory_for_llm(record.content.as_str(), record.occurred_at, now),
                rank: record.rank,
                occurred_at: record.occurred_at,
                stored_at: record.stored_at,
                kind: record.kind,
                concepts: record.concepts,
                tags: record.tags,
                affect_arousal: record.affect_arousal,
                valence: record.valence,
                emotion: record.emotion,
                linked_neighbor_count,
            });
        }
        Ok(SearchMemoryOutput { hits })
    }

    async fn fetch_linked_memories(
        &self,
        args: FetchLinkedMemoriesArgs,
        now: DateTime<Utc>,
    ) -> Result<FetchLinkedMemoriesOutput> {
        let records = self
            .linked_memory
            .linked(&LinkedMemoryQuery {
                memory_indexes: args.memory_indexes,
                relation_filter: Vec::new(),
                direction: MemoryLinkDirection::Both,
                offset: 0,
                limit: DEFAULT_LINKED_MEMORY_LIMIT,
            })
            .await
            .context("fetch linked memory")?;
        let targets = records
            .iter()
            .map(|linked| MemoryUsageTarget::from(&linked.record))
            .collect::<Vec<_>>();
        self.memory.record_accesses(&targets).await;
        Ok(FetchLinkedMemoriesOutput {
            hits: records
                .into_iter()
                .map(|linked| render_linked_hit(linked, now))
                .collect(),
        })
    }

    async fn next_batch(&mut self) -> Result<QueryMemoryBatch> {
        let mut batch = self.await_first_batch().await?;
        self.collect_ready_events_into_batch(&mut batch)?;
        Ok(batch)
    }

    async fn await_first_batch(&mut self) -> Result<QueryMemoryBatch> {
        let _ = self.cognition_updates.next_item().await?;
        Ok(QueryMemoryBatch::cognition_log_update())
    }

    fn collect_ready_events_into_batch(&mut self, batch: &mut QueryMemoryBatch) -> Result<()> {
        if !self.cognition_updates.take_ready_items()?.items.is_empty() {
            batch.mark_cognition_updated();
        }
        Ok(())
    }
}

#[derive(Debug, Default)]
pub struct QueryMemoryBatch {
    pub(crate) cognition_updated: bool,
}

impl QueryMemoryBatch {
    fn cognition_log_update() -> Self {
        Self {
            cognition_updated: true,
        }
    }

    fn mark_cognition_updated(&mut self) {
        self.cognition_updated = true;
    }
}

fn select_hits(hits: &[QueryMemoryHit], indexes: &[MemoryIndex]) -> Vec<QueryMemoryHit> {
    let wanted = indexes.iter().cloned().collect::<HashSet<_>>();
    let mut seen = HashSet::new();
    hits.iter()
        .filter(|hit| wanted.contains(&hit.index) && seen.insert(hit.index.clone()))
        .cloned()
        .collect()
}

fn select_linked_hits(
    hits: &[QueryMemoryLinkedHit],
    indexes: &[MemoryIndex],
) -> Vec<QueryMemoryLinkedHit> {
    let wanted = indexes.iter().cloned().collect::<HashSet<_>>();
    let mut seen = HashSet::new();
    hits.iter()
        .filter(|hit| wanted.contains(&hit.index) && seen.insert(hit.index.clone()))
        .cloned()
        .collect()
}

fn usage_targets(
    hits: &[QueryMemoryHit],
    linked_hits: &[QueryMemoryLinkedHit],
) -> Vec<MemoryUsageTarget> {
    hits.iter()
        .map(|hit| MemoryUsageTarget::new(hit.index.clone(), hit.rank, hit.occurred_at))
        .chain(
            linked_hits
                .iter()
                .map(|hit| MemoryUsageTarget::new(hit.index.clone(), hit.rank, hit.occurred_at)),
        )
        .collect()
}

fn hit_contents(hits: &[QueryMemoryHit]) -> String {
    let mut contents = Vec::new();
    for hit in hits {
        let content = hit.content.trim();
        if !content.is_empty() && !contents.iter().any(|item: &String| item.ends_with(content)) {
            contents.push(format_memory_with_affect(
                content,
                hit.affect_arousal,
                hit.valence,
                &hit.emotion,
            ));
        }
    }
    contents.join("\n\n")
}

fn linked_hit_contents(hits: &[QueryMemoryLinkedHit]) -> String {
    let mut contents = Vec::<String>::new();
    for hit in hits {
        let content = hit.content.trim();
        if !content.is_empty() && !contents.iter().any(|item| item.ends_with(content)) {
            contents.push(format!(
                "[{:?} {:?} {} -> {}] {}",
                hit.link.relation,
                hit.link.freeform_relation,
                hit.link.from_memory,
                hit.link.to_memory,
                format_memory_with_affect(content, hit.affect_arousal, hit.valence, &hit.emotion)
            ));
        }
    }
    contents.join("\n\n")
}

fn format_memory_with_affect(
    content: &str,
    affect_arousal: f32,
    valence: f32,
    emotion: &str,
) -> String {
    let emotion = emotion.trim();
    if emotion.is_empty() && affect_arousal == 0.0 && valence == 0.0 {
        return content.to_owned();
    }
    format!(
        "[stored affect: arousal {:.2}, valence {:.2}, emotion {}] {}",
        affect_arousal,
        valence,
        if emotion.is_empty() {
            "unknown"
        } else {
            emotion
        },
        content
    )
}

fn render_memo(
    _requests: &[String],
    _searches: &[QueryMemoryMemoSearch],
    hits: &[QueryMemoryHit],
    linked_hits: &[QueryMemoryLinkedHit],
) -> String {
    let retrieved = hit_contents(hits);
    let linked = linked_hit_contents(linked_hits);
    if retrieved.is_empty() && linked.is_empty() {
        return String::new();
    }

    let mut sections = Vec::new();
    if !retrieved.is_empty() {
        sections.push(format!("Retrieved memory evidence:\n{retrieved}"));
    }
    if !linked.is_empty() {
        sections.push(format!("Linked memory evidence:\n{linked}"));
    }
    sections.join("\n\n")
}

fn render_linked_hit(linked: LinkedMemoryRecord, now: DateTime<Utc>) -> QueryMemoryLinkedHit {
    let record = linked.record;
    QueryMemoryLinkedHit {
        index: record.index,
        content: render_memory_for_llm(record.content.as_str(), record.occurred_at, now),
        rank: record.rank,
        occurred_at: record.occurred_at,
        stored_at: record.stored_at,
        kind: record.kind,
        concepts: record.concepts,
        tags: record.tags,
        affect_arousal: record.affect_arousal,
        valence: record.valence,
        emotion: record.emotion,
        link: linked.link,
    }
}

fn format_memory_questions(questions: &[String]) -> String {
    let mut out = String::from("Memory questions to investigate:");
    for question in questions
        .iter()
        .map(|question| question.trim())
        .filter(|q| !q.is_empty())
    {
        out.push_str("\n- ");
        out.push_str(question);
    }
    out
}

#[async_trait(?Send)]
impl Module for QueryMemoryModule {
    type Batch = QueryMemoryBatch;

    fn id() -> &'static str {
        "query-memory"
    }

    fn peer_context() -> Option<&'static str> {
        Some("Query-memory brings relevant remembered experience into the current cognitive state.")
    }

    fn allocation_hint() -> Option<&'static str> {
        Some(
            "Raise query-memory when current cognition needs remembered context, past experience, identity facts, or learned background. Keep it low when the needed context is already present or the next step is interpretation rather than recall.",
        )
    }

    async fn next_batch(&mut self) -> Result<Self::Batch> {
        QueryMemoryModule::next_batch(self).await
    }

    async fn activate(
        &mut self,
        cx: &nuillu_module::ActivateCx<'_>,
        batch: &Self::Batch,
    ) -> Result<()> {
        if batch.cognition_updated {
            self.activate_cognition_update(cx).await?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fetch_linked_memories_args_accepts_memory_indexes_only_payload() {
        let args: FetchLinkedMemoriesArgs = serde_json::from_value(serde_json::json!({
            "memory_indexes": ["koro-approach-primary"]
        }))
        .expect("compatible fetch_linked_memories payload should deserialize");

        assert_eq!(args.memory_indexes.len(), 1);
        assert_eq!(args.memory_indexes[0].as_str(), "koro-approach-primary");
    }

    #[test]
    fn broadcast_plaintext_contains_only_retrieved_evidence() {
        let memo = render_memo(
            &["Find the useful rule.".to_owned()],
            &[QueryMemoryMemoSearch {
                query: "useful rule".to_owned(),
                limit: 8,
                hit_indices: vec![],
            }],
            &[QueryMemoryHit {
                index: MemoryIndex::new("rule-1"),
                content: "A concrete remembered rule.".to_owned(),
                rank: MemoryRank::ShortTerm,
                occurred_at: None,
                stored_at: Utc::now(),
                kind: MemoryKind::Statement,
                concepts: vec![],
                tags: vec![],
                affect_arousal: 0.0,
                valence: 0.0,
                emotion: String::new(),
                linked_neighbor_count: 0,
            }],
            &[],
        );

        assert!(memo.starts_with("Retrieved memory evidence:\nA concrete remembered rule."));
        assert!(!memo.contains("Query intent:"));
        assert!(!memo.contains("Search queries:"));
    }
}
