use std::collections::HashSet;

use anyhow::{Context, Result, bail};
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use lutum::{ModelInput, Session, StagedTextStepOutcomeWithTools, TextStepOutcomeWithTools};
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
Each activation has two possible LLM tool-call stages.
First, plan retrieval by calling plan_memory_queries exactly once. Choose no_op only when the
current state, previous session history, and recent cognition already contain enough context, or
when memory evidence is clearly irrelevant. Do not use no_op to claim no relevant memory exists
before searching when the current cognition asks for remembered facts, rules, procedures, people,
places, objects, or past events. Memory inventory counts alone are not evidence that no relevant
memory exists. Otherwise provide up to three concrete memory search queries.
Search for concrete requested facts, proper nouns, species/body/peer/world terms, route rules, and
needed_fact phrases. Prefer limit 8 unless the request is obviously exact and narrow. Set
include_linked when related context, procedures, rules, approach guidance, peer/body cues, or
associations may be needed. Do not search for generic phrases such as "useful memory context" when
a concrete question is available.
The runtime executes searches and linked-memory fetches deterministically, then asks for evidence
selection. In that second stage, call select_memory_evidence exactly once. Select only exact memory
index strings from the provided tool result. Search hits are candidates, not automatic output.
You may select flat hits and linked hits, or reject all evidence.
Do not produce user-facing answers, decide final truth, explain results from outside tool output, or
use plain assistant text as a data channel."#;

const COMPACTED_QUERY_MEMORY_SESSION_PREFIX: &str = "Compacted query-memory session history:";
const MEMO_CONTEXT_WINDOW: LlmContextWindow = LlmContextWindow::new(8, 1_200, 4_800);
const TOOL_TURN_MAX_OUTPUT_TOKENS: u32 = 768;
const MAX_PLANNED_SEARCHES: usize = 3;
const MAX_SEARCH_LIMIT: usize = 16;
const SESSION_COMPACTION_FOCUS: &str = r#"Preserve memo evidence, query requests, memory search
arguments, useful memory hits, rejected broad searches, and allocation/cognition context that future
retrieval should remember."#;
const QUERY_PLAN_INSTRUCTION: &str = r#"Output instruction: call plan_memory_queries exactly once.
Use disposition no_op when no memory retrieval is useful for this activation. Use disposition search
with up to three concrete searches when memory evidence may help. If the current cognition asks
about remembered facts, rules, procedures, people, places, objects, or past events and the memory
inventory is nonempty, search before concluding there is no relevant memory. Memory inventory
counts alone are not evidence that no relevant memory exists. Use limit 8 by default. Set
include_linked=true for procedural/context questions where related memories may carry supporting
evidence. Do not write plain assistant text."#;
const EVIDENCE_SELECTION_INSTRUCTION: &str = r#"Output instruction: call select_memory_evidence
exactly once. Select exact memory index strings from the fresh search result above, or reject all
evidence. Include linked_hit_indexes when a linked hit adds supporting evidence or its link
relationship is relevant to why the evidence belongs in memo, even if the same index is also
present as a flat hit. Do not write plain assistant text."#;

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

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
struct SearchMemoryArgs {
    query: String,
    limit: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
struct SearchMemoryOutput {
    hits: Vec<QueryMemoryHit>,
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

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
struct FetchLinkedMemoriesArgs {
    memory_indexes: Vec<MemoryIndex>,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
struct FetchLinkedMemoriesOutput {
    hits: Vec<QueryMemoryLinkedHit>,
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

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum QueryMemoryPlanDisposition {
    Search,
    NoOp,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct PlannedMemorySearch {
    pub query: String,
    pub limit: usize,
    #[serde(default)]
    pub include_linked: bool,
}

#[lutum::tool_input(name = "plan_memory_queries", output = PlanMemoryQueriesOutput)]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct PlanMemoryQueriesArgs {
    pub disposition: QueryMemoryPlanDisposition,
    #[serde(default)]
    pub searches: Vec<PlannedMemorySearch>,
    #[serde(default)]
    pub reason: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct PlanMemoryExecutedSearch {
    pub query: String,
    pub limit: usize,
    pub include_linked: bool,
    pub hit_indices: Vec<MemoryIndex>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct PlanMemoryQueriesOutput {
    pub disposition: QueryMemoryPlanDisposition,
    pub searches: Vec<PlanMemoryExecutedSearch>,
    pub hits: Vec<QueryMemoryHit>,
    pub linked_hits: Vec<QueryMemoryLinkedHit>,
    pub message: String,
}

#[lutum::tool_input(name = "select_memory_evidence", output = SelectMemoryEvidenceOutput)]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct SelectMemoryEvidenceArgs {
    #[serde(default)]
    pub hit_indexes: Vec<MemoryIndex>,
    #[serde(default)]
    pub linked_hit_indexes: Vec<MemoryIndex>,
    #[serde(default)]
    pub reject_all: bool,
    #[serde(default)]
    pub reason: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct SelectMemoryEvidenceOutput {
    pub memo_written: bool,
    pub selected_hits: usize,
    pub selected_linked_hits: usize,
    pub used_indexes: Vec<MemoryIndex>,
    pub message: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema, lutum::Toolset)]
pub enum QueryMemoryPlanTools {
    PlanMemoryQueries(PlanMemoryQueriesArgs),
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema, lutum::Toolset)]
pub enum QueryMemoryFilterTools {
    SelectMemoryEvidence(SelectMemoryEvidenceArgs),
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
    executed_searches: Vec<PlanMemoryExecutedSearch>,
    hits: Vec<QueryMemoryHit>,
    linked_hits: Vec<QueryMemoryLinkedHit>,
}

impl QueryMemoryRetrieval {
    fn has_evidence(&self) -> bool {
        !self.hits.is_empty() || !self.linked_hits.is_empty()
    }
}

#[derive(Clone, Debug)]
struct QueryMemoryFilterOutcome {
    args: SelectMemoryEvidenceArgs,
    output: SelectMemoryEvidenceOutput,
    memo: Option<(QueryMemoryMemo, String)>,
    access_targets: Vec<MemoryUsageTarget>,
    use_targets: Vec<MemoryUsageTarget>,
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
        let session_len_before_activation = self.session.input().items().len();
        let result = self.run_query_memory_lifecycle(cx).await;
        if result.is_err() {
            truncate_session_to(&mut self.session, session_len_before_activation);
        }
        result
    }

    async fn fresh_hits(&self, hits: &[QueryMemoryHit]) -> Vec<QueryMemoryHit> {
        let mut seen = self.memoed_evidence_indexes().await;
        hits.iter()
            .filter(|hit| seen.insert(hit.index.clone()))
            .cloned()
            .collect()
    }

    async fn fresh_linked_hits(&self, hits: &[QueryMemoryLinkedHit]) -> Vec<QueryMemoryLinkedHit> {
        let mut seen = self.memoed_evidence_indexes().await;
        hits.iter()
            .filter(|hit| seen.insert(hit.index.clone()))
            .cloned()
            .collect()
    }

    async fn memoed_evidence_indexes(&self) -> HashSet<MemoryIndex> {
        self.memo
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
            .collect()
    }

    async fn filter_recent_memo_evidence(
        &self,
        mut retrieval: QueryMemoryRetrieval,
    ) -> QueryMemoryRetrieval {
        let memoed = self.memoed_evidence_indexes().await;
        if memoed.is_empty() {
            return retrieval;
        }

        retrieval.hits.retain(|hit| !memoed.contains(&hit.index));
        retrieval
            .linked_hits
            .retain(|hit| !memoed.contains(&hit.index));
        let fresh_hit_indexes = retrieval
            .hits
            .iter()
            .map(|hit| hit.index.clone())
            .collect::<HashSet<_>>();
        for search in &mut retrieval.searches {
            search
                .hit_indices
                .retain(|index| fresh_hit_indexes.contains(index));
        }
        for search in &mut retrieval.executed_searches {
            search
                .hit_indices
                .retain(|index| fresh_hit_indexes.contains(index));
        }
        retrieval
    }

    async fn run_query_memory_lifecycle(
        &mut self,
        cx: &nuillu_module::ActivateCx<'_>,
    ) -> Result<()> {
        let requests = vec![self.activation_request().await];
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
        self.session.push_user(format_memory_questions(&requests));
        if let Some(prior) = prior_query_memory_searches {
            self.session.push_ephemeral_user(prior);
        }
        self.session
            .push_ephemeral_system(format_memory_context(&rank_counts));
        self.session.push_ephemeral_user(QUERY_PLAN_INSTRUCTION);

        let lutum = self.llm.lutum().await;
        let outcome = self
            .session
            .text_turn()
            .tools::<QueryMemoryPlanTools>()
            .available_tools([QueryMemoryPlanToolsSelector::PlanMemoryQueries])
            .require_tool(QueryMemoryPlanToolsSelector::PlanMemoryQueries)
            .max_output_tokens(TOOL_TURN_MAX_OUTPUT_TOKENS)
            .collect_staged_controlled_with(
                &lutum,
                nuillu_module::AbortOnAvailableToolNameInText::new(),
            )
            .await
            .context("query-memory planner text turn failed")?;

        match outcome {
            StagedTextStepOutcomeWithTools::Finished(_) => {
                bail!("query-memory planner finished without plan_memory_queries tool call")
            }
            StagedTextStepOutcomeWithTools::FinishedNoOutput(_) => {
                bail!("query-memory planner finished with no output")
            }
            StagedTextStepOutcomeWithTools::NeedsTools(round) => {
                let planner_usage = round.usage;
                nuillu_module::emit_trace_tool_calls(&round.tool_calls);
                if round.tool_calls.len() != 1 {
                    bail!(
                        "query-memory planner must call plan_memory_queries exactly once, got {} calls",
                        round.tool_calls.len()
                    );
                }
                let QueryMemoryPlanToolsCall::PlanMemoryQueries(call) = round.tool_calls[0].clone();
                let plan = normalize_plan(call.input.clone())?;

                if matches!(plan.disposition, QueryMemoryPlanDisposition::NoOp) {
                    let output = PlanMemoryQueriesOutput {
                        disposition: QueryMemoryPlanDisposition::NoOp,
                        searches: Vec::new(),
                        hits: Vec::new(),
                        linked_hits: Vec::new(),
                        message: "planner chose no memory retrieval for this activation".to_owned(),
                    };
                    let result = call
                        .complete(output)
                        .context("complete plan_memory_queries no-op tool call")?;
                    round
                        .commit(&mut self.session, [result])
                        .context("commit query-memory no-op planner round")?;
                    cx.compact_and_save(&mut self.session, planner_usage)
                        .await?;
                    return Ok(());
                }

                let retrieval = self
                    .execute_planned_searches(&plan.searches, cx.now())
                    .await
                    .context("execute planned memory searches")?;
                let had_any_evidence = retrieval.has_evidence();
                let retrieval = self.filter_recent_memo_evidence(retrieval).await;
                let plan_output = PlanMemoryQueriesOutput {
                    disposition: QueryMemoryPlanDisposition::Search,
                    searches: retrieval.executed_searches.clone(),
                    hits: retrieval.hits.clone(),
                    linked_hits: retrieval.linked_hits.clone(),
                    message: if !had_any_evidence {
                        "planned searches returned no memory hits".to_owned()
                    } else if !retrieval.has_evidence() {
                        "planned searches returned only evidence already present in query-memory memo"
                            .to_owned()
                    } else {
                        "planned searches executed; filter stage will select evidence".to_owned()
                    },
                };
                let result = call
                    .complete(plan_output)
                    .context("complete plan_memory_queries search tool call")?;
                round
                    .commit(&mut self.session, [result])
                    .context("commit query-memory planner round")?;

                if !retrieval.has_evidence() {
                    cx.compact_and_save(&mut self.session, planner_usage)
                        .await?;
                    return Ok(());
                }

                let filter = self
                    .select_memory_evidence(cx, &requests, &retrieval)
                    .await
                    .context("select memory evidence")?;
                self.session
                    .push_user(format_filter_decision_summary(&filter.args, &filter.output));
                cx.compact_and_save(&mut self.session, planner_usage)
                    .await?;

                if !filter.access_targets.is_empty() {
                    self.memory.record_accesses(&filter.access_targets).await;
                }
                if let Some((payload, content)) = filter.memo {
                    self.memo.write_cognitive(payload, content).await;
                }
                if !filter.use_targets.is_empty() {
                    self.memory.record_uses(&filter.use_targets).await;
                }
                Ok(())
            }
        }
    }

    async fn activation_request(&self) -> String {
        self.blackboard
            .read(|bb| {
                let entries = bb.cognition_log().entries().to_vec();
                let latest = entries
                    .last()
                    .map(|entry| entry.text.as_str())
                    .unwrap_or("current cognition log");
                format!("Find stable memory that clarifies this cognition-log context: {latest}")
            })
            .await
    }

    async fn execute_planned_searches(
        &self,
        searches: &[PlannedMemorySearch],
        now: DateTime<Utc>,
    ) -> Result<QueryMemoryRetrieval> {
        let mut retrieval = QueryMemoryRetrieval::default();
        let mut seen_hits = HashSet::new();
        let mut seen_linked_hits = HashSet::new();

        for search in searches {
            let output = self
                .search_memory(
                    SearchMemoryArgs {
                        query: search.query.clone(),
                        limit: search.limit,
                    },
                    now,
                )
                .await
                .context("run planned memory search")?;
            let mut local_seen_hits = HashSet::new();
            let mut search_hit_indices = Vec::new();
            let mut linked_seed_indexes = Vec::new();

            for hit in output.hits {
                if !local_seen_hits.insert(hit.index.clone()) {
                    continue;
                }
                search_hit_indices.push(hit.index.clone());
                if search.include_linked && hit.linked_neighbor_count > 0 {
                    linked_seed_indexes.push(hit.index.clone());
                }
                if seen_hits.insert(hit.index.clone()) {
                    retrieval.hits.push(hit);
                }
            }

            retrieval.searches.push(QueryMemoryMemoSearch {
                query: search.query.clone(),
                limit: search.limit,
                hit_indices: search_hit_indices.clone(),
            });
            retrieval.executed_searches.push(PlanMemoryExecutedSearch {
                query: search.query.clone(),
                limit: search.limit,
                include_linked: search.include_linked,
                hit_indices: search_hit_indices,
            });

            if linked_seed_indexes.is_empty() {
                continue;
            }
            let output = self
                .fetch_linked_memories(
                    FetchLinkedMemoriesArgs {
                        memory_indexes: linked_seed_indexes,
                    },
                    now,
                )
                .await
                .context("fetch linked memories for planned search")?;
            for linked_hit in output.hits {
                if seen_linked_hits.insert(linked_hit.index.clone()) {
                    retrieval.linked_hits.push(linked_hit);
                }
            }
        }

        Ok(retrieval)
    }

    async fn select_memory_evidence(
        &self,
        cx: &nuillu_module::ActivateCx<'_>,
        requests: &[String],
        retrieval: &QueryMemoryRetrieval,
    ) -> Result<QueryMemoryFilterOutcome> {
        let mut input = ModelInput::new()
            .system(nuillu_module::format_system_seed(
                self.system_prompt(cx),
                false,
                cx.identity_memories(),
                cx.now(),
            ))
            .user(format_evidence_selection_request(requests, retrieval))
            .user(EVIDENCE_SELECTION_INSTRUCTION);
        let lutum = self.llm.lutum().await;
        let outcome = lutum
            .text_turn(input.clone())
            .tools::<QueryMemoryFilterTools>()
            .available_tools([QueryMemoryFilterToolsSelector::SelectMemoryEvidence])
            .require_tool(QueryMemoryFilterToolsSelector::SelectMemoryEvidence)
            .max_output_tokens(TOOL_TURN_MAX_OUTPUT_TOKENS)
            .collect_controlled_with(nuillu_module::AbortOnAvailableToolNameInText::new())
            .await
            .context("query-memory filter text turn failed")?;

        match outcome {
            TextStepOutcomeWithTools::Finished(_) => {
                bail!("query-memory filter finished without select_memory_evidence tool call")
            }
            TextStepOutcomeWithTools::FinishedNoOutput(_) => {
                bail!("query-memory filter finished with no output")
            }
            TextStepOutcomeWithTools::NeedsTools(round) => {
                nuillu_module::emit_trace_tool_calls(&round.tool_calls);
                if round.tool_calls.len() != 1 {
                    bail!(
                        "query-memory filter must call select_memory_evidence exactly once, got {} calls",
                        round.tool_calls.len()
                    );
                }
                let QueryMemoryFilterToolsCall::SelectMemoryEvidence(call) =
                    round.tool_calls[0].clone();
                let args = call.input.clone();
                let outcome = self
                    .build_filter_outcome(requests, retrieval, args.clone())
                    .await?;
                let result = call
                    .complete(outcome.output.clone())
                    .context("complete select_memory_evidence tool call")?;
                round
                    .commit_into(&mut input, [result])
                    .context("commit query-memory filter tool round")?;
                Ok(outcome)
            }
        }
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
                "\n- memo {} at {}",
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

    async fn build_filter_outcome(
        &self,
        requests: &[String],
        retrieval: &QueryMemoryRetrieval,
        args: SelectMemoryEvidenceArgs,
    ) -> Result<QueryMemoryFilterOutcome> {
        validate_filter_selection(retrieval, &args)?;

        let access_targets = usage_targets(&retrieval.hits, &retrieval.linked_hits);
        if args.reject_all || (args.hit_indexes.is_empty() && args.linked_hit_indexes.is_empty()) {
            return Ok(QueryMemoryFilterOutcome {
                args,
                output: SelectMemoryEvidenceOutput {
                    memo_written: false,
                    selected_hits: 0,
                    selected_linked_hits: 0,
                    used_indexes: Vec::new(),
                    message: "filter rejected all memory evidence".to_owned(),
                },
                memo: None,
                access_targets,
                use_targets: Vec::new(),
            });
        }

        let selected_hits = select_hits(&retrieval.hits, &args.hit_indexes);
        let selected_linked_hits =
            select_linked_hits(&retrieval.linked_hits, &args.linked_hit_indexes);
        let hits = self.fresh_hits(&selected_hits).await;
        let linked_hits = self.fresh_linked_hits(&selected_linked_hits).await;
        let content = render_memo(requests, &retrieval.searches, &hits, &linked_hits);
        let use_targets = usage_targets(&hits, &linked_hits);
        let used_indexes = unique_target_indexes(&use_targets);

        if content.is_empty() {
            return Ok(QueryMemoryFilterOutcome {
                args,
                output: SelectMemoryEvidenceOutput {
                    memo_written: false,
                    selected_hits: hits.len(),
                    selected_linked_hits: linked_hits.len(),
                    used_indexes,
                    message: "no fresh selected evidence was available for memo".to_owned(),
                },
                memo: None,
                access_targets,
                use_targets,
            });
        }

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
        Ok(QueryMemoryFilterOutcome {
            args,
            output: SelectMemoryEvidenceOutput {
                memo_written: true,
                selected_hits: hits.len(),
                selected_linked_hits: linked_hits.len(),
                used_indexes,
                message: "selected memory evidence was written to memo".to_owned(),
            },
            memo: Some((payload, content)),
            access_targets,
            use_targets,
        })
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

fn normalize_plan(args: PlanMemoryQueriesArgs) -> Result<PlanMemoryQueriesArgs> {
    match args.disposition {
        QueryMemoryPlanDisposition::NoOp => {
            if !args.searches.is_empty() {
                bail!("query-memory planner used no_op disposition with planned searches");
            }
            Ok(PlanMemoryQueriesArgs {
                disposition: QueryMemoryPlanDisposition::NoOp,
                searches: Vec::new(),
                reason: args.reason,
            })
        }
        QueryMemoryPlanDisposition::Search => {
            let searches = args
                .searches
                .into_iter()
                .take(MAX_PLANNED_SEARCHES)
                .map(|search| {
                    let query = search.query.trim().to_owned();
                    if query.is_empty() {
                        bail!("query-memory planner emitted a blank search query");
                    }
                    Ok(PlannedMemorySearch {
                        query,
                        limit: search.limit.clamp(1, MAX_SEARCH_LIMIT),
                        include_linked: search.include_linked,
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            if searches.is_empty() {
                bail!("query-memory planner used search disposition without searches");
            }
            Ok(PlanMemoryQueriesArgs {
                disposition: QueryMemoryPlanDisposition::Search,
                searches,
                reason: args.reason,
            })
        }
    }
}

fn validate_filter_selection(
    retrieval: &QueryMemoryRetrieval,
    args: &SelectMemoryEvidenceArgs,
) -> Result<()> {
    if args.reject_all && (!args.hit_indexes.is_empty() || !args.linked_hit_indexes.is_empty()) {
        bail!("query-memory filter cannot reject_all and select evidence indexes");
    }

    let flat_indexes = retrieval
        .hits
        .iter()
        .map(|hit| hit.index.clone())
        .collect::<HashSet<_>>();
    let linked_indexes = retrieval
        .linked_hits
        .iter()
        .map(|hit| hit.index.clone())
        .collect::<HashSet<_>>();

    let unknown_flat = args
        .hit_indexes
        .iter()
        .filter(|index| !flat_indexes.contains(*index))
        .map(ToString::to_string)
        .collect::<Vec<_>>();
    let unknown_linked = args
        .linked_hit_indexes
        .iter()
        .filter(|index| !linked_indexes.contains(*index))
        .map(ToString::to_string)
        .collect::<Vec<_>>();

    if !unknown_flat.is_empty() || !unknown_linked.is_empty() {
        bail!(
            "query-memory filter selected unknown evidence indexes; flat={}; linked={}",
            unknown_flat.join(", "),
            unknown_linked.join(", ")
        );
    }

    Ok(())
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

fn unique_target_indexes(targets: &[MemoryUsageTarget]) -> Vec<MemoryIndex> {
    let mut seen = HashSet::new();
    targets
        .iter()
        .filter_map(|target| {
            seen.insert(target.index.clone())
                .then_some(target.index.clone())
        })
        .collect()
}

fn format_evidence_selection_request(
    requests: &[String],
    retrieval: &QueryMemoryRetrieval,
) -> String {
    let mut out = String::from("Select memo evidence from these deterministic memory results.");
    out.push_str(
        " Use hit_indexes for directly retrieved flat evidence. Use linked_hit_indexes when the \
         linked hit's content or relation adds evidence that should remain visible in memo.",
    );
    out.push_str("\n\nMemory request(s):");
    for request in requests {
        out.push_str("\n- ");
        out.push_str(request.trim());
    }

    out.push_str("\n\nExecuted searches:");
    for search in &retrieval.executed_searches {
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
            "\n- query: {}; limit: {}; include_linked: {}; hit indexes: {}",
            search.query, search.limit, search.include_linked, hit_indices
        ));
    }

    out.push_str("\n\nFlat hits available for hit_indexes:");
    if retrieval.hits.is_empty() {
        out.push_str("\n- none");
    }
    for hit in &retrieval.hits {
        out.push_str(&format!(
            "\n- index: {}; rank: {:?}; kind: {:?}; linked_neighbor_count: {}; content: {}",
            hit.index,
            hit.rank,
            hit.kind,
            hit.linked_neighbor_count,
            hit.content.trim()
        ));
    }

    out.push_str("\n\nLinked hits available for linked_hit_indexes:");
    if retrieval.linked_hits.is_empty() {
        out.push_str("\n- none");
    }
    for hit in &retrieval.linked_hits {
        out.push_str(&format!(
            "\n- index: {}; rank: {:?}; kind: {:?}; link: {:?} {:?} {} -> {}; content: {}",
            hit.index,
            hit.rank,
            hit.kind,
            hit.link.relation,
            hit.link.freeform_relation,
            hit.link.from_memory,
            hit.link.to_memory,
            hit.content.trim()
        ));
    }

    out
}

fn format_filter_decision_summary(
    args: &SelectMemoryEvidenceArgs,
    output: &SelectMemoryEvidenceOutput,
) -> String {
    format!(
        "Query-memory filter decision: reject_all={}; selected hit_indexes=[{}]; selected linked_hit_indexes=[{}]; memo_written={}; used_indexes=[{}]; reason={}",
        args.reject_all,
        args.hit_indexes
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join(", "),
        args.linked_hit_indexes
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join(", "),
        output.memo_written,
        output
            .used_indexes
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join(", "),
        args.reason.trim()
    )
}

fn truncate_session_to(session: &mut Session, len: usize) {
    session.input_mut().items_mut().truncate(len);
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

    use std::cell::{Cell, RefCell};
    use std::rc::Rc;
    use std::sync::{Arc, Mutex};

    use lutum::{
        AdapterStructuredTurn, AdapterTextTurn, AgentError, ErasedStructuredTurnEventStream,
        ErasedTextTurnEventStream, FinishReason, Lutum, MockLlmAdapter, MockTextScenario,
        ModelInputItem, RawTextTurnEvent, SharedPoolBudgetManager, SharedPoolBudgetOptions,
        TurnAdapter, Usage,
    };
    use nuillu_blackboard::{Blackboard, Bpm, ModulePolicy, TypedMemoLogRecord, linear_ratio_fn};
    use nuillu_module::ports::{NoopCognitionLogRepository, PortError, SystemClock};
    use nuillu_module::{
        CapabilityProviderPorts, CapabilityProviders, CognitionLogUpdated, LlmConcurrencyLimiter,
        LutumTiers, ModuleRegistry, SessionCompactionPolicy, SessionCompactionRuntime,
    };
    use nuillu_types::{
        MemoryContent, ModelTier, ModuleInstanceId, ReplicaCapRange, ReplicaIndex, builtin,
    };

    use crate::store::{
        IndexedMemory, MemoryLinkRelation, MemoryQuery, MemoryRecord, MemoryStore, NewMemory,
        NewMemoryLink,
    };

    #[derive(Clone)]
    struct CapturingAdapter {
        inner: MockLlmAdapter,
        text_inputs: Arc<Mutex<Vec<lutum::ModelInput>>>,
    }

    impl CapturingAdapter {
        fn new(inner: MockLlmAdapter) -> Self {
            Self {
                inner,
                text_inputs: Arc::new(Mutex::new(Vec::new())),
            }
        }

        fn text_inputs(&self) -> Vec<lutum::ModelInput> {
            self.text_inputs.lock().unwrap().clone()
        }
    }

    #[async_trait]
    impl TurnAdapter for CapturingAdapter {
        async fn text_turn(
            &self,
            input: lutum::ModelInput,
            turn: AdapterTextTurn,
        ) -> Result<ErasedTextTurnEventStream, AgentError> {
            self.text_inputs.lock().unwrap().push(input.clone());
            self.inner.text_turn(input, turn).await
        }

        async fn structured_turn(
            &self,
            input: lutum::ModelInput,
            turn: AdapterStructuredTurn,
        ) -> Result<ErasedStructuredTurnEventStream, AgentError> {
            self.inner.structured_turn(input, turn).await
        }
    }

    #[derive(Default)]
    struct QueryMemoryTestStore {
        records: RefCell<Vec<MemoryRecord>>,
        linked_records: RefCell<Vec<LinkedMemoryRecord>>,
        next_index: Cell<u64>,
    }

    #[async_trait(?Send)]
    impl MemoryStore for QueryMemoryTestStore {
        async fn insert(
            &self,
            mem: NewMemory,
            stored_at: DateTime<Utc>,
        ) -> Result<MemoryRecord, PortError> {
            let id = self.next_index.get();
            self.next_index.set(id + 1);
            let record = MemoryRecord {
                index: MemoryIndex::new(format!("query-test-memory-{id}")),
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
            self.records.borrow_mut().push(record.clone());
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
            self.records.borrow_mut().push(record.clone());
            Ok(record)
        }

        async fn compact(
            &self,
            mem: NewMemory,
            _sources: &[MemoryIndex],
            stored_at: DateTime<Utc>,
        ) -> Result<MemoryRecord, PortError> {
            self.insert(mem, stored_at).await
        }

        async fn put_compacted(
            &self,
            mem: IndexedMemory,
            _sources: &[MemoryIndex],
        ) -> Result<MemoryRecord, PortError> {
            self.put(mem).await
        }

        async fn get(&self, index: &MemoryIndex) -> Result<Option<MemoryRecord>, PortError> {
            Ok(self
                .records
                .borrow()
                .iter()
                .find(|record| &record.index == index)
                .cloned())
        }

        async fn list_by_rank(&self, rank: MemoryRank) -> Result<Vec<MemoryRecord>, PortError> {
            Ok(self
                .records
                .borrow()
                .iter()
                .filter(|record| record.rank == rank)
                .cloned()
                .collect())
        }

        async fn search(&self, q: &MemoryQuery) -> Result<Vec<MemoryRecord>, PortError> {
            Ok(self
                .records
                .borrow()
                .iter()
                .take(q.limit)
                .cloned()
                .collect())
        }

        async fn linked(
            &self,
            q: &LinkedMemoryQuery,
        ) -> Result<Vec<LinkedMemoryRecord>, PortError> {
            Ok(self
                .linked_records
                .borrow()
                .iter()
                .filter(|linked| {
                    q.memory_indexes.contains(&linked.link.from_memory)
                        || q.memory_indexes.contains(&linked.link.to_memory)
                })
                .take(q.limit)
                .cloned()
                .collect())
        }

        async fn upsert_link(
            &self,
            link: NewMemoryLink,
            updated_at: DateTime<Utc>,
        ) -> Result<MemoryLink, PortError> {
            Ok(MemoryLink {
                from_memory: link.from_memory,
                to_memory: link.to_memory,
                relation: link.relation,
                freeform_relation: link.freeform_relation,
                strength: link.strength,
                confidence: link.confidence,
                updated_at,
            })
        }

        async fn delete(&self, index: &MemoryIndex) -> Result<(), PortError> {
            self.records
                .borrow_mut()
                .retain(|record| &record.index != index);
            Ok(())
        }
    }

    fn test_now() -> DateTime<Utc> {
        DateTime::from_timestamp(1_700_000_000, 0).unwrap()
    }

    fn usage() -> Usage {
        Usage {
            input_tokens: 1,
            output_tokens: 1,
            total_tokens: 2,
            ..Usage::zero()
        }
    }

    fn plan_scenario(disposition: &str, searches: serde_json::Value) -> MockTextScenario {
        tool_scenario(
            "query-memory-plan",
            "call-plan",
            "plan_memory_queries",
            serde_json::json!({
                "disposition": disposition,
                "searches": searches,
                "reason": "test",
            }),
        )
    }

    fn filter_scenario(
        hit_indexes: Vec<&str>,
        linked_hit_indexes: Vec<&str>,
        reject_all: bool,
    ) -> MockTextScenario {
        tool_scenario(
            "query-memory-filter",
            "call-filter",
            "select_memory_evidence",
            serde_json::json!({
                "hit_indexes": hit_indexes,
                "linked_hit_indexes": linked_hit_indexes,
                "reject_all": reject_all,
                "reason": "test",
            }),
        )
    }

    fn tool_scenario(
        request_id: &str,
        call_id: &str,
        name: &str,
        arguments: serde_json::Value,
    ) -> MockTextScenario {
        MockTextScenario::events(vec![
            Ok(RawTextTurnEvent::Started {
                request_id: Some(request_id.to_owned()),
                model: "mock".into(),
            }),
            Ok(RawTextTurnEvent::ToolCallChunk {
                id: call_id.to_owned().into(),
                name: name.to_owned().into(),
                arguments_json_delta: arguments.to_string(),
            }),
            Ok(RawTextTurnEvent::Completed {
                request_id: Some(request_id.to_owned()),
                finish_reason: FinishReason::ToolCall,
                usage: usage(),
            }),
        ])
    }

    fn text_scenario(text: &str) -> MockTextScenario {
        MockTextScenario::events(vec![
            Ok(RawTextTurnEvent::Started {
                request_id: Some("query-memory-text".into()),
                model: "mock".into(),
            }),
            Ok(RawTextTurnEvent::TextDelta { delta: text.into() }),
            Ok(RawTextTurnEvent::Completed {
                request_id: Some("query-memory-text".into()),
                finish_reason: FinishReason::Stop,
                usage: usage(),
            }),
        ])
    }

    fn record(index: &str, content: &str) -> MemoryRecord {
        MemoryRecord {
            index: MemoryIndex::new(index),
            content: MemoryContent::new(content),
            rank: MemoryRank::Permanent,
            occurred_at: Some(test_now()),
            stored_at: test_now(),
            kind: MemoryKind::Statement,
            concepts: Vec::new(),
            tags: Vec::new(),
            affect_arousal: 0.0,
            valence: 0.0,
            emotion: String::new(),
        }
    }

    fn test_caps_with_adapter(
        adapter: Arc<CapturingAdapter>,
        store: Rc<QueryMemoryTestStore>,
    ) -> (
        Blackboard,
        CapabilityProviders,
        crate::MemoryCapabilities,
        Lutum,
    ) {
        let blackboard = Blackboard::default();
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
            crate::MemoryCapabilities::new(blackboard.clone(), clock, store, Vec::new());
        (blackboard, caps, memory_caps, lutum)
    }

    async fn build_query_module(
        caps: &CapabilityProviders,
        memory_caps: crate::MemoryCapabilities,
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
                    async move {
                        Ok(QueryMemoryModule::new(
                            caps.cognition_log_updated_inbox(),
                            caps.blackboard_reader(),
                            memory_caps.retriever(),
                            memory_caps.content_reader(),
                            caps.typed_memo::<QueryMemoryMemo>(),
                            caps.llm("main").with_tier(ModelTier::Cheap).into(),
                            caps.session("main")
                                .with_tier(ModelTier::Cheap)
                                .with_auto_compaction(query_session_auto_compaction())
                                .await?,
                        ))
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

    fn activate_cx(lutum: &Lutum) -> nuillu_module::ActivateCx<'static> {
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
            test_now(),
        )
    }

    async fn run_query_memory_once(
        blackboard: &Blackboard,
        caps: &CapabilityProviders,
        module: &mut nuillu_module::AllocatedModule,
        lutum: &Lutum,
    ) -> Result<()> {
        let source = ModuleInstanceId::new(builtin::cognition_gate(), ReplicaIndex::ZERO);
        blackboard
            .append_cognition_log(
                source.clone(),
                nuillu_blackboard::CognitionLogEntry {
                    at: test_now(),
                    text: "How should Nui approach Koro near food?".to_owned(),
                    origin: nuillu_blackboard::CognitionLogOrigin::direct(source.clone()),
                },
            )
            .await;
        caps.internal_harness_io()
            .cognition_log_updated_mailbox()
            .publish(CognitionLogUpdated::EntryAppended { source })
            .await
            .expect("query-memory cognition-update subscriber exists");
        let batch = module.next_batch().await?;
        module.activate(&activate_cx(lutum), &batch).await
    }

    async fn query_memory_memos(
        blackboard: &Blackboard,
    ) -> Vec<TypedMemoLogRecord<QueryMemoryMemo>> {
        let owner = ModuleInstanceId::new(builtin::query_memory(), ReplicaIndex::ZERO);
        blackboard.typed_memo_logs::<QueryMemoryMemo>(&owner).await
    }

    #[test]
    fn memo_plaintext_contains_only_retrieved_evidence() {
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

    #[tokio::test(flavor = "current_thread")]
    async fn planner_no_op_ends_after_one_llm_turn_with_no_memo() {
        tokio::task::LocalSet::new()
            .run_until(async {
                let adapter = Arc::new(CapturingAdapter::new(
                    MockLlmAdapter::new()
                        .with_text_scenario(plan_scenario("no_op", serde_json::json!([]))),
                ));
                let store = Rc::new(QueryMemoryTestStore::default());
                let (blackboard, caps, memory_caps, lutum) =
                    test_caps_with_adapter(adapter.clone(), store);
                let mut module = build_query_module(&caps, memory_caps).await;

                run_query_memory_once(&blackboard, &caps, &mut module, &lutum)
                    .await
                    .expect("no-op activation should succeed");

                assert_eq!(adapter.text_inputs().len(), 1);
                assert!(query_memory_memos(&blackboard).await.is_empty());
            })
            .await;
    }

    #[tokio::test(flavor = "current_thread")]
    async fn search_miss_skips_filter_and_writes_no_memo() {
        tokio::task::LocalSet::new()
            .run_until(async {
                let adapter = Arc::new(CapturingAdapter::new(
                    MockLlmAdapter::new().with_text_scenario(plan_scenario(
                        "search",
                        serde_json::json!([
                            {"query": "Koro food approach", "limit": 99, "include_linked": true}
                        ]),
                    )),
                ));
                let store = Rc::new(QueryMemoryTestStore::default());
                let (blackboard, caps, memory_caps, lutum) =
                    test_caps_with_adapter(adapter.clone(), store);
                let mut module = build_query_module(&caps, memory_caps).await;

                run_query_memory_once(&blackboard, &caps, &mut module, &lutum)
                    .await
                    .expect("search miss activation should succeed");

                assert_eq!(adapter.text_inputs().len(), 1);
                assert!(query_memory_memos(&blackboard).await.is_empty());
            })
            .await;
    }

    #[tokio::test(flavor = "current_thread")]
    async fn already_memoed_evidence_is_removed_before_filter() {
        tokio::task::LocalSet::new()
            .run_until(async {
                let adapter = Arc::new(CapturingAdapter::new(
                    MockLlmAdapter::new().with_text_scenario(plan_scenario(
                        "search",
                        serde_json::json!([
                            {"query": "Koro food approach", "limit": 8, "include_linked": false}
                        ]),
                    )),
                ));
                let store = Rc::new(QueryMemoryTestStore::default());
                store.records.borrow_mut().push(record(
                    "koro-rule",
                    "Koro relaxes when approached slowly from the side.",
                ));
                let (blackboard, caps, memory_caps, lutum) =
                    test_caps_with_adapter(adapter.clone(), store);
                let mut module = build_query_module(&caps, memory_caps).await;
                let owner = ModuleInstanceId::new(builtin::query_memory(), ReplicaIndex::ZERO);
                blackboard
                    .update_typed_cognitive_memo(
                        owner,
                        "Retrieved memory evidence:\nKoro relaxes when approached slowly from the side."
                            .to_owned(),
                        QueryMemoryMemo {
                            requests: vec!["prior request".to_owned()],
                            searches: vec![QueryMemoryMemoSearch {
                                query: "Koro food approach".to_owned(),
                                limit: 8,
                                hit_indices: vec![MemoryIndex::new("koro-rule")],
                            }],
                            hits: vec![QueryMemoryMemoHit {
                                index: MemoryIndex::new("koro-rule"),
                                rank: MemoryRank::Permanent,
                                occurred_at: Some(test_now()),
                                stored_at: test_now(),
                                kind: MemoryKind::Statement,
                                affect_arousal: 0.0,
                                valence: 0.0,
                                emotion: String::new(),
                            }],
                            linked_hits: Vec::new(),
                        },
                        test_now(),
                    )
                    .await;

                run_query_memory_once(&blackboard, &caps, &mut module, &lutum)
                    .await
                    .expect("already memoed evidence should be a successful no-filter activation");

                assert_eq!(adapter.text_inputs().len(), 1);
                assert_eq!(query_memory_memos(&blackboard).await.len(), 1);
            })
            .await;
    }

    #[tokio::test(flavor = "current_thread")]
    async fn flat_hit_selected_by_filter_writes_evidence_only_memo() {
        tokio::task::LocalSet::new()
            .run_until(async {
                let adapter = Arc::new(CapturingAdapter::new(
                    MockLlmAdapter::new()
                        .with_text_scenario(plan_scenario(
                            "search",
                            serde_json::json!([
                                {"query": "Koro food approach", "limit": 8, "include_linked": false}
                            ]),
                        ))
                        .with_text_scenario(filter_scenario(vec!["koro-rule"], Vec::new(), false)),
                ));
                let store = Rc::new(QueryMemoryTestStore::default());
                store.records.borrow_mut().push(record(
                    "koro-rule",
                    "Koro relaxes when approached slowly from the side.",
                ));
                let (blackboard, caps, memory_caps, lutum) =
                    test_caps_with_adapter(adapter.clone(), store);
                let mut module = build_query_module(&caps, memory_caps).await;

                run_query_memory_once(&blackboard, &caps, &mut module, &lutum)
                    .await
                    .expect("flat hit activation should succeed");

                let memos = query_memory_memos(&blackboard).await;
                assert_eq!(memos.len(), 1);
                assert_eq!(memos[0].data().hits[0].index.as_str(), "koro-rule");
                assert!(
                    memos[0]
                        .content
                        .contains("Koro relaxes when approached slowly from the side.")
                );
                assert!(!memos[0].content.contains("How should Nui approach"));

                let inputs = adapter.text_inputs();
                assert_eq!(inputs.len(), 2);
                assert!(
                    inputs[1]
                        .items()
                        .iter()
                        .all(|item| !matches!(item, ModelInputItem::Turn(_)))
                );
                assert!(
                    inputs[1]
                        .items()
                        .iter()
                        .all(|item| !matches!(item, ModelInputItem::ToolResult(_)))
                );
            })
            .await;
    }

    #[tokio::test(flavor = "current_thread")]
    async fn linked_hit_is_fetched_before_filter_and_can_be_selected() {
        tokio::task::LocalSet::new()
            .run_until(async {
                let adapter = Arc::new(CapturingAdapter::new(
                    MockLlmAdapter::new()
                        .with_text_scenario(plan_scenario(
                            "search",
                            serde_json::json!([
                                {"query": "Koro food approach", "limit": 8, "include_linked": true}
                            ]),
                        ))
                        .with_text_scenario(filter_scenario(
                            vec!["koro-primary"],
                            vec!["koro-linked"],
                            false,
                        )),
                ));
                let store = Rc::new(QueryMemoryTestStore::default());
                let primary = record(
                    "koro-primary",
                    "Koro guards food, but relaxes when approached slowly from the side.",
                );
                let linked = record(
                    "koro-linked",
                    "Pibi coached Nui to use short clear signals before touch.",
                );
                store.records.borrow_mut().push(primary.clone());
                store.linked_records.borrow_mut().push(LinkedMemoryRecord {
                    record: linked,
                    link: MemoryLink {
                        from_memory: primary.index.clone(),
                        to_memory: MemoryIndex::new("koro-linked"),
                        relation: MemoryLinkRelation::Supports,
                        freeform_relation: None,
                        strength: 1.0,
                        confidence: 1.0,
                        updated_at: test_now(),
                    },
                });
                let (blackboard, caps, memory_caps, lutum) =
                    test_caps_with_adapter(adapter.clone(), store);
                let mut module = build_query_module(&caps, memory_caps).await;

                run_query_memory_once(&blackboard, &caps, &mut module, &lutum)
                    .await
                    .expect("linked hit activation should succeed");

                let memos = query_memory_memos(&blackboard).await;
                assert_eq!(memos.len(), 1);
                assert_eq!(memos[0].data().linked_hits[0].index.as_str(), "koro-linked");
                assert!(
                    memos[0]
                        .content
                        .contains("Pibi coached Nui to use short clear signals")
                );
            })
            .await;
    }

    #[tokio::test(flavor = "current_thread")]
    async fn filter_reject_all_persists_no_memo() {
        tokio::task::LocalSet::new()
            .run_until(async {
                let adapter = Arc::new(CapturingAdapter::new(
                    MockLlmAdapter::new()
                        .with_text_scenario(plan_scenario(
                            "search",
                            serde_json::json!([
                                {"query": "Koro food approach", "limit": 8, "include_linked": false}
                            ]),
                        ))
                        .with_text_scenario(filter_scenario(Vec::new(), Vec::new(), true)),
                ));
                let store = Rc::new(QueryMemoryTestStore::default());
                store.records.borrow_mut().push(record(
                    "koro-rule",
                    "Koro relaxes when approached slowly from the side.",
                ));
                let (blackboard, caps, memory_caps, lutum) =
                    test_caps_with_adapter(adapter.clone(), store);
                let mut module = build_query_module(&caps, memory_caps).await;

                run_query_memory_once(&blackboard, &caps, &mut module, &lutum)
                    .await
                    .expect("reject-all activation should succeed");

                assert_eq!(adapter.text_inputs().len(), 2);
                assert!(query_memory_memos(&blackboard).await.is_empty());
            })
            .await;
    }

    #[tokio::test(flavor = "current_thread")]
    async fn missing_filter_tool_call_returns_activation_error() {
        tokio::task::LocalSet::new()
            .run_until(async {
                let adapter = Arc::new(CapturingAdapter::new(
                    MockLlmAdapter::new()
                        .with_text_scenario(plan_scenario(
                            "search",
                            serde_json::json!([
                                {"query": "Koro food approach", "limit": 8, "include_linked": false}
                            ]),
                        ))
                        .with_text_scenario(text_scenario("plain answer")),
                ));
                let store = Rc::new(QueryMemoryTestStore::default());
                store.records.borrow_mut().push(record(
                    "koro-rule",
                    "Koro relaxes when approached slowly from the side.",
                ));
                let (blackboard, caps, memory_caps, lutum) =
                    test_caps_with_adapter(adapter.clone(), store);
                let mut module = build_query_module(&caps, memory_caps).await;

                let err = run_query_memory_once(&blackboard, &caps, &mut module, &lutum)
                    .await
                    .expect_err("missing filter tool should fail activation");
                assert!(err.to_string().contains("select memory evidence"));
                assert!(query_memory_memos(&blackboard).await.is_empty());
            })
            .await;
    }

    #[tokio::test(flavor = "current_thread")]
    async fn multiple_planner_tool_calls_return_activation_error() {
        tokio::task::LocalSet::new()
            .run_until(async {
                let adapter = Arc::new(CapturingAdapter::new(
                    MockLlmAdapter::new().with_text_scenario(MockTextScenario::events(vec![
                        Ok(RawTextTurnEvent::Started {
                            request_id: Some("query-memory-plan".into()),
                            model: "mock".into(),
                        }),
                        Ok(RawTextTurnEvent::ToolCallChunk {
                            id: "call-plan-a".into(),
                            name: "plan_memory_queries".into(),
                            arguments_json_delta: serde_json::json!({
                                "disposition": "search",
                                "searches": [{"query": "Koro", "limit": 8, "include_linked": false}]
                            })
                            .to_string(),
                        }),
                        Ok(RawTextTurnEvent::ToolCallChunk {
                            id: "call-plan-b".into(),
                            name: "plan_memory_queries".into(),
                            arguments_json_delta: serde_json::json!({
                                "disposition": "search",
                                "searches": [{"query": "food", "limit": 8, "include_linked": false}]
                            })
                            .to_string(),
                        }),
                        Ok(RawTextTurnEvent::Completed {
                            request_id: Some("query-memory-plan".into()),
                            finish_reason: FinishReason::ToolCall,
                            usage: usage(),
                        }),
                    ])),
                ));
                let store = Rc::new(QueryMemoryTestStore::default());
                let (blackboard, caps, memory_caps, lutum) = test_caps_with_adapter(adapter, store);
                let mut module = build_query_module(&caps, memory_caps).await;

                let err = run_query_memory_once(&blackboard, &caps, &mut module, &lutum)
                    .await
                    .expect_err("multiple planner tool calls should fail activation");
                assert!(err.to_string().contains("exactly once"));
                assert!(query_memory_memos(&blackboard).await.is_empty());
            })
            .await;
    }

    #[tokio::test(flavor = "current_thread")]
    async fn unknown_filter_index_returns_activation_error() {
        tokio::task::LocalSet::new()
            .run_until(async {
                let adapter = Arc::new(CapturingAdapter::new(
                    MockLlmAdapter::new()
                        .with_text_scenario(plan_scenario(
                            "search",
                            serde_json::json!([
                                {"query": "Koro food approach", "limit": 8, "include_linked": false}
                            ]),
                        ))
                        .with_text_scenario(filter_scenario(
                            vec!["unknown-index"],
                            Vec::new(),
                            false,
                        )),
                ));
                let store = Rc::new(QueryMemoryTestStore::default());
                store.records.borrow_mut().push(record(
                    "koro-rule",
                    "Koro relaxes when approached slowly from the side.",
                ));
                let (blackboard, caps, memory_caps, lutum) = test_caps_with_adapter(adapter, store);
                let mut module = build_query_module(&caps, memory_caps).await;

                let err = run_query_memory_once(&blackboard, &caps, &mut module, &lutum)
                    .await
                    .expect_err("unknown evidence index should fail activation");
                let error_chain = err.chain().map(ToString::to_string).collect::<Vec<_>>();
                assert!(
                    error_chain
                        .iter()
                        .any(|error| error.contains("unknown evidence indexes"))
                );
                assert!(query_memory_memos(&blackboard).await.is_empty());
            })
            .await;
    }
}
