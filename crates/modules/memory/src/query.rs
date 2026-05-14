use std::collections::HashSet;

use anyhow::{Context, Result};
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use lutum::{Session, TextStepOutcomeWithTools, ToolResult};
use nuillu_module::{
    AllocationReader, AllocationUpdatedInbox, BlackboardReader, CognitionLogUpdatedInbox,
    LlmAccess, Module, SessionCompactionConfig, SessionCompactionProtectedPrefix, TypedMemo,
    compact_session_if_needed, format_current_attention_guidance, format_memory_trace_inventory,
    memory_rank_counts, push_formatted_memo_log_batch, render_memory_for_llm,
    seed_persistent_faculty_session,
};
use nuillu_types::{MemoryIndex, MemoryRank};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::store::{
    LinkedMemoryQuery, LinkedMemoryRecord, MemoryConcept, MemoryContentReader, MemoryKind,
    MemoryLink, MemoryLinkDirection, MemoryLinkRelation, MemoryTag, VectorMemorySearcher,
};

const SYSTEM_PROMPT: &str = r#"You are the query-vector module.
Retrieve memory evidence. Memory is not a fact table and retrieval must not decide current truth.
Use search_vector_memory for ordinary flat memory search. Use fetch_linked_memories only after a
specific search hit or prior memo makes linked context useful. Linked lookup is explicit; ordinary
search results are flat and do not include hidden bundles.
If the question contains allocation guidance or a speak-gate evidence request, search for the concrete
requested facts, proper nouns, species/body/peer/world terms, route rules, and the needed_fact
phrases. Do not search for generic phrases such as "useful memory context" when a concrete guidance
question is available.
You may call search_vector_memory multiple times in the same turn when the input contains multiple
distinct questions or evidence requests. Prefer multiple targeted searches in one turn over broad
generic searches or later follow-up turns.
If the requested facts are already covered by prior query-vector memo logs, previous tool results,
or the cognition log, finish without calling tools; no memo will be written for a no-op turn.
You may summarize conflict or link structure only through retrieved tool evidence in the memo. Do not
produce user-facing answers, decide final truth, explain results from outside tool output, or use a
final answer as a data channel."#;

const COMPACTED_QUERY_VECTOR_SESSION_PREFIX: &str = "Compacted query-vector session history:";
const SESSION_COMPACTION_PROMPT: &str = r#"You compact the query-vector module's persistent session history.
Summarize only the prefix transcript you receive. Preserve memo-log facts, query requests, vector
search arguments, useful memory hits, rejected broad searches, and allocation/cognition context that
future retrieval should remember. Do not invent facts. Return plain text only."#;

fn format_vector_memory_context(
    rank_counts: &nuillu_module::MemoryRankCounts,
    allocation: &nuillu_module::ResourceAllocation,
) -> String {
    let mut sections = vec!["Vector-memory retrieval context:".to_owned()];
    if let Some(section) = format_memory_trace_inventory(rank_counts) {
        sections.push(section);
    }
    if let Some(section) = format_current_attention_guidance(allocation) {
        sections.push(section);
    }
    sections.join("\n\n")
}

#[lutum::tool_input(name = "search_vector_memory", output = SearchVectorMemoryOutput)]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct SearchVectorMemoryArgs {
    pub query: String,
    pub limit: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct SearchVectorMemoryOutput {
    pub hits: Vec<QueryVectorMemoryHit>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct QueryVectorMemoryHit {
    pub index: MemoryIndex,
    pub content: String,
    pub rank: MemoryRank,
    pub occurred_at: Option<DateTime<Utc>>,
    pub stored_at: DateTime<Utc>,
    pub kind: MemoryKind,
    pub concepts: Vec<MemoryConcept>,
    pub tags: Vec<MemoryTag>,
}

#[lutum::tool_input(name = "fetch_linked_memories", output = FetchLinkedMemoriesOutput)]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct FetchLinkedMemoriesArgs {
    pub memory_indexes: Vec<MemoryIndex>,
    pub relation_filter: Vec<MemoryLinkRelation>,
    pub direction: MemoryLinkDirection,
    pub limit: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct FetchLinkedMemoriesOutput {
    pub hits: Vec<QueryVectorLinkedMemoryHit>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct QueryVectorLinkedMemoryHit {
    pub index: MemoryIndex,
    pub content: String,
    pub rank: MemoryRank,
    pub occurred_at: Option<DateTime<Utc>>,
    pub stored_at: DateTime<Utc>,
    pub kind: MemoryKind,
    pub concepts: Vec<MemoryConcept>,
    pub tags: Vec<MemoryTag>,
    pub link: MemoryLink,
}

#[derive(Clone, Debug, PartialEq)]
pub struct QueryVectorMemo {
    pub requests: Vec<String>,
    pub searches: Vec<QueryVectorMemoSearch>,
    pub hits: Vec<QueryVectorMemoHit>,
    pub linked_hits: Vec<QueryVectorMemoLinkedHit>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct QueryVectorMemoSearch {
    pub query: String,
    pub limit: usize,
    pub hit_indices: Vec<MemoryIndex>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct QueryVectorMemoHit {
    pub index: MemoryIndex,
    pub rank: MemoryRank,
    pub occurred_at: Option<DateTime<Utc>>,
    pub stored_at: DateTime<Utc>,
    pub kind: MemoryKind,
}

#[derive(Clone, Debug, PartialEq)]
pub struct QueryVectorMemoLinkedHit {
    pub index: MemoryIndex,
    pub rank: MemoryRank,
    pub occurred_at: Option<DateTime<Utc>>,
    pub stored_at: DateTime<Utc>,
    pub kind: MemoryKind,
    pub link: MemoryLink,
}

#[derive(Debug, Default)]
struct QueryVectorRetrieval {
    searches: Vec<QueryVectorMemoSearch>,
    hits: Vec<QueryVectorMemoryHit>,
    linked_hits: Vec<QueryVectorLinkedMemoryHit>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema, lutum::Toolset)]
pub enum QueryVectorTools {
    SearchVectorMemory(SearchVectorMemoryArgs),
    FetchLinkedMemories(FetchLinkedMemoriesArgs),
}

pub struct QueryVectorModule {
    owner: nuillu_types::ModuleId,
    allocation_updates: AllocationUpdatedInbox,
    cognition_updates: CognitionLogUpdatedInbox,
    allocation: AllocationReader,
    blackboard: BlackboardReader,
    memory: VectorMemorySearcher,
    linked_memory: MemoryContentReader,
    memo: TypedMemo<QueryVectorMemo>,
    llm: LlmAccess,
    session: Session,
    session_seeded: bool,
    session_compaction: SessionCompactionConfig,
    system_prompt: std::sync::OnceLock<String>,
}

impl QueryVectorModule {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        allocation_updates: AllocationUpdatedInbox,
        cognition_updates: CognitionLogUpdatedInbox,
        allocation: AllocationReader,
        blackboard: BlackboardReader,
        memory: VectorMemorySearcher,
        linked_memory: MemoryContentReader,
        memo: TypedMemo<QueryVectorMemo>,
        llm: LlmAccess,
    ) -> Self {
        Self {
            owner: nuillu_types::ModuleId::new(<Self as Module>::id())
                .expect("query-vector id is valid"),
            allocation_updates,
            cognition_updates,
            allocation,
            blackboard,
            memory,
            linked_memory,
            memo,
            llm,
            session: Session::new(),
            session_seeded: false,
            session_compaction: SessionCompactionConfig::default(),
            system_prompt: std::sync::OnceLock::new(),
        }
    }

    fn system_prompt(&self, cx: &nuillu_module::ActivateCx<'_>) -> &str {
        self.system_prompt.get_or_init(|| {
            nuillu_module::format_faculty_system_prompt(SYSTEM_PROMPT, cx.modules(), &self.owner)
        })
    }

    fn ensure_session_seeded(&mut self, cx: &nuillu_module::ActivateCx<'_>) {
        if self.session_seeded {
            return;
        }
        let system_prompt = self.system_prompt(cx).to_owned();
        seed_persistent_faculty_session(
            &mut self.session,
            system_prompt,
            cx.identity_memories(),
            cx.now(),
        );
        self.session_seeded = true;
    }

    #[tracing::instrument(skip_all, err(Debug, level = "warn"))]
    async fn activate_allocation_guidance(
        &mut self,
        cx: &nuillu_module::ActivateCx<'_>,
    ) -> Result<()> {
        let guidance = self
            .allocation
            .snapshot()
            .await
            .for_module(&self.owner)
            .guidance;
        let questions = guidance_questions(&guidance);
        if questions.is_empty() {
            return Ok(());
        }
        let retrieval = self.search_with_memory(cx, &questions).await?;
        self.write_retrieval(&questions, retrieval).await
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
        let retrieval = self.search_with_memory(cx, &questions).await?;
        self.write_retrieval(&questions, retrieval).await
    }

    async fn write_retrieval(
        &self,
        requests: &[String],
        retrieval: QueryVectorRetrieval,
    ) -> Result<()> {
        let hits = self.fresh_hits(&retrieval.hits).await;
        let linked_hits = self.fresh_linked_hits(&retrieval.linked_hits).await;
        let content = render_memo(requests, &retrieval.searches, &hits, &linked_hits);
        if !content.is_empty() {
            let payload = QueryVectorMemo {
                requests: requests.to_vec(),
                searches: retrieval.searches,
                hits: hits
                    .iter()
                    .map(|hit| QueryVectorMemoHit {
                        index: hit.index.clone(),
                        rank: hit.rank,
                        occurred_at: hit.occurred_at,
                        stored_at: hit.stored_at,
                        kind: hit.kind,
                    })
                    .collect(),
                linked_hits: linked_hits
                    .iter()
                    .map(|hit| QueryVectorMemoLinkedHit {
                        index: hit.index.clone(),
                        rank: hit.rank,
                        occurred_at: hit.occurred_at,
                        stored_at: hit.stored_at,
                        kind: hit.kind,
                        link: hit.link.clone(),
                    })
                    .collect(),
            };
            self.memo.write(payload, content).await;
        }
        Ok(())
    }

    async fn fresh_hits(&self, hits: &[QueryVectorMemoryHit]) -> Vec<QueryVectorMemoryHit> {
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

    async fn fresh_linked_hits(
        &self,
        hits: &[QueryVectorLinkedMemoryHit],
    ) -> Vec<QueryVectorLinkedMemoryHit> {
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
    ) -> Result<QueryVectorRetrieval> {
        self.ensure_session_seeded(cx);
        let unread_memo_logs = self.blackboard.unread_memo_logs().await;
        push_formatted_memo_log_batch(&mut self.session, &unread_memo_logs, cx.now());
        let prior_query_vector_searches = self.prior_query_vector_searches().await;
        let rank_counts = self
            .blackboard
            .read(|bb| memory_rank_counts(bb.memory_metadata()))
            .await;
        let allocation = self.allocation.snapshot().await;
        self.session
            .push_user(format_vector_memory_questions(questions));
        if let Some(prior) = prior_query_vector_searches {
            self.session.push_ephemeral_user(prior);
        }
        self.session
            .push_ephemeral_system(format_vector_memory_context(&rank_counts, &allocation));

        let mut retrieval = QueryVectorRetrieval::default();
        for _ in 0..3 {
            let lutum = self.llm.lutum().await;
            let outcome = self
                .session
                .text_turn(&lutum)
                .tools::<QueryVectorTools>()
                .available_tools([
                    QueryVectorToolsSelector::SearchVectorMemory,
                    QueryVectorToolsSelector::FetchLinkedMemories,
                ])
                .collect()
                .await
                .context("query-vector text turn failed")?;

            match outcome {
                TextStepOutcomeWithTools::Finished(result) => {
                    compact_session_if_needed(
                        &mut self.session,
                        result.usage.input_tokens,
                        cx.session_compaction(),
                        self.session_compaction,
                        SessionCompactionProtectedPrefix::LeadingSystemAndIdentitySeed,
                        Self::id(),
                        COMPACTED_QUERY_VECTOR_SESSION_PREFIX,
                        SESSION_COMPACTION_PROMPT,
                    )
                    .await;
                    return Ok(retrieval);
                }
                TextStepOutcomeWithTools::FinishedNoOutput(result) => {
                    compact_session_if_needed(
                        &mut self.session,
                        result.usage.input_tokens,
                        cx.session_compaction(),
                        self.session_compaction,
                        SessionCompactionProtectedPrefix::LeadingSystemAndIdentitySeed,
                        Self::id(),
                        COMPACTED_QUERY_VECTOR_SESSION_PREFIX,
                        SESSION_COMPACTION_PROMPT,
                    )
                    .await;
                    return Ok(retrieval);
                }
                TextStepOutcomeWithTools::NeedsTools(round) => {
                    let input_tokens = round.usage.input_tokens;
                    if round.tool_calls.is_empty() {
                        compact_session_if_needed(
                            &mut self.session,
                            input_tokens,
                            cx.session_compaction(),
                            self.session_compaction,
                            SessionCompactionProtectedPrefix::LeadingSystemAndIdentitySeed,
                            Self::id(),
                            COMPACTED_QUERY_VECTOR_SESSION_PREFIX,
                            SESSION_COMPACTION_PROMPT,
                        )
                        .await;
                        return Ok(retrieval);
                    }
                    let mut tool_results: Vec<ToolResult> = Vec::new();
                    for call in round.tool_calls.iter().cloned() {
                        match call {
                            QueryVectorToolsCall::SearchVectorMemory(call) => {
                                let input = call.input.clone();
                                let output = self
                                    .search_vector_memory(input.clone(), cx.now())
                                    .await
                                    .context("run search_vector_memory tool")?;
                                retrieval.searches.push(QueryVectorMemoSearch {
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
                                        .context("complete search_vector_memory tool call")?,
                                );
                            }
                            QueryVectorToolsCall::FetchLinkedMemories(call) => {
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
                        }
                    }
                    round
                        .commit(&mut self.session, tool_results)
                        .context("commit query-vector tool round")?;
                    compact_session_if_needed(
                        &mut self.session,
                        input_tokens,
                        cx.session_compaction(),
                        self.session_compaction,
                        SessionCompactionProtectedPrefix::LeadingSystemAndIdentitySeed,
                        Self::id(),
                        COMPACTED_QUERY_VECTOR_SESSION_PREFIX,
                        SESSION_COMPACTION_PROMPT,
                    )
                    .await;
                }
            }
        }
        Ok(retrieval)
    }

    async fn prior_query_vector_searches(&self) -> Option<String> {
        let records = self.memo.recent_logs().await;
        if records.is_empty() {
            return None;
        }
        let mut out = String::from("Recent query-vector searches already attempted:");
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

    async fn search_vector_memory(
        &self,
        args: SearchVectorMemoryArgs,
        now: DateTime<Utc>,
    ) -> Result<SearchVectorMemoryOutput> {
        let limit = args.limit.clamp(1, 16);
        let records = self
            .memory
            .search(&args.query, limit)
            .await
            .context("search vector memory")?;
        Ok(SearchVectorMemoryOutput {
            hits: records
                .into_iter()
                .map(|record| QueryVectorMemoryHit {
                    index: record.index,
                    content: render_memory_for_llm(
                        record.content.as_str(),
                        record.occurred_at,
                        now,
                    ),
                    rank: record.rank,
                    occurred_at: record.occurred_at,
                    stored_at: record.stored_at,
                    kind: record.kind,
                    concepts: record.concepts,
                    tags: record.tags,
                })
                .collect(),
        })
    }

    async fn fetch_linked_memories(
        &self,
        args: FetchLinkedMemoriesArgs,
        now: DateTime<Utc>,
    ) -> Result<FetchLinkedMemoriesOutput> {
        let limit = args.limit.clamp(1, 16);
        let records = self
            .linked_memory
            .linked(&LinkedMemoryQuery {
                memory_indexes: args.memory_indexes,
                relation_filter: args.relation_filter,
                direction: args.direction,
                limit,
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

    async fn next_batch(&mut self) -> Result<QueryVectorBatch> {
        let mut batch = self.await_first_batch().await?;
        self.collect_ready_events_into_batch(&mut batch)?;
        Ok(batch)
    }

    async fn await_first_batch(&mut self) -> Result<QueryVectorBatch> {
        let batch = tokio::select! {
            update = self.allocation_updates.next_item() => {
                let _ = update?;
                QueryVectorBatch::allocation_update()
            }
            update = self.cognition_updates.next_item() => {
                let _ = update?;
                QueryVectorBatch::cognition_log_update()
            }
        };
        Ok(batch)
    }

    fn collect_ready_events_into_batch(&mut self, batch: &mut QueryVectorBatch) -> Result<()> {
        if !self.allocation_updates.take_ready_items()?.items.is_empty() {
            batch.mark_allocation_updated();
        }
        if !self.cognition_updates.take_ready_items()?.items.is_empty() {
            batch.mark_cognition_updated();
        }
        Ok(())
    }
}

#[derive(Debug, Default)]
pub struct QueryVectorBatch {
    pub(crate) allocation_updated: bool,
    pub(crate) cognition_updated: bool,
}

impl QueryVectorBatch {
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

fn hit_contents(hits: &[QueryVectorMemoryHit]) -> String {
    let mut contents = Vec::new();
    for hit in hits {
        let content = hit.content.trim();
        if !content.is_empty() && !contents.contains(&content) {
            contents.push(content);
        }
    }
    contents.join("\n\n")
}

fn linked_hit_contents(hits: &[QueryVectorLinkedMemoryHit]) -> String {
    let mut contents = Vec::<String>::new();
    for hit in hits {
        let content = hit.content.trim();
        if !content.is_empty() && !contents.iter().any(|item| item.ends_with(content)) {
            contents.push(format!(
                "[{:?} {:?} {} -> {}] {content}",
                hit.link.relation,
                hit.link.freeform_relation,
                hit.link.from_memory,
                hit.link.to_memory
            ));
        }
    }
    contents.join("\n\n")
}

fn render_memo(
    requests: &[String],
    searches: &[QueryVectorMemoSearch],
    hits: &[QueryVectorMemoryHit],
    linked_hits: &[QueryVectorLinkedMemoryHit],
) -> String {
    let retrieved = hit_contents(hits);
    let linked = linked_hit_contents(linked_hits);
    if retrieved.is_empty() && linked.is_empty() {
        return String::new();
    }

    let mut sections = Vec::new();
    let request_lines = bullet_lines(requests.iter().map(String::as_str));
    if !request_lines.is_empty() {
        sections.push(format!("Query intent:\n{request_lines}"));
    }
    let search_lines = bullet_lines(searches.iter().map(|search| search.query.as_str()));
    if !search_lines.is_empty() {
        sections.push(format!("Search queries:\n{search_lines}"));
    }
    if !retrieved.is_empty() {
        sections.push(format!("Retrieved memory evidence:\n{retrieved}"));
    }
    if !linked.is_empty() {
        sections.push(format!("Linked memory evidence:\n{linked}"));
    }
    sections.join("\n\n")
}

fn render_linked_hit(linked: LinkedMemoryRecord, now: DateTime<Utc>) -> QueryVectorLinkedMemoryHit {
    let record = linked.record;
    QueryVectorLinkedMemoryHit {
        index: record.index,
        content: render_memory_for_llm(record.content.as_str(), record.occurred_at, now),
        rank: record.rank,
        occurred_at: record.occurred_at,
        stored_at: record.stored_at,
        kind: record.kind,
        concepts: record.concepts,
        tags: record.tags,
        link: linked.link,
    }
}

fn bullet_lines<'a>(items: impl Iterator<Item = &'a str>) -> String {
    items
        .map(str::trim)
        .filter(|item| !item.is_empty())
        .map(|item| format!("- {item}"))
        .collect::<Vec<_>>()
        .join("\n")
}

fn format_vector_memory_questions(questions: &[String]) -> String {
    let mut out = String::from("Vector-memory questions to investigate:");
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
impl Module for QueryVectorModule {
    type Batch = QueryVectorBatch;

    fn id() -> &'static str {
        "query-vector"
    }

    fn role_description() -> &'static str {
        "Recalls stored memories by semantic similarity when evidence is not already available: writes query intent and fresh memory evidence to its memo log from allocation guidance or cognition-log updates; cognition-gate must promote useful hits before speech uses them; never synthesizes answers."
    }

    async fn next_batch(&mut self) -> Result<Self::Batch> {
        QueryVectorModule::next_batch(self).await
    }

    async fn activate(
        &mut self,
        cx: &nuillu_module::ActivateCx<'_>,
        batch: &Self::Batch,
    ) -> Result<()> {
        if batch.allocation_updated {
            self.activate_allocation_guidance(cx).await?;
        }
        if batch.cognition_updated {
            self.activate_cognition_update(cx).await?;
        }
        Ok(())
    }
}

fn guidance_questions(guidance: &str) -> Vec<String> {
    let trimmed = guidance.trim();
    if trimmed.is_empty() {
        Vec::new()
    } else {
        vec![trimmed.to_owned()]
    }
}
