use std::collections::HashSet;

use anyhow::{Context, Result};
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use lutum::{Session, TextStepOutcomeWithTools, ToolResult};
use nuillu_module::{
    AllocationReader, AllocationUpdatedInbox, BlackboardReader, CognitionLogUpdatedInbox,
    LlmAccess, Module, SessionCompactionConfig, TypedMemo, VectorMemorySearcher,
    compact_session_if_needed, format_current_attention_guidance, format_memory_trace_inventory,
    memory_rank_counts, push_formatted_memo_log_batch, render_memory_for_llm,
};
use nuillu_types::{MemoryIndex, MemoryRank};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

mod batch;
pub use batch::NextBatch as QueryVectorBatch;

const SYSTEM_PROMPT: &str = r#"You are the query-vector module.
Choose vector-memory searches only. Use search_vector_memory for factual memory lookup when needed,
then stop.
If the question contains allocation guidance or a speak-gate evidence request, search for the concrete
requested facts, proper nouns, species/body/peer/world terms, route rules, and the needed_fact
phrases. Do not search for generic phrases such as "useful memory context" when a concrete guidance
question is available.
You may call search_vector_memory multiple times in the same turn when the input contains multiple
distinct questions or evidence requests. Prefer multiple targeted searches in one turn over broad
generic searches or later follow-up turns.
If the requested facts are already covered by prior query-vector memo logs, previous tool results,
or the cognition log, finish without calling tools; no memo will be written for a no-op turn.
Do not answer questions, explain results, describe this module, or add any text from outside tool
results. The runtime memoizes only memory hit content returned by tools. Any final text is ignored;
do not use a final answer as a data channel."#;

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

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct QueryVectorMemoryHit {
    pub index: MemoryIndex,
    pub content: String,
    pub rank: MemoryRank,
    pub occurred_at: Option<DateTime<Utc>>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct QueryVectorMemo {
    pub requests: Vec<String>,
    pub searches: Vec<QueryVectorMemoSearch>,
    pub hits: Vec<QueryVectorMemoHit>,
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
}

#[derive(Debug, Default)]
struct QueryVectorRetrieval {
    searches: Vec<QueryVectorMemoSearch>,
    hits: Vec<QueryVectorMemoryHit>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema, lutum::Toolset)]
pub enum QueryVectorTools {
    SearchVectorMemory(SearchVectorMemoryArgs),
}

pub struct QueryVectorModule {
    owner: nuillu_types::ModuleId,
    allocation_updates: AllocationUpdatedInbox,
    cognition_updates: CognitionLogUpdatedInbox,
    allocation: AllocationReader,
    blackboard: BlackboardReader,
    memory: VectorMemorySearcher,
    memo: TypedMemo<QueryVectorMemo>,
    llm: LlmAccess,
    session: Session,
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
            memo,
            llm,
            session: Session::new(),
            session_compaction: SessionCompactionConfig::default(),
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
                cx.now(),
            )
        })
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
        let content = render_memo(requests, &retrieval.searches, &hits);
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

    async fn search_with_memory(
        &mut self,
        cx: &nuillu_module::ActivateCx<'_>,
        questions: &[String],
    ) -> Result<QueryVectorRetrieval> {
        let unread_memo_logs = self.blackboard.unread_memo_logs().await;
        push_formatted_memo_log_batch(&mut self.session, &unread_memo_logs, cx.now());
        let prior_query_vector_searches = self.prior_query_vector_searches().await;
        let rank_counts = self
            .blackboard
            .read(|bb| memory_rank_counts(bb.memory_metadata()))
            .await;
        let allocation = self.allocation.snapshot().await;
        let system_prompt = self.system_prompt(cx).to_owned();
        self.session.push_ephemeral_system(system_prompt);
        self.session
            .push_user(format_vector_memory_questions(questions));
        if let Some(prior) = prior_query_vector_searches {
            self.session.push_ephemeral_user(prior);
        }
        self.session
            .push_ephemeral_system(format_vector_memory_context(&rank_counts, &allocation));

        let lutum = self.llm.lutum().await;
        let outcome = self
            .session
            .text_turn(&lutum)
            .tools::<QueryVectorTools>()
            .available_tools([QueryVectorToolsSelector::SearchVectorMemory])
            .collect()
            .await
            .context("query-vector text turn failed")?;

        match outcome {
            TextStepOutcomeWithTools::Finished(result) => {
                compact_session_if_needed(
                    &mut self.session,
                    result.usage.input_tokens,
                    cx.session_compaction_lutum(),
                    self.session_compaction,
                    Self::id(),
                    COMPACTED_QUERY_VECTOR_SESSION_PREFIX,
                    SESSION_COMPACTION_PROMPT,
                )
                .await;
                Ok(QueryVectorRetrieval::default())
            }
            TextStepOutcomeWithTools::FinishedNoOutput(result) => {
                compact_session_if_needed(
                    &mut self.session,
                    result.usage.input_tokens,
                    cx.session_compaction_lutum(),
                    self.session_compaction,
                    Self::id(),
                    COMPACTED_QUERY_VECTOR_SESSION_PREFIX,
                    SESSION_COMPACTION_PROMPT,
                )
                .await;
                Ok(QueryVectorRetrieval::default())
            }
            TextStepOutcomeWithTools::NeedsTools(round) => {
                let input_tokens = round.usage.input_tokens;
                if round.tool_calls.is_empty() {
                    compact_session_if_needed(
                        &mut self.session,
                        input_tokens,
                        cx.session_compaction_lutum(),
                        self.session_compaction,
                        Self::id(),
                        COMPACTED_QUERY_VECTOR_SESSION_PREFIX,
                        SESSION_COMPACTION_PROMPT,
                    )
                    .await;
                    return Ok(QueryVectorRetrieval::default());
                }
                let mut all_hits = Vec::new();
                let mut searches = Vec::new();
                let mut tool_results: Vec<ToolResult> = Vec::new();
                for call in round.tool_calls.iter().cloned() {
                    let QueryVectorToolsCall::SearchVectorMemory(call) = call;
                    let input = call.input.clone();
                    let output = self
                        .search_vector_memory(input.clone(), cx.now())
                        .await
                        .context("run search_vector_memory tool")?;
                    searches.push(QueryVectorMemoSearch {
                        query: input.query,
                        limit: input.limit.clamp(1, 16),
                        hit_indices: output.hits.iter().map(|hit| hit.index.clone()).collect(),
                    });
                    all_hits.extend(output.hits.clone());
                    tool_results.push(
                        call.complete(output)
                            .context("complete search_vector_memory tool call")?,
                    );
                }
                round
                    .commit(&mut self.session, tool_results)
                    .context("commit query-vector tool round")?;
                compact_session_if_needed(
                    &mut self.session,
                    input_tokens,
                    cx.session_compaction_lutum(),
                    self.session_compaction,
                    Self::id(),
                    COMPACTED_QUERY_VECTOR_SESSION_PREFIX,
                    SESSION_COMPACTION_PROMPT,
                )
                .await;
                Ok(QueryVectorRetrieval {
                    searches,
                    hits: all_hits,
                })
            }
        }
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
                })
                .collect(),
        })
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

fn render_memo(
    requests: &[String],
    searches: &[QueryVectorMemoSearch],
    hits: &[QueryVectorMemoryHit],
) -> String {
    let retrieved = hit_contents(hits);
    if retrieved.is_empty() {
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
    sections.push(format!("Retrieved memories:\n{retrieved}"));
    sections.join("\n\n")
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

#[cfg(test)]
mod tests {
    use super::*;

    use std::sync::{Arc, Mutex};

    use async_trait::async_trait;
    use chrono::TimeZone as _;
    use lutum::{
        FinishReason, Lutum, MockLlmAdapter, MockTextScenario, RawTextTurnEvent,
        SharedPoolBudgetManager, SharedPoolBudgetOptions, Usage,
    };
    use nuillu_blackboard::{
        Blackboard, BlackboardCommand, Bpm, ModuleConfig, ResourceAllocation, linear_ratio_fn,
    };
    use nuillu_module::ports::{
        IndexedMemory, MemoryQuery, MemoryRecord, MemoryStore, NewMemory,
        NoopCognitionLogRepository, NoopFileSearchProvider, NoopUtteranceSink, PortError,
        SystemClock,
    };
    use nuillu_module::{
        ActivateCx, AllocationUpdated, CapabilityProviderPorts, CapabilityProviders, LutumTiers,
        ModuleRegistry,
    };
    use nuillu_types::{MemoryContent, ModuleInstanceId, ReplicaIndex, builtin};

    #[derive(Debug, Default)]
    struct MutableMemoryStore {
        records: Mutex<Vec<MemoryRecord>>,
    }

    impl MutableMemoryStore {
        fn set_records(&self, records: Vec<MemoryRecord>) {
            *self.records.lock().expect("memory records poisoned") = records;
        }
    }

    #[async_trait(?Send)]
    impl MemoryStore for MutableMemoryStore {
        async fn insert(&self, _mem: NewMemory) -> Result<MemoryIndex, PortError> {
            Ok(MemoryIndex::new("unused"))
        }

        async fn put(&self, _mem: IndexedMemory) -> Result<(), PortError> {
            Ok(())
        }

        async fn compact(
            &self,
            _mem: NewMemory,
            _sources: &[MemoryIndex],
        ) -> Result<MemoryIndex, PortError> {
            Ok(MemoryIndex::new("unused"))
        }

        async fn put_compacted(
            &self,
            _mem: IndexedMemory,
            _sources: &[MemoryIndex],
        ) -> Result<(), PortError> {
            Ok(())
        }

        async fn get(&self, _index: &MemoryIndex) -> Result<Option<MemoryRecord>, PortError> {
            Ok(None)
        }

        async fn list_by_rank(&self, _rank: MemoryRank) -> Result<Vec<MemoryRecord>, PortError> {
            Ok(Vec::new())
        }

        async fn search(&self, _q: &MemoryQuery) -> Result<Vec<MemoryRecord>, PortError> {
            Ok(self
                .records
                .lock()
                .expect("memory records poisoned")
                .clone())
        }

        async fn delete(&self, _index: &MemoryIndex) -> Result<(), PortError> {
            Ok(())
        }
    }

    fn tool_scenario(id: &str) -> MockTextScenario {
        MockTextScenario::events(vec![
            Ok(RawTextTurnEvent::Started {
                request_id: Some(id.into()),
                model: "mock".into(),
            }),
            Ok(RawTextTurnEvent::ToolCallChunk {
                id: format!("{id}-tool").into(),
                name: "search_vector_memory".into(),
                arguments_json_delta: r#"{"query":"koro","limit":8}"#.into(),
            }),
            Ok(RawTextTurnEvent::Completed {
                request_id: Some(id.into()),
                finish_reason: FinishReason::ToolCall,
                usage: Usage::zero(),
            }),
        ])
    }

    fn final_text_scenario(id: &str) -> MockTextScenario {
        MockTextScenario::events(vec![
            Ok(RawTextTurnEvent::Started {
                request_id: Some(id.into()),
                model: "mock".into(),
            }),
            Ok(RawTextTurnEvent::TextDelta {
                delta: "already covered".into(),
            }),
            Ok(RawTextTurnEvent::Completed {
                request_id: Some(id.into()),
                finish_reason: FinishReason::Stop,
                usage: Usage::zero(),
            }),
        ])
    }

    fn memory_record(index: &str, content: &str) -> MemoryRecord {
        MemoryRecord {
            index: MemoryIndex::new(index),
            content: MemoryContent::new(content),
            rank: MemoryRank::Permanent,
            occurred_at: None,
        }
    }

    fn memory_record_at(
        index: &str,
        content: &str,
        occurred_at: chrono::DateTime<chrono::Utc>,
    ) -> MemoryRecord {
        MemoryRecord {
            index: MemoryIndex::new(index),
            content: MemoryContent::new(content),
            rank: MemoryRank::Permanent,
            occurred_at: Some(occurred_at),
        }
    }

    fn test_now() -> chrono::DateTime<chrono::Utc> {
        use chrono::TimeZone as _;
        chrono::Utc.with_ymd_and_hms(2026, 5, 10, 0, 0, 0).unwrap()
    }

    fn test_caps(
        blackboard: Blackboard,
        store: Arc<MutableMemoryStore>,
        adapter: MockLlmAdapter,
    ) -> CapabilityProviders {
        let budget = SharedPoolBudgetManager::new(SharedPoolBudgetOptions::default());
        let lutum = Lutum::new(Arc::new(adapter), budget);
        CapabilityProviders::new(CapabilityProviderPorts {
            blackboard,
            cognition_log_port: Arc::new(NoopCognitionLogRepository),
            primary_memory_store: store,
            memory_replicas: Vec::new(),
            file_search: Arc::new(NoopFileSearchProvider),
            utterance_sink: Arc::new(NoopUtteranceSink),
            clock: Arc::new(SystemClock),
            tiers: LutumTiers {
                cheap: lutum.clone(),
                default: lutum.clone(),
                premium: lutum,
            },
        })
    }

    async fn build_query_vector(
        caps: &CapabilityProviders,
    ) -> (
        nuillu_module::AgentRuntimeControl,
        nuillu_module::AllocatedModule,
    ) {
        let modules = ModuleRegistry::new()
            .register(
                nuillu_blackboard::ModulePolicy::new(
                    nuillu_types::ReplicaCapRange::new(1, 1).unwrap(),
                    Bpm::from_f64(60.0)..=Bpm::from_f64(60.0),
                    linear_ratio_fn,
                ),
                |caps| {
                    QueryVectorModule::new(
                        caps.allocation_updated_inbox(),
                        caps.cognition_log_updated_inbox(),
                        caps.allocation_reader(),
                        caps.blackboard_reader(),
                        caps.vector_memory_searcher(),
                        caps.typed_memo::<QueryVectorMemo>(),
                        caps.llm_access(),
                    )
                },
            )
            .unwrap()
            .build(caps)
            .await
            .unwrap();
        let (runtime, mut modules) = modules.into_parts();
        (runtime, modules.pop().expect("query-vector module built"))
    }

    async fn activate_query(
        blackboard: &Blackboard,
        caps: &CapabilityProviders,
        runtime: &nuillu_module::AgentRuntimeControl,
        module: &mut nuillu_module::AllocatedModule,
    ) {
        activate_query_with_question(blackboard, caps, runtime, module, "koro").await;
    }

    async fn activate_query_with_question(
        blackboard: &Blackboard,
        caps: &CapabilityProviders,
        runtime: &nuillu_module::AgentRuntimeControl,
        module: &mut nuillu_module::AllocatedModule,
        question: &str,
    ) {
        let mut allocation = ResourceAllocation::default();
        let mut config = ModuleConfig::default();
        config.guidance = question.into();
        allocation.set(builtin::query_vector(), config);
        blackboard
            .apply(BlackboardCommand::SetAllocation(allocation))
            .await;
        caps.internal_harness_io()
            .allocation_updated_mailbox()
            .publish(AllocationUpdated)
            .await
            .expect("allocation update published");
        let batch = module.next_batch().await.expect("query batch received");
        let catalog = runtime.module_catalog();
        let identity_memories = Vec::new();
        let cx = ActivateCx::new(
            &catalog,
            &identity_memories,
            runtime.session_compaction_lutum().clone(),
            test_now(),
        );
        module.activate(&cx, &batch).await.expect("query activated");
    }

    #[tokio::test(flavor = "current_thread")]
    async fn query_vector_memo_includes_request_search_query_and_fresh_memory() {
        let blackboard = Blackboard::default();
        let store = Arc::new(MutableMemoryStore::default());
        store.set_records(vec![memory_record(
            "mem-1",
            "Koro approaches from the left.",
        )]);
        let adapter = MockLlmAdapter::new().with_text_scenario(tool_scenario("first"));
        let caps = test_caps(blackboard.clone(), store.clone(), adapter);
        let (runtime, mut module) = build_query_vector(&caps).await;

        activate_query_with_question(&blackboard, &caps, &runtime, &mut module, "food bowl clue")
            .await;

        let owner = ModuleInstanceId::new(builtin::query_vector(), ReplicaIndex::ZERO);
        let logs = blackboard.read(|bb| bb.recent_memo_logs()).await;
        assert_eq!(logs.len(), 1);
        assert_eq!(
            logs[0].content,
            "Query intent:\n- food bowl clue\n\nSearch queries:\n- koro\n\nRetrieved memories:\nKoro approaches from the left."
        );

        let typed_logs = blackboard.typed_memo_logs::<QueryVectorMemo>(&owner).await;
        assert_eq!(typed_logs.len(), 1);
        assert_eq!(
            typed_logs[0].data(),
            &QueryVectorMemo {
                requests: vec!["food bowl clue".into()],
                searches: vec![QueryVectorMemoSearch {
                    query: "koro".into(),
                    limit: 8,
                    hit_indices: vec![MemoryIndex::new("mem-1")],
                }],
                hits: vec![QueryVectorMemoHit {
                    index: MemoryIndex::new("mem-1"),
                    rank: MemoryRank::Permanent,
                    occurred_at: None,
                }],
            }
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn query_vector_memo_renders_memory_occurred_at() {
        let blackboard = Blackboard::default();
        let store = Arc::new(MutableMemoryStore::default());
        let occurred_at = chrono::Utc.with_ymd_and_hms(2025, 5, 10, 0, 0, 0).unwrap();
        store.set_records(vec![memory_record_at(
            "mem-1",
            "Koro approaches from the left.",
            occurred_at,
        )]);
        let adapter = MockLlmAdapter::new().with_text_scenario(tool_scenario("first"));
        let caps = test_caps(blackboard.clone(), store.clone(), adapter);
        let (runtime, mut module) = build_query_vector(&caps).await;

        activate_query_with_question(&blackboard, &caps, &runtime, &mut module, "food bowl clue")
            .await;

        let owner = ModuleInstanceId::new(builtin::query_vector(), ReplicaIndex::ZERO);
        let logs = blackboard.read(|bb| bb.recent_memo_logs()).await;
        assert!(logs[0].content.contains(
            "<one-year-ago occurred-at=\"2025-05-10T00:00:00Z\">\nKoro approaches from the left.\n</one-year-ago>"
        ));
        let typed_logs = blackboard.typed_memo_logs::<QueryVectorMemo>(&owner).await;
        assert_eq!(typed_logs[0].data().hits[0].occurred_at, Some(occurred_at));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn repeated_equivalent_query_can_finish_without_tool_and_writes_no_memo() {
        let blackboard = Blackboard::default();
        let store = Arc::new(MutableMemoryStore::default());
        store.set_records(vec![memory_record(
            "mem-1",
            "Koro approaches from the left.",
        )]);
        let adapter = MockLlmAdapter::new()
            .with_text_scenario(tool_scenario("first"))
            .with_text_scenario(final_text_scenario("second"));
        let caps = test_caps(blackboard.clone(), store.clone(), adapter);
        let (runtime, mut module) = build_query_vector(&caps).await;

        activate_query(&blackboard, &caps, &runtime, &mut module).await;
        activate_query(&blackboard, &caps, &runtime, &mut module).await;

        let logs = blackboard.read(|bb| bb.recent_memo_logs()).await;
        assert_eq!(logs.len(), 1);
        assert!(logs[0].content.contains("Koro approaches from the left."));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn search_returning_only_previously_memoized_hits_writes_no_memo() {
        let blackboard = Blackboard::default();
        let store = Arc::new(MutableMemoryStore::default());
        store.set_records(vec![memory_record(
            "mem-1",
            "Koro approaches from the left.",
        )]);
        let adapter = MockLlmAdapter::new()
            .with_text_scenario(tool_scenario("first"))
            .with_text_scenario(tool_scenario("second"));
        let caps = test_caps(blackboard.clone(), store.clone(), adapter);
        let (runtime, mut module) = build_query_vector(&caps).await;

        activate_query(&blackboard, &caps, &runtime, &mut module).await;
        activate_query(&blackboard, &caps, &runtime, &mut module).await;

        let owner = ModuleInstanceId::new(builtin::query_vector(), ReplicaIndex::ZERO);
        let logs = blackboard.read(|bb| bb.recent_memo_logs()).await;
        assert_eq!(logs.len(), 1);
        assert!(logs[0].content.contains("Koro approaches from the left."));

        let typed_logs = blackboard.typed_memo_logs::<QueryVectorMemo>(&owner).await;
        assert_eq!(typed_logs.len(), 1);
        assert_eq!(
            typed_logs[0].data(),
            &QueryVectorMemo {
                requests: vec!["koro".into()],
                searches: vec![QueryVectorMemoSearch {
                    query: "koro".into(),
                    limit: 8,
                    hit_indices: vec![MemoryIndex::new("mem-1")],
                }],
                hits: vec![QueryVectorMemoHit {
                    index: MemoryIndex::new("mem-1"),
                    rank: MemoryRank::Permanent,
                    occurred_at: None,
                }],
            }
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn query_vector_writes_only_fresh_hits_from_mixed_results() {
        let blackboard = Blackboard::default();
        let store = Arc::new(MutableMemoryStore::default());
        store.set_records(vec![memory_record(
            "mem-1",
            "Koro approaches from the left.",
        )]);
        let adapter = MockLlmAdapter::new()
            .with_text_scenario(tool_scenario("first"))
            .with_text_scenario(tool_scenario("second"));
        let caps = test_caps(blackboard.clone(), store.clone(), adapter);
        let (runtime, mut module) = build_query_vector(&caps).await;

        activate_query(&blackboard, &caps, &runtime, &mut module).await;
        store.set_records(vec![
            memory_record("mem-1", "Koro approaches from the left."),
            memory_record("mem-2", "Koro relaxes when the route is clear."),
        ]);
        activate_query(&blackboard, &caps, &runtime, &mut module).await;

        let owner = ModuleInstanceId::new(builtin::query_vector(), ReplicaIndex::ZERO);
        let logs = blackboard.read(|bb| bb.recent_memo_logs()).await;
        assert_eq!(logs.len(), 2);
        assert!(logs[0].content.contains("Koro approaches from the left."));
        assert!(logs[1].content.contains("Query intent:\n- koro"));
        assert!(logs[1].content.contains("Search queries:\n- koro"));
        assert!(
            logs[1]
                .content
                .contains("Koro relaxes when the route is clear.")
        );
        assert!(!logs[1].content.contains("Koro approaches from the left."));

        let typed_logs = blackboard.typed_memo_logs::<QueryVectorMemo>(&owner).await;
        assert_eq!(typed_logs.len(), 2);
        assert_eq!(
            typed_logs[1].data(),
            &QueryVectorMemo {
                requests: vec!["koro".into()],
                searches: vec![QueryVectorMemoSearch {
                    query: "koro".into(),
                    limit: 8,
                    hit_indices: vec![MemoryIndex::new("mem-1"), MemoryIndex::new("mem-2")],
                }],
                hits: vec![QueryVectorMemoHit {
                    index: MemoryIndex::new("mem-2"),
                    rank: MemoryRank::Permanent,
                    occurred_at: None,
                }],
            }
        );
    }
}
