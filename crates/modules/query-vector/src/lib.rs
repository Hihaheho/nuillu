use std::collections::HashSet;

use anyhow::{Context, Result};
use async_trait::async_trait;
use lutum::{Session, TextStepOutcomeWithTools, ToolResult};
use nuillu_module::{
    AllocationReader, BlackboardReader, CognitionLogUpdatedInbox, LlmAccess, Module, QueryInbox,
    QueryRequest, SessionCompactionConfig, TypedMemo, VectorMemorySearcher,
    compact_session_if_needed, push_unread_memo_logs,
};
use nuillu_types::{MemoryIndex, MemoryRank};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

mod batch;
pub use batch::NextBatch as QueryVectorBatch;

const SYSTEM_PROMPT: &str = r#"You are the query-vector module.
Choose vector-memory searches only. Use search_vector_memory for factual memory lookup, then stop.
If the question contains allocation guidance or a speak-gate evidence request, search for the concrete
requested facts, proper nouns, species/body/peer/world terms, route rules, and the needed_fact
phrases. Do not search for generic phrases such as "useful memory context" when a concrete guidance
question is available.
You may call search_vector_memory multiple times in the same turn when the input contains multiple
distinct questions or evidence requests. Prefer multiple targeted searches in one turn over broad
generic searches or later follow-up turns.
Do not answer questions, explain results, describe this module, or add any text from outside tool
results. You must call search_vector_memory. The runtime memoizes only memory hit content returned
by tools. Any final text is ignored; do not use a final answer as a data channel."#;

const COMPACTED_QUERY_VECTOR_SESSION_PREFIX: &str = "Compacted query-vector session history:";
const SESSION_COMPACTION_PROMPT: &str = r#"You compact the query-vector module's persistent session history.
Summarize only the prefix transcript you receive. Preserve memo-log facts, query requests, vector
search arguments, useful memory hits, rejected broad searches, and allocation/cognition context that
future retrieval should remember. Do not invent facts. Return plain text only."#;

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
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct QueryVectorMemo {
    pub hits: Vec<QueryVectorMemoHit>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct QueryVectorMemoHit {
    pub index: MemoryIndex,
    pub rank: MemoryRank,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema, lutum::Toolset)]
pub enum QueryVectorTools {
    SearchVectorMemory(SearchVectorMemoryArgs),
}

pub struct QueryVectorModule {
    owner: nuillu_types::ModuleId,
    query: QueryInbox,
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
        query: QueryInbox,
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
            query,
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
            )
        })
    }

    #[tracing::instrument(skip_all, err(Debug, level = "warn"))]
    async fn handle_queries(
        &mut self,
        cx: &nuillu_module::ActivateCx<'_>,
        requests: Vec<QueryRequest>,
    ) -> Result<()> {
        let questions = requests
            .into_iter()
            .map(|request| request.question)
            .collect::<Vec<_>>();
        if questions.is_empty() {
            return Ok(());
        }
        let hits = self.search_with_memory(cx, &questions).await?;
        self.write_hits(&hits).await
    }

    #[tracing::instrument(skip_all, err(Debug, level = "warn"))]
    async fn activate_cognition_update(
        &mut self,
        cx: &nuillu_module::ActivateCx<'_>,
    ) -> Result<()> {
        let question = self
            .blackboard
            .read(|bb| {
                let entries = bb.cognition_log().entries().to_vec();
                let latest = entries
                    .last()
                    .map(|entry| entry.text.as_str())
                    .unwrap_or("current cognition log");
                format!("Find stable memory that clarifies this cognition-log context: {latest}")
            })
            .await;
        let hits = self.search_with_memory(cx, &[question]).await?;
        self.write_hits(&hits).await
    }

    async fn write_hits(&self, hits: &[QueryVectorMemoryHit]) -> Result<()> {
        let hits = self.fresh_hits(hits).await;
        let content = hit_contents(&hits);
        if !content.is_empty() {
            let payload = QueryVectorMemo {
                hits: hits
                    .iter()
                    .map(|hit| QueryVectorMemoHit {
                        index: hit.index.clone(),
                        rank: hit.rank,
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
    ) -> Result<Vec<QueryVectorMemoryHit>> {
        let unread_memo_logs = self.blackboard.unread_memo_logs().await;
        push_unread_memo_logs(&mut self.session, &unread_memo_logs);
        let snapshot = self
            .blackboard
            .read(|bb| {
                serde_json::json!({
                    "cognition_log": bb.cognition_log().entries(),
                    "memory_metadata": bb.memory_metadata(),
                })
            })
            .await;
        let allocation = self.allocation.snapshot().await;
        let system_prompt = self.system_prompt(cx).to_owned();
        self.session.push_ephemeral_system(system_prompt);
        self.session.push_user(
            serde_json::json!({
                "questions": questions,
            })
            .to_string(),
        );
        self.session.push_ephemeral_user(
            serde_json::json!({
                "blackboard": snapshot,
                "allocation": allocation,
            })
            .to_string(),
        );

        let lutum = self.llm.lutum().await;
        let outcome = self
            .session
            .text_turn(&lutum)
            .tools::<QueryVectorTools>()
            .available_tools([QueryVectorToolsSelector::SearchVectorMemory])
            .require_tool(QueryVectorToolsSelector::SearchVectorMemory)
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
                anyhow::bail!("query-vector completed without calling search_vector_memory");
            }
            TextStepOutcomeWithTools::NeedsTools(round) => {
                let input_tokens = round.usage.input_tokens;
                if round.tool_calls.is_empty() {
                    anyhow::bail!("query-vector produced no valid search_vector_memory tool calls");
                }
                let mut all_hits = Vec::new();
                let mut tool_results: Vec<ToolResult> = Vec::new();
                for call in round.tool_calls.iter().cloned() {
                    let QueryVectorToolsCall::SearchVectorMemory(call) = call;
                    let output = self
                        .search_vector_memory(call.input.clone())
                        .await
                        .context("run search_vector_memory tool")?;
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
                Ok(all_hits)
            }
        }
    }

    async fn search_vector_memory(
        &self,
        args: SearchVectorMemoryArgs,
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
                    content: record.content.as_str().to_owned(),
                    rank: record.rank,
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

#[async_trait(?Send)]
impl Module for QueryVectorModule {
    type Batch = QueryVectorBatch;

    fn id() -> &'static str {
        "query-vector"
    }

    fn role_description() -> &'static str {
        "Recalls stored memories by semantic similarity: writes relevant memory evidence to its memo log on QueryRequest or cognition-log updates; cognition-gate must promote useful hits before speech uses them; never synthesizes answers."
    }

    async fn next_batch(&mut self) -> Result<Self::Batch> {
        QueryVectorModule::next_batch(self).await
    }

    async fn activate(
        &mut self,
        cx: &nuillu_module::ActivateCx<'_>,
        batch: &Self::Batch,
    ) -> Result<()> {
        if !batch.queries.is_empty() {
            self.handle_queries(cx, batch.queries.clone()).await?;
        }
        if batch.cognition_updated {
            self.activate_cognition_update(cx).await?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::sync::{Arc, Mutex};

    use async_trait::async_trait;
    use lutum::{
        FinishReason, Lutum, MockLlmAdapter, MockTextScenario, RawTextTurnEvent,
        SharedPoolBudgetManager, SharedPoolBudgetOptions, Usage,
    };
    use nuillu_blackboard::{Blackboard, Bpm, linear_ratio_fn};
    use nuillu_module::ports::{
        IndexedMemory, MemoryQuery, MemoryRecord, MemoryStore, NewMemory,
        NoopCognitionLogRepository, NoopFileSearchProvider, NoopUtteranceSink, PortError,
        SystemClock,
    };
    use nuillu_module::{
        ActivateCx, CapabilityProviderPorts, CapabilityProviders, LutumTiers, ModuleRegistry,
        QueryRequest,
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

    fn memory_record(index: &str, content: &str) -> MemoryRecord {
        MemoryRecord {
            index: MemoryIndex::new(index),
            content: MemoryContent::new(content),
            rank: MemoryRank::Permanent,
        }
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
                1..=1,
                Bpm::from_f64(60.0)..=Bpm::from_f64(60.0),
                linear_ratio_fn,
                |caps| {
                    QueryVectorModule::new(
                        caps.query_inbox(),
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
        caps: &CapabilityProviders,
        runtime: &nuillu_module::AgentRuntimeControl,
        module: &mut nuillu_module::AllocatedModule,
    ) {
        caps.internal_harness_io()
            .query_mailbox()
            .publish(QueryRequest::new("koro"))
            .await
            .expect("query published");
        let batch = module.next_batch().await.expect("query batch received");
        let catalog = runtime.module_catalog();
        let identity_memories = Vec::new();
        let cx = ActivateCx::new(
            &catalog,
            &identity_memories,
            runtime.session_compaction_lutum(),
        );
        module.activate(&cx, &batch).await.expect("query activated");
    }

    #[tokio::test(flavor = "current_thread")]
    async fn query_vector_does_not_write_repeated_memory_index() {
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

        activate_query(&caps, &runtime, &mut module).await;
        activate_query(&caps, &runtime, &mut module).await;

        let owner = ModuleInstanceId::new(builtin::query_vector(), ReplicaIndex::ZERO);
        let logs = blackboard.read(|bb| bb.recent_memo_logs()).await;
        assert_eq!(logs.len(), 1);
        assert_eq!(logs[0].content, "Koro approaches from the left.");

        let typed_logs = blackboard.typed_memo_logs::<QueryVectorMemo>(&owner).await;
        assert_eq!(typed_logs.len(), 1);
        assert_eq!(
            typed_logs[0].data(),
            &QueryVectorMemo {
                hits: vec![QueryVectorMemoHit {
                    index: MemoryIndex::new("mem-1"),
                    rank: MemoryRank::Permanent,
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

        activate_query(&caps, &runtime, &mut module).await;
        store.set_records(vec![
            memory_record("mem-1", "Koro approaches from the left."),
            memory_record("mem-2", "Koro relaxes when the route is clear."),
        ]);
        activate_query(&caps, &runtime, &mut module).await;

        let owner = ModuleInstanceId::new(builtin::query_vector(), ReplicaIndex::ZERO);
        let logs = blackboard.read(|bb| bb.recent_memo_logs()).await;
        assert_eq!(logs.len(), 2);
        assert_eq!(logs[0].content, "Koro approaches from the left.");
        assert_eq!(logs[1].content, "Koro relaxes when the route is clear.");

        let typed_logs = blackboard.typed_memo_logs::<QueryVectorMemo>(&owner).await;
        assert_eq!(typed_logs.len(), 2);
        assert_eq!(
            typed_logs[1].data(),
            &QueryVectorMemo {
                hits: vec![QueryVectorMemoHit {
                    index: MemoryIndex::new("mem-2"),
                    rank: MemoryRank::Permanent,
                }],
            }
        );
    }
}
