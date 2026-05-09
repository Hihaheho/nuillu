use anyhow::{Context, Result};
use async_trait::async_trait;
use lutum::{Session, StructuredStepOutcomeWithTools, StructuredTurnOutcome, ToolResult};
use nuillu_module::{
    AllocationReader, AttentionStreamUpdatedInbox, BlackboardReader, LlmAccess, Memo, Module,
    QueryInbox, QueryRequest, VectorMemorySearcher,
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
Do not answer questions, explain results, describe this module, or add any text from outside tool
results. You must call search_vector_memory before returning the structured completion. The runtime
memoizes only memory hit content returned by tools. Return only raw JSON for the structured
completion; do not wrap JSON in Markdown or code fences."#;

const MAX_QUERY_ROUNDS: usize = 4;

#[lutum::tool_input(name = "search_vector_memory", output = SearchVectorMemoryOutput)]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct SearchVectorMemoryArgs {
    pub query: String,
    pub limit: usize,
    pub filter_rank: Option<MemoryRank>,
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

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct QueryVectorSearchCompletion {
    pub done: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema, lutum::Toolset)]
pub enum QueryVectorTools {
    SearchVectorMemory(SearchVectorMemoryArgs),
}

pub struct QueryVectorModule {
    owner: nuillu_types::ModuleId,
    query: QueryInbox,
    attention_updates: AttentionStreamUpdatedInbox,
    allocation: AllocationReader,
    blackboard: BlackboardReader,
    memory: VectorMemorySearcher,
    memo: Memo,
    llm: LlmAccess,
    system_prompt: std::sync::OnceLock<String>,
}

impl QueryVectorModule {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        query: QueryInbox,
        attention_updates: AttentionStreamUpdatedInbox,
        allocation: AllocationReader,
        blackboard: BlackboardReader,
        memory: VectorMemorySearcher,
        memo: Memo,
        llm: LlmAccess,
    ) -> Self {
        Self {
            owner: nuillu_types::ModuleId::new(<Self as Module>::id())
                .expect("query-vector id is valid"),
            query,
            attention_updates,
            allocation,
            blackboard,
            memory,
            memo,
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
    async fn handle_queries(
        &self,
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
    async fn activate_attention_update(&self, cx: &nuillu_module::ActivateCx<'_>) -> Result<()> {
        let question = self
            .blackboard
            .read(|bb| {
                let entries = bb.attention_stream().entries().to_vec();
                let latest = entries
                    .last()
                    .map(|entry| entry.text.as_str())
                    .unwrap_or("current attention stream");
                format!("Find stable memory that clarifies this attended context: {latest}")
            })
            .await;
        let hits = self.search_with_memory(cx, &[question]).await?;
        self.write_hits(&hits).await
    }

    async fn write_hits(&self, hits: &[QueryVectorMemoryHit]) -> Result<()> {
        let content = hit_contents(hits);
        if !content.is_empty() {
            self.memo.write(content).await;
        }
        Ok(())
    }

    async fn search_with_memory(
        &self,
        cx: &nuillu_module::ActivateCx<'_>,
        questions: &[String],
    ) -> Result<Vec<QueryVectorMemoryHit>> {
        let snapshot = self
            .blackboard
            .read(|bb| {
                serde_json::json!({
                    "memos": bb.memos(),
                    "attention_stream": bb.attention_stream().entries(),
                    "memory_metadata": bb.memory_metadata(),
                })
            })
            .await;
        let allocation = self.allocation.snapshot().await;
        let mut session = Session::new();
        session.push_system(self.system_prompt(cx));
        session.push_user(
            serde_json::json!({
                "questions": questions,
                "blackboard": snapshot,
                "allocation": allocation,
            })
            .to_string(),
        );

        let mut all_hits = Vec::new();
        for _ in 0..MAX_QUERY_ROUNDS {
            let lutum = self.llm.lutum().await;
            let turn = session
                .structured_turn::<QueryVectorSearchCompletion>(&lutum)
                .tools::<QueryVectorTools>()
                .available_tools([QueryVectorToolsSelector::SearchVectorMemory]);
            let turn = if all_hits.is_empty() {
                turn.require_tool(QueryVectorToolsSelector::SearchVectorMemory)
            } else {
                turn
            };
            let outcome = turn
                .collect()
                .await
                .context("query-vector structured turn failed")?;

            match outcome {
                StructuredStepOutcomeWithTools::Finished(result) => {
                    let StructuredTurnOutcome::Structured(_completion) = result.semantic else {
                        anyhow::bail!("query-vector search completion refused");
                    };
                    return Ok(all_hits);
                }
                StructuredStepOutcomeWithTools::NeedsTools(round) => {
                    let mut tool_results: Vec<ToolResult> = Vec::new();
                    for call in round.tool_calls.iter().cloned() {
                        let QueryVectorToolsCall::SearchVectorMemory(call) = call;
                        let output = self
                            .search_vector_memory(call.input.clone())
                            .await
                            .context("run search_vector_memory tool")?;
                        all_hits.extend(output.hits.clone());
                        let result = call
                            .complete(output)
                            .context("complete search_vector_memory tool call")?;
                        tool_results.push(result);
                    }
                    round
                        .commit(&mut session, tool_results)
                        .context("commit query-vector tool round")?;
                }
            }
        }

        Ok(all_hits)
    }

    async fn search_vector_memory(
        &self,
        args: SearchVectorMemoryArgs,
    ) -> Result<SearchVectorMemoryOutput> {
        let limit = args.limit.clamp(1, 16);
        let records = self
            .memory
            .search(&args.query, limit, args.filter_rank)
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
        if !content.is_empty() && !contents.iter().any(|seen| *seen == content) {
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
        "Vector-memory/RAG retrieval: surfaces stored memory content into its memo on QueryRequest or attention updates; never synthesizes answers."
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
        if batch.attention_updated {
            self.activate_attention_update(cx).await?;
        }
        Ok(())
    }
}
