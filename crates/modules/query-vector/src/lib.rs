use anyhow::{Context, Result};
use async_trait::async_trait;
use lutum::{Session, StructuredStepOutcomeWithTools, StructuredTurnOutcome, ToolResult};
use nuillu_module::{
    BlackboardReader, LlmAccess, Memo, Module, PeriodicInbox, QueryInbox, QueryRequest,
    VectorMemorySearcher,
};
use nuillu_types::{MemoryIndex, MemoryRank};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

mod batch;

const SYSTEM_PROMPT: &str = r#"You are the query-vector module.
Answer vector-memory/RAG questions only. Use the search_vector_memory tool for factual memory lookup. Do not
answer self-referential attention, awareness, or self-model questions."#;

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
pub struct QueryResultMemo {
    pub question: String,
    pub answer: String,
    pub vector_memory_hits: Vec<QueryVectorMemoryHit>,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct QueryBatchAnswer {
    pub question: String,
    pub answer: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct QueryVectorBatchResultMemo {
    pub answers: Vec<QueryBatchAnswer>,
    pub vector_memory_hits: Vec<QueryVectorMemoryHit>,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct QueryVectorBatchAnswerSet {
    pub answers: Vec<QueryBatchAnswer>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema, lutum::Toolset)]
pub enum QueryVectorTools {
    SearchVectorMemory(SearchVectorMemoryArgs),
}

pub struct QueryVectorModule {
    query: QueryInbox,
    periodic: PeriodicInbox,
    blackboard: BlackboardReader,
    memory: VectorMemorySearcher,
    memo: Memo,
    llm: LlmAccess,
}

impl QueryVectorModule {
    pub fn new(
        query: QueryInbox,
        periodic: PeriodicInbox,
        blackboard: BlackboardReader,
        memory: VectorMemorySearcher,
        memo: Memo,
        llm: LlmAccess,
    ) -> Self {
        Self {
            query,
            periodic,
            blackboard,
            memory,
            memo,
            llm,
        }
    }

    #[tracing::instrument(skip_all, err(Debug, level = "warn"))]
    async fn handle_queries(&self, requests: Vec<QueryRequest>) -> Result<()> {
        let questions = requests
            .into_iter()
            .map(|request| request.question)
            .collect::<Vec<_>>();
        if questions.is_empty() {
            return Ok(());
        }
        let (answers, hits) = self.answer_batch_with_memory(&questions).await?;
        self.write_result(QueryVectorBatchResultMemo {
            answers,
            vector_memory_hits: hits,
        })
        .await
    }

    #[tracing::instrument(skip_all, err(Debug, level = "warn"))]
    async fn activate_periodic(&self) -> Result<()> {
        let question = "Review current blackboard state for useful memory context.";
        let (answers, hits) = self
            .answer_batch_with_memory(&[question.to_owned()])
            .await?;
        self.write_result(QueryVectorBatchResultMemo {
            answers,
            vector_memory_hits: hits,
        })
        .await
    }

    async fn write_result(&self, result: QueryVectorBatchResultMemo) -> Result<()> {
        let serialized =
            serde_json::to_string(&result).context("serialize query-vector result memo")?;
        self.memo.write(serialized).await;
        Ok(())
    }

    async fn answer_batch_with_memory(
        &self,
        questions: &[String],
    ) -> Result<(Vec<QueryBatchAnswer>, Vec<QueryVectorMemoryHit>)> {
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
        let lutum = self.llm.lutum().await;
        let mut session = Session::new(lutum);
        session.push_system(SYSTEM_PROMPT);
        session.push_user(
            serde_json::json!({
                "questions": questions,
                "blackboard": snapshot,
            })
            .to_string(),
        );

        let mut all_hits = Vec::new();
        for _ in 0..MAX_QUERY_ROUNDS {
            let outcome = session
                .structured_turn::<QueryVectorBatchAnswerSet>()
                .tools::<QueryVectorTools>()
                .available_tools([QueryVectorToolsSelector::SearchVectorMemory])
                .collect()
                .await
                .context("query-vector structured turn failed")?;

            match outcome {
                StructuredStepOutcomeWithTools::Finished(result) => {
                    let StructuredTurnOutcome::Structured(batch) = result.semantic else {
                        anyhow::bail!("query-vector batch answer refused");
                    };
                    return Ok((batch.answers, all_hits));
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

        Ok((fallback_answers(questions), all_hits))
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

    async fn run_loop(&mut self) -> Result<()> {
        loop {
            let batch = self.next_batch().await?;
            if !batch.queries.is_empty() {
                let _ = self.handle_queries(batch.queries).await;
            }
            if batch.periodic {
                let _ = self.activate_periodic().await;
            }
        }
    }
}

fn fallback_answers(questions: &[String]) -> Vec<QueryBatchAnswer> {
    questions
        .iter()
        .map(|question| QueryBatchAnswer {
            question: question.clone(),
            answer: "I could not finish the query within the round limit.".into(),
        })
        .collect()
}

#[async_trait(?Send)]
impl Module for QueryVectorModule {
    async fn run(&mut self) {
        if let Err(error) = self.run_loop().await {
            tracing::debug!(?error, "query-vector module loop stopped");
        }
    }
}
