use anyhow::{Context, Result};
use async_trait::async_trait;
use lutum::{Session, StructuredStepOutcomeWithTools, StructuredTurnOutcome, ToolResult};
use nuillu_module::{
    BlackboardReader, FileSearcher, LlmAccess, Memo, Module, PeriodicInbox, QueryInbox,
    QueryRequest, ports::FileSearchQuery,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

mod batch;

const SYSTEM_PROMPT: &str = r#"You are the query-agentic module.
Answer questions by searching read-only files. Use the search_files tool for file and text lookup.
The tool intentionally exposes only ripgrep-like controls: pattern, regex, invert_match,
case_sensitive, context, and max_matches.
Do not answer self-referential attention, awareness, or self-model questions."#;

const MAX_QUERY_ROUNDS: usize = 4;

#[lutum::tool_input(name = "search_files", output = SearchFilesOutput)]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct SearchFilesArgs {
    pub pattern: String,
    pub regex: bool,
    pub invert_match: bool,
    pub case_sensitive: bool,
    pub context: usize,
    pub max_matches: usize,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct QueryFileHit {
    pub path: String,
    pub line: usize,
    pub snippet: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct SearchFilesOutput {
    pub hits: Vec<QueryFileHit>,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct QueryAgenticResultMemo {
    pub question: String,
    pub answer: String,
    pub file_hits: Vec<QueryFileHit>,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct QueryBatchAnswer {
    pub question: String,
    pub answer: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct QueryAgenticBatchResultMemo {
    pub answers: Vec<QueryBatchAnswer>,
    pub file_hits: Vec<QueryFileHit>,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct QueryAgenticBatchAnswerSet {
    pub answers: Vec<QueryBatchAnswer>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema, lutum::Toolset)]
pub enum QueryAgenticTools {
    SearchFiles(SearchFilesArgs),
}

pub struct QueryAgenticModule {
    query: QueryInbox,
    periodic: PeriodicInbox,
    blackboard: BlackboardReader,
    files: FileSearcher,
    memo: Memo,
    llm: LlmAccess,
}

impl QueryAgenticModule {
    pub fn new(
        query: QueryInbox,
        periodic: PeriodicInbox,
        blackboard: BlackboardReader,
        files: FileSearcher,
        memo: Memo,
        llm: LlmAccess,
    ) -> Self {
        Self {
            query,
            periodic,
            blackboard,
            files,
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
        let (answers, hits) = self.answer_batch_with_files(&questions).await?;
        self.write_result(QueryAgenticBatchResultMemo {
            answers,
            file_hits: hits,
        })
        .await
    }

    #[tracing::instrument(skip_all, err(Debug, level = "warn"))]
    async fn activate_periodic(&self) -> Result<()> {
        let question = "Review current blackboard state for useful file context.";
        let (answers, hits) = self.answer_batch_with_files(&[question.to_owned()]).await?;
        self.write_result(QueryAgenticBatchResultMemo {
            answers,
            file_hits: hits,
        })
        .await
    }

    async fn write_result(&self, result: QueryAgenticBatchResultMemo) -> Result<()> {
        let serialized =
            serde_json::to_string(&result).context("serialize query-agentic result memo")?;
        self.memo.write(serialized).await;
        Ok(())
    }

    async fn answer_batch_with_files(
        &self,
        questions: &[String],
    ) -> Result<(Vec<QueryBatchAnswer>, Vec<QueryFileHit>)> {
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
                .structured_turn::<QueryAgenticBatchAnswerSet>()
                .tools::<QueryAgenticTools>()
                .available_tools([QueryAgenticToolsSelector::SearchFiles])
                .collect()
                .await
                .context("query-agentic structured turn failed")?;

            match outcome {
                StructuredStepOutcomeWithTools::Finished(result) => {
                    let StructuredTurnOutcome::Structured(batch) = result.semantic else {
                        anyhow::bail!("query-agentic batch answer refused");
                    };
                    return Ok((batch.answers, all_hits));
                }
                StructuredStepOutcomeWithTools::NeedsTools(round) => {
                    let mut tool_results: Vec<ToolResult> = Vec::new();
                    for call in round.tool_calls.iter().cloned() {
                        let QueryAgenticToolsCall::SearchFiles(call) = call;
                        let output = self
                            .search_files(call.input.clone())
                            .await
                            .context("run search_files tool")?;
                        all_hits.extend(output.hits.clone());
                        let result = call
                            .complete(output)
                            .context("complete search_files tool call")?;
                        tool_results.push(result);
                    }
                    round
                        .commit(&mut session, tool_results)
                        .context("commit query-agentic tool round")?;
                }
            }
        }

        Ok((fallback_answers(questions), all_hits))
    }

    async fn search_files(&self, args: SearchFilesArgs) -> Result<SearchFilesOutput> {
        let query = FileSearchQuery {
            pattern: args.pattern,
            regex: args.regex,
            invert_match: args.invert_match,
            case_sensitive: args.case_sensitive,
            context: args.context,
            max_matches: args.max_matches,
        };
        let records = self.files.search(query).await.context("search files")?;
        Ok(SearchFilesOutput {
            hits: records
                .into_iter()
                .map(|record| QueryFileHit {
                    path: record.path,
                    line: record.line,
                    snippet: record.snippet,
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
            answer: "I could not finish the file query within the round limit.".into(),
        })
        .collect()
}

#[async_trait(?Send)]
impl Module for QueryAgenticModule {
    async fn run(&mut self) {
        if let Err(error) = self.run_loop().await {
            tracing::debug!(?error, "query-agentic module loop stopped");
        }
    }
}
