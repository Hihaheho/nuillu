use anyhow::{Context, Result};
use async_trait::async_trait;
use lutum::{Session, StructuredStepOutcomeWithTools, StructuredTurnOutcome, ToolResult};
use nuillu_module::{
    AllocationReader, AllocationUpdatedInbox, BlackboardReader, FileSearcher, LlmAccess, Memo,
    Module, QueryInbox, QueryRequest, ports::FileSearchQuery,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

mod batch;
pub use batch::NextBatch as QueryAgenticBatch;

const SYSTEM_PROMPT: &str = r#"You are the query-agentic module.
Choose read-only file searches only. Use the search_files tool for file and text lookup.
If the question contains allocation guidance or a speak-gate evidence request, search for the concrete
requested facts, proper nouns, route/world terms, filenames hinted by the guidance, and the
needed_fact phrases.
The tool intentionally exposes only ripgrep-like controls: pattern, regex, invert_match,
case_sensitive, context, and max_matches.
The backend is lexical line search like ripgrep, not semantic search. Exact words and short
regex anchors matter. Prefer simple case-insensitive literal searches or broad regex OR patterns
over paraphrases. If a search returns no hits, retry with shorter terms taken from the question:
proper nouns, domain nouns, direction words, filenames hinted by the question, and key verbs.
Do not drift into general associations that are not words from the question or current context.
Do not answer questions, explain results, describe this module, or add any text from outside tool
results. You must call search_files before returning the structured completion. The runtime
memoizes only file hit snippets returned by tools. Return only raw JSON for the structured
completion; do not wrap JSON in Markdown or code fences."#;

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
pub struct QueryAgenticSearchCompletion {
    pub done: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema, lutum::Toolset)]
pub enum QueryAgenticTools {
    SearchFiles(SearchFilesArgs),
}

pub struct QueryAgenticModule {
    query: QueryInbox,
    allocation_updates: AllocationUpdatedInbox,
    allocation: AllocationReader,
    blackboard: BlackboardReader,
    files: FileSearcher,
    memo: Memo,
    llm: LlmAccess,
}

impl QueryAgenticModule {
    pub fn new(
        query: QueryInbox,
        allocation_updates: AllocationUpdatedInbox,
        allocation: AllocationReader,
        blackboard: BlackboardReader,
        files: FileSearcher,
        memo: Memo,
        llm: LlmAccess,
    ) -> Self {
        Self {
            query,
            allocation_updates,
            allocation,
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
        let hits = self.search_with_files(&questions).await?;
        self.write_hits(&hits).await
    }

    #[tracing::instrument(skip_all, err(Debug, level = "warn"))]
    async fn activate_guidance(&self) -> Result<()> {
        let allocation = self.allocation.snapshot().await;
        let guidance = allocation
            .iter()
            .find(|(id, _config)| id.as_str() == "query-agentic")
            .map(|(_id, config)| config.guidance.as_str())
            .unwrap_or_default();
        let question = guidance_question(guidance, "file");
        let hits = self.search_with_files(&[question]).await?;
        self.write_hits(&hits).await
    }

    async fn write_hits(&self, hits: &[QueryFileHit]) -> Result<()> {
        let content = hit_snippets(hits);
        if !content.is_empty() {
            self.memo.write(content).await;
        }
        Ok(())
    }

    async fn search_with_files(&self, questions: &[String]) -> Result<Vec<QueryFileHit>> {
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
        let lutum = self.llm.lutum().await;
        let mut session = Session::new(lutum);
        session.push_system(SYSTEM_PROMPT);
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
            let turn = session
                .structured_turn::<QueryAgenticSearchCompletion>()
                .tools::<QueryAgenticTools>()
                .available_tools([QueryAgenticToolsSelector::SearchFiles]);
            let turn = if all_hits.is_empty() {
                turn.require_tool(QueryAgenticToolsSelector::SearchFiles)
            } else {
                turn
            };
            let outcome = turn
                .collect()
                .await
                .context("query-agentic structured turn failed")?;

            match outcome {
                StructuredStepOutcomeWithTools::Finished(result) => {
                    let StructuredTurnOutcome::Structured(_completion) = result.semantic else {
                        anyhow::bail!("query-agentic search completion refused");
                    };
                    return Ok(all_hits);
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

        Ok(all_hits)
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
}

fn hit_snippets(hits: &[QueryFileHit]) -> String {
    let mut snippets = Vec::new();
    for hit in hits {
        let snippet = hit.snippet.trim();
        if !snippet.is_empty() && !snippets.iter().any(|seen| *seen == snippet) {
            snippets.push(snippet);
        }
    }
    snippets.join("\n\n")
}

fn guidance_question(guidance: &str, context_kind: &str) -> String {
    let trimmed = guidance.trim();
    if trimmed.is_empty() {
        return format!("Act on current allocation guidance for useful {context_kind} context.");
    }
    format!("Act on this allocation guidance for {context_kind} lookup: {trimmed}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn guidance_question_carries_concrete_guidance() {
        let question = guidance_question(
            "speech-evidence request: question=torus route; needed_fact=eastbound appears west",
            "file",
        );

        assert_eq!(
            question,
            "Act on this allocation guidance for file lookup: speech-evidence request: question=torus route; needed_fact=eastbound appears west"
        );
    }
}

#[async_trait(?Send)]
impl Module for QueryAgenticModule {
    type Batch = QueryAgenticBatch;

    fn id() -> &'static str {
        "query-agentic"
    }

    fn role_description() -> &'static str {
        "Read-only file-search retrieval: surfaces snippets from project files into its memo on QueryRequest or allocation cues; never synthesizes answers."
    }

    async fn next_batch(&mut self) -> Result<Self::Batch> {
        QueryAgenticModule::next_batch(self).await
    }

    async fn activate(&mut self, batch: &Self::Batch) -> Result<()> {
        if !batch.queries.is_empty() {
            self.handle_queries(batch.queries.clone()).await?;
        }
        if batch.guidance {
            self.activate_guidance().await?;
        }
        Ok(())
    }
}
