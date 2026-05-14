use anyhow::{Context, Result};
use async_trait::async_trait;
use lutum::{Session, TextStepOutcomeWithTools, ToolResult};
use nuillu_module::{
    AllocationReader, AllocationUpdatedInbox, BlackboardReader, LlmAccess, Memo, Module,
    SessionCompactionConfig, SessionCompactionProtectedPrefix, compact_session_if_needed,
    push_formatted_memo_log_batch,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

mod batch;
mod file_search;
pub use batch::NextBatch as QueryAgenticBatch;
pub use file_search::{
    FileSearchHit, FileSearchProvider, FileSearchQuery, FileSearcher, NoopFileSearchProvider,
};

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
results. You must call search_files before finishing. The runtime memoizes only file hit snippets
returned by tools. Any final text is ignored; do not use a final answer as a data channel."#;

const MAX_QUERY_ROUNDS: usize = 4;
const COMPACTED_QUERY_AGENTIC_SESSION_PREFIX: &str = "Compacted query-agentic session history:";
const SESSION_COMPACTION_PROMPT: &str = r#"You compact the query-agentic module's persistent session history.
Summarize only the prefix transcript you receive. Preserve memo-log facts, query requests,
allocation guidance, file-search arguments, useful file hits, and failed searches future retrieval
should not repeat. Do not invent facts. Return plain text only."#;

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

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema, lutum::Toolset)]
pub enum QueryAgenticTools {
    SearchFiles(SearchFilesArgs),
}

pub struct QueryAgenticModule {
    owner: nuillu_module::ModuleId,
    allocation_updates: AllocationUpdatedInbox,
    allocation: AllocationReader,
    blackboard: BlackboardReader,
    files: FileSearcher,
    memo: Memo,
    llm: LlmAccess,
    session: Session,
    session_compaction: SessionCompactionConfig,
    system_prompt: std::sync::OnceLock<String>,
}

impl QueryAgenticModule {
    pub fn new(
        allocation_updates: AllocationUpdatedInbox,
        allocation: AllocationReader,
        blackboard: BlackboardReader,
        files: FileSearcher,
        memo: Memo,
        llm: LlmAccess,
    ) -> Self {
        Self {
            owner: nuillu_module::ModuleId::new(<Self as Module>::id())
                .expect("query-agentic id is valid"),
            allocation_updates,
            allocation,
            blackboard,
            files,
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
                cx.core_policies(),
                cx.now(),
            )
        })
    }

    #[tracing::instrument(skip_all, err(Debug, level = "warn"))]
    async fn activate_guidance(&mut self, cx: &nuillu_module::ActivateCx<'_>) -> Result<()> {
        let allocation = self.allocation.snapshot().await;
        let guidance = allocation
            .iter()
            .find(|(id, _config)| id.as_str() == "query-agentic")
            .map(|(_id, config)| config.guidance.as_str())
            .unwrap_or_default();
        let question = guidance_question(guidance, "file");
        let hits = self.search_with_files(cx, &[question]).await?;
        self.write_hits(&hits).await
    }

    async fn write_hits(&self, hits: &[QueryFileHit]) -> Result<()> {
        let content = hit_snippets(hits);
        if !content.is_empty() {
            self.memo.write(content).await;
        }
        Ok(())
    }

    async fn search_with_files(
        &mut self,
        cx: &nuillu_module::ActivateCx<'_>,
        questions: &[String],
    ) -> Result<Vec<QueryFileHit>> {
        let unread_memo_logs = self.blackboard.unread_memo_logs().await;
        push_formatted_memo_log_batch(&mut self.session, &unread_memo_logs, cx.now());
        let system_prompt = self.system_prompt(cx).to_owned();
        self.session.push_ephemeral_system(system_prompt);
        self.session
            .push_user(format_file_search_questions(questions));

        let mut all_hits = Vec::new();
        for _ in 0..MAX_QUERY_ROUNDS {
            let lutum = self.llm.lutum().await;
            let turn = self
                .session
                .text_turn(&lutum)
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
                .context("query-agentic text turn failed")?;

            match outcome {
                TextStepOutcomeWithTools::Finished(result) => {
                    compact_session_if_needed(
                        &mut self.session,
                        result.usage.input_tokens,
                        cx.session_compaction_lutum(),
                        self.session_compaction,
                        SessionCompactionProtectedPrefix::None,
                        Self::id(),
                        COMPACTED_QUERY_AGENTIC_SESSION_PREFIX,
                        SESSION_COMPACTION_PROMPT,
                    )
                    .await;
                    return Ok(all_hits);
                }
                TextStepOutcomeWithTools::FinishedNoOutput(result) => {
                    compact_session_if_needed(
                        &mut self.session,
                        result.usage.input_tokens,
                        cx.session_compaction_lutum(),
                        self.session_compaction,
                        SessionCompactionProtectedPrefix::None,
                        Self::id(),
                        COMPACTED_QUERY_AGENTIC_SESSION_PREFIX,
                        SESSION_COMPACTION_PROMPT,
                    )
                    .await;
                    return Ok(all_hits);
                }
                TextStepOutcomeWithTools::NeedsTools(round) => {
                    let input_tokens = round.usage.input_tokens;
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
                        .commit(&mut self.session, tool_results)
                        .context("commit query-agentic tool round")?;
                    compact_session_if_needed(
                        &mut self.session,
                        input_tokens,
                        cx.session_compaction_lutum(),
                        self.session_compaction,
                        SessionCompactionProtectedPrefix::None,
                        Self::id(),
                        COMPACTED_QUERY_AGENTIC_SESSION_PREFIX,
                        SESSION_COMPACTION_PROMPT,
                    )
                    .await;
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

fn format_file_search_questions(questions: &[String]) -> String {
    let mut out = String::from("File-search questions to investigate:");
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

fn hit_snippets(hits: &[QueryFileHit]) -> String {
    let mut snippets = Vec::new();
    for hit in hits {
        let snippet = hit.snippet.trim();
        if !snippet.is_empty() && !snippets.contains(&snippet) {
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

#[async_trait(?Send)]
impl Module for QueryAgenticModule {
    type Batch = QueryAgenticBatch;

    fn id() -> &'static str {
        "query-agentic"
    }

    fn role_description() -> &'static str {
        "Read-only file-search retrieval: writes relevant file snippets to its memo log on allocation guidance cues; cognition-gate must promote useful snippets before speech uses them; never synthesizes answers."
    }

    async fn next_batch(&mut self) -> Result<Self::Batch> {
        QueryAgenticModule::next_batch(self).await
    }

    async fn activate(
        &mut self,
        cx: &nuillu_module::ActivateCx<'_>,
        batch: &Self::Batch,
    ) -> Result<()> {
        if batch.guidance {
            self.activate_guidance(cx).await?;
        }
        Ok(())
    }
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
