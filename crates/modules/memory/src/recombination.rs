use anyhow::{Context, Result};
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use lutum::{ModelInput, TextStepOutcomeWithTools, ToolResult};
use nuillu_module::{
    AllocationReader, BlackboardReader, CognitionWriter, InteroceptiveUpdatedInbox, LlmAccess,
    Module, compact_llm_context_text, render_memory_for_llm,
};
use nuillu_types::{MemoryIndex, MemoryRank, builtin};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::store::{MemoryRecord, MemoryRetriever, MemoryUsageTarget};

const MAX_RECOMBINATION_SEARCH_QUERY_CHARS: usize = 300;
const MAX_RECOMBINATION_SEARCHES: usize = 3;
const MAX_RECOMBINATION_SEARCH_LIMIT: usize = 5;
const MAX_RECOMBINATION_TOOL_ROUNDS: usize = 4;
const RECENT_COGNITION_ENTRY_CHARS: usize = 800;
const TOOL_TURN_MAX_OUTPUT_TOKENS: u32 = 768;

const SYSTEM_PROMPT: &str = r#"You are the memory-recombination module.
You run a REM-like internal dream simulation. Combine recent non-dream cognition, retrieved memory
fragments, identity memory, and core policies. Produce one short associative counterfactual or
scenario that may help future planning or memory reweighting.

When memory context would help, call search_memory with a short, concrete natural-language query.
You may call search_memory at most three times per activation; the runtime enforces a 300-character
query limit. Search results are unverified memory fragments, not current truth.

This is not a verified fact and not an outward reply. Do not repeat the logs faithfully. Do not
violate core policies. Keep the appended text under 200 Japanese characters or 120 English words.
Call append_recombination exactly once only when the recent seeds or retrieved fragments are enough
for a useful internal simulation; otherwise finish without tools."#;

#[lutum::tool_input(name = "search_memory", output = SearchMemoryOutput)]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct SearchMemoryArgs {
    pub query: String,
    pub limit: usize,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct SearchMemoryOutput {
    pub query: String,
    pub limit: usize,
    pub truncated: bool,
    pub hits: Vec<RecombinationMemoryHit>,
    pub message: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct RecombinationMemoryHit {
    pub index: MemoryIndex,
    pub rank: MemoryRank,
    pub occurred_at: Option<DateTime<Utc>>,
    pub content: String,
}

#[lutum::tool_input(name = "append_recombination", output = AppendRecombinationOutput)]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct AppendRecombinationArgs {
    pub text: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct AppendRecombinationOutput {
    pub appended: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema, lutum::Toolset)]
pub enum RecombinationTools {
    SearchMemory(SearchMemoryArgs),
    AppendRecombination(AppendRecombinationArgs),
}

pub struct MemoryRecombinationModule {
    owner: nuillu_types::ModuleId,
    interoception_updates: InteroceptiveUpdatedInbox,
    allocation: AllocationReader,
    blackboard: BlackboardReader,
    memory: MemoryRetriever,
    cognition: CognitionWriter,
    llm: LlmAccess,
}

impl MemoryRecombinationModule {
    pub fn new(
        interoception_updates: InteroceptiveUpdatedInbox,
        allocation: AllocationReader,
        blackboard: BlackboardReader,
        memory: MemoryRetriever,
        cognition: CognitionWriter,
        llm: LlmAccess,
    ) -> Self {
        Self {
            owner: nuillu_types::ModuleId::new(<Self as Module>::id())
                .expect("memory-recombination id is valid"),
            interoception_updates,
            allocation,
            blackboard,
            memory,
            cognition,
            llm,
        }
    }

    #[tracing::instrument(skip_all, err(Debug, level = "warn"))]
    async fn activate(&mut self, cx: &nuillu_module::ActivateCx<'_>) -> Result<()> {
        let allocation = self.allocation.snapshot().await;
        if allocation.activation_for(&self.owner) == nuillu_blackboard::ActivationRatio::ZERO {
            return Ok(());
        }

        let recent = self
            .blackboard
            .read(|bb| {
                let mut entries = bb
                    .unread_cognition_log_entries(None)
                    .into_iter()
                    .filter(|record| record.source.module != builtin::memory_recombination())
                    .collect::<Vec<_>>();
                entries.sort_by_key(|record| record.index);
                entries
                    .into_iter()
                    .rev()
                    .take(8)
                    .collect::<Vec<_>>()
                    .into_iter()
                    .rev()
                    .collect::<Vec<_>>()
            })
            .await;
        if recent.is_empty() {
            return Ok(());
        }

        let mut input = ModelInput::new()
            .system(nuillu_module::format_system_seed(
                nuillu_module::format_system_prompt(
                    SYSTEM_PROMPT,
                    cx.peer_contexts(),
                    &self.owner,
                    cx.core_policies(),
                ),
                false,
                cx.identity_memories(),
                cx.now(),
            ))
            .user(format_recombination_context(&recent));

        let mut search_count = 0_usize;
        let mut append_used = false;
        for _ in 0..MAX_RECOMBINATION_TOOL_ROUNDS {
            let available_tools = if search_count >= MAX_RECOMBINATION_SEARCHES {
                vec![RecombinationToolsSelector::AppendRecombination]
            } else {
                vec![
                    RecombinationToolsSelector::SearchMemory,
                    RecombinationToolsSelector::AppendRecombination,
                ]
            };
            let lutum = self.llm.lutum().await;
            let outcome = lutum
                .text_turn(input.clone())
                .tools::<RecombinationTools>()
                .available_tools(available_tools)
                .max_output_tokens(TOOL_TURN_MAX_OUTPUT_TOKENS)
                .collect_controlled_with(nuillu_module::AbortOnAvailableToolNameInText::new())
                .await
                .context("memory-recombination text turn failed")?;

            let TextStepOutcomeWithTools::NeedsTools(round) = outcome else {
                return Ok(());
            };
            if round.tool_calls.is_empty() {
                return Ok(());
            }

            let mut results = Vec::<ToolResult>::new();
            nuillu_module::emit_trace_tool_calls(&round.tool_calls);
            for call in round.tool_calls.iter().cloned() {
                match call {
                    RecombinationToolsCall::SearchMemory(call) => {
                        search_count = search_count.saturating_add(1);
                        let output = if search_count > MAX_RECOMBINATION_SEARCHES {
                            search_limit_reached_output(call.input.clone())
                        } else {
                            run_recombination_search(&self.memory, call.input.clone(), cx.now())
                                .await
                        };
                        results.push(
                            call.complete(output)
                                .context("complete search_memory tool call")?,
                        );
                    }
                    RecombinationToolsCall::AppendRecombination(call) => {
                        let output = self
                            .append_recombination(call.input.clone(), &mut append_used)
                            .await
                            .context("run append_recombination tool")?;
                        results.push(
                            call.complete(output)
                                .context("complete append_recombination tool call")?,
                        );
                    }
                }
            }
            round
                .commit_into(&mut input, results)
                .context("commit memory-recombination tool round")?;
            if append_used {
                return Ok(());
            }
        }
        Ok(())
    }

    async fn append_recombination(
        &self,
        args: AppendRecombinationArgs,
        append_used: &mut bool,
    ) -> Result<AppendRecombinationOutput> {
        let (entry, output) = prepare_recombination_append(args, append_used);
        if let Some(entry) = entry {
            self.cognition.append(entry).await;
        }
        Ok(output)
    }

    async fn next_batch(&mut self) -> Result<()> {
        let _ = self.interoception_updates.next_item().await?;
        let _ = self.interoception_updates.take_ready_items()?;
        Ok(())
    }
}

fn format_recombination_context(recent: &[nuillu_blackboard::CognitionLogEntryRecord]) -> String {
    let mut out = String::from("Recent non-dream cognition seeds:");
    for record in recent {
        let text = record.entry.text.trim();
        if !text.is_empty() {
            out.push_str("\n- ");
            out.push_str(&compact_llm_context_text(
                text,
                RECENT_COGNITION_ENTRY_CHARS,
            ));
        }
    }
    out.push_str("\n\nUse search_memory only if targeted memory fragments would help.");
    out
}

async fn run_recombination_search(
    memory: &MemoryRetriever,
    args: SearchMemoryArgs,
    now: DateTime<Utc>,
) -> SearchMemoryOutput {
    let (query, truncated) = bounded_search_query(&args.query);
    let limit = args.limit.clamp(1, MAX_RECOMBINATION_SEARCH_LIMIT);
    if query.is_empty() {
        return SearchMemoryOutput {
            query,
            limit,
            truncated,
            hits: Vec::new(),
            message: "empty memory search query; no search was run".to_owned(),
        };
    }

    let records = match memory.search(&query, limit).await {
        Ok(records) => records,
        Err(error) => {
            tracing::warn!(
                error = ?error,
                "memory-recombination search_memory backend failed"
            );
            return SearchMemoryOutput {
                query,
                limit,
                truncated,
                hits: Vec::new(),
                message: "memory search unavailable for this activation".to_owned(),
            };
        }
    };
    let targets = records
        .iter()
        .map(MemoryUsageTarget::from)
        .collect::<Vec<_>>();
    memory.record_accesses(&targets).await;
    let hits = records
        .into_iter()
        .map(|record| recombination_hit(record, now))
        .collect::<Vec<_>>();
    let hit_count = hits.len();
    SearchMemoryOutput {
        query,
        limit,
        truncated,
        hits,
        message: format!("memory search returned {hit_count} hit(s)"),
    }
}

fn search_limit_reached_output(args: SearchMemoryArgs) -> SearchMemoryOutput {
    let (query, truncated) = bounded_search_query(&args.query);
    SearchMemoryOutput {
        query,
        limit: args.limit.clamp(1, MAX_RECOMBINATION_SEARCH_LIMIT),
        truncated,
        hits: Vec::new(),
        message: "memory search limit reached for this activation".to_owned(),
    }
}

fn bounded_search_query(query: &str) -> (String, bool) {
    let trimmed = query.trim();
    let Some((end, _)) = trimmed
        .char_indices()
        .nth(MAX_RECOMBINATION_SEARCH_QUERY_CHARS)
    else {
        return (trimmed.to_owned(), false);
    };
    (trimmed[..end].to_owned(), true)
}

fn recombination_hit(record: MemoryRecord, now: DateTime<Utc>) -> RecombinationMemoryHit {
    RecombinationMemoryHit {
        index: record.index,
        rank: record.rank,
        occurred_at: record.occurred_at,
        content: render_memory_for_llm(record.content.as_str(), record.occurred_at, now),
    }
}

fn prepare_recombination_append(
    args: AppendRecombinationArgs,
    append_used: &mut bool,
) -> (Option<String>, AppendRecombinationOutput) {
    if *append_used {
        return (None, AppendRecombinationOutput { appended: false });
    }
    *append_used = true;
    let text = args.text.trim();
    if text.is_empty() {
        return (None, AppendRecombinationOutput { appended: false });
    }
    (
        Some(format!(
            "Internal dream simulation, not a verified fact: {text}"
        )),
        AppendRecombinationOutput { appended: true },
    )
}

#[async_trait(?Send)]
impl Module for MemoryRecombinationModule {
    type Batch = ();

    fn id() -> &'static str {
        "memory-recombination"
    }

    fn peer_context() -> Option<&'static str> {
        None
    }

    fn allocation_hint() -> Option<&'static str> {
        Some(
            "Raise memory-recombination during REM-like exploratory phases when memory fragments can be safely combined into imaginative simulations. Keep it low for factual recall, direct answers, urgent action, or when unverified imagery would disturb current cognition.",
        )
    }

    async fn next_batch(&mut self) -> Result<Self::Batch> {
        MemoryRecombinationModule::next_batch(self).await
    }

    async fn activate(
        &mut self,
        cx: &nuillu_module::ActivateCx<'_>,
        _batch: &Self::Batch,
    ) -> Result<()> {
        MemoryRecombinationModule::activate(self, cx).await
    }
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::rc::Rc;

    use chrono::TimeZone;
    use nuillu_blackboard::Blackboard;
    use nuillu_module::ports::{Clock, PortError};
    use nuillu_types::MemoryContent;

    use super::*;
    use crate::store::{
        IndexedMemory, LinkedMemoryQuery, LinkedMemoryRecord, MemoryKind, MemoryLink, MemoryQuery,
        MemoryStore, NewMemory, NewMemoryLink,
    };

    #[derive(Debug)]
    struct FixedClock(DateTime<Utc>);

    #[async_trait(?Send)]
    impl Clock for FixedClock {
        fn now(&self) -> DateTime<Utc> {
            self.0
        }

        async fn sleep_until(&self, _deadline: DateTime<Utc>) {}
    }

    #[derive(Default)]
    struct RecordingSearchStore {
        queries: RefCell<Vec<MemoryQuery>>,
        fail_search: bool,
        records: Vec<MemoryRecord>,
    }

    #[async_trait(?Send)]
    impl MemoryStore for RecordingSearchStore {
        async fn insert(
            &self,
            mem: NewMemory,
            stored_at: DateTime<Utc>,
        ) -> std::result::Result<MemoryRecord, PortError> {
            Ok(record("inserted-memory", mem.content.as_str(), stored_at))
        }

        async fn put(&self, mem: IndexedMemory) -> std::result::Result<MemoryRecord, PortError> {
            Ok(MemoryRecord {
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
            })
        }

        async fn compact(
            &self,
            mem: NewMemory,
            _sources: &[MemoryIndex],
            stored_at: DateTime<Utc>,
        ) -> std::result::Result<MemoryRecord, PortError> {
            Ok(record("compacted-memory", mem.content.as_str(), stored_at))
        }

        async fn put_compacted(
            &self,
            mem: IndexedMemory,
            _sources: &[MemoryIndex],
        ) -> std::result::Result<MemoryRecord, PortError> {
            self.put(mem).await
        }

        async fn get(
            &self,
            _index: &MemoryIndex,
        ) -> std::result::Result<Option<MemoryRecord>, PortError> {
            Ok(None)
        }

        async fn list_by_rank(
            &self,
            _rank: MemoryRank,
        ) -> std::result::Result<Vec<MemoryRecord>, PortError> {
            Ok(Vec::new())
        }

        async fn search(
            &self,
            q: &MemoryQuery,
        ) -> std::result::Result<Vec<MemoryRecord>, PortError> {
            self.queries.borrow_mut().push(q.clone());
            if self.fail_search {
                return Err(PortError::Backend("embedding backend unavailable".into()));
            }
            Ok(self.records.clone())
        }

        async fn linked(
            &self,
            _q: &LinkedMemoryQuery,
        ) -> std::result::Result<Vec<LinkedMemoryRecord>, PortError> {
            Ok(Vec::new())
        }

        async fn upsert_link(
            &self,
            link: NewMemoryLink,
            updated_at: DateTime<Utc>,
        ) -> std::result::Result<MemoryLink, PortError> {
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

        async fn delete(&self, _index: &MemoryIndex) -> std::result::Result<(), PortError> {
            Ok(())
        }
    }

    fn now() -> DateTime<Utc> {
        Utc.with_ymd_and_hms(2026, 6, 14, 12, 0, 0).unwrap()
    }

    fn retriever(store: Rc<RecordingSearchStore>) -> MemoryRetriever {
        MemoryRetriever::new(store, Blackboard::default(), Rc::new(FixedClock(now())))
    }

    fn record(index: &str, content: &str, at: DateTime<Utc>) -> MemoryRecord {
        MemoryRecord {
            index: MemoryIndex::new(index),
            content: MemoryContent::new(content),
            rank: MemoryRank::ShortTerm,
            occurred_at: Some(at),
            stored_at: at,
            kind: MemoryKind::Statement,
            concepts: Vec::new(),
            tags: Vec::new(),
            affect_arousal: 0.0,
            valence: 0.0,
            emotion: String::new(),
        }
    }

    #[tokio::test(flavor = "current_thread")]
    async fn search_query_is_truncated_before_backend_call() {
        let store = Rc::new(RecordingSearchStore::default());
        let memory = retriever(store.clone());
        let output = run_recombination_search(
            &memory,
            SearchMemoryArgs {
                query: "あ".repeat(MAX_RECOMBINATION_SEARCH_QUERY_CHARS + 25),
                limit: 5,
            },
            now(),
        )
        .await;

        assert!(output.truncated);
        assert_eq!(
            output.query.chars().count(),
            MAX_RECOMBINATION_SEARCH_QUERY_CHARS
        );
        assert_eq!(store.queries.borrow().len(), 1);
        assert_eq!(store.queries.borrow()[0].text, output.query);
        assert_eq!(
            store.queries.borrow()[0].text.chars().count(),
            MAX_RECOMBINATION_SEARCH_QUERY_CHARS
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn search_limit_is_clamped() {
        let store = Rc::new(RecordingSearchStore::default());
        let memory = retriever(store.clone());

        let min_output = run_recombination_search(
            &memory,
            SearchMemoryArgs {
                query: "seed".to_owned(),
                limit: 0,
            },
            now(),
        )
        .await;
        let max_output = run_recombination_search(
            &memory,
            SearchMemoryArgs {
                query: "seed".to_owned(),
                limit: 99,
            },
            now(),
        )
        .await;

        assert_eq!(min_output.limit, 1);
        assert_eq!(max_output.limit, MAX_RECOMBINATION_SEARCH_LIMIT);
        let queries = store.queries.borrow();
        assert_eq!(queries[0].limit, 1);
        assert_eq!(queries[1].limit, MAX_RECOMBINATION_SEARCH_LIMIT);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn empty_search_query_does_not_call_backend() {
        let store = Rc::new(RecordingSearchStore::default());
        let memory = retriever(store.clone());
        let output = run_recombination_search(
            &memory,
            SearchMemoryArgs {
                query: "   \n\t  ".to_owned(),
                limit: 5,
            },
            now(),
        )
        .await;

        assert_eq!(
            output,
            SearchMemoryOutput {
                query: String::new(),
                limit: 5,
                truncated: false,
                hits: Vec::new(),
                message: "empty memory search query; no search was run".to_owned(),
            }
        );
        assert!(store.queries.borrow().is_empty());
    }

    #[tokio::test(flavor = "current_thread")]
    async fn backend_error_returns_empty_tool_output() {
        let store = Rc::new(RecordingSearchStore {
            fail_search: true,
            ..Default::default()
        });
        let memory = retriever(store.clone());
        let output = run_recombination_search(
            &memory,
            SearchMemoryArgs {
                query: "related memory".to_owned(),
                limit: 5,
            },
            now(),
        )
        .await;

        assert_eq!(output.query, "related memory");
        assert_eq!(output.limit, 5);
        assert!(!output.truncated);
        assert!(output.hits.is_empty());
        assert_eq!(
            output.message,
            "memory search unavailable for this activation"
        );
        assert_eq!(store.queries.borrow().len(), 1);
    }

    #[test]
    fn append_recombination_only_prepares_one_entry() {
        let mut appended = false;
        let (first_entry, first_output) = prepare_recombination_append(
            AppendRecombinationArgs {
                text: "  associative scenario  ".to_owned(),
            },
            &mut appended,
        );
        let (second_entry, second_output) = prepare_recombination_append(
            AppendRecombinationArgs {
                text: "another scenario".to_owned(),
            },
            &mut appended,
        );

        assert!(first_output.appended);
        assert_eq!(
            first_entry,
            Some("Internal dream simulation, not a verified fact: associative scenario".to_owned())
        );
        assert!(!second_output.appended);
        assert_eq!(second_entry, None);
    }

    #[test]
    fn recombination_context_bounds_recent_cognition_entries() {
        let record = nuillu_blackboard::CognitionLogEntryRecord {
            index: 0,
            source: nuillu_types::ModuleInstanceId::new(
                builtin::cognition_gate(),
                nuillu_types::ReplicaIndex::ZERO,
            ),
            entry: nuillu_blackboard::CognitionLogEntry {
                at: now(),
                text: format!("topic {}", "detail ".repeat(200)),
            },
        };

        let context = format_recombination_context(&[record]);

        assert!(context.contains("Recent non-dream cognition seeds:"));
        assert!(
            context.contains("Use search_memory only if targeted memory fragments would help.")
        );
        assert!(context.len() < 1_000);
    }
}
