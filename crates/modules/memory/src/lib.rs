use anyhow::{Context, Result};
use async_trait::async_trait;
use lutum::{Session, TextStepOutcomeWithTools, ToolResult};
use nuillu_module::{
    AllocationReader, BlackboardReader, CognitionLogUpdatedInbox, LlmAccess, MemoryRequest,
    MemoryRequestInbox, MemoryWriter, Module, SessionCompactionConfig, compact_session_if_needed,
    push_unread_memo_logs,
};
use nuillu_types::MemoryRank;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

mod batch;
pub use batch::NextBatch as MemoryBatch;

const SYSTEM_PROMPT: &str = r#"You are the memory module.
Inspect the current cognitive workspace and decide whether to preserve short, useful memories.
Use the current cognition log plus unread/recent module memo logs as candidate evidence. MemoryRequest
messages are explicit preservation candidates from other modules, not write commands. You may
reject, normalize, merge, and deduplicate observations and requests. Use insert_memory only for
concrete information likely to matter later."#;

const NORMAL_REQUEST_DECAY_SECS: i64 = 86_400;
const HIGH_REQUEST_DECAY_SECS: i64 = 604_800;
const COMPACTED_MEMORY_SESSION_PREFIX: &str = "Compacted memory session history:";
const SESSION_COMPACTION_PROMPT: &str = r#"You compact the memory module's persistent session history.
Summarize only the prefix transcript you receive. Preserve memo-log facts, memory requests,
inserted memory content, rejected candidates, deduplication decisions, and relevant cognition-log
context future memory decisions need. Do not invent facts. Return plain text only."#;

#[lutum::tool_input(name = "insert_memory", output = InsertMemoryOutput)]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct InsertMemoryArgs {
    pub content: String,
    pub rank: MemoryRank,
    pub decay_secs: i64,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct InsertMemoryOutput {
    pub index: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema, lutum::Toolset)]
pub enum MemoryTools {
    InsertMemory(InsertMemoryArgs),
}

pub struct MemoryModule {
    owner: nuillu_types::ModuleId,
    cognition_updates: CognitionLogUpdatedInbox,
    requests: MemoryRequestInbox,
    allocation: AllocationReader,
    blackboard: BlackboardReader,
    memory: MemoryWriter,
    llm: LlmAccess,
    session: Session,
    session_compaction: SessionCompactionConfig,
    system_prompt: std::sync::OnceLock<String>,
}

impl MemoryModule {
    pub fn new(
        cognition_updates: CognitionLogUpdatedInbox,
        requests: MemoryRequestInbox,
        allocation: AllocationReader,
        blackboard: BlackboardReader,
        memory: MemoryWriter,
        llm: LlmAccess,
    ) -> Self {
        Self {
            owner: nuillu_types::ModuleId::new(<Self as Module>::id()).expect("memory id is valid"),
            cognition_updates,
            requests,
            allocation,
            blackboard,
            memory,
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
    async fn activate(
        &mut self,
        cx: &nuillu_module::ActivateCx<'_>,
        requests: Vec<MemoryRequest>,
        cognition_updated: bool,
    ) -> Result<()> {
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
                "memory_requests": requests,
                "cognition_updated": cognition_updated,
            })
            .to_string(),
        );
        self.session.push_ephemeral_user(
            serde_json::json!({
                "blackboard": snapshot,
                "allocation": allocation,
                "request_policy": {
                    "normal_request_default_decay_secs": NORMAL_REQUEST_DECAY_SECS,
                    "high_request_default_decay_secs": HIGH_REQUEST_DECAY_SECS,
                    "explicit_requests_are_candidates": true,
                    "deduplication_and_rejection_allowed": true,
                },
            })
            .to_string(),
        );

        for _ in 0..4 {
            let outcome = self
                .session
                .text_turn(&self.llm.lutum().await)
                .tools::<MemoryTools>()
                .available_tools([MemoryToolsSelector::InsertMemory])
                .collect()
                .await
                .context("memory text turn failed")?;

            match outcome {
                TextStepOutcomeWithTools::Finished(result) => {
                    compact_session_if_needed(
                        &mut self.session,
                        result.usage.input_tokens,
                        cx.session_compaction_lutum(),
                        self.session_compaction,
                        Self::id(),
                        COMPACTED_MEMORY_SESSION_PREFIX,
                        SESSION_COMPACTION_PROMPT,
                    )
                    .await;
                    return Ok(());
                }
                TextStepOutcomeWithTools::FinishedNoOutput(result) => {
                    compact_session_if_needed(
                        &mut self.session,
                        result.usage.input_tokens,
                        cx.session_compaction_lutum(),
                        self.session_compaction,
                        Self::id(),
                        COMPACTED_MEMORY_SESSION_PREFIX,
                        SESSION_COMPACTION_PROMPT,
                    )
                    .await;
                    return Ok(());
                }
                TextStepOutcomeWithTools::NeedsTools(round) => {
                    let input_tokens = round.usage.input_tokens;
                    let mut results: Vec<ToolResult> = Vec::new();
                    for call in round.tool_calls.iter().cloned() {
                        let MemoryToolsCall::InsertMemory(call) = call;
                        let output = self
                            .insert_memory(call.input.clone())
                            .await
                            .context("run insert_memory tool")?;
                        let result = call
                            .complete(output)
                            .context("complete insert_memory tool call")?;
                        results.push(result);
                    }
                    round
                        .commit(&mut self.session, results)
                        .context("commit memory tool round")?;
                    compact_session_if_needed(
                        &mut self.session,
                        input_tokens,
                        cx.session_compaction_lutum(),
                        self.session_compaction,
                        Self::id(),
                        COMPACTED_MEMORY_SESSION_PREFIX,
                        SESSION_COMPACTION_PROMPT,
                    )
                    .await;
                }
            }
        }
        Ok(())
    }

    async fn insert_memory(&self, args: InsertMemoryArgs) -> Result<InsertMemoryOutput> {
        let index = self
            .memory
            .insert(args.content, args.rank, args.decay_secs)
            .await
            .context("insert memory")?;
        Ok(InsertMemoryOutput {
            index: index.to_string(),
        })
    }
}

#[async_trait(?Send)]
impl Module for MemoryModule {
    type Batch = MemoryBatch;

    fn id() -> &'static str {
        "memory"
    }

    fn role_description() -> &'static str {
        "Preserves useful information by inserting normalized, deduplicated memory entries from cognition-log evidence and surprise-driven preservation requests."
    }

    async fn next_batch(&mut self) -> Result<Self::Batch> {
        MemoryModule::next_batch(self).await
    }

    async fn activate(
        &mut self,
        cx: &nuillu_module::ActivateCx<'_>,
        batch: &Self::Batch,
    ) -> Result<()> {
        MemoryModule::activate(self, cx, batch.requests.clone(), batch.cognition_updated).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::cell::RefCell;
    use std::rc::Rc;
    use std::sync::{Arc, atomic::AtomicU64};

    use lutum::{
        FinishReason, Lutum, MockLlmAdapter, MockTextScenario, RawTextTurnEvent,
        SharedPoolBudgetManager, SharedPoolBudgetOptions, Usage,
    };
    use nuillu_blackboard::{Bpm, linear_ratio_fn};
    use nuillu_module::ports::{
        IndexedMemory, MemoryQuery, MemoryRecord, MemoryStore, NewMemory,
        NoopCognitionLogRepository, NoopFileSearchProvider, NoopUtteranceSink, PortError,
        SystemClock,
    };
    use nuillu_module::{
        CapabilityProviderPorts, CapabilityProviders, CognitionLogUpdated, LutumTiers,
        MemoryImportance, MemoryRequestMailbox, ModuleRegistry,
    };
    use nuillu_types::{MemoryIndex, MemoryRank, ModuleInstanceId, ReplicaIndex, builtin};
    use tokio::sync::Mutex;
    use tokio::task::LocalSet;

    fn test_caps_with_adapter(
        primary: Arc<dyn MemoryStore>,
        adapter: MockLlmAdapter,
    ) -> CapabilityProviders {
        let adapter = Arc::new(adapter);
        let budget = SharedPoolBudgetManager::new(SharedPoolBudgetOptions::default());
        let lutum = Lutum::new(adapter, budget);
        CapabilityProviders::new(CapabilityProviderPorts {
            blackboard: nuillu_blackboard::Blackboard::default(),
            cognition_log_port: Arc::new(NoopCognitionLogRepository),
            primary_memory_store: primary,
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

    #[derive(Default, Clone)]
    struct RecordingMemoryStore {
        inserted: Arc<Mutex<Vec<MemoryIndex>>>,
        next_index: Arc<AtomicU64>,
    }

    impl RecordingMemoryStore {
        async fn inserted_indexes(&self) -> Vec<MemoryIndex> {
            self.inserted.lock().await.clone()
        }
    }

    #[async_trait(?Send)]
    impl MemoryStore for RecordingMemoryStore {
        async fn insert(&self, _mem: NewMemory) -> Result<MemoryIndex, PortError> {
            let id = self
                .next_index
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let index = MemoryIndex::new(format!("recorded-{id}"));
            self.inserted.lock().await.push(index.clone());
            Ok(index)
        }
        async fn put(&self, _mem: IndexedMemory) -> Result<(), PortError> {
            Ok(())
        }
        async fn compact(
            &self,
            mem: NewMemory,
            _sources: &[MemoryIndex],
        ) -> Result<MemoryIndex, PortError> {
            self.insert(mem).await
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
            Ok(Vec::new())
        }
        async fn delete(&self, _index: &MemoryIndex) -> Result<(), PortError> {
            Ok(())
        }
    }

    struct PublisherStub;

    #[async_trait(?Send)]
    impl Module for PublisherStub {
        type Batch = ();

        fn id() -> &'static str {
            "surprise"
        }

        fn role_description() -> &'static str {
            "test stub"
        }

        async fn next_batch(&mut self) -> Result<Self::Batch> {
            std::future::pending().await
        }

        async fn activate(
            &mut self,
            _cx: &nuillu_module::ActivateCx<'_>,
            _batch: &Self::Batch,
        ) -> Result<()> {
            Ok(())
        }
    }

    fn memory_insert_tool_scenario(content: &str) -> MockTextScenario {
        MockTextScenario::events(vec![
            Ok(RawTextTurnEvent::Started {
                request_id: Some("memory-tool".into()),
                model: "mock".into(),
            }),
            Ok(RawTextTurnEvent::ToolCallChunk {
                id: "insert-1".into(),
                name: "insert_memory".into(),
                arguments_json_delta: format!(
                    r#"{{"content":"{content}","rank":"ShortTerm","decay_secs":86400}}"#
                ),
            }),
            Ok(RawTextTurnEvent::Completed {
                request_id: Some("memory-tool".into()),
                finish_reason: FinishReason::ToolCall,
                usage: Usage::zero(),
            }),
        ])
    }

    fn final_text_scenario(text: &str) -> MockTextScenario {
        MockTextScenario::events(vec![
            Ok(RawTextTurnEvent::Started {
                request_id: Some("memory-final".into()),
                model: "mock".into(),
            }),
            Ok(RawTextTurnEvent::TextDelta { delta: text.into() }),
            Ok(RawTextTurnEvent::Completed {
                request_id: Some("memory-final".into()),
                finish_reason: FinishReason::Stop,
                usage: Usage::zero(),
            }),
        ])
    }

    fn test_bpm() -> std::ops::RangeInclusive<Bpm> {
        Bpm::from_f64(60.0)..=Bpm::from_f64(60.0)
    }

    async fn build_memory(caps: &CapabilityProviders) -> nuillu_module::AllocatedModules {
        ModuleRegistry::new()
            .register(1..=1, test_bpm(), linear_ratio_fn, |caps| {
                MemoryModule::new(
                    caps.cognition_log_updated_inbox(),
                    caps.memory_request_inbox(),
                    caps.allocation_reader(),
                    caps.blackboard_reader(),
                    caps.memory_writer(),
                    caps.llm_access(),
                )
            })
            .unwrap()
            .build(caps)
            .await
            .unwrap()
    }

    async fn build_memory_with_publisher(
        caps: &CapabilityProviders,
    ) -> (nuillu_module::AllocatedModules, MemoryRequestMailbox) {
        let publisher_cell: Rc<RefCell<Option<MemoryRequestMailbox>>> = Rc::new(RefCell::new(None));
        let publisher_clone = Rc::clone(&publisher_cell);
        let modules = ModuleRegistry::new()
            .register(0..=0, test_bpm(), linear_ratio_fn, move |caps| {
                *publisher_clone.borrow_mut() = Some(caps.memory_request_mailbox());
                PublisherStub
            })
            .unwrap()
            .register(1..=1, test_bpm(), linear_ratio_fn, |caps| {
                MemoryModule::new(
                    caps.cognition_log_updated_inbox(),
                    caps.memory_request_inbox(),
                    caps.allocation_reader(),
                    caps.blackboard_reader(),
                    caps.memory_writer(),
                    caps.llm_access(),
                )
            })
            .unwrap()
            .build(caps)
            .await
            .unwrap();
        let publisher = publisher_cell
            .borrow_mut()
            .take()
            .expect("publisher captured");
        (modules, publisher)
    }

    async fn run_modules<F: std::future::Future<Output = ()>>(
        modules: nuillu_module::AllocatedModules,
        body: F,
    ) {
        nuillu_agent::run(
            modules,
            nuillu_agent::AgentEventLoopConfig {
                idle_threshold: std::time::Duration::from_millis(50),
                activate_retries: 2,
            },
            body,
        )
        .await
        .expect("memory test runtime should not fail");
    }

    #[tokio::test]
    async fn cognition_log_update_does_not_insert_without_llm_tool_decision() {
        let local = LocalSet::new();
        local
            .run_until(async {
                let primary = RecordingMemoryStore::default();
                let adapter = MockLlmAdapter::new().with_text_scenario(final_text_scenario("done"));
                let caps = test_caps_with_adapter(Arc::new(primary.clone()), adapter);
                let modules = build_memory(&caps).await;
                let cognition_log = caps.internal_harness_io().cognition_log_updated_mailbox();

                run_modules(modules, async {
                    cognition_log
                        .publish(CognitionLogUpdated::EntryAppended {
                            source: ModuleInstanceId::new(
                                builtin::cognition_gate(),
                                ReplicaIndex::ZERO,
                            ),
                        })
                        .await
                        .expect("memory module subscribed");

                    for _ in 0..10 {
                        tokio::task::yield_now().await;
                    }
                })
                .await;

                assert!(primary.inserted_indexes().await.is_empty());
            })
            .await;
    }

    #[tokio::test]
    async fn cognition_log_update_allows_llm_to_insert_memory() {
        let local = LocalSet::new();
        local
            .run_until(async {
                let primary = RecordingMemoryStore::default();
                let adapter = MockLlmAdapter::new()
                    .with_text_scenario(memory_insert_tool_scenario(
                        "Ryo wants cognition-driven memory implemented.",
                    ))
                    .with_text_scenario(final_text_scenario("memory-complete"));
                let caps = test_caps_with_adapter(Arc::new(primary.clone()), adapter);
                let modules = build_memory(&caps).await;
                let cognition_log = caps.internal_harness_io().cognition_log_updated_mailbox();

                run_modules(modules, async {
                    cognition_log
                        .publish(CognitionLogUpdated::EntryAppended {
                            source: ModuleInstanceId::new(
                                builtin::cognition_gate(),
                                ReplicaIndex::ZERO,
                            ),
                        })
                        .await
                        .expect("memory module subscribed");

                    for _ in 0..20 {
                        if primary.inserted_indexes().await.len() == 1 {
                            break;
                        }
                        tokio::task::yield_now().await;
                    }
                })
                .await;

                let indexes = primary.inserted_indexes().await;
                assert_eq!(indexes.len(), 1);
            })
            .await;
    }

    #[tokio::test]
    async fn memory_request_allows_surprise_to_trigger_insert_decision() {
        let local = LocalSet::new();
        local
            .run_until(async {
                let primary = RecordingMemoryStore::default();
                let adapter = MockLlmAdapter::new()
                    .with_text_scenario(memory_insert_tool_scenario(
                        "Koro stiffened at the food bowl.",
                    ))
                    .with_text_scenario(final_text_scenario("memory-complete"));
                let caps = test_caps_with_adapter(Arc::new(primary.clone()), adapter);
                let (modules, publisher) = build_memory_with_publisher(&caps).await;

                run_modules(modules, async {
                    publisher
                        .publish(MemoryRequest {
                            content: "Koro stiffened at the food bowl.".into(),
                            importance: MemoryImportance::High,
                            reason: "surprising peer food-guarding posture".into(),
                        })
                        .await
                        .expect("memory module subscribed");

                    for _ in 0..20 {
                        if primary.inserted_indexes().await.len() == 1 {
                            break;
                        }
                        tokio::task::yield_now().await;
                    }
                })
                .await;

                let indexes = primary.inserted_indexes().await;
                assert_eq!(indexes.len(), 1);
            })
            .await;
    }
}
