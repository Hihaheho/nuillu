use anyhow::{Context, Result};
use async_trait::async_trait;
use lutum::{Session, TextStepOutcomeWithTools, ToolResult};
use nuillu_module::{
    ActivationGate, BlackboardReader, LlmAccess, MemoryRequest, MemoryRequestInbox, MemoryWriter,
    Module, PeriodicInbox,
};
use nuillu_types::MemoryRank;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

mod batch;

const SYSTEM_PROMPT: &str = r#"You are the memory module.
Inspect the current cognitive workspace and decide whether to preserve short, useful memories.
MemoryRequest messages are candidates from other modules, not write commands. Evaluate them before
any periodic scan work. You may reject, normalize, or merge candidates, including deduplicating
multiple requests in the same batch. Use insert_memory only for concrete information likely to
matter later."#;

const NORMAL_REQUEST_DECAY_SECS: i64 = 86_400;
const HIGH_REQUEST_DECAY_SECS: i64 = 604_800;

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
    periodic: PeriodicInbox,
    requests: MemoryRequestInbox,
    gate: ActivationGate,
    blackboard: BlackboardReader,
    memory: MemoryWriter,
    llm: LlmAccess,
}

impl MemoryModule {
    pub fn new(
        periodic: PeriodicInbox,
        requests: MemoryRequestInbox,
        gate: ActivationGate,
        blackboard: BlackboardReader,
        memory: MemoryWriter,
        llm: LlmAccess,
    ) -> Self {
        Self {
            periodic,
            requests,
            gate,
            blackboard,
            memory,
            llm,
        }
    }

    #[tracing::instrument(skip_all, err(Debug, level = "warn"))]
    async fn activate(&self, requests: Vec<MemoryRequest>, periodic_scan: bool) -> Result<()> {
        if requests.is_empty() && !periodic_scan {
            return Ok(());
        }
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
                "blackboard": snapshot,
                "memory_requests": requests,
                "periodic_scan": periodic_scan,
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
            let outcome = session
                .text_turn()
                .tools::<MemoryTools>()
                .available_tools([MemoryToolsSelector::InsertMemory])
                .collect()
                .await
                .context("memory text turn failed")?;

            match outcome {
                TextStepOutcomeWithTools::Finished(_) => return Ok(()),
                TextStepOutcomeWithTools::NeedsTools(round) => {
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
                        .commit(&mut session, results)
                        .context("commit memory tool round")?;
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

    async fn run_loop(&mut self) -> Result<()> {
        loop {
            let batch = self.next_batch().await?;
            let _ = self.activate(batch.requests, batch.periodic).await;
        }
    }
}

#[async_trait(?Send)]
impl Module for MemoryModule {
    async fn run(&mut self) {
        if let Err(error) = self.run_loop().await {
            tracing::debug!(?error, "memory module loop stopped");
        }
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
    use nuillu_module::ports::{
        IndexedMemory, MemoryQuery, MemoryRecord, MemoryStore, NewMemory, NoopAttentionRepository,
        NoopFileSearchProvider, NoopUtteranceSink, PortError, SystemClock,
    };
    use nuillu_module::{CapabilityProviders, LutumTiers, MemoryImportance, ModuleRegistry};
    use nuillu_types::{MemoryIndex, builtin};
    use tokio::sync::Mutex;
    use tokio::task::LocalSet;

    fn test_caps_with_adapter(
        primary: Arc<dyn MemoryStore>,
        adapter: MockLlmAdapter,
    ) -> CapabilityProviders {
        let adapter = Arc::new(adapter);
        let budget = SharedPoolBudgetManager::new(SharedPoolBudgetOptions::default());
        let lutum = Lutum::new(adapter, budget);
        CapabilityProviders::new(
            nuillu_blackboard::Blackboard::default(),
            Arc::new(NoopAttentionRepository),
            primary,
            Vec::new(),
            Arc::new(NoopFileSearchProvider),
            Arc::new(NoopUtteranceSink),
            Arc::new(SystemClock),
            LutumTiers {
                cheap: lutum.clone(),
                default: lutum.clone(),
                premium: lutum,
            },
        )
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
        async fn run(&mut self) {}
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

    async fn build_memory_with_publisher(
        caps: &CapabilityProviders,
    ) -> (Vec<Box<dyn Module>>, nuillu_module::MemoryRequestMailbox) {
        let publisher_cell: Rc<RefCell<Option<nuillu_module::MemoryRequestMailbox>>> =
            Rc::new(RefCell::new(None));
        let publisher_clone = Rc::clone(&publisher_cell);
        let modules = ModuleRegistry::new()
            .register(builtin::surprise(), 0..=1, move |caps| {
                *publisher_clone.borrow_mut() = Some(caps.memory_request_mailbox());
                PublisherStub
            })
            .unwrap()
            .register(builtin::memory(), 0..=1, |caps| {
                MemoryModule::new(
                    caps.periodic_inbox(),
                    caps.memory_request_inbox(),
                    caps.activation_gate(),
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
        (modules.into_modules().collect(), publisher)
    }

    async fn run_modules<F: std::future::Future<Output = ()>>(
        modules: Vec<Box<dyn Module>>,
        body: F,
    ) {
        for mut module in modules {
            tokio::task::spawn_local(async move {
                module.run().await;
            });
        }
        body.await;
    }

    #[tokio::test]
    async fn memory_request_does_not_insert_without_llm_tool_decision() {
        let local = LocalSet::new();
        local
            .run_until(async {
                let primary = RecordingMemoryStore::default();
                let caps = test_caps_with_adapter(Arc::new(primary.clone()), MockLlmAdapter::new());
                let (modules, publisher) = build_memory_with_publisher(&caps).await;

                run_modules(modules, async {
                    publisher
                        .publish(MemoryRequest {
                            content: "candidate memory".into(),
                            importance: MemoryImportance::High,
                            reason: "test request".into(),
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
    async fn memory_request_batch_allows_llm_to_consolidate_to_one_insert() {
        let local = LocalSet::new();
        local
            .run_until(async {
                let primary = RecordingMemoryStore::default();
                let adapter = MockLlmAdapter::new()
                    .with_text_scenario(memory_insert_tool_scenario(
                        "Ryo wants inbox batching implemented.",
                    ))
                    .with_text_scenario(final_text_scenario("memory-complete"));
                let caps = test_caps_with_adapter(Arc::new(primary.clone()), adapter);
                let (modules, publisher) = build_memory_with_publisher(&caps).await;

                run_modules(modules, async {
                    publisher
                        .publish(MemoryRequest {
                            content: "Ryo wants inbox batching implemented".into(),
                            importance: MemoryImportance::Normal,
                            reason: "first signal".into(),
                        })
                        .await
                        .expect("memory module subscribed");
                    publisher
                        .publish(MemoryRequest {
                            content: "Inbox batching implementation is important to Ryo".into(),
                            importance: MemoryImportance::High,
                            reason: "duplicate signal".into(),
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
