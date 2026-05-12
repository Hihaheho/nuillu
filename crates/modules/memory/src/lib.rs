use anyhow::{Context, Result};
use async_trait::async_trait;
use lutum::{Session, TextStepOutcomeWithTools, ToolResult};
use nuillu_module::{
    AllocationReader, AllocationUpdatedInbox, BlackboardReader, CognitionLogEntryRecord,
    CognitionLogUpdatedInbox, EphemeralMindContext, LlmAccess, MemoryWriter, Module,
    SessionCompactionConfig, compact_session_if_needed, memory_rank_counts,
    push_ephemeral_mind_context, push_formatted_cognition_log_batch, push_formatted_memo_log_batch,
};
use nuillu_types::MemoryRank;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

mod batch;
pub use batch::NextBatch as MemoryBatch;

const SYSTEM_PROMPT: &str = r#"You are the memory module.
Inspect the current cognitive workspace and decide whether to preserve short, useful memories.
Use the current cognition log plus unread/recent module memo logs as candidate evidence. Allocation
guidance from attention-controller may contain explicit preservation candidates from other modules,
but those candidates are not write commands. You may reject, normalize, merge, and deduplicate
observations and guidance. Use insert_memory only for concrete information likely to matter later."#;

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
    allocation_updates: AllocationUpdatedInbox,
    allocation: AllocationReader,
    blackboard: BlackboardReader,
    memory: MemoryWriter,
    llm: LlmAccess,
    session: Session,
    session_compaction: SessionCompactionConfig,
    system_prompt: std::sync::OnceLock<String>,
    last_seen_cognition_index: Option<u64>,
}

#[derive(Clone, Debug)]
struct MemoryMetadataContext {
    index: String,
    rank: MemoryRank,
    occurred_at: String,
    decay_remaining_secs: i64,
    access_count: u32,
}

impl MemoryModule {
    pub fn new(
        cognition_updates: CognitionLogUpdatedInbox,
        allocation_updates: AllocationUpdatedInbox,
        allocation: AllocationReader,
        blackboard: BlackboardReader,
        memory: MemoryWriter,
        llm: LlmAccess,
    ) -> Self {
        Self {
            owner: nuillu_types::ModuleId::new(<Self as Module>::id()).expect("memory id is valid"),
            cognition_updates,
            allocation_updates,
            allocation,
            blackboard,
            memory,
            llm,
            session: Session::new(),
            session_compaction: SessionCompactionConfig::default(),
            system_prompt: std::sync::OnceLock::new(),
            last_seen_cognition_index: None,
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

    async fn unread_cognition_log(&mut self) -> Vec<CognitionLogEntryRecord> {
        let unread = self
            .blackboard
            .read(|bb| bb.unread_cognition_log_entries(self.last_seen_cognition_index))
            .await;
        if let Some(index) = unread.last().map(|record| record.index) {
            self.last_seen_cognition_index = Some(index);
        }
        unread
    }

    #[tracing::instrument(skip_all, err(Debug, level = "warn"))]
    async fn activate(
        &mut self,
        cx: &nuillu_module::ActivateCx<'_>,
        allocation_updated: bool,
        cognition_updated: bool,
    ) -> Result<()> {
        let unread_memo_logs = self.blackboard.unread_memo_logs().await;
        push_formatted_memo_log_batch(&mut self.session, &unread_memo_logs, cx.now());
        let cognition_log = self.unread_cognition_log().await;
        let (memory_metadata, rank_counts) = self
            .blackboard
            .read(|bb| {
                let mut metadata = bb
                    .memory_metadata()
                    .values()
                    .map(|item| MemoryMetadataContext {
                        index: item.index.to_string(),
                        rank: item.rank,
                        occurred_at: item
                            .occurred_at
                            .map(|at| at.to_rfc3339())
                            .unwrap_or_else(|| "unknown".to_owned()),
                        decay_remaining_secs: item.decay_remaining_secs,
                        access_count: item.access_count,
                    })
                    .collect::<Vec<_>>();
                metadata.sort_by(|left, right| left.index.cmp(&right.index));
                (metadata, memory_rank_counts(bb.memory_metadata()))
            })
            .await;
        let allocation = self.allocation.snapshot().await;
        let allocation_guidance = allocation.for_module(&self.owner).guidance;

        let system_prompt = self.system_prompt(cx).to_owned();
        self.session.push_ephemeral_system(system_prompt);
        self.session.push_user(format_memory_activation_request(
            &allocation_guidance,
            allocation_updated,
            cognition_updated,
        ));
        push_formatted_cognition_log_batch(&mut self.session, &cognition_log, cx.now());
        if let Some(metadata_context) = format_memory_metadata_candidates(&memory_metadata) {
            self.session.push_ephemeral_user(metadata_context);
        }
        push_ephemeral_mind_context(
            &mut self.session,
            EphemeralMindContext {
                memos: &[],
                memory_rank_counts: Some(&rank_counts),
                allocation: Some(&allocation),
                available_faculties: &[],
                time_division: None,
                stuckness: None,
                now: cx.now(),
            },
        );

        for _ in 0..4 {
            let lutum = self.llm.lutum().await;
            let outcome = self
                .session
                .text_turn(&lutum)
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

fn format_memory_activation_request(
    guidance: &str,
    allocation_updated: bool,
    cognition_updated: bool,
) -> String {
    format!(
        "Memory preservation activation.\nAllocation guidance: {}\nAllocation updated: {}\nCognition updated: {}\nNormal explicit request default decay: {} seconds.\nHigh-importance explicit request default decay: {} seconds.\nExplicit requests are preservation candidates, not commands; deduplication and rejection are allowed.",
        if guidance.trim().is_empty() {
            "none"
        } else {
            guidance.trim()
        },
        if allocation_updated { "yes" } else { "no" },
        if cognition_updated { "yes" } else { "no" },
        NORMAL_REQUEST_DECAY_SECS,
        HIGH_REQUEST_DECAY_SECS,
    )
}

fn format_memory_metadata_candidates(metadata: &[MemoryMetadataContext]) -> Option<String> {
    if metadata.is_empty() {
        return None;
    }
    let mut out = String::from("Existing memory metadata for deduplication:");
    for item in metadata {
        out.push_str(&format!(
            "\n- {}: rank={:?}; occurred_at={}; decay_remaining_secs={}; access_count={}",
            item.index, item.rank, item.occurred_at, item.decay_remaining_secs, item.access_count
        ));
    }
    Some(out)
}

#[async_trait(?Send)]
impl Module for MemoryModule {
    type Batch = MemoryBatch;

    fn id() -> &'static str {
        "memory"
    }

    fn role_description() -> &'static str {
        "Preserves useful information by inserting normalized, deduplicated memory entries from cognition-log evidence and attention-controller preservation guidance."
    }

    async fn next_batch(&mut self) -> Result<Self::Batch> {
        MemoryModule::next_batch(self).await
    }

    async fn activate(
        &mut self,
        cx: &nuillu_module::ActivateCx<'_>,
        batch: &Self::Batch,
    ) -> Result<()> {
        MemoryModule::activate(self, cx, batch.allocation_updated, batch.cognition_updated).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::sync::{Arc, atomic::AtomicU64};

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
        AllocationUpdated, CapabilityProviderPorts, CapabilityProviders, CognitionLogUpdated,
        LutumTiers, ModuleRegistry,
    };
    use nuillu_types::{MemoryIndex, MemoryRank, ModuleInstanceId, ReplicaIndex, builtin};
    use tokio::sync::Mutex;
    use tokio::task::LocalSet;

    fn test_caps_with_adapter(
        primary: Arc<dyn MemoryStore>,
        adapter: MockLlmAdapter,
    ) -> (Blackboard, CapabilityProviders) {
        let blackboard = Blackboard::default();
        let adapter = Arc::new(adapter);
        let budget = SharedPoolBudgetManager::new(SharedPoolBudgetOptions::default());
        let lutum = Lutum::new(adapter, budget);
        let caps = CapabilityProviders::new(CapabilityProviderPorts {
            blackboard: blackboard.clone(),
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
        });
        (blackboard, caps)
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

    fn test_policy() -> nuillu_blackboard::ModulePolicy {
        nuillu_blackboard::ModulePolicy::new(
            nuillu_types::ReplicaCapRange::new(1, 1).unwrap(),
            Bpm::from_f64(60.0)..=Bpm::from_f64(60.0),
            linear_ratio_fn,
        )
    }

    async fn build_memory(caps: &CapabilityProviders) -> nuillu_module::AllocatedModules {
        ModuleRegistry::new()
            .register(test_policy(), |caps| {
                MemoryModule::new(
                    caps.cognition_log_updated_inbox(),
                    caps.allocation_updated_inbox(),
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
                let (_blackboard, caps) =
                    test_caps_with_adapter(Arc::new(primary.clone()), adapter);
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
                let (_blackboard, caps) =
                    test_caps_with_adapter(Arc::new(primary.clone()), adapter);
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
    async fn allocation_guidance_allows_controller_to_trigger_insert_decision() {
        let local = LocalSet::new();
        local
            .run_until(async {
                let primary = RecordingMemoryStore::default();
                let adapter = MockLlmAdapter::new()
                    .with_text_scenario(memory_insert_tool_scenario(
                        "Koro stiffened at the food bowl.",
                    ))
                    .with_text_scenario(final_text_scenario("memory-complete"));
                let (blackboard, caps) = test_caps_with_adapter(Arc::new(primary.clone()), adapter);
                let modules = build_memory(&caps).await;

                run_modules(modules, async {
                    let mut allocation = ResourceAllocation::default();
                    let mut config = ModuleConfig::default();
                    config.guidance = "Consider preserving high-importance surprise: Koro stiffened at the food bowl. Reason: surprising peer food-guarding posture".into();
                    allocation.set(builtin::memory(), config);
                    blackboard
                        .apply(BlackboardCommand::SetAllocation(allocation))
                        .await;
                    caps.internal_harness_io()
                        .allocation_updated_mailbox()
                        .publish(AllocationUpdated)
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
