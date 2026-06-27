use anyhow::{Context, Result};
use async_trait::async_trait;
use lutum::{ModelInput, TextStepOutcomeWithTools, ToolResult};
use nuillu_module::{BlackboardReader, InteroceptiveUpdatedInbox, LlmAccess, Module};
use nuillu_types::{MemoryContent, MemoryIndex, MemoryRank};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::common::{GetMemoriesOutput, MemoryMetadataContext, memory_record_to_view};
use crate::memory::{
    MemoryConceptInput, MemoryTagInput, SHORT_TERM_MEMORY_DECAY_SECS, memory_concept_from_input,
    memory_tag_from_input,
};
use crate::store::{
    MemoryAssociator, MemoryContentReader, MemoryKind, MemoryLinkRelation, MemoryRecord,
    MemoryWriter, NewMemory, NewMemoryLink,
};

const SYSTEM_PROMPT: &str = r#"You are the memory-association module.
Inspect candidate memories, fetch source contents when useful, and write non-destructive memory
associations. Source memories remain live. Use get_association_memories to inspect sources. Use
write_association_summary when a reflection memory would help explain a group of source memories,
and write_memory_links when only direct memory-to-memory relationships are needed. Links can state
derived_from, updates, corrects, contradicts, supports, or related relationships justified by the
source memories. Do not delete, rewrite, or compact source memories.

The candidate list is the maintenance work item. When multiple candidates are present, inspect their
contents with get_association_memories before deciding whether to write a summary, write links, or
leave them unchanged.

Tool input rules:
- write_association_summary takes source_indexes, summary_content, concepts, and tags only.
  The runtime chooses rank and decay and automatically writes derived_from links from the summary
  to every source memory.
- write_memory_links link objects use from_index, to_index, and relation only. The runtime chooses
  link strength and confidence."#;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct AssociationLinkArgs {
    pub from_index: String,
    pub to_index: String,
    pub relation: MemoryLinkRelation,
}

#[lutum::tool_input(name = "get_association_memories", output = GetMemoriesOutput)]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct GetAssociationMemoriesArgs {
    pub indexes: Vec<String>,
}

#[lutum::tool_input(name = "write_association_summary", output = WriteAssociationSummaryOutput)]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct WriteAssociationSummaryArgs {
    pub source_indexes: Vec<String>,
    pub summary_content: String,
    #[serde(default)]
    pub concepts: Vec<MemoryConceptInput>,
    #[serde(default)]
    pub tags: Vec<MemoryTagInput>,
}

#[lutum::tool_input(name = "write_memory_links", output = WriteMemoryLinksOutput)]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct WriteMemoryLinksArgs {
    pub links: Vec<AssociationLinkArgs>,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct WriteAssociationSummaryOutput {
    pub summary_index: String,
    pub source_count: usize,
    pub links_written: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct WriteMemoryLinksOutput {
    pub links_written: usize,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema, lutum::Toolset)]
pub enum AssociationTools {
    GetAssociationMemories(GetAssociationMemoriesArgs),
    WriteAssociationSummary(WriteAssociationSummaryArgs),
    WriteMemoryLinks(WriteMemoryLinksArgs),
}

pub struct MemoryAssociationModule {
    owner: nuillu_types::ModuleId,
    interoception_updates: InteroceptiveUpdatedInbox,
    blackboard: BlackboardReader,
    reader: MemoryContentReader,
    writer: MemoryWriter,
    associator: MemoryAssociator,
    llm: LlmAccess,
    system_prompt: std::sync::OnceLock<String>,
}

impl MemoryAssociationModule {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        interoception_updates: InteroceptiveUpdatedInbox,
        blackboard: BlackboardReader,
        reader: MemoryContentReader,
        writer: MemoryWriter,
        associator: MemoryAssociator,
        llm: LlmAccess,
    ) -> Self {
        Self {
            owner: nuillu_types::ModuleId::new(<Self as Module>::id())
                .expect("memory-association id is valid"),
            interoception_updates,
            blackboard,
            reader,
            writer,
            associator,
            llm,
            system_prompt: std::sync::OnceLock::new(),
        }
    }

    fn system_prompt(&self, cx: &nuillu_module::ActivateCx<'_>) -> &str {
        self.system_prompt.get_or_init(|| {
            nuillu_module::format_system_seed(
                nuillu_module::format_system_prompt(
                    SYSTEM_PROMPT,
                    cx.peer_contexts(),
                    &self.owner,
                    cx.core_policies(),
                ),
                false,
                cx.identity_memories(),
                cx.now(),
            )
        })
    }

    #[tracing::instrument(skip_all, err(Debug, level = "warn"))]
    async fn activate(&self, cx: &nuillu_module::ActivateCx<'_>) -> Result<()> {
        let memory_metadata = self
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
                    })
                    .collect::<Vec<_>>();
                metadata.sort_by(|left, right| left.index.cmp(&right.index));
                metadata
            })
            .await;
        if memory_metadata.is_empty() {
            return Ok(());
        }

        let mut input = ModelInput::new()
            .system(self.system_prompt(cx))
            .user(format_association_context(&memory_metadata));

        for _ in 0..6 {
            let lutum = self.llm.lutum().await;
            let outcome = lutum
                .text_turn(input.clone())
                .tools::<AssociationTools>()
                .available_tools([
                    AssociationToolsSelector::GetAssociationMemories,
                    AssociationToolsSelector::WriteAssociationSummary,
                    AssociationToolsSelector::WriteMemoryLinks,
                ])
                .collect_controlled_with(nuillu_module::AbortOnAvailableToolNameInText::new())
                .await
                .context("memory-association text turn failed")?;

            match outcome {
                TextStepOutcomeWithTools::Finished(_) => return Ok(()),
                TextStepOutcomeWithTools::FinishedNoOutput(_) => return Ok(()),
                TextStepOutcomeWithTools::NeedsTools(round) => {
                    let mut results: Vec<ToolResult> = Vec::new();
                    nuillu_module::emit_trace_tool_calls(&round.tool_calls);
                    for call in round.tool_calls.iter().cloned() {
                        match call {
                            AssociationToolsCall::GetAssociationMemories(call) => {
                                let output = self
                                    .get_association_memories(call.input.clone())
                                    .await
                                    .context("run get_association_memories tool")?;
                                let result = call
                                    .complete(output)
                                    .context("complete get_association_memories tool call")?;
                                results.push(result);
                            }
                            AssociationToolsCall::WriteAssociationSummary(call) => {
                                let output = self
                                    .write_association_summary(call.input.clone())
                                    .await
                                    .context("run write_association_summary tool")?;
                                let result = call
                                    .complete(output)
                                    .context("complete write_association_summary tool call")?;
                                results.push(result);
                            }
                            AssociationToolsCall::WriteMemoryLinks(call) => {
                                let output = self
                                    .write_memory_links(call.input.clone())
                                    .await
                                    .context("run write_memory_links tool")?;
                                let result = call
                                    .complete(output)
                                    .context("complete write_memory_links tool call")?;
                                results.push(result);
                            }
                        }
                    }
                    round
                        .commit_into(&mut input, results)
                        .context("commit memory-association tool round")?;
                }
            }
        }
        Ok(())
    }

    async fn get_association_memories(
        &self,
        args: GetAssociationMemoriesArgs,
    ) -> Result<GetMemoriesOutput> {
        let indexes = args
            .indexes
            .into_iter()
            .map(MemoryIndex::new)
            .collect::<Vec<_>>();
        Ok(GetMemoriesOutput {
            memories: self
                .get_many(&indexes)
                .await?
                .into_iter()
                .map(memory_record_to_view)
                .collect(),
        })
    }

    async fn write_association_summary(
        &self,
        args: WriteAssociationSummaryArgs,
    ) -> Result<WriteAssociationSummaryOutput> {
        let sources = args
            .source_indexes
            .iter()
            .cloned()
            .map(MemoryIndex::new)
            .collect::<Vec<_>>();
        let source_records = self.get_many(&sources).await?;
        let occurred_at = source_records
            .iter()
            .filter_map(|record| record.occurred_at)
            .min();
        let record = self
            .writer
            .insert_entry(
                NewMemory {
                    content: MemoryContent::new(args.summary_content),
                    rank: MemoryRank::ShortTerm,
                    occurred_at,
                    kind: MemoryKind::Reflection,
                    concepts: args
                        .concepts
                        .into_iter()
                        .map(memory_concept_from_input)
                        .collect(),
                    tags: args.tags.into_iter().map(memory_tag_from_input).collect(),
                    affect_arousal: 0.0,
                    valence: 0.0,
                    emotion: String::new(),
                },
                SHORT_TERM_MEMORY_DECAY_SECS,
            )
            .await
            .context("write association summary memory")?;

        let links = sources
            .iter()
            .map(|source| AssociationLinkArgs {
                from_index: record.index.to_string(),
                to_index: source.to_string(),
                relation: MemoryLinkRelation::DerivedFrom,
            })
            .collect::<Vec<_>>();
        let links_written = self.upsert_links(links).await?;

        Ok(WriteAssociationSummaryOutput {
            summary_index: record.index.to_string(),
            source_count: source_records.len(),
            links_written,
        })
    }

    async fn write_memory_links(
        &self,
        args: WriteMemoryLinksArgs,
    ) -> Result<WriteMemoryLinksOutput> {
        Ok(WriteMemoryLinksOutput {
            links_written: self.upsert_links(args.links).await?,
        })
    }

    async fn upsert_links(&self, links: Vec<AssociationLinkArgs>) -> Result<usize> {
        let mut links_written = 0;
        for link in links {
            self.associator
                .upsert_link(NewMemoryLink {
                    from_memory: MemoryIndex::new(link.from_index),
                    to_memory: MemoryIndex::new(link.to_index),
                    relation: link.relation,
                    freeform_relation: None,
                    strength: 1.0,
                    confidence: 1.0,
                })
                .await
                .context("upsert association link")?;
            links_written += 1;
        }
        Ok(links_written)
    }

    async fn get_many(&self, indexes: &[MemoryIndex]) -> Result<Vec<MemoryRecord>> {
        let mut records = Vec::new();
        for index in indexes {
            if let Some(record) = self
                .reader
                .get(index)
                .await
                .with_context(|| format!("get memory {}", index.as_str()))?
            {
                records.push(record);
            }
        }
        Ok(records)
    }

    async fn next_batch(&mut self) -> Result<()> {
        let _ = self.interoception_updates.next_item().await?;
        let _ = self.interoception_updates.take_ready_items()?;
        Ok(())
    }
}

fn format_association_context(memory_metadata: &[MemoryMetadataContext]) -> String {
    let mut out = String::from("Memory association context.");
    out.push_str("\n\nMemory candidates:");
    if memory_metadata.is_empty() {
        out.push_str("\n- none");
    } else {
        for item in memory_metadata {
            out.push_str(&format!(
                "\n- {}: rank={:?}; occurred_at={}",
                item.index, item.rank, item.occurred_at
            ));
        }
    }
    out
}

#[async_trait(?Send)]
impl Module for MemoryAssociationModule {
    type Batch = ();

    fn id() -> &'static str {
        "memory-association"
    }

    fn peer_context() -> Option<&'static str> {
        None
    }

    async fn next_batch(&mut self) -> Result<Self::Batch> {
        MemoryAssociationModule::next_batch(self).await
    }

    async fn activate(
        &mut self,
        cx: &nuillu_module::ActivateCx<'_>,
        _batch: &Self::Batch,
    ) -> Result<()> {
        MemoryAssociationModule::activate(self, cx).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::cell::RefCell;
    use std::rc::Rc;
    use std::sync::Arc;

    use chrono::{TimeZone, Utc};
    use lutum::{
        FinishReason, Lutum, MockLlmAdapter, MockTextScenario, RawTextTurnEvent,
        SharedPoolBudgetManager, SharedPoolBudgetOptions, Usage,
    };
    use nuillu_blackboard::{Blackboard, BlackboardCommand, Bpm, MemoryMetaPatch, ModulePolicy};
    use nuillu_module::ports::{NoopCognitionLogRepository, PortError, SystemClock};
    use nuillu_module::{
        ActivateCx, CapabilityProviderConfig, CapabilityProviderPorts, CapabilityProviderRuntime,
        CapabilityProviders, InteroceptiveUpdated, LlmConcurrencyLimiter, LutumTiers,
        ModuleRegistry, RuntimeEvent, RuntimeEventSink, SessionCompactionPolicy,
        SessionCompactionRuntime,
    };
    use nuillu_types::{ModelTier, ReplicaCapRange};

    use crate::{MemoryCapabilities, NoopMemoryStore};

    #[derive(Clone, Default)]
    struct RecordingRuntimeEventSink {
        events: Rc<RefCell<Vec<RuntimeEvent>>>,
    }

    impl RecordingRuntimeEventSink {
        fn events(&self) -> Vec<RuntimeEvent> {
            self.events.borrow().clone()
        }
    }

    impl RuntimeEventSink for RecordingRuntimeEventSink {
        fn on_event(&self, event: RuntimeEvent) -> Result<(), PortError> {
            self.events.borrow_mut().push(event);
            Ok(())
        }
    }

    fn test_policy() -> ModulePolicy {
        ModulePolicy::new(
            ReplicaCapRange::new(1, 1).unwrap(),
            Bpm::from_f64(60_000.0)..=Bpm::from_f64(60_000.0),
            nuillu_blackboard::linear_ratio_fn,
        )
    }

    fn test_caps_with_adapter(
        blackboard: Blackboard,
        adapter: MockLlmAdapter,
    ) -> (CapabilityProviders, Lutum, RecordingRuntimeEventSink) {
        let sink = RecordingRuntimeEventSink::default();
        let budget = SharedPoolBudgetManager::new(SharedPoolBudgetOptions::default());
        let lutum = Lutum::new(Arc::new(adapter), budget);
        let caps = CapabilityProviders::new(CapabilityProviderConfig {
            ports: CapabilityProviderPorts {
                blackboard,
                cognition_log_port: Rc::new(NoopCognitionLogRepository),
                clock: Rc::new(SystemClock),
                tiers: LutumTiers::from_shared_lutum(lutum.clone()),
            },
            runtime: CapabilityProviderRuntime {
                event_sink: Rc::new(sink.clone()),
                ..CapabilityProviderRuntime::default()
            },
        });
        (caps, lutum, sink)
    }

    fn activate_cx(lutum: &Lutum, now: chrono::DateTime<chrono::Utc>) -> ActivateCx<'static> {
        ActivateCx::new(
            &[],
            &[],
            &[],
            SessionCompactionRuntime::new(
                lutum.clone(),
                LlmConcurrencyLimiter::new(None),
                ModelTier::Cheap,
                SessionCompactionPolicy::default(),
            ),
            now,
        )
    }

    fn finished_text_scenario() -> MockTextScenario {
        MockTextScenario::events(vec![
            Ok(RawTextTurnEvent::Started {
                request_id: Some("memory-association-text".into()),
                model: "mock".into(),
            }),
            Ok(RawTextTurnEvent::TextDelta {
                delta: "no association changes".into(),
            }),
            Ok(RawTextTurnEvent::Completed {
                request_id: Some("memory-association-text".into()),
                finish_reason: FinishReason::Stop,
                usage: Usage {
                    input_tokens: 1,
                    ..Usage::zero()
                },
            }),
        ])
    }

    async fn build_memory_association_module(
        caps: &CapabilityProviders,
        memory_caps: MemoryCapabilities,
    ) -> nuillu_module::AllocatedModule {
        let modules = ModuleRegistry::new()
            .register(test_policy(), move |caps| {
                let memory_caps = memory_caps.clone();
                async move {
                    Ok(MemoryAssociationModule::new(
                        caps.interoception_updated_inbox(),
                        caps.blackboard_reader(),
                        memory_caps.content_reader(),
                        memory_caps.writer(),
                        memory_caps.associator(),
                        caps.llm("main")
                            .with_tier(nuillu_types::ModelTier::Cheap)
                            .into(),
                    ))
                }
            })
            .unwrap()
            .build(caps)
            .await
            .unwrap();
        let (_, mut modules) = modules.into_parts();
        modules.remove(0)
    }

    async fn activate_module_once(
        caps: &CapabilityProviders,
        lutum: &Lutum,
        module: &mut nuillu_module::AllocatedModule,
        now: chrono::DateTime<chrono::Utc>,
    ) {
        caps.internal_harness_io()
            .interoception_updated_mailbox()
            .publish(InteroceptiveUpdated)
            .await
            .unwrap();
        let batch = module.next_batch().await.unwrap();
        module
            .activate(&activate_cx(lutum, now), &batch)
            .await
            .unwrap();
    }

    #[tokio::test(flavor = "current_thread")]
    async fn activation_without_memory_candidates_does_not_access_llm() {
        let now = Utc.timestamp_opt(10, 0).unwrap();
        let blackboard = Blackboard::default();
        let (caps, lutum, events) =
            test_caps_with_adapter(blackboard.clone(), MockLlmAdapter::new());
        let memory_caps = MemoryCapabilities::new(
            blackboard,
            Rc::new(SystemClock),
            Rc::new(NoopMemoryStore),
            Vec::new(),
        );
        let mut module = build_memory_association_module(&caps, memory_caps).await;

        activate_module_once(&caps, &lutum, &mut module, now).await;

        assert!(
            !events
                .events()
                .iter()
                .any(|event| matches!(event, RuntimeEvent::LlmAccessed { .. }))
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn activation_with_memory_candidates_accesses_llm() {
        let now = Utc.timestamp_opt(10, 0).unwrap();
        let blackboard = Blackboard::default();
        blackboard
            .apply(BlackboardCommand::UpsertMemoryMetadata {
                index: MemoryIndex::new("candidate-memory"),
                rank_if_new: MemoryRank::LongTerm,
                occurred_at_if_new: Some(now),
                decay_if_new_secs: 60,
                now,
                patch: MemoryMetaPatch::default(),
            })
            .await;
        let (caps, lutum, events) = test_caps_with_adapter(
            blackboard.clone(),
            MockLlmAdapter::new().with_text_scenario(finished_text_scenario()),
        );
        let memory_caps = MemoryCapabilities::new(
            blackboard,
            Rc::new(SystemClock),
            Rc::new(NoopMemoryStore),
            Vec::new(),
        );
        let mut module = build_memory_association_module(&caps, memory_caps).await;

        activate_module_once(&caps, &lutum, &mut module, now).await;

        assert_eq!(
            events
                .events()
                .iter()
                .filter(|event| matches!(event, RuntimeEvent::LlmAccessed { .. }))
                .count(),
            1
        );
    }

    #[test]
    fn association_summary_schema_does_not_expose_runtime_metadata_or_links() {
        let schema = serde_json::to_value(schemars::schema_for!(WriteAssociationSummaryArgs))
            .expect("association summary schema should serialize");

        assert_eq!(schema.pointer("/properties/summary_rank"), None);
        assert_eq!(schema.pointer("/properties/decay_secs"), None);
        assert_eq!(schema.pointer("/properties/occurred_at"), None);
        assert_eq!(schema.pointer("/properties/links"), None);
        assert_eq!(
            schema.pointer("/properties/concepts/items/type"),
            Some(&serde_json::json!("string"))
        );
        assert_eq!(
            schema.pointer("/properties/tags/items/type"),
            Some(&serde_json::json!("string"))
        );
    }

    #[test]
    fn association_context_exposes_only_candidate_identity() {
        let context = format_association_context(&[MemoryMetadataContext {
            index: "mem-1".to_owned(),
            rank: MemoryRank::LongTerm,
            occurred_at: "2026-06-07T00:00:00Z".to_owned(),
        }]);

        assert_eq!(
            context,
            "Memory association context.\n\nMemory candidates:\n- mem-1: rank=LongTerm; occurred_at=2026-06-07T00:00:00Z"
        );
        assert!(!context.contains("guidance"));
        assert!(!context.contains("decay_remaining_secs"));
        assert!(!context.contains("access_count"));
        assert!(!context.contains("reinforcement_count"));
    }

    #[test]
    fn association_link_schema_does_not_expose_runtime_scores() {
        let schema = serde_json::to_value(schemars::schema_for!(WriteMemoryLinksArgs))
            .expect("memory link schema should serialize");

        assert_eq!(
            schema.pointer("/$defs/AssociationLinkArgs/properties/strength_percent"),
            None
        );
        assert_eq!(
            schema.pointer("/$defs/AssociationLinkArgs/properties/confidence_percent"),
            None
        );
        assert_eq!(
            schema.pointer("/$defs/AssociationLinkArgs/properties/freeform_relation"),
            None
        );
        assert_eq!(
            schema.pointer("/$defs/AssociationLinkArgs/properties/from_index/type"),
            Some(&serde_json::json!("string"))
        );
        assert_eq!(
            schema.pointer("/$defs/AssociationLinkArgs/properties/to_index/type"),
            Some(&serde_json::json!("string"))
        );
        assert_eq!(
            schema.pointer("/$defs/AssociationLinkArgs/properties/relation/$ref"),
            Some(&serde_json::json!("#/$defs/MemoryLinkRelation"))
        );
    }
}
