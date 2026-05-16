use anyhow::{Context, Result};
use async_trait::async_trait;
use lutum::{Session, TextStepOutcomeWithTools, ToolResult};
use nuillu_module::{
    AllocationReader, AllocationUpdatedInbox, BlackboardReader, LlmAccess, Module,
};
use nuillu_types::{MemoryContent, MemoryIndex, MemoryRank};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::common::{GetMemoriesOutput, MemoryMetadataContext, memory_record_to_view};
use crate::memory::{
    MemoryConceptInput, MemoryTagInput, confidence_percent_to_f32, memory_concept_from_input,
    memory_tag_from_input,
};
use crate::store::{
    MemoryAssociator, MemoryContentReader, MemoryKind, MemoryLinkRelation, MemoryRecord,
    MemoryWriter, NewMemory, NewMemoryLink,
};

const SYSTEM_PROMPT: &str = r#"You are the memory-association module.
Inspect memory metadata, fetch source contents when useful, and write non-destructive memory
associations. Source memories remain live. Use get_association_memories to inspect sources. Use
write_association_summary when a reflection memory would help explain a group of source memories,
and write_memory_links when only direct memory-to-memory relationships are needed. Links can state
derived_from, updates, corrects, contradicts, supports, or related relationships justified by the
source memories. Do not delete, rewrite, or compact source memories."#;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct AssociationLinkArgs {
    pub from_index: String,
    pub to_index: String,
    pub relation: MemoryLinkRelation,
    pub freeform_relation: Option<String>,
    pub strength_percent: u8,
    pub confidence_percent: u8,
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
    pub summary_rank: MemoryRank,
    pub decay_secs: i64,
    pub concepts: Vec<MemoryConceptInput>,
    pub tags: Vec<MemoryTagInput>,
    pub links: Vec<AssociationLinkArgs>,
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
    allocation_updates: AllocationUpdatedInbox,
    allocation: AllocationReader,
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
        allocation_updates: AllocationUpdatedInbox,
        allocation: AllocationReader,
        blackboard: BlackboardReader,
        reader: MemoryContentReader,
        writer: MemoryWriter,
        associator: MemoryAssociator,
        llm: LlmAccess,
    ) -> Self {
        Self {
            owner: nuillu_types::ModuleId::new(<Self as Module>::id())
                .expect("memory-association id is valid"),
            allocation_updates,
            allocation,
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
                        decay_remaining_secs: item.decay_remaining_secs,
                        access_count: item.access_count,
                        use_count: item.use_count,
                        reinforcement_count: item.reinforcement_count,
                    })
                    .collect::<Vec<_>>();
                metadata.sort_by(|left, right| left.index.cmp(&right.index));
                metadata
            })
            .await;
        let allocation = self.allocation.snapshot().await;
        let allocation_guidance = allocation
            .iter()
            .filter_map(|(id, config)| {
                let guidance = config.guidance.trim();
                (!guidance.is_empty()).then(|| (id.to_string(), guidance.to_owned()))
            })
            .collect::<Vec<_>>();

        let mut session = Session::new();
        session.push_system(self.system_prompt(cx));
        session.push_user(format_association_context(
            &memory_metadata,
            &allocation_guidance,
        ));

        for _ in 0..6 {
            let lutum = self.llm.lutum().await;
            let outcome = session
                .text_turn(&lutum)
                .tools::<AssociationTools>()
                .available_tools([
                    AssociationToolsSelector::GetAssociationMemories,
                    AssociationToolsSelector::WriteAssociationSummary,
                    AssociationToolsSelector::WriteMemoryLinks,
                ])
                .collect()
                .await
                .context("memory-association text turn failed")?;

            match outcome {
                TextStepOutcomeWithTools::Finished(_) => return Ok(()),
                TextStepOutcomeWithTools::FinishedNoOutput(_) => return Ok(()),
                TextStepOutcomeWithTools::NeedsTools(round) => {
                    let mut results: Vec<ToolResult> = Vec::new();
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
                        .commit(&mut session, results)
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
                    rank: args.summary_rank,
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
                args.decay_secs,
            )
            .await
            .context("write association summary memory")?;

        let mut links = sources
            .iter()
            .map(|source| AssociationLinkArgs {
                from_index: record.index.to_string(),
                to_index: source.to_string(),
                relation: MemoryLinkRelation::DerivedFrom,
                freeform_relation: None,
                strength_percent: 100,
                confidence_percent: 100,
            })
            .collect::<Vec<_>>();
        links.extend(args.links);
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
                    freeform_relation: link.freeform_relation,
                    strength: confidence_percent_to_f32(link.strength_percent),
                    confidence: confidence_percent_to_f32(link.confidence_percent),
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
        let _ = self.allocation_updates.next_item().await?;
        let _ = self.allocation_updates.take_ready_items()?;
        Ok(())
    }
}

fn format_association_context(
    memory_metadata: &[MemoryMetadataContext],
    allocation_guidance: &[(String, String)],
) -> String {
    let mut out = String::from("Memory association context.");
    out.push_str("\n\nMemory metadata candidates:");
    if memory_metadata.is_empty() {
        out.push_str("\n- none");
    } else {
        for item in memory_metadata {
            out.push_str(&format!(
                "\n- {}: rank={:?}; occurred_at={}; decay_remaining_secs={}; access_count={}; use_count={}; reinforcement_count={}",
                item.index,
                item.rank,
                item.occurred_at,
                item.decay_remaining_secs,
                item.access_count,
                item.use_count,
                item.reinforcement_count
            ));
        }
    }
    out.push_str("\n\nCurrent association guidance:");
    if allocation_guidance.is_empty() {
        out.push_str("\n- none");
    } else {
        for (id, guidance) in allocation_guidance {
            out.push_str(&format!("\n- {id}: {guidance}"));
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

    fn role_description() -> &'static str {
        "Writes non-destructive memory associations, reflection summaries, and links; wakes on allocation guidance, never on raw memos."
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
