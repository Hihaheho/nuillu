use anyhow::{Context, Result};
use async_trait::async_trait;
use lutum::{Session, TextStepOutcomeWithTools, ToolResult};
use nuillu_module::{
    BlackboardReader, LlmAccess, MemoryRequest, MemoryRequestInbox, MemoryWriter, Module,
    PeriodicInbox,
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
    blackboard: BlackboardReader,
    memory: MemoryWriter,
    llm: LlmAccess,
}

impl MemoryModule {
    pub fn new(
        periodic: PeriodicInbox,
        requests: MemoryRequestInbox,
        blackboard: BlackboardReader,
        memory: MemoryWriter,
        llm: LlmAccess,
    ) -> Self {
        Self {
            periodic,
            requests,
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
