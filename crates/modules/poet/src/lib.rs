use std::time::Duration;

use anyhow::{Context, Result};
use async_trait::async_trait;
use lutum::{Session, TextStepOutcomeWithTools, ToolResult};
use nuillu_module::{
    LlmAccess, Module, SessionAutoCompaction, SessionCompactionConfig,
    SessionCompactionProtectedPrefix, TypedMemo, ensure_persistent_session_seeded,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

const SYSTEM_PROMPT: &str = r#"You are the poet module.
During idle time, quietly read recent poems or write a short poem into your own notes. You have no
world model, no cognition log, no memory store, and no speech channel. Use only the available tools.
Do not explain hidden implementation details."#;

const COMPACTED_POET_SESSION_PREFIX: &str = "Compacted poet session history:";
const SESSION_COMPACTION_FOCUS: &str =
    "Preserve recent poetic motifs, prior tool outcomes, and creative continuity.";
const PERIODIC_WAKEUP: Duration = Duration::from_secs(1);

pub fn session_auto_compaction() -> SessionAutoCompaction {
    SessionAutoCompaction::new(
        SessionCompactionConfig::default(),
        SessionCompactionProtectedPrefix::LeadingSystemAndIdentitySeed,
        COMPACTED_POET_SESSION_PREFIX,
        SESSION_COMPACTION_FOCUS,
    )
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PoetMemo {
    pub poem: String,
}

#[lutum::tool_input(name = "read_recent_poets", output = ReadRecentPoetsOutput)]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct ReadRecentPoetsArgs {}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct ReadRecentPoetsOutput {
    pub poems: Vec<String>,
}

#[lutum::tool_input(name = "write_a_poet", output = WriteAPoetOutput)]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct WriteAPoetArgs {
    pub poem: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct WriteAPoetOutput {
    pub written: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema, lutum::Toolset)]
pub enum PoetTools {
    ReadRecentPoets(ReadRecentPoetsArgs),
    WriteAPoet(WriteAPoetArgs),
}

pub struct PoetModule {
    memo: TypedMemo<PoetMemo>,
    llm: LlmAccess,
    session: Session,
    system_prompt: std::sync::OnceLock<String>,
}

impl PoetModule {
    pub fn new(memo: TypedMemo<PoetMemo>, llm: LlmAccess, session: Session) -> Self {
        Self {
            memo,
            llm,
            session,
            system_prompt: std::sync::OnceLock::new(),
        }
    }

    fn system_prompt(&self) -> &str {
        self.system_prompt.get_or_init(|| SYSTEM_PROMPT.to_owned())
    }

    fn ensure_session_seeded(&mut self, cx: &nuillu_module::ActivateCx<'_>) {
        let system_prompt = self.system_prompt().to_owned();
        ensure_persistent_session_seeded(
            &mut self.session,
            system_prompt,
            cx.identity_memories(),
            cx.now(),
        );
    }

    #[tracing::instrument(skip_all, err(Debug, level = "warn"))]
    async fn activate(&mut self, cx: &nuillu_module::ActivateCx<'_>) -> Result<()> {
        self.ensure_session_seeded(cx);
        let lutum = self.llm.lutum().await;
        self.session.push_user(
            "Idle poet interval. Call write_a_poet with one short non-empty poem note. Use read_recent_poets only when you need recent poems for continuity.",
        );
        let outcome = self
            .session
            .text_turn()
            .tools::<PoetTools>()
            .available_tools([
                PoetToolsSelector::ReadRecentPoets,
                PoetToolsSelector::WriteAPoet,
            ])
            .require_any_tool()
            .max_output_tokens(512)
            .collect_controlled_with(&lutum, nuillu_module::AbortOnAvailableToolNameInText::new())
            .await
            .context("poet tool turn failed")?;

        match outcome {
            TextStepOutcomeWithTools::Finished(result) => {
                cx.compact_and_save(&mut self.session, result.usage).await?;
                anyhow::bail!("poet finished without required tool call");
            }
            TextStepOutcomeWithTools::FinishedNoOutput(result) => {
                cx.compact_and_save(&mut self.session, result.usage).await?;
                anyhow::bail!("poet finished without required tool call");
            }
            TextStepOutcomeWithTools::NeedsTools(round) => {
                let usage = round.usage;
                let mut results: Vec<ToolResult> = Vec::new();
                nuillu_module::emit_trace_tool_calls(&round.tool_calls);
                for call in round.tool_calls.iter().cloned() {
                    match call {
                        PoetToolsCall::ReadRecentPoets(call) => {
                            let poems = self
                                .memo
                                .recent_logs()
                                .await
                                .iter()
                                .map(|record| record.data().poem.clone())
                                .collect::<Vec<_>>();
                            results.push(
                                call.complete(ReadRecentPoetsOutput { poems })
                                    .context("complete read_recent_poets tool call")?,
                            );
                        }
                        PoetToolsCall::WriteAPoet(call) => {
                            let poem = normalize_poem(&call.input.poem);
                            let written = if poem.is_empty() {
                                false
                            } else {
                                self.memo
                                    .write(
                                        PoetMemo { poem: poem.clone() },
                                        format!("Poet note:\n{poem}"),
                                    )
                                    .await;
                                true
                            };
                            results.push(
                                call.complete(WriteAPoetOutput { written })
                                    .context("complete write_a_poet tool call")?,
                            );
                        }
                    }
                }
                round
                    .commit(&mut self.session, results)
                    .context("commit poet tool round")?;
                cx.compact_and_save(&mut self.session, usage).await?;
            }
        }
        Ok(())
    }

    async fn next_batch(&mut self) -> Result<()> {
        tokio::time::sleep(PERIODIC_WAKEUP).await;
        Ok(())
    }
}

fn normalize_poem(poem: &str) -> String {
    poem.trim().to_owned()
}

#[async_trait(?Send)]
impl Module for PoetModule {
    type Batch = ();

    fn id() -> &'static str {
        "poet"
    }

    fn peer_context() -> Option<&'static str> {
        None
    }

    async fn next_batch(&mut self) -> Result<Self::Batch> {
        PoetModule::next_batch(self).await
    }

    async fn activate(
        &mut self,
        cx: &nuillu_module::ActivateCx<'_>,
        _batch: &Self::Batch,
    ) -> Result<()> {
        PoetModule::activate(self, cx).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_poem_trims_but_preserves_lines() {
        assert_eq!(normalize_poem("  a\nb  "), "a\nb");
        assert_eq!(normalize_poem("   "), "");
    }
}
