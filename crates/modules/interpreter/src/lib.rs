use anyhow::{Context, Result};
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use lutum::{Session, TextStepOutcomeWithTools, ToolResult};
use nuillu_module::{
    CognitionLogReader, CognitionLogUpdatedInbox, CognitionWriter, LlmAccess, LlmContextWindow,
    Module, SessionAutoCompaction, SessionCompactionConfig, SessionCompactionProtectedPrefix,
    ensure_persistent_session_seeded, format_new_cognition_log_entries,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

mod batch;
pub use batch::InterpreterBatch;

const MODEL_PROMPT: &str = r#"You are the interpreter module.
You are a wakeful associative interpretation faculty. Read only the admitted cognition log and
produce concise inner thought when the current conscious state calls for meaning-making,
hypothesis, analogy, narrative angle, or story material.

Use exactly one tool per activation:
- append_interpretation when a useful interpretation should become part of cognitive processing.
- leave_interpretation_unchanged when the current cognition only needs an ordinary factual
  response, direct action, or no additional interpretation.

Do not use final assistant text as an output channel.

The text field is the exact cognition-log entry to append. Write it as plain inner experience, not
as hidden reasoning, instructions, or a report about this module. Keep it short and directly tied to
the current cognition target. Distinguish observed or recalled facts from imagined, hypothetical,
or fictional material. If you generate story material without admitted factual grounding, label it
as a possible imagined story seed rather than as something that happened. Do not mention mechanical
internals such as modules, memos, allocation, tools, prompts, schemas, blackboards, logs, or
implementation details in the appended text."#;

const COMPACTED_INTERPRETER_SESSION_PREFIX: &str = "Compacted interpreter session history:";
const COGNITION_CONTEXT_WINDOW: LlmContextWindow = LlmContextWindow::new(12, 600, 4_800);
const TOOL_TURN_MAX_OUTPUT_TOKENS: u32 = 512;
const SESSION_COMPACTION_FOCUS: &str = r#"Preserve prior interpretations, hypotheses, story seeds,
rejected/no-change decisions, and cognition-log context needed for future interpretation updates."#;

pub fn session_auto_compaction() -> SessionAutoCompaction {
    SessionAutoCompaction::new(
        SessionCompactionConfig::default(),
        SessionCompactionProtectedPrefix::LeadingSystemAndIdentitySeed,
        COMPACTED_INTERPRETER_SESSION_PREFIX,
        SESSION_COMPACTION_FOCUS,
    )
}

fn format_interpreter_update_input(
    cognition_log: &[nuillu_module::CognitionLogEntryRecord],
    now: DateTime<Utc>,
) -> Option<String> {
    let cognition = format_new_cognition_log_entries(cognition_log, now, COGNITION_CONTEXT_WINDOW)?;
    Some(format!(
        "{cognition}\n\nInstruction: Interpret the new cognition above only when useful for the current conscious state."
    ))
}

#[lutum::tool_input(name = "append_interpretation", output = AppendInterpretationOutput)]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct AppendInterpretationArgs {
    pub text: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct AppendInterpretationOutput {
    pub appended: bool,
}

#[lutum::tool_input(
    name = "leave_interpretation_unchanged",
    output = LeaveInterpretationUnchangedOutput
)]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct LeaveInterpretationUnchangedArgs {
    pub reason: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct LeaveInterpretationUnchangedOutput {
    pub unchanged: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema, lutum::Toolset)]
pub enum InterpreterTools {
    AppendInterpretation(AppendInterpretationArgs),
    LeaveInterpretationUnchanged(LeaveInterpretationUnchangedArgs),
}

pub struct InterpreterModule {
    cognition_updates: CognitionLogUpdatedInbox,
    cognition_log: CognitionLogReader,
    cognition: CognitionWriter,
    llm: LlmAccess,
    session: Session,
    model_prompt: std::sync::OnceLock<String>,
}

impl InterpreterModule {
    pub fn new(
        cognition_updates: CognitionLogUpdatedInbox,
        cognition_log: CognitionLogReader,
        cognition: CognitionWriter,
        llm: LlmAccess,
        session: Session,
    ) -> Self {
        Self {
            cognition_updates,
            cognition_log,
            cognition,
            llm,
            session,
            model_prompt: std::sync::OnceLock::new(),
        }
    }

    fn ensure_session_seeded(&mut self, cx: &nuillu_module::ActivateCx<'_>) {
        let model_prompt = self.model_prompt(cx).to_owned();
        ensure_persistent_session_seeded(
            &mut self.session,
            model_prompt,
            cx.identity_memories(),
            cx.now(),
        );
    }

    fn model_prompt(&self, cx: &nuillu_module::ActivateCx<'_>) -> &str {
        self.model_prompt.get_or_init(|| {
            nuillu_module::format_policy_system_prompt(MODEL_PROMPT, cx.core_policies())
        })
    }

    #[tracing::instrument(skip_all, err(Debug, level = "warn"))]
    async fn update_model(
        &mut self,
        cx: &nuillu_module::ActivateCx<'_>,
        batch: &InterpreterBatch,
    ) -> Result<()> {
        self.ensure_session_seeded(cx);

        let Some(update_input) = format_interpreter_update_input(&batch.cognition_log, cx.now())
        else {
            return Ok(());
        };

        let lutum = self.llm.lutum().await;
        let outcome = {
            self.session.push_user(update_input);
            self.session
                .text_turn()
                .tools::<InterpreterTools>()
                .available_tools([
                    InterpreterToolsSelector::AppendInterpretation,
                    InterpreterToolsSelector::LeaveInterpretationUnchanged,
                ])
                .require_any_tool()
                .max_output_tokens(TOOL_TURN_MAX_OUTPUT_TOKENS)
                .collect_controlled_with(
                    &lutum,
                    nuillu_module::AbortOnAvailableToolNameInText::new(),
                )
                .await
                .context("interpreter interpretation turn failed")?
        };

        let round = match outcome {
            TextStepOutcomeWithTools::Finished(result) => {
                cx.compact_and_save(&mut self.session, result.usage).await?;
                anyhow::bail!("interpreter finished without required tool call");
            }
            TextStepOutcomeWithTools::FinishedNoOutput(result) => {
                cx.compact_and_save(&mut self.session, result.usage).await?;
                anyhow::bail!("interpreter finished without required tool call");
            }
            TextStepOutcomeWithTools::NeedsTools(round) => round,
        };
        let usage = round.usage;

        let mut results: Vec<ToolResult> = Vec::new();
        nuillu_module::emit_trace_tool_calls(&round.tool_calls);
        for call in round.tool_calls.iter().cloned() {
            match call {
                InterpreterToolsCall::AppendInterpretation(call) => {
                    let output = self
                        .append_interpretation(call.input.clone())
                        .await
                        .context("run append_interpretation tool")?;
                    results.push(
                        call.complete(output)
                            .context("complete append_interpretation tool call")?,
                    );
                }
                InterpreterToolsCall::LeaveInterpretationUnchanged(call) => {
                    let output = self.leave_interpretation_unchanged(call.input.clone());
                    results.push(
                        call.complete(output)
                            .context("complete leave_interpretation_unchanged tool call")?,
                    );
                }
            }
        }
        round
            .commit(&mut self.session, results)
            .context("commit interpreter tool round")?;
        cx.compact_and_save(&mut self.session, usage).await?;
        Ok(())
    }

    async fn append_interpretation(
        &self,
        args: AppendInterpretationArgs,
    ) -> Result<AppendInterpretationOutput> {
        let text = args.text.trim();
        if text.is_empty() {
            return Ok(AppendInterpretationOutput { appended: false });
        }
        self.cognition.append(text.to_owned()).await;
        Ok(AppendInterpretationOutput { appended: true })
    }

    fn leave_interpretation_unchanged(
        &self,
        _args: LeaveInterpretationUnchangedArgs,
    ) -> LeaveInterpretationUnchangedOutput {
        LeaveInterpretationUnchangedOutput { unchanged: true }
    }
}

#[async_trait(?Send)]
impl Module for InterpreterModule {
    type Batch = InterpreterBatch;

    fn id() -> &'static str {
        "interpreter"
    }

    fn peer_context() -> Option<&'static str> {
        Some(
            "Interpreter forms concise inner interpretations, hypotheses, analogies, and story seeds from the current cognition log.",
        )
    }

    fn allocation_hint() -> Option<&'static str> {
        Some(
            "Raise interpreter when current cognition needs meaning-making, free thought, narrative generation, analogy, or interpretation before speech. Keep it low for direct factual response, routine action, or when extra inner thought would add noise.",
        )
    }

    async fn next_batch(&mut self) -> Result<Self::Batch> {
        InterpreterModule::next_batch(self).await
    }

    async fn activate(
        &mut self,
        cx: &nuillu_module::ActivateCx<'_>,
        batch: &Self::Batch,
    ) -> Result<()> {
        if batch.has_updates() {
            self.update_model(cx, batch).await?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::rc::Rc;
    use std::sync::Arc;

    use chrono::TimeZone as _;
    use lutum::{
        FinishReason, Lutum, MockLlmAdapter, MockTextScenario, RawTextTurnEvent,
        SharedPoolBudgetManager, SharedPoolBudgetOptions, Usage,
    };
    use nuillu_blackboard::{
        Blackboard, BlackboardCommand, Bpm, CognitionLogEntry, CognitionLogOrigin, ModulePolicy,
    };
    use nuillu_module::ports::{NoopCognitionLogRepository, SystemClock};
    use nuillu_module::{
        CapabilityProviderPorts, CapabilityProviders, CognitionLogUpdated, LlmConcurrencyLimiter,
        LutumTiers, ModuleRegistry, SessionCompactionPolicy, SessionCompactionRuntime,
    };
    use nuillu_types::{ModelTier, ModuleInstanceId, ReplicaCapRange, ReplicaIndex, builtin};

    fn append_interpretation_scenario(text: &str) -> MockTextScenario {
        MockTextScenario::events(vec![
            Ok(RawTextTurnEvent::Started {
                request_id: Some("interpreter-append".into()),
                model: "mock".into(),
            }),
            Ok(RawTextTurnEvent::ToolCallChunk {
                id: "append-interpretation".into(),
                name: "append_interpretation".into(),
                arguments_json_delta: serde_json::json!({ "text": text }).to_string(),
            }),
            Ok(RawTextTurnEvent::Completed {
                request_id: Some("interpreter-append".into()),
                finish_reason: FinishReason::ToolCall,
                usage: Usage::zero(),
            }),
        ])
    }

    fn test_caps(blackboard: Blackboard, lutum: Lutum) -> CapabilityProviders {
        CapabilityProviders::new(CapabilityProviderPorts {
            blackboard,
            cognition_log_port: Rc::new(NoopCognitionLogRepository),
            clock: Rc::new(SystemClock),
            tiers: LutumTiers::from_shared_lutum(lutum),
        })
    }

    fn module_policy() -> ModulePolicy {
        ModulePolicy::new(
            ReplicaCapRange::new(1, 1).unwrap(),
            Bpm::from_f64(60_000.0)..=Bpm::from_f64(60_000.0),
            nuillu_blackboard::linear_ratio_fn,
        )
    }

    fn activate_cx(lutum: &Lutum, now: DateTime<Utc>) -> nuillu_module::ActivateCx<'static> {
        nuillu_module::ActivateCx::new(
            &[],
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

    async fn build_interpreter_module(
        caps: &CapabilityProviders,
    ) -> nuillu_module::AllocatedModule {
        let modules = ModuleRegistry::new()
            .register(module_policy(), |caps| async move {
                Ok(InterpreterModule::new(
                    caps.cognition_log_updated_inbox(),
                    caps.cognition_log_reader(),
                    caps.cognition_writer(),
                    caps.llm_access(),
                    caps.session("main")
                        .with_auto_compaction(session_auto_compaction())
                        .await?,
                ))
            })
            .unwrap()
            .build(caps)
            .await
            .unwrap();
        let (_, mut modules) = modules.into_parts();
        modules.remove(0)
    }

    #[tokio::test(flavor = "current_thread")]
    async fn append_interpretation_writes_cognition_log_entry() {
        let now = Utc.with_ymd_and_hms(2026, 6, 13, 15, 0, 0).unwrap();
        let blackboard = Blackboard::default();
        let adapter = Arc::new(MockLlmAdapter::new().with_text_scenario(
            append_interpretation_scenario(
                "A possible imagined story seed: a tiny debugging ritual becomes a lantern.",
            ),
        ));
        let lutum = Lutum::new(
            adapter,
            SharedPoolBudgetManager::new(SharedPoolBudgetOptions::default()),
        );
        let caps = test_caps(blackboard.clone(), lutum.clone());
        let mut module = build_interpreter_module(&caps).await;
        let harness = caps.internal_harness_io();
        let source = ModuleInstanceId::new(builtin::cognition_gate(), ReplicaIndex::ZERO);

        blackboard
            .apply(BlackboardCommand::AppendCognitionLog {
                source: source.clone(),
                entry: CognitionLogEntry {
                    at: now,
                    text: "Alice asks for an interesting story.".to_owned(),
                    origin: CognitionLogOrigin::direct(source.clone()),
                },
            })
            .await;
        harness
            .cognition_log_updated_mailbox()
            .publish(CognitionLogUpdated::EntryAppended { source })
            .await
            .unwrap();

        let batch = module.next_batch().await.unwrap();
        module
            .activate(&activate_cx(&lutum, now), &batch)
            .await
            .unwrap();

        let logs = blackboard.read(|bb| bb.cognition_log_set()).await;
        let interpreter_entries = logs
            .logs()
            .iter()
            .filter(|record| record.source.module == builtin::interpreter())
            .flat_map(|record| record.entries.iter())
            .map(|entry| entry.text.as_str())
            .collect::<Vec<_>>();

        assert_eq!(
            interpreter_entries,
            vec!["A possible imagined story seed: a tiny debugging ritual becomes a lantern."]
        );
    }
}
