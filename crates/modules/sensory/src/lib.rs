use std::collections::{BTreeMap, BTreeSet};
use std::rc::Rc;
use std::time::Duration;

use anyhow::{Context, Result};
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use lutum::{MessageContent, ModelInputItem, Session, TextStepOutcomeWithTools, ToolResult};
use nuillu_module::{
    AllocationReader, AmbientSensoryEntry, LlmAccess, Memo, Module, SensoryInput,
    SensoryInputInbox, SensoryModality, SessionAutoCompaction, SessionCompactionConfig,
    SessionCompactionProtectedPrefix, ensure_persistent_session_seeded, ports::Clock,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use tokio::time::Instant;

const ONE_SHOT_SYSTEM_PROMPT: &str = r#"You are the one-shot sensory filter.
You receive one-shot sensory events from the environment. A one-shot event is an individual event
that occurred at its own observed_at time; it is not a snapshot, a diff, or a code-labeled repeated
stimulus. Use the session history to understand temporal context, but evaluate the current event as
its own occurrence. Call dispose_sensory only for current one-shot events that should not become
durable sensory memo-log output. If no current event should be disposed, finish without function calls
and without assistant text. Do not write memo text yourself; kept events are recorded by the
runtime. Do not mention tools, prompts, scores, schemas, rubrics, or implementation details. Do not
write to the cognition log, memory, or emit utterances."#;

const AMBIENT_SYSTEM_PROMPT: &str = r#"You are the ambient sensory diff filter.
You receive changes derived from ambient sensory snapshots. Ambient rows are background conditions;
the input describes what changed since the previous active ambient snapshot. Use tools for every
decision: call broadcast_sensory for durable sensory memo-log output, or ignore_ambient_diff when
the diff should not be written. Do not use final assistant text as an output channel. Memo text must
contain observed scene facts only. Do not mention tools, prompts, scores, schemas, rubrics, or
implementation details in memo text. Do not write to the cognition log, memory, or emit
utterances."#;

const SENSORY_LLM_TURN_TIMEOUT: Duration = Duration::from_secs(20);
const TOOL_TURN_MAX_OUTPUT_TOKENS: u32 = 512;

const COMPACTED_ONE_SHOT_SESSION_PREFIX: &str = "Compacted one-shot sensory session history:";
const COMPACTED_AMBIENT_SESSION_PREFIX: &str = "Compacted ambient sensory session history:";
const ONE_SHOT_SESSION_COMPACTION_FOCUS: &str = r#"Preserve one-shot event facts, source/direction
details, observed_at times, dispose_sensory decisions, and uncertainty. Do not convert separate
events into a single repeated stimulus label."#;
const AMBIENT_SESSION_COMPACTION_FOCUS: &str = r#"Preserve ambient diff facts, memo-log outputs
written through tools, ignored background context, and uncertainty."#;

pub fn one_shot_session_auto_compaction() -> SessionAutoCompaction {
    SessionAutoCompaction::new(
        SessionCompactionConfig::default(),
        SessionCompactionProtectedPrefix::LeadingSystemAndIdentitySeed,
        COMPACTED_ONE_SHOT_SESSION_PREFIX,
        ONE_SHOT_SESSION_COMPACTION_FOCUS,
    )
}

pub fn ambient_session_auto_compaction() -> SessionAutoCompaction {
    SessionAutoCompaction::new(
        SessionCompactionConfig::default(),
        SessionCompactionProtectedPrefix::LeadingSystemAndIdentitySeed,
        COMPACTED_AMBIENT_SESSION_PREFIX,
        AMBIENT_SESSION_COMPACTION_FOCUS,
    )
}

const DEFAULT_BURST_SILENT_WINDOW: Duration = Duration::from_millis(100);
const DEFAULT_BURST_BUDGET: Duration = Duration::from_secs(1);

#[derive(Clone, Debug)]
struct PreparedOneShotObservation {
    id: String,
    modality: SensoryModality,
    direction: Option<String>,
    content: String,
    observed_at: DateTime<Utc>,
    relative_age: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AmbientObservationKind {
    AmbientAdded,
    AmbientUpdated,
    AmbientRemoved,
}

impl AmbientObservationKind {
    fn label(self) -> &'static str {
        match self {
            Self::AmbientAdded => "ambient added",
            Self::AmbientUpdated => "ambient updated",
            Self::AmbientRemoved => "ambient removed",
        }
    }
}

#[derive(Clone, Debug)]
enum AmbientDiff {
    Added {
        id: String,
        entry: AmbientSensoryEntry,
    },
    Updated {
        id: String,
        previous: AmbientSensoryEntry,
        current: AmbientSensoryEntry,
    },
    Removed {
        id: String,
        previous: AmbientSensoryEntry,
    },
}

#[derive(Clone, Debug)]
struct PreparedAmbientObservation {
    kind: AmbientObservationKind,
    modality: SensoryModality,
    ambient_id: String,
    content: String,
    observed_at: DateTime<Utc>,
    relative_age: String,
}

pub struct SensoryModule {
    owner: nuillu_types::ModuleId,
    inbox: SensoryInputInbox,
    allocation: AllocationReader,
    memo: Memo,
    clock: Rc<dyn Clock>,
    llm: LlmAccess,
    one_shot_session: Session,
    ambient_session: Session,
    next_one_shot_sequence: u64,
    burst: SensoryBurstConfig,
    one_shot_system_prompt: std::sync::OnceLock<String>,
    ambient_system_prompt: std::sync::OnceLock<String>,
    ambient_entries: BTreeMap<String, AmbientSensoryEntry>,
}

#[lutum::tool_input(name = "broadcast_sensory", output = BroadcastSensoryOutput)]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
struct BroadcastSensoryArgs {
    memo: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
struct BroadcastSensoryOutput {
    written: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "kebab-case")]
enum IgnoreAmbientCategory {
    Background,
    NoMeaningfulChange,
    StaleRemoval,
    TooAmbiguous,
}

#[lutum::tool_input(name = "ignore_ambient_diff", output = IgnoreAmbientDiffOutput)]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
struct IgnoreAmbientDiffArgs {
    category: IgnoreAmbientCategory,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
struct IgnoreAmbientDiffOutput {
    ignored: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema, lutum::Toolset)]
enum AmbientSensoryTools {
    BroadcastSensory(BroadcastSensoryArgs),
    IgnoreAmbientDiff(IgnoreAmbientDiffArgs),
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "kebab-case")]
enum DisposeSensoryCategory {
    EmptyOrInvalid,
    SensorArtifact,
    NonsocialBackground,
    TooAmbiguous,
    Other,
}

#[lutum::tool_input(name = "dispose_sensory", output = DisposeSensoryOutput)]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
struct DisposeSensoryArgs {
    observation_ids: Vec<String>,
    category: DisposeSensoryCategory,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
struct DisposeSensoryOutput {
    disposed: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema, lutum::Toolset)]
enum OneShotSensoryTools {
    DisposeSensory(DisposeSensoryArgs),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct SensoryBurstConfig {
    silent_window: Duration,
    budget: Duration,
}

impl Default for SensoryBurstConfig {
    fn default() -> Self {
        Self {
            silent_window: DEFAULT_BURST_SILENT_WINDOW,
            budget: DEFAULT_BURST_BUDGET,
        }
    }
}

impl SensoryModule {
    pub fn new(
        inbox: SensoryInputInbox,
        allocation: AllocationReader,
        memo: Memo,
        clock: Rc<dyn Clock>,
        llm: LlmAccess,
        one_shot_session: Session,
        ambient_session: Session,
    ) -> Self {
        let next_one_shot_sequence = next_one_shot_sequence_from_session(&one_shot_session);
        Self {
            owner: nuillu_types::ModuleId::new(<Self as Module>::id())
                .expect("sensory id is valid"),
            inbox,
            allocation,
            memo,
            clock,
            llm,
            one_shot_session,
            ambient_session,
            next_one_shot_sequence,
            burst: SensoryBurstConfig::default(),
            one_shot_system_prompt: std::sync::OnceLock::new(),
            ambient_system_prompt: std::sync::OnceLock::new(),
            ambient_entries: BTreeMap::new(),
        }
    }

    #[cfg(test)]
    fn with_burst_config(mut self, burst: SensoryBurstConfig) -> Self {
        self.burst = burst;
        self
    }

    fn one_shot_system_prompt(&self, cx: &nuillu_module::ActivateCx<'_>) -> &str {
        self.one_shot_system_prompt.get_or_init(|| {
            nuillu_module::format_faculty_system_prompt(
                ONE_SHOT_SYSTEM_PROMPT,
                cx.peer_contexts(),
                &self.owner,
            )
        })
    }

    fn ambient_system_prompt(&self, cx: &nuillu_module::ActivateCx<'_>) -> &str {
        self.ambient_system_prompt.get_or_init(|| {
            nuillu_module::format_faculty_system_prompt(
                AMBIENT_SYSTEM_PROMPT,
                cx.peer_contexts(),
                &self.owner,
            )
        })
    }

    fn ensure_one_shot_session_seeded(&mut self, cx: &nuillu_module::ActivateCx<'_>) {
        let system_prompt = self.one_shot_system_prompt(cx).to_owned();
        ensure_persistent_session_seeded(
            &mut self.one_shot_session,
            system_prompt,
            cx.identity_memories(),
            cx.now(),
        );
    }

    fn ensure_ambient_session_seeded(&mut self, cx: &nuillu_module::ActivateCx<'_>) {
        let system_prompt = self.ambient_system_prompt(cx).to_owned();
        ensure_persistent_session_seeded(
            &mut self.ambient_session,
            system_prompt,
            cx.identity_memories(),
            cx.now(),
        );
    }

    fn format_age(now: DateTime<Utc>, observed_at: DateTime<Utc>) -> String {
        let secs = (now - observed_at).num_seconds().max(0) as u64;
        if secs < 60 {
            format!("{secs} seconds ago")
        } else if secs < 3600 {
            let mut m = secs / 60;
            let mut s = ((secs % 60) + 5) / 10 * 10;
            if s == 60 {
                m += 1;
                s = 0;
            }
            if m == 60 {
                return "1 hour 0 minutes ago".into();
            }
            format!("{m} minute{} {s} second{} ago", plural(m), plural(s))
        } else if secs < 86400 {
            let mut h = secs / 3600;
            let mut m = ((secs % 3600) / 60 + 5) / 10 * 10;
            if m == 60 {
                h += 1;
                m = 0;
            }
            format!("{h} hour{} {m} minute{} ago", plural(h), plural(m))
        } else {
            let d = secs / 86400;
            let h = (secs % 86400) / 3600;
            format!("{d} day{} {h} hour{} ago", plural(d), plural(h))
        }
    }

    #[tracing::instrument(skip_all, err(Debug, level = "warn"))]
    async fn handle_inputs(
        &mut self,
        cx: &nuillu_module::ActivateCx<'_>,
        inputs: &[SensoryInput],
    ) -> Result<()> {
        if inputs.is_empty() {
            return Ok(());
        }
        let now = self.clock.now();
        let mut one_shot_observations = Vec::new();
        let mut ambient_observations = Vec::new();
        for input in inputs.iter().cloned() {
            match input {
                SensoryInput::OneShot {
                    modality,
                    direction,
                    content,
                    observed_at,
                } => {
                    if content.trim().is_empty() {
                        continue;
                    }
                    one_shot_observations.push(self.prepare_one_shot_observation(
                        now,
                        modality,
                        direction,
                        content,
                        observed_at,
                    ));
                }
                SensoryInput::AmbientSnapshot {
                    entries,
                    observed_at,
                } => ambient_observations.extend(self.prepare_ambient_snapshot_observations(
                    now,
                    entries,
                    observed_at,
                )),
            }
        }
        if !ambient_observations.is_empty() {
            self.handle_ambient_observations(cx, &ambient_observations, now)
                .await?;
        }
        if !one_shot_observations.is_empty() {
            self.handle_one_shot_observations(cx, &one_shot_observations, now)
                .await?;
        }
        Ok(())
    }

    async fn handle_one_shot_observations(
        &mut self,
        cx: &nuillu_module::ActivateCx<'_>,
        observations: &[PreparedOneShotObservation],
        now: DateTime<Utc>,
    ) -> Result<()> {
        self.ensure_one_shot_session_seeded(cx);
        let allocation = self.allocation.snapshot().await;
        let guidance = allocation.for_module(&self.owner).guidance;
        let lutum = self.llm.lutum().await;
        let current_ids = observations
            .iter()
            .map(|observation| observation.id.clone())
            .collect::<BTreeSet<_>>();
        let (outcome, session_len_after_user) = {
            self.one_shot_session
                .push_user(format_one_shot_turn_user_message(observations, now));
            let session_len_after_user = self.one_shot_session.input().items().len();
            self.one_shot_session
                .push_ephemeral_developer(format_one_shot_decision_context(&guidance));
            let turn = self
                .one_shot_session
                .text_turn()
                .tools::<OneShotSensoryTools>()
                .available_tools([OneShotSensoryToolsSelector::DisposeSensory])
                .max_output_tokens(TOOL_TURN_MAX_OUTPUT_TOKENS);
            let outcome = match tokio::time::timeout(
                SENSORY_LLM_TURN_TIMEOUT,
                turn.collect_controlled_with(
                    &lutum,
                    nuillu_module::AbortOnAvailableToolNameInText::new(),
                ),
            )
            .await
            {
                Ok(result) => result.context("one-shot sensory text turn failed")?,
                Err(_) => anyhow::bail!(
                    "one-shot sensory LLM turn timed out after {}s",
                    SENSORY_LLM_TURN_TIMEOUT.as_secs()
                ),
            };
            (outcome, session_len_after_user)
        };

        let mut disposed = BTreeSet::new();
        match outcome {
            TextStepOutcomeWithTools::NeedsTools(round) => {
                if round.tool_calls.is_empty() {
                    let usage = round.usage;
                    self.one_shot_session
                        .input_mut()
                        .items_mut()
                        .truncate(session_len_after_user);
                    cx.compact_and_save(&mut self.one_shot_session, usage)
                        .await?;
                } else {
                    let usage = round.usage;
                    let mut results: Vec<ToolResult> = Vec::new();
                    nuillu_module::emit_trace_tool_calls(&round.tool_calls);
                    for call in round.tool_calls.iter().cloned() {
                        match call {
                            OneShotSensoryToolsCall::DisposeSensory(call) => {
                                let output = self.dispose_sensory(call.input.clone(), &current_ids);
                                disposed.extend(output.disposed.iter().cloned());
                                results.push(
                                    call.complete(output)
                                        .context("complete dispose_sensory tool call")?,
                                );
                            }
                        }
                    }
                    round
                        .commit(&mut self.one_shot_session, results)
                        .context("commit one-shot sensory tool round")?;
                    cx.compact_and_save(&mut self.one_shot_session, usage)
                        .await?;
                }
            }
            TextStepOutcomeWithTools::Finished(result) => {
                self.one_shot_session
                    .input_mut()
                    .items_mut()
                    .truncate(session_len_after_user);
                cx.compact_and_save(&mut self.one_shot_session, result.usage)
                    .await?;
            }
            TextStepOutcomeWithTools::FinishedNoOutput(result) => {
                self.one_shot_session
                    .input_mut()
                    .items_mut()
                    .truncate(session_len_after_user);
                cx.compact_and_save(&mut self.one_shot_session, result.usage)
                    .await?;
            }
        }
        let kept = observations
            .iter()
            .filter(|observation| !disposed.contains(&observation.id))
            .cloned()
            .collect::<Vec<_>>();
        if !kept.is_empty() {
            self.memo.write(format_one_shot_memo(&kept)).await;
        }
        Ok(())
    }

    async fn handle_ambient_observations(
        &mut self,
        cx: &nuillu_module::ActivateCx<'_>,
        observations: &[PreparedAmbientObservation],
        now: DateTime<Utc>,
    ) -> Result<()> {
        self.ensure_ambient_session_seeded(cx);
        let allocation = self.allocation.snapshot().await;
        let guidance = allocation.for_module(&self.owner).guidance;
        let lutum = self.llm.lutum().await;
        let (outcome, session_len_before_turn) = {
            let session_len_before_turn = self.ambient_session.input().items().len();
            self.ambient_session
                .push_user(format_ambient_diff(observations, now));
            self.ambient_session
                .push_ephemeral_developer(format_ambient_decision_context(
                    &guidance,
                    &format_current_ambient_context(&self.ambient_entries, now),
                ));
            let turn = self
                .ambient_session
                .text_turn()
                .tools::<AmbientSensoryTools>()
                .available_tools([
                    AmbientSensoryToolsSelector::BroadcastSensory,
                    AmbientSensoryToolsSelector::IgnoreAmbientDiff,
                ])
                .require_any_tool()
                .max_output_tokens(TOOL_TURN_MAX_OUTPUT_TOKENS);
            let outcome = match tokio::time::timeout(
                SENSORY_LLM_TURN_TIMEOUT,
                turn.collect_controlled_with(
                    &lutum,
                    nuillu_module::AbortOnAvailableToolNameInText::new(),
                ),
            )
            .await
            {
                Ok(result) => result.context("ambient sensory text turn failed")?,
                Err(_) => anyhow::bail!(
                    "ambient sensory LLM turn timed out after {}s",
                    SENSORY_LLM_TURN_TIMEOUT.as_secs()
                ),
            };
            (outcome, session_len_before_turn)
        };

        match outcome {
            TextStepOutcomeWithTools::NeedsTools(round) => {
                if round.tool_calls.is_empty() {
                    let usage = round.usage;
                    self.ambient_session
                        .input_mut()
                        .items_mut()
                        .truncate(session_len_before_turn);
                    cx.compact_and_save(&mut self.ambient_session, usage)
                        .await?;
                    anyhow::bail!("ambient sensory text turn finished without required tool calls");
                }
                let usage = round.usage;
                let mut results: Vec<ToolResult> = Vec::new();
                nuillu_module::emit_trace_tool_calls(&round.tool_calls);
                for call in round.tool_calls.iter().cloned() {
                    match call {
                        AmbientSensoryToolsCall::BroadcastSensory(call) => {
                            let output = self.broadcast_sensory(call.input.clone()).await;
                            results.push(
                                call.complete(output)
                                    .context("complete broadcast_sensory tool call")?,
                            );
                        }
                        AmbientSensoryToolsCall::IgnoreAmbientDiff(call) => {
                            let output = self.ignore_ambient_diff(call.input.clone());
                            results.push(
                                call.complete(output)
                                    .context("complete ignore_ambient_diff tool call")?,
                            );
                        }
                    }
                }
                round
                    .commit(&mut self.ambient_session, results)
                    .context("commit ambient sensory tool round")?;
                cx.compact_and_save(&mut self.ambient_session, usage)
                    .await?;
            }
            TextStepOutcomeWithTools::Finished(result) => {
                self.ambient_session
                    .input_mut()
                    .items_mut()
                    .truncate(session_len_before_turn);
                cx.compact_and_save(&mut self.ambient_session, result.usage)
                    .await?;
                anyhow::bail!("ambient sensory text turn finished without required tool calls");
            }
            TextStepOutcomeWithTools::FinishedNoOutput(result) => {
                self.ambient_session
                    .input_mut()
                    .items_mut()
                    .truncate(session_len_before_turn);
                cx.compact_and_save(&mut self.ambient_session, result.usage)
                    .await?;
                anyhow::bail!("ambient sensory text turn finished without required tool calls");
            }
        }
        Ok(())
    }

    fn prepare_ambient_snapshot_observations(
        &mut self,
        now: DateTime<Utc>,
        entries: Vec<AmbientSensoryEntry>,
        observed_at: DateTime<Utc>,
    ) -> Vec<PreparedAmbientObservation> {
        let mut next = BTreeMap::new();
        for entry in entries {
            next.insert(entry.id.clone(), entry);
        }

        let mut diffs = Vec::new();
        for (id, entry) in &next {
            match self.ambient_entries.get(id) {
                None => diffs.push(AmbientDiff::Added {
                    id: id.clone(),
                    entry: entry.clone(),
                }),
                Some(previous)
                    if previous.modality != entry.modality || previous.content != entry.content =>
                {
                    diffs.push(AmbientDiff::Updated {
                        id: id.clone(),
                        previous: previous.clone(),
                        current: entry.clone(),
                    });
                }
                Some(_) => {}
            }
        }
        for (id, previous) in &self.ambient_entries {
            if !next.contains_key(id) {
                diffs.push(AmbientDiff::Removed {
                    id: id.clone(),
                    previous: previous.clone(),
                });
            }
        }

        self.ambient_entries = next;
        diffs
            .into_iter()
            .map(|diff| self.prepare_ambient_diff_observation(now, diff, observed_at))
            .collect()
    }

    fn prepare_one_shot_observation(
        &mut self,
        now: DateTime<Utc>,
        modality: SensoryModality,
        direction: Option<String>,
        raw_content: String,
        observed_at: DateTime<Utc>,
    ) -> PreparedOneShotObservation {
        let id = self.next_one_shot_observation_id();
        PreparedOneShotObservation {
            id,
            modality,
            direction,
            content: raw_content,
            observed_at,
            relative_age: Self::format_age(now, observed_at),
        }
    }

    fn next_one_shot_observation_id(&mut self) -> String {
        let sequence = self.next_one_shot_sequence;
        self.next_one_shot_sequence = self.next_one_shot_sequence.saturating_add(1);
        format!("one-shot-{sequence}")
    }

    fn prepare_ambient_diff_observation(
        &self,
        now: DateTime<Utc>,
        diff: AmbientDiff,
        observed_at: DateTime<Utc>,
    ) -> PreparedAmbientObservation {
        match diff {
            AmbientDiff::Added { id, entry } => self.prepare_ambient_observation(
                now,
                AmbientObservationKind::AmbientAdded,
                entry.modality,
                id,
                entry.content,
                observed_at,
            ),
            AmbientDiff::Updated {
                id,
                previous,
                current,
            } => self.prepare_ambient_observation(
                now,
                AmbientObservationKind::AmbientUpdated,
                current.modality.clone(),
                id,
                format!(
                    "was {}: {}; now {}: {}",
                    previous.modality.as_str(),
                    previous.content.trim(),
                    current.modality.as_str(),
                    current.content.trim()
                ),
                observed_at,
            ),
            AmbientDiff::Removed { id, previous } => self.prepare_ambient_observation(
                now,
                AmbientObservationKind::AmbientRemoved,
                previous.modality.clone(),
                id,
                format!(
                    "removed; previous {}: {}",
                    previous.modality.as_str(),
                    previous.content.trim()
                ),
                observed_at,
            ),
        }
    }

    fn prepare_ambient_observation(
        &self,
        now: DateTime<Utc>,
        kind: AmbientObservationKind,
        modality: SensoryModality,
        ambient_id: String,
        raw_content: String,
        observed_at: DateTime<Utc>,
    ) -> PreparedAmbientObservation {
        PreparedAmbientObservation {
            kind,
            modality,
            ambient_id,
            content: raw_content,
            observed_at,
            relative_age: Self::format_age(now, observed_at),
        }
    }

    async fn broadcast_sensory(&self, args: BroadcastSensoryArgs) -> BroadcastSensoryOutput {
        let memo = args.memo.trim();
        let written = !memo.is_empty();
        if written {
            self.memo.write(memo.to_owned()).await;
        }
        BroadcastSensoryOutput { written }
    }

    fn ignore_ambient_diff(&self, _args: IgnoreAmbientDiffArgs) -> IgnoreAmbientDiffOutput {
        IgnoreAmbientDiffOutput { ignored: true }
    }

    fn dispose_sensory(
        &self,
        args: DisposeSensoryArgs,
        current_ids: &BTreeSet<String>,
    ) -> DisposeSensoryOutput {
        let disposed = args
            .observation_ids
            .into_iter()
            .filter(|id| current_ids.contains(id))
            .collect();
        DisposeSensoryOutput { disposed }
    }

    async fn next_batch(&mut self) -> Result<SensoryBatch> {
        let input = self.inbox.next_item().await?;
        let mut batch = SensoryBatch {
            inputs: vec![input.body],
        };
        self.collect_ready_events_into_batch(&mut batch)?;
        if !batch.inputs.is_empty() {
            self.collect_sensory_burst(&mut batch).await?;
        }
        Ok(batch)
    }

    async fn collect_sensory_burst(&mut self, batch: &mut SensoryBatch) -> Result<()> {
        let mut waited = Duration::ZERO;
        while waited < self.burst.budget {
            let remaining = self.burst.budget.saturating_sub(waited);
            let wait_for = std::cmp::min(self.burst.silent_window, remaining);
            if wait_for.is_zero() {
                break;
            }

            let started = Instant::now();
            tokio::select! {
                input = self.inbox.next_item() => {
                    batch.inputs.push(input?.body);
                    waited += std::cmp::min(started.elapsed(), wait_for);
                    let _ = self.collect_ready_events_into_batch(batch)?;
                }
                _ = tokio::time::sleep(wait_for) => {
                    waited += wait_for;
                    let ready = self.collect_ready_events_into_batch(batch)?;
                    if ready.inputs == 0 {
                        break;
                    }
                }
            }
        }
        Ok(())
    }

    fn collect_ready_events_into_batch(&mut self, batch: &mut SensoryBatch) -> Result<ReadyCounts> {
        let ready_inputs = self.inbox.take_ready_items()?;
        let input_count = ready_inputs.items.len();
        batch
            .inputs
            .extend(ready_inputs.items.into_iter().map(|envelope| envelope.body));

        Ok(ReadyCounts {
            inputs: input_count,
        })
    }
}

#[derive(Debug, Default)]
pub struct SensoryBatch {
    inputs: Vec<SensoryInput>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct ReadyCounts {
    inputs: usize,
}

fn plural(n: u64) -> &'static str {
    if n == 1 { "" } else { "s" }
}

fn next_one_shot_sequence_from_session(session: &Session) -> u64 {
    let max_sequence = session
        .input()
        .items()
        .iter()
        .filter_map(model_input_text)
        .flat_map(one_shot_sequences)
        .max()
        .unwrap_or(0);
    max_sequence.saturating_add(1)
}

fn model_input_text(item: &ModelInputItem) -> Option<&str> {
    match item {
        ModelInputItem::Message { content, .. } => match content.as_slice() {
            [MessageContent::Text(text)] => Some(text.as_str()),
            _ => None,
        },
        ModelInputItem::Assistant(lutum::AssistantInputItem::Text(text)) => Some(text.as_str()),
        _ => None,
    }
}

fn one_shot_sequences(text: &str) -> impl Iterator<Item = u64> + '_ {
    text.match_indices("one-shot-")
        .filter_map(|(start, prefix)| {
            let digits_start = start + prefix.len();
            let digits = text[digits_start..]
                .chars()
                .take_while(|ch| ch.is_ascii_digit())
                .collect::<String>();
            if digits.is_empty() {
                None
            } else {
                digits.parse::<u64>().ok()
            }
        })
}

fn format_one_shot_turn_user_message(
    observations: &[PreparedOneShotObservation],
    now: DateTime<Utc>,
) -> String {
    let mut out = format!("Current one-shot sensory batch at {}:", now.to_rfc3339());
    out.push_str("\nThese are individual events with their own observed_at times, not a snapshot or a code-generated repetition/diff.");
    out.push_str("\nOnly ids listed in this current batch are valid dispose_sensory observation_ids for this turn.");
    if observations.is_empty() {
        out.push_str("\n- none");
    } else {
        for observation in observations {
            out.push_str(&format!(
                "\n- [{}] {}{} observed {} ({}): {}",
                observation.id,
                observation.modality.as_str(),
                observation
                    .direction
                    .as_deref()
                    .map(|direction| format!(" from {direction}"))
                    .unwrap_or_default(),
                observation.relative_age,
                observation.observed_at.to_rfc3339(),
                observation.content.trim()
            ));
        }
    }

    out
}

fn format_one_shot_memo(observations: &[PreparedOneShotObservation]) -> String {
    observations
        .iter()
        .map(format_one_shot_memo_line)
        .collect::<Vec<_>>()
        .join("\n")
}

fn format_one_shot_memo_line(observation: &PreparedOneShotObservation) -> String {
    let source = observation
        .direction
        .as_deref()
        .map(|direction| format!(" from {direction}"))
        .unwrap_or_default();
    match observation.modality {
        SensoryModality::Audition => format!(
            "Heard{} at {}: {}",
            source,
            observation.observed_at.to_rfc3339(),
            observation.content.trim()
        ),
        SensoryModality::Vision => format!(
            "Saw{} at {}: {}",
            source,
            observation.observed_at.to_rfc3339(),
            observation.content.trim()
        ),
        _ => format!(
            "Observed {}{} at {}: {}",
            observation.modality.as_str(),
            source,
            observation.observed_at.to_rfc3339(),
            observation.content.trim()
        ),
    }
}

fn format_current_ambient_context(
    entries: &BTreeMap<String, AmbientSensoryEntry>,
    now: DateTime<Utc>,
) -> String {
    let mut out = format!("Current ambient sensory field at {}:", now.to_rfc3339());
    if entries.is_empty() {
        out.push_str("\n- none");
    } else {
        out.push_str(
            "\nThese are currently enabled background rows. They are dynamic context, not durable conclusions.",
        );
        for (id, entry) in entries {
            out.push_str(&format!(
                "\n- [{}] {}: {}",
                id,
                entry.modality.as_str(),
                entry.content.trim()
            ));
        }
    }
    out
}

fn format_ambient_diff(observations: &[PreparedAmbientObservation], now: DateTime<Utc>) -> String {
    let mut out = format!(
        "Ambient sensory snapshot diff received at {}:",
        now.to_rfc3339()
    );
    out.push_str(
        "\nAmbient entries are enabled background rows only; removal from the active ambient field is not proof that a condition disappeared.",
    );
    if observations.is_empty() {
        out.push_str("\n- none");
    } else {
        for observation in observations {
            out.push_str(&format!(
                "\n- {} {} [{}] observed {} ({}): {}",
                observation.kind.label(),
                observation.modality.as_str(),
                observation.ambient_id,
                observation.relative_age,
                observation.observed_at.to_rfc3339(),
                observation.content.trim()
            ));
        }
    }
    out
}

fn format_one_shot_decision_context(guidance: &str) -> String {
    let mut out = String::from("Current sensory guidance: ");
    let guidance = guidance.trim();
    if guidance.is_empty() {
        out.push_str("none");
    } else {
        out.push_str(guidance);
    }
    out.push_str(
        "\nCurrent batch ids are the only valid dispose_sensory observation_ids for this turn.",
    );
    out
}

fn format_ambient_decision_context(guidance: &str, current_ambient_context: &str) -> String {
    let mut out = String::from("Current sensory guidance: ");
    let guidance = guidance.trim();
    if guidance.is_empty() {
        out.push_str("none");
    } else {
        out.push_str(guidance);
    }
    out.push_str("\n\n");
    out.push_str(current_ambient_context);
    out
}

#[async_trait(?Send)]
impl Module for SensoryModule {
    type Batch = SensoryBatch;

    fn id() -> &'static str {
        "sensory"
    }

    fn peer_context() -> Option<&'static str> {
        None
    }

    fn allocation_hint() -> Option<&'static str> {
        None
    }

    async fn next_batch(&mut self) -> Result<Self::Batch> {
        SensoryModule::next_batch(self).await
    }

    async fn activate(
        &mut self,
        cx: &nuillu_module::ActivateCx<'_>,
        batch: &Self::Batch,
    ) -> Result<()> {
        if !batch.inputs.is_empty() {
            self.handle_inputs(cx, &batch.inputs).await?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::cell::RefCell;
    use std::rc::Rc;
    use std::sync::Arc;

    use lutum::{
        AdapterStructuredTurn, AdapterTextTurn, AgentError, ErasedStructuredTurnEventStream,
        ErasedTextTurnEventStream, FinishReason, InputMessageRole, Lutum, MaxOutputTokens,
        MessageContent, MockLlmAdapter, MockTextScenario, ModelInputItem, RawTextTurnEvent,
        SharedPoolBudgetManager, SharedPoolBudgetOptions, TurnAdapter, Usage,
    };
    use nuillu_blackboard::{
        ActivationRatio, Blackboard, BlackboardCommand, Bpm, ModuleConfig, ResourceAllocation,
        linear_ratio_fn,
    };
    use nuillu_module::ports::{NoopCognitionLogRepository, SystemClock};
    use nuillu_module::{
        CapabilityProviderConfig, CapabilityProviderPorts, CapabilityProviderRuntime,
        CapabilityProviders, LutumTiers, ModuleRegistry, RuntimePolicy, SessionCompactionPolicy,
    };
    use nuillu_types::builtin;
    use tokio::task::LocalSet;

    fn reference_now() -> DateTime<Utc> {
        DateTime::parse_from_rfc3339("2026-05-07T12:00:00Z")
            .unwrap()
            .with_timezone(&Utc)
    }

    #[derive(Clone, Default)]
    struct SensoryTestRecorder {
        batches: Rc<RefCell<Vec<usize>>>,
        one_shot_session_inputs: Rc<RefCell<Vec<Vec<ModelInputItem>>>>,
        ambient_session_inputs: Rc<RefCell<Vec<Vec<ModelInputItem>>>>,
    }

    struct RecordingSensoryModule {
        inner: SensoryModule,
        recorder: SensoryTestRecorder,
    }

    #[async_trait(?Send)]
    impl Module for RecordingSensoryModule {
        type Batch = SensoryBatch;

        fn id() -> &'static str {
            SensoryModule::id()
        }

        fn peer_context() -> Option<&'static str> {
            SensoryModule::peer_context()
        }

        fn allocation_hint() -> Option<&'static str> {
            SensoryModule::allocation_hint()
        }

        async fn next_batch(&mut self) -> Result<Self::Batch> {
            self.inner.next_batch().await
        }

        async fn activate(
            &mut self,
            cx: &nuillu_module::ActivateCx<'_>,
            batch: &Self::Batch,
        ) -> Result<()> {
            self.recorder.batches.borrow_mut().push(batch.inputs.len());
            <SensoryModule as Module>::activate(&mut self.inner, cx, batch).await?;
            let one_shot_items = self.inner.one_shot_session.input().items().to_vec();
            if !one_shot_items.is_empty() {
                self.recorder
                    .one_shot_session_inputs
                    .borrow_mut()
                    .push(one_shot_items);
            }
            let ambient_items = self.inner.ambient_session.input().items().to_vec();
            if !ambient_items.is_empty() {
                self.recorder
                    .ambient_session_inputs
                    .borrow_mut()
                    .push(ambient_items);
            }
            Ok(())
        }
    }

    fn tool_scenario(name: &str, arguments_json: String, input_tokens: u64) -> MockTextScenario {
        let mut usage = Usage::zero();
        usage.input_tokens = input_tokens;
        usage.total_tokens = input_tokens;
        MockTextScenario::events(vec![
            Ok(RawTextTurnEvent::Started {
                request_id: Some("sensory-text".into()),
                model: "mock".into(),
            }),
            Ok(RawTextTurnEvent::ToolCallChunk {
                id: "call-sensory".into(),
                name: name.into(),
                arguments_json_delta: arguments_json,
            }),
            Ok(RawTextTurnEvent::Completed {
                request_id: Some("sensory-text".into()),
                finish_reason: FinishReason::ToolCall,
                usage,
            }),
        ])
    }

    fn write_memo_scenario(memo: &str, input_tokens: u64) -> MockTextScenario {
        tool_scenario(
            "broadcast_sensory",
            serde_json::json!({ "memo": memo }).to_string(),
            input_tokens,
        )
    }

    fn ignore_ambient_scenario(input_tokens: u64) -> MockTextScenario {
        tool_scenario(
            "ignore_ambient_diff",
            serde_json::json!({ "category": "background" }).to_string(),
            input_tokens,
        )
    }

    fn dispose_scenario(ids: Vec<&str>, input_tokens: u64) -> MockTextScenario {
        tool_scenario(
            "dispose_sensory",
            serde_json::json!({
                "observation_ids": ids,
                "category": "nonsocial-background",
            })
            .to_string(),
            input_tokens,
        )
    }

    fn text_scenario(text: &str, input_tokens: u64) -> MockTextScenario {
        let mut usage = Usage::zero();
        usage.input_tokens = input_tokens;
        usage.total_tokens = input_tokens;
        MockTextScenario::events(vec![
            Ok(RawTextTurnEvent::Started {
                request_id: Some("sensory-text".into()),
                model: "mock".into(),
            }),
            Ok(RawTextTurnEvent::TextDelta { delta: text.into() }),
            Ok(RawTextTurnEvent::Completed {
                request_id: Some("sensory-text".into()),
                finish_reason: FinishReason::Stop,
                usage,
            }),
        ])
    }

    fn sensory_allocation() -> ResourceAllocation {
        let mut allocation = ResourceAllocation::default();
        let module = builtin::sensory();
        allocation.set(module.clone(), ModuleConfig::default());
        allocation.set_activation(module, ActivationRatio::ONE);
        allocation
    }

    fn test_caps_with_adapter(adapter: MockLlmAdapter) -> (Blackboard, CapabilityProviders) {
        test_caps_with_adapter_and_policy(adapter, RuntimePolicy::default())
    }

    fn test_caps_with_adapter_and_policy<T>(
        adapter: T,
        policy: RuntimePolicy,
    ) -> (Blackboard, CapabilityProviders)
    where
        T: TurnAdapter,
    {
        let blackboard = Blackboard::with_allocation(sensory_allocation());
        let adapter = Arc::new(adapter);
        let budget = SharedPoolBudgetManager::new(SharedPoolBudgetOptions::default());
        let lutum = Lutum::new(adapter, budget);
        let caps = CapabilityProviders::new(CapabilityProviderConfig {
            ports: CapabilityProviderPorts {
                blackboard: blackboard.clone(),
                cognition_log_port: Rc::new(NoopCognitionLogRepository),
                clock: Rc::new(SystemClock),
                tiers: LutumTiers::from_shared_lutum(lutum),
            },
            runtime: CapabilityProviderRuntime {
                policy,
                ..CapabilityProviderRuntime::default()
            },
        });
        (blackboard, caps)
    }

    #[derive(Clone)]
    struct CapturingAdapter {
        inner: MockLlmAdapter,
        text_turns: Arc<std::sync::Mutex<Vec<AdapterTextTurn>>>,
    }

    impl CapturingAdapter {
        fn new(inner: MockLlmAdapter) -> Self {
            Self {
                inner,
                text_turns: Arc::new(std::sync::Mutex::new(Vec::new())),
            }
        }

        fn text_turns(&self) -> Vec<AdapterTextTurn> {
            self.text_turns.lock().unwrap().clone()
        }
    }

    #[async_trait::async_trait]
    impl TurnAdapter for CapturingAdapter {
        async fn text_turn(
            &self,
            input: lutum::ModelInput,
            turn: AdapterTextTurn,
        ) -> Result<ErasedTextTurnEventStream, AgentError> {
            self.text_turns.lock().unwrap().push(turn.clone());
            self.inner.text_turn(input, turn).await
        }

        async fn structured_turn(
            &self,
            input: lutum::ModelInput,
            turn: AdapterStructuredTurn,
        ) -> Result<ErasedStructuredTurnEventStream, AgentError> {
            self.inner.structured_turn(input, turn).await
        }
    }

    fn test_policy() -> nuillu_blackboard::ModulePolicy {
        nuillu_blackboard::ModulePolicy::new(
            nuillu_types::ReplicaCapRange::new(1, 1).unwrap(),
            Bpm::from_f64(60_000.0)..=Bpm::from_f64(60_000.0),
            linear_ratio_fn,
        )
    }

    async fn build_recording_sensory(
        caps: &CapabilityProviders,
        recorder: SensoryTestRecorder,
        burst: SensoryBurstConfig,
        _compaction: Option<SessionCompactionConfig>,
        session_history: Vec<String>,
    ) -> nuillu_module::AllocatedModules {
        ModuleRegistry::new()
            .register(test_policy(), move |caps| {
                let recorder = recorder.clone();
                let session_history = session_history.clone();
                async move {
                    let mut inner = SensoryModule::new(
                        caps.sensory_input_inbox(),
                        caps.allocation_reader(),
                        caps.memo(),
                        caps.clock(),
                        caps.llm_access(),
                        caps.session("one-shot")
                            .with_auto_compaction(one_shot_session_auto_compaction())
                            .await?,
                        caps.session("ambient")
                            .with_auto_compaction(ambient_session_auto_compaction())
                            .await?,
                    )
                    .with_burst_config(burst);
                    for item in &session_history {
                        inner.one_shot_session.push_user(item.clone());
                    }
                    Ok(RecordingSensoryModule { inner, recorder })
                }
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
        run_modules_with_max_activation_attempts(modules, 3, body).await;
    }

    async fn run_modules_with_max_activation_attempts<F: std::future::Future<Output = ()>>(
        modules: nuillu_module::AllocatedModules,
        max_activation_attempts: u8,
        body: F,
    ) {
        nuillu_agent::run(
            modules,
            nuillu_agent::AgentEventLoopConfig {
                idle_threshold: Duration::from_millis(50),
                max_activation_attempts,
                dependency_idle_timeout: Duration::from_secs(2),
                dependency_hard_timeout: Duration::from_secs(10),
            },
            body,
        )
        .await
        .expect("sensory test runtime should not fail");
    }

    fn test_activate_cx(now: DateTime<Utc>) -> nuillu_module::ActivateCx<'static> {
        let lutum = Lutum::new(
            Arc::new(MockLlmAdapter::new()),
            SharedPoolBudgetManager::new(SharedPoolBudgetOptions::default()),
        );
        nuillu_module::ActivateCx::new(
            &[],
            &[],
            &[],
            &[],
            nuillu_module::SessionCompactionRuntime::new(
                lutum,
                nuillu_module::LlmConcurrencyLimiter::new(None),
                nuillu_types::ModelTier::Cheap,
                SessionCompactionPolicy::default(),
            ),
            now,
        )
    }

    fn heard(content: &str) -> SensoryInput {
        SensoryInput::OneShot {
            modality: SensoryModality::Audition,
            direction: Some("front".to_string()),
            content: content.to_string(),
            observed_at: reference_now(),
        }
    }

    async fn wait_for_memo_log_count(blackboard: &Blackboard, count: usize) {
        for _ in 0..50 {
            if blackboard.read(|bb| bb.recent_memo_logs().len()).await >= count {
                return;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    }

    async fn wait_for_batch_count(recorder: &SensoryTestRecorder, count: usize) {
        for _ in 0..50 {
            if recorder.batches.borrow().len() >= count {
                return;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    }

    async fn wait_for_one_shot_session_count(recorder: &SensoryTestRecorder, count: usize) {
        for _ in 0..50 {
            if recorder.one_shot_session_inputs.borrow().len() >= count {
                return;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    }

    async fn wait_for_ambient_session_count(recorder: &SensoryTestRecorder, count: usize) {
        for _ in 0..50 {
            if recorder.ambient_session_inputs.borrow().len() >= count {
                return;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    }

    fn ambient_entry(id: &str, modality: SensoryModality, content: &str) -> AmbientSensoryEntry {
        AmbientSensoryEntry {
            id: id.to_string(),
            modality,
            content: content.to_string(),
        }
    }

    fn ambient(entries: Vec<AmbientSensoryEntry>) -> SensoryInput {
        SensoryInput::AmbientSnapshot {
            entries,
            observed_at: reference_now(),
        }
    }

    #[test]
    fn format_age_rounds_boundary_values() {
        let now = reference_now();

        assert_eq!(
            SensoryModule::format_age(now, now - chrono::Duration::seconds(59)),
            "59 seconds ago"
        );
        assert_eq!(
            SensoryModule::format_age(now, now - chrono::Duration::seconds(60)),
            "1 minute 0 seconds ago"
        );
        assert_eq!(
            SensoryModule::format_age(now, now - chrono::Duration::seconds(3599)),
            "1 hour 0 minutes ago"
        );
        assert_eq!(
            SensoryModule::format_age(now, now - chrono::Duration::seconds(3600)),
            "1 hour 0 minutes ago"
        );
    }

    #[test]
    fn sensory_modality_round_trips_known_and_custom_strings() {
        let known = serde_json::to_string(&SensoryModality::Audition).unwrap();
        assert_eq!(known, "\"audition\"");
        assert_eq!(
            serde_json::from_str::<SensoryModality>("\"audio\"").unwrap(),
            SensoryModality::Audition
        );
        assert_eq!(
            serde_json::from_str::<SensoryModality>("\"thermal\"").unwrap(),
            SensoryModality::Other("thermal".to_string())
        );
    }

    #[test]
    fn one_shot_turn_user_message_has_event_facts_without_salience_metadata() {
        let observations = vec![PreparedOneShotObservation {
            id: "one-shot-1".to_string(),
            modality: SensoryModality::Audition,
            direction: Some("Pibi".to_string()),
            content: "Pibi asks a question.".to_string(),
            observed_at: reference_now(),
            relative_age: "0 seconds ago".to_string(),
        }];
        let text = format_one_shot_turn_user_message(&observations, reference_now());

        assert!(text.contains("Current one-shot sensory batch at"));
        assert!(text.contains("Only ids listed in this current batch"));
        assert!(text.contains("[one-shot-1] audition from Pibi"));
        assert!(text.contains("Pibi asks a question."));
        assert!(!text.contains("signature="));
        assert!(!text.contains("repetition_count"));
        assert!(!text.contains("novelty_score"));
        assert!(!text.contains("salience_score"));
    }

    #[test]
    fn one_shot_sequence_resumes_from_session_history() {
        let mut session = Session::new();
        session.push_user(
            "Current one-shot sensory batch at 2026-05-07T12:00:00+00:00:\n\
             - [one-shot-41] audition from front observed 0 seconds ago \
             (2026-05-07T12:00:00+00:00): sound",
        );

        assert_eq!(next_one_shot_sequence_from_session(&session), 42);
    }

    #[test]
    fn ambient_snapshot_format_marks_background_context() {
        let observations = vec![PreparedAmbientObservation {
            kind: AmbientObservationKind::AmbientAdded,
            modality: SensoryModality::Smell,
            ambient_id: "row-1".to_string(),
            content: "wet stone smell".to_string(),
            observed_at: reference_now(),
            relative_age: "0 seconds ago".to_string(),
        }];
        let text = format_ambient_diff(&observations, reference_now());

        assert!(text.contains("Ambient entries are enabled background rows only"));
        assert!(text.contains("ambient added smell [row-1]"));
        assert!(text.contains("wet stone smell"));
        assert!(!text.contains("salience_score"));
    }

    #[test]
    fn current_ambient_context_formats_full_snapshot() {
        let mut entries = BTreeMap::new();
        entries.insert(
            "ambient-1".to_string(),
            ambient_entry("ambient-1", SensoryModality::Vision, "lamp is on"),
        );
        entries.insert(
            "ambient-2".to_string(),
            ambient_entry("ambient-2", SensoryModality::Audition, "fan hum"),
        );

        let text = format_current_ambient_context(&entries, reference_now());

        assert!(text.contains("Current ambient sensory field at 2026-05-07T12:00:00+00:00"));
        assert!(text.contains("[ambient-1] vision: lamp is on"));
        assert!(text.contains("[ambient-2] audition: fan hum"));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn activation_caps_tool_turn_output_tokens() {
        let local = LocalSet::new();
        local
            .run_until(async {
                let capture = CapturingAdapter::new(
                    MockLlmAdapter::new().with_text_scenario(text_scenario("", 0)),
                );
                let observed = capture.clone();
                let (_blackboard, caps) =
                    test_caps_with_adapter_and_policy(capture, RuntimePolicy::default());
                let recorder = SensoryTestRecorder::default();
                let modules = build_recording_sensory(
                    &caps,
                    recorder.clone(),
                    SensoryBurstConfig {
                        silent_window: Duration::from_millis(1),
                        budget: Duration::from_millis(1),
                    },
                    None,
                    Vec::new(),
                )
                .await;
                let sensory = caps.host_io().sensory_input_mailbox();

                run_modules(modules, async {
                    sensory
                        .publish(heard("fan hum continues"))
                        .await
                        .expect("sensory subscriber exists");
                    wait_for_one_shot_session_count(&recorder, 1).await;
                })
                .await;

                let turns = observed.text_turns();
                assert_eq!(turns.len(), 1);
                assert_eq!(
                    turns[0].config.generation.max_output_tokens,
                    Some(MaxOutputTokens::new(TOOL_TURN_MAX_OUTPUT_TOKENS))
                );
            })
            .await;
    }

    #[tokio::test(flavor = "current_thread")]
    async fn burst_batches_inputs_arriving_inside_silent_window() {
        let local = LocalSet::new();
        local
            .run_until(async {
                let mut adapter = MockLlmAdapter::new();
                for _ in 0..10 {
                    adapter = adapter.with_text_scenario(text_scenario("", 0));
                }
                let (_blackboard, caps) = test_caps_with_adapter(adapter);
                let recorder = SensoryTestRecorder::default();
                let modules = build_recording_sensory(
                    &caps,
                    recorder.clone(),
                    SensoryBurstConfig {
                        silent_window: Duration::from_millis(30),
                        budget: Duration::from_millis(100),
                    },
                    None,
                    Vec::new(),
                )
                .await;
                let mailbox = caps.host_io().sensory_input_mailbox();

                run_modules(modules, async {
                    mailbox
                        .publish(heard("first sound"))
                        .await
                        .expect("sensory subscriber exists");
                    mailbox
                        .publish(heard("second sound"))
                        .await
                        .expect("sensory subscriber exists");
                    for _ in 0..50 {
                        if !recorder.batches.borrow().is_empty() {
                            break;
                        }
                        tokio::time::sleep(Duration::from_millis(5)).await;
                    }
                    tokio::time::sleep(Duration::from_millis(20)).await;
                })
                .await;

                assert_eq!(recorder.batches.borrow().as_slice(), &[2]);
            })
            .await;
    }

    #[tokio::test(flavor = "current_thread")]
    async fn one_shot_events_are_persisted_as_user_history() {
        let local = LocalSet::new();
        local
            .run_until(async {
                let adapter = MockLlmAdapter::new()
                    .with_text_scenario(text_scenario("", 0))
                    .with_text_scenario(text_scenario("", 0));
                let (blackboard, caps) = test_caps_with_adapter(adapter);
                let recorder = SensoryTestRecorder::default();
                let modules = build_recording_sensory(
                    &caps,
                    recorder.clone(),
                    SensoryBurstConfig {
                        silent_window: Duration::from_millis(30),
                        budget: Duration::from_millis(100),
                    },
                    None,
                    Vec::new(),
                )
                .await;
                let sensory = caps.host_io().sensory_input_mailbox();

                run_modules(modules, async {
                    sensory
                        .publish(heard("Koro is standing at the front."))
                        .await
                        .expect("sensory subscriber exists");
                    wait_for_memo_log_count(&blackboard, 1).await;
                    sensory
                        .publish(heard("Koro growled from the front."))
                        .await
                        .expect("sensory subscriber exists");
                    wait_for_memo_log_count(&blackboard, 2).await;
                })
                .await;

                let logs = blackboard.read(|bb| bb.recent_memo_logs()).await;
                let contents = logs
                    .iter()
                    .map(|record| record.content.as_str())
                    .collect::<Vec<_>>();
                assert_eq!(contents.len(), 2);
                assert!(contents[0].contains("Heard from front at 2026-05-07T12:00:00+00:00"));
                assert!(contents[0].contains("Koro is standing at the front."));
                assert!(contents[1].contains("Heard from front at 2026-05-07T12:00:00+00:00"));
                assert!(contents[1].contains("Koro growled from the front."));
                assert!(
                    contents
                        .iter()
                        .all(|text| !text.contains("novelty_score") && !text.contains("salience"))
                );
                assert_eq!(recorder.batches.borrow().as_slice(), &[1, 1]);
                let sessions = recorder.one_shot_session_inputs.borrow();
                let first_session = sessions.first().expect("first sensory activation recorded");
                let ModelInputItem::Message { role, content } = &first_session[0] else {
                    panic!("expected sensory system prompt as first session item");
                };
                assert_eq!(role, &InputMessageRole::System);
                let [MessageContent::Text(system)] = content.as_slice() else {
                    panic!("expected sensory system prompt text");
                };
                assert!(system.contains("You are the one-shot sensory filter"));
                assert_eq!(
                    count_message_role(first_session, InputMessageRole::System),
                    1
                );
                assert_eq!(
                    count_message_role(first_session, InputMessageRole::Developer),
                    0
                );
                assert_eq!(count_message_role(first_session, InputMessageRole::User), 1);
                let first_user_texts = user_texts(first_session);
                assert!(
                    first_user_texts
                        .iter()
                        .all(|text| !text.contains("Output instruction"))
                );
                assert!(
                    first_user_texts.iter().any(|text| {
                        text.contains("Current one-shot sensory batch")
                            && text.contains("[one-shot-1] audition from front")
                            && text.contains("Koro is standing at the front.")
                            && !text.contains("repetition_count")
                            && !text.contains("salience_score")
                    }),
                    "one-shot sensory input should be persisted as user-side event history"
                );
                assert!(
                    first_user_texts
                        .iter()
                        .all(|text| !text.contains("Current ambient sensory field")),
                    "ambient snapshot context should not persist in one-shot event history"
                );

                let second_session = sessions.get(1).expect("second sensory activation recorded");
                assert!(
                    second_session.iter().any(|item| {
                        matches!(
                            item,
                            ModelInputItem::Message {
                                role: InputMessageRole::User,
                                content,
                            } if matches!(
                                content.as_slice(),
                                [MessageContent::Text(text)]
                                    if text.contains("Koro growled from the front.")
                                        && text.contains("[one-shot-2] audition from front")
                            )
                        )
                    }),
                    "second sensory session should retain separate one-shot user turns"
                );
                assert_eq!(
                    count_message_role(second_session, InputMessageRole::System),
                    1
                );
                assert_eq!(
                    count_message_role(second_session, InputMessageRole::Developer),
                    0
                );
                assert_eq!(
                    count_message_role(second_session, InputMessageRole::User),
                    2
                );
                assert!(
                    user_texts(second_session)
                        .iter()
                        .all(|text| !text.contains("Output instruction"))
                );
            })
            .await;
    }

    #[tokio::test(flavor = "current_thread")]
    async fn ambient_diffs_are_persisted_as_user_turns_and_ephemeral_context_is_stripped() {
        let local = LocalSet::new();
        local
            .run_until(async {
                let adapter = MockLlmAdapter::new()
                    .with_text_scenario(write_memo_scenario("Lamp changed.", 0));
                let observed = adapter.clone();
                let (_blackboard, caps) = test_caps_with_adapter(adapter);
                let recorder = SensoryTestRecorder::default();
                let modules = build_recording_sensory(
                    &caps,
                    recorder.clone(),
                    SensoryBurstConfig {
                        silent_window: Duration::from_millis(30),
                        budget: Duration::from_millis(100),
                    },
                    None,
                    Vec::new(),
                )
                .await;
                let sensory = caps.host_io().sensory_input_mailbox();

                run_modules(modules, async {
                    sensory
                        .publish(ambient(vec![ambient_entry(
                            "ambient-1",
                            SensoryModality::Vision,
                            "lamp is on",
                        )]))
                        .await
                        .expect("sensory subscriber exists");
                    sensory
                        .publish(ambient(vec![ambient_entry(
                            "ambient-1",
                            SensoryModality::Vision,
                            "lamp is off",
                        )]))
                        .await
                        .expect("sensory subscriber exists");
                    sensory
                        .publish(ambient(Vec::new()))
                        .await
                        .expect("sensory subscriber exists");
                    wait_for_ambient_session_count(&recorder, 1).await;
                })
                .await;

                let sessions = recorder.ambient_session_inputs.borrow();
                assert_eq!(sessions.len(), 1);

                let first = user_texts(&sessions[0]).join("\n");
                assert!(first.contains("ambient added vision [ambient-1]"));
                assert!(first.contains("lamp is on"));
                assert!(!first.contains("Current ambient sensory field"));
                assert!(first.contains("ambient updated vision [ambient-1]"));
                assert!(first.contains("was vision: lamp is on; now vision: lamp is off"));
                assert!(first.contains("ambient removed vision [ambient-1]"));
                assert!(first.contains("removed; previous vision: lamp is off"));

                let ephemeral_indices = observed.observed_ephemeral_indices();
                assert_eq!(ephemeral_indices.len(), 1);
                assert!(
                    ephemeral_indices.iter().all(|indices| indices.len() == 1),
                    "ambient sensory LLM turn should carry decision context as an ephemeral item"
                );
            })
            .await;
    }

    #[tokio::test(flavor = "current_thread")]
    async fn ambient_plain_text_finish_is_rejected_without_memo() {
        let local = LocalSet::new();
        local
            .run_until(async {
                let adapter = MockLlmAdapter::new()
                    .with_text_scenario(text_scenario("plain response without tool call", 0));
                let observed = adapter.clone();
                let (blackboard, caps) = test_caps_with_adapter(adapter);
                let recorder = SensoryTestRecorder::default();
                let allocated = build_recording_sensory(
                    &caps,
                    recorder.clone(),
                    SensoryBurstConfig {
                        silent_window: Duration::from_millis(1),
                        budget: Duration::from_millis(1),
                    },
                    None,
                    Vec::new(),
                )
                .await;
                let sensory = caps.host_io().sensory_input_mailbox();
                let (_runtime, mut modules) = allocated.into_parts();
                let mut module = modules.pop().expect("sensory module should be built");

                sensory
                    .publish(ambient(vec![ambient_entry(
                        "ambient-1",
                        SensoryModality::Audition,
                        "unexpected plain text path",
                    )]))
                    .await
                    .expect("sensory subscriber exists");
                let batch = module
                    .next_batch()
                    .await
                    .expect("sensory batch should be ready");
                let cx = test_activate_cx(reference_now());
                let error = module
                    .activate(&cx, &batch)
                    .await
                    .expect_err("plain text finish should fail sensory activation");
                let logs = blackboard.read(|bb| bb.recent_memo_logs()).await;
                let message = format!("{error:#}");
                assert!(
                    message.contains("ambient sensory text turn failed")
                        || message.contains(
                            "ambient sensory text turn finished without required tool calls"
                        ),
                    "unexpected activation error: {message}"
                );

                assert!(logs.is_empty());
                assert_eq!(recorder.batches.borrow().as_slice(), &[1]);
                let ephemeral_indices = observed.observed_ephemeral_indices();
                assert_eq!(ephemeral_indices.len(), 1);
                assert!(
                    ephemeral_indices.iter().all(|indices| indices.len() == 1),
                    "ambient sensory LLM turn should carry decision context as an ephemeral item"
                );
            })
            .await;
    }

    #[tokio::test(flavor = "current_thread")]
    async fn unchanged_ambient_snapshot_skips_llm_turn() {
        let local = LocalSet::new();
        local
            .run_until(async {
                let adapter = MockLlmAdapter::new().with_text_scenario(ignore_ambient_scenario(0));
                let observed = adapter.clone();
                let (_blackboard, caps) = test_caps_with_adapter(adapter);
                let recorder = SensoryTestRecorder::default();
                let modules = build_recording_sensory(
                    &caps,
                    recorder.clone(),
                    SensoryBurstConfig {
                        silent_window: Duration::from_millis(1),
                        budget: Duration::from_millis(1),
                    },
                    None,
                    Vec::new(),
                )
                .await;
                let sensory = caps.host_io().sensory_input_mailbox();
                let snapshot = ambient(vec![ambient_entry(
                    "ambient-1",
                    SensoryModality::Audition,
                    "fan hum",
                )]);

                run_modules(modules, async {
                    sensory
                        .publish(snapshot.clone())
                        .await
                        .expect("sensory subscriber exists");
                    wait_for_batch_count(&recorder, 1).await;
                    sensory
                        .publish(snapshot)
                        .await
                        .expect("sensory subscriber exists");
                    wait_for_ambient_session_count(&recorder, 2).await;
                })
                .await;

                assert_eq!(recorder.batches.borrow().as_slice(), &[1, 1]);
                assert_eq!(
                    observed.observed_ephemeral_indices().len(),
                    1,
                    "unchanged ambient snapshots should not start another LLM turn"
                );
            })
            .await;
    }

    #[tokio::test(flavor = "current_thread")]
    async fn dispose_tool_does_not_write_one_shot_memo() {
        let local = LocalSet::new();
        local
            .run_until(async {
                let adapter = MockLlmAdapter::new()
                    .with_text_scenario(dispose_scenario(vec!["one-shot-1"], 0));
                let (blackboard, caps) = test_caps_with_adapter(adapter);
                let recorder = SensoryTestRecorder::default();
                let modules = build_recording_sensory(
                    &caps,
                    recorder.clone(),
                    SensoryBurstConfig {
                        silent_window: Duration::from_millis(1),
                        budget: Duration::from_millis(1),
                    },
                    None,
                    Vec::new(),
                )
                .await;
                let sensory = caps.host_io().sensory_input_mailbox();

                run_modules(modules, async {
                    sensory
                        .publish(heard("fan hum continues"))
                        .await
                        .expect("sensory subscriber exists");
                    for _ in 0..50 {
                        if !recorder.batches.borrow().is_empty() {
                            break;
                        }
                        tokio::time::sleep(Duration::from_millis(5)).await;
                    }
                    tokio::time::sleep(Duration::from_millis(20)).await;
                })
                .await;

                let logs = blackboard.read(|bb| bb.recent_memo_logs()).await;
                assert!(logs.is_empty());
                assert_eq!(recorder.batches.borrow().as_slice(), &[1]);
                let sessions = recorder.one_shot_session_inputs.borrow();
                let first_session = sessions.first().expect("sensory activation recorded");
                assert!(has_tool_call(first_session, "dispose_sensory"));
                assert!(has_tool_result(first_session, "dispose_sensory"));
            })
            .await;
    }

    #[tokio::test(flavor = "current_thread")]
    async fn stale_dispose_id_does_not_dispose_later_one_shot() {
        let local = LocalSet::new();
        local
            .run_until(async {
                let adapter = MockLlmAdapter::new()
                    .with_text_scenario(dispose_scenario(vec!["one-shot-1"], 0))
                    .with_text_scenario(dispose_scenario(vec!["one-shot-1"], 0));
                let (blackboard, caps) = test_caps_with_adapter(adapter);
                let recorder = SensoryTestRecorder::default();
                let modules = build_recording_sensory(
                    &caps,
                    recorder.clone(),
                    SensoryBurstConfig {
                        silent_window: Duration::from_millis(1),
                        budget: Duration::from_millis(1),
                    },
                    None,
                    Vec::new(),
                )
                .await;
                let sensory = caps.host_io().sensory_input_mailbox();

                run_modules_with_max_activation_attempts(modules, 4, async {
                    sensory
                        .publish(heard("first transient sound"))
                        .await
                        .expect("sensory subscriber exists");
                    wait_for_batch_count(&recorder, 1).await;
                    sensory
                        .publish(heard("second transient sound"))
                        .await
                        .expect("sensory subscriber exists");
                    wait_for_memo_log_count(&blackboard, 1).await;
                })
                .await;

                let logs = blackboard.read(|bb| bb.recent_memo_logs()).await;
                assert_eq!(logs.len(), 1);
                assert!(logs[0].content.contains("second transient sound"));
                let sessions = recorder.one_shot_session_inputs.borrow();
                let second_session = sessions.last().expect("second sensory activation recorded");
                assert!(
                    user_texts(second_session)
                        .iter()
                        .any(|text| text.contains("[one-shot-2] audition from front"))
                );
            })
            .await;
    }

    #[tokio::test(flavor = "current_thread")]
    async fn allocation_change_alone_does_not_wake_sensory() {
        let local = LocalSet::new();
        local
            .run_until(async {
                let adapter = MockLlmAdapter::new();
                let (blackboard, caps) = test_caps_with_adapter(adapter);
                let recorder = SensoryTestRecorder::default();
                let modules = build_recording_sensory(
                    &caps,
                    recorder.clone(),
                    SensoryBurstConfig {
                        silent_window: Duration::from_millis(1),
                        budget: Duration::from_millis(1),
                    },
                    None,
                    Vec::new(),
                )
                .await;

                run_modules(modules, async {
                    let mut allocation = sensory_allocation();
                    let mut config = allocation.for_module(&builtin::sensory());
                    config.guidance = "keep watching the front".into();
                    allocation.set(builtin::sensory(), config);
                    blackboard
                        .apply(BlackboardCommand::SetAllocation(allocation))
                        .await;
                    tokio::time::sleep(Duration::from_millis(30)).await;
                })
                .await;

                let logs = blackboard.read(|bb| bb.recent_memo_logs()).await;
                assert!(logs.is_empty());
                assert!(recorder.batches.borrow().is_empty());
            })
            .await;
    }

    #[tokio::test(flavor = "current_thread")]
    async fn sensory_session_compacts_after_large_text_turn() {
        let local = LocalSet::new();
        local
            .run_until(async {
                let adapter = MockLlmAdapter::new()
                    .with_text_scenario(text_scenario("", 2))
                    .with_text_scenario(text_scenario("old sensory history summarized", 0));
                let (_blackboard, caps) = test_caps_with_adapter_and_policy(
                    adapter,
                    RuntimePolicy {
                        session_compaction: SessionCompactionPolicy::new(1, 1, 1),
                        ..RuntimePolicy::default()
                    },
                );
                let recorder = SensoryTestRecorder::default();
                let modules = build_recording_sensory(
                    &caps,
                    recorder.clone(),
                    SensoryBurstConfig {
                        silent_window: Duration::from_millis(1),
                        budget: Duration::from_millis(1),
                    },
                    Some(SessionCompactionConfig { prefix_ratio: 0.8 }),
                    (0..10).map(|index| format!("history-{index}")).collect(),
                )
                .await;
                let sensory = caps.host_io().sensory_input_mailbox();

                run_modules(modules, async {
                    sensory
                        .publish(heard("new sound"))
                        .await
                        .expect("sensory subscriber exists");
                    for _ in 0..50 {
                        if recorder
                            .one_shot_session_inputs
                            .borrow()
                            .iter()
                            .any(|items| {
                                matches!(
                                    &items[1],
                                    ModelInputItem::Assistant(lutum::AssistantInputItem::Text(text))
                                        if text.starts_with(COMPACTED_ONE_SHOT_SESSION_PREFIX)
                                )
                            })
                        {
                            break;
                        }
                        tokio::time::sleep(Duration::from_millis(5)).await;
                    }
                })
                .await;

                let sessions = recorder.one_shot_session_inputs.borrow();
                let compacted = sessions
                    .iter()
                    .find(|items| {
                        matches!(
                            &items[1],
                            ModelInputItem::Assistant(lutum::AssistantInputItem::Text(text))
                                if text.starts_with(COMPACTED_ONE_SHOT_SESSION_PREFIX)
                        )
                    })
                    .expect("expected a compacted sensory session");
                let ModelInputItem::Message { role, content } = &compacted[0] else {
                    panic!("expected stable system prompt to remain first");
                };
                assert_eq!(role, &InputMessageRole::System);
                let [MessageContent::Text(system)] = content.as_slice() else {
                    panic!("expected system prompt text");
                };
                assert!(system.contains("You are the one-shot sensory filter"));
                let ModelInputItem::Assistant(lutum::AssistantInputItem::Text(summary)) =
                    &compacted[1]
                else {
                    panic!("expected compacted summary text");
                };
                assert!(summary.contains("old sensory history summarized"));

                let user_texts = compacted
                    .iter()
                    .filter_map(|item| match item {
                        ModelInputItem::Message {
                            role: lutum::InputMessageRole::User,
                            content,
                        } => {
                            let [MessageContent::Text(text)] = content.as_slice() else {
                                panic!("expected one text content item");
                            };
                            Some(text.as_str())
                        }
                        _ => None,
                    })
                    .collect::<Vec<_>>();
                assert!(!user_texts.contains(&"history-0"));
            })
            .await;
    }

    fn count_message_role(items: &[ModelInputItem], expected: InputMessageRole) -> usize {
        items
            .iter()
            .filter(|item| {
                matches!(
                    item,
                    ModelInputItem::Message { role, .. } if role == &expected
                )
            })
            .count()
    }

    fn has_tool_call(items: &[ModelInputItem], expected: &str) -> bool {
        items.iter().any(|item| {
            let ModelInputItem::Turn(turn) = item else {
                return false;
            };
            (0..turn.item_count()).any(|index| {
                turn.item_at(index)
                    .and_then(|item| item.as_tool_call())
                    .is_some_and(|call| call.name.as_str() == expected)
            })
        })
    }

    fn has_tool_result(items: &[ModelInputItem], expected: &str) -> bool {
        items.iter().any(|item| {
            matches!(
                item,
                ModelInputItem::ToolResult(result) if result.name.as_str() == expected
            )
        })
    }

    fn user_texts(items: &[ModelInputItem]) -> Vec<&str> {
        items
            .iter()
            .filter_map(|item| match item {
                ModelInputItem::Message {
                    role: InputMessageRole::User,
                    content,
                } => {
                    let [MessageContent::Text(text)] = content.as_slice() else {
                        panic!("expected one text content item");
                    };
                    Some(text.as_str())
                }
                _ => None,
            })
            .collect()
    }
}
