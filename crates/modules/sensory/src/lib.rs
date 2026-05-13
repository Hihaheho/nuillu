use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use lutum::Session;
use nuillu_module::{
    AllocationReader, AllocationUpdatedInbox, AmbientSensoryEntry, LlmAccess, Memo, Module,
    SensoryInput, SensoryInputInbox, SensoryModality, SessionCompactionConfig,
    compact_session_if_needed, ports::Clock, seed_persistent_faculty_session,
};
use tokio::time::Instant;

const SYSTEM_PROMPT: &str = r#"You are the sensory module.
You receive batches of raw observations from the environment and decide whether anything is notable
enough to write as sensory memo-log output. Use the deterministic salience features as guidance;
repeated low-change stimuli should usually be ignored or summarized only when the summary is useful.
If there is nothing notable, return an empty response. Otherwise return concise plain text only.
Do not write JSON, YAML, Markdown code fences, headings, or implementation details. Do not write to
the cognition log, memory, or emit utterances."#;

const DETAIL_PROMPT: &str = r#"You are the sensory module answering a detailed sensory request.
Use only retained sensory observations and prior sensory session context. If the observations do not
contain the requested detail, say that the detail is unavailable. Do not infer hidden causes,
intentions, or facts outside sensory observations. Return concise plain text only. Do not write
JSON, YAML, Markdown code fences, headings, or implementation details."#;

const COMPACTED_SENSORY_SESSION_PREFIX: &str = "Compacted sensory session history:";
const SESSION_COMPACTION_PROMPT: &str = r#"You compact the sensory module's persistent session history.
Summarize only the prefix transcript you receive. Preserve observed sensory facts, source/direction
details, relative timing, salience/habituation cues, notable sensory memo-log outputs, detail
requests and answers, ignored/background context that may affect future salience, and uncertainty.
Do not invent facts. Return plain text only."#;

const MAX_OBSERVATIONS: usize = 20;
const DEFAULT_BURST_SILENT_WINDOW: Duration = Duration::from_millis(100);
const DEFAULT_BURST_BUDGET: Duration = Duration::from_secs(1);
const USER_DIRECTED_DIRECTIONS: &[&str] = &["user", "front"];

#[derive(Clone, Debug)]
struct PreparedSensoryObservation {
    modality: SensoryModality,
    ambient: bool,
    direction: Option<String>,
    content: String,
    observed_at: DateTime<Utc>,
    relative_age: String,
    salience: SalienceFeatures,
}

#[derive(Clone, Debug)]
struct SalienceFeatures {
    signature: String,
    repeated: bool,
    repetition_count: u32,
    seconds_since_previous: Option<i64>,
    user_directed: bool,
    novelty_score: f32,
    salience_score: f32,
}

#[derive(Clone, Debug)]
struct StimulusState {
    last_observed_at: DateTime<Utc>,
    repetition_count: u32,
}

pub struct SensoryModule {
    owner: nuillu_types::ModuleId,
    inbox: SensoryInputInbox,
    allocation_updates: AllocationUpdatedInbox,
    allocation: AllocationReader,
    memo: Memo,
    clock: Arc<dyn Clock>,
    llm: LlmAccess,
    session: Session,
    session_seeded: bool,
    session_compaction: SessionCompactionConfig,
    burst: SensoryBurstConfig,
    system_prompt: std::sync::OnceLock<String>,
    detail_prompt: std::sync::OnceLock<String>,
    observations: Vec<String>,
    stimuli: HashMap<String, StimulusState>,
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
        allocation_updates: AllocationUpdatedInbox,
        allocation: AllocationReader,
        memo: Memo,
        clock: Arc<dyn Clock>,
        llm: LlmAccess,
    ) -> Self {
        Self {
            owner: nuillu_types::ModuleId::new(<Self as Module>::id())
                .expect("sensory id is valid"),
            inbox,
            allocation_updates,
            allocation,
            memo,
            clock,
            llm,
            session: Session::new(),
            session_seeded: false,
            session_compaction: SessionCompactionConfig::default(),
            burst: SensoryBurstConfig::default(),
            system_prompt: std::sync::OnceLock::new(),
            detail_prompt: std::sync::OnceLock::new(),
            observations: Vec::new(),
            stimuli: HashMap::new(),
        }
    }

    #[cfg(test)]
    fn with_burst_config(mut self, burst: SensoryBurstConfig) -> Self {
        self.burst = burst;
        self
    }

    fn system_prompt(&self, cx: &nuillu_module::ActivateCx<'_>) -> &str {
        self.system_prompt.get_or_init(|| {
            nuillu_module::format_faculty_system_prompt(SYSTEM_PROMPT, cx.modules(), &self.owner)
        })
    }

    fn detail_prompt(&self, cx: &nuillu_module::ActivateCx<'_>) -> &str {
        self.detail_prompt.get_or_init(|| {
            nuillu_module::format_faculty_system_prompt(DETAIL_PROMPT, cx.modules(), &self.owner)
        })
    }

    fn ensure_session_seeded(&mut self, cx: &nuillu_module::ActivateCx<'_>) {
        if self.session_seeded {
            return;
        }
        let system_prompt = self.system_prompt(cx).to_owned();
        seed_persistent_faculty_session(
            &mut self.session,
            system_prompt,
            cx.identity_memories(),
            cx.now(),
        );
        self.session_seeded = true;
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
        let observations = inputs
            .iter()
            .cloned()
            .flat_map(|input| self.prepare_input_observations(now, input))
            .collect::<Vec<_>>();
        self.ensure_session_seeded(cx);
        let allocation = self.allocation.snapshot().await;
        let guidance = allocation.for_module(&self.owner).guidance;
        self.session.push_user(format_sensory_batch(&observations));
        if let Some(context) = format_sensory_guidance_context(&guidance) {
            self.session.push_ephemeral_system(context);
        }

        let lutum = self.llm.lutum().await;
        let result = self
            .session
            .text_turn(&lutum)
            .collect()
            .await
            .context("sensory text turn failed")?;
        compact_session_if_needed(
            &mut self.session,
            result.usage.input_tokens,
            cx.session_compaction_lutum(),
            self.session_compaction,
            Self::id(),
            COMPACTED_SENSORY_SESSION_PREFIX,
            SESSION_COMPACTION_PROMPT,
        )
        .await;

        let text = result.assistant_text();
        let text = text.trim();
        if !text.is_empty() {
            self.remember_observation(text);
            self.memo.write(text.to_owned()).await;
        }
        Ok(())
    }

    fn prepare_input_observations(
        &mut self,
        now: DateTime<Utc>,
        input: SensoryInput,
    ) -> Vec<PreparedSensoryObservation> {
        match input {
            SensoryInput::Observed {
                modality,
                direction,
                content,
                observed_at,
            } => vec![self.prepare_observation(
                now,
                modality,
                false,
                direction,
                content,
                observed_at,
            )],
            SensoryInput::AmbientSnapshot {
                entries,
                observed_at,
            } => entries
                .into_iter()
                .map(|entry| self.prepare_ambient_observation(now, entry, observed_at))
                .collect(),
        }
    }

    fn prepare_ambient_observation(
        &mut self,
        now: DateTime<Utc>,
        entry: AmbientSensoryEntry,
        observed_at: DateTime<Utc>,
    ) -> PreparedSensoryObservation {
        self.prepare_observation(
            now,
            entry.modality,
            true,
            Some(format!("ambient:{}", entry.id)),
            entry.content,
            observed_at,
        )
    }

    fn prepare_observation(
        &mut self,
        now: DateTime<Utc>,
        modality: SensoryModality,
        ambient: bool,
        direction: Option<String>,
        raw_content: String,
        observed_at: DateTime<Utc>,
    ) -> PreparedSensoryObservation {
        let relative_age = Self::format_age(now, observed_at);
        let salience = self.salience_features(
            modality.as_str(),
            ambient,
            direction.as_deref(),
            &raw_content,
            observed_at,
        );
        PreparedSensoryObservation {
            modality,
            ambient,
            direction,
            content: raw_content,
            observed_at,
            relative_age,
            salience,
        }
    }

    fn remember_observation(&mut self, text: &str) {
        self.observations.push(text.to_owned());
        if self.observations.len() > MAX_OBSERVATIONS {
            self.observations.remove(0);
        }
    }

    async fn handle_detail_guidance(&mut self, cx: &nuillu_module::ActivateCx<'_>) -> Result<()> {
        let allocation = self.allocation.snapshot().await;
        let guidance = allocation.for_module(&self.owner).guidance;
        if guidance.trim().is_empty() {
            return Ok(());
        }
        self.ensure_session_seeded(cx);
        let detail_prompt = self.detail_prompt(cx).to_owned();
        self.session.push_ephemeral_developer(detail_prompt);
        self.session
            .push_user(format_sensory_detail_request(&guidance, &self.observations));

        let lutum = self.llm.lutum().await;
        let result = self
            .session
            .text_turn(&lutum)
            .collect()
            .await
            .context("sensory detail text turn failed")?;
        compact_session_if_needed(
            &mut self.session,
            result.usage.input_tokens,
            cx.session_compaction_lutum(),
            self.session_compaction,
            Self::id(),
            COMPACTED_SENSORY_SESSION_PREFIX,
            SESSION_COMPACTION_PROMPT,
        )
        .await;

        let text = result.assistant_text();
        let text = text.trim();
        if !text.is_empty() {
            self.memo.write(text.to_owned()).await;
        }
        Ok(())
    }

    fn salience_features(
        &mut self,
        modality: &str,
        ambient: bool,
        direction: Option<&str>,
        content: &str,
        observed_at: DateTime<Utc>,
    ) -> SalienceFeatures {
        let signature = format!(
            "{}:{}:{}:{}",
            if ambient { "ambient" } else { "observed" },
            modality,
            direction.unwrap_or_default().to_ascii_lowercase(),
            content.trim().to_ascii_lowercase()
        );
        let previous = self.stimuli.get(&signature).cloned();
        let repetition_count = previous
            .as_ref()
            .map(|state| state.repetition_count.saturating_add(1))
            .unwrap_or(1);
        let seconds_since_previous = previous
            .as_ref()
            .map(|state| (observed_at - state.last_observed_at).num_seconds().max(0));
        let repeated = previous.is_some();
        let user_directed = direction.map(is_user_directed_direction).unwrap_or(false);
        let novelty_score = if repeated { 0.15 } else { 0.85 };
        let repetition_penalty = if repeated {
            (repetition_count.saturating_sub(1) as f32 * 0.15).min(0.45)
        } else {
            0.0
        };
        let salience_score = (novelty_score + if user_directed { 0.2 } else { 0.0 }
            - repetition_penalty)
            .clamp(0.0, 1.0);

        self.stimuli.insert(
            signature.clone(),
            StimulusState {
                last_observed_at: observed_at,
                repetition_count,
            },
        );

        SalienceFeatures {
            signature,
            repeated,
            repetition_count,
            seconds_since_previous,
            user_directed,
            novelty_score,
            salience_score,
        }
    }

    async fn next_batch(&mut self) -> Result<SensoryBatch> {
        let mut batch = tokio::select! {
            input = self.inbox.next_item() => SensoryBatch {
                inputs: vec![input?.body],
                allocation_updated: false,
            },
            update = self.allocation_updates.next_item() => {
                let _ = update?;
                SensoryBatch {
                inputs: Vec::new(),
                    allocation_updated: true,
                }
            }
        };
        let ready = self.collect_ready_events_into_batch(&mut batch)?;
        if !batch.inputs.is_empty() && ready.allocation_updates == 0 && !batch.allocation_updated {
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
                    let ready = self.collect_ready_events_into_batch(batch)?;
                    if ready.allocation_updates > 0 {
                        break;
                    }
                }
                update = self.allocation_updates.next_item() => {
                    let _ = update?;
                    batch.allocation_updated = true;
                    let _ = self.collect_ready_events_into_batch(batch)?;
                    break;
                }
                _ = tokio::time::sleep(wait_for) => {
                    waited += wait_for;
                    let ready = self.collect_ready_events_into_batch(batch)?;
                    if ready.allocation_updates > 0 {
                        break;
                    }
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

        let allocation_updates = self.allocation_updates.take_ready_items()?.items.len();
        if allocation_updates > 0 {
            batch.allocation_updated = true;
        }

        Ok(ReadyCounts {
            inputs: input_count,
            allocation_updates,
        })
    }
}

#[derive(Debug, Default)]
pub struct SensoryBatch {
    inputs: Vec<SensoryInput>,
    allocation_updated: bool,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct ReadyCounts {
    inputs: usize,
    allocation_updates: usize,
}

fn plural(n: u64) -> &'static str {
    if n == 1 { "" } else { "s" }
}

fn is_user_directed_direction(direction: &str) -> bool {
    let direction = direction.to_ascii_lowercase();
    USER_DIRECTED_DIRECTIONS.contains(&direction.as_str())
}

fn format_sensory_batch(observations: &[PreparedSensoryObservation]) -> String {
    let mut out = String::from("New sensory observations:");
    if observations.is_empty() {
        out.push_str("\n- none");
        return out;
    }
    if observations.iter().any(|observation| observation.ambient) {
        out.push_str(
            "\nAmbient snapshot entries are enabled background rows only; omitted rows are not evidence that a condition disappeared.",
        );
    }
    for observation in observations {
        let mode = if observation.ambient {
            "ambient"
        } else {
            "observed"
        };
        let direction = observation
            .direction
            .as_deref()
            .map(|direction| format!(" from {direction}"))
            .unwrap_or_default();
        out.push_str(&format!(
            "\n- {} {}{} observed {} ({}): {}",
            mode,
            observation.modality.as_str(),
            direction,
            observation.relative_age,
            observation.observed_at.to_rfc3339(),
            observation.content.trim()
        ));
        out.push_str(&format!(
            "\n  salience: signature={}; repeated={}; repetition_count={}; seconds_since_previous={}; user_directed={}; novelty_score={:.2}; salience_score={:.2}",
            observation.salience.signature,
            if observation.salience.repeated { "yes" } else { "no" },
            observation.salience.repetition_count,
            observation
                .salience
                .seconds_since_previous
                .map(|secs| secs.to_string())
                .unwrap_or_else(|| "none".to_owned()),
            if observation.salience.user_directed { "yes" } else { "no" },
            observation.salience.novelty_score,
            observation.salience.salience_score,
        ));
    }
    out
}

fn format_sensory_guidance_context(guidance: &str) -> Option<String> {
    let guidance = guidance.trim();
    if guidance.is_empty() {
        return None;
    }
    Some(format!(
        "Sensory context for judging the new observations:\nCurrent sensory guidance: {guidance}"
    ))
}

fn format_sensory_detail_request(guidance: &str, observations: &[String]) -> String {
    let mut out = format!("Sensory detail request:\n{}", guidance.trim());
    out.push_str("\n\nRetained sensory observations:");
    if observations.is_empty() {
        out.push_str("\n- none");
    } else {
        for observation in observations {
            out.push_str("\n- ");
            out.push_str(observation.trim());
        }
    }
    out
}

#[async_trait(?Send)]
impl Module for SensoryModule {
    type Batch = SensoryBatch;

    fn id() -> &'static str {
        "sensory"
    }

    fn role_description() -> &'static str {
        "Pre-attentive filter for the agent's senses: receives sights, sounds, and other external observations from the world, scores their salience, and writes selected normalized observations to its memo."
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
        if batch.allocation_updated {
            self.handle_detail_guidance(cx).await?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::cell::RefCell;
    use std::rc::Rc;

    use lutum::{
        FinishReason, InputMessageRole, Lutum, MessageContent, MockLlmAdapter, MockTextScenario,
        ModelInputItem, RawTextTurnEvent, SharedPoolBudgetManager, SharedPoolBudgetOptions, Usage,
    };
    use nuillu_blackboard::{
        ActivationRatio, Blackboard, BlackboardCommand, Bpm, ModuleConfig, ResourceAllocation,
        linear_ratio_fn,
    };
    use nuillu_module::ports::{NoopCognitionLogRepository, SystemClock};
    use nuillu_module::{
        AllocationUpdated, CapabilityProviderPorts, CapabilityProviders, LutumTiers, ModuleRegistry,
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
        batches: Rc<RefCell<Vec<(usize, bool)>>>,
        session_inputs: Rc<RefCell<Vec<Vec<ModelInputItem>>>>,
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

        fn role_description() -> &'static str {
            SensoryModule::role_description()
        }

        async fn next_batch(&mut self) -> Result<Self::Batch> {
            self.inner.next_batch().await
        }

        async fn activate(
            &mut self,
            cx: &nuillu_module::ActivateCx<'_>,
            batch: &Self::Batch,
        ) -> Result<()> {
            self.recorder
                .batches
                .borrow_mut()
                .push((batch.inputs.len(), batch.allocation_updated));
            <SensoryModule as Module>::activate(&mut self.inner, cx, batch).await?;
            self.recorder
                .session_inputs
                .borrow_mut()
                .push(self.inner.session.input().items().to_vec());
            Ok(())
        }
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
        let blackboard = Blackboard::with_allocation(sensory_allocation());
        let adapter = Arc::new(adapter);
        let budget = SharedPoolBudgetManager::new(SharedPoolBudgetOptions::default());
        let lutum = Lutum::new(adapter, budget);
        let caps = CapabilityProviders::new(CapabilityProviderPorts {
            blackboard: blackboard.clone(),
            cognition_log_port: Arc::new(NoopCognitionLogRepository),
            clock: Arc::new(SystemClock),
            tiers: LutumTiers {
                cheap: lutum.clone(),
                default: lutum.clone(),
                premium: lutum,
            },
        });
        (blackboard, caps)
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
        compaction: Option<SessionCompactionConfig>,
        session_history: Vec<String>,
    ) -> nuillu_module::AllocatedModules {
        ModuleRegistry::new()
            .register(test_policy(), move |caps| {
                let mut inner = SensoryModule::new(
                    caps.sensory_input_inbox(),
                    caps.allocation_updated_inbox(),
                    caps.allocation_reader(),
                    caps.memo(),
                    caps.clock(),
                    caps.llm_access(),
                )
                .with_burst_config(burst);
                if let Some(compaction) = compaction {
                    inner.session_compaction = compaction;
                }
                for item in &session_history {
                    inner.session.push_user(item.clone());
                }
                RecordingSensoryModule {
                    inner,
                    recorder: recorder.clone(),
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
        nuillu_agent::run(
            modules,
            nuillu_agent::AgentEventLoopConfig {
                idle_threshold: Duration::from_millis(50),
                activate_retries: 2,
            },
            body,
        )
        .await
        .expect("sensory test runtime should not fail");
    }

    fn heard(content: &str) -> SensoryInput {
        SensoryInput::Observed {
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
    fn user_directed_direction_is_generic_not_username_specific() {
        assert!(is_user_directed_direction("user"));
        assert!(is_user_directed_direction("front"));
        assert!(is_user_directed_direction("USER"));
        assert!(!is_user_directed_direction("ryo"));
        assert!(!is_user_directed_direction("left"));
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
    fn ambient_snapshot_format_marks_background_context() {
        let observations = vec![PreparedSensoryObservation {
            modality: SensoryModality::Smell,
            ambient: true,
            direction: Some("ambient:row-1".to_string()),
            content: "wet stone smell".to_string(),
            observed_at: reference_now(),
            relative_age: "0 seconds ago".to_string(),
            salience: SalienceFeatures {
                signature: "ambient:smell:ambient:row-1:wet stone smell".to_string(),
                repeated: false,
                repetition_count: 1,
                seconds_since_previous: None,
                user_directed: false,
                novelty_score: 0.85,
                salience_score: 0.85,
            },
        }];
        let text = format_sensory_batch(&observations);

        assert!(text.contains("Ambient snapshot entries are enabled background rows only"));
        assert!(text.contains("ambient smell"));
        assert!(text.contains("wet stone smell"));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn burst_batches_inputs_arriving_inside_silent_window() {
        let local = LocalSet::new();
        local
            .run_until(async {
                let mut adapter = MockLlmAdapter::new();
                for _ in 0..10 {
                    adapter = adapter.with_text_scenario(text_scenario("batched sensory note", 0));
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

                assert_eq!(recorder.batches.borrow().as_slice(), &[(2, false)]);
            })
            .await;
    }

    #[tokio::test(flavor = "current_thread")]
    async fn input_and_detail_outputs_write_separate_plaintext_memo_logs() {
        let local = LocalSet::new();
        local
            .run_until(async {
                let adapter = MockLlmAdapter::new()
                    .with_text_scenario(text_scenario("Koro is standing at the front.", 0))
                    .with_text_scenario(text_scenario("Koro growled from the front.", 0));
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
                        .publish(heard("Koro is standing at the front."))
                        .await
                        .expect("sensory subscriber exists");
                    wait_for_memo_log_count(&blackboard, 1).await;
                    let mut allocation = sensory_allocation();
                    let mut config = allocation.for_module(&builtin::sensory());
                    config.guidance = "What did the agent hear from Koro?".into();
                    allocation.set(builtin::sensory(), config);
                    blackboard
                        .apply(BlackboardCommand::SetAllocation(allocation))
                        .await;
                    caps.internal_harness_io()
                        .allocation_updated_mailbox()
                        .publish(AllocationUpdated)
                        .await
                        .expect("allocation update subscriber exists");
                    wait_for_memo_log_count(&blackboard, 2).await;
                })
                .await;

                let logs = blackboard.read(|bb| bb.recent_memo_logs()).await;
                let contents = logs
                    .iter()
                    .map(|record| record.content.as_str())
                    .collect::<Vec<_>>();
                assert_eq!(
                    contents,
                    vec![
                        "Koro is standing at the front.",
                        "Koro growled from the front.",
                    ]
                );
                assert_eq!(
                    recorder.batches.borrow().as_slice(),
                    &[(1, false), (0, true)]
                );
                let sessions = recorder.session_inputs.borrow();
                let first_session = sessions.first().expect("first sensory activation recorded");
                let ModelInputItem::Message { role, content } = &first_session[0] else {
                    panic!("expected sensory system prompt as first session item");
                };
                assert_eq!(role, &InputMessageRole::System);
                let [MessageContent::Text(system)] = content.as_slice() else {
                    panic!("expected sensory system prompt text");
                };
                assert!(system.contains("You are the sensory module"));
            })
            .await;
    }

    #[tokio::test(flavor = "current_thread")]
    async fn sensory_session_compacts_after_large_text_turn() {
        let local = LocalSet::new();
        local
            .run_until(async {
                let adapter = MockLlmAdapter::new()
                    .with_text_scenario(text_scenario("Notable sensory change.", 2))
                    .with_text_scenario(text_scenario("old sensory history summarized", 0));
                let (_blackboard, caps) = test_caps_with_adapter(adapter);
                let recorder = SensoryTestRecorder::default();
                let modules = build_recording_sensory(
                    &caps,
                    recorder.clone(),
                    SensoryBurstConfig {
                        silent_window: Duration::from_millis(1),
                        budget: Duration::from_millis(1),
                    },
                    Some(SessionCompactionConfig {
                        input_token_threshold: 1,
                        prefix_ratio: 0.8,
                    }),
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
                        if recorder.session_inputs.borrow().iter().any(|items| {
                            matches!(
                                &items[0],
                                ModelInputItem::Message { content, .. }
                                    if matches!(
                                        content.as_slice(),
                                        [MessageContent::Text(text)]
                                            if text.starts_with(COMPACTED_SENSORY_SESSION_PREFIX)
                                    )
                            )
                        }) {
                            break;
                        }
                        tokio::time::sleep(Duration::from_millis(5)).await;
                    }
                })
                .await;

                let sessions = recorder.session_inputs.borrow();
                let compacted = sessions
                    .iter()
                    .find(|items| {
                        matches!(
                            &items[0],
                            ModelInputItem::Message { content, .. }
                                if matches!(
                                    content.as_slice(),
                                    [MessageContent::Text(text)]
                                        if text.starts_with(COMPACTED_SENSORY_SESSION_PREFIX)
                                )
                        )
                    })
                    .expect("expected a compacted sensory session");
                let ModelInputItem::Message { content, .. } = &compacted[0] else {
                    panic!("expected compacted summary message");
                };
                let [MessageContent::Text(summary)] = content.as_slice() else {
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
}
