use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use lutum::Session;
use nuillu_module::{
    AllocationReader, LlmAccess, Memo, Module, SensoryDetailRequest, SensoryDetailRequestInbox,
    SensoryInput, SensoryInputInbox, SessionCompactionConfig, compact_session_if_needed,
    ports::Clock,
};
use serde::Serialize;
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

#[derive(Clone, Debug, Serialize)]
struct PreparedSensoryObservation {
    kind: &'static str,
    direction: Option<String>,
    content: String,
    observed_at: DateTime<Utc>,
    relative_age: String,
    salience: SalienceFeatures,
}

#[derive(Clone, Debug, Serialize)]
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
    detail_requests: SensoryDetailRequestInbox,
    allocation: AllocationReader,
    memo: Memo,
    clock: Arc<dyn Clock>,
    llm: LlmAccess,
    session: Session,
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
        detail_requests: SensoryDetailRequestInbox,
        allocation: AllocationReader,
        memo: Memo,
        clock: Arc<dyn Clock>,
        llm: LlmAccess,
    ) -> Self {
        Self {
            owner: nuillu_types::ModuleId::new(<Self as Module>::id())
                .expect("sensory id is valid"),
            inbox,
            detail_requests,
            allocation,
            memo,
            clock,
            llm,
            session: Session::new(),
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
            nuillu_module::format_system_prompt(
                SYSTEM_PROMPT,
                cx.modules(),
                &self.owner,
                cx.identity_memories(),
                cx.now(),
            )
        })
    }

    fn detail_prompt(&self, cx: &nuillu_module::ActivateCx<'_>) -> &str {
        self.detail_prompt.get_or_init(|| {
            nuillu_module::format_system_prompt(
                DETAIL_PROMPT,
                cx.modules(),
                &self.owner,
                cx.identity_memories(),
                cx.now(),
            )
        })
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
            .map(|input| self.prepare_observation(now, input))
            .collect::<Vec<_>>();
        let allocation = self.allocation.snapshot().await;
        let system_prompt = self.system_prompt(cx).to_owned();
        self.session.push_ephemeral_system(system_prompt);
        self.session.push_user(
            serde_json::json!({
                "new_sensory_batch": observations,
            })
            .to_string(),
        );
        self.session.push_ephemeral_user(
            serde_json::json!({
                "allocation": allocation,
            })
            .to_string(),
        );

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

    fn prepare_observation(
        &mut self,
        now: DateTime<Utc>,
        input: SensoryInput,
    ) -> PreparedSensoryObservation {
        let (kind, direction, raw_content, observed_at) = match &input {
            SensoryInput::Heard {
                direction,
                content,
                observed_at,
            } => ("heard", direction.clone(), content.clone(), *observed_at),
            SensoryInput::Seen {
                direction,
                appearance,
                observed_at,
            } => ("seen", direction.clone(), appearance.clone(), *observed_at),
        };
        let relative_age = Self::format_age(now, observed_at);
        let salience =
            self.salience_features(kind, direction.as_deref(), &raw_content, observed_at);
        PreparedSensoryObservation {
            kind,
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

    async fn handle_detail_request(
        &mut self,
        cx: &nuillu_module::ActivateCx<'_>,
        request: SensoryDetailRequest,
    ) -> Result<()> {
        let allocation = self.allocation.snapshot().await;
        let system_prompt = self.detail_prompt(cx).to_owned();
        self.session.push_ephemeral_system(system_prompt);
        self.session.push_user(
            serde_json::json!({
                "sensory_detail_request": {
                    "question": request.question,
                },
                "retained_sensory_observations": &self.observations,
            })
            .to_string(),
        );
        self.session.push_ephemeral_user(
            serde_json::json!({
                "allocation": allocation,
            })
            .to_string(),
        );

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
        kind: &str,
        direction: Option<&str>,
        content: &str,
        observed_at: DateTime<Utc>,
    ) -> SalienceFeatures {
        let signature = format!(
            "{}:{}:{}",
            kind,
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
                detail_requests: Vec::new(),
            },
            request = self.detail_requests.next_item() => SensoryBatch {
                inputs: Vec::new(),
                detail_requests: vec![request?.body],
            },
        };
        let ready = self.collect_ready_events_into_batch(&mut batch)?;
        if !batch.inputs.is_empty()
            && ready.detail_requests == 0
            && batch.detail_requests.is_empty()
        {
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
                    if ready.detail_requests > 0 {
                        break;
                    }
                }
                request = self.detail_requests.next_item() => {
                    batch.detail_requests.push(request?.body);
                    let _ = self.collect_ready_events_into_batch(batch)?;
                    break;
                }
                _ = tokio::time::sleep(wait_for) => {
                    waited += wait_for;
                    let ready = self.collect_ready_events_into_batch(batch)?;
                    if ready.detail_requests > 0 {
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

        let ready_detail_requests = self.detail_requests.take_ready_items()?;
        let detail_request_count = ready_detail_requests.items.len();
        batch.detail_requests.extend(
            ready_detail_requests
                .items
                .into_iter()
                .map(|envelope| envelope.body),
        );

        Ok(ReadyCounts {
            inputs: input_count,
            detail_requests: detail_request_count,
        })
    }
}

#[derive(Debug, Default)]
pub struct SensoryBatch {
    inputs: Vec<SensoryInput>,
    detail_requests: Vec<SensoryDetailRequest>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct ReadyCounts {
    inputs: usize,
    detail_requests: usize,
}

fn plural(n: u64) -> &'static str {
    if n == 1 { "" } else { "s" }
}

fn is_user_directed_direction(direction: &str) -> bool {
    let direction = direction.to_ascii_lowercase();
    USER_DIRECTED_DIRECTIONS.contains(&direction.as_str())
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
        for request in batch.detail_requests.iter().cloned() {
            self.handle_detail_request(cx, request).await?;
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
        FinishReason, Lutum, MockLlmAdapter, MockTextScenario, RawTextTurnEvent,
        SharedPoolBudgetManager, SharedPoolBudgetOptions, Usage,
    };
    use nuillu_blackboard::{
        ActivationRatio, Blackboard, Bpm, ModuleConfig, ResourceAllocation, linear_ratio_fn,
    };
    use nuillu_module::ports::{
        NoopCognitionLogRepository, NoopFileSearchProvider, NoopMemoryStore, NoopUtteranceSink,
        SystemClock,
    };
    use nuillu_module::{
        CapabilityProviderPorts, CapabilityProviders, LutumTiers, ModuleRegistry,
        render_session_items_for_compaction,
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
        batches: Rc<RefCell<Vec<(usize, usize)>>>,
        rendered_sessions: Rc<RefCell<Vec<String>>>,
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
                .push((batch.inputs.len(), batch.detail_requests.len()));
            <SensoryModule as Module>::activate(&mut self.inner, cx, batch).await?;
            self.recorder.rendered_sessions.borrow_mut().push(
                render_session_items_for_compaction(self.inner.session.input().items()).to_string(),
            );
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
            primary_memory_store: Arc::new(NoopMemoryStore),
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

    fn test_bpm() -> std::ops::RangeInclusive<Bpm> {
        Bpm::from_f64(60_000.0)..=Bpm::from_f64(60_000.0)
    }

    async fn build_recording_sensory(
        caps: &CapabilityProviders,
        recorder: SensoryTestRecorder,
        burst: SensoryBurstConfig,
        compaction: Option<SessionCompactionConfig>,
        session_history: Vec<String>,
    ) -> nuillu_module::AllocatedModules {
        ModuleRegistry::new()
            .register(1..=1, test_bpm(), linear_ratio_fn, move |caps| {
                let mut inner = SensoryModule::new(
                    caps.sensory_input_inbox(),
                    caps.sensory_detail_inbox(),
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
        SensoryInput::Heard {
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

                assert_eq!(recorder.batches.borrow().as_slice(), &[(2, 0)]);
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
                let detail = caps.internal_harness_io().sensory_detail_mailbox();

                run_modules(modules, async {
                    sensory
                        .publish(heard("Koro is standing at the front."))
                        .await
                        .expect("sensory subscriber exists");
                    wait_for_memo_log_count(&blackboard, 1).await;
                    detail
                        .publish(SensoryDetailRequest::new(
                            "What did the agent hear from Koro?",
                        ))
                        .await
                        .expect("sensory detail subscriber exists");
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
                assert_eq!(recorder.batches.borrow().as_slice(), &[(1, 0), (0, 1)]);
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
                        if recorder
                            .rendered_sessions
                            .borrow()
                            .iter()
                            .any(|session| session.contains(COMPACTED_SENSORY_SESSION_PREFIX))
                        {
                            break;
                        }
                        tokio::time::sleep(Duration::from_millis(5)).await;
                    }
                })
                .await;

                let rendered = recorder.rendered_sessions.borrow().join("\n");
                assert!(rendered.contains(COMPACTED_SENSORY_SESSION_PREFIX));
                assert!(rendered.contains("old sensory history summarized"));
                assert!(!rendered.contains("history-0"));
            })
            .await;
    }
}
