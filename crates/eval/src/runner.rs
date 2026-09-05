use std::{
    any::Any,
    collections::{BTreeMap, HashSet, VecDeque},
    fs::{File, OpenOptions},
    io::{self, Write},
    num::NonZeroUsize,
    panic::AssertUnwindSafe,
    path::{Path, PathBuf},
    rc::Rc,
    sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

use anyhow::{Context, Result};
use async_trait::async_trait;
use chrono::{DateTime, Duration as ChronoDuration, FixedOffset, Utc};
use futures::{FutureExt as _, StreamExt as _, future::LocalBoxFuture, stream::FuturesUnordered};
use lutum::{LutumHooksSet, ModelInputHookContext, OnModelInput};
use lutum_eval::{RawTraceSnapshot, TraceSnapshot};
use lutum_libsql_adapter::{LibsqlAgentStore, LibsqlAgentStoreConfig};
use nuillu_agent::{AgentEventLoopConfig, AgentRunController, run_controlled as run_agent};
use nuillu_blackboard::{
    ActivationRatio, Blackboard, BlackboardCommand, BlackboardInner, Bpm, CognitionLogEntry,
    CognitionLogEntryRecord, CognitionLogOrigin, MemoLogRecord, MemoryMetadata, ModulePolicy,
    ModuleRunStatus, PolicyMetaPatch, ResourceAllocation, ZeroReplicaWindowPolicy,
};
use nuillu_memory::{
    LinkedMemoryQuery, MemoryCapabilities, MemoryLinkDirection, MemoryLinkRelation,
    MemoryNamespace, MemoryQuery, MemoryRecord, MemoryStore, NewMemoryLink,
};
use nuillu_module::ports::{Clock, CognitionLogRepository, PortError, SystemClock};
use nuillu_module::{
    AmbientSensoryEntry, CapabilityProviderConfig, CapabilityProviderPorts,
    CapabilityProviderRuntime, CapabilityProviders, CognitionLogUpdated, ExternalActionExecutor,
    ExternalActionInvocation, ExternalActionInvocationResult, InternalHarnessIo,
    InteroceptionRuntimePolicy, LlmConcurrencyPool, Participant, RuntimeEvent, RuntimeEventSink,
    RuntimePolicy, SceneRegistry, SensoryInput, SensoryInputMailbox, SensoryModality,
    SessionCompactionPolicy,
};
use nuillu_reward::{IndexedPolicy, PolicyCapabilities, PolicyRecord, PolicyStore};
use nuillu_speak::{Utterance, UtteranceDelta, UtteranceSink};
use nuillu_types::{
    MemoryIndex, MemoryRank, ModuleId, ModuleInstanceId, PolicyIndex, PolicyRank, ReplicaCapRange,
    ReplicaIndex, SignedUnitF32, UnitF32, builtin,
};
use nuillu_visualizer_protocol::{
    AllocationView, BlackboardSnapshot, CognitionEntryView, CognitionLogView, InteroceptionView,
    MemoView, MemoryMetadataView, MemoryRecordScope, MemoryRecordView, ModuleSettingsView,
    ModuleStatusView, PersistedCognitionEntryView, TabStatus, UtteranceDeltaView,
    UtteranceProgressView, UtteranceView, VisualizerAction, VisualizerClientMessage,
    VisualizerCommand, VisualizerErrorView, VisualizerEvent, VisualizerServerMessage,
    VisualizerTabId, ZeroReplicaWindowView, start_activation_action_id,
};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tokio::task::LocalSet;

use crate::{
    artifact::CaseArtifact,
    cases::{
        ActivateAllocation, ArtifactTextField, Assertion, CaseFileError, EvalCase,
        EvalInteroceptiveMode, EvalLimits, EvalStep, MemoryLinkSeed, PolicySeed, RuntimeCase,
        Stimulus, WaitFor, discover_case_files, parse_case_file, parse_case_now,
        parse_memory_datetime, parse_scope_id, wake_arousal_max_is_set, wake_arousal_min_is_set,
    },
    evaluation::{
        AssertionOutcome, CaseReport, CaseSummary, CaseTiming, CaseTrialSummary,
        MeasurementStatistics, ModuleActivationRecord, SuiteMetrics, SuiteModelNames, SuiteReport,
        SuiteRunReport, SuiteTiming, aggregate_trial_timing, artifact_text,
        build_activation_timeline, evaluate_assertion, evaluate_case_with_overrides, field_label,
        normalize_text_block, numeric_range_outcome, pointer_number, pointer_text,
    },
    judge::{LlmRubricJudge, RubricJudge},
    state_dump::{
        AgenticDeadlockDump, AllocationModuleDump, AllocationProposalDump, BlackboardLastStateDump,
        CognitionEntryDump, CognitionLogDump, DumpText, InteroceptionDump, MemoLogDump,
        MemoryEntryDump, MemoryLastStateDump, MemoryMetadataDump, ModuleInstanceDump,
        ReplicaCapDump, RuntimeLastStateCaseDump, RuntimeLastStateDump, UtteranceDump,
        render_runtime_last_state_eure,
    },
    trace_json::{raw_trace_has_error, raw_trace_snapshot_json, trace_snapshot_json},
};

const IDLE_REPORT_INTERVAL: Duration = Duration::from_secs(30);
const RUNTIME_ACTION_SILENCE_WINDOW: Duration = Duration::from_millis(200);
const RUNTIME_SILENCE_WINDOW: Duration = Duration::from_millis(200);
const RUNTIME_IDLE_TIMEOUT: Duration = Duration::from_secs(5);
const RUNTIME_STEP_SETTLE_TIMEOUT: Duration = Duration::from_secs(12);
const EVAL_POLL_INTERVAL: Duration = Duration::from_millis(100);
const EVAL_MEMO_RETAINED_PER_OWNER: usize = 8;
const EVAL_COGNITION_LOG_RETAINED_ENTRIES: usize = 16;

pub use nuillu_server::{
    EmbeddingBackendConfig, LlmBackendConfig, LlmGenerationConfig, model_concurrency_from_backends,
};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord)]
pub enum LiveOutput {
    Quiet,
    #[default]
    Normal,
    Verbose,
}

#[derive(Clone)]
pub struct RunnerConfig {
    pub cases_root: PathBuf,
    pub output_root: PathBuf,
    pub llm_log_root: PathBuf,
    pub run_id: String,
    pub judge_backend: LlmBackendConfig,
    pub cheap_backend: LlmBackendConfig,
    pub default_backend: LlmBackendConfig,
    pub premium_backend: LlmBackendConfig,
    pub image_backend: LlmBackendConfig,
    pub embedding_backend: EmbeddingBackendConfig,
    pub fail_fast: bool,
    pub failed_only: bool,
    pub failed_from: Option<PathBuf>,
    pub model_concurrency: BTreeMap<String, Option<NonZeroUsize>>,
    pub llm_concurrency_pool: LlmConcurrencyPool,
    pub trials: NonZeroUsize,
    pub case_concurrency: NonZeroUsize,
    pub case_patterns: Vec<String>,
    /// Implementations for user-defined module ids referenced by runtime configs.
    pub module_factories: Vec<Arc<dyn nuillu_server::ServerModuleFactory>>,
    pub runtime_config_override: Option<PathBuf>,
    pub live_output: LiveOutput,
}

pub struct RunnerHooks {
    pub visualizer: Option<VisualizerHook>,
}

impl RunnerHooks {
    pub fn none() -> Self {
        Self { visualizer: None }
    }

    pub fn with_visualizer(visualizer: VisualizerHook) -> Self {
        Self {
            visualizer: Some(visualizer),
        }
    }
}

pub struct VisualizerHook {
    events: std::sync::mpsc::Sender<VisualizerServerMessage>,
    commands: std::sync::mpsc::Receiver<VisualizerClientMessage>,
    memory_cache: BTreeMap<String, Vec<MemoryRecordView>>,
    shutdown_requested: bool,
}

impl VisualizerHook {
    pub fn new(
        events: std::sync::mpsc::Sender<VisualizerServerMessage>,
        commands: std::sync::mpsc::Receiver<VisualizerClientMessage>,
    ) -> Self {
        Self {
            events,
            commands,
            memory_cache: BTreeMap::new(),
            shutdown_requested: false,
        }
    }

    pub fn event_sender(&self) -> VisualizerEventSink {
        VisualizerEventSink::new(self.events.clone())
    }

    pub(crate) fn send_event(&self, event: VisualizerEvent) {
        let _ = self.events.send(VisualizerServerMessage::event(event));
    }

    fn offer_action(&self, action: VisualizerAction) {
        let _ = self
            .events
            .send(VisualizerServerMessage::OfferAction { action });
    }

    fn revoke_action(&self, action_id: String) {
        let _ = self
            .events
            .send(VisualizerServerMessage::RevokeAction { action_id });
    }

    pub(crate) fn request_shutdown(&mut self) {
        self.shutdown_requested = true;
    }

    pub fn shutdown_requested(&self) -> bool {
        self.shutdown_requested
    }

    fn set_memory_cache(&mut self, case_id: &str, records: Vec<MemoryRecordView>) {
        self.memory_cache.insert(case_id.to_string(), records);
    }

    fn cached_memory_records(
        &self,
        case_id: &str,
        scope: &MemoryRecordScope,
        offset: usize,
        limit: usize,
    ) -> (Vec<MemoryRecordView>, bool) {
        let records = self
            .memory_cache
            .get(case_id)
            .map(Vec::as_slice)
            .unwrap_or_default();
        let matches = match scope {
            MemoryRecordScope::Latest => records.to_vec(),
            MemoryRecordScope::Search { query } => {
                let needle = query.to_lowercase();
                records
                    .iter()
                    .filter(|record| record.content.to_lowercase().contains(&needle))
                    .cloned()
                    .collect()
            }
        };
        memory_chunk_from_records(&matches, offset, limit)
    }

    fn drain_cached_commands_until_shutdown(&mut self) {
        while let Ok(message) = self.commands.recv() {
            let VisualizerClientMessage::Command { command } = message else {
                continue;
            };
            match command {
                VisualizerCommand::Shutdown => {
                    self.request_shutdown();
                    break;
                }
                VisualizerCommand::LoadMemoryRecords {
                    tab_id,
                    scope,
                    offset,
                    limit,
                } => {
                    let (records, has_more) =
                        self.cached_memory_records(tab_id.as_str(), &scope, offset, limit);
                    self.send_event(VisualizerEvent::MemoryRecordsLoaded {
                        tab_id,
                        scope,
                        offset,
                        records,
                        has_more,
                    });
                }
                VisualizerCommand::PublishSensoryInput { tab_id, .. }
                | VisualizerCommand::SendOneShotSensoryInput { tab_id, .. } => {
                    self.send_event(VisualizerEvent::Log {
                        tab_id,
                        message: "eval case is no longer running".to_string(),
                    });
                }
                VisualizerCommand::RequestSnapshot { tab_id }
                | VisualizerCommand::CreateAmbientSensoryRow { tab_id, .. }
                | VisualizerCommand::UpdateAmbientSensoryRow { tab_id, .. }
                | VisualizerCommand::RemoveAmbientSensoryRow { tab_id, .. }
                | VisualizerCommand::CreateSceneRow { tab_id, .. }
                | VisualizerCommand::UpdateSceneRow { tab_id, .. }
                | VisualizerCommand::RemoveSceneRow { tab_id, .. }
                | VisualizerCommand::SaveSceneState { tab_id, .. }
                | VisualizerCommand::SendScenePersonMessage { tab_id, .. }
                | VisualizerCommand::LoadLinkedMemories { tab_id, .. }
                | VisualizerCommand::DeleteMemory { tab_id, .. }
                | VisualizerCommand::SetModuleDisabled { tab_id, .. }
                | VisualizerCommand::SetModuleSettings { tab_id, .. }
                | VisualizerCommand::SetAgentActionAffordances { tab_id, .. }
                | VisualizerCommand::UpsertAgentActionAffordance { tab_id, .. }
                | VisualizerCommand::RemoveAgentActionAffordance { tab_id, .. }
                | VisualizerCommand::CompleteAgentActionInvocation { tab_id, .. }
                | VisualizerCommand::ResetModuleSessionHistory { tab_id, .. }
                | VisualizerCommand::LoadActivityRows { tab_id, .. }
                | VisualizerCommand::LoadLlmTranscriptTurns { tab_id, .. }
                | VisualizerCommand::LoadCognitionLogEntries { tab_id, .. } => {
                    self.send_event(VisualizerEvent::Log {
                        tab_id,
                        message: "eval case is no longer running".to_string(),
                    });
                }
            }
        }
    }
}

use nuillu_server::{
    LlmLogContext, VisualizerEventSink, VisualizerLlmObserver, build_embedder, build_model_handle,
    build_tiers, duration_millis_u64, linked_memory_record_view, memory_rank_name,
    memory_record_view, module_policy_views,
};

#[derive(Debug, Clone)]
pub struct CaseRunOutput {
    pub case_path: PathBuf,
    pub output_dir: PathBuf,
    pub summary: CaseSummary,
    pub artifact: CaseArtifact,
    pub events: Vec<RuntimeEvent>,
    pub trace: TraceSnapshot,
    pub raw_trace: RawTraceSnapshot,
}

#[derive(Debug, Error)]
pub enum RunnerError {
    #[error(transparent)]
    Case(#[from] CaseFileError),
    #[error("failed to discover eval cases under {path}: {source}")]
    DiscoverCases {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("case runner failed for {path}: {message}")]
    Driver { path: PathBuf, message: String },
    #[error("failed to write eval output under {path}: {source}")]
    WriteOutput {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to install lutum trace subscriber: {message}")]
    TraceSubscriber { message: String },
    #[error("case patterns matched no eval cases: {patterns}")]
    NoCasesMatched { patterns: String },
    #[error("failed-only requested but no suite-report.json files were found under {path}")]
    FailedOnlyNoReference { path: PathBuf },
    #[error("failed-only reference report not found at {path}")]
    FailedOnlyReferenceNotFound { path: PathBuf },
    #[error("failed to discover failed-only reference reports under {path}: {source}")]
    DiscoverFailedOnlyReference {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to read failed-only reference report {path}: {source}")]
    ReadFailedOnlyReference {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to parse failed-only reference report {path}: {source}")]
    ParseFailedOnlyReference {
        path: PathBuf,
        #[source]
        source: serde_json::Error,
    },
    #[error(
        "failed-only reference case is not present under current cases root: id={id} path={path}"
    )]
    FailedOnlyCaseNotFound { id: String, path: String },
    #[error("--gui does not support --trials > 1 (got {trials})")]
    GuiTrialsUnsupported { trials: usize },
}

struct CaseExecution {
    artifact: CaseArtifact,
    events: Vec<RuntimeEvent>,
    activations: Vec<ModuleActivationRecord>,
    assertion_overrides: BTreeMap<String, AssertionOutcome>,
}

struct CaseOutputContext<'a> {
    case_path: &'a Path,
    output_dir: &'a Path,
    case: &'a EvalCase,
    id: &'a str,
    runtime_id: &'a str,
    trial_number: usize,
    reporter: &'a LiveReporter,
    llm_log_directory: Option<PathBuf>,
}

#[derive(Debug, Clone)]
struct CaseSelection {
    case_paths: Vec<PathBuf>,
    failed_from: Option<PathBuf>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct CaseIdentity {
    path: String,
    id: String,
}

#[derive(Debug, Clone)]
struct EvalWorkItem {
    case_order: usize,
    case_path: PathBuf,
    case: EvalCase,
    id: String,
    case_output_dir: PathBuf,
    output_dir: PathBuf,
    runtime_id: String,
    trial_number: usize,
    trial_count: usize,
}

#[derive(Debug, Clone)]
struct EvalWorkOutput {
    item: EvalWorkItem,
    output: CaseRunOutput,
    started_at: Instant,
    completed_at: Instant,
}

#[derive(Debug, Deserialize)]
struct FailedOnlySuiteReport {
    cases: Vec<FailedOnlyCaseSummary>,
}

#[derive(Debug, Deserialize)]
struct FailedOnlyCaseSummary {
    path: String,
    id: String,
    passed: bool,
    invalid: bool,
}

pub async fn run_suite(config: &RunnerConfig) -> Result<SuiteReport, RunnerError> {
    let mut hooks = RunnerHooks::none();
    run_suite_with_hooks(config, &mut hooks).await
}

pub(crate) fn visualizer_planned_tabs(
    config: &RunnerConfig,
) -> Result<Vec<(VisualizerTabId, String)>, RunnerError> {
    select_case_paths(config)?
        .case_paths
        .into_iter()
        .map(|path| {
            let case = parse_case_file(&path)?;
            let id = case_id(&path, &case);
            Ok((VisualizerTabId::new(id.clone()), id))
        })
        .collect()
}

pub async fn run_suite_with_hooks(
    config: &RunnerConfig,
    hooks: &mut RunnerHooks,
) -> Result<SuiteReport, RunnerError> {
    let suite_started = Instant::now();
    install_trace_subscriber_for_runner()?;
    if hooks.visualizer.is_some() && config.trials.get() > 1 {
        return Err(RunnerError::GuiTrialsUnsupported {
            trials: config.trials.get(),
        });
    }
    let selection = select_case_paths(config)?;
    let case_paths = selection.case_paths;
    let run_dir = config.output_root.join(&config.run_id);
    let planned_case_count = case_paths.len();
    let run_report = suite_run_report(
        config,
        &run_dir,
        planned_case_count,
        selection.failed_from.as_deref(),
    );
    std::fs::create_dir_all(&run_dir).map_err(|source| RunnerError::WriteOutput {
        path: run_dir.clone(),
        source,
    })?;
    eprintln!(
        "🚀 eval suite start run={} cases={} trials={} concurrency={} output={}",
        config.run_id,
        case_paths.len(),
        config.trials.get(),
        config.case_concurrency.get(),
        run_dir.display()
    );

    let judge_handle = build_model_handle(
        &config.judge_backend,
        &config.llm_concurrency_pool,
        None,
        None,
        None,
    )
    .map_err(|error| RunnerError::Driver {
        path: config.cases_root.clone(),
        message: error.to_string(),
    })?;
    let judge = LlmRubricJudge::with_concurrency(judge_handle.lutum, judge_handle.concurrency);

    let cases = if hooks.visualizer.is_some() {
        run_suite_cases_sequential(case_paths, config, Some(&judge), hooks).await?
    } else {
        run_suite_cases_parallel(case_paths, config, Some(&judge)).await?
    };

    let mut report = aggregate_suite(run_report, cases);
    report.timing = SuiteTiming {
        elapsed_ms: duration_millis_u64(suite_started.elapsed()),
    };
    let suite_path = run_dir.join("suite-report.json");
    write_json_file(&suite_path, &report)?;
    eprintln!("\n════════════════════════════════════════════════════════════");
    eprintln!(
        "🏁 eval suite end run={} ✅passed={} ❌failed={} 💥invalid={} mean_score={:.3} elapsed_ms={}{}",
        config.run_id,
        report.passed_cases,
        report.failed_cases,
        report.invalid_cases,
        report.mean_score,
        report.timing.elapsed_ms,
        format_suite_metrics_inline(&report.metrics)
    );
    if let Some(visualizer) = hooks.visualizer.as_mut() {
        eprintln!("eval suite finished; visualizer remains open until its window is closed");
        visualizer.drain_cached_commands_until_shutdown();
    }
    Ok(report)
}

fn suite_run_report(
    config: &RunnerConfig,
    run_dir: &Path,
    planned_case_count: usize,
    failed_from: Option<&Path>,
) -> SuiteRunReport {
    SuiteRunReport {
        run_id: config.run_id.clone(),
        cases_root: config.cases_root.display().to_string(),
        output_dir: run_dir.display().to_string(),
        case_patterns: config.case_patterns.clone(),
        runtime_config_override: config
            .runtime_config_override
            .as_ref()
            .map(|path| path.display().to_string()),
        failed_only: failed_only_requested(config),
        failed_from: failed_from.map(|path| path.display().to_string()),
        fail_fast: config.fail_fast,
        model_concurrency: config
            .model_concurrency
            .iter()
            .map(|(model, limit)| (model.clone(), limit.map(NonZeroUsize::get)))
            .collect(),
        trials: config.trials.get(),
        case_concurrency: config.case_concurrency.get(),
        planned_case_count,
        models: SuiteModelNames {
            judge: config.judge_backend.model.clone(),
            cheap: config.cheap_backend.model.clone(),
            default: config.default_backend.model.clone(),
            premium: config.premium_backend.model.clone(),
            image: config.image_backend.model.clone(),
        },
    }
}

async fn run_suite_cases_sequential(
    case_paths: Vec<PathBuf>,
    config: &RunnerConfig,
    judge: Option<&dyn RubricJudge>,
    hooks: &mut RunnerHooks,
) -> Result<Vec<CaseSummary>, RunnerError> {
    let mut cases = Vec::new();
    for path in case_paths {
        let output = run_case_detailed_sequential(&path, config, judge, hooks).await?;
        let failed = !output.summary.passed || output.summary.invalid;
        cases.push(output.summary);
        if hooks
            .visualizer
            .as_ref()
            .is_some_and(VisualizerHook::shutdown_requested)
        {
            break;
        }
        if failed && config.fail_fast {
            break;
        }
    }
    Ok(cases)
}

async fn run_suite_cases_parallel(
    case_paths: Vec<PathBuf>,
    config: &RunnerConfig,
    judge: Option<&dyn RubricJudge>,
) -> Result<Vec<CaseSummary>, RunnerError> {
    let items = plan_eval_work_items(case_paths, config)?;
    let mut pending = VecDeque::from(items);
    let concurrency = config.case_concurrency.get();
    let mut stop_launching = false;
    let mut running = FuturesUnordered::new();
    let mut outputs = Vec::new();

    loop {
        if !stop_launching {
            while running.len() < concurrency {
                let Some(item) = pending.pop_front() else {
                    break;
                };
                running.push(run_eval_work_item(item, config, judge));
            }
        }

        let Some(output) = running.next().await else {
            break;
        };
        let output = output?;
        if config.fail_fast && (!output.output.summary.passed || output.output.summary.invalid) {
            stop_launching = true;
        }
        outputs.push(output);
    }

    aggregate_parallel_case_outputs(outputs)
}

async fn run_eval_work_item(
    item: EvalWorkItem,
    config: &RunnerConfig,
    judge: Option<&dyn RubricJudge>,
) -> Result<EvalWorkOutput, RunnerError> {
    let started_at = Instant::now();
    let reporter = LiveReporter::new(&config.run_id, &item.output_dir, config.live_output)?;
    let mut hooks = RunnerHooks::none();
    if item.trial_count == 1 {
        reporter.emit(
            Some(&item.id),
            "case_started",
            serde_json::json!({
                "path": item.case_path.display().to_string(),
                "output_dir": item.output_dir.display().to_string(),
                "trials": item.trial_count,
            }),
            format!(
                "▶️  eval case start id={} path={} trials={} output={}",
                item.id,
                item.case_path.display(),
                item.trial_count,
                item.output_dir.display()
            ),
        )?;
    } else {
        reporter.emit(
            Some(&item.runtime_id),
            "trial_started",
            serde_json::json!({
                "path": item.case_path.display().to_string(),
                "output_dir": item.output_dir.display().to_string(),
                "trial": item.trial_number,
                "trials": item.trial_count,
            }),
            format!(
                "▶️  eval trial start id={} trial={}/{} output={}",
                item.id,
                item.trial_number,
                item.trial_count,
                item.output_dir.display()
            ),
        )?;
    }

    let output = run_case_trial_with_timeout(
        &item.case_path,
        config,
        judge,
        &reporter,
        &mut hooks,
        &item.case,
        &item.id,
        &item.runtime_id,
        &item.output_dir,
        item.trial_number,
    )
    .await?;
    if item.trial_count == 1 {
        emit_case_finished(&reporter, &output.summary, output.events.len())?;
    } else {
        emit_trial_finished(
            &reporter,
            &item.runtime_id,
            &output.summary,
            output.events.len(),
        )?;
    }
    Ok(EvalWorkOutput {
        item,
        output,
        started_at,
        completed_at: Instant::now(),
    })
}

fn aggregate_parallel_case_outputs(
    mut outputs: Vec<EvalWorkOutput>,
) -> Result<Vec<CaseSummary>, RunnerError> {
    outputs.sort_by_key(|output| (output.item.case_order, output.item.trial_number));
    let mut grouped: BTreeMap<usize, Vec<EvalWorkOutput>> = BTreeMap::new();
    for output in outputs {
        grouped
            .entry(output.item.case_order)
            .or_default()
            .push(output);
    }

    let mut cases = Vec::with_capacity(grouped.len());
    for (_case_order, mut outputs) in grouped {
        outputs.sort_by_key(|output| output.item.trial_number);
        let first = outputs
            .first()
            .expect("grouped parallel case output is never empty");
        if first.item.trial_count == 1 {
            cases.push(first.output.summary.clone());
            continue;
        }

        let event_count = outputs
            .iter()
            .map(|output| output.output.events.len())
            .sum::<usize>();
        let started_at = outputs
            .iter()
            .map(|output| output.started_at)
            .min()
            .expect("grouped parallel case output has a start time");
        let completed_at = outputs
            .iter()
            .map(|output| output.completed_at)
            .max()
            .expect("grouped parallel case output has a completion time");
        let trial_outputs = outputs
            .iter()
            .map(|output| output.output.clone())
            .collect::<Vec<_>>();
        let summary = aggregate_case_summary(
            &first.item.case_path,
            &first.item.case,
            &first.item.id,
            &trial_outputs,
            duration_millis_u64(completed_at.duration_since(started_at)),
        );
        write_json_file(&first.item.case_output_dir.join("report.json"), &summary)?;
        eprintln!("{}", case_finished_message(&summary, event_count));
        cases.push(summary);
    }
    Ok(cases)
}

pub async fn run_case_detailed(
    case_path: &Path,
    config: &RunnerConfig,
    judge: Option<&dyn RubricJudge>,
) -> Result<CaseRunOutput, RunnerError> {
    install_trace_subscriber_for_runner()?;
    let mut hooks = RunnerHooks::none();
    run_case_detailed_sequential(case_path, config, judge, &mut hooks).await
}

fn select_case_paths(config: &RunnerConfig) -> Result<CaseSelection, RunnerError> {
    let failed_from = resolve_failed_only_reference(config)?;
    let failed_cases = failed_from
        .as_ref()
        .map(|report_path| read_failed_only_reference(report_path))
        .transpose()?;
    if failed_cases.as_ref().is_some_and(Vec::is_empty) {
        return Ok(CaseSelection {
            case_paths: Vec::new(),
            failed_from,
        });
    }

    let mut case_paths =
        discover_case_files(&config.cases_root).map_err(|source| RunnerError::DiscoverCases {
            path: config.cases_root.clone(),
            source,
        })?;
    if let Some(failed_cases) = failed_cases.as_ref() {
        case_paths = filter_failed_only_case_paths(case_paths, failed_cases)?;
    }
    if !case_paths.is_empty() || failed_from.is_none() {
        case_paths = filter_case_paths(case_paths, &config.case_patterns)?;
    }
    Ok(CaseSelection {
        case_paths,
        failed_from,
    })
}

fn failed_only_requested(config: &RunnerConfig) -> bool {
    config.failed_only || config.failed_from.is_some()
}

fn resolve_failed_only_reference(config: &RunnerConfig) -> Result<Option<PathBuf>, RunnerError> {
    if !failed_only_requested(config) {
        return Ok(None);
    }

    let report_path = if let Some(reference) = config.failed_from.as_ref() {
        resolve_explicit_failed_only_reference(&config.output_root, reference)
    } else {
        latest_failed_only_reference(&config.output_root)?
    };
    if !report_path.is_file() {
        return Err(RunnerError::FailedOnlyReferenceNotFound { path: report_path });
    }
    Ok(Some(report_path))
}

fn resolve_explicit_failed_only_reference(output_root: &Path, reference: &Path) -> PathBuf {
    if reference.is_file() {
        reference.to_path_buf()
    } else if reference.is_dir() {
        reference.join("suite-report.json")
    } else {
        output_root.join(reference).join("suite-report.json")
    }
}

fn latest_failed_only_reference(output_root: &Path) -> Result<PathBuf, RunnerError> {
    let entries = std::fs::read_dir(output_root).map_err(|source| match source.kind() {
        io::ErrorKind::NotFound => RunnerError::FailedOnlyNoReference {
            path: output_root.to_path_buf(),
        },
        _ => RunnerError::DiscoverFailedOnlyReference {
            path: output_root.to_path_buf(),
            source,
        },
    })?;

    let mut newest: Option<(std::time::SystemTime, PathBuf)> = None;
    for entry in entries {
        let entry = entry.map_err(|source| RunnerError::DiscoverFailedOnlyReference {
            path: output_root.to_path_buf(),
            source,
        })?;
        let file_type =
            entry
                .file_type()
                .map_err(|source| RunnerError::DiscoverFailedOnlyReference {
                    path: output_root.to_path_buf(),
                    source,
                })?;
        if !file_type.is_dir() {
            continue;
        }
        let report_path = entry.path().join("suite-report.json");
        let metadata = match report_path.metadata() {
            Ok(metadata) if metadata.is_file() => metadata,
            Ok(_) => continue,
            Err(error) if error.kind() == io::ErrorKind::NotFound => continue,
            Err(source) => {
                return Err(RunnerError::DiscoverFailedOnlyReference {
                    path: output_root.to_path_buf(),
                    source,
                });
            }
        };
        let modified =
            metadata
                .modified()
                .map_err(|source| RunnerError::DiscoverFailedOnlyReference {
                    path: report_path.clone(),
                    source,
                })?;
        if newest
            .as_ref()
            .is_none_or(|(newest_modified, _)| modified > *newest_modified)
        {
            newest = Some((modified, report_path));
        }
    }

    newest
        .map(|(_, path)| path)
        .ok_or_else(|| RunnerError::FailedOnlyNoReference {
            path: output_root.to_path_buf(),
        })
}

fn read_failed_only_reference(path: &Path) -> Result<Vec<CaseIdentity>, RunnerError> {
    let bytes = std::fs::read(path).map_err(|source| RunnerError::ReadFailedOnlyReference {
        path: path.to_path_buf(),
        source,
    })?;
    let report: FailedOnlySuiteReport =
        serde_json::from_slice(&bytes).map_err(|source| RunnerError::ParseFailedOnlyReference {
            path: path.to_path_buf(),
            source,
        })?;
    Ok(report
        .cases
        .into_iter()
        .filter(|case| !case.passed || case.invalid)
        .map(|case| CaseIdentity {
            path: case.path,
            id: case.id,
        })
        .collect())
}

fn filter_failed_only_case_paths(
    case_paths: Vec<PathBuf>,
    failed_cases: &[CaseIdentity],
) -> Result<Vec<PathBuf>, RunnerError> {
    if failed_cases.is_empty() {
        return Ok(Vec::new());
    }

    let target_cases = failed_cases.iter().cloned().collect::<HashSet<_>>();
    let mut available_cases = HashSet::new();
    let mut indexed_paths = Vec::with_capacity(case_paths.len());
    for path in case_paths {
        let case = parse_case_file(&path)?;
        let identity = CaseIdentity {
            path: path.display().to_string(),
            id: case_id(&path, &case),
        };
        available_cases.insert(identity.clone());
        indexed_paths.push((path, identity));
    }

    for failed_case in failed_cases {
        if !available_cases.contains(failed_case) {
            return Err(RunnerError::FailedOnlyCaseNotFound {
                id: failed_case.id.clone(),
                path: failed_case.path.clone(),
            });
        }
    }

    Ok(indexed_paths
        .into_iter()
        .filter_map(|(path, identity)| target_cases.contains(&identity).then_some(path))
        .collect())
}

fn filter_case_paths(
    case_paths: Vec<PathBuf>,
    patterns: &[String],
) -> Result<Vec<PathBuf>, RunnerError> {
    let normalized_patterns = patterns
        .iter()
        .map(|pattern| normalize_case_pattern(pattern))
        .filter(|pattern| !pattern.is_empty())
        .collect::<Vec<_>>();
    if normalized_patterns.is_empty() {
        return Ok(case_paths);
    }

    let mut matched = Vec::new();
    for path in case_paths {
        let mut haystacks = vec![
            normalize_case_pattern(&path.display().to_string()),
            path.file_stem()
                .and_then(|stem| stem.to_str())
                .map(normalize_case_pattern)
                .unwrap_or_default(),
        ];
        if let Ok(case) = parse_case_file(&path)
            && let Some(id) = case.id()
        {
            haystacks.push(normalize_case_pattern(id));
        }

        if normalized_patterns.iter().any(|pattern| {
            haystacks
                .iter()
                .any(|haystack| haystack.contains(pattern.as_str()))
        }) {
            matched.push(path);
        }
    }

    if matched.is_empty() {
        return Err(RunnerError::NoCasesMatched {
            patterns: patterns.join(", "),
        });
    }
    Ok(matched)
}

fn plan_eval_work_items(
    case_paths: Vec<PathBuf>,
    config: &RunnerConfig,
) -> Result<Vec<EvalWorkItem>, RunnerError> {
    let trial_count = config.trials.get();
    let mut items = Vec::with_capacity(case_paths.len().saturating_mul(trial_count));
    for (case_order, case_path) in case_paths.into_iter().enumerate() {
        let mut case = parse_case_file(&case_path)?;
        apply_runtime_config_override(&mut case, config);
        let id = case_id(&case_path, &case);
        let case_output_dir = config
            .output_root
            .join(&config.run_id)
            .join(sanitize_id(&id));
        for trial_number in 1..=trial_count {
            let output_dir = if trial_count == 1 {
                case_output_dir.clone()
            } else {
                case_output_dir.join(trial_dir_name(trial_number))
            };
            let runtime_id = if trial_count == 1 {
                id.clone()
            } else {
                trial_runtime_id(&id, trial_number)
            };
            items.push(EvalWorkItem {
                case_order,
                case_path: case_path.clone(),
                case: case.clone(),
                id: id.clone(),
                case_output_dir: case_output_dir.clone(),
                output_dir,
                runtime_id,
                trial_number,
                trial_count,
            });
        }
    }
    Ok(items)
}

fn normalize_case_pattern(value: &str) -> String {
    value
        .chars()
        .flat_map(char::to_lowercase)
        .map(|ch| match ch {
            '/' | '\\' | '_' => '-',
            other => other,
        })
        .collect::<String>()
}

async fn run_case_detailed_sequential(
    case_path: &Path,
    config: &RunnerConfig,
    judge: Option<&dyn RubricJudge>,
    hooks: &mut RunnerHooks,
) -> Result<CaseRunOutput, RunnerError> {
    let mut case = parse_case_file(case_path)?;
    apply_runtime_config_override(&mut case, config);
    let id = case_id(case_path, &case);
    let output_dir = config
        .output_root
        .join(&config.run_id)
        .join(sanitize_id(&id));
    std::fs::create_dir_all(&output_dir).map_err(|source| RunnerError::WriteOutput {
        path: output_dir.clone(),
        source,
    })?;
    emit_visualizer_open_tab(hooks, &id);

    let trial_count = config.trials.get();
    let case_started = Instant::now();
    if trial_count == 1 {
        let reporter = LiveReporter::new(&config.run_id, &output_dir, config.live_output)?;
        reporter.emit(
            Some(&id),
            "case_started",
            serde_json::json!({
                "path": case_path.display().to_string(),
                "output_dir": output_dir.display().to_string(),
                "trials": trial_count,
            }),
            format!(
                "▶️  eval case start id={} path={} trials={} output={}",
                id,
                case_path.display(),
                trial_count,
                output_dir.display()
            ),
        )?;
        let output = run_case_trial_with_timeout(
            case_path,
            config,
            judge,
            &reporter,
            hooks,
            &case,
            &id,
            &id,
            &output_dir,
            1,
        )
        .await?;
        emit_case_finished(&reporter, &output.summary, output.events.len())?;
        emit_visualizer_case_status(hooks, &output.summary);
        return Ok(output);
    }

    let mut trial_outputs = Vec::with_capacity(trial_count);
    for trial_number in 1..=trial_count {
        let runtime_id = trial_runtime_id(&id, trial_number);
        let trial_output_dir = output_dir.join(trial_dir_name(trial_number));
        let reporter = LiveReporter::new(&config.run_id, &trial_output_dir, config.live_output)?;
        reporter.emit(
            Some(&runtime_id),
            "trial_started",
            serde_json::json!({
                "path": case_path.display().to_string(),
                "output_dir": trial_output_dir.display().to_string(),
                "trial": trial_number,
                "trials": trial_count,
            }),
            format!(
                "▶️  eval trial start id={} trial={}/{} output={}",
                id,
                trial_number,
                trial_count,
                trial_output_dir.display()
            ),
        )?;
        let output = run_case_trial_with_timeout(
            case_path,
            config,
            judge,
            &reporter,
            hooks,
            &case,
            &id,
            &runtime_id,
            &trial_output_dir,
            trial_number,
        )
        .await?;
        emit_trial_finished(&reporter, &runtime_id, &output.summary, output.events.len())?;
        trial_outputs.push(output);
    }

    let event_count = trial_outputs
        .iter()
        .map(|output| output.events.len())
        .sum::<usize>();
    let summary = aggregate_case_summary(
        case_path,
        &case,
        &id,
        &trial_outputs,
        duration_millis_u64(case_started.elapsed()),
    );
    write_json_file(&output_dir.join("report.json"), &summary)?;
    eprintln!("{}", case_finished_message(&summary, event_count));

    let artifact = trial_outputs
        .first()
        .map(|output| output.artifact.clone())
        .unwrap_or_default();
    let events = trial_outputs
        .iter()
        .flat_map(|output| output.events.clone())
        .collect::<Vec<_>>();
    let trace = trial_outputs
        .first()
        .map(|output| output.trace.clone())
        .unwrap_or_else(empty_trace_snapshot);
    let raw_trace = trial_outputs
        .first()
        .map(|output| output.raw_trace.clone())
        .unwrap_or_default();

    Ok(CaseRunOutput {
        case_path: case_path.to_path_buf(),
        output_dir,
        summary,
        artifact,
        events,
        trace,
        raw_trace,
    })
}

fn apply_runtime_config_override(case: &mut EvalCase, config: &RunnerConfig) {
    let Some(path) = &config.runtime_config_override else {
        return;
    };
    let EvalCase::Runtime(case) = case;
    case.runtime_config = path.display().to_string();
}

#[allow(clippy::too_many_arguments)]
async fn run_case_trial_with_timeout(
    case_path: &Path,
    config: &RunnerConfig,
    judge: Option<&dyn RubricJudge>,
    reporter: &LiveReporter,
    hooks: &mut RunnerHooks,
    case: &EvalCase,
    id: &str,
    runtime_id: &str,
    output_dir: &Path,
    trial_number: usize,
) -> Result<CaseRunOutput, RunnerError> {
    std::fs::create_dir_all(output_dir).map_err(|source| RunnerError::WriteOutput {
        path: output_dir.to_path_buf(),
        source,
    })?;

    let started = Instant::now();
    let case_timeout = Duration::from_millis(case.limits().timeout_ms);
    let output = match tokio::time::timeout(
        case_timeout,
        run_case_detailed_body(
            case_path,
            config,
            judge,
            reporter,
            hooks,
            case,
            id,
            runtime_id,
            output_dir,
            trial_number,
        ),
    )
    .await
    {
        Ok(result) => result?,
        Err(_) => {
            let message = format!("eval case timed out after {}ms", case_timeout.as_millis());
            emit_visualizer_error(
                hooks,
                runtime_id,
                "eval",
                "case-timeout",
                None,
                message.clone(),
            );
            if let Some(visualizer) = hooks.visualizer.as_ref() {
                visualizer.send_event(VisualizerEvent::SetTabStatus {
                    tab_id: VisualizerTabId::new(id.to_string()),
                    status: TabStatus::Invalid,
                });
            }
            write_runtime_failure_case_output(
                CaseOutputContext {
                    case_path,
                    output_dir,
                    case,
                    id,
                    runtime_id,
                    trial_number,
                    reporter,
                    llm_log_directory: Some(eval_llm_log_directory(config, runtime_id)),
                },
                message,
                empty_trace_snapshot(),
                RawTraceSnapshot::default(),
            )?
        }
    };
    Ok(apply_trial_timing(output, started))
}

fn apply_trial_timing(mut output: CaseRunOutput, started: Instant) -> CaseRunOutput {
    let elapsed_ms = duration_millis_u64(started.elapsed());
    output.summary.timing = CaseTiming { elapsed_ms };
    if let Some(trial) = output.summary.trials.first_mut() {
        trial.timing = CaseTiming { elapsed_ms };
    }
    output
}

#[allow(clippy::too_many_arguments)]
async fn run_case_detailed_body(
    case_path: &Path,
    config: &RunnerConfig,
    judge: Option<&dyn RubricJudge>,
    reporter: &LiveReporter,
    hooks: &mut RunnerHooks,
    case: &EvalCase,
    id: &str,
    runtime_id: &str,
    output_dir: &Path,
    trial_number: usize,
) -> Result<CaseRunOutput, RunnerError> {
    let local = LocalSet::new();
    let capture = lutum_trace::capture_raw(
        AssertUnwindSafe(execute_case(
            case, config, output_dir, runtime_id, reporter, hooks, judge,
        ))
        .catch_unwind(),
    );
    let collected = local.run_until(capture).await;

    let trace = collected.trace;
    let raw_trace = collected.raw;
    let execution = match collected.output {
        Ok(Ok(execution)) => execution,
        Ok(Err(error)) => {
            let message = format!("{error:#}");
            emit_visualizer_error(
                hooks,
                runtime_id,
                "eval",
                "execute_case",
                None,
                message.clone(),
            );
            if let Some(visualizer) = hooks.visualizer.as_ref() {
                visualizer.send_event(VisualizerEvent::SetTabStatus {
                    tab_id: VisualizerTabId::new(id.to_string()),
                    status: TabStatus::Invalid,
                });
            }
            return write_runtime_failure_case_output(
                CaseOutputContext {
                    case_path,
                    output_dir,
                    case,
                    id,
                    runtime_id,
                    trial_number,
                    reporter,
                    llm_log_directory: Some(eval_llm_log_directory(config, runtime_id)),
                },
                message,
                trace,
                raw_trace,
            );
        }
        Err(payload) => {
            let message = format!("panic: {}", panic_payload_message(payload.as_ref()));
            emit_visualizer_error(hooks, runtime_id, "eval", "panic", None, message.clone());
            if let Some(visualizer) = hooks.visualizer.as_ref() {
                visualizer.send_event(VisualizerEvent::SetTabStatus {
                    tab_id: VisualizerTabId::new(id.to_string()),
                    status: TabStatus::Invalid,
                });
            }
            return write_runtime_failure_case_output(
                CaseOutputContext {
                    case_path,
                    output_dir,
                    case,
                    id,
                    runtime_id,
                    trial_number,
                    reporter,
                    llm_log_directory: Some(eval_llm_log_directory(config, runtime_id)),
                },
                message,
                trace,
                raw_trace,
            );
        }
    };
    let artifact = execution.artifact;
    let events = execution.events;
    let activations = execution.activations;
    let report = evaluate_case_with_overrides(
        case,
        &trace,
        &artifact,
        judge,
        &execution.assertion_overrides,
    )
    .await;
    let summary = case_summary_from_report(
        case_path,
        case,
        id,
        output_dir,
        trial_number,
        report,
        activations,
    );

    write_json_file(&output_dir.join("artifact.json"), &artifact)?;
    write_json_file(&output_dir.join("report.json"), &summary)?;
    write_json_file(&output_dir.join("events.json"), &events)?;
    write_json_file(&output_dir.join("trace.json"), &trace_snapshot_json(&trace))?;
    if !summary.passed
        || summary.invalid
        || artifact.failure.is_some()
        || raw_trace_has_error(&raw_trace)
    {
        write_json_file(
            &output_dir.join("raw-trace.json"),
            &raw_trace_snapshot_json(&raw_trace),
        )?;
    }
    Ok(CaseRunOutput {
        case_path: case_path.to_path_buf(),
        output_dir: output_dir.to_path_buf(),
        summary,
        artifact,
        events,
        trace,
        raw_trace,
    })
}

fn write_runtime_failure_case_output(
    ctx: CaseOutputContext<'_>,
    message: String,
    trace: TraceSnapshot,
    raw_trace: RawTraceSnapshot,
) -> Result<CaseRunOutput, RunnerError> {
    let llm_log_directory = ctx
        .llm_log_directory
        .as_ref()
        .map(|path| path.display().to_string());
    let mut artifact = CaseArtifact::failed(message.clone());
    if let Some(directory) = &llm_log_directory {
        artifact = artifact.with_observation("llm_log_directory", directory.clone());
    }
    let events = Vec::new();
    let report = CaseReport {
        runtime_failure: Some(message.clone()),
        llm_log_directory: llm_log_directory.clone(),
        assertions: Vec::new(),
        measurements: BTreeMap::new(),
        invalid: true,
        must_pass_ok: false,
        weighted_points_earned: 0,
        weighted_points_total: 0,
        score: 0.0,
    };
    let summary = case_summary_from_report(
        ctx.case_path,
        ctx.case,
        ctx.id,
        ctx.output_dir,
        ctx.trial_number,
        report,
        Vec::new(),
    );

    write_json_file(&ctx.output_dir.join("artifact.json"), &artifact)?;
    write_json_file(&ctx.output_dir.join("report.json"), &summary)?;
    write_json_file(&ctx.output_dir.join("events.json"), &events)?;
    write_json_file(
        &ctx.output_dir.join("trace.json"),
        &trace_snapshot_json(&trace),
    )?;
    write_json_file(
        &ctx.output_dir.join("raw-trace.json"),
        &raw_trace_snapshot_json(&raw_trace),
    )?;
    ctx.reporter.emit(
        Some(ctx.runtime_id),
        "case_error",
        serde_json::json!({
            "path": summary.path.as_str(),
            "error": message,
            "llm_log_directory": llm_log_directory,
        }),
        format!("eval case error id={} error={}", summary.id, message),
    )?;

    Ok(CaseRunOutput {
        case_path: ctx.case_path.to_path_buf(),
        output_dir: ctx.output_dir.to_path_buf(),
        summary,
        artifact,
        events,
        trace,
        raw_trace,
    })
}

fn case_summary_from_report(
    case_path: &Path,
    case: &EvalCase,
    id: &str,
    output_dir: &Path,
    trial_number: usize,
    report: CaseReport,
    activations: Vec<ModuleActivationRecord>,
) -> CaseSummary {
    let description = case_description(case);
    let passed = report.passed();
    let invalid = report.invalid;
    let score = report.score;
    let timing = CaseTiming { elapsed_ms: 0 };
    let measurement_statistics = measurement_statistics(std::iter::once(&report));
    let trial = CaseTrialSummary {
        trial: trial_number,
        output_dir: output_dir.display().to_string(),
        path: case_path.display().to_string(),
        runtime_config: case.runtime().runtime_config.clone(),
        id: id.to_string(),
        description: description.clone(),
        passed,
        invalid,
        score,
        report: report.clone(),
        timing: timing.clone(),
    };

    CaseSummary {
        path: case_path.display().to_string(),
        runtime_config: case.runtime().runtime_config.clone(),
        id: id.to_string(),
        description,
        passed,
        invalid,
        score,
        report,
        timing,
        trial_timing: None,
        activations,
        measurement_statistics,
        trial_count: 1,
        passed_trials: usize::from(passed),
        failed_trials: usize::from(!passed && !invalid),
        invalid_trials: usize::from(invalid),
        trials: vec![trial],
    }
}

fn aggregate_case_summary(
    case_path: &Path,
    case: &EvalCase,
    id: &str,
    trial_outputs: &[CaseRunOutput],
    elapsed_ms: u64,
) -> CaseSummary {
    let trial_count = trial_outputs.len();
    let passed_trials = trial_outputs
        .iter()
        .filter(|output| output.summary.passed)
        .count();
    let invalid_trials = trial_outputs
        .iter()
        .filter(|output| output.summary.invalid)
        .count();
    let failed_trials = trial_count.saturating_sub(passed_trials + invalid_trials);
    let passed = trial_count > 0 && passed_trials == trial_count;
    let invalid = invalid_trials > 0;
    let score = if trial_outputs.is_empty() {
        0.0
    } else {
        trial_outputs
            .iter()
            .map(|output| output.summary.score)
            .sum::<f64>()
            / trial_outputs.len() as f64
    };
    let report = aggregate_case_report(trial_outputs, passed, invalid, score);
    let description = case_description(case);
    let trials = trial_outputs
        .iter()
        .enumerate()
        .map(|(index, output)| {
            output
                .summary
                .trials
                .first()
                .cloned()
                .unwrap_or_else(|| CaseTrialSummary {
                    trial: index + 1,
                    output_dir: output.output_dir.display().to_string(),
                    path: output.summary.path.clone(),
                    runtime_config: output.summary.runtime_config.clone(),
                    id: output.summary.id.clone(),
                    description: output.summary.description.clone(),
                    passed: output.summary.passed,
                    invalid: output.summary.invalid,
                    score: output.summary.score,
                    report: output.summary.report.clone(),
                    timing: output.summary.timing.clone(),
                })
        })
        .collect::<Vec<_>>();
    let trial_timing = (trial_count > 1).then(|| aggregate_trial_timing(&trials));

    CaseSummary {
        path: case_path.display().to_string(),
        runtime_config: case.runtime().runtime_config.clone(),
        id: id.to_string(),
        description,
        passed,
        invalid,
        score,
        report,
        timing: CaseTiming { elapsed_ms },
        trial_timing,
        activations: Vec::new(),
        measurement_statistics: measurement_statistics(
            trial_outputs.iter().map(|output| &output.summary.report),
        ),
        trial_count,
        passed_trials,
        failed_trials,
        invalid_trials,
        trials,
    }
}

fn aggregate_case_report(
    trial_outputs: &[CaseRunOutput],
    passed: bool,
    invalid: bool,
    score: f64,
) -> CaseReport {
    let runtime_failure_count = trial_outputs
        .iter()
        .filter(|output| output.summary.report.runtime_failure.is_some())
        .count();
    CaseReport {
        runtime_failure: (runtime_failure_count > 0).then(|| {
            format!(
                "{runtime_failure_count} trial(s) had runtime failures out of {}",
                trial_outputs.len()
            )
        }),
        llm_log_directory: None,
        assertions: Vec::new(),
        measurements: aggregate_measurements(trial_outputs),
        invalid,
        must_pass_ok: passed,
        weighted_points_earned: 0,
        weighted_points_total: 0,
        score,
    }
}

fn aggregate_measurements(
    trial_outputs: &[CaseRunOutput],
) -> BTreeMap<String, crate::measure::MeasurementValue> {
    use crate::measure::MeasurementValue;

    let mut scalar_values = BTreeMap::<String, Vec<f64>>::new();
    let mut scoped_values = BTreeMap::<String, BTreeMap<String, Vec<f64>>>::new();
    for output in trial_outputs {
        for (name, value) in &output.summary.report.measurements {
            match value {
                MeasurementValue::Scalar(Some(value)) if value.is_finite() => {
                    scalar_values.entry(name.clone()).or_default().push(*value);
                }
                MeasurementValue::ByScope(values) => {
                    let entry = scoped_values.entry(name.clone()).or_default();
                    for (scope, value) in values {
                        if value.is_finite() {
                            entry.entry(scope.clone()).or_default().push(*value);
                        }
                    }
                }
                MeasurementValue::Scalar(_) => {}
            }
        }
    }
    let mut aggregated = scalar_values
        .into_iter()
        .map(|(name, values)| (name, MeasurementValue::Scalar(Some(mean(&values)))))
        .collect::<BTreeMap<_, _>>();
    for (name, scopes) in scoped_values {
        aggregated.insert(
            name,
            MeasurementValue::ByScope(
                scopes
                    .into_iter()
                    .map(|(scope, values)| (scope, mean(&values)))
                    .collect(),
            ),
        );
    }
    aggregated
}

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

fn measurement_statistics<'a>(
    reports: impl IntoIterator<Item = &'a CaseReport>,
) -> BTreeMap<String, MeasurementStatistics> {
    use crate::measure::MeasurementValue;

    let mut samples = BTreeMap::<String, Vec<f64>>::new();
    for report in reports {
        for (name, value) in &report.measurements {
            match value {
                MeasurementValue::Scalar(Some(value)) if value.is_finite() => {
                    samples.entry(name.clone()).or_default().push(*value);
                }
                MeasurementValue::ByScope(scopes) => {
                    for (scope, value) in scopes {
                        if value.is_finite() {
                            samples
                                .entry(format!("{name}@{scope}"))
                                .or_default()
                                .push(*value);
                        }
                    }
                }
                MeasurementValue::Scalar(_) => {}
            }
        }
    }
    samples
        .into_iter()
        .map(|(name, mut values)| {
            values.sort_by(f64::total_cmp);
            let mean = mean(&values);
            let variance = values
                .iter()
                .map(|value| (value - mean).powi(2))
                .sum::<f64>()
                / values.len() as f64;
            let percentile = |fraction: f64| {
                let index = ((values.len() - 1) as f64 * fraction).ceil() as usize;
                values[index]
            };
            (
                name,
                MeasurementStatistics {
                    samples: values.len(),
                    min: values[0],
                    max: values[values.len() - 1],
                    mean,
                    standard_deviation: variance.sqrt(),
                    p50: percentile(0.50),
                    p95: percentile(0.95),
                },
            )
        })
        .collect()
}

fn case_description(case: &EvalCase) -> Option<String> {
    case.description()
        .map(|text| normalize_text_block(&text.content))
}

fn trial_runtime_id(id: &str, trial_number: usize) -> String {
    format!("{id}/{}", trial_dir_name(trial_number))
}

fn trial_dir_name(trial_number: usize) -> String {
    format!("trial-{trial_number:03}")
}

fn empty_trace_snapshot() -> TraceSnapshot {
    TraceSnapshot {
        roots: Vec::new(),
        root_events: Vec::new(),
    }
}

fn case_finished_message(summary: &CaseSummary, event_count: usize) -> String {
    let status_icon = if summary.invalid {
        "💥"
    } else if summary.passed {
        "✅"
    } else {
        "❌"
    };
    if let Some(runtime_failure) = &summary.report.runtime_failure {
        format!(
            "{status_icon} eval case end id={} passed={} invalid={} score={:.3} elapsed_ms={} events={} failure={}",
            summary.id,
            summary.passed,
            summary.invalid,
            summary.score,
            summary.timing.elapsed_ms,
            event_count,
            runtime_failure
        )
    } else {
        format!(
            "{status_icon} eval case end id={} passed={} invalid={} score={:.3} elapsed_ms={} events={}",
            summary.id,
            summary.passed,
            summary.invalid,
            summary.score,
            summary.timing.elapsed_ms,
            event_count
        )
    }
}

fn emit_case_finished(
    reporter: &LiveReporter,
    summary: &CaseSummary,
    event_count: usize,
) -> Result<(), RunnerError> {
    let case_finished_message = case_finished_message(summary, event_count);
    reporter.emit(
        Some(&summary.id),
        "case_finished",
        serde_json::json!({
            "path": summary.path.as_str(),
            "passed": summary.passed,
            "invalid": summary.invalid,
            "score": summary.score,
            "elapsed_ms": summary.timing.elapsed_ms,
            "runtime_failure": summary.report.runtime_failure.as_deref(),
            "events": event_count,
        }),
        case_finished_message,
    )
}

fn emit_trial_finished(
    reporter: &LiveReporter,
    runtime_id: &str,
    summary: &CaseSummary,
    event_count: usize,
) -> Result<(), RunnerError> {
    let status_icon = if summary.invalid {
        "💥"
    } else if summary.passed {
        "✅"
    } else {
        "❌"
    };
    reporter.emit(
        Some(runtime_id),
        "trial_finished",
        serde_json::json!({
            "path": summary.path.as_str(),
            "passed": summary.passed,
            "invalid": summary.invalid,
            "score": summary.score,
            "elapsed_ms": summary.timing.elapsed_ms,
            "runtime_failure": summary.report.runtime_failure.as_deref(),
            "events": event_count,
        }),
        format!(
            "{status_icon} eval trial end id={} runtime_id={} passed={} invalid={} score={:.3} elapsed_ms={} events={}",
            summary.id,
            runtime_id,
            summary.passed,
            summary.invalid,
            summary.score,
            summary.timing.elapsed_ms,
            event_count
        ),
    )
}

fn format_suite_metrics_inline(metrics: &SuiteMetrics) -> String {
    let pass_at = metrics
        .pass_at
        .iter()
        .map(|metric| format!("pass@{}={:.3}", metric.k, metric.value));
    let pass_hat = metrics
        .pass_hat
        .iter()
        .map(|metric| format!("pass^{}={:.3}", metric.k, metric.value));
    let values = pass_at.chain(pass_hat).collect::<Vec<_>>();
    if values.is_empty() {
        String::new()
    } else {
        format!(" {}", values.join(" "))
    }
}

fn emit_visualizer_case_status(hooks: &RunnerHooks, summary: &CaseSummary) {
    let Some(visualizer) = hooks.visualizer.as_ref() else {
        return;
    };
    let status = if summary.invalid {
        TabStatus::Invalid
    } else if summary.passed {
        TabStatus::Passed
    } else {
        TabStatus::Failed
    };
    visualizer.send_event(VisualizerEvent::SetTabStatus {
        tab_id: VisualizerTabId::new(summary.id.clone()),
        status,
    });
}

fn emit_visualizer_open_tab(hooks: &RunnerHooks, id: &str) {
    let Some(visualizer) = hooks.visualizer.as_ref() else {
        return;
    };
    visualizer.send_event(VisualizerEvent::OpenTab {
        tab_id: VisualizerTabId::new(id.to_string()),
        title: id.to_string(),
    });
}

fn emit_visualizer_error(
    hooks: &RunnerHooks,
    id: &str,
    source: impl Into<String>,
    phase: impl Into<String>,
    owner: Option<String>,
    message: String,
) {
    let Some(visualizer) = hooks.visualizer.as_ref() else {
        return;
    };
    visualizer.send_event(VisualizerEvent::Error {
        tab_id: VisualizerTabId::new(id.to_string()),
        error: VisualizerErrorView {
            at: Utc::now(),
            source: source.into(),
            phase: phase.into(),
            owner,
            message,
        },
    });
}

async fn execute_case(
    case: &EvalCase,
    config: &RunnerConfig,
    output_dir: &Path,
    case_id: &str,
    reporter: &LiveReporter,
    hooks: &mut RunnerHooks,
    judge: Option<&dyn RubricJudge>,
) -> Result<CaseExecution> {
    let EvalCase::Runtime(case) = case;
    execute_runtime_case(case, config, output_dir, case_id, reporter, hooks, judge).await
}

async fn execute_runtime_case(
    case: &RuntimeCase,
    config: &RunnerConfig,
    output_dir: &Path,
    case_id: &str,
    reporter: &LiveReporter,
    hooks: &mut RunnerHooks,
    judge: Option<&dyn RubricJudge>,
) -> Result<CaseExecution> {
    let runtime_config_path = PathBuf::from(&case.runtime_config);
    let runtime_config_content = std::fs::read_to_string(&runtime_config_path)
        .with_context(|| format!("read runtime config {}", runtime_config_path.display()))?;
    let boot_config = nuillu_server::parse_server_boot_config_content(
        &runtime_config_content,
        &runtime_config_path,
    )?;
    let case_modules = boot_config
        .active_modules()
        .into_iter()
        .chain(
            boot_config
                .expanded_subsystems()
                .into_iter()
                .flat_map(|expanded| {
                    expanded
                        .definition
                        .modules
                        .iter()
                        .map(nuillu_server::ServerModuleSpec::module_id)
                        .collect::<Vec<_>>()
                }),
        )
        .collect::<Vec<_>>();
    for activation in &case.activate_allocation {
        anyhow::ensure!(
            case_modules.contains(&activation.module.module_id()),
            "activate-allocation module {:?} is absent from runtime config {}",
            activation.module.as_str(),
            runtime_config_path.display()
        );
    }
    let gui_deferred_start = hooks.visualizer.is_some();
    let case_now = parse_case_now(case.now.as_deref())
        .map_err(anyhow::Error::msg)
        .context("parse runtime case now")?;
    let mut allocation = if gui_deferred_start {
        let mut allocation = nuillu_server::server_initial_allocation(&boot_config);
        for module in &case_modules {
            allocation.set_activation(module.clone(), ActivationRatio::ZERO);
        }
        allocation
    } else {
        nuillu_server::server_initial_allocation(&boot_config)
    };
    if !gui_deferred_start && !case.activate_allocation.is_empty() {
        apply_case_activation_allocation(&mut allocation, &case.activate_allocation);
    }
    let env = build_eval_environment(
        output_dir,
        config,
        allocation,
        &case.limits,
        case_modules.clone(),
        case_now,
        &case.memories,
        &case.memory_links,
        &case.policies,
        case_id,
        reporter,
        hooks.visualizer.as_ref().map(VisualizerHook::event_sender),
    )
    .await?;
    let timeline_runtime_origin_ms = env.events.elapsed_ms();
    let timeline_started_at = env.clock.now();
    seed_eval_scene_participants(env.caps.scene(), &case.participants);
    let memory_baseline = memory_snapshot(env.memory.as_ref()).await?;
    let memo_seed_records =
        seed_memos(&env.blackboard, env.clock.as_ref(), &case.memos, false).await?;
    let cognition_seed_records =
        seed_cognition_log(&env.blackboard, env.clock.as_ref(), &case.cognition_log).await;
    if let Some(visualizer) = hooks.visualizer.as_mut() {
        emit_visualizer_blackboard_snapshot(case_id, &env.blackboard, Some(visualizer)).await;
        emit_visualizer_memory_records(
            case_id,
            visualizer,
            &env.blackboard,
            env.memory.as_ref(),
            0,
            25,
        )
        .await;
        visualizer.offer_action(VisualizerAction::start_activation(VisualizerTabId::new(
            case_id.to_string(),
        )));
    }

    let host = env.caps.host_io();
    let sensory = host.sensory_input_mailbox();
    let inputs = case.inputs.clone();
    let steps = case.steps.clone();
    let eval_case = EvalCase::Runtime(case.clone());
    let step_driven_case = !case.steps.is_empty();
    let activate_allocation = case.activate_allocation.clone();
    let actions = env.actions.clone();
    let events = env.events.clone();
    let clock = env.clock.clone();
    let memory = env.memory.clone();
    let cognition_log_repository = env.cognition_log_repository.clone();
    let utterances = env.utterances.clone();
    let allocation_blackboard = env.blackboard.clone();
    let allow_empty_output = case.allow_empty_output;
    let mut allocation_reporter =
        AllocationChangeReporter::new(case_id.to_string(), reporter.clone());
    let live_reporter = reporter.clone();
    let case_id_for_idle = case_id.to_string();
    let modules = nuillu_server::server_registry_with_factories(
        &runtime_config_path,
        &boot_config,
        &config.module_factories,
        &env.memory_caps,
        &env.policy_caps,
        &env.utterance_sink,
    )?
    .build(&env.caps)
    .await?;
    let mut visualizer = hooks.visualizer.as_mut();
    let step_failure: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));
    let step_outcomes: Arc<Mutex<Vec<serde_json::Value>>> = Arc::new(Mutex::new(Vec::new()));
    let assertion_overrides: Arc<Mutex<BTreeMap<String, AssertionOutcome>>> =
        Arc::new(Mutex::new(BTreeMap::new()));
    let terminal_output: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));
    let step_failure_for_loop = step_failure.clone();
    let step_outcomes_for_loop = step_outcomes.clone();
    let assertion_overrides_for_loop = assertion_overrides.clone();
    let terminal_output_for_loop = terminal_output.clone();
    let live_reporter_for_loop = live_reporter.clone();
    let harness = env.caps.internal_harness_io();
    let setup_memos_for_loop = memo_seed_records.clone();
    let setup_cognition_for_loop = cognition_seed_records.clone();

    let (run_controller, run_control) = AgentRunController::new();
    run_agent(
        modules,
        AgentEventLoopConfig {
            idle_threshold: Duration::from_secs(1),
            max_activation_attempts: 3,
            dependency_idle_timeout: Duration::from_secs(2),
            dependency_hard_timeout: Duration::from_secs(10),
        },
        run_control,
        async move {
            if !gui_deferred_start {
                publish_setup_updates(
                    &harness,
                    &setup_memos_for_loop,
                    &setup_cognition_for_loop,
                )
                .await;
                let _ = allocation_reporter
                    .emit_if_changed(&allocation_blackboard)
                    .await;
            }
            emit_visualizer_blackboard_snapshot(
                &case_id_for_idle,
                &allocation_blackboard,
                visualizer.as_deref(),
            )
            .await;
            let mut started = !gui_deferred_start;
            let mut input_phase_finished = false;
            if started {
                run_input_phase(
                    &case_id_for_idle,
                    &inputs,
                    &steps,
                    &sensory,
                    &harness,
                    &allocation_blackboard,
                    utterances.as_ref(),
                    events.as_ref(),
                    clock.as_ref(),
                    visualizer.as_deref(),
                    &live_reporter_for_loop,
                    &run_controller,
                    &eval_case,
                    judge,
                    memory.as_ref(),
                    timeline_runtime_origin_ms,
                    timeline_started_at,
                    &assertion_overrides_for_loop,
                    &terminal_output_for_loop,
                    &step_failure_for_loop,
                    &step_outcomes_for_loop,
                )
                .await;
                input_phase_finished = true;
            }

            let initial_progress_count = events.progress_event_count();
            let mut settle = RuntimeSettleTracker::new(initial_progress_count, Instant::now());
            let mut last_progress_count = initial_progress_count;
            let mut idle_ticks = 0_u64;
            let poll_ms = duration_millis_u64(EVAL_POLL_INTERVAL);
            let idle_report_every_ticks = ticks_for_interval(IDLE_REPORT_INTERVAL, poll_ms);
            let mut tick: u64 = 0;
            loop {
                let now = Instant::now();
                settle.observe_progress_count(events.progress_event_count(), now);
                if events.stop_requested()
                    || runtime_ready_to_score_at(
                        &actions,
                        &settle,
                        events.llm_in_flight_count(),
                        input_phase_finished,
                        allow_empty_output,
                        step_driven_case,
                        now,
                    )
                {
                    break;
                }
                tokio::task::yield_now().await;
                tokio::time::sleep(EVAL_POLL_INTERVAL).await;
                tick = tick.saturating_add(1);
                if let Some(visualizer) = visualizer.as_deref_mut() {
                    let command_outcome = handle_visualizer_commands(
                        &case_id_for_idle,
                        visualizer,
                        Some(&sensory),
                        &allocation_blackboard,
                        memory.as_ref(),
                        cognition_log_repository.as_ref(),
                        clock.as_ref(),
                    )
                    .await;
                    if command_outcome.shutdown {
                        break;
                    }
                    if command_outcome.start_requested && !started {
                        visualizer.revoke_action(start_activation_action_id(
                            &VisualizerTabId::new(case_id_for_idle.clone()),
                        ));
                        activate_gui_start_modules(&allocation_blackboard, &activate_allocation)
                            .await;
                        let _ = allocation_reporter
                            .emit_if_changed(&allocation_blackboard)
                            .await;
                        run_input_phase(
                            &case_id_for_idle,
                            &inputs,
                            &steps,
                            &sensory,
                            &harness,
                            &allocation_blackboard,
                            utterances.as_ref(),
                            events.as_ref(),
                            clock.as_ref(),
                            Some(visualizer),
                            &live_reporter_for_loop,
                            &run_controller,
                            &eval_case,
                            judge,
                            memory.as_ref(),
                            timeline_runtime_origin_ms,
                            timeline_started_at,
                            &assertion_overrides_for_loop,
                            &terminal_output_for_loop,
                            &step_failure_for_loop,
                            &step_outcomes_for_loop,
                        )
                        .await;
                        started = true;
                        input_phase_finished = true;
                    }
                }
                if started {
                    let _ = allocation_reporter
                        .emit_if_changed(&allocation_blackboard)
                        .await;
                }
                emit_visualizer_blackboard_snapshot(
                    &case_id_for_idle,
                    &allocation_blackboard,
                    visualizer.as_deref(),
                )
                .await;
                if !started {
                    continue;
                }
                let progress_count = events.progress_event_count();
                let llm_in_flight = events.llm_in_flight_count();
                let scheduled_wait_remaining = events.scheduled_wait_remaining();
                if progress_count != last_progress_count {
                    last_progress_count = progress_count;
                    idle_ticks = 0;
                } else if llm_in_flight > 0 || scheduled_wait_remaining.is_some() {
                    idle_ticks = 0;
                } else {
                    idle_ticks = idle_ticks.saturating_add(1);
                }
                let idle_for_ms = idle_ticks.saturating_mul(poll_ms);
                if idle_ticks > 0 && idle_ticks.is_multiple_of(idle_report_every_ticks) {
                    let event_count = events.event_count();
                    let active_modules =
                        allocation_blackboard.read(active_module_observations).await;
                    let active_summary = active_modules_live_summary(&active_modules);
                    live_reporter
                        .emit_port(
                            Some(&case_id_for_idle),
                            "idle",
                            serde_json::json!({
                                "tick": tick,
                                "events": event_count,
                                "progress_events": progress_count,
                                "llm_in_flight": llm_in_flight,
                                "idle_ticks": idle_ticks,
                                "idle_for_ms": idle_for_ms,
                                "tick_ms": poll_ms,
                                "report_interval_ms": duration_millis_u64(IDLE_REPORT_INTERVAL),
                                "active_modules": active_modules,
                            }),
                            format!(
                                "💤 eval idle case={} idle_for_ms={} progress_events={} llm_in_flight={} events={} active=[{}]",
                                case_id_for_idle, idle_for_ms, progress_count, llm_in_flight, event_count, active_summary
                            ),
                        )
                        .expect("runtime eval failed to write idle event");
                }
                if idle_for_ms >= duration_millis_u64(RUNTIME_IDLE_TIMEOUT) {
                    let seconds = idle_for_ms / 1000;
                    let event_snapshot = events.snapshot();
                    let active_modules =
                        allocation_blackboard.read(active_module_observations).await;
                    let message =
                        idle_timeout_message(seconds, &event_snapshot, &active_modules);
                    step_failure_for_loop
                        .lock()
                        .expect("step failure mutex poisoned")
                        .get_or_insert(message);
                    events.request_stop("idle-timeout");
                    break;
                }
            }
        },
    )
    .await?;

    let step_failure_message = step_failure
        .lock()
        .expect("step failure mutex poisoned")
        .take();
    let recorded_step_outcomes = step_outcomes
        .lock()
        .expect("step outcomes mutex poisoned")
        .clone();
    let recorded_assertion_overrides = assertion_overrides
        .lock()
        .expect("assertion override mutex poisoned")
        .clone();
    let selected_terminal_output = terminal_output
        .lock()
        .expect("terminal output mutex poisoned")
        .clone();
    let steps_ok = step_driven_case && step_outcomes_all_ok(&recorded_step_outcomes);
    let output = configured_runtime_output(
        &env.blackboard,
        &env.utterances,
        selected_terminal_output.as_deref(),
        case.steps.last().filter(|step| step.terminal),
        &memory_baseline,
        env.memory.as_ref(),
        &memo_seed_records,
        &cognition_seed_records,
    )
    .await;
    let mut artifact = if let Some(failure) = step_failure_message {
        let mut artifact = CaseArtifact::failed(failure);
        artifact.output = output;
        artifact
    } else if !output.is_empty() || case.allow_empty_output || steps_ok {
        CaseArtifact::new(output)
    } else if env.events.stop_requested() {
        CaseArtifact::failed("stopped after max-llm-calls")
    } else {
        CaseArtifact::failed("no utterance produced")
    };
    add_observations(&mut artifact, &env.blackboard, &env.utterances).await;
    add_memory_diff_observation(&mut artifact, &memory_baseline, env.memory.as_ref()).await?;
    if !recorded_step_outcomes.is_empty() {
        artifact.observations.insert(
            "steps".to_string(),
            serde_json::Value::Array(recorded_step_outcomes),
        );
    }
    let events = env.events.snapshot();
    let timeline = build_eval_timeline(
        &env.blackboard,
        &env.utterances,
        &env.events,
        timeline_runtime_origin_ms,
        timeline_started_at,
    )
    .await;
    let measurements = crate::measure::evaluate_declared(&timeline, &case.measurements);
    artifact.observations.insert(
        "timeline".to_string(),
        serde_json::to_value(&timeline).context("serialize eval timeline")?,
    );
    artifact.observations.insert(
        "measurements".to_string(),
        serde_json::to_value(&measurements).context("serialize eval measurements")?,
    );
    write_timeline_jsonl(output_dir, &timeline)?;
    let last_state = build_runtime_last_state_dump(
        case_id,
        &artifact,
        &env.blackboard,
        env.memory.as_ref(),
        &env.utterances,
        events.len(),
    )
    .await?;
    add_last_state_observation(&mut artifact, &last_state)?;
    write_runtime_last_state_eure(output_dir, last_state)?;
    if let Some(visualizer) = hooks.visualizer.as_mut() {
        emit_visualizer_blackboard_snapshot(case_id, &env.blackboard, Some(visualizer)).await;
        emit_visualizer_memory_records(
            case_id,
            visualizer,
            &env.blackboard,
            env.memory.as_ref(),
            0,
            25,
        )
        .await;
    }
    Ok(CaseExecution {
        artifact,
        events,
        activations: env.events.activation_timeline(),
        assertion_overrides: recorded_assertion_overrides,
    })
}

async fn build_eval_timeline(
    blackboard: &Blackboard,
    utterances: &RecordingUtteranceSink,
    events: &RecordingRuntimeEventSink,
    runtime_origin_ms: u64,
    started_at: DateTime<Utc>,
) -> Vec<crate::timeline::EvalEvent> {
    use crate::timeline::{EvalEvent, EvalEventPayload};

    let mut timeline = crate::timeline::project_runtime_timeline(&events.timed_snapshot());
    for event in &mut timeline {
        event.offset_ms = event.offset_ms.saturating_sub(runtime_origin_ms);
    }
    timeline.extend(events.eval_event_snapshot().into_iter().map(|mut event| {
        event.offset_ms = event.offset_ms.saturating_sub(runtime_origin_ms);
        event
    }));

    let (memos, cognition_logs) = blackboard
        .read(|bb| {
            (
                bb.recent_memo_logs(),
                bb.cognition_log_set().logs().to_vec(),
            )
        })
        .await;
    timeline.extend(memos.into_iter().map(|record| EvalEvent {
        sequence: 0,
        offset_ms: datetime_offset_ms(started_at, record.written_at),
        scope: record.owner.scope,
        module: record.owner.module,
        replica: record.owner.replica.get(),
        step: None,
        payload: EvalEventPayload::MemoWritten {
            cognitive: record.cognitive,
            content: record.content,
        },
    }));
    for log in cognition_logs {
        for entry in log.entries {
            timeline.push(EvalEvent {
                sequence: 0,
                offset_ms: datetime_offset_ms(started_at, entry.at),
                scope: log.source.scope.clone(),
                module: log.source.module.clone(),
                replica: log.source.replica.get(),
                step: None,
                payload: EvalEventPayload::CognitionAppended {
                    content: entry.text,
                    origin: entry.origin.owner.to_string(),
                },
            });
        }
    }
    for utterance in utterances.snapshot() {
        let emitted_at = DateTime::parse_from_rfc3339(&utterance.emitted_at)
            .map(|value| value.with_timezone(&Utc))
            .unwrap_or(started_at);
        timeline.push(EvalEvent {
            sequence: 0,
            offset_ms: datetime_offset_ms(started_at, emitted_at),
            scope: parse_scope_id(&utterance.scope).unwrap_or_default(),
            module: ModuleId::new(utterance.module).expect("utterance sender module is valid"),
            replica: utterance.replica,
            step: None,
            payload: EvalEventPayload::UtteranceCompleted {
                target: utterance.target,
                content: utterance.text,
            },
        });
    }

    timeline.sort_by(|left, right| {
        left.offset_ms
            .cmp(&right.offset_ms)
            .then_with(|| {
                timeline_event_priority(&left.payload).cmp(&timeline_event_priority(&right.payload))
            })
            .then_with(|| left.sequence.cmp(&right.sequence))
    });
    let mut current_step = None;
    for (index, event) in timeline.iter_mut().enumerate() {
        if let EvalEventPayload::StimulusPublished { step_id, .. } = &event.payload {
            current_step = Some(step_id.clone());
        }
        event.sequence = index as u64 + 1;
        event.step = current_step.clone();
    }
    timeline
}

fn datetime_offset_ms(started_at: DateTime<Utc>, at: DateTime<Utc>) -> u64 {
    (at - started_at).num_milliseconds().max(0) as u64
}

fn timeline_event_priority(payload: &crate::timeline::EvalEventPayload) -> u8 {
    match payload {
        crate::timeline::EvalEventPayload::StimulusPublished { .. } => 0,
        crate::timeline::EvalEventPayload::CognitionAppended { .. }
        | crate::timeline::EvalEventPayload::MemoWritten { .. }
        | crate::timeline::EvalEventPayload::UtteranceCompleted { .. } => 2,
        _ => 1,
    }
}

fn write_timeline_jsonl(output_dir: &Path, timeline: &[crate::timeline::EvalEvent]) -> Result<()> {
    let path = output_dir.join("timeline.jsonl");
    let mut file = File::create(&path)
        .with_context(|| format!("create normalized timeline {}", path.display()))?;
    for event in timeline {
        serde_json::to_writer(&mut file, event)
            .with_context(|| format!("serialize normalized timeline {}", path.display()))?;
        file.write_all(b"\n")
            .with_context(|| format!("write normalized timeline {}", path.display()))?;
    }
    Ok(())
}

async fn configured_runtime_output(
    blackboard: &Blackboard,
    utterances: &RecordingUtteranceSink,
    selected_terminal_output: Option<&str>,
    terminal_step: Option<&EvalStep>,
    memory_baseline: &BTreeMap<String, MemoryRecord>,
    memory: &dyn MemoryStore,
    seeded_memos: &[MemoLogRecord],
    seeded_cognition: &[CognitionLogEntryRecord],
) -> String {
    if let Some(output) = selected_terminal_output {
        return output.to_string();
    }
    if let Some(WaitFor::UtteranceFrom {
        scope,
        module,
        target,
        ..
    }) = terminal_step.and_then(|step| step.wait_for.as_ref())
        && let Some(utterance) = utterances.last_matching(scope.as_deref(), module.as_str(), target)
    {
        return utterance.text;
    }
    if let Some(utterance) = utterances.last_complete() {
        return utterance.text;
    }
    let seeded_memo_keys = seeded_memos
        .iter()
        .map(|record| (record.owner.clone(), record.index))
        .collect::<HashSet<_>>();
    let seeded_cognition_keys = seeded_cognition
        .iter()
        .map(|record| (record.source.clone(), record.index))
        .collect::<HashSet<_>>();
    let (memo, cognition) = blackboard
        .read(|bb| {
            let memo = bb
                .recent_memo_logs()
                .into_iter()
                .rev()
                .find(|record| !seeded_memo_keys.contains(&(record.owner.clone(), record.index)))
                .map(|record| record.content);
            let cognition = bb
                .unread_cognition_log_entries(None)
                .into_iter()
                .rev()
                .find(|record| {
                    !seeded_cognition_keys.contains(&(record.source.clone(), record.index))
                })
                .map(|record| record.entry.text);
            (memo, cognition)
        })
        .await;
    if let Some(output) = memo.or(cognition) {
        return output;
    }
    render_memory_store_artifact(memory_baseline, memory).await
}

fn step_outcomes_all_ok(outcomes: &[serde_json::Value]) -> bool {
    !outcomes.is_empty()
        && outcomes
            .iter()
            .all(|outcome| outcome.get("status").and_then(serde_json::Value::as_str) == Some("ok"))
}

async fn memory_snapshot(memory: &dyn MemoryStore) -> Result<BTreeMap<String, MemoryRecord>> {
    let mut records = BTreeMap::new();
    for rank in [
        MemoryRank::Identity,
        MemoryRank::Permanent,
        MemoryRank::LongTerm,
        MemoryRank::MidTerm,
        MemoryRank::ShortTerm,
    ] {
        for record in memory
            .list_by_rank(rank)
            .await
            .with_context(|| format!("list {rank:?} memories for module artifact"))?
        {
            records.insert(record.index.as_str().to_owned(), record);
        }
    }
    Ok(records)
}

async fn memory_diff_records(
    baseline: &BTreeMap<String, MemoryRecord>,
    memory: &dyn MemoryStore,
) -> Vec<MemoryRecord> {
    let Ok(current) = memory_snapshot(memory).await else {
        return Vec::new();
    };
    current
        .into_iter()
        .filter_map(|(index, record)| match baseline.get(&index) {
            Some(previous) if !memory_record_materially_changed(previous, &record) => None,
            _ => Some(record),
        })
        .collect()
}

async fn memory_deleted_indexes(
    baseline: &BTreeMap<String, MemoryRecord>,
    memory: &dyn MemoryStore,
) -> Vec<String> {
    let Ok(current) = memory_snapshot(memory).await else {
        return Vec::new();
    };
    baseline
        .keys()
        .filter(|index| !current.contains_key(*index))
        .cloned()
        .collect()
}

fn memory_record_materially_changed(previous: &MemoryRecord, current: &MemoryRecord) -> bool {
    previous.content.as_str() != current.content.as_str()
        || previous.rank != current.rank
        || previous.occurred_at != current.occurred_at
        || previous.kind != current.kind
        || previous.concepts != current.concepts
        || previous.tags != current.tags
        || previous.affect_arousal != current.affect_arousal
        || previous.valence != current.valence
        || previous.emotion != current.emotion
}

async fn render_memory_store_artifact(
    baseline: &BTreeMap<String, MemoryRecord>,
    memory: &dyn MemoryStore,
) -> String {
    let records = memory_diff_records(baseline, memory).await;
    let deleted = memory_deleted_indexes(baseline, memory).await;
    if records.is_empty() && deleted.is_empty() {
        return String::new();
    }

    let indexes = records
        .iter()
        .map(|record| record.index.clone())
        .collect::<Vec<_>>();
    let links = memory
        .linked(&LinkedMemoryQuery {
            memory_indexes: indexes,
            relation_filter: Vec::new(),
            direction: MemoryLinkDirection::Both,
            offset: 0,
            limit: 128,
        })
        .await
        .unwrap_or_default();

    let mut out = String::from("Memory store changes:");
    for record in records {
        out.push_str("\n\n");
        out.push_str(&render_memory_record_artifact(&record));
    }

    if !links.is_empty() {
        out.push_str("\n\nMemory links:");
        for linked in links {
            out.push_str(&format!(
                "\n- {} -> {} relation={} confidence={:.2} strength={:.2}",
                linked.link.from_memory,
                linked.link.to_memory,
                memory_link_relation_label(linked.link.relation),
                linked.link.confidence,
                linked.link.strength,
            ));
            if let Some(label) = linked.link.freeform_relation.as_deref() {
                out.push_str(&format!(" freeform={label}"));
            }
        }
    }

    if !deleted.is_empty() {
        out.push_str("\n\nDeleted memories:");
        for index in deleted {
            out.push_str(&format!("\n- {index}"));
        }
    }

    out
}

async fn add_memory_diff_observation(
    artifact: &mut CaseArtifact,
    baseline: &BTreeMap<String, MemoryRecord>,
    memory: &dyn MemoryStore,
) -> Result<()> {
    let diff = memory_diff_observation(baseline, memory).await;
    let value = serde_json::to_value(diff).context("serialize memory diff observation")?;
    artifact
        .observations
        .insert("memory_diff".to_owned(), value);
    Ok(())
}

async fn memory_diff_observation(
    baseline: &BTreeMap<String, MemoryRecord>,
    memory: &dyn MemoryStore,
) -> MemoryDiffObservation {
    let records = memory_diff_records(baseline, memory).await;
    let deleted = memory_deleted_indexes(baseline, memory).await;
    let indexes = records
        .iter()
        .map(|record| record.index.clone())
        .collect::<Vec<_>>();
    let links = if indexes.is_empty() {
        Vec::new()
    } else {
        memory
            .linked(&LinkedMemoryQuery {
                memory_indexes: indexes,
                relation_filter: Vec::new(),
                direction: MemoryLinkDirection::Both,
                offset: 0,
                limit: 128,
            })
            .await
            .unwrap_or_default()
    };

    MemoryDiffObservation {
        entries: records
            .into_iter()
            .map(|record| MemoryDiffEntryObservation {
                index: record.index.to_string(),
                kind: format!("{:?}", record.kind),
                rank: format!("{:?}", record.rank),
                content: record.content.as_str().to_owned(),
            })
            .collect(),
        links: links
            .into_iter()
            .map(|linked| MemoryDiffLinkObservation {
                from: linked.link.from_memory.to_string(),
                to: linked.link.to_memory.to_string(),
                relation: memory_link_relation_label(linked.link.relation).to_owned(),
            })
            .collect(),
        deleted,
    }
}

#[derive(Debug, Clone, Serialize)]
struct MemoryDiffObservation {
    entries: Vec<MemoryDiffEntryObservation>,
    links: Vec<MemoryDiffLinkObservation>,
    deleted: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
struct MemoryDiffEntryObservation {
    index: String,
    kind: String,
    rank: String,
    content: String,
}

#[derive(Debug, Clone, Serialize)]
struct MemoryDiffLinkObservation {
    from: String,
    to: String,
    relation: String,
}

fn render_memory_record_artifact(record: &MemoryRecord) -> String {
    let concepts = if record.concepts.is_empty() {
        "none".to_owned()
    } else {
        let mut labels = record
            .concepts
            .iter()
            .map(|concept| match concept.loose_type.as_deref() {
                Some(loose_type) => format!("{}:{loose_type}", concept.label),
                None => concept.label.clone(),
            })
            .collect::<Vec<_>>();
        labels.sort();
        labels.join(", ")
    };
    let tags = if record.tags.is_empty() {
        "none".to_owned()
    } else {
        let mut labels = record
            .tags
            .iter()
            .map(|tag| format!("{}:{}", tag.namespace, tag.label))
            .collect::<Vec<_>>();
        labels.sort();
        labels.join(", ")
    };

    format!(
        "Memory {}\nkind: {:?}\nrank: {:?}\naffect_arousal: {:.2}\nvalence: {:.2}\nemotion: {}\ncontent: {}\nconcepts: {}\ntags: {}",
        record.index,
        record.kind,
        record.rank,
        record.affect_arousal,
        record.valence,
        if record.emotion.trim().is_empty() {
            "unknown"
        } else {
            record.emotion.trim()
        },
        record.content.as_str(),
        concepts,
        tags
    )
}

fn memory_link_relation_label(relation: MemoryLinkRelation) -> &'static str {
    match relation {
        MemoryLinkRelation::Related => "related",
        MemoryLinkRelation::Supports => "supports",
        MemoryLinkRelation::Contradicts => "contradicts",
        MemoryLinkRelation::Updates => "updates",
        MemoryLinkRelation::Corrects => "corrects",
        MemoryLinkRelation::DerivedFrom => "derived_from",
    }
}

pub(crate) async fn emit_visualizer_blackboard_snapshot(
    case_id: &str,
    blackboard: &Blackboard,
    visualizer: Option<&VisualizerHook>,
) {
    let Some(visualizer) = visualizer else {
        return;
    };
    let snapshot = blackboard.read(visualizer_blackboard_snapshot).await;
    visualizer.send_event(VisualizerEvent::BlackboardSnapshot {
        tab_id: VisualizerTabId::new(case_id.to_string()),
        snapshot,
    });
}

async fn activate_gui_start_modules(
    blackboard: &Blackboard,
    activate_allocation: &[ActivateAllocation],
) {
    let mut allocation = blackboard.read(|bb| bb.allocation().clone()).await;
    if activate_allocation.is_empty() {
        apply_gui_activation(&mut allocation, builtin::allocation(), ActivationRatio::ONE);
        apply_gui_activation(&mut allocation, builtin::sensory(), ActivationRatio::ONE);
    } else {
        for activation in activate_allocation {
            apply_gui_activation(
                &mut allocation,
                activation.module.module_id(),
                ActivationRatio::from_f64(activation.activation_ratio),
            );
        }
    }

    blackboard
        .apply(BlackboardCommand::SetAllocation(allocation))
        .await;
}

fn apply_case_activation_allocation(
    allocation: &mut ResourceAllocation,
    activate_allocation: &[ActivateAllocation],
) {
    for activation in activate_allocation {
        apply_gui_activation(
            allocation,
            activation.module.module_id(),
            ActivationRatio::from_f64(activation.activation_ratio),
        );
    }
}

fn apply_gui_activation(
    allocation: &mut ResourceAllocation,
    module: ModuleId,
    activation: ActivationRatio,
) {
    allocation.set_activation(module, activation);
}

fn seed_eval_scene_participants(scene: &SceneRegistry, participants: &[String]) {
    scene.set(participants.iter().map(Participant::new));
    scene.set_broadcast_target_enabled(participants.len() != 1);
}

async fn publish_stimuli(
    case_id: &str,
    inputs: &[Stimulus],
    sensory: &SensoryInputMailbox,
    clock: &dyn Clock,
    events: &RecordingRuntimeEventSink,
    step_id: &str,
    visualizer: Option<&VisualizerHook>,
) {
    let now = clock.now();
    for input in inputs {
        let body = match input {
            Stimulus::Heard { direction, content } => SensoryInput::OneShot {
                modality: SensoryModality::Audition,
                direction: direction.clone(),
                content: content.content.clone(),
                observed_at: now,
            },
            Stimulus::Seen {
                direction,
                appearance,
            } => SensoryInput::OneShot {
                modality: SensoryModality::Vision,
                direction: direction.clone(),
                content: appearance.content.clone(),
                observed_at: now,
            },
            Stimulus::OneShot {
                modality,
                direction,
                content,
            } => SensoryInput::OneShot {
                modality: SensoryModality::parse(modality),
                direction: direction.clone(),
                content: content.content.clone(),
                observed_at: now,
            },
            Stimulus::AmbientSnapshot { entries } => SensoryInput::AmbientSnapshot {
                entries: entries
                    .iter()
                    .map(|entry| AmbientSensoryEntry {
                        id: entry.id.clone(),
                        modality: SensoryModality::parse(&entry.modality),
                        content: entry.content.content.clone(),
                    })
                    .collect(),
                observed_at: now,
            },
        };
        sensory
            .publish(body.clone())
            .await
            .expect("runtime eval failed to publish SensoryInput");
        let (modality, direction, content) = match input {
            Stimulus::Heard { direction, content } => (
                "audition".to_string(),
                direction.clone(),
                content.content.clone(),
            ),
            Stimulus::Seen {
                direction,
                appearance,
            } => (
                "vision".to_string(),
                direction.clone(),
                appearance.content.clone(),
            ),
            Stimulus::OneShot {
                modality,
                direction,
                content,
            } => (modality.clone(), direction.clone(), content.content.clone()),
            Stimulus::AmbientSnapshot { entries } => (
                "ambient-snapshot".to_string(),
                None,
                entries
                    .iter()
                    .map(|entry| entry.content.content.as_str())
                    .collect::<Vec<_>>()
                    .join("\n"),
            ),
        };
        events.record_eval_event(
            nuillu_types::ScopeId::root(),
            builtin::sensory(),
            0,
            Some(step_id.to_string()),
            crate::timeline::EvalEventPayload::StimulusPublished {
                modality,
                direction,
                content,
                step_id: step_id.to_string(),
            },
        );
        if let Some(visualizer) = visualizer {
            visualizer.send_event(VisualizerEvent::SensoryInput {
                tab_id: VisualizerTabId::new(case_id.to_string()),
                input: body,
            });
        }
    }
}

#[allow(clippy::too_many_arguments)]
async fn run_input_phase(
    case_id: &str,
    inputs: &[Stimulus],
    steps: &[EvalStep],
    sensory: &SensoryInputMailbox,
    harness: &InternalHarnessIo,
    blackboard: &Blackboard,
    utterances: &RecordingUtteranceSink,
    events: &RecordingRuntimeEventSink,
    clock: &dyn Clock,
    visualizer: Option<&VisualizerHook>,
    reporter: &LiveReporter,
    run_controller: &AgentRunController,
    eval_case: &EvalCase,
    judge: Option<&dyn RubricJudge>,
    memory: &dyn MemoryStore,
    timeline_runtime_origin_ms: u64,
    timeline_started_at: DateTime<Utc>,
    assertion_overrides: &Arc<Mutex<BTreeMap<String, AssertionOutcome>>>,
    terminal_output: &Arc<Mutex<Option<String>>>,
    step_failure: &Arc<Mutex<Option<String>>>,
    step_outcomes: &Arc<Mutex<Vec<serde_json::Value>>>,
) {
    if steps.is_empty() {
        publish_stimuli(case_id, inputs, sensory, clock, events, "input", visualizer).await;
        return;
    }
    for (index, step) in steps.iter().enumerate() {
        let step_started = Instant::now();
        if index > 0 {
            let settle_modules = step_settle_modules(step.wait_for.as_ref());
            match wait_for_step_modules_to_settle(
                blackboard,
                events,
                &settle_modules,
                RUNTIME_STEP_SETTLE_TIMEOUT,
            )
            .await
            {
                WaitOutcome::Met => {}
                WaitOutcome::Timeout => {
                    let modules = settle_modules
                        .iter()
                        .map(ModuleId::as_str)
                        .collect::<Vec<_>>()
                        .join(", ");
                    let message = format!(
                        "step {index} timed out waiting for prior activity to settle in [{modules}]"
                    );
                    step_failure
                        .lock()
                        .expect("step failure mutex poisoned")
                        .get_or_insert(message);
                    events.request_stop("step-settle-timeout");
                    return;
                }
                WaitOutcome::Stopped => return,
                WaitOutcome::AssertionNotMet => {
                    unreachable!("module settling does not evaluate assertions")
                }
            }
        }
        let step_memos = seed_memos(blackboard, clock, &step.memos, false)
            .await
            .expect("validated eval step memo seeds");
        let step_cognition = seed_cognition_log(blackboard, clock, &step.cognition_log).await;
        publish_setup_updates(harness, &step_memos, &step_cognition).await;
        let memo_wait_baseline = match &step.wait_for {
            Some(WaitFor::MemoFrom { scope, module, .. }) => {
                let scope = scope
                    .as_deref()
                    .map(parse_scope_id)
                    .transpose()
                    .expect("validated wait-for scope");
                Some(memo_count_for_module(blackboard, scope.as_ref(), &module.module_id()).await)
            }
            _ => None,
        };
        let utterance_wait_baseline = match &step.wait_for {
            Some(WaitFor::UtteranceFrom {
                scope,
                module,
                target,
                ..
            }) => Some(utterances.matching_count(scope.as_deref(), module.as_str(), target)),
            _ => None,
        };
        let step_id = step
            .id
            .clone()
            .unwrap_or_else(|| format!("step-{}", index + 1));
        publish_stimuli(
            case_id,
            &step.inputs,
            sensory,
            clock,
            events,
            &step_id,
            visualizer,
        )
        .await;

        let mut wait_result = WaitConditionResult::met();
        if let Some(wait_for) = &step.wait_for {
            wait_result = wait_for_condition(
                case_id,
                blackboard,
                utterances,
                events,
                wait_for,
                memo_wait_baseline,
                utterance_wait_baseline,
                eval_case,
                judge,
                memory,
                timeline_runtime_origin_ms,
                timeline_started_at,
                assertion_overrides,
                terminal_output,
                run_controller,
            )
            .await;
        }
        let wait_outcome = wait_result.outcome;

        if step.terminal && matches!(wait_outcome, WaitOutcome::Met) {
            run_controller.pause();
        }

        let mut check_results: Vec<serde_json::Value> = Vec::new();
        let mut must_pass_failure: Option<String> = None;
        if matches!(wait_outcome, WaitOutcome::Met) && !step.assertions.is_empty() {
            let snapshot = build_step_snapshot(blackboard, utterances).await;
            for check in &step.assertions {
                let (passed, diagnostic) = evaluate_step_check(check, &snapshot);
                let common = check.common();
                check_results.push(serde_json::json!({
                    "name": check.display_name(),
                    "kind": check.kind_name(),
                    "passed": passed,
                    "must_pass": common.must_pass,
                    "diagnostic": diagnostic,
                }));
                if !passed && common.must_pass && must_pass_failure.is_none() {
                    must_pass_failure = Some(format!(
                        "step {index} must-pass check '{name}' failed: {diag}",
                        name = check.display_name(),
                        diag = diagnostic
                            .clone()
                            .unwrap_or_else(|| "no diagnostic".to_string()),
                    ));
                }
            }
        }

        let status = match (&wait_outcome, &must_pass_failure) {
            (WaitOutcome::Timeout, _) => "timed-out",
            (WaitOutcome::Stopped, _) => "stopped",
            (WaitOutcome::AssertionNotMet, _) => "assertion-not-met",
            (_, Some(_)) => "check-failed",
            _ => "ok",
        };
        let mut outcome = serde_json::Map::new();
        outcome.insert("index".to_string(), serde_json::Value::from(index));
        if let Some(description) = &step.description {
            outcome.insert(
                "description".to_string(),
                serde_json::Value::String(description.content.clone()),
            );
        }
        outcome.insert(
            "status".to_string(),
            serde_json::Value::String(status.to_string()),
        );
        outcome.insert(
            "terminal".to_string(),
            serde_json::Value::Bool(step.terminal),
        );
        outcome.insert(
            "elapsed_ms".to_string(),
            serde_json::Value::from(duration_millis_u64(step_started.elapsed())),
        );
        outcome.insert(
            "assertions".to_string(),
            serde_json::Value::Array(check_results),
        );
        if !wait_result.assertion_attempts.is_empty() {
            outcome.insert(
                "until-assertion-attempts".to_string(),
                serde_json::to_value(&wait_result.assertion_attempts)
                    .expect("assertion outcomes serialize"),
            );
        }
        step_outcomes
            .lock()
            .expect("step outcomes mutex poisoned")
            .push(serde_json::Value::Object(outcome));
        let step_elapsed_ms = duration_millis_u64(step_started.elapsed());
        let _ = reporter.emit(
            Some(case_id),
            "step_finished",
            serde_json::json!({
                "index": index,
                "status": status,
                "terminal": step.terminal,
                "elapsed_ms": step_elapsed_ms,
            }),
            format!(
                "eval step end id={case_id} index={index} elapsed_ms={step_elapsed_ms} status={status}"
            ),
        );

        match wait_outcome {
            WaitOutcome::Met => {}
            WaitOutcome::Timeout => {
                let wait_label = wait_for_label(step.wait_for.as_ref());
                let message = format!("step {index} timed out waiting for {wait_label}",);
                step_failure
                    .lock()
                    .expect("step failure mutex poisoned")
                    .get_or_insert(message);
                events.request_stop("step-timeout");
                return;
            }
            WaitOutcome::Stopped => {
                let wait_label = wait_for_label(step.wait_for.as_ref());
                let message = format!("step {index} stopped before satisfying {wait_label}",);
                step_failure
                    .lock()
                    .expect("step failure mutex poisoned")
                    .get_or_insert(message);
                return;
            }
            WaitOutcome::AssertionNotMet => {
                events.request_stop("terminal-assertion-not-met");
                return;
            }
        }

        if let Some(message) = must_pass_failure {
            step_failure
                .lock()
                .expect("step failure mutex poisoned")
                .get_or_insert(message);
            events.request_stop("step-check-failed");
            return;
        }
        if step.terminal {
            events.request_stop("terminal-step-completed");
            return;
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum WaitOutcome {
    Met,
    Timeout,
    Stopped,
    AssertionNotMet,
}

struct WaitConditionResult {
    outcome: WaitOutcome,
    assertion_attempts: Vec<AssertionOutcome>,
}

impl WaitConditionResult {
    fn met() -> Self {
        Self {
            outcome: WaitOutcome::Met,
            assertion_attempts: Vec::new(),
        }
    }
}

fn step_settle_modules(wait_for: Option<&WaitFor>) -> Vec<ModuleId> {
    match wait_for {
        Some(WaitFor::MemoFrom { module, .. } | WaitFor::UtteranceFrom { module, .. }) => {
            vec![module.module_id()]
        }
        Some(WaitFor::Interoception { .. }) => vec![builtin::interoception()],
        None => Vec::new(),
    }
}

async fn wait_for_step_modules_to_settle(
    blackboard: &Blackboard,
    events: &RecordingRuntimeEventSink,
    modules: &[ModuleId],
    timeout: Duration,
) -> WaitOutcome {
    if modules.is_empty() {
        return WaitOutcome::Met;
    }
    let modules = modules
        .iter()
        .cloned()
        .collect::<std::collections::HashSet<_>>();
    let start = Instant::now();
    let poll = Duration::from_millis(50);

    loop {
        if events.stop_requested() {
            return WaitOutcome::Stopped;
        }

        let has_unsettled_target = blackboard
            .read(|bb| {
                bb.module_status_records().into_iter().any(|record| {
                    modules.contains(&record.owner.module)
                        && matches!(
                            record.status,
                            ModuleRunStatus::PendingBatch
                                | ModuleRunStatus::PendingActivationGate
                                | ModuleRunStatus::Activating
                        )
                })
            })
            .await;
        if !has_unsettled_target {
            return WaitOutcome::Met;
        }
        if start.elapsed() >= timeout {
            return WaitOutcome::Timeout;
        }
        tokio::time::sleep(poll).await;
    }
}

async fn wait_for_condition(
    case_id: &str,
    blackboard: &Blackboard,
    utterances: &RecordingUtteranceSink,
    events: &RecordingRuntimeEventSink,
    wait_for: &WaitFor,
    memo_baseline: Option<usize>,
    utterance_baseline: Option<usize>,
    eval_case: &EvalCase,
    judge: Option<&dyn RubricJudge>,
    memory: &dyn MemoryStore,
    timeline_runtime_origin_ms: u64,
    timeline_started_at: DateTime<Utc>,
    assertion_overrides: &Arc<Mutex<BTreeMap<String, AssertionOutcome>>>,
    terminal_output: &Arc<Mutex<Option<String>>>,
    run_controller: &AgentRunController,
) -> WaitConditionResult {
    match wait_for {
        WaitFor::MemoFrom {
            scope,
            module,
            timeout_ms,
        } => {
            let scope = scope
                .as_deref()
                .map(parse_scope_id)
                .transpose()
                .expect("validated wait-for scope");
            let target = module.module_id();
            let baseline = memo_baseline
                .unwrap_or_else(|| panic!("memo wait baseline must be captured before input"));
            let deadline = Duration::from_millis(*timeout_ms);
            let start = Instant::now();
            let poll = Duration::from_millis(50);
            loop {
                let count = memo_count_for_module(blackboard, scope.as_ref(), &target).await;
                if count > baseline {
                    return WaitConditionResult::met();
                }
                if events.stop_requested() {
                    return WaitConditionResult {
                        outcome: WaitOutcome::Stopped,
                        assertion_attempts: Vec::new(),
                    };
                }
                let elapsed = start.elapsed();
                if elapsed >= deadline {
                    return WaitConditionResult {
                        outcome: WaitOutcome::Timeout,
                        assertion_attempts: Vec::new(),
                    };
                }
                let remaining = deadline.saturating_sub(elapsed);
                tokio::time::sleep(remaining.min(poll)).await;
            }
        }
        WaitFor::UtteranceFrom {
            scope,
            module,
            target,
            until_assertion,
            max_matches,
            timeout_ms,
        } => {
            let baseline = utterance_baseline
                .unwrap_or_else(|| panic!("utterance wait baseline must be captured before input"));
            let deadline = Duration::from_millis(*timeout_ms);
            let start = Instant::now();
            let poll = Duration::from_millis(50);
            let Some(assertion_name) = until_assertion.as_deref() else {
                loop {
                    if let Some(candidate) =
                        utterances.matching_at(scope.as_deref(), module.as_str(), target, baseline)
                    {
                        *terminal_output
                            .lock()
                            .expect("terminal output mutex poisoned") = Some(candidate.text);
                        return WaitConditionResult::met();
                    }
                    if events.stop_requested() {
                        return WaitConditionResult {
                            outcome: WaitOutcome::Stopped,
                            assertion_attempts: Vec::new(),
                        };
                    }
                    let elapsed = start.elapsed();
                    if elapsed >= deadline {
                        return WaitConditionResult {
                            outcome: WaitOutcome::Timeout,
                            assertion_attempts: Vec::new(),
                        };
                    }
                    tokio::time::sleep(deadline.saturating_sub(elapsed).min(poll)).await;
                }
            };
            let assertion = eval_case
                .assertions()
                .iter()
                .find(|assertion| assertion.display_name() == assertion_name)
                .expect("validated until-assertion reference");
            let mut matching_utterances = Vec::<RecordedUtterance>::new();
            let mut cumulative_outputs = Vec::<String>::new();
            let mut results = Vec::<Option<AssertionOutcome>>::new();
            let mut judgments: FuturesUnordered<LocalBoxFuture<'_, (usize, AssertionOutcome)>> =
                FuturesUnordered::new();
            let mut closed_with = None;

            loop {
                if closed_with.is_none() {
                    while matching_utterances.len() < *max_matches {
                        let next_index = baseline + matching_utterances.len();
                        let Some(candidate) = utterances.matching_at(
                            scope.as_deref(),
                            module.as_str(),
                            target,
                            next_index,
                        ) else {
                            break;
                        };
                        matching_utterances.push(candidate);
                        let cumulative_output = matching_utterances
                            .iter()
                            .map(|utterance| utterance.text.as_str())
                            .collect::<Vec<_>>()
                            .join("\n");
                        cumulative_outputs.push(cumulative_output.clone());
                        results.push(None);
                        let attempt_index = matching_utterances.len() - 1;
                        let snapshot = build_live_assertion_snapshot(
                            case_id,
                            cumulative_output,
                            &matching_utterances,
                            blackboard,
                            utterances,
                            events,
                            memory,
                            timeline_runtime_origin_ms,
                            timeline_started_at,
                        )
                        .await;
                        let judge_timeout = deadline.saturating_sub(start.elapsed());
                        let judge_timeout_ms = *timeout_ms;
                        let judgment = async move {
                            let outcome = match snapshot {
                                Ok(snapshot) => match tokio::time::timeout(
                                    judge_timeout,
                                    evaluate_assertion(
                                        eval_case,
                                        &empty_trace_snapshot(),
                                        &snapshot,
                                        judge,
                                        assertion,
                                    ),
                                )
                                .await
                                {
                                    Ok(outcome) => outcome,
                                    Err(_) => live_assertion_timeout(assertion, judge_timeout_ms),
                                },
                                Err(error) => live_assertion_error(assertion, error),
                            };
                            (attempt_index, outcome)
                        }
                        .boxed_local();
                        judgments.push(judgment);
                    }
                    if matching_utterances.len() == *max_matches {
                        run_controller.pause();
                        closed_with = Some(WaitOutcome::AssertionNotMet);
                    } else if events.stop_requested() {
                        run_controller.pause();
                        closed_with = Some(WaitOutcome::Stopped);
                    } else if start.elapsed() >= deadline {
                        run_controller.pause();
                        closed_with = Some(WaitOutcome::Timeout);
                    }
                }

                if let Some(pass_index) = results.iter().position(|result| {
                    result
                        .as_ref()
                        .is_some_and(|outcome| outcome.passed && !outcome.errored)
                }) && results[..pass_index].iter().all(Option::is_some)
                {
                    let outcome = results[pass_index]
                        .clone()
                        .expect("passing result is present");
                    assertion_overrides
                        .lock()
                        .expect("assertion override mutex poisoned")
                        .insert(assertion_name.to_string(), outcome);
                    *terminal_output
                        .lock()
                        .expect("terminal output mutex poisoned") =
                        Some(cumulative_outputs[pass_index].clone());
                    run_controller.pause();
                    events.request_stop("terminal-assertion-matched");
                    let assertion_attempts = results
                        .iter()
                        .take(pass_index + 1)
                        .map(|result| result.clone().expect("earlier results are complete"))
                        .collect();
                    return WaitConditionResult {
                        outcome: WaitOutcome::Met,
                        assertion_attempts,
                    };
                }

                if let Some(outcome) = closed_with
                    && judgments.is_empty()
                {
                    let assertion_attempts = results
                        .into_iter()
                        .flatten()
                        .collect::<Vec<AssertionOutcome>>();
                    if let Some(last) = assertion_attempts.last().cloned() {
                        assertion_overrides
                            .lock()
                            .expect("assertion override mutex poisoned")
                            .insert(assertion_name.to_string(), last);
                    }
                    if let Some(output) = cumulative_outputs.last().cloned() {
                        *terminal_output
                            .lock()
                            .expect("terminal output mutex poisoned") = Some(output);
                    }
                    return WaitConditionResult {
                        outcome,
                        assertion_attempts,
                    };
                }

                if judgments.is_empty() {
                    let elapsed = start.elapsed();
                    if elapsed < deadline {
                        tokio::time::sleep(deadline.saturating_sub(elapsed).min(poll)).await;
                    } else {
                        tokio::task::yield_now().await;
                    }
                    continue;
                }

                tokio::select! {
                    result = judgments.next() => {
                        if let Some((attempt_index, outcome)) = result {
                            results[attempt_index] = Some(outcome);
                        }
                    }
                    () = tokio::time::sleep(poll) => {}
                }
            }
        }
        WaitFor::Interoception {
            timeout_ms,
            mode,
            wake_arousal_at_least,
            wake_arousal_at_most,
        } => {
            let deadline = Duration::from_millis(*timeout_ms);
            let start = Instant::now();
            let poll = Duration::from_millis(50);
            loop {
                let matched = blackboard
                    .read(|bb| {
                        let state = bb.interoception();
                        mode.is_none_or(|mode| eval_mode_matches(mode, state.mode))
                            && (!wake_arousal_min_is_set(*wake_arousal_at_least)
                                || f64::from(state.wake_arousal) >= *wake_arousal_at_least)
                            && (!wake_arousal_max_is_set(*wake_arousal_at_most)
                                || f64::from(state.wake_arousal) <= *wake_arousal_at_most)
                    })
                    .await;
                if matched {
                    return WaitConditionResult::met();
                }
                if events.stop_requested() {
                    return WaitConditionResult {
                        outcome: WaitOutcome::Stopped,
                        assertion_attempts: Vec::new(),
                    };
                }
                let elapsed = start.elapsed();
                if elapsed >= deadline {
                    return WaitConditionResult {
                        outcome: WaitOutcome::Timeout,
                        assertion_attempts: Vec::new(),
                    };
                }
                let remaining = deadline.saturating_sub(elapsed);
                tokio::time::sleep(remaining.min(poll)).await;
            }
        }
    }
}

fn wait_for_label(wait_for: Option<&WaitFor>) -> String {
    match wait_for {
        Some(WaitFor::MemoFrom {
            scope,
            module,
            timeout_ms,
        }) => format!(
            "memo from module '{module}'{} within {timeout_ms}ms",
            scope
                .as_deref()
                .map(|scope| format!(" in scope '{scope}'"))
                .unwrap_or_default(),
            module = module.as_str(),
        ),
        Some(WaitFor::UtteranceFrom {
            scope,
            module,
            target,
            until_assertion,
            max_matches,
            timeout_ms,
        }) => format!(
            "utterance from module '{module}'{} to target '{target}'{} within {timeout_ms}ms",
            scope
                .as_deref()
                .map(|scope| format!(" in scope '{scope}'"))
                .unwrap_or_default(),
            until_assertion
                .as_deref()
                .map(|name| format!(" until assertion '{name}' passes (max {max_matches} matches)"))
                .unwrap_or_default(),
            module = module.as_str(),
        ),
        Some(WaitFor::Interoception {
            timeout_ms,
            mode,
            wake_arousal_at_least,
            wake_arousal_at_most,
        }) => {
            let mut conditions = Vec::new();
            if let Some(mode) = mode {
                conditions.push(format!("mode={}", mode.as_str()));
            }
            if wake_arousal_min_is_set(*wake_arousal_at_least) {
                conditions.push(format!("wake_arousal>={wake_arousal_at_least:.2}"));
            }
            if wake_arousal_max_is_set(*wake_arousal_at_most) {
                conditions.push(format!("wake_arousal<={wake_arousal_at_most:.2}"));
            }
            format!(
                "interoception {} within {timeout_ms}ms",
                conditions.join(", ")
            )
        }
        None => "<no wait-for>".to_string(),
    }
}

fn eval_mode_matches(
    expected: EvalInteroceptiveMode,
    actual: nuillu_blackboard::InteroceptiveMode,
) -> bool {
    match expected {
        EvalInteroceptiveMode::Wake => actual == nuillu_blackboard::InteroceptiveMode::Wake,
        EvalInteroceptiveMode::NremPressure => {
            actual == nuillu_blackboard::InteroceptiveMode::NremPressure
        }
        EvalInteroceptiveMode::RemPressure => {
            actual == nuillu_blackboard::InteroceptiveMode::RemPressure
        }
    }
}

async fn memo_count_for_module(
    blackboard: &Blackboard,
    scope: Option<&nuillu_types::ScopeId>,
    module: &ModuleId,
) -> usize {
    blackboard
        .read(|bb| {
            bb.recent_memo_logs()
                .into_iter()
                .filter(|record| {
                    &record.owner.module == module
                        && scope.is_none_or(|scope| &record.owner.scope == scope)
                })
                .count()
        })
        .await
}

async fn build_step_snapshot(
    blackboard: &Blackboard,
    utterances: &RecordingUtteranceSink,
) -> CaseArtifact {
    let mut artifact = CaseArtifact::new(
        utterances
            .last_complete()
            .map(|utterance| utterance.text)
            .unwrap_or_default(),
    );
    add_observations(&mut artifact, blackboard, utterances).await;
    artifact
}

#[allow(clippy::too_many_arguments)]
async fn build_live_assertion_snapshot(
    case_id: &str,
    output: String,
    matching_utterances: &[RecordedUtterance],
    blackboard: &Blackboard,
    utterances: &RecordingUtteranceSink,
    events: &RecordingRuntimeEventSink,
    memory: &dyn MemoryStore,
    timeline_runtime_origin_ms: u64,
    timeline_started_at: DateTime<Utc>,
) -> Result<CaseArtifact> {
    let mut artifact = CaseArtifact::new(output);
    add_observations(&mut artifact, blackboard, utterances).await;
    if let Some(agent) = artifact
        .observations
        .get_mut("agent")
        .and_then(serde_json::Value::as_object_mut)
    {
        agent.insert(
            "utterances".to_string(),
            serde_json::to_value(matching_utterances)
                .context("serialize live assertion utterances")?,
        );
    }
    let mut timeline = build_eval_timeline(
        blackboard,
        utterances,
        events,
        timeline_runtime_origin_ms,
        timeline_started_at,
    )
    .await;
    if let Some(last_utterance) = matching_utterances.last()
        && let Ok(emitted_at) = DateTime::parse_from_rfc3339(&last_utterance.emitted_at)
    {
        let cutoff_ms = datetime_offset_ms(timeline_started_at, emitted_at.with_timezone(&Utc));
        timeline.retain(|event| event.offset_ms <= cutoff_ms);
    }
    artifact.observations.insert(
        "timeline".to_string(),
        serde_json::to_value(timeline).context("serialize live assertion timeline")?,
    );
    let mut last_state = build_runtime_last_state_dump(
        case_id,
        &artifact,
        blackboard,
        memory,
        utterances,
        events.event_count(),
    )
    .await?;
    last_state.utterances = utterance_dumps(matching_utterances.to_vec());
    add_last_state_observation(&mut artifact, &last_state)?;
    Ok(artifact)
}

fn live_assertion_error(assertion: &Assertion, error: anyhow::Error) -> AssertionOutcome {
    AssertionOutcome {
        name: assertion.display_name(),
        kind: assertion.kind_name().to_string(),
        passed: false,
        errored: true,
        must_pass: assertion.common().must_pass,
        weight: assertion.common().weight,
        diagnostic: Some(format!(
            "failed to build live assertion snapshot: {error:#}"
        )),
        rubric: None,
    }
}

fn live_assertion_timeout(assertion: &Assertion, timeout_ms: u64) -> AssertionOutcome {
    AssertionOutcome {
        name: assertion.display_name(),
        kind: assertion.kind_name().to_string(),
        passed: false,
        errored: true,
        must_pass: assertion.common().must_pass,
        weight: assertion.common().weight,
        diagnostic: Some(format!(
            "live assertion judge did not finish within the {timeout_ms}ms step deadline"
        )),
        rubric: None,
    }
}

fn evaluate_step_check(check: &Assertion, artifact: &CaseArtifact) -> (bool, Option<String>) {
    match check {
        Assertion::JsonPointerEquals {
            pointer, expected, ..
        } => {
            let json = artifact.as_json();
            let actual = pointer_text(&json, pointer);
            let passed = actual.as_deref() == Some(expected.as_str());
            let diagnostic = (!passed).then(|| match actual {
                Some(actual) => format!(
                    "expected JSON pointer {pointer:?} to equal {expected:?}, got {actual:?}"
                ),
                None => format!("JSON pointer {pointer:?} did not match artifact"),
            });
            (passed, diagnostic)
        }
        Assertion::JsonPointerContains {
            pointer, contains, ..
        } => {
            let json = artifact.as_json();
            let actual = pointer_text(&json, pointer);
            let passed = actual
                .as_deref()
                .is_some_and(|text| text.contains(contains));
            let diagnostic = (!passed).then(|| match actual {
                Some(actual) => format!(
                    "expected JSON pointer {pointer:?} to contain {contains:?}, got {actual:?}"
                ),
                None => format!("JSON pointer {pointer:?} did not match artifact"),
            });
            (passed, diagnostic)
        }
        Assertion::JsonPointerNumericInRange {
            pointer, min, max, ..
        } => {
            let json = artifact.as_json();
            let actual = pointer_number(&json, pointer);
            numeric_range_outcome(pointer, actual, *min, *max)
        }
        Assertion::ArtifactTextContains {
            field, contains, ..
        } => {
            let field = field.unwrap_or(ArtifactTextField::Output);
            let text = artifact_text(artifact, field);
            let passed = text.contains(contains);
            let diagnostic = (!passed).then(|| {
                format!(
                    "expected {field_name} to contain {contains:?}",
                    field_name = field_label(field),
                )
            });
            (passed, diagnostic)
        }
        Assertion::ArtifactTextExact { field, exact, .. } => {
            let field = field.unwrap_or(ArtifactTextField::Output);
            let expected = normalize_text_block(&exact.content);
            let text = normalize_text_block(artifact_text(artifact, field));
            let passed = text == expected;
            let diagnostic = (!passed).then(|| {
                format!(
                    "expected {field_name} to equal {expected:?}, got {text:?}",
                    field_name = field_label(field),
                )
            });
            (passed, diagnostic)
        }
        _ => (true, None),
    }
}

async fn handle_visualizer_commands(
    case_id: &str,
    visualizer: &mut VisualizerHook,
    sensory: Option<&SensoryInputMailbox>,
    blackboard: &Blackboard,
    memory: &dyn MemoryStore,
    cognition_log_repository: &dyn CognitionLogRepository,
    clock: &dyn Clock,
) -> VisualizerCommandOutcome {
    let mut outcome = VisualizerCommandOutcome::default();
    let start_activation_id =
        start_activation_action_id(&VisualizerTabId::new(case_id.to_string()));
    loop {
        let message = match visualizer.commands.try_recv() {
            Ok(message) => message,
            Err(std::sync::mpsc::TryRecvError::Empty) => break,
            Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                visualizer.request_shutdown();
                outcome.shutdown = true;
                break;
            }
        };
        let command = match message {
            VisualizerClientMessage::Hello { .. } => continue,
            VisualizerClientMessage::InvokeAction { action_id } => {
                if action_id == start_activation_id {
                    outcome.start_requested = true;
                }
                continue;
            }
            VisualizerClientMessage::Command { command } => command,
        };
        match command {
            VisualizerCommand::Shutdown => {
                visualizer.request_shutdown();
                outcome.shutdown = true;
            }
            VisualizerCommand::PublishSensoryInput { tab_id, input }
                if tab_id.as_str() == case_id =>
            {
                let Some(sensory) = sensory else {
                    visualizer.send_event(VisualizerEvent::Log {
                        tab_id,
                        message: "this runtime does not accept sensory input".to_string(),
                    });
                    continue;
                };
                let _ = sensory.publish(input.clone()).await;
                visualizer.send_event(VisualizerEvent::SensoryInput { tab_id, input });
            }
            VisualizerCommand::SendOneShotSensoryInput { tab_id, input }
                if tab_id.as_str() == case_id =>
            {
                let Some(sensory) = sensory else {
                    visualizer.send_event(VisualizerEvent::Log {
                        tab_id,
                        message: "this runtime does not accept sensory input".to_string(),
                    });
                    continue;
                };
                let observed_at = clock.now();
                let body = SensoryInput::OneShot {
                    modality: SensoryModality::parse(input.modality),
                    direction: input.direction,
                    content: input.content,
                    observed_at,
                };
                let _ = sensory.publish(body.clone()).await;
                visualizer.send_event(VisualizerEvent::SensoryInput {
                    tab_id,
                    input: body,
                });
            }
            VisualizerCommand::LoadMemoryRecords {
                tab_id,
                scope,
                offset,
                limit,
            } if tab_id.as_str() == case_id => {
                let scope_for_event = scope.clone();
                let records = match scope {
                    MemoryRecordScope::Latest => memory
                        .list_recent(offset, limit.saturating_add(1))
                        .await
                        .unwrap_or_default(),
                    MemoryRecordScope::Search { query } => {
                        let mut query = MemoryQuery::text(query, limit.saturating_add(1));
                        query.offset = offset;
                        memory.search(&query).await.unwrap_or_default()
                    }
                }
                .into_iter()
                .map(memory_record_view)
                .collect::<Vec<_>>();
                let (records, has_more) = trim_visualizer_chunk(records, limit);
                visualizer.send_event(VisualizerEvent::MemoryRecordsLoaded {
                    tab_id,
                    scope: scope_for_event,
                    offset,
                    records,
                    has_more,
                });
            }
            VisualizerCommand::LoadCognitionLogEntries {
                tab_id,
                cursor,
                limit,
            } if tab_id.as_str() == case_id => {
                let records = cognition_log_repository
                    .page(cursor, limit.saturating_add(1))
                    .await
                    .unwrap_or_default();
                let (records, has_more) = trim_visualizer_chunk(records, limit);
                visualizer.send_event(VisualizerEvent::CognitionLogEntriesLoaded {
                    tab_id,
                    cursor,
                    entries: records
                        .into_iter()
                        .map(|record| PersistedCognitionEntryView {
                            id: record.id,
                            source: record.source.to_string(),
                            at: record.entry.at,
                            origin: record.entry.origin.owner.to_string(),
                            text: record.entry.text,
                        })
                        .collect(),
                    has_more,
                });
            }
            VisualizerCommand::LoadLinkedMemories {
                tab_id,
                memory_index,
                relation_filter,
                offset,
                limit,
            } if tab_id.as_str() == case_id => {
                let relation_filter = relation_filter
                    .into_iter()
                    .filter_map(|relation| parse_memory_relation(&relation))
                    .collect::<Vec<_>>();
                let records = memory
                    .linked(&LinkedMemoryQuery {
                        memory_indexes: vec![MemoryIndex::new(memory_index.clone())],
                        relation_filter,
                        direction: MemoryLinkDirection::Both,
                        offset,
                        limit: limit.saturating_add(1),
                    })
                    .await
                    .map(|records| records.into_iter().map(linked_memory_record_view).collect())
                    .unwrap_or_default();
                let (records, has_more) = trim_visualizer_chunk(records, limit);
                visualizer.send_event(VisualizerEvent::LinkedMemoryRecordsLoaded {
                    tab_id,
                    memory_index,
                    offset,
                    records,
                    has_more,
                });
            }
            VisualizerCommand::DeleteMemory {
                tab_id,
                memory_index,
            } if tab_id.as_str() == case_id => {
                let index = MemoryIndex::new(memory_index.clone());
                let _ = memory.delete(&index).await;
                blackboard
                    .apply(BlackboardCommand::RemoveMemoryMetadata { index })
                    .await;
                visualizer.send_event(VisualizerEvent::MemoryDeleted {
                    tab_id,
                    memory_index,
                });
            }
            VisualizerCommand::SetModuleDisabled {
                tab_id,
                module,
                disabled,
            } if tab_id.as_str() == case_id => match ModuleId::new(module.clone()) {
                Ok(module_id) => {
                    blackboard
                        .apply(BlackboardCommand::SetModuleForcedDisabled {
                            module: module_id,
                            disabled,
                        })
                        .await;
                }
                Err(_) => {
                    visualizer.send_event(VisualizerEvent::Log {
                        tab_id,
                        message: format!("invalid module id: {module}"),
                    });
                }
            },
            VisualizerCommand::SetModuleSettings { tab_id, settings }
                if tab_id.as_str() == case_id =>
            {
                apply_visualizer_module_settings(&tab_id, visualizer, blackboard, settings).await;
            }
            VisualizerCommand::ResetModuleSessionHistory { tab_id, .. }
                if tab_id.as_str() == case_id =>
            {
                visualizer.send_event(VisualizerEvent::Log {
                    tab_id,
                    message: "module session reset is only supported by nuillu-server".to_string(),
                });
            }
            VisualizerCommand::CreateAmbientSensoryRow { tab_id, .. }
            | VisualizerCommand::UpdateAmbientSensoryRow { tab_id, .. }
            | VisualizerCommand::RemoveAmbientSensoryRow { tab_id, .. }
            | VisualizerCommand::CreateSceneRow { tab_id, .. }
            | VisualizerCommand::UpdateSceneRow { tab_id, .. }
            | VisualizerCommand::RemoveSceneRow { tab_id, .. }
            | VisualizerCommand::SaveSceneState { tab_id, .. }
            | VisualizerCommand::SendScenePersonMessage { tab_id, .. }
                if tab_id.as_str() == case_id =>
            {
                visualizer.send_event(VisualizerEvent::Log {
                    tab_id,
                    message: "scene editing is only supported by nuillu-server".to_string(),
                });
            }
            _ => {}
        }
    }
    outcome
}

#[derive(Debug, Default)]
struct VisualizerCommandOutcome {
    shutdown: bool,
    start_requested: bool,
}

fn parse_memory_relation(value: &str) -> Option<MemoryLinkRelation> {
    match value.trim().to_ascii_lowercase().as_str() {
        "related" => Some(MemoryLinkRelation::Related),
        "supports" => Some(MemoryLinkRelation::Supports),
        "contradicts" => Some(MemoryLinkRelation::Contradicts),
        "updates" => Some(MemoryLinkRelation::Updates),
        "corrects" => Some(MemoryLinkRelation::Corrects),
        "derived_from" | "derived-from" => Some(MemoryLinkRelation::DerivedFrom),
        _ => None,
    }
}

pub(crate) async fn apply_visualizer_module_settings(
    tab_id: &VisualizerTabId,
    visualizer: &VisualizerHook,
    blackboard: &Blackboard,
    settings: ModuleSettingsView,
) -> bool {
    let update = match build_module_policy_update(blackboard, &settings).await {
        Ok(update) => update,
        Err(message) => {
            visualizer.send_event(VisualizerEvent::Log {
                tab_id: tab_id.clone(),
                message,
            });
            return false;
        }
    };

    blackboard
        .apply(BlackboardCommand::SetModulePolicies {
            policies: vec![update],
        })
        .await;
    true
}

async fn build_module_policy_update(
    blackboard: &Blackboard,
    settings: &ModuleSettingsView,
) -> Result<(ModuleId, ModulePolicy), String> {
    let module = ModuleId::new(settings.module.clone())
        .map_err(|_| format!("invalid module id: {}", settings.module))?;
    if settings.replica_min > settings.replica_max {
        return Err(format!(
            "{} replica min {} exceeds max {}",
            settings.module, settings.replica_min, settings.replica_max
        ));
    }
    if !settings.bpm_min.is_finite()
        || !settings.bpm_max.is_finite()
        || settings.bpm_min <= 0.0
        || settings.bpm_max <= 0.0
    {
        return Err(format!(
            "{} BPM range must be positive and finite",
            settings.module
        ));
    }
    if settings.bpm_min > settings.bpm_max {
        return Err(format!(
            "{} BPM min {} exceeds max {}",
            settings.module, settings.bpm_min, settings.bpm_max
        ));
    }

    let (policy, capacity) = blackboard
        .read(|bb| {
            let policy = bb.module_policies().get(&module).cloned();
            let capacity = bb.module_replica_capacity(&module);
            (policy, capacity)
        })
        .await;
    let Some(mut policy) = policy else {
        return Err(format!(
            "module settings target is not registered: {}",
            settings.module
        ));
    };
    let capacity = capacity.unwrap_or_else(|| policy.max_active_replicas());
    if settings.replica_max > capacity {
        return Err(format!(
            "{} replica max {} exceeds hard cap {}",
            settings.module, settings.replica_max, capacity
        ));
    }

    policy.replicas_range = ReplicaCapRange::new(settings.replica_min, settings.replica_max)
        .map_err(|error| format!("{} invalid replica range: {error}", settings.module))?;
    policy.rate_limit_range = Bpm::range(settings.bpm_min, settings.bpm_max);
    policy.zero_replica_window = match settings.zero_replica_window {
        ZeroReplicaWindowView::Disabled => ZeroReplicaWindowPolicy::Disabled,
        ZeroReplicaWindowView::EveryControllerActivations { period } => {
            if period == 0 {
                return Err(format!(
                    "{} zero-window period must be greater than zero",
                    settings.module
                ));
            }
            ZeroReplicaWindowPolicy::EveryControllerActivations(period)
        }
    };

    Ok((module, policy))
}

pub(crate) async fn emit_visualizer_memory_records(
    case_id: &str,
    visualizer: &mut VisualizerHook,
    blackboard: &Blackboard,
    memory: &dyn MemoryStore,
    offset: usize,
    limit: usize,
) {
    let records = list_visualizer_memories(blackboard, memory, 0, usize::MAX).await;
    visualizer.set_memory_cache(case_id, records.clone());
    let (records, has_more) = memory_chunk_from_records(&records, offset, limit);
    visualizer.send_event(VisualizerEvent::MemoryRecordsLoaded {
        tab_id: VisualizerTabId::new(case_id.to_string()),
        scope: MemoryRecordScope::Latest,
        offset,
        records,
        has_more,
    });
}

async fn list_visualizer_memories(
    _blackboard: &Blackboard,
    memory: &dyn MemoryStore,
    offset: usize,
    limit: usize,
) -> Vec<MemoryRecordView> {
    memory
        .list_recent(offset, limit)
        .await
        .unwrap_or_default()
        .into_iter()
        .map(memory_record_view)
        .collect()
}

fn memory_chunk_from_records(
    records: &[MemoryRecordView],
    offset: usize,
    limit: usize,
) -> (Vec<MemoryRecordView>, bool) {
    let start = offset.min(records.len());
    let end = start.saturating_add(limit).min(records.len());
    (records[start..end].to_vec(), end < records.len())
}

fn trim_visualizer_chunk<T>(mut records: Vec<T>, limit: usize) -> (Vec<T>, bool) {
    if records.len() > limit {
        records.truncate(limit);
        (records, true)
    } else {
        (records, false)
    }
}

pub(crate) struct EvalEnvironment {
    pub(crate) blackboard: Blackboard,
    pub(crate) caps: CapabilityProviders,
    pub(crate) memory: Rc<dyn MemoryStore>,
    pub(crate) cognition_log_repository: Rc<dyn CognitionLogRepository>,
    pub(crate) memory_caps: MemoryCapabilities,
    pub(crate) policy_caps: PolicyCapabilities,
    pub(crate) utterances: Rc<RecordingUtteranceSink>,
    pub(crate) actions: Rc<ActionActivityTracker>,
    pub(crate) events: Rc<RecordingRuntimeEventSink>,
    pub(crate) clock: Rc<dyn Clock>,
    pub(crate) utterance_sink: Rc<dyn UtteranceSink>,
}

struct AnchoredRealtimeClock {
    base: DateTime<Utc>,
    started: Instant,
}

impl AnchoredRealtimeClock {
    fn new(base: DateTime<Utc>) -> Self {
        Self {
            base,
            started: Instant::now(),
        }
    }
}

#[async_trait(?Send)]
impl Clock for AnchoredRealtimeClock {
    fn now(&self) -> DateTime<Utc> {
        self.base + ChronoDuration::from_std(self.started.elapsed()).unwrap_or_default()
    }

    async fn sleep_until(&self, deadline: DateTime<Utc>) {
        let remaining = deadline - self.now();
        let Ok(duration) = remaining.to_std() else {
            return;
        };
        if duration.is_zero() {
            return;
        }
        tokio::time::sleep(duration).await;
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) async fn build_eval_environment(
    output_dir: &Path,
    config: &RunnerConfig,
    allocation: ResourceAllocation,
    limits: &EvalLimits,
    action_modules: Vec<ModuleId>,
    case_now: Option<DateTime<FixedOffset>>,
    memory_seeds: &[crate::cases::MemorySeed],
    memory_links: &[MemoryLinkSeed],
    policies: &[PolicySeed],
    case_id: &str,
    reporter: &LiveReporter,
    visualizer: Option<VisualizerEventSink>,
) -> Result<EvalEnvironment> {
    let blackboard = Blackboard::with_allocation(allocation);
    let events = Rc::new(RecordingRuntimeEventSink::new(
        case_id.to_string(),
        reporter.clone(),
        visualizer.clone(),
    ));
    let actions = Rc::new(ActionActivityTracker::new(action_modules));
    let utterances = Rc::new(RecordingUtteranceSink::new(
        case_id.to_string(),
        reporter.clone(),
        actions.clone(),
        visualizer.clone(),
    ));
    let external_action_executor = Rc::new(EvalExternalActionExecutor::new(
        case_id.to_string(),
        reporter.clone(),
        actions.clone(),
    ));
    let clock: Rc<dyn Clock> = match case_now {
        Some(now) => Rc::new(AnchoredRealtimeClock::new(now.with_timezone(&Utc))),
        None => Rc::new(SystemClock),
    };
    let agent_store = connect_agent_store(output_dir, config).await?;
    let memory: Rc<dyn MemoryStore> = Rc::new(agent_store.memory_store());
    let policy_store: Rc<dyn PolicyStore> = Rc::new(agent_store.policy_store());
    let cognition_log_repository: Rc<dyn CognitionLogRepository> =
        Rc::new(agent_store.cognition_log_repository());
    let memory_caps = MemoryCapabilities::new(
        blackboard.clone(),
        clock.clone(),
        memory.clone(),
        Vec::new(),
    );
    let policy_caps = PolicyCapabilities::new(
        blackboard.clone(),
        clock.clone(),
        policy_store.clone(),
        Vec::new(),
    );
    seed_and_bootstrap_eval_startup_context(
        &memory_caps,
        &policy_caps,
        memory.as_ref(),
        policy_store.as_ref(),
        &blackboard,
        clock.as_ref(),
        case_now,
        memory_seeds,
        memory_links,
        policies,
    )
    .await?;

    let llm_observer = visualizer
        .clone()
        .map(|sender| VisualizerLlmObserver::new(case_id.to_string(), sender));
    let mut tiers = build_tiers(
        &config.cheap_backend,
        &config.default_backend,
        &config.premium_backend,
        &config.image_backend,
        &config.llm_concurrency_pool,
        llm_observer,
        Some(eval_llm_log_context(config, case_id)),
        None,
    )
    .map_err(|error| RunnerError::Driver {
        path: output_dir.to_path_buf(),
        message: error.to_string(),
    })?;
    let llm_call_limit = LlmCallLimitHook::new(
        case_id.to_string(),
        limits.max_llm_calls,
        events.stop.clone(),
        reporter.clone(),
    );
    for handle in [
        &mut tiers.cheap,
        &mut tiers.default,
        &mut tiers.premium,
        &mut tiers.image,
    ] {
        handle
            .lutum
            .extend_hooks(LutumHooksSet::new().with_on_model_input(llm_call_limit.clone()));
    }
    let runtime_policy = RuntimePolicy {
        memo_retained_per_owner: EVAL_MEMO_RETAINED_PER_OWNER,
        cognition_log_retained_entries: EVAL_COGNITION_LOG_RETAINED_ENTRIES,
        session_compaction: session_compaction_policy(config),
        interoception: interoception_runtime_policy(limits),
        ..RuntimePolicy::default()
    };
    let caps = CapabilityProviders::new(CapabilityProviderConfig {
        ports: CapabilityProviderPorts {
            blackboard: blackboard.clone(),
            cognition_log_port: cognition_log_repository.clone(),
            clock: clock.clone(),
            tiers,
        },
        runtime: CapabilityProviderRuntime {
            event_sink: events.clone(),
            policy: runtime_policy,
            external_action_executor,
            ..CapabilityProviderRuntime::default()
        },
    });
    let utterance_sink: Rc<dyn UtteranceSink> = utterances.clone();

    Ok(EvalEnvironment {
        blackboard,
        caps,
        memory,
        cognition_log_repository,
        memory_caps,
        policy_caps,
        utterances,
        actions,
        events,
        clock,
        utterance_sink,
    })
}

pub(crate) fn eval_llm_log_context(config: &RunnerConfig, case_id: &str) -> LlmLogContext {
    LlmLogContext::new(
        config.llm_log_root.clone(),
        vec![config.run_id.clone(), case_id.to_string()],
    )
}

pub(crate) fn eval_llm_log_directory(config: &RunnerConfig, case_id: &str) -> PathBuf {
    eval_llm_log_context(config, case_id).namespace_dir()
}

fn session_compaction_policy(config: &RunnerConfig) -> SessionCompactionPolicy {
    SessionCompactionPolicy::new_with_image(
        config.cheap_backend.compaction_input_token_threshold,
        config.default_backend.compaction_input_token_threshold,
        config.premium_backend.compaction_input_token_threshold,
        config.image_backend.compaction_input_token_threshold,
    )
}

fn interoception_runtime_policy(limits: &EvalLimits) -> InteroceptionRuntimePolicy {
    InteroceptionRuntimePolicy {
        quiet_sleep_threshold: Duration::from_millis(limits.interoception.quiet_sleep_threshold_ms),
        wake_arousal_change_multiplier: limits.interoception.wake_arousal_change_multiplier as f32,
        affect_arousal_change_multiplier: limits.interoception.affect_arousal_change_multiplier
            as f32,
    }
}

async fn connect_agent_store(output_dir: &Path, config: &RunnerConfig) -> Result<LibsqlAgentStore> {
    let (memory_embedder, memory_profile, memory_dimensions) =
        build_embedder(&config.embedding_backend)?;
    let (policy_embedder, policy_profile, policy_dimensions) =
        build_embedder(&config.embedding_backend)?;
    LibsqlAgentStore::connect(
        LibsqlAgentStoreConfig::local(
            output_dir.join("agent.db"),
            memory_dimensions,
            policy_dimensions,
        )
        .with_memory_active_profile(memory_profile)
        .with_policy_active_profile(policy_profile),
        memory_embedder,
        policy_embedder,
    )
    .await
    .context("connect libsql agent store")
}

async fn seed_memories(
    memory_caps: &MemoryCapabilities,
    blackboard: &Blackboard,
    clock: &dyn Clock,
    case_now: Option<DateTime<FixedOffset>>,
    memories: &[crate::cases::MemorySeed],
) -> Result<Vec<MemoryIndex>> {
    let mut seeded = Vec::with_capacity(memories.len());
    for memory in memories {
        let scope = parse_scope_id(&memory.scope)?;
        let scoped_caps = if scope.is_root() {
            memory_caps.with_namespace(MemoryNamespace::Global)
        } else {
            memory_caps
                .with_namespace(MemoryNamespace::Local(scope.clone()))
                .scoped(blackboard.scoped(scope))
        };
        let writer = scoped_caps.writer();
        let occurred_at = memory_seed_occurred_at(clock, case_now, memory)?;
        let index = if let Some(index) = memory.index.as_deref() {
            writer
                .put_seeded_with_occurred_at(
                    MemoryIndex::new(index),
                    memory.content.content.clone(),
                    MemoryRank::from(memory.rank),
                    memory.decay_secs,
                    occurred_at,
                )
                .await
                .context("seed eval memory with explicit index")?
        } else {
            writer
                .insert_with_occurred_at(
                    memory.content.content.clone(),
                    MemoryRank::from(memory.rank),
                    memory.decay_secs,
                    occurred_at,
                )
                .await
                .context("seed eval memory")?
        };
        seeded.push(index);
    }
    Ok(seeded)
}

async fn seed_memory_links(
    memory: &dyn MemoryStore,
    clock: &dyn Clock,
    seeded_indexes: &[MemoryIndex],
    links: &[MemoryLinkSeed],
) -> Result<()> {
    for link in links {
        let from = seeded_indexes
            .get(link.from_memory)
            .with_context(|| format!("resolve memory-links from-memory {}", link.from_memory))?;
        let to = seeded_indexes
            .get(link.to_memory)
            .with_context(|| format!("resolve memory-links to-memory {}", link.to_memory))?;
        let relation = parse_memory_relation(&link.relation)
            .with_context(|| format!("parse memory link relation {}", link.relation))?;
        memory
            .upsert_link(
                NewMemoryLink {
                    from_memory: from.clone(),
                    to_memory: to.clone(),
                    relation,
                    freeform_relation: None,
                    strength: 1.0,
                    confidence: 1.0,
                },
                clock.now(),
            )
            .await
            .context("seed eval memory link")?;
    }
    Ok(())
}

async fn seed_policies(
    policy_store: &dyn PolicyStore,
    blackboard: &Blackboard,
    policies: &[PolicySeed],
) -> Result<()> {
    for policy in policies {
        let record = PolicyRecord {
            index: PolicyIndex::new(policy.index.clone()),
            trigger: policy.trigger.content.clone(),
            behavior: policy.behavior.content.clone(),
            rank: PolicyRank::from(policy.rank),
            expected_reward: SignedUnitF32::clamp(0.5),
            confidence: UnitF32::clamp(0.7),
            value: SignedUnitF32::clamp(0.5),
            reward_tokens: 0,
            decay_remaining_secs: policy.decay_secs,
        };
        policy_store
            .put(IndexedPolicy {
                index: record.index.clone(),
                trigger: record.trigger.clone(),
                behavior: record.behavior.clone(),
                rank: record.rank,
                expected_reward: record.expected_reward,
                confidence: record.confidence,
                value: record.value,
                reward_tokens: record.reward_tokens,
                decay_remaining_secs: record.decay_remaining_secs,
            })
            .await
            .context("seed eval policy")?;
        blackboard
            .apply(BlackboardCommand::UpsertPolicyMetadata {
                index: record.index.clone(),
                rank_if_new: record.rank,
                decay_if_new_secs: record.decay_remaining_secs,
                patch: PolicyMetaPatch {
                    rank: Some(record.rank),
                    expected_reward: Some(record.expected_reward),
                    confidence: Some(record.confidence),
                    value: Some(record.value),
                    reward_tokens: Some(record.reward_tokens),
                    decay_remaining_secs: Some(record.decay_remaining_secs),
                    ..Default::default()
                },
            })
            .await;
    }
    Ok(())
}

async fn seed_and_bootstrap_eval_startup_context(
    memory_caps: &MemoryCapabilities,
    policy_caps: &PolicyCapabilities,
    memory: &dyn MemoryStore,
    policy_store: &dyn PolicyStore,
    blackboard: &Blackboard,
    clock: &dyn Clock,
    case_now: Option<DateTime<FixedOffset>>,
    memories: &[crate::cases::MemorySeed],
    memory_links: &[MemoryLinkSeed],
    policies: &[PolicySeed],
) -> Result<()> {
    let seeded_indexes = seed_memories(memory_caps, blackboard, clock, case_now, memories).await?;
    seed_memory_links(memory, clock, &seeded_indexes, memory_links).await?;
    seed_policies(policy_store, blackboard, policies).await?;
    memory_caps
        .bootstrap_identity_memories()
        .await
        .map_err(|err| anyhow::anyhow!("failed to load identity memories: {err}"))?;
    policy_caps
        .bootstrap_core_policies()
        .await
        .map_err(|err| anyhow::anyhow!("failed to load core policies: {err}"))?;
    Ok(())
}

fn memory_seed_occurred_at(
    clock: &dyn Clock,
    case_now: Option<DateTime<FixedOffset>>,
    memory: &crate::cases::MemorySeed,
) -> Result<Option<DateTime<Utc>>> {
    if let Some(datetime) = &memory.datetime {
        return parse_memory_datetime(datetime, case_now)
            .map(Some)
            .map_err(anyhow::Error::msg)
            .with_context(|| format!("parse memory datetime {datetime}"));
    }
    if let Some(seconds_ago) = memory.seconds_ago {
        return Ok(Some(clock.now() - ChronoDuration::seconds(seconds_ago)));
    }
    Ok(None)
}

async fn seed_memos(
    blackboard: &Blackboard,
    clock: &dyn Clock,
    memos: &[crate::cases::MemoSeed],
    force_cognitive: bool,
) -> Result<Vec<nuillu_blackboard::MemoLogRecord>> {
    let now = clock.now();
    let mut records = Vec::new();
    for memo in memos {
        let module = ModuleId::new(memo.module.clone())
            .with_context(|| format!("seed memo module id {}", memo.module))?;
        let owner = ModuleInstanceId::in_scope(
            parse_scope_id(&memo.scope)?,
            module,
            ReplicaIndex::new(memo.replica),
        );
        let written_at = now - ChronoDuration::seconds(memo.seconds_ago);
        let cognitive = force_cognitive || memo.cognitive;
        let record = if cognitive {
            blackboard
                .update_cognitive_memo(owner, memo.content.content.clone(), written_at)
                .await
        } else {
            blackboard
                .update_memo(owner, memo.content.content.clone(), written_at)
                .await
        };
        records.push(record);
    }
    Ok(records)
}

async fn seed_cognition_log(
    blackboard: &Blackboard,
    clock: &dyn Clock,
    seeds: &[crate::cases::CognitionLogSeed],
) -> Vec<CognitionLogEntryRecord> {
    let now = clock.now();
    let mut records = Vec::with_capacity(seeds.len());
    for seed in seeds {
        let stream = ModuleInstanceId::in_scope(
            parse_scope_id(&seed.scope).expect("validated cognition scope"),
            ModuleId::new(seed.module.clone()).expect("validated cognition module"),
            ReplicaIndex::new(seed.replica),
        );
        let appended = blackboard
            .append_cognition_log(
                stream.clone(),
                CognitionLogEntry {
                    at: now - ChronoDuration::seconds(seed.seconds_ago),
                    text: seed.text.content.clone(),
                    origin: CognitionLogOrigin::direct(stream.clone()),
                },
            )
            .await;
        records.push(appended.record);
    }
    records
}

async fn publish_setup_updates(
    harness: &InternalHarnessIo,
    memos: &[MemoLogRecord],
    cognition: &[CognitionLogEntryRecord],
) {
    for record in memos {
        harness
            .memo_updated_mailbox()
            .publish(nuillu_module::MemoUpdated {
                owner: record.owner.clone(),
                index: record.index,
            })
            .await
            .expect("eval setup failed to publish MemoUpdated");
    }
    for record in cognition {
        harness
            .cognition_log_updated_mailbox()
            .publish(CognitionLogUpdated::EntryAppended {
                source: record.source.clone(),
            })
            .await
            .expect("eval setup failed to publish CognitionLogUpdated");
    }
    if !memos.is_empty() || !cognition.is_empty() {
        harness
            .interoception_updated_mailbox()
            .publish(nuillu_module::InteroceptiveUpdated)
            .await
            .expect("eval setup failed to publish InteroceptiveUpdated");
    }
}

async fn add_observations(
    artifact: &mut CaseArtifact,
    blackboard: &Blackboard,
    utterances: &RecordingUtteranceSink,
) {
    let observations = blackboard
        .read(|bb| AgentObservation::from_blackboard(bb, utterances.snapshot()))
        .await;
    let observations = match serde_json::to_value(observations) {
        Ok(value) => value,
        Err(error) => serde_json::json!({
            "serialization_error": error.to_string(),
        }),
    };
    artifact
        .observations
        .insert("agent".to_string(), observations);
}

async fn build_runtime_last_state_dump(
    case_id: &str,
    artifact: &CaseArtifact,
    blackboard: &Blackboard,
    memory: &dyn MemoryStore,
    utterances: &RecordingUtteranceSink,
    event_count: usize,
) -> Result<RuntimeLastStateDump> {
    let (blackboard_dump, memory_metadata) = blackboard
        .read(|bb| {
            (
                blackboard_last_state_dump(bb),
                memory_metadata_dump_records(bb),
            )
        })
        .await;
    let memory_dump = memory_last_state_dump(memory_metadata, memory).await?;
    Ok(RuntimeLastStateDump {
        case: RuntimeLastStateCaseDump {
            id: case_id.to_string(),
            dumped_at: Utc::now().to_rfc3339(),
            event_count: event_count as u64,
            output: (!artifact.output.is_empty()).then(|| DumpText::new(artifact.output.clone())),
            failure: artifact.failure.clone().map(DumpText::new),
        },
        blackboard: blackboard_dump,
        memory: memory_dump,
        utterances: utterance_dumps(utterances.snapshot()),
    })
}

fn add_last_state_observation(
    artifact: &mut CaseArtifact,
    last_state: &RuntimeLastStateDump,
) -> Result<()> {
    let value = serde_json::to_value(last_state).context("serialize last state observation")?;
    artifact
        .observations
        .insert("last_state".to_string(), value);
    Ok(())
}

fn write_runtime_last_state_eure(
    output_dir: &Path,
    last_state: RuntimeLastStateDump,
) -> Result<()> {
    let path = output_dir.join("last-state.eure");
    let rendered =
        render_runtime_last_state_eure(last_state).context("render runtime last state Eure")?;
    std::fs::write(&path, rendered)
        .with_context(|| format!("write runtime last state dump to {}", path.display()))
}

fn blackboard_last_state_dump(bb: &BlackboardInner) -> BlackboardLastStateDump {
    let cognition_log_set = bb.cognition_log_set();
    BlackboardLastStateDump {
        memo_logs: memo_log_dumps(bb),
        cognition_logs: cognition_log_set
            .logs()
            .iter()
            .map(|record| CognitionLogDump {
                source: module_instance_dump(&record.source),
                entries: record
                    .entries
                    .iter()
                    .map(|event| CognitionEntryDump {
                        at: event.at.to_rfc3339(),
                        text: DumpText::new(event.text.clone()),
                    })
                    .collect(),
            })
            .collect(),
        interoception: interoception_dump(bb),
        agentic_deadlock: cognition_log_set.agentic_deadlock_marker().map(|marker| {
            AgenticDeadlockDump {
                at: marker.at.to_rfc3339(),
                idle_for_ms: duration_millis_u64(marker.idle_for),
            }
        }),
        base_allocation: allocation_module_dumps(bb.base_allocation()),
        allocation: allocation_module_dumps(bb.allocation()),
        allocation_proposals: allocation_proposal_dumps(bb),
        replica_caps: replica_cap_dumps(bb),
    }
}

fn visualizer_blackboard_snapshot(bb: &BlackboardInner) -> BlackboardSnapshot {
    let cognition_log_set = bb.cognition_log_set();
    let mut memory_metadata = bb
        .memory_metadata()
        .iter()
        .map(|(index, metadata)| MemoryMetadataView {
            index: index.as_str().to_owned(),
            rank: memory_rank_name(metadata.rank).to_owned(),
            occurred_at: metadata.occurred_at,
            last_accessed: metadata.last_accessed,
            access_count: metadata.access_count,
            use_count: metadata.use_count,
            reinforcement_count: metadata.reinforcement_count,
        })
        .collect::<Vec<_>>();
    memory_metadata.sort_by(|left, right| left.index.cmp(&right.index));

    BlackboardSnapshot {
        scopes: Vec::new(),
        module_statuses: bb
            .module_status_records()
            .into_iter()
            .map(|record| ModuleStatusView {
                owner: record.owner.to_string(),
                module: record.owner.module.as_str().to_owned(),
                replica: record.owner.replica.get(),
                status: format!("{:?}", record.status),
            })
            .collect(),
        allocation: allocation_module_dumps(bb.allocation())
            .into_iter()
            .map(|module| {
                let bpm = ModuleId::new(module.module.clone())
                    .ok()
                    .and_then(|id| bb.allocation().bpm_for(&id));
                AllocationView {
                    scope: "/".to_string(),
                    bpm: bpm.map(|bpm| bpm.as_f64()),
                    period_ms: bpm.map(|bpm| duration_millis_u64(bpm.period())),
                    module: module.module,
                    activation_ratio: module.activation_ratio,
                    scope_activation_ratio: 1.0,
                    effective_activation_ratio: module.activation_ratio,
                    active_replicas: module.active_replicas,
                }
            })
            .collect(),
        interoception: interoception_view(bb.interoception()),
        module_policies: module_policy_views(bb),
        forced_disabled_modules: {
            let mut modules = bb
                .forced_disabled_modules()
                .iter()
                .map(|module| module.as_str().to_owned())
                .collect::<Vec<_>>();
            modules.sort();
            modules
        },
        memos: bb
            .recent_memo_logs()
            .into_iter()
            .map(|record| MemoView {
                owner: record.owner.to_string(),
                module: record.owner.module.as_str().to_owned(),
                replica: record.owner.replica.get(),
                index: record.index,
                written_at: record.written_at,
                cognitive: record.cognitive,
                content: record.content,
            })
            .collect(),
        cognition_logs: cognition_log_set
            .logs()
            .iter()
            .map(|record| CognitionLogView {
                source: record.source.to_string(),
                entries: record
                    .entries
                    .iter()
                    .map(|entry| CognitionEntryView {
                        at: entry.at,
                        origin: entry.origin.owner.to_string(),
                        text: entry.text.clone(),
                    })
                    .collect(),
            })
            .collect(),
        utterance_progresses: bb
            .utterance_progress_records()
            .into_iter()
            .map(|record| UtteranceProgressView {
                owner: record.owner.to_string(),
                target: record.progress.target,
                generation_id: record.progress.generation_id,
                sequence: record.progress.sequence,
                state: format!("{:?}", record.progress.state),
                partial_utterance: record.progress.partial_utterance,
            })
            .collect(),
        memory_metadata,
    }
}

fn interoception_view(state: &nuillu_blackboard::InteroceptiveState) -> InteroceptionView {
    InteroceptionView {
        mode: interoceptive_mode_name(state.mode).to_owned(),
        wake_arousal: state.wake_arousal,
        nrem_pressure: state.nrem_pressure,
        rem_pressure: state.rem_pressure,
        affect_arousal: state.affect_arousal,
        valence: state.valence,
        emotion: state.emotion.clone(),
        last_updated: state.last_updated,
    }
}

fn memo_log_dumps(bb: &BlackboardInner) -> Vec<MemoLogDump> {
    bb.recent_memo_logs()
        .into_iter()
        .map(|record| MemoLogDump {
            module: record.owner.module.as_str().to_owned(),
            replica: record.owner.replica.get(),
            index: record.index,
            written_at: record.written_at.to_rfc3339(),
            cognitive: record.cognitive,
            content: DumpText::new(record.content),
        })
        .collect()
}

fn allocation_module_dumps(allocation: &ResourceAllocation) -> Vec<AllocationModuleDump> {
    let mut modules = allocation
        .module_ids()
        .into_iter()
        .map(|module| AllocationModuleDump {
            module: module.as_str().to_owned(),
            activation_ratio: allocation.activation_for(&module).as_f64(),
            active_replicas: allocation.active_replicas(&module),
            period_ms: allocation
                .bpm_for(&module)
                .map(|bpm| duration_millis_u64(bpm.period())),
        })
        .collect::<Vec<_>>();
    modules.sort_by(|left, right| left.module.cmp(&right.module));
    modules
}

fn allocation_proposal_dumps(bb: &BlackboardInner) -> Vec<AllocationProposalDump> {
    let mut proposals = bb
        .allocation_proposals()
        .iter()
        .map(|(controller, proposal)| AllocationProposalDump {
            controller: module_instance_dump(controller),
            modules: allocation_module_dumps(proposal),
        })
        .collect::<Vec<_>>();
    proposals.sort_by(|left, right| {
        left.controller
            .module
            .cmp(&right.controller.module)
            .then_with(|| left.controller.replica.cmp(&right.controller.replica))
    });
    proposals
}

fn replica_cap_dumps(bb: &BlackboardInner) -> Vec<ReplicaCapDump> {
    let mut caps = bb
        .module_policies()
        .iter()
        .map(|(module, policy)| ReplicaCapDump {
            module: module.as_str().to_owned(),
            min: policy.replicas_range.min,
            max: policy.replicas_range.max,
        })
        .collect::<Vec<_>>();
    caps.sort_by(|left, right| left.module.cmp(&right.module));
    caps
}

fn memory_metadata_dump_records(bb: &BlackboardInner) -> Vec<(String, MemoryMetadataDump)> {
    let mut records = bb
        .memory_metadata()
        .iter()
        .map(|(index, metadata)| {
            (
                index.as_str().to_owned(),
                MemoryMetadataDump {
                    rank: memory_rank_name(metadata.rank).to_owned(),
                    occurred_at: metadata.occurred_at.map(|at| at.to_rfc3339()),
                    decay_remaining_secs: metadata.decay_remaining_secs,
                    remember_tokens: metadata.remember_tokens,
                    last_accessed: metadata.last_accessed.to_rfc3339(),
                    access_count: metadata.access_count,
                    use_count: metadata.use_count,
                    last_used: metadata.last_used.map(|at| at.to_rfc3339()),
                    reinforcement_count: metadata.reinforcement_count,
                    last_reinforced_at: metadata.last_reinforced_at.map(|at| at.to_rfc3339()),
                    query_history: metadata
                        .query_history
                        .iter()
                        .map(|at| at.to_rfc3339())
                        .collect(),
                    use_history: metadata
                        .use_history
                        .iter()
                        .map(|at| at.to_rfc3339())
                        .collect(),
                    reinforcement_history: metadata
                        .reinforcement_history
                        .iter()
                        .map(|at| at.to_rfc3339())
                        .collect(),
                },
            )
        })
        .collect::<Vec<_>>();
    records.sort_by(|left, right| left.0.cmp(&right.0));
    records
}

async fn memory_last_state_dump(
    metadata_records: Vec<(String, MemoryMetadataDump)>,
    memory: &dyn MemoryStore,
) -> Result<MemoryLastStateDump> {
    let mut entries = Vec::with_capacity(metadata_records.len());
    for (index, metadata) in metadata_records {
        let memory_index = nuillu_types::MemoryIndex::new(index.clone());
        let record = memory
            .get(&memory_index)
            .await
            .with_context(|| format!("read memory content for {index}"))?;
        entries.push(match record {
            Some(record) => MemoryEntryDump {
                index,
                content: Some(DumpText::new(record.content.as_str().to_owned())),
                content_rank: Some(memory_rank_name(record.rank).to_owned()),
                occurred_at: record.occurred_at.map(|at| at.to_rfc3339()),
                affect_arousal: record.affect_arousal,
                valence: record.valence,
                emotion: record.emotion,
                metadata,
                missing_content: false,
            },
            None => MemoryEntryDump {
                index,
                content: None,
                content_rank: None,
                occurred_at: None,
                affect_arousal: 0.0,
                valence: 0.0,
                emotion: String::new(),
                metadata,
                missing_content: true,
            },
        });
    }
    Ok(MemoryLastStateDump { entries })
}

fn utterance_dumps(utterances: Vec<RecordedUtterance>) -> Vec<UtteranceDump> {
    utterances
        .into_iter()
        .map(|utterance| UtteranceDump {
            sender: utterance.sender,
            target: utterance.target,
            text: DumpText::new(utterance.text),
            emitted_at: utterance.emitted_at,
        })
        .collect()
}

#[derive(Debug, Clone, Serialize)]
struct AgentObservation {
    memo_logs: BTreeMap<String, Vec<MemoLogObservation>>,
    cognition_logs: Vec<CognitionLogObservation>,
    interoception: nuillu_blackboard::InteroceptiveState,
    allocation: BTreeMap<String, AllocationModuleObservation>,
    allocation_proposals: BTreeMap<String, BTreeMap<String, AllocationModuleObservation>>,
    replica_caps: BTreeMap<String, ReplicaCapRange>,
    memory_metadata: BTreeMap<String, MemoryMetadata>,
    utterances: Vec<RecordedUtterance>,
}

impl AgentObservation {
    fn from_blackboard(bb: &BlackboardInner, utterances: Vec<RecordedUtterance>) -> Self {
        Self {
            memo_logs: memo_log_observations(bb),
            cognition_logs: cognition_log_observations(bb),
            interoception: bb.interoception().clone(),
            allocation: allocation_observation(bb.allocation()),
            allocation_proposals: allocation_proposal_observations(bb),
            replica_caps: replica_cap_observations(bb),
            memory_metadata: memory_metadata_observations(bb),
            utterances,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
struct AllocationModuleObservation {
    activation_ratio: ActivationRatio,
    active_replicas: u8,
    period_ms: Option<u64>,
}

#[derive(Debug, Clone, Serialize)]
struct MemoLogObservation {
    scope: String,
    replica: u8,
    index: u64,
    written_at: String,
    cognitive: bool,
    content: String,
}

#[derive(Debug, Clone, Serialize)]
struct CognitionLogObservation {
    source: ModuleInstanceObservation,
    entries: Vec<CognitionLogEntry>,
}

#[derive(Debug, Clone, Serialize)]
struct ModuleInstanceObservation {
    scope: String,
    module: String,
    replica: u8,
}

#[derive(Debug, Clone, Serialize)]
struct ActiveModuleObservation {
    module: String,
    active_replicas: u8,
    activation_ratio: ActivationRatio,
}

fn memo_log_observations(bb: &BlackboardInner) -> BTreeMap<String, Vec<MemoLogObservation>> {
    let mut logs = BTreeMap::<String, Vec<MemoLogObservation>>::new();
    for record in bb.recent_memo_logs() {
        logs.entry(record.owner.to_string())
            .or_default()
            .push(MemoLogObservation {
                scope: record.owner.scope.to_string(),
                replica: record.owner.replica.get(),
                index: record.index,
                written_at: record.written_at.to_rfc3339(),
                cognitive: record.cognitive,
                content: record.content,
            });
    }
    logs
}

fn cognition_log_observations(bb: &BlackboardInner) -> Vec<CognitionLogObservation> {
    bb.cognition_log_set()
        .logs()
        .iter()
        .map(|record| CognitionLogObservation {
            source: module_instance_observation(&record.source),
            entries: record.entries.clone(),
        })
        .collect()
}

fn interoception_dump(bb: &BlackboardInner) -> InteroceptionDump {
    let state = bb.interoception();
    InteroceptionDump {
        mode: interoceptive_mode_name(state.mode).to_owned(),
        wake_arousal: state.wake_arousal,
        nrem_pressure: state.nrem_pressure,
        rem_pressure: state.rem_pressure,
        affect_arousal: state.affect_arousal,
        valence: state.valence,
        emotion: state.emotion.clone(),
        last_updated: state.last_updated.to_rfc3339(),
    }
}

fn interoceptive_mode_name(mode: nuillu_blackboard::InteroceptiveMode) -> &'static str {
    match mode {
        nuillu_blackboard::InteroceptiveMode::Wake => "wake",
        nuillu_blackboard::InteroceptiveMode::NremPressure => "nrem-pressure",
        nuillu_blackboard::InteroceptiveMode::RemPressure => "rem-pressure",
    }
}

fn allocation_observation(
    allocation: &ResourceAllocation,
) -> BTreeMap<String, AllocationModuleObservation> {
    allocation
        .module_ids()
        .into_iter()
        .map(|module| {
            (
                module.as_str().to_owned(),
                AllocationModuleObservation {
                    activation_ratio: allocation.activation_for(&module),
                    active_replicas: allocation.active_replicas(&module),
                    period_ms: allocation
                        .bpm_for(&module)
                        .map(|bpm| duration_millis_u64(bpm.period())),
                },
            )
        })
        .collect()
}

fn allocation_proposal_observations(
    bb: &BlackboardInner,
) -> BTreeMap<String, BTreeMap<String, AllocationModuleObservation>> {
    bb.allocation_proposals()
        .iter()
        .map(|(owner, allocation)| (owner.to_string(), allocation_observation(allocation)))
        .collect()
}

fn replica_cap_observations(bb: &BlackboardInner) -> BTreeMap<String, ReplicaCapRange> {
    bb.module_policies()
        .iter()
        .map(|(module, policy)| (module.as_str().to_owned(), policy.replicas_range))
        .collect()
}

fn active_module_observations(bb: &BlackboardInner) -> Vec<ActiveModuleObservation> {
    let mut modules = bb
        .module_policies()
        .keys()
        .cloned()
        .chain(bb.allocation().module_ids())
        .collect::<Vec<_>>();
    modules.sort_by(|left, right| left.as_str().cmp(right.as_str()));
    modules.dedup();
    modules
        .into_iter()
        .filter_map(|module| {
            let active_replicas = bb.allocation().active_replicas(&module);
            if active_replicas == 0 {
                return None;
            }
            Some(ActiveModuleObservation {
                module: module.as_str().to_owned(),
                active_replicas,
                activation_ratio: bb.allocation().activation_for(&module),
            })
        })
        .collect()
}

fn memory_metadata_observations(bb: &BlackboardInner) -> BTreeMap<String, MemoryMetadata> {
    bb.memory_metadata()
        .iter()
        .map(|(index, metadata)| (index.as_str().to_owned(), metadata.clone()))
        .collect()
}

fn module_instance_observation(owner: &ModuleInstanceId) -> ModuleInstanceObservation {
    ModuleInstanceObservation {
        scope: owner.scope.to_string(),
        module: owner.module.as_str().to_owned(),
        replica: owner.replica.get(),
    }
}

fn module_instance_dump(owner: &ModuleInstanceId) -> ModuleInstanceDump {
    ModuleInstanceDump {
        module: owner.module.as_str().to_owned(),
        replica: owner.replica.get(),
    }
}

struct AllocationChangeReporter {
    case_id: String,
    reporter: LiveReporter,
    last: Option<String>,
}

impl AllocationChangeReporter {
    fn new(case_id: String, reporter: LiveReporter) -> Self {
        Self {
            case_id,
            reporter,
            last: None,
        }
    }

    async fn emit_if_changed(&mut self, blackboard: &Blackboard) -> Result<(), RunnerError> {
        let allocation = blackboard
            .read(|bb| allocation_observation(bb.allocation()))
            .await;
        let value = serde_json::to_value(&allocation).map_err(|error| RunnerError::Driver {
            path: PathBuf::from(&self.case_id),
            message: error.to_string(),
        })?;
        let signature = serde_json::to_string(&value).map_err(|error| RunnerError::Driver {
            path: PathBuf::from(&self.case_id),
            message: error.to_string(),
        })?;
        if self.last.as_deref() == Some(signature.as_str()) {
            return Ok(());
        }
        self.last = Some(signature);
        let live = format!(
            "eval allocation case={} {}",
            self.case_id,
            allocation_live_summary(&allocation)
        );
        self.reporter.emit(
            Some(&self.case_id),
            "allocation_changed",
            serde_json::json!({ "allocation": value }),
            live,
        )
    }
}

fn allocation_live_summary(allocation: &BTreeMap<String, AllocationModuleObservation>) -> String {
    let active = allocation
        .iter()
        .filter(|(_, obs)| obs.activation_ratio > ActivationRatio::ZERO)
        .map(|(module, obs)| format!("{}:{:.2}", module, obs.activation_ratio.as_f64()))
        .collect::<Vec<_>>();
    let inactive = allocation
        .values()
        .filter(|obs| obs.activation_ratio == ActivationRatio::ZERO)
        .count();
    format!("active=[{}] inactive={inactive}", active.join(","))
}

fn active_modules_live_summary(active_modules: &[ActiveModuleObservation]) -> String {
    if active_modules.is_empty() {
        return "none".to_owned();
    }
    active_modules
        .iter()
        .map(|module| {
            format!(
                "{}:{}:{:.2}",
                module.module,
                module.active_replicas,
                module.activation_ratio.as_f64()
            )
        })
        .collect::<Vec<_>>()
        .join(",")
}

fn idle_timeout_message(
    seconds: u64,
    events: &[RuntimeEvent],
    active_modules: &[ActiveModuleObservation],
) -> String {
    let last_event = events
        .last()
        .map(runtime_event_summary)
        .unwrap_or_else(|| "none".to_owned());
    let last_llm = events
        .iter()
        .rev()
        .find_map(|event| match event {
            RuntimeEvent::LlmSemaphoreWaitStarted {
                sequence,
                owner,
                tier,
            } => Some(format!(
                "seq={sequence} semaphore_wait owner={owner} tier={tier:?}"
            )),
            RuntimeEvent::LlmAccessed {
                sequence,
                call,
                owner,
                tier,
            } => Some(format!(
                "seq={sequence} call={call} owner={owner} tier={tier:?}"
            )),
            _ => None,
        })
        .unwrap_or_else(|| "none".to_owned());
    format!(
        "no runtime progress for {seconds}s; agent appears stuck; last_event={last_event}; last_llm={last_llm}; active=[{}]",
        active_modules_live_summary(active_modules)
    )
}

fn runtime_event_summary(event: &RuntimeEvent) -> String {
    match event {
        RuntimeEvent::LlmSemaphoreWaitStarted {
            sequence,
            owner,
            tier,
        } => format!("seq={sequence} llm_semaphore_wait owner={owner} tier={tier:?}"),
        RuntimeEvent::LlmAccessed {
            sequence,
            call,
            owner,
            tier,
        } => format!("seq={sequence} llm_accessed call={call} owner={owner} tier={tier:?}"),
        RuntimeEvent::LlmCompleted {
            sequence,
            call,
            owner,
            tier,
        } => format!("seq={sequence} llm_completed call={call} owner={owner} tier={tier:?}"),
        RuntimeEvent::MemoUpdated {
            sequence,
            owner,
            char_count,
        } => format!("seq={sequence} memo_updated owner={owner} chars={char_count}"),
        RuntimeEvent::ModuleBatchThrottled {
            sequence,
            owner,
            delayed_for,
        } => format!(
            "seq={sequence} module_batch_throttled owner={owner} delayed_for_ms={}",
            duration_millis_u64(*delayed_for)
        ),
        RuntimeEvent::ModuleBatchReady {
            sequence,
            activation_id,
            owner,
            batch_type,
            ..
        } => format!(
            "seq={sequence} module_batch_ready activation={} owner={owner} batch={batch_type}",
            activation_id
        ),
        RuntimeEvent::ModuleActivationCompleted {
            sequence,
            activation_id,
            owner,
            duration,
            succeeded,
            ..
        } => format!(
            "seq={sequence} module_activation_completed activation={} owner={owner} duration_ms={} succeeded={succeeded}",
            activation_id,
            duration_millis_u64(*duration)
        ),
        RuntimeEvent::ModuleActivationAttemptFailed {
            sequence,
            activation_id,
            owner,
            activation_attempt,
            max_attempts,
            message,
            ..
        } => format!(
            "seq={sequence} module_activation_attempt_failed activation={} owner={owner} attempt={activation_attempt}/{max_attempts} message={message}",
            activation_id
        ),
        RuntimeEvent::ModuleTaskFailed {
            sequence,
            owner,
            phase,
            message,
        } => format!(
            "seq={sequence} module_task_failed owner={owner} phase={phase} message={message}"
        ),
        RuntimeEvent::ModuleWarning {
            sequence,
            owner,
            message,
        } => format!("seq={sequence} module_warning owner={owner} message={message}"),
        RuntimeEvent::SessionCompactionStarted {
            sequence,
            owner,
            session_key,
            input_tokens,
            threshold,
            tier,
        } => format!(
            "seq={sequence} session_compaction_started owner={owner} session={session_key} input_tokens={input_tokens} threshold={threshold} tier={tier:?}"
        ),
        RuntimeEvent::SessionCompactionCompleted {
            sequence,
            owner,
            session_key,
            input_tokens,
            before_items,
            after_items,
            tier,
            ..
        } => format!(
            "seq={sequence} session_compaction_completed owner={owner} session={session_key} input_tokens={input_tokens} items={before_items}->{after_items} tier={tier:?}"
        ),
        RuntimeEvent::SessionCompactionFailed {
            sequence,
            owner,
            session_key,
            input_tokens,
            message,
            tier,
            ..
        } => format!(
            "seq={sequence} session_compaction_failed owner={owner} session={session_key} input_tokens={input_tokens} tier={tier:?} message={message}"
        ),
    }
}

fn ticks_for_interval(interval: Duration, tick_ms: u64) -> u64 {
    (duration_millis_u64(interval) / tick_ms.max(1)).max(1)
}

#[derive(Clone)]
pub(crate) struct LiveReporter {
    run_id: String,
    path: PathBuf,
    file: Arc<Mutex<File>>,
    log_prefix: String,
    log_scope: String,
    live_output: LiveOutput,
}

impl std::fmt::Debug for LiveReporter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LiveReporter")
            .field("run_id", &self.run_id)
            .field("path", &self.path)
            .field("log_prefix", &self.log_prefix)
            .field("log_scope", &self.log_scope)
            .field("live_output", &self.live_output)
            .finish_non_exhaustive()
    }
}

impl LiveReporter {
    pub(crate) fn new(
        run_id: &str,
        run_dir: &Path,
        live_output: LiveOutput,
    ) -> Result<Self, RunnerError> {
        Self::new_with_log_context(run_id, run_dir, "eval", "case", live_output)
    }

    pub(crate) fn new_with_log_context(
        run_id: &str,
        run_dir: &Path,
        log_prefix: &str,
        log_scope: &str,
        live_output: LiveOutput,
    ) -> Result<Self, RunnerError> {
        std::fs::create_dir_all(run_dir).map_err(|source| RunnerError::WriteOutput {
            path: run_dir.to_path_buf(),
            source,
        })?;
        let path = run_dir.join("events.jsonl");
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .map_err(|source| RunnerError::WriteOutput { path, source })?;
        Ok(Self {
            run_id: run_id.to_string(),
            path: run_dir.join("events.jsonl"),
            file: Arc::new(Mutex::new(file)),
            log_prefix: log_prefix.to_string(),
            log_scope: log_scope.to_string(),
            live_output,
        })
    }

    fn log_prefix(&self) -> &str {
        &self.log_prefix
    }

    fn log_scope(&self, value: &str) -> String {
        format!("{}={value}", self.log_scope)
    }

    fn emit(
        &self,
        case_id: Option<&str>,
        kind: &str,
        data: serde_json::Value,
        live_message: String,
    ) -> Result<(), RunnerError> {
        self.emit_jsonl(case_id, kind, data, live_message)
            .map_err(|source| RunnerError::WriteOutput {
                path: self.path.clone(),
                source,
            })
    }

    fn emit_port(
        &self,
        case_id: Option<&str>,
        kind: &str,
        data: serde_json::Value,
        live_message: String,
    ) -> Result<(), PortError> {
        self.emit_jsonl(case_id, kind, data, live_message)
            .map_err(|error| {
                PortError::Backend(format!("write {} event: {error}", self.log_prefix))
            })
    }

    fn emit_jsonl(
        &self,
        case_id: Option<&str>,
        kind: &str,
        data: serde_json::Value,
        live_message: String,
    ) -> io::Result<()> {
        let minimum = live_output_minimum(kind);
        self.emit_jsonl_at(case_id, kind, data, live_message, minimum)
    }

    fn emit_port_at(
        &self,
        case_id: Option<&str>,
        kind: &str,
        data: serde_json::Value,
        live_message: String,
        minimum: LiveOutput,
    ) -> Result<(), PortError> {
        self.emit_jsonl_at(case_id, kind, data, live_message, minimum)
            .map_err(|error| {
                PortError::Backend(format!("write {} event: {error}", self.log_prefix))
            })
    }

    fn emit_jsonl_at(
        &self,
        case_id: Option<&str>,
        kind: &str,
        data: serde_json::Value,
        live_message: String,
        minimum: LiveOutput,
    ) -> io::Result<()> {
        if self.live_output >= minimum {
            eprintln!("{live_message}");
        }
        let record = serde_json::json!({
            "ts": Utc::now().to_rfc3339(),
            "run_id": self.run_id,
            "case_id": case_id,
            "kind": kind,
            "data": data,
        });
        let mut file = self
            .file
            .lock()
            .map_err(|_| io::Error::other("events.jsonl lock poisoned"))?;
        serde_json::to_writer(&mut *file, &record).map_err(io::Error::other)?;
        file.write_all(b"\n")?;
        file.flush()
    }
}

fn live_output_minimum(kind: &str) -> LiveOutput {
    match kind {
        "runtime_event" | "allocation_changed" => LiveOutput::Verbose,
        _ => LiveOutput::Normal,
    }
}

#[cfg(test)]
mod live_output_tests {
    use super::*;

    #[test]
    fn normal_output_suppresses_high_volume_events() {
        assert_eq!(live_output_minimum("runtime_event"), LiveOutput::Verbose);
        assert_eq!(
            live_output_minimum("allocation_changed"),
            LiveOutput::Verbose
        );
        assert_eq!(live_output_minimum("case_started"), LiveOutput::Normal);
        assert_eq!(live_output_minimum("step_finished"), LiveOutput::Normal);
        assert_eq!(
            live_output_minimum("utterance_completed"),
            LiveOutput::Normal
        );
    }

    #[test]
    fn runtime_failures_remain_notable() {
        let owner = ModuleInstanceId::new(ModuleId::new("worker").unwrap(), ReplicaIndex::ZERO);
        assert!(runtime_event_is_notable(&RuntimeEvent::ModuleTaskFailed {
            sequence: 1,
            owner,
            phase: "activate".to_string(),
            message: "failed".to_string(),
        }));
    }
}

#[derive(Debug, Clone, Serialize)]
struct RecordedUtterance {
    sender: String,
    scope: String,
    module: String,
    replica: u8,
    target: String,
    text: String,
    emitted_at: String,
}

#[derive(Clone)]
pub(crate) struct ActionActivityTracker {
    action_modules: Arc<HashSet<ModuleId>>,
    last_completed_at: Arc<Mutex<Option<Instant>>>,
}

impl ActionActivityTracker {
    fn new(action_modules: Vec<ModuleId>) -> Self {
        Self {
            action_modules: Arc::new(action_modules.into_iter().collect()),
            last_completed_at: Arc::new(Mutex::new(None)),
        }
    }

    fn record_completed(&self, module: &ModuleId) {
        self.record_completed_at(module, Instant::now());
    }

    fn record_completed_at(&self, module: &ModuleId, completed_at: Instant) {
        if !self.action_modules.contains(module) {
            return;
        }
        let mut last_completed_at = self
            .last_completed_at
            .lock()
            .expect("action activity lock poisoned");
        let should_update = match *last_completed_at {
            Some(previous) => completed_at >= previous,
            None => true,
        };
        if should_update {
            *last_completed_at = Some(completed_at);
        }
    }

    fn silence_window_elapsed_at(&self, window: Duration, now: Instant) -> bool {
        let last_completed_at = *self
            .last_completed_at
            .lock()
            .expect("action activity lock poisoned");
        last_completed_at.is_some_and(|completed_at| {
            now.checked_duration_since(completed_at)
                .is_some_and(|elapsed| elapsed >= window)
        })
    }
}

struct EvalExternalActionExecutor {
    case_id: String,
    reporter: LiveReporter,
    actions: Rc<ActionActivityTracker>,
}

impl EvalExternalActionExecutor {
    fn new(case_id: String, reporter: LiveReporter, actions: Rc<ActionActivityTracker>) -> Self {
        Self {
            case_id,
            reporter,
            actions,
        }
    }
}

#[async_trait(?Send)]
impl ExternalActionExecutor for EvalExternalActionExecutor {
    async fn invoke(
        &self,
        invocation: ExternalActionInvocation,
    ) -> Result<ExternalActionInvocationResult, PortError> {
        self.actions.record_completed(&builtin::action());
        self.reporter.emit_port(
            Some(&self.case_id),
            "external_action_invoked",
            serde_json::json!({
                "sender": invocation.invoked_by.to_string(),
                "action_id": invocation.action_id,
                "arguments": invocation.arguments,
            }),
            format!("{} external_action_invoked", self.reporter.log_prefix()),
        )?;
        Ok(ExternalActionInvocationResult {
            accepted: true,
            message: "external action accepted by eval host".to_owned(),
        })
    }
}

#[derive(Debug, Clone)]
struct RuntimeSettleTracker {
    last_progress_count: usize,
    last_progress_at: Instant,
}

impl RuntimeSettleTracker {
    fn new(progress_count: usize, now: Instant) -> Self {
        Self {
            last_progress_count: progress_count,
            last_progress_at: now,
        }
    }

    fn observe_progress_count(&mut self, progress_count: usize, now: Instant) {
        if progress_count == self.last_progress_count {
            return;
        }
        self.last_progress_count = progress_count;
        self.last_progress_at = now;
    }

    fn runtime_silence_elapsed_at(
        &self,
        window: Duration,
        llm_in_flight: usize,
        now: Instant,
    ) -> bool {
        llm_in_flight == 0
            && now
                .checked_duration_since(self.last_progress_at)
                .is_some_and(|elapsed| elapsed >= window)
    }
}

fn runtime_ready_to_score_at(
    actions: &ActionActivityTracker,
    settle: &RuntimeSettleTracker,
    llm_in_flight: usize,
    input_phase_finished: bool,
    allow_empty_output: bool,
    step_driven_case: bool,
    now: Instant,
) -> bool {
    if !settle.runtime_silence_elapsed_at(RUNTIME_SILENCE_WINDOW, llm_in_flight, now) {
        return false;
    }
    actions.silence_window_elapsed_at(RUNTIME_ACTION_SILENCE_WINDOW, now)
        || (input_phase_finished && (allow_empty_output || step_driven_case))
}

#[derive(Clone)]
pub(crate) struct RecordingUtteranceSink {
    case_id: String,
    reporter: LiveReporter,
    actions: Rc<ActionActivityTracker>,
    complete: Arc<Mutex<Vec<RecordedUtterance>>>,
    visualizer: Option<VisualizerEventSink>,
}

fn normalize_eval_utterance_text(text: String) -> String {
    let trimmed = text.trim();
    if trimmed.starts_with('"')
        && trimmed.ends_with('"')
        && let Ok(unquoted) = serde_json::from_str::<String>(trimmed)
    {
        return unquoted;
    }
    text
}

impl std::fmt::Debug for RecordingUtteranceSink {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RecordingUtteranceSink")
            .field("case_id", &self.case_id)
            .finish_non_exhaustive()
    }
}

impl RecordingUtteranceSink {
    fn new(
        case_id: String,
        reporter: LiveReporter,
        actions: Rc<ActionActivityTracker>,
        visualizer: Option<VisualizerEventSink>,
    ) -> Self {
        Self {
            case_id,
            reporter,
            actions,
            complete: Arc::new(Mutex::new(Vec::new())),
            visualizer,
        }
    }

    fn last_complete(&self) -> Option<RecordedUtterance> {
        self.complete
            .lock()
            .expect("utterance lock poisoned")
            .last()
            .cloned()
    }

    fn snapshot(&self) -> Vec<RecordedUtterance> {
        self.complete
            .lock()
            .expect("utterance lock poisoned")
            .clone()
    }

    fn matching_count(&self, scope: Option<&str>, module: &str, target: &str) -> usize {
        self.complete
            .lock()
            .expect("utterance lock poisoned")
            .iter()
            .filter(|utterance| {
                scope.is_none_or(|scope| utterance.scope == scope)
                    && utterance.module == module
                    && utterance.target == target
            })
            .count()
    }

    fn matching_at(
        &self,
        scope: Option<&str>,
        module: &str,
        target: &str,
        index: usize,
    ) -> Option<RecordedUtterance> {
        self.complete
            .lock()
            .expect("utterance lock poisoned")
            .iter()
            .filter(|utterance| {
                scope.is_none_or(|scope| utterance.scope == scope)
                    && utterance.module == module
                    && utterance.target == target
            })
            .nth(index)
            .cloned()
    }

    fn last_matching(
        &self,
        scope: Option<&str>,
        module: &str,
        target: &str,
    ) -> Option<RecordedUtterance> {
        self.complete
            .lock()
            .expect("utterance lock poisoned")
            .iter()
            .rev()
            .find(|utterance| {
                scope.is_none_or(|scope| utterance.scope == scope)
                    && utterance.module == module
                    && utterance.target == target
            })
            .cloned()
    }
}

#[async_trait(?Send)]
impl UtteranceSink for RecordingUtteranceSink {
    async fn on_complete(&self, utterance: Utterance) -> Result<(), PortError> {
        let sender_module = utterance.sender.module.clone();
        let recorded = RecordedUtterance {
            sender: utterance.sender.to_string(),
            scope: utterance.sender.scope.to_string(),
            module: utterance.sender.module.as_str().to_string(),
            replica: utterance.sender.replica.get(),
            target: utterance.target,
            text: normalize_eval_utterance_text(utterance.text),
            emitted_at: utterance.emitted_at.to_rfc3339(),
        };
        self.complete
            .lock()
            .map_err(|_| PortError::Backend("utterance lock poisoned".into()))?
            .push(recorded.clone());
        self.actions.record_completed(&sender_module);
        self.reporter.emit_port(
            Some(&self.case_id),
            "utterance_completed",
            serde_json::json!({
                "sender": recorded.sender.clone(),
                "target": recorded.target.clone(),
                "text": recorded.text.clone(),
                "emitted_at": recorded.emitted_at.clone(),
            }),
            format!(
                "{} utterance {} sender={} target={} chars={}",
                self.reporter.log_prefix(),
                self.reporter.log_scope(&self.case_id),
                recorded.sender,
                recorded.target,
                recorded.text.chars().count()
            ),
        )?;
        if let Some(visualizer) = &self.visualizer {
            visualizer.send(VisualizerEvent::UtteranceCompleted {
                tab_id: VisualizerTabId::new(self.case_id.clone()),
                utterance: UtteranceView {
                    sender: recorded.sender,
                    target: recorded.target,
                    generation_id: Some(utterance.generation_id),
                    text: recorded.text,
                    emitted_at: utterance.emitted_at,
                },
            });
        }
        Ok(())
    }

    async fn on_delta(&self, delta: UtteranceDelta) -> Result<(), PortError> {
        if let Some(visualizer) = &self.visualizer {
            visualizer.send(VisualizerEvent::UtteranceDelta {
                tab_id: VisualizerTabId::new(self.case_id.clone()),
                utterance: UtteranceDeltaView {
                    sender: delta.sender.to_string(),
                    target: delta.target,
                    generation_id: delta.generation_id,
                    sequence: delta.sequence,
                    delta: delta.delta,
                },
            });
        }
        Ok(())
    }
}

#[derive(Debug)]
pub(crate) struct RecordingRuntimeEventSink {
    events: Mutex<Vec<RuntimeEvent>>,
    timed_events: Mutex<Vec<(u64, RuntimeEvent)>>,
    eval_events: Mutex<Vec<crate::timeline::EvalEvent>>,
    case_started: Instant,
    progress_events: AtomicUsize,
    llm_in_flight: AtomicUsize,
    stop: Arc<AtomicBool>,
    case_id: String,
    reporter: LiveReporter,
    visualizer: Option<VisualizerEventSink>,
}

impl RecordingRuntimeEventSink {
    fn new(
        case_id: String,
        reporter: LiveReporter,
        visualizer: Option<VisualizerEventSink>,
    ) -> Self {
        Self {
            events: Mutex::new(Vec::new()),
            timed_events: Mutex::new(Vec::new()),
            eval_events: Mutex::new(Vec::new()),
            case_started: Instant::now(),
            progress_events: AtomicUsize::new(0),
            llm_in_flight: AtomicUsize::new(0),
            stop: Arc::new(AtomicBool::new(false)),
            case_id,
            reporter,
            visualizer,
        }
    }

    fn snapshot(&self) -> Vec<RuntimeEvent> {
        self.events
            .lock()
            .expect("runtime event lock poisoned")
            .clone()
    }

    fn timed_snapshot(&self) -> Vec<(u64, RuntimeEvent)> {
        self.timed_events
            .lock()
            .expect("timed runtime event lock poisoned")
            .clone()
    }

    fn eval_event_snapshot(&self) -> Vec<crate::timeline::EvalEvent> {
        self.eval_events
            .lock()
            .expect("eval event lock poisoned")
            .clone()
    }

    fn elapsed_ms(&self) -> u64 {
        duration_millis_u64(self.case_started.elapsed())
    }

    fn record_eval_event(
        &self,
        scope: nuillu_types::ScopeId,
        module: ModuleId,
        replica: u8,
        step: Option<String>,
        payload: crate::timeline::EvalEventPayload,
    ) {
        self.eval_events
            .lock()
            .expect("eval event lock poisoned")
            .push(crate::timeline::EvalEvent {
                sequence: 0,
                offset_ms: self.elapsed_ms(),
                scope,
                module,
                replica,
                step,
                payload,
            });
    }

    fn activation_timeline(&self) -> Vec<ModuleActivationRecord> {
        let timed_events = self
            .timed_events
            .lock()
            .expect("timed runtime event lock poisoned");
        build_activation_timeline(&timed_events)
    }

    fn stop_requested(&self) -> bool {
        self.stop.load(Ordering::Relaxed)
    }

    fn request_stop(&self, reason: &str) {
        if !self.stop.swap(true, Ordering::Relaxed) {
            let _ = self.reporter.emit_port(
                Some(&self.case_id),
                "stop_requested",
                serde_json::json!({ "reason": reason }),
                format!(
                    "{} stop requested {} reason={}",
                    self.reporter.log_prefix(),
                    self.reporter.log_scope(&self.case_id),
                    reason
                ),
            );
        }
    }

    fn event_count(&self) -> usize {
        self.events
            .lock()
            .expect("runtime event lock poisoned")
            .len()
    }

    fn progress_event_count(&self) -> usize {
        self.progress_events.load(Ordering::Relaxed)
    }

    fn llm_in_flight_count(&self) -> usize {
        self.llm_in_flight.load(Ordering::Relaxed)
    }

    fn scheduled_wait_remaining(&self) -> Option<Duration> {
        let elapsed = self.case_started.elapsed();
        let timed_events = self
            .timed_events
            .lock()
            .expect("timed runtime event lock poisoned");
        scheduled_wait_remaining_from_timed_events(&timed_events, elapsed)
    }
}

#[derive(Clone)]
struct LlmCallLimitHook {
    case_id: String,
    max_llm_calls: Option<u64>,
    calls: Arc<AtomicU64>,
    stop: Arc<AtomicBool>,
    reporter: LiveReporter,
}

impl LlmCallLimitHook {
    fn new(
        case_id: String,
        max_llm_calls: Option<u64>,
        stop: Arc<AtomicBool>,
        reporter: LiveReporter,
    ) -> Self {
        Self {
            case_id,
            max_llm_calls,
            calls: Arc::new(AtomicU64::new(0)),
            stop,
            reporter,
        }
    }

    fn record_call(&self) {
        let call = self.calls.fetch_add(1, Ordering::Relaxed).saturating_add(1);
        if !self.max_llm_calls.is_some_and(|max| call >= max)
            || self.stop.swap(true, Ordering::Relaxed)
        {
            return;
        }
        let _ = self.reporter.emit_port(
            Some(&self.case_id),
            "stop_requested",
            serde_json::json!({ "reason": "max-llm-calls" }),
            format!(
                "{} stop requested {} reason=max-llm-calls",
                self.reporter.log_prefix(),
                self.reporter.log_scope(&self.case_id)
            ),
        );
    }
}

impl OnModelInput for LlmCallLimitHook {
    async fn call(&self, _cx: &ModelInputHookContext<'_>) {
        self.record_call();
    }
}

fn scheduled_wait_remaining_from_timed_events(
    timed_events: &[(u64, RuntimeEvent)],
    elapsed: Duration,
) -> Option<Duration> {
    let elapsed_ms = duration_millis_u64(elapsed);
    timed_events
        .iter()
        .filter_map(|(offset_ms, event)| {
            let delayed_for = match event {
                RuntimeEvent::ModuleBatchThrottled { delayed_for, .. } => *delayed_for,
                _ => return None,
            };
            let wait_until_ms = offset_ms.saturating_add(duration_millis_u64(delayed_for));
            (wait_until_ms > elapsed_ms)
                .then(|| Duration::from_millis(wait_until_ms.saturating_sub(elapsed_ms)))
        })
        .max()
}

fn runtime_event_counts_as_eval_progress(event: &RuntimeEvent) -> bool {
    match event {
        RuntimeEvent::LlmSemaphoreWaitStarted { .. }
        | RuntimeEvent::LlmAccessed { .. }
        | RuntimeEvent::LlmCompleted { .. }
        | RuntimeEvent::MemoUpdated { .. }
        | RuntimeEvent::SessionCompactionStarted { .. }
        | RuntimeEvent::SessionCompactionCompleted { .. }
        | RuntimeEvent::SessionCompactionFailed { .. }
        | RuntimeEvent::ModuleActivationAttemptFailed { .. }
        | RuntimeEvent::ModuleTaskFailed { .. } => true,
        RuntimeEvent::ModuleBatchThrottled { .. }
        | RuntimeEvent::ModuleBatchReady { .. }
        | RuntimeEvent::ModuleActivationCompleted { .. }
        | RuntimeEvent::ModuleWarning { .. } => false,
    }
}

impl RuntimeEventSink for RecordingRuntimeEventSink {
    fn on_event(&self, event: RuntimeEvent) -> Result<(), PortError> {
        match &event {
            RuntimeEvent::LlmAccessed { .. } => {
                self.llm_in_flight.fetch_add(1, Ordering::Relaxed);
            }
            RuntimeEvent::LlmCompleted { .. } => {
                self.llm_in_flight.fetch_sub(1, Ordering::Relaxed);
            }
            _ => {}
        }
        let live_message = match &event {
            RuntimeEvent::LlmSemaphoreWaitStarted { owner, tier, .. } => format!(
                "{} llm-semaphore-wait-started {} owner={} tier={:?}",
                self.reporter.log_prefix(),
                self.reporter.log_scope(&self.case_id),
                owner,
                tier
            ),
            RuntimeEvent::LlmAccessed {
                call, owner, tier, ..
            } => format!(
                "{} llm-accessed {} call={} owner={} tier={:?}",
                self.reporter.log_prefix(),
                self.reporter.log_scope(&self.case_id),
                call,
                owner,
                tier
            ),
            RuntimeEvent::LlmCompleted {
                call, owner, tier, ..
            } => format!(
                "{} llm-completed {} call={} owner={} tier={:?}",
                self.reporter.log_prefix(),
                self.reporter.log_scope(&self.case_id),
                call,
                owner,
                tier
            ),
            RuntimeEvent::MemoUpdated {
                owner, char_count, ..
            } => format!(
                "{} memo-updated {} owner={} chars={}",
                self.reporter.log_prefix(),
                self.reporter.log_scope(&self.case_id),
                owner,
                char_count
            ),
            RuntimeEvent::ModuleBatchThrottled {
                owner, delayed_for, ..
            } => format!(
                "{} module-batch-throttled {} owner={} delayed_ms={}",
                self.reporter.log_prefix(),
                self.reporter.log_scope(&self.case_id),
                owner,
                delayed_for.as_millis()
            ),
            RuntimeEvent::ModuleBatchReady {
                activation_id,
                owner,
                batch_type,
                batch_debug,
                ..
            } => format!(
                "{} module-batch-ready {} activation={} owner={} type={} chars={}",
                self.reporter.log_prefix(),
                self.reporter.log_scope(&self.case_id),
                activation_id,
                owner,
                batch_type,
                batch_debug.chars().count()
            ),
            RuntimeEvent::ModuleActivationCompleted {
                activation_id,
                owner,
                duration,
                succeeded,
                ..
            } => format!(
                "{} module-activation-completed {} activation={} owner={} duration_ms={} succeeded={}",
                self.reporter.log_prefix(),
                self.reporter.log_scope(&self.case_id),
                activation_id,
                owner,
                duration.as_millis(),
                succeeded
            ),
            RuntimeEvent::ModuleActivationAttemptFailed {
                activation_id,
                owner,
                activation_attempt,
                max_attempts,
                message,
                ..
            } => format!(
                "{} module-activation-attempt-failed {} activation={} owner={} attempt={}/{} error={}",
                self.reporter.log_prefix(),
                self.reporter.log_scope(&self.case_id),
                activation_id,
                owner,
                activation_attempt,
                max_attempts,
                message
            ),
            RuntimeEvent::ModuleTaskFailed {
                owner,
                phase,
                message,
                ..
            } => format!(
                "{} module-task-failed {} owner={} phase={} error={}",
                self.reporter.log_prefix(),
                self.reporter.log_scope(&self.case_id),
                owner,
                phase,
                message
            ),
            RuntimeEvent::ModuleWarning { owner, message, .. } => format!(
                "{} module-warning {} owner={} message={}",
                self.reporter.log_prefix(),
                self.reporter.log_scope(&self.case_id),
                owner,
                message
            ),
            RuntimeEvent::SessionCompactionStarted {
                owner,
                session_key,
                input_tokens,
                threshold,
                ..
            } => format!(
                "{} session-compaction-started {} owner={} session={} input_tokens={} threshold={}",
                self.reporter.log_prefix(),
                self.reporter.log_scope(&self.case_id),
                owner,
                session_key,
                input_tokens,
                threshold
            ),
            RuntimeEvent::SessionCompactionCompleted {
                owner,
                session_key,
                input_tokens,
                before_items,
                after_items,
                ..
            } => format!(
                "{} session-compaction-completed {} owner={} session={} input_tokens={} items={}->{}",
                self.reporter.log_prefix(),
                self.reporter.log_scope(&self.case_id),
                owner,
                session_key,
                input_tokens,
                before_items,
                after_items
            ),
            RuntimeEvent::SessionCompactionFailed {
                owner,
                session_key,
                input_tokens,
                message,
                ..
            } => format!(
                "{} session-compaction-failed {} owner={} session={} input_tokens={} error={}",
                self.reporter.log_prefix(),
                self.reporter.log_scope(&self.case_id),
                owner,
                session_key,
                input_tokens,
                message
            ),
        };
        self.events
            .lock()
            .map_err(|_| PortError::Backend("runtime event lock poisoned".into()))?
            .push(event.clone());
        let offset_ms = duration_millis_u64(self.case_started.elapsed());
        self.timed_events
            .lock()
            .map_err(|_| PortError::Backend("timed runtime event lock poisoned".into()))?
            .push((offset_ms, event.clone()));
        if runtime_event_counts_as_eval_progress(&event) {
            self.progress_events.fetch_add(1, Ordering::Relaxed);
        }
        let minimum = if runtime_event_is_notable(&event) {
            LiveOutput::Normal
        } else {
            LiveOutput::Verbose
        };
        self.reporter.emit_port_at(
            Some(&self.case_id),
            "runtime_event",
            serde_json::json!({ "event": event }),
            live_message,
            minimum,
        )?;
        if let Some(visualizer) = &self.visualizer {
            visualizer.send(VisualizerEvent::RuntimeEvent {
                tab_id: VisualizerTabId::new(self.case_id.clone()),
                event: event.clone(),
            });
            match &event {
                RuntimeEvent::ModuleTaskFailed {
                    owner,
                    phase,
                    message,
                    ..
                } => {
                    visualizer.send(VisualizerEvent::Error {
                        tab_id: VisualizerTabId::new(self.case_id.clone()),
                        error: VisualizerErrorView {
                            at: Utc::now(),
                            source: "runtime".to_string(),
                            phase: phase.clone(),
                            owner: Some(owner.to_string()),
                            message: message.clone(),
                        },
                    });
                }
                RuntimeEvent::ModuleActivationAttemptFailed {
                    owner,
                    activation_attempt,
                    max_attempts,
                    message,
                    ..
                } => {
                    visualizer.send(VisualizerEvent::Error {
                        tab_id: VisualizerTabId::new(self.case_id.clone()),
                        error: VisualizerErrorView {
                            at: Utc::now(),
                            source: "runtime".to_string(),
                            phase: format!(
                                "activate-attempt-{activation_attempt}-of-{max_attempts}"
                            ),
                            owner: Some(owner.to_string()),
                            message: message.clone(),
                        },
                    });
                }
                _ => {}
            }
        }
        Ok(())
    }
}

fn runtime_event_is_notable(event: &RuntimeEvent) -> bool {
    matches!(
        event,
        RuntimeEvent::ModuleActivationAttemptFailed { .. }
            | RuntimeEvent::ModuleTaskFailed { .. }
            | RuntimeEvent::ModuleWarning { .. }
            | RuntimeEvent::SessionCompactionFailed { .. }
    )
}

fn aggregate_suite(run: SuiteRunReport, cases: Vec<CaseSummary>) -> SuiteReport {
    let case_count = cases.len();
    let passed_cases = cases.iter().filter(|case| case.passed).count();
    let invalid_cases = cases.iter().filter(|case| case.invalid).count();
    let failed_cases = case_count.saturating_sub(passed_cases + invalid_cases);
    let mean_score = if cases.is_empty() {
        0.0
    } else {
        cases.iter().map(|case| case.score).sum::<f64>() / cases.len() as f64
    };
    let metrics = SuiteMetrics::from_case_counts(&cases);

    SuiteReport {
        run,
        case_count,
        passed_cases,
        failed_cases,
        invalid_cases,
        mean_score,
        metrics,
        timing: SuiteTiming { elapsed_ms: 0 },
        cases,
    }
}

fn case_id(path: &Path, case: &EvalCase) -> String {
    case.id().map(String::from).unwrap_or_else(|| {
        path.with_extension("")
            .file_name()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_else(|| "case".to_string())
    })
}

fn sanitize_id(id: &str) -> String {
    id.chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_') {
                ch
            } else {
                '-'
            }
        })
        .collect()
}

fn write_json_file(path: &Path, value: &impl Serialize) -> Result<(), RunnerError> {
    let bytes = serde_json::to_vec_pretty(value).map_err(|error| RunnerError::Driver {
        path: path.to_path_buf(),
        message: error.to_string(),
    })?;
    std::fs::write(path, bytes).map_err(|source| RunnerError::WriteOutput {
        path: path.to_path_buf(),
        source,
    })
}

pub use nuillu_server::{default_run_id, install_lutum_trace_subscriber};

fn install_trace_subscriber_for_runner() -> Result<(), RunnerError> {
    install_lutum_trace_subscriber().map_err(|error| RunnerError::TraceSubscriber {
        message: error.to_string(),
    })
}

fn panic_payload_message(payload: &(dyn Any + Send)) -> String {
    if let Some(message) = payload.downcast_ref::<&str>() {
        (*message).to_string()
    } else if let Some(message) = payload.downcast_ref::<String>() {
        message.clone()
    } else {
        "non-string panic payload".to_string()
    }
}
