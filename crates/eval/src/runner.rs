use std::{
    any::Any,
    collections::{BTreeMap, BTreeSet, HashSet},
    fs::{File, OpenOptions},
    io::{self, Write},
    num::NonZeroUsize,
    panic::AssertUnwindSafe,
    path::{Path, PathBuf},
    rc::Rc,
    sync::atomic::{AtomicBool, AtomicUsize, Ordering},
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

use anyhow::{Context, Result};
use async_trait::async_trait;
use chrono::{DateTime, Duration as ChronoDuration, FixedOffset, Utc};
use futures::FutureExt as _;
use lutum_eval::{RawTraceSnapshot, TraceSnapshot};
use lutum_in_memory_adapter::InMemoryCognitionLogRepository;
use lutum_libsql_adapter::{LibsqlAgentStore, LibsqlAgentStoreConfig};
use nuillu_agent::{AgentEventLoopConfig, run as run_agent};
use nuillu_blackboard::{
    ActivationRatio, Blackboard, BlackboardCommand, BlackboardInner, Bpm, CognitionLogEntry,
    CognitionLogEntryRecord, MemoLogRecord, MemoryMetadata, ModuleConfig, ModulePolicy,
    ModuleRunStatus, PolicyMetaPatch, ResourceAllocation, ZeroReplicaWindowPolicy, linear_ratio_fn,
};
use nuillu_memory::{
    LinkedMemoryQuery, MemoryCapabilities, MemoryLinkDirection, MemoryLinkRelation, MemoryQuery,
    MemoryRecord, MemoryStore, NewMemoryLink,
};
use nuillu_module::ports::{Clock, PortError, SystemClock};
use nuillu_module::{
    CapabilityProviderConfig, CapabilityProviderPorts, CapabilityProviderRuntime,
    CapabilityProviders, CognitionLogUpdated, InternalHarnessIo, InteroceptionRuntimePolicy,
    LlmConcurrencyPool, ModuleRegistry, Participant, RuntimeEvent, RuntimeEventSink, RuntimePolicy,
    SceneRegistry, SensoryInput, SensoryInputMailbox, SensoryModality, SessionCompactionPolicy,
    apply_standard_dependencies,
};
use nuillu_reward::{IndexedPolicy, PolicyCapabilities, PolicyRecord, PolicyStore};
use nuillu_speak::{Utterance, UtteranceDelta, UtteranceSink, UtteranceWriter};
use nuillu_types::{
    MemoryIndex, MemoryRank, ModelTier, ModuleId, ModuleInstanceId, PolicyIndex, PolicyRank,
    ReplicaCapRange, ReplicaIndex, SignedUnitF32, UnitF32, builtin,
};
use nuillu_visualizer_protocol::{
    AllocationView, BlackboardSnapshot, CognitionEntryView, CognitionLogView, MemoView,
    MemoryMetadataView, MemoryPage, MemoryRecordView, ModuleSettingsView, ModuleStatusView,
    TabStatus, UtteranceDeltaView, UtteranceProgressView, UtteranceView, VisualizerAction,
    VisualizerClientMessage, VisualizerCommand, VisualizerErrorView, VisualizerEvent,
    VisualizerServerMessage, VisualizerTabId, ZeroReplicaWindowView, start_activation_action_id,
};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tokio::task::LocalSet;

use crate::{
    artifact::CaseArtifact,
    cases::{
        ActivateAllocation, ArtifactTextField, CaseFileError, Check, DEFAULT_FULL_AGENT_MODULES,
        EvalCase, EvalInteroceptiveMode, EvalLimits, EvalModule, EvalStep, FullAgentCase,
        FullAgentInput, MemoryLinkSeed, ModuleCase, ModuleEvalStep, ModuleEvalTarget, PolicySeed,
        WaitFor, discover_case_files, is_full_agent_case_path, parse_case_file, parse_case_now,
        parse_memory_datetime, wake_arousal_max_is_set, wake_arousal_min_is_set,
    },
    evaluation::{
        CaseReport, CaseSummary, CaseTiming, CaseTrialSummary, ModuleActivationRecord,
        SuiteMetrics, SuiteModelNames, SuiteReport, SuiteRunReport, SuiteTiming,
        aggregate_trial_timing, artifact_text, build_activation_timeline, evaluate_case,
        field_label, normalize_text_block, numeric_range_outcome, pointer_number, pointer_text,
    },
    judge::{LlmRubricJudge, RubricJudge},
    state_dump::{
        AgenticDeadlockDump, AllocationModuleDump, AllocationProposalDump, BlackboardLastStateDump,
        CognitionEntryDump, CognitionLogDump, DumpText, FullAgentLastStateCaseDump,
        FullAgentLastStateDump, InteroceptionDump, MemoLogDump, MemoryEntryDump,
        MemoryLastStateDump, MemoryMetadataDump, ModuleInstanceDump, ReplicaCapDump, UtteranceDump,
        render_full_agent_last_state_eure,
    },
    trace_json::{raw_trace_has_error, raw_trace_snapshot_json, trace_snapshot_json},
};

const IDLE_REPORT_INTERVAL: Duration = Duration::from_secs(30);
const FULL_AGENT_ACTION_SILENCE_WINDOW: Duration = Duration::from_millis(200);
const FULL_AGENT_RUNTIME_SILENCE_WINDOW: Duration = Duration::from_millis(200);
const FULL_AGENT_IDLE_TIMEOUT: Duration = Duration::from_secs(5);
const FULL_AGENT_STEP_SETTLE_TIMEOUT: Duration = Duration::from_secs(12);
const EVAL_CASE_TIMEOUT: Duration = Duration::from_secs(60);
const EVAL_POLL_INTERVAL: Duration = Duration::from_millis(100);
const EVAL_MEMO_RETAINED_PER_OWNER: usize = 8;
const EVAL_COGNITION_LOG_RETAINED_ENTRIES: usize = 16;

pub use nuillu_server::{
    EmbeddingBackendConfig, LlmBackendConfig, model_concurrency_from_backends,
};

#[derive(Debug, Clone)]
pub struct RunnerConfig {
    pub cases_root: PathBuf,
    pub output_root: PathBuf,
    pub llm_log_root: PathBuf,
    pub run_id: String,
    pub judge_backend: LlmBackendConfig,
    pub cheap_backend: LlmBackendConfig,
    pub default_backend: LlmBackendConfig,
    pub premium_backend: LlmBackendConfig,
    pub model_dir: PathBuf,
    pub embedding_backend: Option<EmbeddingBackendConfig>,
    pub fail_fast: bool,
    pub failed_only: bool,
    pub failed_from: Option<PathBuf>,
    pub model_concurrency: BTreeMap<String, Option<NonZeroUsize>>,
    pub llm_concurrency_pool: LlmConcurrencyPool,
    pub trials: NonZeroUsize,
    pub case_patterns: Vec<String>,
    pub module_filters: Vec<EvalModule>,
    pub disabled_modules: Vec<EvalModule>,
    pub exclude_full_agent: bool,
    pub full_agent_only: bool,
}

/// Modules that may never be disabled via `RunnerConfig::disabled_modules` —
/// removing them breaks the basic observe → cognize → speak pipeline that the
/// full-agent eval cases assume.
pub const REQUIRED_FULL_AGENT_MODULES: &[EvalModule] = &[
    EvalModule::AllocationController,
    EvalModule::Sensory,
    EvalModule::Speak,
];

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

    fn cached_memory_page(&self, case_id: &str, page: usize, per_page: usize) -> MemoryPage {
        memory_page_from_records(
            self.memory_cache
                .get(case_id)
                .map(Vec::as_slice)
                .unwrap_or_default(),
            page,
            per_page,
        )
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
                VisualizerCommand::ListMemories {
                    tab_id,
                    page,
                    per_page,
                } => {
                    let page = self.cached_memory_page(tab_id.as_str(), page, per_page);
                    self.send_event(VisualizerEvent::MemoryPage { tab_id, page });
                }
                VisualizerCommand::QueryMemory {
                    tab_id,
                    query,
                    limit,
                } => {
                    let needle = query.to_lowercase();
                    let records = self
                        .memory_cache
                        .get(tab_id.as_str())
                        .map(Vec::as_slice)
                        .unwrap_or_default()
                        .iter()
                        .filter(|record| record.content.to_lowercase().contains(&needle))
                        .take(limit)
                        .cloned()
                        .collect();
                    self.send_event(VisualizerEvent::MemoryQueryResult {
                        tab_id,
                        query,
                        records,
                    });
                }
                VisualizerCommand::SendOneShotSensoryInput { tab_id, .. } => {
                    self.send_event(VisualizerEvent::Log {
                        tab_id,
                        message: "eval case is no longer running".to_string(),
                    });
                }
                VisualizerCommand::CreateAmbientSensoryRow { tab_id, .. }
                | VisualizerCommand::UpdateAmbientSensoryRow { tab_id, .. }
                | VisualizerCommand::RemoveAmbientSensoryRow { tab_id, .. }
                | VisualizerCommand::FetchLinkedMemories { tab_id, .. }
                | VisualizerCommand::DeleteMemory { tab_id, .. }
                | VisualizerCommand::SetModuleDisabled { tab_id, .. }
                | VisualizerCommand::SetModuleSettings { tab_id, .. } => {
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
    LlmLogContext, VisualizerEventSink, VisualizerLlmObserver, bpm_from_cooldown, build_embedder,
    build_model_handle, build_tiers, duration_millis_u64, linked_memory_record_view,
    memory_rank_name, memory_record_view, model_tier_name, module_policy_views,
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
    #[error("module filters matched no eval cases: {modules}")]
    NoModuleCasesMatched { modules: String },
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
    #[error("cannot disable required module '{module}'")]
    DisableRequiredModule { module: &'static str },
    #[error("module cases are not supported with --gui")]
    GuiModuleCasesUnsupported,
    #[error("--gui requires full-agent cases; do not combine with --no-full-agent")]
    GuiExcludeFullAgent,
    #[error("--gui does not support --trials > 1 (got {trials})")]
    GuiTrialsUnsupported { trials: usize },
    #[error("--full-agent-only cannot be combined with --no-full-agent")]
    ConflictingFullAgentFilters,
}

struct CaseExecution {
    artifact: CaseArtifact,
    events: Vec<RuntimeEvent>,
    activations: Vec<ModuleActivationRecord>,
}

struct CaseOutputContext<'a> {
    case_path: &'a Path,
    output_dir: &'a Path,
    case: &'a EvalCase,
    id: &'a str,
    runtime_id: &'a str,
    trial_number: usize,
    reporter: &'a LiveReporter,
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
    select_case_paths(config, true)?
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
    validate_disabled_modules(&config.disabled_modules)?;
    if hooks.visualizer.is_some() && config.trials.get() > 1 {
        return Err(RunnerError::GuiTrialsUnsupported {
            trials: config.trials.get(),
        });
    }
    if hooks.visualizer.is_some() && config.exclude_full_agent {
        return Err(RunnerError::GuiExcludeFullAgent);
    }
    let selection = select_case_paths(config, hooks.visualizer.is_some())?;
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
    let reporter = LiveReporter::new(&config.run_id, &run_dir)?;
    reporter.emit(
        None,
        "suite_started",
        serde_json::json!({
            "cases": case_paths.len(),
            "output_dir": run_dir.display().to_string(),
            "backends": {
                "judge": backend_report(&config.judge_backend),
                "cheap": backend_report(&config.cheap_backend),
                "default": backend_report(&config.default_backend),
                "premium": backend_report(&config.premium_backend),
            },
            "model_concurrency": config.model_concurrency.iter().map(|(model, limit)| {
                (model.clone(), limit.map(NonZeroUsize::get))
            }).collect::<BTreeMap<_, _>>(),
            "trials": config.trials.get(),
        }),
        format!(
            "🚀 eval suite start run={} cases={} trials={} output={}",
            config.run_id,
            case_paths.len(),
            config.trials.get(),
            run_dir.display()
        ),
    )?;

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

    let mut cases = Vec::new();
    for path in case_paths {
        let output =
            run_case_detailed_with_reporter(&path, config, Some(&judge), &reporter, hooks).await?;
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

    let mut report = aggregate_suite(run_report, cases);
    report.timing = SuiteTiming {
        elapsed_ms: duration_millis_u64(suite_started.elapsed()),
    };
    let suite_path = run_dir.join("suite-report.json");
    write_json_file(&suite_path, &report)?;
    eprintln!("\n════════════════════════════════════════════════════════════");
    reporter.emit(
        None,
        "suite_finished",
        serde_json::json!({
            "case_count": report.case_count,
            "passed_cases": report.passed_cases,
            "failed_cases": report.failed_cases,
            "invalid_cases": report.invalid_cases,
            "mean_score": report.mean_score,
            "metrics": report.metrics,
            "elapsed_ms": report.timing.elapsed_ms,
        }),
        format!(
            "🏁 eval suite end run={} ✅passed={} ❌failed={} 💥invalid={} mean_score={:.3} elapsed_ms={}{}",
            config.run_id,
            report.passed_cases,
            report.failed_cases,
            report.invalid_cases,
            report.mean_score,
            report.timing.elapsed_ms,
            format_suite_metrics_inline(&report.metrics)
        ),
    )?;
    if let Some(visualizer) = hooks.visualizer.as_mut() {
        eprintln!("eval suite finished; visualizer remains open until its window is closed");
        visualizer.drain_cached_commands_until_shutdown();
    }
    Ok(report)
}

fn backend_report(backend: &LlmBackendConfig) -> serde_json::Value {
    serde_json::json!({
        "endpoint": backend.endpoint.as_str(),
        "model": backend.model.as_str(),
        "reasoning_effort": backend.reasoning_effort,
        "use_responses_api": backend.use_responses_api,
        "compaction_input_token_threshold": backend.compaction_input_token_threshold,
    })
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
        failed_only: failed_only_requested(config),
        failed_from: failed_from.map(|path| path.display().to_string()),
        fail_fast: config.fail_fast,
        model_concurrency: config
            .model_concurrency
            .iter()
            .map(|(model, limit)| (model.clone(), limit.map(NonZeroUsize::get)))
            .collect(),
        trials: config.trials.get(),
        planned_case_count,
        models: SuiteModelNames {
            judge: config.judge_backend.model.clone(),
            cheap: config.cheap_backend.model.clone(),
            default: config.default_backend.model.clone(),
            premium: config.premium_backend.model.clone(),
        },
        module_filters: config
            .module_filters
            .iter()
            .map(|module| module.as_str().to_string())
            .collect(),
        disabled_modules: config
            .disabled_modules
            .iter()
            .map(|module| module.as_str().to_string())
            .collect(),
        exclude_full_agent: config.exclude_full_agent,
        full_agent_only: config.full_agent_only,
    }
}

pub async fn run_case_detailed(
    case_path: &Path,
    config: &RunnerConfig,
    judge: Option<&dyn RubricJudge>,
) -> Result<CaseRunOutput, RunnerError> {
    install_trace_subscriber_for_runner()?;
    validate_disabled_modules(&config.disabled_modules)?;
    let run_dir = config.output_root.join(&config.run_id);
    std::fs::create_dir_all(&run_dir).map_err(|source| RunnerError::WriteOutput {
        path: run_dir.clone(),
        source,
    })?;
    let reporter = LiveReporter::new(&config.run_id, &run_dir)?;
    let mut hooks = RunnerHooks::none();
    run_case_detailed_with_reporter(case_path, config, judge, &reporter, &mut hooks).await
}

fn select_case_paths(config: &RunnerConfig, gui_only: bool) -> Result<CaseSelection, RunnerError> {
    if config.exclude_full_agent && config.full_agent_only {
        return Err(RunnerError::ConflictingFullAgentFilters);
    }

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
        case_paths = filter_failed_only_case_paths(case_paths, &failed_cases)?;
    }
    if !case_paths.is_empty() || failed_from.is_none() {
        case_paths = filter_case_paths(case_paths, &config.case_patterns)?;
    }
    if !case_paths.is_empty() || failed_from.is_none() {
        case_paths = filter_exclude_full_agent_case_paths(case_paths, config.exclude_full_agent);
    }
    if !case_paths.is_empty() || failed_from.is_none() {
        case_paths = filter_full_agent_only_case_paths(case_paths, config.full_agent_only);
    }
    if !case_paths.is_empty() || failed_from.is_none() {
        case_paths = filter_module_case_paths(case_paths, &config.module_filters)?;
    }
    if gui_only && (!case_paths.is_empty() || failed_from.is_none()) {
        case_paths = filter_gui_case_paths(case_paths)?;
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

fn filter_exclude_full_agent_case_paths(case_paths: Vec<PathBuf>, exclude: bool) -> Vec<PathBuf> {
    if !exclude {
        return case_paths;
    }
    case_paths
        .into_iter()
        .filter(|path| !is_full_agent_case_path(path))
        .collect()
}

fn filter_full_agent_only_case_paths(
    case_paths: Vec<PathBuf>,
    full_agent_only: bool,
) -> Vec<PathBuf> {
    if !full_agent_only {
        return case_paths;
    }
    case_paths
        .into_iter()
        .filter(|path| is_full_agent_case_path(path))
        .collect()
}

fn filter_module_case_paths(
    case_paths: Vec<PathBuf>,
    modules: &[EvalModule],
) -> Result<Vec<PathBuf>, RunnerError> {
    if modules.is_empty() {
        return Ok(case_paths);
    }

    let filters = modules.iter().copied().collect::<HashSet<_>>();
    let mut matched = Vec::new();
    for path in case_paths {
        let case = parse_case_file(&path)?;
        if case_matches_module_filter(&case, &filters) {
            matched.push(path);
        }
    }

    if matched.is_empty() {
        return Err(RunnerError::NoModuleCasesMatched {
            modules: modules
                .iter()
                .map(|module| module.as_str())
                .collect::<Vec<_>>()
                .join(", "),
        });
    }
    Ok(matched)
}

fn case_matches_module_filter(case: &EvalCase, modules: &HashSet<EvalModule>) -> bool {
    match case {
        EvalCase::FullAgent(case) => case
            .effective_modules()
            .iter()
            .any(|module| modules.contains(module)),
        EvalCase::Module { target, .. } => modules.contains(&target.module()),
    }
}

fn filter_gui_case_paths(case_paths: Vec<PathBuf>) -> Result<Vec<PathBuf>, RunnerError> {
    let full_agent_paths = case_paths
        .into_iter()
        .filter(|path| is_full_agent_case_path(path))
        .collect::<Vec<_>>();
    if full_agent_paths.is_empty() {
        return Err(RunnerError::GuiModuleCasesUnsupported);
    }
    Ok(full_agent_paths)
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

async fn run_case_detailed_with_reporter(
    case_path: &Path,
    config: &RunnerConfig,
    judge: Option<&dyn RubricJudge>,
    reporter: &LiveReporter,
    hooks: &mut RunnerHooks,
) -> Result<CaseRunOutput, RunnerError> {
    let case = parse_case_file(case_path)?;
    let id = case_id(case_path, &case);
    let output_dir = config
        .output_root
        .join(&config.run_id)
        .join(sanitize_id(&id));
    std::fs::create_dir_all(&output_dir).map_err(|source| RunnerError::WriteOutput {
        path: output_dir.clone(),
        source,
    })?;
    eprintln!("\n────────────────────────────────────────────────────────────");
    reporter.emit(
        Some(&id),
        "case_started",
        serde_json::json!({
            "path": case_path.display().to_string(),
            "output_dir": output_dir.display().to_string(),
            "trials": config.trials.get(),
        }),
        format!(
            "▶️  eval case start id={} path={} trials={} output={}",
            id,
            case_path.display(),
            config.trials.get(),
            output_dir.display()
        ),
    )?;
    emit_visualizer_open_tab(hooks, &id);

    let trial_count = config.trials.get();
    let case_started = Instant::now();
    if trial_count == 1 {
        let output = run_case_trial_with_timeout(
            case_path,
            config,
            judge,
            reporter,
            hooks,
            &case,
            &id,
            &id,
            &output_dir,
            1,
        )
        .await?;
        emit_case_finished(reporter, &output.summary, output.events.len())?;
        emit_visualizer_case_status(hooks, &output.summary);
        return Ok(output);
    }

    let mut trial_outputs = Vec::with_capacity(trial_count);
    for trial_number in 1..=trial_count {
        let runtime_id = trial_runtime_id(&id, trial_number);
        let trial_output_dir = output_dir.join(trial_dir_name(trial_number));
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
            reporter,
            hooks,
            &case,
            &id,
            &runtime_id,
            &trial_output_dir,
            trial_number,
        )
        .await?;
        emit_trial_finished(reporter, &runtime_id, &output.summary, output.events.len())?;
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
    emit_case_finished(reporter, &summary, event_count)?;

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
    let output = match tokio::time::timeout(
        EVAL_CASE_TIMEOUT,
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
            let message = format!("eval case timed out after {}s", EVAL_CASE_TIMEOUT.as_secs());
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
            case, config, output_dir, runtime_id, reporter, hooks,
        ))
        .catch_unwind(),
    );
    let collected = local.run_until(capture).await;

    let trace = collected.trace;
    let raw_trace = collected.raw;
    let execution = match collected.output {
        Ok(Ok(execution)) => execution,
        Ok(Err(error)) => {
            let message = error.to_string();
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
    let report = evaluate_case(case, &trace, &artifact, judge).await;
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
    let artifact = CaseArtifact::failed(message.clone());
    let events = Vec::new();
    let report = CaseReport {
        runtime_failure: Some(message.clone()),
        checks: Vec::new(),
        modules_checks: Vec::new(),
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
    let trial = CaseTrialSummary {
        trial: trial_number,
        output_dir: output_dir.display().to_string(),
        path: case_path.display().to_string(),
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
        id: id.to_string(),
        description,
        passed,
        invalid,
        score,
        report,
        timing,
        trial_timing: None,
        activations,
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
        id: id.to_string(),
        description,
        passed,
        invalid,
        score,
        report,
        timing: CaseTiming { elapsed_ms },
        trial_timing,
        activations: Vec::new(),
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
        checks: Vec::new(),
        modules_checks: Vec::new(),
        invalid,
        must_pass_ok: passed,
        weighted_points_earned: 0,
        weighted_points_total: 0,
        score,
    }
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

fn emit_case_finished(
    reporter: &LiveReporter,
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
    let case_finished_message = if let Some(runtime_failure) = &summary.report.runtime_failure {
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
    };
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
) -> Result<CaseExecution> {
    match case {
        EvalCase::FullAgent(case) => {
            execute_full_agent_case(case, config, output_dir, case_id, reporter, hooks).await
        }
        EvalCase::Module { target, case } => {
            execute_module_case(*target, case, config, output_dir, case_id, reporter, hooks).await
        }
    }
}

async fn execute_full_agent_case(
    case: &FullAgentCase,
    config: &RunnerConfig,
    output_dir: &Path,
    case_id: &str,
    reporter: &LiveReporter,
    hooks: &mut RunnerHooks,
) -> Result<CaseExecution> {
    let case_modules = full_agent_case_modules(case, &config.disabled_modules);
    let gui_deferred_start = hooks.visualizer.is_some();
    let case_now = parse_case_now(case.now.as_deref())
        .map_err(anyhow::Error::msg)
        .context("parse full-agent case now")?;
    let mut allocation = if gui_deferred_start {
        full_agent_gui_initial_allocation(&case.limits, &case_modules)
    } else {
        full_agent_allocation(&case.limits, &case_modules)
    };
    if !gui_deferred_start && !case.activate_allocation.is_empty() {
        apply_case_activation_allocation(&mut allocation, &case.activate_allocation);
    }
    let env = build_eval_environment(
        output_dir,
        config,
        allocation,
        &case.limits,
        action_module_ids(&case_modules),
        case_now,
        &case.memories,
        &[],
        &[],
        case_id,
        reporter,
        hooks.visualizer.as_ref().map(VisualizerHook::event_sender),
    )
    .await?;
    seed_eval_scene_participants(env.caps.scene(), &case.participants);
    let memory_baseline = memory_snapshot(env.memory.as_ref()).await?;
    let _ = seed_memos(&env.blackboard, env.clock.as_ref(), &case.memos).await?;
    if let Some(visualizer) = hooks.visualizer.as_mut() {
        emit_visualizer_blackboard_snapshot(case_id, &env.blackboard, Some(visualizer)).await;
        emit_visualizer_memory_page(
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
    let step_driven_case = !case.steps.is_empty();
    let activate_allocation = case.activate_allocation.clone();
    let actions = env.actions.clone();
    let events = env.events.clone();
    let clock = env.clock.clone();
    let memory = env.memory.clone();
    let utterances = env.utterances.clone();
    let allocation_blackboard = env.blackboard.clone();
    let allow_empty_output = case.allow_empty_output;
    let mut allocation_reporter =
        AllocationChangeReporter::new(case_id.to_string(), reporter.clone());
    let live_reporter = reporter.clone();
    let case_id_for_idle = case_id.to_string();
    let modules = eval_registry(
        &case_modules,
        &env.memory_caps,
        &env.policy_caps,
        &env.utterance_sink,
        if gui_deferred_start {
            ReplicaHardCap::V1Max
        } else {
            ReplicaHardCap::PolicyMax
        },
    )
    .build(&env.caps)
    .await?;
    let mut visualizer = hooks.visualizer.as_mut();
    let step_failure: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));
    let step_outcomes: Arc<Mutex<Vec<serde_json::Value>>> = Arc::new(Mutex::new(Vec::new()));
    let step_failure_for_loop = step_failure.clone();
    let step_outcomes_for_loop = step_outcomes.clone();
    let live_reporter_for_loop = live_reporter.clone();

    run_agent(
        modules,
        AgentEventLoopConfig {
            idle_threshold: Duration::from_secs(1),
            activate_retries: 2,
            module_failure_limit: 3,
        },
        async move {
            if !gui_deferred_start {
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
                    &allocation_blackboard,
                    utterances.as_ref(),
                    events.as_ref(),
                    clock.as_ref(),
                    visualizer.as_deref(),
                    &live_reporter_for_loop,
                    &step_failure_for_loop,
                    &step_outcomes_for_loop,
                )
                .await;
                input_phase_finished = true;
            }

            let initial_progress_count = events.progress_event_count();
            let mut settle = FullAgentSettleTracker::new(initial_progress_count, Instant::now());
            let mut last_progress_count = initial_progress_count;
            let mut idle_ticks = 0_u64;
            let poll_ms = duration_millis_u64(EVAL_POLL_INTERVAL);
            let idle_report_every_ticks = ticks_for_interval(IDLE_REPORT_INTERVAL, poll_ms);
            let mut tick: u64 = 0;
            loop {
                let now = Instant::now();
                settle.observe_progress_count(events.progress_event_count(), now);
                if events.stop_requested()
                    || full_agent_ready_to_score_at(
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
                            &allocation_blackboard,
                            utterances.as_ref(),
                            events.as_ref(),
                            clock.as_ref(),
                            Some(visualizer),
                            &live_reporter_for_loop,
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
                        .expect("full-agent eval failed to write idle event");
                }
                if idle_for_ms >= duration_millis_u64(FULL_AGENT_IDLE_TIMEOUT) {
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
    let steps_ok = step_driven_case && step_outcomes_all_ok(&recorded_step_outcomes);
    let mut artifact = if let Some(failure) = step_failure_message {
        let mut artifact = CaseArtifact::failed(failure);
        if let Some(utterance) = env.utterances.last_complete() {
            artifact.output = utterance.text;
        }
        artifact
    } else if let Some(utterance) = env.utterances.last_complete() {
        CaseArtifact::new(utterance.text)
    } else if case.allow_empty_output || steps_ok {
        CaseArtifact::new(String::new())
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
    let last_state = build_full_agent_last_state_dump(
        case_id,
        &artifact,
        &env.blackboard,
        env.memory.as_ref(),
        &env.utterances,
        events.len(),
    )
    .await?;
    add_last_state_observation(&mut artifact, &last_state)?;
    write_full_agent_last_state_eure(output_dir, last_state)?;
    if let Some(visualizer) = hooks.visualizer.as_mut() {
        emit_visualizer_blackboard_snapshot(case_id, &env.blackboard, Some(visualizer)).await;
        emit_visualizer_memory_page(
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
    })
}

fn step_outcomes_all_ok(outcomes: &[serde_json::Value]) -> bool {
    !outcomes.is_empty()
        && outcomes
            .iter()
            .all(|outcome| outcome.get("status").and_then(serde_json::Value::as_str) == Some("ok"))
}

async fn execute_module_case(
    target: ModuleEvalTarget,
    case: &ModuleCase,
    config: &RunnerConfig,
    output_dir: &Path,
    case_id: &str,
    reporter: &LiveReporter,
    hooks: &mut RunnerHooks,
) -> Result<CaseExecution> {
    if hooks.visualizer.is_some() {
        anyhow::bail!("module cases are not supported with --gui");
    }
    let case_modules = module_case_modules(target, case);
    let case_now = parse_case_now(case.now.as_deref())
        .map_err(anyhow::Error::msg)
        .context("parse module case now")?;
    let env = build_eval_environment(
        output_dir,
        config,
        module_allocation(target, &case.limits, &case_modules),
        &case.limits,
        action_module_ids(&case_modules),
        case_now,
        &case.memories,
        &case.memory_links,
        &case.policies,
        case_id,
        reporter,
        hooks.visualizer.as_ref().map(VisualizerHook::event_sender),
    )
    .await?;
    seed_eval_scene_participants(env.caps.scene(), &case.participants);
    let memory_baseline = if module_target_uses_memory_store_artifact(target) {
        memory_snapshot(env.memory.as_ref()).await?
    } else {
        BTreeMap::new()
    };
    let policy_baseline = if module_target_uses_policy_store_artifact(target) {
        policy_snapshot(env.policy_store.as_ref()).await?
    } else {
        BTreeSet::new()
    };
    let memo_seed_records = seed_memos(&env.blackboard, env.clock.as_ref(), &case.memos).await?;
    let cognition_seed_records =
        seed_cognition_log(&env.blackboard, env.clock.as_ref(), &case.cognition_log).await;
    if let Some(visualizer) = hooks.visualizer.as_mut() {
        emit_visualizer_blackboard_snapshot(case_id, &env.blackboard, Some(visualizer)).await;
        emit_visualizer_memory_page(
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

    let gui_deferred_start = hooks.visualizer.is_some();
    let target_module = module_id_for_target(target);
    let cognition_baseline_for_target =
        cognition_output_for_module(&env.blackboard, &target_module).await;
    let shutdown_target_module = target_module.clone();
    let run_target_module = target_module.clone();
    let modules = eval_registry(
        &case_modules,
        &env.memory_caps,
        &env.policy_caps,
        &env.utterance_sink,
        if gui_deferred_start {
            ReplicaHardCap::V1Max
        } else {
            ReplicaHardCap::PolicyMax
        },
    )
    .build(&env.caps)
    .await?;
    let harness = env.caps.internal_harness_io();
    let sensory = env.caps.host_io().sensory_input_mailbox();
    let prompt = case.prompt.content.clone();
    let case_inputs = case.inputs.clone();
    let module_steps = case.steps.clone();
    let step_driven_module_case = !module_steps.is_empty();
    let case_id_for_activation = case_id.to_string();
    let has_cognition_log_seed = !case.cognition_log.is_empty();
    let events = env.events.clone();
    let blackboard = env.blackboard.clone();
    let utterances = env.utterances.clone();
    let memory = env.memory.clone();
    let policy_store = env.policy_store.clone();
    let memory_baseline_for_loop = memory_baseline.clone();
    let policy_baseline_for_loop = policy_baseline.clone();
    let cognition_baseline_for_loop = cognition_baseline_for_target.clone();
    let clock = env.clock.clone();
    let case_id_for_gui = case_id.to_string();
    let mut visualizer = hooks.visualizer.as_mut();

    run_agent(
        modules,
        AgentEventLoopConfig {
            idle_threshold: Duration::from_secs(1),
            activate_retries: 2,
            module_failure_limit: 3,
        },
        async move {
            let mut started = !gui_deferred_start;
            let mut module_step_index = 0usize;
            let mut module_step_activation_count = 0usize;
            if started {
                if let Some(step) = module_steps.first() {
                    module_step_activation_count =
                        module_activation_completion_count(events.as_ref(), &run_target_module);
                    activate_module_case_step(
                        target,
                        &blackboard,
                        &harness,
                        &prompt,
                        &run_target_module,
                        step,
                        &case_id_for_activation,
                        &sensory,
                        clock.as_ref(),
                    )
                    .await;
                } else {
                    activate_module_case_target(
                        target,
                        &blackboard,
                        &harness,
                        &prompt,
                        &run_target_module,
                        &memo_seed_records,
                        &cognition_seed_records,
                        has_cognition_log_seed,
                        &case_id_for_activation,
                        &case_inputs,
                        &sensory,
                        clock.as_ref(),
                    )
                    .await;
                }
            }

            loop {
                if let Some(visualizer) = visualizer.as_deref_mut() {
                    let command_outcome = handle_visualizer_commands(
                        &case_id_for_gui,
                        visualizer,
                        None,
                        &blackboard,
                        memory.as_ref(),
                        clock.as_ref(),
                    )
                    .await;
                    if command_outcome.shutdown {
                        break;
                    }
                    if command_outcome.start_requested && !started {
                        visualizer.revoke_action(start_activation_action_id(
                            &VisualizerTabId::new(case_id_for_gui.clone()),
                        ));
                        if let Some(step) = module_steps.first() {
                            module_step_activation_count = module_activation_completion_count(
                                events.as_ref(),
                                &run_target_module,
                            );
                            activate_module_case_step(
                                target,
                                &blackboard,
                                &harness,
                                &prompt,
                                &run_target_module,
                                step,
                                &case_id_for_activation,
                                &sensory,
                                clock.as_ref(),
                            )
                            .await;
                        } else {
                            activate_module_case_target(
                                target,
                                &blackboard,
                                &harness,
                                &prompt,
                                &run_target_module,
                                &memo_seed_records,
                                &cognition_seed_records,
                                has_cognition_log_seed,
                                &case_id_for_activation,
                                &case_inputs,
                                &sensory,
                                clock.as_ref(),
                            )
                            .await;
                        }
                        started = true;
                    }
                }
                emit_visualizer_blackboard_snapshot(
                    &case_id_for_gui,
                    &blackboard,
                    visualizer.as_deref(),
                )
                .await;
                if !started {
                    tokio::task::yield_now().await;
                    tokio::time::sleep(EVAL_POLL_INTERVAL).await;
                    continue;
                }
                let target_done = if step_driven_module_case {
                    module_activation_completion_count(events.as_ref(), &shutdown_target_module)
                        > module_step_activation_count
                } else {
                    match target {
                        ModuleEvalTarget::AttentionSchema | ModuleEvalTarget::CognitionGate => {
                            cognition_eval_has_new_output(
                                &cognition_output_for_module(&blackboard, &shutdown_target_module)
                                    .await,
                                &cognition_baseline_for_loop,
                            ) || module_activation_finished(
                                &blackboard,
                                events.as_ref(),
                                &shutdown_target_module,
                            )
                            .await
                        }
                        ModuleEvalTarget::MemoryRecombination => {
                            !cognition_output_for_module(&blackboard, &shutdown_target_module)
                                .await
                                .is_empty()
                                || module_activation_finished(
                                    &blackboard,
                                    events.as_ref(),
                                    &shutdown_target_module,
                                )
                                .await
                        }
                        ModuleEvalTarget::Memory
                        | ModuleEvalTarget::MemoryCompaction
                        | ModuleEvalTarget::MemoryAssociation => {
                            !memory_diff_records(&memory_baseline_for_loop, memory.as_ref())
                                .await
                                .is_empty()
                                || module_activation_finished(
                                    &blackboard,
                                    events.as_ref(),
                                    &shutdown_target_module,
                                )
                                .await
                        }
                        ModuleEvalTarget::PolicyCompaction => {
                            !policy_diff_observation(
                                &policy_baseline_for_loop,
                                &policy_snapshot(policy_store.as_ref())
                                    .await
                                    .unwrap_or_default(),
                            )
                            .deleted
                            .is_empty()
                                || module_activation_finished(
                                    &blackboard,
                                    events.as_ref(),
                                    &shutdown_target_module,
                                )
                                .await
                        }
                        ModuleEvalTarget::Speak => {
                            utterances.last_complete().is_some()
                                || module_activation_finished(
                                    &blackboard,
                                    events.as_ref(),
                                    &shutdown_target_module,
                                )
                                .await
                        }
                        ModuleEvalTarget::AllocationController => {
                            last_memo_log_content_for_module(&blackboard, &shutdown_target_module)
                                .await
                                .is_some()
                                || module_activation_finished(
                                    &blackboard,
                                    events.as_ref(),
                                    &shutdown_target_module,
                                )
                                .await
                        }
                        _ => last_memo_log_content_for_module(&blackboard, &shutdown_target_module)
                            .await
                            .is_some(),
                    }
                };
                if events.stop_requested() || target_done {
                    if step_driven_module_case
                        && target_done
                        && module_step_index + 1 < module_steps.len()
                    {
                        module_step_index += 1;
                        module_step_activation_count =
                            module_activation_completion_count(events.as_ref(), &run_target_module);
                        activate_module_case_step(
                            target,
                            &blackboard,
                            &harness,
                            &prompt,
                            &run_target_module,
                            &module_steps[module_step_index],
                            &case_id_for_activation,
                            &sensory,
                            clock.as_ref(),
                        )
                        .await;
                        continue;
                    }
                    break;
                }
                tokio::task::yield_now().await;
                tokio::time::sleep(EVAL_POLL_INTERVAL).await;
            }
        },
    )
    .await?;

    let output = match target {
        ModuleEvalTarget::AttentionSchema
        | ModuleEvalTarget::CognitionGate
        | ModuleEvalTarget::MemoryRecombination => {
            cognition_output_for_module(&env.blackboard, &target_module).await
        }
        ModuleEvalTarget::Memory
        | ModuleEvalTarget::MemoryCompaction
        | ModuleEvalTarget::MemoryAssociation => {
            render_memory_store_artifact(&memory_baseline, env.memory.as_ref()).await
        }
        ModuleEvalTarget::Speak => env
            .utterances
            .last_complete()
            .map(|utterance| utterance.text)
            .unwrap_or_default(),
        ModuleEvalTarget::AllocationController => {
            allocation_controller_artifact(&env.blackboard, &target_module).await
        }
        _ => last_memo_log_content_for_module(&env.blackboard, &target_module)
            .await
            .unwrap_or_default(),
    };
    let mut artifact = if output.is_empty() && !case.allow_empty_output {
        if env.events.stop_requested() {
            CaseArtifact::failed("stopped after max-llm-calls before target module produced output")
        } else {
            CaseArtifact::failed("target module did not produce output")
        }
    } else {
        CaseArtifact::new(output)
    };
    add_observations(&mut artifact, &env.blackboard, &env.utterances).await;
    if module_target_uses_memory_store_artifact(target) {
        add_memory_diff_observation(&mut artifact, &memory_baseline, env.memory.as_ref()).await?;
    }
    if module_target_uses_policy_store_artifact(target) {
        add_policy_diff_observation(&mut artifact, &policy_baseline, env.policy_store.as_ref())
            .await?;
    }
    if let Some(visualizer) = hooks.visualizer.as_mut() {
        emit_visualizer_blackboard_snapshot(case_id, &env.blackboard, Some(visualizer)).await;
        emit_visualizer_memory_page(
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
        events: env.events.snapshot(),
        activations: env.events.activation_timeline(),
    })
}

async fn activate_module_case_target(
    target: ModuleEvalTarget,
    blackboard: &Blackboard,
    harness: &InternalHarnessIo,
    prompt: &str,
    run_target_module: &ModuleId,
    memo_seed_records: &[MemoLogRecord],
    cognition_seed_records: &[CognitionLogEntryRecord],
    has_cognition_log_seed: bool,
    case_id: &str,
    inputs: &[FullAgentInput],
    sensory: &SensoryInputMailbox,
    clock: &dyn Clock,
) {
    match target {
        ModuleEvalTarget::Sensory => {
            let mut allocation = blackboard.read(|bb| bb.allocation().clone()).await;
            let mut config = allocation.for_module(run_target_module);
            config.guidance = prompt.to_string();
            allocation.set(run_target_module.clone(), config);
            blackboard
                .apply(BlackboardCommand::SetAllocation(allocation))
                .await;
            publish_full_agent_inputs(case_id, inputs, sensory, clock, None).await;
        }
        ModuleEvalTarget::CognitionGate => {
            let mut allocation = blackboard.read(|bb| bb.allocation().clone()).await;
            let mut config = allocation.for_module(run_target_module);
            config.guidance = prompt.to_string();
            allocation.set(run_target_module.clone(), config);
            blackboard
                .apply(BlackboardCommand::SetAllocation(allocation))
                .await;
            for record in memo_seed_records {
                harness
                    .memo_updated_mailbox()
                    .publish(nuillu_module::MemoUpdated {
                        owner: record.owner.clone(),
                        index: record.index,
                    })
                    .await
                    .expect("module eval failed to publish MemoUpdated");
            }
            harness
                .memo_updated_mailbox()
                .publish(nuillu_module::MemoUpdated {
                    owner: ModuleInstanceId::new(
                        ModuleId::new("eval-harness").expect("eval-harness id is valid"),
                        ReplicaIndex::ZERO,
                    ),
                    index: 0,
                })
                .await
                .expect("module eval failed to publish MemoUpdated");
        }
        ModuleEvalTarget::QueryMemory => {
            let mut allocation = blackboard.read(|bb| bb.allocation().clone()).await;
            let mut config = allocation.for_module(run_target_module);
            config.guidance = prompt.to_string();
            allocation.set(run_target_module.clone(), config);
            blackboard
                .apply(BlackboardCommand::SetAllocation(allocation))
                .await;
            harness
                .cognition_log_updated_mailbox()
                .publish(CognitionLogUpdated::EntryAppended {
                    source: ModuleInstanceId::new(builtin::cognition_gate(), ReplicaIndex::ZERO),
                })
                .await
                .expect("module eval failed to publish CognitionLogUpdated");
        }
        ModuleEvalTarget::AttentionSchema => {
            let mut allocation = blackboard.read(|bb| bb.allocation().clone()).await;
            let mut config = allocation.for_module(run_target_module);
            config.guidance = prompt.to_string();
            allocation.set(run_target_module.clone(), config);
            blackboard
                .apply(BlackboardCommand::SetAllocation(allocation))
                .await;
            for record in memo_seed_records {
                harness
                    .memo_updated_mailbox()
                    .publish(nuillu_module::MemoUpdated {
                        owner: record.owner.clone(),
                        index: record.index,
                    })
                    .await
                    .expect("module eval failed to publish MemoUpdated");
            }
            if has_cognition_log_seed {
                harness
                    .cognition_log_updated_mailbox()
                    .publish(CognitionLogUpdated::EntryAppended {
                        source: ModuleInstanceId::new(
                            builtin::cognition_gate(),
                            ReplicaIndex::ZERO,
                        ),
                    })
                    .await
                    .expect("module eval failed to publish CognitionLogUpdated");
            }
        }
        ModuleEvalTarget::SelfModel => {
            let mut allocation = blackboard.read(|bb| bb.allocation().clone()).await;
            let mut config = allocation.for_module(run_target_module);
            config.guidance = prompt.to_string();
            allocation.set(run_target_module.clone(), config);
            blackboard
                .apply(BlackboardCommand::SetAllocation(allocation))
                .await;
            harness
                .cognition_log_updated_mailbox()
                .publish(CognitionLogUpdated::EntryAppended {
                    source: ModuleInstanceId::new(builtin::attention_schema(), ReplicaIndex::ZERO),
                })
                .await
                .expect("module eval failed to publish CognitionLogUpdated");
        }
        ModuleEvalTarget::Memory => {
            let mut allocation = blackboard.read(|bb| bb.allocation().clone()).await;
            let mut config = allocation.for_module(run_target_module);
            config.guidance = prompt.to_string();
            allocation.set(run_target_module.clone(), config);
            blackboard
                .apply(BlackboardCommand::SetAllocation(allocation))
                .await;
            if has_cognition_log_seed {
                for record in cognition_seed_records {
                    harness
                        .cognition_log_evicted_mailbox()
                        .publish(record.clone())
                        .await
                        .expect("module eval failed to publish CognitionLogEvicted");
                }
            }
        }
        ModuleEvalTarget::MemoryCompaction
        | ModuleEvalTarget::MemoryAssociation
        | ModuleEvalTarget::MemoryRecombination
        | ModuleEvalTarget::PolicyCompaction => {
            let mut allocation = blackboard.read(|bb| bb.allocation().clone()).await;
            let mut config = allocation.for_module(run_target_module);
            config.guidance = prompt.to_string();
            allocation.set(run_target_module.clone(), config);
            blackboard
                .apply(BlackboardCommand::SetAllocation(allocation))
                .await;
            harness
                .interoception_updated_mailbox()
                .publish(nuillu_module::InteroceptiveUpdated)
                .await
                .expect("module eval failed to publish InteroceptiveUpdated");
        }
        ModuleEvalTarget::Speak => {
            if has_cognition_log_seed {
                harness
                    .cognition_log_updated_mailbox()
                    .publish(CognitionLogUpdated::EntryAppended {
                        source: ModuleInstanceId::new(
                            builtin::cognition_gate(),
                            ReplicaIndex::ZERO,
                        ),
                    })
                    .await
                    .expect("module eval failed to publish CognitionLogUpdated");
            }
        }
        ModuleEvalTarget::Predict => {
            if has_cognition_log_seed {
                harness
                    .cognition_log_updated_mailbox()
                    .publish(CognitionLogUpdated::EntryAppended {
                        source: ModuleInstanceId::new(
                            builtin::cognition_gate(),
                            ReplicaIndex::ZERO,
                        ),
                    })
                    .await
                    .expect("module eval failed to publish CognitionLogUpdated");
            }
        }
        ModuleEvalTarget::Surprise => {
            let mut allocation = blackboard.read(|bb| bb.allocation().clone()).await;
            let mut config = allocation.for_module(run_target_module);
            config.guidance = prompt.to_string();
            allocation.set(run_target_module.clone(), config);
            blackboard
                .apply(BlackboardCommand::SetAllocation(allocation))
                .await;
            if has_cognition_log_seed {
                harness
                    .cognition_log_updated_mailbox()
                    .publish(CognitionLogUpdated::EntryAppended {
                        source: ModuleInstanceId::new(
                            builtin::cognition_gate(),
                            ReplicaIndex::ZERO,
                        ),
                    })
                    .await
                    .expect("module eval failed to publish CognitionLogUpdated");
            }
        }
        ModuleEvalTarget::AllocationController => {
            let mut allocation = blackboard.read(|bb| bb.allocation().clone()).await;
            let mut config = allocation.for_module(run_target_module);
            config.guidance = prompt.to_string();
            allocation.set(run_target_module.clone(), config);
            blackboard
                .apply(BlackboardCommand::SetAllocation(allocation))
                .await;
            for record in memo_seed_records {
                harness
                    .memo_updated_mailbox()
                    .publish(nuillu_module::MemoUpdated {
                        owner: record.owner.clone(),
                        index: record.index,
                    })
                    .await
                    .expect("module eval failed to publish MemoUpdated");
            }
            harness
                .memo_updated_mailbox()
                .publish(nuillu_module::MemoUpdated {
                    owner: ModuleInstanceId::new(
                        ModuleId::new("eval-harness").expect("eval-harness id is valid"),
                        ReplicaIndex::ZERO,
                    ),
                    index: 0,
                })
                .await
                .expect("module eval failed to publish MemoUpdated");
        }
    }
}

#[allow(clippy::too_many_arguments)]
async fn activate_module_case_step(
    target: ModuleEvalTarget,
    blackboard: &Blackboard,
    harness: &InternalHarnessIo,
    prompt: &str,
    run_target_module: &ModuleId,
    step: &ModuleEvalStep,
    case_id: &str,
    sensory: &SensoryInputMailbox,
    clock: &dyn Clock,
) {
    let memo_seed_records = seed_memos(blackboard, clock, &step.memos)
        .await
        .expect("module eval step memo seeds should be valid");
    let cognition_seed_records = seed_cognition_log(blackboard, clock, &step.cognition_log).await;
    activate_module_case_target(
        target,
        blackboard,
        harness,
        prompt,
        run_target_module,
        &memo_seed_records,
        &cognition_seed_records,
        !step.cognition_log.is_empty(),
        case_id,
        &step.inputs,
        sensory,
        clock,
    )
    .await;
}

fn cognition_eval_has_new_output(current: &str, baseline: &str) -> bool {
    current != baseline
}

async fn cognition_output_for_module(blackboard: &Blackboard, module: &ModuleId) -> String {
    blackboard
        .read(|bb| {
            bb.cognition_log_set()
                .logs()
                .iter()
                .filter(|record| &record.source.module == module)
                .flat_map(|record| record.entries.iter().map(|entry| entry.text.as_str()))
                .collect::<Vec<_>>()
                .join("\n\n")
        })
        .await
}

#[cfg(test)]
async fn attention_schema_cognition_output(blackboard: &Blackboard) -> String {
    cognition_output_for_module(blackboard, &builtin::attention_schema()).await
}

async fn last_memo_log_content_for_module(
    blackboard: &Blackboard,
    module: &ModuleId,
) -> Option<String> {
    blackboard
        .read(|bb| {
            bb.recent_memo_logs()
                .into_iter()
                .filter(|record| &record.owner.module == module)
                .max_by(|a, b| {
                    a.written_at
                        .cmp(&b.written_at)
                        .then_with(|| a.owner.replica.cmp(&b.owner.replica))
                        .then_with(|| a.index.cmp(&b.index))
                })
                .map(|record| record.content)
        })
        .await
}

async fn allocation_controller_artifact(blackboard: &Blackboard, module: &ModuleId) -> String {
    let memo = last_memo_log_content_for_module(blackboard, module)
        .await
        .unwrap_or_default();
    let allocation = blackboard
        .read(|bb| format_allocation_snapshot(bb.allocation()))
        .await;
    if memo.is_empty() {
        allocation
    } else {
        format!("Controller memo:\n{memo}\n\nAllocation snapshot:\n{allocation}")
    }
}

fn format_allocation_snapshot(allocation: &ResourceAllocation) -> String {
    allocation_observation(allocation)
        .into_iter()
        .map(|(module, obs)| {
            format!(
                "{}: activation_ratio={:.2}, active_replicas={}, guidance={}",
                module,
                obs.activation_ratio.as_f64(),
                obs.active_replicas,
                if obs.guidance.trim().is_empty() {
                    "(none)".to_string()
                } else {
                    obs.guidance.trim().to_string()
                }
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn module_target_uses_memory_store_artifact(target: ModuleEvalTarget) -> bool {
    matches!(
        target,
        ModuleEvalTarget::Memory
            | ModuleEvalTarget::MemoryCompaction
            | ModuleEvalTarget::MemoryAssociation
    )
}

fn module_target_uses_policy_store_artifact(target: ModuleEvalTarget) -> bool {
    matches!(target, ModuleEvalTarget::PolicyCompaction)
}

async fn module_activation_finished(
    blackboard: &Blackboard,
    events: &RecordingRuntimeEventSink,
    module: &ModuleId,
) -> bool {
    let saw_batch = events.snapshot().into_iter().any(|event| {
        matches!(
            event,
            RuntimeEvent::ModuleBatchReady { owner, .. } if owner.module == *module
        )
    });
    if !saw_batch {
        return false;
    }

    blackboard
        .read(|bb| {
            bb.module_status_records()
                .into_iter()
                .find(|record| record.owner.module == *module)
                .is_some_and(|record| {
                    matches!(
                        record.status,
                        ModuleRunStatus::Inactive | ModuleRunStatus::AwaitingBatch
                    )
                })
        })
        .await
}

fn module_activation_completion_count(
    events: &RecordingRuntimeEventSink,
    module: &ModuleId,
) -> usize {
    events
        .snapshot()
        .into_iter()
        .filter(|event| {
            matches!(
                event,
                RuntimeEvent::ModuleActivationCompleted { owner, .. } if owner.module == *module
            )
        })
        .count()
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

#[derive(Debug, Clone, Serialize)]
struct PolicyDiffObservation {
    deleted: Vec<String>,
    remaining: Vec<String>,
}

async fn policy_snapshot(store: &dyn PolicyStore) -> Result<BTreeSet<String>> {
    let mut indexes = BTreeSet::new();
    for rank in [
        PolicyRank::Tentative,
        PolicyRank::Provisional,
        PolicyRank::Established,
        PolicyRank::Habit,
        PolicyRank::Core,
    ] {
        for record in store
            .list_by_rank(rank)
            .await
            .context("list policies for eval snapshot")?
        {
            indexes.insert(record.index.to_string());
        }
    }
    Ok(indexes)
}

fn policy_diff_observation(
    baseline: &BTreeSet<String>,
    current: &BTreeSet<String>,
) -> PolicyDiffObservation {
    PolicyDiffObservation {
        deleted: baseline.difference(current).cloned().collect::<Vec<_>>(),
        remaining: current.iter().cloned().collect::<Vec<_>>(),
    }
}

async fn add_policy_diff_observation(
    artifact: &mut CaseArtifact,
    baseline: &BTreeSet<String>,
    store: &dyn PolicyStore,
) -> Result<()> {
    let current = policy_snapshot(store).await?;
    let diff = policy_diff_observation(baseline, &current);
    let value = serde_json::to_value(diff).context("serialize policy diff observation")?;
    artifact
        .observations
        .insert("policy_diff".to_owned(), value);
    Ok(())
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
        apply_gui_activation(
            &mut allocation,
            builtin::allocation_controller(),
            ActivationRatio::ONE,
        );
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

async fn publish_full_agent_inputs(
    case_id: &str,
    inputs: &[FullAgentInput],
    sensory: &SensoryInputMailbox,
    clock: &dyn Clock,
    visualizer: Option<&VisualizerHook>,
) {
    let now = clock.now();
    for input in inputs {
        let body = match input {
            FullAgentInput::Heard { direction, content } => SensoryInput::OneShot {
                modality: SensoryModality::Audition,
                direction: direction.clone(),
                content: content.content.clone(),
                observed_at: now,
            },
            FullAgentInput::Seen {
                direction,
                appearance,
            } => SensoryInput::OneShot {
                modality: SensoryModality::Vision,
                direction: direction.clone(),
                content: appearance.content.clone(),
                observed_at: now,
            },
            FullAgentInput::OneShot {
                modality,
                direction,
                content,
            } => SensoryInput::OneShot {
                modality: SensoryModality::parse(modality),
                direction: direction.clone(),
                content: content.content.clone(),
                observed_at: now,
            },
        };
        sensory
            .publish(body.clone())
            .await
            .expect("full-agent eval failed to publish SensoryInput");
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
    inputs: &[FullAgentInput],
    steps: &[EvalStep],
    sensory: &SensoryInputMailbox,
    blackboard: &Blackboard,
    utterances: &RecordingUtteranceSink,
    events: &RecordingRuntimeEventSink,
    clock: &dyn Clock,
    visualizer: Option<&VisualizerHook>,
    reporter: &LiveReporter,
    step_failure: &Arc<Mutex<Option<String>>>,
    step_outcomes: &Arc<Mutex<Vec<serde_json::Value>>>,
) {
    if steps.is_empty() {
        publish_full_agent_inputs(case_id, inputs, sensory, clock, visualizer).await;
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
                FULL_AGENT_STEP_SETTLE_TIMEOUT,
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
            }
        }
        publish_full_agent_inputs(case_id, &step.inputs, sensory, clock, visualizer).await;

        let mut wait_outcome = WaitOutcome::Met;
        if let Some(wait_for) = &step.wait_for {
            wait_outcome = wait_for_condition(blackboard, events, wait_for).await;
        }

        let mut check_results: Vec<serde_json::Value> = Vec::new();
        let mut must_pass_failure: Option<String> = None;
        if matches!(wait_outcome, WaitOutcome::Met) && !step.checks.is_empty() {
            let snapshot = build_step_snapshot(blackboard, utterances).await;
            for check in &step.checks {
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
            "elapsed_ms".to_string(),
            serde_json::Value::from(duration_millis_u64(step_started.elapsed())),
        );
        outcome.insert(
            "checks".to_string(),
            serde_json::Value::Array(check_results),
        );
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
            WaitOutcome::Stopped => return,
        }

        if let Some(message) = must_pass_failure {
            step_failure
                .lock()
                .expect("step failure mutex poisoned")
                .get_or_insert(message);
            events.request_stop("step-check-failed");
            return;
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum WaitOutcome {
    Met,
    Timeout,
    Stopped,
}

fn step_settle_modules(wait_for: Option<&WaitFor>) -> Vec<ModuleId> {
    match wait_for {
        Some(WaitFor::MemoFrom { module, .. }) => vec![module.module_id()],
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
    blackboard: &Blackboard,
    events: &RecordingRuntimeEventSink,
    wait_for: &WaitFor,
) -> WaitOutcome {
    match wait_for {
        WaitFor::MemoFrom { module, timeout_ms } => {
            let target = module.module_id();
            let baseline = memo_count_for_module(blackboard, &target).await;
            let deadline = Duration::from_millis(*timeout_ms);
            let start = Instant::now();
            let poll = Duration::from_millis(50);
            loop {
                let count = memo_count_for_module(blackboard, &target).await;
                if count > baseline {
                    return WaitOutcome::Met;
                }
                if events.stop_requested() {
                    return WaitOutcome::Stopped;
                }
                let elapsed = start.elapsed();
                if elapsed >= deadline {
                    return WaitOutcome::Timeout;
                }
                let remaining = deadline.saturating_sub(elapsed);
                tokio::time::sleep(remaining.min(poll)).await;
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
                    return WaitOutcome::Met;
                }
                if events.stop_requested() {
                    return WaitOutcome::Stopped;
                }
                let elapsed = start.elapsed();
                if elapsed >= deadline {
                    return WaitOutcome::Timeout;
                }
                let remaining = deadline.saturating_sub(elapsed);
                tokio::time::sleep(remaining.min(poll)).await;
            }
        }
    }
}

fn wait_for_label(wait_for: Option<&WaitFor>) -> String {
    match wait_for {
        Some(WaitFor::MemoFrom { module, timeout_ms }) => format!(
            "memo from module '{module}' within {timeout_ms}ms",
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

async fn memo_count_for_module(blackboard: &Blackboard, module: &ModuleId) -> usize {
    blackboard
        .read(|bb| {
            bb.recent_memo_logs()
                .into_iter()
                .filter(|record| &record.owner.module == module)
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

fn evaluate_step_check(check: &Check, artifact: &CaseArtifact) -> (bool, Option<String>) {
    match check {
        Check::JsonPointerEquals {
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
        Check::JsonPointerContains {
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
        Check::JsonPointerNumericInRange {
            pointer, min, max, ..
        } => {
            let json = artifact.as_json();
            let actual = pointer_number(&json, pointer);
            numeric_range_outcome(pointer, actual, *min, *max)
        }
        Check::ArtifactTextContains {
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
        Check::ArtifactTextExact { field, exact, .. } => {
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
                    direction: None,
                    content: input.content,
                    observed_at,
                };
                let _ = sensory.publish(body.clone()).await;
                visualizer.send_event(VisualizerEvent::SensoryInput {
                    tab_id,
                    input: body,
                });
            }
            VisualizerCommand::QueryMemory {
                tab_id,
                query,
                limit,
            } if tab_id.as_str() == case_id => {
                let records = memory
                    .search(&MemoryQuery::text(query.clone(), limit))
                    .await
                    .map(|records| records.into_iter().map(memory_record_view).collect())
                    .unwrap_or_default();
                visualizer.send_event(VisualizerEvent::MemoryQueryResult {
                    tab_id,
                    query,
                    records,
                });
            }
            VisualizerCommand::ListMemories {
                tab_id,
                page,
                per_page,
            } if tab_id.as_str() == case_id => {
                let all_records = list_all_visualizer_memories(blackboard, memory).await;
                visualizer.set_memory_cache(case_id, all_records.clone());
                let records = memory_page_from_records(&all_records, page, per_page);
                visualizer.send_event(VisualizerEvent::MemoryPage {
                    tab_id,
                    page: records,
                });
            }
            VisualizerCommand::FetchLinkedMemories {
                tab_id,
                memory_index,
                relation_filter,
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
                        limit,
                    })
                    .await
                    .map(|records| records.into_iter().map(linked_memory_record_view).collect())
                    .unwrap_or_default();
                visualizer.send_event(VisualizerEvent::MemoryLinkedResult {
                    tab_id,
                    memory_index,
                    records,
                });
            }
            VisualizerCommand::DeleteMemory {
                tab_id,
                memory_index,
                page,
                per_page,
            } if tab_id.as_str() == case_id => {
                let index = MemoryIndex::new(memory_index);
                let _ = memory.delete(&index).await;
                blackboard
                    .apply(BlackboardCommand::RemoveMemoryMetadata { index })
                    .await;
                let all_records = list_all_visualizer_memories(blackboard, memory).await;
                visualizer.set_memory_cache(case_id, all_records.clone());
                visualizer.send_event(VisualizerEvent::MemoryPage {
                    tab_id,
                    page: memory_page_from_records(&all_records, page, per_page),
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
            VisualizerCommand::CreateAmbientSensoryRow { tab_id, .. }
            | VisualizerCommand::UpdateAmbientSensoryRow { tab_id, .. }
            | VisualizerCommand::RemoveAmbientSensoryRow { tab_id, .. }
                if tab_id.as_str() == case_id =>
            {
                visualizer.send_event(VisualizerEvent::Log {
                    tab_id,
                    message: "ambient sensory rows are only supported by nuillu-server".to_string(),
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

pub(crate) async fn emit_visualizer_memory_page(
    case_id: &str,
    visualizer: &mut VisualizerHook,
    blackboard: &Blackboard,
    memory: &dyn MemoryStore,
    page: usize,
    per_page: usize,
) {
    let records = list_all_visualizer_memories(blackboard, memory).await;
    visualizer.set_memory_cache(case_id, records.clone());
    let page = memory_page_from_records(&records, page, per_page);
    visualizer.send_event(VisualizerEvent::MemoryPage {
        tab_id: VisualizerTabId::new(case_id.to_string()),
        page,
    });
}

async fn list_all_visualizer_memories(
    blackboard: &Blackboard,
    memory: &dyn MemoryStore,
) -> Vec<MemoryRecordView> {
    let indexes = blackboard
        .read(|bb| {
            let mut records = bb
                .memory_metadata()
                .iter()
                .map(|(index, metadata)| {
                    (index.clone(), metadata.occurred_at, metadata.last_accessed)
                })
                .collect::<Vec<_>>();
            records.sort_by(|left, right| {
                right
                    .1
                    .cmp(&left.1)
                    .then_with(|| right.2.cmp(&left.2))
                    .then_with(|| left.0.as_str().cmp(right.0.as_str()))
            });
            records
                .into_iter()
                .map(|(index, _, _)| index)
                .collect::<Vec<_>>()
        })
        .await;

    let mut records = Vec::new();
    for index in indexes {
        if let Ok(Some(record)) = memory.get(&index).await {
            records.push(memory_record_view(record));
        }
    }

    if records.is_empty() {
        let mut seen = HashSet::new();
        for rank in [
            MemoryRank::Identity,
            MemoryRank::Permanent,
            MemoryRank::LongTerm,
            MemoryRank::MidTerm,
            MemoryRank::ShortTerm,
        ] {
            if let Ok(rank_records) = memory.list_by_rank(rank).await {
                for record in rank_records {
                    if seen.insert(record.index.clone()) {
                        records.push(memory_record_view(record));
                    }
                }
            }
        }
        records.sort_by(|left, right| {
            right
                .occurred_at
                .cmp(&left.occurred_at)
                .then_with(|| left.index.cmp(&right.index))
        });
    }

    records
}

fn memory_page_from_records(
    records: &[MemoryRecordView],
    page: usize,
    per_page: usize,
) -> MemoryPage {
    let total = records.len();
    let start = page.saturating_mul(per_page).min(records.len());
    let end = start.saturating_add(per_page).min(records.len());
    MemoryPage {
        page,
        per_page,
        total,
        records: records[start..end].to_vec(),
    }
}

pub(crate) struct EvalEnvironment {
    pub(crate) blackboard: Blackboard,
    pub(crate) caps: CapabilityProviders,
    pub(crate) memory: Rc<dyn MemoryStore>,
    pub(crate) policy_store: Rc<dyn PolicyStore>,
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
        limits.max_llm_calls,
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
    let clock: Rc<dyn Clock> = match case_now {
        Some(now) => Rc::new(AnchoredRealtimeClock::new(now.with_timezone(&Utc))),
        None => Rc::new(SystemClock),
    };
    let agent_store = connect_agent_store(output_dir, config).await?;
    let memory: Rc<dyn MemoryStore> = Rc::new(agent_store.memory_store());
    let policy_store: Rc<dyn PolicyStore> = Rc::new(agent_store.policy_store());
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
    let tiers = build_tiers(
        &config.cheap_backend,
        &config.default_backend,
        &config.premium_backend,
        &config.llm_concurrency_pool,
        llm_observer,
        Some(eval_llm_log_context(config, case_id)),
        None,
    )
    .map_err(|error| RunnerError::Driver {
        path: output_dir.to_path_buf(),
        message: error.to_string(),
    })?;
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
            cognition_log_port: Rc::new(InMemoryCognitionLogRepository::new()),
            clock: clock.clone(),
            tiers,
        },
        runtime: CapabilityProviderRuntime {
            event_sink: events.clone(),
            policy: runtime_policy,
            ..CapabilityProviderRuntime::default()
        },
    });

    let utterance_sink: Rc<dyn UtteranceSink> = utterances.clone();

    Ok(EvalEnvironment {
        blackboard,
        caps,
        memory,
        policy_store,
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

fn session_compaction_policy(config: &RunnerConfig) -> SessionCompactionPolicy {
    SessionCompactionPolicy::new(
        config.cheap_backend.compaction_input_token_threshold,
        config.default_backend.compaction_input_token_threshold,
        config.premium_backend.compaction_input_token_threshold,
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

pub(crate) fn action_module_ids(modules: &[EvalModule]) -> Vec<ModuleId> {
    modules
        .iter()
        .copied()
        .filter(|module| module.is_action_module())
        .map(EvalModule::module_id)
        .collect()
}

async fn connect_agent_store(output_dir: &Path, config: &RunnerConfig) -> Result<LibsqlAgentStore> {
    let (memory_embedder, memory_profile, memory_dimensions) =
        build_embedder(config.embedding_backend.as_ref(), &config.model_dir)?;
    let (policy_embedder, policy_profile, policy_dimensions) =
        build_embedder(config.embedding_backend.as_ref(), &config.model_dir)?;
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
    clock: &dyn Clock,
    case_now: Option<DateTime<FixedOffset>>,
    memories: &[crate::cases::MemorySeed],
) -> Result<Vec<MemoryIndex>> {
    let writer = memory_caps.writer();
    let mut seeded = Vec::with_capacity(memories.len());
    for memory in memories {
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
    let seeded_indexes = seed_memories(memory_caps, clock, case_now, memories).await?;
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
) -> Result<Vec<nuillu_blackboard::MemoLogRecord>> {
    let now = clock.now();
    let mut records = Vec::new();
    for memo in memos {
        let module = ModuleId::new(memo.module.clone())
            .with_context(|| format!("seed memo module id {}", memo.module))?;
        let owner = ModuleInstanceId::new(module, ReplicaIndex::new(memo.replica));
        let written_at = now - ChronoDuration::seconds(memo.seconds_ago);
        records.push(
            blackboard
                .update_memo(owner, memo.content.content.clone(), written_at)
                .await,
        );
    }
    Ok(records)
}

async fn seed_cognition_log(
    blackboard: &Blackboard,
    clock: &dyn Clock,
    seeds: &[crate::cases::CognitionLogSeed],
) -> Vec<CognitionLogEntryRecord> {
    let stream = ModuleInstanceId::new(builtin::cognition_gate(), ReplicaIndex::ZERO);
    let now = clock.now();
    let mut records = Vec::with_capacity(seeds.len());
    for seed in seeds {
        let appended = blackboard
            .append_cognition_log(
                stream.clone(),
                CognitionLogEntry {
                    at: now - ChronoDuration::seconds(seed.seconds_ago),
                    text: seed.text.content.clone(),
                },
            )
            .await;
        records.push(appended.record);
    }
    records
}

fn full_agent_case_modules(case: &FullAgentCase, disabled: &[EvalModule]) -> Vec<EvalModule> {
    let mut modules = case
        .modules
        .clone()
        .unwrap_or_else(|| DEFAULT_FULL_AGENT_MODULES.to_vec());
    if !disabled.is_empty() {
        modules.retain(|module| !disabled.contains(module));
    }
    modules
}

fn validate_disabled_modules(disabled: &[EvalModule]) -> Result<(), RunnerError> {
    for module in disabled {
        if REQUIRED_FULL_AGENT_MODULES.contains(module) {
            return Err(RunnerError::DisableRequiredModule {
                module: module.as_str(),
            });
        }
    }
    Ok(())
}

fn module_case_modules(target: ModuleEvalTarget, case: &ModuleCase) -> Vec<EvalModule> {
    case.modules
        .clone()
        .unwrap_or_else(|| vec![target.module()])
}

pub(crate) fn eval_registry(
    modules: &[EvalModule],
    memory_caps: &MemoryCapabilities,
    policy_caps: &PolicyCapabilities,
    utterance_sink: &Rc<dyn UtteranceSink>,
    replica_hard_cap: ReplicaHardCap,
) -> ModuleRegistry {
    let mut registry = ModuleRegistry::new();
    for module in modules {
        registry = register_eval_module(
            registry,
            *module,
            modules,
            memory_caps,
            policy_caps,
            utterance_sink,
            replica_hard_cap,
        );
    }
    apply_standard_dependencies(registry, modules.iter().copied().map(EvalModule::module_id))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ReplicaHardCap {
    PolicyMax,
    V1Max,
}

trait EvalRegistryExt {
    fn register_eval<B>(
        self,
        policy: ModulePolicy,
        replica_hard_cap: ReplicaHardCap,
        builder: B,
    ) -> Result<ModuleRegistry, nuillu_module::ModuleRegistryError>
    where
        B: nuillu_module::ModuleRegisterer + 'static;
}

impl EvalRegistryExt for ModuleRegistry {
    fn register_eval<B>(
        self,
        policy: ModulePolicy,
        replica_hard_cap: ReplicaHardCap,
        builder: B,
    ) -> Result<ModuleRegistry, nuillu_module::ModuleRegistryError>
    where
        B: nuillu_module::ModuleRegisterer + 'static,
    {
        let replica_capacity = match replica_hard_cap {
            ReplicaHardCap::PolicyMax => policy.max_active_replicas(),
            ReplicaHardCap::V1Max => ReplicaCapRange::V1_MAX,
        };
        self.register_with_replica_capacity(policy, replica_capacity, builder)
    }
}

fn hidden_from_attention_modules() -> Vec<ModuleId> {
    vec![
        nuillu_types::builtin::interoception(),
        nuillu_types::builtin::homeostatic_controller(),
        nuillu_types::builtin::memory_compaction(),
        nuillu_types::builtin::memory_association(),
        nuillu_types::builtin::memory_recombination(),
    ]
}

fn homeostatic_drive_modules() -> Vec<ModuleId> {
    vec![
        nuillu_types::builtin::memory_compaction(),
        nuillu_types::builtin::memory_association(),
        nuillu_types::builtin::memory_recombination(),
        nuillu_types::builtin::policy_compaction(),
    ]
}

fn sleep_suppressed_modules() -> Vec<ModuleId> {
    vec![
        nuillu_types::builtin::cognition_gate(),
        nuillu_types::builtin::attention_schema(),
        nuillu_types::builtin::self_model(),
        nuillu_types::builtin::query_memory(),
        nuillu_types::builtin::memory(),
        nuillu_types::builtin::policy(),
        nuillu_types::builtin::reward(),
        nuillu_types::builtin::predict(),
        nuillu_types::builtin::surprise(),
        nuillu_types::builtin::speak(),
    ]
}

fn voluntary_modules(modules: &[EvalModule]) -> Vec<ModuleId> {
    let hidden = hidden_from_attention_modules()
        .into_iter()
        .collect::<std::collections::HashSet<_>>();
    modules
        .iter()
        .map(|module| module.module_id())
        .filter(|id| *id != builtin::sensory() && !hidden.contains(id))
        .collect()
}

fn eval_policy(
    replicas_range: std::ops::RangeInclusive<u8>,
    rate_limit_range: std::ops::RangeInclusive<Bpm>,
) -> ModulePolicy {
    ModulePolicy::new(
        ReplicaCapRange::new(*replicas_range.start(), *replicas_range.end()).unwrap(),
        rate_limit_range,
        linear_ratio_fn,
    )
}

fn register_eval_module(
    registry: ModuleRegistry,
    module: EvalModule,
    all_modules: &[EvalModule],
    memory_caps: &MemoryCapabilities,
    policy_caps: &PolicyCapabilities,
    utterance_sink: &Rc<dyn UtteranceSink>,
    replica_hard_cap: ReplicaHardCap,
) -> ModuleRegistry {
    match module {
        // Input-driven and bursty: bursts of inputs need a fast active pace
        // so observations are normalized within the same tick window.
        EvalModule::Sensory => registry
            .register_eval(
                eval_policy(0..=1, Bpm::range(6.0, 18.0)),
                replica_hard_cap,
                |caps| async move {
                    Ok(nuillu_sensory::SensoryModule::new(
                        caps.sensory_input_inbox(),
                        caps.allocation_reader(),
                        caps.memo(),
                        caps.clock(),
                        caps.llm_access(),
                        caps.session("main")
                            .with_auto_compaction(nuillu_sensory::session_auto_compaction())
                            .await?,
                    ))
                },
            )
            .expect("eval module registration should be unique"),
        // Must re-fire fast as memos accumulate so the cognition log is
        // current by the time speak considers it.
        EvalModule::CognitionGate => registry
            .register_eval(
                eval_policy(0..=1, Bpm::range(6.0, 18.0)),
                replica_hard_cap,
                |caps| async move {
                    Ok(nuillu_cognition_gate::CognitionGateModule::new(
                        caps.memo_updated_inbox(),
                        caps.blackboard_reader(),
                        caps.allocation_reader(),
                        caps.cognition_writer(),
                        caps.time_division(),
                        caps.llm_access(),
                        caps.session("main")
                            .with_auto_compaction(nuillu_cognition_gate::session_auto_compaction())
                            .await?,
                    ))
                },
            )
            .expect("eval module registration should be unique"),
        // Expensive (premium tier in default model-set), heavy reasoning.
        // Should only fire on meaningful state shifts — slow base pace so
        // it doesn't burn budget reacting to every memo update.
        EvalModule::AllocationController => registry
            .register_eval(
                eval_policy(0..=1, Bpm::range(3.0, 6.0)),
                replica_hard_cap,
                {
                    let voluntary = voluntary_modules(all_modules);
                    move |caps| {
                        let voluntary = voluntary.clone();
                        async move {
                            Ok(
                                nuillu_allocation_controller::AllocationControllerModule::new(
                                    caps.memo_updated_inbox(),
                                    caps.attention_control_inbox(),
                                    caps.blackboard_reader(),
                                    caps.cognition_log_reader(),
                                    caps.allocation_reader(),
                                    caps.interoception_reader(),
                                    caps.allocation_writer(voluntary.clone(), Vec::new()),
                                    caps.memo(),
                                    caps.llm_access(),
                                    caps.session("main")
                                        .with_auto_compaction(
                                            nuillu_allocation_controller::session_auto_compaction(),
                                        )
                                        .await?,
                                ),
                            )
                        }
                    }
                },
            )
            .expect("eval module registration should be unique"),
        // Periodic first-person attention narration; not on the critical
        // path for the speak loop.
        EvalModule::AttentionSchema => {
            registry
                .register_eval(
                    eval_policy(0..=1, Bpm::range(3.0, 6.0)),
                    replica_hard_cap,
                    |caps| async move {
                        Ok(nuillu_attention_schema::AttentionSchemaModule::new(
                            caps.memo_updated_inbox(),
                            caps.cognition_log_updated_inbox(),
                            caps.blackboard_reader(),
                            caps.allocation_reader(),
                            caps.cognition_log_reader(),
                            caps.cognition_writer(),
                            caps.llm_access(),
                            caps.session("main")
                                .with_auto_compaction(
                                    nuillu_attention_schema::session_auto_compaction(),
                                )
                                .await?,
                        ))
                    },
                )
                .expect("eval module registration should be unique")
        }
        // On-demand: fires on cognition-log updates and reads controller guidance as context.
        EvalModule::SelfModel => registry
            .register_eval(
                eval_policy(0..=1, Bpm::range(3.0, 6.0)),
                replica_hard_cap,
                |caps| async move {
                    Ok(nuillu_self_model::SelfModelModule::new(
                        caps.cognition_log_updated_inbox(),
                        caps.allocation_reader(),
                        caps.blackboard_reader(),
                        caps.cognition_log_reader(),
                        caps.memo(),
                        caps.llm_access(),
                        caps.session("main")
                            .with_auto_compaction(nuillu_self_model::session_auto_compaction())
                            .await?,
                    ))
                },
            )
            .expect("eval module registration should be unique"),
        // Memory retrieval is on the critical path between cognition-gate
        // and speak; needs a quick active pace.
        EvalModule::QueryMemory => registry
            .register_eval(
                eval_policy(0..=1, Bpm::range(6.0, 15.0)),
                replica_hard_cap,
                {
                    let memory_caps = memory_caps.clone();
                    move |caps| {
                        let memory_caps = memory_caps.clone();
                        async move {
                            Ok(nuillu_memory::QueryMemoryModule::new(
                                caps.cognition_log_updated_inbox(),
                                caps.allocation_reader(),
                                caps.blackboard_reader(),
                                memory_caps.retriever(),
                                memory_caps.content_reader(),
                                caps.typed_memo::<nuillu_memory::QueryMemoryMemo>(),
                                caps.llm_access(),
                                caps.session("main")
                                    .with_auto_compaction(
                                        nuillu_memory::query_session_auto_compaction(),
                                    )
                                    .await?,
                            ))
                        }
                    }
                },
            )
            .expect("eval module registration should be unique"),
        // Background durability writer. Cognition-log triggered.
        EvalModule::Memory => registry
            .register_eval(
                eval_policy(0..=1, Bpm::range(6.0, 18.0)),
                replica_hard_cap,
                {
                    let memory_caps = memory_caps.clone();
                    move |caps| {
                        let memory_caps = memory_caps.clone();
                        async move {
                            Ok(nuillu_memory::MemoryModule::new(
                                caps.cognition_log_evicted_inbox(),
                                caps.allocation_reader(),
                                caps.memory_metadata_reader(),
                                memory_caps.writer(),
                                memory_caps.retriever(),
                                caps.llm_access(),
                                caps.session("main")
                                    .with_auto_compaction(nuillu_memory::session_auto_compaction())
                                    .await?,
                            ))
                        }
                    }
                },
            )
            .expect("eval module registration should be unique"),
        // Rare; runs on interoceptive state changes and reads allocation guidance as context.
        EvalModule::MemoryCompaction => registry
            .register_eval(
                eval_policy(0..=1, Bpm::range(2.0, 6.0)),
                replica_hard_cap,
                {
                    let memory_caps = memory_caps.clone();
                    move |caps| {
                        let memory_caps = memory_caps.clone();
                        async move {
                            Ok(nuillu_memory::MemoryCompactionModule::new(
                                caps.interoception_updated_inbox(),
                                caps.allocation_reader(),
                                caps.blackboard_reader(),
                                memory_caps.compactor(),
                                caps.llm_access(),
                            ))
                        }
                    }
                },
            )
            .expect("eval module registration should be unique"),
        EvalModule::MemoryAssociation => registry
            .register_eval(
                eval_policy(0..=1, Bpm::range(2.0, 6.0)),
                replica_hard_cap,
                {
                    let memory_caps = memory_caps.clone();
                    move |caps| {
                        let memory_caps = memory_caps.clone();
                        async move {
                            Ok(nuillu_memory::MemoryAssociationModule::new(
                                caps.interoception_updated_inbox(),
                                caps.allocation_reader(),
                                caps.blackboard_reader(),
                                memory_caps.content_reader(),
                                memory_caps.writer(),
                                memory_caps.associator(),
                                caps.llm_access(),
                            ))
                        }
                    }
                },
            )
            .expect("eval module registration should be unique"),
        EvalModule::MemoryRecombination => registry
            .register_eval(
                eval_policy(0..=1, Bpm::range(2.0, 6.0)),
                replica_hard_cap,
                {
                    let memory_caps = memory_caps.clone();
                    move |caps| {
                        let memory_caps = memory_caps.clone();
                        async move {
                            Ok(nuillu_memory::MemoryRecombinationModule::new(
                                caps.interoception_updated_inbox(),
                                caps.allocation_reader(),
                                caps.blackboard_reader(),
                                memory_caps.retriever(),
                                caps.cognition_writer(),
                                caps.llm_access(),
                            ))
                        }
                    }
                },
            )
            .expect("eval module registration should be unique"),
        EvalModule::Interoception => registry
            .register_eval(
                eval_policy(0..=1, Bpm::range(2.0, 6.0)),
                replica_hard_cap,
                {
                    let suppressed = sleep_suppressed_modules();
                    move |caps| {
                        let suppressed = suppressed.clone();
                        async move {
                            Ok(nuillu_interoception::InteroceptionModule::new(
                                caps.memo_updated_inbox(),
                                caps.cognition_log_updated_inbox(),
                                caps.blackboard_reader(),
                                caps.allocation_writer(Vec::new(), suppressed.clone()),
                                caps.interoception_policy(),
                                caps.interoception_writer(),
                                caps.llm_access(),
                                caps.session("main")
                                    .with_auto_compaction(
                                        nuillu_interoception::session_auto_compaction(),
                                    )
                                    .await?,
                            ))
                        }
                    }
                },
            )
            .expect("eval module registration should be unique"),
        EvalModule::HomeostaticController => registry
            .register_eval(
                eval_policy(0..=1, Bpm::range(6.0, 20.0)),
                replica_hard_cap,
                |caps| async move {
                    Ok(
                        nuillu_homeostatic_controller::HomeostaticControllerModule::new(
                            caps.interoception_updated_inbox(),
                            caps.interoception_reader(),
                            caps.allocation_writer(
                                homeostatic_drive_modules(),
                                sleep_suppressed_modules(),
                            ),
                        ),
                    )
                },
            )
            .expect("eval module registration should be unique"),
        EvalModule::Policy => registry
            .register_eval(
                eval_policy(0..=1, Bpm::range(2.0, 6.0)),
                replica_hard_cap,
                {
                    let policy_caps = policy_caps.clone();
                    move |caps| {
                        let policy_caps = policy_caps.clone();
                        async move {
                            let consideration_writer =
                                policy_caps.consideration_writer(caps.owner().clone());
                            Ok(nuillu_reward::PolicyModule::new(
                                caps.memo_updated_inbox(),
                                caps.cognition_log_updated_inbox(),
                                caps.blackboard_reader(),
                                caps.cognition_log_reader(),
                                caps.allocation_reader(),
                                caps.interoception_reader(),
                                policy_caps.searcher(),
                                caps.memo(),
                                consideration_writer,
                                caps.llm_access(),
                                caps.session("main")
                                    .with_auto_compaction(
                                        nuillu_reward::policy_session_auto_compaction(),
                                    )
                                    .await?,
                            ))
                        }
                    }
                },
            )
            .expect("eval module registration should be unique"),
        EvalModule::PolicyCompaction => registry
            .register_eval(
                eval_policy(0..=1, Bpm::range(2.0, 6.0)),
                replica_hard_cap,
                {
                    let policy_caps = policy_caps.clone();
                    move |caps| {
                        let policy_caps = policy_caps.clone();
                        async move {
                            Ok(nuillu_reward::PolicyCompactionModule::new(
                                caps.interoception_updated_inbox(),
                                caps.allocation_reader(),
                                caps.blackboard_reader(),
                                policy_caps.compactor(),
                                caps.llm_access(),
                            ))
                        }
                    }
                },
            )
            .expect("eval module registration should be unique"),
        EvalModule::Reward => registry
            .register_eval(
                eval_policy(0..=1, Bpm::range(3.0, 9.0)),
                replica_hard_cap,
                {
                    let policy_caps = policy_caps.clone();
                    move |caps| {
                        let policy_caps = policy_caps.clone();
                        async move {
                            Ok(nuillu_reward::RewardModule::new(
                                policy_caps.consideration_evicted_inbox(),
                                caps.blackboard_reader(),
                                caps.cognition_log_reader(),
                                caps.allocation_reader(),
                                caps.interoception_reader(),
                                policy_caps.searcher(),
                                policy_caps.upserter(),
                                caps.memo(),
                                caps.llm_access(),
                                caps.session("main")
                                    .with_auto_compaction(
                                        nuillu_reward::reward_session_auto_compaction(),
                                    )
                                    .await?,
                            ))
                        }
                    }
                },
            )
            .expect("eval module registration should be unique"),
        // Cognition-log triggered; not on speak critical path.
        EvalModule::Predict => registry
            .register_eval(
                eval_policy(0..=1, Bpm::range(6.0, 18.0)),
                replica_hard_cap,
                |caps| async move {
                    Ok(nuillu_predict::PredictModule::new(
                        caps.cognition_log_updated_inbox(),
                        caps.cognition_log_reader(),
                        caps.memo(),
                        caps.llm_access(),
                        caps.session("main")
                            .with_auto_compaction(nuillu_predict::session_auto_compaction())
                            .await?,
                    ))
                },
            )
            .expect("eval module registration should be unique"),
        // Cognition-log triggered; should be quick enough to flag
        // unexpected events while they're still relevant.
        EvalModule::Surprise => registry
            .register_eval(
                eval_policy(0..=1, Bpm::range(6.0, 18.0)),
                replica_hard_cap,
                |caps| async move {
                    Ok(nuillu_surprise::SurpriseModule::new(
                        caps.cognition_log_updated_inbox(),
                        caps.cognition_log_reader(),
                        caps.allocation_reader(),
                        caps.blackboard_reader(),
                        caps.attention_control_mailbox(),
                        caps.memo(),
                        caps.llm_access(),
                        caps.session("main")
                            .with_auto_compaction(nuillu_surprise::session_auto_compaction())
                            .await?,
                    ))
                },
            )
            .expect("eval module registration should be unique"),
        // Reactive on cognition-log updates. The target-selection tool
        // decides whether this activation emits speech or stays silent.
        EvalModule::Speak => registry
            .register_eval(
                eval_policy(0..=1, Bpm::range(6.0, 18.0)),
                replica_hard_cap,
                {
                    let utterance_sink = utterance_sink.clone();
                    move |caps| {
                        let utterance_sink = utterance_sink.clone();
                        async move {
                            Ok(nuillu_speak::SpeakModule::new(
                                caps.cognition_log_updated_inbox(),
                                caps.cognition_log_reader(),
                                caps.memo(),
                                UtteranceWriter::new(
                                    caps.owner().clone(),
                                    caps.blackboard(),
                                    utterance_sink.clone(),
                                    caps.clock(),
                                ),
                                caps.llm_access(),
                                caps.scene_reader(),
                            ))
                        }
                    }
                },
            )
            .expect("eval module registration should be unique"),
    }
}

pub(crate) fn full_agent_allocation(
    _limits: &crate::cases::EvalLimits,
    modules: &[EvalModule],
) -> ResourceAllocation {
    let mut allocation = ResourceAllocation::default();
    allocation.set_activation_table(eval_activation_table());

    for module in modules {
        let (activation, tier) = match module {
            EvalModule::Sensory => (1.0, ModelTier::Cheap),
            EvalModule::CognitionGate => (0.0, ModelTier::Cheap),
            EvalModule::AllocationController => (1.0, ModelTier::Default),
            EvalModule::AttentionSchema => (0.0, ModelTier::Default),
            EvalModule::SelfModel => (0.0, ModelTier::Default),
            EvalModule::QueryMemory => (0.0, ModelTier::Cheap),
            EvalModule::Memory => (0.0, ModelTier::Cheap),
            EvalModule::MemoryCompaction => (0.0, ModelTier::Cheap),
            EvalModule::MemoryAssociation => (0.0, ModelTier::Cheap),
            EvalModule::MemoryRecombination => (0.0, ModelTier::Cheap),
            EvalModule::Interoception => (1.0, ModelTier::Cheap),
            EvalModule::HomeostaticController => (1.0, ModelTier::Cheap),
            EvalModule::Policy => (0.0, ModelTier::Default),
            EvalModule::PolicyCompaction => (0.0, ModelTier::Cheap),
            EvalModule::Reward => (0.0, ModelTier::Default),
            EvalModule::Predict => (0.0, ModelTier::Cheap),
            EvalModule::Surprise => (0.0, ModelTier::Default),
            EvalModule::Speak => (0.0, ModelTier::Premium),
        };
        set_allocation_module(&mut allocation, module.module_id(), activation, tier);
    }
    allocation
}

fn full_agent_gui_initial_allocation(
    limits: &crate::cases::EvalLimits,
    modules: &[EvalModule],
) -> ResourceAllocation {
    let mut allocation = full_agent_allocation(limits, modules);
    for module in modules {
        allocation.set_activation(module.module_id(), ActivationRatio::ZERO);
    }
    allocation
}

fn module_allocation(
    target: ModuleEvalTarget,
    _limits: &crate::cases::EvalLimits,
    modules: &[EvalModule],
) -> ResourceAllocation {
    let mut allocation = ResourceAllocation::default();
    allocation.set_activation_table(eval_activation_table());
    let target_module = target.module();
    for module in modules {
        let is_target = *module == target_module;
        let id = module.module_id();
        allocation.set_model_override(id.clone(), eval_module_tier(*module));
        allocation.set(id.clone(), ModuleConfig::default());
        allocation.set_activation(
            id,
            if is_target {
                ActivationRatio::ONE
            } else {
                ActivationRatio::ZERO
            },
        );
    }
    allocation
}

fn set_allocation_module(
    allocation: &mut ResourceAllocation,
    id: ModuleId,
    activation_ratio: f64,
    tier: ModelTier,
) {
    allocation.set_model_override(id.clone(), tier);
    allocation.set(id.clone(), ModuleConfig::default());
    allocation.set_activation(id, ActivationRatio::from_f64(activation_ratio));
}

fn eval_activation_table() -> Vec<ActivationRatio> {
    [1.0, 0.85, 0.7, 0.5, 0.3, 0.0]
        .into_iter()
        .map(ActivationRatio::from_f64)
        .collect()
}

fn module_id_for_target(target: ModuleEvalTarget) -> ModuleId {
    target.module().module_id()
}

fn eval_module_tier(module: EvalModule) -> ModelTier {
    match module {
        EvalModule::Sensory
        | EvalModule::CognitionGate
        | EvalModule::QueryMemory
        | EvalModule::Memory
        | EvalModule::MemoryCompaction
        | EvalModule::MemoryAssociation
        | EvalModule::MemoryRecombination
        | EvalModule::Interoception
        | EvalModule::HomeostaticController
        | EvalModule::PolicyCompaction
        | EvalModule::Predict => ModelTier::Cheap,
        EvalModule::Speak => ModelTier::Premium,
        EvalModule::AllocationController => ModelTier::Default,
        EvalModule::AttentionSchema
        | EvalModule::SelfModel
        | EvalModule::Policy
        | EvalModule::Reward
        | EvalModule::Surprise => ModelTier::Default,
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

async fn build_full_agent_last_state_dump(
    case_id: &str,
    artifact: &CaseArtifact,
    blackboard: &Blackboard,
    memory: &dyn MemoryStore,
    utterances: &RecordingUtteranceSink,
    event_count: usize,
) -> Result<FullAgentLastStateDump> {
    let (blackboard_dump, memory_metadata) = blackboard
        .read(|bb| {
            (
                blackboard_last_state_dump(bb),
                memory_metadata_dump_records(bb),
            )
        })
        .await;
    let memory_dump = memory_last_state_dump(memory_metadata, memory).await?;
    Ok(FullAgentLastStateDump {
        case: FullAgentLastStateCaseDump {
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
    last_state: &FullAgentLastStateDump,
) -> Result<()> {
    let value = serde_json::to_value(last_state).context("serialize last state observation")?;
    artifact
        .observations
        .insert("last_state".to_string(), value);
    Ok(())
}

fn write_full_agent_last_state_eure(
    output_dir: &Path,
    last_state: FullAgentLastStateDump,
) -> Result<()> {
    let path = output_dir.join("last-state.eure");
    let rendered = render_full_agent_last_state_eure(last_state)
        .context("render full-agent last state Eure")?;
    std::fs::write(&path, rendered)
        .with_context(|| format!("write full-agent last state dump to {}", path.display()))
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
            .map(|module| AllocationView {
                bpm: ModuleId::new(module.module.clone())
                    .ok()
                    .and_then(|id| bb.allocation().cooldown_for(&id))
                    .and_then(bpm_from_cooldown),
                cooldown_ms: ModuleId::new(module.module.clone())
                    .ok()
                    .and_then(|id| bb.allocation().cooldown_for(&id))
                    .map(duration_millis_u64),
                module: module.module,
                activation_ratio: module.activation_ratio,
                active_replicas: module.active_replicas,
                tier: module.tier,
                guidance: module.guidance.as_str().to_owned(),
            })
            .collect(),
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

fn memo_log_dumps(bb: &BlackboardInner) -> Vec<MemoLogDump> {
    bb.recent_memo_logs()
        .into_iter()
        .map(|record| MemoLogDump {
            module: record.owner.module.as_str().to_owned(),
            replica: record.owner.replica.get(),
            index: record.index,
            written_at: record.written_at.to_rfc3339(),
            content: DumpText::new(record.content),
        })
        .collect()
}

fn allocation_module_dumps(allocation: &ResourceAllocation) -> Vec<AllocationModuleDump> {
    let mut modules = allocation
        .iter()
        .map(|(module, config)| AllocationModuleDump {
            module: module.as_str().to_owned(),
            activation_ratio: allocation.activation_for(module).as_f64(),
            active_replicas: allocation.active_replicas(module),
            cooldown_ms: allocation.cooldown_for(module).map(duration_millis_u64),
            tier: model_tier_name(allocation.tier_for(module)).to_owned(),
            guidance: DumpText::new(config.guidance.clone()),
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
    cooldown_ms: Option<u64>,
    guidance: String,
    tier: ModelTier,
}

#[derive(Debug, Clone, Serialize)]
struct MemoLogObservation {
    replica: u8,
    index: u64,
    written_at: String,
    content: String,
}

#[derive(Debug, Clone, Serialize)]
struct CognitionLogObservation {
    source: ModuleInstanceObservation,
    entries: Vec<CognitionLogEntry>,
}

#[derive(Debug, Clone, Serialize)]
struct ModuleInstanceObservation {
    module: String,
    replica: u8,
}

#[derive(Debug, Clone, Serialize)]
struct ActiveModuleObservation {
    module: String,
    active_replicas: u8,
    activation_ratio: ActivationRatio,
    tier: ModelTier,
}

fn memo_log_observations(bb: &BlackboardInner) -> BTreeMap<String, Vec<MemoLogObservation>> {
    let mut logs = BTreeMap::<String, Vec<MemoLogObservation>>::new();
    for record in bb.recent_memo_logs() {
        logs.entry(record.owner.module.as_str().to_owned())
            .or_default()
            .push(MemoLogObservation {
                replica: record.owner.replica.get(),
                index: record.index,
                written_at: record.written_at.to_rfc3339(),
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
        .iter()
        .map(|(module, config)| {
            (
                module.as_str().to_owned(),
                AllocationModuleObservation {
                    activation_ratio: allocation.activation_for(module),
                    active_replicas: allocation.active_replicas(module),
                    cooldown_ms: allocation.cooldown_for(module).map(duration_millis_u64),
                    guidance: config.guidance.clone(),
                    tier: allocation.tier_for(module),
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
        .chain(bb.allocation().iter().map(|(module, _)| module.clone()))
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
                tier: bb.allocation().tier_for(&module),
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
        .map(|(module, obs)| {
            format!(
                "{}:{:.2}/{:?}",
                module,
                obs.activation_ratio.as_f64(),
                obs.tier
            )
        })
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
                "{}:{}:{:.2}/{:?}",
                module.module,
                module.active_replicas,
                module.activation_ratio.as_f64(),
                module.tier
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
        RuntimeEvent::RateLimitDelayed {
            sequence,
            owner,
            capability,
            delayed_for,
        } => format!(
            "seq={sequence} rate_limit_delayed owner={owner} capability={capability:?} delayed_for_ms={}",
            duration_millis_u64(*delayed_for)
        ),
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
            owner,
            batch_type,
            ..
        } => format!("seq={sequence} module_batch_ready owner={owner} batch={batch_type}"),
        RuntimeEvent::ModuleActivationCompleted {
            sequence,
            owner,
            duration,
            succeeded,
        } => format!(
            "seq={sequence} module_activation_completed owner={owner} duration_ms={} succeeded={succeeded}",
            duration_millis_u64(*duration)
        ),
        RuntimeEvent::ModuleTaskFailed {
            sequence,
            owner,
            phase,
            message,
        } => format!(
            "seq={sequence} module_task_failed owner={owner} phase={phase} message={message}"
        ),
        RuntimeEvent::ModuleRestarted {
            sequence,
            owner,
            consecutive_failures,
            failure_limit,
        } => format!(
            "seq={sequence} module_restarted owner={owner} failures={consecutive_failures} limit={failure_limit}"
        ),
        RuntimeEvent::ModuleStopped {
            sequence,
            owner,
            phase,
            message,
            consecutive_failures,
        } => format!(
            "seq={sequence} module_stopped owner={owner} phase={phase} failures={consecutive_failures} message={message}"
        ),
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
}

impl std::fmt::Debug for LiveReporter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LiveReporter")
            .field("run_id", &self.run_id)
            .field("path", &self.path)
            .field("log_prefix", &self.log_prefix)
            .field("log_scope", &self.log_scope)
            .finish_non_exhaustive()
    }
}

impl LiveReporter {
    pub(crate) fn new(run_id: &str, run_dir: &Path) -> Result<Self, RunnerError> {
        Self::new_with_log_context(run_id, run_dir, "eval", "case")
    }

    pub(crate) fn new_with_log_context(
        run_id: &str,
        run_dir: &Path,
        log_prefix: &str,
        log_scope: &str,
    ) -> Result<Self, RunnerError> {
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
        eprintln!("{live_message}");
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

#[derive(Debug, Clone, Serialize)]
struct RecordedUtterance {
    sender: String,
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

#[derive(Debug, Clone)]
struct FullAgentSettleTracker {
    last_progress_count: usize,
    last_progress_at: Instant,
}

impl FullAgentSettleTracker {
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

fn full_agent_ready_to_score_at(
    actions: &ActionActivityTracker,
    settle: &FullAgentSettleTracker,
    llm_in_flight: usize,
    input_phase_finished: bool,
    allow_empty_output: bool,
    step_driven_case: bool,
    now: Instant,
) -> bool {
    if !settle.runtime_silence_elapsed_at(FULL_AGENT_RUNTIME_SILENCE_WINDOW, llm_in_flight, now) {
        return false;
    }
    actions.silence_window_elapsed_at(FULL_AGENT_ACTION_SILENCE_WINDOW, now)
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
}

#[async_trait(?Send)]
impl UtteranceSink for RecordingUtteranceSink {
    async fn on_complete(&self, utterance: Utterance) -> Result<(), PortError> {
        let sender_module = utterance.sender.module.clone();
        let recorded = RecordedUtterance {
            sender: utterance.sender.to_string(),
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
    case_started: Instant,
    progress_events: AtomicUsize,
    llm_in_flight: AtomicUsize,
    stop: AtomicBool,
    case_id: String,
    max_llm_calls: Option<u64>,
    reporter: LiveReporter,
    visualizer: Option<VisualizerEventSink>,
}

impl RecordingRuntimeEventSink {
    fn new(
        case_id: String,
        max_llm_calls: Option<u64>,
        reporter: LiveReporter,
        visualizer: Option<VisualizerEventSink>,
    ) -> Self {
        Self {
            events: Mutex::new(Vec::new()),
            timed_events: Mutex::new(Vec::new()),
            case_started: Instant::now(),
            progress_events: AtomicUsize::new(0),
            llm_in_flight: AtomicUsize::new(0),
            stop: AtomicBool::new(false),
            case_id,
            max_llm_calls,
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

fn scheduled_wait_remaining_from_timed_events(
    timed_events: &[(u64, RuntimeEvent)],
    elapsed: Duration,
) -> Option<Duration> {
    let elapsed_ms = duration_millis_u64(elapsed);
    timed_events
        .iter()
        .filter_map(|(offset_ms, event)| {
            let delayed_for = match event {
                RuntimeEvent::RateLimitDelayed { delayed_for, .. }
                | RuntimeEvent::ModuleBatchThrottled { delayed_for, .. } => *delayed_for,
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
        RuntimeEvent::LlmAccessed { .. }
        | RuntimeEvent::LlmCompleted { .. }
        | RuntimeEvent::MemoUpdated { .. }
        | RuntimeEvent::SessionCompactionStarted { .. }
        | RuntimeEvent::SessionCompactionCompleted { .. }
        | RuntimeEvent::SessionCompactionFailed { .. }
        | RuntimeEvent::ModuleTaskFailed { .. }
        | RuntimeEvent::ModuleRestarted { .. }
        | RuntimeEvent::ModuleStopped { .. } => true,
        RuntimeEvent::RateLimitDelayed { .. }
        | RuntimeEvent::ModuleBatchThrottled { .. }
        | RuntimeEvent::ModuleBatchReady { .. }
        | RuntimeEvent::ModuleActivationCompleted { .. } => false,
    }
}

impl RuntimeEventSink for RecordingRuntimeEventSink {
    fn on_event(&self, event: RuntimeEvent) -> Result<(), PortError> {
        let should_stop = match &event {
            RuntimeEvent::LlmAccessed { call, .. } => self
                .max_llm_calls
                .is_some_and(|max| call.saturating_add(1) >= max),
            RuntimeEvent::LlmCompleted { .. } => false,
            RuntimeEvent::MemoUpdated { .. } => false,
            RuntimeEvent::RateLimitDelayed { .. } => false,
            RuntimeEvent::ModuleBatchThrottled { .. } => false,
            RuntimeEvent::ModuleBatchReady { .. } => false,
            RuntimeEvent::ModuleActivationCompleted { .. } => false,
            RuntimeEvent::ModuleTaskFailed { .. } => false,
            RuntimeEvent::ModuleRestarted { .. } => false,
            RuntimeEvent::ModuleStopped { .. } => false,
            RuntimeEvent::SessionCompactionStarted { .. } => false,
            RuntimeEvent::SessionCompactionCompleted { .. } => false,
            RuntimeEvent::SessionCompactionFailed { .. } => false,
        };
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
            RuntimeEvent::RateLimitDelayed {
                owner,
                capability,
                delayed_for,
                ..
            } => format!(
                "{} rate-limit-delayed {} owner={} capability={:?} delayed_ms={}",
                self.reporter.log_prefix(),
                self.reporter.log_scope(&self.case_id),
                owner,
                capability,
                delayed_for.as_millis()
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
                owner,
                batch_type,
                batch_debug,
                ..
            } => format!(
                "{} module-batch-ready {} owner={} type={} chars={}",
                self.reporter.log_prefix(),
                self.reporter.log_scope(&self.case_id),
                owner,
                batch_type,
                batch_debug.chars().count()
            ),
            RuntimeEvent::ModuleActivationCompleted {
                owner,
                duration,
                succeeded,
                ..
            } => format!(
                "{} module-activation-completed {} owner={} duration_ms={} succeeded={}",
                self.reporter.log_prefix(),
                self.reporter.log_scope(&self.case_id),
                owner,
                duration.as_millis(),
                succeeded
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
            RuntimeEvent::ModuleRestarted {
                owner,
                consecutive_failures,
                failure_limit,
                ..
            } => format!(
                "{} module-restarted {} owner={} failures={} limit={}",
                self.reporter.log_prefix(),
                self.reporter.log_scope(&self.case_id),
                owner,
                consecutive_failures,
                failure_limit
            ),
            RuntimeEvent::ModuleStopped {
                owner,
                phase,
                message,
                consecutive_failures,
                ..
            } => format!(
                "{} module-stopped {} owner={} phase={} failures={} error={}",
                self.reporter.log_prefix(),
                self.reporter.log_scope(&self.case_id),
                owner,
                phase,
                consecutive_failures,
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
        self.reporter.emit_port(
            Some(&self.case_id),
            "runtime_event",
            serde_json::json!({ "event": event }),
            live_message,
        )?;
        if let Some(visualizer) = &self.visualizer {
            visualizer.send(VisualizerEvent::RuntimeEvent {
                tab_id: VisualizerTabId::new(self.case_id.clone()),
                event: event.clone(),
            });
            if let RuntimeEvent::ModuleTaskFailed {
                owner,
                phase,
                message,
                ..
            } = &event
            {
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
        }
        if should_stop && !self.stop.swap(true, Ordering::Relaxed) {
            self.reporter.emit_port(
                Some(&self.case_id),
                "stop_requested",
                serde_json::json!({ "reason": "max-llm-calls" }),
                format!(
                    "{} stop requested {} reason=max-llm-calls",
                    self.reporter.log_prefix(),
                    self.reporter.log_scope(&self.case_id)
                ),
            )?;
        }
        Ok(())
    }
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

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::path::Path;
    use std::sync::Arc;

    use crate::evaluation::KMetricReport;
    use chrono::TimeZone as _;
    use lutum::{
        FinishReason, Lutum, MockLlmAdapter, MockTextScenario, RawTextTurnEvent,
        SharedPoolBudgetManager, SharedPoolBudgetOptions, Usage,
    };
    use nuillu_blackboard::{BlackboardCommand, CognitionLogEntry, MemoryMetaPatch};
    use nuillu_memory::NoopMemoryStore;
    use nuillu_module::ports::{NoopCognitionLogRepository, SystemClock};
    use nuillu_module::{LlmConcurrencyPool, LutumTiers, MemoUpdated};
    use nuillu_types::{MemoryIndex, ModuleInstanceId, ReplicaIndex};

    use super::*;

    struct FixedClock(chrono::DateTime<Utc>);

    #[async_trait::async_trait(?Send)]
    impl Clock for FixedClock {
        fn now(&self) -> chrono::DateTime<Utc> {
            self.0
        }

        async fn sleep_until(&self, _deadline: chrono::DateTime<Utc>) {
            // Test clock: sleeps complete immediately so test wall-clock stays
            // independent of the registered BPM cooldown ranges.
        }
    }

    fn test_backend_config() -> LlmBackendConfig {
        test_backend_config_with_model("gpt-oss:20b")
    }

    fn test_backend_config_with_model(model: &str) -> LlmBackendConfig {
        LlmBackendConfig {
            model_key: model.to_string(),
            endpoint: "http://localhost:11434/v1".to_string(),
            token: "local".to_string(),
            model: model.to_string(),
            reasoning_effort: None,
            use_responses_api: false,
            compaction_input_token_threshold: 16_000,
            max_concurrent_llm_calls: None,
        }
    }

    fn test_model_concurrency() -> BTreeMap<String, Option<NonZeroUsize>> {
        BTreeMap::from([
            ("judge-model".to_string(), None),
            ("cheap-model".to_string(), None),
            ("default-model".to_string(), None),
            ("premium-model".to_string(), None),
        ])
    }

    fn test_runner_config(dir: &Path) -> RunnerConfig {
        RunnerConfig {
            cases_root: dir.join("eval-cases"),
            output_root: dir.join("out"),
            llm_log_root: dir.join("llm-logs"),
            run_id: "run-1".to_string(),
            judge_backend: test_backend_config_with_model("judge-model"),
            cheap_backend: test_backend_config_with_model("cheap-model"),
            default_backend: test_backend_config_with_model("default-model"),
            premium_backend: test_backend_config_with_model("premium-model"),
            model_dir: dir.join("missing-model"),
            embedding_backend: None,
            fail_fast: false,
            failed_only: false,
            failed_from: None,
            model_concurrency: test_model_concurrency(),
            llm_concurrency_pool: LlmConcurrencyPool::default(),
            trials: NonZeroUsize::new(1).unwrap(),
            case_patterns: Vec::new(),
            module_filters: Vec::new(),
            disabled_modules: Vec::new(),
            exclude_full_agent: false,
            full_agent_only: false,
        }
    }

    fn test_suite_run_report(trials: usize) -> SuiteRunReport {
        SuiteRunReport {
            run_id: "run".to_string(),
            cases_root: "eval-cases".to_string(),
            output_dir: "out/run".to_string(),
            case_patterns: Vec::new(),
            failed_only: false,
            failed_from: None,
            fail_fast: false,
            model_concurrency: BTreeMap::new(),
            trials,
            planned_case_count: 0,
            models: SuiteModelNames {
                judge: "judge".to_string(),
                cheap: "cheap".to_string(),
                default: "default".to_string(),
                premium: "premium".to_string(),
            },
            module_filters: Vec::new(),
            disabled_modules: Vec::new(),
            exclude_full_agent: false,
            full_agent_only: false,
        }
    }

    fn test_report(passed: bool, invalid: bool, score: f64) -> CaseReport {
        CaseReport {
            runtime_failure: invalid.then(|| "invalid".to_string()),
            checks: Vec::new(),
            modules_checks: Vec::new(),
            invalid,
            must_pass_ok: passed,
            weighted_points_earned: 0,
            weighted_points_total: 0,
            score,
        }
    }

    fn test_case_summary(
        id: &str,
        trial_count: usize,
        passed_trials: usize,
        invalid_trials: usize,
        score: f64,
    ) -> CaseSummary {
        let passed = passed_trials == trial_count;
        let invalid = invalid_trials > 0;
        let report = test_report(passed, invalid, score);
        CaseSummary {
            path: format!("{id}.eure"),
            id: id.to_string(),
            description: None,
            passed,
            invalid,
            score,
            report,
            timing: CaseTiming { elapsed_ms: 0 },
            trial_timing: None,
            activations: Vec::new(),
            trial_count,
            passed_trials,
            failed_trials: trial_count.saturating_sub(passed_trials + invalid_trials),
            invalid_trials,
            trials: Vec::new(),
        }
    }

    fn test_case_run_output(
        id: &str,
        trial: usize,
        passed: bool,
        invalid: bool,
        score: f64,
    ) -> CaseRunOutput {
        let report = test_report(passed, invalid, score);
        let summary = CaseSummary {
            path: format!("{id}.eure"),
            id: id.to_string(),
            description: None,
            passed,
            invalid,
            score,
            report: report.clone(),
            timing: CaseTiming { elapsed_ms: 0 },
            trial_timing: None,
            activations: Vec::new(),
            trial_count: 1,
            passed_trials: usize::from(passed),
            failed_trials: usize::from(!passed && !invalid),
            invalid_trials: usize::from(invalid),
            trials: vec![CaseTrialSummary {
                trial,
                output_dir: format!("out/{id}/{}", trial_dir_name(trial)),
                path: format!("{id}.eure"),
                id: id.to_string(),
                description: None,
                passed,
                invalid,
                score,
                report,
                timing: CaseTiming { elapsed_ms: 0 },
            }],
        };
        CaseRunOutput {
            case_path: PathBuf::from(format!("{id}.eure")),
            output_dir: PathBuf::from(format!("out/{id}/{}", trial_dir_name(trial))),
            summary,
            artifact: CaseArtifact::new(""),
            events: Vec::new(),
            trace: empty_trace_snapshot(),
            raw_trace: RawTraceSnapshot::default(),
        }
    }

    fn assert_metric_values(actual: &[KMetricReport], expected: &[(usize, f64)]) {
        assert_eq!(actual.len(), expected.len());
        for (actual, (k, value)) in actual.iter().zip(expected.iter()) {
            assert_eq!(actual.k, *k);
            assert!(
                (actual.value - value).abs() < 1e-12,
                "k={} expected {} got {}",
                k,
                value,
                actual.value
            );
        }
    }

    fn write_query_memory_case(root: &Path, name: &str, id: &str) -> PathBuf {
        let case_dir = root.join("eval-cases/modules/query-memory");
        std::fs::create_dir_all(&case_dir).unwrap();
        let path = case_dir.join(format!("{name}.eure"));
        std::fs::write(
            &path,
            format!(
                r#"
id = "{id}"
prompt = "Find memory."
"#
            ),
        )
        .unwrap();
        path
    }

    fn write_module_case(
        root: &Path,
        target: EvalModule,
        name: &str,
        id: &str,
        modules: &[EvalModule],
    ) -> PathBuf {
        let case_dir = root.join("eval-cases/modules").join(target.as_str());
        std::fs::create_dir_all(&case_dir).unwrap();
        let path = case_dir.join(format!("{name}.eure"));
        std::fs::write(
            &path,
            format!(
                r#"
id = "{id}"
modules = [{modules}]
prompt = "Run module."
"#,
                modules = module_list(modules),
            ),
        )
        .unwrap();
        path
    }

    fn write_full_agent_case(
        root: &Path,
        name: &str,
        id: &str,
        modules: Option<&[EvalModule]>,
    ) -> PathBuf {
        let case_dir = root.join("eval-cases/full-agent");
        std::fs::create_dir_all(&case_dir).unwrap();
        let path = case_dir.join(format!("{name}.eure"));
        let modules = modules
            .map(|modules| format!("modules = [{}]\n", module_list(modules)))
            .unwrap_or_default();
        std::fs::write(
            &path,
            format!(
                r#"
id = "{id}"
{modules}
@ inputs[] {{
  $variant: heard
  content = "Hello?"
}}
"#
            ),
        )
        .unwrap();
        path
    }

    fn module_list(modules: &[EvalModule]) -> String {
        modules
            .iter()
            .map(|module| format!(r#""{}""#, module.as_str()))
            .collect::<Vec<_>>()
            .join(", ")
    }

    fn write_suite_report(path: &Path, cases: serde_json::Value) {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(
            path,
            serde_json::to_vec_pretty(&serde_json::json!({ "cases": cases })).unwrap(),
        )
        .unwrap();
    }

    #[test]
    fn eval_llm_log_context_uses_run_and_case_namespace() {
        let dir = tempfile::tempdir().unwrap();
        let config = test_runner_config(dir.path());

        let context = eval_llm_log_context(&config, "case-1");

        assert_eq!(context.root, dir.path().join("llm-logs"));
        assert_eq!(context.namespace, vec!["run-1", "case-1"]);
    }

    #[test]
    fn aggregate_suite_computes_trial_metrics_from_case_counts() {
        let report = aggregate_suite(
            test_suite_run_report(3),
            vec![
                test_case_summary("case-one", 3, 2, 0, 0.8),
                test_case_summary("case-two", 3, 1, 0, 0.4),
            ],
        );

        assert_metric_values(
            &report.metrics.pass_at,
            &[(1, 0.5), (2, 5.0 / 6.0), (3, 1.0)],
        );
        assert_metric_values(
            &report.metrics.pass_hat,
            &[(1, 0.5), (2, 1.0 / 6.0), (3, 0.0)],
        );
    }

    #[test]
    fn aggregate_case_summary_requires_all_trials_to_pass() {
        let dir = tempfile::tempdir().unwrap();
        let case_path = write_query_memory_case(dir.path(), "case", "case-id");
        let case = parse_case_file(&case_path).unwrap();

        let all_pass = aggregate_case_summary(
            &case_path,
            &case,
            "case-id",
            &[
                test_case_run_output("case-id", 1, true, false, 1.0),
                test_case_run_output("case-id", 2, true, false, 0.8),
            ],
            0,
        );
        assert!(all_pass.passed);
        assert!(!all_pass.invalid);
        assert_eq!(all_pass.trial_count, 2);
        assert_eq!(all_pass.passed_trials, 2);
        assert_eq!(all_pass.failed_trials, 0);
        assert_eq!(all_pass.invalid_trials, 0);
        assert_eq!(all_pass.trials.len(), 2);

        let some_fail = aggregate_case_summary(
            &case_path,
            &case,
            "case-id",
            &[
                test_case_run_output("case-id", 1, true, false, 1.0),
                test_case_run_output("case-id", 2, false, false, 0.2),
            ],
            0,
        );
        assert!(!some_fail.passed);
        assert!(!some_fail.invalid);
        assert_eq!(some_fail.passed_trials, 1);
        assert_eq!(some_fail.failed_trials, 1);
        assert_eq!(some_fail.invalid_trials, 0);

        let invalid = aggregate_case_summary(
            &case_path,
            &case,
            "case-id",
            &[
                test_case_run_output("case-id", 1, true, false, 1.0),
                test_case_run_output("case-id", 2, false, true, 0.0),
            ],
            0,
        );
        assert!(!invalid.passed);
        assert!(invalid.invalid);
        assert_eq!(invalid.passed_trials, 1);
        assert_eq!(invalid.failed_trials, 0);
        assert_eq!(invalid.invalid_trials, 1);
    }

    #[test]
    fn aggregate_case_summary_records_trial_timing() {
        let dir = tempfile::tempdir().unwrap();
        let case_path = write_query_memory_case(dir.path(), "case", "case-id");
        let case = parse_case_file(&case_path).unwrap();

        let mut first = test_case_run_output("case-id", 1, true, false, 1.0);
        first.summary.timing = CaseTiming { elapsed_ms: 100 };
        first.summary.trials[0].timing = CaseTiming { elapsed_ms: 100 };
        let mut second = test_case_run_output("case-id", 2, true, false, 0.8);
        second.summary.timing = CaseTiming { elapsed_ms: 300 };
        second.summary.trials[0].timing = CaseTiming { elapsed_ms: 300 };

        let summary = aggregate_case_summary(&case_path, &case, "case-id", &[first, second], 450);

        assert_eq!(summary.timing.elapsed_ms, 450);
        let trial_timing = summary.trial_timing.expect("trial timing");
        assert_eq!(trial_timing.min_ms, 100);
        assert_eq!(trial_timing.max_ms, 300);
        assert_eq!(trial_timing.mean_ms, 200);
    }

    #[test]
    fn visualizer_open_tab_uses_case_id_as_tab_id() {
        let (event_tx, event_rx) = std::sync::mpsc::channel();
        let (_command_tx, command_rx) = std::sync::mpsc::channel();
        let hooks = RunnerHooks::with_visualizer(VisualizerHook::new(event_tx, command_rx));

        emit_visualizer_open_tab(&hooks, "case-1");

        let VisualizerServerMessage::Event { event } = event_rx.recv().expect("visualizer event")
        else {
            panic!("expected visualizer event");
        };
        let VisualizerEvent::OpenTab { tab_id, title } = event else {
            panic!("expected open-tab event");
        };
        assert_eq!(tab_id.as_str(), "case-1");
        assert_eq!(title, "case-1");
    }

    #[test]
    fn visualizer_hook_serves_cached_memory_query_results() {
        let (event_tx, event_rx) = std::sync::mpsc::channel();
        let (command_tx, command_rx) = std::sync::mpsc::channel();
        let mut hook = VisualizerHook::new(event_tx, command_rx);
        hook.set_memory_cache(
            "case-1",
            vec![MemoryRecordView {
                index: "m1".to_string(),
                kind: "Statement".to_string(),
                rank: "long-term".to_string(),
                occurred_at: None,
                stored_at: Utc::now(),
                concepts: Vec::new(),
                tags: Vec::new(),
                affect_arousal: 0.0,
                valence: 0.0,
                emotion: String::new(),
                content: "rust memory".to_string(),
            }],
        );

        command_tx
            .send(VisualizerClientMessage::Command {
                command: VisualizerCommand::QueryMemory {
                    tab_id: VisualizerTabId::new("case-1"),
                    query: "rust".to_string(),
                    limit: 10,
                },
            })
            .unwrap();
        command_tx
            .send(VisualizerClientMessage::Command {
                command: VisualizerCommand::Shutdown,
            })
            .unwrap();
        hook.drain_cached_commands_until_shutdown();

        let VisualizerServerMessage::Event { event } = event_rx.recv().expect("memory query event")
        else {
            panic!("expected visualizer event");
        };
        let VisualizerEvent::MemoryQueryResult { records, .. } = event else {
            panic!("expected memory query result");
        };
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].content, "rust memory");
    }

    fn test_caps(blackboard: Blackboard) -> CapabilityProviders {
        test_caps_with_adapter(blackboard, MockLlmAdapter::new())
    }

    fn test_caps_with_adapter(
        blackboard: Blackboard,
        adapter: MockLlmAdapter,
    ) -> CapabilityProviders {
        let adapter = Arc::new(adapter);
        let budget = SharedPoolBudgetManager::new(SharedPoolBudgetOptions::default());
        let lutum = Lutum::new(adapter, budget);
        CapabilityProviders::new(CapabilityProviderPorts {
            blackboard,
            cognition_log_port: Rc::new(NoopCognitionLogRepository),
            clock: Rc::new(SystemClock),
            tiers: LutumTiers::from_shared_lutum(lutum),
        })
    }

    fn attention_schema_tool_scenario(tool_call_id: &str, text: &str) -> MockTextScenario {
        MockTextScenario::events(vec![
            Ok(RawTextTurnEvent::Started {
                request_id: Some("attention-schema".into()),
                model: "mock".into(),
            }),
            Ok(RawTextTurnEvent::ToolCallChunk {
                id: tool_call_id.into(),
                name: "append_attention_experience".into(),
                arguments_json_delta: serde_json::json!({ "plaintext": text }).to_string(),
            }),
            Ok(RawTextTurnEvent::Completed {
                request_id: Some("attention-schema".into()),
                finish_reason: FinishReason::ToolCall,
                usage: Usage::zero(),
            }),
        ])
    }

    fn attention_schema_no_tool_scenario() -> MockTextScenario {
        MockTextScenario::events(vec![
            Ok(RawTextTurnEvent::Started {
                request_id: Some("attention-schema-noop".into()),
                model: "mock".into(),
            }),
            Ok(RawTextTurnEvent::ToolCallChunk {
                id: "leave-attention-unchanged".into(),
                name: "leave_attention_unchanged".into(),
                arguments_json_delta: serde_json::json!({
                    "reason": "no new attention experience"
                })
                .to_string(),
            }),
            Ok(RawTextTurnEvent::Completed {
                request_id: Some("attention-schema".into()),
                finish_reason: FinishReason::ToolCall,
                usage: Usage::zero(),
            }),
        ])
    }

    #[test]
    fn case_patterns_match_case_id_or_path_substrings() {
        let dir = tempfile::tempdir().unwrap();
        let case_dir = dir.path().join("eval-cases/modules/query-memory");
        std::fs::create_dir_all(&case_dir).unwrap();
        let first = case_dir.join("first-route.eure");
        let second = case_dir.join("second-memory.eure");
        std::fs::write(
            &first,
            r#"
id = "module-query-memory-first-route"
prompt = "First?"
"#,
        )
        .unwrap();
        std::fs::write(
            &second,
            r#"
id = "module-query-memory-special-memory"
prompt = "Second?"
"#,
        )
        .unwrap();

        let by_path = filter_case_paths(
            vec![first.clone(), second.clone()],
            &["first-route".to_string()],
        )
        .unwrap();
        assert_eq!(by_path, vec![first.clone()]);

        let by_id = filter_case_paths(vec![first, second.clone()], &["special-memory".to_string()])
            .unwrap();
        assert_eq!(by_id, vec![second]);
    }

    #[test]
    fn module_filters_select_full_agent_membership_or_module_targets() {
        let dir = tempfile::tempdir().unwrap();
        let full_speak = write_full_agent_case(
            dir.path(),
            "full-speak",
            "full-agent-speak",
            Some(&[EvalModule::Sensory, EvalModule::Speak]),
        );
        let full_cognition_gate = write_full_agent_case(
            dir.path(),
            "full-cognition-gate",
            "full-agent-cognition-gate",
            Some(&[EvalModule::Sensory, EvalModule::CognitionGate]),
        );
        let full_default = write_full_agent_case(
            dir.path(),
            "full-default",
            "full-agent-default-modules",
            None,
        );
        let _full_unrelated = write_full_agent_case(
            dir.path(),
            "full-query-memory",
            "full-agent-query-memory",
            Some(&[EvalModule::Sensory, EvalModule::QueryMemory]),
        );
        let speak = write_module_case(
            dir.path(),
            EvalModule::Speak,
            "speak-target",
            "module-speak-target",
            &[EvalModule::Speak],
        );
        let cognition_gate = write_module_case(
            dir.path(),
            EvalModule::CognitionGate,
            "cognition-gate-target",
            "module-cognition-gate-target",
            &[EvalModule::CognitionGate],
        );
        let _query_memory_with_speak_support = write_module_case(
            dir.path(),
            EvalModule::QueryMemory,
            "query-memory-with-speak-support",
            "module-query-memory-with-speak-support",
            &[EvalModule::QueryMemory, EvalModule::Speak],
        );
        let _memory = write_module_case(
            dir.path(),
            EvalModule::Memory,
            "memory-target",
            "module-memory-target",
            &[EvalModule::Memory],
        );
        let mut config = test_runner_config(dir.path());
        config.module_filters = vec![EvalModule::Speak, EvalModule::CognitionGate];

        let selection = select_case_paths(&config, false).unwrap();

        let mut expected = vec![
            full_cognition_gate,
            full_default,
            full_speak,
            cognition_gate,
            speak,
        ];
        expected.sort();
        assert_eq!(selection.case_paths, expected);
    }

    #[test]
    fn exclude_full_agent_drops_full_agent_paths() {
        let dir = tempfile::tempdir().unwrap();
        let full_speak = write_full_agent_case(
            dir.path(),
            "full-speak",
            "full-agent-speak",
            Some(&[EvalModule::Sensory, EvalModule::Speak]),
        );
        let speak = write_module_case(
            dir.path(),
            EvalModule::Speak,
            "speak-target",
            "module-speak-target",
            &[EvalModule::Speak],
        );
        let memory = write_module_case(
            dir.path(),
            EvalModule::Memory,
            "memory-target",
            "module-memory-target",
            &[EvalModule::Memory],
        );
        let mut config = test_runner_config(dir.path());
        config.exclude_full_agent = true;

        let selection = select_case_paths(&config, false).unwrap();

        assert_eq!(selection.case_paths, vec![memory, speak]);
        assert!(!selection.case_paths.contains(&full_speak));
    }

    #[test]
    fn exclude_full_agent_intersects_module_filters() {
        let dir = tempfile::tempdir().unwrap();
        let _full_speak = write_full_agent_case(
            dir.path(),
            "full-speak",
            "full-agent-speak",
            Some(&[EvalModule::Sensory, EvalModule::Speak]),
        );
        let speak = write_module_case(
            dir.path(),
            EvalModule::Speak,
            "speak-target",
            "module-speak-target",
            &[EvalModule::Speak],
        );
        let _memory = write_module_case(
            dir.path(),
            EvalModule::Memory,
            "memory-target",
            "module-memory-target",
            &[EvalModule::Memory],
        );
        let mut config = test_runner_config(dir.path());
        config.exclude_full_agent = true;
        config.module_filters = vec![EvalModule::Speak];

        let selection = select_case_paths(&config, false).unwrap();

        assert_eq!(selection.case_paths, vec![speak]);
    }

    #[test]
    fn full_agent_only_drops_module_paths() {
        let dir = tempfile::tempdir().unwrap();
        let full_speak = write_full_agent_case(
            dir.path(),
            "full-speak",
            "full-agent-speak",
            Some(&[EvalModule::Sensory, EvalModule::Speak]),
        );
        let _speak = write_module_case(
            dir.path(),
            EvalModule::Speak,
            "speak-target",
            "module-speak-target",
            &[EvalModule::Speak],
        );
        let _memory = write_module_case(
            dir.path(),
            EvalModule::Memory,
            "memory-target",
            "module-memory-target",
            &[EvalModule::Memory],
        );
        let mut config = test_runner_config(dir.path());
        config.full_agent_only = true;

        let selection = select_case_paths(&config, false).unwrap();

        assert_eq!(selection.case_paths, vec![full_speak]);
    }

    #[test]
    fn full_agent_only_intersects_module_filters() {
        let dir = tempfile::tempdir().unwrap();
        let full_speak = write_full_agent_case(
            dir.path(),
            "full-speak",
            "full-agent-speak",
            Some(&[EvalModule::Sensory, EvalModule::Speak]),
        );
        let _full_memory = write_full_agent_case(
            dir.path(),
            "full-memory",
            "full-agent-memory",
            Some(&[EvalModule::Memory]),
        );
        let _speak = write_module_case(
            dir.path(),
            EvalModule::Speak,
            "speak-target",
            "module-speak-target",
            &[EvalModule::Speak],
        );
        let mut config = test_runner_config(dir.path());
        config.full_agent_only = true;
        config.module_filters = vec![EvalModule::Speak];

        let selection = select_case_paths(&config, false).unwrap();

        assert_eq!(selection.case_paths, vec![full_speak]);
    }

    #[test]
    fn full_agent_case_filters_conflict() {
        let dir = tempfile::tempdir().unwrap();
        let mut config = test_runner_config(dir.path());
        config.exclude_full_agent = true;
        config.full_agent_only = true;

        let error = select_case_paths(&config, false).unwrap_err();

        assert!(matches!(error, RunnerError::ConflictingFullAgentFilters));
    }

    #[test]
    fn module_filters_intersect_case_patterns() {
        let dir = tempfile::tempdir().unwrap();
        let special_speak = write_module_case(
            dir.path(),
            EvalModule::Speak,
            "special-speak",
            "module-speak-special",
            &[EvalModule::Speak],
        );
        let _plain_speak = write_module_case(
            dir.path(),
            EvalModule::Speak,
            "plain-speak",
            "module-speak-plain",
            &[EvalModule::Speak],
        );
        let _special_memory = write_module_case(
            dir.path(),
            EvalModule::Memory,
            "special-memory",
            "module-memory-special",
            &[EvalModule::Memory],
        );
        let mut config = test_runner_config(dir.path());
        config.case_patterns = vec!["special".to_string()];
        config.module_filters = vec![EvalModule::Speak];

        let selection = select_case_paths(&config, false).unwrap();

        assert_eq!(selection.case_paths, vec![special_speak]);
    }

    #[test]
    fn visualizer_planned_tabs_use_filtered_case_ids() {
        let dir = tempfile::tempdir().unwrap();
        let case_dir = dir.path().join("eval-cases/full-agent");
        std::fs::create_dir_all(&case_dir).unwrap();
        std::fs::write(
            case_dir.join("first-route.eure"),
            r#"
id = "module-query-memory-first-route"

@ inputs[] {
  $variant: heard
  content = "First?"
}
"#,
        )
        .unwrap();
        std::fs::write(
            case_dir.join("second-memory.eure"),
            r#"
id = "module-query-memory-special-memory"

@ inputs[] {
  $variant: heard
  content = "Second?"
}
"#,
        )
        .unwrap();
        let config = RunnerConfig {
            cases_root: dir.path().join("eval-cases"),
            output_root: dir.path().join("out"),
            llm_log_root: dir.path().join("llm-logs"),
            run_id: "run".to_string(),
            judge_backend: test_backend_config(),
            cheap_backend: test_backend_config(),
            default_backend: test_backend_config(),
            premium_backend: test_backend_config(),
            model_dir: dir.path().join("models"),
            embedding_backend: None,
            fail_fast: false,
            failed_only: false,
            failed_from: None,
            model_concurrency: test_model_concurrency(),
            llm_concurrency_pool: LlmConcurrencyPool::default(),
            trials: NonZeroUsize::new(1).unwrap(),
            case_patterns: vec!["special-memory".to_string()],
            module_filters: Vec::new(),
            disabled_modules: Vec::new(),
            exclude_full_agent: false,
            full_agent_only: false,
        };

        let tabs = visualizer_planned_tabs(&config).unwrap();

        assert_eq!(tabs.len(), 1);
        assert_eq!(tabs[0].0.as_str(), "module-query-memory-special-memory");
        assert_eq!(tabs[0].1, "module-query-memory-special-memory");
    }

    #[test]
    fn failed_only_selects_latest_failed_and_invalid_cases() {
        let dir = tempfile::tempdir().unwrap();
        let passed = write_query_memory_case(dir.path(), "passed", "module-query-memory-passed");
        let failed = write_query_memory_case(dir.path(), "failed", "module-query-memory-failed");
        let invalid = write_query_memory_case(dir.path(), "invalid", "module-query-memory-invalid");
        let output_root = dir.path().join("out");
        let old_report = output_root.join("old-run/suite-report.json");
        write_suite_report(
            &old_report,
            serde_json::json!([
                {
                    "path": passed.display().to_string(),
                    "id": "module-query-memory-passed",
                    "passed": false,
                    "invalid": false
                }
            ]),
        );
        std::thread::sleep(std::time::Duration::from_millis(10));
        let latest_report = output_root.join("latest-run/suite-report.json");
        write_suite_report(
            &latest_report,
            serde_json::json!([
                {
                    "path": passed.display().to_string(),
                    "id": "module-query-memory-passed",
                    "passed": true,
                    "invalid": false
                },
                {
                    "path": failed.display().to_string(),
                    "id": "module-query-memory-failed",
                    "passed": false,
                    "invalid": false
                },
                {
                    "path": invalid.display().to_string(),
                    "id": "module-query-memory-invalid",
                    "passed": false,
                    "invalid": true
                }
            ]),
        );
        let mut config = test_runner_config(dir.path());
        config.output_root = output_root;
        config.failed_only = true;

        let selection = select_case_paths(&config, false).unwrap();

        assert_eq!(selection.failed_from, Some(latest_report));
        assert_eq!(selection.case_paths, vec![failed, invalid]);
    }

    #[test]
    fn failed_from_resolves_run_id_run_dir_and_report_path() {
        let dir = tempfile::tempdir().unwrap();
        let report = dir.path().join("out/run-a/suite-report.json");
        write_suite_report(&report, serde_json::json!([]));
        let mut config = test_runner_config(dir.path());
        config.output_root = dir.path().join("out");

        config.failed_from = Some(PathBuf::from("run-a"));
        assert_eq!(
            resolve_failed_only_reference(&config).unwrap(),
            Some(report.clone())
        );

        config.failed_from = Some(report.parent().unwrap().to_path_buf());
        assert_eq!(
            resolve_failed_only_reference(&config).unwrap(),
            Some(report.clone())
        );

        config.failed_from = Some(report.clone());
        assert_eq!(
            resolve_failed_only_reference(&config).unwrap(),
            Some(report)
        );
    }

    #[test]
    fn failed_only_patterns_intersect_reference_cases() {
        let dir = tempfile::tempdir().unwrap();
        let first = write_query_memory_case(dir.path(), "first", "module-query-memory-first");
        let special = write_query_memory_case(dir.path(), "special", "module-query-memory-special");
        let report = dir.path().join("out/run-a/suite-report.json");
        write_suite_report(
            &report,
            serde_json::json!([
                {
                    "path": first.display().to_string(),
                    "id": "module-query-memory-first",
                    "passed": false,
                    "invalid": false
                },
                {
                    "path": special.display().to_string(),
                    "id": "module-query-memory-special",
                    "passed": false,
                    "invalid": true
                }
            ]),
        );
        let mut config = test_runner_config(dir.path());
        config.output_root = dir.path().join("out");
        config.failed_from = Some(PathBuf::from("run-a"));
        config.case_patterns = vec!["special".to_string()];

        let selection = select_case_paths(&config, false).unwrap();

        assert_eq!(selection.case_paths, vec![special]);
    }

    #[test]
    fn failed_only_all_passed_reference_selects_no_cases() {
        let dir = tempfile::tempdir().unwrap();
        let passed = write_query_memory_case(dir.path(), "passed", "module-query-memory-passed");
        let report = dir.path().join("out/run-a/suite-report.json");
        write_suite_report(
            &report,
            serde_json::json!([
                {
                    "path": passed.display().to_string(),
                    "id": "module-query-memory-passed",
                    "passed": true,
                    "invalid": false
                }
            ]),
        );
        let mut config = test_runner_config(dir.path());
        config.output_root = dir.path().join("out");
        config.failed_from = Some(PathBuf::from("run-a"));

        let selection = select_case_paths(&config, false).unwrap();

        assert!(selection.case_paths.is_empty());
    }

    #[test]
    fn failed_only_reports_reference_errors() {
        let dir = tempfile::tempdir().unwrap();
        let mut config = test_runner_config(dir.path());
        config.failed_only = true;
        let error = select_case_paths(&config, false).unwrap_err();
        assert!(matches!(error, RunnerError::FailedOnlyNoReference { .. }));

        let bad_report = dir.path().join("out/bad/suite-report.json");
        std::fs::create_dir_all(bad_report.parent().unwrap()).unwrap();
        std::fs::write(&bad_report, "not json").unwrap();
        config.failed_only = false;
        config.failed_from = Some(bad_report);
        let error = select_case_paths(&config, false).unwrap_err();
        assert!(matches!(
            error,
            RunnerError::ParseFailedOnlyReference { .. }
        ));

        let missing_report = dir.path().join("out/missing-case/suite-report.json");
        write_suite_report(
            &missing_report,
            serde_json::json!([
                {
                    "path": "eval-cases/modules/query-memory/missing.eure",
                    "id": "module-query-memory-missing",
                    "passed": false,
                    "invalid": false
                }
            ]),
        );
        config.failed_from = Some(missing_report);
        std::fs::create_dir_all(dir.path().join("eval-cases")).unwrap();
        let error = select_case_paths(&config, false).unwrap_err();
        assert!(matches!(error, RunnerError::FailedOnlyCaseNotFound { .. }));
    }

    #[test]
    fn suite_run_report_records_failed_only_reference() {
        let dir = tempfile::tempdir().unwrap();
        let mut config = test_runner_config(dir.path());
        config.failed_from = Some(PathBuf::from("run-a"));
        let run_dir = dir.path().join("out/rerun");
        let failed_from = dir.path().join("out/run-a/suite-report.json");

        let report = suite_run_report(&config, &run_dir, 0, Some(&failed_from));
        let json = serde_json::to_value(&report).unwrap();

        assert_eq!(json["failed_only"], serde_json::json!(true));
        assert_eq!(
            json["failed_from"],
            serde_json::json!(failed_from.display().to_string())
        );
    }

    #[test]
    fn eval_utterance_output_strips_balanced_outer_quotes() {
        assert_eq!(
            normalize_eval_utterance_text("\"short signal\"".to_string()),
            "short signal"
        );
        assert_eq!(
            normalize_eval_utterance_text("  \"short signal\"  ".to_string()),
            "short signal"
        );
        assert_eq!(
            normalize_eval_utterance_text("short signal".to_string()),
            "short signal"
        );
        assert_eq!(
            normalize_eval_utterance_text("\"an apple\" is \"red\"".to_string()),
            "\"an apple\" is \"red\""
        );
        assert_eq!(
            normalize_eval_utterance_text("\"an \\\"apple\\\"\"".to_string()),
            "an \"apple\""
        );
    }

    #[tokio::test]
    async fn recording_utterance_sink_returns_last_complete() {
        let dir = tempfile::tempdir().unwrap();
        let reporter = LiveReporter::new("test-run", dir.path()).unwrap();
        let actions = Rc::new(ActionActivityTracker::new(vec![builtin::speak()]));
        let sink =
            RecordingUtteranceSink::new("test-case".to_string(), reporter, actions.clone(), None);
        let emitted_at = Utc.with_ymd_and_hms(2026, 5, 7, 12, 0, 0).unwrap();
        let sender = ModuleInstanceId::new(builtin::speak(), ReplicaIndex::ZERO);

        sink.on_complete(Utterance {
            sender: sender.clone(),
            target: "peer".to_string(),
            text: "first".to_string(),
            emitted_at,
        })
        .await
        .unwrap();
        sink.on_complete(Utterance {
            sender,
            target: "peer".to_string(),
            text: "second".to_string(),
            emitted_at,
        })
        .await
        .unwrap();

        assert_eq!(sink.snapshot().len(), 2);
        assert_eq!(sink.last_complete().unwrap().text, "second");
        assert!(
            !actions.silence_window_elapsed_at(FULL_AGENT_ACTION_SILENCE_WINDOW, Instant::now())
        );
    }

    #[tokio::test]
    async fn memory_diff_observation_exposes_entries_and_links_structurally() {
        let summary_index = MemoryIndex::new("summary");
        let source_index = MemoryIndex::new("source");
        let now = Utc.with_ymd_and_hms(2026, 5, 17, 0, 0, 0).unwrap();
        let summary = MemoryRecord {
            index: summary_index.clone(),
            content: nuillu_types::MemoryContent::new("Koro food boundary summary."),
            rank: MemoryRank::ShortTerm,
            occurred_at: None,
            stored_at: now,
            kind: nuillu_memory::MemoryKind::Reflection,
            concepts: Vec::new(),
            tags: Vec::new(),
            affect_arousal: 0.0,
            valence: 0.0,
            emotion: String::new(),
        };
        let store = StaticMemoryStore {
            records: RefCell::new(vec![summary]),
            links: vec![nuillu_memory::MemoryLink {
                from_memory: summary_index,
                to_memory: source_index,
                relation: MemoryLinkRelation::DerivedFrom,
                freeform_relation: None,
                strength: 1.0,
                confidence: 1.0,
                updated_at: now,
            }],
        };

        let diff = memory_diff_observation(&BTreeMap::new(), &store).await;
        let value = serde_json::to_value(diff).unwrap();

        assert_eq!(
            pointer_text(&value, "/entries/0/kind").as_deref(),
            Some("Reflection")
        );
        assert_eq!(
            pointer_text(&value, "/links/0/relation").as_deref(),
            Some("derived_from")
        );
    }

    #[tokio::test]
    async fn memory_diff_observation_exposes_deleted_baseline_records() {
        let deleted_index = MemoryIndex::new("source");
        let now = Utc.with_ymd_and_hms(2026, 5, 17, 0, 0, 0).unwrap();
        let deleted = MemoryRecord {
            index: deleted_index.clone(),
            content: nuillu_types::MemoryContent::new("Redundant source memory."),
            rank: MemoryRank::ShortTerm,
            occurred_at: None,
            stored_at: now,
            kind: nuillu_memory::MemoryKind::Episode,
            concepts: Vec::new(),
            tags: Vec::new(),
            affect_arousal: 0.0,
            valence: 0.0,
            emotion: String::new(),
        };
        let baseline = BTreeMap::from([(deleted_index.to_string(), deleted)]);
        let store = StaticMemoryStore {
            records: RefCell::new(Vec::new()),
            links: Vec::new(),
        };

        let diff = memory_diff_observation(&baseline, &store).await;
        let value = serde_json::to_value(diff).unwrap();

        assert_eq!(
            pointer_text(&value, "/deleted/0").as_deref(),
            Some("source")
        );
    }

    #[tokio::test]
    async fn seed_and_bootstrap_eval_startup_context_seeds_before_identity_bootstrap() {
        let blackboard = Blackboard::default();
        let clock: Rc<dyn Clock> = Rc::new(FixedClock(
            Utc.with_ymd_and_hms(2026, 5, 21, 0, 0, 0).unwrap(),
        ));
        let memory_store = Rc::new(StaticMemoryStore {
            records: RefCell::new(Vec::new()),
            links: Vec::new(),
        });
        let memory_caps = MemoryCapabilities::new(
            blackboard.clone(),
            clock.clone(),
            memory_store.clone(),
            Vec::new(),
        );
        let policy_store = Rc::new(nuillu_reward::NoopPolicyStore);
        let policy_caps = PolicyCapabilities::new(
            blackboard.clone(),
            clock.clone(),
            policy_store.clone(),
            Vec::new(),
        );
        let memories = vec![
            crate::cases::MemorySeed {
                index: None,
                rank: crate::cases::MemorySeedRank::Identity,
                decay_secs: 86_400,
                datetime: None,
                seconds_ago: None,
                content: eure::value::Text::plaintext("I am Nui."),
            },
            crate::cases::MemorySeed {
                index: None,
                rank: crate::cases::MemorySeedRank::ShortTerm,
                decay_secs: 86_400,
                datetime: None,
                seconds_ago: None,
                content: eure::value::Text::plaintext("Ordinary memory."),
            },
        ];

        seed_and_bootstrap_eval_startup_context(
            &memory_caps,
            &policy_caps,
            memory_store.as_ref(),
            policy_store.as_ref(),
            &blackboard,
            clock.as_ref(),
            None,
            &memories,
            &[],
            &[],
        )
        .await
        .unwrap();

        let identity_memories = blackboard.read(|bb| bb.identity_memories().to_vec()).await;
        assert_eq!(identity_memories.len(), 1);
        assert_eq!(identity_memories[0].index.as_str(), "seed-1");
        assert_eq!(identity_memories[0].content.as_str(), "I am Nui.");
    }

    struct StaticMemoryStore {
        records: RefCell<Vec<MemoryRecord>>,
        links: Vec<nuillu_memory::MemoryLink>,
    }

    #[async_trait::async_trait(?Send)]
    impl MemoryStore for StaticMemoryStore {
        async fn insert(
            &self,
            mem: nuillu_memory::NewMemory,
            stored_at: DateTime<Utc>,
        ) -> std::result::Result<MemoryRecord, PortError> {
            let index = MemoryIndex::new(format!("seed-{}", self.records.borrow().len() + 1));
            let record = MemoryRecord {
                index,
                content: mem.content,
                rank: mem.rank,
                occurred_at: mem.occurred_at,
                stored_at,
                kind: mem.kind,
                concepts: mem.concepts,
                tags: mem.tags,
                affect_arousal: mem.affect_arousal,
                valence: mem.valence,
                emotion: mem.emotion,
            };
            self.records.borrow_mut().push(record.clone());
            Ok(record)
        }

        async fn put(
            &self,
            _mem: nuillu_memory::IndexedMemory,
        ) -> std::result::Result<MemoryRecord, PortError> {
            unimplemented!("static test store does not support writes")
        }

        async fn compact(
            &self,
            _mem: nuillu_memory::NewMemory,
            _sources: &[MemoryIndex],
            _stored_at: DateTime<Utc>,
        ) -> std::result::Result<MemoryRecord, PortError> {
            unimplemented!("static test store does not support writes")
        }

        async fn put_compacted(
            &self,
            _mem: nuillu_memory::IndexedMemory,
            _sources: &[MemoryIndex],
        ) -> std::result::Result<MemoryRecord, PortError> {
            unimplemented!("static test store does not support writes")
        }

        async fn get(
            &self,
            index: &MemoryIndex,
        ) -> std::result::Result<Option<MemoryRecord>, PortError> {
            Ok(self
                .records
                .borrow()
                .iter()
                .find(|record| &record.index == index)
                .cloned())
        }

        async fn list_by_rank(
            &self,
            rank: MemoryRank,
        ) -> std::result::Result<Vec<MemoryRecord>, PortError> {
            Ok(self
                .records
                .borrow()
                .iter()
                .filter(|record| record.rank == rank)
                .cloned()
                .collect())
        }

        async fn search(
            &self,
            _q: &MemoryQuery,
        ) -> std::result::Result<Vec<MemoryRecord>, PortError> {
            Ok(Vec::new())
        }

        async fn linked(
            &self,
            q: &LinkedMemoryQuery,
        ) -> std::result::Result<Vec<nuillu_memory::LinkedMemoryRecord>, PortError> {
            Ok(self
                .links
                .iter()
                .filter(|link| {
                    q.memory_indexes.contains(&link.from_memory)
                        || q.memory_indexes.contains(&link.to_memory)
                })
                .filter_map(|link| {
                    self.records
                        .borrow()
                        .iter()
                        .find(|record| record.index == link.from_memory)
                        .cloned()
                        .map(|record| nuillu_memory::LinkedMemoryRecord {
                            record,
                            link: link.clone(),
                        })
                })
                .collect())
        }

        async fn upsert_link(
            &self,
            _link: nuillu_memory::NewMemoryLink,
            _updated_at: DateTime<Utc>,
        ) -> std::result::Result<nuillu_memory::MemoryLink, PortError> {
            unimplemented!("static test store does not support writes")
        }

        async fn delete(&self, _index: &MemoryIndex) -> std::result::Result<(), PortError> {
            unimplemented!("static test store does not support writes")
        }
    }

    #[test]
    fn action_tracker_waits_for_silence_window_after_action_completion() {
        let tracker = ActionActivityTracker::new(vec![builtin::speak()]);
        let now = Instant::now();

        tracker.record_completed_at(&builtin::speak(), now);

        assert!(!tracker.silence_window_elapsed_at(Duration::from_secs(1), now));
        assert!(
            !tracker.silence_window_elapsed_at(
                Duration::from_secs(1),
                now + Duration::from_millis(999)
            )
        );
        assert!(
            tracker.silence_window_elapsed_at(Duration::from_secs(1), now + Duration::from_secs(1))
        );
    }

    #[test]
    fn action_tracker_ignores_non_action_module_completion() {
        let tracker = ActionActivityTracker::new(vec![builtin::speak()]);
        let completed_at = Instant::now();
        let after_window = completed_at + Duration::from_secs(2);

        tracker.record_completed_at(&builtin::query_memory(), completed_at);
        assert!(!tracker.silence_window_elapsed_at(Duration::from_secs(1), after_window));

        tracker.record_completed_at(&builtin::speak(), completed_at);
        tracker.record_completed_at(&builtin::query_memory(), after_window);
        assert!(tracker.silence_window_elapsed_at(Duration::from_secs(1), after_window));
    }

    #[test]
    fn full_agent_settle_waits_for_runtime_after_action_silence() {
        let actions = ActionActivityTracker::new(vec![builtin::speak()]);
        let now = Instant::now();
        actions.record_completed_at(&builtin::speak(), now);
        let mut settle = FullAgentSettleTracker::new(0, now);

        let after_action_silence = now + FULL_AGENT_ACTION_SILENCE_WINDOW;
        assert!(!full_agent_ready_to_score_at(
            &actions,
            &settle,
            1,
            true,
            false,
            false,
            after_action_silence,
        ));

        let late_progress = after_action_silence + Duration::from_millis(50);
        settle.observe_progress_count(1, late_progress);
        assert!(!full_agent_ready_to_score_at(
            &actions,
            &settle,
            0,
            true,
            false,
            false,
            late_progress + Duration::from_millis(199),
        ));
        assert!(full_agent_ready_to_score_at(
            &actions,
            &settle,
            0,
            true,
            false,
            false,
            late_progress + FULL_AGENT_RUNTIME_SILENCE_WINDOW,
        ));
    }

    #[test]
    fn full_agent_settle_waits_for_runtime_silence_without_action_output() {
        let actions = ActionActivityTracker::new(vec![builtin::speak()]);
        let now = Instant::now();
        let settle = FullAgentSettleTracker::new(0, now);

        assert!(!full_agent_ready_to_score_at(
            &actions,
            &settle,
            0,
            true,
            true,
            false,
            now + Duration::from_millis(199),
        ));
        assert!(full_agent_ready_to_score_at(
            &actions,
            &settle,
            0,
            true,
            true,
            false,
            now + FULL_AGENT_RUNTIME_SILENCE_WINDOW,
        ));
    }

    #[test]
    fn max_llm_calls_requests_stop_after_limit_event() {
        let dir = tempfile::tempdir().unwrap();
        let reporter = LiveReporter::new("test-run", dir.path()).unwrap();
        let sink = RecordingRuntimeEventSink::new("test-case".to_string(), Some(3), reporter, None);
        let owner = ModuleInstanceId::new(builtin::query_memory(), ReplicaIndex::ZERO);

        sink.on_event(RuntimeEvent::LlmAccessed {
            sequence: 0,
            call: 0,
            owner: owner.clone(),
            tier: ModelTier::Default,
        })
        .unwrap();
        assert!(!sink.stop_requested());
        assert_eq!(sink.progress_event_count(), 1);

        sink.on_event(RuntimeEvent::LlmAccessed {
            sequence: 1,
            call: 1,
            owner: owner.clone(),
            tier: ModelTier::Default,
        })
        .unwrap();
        assert!(!sink.stop_requested());

        sink.on_event(RuntimeEvent::LlmAccessed {
            sequence: 2,
            call: 2,
            owner: owner.clone(),
            tier: ModelTier::Default,
        })
        .unwrap();

        assert!(sink.stop_requested());
        sink.on_event(RuntimeEvent::LlmAccessed {
            sequence: 3,
            call: 3,
            owner,
            tier: ModelTier::Default,
        })
        .unwrap();
        assert_eq!(sink.snapshot().len(), 4);
        let jsonl = std::fs::read_to_string(dir.path().join("events.jsonl")).unwrap();
        assert!(jsonl.contains("\"kind\":\"runtime_event\""));
        assert!(jsonl.contains("\"kind\":\"stop_requested\""));
        assert_eq!(jsonl.matches("\"kind\":\"stop_requested\"").count(), 1);
    }

    #[test]
    fn recording_runtime_event_sink_builds_activation_timeline() {
        let dir = tempfile::tempdir().unwrap();
        let reporter = LiveReporter::new("test-run", dir.path()).unwrap();
        let sink = RecordingRuntimeEventSink::new("test-case".to_string(), None, reporter, None);
        let owner = ModuleInstanceId::new(builtin::speak(), ReplicaIndex::ZERO);

        sink.on_event(RuntimeEvent::ModuleBatchReady {
            sequence: 0,
            owner: owner.clone(),
            batch_type: "cognition".to_string(),
            batch_debug: String::new(),
        })
        .unwrap();
        sink.on_event(RuntimeEvent::ModuleActivationCompleted {
            sequence: 1,
            owner,
            duration: Duration::from_millis(42),
            succeeded: true,
        })
        .unwrap();

        let activations = sink.activation_timeline();
        assert_eq!(activations.len(), 1);
        assert_eq!(activations[0].module, "speak");
        assert_eq!(activations[0].duration_ms, 42);
        assert!(activations[0].succeeded);
    }

    #[test]
    fn runtime_progress_count_ignores_scheduler_noise() {
        let dir = tempfile::tempdir().unwrap();
        let reporter = LiveReporter::new("test-run", dir.path()).unwrap();
        let sink = RecordingRuntimeEventSink::new("test-case".to_string(), None, reporter, None);
        let owner = ModuleInstanceId::new(builtin::homeostatic_controller(), ReplicaIndex::ZERO);

        sink.on_event(RuntimeEvent::ModuleBatchThrottled {
            sequence: 0,
            owner: owner.clone(),
            delayed_for: Duration::from_secs(3),
        })
        .unwrap();
        sink.on_event(RuntimeEvent::ModuleBatchReady {
            sequence: 1,
            owner: owner.clone(),
            batch_type: "()".to_string(),
            batch_debug: "()".to_string(),
        })
        .unwrap();
        sink.on_event(RuntimeEvent::RateLimitDelayed {
            sequence: 2,
            owner: owner.clone(),
            capability: nuillu_module::CapabilityKind::LlmCall,
            delayed_for: Duration::from_millis(250),
        })
        .unwrap();

        assert_eq!(sink.event_count(), 3);
        assert_eq!(sink.progress_event_count(), 0);

        sink.on_event(RuntimeEvent::MemoUpdated {
            sequence: 3,
            owner,
            char_count: 42,
        })
        .unwrap();

        assert_eq!(sink.event_count(), 4);
        assert_eq!(sink.progress_event_count(), 1);
    }

    #[test]
    fn scheduled_wait_remaining_tracks_throttle_and_rate_limit_deadlines() {
        let owner = ModuleInstanceId::new(builtin::homeostatic_controller(), ReplicaIndex::ZERO);
        let timed_events = vec![
            (
                100,
                RuntimeEvent::ModuleBatchThrottled {
                    sequence: 0,
                    owner: owner.clone(),
                    delayed_for: Duration::from_millis(1_000),
                },
            ),
            (
                400,
                RuntimeEvent::RateLimitDelayed {
                    sequence: 1,
                    owner,
                    capability: nuillu_module::CapabilityKind::LlmCall,
                    delayed_for: Duration::from_millis(200),
                },
            ),
            (
                450,
                RuntimeEvent::MemoUpdated {
                    sequence: 2,
                    owner: ModuleInstanceId::new(builtin::sensory(), ReplicaIndex::ZERO),
                    char_count: 12,
                },
            ),
        ];

        assert_eq!(
            scheduled_wait_remaining_from_timed_events(&timed_events, Duration::from_millis(500)),
            Some(Duration::from_millis(600))
        );
        assert_eq!(
            scheduled_wait_remaining_from_timed_events(&timed_events, Duration::from_millis(1_000)),
            Some(Duration::from_millis(100))
        );
        assert_eq!(
            scheduled_wait_remaining_from_timed_events(&timed_events, Duration::from_millis(1_100)),
            None
        );
    }

    #[test]
    fn llm_in_flight_tracks_access_and_completion() {
        let dir = tempfile::tempdir().unwrap();
        let reporter = LiveReporter::new("test-run", dir.path()).unwrap();
        let sink = RecordingRuntimeEventSink::new("test-case".to_string(), None, reporter, None);
        let owner = ModuleInstanceId::new(builtin::cognition_gate(), ReplicaIndex::ZERO);

        assert_eq!(sink.llm_in_flight_count(), 0);

        sink.on_event(RuntimeEvent::LlmAccessed {
            sequence: 0,
            call: 0,
            owner: owner.clone(),
            tier: ModelTier::Cheap,
        })
        .unwrap();
        assert_eq!(sink.llm_in_flight_count(), 1);

        sink.on_event(RuntimeEvent::LlmAccessed {
            sequence: 1,
            call: 1,
            owner: owner.clone(),
            tier: ModelTier::Cheap,
        })
        .unwrap();
        assert_eq!(sink.llm_in_flight_count(), 2);

        sink.on_event(RuntimeEvent::LlmCompleted {
            sequence: 2,
            call: 0,
            owner: owner.clone(),
            tier: ModelTier::Cheap,
        })
        .unwrap();
        assert_eq!(sink.llm_in_flight_count(), 1);

        sink.on_event(RuntimeEvent::LlmCompleted {
            sequence: 3,
            call: 1,
            owner,
            tier: ModelTier::Cheap,
        })
        .unwrap();
        assert_eq!(sink.llm_in_flight_count(), 0);
        assert_eq!(sink.progress_event_count(), 4);
    }

    #[test]
    fn step_settle_modules_follow_next_wait_target() {
        assert_eq!(
            step_settle_modules(Some(&WaitFor::MemoFrom {
                module: EvalModule::Surprise,
                timeout_ms: 1000,
            })),
            vec![builtin::surprise()]
        );
        assert_eq!(
            step_settle_modules(Some(&WaitFor::Interoception {
                timeout_ms: 1000,
                mode: None,
                wake_arousal_at_least: f64::NEG_INFINITY,
                wake_arousal_at_most: f64::INFINITY,
            })),
            vec![builtin::interoception()]
        );
        assert!(step_settle_modules(None).is_empty());
    }

    #[tokio::test]
    async fn step_settle_wait_ignores_unrelated_inflight_modules() {
        let blackboard = Blackboard::default();
        blackboard
            .apply(BlackboardCommand::SetModuleRunStatus {
                owner: ModuleInstanceId::new(builtin::allocation_controller(), ReplicaIndex::ZERO),
                status: ModuleRunStatus::Activating,
            })
            .await;
        let dir = tempfile::tempdir().unwrap();
        let reporter = LiveReporter::new("test-run", dir.path()).unwrap();
        let events = RecordingRuntimeEventSink::new("test-case".to_string(), None, reporter, None);

        let outcome = wait_for_step_modules_to_settle(
            &blackboard,
            &events,
            &[builtin::surprise()],
            Duration::from_millis(1),
        )
        .await;

        assert!(matches!(outcome, WaitOutcome::Met));
    }

    #[tokio::test]
    async fn step_settle_wait_times_out_for_target_inflight_module() {
        let blackboard = Blackboard::default();
        blackboard
            .apply(BlackboardCommand::SetModuleRunStatus {
                owner: ModuleInstanceId::new(builtin::surprise(), ReplicaIndex::ZERO),
                status: ModuleRunStatus::Activating,
            })
            .await;
        let dir = tempfile::tempdir().unwrap();
        let reporter = LiveReporter::new("test-run", dir.path()).unwrap();
        let events = RecordingRuntimeEventSink::new("test-case".to_string(), None, reporter, None);

        let outcome = wait_for_step_modules_to_settle(
            &blackboard,
            &events,
            &[builtin::surprise()],
            Duration::from_millis(1),
        )
        .await;

        assert!(matches!(outcome, WaitOutcome::Timeout));
    }

    #[tokio::test]
    async fn wait_for_condition_prefers_new_memo_over_later_stop() {
        let blackboard = Blackboard::default();
        let dir = tempfile::tempdir().unwrap();
        let reporter = LiveReporter::new("test-run", dir.path()).unwrap();
        let events = RecordingRuntimeEventSink::new("test-case".to_string(), None, reporter, None);
        let wait_for = WaitFor::MemoFrom {
            module: EvalModule::Surprise,
            timeout_ms: 500,
        };
        let owner = ModuleInstanceId::new(builtin::surprise(), ReplicaIndex::ZERO);
        let now = Utc.with_ymd_and_hms(2026, 5, 18, 0, 0, 0).unwrap();

        let wait = wait_for_condition(&blackboard, &events, &wait_for);
        let publish_then_stop = async {
            tokio::time::sleep(Duration::from_millis(1)).await;
            blackboard
                .apply(BlackboardCommand::UpdateMemo {
                    owner,
                    memo: "surprise landed".to_string(),
                    written_at: now,
                })
                .await;
            events.request_stop("after-memo");
        };
        let (outcome, _) = tokio::join!(wait, publish_then_stop);

        assert!(matches!(outcome, WaitOutcome::Met));
    }

    #[test]
    fn step_outcomes_all_ok_requires_only_ok_statuses() {
        assert!(step_outcomes_all_ok(&[
            serde_json::json!({"index": 0, "status": "ok", "checks": []}),
            serde_json::json!({"index": 1, "status": "ok", "checks": []}),
        ]));
        assert!(!step_outcomes_all_ok(&[]));
        assert!(!step_outcomes_all_ok(&[
            serde_json::json!({"index": 0, "status": "ok", "checks": []}),
            serde_json::json!({"index": 1, "status": "stopped", "checks": []}),
        ]));
    }

    #[tokio::test]
    async fn run_suite_records_case_runtime_failures_and_continues() {
        let dir = tempfile::tempdir().unwrap();
        let case_dir = dir.path().join("eval-cases/modules/query-memory");
        std::fs::create_dir_all(&case_dir).unwrap();
        for id in ["runtime-failure-one", "runtime-failure-two"] {
            std::fs::write(
                case_dir.join(format!("{id}.eure")),
                format!(
                    r#"
id = "{id}"
prompt = "Who are you?"

limits {{
  max-llm-calls = 1
}}
"#
                ),
            )
            .unwrap();
        }

        let output_root = dir.path().join("out");
        let mut cheap_backend = test_backend_config_with_model("cheap-model");
        cheap_backend.max_concurrent_llm_calls = NonZeroUsize::new(7);
        let config = RunnerConfig {
            cases_root: dir.path().join("eval-cases"),
            output_root: output_root.clone(),
            llm_log_root: dir.path().join("llm-logs"),
            run_id: "runtime-failures".to_string(),
            judge_backend: test_backend_config_with_model("judge-model"),
            cheap_backend,
            default_backend: test_backend_config_with_model("default-model"),
            premium_backend: test_backend_config_with_model("premium-model"),
            model_dir: dir.path().join("missing-model"),
            embedding_backend: None,
            fail_fast: false,
            failed_only: false,
            failed_from: None,
            model_concurrency: model_concurrency_from_backends([
                test_backend_config_with_model("judge-model"),
                {
                    let mut cheap = test_backend_config_with_model("cheap-model");
                    cheap.max_concurrent_llm_calls = NonZeroUsize::new(7);
                    cheap
                },
                test_backend_config_with_model("default-model"),
                test_backend_config_with_model("premium-model"),
            ]),
            llm_concurrency_pool: LlmConcurrencyPool::default(),
            trials: NonZeroUsize::new(1).unwrap(),
            case_patterns: Vec::new(),
            module_filters: Vec::new(),
            disabled_modules: Vec::new(),
            exclude_full_agent: false,
            full_agent_only: false,
        };

        let report = run_suite(&config).await.unwrap();

        let run_dir = output_root.join("runtime-failures");
        assert_eq!(report.run.run_id, "runtime-failures");
        assert_eq!(
            report.run.cases_root,
            config.cases_root.display().to_string()
        );
        assert_eq!(report.run.output_dir, run_dir.display().to_string());
        assert_eq!(report.run.case_patterns, Vec::<String>::new());
        assert!(!report.run.fail_fast);
        assert_eq!(
            report.run.model_concurrency.get("cheap-model"),
            Some(&Some(7))
        );
        assert_eq!(report.run.trials, 1);
        assert_eq!(report.run.planned_case_count, 2);
        assert_eq!(report.run.models.judge, "judge-model");
        assert_eq!(report.run.models.cheap, "cheap-model");
        assert_eq!(report.run.models.default, "default-model");
        assert_eq!(report.run.models.premium, "premium-model");
        assert_eq!(report.run.module_filters, Vec::<String>::new());
        assert_eq!(report.case_count, 2);
        assert_eq!(report.passed_cases, 0);
        assert_eq!(report.invalid_cases, 2);
        assert!(report.cases.iter().all(|case| {
            !case.passed
                && case.invalid
                && case.score == 0.0
                && case.report.runtime_failure.is_some()
                && case.report.checks.is_empty()
                && case.trial_count == 1
                && case.passed_trials == 0
                && case.failed_trials == 0
                && case.invalid_trials == 1
                && case.trials.len() == 1
        }));
        assert_metric_values(&report.metrics.pass_at, &[(1, 0.0)]);
        assert_metric_values(&report.metrics.pass_hat, &[(1, 0.0)]);

        assert!(run_dir.join("suite-report.json").exists());
        let suite_json: serde_json::Value =
            serde_json::from_slice(&std::fs::read(run_dir.join("suite-report.json")).unwrap())
                .unwrap();
        assert_eq!(
            suite_json["run"],
            serde_json::json!({
                "run_id": "runtime-failures",
                "cases_root": config.cases_root.display().to_string(),
                "output_dir": run_dir.display().to_string(),
                "case_patterns": [],
                "failed_only": false,
                "failed_from": null,
                "fail_fast": false,
                "model_concurrency": {
                    "cheap-model": 7,
                    "default-model": null,
                    "judge-model": null,
                    "premium-model": null,
                },
                "trials": 1,
                "planned_case_count": 2,
                "models": {
                    "judge": "judge-model",
                    "cheap": "cheap-model",
                    "default": "default-model",
                    "premium": "premium-model",
                },
                "module_filters": [],
                "disabled_modules": [],
                "exclude_full_agent": false,
                "full_agent_only": false,
            })
        );
        for summary in &report.cases {
            let output_dir = run_dir.join(sanitize_id(&summary.id));
            assert!(output_dir.join("report.json").exists());
            assert!(output_dir.join("artifact.json").exists());
            assert!(output_dir.join("raw-trace.json").exists());
            assert!(!output_dir.join("trial-001").exists());
            let events: serde_json::Value =
                serde_json::from_slice(&std::fs::read(output_dir.join("events.json")).unwrap())
                    .unwrap();
            assert_eq!(events, serde_json::json!([]));
        }
    }

    #[tokio::test]
    async fn run_suite_records_multiple_trials_per_case() {
        let dir = tempfile::tempdir().unwrap();
        let case_dir = dir.path().join("eval-cases/modules/query-memory");
        std::fs::create_dir_all(&case_dir).unwrap();
        std::fs::write(
            case_dir.join("runtime-failure.eure"),
            r#"
id = "runtime-failure"
prompt = "Who are you?"

limits {
  max-llm-calls = 1
}
"#,
        )
        .unwrap();

        let output_root = dir.path().join("out");
        let mut config = test_runner_config(dir.path());
        config.output_root = output_root.clone();
        config.run_id = "multi-trial".to_string();
        config.model_concurrency =
            BTreeMap::from([("cheap-model".to_string(), NonZeroUsize::new(7))]);
        config.trials = NonZeroUsize::new(2).unwrap();

        let report = run_suite(&config).await.unwrap();

        assert_eq!(report.run.trials, 2);
        assert_eq!(report.case_count, 1);
        assert_eq!(report.invalid_cases, 1);
        assert_metric_values(&report.metrics.pass_at, &[(1, 0.0), (2, 0.0)]);
        assert_metric_values(&report.metrics.pass_hat, &[(1, 0.0), (2, 0.0)]);

        let summary = &report.cases[0];
        assert_eq!(summary.trial_count, 2);
        assert_eq!(summary.passed_trials, 0);
        assert_eq!(summary.failed_trials, 0);
        assert_eq!(summary.invalid_trials, 2);
        assert_eq!(summary.trials.len(), 2);

        let output_dir = output_root
            .join("multi-trial")
            .join(sanitize_id(&summary.id));
        assert!(output_dir.join("report.json").exists());
        assert!(!output_dir.join("artifact.json").exists());
        for trial in ["trial-001", "trial-002"] {
            let trial_dir = output_dir.join(trial);
            assert!(trial_dir.join("report.json").exists());
            assert!(trial_dir.join("artifact.json").exists());
            assert!(trial_dir.join("raw-trace.json").exists());
        }

        let suite_json: serde_json::Value = serde_json::from_slice(
            &std::fs::read(output_dir.parent().unwrap().join("suite-report.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(suite_json["run"]["trials"], serde_json::json!(2));
        assert_eq!(suite_json["cases"][0]["trial_count"], serde_json::json!(2));
        assert_eq!(
            suite_json["metrics"]["pass_at"],
            serde_json::json!([
                {"k": 1, "value": 0.0},
                {"k": 2, "value": 0.0},
            ])
        );
        assert_eq!(
            suite_json["metrics"]["pass_hat"],
            serde_json::json!([
                {"k": 1, "value": 0.0},
                {"k": 2, "value": 0.0},
            ])
        );
    }

    #[tokio::test]
    async fn agent_observation_serializes_string_keyed_blackboard_maps() {
        let now = Utc.with_ymd_and_hms(2026, 5, 7, 0, 0, 0).unwrap();
        let mut allocation = ResourceAllocation::default();
        allocation.set_model_override(builtin::query_memory(), ModelTier::Default);
        allocation.set(
            builtin::query_memory(),
            ModuleConfig {
                guidance: "test guidance".into(),
            },
        );
        allocation.set_activation(builtin::query_memory(), ActivationRatio::ONE);
        let blackboard = Blackboard::with_allocation(allocation.clone());
        let owner = ModuleInstanceId::new(builtin::query_memory(), ReplicaIndex::ZERO);

        blackboard
            .apply(BlackboardCommand::UpdateMemo {
                owner: owner.clone(),
                memo: "memo".to_string(),
                written_at: now,
            })
            .await;
        blackboard
            .apply(BlackboardCommand::AppendCognitionLog {
                source: owner,
                entry: CognitionLogEntry {
                    at: now,
                    text: "cognition".to_string(),
                },
            })
            .await;
        blackboard
            .apply(BlackboardCommand::SetModulePolicies {
                policies: vec![(
                    builtin::query_memory(),
                    nuillu_blackboard::ModulePolicy::new(
                        ReplicaCapRange::new(0, 0).unwrap(),
                        Bpm::from_f64(60.0)..=Bpm::from_f64(60.0),
                        linear_ratio_fn,
                    ),
                )],
            })
            .await;
        blackboard
            .apply(BlackboardCommand::RecordAllocationProposal {
                controller: ModuleInstanceId::new(
                    builtin::allocation_controller(),
                    ReplicaIndex::ZERO,
                ),
                proposal: allocation,
            })
            .await;
        blackboard
            .apply(BlackboardCommand::UpsertMemoryMetadata {
                index: MemoryIndex::new("mem-1"),
                rank_if_new: MemoryRank::Permanent,
                occurred_at_if_new: None,
                decay_if_new_secs: 0,
                now,
                patch: MemoryMetaPatch::default(),
            })
            .await;

        let observation = blackboard
            .read(|bb| AgentObservation::from_blackboard(bb, Vec::new()))
            .await;
        let actual = serde_json::to_value(observation).unwrap();
        let expected = serde_json::json!({
            "memo_logs": {
                "query-memory": [{
                    "replica": 0,
                    "index": 0,
                    "written_at": "2026-05-07T00:00:00+00:00",
                    "content": "memo",
                }],
            },
            "cognition_logs": [{
                "source": {
                    "module": "query-memory",
                    "replica": 0,
                },
                "entries": [{
                    "at": "2026-05-07T00:00:00Z",
                    "text": "cognition",
                }],
            }],
            "interoception": {
                "mode": "wake",
                "wake_arousal": 0.0,
                "nrem_pressure": 0.0,
                "rem_pressure": 0.0,
                "affect_arousal": 0.0,
                "valence": 0.0,
                "emotion": "",
                "last_updated": "1970-01-01T00:00:00Z",
            },
            "allocation": {
                "query-memory": {
                    "activation_ratio": 1.0,
                    "active_replicas": 0,
                    "cooldown_ms": 1000,
                    "guidance": "test guidance",
                    "tier": "Default",
                },
            },
            "allocation_proposals": {
                "allocation-controller": {
                    "query-memory": {
                        "activation_ratio": 1.0,
                        "active_replicas": 0,
                        "cooldown_ms": null,
                        "guidance": "test guidance",
                        "tier": "Default",
                    },
                },
            },
            "replica_caps": {
                "query-memory": {
                    "min": 0,
                    "max": 0,
                },
            },
            "memory_metadata": {
                "mem-1": {
                    "index": "mem-1",
                    "rank": "Permanent",
                    "occurred_at": null,
                    "decay_remaining_secs": 0,
                    "remember_tokens": 0,
                    "last_accessed": "2026-05-07T00:00:00Z",
                    "access_count": 0,
                    "last_used": null,
                    "use_count": 0,
                    "last_reinforced_at": null,
                    "reinforcement_count": 0,
                    "query_history": [],
                    "use_history": [],
                    "reinforcement_history": [],
                },
            },
            "utterances": [],
        });
        assert_eq!(actual, expected);
    }

    #[tokio::test]
    async fn seed_cognition_log_stamps_cognition_gate_replica_and_offsets_time() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("cognition-seed.eure");
        std::fs::write(
            &path,
            r#"
id = "cognition-seed"
prompt = "What am I attending to?"

@ cognition-log[] {
  text = "Older cognition-log topic"
  seconds-ago = 30
}

@ cognition-log[] {
  text = "Current cognition-log topic"
}
"#,
        )
        .unwrap();
        let case = crate::cases::parse_module_case_file(&path).unwrap();
        let blackboard = Blackboard::default();
        let now = Utc.with_ymd_and_hms(2026, 5, 7, 12, 0, 0).unwrap();

        seed_cognition_log(&blackboard, &FixedClock(now), &case.cognition_log).await;

        let log_set = blackboard.read(|bb| bb.cognition_log_set()).await;
        let records = log_set.logs();
        assert_eq!(records.len(), 1);
        let record = &records[0];
        assert_eq!(record.source.module, builtin::cognition_gate());
        assert_eq!(record.source.replica, ReplicaIndex::ZERO);
        assert_eq!(record.entries.len(), 2);
        assert_eq!(record.entries[0].text, "Older cognition-log topic");
        assert_eq!(record.entries[0].at, now - ChronoDuration::seconds(30));
        assert_eq!(record.entries[1].text, "Current cognition-log topic");
        assert_eq!(record.entries[1].at, now);
    }

    #[tokio::test]
    async fn cognition_gate_eval_waits_for_output_beyond_seed_baseline() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("cognition-seed-baseline.eure");
        std::fs::write(
            &path,
            r#"
id = "cognition-seed-baseline"
prompt = "Admit retrieved memory evidence only if it is load-bearing for the current situation."

@ cognition-log[] {
  text = "Pibi asks how to approach Koro while Koro is standing beside a food bowl."
}
"#,
        )
        .unwrap();
        let case = crate::cases::parse_module_case_file(&path).unwrap();
        let blackboard = Blackboard::default();
        let now = Utc.with_ymd_and_hms(2026, 5, 7, 12, 0, 0).unwrap();

        seed_cognition_log(&blackboard, &FixedClock(now), &case.cognition_log).await;

        let baseline = cognition_output_for_module(&blackboard, &builtin::cognition_gate()).await;
        assert!(!baseline.is_empty());
        assert!(!cognition_eval_has_new_output(
            &cognition_output_for_module(&blackboard, &builtin::cognition_gate()).await,
            &baseline,
        ));

        blackboard
            .append_cognition_log(
                ModuleInstanceId::new(builtin::cognition_gate(), ReplicaIndex::ZERO),
                CognitionLogEntry {
                    at: now,
                    text: "Approach Koro slowly from the side when he guards food.".to_string(),
                },
            )
            .await;

        assert!(cognition_eval_has_new_output(
            &cognition_output_for_module(&blackboard, &builtin::cognition_gate()).await,
            &baseline,
        ));
    }

    #[tokio::test]
    async fn allocation_controller_eval_bootstrap_guidance_is_not_completion() {
        let blackboard = Blackboard::default();
        let controller = builtin::allocation_controller();

        let mut allocation = blackboard.read(|bb| bb.allocation().clone()).await;
        let mut config = allocation.for_module(&controller);
        config.guidance = "Assign activation priorities for the current memo batch.".to_string();
        allocation.set(controller.clone(), config);
        blackboard
            .apply(BlackboardCommand::SetAllocation(allocation))
            .await;

        assert!(
            last_memo_log_content_for_module(&blackboard, &controller)
                .await
                .is_none()
        );
    }

    #[tokio::test]
    async fn seed_memos_stamps_requested_module_replica_and_offsets_time() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("memo-seed.eure");
        std::fs::write(
            &path,
            r#"
id = "memo-seed"
prompt = "What am I attending to?"

@ memos[] {
  module = "sensory"
  replica = 1
  content = "Koro gave a boundary signal"
  seconds-ago = 12
}
"#,
        )
        .unwrap();
        let case = crate::cases::parse_module_case_file(&path).unwrap();
        let blackboard = Blackboard::default();
        let now = Utc.with_ymd_and_hms(2026, 5, 7, 12, 0, 0).unwrap();

        let records = seed_memos(&blackboard, &FixedClock(now), &case.memos)
            .await
            .unwrap();

        assert_eq!(records.len(), 1);
        assert_eq!(records[0].owner.module, builtin::sensory());
        assert_eq!(records[0].owner.replica, ReplicaIndex::new(1));
        assert_eq!(records[0].content, "Koro gave a boundary signal");
        assert_eq!(records[0].written_at, now - ChronoDuration::seconds(12));

        let memo_logs = blackboard.read(|bb| bb.recent_memo_logs()).await;
        assert_eq!(memo_logs, records);
    }

    #[tokio::test]
    async fn attention_schema_appends_attention_experience_and_skips_noop_wakes() {
        let local = LocalSet::new();
        local
            .run_until(async {
                let blackboard = Blackboard::default();
                let first_entry = "I notice Koro's food-boundary signal.";
                let second_entry = "I feel my attention settle on Koro relaxing.";
                let adapter = MockLlmAdapter::new()
                    .with_text_scenario(attention_schema_tool_scenario(
                        "append-attention-1",
                        first_entry,
                    ))
                    .with_text_scenario(attention_schema_no_tool_scenario())
                    .with_text_scenario(attention_schema_tool_scenario(
                        "append-attention-2",
                        second_entry,
                    ))
                    .with_text_scenario(attention_schema_no_tool_scenario())
                    .with_text_scenario(attention_schema_no_tool_scenario())
                    .with_text_scenario(attention_schema_no_tool_scenario());
                let caps = test_caps_with_adapter(blackboard.clone(), adapter);
                let modules = ModuleRegistry::new()
                    .register(
                        eval_policy(1..=1, Bpm::from_f64(60_000.0)..=Bpm::from_f64(60_000.0)),
                        |caps| async move {
                            Ok(nuillu_attention_schema::AttentionSchemaModule::new(
                                caps.memo_updated_inbox(),
                                caps.cognition_log_updated_inbox(),
                                caps.blackboard_reader(),
                                caps.allocation_reader(),
                                caps.cognition_log_reader(),
                                caps.cognition_writer(),
                                caps.llm_access(),
                                caps.session("main")
                                    .with_auto_compaction(
                                        nuillu_attention_schema::session_auto_compaction(),
                                    )
                                    .await?,
                            ))
                        },
                    )
                    .unwrap()
                    .build(&caps)
                    .await
                    .unwrap();
                let memo_mailbox = caps.internal_harness_io().memo_updated_mailbox();
                let cognition_mailbox = caps.internal_harness_io().cognition_log_updated_mailbox();
                let sensory = ModuleInstanceId::new(builtin::sensory(), ReplicaIndex::ZERO);
                let cognition_source =
                    ModuleInstanceId::new(builtin::cognition_gate(), ReplicaIndex::ZERO);
                let now = Utc.with_ymd_and_hms(2026, 5, 7, 12, 0, 0).unwrap();
                let run_blackboard = blackboard.clone();

                run_agent(
                    modules,
                    AgentEventLoopConfig {
                        idle_threshold: Duration::from_millis(50),
                        activate_retries: 1,
                        module_failure_limit: 3,
                    },
                    async {
                        let record = run_blackboard
                            .update_memo(
                                sensory.clone(),
                                "Koro gave a food-boundary signal".to_owned(),
                                now,
                            )
                            .await;
                        memo_mailbox
                            .publish(MemoUpdated {
                                owner: sensory.clone(),
                                index: record.index,
                            })
                            .await
                            .expect("attention-schema receives seeded memo update");

                        for _ in 0..50 {
                            let count = attention_schema_cognition_output(&run_blackboard)
                                .await
                                .lines()
                                .count();
                            if count >= 1 {
                                break;
                            }
                            tokio::task::yield_now().await;
                            tokio::time::sleep(Duration::from_millis(1)).await;
                        }

                        run_blackboard
                            .apply(BlackboardCommand::AppendCognitionLog {
                                source: cognition_source.clone(),
                                entry: CognitionLogEntry {
                                    at: now,
                                    text: "Koro food-boundary signal is cognitively relevant"
                                        .to_owned(),
                                },
                            })
                            .await;
                        cognition_mailbox
                            .publish(CognitionLogUpdated::EntryAppended {
                                source: cognition_source.clone(),
                            })
                            .await
                            .expect("attention-schema receives cognition-log update");

                        for _ in 0..50 {
                            tokio::task::yield_now().await;
                            tokio::time::sleep(Duration::from_millis(1)).await;
                        }

                        let record = run_blackboard
                            .update_memo(
                                sensory.clone(),
                                "Koro relaxed after the boundary was respected".to_owned(),
                                now + ChronoDuration::seconds(1),
                            )
                            .await;
                        memo_mailbox
                            .publish(MemoUpdated {
                                owner: sensory,
                                index: record.index,
                            })
                            .await
                            .expect("attention-schema receives second memo update");

                        for _ in 0..80 {
                            let count = attention_schema_cognition_output(&run_blackboard)
                                .await
                                .lines()
                                .count();
                            if count >= 2 {
                                break;
                            }
                            tokio::task::yield_now().await;
                            tokio::time::sleep(Duration::from_millis(1)).await;
                        }
                    },
                )
                .await
                .unwrap();

                assert_eq!(
                    attention_schema_cognition_output(&blackboard).await,
                    [first_entry, second_entry].join("\n\n")
                );
            })
            .await;
    }

    #[test]
    fn full_agent_case_modules_filters_disabled() {
        let case = FullAgentCase {
            id: Some("x".to_string()),
            description: None,
            now: None,
            modules: Some(DEFAULT_FULL_AGENT_MODULES.to_vec()),
            inputs: Vec::new(),
            steps: Vec::new(),
            participants: Vec::new(),
            allow_empty_output: false,
            activate_allocation: Vec::new(),
            memories: Vec::new(),
            memos: Vec::new(),
            limits: crate::cases::EvalLimits {
                max_llm_calls: None,
                ..Default::default()
            },
            checks: Vec::new(),
            modules_checks: Vec::new(),
            scoring: Default::default(),
        };
        let modules = full_agent_case_modules(&case, &[EvalModule::QueryMemory]);
        assert!(!modules.contains(&EvalModule::QueryMemory));
        assert!(modules.contains(&EvalModule::Speak));
        assert!(modules.contains(&EvalModule::AllocationController));
    }

    #[test]
    fn validate_disabled_modules_rejects_required_modules() {
        for required in REQUIRED_FULL_AGENT_MODULES {
            let err = validate_disabled_modules(std::slice::from_ref(required)).unwrap_err();
            assert!(matches!(
                err,
                RunnerError::DisableRequiredModule { module } if module == required.as_str()
            ));
        }
        assert!(validate_disabled_modules(&[EvalModule::QueryMemory]).is_ok());
    }

    #[tokio::test]
    async fn eval_registry_and_allocation_include_only_selected_modules() {
        let selected = [
            EvalModule::Sensory,
            EvalModule::AllocationController,
            EvalModule::Speak,
        ];
        let allocation = full_agent_allocation(
            &crate::cases::EvalLimits {
                max_llm_calls: None,
                ..Default::default()
            },
            &selected,
        );
        assert!(allocation.get(&builtin::query_memory()).is_none());
        assert!(allocation.get(&builtin::speak_gate()).is_none());

        let blackboard = Blackboard::with_allocation(allocation);
        let caps = test_caps(blackboard.clone());
        let clock: Rc<dyn Clock> = Rc::new(SystemClock);
        let memory_caps = MemoryCapabilities::new(
            blackboard.clone(),
            clock.clone(),
            Rc::new(NoopMemoryStore),
            Vec::new(),
        );
        let policy_caps = PolicyCapabilities::new(
            blackboard.clone(),
            clock,
            Rc::new(nuillu_reward::NoopPolicyStore),
            Vec::new(),
        );
        let utterance_sink: Rc<dyn UtteranceSink> = Rc::new(nuillu_speak::NoopUtteranceSink);
        let allocated = eval_registry(
            &selected,
            &memory_caps,
            &policy_caps,
            &utterance_sink,
            ReplicaHardCap::PolicyMax,
        )
        .build(&caps)
        .await
        .unwrap();
        assert_eq!(allocated.len(), selected.len());

        let (replica_caps, allocation_modules) = blackboard
            .read(|bb| {
                let mut replica_caps = bb
                    .module_policies()
                    .keys()
                    .map(|module| module.as_str().to_owned())
                    .collect::<Vec<_>>();
                replica_caps.sort();
                let mut allocation_modules = bb
                    .allocation()
                    .iter()
                    .map(|(module, _)| module.as_str().to_owned())
                    .collect::<Vec<_>>();
                allocation_modules.sort();
                (replica_caps, allocation_modules)
            })
            .await;

        assert_eq!(
            replica_caps,
            vec!["allocation-controller", "sensory", "speak"]
        );
        assert_eq!(
            allocation_modules,
            vec!["allocation-controller", "sensory", "speak"]
        );
    }

    #[test]
    fn full_agent_allocation_bootstraps_input_and_autonomic_controllers() {
        let selected = DEFAULT_FULL_AGENT_MODULES.to_vec();
        let allocation = full_agent_allocation(
            &crate::cases::EvalLimits {
                max_llm_calls: None,
                ..Default::default()
            },
            &selected,
        );

        assert_eq!(
            allocation.activation_for(&builtin::sensory()),
            ActivationRatio::ONE
        );
        assert_eq!(allocation.tier_for(&builtin::sensory()), ModelTier::Cheap);

        assert_eq!(
            allocation.activation_for(&builtin::cognition_gate()),
            ActivationRatio::ZERO
        );
        assert_eq!(
            allocation.tier_for(&builtin::cognition_gate()),
            ModelTier::Cheap
        );

        assert_eq!(
            allocation.activation_for(&builtin::allocation_controller()),
            ActivationRatio::ONE
        );
        assert_eq!(
            allocation.tier_for(&builtin::allocation_controller()),
            ModelTier::Default
        );

        assert_eq!(
            allocation.activation_for(&builtin::interoception()),
            ActivationRatio::ONE
        );
        assert_eq!(
            allocation.tier_for(&builtin::interoception()),
            ModelTier::Cheap
        );
        assert_eq!(
            allocation.activation_for(&builtin::homeostatic_controller()),
            ActivationRatio::ONE
        );
        assert_eq!(
            allocation.tier_for(&builtin::homeostatic_controller()),
            ModelTier::Cheap
        );

        assert_eq!(
            allocation.activation_for(&builtin::speak()),
            ActivationRatio::ZERO
        );
        assert_eq!(allocation.tier_for(&builtin::speak()), ModelTier::Premium);

        for module in [
            builtin::attention_schema(),
            builtin::self_model(),
            builtin::query_memory(),
            builtin::memory(),
            builtin::memory_compaction(),
            builtin::memory_association(),
            builtin::memory_recombination(),
            builtin::predict(),
            builtin::surprise(),
        ] {
            assert_eq!(allocation.activation_for(&module), ActivationRatio::ZERO);
        }
    }

    #[tokio::test]
    async fn full_agent_gui_initial_allocation_starts_every_module_detached() {
        let selected = DEFAULT_FULL_AGENT_MODULES.to_vec();
        let allocation = full_agent_gui_initial_allocation(
            &crate::cases::EvalLimits {
                max_llm_calls: None,
                ..Default::default()
            },
            &selected,
        );
        let blackboard = Blackboard::with_allocation(allocation);
        let caps = test_caps(blackboard.clone());
        let clock: Rc<dyn Clock> = Rc::new(SystemClock);
        let memory_caps = MemoryCapabilities::new(
            blackboard.clone(),
            clock.clone(),
            Rc::new(NoopMemoryStore),
            Vec::new(),
        );
        let policy_caps = PolicyCapabilities::new(
            blackboard.clone(),
            clock,
            Rc::new(nuillu_reward::NoopPolicyStore),
            Vec::new(),
        );
        let utterance_sink: Rc<dyn UtteranceSink> = Rc::new(nuillu_speak::NoopUtteranceSink);
        let _allocated = eval_registry(
            &selected,
            &memory_caps,
            &policy_caps,
            &utterance_sink,
            ReplicaHardCap::PolicyMax,
        )
        .build(&caps)
        .await
        .unwrap();

        blackboard
            .read(|bb| {
                for module in &selected {
                    let id = module.module_id();
                    assert_eq!(bb.allocation().activation_for(&id), ActivationRatio::ZERO);
                    assert_eq!(bb.allocation().active_replicas(&id), 0);
                }
            })
            .await;
    }

    #[tokio::test]
    async fn gui_activate_applies_case_activation_allocation() {
        let selected = DEFAULT_FULL_AGENT_MODULES.to_vec();
        let allocation = full_agent_gui_initial_allocation(
            &crate::cases::EvalLimits {
                max_llm_calls: None,
                ..Default::default()
            },
            &selected,
        );
        let blackboard = Blackboard::with_allocation(allocation);
        let activate_allocation = vec![
            ActivateAllocation {
                module: EvalModule::Interoception,
                activation_ratio: 1.0,
            },
            ActivateAllocation {
                module: EvalModule::HomeostaticController,
                activation_ratio: 0.5,
            },
        ];

        activate_gui_start_modules(&blackboard, &activate_allocation).await;

        let allocation = blackboard.read(|bb| bb.allocation().clone()).await;
        assert_eq!(
            allocation.activation_for(&builtin::interoception()),
            ActivationRatio::ONE
        );
        assert_eq!(
            allocation.for_module(&builtin::interoception()).guidance,
            ""
        );
        assert_eq!(
            allocation.activation_for(&builtin::homeostatic_controller()),
            ActivationRatio::from_f64(0.5)
        );
        assert_eq!(
            allocation.activation_for(&builtin::sensory()),
            ActivationRatio::ZERO
        );
        assert_eq!(
            allocation.activation_for(&builtin::allocation_controller()),
            ActivationRatio::ZERO
        );
    }
}
