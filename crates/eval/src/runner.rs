use std::{
    any::Any,
    collections::BTreeMap,
    fs::{File, OpenOptions},
    io::{self, Write},
    panic::AssertUnwindSafe,
    path::{Path, PathBuf},
    sync::atomic::{AtomicBool, Ordering},
    sync::{Arc, Mutex, OnceLock},
    time::Duration,
};

use anyhow::{Context, Result};
use async_trait::async_trait;
use chrono::{Duration as ChronoDuration, Utc};
use futures::FutureExt as _;
use lutum::{
    Lutum, ModelName, RawTelemetryConfig, RequestExtensions, SharedPoolBudgetManager,
    SharedPoolBudgetOptions,
};
use lutum_eval::{RawTraceSnapshot, TraceSnapshot};
use lutum_in_memory_adapter::InMemoryAttentionRepository;
use lutum_libsql_adapter::{EmbeddingProfile, LibsqlMemoryStore, LibsqlMemoryStoreConfig};
use lutum_model2vec_adapter::PotionBase8MEmbedder;
use lutum_openai::{OpenAiAdapter, OpenAiReasoningEffort};
use nuillu_agent::{AgentEventLoopConfig, run as run_agent};
use nuillu_blackboard::{
    ActivationRatio, AttentionStreamEvent, Blackboard, BlackboardCommand, BlackboardInner,
    MemoryMetadata, ModuleConfig, ResourceAllocation,
};
use nuillu_module::ports::{
    Clock, Embedder, FileSearchHit, FileSearchProvider, FileSearchQuery, MemoryStore,
    NoopFileSearchProvider, PortError, SystemClock, Utterance, UtteranceSink,
};
use nuillu_module::{
    AttentionStreamUpdated, CapabilityProviders, LutumTiers, ModuleRegistry, RuntimeEvent,
    RuntimeEventSink, RuntimePolicy, SensoryInput,
};
use nuillu_types::{
    MemoryRank, ModelTier, ModuleId, ModuleInstanceId, ReplicaCapRange, ReplicaIndex, builtin,
};
use regex::RegexBuilder;
use serde::Serialize;
use thiserror::Error;
use tokio::task::LocalSet;
use tracing_subscriber::layer::SubscriberExt as _;

use crate::{
    artifact::CaseArtifact,
    cases::{
        CaseFileError, DEFAULT_FULL_AGENT_MODULES, EvalCase, EvalModule, FullAgentCase,
        FullAgentInput, ModuleCase, ModuleEvalTarget, discover_case_files, parse_case_file,
    },
    evaluation::{CaseReport, CaseSummary, SuiteReport, evaluate_case, normalize_text_block},
    judge::{LlmRubricJudge, RubricJudge},
    model_set::ReasoningEffort,
    state_dump::{
        AgenticDeadlockDump, AllocationModuleDump, AllocationProposalDump, AttentionEntryDump,
        AttentionStreamDump, BlackboardLastStateDump, DumpText, FullAgentLastStateCaseDump,
        FullAgentLastStateDump, MemoDump, MemoLogDump, MemoryEntryDump, MemoryLastStateDump,
        MemoryMetadataDump, ModuleInstanceDump, ReplicaCapDump, UtteranceDump,
        render_full_agent_last_state_eure,
    },
    trace_json::{raw_trace_has_error, raw_trace_snapshot_json, trace_snapshot_json},
};

const IDLE_REPORT_INTERVAL: Duration = Duration::from_secs(30);
const EVAL_MEMO_RETAINED_PER_OWNER: usize = 256;

#[derive(Debug, Clone)]
pub struct LlmBackendConfig {
    pub endpoint: String,
    pub token: String,
    pub model: String,
    pub reasoning_effort: Option<ReasoningEffort>,
    pub use_responses_api: bool,
}

#[derive(Debug, Clone)]
pub struct RunnerConfig {
    pub cases_root: PathBuf,
    pub output_root: PathBuf,
    pub run_id: String,
    pub judge_backend: LlmBackendConfig,
    pub cheap_backend: LlmBackendConfig,
    pub default_backend: LlmBackendConfig,
    pub premium_backend: LlmBackendConfig,
    pub model_dir: PathBuf,
    pub fail_fast: bool,
    pub case_patterns: Vec<String>,
}

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
}

struct CaseExecution {
    artifact: CaseArtifact,
    events: Vec<RuntimeEvent>,
}

struct CaseOutputContext<'a> {
    case_path: &'a Path,
    output_dir: &'a Path,
    case: &'a EvalCase,
    id: &'a str,
    reporter: &'a LiveReporter,
}

pub async fn run_suite(config: &RunnerConfig) -> Result<SuiteReport, RunnerError> {
    install_lutum_trace_subscriber()?;
    let mut case_paths =
        discover_case_files(&config.cases_root).map_err(|source| RunnerError::DiscoverCases {
            path: config.cases_root.clone(),
            source,
        })?;
    case_paths = filter_case_paths(case_paths, &config.case_patterns)?;
    let run_dir = config.output_root.join(&config.run_id);
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
        }),
        format!(
            "eval suite start run={} cases={} output={}",
            config.run_id,
            case_paths.len(),
            run_dir.display()
        ),
    )?;

    let judge_llm = build_lutum(
        &config.judge_backend.endpoint,
        &config.judge_backend.token,
        &config.judge_backend.model,
        config.judge_backend.reasoning_effort,
        config.judge_backend.use_responses_api,
    )
    .map_err(|error| RunnerError::Driver {
        path: config.cases_root.clone(),
        message: error.to_string(),
    })?;
    let judge = LlmRubricJudge::new(judge_llm);

    let mut cases = Vec::new();
    for path in case_paths {
        let output =
            run_case_detailed_with_reporter(&path, config, Some(&judge), &reporter).await?;
        let failed = !output.summary.passed || output.summary.invalid;
        cases.push(output.summary);
        if failed && config.fail_fast {
            break;
        }
    }

    let report = aggregate_suite(cases);
    let suite_path = run_dir.join("suite-report.json");
    write_json_file(&suite_path, &report)?;
    reporter.emit(
        None,
        "suite_finished",
        serde_json::json!({
            "case_count": report.case_count,
            "passed_cases": report.passed_cases,
            "failed_cases": report.failed_cases,
            "invalid_cases": report.invalid_cases,
            "mean_score": report.mean_score,
        }),
        format!(
            "eval suite end run={} passed={} failed={} invalid={} mean_score={:.3}",
            config.run_id,
            report.passed_cases,
            report.failed_cases,
            report.invalid_cases,
            report.mean_score
        ),
    )?;
    Ok(report)
}

fn backend_report(backend: &LlmBackendConfig) -> serde_json::Value {
    serde_json::json!({
        "endpoint": backend.endpoint.as_str(),
        "model": backend.model.as_str(),
        "reasoning_effort": backend.reasoning_effort,
        "use_responses_api": backend.use_responses_api,
    })
}

pub async fn run_case_detailed(
    case_path: &Path,
    config: &RunnerConfig,
    judge: Option<&dyn RubricJudge>,
) -> Result<CaseRunOutput, RunnerError> {
    install_lutum_trace_subscriber()?;
    let run_dir = config.output_root.join(&config.run_id);
    std::fs::create_dir_all(&run_dir).map_err(|source| RunnerError::WriteOutput {
        path: run_dir.clone(),
        source,
    })?;
    let reporter = LiveReporter::new(&config.run_id, &run_dir)?;
    run_case_detailed_with_reporter(case_path, config, judge, &reporter).await
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
    reporter.emit(
        Some(&id),
        "case_started",
        serde_json::json!({
            "path": case_path.display().to_string(),
            "output_dir": output_dir.display().to_string(),
        }),
        format!(
            "eval case start id={} path={} output={}",
            id,
            case_path.display(),
            output_dir.display()
        ),
    )?;

    let local = LocalSet::new();
    let collected = local
        .run_until(lutum_trace::capture_raw(
            AssertUnwindSafe(execute_case(&case, config, &output_dir, &id, reporter))
                .catch_unwind(),
        ))
        .await;

    let trace = collected.trace;
    let raw_trace = collected.raw;
    let execution = match collected.output {
        Ok(Ok(execution)) => execution,
        Ok(Err(error)) => {
            return write_runtime_failure_case_output(
                CaseOutputContext {
                    case_path,
                    output_dir: &output_dir,
                    case: &case,
                    id: &id,
                    reporter,
                },
                error.to_string(),
                trace,
                raw_trace,
            );
        }
        Err(payload) => {
            let message = format!("panic: {}", panic_payload_message(payload.as_ref()));
            return write_runtime_failure_case_output(
                CaseOutputContext {
                    case_path,
                    output_dir: &output_dir,
                    case: &case,
                    id: &id,
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
    let report = evaluate_case(&case, &trace, &artifact, judge).await;
    let summary = CaseSummary {
        path: case_path.display().to_string(),
        id,
        description: case
            .description()
            .map(|text| normalize_text_block(&text.content)),
        passed: report.passed(),
        invalid: report.invalid,
        score: report.score,
        report,
    };

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
    emit_case_finished(reporter, &summary, events.len())?;

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
    let summary = CaseSummary {
        path: ctx.case_path.display().to_string(),
        id: ctx.id.to_string(),
        description: ctx
            .case
            .description()
            .map(|text| normalize_text_block(&text.content)),
        passed: false,
        invalid: true,
        score: 0.0,
        report,
    };

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
        Some(&summary.id),
        "case_error",
        serde_json::json!({
            "path": summary.path.as_str(),
            "error": message,
        }),
        format!("eval case error id={} error={}", summary.id, message),
    )?;
    emit_case_finished(ctx.reporter, &summary, events.len())?;

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

fn emit_case_finished(
    reporter: &LiveReporter,
    summary: &CaseSummary,
    event_count: usize,
) -> Result<(), RunnerError> {
    let case_finished_message = if let Some(runtime_failure) = &summary.report.runtime_failure {
        format!(
            "eval case end id={} passed={} invalid={} score={:.3} events={} failure={}",
            summary.id,
            summary.passed,
            summary.invalid,
            summary.score,
            event_count,
            runtime_failure
        )
    } else {
        format!(
            "eval case end id={} passed={} invalid={} score={:.3} events={}",
            summary.id, summary.passed, summary.invalid, summary.score, event_count
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
            "runtime_failure": summary.report.runtime_failure.as_deref(),
            "events": event_count,
        }),
        case_finished_message,
    )
}

async fn execute_case(
    case: &EvalCase,
    config: &RunnerConfig,
    output_dir: &Path,
    case_id: &str,
    reporter: &LiveReporter,
) -> Result<CaseExecution> {
    match case {
        EvalCase::FullAgent(case) => {
            execute_full_agent_case(case, config, output_dir, case_id, reporter).await
        }
        EvalCase::Module { target, case } => {
            execute_module_case(*target, case, config, output_dir, case_id, reporter).await
        }
    }
}

async fn execute_full_agent_case(
    case: &FullAgentCase,
    config: &RunnerConfig,
    output_dir: &Path,
    case_id: &str,
    reporter: &LiveReporter,
) -> Result<CaseExecution> {
    let case_modules = full_agent_case_modules(case);
    let env = build_eval_environment(
        output_dir,
        config,
        full_agent_allocation(&case.limits, &case_modules),
        case.limits.max_llm_calls,
        Arc::new(NoopFileSearchProvider),
        case_id,
        reporter,
    )
    .await?;
    seed_memories(&env.caps, &case.memories).await?;

    let host = env.caps.host_io();
    let sensory = host.sensory_input_mailbox();
    let inputs = case.inputs.clone();
    let limits = case.limits.clone();
    let utterances = env.utterances.clone();
    let events = env.events.clone();
    let clock = env.clock.clone();
    let allocation_blackboard = env.blackboard.clone();
    let mut allocation_reporter =
        AllocationChangeReporter::new(case_id.to_string(), reporter.clone());
    let live_reporter = reporter.clone();
    let case_id_for_idle = case_id.to_string();
    let modules = eval_registry(&case_modules).build(&env.caps).await?;

    run_agent(
        modules,
        AgentEventLoopConfig {
            idle_threshold: Duration::from_secs(1),
            activate_retries: 2,
        },
        async move {
            let _ = allocation_reporter
                .emit_if_changed(&allocation_blackboard)
                .await;
            let now = clock.now();
            for input in inputs {
                let body = match input {
                    FullAgentInput::Heard { direction, content } => SensoryInput::Heard {
                        direction,
                        content: content.content,
                        observed_at: now,
                    },
                    FullAgentInput::Seen {
                        direction,
                        appearance,
                    } => SensoryInput::Seen {
                        direction,
                        appearance: appearance.content,
                        observed_at: now,
                    },
                };
                sensory
                    .publish(body)
                    .await
                    .expect("full-agent eval failed to publish SensoryInput");
            }

            let mut last_event_count = events.event_count();
            let mut idle_ticks = 0_u64;
            let idle_report_every_ticks = ticks_for_interval(IDLE_REPORT_INTERVAL, limits.tick_ms);
            for tick in 0..limits.max_ticks {
                if utterances.has_completed() || events.stop_requested() {
                    break;
                }
                tokio::task::yield_now().await;
                tokio::time::sleep(Duration::from_millis(limits.tick_ms)).await;
                let _ = allocation_reporter
                    .emit_if_changed(&allocation_blackboard)
                    .await;
                let event_count = events.event_count();
                if event_count == last_event_count {
                    idle_ticks = idle_ticks.saturating_add(1);
                } else {
                    last_event_count = event_count;
                    idle_ticks = 0;
                }
                if idle_ticks > 0 && idle_ticks.is_multiple_of(idle_report_every_ticks) {
                    let active_modules =
                        allocation_blackboard.read(active_module_observations).await;
                    let active_summary = active_modules_live_summary(&active_modules);
                    live_reporter
                        .emit_port(
                            Some(&case_id_for_idle),
                            "idle",
                            serde_json::json!({
                                "tick": tick + 1,
                                "events": event_count,
                                "idle_ticks": idle_ticks,
                                "idle_for_ms": idle_ticks.saturating_mul(limits.tick_ms),
                                "tick_ms": limits.tick_ms,
                                "report_interval_ms": duration_millis_u64(IDLE_REPORT_INTERVAL),
                                "active_modules": active_modules,
                            }),
                            format!(
                                "eval idle case={} idle_for_ms={} events={} active=[{}]",
                                case_id_for_idle,
                                idle_ticks.saturating_mul(limits.tick_ms),
                                event_count,
                                active_summary
                            ),
                        )
                        .expect("full-agent eval failed to write idle event");
                }
            }
        },
    )
    .await?;

    let mut artifact = if let Some(utterance) = env.utterances.first_complete() {
        CaseArtifact::new(utterance.text)
    } else if env.events.stop_requested() {
        CaseArtifact::failed("stopped after max-llm-calls")
    } else {
        CaseArtifact::failed("no utterance before max-ticks")
    };
    add_observations(&mut artifact, &env.blackboard, &env.utterances).await;
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
    Ok(CaseExecution { artifact, events })
}

async fn execute_module_case(
    target: ModuleEvalTarget,
    case: &ModuleCase,
    config: &RunnerConfig,
    output_dir: &Path,
    case_id: &str,
    reporter: &LiveReporter,
) -> Result<CaseExecution> {
    let case_modules = module_case_modules(target, case);
    let env = build_eval_environment(
        output_dir,
        config,
        module_allocation(target, &case.limits, &case_modules),
        case.limits.max_llm_calls,
        module_file_search_provider(target, case),
        case_id,
        reporter,
    )
    .await?;
    seed_memories(&env.caps, &case.memories).await?;
    seed_attention_stream(&env.blackboard, env.clock.as_ref(), &case.attention_stream).await;

    let target_module = module_id_for_target(target);
    let shutdown_target_module = target_module.clone();
    let modules = eval_registry(&case_modules).build(&env.caps).await?;
    let harness = env.caps.internal_harness_io();
    let limits = case.limits.clone();
    let prompt = case.prompt.content.clone();
    let events = env.events.clone();
    let blackboard = env.blackboard.clone();

    run_agent(
        modules,
        AgentEventLoopConfig {
            idle_threshold: Duration::from_secs(1),
            activate_retries: 2,
        },
        async move {
            match target {
                ModuleEvalTarget::QueryVector | ModuleEvalTarget::QueryAgentic => {
                    harness
                        .query_mailbox()
                        .publish(nuillu_module::QueryRequest::new(prompt))
                        .await
                        .expect("module eval failed to publish QueryRequest");
                }
                ModuleEvalTarget::AttentionSchema => {
                    harness
                        .attention_stream_updated_mailbox()
                        .publish(AttentionStreamUpdated::StreamAppended {
                            stream: ModuleInstanceId::new(
                                builtin::attention_gate(),
                                ReplicaIndex::ZERO,
                            ),
                        })
                        .await
                        .expect("module eval failed to publish AttentionStreamUpdated");
                }
                ModuleEvalTarget::SelfModel => {
                    harness
                        .self_model_mailbox()
                        .publish(nuillu_module::SelfModelRequest::new(prompt))
                        .await
                        .expect("module eval failed to publish SelfModelRequest");
                }
            }

            for _ in 0..limits.max_ticks {
                if events.stop_requested()
                    || blackboard.memo(&shutdown_target_module).await.is_some()
                {
                    break;
                }
                tokio::task::yield_now().await;
                tokio::time::sleep(Duration::from_millis(limits.tick_ms)).await;
            }
        },
    )
    .await?;

    let output = env
        .blackboard
        .memo(&target_module)
        .await
        .unwrap_or_default();
    let mut artifact = if output.is_empty() {
        if env.events.stop_requested() {
            CaseArtifact::failed("stopped after max-llm-calls before target module wrote a memo")
        } else {
            CaseArtifact::failed("target module did not write a memo before max-ticks")
        }
    } else {
        CaseArtifact::new(output)
    };
    add_observations(&mut artifact, &env.blackboard, &env.utterances).await;
    Ok(CaseExecution {
        artifact,
        events: env.events.snapshot(),
    })
}

struct EvalEnvironment {
    blackboard: Blackboard,
    caps: CapabilityProviders,
    memory: Arc<dyn MemoryStore>,
    utterances: Arc<RecordingUtteranceSink>,
    events: Arc<RecordingRuntimeEventSink>,
    clock: Arc<dyn Clock>,
}

async fn build_eval_environment(
    output_dir: &Path,
    config: &RunnerConfig,
    allocation: ResourceAllocation,
    max_llm_calls: Option<u64>,
    file_search: Arc<dyn FileSearchProvider>,
    case_id: &str,
    reporter: &LiveReporter,
) -> Result<EvalEnvironment> {
    let blackboard = Blackboard::with_allocation(allocation);
    let events = Arc::new(RecordingRuntimeEventSink::new(
        case_id.to_string(),
        max_llm_calls,
        reporter.clone(),
    ));
    let utterances = Arc::new(RecordingUtteranceSink::new(
        case_id.to_string(),
        reporter.clone(),
    ));
    let clock: Arc<dyn Clock> = Arc::new(SystemClock);
    let memory: Arc<dyn MemoryStore> = Arc::new(connect_memory_store(output_dir, config).await?);
    let tiers = build_tiers(config)?;
    let runtime_policy = RuntimePolicy {
        memo_retained_per_owner: EVAL_MEMO_RETAINED_PER_OWNER,
        ..RuntimePolicy::default()
    };
    let caps = CapabilityProviders::new_with_runtime_policy(
        blackboard.clone(),
        Arc::new(InMemoryAttentionRepository::new()),
        memory.clone(),
        Vec::new(),
        file_search,
        utterances.clone(),
        clock.clone(),
        tiers,
        events.clone(),
        runtime_policy,
    );

    Ok(EvalEnvironment {
        blackboard,
        caps,
        memory,
        utterances,
        events,
        clock,
    })
}

async fn connect_memory_store(
    output_dir: &Path,
    config: &RunnerConfig,
) -> Result<LibsqlMemoryStore> {
    let embedder = PotionBase8MEmbedder::from_local_dir(&config.model_dir)
        .with_context(|| format!("load model2vec model from {}", config.model_dir.display()))?;
    let dimensions = embedder.dimensions();
    let profile = EmbeddingProfile::new("potion-base-8M", "local", dimensions);
    let db_path = output_dir.join("memory.db");
    LibsqlMemoryStore::connect(
        LibsqlMemoryStoreConfig::local(db_path, dimensions).with_active_profile(profile),
        Box::new(embedder),
    )
    .await
    .context("connect libsql memory store")
}

fn module_file_search_provider(
    target: ModuleEvalTarget,
    case: &ModuleCase,
) -> Arc<dyn FileSearchProvider> {
    if target == ModuleEvalTarget::QueryAgentic && !case.files.is_empty() {
        Arc::new(SeededFileSearchProvider::new(case.files.clone()))
    } else {
        Arc::new(NoopFileSearchProvider)
    }
}

#[derive(Debug)]
struct SeededFileSearchProvider {
    files: Vec<SeededFile>,
}

#[derive(Debug)]
struct SeededFile {
    path: String,
    lines: Vec<String>,
}

impl SeededFileSearchProvider {
    fn new(files: Vec<crate::cases::FileSeed>) -> Self {
        Self {
            files: files
                .into_iter()
                .map(|file| SeededFile {
                    path: file.path,
                    lines: file.content.content.lines().map(str::to_owned).collect(),
                })
                .collect(),
        }
    }
}

#[async_trait(?Send)]
impl FileSearchProvider for SeededFileSearchProvider {
    async fn search(&self, query: &FileSearchQuery) -> Result<Vec<FileSearchHit>, PortError> {
        if query.pattern.is_empty() || query.max_matches == 0 {
            return Ok(Vec::new());
        }

        let regex = if query.regex {
            Some(
                RegexBuilder::new(&query.pattern)
                    .case_insensitive(!query.case_sensitive)
                    .build()
                    .map_err(|error| PortError::InvalidInput(error.to_string()))?,
            )
        } else {
            None
        };
        let literal = if query.case_sensitive {
            query.pattern.clone()
        } else {
            query.pattern.to_lowercase()
        };

        let mut hits = Vec::new();
        for file in &self.files {
            for (line_index, line) in file.lines.iter().enumerate() {
                let line_matches = match &regex {
                    Some(regex) => regex.is_match(line),
                    None if query.case_sensitive => line.contains(&literal),
                    None => line.to_lowercase().contains(&literal),
                };
                let line_matches = if query.invert_match {
                    !line_matches
                } else {
                    line_matches
                };
                if !line_matches {
                    continue;
                }

                let start = line_index.saturating_sub(query.context);
                let end = (line_index + query.context + 1).min(file.lines.len());
                hits.push(FileSearchHit {
                    path: file.path.clone(),
                    line: line_index + 1,
                    snippet: file.lines[start..end].join("\n"),
                });
                if hits.len() >= query.max_matches {
                    return Ok(hits);
                }
            }
        }
        Ok(hits)
    }
}

async fn seed_memories(
    caps: &CapabilityProviders,
    memories: &[crate::cases::MemorySeed],
) -> Result<()> {
    let writer = caps.memory_writer();
    for memory in memories {
        writer
            .insert(
                memory.content.content.clone(),
                MemoryRank::from(memory.rank),
                memory.decay_secs,
            )
            .await
            .context("seed eval memory")?;
    }
    Ok(())
}

async fn seed_attention_stream(
    blackboard: &Blackboard,
    clock: &dyn Clock,
    seeds: &[crate::cases::AttentionSeed],
) {
    let stream = ModuleInstanceId::new(builtin::attention_gate(), ReplicaIndex::ZERO);
    let now = clock.now();
    for seed in seeds {
        blackboard
            .apply(BlackboardCommand::AppendAttentionStream {
                stream: stream.clone(),
                event: AttentionStreamEvent {
                    at: now - ChronoDuration::seconds(seed.seconds_ago),
                    text: seed.text.content.clone(),
                },
            })
            .await;
    }
}

fn build_tiers(config: &RunnerConfig) -> Result<LutumTiers> {
    let cheap = build_lutum(
        &config.cheap_backend.endpoint,
        &config.cheap_backend.token,
        &config.cheap_backend.model,
        config.cheap_backend.reasoning_effort,
        config.cheap_backend.use_responses_api,
    )?;
    let default = build_lutum(
        &config.default_backend.endpoint,
        &config.default_backend.token,
        &config.default_backend.model,
        config.default_backend.reasoning_effort,
        config.default_backend.use_responses_api,
    )?;
    let premium = build_lutum(
        &config.premium_backend.endpoint,
        &config.premium_backend.token,
        &config.premium_backend.model,
        config.premium_backend.reasoning_effort,
        config.premium_backend.use_responses_api,
    )?;
    Ok(LutumTiers {
        cheap,
        default,
        premium,
    })
}

fn build_lutum(
    endpoint: &str,
    token: &str,
    model: &str,
    reasoning_effort: Option<ReasoningEffort>,
    use_responses_api: bool,
) -> Result<Lutum> {
    let adapter = OpenAiAdapter::new(token.to_owned())
        .with_base_url(endpoint.to_owned())
        .with_default_model(ModelName::new(model)?)
        .with_resolve_reasoning_effort(ConfiguredReasoningEffort);
    let adapter = if use_responses_api {
        adapter
    } else {
        adapter.with_chat_completions()
    };
    let lutum = Lutum::new(
        Arc::new(adapter),
        SharedPoolBudgetManager::new(SharedPoolBudgetOptions::default()),
    )
    .with_extension(RawTelemetryConfig::all());
    Ok(match reasoning_effort {
        Some(reasoning_effort) => {
            lutum.with_extension(ReasoningEffortConfig(reasoning_effort.into()))
        }
        None => lutum,
    })
}

#[derive(Clone, Copy)]
struct ReasoningEffortConfig(OpenAiReasoningEffort);

#[lutum::impl_hook(lutum_openai::ResolveReasoningEffort)]
async fn configured_reasoning_effort(
    extensions: &RequestExtensions,
) -> Option<OpenAiReasoningEffort> {
    extensions
        .get::<ReasoningEffortConfig>()
        .map(|value| value.0)
}

fn full_agent_case_modules(case: &FullAgentCase) -> Vec<EvalModule> {
    case.modules
        .clone()
        .unwrap_or_else(|| DEFAULT_FULL_AGENT_MODULES.to_vec())
}

fn module_case_modules(target: ModuleEvalTarget, case: &ModuleCase) -> Vec<EvalModule> {
    case.modules
        .clone()
        .unwrap_or_else(|| vec![target.module()])
}

fn eval_registry(modules: &[EvalModule]) -> ModuleRegistry {
    let mut registry = ModuleRegistry::new();
    for module in modules {
        registry = register_eval_module(registry, *module);
    }
    registry
}

fn register_eval_module(registry: ModuleRegistry, module: EvalModule) -> ModuleRegistry {
    match module {
        EvalModule::Sensory => registry
            .register(builtin::sensory(), 0..=1, |caps| {
                nuillu_sensory::SensoryModule::new(
                    caps.sensory_input_inbox(),
                    caps.sensory_detail_inbox(),
                    caps.allocation_reader(),
                    caps.memo(),
                    caps.clock(),
                    caps.llm_access(),
                )
            })
            .expect("eval module registration should be unique"),
        EvalModule::AttentionGate => registry
            .register(builtin::attention_gate(), 0..=1, |caps| {
                nuillu_attention_gate::AttentionGateModule::new(
                    caps.memo_updated_inbox(),
                    caps.blackboard_reader(),
                    caps.allocation_reader(),
                    caps.attention_writer(),
                    caps.time_division(),
                    caps.llm_access(),
                )
            })
            .expect("eval module registration should be unique"),
        EvalModule::AttentionController => registry
            .register(builtin::attention_controller(), 0..=1, |caps| {
                nuillu_attention_controller::AttentionControllerModule::new(
                    caps.memo_updated_inbox(),
                    caps.blackboard_reader(),
                    caps.attention_reader(),
                    caps.allocation_reader(),
                    caps.allocation_writer(),
                    caps.memo(),
                    caps.llm_access(),
                )
            })
            .expect("eval module registration should be unique"),
        EvalModule::AttentionSchema => registry
            .register(builtin::attention_schema(), 0..=1, |caps| {
                nuillu_attention_schema::AttentionSchemaModule::new(
                    caps.attention_stream_updated_inbox(),
                    caps.attention_reader(),
                    caps.memo(),
                    caps.llm_access(),
                )
            })
            .expect("eval module registration should be unique"),
        EvalModule::SelfModel => registry
            .register(builtin::self_model(), 0..=1, |caps| {
                nuillu_self_model::SelfModelModule::new(
                    caps.self_model_inbox(),
                    caps.blackboard_reader(),
                    caps.memo(),
                    caps.llm_access(),
                )
            })
            .expect("eval module registration should be unique"),
        EvalModule::QueryVector => registry
            .register(builtin::query_vector(), 1..=1, |caps| {
                nuillu_query_vector::QueryVectorModule::new(
                    caps.query_inbox(),
                    caps.attention_stream_updated_inbox(),
                    caps.allocation_reader(),
                    caps.blackboard_reader(),
                    caps.vector_memory_searcher(),
                    caps.memo(),
                    caps.llm_access(),
                )
            })
            .expect("eval module registration should be unique"),
        EvalModule::QueryAgentic => registry
            .register(builtin::query_agentic(), 0..=1, |caps| {
                nuillu_query_agentic::QueryAgenticModule::new(
                    caps.query_inbox(),
                    caps.allocation_updated_inbox(),
                    caps.allocation_reader(),
                    caps.blackboard_reader(),
                    caps.file_searcher(),
                    caps.memo(),
                    caps.llm_access(),
                )
            })
            .expect("eval module registration should be unique"),
        EvalModule::Memory => registry
            .register(builtin::memory(), 0..=1, |caps| {
                nuillu_memory::MemoryModule::new(
                    caps.attention_stream_updated_inbox(),
                    caps.memory_request_inbox(),
                    caps.allocation_reader(),
                    caps.blackboard_reader(),
                    caps.memory_writer(),
                    caps.llm_access(),
                )
            })
            .expect("eval module registration should be unique"),
        EvalModule::MemoryCompaction => registry
            .register(builtin::memory_compaction(), 0..=1, |caps| {
                nuillu_memory_compaction::MemoryCompactionModule::new(
                    caps.allocation_updated_inbox(),
                    caps.allocation_reader(),
                    caps.blackboard_reader(),
                    caps.memory_compactor(),
                    caps.llm_access(),
                )
            })
            .expect("eval module registration should be unique"),
        EvalModule::Predict => registry
            .register(builtin::predict(), 0..=1, |caps| {
                nuillu_predict::PredictModule::new(
                    caps.attention_stream_updated_inbox(),
                    caps.attention_reader(),
                    caps.allocation_reader(),
                    caps.blackboard_reader(),
                    caps.memo(),
                    caps.llm_access(),
                )
            })
            .expect("eval module registration should be unique"),
        EvalModule::Surprise => registry
            .register(builtin::surprise(), 0..=1, |caps| {
                nuillu_surprise::SurpriseModule::new(
                    caps.attention_stream_updated_inbox(),
                    caps.attention_reader(),
                    caps.allocation_reader(),
                    caps.blackboard_reader(),
                    caps.memory_request_mailbox(),
                    caps.memo(),
                    caps.llm_access(),
                )
            })
            .expect("eval module registration should be unique"),
        EvalModule::SpeakGate => registry
            .register(builtin::speak_gate(), 0..=1, |caps| {
                nuillu_speak::SpeakGateModule::new(
                    caps.attention_stream_updated_inbox(),
                    caps.attention_reader(),
                    caps.blackboard_reader(),
                    caps.module_status_reader(),
                    caps.query_mailbox(),
                    caps.self_model_mailbox(),
                    caps.sensory_detail_mailbox(),
                    caps.memo(),
                    caps.speak_mailbox(),
                    caps.llm_access(),
                )
            })
            .expect("eval module registration should be unique"),
        EvalModule::Speak => registry
            .register(builtin::speak(), 0..=1, |caps| {
                nuillu_speak::SpeakModule::new(
                    caps.speak_inbox(),
                    caps.attention_reader(),
                    caps.memo(),
                    caps.utterance_writer(),
                    caps.llm_access(),
                )
            })
            .expect("eval module registration should be unique"),
    }
}

fn full_agent_allocation(
    _limits: &crate::cases::EvalLimits,
    modules: &[EvalModule],
) -> ResourceAllocation {
    let mut allocation = ResourceAllocation::default();

    for module in modules {
        match module {
            EvalModule::Sensory => set_allocation_module(
                &mut allocation,
                module.module_id(),
                1.0,
                ModelTier::Cheap,
                "Filter incoming observations into sensory memo.",
            ),
            EvalModule::AttentionGate => set_allocation_module(
                &mut allocation,
                module.module_id(),
                0.0,
                ModelTier::Cheap,
                "Wait for controller guidance before promoting memos into attention.",
            ),
            EvalModule::AttentionController => set_allocation_module(
                &mut allocation,
                module.module_id(),
                1.0,
                ModelTier::Premium,
                "Allocate modules from memo updates and write natural-language guidance.",
            ),
            EvalModule::AttentionSchema => set_allocation_module(
                &mut allocation,
                module.module_id(),
                0.0,
                ModelTier::Default,
                "Idle until attention-stream updates require attention-state modeling.",
            ),
            EvalModule::SelfModel => set_allocation_module(
                &mut allocation,
                module.module_id(),
                0.0,
                ModelTier::Default,
                "Idle until explicit self-model requests require work.",
            ),
            EvalModule::QueryVector => set_allocation_module(
                &mut allocation,
                module.module_id(),
                0.0,
                ModelTier::Cheap,
                "Idle until memory retrieval is needed.",
            ),
            EvalModule::QueryAgentic => set_allocation_module(
                &mut allocation,
                module.module_id(),
                0.0,
                ModelTier::Premium,
                "Idle until file lookup is needed.",
            ),
            EvalModule::Memory => set_allocation_module(
                &mut allocation,
                module.module_id(),
                0.0,
                ModelTier::Cheap,
                "Idle until preservation guidance or memory requests arrive.",
            ),
            EvalModule::MemoryCompaction => set_allocation_module(
                &mut allocation,
                module.module_id(),
                0.0,
                ModelTier::Cheap,
                "Idle until compaction guidance arrives.",
            ),
            EvalModule::Predict => set_allocation_module(
                &mut allocation,
                module.module_id(),
                0.0,
                ModelTier::Cheap,
                "Idle until prediction guidance arrives.",
            ),
            EvalModule::Surprise => set_allocation_module(
                &mut allocation,
                module.module_id(),
                0.0,
                ModelTier::Default,
                "Idle until surprise detection is useful.",
            ),
            EvalModule::SpeakGate => set_allocation_module(
                &mut allocation,
                module.module_id(),
                1.0,
                ModelTier::Premium,
                "Decide whether attention is ready for speech or which evidence is missing.",
            ),
            EvalModule::Speak => set_allocation_module(
                &mut allocation,
                module.module_id(),
                1.0,
                ModelTier::Premium,
                "Wait for typed SpeakRequest.",
            ),
        }
    }
    allocation
}

fn module_allocation(
    target: ModuleEvalTarget,
    _limits: &crate::cases::EvalLimits,
    modules: &[EvalModule],
) -> ResourceAllocation {
    let mut allocation = ResourceAllocation::default();
    let target_module = target.module();
    for module in modules {
        let is_target = *module == target_module;
        allocation.set(
            module.module_id(),
            ModuleConfig {
                activation_ratio: if is_target {
                    ActivationRatio::ONE
                } else {
                    ActivationRatio::ZERO
                },
                guidance: if is_target {
                    "Handle the module eval request.".into()
                } else {
                    "Registered for this module eval; idle unless activated by a typed signal."
                        .into()
                },
                tier: eval_module_tier(*module),
                ..Default::default()
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
    guidance: impl Into<String>,
) {
    allocation.set(
        id,
        ModuleConfig {
            activation_ratio: ActivationRatio::from_f64(activation_ratio),
            guidance: guidance.into(),
            tier,
            ..Default::default()
        },
    );
}

fn module_id_for_target(target: ModuleEvalTarget) -> ModuleId {
    target.module().module_id()
}

fn eval_module_tier(module: EvalModule) -> ModelTier {
    match module {
        EvalModule::Sensory
        | EvalModule::AttentionGate
        | EvalModule::QueryVector
        | EvalModule::Memory
        | EvalModule::MemoryCompaction
        | EvalModule::Predict => ModelTier::Cheap,
        EvalModule::AttentionController
        | EvalModule::QueryAgentic
        | EvalModule::SpeakGate
        | EvalModule::Speak => ModelTier::Premium,
        EvalModule::AttentionSchema | EvalModule::SelfModel | EvalModule::Surprise => {
            ModelTier::Default
        }
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
    let attention_stream_set = bb.attention_stream_set();
    BlackboardLastStateDump {
        memos: memo_dumps(bb),
        memo_logs: memo_log_dumps(bb),
        attention_streams: attention_stream_set
            .streams()
            .iter()
            .map(|record| AttentionStreamDump {
                stream: module_instance_dump(&record.stream),
                entries: record
                    .entries
                    .iter()
                    .map(|event| AttentionEntryDump {
                        at: event.at.to_rfc3339(),
                        text: DumpText::new(event.text.clone()),
                    })
                    .collect(),
            })
            .collect(),
        agentic_deadlock: attention_stream_set
            .agentic_deadlock_marker()
            .map(|marker| AgenticDeadlockDump {
                at: marker.at.to_rfc3339(),
                idle_for_ms: duration_millis_u64(marker.idle_for),
            }),
        base_allocation: allocation_module_dumps(bb.base_allocation()),
        allocation: allocation_module_dumps(bb.allocation()),
        allocation_proposals: allocation_proposal_dumps(bb),
        replica_caps: replica_cap_dumps(bb),
    }
}

fn memo_dumps(bb: &BlackboardInner) -> Vec<MemoDump> {
    bb.memo_records()
        .into_iter()
        .map(|record| MemoDump {
            module: record.owner.module.as_str().to_owned(),
            replica: record.owner.replica.get(),
            memo: DumpText::new(record.memo),
        })
        .collect()
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
            activation_ratio: config.activation_ratio.as_f64(),
            active_replicas: allocation.active_replicas(module),
            tier: model_tier_name(config.tier).to_owned(),
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
        .replica_caps()
        .iter()
        .map(|(module, range)| ReplicaCapDump {
            module: module.as_str().to_owned(),
            min: range.min,
            max: range.max,
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
                    decay_remaining_secs: metadata.decay_remaining_secs,
                    remember_tokens: metadata.remember_tokens,
                    last_accessed: metadata.last_accessed.to_rfc3339(),
                    access_count: metadata.access_count,
                    query_history: metadata
                        .query_history
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
                metadata,
                missing_content: false,
            },
            None => MemoryEntryDump {
                index,
                content: None,
                content_rank: None,
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
            text: DumpText::new(utterance.text),
            emitted_at: utterance.emitted_at,
        })
        .collect()
}

#[derive(Debug, Clone, Serialize)]
struct AgentObservation {
    memos: BTreeMap<String, Vec<ReplicaMemoObservation>>,
    memo_logs: BTreeMap<String, Vec<MemoLogObservation>>,
    attention_streams: Vec<AttentionStreamObservation>,
    allocation: BTreeMap<String, ModuleConfig>,
    allocation_proposals: BTreeMap<String, BTreeMap<String, ModuleConfig>>,
    replica_caps: BTreeMap<String, ReplicaCapRange>,
    memory_metadata: BTreeMap<String, MemoryMetadata>,
    utterances: Vec<RecordedUtterance>,
}

impl AgentObservation {
    fn from_blackboard(bb: &BlackboardInner, utterances: Vec<RecordedUtterance>) -> Self {
        Self {
            memos: memo_observations(bb),
            memo_logs: memo_log_observations(bb),
            attention_streams: attention_stream_observations(bb),
            allocation: allocation_observation(bb.allocation()),
            allocation_proposals: allocation_proposal_observations(bb),
            replica_caps: replica_cap_observations(bb),
            memory_metadata: memory_metadata_observations(bb),
            utterances,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
struct ReplicaMemoObservation {
    replica: u8,
    memo: String,
}

#[derive(Debug, Clone, Serialize)]
struct MemoLogObservation {
    replica: u8,
    index: u64,
    written_at: String,
    content: String,
}

#[derive(Debug, Clone, Serialize)]
struct AttentionStreamObservation {
    stream: ModuleInstanceObservation,
    entries: Vec<AttentionStreamEvent>,
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

fn memo_observations(bb: &BlackboardInner) -> BTreeMap<String, Vec<ReplicaMemoObservation>> {
    let mut memos = BTreeMap::<String, Vec<ReplicaMemoObservation>>::new();
    for record in bb.memo_records() {
        memos
            .entry(record.owner.module.as_str().to_owned())
            .or_default()
            .push(ReplicaMemoObservation {
                replica: record.owner.replica.get(),
                memo: record.memo,
            });
    }
    memos
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

fn attention_stream_observations(bb: &BlackboardInner) -> Vec<AttentionStreamObservation> {
    bb.attention_stream_set()
        .streams()
        .iter()
        .map(|record| AttentionStreamObservation {
            stream: module_instance_observation(&record.stream),
            entries: record.entries.clone(),
        })
        .collect()
}

fn allocation_observation(allocation: &ResourceAllocation) -> BTreeMap<String, ModuleConfig> {
    allocation
        .iter()
        .map(|(module, config)| (module.as_str().to_owned(), config.clone()))
        .collect()
}

fn allocation_proposal_observations(
    bb: &BlackboardInner,
) -> BTreeMap<String, BTreeMap<String, ModuleConfig>> {
    bb.allocation_proposals()
        .iter()
        .map(|(owner, allocation)| (owner.to_string(), allocation_observation(allocation)))
        .collect()
}

fn replica_cap_observations(bb: &BlackboardInner) -> BTreeMap<String, ReplicaCapRange> {
    bb.replica_caps()
        .iter()
        .map(|(module, range)| (module.as_str().to_owned(), *range))
        .collect()
}

fn active_module_observations(bb: &BlackboardInner) -> Vec<ActiveModuleObservation> {
    let mut modules = bb
        .replica_caps()
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
            let config = bb.allocation().for_module(&module);
            Some(ActiveModuleObservation {
                module: module.as_str().to_owned(),
                active_replicas,
                activation_ratio: config.activation_ratio,
                tier: config.tier,
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

fn memory_rank_name(rank: MemoryRank) -> &'static str {
    match rank {
        MemoryRank::ShortTerm => "short-term",
        MemoryRank::MidTerm => "mid-term",
        MemoryRank::LongTerm => "long-term",
        MemoryRank::Permanent => "permanent",
    }
}

fn model_tier_name(tier: ModelTier) -> &'static str {
    match tier {
        ModelTier::Cheap => "cheap",
        ModelTier::Default => "default",
        ModelTier::Premium => "premium",
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

fn allocation_live_summary(allocation: &BTreeMap<String, ModuleConfig>) -> String {
    let active = allocation
        .iter()
        .filter(|(_, config)| config.activation_ratio > ActivationRatio::ZERO)
        .map(|(module, config)| {
            format!(
                "{}:{:.2}/{:?}",
                module,
                config.activation_ratio.as_f64(),
                config.tier
            )
        })
        .collect::<Vec<_>>();
    let inactive = allocation
        .values()
        .filter(|config| config.activation_ratio == ActivationRatio::ZERO)
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

fn ticks_for_interval(interval: Duration, tick_ms: u64) -> u64 {
    (duration_millis_u64(interval) / tick_ms.max(1)).max(1)
}

fn duration_millis_u64(duration: Duration) -> u64 {
    duration.as_millis().min(u128::from(u64::MAX)) as u64
}

#[derive(Clone)]
struct LiveReporter {
    run_id: String,
    path: PathBuf,
    file: Arc<Mutex<File>>,
}

impl std::fmt::Debug for LiveReporter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LiveReporter")
            .field("run_id", &self.run_id)
            .field("path", &self.path)
            .finish_non_exhaustive()
    }
}

impl LiveReporter {
    fn new(run_id: &str, run_dir: &Path) -> Result<Self, RunnerError> {
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
        })
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
            .map_err(|error| PortError::Backend(format!("write live eval event: {error}")))
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
    text: String,
    emitted_at: String,
}

#[derive(Clone)]
struct RecordingUtteranceSink {
    case_id: String,
    reporter: LiveReporter,
    complete: Arc<Mutex<Vec<RecordedUtterance>>>,
}

impl std::fmt::Debug for RecordingUtteranceSink {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RecordingUtteranceSink")
            .field("case_id", &self.case_id)
            .finish_non_exhaustive()
    }
}

impl RecordingUtteranceSink {
    fn new(case_id: String, reporter: LiveReporter) -> Self {
        Self {
            case_id,
            reporter,
            complete: Arc::new(Mutex::new(Vec::new())),
        }
    }

    fn has_completed(&self) -> bool {
        !self
            .complete
            .lock()
            .expect("utterance lock poisoned")
            .is_empty()
    }

    fn first_complete(&self) -> Option<RecordedUtterance> {
        self.complete
            .lock()
            .expect("utterance lock poisoned")
            .first()
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
        let recorded = RecordedUtterance {
            sender: utterance.sender.to_string(),
            text: utterance.text,
            emitted_at: utterance.emitted_at.to_rfc3339(),
        };
        self.complete
            .lock()
            .map_err(|_| PortError::Backend("utterance lock poisoned".into()))?
            .push(recorded.clone());
        self.reporter.emit_port(
            Some(&self.case_id),
            "utterance_completed",
            serde_json::json!({
                "sender": recorded.sender.clone(),
                "text": recorded.text.clone(),
                "emitted_at": recorded.emitted_at.clone(),
            }),
            format!(
                "eval utterance case={} sender={} chars={}",
                self.case_id,
                recorded.sender,
                recorded.text.chars().count()
            ),
        )?;
        Ok(())
    }
}

#[derive(Debug)]
struct RecordingRuntimeEventSink {
    events: Mutex<Vec<RuntimeEvent>>,
    stop: AtomicBool,
    case_id: String,
    max_llm_calls: Option<u64>,
    reporter: LiveReporter,
}

impl RecordingRuntimeEventSink {
    fn new(case_id: String, max_llm_calls: Option<u64>, reporter: LiveReporter) -> Self {
        Self {
            events: Mutex::new(Vec::new()),
            stop: AtomicBool::new(false),
            case_id,
            max_llm_calls,
            reporter,
        }
    }

    fn snapshot(&self) -> Vec<RuntimeEvent> {
        self.events
            .lock()
            .expect("runtime event lock poisoned")
            .clone()
    }

    fn stop_requested(&self) -> bool {
        self.stop.load(Ordering::Relaxed)
    }

    fn event_count(&self) -> usize {
        self.events
            .lock()
            .expect("runtime event lock poisoned")
            .len()
    }
}

#[async_trait(?Send)]
impl RuntimeEventSink for RecordingRuntimeEventSink {
    async fn on_event(&self, event: RuntimeEvent) -> Result<(), PortError> {
        let should_stop = match &event {
            RuntimeEvent::LlmAccessed { call, .. } => self
                .max_llm_calls
                .is_some_and(|max| call.saturating_add(1) >= max),
            RuntimeEvent::MemoUpdated { .. } => false,
            RuntimeEvent::RateLimitDelayed { .. } => false,
            RuntimeEvent::ModuleBatchThrottled { .. } => false,
        };
        let live_message = match &event {
            RuntimeEvent::LlmAccessed {
                call, owner, tier, ..
            } => format!(
                "eval llm-accessed case={} call={} owner={} tier={:?}",
                self.case_id, call, owner, tier
            ),
            RuntimeEvent::MemoUpdated {
                owner, char_count, ..
            } => format!(
                "eval memo-updated case={} owner={} chars={}",
                self.case_id, owner, char_count
            ),
            RuntimeEvent::RateLimitDelayed {
                owner,
                capability,
                delayed_for,
                ..
            } => format!(
                "eval rate-limit-delayed case={} owner={} capability={:?} delayed_ms={}",
                self.case_id,
                owner,
                capability,
                delayed_for.as_millis()
            ),
            RuntimeEvent::ModuleBatchThrottled {
                owner, delayed_for, ..
            } => format!(
                "eval module-batch-throttled case={} owner={} delayed_ms={}",
                self.case_id,
                owner,
                delayed_for.as_millis()
            ),
        };
        self.events
            .lock()
            .map_err(|_| PortError::Backend("runtime event lock poisoned".into()))?
            .push(event.clone());
        self.reporter.emit_port(
            Some(&self.case_id),
            "runtime_event",
            serde_json::json!({ "event": event }),
            live_message,
        )?;
        if should_stop && !self.stop.swap(true, Ordering::Relaxed) {
            self.reporter.emit_port(
                Some(&self.case_id),
                "stop_requested",
                serde_json::json!({ "reason": "max-llm-calls" }),
                format!(
                    "eval stop requested case={} reason=max-llm-calls",
                    self.case_id
                ),
            )?;
        }
        Ok(())
    }
}

fn aggregate_suite(cases: Vec<CaseSummary>) -> SuiteReport {
    let case_count = cases.len();
    let passed_cases = cases.iter().filter(|case| case.passed).count();
    let invalid_cases = cases.iter().filter(|case| case.invalid).count();
    let failed_cases = case_count.saturating_sub(passed_cases + invalid_cases);
    let mean_score = if cases.is_empty() {
        0.0
    } else {
        cases.iter().map(|case| case.score).sum::<f64>() / cases.len() as f64
    };

    SuiteReport {
        case_count,
        passed_cases,
        failed_cases,
        invalid_cases,
        mean_score,
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

pub fn install_lutum_trace_subscriber() -> Result<(), RunnerError> {
    static INSTALL_RESULT: OnceLock<Result<(), String>> = OnceLock::new();
    let result = INSTALL_RESULT.get_or_init(|| {
        let subscriber = tracing_subscriber::registry().with(lutum_trace::layer());
        tracing::subscriber::set_global_default(subscriber).map_err(|error| error.to_string())
    });
    result
        .as_ref()
        .map(|_| ())
        .map_err(|message| RunnerError::TraceSubscriber {
            message: message.clone(),
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

pub fn default_run_id() -> String {
    Utc::now().format("%Y%m%dT%H%M%SZ").to_string()
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use chrono::TimeZone as _;
    use lutum::{Lutum, MockLlmAdapter, SharedPoolBudgetManager, SharedPoolBudgetOptions};
    use nuillu_blackboard::{AttentionStreamEvent, BlackboardCommand, MemoryMetaPatch};
    use nuillu_module::ports::{
        NoopAttentionRepository, NoopFileSearchProvider, NoopMemoryStore, NoopUtteranceSink,
        SystemClock,
    };
    use nuillu_types::{MemoryIndex, ModuleInstanceId, ReplicaIndex};

    use super::*;

    struct FixedClock(chrono::DateTime<Utc>);

    impl Clock for FixedClock {
        fn now(&self) -> chrono::DateTime<Utc> {
            self.0
        }
    }

    fn test_backend_config() -> LlmBackendConfig {
        LlmBackendConfig {
            endpoint: "http://localhost:11434/v1".to_string(),
            token: "local".to_string(),
            model: "gpt-oss:20b".to_string(),
            reasoning_effort: None,
            use_responses_api: false,
        }
    }

    fn test_caps(blackboard: Blackboard) -> CapabilityProviders {
        let adapter = Arc::new(MockLlmAdapter::new());
        let budget = SharedPoolBudgetManager::new(SharedPoolBudgetOptions::default());
        let lutum = Lutum::new(adapter, budget);
        CapabilityProviders::new(
            blackboard,
            Arc::new(NoopAttentionRepository),
            Arc::new(NoopMemoryStore),
            Vec::new(),
            Arc::new(NoopFileSearchProvider),
            Arc::new(NoopUtteranceSink),
            Arc::new(SystemClock),
            LutumTiers {
                cheap: lutum.clone(),
                default: lutum.clone(),
                premium: lutum,
            },
        )
    }

    #[test]
    fn case_patterns_match_case_id_or_path_substrings() {
        let dir = tempfile::tempdir().unwrap();
        let case_dir = dir.path().join("eval-cases/modules/query-vector");
        std::fs::create_dir_all(&case_dir).unwrap();
        let first = case_dir.join("first-route.eure");
        let second = case_dir.join("second-memory.eure");
        std::fs::write(
            &first,
            r#"
id = "module-query-vector-first-route"
prompt = "First?"
"#,
        )
        .unwrap();
        std::fs::write(
            &second,
            r#"
id = "module-query-vector-special-memory"
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

    #[tokio::test]
    async fn max_llm_calls_requests_stop_after_limit_event() {
        let dir = tempfile::tempdir().unwrap();
        let reporter = LiveReporter::new("test-run", dir.path()).unwrap();
        let sink = RecordingRuntimeEventSink::new("test-case".to_string(), Some(3), reporter);
        let owner = ModuleInstanceId::new(builtin::query_vector(), ReplicaIndex::ZERO);

        sink.on_event(RuntimeEvent::LlmAccessed {
            sequence: 0,
            call: 0,
            owner: owner.clone(),
            tier: ModelTier::Default,
        })
        .await
        .unwrap();
        assert!(!sink.stop_requested());

        sink.on_event(RuntimeEvent::LlmAccessed {
            sequence: 1,
            call: 1,
            owner: owner.clone(),
            tier: ModelTier::Default,
        })
        .await
        .unwrap();
        assert!(!sink.stop_requested());

        sink.on_event(RuntimeEvent::LlmAccessed {
            sequence: 2,
            call: 2,
            owner: owner.clone(),
            tier: ModelTier::Default,
        })
        .await
        .unwrap();

        assert!(sink.stop_requested());
        sink.on_event(RuntimeEvent::LlmAccessed {
            sequence: 3,
            call: 3,
            owner,
            tier: ModelTier::Default,
        })
        .await
        .unwrap();
        assert_eq!(sink.snapshot().len(), 4);
        let jsonl = std::fs::read_to_string(dir.path().join("events.jsonl")).unwrap();
        assert!(jsonl.contains("\"kind\":\"runtime_event\""));
        assert!(jsonl.contains("\"kind\":\"stop_requested\""));
        assert_eq!(jsonl.matches("\"kind\":\"stop_requested\"").count(), 1);
    }

    #[tokio::test]
    async fn run_suite_records_case_runtime_failures_and_continues() {
        let dir = tempfile::tempdir().unwrap();
        let case_dir = dir.path().join("eval-cases/modules/query-vector");
        std::fs::create_dir_all(&case_dir).unwrap();
        for id in ["runtime-failure-one", "runtime-failure-two"] {
            std::fs::write(
                case_dir.join(format!("{id}.eure")),
                format!(
                    r#"
id = "{id}"
prompt = "Who are you?"

limits {{
  max-ticks = 1
  tick-ms = 1
  max-llm-calls = 1
}}
"#
                ),
            )
            .unwrap();
        }

        let output_root = dir.path().join("out");
        let config = RunnerConfig {
            cases_root: dir.path().join("eval-cases"),
            output_root: output_root.clone(),
            run_id: "runtime-failures".to_string(),
            judge_backend: test_backend_config(),
            cheap_backend: test_backend_config(),
            default_backend: test_backend_config(),
            premium_backend: test_backend_config(),
            model_dir: dir.path().join("missing-model"),
            fail_fast: false,
            case_patterns: Vec::new(),
        };

        let report = run_suite(&config).await.unwrap();

        assert_eq!(report.case_count, 2);
        assert_eq!(report.passed_cases, 0);
        assert_eq!(report.invalid_cases, 2);
        assert!(report.cases.iter().all(|case| {
            !case.passed
                && case.invalid
                && case.score == 0.0
                && case.report.runtime_failure.is_some()
                && case.report.checks.is_empty()
        }));

        let run_dir = output_root.join("runtime-failures");
        assert!(run_dir.join("suite-report.json").exists());
        for summary in &report.cases {
            let output_dir = run_dir.join(sanitize_id(&summary.id));
            assert!(output_dir.join("report.json").exists());
            assert!(output_dir.join("artifact.json").exists());
            assert!(output_dir.join("raw-trace.json").exists());
            let events: serde_json::Value =
                serde_json::from_slice(&std::fs::read(output_dir.join("events.json")).unwrap())
                    .unwrap();
            assert_eq!(events, serde_json::json!([]));
        }
    }

    #[tokio::test]
    async fn agent_observation_serializes_string_keyed_blackboard_maps() {
        let now = Utc.with_ymd_and_hms(2026, 5, 7, 0, 0, 0).unwrap();
        let mut allocation = ResourceAllocation::default();
        allocation.set(
            builtin::query_vector(),
            ModuleConfig {
                activation_ratio: ActivationRatio::ONE,
                guidance: "test guidance".into(),
                tier: ModelTier::Default,
            },
        );
        let blackboard = Blackboard::with_allocation(allocation.clone());
        let owner = ModuleInstanceId::new(builtin::query_vector(), ReplicaIndex::ZERO);

        blackboard
            .apply(BlackboardCommand::UpdateMemo {
                owner: owner.clone(),
                memo: "memo".to_string(),
                written_at: now,
            })
            .await;
        blackboard
            .apply(BlackboardCommand::AppendAttentionStream {
                stream: owner,
                event: AttentionStreamEvent {
                    at: now,
                    text: "attention".to_string(),
                },
            })
            .await;
        blackboard
            .apply(BlackboardCommand::SetReplicaCaps {
                caps: vec![(builtin::query_vector(), ReplicaCapRange::new(0, 1).unwrap())],
            })
            .await;
        blackboard
            .apply(BlackboardCommand::RecordAllocationProposal {
                controller: ModuleInstanceId::new(
                    builtin::attention_controller(),
                    ReplicaIndex::ZERO,
                ),
                proposal: allocation,
            })
            .await;
        blackboard
            .apply(BlackboardCommand::UpsertMemoryMetadata {
                index: MemoryIndex::new("mem-1"),
                rank_if_new: MemoryRank::Permanent,
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
            "memos": {
                "query-vector": [{
                    "replica": 0,
                    "memo": "memo",
                }],
            },
            "memo_logs": {
                "query-vector": [{
                    "replica": 0,
                    "index": 0,
                    "written_at": "2026-05-07T00:00:00+00:00",
                    "content": "memo",
                }],
            },
            "attention_streams": [{
                "stream": {
                    "module": "query-vector",
                    "replica": 0,
                },
                "entries": [{
                    "at": "2026-05-07T00:00:00Z",
                    "text": "attention",
                }],
            }],
            "allocation": {
                "query-vector": {
                    "activation_ratio": 1.0,
                    "guidance": "test guidance",
                    "tier": "Default",
                },
            },
            "allocation_proposals": {
                "attention-controller": {
                    "query-vector": {
                        "activation_ratio": 1.0,
                        "guidance": "test guidance",
                        "tier": "Default",
                    },
                },
            },
            "replica_caps": {
                "query-vector": {
                    "min": 0,
                    "max": 1,
                },
            },
            "memory_metadata": {
                "mem-1": {
                    "index": "mem-1",
                    "rank": "Permanent",
                    "decay_remaining_secs": 0,
                    "remember_tokens": 0,
                    "last_accessed": "2026-05-07T00:00:00Z",
                    "access_count": 0,
                    "query_history": [],
                },
            },
            "utterances": [],
        });
        assert_eq!(actual, expected);
    }

    #[tokio::test]
    async fn seed_attention_stream_stamps_attention_gate_replica_and_offsets_time() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("attention-seed.eure");
        std::fs::write(
            &path,
            r#"
id = "attention-seed"
prompt = "What am I attending to?"

@ attention-stream[] {
  text = "Older attended topic"
  seconds-ago = 30
}

@ attention-stream[] {
  text = "Current attended topic"
}
"#,
        )
        .unwrap();
        let case = crate::cases::parse_module_case_file(&path).unwrap();
        let blackboard = Blackboard::default();
        let now = Utc.with_ymd_and_hms(2026, 5, 7, 12, 0, 0).unwrap();

        seed_attention_stream(&blackboard, &FixedClock(now), &case.attention_stream).await;

        let stream_set = blackboard.read(|bb| bb.attention_stream_set()).await;
        let records = stream_set.streams();
        assert_eq!(records.len(), 1);
        let record = &records[0];
        assert_eq!(record.stream.module, builtin::attention_gate());
        assert_eq!(record.stream.replica, ReplicaIndex::ZERO);
        assert_eq!(record.entries.len(), 2);
        assert_eq!(record.entries[0].text, "Older attended topic");
        assert_eq!(record.entries[0].at, now - ChronoDuration::seconds(30));
        assert_eq!(record.entries[1].text, "Current attended topic");
        assert_eq!(record.entries[1].at, now);
    }

    #[tokio::test]
    async fn eval_registry_and_allocation_include_only_selected_modules() {
        let selected = [
            EvalModule::Sensory,
            EvalModule::AttentionController,
            EvalModule::Speak,
        ];
        let allocation = full_agent_allocation(
            &crate::cases::EvalLimits {
                tick_ms: 500,
                max_ticks: 1,
                max_llm_calls: None,
            },
            &selected,
        );
        assert!(allocation.get(&builtin::query_vector()).is_none());
        assert!(allocation.get(&builtin::speak_gate()).is_none());

        let blackboard = Blackboard::with_allocation(allocation);
        let caps = test_caps(blackboard.clone());
        let allocated = eval_registry(&selected).build(&caps).await.unwrap();
        assert_eq!(allocated.len(), selected.len());

        let (replica_caps, allocation_modules) = blackboard
            .read(|bb| {
                let mut replica_caps = bb
                    .replica_caps()
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
            vec!["attention-controller", "sensory", "speak"]
        );
        assert_eq!(
            allocation_modules,
            vec!["attention-controller", "sensory", "speak"]
        );
    }

    #[test]
    fn full_agent_allocation_bootstraps_controller_instead_of_every_module() {
        let selected = DEFAULT_FULL_AGENT_MODULES.to_vec();
        let allocation = full_agent_allocation(
            &crate::cases::EvalLimits {
                tick_ms: 500,
                max_ticks: 1,
                max_llm_calls: None,
            },
            &selected,
        );

        let sensory = allocation.for_module(&builtin::sensory());
        assert_eq!(sensory.activation_ratio, ActivationRatio::ONE);
        assert_eq!(sensory.tier, ModelTier::Cheap);

        let attention_gate = allocation.for_module(&builtin::attention_gate());
        assert_eq!(attention_gate.activation_ratio, ActivationRatio::ZERO);
        assert_eq!(attention_gate.tier, ModelTier::Cheap);

        let controller = allocation.for_module(&builtin::attention_controller());
        assert_eq!(controller.activation_ratio, ActivationRatio::ONE);
        assert_eq!(controller.tier, ModelTier::Premium);

        let speak_gate = allocation.for_module(&builtin::speak_gate());
        assert_eq!(speak_gate.activation_ratio, ActivationRatio::ONE);
        assert_eq!(speak_gate.tier, ModelTier::Premium);

        let speak = allocation.for_module(&builtin::speak());
        assert_eq!(speak.activation_ratio, ActivationRatio::ONE);
        assert_eq!(speak.tier, ModelTier::Premium);

        for module in [
            builtin::attention_schema(),
            builtin::self_model(),
            builtin::query_vector(),
            builtin::memory(),
            builtin::memory_compaction(),
            builtin::predict(),
            builtin::surprise(),
        ] {
            assert_eq!(
                allocation.for_module(&module).activation_ratio,
                ActivationRatio::ZERO
            );
        }
        assert!(allocation.get(&builtin::query_agentic()).is_none());
    }
}
