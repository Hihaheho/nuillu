use std::{
    collections::{BTreeMap, HashSet},
    fs,
    num::NonZeroUsize,
    path::PathBuf,
    time::Instant,
};

use anyhow::Context as _;
use clap::{Parser, ValueEnum};
use futures::{StreamExt, stream};
use lutum::{
    GenerationParams, InputMessageRole, ModelInput, ModelInputItem, Temperature,
    TextStepOutcomeWithTools,
};
use nuillu_eval::{install_lutum_trace_subscriber, parse_model_set_file, resolve_llm_backends};
use schemars::{JsonSchema, Schema, SchemaGenerator};
use serde::{Deserialize, Serialize};

const DEFAULT_OUTPUT_DIR: &str = ".tmp/allocation-schema-probe";

const REGISTERED_MODULES: &[&str] = &[
    "attention-schema",
    "cognition-gate",
    "memory",
    "policy",
    "policy-compaction",
    "predict",
    "query-memory",
    "reward",
    "self-model",
    "speak",
    "surprise",
];

const SYSTEM_PROMPT: &str = r#"You are the allocation module.
Use the current memo batch, cognition context, current allocation, and registered module catalog to
choose which registered modules deserve extra activation now.

Call reprioritize_modules exactly once. Rank modules in descending priority order. Do not invent
module ids. Keep hints concise and specific to the current memo batch."#;

const DEVELOPER_CONTEXT: &str = r#"Allocation context for assigning the next activation priorities:

Current allocation:
- sensory: active, cheap
- cognition-gate: active, default
- allocation: active, default
- speak: inactive, premium
- attention-schema: inactive, default
- query-memory: inactive, cheap

Registered allocation targets:
- attention-schema: maintain the agent's current attention model and append attention experience.
- cognition-gate: admit durable memo evidence into the cognition log.
- query-memory: retrieve relevant memories when current cognition needs evidence.
- speak: produce outward utterances after cognition-log evidence is ready.
- predict: predict near-future states from cognition-log targets.
- surprise: detect violations of expected current state.
- self-model: answer self-model questions from cognition and memory context.
- memory: preserve durable cognition-log facts as memory.
- policy: produce policy advice for the current context.
- policy-compaction: compact redundant policy records.
- reward: evaluate policy outcomes.

Current interoception: mode=Wake; wake_arousal=0.72; affect_arousal=0.45; valence=0.10; emotion=alert

Current memo batch:
These are the new durable notes that triggered this allocation turn; treat them as current evidence
to prioritize now:
- sensory[0]: Audition from Peer: "Hi. Can you hear me?"
- cognition-gate[0]: Peer directly greeted Nui and appears to expect an outward acknowledgement.

Instruction: choose the best current allocation posture for the mind."#;

#[derive(Debug, Parser)]
#[command(
    name = "allocation-schema-probe",
    about = "Compare allocation reprioritize_modules schema variants against a live LLM backend"
)]
struct Args {
    /// Model set Eure file with per-role backend config.
    #[arg(long)]
    model_set: PathBuf,

    /// Trials to run per selected schema candidate.
    #[arg(long, default_value = "100")]
    trials: NonZeroUsize,

    /// Output directory for JSON and Markdown reports.
    #[arg(long, default_value = DEFAULT_OUTPUT_DIR)]
    output: PathBuf,

    /// Schema candidates to run. Comma-separated values are accepted.
    #[arg(long = "schema", value_enum, value_delimiter = ',')]
    schemas: Vec<SchemaKind>,

    /// Generation temperature used for each probe turn.
    #[arg(long, default_value = "0.2")]
    temperature: f32,

    /// Max output tokens for each probe turn.
    #[arg(long, default_value = "1024")]
    max_output_tokens: u32,

    /// Base seed. Trial index is added to this value for every schema candidate.
    #[arg(long, default_value = "1")]
    seed_base: u64,
}

#[derive(
    Clone, Copy, Debug, Eq, PartialEq, Hash, Ord, PartialOrd, Serialize, Deserialize, ValueEnum,
)]
#[serde(rename_all = "kebab-case")]
enum SchemaKind {
    Current,
    #[value(name = "short-keys")]
    ShortKeys,
    #[value(name = "split-arrays")]
    SplitArrays,
    #[value(name = "ids-map")]
    IdsMap,
    #[value(name = "primary-rest")]
    PrimaryRest,
}

impl SchemaKind {
    fn all() -> &'static [Self] {
        &[
            Self::Current,
            Self::ShortKeys,
            Self::SplitArrays,
            Self::IdsMap,
            Self::PrimaryRest,
        ]
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Current => "current",
            Self::ShortKeys => "short-keys",
            Self::SplitArrays => "split-arrays",
            Self::IdsMap => "ids-map",
            Self::PrimaryRest => "primary-rest",
        }
    }

    fn production_distance(self) -> u32 {
        match self {
            Self::IdsMap => 0,
            Self::SplitArrays => 1,
            Self::Current => 2,
            Self::ShortKeys => 3,
            Self::PrimaryRest => 4,
        }
    }
}

impl std::fmt::Display for SchemaKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

tokio::task_local! {
    static MODULE_TARGET_ID_SCHEMA: Schema;
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
struct ModuleTargetId(String);

impl From<ModuleTargetId> for String {
    fn from(value: ModuleTargetId) -> Self {
        value.0
    }
}

impl JsonSchema for ModuleTargetId {
    fn inline_schema() -> bool {
        true
    }

    fn schema_name() -> std::borrow::Cow<'static, str> {
        "ModuleTargetId".into()
    }

    fn schema_id() -> std::borrow::Cow<'static, str> {
        "nuillu_eval::allocation_schema_probe::ModuleTargetId.dynamic".into()
    }

    fn json_schema(_generator: &mut SchemaGenerator) -> Schema {
        MODULE_TARGET_ID_SCHEMA
            .try_with(Clone::clone)
            .unwrap_or_else(|_| module_target_id_schema())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
struct ReprioritizeOutput {
    accepted: bool,
}

mod current_schema {
    use super::*;

    /// Raise activation using priority entries with explicit module_id and hint fields.
    #[lutum::tool_input(name = "reprioritize_modules", output = ReprioritizeOutput)]
    #[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
    pub(super) struct Args {
        pub(super) memo: String,
        pub(super) priority: Vec<PriorityEntry>,
    }

    #[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
    pub(super) struct PriorityEntry {
        pub(super) module_id: ModuleTargetId,
        pub(super) hint: String,
    }

    #[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema, lutum::Toolset)]
    pub(super) enum Tools {
        ReprioritizeModules(Args),
    }
}

mod short_keys_schema {
    use super::*;

    /// Raise activation using short priority entries with id and reason fields.
    #[lutum::tool_input(name = "reprioritize_modules", output = ReprioritizeOutput)]
    #[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
    pub(super) struct Args {
        pub(super) memo: String,
        pub(super) priority: Vec<PriorityEntry>,
    }

    #[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
    pub(super) struct PriorityEntry {
        pub(super) id: ModuleTargetId,
        pub(super) reason: String,
    }

    #[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema, lutum::Toolset)]
    pub(super) enum Tools {
        ReprioritizeModules(Args),
    }
}

mod split_arrays_schema {
    use super::*;

    /// Raise activation using parallel arrays for ranked module ids and hints.
    #[lutum::tool_input(name = "reprioritize_modules", output = ReprioritizeOutput)]
    #[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
    pub(super) struct Args {
        pub(super) memo: String,
        pub(super) priority_module_ids: Vec<ModuleTargetId>,
        pub(super) priority_hints: Vec<String>,
    }

    #[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema, lutum::Toolset)]
    pub(super) enum Tools {
        ReprioritizeModules(Args),
    }
}

mod ids_map_schema {
    use super::*;

    /// Raise activation using ranked module ids plus a module-id keyed hint map.
    #[lutum::tool_input(name = "reprioritize_modules", output = ReprioritizeOutput)]
    #[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
    pub(super) struct Args {
        pub(super) memo: String,
        pub(super) priority_module_ids: Vec<ModuleTargetId>,
        #[serde(default)]
        pub(super) hints_by_module: BTreeMap<String, String>,
    }

    #[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema, lutum::Toolset)]
    pub(super) enum Tools {
        ReprioritizeModules(Args),
    }
}

mod primary_rest_schema {
    use super::*;

    /// Raise activation using one primary module id and ranked secondary module ids.
    #[lutum::tool_input(name = "reprioritize_modules", output = ReprioritizeOutput)]
    #[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
    pub(super) struct Args {
        pub(super) memo: String,
        pub(super) primary_module_id: ModuleTargetId,
        pub(super) secondary_module_ids: Vec<ModuleTargetId>,
        pub(super) hints_by_module: BTreeMap<String, String>,
    }

    #[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema, lutum::Toolset)]
    pub(super) enum Tools {
        ReprioritizeModules(Args),
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
struct CandidateDecision {
    memo: String,
    priority_module_ids: Vec<String>,
    hints: BTreeMap<String, String>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
struct ValidationReport {
    success: bool,
    failure: Option<String>,
    priority_module_ids: Vec<String>,
    invalid_module_ids: Vec<String>,
}

#[derive(Clone, Debug, Serialize)]
struct TrialReport {
    schema: SchemaKind,
    trial: usize,
    success: bool,
    failure: Option<String>,
    transport_error: bool,
    priority_module_count: usize,
    priority_module_ids: Vec<String>,
    invalid_module_ids: Vec<String>,
    tool_call_count: usize,
    tool_issue_count: usize,
    latency_ms: u128,
    usage: Option<lutum::Usage>,
}

#[derive(Clone, Debug, Serialize)]
struct CandidateSummary {
    schema: SchemaKind,
    attempted_trials: usize,
    evaluable_trials: usize,
    transport_errors: usize,
    successes: usize,
    schema_failures: usize,
    success_rate: f64,
    invalid_module_ids: usize,
    tool_issues: usize,
    production_distance: u32,
    usage: UsageSummary,
    success_priority_module_count: NumericStats,
    success_module_id_counts: BTreeMap<String, usize>,
}

#[derive(Clone, Debug, Default, Serialize)]
struct UsageSummary {
    sample_count: usize,
    input_tokens: NumericStats,
    output_tokens: NumericStats,
    total_tokens: NumericStats,
    cache_creation_tokens: NumericStats,
    cache_read_tokens: NumericStats,
    cost_micros_usd: NumericStats,
}

#[derive(Clone, Debug, Default, Serialize)]
struct NumericStats {
    sample_count: usize,
    min: Option<u64>,
    p05: Option<u64>,
    p50: Option<u64>,
    p95: Option<u64>,
    max: Option<u64>,
    mean: Option<f64>,
}

#[derive(Debug, Serialize)]
struct ProbeReport {
    generated_at: String,
    model_set: String,
    model_key: String,
    model: String,
    endpoint: String,
    max_concurrent_llm_calls: usize,
    trials_per_schema: usize,
    schemas: Vec<CandidateSummary>,
    winner: Option<SchemaKind>,
    trials: Vec<TrialReport>,
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> anyhow::Result<()> {
    install_lutum_trace_subscriber()?;
    let args = Args::parse();
    let selected = selected_schemas(&args);
    let temperature = Temperature::try_from(args.temperature)
        .with_context(|| format!("invalid --temperature {}", args.temperature))?;

    let model_set = parse_model_set_file(&args.model_set)?;
    let backends = resolve_llm_backends(&model_set)?;
    let backend = backends.cheap;
    let lutum = nuillu_server::build_lutum(&backend, None).context("build cheap-tier lutum")?;
    let concurrency_limit = backend
        .max_concurrent_llm_calls
        .map_or(1, NonZeroUsize::get);
    let generation = GenerationParams {
        temperature: Some(temperature),
        max_output_tokens: Some(args.max_output_tokens),
        seed: None,
    };

    let module_schema = module_target_id_schema();
    let mut trials = Vec::new();
    let trial_specs = selected
        .into_iter()
        .flat_map(|schema| (0..args.trials.get()).map(move |trial| (schema, trial)))
        .collect::<Vec<_>>();
    let trial_futures = trial_specs.into_iter().map(|(schema, trial)| {
        let module_schema = module_schema.clone();
        let lutum = &lutum;
        let mut generation = generation.clone();
        generation.seed = Some(args.seed_base.saturating_add(trial as u64));
        async move {
            MODULE_TARGET_ID_SCHEMA
                .scope(module_schema, run_trial(schema, trial, generation, lutum))
                .await
        }
    });
    let mut trial_stream = stream::iter(trial_futures).buffer_unordered(concurrency_limit);
    while let Some(report) = trial_stream.next().await {
        println!(
            "schema={} trial={} success={} failure={}",
            report.schema,
            report.trial + 1,
            report.success,
            report.failure.as_deref().unwrap_or("-")
        );
        trials.push(report);
    }

    let summaries = summarize_trials(&trials);
    let winner = pick_winner(&summaries);
    let report = ProbeReport {
        generated_at: chrono::Utc::now().to_rfc3339(),
        model_set: args.model_set.display().to_string(),
        model_key: backend.model_key,
        model: backend.model,
        endpoint: backend.endpoint,
        max_concurrent_llm_calls: concurrency_limit,
        trials_per_schema: args.trials.get(),
        schemas: summaries,
        winner,
        trials,
    };

    fs::create_dir_all(&args.output)
        .with_context(|| format!("create output directory {}", args.output.display()))?;
    let json_path = args.output.join("probe-report.json");
    let md_path = args.output.join("probe-report.md");
    fs::write(&json_path, serde_json::to_string_pretty(&report)?)
        .with_context(|| format!("write {}", json_path.display()))?;
    fs::write(&md_path, render_markdown_report(&report))
        .with_context(|| format!("write {}", md_path.display()))?;

    println!(
        "winner={} json={} markdown={}",
        report
            .winner
            .map(|winner| winner.as_str())
            .unwrap_or("(none)"),
        json_path.display(),
        md_path.display()
    );
    Ok(())
}

fn selected_schemas(args: &Args) -> Vec<SchemaKind> {
    if args.schemas.is_empty() {
        SchemaKind::all().to_vec()
    } else {
        let mut selected = Vec::new();
        for schema in &args.schemas {
            if !selected.contains(schema) {
                selected.push(*schema);
            }
        }
        selected
    }
}

async fn run_trial(
    schema: SchemaKind,
    trial: usize,
    generation: GenerationParams,
    lutum: &lutum::Lutum,
) -> TrialReport {
    match schema {
        SchemaKind::Current => run_current_trial(trial, generation, lutum).await,
        SchemaKind::ShortKeys => run_short_keys_trial(trial, generation, lutum).await,
        SchemaKind::SplitArrays => run_split_arrays_trial(trial, generation, lutum).await,
        SchemaKind::IdsMap => run_ids_map_trial(trial, generation, lutum).await,
        SchemaKind::PrimaryRest => run_primary_rest_trial(trial, generation, lutum).await,
    }
}

async fn run_current_trial(
    trial: usize,
    generation: GenerationParams,
    lutum: &lutum::Lutum,
) -> TrialReport {
    let started = Instant::now();
    let outcome = lutum
        .text_turn(probe_input())
        .tools::<current_schema::Tools>()
        .available_tools([current_schema::ToolsSelector::ReprioritizeModules])
        .require_any_tool()
        .generation_config(generation)
        .collect()
        .await;
    build_trial_report(
        SchemaKind::Current,
        trial,
        started,
        outcome,
        |call| match call {
            current_schema::ToolsCall::ReprioritizeModules(call) => {
                Some(serde_json::to_value(call.input.clone()))
            }
        },
    )
}

async fn run_short_keys_trial(
    trial: usize,
    generation: GenerationParams,
    lutum: &lutum::Lutum,
) -> TrialReport {
    let started = Instant::now();
    let outcome = lutum
        .text_turn(probe_input())
        .tools::<short_keys_schema::Tools>()
        .available_tools([short_keys_schema::ToolsSelector::ReprioritizeModules])
        .require_any_tool()
        .generation_config(generation)
        .collect()
        .await;
    build_trial_report(
        SchemaKind::ShortKeys,
        trial,
        started,
        outcome,
        |call| match call {
            short_keys_schema::ToolsCall::ReprioritizeModules(call) => {
                Some(serde_json::to_value(call.input.clone()))
            }
        },
    )
}

async fn run_split_arrays_trial(
    trial: usize,
    generation: GenerationParams,
    lutum: &lutum::Lutum,
) -> TrialReport {
    let started = Instant::now();
    let outcome = lutum
        .text_turn(probe_input())
        .tools::<split_arrays_schema::Tools>()
        .available_tools([split_arrays_schema::ToolsSelector::ReprioritizeModules])
        .require_any_tool()
        .generation_config(generation)
        .collect()
        .await;
    build_trial_report(
        SchemaKind::SplitArrays,
        trial,
        started,
        outcome,
        |call| match call {
            split_arrays_schema::ToolsCall::ReprioritizeModules(call) => {
                Some(serde_json::to_value(call.input.clone()))
            }
        },
    )
}

async fn run_ids_map_trial(
    trial: usize,
    generation: GenerationParams,
    lutum: &lutum::Lutum,
) -> TrialReport {
    let started = Instant::now();
    let outcome = lutum
        .text_turn(probe_input())
        .tools::<ids_map_schema::Tools>()
        .available_tools([ids_map_schema::ToolsSelector::ReprioritizeModules])
        .require_any_tool()
        .generation_config(generation)
        .collect()
        .await;
    build_trial_report(
        SchemaKind::IdsMap,
        trial,
        started,
        outcome,
        |call| match call {
            ids_map_schema::ToolsCall::ReprioritizeModules(call) => {
                Some(serde_json::to_value(call.input.clone()))
            }
        },
    )
}

async fn run_primary_rest_trial(
    trial: usize,
    generation: GenerationParams,
    lutum: &lutum::Lutum,
) -> TrialReport {
    let started = Instant::now();
    let outcome = lutum
        .text_turn(probe_input())
        .tools::<primary_rest_schema::Tools>()
        .available_tools([primary_rest_schema::ToolsSelector::ReprioritizeModules])
        .require_any_tool()
        .generation_config(generation)
        .collect()
        .await;
    build_trial_report(
        SchemaKind::PrimaryRest,
        trial,
        started,
        outcome,
        |call| match call {
            primary_rest_schema::ToolsCall::ReprioritizeModules(call) => {
                Some(serde_json::to_value(call.input.clone()))
            }
        },
    )
}

fn build_trial_report<T, F>(
    schema: SchemaKind,
    trial: usize,
    started: Instant,
    outcome: Result<
        TextStepOutcomeWithTools<T>,
        lutum::CollectError<
            std::convert::Infallible,
            lutum::TextTurnReductionError,
            lutum::TextTurnStateWithTools<T>,
        >,
    >,
    extract_args: F,
) -> TrialReport
where
    T: lutum::Toolset,
    F: Fn(&T::ToolCall) -> Option<Result<serde_json::Value, serde_json::Error>>,
{
    let latency_ms = started.elapsed().as_millis();
    match outcome {
        Ok(TextStepOutcomeWithTools::NeedsTools(round)) => {
            let mut validation = None;
            for call in &round.tool_calls {
                if let Some(args) = extract_args(call) {
                    validation = Some(match args {
                        Ok(args) => validate_candidate_json(schema, args),
                        Err(error) => ValidationReport {
                            success: false,
                            failure: Some(format!("serialize_parsed_args:{error}")),
                            priority_module_ids: Vec::new(),
                            invalid_module_ids: Vec::new(),
                        },
                    });
                    break;
                }
            }
            let validation = validation.unwrap_or_else(|| {
                let failure = if round.recoverable_tool_call_issues().is_empty() {
                    "no_reprioritize_tool".to_owned()
                } else {
                    let reasons = round
                        .recoverable_tool_call_issues()
                        .iter()
                        .map(|issue| format!("{:?}", issue.reason))
                        .collect::<Vec<_>>()
                        .join(",");
                    format!("tool_call_issue:{reasons}")
                };
                ValidationReport {
                    success: false,
                    failure: Some(failure),
                    priority_module_ids: Vec::new(),
                    invalid_module_ids: Vec::new(),
                }
            });
            let priority_module_count = validation.priority_module_ids.len();
            TrialReport {
                schema,
                trial,
                success: validation.success,
                failure: validation.failure,
                transport_error: false,
                priority_module_count,
                priority_module_ids: validation.priority_module_ids,
                invalid_module_ids: validation.invalid_module_ids,
                tool_call_count: round.tool_calls.len(),
                tool_issue_count: round.recoverable_tool_call_issues().len(),
                latency_ms,
                usage: Some(round.usage),
            }
        }
        Ok(TextStepOutcomeWithTools::Finished(result)) => TrialReport {
            schema,
            trial,
            success: false,
            failure: Some(format!("finished_without_tool:{:?}", result.finish_reason)),
            transport_error: false,
            priority_module_count: 0,
            priority_module_ids: Vec::new(),
            invalid_module_ids: Vec::new(),
            tool_call_count: 0,
            tool_issue_count: 0,
            latency_ms,
            usage: Some(result.usage),
        },
        Ok(TextStepOutcomeWithTools::FinishedNoOutput(result)) => TrialReport {
            schema,
            trial,
            success: false,
            failure: Some(format!(
                "finished_without_output:{:?}",
                result.finish_reason
            )),
            transport_error: false,
            priority_module_count: 0,
            priority_module_ids: Vec::new(),
            invalid_module_ids: Vec::new(),
            tool_call_count: 0,
            tool_issue_count: 0,
            latency_ms,
            usage: Some(result.usage),
        },
        Err(error) => {
            let failure = format!("execution_error:{error}");
            TrialReport {
                schema,
                trial,
                success: false,
                transport_error: is_transport_failure(&failure),
                failure: Some(failure),
                priority_module_count: 0,
                priority_module_ids: Vec::new(),
                invalid_module_ids: Vec::new(),
                tool_call_count: 0,
                tool_issue_count: 0,
                latency_ms,
                usage: None,
            }
        }
    }
}

fn is_transport_failure(failure: &str) -> bool {
    failure.contains("kind=Transport") || failure.contains("error sending request")
}

fn validate_candidate_json(schema: SchemaKind, value: serde_json::Value) -> ValidationReport {
    match parse_candidate_json(schema, value) {
        Ok(decision) => validate_candidate_decision(decision),
        Err(error) => ValidationReport {
            success: false,
            failure: Some(format!("parse_error:{error}")),
            priority_module_ids: Vec::new(),
            invalid_module_ids: Vec::new(),
        },
    }
}

fn parse_candidate_json(
    schema: SchemaKind,
    value: serde_json::Value,
) -> Result<CandidateDecision, serde_json::Error> {
    match schema {
        SchemaKind::Current => {
            let args = serde_json::from_value::<current_schema::Args>(value)?;
            Ok(CandidateDecision {
                memo: args.memo,
                priority_module_ids: args
                    .priority
                    .into_iter()
                    .map(|entry| entry.module_id.into())
                    .collect(),
                hints: BTreeMap::new(),
            })
        }
        SchemaKind::ShortKeys => {
            let args = serde_json::from_value::<short_keys_schema::Args>(value)?;
            Ok(CandidateDecision {
                memo: args.memo,
                priority_module_ids: args
                    .priority
                    .into_iter()
                    .map(|entry| entry.id.into())
                    .collect(),
                hints: BTreeMap::new(),
            })
        }
        SchemaKind::SplitArrays => {
            let args = serde_json::from_value::<split_arrays_schema::Args>(value)?;
            Ok(CandidateDecision {
                memo: args.memo,
                priority_module_ids: args
                    .priority_module_ids
                    .into_iter()
                    .map(Into::into)
                    .collect(),
                hints: BTreeMap::new(),
            })
        }
        SchemaKind::IdsMap => {
            let args = serde_json::from_value::<ids_map_schema::Args>(value)?;
            Ok(CandidateDecision {
                memo: args.memo,
                priority_module_ids: args
                    .priority_module_ids
                    .into_iter()
                    .map(Into::into)
                    .collect(),
                hints: args.hints_by_module,
            })
        }
        SchemaKind::PrimaryRest => {
            let args = serde_json::from_value::<primary_rest_schema::Args>(value)?;
            let mut priority_module_ids = vec![args.primary_module_id.into()];
            priority_module_ids.extend(args.secondary_module_ids.into_iter().map(String::from));
            Ok(CandidateDecision {
                memo: args.memo,
                priority_module_ids,
                hints: args.hints_by_module,
            })
        }
    }
}

fn validate_candidate_decision(decision: CandidateDecision) -> ValidationReport {
    if decision.priority_module_ids.is_empty() {
        return ValidationReport {
            success: false,
            failure: Some("empty_priority".to_owned()),
            priority_module_ids: Vec::new(),
            invalid_module_ids: Vec::new(),
        };
    }
    let registered = registered_modules();
    let invalid_module_ids = decision
        .priority_module_ids
        .iter()
        .filter(|id| !registered.contains(id.as_str()))
        .cloned()
        .collect::<Vec<_>>();
    let success = invalid_module_ids.is_empty();
    ValidationReport {
        success,
        failure: (!success).then(|| "unknown_module_id".to_owned()),
        priority_module_ids: decision.priority_module_ids,
        invalid_module_ids,
    }
}

fn registered_modules() -> HashSet<&'static str> {
    REGISTERED_MODULES.iter().copied().collect()
}

fn module_target_id_schema() -> Schema {
    Schema::try_from(serde_json::json!({
        "type": "string",
        "enum": REGISTERED_MODULES,
    }))
    .expect("module target id schema must be valid")
}

fn probe_input() -> ModelInput {
    ModelInput::from_items(vec![
        ModelInputItem::text(InputMessageRole::System, SYSTEM_PROMPT),
        ModelInputItem::text(InputMessageRole::Developer, DEVELOPER_CONTEXT),
    ])
}

fn summarize_trials(trials: &[TrialReport]) -> Vec<CandidateSummary> {
    SchemaKind::all()
        .iter()
        .filter_map(|schema| {
            let schema_trials = trials
                .iter()
                .filter(|trial| trial.schema == *schema)
                .collect::<Vec<_>>();
            if schema_trials.is_empty() {
                return None;
            }
            let evaluable_trials = schema_trials
                .iter()
                .filter(|trial| !trial.transport_error)
                .copied()
                .collect::<Vec<_>>();
            let successes = evaluable_trials
                .iter()
                .filter(|trial| trial.success)
                .count();
            let invalid_module_ids = evaluable_trials
                .iter()
                .map(|trial| trial.invalid_module_ids.len())
                .sum();
            let tool_issues = evaluable_trials
                .iter()
                .map(|trial| trial.tool_issue_count)
                .sum();
            let transport_errors = schema_trials
                .iter()
                .filter(|trial| trial.transport_error)
                .count();
            let evaluable_count = evaluable_trials.len();
            let success_priority_module_count = numeric_stats(
                evaluable_trials
                    .iter()
                    .filter(|trial| trial.success)
                    .map(|trial| trial.priority_module_count as u64),
            );
            let success_module_id_counts = module_id_counts(
                evaluable_trials
                    .iter()
                    .filter(|trial| trial.success)
                    .flat_map(|trial| trial.priority_module_ids.iter().cloned()),
            );
            Some(CandidateSummary {
                schema: *schema,
                attempted_trials: schema_trials.len(),
                evaluable_trials: evaluable_count,
                transport_errors,
                successes,
                schema_failures: evaluable_count.saturating_sub(successes),
                success_rate: if evaluable_count == 0 {
                    0.0
                } else {
                    successes as f64 / evaluable_count as f64
                },
                invalid_module_ids,
                tool_issues,
                production_distance: schema.production_distance(),
                usage: usage_summary(
                    evaluable_trials
                        .iter()
                        .filter_map(|trial| trial.usage.as_ref()),
                ),
                success_priority_module_count,
                success_module_id_counts,
            })
        })
        .collect()
}

fn usage_summary<'a>(usages: impl IntoIterator<Item = &'a lutum::Usage>) -> UsageSummary {
    let usages = usages.into_iter().collect::<Vec<_>>();
    UsageSummary {
        sample_count: usages.len(),
        input_tokens: numeric_stats(usages.iter().map(|usage| usage.input_tokens)),
        output_tokens: numeric_stats(usages.iter().map(|usage| usage.output_tokens)),
        total_tokens: numeric_stats(usages.iter().map(|usage| usage.total_tokens)),
        cache_creation_tokens: numeric_stats(
            usages.iter().map(|usage| usage.cache_creation_tokens),
        ),
        cache_read_tokens: numeric_stats(usages.iter().map(|usage| usage.cache_read_tokens)),
        cost_micros_usd: numeric_stats(usages.iter().map(|usage| usage.cost_micros_usd)),
    }
}

fn numeric_stats(values: impl IntoIterator<Item = u64>) -> NumericStats {
    let mut values = values.into_iter().collect::<Vec<_>>();
    values.sort_unstable();
    if values.is_empty() {
        return NumericStats::default();
    }
    let sum = values.iter().copied().sum::<u64>();
    NumericStats {
        sample_count: values.len(),
        min: values.first().copied(),
        p05: Some(percentile(&values, 0.05)),
        p50: Some(percentile(&values, 0.50)),
        p95: Some(percentile(&values, 0.95)),
        max: values.last().copied(),
        mean: Some(sum as f64 / values.len() as f64),
    }
}

fn percentile(sorted_values: &[u64], quantile: f64) -> u64 {
    debug_assert!(!sorted_values.is_empty());
    let max_index = sorted_values.len() - 1;
    let index = ((max_index as f64) * quantile).round() as usize;
    sorted_values[index.min(max_index)]
}

fn module_id_counts(ids: impl IntoIterator<Item = String>) -> BTreeMap<String, usize> {
    let mut counts = BTreeMap::new();
    for id in ids {
        *counts.entry(id).or_default() += 1;
    }
    counts
}

fn pick_winner(summaries: &[CandidateSummary]) -> Option<SchemaKind> {
    summaries
        .iter()
        .max_by(|left, right| {
            compare_success_rate(left, right)
                .then_with(|| right.invalid_module_ids.cmp(&left.invalid_module_ids))
                .then_with(|| right.production_distance.cmp(&left.production_distance))
        })
        .map(|summary| summary.schema)
}

fn compare_success_rate(left: &CandidateSummary, right: &CandidateSummary) -> std::cmp::Ordering {
    match (left.evaluable_trials, right.evaluable_trials) {
        (0, 0) => std::cmp::Ordering::Equal,
        (0, _) => std::cmp::Ordering::Less,
        (_, 0) => std::cmp::Ordering::Greater,
        _ => (left.successes * right.evaluable_trials)
            .cmp(&(right.successes * left.evaluable_trials)),
    }
}

fn render_markdown_report(report: &ProbeReport) -> String {
    let mut out = String::new();
    out.push_str("# Allocation Schema Probe\n\n");
    out.push_str(&format!(
        "- generated_at: {}\n- model_set: {}\n- model: {} ({})\n- endpoint: {}\n- max_concurrent_llm_calls: {}\n- trials_per_schema: {}\n- winner: {}\n\n",
        report.generated_at,
        report.model_set,
        report.model,
        report.model_key,
        report.endpoint,
        report.max_concurrent_llm_calls,
        report.trials_per_schema,
        report
            .winner
            .map(|winner| winner.as_str())
            .unwrap_or("(none)")
    ));
    out.push_str(
        "| schema | successes | evaluable | attempted | transport_errors | success_rate | invalid_module_ids | tool_issues | production_distance |\n",
    );
    out.push_str("|---|---:|---:|---:|---:|---:|---:|---:|---:|\n");
    for summary in &report.schemas {
        out.push_str(&format!(
            "| {} | {} | {} | {} | {} | {:.3} | {} | {} | {} |\n",
            summary.schema,
            summary.successes,
            summary.evaluable_trials,
            summary.attempted_trials,
            summary.transport_errors,
            summary.success_rate,
            summary.invalid_module_ids,
            summary.tool_issues,
            summary.production_distance
        ));
    }
    out.push_str("\n## Token Usage\n\n");
    out.push_str(
        "| schema | samples | input avg | input p50 | input p95 | output avg | output p50 | output p95 | total avg | total p50 | total p95 |\n",
    );
    out.push_str("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n");
    for summary in &report.schemas {
        out.push_str(&format!(
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |\n",
            summary.schema,
            summary.usage.sample_count,
            format_mean(summary.usage.input_tokens.mean),
            format_u64(summary.usage.input_tokens.p50),
            format_u64(summary.usage.input_tokens.p95),
            format_mean(summary.usage.output_tokens.mean),
            format_u64(summary.usage.output_tokens.p50),
            format_u64(summary.usage.output_tokens.p95),
            format_mean(summary.usage.total_tokens.mean),
            format_u64(summary.usage.total_tokens.p50),
            format_u64(summary.usage.total_tokens.p95),
        ));
    }
    out.push_str("\n## Priority Module Counts\n\n");
    out.push_str(
        "| schema | success samples | min | p05 | p50 | p95 | max | avg | module id counts |\n",
    );
    out.push_str("|---|---:|---:|---:|---:|---:|---:|---:|---|\n");
    for summary in &report.schemas {
        let stats = &summary.success_priority_module_count;
        out.push_str(&format!(
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} |\n",
            summary.schema,
            stats.sample_count,
            format_u64(stats.min),
            format_u64(stats.p05),
            format_u64(stats.p50),
            format_u64(stats.p95),
            format_u64(stats.max),
            format_mean(stats.mean),
            format_module_counts(&summary.success_module_id_counts),
        ));
    }
    out
}

fn format_u64(value: Option<u64>) -> String {
    value.map_or_else(|| "-".to_owned(), |value| value.to_string())
}

fn format_mean(value: Option<f64>) -> String {
    value.map_or_else(|| "-".to_owned(), |value| format!("{value:.1}"))
}

fn format_module_counts(counts: &BTreeMap<String, usize>) -> String {
    if counts.is_empty() {
        return "-".to_owned();
    }
    let mut counts = counts.iter().collect::<Vec<_>>();
    counts.sort_by(|(left_id, left_count), (right_id, right_count)| {
        right_count
            .cmp(left_count)
            .then_with(|| left_id.cmp(right_id))
    });
    counts
        .into_iter()
        .map(|(id, count)| format!("{id}={count}"))
        .collect::<Vec<_>>()
        .join(", ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validator_accepts_current_valid_args() {
        let report = validate_candidate_json(
            SchemaKind::Current,
            serde_json::json!({
                "memo": "Peer greeted Nui; activate outward path.",
                "priority": [
                    { "module_id": "speak", "hint": "acknowledge Peer" },
                    { "module_id": "cognition-gate", "hint": "admit greeting" }
                ]
            }),
        );

        assert_eq!(
            report,
            ValidationReport {
                success: true,
                failure: None,
                priority_module_ids: vec!["speak".to_owned(), "cognition-gate".to_owned()],
                invalid_module_ids: Vec::new(),
            }
        );
    }

    #[test]
    fn validator_rejects_malformed_current_object() {
        let report = validate_candidate_json(
            SchemaKind::Current,
            serde_json::json!({
                "memo": "bad",
                "priority": [
                    { "\"speak\"": true, "\"reason\"": true }
                ]
            }),
        );

        assert!(!report.success);
        assert!(report.failure.unwrap().starts_with("parse_error:"));
    }

    #[test]
    fn validator_rejects_empty_priority() {
        let report = validate_candidate_json(
            SchemaKind::SplitArrays,
            serde_json::json!({
                "memo": "nothing",
                "priority_module_ids": [],
                "priority_hints": []
            }),
        );

        assert_eq!(
            report,
            ValidationReport {
                success: false,
                failure: Some("empty_priority".to_owned()),
                priority_module_ids: Vec::new(),
                invalid_module_ids: Vec::new(),
            }
        );
    }

    #[test]
    fn validator_rejects_unknown_module_id() {
        let report = validate_candidate_json(
            SchemaKind::ShortKeys,
            serde_json::json!({
                "memo": "activate",
                "priority": [
                    { "id": "speak", "reason": "answer" },
                    { "id": "invented-module", "reason": "bad" }
                ]
            }),
        );

        assert_eq!(
            report,
            ValidationReport {
                success: false,
                failure: Some("unknown_module_id".to_owned()),
                priority_module_ids: vec!["speak".to_owned(), "invented-module".to_owned()],
                invalid_module_ids: vec!["invented-module".to_owned()],
            }
        );
    }

    #[test]
    fn winner_tie_breaks_by_invalid_ids_then_production_distance() {
        let summaries = vec![
            CandidateSummary {
                schema: SchemaKind::SplitArrays,
                attempted_trials: 100,
                evaluable_trials: 100,
                transport_errors: 0,
                successes: 90,
                schema_failures: 10,
                success_rate: 0.9,
                invalid_module_ids: 1,
                tool_issues: 0,
                production_distance: SchemaKind::SplitArrays.production_distance(),
                usage: UsageSummary::default(),
                success_priority_module_count: NumericStats::default(),
                success_module_id_counts: BTreeMap::new(),
            },
            CandidateSummary {
                schema: SchemaKind::ShortKeys,
                attempted_trials: 100,
                evaluable_trials: 100,
                transport_errors: 0,
                successes: 90,
                schema_failures: 10,
                success_rate: 0.9,
                invalid_module_ids: 0,
                tool_issues: 0,
                production_distance: SchemaKind::ShortKeys.production_distance(),
                usage: UsageSummary::default(),
                success_priority_module_count: NumericStats::default(),
                success_module_id_counts: BTreeMap::new(),
            },
            CandidateSummary {
                schema: SchemaKind::IdsMap,
                attempted_trials: 100,
                evaluable_trials: 100,
                transport_errors: 0,
                successes: 90,
                schema_failures: 10,
                success_rate: 0.9,
                invalid_module_ids: 0,
                tool_issues: 0,
                production_distance: SchemaKind::IdsMap.production_distance(),
                usage: UsageSummary::default(),
                success_priority_module_count: NumericStats::default(),
                success_module_id_counts: BTreeMap::new(),
            },
        ];

        assert_eq!(pick_winner(&summaries), Some(SchemaKind::IdsMap));
    }

    #[test]
    fn summary_excludes_transport_errors_from_success_rate() {
        let trials = vec![
            trial_report(SchemaKind::SplitArrays, 0, true, None),
            trial_report(
                SchemaKind::SplitArrays,
                1,
                false,
                Some(
                    "execution_error:execution error: request failure (kind=Transport, status=None)",
                ),
            ),
            trial_report(SchemaKind::PrimaryRest, 0, true, None),
            trial_report(SchemaKind::PrimaryRest, 1, true, None),
        ];

        let summaries = summarize_trials(&trials);

        let split = summaries
            .iter()
            .find(|summary| summary.schema == SchemaKind::SplitArrays)
            .unwrap();
        assert_eq!(split.attempted_trials, 2);
        assert_eq!(split.evaluable_trials, 1);
        assert_eq!(split.transport_errors, 1);
        assert_eq!(split.success_rate, 1.0);
        assert_eq!(pick_winner(&summaries), Some(SchemaKind::SplitArrays));
    }

    #[test]
    fn summary_tracks_usage_and_success_module_count_stats() {
        let trials = vec![
            trial_report_with_output(
                SchemaKind::SplitArrays,
                0,
                true,
                &["speak", "cognition-gate", "query-memory"],
                Some(usage(100, 20, 120)),
            ),
            trial_report_with_output(
                SchemaKind::SplitArrays,
                1,
                true,
                &["speak", "cognition-gate", "query-memory", "memory"],
                Some(usage(120, 30, 150)),
            ),
            trial_report_with_output(
                SchemaKind::SplitArrays,
                2,
                true,
                &[
                    "speak",
                    "cognition-gate",
                    "query-memory",
                    "memory",
                    "policy",
                ],
                Some(usage(140, 40, 180)),
            ),
        ];

        let summaries = summarize_trials(&trials);
        let split = summaries
            .iter()
            .find(|summary| summary.schema == SchemaKind::SplitArrays)
            .unwrap();

        assert_eq!(split.usage.sample_count, 3);
        assert_eq!(split.usage.input_tokens.min, Some(100));
        assert_eq!(split.usage.input_tokens.p50, Some(120));
        assert_eq!(split.usage.total_tokens.max, Some(180));
        assert_eq!(split.success_priority_module_count.min, Some(3));
        assert_eq!(split.success_priority_module_count.p50, Some(4));
        assert_eq!(split.success_priority_module_count.max, Some(5));
        assert_eq!(
            split.success_module_id_counts,
            BTreeMap::from([
                ("cognition-gate".to_owned(), 3),
                ("memory".to_owned(), 2),
                ("policy".to_owned(), 1),
                ("query-memory".to_owned(), 3),
                ("speak".to_owned(), 3),
            ])
        );
    }

    fn trial_report(
        schema: SchemaKind,
        trial: usize,
        success: bool,
        failure: Option<&str>,
    ) -> TrialReport {
        let module_ids = if success { &["speak"][..] } else { &[][..] };
        trial_report_with_output(schema, trial, success, module_ids, None).with_failure(failure)
    }

    fn trial_report_with_output(
        schema: SchemaKind,
        trial: usize,
        success: bool,
        module_ids: &[&str],
        usage: Option<lutum::Usage>,
    ) -> TrialReport {
        let priority_module_ids = module_ids
            .iter()
            .map(|id| (*id).to_owned())
            .collect::<Vec<_>>();
        TrialReport {
            schema,
            trial,
            success,
            transport_error: false,
            failure: None,
            priority_module_count: priority_module_ids.len(),
            priority_module_ids,
            invalid_module_ids: Vec::new(),
            tool_call_count: usize::from(success),
            tool_issue_count: 0,
            latency_ms: 1,
            usage,
        }
    }

    trait TrialReportTestExt {
        fn with_failure(self, failure: Option<&str>) -> Self;
    }

    impl TrialReportTestExt for TrialReport {
        fn with_failure(mut self, failure: Option<&str>) -> Self {
            let failure = failure.map(str::to_owned);
            self.transport_error = failure.as_deref().is_some_and(super::is_transport_failure);
            self.failure = failure;
            self
        }
    }

    fn usage(input_tokens: u64, output_tokens: u64, total_tokens: u64) -> lutum::Usage {
        lutum::Usage {
            input_tokens,
            output_tokens,
            total_tokens,
            cost_micros_usd: 0,
            cache_creation_tokens: 0,
            cache_read_tokens: 0,
        }
    }
}
