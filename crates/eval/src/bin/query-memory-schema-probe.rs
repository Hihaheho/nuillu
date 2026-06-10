use std::{
    collections::{BTreeMap, HashSet},
    fs,
    num::NonZeroUsize,
    path::{Path, PathBuf},
    time::Instant,
};

use anyhow::Context as _;
use clap::{Parser, ValueEnum};
use futures::{StreamExt, stream};
use lutum::{
    GenerationParams, InputMessageRole, ModelInput, ModelInputItem, Temperature,
    TextStepOutcomeWithTools,
};
use lutum_trace::{RawTraceEntry, RawTraceSnapshot};
use nuillu_eval::{
    install_lutum_trace_subscriber, parse_model_set_file, raw_trace_snapshot_json,
    resolve_llm_backends,
};
use schemars::{JsonSchema, Schema, SchemaGenerator};
use serde::{Deserialize, Serialize};

const DEFAULT_OUTPUT_DIR: &str = ".tmp/query-memory-schema-probe";
const RAW_GENERATED_PREVIEW_CHARS: usize = 4096;
const RAW_COLLECT_ERROR_PREVIEW_CHARS: usize = 2048;
const EXPECTED_FLAT_INDEX: &str = "koro-approach-primary";
const EXPECTED_LINKED_INDEX: &str = "koro-signal-drill-linked";
const ALL_MEMORY_INDEXES: &[&str] = &[
    "koro-food-fern-wait",
    "koro-food-circle-announce",
    "koro-approach-primary",
    "koro-food-distance-crest",
    "koro-food-rock-offer",
    "koro-signal-drill-linked",
];

const SYSTEM_PROMPT: &str = r#"You are the query-memory module finalizer.
Call exactly one tool now. Do not write prose, reasoning, markdown, or a plain answer.
Search and linked-memory fetch are closed. Use only existing tool results.
The search_memory and fetch_linked_memories results already in this session are fresh pending
retrieval evidence for this activation. Call broadcast_search_results when any pending retrieved
evidence should be published as module output. Call dispose_search_results only when no pending
retrieved evidence is useful enough to broadcast. Prior assistant prose is ignored by the runtime and
does not count as a successful broadcast. Never call dispose_search_results just because assistant
prose already summarized retrieved evidence. Use exact literal index strings from tool results,
never list positions."#;

const RETRIEVAL_CONTEXT: &str = r#"Memory question:
- How should Nui approach Koro near food?

Pending fresh search_memory result:
- index: koro-food-fern-wait
  content: Koro shares food only after Nui waits at the fern line.
  linked_neighbor_count: 0
- index: koro-food-circle-announce
  content: Nui must announce herself before entering Koro's feeding circle.
  linked_neighbor_count: 0
- index: koro-approach-primary
  content: Koro guards food, but relaxes when approached slowly from the side. Linked memories hold short-signal guidance for this approach.
  linked_neighbor_count: 1
- index: koro-food-distance-crest
  content: Koro lowers his crest when Nui keeps two body-lengths back near stored food.
  linked_neighbor_count: 0
- index: koro-food-rock-offer
  content: The feeding rock marks where Koro accepts offered fruit.
  linked_neighbor_count: 0

Pending fresh fetch_linked_memories result:
- index: koro-signal-drill-linked
  linked to: koro-approach-primary
  content: During dawn greeting drills, Pibi coached Nui to use short clear signals before touching a resting elder's shoulder.

Useful fresh evidence to broadcast:
- flat hit index: koro-approach-primary
- linked hit index: koro-signal-drill-linked"#;

#[derive(Debug, Parser)]
#[command(
    name = "query-memory-schema-probe",
    about = "Compare query-memory broadcast_search_results schema variants against a live LLM backend"
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
    #[value(name = "short-arrays")]
    ShortArrays,
    #[value(name = "object-items")]
    ObjectItems,
    #[value(name = "nested-selection")]
    NestedSelection,
}

impl SchemaKind {
    fn all() -> &'static [Self] {
        &[
            Self::Current,
            Self::ShortArrays,
            Self::ObjectItems,
            Self::NestedSelection,
        ]
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Current => "current",
            Self::ShortArrays => "short-arrays",
            Self::ObjectItems => "object-items",
            Self::NestedSelection => "nested-selection",
        }
    }

    fn production_distance(self) -> u32 {
        match self {
            Self::Current => 0,
            Self::ShortArrays => 1,
            Self::NestedSelection => 2,
            Self::ObjectItems => 3,
        }
    }
}

impl std::fmt::Display for SchemaKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

tokio::task_local! {
    static MEMORY_INDEX_SCHEMA: Schema;
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
#[serde(transparent)]
struct MemoryIndexId(String);

impl MemoryIndexId {
    fn as_str(&self) -> &str {
        self.0.as_str()
    }
}

impl From<MemoryIndexId> for String {
    fn from(value: MemoryIndexId) -> Self {
        value.0
    }
}

impl JsonSchema for MemoryIndexId {
    fn inline_schema() -> bool {
        true
    }

    fn schema_name() -> std::borrow::Cow<'static, str> {
        "MemoryIndexId".into()
    }

    fn schema_id() -> std::borrow::Cow<'static, str> {
        "nuillu_eval::query_memory_schema_probe::MemoryIndexId.dynamic".into()
    }

    fn json_schema(_generator: &mut SchemaGenerator) -> Schema {
        MEMORY_INDEX_SCHEMA
            .try_with(Clone::clone)
            .unwrap_or_else(|_| memory_index_schema())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
struct BroadcastSearchResultsOutput {
    broadcasted: bool,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
struct DisposeSearchResultsOutput {
    disposed: bool,
}

#[lutum::tool_input(name = "dispose_search_results", output = DisposeSearchResultsOutput)]
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
struct DisposeSearchResultsArgs {
    #[serde(default)]
    reason: String,
}

mod current_schema {
    use super::*;

    #[lutum::tool_input(name = "broadcast_search_results", output = BroadcastSearchResultsOutput)]
    #[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
    pub(super) struct BroadcastArgs {
        #[serde(default)]
        pub(super) hit_indexes: Vec<MemoryIndexId>,
        #[serde(default)]
        pub(super) linked_hit_indexes: Vec<MemoryIndexId>,
    }

    #[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema, lutum::Toolset)]
    pub(super) enum Tools {
        BroadcastSearchResults(BroadcastArgs),
        DisposeSearchResults(DisposeSearchResultsArgs),
    }
}

mod short_arrays_schema {
    use super::*;

    #[lutum::tool_input(name = "broadcast_search_results", output = BroadcastSearchResultsOutput)]
    #[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
    pub(super) struct BroadcastArgs {
        #[serde(default)]
        pub(super) hits: Vec<MemoryIndexId>,
        #[serde(default)]
        pub(super) linked: Vec<MemoryIndexId>,
    }

    #[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema, lutum::Toolset)]
    pub(super) enum Tools {
        BroadcastSearchResults(BroadcastArgs),
        DisposeSearchResults(DisposeSearchResultsArgs),
    }
}

mod object_items_schema {
    use super::*;

    #[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
    pub(super) struct MemoryIndexObject {
        pub(super) index: MemoryIndexId,
    }

    #[lutum::tool_input(name = "broadcast_search_results", output = BroadcastSearchResultsOutput)]
    #[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
    pub(super) struct BroadcastArgs {
        #[serde(default)]
        pub(super) hit_indexes: Vec<MemoryIndexObject>,
        #[serde(default)]
        pub(super) linked_hit_indexes: Vec<MemoryIndexObject>,
    }

    #[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema, lutum::Toolset)]
    pub(super) enum Tools {
        BroadcastSearchResults(BroadcastArgs),
        DisposeSearchResults(DisposeSearchResultsArgs),
    }
}

mod nested_selection_schema {
    use super::*;

    #[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
    pub(super) struct Selection {
        #[serde(default)]
        pub(super) flat_hit_indexes: Vec<MemoryIndexId>,
        #[serde(default)]
        pub(super) linked_hit_indexes: Vec<MemoryIndexId>,
    }

    #[lutum::tool_input(name = "broadcast_search_results", output = BroadcastSearchResultsOutput)]
    #[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
    pub(super) struct BroadcastArgs {
        pub(super) selection: Selection,
    }

    #[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema, lutum::Toolset)]
    pub(super) enum Tools {
        BroadcastSearchResults(BroadcastArgs),
        DisposeSearchResults(DisposeSearchResultsArgs),
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
enum CandidateDecision {
    Broadcast {
        hit_indexes: Vec<String>,
        linked_hit_indexes: Vec<String>,
    },
    Dispose {
        reason: String,
    },
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
struct ValidationReport {
    success: bool,
    failure: Option<String>,
    hit_indexes: Vec<String>,
    linked_hit_indexes: Vec<String>,
    invalid_indexes: Vec<String>,
    pseudo_tool_text: bool,
    plain_assistant_answer: bool,
}

#[derive(Clone, Debug, Serialize)]
struct TrialReport {
    schema: SchemaKind,
    trial: usize,
    success: bool,
    failure: Option<String>,
    transport_error: bool,
    hit_indexes: Vec<String>,
    linked_hit_indexes: Vec<String>,
    invalid_indexes: Vec<String>,
    tool_call_count: usize,
    tool_issue_count: usize,
    pseudo_tool_text: bool,
    plain_assistant_answer: bool,
    latency_ms: u128,
    usage: Option<lutum::Usage>,
    #[serde(flatten)]
    raw_trace: RawTrialTraceReport,
}

#[derive(Clone, Debug, Default, Serialize)]
struct RawTrialTraceReport {
    raw_trace_path: Option<String>,
    raw_stream_event_count: usize,
    raw_collect_error_count: usize,
    raw_generated_char_count: usize,
    raw_generated_head: Option<String>,
    raw_generated_tail: Option<String>,
    raw_collect_error_summaries: Vec<String>,
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
    tool_issues: usize,
    pseudo_tool_texts: usize,
    plain_assistant_answers: usize,
    dispose_decisions: usize,
    wrong_selections: usize,
    production_distance: u32,
    failure_reasons: BTreeMap<String, usize>,
}

#[derive(Debug, Serialize)]
struct ProbeReport {
    generated_at: String,
    model_set: String,
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

    fs::create_dir_all(&args.output)
        .with_context(|| format!("create output directory {}", args.output.display()))?;
    let memory_schema = memory_index_schema();
    let output_dir = args.output.clone();
    let trial_specs = selected
        .into_iter()
        .flat_map(|schema| (0..args.trials.get()).map(move |trial| (schema, trial)))
        .collect::<Vec<_>>();
    let trial_futures = trial_specs.into_iter().map(|(schema, trial)| {
        let memory_schema = memory_schema.clone();
        let output_dir = output_dir.clone();
        let lutum = &lutum;
        let mut generation = generation.clone();
        generation.seed = Some(args.seed_base.saturating_add(trial as u64));
        async move {
            MEMORY_INDEX_SCHEMA
                .scope(
                    memory_schema,
                    run_trial(schema, trial, generation, lutum, output_dir),
                )
                .await
        }
    });

    let mut trials = Vec::new();
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
        max_concurrent_llm_calls: concurrency_limit,
        trials_per_schema: args.trials.get(),
        schemas: summaries,
        winner,
        trials,
    };

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
    output_dir: PathBuf,
) -> TrialReport {
    let collected =
        lutum_trace::capture_raw(run_trial_body(schema, trial, generation, lutum)).await;
    let mut report = collected.output;
    attach_raw_trace_report(&mut report, &collected.raw, &output_dir);
    report
}

async fn run_trial_body(
    schema: SchemaKind,
    trial: usize,
    generation: GenerationParams,
    lutum: &lutum::Lutum,
) -> TrialReport {
    match schema {
        SchemaKind::Current => run_current_trial(trial, generation, lutum).await,
        SchemaKind::ShortArrays => run_short_arrays_trial(trial, generation, lutum).await,
        SchemaKind::ObjectItems => run_object_items_trial(trial, generation, lutum).await,
        SchemaKind::NestedSelection => run_nested_selection_trial(trial, generation, lutum).await,
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
        .available_tools([
            current_schema::ToolsSelector::BroadcastSearchResults,
            current_schema::ToolsSelector::DisposeSearchResults,
        ])
        .generation_config(generation)
        .collect()
        .await;
    build_trial_report(
        SchemaKind::Current,
        trial,
        started,
        outcome,
        |call| match call {
            current_schema::ToolsCall::BroadcastSearchResults(call) => {
                Some(CandidateDecision::Broadcast {
                    hit_indexes: ids_to_strings(&call.input.hit_indexes),
                    linked_hit_indexes: ids_to_strings(&call.input.linked_hit_indexes),
                })
            }
            current_schema::ToolsCall::DisposeSearchResults(call) => {
                Some(CandidateDecision::Dispose {
                    reason: call.input.reason.clone(),
                })
            }
        },
    )
}

async fn run_short_arrays_trial(
    trial: usize,
    generation: GenerationParams,
    lutum: &lutum::Lutum,
) -> TrialReport {
    let started = Instant::now();
    let outcome = lutum
        .text_turn(probe_input())
        .tools::<short_arrays_schema::Tools>()
        .available_tools([
            short_arrays_schema::ToolsSelector::BroadcastSearchResults,
            short_arrays_schema::ToolsSelector::DisposeSearchResults,
        ])
        .generation_config(generation)
        .collect()
        .await;
    build_trial_report(
        SchemaKind::ShortArrays,
        trial,
        started,
        outcome,
        |call| match call {
            short_arrays_schema::ToolsCall::BroadcastSearchResults(call) => {
                Some(CandidateDecision::Broadcast {
                    hit_indexes: ids_to_strings(&call.input.hits),
                    linked_hit_indexes: ids_to_strings(&call.input.linked),
                })
            }
            short_arrays_schema::ToolsCall::DisposeSearchResults(call) => {
                Some(CandidateDecision::Dispose {
                    reason: call.input.reason.clone(),
                })
            }
        },
    )
}

async fn run_object_items_trial(
    trial: usize,
    generation: GenerationParams,
    lutum: &lutum::Lutum,
) -> TrialReport {
    let started = Instant::now();
    let outcome = lutum
        .text_turn(probe_input())
        .tools::<object_items_schema::Tools>()
        .available_tools([
            object_items_schema::ToolsSelector::BroadcastSearchResults,
            object_items_schema::ToolsSelector::DisposeSearchResults,
        ])
        .generation_config(generation)
        .collect()
        .await;
    build_trial_report(
        SchemaKind::ObjectItems,
        trial,
        started,
        outcome,
        |call| match call {
            object_items_schema::ToolsCall::BroadcastSearchResults(call) => {
                Some(CandidateDecision::Broadcast {
                    hit_indexes: call
                        .input
                        .hit_indexes
                        .iter()
                        .map(|item| item.index.as_str().to_owned())
                        .collect(),
                    linked_hit_indexes: call
                        .input
                        .linked_hit_indexes
                        .iter()
                        .map(|item| item.index.as_str().to_owned())
                        .collect(),
                })
            }
            object_items_schema::ToolsCall::DisposeSearchResults(call) => {
                Some(CandidateDecision::Dispose {
                    reason: call.input.reason.clone(),
                })
            }
        },
    )
}

async fn run_nested_selection_trial(
    trial: usize,
    generation: GenerationParams,
    lutum: &lutum::Lutum,
) -> TrialReport {
    let started = Instant::now();
    let outcome = lutum
        .text_turn(probe_input())
        .tools::<nested_selection_schema::Tools>()
        .available_tools([
            nested_selection_schema::ToolsSelector::BroadcastSearchResults,
            nested_selection_schema::ToolsSelector::DisposeSearchResults,
        ])
        .generation_config(generation)
        .collect()
        .await;
    build_trial_report(
        SchemaKind::NestedSelection,
        trial,
        started,
        outcome,
        |call| match call {
            nested_selection_schema::ToolsCall::BroadcastSearchResults(call) => {
                Some(CandidateDecision::Broadcast {
                    hit_indexes: ids_to_strings(&call.input.selection.flat_hit_indexes),
                    linked_hit_indexes: ids_to_strings(&call.input.selection.linked_hit_indexes),
                })
            }
            nested_selection_schema::ToolsCall::DisposeSearchResults(call) => {
                Some(CandidateDecision::Dispose {
                    reason: call.input.reason.clone(),
                })
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
        lutum::CollectError<lutum::TextTurnReductionError, lutum::TextTurnStateWithTools<T>>,
    >,
    extract_decision: F,
) -> TrialReport
where
    T: lutum::Toolset,
    F: Fn(&T::ToolCall) -> Option<CandidateDecision>,
{
    let latency_ms = started.elapsed().as_millis();
    match outcome {
        Ok(TextStepOutcomeWithTools::NeedsTools(round)) => {
            let validation = round
                .tool_calls
                .iter()
                .find_map(extract_decision)
                .map(validate_candidate_decision)
                .unwrap_or_else(|| {
                    let failure = if round.recoverable_tool_call_issues().is_empty() {
                        "no_finalization_tool".to_owned()
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
                        hit_indexes: Vec::new(),
                        linked_hit_indexes: Vec::new(),
                        invalid_indexes: Vec::new(),
                        pseudo_tool_text: false,
                        plain_assistant_answer: false,
                    }
                });
            TrialReport {
                schema,
                trial,
                success: validation.success,
                failure: validation.failure,
                transport_error: false,
                hit_indexes: validation.hit_indexes,
                linked_hit_indexes: validation.linked_hit_indexes,
                invalid_indexes: validation.invalid_indexes,
                tool_call_count: round.tool_calls.len(),
                tool_issue_count: round.recoverable_tool_call_issues().len(),
                pseudo_tool_text: validation.pseudo_tool_text,
                plain_assistant_answer: validation.plain_assistant_answer,
                latency_ms,
                usage: Some(round.usage),
                raw_trace: RawTrialTraceReport::default(),
            }
        }
        Ok(TextStepOutcomeWithTools::Finished(result)) => {
            let assistant_text = result.assistant_text();
            let pseudo_tool_text = looks_like_pseudo_tool(&assistant_text);
            TrialReport {
                schema,
                trial,
                success: false,
                failure: Some(if pseudo_tool_text {
                    "pseudo_tool_text".to_owned()
                } else {
                    format!("finished_without_tool:{:?}", result.finish_reason)
                }),
                transport_error: false,
                hit_indexes: Vec::new(),
                linked_hit_indexes: Vec::new(),
                invalid_indexes: Vec::new(),
                tool_call_count: 0,
                tool_issue_count: 0,
                pseudo_tool_text,
                plain_assistant_answer: !pseudo_tool_text,
                latency_ms,
                usage: Some(result.usage),
                raw_trace: RawTrialTraceReport::default(),
            }
        }
        Ok(TextStepOutcomeWithTools::FinishedNoOutput(result)) => TrialReport {
            schema,
            trial,
            success: false,
            failure: Some(format!(
                "finished_without_output:{:?}",
                result.finish_reason
            )),
            transport_error: false,
            hit_indexes: Vec::new(),
            linked_hit_indexes: Vec::new(),
            invalid_indexes: Vec::new(),
            tool_call_count: 0,
            tool_issue_count: 0,
            pseudo_tool_text: false,
            plain_assistant_answer: false,
            latency_ms,
            usage: Some(result.usage),
            raw_trace: RawTrialTraceReport::default(),
        },
        Err(error) => {
            let error_text = error.to_string();
            let failure = classify_execution_error(&error_text);
            TrialReport {
                schema,
                trial,
                success: false,
                failure: Some(failure.clone()),
                transport_error: is_transport_failure(&error_text),
                hit_indexes: Vec::new(),
                linked_hit_indexes: Vec::new(),
                invalid_indexes: Vec::new(),
                tool_call_count: 0,
                tool_issue_count: 0,
                pseudo_tool_text: false,
                plain_assistant_answer: false,
                latency_ms,
                usage: None,
                raw_trace: RawTrialTraceReport::default(),
            }
        }
    }
}

fn attach_raw_trace_report(
    report: &mut TrialReport,
    raw_trace: &RawTraceSnapshot,
    output_dir: &Path,
) {
    let generated = raw_generated_content(raw_trace);
    let generated_char_count = generated.chars().count();
    let (raw_generated_head, raw_generated_tail) =
        raw_head_tail(&generated, RAW_GENERATED_PREVIEW_CHARS);
    let raw_collect_error_summaries = raw_collect_error_summaries(raw_trace);
    let raw_stream_event_count = raw_trace
        .entries
        .iter()
        .filter(|entry| matches!(entry, RawTraceEntry::StreamEvent { .. }))
        .count();
    let raw_collect_error_count = raw_collect_error_summaries.len();

    report.raw_trace = RawTrialTraceReport {
        raw_trace_path: None,
        raw_stream_event_count,
        raw_collect_error_count,
        raw_generated_char_count: generated_char_count,
        raw_generated_head,
        raw_generated_tail,
        raw_collect_error_summaries,
    };

    if report.success && report.raw_trace.raw_collect_error_count == 0 {
        return;
    }

    match write_raw_trace_file(output_dir, report.schema, report.trial, raw_trace) {
        Ok(path) => {
            report.raw_trace.raw_trace_path = Some(path.display().to_string());
        }
        Err(error) => {
            report.raw_trace.raw_trace_path = Some(format!("failed to write raw trace: {error}"));
        }
    }
}

fn write_raw_trace_file(
    output_dir: &Path,
    schema: SchemaKind,
    trial: usize,
    raw_trace: &RawTraceSnapshot,
) -> anyhow::Result<PathBuf> {
    let trace_dir = output_dir.join("raw-traces");
    fs::create_dir_all(&trace_dir)
        .with_context(|| format!("create raw trace directory {}", trace_dir.display()))?;
    let path = trace_dir.join(format!("{}-{:03}.json", schema.as_str(), trial + 1));
    fs::write(
        &path,
        serde_json::to_string_pretty(&raw_trace_snapshot_json(raw_trace))?,
    )
    .with_context(|| format!("write raw trace {}", path.display()))?;
    Ok(path)
}

fn raw_generated_content(raw_trace: &RawTraceSnapshot) -> String {
    let mut generated = String::new();
    for entry in &raw_trace.entries {
        let RawTraceEntry::StreamEvent { payload, .. } = entry else {
            continue;
        };
        append_generated_payload(payload, &mut generated);
    }
    generated
}

fn append_generated_payload(payload: &str, output: &mut String) {
    let payload = payload.trim();
    if payload.is_empty() {
        return;
    }
    if let Ok(value) = serde_json::from_str::<serde_json::Value>(payload) {
        append_generated_json_strings(&value, output);
        return;
    }
    for line in payload.lines() {
        let Some(data) = line.trim().strip_prefix("data:") else {
            continue;
        };
        let data = data.trim();
        if data == "[DONE]" {
            continue;
        }
        if let Ok(value) = serde_json::from_str::<serde_json::Value>(data) {
            append_generated_json_strings(&value, output);
        }
    }
}

fn append_generated_json_strings(value: &serde_json::Value, output: &mut String) {
    match value {
        serde_json::Value::Array(items) => {
            for item in items {
                append_generated_json_strings(item, output);
            }
        }
        serde_json::Value::Object(map) => {
            for (key, child) in map {
                if generated_text_key(key) {
                    if let Some(text) = child.as_str() {
                        output.push_str(text);
                        continue;
                    }
                }
                append_generated_json_strings(child, output);
            }
        }
        _ => {}
    }
}

fn generated_text_key(key: &str) -> bool {
    matches!(
        key,
        "arguments"
            | "arguments_json_delta"
            | "content"
            | "delta"
            | "json_delta"
            | "partial_json"
            | "text"
    )
}

fn raw_head_tail(text: &str, limit: usize) -> (Option<String>, Option<String>) {
    if text.is_empty() {
        return (None, None);
    }
    let char_count = text.chars().count();
    let head = text.chars().take(limit).collect::<String>();
    let tail = if char_count > limit {
        let start = char_count.saturating_sub(limit);
        Some(text.chars().skip(start).collect::<String>())
    } else {
        None
    };
    (Some(head), tail)
}

fn raw_collect_error_summaries(raw_trace: &RawTraceSnapshot) -> Vec<String> {
    raw_trace
        .entries
        .iter()
        .filter_map(|entry| {
            let RawTraceEntry::CollectError {
                partial_summary,
                error,
                ..
            } = entry
            else {
                return None;
            };
            let summary = if error.contains("output token limit") {
                "output_token_limit".to_owned()
            } else if is_transport_failure(error) {
                "transport_error".to_owned()
            } else {
                "collect_error".to_owned()
            };
            Some(raw_text_preview(
                &format!("{summary}\n{}", redact_raw_partial_summary(partial_summary)),
                RAW_COLLECT_ERROR_PREVIEW_CHARS,
            ))
        })
        .collect()
}

fn redact_raw_partial_summary(summary: &str) -> String {
    summary
        .split(", ")
        .filter(|field| {
            !field.starts_with("request_id=")
                && !field.starts_with("model=")
                && !field.starts_with("usage=")
        })
        .collect::<Vec<_>>()
        .join(", ")
}

fn raw_text_preview(text: &str, limit: usize) -> String {
    let char_count = text.chars().count();
    if char_count <= limit {
        return text.to_owned();
    }
    let omitted = char_count - limit;
    format!(
        "{}\n[... {omitted} chars omitted]",
        text.chars().take(limit).collect::<String>()
    )
}

fn validate_candidate_decision(decision: CandidateDecision) -> ValidationReport {
    match decision {
        CandidateDecision::Dispose { reason } => ValidationReport {
            success: false,
            failure: Some(if reason.trim().is_empty() {
                "dispose".to_owned()
            } else {
                "dispose_with_reason".to_owned()
            }),
            hit_indexes: Vec::new(),
            linked_hit_indexes: Vec::new(),
            invalid_indexes: Vec::new(),
            pseudo_tool_text: false,
            plain_assistant_answer: false,
        },
        CandidateDecision::Broadcast {
            mut hit_indexes,
            mut linked_hit_indexes,
        } => {
            hit_indexes.sort();
            linked_hit_indexes.sort();
            hit_indexes.dedup();
            linked_hit_indexes.dedup();
            let valid = ALL_MEMORY_INDEXES.iter().copied().collect::<HashSet<_>>();
            let invalid_indexes = hit_indexes
                .iter()
                .chain(linked_hit_indexes.iter())
                .filter(|index| !valid.contains(index.as_str()))
                .cloned()
                .collect::<Vec<_>>();
            let expected_hit_indexes = vec![EXPECTED_FLAT_INDEX.to_owned()];
            let expected_linked_hit_indexes = vec![EXPECTED_LINKED_INDEX.to_owned()];
            let success = invalid_indexes.is_empty()
                && hit_indexes == expected_hit_indexes
                && linked_hit_indexes == expected_linked_hit_indexes;
            let failure = if success {
                None
            } else if !invalid_indexes.is_empty() {
                Some("unknown_memory_index".to_owned())
            } else if hit_indexes.is_empty() && linked_hit_indexes.is_empty() {
                Some("empty_selection".to_owned())
            } else {
                Some("wrong_selection".to_owned())
            };
            ValidationReport {
                success,
                failure,
                hit_indexes,
                linked_hit_indexes,
                invalid_indexes,
                pseudo_tool_text: false,
                plain_assistant_answer: false,
            }
        }
    }
}

fn ids_to_strings(ids: &[MemoryIndexId]) -> Vec<String> {
    ids.iter().map(|id| id.as_str().to_owned()).collect()
}

fn is_transport_failure(failure: &str) -> bool {
    failure.contains("kind=Transport") || failure.contains("error sending request")
}

fn classify_execution_error(error: &str) -> String {
    if error.contains("output token limit") {
        "execution_error:output_token_limit".to_owned()
    } else if is_transport_failure(error) {
        "execution_error:transport".to_owned()
    } else if error.contains("reduction error") {
        "execution_error:reduction".to_owned()
    } else {
        "execution_error:other".to_owned()
    }
}

fn looks_like_pseudo_tool(text: &str) -> bool {
    let normalized = text.trim().to_ascii_lowercase();
    if normalized.is_empty() {
        return false;
    }
    normalized.contains("broadcast_search_results")
        || normalized.contains("dispose_search_results")
        || normalized.contains("<tool_call")
        || (normalized.starts_with('{') && normalized.ends_with('}'))
}

fn memory_index_schema() -> Schema {
    Schema::try_from(serde_json::json!({
        "type": "string",
        "enum": ALL_MEMORY_INDEXES,
    }))
    .expect("memory index schema must be valid")
}

fn probe_input() -> ModelInput {
    ModelInput::from_items(vec![
        ModelInputItem::text(InputMessageRole::System, SYSTEM_PROMPT),
        ModelInputItem::text(InputMessageRole::User, RETRIEVAL_CONTEXT),
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
            let tool_issues = evaluable_trials
                .iter()
                .map(|trial| trial.tool_issue_count)
                .sum();
            let pseudo_tool_texts = evaluable_trials
                .iter()
                .filter(|trial| trial.pseudo_tool_text)
                .count();
            let plain_assistant_answers = evaluable_trials
                .iter()
                .filter(|trial| trial.plain_assistant_answer)
                .count();
            let dispose_decisions = evaluable_trials
                .iter()
                .filter(|trial| {
                    trial
                        .failure
                        .as_deref()
                        .is_some_and(|failure| failure.starts_with("dispose"))
                })
                .count();
            let wrong_selections = evaluable_trials
                .iter()
                .filter(|trial| trial.failure.as_deref() == Some("wrong_selection"))
                .count();
            let mut failure_reasons = BTreeMap::new();
            for trial in &evaluable_trials {
                if let Some(failure) = trial.failure.as_deref() {
                    *failure_reasons.entry(failure.to_owned()).or_insert(0) += 1;
                }
            }
            let transport_errors = schema_trials
                .iter()
                .filter(|trial| trial.transport_error)
                .count();
            let evaluable_count = evaluable_trials.len();
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
                tool_issues,
                pseudo_tool_texts,
                plain_assistant_answers,
                dispose_decisions,
                wrong_selections,
                production_distance: schema.production_distance(),
                failure_reasons,
            })
        })
        .collect()
}

fn pick_winner(summaries: &[CandidateSummary]) -> Option<SchemaKind> {
    summaries
        .iter()
        .filter(|summary| summary.evaluable_trials > 0)
        .max_by(|a, b| {
            a.successes
                .cmp(&b.successes)
                .then_with(|| b.production_distance.cmp(&a.production_distance))
        })
        .map(|summary| summary.schema)
}

fn render_markdown_report(report: &ProbeReport) -> String {
    let mut out = String::new();
    out.push_str("# Query-Memory Schema Probe\n\n");
    out.push_str(&format!(
        "- generated_at: {}\n- model_set: {}\n- max_concurrent_llm_calls: {}\n- trials_per_schema: {}\n- winner: {}\n\n",
        report.generated_at,
        report.model_set,
        report.max_concurrent_llm_calls,
        report.trials_per_schema,
        report
            .winner
            .map(|winner| winner.as_str())
            .unwrap_or("(none)")
    ));
    out.push_str(
        "| schema | successes | evaluable | attempted | success_rate | transport_errors | tool_issues | pseudo_tool_texts | plain_answers | dispose | wrong_selection | production_distance |\n",
    );
    out.push_str(
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n",
    );
    for summary in &report.schemas {
        out.push_str(&format!(
            "| {} | {} | {} | {} | {:.3} | {} | {} | {} | {} | {} | {} | {} |\n",
            summary.schema,
            summary.successes,
            summary.evaluable_trials,
            summary.attempted_trials,
            summary.success_rate,
            summary.transport_errors,
            summary.tool_issues,
            summary.pseudo_tool_texts,
            summary.plain_assistant_answers,
            summary.dispose_decisions,
            summary.wrong_selections,
            summary.production_distance
        ));
    }
    out.push_str("\n## Failure Reasons\n\n");
    out.push_str("| schema | reasons |\n| --- | --- |\n");
    for summary in &report.schemas {
        let reasons = if summary.failure_reasons.is_empty() {
            "-".to_owned()
        } else {
            summary
                .failure_reasons
                .iter()
                .map(|(reason, count)| format!("{reason}: {count}"))
                .collect::<Vec<_>>()
                .join(", ")
        };
        out.push_str(&format!("| {} | {} |\n", summary.schema, reasons));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validation_requires_exact_flat_and_linked_indexes() {
        let valid = validate_candidate_decision(CandidateDecision::Broadcast {
            hit_indexes: vec![EXPECTED_FLAT_INDEX.to_owned()],
            linked_hit_indexes: vec![EXPECTED_LINKED_INDEX.to_owned()],
        });
        assert!(valid.success);

        let extra = validate_candidate_decision(CandidateDecision::Broadcast {
            hit_indexes: vec![
                EXPECTED_FLAT_INDEX.to_owned(),
                "koro-food-fern-wait".to_owned(),
            ],
            linked_hit_indexes: vec![EXPECTED_LINKED_INDEX.to_owned()],
        });
        assert_eq!(extra.failure.as_deref(), Some("wrong_selection"));
    }

    #[test]
    fn winner_prefers_production_distance_on_tie() {
        let summaries = vec![
            CandidateSummary {
                schema: SchemaKind::ObjectItems,
                attempted_trials: 10,
                evaluable_trials: 10,
                transport_errors: 0,
                successes: 8,
                schema_failures: 2,
                success_rate: 0.8,
                tool_issues: 0,
                pseudo_tool_texts: 0,
                plain_assistant_answers: 0,
                dispose_decisions: 0,
                wrong_selections: 2,
                production_distance: SchemaKind::ObjectItems.production_distance(),
                failure_reasons: BTreeMap::new(),
            },
            CandidateSummary {
                schema: SchemaKind::Current,
                attempted_trials: 10,
                evaluable_trials: 10,
                transport_errors: 0,
                successes: 8,
                schema_failures: 2,
                success_rate: 0.8,
                tool_issues: 0,
                pseudo_tool_texts: 0,
                plain_assistant_answers: 0,
                dispose_decisions: 0,
                wrong_selections: 2,
                production_distance: SchemaKind::Current.production_distance(),
                failure_reasons: BTreeMap::new(),
            },
        ];

        assert_eq!(pick_winner(&summaries), Some(SchemaKind::Current));
    }
}
