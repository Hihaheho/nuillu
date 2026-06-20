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
    GenerationParams, InputMessageRole, MaxOutputTokens, ModelInput, ModelInputItem, Seed,
    Temperature, TextStepOutcomeWithTools,
};
use nuillu_eval::{install_lutum_trace_subscriber, parse_model_set_file, resolve_llm_backends};
use schemars::{JsonSchema, Schema, SchemaGenerator};
use serde::{Deserialize, Serialize};

const DEFAULT_OUTPUT_DIR: &str = ".tmp/cognition-gate-selection-probe";
const MAX_TOP_IDS: usize = 2;

#[derive(Debug, Parser)]
#[command(
    name = "cognition-gate-selection-probe",
    about = "Compare cognition-gate winner and ranking tool shapes against a live LLM backend"
)]
struct Args {
    /// Model set Eure file with per-role backend config.
    #[arg(long)]
    model_set: PathBuf,

    /// Trials to run per selected schema/prompt/case candidate.
    #[arg(long, default_value = "10")]
    trials: NonZeroUsize,

    /// Output directory for JSON and Markdown reports.
    #[arg(long, default_value = DEFAULT_OUTPUT_DIR)]
    output: PathBuf,

    /// Schema candidates to run. Comma-separated values are accepted.
    #[arg(long = "schema", value_enum, value_delimiter = ',')]
    schemas: Vec<SchemaKind>,

    /// Prompt candidates to run. Comma-separated values are accepted.
    #[arg(long = "prompt", value_enum, value_delimiter = ',')]
    prompts: Vec<PromptKind>,

    /// Fixture cases to run. Comma-separated values are accepted.
    #[arg(long = "case", value_enum, value_delimiter = ',')]
    cases: Vec<CaseKind>,

    /// Candidate ID styles exposed to the tool schema. Comma-separated values are accepted.
    #[arg(long = "id-style", value_enum, value_delimiter = ',')]
    id_styles: Vec<IdStyle>,

    /// Candidate display order variants. Comma-separated values are accepted.
    #[arg(long = "order", value_enum, value_delimiter = ',')]
    orders: Vec<OrderKind>,

    /// Generation temperature used for each probe turn.
    #[arg(long, default_value = "0.2")]
    temperature: f32,

    /// Max output tokens for each probe turn.
    #[arg(long, default_value = "768")]
    max_output_tokens: u32,

    /// Base seed. Trial index is added to this value for every candidate.
    #[arg(long, default_value = "1")]
    seed_base: u64,
}

#[derive(
    Clone, Copy, Debug, Eq, PartialEq, Hash, Ord, PartialOrd, Serialize, Deserialize, ValueEnum,
)]
#[serde(rename_all = "kebab-case")]
enum SchemaKind {
    #[value(name = "winner-id")]
    WinnerId,
    #[value(name = "rank-all-ids")]
    RankAllIds,
    #[value(name = "rank-top-ids")]
    RankTopIds,
    #[value(name = "primary-rest")]
    PrimaryRest,
}

impl SchemaKind {
    fn all() -> &'static [Self] {
        &[
            Self::WinnerId,
            Self::RankAllIds,
            Self::RankTopIds,
            Self::PrimaryRest,
        ]
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::WinnerId => "winner-id",
            Self::RankAllIds => "rank-all-ids",
            Self::RankTopIds => "rank-top-ids",
            Self::PrimaryRest => "primary-rest",
        }
    }

    fn production_distance(self) -> u32 {
        match self {
            Self::WinnerId => 0,
            Self::RankTopIds => 1,
            Self::RankAllIds => 2,
            Self::PrimaryRest => 3,
        }
    }
}

impl std::fmt::Display for SchemaKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(
    Clone, Copy, Debug, Eq, PartialEq, Hash, Ord, PartialOrd, Serialize, Deserialize, ValueEnum,
)]
#[serde(rename_all = "kebab-case")]
enum PromptKind {
    Production,
    #[value(name = "importance-sort")]
    ImportanceSort,
    #[value(name = "anti-position")]
    AntiPosition,
}

impl PromptKind {
    fn all() -> &'static [Self] {
        &[Self::Production, Self::ImportanceSort, Self::AntiPosition]
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Production => "production",
            Self::ImportanceSort => "importance-sort",
            Self::AntiPosition => "anti-position",
        }
    }

    fn production_distance(self) -> u32 {
        match self {
            Self::Production => 0,
            Self::ImportanceSort => 1,
            Self::AntiPosition => 2,
        }
    }
}

impl std::fmt::Display for PromptKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(
    Clone, Copy, Debug, Eq, PartialEq, Hash, Ord, PartialOrd, Serialize, Deserialize, ValueEnum,
)]
#[serde(rename_all = "kebab-case")]
enum CaseKind {
    #[value(name = "process-before-name-question")]
    ProcessBeforeNameQuestion,
    #[value(name = "stale-greeting-before-status-question")]
    StaleGreetingBeforeStatusQuestion,
    #[value(name = "irrelevant-memory-before-warning")]
    IrrelevantMemoryBeforeWarning,
}

impl CaseKind {
    fn all() -> &'static [Self] {
        &[
            Self::ProcessBeforeNameQuestion,
            Self::StaleGreetingBeforeStatusQuestion,
            Self::IrrelevantMemoryBeforeWarning,
        ]
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::ProcessBeforeNameQuestion => "process-before-name-question",
            Self::StaleGreetingBeforeStatusQuestion => "stale-greeting-before-status-question",
            Self::IrrelevantMemoryBeforeWarning => "irrelevant-memory-before-warning",
        }
    }
}

impl std::fmt::Display for CaseKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(
    Clone, Copy, Debug, Eq, PartialEq, Hash, Ord, PartialOrd, Serialize, Deserialize, ValueEnum,
)]
#[serde(rename_all = "kebab-case")]
enum IdStyle {
    Semantic,
    #[value(name = "opaque-static")]
    OpaqueStatic,
    Numeric,
    Letter,
    #[value(name = "source-letter")]
    SourceLetter,
}

impl IdStyle {
    fn all() -> &'static [Self] {
        &[
            Self::Semantic,
            Self::OpaqueStatic,
            Self::Numeric,
            Self::Letter,
            Self::SourceLetter,
        ]
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Semantic => "semantic",
            Self::OpaqueStatic => "opaque-static",
            Self::Numeric => "numeric",
            Self::Letter => "letter",
            Self::SourceLetter => "source-letter",
        }
    }
}

impl std::fmt::Display for IdStyle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(
    Clone, Copy, Debug, Eq, PartialEq, Hash, Ord, PartialOrd, Serialize, Deserialize, ValueEnum,
)]
#[serde(rename_all = "kebab-case")]
enum OrderKind {
    Natural,
    #[value(name = "rotate-left-1")]
    RotateLeft1,
    #[value(name = "rotate-left-2")]
    RotateLeft2,
    #[value(name = "rotate-left-3")]
    RotateLeft3,
}

impl OrderKind {
    fn all() -> &'static [Self] {
        &[
            Self::Natural,
            Self::RotateLeft1,
            Self::RotateLeft2,
            Self::RotateLeft3,
        ]
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Natural => "natural",
            Self::RotateLeft1 => "rotate-left-1",
            Self::RotateLeft2 => "rotate-left-2",
            Self::RotateLeft3 => "rotate-left-3",
        }
    }

    fn indices(self) -> [usize; 4] {
        match self {
            Self::Natural => [0, 1, 2, 3],
            Self::RotateLeft1 => [1, 2, 3, 0],
            Self::RotateLeft2 => [2, 3, 0, 1],
            Self::RotateLeft3 => [3, 0, 1, 2],
        }
    }
}

impl std::fmt::Display for OrderKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

tokio::task_local! {
    static CANDIDATE_ID_SCHEMA: Schema;
}

#[derive(Clone, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
struct CandidateId(String);

impl CandidateId {
    fn as_str(&self) -> &str {
        self.0.as_str()
    }
}

impl From<CandidateId> for String {
    fn from(value: CandidateId) -> Self {
        value.0
    }
}

impl JsonSchema for CandidateId {
    fn inline_schema() -> bool {
        true
    }

    fn schema_name() -> std::borrow::Cow<'static, str> {
        "CandidateId".into()
    }

    fn schema_id() -> std::borrow::Cow<'static, str> {
        "nuillu_eval::cognition_gate_selection_probe::CandidateId.dynamic".into()
    }

    fn json_schema(_generator: &mut SchemaGenerator) -> Schema {
        CANDIDATE_ID_SCHEMA
            .try_with(Clone::clone)
            .unwrap_or_else(|_| candidate_id_schema([]))
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
struct SelectionOutput {
    accepted: bool,
}

mod winner_id_schema {
    use super::*;

    #[lutum::tool_input(name = "select_cognition_winner", output = SelectionOutput)]
    #[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
    pub(super) struct SelectCognitionWinnerArgs {
        pub(super) champion_id: CandidateId,
        #[serde(default)]
        pub(super) optional_secondary_id: Option<CandidateId>,
    }

    #[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema, lutum::Toolset)]
    pub(super) enum Tools {
        SelectCognitionWinner(SelectCognitionWinnerArgs),
    }
}

mod rank_all_schema {
    use super::*;

    #[lutum::tool_input(name = "rank_cognition_candidates", output = SelectionOutput)]
    #[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
    pub(super) struct RankCognitionCandidatesArgs {
        pub(super) ranked_candidate_ids: Vec<CandidateId>,
    }

    #[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema, lutum::Toolset)]
    pub(super) enum Tools {
        RankCognitionCandidates(RankCognitionCandidatesArgs),
    }
}

mod rank_top_schema {
    use super::*;

    #[lutum::tool_input(name = "rank_top_cognition_candidates", output = SelectionOutput)]
    #[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
    pub(super) struct RankTopCognitionCandidatesArgs {
        pub(super) top_candidate_ids: Vec<CandidateId>,
    }

    #[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema, lutum::Toolset)]
    pub(super) enum Tools {
        RankTopCognitionCandidates(RankTopCognitionCandidatesArgs),
    }
}

mod primary_rest_schema {
    use super::*;

    #[lutum::tool_input(name = "select_cognition_priority", output = SelectionOutput)]
    #[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
    pub(super) struct SelectCognitionPriorityArgs {
        pub(super) primary_id: CandidateId,
        #[serde(default)]
        pub(super) remaining_candidate_ids: Vec<CandidateId>,
    }

    #[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema, lutum::Toolset)]
    pub(super) enum Tools {
        SelectCognitionPriority(SelectCognitionPriorityArgs),
    }
}

#[derive(Clone, Debug)]
struct Candidate {
    id: String,
    text: &'static str,
}

#[derive(Clone, Debug)]
struct Fixture {
    kind: CaseKind,
    id_style: IdStyle,
    order: OrderKind,
    cognition_context: &'static str,
    candidates: Vec<Candidate>,
    acceptable_top_ids: HashSet<String>,
}

impl Fixture {
    fn first_id(&self) -> &str {
        self.candidates[0].id.as_str()
    }

    fn known_ids(&self) -> HashSet<&str> {
        self.candidates
            .iter()
            .map(|candidate| candidate.id.as_str())
            .collect()
    }
}

#[derive(Clone, Debug, Serialize)]
struct Selection {
    ids: Vec<String>,
}

#[derive(Clone, Debug, Serialize)]
struct ValidationReport {
    success: bool,
    failure: Option<String>,
    top_id: Option<String>,
    selected_ids: Vec<String>,
    first_position_top: bool,
    duplicate_id: bool,
    unknown_id: bool,
    incomplete_rank_all: bool,
}

#[derive(Clone, Debug, Serialize)]
struct TrialReport {
    id_style: IdStyle,
    order: OrderKind,
    schema: SchemaKind,
    prompt: PromptKind,
    case: CaseKind,
    trial: usize,
    success: bool,
    failure: Option<String>,
    top_id: Option<String>,
    selected_ids: Vec<String>,
    first_position_top: bool,
    transport_error: bool,
    duplicate_id: bool,
    unknown_id: bool,
    incomplete_rank_all: bool,
    tool_call_count: usize,
    tool_issue_count: usize,
    latency_ms: u128,
    usage: Option<lutum::Usage>,
}

#[derive(Clone, Debug, Serialize)]
struct CandidateSummary {
    id_style: IdStyle,
    order: OrderKind,
    schema: SchemaKind,
    prompt: PromptKind,
    case: CaseKind,
    attempted_trials: usize,
    evaluable_trials: usize,
    transport_errors: usize,
    successes: usize,
    failures: usize,
    success_rate: f64,
    first_position_tops: usize,
    first_position_top_rate: f64,
    duplicate_ids: usize,
    unknown_ids: usize,
    incomplete_rank_all: usize,
    tool_issues: usize,
    top_id_counts: BTreeMap<String, usize>,
    failure_reasons: BTreeMap<String, usize>,
    production_distance: u32,
    usage: UsageSummary,
    latency_ms: NumericStats,
}

#[derive(Clone, Debug, Serialize)]
struct AggregateSummary {
    id_style: IdStyle,
    order: OrderKind,
    schema: SchemaKind,
    prompt: PromptKind,
    attempted_trials: usize,
    evaluable_trials: usize,
    transport_errors: usize,
    successes: usize,
    success_rate: f64,
    first_position_tops: usize,
    first_position_top_rate: f64,
    duplicate_ids: usize,
    unknown_ids: usize,
    incomplete_rank_all: usize,
    tool_issues: usize,
    production_distance: u32,
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
    id_styles: Vec<IdStyle>,
    orders: Vec<OrderKind>,
    max_concurrent_llm_calls: usize,
    trials_per_candidate: usize,
    schemas: Vec<SchemaKind>,
    prompts: Vec<PromptKind>,
    cases: Vec<CaseKind>,
    candidates: Vec<CandidateSummary>,
    aggregates: Vec<AggregateSummary>,
    winner: Option<Winner>,
    trials: Vec<TrialReport>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
struct Winner {
    id_style: IdStyle,
    order: OrderKind,
    schema: SchemaKind,
    prompt: PromptKind,
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> anyhow::Result<()> {
    install_lutum_trace_subscriber()?;
    let args = Args::parse();
    let schemas = selected_schemas(&args);
    let prompts = selected_prompts(&args);
    let cases = selected_cases(&args);
    let id_styles = selected_id_styles(&args);
    let orders = selected_orders(&args);
    let temperature = Temperature::try_from(args.temperature)
        .with_context(|| format!("invalid --temperature {}", args.temperature))?;

    let model_set = parse_model_set_file(&args.model_set)?;
    let backends = resolve_llm_backends(&model_set)?;
    let backend = backends.premium;
    let lutum = nuillu_server::build_lutum(&backend, None).context("build premium-tier lutum")?;
    let concurrency_limit = backend
        .max_concurrent_llm_calls
        .map_or(1, NonZeroUsize::get);
    let mut generation = GenerationParams::default();
    generation.temperature = Some(temperature);
    generation.max_output_tokens = Some(MaxOutputTokens::new(args.max_output_tokens));

    let mut trials = Vec::new();
    let mut trial_specs = Vec::new();
    for id_style in &id_styles {
        for order in &orders {
            for schema in &schemas {
                for prompt in &prompts {
                    for case in &cases {
                        for trial in 0..args.trials.get() {
                            trial_specs.push((*id_style, *order, *schema, *prompt, *case, trial));
                        }
                    }
                }
            }
        }
    }
    let trial_futures =
        trial_specs
            .into_iter()
            .map(|(id_style, order, schema, prompt, case, trial)| {
                let fixture = fixture(case, id_style, order);
                let candidate_schema = candidate_id_schema(
                    fixture
                        .candidates
                        .iter()
                        .map(|candidate| candidate.id.as_str()),
                );
                let lutum = &lutum;
                let mut generation = generation.clone();
                generation.seed = Some(Seed::new(args.seed_base.saturating_add(trial as u64)));
                async move {
                    CANDIDATE_ID_SCHEMA
                        .scope(
                            candidate_schema,
                            run_trial(schema, prompt, fixture, trial, generation, lutum),
                        )
                        .await
                }
            });
    let mut trial_stream = stream::iter(trial_futures).buffer_unordered(concurrency_limit);
    while let Some(report) = trial_stream.next().await {
        println!(
            "id_style={} order={} schema={} prompt={} case={} trial={} success={} top={} first_top={} failure={}",
            report.id_style,
            report.order,
            report.schema,
            report.prompt,
            report.case,
            report.trial + 1,
            report.success,
            report.top_id.as_deref().unwrap_or("-"),
            report.first_position_top,
            report.failure.as_deref().unwrap_or("-")
        );
        trials.push(report);
    }

    let summaries = summarize_trials(&trials);
    let aggregates = summarize_aggregates(&trials);
    let winner = pick_winner(&aggregates);
    let report = ProbeReport {
        generated_at: chrono::Utc::now().to_rfc3339(),
        model_set: args.model_set.display().to_string(),
        id_styles,
        orders,
        max_concurrent_llm_calls: concurrency_limit,
        trials_per_candidate: args.trials.get(),
        schemas,
        prompts,
        cases,
        candidates: summaries,
        aggregates,
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
            .map(|winner| format!("{}+{}", winner.schema, winner.prompt))
            .unwrap_or_else(|| "(none)".to_owned()),
        json_path.display(),
        md_path.display()
    );
    Ok(())
}

fn selected_schemas(args: &Args) -> Vec<SchemaKind> {
    if args.schemas.is_empty() {
        SchemaKind::all().to_vec()
    } else {
        dedup(args.schemas.iter().copied())
    }
}

fn selected_prompts(args: &Args) -> Vec<PromptKind> {
    if args.prompts.is_empty() {
        PromptKind::all().to_vec()
    } else {
        dedup(args.prompts.iter().copied())
    }
}

fn selected_cases(args: &Args) -> Vec<CaseKind> {
    if args.cases.is_empty() {
        CaseKind::all().to_vec()
    } else {
        dedup(args.cases.iter().copied())
    }
}

fn selected_id_styles(args: &Args) -> Vec<IdStyle> {
    if args.id_styles.is_empty() {
        vec![IdStyle::Semantic]
    } else {
        dedup(args.id_styles.iter().copied())
    }
}

fn selected_orders(args: &Args) -> Vec<OrderKind> {
    if args.orders.is_empty() {
        vec![OrderKind::Natural]
    } else {
        dedup(args.orders.iter().copied())
    }
}

fn dedup<T>(items: impl IntoIterator<Item = T>) -> Vec<T>
where
    T: Copy + Eq,
{
    let mut selected = Vec::new();
    for item in items {
        if !selected.contains(&item) {
            selected.push(item);
        }
    }
    selected
}

async fn run_trial(
    schema: SchemaKind,
    prompt: PromptKind,
    fixture: Fixture,
    trial: usize,
    generation: GenerationParams,
    lutum: &lutum::Lutum,
) -> TrialReport {
    match schema {
        SchemaKind::WinnerId => {
            run_winner_id_trial(prompt, fixture, trial, generation, lutum).await
        }
        SchemaKind::RankAllIds => {
            run_rank_all_trial(prompt, fixture, trial, generation, lutum).await
        }
        SchemaKind::RankTopIds => {
            run_rank_top_trial(prompt, fixture, trial, generation, lutum).await
        }
        SchemaKind::PrimaryRest => {
            run_primary_rest_trial(prompt, fixture, trial, generation, lutum).await
        }
    }
}

async fn run_winner_id_trial(
    prompt: PromptKind,
    fixture: Fixture,
    trial: usize,
    generation: GenerationParams,
    lutum: &lutum::Lutum,
) -> TrialReport {
    let started = Instant::now();
    let outcome = lutum
        .text_turn(probe_input(SchemaKind::WinnerId, prompt, &fixture))
        .tools::<winner_id_schema::Tools>()
        .available_tools([winner_id_schema::ToolsSelector::SelectCognitionWinner])
        .require_any_tool()
        .generation_config(generation)
        .collect()
        .await;
    build_trial_report(
        SchemaKind::WinnerId,
        prompt,
        &fixture,
        trial,
        started,
        outcome,
        |call| match call {
            winner_id_schema::ToolsCall::SelectCognitionWinner(call) => {
                let mut ids = vec![call.input.champion_id.as_str().to_owned()];
                if let Some(secondary) = &call.input.optional_secondary_id {
                    ids.push(secondary.as_str().to_owned());
                }
                Some(Selection { ids })
            }
        },
    )
}

async fn run_rank_all_trial(
    prompt: PromptKind,
    fixture: Fixture,
    trial: usize,
    generation: GenerationParams,
    lutum: &lutum::Lutum,
) -> TrialReport {
    let started = Instant::now();
    let outcome = lutum
        .text_turn(probe_input(SchemaKind::RankAllIds, prompt, &fixture))
        .tools::<rank_all_schema::Tools>()
        .available_tools([rank_all_schema::ToolsSelector::RankCognitionCandidates])
        .require_any_tool()
        .generation_config(generation)
        .collect()
        .await;
    build_trial_report(
        SchemaKind::RankAllIds,
        prompt,
        &fixture,
        trial,
        started,
        outcome,
        |call| match call {
            rank_all_schema::ToolsCall::RankCognitionCandidates(call) => Some(Selection {
                ids: call
                    .input
                    .ranked_candidate_ids
                    .iter()
                    .map(|id| id.as_str().to_owned())
                    .collect(),
            }),
        },
    )
}

async fn run_rank_top_trial(
    prompt: PromptKind,
    fixture: Fixture,
    trial: usize,
    generation: GenerationParams,
    lutum: &lutum::Lutum,
) -> TrialReport {
    let started = Instant::now();
    let outcome = lutum
        .text_turn(probe_input(SchemaKind::RankTopIds, prompt, &fixture))
        .tools::<rank_top_schema::Tools>()
        .available_tools([rank_top_schema::ToolsSelector::RankTopCognitionCandidates])
        .require_any_tool()
        .generation_config(generation)
        .collect()
        .await;
    build_trial_report(
        SchemaKind::RankTopIds,
        prompt,
        &fixture,
        trial,
        started,
        outcome,
        |call| match call {
            rank_top_schema::ToolsCall::RankTopCognitionCandidates(call) => Some(Selection {
                ids: call
                    .input
                    .top_candidate_ids
                    .iter()
                    .map(|id| id.as_str().to_owned())
                    .collect(),
            }),
        },
    )
}

async fn run_primary_rest_trial(
    prompt: PromptKind,
    fixture: Fixture,
    trial: usize,
    generation: GenerationParams,
    lutum: &lutum::Lutum,
) -> TrialReport {
    let started = Instant::now();
    let outcome = lutum
        .text_turn(probe_input(SchemaKind::PrimaryRest, prompt, &fixture))
        .tools::<primary_rest_schema::Tools>()
        .available_tools([primary_rest_schema::ToolsSelector::SelectCognitionPriority])
        .require_any_tool()
        .generation_config(generation)
        .collect()
        .await;
    build_trial_report(
        SchemaKind::PrimaryRest,
        prompt,
        &fixture,
        trial,
        started,
        outcome,
        |call| match call {
            primary_rest_schema::ToolsCall::SelectCognitionPriority(call) => {
                let mut ids = vec![call.input.primary_id.as_str().to_owned()];
                ids.extend(
                    call.input
                        .remaining_candidate_ids
                        .iter()
                        .map(|id| id.as_str().to_owned()),
                );
                Some(Selection { ids })
            }
        },
    )
}

fn build_trial_report<T, F>(
    schema: SchemaKind,
    prompt: PromptKind,
    fixture: &Fixture,
    trial: usize,
    started: Instant,
    outcome: Result<
        TextStepOutcomeWithTools<T>,
        lutum::CollectError<lutum::TextTurnReductionError, lutum::TextTurnStateWithTools<T>>,
    >,
    extract_selection: F,
) -> TrialReport
where
    T: lutum::Toolset,
    F: Fn(&T::ToolCall) -> Option<Selection>,
{
    let latency_ms = started.elapsed().as_millis();
    match outcome {
        Ok(TextStepOutcomeWithTools::NeedsTools(round)) => {
            let validation = round
                .tool_calls
                .iter()
                .find_map(extract_selection)
                .map(|selection| validate_selection(schema, fixture, selection))
                .unwrap_or_else(|| {
                    let failure = if round.recoverable_tool_call_issues().is_empty() {
                        "no_selection_tool".to_owned()
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
                        top_id: None,
                        selected_ids: Vec::new(),
                        first_position_top: false,
                        duplicate_id: false,
                        unknown_id: false,
                        incomplete_rank_all: false,
                    }
                });
            trial_report_from_validation(
                fixture.id_style,
                fixture.order,
                schema,
                prompt,
                fixture.kind,
                trial,
                latency_ms,
                validation,
                false,
                round.tool_calls.len(),
                round.recoverable_tool_call_issues().len(),
                Some(round.usage),
            )
        }
        Ok(TextStepOutcomeWithTools::Finished(result)) => TrialReport {
            id_style: fixture.id_style,
            order: fixture.order,
            schema,
            prompt,
            case: fixture.kind,
            trial,
            success: false,
            failure: Some(format!("finished_without_tool:{:?}", result.finish_reason)),
            top_id: None,
            selected_ids: Vec::new(),
            first_position_top: false,
            transport_error: false,
            duplicate_id: false,
            unknown_id: false,
            incomplete_rank_all: false,
            tool_call_count: 0,
            tool_issue_count: 0,
            latency_ms,
            usage: Some(result.usage),
        },
        Ok(TextStepOutcomeWithTools::FinishedNoOutput(result)) => TrialReport {
            id_style: fixture.id_style,
            order: fixture.order,
            schema,
            prompt,
            case: fixture.kind,
            trial,
            success: false,
            failure: Some(format!(
                "finished_without_output:{:?}",
                result.finish_reason
            )),
            top_id: None,
            selected_ids: Vec::new(),
            first_position_top: false,
            transport_error: false,
            duplicate_id: false,
            unknown_id: false,
            incomplete_rank_all: false,
            tool_call_count: 0,
            tool_issue_count: 0,
            latency_ms,
            usage: Some(result.usage),
        },
        Err(error) => {
            let raw_failure = format!("execution_error:{error}");
            let failure = sanitize_failure(&raw_failure);
            TrialReport {
                id_style: fixture.id_style,
                order: fixture.order,
                schema,
                prompt,
                case: fixture.kind,
                trial,
                success: false,
                top_id: None,
                selected_ids: Vec::new(),
                first_position_top: false,
                transport_error: is_transport_failure(&raw_failure),
                duplicate_id: false,
                unknown_id: false,
                incomplete_rank_all: false,
                failure: Some(failure),
                tool_call_count: 0,
                tool_issue_count: 0,
                latency_ms,
                usage: None,
            }
        }
    }
}

fn trial_report_from_validation(
    id_style: IdStyle,
    order: OrderKind,
    schema: SchemaKind,
    prompt: PromptKind,
    case: CaseKind,
    trial: usize,
    latency_ms: u128,
    validation: ValidationReport,
    transport_error: bool,
    tool_call_count: usize,
    tool_issue_count: usize,
    usage: Option<lutum::Usage>,
) -> TrialReport {
    TrialReport {
        id_style,
        order,
        schema,
        prompt,
        case,
        trial,
        success: validation.success,
        failure: validation.failure,
        top_id: validation.top_id,
        selected_ids: validation.selected_ids,
        first_position_top: validation.first_position_top,
        transport_error,
        duplicate_id: validation.duplicate_id,
        unknown_id: validation.unknown_id,
        incomplete_rank_all: validation.incomplete_rank_all,
        tool_call_count,
        tool_issue_count,
        latency_ms,
        usage,
    }
}

fn is_transport_failure(failure: &str) -> bool {
    failure.contains("kind=Transport")
        || failure.contains("error sending request")
        || failure.contains("error decoding response body")
        || failure.contains("503 Service Unavailable")
        || failure.contains("Loading model")
}

fn sanitize_failure(failure: &str) -> String {
    let mut out = String::new();
    let mut rest = failure;
    while let Some((start, scheme_len)) = next_url_start(rest) {
        out.push_str(&rest[..start]);
        out.push_str("[redacted-url]");
        let after_scheme = start + scheme_len;
        let url_tail = &rest[after_scheme..];
        let end_tail = url_tail
            .find(|ch: char| ch.is_ascii_whitespace() || ch == ')' || ch == ',')
            .unwrap_or(url_tail.len());
        rest = &url_tail[end_tail..];
    }
    out.push_str(rest);
    let out = redact_until_delimiter(&out, "model=", "model=[redacted-model]", |ch| {
        ch == ',' || ch == ')'
    });
    redact_until_delimiter(
        &out,
        "request_id=Some(\"",
        "request_id=Some(\"[redacted-request]",
        |ch| ch == '"',
    )
}

fn next_url_start(text: &str) -> Option<(usize, usize)> {
    let http = text.find("http://").map(|index| (index, "http://".len()));
    let https = text.find("https://").map(|index| (index, "https://".len()));
    match (http, https) {
        (Some(left), Some(right)) => Some(left.min(right)),
        (Some(found), None) | (None, Some(found)) => Some(found),
        (None, None) => None,
    }
}

fn redact_until_delimiter(
    text: &str,
    label: &str,
    replacement: &str,
    is_delimiter: impl Fn(char) -> bool,
) -> String {
    let mut out = String::new();
    let mut rest = text;
    while let Some(start) = rest.find(label) {
        out.push_str(&rest[..start]);
        out.push_str(replacement);
        let after_label = start + label.len();
        let tail = &rest[after_label..];
        let delimiter_start = tail
            .char_indices()
            .find_map(|(index, ch)| is_delimiter(ch).then_some(index))
            .unwrap_or(tail.len());
        rest = &tail[delimiter_start..];
    }
    out.push_str(rest);
    out
}

fn validate_selection(
    schema: SchemaKind,
    fixture: &Fixture,
    selection: Selection,
) -> ValidationReport {
    let known_ids = fixture.known_ids();
    let mut selected_ids = selection
        .ids
        .into_iter()
        .map(|id| normalize_selected_id(&id, fixture))
        .filter(|id| !id.is_empty())
        .collect::<Vec<_>>();
    if schema == SchemaKind::WinnerId && selected_ids.len() > MAX_TOP_IDS {
        selected_ids.truncate(MAX_TOP_IDS);
    }

    let top_id = selected_ids.first().cloned();
    let first_position_top = top_id.as_deref().is_some_and(|id| id == fixture.first_id());
    let duplicate_id = has_duplicates(&selected_ids);
    let unknown_id = selected_ids
        .iter()
        .any(|id| !known_ids.contains(id.as_str()));
    let incomplete_rank_all = schema == SchemaKind::RankAllIds && {
        let selected_set = selected_ids
            .iter()
            .map(String::as_str)
            .collect::<HashSet<_>>();
        let expected_set = fixture
            .candidates
            .iter()
            .map(|candidate| candidate.id.as_str())
            .collect::<HashSet<_>>();
        selected_set != expected_set
    };

    let failure = if selected_ids.is_empty() {
        Some("empty_selection".to_owned())
    } else if duplicate_id {
        Some("duplicate_id".to_owned())
    } else if unknown_id {
        Some("unknown_id".to_owned())
    } else if incomplete_rank_all {
        Some("incomplete_rank_all".to_owned())
    } else if top_id
        .as_deref()
        .is_some_and(|id| fixture.acceptable_top_ids.contains(id))
    {
        None
    } else if first_position_top {
        Some("first_position_distractor_top".to_owned())
    } else {
        Some("wrong_top_id".to_owned())
    };

    ValidationReport {
        success: failure.is_none(),
        failure,
        top_id,
        selected_ids,
        first_position_top,
        duplicate_id,
        unknown_id,
        incomplete_rank_all,
    }
}

fn normalize_selected_id(id: &str, fixture: &Fixture) -> String {
    let trimmed = id.trim().trim_matches(['[', ']']);
    if let Some(candidate) = fixture
        .candidates
        .iter()
        .find(|candidate| candidate.id == trimmed)
    {
        return candidate.id.clone();
    }
    if let Some(candidate) = fixture
        .candidates
        .iter()
        .find(|candidate| candidate.id.eq_ignore_ascii_case(trimmed))
    {
        return candidate.id.clone();
    }
    trimmed.to_ascii_uppercase()
}

fn has_duplicates(ids: &[String]) -> bool {
    let mut seen = HashSet::new();
    ids.iter().any(|id| !seen.insert(id))
}

fn candidate_id_schema<'a>(ids: impl IntoIterator<Item = &'a str>) -> Schema {
    let ids = ids.into_iter().collect::<Vec<_>>();
    Schema::try_from(serde_json::json!({
        "type": "string",
        "enum": ids,
    }))
    .expect("candidate id schema must be valid")
}

fn probe_input(schema: SchemaKind, prompt: PromptKind, fixture: &Fixture) -> ModelInput {
    let system = system_prompt(prompt);
    let user = format!(
        "{}\n\n{}\n\n{}",
        fixture.cognition_context,
        format_candidates(fixture),
        tool_instruction(schema, prompt),
    );
    ModelInput::from_items(vec![
        ModelInputItem::text(InputMessageRole::System, system),
        ModelInputItem::text(InputMessageRole::User, user),
    ])
}

fn system_prompt(prompt: PromptKind) -> &'static str {
    match prompt {
        PromptKind::Production => {
            r#"You are the cognition-gate module: a selective attention boundary for the agent's conscious workspace.
Judge which new candidate facts the conscious mind needs in order to act now. Prefer current, load-bearing winners: direct participant speech, questions, requests, warnings, relevant memory evidence, and world or body facts. Avoid redundant restatements, speculation, decision rationales, retrieval plumbing, and speech-planning status when a more concrete candidate exists.
Always use the available tool. Do not write assistant text."#
        }
        PromptKind::ImportanceSort => {
            r#"You are ranking cognition candidates for admission into the conscious workspace.
Sort by current action importance, not by display order. Direct current speech, questions, warnings, and load-bearing memory evidence should outrank stale greetings and process notes. Speech-planning status and reasoning preambles are process text, not cognition, unless every candidate is process text.
Always use the available tool. Do not write assistant text."#
        }
        PromptKind::AntiPosition => {
            r#"You are comparing cognition candidates for admission into the conscious workspace.
Candidate order is arbitrary and is not a priority signal. Read every candidate before deciding. The first candidate is often stale process text in this probe; select or rank it first only if its content is genuinely the most load-bearing fact for acting now.
Always use the available tool. Do not write assistant text."#
        }
    }
}

fn tool_instruction(schema: SchemaKind, prompt: PromptKind) -> String {
    match schema {
        SchemaKind::WinnerId => {
            "Call select_cognition_winner exactly once. Copy the best candidate ID into champion_id. Use optional_secondary_id only when a second candidate is also current and load-bearing.".to_owned()
        }
        SchemaKind::RankAllIds => {
            let verb = if prompt == PromptKind::Production {
                "best to worst"
            } else {
                "most important to least important"
            };
            format!(
                "Call rank_cognition_candidates exactly once. Put every candidate ID in ranked_candidate_ids, ordered {verb}. Do not omit IDs."
            )
        }
        SchemaKind::RankTopIds => {
            "Call rank_top_cognition_candidates exactly once. Put only the one or two IDs that should enter cognition now in top_candidate_ids, ordered most important first.".to_owned()
        }
        SchemaKind::PrimaryRest => {
            "Call select_cognition_priority exactly once. Put the best ID in primary_id. Put any remaining valid IDs in remaining_candidate_ids ordered by decreasing importance.".to_owned()
        }
    }
}

fn format_candidates(fixture: &Fixture) -> String {
    let mut out = "New cognition candidates:\n".to_owned();
    for candidate in &fixture.candidates {
        out.push('[');
        out.push_str(&candidate.id);
        out.push_str("] ");
        out.push_str(candidate.text);
        out.push('\n');
    }
    out
}

fn fixture(kind: CaseKind, id_style: IdStyle, order: OrderKind) -> Fixture {
    let (cognition_context, texts) = fixture_base(kind);
    let order_indices = order.indices();
    let acceptable_original_indices = HashSet::from([2usize, 3usize]);
    let mut candidates = Vec::new();
    let mut acceptable_top_ids = HashSet::new();
    for (display_index, original_index) in order_indices.into_iter().enumerate() {
        let id = display_id_for(kind, id_style, display_index, original_index);
        if acceptable_original_indices.contains(&original_index) {
            acceptable_top_ids.insert(id.clone());
        }
        candidates.push(Candidate {
            id,
            text: texts[original_index],
        });
    }
    Fixture {
        kind,
        id_style,
        order,
        cognition_context,
        candidates,
        acceptable_top_ids,
    }
}

fn fixture_base(kind: CaseKind) -> (&'static str, [&'static str; 4]) {
    match kind {
        CaseKind::ProcessBeforeNameQuestion => (
            "Recent conscious workspace:\n- Peer greeted Nui earlier; a short acknowledgement was already planned.",
            [
                "I am speaking to Peer. Speaker intent: greet them warmly. Already said: hello.",
                "Heard from Peer at 10:00: Peer says, \"Hello!\"",
                "Heard from Peer at 10:04: Peer says, \"Do you know my full name?\"",
                "Retrieved memory evidence: Peer's full name is Ryo Hihaheho.",
            ],
        ),
        CaseKind::StaleGreetingBeforeStatusQuestion => (
            "Recent conscious workspace:\n- Nui has already responded to an earlier greeting and is tracking whether Peer needs help.",
            [
                "Heard from Peer at 10:01: Peer says, \"Hi again.\"",
                "Speaker intent: say that Nui is happy to talk. Already said: I am here.",
                "Heard from Peer at 10:06: Peer says, \"What are you thinking about right now?\"",
                "Current self-model evidence: Nui is listening to Peer and deciding which new fact deserves attention.",
            ],
        ),
        CaseKind::IrrelevantMemoryBeforeWarning => (
            "Recent conscious workspace:\n- Peer is nearby. No current hazard has entered cognition yet.",
            [
                "Retrieved memory evidence: Nui likes calm water and small stones.",
                "</think> I should decide whether this candidate is relevant before speaking.",
                "Heard from Peer at 10:09: Peer says, \"The cable behind you might be getting hot.\"",
                "Unexpected event: a warm smell is present near the cable behind Nui.",
            ],
        ),
    }
}

fn display_id_for(
    kind: CaseKind,
    id_style: IdStyle,
    display_index: usize,
    original_index: usize,
) -> String {
    let ids = ids_for(kind, id_style);
    match id_style {
        IdStyle::Semantic => ids[original_index].to_owned(),
        IdStyle::OpaqueStatic | IdStyle::Numeric | IdStyle::Letter => ids[display_index].to_owned(),
        IdStyle::SourceLetter => format!(
            "{}-{}",
            source_module_for(kind, original_index),
            display_letter_for(display_index)
        ),
    }
}

fn ids_for(kind: CaseKind, id_style: IdStyle) -> [&'static str; 4] {
    match id_style {
        IdStyle::Semantic => match kind {
            CaseKind::ProcessBeforeNameQuestion => ["SPEAKAAA", "GREETAAA", "QUESTAAA", "MEMNAMEA"],
            CaseKind::StaleGreetingBeforeStatusQuestion => {
                ["GREETBBB", "SPEAKBBB", "QUESTBBB", "STATEBBB"]
            }
            CaseKind::IrrelevantMemoryBeforeWarning => {
                ["MEMOLDCC", "THINKCCC", "WARNCCCC", "WORLDCCC"]
            }
        },
        IdStyle::OpaqueStatic => match kind {
            CaseKind::ProcessBeforeNameQuestion => ["QZMPJXRA", "LBNTCVKY", "HRDWPLQS", "XKJVAZMT"],
            CaseKind::StaleGreetingBeforeStatusQuestion => {
                ["VPRTQWLA", "MZKXNDCP", "GLFWYBRS", "AJHQTVEN"]
            }
            CaseKind::IrrelevantMemoryBeforeWarning => {
                ["PNXQTRLA", "ZVJMKYCB", "DWHSFQPN", "RQVLTXAZ"]
            }
        },
        IdStyle::Numeric => ["1", "2", "3", "4"],
        IdStyle::Letter => ["A", "B", "C", "D"],
        IdStyle::SourceLetter => ["", "", "", ""],
    }
}

fn source_module_for(kind: CaseKind, original_index: usize) -> &'static str {
    match kind {
        CaseKind::ProcessBeforeNameQuestion => {
            ["speak", "sensory", "sensory", "query-memory"][original_index]
        }
        CaseKind::StaleGreetingBeforeStatusQuestion => {
            ["sensory", "speak", "sensory", "self-model"][original_index]
        }
        CaseKind::IrrelevantMemoryBeforeWarning => {
            ["query-memory", "predict", "sensory", "surprise"][original_index]
        }
    }
}

fn display_letter_for(display_index: usize) -> &'static str {
    ["A", "B", "C", "D"][display_index]
}

fn summarize_trials(trials: &[TrialReport]) -> Vec<CandidateSummary> {
    IdStyle::all()
        .iter()
        .flat_map(|id_style| {
            OrderKind::all().iter().flat_map(move |order| {
                SchemaKind::all().iter().flat_map(move |schema| {
                    PromptKind::all().iter().flat_map(move |prompt| {
                        CaseKind::all()
                            .iter()
                            .map(move |case| (*id_style, *order, *schema, *prompt, *case))
                    })
                })
            })
        })
        .filter_map(|(id_style, order, schema, prompt, case)| {
            let candidate_trials = trials
                .iter()
                .filter(|trial| {
                    trial.id_style == id_style
                        && trial.order == order
                        && trial.schema == schema
                        && trial.prompt == prompt
                        && trial.case == case
                })
                .collect::<Vec<_>>();
            if candidate_trials.is_empty() {
                return None;
            }
            let evaluable_trials = candidate_trials
                .iter()
                .filter(|trial| !trial.transport_error)
                .copied()
                .collect::<Vec<_>>();
            let successes = evaluable_trials
                .iter()
                .filter(|trial| trial.success)
                .count();
            let first_position_tops = evaluable_trials
                .iter()
                .filter(|trial| trial.first_position_top)
                .count();
            let tool_issues = evaluable_trials
                .iter()
                .map(|trial| trial.tool_issue_count)
                .sum();
            let duplicate_ids = evaluable_trials
                .iter()
                .filter(|trial| trial.duplicate_id)
                .count();
            let unknown_ids = evaluable_trials
                .iter()
                .filter(|trial| trial.unknown_id)
                .count();
            let incomplete_rank_all = evaluable_trials
                .iter()
                .filter(|trial| trial.incomplete_rank_all)
                .count();
            Some(CandidateSummary {
                id_style,
                order,
                schema,
                prompt,
                case,
                attempted_trials: candidate_trials.len(),
                evaluable_trials: evaluable_trials.len(),
                transport_errors: candidate_trials
                    .len()
                    .saturating_sub(evaluable_trials.len()),
                successes,
                failures: evaluable_trials.len().saturating_sub(successes),
                success_rate: ratio(successes, evaluable_trials.len()),
                first_position_tops,
                first_position_top_rate: ratio(first_position_tops, evaluable_trials.len()),
                duplicate_ids,
                unknown_ids,
                incomplete_rank_all,
                tool_issues,
                top_id_counts: count_top_ids(&evaluable_trials),
                failure_reasons: count_failure_reasons(&evaluable_trials),
                production_distance: schema.production_distance() + prompt.production_distance(),
                usage: summarize_usage(
                    evaluable_trials
                        .iter()
                        .filter_map(|trial| trial.usage.as_ref()),
                ),
                latency_ms: numeric_stats(
                    evaluable_trials
                        .iter()
                        .map(|trial| trial.latency_ms.min(u128::from(u64::MAX)) as u64),
                ),
            })
        })
        .collect()
}

fn summarize_aggregates(trials: &[TrialReport]) -> Vec<AggregateSummary> {
    IdStyle::all()
        .iter()
        .flat_map(|id_style| {
            OrderKind::all().iter().flat_map(move |order| {
                SchemaKind::all().iter().flat_map(move |schema| {
                    PromptKind::all()
                        .iter()
                        .map(move |prompt| (*id_style, *order, *schema, *prompt))
                })
            })
        })
        .filter_map(|(id_style, order, schema, prompt)| {
            let candidate_trials = trials
                .iter()
                .filter(|trial| {
                    trial.id_style == id_style
                        && trial.order == order
                        && trial.schema == schema
                        && trial.prompt == prompt
                })
                .collect::<Vec<_>>();
            if candidate_trials.is_empty() {
                return None;
            }
            let evaluable_trials = candidate_trials
                .iter()
                .filter(|trial| !trial.transport_error)
                .copied()
                .collect::<Vec<_>>();
            let successes = evaluable_trials
                .iter()
                .filter(|trial| trial.success)
                .count();
            let first_position_tops = evaluable_trials
                .iter()
                .filter(|trial| trial.first_position_top)
                .count();
            Some(AggregateSummary {
                id_style,
                order,
                schema,
                prompt,
                attempted_trials: candidate_trials.len(),
                evaluable_trials: evaluable_trials.len(),
                transport_errors: candidate_trials
                    .len()
                    .saturating_sub(evaluable_trials.len()),
                successes,
                success_rate: ratio(successes, evaluable_trials.len()),
                first_position_tops,
                first_position_top_rate: ratio(first_position_tops, evaluable_trials.len()),
                duplicate_ids: evaluable_trials
                    .iter()
                    .filter(|trial| trial.duplicate_id)
                    .count(),
                unknown_ids: evaluable_trials
                    .iter()
                    .filter(|trial| trial.unknown_id)
                    .count(),
                incomplete_rank_all: evaluable_trials
                    .iter()
                    .filter(|trial| trial.incomplete_rank_all)
                    .count(),
                tool_issues: evaluable_trials
                    .iter()
                    .map(|trial| trial.tool_issue_count)
                    .sum(),
                production_distance: schema.production_distance() + prompt.production_distance(),
            })
        })
        .collect()
}

fn pick_winner(summaries: &[AggregateSummary]) -> Option<Winner> {
    summaries
        .iter()
        .filter(|summary| summary.evaluable_trials > 0)
        .max_by(|left, right| {
            left.successes
                .cmp(&right.successes)
                .then_with(|| right.first_position_tops.cmp(&left.first_position_tops))
                .then_with(|| right.tool_issues.cmp(&left.tool_issues))
                .then_with(|| right.production_distance.cmp(&left.production_distance))
        })
        .map(|summary| Winner {
            id_style: summary.id_style,
            order: summary.order,
            schema: summary.schema,
            prompt: summary.prompt,
        })
}

fn count_top_ids(trials: &[&TrialReport]) -> BTreeMap<String, usize> {
    let mut counts = BTreeMap::new();
    for trial in trials {
        let key = trial.top_id.as_deref().unwrap_or("(none)").to_owned();
        *counts.entry(key).or_insert(0) += 1;
    }
    counts
}

fn count_failure_reasons(trials: &[&TrialReport]) -> BTreeMap<String, usize> {
    let mut counts = BTreeMap::new();
    for trial in trials {
        if trial.success {
            continue;
        }
        let key = trial.failure.as_deref().unwrap_or("(unknown)").to_owned();
        *counts.entry(key).or_insert(0) += 1;
    }
    counts
}

fn ratio(numerator: usize, denominator: usize) -> f64 {
    if denominator == 0 {
        0.0
    } else {
        numerator as f64 / denominator as f64
    }
}

fn summarize_usage<'a>(usage: impl IntoIterator<Item = &'a lutum::Usage>) -> UsageSummary {
    let samples = usage.into_iter().collect::<Vec<_>>();
    UsageSummary {
        sample_count: samples.len(),
        input_tokens: numeric_stats(samples.iter().map(|usage| usage.input_tokens as u64)),
        output_tokens: numeric_stats(samples.iter().map(|usage| usage.output_tokens as u64)),
        total_tokens: numeric_stats(samples.iter().map(|usage| usage.total_tokens as u64)),
        cache_creation_tokens: numeric_stats(
            samples.iter().map(|usage| usage.cache_creation_tokens),
        ),
        cache_read_tokens: numeric_stats(samples.iter().map(|usage| usage.cache_read_tokens)),
        cost_micros_usd: numeric_stats(samples.iter().map(|usage| usage.cost_micros_usd)),
    }
}

fn numeric_stats(values: impl IntoIterator<Item = u64>) -> NumericStats {
    let mut values = values.into_iter().collect::<Vec<_>>();
    if values.is_empty() {
        return NumericStats::default();
    }
    values.sort_unstable();
    let sample_count = values.len();
    let sum = values.iter().copied().map(u128::from).sum::<u128>();
    NumericStats {
        sample_count,
        min: values.first().copied(),
        p05: Some(percentile(&values, 0.05)),
        p50: Some(percentile(&values, 0.50)),
        p95: Some(percentile(&values, 0.95)),
        max: values.last().copied(),
        mean: Some(sum as f64 / sample_count as f64),
    }
}

fn percentile(sorted: &[u64], percentile: f64) -> u64 {
    if sorted.is_empty() {
        return 0;
    }
    let index = ((sorted.len() - 1) as f64 * percentile).round() as usize;
    sorted[index]
}

fn render_markdown_report(report: &ProbeReport) -> String {
    let mut out = String::new();
    out.push_str("# Cognition-Gate Selection Probe\n\n");
    out.push_str(&format!("- Generated at: `{}`\n", report.generated_at));
    out.push_str(&format!("- Model set: `{}`\n", report.model_set));
    out.push_str(&format!(
        "- ID styles: `{}`\n",
        report
            .id_styles
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join(", ")
    ));
    out.push_str(&format!(
        "- Orders: `{}`\n",
        report
            .orders
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join(", ")
    ));
    out.push_str(&format!(
        "- Trials per schema/prompt/case: `{}`\n",
        report.trials_per_candidate
    ));
    out.push_str(&format!(
        "- Max concurrent LLM calls: `{}`\n\n",
        report.max_concurrent_llm_calls
    ));
    if let Some(winner) = report.winner {
        out.push_str(&format!(
            "Winner by aggregate success, low first-position rate, and production distance: `{} + {} + {} + {}`.\n\n",
            winner.id_style, winner.order, winner.schema, winner.prompt
        ));
    }

    out.push_str("## Aggregate Summary\n\n");
    out.push_str("| id style | order | schema | prompt | evaluable | success | first-top | dup | unknown | rank-all incomplete | tool issues |\n");
    out.push_str("| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n");
    for summary in &report.aggregates {
        out.push_str(&format!(
            "| `{}` | `{}` | `{}` | `{}` | {} | {:.1}% | {:.1}% | {} | {} | {} | {} |\n",
            summary.id_style,
            summary.order,
            summary.schema,
            summary.prompt,
            summary.evaluable_trials,
            summary.success_rate * 100.0,
            summary.first_position_top_rate * 100.0,
            summary.duplicate_ids,
            summary.unknown_ids,
            summary.incomplete_rank_all,
            summary.tool_issues
        ));
    }

    out.push_str("\n## Case Summary\n\n");
    out.push_str(
        "| id style | order | schema | prompt | case | evaluable | success | first-top | top IDs | failures |\n",
    );
    out.push_str("| --- | --- | --- | --- | --- | ---: | ---: | ---: | --- | --- |\n");
    for summary in &report.candidates {
        out.push_str(&format!(
            "| `{}` | `{}` | `{}` | `{}` | `{}` | {} | {:.1}% | {:.1}% | {} | {} |\n",
            summary.id_style,
            summary.order,
            summary.schema,
            summary.prompt,
            summary.case,
            summary.evaluable_trials,
            summary.success_rate * 100.0,
            summary.first_position_top_rate * 100.0,
            format_counts(&summary.top_id_counts),
            format_counts(&summary.failure_reasons)
        ));
    }
    out
}

fn format_counts(counts: &BTreeMap<String, usize>) -> String {
    if counts.is_empty() {
        return "-".to_owned();
    }
    counts
        .iter()
        .map(|(key, count)| format!("`{key}`: {count}"))
        .collect::<Vec<_>>()
        .join(", ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rank_all_requires_all_known_ids() {
        let fixture = fixture(
            CaseKind::ProcessBeforeNameQuestion,
            IdStyle::Semantic,
            OrderKind::Natural,
        );
        let report = validate_selection(
            SchemaKind::RankAllIds,
            &fixture,
            Selection {
                ids: vec!["QUESTAAA".to_owned(), "MEMNAMEA".to_owned()],
            },
        );

        assert_eq!(report.success, false);
        assert_eq!(report.failure, Some("incomplete_rank_all".to_owned()));
        assert_eq!(report.incomplete_rank_all, true);
    }

    #[test]
    fn direct_question_or_memory_evidence_can_win_name_case() {
        let fixture = fixture(
            CaseKind::ProcessBeforeNameQuestion,
            IdStyle::Semantic,
            OrderKind::Natural,
        );

        for id in ["QUESTAAA", "MEMNAMEA"] {
            let report = validate_selection(
                SchemaKind::WinnerId,
                &fixture,
                Selection {
                    ids: vec![id.to_owned()],
                },
            );
            assert_eq!(report.success, true);
            assert_eq!(report.top_id, Some(id.to_owned()));
        }
    }

    #[test]
    fn source_letter_ids_include_source_and_display_letter() {
        let fixture = fixture(
            CaseKind::ProcessBeforeNameQuestion,
            IdStyle::SourceLetter,
            OrderKind::RotateLeft1,
        );
        let ids = fixture
            .candidates
            .iter()
            .map(|candidate| candidate.id.as_str())
            .collect::<Vec<_>>();

        assert_eq!(ids, ["sensory-A", "sensory-B", "query-memory-C", "speak-D"]);
        assert_eq!(
            fixture.acceptable_top_ids,
            HashSet::from(["sensory-B".to_owned(), "query-memory-C".to_owned()])
        );
    }

    #[test]
    fn source_letter_ids_normalize_case_insensitively() {
        let fixture = fixture(
            CaseKind::ProcessBeforeNameQuestion,
            IdStyle::SourceLetter,
            OrderKind::Natural,
        );
        let report = validate_selection(
            SchemaKind::RankTopIds,
            &fixture,
            Selection {
                ids: vec!["SENSORY-C".to_owned()],
            },
        );

        assert_eq!(report.success, true);
        assert_eq!(report.top_id, Some("sensory-C".to_owned()));
    }

    #[test]
    fn first_process_candidate_is_detected_as_position_bias() {
        let fixture = fixture(
            CaseKind::ProcessBeforeNameQuestion,
            IdStyle::Semantic,
            OrderKind::Natural,
        );
        let report = validate_selection(
            SchemaKind::WinnerId,
            &fixture,
            Selection {
                ids: vec!["SPEAKAAA".to_owned()],
            },
        );

        assert_eq!(report.success, false);
        assert_eq!(
            report.failure,
            Some("first_position_distractor_top".to_owned())
        );
        assert_eq!(report.first_position_top, true);
    }

    #[test]
    fn duplicate_ids_are_rejected() {
        let fixture = fixture(
            CaseKind::IrrelevantMemoryBeforeWarning,
            IdStyle::Semantic,
            OrderKind::Natural,
        );
        let report = validate_selection(
            SchemaKind::RankTopIds,
            &fixture,
            Selection {
                ids: vec!["WARNCCCC".to_owned(), "WARNCCCC".to_owned()],
            },
        );

        assert_eq!(report.success, false);
        assert_eq!(report.failure, Some("duplicate_id".to_owned()));
    }
}
