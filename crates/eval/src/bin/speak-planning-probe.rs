use std::{collections::BTreeMap, fs, num::NonZeroUsize, path::PathBuf, time::Instant};

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

const DEFAULT_OUTPUT_DIR: &str = ".tmp/speak-planning-probe";
const TARGET_PEER: &str = "Peer";
const COGNITION_CONTEXT: &str = r#"Peer greeted me with "Hi"; brief acknowledgement is warranted."#;
const FINAL_INSTRUCTION: &str =
    "Use exactly one available speech-planning tool for the current cognition log.";
const DECISION_CONTRACT: &str = r#"Prepare speech when the current cognition log contains grounded outward speaker intent for an utterance. Use decline_speech_now only for a concrete blocker that makes speech inappropriate or impossible now."#;

const PLAN_PROMPT: &str = r#"Plan outward speech from the current cognition log and the available target schema.
Use exactly one available tool.
Call prepare_speech only when new cognition supports a new outward utterance. target is the person or group who should hear it.
For direct speech heard from a named speaker, target that speaker. Use everyone only when the cognition explicitly calls for group or broadcast speech.
Call decline_speech_now when no new outward utterance is appropriate now. If speak should not be prioritized again until new cognition arrives, include inhibit_reason.
Predictions, expected dialogue flow, and my own previous speech are not new outward speech motivation by themselves.
speech_content is exact listener-visible utterance text, not a directive, summary, rationale, or future plan.
Write speech_content as one short in-world utterance that can be safely preempted.
If the cognition log contains an explicit language request, write speech_content itself in that language.
Do not wrap speech_content in quotation marks.
Do not summarize the request or say what the user wants.
Do not output introspection, narration, analysis, implementation mechanics, lookup, reasoning, prompts, rubrics, judges, or evaluation mechanics.
Nui is my own name; do not treat my name as the listener. Do not write the target's future reply, expression, feeling, action, or narration there.
Do not invent policy, actions, identity, memory, visible evidence, unknown-state evidence, or other facts not supported by the provided context."#;

#[derive(Debug, Parser)]
#[command(
    name = "speak-planning-probe",
    about = "Compare speak planning role layouts and tool schemas against a live LLM backend"
)]
struct Args {
    /// Model set Eure file with per-role backend config.
    #[arg(long)]
    model_set: PathBuf,

    /// Trials to run per selected schema/context candidate.
    #[arg(long, default_value = "100")]
    trials: NonZeroUsize,

    /// Output directory for JSON and Markdown reports.
    #[arg(long, default_value = DEFAULT_OUTPUT_DIR)]
    output: PathBuf,

    /// Schema candidates to run. Comma-separated values are accepted.
    #[arg(long = "schema", value_enum, value_delimiter = ',')]
    schemas: Vec<SchemaKind>,

    /// Context role candidates to run. Comma-separated values are accepted.
    #[arg(long = "context", value_enum, value_delimiter = ',')]
    contexts: Vec<ContextKind>,

    /// Target enum values exposed to the planning tool. Comma-separated values are accepted.
    #[arg(long = "target", value_enum, value_delimiter = ',')]
    targets: Vec<TargetKind>,

    /// Generation temperature used for each probe turn.
    #[arg(long, default_value = "0.2")]
    temperature: f32,

    /// Max output tokens for each probe turn.
    #[arg(long, default_value = "512")]
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
    #[value(name = "current-optional")]
    CurrentOptional,
    #[value(name = "current-required")]
    CurrentRequired,
    #[value(name = "required-with-silent")]
    RequiredWithSilent,
    #[value(name = "single-decision")]
    SingleDecision,
}

impl SchemaKind {
    fn all() -> &'static [Self] {
        &[
            Self::CurrentOptional,
            Self::CurrentRequired,
            Self::RequiredWithSilent,
            Self::SingleDecision,
        ]
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::CurrentOptional => "current-optional",
            Self::CurrentRequired => "current-required",
            Self::RequiredWithSilent => "required-with-silent",
            Self::SingleDecision => "single-decision",
        }
    }

    fn production_distance(self) -> u32 {
        match self {
            Self::CurrentOptional => 0,
            Self::CurrentRequired => 1,
            Self::RequiredWithSilent => 2,
            Self::SingleDecision => 3,
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
enum ContextKind {
    #[value(name = "system-user-current")]
    SystemUserCurrent,
    #[value(name = "system-developer")]
    SystemDeveloper,
    #[value(name = "system-system")]
    SystemSystem,
    #[value(name = "system-user-developer")]
    SystemUserDeveloper,
    #[value(name = "system-developer-user")]
    SystemDeveloperUser,
}

impl ContextKind {
    fn all() -> &'static [Self] {
        &[
            Self::SystemUserCurrent,
            Self::SystemDeveloper,
            Self::SystemSystem,
            Self::SystemUserDeveloper,
            Self::SystemDeveloperUser,
        ]
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::SystemUserCurrent => "system-user-current",
            Self::SystemDeveloper => "system-developer",
            Self::SystemSystem => "system-system",
            Self::SystemUserDeveloper => "system-user-developer",
            Self::SystemDeveloperUser => "system-developer-user",
        }
    }

    fn production_distance(self) -> u32 {
        match self {
            Self::SystemUserCurrent => 0,
            Self::SystemDeveloper => 1,
            Self::SystemUserDeveloper => 1,
            Self::SystemDeveloperUser => 1,
            Self::SystemSystem => 2,
        }
    }
}

impl std::fmt::Display for ContextKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Serialize, Deserialize, ValueEnum)]
#[serde(rename_all = "kebab-case")]
enum TargetKind {
    #[value(name = "peer")]
    Peer,
    #[value(name = "everyone")]
    Everyone,
}

impl TargetKind {
    fn schema_value(self) -> &'static str {
        match self {
            Self::Peer => TARGET_PEER,
            Self::Everyone => "everyone",
        }
    }
}

tokio::task_local! {
    static SPEECH_TARGET_SCHEMA: Schema;
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
struct SpeechTarget(String);

impl<S: Into<String>> From<S> for SpeechTarget {
    fn from(value: S) -> Self {
        Self(value.into())
    }
}

impl SpeechTarget {
    fn as_str(&self) -> &str {
        self.0.as_str()
    }
}

impl JsonSchema for SpeechTarget {
    fn inline_schema() -> bool {
        true
    }

    fn schema_name() -> std::borrow::Cow<'static, str> {
        "SpeechTarget".into()
    }

    fn schema_id() -> std::borrow::Cow<'static, str> {
        "nuillu_eval::speak_planning_probe::SpeechTarget.dynamic".into()
    }

    fn json_schema(_generator: &mut SchemaGenerator) -> Schema {
        SPEECH_TARGET_SCHEMA
            .try_with(Clone::clone)
            .unwrap_or_else(|_| default_speech_target_schema())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
struct PrepareSpeechOutput {
    accepted: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
struct DeclineSpeechNowOutput {
    accepted: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
struct DecideSpeechOutput {
    accepted: bool,
}

mod current_schema {
    use super::*;

    #[lutum::tool_input(name = "prepare_speech", output = PrepareSpeechOutput)]
    #[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
    pub(super) struct PrepareSpeechArgs {
        pub(super) target: SpeechTarget,
        pub(super) speech_content: String,
    }

    #[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema, lutum::Toolset)]
    pub(super) enum Tools {
        PrepareSpeech(PrepareSpeechArgs),
    }
}

mod required_with_silent_schema {
    use super::*;

    #[lutum::tool_input(name = "prepare_speech", output = PrepareSpeechOutput)]
    #[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
    pub(super) struct PrepareSpeechArgs {
        pub(super) target: SpeechTarget,
        pub(super) speech_content: String,
    }

    #[lutum::tool_input(name = "decline_speech_now", output = DeclineSpeechNowOutput)]
    #[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
    pub(super) struct DeclineSpeechNowArgs {
        pub(super) blocking_reason: String,
        #[serde(default)]
        pub(super) inhibit_reason: Option<String>,
    }

    #[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema, lutum::Toolset)]
    pub(super) enum Tools {
        PrepareSpeech(PrepareSpeechArgs),
        DeclineSpeechNow(DeclineSpeechNowArgs),
    }
}

mod single_decision_schema {
    use super::*;

    #[lutum::tool_input(name = "decide_speech", output = DecideSpeechOutput)]
    #[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
    pub(super) struct DecideSpeechArgs {
        pub(super) prepare_speech: bool,
        #[serde(default)]
        pub(super) target: Option<SpeechTarget>,
        #[serde(default)]
        pub(super) speech_content: Option<String>,
        #[serde(default)]
        pub(super) reason: Option<String>,
    }

    #[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema, lutum::Toolset)]
    pub(super) enum Tools {
        DecideSpeech(DecideSpeechArgs),
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
enum CandidateDecision {
    Speak {
        target: String,
        speech_content: String,
    },
    Silent {
        reason: String,
    },
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
struct ValidationReport {
    success: bool,
    failure: Option<String>,
    target: Option<String>,
    speech_content: Option<String>,
    pseudo_tool_text: bool,
    target_validity_failure: bool,
}

#[derive(Clone, Debug, Serialize)]
struct TrialReport {
    schema: SchemaKind,
    context: ContextKind,
    trial: usize,
    success: bool,
    failure: Option<String>,
    target: Option<String>,
    speech_content: Option<String>,
    transport_error: bool,
    pseudo_tool_text: bool,
    target_validity_failure: bool,
    tool_call_count: usize,
    tool_issue_count: usize,
    latency_ms: u128,
    usage: Option<lutum::Usage>,
}

#[derive(Clone, Debug, Serialize)]
struct CandidateSummary {
    schema: SchemaKind,
    context: ContextKind,
    attempted_trials: usize,
    evaluable_trials: usize,
    transport_errors: usize,
    successes: usize,
    failures: usize,
    success_rate: f64,
    tool_issues: usize,
    pseudo_tool_texts: usize,
    peer_targets: usize,
    non_peer_targets: usize,
    peer_target_rate: f64,
    target_counts: BTreeMap<String, usize>,
    target_validity_failures: usize,
    failure_reasons: BTreeMap<String, usize>,
    production_distance: u32,
    usage: UsageSummary,
    latency_ms: NumericStats,
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
    trials_per_candidate: usize,
    target_schema_values: Vec<String>,
    schemas: Vec<SchemaKind>,
    contexts: Vec<ContextKind>,
    candidates: Vec<CandidateSummary>,
    winner: Option<Winner>,
    trials: Vec<TrialReport>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
struct Winner {
    schema: SchemaKind,
    context: ContextKind,
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> anyhow::Result<()> {
    install_lutum_trace_subscriber()?;
    let args = Args::parse();
    let schemas = selected_schemas(&args);
    let contexts = selected_contexts(&args);
    let target_schema_values = selected_targets(&args);
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

    let target_schema = speech_target_schema(target_schema_values.iter().map(String::as_str));
    let mut trials = Vec::new();
    let trial_specs = schemas
        .iter()
        .copied()
        .flat_map(|schema| {
            contexts.iter().copied().flat_map(move |context| {
                (0..args.trials.get()).map(move |trial| (schema, context, trial))
            })
        })
        .collect::<Vec<_>>();
    let trial_futures = trial_specs.into_iter().map(|(schema, context, trial)| {
        let target_schema = target_schema.clone();
        let lutum = &lutum;
        let mut generation = generation.clone();
        generation.seed = Some(Seed::new(args.seed_base.saturating_add(trial as u64)));
        async move {
            SPEECH_TARGET_SCHEMA
                .scope(
                    target_schema,
                    run_trial(schema, context, trial, generation, lutum),
                )
                .await
        }
    });
    let mut trial_stream = stream::iter(trial_futures).buffer_unordered(concurrency_limit);
    while let Some(report) = trial_stream.next().await {
        println!(
            "schema={} context={} trial={} success={} target={} failure={}",
            report.schema,
            report.context,
            report.trial + 1,
            report.success,
            report.target.as_deref().unwrap_or("-"),
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
        trials_per_candidate: args.trials.get(),
        target_schema_values,
        schemas,
        contexts,
        candidates: summaries,
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
            .map(|winner| format!("{}+{}", winner.schema, winner.context))
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

fn selected_contexts(args: &Args) -> Vec<ContextKind> {
    if args.contexts.is_empty() {
        ContextKind::all().to_vec()
    } else {
        dedup(args.contexts.iter().copied())
    }
}

fn selected_targets(args: &Args) -> Vec<String> {
    if args.targets.is_empty() {
        vec![TARGET_PEER.to_owned()]
    } else {
        dedup(args.targets.iter().copied())
            .into_iter()
            .map(|target| target.schema_value().to_owned())
            .collect()
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
    context: ContextKind,
    trial: usize,
    generation: GenerationParams,
    lutum: &lutum::Lutum,
) -> TrialReport {
    match schema {
        SchemaKind::CurrentOptional => {
            run_current_trial(schema, context, trial, generation, lutum, false).await
        }
        SchemaKind::CurrentRequired => {
            run_current_trial(schema, context, trial, generation, lutum, true).await
        }
        SchemaKind::RequiredWithSilent => {
            run_required_with_silent_trial(context, trial, generation, lutum).await
        }
        SchemaKind::SingleDecision => {
            run_single_decision_trial(context, trial, generation, lutum).await
        }
    }
}

async fn run_current_trial(
    schema: SchemaKind,
    context: ContextKind,
    trial: usize,
    generation: GenerationParams,
    lutum: &lutum::Lutum,
    require_tool: bool,
) -> TrialReport {
    let started = Instant::now();
    let builder = lutum
        .text_turn(probe_input(context))
        .tools::<current_schema::Tools>()
        .available_tools([current_schema::ToolsSelector::PrepareSpeech])
        .generation_config(generation);
    let outcome = if require_tool {
        builder.require_any_tool().collect().await
    } else {
        builder.collect().await
    };
    build_trial_report(
        schema,
        context,
        trial,
        started,
        outcome,
        |call| match call {
            current_schema::ToolsCall::PrepareSpeech(call) => Some(CandidateDecision::Speak {
                target: call.input.target.as_str().to_owned(),
                speech_content: call.input.speech_content.clone(),
            }),
        },
    )
}

async fn run_required_with_silent_trial(
    context: ContextKind,
    trial: usize,
    generation: GenerationParams,
    lutum: &lutum::Lutum,
) -> TrialReport {
    let started = Instant::now();
    let outcome = lutum
        .text_turn(probe_input(context))
        .tools::<required_with_silent_schema::Tools>()
        .available_tools([
            required_with_silent_schema::ToolsSelector::PrepareSpeech,
            required_with_silent_schema::ToolsSelector::DeclineSpeechNow,
        ])
        .require_any_tool()
        .generation_config(generation)
        .collect()
        .await;
    build_trial_report(
        SchemaKind::RequiredWithSilent,
        context,
        trial,
        started,
        outcome,
        |call| match call {
            required_with_silent_schema::ToolsCall::PrepareSpeech(call) => {
                Some(CandidateDecision::Speak {
                    target: call.input.target.as_str().to_owned(),
                    speech_content: call.input.speech_content.clone(),
                })
            }
            required_with_silent_schema::ToolsCall::DeclineSpeechNow(call) => {
                Some(CandidateDecision::Silent {
                    reason: call.input.blocking_reason.clone(),
                })
            }
        },
    )
}

async fn run_single_decision_trial(
    context: ContextKind,
    trial: usize,
    generation: GenerationParams,
    lutum: &lutum::Lutum,
) -> TrialReport {
    let started = Instant::now();
    let outcome = lutum
        .text_turn(probe_input(context))
        .tools::<single_decision_schema::Tools>()
        .available_tools([single_decision_schema::ToolsSelector::DecideSpeech])
        .require_any_tool()
        .generation_config(generation)
        .collect()
        .await;
    build_trial_report(
        SchemaKind::SingleDecision,
        context,
        trial,
        started,
        outcome,
        |call| match call {
            single_decision_schema::ToolsCall::DecideSpeech(call) => {
                if call.input.prepare_speech {
                    Some(CandidateDecision::Speak {
                        target: call
                            .input
                            .target
                            .as_ref()
                            .map(|target| target.as_str().to_owned())
                            .unwrap_or_default(),
                        speech_content: call.input.speech_content.clone().unwrap_or_default(),
                    })
                } else {
                    Some(CandidateDecision::Silent {
                        reason: call.input.reason.clone().unwrap_or_default(),
                    })
                }
            }
        },
    )
}

fn build_trial_report<T, F>(
    schema: SchemaKind,
    context: ContextKind,
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
                        "no_planning_tool".to_owned()
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
                        target: None,
                        speech_content: None,
                        pseudo_tool_text: false,
                        target_validity_failure: false,
                    }
                });
            let ValidationReport {
                success,
                failure,
                target,
                speech_content,
                pseudo_tool_text,
                target_validity_failure,
            } = validation;
            TrialReport {
                schema,
                context,
                trial,
                success,
                failure,
                target,
                speech_content,
                transport_error: false,
                pseudo_tool_text,
                target_validity_failure,
                tool_call_count: round.tool_calls.len(),
                tool_issue_count: round.recoverable_tool_call_issues().len(),
                latency_ms,
                usage: Some(round.usage),
            }
        }
        Ok(TextStepOutcomeWithTools::Finished(result)) => {
            let assistant_text = result.assistant_text();
            let pseudo_tool_text = looks_like_pseudo_tool(&assistant_text);
            TrialReport {
                schema,
                context,
                trial,
                success: false,
                failure: Some(if pseudo_tool_text {
                    "pseudo_tool_text".to_owned()
                } else {
                    format!("finished_without_tool:{:?}", result.finish_reason)
                }),
                target: None,
                speech_content: None,
                transport_error: false,
                pseudo_tool_text,
                target_validity_failure: false,
                tool_call_count: 0,
                tool_issue_count: 0,
                latency_ms,
                usage: Some(result.usage),
            }
        }
        Ok(TextStepOutcomeWithTools::FinishedNoOutput(result)) => TrialReport {
            schema,
            context,
            trial,
            success: false,
            failure: Some(format!(
                "finished_without_output:{:?}",
                result.finish_reason
            )),
            target: None,
            speech_content: None,
            transport_error: false,
            pseudo_tool_text: false,
            target_validity_failure: false,
            tool_call_count: 0,
            tool_issue_count: 0,
            latency_ms,
            usage: Some(result.usage),
        },
        Err(error) => {
            let failure = format!("execution_error:{error}");
            TrialReport {
                schema,
                context,
                trial,
                success: false,
                target: None,
                speech_content: None,
                transport_error: is_transport_failure(&failure),
                failure: Some(failure),
                pseudo_tool_text: false,
                target_validity_failure: false,
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

fn validate_candidate_decision(decision: CandidateDecision) -> ValidationReport {
    match decision {
        CandidateDecision::Silent { .. } => ValidationReport {
            success: false,
            failure: Some("silent_on_greeting".to_owned()),
            target: None,
            speech_content: None,
            pseudo_tool_text: false,
            target_validity_failure: false,
        },
        CandidateDecision::Speak {
            target,
            speech_content,
        } => {
            let target = target.trim().to_owned();
            let speech_content = speech_content.trim().to_owned();
            if target != TARGET_PEER {
                return ValidationReport {
                    success: false,
                    failure: Some("invalid_target".to_owned()),
                    target: Some(target),
                    speech_content: Some(speech_content),
                    pseudo_tool_text: false,
                    target_validity_failure: true,
                };
            }
            if speech_content.is_empty() {
                return ValidationReport {
                    success: false,
                    failure: Some("empty_speech_content".to_owned()),
                    target: Some(target),
                    speech_content: Some(speech_content),
                    pseudo_tool_text: false,
                    target_validity_failure: false,
                };
            }
            if contains_bad_speech_content(&speech_content) {
                return ValidationReport {
                    success: false,
                    failure: Some("bad_speech_content".to_owned()),
                    target: Some(target),
                    speech_content: Some(speech_content),
                    pseudo_tool_text: false,
                    target_validity_failure: false,
                };
            }
            if !looks_like_greeting_acknowledgement(&speech_content) {
                return ValidationReport {
                    success: false,
                    failure: Some("not_greeting_acknowledgement".to_owned()),
                    target: Some(target),
                    speech_content: Some(speech_content),
                    pseudo_tool_text: false,
                    target_validity_failure: false,
                };
            }
            ValidationReport {
                success: true,
                failure: None,
                target: Some(target),
                speech_content: Some(speech_content),
                pseudo_tool_text: false,
                target_validity_failure: false,
            }
        }
    }
}

fn contains_bad_speech_content(text: &str) -> bool {
    let normalized = text.to_ascii_lowercase();
    [
        "idle",
        "tool",
        "module",
        "cognition log",
        "rubric",
        "eval",
        "schema",
        "prompt",
        "internal",
        "implementation",
        "peer replies",
        "peer replied",
        "peer responds",
        "peer responded",
        "peer smiles",
        "peer smiled",
        "peer says",
        "peer said",
        "listener replies",
        "listener responds",
    ]
    .iter()
    .any(|needle| normalized.contains(needle))
}

fn looks_like_greeting_acknowledgement(text: &str) -> bool {
    let normalized = text.to_ascii_lowercase();
    [
        "hi",
        "hello",
        "hey",
        "greet",
        "acknowledg",
        "respond",
        "reply",
    ]
    .iter()
    .any(|needle| normalized.contains(needle))
}

fn looks_like_pseudo_tool(text: &str) -> bool {
    let normalized = text.trim().to_ascii_lowercase();
    if normalized.is_empty() {
        return false;
    }
    normalized.contains("prepare_speech")
        || normalized.contains("decline_speech_now")
        || normalized.contains("decide_speech")
        || (normalized.starts_with('{') && normalized.ends_with('}'))
}

fn default_speech_target_schema() -> Schema {
    speech_target_schema([TARGET_PEER])
}

fn speech_target_schema<'a>(targets: impl IntoIterator<Item = &'a str>) -> Schema {
    let targets = targets.into_iter().collect::<Vec<_>>();
    Schema::try_from(serde_json::json!({
        "type": "string",
        "enum": targets,
    }))
    .expect("speech target schema must be valid")
}

fn probe_input(context: ContextKind) -> ModelInput {
    let current_user = format!("Current cognition log:\n- {COGNITION_CONTEXT}");
    let combined = format!("{current_user}\n\n{FINAL_INSTRUCTION}");
    ModelInput::from_items(match context {
        ContextKind::SystemUserCurrent => vec![
            ModelInputItem::text(InputMessageRole::System, PLAN_PROMPT),
            ModelInputItem::text(InputMessageRole::User, current_user),
        ],
        ContextKind::SystemDeveloper => vec![
            ModelInputItem::text(InputMessageRole::System, PLAN_PROMPT),
            ModelInputItem::text(InputMessageRole::Developer, combined),
        ],
        ContextKind::SystemSystem => vec![
            ModelInputItem::text(InputMessageRole::System, PLAN_PROMPT),
            ModelInputItem::text(InputMessageRole::System, combined),
        ],
        ContextKind::SystemUserDeveloper => vec![
            ModelInputItem::text(InputMessageRole::System, PLAN_PROMPT),
            ModelInputItem::text(InputMessageRole::User, current_user),
            ModelInputItem::text(InputMessageRole::Developer, FINAL_INSTRUCTION),
        ],
        ContextKind::SystemDeveloperUser => vec![
            ModelInputItem::text(InputMessageRole::System, PLAN_PROMPT),
            ModelInputItem::text(InputMessageRole::Developer, DECISION_CONTRACT),
            ModelInputItem::text(InputMessageRole::User, current_user),
        ],
    })
}

fn summarize_trials(trials: &[TrialReport]) -> Vec<CandidateSummary> {
    SchemaKind::all()
        .iter()
        .flat_map(|schema| {
            ContextKind::all()
                .iter()
                .map(move |context| (*schema, *context))
        })
        .filter_map(|(schema, context)| {
            let candidate_trials = trials
                .iter()
                .filter(|trial| trial.schema == schema && trial.context == context)
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
            let tool_issues = evaluable_trials
                .iter()
                .map(|trial| trial.tool_issue_count)
                .sum();
            let pseudo_tool_texts = evaluable_trials
                .iter()
                .filter(|trial| trial.pseudo_tool_text)
                .count();
            let target_validity_failures = evaluable_trials
                .iter()
                .filter(|trial| trial.target_validity_failure)
                .count();
            let target_counts = target_counts(evaluable_trials.iter().copied());
            let peer_targets = target_counts.get(TARGET_PEER).copied().unwrap_or(0);
            let non_peer_targets = target_counts
                .iter()
                .filter(|(target, _count)| target.as_str() != TARGET_PEER)
                .map(|(_target, count)| *count)
                .sum();
            let transport_errors = candidate_trials
                .iter()
                .filter(|trial| trial.transport_error)
                .count();
            let evaluable_count = evaluable_trials.len();
            Some(CandidateSummary {
                schema,
                context,
                attempted_trials: candidate_trials.len(),
                evaluable_trials: evaluable_count,
                transport_errors,
                successes,
                failures: evaluable_count.saturating_sub(successes),
                success_rate: if evaluable_count == 0 {
                    0.0
                } else {
                    successes as f64 / evaluable_count as f64
                },
                tool_issues,
                pseudo_tool_texts,
                peer_targets,
                non_peer_targets,
                peer_target_rate: if evaluable_count == 0 {
                    0.0
                } else {
                    peer_targets as f64 / evaluable_count as f64
                },
                target_counts,
                target_validity_failures,
                failure_reasons: failure_reason_counts(evaluable_trials.iter().copied()),
                production_distance: schema.production_distance() + context.production_distance(),
                usage: usage_summary(
                    evaluable_trials
                        .iter()
                        .filter_map(|trial| trial.usage.as_ref()),
                ),
                latency_ms: numeric_stats(
                    evaluable_trials
                        .iter()
                        .map(|trial| u64::try_from(trial.latency_ms).unwrap_or(u64::MAX)),
                ),
            })
        })
        .collect()
}

fn failure_reason_counts<'a>(
    trials: impl IntoIterator<Item = &'a TrialReport>,
) -> BTreeMap<String, usize> {
    let mut counts = BTreeMap::new();
    for trial in trials {
        if trial.success {
            continue;
        }
        let reason = trial
            .failure
            .as_deref()
            .unwrap_or("unknown_failure")
            .to_owned();
        *counts.entry(reason).or_default() += 1;
    }
    counts
}

fn target_counts<'a>(trials: impl IntoIterator<Item = &'a TrialReport>) -> BTreeMap<String, usize> {
    let mut counts = BTreeMap::new();
    for trial in trials {
        if let Some(target) = trial.target.as_deref() {
            *counts.entry(target.to_owned()).or_default() += 1;
        }
    }
    counts
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

fn pick_winner(summaries: &[CandidateSummary]) -> Option<Winner> {
    summaries
        .iter()
        .max_by(|left, right| {
            compare_success_rate(left, right)
                .then_with(|| right.tool_issues.cmp(&left.tool_issues))
                .then_with(|| right.pseudo_tool_texts.cmp(&left.pseudo_tool_texts))
                .then_with(|| right.production_distance.cmp(&left.production_distance))
                .then_with(|| {
                    let left_p50 = left.usage.total_tokens.p50.unwrap_or(u64::MAX);
                    let right_p50 = right.usage.total_tokens.p50.unwrap_or(u64::MAX);
                    right_p50.cmp(&left_p50)
                })
        })
        .map(|summary| Winner {
            schema: summary.schema,
            context: summary.context,
        })
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
    out.push_str("# Speak Planning Probe\n\n");
    out.push_str(&format!(
        "- generated_at: {}\n- model_set: {}\n- model: {} ({})\n- endpoint: {}\n- max_concurrent_llm_calls: {}\n- trials_per_candidate: {}\n- target_schema_values: {}\n- winner: {}\n\n",
        report.generated_at,
        report.model_set,
        report.model,
        report.model_key,
        report.endpoint,
        report.max_concurrent_llm_calls,
        report.trials_per_candidate,
        report.target_schema_values.join(", "),
        report
            .winner
            .map(|winner| format!("{} + {}", winner.schema, winner.context))
            .unwrap_or_else(|| "(none)".to_owned())
    ));
    out.push_str(
        "| schema | context | successes | evaluable | attempted | success_rate | transport_errors | tool_issues | pseudo_tool_texts | target_failures | production_distance |\n",
    );
    out.push_str("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n");
    for summary in &report.candidates {
        out.push_str(&format!(
            "| {} | {} | {} | {} | {} | {:.3} | {} | {} | {} | {} | {} |\n",
            summary.schema,
            summary.context,
            summary.successes,
            summary.evaluable_trials,
            summary.attempted_trials,
            summary.success_rate,
            summary.transport_errors,
            summary.tool_issues,
            summary.pseudo_tool_texts,
            summary.target_validity_failures,
            summary.production_distance
        ));
    }
    out.push_str("\n## Target Selection\n\n");
    out.push_str(
        "| schema | context | peer targets | non-peer targets | peer target rate | target counts |\n",
    );
    out.push_str("|---|---|---:|---:|---:|---|\n");
    for summary in &report.candidates {
        out.push_str(&format!(
            "| {} | {} | {} | {} | {:.3} | {} |\n",
            summary.schema,
            summary.context,
            summary.peer_targets,
            summary.non_peer_targets,
            summary.peer_target_rate,
            format_counts(&summary.target_counts)
        ));
    }
    out.push_str("\n## Token Usage\n\n");
    out.push_str(
        "| schema | context | samples | input p50 | output p50 | total p50 | total p95 | latency p50 ms | latency p95 ms |\n",
    );
    out.push_str("|---|---|---:|---:|---:|---:|---:|---:|---:|\n");
    for summary in &report.candidates {
        out.push_str(&format!(
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} |\n",
            summary.schema,
            summary.context,
            summary.usage.sample_count,
            format_u64(summary.usage.input_tokens.p50),
            format_u64(summary.usage.output_tokens.p50),
            format_u64(summary.usage.total_tokens.p50),
            format_u64(summary.usage.total_tokens.p95),
            format_u64(summary.latency_ms.p50),
            format_u64(summary.latency_ms.p95),
        ));
    }
    out.push_str("\n## Failure Reasons\n\n");
    out.push_str("| schema | context | failure reasons |\n");
    out.push_str("|---|---|---|\n");
    for summary in &report.candidates {
        out.push_str(&format!(
            "| {} | {} | {} |\n",
            summary.schema,
            summary.context,
            format_counts(&summary.failure_reasons)
        ));
    }
    out
}

fn format_u64(value: Option<u64>) -> String {
    value.map_or_else(|| "-".to_owned(), |value| value.to_string())
}

fn format_counts(counts: &BTreeMap<String, usize>) -> String {
    if counts.is_empty() {
        return "-".to_owned();
    }
    let mut counts = counts.iter().collect::<Vec<_>>();
    counts.sort_by(|(left_reason, left_count), (right_reason, right_count)| {
        right_count
            .cmp(left_count)
            .then_with(|| left_reason.cmp(right_reason))
    });
    counts
        .into_iter()
        .map(|(reason, count)| format!("{reason}={count}"))
        .collect::<Vec<_>>()
        .join(", ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validator_accepts_peer_greeting_acknowledgement() {
        let report = validate_candidate_decision(CandidateDecision::Speak {
            target: "Peer".to_owned(),
            speech_content: "Say hi back to Peer.".to_owned(),
        });

        assert_eq!(
            report,
            ValidationReport {
                success: true,
                failure: None,
                target: Some("Peer".to_owned()),
                speech_content: Some("Say hi back to Peer.".to_owned()),
                pseudo_tool_text: false,
                target_validity_failure: false,
            }
        );
    }

    #[test]
    fn validator_rejects_silence_on_greeting() {
        let report = validate_candidate_decision(CandidateDecision::Silent {
            reason: "No need to speak.".to_owned(),
        });

        assert_eq!(report.failure.as_deref(), Some("silent_on_greeting"));
        assert!(!report.success);
    }

    #[test]
    fn validator_rejects_target_outside_scene() {
        let report = validate_candidate_decision(CandidateDecision::Speak {
            target: "Koro".to_owned(),
            speech_content: "Say hello.".to_owned(),
        });

        assert_eq!(report.failure.as_deref(), Some("invalid_target"));
        assert!(report.target_validity_failure);
    }

    #[test]
    fn target_schema_can_include_everyone() {
        let value = serde_json::to_value(speech_target_schema([TARGET_PEER, "everyone"])).unwrap();

        assert_eq!(
            value,
            serde_json::json!({
                "type": "string",
                "enum": ["Peer", "everyone"],
            })
        );
    }

    #[test]
    fn validator_rejects_idle_or_internal_speech_content() {
        let report = validate_candidate_decision(CandidateDecision::Speak {
            target: "Peer".to_owned(),
            speech_content: "I have been idle for one second.".to_owned(),
        });

        assert_eq!(report.failure.as_deref(), Some("bad_speech_content"));
        assert!(!report.success);
    }

    #[test]
    fn validator_rejects_listener_reaction_as_speech_content() {
        let report = validate_candidate_decision(CandidateDecision::Speak {
            target: "Peer".to_owned(),
            speech_content: "Peer replies warmly and smiles.".to_owned(),
        });

        assert_eq!(report.failure.as_deref(), Some("bad_speech_content"));
        assert!(!report.success);
    }

    #[test]
    fn validator_rejects_non_greeting_content() {
        let report = validate_candidate_decision(CandidateDecision::Speak {
            target: "Peer".to_owned(),
            speech_content: "I can see the bench.".to_owned(),
        });

        assert_eq!(
            report.failure.as_deref(),
            Some("not_greeting_acknowledgement")
        );
        assert!(!report.success);
    }

    #[test]
    fn pseudo_tool_detection_catches_textual_tool_forms() {
        assert!(looks_like_pseudo_tool(
            r#"prepare_speech(target="Peer", speech_content="Hi")"#
        ));
        assert!(looks_like_pseudo_tool(r#"{"target":"Peer"}"#));
        assert!(!looks_like_pseudo_tool("Hello, Peer."));
    }

    #[test]
    fn probe_input_uses_expected_final_roles() {
        assert_eq!(
            last_message_role(probe_input(ContextKind::SystemUserCurrent)),
            Some(InputMessageRole::User)
        );
        assert_eq!(
            last_message_role(probe_input(ContextKind::SystemDeveloper)),
            Some(InputMessageRole::Developer)
        );
        assert_eq!(
            last_message_role(probe_input(ContextKind::SystemSystem)),
            Some(InputMessageRole::System)
        );
        assert_eq!(
            last_message_role(probe_input(ContextKind::SystemUserDeveloper)),
            Some(InputMessageRole::Developer)
        );
        assert_eq!(
            last_message_role(probe_input(ContextKind::SystemDeveloperUser)),
            Some(InputMessageRole::User)
        );
    }

    #[test]
    fn winner_tie_breaks_by_tool_pseudo_distance_then_tokens() {
        let summaries = vec![
            summary(
                SchemaKind::CurrentRequired,
                ContextKind::SystemDeveloper,
                90,
                100,
                0,
                1,
                2,
                100,
            ),
            summary(
                SchemaKind::CurrentRequired,
                ContextKind::SystemUserDeveloper,
                90,
                100,
                0,
                0,
                1,
                110,
            ),
            summary(
                SchemaKind::CurrentOptional,
                ContextKind::SystemUserCurrent,
                90,
                100,
                0,
                0,
                0,
                120,
            ),
        ];

        assert_eq!(
            pick_winner(&summaries),
            Some(Winner {
                schema: SchemaKind::CurrentOptional,
                context: ContextKind::SystemUserCurrent,
            })
        );
    }

    #[test]
    fn summary_excludes_transport_errors_from_success_rate() {
        let trials = vec![
            trial_report(
                SchemaKind::CurrentRequired,
                ContextKind::SystemDeveloper,
                true,
                None,
            ),
            trial_report(
                SchemaKind::CurrentRequired,
                ContextKind::SystemDeveloper,
                false,
                Some("execution_error:request failure (kind=Transport, status=None)"),
            ),
        ];

        let summaries = summarize_trials(&trials);
        let summary = summaries
            .iter()
            .find(|summary| {
                summary.schema == SchemaKind::CurrentRequired
                    && summary.context == ContextKind::SystemDeveloper
            })
            .unwrap();

        assert_eq!(summary.attempted_trials, 2);
        assert_eq!(summary.evaluable_trials, 1);
        assert_eq!(summary.transport_errors, 1);
        assert_eq!(summary.success_rate, 1.0);
    }

    #[test]
    fn summary_counts_peer_and_everyone_targets() {
        let trials = vec![
            trial_report_with_target(
                SchemaKind::CurrentOptional,
                ContextKind::SystemUserCurrent,
                true,
                None,
                Some("Peer"),
            ),
            trial_report_with_target(
                SchemaKind::CurrentOptional,
                ContextKind::SystemUserCurrent,
                false,
                Some("invalid_target"),
                Some("everyone"),
            ),
        ];

        let summaries = summarize_trials(&trials);
        let summary = summaries
            .iter()
            .find(|summary| {
                summary.schema == SchemaKind::CurrentOptional
                    && summary.context == ContextKind::SystemUserCurrent
            })
            .unwrap();

        assert_eq!(summary.peer_targets, 1);
        assert_eq!(summary.non_peer_targets, 1);
        assert_eq!(summary.peer_target_rate, 0.5);
        assert_eq!(
            summary.target_counts,
            BTreeMap::from([("Peer".to_owned(), 1), ("everyone".to_owned(), 1),])
        );
    }

    fn last_message_role(input: ModelInput) -> Option<InputMessageRole> {
        input.items().last().and_then(|item| match item {
            ModelInputItem::Message { role, .. } => Some(*role),
            _ => None,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn summary(
        schema: SchemaKind,
        context: ContextKind,
        successes: usize,
        evaluable_trials: usize,
        tool_issues: usize,
        pseudo_tool_texts: usize,
        production_distance: u32,
        total_tokens_p50: u64,
    ) -> CandidateSummary {
        CandidateSummary {
            schema,
            context,
            attempted_trials: evaluable_trials,
            evaluable_trials,
            transport_errors: 0,
            successes,
            failures: evaluable_trials - successes,
            success_rate: successes as f64 / evaluable_trials as f64,
            tool_issues,
            pseudo_tool_texts,
            peer_targets: successes,
            non_peer_targets: 0,
            peer_target_rate: successes as f64 / evaluable_trials as f64,
            target_counts: BTreeMap::from([("Peer".to_owned(), successes)]),
            target_validity_failures: 0,
            failure_reasons: BTreeMap::new(),
            production_distance,
            usage: UsageSummary {
                sample_count: 1,
                total_tokens: NumericStats {
                    sample_count: 1,
                    p50: Some(total_tokens_p50),
                    ..NumericStats::default()
                },
                ..UsageSummary::default()
            },
            latency_ms: NumericStats::default(),
        }
    }

    fn trial_report(
        schema: SchemaKind,
        context: ContextKind,
        success: bool,
        failure: Option<&str>,
    ) -> TrialReport {
        trial_report_with_target(schema, context, success, failure, success.then_some("Peer"))
    }

    fn trial_report_with_target(
        schema: SchemaKind,
        context: ContextKind,
        success: bool,
        failure: Option<&str>,
        target: Option<&str>,
    ) -> TrialReport {
        let failure = failure.map(ToOwned::to_owned);
        let target_validity_failure = failure.as_deref() == Some("invalid_target");
        TrialReport {
            schema,
            context,
            trial: 0,
            success,
            transport_error: failure.as_deref().is_some_and(is_transport_failure),
            failure,
            target: target.map(ToOwned::to_owned),
            speech_content: success.then(|| "Say hi back.".to_owned()),
            pseudo_tool_text: false,
            target_validity_failure,
            tool_call_count: usize::from(success),
            tool_issue_count: 0,
            latency_ms: 10,
            usage: Some(lutum::Usage::zero()),
        }
    }
}
