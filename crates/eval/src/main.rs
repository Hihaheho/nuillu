use std::path::PathBuf;

use anyhow::Context as _;
use clap::Parser;
use nuillu_eval::{
    EmbeddingBackendConfig, EmbeddingRole, LiveOutput, RunnerConfig, default_run_id,
    gui::run_suite_with_visualizer, install_lutum_trace_subscriber, parse_model_set_file,
    resolve_llm_backends, resolve_token_fields, run_suite,
};
use nuillu_module::LlmConcurrencyPool;
use tokio::runtime::Builder;

const DEFAULT_OPENAI_EMBEDDING_ENDPOINT: &str = "https://api.openai.com/v1";

#[derive(Debug, Parser)]
#[command(
    name = "nuillu-eval",
    about = "Evaluate and benchmark configured Nuillu agents"
)]
struct Args {
    /// Eure case file or directory to load recursively.
    #[arg(long, default_value = "eval-cases")]
    cases: PathBuf,

    /// Root directory for JSON eval outputs.
    #[arg(long, default_value = ".tmp/eval")]
    output: PathBuf,

    /// Root directory for per-turn LLM trace files.
    #[arg(long, default_value = "llm-logs")]
    llm_log_root: PathBuf,

    /// Run id used as the output subdirectory name.
    #[arg(long)]
    run_id: Option<String>,

    /// Model set Eure file with per-role backend config.
    #[arg(long)]
    model_set: PathBuf,

    /// Override every selected case's runtime config (useful for benchmarking one agent).
    #[arg(long, value_name = "CONFIG")]
    runtime_config: Option<PathBuf>,

    /// Stop the suite after the first failed or invalid case.
    #[arg(long)]
    fail_fast: bool,

    /// Run only cases that failed or were invalid in the latest eval report under --output.
    #[arg(long)]
    failed_only: bool,

    /// Run only cases that failed or were invalid in the referenced eval report.
    ///
    /// Accepts a suite-report.json path, a run directory, or a run id under --output.
    #[arg(long, value_name = "RUN_ID_OR_PATH")]
    failed_from: Option<PathBuf>,

    /// Number of independent trials to run per selected case.
    #[arg(long, default_value = "1")]
    trials: std::num::NonZeroUsize,

    /// Run the reusable egui visualizer while the eval suite executes.
    #[arg(long)]
    gui: bool,

    /// Maximum number of eval cases executed concurrently.
    #[arg(long, default_value = "3")]
    concurrency: std::num::NonZeroUsize,

    /// Print every runtime event live. Detailed events are always saved to events.jsonl.
    #[arg(long, conflicts_with = "quiet")]
    verbose: bool,

    /// Suppress live progress output and print only the final suite summary.
    #[arg(long, conflicts_with = "verbose")]
    quiet: bool,

    /// Optional case id/path substring patterns. When present, only matching cases run.
    #[arg(value_name = "PATTERN")]
    patterns: Vec<String>,
}

fn main() -> anyhow::Result<()> {
    install_lutum_trace_subscriber()?;
    let args = Args::parse();
    let run_id = args.run_id.unwrap_or_else(default_run_id);
    let model_set = parse_model_set_file(&args.model_set)?;

    let backends = resolve_llm_backends(&model_set)?;
    let judge_backend = backends.judge.ok_or_else(|| {
        anyhow::anyhow!("missing judge tier: set judge-model or judge {{ model = ... }}")
    })?;
    let cheap_backend = backends.cheap;
    let default_backend = backends.default;
    let premium_backend = backends.premium;
    let image_backend = backends.image;
    let model_concurrency = backends.model_concurrency;

    let embedding_backend = resolve_embedding(&model_set.embedding)?;
    let config = RunnerConfig {
        cases_root: args.cases,
        output_root: args.output,
        llm_log_root: args.llm_log_root,
        run_id,
        judge_backend,
        cheap_backend,
        default_backend,
        premium_backend,
        image_backend,
        embedding_backend,
        fail_fast: args.fail_fast,
        failed_only: args.failed_only,
        failed_from: args.failed_from,
        model_concurrency,
        llm_concurrency_pool: LlmConcurrencyPool::default(),
        trials: args.trials,
        case_concurrency: args.concurrency,
        case_patterns: args.patterns,
        module_factories: Vec::new(),
        runtime_config_override: args.runtime_config,
        live_output: if args.quiet {
            LiveOutput::Quiet
        } else if args.verbose {
            LiveOutput::Verbose
        } else {
            LiveOutput::Normal
        },
    };

    if args.gui {
        return run_suite_with_visualizer(config);
    }

    let runtime = Builder::new_current_thread()
        .enable_all()
        .build()
        .context("build eval tokio runtime")?;
    let report = runtime.block_on(run_suite(&config))?;
    let metrics = format_cli_metrics(&report.metrics);
    println!(
        "cases={} passed={} failed={} invalid={} elapsed_ms={}{} output={}",
        report.case_count,
        report.passed_cases,
        report.failed_cases,
        report.invalid_cases,
        report.timing.elapsed_ms,
        metrics,
        config.output_root.join(&config.run_id).display()
    );
    Ok(())
}

fn format_cli_metrics(metrics: &nuillu_eval::SuiteMetrics) -> String {
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

fn resolve_embedding(role: &EmbeddingRole) -> anyhow::Result<EmbeddingBackendConfig> {
    let endpoint = role
        .endpoint()
        .unwrap_or(DEFAULT_OPENAI_EMBEDDING_ENDPOINT)
        .to_string();
    let token = resolve_token_fields(
        "embedding",
        role.token_env.as_deref(),
        role.token.as_deref(),
        None,
    )?;
    Ok(EmbeddingBackendConfig {
        endpoint,
        token,
        model: role.model.clone(),
        dimensions: role.dimensions as usize,
    })
}
