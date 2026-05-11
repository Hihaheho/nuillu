use std::{num::NonZeroUsize, path::PathBuf};

use anyhow::Context as _;
use clap::Parser;
use nuillu_eval::{
    EvalModule, LlmBackendConfig, ModelSet, ModelSetRole, ReasoningEffort, RunnerConfig,
    default_run_id, gui::run_suite_with_visualizer, install_lutum_trace_subscriber,
    parse_model_set_file, run_suite,
};
use tokio::runtime::Builder;

const DEFAULT_OPENAI_COMPAT_ENDPOINT: &str = "http://localhost:11434/v1";
const DEFAULT_OPENAI_COMPAT_TOKEN: &str = "local";
const DEFAULT_MODEL_DIR: &str = "models/potion-base-8M";

#[derive(Debug, Parser)]
#[command(
    name = "nuillu-eval",
    about = "Run data-driven nuillu eval cases against real agent/module wiring"
)]
struct Args {
    /// Eure case file or directory to load recursively.
    #[arg(long, default_value = "eval-cases")]
    cases: PathBuf,

    /// Root directory for JSON eval outputs.
    #[arg(long, default_value = ".tmp/eval")]
    output: PathBuf,

    /// Run id used as the output subdirectory name.
    #[arg(long)]
    run_id: Option<String>,

    /// Model set Eure file with per-role backend config.
    #[arg(long)]
    model_set: Option<PathBuf>,

    /// Rubric judge model override.
    #[arg(long)]
    judge_model: Option<String>,

    /// Optional cheap-tier model override.
    #[arg(long)]
    cheap_model: Option<String>,

    /// Optional default-tier model override.
    #[arg(long)]
    default_model: Option<String>,

    /// Optional premium-tier model override.
    #[arg(long)]
    premium_model: Option<String>,

    /// Optional judge reasoning effort override.
    #[arg(long, value_enum)]
    judge_reasoning_effort: Option<ReasoningEffort>,

    /// Optional cheap-tier reasoning effort override.
    #[arg(long, value_enum)]
    cheap_reasoning_effort: Option<ReasoningEffort>,

    /// Optional default-tier reasoning effort override.
    #[arg(long, value_enum)]
    default_reasoning_effort: Option<ReasoningEffort>,

    /// Optional premium-tier reasoning effort override.
    #[arg(long, value_enum)]
    premium_reasoning_effort: Option<ReasoningEffort>,

    /// Local minishlab/potion-base-8M model directory.
    #[arg(long, default_value = DEFAULT_MODEL_DIR)]
    model_dir: PathBuf,

    /// Stop the suite after the first failed or invalid case.
    #[arg(long)]
    fail_fast: bool,

    /// Maximum concurrent agent LLM calls. Defaults to unlimited.
    #[arg(long, env = "NUILLU_EVAL_MAX_CONCURRENT_LLM_CALLS")]
    max_concurrent_llm_calls: Option<NonZeroUsize>,

    /// Run the reusable egui visualizer while the eval suite executes.
    #[arg(long)]
    gui: bool,

    /// Modules to drop from the full-agent wiring (repeatable). Useful for
    /// isolating module-level effects, e.g. `--disable-module speak-gate`.
    /// Required modules (attention-controller, sensory, speak) cannot be disabled.
    #[arg(long = "disable-module", value_enum, value_name = "MODULE")]
    disable_module: Vec<EvalModule>,

    /// Optional case id/path substring patterns. When present, only matching cases run.
    #[arg(value_name = "PATTERN")]
    patterns: Vec<String>,
}

fn main() -> anyhow::Result<()> {
    install_lutum_trace_subscriber()?;
    let args = Args::parse();
    let run_id = args.run_id.unwrap_or_else(default_run_id);
    let model_set = args
        .model_set
        .as_deref()
        .map(parse_model_set_file)
        .transpose()?;
    let judge_backend = resolve_backend(
        "judge",
        role_ref(model_set.as_ref(), ModelRole::Judge),
        args.judge_model.as_deref(),
        args.judge_reasoning_effort,
    )?;
    let cheap_backend = resolve_backend(
        "cheap",
        role_ref(model_set.as_ref(), ModelRole::Cheap),
        args.cheap_model.as_deref(),
        args.cheap_reasoning_effort,
    )?;
    let default_backend = resolve_backend(
        "default",
        role_ref(model_set.as_ref(), ModelRole::Default),
        args.default_model.as_deref(),
        args.default_reasoning_effort,
    )?;
    let premium_backend = resolve_backend(
        "premium",
        role_ref(model_set.as_ref(), ModelRole::Premium),
        args.premium_model.as_deref(),
        args.premium_reasoning_effort,
    )?;
    let config = RunnerConfig {
        cases_root: args.cases,
        output_root: args.output,
        run_id,
        judge_backend,
        cheap_backend,
        default_backend,
        premium_backend,
        model_dir: args.model_dir,
        fail_fast: args.fail_fast,
        max_concurrent_llm_calls: args.max_concurrent_llm_calls,
        case_patterns: args.patterns,
        disabled_modules: args.disable_module,
    };

    if args.gui {
        return run_suite_with_visualizer(config);
    }

    let runtime = Builder::new_current_thread()
        .enable_all()
        .build()
        .context("build eval tokio runtime")?;
    let report = runtime.block_on(run_suite(&config))?;
    println!(
        "cases={} passed={} failed={} invalid={} output={}",
        report.case_count,
        report.passed_cases,
        report.failed_cases,
        report.invalid_cases,
        config.output_root.join(&config.run_id).display()
    );
    Ok(())
}

#[derive(Clone, Copy)]
enum ModelRole {
    Judge,
    Cheap,
    Default,
    Premium,
}

fn resolve_backend(
    label: &str,
    role: Option<&ModelSetRole>,
    explicit_model: Option<&str>,
    explicit_reasoning_effort: Option<ReasoningEffort>,
) -> anyhow::Result<LlmBackendConfig> {
    let endpoint = role
        .and_then(ModelSetRole::endpoint)
        .unwrap_or(DEFAULT_OPENAI_COMPAT_ENDPOINT)
        .to_string();
    let token = resolve_token(label, role)?;
    let model = explicit_model
        .or_else(|| role.and_then(|role| role.model.as_deref()))
        .ok_or_else(|| {
            anyhow::anyhow!(
                "missing {label} model: pass --{label}-model or set {label}.model in --model-set"
            )
        })?
        .to_string();
    let reasoning_effort =
        explicit_reasoning_effort.or_else(|| role.and_then(|role| role.reasoning_effort));
    let use_responses_api = role
        .and_then(|role| role.use_responses_api)
        .unwrap_or(false);

    Ok(LlmBackendConfig {
        endpoint,
        token,
        model,
        reasoning_effort,
        use_responses_api,
    })
}

fn resolve_token(label: &str, role: Option<&ModelSetRole>) -> anyhow::Result<String> {
    let Some(role) = role else {
        return Ok(DEFAULT_OPENAI_COMPAT_TOKEN.to_string());
    };

    if let Some(env_name) = role.token_env.as_deref() {
        match std::env::var(env_name) {
            Ok(token) if !token.trim().is_empty() => return Ok(token),
            Ok(_) => {
                if let Some(token) = role.token.as_ref() {
                    return Ok(token.clone());
                }
                anyhow::bail!("{label}.token-env {env_name} is set but empty");
            }
            Err(source) => {
                if let Some(token) = role.token.as_ref() {
                    return Ok(token.clone());
                }
                return Err(source)
                    .with_context(|| format!("read {label}.token-env {env_name} from model set"));
            }
        }
    }

    Ok(role
        .token
        .clone()
        .unwrap_or_else(|| DEFAULT_OPENAI_COMPAT_TOKEN.to_string()))
}

fn role_ref(model_set: Option<&ModelSet>, role: ModelRole) -> Option<&ModelSetRole> {
    let model_set = model_set?;
    match role {
        ModelRole::Judge => model_set.judge.as_ref(),
        ModelRole::Cheap => model_set.cheap.as_ref(),
        ModelRole::Default => model_set.default.as_ref(),
        ModelRole::Premium => model_set.premium.as_ref(),
    }
}
