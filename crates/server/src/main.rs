use std::{num::NonZeroUsize, path::PathBuf};

use anyhow::Context as _;
use clap::Parser;
use nuillu_module::DEFAULT_SESSION_COMPACTION_INPUT_TOKEN_THRESHOLD;
use nuillu_server::{
    EmbeddingBackendConfig, EmbeddingRole, LlmBackendConfig, ModelSet, ModelSetRole,
    ReasoningEffort, RuntimeModule, ServerConfig, default_run_id, install_lutum_trace_subscriber,
    parse_model_set_file, run_server_with_visualizer,
};

const DEFAULT_OPENAI_COMPAT_ENDPOINT: &str = "http://localhost:11434/v1";
const DEFAULT_OPENAI_COMPAT_TOKEN: &str = "local";
const DEFAULT_MODEL_DIR: &str = "models/potion-base-8M";
const DEFAULT_OPENAI_EMBEDDING_ENDPOINT: &str = "https://api.openai.com/v1";

#[derive(Debug, Parser)]
#[command(
    name = "nuillu-server",
    about = "Run a nuillu server with the visualizer GUI"
)]
struct Args {
    /// Persistent server runtime state directory.
    #[arg(long, default_value = ".tmp/server")]
    state: PathBuf,

    /// Run id used for server event logs.
    #[arg(long)]
    run_id: Option<String>,

    /// Model set Eure file with per-role backend config.
    #[arg(long)]
    model_set: Option<PathBuf>,

    /// Optional cheap-tier model override.
    #[arg(long)]
    cheap_model: Option<String>,

    /// Optional default-tier model override.
    #[arg(long)]
    default_model: Option<String>,

    /// Optional premium-tier model override.
    #[arg(long)]
    premium_model: Option<String>,

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

    /// Maximum concurrent agent LLM calls. Defaults to unlimited.
    #[arg(long, env = "NUILLU_SERVER_MAX_CONCURRENT_LLM_CALLS")]
    max_concurrent_llm_calls: Option<NonZeroUsize>,

    /// Modules to force-disable at startup.
    #[arg(long = "disable-module", value_enum, value_name = "MODULE")]
    disable_module: Vec<RuntimeModule>,

    /// Participants currently available to the speak module as targets.
    #[arg(long = "participant", value_name = "NAME")]
    participants: Vec<String>,
}

fn main() -> anyhow::Result<()> {
    install_lutum_trace_subscriber()?;
    let args = Args::parse();
    let model_set = args
        .model_set
        .as_deref()
        .map(parse_model_set_file)
        .transpose()?;
    let global_compaction_input_token_threshold = model_set
        .as_ref()
        .and_then(|model_set| model_set.compaction_input_token_threshold);
    let cheap_backend = resolve_backend(
        "cheap",
        role_ref(model_set.as_ref(), ModelRole::Cheap),
        global_compaction_input_token_threshold,
        args.cheap_model.as_deref(),
        args.cheap_reasoning_effort,
    )?;
    let default_backend = resolve_backend(
        "default",
        role_ref(model_set.as_ref(), ModelRole::Default),
        global_compaction_input_token_threshold,
        args.default_model.as_deref(),
        args.default_reasoning_effort,
    )?;
    let premium_backend = resolve_backend(
        "premium",
        role_ref(model_set.as_ref(), ModelRole::Premium),
        global_compaction_input_token_threshold,
        args.premium_model.as_deref(),
        args.premium_reasoning_effort,
    )?;
    let embedding_backend =
        resolve_embedding(model_set.as_ref().and_then(|m| m.embedding.as_ref()))?;
    let run_id = args
        .run_id
        .unwrap_or_else(|| format!("server-{}", default_run_id()));

    run_server_with_visualizer(ServerConfig {
        state_dir: args.state,
        run_id,
        cheap_backend,
        default_backend,
        premium_backend,
        model_dir: args.model_dir,
        embedding_backend,
        max_concurrent_llm_calls: args.max_concurrent_llm_calls,
        disabled_modules: args.disable_module,
        participants: args.participants,
    })
    .context("run nuillu server")
}

#[derive(Clone, Copy)]
enum ModelRole {
    Cheap,
    Default,
    Premium,
}

fn resolve_backend(
    label: &str,
    role: Option<&ModelSetRole>,
    global_compaction_input_token_threshold: Option<u64>,
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
    let compaction_input_token_threshold = role
        .and_then(|role| role.compaction_input_token_threshold)
        .or(global_compaction_input_token_threshold)
        .unwrap_or(DEFAULT_SESSION_COMPACTION_INPUT_TOKEN_THRESHOLD);

    Ok(LlmBackendConfig {
        endpoint,
        token,
        model,
        reasoning_effort,
        use_responses_api,
        compaction_input_token_threshold,
    })
}

fn resolve_token(label: &str, role: Option<&ModelSetRole>) -> anyhow::Result<String> {
    let Some(role) = role else {
        return Ok(DEFAULT_OPENAI_COMPAT_TOKEN.to_string());
    };
    resolve_token_fields(
        label,
        role.token_env.as_deref(),
        role.token.as_deref(),
        Some(DEFAULT_OPENAI_COMPAT_TOKEN),
    )
}

fn resolve_token_fields(
    label: &str,
    token_env: Option<&str>,
    token: Option<&str>,
    default: Option<&str>,
) -> anyhow::Result<String> {
    if let Some(env_name) = token_env {
        match std::env::var(env_name) {
            Ok(value) if !value.trim().is_empty() => return Ok(value),
            Ok(_) => {
                if let Some(value) = token {
                    return Ok(value.to_string());
                }
                anyhow::bail!("{label}.token-env {env_name} is set but empty");
            }
            Err(source) => {
                if let Some(value) = token {
                    return Ok(value.to_string());
                }
                return Err(source)
                    .with_context(|| format!("read {label}.token-env {env_name} from model set"));
            }
        }
    }

    if let Some(value) = token {
        return Ok(value.to_string());
    }
    match default {
        Some(value) => Ok(value.to_string()),
        None => anyhow::bail!("{label}.token or {label}.token-env must be set"),
    }
}

fn resolve_embedding(
    role: Option<&EmbeddingRole>,
) -> anyhow::Result<Option<EmbeddingBackendConfig>> {
    let Some(role) = role else {
        return Ok(None);
    };
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
    let model = role
        .model
        .clone()
        .ok_or_else(|| anyhow::anyhow!("missing embedding.model in --model-set"))?;
    let dimensions = role
        .dimensions
        .ok_or_else(|| anyhow::anyhow!("missing embedding.dimensions in --model-set"))?;
    if dimensions == 0 {
        anyhow::bail!("embedding.dimensions must be greater than zero");
    }
    Ok(Some(EmbeddingBackendConfig {
        endpoint,
        token,
        model,
        dimensions: dimensions as usize,
    }))
}

fn role_ref(model_set: Option<&ModelSet>, role: ModelRole) -> Option<&ModelSetRole> {
    let model_set = model_set?;
    match role {
        ModelRole::Cheap => model_set.cheap.as_ref(),
        ModelRole::Default => model_set.default.as_ref(),
        ModelRole::Premium => model_set.premium.as_ref(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model_role(threshold: Option<u64>) -> ModelSetRole {
        ModelSetRole {
            endpoint: None,
            base_url: None,
            token: Some("local".to_string()),
            token_env: None,
            model: Some("model".to_string()),
            reasoning_effort: None,
            use_responses_api: None,
            compaction_input_token_threshold: threshold,
        }
    }

    #[test]
    fn resolve_backend_defaults_compaction_input_token_threshold_to_16k() {
        let role = model_role(None);

        let backend = resolve_backend("cheap", Some(&role), None, None, None).unwrap();

        assert_eq!(
            backend.compaction_input_token_threshold,
            DEFAULT_SESSION_COMPACTION_INPUT_TOKEN_THRESHOLD
        );
    }

    #[test]
    fn resolve_backend_uses_model_set_compaction_input_token_threshold() {
        let role = model_role(Some(4096));

        let backend = resolve_backend("cheap", Some(&role), None, None, None).unwrap();

        assert_eq!(backend.compaction_input_token_threshold, 4096);
    }

    #[test]
    fn resolve_backend_uses_global_compaction_input_token_threshold() {
        let role = model_role(None);

        let backend = resolve_backend("cheap", Some(&role), Some(2048), None, None).unwrap();

        assert_eq!(backend.compaction_input_token_threshold, 2048);
    }

    #[test]
    fn resolve_backend_role_threshold_overrides_global_threshold() {
        let role = model_role(Some(4096));

        let backend = resolve_backend("cheap", Some(&role), Some(2048), None, None).unwrap();

        assert_eq!(backend.compaction_input_token_threshold, 4096);
    }
}
