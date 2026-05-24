use std::path::PathBuf;

use anyhow::Context as _;
use clap::Parser;
use nuillu_server::{
    EmbeddingBackendConfig, EmbeddingRole, RuntimeModule, ServerConfig, default_server_session_id,
    install_lutum_trace_subscriber, parse_model_set_file, resolve_llm_backends,
    resolve_token_fields, run_server_with_visualizer,
};

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

    /// Deprecated alias for --session-id.
    #[arg(long, hide = true)]
    run_id: Option<String>,

    /// Session id used as the LLM trace namespace.
    #[arg(long)]
    session_id: Option<String>,

    /// Root directory for per-turn LLM trace files.
    #[arg(long, default_value = "llm-logs")]
    llm_log_root: PathBuf,

    /// Model set Eure file with per-role backend config.
    #[arg(long)]
    model_set: PathBuf,

    /// Local minishlab/potion-base-8M model directory.
    #[arg(long, default_value = DEFAULT_MODEL_DIR)]
    model_dir: PathBuf,

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
    let model_set = parse_model_set_file(&args.model_set)?;
    let backends = resolve_llm_backends(&model_set)?;
    let cheap_backend = backends.cheap;
    let default_backend = backends.default;
    let premium_backend = backends.premium;
    let embedding_backend = resolve_embedding(model_set.embedding.as_ref())?;
    let session_id = resolve_session_id(args.session_id, args.run_id);

    run_server_with_visualizer(ServerConfig {
        state_dir: args.state,
        session_id,
        llm_log_root: args.llm_log_root,
        cheap_backend,
        default_backend,
        premium_backend,
        model_dir: args.model_dir,
        embedding_backend,
        disabled_modules: args.disable_module,
        participants: args.participants,
    })
    .context("run nuillu server")
}

fn resolve_session_id(session_id: Option<String>, run_id_alias: Option<String>) -> String {
    session_id
        .or(run_id_alias)
        .unwrap_or_else(default_server_session_id)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_session_id_prefers_explicit_session_id() {
        assert_eq!(
            resolve_session_id(Some("session-1".to_string()), Some("run-1".to_string())),
            "session-1"
        );
    }

    #[test]
    fn resolve_session_id_accepts_run_id_alias() {
        assert_eq!(
            resolve_session_id(None, Some("legacy-run".to_string())),
            "legacy-run"
        );
    }

    #[test]
    fn resolve_session_id_generates_server_session_id() {
        let session_id = resolve_session_id(None, None);

        assert!(session_id.starts_with("server-"));
        assert!(session_id.len() > "server-20260517T000000Z-".len());
    }
}
