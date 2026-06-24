use std::path::{Path, PathBuf};

use anyhow::Context as _;
use clap::Parser;
use nuillu_server::{
    EmbeddingBackendConfig, EmbeddingRole, RuntimeModule, ServerConfig, default_server_session_id,
    install_lutum_trace_subscriber, load_server_boot_config, parse_model_set_file,
    resolve_llm_backends, resolve_token_fields, run_server_with_visualizer,
};

const DEFAULT_OPENAI_EMBEDDING_ENDPOINT: &str = "https://api.openai.com/v1";
const STATE_MODEL_SET_FILE: &str = "model-set.eure";

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
    ///
    /// Defaults to <state>/model-set.eure.
    #[arg(long)]
    model_set: Option<PathBuf>,

    /// Modules to force-disable at startup.
    #[arg(long = "disable-module", value_enum, value_name = "MODULE")]
    disable_module: Vec<RuntimeModule>,

    /// Participants currently available to the speak module as targets.
    #[arg(long = "participant", value_name = "NAME")]
    participants: Vec<String>,

    /// Back up existing agent.db under --state before connecting, then start with a fresh DB.
    #[arg(long)]
    fresh_agent_db: bool,

    /// Path to an existing visualizer GUI binary.
    ///
    /// Defaults to building the visualizer bin target.
    #[arg(long = "visualizer-bin", value_name = "PATH")]
    visualizer_bin: Option<PathBuf>,
}

fn main() -> anyhow::Result<()> {
    install_lutum_trace_subscriber()?;
    let args = Args::parse();
    let model_set_path = resolve_model_set_path(&args.state, args.model_set);
    let model_set = parse_model_set_file(&model_set_path)?;
    let backends = resolve_llm_backends(&model_set)?;
    let cheap_backend = backends.cheap;
    let default_backend = backends.default;
    let premium_backend = backends.premium;
    let embedding_backend = resolve_embedding(&model_set.embedding)?;
    let session_id = resolve_session_id(args.session_id, args.run_id);
    let boot_config = load_server_boot_config(&args.state)?;

    run_server_with_visualizer(ServerConfig {
        state_dir: args.state,
        session_id,
        llm_log_root: args.llm_log_root,
        cheap_backend,
        default_backend,
        premium_backend,
        embedding_backend,
        boot_config,
        disabled_modules: args.disable_module,
        participants: args.participants,
        fresh_agent_db: args.fresh_agent_db,
        visualizer_bin: args.visualizer_bin,
    })
    .context("run nuillu server")
}

fn resolve_session_id(session_id: Option<String>, run_id_alias: Option<String>) -> String {
    session_id
        .or(run_id_alias)
        .unwrap_or_else(default_server_session_id)
}

fn resolve_model_set_path(state_dir: &Path, model_set: Option<PathBuf>) -> PathBuf {
    model_set.unwrap_or_else(|| state_dir.join(STATE_MODEL_SET_FILE))
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

    #[test]
    fn resolve_model_set_path_prefers_explicit_path() {
        let explicit = PathBuf::from("configs/modelsets/server.eure");

        assert_eq!(
            resolve_model_set_path(Path::new(".tmp/server"), Some(explicit.clone())),
            explicit
        );
    }

    #[test]
    fn resolve_model_set_path_defaults_under_state_dir() {
        assert_eq!(
            resolve_model_set_path(Path::new(".tmp/custom-server"), None),
            PathBuf::from(".tmp/custom-server/model-set.eure")
        );
    }

    #[test]
    fn args_parse_fresh_agent_db_flag() {
        let args = Args::parse_from(["nuillu-server", "--fresh-agent-db"]);

        assert!(args.fresh_agent_db);
    }

    #[test]
    fn args_default_to_reusing_agent_db() {
        let args = Args::parse_from(["nuillu-server"]);

        assert!(!args.fresh_agent_db);
    }

    #[test]
    fn args_parse_visualizer_bin_path() {
        let args = Args::parse_from(["nuillu-server", "--visualizer-bin", "/custom/visualizer"]);

        assert_eq!(
            args.visualizer_bin,
            Some(PathBuf::from("/custom/visualizer"))
        );
    }
}
