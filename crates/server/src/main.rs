use std::io::Write as _;
use std::path::{Path, PathBuf};

use anyhow::Context as _;
use clap::{Args as ClapArgs, Parser, Subcommand, ValueEnum};
use nuillu_server::{
    EmbeddingBackendConfig, EmbeddingRole, RuntimeModule, ServerConfig, default_server_session_id,
    history::{export_conversation_history, render_conversation_history_markdown},
    install_lutum_trace_subscriber, load_server_boot_config, parse_model_set_file,
    resolve_llm_backends, resolve_token_fields, run_server_with_visualizer,
};

const DEFAULT_OPENAI_EMBEDDING_ENDPOINT: &str = "https://api.openai.com/v1";
const AGENT_DB_FILE: &str = "agent.db";
const STATE_MODEL_SET_FILE: &str = "model-set.eure";

#[derive(Debug, Parser)]
#[command(
    name = "nuillu-server",
    about = "Run a nuillu server with the visualizer GUI"
)]
struct Args {
    #[command(flatten)]
    run: RunArgs,

    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Run a nuillu server with the visualizer GUI.
    Run(RunArgs),
    /// Export user/agent conversation history from the server agent DB.
    History(HistoryArgs),
}

#[derive(Debug, ClapArgs)]
struct RunArgs {
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
    #[arg(long, conflicts_with = "agent_db")]
    fresh_agent_db: bool,

    /// Override the persistent agent DB path. Defaults to <state>/agent.db.
    #[arg(long, value_name = "PATH", conflicts_with = "fresh_agent_db")]
    agent_db: Option<PathBuf>,

    /// Path to an existing visualizer GUI binary.
    ///
    /// Defaults to building the visualizer bin target.
    #[arg(long = "visualizer-bin", value_name = "PATH")]
    visualizer_bin: Option<PathBuf>,
}

#[derive(Debug, ClapArgs)]
struct HistoryArgs {
    /// Persistent server runtime state directory. Defaults to the top-level --state.
    #[arg(long)]
    state: Option<PathBuf>,

    /// Override the persistent agent DB path.
    #[arg(long, value_name = "PATH")]
    agent_db: Option<PathBuf>,

    /// Output format.
    #[arg(long, value_enum, default_value_t = HistoryOutputFormat::Json)]
    output: HistoryOutputFormat,

    /// Server session id to export. Can be repeated.
    #[arg(long = "session-id", value_name = "ID")]
    session_ids: Vec<String>,

    /// Display name used for agent utterances.
    #[arg(long, default_value = "Nui")]
    agent_name: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum HistoryOutputFormat {
    Json,
    Markdown,
}

fn main() -> anyhow::Result<()> {
    install_lutum_trace_subscriber()?;
    let Args { run, command } = Args::parse();
    match command {
        Some(Command::Run(run_args)) => run_server(run_args),
        Some(Command::History(history_args)) => {
            export_history(history_args, run.state, run.agent_db)
        }
        None => run_server(run),
    }
}

fn run_server(args: RunArgs) -> anyhow::Result<()> {
    let model_set_path = resolve_model_set_path(&args.state, args.model_set);
    let agent_db_path = resolve_agent_db_path(&args.state, args.agent_db);
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
        agent_db_path,
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

fn export_history(
    args: HistoryArgs,
    default_state: PathBuf,
    default_agent_db: Option<PathBuf>,
) -> anyhow::Result<()> {
    let agent_db_path = resolve_history_agent_db_path(
        args.state.as_deref(),
        args.agent_db.as_deref(),
        &default_state,
        default_agent_db.as_deref(),
    );
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .context("build history export runtime")?;
    let export = runtime.block_on(export_conversation_history(
        &agent_db_path,
        &args.session_ids,
        &args.agent_name,
    ))?;
    match args.output {
        HistoryOutputFormat::Json => {
            let stdout = std::io::stdout();
            let mut lock = stdout.lock();
            serde_json::to_writer_pretty(&mut lock, &export).context("write history JSON")?;
            writeln!(lock).context("write history JSON newline")?;
        }
        HistoryOutputFormat::Markdown => {
            print!("{}", render_conversation_history_markdown(&export));
        }
    }
    Ok(())
}

fn resolve_session_id(session_id: Option<String>, run_id_alias: Option<String>) -> String {
    session_id
        .or(run_id_alias)
        .unwrap_or_else(default_server_session_id)
}

fn resolve_model_set_path(state_dir: &Path, model_set: Option<PathBuf>) -> PathBuf {
    model_set.unwrap_or_else(|| state_dir.join(STATE_MODEL_SET_FILE))
}

fn resolve_agent_db_path(state_dir: &Path, agent_db: Option<PathBuf>) -> PathBuf {
    agent_db.unwrap_or_else(|| state_dir.join(AGENT_DB_FILE))
}

fn resolve_history_agent_db_path(
    history_state: Option<&Path>,
    history_agent_db: Option<&Path>,
    default_state: &Path,
    default_agent_db: Option<&Path>,
) -> PathBuf {
    history_agent_db
        .or(default_agent_db)
        .map(Path::to_path_buf)
        .unwrap_or_else(|| history_state.unwrap_or(default_state).join(AGENT_DB_FILE))
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
    fn resolve_agent_db_path_defaults_under_state_dir() {
        assert_eq!(
            resolve_agent_db_path(Path::new(".tmp/custom-server"), None),
            PathBuf::from(".tmp/custom-server/agent.db")
        );
    }

    #[test]
    fn resolve_agent_db_path_prefers_explicit_path() {
        let explicit = PathBuf::from(".tmp/custom-agent.db");

        assert_eq!(
            resolve_agent_db_path(Path::new(".tmp/server"), Some(explicit.clone())),
            explicit
        );
    }

    #[test]
    fn resolve_history_agent_db_path_prefers_history_agent_db() {
        assert_eq!(
            resolve_history_agent_db_path(
                Some(Path::new(".tmp/history-state")),
                Some(Path::new(".tmp/history-agent.db")),
                Path::new(".tmp/top-state"),
                Some(Path::new(".tmp/top-agent.db")),
            ),
            PathBuf::from(".tmp/history-agent.db")
        );
    }

    #[test]
    fn resolve_history_agent_db_path_uses_top_level_agent_db() {
        assert_eq!(
            resolve_history_agent_db_path(
                Some(Path::new(".tmp/history-state")),
                None,
                Path::new(".tmp/top-state"),
                Some(Path::new(".tmp/top-agent.db")),
            ),
            PathBuf::from(".tmp/top-agent.db")
        );
    }

    #[test]
    fn resolve_history_agent_db_path_defaults_under_resolved_state() {
        assert_eq!(
            resolve_history_agent_db_path(
                Some(Path::new(".tmp/history-state")),
                None,
                Path::new(".tmp/top-state"),
                None,
            ),
            PathBuf::from(".tmp/history-state/agent.db")
        );
    }

    #[test]
    fn args_parse_fresh_agent_db_flag() {
        let args = Args::parse_from(["nuillu-server", "--fresh-agent-db"]);

        assert!(args.run.fresh_agent_db);
        assert!(args.command.is_none());
    }

    #[test]
    fn args_parse_top_level_agent_db_path() {
        let args = Args::parse_from(["nuillu-server", "--agent-db", ".tmp/custom-agent.db"]);

        assert_eq!(
            args.run.agent_db,
            Some(PathBuf::from(".tmp/custom-agent.db"))
        );
        assert!(args.command.is_none());
    }

    #[test]
    fn args_default_to_reusing_agent_db() {
        let args = Args::parse_from(["nuillu-server"]);

        assert!(!args.run.fresh_agent_db);
        assert_eq!(args.run.agent_db, None);
        assert!(args.command.is_none());
    }

    #[test]
    fn args_parse_visualizer_bin_path() {
        let args = Args::parse_from(["nuillu-server", "--visualizer-bin", "/custom/visualizer"]);

        assert_eq!(
            args.run.visualizer_bin,
            Some(PathBuf::from("/custom/visualizer"))
        );
        assert!(args.command.is_none());
    }

    #[test]
    fn args_parse_explicit_run_subcommand() {
        let args = Args::parse_from([
            "nuillu-server",
            "run",
            "--state",
            ".tmp/exhibition",
            "--fresh-agent-db",
            "--visualizer-bin",
            "/custom/visualizer",
        ]);

        let Some(Command::Run(run)) = args.command else {
            panic!("expected run subcommand");
        };
        assert_eq!(run.state, PathBuf::from(".tmp/exhibition"));
        assert!(run.fresh_agent_db);
        assert_eq!(
            run.visualizer_bin,
            Some(PathBuf::from("/custom/visualizer"))
        );
    }

    #[test]
    fn args_parse_explicit_run_agent_db_path() {
        let args = Args::parse_from([
            "nuillu-server",
            "run",
            "--state",
            ".tmp/exhibition",
            "--agent-db",
            ".tmp/custom-agent.db",
        ]);

        let Some(Command::Run(run)) = args.command else {
            panic!("expected run subcommand");
        };
        assert_eq!(run.state, PathBuf::from(".tmp/exhibition"));
        assert_eq!(run.agent_db, Some(PathBuf::from(".tmp/custom-agent.db")));
    }

    #[test]
    fn args_reject_agent_db_with_fresh_agent_db() {
        assert!(
            Args::try_parse_from([
                "nuillu-server",
                "--agent-db",
                ".tmp/custom-agent.db",
                "--fresh-agent-db",
            ])
            .is_err()
        );
    }

    #[test]
    fn args_parse_history_subcommand() {
        let args = Args::parse_from([
            "nuillu-server",
            "history",
            "--state",
            ".tmp/exhibition",
            "--output",
            "markdown",
            "--session-id",
            "server-a",
            "--session-id",
            "server-b",
            "--agent-name",
            "Nui",
        ]);

        let Some(Command::History(history)) = args.command else {
            panic!("expected history subcommand");
        };
        assert_eq!(history.state, Some(PathBuf::from(".tmp/exhibition")));
        assert_eq!(history.agent_db, None);
        assert_eq!(history.output, HistoryOutputFormat::Markdown);
        assert_eq!(history.session_ids, vec!["server-a", "server-b"]);
        assert_eq!(history.agent_name, "Nui");
    }

    #[test]
    fn args_parse_history_agent_db_path() {
        let args = Args::parse_from([
            "nuillu-server",
            "history",
            "--agent-db",
            ".tmp/history-agent.db",
        ]);

        let Some(Command::History(history)) = args.command else {
            panic!("expected history subcommand");
        };
        assert_eq!(
            history.agent_db,
            Some(PathBuf::from(".tmp/history-agent.db"))
        );
    }

    #[test]
    fn args_parse_history_with_top_level_state() {
        let args = Args::parse_from(["nuillu-server", "--state", ".tmp/exhibition", "history"]);

        assert_eq!(args.run.state, PathBuf::from(".tmp/exhibition"));
        let Some(Command::History(history)) = args.command else {
            panic!("expected history subcommand");
        };
        assert_eq!(history.state, None);
    }

    #[test]
    fn args_history_uses_top_level_agent_db_default() {
        let args = Args::parse_from([
            "nuillu-server",
            "--agent-db",
            ".tmp/top-agent.db",
            "history",
        ]);

        let Some(Command::History(history)) = &args.command else {
            panic!("expected history subcommand");
        };
        assert_eq!(
            resolve_history_agent_db_path(
                history.state.as_deref(),
                history.agent_db.as_deref(),
                &args.run.state,
                args.run.agent_db.as_deref(),
            ),
            PathBuf::from(".tmp/top-agent.db")
        );
    }

    #[test]
    fn args_history_agent_db_overrides_top_level_agent_db() {
        let args = Args::parse_from([
            "nuillu-server",
            "--agent-db",
            ".tmp/top-agent.db",
            "history",
            "--agent-db",
            ".tmp/history-agent.db",
        ]);

        let Some(Command::History(history)) = &args.command else {
            panic!("expected history subcommand");
        };
        assert_eq!(
            resolve_history_agent_db_path(
                history.state.as_deref(),
                history.agent_db.as_deref(),
                &args.run.state,
                args.run.agent_db.as_deref(),
            ),
            PathBuf::from(".tmp/history-agent.db")
        );
    }
}
