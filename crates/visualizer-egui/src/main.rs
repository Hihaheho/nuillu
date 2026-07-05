use std::path::PathBuf;

use clap::{Args as ClapArgs, Parser};
use nuillu_server::{
    RuntimeModule, ServerRunOptions, install_lutum_trace_subscriber,
    load_server_config_from_options, spawn_server_runtime,
};
use nuillu_visualizer_egui::{VisualizerApp, VisualizerChannels};
use nuillu_visualizer_protocol::{VisualizerClientMessage, VisualizerClientPort};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    install_lutum_trace_subscriber()?;
    let args = Args::parse();
    let (server_messages, client_messages, embedded_runtime) = if let Some(host) = args.host {
        let port = VisualizerClientPort::connect(host.as_str())?;
        port.send(VisualizerClientMessage::hello())?;
        let (server_messages, client_messages) = port.into_channels();
        (server_messages, client_messages, None)
    } else {
        let config = load_server_config_from_options(args.server.into_options())?;
        let runtime = spawn_server_runtime(config)?;
        let (server_messages, client_messages) = runtime.visualizer_channels();
        (server_messages, client_messages, Some(runtime))
    };
    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_active(true)
            .with_visible(true),
        ..eframe::NativeOptions::default()
    };
    let result = eframe::run_native(
        "Nuillu Visualizer",
        native_options,
        Box::new(|cc| {
            Ok(Box::new(VisualizerApp::new(
                cc,
                VisualizerChannels {
                    server_messages,
                    client_messages,
                    remote: true,
                },
            )))
        }),
    );
    if let Some(runtime) = embedded_runtime {
        let _ = runtime.shutdown();
        let _ = runtime.join();
    }
    result?;
    Ok(())
}

#[derive(Debug, Parser)]
#[command(name = "nuillu-visualizer-egui", about = "Run the Nuillu visualizer")]
struct Args {
    /// Connect to an already-running visualizer protocol server.
    #[arg(long)]
    host: Option<String>,

    #[command(flatten)]
    server: ServerArgs,
}

#[derive(Debug, Clone, ClapArgs)]
struct ServerArgs {
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
}

impl ServerArgs {
    fn into_options(self) -> ServerRunOptions {
        ServerRunOptions {
            state_dir: self.state,
            run_id: self.run_id,
            session_id: self.session_id,
            llm_log_root: self.llm_log_root,
            model_set: self.model_set,
            disabled_modules: self.disable_module,
            participants: self.participants,
            fresh_agent_db: self.fresh_agent_db,
            agent_db: self.agent_db,
        }
    }
}
