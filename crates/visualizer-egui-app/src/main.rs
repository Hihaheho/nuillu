use std::{
    path::PathBuf,
    sync::mpsc::{Receiver, Sender, TryRecvError},
    time::Duration,
};

use clap::{Args as ClapArgs, Parser};
use nuillu_server::{
    ModuleId, Server, ServerRunOptions, install_lutum_trace_subscriber,
    load_server_config_from_options,
};
use nuillu_visualizer_egui::{Visualizer, VisualizerConfig};
use nuillu_visualizer_protocol::{
    VisualizerClientMessage, VisualizerClientPort, VisualizerCommand, VisualizerServerMessage,
};

const SERVER_MESSAGES_PER_FRAME: usize = 256;

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
        let runtime = Server::new(config).spawn()?;
        let (server_messages, client_messages) = runtime.visualizer_channels();
        (server_messages, client_messages, Some(runtime))
    };
    let native_options = eframe::NativeOptions {
        viewport: eframe::egui::ViewportBuilder::default()
            .with_active(true)
            .with_visible(true),
        ..eframe::NativeOptions::default()
    };
    let result = eframe::run_native(
        "Nuillu Visualizer",
        native_options,
        Box::new(|_cc| {
            Ok(Box::new(StandaloneVisualizerApp::new(
                server_messages,
                client_messages,
                true,
            )))
        }),
    );
    if let Some(runtime) = embedded_runtime {
        let _ = runtime.shutdown();
        match runtime.join_timeout(Duration::from_secs(2)) {
            Ok(true) => {}
            Ok(false) => {
                eprintln!("nuillu-visualizer-egui server runtime did not stop within 2s; exiting");
            }
            Err(error) => {
                eprintln!("nuillu-visualizer-egui server runtime join failed: {error:#}");
            }
        }
    }
    result?;
    Ok(())
}

struct StandaloneVisualizerApp {
    visualizer: Visualizer,
    server_messages: Receiver<VisualizerServerMessage>,
    client_messages: Sender<VisualizerClientMessage>,
    remote: bool,
}

impl StandaloneVisualizerApp {
    fn new(
        server_messages: Receiver<VisualizerServerMessage>,
        client_messages: Sender<VisualizerClientMessage>,
        remote: bool,
    ) -> Self {
        Self {
            visualizer: Visualizer::with_config(
                eframe::egui::Id::new("nuillu-visualizer"),
                VisualizerConfig::standalone(),
            ),
            server_messages,
            client_messages,
            remote,
        }
    }

    fn drain_server_messages(&mut self, ctx: &eframe::egui::Context) {
        let mut drained = 0;
        loop {
            if drained >= SERVER_MESSAGES_PER_FRAME {
                ctx.request_repaint();
                break;
            }
            match self.server_messages.try_recv() {
                Ok(message) => {
                    self.visualizer.apply_server_message(message);
                    drained += 1;
                }
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => {
                    if self.remote {
                        self.visualizer.mark_disconnected();
                    }
                    break;
                }
            }
        }
    }
}

impl eframe::App for StandaloneVisualizerApp {
    fn ui(&mut self, ui: &mut eframe::egui::Ui, _frame: &mut eframe::Frame) {
        self.drain_server_messages(ui.ctx());
        for message in self.visualizer.show(ui).into_messages() {
            if let Err(error) = self.client_messages.send(message) {
                self.visualizer
                    .record_send_failure(format!("failed to send visualizer message: {error}"));
                break;
            }
        }
    }

    fn auto_save_interval(&self) -> Duration {
        Duration::from_millis(1500)
    }

    fn on_exit(&mut self) {
        let _ = self.client_messages.send(VisualizerClientMessage::Command {
            command: VisualizerCommand::Shutdown,
        });
    }
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
    #[arg(long = "disable-module", value_parser = parse_module_id, value_name = "MODULE")]
    disable_module: Vec<ModuleId>,

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

fn parse_module_id(value: &str) -> Result<ModuleId, String> {
    ModuleId::new(value).map_err(|error| error.to_string())
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
