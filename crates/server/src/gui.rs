use std::{
    env,
    io::Read as _,
    net::{TcpListener, TcpStream},
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    sync::mpsc::{Receiver, Sender},
    thread,
    time::Duration,
};

use anyhow::Context as _;
use nuillu_visualizer_protocol::{
    VisualizerClientMessage, VisualizerEvent, VisualizerServerMessage,
};

#[derive(Clone, Debug)]
pub(super) struct VisualizerEventSink {
    events: Sender<VisualizerServerMessage>,
}

impl VisualizerEventSink {
    pub(super) fn new(events: Sender<VisualizerServerMessage>) -> Self {
        Self { events }
    }

    pub(super) fn send(&self, event: VisualizerEvent) {
        let _ = self.events.send(VisualizerServerMessage::event(event));
    }
}

pub(super) struct VisualizerHook {
    events: Sender<VisualizerServerMessage>,
    commands: Receiver<VisualizerClientMessage>,
    shutdown_requested: bool,
}

impl VisualizerHook {
    pub(super) fn new(
        events: Sender<VisualizerServerMessage>,
        commands: Receiver<VisualizerClientMessage>,
    ) -> Self {
        Self {
            events,
            commands,
            shutdown_requested: false,
        }
    }

    pub(super) fn event_sender(&self) -> VisualizerEventSink {
        VisualizerEventSink::new(self.events.clone())
    }

    pub(super) fn send_event(&self, event: VisualizerEvent) {
        let _ = self.events.send(VisualizerServerMessage::event(event));
    }

    pub(super) fn request_shutdown(&mut self) {
        self.shutdown_requested = true;
    }

    pub(super) fn shutdown_requested(&self) -> bool {
        self.shutdown_requested
    }

    pub(super) fn try_recv_command(&mut self) -> Option<VisualizerClientMessage> {
        match self.commands.try_recv() {
            Ok(message) => Some(message),
            Err(std::sync::mpsc::TryRecvError::Empty) => None,
            Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                self.request_shutdown();
                None
            }
        }
    }
}

pub(super) fn accept_visualizer_connection(
    listener: &TcpListener,
    child: &mut Child,
) -> anyhow::Result<TcpStream> {
    loop {
        match listener.accept() {
            Ok((stream, _)) => return Ok(stream),
            Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                if let Some(status) = child
                    .try_wait()
                    .context("poll visualizer GUI process before RPC connection")?
                {
                    anyhow::bail!("visualizer GUI exited before connecting: {status}");
                }
                thread::sleep(Duration::from_millis(50));
            }
            Err(error) => return Err(error).context("accept visualizer RPC connection"),
        }
    }
}

pub(super) fn spawn_visualizer_gui(host: &str) -> anyhow::Result<Child> {
    let mut command = visualizer_gui_command(host)?;
    command
        .stdin(Stdio::null())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .spawn()
        .context("spawn nuillu-visualizer-gui")
}

fn visualizer_gui_command(host: &str) -> anyhow::Result<Command> {
    if let Ok(path) = env::var("NUILLU_VISUALIZER_GUI_BIN")
        && !path.trim().is_empty()
    {
        let mut command = Command::new(path);
        command.arg("--host").arg(host);
        return Ok(command);
    }

    build_visualizer_gui_binary()?;
    if let Some(path) = existing_visualizer_binary() {
        let mut command = Command::new(path);
        command.arg("--host").arg(host);
        return Ok(command);
    }

    anyhow::bail!(
        "built nuillu-visualizer-gui but could not find sibling binary {}",
        visualizer_binary_name()
    )
}

fn build_visualizer_gui_binary() -> anyhow::Result<()> {
    let status = Command::new("cargo")
        .arg("build")
        .arg("-p")
        .arg("nuillu-visualizer-egui")
        .arg("--bin")
        .arg("nuillu-visualizer-gui")
        .current_dir(workspace_root())
        .status()
        .context("build nuillu-visualizer-gui")?;
    if status.success() {
        Ok(())
    } else {
        anyhow::bail!("cargo build for nuillu-visualizer-gui failed with {status}")
    }
}

fn visualizer_binary_name() -> &'static str {
    if cfg!(windows) {
        "nuillu-visualizer-gui.exe"
    } else {
        "nuillu-visualizer-gui"
    }
}

fn existing_visualizer_binary() -> Option<PathBuf> {
    [
        current_exe_sibling(visualizer_binary_name()),
        Some(
            workspace_root()
                .join("target")
                .join("debug")
                .join(visualizer_binary_name()),
        ),
    ]
    .into_iter()
    .flatten()
    .find(|path| path.exists())
}

fn current_exe_sibling(name: &str) -> Option<PathBuf> {
    let exe = env::current_exe().ok()?;
    Some(exe.parent()?.join(name))
}

fn workspace_root() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("server crate should be two levels below workspace root")
}

pub(super) fn wait_for_visualizer_exit_with_context(mut child: Child, context: &str) {
    match child.try_wait() {
        Ok(Some(_)) => return,
        Ok(None) => {
            eprintln!("{context}; visualizer remains open until its window is closed");
        }
        Err(error) => {
            eprintln!("failed to poll visualizer process after {context}: {error}");
            return;
        }
    }
    let _ = child.wait();
}

#[allow(dead_code)]
fn drain_child_stdio(child: &mut Child) {
    if let Some(stdout) = child.stdout.as_mut() {
        let mut output = String::new();
        let _ = stdout.read_to_string(&mut output);
        if !output.trim().is_empty() {
            eprintln!("visualizer stdout:\n{output}");
        }
    }
    if let Some(stderr) = child.stderr.as_mut() {
        let mut output = String::new();
        let _ = stderr.read_to_string(&mut output);
        if !output.trim().is_empty() {
            eprintln!("visualizer stderr:\n{output}");
        }
    }
}
