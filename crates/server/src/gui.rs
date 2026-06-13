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
    VisualizerAction, VisualizerClientMessage, VisualizerEvent, VisualizerServerMessage,
};

const VISUALIZER_GUI_BIN_ENV: &str = "NUILLU_VISUALIZER_GUI_BIN";
const VISUALIZER_GUI_PACKAGE: &str = "nuillu-visualizer-egui";
const VISUALIZER_GUI_BIN: &str = "nuillu-visualizer-gui";

#[derive(Clone, Debug)]
pub struct VisualizerEventSink {
    events: Sender<VisualizerServerMessage>,
}

impl VisualizerEventSink {
    pub fn new(events: Sender<VisualizerServerMessage>) -> Self {
        Self { events }
    }

    pub fn send(&self, event: VisualizerEvent) {
        let _ = self.events.send(VisualizerServerMessage::event(event));
    }
}

pub struct VisualizerHook {
    events: Sender<VisualizerServerMessage>,
    commands: Receiver<VisualizerClientMessage>,
    shutdown_requested: bool,
}

impl VisualizerHook {
    pub fn new(
        events: Sender<VisualizerServerMessage>,
        commands: Receiver<VisualizerClientMessage>,
    ) -> Self {
        Self {
            events,
            commands,
            shutdown_requested: false,
        }
    }

    pub fn event_sender(&self) -> VisualizerEventSink {
        VisualizerEventSink::new(self.events.clone())
    }

    pub fn send_event(&self, event: VisualizerEvent) {
        let _ = self.events.send(VisualizerServerMessage::event(event));
    }

    pub fn offer_action(&self, action: VisualizerAction) {
        let _ = self
            .events
            .send(VisualizerServerMessage::OfferAction { action });
    }

    pub fn revoke_action(&self, action_id: String) {
        let _ = self
            .events
            .send(VisualizerServerMessage::RevokeAction { action_id });
    }

    pub fn request_shutdown(&mut self) {
        self.shutdown_requested = true;
    }

    pub fn shutdown_requested(&self) -> bool {
        self.shutdown_requested
    }

    pub fn try_recv_command(&mut self) -> Option<VisualizerClientMessage> {
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

pub fn accept_visualizer_connection(
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

pub fn spawn_visualizer_gui(host: &str) -> anyhow::Result<Child> {
    let mut command = visualizer_gui_command(host)?;
    command
        .stdin(Stdio::null())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .spawn()
        .context("spawn nuillu-visualizer-gui")
}

fn visualizer_gui_command(host: &str) -> anyhow::Result<Command> {
    let override_path = env::var(VISUALIZER_GUI_BIN_ENV).ok();
    let path =
        resolve_visualizer_gui_binary(override_path.as_deref(), build_visualizer_gui_binary)?;
    let mut command = Command::new(path);
    command.arg("--host").arg(host);
    Ok(command)
}

fn resolve_visualizer_gui_binary(
    override_path: Option<&str>,
    build: impl FnOnce() -> anyhow::Result<PathBuf>,
) -> anyhow::Result<PathBuf> {
    if let Some(path) = override_path
        && !path.trim().is_empty()
    {
        return Ok(PathBuf::from(path));
    }
    build()
}

fn build_visualizer_gui_binary() -> anyhow::Result<PathBuf> {
    let output = Command::new("cargo")
        .arg("build")
        .arg("--release")
        .arg("-p")
        .arg(VISUALIZER_GUI_PACKAGE)
        .arg("--bin")
        .arg(VISUALIZER_GUI_BIN)
        .arg("--message-format=json-render-diagnostics")
        .current_dir(workspace_root())
        .output()
        .context("build nuillu-visualizer-gui")?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!(
            "cargo build for {VISUALIZER_GUI_BIN} failed with {}\n{}",
            output.status,
            stderr.trim_end()
        );
    }
    if !output.stderr.is_empty() {
        eprint!("{}", String::from_utf8_lossy(&output.stderr));
    }
    let stdout = std::str::from_utf8(&output.stdout).context("read cargo build stdout as utf-8")?;
    let executable = required_visualizer_executable_from_cargo_messages(stdout)?;
    if !executable.exists() {
        anyhow::bail!(
            "cargo build for {VISUALIZER_GUI_BIN} reported executable {}, but it does not exist",
            executable.display()
        );
    }
    Ok(executable)
}

fn required_visualizer_executable_from_cargo_messages(messages: &str) -> anyhow::Result<PathBuf> {
    visualizer_executable_from_cargo_messages(messages)?.ok_or_else(|| {
        anyhow::anyhow!(
            "cargo build for {VISUALIZER_GUI_BIN} succeeded but did not report an executable path"
        )
    })
}

fn visualizer_executable_from_cargo_messages(messages: &str) -> anyhow::Result<Option<PathBuf>> {
    for (line_index, line) in messages.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let message: serde_json::Value = serde_json::from_str(line).with_context(|| {
            format!(
                "parse cargo JSON message line {} while building {VISUALIZER_GUI_BIN}",
                line_index + 1
            )
        })?;
        if !is_visualizer_compiler_artifact(&message) {
            continue;
        }
        if let Some(executable) = message
            .get("executable")
            .and_then(serde_json::Value::as_str)
        {
            return Ok(Some(PathBuf::from(executable)));
        }
    }
    Ok(None)
}

fn is_visualizer_compiler_artifact(message: &serde_json::Value) -> bool {
    message.get("reason").and_then(serde_json::Value::as_str) == Some("compiler-artifact")
        && message
            .pointer("/target/name")
            .and_then(serde_json::Value::as_str)
            == Some(VISUALIZER_GUI_BIN)
        && message
            .pointer("/target/kind")
            .and_then(serde_json::Value::as_array)
            .is_some_and(|kinds| {
                kinds
                    .iter()
                    .any(|kind| kind.as_str().is_some_and(|kind| kind == "bin"))
            })
}

fn workspace_root() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("server crate should be two levels below workspace root")
}

pub fn wait_for_visualizer_exit_with_context(mut child: Child, context: &str) {
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

pub fn drain_child_stdio(child: &mut Child) {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cargo_messages_extract_visualizer_gui_executable() {
        let messages = r#"{"reason":"compiler-artifact","target":{"kind":["bin"],"name":"other-bin"},"executable":"/tmp/other-bin"}
{"reason":"compiler-artifact","target":{"kind":["bin"],"name":"nuillu-visualizer-gui"},"executable":"/custom-target/release/nuillu-visualizer-gui"}"#;

        assert_eq!(
            visualizer_executable_from_cargo_messages(messages).unwrap(),
            Some(PathBuf::from(
                "/custom-target/release/nuillu-visualizer-gui"
            ))
        );
    }

    #[test]
    fn cargo_messages_ignore_null_executable_and_other_bins() {
        let messages = r#"{"reason":"compiler-artifact","target":{"kind":["bin"],"name":"nuillu-visualizer-gui"},"executable":null}
{"reason":"compiler-artifact","target":{"kind":["bin"],"name":"other-bin"},"executable":"/tmp/other-bin"}"#;

        assert_eq!(
            visualizer_executable_from_cargo_messages(messages).unwrap(),
            None
        );
    }

    #[test]
    fn required_cargo_messages_error_when_visualizer_artifact_missing() {
        let messages = r#"{"reason":"compiler-artifact","target":{"kind":["lib"],"name":"nuillu_visualizer_egui"},"executable":null}"#;

        let error = required_visualizer_executable_from_cargo_messages(messages).unwrap_err();

        assert!(
            error
                .to_string()
                .contains("did not report an executable path")
        );
    }

    #[test]
    fn resolve_visualizer_gui_binary_prefers_non_empty_override() {
        let path = resolve_visualizer_gui_binary(Some("/custom/visualizer"), || {
            panic!("build should not be called when override is set")
        })
        .unwrap();

        assert_eq!(path, PathBuf::from("/custom/visualizer"));
    }

    #[test]
    fn resolve_visualizer_gui_binary_builds_when_override_is_empty() {
        let path =
            resolve_visualizer_gui_binary(Some("  "), || Ok(PathBuf::from("/built/gui"))).unwrap();

        assert_eq!(path, PathBuf::from("/built/gui"));
    }
}
