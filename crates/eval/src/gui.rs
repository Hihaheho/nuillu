use std::{
    io::Read as _,
    net::{TcpListener, TcpStream},
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    thread,
    time::Duration,
};

use anyhow::Context as _;
use nuillu_visualizer_protocol::{
    START_SUITE_ACTION_ID, TabStatus, VisualizerAction, VisualizerClientMessage, VisualizerEvent,
    VisualizerServerMessage, VisualizerServerPort,
};
use tokio::runtime::Builder;

use crate::{
    RunnerConfig, RunnerError, RunnerHooks, VisualizerHook, run_suite_with_hooks,
    runner::visualizer_planned_tabs,
};

const VISUALIZER_GUI_PACKAGE: &str = "nuillu-visualizer-egui";

pub fn run_suite_with_visualizer(config: RunnerConfig) -> anyhow::Result<()> {
    if config.trials.get() > 1 {
        anyhow::bail!("--gui does not support --trials > 1");
    }

    let planned_tabs = visualizer_planned_tabs(&config)?;
    let listener = TcpListener::bind(("127.0.0.1", 0)).context("bind visualizer RPC listener")?;
    let addr = listener
        .local_addr()
        .context("read visualizer RPC listener address")?;
    eprintln!("visualizer RPC listening on {addr}");
    listener
        .set_nonblocking(true)
        .context("set visualizer RPC listener nonblocking")?;
    let mut child = spawn_visualizer_gui(&addr.to_string(), None)?;
    eprintln!("visualizer process started pid={}", child.id());
    let stream = accept_visualizer_connection(&listener, &mut child)?;
    eprintln!("visualizer RPC connected");
    let port = VisualizerServerPort::from_stream(stream).context("open visualizer RPC port")?;

    port.send(VisualizerServerMessage::hello())
        .context("send visualizer protocol hello")?;
    wait_for_client_hello(&port)?;
    for (tab_id, title) in planned_tabs {
        port.send(VisualizerServerMessage::event(VisualizerEvent::OpenTab {
            tab_id: tab_id.clone(),
            title,
        }))
        .context("send visualizer planned tab")?;
        port.send(VisualizerServerMessage::event(
            VisualizerEvent::SetTabStatus {
                tab_id,
                status: TabStatus::Stopped,
            },
        ))
        .context("send visualizer planned tab status")?;
    }
    port.send(VisualizerServerMessage::OfferAction {
        action: VisualizerAction::start_suite(),
    })
    .context("offer visualizer start suite action")?;

    if !wait_for_start_suite(&port, &mut child)? {
        wait_for_visualizer_exit(child);
        return Ok(());
    }
    port.send(VisualizerServerMessage::RevokeAction {
        action_id: START_SUITE_ACTION_ID.to_string(),
    })
    .context("revoke visualizer start suite action")?;

    let (command_rx, event_tx) = port.into_channels();
    let runtime = Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|source| RunnerError::Driver {
            path: config.cases_root.clone(),
            message: source.to_string(),
        })?;
    let mut hooks = RunnerHooks::with_visualizer(VisualizerHook::new(event_tx, command_rx));
    let result = runtime.block_on(run_suite_with_hooks(&config, &mut hooks));
    wait_for_visualizer_exit(child);
    result?;
    Ok(())
}

fn wait_for_client_hello(port: &VisualizerServerPort) -> anyhow::Result<()> {
    match port.recv() {
        Ok(VisualizerClientMessage::Hello { .. }) => Ok(()),
        Ok(_) => Ok(()),
        Err(error) => Err(error).context("wait for visualizer protocol hello"),
    }
}

fn wait_for_start_suite(port: &VisualizerServerPort, child: &mut Child) -> anyhow::Result<bool> {
    eprintln!("waiting for visualizer Start Suite action");
    loop {
        if let Some(status) = child
            .try_wait()
            .context("poll visualizer GUI process while waiting for Start Suite")?
        {
            eprintln!("visualizer process exited while waiting for Start Suite: {status}");
            drain_child_stdio(child);
            return Ok(false);
        }
        match port.recv_timeout(Duration::from_millis(50)) {
            Ok(Some(VisualizerClientMessage::InvokeAction { action_id }))
                if action_id == START_SUITE_ACTION_ID =>
            {
                eprintln!("visualizer Start Suite action received");
                return Ok(true);
            }
            Ok(Some(VisualizerClientMessage::Command {
                command: nuillu_visualizer_protocol::VisualizerCommand::Shutdown,
            })) => {
                eprintln!("visualizer requested shutdown before suite start");
                return Ok(false);
            }
            Ok(Some(_)) | Ok(None) => {}
            Err(nuillu_visualizer_protocol::VisualizerProtocolError::Disconnected) => {
                eprintln!("visualizer RPC disconnected before suite start");
                report_visualizer_exit(child);
                return Ok(false);
            }
            Err(error) => return Err(error).context("wait for visualizer start suite action"),
        }
    }
}

fn wait_for_visualizer_exit(child: Child) {
    wait_for_visualizer_exit_with_context(child, "eval finished");
}

fn accept_visualizer_connection(
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

fn spawn_visualizer_gui(host: &str, binary_path: Option<&Path>) -> anyhow::Result<Child> {
    let mut command = visualizer_gui_command(host, binary_path)?;
    command
        .stdin(Stdio::null())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .spawn()
        .context("spawn visualizer GUI")
}

fn visualizer_gui_command(host: &str, binary_path: Option<&Path>) -> anyhow::Result<Command> {
    let path = resolve_visualizer_gui_binary(binary_path, build_visualizer_gui_binary)?;
    let mut command = Command::new(path);
    command.arg("--host").arg(host);
    Ok(command)
}

fn resolve_visualizer_gui_binary(
    override_path: Option<&Path>,
    build: impl FnOnce() -> anyhow::Result<PathBuf>,
) -> anyhow::Result<PathBuf> {
    if let Some(path) = override_path
        && !path.as_os_str().is_empty()
    {
        return Ok(path.to_path_buf());
    }
    build()
}

fn build_visualizer_gui_binary() -> anyhow::Result<PathBuf> {
    let output = Command::new("cargo")
        .arg("build")
        .arg("--release")
        .arg("-p")
        .arg(VISUALIZER_GUI_PACKAGE)
        .arg("--message-format=json-render-diagnostics")
        .current_dir(workspace_root())
        .output()
        .context("build visualizer GUI")?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!(
            "cargo build for {VISUALIZER_GUI_PACKAGE} failed with {}\n{}",
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
            "cargo build for {VISUALIZER_GUI_PACKAGE} reported executable {}, but it does not exist",
            executable.display()
        );
    }
    Ok(executable)
}

fn required_visualizer_executable_from_cargo_messages(messages: &str) -> anyhow::Result<PathBuf> {
    visualizer_executable_from_cargo_messages(messages)?.ok_or_else(|| {
        anyhow::anyhow!(
            "cargo build for {VISUALIZER_GUI_PACKAGE} succeeded but did not report an executable path"
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
                "parse cargo JSON message line {} while building {VISUALIZER_GUI_PACKAGE}",
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
            == Some(VISUALIZER_GUI_PACKAGE)
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
        .expect("eval crate should be two levels below workspace root")
}

fn wait_for_visualizer_exit_with_context(mut child: Child, context: &str) {
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

fn report_visualizer_exit(child: &mut Child) {
    match child.try_wait() {
        Ok(Some(status)) => {
            eprintln!("visualizer process status: {status}");
            drain_child_stdio(child);
        }
        Ok(None) => {
            eprintln!(
                "visualizer process is still running after RPC disconnect; close its window to exit"
            );
        }
        Err(error) => {
            eprintln!("failed to poll visualizer process: {error}");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cargo_messages_extract_visualizer_gui_executable() {
        let messages = r#"{"reason":"compiler-artifact","target":{"kind":["bin"],"name":"other-bin"},"executable":"/tmp/other-bin"}
{"reason":"compiler-artifact","target":{"kind":["bin"],"name":"nuillu-visualizer-egui"},"executable":"/custom-target/release/nuillu-visualizer-egui"}"#;

        assert_eq!(
            visualizer_executable_from_cargo_messages(messages).unwrap(),
            Some(PathBuf::from(
                "/custom-target/release/nuillu-visualizer-egui"
            ))
        );
    }

    #[test]
    fn cargo_messages_ignore_null_executable_and_other_bins() {
        let messages = r#"{"reason":"compiler-artifact","target":{"kind":["bin"],"name":"nuillu-visualizer-egui"},"executable":null}
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
        let override_path = Path::new("/custom/visualizer");
        let path = resolve_visualizer_gui_binary(Some(override_path), || {
            panic!("build should not be called when override is set")
        })
        .unwrap();

        assert_eq!(path, PathBuf::from("/custom/visualizer"));
    }

    #[test]
    fn resolve_visualizer_gui_binary_builds_when_override_is_empty() {
        let path =
            resolve_visualizer_gui_binary(Some(Path::new("")), || Ok(PathBuf::from("/built/gui")))
                .unwrap();

        assert_eq!(path, PathBuf::from("/built/gui"));
    }
}
