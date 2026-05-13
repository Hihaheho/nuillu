use std::{
    env,
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

pub fn run_suite_with_visualizer(config: RunnerConfig) -> anyhow::Result<()> {
    let listener = TcpListener::bind(("127.0.0.1", 0)).context("bind visualizer RPC listener")?;
    let addr = listener
        .local_addr()
        .context("read visualizer RPC listener address")?;
    eprintln!("visualizer RPC listening on {addr}");
    listener
        .set_nonblocking(true)
        .context("set visualizer RPC listener nonblocking")?;
    let mut child = spawn_visualizer_gui(&addr.to_string())?;
    eprintln!("visualizer process started pid={}", child.id());
    let stream = accept_visualizer_connection(&listener, &mut child)?;
    eprintln!("visualizer RPC connected");
    let port = VisualizerServerPort::from_stream(stream).context("open visualizer RPC port")?;

    port.send(VisualizerServerMessage::hello())
        .context("send visualizer protocol hello")?;
    wait_for_client_hello(&port)?;
    for (tab_id, title) in visualizer_planned_tabs(&config)? {
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

pub(crate) fn accept_visualizer_connection(
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

pub(crate) fn spawn_visualizer_gui(host: &str) -> anyhow::Result<Child> {
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

fn current_exe_sibling(name: &str) -> Option<PathBuf> {
    let exe = env::current_exe().ok()?;
    Some(exe.parent()?.join(name))
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

fn workspace_root() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("eval crate should be two levels below workspace root")
}

pub(crate) fn wait_for_visualizer_exit(child: Child) {
    wait_for_visualizer_exit_with_context(child, "eval finished");
}

pub(crate) fn wait_for_visualizer_exit_with_context(mut child: Child, context: &str) {
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
