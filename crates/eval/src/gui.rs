use std::{net::TcpListener, process::Child, time::Duration};

use anyhow::Context as _;
use nuillu_server::{
    accept_visualizer_connection, drain_child_stdio, spawn_visualizer_gui,
    wait_for_visualizer_exit_with_context,
};
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

fn wait_for_visualizer_exit(child: Child) {
    wait_for_visualizer_exit_with_context(child, "eval finished");
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
