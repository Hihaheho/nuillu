use std::{fs, net::TcpListener, time::Duration};

use anyhow::Context as _;
use nuillu_agent::{AgentEventLoopConfig, run as run_agent};
use nuillu_blackboard::BlackboardCommand;
use nuillu_visualizer_protocol::{
    TabStatus, VisualizerEvent, VisualizerServerMessage, VisualizerServerPort, VisualizerTabId,
};
use tokio::{runtime::Builder, task::LocalSet};

use crate::SERVER_TAB_ID;
use crate::commands::{apply_persisted_module_settings, drive_server_until_shutdown};
use crate::config::{DEFAULT_MODULES, ServerConfig};
use crate::environment::build_server_environment;
use crate::gui::{
    VisualizerHook, accept_visualizer_connection, spawn_visualizer_gui,
    wait_for_visualizer_exit_with_context,
};
use crate::registry::{full_agent_allocation, server_registry};
use crate::snapshot::{emit_visualizer_blackboard_snapshot, emit_visualizer_memory_page};
use crate::state::{AmbientRows, ModuleSettingsState};

const SERVER_TITLE: &str = "nuillu-server";

pub fn run_server_with_visualizer(config: ServerConfig) -> anyhow::Result<()> {
    fs::create_dir_all(&config.state_dir)
        .with_context(|| format!("create state dir {}", config.state_dir.display()))?;
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
    let _ = port.recv();

    let tab_id = VisualizerTabId::new(SERVER_TAB_ID.to_string());
    port.send(VisualizerServerMessage::event(VisualizerEvent::OpenTab {
        tab_id: tab_id.clone(),
        title: SERVER_TITLE.to_string(),
    }))
    .context("open server tab")?;
    port.send(VisualizerServerMessage::event(
        VisualizerEvent::SetTabStatus {
            tab_id,
            status: TabStatus::Running,
        },
    ))
    .context("set server tab status")?;

    let (command_rx, event_tx) = port.into_channels();
    let runtime = Builder::new_current_thread()
        .enable_all()
        .build()
        .context("build server tokio runtime")?;
    let local = LocalSet::new();
    let mut visualizer = VisualizerHook::new(event_tx, command_rx);
    let result = runtime.block_on(local.run_until(run_server(config, &mut visualizer)));
    if let Err(error) = &result {
        eprintln!("nuillu-server runtime failed: {error:#}");
        let tab_id = VisualizerTabId::new(SERVER_TAB_ID.to_string());
        visualizer.send_event(VisualizerEvent::Log {
            tab_id: tab_id.clone(),
            message: format!("nuillu-server runtime failed: {error:#}"),
        });
        visualizer.send_event(VisualizerEvent::SetTabStatus {
            tab_id,
            status: TabStatus::Invalid,
        });
    }
    wait_for_visualizer_exit_with_context(
        child,
        if result.is_ok() {
            "nuillu-server runtime stopped"
        } else {
            "nuillu-server runtime failed"
        },
    );
    result
}

async fn run_server(config: ServerConfig, visualizer: &mut VisualizerHook) -> anyhow::Result<()> {
    let tab_id = VisualizerTabId::new(SERVER_TAB_ID.to_string());
    visualizer.send_event(VisualizerEvent::Log {
        tab_id: tab_id.clone(),
        message: format!("nuillu-server session_id={}", config.session_id),
    });
    let mut ambient = AmbientRows::load(config.state_dir.join("ambient-sensory.json"))?;
    let mut module_settings =
        ModuleSettingsState::load(config.state_dir.join("module-settings.json"))?;
    visualizer.send_event(VisualizerEvent::AmbientSensoryRows {
        tab_id: tab_id.clone(),
        rows: ambient.rows.clone(),
    });

    let modules = DEFAULT_MODULES.to_vec();
    let env = build_server_environment(
        &config,
        full_agent_allocation(&modules),
        visualizer.event_sender(),
    )
    .await?;
    env.caps.scene().set(
        config
            .participants
            .iter()
            .map(nuillu_module::Participant::new),
    );
    for module in &config.disabled_modules {
        env.blackboard
            .apply(BlackboardCommand::SetModuleForcedDisabled {
                module: module.module_id(),
                disabled: true,
            })
            .await;
    }

    emit_visualizer_blackboard_snapshot(SERVER_TAB_ID, &env.blackboard, visualizer).await;
    emit_visualizer_memory_page(
        SERVER_TAB_ID,
        visualizer,
        &env.blackboard,
        env.memory.as_ref(),
        0,
        25,
    )
    .await;

    let sensory = env.caps.host_io().sensory_input_mailbox();
    let allocation_updates = env.caps.host_io().allocation_updated_mailbox();
    let mut restart_count = 0_u64;
    loop {
        let allocated = server_registry(
            &modules,
            &env.memory_caps,
            &env.policy_caps,
            &env.utterance_sink,
        )
        .build(&env.caps)
        .await?;
        apply_persisted_module_settings(
            &module_settings,
            visualizer,
            &tab_id,
            &env.blackboard,
            &allocation_updates,
        )
        .await;

        let result = run_agent(
            allocated,
            AgentEventLoopConfig {
                idle_threshold: Duration::from_secs(1),
                activate_retries: 2,
            },
            drive_server_until_shutdown(
                visualizer,
                &tab_id,
                &mut ambient,
                &mut module_settings,
                &sensory,
                &allocation_updates,
                &env,
            ),
        )
        .await;

        match result {
            Ok(()) if visualizer.shutdown_requested() => break,
            Ok(()) => {
                restart_count = restart_count.saturating_add(1);
                let message = format!(
                    "agent runtime ended without a GUI shutdown; restarting attempt={restart_count}"
                );
                eprintln!("nuillu-server {message}");
                visualizer.send_event(VisualizerEvent::Log {
                    tab_id: tab_id.clone(),
                    message,
                });
            }
            Err(error) => {
                restart_count = restart_count.saturating_add(1);
                let message =
                    format!("agent runtime error; restarting attempt={restart_count}: {error}");
                eprintln!("nuillu-server {message}");
                visualizer.send_event(VisualizerEvent::Log {
                    tab_id: tab_id.clone(),
                    message,
                });
            }
        }

        if visualizer.shutdown_requested() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(500)).await;
    }
    visualizer.send_event(VisualizerEvent::SetTabStatus {
        tab_id,
        status: TabStatus::Stopped,
    });
    Ok(())
}
