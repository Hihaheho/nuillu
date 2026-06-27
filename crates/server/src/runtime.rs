use std::{fs, net::TcpListener, time::Duration};

use anyhow::Context as _;
use nuillu_agent::{AgentEventLoopConfig, AgentRunController, run_controlled as run_agent};
use nuillu_blackboard::BlackboardCommand;
use nuillu_visualizer_protocol::{
    TabStatus, VisualizerAction, VisualizerEvent, VisualizerServerMessage, VisualizerServerPort,
    VisualizerTabId, run_runtime_action_id, stop_runtime_action_id,
};
use tokio::{runtime::Builder, task::LocalSet};

use crate::SERVER_TAB_ID;
use crate::commands::{
    apply_persisted_module_settings, drive_server_until_shutdown, emit_action_affordances,
    emit_recent_activity_rows, emit_scene_state,
};
use crate::config::ServerConfig;
use crate::environment::build_server_environment;
use crate::gui::{
    VisualizerHook, accept_visualizer_connection, spawn_visualizer_gui,
    wait_for_visualizer_exit_with_context,
};
use crate::llm_db_trace::emit_persisted_llm_transcripts;
use crate::registry::{full_agent_allocation, server_registry};
use crate::snapshot::emit_visualizer_blackboard_snapshot;
use crate::state::{ActionAffordanceState, ModuleSettingsState, SceneState};

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
    let mut child = spawn_visualizer_gui(&addr.to_string(), config.visualizer_bin.as_deref())?;
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
    let legacy_ambient_path = config.state_dir.join("ambient-sensory.json");
    let mut scene = SceneState::load(
        config.state_dir.join("scene-state.json"),
        &legacy_ambient_path,
        &config.participants,
    )?;
    scene.save()?;
    let mut module_settings =
        ModuleSettingsState::load(config.state_dir.join("module-settings.json"))?;
    let mut action_affordances =
        ActionAffordanceState::load(config.state_dir.join("action-affordances.json"))?;
    action_affordances.save()?;

    let active_modules = config.active_modules();
    let env = build_server_environment(
        &config,
        full_agent_allocation(&config.boot_config),
        visualizer.event_sender(),
    )
    .await?;
    let action_snapshot = env
        .caps
        .host_io()
        .action_affordance_writer()
        .set_all(
            config
                .boot_config
                .overlay_action_affordances(action_affordances.affordances()),
        )
        .await
        .context("seed action affordances")?;
    env.caps.scene().set(scene.participants());
    emit_scene_state(&scene, visualizer, &tab_id);
    emit_action_affordances(visualizer, &tab_id, action_snapshot.affordances);
    emit_recent_activity_rows(&env, visualizer, &tab_id).await;
    for module in config
        .disabled_modules
        .iter()
        .filter(|module| active_modules.contains(module))
    {
        env.blackboard
            .apply(BlackboardCommand::SetModuleForcedDisabled {
                module: module.module_id(),
                disabled: true,
            })
            .await;
    }

    emit_visualizer_blackboard_snapshot(SERVER_TAB_ID, &env.blackboard, visualizer).await;
    emit_persisted_llm_transcripts(
        &env.llm_transcript_store,
        SERVER_TAB_ID,
        &visualizer.event_sender(),
    )
    .await;

    let sensory = env.caps.host_io().sensory_input_mailbox();
    let (run_controller, run_control) = AgentRunController::new();
    set_runtime_running(visualizer, &tab_id, &run_controller, true);
    let mut restart_count = 0_u64;
    loop {
        let allocated = server_registry(
            &config.boot_config,
            &env.memory_caps,
            &env.policy_caps,
            &env.utterance_sink,
        )
        .build(&env.caps)
        .await?;
        apply_persisted_module_settings(&module_settings, visualizer, &tab_id, &env.blackboard)
            .await;

        let result = run_agent(
            allocated,
            AgentEventLoopConfig {
                idle_threshold: Duration::from_secs(1),
                max_activation_attempts: 5,
                dependency_idle_timeout: Duration::from_secs(2),
                dependency_hard_timeout: Duration::from_secs(10),
            },
            run_control.clone(),
            drive_server_until_shutdown(
                visualizer,
                &tab_id,
                &mut scene,
                &mut module_settings,
                &mut action_affordances,
                &config.boot_config,
                &sensory,
                &env,
                &run_controller,
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

pub(crate) fn set_runtime_running(
    visualizer: &VisualizerHook,
    tab_id: &VisualizerTabId,
    controller: &AgentRunController,
    running: bool,
) {
    if running {
        controller.resume();
        visualizer.send_event(VisualizerEvent::SetTabStatus {
            tab_id: tab_id.clone(),
            status: TabStatus::Running,
        });
        visualizer.revoke_action(run_runtime_action_id(tab_id));
        visualizer.offer_action(VisualizerAction::stop_runtime(tab_id.clone()));
    } else {
        controller.pause();
        visualizer.send_event(VisualizerEvent::SetTabStatus {
            tab_id: tab_id.clone(),
            status: TabStatus::Stopped,
        });
        visualizer.revoke_action(stop_runtime_action_id(tab_id));
        visualizer.offer_action(VisualizerAction::run_runtime(tab_id.clone()));
    }
}

#[cfg(test)]
mod tests {
    use std::sync::mpsc;

    use nuillu_visualizer_protocol::{VisualizerActionKind, VisualizerClientMessage};

    use super::*;

    #[test]
    fn set_runtime_running_updates_controller_status_and_actions() {
        let (event_tx, event_rx) = mpsc::channel();
        let (_command_tx, command_rx) = mpsc::channel::<VisualizerClientMessage>();
        let visualizer = VisualizerHook::new(event_tx, command_rx);
        let tab_id = VisualizerTabId::new("server");
        let (controller, _control) = AgentRunController::new();

        set_runtime_running(&visualizer, &tab_id, &controller, false);
        assert!(!controller.is_running());
        let messages = event_rx.try_iter().collect::<Vec<_>>();
        assert_eq!(messages.len(), 3);
        assert!(matches!(
            &messages[0],
            VisualizerServerMessage::Event {
                event: VisualizerEvent::SetTabStatus {
                    tab_id: actual_tab_id,
                    status: TabStatus::Stopped,
                    ..
                }
            } if actual_tab_id == &tab_id
        ));
        assert!(matches!(
            &messages[1],
            VisualizerServerMessage::RevokeAction { action_id }
                if action_id == &stop_runtime_action_id(&tab_id)
        ));
        assert!(matches!(
            &messages[2],
            VisualizerServerMessage::OfferAction { action }
                if action.id == run_runtime_action_id(&tab_id)
                    && action.kind == VisualizerActionKind::RunRuntime
        ));

        set_runtime_running(&visualizer, &tab_id, &controller, true);
        assert!(controller.is_running());
        let messages = event_rx.try_iter().collect::<Vec<_>>();
        assert_eq!(messages.len(), 3);
        assert!(matches!(
            &messages[0],
            VisualizerServerMessage::Event {
                event: VisualizerEvent::SetTabStatus {
                    tab_id: actual_tab_id,
                    status: TabStatus::Running,
                    ..
                }
            } if actual_tab_id == &tab_id
        ));
        assert!(matches!(
            &messages[1],
            VisualizerServerMessage::RevokeAction { action_id }
                if action_id == &run_runtime_action_id(&tab_id)
        ));
        assert!(matches!(
            &messages[2],
            VisualizerServerMessage::OfferAction { action }
                if action.id == stop_runtime_action_id(&tab_id)
                    && action.kind == VisualizerActionKind::StopRuntime
        ));
    }
}
