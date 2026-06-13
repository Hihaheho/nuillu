use std::time::Duration;

use nuillu_agent::AgentRunController;
use nuillu_blackboard::{
    Blackboard, BlackboardCommand, Bpm, ModulePolicy, ZeroReplicaWindowPolicy,
};
use nuillu_memory::{LinkedMemoryQuery, MemoryLinkDirection, MemoryLinkRelation};
use nuillu_module::{AmbientSensoryEntry, SensoryInput, SensoryInputMailbox, SensoryModality};
use nuillu_types::{MemoryIndex, ModuleId, ModuleInstanceId, ReplicaCapRange, ReplicaIndex};
use nuillu_visualizer_protocol::{
    ModuleSettingsView, VisualizerClientMessage, VisualizerCommand, VisualizerEvent,
    VisualizerTabId, ZeroReplicaWindowView, memory_page_from_records, run_runtime_action_id,
    stop_runtime_action_id,
};

use crate::SERVER_TAB_ID;
use crate::environment::ServerEnvironment;
use crate::gui::VisualizerHook;
use crate::runtime::set_runtime_running;
use crate::snapshot::{
    emit_visualizer_blackboard_snapshot, linked_memory_record_view, list_all_memories,
    memory_record_view,
};
use crate::state::{ModuleSettingsState, SceneState};

const SNAPSHOT_INTERVAL: Duration = Duration::from_millis(100);

pub(super) async fn drive_server_until_shutdown(
    visualizer: &mut VisualizerHook,
    tab_id: &VisualizerTabId,
    scene: &mut SceneState,
    module_settings: &mut ModuleSettingsState,
    sensory: &SensoryInputMailbox,
    env: &ServerEnvironment,
    run_controller: &AgentRunController,
) {
    publish_scene_snapshot(scene, sensory, visualizer, tab_id, env.clock.as_ref()).await;
    loop {
        if visualizer.shutdown_requested() {
            break;
        }
        while let Some(message) = visualizer.try_recv_command() {
            if handle_server_visualizer_message(
                message,
                visualizer,
                tab_id,
                scene,
                module_settings,
                sensory,
                env,
                run_controller,
            )
            .await
            {
                break;
            }
        }
        if visualizer.shutdown_requested() {
            break;
        }
        emit_visualizer_blackboard_snapshot(SERVER_TAB_ID, &env.blackboard, visualizer).await;
        tokio::time::sleep(SNAPSHOT_INTERVAL).await;
    }
}

#[allow(clippy::too_many_arguments)]
async fn handle_server_visualizer_message(
    message: VisualizerClientMessage,
    visualizer: &mut VisualizerHook,
    tab_id: &VisualizerTabId,
    scene: &mut SceneState,
    module_settings: &mut ModuleSettingsState,
    sensory: &SensoryInputMailbox,
    env: &ServerEnvironment,
    run_controller: &AgentRunController,
) -> bool {
    let command = match message {
        VisualizerClientMessage::Hello { .. } => return false,
        VisualizerClientMessage::InvokeAction { action_id } => {
            if action_id == run_runtime_action_id(tab_id) {
                resume_runtime(visualizer, tab_id, scene, sensory, env, run_controller).await;
            } else if action_id == stop_runtime_action_id(tab_id) {
                set_runtime_running(visualizer, tab_id, run_controller, false);
            }
            return false;
        }
        VisualizerClientMessage::Command { command } => command,
    };
    match command {
        VisualizerCommand::Shutdown => {
            visualizer.request_shutdown();
            true
        }
        VisualizerCommand::SendOneShotSensoryInput {
            tab_id: command_tab,
            input,
        } if command_tab == *tab_id => {
            if !run_controller.is_running() {
                resume_runtime(visualizer, tab_id, scene, sensory, env, run_controller).await;
            }
            let body = SensoryInput::OneShot {
                modality: SensoryModality::parse(input.modality),
                direction: input.direction,
                content: input.content,
                observed_at: env.clock.now(),
            };
            let _ = sensory.publish(body.clone()).await;
            visualizer.send_event(VisualizerEvent::SensoryInput {
                tab_id: tab_id.clone(),
                input: body,
            });
            false
        }
        VisualizerCommand::CreateAmbientSensoryRow {
            tab_id: command_tab,
            modality,
            content,
            disabled,
        } if command_tab == *tab_id => {
            scene.create_legacy_ambient(modality, content, disabled);
            persist_and_emit_scene(
                scene,
                visualizer,
                tab_id,
                sensory,
                env,
                run_controller.is_running(),
            )
            .await;
            false
        }
        VisualizerCommand::UpdateAmbientSensoryRow {
            tab_id: command_tab,
            row,
        } if command_tab == *tab_id => {
            scene.update_legacy_ambient(row);
            persist_and_emit_scene(
                scene,
                visualizer,
                tab_id,
                sensory,
                env,
                run_controller.is_running(),
            )
            .await;
            false
        }
        VisualizerCommand::RemoveAmbientSensoryRow {
            tab_id: command_tab,
            row_id,
        } if command_tab == *tab_id => {
            scene.remove_legacy_ambient(&row_id);
            persist_and_emit_scene(
                scene,
                visualizer,
                tab_id,
                sensory,
                env,
                run_controller.is_running(),
            )
            .await;
            false
        }
        VisualizerCommand::CreateSceneRow {
            tab_id: command_tab,
            kind,
        } if command_tab == *tab_id => {
            scene.create(kind);
            persist_and_emit_scene(
                scene,
                visualizer,
                tab_id,
                sensory,
                env,
                run_controller.is_running(),
            )
            .await;
            false
        }
        VisualizerCommand::UpdateSceneRow {
            tab_id: command_tab,
            row,
        } if command_tab == *tab_id => {
            scene.update(row);
            persist_and_emit_scene(
                scene,
                visualizer,
                tab_id,
                sensory,
                env,
                run_controller.is_running(),
            )
            .await;
            false
        }
        VisualizerCommand::RemoveSceneRow {
            tab_id: command_tab,
            kind,
            row_id,
        } if command_tab == *tab_id => {
            scene.remove(kind, &row_id);
            persist_and_emit_scene(
                scene,
                visualizer,
                tab_id,
                sensory,
                env,
                run_controller.is_running(),
            )
            .await;
            false
        }
        VisualizerCommand::SendScenePersonMessage {
            tab_id: command_tab,
            row_id,
            message,
        } if command_tab == *tab_id => {
            if !run_controller.is_running() {
                resume_runtime(visualizer, tab_id, scene, sensory, env, run_controller).await;
            }
            send_scene_person_message(scene, &row_id, message, visualizer, tab_id, sensory, env)
                .await;
            false
        }
        VisualizerCommand::SetModuleDisabled {
            tab_id: command_tab,
            module,
            disabled,
        } if command_tab == *tab_id => {
            set_module_disabled(&module, disabled, visualizer, tab_id, env).await;
            false
        }
        VisualizerCommand::SetModuleSettings {
            tab_id: command_tab,
            settings,
        } if command_tab == *tab_id => {
            if apply_visualizer_module_settings(
                tab_id,
                visualizer,
                &env.blackboard,
                settings.clone(),
            )
            .await
            {
                module_settings.upsert(settings);
                if let Err(error) = module_settings.save() {
                    visualizer.send_event(VisualizerEvent::Log {
                        tab_id: tab_id.clone(),
                        message: format!("failed to save module settings: {error}"),
                    });
                }
            }
            false
        }
        VisualizerCommand::ResetModuleSessionHistory {
            tab_id: command_tab,
            owner,
        } if command_tab == *tab_id => {
            match parse_module_owner(&owner) {
                Ok(owner_id) => {
                    let controller = run_controller.clone();
                    let events = visualizer.event_sender();
                    let tab_id = tab_id.clone();
                    tokio::task::spawn_local(async move {
                        match controller
                            .reset_module_session_history(owner_id.clone())
                            .await
                        {
                            Ok(reset) => {
                                events.send(VisualizerEvent::Log {
                                    tab_id,
                                    message: format!(
                                        "reset module session history for {}; deleted {} persisted session(s)",
                                        reset.owner, reset.deleted_sessions
                                    ),
                                });
                            }
                            Err(error) => {
                                events.send(VisualizerEvent::Log {
                                    tab_id,
                                    message: format!(
                                        "failed to reset module session history for {owner_id}: {error}"
                                    ),
                                });
                            }
                        }
                    });
                }
                Err(message) => {
                    visualizer.send_event(VisualizerEvent::Log {
                        tab_id: tab_id.clone(),
                        message: format!("invalid module owner {owner:?}: {message}"),
                    });
                }
            }
            false
        }
        VisualizerCommand::QueryMemory {
            tab_id: command_tab,
            query,
            limit,
        } if command_tab == *tab_id => {
            let records = env
                .memory
                .search(&nuillu_memory::MemoryQuery::text(query.clone(), limit))
                .await
                .map(|records| records.into_iter().map(memory_record_view).collect())
                .unwrap_or_default();
            visualizer.send_event(VisualizerEvent::MemoryQueryResult {
                tab_id: tab_id.clone(),
                query,
                records,
            });
            false
        }
        VisualizerCommand::FetchLinkedMemories {
            tab_id: command_tab,
            memory_index,
            relation_filter,
            limit,
        } if command_tab == *tab_id => {
            let relation_filter = relation_filter
                .into_iter()
                .filter_map(|relation| parse_memory_relation(&relation))
                .collect::<Vec<_>>();
            let records = env
                .memory
                .linked(&LinkedMemoryQuery {
                    memory_indexes: vec![MemoryIndex::new(memory_index.clone())],
                    relation_filter,
                    direction: MemoryLinkDirection::Both,
                    limit,
                })
                .await
                .map(|records| records.into_iter().map(linked_memory_record_view).collect())
                .unwrap_or_default();
            visualizer.send_event(VisualizerEvent::MemoryLinkedResult {
                tab_id: tab_id.clone(),
                memory_index,
                records,
            });
            false
        }
        VisualizerCommand::DeleteMemory {
            tab_id: command_tab,
            memory_index,
            page,
            per_page,
        } if command_tab == *tab_id => {
            let index = MemoryIndex::new(memory_index.clone());
            if let Err(error) = env.memory_caps.deleter().delete(&index).await {
                visualizer.send_event(VisualizerEvent::Log {
                    tab_id: tab_id.clone(),
                    message: format!("failed to delete memory {memory_index}: {error}"),
                });
            }
            let records = list_all_memories(env.memory.as_ref()).await;
            visualizer.send_event(VisualizerEvent::MemoryPage {
                tab_id: tab_id.clone(),
                page: memory_page_from_records(&records, page, per_page),
            });
            false
        }
        VisualizerCommand::ListMemories {
            tab_id: command_tab,
            page,
            per_page,
        } if command_tab == *tab_id => {
            let records = list_all_memories(env.memory.as_ref()).await;
            visualizer.send_event(VisualizerEvent::MemoryPage {
                tab_id: tab_id.clone(),
                page: memory_page_from_records(&records, page, per_page),
            });
            false
        }
        _ => false,
    }
}

fn parse_memory_relation(value: &str) -> Option<MemoryLinkRelation> {
    match value.trim().to_ascii_lowercase().as_str() {
        "related" => Some(MemoryLinkRelation::Related),
        "supports" => Some(MemoryLinkRelation::Supports),
        "contradicts" => Some(MemoryLinkRelation::Contradicts),
        "updates" => Some(MemoryLinkRelation::Updates),
        "corrects" => Some(MemoryLinkRelation::Corrects),
        "derived_from" | "derived-from" => Some(MemoryLinkRelation::DerivedFrom),
        _ => None,
    }
}

fn parse_module_owner(value: &str) -> Result<ModuleInstanceId, String> {
    let value = value.trim();
    if value.is_empty() {
        return Err("owner must not be empty".to_string());
    }
    let (module, replica) = if let Some((module, replica)) = value.rsplit_once('[') {
        let replica = replica
            .strip_suffix(']')
            .ok_or_else(|| "replica suffix must end with ']'".to_string())?;
        let replica = replica
            .parse::<u8>()
            .map_err(|_| format!("replica must be a u8 integer: {replica}"))?;
        (module, ReplicaIndex::new(replica))
    } else {
        (value, ReplicaIndex::ZERO)
    };
    let module =
        ModuleId::new(module.to_string()).map_err(|_| format!("invalid module id: {module}"))?;
    Ok(ModuleInstanceId::new(module, replica))
}

async fn set_module_disabled(
    module: &str,
    disabled: bool,
    visualizer: &VisualizerHook,
    tab_id: &VisualizerTabId,
    env: &ServerEnvironment,
) {
    let module_id = match ModuleId::new(module.to_string()) {
        Ok(module_id) => module_id,
        Err(_) => {
            visualizer.send_event(VisualizerEvent::Log {
                tab_id: tab_id.clone(),
                message: format!("invalid module id: {module}"),
            });
            return;
        }
    };
    env.blackboard
        .apply(BlackboardCommand::SetModuleForcedDisabled {
            module: module_id,
            disabled,
        })
        .await;
}

async fn persist_and_emit_scene(
    scene: &SceneState,
    visualizer: &mut VisualizerHook,
    tab_id: &VisualizerTabId,
    sensory: &SensoryInputMailbox,
    env: &ServerEnvironment,
    publish_snapshot: bool,
) {
    if let Err(error) = scene.save() {
        visualizer.send_event(VisualizerEvent::Log {
            tab_id: tab_id.clone(),
            message: format!("failed to save scene state: {error}"),
        });
    }
    env.caps.scene().set(scene.participants());
    emit_scene_state(scene, visualizer, tab_id);
    if publish_snapshot {
        publish_scene_snapshot(scene, sensory, visualizer, tab_id, env.clock.as_ref()).await;
    }
}

async fn resume_runtime(
    visualizer: &VisualizerHook,
    tab_id: &VisualizerTabId,
    scene: &SceneState,
    sensory: &SensoryInputMailbox,
    env: &ServerEnvironment,
    run_controller: &AgentRunController,
) {
    set_runtime_running(visualizer, tab_id, run_controller, true);
    publish_scene_snapshot(scene, sensory, visualizer, tab_id, env.clock.as_ref()).await;
}

pub(super) fn emit_scene_state(
    scene: &SceneState,
    visualizer: &VisualizerHook,
    tab_id: &VisualizerTabId,
) {
    visualizer.send_event(VisualizerEvent::SceneState {
        tab_id: tab_id.clone(),
        state: scene.view(),
    });
}

async fn publish_scene_snapshot(
    scene: &SceneState,
    sensory: &SensoryInputMailbox,
    visualizer: &VisualizerHook,
    tab_id: &VisualizerTabId,
    clock: &dyn nuillu_module::ports::Clock,
) {
    let entries = scene
        .derived_ambient()
        .into_iter()
        .map(|row| AmbientSensoryEntry {
            id: row.id,
            modality: SensoryModality::parse(&row.modality),
            content: row.content,
        })
        .collect::<Vec<_>>();
    let body = SensoryInput::AmbientSnapshot {
        entries,
        observed_at: clock.now(),
    };
    let _ = sensory.publish(body.clone()).await;
    visualizer.send_event(VisualizerEvent::SensoryInput {
        tab_id: tab_id.clone(),
        input: body,
    });
}

async fn send_scene_person_message(
    scene: &SceneState,
    row_id: &str,
    message: String,
    visualizer: &VisualizerHook,
    tab_id: &VisualizerTabId,
    sensory: &SensoryInputMailbox,
    env: &ServerEnvironment,
) {
    let message = message.trim();
    if message.is_empty() {
        return;
    }
    let Some(person) = scene.find_person(row_id) else {
        visualizer.send_event(VisualizerEvent::Log {
            tab_id: tab_id.clone(),
            message: format!("scene person row not found: {row_id}"),
        });
        return;
    };
    let name = person.name.trim();
    if name.is_empty() {
        visualizer.send_event(VisualizerEvent::Log {
            tab_id: tab_id.clone(),
            message: "cannot send a person message from an unnamed scene row".to_string(),
        });
        return;
    }
    let body = SensoryInput::OneShot {
        modality: SensoryModality::parse("audition"),
        direction: Some(name.to_string()),
        content: format!("{name} says, \"{message}\""),
        observed_at: env.clock.now(),
    };
    let _ = sensory.publish(body.clone()).await;
    visualizer.send_event(VisualizerEvent::SensoryInput {
        tab_id: tab_id.clone(),
        input: body,
    });
}

pub(super) async fn apply_persisted_module_settings(
    settings: &ModuleSettingsState,
    visualizer: &VisualizerHook,
    tab_id: &VisualizerTabId,
    blackboard: &Blackboard,
) {
    for setting in settings.iter() {
        apply_visualizer_module_settings(tab_id, visualizer, blackboard, setting.clone()).await;
    }
}

async fn apply_visualizer_module_settings(
    tab_id: &VisualizerTabId,
    visualizer: &VisualizerHook,
    blackboard: &Blackboard,
    settings: ModuleSettingsView,
) -> bool {
    let update = match build_module_policy_update(blackboard, &settings).await {
        Ok(update) => update,
        Err(message) => {
            visualizer.send_event(VisualizerEvent::Log {
                tab_id: tab_id.clone(),
                message,
            });
            return false;
        }
    };

    blackboard
        .apply(BlackboardCommand::SetModulePolicies {
            policies: vec![update],
        })
        .await;
    true
}

async fn build_module_policy_update(
    blackboard: &Blackboard,
    settings: &ModuleSettingsView,
) -> Result<(ModuleId, ModulePolicy), String> {
    let module = ModuleId::new(settings.module.clone())
        .map_err(|_| format!("invalid module id: {}", settings.module))?;
    if settings.replica_min > settings.replica_max {
        return Err(format!(
            "{} replica min {} exceeds max {}",
            settings.module, settings.replica_min, settings.replica_max
        ));
    }
    if !settings.bpm_min.is_finite()
        || !settings.bpm_max.is_finite()
        || settings.bpm_min <= 0.0
        || settings.bpm_max <= 0.0
    {
        return Err(format!(
            "{} BPM range must be positive and finite",
            settings.module
        ));
    }
    if settings.bpm_min > settings.bpm_max {
        return Err(format!(
            "{} BPM min {} exceeds max {}",
            settings.module, settings.bpm_min, settings.bpm_max
        ));
    }

    let (policy, capacity) = blackboard
        .read(|bb| {
            let policy = bb.module_policies().get(&module).cloned();
            let capacity = bb.module_replica_capacity(&module);
            (policy, capacity)
        })
        .await;
    let Some(mut policy) = policy else {
        return Err(format!(
            "module settings target is not registered: {}",
            settings.module
        ));
    };
    let capacity = capacity.unwrap_or_else(|| policy.max_active_replicas());
    if settings.replica_max > capacity {
        return Err(format!(
            "{} replica max {} exceeds hard cap {}",
            settings.module, settings.replica_max, capacity
        ));
    }

    policy.replicas_range = ReplicaCapRange::new(settings.replica_min, settings.replica_max)
        .map_err(|error| format!("{} invalid replica range: {error}", settings.module))?;
    policy.rate_limit_range = Bpm::range(settings.bpm_min, settings.bpm_max);
    policy.zero_replica_window = match settings.zero_replica_window {
        ZeroReplicaWindowView::Disabled => ZeroReplicaWindowPolicy::Disabled,
        ZeroReplicaWindowView::EveryControllerActivations { period } => {
            if period == 0 {
                return Err(format!(
                    "{} zero-window period must be greater than zero",
                    settings.module
                ));
            }
            ZeroReplicaWindowPolicy::EveryControllerActivations(period)
        }
    };

    Ok((module, policy))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_module_owner_accepts_zero_replica_display_form() {
        let owner = parse_module_owner("cognition-gate").expect("owner parses");

        assert_eq!(owner.module.as_str(), "cognition-gate");
        assert_eq!(owner.replica, ReplicaIndex::ZERO);
    }

    #[test]
    fn parse_module_owner_accepts_indexed_display_form() {
        let owner = parse_module_owner("predict[1]").expect("owner parses");

        assert_eq!(owner.module.as_str(), "predict");
        assert_eq!(owner.replica, ReplicaIndex::new(1));
    }

    #[test]
    fn parse_module_owner_rejects_invalid_display_form() {
        assert!(parse_module_owner("").is_err());
        assert!(parse_module_owner("Predict").is_err());
        assert!(parse_module_owner("predict[not-a-replica]").is_err());
        assert!(parse_module_owner("predict[1").is_err());
    }
}
