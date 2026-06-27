use std::{collections::HashSet, time::Duration};

use chrono::{DateTime, Utc};
use lutum_libsql_adapter::{
    AmbientSensorySnapshotRecord, ExternalActionEventRecord, ExternalActionEventStatus,
    NewAmbientSensorySnapshot, NewOneShotSensoryInput, OneShotSensoryInputRecord,
    UtteranceEventKind, UtteranceEventRecord,
};
use nuillu_agent::AgentRunController;
use nuillu_blackboard::{
    Blackboard, BlackboardCommand, Bpm, ModulePolicy, ZeroReplicaWindowPolicy,
};
use nuillu_memory::{LinkedMemoryQuery, MemoryLinkDirection, MemoryLinkRelation, MemoryQuery};
use nuillu_module::{
    ActionAffordance, AmbientSensoryEntry, SensoryInput, SensoryInputMailbox, SensoryModality,
};
use nuillu_types::{MemoryIndex, ModuleId, ModuleInstanceId, ReplicaCapRange, ReplicaIndex};
use nuillu_visualizer_protocol::{
    AmbientSensorySnapshotRowView, ExternalActionEventRowView, ExternalActionEventStatusView,
    MemoryRecordScope, ModuleSettingsView, OneShotSensoryInputRowView, UtteranceEventKindView,
    UtteranceEventRowView, VisualizerClientMessage, VisualizerCommand, VisualizerEvent,
    VisualizerTabId, ZeroReplicaWindowView, run_runtime_action_id, stop_runtime_action_id,
};

use crate::SERVER_TAB_ID;
use crate::config::ServerBootConfig;
use crate::environment::ServerEnvironment;
use crate::gui::VisualizerHook;
use crate::runtime::set_runtime_running;
use crate::snapshot::{
    emit_visualizer_blackboard_snapshot, linked_memory_record_view, memory_record_view,
};
use crate::state::{ActionAffordanceState, ModuleSettingsState, SceneState};

const SNAPSHOT_INTERVAL: Duration = Duration::from_millis(100);
const RECENT_ACTIVITY_ROW_LIMIT: usize = 512;

pub(super) async fn drive_server_until_shutdown(
    visualizer: &mut VisualizerHook,
    tab_id: &VisualizerTabId,
    scene: &mut SceneState,
    module_settings: &mut ModuleSettingsState,
    action_affordances: &mut ActionAffordanceState,
    boot_config: &ServerBootConfig,
    sensory: &SensoryInputMailbox,
    env: &ServerEnvironment,
    run_controller: &AgentRunController,
) {
    publish_scene_snapshot(scene, sensory, visualizer, tab_id, env).await;
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
                action_affordances,
                boot_config,
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
    action_affordances: &mut ActionAffordanceState,
    boot_config: &ServerBootConfig,
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
            record_sensory_input(visualizer, tab_id, env, &body).await;
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
        VisualizerCommand::SaveSceneState {
            tab_id: command_tab,
            state,
        } if command_tab == *tab_id => {
            scene.replace(state);
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
        VisualizerCommand::SetAgentActionAffordances {
            tab_id: command_tab,
            affordances,
        } if command_tab == *tab_id => {
            apply_action_affordance_set(
                action_affordances,
                boot_config,
                affordances,
                visualizer,
                tab_id,
                env,
            )
            .await;
            false
        }
        VisualizerCommand::UpsertAgentActionAffordance {
            tab_id: command_tab,
            affordance,
        } if command_tab == *tab_id => {
            match affordance.validate() {
                Ok(()) => {}
                Err(error) => {
                    visualizer.send_event(VisualizerEvent::Log {
                        tab_id: tab_id.clone(),
                        message: format!("rejected action affordance update: {error}"),
                    });
                    return false;
                }
            }
            let mut base = action_affordances.affordances();
            upsert_affordance(&mut base, affordance);
            match set_merged_action_affordances(base.clone(), boot_config, env).await {
                Ok(snapshot) => {
                    action_affordances.replace(base);
                    if let Err(error) = action_affordances.save() {
                        visualizer.send_event(VisualizerEvent::Log {
                            tab_id: tab_id.clone(),
                            message: format!("failed to save action affordances: {error}"),
                        });
                    }
                    emit_action_affordances(visualizer, tab_id, snapshot.affordances);
                }
                Err(error) => visualizer.send_event(VisualizerEvent::Log {
                    tab_id: tab_id.clone(),
                    message: format!("rejected action affordance update: {error}"),
                }),
            }
            false
        }
        VisualizerCommand::RemoveAgentActionAffordance {
            tab_id: command_tab,
            action_id,
        } if command_tab == *tab_id => {
            let mut base = action_affordances.affordances();
            base.retain(|affordance| affordance.id != action_id);
            match set_merged_action_affordances(base.clone(), boot_config, env).await {
                Ok(snapshot) => {
                    action_affordances.replace(base);
                    if let Err(error) = action_affordances.save() {
                        visualizer.send_event(VisualizerEvent::Log {
                            tab_id: tab_id.clone(),
                            message: format!("failed to save action affordances: {error}"),
                        });
                    }
                    emit_action_affordances(visualizer, tab_id, snapshot.affordances);
                }
                Err(error) => visualizer.send_event(VisualizerEvent::Log {
                    tab_id: tab_id.clone(),
                    message: format!("rejected action affordance removal: {error}"),
                }),
            }
            false
        }
        VisualizerCommand::CompleteAgentActionInvocation {
            tab_id: command_tab,
            completion,
        } if command_tab == *tab_id => {
            if !env.external_actions.complete(completion).await {
                visualizer.send_event(VisualizerEvent::Log {
                    tab_id: tab_id.clone(),
                    message: "external action completion did not match a pending invocation"
                        .to_owned(),
                });
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
        VisualizerCommand::LoadMemoryRecords {
            tab_id: command_tab,
            scope,
            offset,
            limit,
        } if command_tab == *tab_id => {
            let scope_for_event = scope.clone();
            match load_memory_records(env, scope, offset, limit).await {
                Ok((records, has_more)) => {
                    visualizer.send_event(VisualizerEvent::MemoryRecordsLoaded {
                        tab_id: tab_id.clone(),
                        scope: scope_for_event,
                        offset,
                        records,
                        has_more,
                    });
                }
                Err(error) => {
                    visualizer.send_event(VisualizerEvent::Log {
                        tab_id: tab_id.clone(),
                        message: format!("failed to load memory records: {error}"),
                    });
                }
            }
            false
        }
        VisualizerCommand::LoadLinkedMemories {
            tab_id: command_tab,
            memory_index,
            relation_filter,
            offset,
            limit,
        } if command_tab == *tab_id => {
            match load_linked_memory_records(
                env,
                memory_index.clone(),
                relation_filter,
                offset,
                limit,
            )
            .await
            {
                Ok((records, has_more)) => {
                    visualizer.send_event(VisualizerEvent::LinkedMemoryRecordsLoaded {
                        tab_id: tab_id.clone(),
                        memory_index,
                        offset,
                        records,
                        has_more,
                    });
                }
                Err(error) => {
                    visualizer.send_event(VisualizerEvent::Log {
                        tab_id: tab_id.clone(),
                        message: format!("failed to load linked memory records: {error}"),
                    });
                }
            }
            false
        }
        VisualizerCommand::DeleteMemory {
            tab_id: command_tab,
            memory_index,
        } if command_tab == *tab_id => {
            let index = MemoryIndex::new(memory_index.clone());
            if let Err(error) = env.memory_caps.deleter().delete(&index).await {
                visualizer.send_event(VisualizerEvent::Log {
                    tab_id: tab_id.clone(),
                    message: format!("failed to delete memory {memory_index}: {error}"),
                });
            } else {
                visualizer.send_event(VisualizerEvent::MemoryDeleted {
                    tab_id: tab_id.clone(),
                    memory_index,
                });
            }
            false
        }
        _ => false,
    }
}

async fn apply_action_affordance_set(
    action_affordances: &mut ActionAffordanceState,
    boot_config: &ServerBootConfig,
    affordances: Vec<ActionAffordance>,
    visualizer: &VisualizerHook,
    tab_id: &VisualizerTabId,
    env: &ServerEnvironment,
) {
    if let Err(error) = validate_base_action_affordances(&affordances) {
        visualizer.send_event(VisualizerEvent::Log {
            tab_id: tab_id.clone(),
            message: format!("rejected action affordance set: {error}"),
        });
        return;
    }
    match set_merged_action_affordances(affordances.clone(), boot_config, env).await {
        Ok(snapshot) => {
            action_affordances.replace(affordances);
            if let Err(error) = action_affordances.save() {
                visualizer.send_event(VisualizerEvent::Log {
                    tab_id: tab_id.clone(),
                    message: format!("failed to save action affordances: {error}"),
                });
            }
            emit_action_affordances(visualizer, tab_id, snapshot.affordances);
        }
        Err(error) => visualizer.send_event(VisualizerEvent::Log {
            tab_id: tab_id.clone(),
            message: format!("rejected action affordance set: {error}"),
        }),
    }
}

async fn set_merged_action_affordances(
    base: Vec<ActionAffordance>,
    boot_config: &ServerBootConfig,
    env: &ServerEnvironment,
) -> Result<nuillu_module::ActionAffordanceSnapshot, nuillu_module::ActionAffordanceError> {
    env.caps
        .host_io()
        .action_affordance_writer()
        .set_all(boot_config.overlay_action_affordances(base))
        .await
}

fn validate_base_action_affordances(
    affordances: &[ActionAffordance],
) -> Result<(), nuillu_module::ActionAffordanceError> {
    let mut seen = HashSet::new();
    for affordance in affordances {
        affordance.validate()?;
        if !seen.insert(affordance.id.as_str()) {
            return Err(nuillu_module::ActionAffordanceError::DuplicateId(
                affordance.id.clone(),
            ));
        }
    }
    Ok(())
}

fn upsert_affordance(affordances: &mut Vec<ActionAffordance>, affordance: ActionAffordance) {
    match affordances
        .iter_mut()
        .find(|candidate| candidate.id == affordance.id)
    {
        Some(existing) => *existing = affordance,
        None => affordances.push(affordance),
    }
}

pub(super) fn emit_action_affordances(
    visualizer: &VisualizerHook,
    tab_id: &VisualizerTabId,
    affordances: Vec<ActionAffordance>,
) {
    visualizer.send_event(VisualizerEvent::AgentActionAffordances {
        tab_id: tab_id.clone(),
        affordances,
    });
}

async fn load_memory_records(
    env: &ServerEnvironment,
    scope: MemoryRecordScope,
    offset: usize,
    limit: usize,
) -> anyhow::Result<(Vec<nuillu_visualizer_protocol::MemoryRecordView>, bool)> {
    if limit == 0 {
        return Ok((Vec::new(), false));
    }
    let fetch_limit = limit.saturating_add(1);
    let records = match scope {
        MemoryRecordScope::Latest => env.memory.list_recent(offset, fetch_limit).await?,
        MemoryRecordScope::Search { query } => {
            let mut query = MemoryQuery::text(query, fetch_limit);
            query.offset = offset;
            env.memory.search(&query).await?
        }
    };
    let (records, has_more) = trim_chunk(records, limit);
    Ok((
        records.into_iter().map(memory_record_view).collect(),
        has_more,
    ))
}

async fn load_linked_memory_records(
    env: &ServerEnvironment,
    memory_index: String,
    relation_filter: Vec<String>,
    offset: usize,
    limit: usize,
) -> anyhow::Result<(
    Vec<nuillu_visualizer_protocol::LinkedMemoryRecordView>,
    bool,
)> {
    if limit == 0 {
        return Ok((Vec::new(), false));
    }
    let relation_filter = relation_filter
        .into_iter()
        .filter_map(|relation| parse_memory_relation(&relation))
        .collect::<Vec<_>>();
    let records = env
        .memory
        .linked(&LinkedMemoryQuery {
            memory_indexes: vec![MemoryIndex::new(memory_index)],
            relation_filter,
            direction: MemoryLinkDirection::Both,
            offset,
            limit: limit.saturating_add(1),
        })
        .await?;
    let (records, has_more) = trim_chunk(records, limit);
    Ok((
        records.into_iter().map(linked_memory_record_view).collect(),
        has_more,
    ))
}

fn trim_chunk<T>(mut records: Vec<T>, limit: usize) -> (Vec<T>, bool) {
    if records.len() > limit {
        records.truncate(limit);
        (records, true)
    } else {
        (records, false)
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
        publish_scene_snapshot(scene, sensory, visualizer, tab_id, env).await;
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
    publish_scene_snapshot(scene, sensory, visualizer, tab_id, env).await;
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

pub(super) async fn emit_recent_activity_rows(
    env: &ServerEnvironment,
    visualizer: &VisualizerHook,
    tab_id: &VisualizerTabId,
) {
    match env
        .one_shot_sensory_input_store
        .recent(RECENT_ACTIVITY_ROW_LIMIT)
        .await
    {
        Ok(rows) => visualizer.send_event(VisualizerEvent::OneShotSensoryInputRows {
            tab_id: tab_id.clone(),
            rows: rows
                .into_iter()
                .map(one_shot_sensory_input_row_view)
                .collect(),
        }),
        Err(error) => visualizer.send_event(VisualizerEvent::Log {
            tab_id: tab_id.clone(),
            message: format!("failed to load one-shot sensory input rows: {error}"),
        }),
    }
    match env
        .ambient_sensory_snapshot_store
        .recent(RECENT_ACTIVITY_ROW_LIMIT)
        .await
    {
        Ok(rows) => visualizer.send_event(VisualizerEvent::AmbientSensorySnapshotRows {
            tab_id: tab_id.clone(),
            rows: rows
                .into_iter()
                .map(ambient_sensory_snapshot_row_view)
                .collect(),
        }),
        Err(error) => visualizer.send_event(VisualizerEvent::Log {
            tab_id: tab_id.clone(),
            message: format!("failed to load ambient sensory snapshot rows: {error}"),
        }),
    }
    match env
        .utterance_event_store
        .recent(RECENT_ACTIVITY_ROW_LIMIT)
        .await
    {
        Ok(rows) => visualizer.send_event(VisualizerEvent::UtteranceEventRows {
            tab_id: tab_id.clone(),
            rows: rows.into_iter().map(utterance_event_row_view).collect(),
        }),
        Err(error) => visualizer.send_event(VisualizerEvent::Log {
            tab_id: tab_id.clone(),
            message: format!("failed to load utterance event rows: {error}"),
        }),
    }
    match env
        .external_action_event_store
        .recent(RECENT_ACTIVITY_ROW_LIMIT)
        .await
    {
        Ok(rows) => visualizer.send_event(VisualizerEvent::ExternalActionEventRows {
            tab_id: tab_id.clone(),
            rows: rows
                .into_iter()
                .map(external_action_event_row_view)
                .collect(),
        }),
        Err(error) => visualizer.send_event(VisualizerEvent::Log {
            tab_id: tab_id.clone(),
            message: format!("failed to load external action event rows: {error}"),
        }),
    }
}

async fn record_sensory_input(
    visualizer: &VisualizerHook,
    tab_id: &VisualizerTabId,
    env: &ServerEnvironment,
    input: &SensoryInput,
) {
    match input {
        SensoryInput::OneShot {
            modality,
            direction,
            content,
            observed_at,
        } => match env
            .one_shot_sensory_input_store
            .append(NewOneShotSensoryInput {
                server_session_id: env.server_session_id.clone(),
                modality: modality.as_str().to_string(),
                direction: direction.clone(),
                content: content.clone(),
                observed_at_ms: observed_at.timestamp_millis(),
            })
            .await
        {
            Ok(row) => visualizer.send_event(VisualizerEvent::OneShotSensoryInputAppended {
                tab_id: tab_id.clone(),
                row: one_shot_sensory_input_row_view(row),
            }),
            Err(error) => visualizer.send_event(VisualizerEvent::Log {
                tab_id: tab_id.clone(),
                message: format!("failed to persist one-shot sensory input: {error}"),
            }),
        },
        SensoryInput::AmbientSnapshot {
            entries,
            observed_at,
        } => match env
            .ambient_sensory_snapshot_store
            .append(NewAmbientSensorySnapshot {
                server_session_id: env.server_session_id.clone(),
                entries: entries.clone(),
                observed_at_ms: observed_at.timestamp_millis(),
            })
            .await
        {
            Ok(row) => visualizer.send_event(VisualizerEvent::AmbientSensorySnapshotAppended {
                tab_id: tab_id.clone(),
                row: ambient_sensory_snapshot_row_view(row),
            }),
            Err(error) => visualizer.send_event(VisualizerEvent::Log {
                tab_id: tab_id.clone(),
                message: format!("failed to persist ambient sensory snapshot: {error}"),
            }),
        },
    }
}

pub(super) fn one_shot_sensory_input_row_view(
    row: OneShotSensoryInputRecord,
) -> OneShotSensoryInputRowView {
    OneShotSensoryInputRowView {
        id: row.id,
        server_session_id: row.server_session_id,
        modality: row.modality,
        direction: row.direction,
        content: row.content,
        observed_at: timestamp_from_ms(row.observed_at_ms),
        created_at: timestamp_from_ms(row.created_at_ms),
    }
}

pub(super) fn ambient_sensory_snapshot_row_view(
    row: AmbientSensorySnapshotRecord,
) -> AmbientSensorySnapshotRowView {
    AmbientSensorySnapshotRowView {
        id: row.id,
        server_session_id: row.server_session_id,
        entries: row.entries,
        observed_at: timestamp_from_ms(row.observed_at_ms),
        created_at: timestamp_from_ms(row.created_at_ms),
    }
}

pub(super) fn utterance_event_row_view(row: UtteranceEventRecord) -> UtteranceEventRowView {
    UtteranceEventRowView {
        id: row.id,
        server_session_id: row.server_session_id,
        event_kind: match row.event_kind {
            UtteranceEventKind::Delta => UtteranceEventKindView::Delta,
            UtteranceEventKind::Completed => UtteranceEventKindView::Completed,
            UtteranceEventKind::Aborted => UtteranceEventKindView::Aborted,
        },
        sender: row.sender.to_string(),
        target: row.target,
        generation_id: row.generation_id,
        sequence: row.sequence,
        content: row.content,
        reason: row.reason,
        occurred_at: timestamp_from_ms(row.occurred_at_ms),
        created_at: timestamp_from_ms(row.created_at_ms),
    }
}

pub(super) fn external_action_event_row_view(
    row: ExternalActionEventRecord,
) -> ExternalActionEventRowView {
    ExternalActionEventRowView {
        id: row.id,
        server_session_id: row.server_session_id,
        invocation_id: row.invocation_id,
        invoked_by: row.invoked_by.to_string(),
        action_id: row.action_id,
        arguments: row.arguments,
        status: match row.status {
            ExternalActionEventStatus::Pending => ExternalActionEventStatusView::Pending,
            ExternalActionEventStatus::Completed => ExternalActionEventStatusView::Completed,
        },
        accepted: row.accepted,
        message: row.message,
        requested_at: timestamp_from_ms(row.requested_at_ms),
        completed_at: row.completed_at_ms.map(timestamp_from_ms),
        created_at: timestamp_from_ms(row.created_at_ms),
        updated_at: timestamp_from_ms(row.updated_at_ms),
    }
}

fn timestamp_from_ms(value: i64) -> DateTime<Utc> {
    DateTime::<Utc>::from_timestamp_millis(value)
        .unwrap_or_else(|| DateTime::<Utc>::from_timestamp(0, 0).expect("epoch timestamp is valid"))
}

async fn publish_scene_snapshot(
    scene: &SceneState,
    sensory: &SensoryInputMailbox,
    visualizer: &VisualizerHook,
    tab_id: &VisualizerTabId,
    env: &ServerEnvironment,
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
        observed_at: env.clock.now(),
    };
    let _ = sensory.publish(body.clone()).await;
    record_sensory_input(visualizer, tab_id, env, &body).await;
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
    record_sensory_input(visualizer, tab_id, env, &body).await;
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
    use std::path::Path;

    use crate::config::parse_server_boot_config_content;

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

    #[test]
    fn visualizer_base_updates_keep_config_action_precedence() {
        let boot_config = parse_server_boot_config_content(
            r#"
@ actions[] {
  name = "poet"
  description = "Config poet."

  json_schema {
    type = "object"
  }
}
"#,
            Path::new(".tmp/server/config.eure"),
        )
        .unwrap();
        let mut base = vec![action_affordance("poet", "Persisted Poet")];

        upsert_affordance(&mut base, action_affordance("poet", "Visualizer Poet"));
        let merged = boot_config.overlay_action_affordances(base.clone());
        assert_eq!(merged, vec![action_affordance("poet", "Config poet.")]);

        base.retain(|affordance| affordance.id != "poet");
        let merged = boot_config.overlay_action_affordances(base);
        assert_eq!(merged, vec![action_affordance("poet", "Config poet.")]);
    }

    fn action_affordance(id: &str, description: &str) -> ActionAffordance {
        ActionAffordance {
            id: id.to_owned(),
            label: id.to_owned(),
            description: description.to_owned(),
            use_when: String::new(),
            effect: String::new(),
            input_schema: serde_json::json!({"type": "object"}),
        }
    }
}
