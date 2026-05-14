use std::time::Duration;

use nuillu_blackboard::{
    Blackboard, BlackboardCommand, Bpm, ModulePolicy, ZeroReplicaWindowPolicy,
};
use nuillu_memory::{LinkedMemoryQuery, MemoryLinkDirection, MemoryLinkRelation};
use nuillu_module::{
    AllocationUpdated, AllocationUpdatedMailbox, AmbientSensoryEntry, SensoryInput,
    SensoryInputMailbox, SensoryModality,
};
use nuillu_types::{MemoryIndex, ModuleId, ReplicaCapRange};
use nuillu_visualizer_protocol::{
    ModuleSettingsView, VisualizerClientMessage, VisualizerCommand, VisualizerEvent,
    VisualizerTabId, ZeroReplicaWindowView, memory_page_from_records,
};

use crate::SERVER_TAB_ID;
use crate::environment::ServerEnvironment;
use crate::gui::VisualizerHook;
use crate::snapshot::{
    emit_visualizer_blackboard_snapshot, linked_memory_record_view, list_all_memories,
    memory_record_view,
};
use crate::state::{AmbientRows, ModuleSettingsState};

const SNAPSHOT_INTERVAL: Duration = Duration::from_millis(100);

pub(super) async fn drive_server_until_shutdown(
    visualizer: &mut VisualizerHook,
    tab_id: &VisualizerTabId,
    ambient: &mut AmbientRows,
    module_settings: &mut ModuleSettingsState,
    sensory: &SensoryInputMailbox,
    allocation_updates: &AllocationUpdatedMailbox,
    env: &ServerEnvironment,
) {
    publish_ambient_snapshot(ambient, sensory, visualizer, tab_id, env.clock.as_ref()).await;
    loop {
        if visualizer.shutdown_requested() {
            break;
        }
        while let Some(message) = visualizer.try_recv_command() {
            if handle_server_visualizer_message(
                message,
                visualizer,
                tab_id,
                ambient,
                module_settings,
                sensory,
                allocation_updates,
                env,
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

async fn handle_server_visualizer_message(
    message: VisualizerClientMessage,
    visualizer: &mut VisualizerHook,
    tab_id: &VisualizerTabId,
    ambient: &mut AmbientRows,
    module_settings: &mut ModuleSettingsState,
    sensory: &SensoryInputMailbox,
    allocation_updates: &AllocationUpdatedMailbox,
    env: &ServerEnvironment,
) -> bool {
    let command = match message {
        VisualizerClientMessage::Hello { .. } => return false,
        VisualizerClientMessage::InvokeAction { .. } => return false,
        VisualizerClientMessage::Command { command } => command,
    };
    match command {
        VisualizerCommand::Shutdown => {
            visualizer.request_shutdown();
            true
        }
        VisualizerCommand::SendSensoryInput {
            tab_id: command_tab,
            input,
        } if command_tab == *tab_id => {
            let body = SensoryInput::Observed {
                modality: SensoryModality::parse(input.modality),
                direction: None,
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
            ambient.create(modality, content, disabled);
            persist_and_emit_ambient(ambient, visualizer, tab_id, sensory, env.clock.as_ref())
                .await;
            false
        }
        VisualizerCommand::UpdateAmbientSensoryRow {
            tab_id: command_tab,
            row,
        } if command_tab == *tab_id => {
            ambient.update(row);
            persist_and_emit_ambient(ambient, visualizer, tab_id, sensory, env.clock.as_ref())
                .await;
            false
        }
        VisualizerCommand::RemoveAmbientSensoryRow {
            tab_id: command_tab,
            row_id,
        } if command_tab == *tab_id => {
            ambient.remove(&row_id);
            persist_and_emit_ambient(ambient, visualizer, tab_id, sensory, env.clock.as_ref())
                .await;
            false
        }
        VisualizerCommand::SetModuleDisabled {
            tab_id: command_tab,
            module,
            disabled,
        } if command_tab == *tab_id => {
            set_module_disabled(
                &module,
                disabled,
                visualizer,
                tab_id,
                allocation_updates,
                env,
            )
            .await;
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
                allocation_updates,
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

async fn set_module_disabled(
    module: &str,
    disabled: bool,
    visualizer: &VisualizerHook,
    tab_id: &VisualizerTabId,
    allocation_updates: &AllocationUpdatedMailbox,
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
    let before = env.blackboard.read(|bb| bb.allocation().clone()).await;
    env.blackboard
        .apply(BlackboardCommand::SetModuleForcedDisabled {
            module: module_id,
            disabled,
        })
        .await;
    let after = env.blackboard.read(|bb| bb.allocation().clone()).await;
    if before != after && allocation_updates.publish(AllocationUpdated).await.is_err() {
        tracing::trace!("visualizer forced-disable allocation update had no active subscribers");
    }
}

async fn persist_and_emit_ambient(
    ambient: &AmbientRows,
    visualizer: &mut VisualizerHook,
    tab_id: &VisualizerTabId,
    sensory: &SensoryInputMailbox,
    clock: &dyn nuillu_module::ports::Clock,
) {
    if let Err(error) = ambient.save() {
        visualizer.send_event(VisualizerEvent::Log {
            tab_id: tab_id.clone(),
            message: format!("failed to save ambient sensory rows: {error}"),
        });
    }
    visualizer.send_event(VisualizerEvent::AmbientSensoryRows {
        tab_id: tab_id.clone(),
        rows: ambient.rows.clone(),
    });
    publish_ambient_snapshot(ambient, sensory, visualizer, tab_id, clock).await;
}

async fn publish_ambient_snapshot(
    ambient: &AmbientRows,
    sensory: &SensoryInputMailbox,
    visualizer: &VisualizerHook,
    tab_id: &VisualizerTabId,
    clock: &dyn nuillu_module::ports::Clock,
) {
    let entries = ambient
        .rows
        .iter()
        .filter(|row| !row.disabled && !row.content.trim().is_empty())
        .map(|row| AmbientSensoryEntry {
            id: row.id.clone(),
            modality: SensoryModality::parse(&row.modality),
            content: row.content.clone(),
        })
        .collect::<Vec<_>>();
    if entries.is_empty() {
        return;
    }
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

pub(super) async fn apply_persisted_module_settings(
    settings: &ModuleSettingsState,
    visualizer: &VisualizerHook,
    tab_id: &VisualizerTabId,
    blackboard: &Blackboard,
    allocation_updates: &AllocationUpdatedMailbox,
) {
    for setting in settings.iter() {
        apply_visualizer_module_settings(
            tab_id,
            visualizer,
            blackboard,
            allocation_updates,
            setting.clone(),
        )
        .await;
    }
}

async fn apply_visualizer_module_settings(
    tab_id: &VisualizerTabId,
    visualizer: &VisualizerHook,
    blackboard: &Blackboard,
    allocation_updates: &AllocationUpdatedMailbox,
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

    let before = blackboard.read(|bb| bb.allocation().clone()).await;
    blackboard
        .apply(BlackboardCommand::SetModulePolicies {
            policies: vec![update],
        })
        .await;
    let after = blackboard.read(|bb| bb.allocation().clone()).await;
    if before != after && allocation_updates.publish(AllocationUpdated).await.is_err() {
        tracing::trace!("visualizer module settings allocation update had no active subscribers");
    }
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
