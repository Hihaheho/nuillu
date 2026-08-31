use std::time::Duration;

use nuillu_blackboard::{
    Blackboard, BlackboardInner, InteroceptiveMode, InteroceptiveState, MemoryMetadata,
    ResourceAllocation, ZeroReplicaWindowPolicy,
};
use nuillu_memory::{LinkedMemoryRecord, MemoryRecord};
use nuillu_types::{MemoryRank, ScopeId};
use nuillu_visualizer_protocol::{
    AllocationView, BlackboardSnapshot, CognitionEntryView, CognitionLogView, InteroceptionView,
    LinkedMemoryRecordView, MemoView, MemoryConceptView, MemoryLinkView, MemoryMetadataView,
    MemoryRecordView, MemoryTagView, ModulePolicyView, ModuleStatusView, ScopeView,
    UtteranceProgressView, VisualizerEvent, VisualizerTabId, ZeroReplicaWindowView,
};

use super::config::{ServerBootConfig, ServerMemoryScope};
use super::gui::VisualizerHook;

const VISUALIZER_MEMORY_METADATA_LIMIT: usize = 512;

pub(crate) async fn emit_visualizer_blackboard_snapshot(
    tab_id: &str,
    blackboard: &Blackboard,
    boot_config: &ServerBootConfig,
    visualizer: &VisualizerHook,
) {
    let mut snapshot = BlackboardSnapshot {
        scopes: scope_views(boot_config, blackboard).await,
        ..BlackboardSnapshot::default()
    };
    for scoped_blackboard in blackboard.all_scopes() {
        let scope = scoped_blackboard.scope().clone();
        let scope_activation = blackboard.scope_activation_state(&scope).await;
        let effective_allocation = blackboard.effective_module_allocation(&scope).await;
        let scoped_snapshot = scoped_blackboard
            .read(|bb| {
                visualizer_scoped_blackboard_snapshot(
                    &scope,
                    bb,
                    &effective_allocation,
                    scope_activation.effective_activation,
                )
            })
            .await;
        merge_blackboard_snapshot(&mut snapshot, scoped_snapshot, scope.is_root());
    }
    visualizer.send_event(VisualizerEvent::BlackboardSnapshot {
        tab_id: VisualizerTabId::new(tab_id.to_string()),
        snapshot,
    });
}

async fn scope_views(boot_config: &ServerBootConfig, blackboard: &Blackboard) -> Vec<ScopeView> {
    let expanded = boot_config.expanded_subsystems();
    if expanded.is_empty() {
        return Vec::new();
    }
    let mut scopes = vec![ScopeView {
        id: ScopeId::root().to_string(),
        parent: None,
        memory_scope: "global".to_owned(),
        subsystem: None,
        replica: None,
        local_activation: 1.0,
        effective_activation: 1.0,
        active_replicas: 1,
        active: true,
    }];
    for expanded in expanded {
        let state = blackboard.scope_activation_state(&expanded.scope).await;
        let instance = expanded
            .scope
            .path()
            .last()
            .expect("expanded scope is non-root");
        scopes.push(ScopeView {
            id: expanded.scope.to_string(),
            parent: expanded.scope.parent().map(|scope| scope.to_string()),
            memory_scope: match expanded.definition.memory_scope {
                ServerMemoryScope::Global => "global",
                ServerMemoryScope::Local => "local",
            }
            .to_owned(),
            subsystem: Some(instance.subsystem.to_string()),
            replica: Some(instance.replica.get()),
            local_activation: state.local_activation.as_f64(),
            effective_activation: state.effective_activation.as_f64(),
            active_replicas: state.active_replicas,
            active: state.active,
        });
    }
    scopes.sort_by(|left, right| left.id.cmp(&right.id));
    scopes
}

fn merge_blackboard_snapshot(
    target: &mut BlackboardSnapshot,
    mut source: BlackboardSnapshot,
    root: bool,
) {
    target.module_statuses.append(&mut source.module_statuses);
    target.allocation.append(&mut source.allocation);
    target.module_policies.append(&mut source.module_policies);
    if root {
        target.interoception = source.interoception;
        target.forced_disabled_modules = source.forced_disabled_modules;
        target.memos = source.memos;
        target.cognition_logs = source.cognition_logs;
        target.utterance_progresses = source.utterance_progresses;
        target.memory_metadata = source.memory_metadata;
    }
}

pub fn memory_record_view(record: MemoryRecord) -> MemoryRecordView {
    MemoryRecordView {
        index: record.index.as_str().to_string(),
        kind: format!("{:?}", record.kind),
        rank: format!("{:?}", record.rank),
        occurred_at: record.occurred_at,
        stored_at: record.stored_at,
        concepts: record
            .concepts
            .into_iter()
            .map(|concept| MemoryConceptView {
                label: concept.label,
                mention_text: concept.mention_text,
                loose_type: concept.loose_type,
                confidence: concept.confidence,
            })
            .collect(),
        tags: record
            .tags
            .into_iter()
            .map(|tag| MemoryTagView {
                label: tag.label,
                namespace: tag.namespace,
                confidence: tag.confidence,
            })
            .collect(),
        affect_arousal: record.affect_arousal,
        valence: record.valence,
        emotion: record.emotion,
        content: record.content.as_str().to_string(),
    }
}

pub fn linked_memory_record_view(record: LinkedMemoryRecord) -> LinkedMemoryRecordView {
    LinkedMemoryRecordView {
        record: memory_record_view(record.record),
        link: MemoryLinkView {
            from_memory: record.link.from_memory.to_string(),
            to_memory: record.link.to_memory.to_string(),
            relation: format!("{:?}", record.link.relation),
            freeform_relation: record.link.freeform_relation,
            strength: record.link.strength,
            confidence: record.link.confidence,
            updated_at: record.link.updated_at,
        },
    }
}

#[cfg(test)]
fn visualizer_blackboard_snapshot(bb: &BlackboardInner) -> BlackboardSnapshot {
    visualizer_scoped_blackboard_snapshot(
        &ScopeId::root(),
        bb,
        bb.allocation(),
        nuillu_blackboard::ActivationRatio::ONE,
    )
}

fn visualizer_scoped_blackboard_snapshot(
    scope: &ScopeId,
    bb: &BlackboardInner,
    effective_allocation: &ResourceAllocation,
    scope_activation: nuillu_blackboard::ActivationRatio,
) -> BlackboardSnapshot {
    let include_root_content = scope.is_root();
    BlackboardSnapshot {
        scopes: Vec::new(),
        module_statuses: bb
            .module_status_records()
            .into_iter()
            .map(|record| ModuleStatusView {
                owner: record.owner.to_string(),
                module: record.owner.module.as_str().to_owned(),
                replica: record.owner.replica.get(),
                status: format!("{:?}", record.status),
            })
            .collect(),
        allocation: allocation_views(
            scope,
            bb.allocation(),
            effective_allocation,
            scope_activation,
        ),
        interoception: if include_root_content {
            interoception_view(bb.interoception())
        } else {
            Default::default()
        },
        module_policies: scoped_module_policy_views(scope, bb),
        forced_disabled_modules: if include_root_content {
            let mut modules = bb
                .forced_disabled_modules()
                .iter()
                .map(|module| module.as_str().to_owned())
                .collect::<Vec<_>>();
            modules.sort();
            modules
        } else {
            Vec::new()
        },
        memos: if include_root_content {
            bb.recent_memo_logs()
                .into_iter()
                .map(|record| MemoView {
                    owner: record.owner.to_string(),
                    module: record.owner.module.as_str().to_owned(),
                    replica: record.owner.replica.get(),
                    index: record.index,
                    written_at: record.written_at,
                    cognitive: record.cognitive,
                    content: record.content,
                })
                .collect()
        } else {
            Vec::new()
        },
        cognition_logs: if include_root_content {
            bb.cognition_log_set()
                .logs()
                .iter()
                .map(|record| CognitionLogView {
                    source: record.source.to_string(),
                    entries: record
                        .entries
                        .iter()
                        .map(|entry| CognitionEntryView {
                            at: entry.at,
                            origin: entry.origin.owner.to_string(),
                            text: entry.text.clone(),
                        })
                        .collect(),
                })
                .collect()
        } else {
            Vec::new()
        },
        utterance_progresses: if include_root_content {
            bb.utterance_progress_records()
                .into_iter()
                .map(|record| UtteranceProgressView {
                    owner: record.owner.to_string(),
                    target: record.progress.target,
                    generation_id: record.progress.generation_id,
                    sequence: record.progress.sequence,
                    state: format!("{:?}", record.progress.state),
                    partial_utterance: record.progress.partial_utterance,
                })
                .collect()
        } else {
            Vec::new()
        },
        memory_metadata: if include_root_content {
            memory_metadata_views(bb)
        } else {
            Vec::new()
        },
    }
}

fn interoception_view(state: &InteroceptiveState) -> InteroceptionView {
    InteroceptionView {
        mode: interoceptive_mode_name(state.mode).to_owned(),
        wake_arousal: state.wake_arousal,
        nrem_pressure: state.nrem_pressure,
        rem_pressure: state.rem_pressure,
        affect_arousal: state.affect_arousal,
        valence: state.valence,
        emotion: state.emotion.clone(),
        last_updated: state.last_updated,
    }
}

fn interoceptive_mode_name(mode: InteroceptiveMode) -> &'static str {
    match mode {
        InteroceptiveMode::Wake => "wake",
        InteroceptiveMode::NremPressure => "nrem_pressure",
        InteroceptiveMode::RemPressure => "rem_pressure",
    }
}

fn allocation_views(
    scope: &ScopeId,
    allocation: &ResourceAllocation,
    effective: &ResourceAllocation,
    scope_activation: nuillu_blackboard::ActivationRatio,
) -> Vec<AllocationView> {
    let mut modules = allocation
        .module_ids()
        .into_iter()
        .map(|module| {
            let bpm = effective.bpm_for(&module);
            AllocationView {
                scope: scope.to_string(),
                bpm: bpm.map(|bpm| bpm.as_f64()),
                period_ms: bpm.map(|bpm| duration_millis_u64(bpm.period())),
                module: module.as_str().to_owned(),
                activation_ratio: allocation.activation_for(&module).as_f64(),
                scope_activation_ratio: scope_activation.as_f64(),
                effective_activation_ratio: effective.effective_activation_for(&module).as_f64(),
                active_replicas: effective.active_replicas(&module),
            }
        })
        .collect::<Vec<_>>();
    modules.sort_by(|left, right| left.module.cmp(&right.module));
    modules
}

pub fn module_policy_views(bb: &BlackboardInner) -> Vec<ModulePolicyView> {
    scoped_module_policy_views(&ScopeId::root(), bb)
}

fn scoped_module_policy_views(scope: &ScopeId, bb: &BlackboardInner) -> Vec<ModulePolicyView> {
    let mut policies = bb
        .module_policies()
        .iter()
        .map(|(module, policy)| ModulePolicyView {
            scope: scope.to_string(),
            module: module.as_str().to_owned(),
            replica_min: policy.replicas_range.min,
            replica_max: policy.replicas_range.max,
            replica_capacity: bb
                .module_replica_capacity(module)
                .unwrap_or_else(|| policy.max_active_replicas()),
            bpm_min: policy.rate_limit_range.start().as_f64(),
            bpm_max: policy.rate_limit_range.end().as_f64(),
            zero_replica_window: zero_replica_window_view(policy.zero_replica_window),
        })
        .collect::<Vec<_>>();
    policies.sort_by(|left, right| left.module.cmp(&right.module));
    policies
}

pub fn zero_replica_window_view(policy: ZeroReplicaWindowPolicy) -> ZeroReplicaWindowView {
    match policy {
        ZeroReplicaWindowPolicy::Disabled => ZeroReplicaWindowView::Disabled,
        ZeroReplicaWindowPolicy::EveryControllerActivations(period) => {
            ZeroReplicaWindowView::EveryControllerActivations { period }
        }
    }
}

pub fn memory_metadata_views(bb: &BlackboardInner) -> Vec<MemoryMetadataView> {
    let mut memory_metadata = bb.memory_metadata().iter().collect::<Vec<_>>();
    memory_metadata.sort_by(|(left_index, left), (right_index, right)| {
        memory_metadata_activity_order(left_index.as_str(), left, right_index.as_str(), right)
    });
    memory_metadata.truncate(VISUALIZER_MEMORY_METADATA_LIMIT);

    let mut memory_metadata = memory_metadata
        .into_iter()
        .map(|(index, metadata)| MemoryMetadataView {
            index: index.as_str().to_owned(),
            rank: memory_rank_name(metadata.rank).to_owned(),
            occurred_at: metadata.occurred_at,
            last_accessed: metadata.last_accessed,
            access_count: metadata.access_count,
            use_count: metadata.use_count,
            reinforcement_count: metadata.reinforcement_count,
        })
        .collect::<Vec<_>>();
    memory_metadata.sort_by(|left, right| left.index.cmp(&right.index));
    memory_metadata
}

fn memory_metadata_activity_order(
    left_index: &str,
    left: &MemoryMetadata,
    right_index: &str,
    right: &MemoryMetadata,
) -> std::cmp::Ordering {
    right
        .last_reinforced_at
        .cmp(&left.last_reinforced_at)
        .then_with(|| right.last_used.cmp(&left.last_used))
        .then_with(|| right.last_accessed.cmp(&left.last_accessed))
        .then_with(|| right.occurred_at.cmp(&left.occurred_at))
        .then_with(|| right.reinforcement_count.cmp(&left.reinforcement_count))
        .then_with(|| right.use_count.cmp(&left.use_count))
        .then_with(|| right.access_count.cmp(&left.access_count))
        .then_with(|| left_index.cmp(right_index))
}

pub fn memory_rank_name(rank: MemoryRank) -> &'static str {
    match rank {
        MemoryRank::Identity => "identity",
        MemoryRank::Permanent => "permanent",
        MemoryRank::LongTerm => "long-term",
        MemoryRank::MidTerm => "mid-term",
        MemoryRank::ShortTerm => "short-term",
    }
}

pub fn duration_millis_u64(duration: Duration) -> u64 {
    duration.as_millis().min(u128::from(u64::MAX)) as u64
}

#[cfg(test)]
mod tests {
    use chrono::{DateTime, Utc};
    use nuillu_blackboard::{
        ActivationRatio, BlackboardCommand, InteroceptivePatch, MemoryMetaPatch,
        RegisteredSubsystemPolicy, ReplicaProjection, SubsystemPolicy, SubsystemReplicaRange,
    };
    use nuillu_types::{MemoryIndex, SubsystemId};

    use super::*;

    #[tokio::test]
    async fn scope_views_include_parent_activation_and_memory_mode() {
        let config = crate::config::parse_server_boot_config_content(
            r#"
@ subsystem-definitions[] {
  id: arm
  allocation-description = "Test arm subsystem."
  memory-scope: local

  @ modules[] {
    id: predict
    replica-min = 1
    replica-max = 1
    bpm-min = 1.0
    bpm-max = 2.0
    initial-activation = 1.0
  }
}

@ subsystems[] {
  subsystem: arm
  replicas = 2
}
"#,
            std::path::Path::new(".tmp/server/scope-view-test.eure"),
        )
        .unwrap();

        let blackboard = Blackboard::new();
        blackboard
            .apply(BlackboardCommand::SetRegisteredSubsystems {
                registrations: vec![RegisteredSubsystemPolicy {
                    subsystem: SubsystemId::new("arm").unwrap(),
                    policy: SubsystemPolicy::new(
                        SubsystemReplicaRange::new(2, 2).unwrap(),
                        2,
                        ReplicaProjection::Linear,
                    ),
                    initial_activation: ActivationRatio::ONE,
                }],
            })
            .await;
        let scopes = scope_views(&config, &blackboard).await;
        assert_eq!(
            scopes,
            vec![
                ScopeView {
                    id: "/".to_string(),
                    parent: None,
                    memory_scope: "global".to_string(),
                    subsystem: None,
                    replica: None,
                    local_activation: 1.0,
                    effective_activation: 1.0,
                    active_replicas: 1,
                    active: true,
                },
                ScopeView {
                    id: "/arm[0]".to_string(),
                    parent: Some("/".to_string()),
                    memory_scope: "local".to_string(),
                    subsystem: Some("arm".to_string()),
                    replica: Some(0),
                    local_activation: 1.0,
                    effective_activation: 1.0,
                    active_replicas: 2,
                    active: true,
                },
                ScopeView {
                    id: "/arm[1]".to_string(),
                    parent: Some("/".to_string()),
                    memory_scope: "local".to_string(),
                    subsystem: Some("arm".to_string()),
                    replica: Some(1),
                    local_activation: 1.0,
                    effective_activation: 1.0,
                    active_replicas: 2,
                    active: true,
                },
            ]
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn visualizer_snapshot_includes_interoception() {
        let blackboard = Blackboard::new();
        let now = DateTime::<Utc>::from_timestamp(42, 0).unwrap();
        blackboard
            .apply(BlackboardCommand::UpdateInteroceptive {
                patch: InteroceptivePatch {
                    mode: Some(InteroceptiveMode::NremPressure),
                    wake_arousal: Some(0.25),
                    nrem_pressure: Some(0.75),
                    rem_pressure: Some(0.15),
                    affect_arousal: Some(0.4),
                    valence: Some(-0.5),
                    emotion: Some("drowsy".to_string()),
                },
                now,
            })
            .await;

        let snapshot = blackboard.read(visualizer_blackboard_snapshot).await;

        assert_eq!(
            snapshot.interoception,
            InteroceptionView {
                mode: "nrem_pressure".to_string(),
                wake_arousal: 0.25,
                nrem_pressure: 0.75,
                rem_pressure: 0.15,
                affect_arousal: 0.4,
                valence: -0.5,
                emotion: "drowsy".to_string(),
                last_updated: now,
            }
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn visualizer_snapshot_caps_memory_metadata_to_recent_activity() {
        let blackboard = Blackboard::new();
        let base = DateTime::<Utc>::from_timestamp(100, 0).unwrap();
        let total = VISUALIZER_MEMORY_METADATA_LIMIT + 2;
        for index in 0..total {
            blackboard
                .apply(BlackboardCommand::UpsertMemoryMetadata {
                    index: MemoryIndex::new(format!("memory-{index:04}")),
                    rank_if_new: MemoryRank::ShortTerm,
                    occurred_at_if_new: Some(base + chrono::Duration::seconds(index as i64)),
                    decay_if_new_secs: 0,
                    now: base + chrono::Duration::seconds(index as i64),
                    patch: MemoryMetaPatch::default(),
                })
                .await;
        }

        let snapshot = blackboard.read(visualizer_blackboard_snapshot).await;

        assert_eq!(
            snapshot.memory_metadata.len(),
            VISUALIZER_MEMORY_METADATA_LIMIT
        );
        assert!(
            snapshot
                .memory_metadata
                .iter()
                .all(|memory| memory.index != "memory-0000" && memory.index != "memory-0001")
        );
        assert!(
            snapshot
                .memory_metadata
                .iter()
                .any(|memory| memory.index == format!("memory-{:04}", total - 1))
        );
    }
}
