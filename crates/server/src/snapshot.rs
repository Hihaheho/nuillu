use std::time::Duration;

use nuillu_blackboard::{Blackboard, BlackboardInner, ResourceAllocation, ZeroReplicaWindowPolicy};
use nuillu_memory::{LinkedMemoryRecord, MemoryRecord, MemoryStore};
use nuillu_types::{MemoryRank, ModelTier};
use nuillu_visualizer_protocol::{
    AllocationView, BlackboardSnapshot, CognitionEntryView, CognitionLogView,
    LinkedMemoryRecordView, MemoView, MemoryConceptView, MemoryLinkView, MemoryMetadataView,
    MemoryRecordView, MemoryTagView, ModulePolicyView, ModuleStatusView, UtteranceProgressView,
    VisualizerEvent, VisualizerTabId, ZeroReplicaWindowView, memory_page_from_records,
};

use super::gui::VisualizerHook;

pub(crate) async fn emit_visualizer_blackboard_snapshot(
    tab_id: &str,
    blackboard: &Blackboard,
    visualizer: &VisualizerHook,
) {
    let snapshot = blackboard.read(visualizer_blackboard_snapshot).await;
    visualizer.send_event(VisualizerEvent::BlackboardSnapshot {
        tab_id: VisualizerTabId::new(tab_id.to_string()),
        snapshot,
    });
}

pub(crate) async fn emit_visualizer_memory_page(
    tab_id: &str,
    visualizer: &mut VisualizerHook,
    blackboard: &Blackboard,
    memory: &dyn MemoryStore,
    page: usize,
    per_page: usize,
) {
    let records = list_all_visualizer_memories(blackboard, memory).await;
    let page = memory_page_from_records(&records, page, per_page);
    visualizer.send_event(VisualizerEvent::MemoryPage {
        tab_id: VisualizerTabId::new(tab_id.to_string()),
        page,
    });
}

pub async fn list_all_memories(memory: &dyn MemoryStore) -> Vec<MemoryRecordView> {
    let mut out = Vec::new();
    for rank in [
        MemoryRank::Identity,
        MemoryRank::Permanent,
        MemoryRank::LongTerm,
        MemoryRank::MidTerm,
        MemoryRank::ShortTerm,
    ] {
        if let Ok(records) = memory.list_by_rank(rank).await {
            out.extend(records.into_iter().map(memory_record_view));
        }
    }
    out.sort_by(|left, right| {
        right
            .occurred_at
            .cmp(&left.occurred_at)
            .then_with(|| left.index.cmp(&right.index))
    });
    out
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

async fn list_all_visualizer_memories(
    blackboard: &Blackboard,
    memory: &dyn MemoryStore,
) -> Vec<MemoryRecordView> {
    let indexes = blackboard
        .read(|bb| {
            let mut records = bb
                .memory_metadata()
                .iter()
                .map(|(index, metadata)| {
                    (index.clone(), metadata.occurred_at, metadata.last_accessed)
                })
                .collect::<Vec<_>>();
            records.sort_by(|left, right| {
                right
                    .1
                    .cmp(&left.1)
                    .then_with(|| right.2.cmp(&left.2))
                    .then_with(|| left.0.as_str().cmp(right.0.as_str()))
            });
            records
                .into_iter()
                .map(|(index, _, _)| index)
                .collect::<Vec<_>>()
        })
        .await;

    let mut records = Vec::new();
    for index in indexes {
        if let Ok(Some(record)) = memory.get(&index).await {
            records.push(memory_record_view(record));
        }
    }
    records
}

fn visualizer_blackboard_snapshot(bb: &BlackboardInner) -> BlackboardSnapshot {
    let cognition_log_set = bb.cognition_log_set();
    BlackboardSnapshot {
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
        allocation: allocation_views(bb.allocation()),
        module_policies: module_policy_views(bb),
        forced_disabled_modules: {
            let mut modules = bb
                .forced_disabled_modules()
                .iter()
                .map(|module| module.as_str().to_owned())
                .collect::<Vec<_>>();
            modules.sort();
            modules
        },
        memos: bb
            .recent_memo_logs()
            .into_iter()
            .map(|record| MemoView {
                owner: record.owner.to_string(),
                module: record.owner.module.as_str().to_owned(),
                replica: record.owner.replica.get(),
                index: record.index,
                written_at: record.written_at,
                content: record.content,
            })
            .collect(),
        cognition_logs: cognition_log_set
            .logs()
            .iter()
            .map(|record| CognitionLogView {
                source: record.source.to_string(),
                entries: record
                    .entries
                    .iter()
                    .map(|entry| CognitionEntryView {
                        at: entry.at,
                        text: entry.text.clone(),
                    })
                    .collect(),
            })
            .collect(),
        utterance_progresses: bb
            .utterance_progress_records()
            .into_iter()
            .map(|record| UtteranceProgressView {
                owner: record.owner.to_string(),
                target: record.progress.target,
                generation_id: record.progress.generation_id,
                sequence: record.progress.sequence,
                state: format!("{:?}", record.progress.state),
                partial_utterance: record.progress.partial_utterance,
            })
            .collect(),
        memory_metadata: memory_metadata_views(bb),
    }
}

fn allocation_views(allocation: &ResourceAllocation) -> Vec<AllocationView> {
    let mut modules = allocation
        .iter()
        .map(|(module, config)| AllocationView {
            bpm: allocation.cooldown_for(module).and_then(bpm_from_cooldown),
            cooldown_ms: allocation.cooldown_for(module).map(duration_millis_u64),
            module: module.as_str().to_owned(),
            activation_ratio: allocation.activation_for(module).as_f64(),
            active_replicas: allocation.active_replicas(module),
            tier: model_tier_name(allocation.tier_for(module)).to_owned(),
            guidance: config.guidance.clone(),
        })
        .collect::<Vec<_>>();
    modules.sort_by(|left, right| left.module.cmp(&right.module));
    modules
}

pub fn module_policy_views(bb: &BlackboardInner) -> Vec<ModulePolicyView> {
    let mut policies = bb
        .module_policies()
        .iter()
        .map(|(module, policy)| ModulePolicyView {
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
    let mut memory_metadata = bb
        .memory_metadata()
        .iter()
        .map(|(index, metadata)| MemoryMetadataView {
            index: index.as_str().to_owned(),
            rank: memory_rank_name(metadata.rank).to_owned(),
            occurred_at: metadata.occurred_at,
            last_accessed: metadata.last_accessed,
            access_count: metadata.access_count,
        })
        .collect::<Vec<_>>();
    memory_metadata.sort_by(|left, right| left.index.cmp(&right.index));
    memory_metadata
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

pub fn model_tier_name(tier: ModelTier) -> &'static str {
    match tier {
        ModelTier::Cheap => "cheap",
        ModelTier::Default => "default",
        ModelTier::Premium => "premium",
    }
}

pub fn duration_millis_u64(duration: Duration) -> u64 {
    duration.as_millis().min(u128::from(u64::MAX)) as u64
}

pub fn bpm_from_cooldown(cooldown: Duration) -> Option<f64> {
    let seconds = cooldown.as_secs_f64();
    (seconds.is_finite() && seconds > 0.0).then_some(60.0 / seconds)
}
