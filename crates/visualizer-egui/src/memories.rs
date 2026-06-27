use std::sync::mpsc::Sender;

use egui::scroll_area::ScrollAreaOutput;

use crate::{
    LinkedMemoryRecordView, MemoryRecordScope, MemoryRecordView, VisualizerClientMessage,
    VisualizerCommand, VisualizerTabId,
    i18n::{EguiI18nExt as _, I18nArg},
    text::wrapped_label,
    time::format_jst_datetime,
};

const MAIN_CHUNK_SIZE: usize = 50;
const LINKED_CHUNK_SIZE: usize = 32;
const LOAD_MORE_THRESHOLD_PX: f32 = 300.0;
const LATEST_REFRESH_INTERVAL_SECS: f64 = 5.0;

#[derive(Debug)]
pub struct MemoriesState {
    pub scope: MemoryRecordScope,
    pub records: Vec<MemoryRecordView>,
    pub linked_memory_index: String,
    pub linked_results: Vec<LinkedMemoryRecordView>,
    draft_query: String,
    main_has_more: bool,
    main_loading: bool,
    linked_has_more: bool,
    linked_loading: bool,
    requested_initial_load: bool,
    list_generation: u64,
    linked_generation: u64,
    last_latest_refresh_at: f64,
    main_scroll_offset_y: f32,
    main_content_height: f32,
    pending_scroll_restore: Option<ScrollRestore>,
}

impl Default for MemoriesState {
    fn default() -> Self {
        Self {
            scope: MemoryRecordScope::Latest,
            records: Vec::new(),
            linked_memory_index: String::new(),
            linked_results: Vec::new(),
            draft_query: String::new(),
            main_has_more: true,
            main_loading: false,
            linked_has_more: false,
            linked_loading: false,
            requested_initial_load: false,
            list_generation: 0,
            linked_generation: 0,
            last_latest_refresh_at: 0.0,
            main_scroll_offset_y: 0.0,
            main_content_height: 0.0,
            pending_scroll_restore: None,
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct ScrollRestore {
    offset_y: f32,
    content_height: f32,
}

impl MemoriesState {
    pub fn apply_records_loaded(
        &mut self,
        scope: MemoryRecordScope,
        offset: usize,
        mut records: Vec<MemoryRecordView>,
        has_more: bool,
    ) {
        if scope != self.scope {
            return;
        }
        self.main_loading = false;
        self.main_has_more = has_more;
        if offset == 0 {
            if matches!(self.scope, MemoryRecordScope::Latest) && !self.records.is_empty() {
                self.pending_scroll_restore =
                    (self.main_scroll_offset_y > 1.0).then_some(ScrollRestore {
                        offset_y: self.main_scroll_offset_y,
                        content_height: self.main_content_height,
                    });
                merge_latest_records(&mut self.records, records);
            } else {
                self.records = records;
            }
            return;
        }
        if offset != self.records.len() {
            return;
        }
        for record in records.drain(..) {
            if !self
                .records
                .iter()
                .any(|existing| existing.index == record.index)
            {
                self.records.push(record);
            }
        }
    }

    pub fn apply_linked_records_loaded(
        &mut self,
        memory_index: String,
        offset: usize,
        mut records: Vec<LinkedMemoryRecordView>,
        has_more: bool,
    ) {
        if memory_index != self.linked_memory_index {
            return;
        }
        self.linked_loading = false;
        self.linked_has_more = has_more;
        if offset == 0 {
            self.linked_results = records;
            return;
        }
        if offset != self.linked_results.len() {
            return;
        }
        for record in records.drain(..) {
            if !self.linked_results.iter().any(|existing| {
                existing.record.index == record.record.index
                    && existing.link.from_memory == record.link.from_memory
                    && existing.link.to_memory == record.link.to_memory
                    && existing.link.relation == record.link.relation
                    && existing.link.freeform_relation == record.link.freeform_relation
            }) {
                self.linked_results.push(record);
            }
        }
    }

    pub fn apply_memory_deleted(&mut self, memory_index: &str) {
        self.records.retain(|record| record.index != memory_index);
        self.linked_results.retain(|linked| {
            linked.record.index != memory_index
                && linked.link.from_memory != memory_index
                && linked.link.to_memory != memory_index
        });
        if self.linked_memory_index == memory_index {
            self.linked_memory_index.clear();
            self.linked_results.clear();
            self.linked_has_more = false;
            self.linked_loading = false;
        }
    }

    pub fn active_query(&self) -> Option<&str> {
        match &self.scope {
            MemoryRecordScope::Latest => None,
            MemoryRecordScope::Search { query } => Some(query.as_str()),
        }
    }
}

pub fn ui(
    ui: &mut egui::Ui,
    tab_id: &VisualizerTabId,
    state: &mut MemoriesState,
    commands: &Sender<VisualizerClientMessage>,
) {
    let now = ui.input(|input| input.time);
    if !state.requested_initial_load {
        state.requested_initial_load = true;
        request_main_chunk(tab_id, state, commands, 0, now);
    } else if matches!(state.scope, MemoryRecordScope::Latest)
        && !state.main_loading
        && now - state.last_latest_refresh_at >= LATEST_REFRESH_INTERVAL_SECS
    {
        refresh_latest(tab_id, state, commands, now);
    }

    ui.horizontal(|ui| {
        let response = ui.text_edit_singleline(&mut state.draft_query);
        let query_requested = ui.button(ui.ctx().tr("memory-query-button")).clicked()
            || (response.lost_focus() && ui.input(|input| input.key_pressed(egui::Key::Enter)));
        if query_requested {
            let query = state.draft_query.trim().to_owned();
            if !query.is_empty() {
                start_main_scope(
                    tab_id,
                    state,
                    commands,
                    MemoryRecordScope::Search { query },
                    now,
                );
            }
        }
        if ui.button(ui.ctx().tr("memory-latest-button")).clicked() {
            refresh_latest(tab_id, state, commands, now);
        }
        ui.label(ui.ctx().tr_args(
            "memory-loaded-status",
            &[("loaded", state.records.len().into())],
        ));
    });

    if let Some(query) = state.active_query() {
        ui.label(
            ui.ctx()
                .tr_args("memory-query-label", &[("query", I18nArg::from(query))]),
        );
    }

    let mut actions = Vec::new();
    let records = state.records.clone();
    let output = memory_list(
        ui,
        &records,
        state.main_loading,
        state.main_has_more,
        state.list_generation,
        &mut actions,
    );
    let restored_scroll_offset = preserve_scroll_after_soft_refresh(ui, state, &output);
    state.main_scroll_offset_y = restored_scroll_offset.unwrap_or(output.state.offset.y);
    state.main_content_height = output.content_size.y;
    for action in actions {
        match action {
            MemoryListAction::OpenLinked(index) => open_linked(tab_id, state, commands, index),
            MemoryListAction::Delete(index) => delete_memory(tab_id, commands, index),
        }
    }
    if should_load_more(&output) {
        request_main_chunk(tab_id, state, commands, state.records.len(), now);
    }

    if !state.linked_memory_index.is_empty() {
        ui.separator();
        ui.label(ui.ctx().tr_args(
            "memory-linked-title",
            &[("index", I18nArg::from(state.linked_memory_index.as_str()))],
        ));
        let linked_results = state.linked_results.clone();
        let output = linked_memory_list(
            ui,
            &linked_results,
            state.linked_loading,
            state.linked_has_more,
            state.linked_generation,
        );
        if should_load_more(&output) {
            request_linked_chunk(tab_id, state, commands, state.linked_results.len());
        }
    }
}

fn start_main_scope(
    tab_id: &VisualizerTabId,
    state: &mut MemoriesState,
    commands: &Sender<VisualizerClientMessage>,
    scope: MemoryRecordScope,
    now: f64,
) {
    state.scope = scope;
    state.records.clear();
    state.main_has_more = true;
    state.main_loading = false;
    state.linked_memory_index.clear();
    state.linked_results.clear();
    state.linked_has_more = false;
    state.linked_loading = false;
    state.list_generation = state.list_generation.saturating_add(1);
    request_main_chunk(tab_id, state, commands, 0, now);
}

fn refresh_latest(
    tab_id: &VisualizerTabId,
    state: &mut MemoriesState,
    commands: &Sender<VisualizerClientMessage>,
    now: f64,
) {
    if matches!(state.scope, MemoryRecordScope::Latest) && !state.records.is_empty() {
        request_latest_refresh(tab_id, state, commands, now);
    } else {
        start_main_scope(tab_id, state, commands, MemoryRecordScope::Latest, now);
        state.last_latest_refresh_at = now;
    }
}

fn request_main_chunk(
    tab_id: &VisualizerTabId,
    state: &mut MemoriesState,
    commands: &Sender<VisualizerClientMessage>,
    offset: usize,
    now: f64,
) {
    if state.main_loading || !state.main_has_more {
        return;
    }
    let sent = commands.send(VisualizerClientMessage::Command {
        command: VisualizerCommand::LoadMemoryRecords {
            tab_id: tab_id.clone(),
            scope: state.scope.clone(),
            offset,
            limit: MAIN_CHUNK_SIZE,
        },
    });
    if sent.is_ok() {
        state.main_loading = true;
        if matches!(state.scope, MemoryRecordScope::Latest) && offset == 0 {
            state.last_latest_refresh_at = now;
        }
    }
}

fn request_latest_refresh(
    tab_id: &VisualizerTabId,
    state: &mut MemoriesState,
    commands: &Sender<VisualizerClientMessage>,
    now: f64,
) {
    if state.main_loading {
        return;
    }
    if commands
        .send(VisualizerClientMessage::Command {
            command: VisualizerCommand::LoadMemoryRecords {
                tab_id: tab_id.clone(),
                scope: MemoryRecordScope::Latest,
                offset: 0,
                limit: MAIN_CHUNK_SIZE,
            },
        })
        .is_ok()
    {
        state.main_loading = true;
        state.last_latest_refresh_at = now;
    }
}

fn open_linked(
    tab_id: &VisualizerTabId,
    state: &mut MemoriesState,
    commands: &Sender<VisualizerClientMessage>,
    index: String,
) {
    state.linked_memory_index = index;
    state.linked_results.clear();
    state.linked_has_more = true;
    state.linked_loading = false;
    state.linked_generation = state.linked_generation.saturating_add(1);
    request_linked_chunk(tab_id, state, commands, 0);
}

fn request_linked_chunk(
    tab_id: &VisualizerTabId,
    state: &mut MemoriesState,
    commands: &Sender<VisualizerClientMessage>,
    offset: usize,
) {
    if state.linked_loading || !state.linked_has_more || state.linked_memory_index.is_empty() {
        return;
    }
    if commands
        .send(VisualizerClientMessage::Command {
            command: VisualizerCommand::LoadLinkedMemories {
                tab_id: tab_id.clone(),
                memory_index: state.linked_memory_index.clone(),
                relation_filter: Vec::new(),
                offset,
                limit: LINKED_CHUNK_SIZE,
            },
        })
        .is_ok()
    {
        state.linked_loading = true;
    }
}

fn delete_memory(
    tab_id: &VisualizerTabId,
    commands: &Sender<VisualizerClientMessage>,
    index: String,
) {
    let _ = commands.send(VisualizerClientMessage::Command {
        command: VisualizerCommand::DeleteMemory {
            tab_id: tab_id.clone(),
            memory_index: index,
        },
    });
}

enum MemoryListAction {
    OpenLinked(String),
    Delete(String),
}

fn memory_list(
    ui: &mut egui::Ui,
    records: &[MemoryRecordView],
    loading: bool,
    has_more: bool,
    generation: u64,
    actions: &mut Vec<MemoryListAction>,
) -> ScrollAreaOutput<()> {
    egui::ScrollArea::vertical()
        .id_salt(("memory-list", generation))
        .show(ui, |ui| {
            for record in records {
                memory_record_card(ui, record, actions);
                ui.add_space(6.0);
            }
            list_footer(ui, loading, has_more);
        })
}

fn linked_memory_list(
    ui: &mut egui::Ui,
    records: &[LinkedMemoryRecordView],
    loading: bool,
    has_more: bool,
    generation: u64,
) -> ScrollAreaOutput<()> {
    egui::ScrollArea::vertical()
        .id_salt(("linked-memory-list", generation))
        .show(ui, |ui| {
            for linked in records {
                ui.horizontal_wrapped(|ui| {
                    ui.strong(&linked.link.relation);
                    ui.label(format!(
                        "{} -> {}",
                        linked.link.from_memory, linked.link.to_memory
                    ));
                    ui.label(&linked.record.index);
                });
                wrapped_label(ui, &linked.record.content);
                ui.add_space(6.0);
            }
            list_footer(ui, loading, has_more);
        })
}

fn memory_record_card(
    ui: &mut egui::Ui,
    record: &MemoryRecordView,
    actions: &mut Vec<MemoryListAction>,
) {
    egui::Frame::new()
        .stroke(ui.visuals().widgets.noninteractive.bg_stroke)
        .corner_radius(egui::CornerRadius::same(6))
        .inner_margin(egui::Margin::same(8))
        .show(ui, |ui| {
            ui.horizontal_wrapped(|ui| {
                ui.strong(&record.kind);
                ui.strong(&record.rank);
                ui.label(&record.index);
                ui.label(
                    record
                        .occurred_at
                        .map(format_jst_datetime)
                        .unwrap_or_else(|| "-".to_string()),
                );
                ui.label(ui.ctx().tr_args(
                    "memory-stored-at",
                    &[("stored_at", format_jst_datetime(record.stored_at).into())],
                ));
                ui.label(ui.ctx().tr_args(
                    "memory-arousal",
                    &[("value", format!("{:.2}", record.affect_arousal).into())],
                ));
                ui.label(ui.ctx().tr_args(
                    "memory-valence",
                    &[("value", format!("{:.2}", record.valence).into())],
                ));
                if !record.emotion.trim().is_empty() {
                    ui.label(ui.ctx().tr_args(
                        "memory-emotion",
                        &[("emotion", I18nArg::from(record.emotion.trim()))],
                    ));
                }
                if ui
                    .small_button(ui.ctx().tr("memory-links-button"))
                    .clicked()
                {
                    actions.push(MemoryListAction::OpenLinked(record.index.clone()));
                }
                if ui
                    .small_button(ui.ctx().tr("memory-delete-button"))
                    .clicked()
                {
                    actions.push(MemoryListAction::Delete(record.index.clone()));
                }
            });
            if !record.concepts.is_empty() || !record.tags.is_empty() {
                ui.horizontal_wrapped(|ui| {
                    for concept in &record.concepts {
                        ui.label(ui.ctx().tr_args(
                            "memory-concept-label",
                            &[("concept", I18nArg::from(concept.label.as_str()))],
                        ));
                    }
                    for tag in &record.tags {
                        ui.label(ui.ctx().tr_args(
                            "memory-tag-label",
                            &[
                                ("namespace", I18nArg::from(tag.namespace.as_str())),
                                ("tag", I18nArg::from(tag.label.as_str())),
                            ],
                        ));
                    }
                });
            }
            ui.add_space(4.0);
            wrapped_label(ui, &record.content);
        });
}

fn list_footer(ui: &mut egui::Ui, loading: bool, has_more: bool) {
    if loading {
        ui.label(ui.ctx().tr("memory-loading"));
    } else if !has_more {
        ui.label(ui.ctx().tr("memory-end"));
    }
}

fn should_load_more(output: &ScrollAreaOutput<()>) -> bool {
    let remaining = output.content_size.y - output.state.offset.y - output.inner_rect.height();
    remaining <= LOAD_MORE_THRESHOLD_PX
}

fn merge_latest_records(existing: &mut Vec<MemoryRecordView>, refreshed: Vec<MemoryRecordView>) {
    let mut merged = Vec::with_capacity(existing.len().max(refreshed.len()));
    for record in refreshed {
        if !merged
            .iter()
            .any(|existing: &MemoryRecordView| existing.index == record.index)
        {
            merged.push(record);
        }
    }
    for record in existing.drain(..) {
        if !merged
            .iter()
            .any(|existing: &MemoryRecordView| existing.index == record.index)
        {
            merged.push(record);
        }
    }
    *existing = merged;
}

fn preserve_scroll_after_soft_refresh(
    ui: &egui::Ui,
    state: &mut MemoriesState,
    output: &ScrollAreaOutput<()>,
) -> Option<f32> {
    let Some(restore) = state.pending_scroll_restore.take() else {
        return None;
    };
    let delta = output.content_size.y - restore.content_height;
    if delta <= 0.0 {
        return None;
    }
    let mut scroll_state = output.state;
    scroll_state.offset.y = restore.offset_y + delta;
    scroll_state.store(ui.ctx(), output.id);
    ui.ctx().request_repaint();
    Some(scroll_state.offset.y)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn record(index: &str, content: &str) -> MemoryRecordView {
        MemoryRecordView {
            index: index.to_string(),
            kind: "Statement".to_string(),
            rank: "ShortTerm".to_string(),
            occurred_at: None,
            stored_at: chrono::Utc::now(),
            concepts: Vec::new(),
            tags: Vec::new(),
            affect_arousal: 0.0,
            valence: 0.0,
            emotion: String::new(),
            content: content.to_string(),
        }
    }

    #[test]
    fn latest_refresh_merges_records_without_clearing_existing_list() {
        let mut state = MemoriesState::default();
        state.records = vec![record("old", "old content")];
        state.main_scroll_offset_y = 120.0;
        state.main_content_height = 1_000.0;

        state.apply_records_loaded(
            MemoryRecordScope::Latest,
            0,
            vec![record("new", "new content"), record("old", "updated old")],
            true,
        );

        assert_eq!(
            state
                .records
                .iter()
                .map(|record| (record.index.as_str(), record.content.as_str()))
                .collect::<Vec<_>>(),
            vec![("new", "new content"), ("old", "updated old")]
        );
        assert!(state.main_has_more);
        assert!(state.pending_scroll_restore.is_some());
    }

    #[test]
    fn latest_refresh_at_top_does_not_schedule_scroll_restore() {
        let mut state = MemoriesState::default();
        state.records = vec![record("old", "old content")];
        state.main_scroll_offset_y = 0.0;
        state.main_content_height = 1_000.0;

        state.apply_records_loaded(
            MemoryRecordScope::Latest,
            0,
            vec![record("new", "new content"), record("old", "updated old")],
            true,
        );

        assert!(state.pending_scroll_restore.is_none());
    }
}
