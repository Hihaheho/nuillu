use std::sync::mpsc::Sender;

use crate::{
    LinkedMemoryRecordView, MemoryPage, MemoryRecordView, VisualizerClientMessage,
    VisualizerCommand, VisualizerTabId, text::wrapped_label,
};

#[derive(Debug)]
pub struct MemoriesState {
    pub query: String,
    pub query_results: Vec<MemoryRecordView>,
    pub linked_memory_index: String,
    pub linked_results: Vec<LinkedMemoryRecordView>,
    pub page: MemoryPage,
    draft_query: String,
    page_index: usize,
    per_page: usize,
    requested_initial_page: bool,
}

impl Default for MemoriesState {
    fn default() -> Self {
        Self {
            query: String::new(),
            query_results: Vec::new(),
            linked_memory_index: String::new(),
            linked_results: Vec::new(),
            page: MemoryPage {
                page: 0,
                per_page: 25,
                total: 0,
                records: Vec::new(),
            },
            draft_query: String::new(),
            page_index: 0,
            per_page: 25,
            requested_initial_page: false,
        }
    }
}

pub fn ui(
    ui: &mut egui::Ui,
    tab_id: &VisualizerTabId,
    state: &mut MemoriesState,
    commands: &Sender<VisualizerClientMessage>,
) {
    if !state.requested_initial_page {
        state.requested_initial_page = true;
        let _ = commands.send(VisualizerClientMessage::Command {
            command: VisualizerCommand::ListMemories {
                tab_id: tab_id.clone(),
                page: state.page_index,
                per_page: state.per_page,
            },
        });
    }

    ui.horizontal(|ui| {
        let response = ui.text_edit_singleline(&mut state.draft_query);
        let query_requested = ui.button("Query").clicked()
            || (response.lost_focus() && ui.input(|input| input.key_pressed(egui::Key::Enter)));
        if query_requested {
            let query = state.draft_query.trim().to_owned();
            if !query.is_empty() {
                let _ = commands.send(VisualizerClientMessage::Command {
                    command: VisualizerCommand::QueryMemory {
                        tab_id: tab_id.clone(),
                        query,
                        limit: state.per_page,
                    },
                });
            }
        }
        if ui.button("Latest").clicked() {
            state.query.clear();
            state.query_results.clear();
            let _ = commands.send(VisualizerClientMessage::Command {
                command: VisualizerCommand::ListMemories {
                    tab_id: tab_id.clone(),
                    page: state.page_index,
                    per_page: state.per_page,
                },
            });
        }
    });

    ui.horizontal(|ui| {
        if ui.button("Prev").clicked() {
            state.page_index = state.page_index.saturating_sub(1);
            let _ = commands.send(VisualizerClientMessage::Command {
                command: VisualizerCommand::ListMemories {
                    tab_id: tab_id.clone(),
                    page: state.page_index,
                    per_page: state.per_page,
                },
            });
        }
        ui.label(format!(
            "page {} / total {}",
            state.page.page + 1,
            state.page.total
        ));
        if ui.button("Next").clicked() {
            state.page_index = state.page_index.saturating_add(1);
            let _ = commands.send(VisualizerClientMessage::Command {
                command: VisualizerCommand::ListMemories {
                    tab_id: tab_id.clone(),
                    page: state.page_index,
                    per_page: state.per_page,
                },
            });
        }
    });

    let records = if state.query.is_empty() {
        &state.page.records
    } else {
        ui.label(format!("Query: {}", state.query));
        &state.query_results
    };
    memory_list(
        ui,
        tab_id,
        records,
        commands,
        state.page_index,
        state.per_page,
    );

    if !state.linked_results.is_empty() {
        ui.separator();
        ui.label(format!("Linked memories: {}", state.linked_memory_index));
        egui::ScrollArea::vertical()
            .id_salt("linked-memory-list")
            .max_height(220.0)
            .show(ui, |ui| {
                for linked in &state.linked_results {
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
            });
    }
}

fn memory_list(
    ui: &mut egui::Ui,
    tab_id: &VisualizerTabId,
    records: &[MemoryRecordView],
    commands: &Sender<VisualizerClientMessage>,
    page: usize,
    per_page: usize,
) {
    egui::ScrollArea::vertical()
        .id_salt("memory-list")
        .show(ui, |ui| {
            for record in records {
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
                                    .map(|at| at.to_rfc3339())
                                    .unwrap_or_else(|| "-".to_string()),
                            );
                            ui.label(format!("stored {}", record.stored_at.to_rfc3339()));
                            ui.label(format!("arousal {:.2}", record.affect_arousal));
                            ui.label(format!("valence {:.2}", record.valence));
                            if !record.emotion.trim().is_empty() {
                                ui.label(format!("emotion {}", record.emotion.trim()));
                            }
                            if ui.small_button("Links").clicked() {
                                let _ = commands.send(VisualizerClientMessage::Command {
                                    command: VisualizerCommand::FetchLinkedMemories {
                                        tab_id: tab_id.clone(),
                                        memory_index: record.index.clone(),
                                        relation_filter: Vec::new(),
                                        limit: 16,
                                    },
                                });
                            }
                            if ui.small_button("Delete").clicked() {
                                let _ = commands.send(VisualizerClientMessage::Command {
                                    command: VisualizerCommand::DeleteMemory {
                                        tab_id: tab_id.clone(),
                                        memory_index: record.index.clone(),
                                        page,
                                        per_page,
                                    },
                                });
                            }
                        });
                        if !record.concepts.is_empty() || !record.tags.is_empty() {
                            ui.horizontal_wrapped(|ui| {
                                for concept in &record.concepts {
                                    ui.label(format!("concept:{}", concept.label));
                                }
                                for tag in &record.tags {
                                    ui.label(format!("tag:{}:{}", tag.namespace, tag.label));
                                }
                            });
                        }
                        ui.add_space(4.0);
                        wrapped_label(ui, &record.content);
                    });
                ui.add_space(6.0);
            }
        });
}
