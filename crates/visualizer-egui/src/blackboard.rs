use crate::{BlackboardSnapshot, text::wrapped_label};

pub fn ui(ui: &mut egui::Ui, snapshot: &BlackboardSnapshot) {
    egui::ScrollArea::vertical()
        .id_salt("blackboard-window")
        .show(ui, |ui| {
            ui.heading("Allocation");
            for item in &snapshot.allocation {
                egui::Frame::new()
                    .stroke(ui.visuals().widgets.noninteractive.bg_stroke)
                    .corner_radius(egui::CornerRadius::same(6))
                    .inner_margin(egui::Margin::same(8))
                    .show(ui, |ui| {
                        ui.horizontal_wrapped(|ui| {
                            ui.strong(&item.module);
                            ui.label(format!("replicas {}", item.active_replicas));
                            ui.label(format!("ratio {:.2}", item.activation_ratio));
                            ui.label(&item.tier);
                        });
                        if !item.guidance.is_empty() {
                            ui.add_space(4.0);
                            wrapped_label(ui, &item.guidance);
                        }
                    });
                ui.add_space(6.0);
            }

            ui.separator();
            ui.heading("Module Status");
            for item in &snapshot.module_statuses {
                ui.horizontal_wrapped(|ui| {
                    ui.strong(&item.owner);
                    ui.label(&item.status);
                });
            }

            ui.separator();
            ui.heading("Utterance Progress");
            for progress in &snapshot.utterance_progresses {
                egui::Frame::new()
                    .stroke(ui.visuals().widgets.noninteractive.bg_stroke)
                    .corner_radius(egui::CornerRadius::same(6))
                    .inner_margin(egui::Margin::same(8))
                    .show(ui, |ui| {
                        ui.horizontal_wrapped(|ui| {
                            ui.strong(&progress.owner);
                            ui.label(&progress.state);
                            ui.label(format!("#{}:{}", progress.generation_id, progress.sequence));
                            ui.label(format!("target: {}", progress.target));
                        });
                        ui.add_space(4.0);
                        wrapped_label(ui, &progress.partial_utterance);
                    });
                ui.add_space(6.0);
            }

            ui.separator();
            ui.heading("Memory Metadata");
            egui::Grid::new("memory-metadata-grid")
                .striped(true)
                .show(ui, |ui| {
                    ui.strong("Index");
                    ui.strong("Rank");
                    ui.strong("Accesses");
                    ui.strong("Uses");
                    ui.strong("Reinforces");
                    ui.strong("Occurred");
                    ui.end_row();
                    for memory in &snapshot.memory_metadata {
                        ui.label(&memory.index);
                        ui.label(&memory.rank);
                        ui.label(memory.access_count.to_string());
                        ui.label(memory.use_count.to_string());
                        ui.label(memory.reinforcement_count.to_string());
                        ui.label(
                            memory
                                .occurred_at
                                .map(|at| at.to_rfc3339())
                                .unwrap_or_else(|| "-".to_string()),
                        );
                        ui.end_row();
                    }
                });
        });
}
