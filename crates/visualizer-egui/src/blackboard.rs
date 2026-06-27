use crate::{
    BlackboardSnapshot,
    i18n::{EguiI18nExt as _, I18nArg, localized_module_name_with_id},
    text::wrapped_label,
    time::format_jst_datetime,
};

pub fn ui(ui: &mut egui::Ui, snapshot: &BlackboardSnapshot) {
    egui::ScrollArea::vertical()
        .id_salt("blackboard-window")
        .show(ui, |ui| {
            ui.heading(ui.ctx().tr("blackboard-allocation"));
            for item in &snapshot.allocation {
                egui::Frame::new()
                    .stroke(ui.visuals().widgets.noninteractive.bg_stroke)
                    .corner_radius(egui::CornerRadius::same(6))
                    .inner_margin(egui::Margin::same(8))
                    .show(ui, |ui| {
                        ui.horizontal_wrapped(|ui| {
                            ui.strong(localized_module_name_with_id(ui.ctx(), &item.module));
                            ui.label(ui.ctx().tr_args(
                                "blackboard-replicas",
                                &[("count", i64::from(item.active_replicas).into())],
                            ));
                            ui.label(ui.ctx().tr_args(
                                "blackboard-ratio",
                                &[("ratio", format!("{:.2}", item.activation_ratio).into())],
                            ));
                        });
                    });
                ui.add_space(6.0);
            }

            ui.separator();
            ui.heading(ui.ctx().tr("blackboard-module-status"));
            for item in &snapshot.module_statuses {
                ui.horizontal_wrapped(|ui| {
                    ui.strong(localized_module_name_with_id(ui.ctx(), &item.module));
                    if item.owner != item.module {
                        ui.label(&item.owner);
                    }
                    ui.label(&item.status);
                });
            }

            ui.separator();
            ui.heading(ui.ctx().tr("blackboard-utterance-progress"));
            for progress in &snapshot.utterance_progresses {
                egui::Frame::new()
                    .stroke(ui.visuals().widgets.noninteractive.bg_stroke)
                    .corner_radius(egui::CornerRadius::same(6))
                    .inner_margin(egui::Margin::same(8))
                    .show(ui, |ui| {
                        ui.horizontal_wrapped(|ui| {
                            ui.strong(localized_module_name_with_id(ui.ctx(), &progress.owner));
                            ui.label(&progress.state);
                            ui.label(format!("#{}:{}", progress.generation_id, progress.sequence));
                            ui.label(ui.ctx().tr_args(
                                "blackboard-target",
                                &[("target", I18nArg::from(progress.target.as_str()))],
                            ));
                        });
                        ui.add_space(4.0);
                        wrapped_label(ui, &progress.partial_utterance);
                    });
                ui.add_space(6.0);
            }

            ui.separator();
            ui.heading(ui.ctx().tr("blackboard-memory-metadata"));
            egui::Grid::new("memory-metadata-grid")
                .striped(true)
                .show(ui, |ui| {
                    ui.strong(ui.ctx().tr("blackboard-index"));
                    ui.strong(ui.ctx().tr("blackboard-rank"));
                    ui.strong(ui.ctx().tr("blackboard-accesses"));
                    ui.strong(ui.ctx().tr("blackboard-uses"));
                    ui.strong(ui.ctx().tr("blackboard-reinforces"));
                    ui.strong(ui.ctx().tr("blackboard-occurred"));
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
                                .map(format_jst_datetime)
                                .unwrap_or_else(|| "-".to_string()),
                        );
                        ui.end_row();
                    }
                });
        });
}
