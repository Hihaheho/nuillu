use crate::{CognitionLogView, text::wrapped_label};

pub fn ui(ui: &mut egui::Ui, logs: &[CognitionLogView]) {
    egui::ScrollArea::vertical()
        .id_salt("cognition-log-window-list")
        .show(ui, |ui| {
            for log in logs {
                ui.heading(&log.source);
                for entry in &log.entries {
                    egui::Frame::new()
                        .stroke(ui.visuals().widgets.noninteractive.bg_stroke)
                        .corner_radius(egui::CornerRadius::same(6))
                        .inner_margin(egui::Margin::same(8))
                        .show(ui, |ui| {
                            ui.label(entry.at.to_rfc3339());
                            ui.add_space(4.0);
                            wrapped_label(ui, &entry.text);
                        });
                    ui.add_space(6.0);
                }
                ui.separator();
            }
        });
}
