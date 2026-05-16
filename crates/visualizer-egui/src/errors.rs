use std::collections::VecDeque;

use nuillu_visualizer_protocol::VisualizerErrorView;

use crate::text::wrapped_label;

pub fn ui(ui: &mut egui::Ui, errors: &mut VecDeque<VisualizerErrorView>) {
    ui.horizontal_wrapped(|ui| {
        ui.heading("Errors");
        ui.label(format!("count: {}", errors.len()));
        if ui.button("Clear").clicked() {
            errors.clear();
        }
    });
    ui.separator();

    if errors.is_empty() {
        ui.label("No errors reported.");
        return;
    }

    egui::ScrollArea::vertical()
        .id_salt("errors")
        .show(ui, |ui| {
            for (index, error) in errors.iter().rev().enumerate() {
                ui.push_id(("error", index), |ui| {
                    error_row(ui, error);
                });
                ui.add_space(6.0);
            }
        });
}

fn error_row(ui: &mut egui::Ui, error: &VisualizerErrorView) {
    egui::Frame::new()
        .fill(ui.visuals().extreme_bg_color)
        .stroke(ui.visuals().widgets.noninteractive.bg_stroke)
        .corner_radius(egui::CornerRadius::same(6))
        .inner_margin(egui::Margin::same(8))
        .show(ui, |ui| {
            ui.horizontal_wrapped(|ui| {
                ui.colored_label(ui.visuals().error_fg_color, &error.phase);
                ui.label(&error.source);
                ui.label(error.at.format("%H:%M:%S%.3f").to_string());
                if let Some(owner) = &error.owner {
                    wrapped_label(ui, owner);
                }
            });
            ui.add_space(4.0);
            wrapped_label(ui, &error.message);
        });
}
