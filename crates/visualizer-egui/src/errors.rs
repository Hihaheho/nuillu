use std::collections::VecDeque;

use nuillu_visualizer_protocol::VisualizerErrorView;

use crate::{
    i18n::{EguiI18nExt as _, localized_module_name_with_id},
    text::wrapped_label,
};

pub fn ui(
    ui: &mut egui::Ui,
    errors: &mut VecDeque<VisualizerErrorView>,
    session_error_count: u32,
    live_llm_turn_count: u32,
) {
    ui.horizontal_wrapped(|ui| {
        ui.heading(ui.ctx().tr("errors-heading"));
        ui.label(ui.ctx().tr_args(
            "errors-session-status",
            &[
                ("errors", session_error_count.into()),
                ("turns", live_llm_turn_count.into()),
            ],
        ))
        .on_hover_text(ui.ctx().tr("errors-session-status-hover"));
        ui.label(
            ui.ctx()
                .tr_args("errors-shown-count", &[("count", errors.len().into())]),
        );
        if ui.button(ui.ctx().tr("errors-clear-list")).clicked() {
            errors.clear();
        }
    });
    ui.separator();

    if errors.is_empty() {
        ui.label(ui.ctx().tr("errors-empty"));
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
                ui.label(localized_module_name_with_id(ui.ctx(), &error.source));
                ui.label(error.at.format("%H:%M:%S%.3f").to_string());
                if let Some(owner) = &error.owner {
                    wrapped_label(ui, &localized_module_name_with_id(ui.ctx(), owner));
                }
            });
            ui.add_space(4.0);
            wrapped_label(ui, &error.message);
        });
}
