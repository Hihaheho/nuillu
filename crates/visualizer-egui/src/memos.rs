use crate::{MemoView, text::wrapped_label};

pub fn ui(ui: &mut egui::Ui, memos: &[MemoView]) {
    egui::ScrollArea::vertical()
        .id_salt("memo-window-list")
        .show(ui, |ui| {
            for memo in memos {
                render_memo_card(ui, memo);
                ui.add_space(6.0);
            }
        });
}

pub fn render_memo_card(ui: &mut egui::Ui, memo: &MemoView) {
    egui::Frame::new()
        .stroke(ui.visuals().widgets.noninteractive.bg_stroke)
        .corner_radius(egui::CornerRadius::same(6))
        .inner_margin(egui::Margin::same(8))
        .show(ui, |ui| {
            ui.horizontal_wrapped(|ui| {
                ui.strong(&memo.owner);
                ui.label(format!("{}#{}", memo.module, memo.replica));
                ui.label(format!("memo {}", memo.index));
                ui.label(memo.written_at.to_rfc3339());
            });
            ui.add_space(4.0);
            wrapped_label(ui, &memo.content);
        });
}
