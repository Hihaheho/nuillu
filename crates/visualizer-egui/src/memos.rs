use crate::{
    MemoView, i18n::EguiI18nExt as _, module_filter, module_filter::ModuleFilterState,
    text::wrapped_label,
};

pub fn ui(
    ui: &mut egui::Ui,
    memos: &[MemoView],
    filter: &mut ModuleFilterState,
    modules: &[String],
) {
    module_filter::render_module_filter(ui, "memo-module-filter", filter, modules);
    ui.separator();
    let memos = filtered_memos(memos, filter);
    egui::ScrollArea::vertical()
        .id_salt("memo-window-list")
        .show(ui, |ui| {
            if memos.is_empty() {
                ui.label(ui.ctx().tr("memo-empty"));
                return;
            }
            for memo in memos {
                render_memo_card(ui, memo);
                ui.add_space(6.0);
            }
        });
}

pub fn filtered_memos<'a>(memos: &'a [MemoView], filter: &ModuleFilterState) -> Vec<&'a MemoView> {
    memos
        .iter()
        .filter(|memo| filter.is_selected(&memo.module))
        .collect()
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
                ui.label(
                    ui.ctx()
                        .tr_args("memo-index", &[("index", memo.index.to_string().into())]),
                );
                ui.label(memo.written_at.to_rfc3339());
            });
            ui.add_space(4.0);
            wrapped_label(ui, &memo.content);
        });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn filtered_memos_uses_module_filter() {
        let now = chrono::Utc::now();
        let memos = vec![
            MemoView {
                owner: "sensory".to_string(),
                module: "sensory".to_string(),
                replica: 0,
                index: 0,
                written_at: now,
                content: "sensory memo".to_string(),
            },
            MemoView {
                owner: "memory".to_string(),
                module: "memory".to_string(),
                replica: 0,
                index: 1,
                written_at: now,
                content: "memory memo".to_string(),
            },
        ];
        let mut filter = ModuleFilterState::default();
        filter.deselect_all();
        filter.set_selected("memory".to_string(), true);

        let visible = filtered_memos(&memos, &filter)
            .into_iter()
            .map(|memo| memo.module.as_str())
            .collect::<Vec<_>>();

        assert_eq!(visible, vec!["memory"]);
    }
}
