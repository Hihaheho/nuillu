use crate::{
    CognitionEntryView, CognitionLogView, i18n::localized_module_name_with_id, text::wrapped_label,
    time::format_jst_datetime,
};

pub fn ui(ui: &mut egui::Ui, logs: &[CognitionLogView]) {
    let entries = cognition_entries_newest_first(logs);
    egui::ScrollArea::vertical()
        .id_salt("cognition-log-window-list")
        .show(ui, |ui| {
            for entry in entries {
                egui::Frame::new()
                    .stroke(ui.visuals().widgets.noninteractive.bg_stroke)
                    .corner_radius(egui::CornerRadius::same(6))
                    .inner_margin(egui::Margin::same(8))
                    .show(ui, |ui| {
                        ui.horizontal_wrapped(|ui| {
                            ui.strong(cognition_header_label(
                                ui.ctx(),
                                entry.source,
                                &entry.entry.origin,
                            ));
                            ui.label(format_jst_datetime(entry.entry.at));
                        });
                        ui.add_space(4.0);
                        wrapped_label(ui, &entry.entry.text);
                    });
                ui.add_space(6.0);
            }
        });
}

#[derive(Debug, Clone, Copy)]
struct CognitionDisplayEntry<'a> {
    source: &'a str,
    entry: &'a CognitionEntryView,
}

fn cognition_entries_newest_first(logs: &[CognitionLogView]) -> Vec<CognitionDisplayEntry<'_>> {
    let mut entries = logs
        .iter()
        .flat_map(|log| {
            log.entries.iter().map(|entry| CognitionDisplayEntry {
                source: log.source.as_str(),
                entry,
            })
        })
        .collect::<Vec<_>>();
    entries.sort_by(|left, right| {
        right
            .entry
            .at
            .cmp(&left.entry.at)
            .then_with(|| left.source.cmp(right.source))
            .then_with(|| left.entry.text.cmp(&right.entry.text))
    });
    entries
}

fn cognition_header_label(ctx: &egui::Context, source: &str, origin: &str) -> String {
    let source_label = localized_module_name_with_id(ctx, source);
    if origin == source {
        return source_label;
    }
    format!(
        "{source_label} ({})",
        localized_module_name_with_id(ctx, origin)
    )
}

#[cfg(test)]
mod tests {
    use chrono::{TimeZone, Utc};

    use crate::i18n::{EguiI18nExt as _, I18nCatalog, Locale};

    use super::*;

    #[test]
    fn cognition_entries_mix_sources_newest_first() {
        let logs = vec![
            cognition_log(
                "attention-schema",
                &[(20, "middle attention"), (10, "old attention")],
            ),
            cognition_log("surprise", &[(30, "new surprise")]),
        ];

        let entries = cognition_entries_newest_first(&logs);

        assert_eq!(entries.len(), 3);
        assert_eq!(entries[0].source, "surprise");
        assert_eq!(entries[0].entry.text, "new surprise");
        assert_eq!(entries[1].source, "attention-schema");
        assert_eq!(entries[1].entry.text, "middle attention");
        assert_eq!(entries[2].source, "attention-schema");
        assert_eq!(entries[2].entry.text, "old attention");
    }

    #[test]
    fn cognition_header_includes_distinct_origin() {
        let ctx = egui::Context::default();
        let catalog = I18nCatalog::embedded().unwrap();
        ctx.install_i18n(catalog.for_locale(Locale::EnUs));

        assert_eq!(
            cognition_header_label(&ctx, "cognition-gate", "sensory"),
            "cognition-gate (sensory)"
        );
        assert_eq!(
            cognition_header_label(&ctx, "interpreter", "interpreter"),
            "interpreter"
        );
    }

    fn cognition_log(source: &str, entries: &[(i64, &str)]) -> CognitionLogView {
        CognitionLogView {
            source: source.to_string(),
            entries: entries
                .iter()
                .map(|(second, text)| CognitionEntryView {
                    at: Utc.timestamp_opt(*second, 0).unwrap(),
                    origin: source.to_string(),
                    text: (*text).to_string(),
                })
                .collect(),
        }
    }
}
