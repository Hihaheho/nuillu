use crate::{
    CognitionEntryView, CognitionLogView, i18n::localized_module_name_with_id, text::wrapped_label,
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
                            ui.strong(localized_module_name_with_id(ui.ctx(), entry.source));
                            ui.label(entry.entry.at.to_rfc3339());
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

#[cfg(test)]
mod tests {
    use chrono::{TimeZone, Utc};

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

    fn cognition_log(source: &str, entries: &[(i64, &str)]) -> CognitionLogView {
        CognitionLogView {
            source: source.to_string(),
            entries: entries
                .iter()
                .map(|(second, text)| CognitionEntryView {
                    at: Utc.timestamp_opt(*second, 0).unwrap(),
                    text: (*text).to_string(),
                })
                .collect(),
        }
    }
}
