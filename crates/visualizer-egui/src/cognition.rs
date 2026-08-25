use std::{
    collections::{HashMap, HashSet},
    hash::{DefaultHasher, Hash as _, Hasher as _},
};

use crate::{
    CognitionLogView, PersistedCognitionEntryView, VisualizerClientMessage, VisualizerCommand,
    VisualizerTabId, i18n::localized_module_name_with_id, text::hard_wrap_long_segments,
    time::format_jst_datetime,
};

pub(crate) const COGNITION_CHUNK_SIZE: usize = 100;
const CARD_MARGIN: f32 = 8.0;
const CARD_GAP: f32 = 6.0;
const HEADER_GAP: f32 = 4.0;
const HARD_WRAP_LIMIT: usize = 96;

#[derive(Debug, Default)]
pub struct CognitionState {
    entries: Vec<PersistedCognitionEntryView>,
    snapshot_entries: Vec<PersistedCognitionEntryView>,
    loaded_initial: bool,
    has_more: bool,
    loading: bool,
    requested_initial: bool,
    refresh_needed: bool,
    snapshot_fingerprint: Option<u64>,
    layout_width_bits: u32,
    height_cache: HashMap<i64, f32>,
}

impl CognitionState {
    pub fn observe_snapshot(&mut self, logs: &[CognitionLogView]) {
        let mut hasher = DefaultHasher::new();
        for log in logs {
            log.source.hash(&mut hasher);
            for entry in &log.entries {
                entry.at.hash(&mut hasher);
                entry.origin.hash(&mut hasher);
                entry.text.hash(&mut hasher);
            }
        }
        let fingerprint = hasher.finish();
        if self.snapshot_fingerprint.is_some() && self.snapshot_fingerprint != Some(fingerprint) {
            self.refresh_needed = true;
        }
        self.snapshot_fingerprint = Some(fingerprint);
        let mut snapshot_entries = logs
            .iter()
            .flat_map(|log| {
                log.entries.iter().map(|entry| {
                    let mut hasher = DefaultHasher::new();
                    log.source.hash(&mut hasher);
                    entry.at.hash(&mut hasher);
                    entry.origin.hash(&mut hasher);
                    entry.text.hash(&mut hasher);
                    PersistedCognitionEntryView {
                        id: i64::from_ne_bytes(hasher.finish().to_ne_bytes()),
                        source: log.source.clone(),
                        at: entry.at,
                        origin: entry.origin.clone(),
                        text: entry.text.clone(),
                    }
                })
            })
            .collect::<Vec<_>>();
        snapshot_entries
            .sort_by(|left, right| right.at.cmp(&left.at).then_with(|| right.id.cmp(&left.id)));
        self.snapshot_entries = snapshot_entries;
    }

    pub fn apply_page(
        &mut self,
        offset: usize,
        entries: Vec<PersistedCognitionEntryView>,
        has_more: bool,
    ) {
        self.loading = false;
        self.refresh_needed = false;
        self.has_more = has_more;
        if offset == 0 {
            self.loaded_initial = true;
        }
        if offset == 0 && self.entries.is_empty() {
            self.entries = entries;
            return;
        }

        let mut known = self
            .entries
            .iter()
            .map(|entry| entry.id)
            .collect::<HashSet<_>>();
        self.entries
            .extend(entries.into_iter().filter(|entry| known.insert(entry.id)));
        self.entries
            .sort_by_key(|entry| std::cmp::Reverse(entry.id));
    }
}

#[derive(Clone, Copy, Default)]
struct ViewportAnchor {
    first_id: Option<i64>,
    offset: f32,
}

pub fn ui(
    ui: &mut egui::Ui,
    id_salt: impl std::hash::Hash,
    tab_id: &VisualizerTabId,
    state: &mut CognitionState,
    messages: &mut Vec<VisualizerClientMessage>,
) {
    if (!state.requested_initial || state.refresh_needed) && !state.loading {
        request_page(tab_id, state, messages, 0);
    }

    let entries = if state.loaded_initial {
        &state.entries
    } else {
        &state.snapshot_entries
    };

    let id = ui.make_persistent_id(id_salt);
    let available_width = ui.available_width().max(120.0);
    let width_bits = available_width.to_bits();
    if state.layout_width_bits != width_bits {
        state.layout_width_bits = width_bits;
        state.height_cache.clear();
    }
    let heights = entries
        .iter()
        .map(|entry| {
            *state
                .height_cache
                .entry(entry.id)
                .or_insert_with(|| entry_height(ui, entry, available_width))
        })
        .collect::<Vec<_>>();
    let mut offsets = Vec::with_capacity(heights.len() + 1);
    offsets.push(0.0);
    for height in &heights {
        offsets.push(offsets.last().copied().unwrap_or_default() + height);
    }
    let total_height = offsets.last().copied().unwrap_or_default();

    let old_anchor = ui.ctx().data(|data| {
        data.get_temp::<ViewportAnchor>(id.with("cognition-viewport-anchor"))
            .unwrap_or_default()
    });
    let corrected_offset = corrected_scroll_offset(old_anchor, entries, &offsets);

    let mut scroll = egui::ScrollArea::vertical().id_salt(id);
    if let Some(offset) = corrected_offset {
        scroll = scroll.vertical_scroll_offset(offset);
    }
    let output = scroll.show_viewport(ui, |ui, viewport| {
        ui.set_height(total_height);
        let start = offsets
            .partition_point(|offset| *offset <= viewport.min.y)
            .saturating_sub(1)
            .min(entries.len());
        let end = offsets
            .partition_point(|offset| *offset < viewport.max.y)
            .min(entries.len());
        for index in start..end {
            let rect = egui::Rect::from_min_size(
                egui::pos2(ui.max_rect().left(), ui.max_rect().top() + offsets[index]),
                egui::vec2(available_width, heights[index] - CARD_GAP),
            );
            ui.scope_builder(egui::UiBuilder::new().max_rect(rect), |ui| {
                render_entry(ui, &entries[index]);
            });
        }
        viewport.max.y + viewport.height() >= total_height
    });

    ui.ctx().data_mut(|data| {
        data.insert_temp(
            id.with("cognition-viewport-anchor"),
            ViewportAnchor {
                first_id: entries.first().map(|entry| entry.id),
                offset: output.state.offset.y,
            },
        );
    });
    let loaded_count = entries.len();
    if output.inner && state.has_more && !state.loading {
        request_page(tab_id, state, messages, loaded_count);
    }
}

fn corrected_scroll_offset(
    old_anchor: ViewportAnchor,
    entries: &[PersistedCognitionEntryView],
    offsets: &[f32],
) -> Option<f32> {
    let inserted_height = old_anchor
        .first_id
        .and_then(|first_id| entries.iter().position(|entry| entry.id == first_id))
        .map(|position| offsets[position])
        .unwrap_or_default();
    (inserted_height > 0.0).then_some(old_anchor.offset + inserted_height)
}

fn request_page(
    tab_id: &VisualizerTabId,
    state: &mut CognitionState,
    messages: &mut Vec<VisualizerClientMessage>,
    offset: usize,
) {
    state.loading = true;
    state.requested_initial = true;
    messages.push(VisualizerClientMessage::Command {
        command: VisualizerCommand::LoadCognitionLogEntries {
            tab_id: tab_id.clone(),
            offset,
            limit: COGNITION_CHUNK_SIZE,
        },
    });
}

fn entry_height(ui: &egui::Ui, entry: &PersistedCognitionEntryView, available_width: f32) -> f32 {
    let content_width = (available_width - CARD_MARGIN * 2.0).max(40.0);
    let header_height = cognition_header_text(ui.ctx(), entry)
        .into_galley(
            ui,
            Some(egui::TextWrapMode::Wrap),
            content_width,
            egui::FontSelection::Default,
        )
        .size()
        .y;
    let body_height = egui::WidgetText::from(hard_wrap_long_segments(&entry.text, HARD_WRAP_LIMIT))
        .into_galley(
            ui,
            Some(egui::TextWrapMode::Wrap),
            content_width,
            egui::FontSelection::Default,
        )
        .size()
        .y;
    CARD_MARGIN * 2.0 + header_height + HEADER_GAP + body_height + CARD_GAP
}

fn render_entry(ui: &mut egui::Ui, entry: &PersistedCognitionEntryView) {
    egui::Frame::new()
        .stroke(ui.visuals().widgets.noninteractive.bg_stroke)
        .corner_radius(egui::CornerRadius::same(6))
        .inner_margin(egui::Margin::same(CARD_MARGIN as i8))
        .show(ui, |ui| {
            ui.add(egui::Label::new(cognition_header_text(ui.ctx(), entry)).wrap());
            ui.add_space(HEADER_GAP);
            ui.add(egui::Label::new(hard_wrap_long_segments(&entry.text, HARD_WRAP_LIMIT)).wrap());
        });
}

fn cognition_header_text(
    ctx: &egui::Context,
    entry: &PersistedCognitionEntryView,
) -> egui::WidgetText {
    egui::RichText::new(format!(
        "{}  {}",
        cognition_header_label(ctx, &entry.source, &entry.origin),
        format_jst_datetime(entry.at)
    ))
    .strong()
    .into()
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
    fn pages_merge_by_persisted_id_in_newest_first_order() {
        let mut state = CognitionState::default();
        state.apply_page(0, vec![entry(3), entry(2)], true);
        state.apply_page(2, vec![entry(2), entry(1)], false);
        assert_eq!(
            state
                .entries
                .iter()
                .map(|entry| entry.id)
                .collect::<Vec<_>>(),
            vec![3, 2, 1]
        );
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

    #[test]
    fn prepended_entries_keep_the_previous_viewport_content_anchored() {
        let entries = vec![entry(5), entry(4), entry(3)];
        let offset = corrected_scroll_offset(
            ViewportAnchor {
                first_id: Some(3),
                offset: 0.0,
            },
            &entries,
            &[0.0, 40.0, 90.0, 130.0],
        );

        assert_eq!(offset, Some(90.0));
    }

    fn entry(id: i64) -> PersistedCognitionEntryView {
        PersistedCognitionEntryView {
            id,
            source: "cognition-gate".to_owned(),
            at: Utc.timestamp_opt(id, 0).unwrap(),
            origin: "sensory".to_owned(),
            text: format!("entry {id}"),
        }
    }
}
