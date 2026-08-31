use std::{
    collections::HashMap,
    hash::{DefaultHasher, Hash as _, Hasher as _},
};

use crate::{
    CognitionLogCursor, CognitionLogView, PersistedCognitionEntryView, VisualizerClientMessage,
    VisualizerCommand, VisualizerTabId,
    i18n::{EguiI18nExt as _, localized_module_name_with_id},
    text::hard_wrap_long_segments,
    time::format_jst_datetime,
};

pub(crate) const COGNITION_CHUNK_SIZE: usize = 100;
/// Ceiling on entries held client-side.
///
/// The newest end is never evicted: a refresh prepends and drops from the old
/// end, and paging further back stops once the pane holds this many. The log
/// itself stays complete in the repository.
pub(crate) const COGNITION_MAX_RETAINED: usize = 2_000;
const CARD_MARGIN: f32 = 8.0;
const CARD_GAP: f32 = 6.0;
const HEADER_GAP: f32 = 4.0;
const HARD_WRAP_LIMIT: usize = 96;

#[derive(Debug)]
pub struct CognitionState {
    /// Newest-first and contiguous from the newest entry the pane knows about.
    entries: Vec<PersistedCognitionEntryView>,
    /// Position of every rendered entry, so anchoring never scans the list.
    index_by_id: HashMap<i64, usize>,
    /// Bumped whenever the rendered entries change, to invalidate pane layouts.
    revision: u64,
    /// Stand-in shown until the first page lands, derived from the blackboard.
    snapshot_entries: Vec<PersistedCognitionEntryView>,
    snapshot_fingerprint: Option<u64>,
    max_retained: usize,
    loaded_initial: bool,
    has_older: bool,
    loading: bool,
    requested_initial: bool,
    refresh_needed: bool,
    /// Monotonically increasing snapshot generation used to avoid losing an
    /// update that arrives while a persisted page is in flight.
    snapshot_revision: u64,
    requested_snapshot_revision: Option<u64>,
    jump_to_newest: bool,
    panes: HashMap<&'static str, PaneLayout>,
}

impl Default for CognitionState {
    fn default() -> Self {
        Self::with_retention(COGNITION_MAX_RETAINED)
    }
}

/// Cached vertical layout for one pane showing the log.
///
/// Two panes can show the same state at different widths, so the ledger is
/// per-pane and is rebuilt only when the entries or the width actually change.
#[derive(Debug, Default)]
struct PaneLayout {
    ready: bool,
    revision: u64,
    width_bits: u32,
    /// Cumulative tops: `offsets[i]` is the top of entry `i`, and the last
    /// element is the total height.
    offsets: Vec<f32>,
    height_by_id: HashMap<i64, f32>,
}

impl CognitionState {
    fn with_retention(max_retained: usize) -> Self {
        Self {
            entries: Vec::new(),
            index_by_id: HashMap::new(),
            revision: 0,
            snapshot_entries: Vec::new(),
            snapshot_fingerprint: None,
            max_retained: max_retained.max(1),
            loaded_initial: false,
            has_older: false,
            loading: false,
            requested_initial: false,
            refresh_needed: false,
            snapshot_revision: 0,
            requested_snapshot_revision: None,
            jump_to_newest: true,
            panes: HashMap::new(),
        }
    }

    pub fn observe_snapshot(&mut self, logs: &[CognitionLogView]) {
        let fingerprint = snapshot_fingerprint(logs);
        if self.snapshot_fingerprint == Some(fingerprint) {
            return;
        }
        self.snapshot_revision = self.snapshot_revision.wrapping_add(1);
        if self.snapshot_fingerprint.is_some()
            || (self.loaded_initial && snapshot_has_unloaded_entry(logs, &self.entries))
        {
            self.refresh_needed = true;
        }
        self.snapshot_fingerprint = Some(fingerprint);
        if self.loaded_initial {
            // Persisted pages drive the pane from here on; the snapshot only
            // says when new entries exist to pull.
            return;
        }

        let mut snapshot_entries = logs
            .iter()
            .flat_map(|log| {
                log.entries.iter().map(|entry| PersistedCognitionEntryView {
                    id: synthetic_id(&log.source, entry),
                    source: log.source.clone(),
                    at: entry.at,
                    origin: entry.origin.clone(),
                    text: entry.text.clone(),
                })
            })
            .collect::<Vec<_>>();
        snapshot_entries
            .sort_by(|left, right| right.at.cmp(&left.at).then_with(|| right.id.cmp(&left.id)));
        self.snapshot_entries = snapshot_entries;
        self.reindex();
    }

    pub fn apply_page(
        &mut self,
        cursor: CognitionLogCursor,
        entries: Vec<PersistedCognitionEntryView>,
        has_more: bool,
    ) {
        self.loading = false;
        let requested_snapshot_revision = self.requested_snapshot_revision.take();
        match cursor {
            CognitionLogCursor::Newest => {
                self.loaded_initial = true;
                self.refresh_needed = requested_snapshot_revision
                    .is_some_and(|revision| revision != self.snapshot_revision);
                self.jump_to_newest = true;
                self.entries = entries;
                self.has_older = has_more;
            }
            CognitionLogCursor::Older { .. } => {
                let held = &self.index_by_id;
                self.entries.extend(
                    entries
                        .into_iter()
                        .filter(|entry| !held.contains_key(&entry.id)),
                );
                self.has_older = has_more;
            }
            CognitionLogCursor::Newer { .. } => {
                self.refresh_needed = requested_snapshot_revision
                    .is_some_and(|revision| revision != self.snapshot_revision);
                self.jump_to_newest = true;
                if entries.is_empty() {
                    return;
                }
                if has_more {
                    // More arrived than one page holds, so splicing this page
                    // onto what we have would leave a hole. Start over from
                    // the newest instead; the rest is still one page back.
                    self.entries = entries;
                    self.has_older = true;
                } else {
                    let mut merged = entries;
                    merged.retain(|entry| !self.index_by_id.contains_key(&entry.id));
                    merged.append(&mut self.entries);
                    self.entries = merged;
                }
            }
        }
        if self.entries.len() > self.max_retained {
            self.entries.truncate(self.max_retained);
            // The dropped tail is still in the repository, so paging back is
            // possible once the pane has room again.
            self.has_older = true;
        }
        self.reindex();
    }

    /// Entries the pane draws: persisted pages once loaded, else the snapshot.
    fn visible(&self) -> &[PersistedCognitionEntryView] {
        if self.loaded_initial {
            &self.entries
        } else {
            &self.snapshot_entries
        }
    }

    fn can_load_older(&self) -> bool {
        self.has_older && self.entries.len() < self.max_retained
    }

    fn reindex(&mut self) {
        self.revision = self.revision.wrapping_add(1);
        let mut index_by_id = std::mem::take(&mut self.index_by_id);
        index_by_id.clear();
        let source = if self.loaded_initial {
            &self.entries
        } else {
            &self.snapshot_entries
        };
        index_by_id.extend(
            source
                .iter()
                .enumerate()
                .map(|(position, entry)| (entry.id, position)),
        );
        self.index_by_id = index_by_id;
    }
}

fn snapshot_fingerprint(logs: &[CognitionLogView]) -> u64 {
    let mut hasher = DefaultHasher::new();
    for log in logs {
        log.source.hash(&mut hasher);
        for entry in &log.entries {
            entry.at.hash(&mut hasher);
            entry.origin.hash(&mut hasher);
            entry.text.hash(&mut hasher);
        }
    }
    hasher.finish()
}

fn snapshot_has_unloaded_entry(
    logs: &[CognitionLogView],
    loaded: &[PersistedCognitionEntryView],
) -> bool {
    logs.iter().any(|log| {
        log.entries.iter().any(|snapshot| {
            !loaded.iter().any(|entry| {
                entry.source == log.source
                    && entry.at == snapshot.at
                    && entry.origin == snapshot.origin
                    && entry.text == snapshot.text
            })
        })
    })
}

/// Stand-in identity for a snapshot entry, which carries no persisted id.
fn synthetic_id(source: &str, entry: &crate::CognitionEntryView) -> i64 {
    let mut hasher = DefaultHasher::new();
    source.hash(&mut hasher);
    entry.at.hash(&mut hasher);
    entry.origin.hash(&mut hasher);
    entry.text.hash(&mut hasher);
    i64::from_ne_bytes(hasher.finish().to_ne_bytes())
}

#[derive(Clone, Copy, Default)]
struct ViewportAnchor {
    first_id: Option<i64>,
    offset: f32,
}

pub fn ui(
    ui: &mut egui::Ui,
    id_salt: &'static str,
    tab_id: &VisualizerTabId,
    state: &mut CognitionState,
    follow: &mut bool,
    messages: &mut Vec<VisualizerClientMessage>,
) {
    let id = ui.make_persistent_id(id_salt);
    let available_width = ui.available_width().max(120.0);
    let width_bits = available_width.to_bits();
    let mut pane = state.panes.remove(id_salt).unwrap_or_default();
    let entries = state.visible();
    if !pane.ready || pane.revision != state.revision || pane.width_bits != width_bits {
        if pane.width_bits != width_bits {
            pane.height_by_id.clear();
        }
        rebuild_layout(ui, entries, available_width, &mut pane);
        pane.ready = true;
        pane.revision = state.revision;
        pane.width_bits = width_bits;
    }
    let offsets = &pane.offsets;
    let total_height = offsets.last().copied().unwrap_or_default();

    let anchor_key = id.with("cognition-viewport-anchor");
    let old_anchor = ui.ctx().data(|data| {
        data.get_temp::<ViewportAnchor>(anchor_key)
            .unwrap_or_default()
    });
    let corrected_offset = if *follow || state.jump_to_newest {
        Some(0.0)
    } else {
        corrected_scroll_offset(old_anchor, &state.index_by_id, offsets)
    };

    let mut scroll = egui::ScrollArea::vertical()
        .id_salt(id)
        .auto_shrink([false, false]);
    if let Some(offset) = corrected_offset {
        scroll = scroll.vertical_scroll_offset(offset);
    }
    let output = scroll.show_viewport(ui, |ui, viewport| {
        // Virtualized children are placed with explicit rectangles and do not
        // advance the content cursor horizontally. Pin the virtual content to
        // the viewport width so the scroll bar stays at the pane's right edge.
        ui.set_min_width(available_width);
        ui.set_width(available_width);
        ui.set_height(total_height);
        let start = offsets
            .partition_point(|offset| *offset <= viewport.min.y)
            .saturating_sub(1)
            .min(entries.len());
        let end = offsets
            .partition_point(|offset| *offset < viewport.max.y)
            .min(entries.len());
        for index in start..end {
            let height = offsets[index + 1] - offsets[index];
            let rect = egui::Rect::from_min_size(
                egui::pos2(ui.max_rect().left(), ui.max_rect().top() + offsets[index]),
                egui::vec2(available_width, height - CARD_GAP),
            );
            ui.scope_builder(egui::UiBuilder::new().max_rect(rect), |ui| {
                render_entry(ui, &entries[index]);
            });
        }
        viewport.max.y + viewport.height() >= total_height
    });
    let reached_end = output.inner;
    if should_disable_follow(*follow, output.state.offset.y) {
        *follow = false;
    }

    ui.ctx().data_mut(|data| {
        data.insert_temp(
            anchor_key,
            ViewportAnchor {
                first_id: entries.first().map(|entry| entry.id),
                offset: output.state.offset.y,
            },
        );
    });
    state.jump_to_newest = false;
    state.panes.insert(id_salt, pane);

    // Decide whether to refresh only after processing scroll input. If the
    // user scrolls away on the same frame as a snapshot update, Follow turns
    // off first and the update remains pending behind the floating button.
    if let Some(cursor) = automatic_page_cursor(state, *follow) {
        request_page(tab_id, state, messages, cursor);
    }

    if should_show_load_newest(*follow, state) {
        let button_size = egui::vec2(32.0, 24.0);
        let button_pos = egui::pos2(
            output.inner_rect.center().x - button_size.x / 2.0,
            output.inner_rect.top() + 8.0,
        );
        let button_rect = egui::Rect::from_min_size(button_pos, button_size);
        let clicked = ui
            .put(button_rect, egui::Button::new("↑").min_size(button_size))
            .on_hover_text(ui.ctx().tr("cognition-load-newest-hover"))
            .clicked();
        if clicked {
            request_page(tab_id, state, messages, latest_page_cursor(state));
        }
    }

    if reached_end
        && !state.loading
        && state.can_load_older()
        && let Some(oldest) = state.entries.last()
    {
        let cursor = CognitionLogCursor::Older {
            before_id: oldest.id,
        };
        request_page(tab_id, state, messages, cursor);
    }
}

pub fn follow_toggle(ui: &mut egui::Ui, state: &mut CognitionState, follow: &mut bool) {
    if ui
        .checkbox(follow, ui.ctx().tr("cognition-follow"))
        .on_hover_text(ui.ctx().tr("cognition-follow-hover"))
        .changed()
        && *follow
    {
        state.jump_to_newest = true;
    }
    if state.loading && state.requested_snapshot_revision.is_some() {
        ui.small(ui.ctx().tr("cognition-loading-newest"));
    }
}

fn automatic_page_cursor(state: &CognitionState, follow: bool) -> Option<CognitionLogCursor> {
    if state.loading {
        return None;
    }
    if !state.requested_initial {
        return Some(CognitionLogCursor::Newest);
    }
    (follow && state.refresh_needed).then(|| latest_page_cursor(state))
}

fn should_disable_follow(follow: bool, scroll_offset: f32) -> bool {
    follow && scroll_offset > 0.5
}

fn should_show_load_newest(follow: bool, state: &CognitionState) -> bool {
    !follow && state.refresh_needed && !state.loading
}

fn latest_page_cursor(state: &CognitionState) -> CognitionLogCursor {
    match state.entries.first() {
        Some(newest) => CognitionLogCursor::Newer {
            after_id: newest.id,
        },
        None => CognitionLogCursor::Newest,
    }
}

fn rebuild_layout(
    ui: &egui::Ui,
    entries: &[PersistedCognitionEntryView],
    available_width: f32,
    pane: &mut PaneLayout,
) {
    // Rebuilt into a fresh map so heights for evicted entries don't accumulate,
    // while entries that survived keep their measurement.
    let mut height_by_id = HashMap::with_capacity(entries.len());
    let offsets = &mut pane.offsets;
    offsets.clear();
    offsets.reserve(entries.len() + 1);
    offsets.push(0.0);
    let mut running = 0.0;
    for entry in entries {
        let height = pane
            .height_by_id
            .get(&entry.id)
            .copied()
            .unwrap_or_else(|| entry_height(ui, entry, available_width));
        height_by_id.insert(entry.id, height);
        running += height;
        offsets.push(running);
    }
    pane.height_by_id = height_by_id;
}

fn corrected_scroll_offset(
    old_anchor: ViewportAnchor,
    index_by_id: &HashMap<i64, usize>,
    offsets: &[f32],
) -> Option<f32> {
    let inserted_height = old_anchor
        .first_id
        .and_then(|first_id| index_by_id.get(&first_id))
        .and_then(|position| offsets.get(*position))
        .copied()
        .unwrap_or_default();
    (inserted_height > 0.0).then_some(old_anchor.offset + inserted_height)
}

fn request_page(
    tab_id: &VisualizerTabId,
    state: &mut CognitionState,
    messages: &mut Vec<VisualizerClientMessage>,
    cursor: CognitionLogCursor,
) {
    state.loading = true;
    state.requested_initial = true;
    if matches!(
        cursor,
        CognitionLogCursor::Newest | CognitionLogCursor::Newer { .. }
    ) {
        state.requested_snapshot_revision = Some(state.snapshot_revision);
        state.jump_to_newest = true;
    } else {
        state.requested_snapshot_revision = None;
    }
    messages.push(VisualizerClientMessage::Command {
        command: VisualizerCommand::LoadCognitionLogEntries {
            tab_id: tab_id.clone(),
            cursor,
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

    use crate::i18n::{I18nCatalog, Locale};

    use super::*;

    #[test]
    fn older_pages_stitch_onto_the_newest_page_in_newest_first_order() {
        let mut state = CognitionState::default();
        state.apply_page(CognitionLogCursor::Newest, vec![entry(5), entry(4)], true);
        state.apply_page(
            CognitionLogCursor::Older { before_id: 4 },
            vec![entry(4), entry(3)],
            false,
        );

        assert_eq!(loaded_ids(&state), vec![5, 4, 3]);
        assert!(!state.can_load_older());
    }

    #[test]
    fn a_refresh_prepends_newer_entries_and_evicts_only_the_oldest() {
        let mut state = CognitionState::with_retention(3);
        state.apply_page(
            CognitionLogCursor::Newest,
            vec![entry(3), entry(2), entry(1)],
            false,
        );
        state.apply_page(
            CognitionLogCursor::Newer { after_id: 3 },
            vec![entry(5), entry(4)],
            false,
        );

        assert_eq!(loaded_ids(&state), vec![5, 4, 3]);
        // The evicted tail is still in the repository, so paging back reopens.
        assert!(state.has_older);
        // ...but not while the pane is already at its retention ceiling.
        assert!(!state.can_load_older());
    }

    #[test]
    fn a_refresh_larger_than_one_page_restarts_from_the_newest() {
        let mut state = CognitionState::default();
        state.apply_page(CognitionLogCursor::Newest, vec![entry(2), entry(1)], false);
        state.apply_page(
            CognitionLogCursor::Newer { after_id: 2 },
            vec![entry(9), entry(8)],
            true,
        );

        // Splicing 9,8 onto 2,1 would claim 7..3 never existed.
        assert_eq!(loaded_ids(&state), vec![9, 8]);
        assert!(state.can_load_older());
    }

    #[test]
    fn an_empty_refresh_leaves_the_layout_untouched() {
        let mut state = CognitionState::default();
        state.apply_page(CognitionLogCursor::Newest, vec![entry(2), entry(1)], false);
        let revision = state.revision;

        state.apply_page(CognitionLogCursor::Newer { after_id: 2 }, Vec::new(), false);

        assert_eq!(state.revision, revision);
        assert_eq!(loaded_ids(&state), vec![2, 1]);
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
        let mut state = CognitionState::default();
        state.apply_page(
            CognitionLogCursor::Newest,
            vec![entry(5), entry(4), entry(3)],
            false,
        );

        let offset = corrected_scroll_offset(
            ViewportAnchor {
                first_id: Some(3),
                offset: 0.0,
            },
            &state.index_by_id,
            &[0.0, 40.0, 90.0, 130.0],
        );

        assert_eq!(offset, Some(90.0));
    }

    #[test]
    fn initial_page_is_automatic_even_when_follow_is_off() {
        let state = CognitionState::default();

        assert_eq!(
            automatic_page_cursor(&state, false),
            Some(CognitionLogCursor::Newest)
        );
    }

    #[test]
    fn snapshot_update_waits_for_user_when_follow_is_off() {
        let mut state = CognitionState::default();
        state.apply_page(CognitionLogCursor::Newest, vec![entry(1)], false);
        state.requested_initial = true;
        state.observe_snapshot(&snapshot(1));
        state.observe_snapshot(&snapshot(2));

        assert!(state.refresh_needed);
        assert_eq!(automatic_page_cursor(&state, false), None);
        assert_eq!(
            automatic_page_cursor(&state, true),
            Some(CognitionLogCursor::Newer { after_id: 1 })
        );
    }

    #[test]
    fn first_snapshot_after_initial_page_detects_an_unloaded_entry() {
        let mut state = CognitionState::default();
        state.apply_page(CognitionLogCursor::Newest, vec![entry(1)], false);

        state.observe_snapshot(&snapshot(2));

        assert!(state.refresh_needed);
        assert!(should_show_load_newest(false, &state));
    }

    #[test]
    fn first_snapshot_matching_initial_page_does_not_report_an_update() {
        let mut state = CognitionState::default();
        state.apply_page(CognitionLogCursor::Newest, vec![entry(1)], false);

        state.observe_snapshot(&snapshot_with_text(1, "entry 1"));

        assert!(!state.refresh_needed);
    }

    #[test]
    fn snapshot_update_during_refresh_remains_pending() {
        let mut state = CognitionState::default();
        state.observe_snapshot(&snapshot(1));
        let mut messages = Vec::new();
        request_page(
            &VisualizerTabId::new("tab"),
            &mut state,
            &mut messages,
            CognitionLogCursor::Newest,
        );
        state.observe_snapshot(&snapshot(2));

        state.apply_page(CognitionLogCursor::Newest, vec![entry(2), entry(1)], false);

        assert!(state.refresh_needed);
        assert_eq!(
            automatic_page_cursor(&state, true),
            Some(CognitionLogCursor::Newer { after_id: 2 })
        );
    }

    #[test]
    fn completed_refresh_requests_a_jump_to_the_newest_entry() {
        let mut state = CognitionState::default();
        state.apply_page(CognitionLogCursor::Newest, vec![entry(1)], false);
        state.jump_to_newest = false;

        state.apply_page(
            CognitionLogCursor::Newer { after_id: 1 },
            vec![entry(2)],
            false,
        );

        assert!(state.jump_to_newest);
        assert_eq!(loaded_ids(&state), vec![2, 1]);
    }

    #[test]
    fn scrolling_away_from_the_newest_entry_disables_follow() {
        assert!(!should_disable_follow(true, 0.0));
        assert!(!should_disable_follow(true, 0.5));
        assert!(should_disable_follow(true, 1.0));
        assert!(!should_disable_follow(false, 100.0));
    }

    #[test]
    fn pending_update_shows_the_load_newest_button_only_outside_follow() {
        let mut state = CognitionState {
            refresh_needed: true,
            ..Default::default()
        };

        assert!(!should_show_load_newest(true, &state));
        assert!(should_show_load_newest(false, &state));

        state.loading = true;
        assert!(!should_show_load_newest(false, &state));
    }

    fn loaded_ids(state: &CognitionState) -> Vec<i64> {
        state.entries.iter().map(|entry| entry.id).collect()
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

    fn snapshot(id: i64) -> Vec<CognitionLogView> {
        snapshot_with_text(id, &format!("snapshot {id}"))
    }

    fn snapshot_with_text(id: i64, text: &str) -> Vec<CognitionLogView> {
        vec![CognitionLogView {
            source: "cognition-gate".to_owned(),
            entries: vec![crate::CognitionEntryView {
                at: Utc.timestamp_opt(id, 0).unwrap(),
                origin: "sensory".to_owned(),
                text: text.to_owned(),
            }],
        }]
    }
}
