use std::collections::BTreeMap;
use std::sync::mpsc::Sender;

use nuillu_module::{AmbientSensoryEntry, SensoryInput};

use crate::{
    AmbientSensorySnapshotRowView, DerivedAmbientSensoryRowView, OneShotSensoryInputRowView,
    ScenePersonRowView, SceneRowKind, SceneRowView, SceneStateView, UtteranceDeltaView,
    UtteranceEventKindView, UtteranceEventRowView, UtteranceView, VisualizerClientMessage,
    VisualizerCommand, VisualizerTabId, i18n::EguiI18nExt as _, text::wrapped_label,
};

const FIELD_HEIGHT: f32 = 24.0;
const SHORT_FIELD_WIDTH: f32 = 96.0;
const MEDIUM_FIELD_WIDTH: f32 = 150.0;
const LONG_FIELD_WIDTH: f32 = 260.0;
const PREVIEW_MODALITY_WIDTH: f32 = 96.0;
const PREVIEW_CONTENT_WIDTH: f32 = 560.0;

const ATMOSPHERE_ASPECTS: [&str; 6] = [
    "light",
    "smell",
    "temperature",
    "air/weather",
    "surface/feel",
    "other",
];

fn atmosphere_aspect_label(ctx: &egui::Context, aspect: &str) -> String {
    match aspect {
        "light" => ctx.tr("scene-atmosphere-aspect-light"),
        "smell" => ctx.tr("scene-atmosphere-aspect-smell"),
        "temperature" => ctx.tr("scene-atmosphere-aspect-temperature"),
        "air/weather" => ctx.tr("scene-atmosphere-aspect-air-weather"),
        "surface/feel" => ctx.tr("scene-atmosphere-aspect-surface-feel"),
        "other" => ctx.tr("scene-atmosphere-aspect-other"),
        other => other.to_string(),
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum ActivityRole {
    User,
    Environment,
    Assistant,
}

impl ActivityRole {
    fn tr_key(&self) -> &'static str {
        match self {
            Self::User => "scene-activity-role-sensory",
            Self::Environment => "scene-activity-role-ambient",
            Self::Assistant => "scene-activity-role-nui",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ActivityMessage {
    id: Option<String>,
    role: ActivityRole,
    content: String,
    source: Option<String>,
    streaming: bool,
}

impl ActivityMessage {
    fn new(role: ActivityRole, content: impl Into<String>) -> Self {
        Self {
            id: None,
            role,
            content: content.into(),
            source: None,
            streaming: false,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct UtteranceKey {
    sender: String,
    target: String,
    generation_id: u64,
}

#[derive(Debug, Default)]
pub struct SceneUiState {
    scene: SceneStateView,
    activity: Vec<ActivityMessage>,
    activity_authoritative: bool,
    streaming_utterances: BTreeMap<UtteranceKey, usize>,
    completed_utterances: BTreeMap<UtteranceKey, usize>,
    one_shot_sensory_rows: BTreeMap<i64, OneShotSensoryInputRowView>,
    ambient_sensory_rows: BTreeMap<i64, AmbientSensorySnapshotRowView>,
    utterance_event_rows: BTreeMap<i64, UtteranceEventRowView>,
    selected_person_message_row_id: Option<String>,
    person_message_draft: String,
}

impl SceneUiState {
    pub fn scene_view(&self) -> &SceneStateView {
        &self.scene
    }

    pub fn set_scene_state(&mut self, scene: SceneStateView) {
        let next_selection = selected_person_message_row_id(
            self.selected_person_message_row_id.as_deref(),
            &scene.people,
        );
        self.set_person_message_selection(next_selection);
        self.scene = scene;
    }

    pub fn push_sensory_input(&mut self, input: SensoryInput) {
        if self.activity_authoritative {
            return;
        }
        let SensoryInput::OneShot {
            modality,
            direction,
            content,
            observed_at,
        } = input
        else {
            return;
        };
        let source = if let Some(direction) = direction {
            format!(
                "one-shot {} from {} at {}",
                modality.as_str(),
                direction,
                observed_at
            )
        } else {
            format!("one-shot {} at {}", modality.as_str(), observed_at)
        };
        let mut message = ActivityMessage::new(ActivityRole::User, content);
        message.source = Some(source);
        self.activity.push(message);
    }

    pub fn push_utterance_delta(&mut self, delta: UtteranceDeltaView) {
        if self.activity_authoritative {
            return;
        }
        let key = UtteranceKey {
            sender: delta.sender.clone(),
            target: delta.target.clone(),
            generation_id: delta.generation_id,
        };
        if self.completed_utterances.contains_key(&key) {
            return;
        }
        let index = if let Some(index) = self.streaming_utterances.get(&key).copied() {
            index
        } else {
            let index = self.activity.len();
            self.streaming_utterances.insert(key, index);
            self.activity.push(ActivityMessage {
                id: None,
                role: ActivityRole::Assistant,
                content: String::new(),
                streaming: true,
                source: Some(format!("{} -> {}", delta.sender, delta.target)),
            });
            index
        };
        if let Some(message) = self.activity.get_mut(index) {
            message.content.push_str(&delta.delta);
            message.streaming = true;
        }
    }

    pub fn push_utterance_completed(&mut self, utterance: UtteranceView) {
        if self.activity_authoritative {
            return;
        }
        let matching_key = if let Some(generation_id) = utterance.generation_id {
            let key = UtteranceKey {
                sender: utterance.sender.clone(),
                target: utterance.target.clone(),
                generation_id,
            };
            Some(key)
        } else {
            self.streaming_utterances
                .iter()
                .filter(|(key, _)| key.sender == utterance.sender && key.target == utterance.target)
                .max_by_key(|(_, index)| **index)
                .map(|(key, _)| key.clone())
        };
        if let Some(key) = matching_key {
            if let Some(index) = self.streaming_utterances.remove(&key) {
                if let Some(message) = self.activity.get_mut(index) {
                    message.content = utterance.text.clone();
                    message.streaming = false;
                }
                self.completed_utterances.insert(key, index);
                return;
            }
            if let Some(index) = self.completed_utterances.get(&key).copied() {
                if let Some(message) = self.activity.get_mut(index) {
                    message.content = utterance.text.clone();
                    message.streaming = false;
                }
                return;
            }
        }

        let completed_key = utterance.generation_id.map(|generation_id| UtteranceKey {
            sender: utterance.sender.clone(),
            target: utterance.target.clone(),
            generation_id,
        });
        let mut message = ActivityMessage::new(ActivityRole::Assistant, utterance.text);
        message.source = Some(format!(
            "{} -> {} at {}",
            utterance.sender, utterance.target, utterance.emitted_at
        ));
        let index = self.activity.len();
        self.activity.push(message);
        if let Some(key) = completed_key {
            self.completed_utterances.insert(key, index);
        }
    }

    pub fn apply_one_shot_sensory_input_rows(&mut self, rows: Vec<OneShotSensoryInputRowView>) {
        self.one_shot_sensory_rows = rows.into_iter().map(|row| (row.id, row)).collect();
        self.rebuild_activity_from_raw_rows();
    }

    pub fn append_one_shot_sensory_input_row(&mut self, row: OneShotSensoryInputRowView) {
        self.one_shot_sensory_rows.insert(row.id, row);
        self.rebuild_activity_from_raw_rows();
    }

    pub fn apply_ambient_sensory_snapshot_rows(
        &mut self,
        rows: Vec<AmbientSensorySnapshotRowView>,
    ) {
        self.ambient_sensory_rows = rows.into_iter().map(|row| (row.id, row)).collect();
        self.rebuild_activity_from_raw_rows();
    }

    pub fn append_ambient_sensory_snapshot_row(&mut self, row: AmbientSensorySnapshotRowView) {
        self.ambient_sensory_rows.insert(row.id, row);
        self.rebuild_activity_from_raw_rows();
    }

    pub fn apply_utterance_event_rows(&mut self, rows: Vec<UtteranceEventRowView>) {
        self.utterance_event_rows = rows.into_iter().map(|row| (row.id, row)).collect();
        self.rebuild_activity_from_raw_rows();
    }

    pub fn append_utterance_event_row(&mut self, row: UtteranceEventRowView) {
        self.utterance_event_rows.insert(row.id, row);
        self.rebuild_activity_from_raw_rows();
    }

    fn rebuild_activity_from_raw_rows(&mut self) {
        self.activity_authoritative = true;
        self.activity.clear();
        self.streaming_utterances.clear();
        self.completed_utterances.clear();

        let mut utterance_messages = BTreeMap::<UtteranceKey, Vec<usize>>::new();
        let mut ambient_baseline: Option<BTreeMap<String, AmbientSensoryEntry>> = None;
        let mut rows = Vec::new();
        rows.extend(
            self.one_shot_sensory_rows
                .values()
                .map(ActivitySourceRow::OneShot),
        );
        rows.extend(
            self.ambient_sensory_rows
                .values()
                .map(ActivitySourceRow::Ambient),
        );
        rows.extend(
            self.utterance_event_rows
                .values()
                .map(ActivitySourceRow::Utterance),
        );
        rows.sort_by_key(ActivitySourceRow::sort_key);

        for row in rows {
            match row {
                ActivitySourceRow::OneShot(row) => {
                    interrupt_streaming_messages(
                        &mut self.activity,
                        &mut self.streaming_utterances,
                    );
                    let mut message = ActivityMessage::new(ActivityRole::User, row.content.clone());
                    message.id = Some(format!("one-shot:{}", row.id));
                    message.source = Some(one_shot_source(row));
                    self.activity.push(message);
                }
                ActivitySourceRow::Ambient(row) => {
                    let current = ambient_entry_map(&row.entries);
                    if let Some(previous) = &ambient_baseline {
                        if let Some(content) = ambient_diff_content(previous, &current) {
                            interrupt_streaming_messages(
                                &mut self.activity,
                                &mut self.streaming_utterances,
                            );
                            let mut message =
                                ActivityMessage::new(ActivityRole::Environment, content);
                            message.id = Some(format!("ambient:{}", row.id));
                            message.source =
                                Some(format!("ambient snapshot at {}", row.observed_at));
                            self.activity.push(message);
                        }
                    }
                    ambient_baseline = Some(current);
                }
                ActivitySourceRow::Utterance(row) => apply_utterance_event_row(
                    &mut self.activity,
                    &mut self.streaming_utterances,
                    &mut self.completed_utterances,
                    &mut utterance_messages,
                    row,
                ),
            }
        }
    }

    fn send_person_message_commands(
        &mut self,
        tab_id: &VisualizerTabId,
    ) -> Vec<VisualizerClientMessage> {
        let Some(row_id) = self.selected_person_message_row_id.clone() else {
            return Vec::new();
        };
        let message = self.person_message_draft.trim().to_string();
        if message.is_empty() {
            return Vec::new();
        }
        let Some(row) = self
            .scene
            .people
            .iter()
            .find(|row| row.id == row_id)
            .cloned()
        else {
            return Vec::new();
        };
        self.person_message_draft.clear();
        vec![
            VisualizerClientMessage::Command {
                command: VisualizerCommand::UpdateSceneRow {
                    tab_id: tab_id.clone(),
                    row: SceneRowView::Person(row),
                },
            },
            VisualizerClientMessage::Command {
                command: VisualizerCommand::SendScenePersonMessage {
                    tab_id: tab_id.clone(),
                    row_id,
                    message,
                },
            },
        ]
    }

    fn set_person_message_selection(&mut self, next_selection: Option<String>) {
        if self.selected_person_message_row_id != next_selection {
            self.person_message_draft.clear();
        }
        self.selected_person_message_row_id = next_selection;
    }
}

fn selected_person_message_row_id(
    current: Option<&str>,
    people: &[ScenePersonRowView],
) -> Option<String> {
    current
        .filter(|row_id| people.iter().any(|row| row.id == *row_id))
        .map(str::to_string)
        .or_else(|| people.first().map(|row| row.id.clone()))
}

enum ActivitySourceRow<'a> {
    OneShot(&'a OneShotSensoryInputRowView),
    Ambient(&'a AmbientSensorySnapshotRowView),
    Utterance(&'a UtteranceEventRowView),
}

impl ActivitySourceRow<'_> {
    fn sort_key(
        &self,
    ) -> (
        chrono::DateTime<chrono::Utc>,
        chrono::DateTime<chrono::Utc>,
        u8,
        i64,
    ) {
        match self {
            Self::OneShot(row) => (row.observed_at, row.created_at, 0, row.id),
            Self::Ambient(row) => (row.observed_at, row.created_at, 1, row.id),
            Self::Utterance(row) => (row.occurred_at, row.created_at, 2, row.id),
        }
    }
}

fn interrupt_streaming_messages(
    activity: &mut [ActivityMessage],
    streaming: &mut BTreeMap<UtteranceKey, usize>,
) {
    for (_, index) in std::mem::take(streaming) {
        if let Some(message) = activity.get_mut(index) {
            message.streaming = false;
        }
    }
}

fn apply_utterance_event_row(
    activity: &mut Vec<ActivityMessage>,
    streaming: &mut BTreeMap<UtteranceKey, usize>,
    completed: &mut BTreeMap<UtteranceKey, usize>,
    utterance_messages: &mut BTreeMap<UtteranceKey, Vec<usize>>,
    row: &UtteranceEventRowView,
) {
    let key = UtteranceKey {
        sender: row.sender.clone(),
        target: row.target.clone(),
        generation_id: row.generation_id,
    };
    match row.event_kind {
        UtteranceEventKindView::Delta => {
            if completed.contains_key(&key) {
                return;
            }
            let index = if let Some(index) = streaming.get(&key).copied() {
                index
            } else {
                let index = activity.len();
                streaming.insert(key.clone(), index);
                utterance_messages
                    .entry(key.clone())
                    .or_default()
                    .push(index);
                activity.push(ActivityMessage {
                    id: Some(format!("utterance:{}:{}", row.generation_id, row.sequence)),
                    role: ActivityRole::Assistant,
                    content: String::new(),
                    streaming: true,
                    source: Some(format!("{} -> {}", row.sender, row.target)),
                });
                index
            };
            if let Some(message) = activity.get_mut(index) {
                message.content.push_str(&row.content);
                message.streaming = true;
            }
        }
        UtteranceEventKindView::Completed => {
            streaming.remove(&key);
            if let Some(indexes) = utterance_messages.get(&key) {
                if indexes.len() == 1
                    && let Some(message) = activity.get_mut(indexes[0])
                {
                    message.content = row.content.clone();
                }
                for index in indexes {
                    if let Some(message) = activity.get_mut(*index) {
                        message.streaming = false;
                    }
                }
                if let Some(index) = indexes.last().copied() {
                    completed.insert(key, index);
                }
                return;
            }
            let mut message = ActivityMessage::new(ActivityRole::Assistant, row.content.clone());
            message.id = Some(format!("utterance:{}", row.id));
            message.source = Some(format!(
                "{} -> {} at {}",
                row.sender, row.target, row.occurred_at
            ));
            let index = activity.len();
            activity.push(message);
            completed.insert(key, index);
        }
        UtteranceEventKindView::Aborted => {
            streaming.remove(&key);
            if let Some(indexes) = utterance_messages.get(&key) {
                for index in indexes {
                    if let Some(message) = activity.get_mut(*index) {
                        message.streaming = false;
                    }
                }
                return;
            }
            if !row.content.trim().is_empty() {
                let mut message =
                    ActivityMessage::new(ActivityRole::Assistant, row.content.clone());
                message.id = Some(format!("utterance:{}", row.id));
                message.source = Some(format!("{} -> {} interrupted", row.sender, row.target));
                activity.push(message);
            }
        }
    }
}

fn one_shot_source(row: &OneShotSensoryInputRowView) -> String {
    if let Some(direction) = &row.direction {
        format!(
            "one-shot {} from {} at {}",
            row.modality, direction, row.observed_at
        )
    } else {
        format!("one-shot {} at {}", row.modality, row.observed_at)
    }
}

fn ambient_entry_map(entries: &[AmbientSensoryEntry]) -> BTreeMap<String, AmbientSensoryEntry> {
    entries
        .iter()
        .cloned()
        .map(|entry| (entry.id.clone(), entry))
        .collect()
}

fn ambient_diff_content(
    previous: &BTreeMap<String, AmbientSensoryEntry>,
    current: &BTreeMap<String, AmbientSensoryEntry>,
) -> Option<String> {
    let mut parts = Vec::new();
    for (id, entry) in current {
        match previous.get(id) {
            None => parts.push(format!("added {}", ambient_entry_summary(entry))),
            Some(previous) if previous != entry => {
                parts.push(format!("updated {}", ambient_entry_summary(entry)));
            }
            Some(_) => {}
        }
    }
    for (id, entry) in previous {
        if !current.contains_key(id) {
            parts.push(format!("removed {}", ambient_entry_summary(entry)));
        }
    }
    (!parts.is_empty()).then(|| parts.join("\n"))
}

fn ambient_entry_summary(entry: &AmbientSensoryEntry) -> String {
    format!(
        "{} {}: {}",
        entry.modality.as_str(),
        entry.id,
        entry.content
    )
}

pub fn ui(
    ui: &mut egui::Ui,
    tab_id: &VisualizerTabId,
    state: &mut SceneUiState,
    commands: &Sender<VisualizerClientMessage>,
) {
    let available = ui.available_size();
    let width = available.x.max(1.0);
    let height = available.y.max(1.0);
    let composer_height = SCENE_COMPOSER_HEIGHT;
    let content_height = (height - composer_height - SCENE_ROW_GAP * 2.0).max(1.0);
    let config_height = scene_config_height(height, content_height);
    let activity_height = (content_height - config_height).max(1.0);

    ui.allocate_ui_with_layout(
        egui::vec2(width, config_height),
        egui::Layout::top_down(egui::Align::Min),
        |ui| {
            egui::ScrollArea::vertical()
                .id_salt("scene-config-scroll")
                .show(ui, |ui| scene_config_ui(ui, tab_id, state, commands));
        },
    );
    ui.add_space(SCENE_ROW_GAP);
    ui.allocate_ui_with_layout(
        egui::vec2(width, activity_height),
        egui::Layout::top_down(egui::Align::Min),
        |ui| activity_ui(ui, &state.activity),
    );
    ui.add_space(SCENE_ROW_GAP);
    ui.allocate_ui_with_layout(
        egui::vec2(width, composer_height),
        egui::Layout::left_to_right(egui::Align::Center),
        |ui| person_message_composer_ui(ui, tab_id, state, commands),
    );
}

const SCENE_CONFIG_MIN_HEIGHT: f32 = 120.0;
const SCENE_CONFIG_MAX_HEIGHT: f32 = 320.0;
const SCENE_COMPOSER_HEIGHT: f32 = 34.0;
const SCENE_ROW_GAP: f32 = 6.0;
const SCENE_COMPOSER_SPEAKER_WIDTH: f32 = 150.0;
const SCENE_COMPOSER_SEND_WIDTH: f32 = 64.0;
const SCENE_COMPOSER_MESSAGE_MIN_WIDTH: f32 = 96.0;

fn scene_config_height(available_height: f32, content_height: f32) -> f32 {
    let max_config = (content_height * 0.62).max(1.0);
    (available_height * 0.38)
        .clamp(SCENE_CONFIG_MIN_HEIGHT, SCENE_CONFIG_MAX_HEIGHT)
        .min(max_config)
}

fn scene_config_ui(
    ui: &mut egui::Ui,
    tab_id: &VisualizerTabId,
    state: &mut SceneUiState,
    commands: &Sender<VisualizerClientMessage>,
) {
    people_section_ui(ui, tab_id, state, commands);
    ui.separator();
    objects_section_ui(ui, tab_id, state, commands);
    ui.separator();
    sounds_section_ui(ui, tab_id, state, commands);
    ui.separator();
    atmosphere_section_ui(ui, tab_id, state, commands);
    ui.separator();
    derived_ambient_ui(ui, &state.scene.derived_ambient);
}

fn people_section_ui(
    ui: &mut egui::Ui,
    tab_id: &VisualizerTabId,
    state: &mut SceneUiState,
    commands: &Sender<VisualizerClientMessage>,
) {
    let section_title = ui.ctx().tr("scene-people");
    section_header(ui, &section_title, tab_id, commands, SceneRowKind::Person);
    let remove_label = ui.ctx().tr("scene-remove");
    horizontal_grid(ui, "scene-people-grid", 5, |ui| {
        ui.strong(ui.ctx().tr("scene-remove"));
        ui.strong(ui.ctx().tr("scene-name"));
        ui.strong(ui.ctx().tr("scene-direction"));
        ui.strong(ui.ctx().tr("scene-distance"));
        ui.strong(ui.ctx().tr("scene-state"));
        ui.end_row();

        let mut index = 0;
        while index < state.scene.people.len() {
            let row_id = state.scene.people[index].id.clone();
            let mut send_update = false;
            let remove_clicked = ui
                .add_sized(
                    [64.0, FIELD_HEIGHT],
                    egui::Button::new(remove_label.as_str()),
                )
                .clicked();
            {
                let row = &mut state.scene.people[index];
                send_update |= text_field_with_id(
                    ui,
                    &mut row.name,
                    SHORT_FIELD_WIDTH,
                    ("person-name", tab_id.as_str(), row_id.as_str()),
                );
                send_update |= text_field_with_id(
                    ui,
                    &mut row.direction,
                    SHORT_FIELD_WIDTH,
                    ("person-direction", tab_id.as_str(), row_id.as_str()),
                );
                send_update |= text_field_with_id(
                    ui,
                    &mut row.distance,
                    SHORT_FIELD_WIDTH,
                    ("person-distance", tab_id.as_str(), row_id.as_str()),
                );
                send_update |= text_field_with_id(
                    ui,
                    &mut row.state,
                    LONG_FIELD_WIDTH,
                    ("person-state", tab_id.as_str(), row_id.as_str()),
                );
            }

            if remove_clicked {
                let _ = commands.send(VisualizerClientMessage::Command {
                    command: VisualizerCommand::RemoveSceneRow {
                        tab_id: tab_id.clone(),
                        kind: SceneRowKind::Person,
                        row_id: row_id.clone(),
                    },
                });
                state.scene.people.remove(index);
                let next_selection = selected_person_message_row_id(
                    state.selected_person_message_row_id.as_deref(),
                    &state.scene.people,
                );
                state.set_person_message_selection(next_selection);
                ui.end_row();
                continue;
            }
            if send_update {
                let _ = commands.send(VisualizerClientMessage::Command {
                    command: VisualizerCommand::UpdateSceneRow {
                        tab_id: tab_id.clone(),
                        row: SceneRowView::Person(state.scene.people[index].clone()),
                    },
                });
            }
            ui.end_row();
            index += 1;
        }
    });
}

fn objects_section_ui(
    ui: &mut egui::Ui,
    tab_id: &VisualizerTabId,
    state: &mut SceneUiState,
    commands: &Sender<VisualizerClientMessage>,
) {
    let section_title = ui.ctx().tr("scene-objects");
    section_header(ui, &section_title, tab_id, commands, SceneRowKind::Object);
    let remove_label = ui.ctx().tr("scene-remove");
    horizontal_grid(ui, "scene-objects-grid", 6, |ui| {
        ui.strong(ui.ctx().tr("scene-remove"));
        ui.strong(ui.ctx().tr("scene-name"));
        ui.strong(ui.ctx().tr("scene-direction"));
        ui.strong(ui.ctx().tr("scene-distance"));
        ui.strong(ui.ctx().tr("scene-visual-description"));
        ui.strong(ui.ctx().tr("scene-sound-description"));
        ui.end_row();

        let mut index = 0;
        while index < state.scene.objects.len() {
            let row_id = state.scene.objects[index].id.clone();
            let remove_clicked = ui
                .add_sized(
                    [64.0, FIELD_HEIGHT],
                    egui::Button::new(remove_label.as_str()),
                )
                .clicked();
            let mut send_update = false;
            {
                let row = &mut state.scene.objects[index];
                send_update |= text_field(ui, &mut row.name, MEDIUM_FIELD_WIDTH);
                send_update |= text_field(ui, &mut row.direction, SHORT_FIELD_WIDTH);
                send_update |= text_field(ui, &mut row.distance, SHORT_FIELD_WIDTH);
                send_update |= text_field(ui, &mut row.visual_description, LONG_FIELD_WIDTH);
                send_update |= text_field(ui, &mut row.sound_description, LONG_FIELD_WIDTH);
            }
            if remove_clicked {
                let _ = commands.send(VisualizerClientMessage::Command {
                    command: VisualizerCommand::RemoveSceneRow {
                        tab_id: tab_id.clone(),
                        kind: SceneRowKind::Object,
                        row_id,
                    },
                });
                state.scene.objects.remove(index);
                ui.end_row();
                continue;
            }
            if send_update {
                let _ = commands.send(VisualizerClientMessage::Command {
                    command: VisualizerCommand::UpdateSceneRow {
                        tab_id: tab_id.clone(),
                        row: SceneRowView::Object(state.scene.objects[index].clone()),
                    },
                });
            }
            ui.end_row();
            index += 1;
        }
    });
}

fn sounds_section_ui(
    ui: &mut egui::Ui,
    tab_id: &VisualizerTabId,
    state: &mut SceneUiState,
    commands: &Sender<VisualizerClientMessage>,
) {
    let section_title = ui.ctx().tr("scene-sounds");
    section_header(ui, &section_title, tab_id, commands, SceneRowKind::Sound);
    let remove_label = ui.ctx().tr("scene-remove");
    horizontal_grid(ui, "scene-sounds-grid", 4, |ui| {
        ui.strong(ui.ctx().tr("scene-remove"));
        ui.strong(ui.ctx().tr("scene-direction"));
        ui.strong(ui.ctx().tr("scene-distance"));
        ui.strong(ui.ctx().tr("scene-description"));
        ui.end_row();

        let mut index = 0;
        while index < state.scene.sounds.len() {
            let row_id = state.scene.sounds[index].id.clone();
            let remove_clicked = ui
                .add_sized(
                    [64.0, FIELD_HEIGHT],
                    egui::Button::new(remove_label.as_str()),
                )
                .clicked();
            let mut send_update = false;
            {
                let row = &mut state.scene.sounds[index];
                send_update |= text_field(ui, &mut row.direction, SHORT_FIELD_WIDTH);
                send_update |= text_field(ui, &mut row.distance, SHORT_FIELD_WIDTH);
                send_update |= text_field(ui, &mut row.description, LONG_FIELD_WIDTH);
            }
            if remove_clicked {
                let _ = commands.send(VisualizerClientMessage::Command {
                    command: VisualizerCommand::RemoveSceneRow {
                        tab_id: tab_id.clone(),
                        kind: SceneRowKind::Sound,
                        row_id,
                    },
                });
                state.scene.sounds.remove(index);
                ui.end_row();
                continue;
            }
            if send_update {
                let _ = commands.send(VisualizerClientMessage::Command {
                    command: VisualizerCommand::UpdateSceneRow {
                        tab_id: tab_id.clone(),
                        row: SceneRowView::Sound(state.scene.sounds[index].clone()),
                    },
                });
            }
            ui.end_row();
            index += 1;
        }
    });
}

fn atmosphere_section_ui(
    ui: &mut egui::Ui,
    tab_id: &VisualizerTabId,
    state: &mut SceneUiState,
    commands: &Sender<VisualizerClientMessage>,
) {
    let section_title = ui.ctx().tr("scene-atmosphere");
    section_header(
        ui,
        &section_title,
        tab_id,
        commands,
        SceneRowKind::Atmosphere,
    );
    let remove_label = ui.ctx().tr("scene-remove");
    horizontal_grid(ui, "scene-atmosphere-grid", 3, |ui| {
        ui.strong(ui.ctx().tr("scene-remove"));
        ui.strong(ui.ctx().tr("scene-aspect"));
        ui.strong(ui.ctx().tr("scene-description"));
        ui.end_row();

        let mut index = 0;
        while index < state.scene.atmosphere.len() {
            let row_id = state.scene.atmosphere[index].id.clone();
            let remove_clicked = ui
                .add_sized(
                    [64.0, FIELD_HEIGHT],
                    egui::Button::new(remove_label.as_str()),
                )
                .clicked();
            let mut send_update = false;
            {
                let row = &mut state.scene.atmosphere[index];
                let before = row.aspect.clone();
                egui::ComboBox::from_id_salt(format!("atmosphere-aspect-{row_id}"))
                    .selected_text(if row.aspect.is_empty() {
                        ui.ctx().tr("scene-atmosphere-aspect-other")
                    } else {
                        atmosphere_aspect_label(ui.ctx(), row.aspect.as_str())
                    })
                    .show_ui(ui, |ui| {
                        for aspect in ATMOSPHERE_ASPECTS {
                            ui.selectable_value(
                                &mut row.aspect,
                                aspect.to_string(),
                                atmosphere_aspect_label(ui.ctx(), aspect),
                            );
                        }
                    });
                send_update |= row.aspect != before;
                send_update |= text_field(ui, &mut row.description, LONG_FIELD_WIDTH);
            }
            if remove_clicked {
                let _ = commands.send(VisualizerClientMessage::Command {
                    command: VisualizerCommand::RemoveSceneRow {
                        tab_id: tab_id.clone(),
                        kind: SceneRowKind::Atmosphere,
                        row_id,
                    },
                });
                state.scene.atmosphere.remove(index);
                ui.end_row();
                continue;
            }
            if send_update {
                let _ = commands.send(VisualizerClientMessage::Command {
                    command: VisualizerCommand::UpdateSceneRow {
                        tab_id: tab_id.clone(),
                        row: SceneRowView::Atmosphere(state.scene.atmosphere[index].clone()),
                    },
                });
            }
            ui.end_row();
            index += 1;
        }
    });
}

fn derived_ambient_ui(ui: &mut egui::Ui, rows: &[DerivedAmbientSensoryRowView]) {
    ui.strong(ui.ctx().tr("scene-derived-sensory-preview"));
    horizontal_grid(ui, "scene-derived-ambient-grid", 3, |ui| {
        ui.strong(ui.ctx().tr("scene-id"));
        ui.strong(ui.ctx().tr("scene-modality"));
        ui.strong(ui.ctx().tr("scene-description"));
        ui.end_row();

        if rows.is_empty() {
            ui.label("-");
            ui.label("-");
            ui.label(ui.ctx().tr("scene-no-derived-sensory-input"));
            ui.end_row();
            return;
        }
        for row in rows {
            ui.add_sized(
                [MEDIUM_FIELD_WIDTH, FIELD_HEIGHT],
                egui::Label::new(row.id.as_str()),
            );
            ui.add_sized(
                [PREVIEW_MODALITY_WIDTH, FIELD_HEIGHT],
                egui::Label::new(row.modality.as_str()),
            );
            ui.add_sized(
                [PREVIEW_CONTENT_WIDTH, FIELD_HEIGHT],
                egui::Label::new(row.content.as_str()),
            );
            ui.end_row();
        }
    });
}

fn activity_ui(ui: &mut egui::Ui, activity: &[ActivityMessage]) {
    ui.strong(ui.ctx().tr("scene-recent-activity"));
    egui::ScrollArea::vertical()
        .id_salt("scene-activity")
        .stick_to_bottom(true)
        .show(ui, |ui| {
            if activity.is_empty() {
                ui.label(ui.ctx().tr("scene-no-recent-speech"));
                return;
            }
            for message in activity {
                activity_message_ui(ui, message);
                ui.add_space(6.0);
            }
        });
}

fn person_message_composer_ui(
    ui: &mut egui::Ui,
    tab_id: &VisualizerTabId,
    state: &mut SceneUiState,
    commands: &Sender<VisualizerClientMessage>,
) {
    let has_people = !state.scene.people.is_empty();
    let mut send_requested = false;
    ui.horizontal(|ui| {
        let message_width =
            composer_message_input_width(ui.available_width(), ui.spacing().item_spacing.x);
        ui.add_enabled_ui(has_people, |ui| {
            let mut next_selection = state.selected_person_message_row_id.clone();
            egui::ComboBox::from_id_salt(("scene-person-message-speaker", tab_id.as_str()))
                .width(SCENE_COMPOSER_SPEAKER_WIDTH)
                .selected_text(selected_person_message_label(ui, state))
                .show_ui(ui, |ui| {
                    for person in &state.scene.people {
                        ui.selectable_value(
                            &mut next_selection,
                            Some(person.id.clone()),
                            person_display_name(person),
                        );
                    }
                });
            state.set_person_message_selection(next_selection);
        });
        let message_response = ui.add_enabled(
            has_people,
            egui::TextEdit::singleline(&mut state.person_message_draft)
                .desired_width(message_width)
                .hint_text(ui.ctx().tr("scene-message-hint"))
                .id_salt(("scene-person-message-draft", tab_id.as_str())),
        );
        let send_label = ui.ctx().tr("scene-send");
        send_requested |= ui
            .add_enabled_ui(has_people, |ui| {
                ui.add_sized(
                    [SCENE_COMPOSER_SEND_WIDTH, FIELD_HEIGHT],
                    egui::Button::new(send_label.as_str()),
                )
            })
            .inner
            .clicked();
        send_requested |= has_people
            && message_response.lost_focus()
            && ui.input(|input| input.key_pressed(egui::Key::Enter));
    });

    if send_requested {
        for command in state.send_person_message_commands(tab_id) {
            let _ = commands.send(command);
        }
    }
}

fn composer_message_input_width(row_width: f32, item_spacing: f32) -> f32 {
    (row_width - SCENE_COMPOSER_SPEAKER_WIDTH - SCENE_COMPOSER_SEND_WIDTH - item_spacing * 2.0)
        .max(SCENE_COMPOSER_MESSAGE_MIN_WIDTH)
}

fn selected_person_message_label(ui: &egui::Ui, state: &SceneUiState) -> String {
    state
        .selected_person_message_row_id
        .as_deref()
        .and_then(|row_id| state.scene.people.iter().find(|row| row.id == row_id))
        .map(person_display_name)
        .unwrap_or_else(|| ui.ctx().tr("scene-no-people"))
}

fn person_display_name(person: &ScenePersonRowView) -> String {
    let name = person.name.trim();
    if name.is_empty() {
        person.id.clone()
    } else {
        name.to_string()
    }
}

fn activity_message_ui(ui: &mut egui::Ui, message: &ActivityMessage) {
    egui::Frame::new()
        .fill(match message.role {
            ActivityRole::User => ui.visuals().selection.bg_fill.linear_multiply(0.65),
            ActivityRole::Environment => ui.visuals().faint_bg_color,
            ActivityRole::Assistant => ui.visuals().extreme_bg_color,
        })
        .stroke(ui.visuals().widgets.noninteractive.bg_stroke)
        .corner_radius(egui::CornerRadius::same(6))
        .inner_margin(egui::Margin::same(8))
        .show(ui, |ui| {
            ui.horizontal_wrapped(|ui| {
                ui.strong(ui.ctx().tr(message.role.tr_key()));
                if message.streaming {
                    ui.label(ui.ctx().tr("scene-streaming"));
                }
                if let Some(source) = &message.source {
                    wrapped_label(ui, source);
                }
            });
            ui.add_space(3.0);
            wrapped_label(ui, &message.content);
        });
}

fn section_header(
    ui: &mut egui::Ui,
    label: &str,
    tab_id: &VisualizerTabId,
    commands: &Sender<VisualizerClientMessage>,
    kind: SceneRowKind,
) {
    ui.horizontal(|ui| {
        ui.strong(label);
        if ui.button(ui.ctx().tr("scene-add")).clicked() {
            let _ = commands.send(VisualizerClientMessage::Command {
                command: VisualizerCommand::CreateSceneRow {
                    tab_id: tab_id.clone(),
                    kind,
                },
            });
        }
    });
}

fn horizontal_grid(
    ui: &mut egui::Ui,
    id: &'static str,
    columns: usize,
    add_contents: impl FnOnce(&mut egui::Ui),
) {
    egui::ScrollArea::horizontal()
        .id_salt(format!("{id}-scroll"))
        .show(ui, |ui| {
            egui::Grid::new(id)
                .striped(true)
                .num_columns(columns)
                .show(ui, add_contents);
        });
}

fn text_field(ui: &mut egui::Ui, value: &mut String, width: f32) -> bool {
    ui.add_sized(
        [width, FIELD_HEIGHT],
        egui::TextEdit::singleline(value).desired_width(width),
    )
    .changed()
}

fn text_field_with_id(
    ui: &mut egui::Ui,
    value: &mut String,
    width: f32,
    id_salt: impl std::hash::Hash,
) -> bool {
    ui.add_sized(
        [width, FIELD_HEIGHT],
        egui::TextEdit::singleline(value)
            .desired_width(width)
            .id_salt(id_salt),
    )
    .changed()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ScenePersonRowView;
    use chrono::{TimeZone, Utc};

    #[test]
    fn person_message_send_flushes_target_row_before_message() {
        let tab_id = VisualizerTabId::new("live");
        let mut state = scene_with_two_people();
        state.selected_person_message_row_id = Some("person-2".to_string());
        state.person_message_draft = " hello two ".to_string();

        let commands = state.send_person_message_commands(&tab_id);

        assert_eq!(commands.len(), 2);
        let update_row = match &commands[0] {
            VisualizerClientMessage::Command {
                command:
                    VisualizerCommand::UpdateSceneRow {
                        tab_id: command_tab,
                        row: SceneRowView::Person(row),
                    },
            } => {
                assert_eq!(command_tab, &tab_id);
                row
            }
            other => panic!("expected target row update, got {other:?}"),
        };
        assert_eq!(update_row.id, "person-2");
        assert_eq!(update_row.name, "Koro");

        match &commands[1] {
            VisualizerClientMessage::Command {
                command:
                    VisualizerCommand::SendScenePersonMessage {
                        tab_id: command_tab,
                        row_id,
                        message,
                    },
            } => {
                assert_eq!(command_tab, &tab_id);
                assert_eq!(row_id, "person-2");
                assert_eq!(message, "hello two");
            }
            other => panic!("expected target person message, got {other:?}"),
        }
        assert_eq!(
            state.selected_person_message_row_id.as_deref(),
            Some("person-2")
        );
        assert_eq!(state.person_message_draft, "");
    }

    #[test]
    fn empty_person_message_does_not_create_commands_or_clear_draft() {
        let tab_id = VisualizerTabId::new("live");
        let mut state = scene_with_two_people();
        state.selected_person_message_row_id = Some("person-2".to_string());
        state.person_message_draft = "   ".to_string();

        let commands = state.send_person_message_commands(&tab_id);

        assert!(commands.is_empty());
        assert_eq!(state.person_message_draft, "   ");
    }

    #[test]
    fn composer_message_input_width_uses_remaining_row_width() {
        let spacing = 8.0;
        let narrow = composer_message_input_width(240.0, spacing);
        let wide = composer_message_input_width(620.0, spacing);

        assert_eq!(narrow, SCENE_COMPOSER_MESSAGE_MIN_WIDTH);
        assert!((wide - 390.0).abs() < f32::EPSILON);
    }

    #[test]
    fn person_message_selection_survives_scene_refresh_when_person_remains() {
        let mut state = scene_with_two_people();
        state.selected_person_message_row_id = Some("person-2".to_string());
        state.person_message_draft = "pending".to_string();

        state.set_scene_state(SceneStateView {
            people: vec![
                ScenePersonRowView {
                    id: "person-1".to_string(),
                    name: "Pibi renamed".to_string(),
                    direction: "front".to_string(),
                    distance: "2m".to_string(),
                    state: "watching Nui".to_string(),
                },
                ScenePersonRowView {
                    id: "person-2".to_string(),
                    name: "Koro renamed".to_string(),
                    direction: "left".to_string(),
                    distance: "1m".to_string(),
                    state: "waiting".to_string(),
                },
            ],
            ..SceneStateView::default()
        });

        assert_eq!(
            state.selected_person_message_row_id.as_deref(),
            Some("person-2")
        );
        assert_eq!(state.person_message_draft, "pending");
    }

    #[test]
    fn person_message_selection_falls_back_and_clears_draft_when_person_disappears() {
        let mut state = scene_with_two_people();
        state.selected_person_message_row_id = Some("person-2".to_string());
        state.person_message_draft = "pending".to_string();

        state.set_scene_state(SceneStateView {
            people: vec![ScenePersonRowView {
                id: "person-1".to_string(),
                name: "Pibi".to_string(),
                direction: "front".to_string(),
                distance: "2m".to_string(),
                state: "watching Nui".to_string(),
            }],
            ..SceneStateView::default()
        });

        assert_eq!(
            state.selected_person_message_row_id.as_deref(),
            Some("person-1")
        );
        assert_eq!(state.person_message_draft, "");

        state.person_message_draft = "new pending".to_string();
        state.set_scene_state(SceneStateView::default());

        assert_eq!(state.selected_person_message_row_id, None);
        assert_eq!(state.person_message_draft, "");
    }

    #[test]
    fn completed_utterance_updates_matching_generation_not_old_stream() {
        let mut state = SceneUiState::default();

        state.push_utterance_delta(utterance_delta(1, 0, "old partial"));
        state.push_utterance_delta(utterance_delta(2, 0, "latest"));
        state.push_utterance_completed(utterance_completed(Some(2), "latest"));

        assert_eq!(state.activity.len(), 2);
        assert_eq!(state.activity[0].content, "old partial");
        assert!(state.activity[0].streaming);
        assert_eq!(state.activity[1].content, "latest");
        assert!(!state.activity[1].streaming);
    }

    #[test]
    fn late_delta_for_completed_utterance_is_ignored() {
        let mut state = SceneUiState::default();

        state.push_utterance_completed(utterance_completed(Some(7), "finished"));
        state.push_utterance_delta(utterance_delta(7, 0, "finished"));

        assert_eq!(state.activity.len(), 1);
        assert_eq!(state.activity[0].content, "finished");
        assert!(!state.activity[0].streaming);
    }

    #[test]
    fn duplicate_completed_utterance_updates_existing_row() {
        let mut state = SceneUiState::default();

        state.push_utterance_completed(utterance_completed(Some(9), "first"));
        state.push_utterance_completed(utterance_completed(Some(9), "final"));

        assert_eq!(state.activity.len(), 1);
        assert_eq!(state.activity[0].content, "final");
        assert!(!state.activity[0].streaming);
    }

    #[test]
    fn raw_rows_restore_one_shot_ambient_diff_and_split_streaming_speech() {
        let mut state = SceneUiState::default();

        state.apply_ambient_sensory_snapshot_rows(vec![ambient_row(
            1,
            0,
            vec![ambient_entry(
                "scene:person:koro",
                "vision",
                "Koro waits nearby.",
            )],
        )]);
        state.apply_utterance_event_rows(vec![
            utterance_event(1, 1, UtteranceEventKindView::Delta, 0, "Koro, "),
            utterance_event(2, 3, UtteranceEventKindView::Delta, 1, "stay"),
            utterance_event(3, 5, UtteranceEventKindView::Delta, 2, " close."),
            utterance_event(
                4,
                6,
                UtteranceEventKindView::Completed,
                3,
                "Koro, stay close.",
            ),
        ]);
        state.append_ambient_sensory_snapshot_row(ambient_row(
            2,
            2,
            vec![ambient_entry(
                "scene:person:koro",
                "vision",
                "Koro steps toward Nui.",
            )],
        ));
        state.append_one_shot_sensory_input_row(one_shot_row(1, 4, "Koro says, \"wait\""));

        assert_eq!(state.activity.len(), 5);
        assert_eq!(state.activity[0].content, "Koro, ");
        assert!(!state.activity[0].streaming);
        assert_eq!(state.activity[1].role, ActivityRole::Environment);
        assert!(state.activity[1].content.contains("updated vision"));
        assert_eq!(state.activity[2].content, "stay");
        assert!(!state.activity[2].streaming);
        assert_eq!(state.activity[3].role, ActivityRole::User);
        assert_eq!(state.activity[3].content, "Koro says, \"wait\"");
        assert_eq!(state.activity[4].content, " close.");
        assert!(!state.activity[4].streaming);
    }

    fn scene_with_two_people() -> SceneUiState {
        let mut state = SceneUiState::default();
        state.set_scene_state(SceneStateView {
            people: vec![
                ScenePersonRowView {
                    id: "person-1".to_string(),
                    name: "Pibi".to_string(),
                    direction: "front".to_string(),
                    distance: "2m".to_string(),
                    state: "watching Nui".to_string(),
                },
                ScenePersonRowView {
                    id: "person-2".to_string(),
                    name: "Koro".to_string(),
                    direction: "left".to_string(),
                    distance: "1m".to_string(),
                    state: "waiting".to_string(),
                },
            ],
            ..SceneStateView::default()
        });
        state
    }

    fn utterance_delta(generation_id: u64, sequence: u32, delta: &str) -> UtteranceDeltaView {
        UtteranceDeltaView {
            sender: "speak".to_string(),
            target: "Alice".to_string(),
            generation_id,
            sequence,
            delta: delta.to_string(),
        }
    }

    fn utterance_completed(generation_id: Option<u64>, text: &str) -> UtteranceView {
        UtteranceView {
            sender: "speak".to_string(),
            target: "Alice".to_string(),
            generation_id,
            text: text.to_string(),
            emitted_at: Utc.with_ymd_and_hms(2026, 6, 13, 6, 18, 51).unwrap(),
        }
    }

    fn at(second: u32) -> chrono::DateTime<Utc> {
        Utc.with_ymd_and_hms(2026, 6, 13, 6, 18, second).unwrap()
    }

    fn one_shot_row(id: i64, second: u32, content: &str) -> OneShotSensoryInputRowView {
        OneShotSensoryInputRowView {
            id,
            server_session_id: "server-session".to_string(),
            modality: "audition".to_string(),
            direction: Some("Koro".to_string()),
            content: content.to_string(),
            observed_at: at(second),
            created_at: at(second),
        }
    }

    fn ambient_row(
        id: i64,
        second: u32,
        entries: Vec<AmbientSensoryEntry>,
    ) -> AmbientSensorySnapshotRowView {
        AmbientSensorySnapshotRowView {
            id,
            server_session_id: "server-session".to_string(),
            entries,
            observed_at: at(second),
            created_at: at(second),
        }
    }

    fn ambient_entry(id: &str, modality: &str, content: &str) -> AmbientSensoryEntry {
        AmbientSensoryEntry {
            id: id.to_string(),
            modality: nuillu_module::SensoryModality::parse(modality),
            content: content.to_string(),
        }
    }

    fn utterance_event(
        id: i64,
        second: u32,
        event_kind: UtteranceEventKindView,
        sequence: u32,
        content: &str,
    ) -> UtteranceEventRowView {
        UtteranceEventRowView {
            id,
            server_session_id: "server-session".to_string(),
            event_kind,
            sender: "speak".to_string(),
            target: "Koro".to_string(),
            generation_id: 7,
            sequence,
            content: content.to_string(),
            reason: None,
            occurred_at: at(second),
            created_at: at(second),
        }
    }
}
