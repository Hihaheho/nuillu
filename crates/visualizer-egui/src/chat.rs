use std::collections::{BTreeMap, BTreeSet};
use std::sync::mpsc::Sender;

use nuillu_module::SensoryInput;

use crate::{
    DerivedAmbientSensoryRowView, SceneRowKind, SceneRowView, SceneStateView, UtteranceDeltaView,
    UtteranceView, VisualizerClientMessage, VisualizerCommand, VisualizerTabId,
    text::wrapped_label,
};

const FIELD_HEIGHT: f32 = 24.0;
const SHORT_FIELD_WIDTH: f32 = 96.0;
const MEDIUM_FIELD_WIDTH: f32 = 150.0;
const LONG_FIELD_WIDTH: f32 = 260.0;
const MESSAGE_FIELD_WIDTH: f32 = 260.0;
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

#[derive(Debug, Clone, PartialEq, Eq)]
enum ActivityRole {
    User,
    Assistant,
}

impl ActivityRole {
    fn label(&self) -> &'static str {
        match self {
            Self::User => "sensory",
            Self::Assistant => "nui",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ActivityMessage {
    role: ActivityRole,
    content: String,
    source: Option<String>,
    streaming: bool,
}

impl ActivityMessage {
    fn new(role: ActivityRole, content: impl Into<String>) -> Self {
        Self {
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
    streaming_utterances: BTreeMap<UtteranceKey, usize>,
    person_message_drafts: BTreeMap<String, String>,
}

impl SceneUiState {
    pub fn scene_view(&self) -> &SceneStateView {
        &self.scene
    }

    pub fn set_scene_state(&mut self, scene: SceneStateView) {
        let valid_people = scene
            .people
            .iter()
            .map(|row| row.id.clone())
            .collect::<BTreeSet<_>>();
        self.person_message_drafts
            .retain(|row_id, _| valid_people.contains(row_id));
        for row_id in valid_people {
            self.person_message_drafts.entry(row_id).or_default();
        }
        self.scene = scene;
    }

    pub fn push_sensory_input(&mut self, input: SensoryInput) {
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
        let key = UtteranceKey {
            sender: delta.sender.clone(),
            target: delta.target.clone(),
            generation_id: delta.generation_id,
        };
        let index = if let Some(index) = self.streaming_utterances.get(&key).copied() {
            index
        } else {
            let index = self.activity.len();
            self.streaming_utterances.insert(key, index);
            self.activity.push(ActivityMessage {
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
        let matching_key = self
            .streaming_utterances
            .keys()
            .find(|key| key.sender == utterance.sender && key.target == utterance.target)
            .cloned();
        if let Some(key) = matching_key
            && let Some(index) = self.streaming_utterances.remove(&key)
        {
            if let Some(message) = self.activity.get_mut(index) {
                message.content = utterance.text;
                message.streaming = false;
            }
            return;
        }

        let mut message = ActivityMessage::new(ActivityRole::Assistant, utterance.text);
        message.source = Some(format!(
            "{} -> {} at {}",
            utterance.sender, utterance.target, utterance.emitted_at
        ));
        self.activity.push(message);
    }

    fn send_person_message_commands(
        &mut self,
        tab_id: &VisualizerTabId,
        row_id: &str,
    ) -> Vec<VisualizerClientMessage> {
        let message = self
            .person_message_drafts
            .get(row_id)
            .map(|draft| draft.trim().to_string())
            .unwrap_or_default();
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
        if let Some(draft) = self.person_message_drafts.get_mut(row_id) {
            draft.clear();
        }
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
                    row_id: row_id.to_string(),
                    message,
                },
            },
        ]
    }
}

pub fn ui(
    ui: &mut egui::Ui,
    tab_id: &VisualizerTabId,
    state: &mut SceneUiState,
    commands: &Sender<VisualizerClientMessage>,
) {
    egui::ScrollArea::vertical()
        .id_salt("scene-window-scroll")
        .show(ui, |ui| {
            people_section_ui(ui, tab_id, state, commands);
            ui.separator();
            objects_section_ui(ui, tab_id, state, commands);
            ui.separator();
            sounds_section_ui(ui, tab_id, state, commands);
            ui.separator();
            atmosphere_section_ui(ui, tab_id, state, commands);
            ui.separator();
            derived_ambient_ui(ui, &state.scene.derived_ambient);
            ui.separator();
            activity_ui(ui, &state.activity);
        });
}

fn people_section_ui(
    ui: &mut egui::Ui,
    tab_id: &VisualizerTabId,
    state: &mut SceneUiState,
    commands: &Sender<VisualizerClientMessage>,
) {
    section_header(ui, "People", tab_id, commands, SceneRowKind::Person);
    horizontal_grid(ui, "scene-people-grid", 7, |ui| {
        ui.strong("Remove");
        ui.strong("Name");
        ui.strong("Direction");
        ui.strong("Distance");
        ui.strong("State");
        ui.strong("Message");
        ui.strong("Send");
        ui.end_row();

        let mut index = 0;
        while index < state.scene.people.len() {
            let row_id = state.scene.people[index].id.clone();
            let mut send_update = false;
            let remove_clicked = ui
                .add_sized([64.0, FIELD_HEIGHT], egui::Button::new("Remove"))
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
            let draft = state
                .person_message_drafts
                .entry(row_id.clone())
                .or_default();
            let message_response = ui.add_sized(
                [MESSAGE_FIELD_WIDTH, FIELD_HEIGHT],
                egui::TextEdit::singleline(draft)
                    .desired_width(MESSAGE_FIELD_WIDTH)
                    .id_salt(("person-message", tab_id.as_str(), row_id.as_str())),
            );
            let send_clicked = ui
                .add_sized([56.0, FIELD_HEIGHT], egui::Button::new("Send"))
                .clicked()
                || (message_response.lost_focus()
                    && ui.input(|input| input.key_pressed(egui::Key::Enter)));

            if remove_clicked {
                let _ = commands.send(VisualizerClientMessage::Command {
                    command: VisualizerCommand::RemoveSceneRow {
                        tab_id: tab_id.clone(),
                        kind: SceneRowKind::Person,
                        row_id: row_id.clone(),
                    },
                });
                state.scene.people.remove(index);
                state.person_message_drafts.remove(&row_id);
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
            if send_clicked {
                for command in state.send_person_message_commands(tab_id, &row_id) {
                    let _ = commands.send(command);
                }
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
    section_header(ui, "Objects", tab_id, commands, SceneRowKind::Object);
    horizontal_grid(ui, "scene-objects-grid", 6, |ui| {
        ui.strong("Remove");
        ui.strong("Name");
        ui.strong("Direction");
        ui.strong("Distance");
        ui.strong("Visual Description");
        ui.strong("Sound Description");
        ui.end_row();

        let mut index = 0;
        while index < state.scene.objects.len() {
            let row_id = state.scene.objects[index].id.clone();
            let remove_clicked = ui
                .add_sized([64.0, FIELD_HEIGHT], egui::Button::new("Remove"))
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
    section_header(ui, "Sounds", tab_id, commands, SceneRowKind::Sound);
    horizontal_grid(ui, "scene-sounds-grid", 4, |ui| {
        ui.strong("Remove");
        ui.strong("Direction");
        ui.strong("Distance");
        ui.strong("Description");
        ui.end_row();

        let mut index = 0;
        while index < state.scene.sounds.len() {
            let row_id = state.scene.sounds[index].id.clone();
            let remove_clicked = ui
                .add_sized([64.0, FIELD_HEIGHT], egui::Button::new("Remove"))
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
    section_header(ui, "Atmosphere", tab_id, commands, SceneRowKind::Atmosphere);
    horizontal_grid(ui, "scene-atmosphere-grid", 3, |ui| {
        ui.strong("Remove");
        ui.strong("Aspect");
        ui.strong("Description");
        ui.end_row();

        let mut index = 0;
        while index < state.scene.atmosphere.len() {
            let row_id = state.scene.atmosphere[index].id.clone();
            let remove_clicked = ui
                .add_sized([64.0, FIELD_HEIGHT], egui::Button::new("Remove"))
                .clicked();
            let mut send_update = false;
            {
                let row = &mut state.scene.atmosphere[index];
                let before = row.aspect.clone();
                egui::ComboBox::from_id_salt(format!("atmosphere-aspect-{row_id}"))
                    .selected_text(if row.aspect.is_empty() {
                        "other"
                    } else {
                        row.aspect.as_str()
                    })
                    .show_ui(ui, |ui| {
                        for aspect in ATMOSPHERE_ASPECTS {
                            ui.selectable_value(&mut row.aspect, aspect.to_string(), aspect);
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
    ui.strong("Derived sensory preview");
    horizontal_grid(ui, "scene-derived-ambient-grid", 3, |ui| {
        ui.strong("Id");
        ui.strong("Modality");
        ui.strong("Description");
        ui.end_row();

        if rows.is_empty() {
            ui.label("-");
            ui.label("-");
            ui.label("No active derived sensory input.");
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
    ui.strong("Recent activity");
    egui::ScrollArea::vertical()
        .id_salt("scene-activity")
        .stick_to_bottom(true)
        .show(ui, |ui| {
            if activity.is_empty() {
                ui.label("No recent speech.");
                return;
            }
            for message in activity {
                activity_message_ui(ui, message);
                ui.add_space(6.0);
            }
        });
}

fn activity_message_ui(ui: &mut egui::Ui, message: &ActivityMessage) {
    egui::Frame::new()
        .fill(match message.role {
            ActivityRole::User => ui.visuals().selection.bg_fill.linear_multiply(0.65),
            ActivityRole::Assistant => ui.visuals().extreme_bg_color,
        })
        .stroke(ui.visuals().widgets.noninteractive.bg_stroke)
        .corner_radius(egui::CornerRadius::same(6))
        .inner_margin(egui::Margin::same(8))
        .show(ui, |ui| {
            ui.horizontal_wrapped(|ui| {
                ui.strong(message.role.label());
                if message.streaming {
                    ui.label("streaming");
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
        if ui.button("Add").clicked() {
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

    #[test]
    fn person_message_send_flushes_target_row_before_message() {
        let tab_id = VisualizerTabId::new("live");
        let mut state = scene_with_two_people();
        state
            .person_message_drafts
            .insert("person-1".to_string(), "hello one".to_string());
        state
            .person_message_drafts
            .insert("person-2".to_string(), " hello two ".to_string());

        let commands = state.send_person_message_commands(&tab_id, "person-2");

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
            state
                .person_message_drafts
                .get("person-1")
                .map(String::as_str),
            Some("hello one")
        );
        assert_eq!(
            state
                .person_message_drafts
                .get("person-2")
                .map(String::as_str),
            Some("")
        );
    }

    #[test]
    fn empty_person_message_does_not_create_commands_or_clear_draft() {
        let tab_id = VisualizerTabId::new("live");
        let mut state = scene_with_two_people();
        state
            .person_message_drafts
            .insert("person-2".to_string(), "   ".to_string());

        let commands = state.send_person_message_commands(&tab_id, "person-2");

        assert!(commands.is_empty());
        assert_eq!(
            state
                .person_message_drafts
                .get("person-2")
                .map(String::as_str),
            Some("   ")
        );
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
}
