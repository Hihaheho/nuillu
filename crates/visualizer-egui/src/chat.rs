use std::collections::BTreeMap;
use std::sync::mpsc::Sender;

use nuillu_module::SensoryInput;

use crate::{
    AmbientSensoryRowView, ChatInput, UtteranceDeltaView, UtteranceView, VisualizerClientMessage,
    VisualizerCommand, VisualizerTabId, text::wrapped_label,
};

const FIELD_HEIGHT: f32 = 24.0;
const MODALITY_FIELD_WIDTH: f32 = 120.0;
const AMBIENT_INPUT_WIDTH: f32 = 360.0;
const ONE_SHOT_INPUT_MIN_WIDTH: f32 = 320.0;

#[derive(Debug, Clone, PartialEq, Eq)]
enum ChatRole {
    User,
    Assistant,
}

impl ChatRole {
    fn label(&self) -> &'static str {
        match self {
            Self::User => "user",
            Self::Assistant => "assistant",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ChatMessage {
    role: ChatRole,
    content: String,
    source: Option<String>,
    streaming: bool,
}

impl ChatMessage {
    fn new(role: ChatRole, content: impl Into<String>) -> Self {
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

#[derive(Debug)]
pub struct ChatState {
    messages: Vec<ChatMessage>,
    streaming_utterances: BTreeMap<UtteranceKey, usize>,
    ambient_rows: Vec<AmbientSensoryRowView>,
    one_shot_modality: String,
    one_shot_draft: String,
}

impl Default for ChatState {
    fn default() -> Self {
        Self {
            messages: Vec::new(),
            streaming_utterances: BTreeMap::new(),
            ambient_rows: Vec::new(),
            one_shot_modality: "audition".to_string(),
            one_shot_draft: String::new(),
        }
    }
}

impl ChatState {
    pub fn set_ambient_rows(&mut self, rows: Vec<AmbientSensoryRowView>) {
        self.ambient_rows = rows;
    }

    pub fn push_sensory_input(&mut self, input: SensoryInput) {
        let (source, content) = match input {
            SensoryInput::Observed {
                modality,
                content,
                observed_at,
                ..
            } => (format!("{} at {}", modality.as_str(), observed_at), content),
            SensoryInput::AmbientSnapshot {
                entries,
                observed_at,
            } => (
                format!("ambient snapshot at {observed_at}"),
                entries
                    .into_iter()
                    .map(|entry| format!("{}: {}", entry.modality.as_str(), entry.content))
                    .collect::<Vec<_>>()
                    .join("\n"),
            ),
        };
        let mut message = ChatMessage::new(ChatRole::User, content);
        message.source = Some(source);
        self.messages.push(message);
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
            let index = self.messages.len();
            self.streaming_utterances.insert(key, index);
            self.messages.push(ChatMessage {
                role: ChatRole::Assistant,
                content: String::new(),
                streaming: true,
                source: Some(format!("{} -> {}", delta.sender, delta.target)),
            });
            index
        };
        if let Some(message) = self.messages.get_mut(index) {
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
            if let Some(message) = self.messages.get_mut(index) {
                message.content = utterance.text;
                message.streaming = false;
            }
            return;
        }

        let mut message = ChatMessage::new(ChatRole::Assistant, utterance.text);
        message.source = Some(format!(
            "{} -> {} at {}",
            utterance.sender, utterance.target, utterance.emitted_at
        ));
        self.messages.push(message);
    }
}

pub fn ui(
    ui: &mut egui::Ui,
    tab_id: &VisualizerTabId,
    state: &mut ChatState,
    commands: &Sender<VisualizerClientMessage>,
) {
    chat_messages_ui(ui, &state.messages);

    ui.separator();
    ambient_table_ui(ui, tab_id, state, commands);
    ui.separator();
    one_shot_ui(ui, tab_id, state, commands);
}

fn ambient_table_ui(
    ui: &mut egui::Ui,
    tab_id: &VisualizerTabId,
    state: &mut ChatState,
    commands: &Sender<VisualizerClientMessage>,
) {
    ui.horizontal(|ui| {
        ui.strong("Ambient");
        if ui.button("Add row").clicked() {
            let _ = commands.send(VisualizerClientMessage::Command {
                command: VisualizerCommand::CreateAmbientSensoryRow {
                    tab_id: tab_id.clone(),
                    modality: "vision".to_string(),
                    content: String::new(),
                    disabled: false,
                },
            });
        }
    });
    egui::ScrollArea::horizontal()
        .id_salt("ambient-sensory-table-scroll")
        .show(ui, |ui| {
            egui::Grid::new("ambient-sensory-table")
                .striped(true)
                .num_columns(4)
                .show(ui, |ui| {
                    ui.strong("Enabled");
                    ui.strong("Category");
                    ui.strong("Input");
                    ui.end_row();

                    let mut index = 0;
                    while index < state.ambient_rows.len() {
                        let row_id = state.ambient_rows[index].id.clone();
                        let mut send_update = false;

                        let mut enabled = !state.ambient_rows[index].disabled;
                        if ui
                            .add_sized(
                                [56.0, FIELD_HEIGHT],
                                egui::Checkbox::without_text(&mut enabled),
                            )
                            .on_hover_text("Include this row in ambient snapshots")
                            .changed()
                        {
                            state.ambient_rows[index].disabled = !enabled;
                            send_update = true;
                        }

                        let category_response = ui.add_sized(
                            [MODALITY_FIELD_WIDTH, FIELD_HEIGHT],
                            egui::TextEdit::singleline(&mut state.ambient_rows[index].modality)
                                .desired_width(MODALITY_FIELD_WIDTH),
                        );
                        if category_response.lost_focus() {
                            send_update = true;
                        }

                        let input_response = ui.add_sized(
                            [AMBIENT_INPUT_WIDTH, FIELD_HEIGHT],
                            egui::TextEdit::singleline(&mut state.ambient_rows[index].content)
                                .desired_width(AMBIENT_INPUT_WIDTH),
                        );
                        if input_response.lost_focus() {
                            send_update = true;
                        }

                        if ui
                            .add_sized([64.0, FIELD_HEIGHT], egui::Button::new("Delete"))
                            .clicked()
                        {
                            let _ = commands.send(VisualizerClientMessage::Command {
                                command: VisualizerCommand::RemoveAmbientSensoryRow {
                                    tab_id: tab_id.clone(),
                                    row_id,
                                },
                            });
                            state.ambient_rows.remove(index);
                            ui.end_row();
                            continue;
                        } else if send_update {
                            let _ = commands.send(VisualizerClientMessage::Command {
                                command: VisualizerCommand::UpdateAmbientSensoryRow {
                                    tab_id: tab_id.clone(),
                                    row: state.ambient_rows[index].clone(),
                                },
                            });
                        }
                        ui.end_row();
                        index += 1;
                    }
                });
        });
}

fn one_shot_ui(
    ui: &mut egui::Ui,
    tab_id: &VisualizerTabId,
    state: &mut ChatState,
    commands: &Sender<VisualizerClientMessage>,
) {
    ui.vertical(|ui| {
        ui.strong("One shot");
        ui.horizontal(|ui| {
            ui.add_sized(
                [MODALITY_FIELD_WIDTH, FIELD_HEIGHT],
                egui::TextEdit::singleline(&mut state.one_shot_modality)
                    .desired_width(MODALITY_FIELD_WIDTH),
            );
            let input_width = (ui.available_width() - 72.0).max(ONE_SHOT_INPUT_MIN_WIDTH);
            let response = ui.add_sized(
                [input_width, FIELD_HEIGHT],
                egui::TextEdit::singleline(&mut state.one_shot_draft).desired_width(input_width),
            );
            let send = ui
                .add_sized([56.0, FIELD_HEIGHT], egui::Button::new("Send"))
                .clicked()
                || (response.lost_focus() && ui.input(|input| input.key_pressed(egui::Key::Enter)));
            if send {
                let content = state.one_shot_draft.trim().to_owned();
                if !content.is_empty() {
                    let modality = if state.one_shot_modality.trim().is_empty() {
                        "audition".to_string()
                    } else {
                        state.one_shot_modality.trim().to_owned()
                    };
                    let _ = commands.send(VisualizerClientMessage::Command {
                        command: VisualizerCommand::SendSensoryInput {
                            tab_id: tab_id.clone(),
                            input: ChatInput { modality, content },
                        },
                    });
                    state.one_shot_draft.clear();
                }
            }
        });
    });
}

fn chat_messages_ui(ui: &mut egui::Ui, messages: &[ChatMessage]) {
    egui::ScrollArea::vertical()
        .id_salt("chat-messages")
        .stick_to_bottom(true)
        .show(ui, |ui| {
            for message in messages {
                chat_message_ui(ui, message);
                ui.add_space(6.0);
            }
        });
}

fn chat_message_ui(ui: &mut egui::Ui, message: &ChatMessage) {
    let is_user = matches!(message.role, ChatRole::User);
    ui.horizontal(|ui| {
        let row_width = ui.available_width();
        let side_space = row_width * 0.08;
        let bubble_width = (row_width - side_space).clamp(160.0, 720.0);
        if is_user {
            ui.add_space(side_space);
        }
        egui::Frame::new()
            .fill(if is_user {
                ui.visuals().selection.bg_fill.linear_multiply(0.65)
            } else {
                ui.visuals().extreme_bg_color
            })
            .stroke(ui.visuals().widgets.noninteractive.bg_stroke)
            .corner_radius(egui::CornerRadius::same(6))
            .inner_margin(egui::Margin::same(8))
            .show(ui, |ui| {
                let inner_width = (bubble_width - 16.0).max(120.0);
                ui.set_min_width(inner_width);
                ui.set_max_width(inner_width);
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
        if !is_user {
            ui.add_space(side_space);
        }
    });
}
