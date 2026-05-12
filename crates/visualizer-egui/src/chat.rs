use std::collections::BTreeMap;
use std::sync::mpsc::Sender;

use nuillu_module::SensoryInput;

use crate::{
    ChatInput, ChatInputKind, UtteranceDeltaView, UtteranceView, VisualizerClientMessage,
    VisualizerCommand, VisualizerTabId, text::wrapped_label,
};

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

#[derive(Debug, Default)]
pub struct ChatState {
    messages: Vec<ChatMessage>,
    streaming_utterances: BTreeMap<UtteranceKey, usize>,
    draft: String,
    direction: String,
    seen: bool,
}

impl ChatState {
    pub fn push_sensory_input(&mut self, input: SensoryInput) {
        let (source, content) = match input {
            SensoryInput::Heard {
                direction,
                content,
                observed_at,
            } => (
                format!(
                    "heard {} at {}",
                    direction.as_deref().unwrap_or("unknown"),
                    observed_at
                ),
                content,
            ),
            SensoryInput::Seen {
                direction,
                appearance,
                observed_at,
            } => (
                format!(
                    "seen {} at {}",
                    direction.as_deref().unwrap_or("unknown"),
                    observed_at
                ),
                appearance,
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
    ui.horizontal(|ui| {
        ui.checkbox(&mut state.seen, "Seen");
        ui.label("Direction");
        ui.text_edit_singleline(&mut state.direction);
    });
    ui.horizontal(|ui| {
        let response = ui.text_edit_singleline(&mut state.draft);
        let send = ui.button("Send").clicked()
            || (response.lost_focus() && ui.input(|input| input.key_pressed(egui::Key::Enter)));
        if send {
            let content = state.draft.trim().to_owned();
            if !content.is_empty() {
                let direction =
                    (!state.direction.trim().is_empty()).then(|| state.direction.trim().to_owned());
                let kind = if state.seen {
                    ChatInputKind::Seen
                } else {
                    ChatInputKind::Heard
                };
                let _ = commands.send(VisualizerClientMessage::Command {
                    command: VisualizerCommand::SendSensoryInput {
                        tab_id: tab_id.clone(),
                        input: ChatInput {
                            kind,
                            direction,
                            content,
                        },
                    },
                });
                state.draft.clear();
            }
        }
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
