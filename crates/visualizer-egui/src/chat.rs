use std::collections::BTreeMap;
use std::sync::mpsc::Sender;

use nuillu_module::SensoryInput;

use crate::{
    ChatInput, ChatInputKind, UtteranceDeltaView, UtteranceView, VisualizerCommand,
    VisualizerTabId,
    llm_chat::{LlmChatItemKind, LlmChatMessage, LlmChatRole, LlmChatTranscript},
};

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct UtteranceKey {
    sender: String,
    target: String,
    generation_id: u64,
}

#[derive(Debug, Default)]
pub struct ChatState {
    pub transcript: LlmChatTranscript,
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
        let mut message = LlmChatMessage::text(LlmChatRole::User, content);
        message.source = Some(source);
        self.transcript.push(message);
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
            let index = self.transcript.messages.len();
            self.streaming_utterances.insert(key, index);
            self.transcript.push(LlmChatMessage {
                role: LlmChatRole::Assistant,
                kind: LlmChatItemKind::Text,
                content: String::new(),
                ephemeral: false,
                streaming: true,
                source: Some(format!("{} -> {}", delta.sender, delta.target)),
            });
            index
        };
        self.transcript.append_to_message(index, &delta.delta);
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
            self.transcript.replace_message(index, utterance.text);
            return;
        }

        let mut message = LlmChatMessage::text(LlmChatRole::Assistant, utterance.text);
        message.source = Some(format!(
            "{} -> {} at {}",
            utterance.sender, utterance.target, utterance.emitted_at
        ));
        self.transcript.push(message);
    }
}

pub fn ui(
    ui: &mut egui::Ui,
    tab_id: &VisualizerTabId,
    state: &mut ChatState,
    commands: &Sender<VisualizerCommand>,
) {
    crate::llm_chat::ui(ui, &state.transcript);

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
                let _ = commands.send(VisualizerCommand::SendSensoryInput {
                    tab_id: tab_id.clone(),
                    input: ChatInput {
                        kind,
                        direction,
                        content,
                    },
                });
                state.draft.clear();
            }
        }
    });
}
