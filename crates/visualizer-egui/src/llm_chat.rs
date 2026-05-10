use serde::{Deserialize, Serialize};
use std::hash::Hash;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum LlmChatRole {
    System,
    Developer,
    User,
    Assistant,
    Tool,
    Other(String),
}

impl LlmChatRole {
    pub fn label(&self) -> &str {
        match self {
            Self::System => "system",
            Self::Developer => "developer",
            Self::User => "user",
            Self::Assistant => "assistant",
            Self::Tool => "tool",
            Self::Other(role) => role,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum LlmChatItemKind {
    Text,
    Reasoning,
    Refusal,
    ToolCall,
    ToolResult,
    Structured,
    Other(String),
}

impl LlmChatItemKind {
    pub fn label(&self) -> &str {
        match self {
            Self::Text => "text",
            Self::Reasoning => "reasoning",
            Self::Refusal => "refusal",
            Self::ToolCall => "tool call",
            Self::ToolResult => "tool result",
            Self::Structured => "structured",
            Self::Other(kind) => kind,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LlmChatMessage {
    pub role: LlmChatRole,
    pub kind: LlmChatItemKind,
    pub content: String,
    pub ephemeral: bool,
    pub streaming: bool,
    pub source: Option<String>,
}

impl LlmChatMessage {
    pub fn text(role: LlmChatRole, content: impl Into<String>) -> Self {
        Self {
            role,
            kind: LlmChatItemKind::Text,
            content: content.into(),
            ephemeral: false,
            streaming: false,
            source: None,
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct LlmChatTranscript {
    pub messages: Vec<LlmChatMessage>,
}

impl LlmChatTranscript {
    pub fn push(&mut self, message: LlmChatMessage) {
        self.messages.push(message);
    }

    pub fn append_to_message(&mut self, index: usize, delta: &str) {
        if let Some(message) = self.messages.get_mut(index) {
            message.content.push_str(delta);
            message.streaming = true;
        }
    }

    pub fn replace_message(&mut self, index: usize, content: String) {
        if let Some(message) = self.messages.get_mut(index) {
            message.content = content;
            message.streaming = false;
        }
    }
}

pub fn ui(ui: &mut egui::Ui, transcript: &LlmChatTranscript) {
    ui_with_id(ui, "llm-chat", transcript);
}

pub fn ui_with_id(ui: &mut egui::Ui, id_salt: impl Hash, transcript: &LlmChatTranscript) {
    egui::ScrollArea::vertical()
        .id_salt(id_salt)
        .stick_to_bottom(true)
        .show(ui, |ui| {
            for message in &transcript.messages {
                message_ui(ui, message);
                ui.add_space(6.0);
            }
        });
}

fn message_ui(ui: &mut egui::Ui, message: &LlmChatMessage) {
    let is_user = matches!(message.role, LlmChatRole::User);
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
                    if message.kind != LlmChatItemKind::Text {
                        ui.label(message.kind.label());
                    }
                    if message.ephemeral {
                        ui.label("ephemeral");
                    }
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

pub fn wrapped_label(ui: &mut egui::Ui, text: &str) {
    let width = ui.available_width().max(120.0);
    let display = hard_wrap_long_segments(text, 96);
    ui.scope(|ui| {
        ui.set_max_width(width);
        ui.add(egui::Label::new(display).wrap());
    });
}

pub fn hard_wrap_long_segments(text: &str, limit: usize) -> String {
    let mut out = String::with_capacity(text.len());
    let mut run = 0_usize;
    for ch in text.chars() {
        if ch == '\n' {
            run = 0;
            out.push(ch);
            continue;
        }

        if ch.is_whitespace() {
            run = 0;
            out.push(ch);
            continue;
        }

        if run >= limit {
            out.push('\n');
            run = 0;
        }
        out.push(ch);
        run += 1;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hard_wrap_long_segments_breaks_unspaced_content() {
        let wrapped = hard_wrap_long_segments("abcdefghijkl", 4);
        assert_eq!(wrapped, "abcd\nefgh\nijkl");
    }
}
