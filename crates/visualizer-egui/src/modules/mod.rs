pub mod attention_controller;
pub mod attention_schema;
pub mod cognition_gate;
pub mod memory;
pub mod memory_compaction;
pub mod predict;
pub mod query_agentic;
pub mod query_vector;
pub mod self_model;
pub mod sensory;
pub mod speak;
pub mod surprise;

use std::collections::{BTreeMap, VecDeque};

use egui_hooks::UseHookExt as _;
use nuillu_module::RuntimeEvent;

use crate::{
    LlmTraceEvent,
    llm_chat::{LlmChatItemKind, LlmChatMessage, LlmChatRole, LlmChatTranscript, wrapped_label},
};

const UNATTRIBUTED_LLM_OWNER: &str = "unattributed-llm";

#[derive(Debug, Default)]
pub struct ModulesState {
    modules: BTreeMap<String, ModuleState>,
    pending_llm_owners: VecDeque<String>,
    span_to_owner: BTreeMap<String, String>,
}

impl ModulesState {
    pub fn iter(&self) -> impl Iterator<Item = &ModuleState> {
        self.modules.values()
    }
}

#[derive(Debug, Default)]
pub struct ModuleState {
    pub owner: String,
    pub turns: Vec<LlmTurnState>,
    pub status: ModuleSessionStatus,
    pub last_tier: Option<String>,
}

#[derive(Debug, Clone)]
pub struct LlmTurnState {
    pub span_id: String,
    pub kind: Option<String>,
    pub model: Option<String>,
    pub request_id: Option<String>,
    pub finish_reason: Option<String>,
    pub input: LlmChatTranscript,
    pub output: LlmChatTranscript,
    pub status: ModuleSessionStatus,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum ModuleSessionStatus {
    #[default]
    Idle,
    Running,
    Retrying,
    Completed,
    Failed,
}

pub fn apply_runtime_event(state: &mut ModulesState, event: &RuntimeEvent) {
    match event {
        RuntimeEvent::LlmAccessed { owner, tier, .. } => {
            let owner = owner.to_string();
            state.pending_llm_owners.push_back(owner.clone());
            let module = module_mut(state, owner);
            module.status = ModuleSessionStatus::Running;
            module.last_tier = Some(format!("{tier:?}"));
        }
        RuntimeEvent::RateLimitDelayed { owner, .. }
        | RuntimeEvent::ModuleBatchThrottled { owner, .. } => {
            module_mut(state, owner.to_string()).status = ModuleSessionStatus::Retrying;
        }
        RuntimeEvent::MemoUpdated { owner, .. } => {
            let module = module_mut(state, owner.to_string());
            if module.status == ModuleSessionStatus::Running {
                module.status = ModuleSessionStatus::Completed;
            }
        }
    }
}

pub fn apply_trace_event(state: &mut ModulesState, event: LlmTraceEvent) {
    match event {
        LlmTraceEvent::ModuleSpanOpened { span_id, owner } => {
            state.span_to_owner.insert(span_id, owner);
        }
        LlmTraceEvent::SpanOpened {
            span_id,
            parent_span_id,
            kind,
            model,
        } => {
            let owner = parent_span_id
                .as_ref()
                .and_then(|parent| state.span_to_owner.get(parent))
                .cloned()
                .or_else(|| state.pending_llm_owners.pop_front())
                .unwrap_or_else(|| UNATTRIBUTED_LLM_OWNER.to_string());
            state.span_to_owner.insert(span_id.clone(), owner.clone());
            let module = module_mut(state, owner);
            module.status = ModuleSessionStatus::Running;
            module.turns.push(LlmTurnState {
                span_id,
                kind,
                model,
                request_id: None,
                finish_reason: None,
                input: LlmChatTranscript::default(),
                output: LlmChatTranscript::default(),
                status: ModuleSessionStatus::Running,
            });
        }
        LlmTraceEvent::SpanRecorded {
            span_id,
            model,
            request_id,
            finish_reason,
        } => {
            if let Some(turn) = turn_mut(state, &span_id) {
                if model.is_some() {
                    turn.model = model;
                }
                if request_id.is_some() {
                    turn.request_id = request_id;
                }
                if finish_reason.is_some() {
                    turn.finish_reason = finish_reason;
                }
            }
        }
        LlmTraceEvent::Input {
            span_id,
            transcript,
        } => {
            ensure_turn(state, &span_id).input = transcript;
        }
        LlmTraceEvent::OutputDelta {
            span_id,
            kind,
            delta,
        } => {
            let turn = ensure_turn(state, &span_id);
            let index = turn
                .output
                .messages
                .iter()
                .rposition(|message| message.streaming)
                .unwrap_or_else(|| {
                    let index = turn.output.messages.len();
                    turn.output.push(LlmChatMessage {
                        role: LlmChatRole::Assistant,
                        kind,
                        content: String::new(),
                        ephemeral: false,
                        streaming: true,
                        source: None,
                    });
                    index
                });
            turn.output.append_to_message(index, &delta);
            turn.status = ModuleSessionStatus::Running;
        }
        LlmTraceEvent::OutputCompleted {
            span_id,
            transcript,
        } => {
            let turn = ensure_turn(state, &span_id);
            turn.output = transcript;
            turn.status = ModuleSessionStatus::Completed;
            if let Some(owner) = state.span_to_owner.get(&span_id).cloned() {
                module_mut(state, owner).status = ModuleSessionStatus::Completed;
            }
        }
        LlmTraceEvent::Error { span_id, message } => {
            let turn = ensure_turn(state, &span_id);
            turn.output.push(LlmChatMessage {
                role: LlmChatRole::Assistant,
                kind: LlmChatItemKind::Other("error".to_string()),
                content: message,
                ephemeral: false,
                streaming: false,
                source: None,
            });
            turn.status = ModuleSessionStatus::Failed;
        }
        LlmTraceEvent::SpanClosed { span_id } => {
            if let Some(turn) = turn_mut(state, &span_id)
                && turn.status == ModuleSessionStatus::Running
            {
                turn.status = ModuleSessionStatus::Completed;
            }
        }
    }
}

pub fn render_module(ui: &mut egui::Ui, module: &ModuleState) {
    ui.horizontal_wrapped(|ui| {
        ui.heading(module_title(module));
        if let Some(tier) = &module.last_tier {
            ui.label(format!("tier: {tier}"));
        }
        ui.label(format!("turns: {}", module.turns.len()));
    });
    ui.separator();

    let Some(default_turn_id) = active_turn_id(module) else {
        ui.label("No LLM turns yet.");
        return;
    };
    let persisted_turn_id =
        ui.use_persisted_state(|| default_turn_id.clone(), module.owner.clone());
    let selected_turn_id = selected_turn_id(module, persisted_turn_id.as_str(), &default_turn_id);
    if selected_turn_id != persisted_turn_id.as_str() {
        persisted_turn_id.set_next(selected_turn_id.clone());
    }

    let mut next_turn_id = selected_turn_id.clone();
    ui.horizontal(|ui| {
        ui.set_min_height(320.0);
        ui.vertical(|ui| {
            ui.set_width(190.0);
            ui.strong("Turns");
            ui.separator();
            render_turn_list(ui, module, &selected_turn_id, &mut next_turn_id);
        });

        ui.separator();

        ui.vertical(|ui| {
            ui.set_min_width((ui.available_width() - 8.0).max(260.0));
            if let Some(turn) = module
                .turns
                .iter()
                .find(|turn| turn.span_id == next_turn_id)
            {
                render_active_turn(ui, module, turn);
            }
        });
    });
    if next_turn_id != selected_turn_id {
        persisted_turn_id.set_next(next_turn_id);
    }
}

pub fn window_title(module: &ModuleState) -> String {
    module_title(module)
}

fn render_turn_list(
    ui: &mut egui::Ui,
    module: &ModuleState,
    selected_turn_id: &str,
    next_turn_id: &mut String,
) {
    egui::ScrollArea::vertical()
        .id_salt(format!("turn-list:{}", module.owner))
        .show(ui, |ui| {
            for (index, turn) in module.turns.iter().enumerate() {
                let selected = turn.span_id == selected_turn_id;
                let response = ui
                    .selectable_label(selected, turn_list_label(index, turn))
                    .on_hover_text(&turn.span_id);
                if response.clicked() {
                    *next_turn_id = turn.span_id.clone();
                }
            }
        });
}

fn render_active_turn(ui: &mut egui::Ui, module: &ModuleState, turn: &LlmTurnState) {
    ui.horizontal_wrapped(|ui| {
        ui.strong(format!(
            "{} turn {}",
            status_icon(turn.status),
            turn.span_id
        ));
        if let Some(kind) = &turn.kind {
            ui.label(kind);
        }
        if let Some(model) = &turn.model {
            ui.label(model);
        }
        if let Some(request_id) = &turn.request_id {
            wrapped_label(ui, &format!("request: {request_id}"));
        }
        if let Some(finish_reason) = &turn.finish_reason {
            ui.label(format!("finish: {finish_reason}"));
        }
    });
    ui.separator();
    let transcript = merged_turn_transcript(turn);
    let id = format!("active-turn:{}:{}", module.owner, turn.span_id);
    crate::llm_chat::ui_with_id(ui, id, &transcript);
}

fn turn_list_label(index: usize, turn: &LlmTurnState) -> String {
    let kind = turn.kind.as_deref().unwrap_or("turn");
    format!("{} {:02} {}", status_icon(turn.status), index + 1, kind)
}

fn merged_turn_transcript(turn: &LlmTurnState) -> LlmChatTranscript {
    let mut transcript = turn.input.clone();
    transcript.messages.extend(turn.output.messages.clone());
    transcript
}

fn active_turn_id(module: &ModuleState) -> Option<String> {
    module
        .turns
        .iter()
        .rev()
        .find(|turn| turn.status == ModuleSessionStatus::Running)
        .or_else(|| module.turns.last())
        .map(|turn| turn.span_id.clone())
}

fn selected_turn_id(module: &ModuleState, persisted: &str, default_turn_id: &str) -> String {
    let running_turn_id = module
        .turns
        .iter()
        .rev()
        .find(|turn| turn.status == ModuleSessionStatus::Running)
        .map(|turn| turn.span_id.as_str());
    if let Some(running_turn_id) = running_turn_id
        && running_turn_id != persisted
    {
        return running_turn_id.to_string();
    }
    if module.turns.iter().any(|turn| turn.span_id == persisted) {
        persisted.to_string()
    } else {
        default_turn_id.to_string()
    }
}

fn ensure_turn<'a>(state: &'a mut ModulesState, span_id: &str) -> &'a mut LlmTurnState {
    if turn_mut(state, span_id).is_none() {
        let owner = state
            .span_to_owner
            .get(span_id)
            .cloned()
            .unwrap_or_else(|| UNATTRIBUTED_LLM_OWNER.to_string());
        state
            .span_to_owner
            .insert(span_id.to_string(), owner.clone());
        module_mut(state, owner).turns.push(LlmTurnState {
            span_id: span_id.to_string(),
            kind: None,
            model: None,
            request_id: None,
            finish_reason: None,
            input: LlmChatTranscript::default(),
            output: LlmChatTranscript::default(),
            status: ModuleSessionStatus::Running,
        });
    }
    turn_mut(state, span_id).expect("turn inserted")
}

fn turn_mut<'a>(state: &'a mut ModulesState, span_id: &str) -> Option<&'a mut LlmTurnState> {
    let owner = state.span_to_owner.get(span_id)?.clone();
    state
        .modules
        .get_mut(&owner)?
        .turns
        .iter_mut()
        .find(|turn| turn.span_id == span_id)
}

fn module_mut(state: &mut ModulesState, owner: String) -> &mut ModuleState {
    state
        .modules
        .entry(owner.clone())
        .or_insert_with(|| ModuleState {
            owner,
            ..ModuleState::default()
        })
}

fn module_title(module: &ModuleState) -> String {
    format!("{} {}", status_icon(module.status), module.owner)
}

fn status_icon(status: ModuleSessionStatus) -> &'static str {
    match status {
        ModuleSessionStatus::Idle => "⚪",
        ModuleSessionStatus::Running => "🟢",
        ModuleSessionStatus::Retrying => "⏱️",
        ModuleSessionStatus::Completed => "✅",
        ModuleSessionStatus::Failed => "❌",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{LlmTraceEvent, llm_chat::LlmChatRole};
    use nuillu_types::{ModuleInstanceId, ReplicaIndex, builtin};

    #[test]
    fn trace_events_update_module_stream_state() {
        let mut state = ModulesState::default();
        let owner = ModuleInstanceId::new(builtin::sensory(), ReplicaIndex::ZERO);
        apply_runtime_event(
            &mut state,
            &RuntimeEvent::LlmAccessed {
                sequence: 0,
                owner: owner.clone(),
                call: 0,
                tier: nuillu_types::ModelTier::Default,
            },
        );
        apply_trace_event(
            &mut state,
            LlmTraceEvent::SpanOpened {
                span_id: "1".to_string(),
                parent_span_id: None,
                kind: Some("text".to_string()),
                model: None,
            },
        );
        apply_trace_event(
            &mut state,
            LlmTraceEvent::OutputDelta {
                span_id: "1".to_string(),
                kind: LlmChatItemKind::Text,
                delta: "world".to_string(),
            },
        );

        let module = state
            .modules
            .get(&owner.to_string())
            .expect("module exists");
        assert_eq!(module.turns[0].output.messages[0].content, "world");
        assert_eq!(
            module.turns[0].output.messages[0].role,
            LlmChatRole::Assistant
        );
        assert_eq!(module.status, ModuleSessionStatus::Running);
    }

    #[test]
    fn trace_parent_span_maps_llm_turn_to_module_owner() {
        let mut state = ModulesState::default();
        let owner = ModuleInstanceId::new(builtin::memory(), ReplicaIndex::ZERO).to_string();

        apply_trace_event(
            &mut state,
            LlmTraceEvent::ModuleSpanOpened {
                span_id: "activate-1".to_string(),
                owner: owner.clone(),
            },
        );
        apply_trace_event(
            &mut state,
            LlmTraceEvent::SpanOpened {
                span_id: "turn-1".to_string(),
                parent_span_id: Some("activate-1".to_string()),
                kind: Some("text".to_string()),
                model: None,
            },
        );

        assert!(state.modules.contains_key(&owner));
        assert!(!state.modules.contains_key(UNATTRIBUTED_LLM_OWNER));
    }
}
