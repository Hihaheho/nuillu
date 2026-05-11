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

use std::collections::BTreeMap;
use std::hash::Hash;

use egui_hooks::UseHookExt as _;
use nuillu_module::RuntimeEvent;

use crate::{
    LlmInputItemView, LlmObservationEvent, LlmObservationSource, LlmUsageView,
    text::{hard_wrap_long_segments, wrapped_label},
};

#[derive(Debug, Default)]
pub struct ModulesState {
    modules: BTreeMap<String, ModuleState>,
    turn_to_owner: BTreeMap<String, String>,
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
    pub turn_id: String,
    pub operation: String,
    pub source: LlmObservationSource,
    pub tier: String,
    pub model: Option<String>,
    pub request_id: Option<String>,
    pub finish_reason: Option<String>,
    pub usage: Option<LlmUsageView>,
    pub input: Vec<LlmInputItemView>,
    pub output: Vec<LlmOutputItemState>,
    pub status: ModuleSessionStatus,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LlmOutputItemState {
    pub kind: String,
    pub content: String,
    pub streaming: bool,
    pub source: Option<String>,
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
            let module = module_mut(state, owner.to_string());
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

pub fn apply_llm_observation(state: &mut ModulesState, event: LlmObservationEvent) {
    match event {
        LlmObservationEvent::ModelInput {
            turn_id,
            owner,
            tier,
            source,
            operation,
            items,
            ..
        } => {
            state.turn_to_owner.insert(turn_id.clone(), owner.clone());
            let module = module_mut(state, owner);
            module.status = ModuleSessionStatus::Running;
            module.last_tier = Some(tier.clone());
            let turn = ensure_turn(module, turn_id, operation, source, tier);
            turn.input = items;
            turn.status = ModuleSessionStatus::Running;
        }
        LlmObservationEvent::StreamStarted {
            turn_id,
            owner,
            tier,
            source,
            operation,
            request_id,
            model,
            ..
        } => {
            state.turn_to_owner.insert(turn_id.clone(), owner.clone());
            let module = module_mut(state, owner);
            module.status = ModuleSessionStatus::Running;
            module.last_tier = Some(tier.clone());
            let turn = ensure_turn(module, turn_id, operation, source, tier);
            turn.model = Some(model);
            turn.request_id = request_id;
            turn.status = ModuleSessionStatus::Running;
        }
        LlmObservationEvent::StreamDelta {
            turn_id,
            kind,
            delta,
        } => {
            if let Some(turn) = turn_mut(state, &turn_id) {
                append_output_delta(turn, kind, delta);
                turn.status = ModuleSessionStatus::Running;
            }
        }
        LlmObservationEvent::ToolCallChunk {
            turn_id,
            id,
            name,
            arguments_json_delta,
        } => {
            if let Some(turn) = turn_mut(state, &turn_id) {
                append_output_delta(
                    turn,
                    "tool_call".to_string(),
                    format!("{name}({id}) {arguments_json_delta}"),
                );
            }
        }
        LlmObservationEvent::ToolCallReady {
            turn_id,
            id,
            name,
            arguments_json,
        } => {
            if let Some(turn) = turn_mut(state, &turn_id) {
                turn.output.push(LlmOutputItemState {
                    kind: "tool_call_ready".to_string(),
                    content: arguments_json,
                    streaming: false,
                    source: Some(format!("{name}({id})")),
                });
            }
        }
        LlmObservationEvent::StructuredReady { turn_id, json } => {
            if let Some(turn) = turn_mut(state, &turn_id) {
                apply_structured_ready(turn, json);
            }
        }
        LlmObservationEvent::Completed {
            turn_id,
            request_id,
            finish_reason,
            usage,
        } => {
            let owner = state.turn_to_owner.get(&turn_id).cloned();
            if let Some(turn) = turn_mut(state, &turn_id) {
                if request_id.is_some() {
                    turn.request_id = request_id;
                }
                turn.finish_reason = Some(finish_reason);
                turn.usage = Some(usage);
                for row in &mut turn.output {
                    row.streaming = false;
                }
                turn.status = ModuleSessionStatus::Completed;
            }
            if let Some(owner) = owner {
                module_mut(state, owner).status = ModuleSessionStatus::Completed;
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
                .find(|turn| turn.turn_id == next_turn_id)
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
                let selected = turn.turn_id == selected_turn_id;
                let response = ui
                    .selectable_label(selected, turn_list_label(index, turn))
                    .on_hover_text(&turn.turn_id);
                if response.clicked() {
                    *next_turn_id = turn.turn_id.clone();
                }
            }
        });
}

fn render_active_turn(ui: &mut egui::Ui, module: &ModuleState, turn: &LlmTurnState) {
    ui.horizontal_wrapped(|ui| {
        ui.strong(format!(
            "{} turn {}",
            status_label(turn.status),
            turn.turn_id
        ));
        ui.label(&turn.operation);
        ui.label(turn.source.label());
        ui.label(&turn.tier);
        if let Some(model) = &turn.model {
            ui.label(model);
        }
        if let Some(request_id) = &turn.request_id {
            wrapped_label(ui, &format!("request: {request_id}"));
        }
        if let Some(finish_reason) = &turn.finish_reason {
            ui.label(format!("finish: {finish_reason}"));
        }
        if let Some(usage) = &turn.usage {
            ui.label(format!(
                "tokens: {}/{}",
                usage.input_tokens, usage.output_tokens
            ));
        }
    });
    ui.separator();
    let id = format!("active-turn:{}:{}", module.owner, turn.turn_id);
    egui::ScrollArea::vertical()
        .id_salt(id)
        .stick_to_bottom(true)
        .show(ui, |ui| {
            for (index, item) in turn.input.iter().enumerate() {
                render_input_item(ui, item, &turn.turn_id, index);
                ui.add_space(6.0);
            }
            for (index, item) in turn.output.iter().enumerate() {
                render_output_item(ui, item, &turn.turn_id, index);
                ui.add_space(6.0);
            }
        });
}

fn render_input_item(ui: &mut egui::Ui, item: &LlmInputItemView, turn_id: &str, index: usize) {
    egui::Frame::new()
        .fill(ui.visuals().extreme_bg_color)
        .stroke(ui.visuals().widgets.noninteractive.bg_stroke)
        .corner_radius(egui::CornerRadius::same(6))
        .inner_margin(egui::Margin::same(8))
        .show(ui, |ui| {
            ui.horizontal_wrapped(|ui| {
                ui.strong(&item.role);
                ui.label(&item.kind);
                if item.ephemeral {
                    ui.label("ephemeral");
                }
                if let Some(source) = &item.source {
                    wrapped_label(ui, source);
                }
            });
            ui.add_space(3.0);
            render_input_item_content(ui, item, turn_id, index);
        });
}

fn render_output_item(ui: &mut egui::Ui, item: &LlmOutputItemState, turn_id: &str, index: usize) {
    egui::Frame::new()
        .fill(ui.visuals().selection.bg_fill.linear_multiply(0.45))
        .stroke(ui.visuals().widgets.noninteractive.bg_stroke)
        .corner_radius(egui::CornerRadius::same(6))
        .inner_margin(egui::Margin::same(8))
        .show(ui, |ui| {
            ui.horizontal_wrapped(|ui| {
                ui.strong("assistant");
                ui.label(&item.kind);
                if item.streaming {
                    ui.label("streaming");
                }
                if let Some(source) = &item.source {
                    wrapped_label(ui, source);
                }
            });
            ui.add_space(3.0);
            render_output_item_content(ui, item, turn_id, index);
        });
}

fn render_input_item_content(
    ui: &mut egui::Ui,
    item: &LlmInputItemView,
    turn_id: &str,
    index: usize,
) {
    match item.kind.as_str() {
        "tool_call" => {
            if !render_json_block(
                ui,
                (
                    "input-json",
                    turn_id,
                    index,
                    item.kind.as_str(),
                    item.source.as_deref(),
                ),
                "tool input JSON",
                &item.content,
            ) {
                wrapped_label(ui, &item.content);
            }
        }
        "tool_result" => {
            if let Some((arguments, result)) = split_tool_result_content(&item.content) {
                render_json_or_raw_tool_part(
                    ui,
                    (
                        "input-tool-result-arguments",
                        turn_id,
                        index,
                        item.source.as_deref(),
                    ),
                    "tool input JSON",
                    "arguments",
                    arguments,
                );
                ui.add_space(4.0);
                render_json_or_raw_tool_part(
                    ui,
                    (
                        "input-tool-result-output",
                        turn_id,
                        index,
                        item.source.as_deref(),
                    ),
                    "tool output JSON",
                    "result",
                    result,
                );
            } else {
                wrapped_label(ui, &item.content);
            }
        }
        _ => wrapped_label(ui, &item.content),
    }
}

fn render_output_item_content(
    ui: &mut egui::Ui,
    item: &LlmOutputItemState,
    turn_id: &str,
    index: usize,
) {
    let json_label = match item.kind.as_str() {
        "structured" | "structured_ready" => Some("structured JSON"),
        "tool_call_ready" => Some("tool input JSON"),
        _ => None,
    };
    if let Some(label) = json_label
        && render_json_block(
            ui,
            (
                "output-json",
                turn_id,
                index,
                item.kind.as_str(),
                item.source.as_deref(),
            ),
            label,
            &item.content,
        )
    {
        return;
    }
    wrapped_label(ui, &item.content);
}

fn render_json_or_raw_tool_part(
    ui: &mut egui::Ui,
    id_salt: impl Hash,
    json_label: &str,
    raw_label: &str,
    content: &str,
) {
    if render_json_block(ui, id_salt, json_label, content) {
        return;
    }
    wrapped_label(ui, &format!("{raw_label}:\n{content}"));
}

fn render_json_block(ui: &mut egui::Ui, id_salt: impl Hash, label: &str, content: &str) -> bool {
    let Some(json) = format_json_for_display(content) else {
        return false;
    };
    egui::CollapsingHeader::new(format!("{label}: {}", json_preview(&json.compact)))
        .default_open(false)
        .id_salt(id_salt)
        .show(ui, |ui| {
            let display = hard_wrap_long_segments(&json.pretty, 120);
            ui.add(egui::Label::new(egui::RichText::new(display).monospace()).wrap());
        });
    true
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct JsonDisplay {
    pretty: String,
    compact: String,
}

fn format_json_for_display(content: &str) -> Option<JsonDisplay> {
    let value = serde_json::from_str::<serde_json::Value>(content).ok()?;
    let pretty = serde_json::to_string_pretty(&value).ok()?;
    let compact = serde_json::to_string(&value).ok()?;
    Some(JsonDisplay { pretty, compact })
}

fn json_preview(compact: &str) -> String {
    const LIMIT: usize = 96;
    if compact.chars().count() <= LIMIT {
        return compact.to_string();
    }
    let mut preview: String = compact.chars().take(LIMIT).collect();
    preview.push_str("...");
    preview
}

fn split_tool_result_content(content: &str) -> Option<(&str, &str)> {
    let rest = content.strip_prefix("arguments:\n")?;
    rest.split_once("\nresult:\n")
}

fn turn_list_label(index: usize, turn: &LlmTurnState) -> String {
    format!(
        "{} {:02} {} {}",
        status_label(turn.status),
        index + 1,
        turn.operation,
        turn.source.label()
    )
}

fn active_turn_id(module: &ModuleState) -> Option<String> {
    module
        .turns
        .iter()
        .rev()
        .find(|turn| turn.status == ModuleSessionStatus::Running)
        .or_else(|| module.turns.last())
        .map(|turn| turn.turn_id.clone())
}

fn selected_turn_id(module: &ModuleState, persisted: &str, default_turn_id: &str) -> String {
    let running_turn_id = module
        .turns
        .iter()
        .rev()
        .find(|turn| turn.status == ModuleSessionStatus::Running)
        .map(|turn| turn.turn_id.as_str());
    if let Some(running_turn_id) = running_turn_id
        && running_turn_id != persisted
    {
        return running_turn_id.to_string();
    }
    if module.turns.iter().any(|turn| turn.turn_id == persisted) {
        persisted.to_string()
    } else {
        default_turn_id.to_string()
    }
}

fn ensure_turn<'a>(
    module: &'a mut ModuleState,
    turn_id: String,
    operation: String,
    source: LlmObservationSource,
    tier: String,
) -> &'a mut LlmTurnState {
    if let Some(index) = module.turns.iter().position(|turn| turn.turn_id == turn_id) {
        let turn = &mut module.turns[index];
        turn.operation = operation;
        turn.source = source;
        turn.tier = tier;
        return turn;
    }
    module.turns.push(LlmTurnState {
        turn_id,
        operation,
        source,
        tier,
        model: None,
        request_id: None,
        finish_reason: None,
        usage: None,
        input: Vec::new(),
        output: Vec::new(),
        status: ModuleSessionStatus::Running,
    });
    module.turns.last_mut().expect("turn inserted")
}

fn turn_mut<'a>(state: &'a mut ModulesState, turn_id: &str) -> Option<&'a mut LlmTurnState> {
    let owner = state.turn_to_owner.get(turn_id)?.clone();
    state
        .modules
        .get_mut(&owner)?
        .turns
        .iter_mut()
        .find(|turn| turn.turn_id == turn_id)
}

fn append_output_delta(turn: &mut LlmTurnState, kind: String, delta: String) {
    if let Some(row) = turn
        .output
        .iter_mut()
        .rev()
        .find(|row| row.streaming && row.kind == kind)
    {
        row.content.push_str(&delta);
        return;
    }
    turn.output.push(LlmOutputItemState {
        kind,
        content: delta,
        streaming: true,
        source: None,
    });
}

fn apply_structured_ready(turn: &mut LlmTurnState, json: String) {
    if let Some(row) = turn
        .output
        .iter_mut()
        .rev()
        .find(|row| row.kind == "structured" || row.kind == "structured_ready")
    {
        row.kind = "structured_ready".to_string();
        row.content = json;
        row.streaming = false;
        row.source = None;
        return;
    }
    turn.output.push(LlmOutputItemState {
        kind: "structured_ready".to_string(),
        content: json,
        streaming: false,
        source: None,
    });
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
    format!("{} {}", status_label(module.status), module.owner)
}

fn status_label(status: ModuleSessionStatus) -> &'static str {
    match status {
        ModuleSessionStatus::Idle => "idle",
        ModuleSessionStatus::Running => "running",
        ModuleSessionStatus::Retrying => "retrying",
        ModuleSessionStatus::Completed => "done",
        ModuleSessionStatus::Failed => "failed",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{LlmObservationSource, LlmUsageView};
    use nuillu_types::{ModuleInstanceId, ReplicaIndex, builtin};

    #[test]
    fn observation_events_update_module_stream_state() {
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
        apply_llm_observation(
            &mut state,
            LlmObservationEvent::ModelInput {
                turn_id: "turn-1".to_string(),
                owner: owner.to_string(),
                module: "sensory".to_string(),
                replica: 0,
                tier: "Default".to_string(),
                source: LlmObservationSource::ModuleTurn,
                operation: "text_turn".to_string(),
                items: vec![LlmInputItemView {
                    role: "user".to_string(),
                    kind: "text".to_string(),
                    content: "hello".to_string(),
                    ephemeral: false,
                    source: None,
                }],
            },
        );
        apply_llm_observation(
            &mut state,
            LlmObservationEvent::StreamDelta {
                turn_id: "turn-1".to_string(),
                kind: "text".to_string(),
                delta: "world".to_string(),
            },
        );

        let module = state
            .modules
            .get(&owner.to_string())
            .expect("module exists");
        assert_eq!(module.turns[0].output[0].content, "world");
        assert_eq!(module.turns[0].input[0].content, "hello");
        assert_eq!(module.status, ModuleSessionStatus::Running);
    }

    #[test]
    fn completion_marks_turn_done_and_preserves_compaction_source() {
        let mut state = ModulesState::default();
        let owner = ModuleInstanceId::new(builtin::memory(), ReplicaIndex::ZERO).to_string();

        apply_llm_observation(
            &mut state,
            LlmObservationEvent::ModelInput {
                turn_id: "turn-2".to_string(),
                owner: owner.clone(),
                module: "memory".to_string(),
                replica: 0,
                tier: "Cheap".to_string(),
                source: LlmObservationSource::SessionCompaction,
                operation: "text_turn".to_string(),
                items: Vec::new(),
            },
        );
        apply_llm_observation(
            &mut state,
            LlmObservationEvent::Completed {
                turn_id: "turn-2".to_string(),
                request_id: Some("req-2".to_string()),
                finish_reason: "stop".to_string(),
                usage: LlmUsageView {
                    input_tokens: 2,
                    output_tokens: 3,
                    total_tokens: 5,
                    cost_micros_usd: 0,
                    cache_creation_tokens: 0,
                    cache_read_tokens: 0,
                },
            },
        );

        let module = state.modules.get(&owner).expect("module exists");
        assert_eq!(
            module.turns[0].source,
            LlmObservationSource::SessionCompaction
        );
        assert_eq!(module.turns[0].status, ModuleSessionStatus::Completed);
        assert_eq!(module.status, ModuleSessionStatus::Completed);
    }

    #[test]
    fn structured_ready_replaces_streaming_structured_row() {
        let mut state = ModulesState::default();
        let owner = ModuleInstanceId::new(builtin::query_vector(), ReplicaIndex::ZERO).to_string();
        apply_llm_observation(
            &mut state,
            LlmObservationEvent::ModelInput {
                turn_id: "turn-3".to_string(),
                owner: owner.clone(),
                module: "query-vector".to_string(),
                replica: 0,
                tier: "Default".to_string(),
                source: LlmObservationSource::ModuleTurn,
                operation: "structured_turn".to_string(),
                items: Vec::new(),
            },
        );
        apply_llm_observation(
            &mut state,
            LlmObservationEvent::StreamDelta {
                turn_id: "turn-3".to_string(),
                kind: "structured".to_string(),
                delta: "{\"answer\":".to_string(),
            },
        );
        apply_llm_observation(
            &mut state,
            LlmObservationEvent::StructuredReady {
                turn_id: "turn-3".to_string(),
                json: "{\"answer\":true}".to_string(),
            },
        );

        let module = state.modules.get(&owner).expect("module exists");
        assert_eq!(
            module.turns[0].output,
            vec![LlmOutputItemState {
                kind: "structured_ready".to_string(),
                content: "{\"answer\":true}".to_string(),
                streaming: false,
                source: None,
            }]
        );
    }

    #[test]
    fn json_display_prettifies_valid_json() {
        let display = format_json_for_display("{\"answer\":true}").expect("valid json");

        assert_eq!(
            display,
            JsonDisplay {
                pretty: "{\n  \"answer\": true\n}".to_string(),
                compact: "{\"answer\":true}".to_string(),
            }
        );
    }

    #[test]
    fn json_display_rejects_invalid_json() {
        assert_eq!(format_json_for_display("{\"answer\":"), None);
    }

    #[test]
    fn tool_result_content_splits_arguments_and_result() {
        let content = "arguments:\n{\"query\":\"koro\"}\nresult:\n{\"matches\":[]}";

        assert_eq!(
            split_tool_result_content(content),
            Some(("{\"query\":\"koro\"}", "{\"matches\":[]}"))
        );
    }
}
