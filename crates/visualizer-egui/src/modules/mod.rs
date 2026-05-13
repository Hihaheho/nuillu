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
use std::time::Duration;

use egui_hooks::UseHookExt as _;
use nuillu_module::RuntimeEvent;

use crate::{
    AllocationView, BlackboardSnapshot, LlmInputItemView, LlmObservationEvent,
    LlmObservationSource, LlmUsageView, MemoView, ModulePolicyView, ModuleSettingsView,
    ModuleStatusView, ZeroReplicaWindowView, memos,
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

    pub fn get(&self, owner: &str) -> Option<&ModuleState> {
        self.modules.get(owner)
    }
}

#[derive(Debug, Default)]
pub struct ModuleState {
    pub owner: String,
    pub module: String,
    pub replica: u8,
    pub turns: Vec<LlmTurnState>,
    pub status: ModuleSessionStatus,
    pub runtime_status: Option<String>,
    pub last_tier: Option<String>,
    pub last_throttle: Option<ThrottleSummary>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ThrottleSummary {
    pub kind: String,
    pub detail: String,
    pub delayed_ms: u64,
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

#[derive(Debug, Clone, PartialEq)]
pub struct ModuleOverviewRow {
    pub owner: String,
    pub module: String,
    pub replica: u8,
    pub active: bool,
    pub forced_disabled: bool,
    pub runtime_status: String,
    pub llm_status: String,
    pub activation_ratio: Option<f64>,
    pub active_replicas: Option<u8>,
    pub tier: Option<String>,
    pub guidance: Option<String>,
    pub bpm: Option<f64>,
    pub cooldown_ms: Option<u64>,
    pub policy: Option<ModulePolicyView>,
    pub throttle: Option<String>,
    pub latest_llm_output: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ModuleOverviewAction {
    OpenModule { owner: String },
    SetDisabled { module: String, disabled: bool },
    SetModuleSettings { settings: ModuleSettingsView },
}

#[derive(Debug, Clone, PartialEq)]
struct OpenModuleConfig {
    module: String,
    anchor: egui::Pos2,
}

pub fn apply_blackboard_snapshot(state: &mut ModulesState, snapshot: &BlackboardSnapshot) {
    for status in &snapshot.module_statuses {
        apply_module_status(state, status);
    }
    for allocation in &snapshot.allocation {
        let owner = owner_for_replica(&allocation.module, 0);
        let module = module_mut_with_metadata(state, owner, allocation.module.clone(), 0);
        if module.runtime_status.is_none() {
            module.runtime_status = Some("not reported".to_string());
        }
    }
}

pub fn apply_runtime_event(state: &mut ModulesState, event: &RuntimeEvent) {
    match event {
        RuntimeEvent::LlmAccessed { owner, tier, .. } => {
            let module = module_mut_for_owner(state, owner);
            module.status = ModuleSessionStatus::Running;
            module.last_tier = Some(format!("{tier:?}"));
        }
        RuntimeEvent::RateLimitDelayed {
            owner,
            capability,
            delayed_for,
            ..
        } => {
            let module = module_mut_for_owner(state, owner);
            module.last_throttle = Some(ThrottleSummary {
                kind: "rate limit".to_string(),
                detail: format!("{capability:?}"),
                delayed_ms: duration_millis(*delayed_for),
            });
        }
        RuntimeEvent::ModuleBatchThrottled {
            owner, delayed_for, ..
        } => {
            let module = module_mut_for_owner(state, owner);
            module.last_throttle = Some(ThrottleSummary {
                kind: "batch throttle".to_string(),
                detail: "next_batch".to_string(),
                delayed_ms: duration_millis(*delayed_for),
            });
        }
        RuntimeEvent::MemoUpdated { owner, .. } => {
            let module = module_mut_for_owner(state, owner);
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
            module,
            replica,
            tier,
            source,
            operation,
            items,
            ..
        } => {
            state.turn_to_owner.insert(turn_id.clone(), owner.clone());
            let module_state = module_mut_with_metadata(state, owner, module, replica);
            module_state.status = ModuleSessionStatus::Running;
            module_state.last_tier = Some(tier.clone());
            let turn = ensure_turn(module_state, turn_id, operation, source, tier);
            turn.input = items;
            turn.status = ModuleSessionStatus::Running;
        }
        LlmObservationEvent::StreamStarted {
            turn_id,
            owner,
            module,
            replica,
            tier,
            source,
            operation,
            request_id,
            model,
            ..
        } => {
            state.turn_to_owner.insert(turn_id.clone(), owner.clone());
            let module_state = module_mut_with_metadata(state, owner, module, replica);
            module_state.status = ModuleSessionStatus::Running;
            module_state.last_tier = Some(tier.clone());
            let turn = ensure_turn(module_state, turn_id, operation, source, tier);
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

pub fn overview_rows(
    state: &ModulesState,
    snapshot: &BlackboardSnapshot,
) -> Vec<ModuleOverviewRow> {
    let mut rows = BTreeMap::<String, ModuleOverviewRow>::new();

    for allocation in &snapshot.allocation {
        upsert_overview_row(
            &mut rows,
            owner_for_replica(&allocation.module, 0),
            &allocation.module,
            0,
        );
    }
    for status in &snapshot.module_statuses {
        let row = upsert_overview_row(
            &mut rows,
            status.owner.clone(),
            &status.module,
            status.replica,
        );
        row.runtime_status = status.status.clone();
    }
    for module in state.iter() {
        let module_name = module_name(module);
        let row = upsert_overview_row(
            &mut rows,
            module.owner.clone(),
            &module_name,
            module.replica,
        );
        if let Some(runtime_status) = &module.runtime_status {
            row.runtime_status = runtime_status.clone();
        }
        row.llm_status = status_label(module.status).to_string();
        if row.tier.is_none() {
            row.tier = module.last_tier.clone();
        }
        row.throttle = module.last_throttle.as_ref().map(throttle_label);
        row.latest_llm_output = latest_llm_output(module);
    }

    let allocations = snapshot
        .allocation
        .iter()
        .map(|allocation| (allocation.module.as_str(), allocation))
        .collect::<BTreeMap<_, _>>();
    let policies = snapshot
        .module_policies
        .iter()
        .map(|policy| (policy.module.as_str(), policy))
        .collect::<BTreeMap<_, _>>();
    for row in rows.values_mut() {
        if let Some(allocation) = allocations.get(row.module.as_str()) {
            apply_allocation_to_row(row, allocation);
        }
        if let Some(policy) = policies.get(row.module.as_str()) {
            row.policy = Some((*policy).clone());
        }
        row.forced_disabled = snapshot
            .forced_disabled_modules
            .iter()
            .any(|module| module == &row.module);
    }
    rows.retain(|_, row| overview_row_visible(row));

    rows.into_values().collect()
}

pub fn render_modules_overview(
    ui: &mut egui::Ui,
    snapshot: &BlackboardSnapshot,
    state: &ModulesState,
) -> Vec<ModuleOverviewAction> {
    let rows = overview_rows(state, snapshot);
    let mut actions = Vec::new();
    let open_config_id = ui.make_persistent_id("module-config-popup");
    let mut open_config = ui
        .ctx()
        .data(|data| data.get_temp::<OpenModuleConfig>(open_config_id));
    ui.horizontal_wrapped(|ui| {
        ui.heading("Modules");
        ui.label(format!("count: {}", rows.len()));
    });
    ui.separator();

    egui::ScrollArea::both()
        .id_salt("modules-overview")
        .show(ui, |ui| {
            overview_header(ui);
            ui.separator();
            for (index, row) in rows.iter().enumerate() {
                overview_row(ui, row, index, &mut actions, &mut open_config);
            }
        });

    render_open_config_popup(ui, snapshot, &mut open_config, &mut actions);
    if let Some(open_config) = open_config {
        ui.ctx()
            .data_mut(|data| data.insert_temp(open_config_id, open_config));
    } else {
        ui.ctx()
            .data_mut(|data| data.remove::<OpenModuleConfig>(open_config_id));
    }

    actions
}

pub fn render_module(ui: &mut egui::Ui, module: &ModuleState, memos: &[MemoView]) {
    let module_memos = module_memos(module, memos);
    ui.horizontal_wrapped(|ui| {
        ui.heading(module_title(module));
        if let Some(tier) = &module.last_tier {
            ui.label(format!("tier: {tier}"));
        }
        ui.label(format!("memos: {}", module_memos.len()));
        ui.label(format!("turns: {}", module.turns.len()));
    });
    ui.separator();

    let persisted_panel =
        ui.use_persisted_state(|| MODULE_MEMOS_SELECTION.to_string(), module.owner.clone());
    let selected_panel = selected_module_panel(module, persisted_panel.as_str());
    if selected_panel != persisted_panel.as_str() {
        persisted_panel.set_next(selected_panel.clone());
    }

    let mut next_panel = selected_panel.clone();
    let body_height = ui.available_height().max(MODULE_BODY_MIN_HEIGHT);
    ui.horizontal(|ui| {
        ui.set_min_height(body_height);
        ui.vertical(|ui| {
            ui.set_width(190.0);
            ui.strong("Views");
            ui.separator();
            render_module_selector(ui, module, &selected_panel, &mut next_panel);
        });

        ui.separator();

        ui.vertical(|ui| {
            ui.set_min_width((ui.available_width() - 8.0).max(260.0));
            if next_panel == MODULE_MEMOS_SELECTION {
                render_module_memos(ui, module, &module_memos);
            } else if let Some(turn_id) = next_panel.strip_prefix(MODULE_TURN_SELECTION_PREFIX)
                && let Some((turn_index, turn)) = module
                    .turns
                    .iter()
                    .enumerate()
                    .find(|(_, turn)| turn.turn_id == turn_id)
            {
                render_active_turn(ui, module, turn_index, turn);
            } else {
                render_module_memos(ui, module, &module_memos);
            }
        });
    });
    if next_panel != selected_panel {
        persisted_panel.set_next(next_panel);
    }
}

pub fn window_title(module: &ModuleState) -> String {
    format!("Module - {}", module.owner)
}

const MODULE_BODY_MIN_HEIGHT: f32 = 160.0;
const MODULE_MEMOS_SELECTION: &str = "memos";
const MODULE_TURN_SELECTION_PREFIX: &str = "turn:";

fn render_module_selector(
    ui: &mut egui::Ui,
    module: &ModuleState,
    selected_panel: &str,
    next_panel: &mut String,
) {
    egui::ScrollArea::vertical()
        .id_salt(format!("module-panel-list:{}", module.owner))
        .show(ui, |ui| {
            if ui
                .selectable_label(selected_panel == MODULE_MEMOS_SELECTION, "memos")
                .clicked()
            {
                *next_panel = MODULE_MEMOS_SELECTION.to_string();
            }
            for (index, turn) in module.turns.iter().enumerate() {
                let panel_id = turn_selection_id(&turn.turn_id);
                let selected = panel_id == selected_panel;
                ui.push_id(("turn-row", index, turn.turn_id.as_str()), |ui| {
                    let response = ui
                        .selectable_label(selected, turn_selector_label(index))
                        .on_hover_text(turn_selector_hover(turn));
                    if response.clicked() {
                        *next_panel = panel_id;
                    }
                });
            }
        });
}

fn render_module_memos(ui: &mut egui::Ui, module: &ModuleState, memos: &[&MemoView]) {
    ui.horizontal_wrapped(|ui| {
        ui.strong("memos");
        ui.label(format!("module: {}", module_name(module)));
        ui.label(format!("count: {}", memos.len()));
    });
    ui.separator();
    egui::ScrollArea::vertical()
        .id_salt(("module-memos", module.owner.as_str()))
        .show(ui, |ui| {
            if memos.is_empty() {
                ui.label("No memos yet.");
                return;
            }
            for memo in memos {
                memos::render_memo_card(ui, memo);
                ui.add_space(6.0);
            }
        });
}

fn render_active_turn(
    ui: &mut egui::Ui,
    module: &ModuleState,
    turn_index: usize,
    turn: &LlmTurnState,
) {
    ui.push_id(
        (
            "active-turn",
            module.owner.as_str(),
            turn_index,
            turn.turn_id.as_str(),
        ),
        |ui| render_active_turn_contents(ui, turn_index, turn),
    );
}

fn render_active_turn_contents(ui: &mut egui::Ui, turn_index: usize, turn: &LlmTurnState) {
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
    egui::ScrollArea::vertical()
        .id_salt("scroll-area")
        .stick_to_bottom(true)
        .show(ui, |ui| {
            ui.push_id("scroll-content", |ui| {
                for (index, item) in turn.input.iter().enumerate() {
                    render_input_item(ui, item, &turn.turn_id, turn_index, index);
                    ui.add_space(6.0);
                }
                for (index, item) in turn.output.iter().enumerate() {
                    render_output_item(ui, item, &turn.turn_id, turn_index, index);
                    ui.add_space(6.0);
                }
            });
        });
}

fn render_input_item(
    ui: &mut egui::Ui,
    item: &LlmInputItemView,
    turn_id: &str,
    turn_index: usize,
    index: usize,
) {
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
            render_input_item_content(ui, item, turn_id, turn_index, index);
        });
}

fn render_output_item(
    ui: &mut egui::Ui,
    item: &LlmOutputItemState,
    turn_id: &str,
    turn_index: usize,
    index: usize,
) {
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
            render_output_item_content(ui, item, turn_id, turn_index, index);
        });
}

fn render_input_item_content(
    ui: &mut egui::Ui,
    item: &LlmInputItemView,
    turn_id: &str,
    turn_index: usize,
    index: usize,
) {
    match item.kind.as_str() {
        "tool_call" => {
            if !render_json_block(
                ui,
                (
                    "input-json",
                    turn_id,
                    turn_index,
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
                        turn_index,
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
                        turn_index,
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
    turn_index: usize,
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
                turn_index,
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
    let id = ui.make_persistent_id(("json-block", id_salt));
    let header = format!("{label}: {}", json.compact);
    egui::collapsing_header::CollapsingState::load_with_default_open(ui.ctx(), id, false)
        .show_header(ui, |ui| {
            ui.add(
                egui::Label::new(egui::RichText::new(header).monospace())
                    .truncate()
                    .show_tooltip_when_elided(true),
            )
        })
        .body(|ui| {
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

fn split_tool_result_content(content: &str) -> Option<(&str, &str)> {
    let rest = content.strip_prefix("arguments:\n")?;
    rest.split_once("\nresult:\n")
}

const OVERVIEW_ROW_HEIGHT: f32 = 22.0;
const ACTIVE_COLUMN_WIDTH: f32 = 28.0;
const CONFIG_COLUMN_WIDTH: f32 = 54.0;
const MODULE_COLUMN_WIDTH: f32 = 130.0;
const REPLICA_COLUMN_WIDTH: f32 = 30.0;
const STATUS_COLUMN_WIDTH: f32 = 128.0;
const LLM_COLUMN_WIDTH: f32 = 80.0;
const ALLOCATION_COLUMN_WIDTH: f32 = 36.0;
const TIER_COLUMN_WIDTH: f32 = 60.0;
const BPM_COLUMN_WIDTH: f32 = 30.0;
const COOLDOWN_COLUMN_WIDTH: f32 = 40.0;
const THROTTLE_COLUMN_WIDTH: f32 = 40.0;
const LATEST_OUTPUT_COLUMN_WIDTH: f32 = 300.0;
const CONFIG_POPUP_WIDTH: f32 = 310.0;
const CONFIG_POPUP_ESTIMATED_HEIGHT: f32 = 160.0;
const CONFIG_POPUP_GAP: f32 = 6.0;

fn overview_header(ui: &mut egui::Ui) {
    ui.horizontal(|ui| {
        overview_header_cell(ui, "Enabled", ACTIVE_COLUMN_WIDTH);
        overview_header_cell(ui, "Configs", CONFIG_COLUMN_WIDTH);
        overview_header_cell(ui, "Module", MODULE_COLUMN_WIDTH);
        overview_header_cell(ui, "Replica", REPLICA_COLUMN_WIDTH);
        overview_header_cell(ui, "Alloc", ALLOCATION_COLUMN_WIDTH);
        overview_header_cell(ui, "BPM", BPM_COLUMN_WIDTH);
        overview_header_cell(ui, "Cooldown", COOLDOWN_COLUMN_WIDTH);
        overview_header_cell(ui, "Throttle", THROTTLE_COLUMN_WIDTH);
        overview_header_cell(ui, "Tier", TIER_COLUMN_WIDTH);
        overview_header_cell(ui, "Runtime", STATUS_COLUMN_WIDTH);
        overview_header_cell(ui, "LLM", LLM_COLUMN_WIDTH);
        overview_header_cell(ui, "Latest LLM out", LATEST_OUTPUT_COLUMN_WIDTH);
    });
}

fn overview_row(
    ui: &mut egui::Ui,
    row: &ModuleOverviewRow,
    index: usize,
    actions: &mut Vec<ModuleOverviewAction>,
    open_config: &mut Option<OpenModuleConfig>,
) {
    let fill = (index % 2 == 1).then(|| ui.visuals().faint_bg_color);
    let frame = fill.map_or_else(egui::Frame::new, |fill| egui::Frame::new().fill(fill));
    frame.show(ui, |ui| {
        ui.horizontal(|ui| {
            overview_disable_cell(ui, row, actions);
            overview_config_cell(ui, row, open_config);
            overview_module_cell(ui, row, actions);
            overview_label_cell(ui, &replica_label(row), None, REPLICA_COLUMN_WIDTH);
            overview_label_cell(
                ui,
                &row.activation_ratio
                    .map(|ratio| format!("{ratio:.2}"))
                    .unwrap_or_else(|| "-".to_string()),
                row.guidance.as_deref(),
                ALLOCATION_COLUMN_WIDTH,
            );
            overview_label_cell(
                ui,
                &row.bpm.map(format_bpm).unwrap_or_else(|| "-".to_string()),
                None,
                BPM_COLUMN_WIDTH,
            );
            overview_label_cell(
                ui,
                &row.cooldown_ms
                    .map(format_millis)
                    .unwrap_or_else(|| "-".to_string()),
                None,
                COOLDOWN_COLUMN_WIDTH,
            );
            overview_label_cell(
                ui,
                row.throttle.as_deref().unwrap_or("-"),
                None,
                THROTTLE_COLUMN_WIDTH,
            );
            overview_label_cell(
                ui,
                row.tier.as_deref().unwrap_or("-"),
                None,
                TIER_COLUMN_WIDTH,
            );
            overview_label_cell(ui, &row.runtime_status, None, STATUS_COLUMN_WIDTH);
            overview_label_cell(ui, &row.llm_status, None, LLM_COLUMN_WIDTH);
            let output = row.latest_llm_output.as_deref().unwrap_or("-");
            let output_preview = tail_preview_text(output, 96);
            overview_label_cell(
                ui,
                &output_preview,
                row.latest_llm_output.as_deref(),
                LATEST_OUTPUT_COLUMN_WIDTH,
            );
        });
    });
}

fn overview_header_cell(ui: &mut egui::Ui, text: &str, width: f32) {
    ui.add_sized(
        [width, OVERVIEW_ROW_HEIGHT],
        egui::Label::new(egui::RichText::new(text).strong())
            .halign(egui::Align::Min)
            .truncate(),
    );
}

fn overview_disable_cell(
    ui: &mut egui::Ui,
    row: &ModuleOverviewRow,
    actions: &mut Vec<ModuleOverviewAction>,
) {
    ui.allocate_ui_with_layout(
        egui::vec2(ACTIVE_COLUMN_WIDTH, OVERVIEW_ROW_HEIGHT),
        egui::Layout::left_to_right(egui::Align::Center),
        |ui| {
            let mut enabled = !row.forced_disabled;
            if ui
                .add(egui::Checkbox::without_text(&mut enabled))
                .on_hover_text("Allow this module to use allocated replicas")
                .changed()
            {
                actions.push(ModuleOverviewAction::SetDisabled {
                    module: row.module.clone(),
                    disabled: !enabled,
                });
            }
        },
    );
}

fn overview_config_cell(
    ui: &mut egui::Ui,
    row: &ModuleOverviewRow,
    open_config: &mut Option<OpenModuleConfig>,
) {
    ui.allocate_ui_with_layout(
        egui::vec2(CONFIG_COLUMN_WIDTH, OVERVIEW_ROW_HEIGHT),
        egui::Layout::left_to_right(egui::Align::Center),
        |ui| {
            let Some(policy) = &row.policy else {
                ui.label("-");
                return;
            };
            let response = ui.add_sized(
                [CONFIG_COLUMN_WIDTH, OVERVIEW_ROW_HEIGHT],
                egui::Button::new("Edit"),
            );
            let anchor = response.rect.right_top();
            let clicked = response.clicked();
            response.on_hover_text(format!("edit {}", policy.module));
            if clicked {
                if open_config
                    .as_ref()
                    .is_some_and(|open| open.module == policy.module)
                {
                    *open_config = None;
                } else {
                    *open_config = Some(OpenModuleConfig {
                        module: policy.module.clone(),
                        anchor,
                    });
                }
            }
        },
    );
}

fn render_open_config_popup(
    ui: &mut egui::Ui,
    snapshot: &BlackboardSnapshot,
    open_config: &mut Option<OpenModuleConfig>,
    actions: &mut Vec<ModuleOverviewAction>,
) {
    let Some(open) = open_config.clone() else {
        return;
    };
    let Some(policy) = snapshot
        .module_policies
        .iter()
        .find(|policy| policy.module == open.module)
        .cloned()
    else {
        *open_config = None;
        return;
    };

    let mut close = false;
    let popup_pos = clamped_config_popup_pos(ui.ctx(), open.anchor);
    egui::Area::new(egui::Id::new((
        "module-config-popup",
        policy.module.as_str(),
    )))
    .order(egui::Order::Foreground)
    .fixed_pos(popup_pos)
    .show(ui.ctx(), |ui| {
        egui::Frame::popup(ui.style()).show(ui, |ui| {
            ui.set_min_width(CONFIG_POPUP_WIDTH);
            ui.set_max_width(CONFIG_POPUP_WIDTH);
            ui.horizontal(|ui| {
                ui.strong("Configs");
                ui.label(&policy.module);
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui.small_button("x").on_hover_text("close").clicked() {
                        close = true;
                    }
                });
            });
            ui.separator();
            render_config_editor(ui, &policy, actions);
        });
    });
    if close {
        *open_config = None;
    }
}

fn clamped_config_popup_pos(ctx: &egui::Context, anchor: egui::Pos2) -> egui::Pos2 {
    let bounds = ctx.content_rect();
    let mut pos = anchor + egui::vec2(CONFIG_POPUP_GAP, 0.0);
    if pos.x + CONFIG_POPUP_WIDTH > bounds.right() {
        pos.x = (anchor.x - CONFIG_POPUP_WIDTH - CONFIG_POPUP_GAP).max(bounds.left());
    }
    if pos.y + CONFIG_POPUP_ESTIMATED_HEIGHT > bounds.bottom() {
        pos.y = (bounds.bottom() - CONFIG_POPUP_ESTIMATED_HEIGHT).max(bounds.top());
    }
    pos.y = pos.y.max(bounds.top());
    pos
}

fn render_config_editor(
    ui: &mut egui::Ui,
    policy: &ModulePolicyView,
    actions: &mut Vec<ModuleOverviewAction>,
) {
    let mut settings = ModuleSettingsView {
        module: policy.module.clone(),
        replica_min: policy.replica_min,
        replica_max: policy.replica_max,
        bpm_min: policy.bpm_min,
        bpm_max: policy.bpm_max,
        zero_replica_window: policy.zero_replica_window,
    };
    let mut changed = false;

    ui.horizontal(|ui| {
        ui.label("Replicas");
        changed |= ui
            .add(
                egui::DragValue::new(&mut settings.replica_min)
                    .range(0..=policy.replica_capacity)
                    .speed(1.0),
            )
            .changed();
        ui.label("to");
        changed |= ui
            .add(
                egui::DragValue::new(&mut settings.replica_max)
                    .range(0..=policy.replica_capacity)
                    .speed(1.0),
            )
            .changed();
        ui.label(format!("cap {}", policy.replica_capacity));
    });

    ui.horizontal(|ui| {
        ui.label("BPM");
        changed |= ui
            .add(
                egui::DragValue::new(&mut settings.bpm_min)
                    .range(0.001..=100_000.0)
                    .speed(0.25),
            )
            .changed();
        ui.label("to");
        changed |= ui
            .add(
                egui::DragValue::new(&mut settings.bpm_max)
                    .range(0.001..=100_000.0)
                    .speed(0.25),
            )
            .changed();
    });

    let mut zero_enabled = matches!(
        settings.zero_replica_window,
        ZeroReplicaWindowView::EveryControllerActivations { .. }
    );
    let mut zero_period = match settings.zero_replica_window {
        ZeroReplicaWindowView::Disabled => 3,
        ZeroReplicaWindowView::EveryControllerActivations { period } => period.max(1),
    };
    ui.horizontal(|ui| {
        changed |= ui.checkbox(&mut zero_enabled, "Zero window").changed();
        ui.add_enabled_ui(zero_enabled, |ui| {
            ui.label("period");
            changed |= ui
                .add(
                    egui::DragValue::new(&mut zero_period)
                        .range(1..=10_000)
                        .speed(1.0),
                )
                .changed();
        });
    });

    settings.replica_min = settings.replica_min.min(policy.replica_capacity);
    settings.replica_max = settings.replica_max.min(policy.replica_capacity);
    if settings.replica_min > settings.replica_max {
        settings.replica_max = settings.replica_min;
    }
    if settings.bpm_min > settings.bpm_max {
        settings.bpm_max = settings.bpm_min;
    }
    settings.zero_replica_window = if zero_enabled {
        ZeroReplicaWindowView::EveryControllerActivations {
            period: zero_period.max(1),
        }
    } else {
        ZeroReplicaWindowView::Disabled
    };

    if changed {
        actions.push(ModuleOverviewAction::SetModuleSettings { settings });
    }
}

fn overview_module_cell(
    ui: &mut egui::Ui,
    row: &ModuleOverviewRow,
    actions: &mut Vec<ModuleOverviewAction>,
) {
    let mut response = ui.add_sized(
        [MODULE_COLUMN_WIDTH, OVERVIEW_ROW_HEIGHT],
        egui::Button::new(&row.module),
    );
    if row.owner != row.module {
        response = response.on_hover_text(format!("open {}", row.owner));
    }
    if response.clicked() {
        actions.push(ModuleOverviewAction::OpenModule {
            owner: row.owner.clone(),
        });
    }
}

fn overview_label_cell(ui: &mut egui::Ui, text: &str, hover: Option<&str>, width: f32) {
    let response = ui.add_sized(
        [width, OVERVIEW_ROW_HEIGHT],
        egui::Label::new(text)
            .truncate()
            .show_tooltip_when_elided(true),
    );
    if let Some(hover) = hover
        && !hover.is_empty()
        && hover != text
    {
        response.on_hover_text(hover);
    }
}

fn upsert_overview_row<'a>(
    rows: &'a mut BTreeMap<String, ModuleOverviewRow>,
    owner: String,
    module: &str,
    replica: u8,
) -> &'a mut ModuleOverviewRow {
    let row = rows
        .entry(owner.clone())
        .or_insert_with(|| ModuleOverviewRow {
            owner,
            module: module.to_string(),
            replica,
            active: false,
            forced_disabled: false,
            runtime_status: "not reported".to_string(),
            llm_status: status_label(ModuleSessionStatus::Idle).to_string(),
            activation_ratio: None,
            active_replicas: None,
            tier: None,
            guidance: None,
            bpm: None,
            cooldown_ms: None,
            policy: None,
            throttle: None,
            latest_llm_output: None,
        });
    if !module.is_empty() {
        row.module = module.to_string();
    }
    row.replica = replica;
    row
}

fn apply_allocation_to_row(row: &mut ModuleOverviewRow, allocation: &AllocationView) {
    row.active = row.replica < allocation.active_replicas;
    row.activation_ratio = Some(allocation.activation_ratio);
    row.active_replicas = Some(allocation.active_replicas);
    row.bpm = allocation.bpm;
    row.cooldown_ms = allocation.cooldown_ms;
    row.tier = Some(allocation.tier.clone());
    row.guidance = (!allocation.guidance.is_empty()).then(|| allocation.guidance.clone());
}

fn overview_row_visible(row: &ModuleOverviewRow) -> bool {
    if row.replica == 0 {
        return true;
    }
    if let Some(policy) = &row.policy
        && row.replica < policy.replica_max
    {
        return true;
    }
    row.llm_status != status_label(ModuleSessionStatus::Idle)
        || row.latest_llm_output.is_some()
        || row.throttle.is_some()
        || !matches!(row.runtime_status.as_str(), "Inactive" | "not reported")
}

fn apply_module_status(state: &mut ModulesState, status: &ModuleStatusView) {
    module_mut_with_metadata(
        state,
        status.owner.clone(),
        status.module.clone(),
        status.replica,
    )
    .runtime_status = Some(status.status.clone());
}

fn latest_llm_output(module: &ModuleState) -> Option<String> {
    let turn = module
        .turns
        .iter()
        .rev()
        .find(|turn| turn.status == ModuleSessionStatus::Running)
        .or_else(|| module.turns.last())?;
    let output = turn
        .output
        .iter()
        .rev()
        .find(|item| !item.content.trim().is_empty())?;
    Some(format!(
        "{}: {}",
        output.kind,
        tail_preview_text(&output.content, 512)
    ))
}

fn replica_label(row: &ModuleOverviewRow) -> String {
    let index = u16::from(row.replica) + 1;
    row.active_replicas
        .map(|total| format!("{index}/{total}"))
        .unwrap_or_else(|| format!("{index}/-"))
}

fn tail_preview_text(text: &str, max_chars: usize) -> String {
    let normalized = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if normalized.chars().count() <= max_chars {
        return normalized;
    }
    if max_chars <= 3 {
        return ".".repeat(max_chars);
    }
    let mut out = normalized
        .chars()
        .rev()
        .take(max_chars - 3)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect::<String>();
    out.insert_str(0, "...");
    out
}

fn throttle_label(summary: &ThrottleSummary) -> String {
    format_millis(summary.delayed_ms)
}

fn format_bpm(bpm: f64) -> String {
    if bpm >= 100.0 {
        format!("{bpm:.0}")
    } else if bpm >= 10.0 {
        format!("{bpm:.1}")
    } else {
        format!("{bpm:.2}")
    }
}

fn format_millis(ms: u64) -> String {
    if ms >= 1000 {
        format!("{:.1}s", ms as f64 / 1000.0)
    } else {
        format!("{ms}ms")
    }
}

fn duration_millis(duration: Duration) -> u64 {
    duration.as_millis().min(u128::from(u64::MAX)) as u64
}

fn owner_for_replica(module: &str, replica: u8) -> String {
    if replica == 0 {
        module.to_string()
    } else {
        format!("{module}[{replica}]")
    }
}

fn module_name(module: &ModuleState) -> String {
    if !module.module.is_empty() {
        module.module.clone()
    } else {
        infer_owner_parts(&module.owner).0
    }
}

fn module_memos<'a>(module: &ModuleState, memos: &'a [MemoView]) -> Vec<&'a MemoView> {
    let module_name = module_name(module);
    memos
        .iter()
        .filter(|memo| memo.module == module_name)
        .collect()
}

fn selected_module_panel(module: &ModuleState, persisted: &str) -> String {
    if persisted == MODULE_MEMOS_SELECTION {
        return MODULE_MEMOS_SELECTION.to_string();
    }
    let Some(turn_id) = persisted.strip_prefix(MODULE_TURN_SELECTION_PREFIX) else {
        return MODULE_MEMOS_SELECTION.to_string();
    };
    if module.turns.iter().any(|turn| turn.turn_id == turn_id) {
        persisted.to_string()
    } else {
        MODULE_MEMOS_SELECTION.to_string()
    }
}

fn turn_selection_id(turn_id: &str) -> String {
    format!("{MODULE_TURN_SELECTION_PREFIX}{turn_id}")
}

fn turn_selector_label(index: usize) -> String {
    format!("turn {}", index + 1)
}

fn turn_selector_hover(turn: &LlmTurnState) -> String {
    format!(
        "{} {} {} ({})",
        status_label(turn.status),
        turn.operation,
        turn.source.label(),
        turn.turn_id
    )
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
    let (module, replica) = infer_owner_parts(&owner);
    module_mut_with_metadata(state, owner, module, replica)
}

fn module_mut_for_owner<'a>(
    state: &'a mut ModulesState,
    owner: &nuillu_types::ModuleInstanceId,
) -> &'a mut ModuleState {
    module_mut_with_metadata(
        state,
        owner.to_string(),
        owner.module.as_str().to_string(),
        owner.replica.get(),
    )
}

fn module_mut_with_metadata(
    state: &mut ModulesState,
    owner: String,
    module: String,
    replica: u8,
) -> &mut ModuleState {
    let row = state
        .modules
        .entry(owner.clone())
        .or_insert_with(|| ModuleState {
            owner,
            module: module.clone(),
            replica,
            ..ModuleState::default()
        });
    if !module.is_empty() {
        row.module = module;
    }
    row.replica = replica;
    row
}

fn infer_owner_parts(owner: &str) -> (String, u8) {
    if let Some((module, replica)) = owner.rsplit_once('[')
        && let Some(replica) = replica.strip_suffix(']')
        && let Ok(replica) = replica.parse::<u8>()
    {
        return (module.to_string(), replica);
    }
    (owner.to_string(), 0)
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
        assert_eq!(module.module, "sensory");
        assert_eq!(module.replica, 0);
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

    #[test]
    fn tail_preview_omits_prefix_and_keeps_latest_text() {
        assert_eq!(
            tail_preview_text("alpha beta gamma delta epsilon", 18),
            "...a delta epsilon"
        );
    }

    #[test]
    fn blackboard_snapshot_creates_module_rows_without_llm_turns() {
        let mut state = ModulesState::default();
        let snapshot = BlackboardSnapshot {
            module_statuses: vec![ModuleStatusView {
                owner: "predict".to_string(),
                module: "predict".to_string(),
                replica: 0,
                status: "AwaitingBatch".to_string(),
            }],
            allocation: vec![AllocationView {
                module: "surprise".to_string(),
                activation_ratio: 0.25,
                active_replicas: 1,
                bpm: Some(9.0),
                cooldown_ms: Some(6667),
                tier: "Default".to_string(),
                guidance: String::new(),
            }],
            ..BlackboardSnapshot::default()
        };

        apply_blackboard_snapshot(&mut state, &snapshot);

        assert!(state.modules.contains_key("predict"));
        assert!(state.modules.contains_key("surprise"));
        assert_eq!(
            state
                .modules
                .get("predict")
                .and_then(|m| m.runtime_status.as_deref()),
            Some("AwaitingBatch")
        );
        assert_eq!(
            state.modules.get("surprise").map(|m| m.module.as_str()),
            Some("surprise")
        );
    }

    #[test]
    fn overview_rows_hide_capacity_only_inactive_replicas() {
        let state = ModulesState::default();
        let snapshot = BlackboardSnapshot {
            module_statuses: vec![
                ModuleStatusView {
                    owner: "sensory".to_string(),
                    module: "sensory".to_string(),
                    replica: 0,
                    status: "AwaitingBatch".to_string(),
                },
                ModuleStatusView {
                    owner: "sensory[1]".to_string(),
                    module: "sensory".to_string(),
                    replica: 1,
                    status: "Inactive".to_string(),
                },
            ],
            allocation: vec![AllocationView {
                module: "sensory".to_string(),
                activation_ratio: 1.0,
                active_replicas: 1,
                bpm: Some(18.0),
                cooldown_ms: Some(3333),
                tier: "Cheap".to_string(),
                guidance: String::new(),
            }],
            module_policies: vec![ModulePolicyView {
                module: "sensory".to_string(),
                replica_min: 0,
                replica_max: 1,
                replica_capacity: 2,
                bpm_min: 6.0,
                bpm_max: 18.0,
                zero_replica_window: ZeroReplicaWindowView::EveryControllerActivations {
                    period: 3,
                },
            }],
            ..BlackboardSnapshot::default()
        };

        let rows = overview_rows(&state, &snapshot);

        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].owner, "sensory");
    }

    #[test]
    fn module_memos_filter_by_module_across_replicas() {
        let module = ModuleState {
            owner: "query-vector".to_string(),
            module: "query-vector".to_string(),
            replica: 0,
            ..ModuleState::default()
        };
        let memos = vec![
            memo_view("query-vector", "query-vector", 0, 0, "primary memo"),
            memo_view("query-vector[1]", "query-vector", 1, 0, "replica memo"),
            memo_view("sensory", "sensory", 0, 0, "other memo"),
        ];

        let owners = module_memos(&module, &memos)
            .into_iter()
            .map(|memo| memo.owner.as_str())
            .collect::<Vec<_>>();

        assert_eq!(owners, vec!["query-vector", "query-vector[1]"]);
    }

    #[test]
    fn selected_module_panel_defaults_and_falls_back_to_memos() {
        let module = ModuleState {
            owner: "sensory".to_string(),
            module: "sensory".to_string(),
            turns: vec![LlmTurnState {
                turn_id: "turn-1".to_string(),
                operation: "text_turn".to_string(),
                source: LlmObservationSource::ModuleTurn,
                tier: "Default".to_string(),
                model: None,
                request_id: None,
                finish_reason: None,
                usage: None,
                input: Vec::new(),
                output: Vec::new(),
                status: ModuleSessionStatus::Running,
            }],
            ..ModuleState::default()
        };

        assert_eq!(selected_module_panel(&module, ""), "memos");
        assert_eq!(selected_module_panel(&module, "memos"), "memos");
        assert_eq!(selected_module_panel(&module, "turn:turn-1"), "turn:turn-1");
        assert_eq!(selected_module_panel(&module, "turn:missing"), "memos");
        assert_eq!(selected_module_panel(&module, "turn-1"), "memos");
    }

    #[test]
    fn runtime_events_record_throttle_summaries_without_changing_llm_status() {
        let mut state = ModulesState::default();
        let owner = ModuleInstanceId::new(builtin::query_vector(), ReplicaIndex::ZERO);

        apply_runtime_event(
            &mut state,
            &RuntimeEvent::RateLimitDelayed {
                sequence: 1,
                owner: owner.clone(),
                capability: nuillu_module::CapabilityKind::LlmCall,
                delayed_for: std::time::Duration::from_millis(25),
            },
        );

        let module = state
            .modules
            .get(&owner.to_string())
            .expect("module exists");
        assert_eq!(module.status, ModuleSessionStatus::Idle);
        assert_eq!(
            module.last_throttle,
            Some(ThrottleSummary {
                kind: "rate limit".to_string(),
                detail: "LlmCall".to_string(),
                delayed_ms: 25,
            })
        );
    }

    #[test]
    fn overview_rows_merge_allocation_status_throttle_and_latest_output() {
        let mut state = ModulesState::default();
        apply_llm_observation(
            &mut state,
            LlmObservationEvent::ModelInput {
                turn_id: "turn-4".to_string(),
                owner: "sensory".to_string(),
                module: "sensory".to_string(),
                replica: 0,
                tier: "Premium".to_string(),
                source: LlmObservationSource::ModuleTurn,
                operation: "text_turn".to_string(),
                items: Vec::new(),
            },
        );
        apply_llm_observation(
            &mut state,
            LlmObservationEvent::StreamDelta {
                turn_id: "turn-4".to_string(),
                kind: "text".to_string(),
                delta: "filtered observation".to_string(),
            },
        );
        state
            .modules
            .get_mut("sensory")
            .expect("module exists")
            .last_throttle = Some(ThrottleSummary {
            kind: "batch throttle".to_string(),
            detail: "next_batch".to_string(),
            delayed_ms: 500,
        });
        let snapshot = BlackboardSnapshot {
            module_statuses: vec![ModuleStatusView {
                owner: "sensory".to_string(),
                module: "sensory".to_string(),
                replica: 0,
                status: "Activating".to_string(),
            }],
            allocation: vec![AllocationView {
                module: "sensory".to_string(),
                activation_ratio: 0.75,
                active_replicas: 1,
                bpm: Some(12.5),
                cooldown_ms: Some(4800),
                tier: "Premium".to_string(),
                guidance: "inspect recent input".to_string(),
            }],
            ..BlackboardSnapshot::default()
        };

        let rows = overview_rows(&state, &snapshot);

        assert_eq!(
            rows,
            vec![ModuleOverviewRow {
                owner: "sensory".to_string(),
                module: "sensory".to_string(),
                replica: 0,
                active: true,
                forced_disabled: false,
                runtime_status: "Activating".to_string(),
                llm_status: "running".to_string(),
                activation_ratio: Some(0.75),
                active_replicas: Some(1),
                tier: Some("Premium".to_string()),
                guidance: Some("inspect recent input".to_string()),
                bpm: Some(12.5),
                cooldown_ms: Some(4800),
                policy: None,
                throttle: Some("500ms".to_string()),
                latest_llm_output: Some("text: filtered observation".to_string()),
            }]
        );
    }

    #[test]
    fn replica_label_uses_one_based_index_and_active_total() {
        let row = ModuleOverviewRow {
            owner: "query-vector[1]".to_string(),
            module: "query-vector".to_string(),
            replica: 1,
            active: true,
            forced_disabled: false,
            runtime_status: "Activating".to_string(),
            llm_status: "running".to_string(),
            activation_ratio: Some(1.0),
            active_replicas: Some(2),
            tier: Some("Default".to_string()),
            guidance: None,
            bpm: None,
            cooldown_ms: None,
            policy: None,
            throttle: None,
            latest_llm_output: None,
        };

        assert_eq!(replica_label(&row), "2/2");
    }

    #[test]
    fn module_window_title_is_stable_across_status_changes() {
        let mut module = ModuleState {
            owner: "sensory".to_string(),
            module: "sensory".to_string(),
            status: ModuleSessionStatus::Idle,
            ..ModuleState::default()
        };
        let idle_title = window_title(&module);
        module.status = ModuleSessionStatus::Running;

        assert_eq!(idle_title, "Module - sensory");
        assert_eq!(window_title(&module), idle_title);
    }

    fn memo_view(owner: &str, module: &str, replica: u8, index: u64, content: &str) -> MemoView {
        MemoView {
            owner: owner.to_string(),
            module: module.to_string(),
            replica,
            index,
            written_at: chrono::DateTime::parse_from_rfc3339("2026-05-13T00:00:00Z")
                .expect("valid timestamp")
                .with_timezone(&chrono::Utc),
            content: content.to_string(),
        }
    }
}
