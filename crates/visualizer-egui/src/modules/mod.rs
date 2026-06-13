pub mod allocation;
pub mod attention_schema;
pub mod cognition_gate;
pub mod memory;
pub mod memory_compaction;
pub mod predict;
pub mod query_memory;
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
    LlmObservationSource, LlmTranscriptTurnStatus, LlmTranscriptTurnView, LlmUsageView, MemoView,
    ModulePolicyView, ModuleSettingsView, ModuleStatusView, ZeroReplicaWindowView, memos,
    module_filter,
    module_filter::ModuleFilterState,
    text::{hard_wrap_long_segments, wrapped_label},
};

#[derive(Debug, Default)]
pub struct ModulesState {
    modules: BTreeMap<String, ModuleState>,
    turn_to_owner: BTreeMap<String, String>,
    turn_order: Vec<String>,
}

impl ModulesState {
    pub fn iter(&self) -> impl Iterator<Item = &ModuleState> {
        self.modules.values()
    }

    pub fn get(&self, owner: &str) -> Option<&ModuleState> {
        self.modules.get(owner)
    }

    pub fn module_names(&self) -> Vec<String> {
        self.modules
            .values()
            .map(module_name)
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .collect()
    }

    pub fn session_live_llm_turn_count(&self) -> u32 {
        self.modules
            .values()
            .map(|module| llm_turn_counts(module).total)
            .fold(0_u32, u32::saturating_add)
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
    pub latest_batch: Option<ModuleBatchDebugState>,
    pub activation_error_count: u32,
    pub activation_attempt_count: u32,
    pub last_execution_failed: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ThrottleSummary {
    pub kind: String,
    pub detail: String,
    pub delayed_ms: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModuleBatchDebugState {
    pub batch_type: String,
    pub debug: String,
}

#[derive(Debug, Clone)]
pub struct LlmTurnState {
    pub turn_id: String,
    pub operation: String,
    pub source: LlmObservationSource,
    pub session_key: Option<String>,
    pub tier: String,
    pub model: Option<String>,
    pub request_id: Option<String>,
    pub finish_reason: Option<String>,
    pub usage: Option<LlmUsageView>,
    pub error_message: Option<String>,
    pub batch: Option<ModuleBatchDebugState>,
    pub input: Vec<LlmInputItemView>,
    pub output: Vec<LlmOutputItemState>,
    pub status: ModuleSessionStatus,
    pub counted_in_session: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LlmOutputItemState {
    pub kind: String,
    pub content: String,
    pub streaming: bool,
    pub source: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LlmTurnListRow {
    pub owner: String,
    pub module: String,
    pub turn_id: String,
    pub session_key: Option<String>,
    pub turn_number: usize,
    pub label: String,
    pub streaming: bool,
    pub failed: bool,
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
    pub llm_streaming: bool,
    pub activation_ratio: Option<f64>,
    pub active_replicas: Option<u8>,
    pub tier: Option<String>,
    pub guidance: Option<String>,
    pub bpm: Option<f64>,
    pub period_ms: Option<u64>,
    pub policy: Option<ModulePolicyView>,
    pub throttle: Option<String>,
    pub latest_llm_output: Option<String>,
    pub activation_error_count: u32,
    pub activation_attempt_count: u32,
    pub llm_error_count: u32,
    pub llm_turn_count: u32,
    pub last_execution_failed: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ModuleOverviewAction {
    OpenModule { owner: String },
    SetDisabled { module: String, disabled: bool },
    SetModuleSettings { settings: ModuleSettingsView },
}

pub enum ModuleWindowAction {
    ResetSessionHistory { owner: String },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ActiveReplicaHighlight {
    AllocationDriven,
    MinReplicaDriven,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OverviewRowFill {
    Failed,
    LlmStreaming,
    Zebra,
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
        RuntimeEvent::LlmCompleted { owner, .. } => {
            let module = module_mut_for_owner(state, owner);
            if module.status == ModuleSessionStatus::Running {
                module.status = ModuleSessionStatus::Completed;
            }
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
        RuntimeEvent::ModuleBatchReady {
            owner,
            batch_type,
            batch_debug,
            ..
        } => {
            let module = module_mut_for_owner(state, owner);
            module.latest_batch = Some(ModuleBatchDebugState {
                batch_type: batch_type.clone(),
                debug: batch_debug.clone(),
            });
        }
        RuntimeEvent::ModuleActivationCompleted {
            owner, succeeded, ..
        } => {
            let module = module_mut_for_owner(state, owner);
            if *succeeded {
                module.activation_attempt_count = module.activation_attempt_count.saturating_add(1);
                module.last_execution_failed = false;
                if module
                    .runtime_status
                    .as_deref()
                    .is_some_and(|status| status.starts_with("Retrying activation "))
                {
                    module.runtime_status = Some("Activated".to_string());
                }
            }
        }
        RuntimeEvent::ModuleActivationAttemptFailed {
            owner,
            activation_attempt,
            max_attempts,
            message,
            ..
        } => {
            let module = module_mut_for_owner(state, owner);
            module.runtime_status = Some(format!(
                "Retrying activation {activation_attempt}/{max_attempts}: {message}"
            ));
            module.activation_error_count = module.activation_error_count.saturating_add(1);
            module.activation_attempt_count = module.activation_attempt_count.saturating_add(1);
            module.last_execution_failed = true;
        }
        RuntimeEvent::ModuleTaskFailed {
            owner,
            phase,
            message,
            ..
        } => {
            let module = module_mut_for_owner(state, owner);
            module.status = ModuleSessionStatus::Failed;
            module.runtime_status = Some(format!("Failed {phase}: {message}"));
            module.last_execution_failed = true;
        }
        RuntimeEvent::ModuleWarning { owner, message, .. } => {
            let module = module_mut_for_owner(state, owner);
            module.runtime_status = Some(format!("Warning: {message}"));
        }
        RuntimeEvent::SessionCompactionStarted {
            owner, session_key, ..
        } => {
            let module = module_mut_for_owner(state, owner);
            module.runtime_status = Some(format!("Compacting session {session_key}"));
        }
        RuntimeEvent::SessionCompactionCompleted {
            owner, session_key, ..
        } => {
            let module = module_mut_for_owner(state, owner);
            module.runtime_status = Some(format!("Compacted session {session_key}"));
        }
        RuntimeEvent::SessionCompactionFailed {
            owner,
            session_key,
            message,
            ..
        } => {
            let module = module_mut_for_owner(state, owner);
            module.runtime_status = Some(format!("Compaction failed {session_key}: {message}"));
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
            session_key,
            operation,
            items,
            ..
        } => {
            record_turn_owner(state, &turn_id, &owner);
            let module_state = module_mut_with_metadata(state, owner, module, replica);
            module_state.status = ModuleSessionStatus::Running;
            module_state.last_tier = Some(tier.clone());
            let batch = module_state.latest_batch.clone();
            let turn = ensure_turn(
                module_state,
                turn_id,
                operation,
                source,
                session_key,
                tier,
                batch,
            );
            turn.input = items;
            turn.status = ModuleSessionStatus::Running;
            turn.counted_in_session = true;
        }
        LlmObservationEvent::StreamStarted {
            turn_id,
            owner,
            module,
            replica,
            tier,
            source,
            session_key,
            operation,
            request_id,
            model,
            ..
        } => {
            record_turn_owner(state, &turn_id, &owner);
            let module_state = module_mut_with_metadata(state, owner, module, replica);
            module_state.status = ModuleSessionStatus::Running;
            module_state.last_tier = Some(tier.clone());
            let batch = module_state.latest_batch.clone();
            let turn = ensure_turn(
                module_state,
                turn_id,
                operation,
                source,
                session_key,
                tier,
                batch,
            );
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
                append_output_delta(turn, kind, delta, None);
                if turn.status != ModuleSessionStatus::Failed {
                    turn.status = ModuleSessionStatus::Running;
                }
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
                    arguments_json_delta,
                    Some(tool_call_source(&name, &id)),
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
                apply_tool_call_ready(turn, tool_call_source(&name, &id), arguments_json);
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
                if turn.status != ModuleSessionStatus::Failed {
                    turn.status = ModuleSessionStatus::Completed;
                }
            }
            if let Some(owner) = owner {
                let module = module_mut(state, owner);
                if module.status != ModuleSessionStatus::Failed {
                    module.status = ModuleSessionStatus::Completed;
                }
            }
        }
        LlmObservationEvent::Failed { turn_id, message } => {
            let owner = state.turn_to_owner.get(&turn_id).cloned();
            if let Some(turn) = turn_mut(state, &turn_id) {
                turn.error_message = Some(message.clone());
                for row in &mut turn.output {
                    row.streaming = false;
                }
                turn.status = ModuleSessionStatus::Failed;
            }
            if let Some(owner) = owner {
                let module = module_mut(state, owner);
                module.status = ModuleSessionStatus::Failed;
                module.runtime_status = Some(format!("LLM turn failed: {message}"));
                module.last_execution_failed = true;
            }
        }
    }
}

pub fn apply_llm_transcript_snapshot(state: &mut ModulesState, turns: Vec<LlmTranscriptTurnView>) {
    for turn in turns {
        apply_llm_transcript_turn(state, turn);
    }
}

fn apply_llm_transcript_turn(state: &mut ModulesState, turn: LlmTranscriptTurnView) {
    let LlmTranscriptTurnView {
        turn_id,
        owner,
        module,
        replica,
        tier,
        source,
        session_key,
        operation,
        input,
        output,
        request_id,
        model,
        finish_reason,
        usage,
        status,
        error_message,
    } = turn;

    record_turn_owner(state, &turn_id, &owner);
    let module_state = module_mut_with_metadata(state, owner, module, replica);
    let turn_state = ensure_turn(
        module_state,
        turn_id,
        operation,
        source,
        session_key,
        tier,
        None,
    );
    turn_state.input = input;
    turn_state.output = output
        .into_iter()
        .map(|item| LlmOutputItemState {
            kind: item.kind,
            content: item.content,
            streaming: false,
            source: item.source,
        })
        .collect();
    turn_state.request_id = request_id;
    turn_state.model = model;
    turn_state.finish_reason = finish_reason;
    turn_state.usage = usage;
    turn_state.error_message = error_message;
    turn_state.status = match status {
        LlmTranscriptTurnStatus::Completed => ModuleSessionStatus::Completed,
        LlmTranscriptTurnStatus::Failed => ModuleSessionStatus::Failed,
    };
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
        row.llm_streaming = module_llm_in_flight(module);
        if row.tier.is_none() {
            row.tier = module.last_tier.clone();
        }
        row.throttle = module.last_throttle.as_ref().map(throttle_label);
        row.latest_llm_output = latest_llm_output(module);
        row.activation_error_count = module.activation_error_count;
        row.activation_attempt_count = module.activation_attempt_count;
        let llm_counts = llm_turn_counts(module);
        row.llm_error_count = llm_counts.failed;
        row.llm_turn_count = llm_counts.total;
        row.last_execution_failed = module.last_execution_failed;
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

pub fn render_module(
    ui: &mut egui::Ui,
    module: &ModuleState,
    memos: &[MemoView],
) -> Vec<ModuleWindowAction> {
    let mut actions = Vec::new();
    let module_memos = module_memos(module, memos);
    ui.horizontal(|ui| {
        ui.heading(module_title(module));
        if let Some(tier) = &module.last_tier {
            ui.label(format!("tier: {tier}"));
        }
        ui.label(format!("memos: {}", module_memos.len()));
        ui.label(format!("turns: {}", module.turns.len()));
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            if ui.small_button("Reset").clicked() {
                actions.push(ModuleWindowAction::ResetSessionHistory {
                    owner: module.owner.clone(),
                });
            }
        });
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
        render_fixed_pane(ui, MODULE_SELECTOR_WIDTH, body_height, |ui| {
            ui.strong("Views");
            ui.separator();
            render_module_selector(ui, module, &selected_panel, &mut next_panel);
        });
        ui.separator();
        render_remaining_pane(ui, body_height, |ui| {
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
    actions
}

pub fn render_llm_turns(
    ui: &mut egui::Ui,
    state: &ModulesState,
    filter: &mut ModuleFilterState,
    modules: &[String],
) {
    module_filter::render_module_filter(ui, "llm-turns-module-filter", filter, modules);
    ui.separator();

    let rows = llm_turn_rows(state, |module| filter.is_selected(module));
    let persisted_selection = ui.use_persisted_state(String::new, "llm-turns-selection");
    let mut selected_turn_id = selected_llm_turn_id(persisted_selection.as_str(), &rows);

    let body_height = ui.available_height().max(MODULE_BODY_MIN_HEIGHT);
    ui.horizontal(|ui| {
        ui.set_min_height(body_height);
        render_fixed_pane(ui, LLM_TURN_SELECTOR_WIDTH, body_height, |ui| {
            ui.strong("Turns");
            ui.separator();
            render_llm_turn_selector(ui, &rows, &mut selected_turn_id);
        });
        ui.separator();
        render_remaining_pane(ui, body_height, |ui| {
            if rows.is_empty() {
                ui.label("No LLM turns yet.");
            } else if let Some(turn_id) = selected_turn_id.as_deref()
                && let Some((turn_index, module, turn)) = turn_by_id(state, turn_id)
            {
                render_active_turn(ui, module, turn_index, turn);
            } else {
                ui.label("Select a turn.");
            }
        });
    });

    if selected_turn_id.as_deref().unwrap_or_default() != persisted_selection.as_str() {
        persisted_selection.set_next(selected_turn_id.unwrap_or_default());
    }
}

pub fn window_title(module: &ModuleState) -> String {
    format!("Module - {}", module.owner)
}

const MODULE_BODY_MIN_HEIGHT: f32 = 160.0;
const MODULE_SELECTOR_WIDTH: f32 = 190.0;
const LLM_TURN_SELECTOR_WIDTH: f32 = 220.0;
const MODULE_MEMOS_SELECTION: &str = "memos";
const MODULE_TURN_SELECTION_PREFIX: &str = "turn:";

fn render_fixed_pane(
    ui: &mut egui::Ui,
    width: f32,
    body_height: f32,
    content: impl FnOnce(&mut egui::Ui),
) {
    let width = width.min(ui.available_width()).max(1.0);
    ui.allocate_ui_with_layout(
        egui::vec2(width, body_height),
        egui::Layout::top_down(egui::Align::Min),
        content,
    );
}

fn render_remaining_pane(ui: &mut egui::Ui, body_height: f32, content: impl FnOnce(&mut egui::Ui)) {
    let width = ui.available_width().max(1.0);
    ui.allocate_ui_with_layout(
        egui::vec2(width, body_height),
        egui::Layout::top_down(egui::Align::Min),
        content,
    );
}

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
                let turn_number = session_turn_number(module, index);
                ui.push_id(("turn-row", index, turn.turn_id.as_str()), |ui| {
                    failed_turn_frame(ui, turn.status).show(ui, |ui| {
                        let response = ui
                            .selectable_label(selected, turn_selector_label(turn_number, turn))
                            .on_hover_text(turn_selector_hover(turn));
                        if response.clicked() {
                            *next_panel = panel_id;
                        }
                    });
                });
            }
        });
}

fn render_llm_turn_selector(
    ui: &mut egui::Ui,
    rows: &[LlmTurnListRow],
    selected_turn_id: &mut Option<String>,
) {
    egui::ScrollArea::vertical()
        .id_salt("llm-turn-panel-list")
        .show(ui, |ui| {
            for (index, row) in rows.iter().enumerate() {
                let selected = selected_turn_id.as_deref() == Some(row.turn_id.as_str());
                ui.push_id(("llm-turn-row", index, row.turn_id.as_str()), |ui| {
                    let status = if row.failed {
                        ModuleSessionStatus::Failed
                    } else {
                        ModuleSessionStatus::Idle
                    };
                    failed_turn_frame(ui, status).show(ui, |ui| {
                        let response = ui
                            .selectable_label(selected, &row.label)
                            .on_hover_text(llm_turn_row_hover(row));
                        if response.clicked() {
                            *selected_turn_id = Some(row.turn_id.clone());
                        }
                    });
                });
            }
        });
}

fn failed_turn_frame(ui: &egui::Ui, status: ModuleSessionStatus) -> egui::Frame {
    if status == ModuleSessionStatus::Failed {
        egui::Frame::new()
            .fill(ui.visuals().error_fg_color.linear_multiply(0.14))
            .inner_margin(egui::Margin::symmetric(2, 0))
    } else {
        egui::Frame::new().inner_margin(egui::Margin::symmetric(2, 0))
    }
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
    if let Some(message) = &turn.error_message {
        render_turn_error_banner(ui, message);
        ui.add_space(6.0);
    }
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
                if let Some(batch) = &turn.batch {
                    render_batch_item(ui, batch, &turn.turn_id, turn_index);
                    ui.add_space(6.0);
                }
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

fn render_batch_item(
    ui: &mut egui::Ui,
    batch: &ModuleBatchDebugState,
    turn_id: &str,
    turn_index: usize,
) {
    egui::Frame::new()
        .fill(ui.visuals().faint_bg_color)
        .stroke(ui.visuals().widgets.noninteractive.bg_stroke)
        .corner_radius(egui::CornerRadius::same(6))
        .inner_margin(egui::Margin::same(8))
        .show(ui, |ui| {
            ui.horizontal_wrapped(|ui| {
                ui.strong("batch");
                wrapped_label(ui, &batch.batch_type);
            });
            ui.add_space(3.0);
            let display = hard_wrap_long_segments(&batch.debug, 120);
            ui.push_id(("batch-debug", turn_id, turn_index), |ui| {
                ui.add(egui::Label::new(egui::RichText::new(display).monospace()).wrap());
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
const PERIOD_COLUMN_WIDTH: f32 = 40.0;
const THROTTLE_COLUMN_WIDTH: f32 = 40.0;
const ACTIVATION_ERRORS_COLUMN_WIDTH: f32 = 64.0;
const LLM_ERRORS_COLUMN_WIDTH: f32 = 64.0;
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
        overview_header_cell(ui, "Period", PERIOD_COLUMN_WIDTH);
        overview_header_cell(ui, "Throttle", THROTTLE_COLUMN_WIDTH);
        overview_header_cell(ui, "Tier", TIER_COLUMN_WIDTH);
        overview_header_cell(ui, "Runtime", STATUS_COLUMN_WIDTH);
        overview_header_cell(ui, "LLM", LLM_COLUMN_WIDTH);
        overview_header_cell(ui, "Act Err", ACTIVATION_ERRORS_COLUMN_WIDTH);
        overview_header_cell(ui, "LLM Err", LLM_ERRORS_COLUMN_WIDTH);
        overview_header_cell(ui, "Latest LLM out", LATEST_OUTPUT_COLUMN_WIDTH);
    });
}

fn render_turn_error_banner(ui: &mut egui::Ui, message: &str) {
    let display = hard_wrap_long_segments(message, 96);
    egui::Frame::new()
        .fill(ui.visuals().error_fg_color.linear_multiply(0.16))
        .stroke(egui::Stroke::new(1.0, ui.visuals().error_fg_color))
        .corner_radius(egui::CornerRadius::same(6))
        .inner_margin(egui::Margin::same(8))
        .show(ui, |ui| {
            ui.strong(egui::RichText::new("error").color(ui.visuals().error_fg_color));
            ui.add(
                egui::Label::new(egui::RichText::new(display).monospace())
                    .wrap()
                    .selectable(true),
            );
        });
}

fn overview_row(
    ui: &mut egui::Ui,
    row: &ModuleOverviewRow,
    index: usize,
    actions: &mut Vec<ModuleOverviewAction>,
    open_config: &mut Option<OpenModuleConfig>,
) {
    let fill = overview_row_fill(row, index, ui.visuals());
    let frame = fill.map_or_else(egui::Frame::new, |fill| egui::Frame::new().fill(fill));
    frame.show(ui, |ui| {
        ui.horizontal(|ui| {
            overview_disable_cell(ui, row, actions);
            overview_config_cell(ui, row, open_config);
            overview_module_cell(ui, row, actions);
            overview_replica_cell(ui, row);
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
                &row.period_ms
                    .map(format_millis)
                    .unwrap_or_else(|| "-".to_string()),
                None,
                PERIOD_COLUMN_WIDTH,
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
            overview_llm_status_cell(ui, row);
            let activation_error_label = overview_activation_error_label(row);
            let activation_error_hover = format!(
                "{} activation errors / {} activation attempts in this visualizer session",
                row.activation_error_count, row.activation_attempt_count
            );
            overview_label_cell(
                ui,
                &activation_error_label,
                Some(&activation_error_hover),
                ACTIVATION_ERRORS_COLUMN_WIDTH,
            );
            let llm_error_label = overview_llm_error_label(row);
            let llm_error_hover = format!(
                "{} LLM errors / {} live LLM turns in this visualizer session",
                row.llm_error_count, row.llm_turn_count
            );
            overview_label_cell(
                ui,
                &llm_error_label,
                Some(&llm_error_hover),
                LLM_ERRORS_COLUMN_WIDTH,
            );
            let output = row.latest_llm_output.as_deref().unwrap_or("-");
            overview_latest_output_cell(ui, output, row.latest_llm_output.as_deref());
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

fn overview_row_fill(
    row: &ModuleOverviewRow,
    index: usize,
    visuals: &egui::Visuals,
) -> Option<egui::Color32> {
    match overview_row_fill_kind(row, index) {
        Some(OverviewRowFill::Failed) => Some(visuals.error_fg_color.linear_multiply(0.18)),
        Some(OverviewRowFill::LlmStreaming) => {
            Some(visuals.selection.bg_fill.linear_multiply(0.22))
        }
        Some(OverviewRowFill::Zebra) => Some(visuals.faint_bg_color),
        None => None,
    }
}

fn overview_row_fill_kind(row: &ModuleOverviewRow, index: usize) -> Option<OverviewRowFill> {
    if row.last_execution_failed {
        Some(OverviewRowFill::Failed)
    } else if row.llm_streaming {
        Some(OverviewRowFill::LlmStreaming)
    } else if index % 2 == 1 {
        Some(OverviewRowFill::Zebra)
    } else {
        None
    }
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

fn overview_replica_cell(ui: &mut egui::Ui, row: &ModuleOverviewRow) {
    let Some(highlight) = active_replica_highlight(row) else {
        overview_label_cell(ui, &replica_label(row), None, REPLICA_COLUMN_WIDTH);
        return;
    };

    let visuals = ui.visuals();
    let (fill, stroke, hover) = match highlight {
        ActiveReplicaHighlight::AllocationDriven => (
            Some(visuals.selection.bg_fill.linear_multiply(0.55)),
            visuals.selection.stroke,
            "Allocation-active replica",
        ),
        ActiveReplicaHighlight::MinReplicaDriven => (
            None,
            egui::Stroke::new(1.0, visuals.weak_text_color()),
            "Minimum replica kept active",
        ),
    };
    let frame = egui::Frame::new()
        .fill(fill.unwrap_or(egui::Color32::TRANSPARENT))
        .stroke(stroke)
        .inner_margin(egui::Margin::same(0));
    frame.show(ui, |ui| {
        overview_label_cell(ui, &replica_label(row), Some(hover), REPLICA_COLUMN_WIDTH);
    });
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

fn overview_llm_status_cell(ui: &mut egui::Ui, row: &ModuleOverviewRow) {
    let text = if row.llm_streaming {
        egui::RichText::new(&row.llm_status).strong()
    } else {
        egui::RichText::new(&row.llm_status)
    };
    let response = ui.add_sized(
        [LLM_COLUMN_WIDTH, OVERVIEW_ROW_HEIGHT],
        egui::Label::new(text)
            .truncate()
            .show_tooltip_when_elided(true),
    );
    if row.llm_streaming {
        response.on_hover_text("LLM request in flight");
    }
}

fn overview_latest_output_cell(ui: &mut egui::Ui, text: &str, hover: Option<&str>) {
    let (rect, response) = ui.allocate_exact_size(
        egui::vec2(LATEST_OUTPUT_COLUMN_WIDTH, OVERVIEW_ROW_HEIGHT),
        egui::Sense::hover(),
    );
    let color = ui.visuals().text_color();
    let font_id = egui::TextStyle::Body.resolve(ui.style());
    let display = latest_output_cell_text(ui, text, &font_id, color, rect.width());
    let galley = ui.painter().layout_no_wrap(display.clone(), font_id, color);
    let position = egui::pos2(
        rect.right() - galley.size().x,
        rect.center().y - (galley.size().y / 2.0),
    );
    ui.painter()
        .with_clip_rect(rect)
        .galley(position, galley, color);

    if let Some(hover) = hover
        && !hover.is_empty()
        && hover != display
    {
        response.on_hover_text(hover);
    }
}

fn latest_output_cell_text(
    ui: &egui::Ui,
    text: &str,
    font_id: &egui::FontId,
    color: egui::Color32,
    max_width: f32,
) -> String {
    latest_output_cell_text_with_width(text, max_width, |text| text_width(ui, text, font_id, color))
}

fn latest_output_cell_text_with_width(
    text: &str,
    max_width: f32,
    mut text_width: impl FnMut(&str) -> f32,
) -> String {
    if text_width(text) <= max_width {
        return text.to_string();
    }

    let chars = text.chars().collect::<Vec<_>>();
    let mut low = 0_usize;
    let mut high = chars.len();
    while low < high {
        let mid = (low + high).div_ceil(2);
        let suffix = chars[chars.len().saturating_sub(mid)..]
            .iter()
            .collect::<String>();
        if text_width(&suffix) <= max_width {
            low = mid;
        } else {
            high = mid - 1;
        }
    }

    if low == 0 {
        return chars.last().into_iter().collect();
    }
    chars[chars.len() - low..].iter().collect::<String>()
}

fn text_width(ui: &egui::Ui, text: &str, font_id: &egui::FontId, color: egui::Color32) -> f32 {
    ui.painter()
        .layout_no_wrap(text.to_string(), font_id.clone(), color)
        .size()
        .x
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
            llm_streaming: false,
            activation_ratio: None,
            active_replicas: None,
            tier: None,
            guidance: None,
            bpm: None,
            period_ms: None,
            policy: None,
            throttle: None,
            latest_llm_output: None,
            activation_error_count: 0,
            activation_attempt_count: 0,
            llm_error_count: 0,
            llm_turn_count: 0,
            last_execution_failed: false,
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
    row.period_ms = allocation.period_ms;
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
        || row.activation_error_count > 0
        || row.llm_error_count > 0
        || row.last_execution_failed
        || !matches!(row.runtime_status.as_str(), "Inactive" | "not reported")
}

fn overview_activation_error_label(row: &ModuleOverviewRow) -> String {
    format!(
        "{}/{}",
        row.activation_error_count, row.activation_attempt_count
    )
}

fn overview_llm_error_label(row: &ModuleOverviewRow) -> String {
    format!("{}/{}", row.llm_error_count, row.llm_turn_count)
}

fn active_replica_highlight(row: &ModuleOverviewRow) -> Option<ActiveReplicaHighlight> {
    if !row.active {
        return None;
    }
    let ratio = row.activation_ratio?;
    if ratio > 0.0 {
        Some(ActiveReplicaHighlight::AllocationDriven)
    } else {
        Some(ActiveReplicaHighlight::MinReplicaDriven)
    }
}

fn module_llm_in_flight(module: &ModuleState) -> bool {
    module.status == ModuleSessionStatus::Running
        || module
            .turns
            .iter()
            .any(|turn| turn.status != ModuleSessionStatus::Failed && turn_is_streaming(turn))
}

fn apply_module_status(state: &mut ModulesState, status: &ModuleStatusView) {
    let module = module_mut_with_metadata(
        state,
        status.owner.clone(),
        status.module.clone(),
        status.replica,
    );
    module.runtime_status = Some(status.status.clone());
    if status.status.starts_with("Failed") || status.status.starts_with("Stopped") {
        module.last_execution_failed = true;
    }
}

fn latest_llm_output(module: &ModuleState) -> Option<String> {
    if let Some(turn) = module.turns.last()
        && turn.status == ModuleSessionStatus::Running
        && turn
            .output
            .iter()
            .all(|item| item.content.trim().is_empty())
    {
        return Some("running: awaiting output".to_string());
    }
    module.turns.iter().rev().find_map(|turn| {
        turn.output
            .iter()
            .rev()
            .find(|item| !item.content.trim().is_empty())
            .map(output_text)
    })
}

fn output_text(output: &LlmOutputItemState) -> String {
    let content = normalize_latest_output_text(&output.content);
    match output.source.as_deref() {
        Some(source) if !source.is_empty() => format!("{} {source}: {content}", output.kind),
        _ => format!("{}: {content}", output.kind),
    }
}

fn normalize_latest_output_text(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn replica_label(row: &ModuleOverviewRow) -> String {
    let index = u16::from(row.replica) + 1;
    row.active_replicas
        .map(|total| format!("{index}/{total}"))
        .unwrap_or_else(|| format!("{index}/-"))
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

pub fn llm_turn_rows(
    state: &ModulesState,
    module_selected: impl Fn(&str) -> bool,
) -> Vec<LlmTurnListRow> {
    state
        .turn_order
        .iter()
        .filter_map(|turn_id| {
            let owner = state.turn_to_owner.get(turn_id)?;
            let module = state.modules.get(owner)?;
            let turn_index = module
                .turns
                .iter()
                .position(|turn| turn.turn_id == *turn_id)?;
            let turn = &module.turns[turn_index];
            let module_name = module_name(module);
            module_selected(&module_name).then(|| {
                let streaming = turn_is_streaming(turn);
                let turn_number = session_turn_number(module, turn_index);
                LlmTurnListRow {
                    owner: module.owner.clone(),
                    module: module_name,
                    turn_id: turn.turn_id.clone(),
                    session_key: turn.session_key.clone(),
                    turn_number,
                    label: llm_turn_row_label(
                        module_turn_label_module(module),
                        turn_session_label(turn),
                        turn_number,
                        streaming,
                    ),
                    streaming,
                    failed: turn.status == ModuleSessionStatus::Failed,
                }
            })
        })
        .collect()
}

pub fn selected_llm_turn_id(selected: &str, rows: &[LlmTurnListRow]) -> Option<String> {
    if rows.iter().any(|row| row.turn_id == selected) {
        return Some(selected.to_string());
    }
    rows.iter()
        .rev()
        .find(|row| row.streaming)
        .or_else(|| rows.last())
        .map(|row| row.turn_id.clone())
}

fn turn_by_id<'a>(
    state: &'a ModulesState,
    turn_id: &str,
) -> Option<(usize, &'a ModuleState, &'a LlmTurnState)> {
    let owner = state.turn_to_owner.get(turn_id)?;
    let module = state.modules.get(owner)?;
    let turn_index = module
        .turns
        .iter()
        .position(|turn| turn.turn_id == turn_id)?;
    Some((turn_index, module, &module.turns[turn_index]))
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

fn turn_selector_label(turn_number: usize, turn: &LlmTurnState) -> String {
    let label = format!("{} {turn_number}", turn_session_label(turn));
    if let Some(usage) = &turn.usage {
        return format!("{label} ({})", turn_usage_tokens(usage));
    }
    label
}

fn turn_selector_hover(turn: &LlmTurnState) -> String {
    let mut hover = format!(
        "{} {} {} {} ({})",
        status_label(turn.status),
        turn_session_label(turn),
        turn.operation,
        turn.source.label(),
        turn.turn_id
    );
    if let Some(usage) = &turn.usage {
        hover.push_str(&format!(" tokens: {}", turn_usage_tokens(usage)));
    }
    hover
}

fn turn_usage_tokens(usage: &LlmUsageView) -> u64 {
    usage.input_tokens.saturating_add(usage.output_tokens)
}

fn tool_call_source(name: &str, id: &str) -> String {
    format!("{name}({id})")
}

fn record_turn_owner(state: &mut ModulesState, turn_id: &str, owner: &str) -> bool {
    let new_turn = !state.turn_to_owner.contains_key(turn_id);
    if new_turn {
        state.turn_order.push(turn_id.to_string());
    }
    state
        .turn_to_owner
        .insert(turn_id.to_string(), owner.to_string());
    new_turn
}

fn module_turn_label_module(module: &ModuleState) -> String {
    if module.replica == 0 {
        module_name(module)
    } else {
        module.owner.clone()
    }
}

fn turn_session_label(turn: &LlmTurnState) -> &str {
    turn.session_key.as_deref().unwrap_or("turn")
}

fn session_turn_number(module: &ModuleState, turn_index: usize) -> usize {
    let session_key = &module.turns[turn_index].session_key;
    module.turns[..=turn_index]
        .iter()
        .filter(|turn| &turn.session_key == session_key)
        .count()
}

fn llm_turn_row_label(
    module: String,
    session_label: &str,
    turn_number: usize,
    streaming: bool,
) -> String {
    let label = format!("{module}.{session_label} {turn_number}");
    if streaming {
        format!("* {label}")
    } else {
        label
    }
}

fn llm_turn_row_hover(row: &LlmTurnListRow) -> String {
    let status = if row.streaming {
        "streaming"
    } else {
        "not streaming"
    };
    format!("{} {} ({status})", row.owner, row.turn_id)
}

fn turn_is_streaming(turn: &LlmTurnState) -> bool {
    turn.status == ModuleSessionStatus::Running || turn.output.iter().any(|item| item.streaming)
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct LlmTurnCounts {
    failed: u32,
    total: u32,
}

fn llm_turn_counts(module: &ModuleState) -> LlmTurnCounts {
    module
        .turns
        .iter()
        .filter(|turn| turn.counted_in_session)
        .fold(LlmTurnCounts::default(), |mut counts, turn| {
            counts.total = counts.total.saturating_add(1);
            if turn.status == ModuleSessionStatus::Failed {
                counts.failed = counts.failed.saturating_add(1);
            }
            counts
        })
}

fn ensure_turn(
    module: &mut ModuleState,
    turn_id: String,
    operation: String,
    source: LlmObservationSource,
    session_key: Option<String>,
    tier: String,
    batch: Option<ModuleBatchDebugState>,
) -> &mut LlmTurnState {
    if let Some(index) = module.turns.iter().position(|turn| turn.turn_id == turn_id) {
        let turn = &mut module.turns[index];
        turn.operation = operation;
        turn.source = source;
        if session_key.is_some() {
            turn.session_key = session_key;
        }
        turn.tier = tier;
        if turn.batch.is_none() {
            turn.batch = batch;
        }
        return turn;
    }
    module.turns.push(LlmTurnState {
        turn_id,
        operation,
        source,
        session_key,
        tier,
        model: None,
        request_id: None,
        finish_reason: None,
        usage: None,
        error_message: None,
        batch,
        input: Vec::new(),
        output: Vec::new(),
        status: ModuleSessionStatus::Running,
        counted_in_session: false,
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

fn append_output_delta(
    turn: &mut LlmTurnState,
    kind: String,
    delta: String,
    source: Option<String>,
) {
    if let Some(index) = turn
        .output
        .iter()
        .rposition(|row| row.streaming && row.kind == kind && row.source == source)
    {
        let mut row = turn.output.remove(index);
        row.content.push_str(&delta);
        turn.output.push(row);
        return;
    }
    turn.output.push(LlmOutputItemState {
        kind,
        content: delta,
        streaming: true,
        source,
    });
}

fn apply_tool_call_ready(turn: &mut LlmTurnState, source: String, arguments_json: String) {
    if let Some(index) = turn.output.iter().rposition(|row| {
        row.streaming && row.kind == "tool_call" && row.source.as_deref() == Some(source.as_str())
    }) {
        let mut row = turn.output.remove(index);
        row.kind = "tool_call_ready".to_string();
        row.content = arguments_json;
        row.streaming = false;
        row.source = Some(source);
        turn.output.push(row);
        return;
    }
    turn.output.push(LlmOutputItemState {
        kind: "tool_call_ready".to_string(),
        content: arguments_json,
        streaming: false,
        source: Some(source),
    });
}

fn apply_structured_ready(turn: &mut LlmTurnState, json: String) {
    if let Some(row) = turn
        .output
        .iter_mut()
        .rev()
        .find(|row| row.kind == "structured_ready")
    {
        row.content = json;
        row.streaming = false;
        row.source = None;
        return;
    }
    if let Some(index) = turn.output.iter().rposition(|row| row.kind == "structured") {
        let mut row = turn.output.remove(index);
        row.kind = "structured_ready".to_string();
        row.content = json;
        row.streaming = false;
        row.source = None;
        turn.output.push(row);
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
    use crate::{
        LlmObservationSource, LlmOutputItemView, LlmTranscriptTurnStatus, LlmTranscriptTurnView,
        LlmUsageView,
    };
    use nuillu_types::{ModuleInstanceId, ReplicaIndex, builtin};

    fn overview_row_for_highlight(
        active: bool,
        activation_ratio: Option<f64>,
    ) -> ModuleOverviewRow {
        ModuleOverviewRow {
            owner: "sensory".to_string(),
            module: "sensory".to_string(),
            replica: 0,
            active,
            forced_disabled: false,
            runtime_status: "Activating".to_string(),
            llm_status: "idle".to_string(),
            llm_streaming: false,
            activation_ratio,
            active_replicas: active.then_some(1),
            tier: Some("Default".to_string()),
            guidance: None,
            bpm: None,
            period_ms: None,
            policy: None,
            throttle: None,
            latest_llm_output: None,
            activation_error_count: 0,
            activation_attempt_count: 0,
            llm_error_count: 0,
            llm_turn_count: 0,
            last_execution_failed: false,
        }
    }

    #[test]
    fn active_replica_highlight_distinguishes_allocation_and_min_replicas() {
        assert_eq!(
            active_replica_highlight(&overview_row_for_highlight(true, Some(0.75))),
            Some(ActiveReplicaHighlight::AllocationDriven)
        );
        assert_eq!(
            active_replica_highlight(&overview_row_for_highlight(true, Some(0.0))),
            Some(ActiveReplicaHighlight::MinReplicaDriven)
        );
        assert_eq!(
            active_replica_highlight(&overview_row_for_highlight(false, Some(0.75))),
            None
        );
    }

    #[test]
    fn failed_row_keeps_active_replica_highlight() {
        let mut row = overview_row_for_highlight(true, Some(0.75));
        row.last_execution_failed = true;

        assert_eq!(
            active_replica_highlight(&row),
            Some(ActiveReplicaHighlight::AllocationDriven)
        );
    }

    #[test]
    fn failed_row_fill_takes_priority_over_llm_streaming_highlight() {
        let mut row = overview_row_for_highlight(true, Some(0.75));
        row.llm_streaming = true;
        assert_eq!(
            overview_row_fill_kind(&row, 0),
            Some(OverviewRowFill::LlmStreaming)
        );

        row.last_execution_failed = true;
        assert_eq!(
            overview_row_fill_kind(&row, 1),
            Some(OverviewRowFill::Failed)
        );
    }

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
                session_key: None,
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
                session_key: None,
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
    fn module_batch_ready_is_copied_to_new_turns() {
        let mut state = ModulesState::default();
        let owner = ModuleInstanceId::new(builtin::sensory(), ReplicaIndex::ZERO);

        apply_runtime_event(
            &mut state,
            &RuntimeEvent::ModuleBatchReady {
                sequence: 0,
                owner: owner.clone(),
                batch_type: "nuillu_sensory::SensoryBatch".to_string(),
                batch_debug: "SensoryBatch { inputs: [] }".to_string(),
            },
        );
        apply_llm_observation(
            &mut state,
            LlmObservationEvent::ModelInput {
                turn_id: "turn-batch".to_string(),
                owner: owner.to_string(),
                module: "sensory".to_string(),
                replica: 0,
                tier: "Default".to_string(),
                source: LlmObservationSource::ModuleTurn,
                session_key: None,
                operation: "text_turn".to_string(),
                items: Vec::new(),
            },
        );

        let module = state
            .modules
            .get(&owner.to_string())
            .expect("module exists");
        assert_eq!(
            module.turns[0].batch,
            Some(ModuleBatchDebugState {
                batch_type: "nuillu_sensory::SensoryBatch".to_string(),
                debug: "SensoryBatch { inputs: [] }".to_string(),
            })
        );
    }

    #[test]
    fn structured_ready_replaces_streaming_structured_row() {
        let mut state = ModulesState::default();
        let owner = ModuleInstanceId::new(builtin::query_memory(), ReplicaIndex::ZERO).to_string();
        apply_llm_observation(
            &mut state,
            LlmObservationEvent::ModelInput {
                turn_id: "turn-3".to_string(),
                owner: owner.clone(),
                module: "query-memory".to_string(),
                replica: 0,
                tier: "Default".to_string(),
                source: LlmObservationSource::ModuleTurn,
                session_key: None,
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
    fn structured_ready_replay_updates_in_place() {
        let mut state = ModulesState::default();
        let owner = ModuleInstanceId::new(builtin::query_memory(), ReplicaIndex::ZERO).to_string();
        apply_llm_observation(
            &mut state,
            LlmObservationEvent::ModelInput {
                turn_id: "turn-structured-replay".to_string(),
                owner: owner.clone(),
                module: "query-memory".to_string(),
                replica: 0,
                tier: "Default".to_string(),
                source: LlmObservationSource::ModuleTurn,
                session_key: None,
                operation: "structured_turn".to_string(),
                items: Vec::new(),
            },
        );
        apply_llm_observation(
            &mut state,
            LlmObservationEvent::StreamDelta {
                turn_id: "turn-structured-replay".to_string(),
                kind: "structured".to_string(),
                delta: "{\"answer\":".to_string(),
            },
        );
        apply_llm_observation(
            &mut state,
            LlmObservationEvent::StructuredReady {
                turn_id: "turn-structured-replay".to_string(),
                json: "{\"answer\":true}".to_string(),
            },
        );
        apply_llm_observation(
            &mut state,
            LlmObservationEvent::StreamDelta {
                turn_id: "turn-structured-replay".to_string(),
                kind: "text".to_string(),
                delta: "later text".to_string(),
            },
        );
        apply_llm_observation(
            &mut state,
            LlmObservationEvent::StructuredReady {
                turn_id: "turn-structured-replay".to_string(),
                json: "{\"answer\":\"updated\"}".to_string(),
            },
        );

        let module = state.modules.get(&owner).expect("module exists");
        assert_eq!(
            module.turns[0].output,
            vec![
                LlmOutputItemState {
                    kind: "structured_ready".to_string(),
                    content: "{\"answer\":\"updated\"}".to_string(),
                    streaming: false,
                    source: None,
                },
                LlmOutputItemState {
                    kind: "text".to_string(),
                    content: "later text".to_string(),
                    streaming: true,
                    source: None,
                },
            ]
        );
    }

    #[test]
    fn tool_call_chunks_append_arguments_only_and_keep_source() {
        let mut state = ModulesState::default();
        let owner = ModuleInstanceId::new(builtin::allocation(), ReplicaIndex::ZERO).to_string();
        apply_llm_observation(
            &mut state,
            LlmObservationEvent::ModelInput {
                turn_id: "turn-tool".to_string(),
                owner: owner.clone(),
                module: "allocation".to_string(),
                replica: 0,
                tier: "Default".to_string(),
                source: LlmObservationSource::ModuleTurn,
                session_key: None,
                operation: "text_turn".to_string(),
                items: Vec::new(),
            },
        );
        apply_llm_observation(
            &mut state,
            LlmObservationEvent::ToolCallChunk {
                turn_id: "turn-tool".to_string(),
                id: "call-1".to_string(),
                name: "leave_allocation_unchanged".to_string(),
                arguments_json_delta: "{\"memo\":".to_string(),
            },
        );
        apply_llm_observation(
            &mut state,
            LlmObservationEvent::ToolCallChunk {
                turn_id: "turn-tool".to_string(),
                id: "call-1".to_string(),
                name: "leave_allocation_unchanged".to_string(),
                arguments_json_delta: "\"The current".to_string(),
            },
        );

        let module = state.modules.get(&owner).expect("module exists");
        assert_eq!(
            module.turns[0].output,
            vec![LlmOutputItemState {
                kind: "tool_call".to_string(),
                content: "{\"memo\":\"The current".to_string(),
                streaming: true,
                source: Some("leave_allocation_unchanged(call-1)".to_string()),
            }]
        );
    }

    #[test]
    fn tool_call_chunks_with_different_sources_do_not_merge() {
        let mut state = ModulesState::default();
        let owner = ModuleInstanceId::new(builtin::allocation(), ReplicaIndex::ZERO).to_string();
        apply_llm_observation(
            &mut state,
            LlmObservationEvent::ModelInput {
                turn_id: "turn-tools".to_string(),
                owner: owner.clone(),
                module: "allocation".to_string(),
                replica: 0,
                tier: "Default".to_string(),
                source: LlmObservationSource::ModuleTurn,
                session_key: None,
                operation: "text_turn".to_string(),
                items: Vec::new(),
            },
        );
        apply_llm_observation(
            &mut state,
            LlmObservationEvent::ToolCallChunk {
                turn_id: "turn-tools".to_string(),
                id: "call-1".to_string(),
                name: "reprioritize_modules".to_string(),
                arguments_json_delta: "{\"memo\":".to_string(),
            },
        );
        apply_llm_observation(
            &mut state,
            LlmObservationEvent::ToolCallChunk {
                turn_id: "turn-tools".to_string(),
                id: "call-2".to_string(),
                name: "leave_allocation_unchanged".to_string(),
                arguments_json_delta: "{}".to_string(),
            },
        );
        apply_llm_observation(
            &mut state,
            LlmObservationEvent::ToolCallChunk {
                turn_id: "turn-tools".to_string(),
                id: "call-1".to_string(),
                name: "reprioritize_modules".to_string(),
                arguments_json_delta: "\"focus\"".to_string(),
            },
        );

        let module = state.modules.get(&owner).expect("module exists");
        assert_eq!(
            module.turns[0].output,
            vec![
                LlmOutputItemState {
                    kind: "tool_call".to_string(),
                    content: "{}".to_string(),
                    streaming: true,
                    source: Some("leave_allocation_unchanged(call-2)".to_string()),
                },
                LlmOutputItemState {
                    kind: "tool_call".to_string(),
                    content: "{\"memo\":\"focus\"".to_string(),
                    streaming: true,
                    source: Some("reprioritize_modules(call-1)".to_string()),
                },
            ]
        );
    }

    #[test]
    fn tool_call_ready_replaces_streaming_tool_call_row() {
        let mut state = ModulesState::default();
        let owner = ModuleInstanceId::new(builtin::allocation(), ReplicaIndex::ZERO).to_string();
        apply_llm_observation(
            &mut state,
            LlmObservationEvent::ModelInput {
                turn_id: "turn-ready".to_string(),
                owner: owner.clone(),
                module: "allocation".to_string(),
                replica: 0,
                tier: "Default".to_string(),
                source: LlmObservationSource::ModuleTurn,
                session_key: None,
                operation: "text_turn".to_string(),
                items: Vec::new(),
            },
        );
        apply_llm_observation(
            &mut state,
            LlmObservationEvent::ToolCallChunk {
                turn_id: "turn-ready".to_string(),
                id: "call-1".to_string(),
                name: "leave_allocation_unchanged".to_string(),
                arguments_json_delta: "{\"memo\":".to_string(),
            },
        );
        apply_llm_observation(
            &mut state,
            LlmObservationEvent::ToolCallReady {
                turn_id: "turn-ready".to_string(),
                id: "call-1".to_string(),
                name: "leave_allocation_unchanged".to_string(),
                arguments_json: "{\"memo\":\"stable\"}".to_string(),
            },
        );

        let module = state.modules.get(&owner).expect("module exists");
        assert_eq!(
            module.turns[0].output,
            vec![LlmOutputItemState {
                kind: "tool_call_ready".to_string(),
                content: "{\"memo\":\"stable\"}".to_string(),
                streaming: false,
                source: Some("leave_allocation_unchanged(call-1)".to_string()),
            }]
        );
    }

    #[test]
    fn latest_llm_output_reports_running_turn_without_output() {
        let mut state = ModulesState::default();
        apply_llm_observation(
            &mut state,
            LlmObservationEvent::ModelInput {
                turn_id: "turn-empty".to_string(),
                owner: "sensory".to_string(),
                module: "sensory".to_string(),
                replica: 0,
                tier: "Cheap".to_string(),
                source: LlmObservationSource::ModuleTurn,
                session_key: None,
                operation: "text_turn".to_string(),
                items: Vec::new(),
            },
        );

        let module = state.modules.get("sensory").expect("module exists");
        assert_eq!(
            latest_llm_output(module),
            Some("running: awaiting output".to_string())
        );
    }

    #[test]
    fn overview_rows_mark_llm_streaming_from_model_input_until_completion() {
        let mut state = ModulesState::default();
        let owner = "sensory".to_string();
        apply_llm_observation(
            &mut state,
            LlmObservationEvent::ModelInput {
                turn_id: "turn-streaming".to_string(),
                owner: owner.clone(),
                module: "sensory".to_string(),
                replica: 0,
                tier: "Cheap".to_string(),
                source: LlmObservationSource::ModuleTurn,
                session_key: None,
                operation: "text_turn".to_string(),
                items: Vec::new(),
            },
        );

        let rows = overview_rows(&state, &BlackboardSnapshot::default());
        assert_eq!(rows.len(), 1);
        assert!(rows[0].llm_streaming);

        apply_llm_observation(
            &mut state,
            LlmObservationEvent::Completed {
                turn_id: "turn-streaming".to_string(),
                request_id: None,
                finish_reason: "stop".to_string(),
                usage: LlmUsageView {
                    input_tokens: 0,
                    output_tokens: 0,
                    total_tokens: 0,
                    cost_micros_usd: 0,
                    cache_creation_tokens: 0,
                    cache_read_tokens: 0,
                },
            },
        );

        let rows = overview_rows(&state, &BlackboardSnapshot::default());
        assert_eq!(rows.len(), 1);
        assert!(!rows[0].llm_streaming);
        assert_eq!(rows[0].owner, owner);
    }

    #[test]
    fn latest_llm_output_keeps_completed_turn_output() {
        let mut state = ModulesState::default();
        apply_llm_observation(
            &mut state,
            LlmObservationEvent::ModelInput {
                turn_id: "turn-done".to_string(),
                owner: "sensory".to_string(),
                module: "sensory".to_string(),
                replica: 0,
                tier: "Cheap".to_string(),
                source: LlmObservationSource::ModuleTurn,
                session_key: None,
                operation: "text_turn".to_string(),
                items: Vec::new(),
            },
        );
        apply_llm_observation(
            &mut state,
            LlmObservationEvent::StreamDelta {
                turn_id: "turn-done".to_string(),
                kind: "text".to_string(),
                delta: "old output".to_string(),
            },
        );
        apply_llm_observation(
            &mut state,
            LlmObservationEvent::Completed {
                turn_id: "turn-done".to_string(),
                request_id: None,
                finish_reason: "stop".to_string(),
                usage: LlmUsageView {
                    input_tokens: 0,
                    output_tokens: 0,
                    total_tokens: 0,
                    cost_micros_usd: 0,
                    cache_creation_tokens: 0,
                    cache_read_tokens: 0,
                },
            },
        );

        let module = state.modules.get("sensory").expect("module exists");
        assert_eq!(
            latest_llm_output(module),
            Some("text: old output".to_string())
        );
    }

    #[test]
    fn latest_llm_output_falls_back_to_previous_output_when_newer_turn_is_empty() {
        let mut state = ModulesState::default();
        apply_llm_observation(
            &mut state,
            LlmObservationEvent::ModelInput {
                turn_id: "turn-stale".to_string(),
                owner: "sensory".to_string(),
                module: "sensory".to_string(),
                replica: 0,
                tier: "Cheap".to_string(),
                source: LlmObservationSource::ModuleTurn,
                session_key: None,
                operation: "text_turn".to_string(),
                items: Vec::new(),
            },
        );
        apply_llm_observation(
            &mut state,
            LlmObservationEvent::StreamDelta {
                turn_id: "turn-stale".to_string(),
                kind: "text".to_string(),
                delta: "stale output".to_string(),
            },
        );
        apply_llm_observation(
            &mut state,
            LlmObservationEvent::ModelInput {
                turn_id: "turn-newer".to_string(),
                owner: "sensory".to_string(),
                module: "sensory".to_string(),
                replica: 0,
                tier: "Cheap".to_string(),
                source: LlmObservationSource::ModuleTurn,
                session_key: None,
                operation: "text_turn".to_string(),
                items: Vec::new(),
            },
        );
        apply_llm_observation(
            &mut state,
            LlmObservationEvent::Completed {
                turn_id: "turn-newer".to_string(),
                request_id: None,
                finish_reason: "stop".to_string(),
                usage: LlmUsageView {
                    input_tokens: 0,
                    output_tokens: 0,
                    total_tokens: 0,
                    cost_micros_usd: 0,
                    cache_creation_tokens: 0,
                    cache_read_tokens: 0,
                },
            },
        );

        let module = state.modules.get("sensory").expect("module exists");
        assert_eq!(
            latest_llm_output(module),
            Some("text: stale output".to_string())
        );
    }

    #[test]
    fn selected_llm_turn_id_prefers_latest_streaming_then_latest() {
        let rows = vec![
            LlmTurnListRow {
                owner: "sensory".to_string(),
                module: "sensory".to_string(),
                turn_id: "turn-1".to_string(),
                session_key: None,
                turn_number: 1,
                label: "sensory.turn 1".to_string(),
                streaming: false,
                failed: false,
            },
            LlmTurnListRow {
                owner: "memory".to_string(),
                module: "memory".to_string(),
                turn_id: "turn-2".to_string(),
                session_key: None,
                turn_number: 1,
                label: "* memory.turn 1".to_string(),
                streaming: true,
                failed: false,
            },
            LlmTurnListRow {
                owner: "allocation".to_string(),
                module: "allocation".to_string(),
                turn_id: "turn-3".to_string(),
                session_key: None,
                turn_number: 1,
                label: "* allocation.turn 1".to_string(),
                streaming: true,
                failed: false,
            },
        ];

        assert_eq!(
            selected_llm_turn_id("turn-1", &rows),
            Some("turn-1".to_string())
        );
        assert_eq!(
            selected_llm_turn_id("missing", &rows),
            Some("turn-3".to_string())
        );

        let completed_rows = rows
            .into_iter()
            .map(|mut row| {
                row.streaming = false;
                row
            })
            .collect::<Vec<_>>();
        assert_eq!(
            selected_llm_turn_id("missing", &completed_rows),
            Some("turn-3".to_string())
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
    fn latest_llm_output_keeps_full_text_until_cell_rendering() {
        let mut state = ModulesState::default();
        let content = format!("{}final-token", "stream ".repeat(120));
        apply_llm_observation(
            &mut state,
            LlmObservationEvent::ModelInput {
                turn_id: "turn-long".to_string(),
                owner: "sensory".to_string(),
                module: "sensory".to_string(),
                replica: 0,
                tier: "Cheap".to_string(),
                source: LlmObservationSource::ModuleTurn,
                session_key: None,
                operation: "text_turn".to_string(),
                items: Vec::new(),
            },
        );
        apply_llm_observation(
            &mut state,
            LlmObservationEvent::StreamDelta {
                turn_id: "turn-long".to_string(),
                kind: "text".to_string(),
                delta: content,
            },
        );

        let module = state.modules.get("sensory").expect("module exists");
        let output = latest_llm_output(module).expect("latest output");
        assert!(output.starts_with("text: stream stream"));
        assert!(output.ends_with("final-token"));
        assert!(!output.contains("..."));
    }

    #[test]
    fn latest_llm_output_follows_most_recent_delta_row() {
        let mut state = ModulesState::default();
        apply_llm_observation(
            &mut state,
            LlmObservationEvent::ModelInput {
                turn_id: "turn-interleaved".to_string(),
                owner: "sensory".to_string(),
                module: "sensory".to_string(),
                replica: 0,
                tier: "Cheap".to_string(),
                source: LlmObservationSource::ModuleTurn,
                session_key: None,
                operation: "text_turn".to_string(),
                items: Vec::new(),
            },
        );
        apply_llm_observation(
            &mut state,
            LlmObservationEvent::StreamDelta {
                turn_id: "turn-interleaved".to_string(),
                kind: "text".to_string(),
                delta: "first text".to_string(),
            },
        );
        apply_llm_observation(
            &mut state,
            LlmObservationEvent::ToolCallChunk {
                turn_id: "turn-interleaved".to_string(),
                id: "call-1".to_string(),
                name: "lookup".to_string(),
                arguments_json_delta: "{\"query\":\"old\"}".to_string(),
            },
        );
        apply_llm_observation(
            &mut state,
            LlmObservationEvent::StreamDelta {
                turn_id: "turn-interleaved".to_string(),
                kind: "text".to_string(),
                delta: " newest-tail".to_string(),
            },
        );

        let module = state.modules.get("sensory").expect("module exists");
        assert_eq!(
            latest_llm_output(module),
            Some("text: first text newest-tail".to_string())
        );
    }

    #[test]
    fn latest_output_cell_text_elides_only_the_prefix() {
        let output =
            latest_output_cell_text_with_width("abcdefghijklmnopqrstuvwxyz", 12.0, |text| {
                text.chars().count() as f32
            });

        assert_eq!(output, "opqrstuvwxyz");
        assert!(!output.starts_with("..."));
        assert!(!output.ends_with("..."));
    }

    #[test]
    fn latest_output_cell_text_keeps_one_char_for_tiny_width() {
        let output = latest_output_cell_text_with_width("abc", 0.0, |text| {
            if text.is_empty() { 0.0 } else { 10.0 }
        });

        assert_eq!(output, "c");
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
                period_ms: Some(6667),
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
                period_ms: Some(3333),
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
            owner: "query-memory".to_string(),
            module: "query-memory".to_string(),
            replica: 0,
            ..ModuleState::default()
        };
        let memos = vec![
            memo_view("query-memory", "query-memory", 0, 0, "primary memo"),
            memo_view("query-memory[1]", "query-memory", 1, 0, "replica memo"),
            memo_view("sensory", "sensory", 0, 0, "other memo"),
        ];

        let owners = module_memos(&module, &memos)
            .into_iter()
            .map(|memo| memo.owner.as_str())
            .collect::<Vec<_>>();

        assert_eq!(owners, vec!["query-memory", "query-memory[1]"]);
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
                session_key: None,
                tier: "Default".to_string(),
                model: None,
                request_id: None,
                finish_reason: None,
                usage: None,
                batch: None,
                input: Vec::new(),
                output: Vec::new(),
                status: ModuleSessionStatus::Running,
                error_message: None,
                counted_in_session: false,
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
    fn turn_selector_label_shows_input_plus_output_tokens() {
        let mut turn = LlmTurnState {
            turn_id: "turn-1".to_string(),
            operation: "text_turn".to_string(),
            source: LlmObservationSource::ModuleTurn,
            session_key: Some("main".to_string()),
            tier: "Default".to_string(),
            model: None,
            request_id: None,
            finish_reason: None,
            usage: None,
            batch: None,
            input: Vec::new(),
            output: Vec::new(),
            status: ModuleSessionStatus::Running,
            error_message: None,
            counted_in_session: false,
        };

        assert_eq!(turn_selector_label(1, &turn), "main 1");

        turn.usage = Some(LlmUsageView {
            input_tokens: 2,
            output_tokens: 3,
            total_tokens: 99,
            cost_micros_usd: 0,
            cache_creation_tokens: 0,
            cache_read_tokens: 0,
        });

        assert_eq!(turn_selector_label(1, &turn), "main 1 (5)");
        assert!(turn_selector_hover(&turn).contains("main"));
        assert!(turn_selector_hover(&turn).contains("tokens: 5"));
    }

    #[test]
    fn llm_turn_rows_follow_first_seen_order_and_mark_streaming() {
        let mut state = ModulesState::default();
        apply_llm_observation(
            &mut state,
            LlmObservationEvent::ModelInput {
                turn_id: "sensory-turn".to_string(),
                owner: "sensory".to_string(),
                module: "sensory".to_string(),
                replica: 0,
                tier: "Default".to_string(),
                source: LlmObservationSource::ModuleTurn,
                session_key: None,
                operation: "text_turn".to_string(),
                items: Vec::new(),
            },
        );
        apply_llm_observation(
            &mut state,
            LlmObservationEvent::Completed {
                turn_id: "sensory-turn".to_string(),
                request_id: None,
                finish_reason: "stop".to_string(),
                usage: LlmUsageView {
                    input_tokens: 0,
                    output_tokens: 0,
                    total_tokens: 0,
                    cost_micros_usd: 0,
                    cache_creation_tokens: 0,
                    cache_read_tokens: 0,
                },
            },
        );
        apply_llm_observation(
            &mut state,
            LlmObservationEvent::ModelInput {
                turn_id: "memory-turn".to_string(),
                owner: "memory".to_string(),
                module: "memory".to_string(),
                replica: 0,
                tier: "Default".to_string(),
                source: LlmObservationSource::ModuleTurn,
                session_key: None,
                operation: "text_turn".to_string(),
                items: Vec::new(),
            },
        );

        let rows = llm_turn_rows(&state, |_| true);

        assert_eq!(
            rows.iter()
                .map(|row| row.turn_id.as_str())
                .collect::<Vec<_>>(),
            vec!["sensory-turn", "memory-turn"]
        );
        assert_eq!(rows[0].label, "sensory.turn 1");
        assert!(!rows[0].streaming);
        assert_eq!(rows[1].label, "* memory.turn 1");
        assert!(rows[1].streaming);
    }

    #[test]
    fn llm_turn_labels_count_by_session_key() {
        let mut state = ModulesState::default();
        for (turn_id, session_key) in [
            ("speak-planning-1", "planning"),
            ("speak-generation-1", "generation"),
            ("speak-planning-2", "planning"),
            ("speak-generation-2", "generation"),
        ] {
            apply_llm_observation(
                &mut state,
                LlmObservationEvent::ModelInput {
                    turn_id: turn_id.to_string(),
                    owner: "speak".to_string(),
                    module: "speak".to_string(),
                    replica: 0,
                    tier: "Default".to_string(),
                    source: LlmObservationSource::ModuleTurn,
                    session_key: Some(session_key.to_string()),
                    operation: "text_turn".to_string(),
                    items: Vec::new(),
                },
            );
        }

        let module = state.modules.get("speak").expect("module exists");
        let module_labels = module
            .turns
            .iter()
            .enumerate()
            .map(|(index, turn)| turn_selector_label(session_turn_number(module, index), turn))
            .collect::<Vec<_>>();
        assert_eq!(
            module_labels,
            vec!["planning 1", "generation 1", "planning 2", "generation 2"]
        );

        let rows = llm_turn_rows(&state, |_| true);
        assert_eq!(
            rows.iter()
                .map(|row| row.label.as_str())
                .collect::<Vec<_>>(),
            vec![
                "* speak.planning 1",
                "* speak.generation 1",
                "* speak.planning 2",
                "* speak.generation 2"
            ]
        );
        assert_eq!(
            rows.iter().map(|row| row.turn_number).collect::<Vec<_>>(),
            vec![1, 1, 2, 2]
        );
    }

    #[test]
    fn failed_llm_observation_marks_turn_failed_and_keeps_it_listed() {
        let mut state = ModulesState::default();
        apply_llm_observation(
            &mut state,
            LlmObservationEvent::ModelInput {
                turn_id: "predict-turn".to_string(),
                owner: "predict".to_string(),
                module: "predict".to_string(),
                replica: 0,
                tier: "Default".to_string(),
                source: LlmObservationSource::ModuleTurn,
                session_key: None,
                operation: "text_turn".to_string(),
                items: Vec::new(),
            },
        );
        apply_llm_observation(
            &mut state,
            LlmObservationEvent::StreamDelta {
                turn_id: "predict-turn".to_string(),
                kind: "text".to_string(),
                delta: "partial output".to_string(),
            },
        );
        apply_llm_observation(
            &mut state,
            LlmObservationEvent::Failed {
                turn_id: "predict-turn".to_string(),
                message: "request timed out".to_string(),
            },
        );

        let module = state.modules.get("predict").expect("module exists");
        let turn = module.turns.first().expect("turn exists");
        assert_eq!(turn.status, ModuleSessionStatus::Failed);
        assert_eq!(turn.error_message.as_deref(), Some("request timed out"));
        assert_eq!(turn.output.len(), 1);
        assert_eq!(module.status, ModuleSessionStatus::Failed);

        let rows = llm_turn_rows(&state, |_| true);
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].turn_id, "predict-turn");
        assert!(rows[0].failed);
        assert!(!rows[0].streaming);
        assert_eq!(
            module
                .turns
                .iter()
                .filter(|turn| turn.status == ModuleSessionStatus::Failed)
                .count(),
            1
        );
    }

    #[test]
    fn model_input_events_count_live_llm_turns_by_module_and_session() {
        let mut state = ModulesState::default();
        apply_llm_observation(
            &mut state,
            LlmObservationEvent::ModelInput {
                turn_id: "predict-turn".to_string(),
                owner: "predict".to_string(),
                module: "predict".to_string(),
                replica: 0,
                tier: "Default".to_string(),
                source: LlmObservationSource::ModuleTurn,
                session_key: None,
                operation: "text_turn".to_string(),
                items: Vec::new(),
            },
        );
        apply_llm_observation(
            &mut state,
            LlmObservationEvent::ModelInput {
                turn_id: "predict-turn".to_string(),
                owner: "predict".to_string(),
                module: "predict".to_string(),
                replica: 0,
                tier: "Default".to_string(),
                source: LlmObservationSource::ModuleTurn,
                session_key: None,
                operation: "text_turn".to_string(),
                items: Vec::new(),
            },
        );
        apply_llm_observation(
            &mut state,
            LlmObservationEvent::StreamStarted {
                turn_id: "predict-turn".to_string(),
                owner: "predict".to_string(),
                module: "predict".to_string(),
                replica: 0,
                tier: "Default".to_string(),
                source: LlmObservationSource::ModuleTurn,
                session_key: None,
                operation: "text_turn".to_string(),
                request_id: Some("req-1".to_string()),
                model: "model-a".to_string(),
            },
        );
        apply_llm_observation(
            &mut state,
            LlmObservationEvent::ModelInput {
                turn_id: "predict-turn".to_string(),
                owner: "predict".to_string(),
                module: "predict".to_string(),
                replica: 0,
                tier: "Default".to_string(),
                source: LlmObservationSource::ModuleTurn,
                session_key: None,
                operation: "text_turn".to_string(),
                items: Vec::new(),
            },
        );
        apply_llm_observation(
            &mut state,
            LlmObservationEvent::ModelInput {
                turn_id: "memory-turn".to_string(),
                owner: "memory".to_string(),
                module: "memory".to_string(),
                replica: 0,
                tier: "Default".to_string(),
                source: LlmObservationSource::ModuleTurn,
                session_key: None,
                operation: "text_turn".to_string(),
                items: Vec::new(),
            },
        );
        apply_llm_observation(
            &mut state,
            LlmObservationEvent::StreamStarted {
                turn_id: "sensory-turn".to_string(),
                owner: "sensory".to_string(),
                module: "sensory".to_string(),
                replica: 0,
                tier: "Default".to_string(),
                source: LlmObservationSource::ModuleTurn,
                session_key: None,
                operation: "text_turn".to_string(),
                request_id: Some("req-2".to_string()),
                model: "model-a".to_string(),
            },
        );

        assert_eq!(state.session_live_llm_turn_count(), 2);

        apply_runtime_event(
            &mut state,
            &RuntimeEvent::LlmAccessed {
                sequence: 0,
                call: 0,
                owner: ModuleInstanceId::new(builtin::memory(), ReplicaIndex::ZERO),
                tier: nuillu_types::ModelTier::Default,
            },
        );
        assert_eq!(state.session_live_llm_turn_count(), 2);

        let snapshot = BlackboardSnapshot {
            allocation: vec![AllocationView {
                module: "sensory".to_string(),
                activation_ratio: 0.25,
                active_replicas: 1,
                bpm: None,
                period_ms: None,
                tier: "Default".to_string(),
                guidance: String::new(),
            }],
            ..BlackboardSnapshot::default()
        };
        let rows = overview_rows(&state, &snapshot);
        assert_eq!(
            rows.iter()
                .find(|row| row.owner == "predict")
                .map(|row| (row.llm_error_count, row.llm_turn_count)),
            Some((0, 1))
        );
        assert_eq!(
            rows.iter()
                .find(|row| row.owner == "memory")
                .map(|row| (row.llm_error_count, row.llm_turn_count)),
            Some((0, 1))
        );
        assert_eq!(
            rows.iter()
                .find(|row| row.owner == "sensory")
                .map(|row| (row.llm_error_count, row.llm_turn_count)),
            Some((0, 0))
        );
    }

    #[test]
    fn failed_live_llm_turns_count_as_llm_errors() {
        let mut state = ModulesState::default();
        apply_llm_observation(
            &mut state,
            LlmObservationEvent::ModelInput {
                turn_id: "predict-turn".to_string(),
                owner: "predict".to_string(),
                module: "predict".to_string(),
                replica: 0,
                tier: "Default".to_string(),
                source: LlmObservationSource::ModuleTurn,
                session_key: None,
                operation: "text_turn".to_string(),
                items: Vec::new(),
            },
        );
        apply_llm_observation(
            &mut state,
            LlmObservationEvent::Failed {
                turn_id: "predict-turn".to_string(),
                message: "request timed out".to_string(),
            },
        );

        let rows = overview_rows(&state, &BlackboardSnapshot::default());
        assert_eq!(
            rows.iter()
                .find(|row| row.owner == "predict")
                .map(|row| (row.llm_error_count, row.llm_turn_count)),
            Some((1, 1))
        );
    }

    #[test]
    fn activation_events_count_errors_and_attempts() {
        let mut state = ModulesState::default();
        let owner = ModuleInstanceId::new(builtin::predict(), ReplicaIndex::ZERO);

        apply_runtime_event(
            &mut state,
            &RuntimeEvent::ModuleActivationAttemptFailed {
                sequence: 0,
                owner: owner.clone(),
                activation_attempt: 1,
                max_attempts: 3,
                message: "temporary failure".to_string(),
            },
        );
        apply_runtime_event(
            &mut state,
            &RuntimeEvent::ModuleActivationCompleted {
                sequence: 1,
                owner: owner.clone(),
                duration: Duration::from_millis(42),
                succeeded: true,
            },
        );

        let rows = overview_rows(&state, &BlackboardSnapshot::default());
        assert_eq!(
            rows.iter()
                .find(|row| row.owner == owner.to_string())
                .map(|row| (row.activation_error_count, row.activation_attempt_count)),
            Some((1, 2))
        );

        let mut state = ModulesState::default();
        apply_runtime_event(
            &mut state,
            &RuntimeEvent::ModuleActivationAttemptFailed {
                sequence: 0,
                owner: owner.clone(),
                activation_attempt: 1,
                max_attempts: 2,
                message: "first failure".to_string(),
            },
        );
        apply_runtime_event(
            &mut state,
            &RuntimeEvent::ModuleActivationAttemptFailed {
                sequence: 1,
                owner: owner.clone(),
                activation_attempt: 2,
                max_attempts: 2,
                message: "second failure".to_string(),
            },
        );
        apply_runtime_event(
            &mut state,
            &RuntimeEvent::ModuleActivationCompleted {
                sequence: 2,
                owner: owner.clone(),
                duration: Duration::from_millis(42),
                succeeded: false,
            },
        );

        let rows = overview_rows(&state, &BlackboardSnapshot::default());
        assert_eq!(
            rows.iter()
                .find(|row| row.owner == owner.to_string())
                .map(|row| (row.activation_error_count, row.activation_attempt_count)),
            Some((2, 2))
        );
    }

    #[test]
    fn module_task_failed_does_not_count_as_activation_error() {
        let mut state = ModulesState::default();
        let owner = ModuleInstanceId::new(builtin::predict(), ReplicaIndex::ZERO);

        apply_runtime_event(
            &mut state,
            &RuntimeEvent::ModuleTaskFailed {
                sequence: 0,
                owner: owner.clone(),
                phase: "activate".to_string(),
                message: "task failed".to_string(),
            },
        );

        let rows = overview_rows(&state, &BlackboardSnapshot::default());
        assert_eq!(
            rows.iter()
                .find(|row| row.owner == owner.to_string())
                .map(|row| (row.activation_error_count, row.activation_attempt_count)),
            Some((0, 0))
        );
    }

    #[test]
    fn transcript_snapshot_failed_turn_does_not_mark_module_currently_failed() {
        let mut state = ModulesState::default();

        apply_llm_transcript_snapshot(
            &mut state,
            vec![LlmTranscriptTurnView {
                turn_id: "persisted-turn".to_string(),
                owner: "predict".to_string(),
                module: "predict".to_string(),
                replica: 0,
                tier: "Default".to_string(),
                source: LlmObservationSource::ModuleTurn,
                session_key: Some("session".to_string()),
                operation: "text_turn".to_string(),
                input: Vec::new(),
                output: vec![LlmOutputItemView {
                    kind: "text".to_string(),
                    content: "partial output".to_string(),
                    source: None,
                }],
                request_id: Some("req-1".to_string()),
                model: Some("model-a".to_string()),
                finish_reason: Some("Stop".to_string()),
                usage: Some(LlmUsageView {
                    input_tokens: 3,
                    output_tokens: 5,
                    total_tokens: 8,
                    cost_micros_usd: 0,
                    cache_creation_tokens: 0,
                    cache_read_tokens: 0,
                }),
                status: LlmTranscriptTurnStatus::Failed,
                error_message: Some("request timed out".to_string()),
            }],
        );

        let module = state.modules.get("predict").expect("module exists");
        assert_eq!(module.status, ModuleSessionStatus::Idle);
        assert_eq!(module.runtime_status, None);
        assert_eq!(module.activation_error_count, 0);
        assert_eq!(module.activation_attempt_count, 0);
        assert_eq!(state.session_live_llm_turn_count(), 0);
        assert!(!module.last_execution_failed);
        assert_eq!(module.last_tier, None);

        let turn = module.turns.first().expect("turn exists");
        assert_eq!(turn.session_key.as_deref(), Some("session"));
        assert_eq!(turn.status, ModuleSessionStatus::Failed);
        assert_eq!(turn.error_message.as_deref(), Some("request timed out"));
        assert_eq!(turn.output.len(), 1);

        let turn_rows = llm_turn_rows(&state, |_| true);
        assert_eq!(turn_rows.len(), 1);
        assert_eq!(turn_rows[0].label, "predict.session 1");
        assert!(turn_rows[0].failed);

        let overview = overview_rows(&state, &BlackboardSnapshot::default());
        assert_eq!(overview.len(), 1);
        assert_eq!(overview[0].llm_status, "idle");
        assert_eq!(overview[0].activation_error_count, 0);
        assert_eq!(overview[0].activation_attempt_count, 0);
        assert_eq!(overview[0].llm_error_count, 0);
        assert_eq!(overview[0].llm_turn_count, 0);
        assert!(!overview[0].last_execution_failed);
        assert_eq!(overview_row_fill_kind(&overview[0], 0), None);
    }

    #[test]
    fn failed_turn_error_banner_does_not_grow_llm_turns_window_each_frame() {
        let mut state = ModulesState::default();
        apply_llm_observation(
            &mut state,
            LlmObservationEvent::ModelInput {
                turn_id: "predict-turn".to_string(),
                owner: "predict".to_string(),
                module: "predict".to_string(),
                replica: 0,
                tier: "Default".to_string(),
                source: LlmObservationSource::ModuleTurn,
                session_key: None,
                operation: "text_turn".to_string(),
                items: Vec::new(),
            },
        );
        apply_llm_observation(
            &mut state,
            LlmObservationEvent::Failed {
                turn_id: "predict-turn".to_string(),
                message: format!("request failed: {}", "x".repeat(1600)),
            },
        );

        let ctx = egui::Context::default();
        let mut filter = ModuleFilterState::default();
        let modules = state.module_names();
        let mut widths = Vec::new();

        for frame in 0..6 {
            let input = egui::RawInput {
                screen_rect: Some(egui::Rect::from_min_size(
                    egui::Pos2::ZERO,
                    egui::vec2(900.0, 700.0),
                )),
                time: Some(frame as f64 / 60.0),
                ..egui::RawInput::default()
            };
            let _ = ctx.run_ui(input, |ui| {
                let response = egui::Window::new("LLM Turns")
                    .id(egui::Id::new("failed-turn-growth-test"))
                    .default_size(egui::vec2(520.0, 360.0))
                    .show(ui.ctx(), |ui| {
                        render_llm_turns(ui, &state, &mut filter, &modules);
                    })
                    .expect("window is open");
                widths.push(response.response.rect.width());
            });
        }

        let settled_widths = &widths[2..];
        let min = settled_widths.iter().copied().fold(f32::INFINITY, f32::min);
        let max = settled_widths
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        assert!(
            max - min <= 1.0,
            "window width should settle instead of growing: {widths:?}"
        );
    }

    #[test]
    fn llm_turn_rows_filter_by_module_and_selection_falls_back() {
        let mut state = ModulesState::default();
        apply_llm_observation(
            &mut state,
            LlmObservationEvent::ModelInput {
                turn_id: "sensory-turn".to_string(),
                owner: "sensory".to_string(),
                module: "sensory".to_string(),
                replica: 0,
                tier: "Default".to_string(),
                source: LlmObservationSource::ModuleTurn,
                session_key: None,
                operation: "text_turn".to_string(),
                items: Vec::new(),
            },
        );
        apply_llm_observation(
            &mut state,
            LlmObservationEvent::ModelInput {
                turn_id: "memory-turn".to_string(),
                owner: "memory".to_string(),
                module: "memory".to_string(),
                replica: 0,
                tier: "Default".to_string(),
                source: LlmObservationSource::ModuleTurn,
                session_key: None,
                operation: "text_turn".to_string(),
                items: Vec::new(),
            },
        );

        let rows = llm_turn_rows(&state, |module| module == "memory");

        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].turn_id, "memory-turn");
        assert_eq!(
            selected_llm_turn_id("sensory-turn", &rows),
            Some("memory-turn".to_string())
        );
        assert_eq!(
            selected_llm_turn_id("memory-turn", &rows),
            Some("memory-turn".to_string())
        );
        assert_eq!(selected_llm_turn_id("memory-turn", &[]), None);
    }

    #[test]
    fn runtime_events_record_throttle_summaries_without_changing_llm_status() {
        let mut state = ModulesState::default();
        let owner = ModuleInstanceId::new(builtin::query_memory(), ReplicaIndex::ZERO);

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
    fn runtime_error_marks_module_failed() {
        let mut state = ModulesState::default();
        let owner = ModuleInstanceId::new(builtin::predict(), ReplicaIndex::ZERO);

        apply_runtime_event(
            &mut state,
            &RuntimeEvent::ModuleTaskFailed {
                sequence: 2,
                owner: owner.clone(),
                phase: "activate".to_string(),
                message: "llm request failed".to_string(),
            },
        );

        let module = state
            .modules
            .get(&owner.to_string())
            .expect("module exists");
        assert_eq!(module.status, ModuleSessionStatus::Failed);
        assert_eq!(
            module.runtime_status.as_deref(),
            Some("Failed activate: llm request failed")
        );
        assert_eq!(module.activation_error_count, 0);
        assert_eq!(module.activation_attempt_count, 0);
        assert!(module.last_execution_failed);

        let rows = overview_rows(&state, &BlackboardSnapshot::default());
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].activation_error_count, 0);
        assert_eq!(rows[0].activation_attempt_count, 0);
        assert_eq!(rows[0].llm_error_count, 0);
        assert_eq!(rows[0].llm_turn_count, 0);
        assert!(rows[0].last_execution_failed);
    }

    #[test]
    fn successful_activation_clears_failed_highlight_without_counting_task_failure() {
        let mut state = ModulesState::default();
        let owner = ModuleInstanceId::new(builtin::predict(), ReplicaIndex::ZERO);

        apply_runtime_event(
            &mut state,
            &RuntimeEvent::ModuleTaskFailed {
                sequence: 2,
                owner: owner.clone(),
                phase: "activate".to_string(),
                message: "llm request failed".to_string(),
            },
        );
        apply_runtime_event(
            &mut state,
            &RuntimeEvent::ModuleActivationCompleted {
                sequence: 3,
                owner: owner.clone(),
                duration: Duration::from_millis(42),
                succeeded: true,
            },
        );

        let module = state
            .modules
            .get(&owner.to_string())
            .expect("module exists");
        assert_eq!(module.activation_error_count, 0);
        assert_eq!(module.activation_attempt_count, 1);
        assert!(!module.last_execution_failed);
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
                session_key: None,
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
                period_ms: Some(4800),
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
                llm_streaming: true,
                activation_ratio: Some(0.75),
                active_replicas: Some(1),
                tier: Some("Premium".to_string()),
                guidance: Some("inspect recent input".to_string()),
                bpm: Some(12.5),
                period_ms: Some(4800),
                policy: None,
                throttle: Some("500ms".to_string()),
                latest_llm_output: Some("text: filtered observation".to_string()),
                activation_error_count: 0,
                activation_attempt_count: 0,
                llm_error_count: 0,
                llm_turn_count: 1,
                last_execution_failed: false,
            }]
        );
    }

    #[test]
    fn replica_label_uses_one_based_index_and_active_total() {
        let row = ModuleOverviewRow {
            owner: "query-memory[1]".to_string(),
            module: "query-memory".to_string(),
            replica: 1,
            active: true,
            forced_disabled: false,
            runtime_status: "Activating".to_string(),
            llm_status: "running".to_string(),
            llm_streaming: false,
            activation_ratio: Some(1.0),
            active_replicas: Some(2),
            tier: Some("Default".to_string()),
            guidance: None,
            bpm: None,
            period_ms: None,
            policy: None,
            throttle: None,
            latest_llm_output: None,
            activation_error_count: 0,
            activation_attempt_count: 0,
            llm_error_count: 0,
            llm_turn_count: 0,
            last_execution_failed: false,
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
