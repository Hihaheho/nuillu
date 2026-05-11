pub mod blackboard;
pub mod chat;
pub mod cognition;
pub mod memories;
pub mod memos;
pub mod modules;
pub mod text;
pub mod window;

pub use eframe;
pub use egui;
pub use egui_hooks;

use std::collections::{BTreeMap, VecDeque};
use std::sync::mpsc::{Receiver, Sender};

use chrono::{DateTime, Utc};
use nuillu_module::{RuntimeEvent, SensoryInput};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct VisualizerTabId(pub String);

impl VisualizerTabId {
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VisualizerEvent {
    OpenTab {
        tab_id: VisualizerTabId,
        title: String,
    },
    SetTabStatus {
        tab_id: VisualizerTabId,
        status: TabStatus,
    },
    Log {
        tab_id: VisualizerTabId,
        message: String,
    },
    SensoryInput {
        tab_id: VisualizerTabId,
        input: SensoryInput,
    },
    UtteranceDelta {
        tab_id: VisualizerTabId,
        utterance: UtteranceDeltaView,
    },
    UtteranceCompleted {
        tab_id: VisualizerTabId,
        utterance: UtteranceView,
    },
    RuntimeEvent {
        tab_id: VisualizerTabId,
        event: RuntimeEvent,
    },
    LlmObserved {
        tab_id: VisualizerTabId,
        event: LlmObservationEvent,
    },
    BlackboardSnapshot {
        tab_id: VisualizerTabId,
        snapshot: BlackboardSnapshot,
    },
    MemoryPage {
        tab_id: VisualizerTabId,
        page: MemoryPage,
    },
    MemoryQueryResult {
        tab_id: VisualizerTabId,
        query: String,
        records: Vec<MemoryRecordView>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VisualizerCommand {
    StartSuite,
    SendSensoryInput {
        tab_id: VisualizerTabId,
        input: ChatInput,
    },
    QueryMemory {
        tab_id: VisualizerTabId,
        query: String,
        limit: usize,
    },
    ListMemories {
        tab_id: VisualizerTabId,
        page: usize,
        per_page: usize,
    },
    Shutdown,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "event", rename_all = "snake_case")]
pub enum LlmObservationEvent {
    ModelInput {
        turn_id: String,
        owner: String,
        module: String,
        replica: u8,
        tier: String,
        source: LlmObservationSource,
        operation: String,
        items: Vec<LlmInputItemView>,
    },
    StreamStarted {
        turn_id: String,
        owner: String,
        module: String,
        replica: u8,
        tier: String,
        source: LlmObservationSource,
        operation: String,
        request_id: Option<String>,
        model: String,
    },
    StreamDelta {
        turn_id: String,
        kind: String,
        delta: String,
    },
    ToolCallChunk {
        turn_id: String,
        id: String,
        name: String,
        arguments_json_delta: String,
    },
    ToolCallReady {
        turn_id: String,
        id: String,
        name: String,
        arguments_json: String,
    },
    StructuredReady {
        turn_id: String,
        json: String,
    },
    Completed {
        turn_id: String,
        request_id: Option<String>,
        finish_reason: String,
        usage: LlmUsageView,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LlmObservationSource {
    ModuleTurn,
    SessionCompaction,
}

impl LlmObservationSource {
    pub fn label(self) -> &'static str {
        match self {
            Self::ModuleTurn => "module",
            Self::SessionCompaction => "compaction",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LlmInputItemView {
    pub role: String,
    pub kind: String,
    pub content: String,
    pub ephemeral: bool,
    pub source: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct LlmUsageView {
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub total_tokens: u64,
    pub cost_micros_usd: u64,
    pub cache_creation_tokens: u64,
    pub cache_read_tokens: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TabStatus {
    Running,
    Passed,
    Failed,
    Invalid,
    Stopped,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ChatInputKind {
    Heard,
    Seen,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatInput {
    pub kind: ChatInputKind,
    pub direction: Option<String>,
    pub content: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UtteranceDeltaView {
    pub sender: String,
    pub target: String,
    pub generation_id: u64,
    pub sequence: u32,
    pub delta: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UtteranceView {
    pub sender: String,
    pub target: String,
    pub text: String,
    pub emitted_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BlackboardSnapshot {
    pub module_statuses: Vec<ModuleStatusView>,
    pub allocation: Vec<AllocationView>,
    pub memos: Vec<MemoView>,
    pub cognition_logs: Vec<CognitionLogView>,
    pub utterance_progresses: Vec<UtteranceProgressView>,
    pub memory_metadata: Vec<MemoryMetadataView>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModuleStatusView {
    pub owner: String,
    pub module: String,
    pub replica: u8,
    pub status: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AllocationView {
    pub module: String,
    pub activation_ratio: f64,
    pub active_replicas: u8,
    #[serde(default)]
    pub bpm: Option<f64>,
    #[serde(default)]
    pub cooldown_ms: Option<u64>,
    pub tier: String,
    pub guidance: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoView {
    pub owner: String,
    pub module: String,
    pub replica: u8,
    pub index: u64,
    pub written_at: DateTime<Utc>,
    pub content: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CognitionLogView {
    pub source: String,
    pub entries: Vec<CognitionEntryView>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CognitionEntryView {
    pub at: DateTime<Utc>,
    pub text: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UtteranceProgressView {
    pub owner: String,
    pub target: String,
    pub generation_id: u64,
    pub sequence: u32,
    pub state: String,
    pub partial_utterance: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryMetadataView {
    pub index: String,
    pub rank: String,
    pub occurred_at: Option<DateTime<Utc>>,
    pub last_accessed: DateTime<Utc>,
    pub access_count: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryRecordView {
    pub index: String,
    pub rank: String,
    pub occurred_at: Option<DateTime<Utc>>,
    pub content: String,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MemoryPage {
    pub page: usize,
    pub per_page: usize,
    pub total: usize,
    pub records: Vec<MemoryRecordView>,
}

pub struct VisualizerChannels {
    pub events: Receiver<VisualizerEvent>,
    pub commands: Sender<VisualizerCommand>,
    pub start_suite_from_ui: bool,
}

pub struct VisualizerApp {
    events: Receiver<VisualizerEvent>,
    commands: Sender<VisualizerCommand>,
    start_suite_from_ui: bool,
    state: VisualizerState,
}

impl VisualizerApp {
    pub fn new(_cc: &eframe::CreationContext<'_>, channels: VisualizerChannels) -> Self {
        Self {
            events: channels.events,
            commands: channels.commands,
            start_suite_from_ui: channels.start_suite_from_ui,
            state: VisualizerState::default(),
        }
    }

    fn drain_events(&mut self) {
        while let Ok(event) = self.events.try_recv() {
            self.state.apply(event);
        }
    }
}

impl eframe::App for VisualizerApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        self.drain_events();
        ui.ctx()
            .request_repaint_after(std::time::Duration::from_millis(100));

        egui::Panel::top("nuillu-visualizer-tabs").show_inside(ui, |ui| {
            ui.horizontal_wrapped(|ui| {
                for tab in self.state.tabs.values() {
                    let selected = self.state.selected.as_ref() == Some(&tab.id);
                    let label = format!("{} {}", tab.status.icon(), tab.title);
                    if ui.selectable_label(selected, label).clicked() {
                        self.state.selected = Some(tab.id.clone());
                    }
                }
                let view_tab_id = self
                    .state
                    .selected
                    .clone()
                    .or_else(|| self.state.tabs.keys().next().cloned());
                if let Some(tab_id) = view_tab_id {
                    if let Some(tab) = self.state.tabs.get_mut(&tab_id) {
                        tab.view_menu(ui);
                    }
                } else {
                    ui.menu_button("View", |ui| {
                        ui.label("No runtime windows yet.");
                    });
                }
                if self.start_suite_from_ui {
                    let label = if self.state.suite_start_requested {
                        "Eval started"
                    } else {
                        "Start Eval"
                    };
                    if ui
                        .add_enabled(!self.state.suite_start_requested, egui::Button::new(label))
                        .clicked()
                    {
                        self.state.suite_start_requested = true;
                        let _ = self.commands.send(VisualizerCommand::StartSuite);
                    }
                }
                if ui.button("Shutdown").clicked() {
                    let _ = self.commands.send(VisualizerCommand::Shutdown);
                }
            });
        });

        egui::CentralPanel::default().show_inside(ui, |ui| {
            let selected = self
                .state
                .selected
                .clone()
                .or_else(|| self.state.tabs.keys().next().cloned());
            if let Some(tab_id) = selected {
                self.state.selected = Some(tab_id.clone());
                if let Some(tab) = self.state.tabs.get_mut(&tab_id) {
                    ui.push_id(tab_id.as_str(), |ui| {
                        tab.ui(ui, &self.commands);
                    });
                }
            } else {
                ui.centered_and_justified(|ui| {
                    ui.label("No runtime tabs yet.");
                });
            }
        });
    }

    fn auto_save_interval(&self) -> std::time::Duration {
        std::time::Duration::from_millis(1500)
    }
}

#[derive(Default)]
pub struct VisualizerState {
    tabs: BTreeMap<VisualizerTabId, RuntimeTab>,
    selected: Option<VisualizerTabId>,
    suite_start_requested: bool,
}

impl VisualizerState {
    pub fn apply(&mut self, event: VisualizerEvent) {
        match event {
            VisualizerEvent::OpenTab { tab_id, title } => {
                let tab = self
                    .tabs
                    .entry(tab_id.clone())
                    .or_insert_with(|| RuntimeTab::new(tab_id.clone(), title.clone()));
                tab.title = title;
                tab.status = TabStatus::Running;
                self.selected.get_or_insert(tab_id);
            }
            VisualizerEvent::SetTabStatus { tab_id, status } => {
                self.tab_mut(tab_id).status = status;
            }
            VisualizerEvent::Log { tab_id, message } => {
                self.tab_mut(tab_id).push_log(message);
            }
            VisualizerEvent::SensoryInput { tab_id, input } => {
                self.tab_mut(tab_id).chat.push_sensory_input(input);
            }
            VisualizerEvent::UtteranceDelta { tab_id, utterance } => {
                self.tab_mut(tab_id).chat.push_utterance_delta(utterance);
            }
            VisualizerEvent::UtteranceCompleted { tab_id, utterance } => {
                self.tab_mut(tab_id)
                    .chat
                    .push_utterance_completed(utterance);
            }
            VisualizerEvent::RuntimeEvent { tab_id, event } => {
                let tab = self.tab_mut(tab_id);
                modules::apply_runtime_event(&mut tab.modules, &event);
                tab.runtime_events.push_back(event);
                if tab.runtime_events.len() > 256 {
                    tab.runtime_events.pop_front();
                }
            }
            VisualizerEvent::LlmObserved { tab_id, event } => {
                modules::apply_llm_observation(&mut self.tab_mut(tab_id).modules, event);
            }
            VisualizerEvent::BlackboardSnapshot { tab_id, snapshot } => {
                let tab = self.tab_mut(tab_id);
                modules::apply_blackboard_snapshot(&mut tab.modules, &snapshot);
                tab.blackboard = snapshot;
            }
            VisualizerEvent::MemoryPage { tab_id, page } => {
                let tab = self.tab_mut(tab_id);
                tab.memories.query.clear();
                tab.memories.query_results.clear();
                tab.memories.page = page;
            }
            VisualizerEvent::MemoryQueryResult {
                tab_id,
                query,
                records,
            } => {
                let tab = self.tab_mut(tab_id);
                tab.memories.query = query;
                tab.memories.query_results = records;
            }
        }
    }

    pub fn tabs(&self) -> &BTreeMap<VisualizerTabId, RuntimeTab> {
        &self.tabs
    }

    fn tab_mut(&mut self, tab_id: VisualizerTabId) -> &mut RuntimeTab {
        self.tabs
            .entry(tab_id.clone())
            .or_insert_with(|| RuntimeTab::new(tab_id, "runtime".to_string()))
    }
}

pub struct RuntimeTab {
    id: VisualizerTabId,
    title: String,
    status: TabStatus,
    chat: chat::ChatState,
    blackboard: BlackboardSnapshot,
    memories: memories::MemoriesState,
    modules: modules::ModulesState,
    runtime_events: VecDeque<RuntimeEvent>,
    logs: VecDeque<String>,
    window_open: BTreeMap<String, bool>,
    window_requests: BTreeMap<String, bool>,
}

#[derive(Debug, Clone)]
struct ViewWindowSpec {
    id: String,
    title: String,
}

impl RuntimeTab {
    fn new(id: VisualizerTabId, title: String) -> Self {
        Self {
            id,
            title,
            status: TabStatus::Running,
            chat: chat::ChatState::default(),
            blackboard: BlackboardSnapshot::default(),
            memories: memories::MemoriesState::default(),
            modules: modules::ModulesState::default(),
            runtime_events: VecDeque::new(),
            logs: VecDeque::new(),
            window_open: BTreeMap::new(),
            window_requests: BTreeMap::new(),
        }
    }

    fn ui(&mut self, ui: &mut egui::Ui, commands: &Sender<VisualizerCommand>) {
        ui.horizontal_wrapped(|ui| {
            ui.heading(format!("{} {}", self.status.icon(), self.title));
            ui.label(format!("runtime events: {}", self.runtime_events.len()));
            ui.label(format!("modules: {}", self.modules.iter().count()));
        });

        self.windows_ui(ui, commands);
    }

    fn view_menu(&mut self, ui: &mut egui::Ui) {
        let specs = self.window_specs();
        ui.menu_button("View", |ui| {
            for spec in specs {
                let mut open = self.window_open.get(&spec.id).copied().unwrap_or(true);
                if ui.checkbox(&mut open, &spec.title).changed() {
                    self.window_requests.insert(spec.id, open);
                }
            }
        });
    }

    fn window_specs(&self) -> Vec<ViewWindowSpec> {
        let base = self.id.as_str();
        let mut specs = vec![
            ViewWindowSpec {
                id: format!("{base}:chat"),
                title: format!("💬 Chat - {}", self.title),
            },
            ViewWindowSpec {
                id: format!("{base}:blackboard"),
                title: format!("🧾 Blackboard - {}", self.title),
            },
            ViewWindowSpec {
                id: format!("{base}:memories"),
                title: format!("🧠 Memory - {}", self.title),
            },
            ViewWindowSpec {
                id: format!("{base}:memos"),
                title: format!("📝 Memo - {}", self.title),
            },
            ViewWindowSpec {
                id: format!("{base}:cognition"),
                title: format!("🧩 Cognition Log - {}", self.title),
            },
            ViewWindowSpec {
                id: format!("{base}:logs"),
                title: format!("📜 Logs - {}", self.title),
            },
            ViewWindowSpec {
                id: format!("{base}:modules"),
                title: format!("Modules - {}", self.title),
            },
        ];
        for module in self.modules.iter() {
            specs.push(ViewWindowSpec {
                id: format!("{base}:module:{}", module.owner),
                title: modules::window_title(module),
            });
        }
        specs
    }

    fn windows_ui(&mut self, ui: &mut egui::Ui, commands: &Sender<VisualizerCommand>) {
        let base = self.id.as_str().to_string();
        let mut window_requests = std::mem::take(&mut self.window_requests);

        let chat_id = format!("{base}:chat");
        let chat_title = format!("💬 Chat - {}", self.title);
        let open = window::PersistedWindow::new(&chat_id, &chat_title)
            .open_override(window_requests.remove(&chat_id))
            .default_pos(24.0, 88.0)
            .default_size(520.0, 520.0)
            .show(ui, |ui| chat::ui(ui, &self.id, &mut self.chat, commands));
        self.record_window_open(chat_id, open);

        let blackboard_id = format!("{base}:blackboard");
        let blackboard_title = format!("🧾 Blackboard - {}", self.title);
        let open = window::PersistedWindow::new(&blackboard_id, &blackboard_title)
            .open_override(window_requests.remove(&blackboard_id))
            .default_pos(568.0, 88.0)
            .default_size(640.0, 520.0)
            .show(ui, |ui| blackboard::ui(ui, &self.blackboard));
        self.record_window_open(blackboard_id, open);

        let memories_id = format!("{base}:memories");
        let memories_title = format!("🧠 Memory - {}", self.title);
        let open = window::PersistedWindow::new(&memories_id, &memories_title)
            .open_override(window_requests.remove(&memories_id))
            .default_pos(96.0, 636.0)
            .default_size(720.0, 360.0)
            .show(ui, |ui| {
                memories::ui(ui, &self.id, &mut self.memories, commands)
            });
        self.record_window_open(memories_id, open);

        let memos_id = format!("{base}:memos");
        let memos_title = format!("📝 Memo - {}", self.title);
        let open = window::PersistedWindow::new(&memos_id, &memos_title)
            .open_override(window_requests.remove(&memos_id))
            .default_pos(840.0, 636.0)
            .default_size(520.0, 360.0)
            .show(ui, |ui| memos::ui(ui, &self.blackboard.memos));
        self.record_window_open(memos_id, open);

        let cognition_id = format!("{base}:cognition");
        let cognition_title = format!("🧩 Cognition Log - {}", self.title);
        let open = window::PersistedWindow::new(&cognition_id, &cognition_title)
            .open_override(window_requests.remove(&cognition_id))
            .default_pos(1384.0, 636.0)
            .default_size(560.0, 360.0)
            .show(ui, |ui| cognition::ui(ui, &self.blackboard.cognition_logs));
        self.record_window_open(cognition_id, open);

        let logs_id = format!("{base}:logs");
        let logs_title = format!("📜 Logs - {}", self.title);
        let open = window::PersistedWindow::new(&logs_id, &logs_title)
            .open_override(window_requests.remove(&logs_id))
            .default_pos(24.0, 1020.0)
            .default_size(520.0, 360.0)
            .show(ui, |ui| self.logs_ui(ui));
        self.record_window_open(logs_id, open);

        let modules_id = format!("{base}:modules");
        let modules_title = format!("Modules - {}", self.title);
        let mut requested_module = None;
        let open = window::PersistedWindow::new(&modules_id, &modules_title)
            .open_override(window_requests.remove(&modules_id))
            .default_pos(568.0, 1020.0)
            .default_size(640.0, 360.0)
            .show(ui, |ui| {
                requested_module =
                    modules::render_modules_overview(ui, &self.blackboard, &self.modules);
            });
        self.record_window_open(modules_id, open);

        let module_windows = self
            .modules
            .iter()
            .enumerate()
            .map(|(index, module)| (index, module.owner.clone(), modules::window_title(module)))
            .collect::<Vec<_>>();
        for (index, owner, module_title) in module_windows {
            let module_id = format!("{base}:module:{owner}");
            let x = 1232.0 + (index % 2) as f32 * 440.0;
            let y = 88.0 + (index / 2) as f32 * 380.0;
            let requested = if requested_module.as_deref() == Some(owner.as_str()) {
                Some(true)
            } else {
                window_requests.remove(&module_id)
            };
            let open = {
                let Some(module) = self.modules.get(&owner) else {
                    continue;
                };
                window::PersistedWindow::new(&module_id, &module_title)
                    .open_override(requested)
                    .default_pos(x, y)
                    .default_size(420.0, 360.0)
                    .show(ui, |ui| modules::render_module(ui, module))
            };
            self.record_window_open(module_id, open);
        }
    }

    fn record_window_open(&mut self, id: String, open: bool) {
        self.window_open.insert(id, open);
    }

    fn push_log(&mut self, message: String) {
        self.logs.push_back(message);
        if self.logs.len() > 512 {
            self.logs.pop_front();
        }
    }

    fn logs_ui(&self, ui: &mut egui::Ui) {
        egui::ScrollArea::vertical().show(ui, |ui| {
            for log in &self.logs {
                text::wrapped_label(ui, log);
            }
        });
    }
}

impl TabStatus {
    fn icon(self) -> &'static str {
        match self {
            Self::Running => "🟢",
            Self::Passed => "✅",
            Self::Failed => "❌",
            Self::Invalid => "⚠️",
            Self::Stopped => "⚪",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reducer_creates_and_updates_tabs() {
        let mut state = VisualizerState::default();
        let tab_id = VisualizerTabId::new("case-1");
        state.apply(VisualizerEvent::OpenTab {
            tab_id: tab_id.clone(),
            title: "Case 1".to_string(),
        });
        state.apply(VisualizerEvent::SetTabStatus {
            tab_id: tab_id.clone(),
            status: TabStatus::Passed,
        });

        let tab = state.tabs().get(&tab_id).expect("tab exists");
        assert_eq!(tab.title, "Case 1");
        assert_eq!(tab.status, TabStatus::Passed);
    }

    #[test]
    fn reducer_replaces_memory_query_results() {
        let mut state = VisualizerState::default();
        let tab_id = VisualizerTabId::new("case-1");
        state.apply(VisualizerEvent::MemoryQueryResult {
            tab_id: tab_id.clone(),
            query: "rust".to_string(),
            records: vec![MemoryRecordView {
                index: "m1".to_string(),
                rank: "episodic".to_string(),
                occurred_at: None,
                content: "learned rust".to_string(),
            }],
        });

        let tab = state.tabs().get(&tab_id).expect("tab exists");
        assert_eq!(tab.memories.query, "rust");
        assert_eq!(tab.memories.query_results[0].content, "learned rust");
    }

    #[test]
    fn reducer_open_tab_marks_preopened_tab_running() {
        let mut state = VisualizerState::default();
        let tab_id = VisualizerTabId::new("case-1");
        state.apply(VisualizerEvent::OpenTab {
            tab_id: tab_id.clone(),
            title: "Case 1".to_string(),
        });
        state.apply(VisualizerEvent::SetTabStatus {
            tab_id: tab_id.clone(),
            status: TabStatus::Stopped,
        });
        state.apply(VisualizerEvent::OpenTab {
            tab_id: tab_id.clone(),
            title: "Case 1".to_string(),
        });

        let tab = state.tabs().get(&tab_id).expect("tab exists");
        assert_eq!(tab.status, TabStatus::Running);
    }

    #[test]
    fn view_window_specs_include_overview_and_module_windows() {
        let mut state = VisualizerState::default();
        let tab_id = VisualizerTabId::new("case-1");
        state.apply(VisualizerEvent::OpenTab {
            tab_id: tab_id.clone(),
            title: "Case 1".to_string(),
        });
        state.apply(VisualizerEvent::LlmObserved {
            tab_id: tab_id.clone(),
            event: LlmObservationEvent::ModelInput {
                turn_id: "turn-1".to_string(),
                owner: "sensory".to_string(),
                module: "sensory".to_string(),
                replica: 0,
                tier: "Default".to_string(),
                source: LlmObservationSource::ModuleTurn,
                operation: "text_turn".to_string(),
                items: Vec::new(),
            },
        });

        let specs = state
            .tabs()
            .get(&tab_id)
            .expect("tab exists")
            .window_specs();

        assert!(
            specs
                .iter()
                .any(|spec| { spec.id == "case-1:modules" && spec.title == "Modules - Case 1" })
        );
        assert!(specs.iter().any(|spec| {
            spec.id == "case-1:module:sensory" && spec.title == "Module - sensory"
        }));
    }
}
