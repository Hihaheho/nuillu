pub mod blackboard;
pub mod chat;
pub mod cognition;
pub mod errors;
pub mod memories;
pub mod memos;
pub mod modules;
pub mod text;
pub mod window;

pub use eframe;
pub use egui;
pub use egui_hooks;

use std::collections::{BTreeMap, VecDeque};
use std::sync::mpsc::{Receiver, Sender, TryRecvError};

use nuillu_module::RuntimeEvent;
pub use nuillu_visualizer_protocol::*;

pub struct VisualizerChannels {
    pub server_messages: Receiver<VisualizerServerMessage>,
    pub client_messages: Sender<VisualizerClientMessage>,
    pub remote: bool,
}

pub struct VisualizerApp {
    server_messages: Receiver<VisualizerServerMessage>,
    client_messages: Sender<VisualizerClientMessage>,
    remote: bool,
    state: VisualizerState,
}

impl VisualizerApp {
    pub fn new(_cc: &eframe::CreationContext<'_>, channels: VisualizerChannels) -> Self {
        Self {
            server_messages: channels.server_messages,
            client_messages: channels.client_messages,
            remote: channels.remote,
            state: VisualizerState::default(),
        }
    }

    fn drain_server_messages(&mut self) {
        loop {
            match self.server_messages.try_recv() {
                Ok(message) => self.state.apply_server_message(message),
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => {
                    if self.remote {
                        self.state.mark_disconnected();
                    }
                    break;
                }
            }
        }
    }
}

impl eframe::App for VisualizerApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        self.drain_server_messages();
        ui.ctx()
            .request_repaint_after(std::time::Duration::from_millis(100));

        egui::Panel::top("nuillu-visualizer-tabs").show_inside(ui, |ui| {
            ui.horizontal_wrapped(|ui| {
                for tab in self.state.tabs.values() {
                    let selected = self.state.selected.as_ref() == Some(&tab.id);
                    let label = format!("{} {}", tab_status_icon(tab.status), tab.title);
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
                for action in self.state.visible_actions() {
                    if ui.button(&action.label).clicked() {
                        let _ = self
                            .client_messages
                            .send(VisualizerClientMessage::InvokeAction {
                                action_id: action.id,
                            });
                    }
                }
                if self.remote && self.state.disconnected {
                    ui.colored_label(ui.visuals().error_fg_color, "Eval disconnected");
                }
                if ui.button("Shutdown").clicked() {
                    let _ = self.client_messages.send(VisualizerClientMessage::Command {
                        command: VisualizerCommand::Shutdown,
                    });
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
                        tab.ui(ui, &self.client_messages);
                    });
                }
            } else {
                ui.centered_and_justified(|ui| {
                    if self.remote && self.state.disconnected {
                        ui.label("Eval process disconnected.");
                    } else {
                        ui.label("No runtime tabs yet.");
                    }
                });
            }
        });
    }

    fn auto_save_interval(&self) -> std::time::Duration {
        std::time::Duration::from_millis(1500)
    }

    fn on_exit(&mut self) {
        let _ = self.client_messages.send(VisualizerClientMessage::Command {
            command: VisualizerCommand::Shutdown,
        });
    }
}

#[derive(Default)]
pub struct VisualizerState {
    tabs: BTreeMap<VisualizerTabId, RuntimeTab>,
    selected: Option<VisualizerTabId>,
    actions: BTreeMap<String, VisualizerAction>,
    disconnected: bool,
}

impl VisualizerState {
    pub fn apply_server_message(&mut self, message: VisualizerServerMessage) {
        match message {
            VisualizerServerMessage::Hello { .. } => {
                self.disconnected = false;
            }
            VisualizerServerMessage::Event { event } => self.apply(event),
            VisualizerServerMessage::OfferAction { action } => {
                self.actions.insert(action.id.clone(), action);
            }
            VisualizerServerMessage::RevokeAction { action_id } => {
                self.actions.remove(&action_id);
            }
        }
    }

    pub fn mark_disconnected(&mut self) {
        if self.disconnected {
            return;
        }
        self.disconnected = true;
        self.actions.clear();
        for tab in self.tabs.values_mut() {
            tab.push_log("eval process disconnected".to_string());
        }
    }

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
            VisualizerEvent::Error { tab_id, error } => {
                let selected = {
                    let tab = self.tab_mut(tab_id);
                    let errors_id = tab.errors_window_id();
                    tab.push_error(error);
                    tab.window_requests.insert(errors_id, true);
                    tab.id.clone()
                };
                self.selected = Some(selected);
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
                tab.memories.linked_memory_index.clear();
                tab.memories.linked_results.clear();
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
            VisualizerEvent::MemoryLinkedResult {
                tab_id,
                memory_index,
                records,
            } => {
                let tab = self.tab_mut(tab_id);
                tab.memories.linked_memory_index = memory_index;
                tab.memories.linked_results = records;
            }
            VisualizerEvent::AmbientSensoryRows { tab_id, rows } => {
                self.tab_mut(tab_id).chat.set_ambient_rows(rows);
            }
        }
    }

    pub fn tabs(&self) -> &BTreeMap<VisualizerTabId, RuntimeTab> {
        &self.tabs
    }

    pub fn visible_actions(&self) -> Vec<VisualizerAction> {
        self.actions
            .values()
            .filter(|action| match &action.scope {
                VisualizerActionScope::Global => true,
                VisualizerActionScope::Tab { tab_id } => self.selected.as_ref() == Some(tab_id),
            })
            .cloned()
            .collect()
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
    errors: VecDeque<VisualizerErrorView>,
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
            errors: VecDeque::new(),
            logs: VecDeque::new(),
            window_open: BTreeMap::new(),
            window_requests: BTreeMap::new(),
        }
    }

    fn ui(&mut self, ui: &mut egui::Ui, commands: &Sender<VisualizerClientMessage>) {
        ui.horizontal_wrapped(|ui| {
            ui.heading(format!("{} {}", tab_status_icon(self.status), self.title));
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
                id: format!("{base}:errors"),
                title: format!("Errors - {}", self.title),
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

    fn windows_ui(&mut self, ui: &mut egui::Ui, commands: &Sender<VisualizerClientMessage>) {
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

        let errors_id = self.errors_window_id();
        let errors_title = format!("Errors - {}", self.title);
        let open = window::PersistedWindow::new(&errors_id, &errors_title)
            .open_override(window_requests.remove(&errors_id))
            .default_pos(24.0, 1020.0)
            .default_size(640.0, 360.0)
            .show(ui, |ui| errors::ui(ui, &mut self.errors));
        self.record_window_open(errors_id, open);

        let logs_id = format!("{base}:logs");
        let logs_title = format!("📜 Logs - {}", self.title);
        let open = window::PersistedWindow::new(&logs_id, &logs_title)
            .open_override(window_requests.remove(&logs_id))
            .default_pos(688.0, 1020.0)
            .default_size(520.0, 360.0)
            .show(ui, |ui| self.logs_ui(ui));
        self.record_window_open(logs_id, open);

        let modules_id = format!("{base}:modules");
        let modules_title = format!("Modules - {}", self.title);
        let mut requested_module = None;
        let mut module_commands = Vec::new();
        let open = window::PersistedWindow::new(&modules_id, &modules_title)
            .open_override(window_requests.remove(&modules_id))
            .default_pos(568.0, 1020.0)
            .default_size(640.0, 360.0)
            .show(ui, |ui| {
                module_commands =
                    modules::render_modules_overview(ui, &self.blackboard, &self.modules);
            });
        self.record_window_open(modules_id, open);
        for action in module_commands {
            match action {
                modules::ModuleOverviewAction::OpenModule { owner } => {
                    requested_module = Some(owner);
                }
                modules::ModuleOverviewAction::SetDisabled { module, disabled } => {
                    let _ = commands.send(VisualizerClientMessage::Command {
                        command: VisualizerCommand::SetModuleDisabled {
                            tab_id: self.id.clone(),
                            module,
                            disabled,
                        },
                    });
                }
                modules::ModuleOverviewAction::SetModuleSettings { settings } => {
                    let _ = commands.send(VisualizerClientMessage::Command {
                        command: VisualizerCommand::SetModuleSettings {
                            tab_id: self.id.clone(),
                            settings,
                        },
                    });
                }
            }
        }

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
                    .show(ui, |ui| {
                        modules::render_module(ui, module, &self.blackboard.memos)
                    })
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

    fn push_error(&mut self, error: VisualizerErrorView) {
        self.errors.push_back(error);
        if self.errors.len() > 256 {
            self.errors.pop_front();
        }
    }

    fn errors_window_id(&self) -> String {
        format!("{}:errors", self.id.as_str())
    }

    fn logs_ui(&self, ui: &mut egui::Ui) {
        egui::ScrollArea::vertical().show(ui, |ui| {
            for log in &self.logs {
                text::wrapped_label(ui, log);
            }
        });
    }
}

fn tab_status_icon(status: TabStatus) -> &'static str {
    match status {
        TabStatus::Running => "🟢",
        TabStatus::Passed => "✅",
        TabStatus::Failed => "❌",
        TabStatus::Invalid => "⚠️",
        TabStatus::Stopped => "⚪",
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
                kind: "Statement".to_string(),
                rank: "episodic".to_string(),
                occurred_at: None,
                stored_at: chrono::Utc::now(),
                concepts: Vec::new(),
                tags: Vec::new(),
                affect_arousal: 0.0,
                valence: 0.0,
                emotion: String::new(),
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
    fn reducer_tracks_offered_actions_by_scope() {
        let mut state = VisualizerState::default();
        let tab_id = VisualizerTabId::new("case-1");
        state.apply(VisualizerEvent::OpenTab {
            tab_id: tab_id.clone(),
            title: "Case 1".to_string(),
        });
        state.apply_server_message(VisualizerServerMessage::OfferAction {
            action: VisualizerAction::start_suite(),
        });
        state.apply_server_message(VisualizerServerMessage::OfferAction {
            action: VisualizerAction::start_activation(tab_id.clone()),
        });

        let actions = state.visible_actions();
        assert_eq!(actions.len(), 2);
        assert!(
            actions
                .iter()
                .any(|action| action.id == START_SUITE_ACTION_ID)
        );
        assert!(
            actions
                .iter()
                .any(|action| action.id == start_activation_action_id(&tab_id))
        );

        state.apply_server_message(VisualizerServerMessage::RevokeAction {
            action_id: start_activation_action_id(&tab_id),
        });
        let actions = state.visible_actions();
        assert_eq!(actions.len(), 1);
        assert_eq!(actions[0].id, START_SUITE_ACTION_ID);
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

    #[test]
    fn reducer_opens_errors_window_when_error_arrives() {
        let mut state = VisualizerState::default();
        let tab_id = VisualizerTabId::new("case-1");

        state.apply(VisualizerEvent::Error {
            tab_id: tab_id.clone(),
            error: VisualizerErrorView {
                at: chrono::Utc::now(),
                source: "runtime".to_string(),
                phase: "activate".to_string(),
                owner: Some("sensory".to_string()),
                message: "planned failure".to_string(),
            },
        });

        let tab = state.tabs().get(&tab_id).expect("tab exists");
        assert_eq!(tab.errors.len(), 1);
        assert_eq!(state.selected.as_ref(), Some(&tab_id));
        assert_eq!(tab.window_requests.get("case-1:errors"), Some(&true));
    }
}
