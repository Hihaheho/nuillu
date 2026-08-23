use std::{
    collections::VecDeque,
    fs,
    net::TcpListener,
    rc::Rc,
    sync::{
        Arc, Mutex,
        mpsc::{self, Receiver, Sender},
    },
    thread::{self, JoinHandle},
    time::Duration,
};

#[cfg(not(target_arch = "wasm32"))]
use std::time::Instant;

use anyhow::Context as _;
use chrono::{DateTime, Utc};
use nuillu_agent::{
    AgentEventLoopConfig, AgentRunController, run_controlled_with_timer as run_agent,
};
use nuillu_blackboard::BlackboardCommand;
use nuillu_module::{
    ActionAffordance, AmbientSensoryEntry, Participant, RuntimeEvent, SensoryInput,
    ports::{Timer, TokioTimer},
};
use nuillu_visualizer_protocol::{
    AgentActionInvocationCompletion, EditableSceneStateView, ExternalActionEventRowView,
    ExternalActionEventStatusView, LlmObservationEvent, LlmObservationSource, OneShotSensoryInput,
    TabStatus, UtteranceEventKindView, UtteranceEventRowView, VisualizerAction,
    VisualizerClientMessage, VisualizerCommand, VisualizerEvent, VisualizerServerMessage,
    VisualizerServerPort, VisualizerTabId, run_runtime_action_id, stop_runtime_action_id,
};
use tokio::{runtime::Builder, task::LocalSet};

use crate::SERVER_TAB_ID;
use crate::commands::{
    apply_persisted_module_settings, drive_server_until_shutdown, emit_action_affordances,
    emit_recent_activity_rows, emit_scene_state,
};
use crate::config::ServerConfig;
use crate::environment::{build_native_host_ports, build_server_environment};
use crate::gui::VisualizerHook;
use crate::llm_db_trace::emit_persisted_llm_transcripts;
use crate::ports::ServerHostPorts;
use crate::registry::{
    ServerModuleRegistrar, apply_server_module_registrars, full_agent_allocation, server_registry,
};
use crate::snapshot::emit_visualizer_blackboard_snapshot;
use crate::state::{ActionAffordanceState, ModuleSettingsState, SceneState};

pub(crate) const SERVER_TITLE: &str = "nuillu-server";
const EVENT_BACKLOG_LIMIT: usize = 512;
const AGENT_RESTART_LIMIT: u64 = 5;
const AGENT_RESTART_STABLE_AFTER: Duration = Duration::from_secs(30);

/// A configured nuillu server.
///
/// Constructing the server is separate from choosing how it is hosted: [`Server::listen`]
/// connects to an external visualizer, [`Server::spawn`] embeds it on a dedicated thread, and
/// [`Server::run`] lets an async host supply all runtime capabilities directly.
pub struct Server {
    config: ServerConfig,
    registrars: Vec<Arc<dyn ServerModuleRegistrar>>,
}

/// Capabilities owned by an async server host.
pub struct ServerHost {
    visualizer: VisualizerHook,
    timer: Rc<dyn Timer>,
    ports: ServerHostPorts,
}

impl ServerHost {
    pub fn new(visualizer: VisualizerHook, timer: Rc<dyn Timer>, ports: ServerHostPorts) -> Self {
        Self {
            visualizer,
            timer,
            ports,
        }
    }
}

impl Server {
    pub fn new(config: ServerConfig) -> Self {
        Self {
            config,
            registrars: Vec::new(),
        }
    }

    /// Adds host-provided module registrars.
    pub fn module_registrars(
        mut self,
        registrars: impl IntoIterator<Item = Arc<dyn ServerModuleRegistrar>>,
    ) -> Self {
        self.registrars.extend(registrars);
        self
    }

    /// Waits for an external visualizer connection, then runs on the current thread.
    pub fn listen(self) -> anyhow::Result<()> {
        fs::create_dir_all(&self.config.state_dir)
            .with_context(|| format!("create state dir {}", self.config.state_dir.display()))?;
        let listener =
            TcpListener::bind(("127.0.0.1", 0)).context("bind visualizer RPC listener")?;
        let addr = listener
            .local_addr()
            .context("read visualizer RPC listener address")?;
        eprintln!("nuillu-server visualizer RPC listening on {addr}");
        let (stream, _) = listener
            .accept()
            .context("accept visualizer RPC connection")?;
        eprintln!("visualizer RPC connected");
        let port = VisualizerServerPort::from_stream(stream).context("open visualizer RPC port")?;
        port.send(VisualizerServerMessage::hello())
            .context("send visualizer protocol hello")?;
        let _ = port.recv();

        let (command_rx, event_tx) = port.into_channels();
        self.run_on_current_thread(event_tx, command_rx)
    }

    /// Runs the server on a dedicated thread and returns its control handle.
    pub fn spawn(self) -> anyhow::Result<ServerRuntimeHandle> {
        let (command_tx, command_rx) = mpsc::channel();
        let (visualizer_tx, visualizer_rx) = mpsc::channel();
        let events = Broadcast::new(EVENT_BACKLOG_LIMIT);
        let visualizer_messages = Broadcast::new(EVENT_BACKLOG_LIMIT);
        spawn_visualizer_message_pump(visualizer_rx, visualizer_messages.clone(), events.clone());
        let join = thread::spawn(move || self.run_on_current_thread(visualizer_tx, command_rx));

        Ok(ServerRuntimeHandle {
            commands: command_tx,
            events,
            visualizer_messages,
            join: Arc::new(Mutex::new(Some(join))),
        })
    }

    /// Runs on the caller's current local async executor using host-supplied capabilities.
    ///
    /// This does not create a thread, Tokio runtime, or `LocalSet`. Server modules may contain
    /// non-`Send` state, so the caller must provide a local task context.
    pub async fn run(self, host: ServerHost) -> anyhow::Result<()> {
        let ServerHost {
            mut visualizer,
            timer,
            ports,
        } = host;
        send_visualizer_startup_to_hook(&visualizer, self.config.start_paused);
        let result =
            run_server_inner(self.config, &self.registrars, &mut visualizer, timer, ports).await;
        if let Err(error) = &result {
            let tab_id = server_tab_id();
            visualizer.send_event(VisualizerEvent::Log {
                tab_id: tab_id.clone(),
                message: format!("nuillu-server runtime failed: {error:#}"),
            });
            visualizer.send_event(VisualizerEvent::SetTabStatus {
                tab_id,
                status: TabStatus::Invalid,
            });
        }
        result
    }

    fn run_on_current_thread(
        self,
        event_tx: Sender<VisualizerServerMessage>,
        command_rx: Receiver<VisualizerClientMessage>,
    ) -> anyhow::Result<()> {
        let runtime = Builder::new_current_thread()
            .enable_all()
            .build()
            .context("build server tokio runtime")?;
        let local = LocalSet::new();
        let visualizer = VisualizerHook::new(event_tx, command_rx);
        let timer = Rc::new(TokioTimer::new());
        let result = runtime.block_on(local.run_until(async move {
            let host_ports = build_native_host_ports(&self.config).await?;
            self.run(ServerHost::new(visualizer, timer, host_ports))
                .await
        }));
        if let Err(error) = &result {
            eprintln!("nuillu-server runtime failed: {error:#}");
        }
        result
    }
}

#[derive(Clone)]
pub struct ServerRuntimeHandle {
    commands: Sender<VisualizerClientMessage>,
    events: Broadcast<ServerEvent>,
    visualizer_messages: Broadcast<VisualizerServerMessage>,
    join: Arc<Mutex<Option<JoinHandle<anyhow::Result<()>>>>>,
}

impl ServerRuntimeHandle {
    pub fn send_one_shot(
        &self,
        modality: impl Into<String>,
        direction: Option<String>,
        content: impl Into<String>,
    ) -> anyhow::Result<()> {
        self.send_command(VisualizerCommand::SendOneShotSensoryInput {
            tab_id: server_tab_id(),
            input: OneShotSensoryInput {
                modality: modality.into(),
                direction,
                content: content.into(),
            },
        })
    }

    pub fn publish_sensory_input(&self, input: SensoryInput) -> anyhow::Result<()> {
        self.send_command(VisualizerCommand::PublishSensoryInput {
            tab_id: server_tab_id(),
            input,
        })
    }

    pub fn set_participants(
        &self,
        participants: impl IntoIterator<Item = Participant>,
    ) -> anyhow::Result<()> {
        let people = participants
            .into_iter()
            .enumerate()
            .map(
                |(index, participant)| nuillu_visualizer_protocol::ScenePersonRowView {
                    id: format!("participant-{}", index + 1),
                    name: participant.name,
                    direction: String::new(),
                    distance: String::new(),
                    state: String::new(),
                },
            )
            .collect();
        self.send_command(VisualizerCommand::SaveSceneState {
            tab_id: server_tab_id(),
            state: EditableSceneStateView {
                people,
                ..EditableSceneStateView::default()
            },
        })
    }

    pub fn set_action_affordances(&self, affordances: Vec<ActionAffordance>) -> anyhow::Result<()> {
        self.send_command(VisualizerCommand::SetAgentActionAffordances {
            tab_id: server_tab_id(),
            affordances,
        })
    }

    pub fn complete_external_action(
        &self,
        invocation_id: impl Into<String>,
        accepted: bool,
        message: impl Into<String>,
    ) -> anyhow::Result<()> {
        self.send_command(VisualizerCommand::CompleteAgentActionInvocation {
            tab_id: server_tab_id(),
            completion: AgentActionInvocationCompletion {
                invocation_id: invocation_id.into(),
                accepted,
                message: message.into(),
            },
        })
    }

    pub fn pause(&self) -> anyhow::Result<()> {
        self.commands
            .send(VisualizerClientMessage::InvokeAction {
                action_id: stop_runtime_action_id(&server_tab_id()),
            })
            .context("send server pause command")
    }

    pub fn resume(&self) -> anyhow::Result<()> {
        self.commands
            .send(VisualizerClientMessage::InvokeAction {
                action_id: run_runtime_action_id(&server_tab_id()),
            })
            .context("send server resume command")
    }

    pub fn shutdown(&self) -> anyhow::Result<()> {
        self.send_command(VisualizerCommand::Shutdown)
    }

    /// Asks the runtime to re-emit the authoritative visualizer state.
    ///
    /// Use this after recreating a UI view or message bridge. The resulting messages can be
    /// consumed in batches with [`crate::VisualizerServerMessageReceiverExt::drain`].
    pub fn request_visualizer_snapshot(&self) -> anyhow::Result<()> {
        self.commands
            .send(VisualizerClientMessage::request_snapshot(server_tab_id()))
            .context("request server visualizer snapshot")
    }

    pub fn subscribe_events(&self) -> Receiver<ServerEvent> {
        self.events.subscribe()
    }

    pub fn visualizer_channels(
        &self,
    ) -> (
        Receiver<VisualizerServerMessage>,
        Sender<VisualizerClientMessage>,
    ) {
        (self.visualizer_messages.subscribe(), self.commands.clone())
    }

    pub fn join(self) -> anyhow::Result<()> {
        let Some(join) = self
            .join
            .lock()
            .expect("server runtime join lock poisoned")
            .take()
        else {
            return Ok(());
        };
        join.join()
            .map_err(|panic| anyhow::anyhow!("server runtime thread panicked: {panic:?}"))?
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn join_timeout(&self, timeout: Duration) -> anyhow::Result<bool> {
        let deadline = Instant::now() + timeout;
        loop {
            {
                let mut guard = self.join.lock().expect("server runtime join lock poisoned");
                let Some(join) = guard.as_ref() else {
                    return Ok(true);
                };
                if join.is_finished() {
                    let join = guard
                        .take()
                        .expect("server runtime join handle disappeared");
                    join.join().map_err(|panic| {
                        anyhow::anyhow!("server runtime thread panicked: {panic:?}")
                    })??;
                    return Ok(true);
                }
            }
            if Instant::now() >= deadline {
                return Ok(false);
            }
            thread::sleep(Duration::from_millis(20));
        }
    }

    fn send_command(&self, command: VisualizerCommand) -> anyhow::Result<()> {
        self.commands
            .send(VisualizerClientMessage::Command { command })
            .context("send server runtime command")
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum ServerEvent {
    Log {
        message: String,
    },
    StatusChanged {
        status: ServerRuntimeStatus,
    },
    RuntimeEvent {
        event: RuntimeEvent,
    },
    /// An actual Lutum model operation has received its model input.
    ///
    /// This event is observational: the reported call has already started. A
    /// subscriber can enforce its own budget for subsequent calls by pausing
    /// or shutting down the [`ServerRuntimeHandle`].
    LlmCall {
        call: ServerLlmCall,
    },
    SensoryInput {
        input: SensoryInput,
    },
    OneShotSensoryInputAppended {
        record: ServerOneShotSensoryInputRecord,
    },
    AmbientSensorySnapshotAppended {
        record: ServerAmbientSensorySnapshotRecord,
    },
    UtteranceDelta {
        sender: String,
        target: String,
        generation_id: u64,
        sequence: u32,
        delta: String,
    },
    UtteranceCompleted {
        sender: String,
        target: String,
        generation_id: Option<u64>,
        text: String,
        emitted_at: DateTime<Utc>,
    },
    UtteranceEventAppended {
        record: ServerUtteranceEventRecord,
    },
    ExternalActionRequested {
        invocation_id: String,
        action_id: String,
        arguments: serde_json::Value,
    },
    ExternalActionEventAppended {
        record: ServerExternalActionEventRecord,
    },
    ExternalActionEventUpdated {
        record: ServerExternalActionEventRecord,
    },
    ActionAffordancesChanged {
        affordances: Vec<ActionAffordance>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ServerLlmCall {
    pub turn_id: String,
    pub owner: String,
    pub module: String,
    pub replica: u8,
    pub tier: String,
    pub source: ServerLlmCallSource,
    pub session_key: Option<String>,
    pub operation: String,
    pub activation_id: u64,
    pub activation_attempt: u32,
}

pub type ServerLlmCallSource = LlmObservationSource;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ServerRuntimeStatus {
    Running,
    Passed,
    Failed,
    Stopped,
    Invalid,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ServerOneShotSensoryInputRecord {
    pub id: i64,
    pub server_session_id: String,
    pub modality: String,
    pub direction: Option<String>,
    pub content: String,
    pub observed_at: DateTime<Utc>,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ServerAmbientSensorySnapshotRecord {
    pub id: i64,
    pub server_session_id: String,
    pub entries: Vec<AmbientSensoryEntry>,
    pub observed_at: DateTime<Utc>,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ServerUtteranceEventKind {
    Delta,
    Completed,
    Aborted,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ServerUtteranceEventRecord {
    pub id: i64,
    pub server_session_id: String,
    pub event_kind: ServerUtteranceEventKind,
    pub sender: String,
    pub target: String,
    pub generation_id: u64,
    pub sequence: u32,
    pub content: String,
    pub reason: Option<String>,
    pub occurred_at: DateTime<Utc>,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ServerExternalActionEventStatus {
    Pending,
    Completed,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ServerExternalActionEventRecord {
    pub id: i64,
    pub server_session_id: String,
    pub invocation_id: String,
    pub invoked_by: String,
    pub action_id: String,
    pub arguments: serde_json::Value,
    pub status: ServerExternalActionEventStatus,
    pub accepted: Option<bool>,
    pub message: Option<String>,
    pub requested_at: DateTime<Utc>,
    pub completed_at: Option<DateTime<Utc>>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

fn server_tab_id() -> VisualizerTabId {
    VisualizerTabId::new(SERVER_TAB_ID.to_string())
}

#[derive(Clone)]
struct Broadcast<T: Clone> {
    inner: Arc<Mutex<BroadcastState<T>>>,
}

struct BroadcastState<T: Clone> {
    subscribers: Vec<Sender<T>>,
    backlog: VecDeque<T>,
    max_backlog: usize,
}

impl<T: Clone> Broadcast<T> {
    fn new(max_backlog: usize) -> Self {
        Self {
            inner: Arc::new(Mutex::new(BroadcastState {
                subscribers: Vec::new(),
                backlog: VecDeque::new(),
                max_backlog,
            })),
        }
    }

    fn subscribe(&self) -> Receiver<T> {
        let (tx, rx) = mpsc::channel();
        let mut inner = self.inner.lock().expect("broadcast lock poisoned");
        for item in &inner.backlog {
            if tx.send(item.clone()).is_err() {
                return rx;
            }
        }
        inner.subscribers.push(tx);
        rx
    }

    fn publish(&self, item: T) {
        let mut inner = self.inner.lock().expect("broadcast lock poisoned");
        if inner.max_backlog > 0 {
            inner.backlog.push_back(item.clone());
            while inner.backlog.len() > inner.max_backlog {
                inner.backlog.pop_front();
            }
        }
        inner
            .subscribers
            .retain(|subscriber| subscriber.send(item.clone()).is_ok());
    }
}

fn spawn_visualizer_message_pump(
    rx: Receiver<VisualizerServerMessage>,
    raw_messages: Broadcast<VisualizerServerMessage>,
    events: Broadcast<ServerEvent>,
) {
    thread::spawn(move || {
        while let Ok(message) = rx.recv() {
            if let Some(event) = server_event_from_visualizer_message(&message) {
                events.publish(event);
            }
            raw_messages.publish(message);
        }
    });
}

fn server_event_from_visualizer_message(message: &VisualizerServerMessage) -> Option<ServerEvent> {
    let VisualizerServerMessage::Event { event } = message else {
        return None;
    };
    match event {
        VisualizerEvent::SetTabStatus { status, .. } => Some(ServerEvent::StatusChanged {
            status: match status {
                TabStatus::Running => ServerRuntimeStatus::Running,
                TabStatus::Passed => ServerRuntimeStatus::Passed,
                TabStatus::Failed => ServerRuntimeStatus::Failed,
                TabStatus::Stopped => ServerRuntimeStatus::Stopped,
                TabStatus::Invalid => ServerRuntimeStatus::Invalid,
            },
        }),
        VisualizerEvent::Log { message, .. } => Some(ServerEvent::Log {
            message: message.clone(),
        }),
        VisualizerEvent::RuntimeEvent { event, .. } => Some(ServerEvent::RuntimeEvent {
            event: event.clone(),
        }),
        VisualizerEvent::LlmObserved {
            event:
                LlmObservationEvent::ModelInput {
                    turn_id,
                    owner,
                    module,
                    replica,
                    tier,
                    source,
                    session_key,
                    operation,
                    activation_id,
                    activation_attempt,
                    ..
                },
            ..
        } => Some(ServerEvent::LlmCall {
            call: ServerLlmCall {
                turn_id: turn_id.clone(),
                owner: owner.clone(),
                module: module.clone(),
                replica: *replica,
                tier: tier.clone(),
                source: *source,
                session_key: session_key.clone(),
                operation: operation.clone(),
                activation_id: *activation_id,
                activation_attempt: *activation_attempt,
            },
        }),
        VisualizerEvent::SensoryInput { input, .. } => Some(ServerEvent::SensoryInput {
            input: input.clone(),
        }),
        VisualizerEvent::OneShotSensoryInputAppended { row, .. } => {
            Some(ServerEvent::OneShotSensoryInputAppended {
                record: ServerOneShotSensoryInputRecord {
                    id: row.id,
                    server_session_id: row.server_session_id.clone(),
                    modality: row.modality.clone(),
                    direction: row.direction.clone(),
                    content: row.content.clone(),
                    observed_at: row.observed_at,
                    created_at: row.created_at,
                },
            })
        }
        VisualizerEvent::AmbientSensorySnapshotAppended { row, .. } => {
            Some(ServerEvent::AmbientSensorySnapshotAppended {
                record: ServerAmbientSensorySnapshotRecord {
                    id: row.id,
                    server_session_id: row.server_session_id.clone(),
                    entries: row.entries.clone(),
                    observed_at: row.observed_at,
                    created_at: row.created_at,
                },
            })
        }
        VisualizerEvent::UtteranceDelta { utterance, .. } => Some(ServerEvent::UtteranceDelta {
            sender: utterance.sender.clone(),
            target: utterance.target.clone(),
            generation_id: utterance.generation_id,
            sequence: utterance.sequence,
            delta: utterance.delta.clone(),
        }),
        VisualizerEvent::UtteranceCompleted { utterance, .. } => {
            Some(ServerEvent::UtteranceCompleted {
                sender: utterance.sender.clone(),
                target: utterance.target.clone(),
                generation_id: utterance.generation_id,
                text: utterance.text.clone(),
                emitted_at: utterance.emitted_at,
            })
        }
        VisualizerEvent::UtteranceEventAppended { row, .. } => {
            Some(ServerEvent::UtteranceEventAppended {
                record: server_utterance_event_record(row),
            })
        }
        VisualizerEvent::ExternalActionEventAppended { row, .. } => {
            Some(ServerEvent::ExternalActionEventAppended {
                record: server_external_action_event_record(row),
            })
        }
        VisualizerEvent::ExternalActionEventUpdated { row, .. } => {
            Some(ServerEvent::ExternalActionEventUpdated {
                record: server_external_action_event_record(row),
            })
        }
        VisualizerEvent::AgentActionInvocationRequested { request, .. } => {
            Some(ServerEvent::ExternalActionRequested {
                invocation_id: request.invocation_id.clone(),
                action_id: request.action_id.clone(),
                arguments: request.arguments.clone(),
            })
        }
        VisualizerEvent::AgentActionAffordances { affordances, .. } => {
            Some(ServerEvent::ActionAffordancesChanged {
                affordances: affordances.clone(),
            })
        }
        _ => None,
    }
}

fn server_utterance_event_record(row: &UtteranceEventRowView) -> ServerUtteranceEventRecord {
    ServerUtteranceEventRecord {
        id: row.id,
        server_session_id: row.server_session_id.clone(),
        event_kind: match row.event_kind {
            UtteranceEventKindView::Delta => ServerUtteranceEventKind::Delta,
            UtteranceEventKindView::Completed => ServerUtteranceEventKind::Completed,
            UtteranceEventKindView::Aborted => ServerUtteranceEventKind::Aborted,
        },
        sender: row.sender.clone(),
        target: row.target.clone(),
        generation_id: row.generation_id,
        sequence: row.sequence,
        content: row.content.clone(),
        reason: row.reason.clone(),
        occurred_at: row.occurred_at,
        created_at: row.created_at,
    }
}

fn server_external_action_event_record(
    row: &ExternalActionEventRowView,
) -> ServerExternalActionEventRecord {
    ServerExternalActionEventRecord {
        id: row.id,
        server_session_id: row.server_session_id.clone(),
        invocation_id: row.invocation_id.clone(),
        invoked_by: row.invoked_by.clone(),
        action_id: row.action_id.clone(),
        arguments: row.arguments.clone(),
        status: match row.status {
            ExternalActionEventStatusView::Pending => ServerExternalActionEventStatus::Pending,
            ExternalActionEventStatusView::Completed => ServerExternalActionEventStatus::Completed,
        },
        accepted: row.accepted,
        message: row.message.clone(),
        requested_at: row.requested_at,
        completed_at: row.completed_at,
        created_at: row.created_at,
        updated_at: row.updated_at,
    }
}

fn send_visualizer_startup_to_hook(visualizer: &VisualizerHook, start_paused: bool) {
    let tab_id = server_tab_id();
    visualizer.send_server_message(VisualizerServerMessage::hello());
    visualizer.send_event(VisualizerEvent::OpenTab {
        tab_id: tab_id.clone(),
        title: SERVER_TITLE.to_string(),
    });
    visualizer.send_event(VisualizerEvent::SetTabStatus {
        tab_id,
        status: if start_paused {
            TabStatus::Stopped
        } else {
            TabStatus::Running
        },
    });
}

async fn run_server_inner(
    config: ServerConfig,
    registrars: &[Arc<dyn ServerModuleRegistrar>],
    visualizer: &mut VisualizerHook,
    timer: Rc<dyn Timer>,
    host_ports: ServerHostPorts,
) -> anyhow::Result<()> {
    let tab_id = VisualizerTabId::new(SERVER_TAB_ID.to_string());
    visualizer.send_event(VisualizerEvent::Log {
        tab_id: tab_id.clone(),
        message: format!("nuillu-server session_id={}", config.session_id),
    });
    let state_port = host_ports.state().clone();
    let mut scene = SceneState::load(state_port.as_ref(), &config.participants).await?;
    scene.save(state_port.as_ref()).await?;
    let mut module_settings = ModuleSettingsState::load(state_port.as_ref()).await?;
    let mut action_affordances = ActionAffordanceState::load(state_port.as_ref()).await?;
    action_affordances.save(state_port.as_ref()).await?;

    let active_modules = config.active_modules();
    let env = build_server_environment(
        &config,
        &host_ports,
        full_agent_allocation(&config.boot_config),
        visualizer.event_sender(),
        timer.clone(),
    )
    .await?;
    let action_snapshot = env
        .caps
        .host_io()
        .action_affordance_writer()
        .set_all(
            config
                .boot_config
                .overlay_action_affordances(action_affordances.affordances()),
        )
        .await
        .context("seed action affordances")?;
    env.caps.scene().set(scene.participants());
    emit_scene_state(&scene, visualizer, &tab_id);
    emit_action_affordances(visualizer, &tab_id, action_snapshot.affordances);
    emit_recent_activity_rows(&env, visualizer, &tab_id).await;
    for module in config
        .disabled_modules
        .iter()
        .filter(|module| active_modules.contains(module))
    {
        env.blackboard
            .apply(BlackboardCommand::SetModuleForcedDisabled {
                module: module.module_id(),
                disabled: true,
            })
            .await;
    }

    emit_visualizer_blackboard_snapshot(SERVER_TAB_ID, &env.blackboard, visualizer).await;
    emit_persisted_llm_transcripts(
        env.llm_transcript_store.as_ref(),
        SERVER_TAB_ID,
        &visualizer.event_sender(),
    )
    .await;

    let sensory = env.caps.host_io().sensory_input_mailbox();
    let (run_controller, run_control) = if config.start_paused {
        AgentRunController::new_paused()
    } else {
        AgentRunController::new()
    };
    set_runtime_running(visualizer, &tab_id, &run_controller, !config.start_paused);
    let mut restart_count = 0_u64;
    loop {
        let mut registry = server_registry(
            &config.boot_config,
            &env.memory_caps,
            &env.policy_caps,
            &env.utterance_sink,
        );
        registry = apply_server_module_registrars(registry, registrars)
            .context("register host-provided server modules")?;
        let allocated = registry.build(&env.caps).await?;
        apply_persisted_module_settings(&module_settings, visualizer, &tab_id, &env.blackboard)
            .await;

        let run_started_at = timer.elapsed();
        let result = run_agent(
            allocated,
            AgentEventLoopConfig {
                idle_threshold: Duration::from_secs(1),
                max_activation_attempts: 5,
                dependency_idle_timeout: Duration::from_secs(2),
                dependency_hard_timeout: Duration::from_secs(10),
            },
            run_control.clone(),
            timer.clone(),
            drive_server_until_shutdown(
                visualizer,
                &tab_id,
                &mut scene,
                &mut module_settings,
                &mut action_affordances,
                state_port.as_ref(),
                &config.boot_config,
                &sensory,
                &env,
                &run_controller,
                timer.as_ref(),
            ),
        )
        .await;

        if timer.elapsed().saturating_sub(run_started_at) >= AGENT_RESTART_STABLE_AFTER {
            restart_count = 0;
        }
        match result {
            Ok(()) if visualizer.shutdown_requested() => break,
            Ok(()) => {
                restart_count = restart_count.saturating_add(1);
                let Some(delay) = agent_restart_delay(restart_count) else {
                    let message = format!(
                        "agent runtime ended without a GUI shutdown {restart_count} consecutive times; stopping"
                    );
                    eprintln!("nuillu-server {message}");
                    visualizer.send_event(VisualizerEvent::Log {
                        tab_id: tab_id.clone(),
                        message: message.clone(),
                    });
                    visualizer.send_event(VisualizerEvent::SetTabStatus {
                        tab_id,
                        status: TabStatus::Failed,
                    });
                    anyhow::bail!(message);
                };
                let message = format!(
                    "agent runtime ended without a GUI shutdown; restarting attempt={restart_count} next_retry_ms={}",
                    delay.as_millis()
                );
                eprintln!("nuillu-server {message}");
                visualizer.send_event(VisualizerEvent::Log {
                    tab_id: tab_id.clone(),
                    message,
                });
            }
            Err(error) => {
                restart_count = restart_count.saturating_add(1);
                let Some(delay) = agent_restart_delay(restart_count) else {
                    let message = format!(
                        "agent runtime error {restart_count} consecutive times; stopping: {error}"
                    );
                    eprintln!("nuillu-server {message}");
                    visualizer.send_event(VisualizerEvent::Log {
                        tab_id: tab_id.clone(),
                        message: message.clone(),
                    });
                    visualizer.send_event(VisualizerEvent::SetTabStatus {
                        tab_id,
                        status: TabStatus::Failed,
                    });
                    anyhow::bail!(message);
                };
                let message = format!(
                    "agent runtime error; restarting attempt={restart_count} next_retry_ms={}: {error}",
                    delay.as_millis()
                );
                eprintln!("nuillu-server {message}");
                visualizer.send_event(VisualizerEvent::Log {
                    tab_id: tab_id.clone(),
                    message,
                });
            }
        }

        if visualizer.shutdown_requested() {
            break;
        }
        timer
            .sleep(agent_restart_delay(restart_count).unwrap_or_default())
            .await;
    }
    visualizer.send_event(VisualizerEvent::SetTabStatus {
        tab_id,
        status: TabStatus::Stopped,
    });
    Ok(())
}

fn agent_restart_delay(attempt: u64) -> Option<Duration> {
    if attempt >= AGENT_RESTART_LIMIT {
        return None;
    }
    let multiplier = 1_u64 << attempt.saturating_sub(1).min(5);
    Some(Duration::from_millis(500_u64.saturating_mul(multiplier)))
}

pub(crate) fn set_runtime_running(
    visualizer: &VisualizerHook,
    tab_id: &VisualizerTabId,
    controller: &AgentRunController,
    running: bool,
) {
    if running {
        controller.resume();
        visualizer.send_event(VisualizerEvent::SetTabStatus {
            tab_id: tab_id.clone(),
            status: TabStatus::Running,
        });
        visualizer.revoke_action(run_runtime_action_id(tab_id));
        visualizer.offer_action(VisualizerAction::stop_runtime(tab_id.clone()));
    } else {
        controller.pause();
        visualizer.send_event(VisualizerEvent::SetTabStatus {
            tab_id: tab_id.clone(),
            status: TabStatus::Stopped,
        });
        visualizer.revoke_action(stop_runtime_action_id(tab_id));
        visualizer.offer_action(VisualizerAction::run_runtime(tab_id.clone()));
    }
}

#[cfg(test)]
mod tests {
    use std::sync::mpsc;

    use nuillu_module::{ActionAffordance, SensoryModality};
    use nuillu_visualizer_protocol::{
        AgentActionInvocationRequest, VisualizerActionKind, VisualizerClientMessage,
    };

    use super::*;

    fn test_handle() -> (ServerRuntimeHandle, Receiver<VisualizerClientMessage>) {
        let (tx, rx) = mpsc::channel();
        (
            ServerRuntimeHandle {
                commands: tx,
                events: Broadcast::new(EVENT_BACKLOG_LIMIT),
                visualizer_messages: Broadcast::new(EVENT_BACKLOG_LIMIT),
                join: Arc::new(Mutex::new(None)),
            },
            rx,
        )
    }

    #[test]
    fn runtime_handle_sends_sensory_and_control_commands() {
        let (handle, rx) = test_handle();

        handle
            .send_one_shot("audition", Some("Peer".to_string()), "hello")
            .unwrap();
        handle
            .publish_sensory_input(SensoryInput::AmbientSnapshot {
                entries: vec![AmbientSensoryEntry {
                    id: "room".to_string(),
                    modality: SensoryModality::parse("vision"),
                    content: "lamp is on".to_string(),
                }],
                observed_at: chrono::Utc::now(),
            })
            .unwrap();
        handle.pause().unwrap();
        handle.resume().unwrap();
        handle.request_visualizer_snapshot().unwrap();
        handle.shutdown().unwrap();

        let messages = rx.try_iter().collect::<Vec<_>>();
        assert!(matches!(
            &messages[0],
            VisualizerClientMessage::Command {
                command: VisualizerCommand::SendOneShotSensoryInput { input, .. }
            } if input.modality == "audition"
                && input.direction.as_deref() == Some("Peer")
                && input.content == "hello"
        ));
        assert!(matches!(
            &messages[1],
            VisualizerClientMessage::Command {
                command: VisualizerCommand::PublishSensoryInput {
                    input: SensoryInput::AmbientSnapshot { entries, .. },
                    ..
                }
            } if entries.len() == 1 && entries[0].content == "lamp is on"
        ));
        assert!(matches!(
            &messages[2],
            VisualizerClientMessage::InvokeAction { action_id }
                if action_id == &stop_runtime_action_id(&server_tab_id())
        ));
        assert!(matches!(
            &messages[3],
            VisualizerClientMessage::InvokeAction { action_id }
                if action_id == &run_runtime_action_id(&server_tab_id())
        ));
        assert!(matches!(
            &messages[4],
            VisualizerClientMessage::Command {
                command: VisualizerCommand::RequestSnapshot { tab_id }
            } if tab_id == &server_tab_id()
        ));
        assert!(matches!(
            &messages[5],
            VisualizerClientMessage::Command {
                command: VisualizerCommand::Shutdown
            }
        ));
    }

    #[test]
    fn runtime_handle_sends_participants_actions_and_action_completion() {
        let (handle, rx) = test_handle();

        handle
            .set_participants([Participant::new("Pibi"), Participant::new("Koro")])
            .unwrap();
        handle
            .set_action_affordances(vec![ActionAffordance {
                id: "poet".to_string(),
                label: "Poet".to_string(),
                description: "Write a poem".to_string(),
                use_when: "A poem is useful".to_string(),
                effect: "The host records a poem".to_string(),
                input_schema: serde_json::json!({"type": "object"}),
            }])
            .unwrap();
        handle
            .complete_external_action("agent-action-1", true, "accepted")
            .unwrap();

        let messages = rx.try_iter().collect::<Vec<_>>();
        assert!(matches!(
            &messages[0],
            VisualizerClientMessage::Command {
                command: VisualizerCommand::SaveSceneState { state, .. }
            } if state.people.iter().map(|person| person.name.as_str()).collect::<Vec<_>>()
                == vec!["Pibi", "Koro"]
        ));
        assert!(matches!(
            &messages[1],
            VisualizerClientMessage::Command {
                command: VisualizerCommand::SetAgentActionAffordances { affordances, .. }
            } if affordances.len() == 1 && affordances[0].id == "poet"
        ));
        assert!(matches!(
            &messages[2],
            VisualizerClientMessage::Command {
                command: VisualizerCommand::CompleteAgentActionInvocation { completion, .. }
            } if completion.invocation_id == "agent-action-1"
                && completion.accepted
                && completion.message == "accepted"
        ));
    }

    #[test]
    fn server_event_maps_external_action_request() {
        let message =
            VisualizerServerMessage::event(VisualizerEvent::AgentActionInvocationRequested {
                tab_id: server_tab_id(),
                request: AgentActionInvocationRequest {
                    invocation_id: "agent-action-1".to_string(),
                    action_id: "poet".to_string(),
                    arguments: serde_json::json!({ "poem": "quiet rain" }),
                },
            });

        assert_eq!(
            server_event_from_visualizer_message(&message),
            Some(ServerEvent::ExternalActionRequested {
                invocation_id: "agent-action-1".to_string(),
                action_id: "poet".to_string(),
                arguments: serde_json::json!({ "poem": "quiet rain" }),
            })
        );
    }

    #[test]
    fn server_event_maps_lutum_model_input_to_llm_call() {
        let message = VisualizerServerMessage::event(VisualizerEvent::LlmObserved {
            tab_id: server_tab_id(),
            event: LlmObservationEvent::ModelInput {
                turn_id: "predict-0:7:1".to_string(),
                owner: "predict#0".to_string(),
                module: "predict".to_string(),
                replica: 0,
                tier: "Premium".to_string(),
                source: LlmObservationSource::ModuleTurn,
                session_key: Some("main".to_string()),
                operation: "text_turn".to_string(),
                activation_id: 7,
                activation_attempt: 1,
                batch: nuillu_visualizer_protocol::LlmBatchDebugView {
                    batch_type: "cognition".to_string(),
                    debug: String::new(),
                },
                items: Vec::new(),
            },
        });

        assert_eq!(
            server_event_from_visualizer_message(&message),
            Some(ServerEvent::LlmCall {
                call: ServerLlmCall {
                    turn_id: "predict-0:7:1".to_string(),
                    owner: "predict#0".to_string(),
                    module: "predict".to_string(),
                    replica: 0,
                    tier: "Premium".to_string(),
                    source: LlmObservationSource::ModuleTurn,
                    session_key: Some("main".to_string()),
                    operation: "text_turn".to_string(),
                    activation_id: 7,
                    activation_attempt: 1,
                },
            })
        );
    }

    #[test]
    fn visualizer_channels_replay_backlog() {
        let (handle, _rx) = test_handle();
        handle
            .visualizer_messages
            .publish(VisualizerServerMessage::hello());

        let (server_rx, _client_tx) = handle.visualizer_channels();

        assert!(matches!(
            server_rx.recv_timeout(Duration::from_secs(1)).unwrap(),
            VisualizerServerMessage::Hello { .. }
        ));
    }

    #[test]
    fn set_runtime_running_updates_controller_status_and_actions() {
        let (event_tx, event_rx) = mpsc::channel();
        let (_command_tx, command_rx) = mpsc::channel::<VisualizerClientMessage>();
        let visualizer = VisualizerHook::new(event_tx, command_rx);
        let tab_id = VisualizerTabId::new("server");
        let (controller, _control) = AgentRunController::new();

        set_runtime_running(&visualizer, &tab_id, &controller, false);
        assert!(!controller.is_running());
        let messages = event_rx.try_iter().collect::<Vec<_>>();
        assert_eq!(messages.len(), 3);
        assert!(matches!(
            &messages[0],
            VisualizerServerMessage::Event {
                event: VisualizerEvent::SetTabStatus {
                    tab_id: actual_tab_id,
                    status: TabStatus::Stopped,
                    ..
                }
            } if actual_tab_id == &tab_id
        ));
        assert!(matches!(
            &messages[1],
            VisualizerServerMessage::RevokeAction { action_id }
                if action_id == &stop_runtime_action_id(&tab_id)
        ));
        assert!(matches!(
            &messages[2],
            VisualizerServerMessage::OfferAction { action }
                if action.id == run_runtime_action_id(&tab_id)
                    && action.kind == VisualizerActionKind::RunRuntime
        ));

        set_runtime_running(&visualizer, &tab_id, &controller, true);
        assert!(controller.is_running());
        let messages = event_rx.try_iter().collect::<Vec<_>>();
        assert_eq!(messages.len(), 3);
        assert!(matches!(
            &messages[0],
            VisualizerServerMessage::Event {
                event: VisualizerEvent::SetTabStatus {
                    tab_id: actual_tab_id,
                    status: TabStatus::Running,
                    ..
                }
            } if actual_tab_id == &tab_id
        ));
        assert!(matches!(
            &messages[1],
            VisualizerServerMessage::RevokeAction { action_id }
                if action_id == &run_runtime_action_id(&tab_id)
        ));
        assert!(matches!(
            &messages[2],
            VisualizerServerMessage::OfferAction { action }
                if action.id == stop_runtime_action_id(&tab_id)
                    && action.kind == VisualizerActionKind::StopRuntime
        ));
    }

    #[test]
    fn visualizer_startup_status_matches_initial_run_state() {
        let (event_tx, event_rx) = mpsc::channel();
        let (_command_tx, command_rx) = mpsc::channel::<VisualizerClientMessage>();
        let visualizer = VisualizerHook::new(event_tx, command_rx);

        send_visualizer_startup_to_hook(&visualizer, true);

        let messages = event_rx.try_iter().collect::<Vec<_>>();
        assert!(matches!(
            &messages[2],
            VisualizerServerMessage::Event {
                event: VisualizerEvent::SetTabStatus {
                    status: TabStatus::Stopped,
                    ..
                }
            }
        ));
    }
}
