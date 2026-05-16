use std::{
    collections::BTreeMap,
    io::{self, BufRead, BufReader, BufWriter, ErrorKind, Write},
    net::{TcpListener, TcpStream, ToSocketAddrs},
    sync::mpsc::{self, Receiver, RecvTimeoutError, Sender, TryRecvError},
    thread,
    time::Duration,
};

use chrono::{DateTime, Utc};
use nuillu_module::{RuntimeEvent, SensoryInput};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use thiserror::Error;

pub const VISUALIZER_PROTOCOL_VERSION: u32 = 1;
pub const START_SUITE_ACTION_ID: &str = "suite:start";

pub fn start_activation_action_id(tab_id: &VisualizerTabId) -> String {
    format!("tab:{}:start-activation", tab_id.as_str())
}

#[derive(Debug, Error)]
pub enum VisualizerProtocolError {
    #[error("visualizer transport disconnected")]
    Disconnected,
    #[error("visualizer transport I/O error: {0}")]
    Io(#[from] io::Error),
    #[error("visualizer protocol serialization error: {0}")]
    Serde(#[from] serde_json::Error),
}

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
#[serde(tag = "type", rename_all = "snake_case")]
pub enum VisualizerServerMessage {
    Hello { version: u32 },
    Event { event: VisualizerEvent },
    OfferAction { action: VisualizerAction },
    RevokeAction { action_id: String },
}

impl VisualizerServerMessage {
    pub fn hello() -> Self {
        Self::Hello {
            version: VISUALIZER_PROTOCOL_VERSION,
        }
    }

    pub fn event(event: VisualizerEvent) -> Self {
        Self::Event { event }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum VisualizerClientMessage {
    Hello { version: u32 },
    InvokeAction { action_id: String },
    Command { command: VisualizerCommand },
}

impl VisualizerClientMessage {
    pub fn hello() -> Self {
        Self::Hello {
            version: VISUALIZER_PROTOCOL_VERSION,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VisualizerAction {
    pub id: String,
    pub label: String,
    pub scope: VisualizerActionScope,
    pub kind: VisualizerActionKind,
}

impl VisualizerAction {
    pub fn start_suite() -> Self {
        Self {
            id: START_SUITE_ACTION_ID.to_string(),
            label: "Start Suite".to_string(),
            scope: VisualizerActionScope::Global,
            kind: VisualizerActionKind::StartSuite,
        }
    }

    pub fn start_activation(tab_id: VisualizerTabId) -> Self {
        Self {
            id: start_activation_action_id(&tab_id),
            label: "Start Activation".to_string(),
            scope: VisualizerActionScope::Tab { tab_id },
            kind: VisualizerActionKind::StartActivation,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "scope", rename_all = "snake_case")]
pub enum VisualizerActionScope {
    Global,
    Tab { tab_id: VisualizerTabId },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VisualizerActionKind {
    StartSuite,
    StartActivation,
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
    MemoryLinkedResult {
        tab_id: VisualizerTabId,
        memory_index: String,
        records: Vec<LinkedMemoryRecordView>,
    },
    AmbientSensoryRows {
        tab_id: VisualizerTabId,
        rows: Vec<AmbientSensoryRowView>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VisualizerCommand {
    SendOneShotSensoryInput {
        tab_id: VisualizerTabId,
        input: OneShotSensoryInput,
    },
    CreateAmbientSensoryRow {
        tab_id: VisualizerTabId,
        modality: String,
        content: String,
        disabled: bool,
    },
    UpdateAmbientSensoryRow {
        tab_id: VisualizerTabId,
        row: AmbientSensoryRowView,
    },
    RemoveAmbientSensoryRow {
        tab_id: VisualizerTabId,
        row_id: String,
    },
    SetModuleDisabled {
        tab_id: VisualizerTabId,
        module: String,
        disabled: bool,
    },
    SetModuleSettings {
        tab_id: VisualizerTabId,
        settings: ModuleSettingsView,
    },
    QueryMemory {
        tab_id: VisualizerTabId,
        query: String,
        limit: usize,
    },
    FetchLinkedMemories {
        tab_id: VisualizerTabId,
        memory_index: String,
        relation_filter: Vec<String>,
        limit: usize,
    },
    DeleteMemory {
        tab_id: VisualizerTabId,
        memory_index: String,
        page: usize,
        per_page: usize,
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OneShotSensoryInput {
    pub modality: String,
    pub content: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AmbientSensoryRowView {
    pub id: String,
    pub modality: String,
    pub content: String,
    pub disabled: bool,
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
    #[serde(default)]
    pub module_policies: Vec<ModulePolicyView>,
    #[serde(default)]
    pub forced_disabled_modules: Vec<String>,
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

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModulePolicyView {
    pub module: String,
    pub replica_min: u8,
    pub replica_max: u8,
    pub replica_capacity: u8,
    pub bpm_min: f64,
    pub bpm_max: f64,
    pub zero_replica_window: ZeroReplicaWindowView,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModuleSettingsView {
    pub module: String,
    pub replica_min: u8,
    pub replica_max: u8,
    pub bpm_min: f64,
    pub bpm_max: f64,
    pub zero_replica_window: ZeroReplicaWindowView,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "mode", rename_all = "snake_case")]
pub enum ZeroReplicaWindowView {
    Disabled,
    EveryControllerActivations { period: u32 },
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
    pub use_count: u32,
    pub reinforcement_count: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryRecordView {
    pub index: String,
    pub kind: String,
    pub rank: String,
    pub occurred_at: Option<DateTime<Utc>>,
    pub stored_at: DateTime<Utc>,
    pub concepts: Vec<MemoryConceptView>,
    pub tags: Vec<MemoryTagView>,
    pub affect_arousal: f32,
    pub valence: f32,
    pub emotion: String,
    pub content: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryConceptView {
    pub label: String,
    pub mention_text: Option<String>,
    pub loose_type: Option<String>,
    pub confidence: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryTagView {
    pub label: String,
    pub namespace: String,
    pub confidence: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryLinkView {
    pub from_memory: String,
    pub to_memory: String,
    pub relation: String,
    pub freeform_relation: Option<String>,
    pub strength: f32,
    pub confidence: f32,
    pub updated_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinkedMemoryRecordView {
    pub record: MemoryRecordView,
    pub link: MemoryLinkView,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MemoryPage {
    pub page: usize,
    pub per_page: usize,
    pub total: usize,
    pub records: Vec<MemoryRecordView>,
}

pub struct VisualizerServerPort {
    incoming: Receiver<VisualizerClientMessage>,
    outgoing: Sender<VisualizerServerMessage>,
}

impl VisualizerServerPort {
    pub fn accept(listener: TcpListener) -> Result<Self, VisualizerProtocolError> {
        let (stream, _) = listener.accept()?;
        Self::from_stream(stream)
    }

    pub fn from_stream(stream: TcpStream) -> Result<Self, VisualizerProtocolError> {
        stream.set_nonblocking(false)?;
        let (incoming, outgoing) =
            spawn_connection_threads::<VisualizerClientMessage, VisualizerServerMessage>(stream)?;
        Ok(Self { incoming, outgoing })
    }

    pub fn send(&self, message: VisualizerServerMessage) -> Result<(), VisualizerProtocolError> {
        self.outgoing
            .send(message)
            .map_err(|_| VisualizerProtocolError::Disconnected)
    }

    pub fn sender(&self) -> Sender<VisualizerServerMessage> {
        self.outgoing.clone()
    }

    pub fn into_channels(
        self,
    ) -> (
        Receiver<VisualizerClientMessage>,
        Sender<VisualizerServerMessage>,
    ) {
        (self.incoming, self.outgoing)
    }

    pub fn recv(&self) -> Result<VisualizerClientMessage, VisualizerProtocolError> {
        self.incoming
            .recv()
            .map_err(|_| VisualizerProtocolError::Disconnected)
    }

    pub fn recv_timeout(
        &self,
        timeout: Duration,
    ) -> Result<Option<VisualizerClientMessage>, VisualizerProtocolError> {
        match self.incoming.recv_timeout(timeout) {
            Ok(message) => Ok(Some(message)),
            Err(RecvTimeoutError::Timeout) => Ok(None),
            Err(RecvTimeoutError::Disconnected) => Err(VisualizerProtocolError::Disconnected),
        }
    }

    pub fn try_recv(&self) -> Result<Option<VisualizerClientMessage>, VisualizerProtocolError> {
        match self.incoming.try_recv() {
            Ok(message) => Ok(Some(message)),
            Err(TryRecvError::Empty) => Ok(None),
            Err(TryRecvError::Disconnected) => Err(VisualizerProtocolError::Disconnected),
        }
    }
}

pub struct VisualizerClientPort {
    incoming: Receiver<VisualizerServerMessage>,
    outgoing: Sender<VisualizerClientMessage>,
}

impl VisualizerClientPort {
    pub fn connect(addr: impl ToSocketAddrs) -> Result<Self, VisualizerProtocolError> {
        Self::from_stream(TcpStream::connect(addr)?)
    }

    pub fn from_stream(stream: TcpStream) -> Result<Self, VisualizerProtocolError> {
        stream.set_nonblocking(false)?;
        let (incoming, outgoing) =
            spawn_connection_threads::<VisualizerServerMessage, VisualizerClientMessage>(stream)?;
        Ok(Self { incoming, outgoing })
    }

    pub fn send(&self, message: VisualizerClientMessage) -> Result<(), VisualizerProtocolError> {
        self.outgoing
            .send(message)
            .map_err(|_| VisualizerProtocolError::Disconnected)
    }

    pub fn into_channels(
        self,
    ) -> (
        Receiver<VisualizerServerMessage>,
        Sender<VisualizerClientMessage>,
    ) {
        (self.incoming, self.outgoing)
    }
}

fn spawn_connection_threads<Incoming, Outgoing>(
    stream: TcpStream,
) -> Result<(Receiver<Incoming>, Sender<Outgoing>), VisualizerProtocolError>
where
    Incoming: DeserializeOwned + Send + 'static,
    Outgoing: Serialize + Send + 'static,
{
    let read_stream = stream.try_clone()?;
    let write_stream = stream;
    let (incoming_tx, incoming_rx) = mpsc::channel();
    let (outgoing_tx, outgoing_rx) = mpsc::channel();

    thread::spawn(move || {
        let mut reader = BufReader::new(read_stream);
        loop {
            let mut line = String::new();
            match reader.read_line(&mut line) {
                Ok(0) => {
                    eprintln!("visualizer protocol read loop ended: eof");
                    break;
                }
                Ok(_) => match serde_json::from_str::<Incoming>(&line) {
                    Ok(message) => {
                        if incoming_tx.send(message).is_err() {
                            eprintln!("visualizer protocol read loop ended: receiver dropped");
                            break;
                        }
                    }
                    Err(error) => {
                        eprintln!(
                            "visualizer protocol read loop ended: invalid json: {error}; line={line:?}"
                        );
                        break;
                    }
                },
                Err(error) if error.kind() == ErrorKind::WouldBlock => {
                    thread::sleep(Duration::from_millis(10));
                }
                Err(error) => {
                    eprintln!("visualizer protocol read loop ended: io error: {error}");
                    break;
                }
            }
        }
    });

    thread::spawn(move || {
        let mut writer = BufWriter::new(write_stream);
        while let Ok(message) = outgoing_rx.recv() {
            if write_json_line(&mut writer, &message).is_err() {
                eprintln!("visualizer protocol write loop ended: write failed");
                break;
            }
        }
        eprintln!("visualizer protocol write loop ended: sender dropped");
    });

    Ok((incoming_rx, outgoing_tx))
}

fn write_json_line(
    writer: &mut BufWriter<TcpStream>,
    message: &impl Serialize,
) -> Result<(), VisualizerProtocolError> {
    serde_json::to_writer(&mut *writer, message)?;
    writer.write_all(b"\n")?;
    writer.flush()?;
    Ok(())
}

pub fn memory_page_from_records(
    records: &[MemoryRecordView],
    page: usize,
    per_page: usize,
) -> MemoryPage {
    let total = records.len();
    let start = page.saturating_mul(per_page).min(records.len());
    let end = start.saturating_add(per_page).min(records.len());
    MemoryPage {
        page,
        per_page,
        total,
        records: records[start..end].to_vec(),
    }
}

pub type MemoryCache = BTreeMap<String, Vec<MemoryRecordView>>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn messages_round_trip_through_json() {
        let tab_id = VisualizerTabId::new("case-1");
        let message = VisualizerServerMessage::OfferAction {
            action: VisualizerAction::start_activation(tab_id.clone()),
        };

        let json = serde_json::to_string(&message).unwrap();
        let actual: VisualizerServerMessage = serde_json::from_str(&json).unwrap();

        let VisualizerServerMessage::OfferAction { action } = actual else {
            panic!("expected action");
        };
        assert_eq!(action.id, start_activation_action_id(&tab_id));
        assert_eq!(action.scope, VisualizerActionScope::Tab { tab_id });
        assert_eq!(action.kind, VisualizerActionKind::StartActivation);
    }

    #[test]
    fn ambient_rows_and_module_commands_round_trip_through_json() {
        let tab_id = VisualizerTabId::new("live");
        let row = AmbientSensoryRowView {
            id: "ambient-1".to_string(),
            modality: "smell".to_string(),
            content: "wet stone smell".to_string(),
            disabled: false,
        };
        let message = VisualizerServerMessage::event(VisualizerEvent::AmbientSensoryRows {
            tab_id: tab_id.clone(),
            rows: vec![row.clone()],
        });
        let json = serde_json::to_string(&message).unwrap();
        let actual: VisualizerServerMessage = serde_json::from_str(&json).unwrap();
        assert!(matches!(
            actual,
            VisualizerServerMessage::Event {
                event: VisualizerEvent::AmbientSensoryRows { rows, .. },
            } if rows == vec![row]
        ));

        let command = VisualizerClientMessage::Command {
            command: VisualizerCommand::SetModuleDisabled {
                tab_id,
                module: "predict".to_string(),
                disabled: true,
            },
        };
        let json = serde_json::to_string(&command).unwrap();
        let actual: VisualizerClientMessage = serde_json::from_str(&json).unwrap();
        assert!(matches!(
            actual,
            VisualizerClientMessage::Command {
                command: VisualizerCommand::SetModuleDisabled {
                    module,
                    disabled: true,
                    ..
                },
            } if module == "predict"
        ));

        let command = VisualizerClientMessage::Command {
            command: VisualizerCommand::SendOneShotSensoryInput {
                tab_id: VisualizerTabId::new("live"),
                input: OneShotSensoryInput {
                    modality: "touch".to_string(),
                    content: "the tabletop feels cold".to_string(),
                },
            },
        };
        let json = serde_json::to_string(&command).unwrap();
        let actual: VisualizerClientMessage = serde_json::from_str(&json).unwrap();
        assert!(matches!(
            actual,
            VisualizerClientMessage::Command {
                command: VisualizerCommand::SendOneShotSensoryInput { input, .. },
            } if input.modality == "touch" && input.content == "the tabletop feels cold"
        ));

        let command = VisualizerClientMessage::Command {
            command: VisualizerCommand::SetModuleSettings {
                tab_id: VisualizerTabId::new("live"),
                settings: ModuleSettingsView {
                    module: "predict".to_string(),
                    replica_min: 0,
                    replica_max: 2,
                    bpm_min: 3.0,
                    bpm_max: 12.0,
                    zero_replica_window: ZeroReplicaWindowView::EveryControllerActivations {
                        period: 4,
                    },
                },
            },
        };
        let json = serde_json::to_string(&command).unwrap();
        let actual: VisualizerClientMessage = serde_json::from_str(&json).unwrap();
        assert!(matches!(
            actual,
            VisualizerClientMessage::Command {
                command: VisualizerCommand::SetModuleSettings {
                    settings:
                        ModuleSettingsView {
                            module,
                            replica_max: 2,
                            zero_replica_window:
                                ZeroReplicaWindowView::EveryControllerActivations { period: 4 },
                            ..
                        },
                    ..
                },
            } if module == "predict"
        ));
    }

    #[test]
    fn tcp_loopback_streams_events_and_actions() {
        let listener = TcpListener::bind(("127.0.0.1", 0)).unwrap();
        let addr = listener.local_addr().unwrap();
        let server = thread::spawn(move || {
            let port = VisualizerServerPort::accept(listener).unwrap();
            port.send(VisualizerServerMessage::hello()).unwrap();
            let client = port.recv().unwrap();
            assert!(matches!(
                client,
                VisualizerClientMessage::Hello { version: 1 }
            ));
            let tab_id = VisualizerTabId::new("case-1");
            port.send(VisualizerServerMessage::event(VisualizerEvent::OpenTab {
                tab_id: tab_id.clone(),
                title: "case-1".to_string(),
            }))
            .unwrap();
            let message = port.recv().unwrap();
            assert!(matches!(
                message,
                VisualizerClientMessage::InvokeAction { action_id }
                    if action_id == start_activation_action_id(&tab_id)
            ));
        });

        let client = VisualizerClientPort::connect(addr).unwrap();
        client.send(VisualizerClientMessage::hello()).unwrap();
        let (incoming, outgoing) = client.into_channels();
        assert!(matches!(
            incoming.recv().unwrap(),
            VisualizerServerMessage::Hello { version: 1 }
        ));
        let event = incoming.recv().unwrap();
        let VisualizerServerMessage::Event {
            event: VisualizerEvent::OpenTab { tab_id, title },
        } = event
        else {
            panic!("expected open tab event");
        };
        assert_eq!(title, "case-1");
        outgoing
            .send(VisualizerClientMessage::InvokeAction {
                action_id: start_activation_action_id(&tab_id),
            })
            .unwrap();

        server.join().unwrap();
    }

    #[test]
    fn tcp_ports_normalize_nonblocking_streams() {
        let listener = TcpListener::bind(("127.0.0.1", 0)).unwrap();
        let addr = listener.local_addr().unwrap();
        let server = thread::spawn(move || {
            let (stream, _) = listener.accept().unwrap();
            stream.set_nonblocking(true).unwrap();
            let port = VisualizerServerPort::from_stream(stream).unwrap();
            port.send(VisualizerServerMessage::hello()).unwrap();
            let message = port.recv_timeout(Duration::from_secs(1)).unwrap();
            assert!(matches!(
                message,
                Some(VisualizerClientMessage::Hello { version: 1 })
            ));
        });

        let stream = TcpStream::connect(addr).unwrap();
        stream.set_nonblocking(true).unwrap();
        let client = VisualizerClientPort::from_stream(stream).unwrap();
        client.send(VisualizerClientMessage::hello()).unwrap();
        let (incoming, _) = client.into_channels();
        assert!(matches!(
            incoming.recv_timeout(Duration::from_secs(1)).unwrap(),
            VisualizerServerMessage::Hello { version: 1 }
        ));

        server.join().unwrap();
    }
}
