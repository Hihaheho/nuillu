use std::{
    io::{self, BufRead, BufReader, BufWriter, ErrorKind, Write},
    net::{TcpListener, TcpStream, ToSocketAddrs},
    sync::mpsc::{self, Receiver, RecvTimeoutError, Sender, TryRecvError},
    thread,
    time::Duration,
};

use chrono::{DateTime, Utc};
use nuillu_module::{AmbientSensoryEntry, RuntimeEvent, SensoryInput};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use thiserror::Error;

pub const VISUALIZER_PROTOCOL_VERSION: u32 = 3;
pub const START_SUITE_ACTION_ID: &str = "suite:start";

pub fn start_activation_action_id(tab_id: &VisualizerTabId) -> String {
    format!("tab:{}:start-activation", tab_id.as_str())
}

pub fn run_runtime_action_id(tab_id: &VisualizerTabId) -> String {
    format!("tab:{}:run-runtime", tab_id.as_str())
}

pub fn stop_runtime_action_id(tab_id: &VisualizerTabId) -> String {
    format!("tab:{}:stop-runtime", tab_id.as_str())
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

    pub fn run_runtime(tab_id: VisualizerTabId) -> Self {
        Self {
            id: run_runtime_action_id(&tab_id),
            label: "Run".to_string(),
            scope: VisualizerActionScope::Tab { tab_id },
            kind: VisualizerActionKind::RunRuntime,
        }
    }

    pub fn stop_runtime(tab_id: VisualizerTabId) -> Self {
        Self {
            id: stop_runtime_action_id(&tab_id),
            label: "Stop".to_string(),
            scope: VisualizerActionScope::Tab { tab_id },
            kind: VisualizerActionKind::StopRuntime,
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
    RunRuntime,
    StopRuntime,
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
    Error {
        tab_id: VisualizerTabId,
        error: VisualizerErrorView,
    },
    LlmObserved {
        tab_id: VisualizerTabId,
        event: LlmObservationEvent,
    },
    LlmTranscriptSnapshot {
        tab_id: VisualizerTabId,
        turns: Vec<LlmTranscriptTurnView>,
    },
    BlackboardSnapshot {
        tab_id: VisualizerTabId,
        snapshot: BlackboardSnapshot,
    },
    MemoryRecordsLoaded {
        tab_id: VisualizerTabId,
        scope: MemoryRecordScope,
        offset: usize,
        records: Vec<MemoryRecordView>,
        has_more: bool,
    },
    LinkedMemoryRecordsLoaded {
        tab_id: VisualizerTabId,
        memory_index: String,
        offset: usize,
        records: Vec<LinkedMemoryRecordView>,
        has_more: bool,
    },
    MemoryDeleted {
        tab_id: VisualizerTabId,
        memory_index: String,
    },
    AmbientSensoryRows {
        tab_id: VisualizerTabId,
        rows: Vec<AmbientSensoryRowView>,
    },
    OneShotSensoryInputRows {
        tab_id: VisualizerTabId,
        rows: Vec<OneShotSensoryInputRowView>,
    },
    OneShotSensoryInputAppended {
        tab_id: VisualizerTabId,
        row: OneShotSensoryInputRowView,
    },
    AmbientSensorySnapshotRows {
        tab_id: VisualizerTabId,
        rows: Vec<AmbientSensorySnapshotRowView>,
    },
    AmbientSensorySnapshotAppended {
        tab_id: VisualizerTabId,
        row: AmbientSensorySnapshotRowView,
    },
    UtteranceEventRows {
        tab_id: VisualizerTabId,
        rows: Vec<UtteranceEventRowView>,
    },
    UtteranceEventAppended {
        tab_id: VisualizerTabId,
        row: UtteranceEventRowView,
    },
    SceneState {
        tab_id: VisualizerTabId,
        state: SceneStateView,
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
    CreateSceneRow {
        tab_id: VisualizerTabId,
        kind: SceneRowKind,
    },
    UpdateSceneRow {
        tab_id: VisualizerTabId,
        row: SceneRowView,
    },
    RemoveSceneRow {
        tab_id: VisualizerTabId,
        kind: SceneRowKind,
        row_id: String,
    },
    SaveSceneState {
        tab_id: VisualizerTabId,
        state: EditableSceneStateView,
    },
    SendScenePersonMessage {
        tab_id: VisualizerTabId,
        row_id: String,
        message: String,
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
    ResetModuleSessionHistory {
        tab_id: VisualizerTabId,
        owner: String,
    },
    LoadMemoryRecords {
        tab_id: VisualizerTabId,
        scope: MemoryRecordScope,
        offset: usize,
        limit: usize,
    },
    LoadLinkedMemories {
        tab_id: VisualizerTabId,
        memory_index: String,
        relation_filter: Vec<String>,
        offset: usize,
        limit: usize,
    },
    DeleteMemory {
        tab_id: VisualizerTabId,
        memory_index: String,
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
        #[serde(default)]
        session_key: Option<String>,
        operation: String,
        activation_id: u64,
        activation_attempt: u32,
        batch: LlmBatchDebugView,
        items: Vec<LlmInputItemView>,
    },
    StreamStarted {
        turn_id: String,
        owner: String,
        module: String,
        replica: u8,
        tier: String,
        source: LlmObservationSource,
        #[serde(default)]
        session_key: Option<String>,
        operation: String,
        activation_id: u64,
        activation_attempt: u32,
        batch: LlmBatchDebugView,
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
    Failed {
        turn_id: String,
        message: String,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmTranscriptTurnView {
    pub turn_id: String,
    pub owner: String,
    pub module: String,
    pub replica: u8,
    pub tier: String,
    pub source: LlmObservationSource,
    #[serde(default)]
    pub session_key: Option<String>,
    pub operation: String,
    pub activation_id: u64,
    pub activation_attempt: u32,
    pub batch: LlmBatchDebugView,
    pub input: Vec<LlmInputItemView>,
    pub output: Vec<LlmOutputItemView>,
    pub request_id: Option<String>,
    pub model: Option<String>,
    pub finish_reason: Option<String>,
    pub usage: Option<LlmUsageView>,
    pub status: LlmTranscriptTurnStatus,
    pub error_message: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LlmBatchDebugView {
    pub batch_type: String,
    pub debug: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LlmTranscriptTurnStatus {
    Completed,
    Failed,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LlmOutputItemView {
    pub kind: String,
    pub content: String,
    pub source: Option<String>,
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

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VisualizerErrorView {
    pub at: DateTime<Utc>,
    pub source: String,
    pub phase: String,
    pub owner: Option<String>,
    pub message: String,
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub direction: Option<String>,
    pub content: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AmbientSensoryRowView {
    pub id: String,
    pub modality: String,
    pub content: String,
    pub disabled: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct SceneStateView {
    pub people: Vec<ScenePersonRowView>,
    pub objects: Vec<SceneObjectRowView>,
    pub sounds: Vec<SceneSoundRowView>,
    pub atmosphere: Vec<SceneAtmosphereRowView>,
    pub derived_ambient: Vec<DerivedAmbientSensoryRowView>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct EditableSceneStateView {
    pub people: Vec<ScenePersonRowView>,
    pub objects: Vec<SceneObjectRowView>,
    pub sounds: Vec<SceneSoundRowView>,
    pub atmosphere: Vec<SceneAtmosphereRowView>,
}

impl EditableSceneStateView {
    pub fn into_scene_state(self) -> SceneStateView {
        let derived_ambient = derive_scene_ambient(&self);
        SceneStateView {
            people: self.people,
            objects: self.objects,
            sounds: self.sounds,
            atmosphere: self.atmosphere,
            derived_ambient,
        }
    }
}

impl From<&SceneStateView> for EditableSceneStateView {
    fn from(state: &SceneStateView) -> Self {
        Self {
            people: state.people.clone(),
            objects: state.objects.clone(),
            sounds: state.sounds.clone(),
            atmosphere: state.atmosphere.clone(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SceneRowKind {
    Person,
    Object,
    Sound,
    Atmosphere,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", content = "row", rename_all = "snake_case")]
pub enum SceneRowView {
    Person(ScenePersonRowView),
    Object(SceneObjectRowView),
    Sound(SceneSoundRowView),
    Atmosphere(SceneAtmosphereRowView),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ScenePersonRowView {
    pub id: String,
    pub name: String,
    pub direction: String,
    pub distance: String,
    pub state: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SceneObjectRowView {
    pub id: String,
    pub name: String,
    pub direction: String,
    pub distance: String,
    pub visual_description: String,
    pub sound_description: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SceneSoundRowView {
    pub id: String,
    pub direction: String,
    pub distance: String,
    pub description: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SceneAtmosphereRowView {
    pub id: String,
    pub aspect: String,
    pub description: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DerivedAmbientSensoryRowView {
    pub id: String,
    pub modality: String,
    pub content: String,
}

pub fn derive_scene_ambient(state: &EditableSceneStateView) -> Vec<DerivedAmbientSensoryRowView> {
    let mut rows = Vec::new();
    for person in &state.people {
        if let Some(content) = derive_person_content(person) {
            rows.push(DerivedAmbientSensoryRowView {
                id: format!("scene:person:{}", person.id),
                modality: "vision".to_string(),
                content,
            });
        }
    }
    for object in &state.objects {
        if let Some(content) = derive_object_visual_content(object) {
            rows.push(DerivedAmbientSensoryRowView {
                id: format!("scene:object:{}:visual", object.id),
                modality: "vision".to_string(),
                content,
            });
        }
        if let Some(content) = derive_object_sound_content(object) {
            rows.push(DerivedAmbientSensoryRowView {
                id: format!("scene:object:{}:sound", object.id),
                modality: "audition".to_string(),
                content,
            });
        }
    }
    for sound in &state.sounds {
        if let Some(content) = derive_sound_content(sound) {
            rows.push(DerivedAmbientSensoryRowView {
                id: format!("scene:sound:{}", sound.id),
                modality: "audition".to_string(),
                content,
            });
        }
    }
    for atmosphere in &state.atmosphere {
        if let Some(content) = derive_atmosphere_content(atmosphere) {
            rows.push(DerivedAmbientSensoryRowView {
                id: format!("scene:atmosphere:{}", atmosphere.id),
                modality: atmosphere_modality(&atmosphere.aspect).to_string(),
                content,
            });
        }
    }
    rows
}

pub fn atmosphere_modality(aspect: &str) -> &'static str {
    match aspect.trim().to_ascii_lowercase().as_str() {
        "light" => "vision",
        "smell" => "smell",
        "temperature" | "air/weather" | "surface/feel" => "touch",
        _ => "ambient",
    }
}

fn derive_person_content(row: &ScenePersonRowView) -> Option<String> {
    let name = row.name.trim();
    if name.is_empty() {
        return None;
    }
    let mut content = format!("{name} is present");
    append_location(&mut content, &row.direction, &row.distance);
    append_sentence_tail(&mut content, &row.state);
    Some(content)
}

fn derive_object_visual_content(row: &SceneObjectRowView) -> Option<String> {
    let name = row.name.trim();
    let visual = row.visual_description.trim();
    if name.is_empty() && visual.is_empty() {
        return None;
    }
    let mut content = if name.is_empty() {
        "An object is visible".to_string()
    } else {
        format!("{name} is visible")
    };
    append_location(&mut content, &row.direction, &row.distance);
    append_sentence_tail(&mut content, visual);
    Some(content)
}

fn derive_object_sound_content(row: &SceneObjectRowView) -> Option<String> {
    let sound = row.sound_description.trim();
    if sound.is_empty() {
        return None;
    }
    let name = row.name.trim();
    let mut content = if name.is_empty() {
        "An object is making sound".to_string()
    } else {
        format!("{name} is making sound")
    };
    append_location(&mut content, &row.direction, &row.distance);
    append_sentence_tail(&mut content, sound);
    Some(content)
}

fn derive_sound_content(row: &SceneSoundRowView) -> Option<String> {
    let description = row.description.trim();
    if description.is_empty() {
        return None;
    }
    let mut content = "A sound is present".to_string();
    append_sound_location(&mut content, &row.direction, &row.distance);
    append_sentence_tail(&mut content, description);
    Some(content)
}

fn derive_atmosphere_content(row: &SceneAtmosphereRowView) -> Option<String> {
    let description = row.description.trim();
    if description.is_empty() {
        return None;
    }
    let aspect = row.aspect.trim();
    if aspect.is_empty() || aspect == "other" {
        Some(description.to_string())
    } else {
        Some(format!("{aspect}: {description}"))
    }
}

fn append_location(content: &mut String, direction: &str, distance: &str) {
    let direction = direction.trim();
    let distance = distance.trim();
    match (direction.is_empty(), distance.is_empty()) {
        (false, false) => content.push_str(&format!(" at {direction}, {distance} away")),
        (false, true) => content.push_str(&format!(" at {direction}")),
        (true, false) => content.push_str(&format!(" {distance} away")),
        (true, true) => {}
    }
}

fn append_sound_location(content: &mut String, direction: &str, distance: &str) {
    let direction = direction.trim();
    let distance = distance.trim();
    match (direction.is_empty(), distance.is_empty()) {
        (false, false) => content.push_str(&format!(" from {direction}, {distance} away")),
        (false, true) => content.push_str(&format!(" from {direction}")),
        (true, false) => content.push_str(&format!(" {distance} away")),
        (true, true) => {}
    }
}

fn append_sentence_tail(content: &mut String, tail: &str) {
    let tail = tail.trim();
    if tail.is_empty() {
        content.push('.');
    } else {
        content.push_str("; ");
        content.push_str(tail);
        content.push('.');
    }
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub generation_id: Option<u64>,
    pub text: String,
    pub emitted_at: DateTime<Utc>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OneShotSensoryInputRowView {
    pub id: i64,
    pub server_session_id: String,
    pub modality: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub direction: Option<String>,
    pub content: String,
    pub observed_at: DateTime<Utc>,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AmbientSensorySnapshotRowView {
    pub id: i64,
    pub server_session_id: String,
    pub entries: Vec<AmbientSensoryEntry>,
    pub observed_at: DateTime<Utc>,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UtteranceEventKindView {
    Delta,
    Completed,
    Aborted,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct UtteranceEventRowView {
    pub id: i64,
    pub server_session_id: String,
    pub event_kind: UtteranceEventKindView,
    pub sender: String,
    pub target: String,
    pub generation_id: u64,
    pub sequence: u32,
    pub content: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
    pub occurred_at: DateTime<Utc>,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BlackboardSnapshot {
    pub module_statuses: Vec<ModuleStatusView>,
    pub allocation: Vec<AllocationView>,
    #[serde(default)]
    pub interoception: InteroceptionView,
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
    pub period_ms: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InteroceptionView {
    pub mode: String,
    pub wake_arousal: f32,
    pub nrem_pressure: f32,
    pub rem_pressure: f32,
    pub affect_arousal: f32,
    pub valence: f32,
    pub emotion: String,
    pub last_updated: DateTime<Utc>,
}

impl Default for InteroceptionView {
    fn default() -> Self {
        Self {
            mode: "wake".to_string(),
            wake_arousal: 0.0,
            nrem_pressure: 0.0,
            rem_pressure: 0.0,
            affect_arousal: 0.0,
            valence: 0.0,
            emotion: String::new(),
            last_updated: DateTime::<Utc>::from_timestamp(0, 0)
                .expect("unix epoch timestamp is valid"),
        }
    }
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
    #[serde(default)]
    pub cognitive: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CognitionLogView {
    pub source: String,
    pub entries: Vec<CognitionEntryView>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CognitionEntryView {
    pub at: DateTime<Utc>,
    pub origin: String,
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

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MemoryMetadataView {
    pub index: String,
    pub rank: String,
    pub occurred_at: Option<DateTime<Utc>>,
    pub last_accessed: DateTime<Utc>,
    pub access_count: u32,
    pub use_count: u32,
    pub reinforcement_count: u32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
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

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MemoryConceptView {
    pub label: String,
    pub mention_text: Option<String>,
    pub loose_type: Option<String>,
    pub confidence: f32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MemoryTagView {
    pub label: String,
    pub namespace: String,
    pub confidence: f32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MemoryLinkView {
    pub from_memory: String,
    pub to_memory: String,
    pub relation: String,
    pub freeform_relation: Option<String>,
    pub strength: f32,
    pub confidence: f32,
    pub updated_at: DateTime<Utc>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LinkedMemoryRecordView {
    pub record: MemoryRecordView,
    pub link: MemoryLinkView,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "scope", rename_all = "snake_case")]
pub enum MemoryRecordScope {
    Latest,
    Search { query: String },
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
    fn run_stop_actions_round_trip_through_json() {
        let tab_id = VisualizerTabId::new("live");
        for (action, expected_id, expected_kind) in [
            (
                VisualizerAction::run_runtime(tab_id.clone()),
                run_runtime_action_id(&tab_id),
                VisualizerActionKind::RunRuntime,
            ),
            (
                VisualizerAction::stop_runtime(tab_id.clone()),
                stop_runtime_action_id(&tab_id),
                VisualizerActionKind::StopRuntime,
            ),
        ] {
            let message = VisualizerServerMessage::OfferAction { action };
            let json = serde_json::to_string(&message).unwrap();
            let actual: VisualizerServerMessage = serde_json::from_str(&json).unwrap();

            let VisualizerServerMessage::OfferAction { action } = actual else {
                panic!("expected action");
            };
            assert_eq!(action.id, expected_id);
            assert_eq!(
                action.scope,
                VisualizerActionScope::Tab {
                    tab_id: tab_id.clone()
                }
            );
            assert_eq!(action.kind, expected_kind);
        }
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
                    direction: Some("tabletop".to_string()),
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
            } if input.modality == "touch"
                && input.direction.as_deref() == Some("tabletop")
                && input.content == "the tabletop feels cold"
        ));

        let scene = SceneStateView {
            people: vec![ScenePersonRowView {
                id: "person-1".to_string(),
                name: "Pibi".to_string(),
                direction: "front".to_string(),
                distance: "2m".to_string(),
                state: "watching Nui".to_string(),
            }],
            objects: vec![SceneObjectRowView {
                id: "object-1".to_string(),
                name: "bowl".to_string(),
                direction: "left".to_string(),
                distance: "1m".to_string(),
                visual_description: "red bowl".to_string(),
                sound_description: "soft rattling".to_string(),
            }],
            sounds: vec![SceneSoundRowView {
                id: "sound-1".to_string(),
                direction: "behind".to_string(),
                distance: "far".to_string(),
                description: "rain tapping".to_string(),
            }],
            atmosphere: vec![SceneAtmosphereRowView {
                id: "atmosphere-1".to_string(),
                aspect: "light".to_string(),
                description: "dim yellow light".to_string(),
            }],
            derived_ambient: vec![DerivedAmbientSensoryRowView {
                id: "scene:person:person-1".to_string(),
                modality: "vision".to_string(),
                content: "Pibi is present at front, 2m away; watching Nui.".to_string(),
            }],
        };
        let message = VisualizerServerMessage::event(VisualizerEvent::SceneState {
            tab_id: VisualizerTabId::new("live"),
            state: scene.clone(),
        });
        let json = serde_json::to_string(&message).unwrap();
        let actual: VisualizerServerMessage = serde_json::from_str(&json).unwrap();
        assert!(matches!(
            actual,
            VisualizerServerMessage::Event {
                event: VisualizerEvent::SceneState { state, .. },
            } if state == scene
        ));

        let command = VisualizerClientMessage::Command {
            command: VisualizerCommand::UpdateSceneRow {
                tab_id: VisualizerTabId::new("live"),
                row: SceneRowView::Person(ScenePersonRowView {
                    id: "person-1".to_string(),
                    name: "Pibi".to_string(),
                    direction: "front".to_string(),
                    distance: "2m".to_string(),
                    state: "watching Nui".to_string(),
                }),
            },
        };
        let json = serde_json::to_string(&command).unwrap();
        let actual: VisualizerClientMessage = serde_json::from_str(&json).unwrap();
        assert!(matches!(
            actual,
            VisualizerClientMessage::Command {
                command: VisualizerCommand::UpdateSceneRow {
                    row: SceneRowView::Person(row),
                    ..
                },
            } if row.name == "Pibi"
        ));

        let command = VisualizerClientMessage::Command {
            command: VisualizerCommand::SaveSceneState {
                tab_id: VisualizerTabId::new("live"),
                state: EditableSceneStateView::from(&scene),
            },
        };
        let json = serde_json::to_string(&command).unwrap();
        let actual: VisualizerClientMessage = serde_json::from_str(&json).unwrap();
        assert!(matches!(
            actual,
            VisualizerClientMessage::Command {
                command: VisualizerCommand::SaveSceneState { state, .. },
            } if state.people[0].name == "Pibi"
                && state.objects[0].name == "bowl"
                && state.sounds[0].description == "rain tapping"
                && state.atmosphere[0].description == "dim yellow light"
        ));

        let command = VisualizerClientMessage::Command {
            command: VisualizerCommand::SendScenePersonMessage {
                tab_id: VisualizerTabId::new("live"),
                row_id: "person-1".to_string(),
                message: "hello".to_string(),
            },
        };
        let json = serde_json::to_string(&command).unwrap();
        let actual: VisualizerClientMessage = serde_json::from_str(&json).unwrap();
        assert!(matches!(
            actual,
            VisualizerClientMessage::Command {
                command: VisualizerCommand::SendScenePersonMessage {
                    row_id,
                    message,
                    ..
                },
            } if row_id == "person-1" && message == "hello"
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

        let command = VisualizerClientMessage::Command {
            command: VisualizerCommand::ResetModuleSessionHistory {
                tab_id: VisualizerTabId::new("live"),
                owner: "predict[1]".to_string(),
            },
        };
        let json = serde_json::to_string(&command).unwrap();
        let actual: VisualizerClientMessage = serde_json::from_str(&json).unwrap();
        assert!(matches!(
            actual,
            VisualizerClientMessage::Command {
                command: VisualizerCommand::ResetModuleSessionHistory { owner, .. },
            } if owner == "predict[1]"
        ));
    }

    #[test]
    fn raw_recent_activity_rows_round_trip_through_json() {
        let at = DateTime::<Utc>::from_timestamp(1_780_000_000, 0).unwrap();
        let one_shot = OneShotSensoryInputRowView {
            id: 1,
            server_session_id: "server-session".to_string(),
            modality: "audition".to_string(),
            direction: Some("Koro".to_string()),
            content: "Koro says, \"wait\"".to_string(),
            observed_at: at,
            created_at: at,
        };
        let ambient = AmbientSensorySnapshotRowView {
            id: 2,
            server_session_id: "server-session".to_string(),
            entries: vec![AmbientSensoryEntry {
                id: "scene:person:koro".to_string(),
                modality: nuillu_module::SensoryModality::parse("vision"),
                content: "Koro waits nearby.".to_string(),
            }],
            observed_at: at,
            created_at: at,
        };
        let utterance = UtteranceEventRowView {
            id: 3,
            server_session_id: "server-session".to_string(),
            event_kind: UtteranceEventKindView::Delta,
            sender: "speak".to_string(),
            target: "Koro".to_string(),
            generation_id: 7,
            sequence: 1,
            content: "stay".to_string(),
            reason: None,
            occurred_at: at,
            created_at: at,
        };

        let messages = [
            VisualizerServerMessage::event(VisualizerEvent::OneShotSensoryInputRows {
                tab_id: VisualizerTabId::new("live"),
                rows: vec![one_shot.clone()],
            }),
            VisualizerServerMessage::event(VisualizerEvent::AmbientSensorySnapshotAppended {
                tab_id: VisualizerTabId::new("live"),
                row: ambient.clone(),
            }),
            VisualizerServerMessage::event(VisualizerEvent::UtteranceEventAppended {
                tab_id: VisualizerTabId::new("live"),
                row: utterance.clone(),
            }),
        ];

        for message in messages {
            let json = serde_json::to_string(&message).unwrap();
            let actual: VisualizerServerMessage = serde_json::from_str(&json).unwrap();
            match actual {
                VisualizerServerMessage::Event {
                    event: VisualizerEvent::OneShotSensoryInputRows { rows, .. },
                } => assert_eq!(rows, vec![one_shot.clone()]),
                VisualizerServerMessage::Event {
                    event: VisualizerEvent::AmbientSensorySnapshotAppended { row, .. },
                } => assert_eq!(row, ambient.clone()),
                VisualizerServerMessage::Event {
                    event: VisualizerEvent::UtteranceEventAppended { row, .. },
                } => assert_eq!(row, utterance.clone()),
                other => panic!("unexpected raw activity row message: {other:?}"),
            }
        }
    }

    #[test]
    fn memory_chunks_round_trip_through_json() {
        let at = DateTime::<Utc>::from_timestamp(1_780_000_000, 0).unwrap();
        let record = MemoryRecordView {
            index: "m1".to_string(),
            kind: "Statement".to_string(),
            rank: "ShortTerm".to_string(),
            occurred_at: Some(at),
            stored_at: at,
            concepts: Vec::new(),
            tags: Vec::new(),
            affect_arousal: 0.0,
            valence: 0.0,
            emotion: String::new(),
            content: "remembered".to_string(),
        };
        let message = VisualizerServerMessage::event(VisualizerEvent::MemoryRecordsLoaded {
            tab_id: VisualizerTabId::new("live"),
            scope: MemoryRecordScope::Search {
                query: "remembered".to_string(),
            },
            offset: 50,
            records: vec![record.clone()],
            has_more: true,
        });

        let json = serde_json::to_string(&message).unwrap();
        let actual: VisualizerServerMessage = serde_json::from_str(&json).unwrap();

        assert!(matches!(
            actual,
            VisualizerServerMessage::Event {
                event: VisualizerEvent::MemoryRecordsLoaded {
                    scope: MemoryRecordScope::Search { query },
                    offset: 50,
                    records,
                    has_more: true,
                    ..
                },
            } if query == "remembered" && records == vec![record]
        ));
    }

    #[test]
    fn blackboard_snapshot_defaults_missing_interoception() {
        let json = r#"{
            "module_statuses": [],
            "allocation": [],
            "memos": [],
            "cognition_logs": [],
            "utterance_progresses": [],
            "memory_metadata": []
        }"#;

        let actual: BlackboardSnapshot = serde_json::from_str(json).unwrap();

        assert_eq!(actual.interoception, InteroceptionView::default());
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
                VisualizerClientMessage::Hello { version: 3 }
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
            VisualizerServerMessage::Hello { version: 3 }
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
                Some(VisualizerClientMessage::Hello { version: 3 })
            ));
        });

        let stream = TcpStream::connect(addr).unwrap();
        stream.set_nonblocking(true).unwrap();
        let client = VisualizerClientPort::from_stream(stream).unwrap();
        client.send(VisualizerClientMessage::hello()).unwrap();
        let (incoming, _) = client.into_channels();
        assert!(matches!(
            incoming.recv_timeout(Duration::from_secs(1)).unwrap(),
            VisualizerServerMessage::Hello { version: 3 }
        ));

        server.join().unwrap();
    }
}
