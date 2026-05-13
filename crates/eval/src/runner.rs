use std::{
    any::Any,
    collections::{BTreeMap, HashSet},
    fs::{File, OpenOptions},
    io::{self, Write},
    num::NonZeroUsize,
    panic::AssertUnwindSafe,
    path::{Path, PathBuf},
    sync::atomic::{AtomicBool, AtomicU64, Ordering},
    sync::{Arc, Mutex, OnceLock},
    time::{Duration, Instant},
};

use anyhow::{Context, Result};
use async_trait::async_trait;
use chrono::{DateTime, Duration as ChronoDuration, FixedOffset, Utc};
use futures::FutureExt as _;
use lutum::{
    AssistantInputItem, CompletionEvent, ErasedStructuredCompletionEvent,
    ErasedStructuredTurnEvent, ErasedTextTurnEvent, InputMessageRole, ItemView, Lutum,
    LutumHooksSet, LutumStreamEvent, MessageContent, ModelInputHookContext, ModelInputItem,
    ModelName, OnModelInput, OnStreamEvent, OperationKind, RawTelemetryConfig, RequestExtensions,
    SharedPoolBudgetManager, SharedPoolBudgetOptions, StreamEventHookContext, TurnRole, TurnView,
    Usage,
};
use lutum_eval::{RawTraceSnapshot, TraceSnapshot};
use lutum_in_memory_adapter::InMemoryCognitionLogRepository;
use lutum_libsql_adapter::{
    EmbeddingProfile, LibsqlMemoryStore, LibsqlMemoryStoreConfig, LibsqlPolicyStore,
    LibsqlPolicyStoreConfig,
};
use lutum_model2vec_adapter::PotionBase8MEmbedder;
use lutum_openai::{OpenAiAdapter, OpenAiReasoningEffort};
use nuillu_agent::{AgentEventLoopConfig, run as run_agent};
use nuillu_blackboard::{
    ActivationRatio, Blackboard, BlackboardCommand, BlackboardInner, Bpm, CognitionLogEntry,
    MemoLogRecord, MemoryMetadata, ModuleConfig, ModulePolicy, ResourceAllocation, linear_ratio_fn,
};
use nuillu_module::ports::{
    Clock, Embedder, FileSearchHit, FileSearchProvider, FileSearchQuery, MemoryQuery, MemoryStore,
    NoopFileSearchProvider, PolicyStore, PortError, SystemClock, Utterance, UtteranceDelta,
    UtteranceSink,
};
use nuillu_module::{
    AllocationUpdated, CapabilityProviderConfig, CapabilityProviderPorts,
    CapabilityProviderRuntime, CapabilityProviders, CognitionLogUpdated, InternalHarnessIo,
    LlmRequestMetadata, LlmRequestSource, LutumTiers, ModuleRegistry, Participant, RuntimeEvent,
    RuntimeEventSink, RuntimePolicy, SensoryInput, SensoryInputMailbox,
};
use nuillu_types::{
    MemoryRank, ModelTier, ModuleId, ModuleInstanceId, ReplicaCapRange, ReplicaIndex, builtin,
};
use nuillu_visualizer_protocol::{
    AllocationView, BlackboardSnapshot, ChatInputKind, CognitionEntryView, CognitionLogView,
    LlmInputItemView, LlmObservationEvent, LlmObservationSource, LlmUsageView, MemoView,
    MemoryMetadataView, MemoryPage, MemoryRecordView, ModuleStatusView, TabStatus,
    UtteranceDeltaView, UtteranceProgressView, UtteranceView, VisualizerAction,
    VisualizerClientMessage, VisualizerCommand, VisualizerEvent, VisualizerServerMessage,
    VisualizerTabId, start_activation_action_id,
};
use regex::RegexBuilder;
use serde::Serialize;
use thiserror::Error;
use tokio::task::LocalSet;
use tracing_subscriber::layer::SubscriberExt as _;

use crate::{
    artifact::CaseArtifact,
    cases::{
        ArtifactTextField, CaseFileError, Check, DEFAULT_FULL_AGENT_MODULES, EvalCase, EvalModule,
        EvalStep, FullAgentCase, FullAgentInput, ModuleCase, ModuleEvalTarget, WaitFor,
        discover_case_files, parse_case_file, parse_case_now, parse_memory_datetime,
    },
    evaluation::{
        CaseReport, CaseSummary, SuiteModelNames, SuiteReport, SuiteRunReport, artifact_text,
        evaluate_case, field_label, normalize_text_block, pointer_text,
    },
    judge::{LlmRubricJudge, RubricJudge},
    model_set::ReasoningEffort,
    state_dump::{
        AgenticDeadlockDump, AllocationModuleDump, AllocationProposalDump, BlackboardLastStateDump,
        CognitionEntryDump, CognitionLogDump, DumpText, FullAgentLastStateCaseDump,
        FullAgentLastStateDump, MemoLogDump, MemoryEntryDump, MemoryLastStateDump,
        MemoryMetadataDump, ModuleInstanceDump, ReplicaCapDump, UtteranceDump,
        render_full_agent_last_state_eure,
    },
    trace_json::{raw_trace_has_error, raw_trace_snapshot_json, trace_snapshot_json},
};

const IDLE_REPORT_INTERVAL: Duration = Duration::from_secs(30);
const FULL_AGENT_ACTION_SILENCE_WINDOW: Duration = Duration::from_secs(1);
const EVAL_POLL_INTERVAL: Duration = Duration::from_millis(100);
const EVAL_MEMO_RETAINED_PER_OWNER: usize = 256;

#[derive(Debug, Clone)]
pub struct LlmBackendConfig {
    pub endpoint: String,
    pub token: String,
    pub model: String,
    pub reasoning_effort: Option<ReasoningEffort>,
    pub use_responses_api: bool,
}

#[derive(Debug, Clone)]
pub struct RunnerConfig {
    pub cases_root: PathBuf,
    pub output_root: PathBuf,
    pub run_id: String,
    pub judge_backend: LlmBackendConfig,
    pub cheap_backend: LlmBackendConfig,
    pub default_backend: LlmBackendConfig,
    pub premium_backend: LlmBackendConfig,
    pub model_dir: PathBuf,
    pub fail_fast: bool,
    pub max_concurrent_llm_calls: Option<NonZeroUsize>,
    pub case_patterns: Vec<String>,
    pub disabled_modules: Vec<EvalModule>,
}

/// Modules that may never be disabled via `RunnerConfig::disabled_modules` —
/// removing them breaks the basic observe → cognize → speak pipeline that the
/// full-agent eval cases assume.
pub const REQUIRED_FULL_AGENT_MODULES: &[EvalModule] = &[
    EvalModule::AttentionController,
    EvalModule::Sensory,
    EvalModule::Speak,
];

pub struct RunnerHooks {
    pub visualizer: Option<VisualizerHook>,
}

impl RunnerHooks {
    pub fn none() -> Self {
        Self { visualizer: None }
    }

    pub fn with_visualizer(visualizer: VisualizerHook) -> Self {
        Self {
            visualizer: Some(visualizer),
        }
    }
}

pub struct VisualizerHook {
    events: std::sync::mpsc::Sender<VisualizerServerMessage>,
    commands: std::sync::mpsc::Receiver<VisualizerClientMessage>,
    memory_cache: BTreeMap<String, Vec<MemoryRecordView>>,
    shutdown_requested: bool,
}

#[derive(Clone, Debug)]
pub struct VisualizerEventSink {
    events: std::sync::mpsc::Sender<VisualizerServerMessage>,
}

impl VisualizerEventSink {
    fn new(events: std::sync::mpsc::Sender<VisualizerServerMessage>) -> Self {
        Self { events }
    }

    fn send(&self, event: VisualizerEvent) {
        let _ = self.events.send(VisualizerServerMessage::event(event));
    }
}

impl VisualizerHook {
    pub fn new(
        events: std::sync::mpsc::Sender<VisualizerServerMessage>,
        commands: std::sync::mpsc::Receiver<VisualizerClientMessage>,
    ) -> Self {
        Self {
            events,
            commands,
            memory_cache: BTreeMap::new(),
            shutdown_requested: false,
        }
    }

    pub fn event_sender(&self) -> VisualizerEventSink {
        VisualizerEventSink::new(self.events.clone())
    }

    fn send_event(&self, event: VisualizerEvent) {
        let _ = self.events.send(VisualizerServerMessage::event(event));
    }

    fn offer_action(&self, action: VisualizerAction) {
        let _ = self
            .events
            .send(VisualizerServerMessage::OfferAction { action });
    }

    fn revoke_action(&self, action_id: String) {
        let _ = self
            .events
            .send(VisualizerServerMessage::RevokeAction { action_id });
    }

    fn request_shutdown(&mut self) {
        self.shutdown_requested = true;
    }

    pub fn shutdown_requested(&self) -> bool {
        self.shutdown_requested
    }

    fn set_memory_cache(&mut self, case_id: &str, records: Vec<MemoryRecordView>) {
        self.memory_cache.insert(case_id.to_string(), records);
    }

    fn cached_memory_page(&self, case_id: &str, page: usize, per_page: usize) -> MemoryPage {
        memory_page_from_records(
            self.memory_cache
                .get(case_id)
                .map(Vec::as_slice)
                .unwrap_or_default(),
            page,
            per_page,
        )
    }

    fn drain_cached_commands_until_shutdown(&mut self) {
        while let Ok(message) = self.commands.recv() {
            let VisualizerClientMessage::Command { command } = message else {
                continue;
            };
            match command {
                VisualizerCommand::Shutdown => {
                    self.request_shutdown();
                    break;
                }
                VisualizerCommand::ListMemories {
                    tab_id,
                    page,
                    per_page,
                } => {
                    let page = self.cached_memory_page(tab_id.as_str(), page, per_page);
                    self.send_event(VisualizerEvent::MemoryPage { tab_id, page });
                }
                VisualizerCommand::QueryMemory {
                    tab_id,
                    query,
                    limit,
                } => {
                    let needle = query.to_lowercase();
                    let records = self
                        .memory_cache
                        .get(tab_id.as_str())
                        .map(Vec::as_slice)
                        .unwrap_or_default()
                        .iter()
                        .filter(|record| record.content.to_lowercase().contains(&needle))
                        .take(limit)
                        .cloned()
                        .collect();
                    self.send_event(VisualizerEvent::MemoryQueryResult {
                        tab_id,
                        query,
                        records,
                    });
                }
                VisualizerCommand::SendSensoryInput { tab_id, .. } => {
                    self.send_event(VisualizerEvent::Log {
                        tab_id,
                        message: "eval case is no longer running".to_string(),
                    });
                }
            }
        }
    }
}

#[derive(Clone)]
struct VisualizerLlmObserver {
    inner: Arc<VisualizerLlmObserverInner>,
}

struct VisualizerLlmObserverInner {
    case_id: String,
    events: VisualizerEventSink,
    next_turn: AtomicU64,
    extension_turns: Mutex<BTreeMap<usize, String>>,
}

impl VisualizerLlmObserver {
    fn new(case_id: String, events: VisualizerEventSink) -> Self {
        Self {
            inner: Arc::new(VisualizerLlmObserverInner {
                case_id,
                events,
                next_turn: AtomicU64::new(0),
                extension_turns: Mutex::new(BTreeMap::new()),
            }),
        }
    }

    fn hook_set(&self) -> LutumHooksSet<'static> {
        LutumHooksSet::new()
            .with_on_model_input(self.clone())
            .with_on_stream_event(self.clone())
    }

    fn metadata<'a>(&self, extensions: &'a RequestExtensions) -> Option<&'a LlmRequestMetadata> {
        extensions.get::<LlmRequestMetadata>()
    }

    fn turn_id_for(&self, extensions: &RequestExtensions) -> String {
        let key = extension_key(extensions);
        let mut turns = self
            .inner
            .extension_turns
            .lock()
            .expect("visualizer llm turn map lock poisoned");
        turns
            .entry(key)
            .or_insert_with(|| {
                let next = self.inner.next_turn.fetch_add(1, Ordering::Relaxed);
                format!("{}-llm-{next}", self.inner.case_id)
            })
            .clone()
    }

    fn clear_turn_for(&self, extensions: &RequestExtensions) {
        let key = extension_key(extensions);
        self.inner
            .extension_turns
            .lock()
            .expect("visualizer llm turn map lock poisoned")
            .remove(&key);
    }

    fn emit(&self, event: LlmObservationEvent) {
        self.inner.events.send(VisualizerEvent::LlmObserved {
            tab_id: VisualizerTabId::new(self.inner.case_id.clone()),
            event,
        });
    }
}

impl OnModelInput for VisualizerLlmObserver {
    async fn call(&self, cx: &ModelInputHookContext<'_>) {
        let Some(metadata) = self.metadata(cx.extensions()) else {
            return;
        };
        let turn_id = self.turn_id_for(cx.extensions());
        self.emit(LlmObservationEvent::ModelInput {
            turn_id,
            owner: metadata.owner.to_string(),
            module: metadata.owner.module.to_string(),
            replica: metadata.owner.replica.get(),
            tier: format!("{:?}", metadata.tier),
            source: observation_source(metadata.source),
            operation: operation_kind_label(cx.kind()).to_string(),
            items: model_input_views(cx.input().items()),
        });
    }
}

impl OnStreamEvent for VisualizerLlmObserver {
    async fn call(&self, cx: &StreamEventHookContext<'_>) {
        let Some(metadata) = self.metadata(cx.extensions()) else {
            return;
        };
        let turn_id = self.turn_id_for(cx.extensions());
        emit_stream_observation(self, cx, metadata, turn_id);
    }
}

fn extension_key(extensions: &RequestExtensions) -> usize {
    std::ptr::from_ref(extensions) as usize
}

fn emit_stream_observation(
    observer: &VisualizerLlmObserver,
    cx: &StreamEventHookContext<'_>,
    metadata: &LlmRequestMetadata,
    turn_id: String,
) {
    match cx.event() {
        LutumStreamEvent::TextTurn(event) => {
            emit_text_turn_stream_event(observer, cx.extensions(), metadata, turn_id, event);
        }
        LutumStreamEvent::StructuredTurn(event) => {
            emit_structured_turn_stream_event(observer, cx.extensions(), metadata, turn_id, event);
        }
        LutumStreamEvent::Completion(event) => {
            emit_completion_stream_event(observer, cx.extensions(), metadata, turn_id, event);
        }
        LutumStreamEvent::StructuredCompletion(event) => {
            emit_structured_completion_stream_event(
                observer,
                cx.extensions(),
                metadata,
                turn_id,
                event,
            );
        }
    }
}

fn emit_started(
    observer: &VisualizerLlmObserver,
    metadata: &LlmRequestMetadata,
    turn_id: String,
    operation: OperationKind,
    request_id: Option<String>,
    model: String,
) {
    observer.emit(LlmObservationEvent::StreamStarted {
        turn_id,
        owner: metadata.owner.to_string(),
        module: metadata.owner.module.to_string(),
        replica: metadata.owner.replica.get(),
        tier: format!("{:?}", metadata.tier),
        source: observation_source(metadata.source),
        operation: operation_kind_label(operation).to_string(),
        request_id,
        model,
    });
}

fn emit_delta(observer: &VisualizerLlmObserver, turn_id: String, kind: &str, delta: String) {
    observer.emit(LlmObservationEvent::StreamDelta {
        turn_id,
        kind: kind.to_string(),
        delta,
    });
}

fn emit_completed(
    observer: &VisualizerLlmObserver,
    extensions: &RequestExtensions,
    turn_id: String,
    request_id: Option<String>,
    finish_reason: impl std::fmt::Debug,
    usage: Usage,
) {
    observer.emit(LlmObservationEvent::Completed {
        turn_id,
        request_id,
        finish_reason: format!("{finish_reason:?}"),
        usage: usage_view(usage),
    });
    observer.clear_turn_for(extensions);
}

fn emit_tool_call_chunk(
    observer: &VisualizerLlmObserver,
    turn_id: String,
    id: &lutum::ToolCallId,
    name: &lutum::ToolName,
    arguments_json_delta: String,
) {
    observer.emit(LlmObservationEvent::ToolCallChunk {
        turn_id,
        id: id.as_str().to_string(),
        name: name.as_str().to_string(),
        arguments_json_delta,
    });
}

fn emit_tool_call_ready(
    observer: &VisualizerLlmObserver,
    turn_id: String,
    metadata: &lutum::ToolMetadata,
) {
    observer.emit(LlmObservationEvent::ToolCallReady {
        turn_id,
        id: metadata.id.as_str().to_string(),
        name: metadata.name.as_str().to_string(),
        arguments_json: metadata.arguments.to_string(),
    });
}

fn emit_text_turn_stream_event(
    observer: &VisualizerLlmObserver,
    extensions: &RequestExtensions,
    metadata: &LlmRequestMetadata,
    turn_id: String,
    event: &ErasedTextTurnEvent,
) {
    match event {
        ErasedTextTurnEvent::Started { request_id, model } => emit_started(
            observer,
            metadata,
            turn_id,
            OperationKind::TextTurn,
            request_id.clone(),
            model.clone(),
        ),
        ErasedTextTurnEvent::TextDelta { delta } => {
            emit_delta(observer, turn_id, "text", delta.clone());
        }
        ErasedTextTurnEvent::ReasoningDelta { delta } => {
            emit_delta(observer, turn_id, "reasoning", delta.clone());
        }
        ErasedTextTurnEvent::RefusalDelta { delta } => {
            emit_delta(observer, turn_id, "refusal", delta.clone());
        }
        ErasedTextTurnEvent::ToolCallChunk {
            id,
            name,
            arguments_json_delta,
        } => emit_tool_call_chunk(observer, turn_id, id, name, arguments_json_delta.clone()),
        ErasedTextTurnEvent::ToolCallReady(tool) => {
            emit_tool_call_ready(observer, turn_id, tool);
        }
        ErasedTextTurnEvent::Completed {
            request_id,
            finish_reason,
            usage,
            ..
        } => emit_completed(
            observer,
            extensions,
            turn_id,
            request_id.clone(),
            finish_reason,
            *usage,
        ),
    }
}

fn emit_structured_turn_stream_event(
    observer: &VisualizerLlmObserver,
    extensions: &RequestExtensions,
    metadata: &LlmRequestMetadata,
    turn_id: String,
    event: &ErasedStructuredTurnEvent,
) {
    match event {
        ErasedStructuredTurnEvent::Started { request_id, model } => emit_started(
            observer,
            metadata,
            turn_id,
            OperationKind::StructuredTurn,
            request_id.clone(),
            model.clone(),
        ),
        ErasedStructuredTurnEvent::StructuredOutputChunk { json_delta } => {
            emit_delta(observer, turn_id, "structured", json_delta.clone());
        }
        ErasedStructuredTurnEvent::StructuredOutputReady(json) => {
            observer.emit(LlmObservationEvent::StructuredReady {
                turn_id,
                json: json.to_string(),
            });
        }
        ErasedStructuredTurnEvent::ReasoningDelta { delta } => {
            emit_delta(observer, turn_id, "reasoning", delta.clone());
        }
        ErasedStructuredTurnEvent::RefusalDelta { delta } => {
            emit_delta(observer, turn_id, "refusal", delta.clone());
        }
        ErasedStructuredTurnEvent::ToolCallChunk {
            id,
            name,
            arguments_json_delta,
        } => emit_tool_call_chunk(observer, turn_id, id, name, arguments_json_delta.clone()),
        ErasedStructuredTurnEvent::ToolCallReady(tool) => {
            emit_tool_call_ready(observer, turn_id, tool);
        }
        ErasedStructuredTurnEvent::Completed {
            request_id,
            finish_reason,
            usage,
            ..
        } => emit_completed(
            observer,
            extensions,
            turn_id,
            request_id.clone(),
            finish_reason,
            *usage,
        ),
    }
}

fn emit_completion_stream_event(
    observer: &VisualizerLlmObserver,
    extensions: &RequestExtensions,
    metadata: &LlmRequestMetadata,
    turn_id: String,
    event: &CompletionEvent,
) {
    match event {
        CompletionEvent::Started { request_id, model } => emit_started(
            observer,
            metadata,
            turn_id,
            OperationKind::Completion,
            request_id.clone(),
            model.clone(),
        ),
        CompletionEvent::WillRetry {
            attempt,
            after,
            kind,
            status,
            ..
        } => emit_delta(
            observer,
            turn_id,
            "retry",
            format!(
                "attempt={attempt} after_ms={} kind={kind:?} status={status:?}",
                after.as_millis()
            ),
        ),
        CompletionEvent::TextDelta(delta) => {
            emit_delta(observer, turn_id, "text", delta.clone());
        }
        CompletionEvent::Completed {
            request_id,
            finish_reason,
            usage,
        } => emit_completed(
            observer,
            extensions,
            turn_id,
            request_id.clone(),
            finish_reason,
            *usage,
        ),
    }
}

fn emit_structured_completion_stream_event(
    observer: &VisualizerLlmObserver,
    extensions: &RequestExtensions,
    metadata: &LlmRequestMetadata,
    turn_id: String,
    event: &ErasedStructuredCompletionEvent,
) {
    match event {
        ErasedStructuredCompletionEvent::Started { request_id, model } => emit_started(
            observer,
            metadata,
            turn_id,
            OperationKind::StructuredCompletion,
            request_id.clone(),
            model.clone(),
        ),
        ErasedStructuredCompletionEvent::StructuredOutputChunk { json_delta } => {
            emit_delta(observer, turn_id, "structured", json_delta.clone());
        }
        ErasedStructuredCompletionEvent::StructuredOutputReady(json) => {
            observer.emit(LlmObservationEvent::StructuredReady {
                turn_id,
                json: json.to_string(),
            });
        }
        ErasedStructuredCompletionEvent::ReasoningDelta { delta } => {
            emit_delta(observer, turn_id, "reasoning", delta.clone());
        }
        ErasedStructuredCompletionEvent::RefusalDelta { delta } => {
            emit_delta(observer, turn_id, "refusal", delta.clone());
        }
        ErasedStructuredCompletionEvent::Completed {
            request_id,
            finish_reason,
            usage,
        } => emit_completed(
            observer,
            extensions,
            turn_id,
            request_id.clone(),
            finish_reason,
            *usage,
        ),
    }
}

fn model_input_views(items: &[ModelInputItem]) -> Vec<LlmInputItemView> {
    let mut output = Vec::new();
    for item in items {
        match item {
            ModelInputItem::Message { role, content } => {
                for content in content.as_slice() {
                    let MessageContent::Text(text) = content;
                    output.push(LlmInputItemView {
                        role: input_role_label(*role).to_string(),
                        kind: "text".to_string(),
                        content: text.clone(),
                        ephemeral: false,
                        source: None,
                    });
                }
            }
            ModelInputItem::Assistant(item) => {
                let (kind, content) = assistant_input_view(item);
                output.push(LlmInputItemView {
                    role: "assistant".to_string(),
                    kind: kind.to_string(),
                    content: content.to_string(),
                    ephemeral: false,
                    source: None,
                });
            }
            ModelInputItem::ToolResult(result) => {
                output.push(LlmInputItemView {
                    role: "tool".to_string(),
                    kind: "tool_result".to_string(),
                    content: format!(
                        "arguments:\n{}\nresult:\n{}",
                        result.arguments, result.result
                    ),
                    ephemeral: false,
                    source: Some(format!("{}({})", result.name, result.id)),
                });
            }
            ModelInputItem::Turn(turn) => {
                push_turn_view(&mut output, turn.as_ref());
            }
        }
    }
    output
}

fn push_turn_view(output: &mut Vec<LlmInputItemView>, turn: &dyn TurnView) {
    for index in 0..turn.item_count() {
        let Some(item) = turn.item_at(index) else {
            continue;
        };
        output.push(item_view(turn.role(), turn.ephemeral(), item));
    }
}

fn item_view(role: TurnRole, ephemeral: bool, item: &dyn ItemView) -> LlmInputItemView {
    if let Some(text) = item.as_text() {
        return input_item(turn_role_label(role), "text", text, ephemeral, None);
    }
    if let Some(text) = item.as_reasoning() {
        return input_item(turn_role_label(role), "reasoning", text, ephemeral, None);
    }
    if let Some(text) = item.as_refusal() {
        return input_item(turn_role_label(role), "refusal", text, ephemeral, None);
    }
    if let Some(tool) = item.as_tool_call() {
        return input_item(
            turn_role_label(role),
            "tool_call",
            tool.arguments.get(),
            ephemeral,
            Some(format!("{}({})", tool.name, tool.id)),
        );
    }
    if let Some(tool) = item.as_tool_result() {
        return input_item(
            "tool",
            "tool_result",
            &format!("arguments:\n{}\nresult:\n{}", tool.arguments, tool.result),
            ephemeral,
            Some(format!("{}({})", tool.name, tool.id)),
        );
    }
    input_item(turn_role_label(role), "unknown", "", ephemeral, None)
}

fn input_item(
    role: &str,
    kind: &str,
    content: &str,
    ephemeral: bool,
    source: Option<String>,
) -> LlmInputItemView {
    LlmInputItemView {
        role: role.to_string(),
        kind: kind.to_string(),
        content: content.to_string(),
        ephemeral,
        source,
    }
}

fn assistant_input_view(item: &AssistantInputItem) -> (&'static str, &str) {
    match item {
        AssistantInputItem::Text(text) => ("text", text.as_str()),
        AssistantInputItem::Reasoning(text) => ("reasoning", text.as_str()),
        AssistantInputItem::Refusal(text) => ("refusal", text.as_str()),
    }
}

fn input_role_label(role: InputMessageRole) -> &'static str {
    match role {
        InputMessageRole::System => "system",
        InputMessageRole::Developer => "developer",
        InputMessageRole::User => "user",
    }
}

fn turn_role_label(role: TurnRole) -> &'static str {
    match role {
        TurnRole::System => "system",
        TurnRole::Developer => "developer",
        TurnRole::User => "user",
        TurnRole::Assistant => "assistant",
    }
}

fn observation_source(source: LlmRequestSource) -> LlmObservationSource {
    match source {
        LlmRequestSource::ModuleTurn => LlmObservationSource::ModuleTurn,
        LlmRequestSource::SessionCompaction => LlmObservationSource::SessionCompaction,
    }
}

fn operation_kind_label(kind: OperationKind) -> &'static str {
    match kind {
        OperationKind::TextTurn => "text_turn",
        OperationKind::StructuredTurn => "structured_turn",
        OperationKind::StructuredCompletion => "structured_completion",
        OperationKind::Completion => "completion",
    }
}

fn usage_view(usage: Usage) -> LlmUsageView {
    LlmUsageView {
        input_tokens: usage.input_tokens,
        output_tokens: usage.output_tokens,
        total_tokens: usage.total_tokens,
        cost_micros_usd: usage.cost_micros_usd,
        cache_creation_tokens: usage.cache_creation_tokens,
        cache_read_tokens: usage.cache_read_tokens,
    }
}

#[derive(Debug, Clone)]
pub struct CaseRunOutput {
    pub case_path: PathBuf,
    pub output_dir: PathBuf,
    pub summary: CaseSummary,
    pub artifact: CaseArtifact,
    pub events: Vec<RuntimeEvent>,
    pub trace: TraceSnapshot,
    pub raw_trace: RawTraceSnapshot,
}

#[derive(Debug, Error)]
pub enum RunnerError {
    #[error(transparent)]
    Case(#[from] CaseFileError),
    #[error("failed to discover eval cases under {path}: {source}")]
    DiscoverCases {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("case runner failed for {path}: {message}")]
    Driver { path: PathBuf, message: String },
    #[error("failed to write eval output under {path}: {source}")]
    WriteOutput {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to install lutum trace subscriber: {message}")]
    TraceSubscriber { message: String },
    #[error("case patterns matched no eval cases: {patterns}")]
    NoCasesMatched { patterns: String },
    #[error("cannot disable required module '{module}'")]
    DisableRequiredModule { module: &'static str },
}

struct CaseExecution {
    artifact: CaseArtifact,
    events: Vec<RuntimeEvent>,
}

struct CaseOutputContext<'a> {
    case_path: &'a Path,
    output_dir: &'a Path,
    case: &'a EvalCase,
    id: &'a str,
    reporter: &'a LiveReporter,
}

pub async fn run_suite(config: &RunnerConfig) -> Result<SuiteReport, RunnerError> {
    let mut hooks = RunnerHooks::none();
    run_suite_with_hooks(config, &mut hooks).await
}

pub(crate) fn visualizer_planned_tabs(
    config: &RunnerConfig,
) -> Result<Vec<(VisualizerTabId, String)>, RunnerError> {
    let mut case_paths =
        discover_case_files(&config.cases_root).map_err(|source| RunnerError::DiscoverCases {
            path: config.cases_root.clone(),
            source,
        })?;
    case_paths = filter_case_paths(case_paths, &config.case_patterns)?;
    case_paths
        .into_iter()
        .map(|path| {
            let case = parse_case_file(&path)?;
            let id = case_id(&path, &case);
            Ok((VisualizerTabId::new(id.clone()), id))
        })
        .collect()
}

pub async fn run_suite_with_hooks(
    config: &RunnerConfig,
    hooks: &mut RunnerHooks,
) -> Result<SuiteReport, RunnerError> {
    install_lutum_trace_subscriber()?;
    validate_disabled_modules(&config.disabled_modules)?;
    let mut case_paths =
        discover_case_files(&config.cases_root).map_err(|source| RunnerError::DiscoverCases {
            path: config.cases_root.clone(),
            source,
        })?;
    case_paths = filter_case_paths(case_paths, &config.case_patterns)?;
    let run_dir = config.output_root.join(&config.run_id);
    let planned_case_count = case_paths.len();
    let run_report = suite_run_report(config, &run_dir, planned_case_count);
    std::fs::create_dir_all(&run_dir).map_err(|source| RunnerError::WriteOutput {
        path: run_dir.clone(),
        source,
    })?;
    let reporter = LiveReporter::new(&config.run_id, &run_dir)?;
    reporter.emit(
        None,
        "suite_started",
        serde_json::json!({
            "cases": case_paths.len(),
            "output_dir": run_dir.display().to_string(),
            "backends": {
                "judge": backend_report(&config.judge_backend),
                "cheap": backend_report(&config.cheap_backend),
                "default": backend_report(&config.default_backend),
                "premium": backend_report(&config.premium_backend),
            },
            "max_concurrent_llm_calls": config.max_concurrent_llm_calls.map(NonZeroUsize::get),
        }),
        format!(
            "🚀 eval suite start run={} cases={} output={}",
            config.run_id,
            case_paths.len(),
            run_dir.display()
        ),
    )?;

    let judge_llm = build_lutum(
        &config.judge_backend.endpoint,
        &config.judge_backend.token,
        &config.judge_backend.model,
        config.judge_backend.reasoning_effort,
        config.judge_backend.use_responses_api,
        None,
    )
    .map_err(|error| RunnerError::Driver {
        path: config.cases_root.clone(),
        message: error.to_string(),
    })?;
    let judge = LlmRubricJudge::new(judge_llm);

    let mut cases = Vec::new();
    for path in case_paths {
        let output =
            run_case_detailed_with_reporter(&path, config, Some(&judge), &reporter, hooks).await?;
        let failed = !output.summary.passed || output.summary.invalid;
        cases.push(output.summary);
        if hooks
            .visualizer
            .as_ref()
            .is_some_and(VisualizerHook::shutdown_requested)
        {
            break;
        }
        if failed && config.fail_fast {
            break;
        }
    }

    let report = aggregate_suite(run_report, cases);
    let suite_path = run_dir.join("suite-report.json");
    write_json_file(&suite_path, &report)?;
    eprintln!("\n════════════════════════════════════════════════════════════");
    reporter.emit(
        None,
        "suite_finished",
        serde_json::json!({
            "case_count": report.case_count,
            "passed_cases": report.passed_cases,
            "failed_cases": report.failed_cases,
            "invalid_cases": report.invalid_cases,
            "mean_score": report.mean_score,
        }),
        format!(
            "🏁 eval suite end run={} ✅passed={} ❌failed={} 💥invalid={} mean_score={:.3}",
            config.run_id,
            report.passed_cases,
            report.failed_cases,
            report.invalid_cases,
            report.mean_score
        ),
    )?;
    if let Some(visualizer) = hooks.visualizer.as_mut() {
        eprintln!("eval suite finished; visualizer remains open until its window is closed");
        visualizer.drain_cached_commands_until_shutdown();
    }
    Ok(report)
}

fn backend_report(backend: &LlmBackendConfig) -> serde_json::Value {
    serde_json::json!({
        "endpoint": backend.endpoint.as_str(),
        "model": backend.model.as_str(),
        "reasoning_effort": backend.reasoning_effort,
        "use_responses_api": backend.use_responses_api,
    })
}

fn suite_run_report(
    config: &RunnerConfig,
    run_dir: &Path,
    planned_case_count: usize,
) -> SuiteRunReport {
    SuiteRunReport {
        run_id: config.run_id.clone(),
        cases_root: config.cases_root.display().to_string(),
        output_dir: run_dir.display().to_string(),
        case_patterns: config.case_patterns.clone(),
        fail_fast: config.fail_fast,
        max_concurrent_llm_calls: config.max_concurrent_llm_calls.map(NonZeroUsize::get),
        planned_case_count,
        models: SuiteModelNames {
            judge: config.judge_backend.model.clone(),
            cheap: config.cheap_backend.model.clone(),
            default: config.default_backend.model.clone(),
            premium: config.premium_backend.model.clone(),
        },
        disabled_modules: config
            .disabled_modules
            .iter()
            .map(|module| module.as_str().to_string())
            .collect(),
    }
}

pub async fn run_case_detailed(
    case_path: &Path,
    config: &RunnerConfig,
    judge: Option<&dyn RubricJudge>,
) -> Result<CaseRunOutput, RunnerError> {
    install_lutum_trace_subscriber()?;
    validate_disabled_modules(&config.disabled_modules)?;
    let run_dir = config.output_root.join(&config.run_id);
    std::fs::create_dir_all(&run_dir).map_err(|source| RunnerError::WriteOutput {
        path: run_dir.clone(),
        source,
    })?;
    let reporter = LiveReporter::new(&config.run_id, &run_dir)?;
    let mut hooks = RunnerHooks::none();
    run_case_detailed_with_reporter(case_path, config, judge, &reporter, &mut hooks).await
}

fn filter_case_paths(
    case_paths: Vec<PathBuf>,
    patterns: &[String],
) -> Result<Vec<PathBuf>, RunnerError> {
    let normalized_patterns = patterns
        .iter()
        .map(|pattern| normalize_case_pattern(pattern))
        .filter(|pattern| !pattern.is_empty())
        .collect::<Vec<_>>();
    if normalized_patterns.is_empty() {
        return Ok(case_paths);
    }

    let mut matched = Vec::new();
    for path in case_paths {
        let mut haystacks = vec![
            normalize_case_pattern(&path.display().to_string()),
            path.file_stem()
                .and_then(|stem| stem.to_str())
                .map(normalize_case_pattern)
                .unwrap_or_default(),
        ];
        if let Ok(case) = parse_case_file(&path)
            && let Some(id) = case.id()
        {
            haystacks.push(normalize_case_pattern(id));
        }

        if normalized_patterns.iter().any(|pattern| {
            haystacks
                .iter()
                .any(|haystack| haystack.contains(pattern.as_str()))
        }) {
            matched.push(path);
        }
    }

    if matched.is_empty() {
        return Err(RunnerError::NoCasesMatched {
            patterns: patterns.join(", "),
        });
    }
    Ok(matched)
}

fn normalize_case_pattern(value: &str) -> String {
    value
        .chars()
        .flat_map(char::to_lowercase)
        .map(|ch| match ch {
            '/' | '\\' | '_' => '-',
            other => other,
        })
        .collect::<String>()
}

async fn run_case_detailed_with_reporter(
    case_path: &Path,
    config: &RunnerConfig,
    judge: Option<&dyn RubricJudge>,
    reporter: &LiveReporter,
    hooks: &mut RunnerHooks,
) -> Result<CaseRunOutput, RunnerError> {
    let case = parse_case_file(case_path)?;
    let id = case_id(case_path, &case);
    let output_dir = config
        .output_root
        .join(&config.run_id)
        .join(sanitize_id(&id));
    std::fs::create_dir_all(&output_dir).map_err(|source| RunnerError::WriteOutput {
        path: output_dir.clone(),
        source,
    })?;
    eprintln!("\n────────────────────────────────────────────────────────────");
    reporter.emit(
        Some(&id),
        "case_started",
        serde_json::json!({
            "path": case_path.display().to_string(),
            "output_dir": output_dir.display().to_string(),
        }),
        format!(
            "▶️  eval case start id={} path={} output={}",
            id,
            case_path.display(),
            output_dir.display()
        ),
    )?;
    emit_visualizer_open_tab(hooks, &id);

    let local = LocalSet::new();
    let capture = lutum_trace::capture_raw(
        AssertUnwindSafe(execute_case(
            &case,
            config,
            &output_dir,
            &id,
            reporter,
            hooks,
        ))
        .catch_unwind(),
    );
    let collected = local.run_until(capture).await;

    let trace = collected.trace;
    let raw_trace = collected.raw;
    let execution = match collected.output {
        Ok(Ok(execution)) => execution,
        Ok(Err(error)) => {
            if let Some(visualizer) = hooks.visualizer.as_ref() {
                visualizer.send_event(VisualizerEvent::SetTabStatus {
                    tab_id: VisualizerTabId::new(id.clone()),
                    status: TabStatus::Invalid,
                });
            }
            return write_runtime_failure_case_output(
                CaseOutputContext {
                    case_path,
                    output_dir: &output_dir,
                    case: &case,
                    id: &id,
                    reporter,
                },
                error.to_string(),
                trace,
                raw_trace,
            );
        }
        Err(payload) => {
            let message = format!("panic: {}", panic_payload_message(payload.as_ref()));
            if let Some(visualizer) = hooks.visualizer.as_ref() {
                visualizer.send_event(VisualizerEvent::SetTabStatus {
                    tab_id: VisualizerTabId::new(id.clone()),
                    status: TabStatus::Invalid,
                });
            }
            return write_runtime_failure_case_output(
                CaseOutputContext {
                    case_path,
                    output_dir: &output_dir,
                    case: &case,
                    id: &id,
                    reporter,
                },
                message,
                trace,
                raw_trace,
            );
        }
    };
    let artifact = execution.artifact;
    let events = execution.events;
    let report = evaluate_case(&case, &trace, &artifact, judge).await;
    let summary = CaseSummary {
        path: case_path.display().to_string(),
        id,
        description: case
            .description()
            .map(|text| normalize_text_block(&text.content)),
        passed: report.passed(),
        invalid: report.invalid,
        score: report.score,
        report,
    };

    write_json_file(&output_dir.join("artifact.json"), &artifact)?;
    write_json_file(&output_dir.join("report.json"), &summary)?;
    write_json_file(&output_dir.join("events.json"), &events)?;
    write_json_file(&output_dir.join("trace.json"), &trace_snapshot_json(&trace))?;
    if !summary.passed
        || summary.invalid
        || artifact.failure.is_some()
        || raw_trace_has_error(&raw_trace)
    {
        write_json_file(
            &output_dir.join("raw-trace.json"),
            &raw_trace_snapshot_json(&raw_trace),
        )?;
    }
    emit_case_finished(reporter, &summary, events.len())?;
    emit_visualizer_case_status(hooks, &summary);

    Ok(CaseRunOutput {
        case_path: case_path.to_path_buf(),
        output_dir,
        summary,
        artifact,
        events,
        trace,
        raw_trace,
    })
}

fn write_runtime_failure_case_output(
    ctx: CaseOutputContext<'_>,
    message: String,
    trace: TraceSnapshot,
    raw_trace: RawTraceSnapshot,
) -> Result<CaseRunOutput, RunnerError> {
    let artifact = CaseArtifact::failed(message.clone());
    let events = Vec::new();
    let report = CaseReport {
        runtime_failure: Some(message.clone()),
        checks: Vec::new(),
        modules_checks: Vec::new(),
        invalid: true,
        must_pass_ok: false,
        weighted_points_earned: 0,
        weighted_points_total: 0,
        score: 0.0,
    };
    let summary = CaseSummary {
        path: ctx.case_path.display().to_string(),
        id: ctx.id.to_string(),
        description: ctx
            .case
            .description()
            .map(|text| normalize_text_block(&text.content)),
        passed: false,
        invalid: true,
        score: 0.0,
        report,
    };

    write_json_file(&ctx.output_dir.join("artifact.json"), &artifact)?;
    write_json_file(&ctx.output_dir.join("report.json"), &summary)?;
    write_json_file(&ctx.output_dir.join("events.json"), &events)?;
    write_json_file(
        &ctx.output_dir.join("trace.json"),
        &trace_snapshot_json(&trace),
    )?;
    write_json_file(
        &ctx.output_dir.join("raw-trace.json"),
        &raw_trace_snapshot_json(&raw_trace),
    )?;
    ctx.reporter.emit(
        Some(&summary.id),
        "case_error",
        serde_json::json!({
            "path": summary.path.as_str(),
            "error": message,
        }),
        format!("eval case error id={} error={}", summary.id, message),
    )?;
    emit_case_finished(ctx.reporter, &summary, events.len())?;

    Ok(CaseRunOutput {
        case_path: ctx.case_path.to_path_buf(),
        output_dir: ctx.output_dir.to_path_buf(),
        summary,
        artifact,
        events,
        trace,
        raw_trace,
    })
}

fn emit_case_finished(
    reporter: &LiveReporter,
    summary: &CaseSummary,
    event_count: usize,
) -> Result<(), RunnerError> {
    let status_icon = if summary.invalid {
        "💥"
    } else if summary.passed {
        "✅"
    } else {
        "❌"
    };
    let case_finished_message = if let Some(runtime_failure) = &summary.report.runtime_failure {
        format!(
            "{status_icon} eval case end id={} passed={} invalid={} score={:.3} events={} failure={}",
            summary.id,
            summary.passed,
            summary.invalid,
            summary.score,
            event_count,
            runtime_failure
        )
    } else {
        format!(
            "{status_icon} eval case end id={} passed={} invalid={} score={:.3} events={}",
            summary.id, summary.passed, summary.invalid, summary.score, event_count
        )
    };
    reporter.emit(
        Some(&summary.id),
        "case_finished",
        serde_json::json!({
            "path": summary.path.as_str(),
            "passed": summary.passed,
            "invalid": summary.invalid,
            "score": summary.score,
            "runtime_failure": summary.report.runtime_failure.as_deref(),
            "events": event_count,
        }),
        case_finished_message,
    )
}

fn emit_visualizer_case_status(hooks: &RunnerHooks, summary: &CaseSummary) {
    let Some(visualizer) = hooks.visualizer.as_ref() else {
        return;
    };
    let status = if summary.invalid {
        TabStatus::Invalid
    } else if summary.passed {
        TabStatus::Passed
    } else {
        TabStatus::Failed
    };
    visualizer.send_event(VisualizerEvent::SetTabStatus {
        tab_id: VisualizerTabId::new(summary.id.clone()),
        status,
    });
}

fn emit_visualizer_open_tab(hooks: &RunnerHooks, id: &str) {
    let Some(visualizer) = hooks.visualizer.as_ref() else {
        return;
    };
    visualizer.send_event(VisualizerEvent::OpenTab {
        tab_id: VisualizerTabId::new(id.to_string()),
        title: id.to_string(),
    });
}

async fn execute_case(
    case: &EvalCase,
    config: &RunnerConfig,
    output_dir: &Path,
    case_id: &str,
    reporter: &LiveReporter,
    hooks: &mut RunnerHooks,
) -> Result<CaseExecution> {
    match case {
        EvalCase::FullAgent(case) => {
            execute_full_agent_case(case, config, output_dir, case_id, reporter, hooks).await
        }
        EvalCase::Module { target, case } => {
            execute_module_case(*target, case, config, output_dir, case_id, reporter, hooks).await
        }
    }
}

async fn execute_full_agent_case(
    case: &FullAgentCase,
    config: &RunnerConfig,
    output_dir: &Path,
    case_id: &str,
    reporter: &LiveReporter,
    hooks: &mut RunnerHooks,
) -> Result<CaseExecution> {
    let case_modules = full_agent_case_modules(case, &config.disabled_modules);
    let gui_deferred_start = hooks.visualizer.is_some();
    let case_now = parse_case_now(case.now.as_deref())
        .map_err(anyhow::Error::msg)
        .context("parse full-agent case now")?;
    let allocation = if gui_deferred_start {
        full_agent_gui_initial_allocation(&case.limits, &case_modules)
    } else {
        full_agent_allocation(&case.limits, &case_modules)
    };
    let env = build_eval_environment(
        output_dir,
        config,
        allocation,
        case.limits.max_llm_calls,
        action_module_ids(&case_modules),
        Arc::new(NoopFileSearchProvider),
        case_now,
        case_id,
        reporter,
        hooks.visualizer.as_ref().map(VisualizerHook::event_sender),
    )
    .await?;
    env.caps
        .scene()
        .set(case.participants.iter().map(Participant::new));
    seed_memories(&env.caps, env.clock.as_ref(), case_now, &case.memories).await?;
    let _ = seed_memos(&env.blackboard, env.clock.as_ref(), &case.memos).await?;
    if let Some(visualizer) = hooks.visualizer.as_mut() {
        emit_visualizer_blackboard_snapshot(case_id, &env.blackboard, Some(visualizer)).await;
        emit_visualizer_memory_page(
            case_id,
            visualizer,
            &env.blackboard,
            env.memory.as_ref(),
            0,
            25,
        )
        .await;
        visualizer.offer_action(VisualizerAction::start_activation(VisualizerTabId::new(
            case_id.to_string(),
        )));
    }

    let host = env.caps.host_io();
    let sensory = host.sensory_input_mailbox();
    let inputs = case.inputs.clone();
    let steps = case.steps.clone();
    let actions = env.actions.clone();
    let events = env.events.clone();
    let clock = env.clock.clone();
    let memory = env.memory.clone();
    let utterances = env.utterances.clone();
    let allocation_blackboard = env.blackboard.clone();
    let mut allocation_reporter =
        AllocationChangeReporter::new(case_id.to_string(), reporter.clone());
    let live_reporter = reporter.clone();
    let case_id_for_idle = case_id.to_string();
    let modules = eval_registry(&case_modules).build(&env.caps).await?;
    let mut visualizer = hooks.visualizer.as_mut();
    let step_failure: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));
    let step_outcomes: Arc<Mutex<Vec<serde_json::Value>>> = Arc::new(Mutex::new(Vec::new()));
    let step_failure_for_loop = step_failure.clone();
    let step_outcomes_for_loop = step_outcomes.clone();

    run_agent(
        modules,
        AgentEventLoopConfig {
            idle_threshold: Duration::from_secs(1),
            activate_retries: 2,
        },
        async move {
            let _ = allocation_reporter
                .emit_if_changed(&allocation_blackboard)
                .await;
            emit_visualizer_blackboard_snapshot(
                &case_id_for_idle,
                &allocation_blackboard,
                visualizer.as_deref(),
            )
            .await;
            let mut started = !gui_deferred_start;
            if started {
                run_input_phase(
                    &case_id_for_idle,
                    &inputs,
                    &steps,
                    &sensory,
                    &allocation_blackboard,
                    utterances.as_ref(),
                    events.as_ref(),
                    clock.as_ref(),
                    visualizer.as_deref(),
                    &step_failure_for_loop,
                    &step_outcomes_for_loop,
                )
                .await;
            }

            let mut last_event_count = events.event_count();
            let mut idle_ticks = 0_u64;
            let poll_ms = duration_millis_u64(EVAL_POLL_INTERVAL);
            let idle_report_every_ticks = ticks_for_interval(IDLE_REPORT_INTERVAL, poll_ms);
            let mut tick: u64 = 0;
            loop {
                if actions.silence_window_elapsed(FULL_AGENT_ACTION_SILENCE_WINDOW)
                    || events.stop_requested()
                {
                    break;
                }
                tokio::task::yield_now().await;
                tokio::time::sleep(EVAL_POLL_INTERVAL).await;
                tick = tick.saturating_add(1);
                let _ = allocation_reporter
                    .emit_if_changed(&allocation_blackboard)
                    .await;
                if let Some(visualizer) = visualizer.as_deref_mut() {
                    let command_outcome = handle_visualizer_commands(
                        &case_id_for_idle,
                        visualizer,
                        Some(&sensory),
                        &allocation_blackboard,
                        memory.as_ref(),
                        clock.as_ref(),
                    )
                    .await;
                    if command_outcome.shutdown {
                        break;
                    }
                    if command_outcome.start_requested && !started {
                        visualizer.revoke_action(start_activation_action_id(
                            &VisualizerTabId::new(case_id_for_idle.clone()),
                        ));
                        activate_gui_start_modules(&allocation_blackboard).await;
                        run_input_phase(
                            &case_id_for_idle,
                            &inputs,
                            &steps,
                            &sensory,
                            &allocation_blackboard,
                            utterances.as_ref(),
                            events.as_ref(),
                            clock.as_ref(),
                            Some(visualizer),
                            &step_failure_for_loop,
                            &step_outcomes_for_loop,
                        )
                        .await;
                        started = true;
                    }
                }
                emit_visualizer_blackboard_snapshot(
                    &case_id_for_idle,
                    &allocation_blackboard,
                    visualizer.as_deref(),
                )
                .await;
                if !started {
                    continue;
                }
                let event_count = events.event_count();
                if event_count == last_event_count {
                    idle_ticks = idle_ticks.saturating_add(1);
                } else {
                    last_event_count = event_count;
                    idle_ticks = 0;
                }
                if idle_ticks > 0 && idle_ticks.is_multiple_of(idle_report_every_ticks) {
                    let active_modules =
                        allocation_blackboard.read(active_module_observations).await;
                    let active_summary = active_modules_live_summary(&active_modules);
                    live_reporter
                        .emit_port(
                            Some(&case_id_for_idle),
                            "idle",
                            serde_json::json!({
                                "tick": tick,
                                "events": event_count,
                                "idle_ticks": idle_ticks,
                                "idle_for_ms": idle_ticks.saturating_mul(poll_ms),
                                "tick_ms": poll_ms,
                                "report_interval_ms": duration_millis_u64(IDLE_REPORT_INTERVAL),
                                "active_modules": active_modules,
                            }),
                            format!(
                                "💤 eval idle case={} idle_for_ms={} events={} active=[{}]",
                                case_id_for_idle,
                                idle_ticks.saturating_mul(poll_ms),
                                event_count,
                                active_summary
                            ),
                        )
                        .expect("full-agent eval failed to write idle event");
                }
            }
        },
    )
    .await?;

    let step_failure_message = step_failure
        .lock()
        .expect("step failure mutex poisoned")
        .take();
    let recorded_step_outcomes = step_outcomes
        .lock()
        .expect("step outcomes mutex poisoned")
        .clone();
    let mut artifact = if let Some(failure) = step_failure_message {
        let mut artifact = CaseArtifact::failed(failure);
        if let Some(utterance) = env.utterances.last_complete() {
            artifact.output = utterance.text;
        }
        artifact
    } else if let Some(utterance) = env.utterances.last_complete() {
        CaseArtifact::new(utterance.text)
    } else if env.events.stop_requested() {
        CaseArtifact::failed("stopped after max-llm-calls")
    } else {
        CaseArtifact::failed("no utterance produced")
    };
    add_observations(&mut artifact, &env.blackboard, &env.utterances).await;
    if !recorded_step_outcomes.is_empty() {
        artifact
            .observations
            .insert("steps".to_string(), serde_json::Value::Array(recorded_step_outcomes));
    }
    let events = env.events.snapshot();
    let last_state = build_full_agent_last_state_dump(
        case_id,
        &artifact,
        &env.blackboard,
        env.memory.as_ref(),
        &env.utterances,
        events.len(),
    )
    .await?;
    add_last_state_observation(&mut artifact, &last_state)?;
    write_full_agent_last_state_eure(output_dir, last_state)?;
    if let Some(visualizer) = hooks.visualizer.as_mut() {
        emit_visualizer_blackboard_snapshot(case_id, &env.blackboard, Some(visualizer)).await;
        emit_visualizer_memory_page(
            case_id,
            visualizer,
            &env.blackboard,
            env.memory.as_ref(),
            0,
            25,
        )
        .await;
    }
    Ok(CaseExecution { artifact, events })
}

async fn execute_module_case(
    target: ModuleEvalTarget,
    case: &ModuleCase,
    config: &RunnerConfig,
    output_dir: &Path,
    case_id: &str,
    reporter: &LiveReporter,
    hooks: &mut RunnerHooks,
) -> Result<CaseExecution> {
    let case_modules = module_case_modules(target, case);
    let case_now = parse_case_now(case.now.as_deref())
        .map_err(anyhow::Error::msg)
        .context("parse module case now")?;
    let env = build_eval_environment(
        output_dir,
        config,
        module_allocation(target, &case.limits, &case_modules),
        case.limits.max_llm_calls,
        action_module_ids(&case_modules),
        module_file_search_provider(target, case),
        case_now,
        case_id,
        reporter,
        hooks.visualizer.as_ref().map(VisualizerHook::event_sender),
    )
    .await?;
    seed_memories(&env.caps, env.clock.as_ref(), case_now, &case.memories).await?;
    let memo_seed_records = seed_memos(&env.blackboard, env.clock.as_ref(), &case.memos).await?;
    seed_cognition_log(&env.blackboard, env.clock.as_ref(), &case.cognition_log).await;
    if let Some(visualizer) = hooks.visualizer.as_mut() {
        emit_visualizer_blackboard_snapshot(case_id, &env.blackboard, Some(visualizer)).await;
        emit_visualizer_memory_page(
            case_id,
            visualizer,
            &env.blackboard,
            env.memory.as_ref(),
            0,
            25,
        )
        .await;
        visualizer.offer_action(VisualizerAction::start_activation(VisualizerTabId::new(
            case_id.to_string(),
        )));
    }

    let gui_deferred_start = hooks.visualizer.is_some();
    let target_module = module_id_for_target(target);
    let shutdown_target_module = target_module.clone();
    let run_target_module = target_module.clone();
    let modules = eval_registry(&case_modules).build(&env.caps).await?;
    let harness = env.caps.internal_harness_io();
    let prompt = case.prompt.content.clone();
    let has_cognition_log_seed = !case.cognition_log.is_empty();
    let events = env.events.clone();
    let blackboard = env.blackboard.clone();
    let memory = env.memory.clone();
    let clock = env.clock.clone();
    let case_id_for_gui = case_id.to_string();
    let mut visualizer = hooks.visualizer.as_mut();

    run_agent(
        modules,
        AgentEventLoopConfig {
            idle_threshold: Duration::from_secs(1),
            activate_retries: 2,
        },
        async move {
            let mut started = !gui_deferred_start;
            if started {
                activate_module_case_target(
                    target,
                    &blackboard,
                    &harness,
                    &prompt,
                    &run_target_module,
                    &memo_seed_records,
                    has_cognition_log_seed,
                )
                .await;
            }

            loop {
                if let Some(visualizer) = visualizer.as_deref_mut() {
                    let command_outcome = handle_visualizer_commands(
                        &case_id_for_gui,
                        visualizer,
                        None,
                        &blackboard,
                        memory.as_ref(),
                        clock.as_ref(),
                    )
                    .await;
                    if command_outcome.shutdown {
                        break;
                    }
                    if command_outcome.start_requested && !started {
                        visualizer.revoke_action(start_activation_action_id(
                            &VisualizerTabId::new(case_id_for_gui.clone()),
                        ));
                        activate_module_case_target(
                            target,
                            &blackboard,
                            &harness,
                            &prompt,
                            &run_target_module,
                            &memo_seed_records,
                            has_cognition_log_seed,
                        )
                        .await;
                        started = true;
                    }
                }
                emit_visualizer_blackboard_snapshot(
                    &case_id_for_gui,
                    &blackboard,
                    visualizer.as_deref(),
                )
                .await;
                if !started {
                    tokio::task::yield_now().await;
                    tokio::time::sleep(EVAL_POLL_INTERVAL).await;
                    continue;
                }
                let target_done = match target {
                    ModuleEvalTarget::AttentionSchema => {
                        !attention_schema_cognition_output(&blackboard)
                            .await
                            .is_empty()
                    }
                    _ => last_memo_log_content_for_module(&blackboard, &shutdown_target_module)
                        .await
                        .is_some(),
                };
                if events.stop_requested() || target_done {
                    break;
                }
                tokio::task::yield_now().await;
                tokio::time::sleep(EVAL_POLL_INTERVAL).await;
            }
        },
    )
    .await?;

    let output = match target {
        ModuleEvalTarget::AttentionSchema => {
            attention_schema_cognition_output(&env.blackboard).await
        }
        _ => last_memo_log_content_for_module(&env.blackboard, &target_module)
            .await
            .unwrap_or_default(),
    };
    let mut artifact = if output.is_empty() {
        if env.events.stop_requested() {
            CaseArtifact::failed("stopped after max-llm-calls before target module produced output")
        } else {
            CaseArtifact::failed("target module did not produce output")
        }
    } else {
        CaseArtifact::new(output)
    };
    add_observations(&mut artifact, &env.blackboard, &env.utterances).await;
    if let Some(visualizer) = hooks.visualizer.as_mut() {
        emit_visualizer_blackboard_snapshot(case_id, &env.blackboard, Some(visualizer)).await;
        emit_visualizer_memory_page(
            case_id,
            visualizer,
            &env.blackboard,
            env.memory.as_ref(),
            0,
            25,
        )
        .await;
    }
    Ok(CaseExecution {
        artifact,
        events: env.events.snapshot(),
    })
}

async fn activate_module_case_target(
    target: ModuleEvalTarget,
    blackboard: &Blackboard,
    harness: &InternalHarnessIo,
    prompt: &str,
    run_target_module: &ModuleId,
    memo_seed_records: &[MemoLogRecord],
    has_cognition_log_seed: bool,
) {
    match target {
        ModuleEvalTarget::QueryVector | ModuleEvalTarget::QueryAgentic => {
            let mut allocation = blackboard.read(|bb| bb.allocation().clone()).await;
            let mut config = allocation.for_module(run_target_module);
            config.guidance = prompt.to_string();
            allocation.set(run_target_module.clone(), config);
            blackboard
                .apply(BlackboardCommand::SetAllocation(allocation))
                .await;
            harness
                .allocation_updated_mailbox()
                .publish(AllocationUpdated)
                .await
                .expect("module eval failed to publish AllocationUpdated");
        }
        ModuleEvalTarget::AttentionSchema => {
            for record in memo_seed_records {
                harness
                    .memo_updated_mailbox()
                    .publish(nuillu_module::MemoUpdated {
                        owner: record.owner.clone(),
                        index: record.index,
                    })
                    .await
                    .expect("module eval failed to publish MemoUpdated");
            }
            if has_cognition_log_seed {
                harness
                    .cognition_log_updated_mailbox()
                    .publish(CognitionLogUpdated::EntryAppended {
                        source: ModuleInstanceId::new(
                            builtin::cognition_gate(),
                            ReplicaIndex::ZERO,
                        ),
                    })
                    .await
                    .expect("module eval failed to publish CognitionLogUpdated");
            }
        }
        ModuleEvalTarget::SelfModel => {
            let mut allocation = blackboard.read(|bb| bb.allocation().clone()).await;
            let mut config = allocation.for_module(run_target_module);
            config.guidance = prompt.to_string();
            allocation.set(run_target_module.clone(), config);
            blackboard
                .apply(BlackboardCommand::SetAllocation(allocation))
                .await;
            harness
                .allocation_updated_mailbox()
                .publish(AllocationUpdated)
                .await
                .expect("module eval failed to publish AllocationUpdated");
        }
    }
}

async fn attention_schema_cognition_output(blackboard: &Blackboard) -> String {
    blackboard
        .read(|bb| {
            bb.cognition_log_set()
                .logs()
                .iter()
                .filter(|record| record.source.module == builtin::attention_schema())
                .flat_map(|record| record.entries.iter().map(|entry| entry.text.as_str()))
                .collect::<Vec<_>>()
                .join("\n\n")
        })
        .await
}

async fn last_memo_log_content_for_module(
    blackboard: &Blackboard,
    module: &ModuleId,
) -> Option<String> {
    blackboard
        .read(|bb| {
            bb.recent_memo_logs()
                .into_iter()
                .filter(|record| &record.owner.module == module)
                .max_by(|a, b| {
                    a.written_at
                        .cmp(&b.written_at)
                        .then_with(|| a.owner.replica.cmp(&b.owner.replica))
                        .then_with(|| a.index.cmp(&b.index))
                })
                .map(|record| record.content)
        })
        .await
}

async fn emit_visualizer_blackboard_snapshot(
    case_id: &str,
    blackboard: &Blackboard,
    visualizer: Option<&VisualizerHook>,
) {
    let Some(visualizer) = visualizer else {
        return;
    };
    let snapshot = blackboard.read(visualizer_blackboard_snapshot).await;
    visualizer.send_event(VisualizerEvent::BlackboardSnapshot {
        tab_id: VisualizerTabId::new(case_id.to_string()),
        snapshot,
    });
}

async fn activate_gui_start_modules(blackboard: &Blackboard) {
    let mut allocation = blackboard.read(|bb| bb.allocation().clone()).await;
    let controller_id = builtin::attention_controller();
    let mut controller_config = allocation.for_module(&controller_id);
    controller_config.guidance =
        "GUI start: process the sensory input and allocate the next active faculties.".to_string();
    allocation.set(controller_id.clone(), controller_config);
    allocation.set_activation(controller_id, ActivationRatio::ONE);

    let sensory_id = builtin::sensory();
    let mut sensory_config = allocation.for_module(&sensory_id);
    sensory_config.guidance = "GUI start: normalize the queued sensory input.".to_string();
    allocation.set(sensory_id.clone(), sensory_config);
    allocation.set_activation(sensory_id, ActivationRatio::ONE);

    blackboard
        .apply(BlackboardCommand::SetAllocation(allocation))
        .await;
}

async fn publish_full_agent_inputs(
    case_id: &str,
    inputs: &[FullAgentInput],
    sensory: &SensoryInputMailbox,
    clock: &dyn Clock,
    visualizer: Option<&VisualizerHook>,
) {
    let now = clock.now();
    for input in inputs {
        let body = match input {
            FullAgentInput::Heard { direction, content } => SensoryInput::Heard {
                direction: direction.clone(),
                content: content.content.clone(),
                observed_at: now,
            },
            FullAgentInput::Seen {
                direction,
                appearance,
            } => SensoryInput::Seen {
                direction: direction.clone(),
                appearance: appearance.content.clone(),
                observed_at: now,
            },
        };
        sensory
            .publish(body.clone())
            .await
            .expect("full-agent eval failed to publish SensoryInput");
        if let Some(visualizer) = visualizer {
            visualizer.send_event(VisualizerEvent::SensoryInput {
                tab_id: VisualizerTabId::new(case_id.to_string()),
                input: body,
            });
        }
    }
}

#[allow(clippy::too_many_arguments)]
async fn run_input_phase(
    case_id: &str,
    inputs: &[FullAgentInput],
    steps: &[EvalStep],
    sensory: &SensoryInputMailbox,
    blackboard: &Blackboard,
    utterances: &RecordingUtteranceSink,
    events: &RecordingRuntimeEventSink,
    clock: &dyn Clock,
    visualizer: Option<&VisualizerHook>,
    step_failure: &Arc<Mutex<Option<String>>>,
    step_outcomes: &Arc<Mutex<Vec<serde_json::Value>>>,
) {
    if steps.is_empty() {
        publish_full_agent_inputs(case_id, inputs, sensory, clock, visualizer).await;
        return;
    }
    for (index, step) in steps.iter().enumerate() {
        publish_full_agent_inputs(case_id, &step.inputs, sensory, clock, visualizer).await;

        let mut wait_outcome = WaitOutcome::Met;
        if let Some(wait_for) = &step.wait_for {
            wait_outcome = wait_for_condition(blackboard, events, wait_for).await;
        }

        let mut check_results: Vec<serde_json::Value> = Vec::new();
        let mut must_pass_failure: Option<String> = None;
        if matches!(wait_outcome, WaitOutcome::Met) && !step.checks.is_empty() {
            let snapshot = build_step_snapshot(blackboard, utterances).await;
            for check in &step.checks {
                let (passed, diagnostic) = evaluate_step_check(check, &snapshot);
                let common = check.common();
                check_results.push(serde_json::json!({
                    "name": check.display_name(),
                    "kind": check.kind_name(),
                    "passed": passed,
                    "must_pass": common.must_pass,
                    "diagnostic": diagnostic,
                }));
                if !passed && common.must_pass && must_pass_failure.is_none() {
                    must_pass_failure = Some(format!(
                        "step {index} must-pass check '{name}' failed: {diag}",
                        name = check.display_name(),
                        diag = diagnostic.clone().unwrap_or_else(|| "no diagnostic".to_string()),
                    ));
                }
            }
        }

        let status = match (&wait_outcome, &must_pass_failure) {
            (WaitOutcome::Timeout, _) => "timed-out",
            (WaitOutcome::Stopped, _) => "stopped",
            (_, Some(_)) => "check-failed",
            _ => "ok",
        };
        let mut outcome = serde_json::Map::new();
        outcome.insert("index".to_string(), serde_json::Value::from(index));
        if let Some(description) = &step.description {
            outcome.insert(
                "description".to_string(),
                serde_json::Value::String(description.content.clone()),
            );
        }
        outcome.insert(
            "status".to_string(),
            serde_json::Value::String(status.to_string()),
        );
        outcome.insert(
            "checks".to_string(),
            serde_json::Value::Array(check_results),
        );
        step_outcomes
            .lock()
            .expect("step outcomes mutex poisoned")
            .push(serde_json::Value::Object(outcome));

        match wait_outcome {
            WaitOutcome::Met => {}
            WaitOutcome::Timeout => {
                let wait_label = wait_for_label(step.wait_for.as_ref());
                let message = format!(
                    "step {index} timed out waiting for {wait_label}",
                );
                step_failure
                    .lock()
                    .expect("step failure mutex poisoned")
                    .get_or_insert(message);
                events.request_stop("step-timeout");
                return;
            }
            WaitOutcome::Stopped => return,
        }

        if let Some(message) = must_pass_failure {
            step_failure
                .lock()
                .expect("step failure mutex poisoned")
                .get_or_insert(message);
            events.request_stop("step-check-failed");
            return;
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum WaitOutcome {
    Met,
    Timeout,
    Stopped,
}

async fn wait_for_condition(
    blackboard: &Blackboard,
    events: &RecordingRuntimeEventSink,
    wait_for: &WaitFor,
) -> WaitOutcome {
    match wait_for {
        WaitFor::MemoFrom { module, timeout_ms } => {
            let target = module.module_id();
            let baseline = memo_count_for_module(blackboard, &target).await;
            let deadline = Duration::from_millis(*timeout_ms);
            let start = Instant::now();
            let poll = Duration::from_millis(50);
            loop {
                if events.stop_requested() {
                    return WaitOutcome::Stopped;
                }
                let count = memo_count_for_module(blackboard, &target).await;
                if count > baseline {
                    return WaitOutcome::Met;
                }
                let elapsed = start.elapsed();
                if elapsed >= deadline {
                    return WaitOutcome::Timeout;
                }
                let remaining = deadline.saturating_sub(elapsed);
                tokio::time::sleep(remaining.min(poll)).await;
            }
        }
    }
}

fn wait_for_label(wait_for: Option<&WaitFor>) -> String {
    match wait_for {
        Some(WaitFor::MemoFrom { module, timeout_ms }) => format!(
            "memo from module '{module}' within {timeout_ms}ms",
            module = module.as_str(),
        ),
        None => "<no wait-for>".to_string(),
    }
}

async fn memo_count_for_module(blackboard: &Blackboard, module: &ModuleId) -> usize {
    blackboard
        .read(|bb| {
            bb.recent_memo_logs()
                .into_iter()
                .filter(|record| &record.owner.module == module)
                .count()
        })
        .await
}

async fn build_step_snapshot(
    blackboard: &Blackboard,
    utterances: &RecordingUtteranceSink,
) -> CaseArtifact {
    let mut artifact = CaseArtifact::new(
        utterances
            .last_complete()
            .map(|utterance| utterance.text)
            .unwrap_or_default(),
    );
    add_observations(&mut artifact, blackboard, utterances).await;
    artifact
}

fn evaluate_step_check(check: &Check, artifact: &CaseArtifact) -> (bool, Option<String>) {
    match check {
        Check::JsonPointerEquals {
            pointer, expected, ..
        } => {
            let json = artifact.as_json();
            let actual = pointer_text(&json, pointer);
            let passed = actual.as_deref() == Some(expected.as_str());
            let diagnostic = (!passed).then(|| match actual {
                Some(actual) => format!(
                    "expected JSON pointer {pointer:?} to equal {expected:?}, got {actual:?}"
                ),
                None => format!("JSON pointer {pointer:?} did not match artifact"),
            });
            (passed, diagnostic)
        }
        Check::JsonPointerContains {
            pointer, contains, ..
        } => {
            let json = artifact.as_json();
            let actual = pointer_text(&json, pointer);
            let passed = actual
                .as_deref()
                .is_some_and(|text| text.contains(contains));
            let diagnostic = (!passed).then(|| match actual {
                Some(actual) => format!(
                    "expected JSON pointer {pointer:?} to contain {contains:?}, got {actual:?}"
                ),
                None => format!("JSON pointer {pointer:?} did not match artifact"),
            });
            (passed, diagnostic)
        }
        Check::ArtifactTextContains {
            field, contains, ..
        } => {
            let field = field.unwrap_or(ArtifactTextField::Output);
            let text = artifact_text(artifact, field);
            let passed = text.contains(contains);
            let diagnostic = (!passed).then(|| {
                format!(
                    "expected {field_name} to contain {contains:?}",
                    field_name = field_label(field),
                )
            });
            (passed, diagnostic)
        }
        Check::ArtifactTextExact { field, exact, .. } => {
            let field = field.unwrap_or(ArtifactTextField::Output);
            let expected = normalize_text_block(&exact.content);
            let text = normalize_text_block(artifact_text(artifact, field));
            let passed = text == expected;
            let diagnostic = (!passed).then(|| {
                format!(
                    "expected {field_name} to equal {expected:?}, got {text:?}",
                    field_name = field_label(field),
                )
            });
            (passed, diagnostic)
        }
        _ => (true, None),
    }
}

async fn handle_visualizer_commands(
    case_id: &str,
    visualizer: &mut VisualizerHook,
    sensory: Option<&SensoryInputMailbox>,
    blackboard: &Blackboard,
    memory: &dyn MemoryStore,
    clock: &dyn Clock,
) -> VisualizerCommandOutcome {
    let mut outcome = VisualizerCommandOutcome::default();
    let start_activation_id =
        start_activation_action_id(&VisualizerTabId::new(case_id.to_string()));
    loop {
        let message = match visualizer.commands.try_recv() {
            Ok(message) => message,
            Err(std::sync::mpsc::TryRecvError::Empty) => break,
            Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                visualizer.request_shutdown();
                outcome.shutdown = true;
                break;
            }
        };
        let command = match message {
            VisualizerClientMessage::Hello { .. } => continue,
            VisualizerClientMessage::InvokeAction { action_id } => {
                if action_id == start_activation_id {
                    outcome.start_requested = true;
                }
                continue;
            }
            VisualizerClientMessage::Command { command } => command,
        };
        match command {
            VisualizerCommand::Shutdown => {
                visualizer.request_shutdown();
                outcome.shutdown = true;
            }
            VisualizerCommand::SendSensoryInput { tab_id, input } if tab_id.as_str() == case_id => {
                let Some(sensory) = sensory else {
                    visualizer.send_event(VisualizerEvent::Log {
                        tab_id,
                        message: "this runtime does not accept sensory input".to_string(),
                    });
                    continue;
                };
                let observed_at = clock.now();
                let body = match input.kind {
                    ChatInputKind::Heard => SensoryInput::Heard {
                        direction: input.direction,
                        content: input.content,
                        observed_at,
                    },
                    ChatInputKind::Seen => SensoryInput::Seen {
                        direction: input.direction,
                        appearance: input.content,
                        observed_at,
                    },
                };
                let _ = sensory.publish(body.clone()).await;
                visualizer.send_event(VisualizerEvent::SensoryInput {
                    tab_id,
                    input: body,
                });
            }
            VisualizerCommand::QueryMemory {
                tab_id,
                query,
                limit,
            } if tab_id.as_str() == case_id => {
                let records = memory
                    .search(&MemoryQuery {
                        text: query.clone(),
                        limit,
                    })
                    .await
                    .map(|records| records.into_iter().map(memory_record_view).collect())
                    .unwrap_or_default();
                visualizer.send_event(VisualizerEvent::MemoryQueryResult {
                    tab_id,
                    query,
                    records,
                });
            }
            VisualizerCommand::ListMemories {
                tab_id,
                page,
                per_page,
            } if tab_id.as_str() == case_id => {
                let all_records = list_all_visualizer_memories(blackboard, memory).await;
                visualizer.set_memory_cache(case_id, all_records.clone());
                let records = memory_page_from_records(&all_records, page, per_page);
                visualizer.send_event(VisualizerEvent::MemoryPage {
                    tab_id,
                    page: records,
                });
            }
            _ => {}
        }
    }
    outcome
}

#[derive(Debug, Default)]
struct VisualizerCommandOutcome {
    shutdown: bool,
    start_requested: bool,
}

async fn emit_visualizer_memory_page(
    case_id: &str,
    visualizer: &mut VisualizerHook,
    blackboard: &Blackboard,
    memory: &dyn MemoryStore,
    page: usize,
    per_page: usize,
) {
    let records = list_all_visualizer_memories(blackboard, memory).await;
    visualizer.set_memory_cache(case_id, records.clone());
    let page = memory_page_from_records(&records, page, per_page);
    visualizer.send_event(VisualizerEvent::MemoryPage {
        tab_id: VisualizerTabId::new(case_id.to_string()),
        page,
    });
}

async fn list_all_visualizer_memories(
    blackboard: &Blackboard,
    memory: &dyn MemoryStore,
) -> Vec<MemoryRecordView> {
    let indexes = blackboard
        .read(|bb| {
            let mut records = bb
                .memory_metadata()
                .iter()
                .map(|(index, metadata)| {
                    (index.clone(), metadata.occurred_at, metadata.last_accessed)
                })
                .collect::<Vec<_>>();
            records.sort_by(|left, right| {
                right
                    .1
                    .cmp(&left.1)
                    .then_with(|| right.2.cmp(&left.2))
                    .then_with(|| left.0.as_str().cmp(right.0.as_str()))
            });
            records
                .into_iter()
                .map(|(index, _, _)| index)
                .collect::<Vec<_>>()
        })
        .await;

    let mut records = Vec::new();
    for index in indexes {
        if let Ok(Some(record)) = memory.get(&index).await {
            records.push(memory_record_view(record));
        }
    }

    if records.is_empty() {
        let mut seen = HashSet::new();
        for rank in [
            MemoryRank::Identity,
            MemoryRank::Permanent,
            MemoryRank::LongTerm,
            MemoryRank::MidTerm,
            MemoryRank::ShortTerm,
        ] {
            if let Ok(rank_records) = memory.list_by_rank(rank).await {
                for record in rank_records {
                    if seen.insert(record.index.clone()) {
                        records.push(memory_record_view(record));
                    }
                }
            }
        }
        records.sort_by(|left, right| {
            right
                .occurred_at
                .cmp(&left.occurred_at)
                .then_with(|| left.index.cmp(&right.index))
        });
    }

    records
}

fn memory_page_from_records(
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

fn memory_record_view(record: nuillu_module::ports::MemoryRecord) -> MemoryRecordView {
    MemoryRecordView {
        index: record.index.as_str().to_owned(),
        rank: memory_rank_name(record.rank).to_owned(),
        occurred_at: record.occurred_at,
        content: record.content.as_str().to_owned(),
    }
}

struct EvalEnvironment {
    blackboard: Blackboard,
    caps: CapabilityProviders,
    memory: Arc<dyn MemoryStore>,
    utterances: Arc<RecordingUtteranceSink>,
    actions: Arc<ActionActivityTracker>,
    events: Arc<RecordingRuntimeEventSink>,
    clock: Arc<dyn Clock>,
}

struct AnchoredRealtimeClock {
    base: DateTime<Utc>,
    started: Instant,
}

impl AnchoredRealtimeClock {
    fn new(base: DateTime<Utc>) -> Self {
        Self {
            base,
            started: Instant::now(),
        }
    }
}

#[async_trait(?Send)]
impl Clock for AnchoredRealtimeClock {
    fn now(&self) -> DateTime<Utc> {
        self.base + ChronoDuration::from_std(self.started.elapsed()).unwrap_or_default()
    }

    async fn sleep_until(&self, deadline: DateTime<Utc>) {
        let remaining = deadline - self.now();
        let Ok(duration) = remaining.to_std() else {
            return;
        };
        if duration.is_zero() {
            return;
        }
        tokio::time::sleep(duration).await;
    }
}

async fn build_eval_environment(
    output_dir: &Path,
    config: &RunnerConfig,
    allocation: ResourceAllocation,
    max_llm_calls: Option<u64>,
    action_modules: Vec<ModuleId>,
    file_search: Arc<dyn FileSearchProvider>,
    case_now: Option<DateTime<FixedOffset>>,
    case_id: &str,
    reporter: &LiveReporter,
    visualizer: Option<VisualizerEventSink>,
) -> Result<EvalEnvironment> {
    let blackboard = Blackboard::with_allocation(allocation);
    let events = Arc::new(RecordingRuntimeEventSink::new(
        case_id.to_string(),
        max_llm_calls,
        reporter.clone(),
        visualizer.clone(),
    ));
    let actions = Arc::new(ActionActivityTracker::new(action_modules));
    let utterances = Arc::new(RecordingUtteranceSink::new(
        case_id.to_string(),
        reporter.clone(),
        actions.clone(),
        visualizer.clone(),
    ));
    let clock: Arc<dyn Clock> = match case_now {
        Some(now) => Arc::new(AnchoredRealtimeClock::new(now.with_timezone(&Utc))),
        None => Arc::new(SystemClock),
    };
    let memory: Arc<dyn MemoryStore> = Arc::new(connect_memory_store(output_dir, config).await?);
    let policy_store: Arc<dyn PolicyStore> =
        Arc::new(connect_policy_store(output_dir, config).await?);
    let llm_observer = visualizer
        .clone()
        .map(|sender| VisualizerLlmObserver::new(case_id.to_string(), sender));
    let tiers = build_tiers(config, llm_observer)?;
    let runtime_policy = RuntimePolicy {
        memo_retained_per_owner: EVAL_MEMO_RETAINED_PER_OWNER,
        max_concurrent_llm_calls: config.max_concurrent_llm_calls,
        ..RuntimePolicy::default()
    };
    let caps = CapabilityProviders::new(CapabilityProviderConfig {
        ports: CapabilityProviderPorts {
            blackboard: blackboard.clone(),
            cognition_log_port: Arc::new(InMemoryCognitionLogRepository::new()),
            primary_memory_store: memory.clone(),
            memory_replicas: Vec::new(),
            primary_policy_store: policy_store,
            policy_replicas: Vec::new(),
            file_search,
            utterance_sink: utterances.clone(),
            clock: clock.clone(),
            tiers,
        },
        runtime: CapabilityProviderRuntime {
            event_sink: events.clone(),
            policy: runtime_policy,
        },
    });

    Ok(EvalEnvironment {
        blackboard,
        caps,
        memory,
        utterances,
        actions,
        events,
        clock,
    })
}

fn action_module_ids(modules: &[EvalModule]) -> Vec<ModuleId> {
    modules
        .iter()
        .copied()
        .filter(|module| module.is_action_module())
        .map(EvalModule::module_id)
        .collect()
}

async fn connect_memory_store(
    output_dir: &Path,
    config: &RunnerConfig,
) -> Result<LibsqlMemoryStore> {
    let embedder = PotionBase8MEmbedder::from_local_dir(&config.model_dir)
        .with_context(|| format!("load model2vec model from {}", config.model_dir.display()))?;
    let dimensions = embedder.dimensions();
    let profile = EmbeddingProfile::new("potion-base-8M", "local", dimensions);
    let db_path = output_dir.join("memory.db");
    LibsqlMemoryStore::connect(
        LibsqlMemoryStoreConfig::local(db_path, dimensions).with_active_profile(profile),
        Box::new(embedder),
    )
    .await
    .context("connect libsql memory store")
}

async fn connect_policy_store(
    output_dir: &Path,
    config: &RunnerConfig,
) -> Result<LibsqlPolicyStore> {
    let embedder = PotionBase8MEmbedder::from_local_dir(&config.model_dir)
        .with_context(|| format!("load model2vec model from {}", config.model_dir.display()))?;
    let dimensions = embedder.dimensions();
    let profile = EmbeddingProfile::new("potion-base-8M", "local", dimensions);
    let db_path = output_dir.join("policy.db");
    LibsqlPolicyStore::connect(
        LibsqlPolicyStoreConfig::local(db_path, dimensions).with_active_profile(profile),
        Box::new(embedder),
    )
    .await
    .context("connect libsql policy store")
}

fn module_file_search_provider(
    target: ModuleEvalTarget,
    case: &ModuleCase,
) -> Arc<dyn FileSearchProvider> {
    if target == ModuleEvalTarget::QueryAgentic && !case.files.is_empty() {
        Arc::new(SeededFileSearchProvider::new(case.files.clone()))
    } else {
        Arc::new(NoopFileSearchProvider)
    }
}

#[derive(Debug)]
struct SeededFileSearchProvider {
    files: Vec<SeededFile>,
}

#[derive(Debug)]
struct SeededFile {
    path: String,
    lines: Vec<String>,
}

impl SeededFileSearchProvider {
    fn new(files: Vec<crate::cases::FileSeed>) -> Self {
        Self {
            files: files
                .into_iter()
                .map(|file| SeededFile {
                    path: file.path,
                    lines: file.content.content.lines().map(str::to_owned).collect(),
                })
                .collect(),
        }
    }
}

#[async_trait(?Send)]
impl FileSearchProvider for SeededFileSearchProvider {
    async fn search(&self, query: &FileSearchQuery) -> Result<Vec<FileSearchHit>, PortError> {
        if query.pattern.is_empty() || query.max_matches == 0 {
            return Ok(Vec::new());
        }

        let regex = if query.regex {
            Some(
                RegexBuilder::new(&query.pattern)
                    .case_insensitive(!query.case_sensitive)
                    .build()
                    .map_err(|error| PortError::InvalidInput(error.to_string()))?,
            )
        } else {
            None
        };
        let literal = if query.case_sensitive {
            query.pattern.clone()
        } else {
            query.pattern.to_lowercase()
        };

        let mut hits = Vec::new();
        for file in &self.files {
            for (line_index, line) in file.lines.iter().enumerate() {
                let line_matches = match &regex {
                    Some(regex) => regex.is_match(line),
                    None if query.case_sensitive => line.contains(&literal),
                    None => line.to_lowercase().contains(&literal),
                };
                let line_matches = if query.invert_match {
                    !line_matches
                } else {
                    line_matches
                };
                if !line_matches {
                    continue;
                }

                let start = line_index.saturating_sub(query.context);
                let end = (line_index + query.context + 1).min(file.lines.len());
                hits.push(FileSearchHit {
                    path: file.path.clone(),
                    line: line_index + 1,
                    snippet: file.lines[start..end].join("\n"),
                });
                if hits.len() >= query.max_matches {
                    return Ok(hits);
                }
            }
        }
        Ok(hits)
    }
}

async fn seed_memories(
    caps: &CapabilityProviders,
    clock: &dyn Clock,
    case_now: Option<DateTime<FixedOffset>>,
    memories: &[crate::cases::MemorySeed],
) -> Result<()> {
    let writer = caps.memory_writer();
    for memory in memories {
        let occurred_at = memory_seed_occurred_at(clock, case_now, memory)?;
        writer
            .insert_with_occurred_at(
                memory.content.content.clone(),
                MemoryRank::from(memory.rank),
                memory.decay_secs,
                occurred_at,
            )
            .await
            .context("seed eval memory")?;
    }
    Ok(())
}

fn memory_seed_occurred_at(
    clock: &dyn Clock,
    case_now: Option<DateTime<FixedOffset>>,
    memory: &crate::cases::MemorySeed,
) -> Result<Option<DateTime<Utc>>> {
    if let Some(datetime) = &memory.datetime {
        return parse_memory_datetime(datetime, case_now)
            .map(Some)
            .map_err(anyhow::Error::msg)
            .with_context(|| format!("parse memory datetime {datetime}"));
    }
    if let Some(seconds_ago) = memory.seconds_ago {
        return Ok(Some(clock.now() - ChronoDuration::seconds(seconds_ago)));
    }
    Ok(None)
}

async fn seed_memos(
    blackboard: &Blackboard,
    clock: &dyn Clock,
    memos: &[crate::cases::MemoSeed],
) -> Result<Vec<nuillu_blackboard::MemoLogRecord>> {
    let now = clock.now();
    let mut records = Vec::new();
    for memo in memos {
        let module = ModuleId::new(memo.module.clone())
            .with_context(|| format!("seed memo module id {}", memo.module))?;
        let owner = ModuleInstanceId::new(module, ReplicaIndex::new(memo.replica));
        let written_at = now - ChronoDuration::seconds(memo.seconds_ago);
        records.push(
            blackboard
                .update_memo(owner, memo.content.content.clone(), written_at)
                .await,
        );
    }
    Ok(records)
}

async fn seed_cognition_log(
    blackboard: &Blackboard,
    clock: &dyn Clock,
    seeds: &[crate::cases::CognitionLogSeed],
) {
    let stream = ModuleInstanceId::new(builtin::cognition_gate(), ReplicaIndex::ZERO);
    let now = clock.now();
    for seed in seeds {
        blackboard
            .apply(BlackboardCommand::AppendCognitionLog {
                source: stream.clone(),
                entry: CognitionLogEntry {
                    at: now - ChronoDuration::seconds(seed.seconds_ago),
                    text: seed.text.content.clone(),
                },
            })
            .await;
    }
}

fn build_tiers(
    config: &RunnerConfig,
    llm_observer: Option<VisualizerLlmObserver>,
) -> Result<LutumTiers> {
    let cheap = build_lutum(
        &config.cheap_backend.endpoint,
        &config.cheap_backend.token,
        &config.cheap_backend.model,
        config.cheap_backend.reasoning_effort,
        config.cheap_backend.use_responses_api,
        llm_observer.clone(),
    )?;
    let default = build_lutum(
        &config.default_backend.endpoint,
        &config.default_backend.token,
        &config.default_backend.model,
        config.default_backend.reasoning_effort,
        config.default_backend.use_responses_api,
        llm_observer.clone(),
    )?;
    let premium = build_lutum(
        &config.premium_backend.endpoint,
        &config.premium_backend.token,
        &config.premium_backend.model,
        config.premium_backend.reasoning_effort,
        config.premium_backend.use_responses_api,
        llm_observer,
    )?;
    Ok(LutumTiers {
        cheap,
        default,
        premium,
    })
}

fn build_lutum(
    endpoint: &str,
    token: &str,
    model: &str,
    reasoning_effort: Option<ReasoningEffort>,
    use_responses_api: bool,
    llm_observer: Option<VisualizerLlmObserver>,
) -> Result<Lutum> {
    let adapter = OpenAiAdapter::new(token.to_owned())
        .with_base_url(endpoint.to_owned())
        .with_default_model(ModelName::new(model)?)
        .with_resolve_reasoning_effort(ConfiguredReasoningEffort);
    let adapter = if use_responses_api {
        adapter
    } else {
        adapter.with_chat_completions()
    };
    let mut lutum = Lutum::new(
        Arc::new(adapter),
        SharedPoolBudgetManager::new(SharedPoolBudgetOptions::default()),
    )
    .with_extension(RawTelemetryConfig::all());
    if let Some(observer) = llm_observer {
        lutum.extend_hooks(observer.hook_set());
    }
    Ok(match reasoning_effort {
        Some(reasoning_effort) => {
            lutum.with_extension(ReasoningEffortConfig(reasoning_effort.into()))
        }
        None => lutum,
    })
}

#[derive(Clone, Copy)]
struct ReasoningEffortConfig(OpenAiReasoningEffort);

#[lutum::impl_hook(lutum_openai::ResolveReasoningEffort)]
async fn configured_reasoning_effort(
    extensions: &RequestExtensions,
) -> Option<OpenAiReasoningEffort> {
    extensions
        .get::<ReasoningEffortConfig>()
        .map(|value| value.0)
}

fn full_agent_case_modules(case: &FullAgentCase, disabled: &[EvalModule]) -> Vec<EvalModule> {
    let mut modules = case
        .modules
        .clone()
        .unwrap_or_else(|| DEFAULT_FULL_AGENT_MODULES.to_vec());
    if !disabled.is_empty() {
        modules.retain(|module| !disabled.contains(module));
    }
    modules
}

fn validate_disabled_modules(disabled: &[EvalModule]) -> Result<(), RunnerError> {
    for module in disabled {
        if REQUIRED_FULL_AGENT_MODULES.contains(module) {
            return Err(RunnerError::DisableRequiredModule {
                module: module.as_str(),
            });
        }
    }
    Ok(())
}

fn module_case_modules(target: ModuleEvalTarget, case: &ModuleCase) -> Vec<EvalModule> {
    case.modules
        .clone()
        .unwrap_or_else(|| vec![target.module()])
}

fn eval_registry(modules: &[EvalModule]) -> ModuleRegistry {
    let mut registry = ModuleRegistry::new();
    for module in modules {
        registry = register_eval_module(registry, *module);
    }
    declare_eval_dependencies(registry, modules)
}

fn declare_eval_dependencies(registry: ModuleRegistry, modules: &[EvalModule]) -> ModuleRegistry {
    use nuillu_types::builtin;

    let present = modules
        .iter()
        .copied()
        .map(EvalModule::module_id)
        .collect::<std::collections::HashSet<_>>();
    let edges = [
        (builtin::speak_gate(), builtin::cognition_gate()),
        (builtin::self_model(), builtin::query_vector()),
        (builtin::cognition_gate(), builtin::sensory()),
        (builtin::cognition_gate(), builtin::query_vector()),
        (builtin::cognition_gate(), builtin::query_policy()),
        (builtin::cognition_gate(), builtin::query_agentic()),
        (builtin::cognition_gate(), builtin::self_model()),
        (builtin::cognition_gate(), builtin::surprise()),
        (builtin::value_estimator(), builtin::query_policy()),
        (builtin::reward(), builtin::value_estimator()),
        (builtin::policy(), builtin::reward()),
    ];

    edges
        .into_iter()
        .fold(registry, |registry, (dependent, dependency)| {
            if present.contains(&dependent) && present.contains(&dependency) {
                registry.depends_on(dependent, dependency)
            } else {
                registry
            }
        })
}

fn eval_policy(
    replicas_range: std::ops::RangeInclusive<u8>,
    rate_limit_range: std::ops::RangeInclusive<Bpm>,
) -> ModulePolicy {
    ModulePolicy::new(
        ReplicaCapRange::new(*replicas_range.start(), *replicas_range.end()).unwrap(),
        rate_limit_range,
        linear_ratio_fn,
    )
}

fn register_eval_module(registry: ModuleRegistry, module: EvalModule) -> ModuleRegistry {
    match module {
        // Input-driven and bursty: bursts of inputs need a fast active pace
        // so observations are normalized within the same tick window.
        EvalModule::Sensory => registry
            .register(eval_policy(0..=1, Bpm::range(6.0, 18.0)), |caps| {
                nuillu_sensory::SensoryModule::new(
                    caps.sensory_input_inbox(),
                    caps.allocation_updated_inbox(),
                    caps.allocation_reader(),
                    caps.memo(),
                    caps.clock(),
                    caps.llm_access(),
                )
            })
            .expect("eval module registration should be unique"),
        // Gates the speak path; must re-fire fast as memos accumulate so
        // the cognition log is current by the time speak-gate considers it.
        EvalModule::CognitionGate => registry
            .register(eval_policy(0..=1, Bpm::range(6.0, 18.0)), |caps| {
                nuillu_cognition_gate::CognitionGateModule::new(
                    caps.memo_updated_inbox(),
                    caps.allocation_updated_inbox(),
                    caps.blackboard_reader(),
                    caps.allocation_reader(),
                    caps.cognition_writer(),
                    caps.time_division(),
                    caps.llm_access(),
                )
            })
            .expect("eval module registration should be unique"),
        // Expensive (premium tier in default model-set), heavy reasoning.
        // Should only fire on meaningful state shifts — slow base pace so
        // it doesn't burn budget reacting to every memo update.
        EvalModule::AttentionController => registry
            .register(eval_policy(0..=1, Bpm::range(3.0, 6.0)), |caps| {
                nuillu_attention_controller::AttentionControllerModule::new(
                    caps.memo_updated_inbox(),
                    caps.attention_control_inbox(),
                    caps.blackboard_reader(),
                    caps.cognition_log_reader(),
                    caps.allocation_reader(),
                    caps.allocation_writer(),
                    caps.memo(),
                    caps.llm_access(),
                )
            })
            .expect("eval module registration should be unique"),
        // Periodic first-person attention narration; not on the critical
        // path for the speak loop.
        EvalModule::AttentionSchema => registry
            .register(eval_policy(0..=1, Bpm::range(3.0, 6.0)), |caps| {
                nuillu_attention_schema::AttentionSchemaModule::new(
                    caps.memo_updated_inbox(),
                    caps.allocation_updated_inbox(),
                    caps.cognition_log_updated_inbox(),
                    caps.blackboard_reader(),
                    caps.allocation_reader(),
                    caps.cognition_log_reader(),
                    caps.cognition_writer(),
                    caps.llm_access(),
                )
            })
            .expect("eval module registration should be unique"),
        // On-demand: fires on controller allocation guidance.
        EvalModule::SelfModel => registry
            .register(eval_policy(0..=1, Bpm::range(3.0, 6.0)), |caps| {
                nuillu_self_model::SelfModelModule::new(
                    caps.allocation_updated_inbox(),
                    caps.allocation_reader(),
                    caps.blackboard_reader(),
                    caps.cognition_log_reader(),
                    caps.memo(),
                    caps.llm_access(),
                )
            })
            .expect("eval module registration should be unique"),
        // Memory retrieval is on the critical path between cognition-gate
        // and speak-gate; needs a quick active pace.
        EvalModule::QueryVector => registry
            .register(eval_policy(0..=1, Bpm::range(6.0, 15.0)), |caps| {
                nuillu_query_vector::QueryVectorModule::new(
                    caps.allocation_updated_inbox(),
                    caps.cognition_log_updated_inbox(),
                    caps.allocation_reader(),
                    caps.blackboard_reader(),
                    caps.vector_memory_searcher(),
                    caps.typed_memo::<nuillu_query_vector::QueryVectorMemo>(),
                    caps.llm_access(),
                )
            })
            .expect("eval module registration should be unique"),
        EvalModule::QueryPolicy => registry
            .register(eval_policy(0..=1, Bpm::range(6.0, 15.0)), |caps| {
                nuillu_query_policy::QueryPolicyModule::new(
                    caps.allocation_updated_inbox(),
                    caps.cognition_log_updated_inbox(),
                    caps.allocation_reader(),
                    caps.blackboard_reader(),
                    caps.policy_searcher(),
                    caps.typed_memo::<nuillu_module::PolicyRetrievalMemo>(),
                    caps.llm_access(),
                )
            })
            .expect("eval module registration should be unique"),
        // File search is heavier and not on the speak critical path.
        EvalModule::QueryAgentic => registry
            .register(eval_policy(0..=1, Bpm::range(2.0, 6.0)), |caps| {
                nuillu_query_agentic::QueryAgenticModule::new(
                    caps.allocation_updated_inbox(),
                    caps.allocation_reader(),
                    caps.blackboard_reader(),
                    caps.file_searcher(),
                    caps.memo(),
                    caps.llm_access(),
                )
            })
            .expect("eval module registration should be unique"),
        // Background durability writer. Cognition-log triggered.
        EvalModule::Memory => registry
            .register(eval_policy(0..=1, Bpm::range(6.0, 18.0)), |caps| {
                nuillu_memory::MemoryModule::new(
                    caps.cognition_log_updated_inbox(),
                    caps.allocation_updated_inbox(),
                    caps.allocation_reader(),
                    caps.blackboard_reader(),
                    caps.memory_writer(),
                    caps.llm_access(),
                )
            })
            .expect("eval module registration should be unique"),
        // Rare; runs on allocation guidance only.
        EvalModule::MemoryCompaction => registry
            .register(eval_policy(0..=1, Bpm::range(2.0, 6.0)), |caps| {
                nuillu_memory_compaction::MemoryCompactionModule::new(
                    caps.allocation_updated_inbox(),
                    caps.allocation_reader(),
                    caps.blackboard_reader(),
                    caps.memory_compactor(),
                    caps.llm_access(),
                )
            })
            .expect("eval module registration should be unique"),
        EvalModule::Policy => registry
            .register(eval_policy(0..=1, Bpm::range(2.0, 6.0)), |caps| {
                nuillu_policy::PolicyModule::new(
                    caps.cognition_log_updated_inbox(),
                    caps.allocation_updated_inbox(),
                    caps.blackboard_reader(),
                    caps.allocation_reader(),
                    caps.policy_writer(),
                    caps.llm_access(),
                )
            })
            .expect("eval module registration should be unique"),
        EvalModule::ValueEstimator => registry
            .register(eval_policy(0..=1, Bpm::range(6.0, 15.0)), |caps| {
                nuillu_value_estimator::ValueEstimatorModule::new(
                    caps.memo_updated_inbox(),
                    caps.cognition_log_updated_inbox(),
                    caps.allocation_updated_inbox(),
                    caps.blackboard_reader(),
                    caps.cognition_log_reader(),
                    caps.allocation_reader(),
                    caps.policy_window_reader(),
                    caps.typed_memo::<nuillu_module::ValueEstimateMemo>(),
                    caps.llm_access(),
                )
            })
            .expect("eval module registration should be unique"),
        EvalModule::Reward => registry
            .register(eval_policy(0..=1, Bpm::range(3.0, 9.0)), |caps| {
                nuillu_reward::RewardModule::new(
                    caps.cognition_log_updated_inbox(),
                    caps.memo_updated_inbox(),
                    caps.allocation_updated_inbox(),
                    caps.blackboard_reader(),
                    caps.cognition_log_reader(),
                    caps.allocation_reader(),
                    caps.policy_window_reader(),
                    caps.policy_value_updater(),
                    caps.attention_control_mailbox(),
                    caps.typed_memo::<nuillu_reward::RewardMemo>(),
                    caps.llm_access(),
                )
            })
            .expect("eval module registration should be unique"),
        // Cognition-log triggered; not on speak critical path.
        EvalModule::Predict => registry
            .register(eval_policy(0..=1, Bpm::range(6.0, 18.0)), |caps| {
                nuillu_predict::PredictModule::new(
                    caps.cognition_log_updated_inbox(),
                    caps.cognition_log_reader(),
                    caps.allocation_reader(),
                    caps.blackboard_reader(),
                    caps.memo(),
                    caps.llm_access(),
                )
            })
            .expect("eval module registration should be unique"),
        // Cognition-log triggered; should be quick enough to flag
        // unexpected events while they're still relevant.
        EvalModule::Surprise => registry
            .register(eval_policy(0..=1, Bpm::range(6.0, 18.0)), |caps| {
                nuillu_surprise::SurpriseModule::new(
                    caps.cognition_log_updated_inbox(),
                    caps.cognition_log_reader(),
                    caps.allocation_reader(),
                    caps.blackboard_reader(),
                    caps.attention_control_mailbox(),
                    caps.memo(),
                    caps.llm_access(),
                )
            })
            .expect("eval module registration should be unique"),
        // On the critical path: must respond quickly to cognition-log
        // updates so it can decide to speak before the moment passes.
        EvalModule::SpeakGate => registry
            .register(eval_policy(0..=1, Bpm::range(6.0, 18.0)), |caps| {
                nuillu_speak::SpeakGateModule::new(
                    caps.activation_gate_for::<nuillu_speak::SpeakModule>(),
                    caps.cognition_log_reader(),
                    caps.blackboard_reader(),
                    caps.attention_control_mailbox(),
                    caps.typed_memo::<nuillu_speak::SpeakGateMemo>(),
                    caps.llm_access(),
                )
            })
            .expect("eval module registration should be unique"),
        // Reactive on cognition-log updates after speak-gate allows the
        // activation. Match speak-gate so the pair stays in sync.
        EvalModule::Speak => registry
            .register(eval_policy(0..=1, Bpm::range(6.0, 18.0)), |caps| {
                nuillu_speak::SpeakModule::new(
                    caps.cognition_log_updated_inbox(),
                    caps.cognition_log_reader(),
                    caps.memo(),
                    caps.utterance_writer(),
                    caps.llm_access(),
                    caps.scene_reader(),
                )
            })
            .expect("eval module registration should be unique"),
    }
}

fn full_agent_allocation(
    _limits: &crate::cases::EvalLimits,
    modules: &[EvalModule],
) -> ResourceAllocation {
    let mut allocation = ResourceAllocation::default();
    allocation.set_activation_table(eval_activation_table());

    for module in modules {
        match module {
            EvalModule::Sensory => set_allocation_module(
                &mut allocation,
                module.module_id(),
                1.0,
                ModelTier::Cheap,
                "Queued sensory input is waiting; activate when the controller is ready to process external observations.",
            ),
            EvalModule::CognitionGate => set_allocation_module(
                &mut allocation,
                module.module_id(),
                0.0,
                ModelTier::Cheap,
                "Wait for memo or controller guidance before promoting relevant memos into cognition.",
            ),
            EvalModule::AttentionController => set_allocation_module(
                &mut allocation,
                module.module_id(),
                1.0,
                ModelTier::Default,
                "Bootstrap the case: activate sensory first so queued external observations become memo logs, then allocate cognition, query, and speech modules as evidence becomes ready.",
            ),
            EvalModule::AttentionSchema => set_allocation_module(
                &mut allocation,
                module.module_id(),
                0.0,
                ModelTier::Default,
                "Idle until memo, allocation, or cognition-log updates require attention-experience integration.",
            ),
            EvalModule::SelfModel => set_allocation_module(
                &mut allocation,
                module.module_id(),
                0.0,
                ModelTier::Default,
                "Idle until explicit self-model requests require work.",
            ),
            EvalModule::QueryVector => set_allocation_module(
                &mut allocation,
                module.module_id(),
                0.0,
                ModelTier::Cheap,
                "Idle until memory retrieval is needed.",
            ),
            EvalModule::QueryPolicy => set_allocation_module(
                &mut allocation,
                module.module_id(),
                0.0,
                ModelTier::Cheap,
                "Idle until policy retrieval is needed.",
            ),
            EvalModule::QueryAgentic => set_allocation_module(
                &mut allocation,
                module.module_id(),
                0.0,
                ModelTier::Premium,
                "Idle until file lookup is needed.",
            ),
            EvalModule::Memory => set_allocation_module(
                &mut allocation,
                module.module_id(),
                0.0,
                ModelTier::Cheap,
                "Idle until preservation guidance or memory requests arrive.",
            ),
            EvalModule::MemoryCompaction => set_allocation_module(
                &mut allocation,
                module.module_id(),
                0.0,
                ModelTier::Cheap,
                "Idle until compaction guidance arrives.",
            ),
            EvalModule::Policy => set_allocation_module(
                &mut allocation,
                module.module_id(),
                0.0,
                ModelTier::Default,
                "Idle until policy formation guidance or distinctive outcomes arrive.",
            ),
            EvalModule::ValueEstimator => set_allocation_module(
                &mut allocation,
                module.module_id(),
                0.0,
                ModelTier::Cheap,
                "Idle until query-policy retrieval windows need value estimates.",
            ),
            EvalModule::Reward => set_allocation_module(
                &mut allocation,
                module.module_id(),
                0.0,
                ModelTier::Default,
                "Idle until outcomes settle value-estimate windows.",
            ),
            EvalModule::Predict => set_allocation_module(
                &mut allocation,
                module.module_id(),
                0.0,
                ModelTier::Cheap,
                "Idle until prediction guidance arrives.",
            ),
            EvalModule::Surprise => set_allocation_module(
                &mut allocation,
                module.module_id(),
                0.0,
                ModelTier::Default,
                "Idle until surprise detection is useful.",
            ),
            EvalModule::SpeakGate => set_allocation_module(
                &mut allocation,
                module.module_id(),
                0.0,
                ModelTier::Premium,
                "Idle until cognition contains the evidence needed for speech readiness.",
            ),
            EvalModule::Speak => set_allocation_module(
                &mut allocation,
                module.module_id(),
                0.0,
                ModelTier::Premium,
                "Idle until cognition-log updates are allowed through speak-gate.",
            ),
        }
    }
    allocation
}

fn full_agent_gui_initial_allocation(
    limits: &crate::cases::EvalLimits,
    modules: &[EvalModule],
) -> ResourceAllocation {
    let mut allocation = full_agent_allocation(limits, modules);
    for module in modules {
        allocation.set_activation(module.module_id(), ActivationRatio::ZERO);
    }
    allocation
}

fn module_allocation(
    target: ModuleEvalTarget,
    _limits: &crate::cases::EvalLimits,
    modules: &[EvalModule],
) -> ResourceAllocation {
    let mut allocation = ResourceAllocation::default();
    allocation.set_activation_table(eval_activation_table());
    let target_module = target.module();
    for module in modules {
        let is_target = *module == target_module;
        let id = module.module_id();
        allocation.set_model_override(id.clone(), eval_module_tier(*module));
        allocation.set(
            id.clone(),
            ModuleConfig {
                guidance: if is_target {
                    "Handle the module eval request.".into()
                } else {
                    "Registered for this module eval; idle unless activated by allocation guidance."
                        .into()
                },
            },
        );
        allocation.set_activation(
            id,
            if is_target {
                ActivationRatio::ONE
            } else {
                ActivationRatio::ZERO
            },
        );
    }
    allocation
}

fn set_allocation_module(
    allocation: &mut ResourceAllocation,
    id: ModuleId,
    activation_ratio: f64,
    tier: ModelTier,
    guidance: impl Into<String>,
) {
    allocation.set_model_override(id.clone(), tier);
    allocation.set(
        id.clone(),
        ModuleConfig {
            guidance: guidance.into(),
        },
    );
    allocation.set_activation(id, ActivationRatio::from_f64(activation_ratio));
}

fn eval_activation_table() -> Vec<ActivationRatio> {
    [1.0, 0.85, 0.7, 0.5, 0.3, 0.0]
        .into_iter()
        .map(ActivationRatio::from_f64)
        .collect()
}

fn module_id_for_target(target: ModuleEvalTarget) -> ModuleId {
    target.module().module_id()
}

fn eval_module_tier(module: EvalModule) -> ModelTier {
    match module {
        EvalModule::Sensory
        | EvalModule::CognitionGate
        | EvalModule::QueryVector
        | EvalModule::QueryPolicy
        | EvalModule::Memory
        | EvalModule::MemoryCompaction
        | EvalModule::ValueEstimator
        | EvalModule::Predict => ModelTier::Cheap,
        EvalModule::QueryAgentic | EvalModule::SpeakGate | EvalModule::Speak => ModelTier::Premium,
        EvalModule::AttentionController => ModelTier::Default,
        EvalModule::AttentionSchema
        | EvalModule::SelfModel
        | EvalModule::Policy
        | EvalModule::Reward
        | EvalModule::Surprise => ModelTier::Default,
    }
}

async fn add_observations(
    artifact: &mut CaseArtifact,
    blackboard: &Blackboard,
    utterances: &RecordingUtteranceSink,
) {
    let observations = blackboard
        .read(|bb| AgentObservation::from_blackboard(bb, utterances.snapshot()))
        .await;
    let observations = match serde_json::to_value(observations) {
        Ok(value) => value,
        Err(error) => serde_json::json!({
            "serialization_error": error.to_string(),
        }),
    };
    artifact
        .observations
        .insert("agent".to_string(), observations);
}

async fn build_full_agent_last_state_dump(
    case_id: &str,
    artifact: &CaseArtifact,
    blackboard: &Blackboard,
    memory: &dyn MemoryStore,
    utterances: &RecordingUtteranceSink,
    event_count: usize,
) -> Result<FullAgentLastStateDump> {
    let (blackboard_dump, memory_metadata) = blackboard
        .read(|bb| {
            (
                blackboard_last_state_dump(bb),
                memory_metadata_dump_records(bb),
            )
        })
        .await;
    let memory_dump = memory_last_state_dump(memory_metadata, memory).await?;
    Ok(FullAgentLastStateDump {
        case: FullAgentLastStateCaseDump {
            id: case_id.to_string(),
            dumped_at: Utc::now().to_rfc3339(),
            event_count: event_count as u64,
            output: (!artifact.output.is_empty()).then(|| DumpText::new(artifact.output.clone())),
            failure: artifact.failure.clone().map(DumpText::new),
        },
        blackboard: blackboard_dump,
        memory: memory_dump,
        utterances: utterance_dumps(utterances.snapshot()),
    })
}

fn add_last_state_observation(
    artifact: &mut CaseArtifact,
    last_state: &FullAgentLastStateDump,
) -> Result<()> {
    let value = serde_json::to_value(last_state).context("serialize last state observation")?;
    artifact
        .observations
        .insert("last_state".to_string(), value);
    Ok(())
}

fn write_full_agent_last_state_eure(
    output_dir: &Path,
    last_state: FullAgentLastStateDump,
) -> Result<()> {
    let path = output_dir.join("last-state.eure");
    let rendered = render_full_agent_last_state_eure(last_state)
        .context("render full-agent last state Eure")?;
    std::fs::write(&path, rendered)
        .with_context(|| format!("write full-agent last state dump to {}", path.display()))
}

fn blackboard_last_state_dump(bb: &BlackboardInner) -> BlackboardLastStateDump {
    let cognition_log_set = bb.cognition_log_set();
    BlackboardLastStateDump {
        memo_logs: memo_log_dumps(bb),
        cognition_logs: cognition_log_set
            .logs()
            .iter()
            .map(|record| CognitionLogDump {
                source: module_instance_dump(&record.source),
                entries: record
                    .entries
                    .iter()
                    .map(|event| CognitionEntryDump {
                        at: event.at.to_rfc3339(),
                        text: DumpText::new(event.text.clone()),
                    })
                    .collect(),
            })
            .collect(),
        agentic_deadlock: cognition_log_set.agentic_deadlock_marker().map(|marker| {
            AgenticDeadlockDump {
                at: marker.at.to_rfc3339(),
                idle_for_ms: duration_millis_u64(marker.idle_for),
            }
        }),
        base_allocation: allocation_module_dumps(bb.base_allocation()),
        allocation: allocation_module_dumps(bb.allocation()),
        allocation_proposals: allocation_proposal_dumps(bb),
        replica_caps: replica_cap_dumps(bb),
    }
}

fn visualizer_blackboard_snapshot(bb: &BlackboardInner) -> BlackboardSnapshot {
    let cognition_log_set = bb.cognition_log_set();
    let mut memory_metadata = bb
        .memory_metadata()
        .iter()
        .map(|(index, metadata)| MemoryMetadataView {
            index: index.as_str().to_owned(),
            rank: memory_rank_name(metadata.rank).to_owned(),
            occurred_at: metadata.occurred_at,
            last_accessed: metadata.last_accessed,
            access_count: metadata.access_count,
        })
        .collect::<Vec<_>>();
    memory_metadata.sort_by(|left, right| left.index.cmp(&right.index));

    BlackboardSnapshot {
        module_statuses: bb
            .module_status_records()
            .into_iter()
            .map(|record| ModuleStatusView {
                owner: record.owner.to_string(),
                module: record.owner.module.as_str().to_owned(),
                replica: record.owner.replica.get(),
                status: format!("{:?}", record.status),
            })
            .collect(),
        allocation: allocation_module_dumps(bb.allocation())
            .into_iter()
            .map(|module| AllocationView {
                bpm: ModuleId::new(module.module.clone())
                    .ok()
                    .and_then(|id| bb.allocation().cooldown_for(&id))
                    .and_then(bpm_from_cooldown),
                cooldown_ms: ModuleId::new(module.module.clone())
                    .ok()
                    .and_then(|id| bb.allocation().cooldown_for(&id))
                    .map(duration_millis_u64),
                module: module.module,
                activation_ratio: module.activation_ratio,
                active_replicas: module.active_replicas,
                tier: module.tier,
                guidance: module.guidance.as_str().to_owned(),
            })
            .collect(),
        memos: bb
            .recent_memo_logs()
            .into_iter()
            .map(|record| MemoView {
                owner: record.owner.to_string(),
                module: record.owner.module.as_str().to_owned(),
                replica: record.owner.replica.get(),
                index: record.index,
                written_at: record.written_at,
                content: record.content,
            })
            .collect(),
        cognition_logs: cognition_log_set
            .logs()
            .iter()
            .map(|record| CognitionLogView {
                source: record.source.to_string(),
                entries: record
                    .entries
                    .iter()
                    .map(|entry| CognitionEntryView {
                        at: entry.at,
                        text: entry.text.clone(),
                    })
                    .collect(),
            })
            .collect(),
        utterance_progresses: bb
            .utterance_progress_records()
            .into_iter()
            .map(|record| UtteranceProgressView {
                owner: record.owner.to_string(),
                target: record.progress.target,
                generation_id: record.progress.generation_id,
                sequence: record.progress.sequence,
                state: format!("{:?}", record.progress.state),
                partial_utterance: record.progress.partial_utterance,
            })
            .collect(),
        memory_metadata,
    }
}

fn memo_log_dumps(bb: &BlackboardInner) -> Vec<MemoLogDump> {
    bb.recent_memo_logs()
        .into_iter()
        .map(|record| MemoLogDump {
            module: record.owner.module.as_str().to_owned(),
            replica: record.owner.replica.get(),
            index: record.index,
            written_at: record.written_at.to_rfc3339(),
            content: DumpText::new(record.content),
        })
        .collect()
}

fn allocation_module_dumps(allocation: &ResourceAllocation) -> Vec<AllocationModuleDump> {
    let mut modules = allocation
        .iter()
        .map(|(module, config)| AllocationModuleDump {
            module: module.as_str().to_owned(),
            activation_ratio: allocation.activation_for(module).as_f64(),
            active_replicas: allocation.active_replicas(module),
            tier: model_tier_name(allocation.tier_for(module)).to_owned(),
            guidance: DumpText::new(config.guidance.clone()),
        })
        .collect::<Vec<_>>();
    modules.sort_by(|left, right| left.module.cmp(&right.module));
    modules
}

fn allocation_proposal_dumps(bb: &BlackboardInner) -> Vec<AllocationProposalDump> {
    let mut proposals = bb
        .allocation_proposals()
        .iter()
        .map(|(controller, proposal)| AllocationProposalDump {
            controller: module_instance_dump(controller),
            modules: allocation_module_dumps(proposal),
        })
        .collect::<Vec<_>>();
    proposals.sort_by(|left, right| {
        left.controller
            .module
            .cmp(&right.controller.module)
            .then_with(|| left.controller.replica.cmp(&right.controller.replica))
    });
    proposals
}

fn replica_cap_dumps(bb: &BlackboardInner) -> Vec<ReplicaCapDump> {
    let mut caps = bb
        .module_policies()
        .iter()
        .map(|(module, policy)| ReplicaCapDump {
            module: module.as_str().to_owned(),
            min: policy.replicas_range.min,
            max: policy.replicas_range.max,
        })
        .collect::<Vec<_>>();
    caps.sort_by(|left, right| left.module.cmp(&right.module));
    caps
}

fn memory_metadata_dump_records(bb: &BlackboardInner) -> Vec<(String, MemoryMetadataDump)> {
    let mut records = bb
        .memory_metadata()
        .iter()
        .map(|(index, metadata)| {
            (
                index.as_str().to_owned(),
                MemoryMetadataDump {
                    rank: memory_rank_name(metadata.rank).to_owned(),
                    occurred_at: metadata.occurred_at.map(|at| at.to_rfc3339()),
                    decay_remaining_secs: metadata.decay_remaining_secs,
                    remember_tokens: metadata.remember_tokens,
                    last_accessed: metadata.last_accessed.to_rfc3339(),
                    access_count: metadata.access_count,
                    query_history: metadata
                        .query_history
                        .iter()
                        .map(|at| at.to_rfc3339())
                        .collect(),
                },
            )
        })
        .collect::<Vec<_>>();
    records.sort_by(|left, right| left.0.cmp(&right.0));
    records
}

async fn memory_last_state_dump(
    metadata_records: Vec<(String, MemoryMetadataDump)>,
    memory: &dyn MemoryStore,
) -> Result<MemoryLastStateDump> {
    let mut entries = Vec::with_capacity(metadata_records.len());
    for (index, metadata) in metadata_records {
        let memory_index = nuillu_types::MemoryIndex::new(index.clone());
        let record = memory
            .get(&memory_index)
            .await
            .with_context(|| format!("read memory content for {index}"))?;
        entries.push(match record {
            Some(record) => MemoryEntryDump {
                index,
                content: Some(DumpText::new(record.content.as_str().to_owned())),
                content_rank: Some(memory_rank_name(record.rank).to_owned()),
                occurred_at: record.occurred_at.map(|at| at.to_rfc3339()),
                metadata,
                missing_content: false,
            },
            None => MemoryEntryDump {
                index,
                content: None,
                content_rank: None,
                occurred_at: None,
                metadata,
                missing_content: true,
            },
        });
    }
    Ok(MemoryLastStateDump { entries })
}

fn utterance_dumps(utterances: Vec<RecordedUtterance>) -> Vec<UtteranceDump> {
    utterances
        .into_iter()
        .map(|utterance| UtteranceDump {
            sender: utterance.sender,
            target: utterance.target,
            text: DumpText::new(utterance.text),
            emitted_at: utterance.emitted_at,
        })
        .collect()
}

#[derive(Debug, Clone, Serialize)]
struct AgentObservation {
    memo_logs: BTreeMap<String, Vec<MemoLogObservation>>,
    cognition_logs: Vec<CognitionLogObservation>,
    allocation: BTreeMap<String, AllocationModuleObservation>,
    allocation_proposals: BTreeMap<String, BTreeMap<String, AllocationModuleObservation>>,
    replica_caps: BTreeMap<String, ReplicaCapRange>,
    memory_metadata: BTreeMap<String, MemoryMetadata>,
    utterances: Vec<RecordedUtterance>,
}

impl AgentObservation {
    fn from_blackboard(bb: &BlackboardInner, utterances: Vec<RecordedUtterance>) -> Self {
        Self {
            memo_logs: memo_log_observations(bb),
            cognition_logs: cognition_log_observations(bb),
            allocation: allocation_observation(bb.allocation()),
            allocation_proposals: allocation_proposal_observations(bb),
            replica_caps: replica_cap_observations(bb),
            memory_metadata: memory_metadata_observations(bb),
            utterances,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
struct AllocationModuleObservation {
    activation_ratio: ActivationRatio,
    guidance: String,
    tier: ModelTier,
}

#[derive(Debug, Clone, Serialize)]
struct MemoLogObservation {
    replica: u8,
    index: u64,
    written_at: String,
    content: String,
}

#[derive(Debug, Clone, Serialize)]
struct CognitionLogObservation {
    source: ModuleInstanceObservation,
    entries: Vec<CognitionLogEntry>,
}

#[derive(Debug, Clone, Serialize)]
struct ModuleInstanceObservation {
    module: String,
    replica: u8,
}

#[derive(Debug, Clone, Serialize)]
struct ActiveModuleObservation {
    module: String,
    active_replicas: u8,
    activation_ratio: ActivationRatio,
    tier: ModelTier,
}

fn memo_log_observations(bb: &BlackboardInner) -> BTreeMap<String, Vec<MemoLogObservation>> {
    let mut logs = BTreeMap::<String, Vec<MemoLogObservation>>::new();
    for record in bb.recent_memo_logs() {
        logs.entry(record.owner.module.as_str().to_owned())
            .or_default()
            .push(MemoLogObservation {
                replica: record.owner.replica.get(),
                index: record.index,
                written_at: record.written_at.to_rfc3339(),
                content: record.content,
            });
    }
    logs
}

fn cognition_log_observations(bb: &BlackboardInner) -> Vec<CognitionLogObservation> {
    bb.cognition_log_set()
        .logs()
        .iter()
        .map(|record| CognitionLogObservation {
            source: module_instance_observation(&record.source),
            entries: record.entries.clone(),
        })
        .collect()
}

fn allocation_observation(
    allocation: &ResourceAllocation,
) -> BTreeMap<String, AllocationModuleObservation> {
    allocation
        .iter()
        .map(|(module, config)| {
            (
                module.as_str().to_owned(),
                AllocationModuleObservation {
                    activation_ratio: allocation.activation_for(module),
                    guidance: config.guidance.clone(),
                    tier: allocation.tier_for(module),
                },
            )
        })
        .collect()
}

fn allocation_proposal_observations(
    bb: &BlackboardInner,
) -> BTreeMap<String, BTreeMap<String, AllocationModuleObservation>> {
    bb.allocation_proposals()
        .iter()
        .map(|(owner, allocation)| (owner.to_string(), allocation_observation(allocation)))
        .collect()
}

fn replica_cap_observations(bb: &BlackboardInner) -> BTreeMap<String, ReplicaCapRange> {
    bb.module_policies()
        .iter()
        .map(|(module, policy)| (module.as_str().to_owned(), policy.replicas_range))
        .collect()
}

fn active_module_observations(bb: &BlackboardInner) -> Vec<ActiveModuleObservation> {
    let mut modules = bb
        .module_policies()
        .keys()
        .cloned()
        .chain(bb.allocation().iter().map(|(module, _)| module.clone()))
        .collect::<Vec<_>>();
    modules.sort_by(|left, right| left.as_str().cmp(right.as_str()));
    modules.dedup();
    modules
        .into_iter()
        .filter_map(|module| {
            let active_replicas = bb.allocation().active_replicas(&module);
            if active_replicas == 0 {
                return None;
            }
            Some(ActiveModuleObservation {
                module: module.as_str().to_owned(),
                active_replicas,
                activation_ratio: bb.allocation().activation_for(&module),
                tier: bb.allocation().tier_for(&module),
            })
        })
        .collect()
}

fn memory_metadata_observations(bb: &BlackboardInner) -> BTreeMap<String, MemoryMetadata> {
    bb.memory_metadata()
        .iter()
        .map(|(index, metadata)| (index.as_str().to_owned(), metadata.clone()))
        .collect()
}

fn module_instance_observation(owner: &ModuleInstanceId) -> ModuleInstanceObservation {
    ModuleInstanceObservation {
        module: owner.module.as_str().to_owned(),
        replica: owner.replica.get(),
    }
}

fn module_instance_dump(owner: &ModuleInstanceId) -> ModuleInstanceDump {
    ModuleInstanceDump {
        module: owner.module.as_str().to_owned(),
        replica: owner.replica.get(),
    }
}

fn memory_rank_name(rank: MemoryRank) -> &'static str {
    match rank {
        MemoryRank::ShortTerm => "short-term",
        MemoryRank::MidTerm => "mid-term",
        MemoryRank::LongTerm => "long-term",
        MemoryRank::Permanent => "permanent",
        MemoryRank::Identity => "identity",
    }
}

fn model_tier_name(tier: ModelTier) -> &'static str {
    match tier {
        ModelTier::Cheap => "cheap",
        ModelTier::Default => "default",
        ModelTier::Premium => "premium",
    }
}

struct AllocationChangeReporter {
    case_id: String,
    reporter: LiveReporter,
    last: Option<String>,
}

impl AllocationChangeReporter {
    fn new(case_id: String, reporter: LiveReporter) -> Self {
        Self {
            case_id,
            reporter,
            last: None,
        }
    }

    async fn emit_if_changed(&mut self, blackboard: &Blackboard) -> Result<(), RunnerError> {
        let allocation = blackboard
            .read(|bb| allocation_observation(bb.allocation()))
            .await;
        let value = serde_json::to_value(&allocation).map_err(|error| RunnerError::Driver {
            path: PathBuf::from(&self.case_id),
            message: error.to_string(),
        })?;
        let signature = serde_json::to_string(&value).map_err(|error| RunnerError::Driver {
            path: PathBuf::from(&self.case_id),
            message: error.to_string(),
        })?;
        if self.last.as_deref() == Some(signature.as_str()) {
            return Ok(());
        }
        self.last = Some(signature);
        let live = format!(
            "eval allocation case={} {}",
            self.case_id,
            allocation_live_summary(&allocation)
        );
        self.reporter.emit(
            Some(&self.case_id),
            "allocation_changed",
            serde_json::json!({ "allocation": value }),
            live,
        )
    }
}

fn allocation_live_summary(allocation: &BTreeMap<String, AllocationModuleObservation>) -> String {
    let active = allocation
        .iter()
        .filter(|(_, obs)| obs.activation_ratio > ActivationRatio::ZERO)
        .map(|(module, obs)| {
            format!(
                "{}:{:.2}/{:?}",
                module,
                obs.activation_ratio.as_f64(),
                obs.tier
            )
        })
        .collect::<Vec<_>>();
    let inactive = allocation
        .values()
        .filter(|obs| obs.activation_ratio == ActivationRatio::ZERO)
        .count();
    format!("active=[{}] inactive={inactive}", active.join(","))
}

fn active_modules_live_summary(active_modules: &[ActiveModuleObservation]) -> String {
    if active_modules.is_empty() {
        return "none".to_owned();
    }
    active_modules
        .iter()
        .map(|module| {
            format!(
                "{}:{}:{:.2}/{:?}",
                module.module,
                module.active_replicas,
                module.activation_ratio.as_f64(),
                module.tier
            )
        })
        .collect::<Vec<_>>()
        .join(",")
}

fn ticks_for_interval(interval: Duration, tick_ms: u64) -> u64 {
    (duration_millis_u64(interval) / tick_ms.max(1)).max(1)
}

fn duration_millis_u64(duration: Duration) -> u64 {
    duration.as_millis().min(u128::from(u64::MAX)) as u64
}

fn bpm_from_cooldown(cooldown: Duration) -> Option<f64> {
    let seconds = cooldown.as_secs_f64();
    (seconds.is_finite() && seconds > 0.0).then_some(60.0 / seconds)
}

#[derive(Clone)]
struct LiveReporter {
    run_id: String,
    path: PathBuf,
    file: Arc<Mutex<File>>,
}

impl std::fmt::Debug for LiveReporter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LiveReporter")
            .field("run_id", &self.run_id)
            .field("path", &self.path)
            .finish_non_exhaustive()
    }
}

impl LiveReporter {
    fn new(run_id: &str, run_dir: &Path) -> Result<Self, RunnerError> {
        let path = run_dir.join("events.jsonl");
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .map_err(|source| RunnerError::WriteOutput { path, source })?;
        Ok(Self {
            run_id: run_id.to_string(),
            path: run_dir.join("events.jsonl"),
            file: Arc::new(Mutex::new(file)),
        })
    }

    fn emit(
        &self,
        case_id: Option<&str>,
        kind: &str,
        data: serde_json::Value,
        live_message: String,
    ) -> Result<(), RunnerError> {
        self.emit_jsonl(case_id, kind, data, live_message)
            .map_err(|source| RunnerError::WriteOutput {
                path: self.path.clone(),
                source,
            })
    }

    fn emit_port(
        &self,
        case_id: Option<&str>,
        kind: &str,
        data: serde_json::Value,
        live_message: String,
    ) -> Result<(), PortError> {
        self.emit_jsonl(case_id, kind, data, live_message)
            .map_err(|error| PortError::Backend(format!("write live eval event: {error}")))
    }

    fn emit_jsonl(
        &self,
        case_id: Option<&str>,
        kind: &str,
        data: serde_json::Value,
        live_message: String,
    ) -> io::Result<()> {
        eprintln!("{live_message}");
        let record = serde_json::json!({
            "ts": Utc::now().to_rfc3339(),
            "run_id": self.run_id,
            "case_id": case_id,
            "kind": kind,
            "data": data,
        });
        let mut file = self
            .file
            .lock()
            .map_err(|_| io::Error::other("events.jsonl lock poisoned"))?;
        serde_json::to_writer(&mut *file, &record).map_err(io::Error::other)?;
        file.write_all(b"\n")?;
        file.flush()
    }
}

#[derive(Debug, Clone, Serialize)]
struct RecordedUtterance {
    sender: String,
    target: String,
    text: String,
    emitted_at: String,
}

#[derive(Clone)]
struct ActionActivityTracker {
    action_modules: Arc<HashSet<ModuleId>>,
    last_completed_at: Arc<Mutex<Option<Instant>>>,
}

impl ActionActivityTracker {
    fn new(action_modules: Vec<ModuleId>) -> Self {
        Self {
            action_modules: Arc::new(action_modules.into_iter().collect()),
            last_completed_at: Arc::new(Mutex::new(None)),
        }
    }

    fn record_completed(&self, module: &ModuleId) {
        self.record_completed_at(module, Instant::now());
    }

    fn record_completed_at(&self, module: &ModuleId, completed_at: Instant) {
        if !self.action_modules.contains(module) {
            return;
        }
        let mut last_completed_at = self
            .last_completed_at
            .lock()
            .expect("action activity lock poisoned");
        let should_update = match *last_completed_at {
            Some(previous) => completed_at >= previous,
            None => true,
        };
        if should_update {
            *last_completed_at = Some(completed_at);
        }
    }

    fn silence_window_elapsed(&self, window: Duration) -> bool {
        self.silence_window_elapsed_at(window, Instant::now())
    }

    fn silence_window_elapsed_at(&self, window: Duration, now: Instant) -> bool {
        let last_completed_at = *self
            .last_completed_at
            .lock()
            .expect("action activity lock poisoned");
        last_completed_at.is_some_and(|completed_at| {
            now.checked_duration_since(completed_at)
                .is_some_and(|elapsed| elapsed >= window)
        })
    }
}

#[derive(Clone)]
struct RecordingUtteranceSink {
    case_id: String,
    reporter: LiveReporter,
    actions: Arc<ActionActivityTracker>,
    complete: Arc<Mutex<Vec<RecordedUtterance>>>,
    visualizer: Option<VisualizerEventSink>,
}

fn normalize_eval_utterance_text(text: String) -> String {
    let trimmed = text.trim();
    if trimmed.starts_with('"')
        && trimmed.ends_with('"')
        && let Ok(unquoted) = serde_json::from_str::<String>(trimmed)
    {
        return unquoted;
    }
    text
}

impl std::fmt::Debug for RecordingUtteranceSink {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RecordingUtteranceSink")
            .field("case_id", &self.case_id)
            .finish_non_exhaustive()
    }
}

impl RecordingUtteranceSink {
    fn new(
        case_id: String,
        reporter: LiveReporter,
        actions: Arc<ActionActivityTracker>,
        visualizer: Option<VisualizerEventSink>,
    ) -> Self {
        Self {
            case_id,
            reporter,
            actions,
            complete: Arc::new(Mutex::new(Vec::new())),
            visualizer,
        }
    }

    fn last_complete(&self) -> Option<RecordedUtterance> {
        self.complete
            .lock()
            .expect("utterance lock poisoned")
            .last()
            .cloned()
    }

    fn snapshot(&self) -> Vec<RecordedUtterance> {
        self.complete
            .lock()
            .expect("utterance lock poisoned")
            .clone()
    }
}

#[async_trait(?Send)]
impl UtteranceSink for RecordingUtteranceSink {
    async fn on_complete(&self, utterance: Utterance) -> Result<(), PortError> {
        let sender_module = utterance.sender.module.clone();
        let recorded = RecordedUtterance {
            sender: utterance.sender.to_string(),
            target: utterance.target,
            text: normalize_eval_utterance_text(utterance.text),
            emitted_at: utterance.emitted_at.to_rfc3339(),
        };
        self.complete
            .lock()
            .map_err(|_| PortError::Backend("utterance lock poisoned".into()))?
            .push(recorded.clone());
        self.actions.record_completed(&sender_module);
        self.reporter.emit_port(
            Some(&self.case_id),
            "utterance_completed",
            serde_json::json!({
                "sender": recorded.sender.clone(),
                "target": recorded.target.clone(),
                "text": recorded.text.clone(),
                "emitted_at": recorded.emitted_at.clone(),
            }),
            format!(
                "eval utterance case={} sender={} target={} chars={}",
                self.case_id,
                recorded.sender,
                recorded.target,
                recorded.text.chars().count()
            ),
        )?;
        if let Some(visualizer) = &self.visualizer {
            visualizer.send(VisualizerEvent::UtteranceCompleted {
                tab_id: VisualizerTabId::new(self.case_id.clone()),
                utterance: UtteranceView {
                    sender: recorded.sender,
                    target: recorded.target,
                    text: recorded.text,
                    emitted_at: utterance.emitted_at,
                },
            });
        }
        Ok(())
    }

    async fn on_delta(&self, delta: UtteranceDelta) -> Result<(), PortError> {
        if let Some(visualizer) = &self.visualizer {
            visualizer.send(VisualizerEvent::UtteranceDelta {
                tab_id: VisualizerTabId::new(self.case_id.clone()),
                utterance: UtteranceDeltaView {
                    sender: delta.sender.to_string(),
                    target: delta.target,
                    generation_id: delta.generation_id,
                    sequence: delta.sequence,
                    delta: delta.delta,
                },
            });
        }
        Ok(())
    }
}

#[derive(Debug)]
struct RecordingRuntimeEventSink {
    events: Mutex<Vec<RuntimeEvent>>,
    stop: AtomicBool,
    case_id: String,
    max_llm_calls: Option<u64>,
    reporter: LiveReporter,
    visualizer: Option<VisualizerEventSink>,
}

impl RecordingRuntimeEventSink {
    fn new(
        case_id: String,
        max_llm_calls: Option<u64>,
        reporter: LiveReporter,
        visualizer: Option<VisualizerEventSink>,
    ) -> Self {
        Self {
            events: Mutex::new(Vec::new()),
            stop: AtomicBool::new(false),
            case_id,
            max_llm_calls,
            reporter,
            visualizer,
        }
    }

    fn snapshot(&self) -> Vec<RuntimeEvent> {
        self.events
            .lock()
            .expect("runtime event lock poisoned")
            .clone()
    }

    fn stop_requested(&self) -> bool {
        self.stop.load(Ordering::Relaxed)
    }

    fn request_stop(&self, reason: &str) {
        if !self.stop.swap(true, Ordering::Relaxed) {
            let _ = self.reporter.emit_port(
                Some(&self.case_id),
                "stop_requested",
                serde_json::json!({ "reason": reason }),
                format!(
                    "eval stop requested case={} reason={}",
                    self.case_id, reason
                ),
            );
        }
    }

    fn event_count(&self) -> usize {
        self.events
            .lock()
            .expect("runtime event lock poisoned")
            .len()
    }
}

#[async_trait(?Send)]
impl RuntimeEventSink for RecordingRuntimeEventSink {
    async fn on_event(&self, event: RuntimeEvent) -> Result<(), PortError> {
        let should_stop = match &event {
            RuntimeEvent::LlmAccessed { call, .. } => self
                .max_llm_calls
                .is_some_and(|max| call.saturating_add(1) >= max),
            RuntimeEvent::MemoUpdated { .. } => false,
            RuntimeEvent::RateLimitDelayed { .. } => false,
            RuntimeEvent::ModuleBatchThrottled { .. } => false,
        };
        let live_message = match &event {
            RuntimeEvent::LlmAccessed {
                call, owner, tier, ..
            } => format!(
                "eval llm-accessed case={} call={} owner={} tier={:?}",
                self.case_id, call, owner, tier
            ),
            RuntimeEvent::MemoUpdated {
                owner, char_count, ..
            } => format!(
                "eval memo-updated case={} owner={} chars={}",
                self.case_id, owner, char_count
            ),
            RuntimeEvent::RateLimitDelayed {
                owner,
                capability,
                delayed_for,
                ..
            } => format!(
                "eval rate-limit-delayed case={} owner={} capability={:?} delayed_ms={}",
                self.case_id,
                owner,
                capability,
                delayed_for.as_millis()
            ),
            RuntimeEvent::ModuleBatchThrottled {
                owner, delayed_for, ..
            } => format!(
                "eval module-batch-throttled case={} owner={} delayed_ms={}",
                self.case_id,
                owner,
                delayed_for.as_millis()
            ),
        };
        self.events
            .lock()
            .map_err(|_| PortError::Backend("runtime event lock poisoned".into()))?
            .push(event.clone());
        self.reporter.emit_port(
            Some(&self.case_id),
            "runtime_event",
            serde_json::json!({ "event": event }),
            live_message,
        )?;
        if let Some(visualizer) = &self.visualizer {
            visualizer.send(VisualizerEvent::RuntimeEvent {
                tab_id: VisualizerTabId::new(self.case_id.clone()),
                event,
            });
        }
        if should_stop && !self.stop.swap(true, Ordering::Relaxed) {
            self.reporter.emit_port(
                Some(&self.case_id),
                "stop_requested",
                serde_json::json!({ "reason": "max-llm-calls" }),
                format!(
                    "eval stop requested case={} reason=max-llm-calls",
                    self.case_id
                ),
            )?;
        }
        Ok(())
    }
}

fn aggregate_suite(run: SuiteRunReport, cases: Vec<CaseSummary>) -> SuiteReport {
    let case_count = cases.len();
    let passed_cases = cases.iter().filter(|case| case.passed).count();
    let invalid_cases = cases.iter().filter(|case| case.invalid).count();
    let failed_cases = case_count.saturating_sub(passed_cases + invalid_cases);
    let mean_score = if cases.is_empty() {
        0.0
    } else {
        cases.iter().map(|case| case.score).sum::<f64>() / cases.len() as f64
    };

    SuiteReport {
        run,
        case_count,
        passed_cases,
        failed_cases,
        invalid_cases,
        mean_score,
        cases,
    }
}

fn case_id(path: &Path, case: &EvalCase) -> String {
    case.id().map(String::from).unwrap_or_else(|| {
        path.with_extension("")
            .file_name()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_else(|| "case".to_string())
    })
}

fn sanitize_id(id: &str) -> String {
    id.chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_') {
                ch
            } else {
                '-'
            }
        })
        .collect()
}

fn write_json_file(path: &Path, value: &impl Serialize) -> Result<(), RunnerError> {
    let bytes = serde_json::to_vec_pretty(value).map_err(|error| RunnerError::Driver {
        path: path.to_path_buf(),
        message: error.to_string(),
    })?;
    std::fs::write(path, bytes).map_err(|source| RunnerError::WriteOutput {
        path: path.to_path_buf(),
        source,
    })
}

pub fn install_lutum_trace_subscriber() -> Result<(), RunnerError> {
    static INSTALL_RESULT: OnceLock<Result<(), String>> = OnceLock::new();
    let result = INSTALL_RESULT.get_or_init(|| {
        let subscriber = tracing_subscriber::registry().with(lutum_trace::layer());
        tracing::subscriber::set_global_default(subscriber).map_err(|error| error.to_string())
    });
    result
        .as_ref()
        .map(|_| ())
        .map_err(|message| RunnerError::TraceSubscriber {
            message: message.clone(),
        })
}

fn panic_payload_message(payload: &(dyn Any + Send)) -> String {
    if let Some(message) = payload.downcast_ref::<&str>() {
        (*message).to_string()
    } else if let Some(message) = payload.downcast_ref::<String>() {
        message.clone()
    } else {
        "non-string panic payload".to_string()
    }
}

pub fn default_run_id() -> String {
    Utc::now().format("%Y%m%dT%H%M%SZ").to_string()
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use chrono::TimeZone as _;
    use lutum::{
        FinishReason, Lutum, MockLlmAdapter, MockTextScenario, RawTextTurnEvent,
        SharedPoolBudgetManager, SharedPoolBudgetOptions, Usage,
    };
    use nuillu_blackboard::{BlackboardCommand, CognitionLogEntry, MemoryMetaPatch};
    use nuillu_module::MemoUpdated;
    use nuillu_module::ports::{
        NoopCognitionLogRepository, NoopFileSearchProvider, NoopMemoryStore, NoopPolicyStore,
        NoopUtteranceSink, SystemClock,
    };
    use nuillu_types::{MemoryIndex, ModuleInstanceId, ReplicaIndex};

    use super::*;

    struct FixedClock(chrono::DateTime<Utc>);

    #[async_trait::async_trait(?Send)]
    impl Clock for FixedClock {
        fn now(&self) -> chrono::DateTime<Utc> {
            self.0
        }

        async fn sleep_until(&self, _deadline: chrono::DateTime<Utc>) {
            // Test clock: sleeps complete immediately so test wall-clock stays
            // independent of the registered BPM cooldown ranges.
        }
    }

    fn test_backend_config() -> LlmBackendConfig {
        test_backend_config_with_model("gpt-oss:20b")
    }

    fn test_backend_config_with_model(model: &str) -> LlmBackendConfig {
        LlmBackendConfig {
            endpoint: "http://localhost:11434/v1".to_string(),
            token: "local".to_string(),
            model: model.to_string(),
            reasoning_effort: None,
            use_responses_api: false,
        }
    }

    #[test]
    fn visualizer_open_tab_uses_case_id_as_tab_id() {
        let (event_tx, event_rx) = std::sync::mpsc::channel();
        let (_command_tx, command_rx) = std::sync::mpsc::channel();
        let hooks = RunnerHooks::with_visualizer(VisualizerHook::new(event_tx, command_rx));

        emit_visualizer_open_tab(&hooks, "case-1");

        let VisualizerServerMessage::Event { event } = event_rx.recv().expect("visualizer event")
        else {
            panic!("expected visualizer event");
        };
        let VisualizerEvent::OpenTab { tab_id, title } = event else {
            panic!("expected open-tab event");
        };
        assert_eq!(tab_id.as_str(), "case-1");
        assert_eq!(title, "case-1");
    }

    #[tokio::test]
    async fn visualizer_llm_observer_emits_hook_events() {
        let (event_tx, event_rx) = std::sync::mpsc::channel();
        let observer =
            VisualizerLlmObserver::new("case-1".to_string(), VisualizerEventSink::new(event_tx));
        let owner = ModuleInstanceId::new(builtin::sensory(), ReplicaIndex::ZERO);
        let mut extensions = RequestExtensions::new();
        extensions.insert(LlmRequestMetadata {
            owner: owner.clone(),
            tier: ModelTier::Default,
            source: LlmRequestSource::ModuleTurn,
        });
        let input = lutum::ModelInput::new().user("hello");
        let input_cx = ModelInputHookContext::new(&extensions, OperationKind::TextTurn, &input);

        OnModelInput::call(&observer, &input_cx).await;

        let event = match event_rx.recv().expect("model input event") {
            VisualizerServerMessage::Event { event } => event,
            _ => panic!("expected visualizer event"),
        };
        let VisualizerEvent::LlmObserved {
            event:
                LlmObservationEvent::ModelInput {
                    turn_id,
                    owner: observed_owner,
                    items,
                    ..
                },
            ..
        } = event
        else {
            panic!("expected model input observation");
        };
        assert_eq!(observed_owner, owner.to_string());
        assert_eq!(items[0].content, "hello");

        let stream_event = ErasedTextTurnEvent::TextDelta {
            delta: "world".to_string(),
        };
        let stream_cx = StreamEventHookContext::new(
            &extensions,
            OperationKind::TextTurn,
            LutumStreamEvent::TextTurn(&stream_event),
        );

        OnStreamEvent::call(&observer, &stream_cx).await;

        let event = match event_rx.recv().expect("stream event") {
            VisualizerServerMessage::Event { event } => event,
            _ => panic!("expected visualizer event"),
        };
        let VisualizerEvent::LlmObserved {
            event:
                LlmObservationEvent::StreamDelta {
                    turn_id: delta_turn_id,
                    kind,
                    delta,
                },
            ..
        } = event
        else {
            panic!("expected stream delta observation");
        };
        assert_eq!(delta_turn_id, turn_id);
        assert_eq!(kind, "text");
        assert_eq!(delta, "world");
    }

    #[test]
    fn visualizer_hook_serves_cached_memory_query_results() {
        let (event_tx, event_rx) = std::sync::mpsc::channel();
        let (command_tx, command_rx) = std::sync::mpsc::channel();
        let mut hook = VisualizerHook::new(event_tx, command_rx);
        hook.set_memory_cache(
            "case-1",
            vec![MemoryRecordView {
                index: "m1".to_string(),
                rank: "long-term".to_string(),
                occurred_at: None,
                content: "rust memory".to_string(),
            }],
        );

        command_tx
            .send(VisualizerClientMessage::Command {
                command: VisualizerCommand::QueryMemory {
                    tab_id: VisualizerTabId::new("case-1"),
                    query: "rust".to_string(),
                    limit: 10,
                },
            })
            .unwrap();
        command_tx
            .send(VisualizerClientMessage::Command {
                command: VisualizerCommand::Shutdown,
            })
            .unwrap();
        hook.drain_cached_commands_until_shutdown();

        let VisualizerServerMessage::Event { event } = event_rx.recv().expect("memory query event")
        else {
            panic!("expected visualizer event");
        };
        let VisualizerEvent::MemoryQueryResult { records, .. } = event else {
            panic!("expected memory query result");
        };
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].content, "rust memory");
    }

    fn test_caps(blackboard: Blackboard) -> CapabilityProviders {
        test_caps_with_adapter(blackboard, MockLlmAdapter::new())
    }

    fn test_caps_with_adapter(
        blackboard: Blackboard,
        adapter: MockLlmAdapter,
    ) -> CapabilityProviders {
        let adapter = Arc::new(adapter);
        let budget = SharedPoolBudgetManager::new(SharedPoolBudgetOptions::default());
        let lutum = Lutum::new(adapter, budget);
        CapabilityProviders::new(CapabilityProviderPorts {
            blackboard,
            cognition_log_port: Arc::new(NoopCognitionLogRepository),
            primary_memory_store: Arc::new(NoopMemoryStore),
            memory_replicas: Vec::new(),
            primary_policy_store: Arc::new(NoopPolicyStore),
            policy_replicas: Vec::new(),
            file_search: Arc::new(NoopFileSearchProvider),
            utterance_sink: Arc::new(NoopUtteranceSink),
            clock: Arc::new(SystemClock),
            tiers: LutumTiers {
                cheap: lutum.clone(),
                default: lutum.clone(),
                premium: lutum,
            },
        })
    }

    fn attention_schema_tool_scenario(tool_call_id: &str, text: &str) -> MockTextScenario {
        MockTextScenario::events(vec![
            Ok(RawTextTurnEvent::Started {
                request_id: Some("attention-schema".into()),
                model: "mock".into(),
            }),
            Ok(RawTextTurnEvent::ToolCallChunk {
                id: tool_call_id.into(),
                name: "append_attention_experience".into(),
                arguments_json_delta: serde_json::json!({ "plaintext": text }).to_string(),
            }),
            Ok(RawTextTurnEvent::Completed {
                request_id: Some("attention-schema".into()),
                finish_reason: FinishReason::ToolCall,
                usage: Usage::zero(),
            }),
        ])
    }

    fn attention_schema_no_tool_scenario() -> MockTextScenario {
        MockTextScenario::events(vec![
            Ok(RawTextTurnEvent::Started {
                request_id: Some("attention-schema-noop".into()),
                model: "mock".into(),
            }),
            Ok(RawTextTurnEvent::TextDelta {
                delta: "no new attention experience".into(),
            }),
            Ok(RawTextTurnEvent::Completed {
                request_id: Some("attention-schema".into()),
                finish_reason: FinishReason::Stop,
                usage: Usage::zero(),
            }),
        ])
    }

    #[test]
    fn case_patterns_match_case_id_or_path_substrings() {
        let dir = tempfile::tempdir().unwrap();
        let case_dir = dir.path().join("eval-cases/modules/query-vector");
        std::fs::create_dir_all(&case_dir).unwrap();
        let first = case_dir.join("first-route.eure");
        let second = case_dir.join("second-memory.eure");
        std::fs::write(
            &first,
            r#"
id = "module-query-vector-first-route"
prompt = "First?"
"#,
        )
        .unwrap();
        std::fs::write(
            &second,
            r#"
id = "module-query-vector-special-memory"
prompt = "Second?"
"#,
        )
        .unwrap();

        let by_path = filter_case_paths(
            vec![first.clone(), second.clone()],
            &["first-route".to_string()],
        )
        .unwrap();
        assert_eq!(by_path, vec![first.clone()]);

        let by_id = filter_case_paths(vec![first, second.clone()], &["special-memory".to_string()])
            .unwrap();
        assert_eq!(by_id, vec![second]);
    }

    #[test]
    fn visualizer_planned_tabs_use_filtered_case_ids() {
        let dir = tempfile::tempdir().unwrap();
        let case_dir = dir.path().join("eval-cases/modules/query-vector");
        std::fs::create_dir_all(&case_dir).unwrap();
        std::fs::write(
            case_dir.join("first-route.eure"),
            r#"
id = "module-query-vector-first-route"
prompt = "First?"
"#,
        )
        .unwrap();
        std::fs::write(
            case_dir.join("second-memory.eure"),
            r#"
id = "module-query-vector-special-memory"
prompt = "Second?"
"#,
        )
        .unwrap();
        let config = RunnerConfig {
            cases_root: dir.path().join("eval-cases"),
            output_root: dir.path().join("out"),
            run_id: "run".to_string(),
            judge_backend: test_backend_config(),
            cheap_backend: test_backend_config(),
            default_backend: test_backend_config(),
            premium_backend: test_backend_config(),
            model_dir: dir.path().join("models"),
            fail_fast: false,
            max_concurrent_llm_calls: None,
            case_patterns: vec!["special-memory".to_string()],
            disabled_modules: Vec::new(),
        };

        let tabs = visualizer_planned_tabs(&config).unwrap();

        assert_eq!(tabs.len(), 1);
        assert_eq!(tabs[0].0.as_str(), "module-query-vector-special-memory");
        assert_eq!(tabs[0].1, "module-query-vector-special-memory");
    }

    #[test]
    fn eval_utterance_output_strips_balanced_outer_quotes() {
        assert_eq!(
            normalize_eval_utterance_text("\"short signal\"".to_string()),
            "short signal"
        );
        assert_eq!(
            normalize_eval_utterance_text("  \"short signal\"  ".to_string()),
            "short signal"
        );
        assert_eq!(
            normalize_eval_utterance_text("short signal".to_string()),
            "short signal"
        );
        assert_eq!(
            normalize_eval_utterance_text("\"an apple\" is \"red\"".to_string()),
            "\"an apple\" is \"red\""
        );
        assert_eq!(
            normalize_eval_utterance_text("\"an \\\"apple\\\"\"".to_string()),
            "an \"apple\""
        );
    }

    #[tokio::test]
    async fn recording_utterance_sink_returns_last_complete() {
        let dir = tempfile::tempdir().unwrap();
        let reporter = LiveReporter::new("test-run", dir.path()).unwrap();
        let actions = Arc::new(ActionActivityTracker::new(vec![builtin::speak()]));
        let sink =
            RecordingUtteranceSink::new("test-case".to_string(), reporter, actions.clone(), None);
        let emitted_at = Utc.with_ymd_and_hms(2026, 5, 7, 12, 0, 0).unwrap();
        let sender = ModuleInstanceId::new(builtin::speak(), ReplicaIndex::ZERO);

        sink.on_complete(Utterance {
            sender: sender.clone(),
            target: "peer".to_string(),
            text: "first".to_string(),
            emitted_at,
        })
        .await
        .unwrap();
        sink.on_complete(Utterance {
            sender,
            target: "peer".to_string(),
            text: "second".to_string(),
            emitted_at,
        })
        .await
        .unwrap();

        assert_eq!(sink.snapshot().len(), 2);
        assert_eq!(sink.last_complete().unwrap().text, "second");
        assert!(!actions.silence_window_elapsed(FULL_AGENT_ACTION_SILENCE_WINDOW));
    }

    #[test]
    fn action_tracker_waits_for_silence_window_after_action_completion() {
        let tracker = ActionActivityTracker::new(vec![builtin::speak()]);
        let now = Instant::now();

        tracker.record_completed_at(&builtin::speak(), now);

        assert!(!tracker.silence_window_elapsed_at(Duration::from_secs(1), now));
        assert!(
            !tracker.silence_window_elapsed_at(
                Duration::from_secs(1),
                now + Duration::from_millis(999)
            )
        );
        assert!(
            tracker.silence_window_elapsed_at(Duration::from_secs(1), now + Duration::from_secs(1))
        );
    }

    #[test]
    fn action_tracker_ignores_non_action_module_completion() {
        let tracker = ActionActivityTracker::new(vec![builtin::speak()]);
        let completed_at = Instant::now();
        let after_window = completed_at + Duration::from_secs(2);

        tracker.record_completed_at(&builtin::query_vector(), completed_at);
        assert!(!tracker.silence_window_elapsed_at(Duration::from_secs(1), after_window));

        tracker.record_completed_at(&builtin::speak(), completed_at);
        tracker.record_completed_at(&builtin::query_vector(), after_window);
        assert!(tracker.silence_window_elapsed_at(Duration::from_secs(1), after_window));
    }

    #[tokio::test]
    async fn max_llm_calls_requests_stop_after_limit_event() {
        let dir = tempfile::tempdir().unwrap();
        let reporter = LiveReporter::new("test-run", dir.path()).unwrap();
        let sink = RecordingRuntimeEventSink::new("test-case".to_string(), Some(3), reporter, None);
        let owner = ModuleInstanceId::new(builtin::query_vector(), ReplicaIndex::ZERO);

        sink.on_event(RuntimeEvent::LlmAccessed {
            sequence: 0,
            call: 0,
            owner: owner.clone(),
            tier: ModelTier::Default,
        })
        .await
        .unwrap();
        assert!(!sink.stop_requested());

        sink.on_event(RuntimeEvent::LlmAccessed {
            sequence: 1,
            call: 1,
            owner: owner.clone(),
            tier: ModelTier::Default,
        })
        .await
        .unwrap();
        assert!(!sink.stop_requested());

        sink.on_event(RuntimeEvent::LlmAccessed {
            sequence: 2,
            call: 2,
            owner: owner.clone(),
            tier: ModelTier::Default,
        })
        .await
        .unwrap();

        assert!(sink.stop_requested());
        sink.on_event(RuntimeEvent::LlmAccessed {
            sequence: 3,
            call: 3,
            owner,
            tier: ModelTier::Default,
        })
        .await
        .unwrap();
        assert_eq!(sink.snapshot().len(), 4);
        let jsonl = std::fs::read_to_string(dir.path().join("events.jsonl")).unwrap();
        assert!(jsonl.contains("\"kind\":\"runtime_event\""));
        assert!(jsonl.contains("\"kind\":\"stop_requested\""));
        assert_eq!(jsonl.matches("\"kind\":\"stop_requested\"").count(), 1);
    }

    #[tokio::test]
    async fn run_suite_records_case_runtime_failures_and_continues() {
        let dir = tempfile::tempdir().unwrap();
        let case_dir = dir.path().join("eval-cases/modules/query-vector");
        std::fs::create_dir_all(&case_dir).unwrap();
        for id in ["runtime-failure-one", "runtime-failure-two"] {
            std::fs::write(
                case_dir.join(format!("{id}.eure")),
                format!(
                    r#"
id = "{id}"
prompt = "Who are you?"

limits {{
  max-llm-calls = 1
}}
"#
                ),
            )
            .unwrap();
        }

        let output_root = dir.path().join("out");
        let config = RunnerConfig {
            cases_root: dir.path().join("eval-cases"),
            output_root: output_root.clone(),
            run_id: "runtime-failures".to_string(),
            judge_backend: test_backend_config_with_model("judge-model"),
            cheap_backend: test_backend_config_with_model("cheap-model"),
            default_backend: test_backend_config_with_model("default-model"),
            premium_backend: test_backend_config_with_model("premium-model"),
            model_dir: dir.path().join("missing-model"),
            fail_fast: false,
            max_concurrent_llm_calls: NonZeroUsize::new(7),
            case_patterns: Vec::new(),
            disabled_modules: Vec::new(),
        };

        let report = run_suite(&config).await.unwrap();

        let run_dir = output_root.join("runtime-failures");
        assert_eq!(report.run.run_id, "runtime-failures");
        assert_eq!(
            report.run.cases_root,
            config.cases_root.display().to_string()
        );
        assert_eq!(report.run.output_dir, run_dir.display().to_string());
        assert_eq!(report.run.case_patterns, Vec::<String>::new());
        assert!(!report.run.fail_fast);
        assert_eq!(report.run.max_concurrent_llm_calls, Some(7));
        assert_eq!(report.run.planned_case_count, 2);
        assert_eq!(report.run.models.judge, "judge-model");
        assert_eq!(report.run.models.cheap, "cheap-model");
        assert_eq!(report.run.models.default, "default-model");
        assert_eq!(report.run.models.premium, "premium-model");
        assert_eq!(report.case_count, 2);
        assert_eq!(report.passed_cases, 0);
        assert_eq!(report.invalid_cases, 2);
        assert!(report.cases.iter().all(|case| {
            !case.passed
                && case.invalid
                && case.score == 0.0
                && case.report.runtime_failure.is_some()
                && case.report.checks.is_empty()
        }));

        assert!(run_dir.join("suite-report.json").exists());
        let suite_json: serde_json::Value =
            serde_json::from_slice(&std::fs::read(run_dir.join("suite-report.json")).unwrap())
                .unwrap();
        assert_eq!(
            suite_json["run"],
            serde_json::json!({
                "run_id": "runtime-failures",
                "cases_root": config.cases_root.display().to_string(),
                "output_dir": run_dir.display().to_string(),
                "case_patterns": [],
                "fail_fast": false,
                "max_concurrent_llm_calls": 7,
                "planned_case_count": 2,
                "models": {
                    "judge": "judge-model",
                    "cheap": "cheap-model",
                    "default": "default-model",
                    "premium": "premium-model",
                },
                "disabled_modules": [],
            })
        );
        for summary in &report.cases {
            let output_dir = run_dir.join(sanitize_id(&summary.id));
            assert!(output_dir.join("report.json").exists());
            assert!(output_dir.join("artifact.json").exists());
            assert!(output_dir.join("raw-trace.json").exists());
            let events: serde_json::Value =
                serde_json::from_slice(&std::fs::read(output_dir.join("events.json")).unwrap())
                    .unwrap();
            assert_eq!(events, serde_json::json!([]));
        }
    }

    #[tokio::test]
    async fn agent_observation_serializes_string_keyed_blackboard_maps() {
        let now = Utc.with_ymd_and_hms(2026, 5, 7, 0, 0, 0).unwrap();
        let mut allocation = ResourceAllocation::default();
        allocation.set_model_override(builtin::query_vector(), ModelTier::Default);
        allocation.set(
            builtin::query_vector(),
            ModuleConfig {
                guidance: "test guidance".into(),
            },
        );
        allocation.set_activation(builtin::query_vector(), ActivationRatio::ONE);
        let blackboard = Blackboard::with_allocation(allocation.clone());
        let owner = ModuleInstanceId::new(builtin::query_vector(), ReplicaIndex::ZERO);

        blackboard
            .apply(BlackboardCommand::UpdateMemo {
                owner: owner.clone(),
                memo: "memo".to_string(),
                written_at: now,
            })
            .await;
        blackboard
            .apply(BlackboardCommand::AppendCognitionLog {
                source: owner,
                entry: CognitionLogEntry {
                    at: now,
                    text: "cognition".to_string(),
                },
            })
            .await;
        blackboard
            .apply(BlackboardCommand::SetModulePolicies {
                policies: vec![(
                    builtin::query_vector(),
                    nuillu_blackboard::ModulePolicy::new(
                        ReplicaCapRange::new(0, 0).unwrap(),
                        Bpm::from_f64(60.0)..=Bpm::from_f64(60.0),
                        linear_ratio_fn,
                    ),
                )],
            })
            .await;
        blackboard
            .apply(BlackboardCommand::RecordAllocationProposal {
                controller: ModuleInstanceId::new(
                    builtin::attention_controller(),
                    ReplicaIndex::ZERO,
                ),
                proposal: allocation,
            })
            .await;
        blackboard
            .apply(BlackboardCommand::UpsertMemoryMetadata {
                index: MemoryIndex::new("mem-1"),
                rank_if_new: MemoryRank::Permanent,
                occurred_at_if_new: None,
                decay_if_new_secs: 0,
                now,
                patch: MemoryMetaPatch::default(),
            })
            .await;

        let observation = blackboard
            .read(|bb| AgentObservation::from_blackboard(bb, Vec::new()))
            .await;
        let actual = serde_json::to_value(observation).unwrap();
        let expected = serde_json::json!({
            "memo_logs": {
                "query-vector": [{
                    "replica": 0,
                    "index": 0,
                    "written_at": "2026-05-07T00:00:00+00:00",
                    "content": "memo",
                }],
            },
            "cognition_logs": [{
                "source": {
                    "module": "query-vector",
                    "replica": 0,
                },
                "entries": [{
                    "at": "2026-05-07T00:00:00Z",
                    "text": "cognition",
                }],
            }],
            "allocation": {
                "query-vector": {
                    "activation_ratio": 1.0,
                    "guidance": "test guidance",
                    "tier": "Default",
                },
            },
            "allocation_proposals": {
                "attention-controller": {
                    "query-vector": {
                        "activation_ratio": 1.0,
                        "guidance": "test guidance",
                        "tier": "Default",
                    },
                },
            },
            "replica_caps": {
                "query-vector": {
                    "min": 0,
                    "max": 0,
                },
            },
            "memory_metadata": {
                "mem-1": {
                    "index": "mem-1",
                    "rank": "Permanent",
                    "occurred_at": null,
                    "decay_remaining_secs": 0,
                    "remember_tokens": 0,
                    "last_accessed": "2026-05-07T00:00:00Z",
                    "access_count": 0,
                    "query_history": [],
                },
            },
            "utterances": [],
        });
        assert_eq!(actual, expected);
    }

    #[tokio::test]
    async fn seed_cognition_log_stamps_cognition_gate_replica_and_offsets_time() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("cognition-seed.eure");
        std::fs::write(
            &path,
            r#"
id = "cognition-seed"
prompt = "What am I attending to?"

@ cognition-log[] {
  text = "Older cognition-log topic"
  seconds-ago = 30
}

@ cognition-log[] {
  text = "Current cognition-log topic"
}
"#,
        )
        .unwrap();
        let case = crate::cases::parse_module_case_file(&path).unwrap();
        let blackboard = Blackboard::default();
        let now = Utc.with_ymd_and_hms(2026, 5, 7, 12, 0, 0).unwrap();

        seed_cognition_log(&blackboard, &FixedClock(now), &case.cognition_log).await;

        let log_set = blackboard.read(|bb| bb.cognition_log_set()).await;
        let records = log_set.logs();
        assert_eq!(records.len(), 1);
        let record = &records[0];
        assert_eq!(record.source.module, builtin::cognition_gate());
        assert_eq!(record.source.replica, ReplicaIndex::ZERO);
        assert_eq!(record.entries.len(), 2);
        assert_eq!(record.entries[0].text, "Older cognition-log topic");
        assert_eq!(record.entries[0].at, now - ChronoDuration::seconds(30));
        assert_eq!(record.entries[1].text, "Current cognition-log topic");
        assert_eq!(record.entries[1].at, now);
    }

    #[tokio::test]
    async fn seed_memos_stamps_requested_module_replica_and_offsets_time() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("memo-seed.eure");
        std::fs::write(
            &path,
            r#"
id = "memo-seed"
prompt = "What am I attending to?"

@ memos[] {
  module = "sensory"
  replica = 1
  content = "Koro gave a boundary signal"
  seconds-ago = 12
}
"#,
        )
        .unwrap();
        let case = crate::cases::parse_module_case_file(&path).unwrap();
        let blackboard = Blackboard::default();
        let now = Utc.with_ymd_and_hms(2026, 5, 7, 12, 0, 0).unwrap();

        let records = seed_memos(&blackboard, &FixedClock(now), &case.memos)
            .await
            .unwrap();

        assert_eq!(records.len(), 1);
        assert_eq!(records[0].owner.module, builtin::sensory());
        assert_eq!(records[0].owner.replica, ReplicaIndex::new(1));
        assert_eq!(records[0].content, "Koro gave a boundary signal");
        assert_eq!(records[0].written_at, now - ChronoDuration::seconds(12));

        let memo_logs = blackboard.read(|bb| bb.recent_memo_logs()).await;
        assert_eq!(memo_logs, records);
    }

    #[tokio::test]
    async fn attention_schema_appends_attention_experience_and_skips_no_tool_wakes() {
        let local = LocalSet::new();
        local
            .run_until(async {
                let blackboard = Blackboard::default();
                let first_entry = "I notice Koro's food-boundary signal.";
                let second_entry = "I feel my attention settle on Koro relaxing.";
                let adapter = MockLlmAdapter::new()
                    .with_text_scenario(attention_schema_tool_scenario(
                        "append-attention-1",
                        first_entry,
                    ))
                    .with_text_scenario(attention_schema_no_tool_scenario())
                    .with_text_scenario(attention_schema_tool_scenario(
                        "append-attention-2",
                        second_entry,
                    ))
                    .with_text_scenario(attention_schema_no_tool_scenario())
                    .with_text_scenario(attention_schema_no_tool_scenario())
                    .with_text_scenario(attention_schema_no_tool_scenario());
                let caps = test_caps_with_adapter(blackboard.clone(), adapter);
                let modules = ModuleRegistry::new()
                    .register(
                        eval_policy(1..=1, Bpm::from_f64(60_000.0)..=Bpm::from_f64(60_000.0)),
                        |caps| {
                            nuillu_attention_schema::AttentionSchemaModule::new(
                                caps.memo_updated_inbox(),
                                caps.allocation_updated_inbox(),
                                caps.cognition_log_updated_inbox(),
                                caps.blackboard_reader(),
                                caps.allocation_reader(),
                                caps.cognition_log_reader(),
                                caps.cognition_writer(),
                                caps.llm_access(),
                            )
                        },
                    )
                    .unwrap()
                    .build(&caps)
                    .await
                    .unwrap();
                let memo_mailbox = caps.internal_harness_io().memo_updated_mailbox();
                let cognition_mailbox = caps.internal_harness_io().cognition_log_updated_mailbox();
                let sensory = ModuleInstanceId::new(builtin::sensory(), ReplicaIndex::ZERO);
                let cognition_source =
                    ModuleInstanceId::new(builtin::cognition_gate(), ReplicaIndex::ZERO);
                let now = Utc.with_ymd_and_hms(2026, 5, 7, 12, 0, 0).unwrap();
                let run_blackboard = blackboard.clone();

                run_agent(
                    modules,
                    AgentEventLoopConfig {
                        idle_threshold: Duration::from_millis(50),
                        activate_retries: 1,
                    },
                    async {
                        let record = run_blackboard
                            .update_memo(
                                sensory.clone(),
                                "Koro gave a food-boundary signal".to_owned(),
                                now,
                            )
                            .await;
                        memo_mailbox
                            .publish(MemoUpdated {
                                owner: sensory.clone(),
                                index: record.index,
                            })
                            .await
                            .expect("attention-schema receives seeded memo update");

                        for _ in 0..50 {
                            let count = attention_schema_cognition_output(&run_blackboard)
                                .await
                                .lines()
                                .count();
                            if count >= 1 {
                                break;
                            }
                            tokio::task::yield_now().await;
                            tokio::time::sleep(Duration::from_millis(1)).await;
                        }

                        run_blackboard
                            .apply(BlackboardCommand::AppendCognitionLog {
                                source: cognition_source.clone(),
                                entry: CognitionLogEntry {
                                    at: now,
                                    text: "Koro food-boundary signal is cognitively relevant"
                                        .to_owned(),
                                },
                            })
                            .await;
                        cognition_mailbox
                            .publish(CognitionLogUpdated::EntryAppended {
                                source: cognition_source.clone(),
                            })
                            .await
                            .expect("attention-schema receives cognition-log update");

                        for _ in 0..50 {
                            tokio::task::yield_now().await;
                            tokio::time::sleep(Duration::from_millis(1)).await;
                        }

                        let record = run_blackboard
                            .update_memo(
                                sensory.clone(),
                                "Koro relaxed after the boundary was respected".to_owned(),
                                now + ChronoDuration::seconds(1),
                            )
                            .await;
                        memo_mailbox
                            .publish(MemoUpdated {
                                owner: sensory,
                                index: record.index,
                            })
                            .await
                            .expect("attention-schema receives second memo update");

                        for _ in 0..80 {
                            let count = attention_schema_cognition_output(&run_blackboard)
                                .await
                                .lines()
                                .count();
                            if count >= 2 {
                                break;
                            }
                            tokio::task::yield_now().await;
                            tokio::time::sleep(Duration::from_millis(1)).await;
                        }
                    },
                )
                .await
                .unwrap();

                assert_eq!(
                    attention_schema_cognition_output(&blackboard).await,
                    [first_entry, second_entry].join("\n\n")
                );
            })
            .await;
    }

    #[test]
    fn full_agent_case_modules_filters_disabled() {
        let case = FullAgentCase {
            id: Some("x".to_string()),
            description: None,
            now: None,
            modules: Some(DEFAULT_FULL_AGENT_MODULES.to_vec()),
            inputs: Vec::new(),
            steps: Vec::new(),
            participants: Vec::new(),
            memories: Vec::new(),
            memos: Vec::new(),
            limits: crate::cases::EvalLimits {
                max_llm_calls: None,
            },
            checks: Vec::new(),
            modules_checks: Vec::new(),
            scoring: Default::default(),
        };
        let modules = full_agent_case_modules(&case, &[EvalModule::SpeakGate]);
        assert!(!modules.contains(&EvalModule::SpeakGate));
        assert!(modules.contains(&EvalModule::Speak));
        assert!(modules.contains(&EvalModule::AttentionController));
    }

    #[test]
    fn validate_disabled_modules_rejects_required_modules() {
        for required in REQUIRED_FULL_AGENT_MODULES {
            let err = validate_disabled_modules(std::slice::from_ref(required)).unwrap_err();
            assert!(matches!(
                err,
                RunnerError::DisableRequiredModule { module } if module == required.as_str()
            ));
        }
        assert!(validate_disabled_modules(&[EvalModule::SpeakGate]).is_ok());
    }

    #[tokio::test]
    async fn eval_registry_and_allocation_include_only_selected_modules() {
        let selected = [
            EvalModule::Sensory,
            EvalModule::AttentionController,
            EvalModule::Speak,
        ];
        let allocation = full_agent_allocation(
            &crate::cases::EvalLimits {
                max_llm_calls: None,
            },
            &selected,
        );
        assert!(allocation.get(&builtin::query_vector()).is_none());
        assert!(allocation.get(&builtin::speak_gate()).is_none());

        let blackboard = Blackboard::with_allocation(allocation);
        let caps = test_caps(blackboard.clone());
        let allocated = eval_registry(&selected).build(&caps).await.unwrap();
        assert_eq!(allocated.len(), selected.len());

        let (replica_caps, allocation_modules) = blackboard
            .read(|bb| {
                let mut replica_caps = bb
                    .module_policies()
                    .keys()
                    .map(|module| module.as_str().to_owned())
                    .collect::<Vec<_>>();
                replica_caps.sort();
                let mut allocation_modules = bb
                    .allocation()
                    .iter()
                    .map(|(module, _)| module.as_str().to_owned())
                    .collect::<Vec<_>>();
                allocation_modules.sort();
                (replica_caps, allocation_modules)
            })
            .await;

        assert_eq!(
            replica_caps,
            vec!["attention-controller", "sensory", "speak"]
        );
        assert_eq!(
            allocation_modules,
            vec!["attention-controller", "sensory", "speak"]
        );
    }

    #[test]
    fn full_agent_allocation_bootstraps_input_and_controller_only() {
        let selected = DEFAULT_FULL_AGENT_MODULES.to_vec();
        let allocation = full_agent_allocation(
            &crate::cases::EvalLimits {
                max_llm_calls: None,
            },
            &selected,
        );

        assert_eq!(
            allocation.activation_for(&builtin::sensory()),
            ActivationRatio::ONE
        );
        assert_eq!(allocation.tier_for(&builtin::sensory()), ModelTier::Cheap);

        assert_eq!(
            allocation.activation_for(&builtin::cognition_gate()),
            ActivationRatio::ZERO
        );
        assert_eq!(
            allocation.tier_for(&builtin::cognition_gate()),
            ModelTier::Cheap
        );

        assert_eq!(
            allocation.activation_for(&builtin::attention_controller()),
            ActivationRatio::ONE
        );
        assert_eq!(
            allocation.tier_for(&builtin::attention_controller()),
            ModelTier::Default
        );

        assert_eq!(
            allocation.activation_for(&builtin::speak_gate()),
            ActivationRatio::ZERO
        );
        assert_eq!(
            allocation.tier_for(&builtin::speak_gate()),
            ModelTier::Premium
        );

        assert_eq!(
            allocation.activation_for(&builtin::speak()),
            ActivationRatio::ZERO
        );
        assert_eq!(allocation.tier_for(&builtin::speak()), ModelTier::Premium);

        for module in [
            builtin::attention_schema(),
            builtin::self_model(),
            builtin::query_vector(),
            builtin::memory(),
            builtin::memory_compaction(),
            builtin::predict(),
            builtin::surprise(),
        ] {
            assert_eq!(allocation.activation_for(&module), ActivationRatio::ZERO);
        }
        assert!(allocation.get(&builtin::query_agentic()).is_none());
    }

    #[tokio::test]
    async fn full_agent_gui_initial_allocation_starts_every_module_detached() {
        let selected = DEFAULT_FULL_AGENT_MODULES.to_vec();
        let allocation = full_agent_gui_initial_allocation(
            &crate::cases::EvalLimits {
                max_llm_calls: None,
            },
            &selected,
        );
        let blackboard = Blackboard::with_allocation(allocation);
        let caps = test_caps(blackboard.clone());
        let _allocated = eval_registry(&selected).build(&caps).await.unwrap();

        blackboard
            .read(|bb| {
                for module in &selected {
                    let id = module.module_id();
                    assert_eq!(bb.allocation().activation_for(&id), ActivationRatio::ZERO);
                    assert_eq!(bb.allocation().active_replicas(&id), 0);
                }
            })
            .await;
    }
}
