use std::collections::BTreeMap;
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicU64, Ordering},
};

use chrono::Utc;
use lutum::{
    CompletionEvent, ErasedStructuredCompletionEvent, ErasedStructuredTurnEvent,
    ErasedTextTurnEvent, LutumHooksSet, LutumStreamEvent, ModelInputHookContext, OnModelInput,
    OnStreamEvent, OperationKind, RequestExtensions, StreamEventHookContext, Usage,
};
use lutum_libsql_adapter::{LibsqlLlmTranscriptStore, NewLlmTranscriptTurn};
use nuillu_module::{LlmRequestMetadata, ModuleSessionMetadata};
use nuillu_visualizer_protocol::{
    LlmInputItemView, LlmObservationEvent, LlmObservationSource, LlmUsageView, VisualizerEvent,
    VisualizerTabId,
};
use serde::{Deserialize, Serialize};

use crate::gui::VisualizerEventSink;
use crate::llm_observer::{
    model_input_views, observation_source, operation_kind_label, usage_view,
};

const LLM_TRANSCRIPT_RETAINED_TURNS: usize = 200;

#[derive(Clone)]
pub struct DbLlmTraceSink {
    inner: Arc<DbLlmTraceSinkInner>,
}

struct DbLlmTraceSinkInner {
    server_session_id: String,
    store: LibsqlLlmTranscriptStore,
    next_turn: AtomicU64,
    turns: Mutex<BTreeMap<usize, CompletedTurnTrace>>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CompletedTurnTrace {
    version: u32,
    turn_id: String,
    owner: String,
    module: String,
    replica: u8,
    tier: String,
    source: LlmObservationSource,
    #[serde(default)]
    session_key: Option<String>,
    operation: String,
    started_at_ms: i64,
    completed_at_ms: Option<i64>,
    input: Vec<LlmInputItemView>,
    request_id: Option<String>,
    model: Option<String>,
    deltas: Vec<DeltaTrace>,
    tool_calls: Vec<ToolCallTrace>,
    structured_output: Option<String>,
    finish_reason: Option<String>,
    usage: Option<LlmUsageView>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct DeltaTrace {
    kind: String,
    delta: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct ToolCallTrace {
    id: String,
    name: String,
    arguments_json_delta: String,
    arguments_json: Option<String>,
}

impl DbLlmTraceSink {
    pub fn new(server_session_id: String, store: LibsqlLlmTranscriptStore) -> Self {
        Self {
            inner: Arc::new(DbLlmTraceSinkInner {
                server_session_id,
                store,
                next_turn: AtomicU64::new(0),
                turns: Mutex::new(BTreeMap::new()),
            }),
        }
    }

    pub fn hook_set(&self) -> LutumHooksSet<'static> {
        LutumHooksSet::new()
            .with_on_model_input(self.clone())
            .with_on_stream_event(self.clone())
    }

    fn metadata(extensions: &RequestExtensions) -> Option<&LlmRequestMetadata> {
        extensions.get::<LlmRequestMetadata>()
    }

    fn update_trace(
        &self,
        extensions: &RequestExtensions,
        metadata: &LlmRequestMetadata,
        operation: OperationKind,
        update: impl FnOnce(&mut CompletedTurnTrace),
    ) -> CompletedTurnTrace {
        let key = extension_key(extensions);
        let mut turns = self
            .inner
            .turns
            .lock()
            .expect("DB LLM trace turn map lock poisoned");
        let trace = turns.entry(key).or_insert_with(|| {
            let next = self.inner.next_turn.fetch_add(1, Ordering::Relaxed);
            let turn_id = format!("{}-db-llm-{next}", self.inner.server_session_id);
            let now = Utc::now().timestamp_millis();
            CompletedTurnTrace {
                version: 1,
                turn_id,
                owner: metadata.owner.to_string(),
                module: metadata.owner.module.to_string(),
                replica: metadata.owner.replica.get(),
                tier: format!("{:?}", metadata.tier),
                source: observation_source(metadata.source),
                session_key: observation_session_key(extensions, metadata),
                operation: operation_kind_label(operation).to_string(),
                started_at_ms: now,
                completed_at_ms: None,
                input: Vec::new(),
                request_id: None,
                model: None,
                deltas: Vec::new(),
                tool_calls: Vec::new(),
                structured_output: None,
                finish_reason: None,
                usage: None,
            }
        });
        update(trace);
        trace.clone()
    }

    async fn complete_trace(&self, extensions: &RequestExtensions, trace: CompletedTurnTrace) {
        self.inner
            .turns
            .lock()
            .expect("DB LLM trace turn map lock poisoned")
            .remove(&extension_key(extensions));
        let trace_json = match serde_json::to_value(&trace) {
            Ok(value) => value,
            Err(error) => {
                tracing::warn!(?error, "failed to serialize completed LLM trace");
                return;
            }
        };
        if let Err(error) = self
            .inner
            .store
            .insert_completed_turn(NewLlmTranscriptTurn {
                server_session_id: self.inner.server_session_id.clone(),
                turn_id: trace.turn_id.clone(),
                owner: trace.owner.clone(),
                owner_module: trace.module.clone(),
                owner_replica: trace.replica,
                tier: trace.tier.clone(),
                source: source_label(trace.source).to_string(),
                session_key: trace.session_key.clone(),
                operation: trace.operation.clone(),
                started_at_ms: trace.started_at_ms,
                completed_at_ms: trace
                    .completed_at_ms
                    .unwrap_or_else(|| Utc::now().timestamp_millis()),
                trace_json,
            })
            .await
        {
            tracing::warn!(?error, "failed to persist completed LLM trace");
            return;
        }
        if let Err(error) = self
            .inner
            .store
            .prune_completed_turns(LLM_TRANSCRIPT_RETAINED_TURNS)
            .await
        {
            tracing::warn!(?error, "failed to prune completed LLM traces");
        }
    }
}

impl OnModelInput for DbLlmTraceSink {
    async fn call(&self, cx: &ModelInputHookContext<'_>) {
        let Some(metadata) = Self::metadata(cx.extensions()) else {
            return;
        };
        self.update_trace(cx.extensions(), metadata, cx.kind(), |trace| {
            trace.input = model_input_views(cx.input().items());
        });
    }
}

impl OnStreamEvent for DbLlmTraceSink {
    async fn call(&self, cx: &StreamEventHookContext<'_>) {
        let Some(metadata) = Self::metadata(cx.extensions()) else {
            return;
        };
        let mut completed = false;
        let trace = self.update_trace(cx.extensions(), metadata, cx.kind(), |trace| {
            completed = apply_event(trace, cx.event());
        });
        if completed {
            self.complete_trace(cx.extensions(), trace).await;
        }
    }
}

pub async fn emit_persisted_llm_transcripts(
    store: &LibsqlLlmTranscriptStore,
    tab_id: &str,
    visualizer: &VisualizerEventSink,
) {
    let records = match store
        .recent_completed_turns(LLM_TRANSCRIPT_RETAINED_TURNS)
        .await
    {
        Ok(records) => records,
        Err(error) => {
            tracing::warn!(?error, "failed to load persisted LLM transcripts");
            return;
        }
    };
    let tab_id = VisualizerTabId::new(tab_id.to_string());
    for record in records {
        let Ok(trace) = serde_json::from_value::<CompletedTurnTrace>(record.trace_json) else {
            tracing::warn!(id = record.id, "failed to parse persisted LLM transcript");
            continue;
        };
        for event in trace.to_observation_events(format!("server-persisted-{}", record.id)) {
            visualizer.send(VisualizerEvent::LlmObserved {
                tab_id: tab_id.clone(),
                event,
            });
        }
    }
}

impl CompletedTurnTrace {
    fn to_observation_events(&self, restored_turn_id: String) -> Vec<LlmObservationEvent> {
        let mut events = vec![
            LlmObservationEvent::ModelInput {
                turn_id: restored_turn_id.clone(),
                owner: self.owner.clone(),
                module: self.module.clone(),
                replica: self.replica,
                tier: self.tier.clone(),
                source: self.source,
                session_key: self.session_key.clone(),
                operation: self.operation.clone(),
                items: self.input.clone(),
            },
            LlmObservationEvent::StreamStarted {
                turn_id: restored_turn_id.clone(),
                owner: self.owner.clone(),
                module: self.module.clone(),
                replica: self.replica,
                tier: self.tier.clone(),
                source: self.source,
                session_key: self.session_key.clone(),
                operation: self.operation.clone(),
                request_id: self.request_id.clone(),
                model: self.model.clone().unwrap_or_default(),
            },
        ];
        for delta in &self.deltas {
            events.push(LlmObservationEvent::StreamDelta {
                turn_id: restored_turn_id.clone(),
                kind: delta.kind.clone(),
                delta: delta.delta.clone(),
            });
        }
        for tool in &self.tool_calls {
            if let Some(arguments_json) = &tool.arguments_json {
                events.push(LlmObservationEvent::ToolCallReady {
                    turn_id: restored_turn_id.clone(),
                    id: tool.id.clone(),
                    name: tool.name.clone(),
                    arguments_json: arguments_json.clone(),
                });
            } else {
                events.push(LlmObservationEvent::ToolCallChunk {
                    turn_id: restored_turn_id.clone(),
                    id: tool.id.clone(),
                    name: tool.name.clone(),
                    arguments_json_delta: tool.arguments_json_delta.clone(),
                });
            }
        }
        if let Some(json) = &self.structured_output {
            events.push(LlmObservationEvent::StructuredReady {
                turn_id: restored_turn_id.clone(),
                json: json.clone(),
            });
        }
        if let (Some(finish_reason), Some(usage)) = (&self.finish_reason, self.usage) {
            events.push(LlmObservationEvent::Completed {
                turn_id: restored_turn_id,
                request_id: self.request_id.clone(),
                finish_reason: finish_reason.clone(),
                usage,
            });
        }
        events
    }
}

fn extension_key(extensions: &RequestExtensions) -> usize {
    std::ptr::from_ref(extensions) as usize
}

fn observation_session_key(
    extensions: &RequestExtensions,
    metadata: &LlmRequestMetadata,
) -> Option<String> {
    metadata.session_key.clone().or_else(|| {
        extensions
            .get::<ModuleSessionMetadata>()
            .map(|session| session.session_key.as_str().to_owned())
    })
}

fn apply_event(trace: &mut CompletedTurnTrace, event: LutumStreamEvent<'_>) -> bool {
    match event {
        LutumStreamEvent::TextTurn(event) => apply_text_turn_event(trace, event),
        LutumStreamEvent::StructuredTurn(event) => apply_structured_turn_event(trace, event),
        LutumStreamEvent::Completion(event) => apply_completion_event(trace, event),
        LutumStreamEvent::StructuredCompletion(event) => {
            apply_structured_completion_event(trace, event)
        }
    }
}

fn apply_started(trace: &mut CompletedTurnTrace, request_id: &Option<String>, model: &str) {
    trace.request_id = request_id.clone();
    trace.model = Some(model.to_string());
}

fn apply_completed(
    trace: &mut CompletedTurnTrace,
    request_id: &Option<String>,
    finish_reason: &impl std::fmt::Debug,
    usage: Usage,
) {
    trace.request_id = request_id.clone().or_else(|| trace.request_id.clone());
    trace.completed_at_ms = Some(Utc::now().timestamp_millis());
    trace.finish_reason = Some(format!("{finish_reason:?}"));
    trace.usage = Some(usage_view(usage));
}

fn push_delta(trace: &mut CompletedTurnTrace, kind: &str, delta: &str) {
    trace.deltas.push(DeltaTrace {
        kind: kind.to_string(),
        delta: delta.to_string(),
    });
}

fn push_tool_chunk(
    trace: &mut CompletedTurnTrace,
    id: &lutum::ToolCallId,
    name: &lutum::ToolName,
    arguments_json_delta: &str,
) {
    trace.tool_calls.push(ToolCallTrace {
        id: id.as_str().to_string(),
        name: name.as_str().to_string(),
        arguments_json_delta: arguments_json_delta.to_string(),
        arguments_json: None,
    });
}

fn push_tool_ready(trace: &mut CompletedTurnTrace, metadata: &lutum::ToolMetadata) {
    trace.tool_calls.push(ToolCallTrace {
        id: metadata.id.as_str().to_string(),
        name: metadata.name.as_str().to_string(),
        arguments_json_delta: String::new(),
        arguments_json: Some(metadata.arguments.to_string()),
    });
}

fn apply_text_turn_event(trace: &mut CompletedTurnTrace, event: &ErasedTextTurnEvent) -> bool {
    match event {
        ErasedTextTurnEvent::Started { request_id, model } => {
            apply_started(trace, request_id, model)
        }
        ErasedTextTurnEvent::TextDelta { delta } => push_delta(trace, "text", delta),
        ErasedTextTurnEvent::ReasoningDelta { delta } => push_delta(trace, "reasoning", delta),
        ErasedTextTurnEvent::RefusalDelta { delta } => push_delta(trace, "refusal", delta),
        ErasedTextTurnEvent::ToolCallChunk {
            id,
            name,
            arguments_json_delta,
        } => push_tool_chunk(trace, id, name, arguments_json_delta),
        ErasedTextTurnEvent::ToolCallReady(tool) => push_tool_ready(trace, tool),
        ErasedTextTurnEvent::Completed {
            request_id,
            finish_reason,
            usage,
            ..
        } => {
            apply_completed(trace, request_id, finish_reason, *usage);
            return true;
        }
    }
    false
}

fn apply_structured_turn_event(
    trace: &mut CompletedTurnTrace,
    event: &ErasedStructuredTurnEvent,
) -> bool {
    match event {
        ErasedStructuredTurnEvent::Started { request_id, model } => {
            apply_started(trace, request_id, model)
        }
        ErasedStructuredTurnEvent::StructuredOutputChunk { json_delta } => {
            push_delta(trace, "structured", json_delta);
        }
        ErasedStructuredTurnEvent::StructuredOutputReady(json) => {
            trace.structured_output = Some(json.to_string());
        }
        ErasedStructuredTurnEvent::ReasoningDelta { delta } => {
            push_delta(trace, "reasoning", delta)
        }
        ErasedStructuredTurnEvent::RefusalDelta { delta } => push_delta(trace, "refusal", delta),
        ErasedStructuredTurnEvent::ToolCallChunk {
            id,
            name,
            arguments_json_delta,
        } => push_tool_chunk(trace, id, name, arguments_json_delta),
        ErasedStructuredTurnEvent::ToolCallReady(tool) => push_tool_ready(trace, tool),
        ErasedStructuredTurnEvent::Completed {
            request_id,
            finish_reason,
            usage,
            ..
        } => {
            apply_completed(trace, request_id, finish_reason, *usage);
            return true;
        }
    }
    false
}

fn apply_completion_event(trace: &mut CompletedTurnTrace, event: &CompletionEvent) -> bool {
    match event {
        CompletionEvent::Started { request_id, model } => apply_started(trace, request_id, model),
        CompletionEvent::WillRetry { .. } => {}
        CompletionEvent::TextDelta(delta) => push_delta(trace, "text", delta),
        CompletionEvent::Completed {
            request_id,
            finish_reason,
            usage,
        } => {
            apply_completed(trace, request_id, finish_reason, *usage);
            return true;
        }
    }
    false
}

fn apply_structured_completion_event(
    trace: &mut CompletedTurnTrace,
    event: &ErasedStructuredCompletionEvent,
) -> bool {
    match event {
        ErasedStructuredCompletionEvent::Started { request_id, model } => {
            apply_started(trace, request_id, model);
        }
        ErasedStructuredCompletionEvent::StructuredOutputChunk { json_delta } => {
            push_delta(trace, "structured", json_delta);
        }
        ErasedStructuredCompletionEvent::StructuredOutputReady(json) => {
            trace.structured_output = Some(json.to_string());
        }
        ErasedStructuredCompletionEvent::ReasoningDelta { delta } => {
            push_delta(trace, "reasoning", delta);
        }
        ErasedStructuredCompletionEvent::RefusalDelta { delta } => {
            push_delta(trace, "refusal", delta)
        }
        ErasedStructuredCompletionEvent::Completed {
            request_id,
            finish_reason,
            usage,
        } => {
            apply_completed(trace, request_id, finish_reason, *usage);
            return true;
        }
    }
    false
}

fn source_label(source: LlmObservationSource) -> &'static str {
    match source {
        LlmObservationSource::ModuleTurn => "module_turn",
        LlmObservationSource::SessionCompaction => "session_compaction",
    }
}
