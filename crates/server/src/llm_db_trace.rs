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
use nuillu_types::ModuleInstanceId;
use nuillu_visualizer_protocol::{
    LlmInputItemView, LlmObservationSource, LlmOutputItemView, LlmTranscriptTurnStatus,
    LlmTranscriptTurnView, LlmUsageView, VisualizerEvent, VisualizerTabId,
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
    activation_attempt_turns: Mutex<BTreeMap<(String, u32), String>>,
    terminal_turns: Mutex<BTreeMap<String, CompletedTurnTrace>>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CompletedTurnTrace {
    version: u32,
    #[serde(default = "default_turn_status")]
    status: TranscriptTurnStatus,
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
    #[serde(default)]
    error_message: Option<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum TranscriptTurnStatus {
    InProgress,
    Completed,
    Failed,
}

fn default_turn_status() -> TranscriptTurnStatus {
    TranscriptTurnStatus::Completed
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
                activation_attempt_turns: Mutex::new(BTreeMap::new()),
                terminal_turns: Mutex::new(BTreeMap::new()),
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
                status: TranscriptTurnStatus::InProgress,
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
                error_message: None,
            }
        });
        if let Some(activation_attempt) = metadata.activation_attempt {
            self.inner
                .activation_attempt_turns
                .lock()
                .expect("DB LLM trace activation turn map lock poisoned")
                .insert(
                    (metadata.owner.to_string(), activation_attempt),
                    trace.turn_id.clone(),
                );
        }
        update(trace);
        trace.clone()
    }

    async fn complete_trace(&self, extensions: &RequestExtensions, trace: CompletedTurnTrace) {
        self.inner
            .turns
            .lock()
            .expect("DB LLM trace turn map lock poisoned")
            .remove(&extension_key(extensions));
        self.persist_terminal_trace(trace).await;
    }

    async fn persist_terminal_trace(&self, trace: CompletedTurnTrace) {
        self.inner
            .terminal_turns
            .lock()
            .expect("DB LLM trace terminal turn map lock poisoned")
            .insert(trace.turn_id.clone(), trace.clone());
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

    pub async fn mark_activation_attempt_failed(
        &self,
        owner: ModuleInstanceId,
        activation_attempt: u32,
        message: String,
    ) {
        let turn_id = self
            .inner
            .activation_attempt_turns
            .lock()
            .expect("DB LLM trace activation turn map lock poisoned")
            .remove(&(owner.to_string(), activation_attempt));
        let Some(turn_id) = turn_id else {
            return;
        };

        let active_trace = {
            let mut turns = self
                .inner
                .turns
                .lock()
                .expect("DB LLM trace turn map lock poisoned");
            let key = turns
                .iter()
                .find_map(|(key, trace)| (trace.turn_id == turn_id).then_some(*key));
            key.and_then(|key| turns.remove(&key))
        };
        let mut trace = active_trace.or_else(|| {
            self.inner
                .terminal_turns
                .lock()
                .expect("DB LLM trace terminal turn map lock poisoned")
                .get(&turn_id)
                .cloned()
        });
        let Some(mut trace) = trace.take() else {
            return;
        };

        mark_trace_failed(&mut trace, message);
        self.persist_terminal_trace(trace).await;
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
    let mut turns = Vec::new();
    for record in records {
        let Ok(trace) = serde_json::from_value::<CompletedTurnTrace>(record.trace_json) else {
            tracing::warn!(id = record.id, "failed to parse persisted LLM transcript");
            continue;
        };
        turns.push(trace.to_transcript_turn_view(format!("server-persisted-{}", record.id)));
    }
    if !turns.is_empty() {
        visualizer.send(VisualizerEvent::LlmTranscriptSnapshot { tab_id, turns });
    }
}

impl CompletedTurnTrace {
    fn to_transcript_turn_view(&self, restored_turn_id: String) -> LlmTranscriptTurnView {
        LlmTranscriptTurnView {
            turn_id: restored_turn_id,
            owner: self.owner.clone(),
            module: self.module.clone(),
            replica: self.replica,
            tier: self.tier.clone(),
            source: self.source,
            session_key: self.session_key.clone(),
            operation: self.operation.clone(),
            input: self.input.clone(),
            output: transcript_output_items(self),
            request_id: self.request_id.clone(),
            model: self.model.clone(),
            finish_reason: self.finish_reason.clone(),
            usage: self.usage,
            status: match self.status {
                TranscriptTurnStatus::Failed => LlmTranscriptTurnStatus::Failed,
                TranscriptTurnStatus::InProgress | TranscriptTurnStatus::Completed => {
                    LlmTranscriptTurnStatus::Completed
                }
            },
            error_message: self.error_message.clone(),
        }
    }
}

fn transcript_output_items(trace: &CompletedTurnTrace) -> Vec<LlmOutputItemView> {
    let mut output = Vec::new();
    for delta in &trace.deltas {
        append_transcript_output_delta(&mut output, delta.kind.clone(), delta.delta.clone(), None);
    }
    for tool in &trace.tool_calls {
        let source = transcript_tool_call_source(&tool.name, &tool.id);
        if let Some(arguments_json) = &tool.arguments_json {
            apply_transcript_tool_call_ready(&mut output, source, arguments_json.clone());
        } else {
            append_transcript_output_delta(
                &mut output,
                "tool_call".to_string(),
                tool.arguments_json_delta.clone(),
                Some(source),
            );
        }
    }
    if let Some(json) = &trace.structured_output {
        apply_transcript_structured_ready(&mut output, json.clone());
    }
    output
}

fn append_transcript_output_delta(
    output: &mut Vec<LlmOutputItemView>,
    kind: String,
    delta: String,
    source: Option<String>,
) {
    if let Some(index) = output
        .iter()
        .rposition(|row| row.kind == kind && row.source == source)
    {
        let mut row = output.remove(index);
        row.content.push_str(&delta);
        output.push(row);
        return;
    }
    output.push(LlmOutputItemView {
        kind,
        content: delta,
        source,
    });
}

fn apply_transcript_tool_call_ready(
    output: &mut Vec<LlmOutputItemView>,
    source: String,
    arguments_json: String,
) {
    if let Some(index) = output
        .iter()
        .rposition(|row| row.kind == "tool_call" && row.source.as_deref() == Some(source.as_str()))
    {
        let mut row = output.remove(index);
        row.kind = "tool_call_ready".to_string();
        row.content = arguments_json;
        row.source = Some(source);
        output.push(row);
        return;
    }
    output.push(LlmOutputItemView {
        kind: "tool_call_ready".to_string(),
        content: arguments_json,
        source: Some(source),
    });
}

fn apply_transcript_structured_ready(output: &mut Vec<LlmOutputItemView>, json: String) {
    if let Some(row) = output
        .iter_mut()
        .rev()
        .find(|row| row.kind == "structured_ready")
    {
        row.content = json;
        row.source = None;
        return;
    }
    if let Some(index) = output.iter().rposition(|row| row.kind == "structured") {
        let mut row = output.remove(index);
        row.kind = "structured_ready".to_string();
        row.content = json;
        row.source = None;
        output.push(row);
        return;
    }
    output.push(LlmOutputItemView {
        kind: "structured_ready".to_string(),
        content: json,
        source: None,
    });
}

fn transcript_tool_call_source(name: &str, id: &str) -> String {
    format!("{name}({id})")
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
    trace.status = TranscriptTurnStatus::Completed;
    trace.request_id = request_id.clone().or_else(|| trace.request_id.clone());
    trace.completed_at_ms = Some(Utc::now().timestamp_millis());
    trace.finish_reason = Some(format!("{finish_reason:?}"));
    trace.usage = Some(usage_view(usage));
}

fn mark_trace_failed(trace: &mut CompletedTurnTrace, message: String) {
    trace.status = TranscriptTurnStatus::Failed;
    trace.completed_at_ms = Some(Utc::now().timestamp_millis());
    trace.error_message = Some(message);
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

#[cfg(test)]
mod tests {
    use super::*;

    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::mpsc;

    use lutum_libsql_adapter::{LibsqlAgentStore, LibsqlAgentStoreConfig};
    use nuillu_module::ports::{Embedder, PortError};
    use nuillu_types::{ReplicaIndex, builtin};
    use nuillu_visualizer_protocol::VisualizerServerMessage;

    static NEXT_TEST_DIR: AtomicU64 = AtomicU64::new(0);

    #[derive(Debug)]
    struct TestEmbedder {
        dims: usize,
    }

    impl TestEmbedder {
        fn boxed(dims: usize) -> Box<dyn Embedder> {
            Box::new(Self { dims })
        }
    }

    #[async_trait::async_trait(?Send)]
    impl Embedder for TestEmbedder {
        fn dimensions(&self) -> usize {
            self.dims
        }

        async fn embed(&self, _text: &str) -> Result<Vec<f32>, PortError> {
            Ok(vec![0.0; self.dims])
        }
    }

    #[tokio::test]
    async fn failed_active_turn_is_inserted_into_transcript_db() {
        let store = test_transcript_store().await;
        let sink = DbLlmTraceSink::new("server-session".to_string(), store.clone());
        let owner = ModuleInstanceId::new(builtin::predict(), ReplicaIndex::ZERO);
        let trace = test_trace(&owner, "turn-active", TranscriptTurnStatus::InProgress);

        sink.inner
            .turns
            .lock()
            .expect("turn map lock poisoned")
            .insert(1, trace);
        sink.inner
            .activation_attempt_turns
            .lock()
            .expect("activation turn map lock poisoned")
            .insert((owner.to_string(), 5), "turn-active".to_string());

        sink.mark_activation_attempt_failed(owner, 5, "activation failed".to_string())
            .await;

        let recent = store.recent_completed_turns(10).await.unwrap();
        assert_eq!(recent.len(), 1);
        assert_eq!(recent[0].turn_id, "turn-active");
        assert_eq!(recent[0].trace_json["status"], "failed");
        assert_eq!(recent[0].trace_json["error_message"], "activation failed");
        assert!(recent[0].trace_json["completed_at_ms"].is_i64());
    }

    #[tokio::test]
    async fn completed_turn_can_be_updated_and_replayed_as_failed() {
        let store = test_transcript_store().await;
        let sink = DbLlmTraceSink::new("server-session".to_string(), store.clone());
        let owner = ModuleInstanceId::new(builtin::predict(), ReplicaIndex::ZERO);
        let trace = test_trace(&owner, "turn-completed", TranscriptTurnStatus::Completed);

        sink.persist_terminal_trace(trace).await;
        sink.inner
            .activation_attempt_turns
            .lock()
            .expect("activation turn map lock poisoned")
            .insert((owner.to_string(), 3), "turn-completed".to_string());

        sink.mark_activation_attempt_failed(owner, 3, "module failed after output".to_string())
            .await;

        let recent = store.recent_completed_turns(10).await.unwrap();
        assert_eq!(recent.len(), 1);
        let restored: CompletedTurnTrace =
            serde_json::from_value(recent[0].trace_json.clone()).unwrap();
        assert_eq!(restored.status, TranscriptTurnStatus::Failed);
        assert_eq!(
            restored.error_message.as_deref(),
            Some("module failed after output")
        );
        assert_eq!(restored.finish_reason.as_deref(), Some("Stop"));
        assert!(restored.usage.is_some());

        let direct_turn = restored.to_transcript_turn_view("restored".to_string());
        assert_eq!(direct_turn.turn_id, "restored");
        assert_eq!(direct_turn.status, LlmTranscriptTurnStatus::Failed);
        assert_eq!(
            direct_turn.error_message.as_deref(),
            Some("module failed after output")
        );
        assert_eq!(direct_turn.finish_reason.as_deref(), Some("Stop"));
        assert!(direct_turn.usage.is_some());

        let (tx, rx) = mpsc::channel();
        let visualizer = VisualizerEventSink::new(tx);
        emit_persisted_llm_transcripts(&store, "server", &visualizer).await;
        let snapshot = rx.try_iter().collect::<Vec<_>>();
        assert_eq!(snapshot.len(), 1);
        let VisualizerServerMessage::Event {
            event: VisualizerEvent::LlmTranscriptSnapshot { tab_id, turns },
        } = &snapshot[0]
        else {
            panic!("expected LLM transcript snapshot event");
        };
        assert_eq!(tab_id.as_str(), "server");
        assert_eq!(turns.len(), 1);
        assert_eq!(turns[0].status, LlmTranscriptTurnStatus::Failed);
        assert_eq!(
            turns[0].error_message.as_deref(),
            Some("module failed after output")
        );
    }

    async fn test_transcript_store() -> LibsqlLlmTranscriptStore {
        let agent = LibsqlAgentStore::connect(
            LibsqlAgentStoreConfig::local(test_db_path(), 3, 3),
            TestEmbedder::boxed(3),
            TestEmbedder::boxed(3),
        )
        .await
        .unwrap();
        agent.llm_transcript_store()
    }

    fn test_trace(
        owner: &ModuleInstanceId,
        turn_id: &str,
        status: TranscriptTurnStatus,
    ) -> CompletedTurnTrace {
        CompletedTurnTrace {
            version: 1,
            status,
            turn_id: turn_id.to_string(),
            owner: owner.to_string(),
            module: owner.module.to_string(),
            replica: owner.replica.get(),
            tier: "Default".to_string(),
            source: LlmObservationSource::ModuleTurn,
            session_key: Some("session".to_string()),
            operation: "text_turn".to_string(),
            started_at_ms: 10,
            completed_at_ms: (status != TranscriptTurnStatus::InProgress).then_some(20),
            input: vec![LlmInputItemView {
                role: "user".to_string(),
                kind: "text".to_string(),
                content: "hello".to_string(),
                ephemeral: false,
                source: None,
            }],
            request_id: Some("request".to_string()),
            model: Some("model".to_string()),
            deltas: vec![DeltaTrace {
                kind: "text".to_string(),
                delta: "partial".to_string(),
            }],
            tool_calls: Vec::new(),
            structured_output: None,
            finish_reason: (status == TranscriptTurnStatus::Completed)
                .then_some("Stop".to_string()),
            usage: (status == TranscriptTurnStatus::Completed).then_some(LlmUsageView {
                input_tokens: 1,
                output_tokens: 2,
                total_tokens: 3,
                cost_micros_usd: 0,
                cache_creation_tokens: 0,
                cache_read_tokens: 0,
            }),
            error_message: None,
        }
    }

    fn test_db_path() -> PathBuf {
        let dir = std::env::current_dir()
            .unwrap()
            .join(".tmp")
            .join("server-llm-db-trace-tests")
            .join(format!(
                "{}-{}",
                std::process::id(),
                NEXT_TEST_DIR.fetch_add(1, Ordering::Relaxed)
            ));
        std::fs::create_dir_all(&dir).unwrap();
        dir.join("agent.db")
    }
}
