use std::{
    collections::{BTreeMap, HashMap},
    fs::{self, OpenOptions},
    io::{self, Write as _},
    path::{Path, PathBuf},
    sync::{
        Arc, Mutex,
        atomic::{AtomicU64, Ordering},
    },
};

use chrono::{DateTime, Utc};
use lutum::{
    AssistantInputItem, CompletionEvent, ErasedStructuredCompletionEvent,
    ErasedStructuredTurnEvent, ErasedTextTurnEvent, InputMessageRole, ItemView, LutumHooksSet,
    LutumStreamEvent, MessageContent, ModelInputHookContext, ModelInputItem, OnModelInput,
    OnStreamEvent, OperationKind, RequestExtensions, StreamEventHookContext, TurnRole, TurnView,
    Usage,
};
use nuillu_module::{LlmRequestMetadata, LlmRequestSource};
use serde::Serialize;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LlmLogContext {
    pub root: PathBuf,
    pub namespace: Vec<String>,
}

impl LlmLogContext {
    pub fn new(root: impl Into<PathBuf>, namespace: impl IntoIterator<Item = String>) -> Self {
        Self {
            root: root.into(),
            namespace: namespace.into_iter().collect(),
        }
    }
}

#[derive(Clone)]
pub struct FileLlmTraceSink {
    inner: Arc<FileLlmTraceSinkInner>,
}

struct FileLlmTraceSinkInner {
    context: LlmLogContext,
    next_turn_by_module: Mutex<BTreeMap<String, u64>>,
    active_turns: Mutex<HashMap<usize, TurnState>>,
}

#[derive(Clone)]
struct TurnState {
    dir: PathBuf,
    trace: TurnTrace,
}

impl FileLlmTraceSink {
    pub fn new(context: LlmLogContext) -> Self {
        Self {
            inner: Arc::new(FileLlmTraceSinkInner {
                context,
                next_turn_by_module: Mutex::new(BTreeMap::new()),
                active_turns: Mutex::new(HashMap::new()),
            }),
        }
    }

    pub fn hook_set(&self) -> LutumHooksSet<'static> {
        LutumHooksSet::new()
            .with_on_model_input(self.clone())
            .with_on_stream_event(self.clone())
    }

    pub fn turn_dir_for_test(
        root: &Path,
        namespace: &[String],
        module: &str,
        turn: u64,
    ) -> PathBuf {
        turn_dir(root, namespace, module, turn)
    }

    fn metadata(extensions: &RequestExtensions) -> Option<&LlmRequestMetadata> {
        extensions.get::<LlmRequestMetadata>()
    }

    fn turn_for(
        &self,
        extensions: &RequestExtensions,
        metadata: &LlmRequestMetadata,
        operation: OperationKind,
    ) -> TurnState {
        let key = extension_key(extensions);
        let mut active = self
            .inner
            .active_turns
            .lock()
            .expect("LLM file trace active turn lock poisoned");
        if let Some(state) = active.get(&key) {
            return state.clone();
        }

        let module = metadata.owner.module.to_string();
        let turn = {
            let mut next_by_module = self
                .inner
                .next_turn_by_module
                .lock()
                .expect("LLM file trace turn counter lock poisoned");
            let next = next_by_module.entry(module.clone()).or_insert(0);
            let turn = *next;
            *next = next.saturating_add(1);
            turn
        };
        let dir = turn_dir(
            &self.inner.context.root,
            &self.inner.context.namespace,
            &module,
            turn,
        );
        if let Err(error) = fs::create_dir_all(&dir) {
            tracing::warn!(path = %dir.display(), ?error, "failed to create LLM trace dir");
        }
        let now = Utc::now();
        let state = TurnState {
            dir,
            trace: TurnTrace::new(turn, operation, metadata, now),
        };
        active.insert(key, state.clone());
        state
    }

    fn update_turn(
        &self,
        extensions: &RequestExtensions,
        metadata: &LlmRequestMetadata,
        operation: OperationKind,
        event_kind: &'static str,
        event_payload: serde_json::Value,
        update: impl FnOnce(&mut TurnTrace),
    ) {
        let key = extension_key(extensions);
        let mut state = self.turn_for(extensions, metadata, operation);
        update(&mut state.trace);
        state.trace.updated_at = Utc::now();
        let record = EventRecord {
            ts: state.trace.updated_at,
            event_kind,
            metadata,
            operation: operation_kind_label(operation),
            event: event_payload,
        };
        if let Err(error) = append_event(&state.dir, &record) {
            tracing::warn!(path = %state.dir.display(), ?error, "failed to append LLM trace event");
        }
        if let Err(error) = rewrite_trace(&state.dir, &state.trace) {
            tracing::warn!(path = %state.dir.display(), ?error, "failed to rewrite LLM trace");
        }

        let completed = state.trace.status == TurnStatus::Completed;
        let mut active = self
            .inner
            .active_turns
            .lock()
            .expect("LLM file trace active turn lock poisoned");
        if completed {
            active.remove(&key);
        } else {
            active.insert(key, state);
        }
    }
}

impl OnModelInput for FileLlmTraceSink {
    async fn call(&self, cx: &ModelInputHookContext<'_>) {
        let Some(metadata) = Self::metadata(cx.extensions()) else {
            return;
        };
        self.update_turn(
            cx.extensions(),
            metadata,
            cx.kind(),
            "model_input",
            serde_json::json!({
                "kind": "model_input",
                "items": model_input_views(cx.input().items()),
            }),
            |trace| {
                trace.model_input = model_input_views(cx.input().items());
            },
        );
    }
}

impl OnStreamEvent for FileLlmTraceSink {
    async fn call(&self, cx: &StreamEventHookContext<'_>) {
        let Some(metadata) = Self::metadata(cx.extensions()) else {
            return;
        };
        let event_payload = stream_event_json(cx.event());
        self.update_turn(
            cx.extensions(),
            metadata,
            cx.kind(),
            stream_event_kind(cx.event()),
            event_payload,
            |trace| apply_stream_event(trace, cx.event()),
        );
    }
}

fn extension_key(extensions: &RequestExtensions) -> usize {
    std::ptr::from_ref(extensions) as usize
}

fn turn_dir(root: &Path, namespace: &[String], module: &str, turn: u64) -> PathBuf {
    let mut path = root.to_path_buf();
    for segment in namespace {
        path.push(sanitize_segment(segment));
    }
    path.push(sanitize_segment(module));
    path.push(format!("{turn:06}"));
    path
}

fn sanitize_segment(input: &str) -> String {
    let mut out = String::with_capacity(input.len().max(1));
    let mut previous_dash = false;
    for ch in input.chars() {
        let mapped = if ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_') {
            ch
        } else {
            '-'
        };
        if mapped == '-' {
            if previous_dash {
                continue;
            }
            previous_dash = true;
        } else {
            previous_dash = false;
        }
        out.push(mapped);
    }
    let trimmed = out.trim_matches('-');
    if trimmed.is_empty() {
        "unknown".to_string()
    } else {
        trimmed.to_string()
    }
}

fn append_event(dir: &Path, record: &EventRecord<'_>) -> io::Result<()> {
    let path = dir.join("events.jsonl");
    let mut file = OpenOptions::new().create(true).append(true).open(path)?;
    serde_json::to_writer(&mut file, record).map_err(io::Error::other)?;
    file.write_all(b"\n")?;
    file.flush()
}

fn rewrite_trace(dir: &Path, trace: &TurnTrace) -> io::Result<()> {
    let final_path = dir.join("trace.json");
    let counter = TRACE_TMP_COUNTER.fetch_add(1, Ordering::Relaxed);
    let tmp_path = dir.join(format!("trace.json.{counter}.tmp"));
    {
        let mut file = OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&tmp_path)?;
        serde_json::to_writer_pretty(&mut file, trace).map_err(io::Error::other)?;
        file.write_all(b"\n")?;
        file.flush()?;
    }
    fs::rename(tmp_path, final_path)
}

static TRACE_TMP_COUNTER: AtomicU64 = AtomicU64::new(0);

#[derive(Serialize)]
struct EventRecord<'a> {
    ts: DateTime<Utc>,
    event_kind: &'static str,
    metadata: &'a LlmRequestMetadata,
    operation: &'static str,
    event: serde_json::Value,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
enum TurnStatus {
    InProgress,
    Completed,
}

#[derive(Clone, Debug, Serialize)]
struct TurnTrace {
    version: u32,
    status: TurnStatus,
    turn: u64,
    operation: &'static str,
    metadata: LlmRequestMetadata,
    source: &'static str,
    started_at: DateTime<Utc>,
    updated_at: DateTime<Utc>,
    completed_at: Option<DateTime<Utc>>,
    model_input: Vec<ModelInputItemTrace>,
    request_id: Option<String>,
    model: Option<String>,
    stream_deltas: Vec<StreamDeltaTrace>,
    tool_calls: BTreeMap<String, ToolCallTrace>,
    structured_output: Option<serde_json::Value>,
    retries: Vec<RetryTrace>,
    finish_reason: Option<String>,
    usage: Option<Usage>,
}

impl TurnTrace {
    fn new(
        turn: u64,
        operation: OperationKind,
        metadata: &LlmRequestMetadata,
        now: DateTime<Utc>,
    ) -> Self {
        Self {
            version: 1,
            status: TurnStatus::InProgress,
            turn,
            operation: operation_kind_label(operation),
            source: request_source_label(metadata.source),
            metadata: metadata.clone(),
            started_at: now,
            updated_at: now,
            completed_at: None,
            model_input: Vec::new(),
            request_id: None,
            model: None,
            stream_deltas: Vec::new(),
            tool_calls: BTreeMap::new(),
            structured_output: None,
            retries: Vec::new(),
            finish_reason: None,
            usage: None,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
struct ModelInputItemTrace {
    role: String,
    kind: String,
    content: String,
    ephemeral: bool,
    source: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
struct StreamDeltaTrace {
    kind: String,
    delta: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
struct ToolCallTrace {
    id: String,
    name: String,
    argument_deltas: Vec<String>,
    arguments_json: Option<serde_json::Value>,
}

impl ToolCallTrace {
    fn new(id: String, name: String) -> Self {
        Self {
            id,
            name,
            argument_deltas: Vec::new(),
            arguments_json: None,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
struct RetryTrace {
    attempt: u32,
    after_ms: u128,
    kind: String,
    status: Option<u16>,
    request_id: Option<String>,
    accounted_usage: Usage,
    cumulative_usage: Usage,
}

fn apply_stream_event(trace: &mut TurnTrace, event: LutumStreamEvent<'_>) {
    match event {
        LutumStreamEvent::TextTurn(event) => apply_text_turn_event(trace, event),
        LutumStreamEvent::StructuredTurn(event) => apply_structured_turn_event(trace, event),
        LutumStreamEvent::Completion(event) => apply_completion_event(trace, event),
        LutumStreamEvent::StructuredCompletion(event) => {
            apply_structured_completion_event(trace, event);
        }
    }
}

fn apply_started(trace: &mut TurnTrace, request_id: &Option<String>, model: &str) {
    trace.request_id = request_id.clone();
    trace.model = Some(model.to_string());
}

fn apply_completed(
    trace: &mut TurnTrace,
    request_id: &Option<String>,
    finish_reason: &impl std::fmt::Debug,
    usage: Usage,
) {
    trace.status = TurnStatus::Completed;
    trace.completed_at = Some(Utc::now());
    trace.request_id = request_id.clone().or_else(|| trace.request_id.clone());
    trace.finish_reason = Some(format!("{finish_reason:?}"));
    trace.usage = Some(usage);
}

fn push_delta(trace: &mut TurnTrace, kind: &str, delta: &str) {
    trace.stream_deltas.push(StreamDeltaTrace {
        kind: kind.to_string(),
        delta: delta.to_string(),
    });
}

fn push_tool_chunk(
    trace: &mut TurnTrace,
    id: &lutum::ToolCallId,
    name: &lutum::ToolName,
    arguments_json_delta: &str,
) {
    let key = id.as_str().to_string();
    let entry = trace
        .tool_calls
        .entry(key.clone())
        .or_insert_with(|| ToolCallTrace::new(key, name.as_str().to_string()));
    entry.argument_deltas.push(arguments_json_delta.to_string());
}

fn set_tool_ready(trace: &mut TurnTrace, metadata: &lutum::ToolMetadata) {
    let key = metadata.id.as_str().to_string();
    let entry = trace
        .tool_calls
        .entry(key.clone())
        .or_insert_with(|| ToolCallTrace::new(key, metadata.name.as_str().to_string()));
    entry.name = metadata.name.as_str().to_string();
    entry.arguments_json = Some(raw_json_value(&metadata.arguments));
}

fn apply_text_turn_event(trace: &mut TurnTrace, event: &ErasedTextTurnEvent) {
    match event {
        ErasedTextTurnEvent::Started { request_id, model } => {
            apply_started(trace, request_id, model);
        }
        ErasedTextTurnEvent::TextDelta { delta } => push_delta(trace, "text", delta),
        ErasedTextTurnEvent::ReasoningDelta { delta } => push_delta(trace, "reasoning", delta),
        ErasedTextTurnEvent::RefusalDelta { delta } => push_delta(trace, "refusal", delta),
        ErasedTextTurnEvent::ToolCallChunk {
            id,
            name,
            arguments_json_delta,
        } => push_tool_chunk(trace, id, name, arguments_json_delta),
        ErasedTextTurnEvent::ToolCallReady(tool) => set_tool_ready(trace, tool),
        ErasedTextTurnEvent::Completed {
            request_id,
            finish_reason,
            usage,
            ..
        } => apply_completed(trace, request_id, finish_reason, *usage),
    }
}

fn apply_structured_turn_event(trace: &mut TurnTrace, event: &ErasedStructuredTurnEvent) {
    match event {
        ErasedStructuredTurnEvent::Started { request_id, model } => {
            apply_started(trace, request_id, model);
        }
        ErasedStructuredTurnEvent::StructuredOutputChunk { json_delta } => {
            push_delta(trace, "structured", json_delta);
        }
        ErasedStructuredTurnEvent::StructuredOutputReady(json) => {
            trace.structured_output = Some(raw_json_value(json));
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
        ErasedStructuredTurnEvent::ToolCallReady(tool) => set_tool_ready(trace, tool),
        ErasedStructuredTurnEvent::Completed {
            request_id,
            finish_reason,
            usage,
            ..
        } => apply_completed(trace, request_id, finish_reason, *usage),
    }
}

fn apply_completion_event(trace: &mut TurnTrace, event: &CompletionEvent) {
    match event {
        CompletionEvent::Started { request_id, model } => {
            apply_started(trace, request_id, model);
        }
        CompletionEvent::WillRetry {
            attempt,
            after,
            kind,
            status,
            request_id,
            accounted_usage,
            cumulative_usage,
        } => trace.retries.push(RetryTrace {
            attempt: *attempt,
            after_ms: after.as_millis(),
            kind: format!("{kind:?}"),
            status: *status,
            request_id: request_id.clone(),
            accounted_usage: *accounted_usage,
            cumulative_usage: *cumulative_usage,
        }),
        CompletionEvent::TextDelta(delta) => push_delta(trace, "text", delta),
        CompletionEvent::Completed {
            request_id,
            finish_reason,
            usage,
        } => apply_completed(trace, request_id, finish_reason, *usage),
    }
}

fn apply_structured_completion_event(
    trace: &mut TurnTrace,
    event: &ErasedStructuredCompletionEvent,
) {
    match event {
        ErasedStructuredCompletionEvent::Started { request_id, model } => {
            apply_started(trace, request_id, model);
        }
        ErasedStructuredCompletionEvent::StructuredOutputChunk { json_delta } => {
            push_delta(trace, "structured", json_delta);
        }
        ErasedStructuredCompletionEvent::StructuredOutputReady(json) => {
            trace.structured_output = Some(raw_json_value(json));
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
        } => apply_completed(trace, request_id, finish_reason, *usage),
    }
}

fn stream_event_kind(event: LutumStreamEvent<'_>) -> &'static str {
    match event {
        LutumStreamEvent::TextTurn(ErasedTextTurnEvent::Started { .. })
        | LutumStreamEvent::StructuredTurn(ErasedStructuredTurnEvent::Started { .. })
        | LutumStreamEvent::Completion(CompletionEvent::Started { .. })
        | LutumStreamEvent::StructuredCompletion(ErasedStructuredCompletionEvent::Started {
            ..
        }) => "stream_started",
        LutumStreamEvent::TextTurn(ErasedTextTurnEvent::Completed { .. })
        | LutumStreamEvent::StructuredTurn(ErasedStructuredTurnEvent::Completed { .. })
        | LutumStreamEvent::Completion(CompletionEvent::Completed { .. })
        | LutumStreamEvent::StructuredCompletion(ErasedStructuredCompletionEvent::Completed {
            ..
        }) => "completed",
        _ => "stream_event",
    }
}

fn stream_event_json(event: LutumStreamEvent<'_>) -> serde_json::Value {
    match event {
        LutumStreamEvent::TextTurn(event) => text_turn_event_json(event),
        LutumStreamEvent::StructuredTurn(event) => structured_turn_event_json(event),
        LutumStreamEvent::Completion(event) => completion_event_json(event),
        LutumStreamEvent::StructuredCompletion(event) => structured_completion_event_json(event),
    }
}

fn text_turn_event_json(event: &ErasedTextTurnEvent) -> serde_json::Value {
    match event {
        ErasedTextTurnEvent::Started { request_id, model } => serde_json::json!({
            "kind": "started",
            "request_id": request_id,
            "model": model,
        }),
        ErasedTextTurnEvent::TextDelta { delta } => delta_event_json("text", delta),
        ErasedTextTurnEvent::ReasoningDelta { delta } => delta_event_json("reasoning", delta),
        ErasedTextTurnEvent::RefusalDelta { delta } => delta_event_json("refusal", delta),
        ErasedTextTurnEvent::ToolCallChunk {
            id,
            name,
            arguments_json_delta,
        } => serde_json::json!({
            "kind": "tool_call_chunk",
            "id": id.as_str(),
            "name": name.as_str(),
            "arguments_json_delta": arguments_json_delta,
        }),
        ErasedTextTurnEvent::ToolCallReady(tool) => tool_ready_json(tool),
        ErasedTextTurnEvent::Completed {
            request_id,
            finish_reason,
            usage,
            ..
        } => completed_event_json(request_id, finish_reason, *usage),
    }
}

fn structured_turn_event_json(event: &ErasedStructuredTurnEvent) -> serde_json::Value {
    match event {
        ErasedStructuredTurnEvent::Started { request_id, model } => serde_json::json!({
            "kind": "started",
            "request_id": request_id,
            "model": model,
        }),
        ErasedStructuredTurnEvent::StructuredOutputChunk { json_delta } => {
            delta_event_json("structured", json_delta)
        }
        ErasedStructuredTurnEvent::StructuredOutputReady(json) => serde_json::json!({
            "kind": "structured_output_ready",
            "json": raw_json_value(json),
        }),
        ErasedStructuredTurnEvent::ReasoningDelta { delta } => delta_event_json("reasoning", delta),
        ErasedStructuredTurnEvent::RefusalDelta { delta } => delta_event_json("refusal", delta),
        ErasedStructuredTurnEvent::ToolCallChunk {
            id,
            name,
            arguments_json_delta,
        } => serde_json::json!({
            "kind": "tool_call_chunk",
            "id": id.as_str(),
            "name": name.as_str(),
            "arguments_json_delta": arguments_json_delta,
        }),
        ErasedStructuredTurnEvent::ToolCallReady(tool) => tool_ready_json(tool),
        ErasedStructuredTurnEvent::Completed {
            request_id,
            finish_reason,
            usage,
            ..
        } => completed_event_json(request_id, finish_reason, *usage),
    }
}

fn completion_event_json(event: &CompletionEvent) -> serde_json::Value {
    match event {
        CompletionEvent::Started { request_id, model } => serde_json::json!({
            "kind": "started",
            "request_id": request_id,
            "model": model,
        }),
        CompletionEvent::WillRetry {
            attempt,
            after,
            kind,
            status,
            request_id,
            accounted_usage,
            cumulative_usage,
        } => serde_json::json!({
            "kind": "will_retry",
            "attempt": attempt,
            "after_ms": after.as_millis(),
            "request_failure_kind": format!("{kind:?}"),
            "status": status,
            "request_id": request_id,
            "accounted_usage": accounted_usage,
            "cumulative_usage": cumulative_usage,
        }),
        CompletionEvent::TextDelta(delta) => delta_event_json("text", delta),
        CompletionEvent::Completed {
            request_id,
            finish_reason,
            usage,
        } => completed_event_json(request_id, finish_reason, *usage),
    }
}

fn structured_completion_event_json(event: &ErasedStructuredCompletionEvent) -> serde_json::Value {
    match event {
        ErasedStructuredCompletionEvent::Started { request_id, model } => serde_json::json!({
            "kind": "started",
            "request_id": request_id,
            "model": model,
        }),
        ErasedStructuredCompletionEvent::StructuredOutputChunk { json_delta } => {
            delta_event_json("structured", json_delta)
        }
        ErasedStructuredCompletionEvent::StructuredOutputReady(json) => serde_json::json!({
            "kind": "structured_output_ready",
            "json": raw_json_value(json),
        }),
        ErasedStructuredCompletionEvent::ReasoningDelta { delta } => {
            delta_event_json("reasoning", delta)
        }
        ErasedStructuredCompletionEvent::RefusalDelta { delta } => {
            delta_event_json("refusal", delta)
        }
        ErasedStructuredCompletionEvent::Completed {
            request_id,
            finish_reason,
            usage,
        } => completed_event_json(request_id, finish_reason, *usage),
    }
}

fn delta_event_json(kind: &str, delta: &str) -> serde_json::Value {
    serde_json::json!({
        "kind": "delta",
        "delta_kind": kind,
        "delta": delta,
    })
}

fn tool_ready_json(tool: &lutum::ToolMetadata) -> serde_json::Value {
    serde_json::json!({
        "kind": "tool_call_ready",
        "id": tool.id.as_str(),
        "name": tool.name.as_str(),
        "arguments_json": raw_json_value(&tool.arguments),
    })
}

fn raw_json_value(json: &lutum::RawJson) -> serde_json::Value {
    serde_json::from_str(json.get())
        .unwrap_or_else(|_| serde_json::Value::String(json.get().to_string()))
}

fn completed_event_json(
    request_id: &Option<String>,
    finish_reason: &impl std::fmt::Debug,
    usage: Usage,
) -> serde_json::Value {
    serde_json::json!({
        "kind": "completed",
        "request_id": request_id,
        "finish_reason": format!("{finish_reason:?}"),
        "usage": usage,
    })
}

fn model_input_views(items: &[ModelInputItem]) -> Vec<ModelInputItemTrace> {
    let mut output = Vec::new();
    for item in items {
        match item {
            ModelInputItem::Message { role, content } => {
                for content in content.as_slice() {
                    let MessageContent::Text(text) = content;
                    output.push(input_item(
                        input_role_label(*role),
                        "text",
                        text,
                        false,
                        None,
                    ));
                }
            }
            ModelInputItem::Assistant(item) => {
                let (kind, content) = assistant_input_view(item);
                output.push(input_item("assistant", kind, content, false, None));
            }
            ModelInputItem::ToolResult(result) => {
                output.push(input_item(
                    "tool",
                    "tool_result",
                    &format!(
                        "arguments:\n{}\nresult:\n{}",
                        result.arguments, result.result
                    ),
                    false,
                    Some(format!("{}({})", result.name, result.id)),
                ));
            }
            ModelInputItem::Turn(turn) => push_turn_view(&mut output, turn.as_ref()),
        }
    }
    output
}

fn push_turn_view(output: &mut Vec<ModelInputItemTrace>, turn: &dyn TurnView) {
    for index in 0..turn.item_count() {
        let Some(item) = turn.item_at(index) else {
            continue;
        };
        output.push(item_view(turn.role(), turn.ephemeral(), item));
    }
}

fn item_view(role: TurnRole, ephemeral: bool, item: &dyn ItemView) -> ModelInputItemTrace {
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
) -> ModelInputItemTrace {
    ModelInputItemTrace {
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

fn request_source_label(source: LlmRequestSource) -> &'static str {
    match source {
        LlmRequestSource::ModuleTurn => "module_turn",
        LlmRequestSource::SessionCompaction => "session_compaction",
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

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::{Path, PathBuf};

    use lutum::{
        CompletionEvent, FinishReason, ModelInput, ModelInputHookContext, OperationKind,
        RequestExtensions, StreamEventHookContext,
    };
    use nuillu_module::{LlmBatchDebug, LlmRequestMetadata, LlmRequestSource};
    use nuillu_types::{ModelTier, ModuleInstanceId, ReplicaIndex, builtin};

    use super::*;

    fn metadata() -> LlmRequestMetadata {
        LlmRequestMetadata {
            owner: ModuleInstanceId::new(builtin::cognition_gate(), ReplicaIndex::ZERO),
            tier: ModelTier::Premium,
            source: LlmRequestSource::ModuleTurn,
            activation_attempt: Some(1),
            batch: Some(LlmBatchDebug {
                batch_type: "test::Batch".to_string(),
                batch_debug: "Batch(\"debug\")".to_string(),
            }),
        }
    }

    fn extensions() -> RequestExtensions {
        let mut extensions = RequestExtensions::new();
        extensions.insert(metadata());
        extensions
    }

    fn read_json(path: &Path) -> serde_json::Value {
        serde_json::from_slice(&fs::read(path).expect("read json file")).expect("parse json")
    }

    #[test]
    fn eval_turn_dir_sanitizes_namespace_module_and_pads_turn() {
        let root = Path::new("llm-logs");
        let namespace = vec!["run/../1".to_string(), "case one".to_string()];
        assert_eq!(
            FileLlmTraceSink::turn_dir_for_test(root, &namespace, "module/name", 7),
            PathBuf::from("llm-logs/run-1/case-one/module-name/000007")
        );
    }

    #[test]
    fn server_turn_dir_uses_single_session_namespace() {
        let root = Path::new("llm-logs");
        let namespace = vec!["server-session".to_string()];
        assert_eq!(
            FileLlmTraceSink::turn_dir_for_test(root, &namespace, "speak", 0),
            PathBuf::from("llm-logs/server-session/speak/000000")
        );
    }

    #[tokio::test]
    async fn partial_stream_event_remains_in_events_jsonl() {
        let dir = tempfile::tempdir().unwrap();
        let sink = FileLlmTraceSink::new(LlmLogContext::new(
            dir.path(),
            vec!["run".to_string(), "case".to_string()],
        ));
        let extensions = extensions();
        let input = ModelInput::new().user("hello");
        let input_cx = ModelInputHookContext::new(&extensions, OperationKind::Completion, &input);
        OnModelInput::call(&sink, &input_cx).await;

        let started = CompletionEvent::Started {
            request_id: Some("req".to_string()),
            model: "model-a".to_string(),
        };
        let started_cx = StreamEventHookContext::new(
            &extensions,
            OperationKind::Completion,
            LutumStreamEvent::Completion(&started),
        );
        OnStreamEvent::call(&sink, &started_cx).await;
        let delta = CompletionEvent::TextDelta("partial".to_string());
        let delta_cx = StreamEventHookContext::new(
            &extensions,
            OperationKind::Completion,
            LutumStreamEvent::Completion(&delta),
        );
        OnStreamEvent::call(&sink, &delta_cx).await;

        let turn_dir = dir.path().join("run/case/cognition-gate/000000");
        let jsonl = fs::read_to_string(turn_dir.join("events.jsonl")).unwrap();
        assert!(jsonl.contains("\"event_kind\":\"model_input\""));
        assert!(jsonl.contains("partial"));
        assert!(!jsonl.contains("\"event_kind\":\"completed\""));
        let trace = read_json(&turn_dir.join("trace.json"));
        assert_eq!(trace["status"], "in_progress");
        assert_eq!(trace["stream_deltas"][0]["delta"], "partial");
    }

    #[tokio::test]
    async fn completed_turn_trace_aggregates_input_stream_usage_and_batch_debug() {
        let dir = tempfile::tempdir().unwrap();
        let sink =
            FileLlmTraceSink::new(LlmLogContext::new(dir.path(), vec!["session".to_string()]));
        let extensions = extensions();
        let input = ModelInput::new().system("SYSTEM").user("hello");
        let input_cx = ModelInputHookContext::new(&extensions, OperationKind::Completion, &input);
        OnModelInput::call(&sink, &input_cx).await;

        let started = CompletionEvent::Started {
            request_id: Some("req-1".to_string()),
            model: "model-a".to_string(),
        };
        OnStreamEvent::call(
            &sink,
            &StreamEventHookContext::new(
                &extensions,
                OperationKind::Completion,
                LutumStreamEvent::Completion(&started),
            ),
        )
        .await;
        let delta = CompletionEvent::TextDelta("hello".to_string());
        OnStreamEvent::call(
            &sink,
            &StreamEventHookContext::new(
                &extensions,
                OperationKind::Completion,
                LutumStreamEvent::Completion(&delta),
            ),
        )
        .await;
        let completed = CompletionEvent::Completed {
            request_id: Some("req-1".to_string()),
            finish_reason: FinishReason::Stop,
            usage: Usage {
                input_tokens: 2,
                output_tokens: 3,
                total_tokens: 5,
                ..Usage::zero()
            },
        };
        OnStreamEvent::call(
            &sink,
            &StreamEventHookContext::new(
                &extensions,
                OperationKind::Completion,
                LutumStreamEvent::Completion(&completed),
            ),
        )
        .await;

        let trace = read_json(&dir.path().join("session/cognition-gate/000000/trace.json"));
        assert_eq!(trace["status"], "completed");
        assert_eq!(trace["operation"], "completion");
        assert_eq!(trace["request_id"], "req-1");
        assert_eq!(trace["model"], "model-a");
        assert_eq!(trace["model_input"][0]["role"], "system");
        assert_eq!(trace["model_input"][1]["content"], "hello");
        assert_eq!(trace["stream_deltas"][0]["delta"], "hello");
        assert_eq!(trace["usage"]["total_tokens"], 5);
        assert_eq!(trace["metadata"]["activation_attempt"], 1);
        assert_eq!(
            trace["metadata"]["batch"]["batch_debug"],
            "Batch(\"debug\")"
        );
    }
}
