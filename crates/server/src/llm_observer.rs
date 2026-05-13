use std::{
    collections::BTreeMap,
    sync::{
        Arc, Mutex,
        atomic::{AtomicU64, Ordering},
    },
};

use lutum::{
    AssistantInputItem, CompletionEvent, ErasedStructuredCompletionEvent,
    ErasedStructuredTurnEvent, ErasedTextTurnEvent, InputMessageRole, ItemView, LutumHooksSet,
    LutumStreamEvent, MessageContent, ModelInputHookContext, ModelInputItem, OnModelInput,
    OnStreamEvent, OperationKind, RequestExtensions, StreamEventHookContext, TurnRole, TurnView,
    Usage,
};
use nuillu_module::{LlmRequestMetadata, LlmRequestSource};
use nuillu_visualizer_protocol::{
    LlmInputItemView, LlmObservationEvent, LlmObservationSource, LlmUsageView, VisualizerEvent,
    VisualizerTabId,
};

use crate::gui::VisualizerEventSink;

#[derive(Clone)]
pub(super) struct ServerLlmObserver {
    inner: Arc<ServerLlmObserverInner>,
}

struct ServerLlmObserverInner {
    tab_id: String,
    events: VisualizerEventSink,
    next_turn: AtomicU64,
    extension_turns: Mutex<BTreeMap<usize, String>>,
}

impl ServerLlmObserver {
    pub(super) fn new(tab_id: String, events: VisualizerEventSink) -> Self {
        Self {
            inner: Arc::new(ServerLlmObserverInner {
                tab_id,
                events,
                next_turn: AtomicU64::new(0),
                extension_turns: Mutex::new(BTreeMap::new()),
            }),
        }
    }

    pub(super) fn hook_set(&self) -> LutumHooksSet<'static> {
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
            .expect("server visualizer LLM turn map lock poisoned");
        turns
            .entry(key)
            .or_insert_with(|| {
                let next = self.inner.next_turn.fetch_add(1, Ordering::Relaxed);
                format!("{}-llm-{next}", self.inner.tab_id)
            })
            .clone()
    }

    fn clear_turn_for(&self, extensions: &RequestExtensions) {
        let key = extension_key(extensions);
        self.inner
            .extension_turns
            .lock()
            .expect("server visualizer LLM turn map lock poisoned")
            .remove(&key);
    }

    fn emit(&self, event: LlmObservationEvent) {
        self.inner.events.send(VisualizerEvent::LlmObserved {
            tab_id: VisualizerTabId::new(self.inner.tab_id.clone()),
            event,
        });
    }
}

impl OnModelInput for ServerLlmObserver {
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

impl OnStreamEvent for ServerLlmObserver {
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
    observer: &ServerLlmObserver,
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
    observer: &ServerLlmObserver,
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

fn emit_delta(observer: &ServerLlmObserver, turn_id: String, kind: &str, delta: String) {
    observer.emit(LlmObservationEvent::StreamDelta {
        turn_id,
        kind: kind.to_string(),
        delta,
    });
}

fn emit_completed(
    observer: &ServerLlmObserver,
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
    observer: &ServerLlmObserver,
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
    observer: &ServerLlmObserver,
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
    observer: &ServerLlmObserver,
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
    observer: &ServerLlmObserver,
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
    observer: &ServerLlmObserver,
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
    observer: &ServerLlmObserver,
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
