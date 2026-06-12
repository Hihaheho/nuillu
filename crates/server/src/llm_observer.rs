use std::{
    collections::BTreeMap,
    sync::{
        Arc, Mutex,
        atomic::{AtomicU64, Ordering},
    },
};

use lutum::{
    AssistantInputItem, CompletionEvent, ErasedStructuredCompletionEvent,
    ErasedStructuredTurnEvent, ErasedTextTurnEvent, Image, InputMessageRole, ItemView,
    LutumHooksSet, LutumStreamEvent, MessageContent, ModelInputHookContext, ModelInputItem,
    OnModelInput, OnStreamEvent, OperationKind, RequestExtensions, StreamEventHookContext,
    TurnRole, TurnView, Usage,
};
use nuillu_module::{LlmRequestMetadata, LlmRequestSource, ModuleSessionMetadata};
use nuillu_visualizer_protocol::{
    LlmInputItemView, LlmObservationEvent, LlmObservationSource, LlmUsageView, VisualizerEvent,
    VisualizerTabId,
};

use crate::gui::VisualizerEventSink;

const SUPERSEDED_TURN_MESSAGE: &str =
    "superseded by a new LLM model input before terminal stream event";

#[derive(Clone)]
pub struct VisualizerLlmObserver {
    inner: Arc<VisualizerLlmObserverInner>,
}

struct VisualizerLlmObserverInner {
    tab_id: String,
    events: VisualizerEventSink,
    next_turn: AtomicU64,
    extension_turns: Mutex<BTreeMap<usize, String>>,
    activation_attempt_turns: Mutex<BTreeMap<(String, u32), String>>,
}

impl VisualizerLlmObserver {
    pub fn new(tab_id: String, events: VisualizerEventSink) -> Self {
        Self {
            inner: Arc::new(VisualizerLlmObserverInner {
                tab_id,
                events,
                next_turn: AtomicU64::new(0),
                extension_turns: Mutex::new(BTreeMap::new()),
                activation_attempt_turns: Mutex::new(BTreeMap::new()),
            }),
        }
    }

    pub fn hook_set(&self) -> LutumHooksSet<'static> {
        LutumHooksSet::new()
            .with_on_model_input(self.clone())
            .with_on_stream_event(self.clone())
    }

    fn metadata<'a>(&self, extensions: &'a RequestExtensions) -> Option<&'a LlmRequestMetadata> {
        extensions.get::<LlmRequestMetadata>()
    }

    pub fn mark_activation_attempt_failed(
        &self,
        owner: &nuillu_types::ModuleInstanceId,
        activation_attempt: u32,
        message: String,
    ) {
        let key = (owner.to_string(), activation_attempt);
        let turn_id = self
            .inner
            .activation_attempt_turns
            .lock()
            .expect("server visualizer LLM activation turn map lock poisoned")
            .remove(&key);
        let Some(turn_id) = turn_id else {
            return;
        };
        self.inner
            .extension_turns
            .lock()
            .expect("server visualizer LLM turn map lock poisoned")
            .retain(|_, existing| existing != &turn_id);
        self.emit(LlmObservationEvent::Failed { turn_id, message });
    }

    fn turn_id_for(&self, extensions: &RequestExtensions, metadata: &LlmRequestMetadata) -> String {
        let key = extension_key(extensions);
        let turn_id = {
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
        };
        if let Some(activation_attempt) = metadata.activation_attempt {
            self.inner
                .activation_attempt_turns
                .lock()
                .expect("server visualizer LLM activation turn map lock poisoned")
                .insert(
                    (metadata.owner.to_string(), activation_attempt),
                    turn_id.clone(),
                );
        }
        turn_id
    }

    fn start_turn_for_model_input(
        &self,
        extensions: &RequestExtensions,
        metadata: &LlmRequestMetadata,
    ) -> (Option<String>, String) {
        let key = extension_key(extensions);
        let next = self.inner.next_turn.fetch_add(1, Ordering::Relaxed);
        let turn_id = format!("{}-llm-{next}", self.inner.tab_id);
        let superseded = self
            .inner
            .extension_turns
            .lock()
            .expect("server visualizer LLM turn map lock poisoned")
            .insert(key, turn_id.clone());
        if let Some(superseded) = &superseded {
            self.remove_activation_mappings_for_turn(superseded);
        }
        if let Some(activation_attempt) = metadata.activation_attempt {
            self.inner
                .activation_attempt_turns
                .lock()
                .expect("server visualizer LLM activation turn map lock poisoned")
                .insert(
                    (metadata.owner.to_string(), activation_attempt),
                    turn_id.clone(),
                );
        }
        (superseded, turn_id)
    }

    fn clear_turn_for(&self, extensions: &RequestExtensions) {
        let key = extension_key(extensions);
        self.inner
            .extension_turns
            .lock()
            .expect("server visualizer LLM turn map lock poisoned")
            .remove(&key);
    }

    fn remove_activation_mappings_for_turn(&self, turn_id: &str) {
        self.inner
            .activation_attempt_turns
            .lock()
            .expect("server visualizer LLM activation turn map lock poisoned")
            .retain(|_, existing| existing != turn_id);
    }

    fn emit(&self, event: LlmObservationEvent) {
        self.inner.events.send(VisualizerEvent::LlmObserved {
            tab_id: VisualizerTabId::new(self.inner.tab_id.clone()),
            event,
        });
    }
}

impl OnModelInput for VisualizerLlmObserver {
    async fn call(&self, cx: &ModelInputHookContext<'_>) {
        let Some(metadata) = self.metadata(cx.extensions()) else {
            return;
        };
        let (superseded, turn_id) = self.start_turn_for_model_input(cx.extensions(), metadata);
        if let Some(turn_id) = superseded {
            self.emit(LlmObservationEvent::Failed {
                turn_id,
                message: SUPERSEDED_TURN_MESSAGE.to_string(),
            });
        }
        self.emit(LlmObservationEvent::ModelInput {
            turn_id,
            owner: metadata.owner.to_string(),
            module: metadata.owner.module.to_string(),
            replica: metadata.owner.replica.get(),
            tier: format!("{:?}", metadata.tier),
            source: observation_source(metadata.source),
            session_key: observation_session_key(cx.extensions(), metadata),
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
        let turn_id = self.turn_id_for(cx.extensions(), metadata);
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
    extensions: &RequestExtensions,
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
        session_key: observation_session_key(extensions, metadata),
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
            extensions,
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
            extensions,
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
            extensions,
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
            extensions,
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

pub(crate) fn model_input_views(items: &[ModelInputItem]) -> Vec<LlmInputItemView> {
    let mut output = Vec::new();
    for item in items {
        match item {
            ModelInputItem::Message { role, content } => {
                for content in content.as_slice() {
                    let (kind, content) = message_content_view(content);
                    output.push(LlmInputItemView {
                        role: input_role_label(*role).to_string(),
                        kind: kind.to_string(),
                        content,
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
                    content,
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

fn message_content_view(content: &MessageContent) -> (&'static str, String) {
    match content {
        MessageContent::Text(text) => ("text", text.clone()),
        MessageContent::Image(image) => ("image", image_trace_content(image)),
    }
}

fn assistant_input_view(item: &AssistantInputItem) -> (&'static str, String) {
    match item {
        AssistantInputItem::Text(text) => ("text", text.clone()),
        AssistantInputItem::Image(image) => ("image", image_trace_content(image)),
        AssistantInputItem::Reasoning(text) => ("reasoning", text.clone()),
        AssistantInputItem::Refusal(text) => ("refusal", text.clone()),
    }
}

fn image_trace_content(image: &Image) -> String {
    match image {
        Image::Base64 { data, media_type } => {
            format!("base64 image media_type={media_type} bytes={}", data.len())
        }
        Image::Uri(uri) => uri.to_string(),
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

pub(crate) fn observation_source(source: LlmRequestSource) -> LlmObservationSource {
    match source {
        LlmRequestSource::ModuleTurn => LlmObservationSource::ModuleTurn,
        LlmRequestSource::SessionCompaction => LlmObservationSource::SessionCompaction,
    }
}

pub(crate) fn operation_kind_label(kind: OperationKind) -> &'static str {
    match kind {
        OperationKind::TextTurn => "text_turn",
        OperationKind::StructuredTurn => "structured_turn",
        OperationKind::StructuredCompletion => "structured_completion",
        OperationKind::Completion => "completion",
    }
}

pub(crate) fn usage_view(usage: Usage) -> LlmUsageView {
    LlmUsageView {
        input_tokens: usage.input_tokens,
        output_tokens: usage.output_tokens,
        total_tokens: usage.total_tokens,
        cost_micros_usd: usage.cost_micros_usd,
        cache_creation_tokens: usage.cache_creation_tokens,
        cache_read_tokens: usage.cache_read_tokens,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::sync::mpsc;

    use lutum::ModelInput;
    use nuillu_types::{ModelTier, ModuleInstanceId, ReplicaIndex, builtin};
    use nuillu_visualizer_protocol::VisualizerServerMessage;

    #[tokio::test]
    async fn model_input_supersedes_stale_turn_before_starting_new_turn() {
        let (tx, rx) = mpsc::channel();
        let observer =
            VisualizerLlmObserver::new("server".to_string(), VisualizerEventSink::new(tx));
        let owner = ModuleInstanceId::new(builtin::predict(), ReplicaIndex::ZERO);
        let mut extensions = RequestExtensions::new();
        extensions.insert(metadata(&owner, 1));

        let first_input = ModelInput::new().user("first");
        let first_cx =
            ModelInputHookContext::new(&extensions, OperationKind::Completion, &first_input);
        OnModelInput::call(&observer, &first_cx).await;

        extensions.insert(metadata(&owner, 2));
        let second_input = ModelInput::new().user("second");
        let second_cx =
            ModelInputHookContext::new(&extensions, OperationKind::Completion, &second_input);
        OnModelInput::call(&observer, &second_cx).await;

        let events = observed_events(&rx);
        assert_eq!(events.len(), 3);

        let first_turn_id = model_input_turn_id(&events[0]);
        let failed_turn_id = failed_turn_id(&events[1]);
        let second_turn_id = model_input_turn_id(&events[2]);

        assert_eq!(failed_turn_id, first_turn_id);
        assert_ne!(second_turn_id, first_turn_id);
        assert_eq!(
            failed_message(&events[1]).as_deref(),
            Some(SUPERSEDED_TURN_MESSAGE)
        );

        let activation_turns = observer
            .inner
            .activation_attempt_turns
            .lock()
            .expect("activation turn map lock poisoned")
            .clone();
        assert!(!activation_turns.contains_key(&(owner.to_string(), 1)));
        assert_eq!(
            activation_turns.get(&(owner.to_string(), 2)),
            Some(&second_turn_id)
        );
    }

    fn metadata(owner: &ModuleInstanceId, activation_attempt: u32) -> LlmRequestMetadata {
        LlmRequestMetadata {
            owner: owner.clone(),
            tier: ModelTier::Default,
            source: LlmRequestSource::ModuleTurn,
            session_key: None,
            activation_attempt: Some(activation_attempt),
            batch: None,
        }
    }

    fn observed_events(rx: &mpsc::Receiver<VisualizerServerMessage>) -> Vec<LlmObservationEvent> {
        rx.try_iter()
            .map(|message| match message {
                VisualizerServerMessage::Event {
                    event: VisualizerEvent::LlmObserved { event, .. },
                } => event,
                other => panic!("unexpected visualizer message: {other:?}"),
            })
            .collect()
    }

    fn model_input_turn_id(event: &LlmObservationEvent) -> String {
        match event {
            LlmObservationEvent::ModelInput { turn_id, .. } => turn_id.clone(),
            other => panic!("expected model input event: {other:?}"),
        }
    }

    fn failed_turn_id(event: &LlmObservationEvent) -> String {
        match event {
            LlmObservationEvent::Failed { turn_id, .. } => turn_id.clone(),
            other => panic!("expected failed event: {other:?}"),
        }
    }

    fn failed_message(event: &LlmObservationEvent) -> Option<String> {
        match event {
            LlmObservationEvent::Failed { message, .. } => Some(message.clone()),
            _ => None,
        }
    }
}
