use lutum_eval::{EventRecord, FieldValue, RawTraceSnapshot, SpanNode, TraceSnapshot};
use lutum_trace::RawTraceEntry;

pub fn trace_snapshot_json(trace: &TraceSnapshot) -> serde_json::Value {
    serde_json::json!({
        "roots": trace.roots.iter().map(span_json).collect::<Vec<_>>(),
        "root_events": trace.root_events.iter().map(event_json).collect::<Vec<_>>(),
    })
}

pub fn raw_trace_snapshot_json(raw: &RawTraceSnapshot) -> serde_json::Value {
    serde_json::json!({
        "entries": raw.entries.iter().map(raw_entry_json).collect::<Vec<_>>(),
    })
}

pub fn raw_trace_has_error(raw: &RawTraceSnapshot) -> bool {
    raw.entries.iter().any(|entry| {
        matches!(
            entry,
            RawTraceEntry::ParseError { .. }
                | RawTraceEntry::RequestError { .. }
                | RawTraceEntry::CollectError { .. }
        )
    })
}

fn span_json(span: &SpanNode) -> serde_json::Value {
    serde_json::json!({
        "name": &span.name,
        "target": &span.target,
        "level": &span.level,
        "fields": fields_json(&span.fields),
        "events": span.events.iter().map(event_json).collect::<Vec<_>>(),
        "children": span.children.iter().map(span_json).collect::<Vec<_>>(),
    })
}

fn event_json(event: &EventRecord) -> serde_json::Value {
    serde_json::json!({
        "target": &event.target,
        "level": &event.level,
        "message": &event.message,
        "fields": fields_json(&event.fields),
    })
}

fn fields_json(fields: &[(String, FieldValue)]) -> serde_json::Value {
    let mut object = serde_json::Map::new();
    for (key, value) in fields {
        object.insert(key.clone(), field_value_json(value));
    }
    serde_json::Value::Object(object)
}

fn field_value_json(value: &FieldValue) -> serde_json::Value {
    match value {
        FieldValue::Bool(value) => serde_json::Value::Bool(*value),
        FieldValue::I64(value) => serde_json::json!(value),
        FieldValue::U64(value) => serde_json::json!(value),
        FieldValue::I128(value) => serde_json::json!(value.to_string()),
        FieldValue::U128(value) => serde_json::json!(value.to_string()),
        FieldValue::F64(value) => serde_json::Number::from_f64(*value)
            .map(serde_json::Value::Number)
            .unwrap_or_else(|| serde_json::Value::String(value.to_string())),
        FieldValue::Str(value) => serde_json::Value::String(value.clone()),
    }
}

fn raw_entry_json(entry: &RawTraceEntry) -> serde_json::Value {
    match entry {
        RawTraceEntry::Request {
            provider,
            api,
            operation,
            request_id,
            body,
        } => serde_json::json!({
            "kind": "request",
            "provider": provider,
            "api": api,
            "operation": operation,
            "request_id": request_id,
            "body": body,
        }),
        RawTraceEntry::StreamEvent {
            provider,
            api,
            operation,
            request_id,
            sequence,
            payload,
            event_name,
        } => serde_json::json!({
            "kind": "stream_event",
            "provider": provider,
            "api": api,
            "operation": operation,
            "request_id": request_id,
            "sequence": sequence,
            "payload": payload,
            "event_name": event_name,
        }),
        RawTraceEntry::ParseError {
            provider,
            api,
            operation,
            request_id,
            stage,
            payload,
            error,
        } => serde_json::json!({
            "kind": "parse_error",
            "provider": provider,
            "api": api,
            "operation": operation,
            "request_id": request_id,
            "stage": stage.as_str(),
            "payload": payload,
            "error": error,
        }),
        RawTraceEntry::RequestError {
            provider,
            api,
            operation,
            request_id,
            kind,
            status,
            payload,
            error,
            error_debug,
            source_chain,
            is_timeout,
            is_connect,
            is_request,
            is_body,
            is_decode,
        } => serde_json::json!({
            "kind": "request_error",
            "provider": provider,
            "api": api,
            "operation": operation,
            "request_id": request_id,
            "request_error_kind": kind.as_str(),
            "status": status,
            "payload": payload,
            "error": error,
            "error_debug": error_debug,
            "source_chain": source_chain,
            "is_timeout": is_timeout,
            "is_connect": is_connect,
            "is_request": is_request,
            "is_body": is_body,
            "is_decode": is_decode,
        }),
        RawTraceEntry::CollectError {
            operation_kind,
            request_id,
            kind,
            partial_summary,
            error,
        } => serde_json::json!({
            "kind": "collect_error",
            "operation": operation_kind_json(*operation_kind),
            "request_id": request_id,
            "collect_kind": kind.as_str(),
            "partial_summary": partial_summary,
            "error": error,
        }),
    }
}

fn operation_kind_json(kind: lutum::OperationKind) -> &'static str {
    match kind {
        lutum::OperationKind::TextTurn => "text_turn",
        lutum::OperationKind::StructuredTurn => "structured_turn",
        lutum::OperationKind::StructuredCompletion => "structured_completion",
        lutum::OperationKind::Completion => "completion",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trace_snapshot_json_preserves_span_tree_and_fields() {
        let trace = TraceSnapshot {
            roots: vec![SpanNode {
                name: "root".to_string(),
                target: "test-target".to_string(),
                level: "INFO".to_string(),
                fields: vec![
                    ("ok".to_string(), FieldValue::Bool(true)),
                    ("name".to_string(), FieldValue::Str("nuillu".to_string())),
                ],
                events: vec![EventRecord {
                    target: "test-event-target".to_string(),
                    level: "WARN".to_string(),
                    message: Some("event message".to_string()),
                    fields: vec![("count".to_string(), FieldValue::U64(3))],
                }],
                children: vec![SpanNode {
                    name: "child".to_string(),
                    target: "child-target".to_string(),
                    level: "DEBUG".to_string(),
                    fields: Vec::new(),
                    events: Vec::new(),
                    children: Vec::new(),
                }],
            }],
            root_events: Vec::new(),
        };

        let actual = trace_snapshot_json(&trace);
        let expected = serde_json::json!({
            "roots": [{
                "name": "root",
                "target": "test-target",
                "level": "INFO",
                "fields": {
                    "ok": true,
                    "name": "nuillu",
                },
                "events": [{
                    "target": "test-event-target",
                    "level": "WARN",
                    "message": "event message",
                    "fields": {
                        "count": 3,
                    },
                }],
                "children": [{
                    "name": "child",
                    "target": "child-target",
                    "level": "DEBUG",
                    "fields": {},
                    "events": [],
                    "children": [],
                }],
            }],
            "root_events": [],
        });
        assert_eq!(actual, expected);
    }
}
