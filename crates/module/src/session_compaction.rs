use anyhow::{Context, Result};
use lutum::{
    AssistantInputItem, InputMessageRole, Lutum, MessageContent, ModelInput, ModelInputItem,
    RawJson, Session, TurnRole,
};
use nuillu_blackboard::MemoLogRecord;

pub const DEFAULT_SESSION_COMPACTION_INPUT_TOKEN_THRESHOLD: u64 = 16_000;
pub const DEFAULT_SESSION_COMPACTION_PREFIX_RATIO: f64 = 0.8;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SessionCompactionConfig {
    pub input_token_threshold: u64,
    pub prefix_ratio: f64,
}

impl Default for SessionCompactionConfig {
    fn default() -> Self {
        Self {
            input_token_threshold: DEFAULT_SESSION_COMPACTION_INPUT_TOKEN_THRESHOLD,
            prefix_ratio: DEFAULT_SESSION_COMPACTION_PREFIX_RATIO,
        }
    }
}

pub fn session_compaction_cutoff(item_count: usize, prefix_ratio: f64) -> Option<usize> {
    if item_count < 2 {
        return None;
    }
    let ratio = if prefix_ratio.is_finite() {
        prefix_ratio.clamp(f64::EPSILON, 1.0)
    } else {
        DEFAULT_SESSION_COMPACTION_PREFIX_RATIO
    };
    let cutoff = ((item_count as f64) * ratio).floor() as usize;
    Some(cutoff.clamp(1, item_count.saturating_sub(1)))
}

pub async fn compact_session_if_needed(
    session: &mut Session,
    input_tokens: u64,
    lutum: &Lutum,
    config: SessionCompactionConfig,
    module_name: &str,
    compacted_prefix: &str,
    compaction_prompt: &str,
) {
    if input_tokens <= config.input_token_threshold {
        return;
    }
    if let Err(error) =
        compact_session(session, lutum, config, compacted_prefix, compaction_prompt).await
    {
        tracing::warn!(
            module = module_name,
            input_tokens,
            threshold = config.input_token_threshold,
            error = ?error,
            "module session compaction failed"
        );
    }
}

pub fn push_unread_memo_logs(session: &mut Session, records: &[MemoLogRecord]) {
    if records.is_empty() {
        return;
    }
    session.push_user(
        serde_json::json!({
            "new_memo_log_items": records,
        })
        .to_string(),
    );
}

pub async fn compact_session(
    session: &mut Session,
    lutum: &Lutum,
    config: SessionCompactionConfig,
    compacted_prefix: &str,
    compaction_prompt: &str,
) -> Result<()> {
    let items = session.input().items();
    let Some(cutoff) = session_compaction_cutoff(items.len(), config.prefix_ratio) else {
        return Ok(());
    };

    let prefix = items[..cutoff].to_vec();
    let suffix = items[cutoff..].to_vec();
    let transcript = serde_json::to_string_pretty(&render_session_items_for_compaction(&prefix))
        .context("render session compaction transcript")?;

    let mut summary_session = Session::new();
    summary_session.push_system(compaction_prompt);
    summary_session.push_user(transcript);
    let summary = summary_session
        .text_turn(lutum)
        .collect()
        .await
        .context("summarize session prefix")?
        .assistant_text();
    let summary = summary.trim();
    if summary.is_empty() {
        tracing::warn!("module session compaction produced an empty summary");
        return Ok(());
    }

    let mut compacted_items = Vec::with_capacity(suffix.len().saturating_add(1));
    compacted_items.push(ModelInputItem::text(
        InputMessageRole::System,
        format!("{compacted_prefix}\n{summary}"),
    ));
    compacted_items.extend(suffix);
    *session = Session::from_input(ModelInput::from_items(compacted_items));
    Ok(())
}

fn input_message_role_text(role: InputMessageRole) -> &'static str {
    match role {
        InputMessageRole::System => "system",
        InputMessageRole::Developer => "developer",
        InputMessageRole::User => "user",
    }
}

fn turn_role_text(role: TurnRole) -> &'static str {
    match role {
        TurnRole::System => "system",
        TurnRole::Developer => "developer",
        TurnRole::User => "user",
        TurnRole::Assistant => "assistant",
    }
}

fn raw_json_value(raw: &RawJson) -> serde_json::Value {
    serde_json::from_str(raw.get()).unwrap_or_else(|_| serde_json::Value::String(raw.to_string()))
}

fn render_message_content_for_compaction(content: &MessageContent) -> serde_json::Value {
    match content {
        MessageContent::Text(text) => serde_json::json!({
            "type": "text",
            "text": text,
        }),
    }
}

fn render_assistant_input_for_compaction(item: &AssistantInputItem) -> serde_json::Value {
    match item {
        AssistantInputItem::Text(text) => serde_json::json!({
            "type": "text",
            "text": text,
        }),
        AssistantInputItem::Reasoning(text) => serde_json::json!({
            "type": "reasoning",
            "text": text,
        }),
        AssistantInputItem::Refusal(text) => serde_json::json!({
            "type": "refusal",
            "text": text,
        }),
    }
}

fn render_turn_item_for_compaction(item: &dyn lutum::ItemView) -> serde_json::Value {
    if let Some(text) = item.as_text() {
        return serde_json::json!({
            "type": "text",
            "text": text,
        });
    }
    if let Some(text) = item.as_reasoning() {
        return serde_json::json!({
            "type": "reasoning",
            "text": text,
        });
    }
    if let Some(text) = item.as_refusal() {
        return serde_json::json!({
            "type": "refusal",
            "text": text,
        });
    }
    if let Some(call) = item.as_tool_call() {
        return serde_json::json!({
            "type": "tool_call",
            "id": call.id.to_string(),
            "name": call.name.to_string(),
            "arguments": raw_json_value(call.arguments),
        });
    }
    if let Some(result) = item.as_tool_result() {
        return serde_json::json!({
            "type": "tool_result",
            "id": result.id.to_string(),
            "name": result.name.to_string(),
            "arguments": raw_json_value(result.arguments),
            "result": raw_json_value(result.result),
        });
    }
    serde_json::json!({
        "type": "unknown",
    })
}

fn render_session_item_for_compaction(index: usize, item: &ModelInputItem) -> serde_json::Value {
    match item {
        ModelInputItem::Message { role, content } => serde_json::json!({
            "index": index,
            "kind": "message",
            "role": input_message_role_text(*role),
            "content": content
                .as_slice()
                .iter()
                .map(render_message_content_for_compaction)
                .collect::<Vec<_>>(),
        }),
        ModelInputItem::Assistant(item) => serde_json::json!({
            "index": index,
            "kind": "assistant_input",
            "item": render_assistant_input_for_compaction(item),
        }),
        ModelInputItem::ToolResult(result) => serde_json::json!({
            "index": index,
            "kind": "tool_result",
            "id": result.id.to_string(),
            "name": result.name.to_string(),
            "arguments": raw_json_value(&result.arguments),
            "result": raw_json_value(&result.result),
        }),
        ModelInputItem::Turn(turn) => {
            let items = (0..turn.item_count())
                .filter_map(|item_index| turn.item_at(item_index))
                .map(render_turn_item_for_compaction)
                .collect::<Vec<_>>();
            serde_json::json!({
                "index": index,
                "kind": "turn",
                "role": turn_role_text(turn.role()),
                "items": items,
            })
        }
    }
}

pub fn render_session_items_for_compaction(items: &[ModelInputItem]) -> serde_json::Value {
    serde_json::Value::Array(
        items
            .iter()
            .enumerate()
            .map(|(index, item)| render_session_item_for_compaction(index, item))
            .collect(),
    )
}
