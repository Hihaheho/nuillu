use std::collections::BTreeMap;
use std::path::Path;

use anyhow::Context as _;
use chrono::{DateTime, NaiveDateTime, Utc};
use lutum_libsql_adapter::{
    ConversationHistoryEntryRecord, ConversationHistoryRole, LibsqlConversationHistoryStore,
};
use serde::Serialize;

const AGENT_DB_FILE: &str = "agent.db";

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ConversationHistoryExport {
    pub source: String,
    pub sessions: Vec<ConversationHistorySession>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ConversationHistorySession {
    pub id: String,
    pub label: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub started_at: Option<DateTime<Utc>>,
    pub entries: Vec<ConversationHistoryEntry>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ConversationHistoryEntryRole {
    User,
    Agent,
    ExternalAction,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ConversationHistoryEntry {
    pub timestamp: DateTime<Utc>,
    pub role: ConversationHistoryEntryRole,
    pub speaker: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub target: Option<String>,
    pub text: String,
    pub source_table: String,
    pub source_id: i64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub generation_id: Option<u64>,
}

pub async fn export_conversation_history(
    state_dir: &Path,
    session_ids: &[String],
    agent_name: &str,
) -> anyhow::Result<ConversationHistoryExport> {
    let db_path = state_dir.join(AGENT_DB_FILE);
    let store = LibsqlConversationHistoryStore::open(&db_path)
        .await
        .with_context(|| format!("open conversation history db {}", db_path.display()))?;
    let entries = store
        .entries(session_ids)
        .await
        .with_context(|| format!("load conversation history from {}", db_path.display()))?;
    build_conversation_history_export(db_path.display().to_string(), entries, agent_name)
}

pub fn build_conversation_history_export(
    source: String,
    records: Vec<ConversationHistoryEntryRecord>,
    agent_name: &str,
) -> anyhow::Result<ConversationHistoryExport> {
    let mut sessions = BTreeMap::<String, Vec<ConversationHistoryEntry>>::new();
    for record in records {
        let session_id = record.server_session_id.clone();
        sessions
            .entry(session_id)
            .or_default()
            .push(conversation_history_entry(record, agent_name)?);
    }
    let sessions = sessions
        .into_iter()
        .map(|(id, entries)| {
            let (label, started_at) = session_label_and_started_at(&id);
            ConversationHistorySession {
                id,
                label,
                started_at,
                entries,
            }
        })
        .collect();
    Ok(ConversationHistoryExport { source, sessions })
}

pub fn render_conversation_history_markdown(export: &ConversationHistoryExport) -> String {
    let mut out = String::from("# 会話履歴\n\n");
    out.push_str(&format!(
        "- 出典: `{}`\n",
        export.source.replace('`', "\\`")
    ));
    match export.sessions.as_slice() {
        [] => {
            out.push_str("\n履歴はありません。\n");
        }
        [session] => {
            out.push_str(&format!(
                "- セッション: {}\n\n",
                render_session_label(session)
            ));
            append_markdown_list(&mut out, &session.entries);
        }
        sessions => {
            out.push('\n');
            for (index, session) in sessions.iter().enumerate() {
                if index > 0 {
                    out.push('\n');
                }
                out.push_str(&format!(
                    "## セッション: {}\n\n",
                    render_session_label(session)
                ));
                append_markdown_list(&mut out, &session.entries);
            }
        }
    }
    out
}

fn conversation_history_entry(
    record: ConversationHistoryEntryRecord,
    agent_name: &str,
) -> anyhow::Result<ConversationHistoryEntry> {
    let timestamp = DateTime::from_timestamp_millis(record.occurred_at_ms).with_context(|| {
        format!(
            "invalid conversation history timestamp source={} id={} value={}",
            record.source_table, record.source_id, record.occurred_at_ms
        )
    })?;
    let (role, speaker, target) = match record.role {
        ConversationHistoryRole::User => (
            ConversationHistoryEntryRole::User,
            record.speaker,
            record.target,
        ),
        ConversationHistoryRole::Agent => (
            ConversationHistoryEntryRole::Agent,
            agent_name.to_string(),
            record.target,
        ),
        ConversationHistoryRole::ExternalAction => (
            ConversationHistoryEntryRole::ExternalAction,
            record.speaker,
            record.target,
        ),
    };
    Ok(ConversationHistoryEntry {
        timestamp,
        role,
        speaker,
        target,
        text: record.content,
        source_table: record.source_table,
        source_id: record.source_id,
        generation_id: record.generation_id,
    })
}

fn session_label_and_started_at(id: &str) -> (String, Option<DateTime<Utc>>) {
    let Some(rest) = id.strip_prefix("server-") else {
        return (id.to_string(), None);
    };
    let Some(stamp) = rest.get(..16) else {
        return (id.to_string(), None);
    };
    let suffix_ok = rest.len() == 16 || rest.as_bytes().get(16) == Some(&b'-');
    if !suffix_ok {
        return (id.to_string(), None);
    }
    let Ok(naive) = NaiveDateTime::parse_from_str(stamp, "%Y%m%dT%H%M%SZ") else {
        return (id.to_string(), None);
    };
    (
        format!("server-{stamp}"),
        Some(DateTime::from_naive_utc_and_offset(naive, Utc)),
    )
}

fn render_session_label(session: &ConversationHistorySession) -> String {
    if let Some(started_at) = session.started_at {
        format!("{}（{} UTC）", session.label, started_at.format("%Y-%m-%d"))
    } else {
        session.label.clone()
    }
}

fn append_markdown_list(out: &mut String, entries: &[ConversationHistoryEntry]) {
    for entry in entries {
        append_markdown_entry(out, entry);
    }
}

fn markdown_speaker(entry: &ConversationHistoryEntry) -> String {
    match entry.role {
        ConversationHistoryEntryRole::User => {
            format!("**{}**", markdown_inline_text(&entry.speaker))
        }
        ConversationHistoryEntryRole::Agent => match &entry.target {
            Some(target) if !target.trim().is_empty() => format!(
                "{} → {}",
                markdown_inline_text(&entry.speaker),
                markdown_inline_text(target)
            ),
            _ => markdown_inline_text(&entry.speaker),
        },
        ConversationHistoryEntryRole::ExternalAction => match &entry.target {
            Some(target) if !target.trim().is_empty() => format!(
                "{} → {}",
                markdown_inline_text(&entry.speaker),
                markdown_inline_text(target)
            ),
            _ => markdown_inline_text(&entry.speaker),
        },
    }
}

fn append_markdown_entry(out: &mut String, entry: &ConversationHistoryEntry) {
    let text = entry.text.replace('\r', "");
    let mut lines = text.split('\n');
    let first_line = lines.next().unwrap_or_default();

    out.push_str("- ");
    if entry.role != ConversationHistoryEntryRole::User {
        out.push_str(&markdown_speaker(entry));
        out.push_str(": ");
    }
    out.push_str(first_line);
    out.push('\n');

    for line in lines {
        out.push_str("  ");
        out.push_str(line);
        out.push('\n');
    }
}

fn markdown_inline_text(value: &str) -> String {
    let value = value.replace('\r', "").replace('\n', " ");
    let mut escaped = String::with_capacity(value.len());
    for ch in value.chars() {
        match ch {
            '\\' | '`' | '*' | '_' | '[' | ']' => {
                escaped.push('\\');
                escaped.push(ch);
            }
            _ => escaped.push(ch),
        }
    }
    escaped
}

#[cfg(test)]
mod tests {
    use super::*;

    use chrono::TimeZone as _;

    fn record(
        source_id: i64,
        session: &str,
        occurred_at_ms: i64,
        role: ConversationHistoryRole,
        speaker: &str,
        target: Option<&str>,
        content: &str,
        generation_id: Option<u64>,
    ) -> ConversationHistoryEntryRecord {
        ConversationHistoryEntryRecord {
            source_table: match role {
                ConversationHistoryRole::User => "one_shot_sensory_inputs",
                ConversationHistoryRole::Agent => "utterance_events",
                ConversationHistoryRole::ExternalAction => "external_action_events",
            }
            .to_string(),
            source_id,
            server_session_id: session.to_string(),
            occurred_at_ms,
            role,
            speaker: speaker.to_string(),
            target: target.map(str::to_string),
            content: content.to_string(),
            generation_id,
        }
    }

    #[test]
    fn build_export_groups_sessions_and_formats_agent_speaker() {
        let export = build_conversation_history_export(
            ".tmp/exhibition/agent.db".to_string(),
            vec![
                record(
                    1,
                    "server-20260622T023702Z-uuid",
                    Utc.with_ymd_and_hms(2026, 6, 22, 2, 37, 6)
                        .unwrap()
                        .timestamp_millis(),
                    ConversationHistoryRole::User,
                    "Ryo",
                    None,
                    "Ryo says, \"こんにちは\"",
                    None,
                ),
                record(
                    3,
                    "server-20260622T023702Z-uuid",
                    Utc.with_ymd_and_hms(2026, 6, 22, 2, 38, 8)
                        .unwrap()
                        .timestamp_millis(),
                    ConversationHistoryRole::ExternalAction,
                    "action",
                    Some("poet"),
                    "action: poet\narguments: {\"poem\":\"quiet rain\"}\nstatus: accepted: poem recorded",
                    None,
                ),
                record(
                    2,
                    "server-20260622T023702Z-uuid",
                    Utc.with_ymd_and_hms(2026, 6, 22, 2, 38, 7)
                        .unwrap()
                        .timestamp_millis(),
                    ConversationHistoryRole::Agent,
                    "speak",
                    Some("Ryo"),
                    "こんにちは。",
                    Some(7),
                ),
            ],
            "Nui",
        )
        .unwrap();

        assert_eq!(
            export,
            ConversationHistoryExport {
                source: ".tmp/exhibition/agent.db".to_string(),
                sessions: vec![ConversationHistorySession {
                    id: "server-20260622T023702Z-uuid".to_string(),
                    label: "server-20260622T023702Z".to_string(),
                    started_at: Some(Utc.with_ymd_and_hms(2026, 6, 22, 2, 37, 2).unwrap()),
                    entries: vec![
                        ConversationHistoryEntry {
                            timestamp: Utc.with_ymd_and_hms(2026, 6, 22, 2, 37, 6).unwrap(),
                            role: ConversationHistoryEntryRole::User,
                            speaker: "Ryo".to_string(),
                            target: None,
                            text: "Ryo says, \"こんにちは\"".to_string(),
                            source_table: "one_shot_sensory_inputs".to_string(),
                            source_id: 1,
                            generation_id: None,
                        },
                        ConversationHistoryEntry {
                            timestamp: Utc.with_ymd_and_hms(2026, 6, 22, 2, 38, 8).unwrap(),
                            role: ConversationHistoryEntryRole::ExternalAction,
                            speaker: "action".to_string(),
                            target: Some("poet".to_string()),
                            text: "action: poet\narguments: {\"poem\":\"quiet rain\"}\nstatus: accepted: poem recorded"
                                .to_string(),
                            source_table: "external_action_events".to_string(),
                            source_id: 3,
                            generation_id: None,
                        },
                        ConversationHistoryEntry {
                            timestamp: Utc.with_ymd_and_hms(2026, 6, 22, 2, 38, 7).unwrap(),
                            role: ConversationHistoryEntryRole::Agent,
                            speaker: "Nui".to_string(),
                            target: Some("Ryo".to_string()),
                            text: "こんにちは。".to_string(),
                            source_table: "utterance_events".to_string(),
                            source_id: 2,
                            generation_id: Some(7),
                        },
                    ],
                }],
            }
        );
    }

    #[test]
    fn render_markdown_matches_single_session_shape() {
        let export = ConversationHistoryExport {
            source: ".tmp/exhibition/agent.db".to_string(),
            sessions: vec![ConversationHistorySession {
                id: "server-20260622T023702Z-uuid".to_string(),
                label: "server-20260622T023702Z".to_string(),
                started_at: Some(Utc.with_ymd_and_hms(2026, 6, 22, 2, 37, 2).unwrap()),
                entries: vec![
                    ConversationHistoryEntry {
                        timestamp: Utc.with_ymd_and_hms(2026, 6, 22, 2, 37, 6).unwrap(),
                        role: ConversationHistoryEntryRole::User,
                        speaker: "Ryo".to_string(),
                        target: None,
                        text: "Ryo says, \"こんにちは\"".to_string(),
                        source_table: "one_shot_sensory_inputs".to_string(),
                        source_id: 1,
                        generation_id: None,
                    },
                    ConversationHistoryEntry {
                        timestamp: Utc.with_ymd_and_hms(2026, 6, 22, 2, 38, 7).unwrap(),
                        role: ConversationHistoryEntryRole::Agent,
                        speaker: "Nui".to_string(),
                        target: Some("Ryo".to_string()),
                        text: "こんにちは。元気だよ".to_string(),
                        source_table: "utterance_events".to_string(),
                        source_id: 2,
                        generation_id: Some(7),
                    },
                    ConversationHistoryEntry {
                        timestamp: Utc.with_ymd_and_hms(2026, 6, 22, 2, 38, 8).unwrap(),
                        role: ConversationHistoryEntryRole::ExternalAction,
                        speaker: "action".to_string(),
                        target: Some("poet".to_string()),
                        text: "action: poet\narguments: {\"poem\":\"quiet rain\"}\nstatus: accepted: poem recorded"
                            .to_string(),
                        source_table: "external_action_events".to_string(),
                        source_id: 3,
                        generation_id: None,
                    },
                ],
            }],
        };

        assert_eq!(
            render_conversation_history_markdown(&export),
            concat!(
                "# 会話履歴\n\n",
                "- 出典: `.tmp/exhibition/agent.db`\n",
                "- セッション: server-20260622T023702Z（2026-06-22 UTC）\n\n",
                "- Ryo says, \"こんにちは\"\n",
                "- Nui → Ryo: こんにちは。元気だよ\n",
                "- action → poet: action: poet\n",
                "  arguments: {\"poem\":\"quiet rain\"}\n",
                "  status: accepted: poem recorded\n",
            )
        );
    }

    #[test]
    fn render_markdown_preserves_multiline_entry_text_as_list_body() {
        let export = ConversationHistoryExport {
            source: "agent.db".to_string(),
            sessions: vec![ConversationHistorySession {
                id: "session".to_string(),
                label: "session".to_string(),
                started_at: None,
                entries: vec![ConversationHistoryEntry {
                    timestamp: Utc.with_ymd_and_hms(2026, 6, 22, 2, 37, 6).unwrap(),
                    role: ConversationHistoryEntryRole::User,
                    speaker: "Ry|o".to_string(),
                    target: None,
                    text: "line|one\nline two".to_string(),
                    source_table: "one_shot_sensory_inputs".to_string(),
                    source_id: 1,
                    generation_id: None,
                }],
            }],
        };

        assert_eq!(
            render_conversation_history_markdown(&export),
            concat!(
                "# 会話履歴\n\n",
                "- 出典: `agent.db`\n",
                "- セッション: session\n\n",
                "- line|one\n",
                "  line two\n",
            )
        );
    }

    #[test]
    fn json_export_shape_uses_snake_case_roles() {
        let export = ConversationHistoryExport {
            source: "agent.db".to_string(),
            sessions: vec![ConversationHistorySession {
                id: "session".to_string(),
                label: "session".to_string(),
                started_at: None,
                entries: vec![
                    ConversationHistoryEntry {
                        timestamp: Utc.with_ymd_and_hms(2026, 6, 22, 2, 37, 6).unwrap(),
                        role: ConversationHistoryEntryRole::Agent,
                        speaker: "Nui".to_string(),
                        target: Some("Ryo".to_string()),
                        text: "hello".to_string(),
                        source_table: "utterance_events".to_string(),
                        source_id: 2,
                        generation_id: Some(7),
                    },
                    ConversationHistoryEntry {
                        timestamp: Utc.with_ymd_and_hms(2026, 6, 22, 2, 37, 8).unwrap(),
                        role: ConversationHistoryEntryRole::ExternalAction,
                        speaker: "action".to_string(),
                        target: Some("poet".to_string()),
                        text: "action: poet\narguments: {}\nstatus: pending".to_string(),
                        source_table: "external_action_events".to_string(),
                        source_id: 3,
                        generation_id: None,
                    },
                ],
            }],
        };

        assert_eq!(
            serde_json::to_value(&export).unwrap(),
            serde_json::json!({
                "source": "agent.db",
                "sessions": [{
                    "id": "session",
                    "label": "session",
                    "entries": [{
                        "timestamp": "2026-06-22T02:37:06Z",
                        "role": "agent",
                        "speaker": "Nui",
                        "target": "Ryo",
                        "text": "hello",
                        "source_table": "utterance_events",
                        "source_id": 2,
                        "generation_id": 7
                    }, {
                        "timestamp": "2026-06-22T02:37:08Z",
                        "role": "external_action",
                        "speaker": "action",
                        "target": "poet",
                        "text": "action: poet\narguments: {}\nstatus: pending",
                        "source_table": "external_action_events",
                        "source_id": 3
                    }]
                }]
            })
        );
    }
}
