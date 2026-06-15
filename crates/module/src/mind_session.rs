use chrono::{DateTime, Utc};
use lutum::{InputMessageRole, ModelInputItem, Session};
use nuillu_blackboard::{CognitionLogEntryRecord, IdentityMemoryRecord, MemoLogRecord};

use crate::{
    LlmContextWindow, format_bounded_cognition_log_batch, format_bounded_memo_log_batch,
    format_identity_memory_seed,
};

pub const REASONING_SYSTEM_PROMPT: &str = "During reasoning, reason extremely concisely: use at most 4 short sentences or 128 tokens of internal deliberation before deciding.";

pub fn format_system_seed(
    system_prompt: impl Into<String>,
    reasoning: bool,
    identity_memories: &[IdentityMemoryRecord],
    now: DateTime<Utc>,
) -> String {
    let mut sections = vec![system_prompt.into().trim_end().to_owned()];
    if reasoning {
        sections.push(REASONING_SYSTEM_PROMPT.to_owned());
    }
    if let Some(seed) = format_identity_memory_seed(identity_memories, now) {
        sections.push(seed);
    }
    sections
        .into_iter()
        .filter(|section| !section.trim().is_empty())
        .collect::<Vec<_>>()
        .join("\n\n")
}

pub fn seed_persistent_faculty_session(
    session: &mut Session,
    system_prompt: impl Into<String>,
    reasoning: bool,
    identity_memories: &[IdentityMemoryRecord],
    now: DateTime<Utc>,
) {
    let seed_item = ModelInputItem::text(
        InputMessageRole::System,
        format_system_seed(system_prompt, reasoning, identity_memories, now),
    );
    session.input_mut().items_mut().insert(0, seed_item);
}

pub fn push_formatted_cognition_log_batch(
    session: &mut Session,
    records: &[CognitionLogEntryRecord],
    now: DateTime<Utc>,
    window: LlmContextWindow,
) {
    if let Some(batch) = format_bounded_cognition_log_batch(records, now, window) {
        session.push_user(batch);
    }
}

pub fn push_formatted_memo_log_batch(
    session: &mut Session,
    records: &[MemoLogRecord],
    now: DateTime<Utc>,
    window: LlmContextWindow,
) {
    if let Some(batch) = format_bounded_memo_log_batch(records, now, window) {
        session.push_system(batch);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use chrono::TimeZone as _;
    use nuillu_blackboard::{CognitionLogEntry, MemoLogRecord};
    use nuillu_types::{MemoryContent, MemoryIndex, ReplicaIndex, builtin};

    use lutum::{InputMessageRole, MessageContent, ModelInputItem};

    fn now() -> DateTime<Utc> {
        Utc.with_ymd_and_hms(2026, 5, 11, 6, 23, 0).unwrap()
    }

    #[test]
    fn helpers_push_blackboard_context_with_expected_session_roles() {
        let owner = nuillu_types::ModuleInstanceId::new(builtin::sensory(), ReplicaIndex::ZERO);
        let mut session = Session::new();

        seed_persistent_faculty_session(
            &mut session,
            "SYSTEM",
            true,
            &[IdentityMemoryRecord {
                index: MemoryIndex::new("identity-1"),
                content: MemoryContent::new("The agent is named Nuillu."),
                occurred_at: None,
            }],
            now(),
        );
        push_formatted_cognition_log_batch(
            &mut session,
            &[CognitionLogEntryRecord {
                index: 0,
                source: owner.clone(),
                entry: CognitionLogEntry {
                    at: Utc.with_ymd_and_hms(2026, 5, 11, 6, 22, 50).unwrap(),
                    text: "The door is open.".into(),
                },
            }],
            now(),
            LlmContextWindow::new(8, 360, 3_000),
        );
        push_formatted_memo_log_batch(
            &mut session,
            &[MemoLogRecord {
                owner,
                index: 0,
                written_at: Utc.with_ymd_and_hms(2026, 5, 11, 6, 22, 0).unwrap(),
                content: "A memo fact.".into(),
                cognitive: false,
            }],
            now(),
            LlmContextWindow::new(8, 420, 3_000),
        );

        let items = session.input().items();
        let ModelInputItem::Message { role, content } = &items[0] else {
            panic!("expected system message first");
        };
        assert_eq!(role, &InputMessageRole::System);
        let [MessageContent::Text(system)] = content.as_slice() else {
            panic!("expected system text");
        };
        assert!(system.starts_with("SYSTEM\n\n"));
        assert!(system.contains(REASONING_SYSTEM_PROMPT));
        assert!(system.contains("What I already remember about myself"));
        assert_eq!(
            system
                .matches("What I already remember about myself")
                .count(),
            1
        );
        assert!(!system.contains("Identity memory loaded at agent startup"));

        let ModelInputItem::Message { role, content } = &items[1] else {
            panic!("expected cognition batch user text second");
        };
        assert_eq!(role, &InputMessageRole::User);
        let [MessageContent::Text(cognition)] = content.as_slice() else {
            panic!("expected cognition batch text");
        };
        assert!(cognition.contains("Current cognition log at 2026-05-11T06:23:00Z"));

        let ModelInputItem::Message { role, content } = &items[2] else {
            panic!("expected memo batch system message third");
        };
        assert_eq!(role, &InputMessageRole::System);
        let [MessageContent::Text(memos)] = content.as_slice() else {
            panic!("expected memo batch text");
        };
        assert!(memos.contains("Held-in-mind notes at 2026-05-11T06:23:00Z"));
    }

    #[test]
    fn persistent_faculty_seed_prepends_existing_history() {
        let mut session = Session::new();
        session.push_user("history before seed");

        seed_persistent_faculty_session(&mut session, "SYSTEM", false, &[], now());

        let items = session.input().items();
        let ModelInputItem::Message { role, content } = &items[0] else {
            panic!("expected prepended system message");
        };
        assert_eq!(role, &InputMessageRole::System);
        let [MessageContent::Text(system)] = content.as_slice() else {
            panic!("expected system text");
        };
        assert_eq!(system, "SYSTEM");

        let ModelInputItem::Message { role, content } = &items[1] else {
            panic!("expected existing history after system seed");
        };
        assert_eq!(role, &InputMessageRole::User);
        let [MessageContent::Text(history)] = content.as_slice() else {
            panic!("expected history text");
        };
        assert_eq!(history, "history before seed");
    }
}
