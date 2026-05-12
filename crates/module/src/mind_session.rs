use chrono::{DateTime, Utc};
use lutum::Session;
use nuillu_blackboard::{CognitionLogEntryRecord, IdentityMemoryRecord, MemoLogRecord};

use crate::{
    EphemeralMindContext, format_cognition_log_batch, format_ephemeral_mind_context,
    format_identity_memory_seed, format_memo_log_batch,
};

pub fn seed_persistent_faculty_session(
    session: &mut Session,
    system_prompt: impl Into<String>,
    identity_memories: &[IdentityMemoryRecord],
    now: DateTime<Utc>,
) {
    session.push_system(system_prompt);
    if let Some(seed) = format_identity_memory_seed(identity_memories, now) {
        session.push_assistant_text(seed);
    }
}

pub fn push_formatted_cognition_log_batch(
    session: &mut Session,
    records: &[CognitionLogEntryRecord],
    now: DateTime<Utc>,
) {
    if let Some(batch) = format_cognition_log_batch(records, now) {
        session.push_assistant_text(batch);
    }
}

pub fn push_formatted_memo_log_batch(
    session: &mut Session,
    records: &[MemoLogRecord],
    now: DateTime<Utc>,
) {
    if let Some(batch) = format_memo_log_batch(records, now) {
        session.push_system(batch);
    }
}

pub fn push_ephemeral_mind_context(session: &mut Session, context: EphemeralMindContext<'_>) {
    session.push_ephemeral_system(format_ephemeral_mind_context(context));
}

#[cfg(test)]
mod tests {
    use super::*;

    use chrono::TimeZone as _;
    use nuillu_blackboard::{CognitionLogEntry, MemoLogRecord};
    use nuillu_types::{MemoryContent, MemoryIndex, ReplicaIndex, builtin};

    use lutum::{AssistantInputItem, InputMessageRole, MessageContent, ModelInputItem};

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
        );
        push_formatted_memo_log_batch(
            &mut session,
            &[MemoLogRecord {
                owner,
                index: 0,
                written_at: Utc.with_ymd_and_hms(2026, 5, 11, 6, 22, 0).unwrap(),
                content: "A memo fact.".into(),
            }],
            now(),
        );

        let items = session.input().items();
        let ModelInputItem::Message { role, content } = &items[0] else {
            panic!("expected system message first");
        };
        assert_eq!(role, &InputMessageRole::System);
        let [MessageContent::Text(system)] = content.as_slice() else {
            panic!("expected system text");
        };
        assert_eq!(system, "SYSTEM");

        let ModelInputItem::Assistant(AssistantInputItem::Text(identity)) = &items[1] else {
            panic!("expected identity seed assistant text second");
        };
        assert!(identity.contains("What I already remember about myself"));

        let ModelInputItem::Assistant(AssistantInputItem::Text(cognition)) = &items[2] else {
            panic!("expected cognition batch assistant text third");
        };
        assert!(cognition.contains("My cognition at 2026-05-11T06:23:00Z"));

        let ModelInputItem::Message { role, content } = &items[3] else {
            panic!("expected memo batch system message fourth");
        };
        assert_eq!(role, &InputMessageRole::System);
        let [MessageContent::Text(memos)] = content.as_slice() else {
            panic!("expected memo batch text");
        };
        assert!(memos.contains("Held-in-mind notes at 2026-05-11T06:23:00Z"));
    }
}
