use std::fs::{self, File, OpenOptions};
use std::io::{self, Write as _};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use chrono::{DateTime, Utc};
use nuillu_module::RuntimeEvent;
use serde::Serialize;

#[derive(Debug)]
pub(crate) struct RuntimeEventLogWriter {
    path: PathBuf,
    session_id: String,
    tab_id: String,
    file: Mutex<File>,
}

impl RuntimeEventLogWriter {
    pub(crate) fn open(path: PathBuf, session_id: String, tab_id: String) -> io::Result<Self> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let file = OpenOptions::new().create(true).append(true).open(&path)?;
        Ok(Self {
            path,
            session_id,
            tab_id,
            file: Mutex::new(file),
        })
    }

    pub(crate) fn path(&self) -> &Path {
        &self.path
    }

    pub(crate) fn append(&self, message: &str, event: &RuntimeEvent) -> io::Result<()> {
        let record = runtime_event_log_record(&self.session_id, &self.tab_id, message, event);
        let mut file = self
            .file
            .lock()
            .map_err(|_| io::Error::other("runtime event log lock poisoned"))?;
        serde_json::to_writer(&mut *file, &record).map_err(io::Error::other)?;
        file.write_all(b"\n")?;
        file.flush()
    }
}

#[derive(Debug, Serialize)]
struct RuntimeEventLogRecord {
    ts: String,
    session_id: String,
    tab_id: String,
    message: String,
    event: serde_json::Value,
}

pub(crate) fn runtime_event_log_path(state_dir: &Path, session_id: &str) -> PathBuf {
    state_dir
        .join("logs")
        .join(session_id)
        .join("runtime-events.jsonl")
}

pub(crate) fn runtime_event_message(tab_id: &str, event: &RuntimeEvent) -> String {
    match event {
        RuntimeEvent::LlmSemaphoreWaitStarted { owner, tier, .. } => format!(
            "nuillu-server llm-semaphore-wait-started tab={tab_id} owner={owner} tier={tier:?}"
        ),
        RuntimeEvent::LlmAccessed {
            call, owner, tier, ..
        } => format!(
            "nuillu-server llm-accessed tab={tab_id} call={call} owner={owner} tier={tier:?}"
        ),
        RuntimeEvent::LlmCompleted {
            call, owner, tier, ..
        } => format!(
            "nuillu-server llm-completed tab={tab_id} call={call} owner={owner} tier={tier:?}"
        ),
        RuntimeEvent::MemoUpdated {
            owner, char_count, ..
        } => format!("nuillu-server memo-updated tab={tab_id} owner={owner} chars={char_count}"),
        RuntimeEvent::ModuleBatchThrottled {
            owner, delayed_for, ..
        } => format!(
            "nuillu-server module-batch-throttled tab={tab_id} owner={owner} delayed_ms={}",
            delayed_for.as_millis()
        ),
        RuntimeEvent::ModuleBatchReady {
            activation_id,
            owner,
            batch_type,
            batch_debug,
            ..
        } => format!(
            "nuillu-server module-batch-ready tab={tab_id} activation={} owner={owner} type={batch_type} chars={}",
            activation_id,
            batch_debug.chars().count()
        ),
        RuntimeEvent::ModuleActivationCompleted {
            activation_id,
            owner,
            duration,
            succeeded,
            ..
        } => format!(
            "nuillu-server module-activation-completed tab={tab_id} activation={} owner={owner} duration_ms={} succeeded={succeeded}",
            activation_id,
            duration.as_millis()
        ),
        RuntimeEvent::ModuleActivationAttemptFailed {
            activation_id,
            owner,
            activation_attempt,
            max_attempts,
            message,
            ..
        } => format!(
            "nuillu-server module-activation-attempt-failed tab={tab_id} activation={} owner={owner} attempt={activation_attempt}/{max_attempts} error={message}",
            activation_id
        ),
        RuntimeEvent::ModuleTaskFailed {
            owner,
            phase,
            message,
            ..
        } => format!(
            "nuillu-server module-task-failed tab={tab_id} owner={owner} phase={phase} error={message}"
        ),
        RuntimeEvent::ModuleWarning { owner, message, .. } => {
            format!("nuillu-server module-warning tab={tab_id} owner={owner} message={message}")
        }
        RuntimeEvent::SessionCompactionStarted {
            owner,
            session_key,
            input_tokens,
            threshold,
            ..
        } => format!(
            "nuillu-server session-compaction-started tab={tab_id} owner={owner} session={session_key} input_tokens={input_tokens} threshold={threshold}"
        ),
        RuntimeEvent::SessionCompactionCompleted {
            owner,
            session_key,
            input_tokens,
            before_items,
            after_items,
            ..
        } => format!(
            "nuillu-server session-compaction-completed tab={tab_id} owner={owner} session={session_key} input_tokens={input_tokens} items={before_items}->{after_items}"
        ),
        RuntimeEvent::SessionCompactionFailed {
            owner,
            session_key,
            input_tokens,
            message,
            ..
        } => format!(
            "nuillu-server session-compaction-failed tab={tab_id} owner={owner} session={session_key} input_tokens={input_tokens} error={message}"
        ),
    }
}

fn runtime_event_log_record(
    session_id: &str,
    tab_id: &str,
    message: &str,
    event: &RuntimeEvent,
) -> RuntimeEventLogRecord {
    runtime_event_log_record_at(Utc::now(), session_id, tab_id, message, event)
}

fn runtime_event_log_record_at(
    ts: DateTime<Utc>,
    session_id: &str,
    tab_id: &str,
    message: &str,
    event: &RuntimeEvent,
) -> RuntimeEventLogRecord {
    RuntimeEventLogRecord {
        ts: ts.to_rfc3339(),
        session_id: session_id.to_string(),
        tab_id: tab_id.to_string(),
        message: message.to_string(),
        event: serde_json::to_value(event).unwrap_or_else(|error| {
            serde_json::json!({
                "kind": "serialization_failed",
                "message": error.to_string(),
            })
        }),
    }
}

#[cfg(test)]
mod tests {
    use std::path::{Path, PathBuf};
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::Duration;

    use chrono::{TimeZone as _, Utc};
    use nuillu_module::RuntimeEvent;
    use nuillu_types::{ModelTier, ModuleActivationId, ModuleInstanceId, ReplicaIndex, builtin};

    use super::{RuntimeEventLogWriter, runtime_event_log_record_at, runtime_event_message};

    static NEXT_TEST_DIR: AtomicU64 = AtomicU64::new(0);

    #[test]
    fn runtime_event_message_formats_human_line() {
        let owner = ModuleInstanceId::new(builtin::sensory(), ReplicaIndex::ZERO);
        let event = RuntimeEvent::ModuleActivationCompleted {
            sequence: 17,
            activation_id: ModuleActivationId::new(3),
            owner,
            duration: Duration::from_millis(42),
            succeeded: true,
        };

        assert_eq!(
            runtime_event_message("server", &event),
            "nuillu-server module-activation-completed tab=server activation=3 owner=sensory duration_ms=42 succeeded=true"
        );
    }

    #[test]
    fn runtime_event_message_formats_module_warning() {
        let owner = ModuleInstanceId::new(builtin::cognition_gate(), ReplicaIndex::ZERO);
        let event = RuntimeEvent::ModuleWarning {
            sequence: 4,
            owner,
            message: "decision attempt 1/3 failed: model finished with no output".to_string(),
        };

        assert_eq!(
            runtime_event_message("server", &event),
            "nuillu-server module-warning tab=server owner=cognition-gate message=decision attempt 1/3 failed: model finished with no output"
        );
    }

    #[test]
    fn runtime_event_message_covers_session_compaction() {
        let owner = ModuleInstanceId::new(builtin::memory(), ReplicaIndex::ZERO);
        let event = RuntimeEvent::SessionCompactionStarted {
            sequence: 3,
            owner,
            session_key: "main".to_string(),
            input_tokens: 12_345,
            threshold: 8_000,
            tier: ModelTier::Cheap,
        };

        assert_eq!(
            runtime_event_message("server", &event),
            "nuillu-server session-compaction-started tab=server owner=memory session=main input_tokens=12345 threshold=8000"
        );
    }

    #[test]
    fn runtime_event_record_contains_metadata_message_and_event() {
        let owner = ModuleInstanceId::new(builtin::sensory(), ReplicaIndex::ZERO);
        let event = RuntimeEvent::LlmAccessed {
            sequence: 9,
            call: 2,
            owner,
            tier: ModelTier::Default,
        };
        let message = runtime_event_message("server", &event);
        let ts = Utc.with_ymd_and_hms(2026, 6, 2, 1, 2, 3).unwrap();
        let value = serde_json::to_value(runtime_event_log_record_at(
            ts,
            "session-1",
            "server",
            &message,
            &event,
        ))
        .unwrap();

        assert_eq!(
            value,
            serde_json::json!({
                "ts": "2026-06-02T01:02:03+00:00",
                "session_id": "session-1",
                "tab_id": "server",
                "message": "nuillu-server llm-accessed tab=server call=2 owner=sensory tier=Default",
                "event": {
                    "kind": "llm_accessed",
                    "sequence": 9,
                    "call": 2,
                    "owner": {
                        "module": "sensory",
                        "replica": 0
                    },
                    "tier": "Default"
                }
            })
        );
    }

    #[test]
    fn runtime_event_writer_appends_jsonl_records() {
        let dir = runtime_event_path_for_test();
        let path = dir.join("runtime-events.jsonl");
        let writer = RuntimeEventLogWriter::open(
            path.clone(),
            "session-1".to_string(),
            "server".to_string(),
        )
        .unwrap();
        let owner = ModuleInstanceId::new(builtin::sensory(), ReplicaIndex::ZERO);
        let first = RuntimeEvent::MemoUpdated {
            sequence: 1,
            owner: owner.clone(),
            char_count: 10,
        };
        let second = RuntimeEvent::ModuleBatchThrottled {
            sequence: 2,
            owner,
            delayed_for: Duration::from_millis(25),
        };

        writer
            .append(&runtime_event_message("server", &first), &first)
            .unwrap();
        writer
            .append(&runtime_event_message("server", &second), &second)
            .unwrap();

        let content = std::fs::read_to_string(path).unwrap();
        let lines = content.lines().collect::<Vec<_>>();
        assert_eq!(lines.len(), 2);
        let first_record = serde_json::from_str::<serde_json::Value>(lines[0]).unwrap();
        let second_record = serde_json::from_str::<serde_json::Value>(lines[1]).unwrap();
        assert_eq!(first_record["event"]["sequence"], 1);
        assert_eq!(second_record["event"]["kind"], "module_batch_throttled");
    }

    fn workspace_root() -> &'static Path {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(Path::parent)
            .expect("server crate lives under crates/server")
    }

    fn runtime_event_path_for_test() -> PathBuf {
        workspace_root()
            .join(".tmp")
            .join("server-runtime-event-log-tests")
            .join(format!(
                "{}-{}",
                std::process::id(),
                NEXT_TEST_DIR.fetch_add(1, Ordering::Relaxed)
            ))
    }
}
