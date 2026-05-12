use chrono::{DateTime, Utc};
use nuillu_types::ModuleInstanceId;
use serde::{Deserialize, Serialize};
use std::time::Duration;

/// A single (time, event) entry on the cognition log.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CognitionLogEntry {
    pub at: DateTime<Utc>,
    pub text: String,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CognitionLog {
    entries: Vec<CognitionLogEntry>,
}

impl CognitionLog {
    pub fn append(&mut self, entry: CognitionLogEntry) {
        self.entries.push(entry);
    }

    pub fn entries(&self) -> &[CognitionLogEntry] {
        &self.entries
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CognitionLogRecord {
    pub source: ModuleInstanceId,
    pub entries: Vec<CognitionLogEntry>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CognitionLogEntryRecord {
    pub index: u64,
    pub source: ModuleInstanceId,
    pub entry: CognitionLogEntry,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AgenticDeadlockMarker {
    pub at: DateTime<Utc>,
    pub idle_for: Duration,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct CognitionLogSet {
    logs: Vec<CognitionLogRecord>,
    agentic_deadlock_marker: Option<AgenticDeadlockMarker>,
}

impl CognitionLogSet {
    pub fn new(
        logs: Vec<CognitionLogRecord>,
        agentic_deadlock_marker: Option<AgenticDeadlockMarker>,
    ) -> Self {
        Self {
            logs,
            agentic_deadlock_marker,
        }
    }

    pub fn logs(&self) -> &[CognitionLogRecord] {
        &self.logs
    }

    pub fn agentic_deadlock_marker(&self) -> Option<&AgenticDeadlockMarker> {
        self.agentic_deadlock_marker.as_ref()
    }
}
