use chrono::{DateTime, Utc};
use nuillu_types::ModuleInstanceId;
use serde::{Deserialize, Serialize};
use std::time::Duration;

/// A single (time, event) entry on the cognitive attention stream.
///
/// Per the design, the summarize module is the *only* module that may
/// produce these. Enforcement lives in the agent scheduler and a workspace
/// test, not in this struct.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AttentionStreamEvent {
    pub at: DateTime<Utc>,
    pub text: String,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AttentionStream {
    entries: Vec<AttentionStreamEvent>,
}

impl AttentionStream {
    pub fn append(&mut self, event: AttentionStreamEvent) {
        self.entries.push(event);
    }

    pub fn entries(&self) -> &[AttentionStreamEvent] {
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
pub struct AttentionStreamRecord {
    pub stream: ModuleInstanceId,
    pub entries: Vec<AttentionStreamEvent>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AgenticDeadlockMarker {
    pub at: DateTime<Utc>,
    pub idle_for: Duration,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct AttentionStreamSet {
    streams: Vec<AttentionStreamRecord>,
    agentic_deadlock_marker: Option<AgenticDeadlockMarker>,
}

impl AttentionStreamSet {
    pub fn new(
        streams: Vec<AttentionStreamRecord>,
        agentic_deadlock_marker: Option<AgenticDeadlockMarker>,
    ) -> Self {
        Self {
            streams,
            agentic_deadlock_marker,
        }
    }

    pub fn streams(&self) -> &[AttentionStreamRecord] {
        &self.streams
    }

    pub fn agentic_deadlock_marker(&self) -> Option<&AgenticDeadlockMarker> {
        self.agentic_deadlock_marker.as_ref()
    }

    pub fn compact_json(&self) -> serde_json::Value {
        if self.agentic_deadlock_marker.is_none() && self.streams.len() == 1 {
            return serde_json::json!(self.streams[0].entries);
        }
        if let Some(marker) = &self.agentic_deadlock_marker {
            return serde_json::json!({
                "streams": self.streams,
                "agentic_deadlock_marker": marker,
            });
        }
        serde_json::json!(self.streams)
    }
}
