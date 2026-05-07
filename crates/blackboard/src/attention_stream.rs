use chrono::{DateTime, Utc};
use nuillu_types::ModuleInstanceId;
use serde::{Deserialize, Serialize};

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

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct AttentionStreamSet {
    streams: Vec<AttentionStreamRecord>,
}

impl AttentionStreamSet {
    pub fn new(streams: Vec<AttentionStreamRecord>) -> Self {
        Self { streams }
    }

    pub fn streams(&self) -> &[AttentionStreamRecord] {
        &self.streams
    }

    pub fn compact_json(&self) -> serde_json::Value {
        if self.streams.len() == 1 {
            return serde_json::json!(self.streams[0].entries);
        }
        serde_json::json!(self.streams)
    }
}
