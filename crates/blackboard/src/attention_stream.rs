use chrono::{DateTime, Utc};
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
