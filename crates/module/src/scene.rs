use std::sync::{Arc, RwLock};

use schemars::Schema;

/// Canonical target value for self-directed speech (soliloquy).
pub const TARGET_SELF: &str = "self";
/// Canonical target value for broadcast speech to everyone in the scene.
pub const TARGET_EVERYONE: &str = "everyone";

/// A being currently in the agent's scene.
///
/// `name` is the canonical identifier the agent uses to address the participant
/// in speech (`SpeakRequest.target`, `Utterance.target`). The host (game runtime,
/// eval harness, etc.) decides naming.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Participant {
    pub name: String,
}

impl Participant {
    pub fn new(name: impl Into<String>) -> Self {
        Self { name: name.into() }
    }
}

impl<S: Into<String>> From<S> for Participant {
    fn from(name: S) -> Self {
        Self::new(name)
    }
}

/// Externally-driven registry of who is currently present in the scene.
///
/// The host updates this via [`SceneRegistry::set`] as participants enter or
/// leave; capabilities (currently the Utterance capability) read snapshots when
/// they need the current addressable set. The registry is shared via `Arc` and
/// guarded by an `RwLock` so the host may push from a different task than the
/// agent's `LocalSet`.
#[derive(Clone, Debug)]
pub struct SceneRegistry {
    inner: Arc<RwLock<Vec<Participant>>>,
}

impl SceneRegistry {
    pub fn new(initial: impl IntoIterator<Item = Participant>) -> Self {
        Self {
            inner: Arc::new(RwLock::new(initial.into_iter().collect())),
        }
    }

    pub fn empty() -> Self {
        Self::new(std::iter::empty())
    }

    pub fn set(&self, participants: impl IntoIterator<Item = Participant>) {
        let mut guard = self
            .inner
            .write()
            .expect("scene registry lock poisoned on write");
        *guard = participants.into_iter().collect();
    }

    pub fn snapshot(&self) -> Vec<Participant> {
        self.inner
            .read()
            .expect("scene registry lock poisoned on read")
            .clone()
    }
}

impl Default for SceneRegistry {
    fn default() -> Self {
        Self::empty()
    }
}

/// Read-only capability over the scene registry.
///
/// Modules that need to know who is in the scene — for prompt construction,
/// for constraining structured outputs to valid speech targets, etc. —
/// receive this handle. Cloning is cheap (shares the registry's `Arc`).
#[derive(Clone, Debug)]
pub struct SceneReader {
    scene: SceneRegistry,
}

impl SceneReader {
    pub(crate) fn new(scene: SceneRegistry) -> Self {
        Self { scene }
    }

    pub fn snapshot(&self) -> Vec<Participant> {
        self.scene.snapshot()
    }

    /// JSON Schema for a speech-target value, constrained to
    /// `[self, everyone, ...participants]` from the current scene snapshot.
    ///
    /// Callers should invoke this immediately before constructing the
    /// structured-output schema for an LLM call so the enum reflects the
    /// host's latest scene state.
    pub fn target_schema(&self) -> Schema {
        let participants = self.scene.snapshot();
        let mut values: Vec<serde_json::Value> = Vec::with_capacity(participants.len() + 2);
        values.push(serde_json::Value::String(TARGET_SELF.to_owned()));
        values.push(serde_json::Value::String(TARGET_EVERYONE.to_owned()));
        for participant in participants {
            values.push(serde_json::Value::String(participant.name));
        }
        Schema::try_from(serde_json::json!({
            "type": "string",
            "enum": values,
        }))
        .expect("speech target schema must be a JSON object")
    }
}
