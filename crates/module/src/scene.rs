use std::sync::{Arc, RwLock};

use schemars::Schema;

/// Canonical target value for self-directed speech (soliloquy).
pub const TARGET_SELF: &str = "self";
/// Canonical target value for broadcast speech to everyone in the scene.
pub const TARGET_EVERYONE: &str = "everyone";

/// A being currently in the agent's scene.
///
/// `name` is the canonical identifier the agent uses to address the participant
/// in speech (`Utterance.target`). The host (game runtime, eval harness, etc.)
/// decides naming.
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
    inner: Arc<RwLock<SceneState>>,
}

#[derive(Clone, Debug)]
struct SceneState {
    participants: Vec<Participant>,
    include_self_target: bool,
    include_broadcast_target: bool,
}

impl SceneRegistry {
    pub fn new(initial: impl IntoIterator<Item = Participant>) -> Self {
        Self {
            inner: Arc::new(RwLock::new(SceneState {
                participants: initial.into_iter().collect(),
                include_self_target: false,
                include_broadcast_target: true,
            })),
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
        guard.participants = participants.into_iter().collect();
    }

    /// Host-controlled speech target policy. Defaults to `true`.
    ///
    /// Eval harnesses can set this to `false` when a case has exactly one
    /// valid peer target and broadcast speech would only add an invalid
    /// structured-output option.
    pub fn set_broadcast_target_enabled(&self, enabled: bool) {
        let mut guard = self
            .inner
            .write()
            .expect("scene registry lock poisoned on write");
        guard.include_broadcast_target = enabled;
    }

    /// Host-controlled speech target policy. Defaults to `false`.
    ///
    /// Self-directed speech is an internal cognition/memo concern by default;
    /// hosts that intentionally expose audible soliloquy can opt in.
    pub fn set_self_target_enabled(&self, enabled: bool) {
        let mut guard = self
            .inner
            .write()
            .expect("scene registry lock poisoned on write");
        guard.include_self_target = enabled;
    }

    pub fn snapshot(&self) -> Vec<Participant> {
        self.inner
            .read()
            .expect("scene registry lock poisoned on read")
            .participants
            .clone()
    }

    fn target_snapshot(&self) -> (Vec<Participant>, bool, bool) {
        let guard = self
            .inner
            .read()
            .expect("scene registry lock poisoned on read");
        (
            guard.participants.clone(),
            guard.include_self_target,
            guard.include_broadcast_target,
        )
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

    /// JSON Schema for a speech-target value, constrained to the current
    /// scene snapshot and host self/broadcast target policy.
    ///
    /// Callers should invoke this immediately before constructing the
    /// structured-output schema for an LLM call so the enum reflects the
    /// host's latest scene state.
    pub fn target_schema(&self) -> Schema {
        let (participants, include_self_target, include_broadcast_target) =
            self.scene.target_snapshot();
        let mut values: Vec<serde_json::Value> = Vec::with_capacity(
            participants.len()
                + usize::from(include_self_target)
                + usize::from(include_broadcast_target),
        );
        if include_self_target {
            values.push(serde_json::Value::String(TARGET_SELF.to_owned()));
        }
        if include_broadcast_target {
            values.push(serde_json::Value::String(TARGET_EVERYONE.to_owned()));
        }
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

#[cfg(test)]
mod tests {
    use super::*;

    fn schema_value(reader: &SceneReader) -> serde_json::Value {
        serde_json::to_value(reader.target_schema()).expect("target schema should serialize")
    }

    #[test]
    fn target_schema_includes_broadcast_target_by_default() {
        let scene = SceneRegistry::new([Participant::new("Pibi")]);
        let reader = SceneReader::new(scene);

        assert_eq!(
            schema_value(&reader),
            serde_json::json!({
                "type": "string",
                "enum": ["everyone", "Pibi"],
            })
        );
    }

    #[test]
    fn target_schema_can_exclude_broadcast_target() {
        let scene = SceneRegistry::new([Participant::new("Pibi")]);
        scene.set_broadcast_target_enabled(false);
        let reader = SceneReader::new(scene);

        assert_eq!(
            schema_value(&reader),
            serde_json::json!({
                "type": "string",
                "enum": ["Pibi"],
            })
        );
    }

    #[test]
    fn target_schema_includes_self_only_when_enabled() {
        let scene = SceneRegistry::new([Participant::new("Pibi")]);
        scene.set_self_target_enabled(true);
        let reader = SceneReader::new(scene);

        assert_eq!(
            schema_value(&reader),
            serde_json::json!({
                "type": "string",
                "enum": ["self", "everyone", "Pibi"],
            })
        );
    }
}
