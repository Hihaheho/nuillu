use std::collections::HashSet;
use std::sync::{Arc, RwLock};

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::ports::PortError;
use crate::{ActionAffordancesUpdated, ActionAffordancesUpdatedMailbox};
use nuillu_types::ModuleInstanceId;

const BUILTIN_ACTION_TOOL_NAMES: &[&str] = &["activate_speak", "hold_actions", "sleep"];

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct ActionAffordance {
    pub id: String,
    pub label: String,
    pub description: String,
    pub use_when: String,
    pub effect: String,
    pub input_schema: serde_json::Value,
}

impl ActionAffordance {
    pub fn validate(&self) -> Result<(), ActionAffordanceError> {
        validate_action_tool_name(&self.id)?;
        if BUILTIN_ACTION_TOOL_NAMES.contains(&self.id.as_str()) {
            return Err(ActionAffordanceError::ReservedToolName(self.id.clone()));
        }
        if !self.input_schema.is_object() {
            return Err(ActionAffordanceError::SchemaMustBeObject(self.id.clone()));
        }
        Ok(())
    }

    pub fn tool_description(&self) -> String {
        let mut sections = Vec::new();
        let label = self.label.trim();
        if !label.is_empty() {
            sections.push(label.to_owned());
        }
        let description = self.description.trim();
        if !description.is_empty() {
            sections.push(description.to_owned());
        }
        let use_when = self.use_when.trim();
        if !use_when.is_empty() {
            sections.push(format!("Use when: {use_when}"));
        }
        let effect = self.effect.trim();
        if !effect.is_empty() {
            sections.push(format!("Effect: {effect}"));
        }
        sections.join("\n")
    }
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ActionAffordanceError {
    #[error("action id must be non-empty ASCII [a-zA-Z][a-zA-Z0-9_-]*: {0:?}")]
    InvalidToolName(String),
    #[error("action id is reserved for a built-in action: {0}")]
    ReservedToolName(String),
    #[error("action input schema must be a JSON object: {0}")]
    SchemaMustBeObject(String),
    #[error("duplicate action id: {0}")]
    DuplicateId(String),
}

fn validate_action_tool_name(value: &str) -> Result<(), ActionAffordanceError> {
    let mut chars = value.chars();
    let Some(first) = chars.next() else {
        return Err(ActionAffordanceError::InvalidToolName(value.to_owned()));
    };
    if !first.is_ascii_alphabetic() {
        return Err(ActionAffordanceError::InvalidToolName(value.to_owned()));
    }
    if chars.all(|ch| ch.is_ascii_alphanumeric() || ch == '_' || ch == '-') {
        Ok(())
    } else {
        Err(ActionAffordanceError::InvalidToolName(value.to_owned()))
    }
}

#[derive(Clone, Debug, Default)]
pub struct ActionAffordanceRegistry {
    inner: Arc<RwLock<ActionAffordanceRegistryState>>,
}

#[derive(Debug, Default)]
struct ActionAffordanceRegistryState {
    version: u64,
    affordances: Vec<ActionAffordance>,
}

impl ActionAffordanceRegistry {
    pub fn snapshot(&self) -> ActionAffordanceSnapshot {
        let guard = self
            .inner
            .read()
            .expect("action affordance registry lock poisoned on read");
        ActionAffordanceSnapshot {
            version: guard.version,
            affordances: guard.affordances.clone(),
        }
    }

    fn replace(&self, affordances: Vec<ActionAffordance>) -> u64 {
        let mut guard = self
            .inner
            .write()
            .expect("action affordance registry lock poisoned on write");
        guard.version = guard.version.wrapping_add(1);
        guard.affordances = affordances;
        guard.version
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct ActionAffordanceSnapshot {
    pub version: u64,
    pub affordances: Vec<ActionAffordance>,
}

#[derive(Clone)]
pub struct ActionAffordanceWriter {
    registry: ActionAffordanceRegistry,
    updates: ActionAffordancesUpdatedMailbox,
}

impl ActionAffordanceWriter {
    pub(crate) fn new(
        registry: ActionAffordanceRegistry,
        updates: ActionAffordancesUpdatedMailbox,
    ) -> Self {
        Self { registry, updates }
    }

    pub async fn set_all(
        &self,
        affordances: Vec<ActionAffordance>,
    ) -> Result<ActionAffordanceSnapshot, ActionAffordanceError> {
        validate_affordance_list(&affordances)?;
        let version = self.registry.replace(affordances);
        self.publish_update(version).await;
        Ok(self.registry.snapshot())
    }

    pub async fn upsert(
        &self,
        affordance: ActionAffordance,
    ) -> Result<ActionAffordanceSnapshot, ActionAffordanceError> {
        affordance.validate()?;
        let mut affordances = self.registry.snapshot().affordances;
        match affordances
            .iter_mut()
            .find(|candidate| candidate.id == affordance.id)
        {
            Some(existing) => *existing = affordance,
            None => affordances.push(affordance),
        }
        self.set_all(affordances).await
    }

    pub async fn remove(&self, id: &str) -> ActionAffordanceSnapshot {
        let affordances = self
            .registry
            .snapshot()
            .affordances
            .into_iter()
            .filter(|affordance| affordance.id != id)
            .collect::<Vec<_>>();
        let version = self.registry.replace(affordances);
        self.publish_update(version).await;
        self.registry.snapshot()
    }

    pub fn snapshot(&self) -> ActionAffordanceSnapshot {
        self.registry.snapshot()
    }

    async fn publish_update(&self, version: u64) {
        let _ = self
            .updates
            .publish(ActionAffordancesUpdated { version })
            .await;
    }
}

fn validate_affordance_list(affordances: &[ActionAffordance]) -> Result<(), ActionAffordanceError> {
    let mut seen = HashSet::new();
    for affordance in affordances {
        affordance.validate()?;
        if !seen.insert(affordance.id.as_str()) {
            return Err(ActionAffordanceError::DuplicateId(affordance.id.clone()));
        }
    }
    Ok(())
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ExternalActionInvocation {
    pub invoked_by: ModuleInstanceId,
    pub action_id: String,
    pub arguments: serde_json::Value,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct ExternalActionInvocationResult {
    pub accepted: bool,
    pub message: String,
}

#[async_trait(?Send)]
pub trait ExternalActionExecutor {
    async fn invoke(
        &self,
        invocation: ExternalActionInvocation,
    ) -> Result<ExternalActionInvocationResult, PortError>;
}

#[derive(Debug, Default)]
pub struct NoopExternalActionExecutor;

#[async_trait(?Send)]
impl ExternalActionExecutor for NoopExternalActionExecutor {
    async fn invoke(
        &self,
        invocation: ExternalActionInvocation,
    ) -> Result<ExternalActionInvocationResult, PortError> {
        Ok(ExternalActionInvocationResult {
            accepted: false,
            message: format!("no external action handler for {}", invocation.action_id),
        })
    }
}

#[derive(Clone)]
pub struct ExternalActionInvoker {
    owner: ModuleInstanceId,
    executor: std::rc::Rc<dyn ExternalActionExecutor>,
}

impl ExternalActionInvoker {
    pub(crate) fn new(
        owner: ModuleInstanceId,
        executor: std::rc::Rc<dyn ExternalActionExecutor>,
    ) -> Self {
        Self { owner, executor }
    }

    pub async fn invoke(
        &self,
        action_id: String,
        arguments: serde_json::Value,
    ) -> Result<ExternalActionInvocationResult, PortError> {
        self.executor
            .invoke(ExternalActionInvocation {
                invoked_by: self.owner.clone(),
                action_id,
                arguments,
            })
            .await
    }
}

#[derive(Clone, Debug)]
pub struct ActionAffordanceReader {
    registry: ActionAffordanceRegistry,
}

impl ActionAffordanceReader {
    pub(crate) fn new(registry: ActionAffordanceRegistry) -> Self {
        Self { registry }
    }

    pub fn snapshot(&self) -> ActionAffordanceSnapshot {
        self.registry.snapshot()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{TopicMailbox, channels::Topic};
    use nuillu_blackboard::Blackboard;
    use nuillu_types::{ModuleId, ReplicaIndex};

    fn affordance(id: &str) -> ActionAffordance {
        ActionAffordance {
            id: id.to_owned(),
            label: "Action".to_owned(),
            description: "Do a thing".to_owned(),
            use_when: "when useful".to_owned(),
            effect: "a thing happens".to_owned(),
            input_schema: serde_json::json!({"type": "object"}),
        }
    }

    #[test]
    fn validation_rejects_invalid_reserved_and_duplicate_ids() {
        let mut invalid = affordance("not ok");
        assert!(matches!(
            invalid.validate(),
            Err(ActionAffordanceError::InvalidToolName(_))
        ));

        invalid.id = "sleep".to_owned();
        assert!(matches!(
            invalid.validate(),
            Err(ActionAffordanceError::ReservedToolName(_))
        ));

        let duplicate = vec![affordance("poet"), affordance("poet")];
        assert!(matches!(
            validate_affordance_list(&duplicate),
            Err(ActionAffordanceError::DuplicateId(_))
        ));
    }

    #[test]
    fn validation_requires_object_schema() {
        let mut affordance = affordance("poet");
        affordance.input_schema = serde_json::json!(true);
        assert_eq!(
            affordance.validate(),
            Err(ActionAffordanceError::SchemaMustBeObject("poet".to_owned()))
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn writer_persists_snapshot_order_and_version() {
        let registry = ActionAffordanceRegistry::default();
        let wakes = crate::channels::WakeRegistry::default();
        let topic = Topic::new(
            Blackboard::default(),
            wakes,
            crate::channels::TopicPolicy::Fanout,
        );
        let owner = ModuleInstanceId::new(ModuleId::new("host").unwrap(), ReplicaIndex::ZERO);
        let writer = ActionAffordanceWriter::new(registry, TopicMailbox::new(owner, topic));

        let snapshot = writer.set_all(vec![affordance("poet")]).await.unwrap();

        assert_eq!(snapshot.version, 1);
        assert_eq!(snapshot.affordances, vec![affordance("poet")]);
        assert_eq!(writer.snapshot(), snapshot);
    }
}
