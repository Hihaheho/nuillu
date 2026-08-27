use std::rc::Rc;

use async_trait::async_trait;
use nuillu_memory::{MemoryCapabilities, MemoryNamespace};
use nuillu_module::{ActionAffordance, RuntimeEvent};
use nuillu_storage::AgentStore;
use nuillu_types::ScopeId;
use nuillu_visualizer_protocol::{EditableSceneStateView, ModuleSettingsView};

#[async_trait(?Send)]
pub trait ServerStatePort {
    async fn load_scene(
        &self,
        seed_participants: &[String],
    ) -> anyhow::Result<EditableSceneStateView>;

    async fn save_scene(&self, state: &EditableSceneStateView) -> anyhow::Result<()>;

    async fn load_module_settings(&self) -> anyhow::Result<Vec<ModuleSettingsView>>;

    async fn save_module_settings(&self, settings: &[ModuleSettingsView]) -> anyhow::Result<()>;

    async fn load_action_affordances(&self) -> anyhow::Result<Vec<ActionAffordance>>;

    async fn save_action_affordances(&self, affordances: &[ActionAffordance])
    -> anyhow::Result<()>;
}

pub trait RuntimeEventLogPort {
    fn append(&self, message: &str, event: &RuntimeEvent) -> anyhow::Result<()>;

    fn destination(&self) -> Option<String> {
        None
    }
}

#[async_trait(?Send)]
pub trait MemorySeedPort {
    async fn seed(&self, targets: &[MemorySeedTarget]) -> anyhow::Result<MemorySeedSummary>;
}

/// What a startup seed pass actually persisted.
///
/// `memories` counts distinct persisted memory indexes rather than write calls: a
/// global-namespace seed declared at a subsystem mount is written once per replica scope
/// but stores a single memory, so counting writes would over-report the seeded set.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MemorySeedSummary {
    pub memories: usize,
    pub scopes: usize,
}

#[derive(Clone)]
pub struct MemorySeedTarget {
    scope: ScopeId,
    namespace: MemoryNamespace,
    memory: MemoryCapabilities,
}

impl MemorySeedTarget {
    pub fn new(scope: ScopeId, namespace: MemoryNamespace, memory: MemoryCapabilities) -> Self {
        Self {
            scope,
            namespace,
            memory,
        }
    }

    pub fn scope(&self) -> &ScopeId {
        &self.scope
    }

    pub fn namespace(&self) -> &MemoryNamespace {
        &self.namespace
    }

    pub fn memory(&self) -> &MemoryCapabilities {
        &self.memory
    }
}

#[derive(Clone)]
pub struct ServerHostPorts {
    state: Rc<dyn ServerStatePort>,
    agent_store: Rc<dyn AgentStore>,
    runtime_event_log: Rc<dyn RuntimeEventLogPort>,
    memory_seed: Rc<dyn MemorySeedPort>,
}

impl ServerHostPorts {
    pub fn new(
        state: Rc<dyn ServerStatePort>,
        agent_store: Rc<dyn AgentStore>,
        runtime_event_log: Rc<dyn RuntimeEventLogPort>,
        memory_seed: Rc<dyn MemorySeedPort>,
    ) -> Self {
        Self {
            state,
            agent_store,
            runtime_event_log,
            memory_seed,
        }
    }

    pub(crate) fn state(&self) -> &Rc<dyn ServerStatePort> {
        &self.state
    }

    pub(crate) fn agent_store(&self) -> &Rc<dyn AgentStore> {
        &self.agent_store
    }

    pub(crate) fn runtime_event_log(&self) -> &Rc<dyn RuntimeEventLogPort> {
        &self.runtime_event_log
    }

    pub(crate) fn memory_seed(&self) -> &Rc<dyn MemorySeedPort> {
        &self.memory_seed
    }
}

#[derive(Debug, Default)]
pub struct NoopRuntimeEventLog;

impl RuntimeEventLogPort for NoopRuntimeEventLog {
    fn append(&self, _message: &str, _event: &RuntimeEvent) -> anyhow::Result<()> {
        Ok(())
    }
}

#[derive(Debug, Default)]
pub struct NoopMemorySeed;

#[async_trait(?Send)]
impl MemorySeedPort for NoopMemorySeed {
    async fn seed(&self, _targets: &[MemorySeedTarget]) -> anyhow::Result<MemorySeedSummary> {
        Ok(MemorySeedSummary::default())
    }
}
