use std::any::Any;

use anyhow::{Result, anyhow};
use async_trait::async_trait;
use lutum::Lutum;
use nuillu_blackboard::IdentityMemoryRecord;
use nuillu_types::ModuleId;

/// Read-only context passed to `Module::activate` carrying agent-wide
/// information that is shared across all modules. **Capabilities are
/// per-module and owner-stamped — they do not belong here.** Use this for
/// agent-global state that any module may consult, such as the registered
/// module catalog used to build peer-aware system prompts.
pub struct ActivateCx<'a> {
    modules: &'a [(ModuleId, &'static str)],
    identity_memories: &'a [IdentityMemoryRecord],
    session_compaction_lutum: &'a Lutum,
}

impl<'a> ActivateCx<'a> {
    pub fn new(
        modules: &'a [(ModuleId, &'static str)],
        identity_memories: &'a [IdentityMemoryRecord],
        session_compaction_lutum: &'a Lutum,
    ) -> Self {
        Self {
            modules,
            identity_memories,
            session_compaction_lutum,
        }
    }

    /// All modules registered in the agent: `(id, role_description)`.
    pub fn modules(&self) -> &[(ModuleId, &'static str)] {
        self.modules
    }

    /// Boot-time identity memory snapshot loaded from the primary memory
    /// store before modules are activated.
    pub fn identity_memories(&self) -> &[IdentityMemoryRecord] {
        self.identity_memories
    }

    /// Cheap shared LLM handle for module-owned session compaction.
    pub fn session_compaction_lutum(&self) -> &Lutum {
        self.session_compaction_lutum
    }
}

/// A cognitive or functional module.
///
/// Capabilities (including typed inboxes that yield activations) are owned
/// by the module struct, passed in at construction. Modules do not own
/// their persistent run loop: the agent event loop awaits `next_batch`,
/// then invokes `activate` with a shared reference to the returned batch
/// plus an [`ActivateCx`] for agent-global information.
///
/// `?Send` is intentional: the runtime is single-threaded
/// (`spawn_local` / `LocalSet`) and capabilities may capture non-`Send`
/// state (e.g. `Rc`-bearing `lutum::Session`).
#[async_trait(?Send)]
pub trait Module {
    type Batch: 'static;

    /// Stable kebab-case identifier for this module type.
    fn id() -> &'static str
    where
        Self: Sized;

    /// One-sentence description of this module's role, used in system prompts
    /// of peer modules so each module knows what its siblings do.
    fn role_description() -> &'static str
    where
        Self: Sized;

    async fn next_batch(&mut self) -> Result<Self::Batch>;
    async fn activate(&mut self, cx: &ActivateCx<'_>, batch: &Self::Batch) -> Result<()>;
}

pub struct ModuleBatch {
    inner: Box<dyn Any>,
}

impl ModuleBatch {
    fn new(inner: Box<dyn Any>) -> Self {
        Self { inner }
    }

    fn as_any(&self) -> &dyn Any {
        self.inner.as_ref()
    }
}

#[async_trait(?Send)]
pub trait ErasedModule {
    async fn next_batch(&mut self) -> Result<ModuleBatch>;
    async fn activate(&mut self, cx: &ActivateCx<'_>, batch: &ModuleBatch) -> Result<()>;
}

#[async_trait(?Send)]
impl<M> ErasedModule for M
where
    M: Module + 'static,
{
    async fn next_batch(&mut self) -> Result<ModuleBatch> {
        Module::next_batch(self)
            .await
            .map(|batch| ModuleBatch::new(Box::new(batch)))
    }

    async fn activate(&mut self, cx: &ActivateCx<'_>, batch: &ModuleBatch) -> Result<()> {
        let batch = batch
            .as_any()
            .downcast_ref::<M::Batch>()
            .ok_or_else(|| anyhow!("module batch type mismatch"))?;
        Module::activate(self, cx, batch).await
    }
}
