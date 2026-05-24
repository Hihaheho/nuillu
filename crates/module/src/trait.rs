use std::any::{Any, TypeId};
use std::fmt::Debug;
use std::rc::Rc;

use anyhow::{Result, anyhow};
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use nuillu_blackboard::{CorePolicyRecord, IdentityMemoryRecord};
use nuillu_types::ModuleId;

use crate::SessionCompactionRuntime;

/// Read-only context passed to `Module::activate` carrying agent-wide
/// information that is shared across all modules. **Capabilities are
/// per-module and owner-stamped — they do not belong here.** Use this for
/// agent-global state that any module may consult, such as registered peer
/// contexts and allocation hints used to build module prompts.
pub struct ActivateCx<'a> {
    peer_contexts: &'a [(ModuleId, &'static str)],
    allocation_hints: &'a [(ModuleId, &'static str)],
    identity_memories: &'a [IdentityMemoryRecord],
    core_policies: &'a [CorePolicyRecord],
    session_compaction: SessionCompactionRuntime,
    now: DateTime<Utc>,
}

impl<'a> ActivateCx<'a> {
    pub fn new(
        peer_contexts: &'a [(ModuleId, &'static str)],
        allocation_hints: &'a [(ModuleId, &'static str)],
        identity_memories: &'a [IdentityMemoryRecord],
        core_policies: &'a [CorePolicyRecord],
        session_compaction: SessionCompactionRuntime,
        now: DateTime<Utc>,
    ) -> Self {
        Self {
            peer_contexts,
            allocation_hints,
            identity_memories,
            core_policies,
            session_compaction,
            now,
        }
    }

    /// Peer-context entries for modules that should be described in sibling
    /// system prompts: `(id, peer_context)`.
    pub fn peer_contexts(&self) -> &[(ModuleId, &'static str)] {
        self.peer_contexts
    }

    /// Allocation hints for modules that the allocation controller may target:
    /// `(id, allocation_hint)`.
    pub fn allocation_hints(&self) -> &[(ModuleId, &'static str)] {
        self.allocation_hints
    }

    /// Boot-time identity memory snapshot loaded from the primary memory
    /// store before modules are activated.
    pub fn identity_memories(&self) -> &[IdentityMemoryRecord] {
        self.identity_memories
    }

    pub fn core_policies(&self) -> &[CorePolicyRecord] {
        self.core_policies
    }

    /// Runtime state for module-owned session compaction.
    pub fn session_compaction(&self) -> &SessionCompactionRuntime {
        &self.session_compaction
    }

    pub fn now(&self) -> DateTime<Utc> {
        self.now
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
    type Batch: Debug + 'static;

    /// Stable kebab-case identifier for this module type.
    fn id() -> &'static str
    where
        Self: Sized;

    /// Context about this module shown in peer module system prompts. Return
    /// `None` to keep this module out of peer-context catalogs.
    fn peer_context() -> Option<&'static str>
    where
        Self: Sized;

    /// Hint shown to the allocation controller when this module is eligible as
    /// an allocation target. Return `None` to keep this module out of the
    /// allocation target catalog.
    fn allocation_hint() -> Option<&'static str>
    where
        Self: Sized,
    {
        None
    }

    async fn next_batch(&mut self) -> Result<Self::Batch>;
    async fn activate(&mut self, cx: &ActivateCx<'_>, batch: &Self::Batch) -> Result<()>;
}

pub struct ModuleBatch {
    inner: Rc<dyn Any>,
    type_name: &'static str,
    debug: String,
}

impl ModuleBatch {
    pub(crate) fn new<T: Debug + 'static>(inner: T) -> Self {
        let debug = format!("{inner:?}");
        Self {
            inner: Rc::new(inner),
            type_name: std::any::type_name::<T>(),
            debug,
        }
    }

    pub fn type_id(&self) -> TypeId {
        self.inner.as_ref().type_id()
    }

    pub fn downcast_rc<T: 'static>(&self) -> Option<Rc<T>> {
        self.inner.clone().downcast::<T>().ok()
    }

    pub(crate) fn as_any(&self) -> &dyn Any {
        self.inner.as_ref()
    }

    pub fn type_name(&self) -> &'static str {
        self.type_name
    }

    pub fn debug(&self) -> &str {
        &self.debug
    }
}

impl Clone for ModuleBatch {
    fn clone(&self) -> Self {
        Self {
            inner: Rc::clone(&self.inner),
            type_name: self.type_name,
            debug: self.debug.clone(),
        }
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
        Module::next_batch(self).await.map(ModuleBatch::new)
    }

    async fn activate(&mut self, cx: &ActivateCx<'_>, batch: &ModuleBatch) -> Result<()> {
        let batch = batch
            .as_any()
            .downcast_ref::<M::Batch>()
            .ok_or_else(|| anyhow!("module batch type mismatch"))?;
        Module::activate(self, cx, batch).await
    }
}
