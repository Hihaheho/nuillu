use std::any::{Any, TypeId};
use std::fmt::Debug;
use std::rc::Rc;
use std::sync::Arc;

use anyhow::{Result, anyhow};
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use lutum::{Session, Usage};
use nuillu_blackboard::{CorePolicyRecord, IdentityMemoryRecord};
use nuillu_types::{ModuleId, ModuleInstanceId};

use crate::ports::{Clock, PortError};
use crate::runtime_events::{NoopRuntimeEventSink, RuntimeEventEmitter};
use crate::session::{
    NoopSessionStore, SessionCheckpointError, SessionStore, persistent_session_metadata,
    restore_persistent_session_metadata, strip_reasoning_blocks_from_session,
};
use crate::{PersistedSessionSnapshot, SessionCompactionRuntime, compact_session};

/// Read-only context passed to `Module::activate` carrying agent-wide
/// information that is shared across all modules. **Capabilities are
/// per-module and owner-stamped — they do not belong here.** Use this for
/// agent-global state that any module may consult, such as registered peer
/// contexts used to build module prompts.
pub struct ActivateCx<'a> {
    peer_contexts: &'a [(ModuleId, Arc<str>)],
    identity_memories: &'a [IdentityMemoryRecord],
    core_policies: &'a [CorePolicyRecord],
    session_compaction: SessionCompactionRuntime,
    session_store: Rc<dyn SessionStore>,
    runtime_events: RuntimeEventEmitter,
    owner: Option<ModuleInstanceId>,
    clock: Rc<dyn Clock>,
}

impl<'a> ActivateCx<'a> {
    pub fn new(
        peer_contexts: &'a [(ModuleId, Arc<str>)],
        identity_memories: &'a [IdentityMemoryRecord],
        core_policies: &'a [CorePolicyRecord],
        session_compaction: SessionCompactionRuntime,
        clock: Rc<dyn Clock>,
    ) -> Self {
        Self {
            peer_contexts,
            identity_memories,
            core_policies,
            session_compaction,
            session_store: Rc::new(NoopSessionStore),
            runtime_events: RuntimeEventEmitter::new(Rc::new(NoopRuntimeEventSink)),
            owner: None,
            clock,
        }
    }

    pub(crate) fn with_session_checkpoint_runtime(
        mut self,
        session_store: Rc<dyn SessionStore>,
        runtime_events: RuntimeEventEmitter,
        owner: ModuleInstanceId,
    ) -> Self {
        self.session_store = session_store;
        self.runtime_events = runtime_events;
        self.owner = Some(owner);
        self
    }

    /// Stamp the activating module for runtime diagnostics such as [`Self::warn`].
    pub fn with_owner(mut self, owner: ModuleInstanceId) -> Self {
        self.owner = Some(owner);
        self
    }

    /// Emit a module-scoped activation warning to runtime observers and tracing.
    pub fn warn(&self, message: impl Into<String>) {
        let message = message.into();
        if let Some(owner) = &self.owner {
            tracing::warn!(
                owner = %owner,
                message = %message,
                "module activation warning"
            );
            self.runtime_events.module_warning(owner.clone(), message);
        } else {
            tracing::warn!(message = %message, "module activation warning");
        }
    }

    /// Peer-context entries for modules that should be described in sibling
    /// system prompts: `(id, peer_context)`.
    pub fn peer_contexts(&self) -> &[(ModuleId, Arc<str>)] {
        self.peer_contexts
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

    /// Current injected wall-clock time at the instant of this call.
    pub fn now(&self) -> DateTime<Utc> {
        self.clock.now()
    }

    pub async fn compact_and_save(
        &self,
        session: &mut Session,
        usage: Usage,
    ) -> std::result::Result<(), SessionCheckpointError> {
        let metadata = persistent_session_metadata(session)
            .cloned()
            .ok_or(SessionCheckpointError::MissingMetadata)?;
        strip_reasoning_blocks_from_session(session);
        let threshold = self.session_compaction.input_token_threshold();
        if let Some(profile) = metadata.auto_compaction
            && usage.input_tokens > threshold
        {
            let tier = self.session_compaction.module_tier();
            let before_items = session.input().items().len();
            self.runtime_events.session_compaction_started(
                metadata.owner.clone(),
                metadata.key.as_str().to_owned(),
                usage.input_tokens,
                threshold,
                tier,
            );
            let _permit = self.session_compaction.concurrency().acquire().await;
            let lutum = self
                .session_compaction
                .lutum_for_session(metadata.owner.clone(), metadata.key.as_str().to_owned());
            match compact_session(
                session,
                &lutum,
                profile.config,
                profile.protected_prefix,
                profile.compacted_prefix,
                profile.compaction_focus,
            )
            .await
            {
                Ok(()) => {
                    restore_persistent_session_metadata(session, &metadata);
                    self.runtime_events.session_compaction_completed(
                        metadata.owner.clone(),
                        metadata.key.as_str().to_owned(),
                        usage.input_tokens,
                        threshold,
                        before_items,
                        session.input().items().len(),
                        tier,
                    );
                }
                Err(error) => {
                    let message = format!("{error:#}");
                    tracing::warn!(
                        owner = %metadata.owner,
                        session_key = %metadata.key,
                        input_tokens = usage.input_tokens,
                        threshold,
                        module_tier = ?tier,
                        error = ?error,
                        "module session compaction failed"
                    );
                    self.runtime_events.session_compaction_failed(
                        metadata.owner.clone(),
                        metadata.key.as_str().to_owned(),
                        usage.input_tokens,
                        threshold,
                        tier,
                        message,
                    );
                }
            }
        }

        let snapshot = PersistedSessionSnapshot::from_session(session);
        if let Err(error) = self
            .session_store
            .save(&metadata.owner, &metadata.key, &snapshot)
            .await
        {
            let message = format_session_save_error(&metadata.owner, &metadata.key, error);
            tracing::warn!(
                owner = %metadata.owner,
                session_key = %metadata.key,
                message,
                "module session save failed"
            );
            self.runtime_events.module_task_failed(
                metadata.owner,
                "session-save".to_owned(),
                message,
            );
        }
        Ok(())
    }
}

fn format_session_save_error(
    owner: &nuillu_types::ModuleInstanceId,
    key: &crate::SessionKey,
    error: PortError,
) -> String {
    format!("failed to save session {owner}/{key}: {error}")
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

    async fn next_batch(&mut self) -> Result<Self::Batch>;
    async fn activate(&mut self, cx: &ActivateCx<'_>, batch: &Self::Batch) -> Result<()>;
}

/// Static registration metadata for built-in and other one-role module types.
///
/// Dynamic registrations do not implement identity here: callers construct a
/// [`crate::ModuleRegistrationSpec`] with the runtime [`ModuleId`] instead.
pub trait StaticModule: Module {
    /// Stable kebab-case default identifier for this module type.
    fn id() -> &'static str;

    /// Context about this module shown in peer module system prompts. Return
    /// `None` to keep this module out of peer-context catalogs.
    fn peer_context() -> Option<&'static str>;
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

#[cfg(test)]
mod tests {
    use std::cell::Cell;

    use chrono::TimeZone as _;
    use nuillu_types::ModelTier;

    use super::*;
    use crate::{LlmConcurrencyLimiter, SessionCompactionPolicy};

    struct MutableClock(Cell<DateTime<Utc>>);

    #[async_trait(?Send)]
    impl Clock for MutableClock {
        fn now(&self) -> DateTime<Utc> {
            self.0.get()
        }

        async fn sleep_until(&self, _deadline: DateTime<Utc>) {}
    }

    fn compaction_runtime() -> SessionCompactionRuntime {
        SessionCompactionRuntime::new(
            lutum::Lutum::new(
                Arc::new(lutum::MockLlmAdapter::new()),
                lutum::SharedPoolBudgetManager::new(lutum::SharedPoolBudgetOptions::default()),
            ),
            LlmConcurrencyLimiter::new(None),
            ModelTier::Cheap,
            SessionCompactionPolicy::default(),
        )
    }

    #[test]
    fn now_tracks_the_live_clock_instead_of_the_activation_start() {
        let started_at = Utc.with_ymd_and_hms(2026, 8, 24, 14, 11, 23).unwrap();
        let later = Utc.with_ymd_and_hms(2026, 8, 24, 14, 12, 3).unwrap();
        let clock = Rc::new(MutableClock(Cell::new(started_at)));
        let cx = ActivateCx::new(&[], &[], &[], compaction_runtime(), clock.clone());

        assert_eq!(cx.now(), started_at);

        clock.0.set(later);

        assert_eq!(cx.now(), later);
    }
}
