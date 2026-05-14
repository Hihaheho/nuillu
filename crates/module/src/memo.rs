use std::marker::PhantomData;
use std::rc::Rc;

use nuillu_blackboard::{Blackboard, MemoLogRecord, TypedMemoLogRecord};
use nuillu_types::ModuleInstanceId;

use crate::ports::Clock;
use crate::runtime_events::RuntimeEventEmitter;
use crate::{MemoUpdated, MemoUpdatedMailbox};

/// Plaintext write handle for the activating module's own memo log.
///
/// The owner identity is stamped at construction time by
/// [`ModuleCapabilityFactory`](crate::ModuleCapabilityFactory); the module
/// cannot change it. A module that does not hold a memo capability has no memo
/// log allocated on the blackboard at all.
#[derive(Clone)]
pub struct Memo {
    core: MemoCore,
}

/// Typed write handle for deterministic Rust-side memo payloads.
///
/// LLM-facing memo readers still receive only plaintext content. The typed
/// payload is available to harnesses and deterministic module logic through
/// typed memo views.
#[derive(Clone)]
pub struct TypedMemo<T> {
    core: MemoCore,
    _marker: PhantomData<fn() -> T>,
}

#[derive(Clone)]
struct MemoCore {
    owner: ModuleInstanceId,
    blackboard: Blackboard,
    updates: MemoUpdatedMailbox,
    clock: Rc<dyn Clock>,
    events: RuntimeEventEmitter,
}

impl Memo {
    pub(crate) fn new(
        owner: ModuleInstanceId,
        blackboard: Blackboard,
        updates: MemoUpdatedMailbox,
        clock: Rc<dyn Clock>,
        events: RuntimeEventEmitter,
    ) -> Self {
        Self {
            core: MemoCore::new(owner, blackboard, updates, clock, events),
        }
    }

    /// Append a new plaintext owner memo item. Allocates the queue on first
    /// call.
    pub async fn write(&self, memo: impl Into<String>) -> MemoLogRecord {
        self.core.write_plain(memo.into()).await
    }
}

impl<T: 'static> TypedMemo<T> {
    pub(crate) fn new(
        owner: ModuleInstanceId,
        blackboard: Blackboard,
        updates: MemoUpdatedMailbox,
        clock: Rc<dyn Clock>,
        events: RuntimeEventEmitter,
    ) -> Self {
        Self {
            core: MemoCore::new(owner, blackboard, updates, clock, events),
            _marker: PhantomData,
        }
    }

    /// Append a new typed owner memo item plus its plaintext representation.
    pub async fn write(&self, payload: T, memo: impl Into<String>) -> MemoLogRecord {
        self.core.write_typed(payload, memo.into()).await
    }

    /// Return retained typed memo entries for this owner.
    pub async fn recent_logs(&self) -> Vec<TypedMemoLogRecord<T>> {
        self.core.blackboard.typed_memo_logs(&self.core.owner).await
    }
}

impl MemoCore {
    fn new(
        owner: ModuleInstanceId,
        blackboard: Blackboard,
        updates: MemoUpdatedMailbox,
        clock: Rc<dyn Clock>,
        events: RuntimeEventEmitter,
    ) -> Self {
        Self {
            owner,
            blackboard,
            updates,
            clock,
            events,
        }
    }

    async fn write_plain(&self, memo: String) -> MemoLogRecord {
        let char_count = memo.chars().count();
        let record = self
            .blackboard
            .update_memo(self.owner.clone(), memo, self.clock.now())
            .await;
        self.publish_update(record.index, char_count).await;
        record
    }

    async fn write_typed<T: 'static>(&self, payload: T, memo: String) -> MemoLogRecord {
        let char_count = memo.chars().count();
        let record = self
            .blackboard
            .update_typed_memo(self.owner.clone(), memo, payload, self.clock.now())
            .await;
        self.publish_update(record.index, char_count).await;
        record
    }

    async fn publish_update(&self, index: u64, char_count: usize) {
        self.events
            .memo_updated(self.owner.clone(), char_count)
            .await;
        if self
            .updates
            .publish(MemoUpdated {
                owner: self.owner.clone(),
                index,
            })
            .await
            .is_err()
        {
            tracing::trace!("memo update had no active subscribers");
        }
    }
}
