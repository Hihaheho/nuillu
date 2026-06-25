use std::marker::PhantomData;
use std::rc::Rc;

use nuillu_blackboard::{Blackboard, MemoLogPayload, MemoLogRecord, TypedMemoLogRecord};
use nuillu_types::ModuleInstanceId;
use serde::{Serialize, de::DeserializeOwned};

use crate::ports::Clock;
use crate::runtime_events::RuntimeEventEmitter;
use crate::{
    MemoLogEvictedMailbox, MemoLogRepository, MemoUpdated, MemoUpdatedMailbox,
    PersistedMemoLogEntry,
};

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
    memo_log_repository: Rc<dyn MemoLogRepository>,
    updates: MemoUpdatedMailbox,
    evictions: MemoLogEvictedMailbox,
    clock: Rc<dyn Clock>,
    events: RuntimeEventEmitter,
}

impl Memo {
    pub(crate) fn new(
        owner: ModuleInstanceId,
        blackboard: Blackboard,
        memo_log_repository: Rc<dyn MemoLogRepository>,
        updates: MemoUpdatedMailbox,
        evictions: MemoLogEvictedMailbox,
        clock: Rc<dyn Clock>,
        events: RuntimeEventEmitter,
    ) -> Self {
        Self {
            core: MemoCore::new(
                owner,
                blackboard,
                memo_log_repository,
                updates,
                evictions,
                clock,
                events,
            ),
        }
    }

    /// Append a new plaintext owner memo item. Allocates the queue on first
    /// call.
    pub async fn write(&self, memo: impl Into<String>) -> MemoLogRecord {
        self.core.write_plain(memo.into(), false).await
    }

    /// Append a plaintext owner memo item that is eligible for cognition-gate
    /// promotion.
    pub async fn write_cognitive(&self, memo: impl Into<String>) -> MemoLogRecord {
        self.core.write_plain(memo.into(), true).await
    }
}

impl<T> TypedMemo<T>
where
    T: Serialize + DeserializeOwned + 'static,
{
    pub(crate) fn new(
        owner: ModuleInstanceId,
        blackboard: Blackboard,
        memo_log_repository: Rc<dyn MemoLogRepository>,
        updates: MemoUpdatedMailbox,
        evictions: MemoLogEvictedMailbox,
        clock: Rc<dyn Clock>,
        events: RuntimeEventEmitter,
    ) -> Self {
        Self {
            core: MemoCore::new(
                owner,
                blackboard,
                memo_log_repository,
                updates,
                evictions,
                clock,
                events,
            ),
            _marker: PhantomData,
        }
    }

    /// Append a new typed owner memo item plus its plaintext representation.
    pub async fn write(&self, payload: T, memo: impl Into<String>) -> MemoLogRecord {
        self.core.write_typed(payload, memo.into(), false).await
    }

    /// Append a typed owner memo item whose plaintext representation is
    /// eligible for cognition-gate promotion.
    pub async fn write_cognitive(&self, payload: T, memo: impl Into<String>) -> MemoLogRecord {
        self.core.write_typed(payload, memo.into(), true).await
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
        memo_log_repository: Rc<dyn MemoLogRepository>,
        updates: MemoUpdatedMailbox,
        evictions: MemoLogEvictedMailbox,
        clock: Rc<dyn Clock>,
        events: RuntimeEventEmitter,
    ) -> Self {
        Self {
            owner,
            blackboard,
            memo_log_repository,
            updates,
            evictions,
            clock,
            events,
        }
    }

    async fn write_plain(&self, memo: String, cognitive: bool) -> MemoLogRecord {
        let char_count = memo.chars().count();
        let result = if cognitive {
            self.blackboard
                .update_cognitive_memo_with_evictions(self.owner.clone(), memo, self.clock.now())
                .await
        } else {
            self.blackboard
                .update_memo_with_evictions(self.owner.clone(), memo, self.clock.now())
                .await
        };
        self.persist_memo(&result.record, MemoLogPayload::Plain)
            .await;
        self.publish_update(result.record.index, char_count).await;
        self.publish_evictions(result.evicted).await;
        result.record
    }

    async fn write_typed<T: Serialize + 'static>(
        &self,
        payload: T,
        memo: String,
        cognitive: bool,
    ) -> MemoLogRecord {
        let char_count = memo.chars().count();
        let persisted_payload = match serde_json::to_value(&payload) {
            Ok(json) => MemoLogPayload::Typed {
                type_name: std::any::type_name::<T>().to_owned(),
                json,
            },
            Err(error) => {
                tracing::warn!(
                    owner = %self.owner,
                    payload_type = std::any::type_name::<T>(),
                    error = %error,
                    "typed memo payload serialization failed; persisting plaintext memo only"
                );
                MemoLogPayload::Plain
            }
        };
        let result = if cognitive {
            self.blackboard
                .update_typed_cognitive_memo_with_evictions(
                    self.owner.clone(),
                    memo,
                    payload,
                    self.clock.now(),
                )
                .await
        } else {
            self.blackboard
                .update_typed_memo_with_evictions(
                    self.owner.clone(),
                    memo,
                    payload,
                    self.clock.now(),
                )
                .await
        };
        self.persist_memo(&result.record, persisted_payload).await;
        self.publish_update(result.record.index, char_count).await;
        self.publish_evictions(result.evicted).await;
        result.record
    }

    async fn persist_memo(&self, record: &MemoLogRecord, payload: MemoLogPayload) {
        let entry = PersistedMemoLogEntry {
            record: record.clone(),
            payload,
        };
        if let Err(error) = self.memo_log_repository.append(&entry).await {
            tracing::warn!(
                owner = %record.owner,
                index = record.index,
                error = ?error,
                "memo log repository append failed"
            );
        }
    }

    async fn publish_update(&self, index: u64, char_count: usize) {
        self.events.memo_updated(self.owner.clone(), char_count);
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

    async fn publish_evictions(&self, evicted: Vec<MemoLogRecord>) {
        for record in evicted {
            if self.evictions.publish(record).await.is_err() {
                tracing::trace!("memo-log eviction had no active subscribers");
            }
        }
    }
}
