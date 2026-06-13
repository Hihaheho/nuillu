use std::rc::Rc;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use nuillu_types::{ModelTier, ModuleInstanceId};
use serde::{Deserialize, Serialize};

use crate::llm::LlmBatchDebug;
use crate::ports::PortError;
use crate::r#trait::ModuleBatch;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum RuntimeEvent {
    LlmAccessed {
        sequence: u64,
        call: u64,
        owner: ModuleInstanceId,
        tier: ModelTier,
    },
    LlmCompleted {
        sequence: u64,
        call: u64,
        owner: ModuleInstanceId,
        tier: ModelTier,
    },
    MemoUpdated {
        sequence: u64,
        owner: ModuleInstanceId,
        char_count: usize,
    },
    ModuleBatchThrottled {
        sequence: u64,
        owner: ModuleInstanceId,
        delayed_for: Duration,
    },
    ModuleBatchReady {
        sequence: u64,
        owner: ModuleInstanceId,
        batch_type: String,
        batch_debug: String,
    },
    ModuleActivationCompleted {
        sequence: u64,
        owner: ModuleInstanceId,
        duration: Duration,
        succeeded: bool,
    },
    ModuleActivationAttemptFailed {
        sequence: u64,
        owner: ModuleInstanceId,
        activation_attempt: u32,
        max_attempts: u32,
        message: String,
    },
    ModuleTaskFailed {
        sequence: u64,
        owner: ModuleInstanceId,
        phase: String,
        message: String,
    },
    ModuleWarning {
        sequence: u64,
        owner: ModuleInstanceId,
        message: String,
    },
    SessionCompactionStarted {
        sequence: u64,
        owner: ModuleInstanceId,
        session_key: String,
        input_tokens: u64,
        threshold: u64,
        tier: ModelTier,
    },
    SessionCompactionCompleted {
        sequence: u64,
        owner: ModuleInstanceId,
        session_key: String,
        input_tokens: u64,
        threshold: u64,
        before_items: usize,
        after_items: usize,
        tier: ModelTier,
    },
    SessionCompactionFailed {
        sequence: u64,
        owner: ModuleInstanceId,
        session_key: String,
        input_tokens: u64,
        threshold: u64,
        tier: ModelTier,
        message: String,
    },
}

pub trait RuntimeEventSink {
    fn on_event(&self, event: RuntimeEvent) -> Result<(), PortError>;
}

#[derive(Debug, Default)]
pub struct NoopRuntimeEventSink;

impl RuntimeEventSink for NoopRuntimeEventSink {
    fn on_event(&self, _event: RuntimeEvent) -> Result<(), PortError> {
        Ok(())
    }
}

#[derive(Clone)]
pub(crate) struct RuntimeEventEmitter {
    sink: Rc<dyn RuntimeEventSink>,
    next_sequence: Arc<AtomicU64>,
    next_llm_call: Arc<AtomicU64>,
}

impl RuntimeEventEmitter {
    pub(crate) fn new(sink: Rc<dyn RuntimeEventSink>) -> Self {
        Self {
            sink,
            next_sequence: Arc::new(AtomicU64::new(0)),
            next_llm_call: Arc::new(AtomicU64::new(0)),
        }
    }

    pub(crate) fn llm_accessed(&self, owner: ModuleInstanceId, tier: ModelTier) -> u64 {
        let call = self.next_llm_call.fetch_add(1, Ordering::Relaxed);
        let owner_for_event = owner.clone();
        self.emit(|sequence| RuntimeEvent::LlmAccessed {
            sequence,
            call,
            owner: owner_for_event,
            tier,
        });
        call
    }

    pub(crate) fn llm_completed(&self, owner: ModuleInstanceId, tier: ModelTier, call: u64) {
        self.emit(|sequence| RuntimeEvent::LlmCompleted {
            sequence,
            call,
            owner,
            tier,
        });
    }

    pub(crate) fn memo_updated(&self, owner: ModuleInstanceId, char_count: usize) {
        self.emit(|sequence| RuntimeEvent::MemoUpdated {
            sequence,
            owner,
            char_count,
        });
    }

    pub(crate) fn module_batch_throttled(&self, owner: ModuleInstanceId, delayed_for: Duration) {
        self.emit(|sequence| RuntimeEvent::ModuleBatchThrottled {
            sequence,
            owner,
            delayed_for,
        });
    }

    pub(crate) fn module_batch_ready(&self, owner: ModuleInstanceId, batch: &ModuleBatch) {
        let LlmBatchDebug {
            batch_type,
            batch_debug,
        } = LlmBatchDebug::from_batch(batch);
        self.emit(|sequence| RuntimeEvent::ModuleBatchReady {
            sequence,
            owner,
            batch_type,
            batch_debug,
        });
    }

    pub(crate) fn module_activation_completed(
        &self,
        owner: ModuleInstanceId,
        duration: Duration,
        succeeded: bool,
    ) {
        self.emit(|sequence| RuntimeEvent::ModuleActivationCompleted {
            sequence,
            owner,
            duration,
            succeeded,
        });
    }

    pub(crate) fn module_activation_attempt_failed(
        &self,
        owner: ModuleInstanceId,
        activation_attempt: u32,
        max_attempts: u32,
        message: String,
    ) {
        self.emit(|sequence| RuntimeEvent::ModuleActivationAttemptFailed {
            sequence,
            owner,
            activation_attempt,
            max_attempts,
            message,
        });
    }

    pub(crate) fn module_task_failed(
        &self,
        owner: ModuleInstanceId,
        phase: String,
        message: String,
    ) {
        self.emit(|sequence| RuntimeEvent::ModuleTaskFailed {
            sequence,
            owner,
            phase,
            message,
        });
    }

    pub(crate) fn module_warning(&self, owner: ModuleInstanceId, message: String) {
        self.emit(|sequence| RuntimeEvent::ModuleWarning {
            sequence,
            owner,
            message,
        });
    }

    pub(crate) fn session_compaction_started(
        &self,
        owner: ModuleInstanceId,
        session_key: String,
        input_tokens: u64,
        threshold: u64,
        tier: ModelTier,
    ) {
        self.emit(|sequence| RuntimeEvent::SessionCompactionStarted {
            sequence,
            owner,
            session_key,
            input_tokens,
            threshold,
            tier,
        });
    }

    pub(crate) fn session_compaction_completed(
        &self,
        owner: ModuleInstanceId,
        session_key: String,
        input_tokens: u64,
        threshold: u64,
        before_items: usize,
        after_items: usize,
        tier: ModelTier,
    ) {
        self.emit(|sequence| RuntimeEvent::SessionCompactionCompleted {
            sequence,
            owner,
            session_key,
            input_tokens,
            threshold,
            before_items,
            after_items,
            tier,
        });
    }

    pub(crate) fn session_compaction_failed(
        &self,
        owner: ModuleInstanceId,
        session_key: String,
        input_tokens: u64,
        threshold: u64,
        tier: ModelTier,
        message: String,
    ) {
        self.emit(|sequence| RuntimeEvent::SessionCompactionFailed {
            sequence,
            owner,
            session_key,
            input_tokens,
            threshold,
            tier,
            message,
        });
    }

    fn emit(&self, build: impl FnOnce(u64) -> RuntimeEvent) {
        let sequence = self.next_sequence.fetch_add(1, Ordering::Relaxed);
        if let Err(error) = self.sink.on_event(build(sequence)) {
            tracing::warn!(?error, "runtime event sink failed");
        }
    }
}
