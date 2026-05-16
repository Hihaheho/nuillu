use std::rc::Rc;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use async_trait::async_trait;
use nuillu_types::{ModelTier, ModuleInstanceId};
use serde::{Deserialize, Serialize};

use crate::ports::PortError;
use crate::rate_limit::CapabilityKind;
use crate::r#trait::ModuleBatch;

const MAX_BATCH_DEBUG_CHARS: usize = 20_000;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum RuntimeEvent {
    LlmAccessed {
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
    RateLimitDelayed {
        sequence: u64,
        owner: ModuleInstanceId,
        capability: CapabilityKind,
        delayed_for: Duration,
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
    ModuleTaskFailed {
        sequence: u64,
        owner: ModuleInstanceId,
        phase: String,
        message: String,
    },
}

#[async_trait(?Send)]
pub trait RuntimeEventSink {
    async fn on_event(&self, event: RuntimeEvent) -> Result<(), PortError>;
}

#[derive(Debug, Default)]
pub struct NoopRuntimeEventSink;

#[async_trait(?Send)]
impl RuntimeEventSink for NoopRuntimeEventSink {
    async fn on_event(&self, _event: RuntimeEvent) -> Result<(), PortError> {
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

    pub(crate) async fn llm_accessed(&self, owner: ModuleInstanceId, tier: ModelTier) {
        let call = self.next_llm_call.fetch_add(1, Ordering::Relaxed);
        self.emit(|sequence| RuntimeEvent::LlmAccessed {
            sequence,
            call,
            owner,
            tier,
        })
        .await;
    }

    pub(crate) async fn memo_updated(&self, owner: ModuleInstanceId, char_count: usize) {
        self.emit(|sequence| RuntimeEvent::MemoUpdated {
            sequence,
            owner,
            char_count,
        })
        .await;
    }

    pub(crate) async fn rate_limit_delayed(
        &self,
        owner: ModuleInstanceId,
        capability: CapabilityKind,
        delayed_for: Duration,
    ) {
        self.emit(|sequence| RuntimeEvent::RateLimitDelayed {
            sequence,
            owner,
            capability,
            delayed_for,
        })
        .await;
    }

    pub(crate) async fn module_batch_throttled(
        &self,
        owner: ModuleInstanceId,
        delayed_for: Duration,
    ) {
        self.emit(|sequence| RuntimeEvent::ModuleBatchThrottled {
            sequence,
            owner,
            delayed_for,
        })
        .await;
    }

    pub(crate) async fn module_batch_ready(&self, owner: ModuleInstanceId, batch: &ModuleBatch) {
        let batch_type = batch.type_name().to_string();
        let batch_debug = truncated_debug(batch.debug());
        self.emit(|sequence| RuntimeEvent::ModuleBatchReady {
            sequence,
            owner,
            batch_type,
            batch_debug,
        })
        .await;
    }

    pub(crate) async fn module_task_failed(
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
        })
        .await;
    }

    async fn emit(&self, build: impl FnOnce(u64) -> RuntimeEvent) {
        let sequence = self.next_sequence.fetch_add(1, Ordering::Relaxed);
        if let Err(error) = self.sink.on_event(build(sequence)).await {
            tracing::warn!(?error, "runtime event sink failed");
        }
    }
}

fn truncated_debug(debug: &str) -> String {
    let mut out = String::with_capacity(debug.len().min(MAX_BATCH_DEBUG_CHARS));
    for (index, ch) in debug.chars().enumerate() {
        if index == MAX_BATCH_DEBUG_CHARS {
            out.push_str("\n... [truncated]");
            return out;
        }
        out.push(ch);
    }
    out
}
