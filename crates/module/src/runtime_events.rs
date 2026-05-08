use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use async_trait::async_trait;
use nuillu_types::{ModelTier, ModuleInstanceId};
use serde::{Deserialize, Serialize};

use crate::ports::PortError;

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
    sink: Arc<dyn RuntimeEventSink>,
    next_sequence: Arc<AtomicU64>,
    next_llm_call: Arc<AtomicU64>,
}

impl RuntimeEventEmitter {
    pub(crate) fn new(sink: Arc<dyn RuntimeEventSink>) -> Self {
        Self {
            sink,
            next_sequence: Arc::new(AtomicU64::new(0)),
            next_llm_call: Arc::new(AtomicU64::new(0)),
        }
    }

    pub(crate) async fn llm_accessed(&self, owner: ModuleInstanceId, tier: ModelTier) {
        let sequence = self.next_sequence.fetch_add(1, Ordering::Relaxed);
        let call = self.next_llm_call.fetch_add(1, Ordering::Relaxed);
        if let Err(error) = self
            .sink
            .on_event(RuntimeEvent::LlmAccessed {
                sequence,
                call,
                owner,
                tier,
            })
            .await
        {
            tracing::warn!(?error, "runtime event sink failed");
        }
    }

    pub(crate) async fn memo_updated(&self, owner: ModuleInstanceId, char_count: usize) {
        let sequence = self.next_sequence.fetch_add(1, Ordering::Relaxed);
        if let Err(error) = self
            .sink
            .on_event(RuntimeEvent::MemoUpdated {
                sequence,
                owner,
                char_count,
            })
            .await
        {
            tracing::warn!(?error, "runtime event sink failed");
        }
    }
}
