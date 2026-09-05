use std::time::Duration;

use nuillu_module::RuntimeEvent;
use nuillu_types::{ModelTier, ModuleId, ScopeId};
use serde::Serialize;

pub const EVENT_VARIANTS: &[&str] = &[
    "llm-wait-started",
    "llm-accessed",
    "llm-completed",
    "memo-updated",
    "module-batch-throttled",
    "module-batch-ready",
    "module-activation-completed",
    "module-activation-failed",
    "module-task-failed",
    "module-warning",
    "session-compaction-started",
    "session-compaction-completed",
    "session-compaction-failed",
    "stimulus-published",
    "memo-written",
    "cognition-appended",
    "utterance-completed",
];

/// Stable, scope-aware projection of runtime instrumentation used by evals.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct EvalEvent {
    pub sequence: u64,
    pub offset_ms: u64,
    pub scope: ScopeId,
    pub module: ModuleId,
    pub replica: u8,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub step: Option<String>,
    #[serde(flatten)]
    pub payload: EvalEventPayload,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "variant", rename_all = "kebab-case")]
pub enum EvalEventPayload {
    LlmWaitStarted {
        tier: ModelTier,
    },
    LlmAccessed {
        call: u64,
        tier: ModelTier,
    },
    LlmCompleted {
        call: u64,
        tier: ModelTier,
    },
    MemoUpdated {
        char_count: usize,
    },
    ModuleBatchThrottled {
        delayed_ms: u64,
    },
    ModuleBatchReady {
        batch_type: String,
    },
    ModuleActivationCompleted {
        duration_ms: u64,
        succeeded: bool,
    },
    ModuleActivationFailed {
        attempt: u32,
        max_attempts: u32,
        message: String,
    },
    ModuleTaskFailed {
        phase: String,
        message: String,
    },
    ModuleWarning {
        message: String,
    },
    SessionCompactionStarted {
        session_key: String,
        input_tokens: u64,
    },
    SessionCompactionCompleted {
        session_key: String,
        input_tokens: u64,
        before_items: usize,
        after_items: usize,
    },
    SessionCompactionFailed {
        session_key: String,
        message: String,
    },
    StimulusPublished {
        modality: String,
        direction: Option<String>,
        content: String,
        step_id: String,
    },
    MemoWritten {
        cognitive: bool,
        content: String,
    },
    CognitionAppended {
        content: String,
        origin: String,
    },
    UtteranceCompleted {
        target: String,
        content: String,
    },
}

impl EvalEventPayload {
    pub fn variant(&self) -> &'static str {
        match self {
            Self::LlmWaitStarted { .. } => "llm-wait-started",
            Self::LlmAccessed { .. } => "llm-accessed",
            Self::LlmCompleted { .. } => "llm-completed",
            Self::MemoUpdated { .. } => "memo-updated",
            Self::ModuleBatchThrottled { .. } => "module-batch-throttled",
            Self::ModuleBatchReady { .. } => "module-batch-ready",
            Self::ModuleActivationCompleted { .. } => "module-activation-completed",
            Self::ModuleActivationFailed { .. } => "module-activation-failed",
            Self::ModuleTaskFailed { .. } => "module-task-failed",
            Self::ModuleWarning { .. } => "module-warning",
            Self::SessionCompactionStarted { .. } => "session-compaction-started",
            Self::SessionCompactionCompleted { .. } => "session-compaction-completed",
            Self::SessionCompactionFailed { .. } => "session-compaction-failed",
            Self::StimulusPublished { .. } => "stimulus-published",
            Self::MemoWritten { .. } => "memo-written",
            Self::CognitionAppended { .. } => "cognition-appended",
            Self::UtteranceCompleted { .. } => "utterance-completed",
        }
    }
}

impl EvalEvent {
    /// Returns the scope of the module that originated an event, when the
    /// payload preserves origin metadata independently of the event's storage
    /// scope.
    pub fn origin_scope(&self) -> Option<&str> {
        let EvalEventPayload::CognitionAppended { origin, .. } = &self.payload else {
            return None;
        };
        Some(
            origin
                .rsplit_once('/')
                .map_or("/", |(scope, _module)| scope),
        )
    }
}

pub fn project_runtime_timeline(events: &[(u64, RuntimeEvent)]) -> Vec<EvalEvent> {
    let mut projected = events
        .iter()
        .map(|(offset_ms, event)| project_runtime_event(*offset_ms, event))
        .collect::<Vec<_>>();
    projected.sort_by_key(|event| event.sequence);
    projected
}

fn project_runtime_event(offset_ms: u64, event: &RuntimeEvent) -> EvalEvent {
    let (sequence, owner, payload) = match event {
        RuntimeEvent::LlmSemaphoreWaitStarted {
            sequence,
            owner,
            tier,
        } => (
            *sequence,
            owner,
            EvalEventPayload::LlmWaitStarted { tier: *tier },
        ),
        RuntimeEvent::LlmAccessed {
            sequence,
            call,
            owner,
            tier,
        } => (
            *sequence,
            owner,
            EvalEventPayload::LlmAccessed {
                call: *call,
                tier: *tier,
            },
        ),
        RuntimeEvent::LlmCompleted {
            sequence,
            call,
            owner,
            tier,
        } => (
            *sequence,
            owner,
            EvalEventPayload::LlmCompleted {
                call: *call,
                tier: *tier,
            },
        ),
        RuntimeEvent::MemoUpdated {
            sequence,
            owner,
            char_count,
        } => (
            *sequence,
            owner,
            EvalEventPayload::MemoUpdated {
                char_count: *char_count,
            },
        ),
        RuntimeEvent::ModuleBatchThrottled {
            sequence,
            owner,
            delayed_for,
        } => (
            *sequence,
            owner,
            EvalEventPayload::ModuleBatchThrottled {
                delayed_ms: millis(*delayed_for),
            },
        ),
        RuntimeEvent::ModuleBatchReady {
            sequence,
            owner,
            batch_type,
            ..
        } => (
            *sequence,
            owner,
            EvalEventPayload::ModuleBatchReady {
                batch_type: batch_type.clone(),
            },
        ),
        RuntimeEvent::ModuleActivationCompleted {
            sequence,
            owner,
            duration,
            succeeded,
            ..
        } => (
            *sequence,
            owner,
            EvalEventPayload::ModuleActivationCompleted {
                duration_ms: millis(*duration),
                succeeded: *succeeded,
            },
        ),
        RuntimeEvent::ModuleActivationAttemptFailed {
            sequence,
            owner,
            activation_attempt,
            max_attempts,
            message,
            ..
        } => (
            *sequence,
            owner,
            EvalEventPayload::ModuleActivationFailed {
                attempt: *activation_attempt,
                max_attempts: *max_attempts,
                message: message.clone(),
            },
        ),
        RuntimeEvent::ModuleTaskFailed {
            sequence,
            owner,
            phase,
            message,
        } => (
            *sequence,
            owner,
            EvalEventPayload::ModuleTaskFailed {
                phase: phase.clone(),
                message: message.clone(),
            },
        ),
        RuntimeEvent::ModuleWarning {
            sequence,
            owner,
            message,
        } => (
            *sequence,
            owner,
            EvalEventPayload::ModuleWarning {
                message: message.clone(),
            },
        ),
        RuntimeEvent::SessionCompactionStarted {
            sequence,
            owner,
            session_key,
            input_tokens,
            ..
        } => (
            *sequence,
            owner,
            EvalEventPayload::SessionCompactionStarted {
                session_key: session_key.clone(),
                input_tokens: *input_tokens,
            },
        ),
        RuntimeEvent::SessionCompactionCompleted {
            sequence,
            owner,
            session_key,
            input_tokens,
            before_items,
            after_items,
            ..
        } => (
            *sequence,
            owner,
            EvalEventPayload::SessionCompactionCompleted {
                session_key: session_key.clone(),
                input_tokens: *input_tokens,
                before_items: *before_items,
                after_items: *after_items,
            },
        ),
        RuntimeEvent::SessionCompactionFailed {
            sequence,
            owner,
            session_key,
            message,
            ..
        } => (
            *sequence,
            owner,
            EvalEventPayload::SessionCompactionFailed {
                session_key: session_key.clone(),
                message: message.clone(),
            },
        ),
    };
    EvalEvent {
        sequence,
        offset_ms,
        scope: owner.scope.clone(),
        module: owner.module.clone(),
        replica: owner.replica.get(),
        step: None,
        payload,
    }
}

fn millis(duration: Duration) -> u64 {
    duration.as_millis().try_into().unwrap_or(u64::MAX)
}
