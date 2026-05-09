//! Passive data hub for the agent.
//!
//! - per-module memos
//! - cognitive attention stream
//! - scheduler-owned module run status
//! - current utterance stream progress
//! - memory metadata mirror (content lives in the external `MemoryStore`)
//! - the attention controller's `ResourceAllocation` snapshot
//!
//! The blackboard mutates only through `apply(BlackboardCommand)`. Modules
//! do not construct `BlackboardCommand` directly — they hold typed
//! [capability handles](nuillu_module) whose methods build and apply the
//! appropriate command. That is what makes module impersonation
//! impossible at the type level.
//!
//! Typed channel messages are *not* persisted on the blackboard; they are
//! transient activation signals. Durable module output belongs in memos or
//! other blackboard state.

mod allocation;
mod attention_stream;
mod command;
mod memory;
mod state;

pub use allocation::{
    ActivationRatio, ActivationRatioFn, AllocationLimits, Bpm, ModuleConfig, ModulePolicy,
    RateLimitRatio, ReplicasRatio, ResourceAllocation, linear_ratio_fn, rate_only_ratio_fn,
    replicas_only_ratio_fn,
};
pub use attention_stream::{
    AgenticDeadlockMarker, AttentionLogRecord, AttentionStream, AttentionStreamEvent,
    AttentionStreamRecord, AttentionStreamSet,
};
pub use command::BlackboardCommand;
pub use memory::{MemoryMetaPatch, MemoryMetadata};
pub use state::{
    Blackboard, BlackboardInner, MemoLogRecord, MemoRecord, ModuleRunStatus, ModuleRunStatusRecord,
    UtteranceProgress, UtteranceProgressRecord, UtteranceProgressState,
};
