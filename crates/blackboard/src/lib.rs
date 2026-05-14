//! Passive data hub for the agent.
//!
//! - per-module memos
//! - cognition log
//! - scheduler-owned module run status
//! - current utterance stream progress
//! - memory metadata mirror (content lives in the external `MemoryStore`)
//! - boot-time identity memory snapshot
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
mod cognition_log;
mod command;
mod interoception;
mod memory;
mod policy;
mod state;

pub use allocation::{
    ActivationRatio, ActivationRatioFn, AllocationLimits, Bpm, ModuleConfig, ModulePolicy,
    RateLimitRatio, ReplicasRatio, ResourceAllocation, ZeroReplicaWindowPolicy, linear_ratio_fn,
    rate_only_ratio_fn, replicas_only_ratio_fn,
};
pub use cognition_log::{
    AgenticDeadlockMarker, CognitionLog, CognitionLogEntry, CognitionLogEntryRecord,
    CognitionLogRecord, CognitionLogSet,
};
pub use command::BlackboardCommand;
pub use interoception::{InteroceptiveMode, InteroceptivePatch, InteroceptiveState};
pub use memory::{IdentityMemoryRecord, MemoryMetaPatch, MemoryMetadata};
pub use policy::{CorePolicyRecord, PolicyMetaPatch, PolicyMetadata};
pub use state::{
    Blackboard, BlackboardInner, MemoLogRecord, ModuleRunStatus, ModuleRunStatusRecord,
    TypedMemoLogRecord, UtteranceProgress, UtteranceProgressRecord, UtteranceProgressState,
};
