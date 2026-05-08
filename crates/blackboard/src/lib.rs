//! Passive data hub for the agent.
//!
//! - per-module memos
//! - cognitive attention stream
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

pub use allocation::{ActivationRatio, ModuleConfig, ResourceAllocation};
pub use attention_stream::{
    AgenticDeadlockMarker, AttentionStream, AttentionStreamEvent, AttentionStreamRecord,
    AttentionStreamSet,
};
pub use command::BlackboardCommand;
pub use memory::{MemoryMetaPatch, MemoryMetadata};
pub use state::{Blackboard, BlackboardInner, MemoRecord};
