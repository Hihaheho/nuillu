//! `Module` trait + capability handles.
//!
//! Each module's *role* is the set of capability handles it holds. The
//! agent's design invariants are upheld by construction — a module
//! cannot perform an operation it was not granted.
//!
//! - Without an inbox capability, a module has no triggers to react to.
//! - Without [`CognitionWriter`], a module has no path to append to the
//!   cognition log.
//! - Without [`AllocationWriter`], a module cannot change resource
//!   allocation.
//! - Without [`Memo`], a module has no memo slot at all.
//! - Typed mailbox sends are owner-stamped; identities cannot be forged.
//! - LLM tier is read from allocation per-call inside [`LlmAccess`];
//!   modules don't pick tiers themselves.

pub use nuillu_blackboard::{
    CognitionLogEntryRecord, ModuleRunStatus, UtteranceProgress, UtteranceProgressState,
};

mod allocation_writer;
mod capabilities;
mod channels;
mod cognition;
mod llm;
mod memo;
mod memory_caps;
pub mod ports;
mod prompt;
mod rate_limit;
mod readers;
mod runtime_events;
mod tiers;
mod time_division;
mod r#trait;
mod utterance;

#[cfg(test)]
mod test_support;

pub use allocation_writer::AllocationWriter;
pub use capabilities::{
    AgentRuntimeControl, AllocatedModule, AllocatedModules, CapabilityProviders, HostIo,
    InternalHarnessIo, ModuleCapabilityFactory, ModuleRegisterer, ModuleRegistry,
    ModuleRegistryError,
};
pub use channels::{
    AllocationUpdated, AllocationUpdatedInbox, AllocationUpdatedMailbox, CognitionLogUpdated,
    CognitionLogUpdatedInbox, CognitionLogUpdatedMailbox, Envelope, MemoUpdated, MemoUpdatedInbox,
    MemoUpdatedMailbox, MemoryImportance, MemoryRequest, MemoryRequestInbox, MemoryRequestMailbox,
    QueryInbox, QueryMailbox, QueryRequest, ReadyItems, SelfModelInbox, SelfModelMailbox,
    SelfModelRequest, SensoryDetailRequest, SensoryDetailRequestInbox, SensoryDetailRequestMailbox,
    SensoryInput, SensoryInputInbox, SensoryInputMailbox, SpeakInbox, SpeakMailbox, SpeakRequest,
    TopicInbox, TopicMailbox, TopicRecvError,
};
pub use cognition::CognitionWriter;
pub use llm::LlmAccess;
pub use memo::Memo;
pub use memory_caps::{
    FileSearcher, MemoryCompactor, MemoryContentReader, MemoryWriter, VectorMemorySearcher,
};
pub use nuillu_types::ModuleId;
pub use ports::Embedder;
pub use prompt::format_system_prompt;
pub use rate_limit::{
    ActivitySnapshot, CapabilityKind, RateLimitConfig, RateLimitOutcome, RateLimitPolicy,
    RateLimitPolicyError, RateLimiter, RuntimePolicy, TopicKind,
};
pub use readers::{AllocationReader, BlackboardReader, CognitionLogReader, ModuleStatusReader};
pub use runtime_events::{NoopRuntimeEventSink, RuntimeEvent, RuntimeEventSink};
pub use tiers::LutumTiers;
pub use time_division::{TimeDivision, TimeDivisionBucket, TimeDivisionError};
pub use r#trait::{ActivateCx, ErasedModule, Module, ModuleBatch};
pub use utterance::UtteranceWriter;
