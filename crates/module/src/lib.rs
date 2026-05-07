//! `Module` trait + capability handles.
//!
//! Each module's *role* is the set of capability handles it holds. The
//! agent's design invariants are upheld by construction — a module
//! cannot perform an operation it was not granted.
//!
//! - Without an inbox capability, a module has no triggers to react to.
//! - Without [`AttentionWriter`], a module has no path to the cognitive
//!   stream.
//! - Without [`AllocationWriter`], a module cannot change resource
//!   allocation.
//! - Without [`Memo`], a module has no memo slot at all.
//! - Typed mailbox sends are owner-stamped; identities cannot be forged.
//! - LLM tier is read from allocation per-call inside [`LlmAccess`];
//!   modules don't pick tiers themselves.

mod activation;
mod allocation_writer;
mod attention;
mod channels;
mod factory;
mod llm;
mod memo;
mod memory_caps;
mod periodic;
pub mod ports;
mod readers;
mod tiers;
mod time_division;
mod r#trait;
mod utterance;

pub use activation::ActivationGate;
pub use allocation_writer::AllocationWriter;
pub use attention::AttentionWriter;
pub use channels::{
    AttentionStreamUpdated, AttentionStreamUpdatedInbox, AttentionStreamUpdatedMailbox, Envelope,
    MemoryImportance, MemoryRequest, MemoryRequestInbox, MemoryRequestMailbox, QueryInbox,
    QueryMailbox, QueryRequest, ReadyItems, SelfModelInbox, SelfModelMailbox, SelfModelRequest,
    SensoryInput, SensoryInputInbox, SensoryInputMailbox, TopicInbox, TopicMailbox, TopicRecvError,
};
pub use factory::{
    AllocatedModules, CapabilityFactory, ModuleCapabilityFactory, ModuleRegistry,
    ModuleRegistryError,
};
pub use llm::LlmAccess;
pub use memo::Memo;
pub use memory_caps::{
    FileSearcher, MemoryCompactor, MemoryContentReader, MemoryWriter, VectorMemorySearcher,
};
pub use periodic::{PeriodicActivation, PeriodicInbox, PeriodicRecvError, PeriodicTick};
pub use readers::{AllocationReader, AttentionReader, BlackboardReader};
pub use tiers::LutumTiers;
pub use time_division::{TimeDivision, TimeDivisionBucket, TimeDivisionError};
pub use r#trait::Module;
pub use utterance::UtteranceWriter;
