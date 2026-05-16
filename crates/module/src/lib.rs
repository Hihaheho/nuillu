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
//! - Without [`Memo`] or [`TypedMemo`], a module has no memo slot at all.
//!   Memo is single-issued per module construction so a module owner has one
//!   payload type.
//! - Typed mailbox sends are owner-stamped; identities cannot be forged.
//! - LLM tier is read from allocation per-call inside [`LlmAccess`];
//!   modules don't pick tiers themselves.

pub use nuillu_blackboard::{
    AgenticDeadlockMarker, CognitionLogEntryRecord, ModuleRunStatus, ModuleRunStatusRecord,
    ResourceAllocation, UtteranceProgress, UtteranceProgressState,
};

mod activation_gate;
mod allocation_writer;
mod capabilities;
mod channels;
mod cognition;
mod interoception;
mod llm;
mod memo;
mod memory_render;
mod mind_format;
mod mind_session;
pub mod ports;
mod prompt;
mod rate_limit;
mod readers;
mod runtime_events;
mod scene;
mod session_compaction;
mod tiers;
mod time_division;
mod r#trait;

#[cfg(test)]
mod test_support;

pub use activation_gate::{
    ActivationGate, ActivationGateEvent, ActivationGateRecvError, ActivationGateVote,
};
pub use allocation_writer::AllocationWriter;
pub use capabilities::{
    AgentRuntimeControl, AllocatedModule, AllocatedModules, CapabilityProviderConfig,
    CapabilityProviderPorts, CapabilityProviderRuntime, CapabilityProviders, HostIo,
    InternalHarnessIo, ModuleCapabilityFactory, ModuleDependencies, ModuleRegisterer,
    ModuleRegistry, ModuleRegistryError,
};
pub use channels::{
    AllocationUpdated, AllocationUpdatedInbox, AllocationUpdatedMailbox, AmbientSensoryEntry,
    AttentionControlRequest, AttentionControlRequestInbox, AttentionControlRequestMailbox,
    CognitionLogEvictedInbox, CognitionLogEvictedMailbox, CognitionLogUpdated,
    CognitionLogUpdatedInbox, CognitionLogUpdatedMailbox, Envelope, InteroceptiveUpdated,
    InteroceptiveUpdatedInbox, InteroceptiveUpdatedMailbox, MemoLogEvictedInbox,
    MemoLogEvictedMailbox, MemoUpdated, MemoUpdatedInbox, MemoUpdatedMailbox, MemoryImportance,
    ReadyItems, SensoryInput, SensoryInputInbox, SensoryInputMailbox, SensoryModality, TopicInbox,
    TopicMailbox, TopicRecvError,
};
pub use cognition::CognitionWriter;
pub use interoception::InteroceptiveWriter;
pub use llm::{LlmAccess, LlmLease, LlmRequestMetadata, LlmRequestSource};
pub use memo::{Memo, TypedMemo};
pub use memory_render::render_memory_for_llm;
pub use mind_format::{
    MemoryRankCounts, format_available_faculties, format_cognition_log_batch,
    format_current_attention_guidance, format_identity_memory_seed, format_memo_log_batch,
    format_memory_trace_inventory, format_stuckness, format_time_division_guidance,
    memory_rank_counts,
};
pub use mind_session::{
    push_formatted_cognition_log_batch, push_formatted_memo_log_batch,
    seed_persistent_faculty_session,
};
pub use nuillu_types::ModuleId;
pub use ports::Embedder;
pub use prompt::{format_faculty_system_prompt, format_system_prompt};
pub use rate_limit::{
    ActivitySnapshot, CapabilityKind, RateLimitConfig, RateLimitOutcome, RateLimitPolicy,
    RateLimitPolicyError, RateLimiter, RuntimePolicy, TopicKind,
};
pub use readers::{
    AllocationReader, BlackboardReader, CognitionLogReader, InteroceptiveReader,
    MemoryMetadataReader, ModuleStatusReader,
};
pub use runtime_events::{NoopRuntimeEventSink, RuntimeEvent, RuntimeEventSink};
pub use scene::{Participant, SceneReader, SceneRegistry, TARGET_EVERYONE, TARGET_SELF};
pub use session_compaction::{
    DEFAULT_SESSION_COMPACTION_INPUT_TOKEN_THRESHOLD, DEFAULT_SESSION_COMPACTION_PREFIX_RATIO,
    SessionCompactionConfig, SessionCompactionPolicy, SessionCompactionProtectedPrefix,
    SessionCompactionRuntime, compact_session, compact_session_if_needed,
    session_compaction_cutoff,
};
pub use tiers::LutumTiers;
pub use time_division::{TimeDivision, TimeDivisionBucket, TimeDivisionError};
pub use r#trait::{ActivateCx, ErasedModule, Module, ModuleBatch};
