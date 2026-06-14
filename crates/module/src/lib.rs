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
    AgenticDeadlockMarker, AllocationCommand, AllocationEffectKind, AllocationEffectLevel,
    AllocationEffectPolicy, CognitionLogEntryRecord, MemoLogRecord, ModuleRunStatus,
    ModuleRunStatusRecord, ResourceAllocation, UtteranceProgress, UtteranceProgressState,
};

mod activation_gate;
mod allocation_persistence;
mod allocation_writer;
mod capabilities;
mod channels;
mod cognition;
mod dependencies;
mod interoception;
mod llm;
mod memo;
mod memory_render;
mod mind_format;
mod mind_session;
pub mod ports;
mod prompt;
mod readers;
mod runtime_events;
mod runtime_policy;
mod scene;
mod session;
mod session_compaction;
mod tiers;
mod time_division;
mod tool_text_guard;
mod tool_trace;
mod r#trait;

#[cfg(test)]
mod test_support;

pub use activation_gate::{
    ActivationGate, ActivationGateEvent, ActivationGateRecvError, ActivationGateVote,
};
pub use allocation_persistence::{
    AllocationStore, NoopAllocationStore, PersistedAllocationSnapshot,
};
pub use allocation_writer::AllocationWriter;
pub use capabilities::{
    AgentRuntimeControl, AllocatedModule, AllocatedModules, CapabilityProviderConfig,
    CapabilityProviderPorts, CapabilityProviderRuntime, CapabilityProviders, HostIo,
    InternalHarnessIo, ModuleCapabilityFactory, ModuleDependencies, ModuleRegisterer,
    ModuleRegistry, ModuleRegistryError, SelfWake, SelfWakePermitClaim,
};
pub use channels::{
    AmbientSensoryEntry, AttentionControlRequest, AttentionControlRequestInbox,
    AttentionControlRequestMailbox, CognitionLogEvictedInbox, CognitionLogEvictedMailbox,
    CognitionLogUpdated, CognitionLogUpdatedInbox, CognitionLogUpdatedMailbox, Envelope,
    InteroceptiveUpdated, InteroceptiveUpdatedInbox, InteroceptiveUpdatedMailbox,
    MemoLogEvictedInbox, MemoLogEvictedMailbox, MemoUpdated, MemoUpdatedInbox, MemoUpdatedMailbox,
    ReadyItems, SensoryInput, SensoryInputInbox, SensoryInputMailbox, SensoryModality, TopicInbox,
    TopicMailbox, TopicRecvError, WakeClaim,
};
pub use cognition::CognitionWriter;
pub use dependencies::{apply_standard_dependencies, standard_dependency_edges};
pub use interoception::InteroceptiveWriter;
pub use llm::{
    FixedTierLlmAccess, LlmAccess, LlmBatchDebug, LlmConcurrencyLimiter, LlmLease,
    LlmRequestMetadata, LlmRequestSource, current_activation_llm_request_metadata,
    with_activation_llm_request_metadata,
};
pub use memo::{Memo, TypedMemo};
pub use memory_render::render_memory_for_llm;
pub use mind_format::{
    CognitionLogBatchFormat, LlmContextWindow, MemoLogBatchFormat, MemoryRankCounts,
    compact_llm_context_text, format_available_faculties, format_bounded_cognition_log_batch,
    format_bounded_cognition_log_batch_with_format, format_bounded_memo_log_batch,
    format_bounded_memo_log_batch_with_format, format_cognition_log_batch,
    format_current_allocation_state, format_identity_memory_seed, format_memo_log_batch,
    format_memory_trace_inventory, format_new_cognition_log_entries,
    format_source_blind_memo_log_batch, format_stuckness, format_time_division_guidance,
    memory_rank_counts,
};
pub use mind_session::{
    REASONING_SYSTEM_PROMPT, format_persistent_system_seed, push_formatted_cognition_log_batch,
    push_formatted_memo_log_batch, seed_persistent_faculty_session,
};
pub use nuillu_types::ModuleId;
pub use ports::Embedder;
pub use prompt::{
    format_faculty_system_prompt, format_identity_system_prompt, format_system_prompt,
};
pub use readers::{
    AllocationReader, BlackboardReader, CognitionLogReader, InteroceptiveReader,
    MemoryMetadataReader, ModuleStatusReader,
};
pub use runtime_events::{NoopRuntimeEventSink, RuntimeEvent, RuntimeEventSink};
pub use runtime_policy::{InteroceptionRuntimePolicy, RuntimePolicy};
pub use scene::{Participant, SceneReader, SceneRegistry, TARGET_EVERYONE, TARGET_SELF};
pub use session::{
    ModuleSessionMetadata, NoopSessionStore, PersistedModelInputItem, PersistedSessionSnapshot,
    SessionAutoCompaction, SessionCheckpointError, SessionKey, SessionStore,
    ensure_persistent_session_seeded, push_persistent_identity_seed_if_absent,
};
pub use session_compaction::{
    DEFAULT_SESSION_COMPACTION_INPUT_TOKEN_THRESHOLD, DEFAULT_SESSION_COMPACTION_MAX_OUTPUT_TOKENS,
    DEFAULT_SESSION_COMPACTION_PREFIX_RATIO, SessionCompactionConfig, SessionCompactionPolicy,
    SessionCompactionProtectedPrefix, SessionCompactionRuntime, compact_session,
    compact_session_if_needed, session_compaction_cutoff,
};
pub use tiers::{LlmConcurrencyPool, LlmTierHandle, LutumTiers};
pub use time_division::{TimeDivision, TimeDivisionBucket, TimeDivisionError};
pub use tool_text_guard::{AbortOnAvailableToolNameInText, ToolNameInAssistantText};
pub use tool_trace::emit_trace_tool_calls;
pub use r#trait::{ActivateCx, ErasedModule, Module, ModuleBatch};
