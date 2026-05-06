//! Outbound traits used by capability handles.
//!
//! Adapters provide concrete implementations of these traits and inject
//! them into the [`CapabilityFactory`] at boot; modules should prefer the
//! capability handles that wrap them.
//!
//! All async traits use `?Send` so the agent can run on a single-threaded
//! runtime (current-thread tokio / wasm32) without requiring `Send`
//! bounds.
//!
//! [`CapabilityFactory`]: crate::CapabilityFactory

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use nuillu_blackboard::AttentionStreamEvent;
use nuillu_types::{MemoryContent, MemoryIndex, MemoryRank};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum PortError {
    #[error("storage backend did not find requested resource: {0}")]
    NotFound(String),
    #[error("invalid port input: {0}")]
    InvalidInput(String),
    #[error("storage backend returned invalid data: {0}")]
    InvalidData(String),
    #[error("storage backend reported: {0}")]
    Backend(String),
}

/// Content-store for memory entries. Metadata (rank, decay, access counts)
/// is mirrored on the blackboard; this trait owns durable content and any
/// adapter-local search/indexing state.
///
/// Primary stores assign [`MemoryIndex`] values through [`insert`](Self::insert).
/// Replica stores receive primary-assigned ids through [`put`](Self::put).
/// Compaction is also store-atomic: [`compact`](Self::compact) and
/// [`put_compacted`](Self::put_compacted) must create the merged record and
/// remove all sources as one backend operation.
#[async_trait(?Send)]
pub trait MemoryStore {
    async fn insert(&self, mem: NewMemory) -> Result<MemoryIndex, PortError>;
    async fn put(&self, mem: IndexedMemory) -> Result<(), PortError>;
    async fn compact(
        &self,
        mem: NewMemory,
        sources: &[MemoryIndex],
    ) -> Result<MemoryIndex, PortError>;
    async fn put_compacted(
        &self,
        mem: IndexedMemory,
        sources: &[MemoryIndex],
    ) -> Result<(), PortError>;
    async fn get(&self, index: &MemoryIndex) -> Result<Option<MemoryRecord>, PortError>;
    async fn search(&self, q: &MemoryQuery) -> Result<Vec<MemoryRecord>, PortError>;
    async fn delete(&self, index: &MemoryIndex) -> Result<(), PortError>;
}

#[derive(Debug, Clone)]
pub struct NewMemory {
    pub content: MemoryContent,
    pub rank: MemoryRank,
}

#[derive(Debug, Clone)]
pub struct IndexedMemory {
    pub index: MemoryIndex,
    pub content: MemoryContent,
    pub rank: MemoryRank,
}

#[derive(Debug, Clone)]
pub struct MemoryRecord {
    pub index: MemoryIndex,
    pub content: MemoryContent,
    pub rank: MemoryRank,
}

#[derive(Debug, Clone)]
pub struct MemoryQuery {
    pub text: String,
    pub limit: usize,
    pub filter_rank: Option<MemoryRank>,
}

/// Read-only grep-like search over application files.
#[async_trait(?Send)]
pub trait FileSearchProvider {
    async fn search(&self, query: &FileSearchQuery) -> Result<Vec<FileSearchHit>, PortError>;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FileSearchQuery {
    pub pattern: String,
    pub regex: bool,
    pub invert_match: bool,
    pub case_sensitive: bool,
    pub context: usize,
    pub max_matches: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FileSearchHit {
    pub path: String,
    pub line: usize,
    pub snippet: String,
}

/// Append-only persistence for the cognitive attention stream.
#[async_trait(?Send)]
pub trait AttentionRepository {
    async fn append(&self, event: AttentionStreamEvent) -> Result<(), PortError>;
    async fn since(&self, from: DateTime<Utc>) -> Result<Vec<AttentionStreamEvent>, PortError>;
}

/// Time source. Indirected so tests can use a mock clock.
pub trait Clock {
    fn now(&self) -> DateTime<Utc>;
}

/// System clock: adequate default for non-test use.
#[derive(Debug, Clone, Copy, Default)]
pub struct SystemClock;

impl Clock for SystemClock {
    fn now(&self) -> DateTime<Utc> {
        Utc::now()
    }
}

/// A single user-visible utterance emitted by the speak module.
pub struct Utterance {
    pub text: String,
    pub emitted_at: DateTime<Utc>,
}

/// Append-only sink for utterances. Adapters provide concrete implementations
/// (e.g. in-memory channel, persistent log).
#[async_trait(?Send)]
pub trait UtteranceSink {
    async fn append(&self, utterance: Utterance) -> Result<(), PortError>;
}
