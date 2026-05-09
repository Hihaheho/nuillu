//! Outbound traits used by capability handles.
//!
//! Adapters provide concrete implementations of these traits and inject
//! them into the [`CapabilityProviders`] at boot; modules should prefer the
//! capability handles that wrap them.
//!
//! All async traits use `?Send` so the agent can run on a single-threaded
//! runtime (current-thread tokio / wasm32) without requiring `Send`
//! bounds.
//!
//! [`CapabilityProviders`]: crate::CapabilityProviders

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use nuillu_blackboard::AttentionStreamEvent;
use nuillu_types::{MemoryContent, MemoryIndex, MemoryRank, ModuleInstanceId};
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

/// Text embedding provider used by vector-capable memory adapters.
#[async_trait(?Send)]
pub trait Embedder {
    fn dimensions(&self) -> usize;
    async fn embed(&self, text: &str) -> Result<Vec<f32>, PortError>;
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
    async fn append(
        &self,
        stream: ModuleInstanceId,
        event: AttentionStreamEvent,
    ) -> Result<(), PortError>;
    async fn since(
        &self,
        stream: &ModuleInstanceId,
        from: DateTime<Utc>,
    ) -> Result<Vec<AttentionStreamEvent>, PortError>;
}

/// Time source plus sleep. Indirected so tests can fully inject time —
/// `sleep_until` lets the scheduler wait for a virtual deadline without
/// blocking real time in tests.
#[async_trait(?Send)]
pub trait Clock {
    fn now(&self) -> DateTime<Utc>;

    /// Sleep until the given absolute deadline. Implementations should return
    /// immediately if the deadline is already in the past.
    async fn sleep_until(&self, deadline: DateTime<Utc>);

    /// Sleep for the given duration. Default impl computes the deadline via
    /// `now()` and delegates to `sleep_until`.
    async fn sleep_for(&self, duration: std::time::Duration) {
        let deadline = self.now() + chrono::Duration::from_std(duration).unwrap_or_default();
        self.sleep_until(deadline).await;
    }
}

/// System clock: adequate default for non-test use.
#[derive(Debug, Clone, Copy, Default)]
pub struct SystemClock;

#[async_trait(?Send)]
impl Clock for SystemClock {
    fn now(&self) -> DateTime<Utc> {
        Utc::now()
    }

    async fn sleep_until(&self, deadline: DateTime<Utc>) {
        let remaining = deadline - Utc::now();
        let Ok(duration) = remaining.to_std() else {
            return; // deadline already past or non-positive
        };
        if duration.is_zero() {
            return;
        }
        tokio::time::sleep(duration).await;
    }
}

/// Memory store that accepts every write, returns empty reads, and assigns
/// a unique synthetic [`MemoryIndex`] to each insert.
#[derive(Debug, Default)]
pub struct NoopMemoryStore;

#[async_trait(?Send)]
impl MemoryStore for NoopMemoryStore {
    async fn insert(&self, _mem: NewMemory) -> Result<MemoryIndex, PortError> {
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let id = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Ok(MemoryIndex::new(format!("noop-memory-{id}")))
    }

    async fn put(&self, _mem: IndexedMemory) -> Result<(), PortError> {
        Ok(())
    }

    async fn compact(
        &self,
        mem: NewMemory,
        _sources: &[MemoryIndex],
    ) -> Result<MemoryIndex, PortError> {
        self.insert(mem).await
    }

    async fn put_compacted(
        &self,
        _mem: IndexedMemory,
        _sources: &[MemoryIndex],
    ) -> Result<(), PortError> {
        Ok(())
    }

    async fn get(&self, _index: &MemoryIndex) -> Result<Option<MemoryRecord>, PortError> {
        Ok(None)
    }

    async fn search(&self, _q: &MemoryQuery) -> Result<Vec<MemoryRecord>, PortError> {
        Ok(Vec::new())
    }

    async fn delete(&self, _index: &MemoryIndex) -> Result<(), PortError> {
        Ok(())
    }
}

/// File search provider that returns no hits.
#[derive(Debug, Default)]
pub struct NoopFileSearchProvider;

#[async_trait(?Send)]
impl FileSearchProvider for NoopFileSearchProvider {
    async fn search(&self, _query: &FileSearchQuery) -> Result<Vec<FileSearchHit>, PortError> {
        Ok(Vec::new())
    }
}

/// Attention repository that discards appends and reports no history.
#[derive(Debug, Default)]
pub struct NoopAttentionRepository;

#[async_trait(?Send)]
impl AttentionRepository for NoopAttentionRepository {
    async fn append(
        &self,
        _stream: ModuleInstanceId,
        _event: AttentionStreamEvent,
    ) -> Result<(), PortError> {
        Ok(())
    }

    async fn since(
        &self,
        _stream: &ModuleInstanceId,
        _from: DateTime<Utc>,
    ) -> Result<Vec<AttentionStreamEvent>, PortError> {
        Ok(Vec::new())
    }
}

/// Utterance sink that drops every event.
#[derive(Debug, Default)]
pub struct NoopUtteranceSink;

#[async_trait(?Send)]
impl UtteranceSink for NoopUtteranceSink {
    async fn on_complete(&self, _utterance: Utterance) -> Result<(), PortError> {
        Ok(())
    }
}

/// A single user-visible utterance emitted by the speak module.
pub struct Utterance {
    pub sender: ModuleInstanceId,
    pub text: String,
    pub emitted_at: DateTime<Utc>,
}

/// A streaming chunk emitted while a user-visible utterance is being generated.
///
/// A generation may be interrupted and resumed. Resumed chunks keep the same
/// `generation_id` and continue the `sequence`; sinks should append them to the
/// chunks they already accepted instead of discarding partial text.
pub struct UtteranceDelta {
    pub sender: ModuleInstanceId,
    pub generation_id: u64,
    pub sequence: u32,
    pub delta: String,
}

/// Append-only sink for utterances. Adapters provide concrete implementations.
#[async_trait(?Send)]
pub trait UtteranceSink {
    async fn on_complete(&self, utterance: Utterance) -> Result<(), PortError>;

    async fn on_delta(&self, _delta: UtteranceDelta) -> Result<(), PortError> {
        Ok(())
    }
}
