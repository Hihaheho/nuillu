//! `MemoryStore` port + memory-domain capability handles.
//!
//! Memory content lives in this trait's implementations; the blackboard
//! mirrors metadata only. Modules never touch the port directly: they hold
//! one of the capability wrappers below, all of which thread blackboard
//! metadata updates alongside the storage calls.

use std::sync::Arc;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use nuillu_blackboard::{Blackboard, BlackboardCommand, MemoryMetaPatch};
use nuillu_module::ports::{Clock, PortError};
use nuillu_types::{MemoryContent, MemoryIndex, MemoryRank};

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
    async fn list_by_rank(&self, rank: MemoryRank) -> Result<Vec<MemoryRecord>, PortError>;
    async fn search(&self, q: &MemoryQuery) -> Result<Vec<MemoryRecord>, PortError>;
    async fn delete(&self, index: &MemoryIndex) -> Result<(), PortError>;
}

#[derive(Debug, Clone)]
pub struct NewMemory {
    pub content: MemoryContent,
    pub rank: MemoryRank,
    pub occurred_at: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone)]
pub struct IndexedMemory {
    pub index: MemoryIndex,
    pub content: MemoryContent,
    pub rank: MemoryRank,
    pub occurred_at: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone)]
pub struct MemoryRecord {
    pub index: MemoryIndex,
    pub content: MemoryContent,
    pub rank: MemoryRank,
    pub occurred_at: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone)]
pub struct MemoryQuery {
    pub text: String,
    pub limit: usize,
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

    async fn list_by_rank(&self, _rank: MemoryRank) -> Result<Vec<MemoryRecord>, PortError> {
        Ok(Vec::new())
    }

    async fn search(&self, _q: &MemoryQuery) -> Result<Vec<MemoryRecord>, PortError> {
        Ok(Vec::new())
    }

    async fn delete(&self, _index: &MemoryIndex) -> Result<(), PortError> {
        Ok(())
    }
}

/// Vector search over the primary memory store + automatic access patching.
#[derive(Clone)]
pub struct VectorMemorySearcher {
    primary_store: Arc<dyn MemoryStore>,
    blackboard: Blackboard,
    clock: Arc<dyn Clock>,
}

impl VectorMemorySearcher {
    pub fn new(
        primary_store: Arc<dyn MemoryStore>,
        blackboard: Blackboard,
        clock: Arc<dyn Clock>,
    ) -> Self {
        Self {
            primary_store,
            blackboard,
            clock,
        }
    }

    pub async fn search(&self, query: &str, limit: usize) -> Result<Vec<MemoryRecord>, PortError> {
        let q = MemoryQuery {
            text: query.to_owned(),
            limit,
        };
        let hits = self.primary_store.search(&q).await?;

        if !hits.is_empty() {
            for hit in &hits {
                self.blackboard
                    .apply(BlackboardCommand::UpsertMemoryMetadata {
                        index: hit.index.clone(),
                        rank_if_new: hit.rank,
                        occurred_at_if_new: hit.occurred_at,
                        decay_if_new_secs: 0,
                        now: self.clock.now(),
                        patch: MemoryMetaPatch {
                            record_access: true,
                            ..Default::default()
                        },
                    })
                    .await;
            }
        }

        Ok(hits)
    }
}

/// Content lookup by memory index without implying vector search.
#[derive(Clone)]
pub struct MemoryContentReader {
    primary_store: Arc<dyn MemoryStore>,
}

impl MemoryContentReader {
    pub fn new(primary_store: Arc<dyn MemoryStore>) -> Self {
        Self { primary_store }
    }

    pub async fn get(&self, index: &MemoryIndex) -> Result<Option<MemoryRecord>, PortError> {
        self.primary_store.get(index).await
    }
}

/// Insert a new memory entry. Mirrors metadata onto the blackboard.
#[derive(Clone)]
pub struct MemoryWriter {
    primary_store: Arc<dyn MemoryStore>,
    replicas: Vec<Arc<dyn MemoryStore>>,
    blackboard: Blackboard,
    clock: Arc<dyn Clock>,
}

impl MemoryWriter {
    pub fn new(
        primary_store: Arc<dyn MemoryStore>,
        replicas: Vec<Arc<dyn MemoryStore>>,
        blackboard: Blackboard,
        clock: Arc<dyn Clock>,
    ) -> Self {
        Self {
            primary_store,
            replicas,
            blackboard,
            clock,
        }
    }

    pub async fn insert(
        &self,
        content: String,
        rank: MemoryRank,
        decay_secs: i64,
    ) -> Result<MemoryIndex, PortError> {
        self.insert_with_occurred_at(content, rank, decay_secs, Some(self.clock.now()))
            .await
    }

    pub async fn insert_with_occurred_at(
        &self,
        content: String,
        rank: MemoryRank,
        decay_secs: i64,
        occurred_at: Option<DateTime<Utc>>,
    ) -> Result<MemoryIndex, PortError> {
        let new = NewMemory {
            content: MemoryContent::new(content.clone()),
            rank,
            occurred_at,
        };

        let index = self.primary_store.insert(new).await?;
        let indexed = IndexedMemory {
            index: index.clone(),
            content: MemoryContent::new(content),
            rank,
            occurred_at,
        };

        let replica_writes = self.replicas.iter().enumerate().map(|(replica, store)| {
            let indexed = indexed.clone();
            async move {
                if let Err(error) = store.put(indexed).await {
                    tracing::warn!(replica, ?error, "secondary memory insert failed");
                }
            }
        });
        futures::future::join_all(replica_writes).await;

        self.blackboard
            .apply(BlackboardCommand::UpsertMemoryMetadata {
                index: index.clone(),
                rank_if_new: rank,
                occurred_at_if_new: occurred_at,
                decay_if_new_secs: decay_secs,
                now: self.clock.now(),
                patch: MemoryMetaPatch {
                    rank: Some(rank),
                    occurred_at: Some(occurred_at),
                    decay_remaining_secs: Some(decay_secs),
                    ..Default::default()
                },
            })
            .await;
        Ok(index)
    }
}

/// Delete + merge memories. The compaction module is the canonical holder;
/// per design it accumulates `remember_tokens` when merging.
#[derive(Clone)]
pub struct MemoryCompactor {
    primary_store: Arc<dyn MemoryStore>,
    replicas: Vec<Arc<dyn MemoryStore>>,
    blackboard: Blackboard,
    clock: Arc<dyn Clock>,
}

impl MemoryCompactor {
    pub fn new(
        primary_store: Arc<dyn MemoryStore>,
        replicas: Vec<Arc<dyn MemoryStore>>,
        blackboard: Blackboard,
        clock: Arc<dyn Clock>,
    ) -> Self {
        Self {
            primary_store,
            replicas,
            blackboard,
            clock,
        }
    }

    /// Insert the merged entry, delete the sources, and increment
    /// `remember_tokens` on the merged entry by the count of merged sources.
    pub async fn merge(
        &self,
        sources: &[MemoryIndex],
        merged_content: String,
        merged_rank: MemoryRank,
        decay_secs: i64,
    ) -> Result<MemoryIndex, PortError> {
        let occurred_at = self
            .get_many(sources)
            .await?
            .into_iter()
            .filter_map(|record| record.occurred_at)
            .min()
            .or_else(|| Some(self.clock.now()));
        let new = NewMemory {
            content: MemoryContent::new(merged_content.clone()),
            rank: merged_rank,
            occurred_at,
        };

        let id = self.primary_store.compact(new, sources).await?;
        let indexed = IndexedMemory {
            index: id.clone(),
            content: MemoryContent::new(merged_content),
            rank: merged_rank,
            occurred_at,
        };

        let replica_writes = self.replicas.iter().enumerate().map(|(replica, store)| {
            let indexed = indexed.clone();
            async move {
                if let Err(error) = store.put_compacted(indexed, sources).await {
                    tracing::warn!(replica, ?error, "secondary memory compact failed");
                }
            }
        });
        futures::future::join_all(replica_writes).await;

        self.blackboard
            .apply(BlackboardCommand::UpsertMemoryMetadata {
                index: id.clone(),
                rank_if_new: merged_rank,
                occurred_at_if_new: occurred_at,
                decay_if_new_secs: decay_secs,
                now: self.clock.now(),
                patch: MemoryMetaPatch {
                    rank: Some(merged_rank),
                    occurred_at: Some(occurred_at),
                    decay_remaining_secs: Some(decay_secs),
                    increment_remember_tokens: Some(sources.len() as u32),
                    ..Default::default()
                },
            })
            .await;
        for src in sources {
            self.blackboard
                .apply(BlackboardCommand::RemoveMemoryMetadata { index: src.clone() })
                .await;
        }

        Ok(id)
    }

    pub async fn get(&self, index: &MemoryIndex) -> Result<Option<MemoryRecord>, PortError> {
        self.primary_store.get(index).await
    }

    pub async fn get_many(&self, indexes: &[MemoryIndex]) -> Result<Vec<MemoryRecord>, PortError> {
        let mut records = Vec::new();
        for index in indexes {
            if let Some(record) = self.primary_store.get(index).await? {
                records.push(record);
            }
        }
        Ok(records)
    }
}
