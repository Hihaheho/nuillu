//! Memory capabilities.
//!
//! Capabilities bundle storage calls with the associated blackboard-side
//! metadata mirror. Modules never touch the ports directly.

use std::sync::Arc;

use nuillu_blackboard::{Blackboard, BlackboardCommand, MemoryMetaPatch};
use nuillu_types::{MemoryContent, MemoryIndex, MemoryRank};

use crate::ports::{
    FileSearchHit, FileSearchProvider, FileSearchQuery, IndexedMemory, MemoryQuery, MemoryRecord,
    MemoryStore, NewMemory, PortError,
};

/// Vector search over the primary memory store + automatic access patching.
#[derive(Clone)]
pub struct VectorMemorySearcher {
    primary_store: Arc<dyn MemoryStore>,
    blackboard: Blackboard,
}

impl VectorMemorySearcher {
    pub(crate) fn new(primary_store: Arc<dyn MemoryStore>, blackboard: Blackboard) -> Self {
        Self {
            primary_store,
            blackboard,
        }
    }

    pub async fn search(
        &self,
        query: &str,
        limit: usize,
        filter_rank: Option<MemoryRank>,
    ) -> Result<Vec<MemoryRecord>, PortError> {
        let q = MemoryQuery {
            text: query.to_owned(),
            limit,
            filter_rank,
        };
        let hits = self.primary_store.search(&q).await?;

        if !hits.is_empty() {
            for hit in &hits {
                self.blackboard
                    .apply(BlackboardCommand::UpsertMemoryMetadata {
                        index: hit.index.clone(),
                        rank_if_new: hit.rank,
                        decay_if_new_secs: 0,
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
    pub(crate) fn new(primary_store: Arc<dyn MemoryStore>) -> Self {
        Self { primary_store }
    }

    pub async fn get(&self, index: &MemoryIndex) -> Result<Option<MemoryRecord>, PortError> {
        self.primary_store.get(index).await
    }
}

/// Read-only ripgrep-like file search capability.
#[derive(Clone)]
pub struct FileSearcher {
    search: Arc<dyn FileSearchProvider>,
}

impl FileSearcher {
    pub(crate) fn new(search: Arc<dyn FileSearchProvider>) -> Self {
        Self { search }
    }

    pub async fn search(
        &self,
        mut query: FileSearchQuery,
    ) -> Result<Vec<FileSearchHit>, PortError> {
        query.context = query.context.min(8);
        query.max_matches = query.max_matches.clamp(1, 256);
        self.search.search(&query).await
    }
}

/// Insert a new memory entry. Mirrors metadata onto the blackboard.
#[derive(Clone)]
pub struct MemoryWriter {
    primary_store: Arc<dyn MemoryStore>,
    replicas: Vec<Arc<dyn MemoryStore>>,
    blackboard: Blackboard,
}

impl MemoryWriter {
    pub(crate) fn new(
        primary_store: Arc<dyn MemoryStore>,
        replicas: Vec<Arc<dyn MemoryStore>>,
        blackboard: Blackboard,
    ) -> Self {
        Self {
            primary_store,
            replicas,
            blackboard,
        }
    }

    pub async fn insert(
        &self,
        content: String,
        rank: MemoryRank,
        decay_secs: i64,
    ) -> Result<MemoryIndex, PortError> {
        let new = NewMemory {
            content: MemoryContent::new(content.clone()),
            rank,
        };

        let index = self.primary_store.insert(new).await?;
        let indexed = IndexedMemory {
            index: index.clone(),
            content: MemoryContent::new(content),
            rank,
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
                decay_if_new_secs: decay_secs,
                patch: MemoryMetaPatch {
                    rank: Some(rank),
                    decay_remaining_secs: Some(decay_secs),
                    ..Default::default()
                },
            })
            .await;
        Ok(index)
    }
}

/// Delete + merge memories. The compaction module is the canonical
/// holder; per design it accumulates `remember_tokens` when merging.
#[derive(Clone)]
pub struct MemoryCompactor {
    primary_store: Arc<dyn MemoryStore>,
    replicas: Vec<Arc<dyn MemoryStore>>,
    blackboard: Blackboard,
}

impl MemoryCompactor {
    pub(crate) fn new(
        primary_store: Arc<dyn MemoryStore>,
        replicas: Vec<Arc<dyn MemoryStore>>,
        blackboard: Blackboard,
    ) -> Self {
        Self {
            primary_store,
            replicas,
            blackboard,
        }
    }

    /// Insert the merged entry, delete the sources, and increment
    /// `remember_tokens` on the merged entry by the count of merged
    /// sources.
    pub async fn merge(
        &self,
        sources: &[MemoryIndex],
        merged_content: String,
        merged_rank: MemoryRank,
        decay_secs: i64,
    ) -> Result<MemoryIndex, PortError> {
        let new = NewMemory {
            content: MemoryContent::new(merged_content.clone()),
            rank: merged_rank,
        };

        let id = self.primary_store.compact(new, sources).await?;
        let indexed = IndexedMemory {
            index: id.clone(),
            content: MemoryContent::new(merged_content),
            rank: merged_rank,
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
                decay_if_new_secs: decay_secs,
                patch: MemoryMetaPatch {
                    rank: Some(merged_rank),
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
