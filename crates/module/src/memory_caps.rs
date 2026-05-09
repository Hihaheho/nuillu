//! Memory capabilities.
//!
//! Capabilities bundle storage calls with the associated blackboard-side
//! metadata mirror. Modules never touch the ports directly.

use std::sync::Arc;

use nuillu_blackboard::{Blackboard, BlackboardCommand, MemoryMetaPatch};
use nuillu_types::{MemoryContent, MemoryIndex, MemoryRank};

use crate::ports::{
    Clock, FileSearchHit, FileSearchProvider, FileSearchQuery, IndexedMemory, MemoryQuery,
    MemoryRecord, MemoryStore, NewMemory, PortError,
};

/// Vector search over the primary memory store + automatic access patching.
#[derive(Clone)]
pub struct VectorMemorySearcher {
    primary_store: Arc<dyn MemoryStore>,
    blackboard: Blackboard,
    clock: Arc<dyn Clock>,
}

impl VectorMemorySearcher {
    pub(crate) fn new(
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
    clock: Arc<dyn Clock>,
}

impl MemoryWriter {
    pub(crate) fn new(
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
                now: self.clock.now(),
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
    clock: Arc<dyn Clock>,
}

impl MemoryCompactor {
    pub(crate) fn new(
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
                now: self.clock.now(),
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

#[cfg(test)]
mod tests {
    use super::*;

    use async_trait::async_trait;
    use chrono::Utc;
    use tokio::sync::Mutex;

    use crate::ports::NoopFileSearchProvider;
    use crate::test_support::test_caps_with_stores;

    #[derive(Clone)]
    struct RecordingMemoryStore {
        state: Arc<Mutex<RecordingMemoryStoreState>>,
        generated_index: MemoryIndex,
        next_index: Arc<std::sync::atomic::AtomicU64>,
        search_hits: Vec<MemoryRecord>,
        fail_insert: bool,
        fail_put: bool,
        fail_compact: bool,
    }

    impl Default for RecordingMemoryStore {
        fn default() -> Self {
            Self {
                state: Arc::new(Mutex::new(RecordingMemoryStoreState::default())),
                generated_index: MemoryIndex::new("generated"),
                next_index: Arc::new(std::sync::atomic::AtomicU64::new(0)),
                search_hits: Vec::new(),
                fail_insert: false,
                fail_put: false,
                fail_compact: false,
            }
        }
    }

    #[derive(Default)]
    struct RecordingMemoryStoreState {
        inserted_indexes: Vec<MemoryIndex>,
        put_indexes: Vec<MemoryIndex>,
        compacted_indexes: Vec<MemoryIndex>,
        compacted_sources: Vec<MemoryIndex>,
    }

    impl RecordingMemoryStore {
        async fn inserted_indexes(&self) -> Vec<MemoryIndex> {
            self.state.lock().await.inserted_indexes.clone()
        }

        async fn put_indexes(&self) -> Vec<MemoryIndex> {
            self.state.lock().await.put_indexes.clone()
        }

        async fn compacted_indexes(&self) -> Vec<MemoryIndex> {
            self.state.lock().await.compacted_indexes.clone()
        }

        async fn compacted_sources(&self) -> Vec<MemoryIndex> {
            self.state.lock().await.compacted_sources.clone()
        }

        fn next_generated_index(&self) -> MemoryIndex {
            if self.generated_index.as_str() != "generated" {
                return self.generated_index.clone();
            }
            let id = self
                .next_index
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            MemoryIndex::new(format!("generated-{id}"))
        }
    }

    #[async_trait(?Send)]
    impl MemoryStore for RecordingMemoryStore {
        async fn insert(&self, _mem: NewMemory) -> Result<MemoryIndex, PortError> {
            if self.fail_insert {
                return Err(PortError::Backend("insert failed".into()));
            }
            let index = self.next_generated_index();
            self.state.lock().await.inserted_indexes.push(index.clone());
            Ok(index)
        }

        async fn put(&self, mem: IndexedMemory) -> Result<(), PortError> {
            if self.fail_put {
                return Err(PortError::Backend("put failed".into()));
            }
            self.state.lock().await.put_indexes.push(mem.index);
            Ok(())
        }

        async fn compact(
            &self,
            _mem: NewMemory,
            sources: &[MemoryIndex],
        ) -> Result<MemoryIndex, PortError> {
            if self.fail_compact {
                return Err(PortError::Backend("compact failed".into()));
            }
            let index = self.next_generated_index();
            let mut state = self.state.lock().await;
            state.compacted_indexes.push(index.clone());
            state.compacted_sources.extend_from_slice(sources);
            Ok(index)
        }

        async fn put_compacted(
            &self,
            mem: IndexedMemory,
            sources: &[MemoryIndex],
        ) -> Result<(), PortError> {
            if self.fail_put {
                return Err(PortError::Backend("put compacted failed".into()));
            }
            let mut state = self.state.lock().await;
            state.compacted_indexes.push(mem.index);
            state.compacted_sources.extend_from_slice(sources);
            Ok(())
        }

        async fn get(&self, index: &MemoryIndex) -> Result<Option<MemoryRecord>, PortError> {
            Ok(self
                .search_hits
                .iter()
                .find(|record| &record.index == index)
                .cloned())
        }

        async fn list_by_rank(&self, rank: MemoryRank) -> Result<Vec<MemoryRecord>, PortError> {
            Ok(self
                .search_hits
                .iter()
                .filter(|record| record.rank == rank)
                .cloned()
                .collect())
        }

        async fn search(&self, _q: &MemoryQuery) -> Result<Vec<MemoryRecord>, PortError> {
            Ok(self.search_hits.clone())
        }

        async fn delete(&self, _index: &MemoryIndex) -> Result<(), PortError> {
            Ok(())
        }
    }

    #[derive(Clone, Default)]
    struct RecordingFileSearchProvider {
        last_query: Arc<Mutex<Option<FileSearchQuery>>>,
    }

    impl RecordingFileSearchProvider {
        async fn last_query(&self) -> Option<FileSearchQuery> {
            self.last_query.lock().await.clone()
        }
    }

    #[async_trait(?Send)]
    impl FileSearchProvider for RecordingFileSearchProvider {
        async fn search(&self, query: &FileSearchQuery) -> Result<Vec<FileSearchHit>, PortError> {
            *self.last_query.lock().await = Some(query.clone());
            Ok(Vec::new())
        }
    }

    #[tokio::test]
    async fn file_searcher_exposes_only_rg_like_controls_and_clamps_bounds() {
        let file_search = RecordingFileSearchProvider::default();
        let caps = test_caps_with_stores(
            Blackboard::default(),
            Arc::new(crate::ports::NoopMemoryStore),
            Vec::new(),
            Arc::new(file_search.clone()),
        );

        caps.file_searcher()
            .search(FileSearchQuery {
                pattern: "fn main".into(),
                regex: true,
                invert_match: true,
                case_sensitive: false,
                context: 100,
                max_matches: 0,
            })
            .await
            .expect("file search succeeds");

        assert_eq!(
            file_search.last_query().await,
            Some(FileSearchQuery {
                pattern: "fn main".into(),
                regex: true,
                invert_match: true,
                case_sensitive: false,
                context: 8,
                max_matches: 1,
            })
        );
    }

    #[tokio::test]
    async fn memory_writer_fans_out_and_writes_single_metadata_after_primary_success() {
        let blackboard = Blackboard::default();
        let index = MemoryIndex::new("fanout");
        let primary = RecordingMemoryStore {
            generated_index: index.clone(),
            ..Default::default()
        };
        let secondary = RecordingMemoryStore {
            fail_put: true,
            ..Default::default()
        };
        let caps = test_caps_with_stores(
            blackboard.clone(),
            Arc::new(primary.clone()),
            vec![Arc::new(secondary.clone())],
            Arc::new(NoopFileSearchProvider),
        );

        let written = caps
            .memory_writer()
            .insert("replicated memory".into(), MemoryRank::LongTerm, 30)
            .await
            .expect("primary insert succeeds");

        assert_eq!(written, index);
        assert_eq!(primary.inserted_indexes().await, vec![index.clone()]);
        assert!(secondary.put_indexes().await.is_empty());
        let metadata_present = blackboard
            .read(|bb| bb.memory_metadata().contains_key(&index))
            .await;
        assert!(metadata_present);
    }

    #[tokio::test]
    async fn memory_writer_fans_out_to_secondary_on_happy_path() {
        let blackboard = Blackboard::default();
        let index = MemoryIndex::new("happy-fanout");
        let primary = RecordingMemoryStore {
            generated_index: index.clone(),
            ..Default::default()
        };
        let secondary = RecordingMemoryStore::default();
        let caps = test_caps_with_stores(
            blackboard.clone(),
            Arc::new(primary.clone()),
            vec![Arc::new(secondary.clone())],
            Arc::new(NoopFileSearchProvider),
        );

        let written = caps
            .memory_writer()
            .insert("replicated memory".into(), MemoryRank::LongTerm, 30)
            .await
            .expect("primary insert succeeds");

        assert_eq!(written, index.clone());
        assert_eq!(primary.inserted_indexes().await, vec![index.clone()]);
        assert_eq!(secondary.put_indexes().await, vec![index]);
    }

    #[tokio::test]
    async fn memory_writer_primary_failure_prevents_metadata_and_secondary_writes() {
        let blackboard = Blackboard::default();
        let index = MemoryIndex::new("primary-fails");
        let primary = RecordingMemoryStore {
            generated_index: index.clone(),
            fail_insert: true,
            ..Default::default()
        };
        let secondary = RecordingMemoryStore::default();
        let caps = test_caps_with_stores(
            blackboard.clone(),
            Arc::new(primary.clone()),
            vec![Arc::new(secondary.clone())],
            Arc::new(NoopFileSearchProvider),
        );

        let result = caps
            .memory_writer()
            .insert("not durable".into(), MemoryRank::ShortTerm, 5)
            .await;

        assert!(result.is_err());
        assert!(secondary.put_indexes().await.is_empty());
        let has_metadata = blackboard
            .read(|bb| bb.memory_metadata().contains_key(&index))
            .await;
        assert!(!has_metadata);
    }

    #[tokio::test]
    async fn vector_memory_searcher_records_access_for_hits_only() {
        let blackboard = Blackboard::default();
        let hit_index = MemoryIndex::new("hit");
        let miss_index = MemoryIndex::new("miss");
        blackboard
            .apply(BlackboardCommand::UpsertMemoryMetadata {
                index: miss_index.clone(),
                rank_if_new: MemoryRank::MidTerm,
                decay_if_new_secs: 0,
                now: Utc::now(),
                patch: MemoryMetaPatch::default(),
            })
            .await;
        let primary = RecordingMemoryStore {
            search_hits: vec![MemoryRecord {
                index: hit_index.clone(),
                content: MemoryContent::new("matched content"),
                rank: MemoryRank::MidTerm,
            }],
            ..Default::default()
        };
        let caps = test_caps_with_stores(
            blackboard.clone(),
            Arc::new(primary),
            Vec::new(),
            Arc::new(NoopFileSearchProvider),
        );

        let hits = caps
            .vector_memory_searcher()
            .search("matched", 10)
            .await
            .expect("search succeeds");

        assert_eq!(hits.len(), 1);
        let (hit_access_count, miss_access_count) = blackboard
            .read(|bb| {
                (
                    bb.memory_metadata()
                        .get(&hit_index)
                        .map(|meta| meta.access_count),
                    bb.memory_metadata()
                        .get(&miss_index)
                        .map(|meta| meta.access_count),
                )
            })
            .await;
        assert_eq!(hit_access_count, Some(1));
        assert_eq!(miss_access_count, Some(0));
    }

    #[tokio::test]
    async fn memory_compactor_uses_atomic_compaction_and_updates_metadata() {
        let blackboard = Blackboard::default();
        let source = MemoryIndex::new("source");
        let merged = MemoryIndex::new("merged");
        let primary = RecordingMemoryStore {
            generated_index: merged.clone(),
            ..Default::default()
        };
        let secondary = RecordingMemoryStore::default();
        let caps = test_caps_with_stores(
            blackboard.clone(),
            Arc::new(primary.clone()),
            vec![Arc::new(secondary.clone())],
            Arc::new(NoopFileSearchProvider),
        );

        let result = caps
            .memory_compactor()
            .merge(
                std::slice::from_ref(&source),
                "merged content".into(),
                MemoryRank::LongTerm,
                30,
            )
            .await;

        assert_eq!(result.unwrap(), merged.clone());
        assert_eq!(primary.compacted_indexes().await, vec![merged.clone()]);
        assert_eq!(primary.compacted_sources().await, vec![source.clone()]);
        assert_eq!(secondary.compacted_indexes().await, vec![merged.clone()]);
        assert_eq!(secondary.compacted_sources().await, vec![source]);
        let has_metadata = blackboard
            .read(|bb| bb.memory_metadata().contains_key(&merged))
            .await;
        assert!(has_metadata);
    }

    #[tokio::test]
    async fn memory_compactor_primary_compact_failure_prevents_metadata_and_replica_writes() {
        let blackboard = Blackboard::default();
        let source = MemoryIndex::new("source");
        let merged = MemoryIndex::new("merged");
        let primary = RecordingMemoryStore {
            generated_index: merged.clone(),
            fail_compact: true,
            ..Default::default()
        };
        let secondary = RecordingMemoryStore::default();
        let caps = test_caps_with_stores(
            blackboard.clone(),
            Arc::new(primary.clone()),
            vec![Arc::new(secondary.clone())],
            Arc::new(NoopFileSearchProvider),
        );

        let result = caps
            .memory_compactor()
            .merge(&[source], "merged content".into(), MemoryRank::LongTerm, 30)
            .await;

        assert!(result.is_err());
        assert!(secondary.compacted_indexes().await.is_empty());
        let has_metadata = blackboard
            .read(|bb| bb.memory_metadata().contains_key(&merged))
            .await;
        assert!(!has_metadata);
    }
}
