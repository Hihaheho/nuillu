//! `MemoryStore` port + memory-domain capability handles.
//!
//! Memory content lives in this trait's implementations; the blackboard
//! mirrors metadata only. Modules never touch the port directly: they hold
//! one of the capability wrappers below, all of which thread blackboard
//! metadata updates alongside the storage calls.

use std::rc::Rc;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use nuillu_blackboard::{Blackboard, BlackboardCommand, MemoryMetaPatch};
use nuillu_module::ports::{Clock, PortError};
use nuillu_types::{MemoryContent, MemoryIndex, MemoryRank};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

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
    async fn insert(
        &self,
        mem: NewMemory,
        stored_at: DateTime<Utc>,
    ) -> Result<MemoryRecord, PortError>;
    async fn put(&self, mem: IndexedMemory) -> Result<MemoryRecord, PortError>;
    async fn compact(
        &self,
        mem: NewMemory,
        sources: &[MemoryIndex],
        stored_at: DateTime<Utc>,
    ) -> Result<MemoryRecord, PortError>;
    async fn put_compacted(
        &self,
        mem: IndexedMemory,
        sources: &[MemoryIndex],
    ) -> Result<MemoryRecord, PortError>;
    async fn get(&self, index: &MemoryIndex) -> Result<Option<MemoryRecord>, PortError>;
    async fn list_by_rank(&self, rank: MemoryRank) -> Result<Vec<MemoryRecord>, PortError>;
    async fn search(&self, q: &MemoryQuery) -> Result<Vec<MemoryRecord>, PortError>;
    async fn linked(&self, q: &LinkedMemoryQuery) -> Result<Vec<LinkedMemoryRecord>, PortError>;
    async fn upsert_link(
        &self,
        link: NewMemoryLink,
        updated_at: DateTime<Utc>,
    ) -> Result<MemoryLink, PortError>;
    async fn delete(&self, index: &MemoryIndex) -> Result<(), PortError>;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema, Default)]
#[serde(rename_all = "snake_case")]
pub enum MemoryKind {
    Episode,
    #[default]
    Statement,
    Reflection,
    Hypothesis,
    Dream,
    Procedure,
    Plan,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct MemoryConcept {
    pub label: String,
    pub mention_text: Option<String>,
    pub loose_type: Option<String>,
    pub confidence: f32,
}

impl MemoryConcept {
    pub fn new(label: impl Into<String>) -> Self {
        Self {
            label: label.into(),
            mention_text: None,
            loose_type: None,
            confidence: 1.0,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct MemoryTag {
    pub label: String,
    pub namespace: String,
    pub confidence: f32,
}

impl MemoryTag {
    pub fn operational(label: impl Into<String>) -> Self {
        Self {
            label: label.into(),
            namespace: "operation".to_owned(),
            confidence: 1.0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum MemoryLinkRelation {
    Related,
    Supports,
    Contradicts,
    Updates,
    Corrects,
    DerivedFrom,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema, Default)]
#[serde(rename_all = "snake_case")]
pub enum MemoryLinkDirection {
    Outgoing,
    Incoming,
    #[default]
    Both,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct NewMemoryLink {
    pub from_memory: MemoryIndex,
    pub to_memory: MemoryIndex,
    pub relation: MemoryLinkRelation,
    pub freeform_relation: Option<String>,
    pub strength: f32,
    pub confidence: f32,
}

impl NewMemoryLink {
    pub fn derived_from(from_memory: MemoryIndex, to_memory: MemoryIndex) -> Self {
        Self {
            from_memory,
            to_memory,
            relation: MemoryLinkRelation::DerivedFrom,
            freeform_relation: None,
            strength: 1.0,
            confidence: 1.0,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct MemoryLink {
    pub from_memory: MemoryIndex,
    pub to_memory: MemoryIndex,
    pub relation: MemoryLinkRelation,
    pub freeform_relation: Option<String>,
    pub strength: f32,
    pub confidence: f32,
    pub updated_at: DateTime<Utc>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct LinkedMemoryRecord {
    pub record: MemoryRecord,
    pub link: MemoryLink,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LinkedMemoryQuery {
    pub memory_indexes: Vec<MemoryIndex>,
    pub relation_filter: Vec<MemoryLinkRelation>,
    pub direction: MemoryLinkDirection,
    pub limit: usize,
}

impl LinkedMemoryQuery {
    pub fn around(memory_indexes: Vec<MemoryIndex>, limit: usize) -> Self {
        Self {
            memory_indexes,
            relation_filter: Vec::new(),
            direction: MemoryLinkDirection::Both,
            limit,
        }
    }
}

#[derive(Debug, Clone)]
pub struct NewMemory {
    pub content: MemoryContent,
    pub rank: MemoryRank,
    pub occurred_at: Option<DateTime<Utc>>,
    pub kind: MemoryKind,
    pub concepts: Vec<MemoryConcept>,
    pub tags: Vec<MemoryTag>,
    pub affect_arousal: f32,
    pub valence: f32,
    pub emotion: String,
}

impl NewMemory {
    pub fn statement(
        content: impl Into<MemoryContent>,
        rank: MemoryRank,
        occurred_at: Option<DateTime<Utc>>,
    ) -> Self {
        Self {
            content: content.into(),
            rank,
            occurred_at,
            kind: MemoryKind::Statement,
            concepts: Vec::new(),
            tags: Vec::new(),
            affect_arousal: 0.0,
            valence: 0.0,
            emotion: String::new(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct IndexedMemory {
    pub index: MemoryIndex,
    pub content: MemoryContent,
    pub rank: MemoryRank,
    pub occurred_at: Option<DateTime<Utc>>,
    pub stored_at: DateTime<Utc>,
    pub kind: MemoryKind,
    pub concepts: Vec<MemoryConcept>,
    pub tags: Vec<MemoryTag>,
    pub affect_arousal: f32,
    pub valence: f32,
    pub emotion: String,
}

impl IndexedMemory {
    pub fn from_record(record: MemoryRecord) -> Self {
        Self {
            index: record.index,
            content: record.content,
            rank: record.rank,
            occurred_at: record.occurred_at,
            stored_at: record.stored_at,
            kind: record.kind,
            concepts: record.concepts,
            tags: record.tags,
            affect_arousal: record.affect_arousal,
            valence: record.valence,
            emotion: record.emotion,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct MemoryRecord {
    pub index: MemoryIndex,
    pub content: MemoryContent,
    pub rank: MemoryRank,
    pub occurred_at: Option<DateTime<Utc>>,
    pub stored_at: DateTime<Utc>,
    pub kind: MemoryKind,
    pub concepts: Vec<MemoryConcept>,
    pub tags: Vec<MemoryTag>,
    pub affect_arousal: f32,
    pub valence: f32,
    pub emotion: String,
}

#[derive(Debug, Clone)]
pub struct MemoryQuery {
    pub text: String,
    pub limit: usize,
    pub kinds: Vec<MemoryKind>,
    pub concepts: Vec<String>,
    pub tags: Vec<String>,
}

impl MemoryQuery {
    pub fn text(text: impl Into<String>, limit: usize) -> Self {
        Self {
            text: text.into(),
            limit,
            kinds: Vec::new(),
            concepts: Vec::new(),
            tags: Vec::new(),
        }
    }
}

/// Memory store that accepts every write, returns empty reads, and assigns
/// a unique synthetic [`MemoryIndex`] to each insert.
#[derive(Debug, Default)]
pub struct NoopMemoryStore;

#[async_trait(?Send)]
impl MemoryStore for NoopMemoryStore {
    async fn insert(
        &self,
        mem: NewMemory,
        stored_at: DateTime<Utc>,
    ) -> Result<MemoryRecord, PortError> {
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let id = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Ok(MemoryRecord {
            index: MemoryIndex::new(format!("noop-memory-{id}")),
            content: mem.content,
            rank: mem.rank,
            occurred_at: mem.occurred_at,
            stored_at,
            kind: mem.kind,
            concepts: mem.concepts,
            tags: mem.tags,
            affect_arousal: mem.affect_arousal,
            valence: mem.valence,
            emotion: mem.emotion,
        })
    }

    async fn put(&self, mem: IndexedMemory) -> Result<MemoryRecord, PortError> {
        Ok(MemoryRecord {
            index: mem.index,
            content: mem.content,
            rank: mem.rank,
            occurred_at: mem.occurred_at,
            stored_at: mem.stored_at,
            kind: mem.kind,
            concepts: mem.concepts,
            tags: mem.tags,
            affect_arousal: mem.affect_arousal,
            valence: mem.valence,
            emotion: mem.emotion,
        })
    }

    async fn compact(
        &self,
        mem: NewMemory,
        _sources: &[MemoryIndex],
        stored_at: DateTime<Utc>,
    ) -> Result<MemoryRecord, PortError> {
        self.insert(mem, stored_at).await
    }

    async fn put_compacted(
        &self,
        mem: IndexedMemory,
        _sources: &[MemoryIndex],
    ) -> Result<MemoryRecord, PortError> {
        self.put(mem).await
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

    async fn linked(&self, _q: &LinkedMemoryQuery) -> Result<Vec<LinkedMemoryRecord>, PortError> {
        Ok(Vec::new())
    }

    async fn upsert_link(
        &self,
        link: NewMemoryLink,
        updated_at: DateTime<Utc>,
    ) -> Result<MemoryLink, PortError> {
        Ok(MemoryLink {
            from_memory: link.from_memory,
            to_memory: link.to_memory,
            relation: link.relation,
            freeform_relation: link.freeform_relation,
            strength: link.strength,
            confidence: link.confidence,
            updated_at,
        })
    }

    async fn delete(&self, _index: &MemoryIndex) -> Result<(), PortError> {
        Ok(())
    }
}

/// Vector search over the primary memory store + automatic access patching.
#[derive(Clone)]
pub struct MemorySearcher {
    primary_store: Rc<dyn MemoryStore>,
    blackboard: Blackboard,
    clock: Rc<dyn Clock>,
}

impl MemorySearcher {
    pub fn new(
        primary_store: Rc<dyn MemoryStore>,
        blackboard: Blackboard,
        clock: Rc<dyn Clock>,
    ) -> Self {
        Self {
            primary_store,
            blackboard,
            clock,
        }
    }

    pub async fn search(&self, query: &str, limit: usize) -> Result<Vec<MemoryRecord>, PortError> {
        let q = MemoryQuery::text(query, limit);
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
    primary_store: Rc<dyn MemoryStore>,
}

impl MemoryContentReader {
    pub fn new(primary_store: Rc<dyn MemoryStore>) -> Self {
        Self { primary_store }
    }

    pub async fn get(&self, index: &MemoryIndex) -> Result<Option<MemoryRecord>, PortError> {
        self.primary_store.get(index).await
    }

    pub async fn linked(
        &self,
        query: &LinkedMemoryQuery,
    ) -> Result<Vec<LinkedMemoryRecord>, PortError> {
        self.primary_store.linked(query).await
    }
}

/// Insert a new memory entry. Mirrors metadata onto the blackboard.
#[derive(Clone)]
pub struct MemoryWriter {
    primary_store: Rc<dyn MemoryStore>,
    replicas: Vec<Rc<dyn MemoryStore>>,
    blackboard: Blackboard,
    clock: Rc<dyn Clock>,
}

impl MemoryWriter {
    pub fn new(
        primary_store: Rc<dyn MemoryStore>,
        replicas: Vec<Rc<dyn MemoryStore>>,
        blackboard: Blackboard,
        clock: Rc<dyn Clock>,
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
        let record = self
            .insert_entry(
                NewMemory::statement(MemoryContent::new(content), rank, occurred_at),
                decay_secs,
            )
            .await?;
        Ok(record.index)
    }

    pub async fn insert_entry(
        &self,
        new: NewMemory,
        decay_secs: i64,
    ) -> Result<MemoryRecord, PortError> {
        let mut new = new;
        self.stamp_interoception(&mut new).await;
        let rank = new.rank;
        let occurred_at = new.occurred_at;
        let now = self.clock.now();
        let record = self.primary_store.insert(new, now).await?;
        let indexed = IndexedMemory::from_record(record.clone());

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
                index: record.index.clone(),
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
        Ok(record)
    }

    async fn stamp_interoception(&self, new: &mut NewMemory) {
        let state = self.blackboard.read(|bb| bb.interoception().clone()).await;
        new.affect_arousal = state.affect_arousal;
        new.valence = state.valence;
        new.emotion = state.emotion;
    }
}

/// Delete + merge memories. The compaction module is the canonical holder;
/// per design it accumulates `remember_tokens` when merging.
#[derive(Clone)]
pub struct MemoryCompactor {
    primary_store: Rc<dyn MemoryStore>,
    replicas: Vec<Rc<dyn MemoryStore>>,
    blackboard: Blackboard,
    clock: Rc<dyn Clock>,
}

impl MemoryCompactor {
    pub fn new(
        primary_store: Rc<dyn MemoryStore>,
        replicas: Vec<Rc<dyn MemoryStore>>,
        blackboard: Blackboard,
        clock: Rc<dyn Clock>,
    ) -> Self {
        Self {
            primary_store,
            replicas,
            blackboard,
            clock,
        }
    }

    /// Insert a compaction summary and remove source memories from live
    /// retrieval. `remember_tokens` are accumulated on the summary by the
    /// count of source memories.
    pub async fn merge(
        &self,
        sources: &[MemoryIndex],
        merged_content: String,
        merged_rank: MemoryRank,
        decay_secs: i64,
    ) -> Result<MemoryIndex, PortError> {
        self.write_summary(
            sources,
            merged_content,
            merged_rank,
            decay_secs,
            Vec::new(),
            vec![MemoryTag::operational("compaction-summary")],
        )
        .await
    }

    pub async fn write_summary(
        &self,
        sources: &[MemoryIndex],
        summary_content: String,
        summary_rank: MemoryRank,
        decay_secs: i64,
        concepts: Vec<MemoryConcept>,
        tags: Vec<MemoryTag>,
    ) -> Result<MemoryIndex, PortError> {
        let occurred_at = self
            .get_many(sources)
            .await?
            .into_iter()
            .filter_map(|record| record.occurred_at)
            .min()
            .or_else(|| Some(self.clock.now()));
        let mut new = NewMemory {
            content: MemoryContent::new(summary_content),
            rank: summary_rank,
            occurred_at,
            kind: MemoryKind::Reflection,
            concepts,
            tags,
            affect_arousal: 0.0,
            valence: 0.0,
            emotion: String::new(),
        };
        self.stamp_interoception(&mut new).await;

        let now = self.clock.now();
        let record = self.primary_store.compact(new, sources, now).await?;
        let indexed = IndexedMemory::from_record(record.clone());

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
                index: record.index.clone(),
                rank_if_new: summary_rank,
                occurred_at_if_new: occurred_at,
                decay_if_new_secs: decay_secs,
                now: self.clock.now(),
                patch: MemoryMetaPatch {
                    rank: Some(summary_rank),
                    occurred_at: Some(occurred_at),
                    decay_remaining_secs: Some(decay_secs),
                    increment_remember_tokens: Some(sources.len() as u32),
                    ..Default::default()
                },
            })
            .await;
        for source in sources {
            self.blackboard
                .apply(BlackboardCommand::RemoveMemoryMetadata {
                    index: source.clone(),
                })
                .await;
        }

        Ok(record.index)
    }

    async fn stamp_interoception(&self, new: &mut NewMemory) {
        let state = self.blackboard.read(|bb| bb.interoception().clone()).await;
        new.affect_arousal = state.affect_arousal;
        new.valence = state.valence;
        new.emotion = state.emotion;
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

/// Non-destructive writer for memory-to-memory associations.
#[derive(Clone)]
pub struct MemoryAssociator {
    primary_store: Rc<dyn MemoryStore>,
    replicas: Vec<Rc<dyn MemoryStore>>,
    clock: Rc<dyn Clock>,
}

impl MemoryAssociator {
    pub fn new(
        primary_store: Rc<dyn MemoryStore>,
        replicas: Vec<Rc<dyn MemoryStore>>,
        clock: Rc<dyn Clock>,
    ) -> Self {
        Self {
            primary_store,
            replicas,
            clock,
        }
    }

    pub async fn upsert_link(&self, link: NewMemoryLink) -> Result<MemoryLink, PortError> {
        let updated_at = self.clock.now();
        let record = self
            .primary_store
            .upsert_link(link.clone(), updated_at)
            .await?;
        let replica_writes = self.replicas.iter().enumerate().map(|(replica, store)| {
            let link = link.clone();
            async move {
                if let Err(error) = store.upsert_link(link, updated_at).await {
                    tracing::warn!(replica, ?error, "secondary memory link upsert failed");
                }
            }
        });
        futures::future::join_all(replica_writes).await;
        Ok(record)
    }
}

/// Hard-delete a memory entry and remove the blackboard metadata mirror.
#[derive(Clone)]
pub struct MemoryDeleter {
    primary_store: Rc<dyn MemoryStore>,
    replicas: Vec<Rc<dyn MemoryStore>>,
    blackboard: Blackboard,
}

impl MemoryDeleter {
    pub fn new(
        primary_store: Rc<dyn MemoryStore>,
        replicas: Vec<Rc<dyn MemoryStore>>,
        blackboard: Blackboard,
    ) -> Self {
        Self {
            primary_store,
            replicas,
            blackboard,
        }
    }

    pub async fn delete(&self, index: &MemoryIndex) -> Result<(), PortError> {
        self.primary_store.delete(index).await?;
        for (replica, store) in self.replicas.iter().enumerate() {
            if let Err(error) = store.delete(index).await {
                tracing::warn!(replica, ?error, "secondary memory delete failed");
            }
        }
        self.blackboard
            .apply(BlackboardCommand::RemoveMemoryMetadata {
                index: index.clone(),
            })
            .await;
        let identity_memories = self
            .blackboard
            .read(|bb| {
                bb.identity_memories()
                    .iter()
                    .filter(|memory| memory.index != *index)
                    .cloned()
                    .collect::<Vec<_>>()
            })
            .await;
        self.blackboard
            .apply(BlackboardCommand::SetIdentityMemories(identity_memories))
            .await;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;

    use super::*;
    use nuillu_module::ports::Clock;

    #[derive(Debug)]
    struct FixedClock(DateTime<Utc>);

    #[async_trait(?Send)]
    impl Clock for FixedClock {
        fn now(&self) -> DateTime<Utc> {
            self.0
        }

        async fn sleep_until(&self, _deadline: DateTime<Utc>) {}
    }

    #[derive(Clone)]
    struct StaticMemoryStore {
        record: MemoryRecord,
    }

    #[derive(Clone, Default)]
    struct RecordingMemoryStore {
        writes: Rc<RefCell<Vec<RecordedMemoryWrite>>>,
    }

    #[derive(Clone)]
    enum RecordedMemoryWrite {
        Insert(NewMemory),
        Compact(NewMemory, Vec<MemoryIndex>),
    }

    impl RecordingMemoryStore {
        fn record_from_new(
            index: MemoryIndex,
            mem: NewMemory,
            stored_at: DateTime<Utc>,
        ) -> MemoryRecord {
            MemoryRecord {
                index,
                content: mem.content,
                rank: mem.rank,
                occurred_at: mem.occurred_at,
                stored_at,
                kind: mem.kind,
                concepts: mem.concepts,
                tags: mem.tags,
                affect_arousal: mem.affect_arousal,
                valence: mem.valence,
                emotion: mem.emotion,
            }
        }
    }

    #[async_trait(?Send)]
    impl MemoryStore for RecordingMemoryStore {
        async fn insert(
            &self,
            mem: NewMemory,
            stored_at: DateTime<Utc>,
        ) -> Result<MemoryRecord, PortError> {
            self.writes
                .borrow_mut()
                .push(RecordedMemoryWrite::Insert(mem.clone()));
            Ok(Self::record_from_new(
                MemoryIndex::new("recorded-memory"),
                mem,
                stored_at,
            ))
        }

        async fn put(&self, mem: IndexedMemory) -> Result<MemoryRecord, PortError> {
            Ok(MemoryRecord {
                index: mem.index,
                content: mem.content,
                rank: mem.rank,
                occurred_at: mem.occurred_at,
                stored_at: mem.stored_at,
                kind: mem.kind,
                concepts: mem.concepts,
                tags: mem.tags,
                affect_arousal: mem.affect_arousal,
                valence: mem.valence,
                emotion: mem.emotion,
            })
        }

        async fn compact(
            &self,
            mem: NewMemory,
            sources: &[MemoryIndex],
            stored_at: DateTime<Utc>,
        ) -> Result<MemoryRecord, PortError> {
            self.writes
                .borrow_mut()
                .push(RecordedMemoryWrite::Compact(mem.clone(), sources.to_vec()));
            Ok(Self::record_from_new(
                MemoryIndex::new("recorded-compaction"),
                mem,
                stored_at,
            ))
        }

        async fn put_compacted(
            &self,
            mem: IndexedMemory,
            _sources: &[MemoryIndex],
        ) -> Result<MemoryRecord, PortError> {
            self.put(mem).await
        }

        async fn get(&self, index: &MemoryIndex) -> Result<Option<MemoryRecord>, PortError> {
            let at = DateTime::from_timestamp(1_700_000_000, 0).unwrap();
            Ok(Some(MemoryRecord {
                index: index.clone(),
                content: MemoryContent::new("source memory"),
                rank: MemoryRank::ShortTerm,
                occurred_at: Some(at),
                stored_at: at,
                kind: MemoryKind::Statement,
                concepts: Vec::new(),
                tags: Vec::new(),
                affect_arousal: 0.0,
                valence: 0.0,
                emotion: String::new(),
            }))
        }

        async fn list_by_rank(&self, _rank: MemoryRank) -> Result<Vec<MemoryRecord>, PortError> {
            Ok(Vec::new())
        }

        async fn search(&self, _q: &MemoryQuery) -> Result<Vec<MemoryRecord>, PortError> {
            Ok(Vec::new())
        }

        async fn linked(
            &self,
            _q: &LinkedMemoryQuery,
        ) -> Result<Vec<LinkedMemoryRecord>, PortError> {
            Ok(Vec::new())
        }

        async fn upsert_link(
            &self,
            _link: NewMemoryLink,
            _updated_at: DateTime<Utc>,
        ) -> Result<MemoryLink, PortError> {
            Err(PortError::InvalidInput(
                "recording store does not support links".into(),
            ))
        }

        async fn delete(&self, _index: &MemoryIndex) -> Result<(), PortError> {
            Ok(())
        }
    }

    #[async_trait(?Send)]
    impl MemoryStore for StaticMemoryStore {
        async fn insert(
            &self,
            _mem: NewMemory,
            _stored_at: DateTime<Utc>,
        ) -> Result<MemoryRecord, PortError> {
            Ok(self.record.clone())
        }

        async fn put(&self, _mem: IndexedMemory) -> Result<MemoryRecord, PortError> {
            Ok(self.record.clone())
        }

        async fn compact(
            &self,
            _mem: NewMemory,
            _sources: &[MemoryIndex],
            _stored_at: DateTime<Utc>,
        ) -> Result<MemoryRecord, PortError> {
            Ok(self.record.clone())
        }

        async fn put_compacted(
            &self,
            _mem: IndexedMemory,
            _sources: &[MemoryIndex],
        ) -> Result<MemoryRecord, PortError> {
            Ok(self.record.clone())
        }

        async fn get(&self, index: &MemoryIndex) -> Result<Option<MemoryRecord>, PortError> {
            Ok((index == &self.record.index).then(|| self.record.clone()))
        }

        async fn list_by_rank(&self, _rank: MemoryRank) -> Result<Vec<MemoryRecord>, PortError> {
            Ok(Vec::new())
        }

        async fn search(&self, _q: &MemoryQuery) -> Result<Vec<MemoryRecord>, PortError> {
            Ok(vec![self.record.clone()])
        }

        async fn linked(
            &self,
            _q: &LinkedMemoryQuery,
        ) -> Result<Vec<LinkedMemoryRecord>, PortError> {
            Ok(vec![LinkedMemoryRecord {
                record: self.record.clone(),
                link: MemoryLink {
                    from_memory: MemoryIndex::new("source"),
                    to_memory: self.record.index.clone(),
                    relation: MemoryLinkRelation::Related,
                    freeform_relation: None,
                    strength: 1.0,
                    confidence: 1.0,
                    updated_at: self.record.stored_at,
                },
            }])
        }

        async fn upsert_link(
            &self,
            link: NewMemoryLink,
            _updated_at: DateTime<Utc>,
        ) -> Result<MemoryLink, PortError> {
            Ok(MemoryLink {
                from_memory: link.from_memory,
                to_memory: link.to_memory,
                relation: link.relation,
                freeform_relation: link.freeform_relation,
                strength: link.strength,
                confidence: link.confidence,
                updated_at: self.record.stored_at,
            })
        }

        async fn delete(&self, _index: &MemoryIndex) -> Result<(), PortError> {
            Ok(())
        }
    }

    fn test_record() -> MemoryRecord {
        let at = DateTime::from_timestamp(1_700_000_000, 0).unwrap();
        MemoryRecord {
            index: MemoryIndex::new("memory-1"),
            content: MemoryContent::new("alpha memory"),
            rank: MemoryRank::LongTerm,
            occurred_at: Some(at),
            stored_at: at,
            kind: MemoryKind::Statement,
            concepts: vec![MemoryConcept::new("alpha")],
            tags: vec![MemoryTag::operational("test")],
            affect_arousal: 0.4,
            valence: 0.2,
            emotion: "curious".to_owned(),
        }
    }

    async fn set_test_interoception(blackboard: &Blackboard) {
        blackboard
            .apply(BlackboardCommand::UpdateInteroceptive {
                patch: nuillu_blackboard::InteroceptivePatch {
                    affect_arousal: Some(0.8),
                    valence: Some(-0.35),
                    emotion: Some("  tense focus  ".to_owned()),
                    ..Default::default()
                },
                now: DateTime::from_timestamp(1_700_000_001, 0).unwrap(),
            })
            .await;
    }

    #[tokio::test]
    async fn memory_writer_stamps_current_interoception_on_insert() {
        let now = DateTime::from_timestamp(1_700_000_010, 0).unwrap();
        let blackboard = Blackboard::new();
        set_test_interoception(&blackboard).await;
        let store = RecordingMemoryStore::default();
        let writer = MemoryWriter::new(
            Rc::new(store.clone()),
            Vec::new(),
            blackboard,
            Rc::new(FixedClock(now)),
        );

        let record = writer
            .insert_entry(
                NewMemory::statement(
                    MemoryContent::new("affect stamped memory"),
                    MemoryRank::ShortTerm,
                    Some(now),
                ),
                60,
            )
            .await
            .unwrap();

        assert_eq!(record.affect_arousal, 0.8);
        assert_eq!(record.valence, -0.35);
        assert_eq!(record.emotion, "tense focus");
        let writes = store.writes.borrow();
        let RecordedMemoryWrite::Insert(new) = &writes[0] else {
            panic!("expected insert write");
        };
        assert_eq!(new.affect_arousal, 0.8);
        assert_eq!(new.valence, -0.35);
        assert_eq!(new.emotion, "tense focus");
    }

    #[tokio::test]
    async fn memory_compactor_stamps_current_interoception_on_summary() {
        let now = DateTime::from_timestamp(1_700_000_010, 0).unwrap();
        let blackboard = Blackboard::new();
        set_test_interoception(&blackboard).await;
        let store = RecordingMemoryStore::default();
        let compactor = MemoryCompactor::new(
            Rc::new(store.clone()),
            Vec::new(),
            blackboard,
            Rc::new(FixedClock(now)),
        );

        let source = MemoryIndex::new("source-memory");
        let index = compactor
            .write_summary(
                std::slice::from_ref(&source),
                "summary memory".to_owned(),
                MemoryRank::MidTerm,
                120,
                Vec::new(),
                Vec::new(),
            )
            .await
            .unwrap();

        assert_eq!(index.as_str(), "recorded-compaction");
        let writes = store.writes.borrow();
        let RecordedMemoryWrite::Compact(new, sources) = &writes[0] else {
            panic!("expected compact write");
        };
        assert_eq!(sources, &[source]);
        assert_eq!(new.affect_arousal, 0.8);
        assert_eq!(new.valence, -0.35);
        assert_eq!(new.emotion, "tense focus");
    }

    #[tokio::test]
    async fn vector_search_records_direct_access() {
        let record = test_record();
        let now = record.stored_at;
        let blackboard = Blackboard::new();
        let searcher = MemorySearcher::new(
            Rc::new(StaticMemoryStore {
                record: record.clone(),
            }),
            blackboard.clone(),
            Rc::new(FixedClock(now)),
        );

        let hits = searcher.search("alpha", 1).await.unwrap();

        assert_eq!(hits[0].index, record.index);
        let metadata = blackboard
            .read(|bb| bb.memory_metadata().get(&record.index).cloned())
            .await
            .unwrap();
        assert_eq!(metadata.access_count, 1);
        assert_eq!(metadata.last_accessed, now);
    }

    #[tokio::test]
    async fn linked_lookup_does_not_record_access() {
        let record = test_record();
        let blackboard = Blackboard::new();
        let reader = MemoryContentReader::new(Rc::new(StaticMemoryStore {
            record: record.clone(),
        }));

        let linked = reader
            .linked(&LinkedMemoryQuery::around(vec![record.index.clone()], 8))
            .await
            .unwrap();

        assert_eq!(linked[0].record.index, record.index);
        assert!(
            blackboard
                .read(|bb| bb.memory_metadata().get(&record.index).cloned())
                .await
                .is_none()
        );
    }
}
