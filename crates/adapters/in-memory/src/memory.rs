use std::collections::{HashMap, HashSet};
use std::rc::Rc;
use std::sync::Mutex;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use nuillu_memory::{
    IndexedMemory, LinkedMemoryQuery, LinkedMemoryRecord, MemoryLink, MemoryLinkDirection,
    MemoryLinkRelation, MemoryQuery, MemoryRecord, MemoryStore, NewMemory, NewMemoryLink,
};
use nuillu_module::ports::{Embedder, PortError};
use nuillu_types::{MemoryIndex, MemoryRank};
use uuid::Uuid;

#[derive(Debug, Default)]
struct MemoryState {
    records: HashMap<MemoryIndex, StoredMemory>,
    links: Vec<StoredLink>,
    next_sequence: u64,
    next_link_sequence: u64,
}

#[derive(Debug, Clone)]
struct StoredMemory {
    record: MemoryRecord,
    embedding: Vec<f32>,
    sequence: u64,
}

#[derive(Debug, Clone)]
struct StoredLink {
    link: MemoryLink,
    sequence: u64,
}

/// Embedding-backed memory store that keeps records and vectors in process memory.
///
/// Every content write and every search with a nonzero limit calls the configured
/// [`Embedder`]. No lexical-search fallback is used.
pub struct InMemoryMemoryStore {
    embedder: Rc<dyn Embedder>,
    dimensions: usize,
    state: Mutex<MemoryState>,
}

impl std::fmt::Debug for InMemoryMemoryStore {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("InMemoryMemoryStore")
            .field("dimensions", &self.dimensions)
            .field("state", &self.state)
            .finish_non_exhaustive()
    }
}

impl InMemoryMemoryStore {
    pub fn new(embedder: Rc<dyn Embedder>) -> Self {
        Self {
            dimensions: embedder.dimensions(),
            embedder,
            state: Mutex::new(MemoryState::default()),
        }
    }

    async fn embed(&self, text: &str) -> Result<Vec<f32>, PortError> {
        let embedding = self.embedder.embed(text).await?;
        crate::embedding::validate("memory", self.dimensions, embedding)
    }

    fn lock(&self) -> Result<std::sync::MutexGuard<'_, MemoryState>, PortError> {
        self.state
            .lock()
            .map_err(|_| PortError::Backend("memory store lock poisoned".into()))
    }

    fn put_locked(
        state: &mut MemoryState,
        mem: IndexedMemory,
        embedding: Vec<f32>,
    ) -> MemoryRecord {
        let sequence = state.records.get(&mem.index).map_or_else(
            || {
                let sequence = state.next_sequence;
                state.next_sequence = state.next_sequence.saturating_add(1);
                sequence
            },
            |stored| stored.sequence,
        );
        let record = MemoryRecord {
            index: mem.index.clone(),
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
        };
        state.records.insert(
            mem.index,
            StoredMemory {
                record: record.clone(),
                embedding,
                sequence,
            },
        );
        record
    }

    fn delete_locked(state: &mut MemoryState, index: &MemoryIndex) {
        state.records.remove(index);
        state
            .links
            .retain(|stored| &stored.link.from_memory != index && &stored.link.to_memory != index);
    }

    fn compact_locked(
        state: &mut MemoryState,
        mem: IndexedMemory,
        embedding: Vec<f32>,
        sources: &[MemoryIndex],
    ) -> MemoryRecord {
        let record = Self::put_locked(state, mem, embedding);
        for source in sources {
            Self::delete_locked(state, source);
        }
        record
    }
}

#[async_trait(?Send)]
impl MemoryStore for InMemoryMemoryStore {
    async fn insert(
        &self,
        mem: NewMemory,
        stored_at: DateTime<Utc>,
    ) -> Result<MemoryRecord, PortError> {
        let embedding = self.embed(mem.content.as_str()).await?;
        let indexed = IndexedMemory {
            index: MemoryIndex::new(Uuid::now_v7().to_string()),
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
        };
        let mut state = self.lock()?;
        Ok(Self::put_locked(&mut state, indexed, embedding))
    }

    async fn put(&self, mem: IndexedMemory) -> Result<MemoryRecord, PortError> {
        let embedding = self.embed(mem.content.as_str()).await?;
        let mut state = self.lock()?;
        Ok(Self::put_locked(&mut state, mem, embedding))
    }

    async fn compact(
        &self,
        mem: NewMemory,
        sources: &[MemoryIndex],
        stored_at: DateTime<Utc>,
    ) -> Result<MemoryRecord, PortError> {
        let embedding = self.embed(mem.content.as_str()).await?;
        let indexed = IndexedMemory {
            index: MemoryIndex::new(Uuid::now_v7().to_string()),
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
        };
        let mut state = self.lock()?;
        Ok(Self::compact_locked(
            &mut state, indexed, embedding, sources,
        ))
    }

    async fn put_compacted(
        &self,
        mem: IndexedMemory,
        sources: &[MemoryIndex],
    ) -> Result<MemoryRecord, PortError> {
        let embedding = self.embed(mem.content.as_str()).await?;
        let mut state = self.lock()?;
        Ok(Self::compact_locked(&mut state, mem, embedding, sources))
    }

    async fn get(&self, index: &MemoryIndex) -> Result<Option<MemoryRecord>, PortError> {
        Ok(self
            .lock()?
            .records
            .get(index)
            .map(|stored| stored.record.clone()))
    }

    async fn list_by_rank(&self, rank: MemoryRank) -> Result<Vec<MemoryRecord>, PortError> {
        let state = self.lock()?;
        let mut records = state
            .records
            .values()
            .filter(|stored| stored.record.rank == rank)
            .map(|stored| stored.record.clone())
            .collect::<Vec<_>>();
        records.sort_by(|left, right| left.index.as_str().cmp(right.index.as_str()));
        Ok(records)
    }

    async fn list_recent(
        &self,
        offset: usize,
        limit: usize,
    ) -> Result<Vec<MemoryRecord>, PortError> {
        if limit == 0 {
            return Ok(Vec::new());
        }
        let state = self.lock()?;
        let mut records = state
            .records
            .values()
            .map(|stored| stored.record.clone())
            .collect::<Vec<_>>();
        records.sort_by(|left, right| {
            right
                .stored_at
                .cmp(&left.stored_at)
                .then_with(|| left.index.as_str().cmp(right.index.as_str()))
        });
        Ok(records.into_iter().skip(offset).take(limit).collect())
    }

    async fn search(&self, q: &MemoryQuery) -> Result<Vec<MemoryRecord>, PortError> {
        if q.limit == 0 {
            return Ok(Vec::new());
        }
        let query_embedding = self.embed(&q.text).await?;
        let state = self.lock()?;
        let concepts = q
            .concepts
            .iter()
            .map(|value| normalize_label(value))
            .filter(|value| !value.is_empty())
            .collect::<Vec<_>>();
        let tags = q
            .tags
            .iter()
            .filter_map(|value| normalized_tag_filter(value))
            .collect::<Vec<_>>();
        let mut matches = state
            .records
            .values()
            .filter(|stored| q.kinds.is_empty() || q.kinds.contains(&stored.record.kind))
            .filter(|stored| {
                concepts.iter().all(|required| {
                    stored
                        .record
                        .concepts
                        .iter()
                        .any(|concept| normalize_label(&concept.label) == *required)
                })
            })
            .filter(|stored| {
                tags.iter().all(|(namespace, label)| {
                    stored.record.tags.iter().any(|tag| {
                        normalize_label(&tag.label) == *label
                            && namespace.as_ref().is_none_or(|required| {
                                tag.namespace.to_ascii_lowercase() == *required
                            })
                    })
                })
            })
            .map(|stored| {
                (
                    crate::embedding::cosine_similarity(&query_embedding, &stored.embedding),
                    stored.sequence,
                    stored.record.clone(),
                )
            })
            .collect::<Vec<_>>();
        matches.sort_by(|left, right| {
            right
                .0
                .total_cmp(&left.0)
                .then_with(|| left.1.cmp(&right.1))
        });
        Ok(matches
            .into_iter()
            .skip(q.offset)
            .take(q.limit)
            .map(|(_, _, record)| record)
            .collect())
    }

    async fn linked(&self, q: &LinkedMemoryQuery) -> Result<Vec<LinkedMemoryRecord>, PortError> {
        if q.limit == 0 || q.memory_indexes.is_empty() {
            return Ok(Vec::new());
        }
        let state = self.lock()?;
        let mut records = Vec::new();
        let mut seen = HashSet::<(String, String, MemoryLinkRelation, String)>::new();
        let mut skipped = 0;
        for root in &q.memory_indexes {
            if !state.records.contains_key(root) {
                continue;
            }
            let mut links = state
                .links
                .iter()
                .filter(|stored| &stored.link.from_memory == root || &stored.link.to_memory == root)
                .collect::<Vec<_>>();
            links.sort_by(|left, right| {
                right
                    .link
                    .updated_at
                    .cmp(&left.link.updated_at)
                    .then_with(|| left.sequence.cmp(&right.sequence))
            });
            for stored in links {
                let link = &stored.link;
                if !q.relation_filter.is_empty() && !q.relation_filter.contains(&link.relation) {
                    continue;
                }
                let outgoing = &link.from_memory == root;
                if !matches!(q.direction, MemoryLinkDirection::Both)
                    && !matches!(
                        (q.direction, outgoing),
                        (MemoryLinkDirection::Outgoing, true)
                            | (MemoryLinkDirection::Incoming, false)
                    )
                {
                    continue;
                }
                let key = (
                    link.from_memory.as_str().to_owned(),
                    link.to_memory.as_str().to_owned(),
                    link.relation,
                    link.freeform_relation.clone().unwrap_or_default(),
                );
                if !seen.insert(key) {
                    continue;
                }
                if skipped < q.offset {
                    skipped += 1;
                    continue;
                }
                let linked_index = if outgoing {
                    &link.to_memory
                } else {
                    &link.from_memory
                };
                let Some(linked) = state.records.get(linked_index) else {
                    continue;
                };
                records.push(LinkedMemoryRecord {
                    record: linked.record.clone(),
                    link: link.clone(),
                });
                if records.len() == q.limit {
                    return Ok(records);
                }
            }
        }
        Ok(records)
    }

    async fn upsert_link(
        &self,
        link: NewMemoryLink,
        updated_at: DateTime<Utc>,
    ) -> Result<MemoryLink, PortError> {
        let mut state = self.lock()?;
        if !state.records.contains_key(&link.from_memory) {
            return Err(PortError::NotFound(link.from_memory.to_string()));
        }
        if !state.records.contains_key(&link.to_memory) {
            return Err(PortError::NotFound(link.to_memory.to_string()));
        }
        let memory_link = MemoryLink {
            from_memory: link.from_memory,
            to_memory: link.to_memory,
            relation: link.relation,
            freeform_relation: link.freeform_relation,
            strength: clamp_confidence(link.strength),
            confidence: clamp_confidence(link.confidence),
            updated_at,
        };
        if let Some(stored) = state.links.iter_mut().find(|stored| {
            stored.link.from_memory == memory_link.from_memory
                && stored.link.to_memory == memory_link.to_memory
                && stored.link.relation == memory_link.relation
                && stored.link.freeform_relation.as_deref().unwrap_or_default()
                    == memory_link.freeform_relation.as_deref().unwrap_or_default()
        }) {
            stored.link = memory_link.clone();
        } else {
            let sequence = state.next_link_sequence;
            state.next_link_sequence = state.next_link_sequence.saturating_add(1);
            state.links.push(StoredLink {
                link: memory_link.clone(),
                sequence,
            });
        }
        Ok(memory_link)
    }

    async fn delete(&self, index: &MemoryIndex) -> Result<(), PortError> {
        let mut state = self.lock()?;
        Self::delete_locked(&mut state, index);
        Ok(())
    }
}

fn normalize_label(value: &str) -> String {
    value
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_ascii_lowercase()
}

fn normalized_tag_filter(value: &str) -> Option<(Option<String>, String)> {
    let normalized = normalize_label(value);
    if normalized.is_empty() {
        return None;
    }
    let Some((namespace, label)) = normalized.split_once(':') else {
        return Some((None, normalized));
    };
    let label = label.trim();
    if label.is_empty() {
        return None;
    }
    let namespace = namespace.trim();
    Some((
        (!namespace.is_empty()).then(|| namespace.to_owned()),
        label.to_owned(),
    ))
}

fn clamp_confidence(value: f32) -> f32 {
    if value.is_finite() {
        value.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone as _;
    use nuillu_memory::{MemoryConcept, MemoryTag};
    use nuillu_types::MemoryContent;
    use std::rc::Rc;

    use crate::embedding::TestEmbedder;

    fn indexed(index: &str, content: &str, stored_at: DateTime<Utc>) -> IndexedMemory {
        IndexedMemory {
            index: MemoryIndex::new(index),
            content: MemoryContent::new(content),
            rank: MemoryRank::LongTerm,
            occurred_at: None,
            stored_at,
            kind: nuillu_memory::MemoryKind::Statement,
            concepts: Vec::new(),
            tags: Vec::new(),
            affect_arousal: 0.0,
            valence: 0.0,
            emotion: String::new(),
        }
    }

    #[tokio::test(flavor = "current_thread")]
    async fn search_filters_and_orders_records() {
        let store = InMemoryMemoryStore::new(Rc::new(TestEmbedder));
        let old_at = Utc.with_ymd_and_hms(2026, 1, 1, 0, 0, 0).unwrap();
        let new_at = Utc.with_ymd_and_hms(2026, 1, 2, 0, 0, 0).unwrap();
        store
            .put(IndexedMemory {
                concepts: vec![MemoryConcept::new("Project Alpha")],
                tags: vec![MemoryTag::operational("follow-up")],
                ..indexed("old", "alpha launch checklist", old_at)
            })
            .await
            .unwrap();
        store
            .put(indexed("new", "unrelated note", new_at))
            .await
            .unwrap();

        let records = store
            .search(&MemoryQuery {
                text: "alpha checklist".into(),
                offset: 0,
                limit: 4,
                kinds: vec![nuillu_memory::MemoryKind::Statement],
                concepts: vec![" project   alpha ".into()],
                tags: vec!["operation:follow-up".into()],
            })
            .await
            .unwrap();

        assert_eq!(
            records
                .iter()
                .map(|record| record.index.as_str())
                .collect::<Vec<_>>(),
            vec!["old"]
        );
        assert_eq!(
            store
                .list_recent(0, 2)
                .await
                .unwrap()
                .iter()
                .map(|record| record.index.as_str())
                .collect::<Vec<_>>(),
            vec!["new", "old"]
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn links_and_compaction_are_updated_atomically() {
        let store = InMemoryMemoryStore::new(Rc::new(TestEmbedder));
        let at = Utc.with_ymd_and_hms(2026, 1, 1, 0, 0, 0).unwrap();
        let first = MemoryIndex::new("first");
        let second = MemoryIndex::new("second");
        store.put(indexed("first", "first", at)).await.unwrap();
        store.put(indexed("second", "second", at)).await.unwrap();
        let link = store
            .upsert_link(
                NewMemoryLink {
                    from_memory: first.clone(),
                    to_memory: second.clone(),
                    relation: MemoryLinkRelation::Supports,
                    freeform_relation: None,
                    strength: 2.0,
                    confidence: f32::NAN,
                },
                at,
            )
            .await
            .unwrap();
        assert_eq!((link.strength, link.confidence), (1.0, 0.0));
        assert_eq!(
            store
                .linked(&LinkedMemoryQuery::around(vec![first.clone()], 4))
                .await
                .unwrap()[0]
                .record
                .index,
            second
        );

        let summary = store
            .put_compacted(
                indexed("summary", "summary", at),
                std::slice::from_ref(&first),
            )
            .await
            .unwrap();
        assert_eq!(summary.index.as_str(), "summary");
        assert!(store.get(&first).await.unwrap().is_none());
        assert!(
            store
                .linked(&LinkedMemoryQuery::around(vec![second], 4))
                .await
                .unwrap()
                .is_empty()
        );
    }
}
