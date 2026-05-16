use chrono::{DateTime, Utc};
use nuillu_types::MemoryRank;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::store::{MemoryConcept, MemoryKind, MemoryRecord, MemoryTag};

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct MemoryContentView {
    pub index: String,
    pub content: String,
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

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct GetMemoriesOutput {
    pub memories: Vec<MemoryContentView>,
}

#[derive(Clone, Debug)]
pub(crate) struct MemoryMetadataContext {
    pub(crate) index: String,
    pub(crate) rank: MemoryRank,
    pub(crate) occurred_at: String,
    pub(crate) decay_remaining_secs: i64,
    pub(crate) access_count: u32,
    pub(crate) use_count: u32,
    pub(crate) reinforcement_count: u32,
}

pub(crate) fn memory_record_to_view(record: MemoryRecord) -> MemoryContentView {
    MemoryContentView {
        index: record.index.to_string(),
        content: record.content.as_str().to_owned(),
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
