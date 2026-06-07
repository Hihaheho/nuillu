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
    pub kind: MemoryKind,
    pub concepts: Vec<MemoryConcept>,
    pub tags: Vec<MemoryTag>,
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
}

pub(crate) fn memory_record_to_view(record: MemoryRecord) -> MemoryContentView {
    MemoryContentView {
        index: record.index.to_string(),
        content: record.content.as_str().to_owned(),
        rank: record.rank,
        occurred_at: record.occurred_at,
        kind: record.kind,
        concepts: record.concepts,
        tags: record.tags,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn memory_content_view_schema_exposes_only_memory_evidence_fields() {
        let schema = serde_json::to_value(schemars::schema_for!(GetMemoriesOutput))
            .expect("get memories schema should serialize");
        let memory_properties = schema
            .pointer("/$defs/MemoryContentView/properties")
            .expect("memory content view properties should exist");

        assert_eq!(
            memory_properties.pointer("/index/type"),
            Some(&serde_json::json!("string"))
        );
        assert_eq!(
            memory_properties.pointer("/content/type"),
            Some(&serde_json::json!("string"))
        );
        assert_eq!(
            memory_properties.pointer("/rank/$ref"),
            Some(&serde_json::json!("#/$defs/MemoryRank"))
        );
        assert_eq!(
            memory_properties.pointer("/kind/$ref"),
            Some(&serde_json::json!("#/$defs/MemoryKind"))
        );
        assert!(memory_properties.pointer("/concepts").is_some());
        assert!(memory_properties.pointer("/tags").is_some());
        assert_eq!(memory_properties.pointer("/stored_at"), None);
        assert_eq!(memory_properties.pointer("/affect_arousal"), None);
        assert_eq!(memory_properties.pointer("/valence"), None);
        assert_eq!(memory_properties.pointer("/emotion"), None);
    }
}
