use std::collections::VecDeque;

use chrono::{DateTime, Utc};
use nuillu_types::{MemoryContent, MemoryIndex, MemoryRank};
use serde::{Deserialize, Serialize};

const HISTORY_LIMIT: usize = 10;

/// Startup-loaded identity memory content. This is a boot snapshot of
/// durable identity-ranked memory records, separate from the live metadata
/// mirror used for ordinary memory search/access bookkeeping.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IdentityMemoryRecord {
    pub index: MemoryIndex,
    pub content: MemoryContent,
    pub occurred_at: Option<DateTime<Utc>>,
}

/// Mutable metadata about a memory entry. Durable content and adapter-local
/// search state live in the external `MemoryStore`, not here.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryMetadata {
    pub index: MemoryIndex,
    pub rank: MemoryRank,
    pub occurred_at: Option<DateTime<Utc>>,
    pub decay_remaining_secs: i64,
    pub remember_tokens: u32,
    pub last_accessed: DateTime<Utc>,
    pub access_count: u32,
    #[serde(default)]
    pub last_used: Option<DateTime<Utc>>,
    #[serde(default)]
    pub use_count: u32,
    #[serde(default)]
    pub last_reinforced_at: Option<DateTime<Utc>>,
    #[serde(default)]
    pub reinforcement_count: u32,
    /// Recent access timestamps, capped to 10 most recent. Kept as
    /// `query_history` for serialized-state compatibility.
    pub query_history: VecDeque<DateTime<Utc>>,
    /// Recent use timestamps, capped to 10 most recent.
    #[serde(default)]
    pub use_history: VecDeque<DateTime<Utc>>,
    /// Recent reinforcement timestamps, capped to 10 most recent.
    #[serde(default)]
    pub reinforcement_history: VecDeque<DateTime<Utc>>,
}

impl MemoryMetadata {
    pub fn new_at(
        index: MemoryIndex,
        rank: MemoryRank,
        occurred_at: Option<DateTime<Utc>>,
        decay_remaining_secs: i64,
        now: DateTime<Utc>,
    ) -> Self {
        Self {
            index,
            rank,
            occurred_at,
            decay_remaining_secs,
            remember_tokens: 0,
            last_accessed: now,
            access_count: 0,
            last_used: None,
            use_count: 0,
            last_reinforced_at: None,
            reinforcement_count: 0,
            query_history: VecDeque::new(),
            use_history: VecDeque::new(),
            reinforcement_history: VecDeque::new(),
        }
    }

    pub fn record_access_at(&mut self, now: DateTime<Utc>) {
        self.last_accessed = now;
        self.access_count += 1;
        push_capped(&mut self.query_history, now);
    }

    pub fn record_use_at(&mut self, now: DateTime<Utc>) {
        self.last_used = Some(now);
        self.use_count += 1;
        push_capped(&mut self.use_history, now);
    }

    pub fn record_reinforcement_at(&mut self, now: DateTime<Utc>) {
        self.last_reinforced_at = Some(now);
        self.reinforcement_count += 1;
        push_capped(&mut self.reinforcement_history, now);
    }
}

fn push_capped(history: &mut VecDeque<DateTime<Utc>>, at: DateTime<Utc>) {
    history.push_back(at);
    if history.len() > HISTORY_LIMIT {
        history.pop_front();
    }
}

/// Partial update to memory metadata. Fields set to `Some(_)` are applied;
/// `None` is ignored.
#[derive(Debug, Clone, Default)]
pub struct MemoryMetaPatch {
    pub rank: Option<MemoryRank>,
    pub occurred_at: Option<Option<DateTime<Utc>>>,
    pub decay_remaining_secs: Option<i64>,
    pub increment_remember_tokens: Option<u32>,
    pub record_access: bool,
    pub record_use: bool,
    pub record_reinforcement: bool,
}

impl MemoryMetaPatch {
    pub fn apply_at(&self, meta: &mut MemoryMetadata, now: DateTime<Utc>) {
        if let Some(r) = self.rank {
            meta.rank = r;
        }
        if let Some(occurred_at) = self.occurred_at {
            meta.occurred_at = occurred_at;
        }
        if let Some(d) = self.decay_remaining_secs {
            meta.decay_remaining_secs = d;
        }
        if let Some(n) = self.increment_remember_tokens {
            meta.remember_tokens = meta.remember_tokens.saturating_add(n);
        }
        if self.record_access {
            meta.record_access_at(now);
        }
        if self.record_use {
            meta.record_use_at(now);
        }
        if self.record_reinforcement {
            meta.record_reinforcement_at(now);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use chrono::{TimeZone, Utc};

    fn now() -> DateTime<Utc> {
        Utc.with_ymd_and_hms(2026, 5, 16, 0, 0, 0).unwrap()
    }

    #[test]
    fn metadata_records_access_use_and_reinforcement_separately() {
        let at = now();
        let mut metadata = MemoryMetadata::new_at(
            MemoryIndex::new("memory-1"),
            MemoryRank::ShortTerm,
            None,
            10,
            at,
        );

        metadata.record_access_at(at);
        metadata.record_use_at(at);
        metadata.record_reinforcement_at(at);

        assert_eq!(metadata.access_count, 1);
        assert_eq!(metadata.use_count, 1);
        assert_eq!(metadata.reinforcement_count, 1);
        assert_eq!(metadata.last_accessed, at);
        assert_eq!(metadata.last_used, Some(at));
        assert_eq!(metadata.last_reinforced_at, Some(at));
    }

    #[test]
    fn metadata_histories_are_capped() {
        let at = now();
        let mut metadata = MemoryMetadata::new_at(
            MemoryIndex::new("memory-1"),
            MemoryRank::ShortTerm,
            None,
            10,
            at,
        );

        for offset in 0..12 {
            let timestamp = at + chrono::Duration::seconds(offset);
            metadata.record_access_at(timestamp);
            metadata.record_use_at(timestamp);
            metadata.record_reinforcement_at(timestamp);
        }

        assert_eq!(metadata.query_history.len(), HISTORY_LIMIT);
        assert_eq!(metadata.use_history.len(), HISTORY_LIMIT);
        assert_eq!(metadata.reinforcement_history.len(), HISTORY_LIMIT);
        assert_eq!(
            metadata.query_history.front(),
            Some(&(at + chrono::Duration::seconds(2)))
        );
        assert_eq!(
            metadata.use_history.front(),
            Some(&(at + chrono::Duration::seconds(2)))
        );
        assert_eq!(
            metadata.reinforcement_history.front(),
            Some(&(at + chrono::Duration::seconds(2)))
        );
    }

    #[test]
    fn metadata_deserializes_without_new_usage_fields() {
        let at = now();
        let metadata = MemoryMetadata::new_at(
            MemoryIndex::new("memory-1"),
            MemoryRank::ShortTerm,
            None,
            10,
            at,
        );
        let mut value = serde_json::to_value(metadata).unwrap();
        let object = value.as_object_mut().unwrap();
        object.remove("last_used");
        object.remove("use_count");
        object.remove("last_reinforced_at");
        object.remove("reinforcement_count");
        object.remove("use_history");
        object.remove("reinforcement_history");

        let decoded: MemoryMetadata = serde_json::from_value(value).unwrap();

        assert_eq!(decoded.last_used, None);
        assert_eq!(decoded.use_count, 0);
        assert_eq!(decoded.last_reinforced_at, None);
        assert_eq!(decoded.reinforcement_count, 0);
        assert!(decoded.use_history.is_empty());
        assert!(decoded.reinforcement_history.is_empty());
    }
}
