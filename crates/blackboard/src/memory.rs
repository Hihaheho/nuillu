use std::collections::VecDeque;

use chrono::{DateTime, Utc};
use nuillu_types::{MemoryContent, MemoryIndex, MemoryRank};
use serde::{Deserialize, Serialize};

/// Startup-loaded identity memory content. This is a boot snapshot of
/// durable identity-ranked memory records, separate from the live metadata
/// mirror used for ordinary memory search/access bookkeeping.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IdentityMemoryRecord {
    pub index: MemoryIndex,
    pub content: MemoryContent,
}

/// Mutable metadata about a memory entry. Durable content and adapter-local
/// search state live in the external `MemoryStore`, not here.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryMetadata {
    pub index: MemoryIndex,
    pub rank: MemoryRank,
    pub decay_remaining_secs: i64,
    pub remember_tokens: u32,
    pub last_accessed: DateTime<Utc>,
    pub access_count: u32,
    /// Recent access timestamps, capped to 10 most recent.
    pub query_history: VecDeque<DateTime<Utc>>,
}

impl MemoryMetadata {
    pub fn new_at(
        index: MemoryIndex,
        rank: MemoryRank,
        decay_remaining_secs: i64,
        now: DateTime<Utc>,
    ) -> Self {
        Self {
            index,
            rank,
            decay_remaining_secs,
            remember_tokens: 0,
            last_accessed: now,
            access_count: 0,
            query_history: VecDeque::new(),
        }
    }

    pub fn record_access_at(&mut self, now: DateTime<Utc>) {
        self.last_accessed = now;
        self.access_count += 1;
        self.query_history.push_back(now);
        if self.query_history.len() > 10 {
            self.query_history.pop_front();
        }
    }
}

/// Partial update to memory metadata. Fields set to `Some(_)` are applied;
/// `None` is ignored.
#[derive(Debug, Clone, Default)]
pub struct MemoryMetaPatch {
    pub rank: Option<MemoryRank>,
    pub decay_remaining_secs: Option<i64>,
    pub increment_remember_tokens: Option<u32>,
    pub record_access: bool,
}

impl MemoryMetaPatch {
    pub fn apply_at(&self, meta: &mut MemoryMetadata, now: DateTime<Utc>) {
        if let Some(r) = self.rank {
            meta.rank = r;
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
    }
}
