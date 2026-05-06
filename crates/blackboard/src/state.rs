use std::collections::HashMap;
use std::sync::Arc;

use nuillu_types::{MemoryIndex, ModuleId};
use tokio::sync::RwLock;

use crate::{AttentionStream, BlackboardCommand, MemoryMetadata, ResourceAllocation};

/// The non-cognitive blackboard plus the cognitive surface and its
/// allocation snapshot. This is a cheap cloneable handle; locking is an
/// implementation detail hidden behind its methods.
#[derive(Debug, Clone)]
pub struct Blackboard {
    inner: Arc<RwLock<BlackboardInner>>,
}

/// Inner blackboard state. Public so read closures in other crates can
/// inspect it, but its fields are private and mutations stay behind
/// [`BlackboardCommand`].
#[derive(Debug, Default)]
pub struct BlackboardInner {
    memos: HashMap<ModuleId, String>,
    attention_stream: AttentionStream,
    memory_metadata: HashMap<MemoryIndex, MemoryMetadata>,
    allocation: ResourceAllocation,
}

impl Blackboard {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(RwLock::new(BlackboardInner::default())),
        }
    }

    pub fn with_allocation(allocation: ResourceAllocation) -> Self {
        Self {
            inner: Arc::new(RwLock::new(BlackboardInner {
                allocation,
                ..BlackboardInner::default()
            })),
        }
    }

    /// Apply `f` to a borrowed snapshot. The read lock is held for the
    /// duration of `f`; do not await inside it.
    pub async fn read<R>(&self, f: impl FnOnce(&BlackboardInner) -> R) -> R {
        let guard = self.inner.read().await;
        f(&guard)
    }

    /// Apply one command under the blackboard write lock.
    pub async fn apply(&self, cmd: BlackboardCommand) {
        let mut guard = self.inner.write().await;
        guard.apply(cmd);
    }

    pub async fn memo(&self, id: &ModuleId) -> Option<String> {
        self.read(|bb| bb.memo(id).map(String::from)).await
    }
}

impl Default for Blackboard {
    fn default() -> Self {
        Self::new()
    }
}

impl BlackboardInner {
    pub fn memo(&self, id: &ModuleId) -> Option<&str> {
        self.memos.get(id).map(String::as_str)
    }

    pub fn memos(&self) -> &HashMap<ModuleId, String> {
        &self.memos
    }

    pub fn attention_stream(&self) -> &AttentionStream {
        &self.attention_stream
    }

    pub fn memory_metadata(&self) -> &HashMap<MemoryIndex, MemoryMetadata> {
        &self.memory_metadata
    }

    pub fn allocation(&self) -> &ResourceAllocation {
        &self.allocation
    }

    /// Apply one command. Mutations are localised to the matching arm so
    /// adding a variant is a compile error in any consumer that pattern-
    /// matches.
    fn apply(&mut self, cmd: BlackboardCommand) {
        match cmd {
            BlackboardCommand::UpdateMemo { module, memo } => {
                self.memos.insert(module, memo);
            }
            BlackboardCommand::AppendAttentionStream(event) => {
                self.attention_stream.append(event);
            }
            BlackboardCommand::UpsertMemoryMetadata {
                index,
                rank_if_new,
                decay_if_new_secs,
                patch,
            } => {
                let entry = self
                    .memory_metadata
                    .entry(index.clone())
                    .or_insert_with(|| MemoryMetadata::new(index, rank_if_new, decay_if_new_secs));
                patch.apply(entry);
            }
            BlackboardCommand::RemoveMemoryMetadata { index } => {
                self.memory_metadata.remove(&index);
            }
            BlackboardCommand::SetAllocation(alloc) => {
                self.allocation = alloc;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nuillu_types::builtin;

    #[tokio::test]
    async fn memo_round_trip() {
        let bb = Blackboard::new();
        let id = builtin::summarize();
        bb.apply(BlackboardCommand::UpdateMemo {
            module: id.clone(),
            memo: "noted".into(),
        })
        .await;
        assert_eq!(bb.memo(&id).await.as_deref(), Some("noted"));
    }
}
