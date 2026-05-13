//! Memory domain crate.
//!
//! Owns the [`MemoryStore`] port, the memory-domain capability handles
//! ([`MemoryWriter`], [`MemoryCompactor`], [`VectorMemorySearcher`],
//! [`MemoryContentReader`]), and the three modules that operate on memory:
//! [`MemoryModule`] (writes), [`MemoryCompactionModule`] (merges), and
//! [`MemoryRecombinationModule`] (dream-like recombination), and
//! [`QueryVectorModule`] (vector-search retrieval).
//!
//! Hosts build a [`MemoryCapabilities`] provider once at boot to bundle the
//! store + blackboard + clock, then either pass it to registration closures
//! to construct module-specific capability handles, or call
//! [`MemoryCapabilities::bootstrap_identity_memories`] to seed the
//! blackboard from the primary store before [`ModuleRegistry::build`] runs.
//!
//! [`ModuleRegistry::build`]: nuillu_module::ModuleRegistry::build

use std::sync::Arc;

use nuillu_blackboard::{Blackboard, BlackboardCommand};
use nuillu_module::ports::{Clock, PortError};
use nuillu_types::MemoryRank;

mod memory;
mod query;
mod recombination;
mod store;

pub use memory::{
    CompactionTools, CompactionToolsCall, CompactionToolsSelector, GetMemoriesArgs,
    GetMemoriesOutput, InsertMemoryArgs, InsertMemoryOutput, MemoryBatch, MemoryCompactionModule,
    MemoryContentView, MemoryModule, MemoryTools, MemoryToolsCall, MemoryToolsSelector,
    MergeMemoriesArgs, MergeMemoriesOutput,
};
pub use query::{
    QueryVectorBatch, QueryVectorMemo, QueryVectorMemoHit, QueryVectorMemoSearch,
    QueryVectorMemoryHit, QueryVectorModule, QueryVectorTools, QueryVectorToolsCall,
    QueryVectorToolsSelector, SearchVectorMemoryArgs, SearchVectorMemoryOutput,
};
pub use recombination::{
    AppendRecombinationArgs, AppendRecombinationOutput, MemoryRecombinationModule,
    RecombinationTools, RecombinationToolsCall, RecombinationToolsSelector,
};
pub use store::{
    IndexedMemory, MemoryCompactor, MemoryContentReader, MemoryQuery, MemoryRecord, MemoryStore,
    MemoryWriter, NewMemory, NoopMemoryStore, VectorMemorySearcher,
};

/// Domain-scoped capability provider for the memory subsystem.
///
/// Bundles the primary memory store, optional replica stores, the
/// blackboard, and the clock. Constructs the four memory-domain capability
/// handles on demand and seeds boot-time identity memories.
#[derive(Clone)]
pub struct MemoryCapabilities {
    primary_store: Arc<dyn MemoryStore>,
    replicas: Vec<Arc<dyn MemoryStore>>,
    blackboard: Blackboard,
    clock: Arc<dyn Clock>,
}

impl MemoryCapabilities {
    pub fn new(
        blackboard: Blackboard,
        clock: Arc<dyn Clock>,
        primary_store: Arc<dyn MemoryStore>,
        replicas: Vec<Arc<dyn MemoryStore>>,
    ) -> Self {
        Self {
            primary_store,
            replicas,
            blackboard,
            clock,
        }
    }

    pub fn primary_store(&self) -> &Arc<dyn MemoryStore> {
        &self.primary_store
    }

    pub fn writer(&self) -> MemoryWriter {
        MemoryWriter::new(
            self.primary_store.clone(),
            self.replicas.clone(),
            self.blackboard.clone(),
            self.clock.clone(),
        )
    }

    pub fn compactor(&self) -> MemoryCompactor {
        MemoryCompactor::new(
            self.primary_store.clone(),
            self.replicas.clone(),
            self.blackboard.clone(),
            self.clock.clone(),
        )
    }

    pub fn searcher(&self) -> VectorMemorySearcher {
        VectorMemorySearcher::new(
            self.primary_store.clone(),
            self.blackboard.clone(),
            self.clock.clone(),
        )
    }

    pub fn content_reader(&self) -> MemoryContentReader {
        MemoryContentReader::new(self.primary_store.clone())
    }

    /// Seed `Identity` rank memories onto the blackboard from the primary
    /// store. Hosts call this before `ModuleRegistry::build` so module
    /// constructors can synchronously read identity memories from the
    /// blackboard.
    pub async fn bootstrap_identity_memories(&self) -> Result<(), PortError> {
        let mut records = self
            .primary_store
            .list_by_rank(MemoryRank::Identity)
            .await?
            .into_iter()
            .map(|record| nuillu_blackboard::IdentityMemoryRecord {
                index: record.index,
                content: record.content,
                occurred_at: record.occurred_at,
            })
            .collect::<Vec<_>>();
        records.sort_by(|a, b| a.index.as_str().cmp(b.index.as_str()));
        self.blackboard
            .apply(BlackboardCommand::SetIdentityMemories(records))
            .await;
        Ok(())
    }
}
