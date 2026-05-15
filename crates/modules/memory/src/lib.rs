//! Memory domain crate.
//!
//! Owns the [`MemoryStore`] port, the memory-domain capability handles
//! ([`MemoryWriter`], [`MemoryCompactor`], [`MemoryAssociator`],
//! [`MemorySearcher`], [`MemoryContentReader`]), and the modules that
//! operate on memory: [`MemoryModule`] (writes), [`MemoryCompactionModule`]
//! (destructive merges), [`MemoryAssociationModule`] (non-destructive
//! relationship writes), [`MemoryRecombinationModule`] (dream-like
//! recombination), and [`QueryMemoryModule`] (vector-search retrieval).
//!
//! Hosts build a [`MemoryCapabilities`] provider once at boot to bundle the
//! store + blackboard + clock, then either pass it to registration closures
//! to construct module-specific capability handles, or call
//! [`MemoryCapabilities::bootstrap_identity_memories`] to seed the
//! blackboard from the primary store before [`ModuleRegistry::build`] runs.
//!
//! [`ModuleRegistry::build`]: nuillu_module::ModuleRegistry::build

use std::rc::Rc;

use nuillu_blackboard::{Blackboard, BlackboardCommand};
use nuillu_module::ports::{Clock, PortError};
use nuillu_types::MemoryRank;

mod association;
mod common;
mod compaction;
mod memory;
mod query;
mod recombination;
mod store;

pub use association::{
    AssociationLinkArgs, AssociationTools, AssociationToolsCall, AssociationToolsSelector,
    GetAssociationMemoriesArgs, MemoryAssociationModule, WriteAssociationSummaryArgs,
    WriteAssociationSummaryOutput, WriteMemoryLinksArgs, WriteMemoryLinksOutput,
};
pub use common::{GetMemoriesOutput, MemoryContentView};
pub use compaction::{
    CompactionTools, CompactionToolsCall, CompactionToolsSelector, GetMemoriesArgs,
    MemoryCompactionModule, MergeMemoriesArgs, MergeMemoriesOutput,
};
pub use memory::{
    InsertMemoryArgs, InsertMemoryOutput, MemoryBatch, MemoryConceptInput, MemoryModule,
    MemoryTagInput, MemoryTools, MemoryToolsCall, MemoryToolsSelector,
};
pub use query::{
    FetchLinkedMemoriesArgs, FetchLinkedMemoriesOutput, QueryMemoryBatch, QueryMemoryHit,
    QueryMemoryLinkedHit, QueryMemoryMemo, QueryMemoryMemoHit, QueryMemoryMemoLinkedHit,
    QueryMemoryMemoSearch, QueryMemoryModule, QueryMemoryTools, QueryMemoryToolsCall,
    QueryMemoryToolsSelector, SearchMemoryArgs, SearchMemoryOutput,
};
pub use recombination::{
    AppendRecombinationArgs, AppendRecombinationOutput, MemoryRecombinationModule,
    RecombinationTools, RecombinationToolsCall, RecombinationToolsSelector,
};
pub use store::{
    IndexedMemory, LinkedMemoryQuery, LinkedMemoryRecord, MemoryAssociator, MemoryCompactor,
    MemoryConcept, MemoryContentReader, MemoryDeleter, MemoryKind, MemoryLink, MemoryLinkDirection,
    MemoryLinkRelation, MemoryQuery, MemoryRecord, MemorySearcher, MemoryStore, MemoryTag,
    MemoryWriter, NewMemory, NewMemoryLink, NoopMemoryStore,
};

/// Domain-scoped capability provider for the memory subsystem.
///
/// Bundles the primary memory store, optional replica stores, the
/// blackboard, and the clock. Constructs the four memory-domain capability
/// handles on demand and seeds boot-time identity memories.
#[derive(Clone)]
pub struct MemoryCapabilities {
    primary_store: Rc<dyn MemoryStore>,
    replicas: Vec<Rc<dyn MemoryStore>>,
    blackboard: Blackboard,
    clock: Rc<dyn Clock>,
}

impl MemoryCapabilities {
    pub fn new(
        blackboard: Blackboard,
        clock: Rc<dyn Clock>,
        primary_store: Rc<dyn MemoryStore>,
        replicas: Vec<Rc<dyn MemoryStore>>,
    ) -> Self {
        Self {
            primary_store,
            replicas,
            blackboard,
            clock,
        }
    }

    pub fn primary_store(&self) -> &Rc<dyn MemoryStore> {
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

    pub fn associator(&self) -> MemoryAssociator {
        MemoryAssociator::new(
            self.primary_store.clone(),
            self.replicas.clone(),
            self.clock.clone(),
        )
    }

    pub fn searcher(&self) -> MemorySearcher {
        MemorySearcher::new(
            self.primary_store.clone(),
            self.blackboard.clone(),
            self.clock.clone(),
        )
    }

    pub fn content_reader(&self) -> MemoryContentReader {
        MemoryContentReader::new(self.primary_store.clone())
    }

    pub fn deleter(&self) -> MemoryDeleter {
        MemoryDeleter::new(
            self.primary_store.clone(),
            self.replicas.clone(),
            self.blackboard.clone(),
        )
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
