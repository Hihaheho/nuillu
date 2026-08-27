//! Shared newtypes and value objects for nuillu.
//!
//! This crate must contain **no business logic** — only:
//! - newtypes / wrappers (IDs, ranks, clamped scalars)
//! - small value structs that are passed by value or reference
//!
//! Anything with behaviour belongs in `blackboard`, `module`, or `agent`.

mod ids;
mod memory;
mod model;
mod policy;
mod scalars;

pub use ids::{
    ModuleActivationId, ModuleGroupId, ModuleGroupIdParseError, ModuleId, ModuleIdParseError,
    ModuleInstanceId, ModuleInstanceStorageParseError, ReplicaCapRange, ReplicaCapRangeError,
    ReplicaIndex, ScopeId, ScopedModuleId, SubsystemId, SubsystemIdParseError, SubsystemInstanceId,
    builtin,
};
pub use memory::{MemoryContent, MemoryIndex, MemoryRank};
pub use model::{ModelTier, TokenBudget};
pub use policy::{PolicyIndex, PolicyRank};
pub use scalars::{SignedUnitF32, SignedUnitF32Error, UnitF32, UnitF32Error};
