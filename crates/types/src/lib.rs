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
mod scalars;

pub use ids::{ModuleId, ModuleIdParseError, builtin};
pub use memory::{MemoryContent, MemoryIndex, MemoryRank};
pub use model::{ModelTier, TokenBudget};
pub use scalars::{UnitF32, UnitF32Error};
