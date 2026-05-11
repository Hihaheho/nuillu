//! Agent runtime.
//!
//! The runtime spawns one persistent task per module on a single-threaded
//! `LocalSet` and waits for a shutdown signal. Modules drive their own
//! input loops via typed inbox capabilities and perform side effects
//! through other capabilities.

mod kicks;
pub mod scheduler;

#[cfg(test)]
mod testing;

pub use scheduler::{AgentEventLoopConfig, SchedulerError, run};
