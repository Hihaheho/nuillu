//! Read-only views over agent state.
//!
//! Each reader exposes only the slice of the blackboard the design
//! permits the holding module to see. The compile-time signal is the
//! constructor signature: a module that takes only [`AttentionReader`]
//! cannot read the non-cognitive blackboard, regardless of what its
//! `run` body tries.

use nuillu_blackboard::{AttentionStream, Blackboard, BlackboardInner, ResourceAllocation};

/// Read-only access to the entire blackboard (memos + memory metadata).
///
/// Held by modules that legitimately need a wide view (summarize,
/// query, memory, memory-compaction). Pointedly *not* held by the
/// attention controller, which is restricted to the cognitive surface.
#[derive(Clone)]
pub struct BlackboardReader {
    blackboard: Blackboard,
}

impl BlackboardReader {
    pub(crate) fn new(blackboard: Blackboard) -> Self {
        Self { blackboard }
    }

    /// Apply `f` to a borrowed snapshot. The read lock is held for the
    /// duration of `f`; do not await inside it.
    pub async fn read<R>(&self, f: impl FnOnce(&BlackboardInner) -> R) -> R {
        self.blackboard.read(f).await
    }
}

/// Read-only access to the cognitive attention stream. The holder
/// cannot see memos, memory metadata, or allocation through this
/// capability.
#[derive(Clone)]
pub struct AttentionReader {
    blackboard: Blackboard,
}

impl AttentionReader {
    pub(crate) fn new(blackboard: Blackboard) -> Self {
        Self { blackboard }
    }

    pub async fn read<R>(&self, f: impl FnOnce(&AttentionStream) -> R) -> R {
        self.blackboard.read(|bb| f(bb.attention_stream())).await
    }
}

/// Read-only access to the resource-allocation snapshot. Issued to
/// modules that need to consult allocation for their own decisions
/// (typically only the attention controller).
#[derive(Clone)]
pub struct AllocationReader {
    blackboard: Blackboard,
}

impl AllocationReader {
    pub(crate) fn new(blackboard: Blackboard) -> Self {
        Self { blackboard }
    }

    pub async fn read<R>(&self, f: impl FnOnce(&ResourceAllocation) -> R) -> R {
        self.blackboard.read(|bb| f(bb.allocation())).await
    }

    pub async fn snapshot(&self) -> ResourceAllocation {
        self.blackboard.read(|bb| bb.allocation().clone()).await
    }
}
