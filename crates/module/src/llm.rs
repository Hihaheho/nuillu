use lutum::Lutum;
use nuillu_blackboard::Blackboard;
use nuillu_types::ModuleId;

use crate::LutumTiers;

/// LLM-access capability.
///
/// Owner-stamped: `lutum()` reads the current
/// [`ResourceAllocation`](nuillu_blackboard::ResourceAllocation) for the
/// owning module and returns the [`Lutum`] handle bound to the
/// allocation's tier.
///
/// Modules consume the returned `Lutum` directly — typically by feeding
/// it into `lutum::Session::new(lutum)` for a single-turn or agent-loop
/// activation. The capability deliberately stops at the `Lutum`
/// boundary; everything past it (Session shape, prompts, tools) is the
/// module's own concern.
#[derive(Clone)]
pub struct LlmAccess {
    owner: ModuleId,
    tiers: LutumTiers,
    blackboard: Blackboard,
}

impl LlmAccess {
    pub(crate) fn new(owner: ModuleId, tiers: LutumTiers, blackboard: Blackboard) -> Self {
        Self {
            owner,
            tiers,
            blackboard,
        }
    }

    pub fn owner(&self) -> &ModuleId {
        &self.owner
    }

    /// The [`Lutum`] for the owner module's currently allocated tier.
    /// Tier resolution is per-call so allocation changes between
    /// activations take effect on the next call without re-issuing the
    /// capability.
    pub async fn lutum(&self) -> Lutum {
        let cfg = self
            .blackboard
            .read(|bb| bb.allocation().for_module(&self.owner))
            .await;
        self.tiers.pick(cfg.tier)
    }
}
