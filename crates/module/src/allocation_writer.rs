use std::collections::HashSet;

use nuillu_blackboard::{Blackboard, BlackboardCommand, ResourceAllocation};
use nuillu_types::{ModuleId, ModuleInstanceId};

use crate::{AllocationUpdated, AllocationUpdatedMailbox};

/// Replace the agent's resource-allocation snapshot.
///
/// Capability issuers do not enforce uniqueness: capabilities are non-exclusive.
/// By boot-time wiring convention only the attention controller receives
/// this handle, but multiple writers are structurally permitted.
pub struct AllocationWriter {
    owner: ModuleInstanceId,
    blackboard: Blackboard,
    updates: AllocationUpdatedMailbox,
    allowed_drive_modules: Vec<ModuleId>,
    allowed_cap_modules: Vec<ModuleId>,
}

impl AllocationWriter {
    pub(crate) fn new(
        owner: ModuleInstanceId,
        blackboard: Blackboard,
        updates: AllocationUpdatedMailbox,
        allowed_drive_modules: Vec<ModuleId>,
        allowed_cap_modules: Vec<ModuleId>,
    ) -> Self {
        Self {
            owner,
            blackboard,
            updates,
            allowed_drive_modules,
            allowed_cap_modules,
        }
    }

    /// Record this controller replica's proposal. The runtime derives the
    /// effective allocation from active controller proposals.
    pub async fn set_drive(&self, mut allocation: ResourceAllocation) {
        self.retain_allowed(&mut allocation, &self.allowed_drive_modules, "drive");
        let before = self.blackboard.read(|bb| bb.allocation().clone()).await;
        self.blackboard
            .apply(BlackboardCommand::RecordAllocationProposal {
                controller: self.owner.clone(),
                proposal: allocation,
            })
            .await;
        let after = self.blackboard.read(|bb| bb.allocation().clone()).await;
        if before != after && self.updates.publish(AllocationUpdated).await.is_err() {
            tracing::trace!("allocation update had no active subscribers");
        }
    }

    /// Backwards-compatible name for drive proposals.
    pub async fn set(&self, allocation: ResourceAllocation) {
        self.set_drive(allocation).await;
    }

    /// Record activation ceilings. The runtime applies these after drive
    /// proposal merging by clamping matching module activations downward.
    pub async fn set_cap(&self, mut cap: ResourceAllocation) {
        self.retain_allowed(&mut cap, &self.allowed_cap_modules, "cap");
        let before = self.blackboard.read(|bb| bb.allocation().clone()).await;
        self.blackboard
            .apply(BlackboardCommand::RecordAllocationCap {
                controller: self.owner.clone(),
                cap,
            })
            .await;
        let after = self.blackboard.read(|bb| bb.allocation().clone()).await;
        if before != after && self.updates.publish(AllocationUpdated).await.is_err() {
            tracing::trace!("allocation update had no active subscribers");
        }
    }

    pub fn allowed_drive_modules(&self) -> &[ModuleId] {
        &self.allowed_drive_modules
    }

    pub fn allowed_cap_modules(&self) -> &[ModuleId] {
        &self.allowed_cap_modules
    }

    fn retain_allowed(
        &self,
        allocation: &mut ResourceAllocation,
        allowed_modules: &[ModuleId],
        kind: &'static str,
    ) {
        let allowed = allowed_modules.iter().cloned().collect::<HashSet<_>>();
        let dropped = allocation
            .iter()
            .map(|(id, _)| id.clone())
            .chain(allocation.iter_activation().map(|(id, _)| id.clone()))
            .filter(|id| !allowed.contains(id))
            .collect::<HashSet<_>>();
        if !dropped.is_empty() {
            let mut dropped = dropped.into_iter().collect::<Vec<_>>();
            dropped.sort_by(|left, right| left.as_str().cmp(right.as_str()));
            tracing::warn!(
                owner = %self.owner,
                allocation_kind = kind,
                dropped = ?dropped,
                "allocation writer dropped disallowed module updates"
            );
        }
        allocation.retain_modules(&allowed);
    }
}
