use std::collections::HashMap;

use nuillu_blackboard::{
    ActivationRatio, AllocationEffectLevel, Blackboard, BlackboardCommand,
    SubsystemAllocationCommand,
};
use nuillu_types::{ModuleInstanceId, SubsystemId};

use crate::ports::PortError;

#[derive(Debug, Clone, PartialEq)]
pub struct SubsystemAllocationView {
    pub subsystem: SubsystemId,
    pub local_activation: nuillu_blackboard::ActivationRatio,
    pub effective_activation: nuillu_blackboard::ActivationRatio,
    pub active_replicas: u8,
    pub replica_min: u8,
    pub replica_max: u8,
    pub replica_capacity: u8,
}

#[derive(Clone)]
pub struct SubsystemAllocationReader {
    blackboard: Blackboard,
}

impl SubsystemAllocationReader {
    pub(crate) fn new(blackboard: Blackboard) -> Self {
        Self { blackboard }
    }

    pub async fn snapshot(&self) -> Vec<SubsystemAllocationView> {
        let parent_activation = self
            .blackboard
            .scope_activation_state(self.blackboard.scope())
            .await
            .effective_activation;
        self.blackboard
            .read(|bb| {
                let mut views = bb
                    .subsystem_policies()
                    .iter()
                    .map(|(subsystem, policy)| {
                        let local_activation = bb.subsystem_allocation().activation_for(subsystem);
                        let input = nuillu_blackboard::ActivationInput::new(
                            local_activation,
                            parent_activation,
                        );
                        SubsystemAllocationView {
                            subsystem: subsystem.clone(),
                            local_activation,
                            effective_activation: input.effective,
                            active_replicas: policy.active_replicas_for(input),
                            replica_min: policy.replicas_range.min,
                            replica_max: policy.replicas_range.max,
                            replica_capacity: policy.replica_capacity,
                        }
                    })
                    .collect::<Vec<_>>();
                views.sort_by(|left, right| left.subsystem.as_str().cmp(right.subsystem.as_str()));
                views
            })
            .await
    }
}

/// Owner-stamped writer restricted to immediate child mounts selected by host
/// wiring. Capabilities remain non-exclusive.
pub struct SubsystemAllocationWriter {
    owner: ModuleInstanceId,
    blackboard: Blackboard,
    activation_tables: HashMap<SubsystemId, Vec<ActivationRatio>>,
}

impl SubsystemAllocationWriter {
    pub(crate) fn new(
        owner: ModuleInstanceId,
        blackboard: Blackboard,
        activation_tables: impl IntoIterator<Item = (SubsystemId, Vec<ActivationRatio>)>,
    ) -> Self {
        Self {
            owner,
            blackboard,
            activation_tables: activation_tables.into_iter().collect(),
        }
    }

    pub async fn submit(
        &self,
        commands: impl IntoIterator<Item = SubsystemAllocationCommand>,
    ) -> Result<(), PortError> {
        let mut requested = HashMap::new();
        for command in commands {
            if !self.activation_tables.contains_key(&command.subsystem) {
                tracing::warn!(
                    owner = %self.owner,
                    subsystem = %command.subsystem,
                    "subsystem allocation writer dropped disallowed target"
                );
                continue;
            }
            requested.entry(command.subsystem).or_insert(command.level);
        }
        for (subsystem, table) in &self.activation_tables {
            let activation = requested
                .get(subsystem)
                .and_then(|level| table.get(level_index(*level)))
                .copied()
                .unwrap_or(ActivationRatio::ZERO);
            self.blackboard
                .apply(BlackboardCommand::SetSubsystemActivation {
                    writer: self.owner.clone(),
                    subsystem: subsystem.clone(),
                    activation,
                })
                .await;
        }
        Ok(())
    }

    pub fn allowed_subsystems(&self) -> Vec<SubsystemId> {
        let mut allowed = self.activation_tables.keys().cloned().collect::<Vec<_>>();
        allowed.sort_by(|left, right| left.as_str().cmp(right.as_str()));
        allowed
    }
}

fn level_index(level: AllocationEffectLevel) -> usize {
    match level {
        AllocationEffectLevel::Max => 0,
        AllocationEffectLevel::High => 1,
        AllocationEffectLevel::Normal => 2,
        AllocationEffectLevel::Low => 3,
        AllocationEffectLevel::Minimal => 4,
        AllocationEffectLevel::Off => 5,
    }
}
