use std::collections::HashSet;
use std::rc::Rc;

use nuillu_blackboard::{
    AllocationCommand, AllocationEffectKind, AllocationEffectPolicy, Blackboard, BlackboardCommand,
    ResourceAllocation,
};
use nuillu_types::{ModuleId, ModuleInstanceId};

use crate::allocation_persistence::{AllocationStore, PersistedAllocationSnapshot};
use crate::ports::PortError;

/// Submit owner-stamped allocation effect commands.
///
/// Capability issuers do not enforce uniqueness: capabilities are non-exclusive.
/// By boot-time wiring convention only the attention controller receives
/// this handle, but multiple writers are structurally permitted.
pub struct AllocationWriter {
    owner: ModuleInstanceId,
    blackboard: Blackboard,
    allowed_target_modules: Vec<ModuleId>,
    allowed_suppression_modules: Vec<ModuleId>,
    effect_policy: AllocationEffectPolicy,
    store: Rc<dyn AllocationStore>,
}

impl AllocationWriter {
    pub(crate) fn new(
        owner: ModuleInstanceId,
        blackboard: Blackboard,
        allowed_target_modules: Vec<ModuleId>,
        allowed_suppression_modules: Vec<ModuleId>,
        effect_policy: AllocationEffectPolicy,
        store: Rc<dyn AllocationStore>,
    ) -> Self {
        Self {
            owner,
            blackboard,
            allowed_target_modules,
            allowed_suppression_modules,
            effect_policy,
            store,
        }
    }

    /// Submit this writer's complete allocation effects.
    ///
    /// The command payload is semantic: modules choose an effect kind and
    /// level, while runtime policy resolves those levels to concrete activation
    /// ratios before the blackboard deterministically combines active writers.
    pub async fn submit(
        &self,
        commands: impl IntoIterator<Item = AllocationCommand>,
    ) -> Result<(), PortError> {
        let mut targets = ResourceAllocation::default();
        let mut suppressions = ResourceAllocation::default();

        for command in commands {
            match command.effect {
                AllocationEffectKind::Target => {
                    if !self.is_allowed(&command.module, &self.allowed_target_modules, "target") {
                        continue;
                    }
                    targets.set_activation(
                        command.module,
                        self.effect_policy.target_ratio(command.level),
                    );
                }
                AllocationEffectKind::Suppression => {
                    if !self.is_allowed(
                        &command.module,
                        &self.allowed_suppression_modules,
                        "suppression",
                    ) {
                        continue;
                    }
                    suppressions.set_activation(
                        command.module,
                        self.effect_policy.suppression_multiplier(command.level),
                    );
                }
            }
        }

        self.blackboard
            .apply(BlackboardCommand::RecordAllocationEffects {
                writer: self.owner.clone(),
                targets: targets.clone(),
                suppressions: suppressions.clone(),
            })
            .await;
        self.store
            .save(&PersistedAllocationSnapshot::new(
                self.owner.clone(),
                targets,
                suppressions,
            ))
            .await
    }

    pub fn allowed_target_modules(&self) -> &[ModuleId] {
        &self.allowed_target_modules
    }

    pub fn allowed_suppression_modules(&self) -> &[ModuleId] {
        &self.allowed_suppression_modules
    }

    fn is_allowed(
        &self,
        module: &ModuleId,
        allowed_modules: &[ModuleId],
        kind: &'static str,
    ) -> bool {
        let allowed = allowed_modules.iter().cloned().collect::<HashSet<_>>();
        if allowed.contains(module) {
            true
        } else {
            tracing::warn!(
                owner = %self.owner,
                allocation_kind = kind,
                module = %module,
                "allocation writer dropped disallowed module effect"
            );
            false
        }
    }
}
