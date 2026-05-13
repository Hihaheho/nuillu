use std::sync::Arc;

use nuillu_blackboard::{
    ActivationRatio, Bpm, ModuleConfig, ModulePolicy, ResourceAllocation, linear_ratio_fn,
};
use nuillu_memory::MemoryCapabilities;
use nuillu_module::ModuleRegistry;
use nuillu_query_agentic::FileSearchProvider;
use nuillu_reward::PolicyCapabilities;
use nuillu_speak::{UtteranceSink, UtteranceWriter};
use nuillu_types::{ModelTier, ModuleId, ReplicaCapRange, builtin};

use super::config::ServerModule;

pub(super) fn server_registry(
    modules: &[ServerModule],
    memory_caps: &MemoryCapabilities,
    policy_caps: &PolicyCapabilities,
    file_search: &Arc<dyn FileSearchProvider>,
    utterance_sink: &Arc<dyn UtteranceSink>,
) -> ModuleRegistry {
    let mut registry = ModuleRegistry::new();
    for module in modules {
        registry = register_server_module(
            registry,
            *module,
            modules,
            memory_caps,
            policy_caps,
            file_search,
            utterance_sink,
        );
    }
    declare_dependencies(registry, modules)
}

trait ServerRegistryExt {
    fn register_server<B>(self, policy: ModulePolicy, builder: B) -> ModuleRegistry
    where
        B: nuillu_module::ModuleRegisterer + 'static;
}

impl ServerRegistryExt for ModuleRegistry {
    fn register_server<B>(self, policy: ModulePolicy, builder: B) -> ModuleRegistry
    where
        B: nuillu_module::ModuleRegisterer + 'static,
    {
        self.register_with_replica_capacity(policy, ReplicaCapRange::V1_MAX, builder)
            .expect("server module registration should be unique")
    }
}

fn declare_dependencies(registry: ModuleRegistry, modules: &[ServerModule]) -> ModuleRegistry {
    let present = modules
        .iter()
        .copied()
        .map(ServerModule::module_id)
        .collect::<std::collections::HashSet<_>>();
    let edges = [
        (builtin::speak_gate(), builtin::cognition_gate()),
        (builtin::self_model(), builtin::query_vector()),
        (builtin::cognition_gate(), builtin::sensory()),
        (builtin::cognition_gate(), builtin::query_vector()),
        (builtin::cognition_gate(), builtin::query_policy()),
        (builtin::cognition_gate(), builtin::self_model()),
        (builtin::cognition_gate(), builtin::surprise()),
        (builtin::value_estimator(), builtin::query_policy()),
        (builtin::reward(), builtin::value_estimator()),
        (builtin::policy(), builtin::reward()),
        (
            builtin::memory_recombination(),
            builtin::memory_compaction(),
        ),
    ];
    edges
        .into_iter()
        .fold(registry, |registry, (dependent, dependency)| {
            if present.contains(&dependent) && present.contains(&dependency) {
                registry.depends_on(dependent, dependency)
            } else {
                registry
            }
        })
}

fn register_server_module(
    registry: ModuleRegistry,
    module: ServerModule,
    all_modules: &[ServerModule],
    memory_caps: &MemoryCapabilities,
    policy_caps: &PolicyCapabilities,
    _file_search: &Arc<dyn FileSearchProvider>,
    utterance_sink: &Arc<dyn UtteranceSink>,
) -> ModuleRegistry {
    match module {
        ServerModule::Sensory => {
            registry.register_server(policy(1..=1, Bpm::range(3.0, 8.0)), |caps| {
                nuillu_sensory::SensoryModule::new(
                    caps.sensory_input_inbox(),
                    caps.allocation_reader(),
                    caps.memo(),
                    caps.clock(),
                    caps.llm_access(),
                )
            })
        }
        ServerModule::CognitionGate => {
            registry.register_server(policy(1..=1, Bpm::range(6.0, 12.0)), |caps| {
                nuillu_cognition_gate::CognitionGateModule::new(
                    caps.memo_updated_inbox(),
                    caps.allocation_updated_inbox(),
                    caps.blackboard_reader(),
                    caps.allocation_reader(),
                    caps.cognition_writer(),
                    caps.time_division(),
                    caps.llm_access(),
                )
            })
        }
        ServerModule::AttentionController => {
            let voluntary = voluntary_modules(all_modules);
            registry.register_server(policy(1..=1, Bpm::range(6.0, 6.0)), move |caps| {
                nuillu_attention_controller::AttentionControllerModule::new(
                    caps.memo_updated_inbox(),
                    caps.attention_control_inbox(),
                    caps.blackboard_reader(),
                    caps.cognition_log_reader(),
                    caps.allocation_reader(),
                    caps.allocation_writer(voluntary.clone(), Vec::new()),
                    caps.memo(),
                    caps.llm_access(),
                )
            })
        }
        ServerModule::AttentionSchema => {
            registry.register_server(policy(0..=1, Bpm::range(3.0, 6.0)), |caps| {
                nuillu_attention_schema::AttentionSchemaModule::new(
                    caps.memo_updated_inbox(),
                    caps.allocation_updated_inbox(),
                    caps.cognition_log_updated_inbox(),
                    caps.blackboard_reader(),
                    caps.allocation_reader(),
                    caps.cognition_log_reader(),
                    caps.cognition_writer(),
                    caps.llm_access(),
                )
            })
        }
        ServerModule::SelfModel => {
            registry.register_server(policy(0..=1, Bpm::range(3.0, 6.0)), |caps| {
                nuillu_self_model::SelfModelModule::new(
                    caps.allocation_updated_inbox(),
                    caps.allocation_reader(),
                    caps.blackboard_reader(),
                    caps.cognition_log_reader(),
                    caps.memo(),
                    caps.llm_access(),
                )
            })
        }
        ServerModule::QueryVector => {
            let memory_caps = memory_caps.clone();
            registry.register_server(policy(0..=1, Bpm::range(6.0, 15.0)), move |caps| {
                nuillu_memory::QueryVectorModule::new(
                    caps.allocation_updated_inbox(),
                    caps.cognition_log_updated_inbox(),
                    caps.allocation_reader(),
                    caps.blackboard_reader(),
                    memory_caps.searcher(),
                    caps.typed_memo::<nuillu_memory::QueryVectorMemo>(),
                    caps.llm_access(),
                )
            })
        }
        ServerModule::QueryPolicy => {
            let policy_caps = policy_caps.clone();
            registry.register_server(policy(0..=1, Bpm::range(6.0, 15.0)), move |caps| {
                nuillu_reward::QueryPolicyModule::new(
                    caps.allocation_updated_inbox(),
                    caps.cognition_log_updated_inbox(),
                    caps.allocation_reader(),
                    caps.blackboard_reader(),
                    policy_caps.searcher(),
                    caps.typed_memo::<nuillu_reward::PolicyRetrievalMemo>(),
                    caps.llm_access(),
                )
            })
        }
        ServerModule::Memory => {
            let memory_caps = memory_caps.clone();
            registry.register_server(policy(0..=1, Bpm::range(6.0, 18.0)), move |caps| {
                nuillu_memory::MemoryModule::new(
                    caps.cognition_log_updated_inbox(),
                    caps.allocation_updated_inbox(),
                    caps.allocation_reader(),
                    caps.blackboard_reader(),
                    memory_caps.writer(),
                    caps.llm_access(),
                )
            })
        }
        ServerModule::MemoryCompaction => {
            let memory_caps = memory_caps.clone();
            registry.register_server(policy(0..=1, Bpm::range(2.0, 6.0)), move |caps| {
                nuillu_memory::MemoryCompactionModule::new(
                    caps.allocation_updated_inbox(),
                    caps.allocation_reader(),
                    caps.blackboard_reader(),
                    memory_caps.compactor(),
                    caps.llm_access(),
                )
            })
        }
        ServerModule::MemoryRecombination => {
            let memory_caps = memory_caps.clone();
            registry.register_server(policy(0..=1, Bpm::range(2.0, 6.0)), move |caps| {
                nuillu_memory::MemoryRecombinationModule::new(
                    caps.allocation_updated_inbox(),
                    caps.allocation_reader(),
                    caps.blackboard_reader(),
                    memory_caps.searcher(),
                    caps.cognition_writer(),
                    caps.llm_access(),
                )
            })
        }
        ServerModule::Vital => {
            registry.register_server(policy(1..=1, Bpm::range(1.0, 3.0)), |caps| {
                nuillu_vital::VitalModule::new(
                    caps.cognition_log_updated_inbox(),
                    caps.allocation_updated_inbox(),
                    caps.blackboard_reader(),
                    caps.vital_writer(),
                )
            })
        }
        ServerModule::HomeostaticController => {
            registry.register_server(policy(1..=1, Bpm::range(6.0, 20.0)), |caps| {
                nuillu_homeostatic_controller::HomeostaticControllerModule::new(
                    caps.vital_updated_inbox(),
                    caps.vital_reader(),
                    caps.allocation_writer(
                        homeostatic_drive_modules(),
                        homeostatic_capped_modules(),
                    ),
                )
            })
        }
        ServerModule::Policy => {
            let policy_caps = policy_caps.clone();
            registry.register_server(policy(1..=1, Bpm::range(2.0, 6.0)), move |caps| {
                nuillu_reward::PolicyModule::new(
                    caps.cognition_log_updated_inbox(),
                    caps.allocation_updated_inbox(),
                    caps.blackboard_reader(),
                    caps.allocation_reader(),
                    policy_caps.writer(),
                    caps.llm_access(),
                )
            })
        }
        ServerModule::ValueEstimator => {
            let policy_caps = policy_caps.clone();
            registry.register_server(policy(1..=1, Bpm::range(2.0, 6.0)), move |caps| {
                nuillu_reward::ValueEstimatorModule::new(
                    caps.memo_updated_inbox(),
                    caps.cognition_log_updated_inbox(),
                    caps.allocation_updated_inbox(),
                    caps.blackboard_reader(),
                    caps.cognition_log_reader(),
                    caps.allocation_reader(),
                    policy_caps.window_reader(),
                    caps.typed_memo::<nuillu_reward::ValueEstimateMemo>(),
                    caps.llm_access(),
                )
            })
        }
        ServerModule::Reward => {
            let policy_caps = policy_caps.clone();
            registry.register_server(policy(0..=1, Bpm::range(1.0, 2.0)), move |caps| {
                nuillu_reward::RewardModule::new(
                    caps.cognition_log_updated_inbox(),
                    caps.memo_updated_inbox(),
                    caps.allocation_updated_inbox(),
                    caps.blackboard_reader(),
                    caps.cognition_log_reader(),
                    caps.allocation_reader(),
                    policy_caps.window_reader(),
                    policy_caps.value_updater(),
                    caps.attention_control_mailbox(),
                    caps.typed_memo::<nuillu_reward::RewardMemo>(),
                    caps.llm_access(),
                )
            })
        }
        ServerModule::Predict => {
            registry.register_server(policy(0..=1, Bpm::range(1.0, 6.0)), |caps| {
                nuillu_predict::PredictModule::new(
                    caps.cognition_log_updated_inbox(),
                    caps.cognition_log_reader(),
                    caps.allocation_reader(),
                    caps.blackboard_reader(),
                    caps.memo(),
                    caps.llm_access(),
                )
            })
        }
        ServerModule::Surprise => {
            registry.register_server(policy(0..=1, Bpm::range(1.0, 3.0)), |caps| {
                nuillu_surprise::SurpriseModule::new(
                    caps.cognition_log_updated_inbox(),
                    caps.cognition_log_reader(),
                    caps.allocation_reader(),
                    caps.blackboard_reader(),
                    caps.attention_control_mailbox(),
                    caps.memo(),
                    caps.llm_access(),
                )
            })
        }
        ServerModule::SpeakGate => {
            registry.register_server(policy(0..=1, Bpm::range(3.0, 6.0)), |caps| {
                nuillu_speak::SpeakGateModule::new(
                    caps.activation_gate_for::<nuillu_speak::SpeakModule>(),
                    caps.cognition_log_reader(),
                    caps.blackboard_reader(),
                    caps.attention_control_mailbox(),
                    caps.typed_memo::<nuillu_speak::SpeakGateMemo>(),
                    caps.llm_access(),
                )
            })
        }
        ServerModule::Speak => {
            let utterance_sink = utterance_sink.clone();
            registry.register_server(policy(0..=1, Bpm::range(3.0, 6.0)), move |caps| {
                nuillu_speak::SpeakModule::new(
                    caps.cognition_log_updated_inbox(),
                    caps.cognition_log_reader(),
                    caps.memo(),
                    UtteranceWriter::new(
                        caps.owner().clone(),
                        caps.blackboard(),
                        utterance_sink.clone(),
                        caps.clock(),
                    ),
                    caps.llm_access(),
                    caps.scene_reader(),
                )
            })
        }
    }
}

pub(super) fn full_agent_allocation(modules: &[ServerModule]) -> ResourceAllocation {
    let mut allocation = ResourceAllocation::default();
    allocation.set_activation_table(activation_table());
    for module in modules {
        let (activation, tier, guidance) = match module {
            ServerModule::Sensory => (
                1.0,
                ModelTier::Cheap,
                "Queued sensory input is waiting; activate when the controller is ready to process external observations.",
            ),
            ServerModule::CognitionGate => (
                1.0,
                ModelTier::Cheap,
                "Wait for memo or controller guidance before promoting relevant memos into cognition.",
            ),
            ServerModule::AttentionController => (
                1.0,
                ModelTier::Default,
                "Bootstrap live interaction: activate sensory first, then allocate cognition, query, and speech modules as evidence becomes ready.",
            ),
            ServerModule::AttentionSchema => (
                0.0,
                ModelTier::Default,
                "Idle until memo, allocation, or cognition-log updates require attention-experience integration.",
            ),
            ServerModule::SelfModel => (
                0.0,
                ModelTier::Default,
                "Idle until explicit self-model requests require work.",
            ),
            ServerModule::QueryVector => (
                0.0,
                ModelTier::Cheap,
                "Idle until memory retrieval is needed.",
            ),
            ServerModule::QueryPolicy => (
                0.0,
                ModelTier::Cheap,
                "Idle until policy retrieval is needed.",
            ),
            ServerModule::Memory => (
                0.0,
                ModelTier::Cheap,
                "Idle until preservation guidance or memory requests arrive.",
            ),
            ServerModule::MemoryCompaction => (
                0.0,
                ModelTier::Cheap,
                "Idle until compaction guidance arrives.",
            ),
            ServerModule::MemoryRecombination => (
                0.0,
                ModelTier::Cheap,
                "Idle until REM-like recombination guidance arrives.",
            ),
            ServerModule::Vital => (
                1.0,
                ModelTier::Cheap,
                "Continuously update homeostatic vital state from cognition volume and memory traces.",
            ),
            ServerModule::HomeostaticController => (
                1.0,
                ModelTier::Cheap,
                "Autonomically drive sleep-like memory modules and cap action modules from vital state.",
            ),
            ServerModule::Policy => (
                0.0,
                ModelTier::Default,
                "Idle until policy formation guidance or distinctive outcomes arrive.",
            ),
            ServerModule::ValueEstimator => (
                0.0,
                ModelTier::Cheap,
                "Idle until query-policy retrieval windows need value estimates.",
            ),
            ServerModule::Reward => (
                0.0,
                ModelTier::Default,
                "Idle until outcomes settle value-estimate windows.",
            ),
            ServerModule::Predict => (
                0.0,
                ModelTier::Cheap,
                "Idle until prediction guidance arrives.",
            ),
            ServerModule::Surprise => (
                0.0,
                ModelTier::Default,
                "Idle until surprise detection is useful.",
            ),
            ServerModule::SpeakGate => (
                0.0,
                ModelTier::Premium,
                "Idle until cognition contains the evidence needed for speech readiness.",
            ),
            ServerModule::Speak => (
                0.0,
                ModelTier::Premium,
                "Idle until cognition-log updates are allowed through speak-gate.",
            ),
        };
        set_allocation_module(
            &mut allocation,
            module.module_id(),
            activation,
            tier,
            guidance,
        );
    }
    allocation
}

fn policy(
    replicas_range: std::ops::RangeInclusive<u8>,
    rate_limit_range: std::ops::RangeInclusive<Bpm>,
) -> ModulePolicy {
    ModulePolicy::new(
        ReplicaCapRange::new(*replicas_range.start(), *replicas_range.end()).unwrap(),
        rate_limit_range,
        linear_ratio_fn,
    )
}

fn homeostatic_drive_modules() -> Vec<ModuleId> {
    vec![
        builtin::memory_compaction(),
        builtin::memory_recombination(),
    ]
}

fn homeostatic_capped_modules() -> Vec<ModuleId> {
    vec![builtin::speak_gate(), builtin::speak()]
}

fn voluntary_modules(_modules: &[ServerModule]) -> Vec<ModuleId> {
    vec![
        builtin::sensory(),
        builtin::attention_schema(),
        builtin::self_model(),
        builtin::query_vector(),
        builtin::query_policy(),
        builtin::memory(),
        builtin::policy(),
        builtin::value_estimator(),
        builtin::reward(),
        builtin::predict(),
        builtin::surprise(),
        builtin::speak_gate(),
        builtin::speak(),
    ]
}

fn set_allocation_module(
    allocation: &mut ResourceAllocation,
    id: ModuleId,
    activation_ratio: f64,
    tier: ModelTier,
    guidance: impl Into<String>,
) {
    allocation.set_model_override(id.clone(), tier);
    allocation.set(
        id.clone(),
        ModuleConfig {
            guidance: guidance.into(),
        },
    );
    allocation.set_activation(id, ActivationRatio::from_f64(activation_ratio));
}

fn activation_table() -> Vec<ActivationRatio> {
    [1.0, 0.85, 0.7, 0.5, 0.3, 0.0]
        .into_iter()
        .map(ActivationRatio::from_f64)
        .collect()
}
