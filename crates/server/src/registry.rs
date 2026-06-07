use std::rc::Rc;

use nuillu_blackboard::{
    ActivationRatio, Bpm, ModuleConfig, ModulePolicy, ResourceAllocation, linear_ratio_fn,
};
use nuillu_memory::MemoryCapabilities;
use nuillu_module::{ModuleRegistry, apply_standard_dependencies};
use nuillu_reward::PolicyCapabilities;
use nuillu_speak::{UtteranceSink, UtteranceWriter};
use nuillu_types::{ModelTier, ModuleId, ReplicaCapRange, builtin};

use super::config::RuntimeModule;

pub(super) fn server_registry(
    modules: &[RuntimeModule],
    memory_caps: &MemoryCapabilities,
    policy_caps: &PolicyCapabilities,
    utterance_sink: &Rc<dyn UtteranceSink>,
) -> ModuleRegistry {
    let mut registry = ModuleRegistry::new();
    for module in modules {
        registry = register_server_module(
            registry,
            *module,
            modules,
            memory_caps,
            policy_caps,
            utterance_sink,
        );
    }
    apply_standard_dependencies(
        registry,
        modules.iter().copied().map(RuntimeModule::module_id),
    )
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

fn register_server_module(
    registry: ModuleRegistry,
    module: RuntimeModule,
    all_modules: &[RuntimeModule],
    memory_caps: &MemoryCapabilities,
    policy_caps: &PolicyCapabilities,
    utterance_sink: &Rc<dyn UtteranceSink>,
) -> ModuleRegistry {
    match module {
        RuntimeModule::Sensory => {
            registry.register_server(policy(1..=1, Bpm::range(3.0, 8.0)), |caps| async move {
                Ok(nuillu_sensory::SensoryModule::new(
                    caps.sensory_input_inbox(),
                    caps.allocation_reader(),
                    caps.memo(),
                    caps.clock(),
                    caps.llm_access(),
                    caps.session("main")
                        .with_auto_compaction(nuillu_sensory::session_auto_compaction())
                        .await?,
                ))
            })
        }
        RuntimeModule::CognitionGate => {
            registry.register_server(policy(1..=1, Bpm::range(6.0, 12.0)), |caps| async move {
                Ok(nuillu_cognition_gate::CognitionGateModule::new(
                    caps.memo_updated_inbox(),
                    caps.blackboard_reader(),
                    caps.allocation_reader(),
                    caps.cognition_writer(),
                    caps.time_division(),
                    caps.llm_access(),
                    caps.session("main")
                        .with_auto_compaction(nuillu_cognition_gate::session_auto_compaction())
                        .await?,
                ))
            })
        }
        RuntimeModule::Allocation => {
            let voluntary = voluntary_modules(all_modules);
            registry.register_server(policy(1..=1, Bpm::range(6.0, 6.0)), move |caps| {
                let voluntary = voluntary.clone();
                async move {
                    Ok(nuillu_allocation::AllocationModule::new(
                        caps.memo_updated_inbox(),
                        caps.attention_control_inbox(),
                        caps.blackboard_reader(),
                        caps.cognition_log_reader(),
                        caps.allocation_reader(),
                        caps.interoception_reader(),
                        caps.allocation_writer(voluntary.clone(), Vec::new()),
                        caps.memo(),
                        caps.llm_access(),
                        caps.session("main")
                            .with_auto_compaction(nuillu_allocation::session_auto_compaction())
                            .await?,
                    ))
                }
            })
        }
        RuntimeModule::AttentionSchema => {
            registry.register_server(policy(0..=1, Bpm::range(3.0, 6.0)), |caps| async move {
                Ok(nuillu_attention_schema::AttentionSchemaModule::new(
                    caps.memo_updated_inbox(),
                    caps.cognition_log_updated_inbox(),
                    caps.blackboard_reader(),
                    caps.allocation_reader(),
                    caps.cognition_log_reader(),
                    caps.cognition_writer(),
                    caps.llm_access(),
                    caps.session("main")
                        .with_auto_compaction(nuillu_attention_schema::session_auto_compaction())
                        .await?,
                ))
            })
        }
        RuntimeModule::SelfModel => {
            registry.register_server(policy(0..=1, Bpm::range(3.0, 6.0)), |caps| async move {
                Ok(nuillu_self_model::SelfModelModule::new(
                    caps.cognition_log_updated_inbox(),
                    caps.allocation_reader(),
                    caps.blackboard_reader(),
                    caps.cognition_log_reader(),
                    caps.memo(),
                    caps.llm_access(),
                    caps.session("main")
                        .with_auto_compaction(nuillu_self_model::session_auto_compaction())
                        .await?,
                ))
            })
        }
        RuntimeModule::QueryMemory => {
            let memory_caps = memory_caps.clone();
            registry.register_server(policy(1..=1, Bpm::range(6.0, 15.0)), move |caps| {
                let memory_caps = memory_caps.clone();
                async move {
                    Ok(nuillu_memory::QueryMemoryModule::new(
                        caps.cognition_log_updated_inbox(),
                        caps.allocation_reader(),
                        caps.blackboard_reader(),
                        memory_caps.retriever(),
                        memory_caps.content_reader(),
                        caps.typed_memo::<nuillu_memory::QueryMemoryMemo>(),
                        caps.llm_access(),
                        caps.session("main")
                            .with_auto_compaction(nuillu_memory::query_session_auto_compaction())
                            .await?,
                    ))
                }
            })
        }
        RuntimeModule::Memory => {
            let memory_caps = memory_caps.clone();
            registry.register_server(policy(1..=1, Bpm::range(6.0, 18.0)), move |caps| {
                let memory_caps = memory_caps.clone();
                async move {
                    Ok(nuillu_memory::MemoryModule::new(
                        caps.cognition_log_evicted_inbox(),
                        caps.allocation_reader(),
                        caps.memory_metadata_reader(),
                        memory_caps.writer(),
                        memory_caps.retriever(),
                        caps.llm_access(),
                        caps.session("main")
                            .with_auto_compaction(nuillu_memory::session_auto_compaction())
                            .await?,
                    ))
                }
            })
        }
        RuntimeModule::MemoryCompaction => {
            let memory_caps = memory_caps.clone();
            registry.register_server(policy(0..=1, Bpm::range(2.0, 6.0)), move |caps| {
                let memory_caps = memory_caps.clone();
                async move {
                    Ok(nuillu_memory::MemoryCompactionModule::new(
                        caps.interoception_updated_inbox(),
                        caps.allocation_reader(),
                        caps.blackboard_reader(),
                        memory_caps.compactor(),
                        caps.llm_access(),
                    ))
                }
            })
        }
        RuntimeModule::MemoryAssociation => {
            let memory_caps = memory_caps.clone();
            registry.register_server(policy(0..=1, Bpm::range(2.0, 6.0)), move |caps| {
                let memory_caps = memory_caps.clone();
                async move {
                    Ok(nuillu_memory::MemoryAssociationModule::new(
                        caps.interoception_updated_inbox(),
                        caps.allocation_reader(),
                        caps.blackboard_reader(),
                        memory_caps.content_reader(),
                        memory_caps.writer(),
                        memory_caps.associator(),
                        caps.llm_access(),
                    ))
                }
            })
        }
        RuntimeModule::MemoryRecombination => {
            let memory_caps = memory_caps.clone();
            registry.register_server(policy(0..=1, Bpm::range(2.0, 6.0)), move |caps| {
                let memory_caps = memory_caps.clone();
                async move {
                    Ok(nuillu_memory::MemoryRecombinationModule::new(
                        caps.interoception_updated_inbox(),
                        caps.allocation_reader(),
                        caps.blackboard_reader(),
                        memory_caps.retriever(),
                        caps.cognition_writer(),
                        caps.llm_access(),
                    ))
                }
            })
        }
        RuntimeModule::Interoception => {
            let suppressed = sleep_suppressed_modules();
            registry.register_server(policy(1..=1, Bpm::range(1.0, 3.0)), move |caps| {
                let suppressed = suppressed.clone();
                async move {
                    Ok(nuillu_interoception::InteroceptionModule::new(
                        caps.memo_updated_inbox(),
                        caps.cognition_log_updated_inbox(),
                        caps.blackboard_reader(),
                        caps.allocation_writer(Vec::new(), suppressed.clone()),
                        caps.interoception_policy(),
                        caps.interoception_writer(),
                        caps.llm_access(),
                        caps.session("main")
                            .with_auto_compaction(nuillu_interoception::session_auto_compaction())
                            .await?,
                    ))
                }
            })
        }
        RuntimeModule::Homeostasis => {
            registry.register_server(policy(1..=1, Bpm::range(6.0, 20.0)), |caps| async move {
                Ok(nuillu_homeostasis::HomeostasisModule::new(
                    caps.interoception_updated_inbox(),
                    caps.interoception_reader(),
                    caps.allocation_writer(homeostatic_drive_modules(), sleep_suppressed_modules()),
                ))
            })
        }
        RuntimeModule::Policy => {
            let policy_caps = policy_caps.clone();
            registry.register_server(policy(1..=1, Bpm::range(2.0, 6.0)), move |caps| {
                let policy_caps = policy_caps.clone();
                async move {
                    let consideration_writer =
                        policy_caps.consideration_writer(caps.owner().clone());
                    Ok(nuillu_reward::PolicyModule::new(
                        caps.memo_updated_inbox(),
                        caps.cognition_log_updated_inbox(),
                        caps.blackboard_reader(),
                        caps.cognition_log_reader(),
                        caps.allocation_reader(),
                        caps.interoception_reader(),
                        policy_caps.searcher(),
                        caps.memo(),
                        consideration_writer,
                        caps.llm_access(),
                        caps.session("main")
                            .with_auto_compaction(nuillu_reward::policy_session_auto_compaction())
                            .await?,
                    ))
                }
            })
        }
        RuntimeModule::PolicyCompaction => {
            let policy_caps = policy_caps.clone();
            registry.register_server(policy(0..=1, Bpm::range(2.0, 6.0)), move |caps| {
                let policy_caps = policy_caps.clone();
                async move {
                    Ok(nuillu_reward::PolicyCompactionModule::new(
                        caps.interoception_updated_inbox(),
                        caps.allocation_reader(),
                        caps.blackboard_reader(),
                        policy_caps.compactor(),
                        caps.llm_access(),
                    ))
                }
            })
        }
        RuntimeModule::Reward => {
            let policy_caps = policy_caps.clone();
            registry.register_server(policy(1..=1, Bpm::range(1.0, 2.0)), move |caps| {
                let policy_caps = policy_caps.clone();
                async move {
                    Ok(nuillu_reward::RewardModule::new(
                        policy_caps.consideration_evicted_inbox(),
                        caps.blackboard_reader(),
                        caps.cognition_log_reader(),
                        caps.allocation_reader(),
                        caps.interoception_reader(),
                        policy_caps.searcher(),
                        policy_caps.upserter(),
                        caps.memo(),
                        caps.llm_access(),
                        caps.session("main")
                            .with_auto_compaction(nuillu_reward::reward_session_auto_compaction())
                            .await?,
                    ))
                }
            })
        }
        RuntimeModule::Predict => {
            registry.register_server(policy(1..=1, Bpm::range(1.0, 6.0)), |caps| async move {
                Ok(nuillu_predict::PredictModule::new(
                    caps.cognition_log_updated_inbox(),
                    caps.cognition_log_reader(),
                    caps.memo(),
                    caps.llm_access(),
                    caps.session("main")
                        .with_auto_compaction(nuillu_predict::session_auto_compaction())
                        .await?,
                ))
            })
        }
        RuntimeModule::Surprise => {
            registry.register_server(policy(1..=1, Bpm::range(1.0, 3.0)), |caps| async move {
                Ok(nuillu_surprise::SurpriseModule::new(
                    caps.cognition_log_updated_inbox(),
                    caps.cognition_log_reader(),
                    caps.allocation_reader(),
                    caps.blackboard_reader(),
                    caps.attention_control_mailbox(),
                    caps.memo(),
                    caps.llm_access(),
                    caps.session("main")
                        .with_auto_compaction(nuillu_surprise::session_auto_compaction())
                        .await?,
                ))
            })
        }
        RuntimeModule::Speak => {
            let utterance_sink = utterance_sink.clone();
            registry.register_server(policy(0..=1, Bpm::range(3.0, 6.0)), move |caps| {
                let utterance_sink = utterance_sink.clone();
                async move {
                    Ok(nuillu_speak::SpeakModule::new(
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
                    ))
                }
            })
        }
    }
}

pub(super) fn full_agent_allocation(modules: &[RuntimeModule]) -> ResourceAllocation {
    let mut allocation = ResourceAllocation::default();
    allocation.set_activation_table(activation_table());
    for module in modules {
        let (activation, tier) = match module {
            RuntimeModule::Sensory => (1.0, ModelTier::Cheap),
            RuntimeModule::CognitionGate => (1.0, ModelTier::Default),
            RuntimeModule::Allocation => (1.0, ModelTier::Default),
            RuntimeModule::AttentionSchema => (0.0, ModelTier::Default),
            RuntimeModule::SelfModel => (0.0, ModelTier::Default),
            RuntimeModule::QueryMemory => (0.0, ModelTier::Cheap),
            RuntimeModule::Memory => (0.0, ModelTier::Cheap),
            RuntimeModule::MemoryCompaction => (0.0, ModelTier::Cheap),
            RuntimeModule::MemoryAssociation => (0.0, ModelTier::Cheap),
            RuntimeModule::MemoryRecombination => (0.0, ModelTier::Cheap),
            RuntimeModule::Interoception => (1.0, ModelTier::Cheap),
            RuntimeModule::Homeostasis => (1.0, ModelTier::Cheap),
            RuntimeModule::Policy => (0.0, ModelTier::Default),
            RuntimeModule::PolicyCompaction => (0.0, ModelTier::Cheap),
            RuntimeModule::Reward => (0.0, ModelTier::Default),
            RuntimeModule::Predict => (0.0, ModelTier::Cheap),
            RuntimeModule::Surprise => (0.0, ModelTier::Default),
            RuntimeModule::Speak => (0.0, ModelTier::Premium),
        };
        set_allocation_module(&mut allocation, module.module_id(), activation, tier);
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
        builtin::memory_association(),
        builtin::memory_recombination(),
        builtin::policy_compaction(),
    ]
}

fn sleep_suppressed_modules() -> Vec<ModuleId> {
    vec![
        builtin::cognition_gate(),
        builtin::attention_schema(),
        builtin::self_model(),
        builtin::query_memory(),
        builtin::memory(),
        builtin::policy(),
        builtin::reward(),
        builtin::predict(),
        builtin::surprise(),
        builtin::speak(),
    ]
}

fn voluntary_modules(_modules: &[RuntimeModule]) -> Vec<ModuleId> {
    vec![
        builtin::cognition_gate(),
        builtin::attention_schema(),
        builtin::self_model(),
        builtin::query_memory(),
        builtin::memory(),
        builtin::policy(),
        builtin::reward(),
        builtin::predict(),
        builtin::surprise(),
        builtin::speak(),
    ]
}

fn set_allocation_module(
    allocation: &mut ResourceAllocation,
    id: ModuleId,
    activation_ratio: f64,
    tier: ModelTier,
) {
    allocation.set_model_override(id.clone(), tier);
    allocation.set(id.clone(), ModuleConfig::default());
    allocation.set_activation(id, ActivationRatio::from_f64(activation_ratio));
}

fn activation_table() -> Vec<ActivationRatio> {
    [1.0, 0.85, 0.7, 0.5, 0.3, 0.0]
        .into_iter()
        .map(ActivationRatio::from_f64)
        .collect()
}
