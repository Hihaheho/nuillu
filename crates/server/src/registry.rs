use std::rc::Rc;

use nuillu_blackboard::{
    ActivationRatio, Bpm, ModuleConfig, ModulePolicy, ResourceAllocation, linear_ratio_fn,
};
use nuillu_memory::MemoryCapabilities;
use nuillu_module::ModuleRegistry;
use nuillu_reward::PolicyCapabilities;
use nuillu_speak::{UtteranceSink, UtteranceWriter};
use nuillu_types::{ModelTier, ModuleId};

use super::config::{RuntimeModule, ServerBootConfig, ServerModuleGroup, ServerModuleSpec};

pub(super) fn server_registry(
    boot_config: &ServerBootConfig,
    memory_caps: &MemoryCapabilities,
    policy_caps: &PolicyCapabilities,
    utterance_sink: &Rc<dyn UtteranceSink>,
) -> ModuleRegistry {
    let mut registry = ModuleRegistry::new();
    for module in &boot_config.modules {
        registry = register_server_module(
            registry,
            module,
            boot_config,
            memory_caps,
            policy_caps,
            utterance_sink,
        );
    }
    configured_dependency_edges(boot_config)
        .into_iter()
        .fold(registry, |registry, (dependent, dependency)| {
            registry.depends_on(dependent, dependency)
        })
}

trait ServerRegistryExt {
    fn register_server<B>(self, spec: &ServerModuleSpec, builder: B) -> ModuleRegistry
    where
        B: nuillu_module::ModuleRegisterer + 'static;
}

impl ServerRegistryExt for ModuleRegistry {
    fn register_server<B>(self, spec: &ServerModuleSpec, builder: B) -> ModuleRegistry
    where
        B: nuillu_module::ModuleRegisterer + 'static,
    {
        self.register_with_replica_capacity(policy(spec), spec.replica_capacity, builder)
            .expect("server module registration should be unique")
    }
}

fn register_server_module(
    registry: ModuleRegistry,
    spec: &ServerModuleSpec,
    boot_config: &ServerBootConfig,
    memory_caps: &MemoryCapabilities,
    policy_caps: &PolicyCapabilities,
    utterance_sink: &Rc<dyn UtteranceSink>,
) -> ModuleRegistry {
    let module = spec.id;
    match module {
        RuntimeModule::Sensory => registry.register_server(spec, |caps| async move {
            Ok(nuillu_sensory::SensoryModule::new(
                caps.sensory_input_inbox(),
                caps.allocation_reader(),
                caps.memo(),
                caps.scene_reader(),
                caps.clock(),
                caps.llm_access(),
                caps.session("one-shot")
                    .with_auto_compaction(nuillu_sensory::one_shot_session_auto_compaction())
                    .await?,
                caps.session("ambient")
                    .with_auto_compaction(nuillu_sensory::ambient_session_auto_compaction())
                    .await?,
            ))
        }),
        RuntimeModule::CognitionGate => registry.register_server(spec, |caps| async move {
            Ok(nuillu_cognition_gate::CognitionGateModule::new(
                caps.memo_updated_inbox(),
                caps.blackboard_reader(),
                caps.cognition_writer(),
                caps.llm_access(),
                caps.session("main")
                    .with_auto_compaction(nuillu_cognition_gate::session_auto_compaction())
                    .await?,
            ))
        }),
        RuntimeModule::Allocation => {
            let voluntary = group_modules(boot_config, ServerModuleGroup::Voluntary);
            registry.register_server(spec, move |caps| {
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
                        caps.llm_access(),
                        caps.session("main")
                            .with_auto_compaction(nuillu_allocation::session_auto_compaction())
                            .await?,
                    ))
                }
            })
        }
        RuntimeModule::AttentionSchema => registry.register_server(spec, |caps| async move {
            Ok(nuillu_attention_schema::AttentionSchemaModule::new(
                caps.memo_updated_inbox(),
                caps.cognition_log_updated_inbox(),
                caps.blackboard_reader(),
                caps.cognition_log_reader(),
                caps.cognition_writer(),
                caps.llm_access(),
                caps.session("main")
                    .with_auto_compaction(nuillu_attention_schema::session_auto_compaction())
                    .await?,
            ))
        }),
        RuntimeModule::Interpreter => registry.register_server(spec, |caps| async move {
            Ok(nuillu_interpreter::InterpreterModule::new(
                caps.cognition_log_updated_inbox(),
                caps.cognition_log_reader(),
                caps.cognition_writer(),
                caps.llm_access(),
                caps.session("main")
                    .with_auto_compaction(nuillu_interpreter::session_auto_compaction())
                    .await?,
            ))
        }),
        RuntimeModule::SelfModel => registry.register_server(spec, |caps| async move {
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
        }),
        RuntimeModule::QueryMemory => {
            let memory_caps = memory_caps.clone();
            registry.register_server(spec, move |caps| {
                let memory_caps = memory_caps.clone();
                async move {
                    Ok(nuillu_memory::QueryMemoryModule::new(
                        caps.cognition_log_updated_inbox(),
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
            registry.register_server(spec, move |caps| {
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
            registry.register_server(spec, move |caps| {
                let memory_caps = memory_caps.clone();
                async move {
                    Ok(nuillu_memory::MemoryCompactionModule::new(
                        caps.interoception_updated_inbox(),
                        caps.blackboard_reader(),
                        memory_caps.compactor(),
                        caps.llm_access(),
                        caps.default_tier_llm_access(),
                    ))
                }
            })
        }
        RuntimeModule::MemoryAssociation => {
            let memory_caps = memory_caps.clone();
            registry.register_server(spec, move |caps| {
                let memory_caps = memory_caps.clone();
                async move {
                    Ok(nuillu_memory::MemoryAssociationModule::new(
                        caps.interoception_updated_inbox(),
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
            registry.register_server(spec, move |caps| {
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
            let suppressed = group_modules(boot_config, ServerModuleGroup::SleepSuppressed);
            registry.register_server(spec, move |caps| {
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
            let drive_modules = group_modules(boot_config, ServerModuleGroup::HomeostaticDrive);
            let suppressed = group_modules(boot_config, ServerModuleGroup::SleepSuppressed);
            registry.register_server(spec, move |caps| {
                let drive_modules = drive_modules.clone();
                let suppressed = suppressed.clone();
                async move {
                    Ok(nuillu_homeostasis::HomeostasisModule::new(
                        caps.interoception_updated_inbox(),
                        caps.interoception_reader(),
                        caps.allocation_writer(drive_modules, suppressed),
                    ))
                }
            })
        }
        RuntimeModule::Policy => {
            let policy_caps = policy_caps.clone();
            registry.register_server(spec, move |caps| {
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
            registry.register_server(spec, move |caps| {
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
            registry.register_server(spec, move |caps| {
                let policy_caps = policy_caps.clone();
                async move {
                    Ok(nuillu_reward::RewardModule::new(
                        policy_caps.consideration_evicted_inbox(),
                        caps.blackboard_reader(),
                        caps.cognition_log_reader(),
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
        RuntimeModule::Predict => registry.register_server(spec, |caps| async move {
            Ok(nuillu_predict::PredictModule::new(
                caps.cognition_log_updated_inbox(),
                caps.cognition_log_reader(),
                caps.memo(),
                caps.llm_access(),
                caps.session("main")
                    .with_auto_compaction(nuillu_predict::session_auto_compaction())
                    .await?,
            ))
        }),
        RuntimeModule::Surprise => registry.register_server(spec, |caps| async move {
            Ok(nuillu_surprise::SurpriseModule::new(
                caps.cognition_log_updated_inbox(),
                caps.cognition_log_reader(),
                caps.blackboard_reader(),
                caps.attention_control_mailbox(),
                caps.memo(),
                caps.llm_access(),
                caps.session("main")
                    .with_auto_compaction(nuillu_surprise::session_auto_compaction())
                    .await?,
            ))
        }),
        RuntimeModule::Speak => {
            let utterance_sink = utterance_sink.clone();
            registry.register_server(spec, move |caps| {
                let utterance_sink = utterance_sink.clone();
                async move {
                    Ok(nuillu_speak::SpeakModule::new(
                        nuillu_speak::SpeakModuleParts {
                            cognition_updates: caps.cognition_log_updated_inbox(),
                            cognition_log: caps.cognition_log_reader(),
                            memo: caps.memo(),
                            utterance: UtteranceWriter::new(
                                caps.owner().clone(),
                                caps.blackboard(),
                                utterance_sink.clone(),
                                caps.clock(),
                            ),
                            llm: caps.llm_access(),
                            scene: caps.scene_reader(),
                            clock: caps.clock(),
                            self_wake: caps.self_wake(),
                            planning_session: caps
                                .session("planning")
                                .with_auto_compaction(
                                    nuillu_speak::planning_session_auto_compaction(),
                                )
                                .await?,
                            generation_session: caps
                                .session("generation")
                                .with_auto_compaction(
                                    nuillu_speak::generation_session_auto_compaction(),
                                )
                                .await?,
                        },
                    ))
                }
            })
        }
    }
}

pub(super) fn full_agent_allocation(boot_config: &ServerBootConfig) -> ResourceAllocation {
    let mut allocation = ResourceAllocation::default();
    allocation.set_activation_table(
        boot_config
            .activation_table
            .iter()
            .copied()
            .map(ActivationRatio::from_f64)
            .collect(),
    );
    for module in &boot_config.modules {
        set_allocation_module(
            &mut allocation,
            module.module_id(),
            module.initial_activation,
            module.tier(),
        );
    }
    allocation
}

fn configured_dependency_edges(boot_config: &ServerBootConfig) -> Vec<(ModuleId, ModuleId)> {
    let active = boot_config.active_module_ids();
    let mut edges = Vec::new();
    for module in &boot_config.modules {
        let dependent = module.module_id();
        for dependency in &module.depends_on {
            let dependency = dependency.module_id();
            if active.contains(&dependency) {
                edges.push((dependent.clone(), dependency));
            }
        }
    }
    edges
}

fn group_modules(boot_config: &ServerBootConfig, group: ServerModuleGroup) -> Vec<ModuleId> {
    boot_config
        .specs_in_group(group)
        .into_iter()
        .map(ServerModuleSpec::module_id)
        .collect()
}

fn policy(module: &ServerModuleSpec) -> ModulePolicy {
    ModulePolicy::new(
        module.replica_range(),
        Bpm::range(module.bpm_min, module.bpm_max),
        linear_ratio_fn,
    )
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

#[cfg(test)]
mod tests {
    use super::*;

    use lutum::Session;
    use nuillu_module::{CognitionLogReader, CognitionLogUpdatedInbox, CognitionWriter, LlmAccess};
    use nuillu_types::builtin;

    type InterpreterConstructor = fn(
        CognitionLogUpdatedInbox,
        CognitionLogReader,
        CognitionWriter,
        LlmAccess,
        Session,
    ) -> nuillu_interpreter::InterpreterModule;

    #[test]
    fn interpreter_constructor_uses_only_direct_cognition_capabilities() {
        fn accepts_direct_cognition_signature(_constructor: InterpreterConstructor) {}

        accepts_direct_cognition_signature(nuillu_interpreter::InterpreterModule::new);
    }

    #[test]
    fn configured_dependencies_ignore_absent_modules() {
        let mut boot_config = ServerBootConfig::default();
        boot_config
            .modules
            .retain(|module| module.id != RuntimeModule::Policy);

        let edges = configured_dependency_edges(&boot_config);

        assert!(!edges.contains(&(builtin::cognition_gate(), builtin::policy())));
        assert!(edges.contains(&(builtin::cognition_gate(), builtin::sensory())));
    }

    #[test]
    fn group_modules_are_data_driven() {
        let boot_config = ServerBootConfig::default();

        let voluntary = group_modules(&boot_config, ServerModuleGroup::Voluntary);
        let drive = group_modules(&boot_config, ServerModuleGroup::HomeostaticDrive);

        assert!(voluntary.contains(&builtin::speak()));
        assert!(voluntary.contains(&builtin::interpreter()));
        assert!(!voluntary.contains(&builtin::sensory()));
        assert!(drive.contains(&builtin::memory_compaction()));
        assert!(drive.contains(&builtin::policy_compaction()));
    }

    #[test]
    fn full_agent_allocation_uses_boot_config_module_specs() {
        let mut boot_config = ServerBootConfig {
            activation_table: vec![1.0, 0.25, 0.0],
            ..Default::default()
        };
        boot_config
            .modules
            .retain(|module| module.id == RuntimeModule::Speak);
        boot_config.modules[0].initial_activation = 0.75;

        let allocation = full_agent_allocation(&boot_config);

        assert_eq!(
            allocation.activation_table(),
            &[
                ActivationRatio::from_f64(1.0),
                ActivationRatio::from_f64(0.25),
                ActivationRatio::from_f64(0.0),
            ]
        );
        assert_eq!(
            allocation.activation_for(&builtin::speak()),
            ActivationRatio::from_f64(0.75)
        );
        assert_eq!(
            allocation.tier_for(&builtin::speak()),
            nuillu_types::ModelTier::Premium
        );
        assert_eq!(allocation.get(&builtin::policy()), None);
    }
}
