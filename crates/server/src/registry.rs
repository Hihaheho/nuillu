use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::sync::Arc;

use nuillu_blackboard::{
    ActivationRatio, Bpm, ModulePolicy, RateProjection, ReplicaProjection, ResourceAllocation,
    SubsystemPolicy, SubsystemReplicaRange,
};
use nuillu_memory::{MemoryCapabilities, MemoryNamespace};
use nuillu_module::{
    ModuleRegistrationSpec, ModuleRegistry, ModuleRegistryError, StaticModule,
    SubsystemRegistrationSpec,
};
use nuillu_reward::PolicyCapabilities;
use nuillu_speak::{UtteranceSink, UtteranceWriter};
use nuillu_types::{ModelTier, ModuleId, ScopeId, is_kebab_case};

#[cfg(test)]
use super::config::ServerActivationBarrierSpec;
use super::config::{
    RuntimeModule, ServerBootConfig, ServerMemoryScope, ServerModuleGroup, ServerModuleSpec,
    ServerProjectionCurve, ServerProjectionSpec,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ServerModelSlotDescriptor {
    pub key: Arc<str>,
    pub default_tier: ModelTier,
}

impl ServerModelSlotDescriptor {
    pub fn new(key: impl Into<Arc<str>>, default_tier: ModelTier) -> Self {
        Self {
            key: key.into(),
            default_tier,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ServerModuleDescriptor {
    pub id: ModuleId,
    pub peer_context: Option<Arc<str>>,
    pub model_slots: Vec<ServerModelSlotDescriptor>,
    pub root_only: bool,
    pub max_configured_instances: Option<usize>,
    pub max_replica_capacity: Option<u8>,
}

impl ServerModuleDescriptor {
    pub fn new(id: ModuleId) -> Self {
        Self {
            id,
            peer_context: None,
            model_slots: Vec::new(),
            root_only: false,
            max_configured_instances: None,
            max_replica_capacity: None,
        }
    }

    pub fn with_peer_context(mut self, peer_context: impl Into<Arc<str>>) -> Self {
        self.peer_context = Some(peer_context.into());
        self
    }

    pub fn with_model_slot(mut self, key: impl Into<Arc<str>>, tier: ModelTier) -> Self {
        self.model_slots
            .push(ServerModelSlotDescriptor::new(key, tier));
        self
    }

    pub fn root_only(mut self) -> Self {
        self.root_only = true;
        self
    }

    pub fn with_max_configured_instances(mut self, max: usize) -> Self {
        self.max_configured_instances = Some(max);
        self
    }

    pub fn with_max_replica_capacity(mut self, max: u8) -> Self {
        self.max_replica_capacity = Some(max);
        self
    }
}

#[derive(Debug, Clone)]
pub struct ResolvedServerModuleConfig {
    scope: ScopeId,
    spec: ServerModuleSpec,
    model_tiers: BTreeMap<Arc<str>, ModelTier>,
}

impl ResolvedServerModuleConfig {
    pub fn scope(&self) -> &ScopeId {
        &self.scope
    }

    pub fn spec(&self) -> &ServerModuleSpec {
        &self.spec
    }

    pub fn model_tier(&self, key: &str) -> Option<ModelTier> {
        self.model_tiers.get(key).copied()
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ServerModuleFactoryError {
    #[error(transparent)]
    Registry(Box<ModuleRegistryError>),
    #[error("{0}")]
    Implementation(String),
}

impl ServerModuleFactoryError {
    pub fn implementation(message: impl Into<String>) -> Self {
        Self::Implementation(message.into())
    }
}

impl From<ModuleRegistryError> for ServerModuleFactoryError {
    fn from(value: ModuleRegistryError) -> Self {
        Self::Registry(Box::new(value))
    }
}

#[must_use = "a module slot must be filled with exactly one builder"]
pub struct ServerModuleSlot {
    registry: ModuleRegistry,
    registration: ModuleRegistrationSpec,
}

#[must_use = "the filled slot must be returned to the server"]
pub struct FilledServerModuleSlot {
    registry: ModuleRegistry,
}

impl ServerModuleSlot {
    /// Fills this configured role with its implementation builder.
    ///
    /// The builder is retained and may be called once per replica and again
    /// when a module is restarted, so captured construction inputs must be
    /// reusable and must not rely on one-shot `Option::take()` state.
    pub fn with_builder<B>(self, builder: B) -> Result<FilledServerModuleSlot, ModuleRegistryError>
    where
        B: nuillu_module::ModuleRegisterer + 'static,
    {
        Ok(FilledServerModuleSlot {
            registry: self.registry.register(self.registration, builder)?,
        })
    }
}

/// Host-provided implementation for one configured module id.
///
/// The server owns registration metadata and gives the factory a single-use
/// slot. Implementations can supply a builder but cannot alter the registry.
pub trait ServerModuleFactory: Send + Sync {
    fn descriptor(&self) -> &ServerModuleDescriptor;

    fn implement(
        &self,
        slot: ServerModuleSlot,
        config: &ResolvedServerModuleConfig,
    ) -> Result<FilledServerModuleSlot, ServerModuleFactoryError>;
}

pub struct ServerModuleFactoryFn<F> {
    descriptor: ServerModuleDescriptor,
    implement: F,
}

impl<F> ServerModuleFactoryFn<F> {
    pub fn new(descriptor: ServerModuleDescriptor, implement: F) -> Self {
        Self {
            descriptor,
            implement,
        }
    }
}

impl<F> ServerModuleFactory for ServerModuleFactoryFn<F>
where
    F: Fn(
            ServerModuleSlot,
            &ResolvedServerModuleConfig,
        ) -> Result<FilledServerModuleSlot, ServerModuleFactoryError>
        + Send
        + Sync,
{
    fn descriptor(&self) -> &ServerModuleDescriptor {
        &self.descriptor
    }

    fn implement(
        &self,
        slot: ServerModuleSlot,
        config: &ResolvedServerModuleConfig,
    ) -> Result<FilledServerModuleSlot, ServerModuleFactoryError> {
        (self.implement)(slot, config)
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ServerModuleConfigError {
    #[error("server config {path} has invalid factory descriptor for module {module}: {reason}")]
    InvalidFactoryDescriptor {
        path: PathBuf,
        module: ModuleId,
        reason: String,
    },
    #[error("server config {path} has more than one host factory for module {module}")]
    DuplicateFactory { path: PathBuf, module: ModuleId },
    #[error("server config {path} cannot replace built-in module {module} with a host factory")]
    BuiltinFactoryConflict { path: PathBuf, module: ModuleId },
    #[error(
        "server config {path} has no implementation factory for module {module} in scope {scope}"
    )]
    MissingFactory {
        path: PathBuf,
        scope: ScopeId,
        module: ModuleId,
    },
    #[error(
        "server config {path} declares unknown model slot {slot:?} for module {module} in scope {scope}"
    )]
    UnknownModelSlot {
        path: PathBuf,
        scope: ScopeId,
        module: ModuleId,
        slot: String,
    },
    #[error("server config {path} places root-only module {module} in scope {scope}")]
    RootOnly {
        path: PathBuf,
        scope: ScopeId,
        module: ModuleId,
    },
    #[error(
        "server config {path} declares {actual} instances of module {module}, above the factory limit {max}"
    )]
    TooManyInstances {
        path: PathBuf,
        module: ModuleId,
        actual: usize,
        max: usize,
    },
    #[error(
        "server config {path} sets replica-capacity={actual} for module {module} in scope {scope}, above the factory limit {max}"
    )]
    ReplicaCapacityAboveFactoryLimit {
        path: PathBuf,
        scope: ScopeId,
        module: ModuleId,
        actual: u8,
        max: u8,
    },
    #[error("server config {path} factory for module {module} failed in scope {scope}: {source}")]
    Factory {
        path: PathBuf,
        scope: ScopeId,
        module: ModuleId,
        #[source]
        source: ServerModuleFactoryError,
    },
}

pub(super) struct ServerModuleCatalog<'a> {
    hosts: HashMap<ModuleId, &'a Arc<dyn ServerModuleFactory>>,
}

impl<'a> ServerModuleCatalog<'a> {
    pub(super) fn new(
        path: &Path,
        factories: &'a [Arc<dyn ServerModuleFactory>],
    ) -> Result<Self, ServerModuleConfigError> {
        let mut hosts = HashMap::new();
        for factory in factories {
            let module = factory.descriptor().id.clone();
            let mut model_slots = std::collections::HashSet::new();
            for slot in &factory.descriptor().model_slots {
                if !is_kebab_case(&slot.key) {
                    return Err(ServerModuleConfigError::InvalidFactoryDescriptor {
                        path: path.to_path_buf(),
                        module,
                        reason: format!("model slot {:?} is not a kebab-case id", slot.key),
                    });
                }
                if !model_slots.insert(slot.key.as_ref()) {
                    return Err(ServerModuleConfigError::InvalidFactoryDescriptor {
                        path: path.to_path_buf(),
                        module,
                        reason: format!("model slot {:?} is declared more than once", slot.key),
                    });
                }
            }
            if RuntimeModule::from_module_id(&module).is_some() {
                return Err(ServerModuleConfigError::BuiltinFactoryConflict {
                    path: path.to_path_buf(),
                    module,
                });
            }
            if hosts.insert(module.clone(), factory).is_some() {
                return Err(ServerModuleConfigError::DuplicateFactory {
                    path: path.to_path_buf(),
                    module,
                });
            }
        }
        Ok(Self { hosts })
    }

    fn host(&self, module: &ModuleId) -> Option<&Arc<dyn ServerModuleFactory>> {
        self.hosts.get(module).copied()
    }
}

pub(super) fn validate_configured_modules(
    path: &Path,
    boot_config: &ServerBootConfig,
    catalog: &ServerModuleCatalog<'_>,
) -> Result<(), ServerModuleConfigError> {
    let mut instances = HashMap::<ModuleId, usize>::new();
    for module in &boot_config.modules {
        validate_configured_module(path, &ScopeId::root(), module, catalog, &mut instances)?;
    }
    for expanded in boot_config.expanded_subsystems() {
        for module in &expanded.definition.modules {
            validate_configured_module(path, &expanded.scope, module, catalog, &mut instances)?;
        }
    }
    for (module, actual) in instances {
        let Some(descriptor) = catalog.host(&module).map(|factory| factory.descriptor()) else {
            continue;
        };
        if let Some(max) = descriptor.max_configured_instances
            && actual > max
        {
            return Err(ServerModuleConfigError::TooManyInstances {
                path: path.to_path_buf(),
                module,
                actual,
                max,
            });
        }
    }
    Ok(())
}

#[cfg(test)]
fn validate_server_module_factories(
    path: &Path,
    boot_config: &ServerBootConfig,
    factories: &[Arc<dyn ServerModuleFactory>],
) -> Result<(), ServerModuleConfigError> {
    let catalog = ServerModuleCatalog::new(path, factories)?;
    validate_configured_modules(path, boot_config, &catalog)
}

fn validate_configured_module(
    path: &Path,
    scope: &ScopeId,
    module: &ServerModuleSpec,
    catalog: &ServerModuleCatalog<'_>,
    instances: &mut HashMap<ModuleId, usize>,
) -> Result<(), ServerModuleConfigError> {
    let module_id = module.module_id();
    *instances.entry(module_id.clone()).or_default() += 1;
    let host_descriptor = catalog.host(&module_id).map(|factory| factory.descriptor());
    if RuntimeModule::from_module_id(&module_id).is_none() && host_descriptor.is_none() {
        return Err(ServerModuleConfigError::MissingFactory {
            path: path.to_path_buf(),
            scope: scope.clone(),
            module: module_id,
        });
    }
    let slots = match host_descriptor {
        Some(descriptor) => descriptor.model_slots.as_slice(),
        None => {
            let builtin = RuntimeModule::from_module_id(&module_id)
                .expect("known built-in module should resolve");
            for configured in &module.model_slots {
                if builtin
                    .model_slot_defaults()
                    .iter()
                    .all(|(key, _)| *key != configured.key)
                {
                    return Err(ServerModuleConfigError::UnknownModelSlot {
                        path: path.to_path_buf(),
                        scope: scope.clone(),
                        module: module_id,
                        slot: configured.key.clone(),
                    });
                }
            }
            return Ok(());
        }
    };
    for configured in &module.model_slots {
        if slots.iter().all(|slot| slot.key.as_ref() != configured.key) {
            return Err(ServerModuleConfigError::UnknownModelSlot {
                path: path.to_path_buf(),
                scope: scope.clone(),
                module: module_id.clone(),
                slot: configured.key.clone(),
            });
        }
    }
    let descriptor = host_descriptor.expect("host descriptor selected above");
    if descriptor.root_only && !scope.is_root() {
        return Err(ServerModuleConfigError::RootOnly {
            path: path.to_path_buf(),
            scope: scope.clone(),
            module: module_id,
        });
    }
    if let Some(max) = descriptor.max_replica_capacity
        && module.replica_capacity > max
    {
        return Err(ServerModuleConfigError::ReplicaCapacityAboveFactoryLimit {
            path: path.to_path_buf(),
            scope: scope.clone(),
            module: module_id,
            actual: module.replica_capacity,
            max,
        });
    }
    Ok(())
}

pub(super) fn server_registry(
    config_path: &Path,
    boot_config: &ServerBootConfig,
    catalog: &ServerModuleCatalog<'_>,
    memory_caps: &MemoryCapabilities,
    policy_caps: &PolicyCapabilities,
    utterance_sink: &Rc<dyn UtteranceSink>,
) -> Result<ModuleRegistry, ServerModuleConfigError> {
    let mut registry = ModuleRegistry::new().with_registration_scope(ScopeId::root());
    let root_memory_caps = memory_caps.with_namespace(MemoryNamespace::Global);
    let root_resources = ServerModuleResources {
        memory: &root_memory_caps,
        policy: policy_caps,
        utterance_sink,
    };
    for module in &boot_config.modules {
        registry = register_configured_module(
            registry,
            config_path,
            ScopeId::root(),
            module,
            catalog,
            &root_resources,
        )?;
    }
    registry = configured_dependency_edges(&boot_config.modules)
        .into_iter()
        .fold(registry, |registry, (dependent, dependency)| {
            registry.scoped_depends_on(ScopeId::root(), dependent, dependency)
        });
    registry = configured_activation_barriers(&boot_config.modules)
        .into_iter()
        .fold(registry, |registry, (dependent, prerequisites, timeout)| {
            registry.scoped_activation_barrier(ScopeId::root(), dependent, prerequisites, timeout)
        });
    for expanded in boot_config.expanded_subsystems() {
        let mut subsystem_spec = SubsystemRegistrationSpec::new(
            expanded
                .scope
                .parent()
                .expect("expanded subsystem has parent"),
            expanded.definition.subsystem_id(),
            SubsystemPolicy::new(
                SubsystemReplicaRange::new(
                    expanded.mount.replica_min(),
                    expanded.mount.replica_max(),
                )
                .expect("validated subsystem replica range"),
                expanded.mount.replica_capacity(),
                replica_projection(expanded.mount.replica_projection()),
            ),
            ActivationRatio::from_f64(expanded.mount.initial_activation()),
            expanded.definition.allocation_description.clone(),
        );
        if let Some(label) = &expanded.definition.label {
            subsystem_spec = subsystem_spec.with_label(label.clone());
        }
        subsystem_spec = subsystem_spec.with_activation_table(
            expanded
                .mount
                .activation_table
                .iter()
                .copied()
                .map(ActivationRatio::from_f64),
        );
        // One registration describes the mount; all capacity instances share
        // the same parent and subsystem id.
        if expanded
            .scope
            .path()
            .last()
            .is_some_and(|instance| instance.replica == nuillu_types::ReplicaIndex::ZERO)
        {
            registry = registry.with_subsystem(subsystem_spec);
        }
        let scoped_memory_caps = match expanded.definition.memory_scope {
            ServerMemoryScope::Global => memory_caps.with_namespace(MemoryNamespace::Global),
            ServerMemoryScope::Local => {
                memory_caps.with_namespace(MemoryNamespace::Local(expanded.scope.clone()))
            }
        };
        let scoped_resources = ServerModuleResources {
            memory: &scoped_memory_caps,
            policy: policy_caps,
            utterance_sink,
        };
        registry = registry.with_registration_scope(expanded.scope.clone());
        for module in &expanded.definition.modules {
            registry = register_configured_module(
                registry,
                config_path,
                expanded.scope.clone(),
                module,
                catalog,
                &scoped_resources,
            )?;
        }
        registry = configured_dependency_edges(&expanded.definition.modules)
            .into_iter()
            .fold(registry, |registry, (dependent, dependency)| {
                registry.scoped_depends_on(expanded.scope.clone(), dependent, dependency)
            });
        registry = configured_activation_barriers(&expanded.definition.modules)
            .into_iter()
            .fold(registry, |registry, (dependent, prerequisites, timeout)| {
                registry.scoped_activation_barrier(
                    expanded.scope.clone(),
                    dependent,
                    prerequisites,
                    timeout,
                )
            });
    }
    Ok(registry.with_registration_scope(ScopeId::root()))
}

struct ServerModuleResources<'a> {
    memory: &'a MemoryCapabilities,
    policy: &'a PolicyCapabilities,
    utterance_sink: &'a Rc<dyn UtteranceSink>,
}

fn register_configured_module(
    registry: ModuleRegistry,
    config_path: &Path,
    scope: ScopeId,
    spec: &ServerModuleSpec,
    catalog: &ServerModuleCatalog<'_>,
    resources: &ServerModuleResources<'_>,
) -> Result<ModuleRegistry, ServerModuleConfigError> {
    if RuntimeModule::from_module_id(spec.id.as_module_id()).is_some() {
        return Ok(register_server_module(
            registry,
            spec,
            resources.memory,
            resources.policy,
            resources.utterance_sink,
        ));
    }
    let module = spec.module_id();
    let factory = catalog
        .host(&module)
        .expect("catalog-aware validation should require a host factory");
    let descriptor = factory.descriptor();
    let mut registration = ModuleRegistrationSpec::new(
        module.clone(),
        policy(spec),
        ActivationRatio::from_f64(spec.initial_activation),
    )
    .in_scope(scope.clone())
    .with_replica_capacity(spec.replica_capacity);
    if let Some(peer_context) = &descriptor.peer_context {
        registration = registration.with_peer_context(Arc::clone(peer_context));
    }
    for group in &spec.groups {
        registration = registration.in_group(group.as_module_group_id().clone());
    }
    if let Some(sources) = &spec.memo_sources {
        registration = registration
            .with_memo_sources(sources.iter().map(|source| source.as_module_id().clone()));
    }
    let mut model_tiers = descriptor
        .model_slots
        .iter()
        .map(|slot| (Arc::clone(&slot.key), slot.default_tier))
        .collect::<BTreeMap<_, _>>();
    for configured in &spec.model_slots {
        if let Some((_, tier)) = model_tiers
            .iter_mut()
            .find(|(key, _)| key.as_ref() == configured.key)
        {
            *tier = configured.tier.into();
        }
    }
    let resolved = ResolvedServerModuleConfig {
        scope: scope.clone(),
        spec: spec.clone(),
        model_tiers,
    };
    let filled = factory
        .implement(
            ServerModuleSlot {
                registry,
                registration,
            },
            &resolved,
        )
        .map_err(|source| ServerModuleConfigError::Factory {
            path: config_path.to_path_buf(),
            scope,
            module,
            source,
        })?;
    Ok(filled.registry)
}

trait ServerRegistryExt {
    fn register_server<B>(self, spec: &ServerModuleSpec, builder: B) -> ModuleRegistry
    where
        B: nuillu_module::ModuleRegisterer + 'static,
        B::Module: StaticModule;
}

impl ServerRegistryExt for ModuleRegistry {
    fn register_server<B>(self, spec: &ServerModuleSpec, builder: B) -> ModuleRegistry
    where
        B: nuillu_module::ModuleRegisterer + 'static,
        B::Module: StaticModule,
    {
        let mut registration = ModuleRegistrationSpec::for_static::<B::Module>(
            policy(spec),
            ActivationRatio::from_f64(spec.initial_activation),
        )
        .expect("built-in module id should be valid")
        .with_replica_capacity(spec.replica_capacity);
        for group in &spec.groups {
            registration = registration.in_group(group.as_module_group_id().clone());
        }
        if let Some(sources) = &spec.memo_sources {
            registration = registration
                .with_memo_sources(sources.iter().map(|source| source.as_module_id().clone()));
        }
        self.register(registration, builder)
            .expect("server module registration should be unique")
    }
}

fn register_server_module(
    registry: ModuleRegistry,
    spec: &ServerModuleSpec,
    memory_caps: &MemoryCapabilities,
    policy_caps: &PolicyCapabilities,
    utterance_sink: &Rc<dyn UtteranceSink>,
) -> ModuleRegistry {
    let module = RuntimeModule::from_module_id(spec.id.as_module_id())
        .expect("built-in registration requires a built-in module id");
    match module {
        RuntimeModule::Sensory => {
            let one_shot_tier = spec.model_tier("one-shot");
            let ambient_tier = spec.model_tier("ambient");
            registry.register_server(spec, move |caps| async move {
                Ok(nuillu_sensory::SensoryModule::new(
                    caps.sensory_input_inbox(),
                    caps.memo(),
                    caps.scene_reader(),
                    caps.clock(),
                    caps.timer(),
                    caps.llm("one-shot").with_tier(one_shot_tier).into(),
                    caps.session("one-shot")
                        .with_tier(one_shot_tier)
                        .with_auto_compaction(nuillu_sensory::one_shot_session_auto_compaction())
                        .await?,
                    caps.session("ambient")
                        .with_tier(ambient_tier)
                        .with_auto_compaction(nuillu_sensory::ambient_session_auto_compaction())
                        .await?,
                ))
            })
        }
        RuntimeModule::CognitionGate => {
            let main_tier = spec.model_tier("main");
            registry.register_server(spec, move |caps| async move {
                Ok(nuillu_cognition_gate::CognitionGateModule::new(
                    caps.memo_updated_inbox(),
                    caps.blackboard_reader(),
                    caps.cognition_writer(),
                    caps.llm("main").with_tier(main_tier).into(),
                    caps.session("main")
                        .with_tier(main_tier)
                        .with_auto_compaction(nuillu_cognition_gate::session_auto_compaction())
                        .await?,
                ))
            })
        }
        RuntimeModule::Allocation => {
            let main_tier = spec.model_tier("main");
            registry.register_server(spec, move |caps| async move {
                let voluntary = caps
                    .module_catalog()
                    .members(&ServerModuleGroup::Voluntary.module_group_id());
                Ok(nuillu_allocation::AllocationModule::new(
                    caps.memo_updated_inbox(),
                    caps.attention_control_inbox(),
                    caps.blackboard_reader(),
                    caps.cognition_log_reader(),
                    caps.allocation_reader(),
                    caps.interoception_reader(),
                    caps.allocation_writer(voluntary, Vec::new()),
                    caps.llm("main").with_tier(main_tier).into(),
                    caps.session("main")
                        .with_tier(main_tier)
                        .with_auto_compaction(nuillu_allocation::session_auto_compaction())
                        .await?,
                    caps.timer(),
                ))
            })
        }
        RuntimeModule::Action => {
            let main_tier = spec.model_tier("main");
            registry.register_server(spec, move |caps| async move {
                let action_targets = caps
                    .module_catalog()
                    .members(&ServerModuleGroup::ActionTarget.module_group_id());
                Ok(nuillu_action::ActionModule::new(
                    caps.memo_updated_inbox(),
                    caps.cognition_log_updated_inbox(),
                    caps.interoception_updated_inbox(),
                    caps.action_affordances_updated_inbox(),
                    caps.blackboard_reader(),
                    caps.cognition_log_reader(),
                    caps.allocation_reader(),
                    caps.interoception_reader(),
                    caps.action_affordance_reader(),
                    caps.external_action_invoker(),
                    caps.allocation_writer(action_targets.clone(), Vec::new()),
                    caps.interoception_writer(),
                    caps.memo(),
                    caps.llm("main").with_tier(main_tier).into(),
                    caps.session("main")
                        .with_tier(main_tier)
                        .with_auto_compaction(nuillu_action::session_auto_compaction())
                        .await?,
                ))
            })
        }
        RuntimeModule::AttentionSchema => {
            let main_tier = spec.model_tier("main");
            registry.register_server(spec, move |caps| async move {
                Ok(nuillu_attention_schema::AttentionSchemaModule::new(
                    caps.memo_updated_inbox(),
                    caps.cognition_log_updated_inbox(),
                    caps.blackboard_reader(),
                    caps.cognition_log_reader(),
                    caps.memo(),
                    caps.llm("main").with_tier(main_tier).into(),
                    caps.session("main")
                        .with_tier(main_tier)
                        .with_auto_compaction(nuillu_attention_schema::session_auto_compaction())
                        .await?,
                ))
            })
        }
        RuntimeModule::Interpreter => {
            let main_tier = spec.model_tier("main");
            registry.register_server(spec, move |caps| async move {
                Ok(nuillu_interpreter::InterpreterModule::new(
                    caps.cognition_log_updated_inbox(),
                    caps.cognition_log_reader(),
                    caps.cognition_writer(),
                    caps.llm("main").with_tier(main_tier).into(),
                    caps.session("main")
                        .with_tier(main_tier)
                        .with_auto_compaction(nuillu_interpreter::session_auto_compaction())
                        .await?,
                ))
            })
        }
        RuntimeModule::SelfModel => {
            let main_tier = spec.model_tier("main");
            registry.register_server(spec, move |caps| async move {
                Ok(nuillu_self_model::SelfModelModule::new(
                    caps.memo_updated_inbox(),
                    caps.blackboard_reader(),
                    caps.memo(),
                    caps.llm("main").with_tier(main_tier).into(),
                    caps.session("main")
                        .with_tier(main_tier)
                        .with_auto_compaction(nuillu_self_model::session_auto_compaction())
                        .await?,
                ))
            })
        }
        RuntimeModule::QueryMemory => {
            let memory_caps = memory_caps.clone();
            let main_tier = spec.model_tier("main");
            registry.register_server(spec, move |caps| {
                let memory_caps = memory_caps.clone();
                async move {
                    let memory_caps = memory_caps.scoped(caps.blackboard());
                    Ok(nuillu_memory::QueryMemoryModule::new(
                        caps.cognition_log_updated_inbox(),
                        caps.blackboard_reader(),
                        memory_caps.retriever(),
                        memory_caps.content_reader(),
                        caps.typed_memo::<nuillu_memory::QueryMemoryMemo>(),
                        caps.llm("main").with_tier(main_tier).into(),
                        caps.session("main")
                            .with_tier(main_tier)
                            .with_auto_compaction(nuillu_memory::query_session_auto_compaction())
                            .await?,
                    ))
                }
            })
        }
        RuntimeModule::Memory => {
            let memory_caps = memory_caps.clone();
            let main_tier = spec.model_tier("main");
            registry.register_server(spec, move |caps| {
                let memory_caps = memory_caps.clone();
                async move {
                    let memory_caps = memory_caps.scoped(caps.blackboard());
                    Ok(nuillu_memory::MemoryModule::new(
                        caps.memo_updated_inbox(),
                        caps.cognition_log_updated_inbox(),
                        caps.blackboard_reader(),
                        caps.cognition_log_reader(),
                        caps.memory_metadata_reader(),
                        memory_caps.writer(),
                        memory_caps.deleter(),
                        memory_caps.retriever(),
                        caps.llm("main").with_tier(main_tier).into(),
                        caps.session("main")
                            .with_tier(main_tier)
                            .with_auto_compaction(nuillu_memory::session_auto_compaction())
                            .await?,
                        caps.timer(),
                    ))
                }
            })
        }
        RuntimeModule::MemoryCompaction => {
            let memory_caps = memory_caps.clone();
            let main_tier = spec.model_tier("main");
            let audit_tier = spec.model_tier("audit");
            registry.register_server(spec, move |caps| {
                let memory_caps = memory_caps.clone();
                async move {
                    let memory_caps = memory_caps.scoped(caps.blackboard());
                    Ok(nuillu_memory::MemoryCompactionModule::new(
                        caps.interoception_updated_inbox(),
                        caps.blackboard_reader(),
                        memory_caps.compactor(),
                        caps.llm("main").with_tier(main_tier).into(),
                        caps.llm("audit").with_tier(audit_tier).into(),
                    ))
                }
            })
        }
        RuntimeModule::MemoryAssociation => {
            let memory_caps = memory_caps.clone();
            let main_tier = spec.model_tier("main");
            registry.register_server(spec, move |caps| {
                let memory_caps = memory_caps.clone();
                async move {
                    let memory_caps = memory_caps.scoped(caps.blackboard());
                    Ok(nuillu_memory::MemoryAssociationModule::new(
                        caps.interoception_updated_inbox(),
                        caps.blackboard_reader(),
                        memory_caps.content_reader(),
                        memory_caps.writer(),
                        memory_caps.associator(),
                        caps.llm("main").with_tier(main_tier).into(),
                    ))
                }
            })
        }
        RuntimeModule::Dreaming => {
            let main_tier = spec.model_tier("main");
            registry.register_server(spec, move |caps| async move {
                Ok(nuillu_memory::DreamingModule::new(
                    caps.interoception_updated_inbox(),
                    caps.allocation_reader(),
                    caps.blackboard_reader(),
                    caps.memo(),
                    caps.llm("main").with_tier(main_tier).into(),
                ))
            })
        }
        RuntimeModule::Interoception => {
            let main_tier = spec.model_tier("main");
            registry.register_server(spec, move |caps| async move {
                Ok(nuillu_interoception::InteroceptionModule::new(
                    caps.memo_updated_inbox(),
                    caps.cognition_log_updated_inbox(),
                    caps.blackboard_reader(),
                    caps.interoception_policy(),
                    caps.interoception_writer(),
                    caps.llm("main").with_tier(main_tier).into(),
                    caps.session("main")
                        .with_tier(main_tier)
                        .with_auto_compaction(nuillu_interoception::session_auto_compaction())
                        .await?,
                    caps.timer(),
                ))
            })
        }
        RuntimeModule::Homeostasis => registry.register_server(spec, move |caps| async move {
            let drive_modules = caps
                .module_catalog()
                .members(&ServerModuleGroup::HomeostaticDrive.module_group_id());
            let suppressed = caps
                .module_catalog()
                .members(&ServerModuleGroup::SleepSuppressed.module_group_id());
            Ok(nuillu_homeostasis::HomeostasisModule::new(
                caps.interoception_updated_inbox(),
                caps.interoception_reader(),
                caps.allocation_writer(drive_modules, suppressed),
                caps.timer(),
            ))
        }),
        RuntimeModule::Policy => {
            let policy_caps = policy_caps.clone();
            let main_tier = spec.model_tier("main");
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
                        caps.interoception_reader(),
                        policy_caps.searcher(),
                        caps.memo(),
                        consideration_writer,
                        caps.llm("main").with_tier(main_tier).into(),
                        caps.session("main")
                            .with_tier(main_tier)
                            .with_auto_compaction(nuillu_reward::policy_session_auto_compaction())
                            .await?,
                    ))
                }
            })
        }
        RuntimeModule::PolicyCompaction => {
            let policy_caps = policy_caps.clone();
            let main_tier = spec.model_tier("main");
            registry.register_server(spec, move |caps| {
                let policy_caps = policy_caps.clone();
                async move {
                    Ok(nuillu_reward::PolicyCompactionModule::new(
                        caps.interoception_updated_inbox(),
                        caps.blackboard_reader(),
                        policy_caps.compactor(),
                        caps.llm("main").with_tier(main_tier).into(),
                    ))
                }
            })
        }
        RuntimeModule::Reward => {
            let policy_caps = policy_caps.clone();
            let main_tier = spec.model_tier("main");
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
                        caps.llm("main").with_tier(main_tier).into(),
                        caps.session("main")
                            .with_tier(main_tier)
                            .with_auto_compaction(nuillu_reward::reward_session_auto_compaction())
                            .await?,
                    ))
                }
            })
        }
        RuntimeModule::Predict => {
            let main_tier = spec.model_tier("main");
            registry.register_server(spec, move |caps| async move {
                Ok(nuillu_predict::PredictModule::new(
                    caps.cognition_log_updated_inbox(),
                    caps.cognition_log_reader(),
                    caps.memo(),
                    caps.llm("main").with_tier(main_tier).into(),
                    caps.session("main")
                        .with_tier(main_tier)
                        .with_auto_compaction(nuillu_predict::session_auto_compaction())
                        .await?,
                ))
            })
        }
        RuntimeModule::Surprise => {
            let main_tier = spec.model_tier("main");
            registry.register_server(spec, move |caps| async move {
                Ok(nuillu_surprise::SurpriseModule::new(
                    caps.cognition_log_updated_inbox(),
                    caps.cognition_log_reader(),
                    caps.blackboard_reader(),
                    caps.attention_control_mailbox(),
                    caps.memo(),
                    caps.llm("main").with_tier(main_tier).into(),
                    caps.session("main")
                        .with_tier(main_tier)
                        .with_auto_compaction(nuillu_surprise::session_auto_compaction())
                        .await?,
                ))
            })
        }
        RuntimeModule::SubsystemGate => registry.register_server(spec, move |caps| async move {
            let owner = caps.owner().clone();
            let outer_updates = caps.outer_cognition_log_updated_inbox().ok_or_else(|| {
                nuillu_module::ModuleRegistryError::MissingOuterScope {
                    owner: owner.clone(),
                }
            })?;
            let outer_reader = caps.outer_cognition_log_reader().ok_or_else(|| {
                nuillu_module::ModuleRegistryError::MissingOuterScope {
                    owner: owner.clone(),
                }
            })?;
            let outer_memo = caps
                .outer_typed_memo::<nuillu_subsystem_gate::SubsystemGateMemo>()
                .ok_or(nuillu_module::ModuleRegistryError::MissingOuterScope { owner })?;
            Ok(nuillu_subsystem_gate::SubsystemGateModule::new(
                caps.cognition_log_updated_inbox().broadcast().coalesce(),
                caps.cognition_log_reader(),
                caps.typed_memo::<nuillu_subsystem_gate::SubsystemGateMemo>(),
                outer_updates.broadcast().coalesce(),
                outer_reader,
                outer_memo,
            ))
        }),
        RuntimeModule::SubsystemAllocation => {
            let main_tier = spec.model_tier("main");
            registry.register_server(spec, move |caps| {
                let catalog = caps.subsystem_catalog().children();
                let activation_tables = catalog
                    .iter()
                    .map(|item| {
                        (
                            item.subsystem.clone(),
                            item.activation_table.iter().copied().collect(),
                        )
                    })
                    .collect::<Vec<_>>();
                async move {
                    Ok(nuillu_subsystem_allocation::SubsystemAllocationModule::new(
                        caps.memo_updated_inbox(),
                        caps.blackboard_reader(),
                        caps.subsystem_allocation_reader(),
                        caps.subsystem_allocation_writer(activation_tables),
                        catalog,
                        caps.llm("main").with_tier(main_tier).into(),
                        caps.session("main")
                            .with_tier(main_tier)
                            .with_auto_compaction(
                                nuillu_subsystem_allocation::session_auto_compaction(),
                            )
                            .await?,
                    ))
                }
            })
        }
        RuntimeModule::Speak => {
            let utterance_sink = utterance_sink.clone();
            let planning_tier = spec.model_tier("planning");
            let speech_targets = nuillu_speak::SpeechTargetCatalog::new(spec.speech_targets());
            registry.register_server(spec, move |caps| {
                let utterance_sink = utterance_sink.clone();
                let speech_targets = speech_targets.clone();
                async move {
                    Ok(nuillu_speak::SpeakModule::new(
                        nuillu_speak::SpeakModuleParts {
                            cognition_updates: caps.cognition_log_updated_inbox(),
                            cognition_log: caps.cognition_log_reader(),
                            attention_control: caps.attention_control_mailbox(),
                            memo: caps.memo(),
                            utterance: UtteranceWriter::new(
                                caps.owner().clone(),
                                caps.blackboard(),
                                utterance_sink.clone(),
                                caps.clock(),
                            ),
                            planning_llm: caps.llm("planning").with_tier(planning_tier).into(),
                            scene: caps.scene_reader(),
                            speech_targets,
                            clock: caps.clock(),
                            planning_session: caps
                                .session("planning")
                                .with_tier(planning_tier)
                                .with_auto_compaction(
                                    nuillu_speak::planning_session_auto_compaction(),
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
        );
    }
    allocation
}

fn configured_dependency_edges(modules: &[ServerModuleSpec]) -> Vec<(ModuleId, ModuleId)> {
    let active = modules
        .iter()
        .map(ServerModuleSpec::module_id)
        .collect::<std::collections::HashSet<_>>();
    let mut edges = Vec::new();
    for module in modules {
        let dependent = module.module_id();
        for dependency in &module.depends_on {
            let dependency = dependency.as_module_id().clone();
            if active.contains(&dependency) {
                edges.push((dependent.clone(), dependency));
            }
        }
    }
    edges
}

fn configured_activation_barriers(
    modules: &[ServerModuleSpec],
) -> Vec<(ModuleId, Vec<ModuleId>, Option<std::time::Duration>)> {
    modules
        .iter()
        .filter_map(|module| {
            let barrier = module.activation_barrier.as_ref()?;
            Some((
                module.module_id(),
                barrier
                    .prerequisites
                    .iter()
                    .map(|module| module.as_module_id().clone())
                    .collect(),
                barrier.timeout(),
            ))
        })
        .collect()
}

#[cfg(test)]
fn group_modules(boot_config: &ServerBootConfig, group: ServerModuleGroup) -> Vec<ModuleId> {
    boot_config
        .specs_in_group(group)
        .into_iter()
        .map(ServerModuleSpec::module_id)
        .collect()
}

fn policy(module: &ServerModuleSpec) -> ModulePolicy {
    ModulePolicy::with_projections(
        module.replica_range(),
        Bpm::range(module.bpm_min, module.bpm_max),
        replica_projection(module.replica_projection()),
        rate_projection(module.rate_projection()),
    )
}

fn replica_projection(spec: ServerProjectionSpec) -> ReplicaProjection {
    match spec.curve {
        ServerProjectionCurve::Linear => ReplicaProjection::Linear,
        ServerProjectionCurve::Threshold => ReplicaProjection::Threshold(
            ActivationRatio::from_f64(spec.threshold.expect("validated threshold")),
        ),
    }
}

fn rate_projection(spec: ServerProjectionSpec) -> RateProjection {
    match spec.curve {
        ServerProjectionCurve::Linear => RateProjection::Linear,
        ServerProjectionCurve::Threshold => RateProjection::Threshold(ActivationRatio::from_f64(
            spec.threshold.expect("validated threshold"),
        )),
    }
}

fn set_allocation_module(allocation: &mut ResourceAllocation, id: ModuleId, activation_ratio: f64) {
    allocation.set_activation(id, ActivationRatio::from_f64(activation_ratio));
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::sync::atomic::{AtomicUsize, Ordering};

    use crate::config::{ServerModelSlotSpec, ServerModelTier, parse_server_boot_config_content};
    use lutum::Session;
    use nuillu_blackboard::Blackboard;
    use nuillu_memory::NoopMemoryStore;
    use nuillu_module::ports::SystemClock;
    use nuillu_module::{
        ActionAffordanceReader, ActionAffordancesUpdatedInbox, AllocationReader, AllocationWriter,
        BlackboardReader, CognitionLogReader, CognitionLogUpdatedInbox, CognitionWriter,
        ExternalActionInvoker, InteroceptiveReader, InteroceptiveUpdatedInbox, InteroceptiveWriter,
        LlmAccess, Memo, MemoUpdatedInbox,
    };
    use nuillu_reward::NoopPolicyStore;
    use nuillu_speak::NoopUtteranceSink;
    use nuillu_types::builtin;

    struct DynamicServerModule;

    fn external_spec(id: &str) -> ServerModuleSpec {
        let mut spec = ServerBootConfig::default().modules[0].clone();
        spec.id = super::super::config::ConfiguredModuleId::new(id).unwrap();
        spec.replica_min = 1;
        spec.replica_max = 1;
        spec.replica_capacity = 1;
        spec.model_slots.clear();
        spec.groups.clear();
        spec.depends_on.clear();
        spec.activation_barrier = None;
        spec.memo_sources = None;
        spec.scope_targets.clear();
        spec
    }

    fn dynamic_factory(descriptor: ServerModuleDescriptor) -> Arc<dyn ServerModuleFactory> {
        Arc::new(ServerModuleFactoryFn::new(
            descriptor,
            |slot: ServerModuleSlot, _config: &ResolvedServerModuleConfig| {
                Ok(slot.with_builder(|_caps| async { Ok(DynamicServerModule) })?)
            },
        ))
    }

    #[async_trait::async_trait(?Send)]
    impl nuillu_module::Module for DynamicServerModule {
        type Batch = ();

        async fn next_batch(&mut self) -> anyhow::Result<Self::Batch> {
            std::future::pending().await
        }

        async fn activate(
            &mut self,
            _cx: &nuillu_module::ActivateCx<'_>,
            _batch: &Self::Batch,
        ) -> anyhow::Result<()> {
            Ok(())
        }
    }

    type InterpreterConstructor = fn(
        CognitionLogUpdatedInbox,
        CognitionLogReader,
        CognitionWriter,
        LlmAccess,
        Session,
    ) -> nuillu_interpreter::InterpreterModule;

    type ActionConstructor = fn(
        MemoUpdatedInbox,
        CognitionLogUpdatedInbox,
        InteroceptiveUpdatedInbox,
        ActionAffordancesUpdatedInbox,
        BlackboardReader,
        CognitionLogReader,
        AllocationReader,
        InteroceptiveReader,
        ActionAffordanceReader,
        ExternalActionInvoker,
        AllocationWriter,
        InteroceptiveWriter,
        Memo,
        LlmAccess,
        Session,
    ) -> nuillu_action::ActionModule;

    #[test]
    fn interpreter_constructor_uses_only_direct_cognition_capabilities() {
        fn accepts_direct_cognition_signature(_constructor: InterpreterConstructor) {}

        accepts_direct_cognition_signature(nuillu_interpreter::InterpreterModule::new);
    }

    #[test]
    fn new_action_module_constructors_use_expected_capabilities() {
        fn accepts_action_signature(_constructor: ActionConstructor) {}

        accepts_action_signature(nuillu_action::ActionModule::new);
    }

    #[test]
    fn configured_dependencies_ignore_absent_modules() {
        let mut boot_config = ServerBootConfig::default();
        boot_config
            .modules
            .retain(|module| module.id != RuntimeModule::Policy);

        let edges = configured_dependency_edges(&boot_config.modules);

        assert!(!edges.contains(&(builtin::cognition_gate(), builtin::policy())));
        assert!(edges.contains(&(builtin::cognition_gate(), builtin::sensory())));
    }

    #[test]
    fn configured_activation_barriers_preserve_prerequisites_and_timeout() {
        let mut boot_config = ServerBootConfig::default();
        let speak = boot_config
            .modules
            .iter_mut()
            .find(|module| module.id == RuntimeModule::Speak)
            .unwrap();
        speak.activation_barrier = Some(ServerActivationBarrierSpec {
            prerequisites: vec![
                RuntimeModule::Sensory.into(),
                RuntimeModule::CognitionGate.into(),
            ],
            timeout_seconds: Some(4.5),
        });

        assert_eq!(
            configured_activation_barriers(&boot_config.modules),
            vec![(
                builtin::speak(),
                vec![builtin::sensory(), builtin::cognition_gate()],
                Some(std::time::Duration::from_millis(4_500)),
            )]
        );
    }

    #[test]
    fn group_modules_are_data_driven() {
        let boot_config = ServerBootConfig::default();

        let voluntary = group_modules(&boot_config, ServerModuleGroup::Voluntary);
        let action_targets = group_modules(&boot_config, ServerModuleGroup::ActionTarget);
        let drive = group_modules(&boot_config, ServerModuleGroup::HomeostaticDrive);

        assert!(voluntary.contains(&builtin::action()));
        assert!(!voluntary.contains(&builtin::speak()));
        assert!(action_targets.contains(&builtin::speak()));
        assert_eq!(action_targets, vec![builtin::speak()]);
        assert!(voluntary.contains(&builtin::interpreter()));
        assert!(voluntary.contains(&builtin::dreaming()));
        assert!(!voluntary.contains(&builtin::sensory()));
        assert!(drive.contains(&builtin::memory_compaction()));
        assert!(drive.contains(&builtin::dreaming()));
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
        assert!(!allocation.has_activation(&builtin::policy()));
    }

    #[test]
    fn host_factories_are_reusable_for_agent_rebuilds() {
        let calls = Arc::new(AtomicUsize::new(0));
        let module_ids = Arc::new(vec![
            ModuleId::new("mcp-files").unwrap(),
            ModuleId::new("mcp-issues").unwrap(),
        ]);
        let factories = module_ids
            .iter()
            .cloned()
            .map(|module| {
                let calls = calls.clone();
                Arc::new(ServerModuleFactoryFn::new(
                    ServerModuleDescriptor::new(module),
                    move |slot: ServerModuleSlot, _config: &ResolvedServerModuleConfig| {
                        calls.fetch_add(1, Ordering::Relaxed);
                        Ok(slot.with_builder(|_caps| async { Ok(DynamicServerModule) })?)
                    },
                )) as Arc<dyn ServerModuleFactory>
            })
            .collect::<Vec<_>>();
        let build_registry = || {
            factories
                .iter()
                .try_fold(ModuleRegistry::new(), |registry, factory| {
                    let module = factory.descriptor().id.clone();
                    let registration = ModuleRegistrationSpec::new(
                        module.clone(),
                        ModulePolicy::new(
                            nuillu_types::ReplicaCapRange::new(1, 1).unwrap(),
                            Bpm::from_f64(60.0)..=Bpm::from_f64(60.0),
                            nuillu_blackboard::linear_ratio_fn,
                        ),
                        ActivationRatio::ONE,
                    );
                    let config = ResolvedServerModuleConfig {
                        scope: ScopeId::root(),
                        spec: ServerBootConfig::default().modules[0].clone(),
                        model_tiers: BTreeMap::new(),
                    };
                    factory
                        .implement(
                            ServerModuleSlot {
                                registry,
                                registration,
                            },
                            &config,
                        )
                        .map(|filled| filled.registry)
                })
        };

        let first = build_registry().unwrap();
        let second = build_registry().unwrap();

        assert_eq!(calls.load(Ordering::Relaxed), 4);
        for module in module_ids.iter() {
            assert!(format!("{first:?}").contains(module.as_str()));
            assert!(format!("{second:?}").contains(module.as_str()));
        }
    }

    #[test]
    fn catalog_validation_rejects_missing_factory_and_unknown_model_slot() {
        let path = Path::new("agent/config.eure");
        let mut boot_config = ServerBootConfig {
            modules: vec![external_spec("code")],
            ..Default::default()
        };
        let error = validate_server_module_factories(path, &boot_config, &[]).unwrap_err();
        assert!(matches!(
            error,
            ServerModuleConfigError::MissingFactory { module, .. } if module.as_str() == "code"
        ));

        boot_config.modules[0].model_slots = vec![ServerModelSlotSpec {
            key: "conflict".to_owned(),
            tier: ServerModelTier::Premium,
        }];
        let factories = vec![dynamic_factory(ServerModuleDescriptor::new(
            ModuleId::new("code").unwrap(),
        ))];
        let error = validate_server_module_factories(path, &boot_config, &factories).unwrap_err();
        assert!(matches!(
            error,
            ServerModuleConfigError::UnknownModelSlot { module, slot, .. }
                if module.as_str() == "code" && slot == "conflict"
        ));
    }

    #[test]
    fn catalog_validation_rejects_replica_capacity_above_factory_limit() {
        let path = Path::new("agent/config.eure");
        let mut boot_config = ServerBootConfig::default();
        let mut code = external_spec("code");
        code.replica_capacity = 2;
        boot_config.modules = vec![code];
        let factories = vec![dynamic_factory(
            ServerModuleDescriptor::new(ModuleId::new("code").unwrap())
                .with_max_replica_capacity(1),
        )];
        let error = validate_server_module_factories(path, &boot_config, &factories).unwrap_err();
        assert!(matches!(
            error,
            ServerModuleConfigError::ReplicaCapacityAboveFactoryLimit {
                actual: 2,
                max: 1,
                ..
            }
        ));
    }

    #[test]
    fn catalog_validation_enforces_root_only_and_instance_count() {
        let path = Path::new("agent/config.eure");
        let boot_config = parse_server_boot_config_content(
            r#"
@ modules[] {
  id: code
  replica-min = 1
  replica-max = 1
  replica-capacity = 1
  bpm-min = 1.0
  bpm-max = 1.0
  initial-activation = 1.0
}
@ subsystem-definitions[] {
  id: arm
  allocation-description = "Test arm subsystem."
  @ modules[] {
    id: code
    replica-min = 1
    replica-max = 1
    replica-capacity = 1
    bpm-min = 1.0
    bpm-max = 1.0
    initial-activation = 1.0
  }
}
@ subsystems[] {
  subsystem: arm
  replicas = 1
}
"#,
            path,
        )
        .unwrap();
        let root_only = vec![dynamic_factory(
            ServerModuleDescriptor::new(ModuleId::new("code").unwrap()).root_only(),
        )];
        let error = validate_server_module_factories(path, &boot_config, &root_only).unwrap_err();
        assert!(matches!(error, ServerModuleConfigError::RootOnly { .. }));

        let single_instance = vec![dynamic_factory(
            ServerModuleDescriptor::new(ModuleId::new("code").unwrap())
                .with_max_configured_instances(1),
        )];
        let error =
            validate_server_module_factories(path, &boot_config, &single_instance).unwrap_err();
        assert!(matches!(
            error,
            ServerModuleConfigError::TooManyInstances {
                actual: 2,
                max: 1,
                ..
            }
        ));
    }

    #[test]
    fn catalog_validation_rejects_duplicate_and_builtin_factories() {
        let path = Path::new("agent/config.eure");
        let descriptor = || ServerModuleDescriptor::new(ModuleId::new("code").unwrap());
        let duplicates = vec![dynamic_factory(descriptor()), dynamic_factory(descriptor())];
        let error =
            validate_server_module_factories(path, &ServerBootConfig::default(), &duplicates)
                .unwrap_err();
        assert!(matches!(
            error,
            ServerModuleConfigError::DuplicateFactory { .. }
        ));

        let builtin = vec![dynamic_factory(ServerModuleDescriptor::new(
            builtin::speak(),
        ))];
        let error = validate_server_module_factories(path, &ServerBootConfig::default(), &builtin)
            .unwrap_err();
        assert!(matches!(
            error,
            ServerModuleConfigError::BuiltinFactoryConflict { .. }
        ));
    }

    #[test]
    fn config_driven_host_factory_receives_resolved_model_tier() {
        let calls = Arc::new(AtomicUsize::new(0));
        let factory_calls = calls.clone();
        let descriptor = ServerModuleDescriptor::new(ModuleId::new("code").unwrap())
            .with_peer_context("Inspects and edits the configured workspace.")
            .with_model_slot("main", ModelTier::Default);
        let factory: Arc<dyn ServerModuleFactory> = Arc::new(ServerModuleFactoryFn::new(
            descriptor,
            move |slot: ServerModuleSlot, config: &ResolvedServerModuleConfig| {
                assert_eq!(config.scope(), &ScopeId::root());
                assert_eq!(config.model_tier("main"), Some(ModelTier::Premium));
                factory_calls.fetch_add(1, Ordering::Relaxed);
                Ok(slot.with_builder(|_caps| async { Ok(DynamicServerModule) })?)
            },
        ));
        let factories = vec![factory];
        let mut code = external_spec("code");
        code.groups = vec![ServerModuleGroup::Voluntary.into()];
        code.model_slots = vec![ServerModelSlotSpec {
            key: "main".to_owned(),
            tier: ServerModelTier::Premium,
        }];
        let boot_config = ServerBootConfig {
            modules: vec![code],
            ..Default::default()
        };
        let config_path = Path::new("<memory>");
        let catalog = ServerModuleCatalog::new(config_path, &factories).unwrap();
        validate_configured_modules(config_path, &boot_config, &catalog).unwrap();

        let blackboard = Blackboard::new();
        let clock = Rc::new(SystemClock);
        let memory = MemoryCapabilities::new(
            blackboard.clone(),
            clock.clone(),
            Rc::new(NoopMemoryStore),
            Vec::new(),
        );
        let policy =
            PolicyCapabilities::new(blackboard, clock, Rc::new(NoopPolicyStore), Vec::new());
        let sink: Rc<dyn UtteranceSink> = Rc::new(NoopUtteranceSink);
        let registry =
            server_registry(config_path, &boot_config, &catalog, &memory, &policy, &sink).unwrap();

        assert_eq!(calls.load(Ordering::Relaxed), 1);
        let debug = format!("{registry:?}");
        assert!(debug.contains("code"), "{debug}");
        assert!(debug.contains("voluntary"), "{debug}");
        assert!(debug.contains("Inspects and edits"), "{debug}");
    }
}
