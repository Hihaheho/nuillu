use std::{
    collections::{BTreeMap, HashSet},
    fs, io,
    num::NonZeroUsize,
    path::{Path, PathBuf},
    sync::{Arc, OnceLock},
};

use chrono::Utc;
use eure::FromEure;
use eure::document::{
    EureDocument,
    parse::{ParseContext, ParseError, ParseErrorKind},
};
use nuillu_module::{ActionAffordance, ScopeLabels};
use nuillu_types::{
    ModelTier, ModuleGroupId, ModuleId, ReplicaCapRange, ReplicaIndex, ScopeId, SubsystemId,
    SubsystemInstanceId, builtin,
};
use tracing_subscriber::layer::SubscriberExt as _;
use uuid::Uuid;

use crate::model_set::{
    EmbeddingRole, ModelSet, ReasoningEffort, parse_model_set_file, resolve_llm_backends,
    resolve_token_fields,
};

const DEFAULT_OPENAI_EMBEDDING_ENDPOINT: &str = "https://api.openai.com/v1";
const AGENT_DB_FILE: &str = "agent.db";
const STATE_MODEL_SET_FILE: &str = "model-set.eure";

#[derive(Debug, Clone)]
pub struct ServerConfig {
    pub state_dir: PathBuf,
    pub agent_db_path: PathBuf,
    pub session_id: String,
    /// Root for file-based LLM traces. `None` disables file trace output.
    pub llm_log_root: Option<PathBuf>,
    pub cheap_backend: LlmBackendConfig,
    pub default_backend: LlmBackendConfig,
    pub premium_backend: LlmBackendConfig,
    pub image_backend: LlmBackendConfig,
    pub embedding_backend: EmbeddingBackendConfig,
    pub boot_config: ServerBootConfig,
    pub disabled_modules: Vec<RuntimeModule>,
    pub participants: Vec<String>,
    pub fresh_agent_db: bool,
    /// Starts the agent stopped, without activating modules, until Run or sensory input resumes it.
    pub start_paused: bool,
}

#[derive(Debug, Clone)]
pub struct ServerRunOptions {
    pub state_dir: PathBuf,
    pub run_id: Option<String>,
    pub session_id: Option<String>,
    pub llm_log_root: PathBuf,
    pub model_set: Option<PathBuf>,
    pub disabled_modules: Vec<RuntimeModule>,
    pub participants: Vec<String>,
    pub fresh_agent_db: bool,
    pub agent_db: Option<PathBuf>,
}

#[derive(Debug, Clone)]
pub struct LlmBackendConfig {
    pub model_key: String,
    pub endpoint: String,
    pub token: String,
    pub model: String,
    pub reasoning: bool,
    pub reasoning_effort: Option<ReasoningEffort>,
    pub generation: LlmGenerationConfig,
    pub use_responses_api: bool,
    pub compaction_input_token_threshold: u64,
    pub max_concurrent_llm_calls: Option<NonZeroUsize>,
    /// Ordered model backends tried after a provider request failure.
    pub fallbacks: Vec<LlmBackendConfig>,
}

#[derive(Debug, Clone, Default, PartialEq, FromEure)]
#[eure(crate = ::eure::document, rename_all = "kebab-case")]
pub struct LlmGenerationConfig {
    #[eure(default)]
    pub temperature: Option<f64>,
    #[eure(default)]
    pub top_p: Option<f64>,
    #[eure(default)]
    pub top_k: Option<u32>,
    #[eure(default)]
    pub frequency_penalty: Option<f64>,
    #[eure(default)]
    pub presence_penalty: Option<f64>,
    #[eure(default)]
    pub max_output_tokens: Option<u32>,
    #[eure(default)]
    pub seed: Option<u64>,
    #[eure(default)]
    pub stop_sequences: Option<Vec<String>>,
}

#[derive(Debug, Clone)]
pub struct EmbeddingBackendConfig {
    pub endpoint: String,
    pub token: String,
    pub model: String,
    pub dimensions: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, clap::ValueEnum, FromEure)]
#[eure(crate = ::eure::document, rename_all = "kebab-case")]
pub enum RuntimeModule {
    Sensory,
    CognitionGate,
    Allocation,
    Action,
    AttentionSchema,
    Interpreter,
    SelfModel,
    QueryMemory,
    Memory,
    MemoryCompaction,
    MemoryAssociation,
    Dreaming,
    Interoception,
    Homeostasis,
    Policy,
    PolicyCompaction,
    Reward,
    Predict,
    Surprise,
    SubsystemGate,
    Speak,
}

const SERVER_BOOT_CONFIG_FILE: &str = "config.eure";

#[derive(Debug, Clone, FromEure)]
#[eure(crate = ::eure::document, rename_all = "kebab-case")]
pub struct ServerBootConfig {
    #[eure(default = "default_activation_table_values")]
    pub activation_table: Vec<f64>,
    #[eure(default)]
    pub modules: Vec<ServerModuleSpec>,
    #[eure(default)]
    pub subsystem_definitions: Vec<ServerSubsystemDef>,
    #[eure(default)]
    pub subsystems: Vec<ServerSubsystemRef>,
    #[eure(default)]
    pub actions: Vec<ServerActionSpec>,
}

#[derive(Debug, Clone, PartialEq, FromEure)]
#[eure(crate = ::eure::document, rename_all = "kebab-case")]
pub struct ServerSubsystemDef {
    pub id: String,
    #[eure(default)]
    pub label: Option<String>,
    #[eure(default)]
    pub modules: Vec<ServerModuleSpec>,
    pub root: String,
    #[eure(default)]
    pub memory_scope: ServerMemoryScope,
    #[eure(default)]
    pub subsystems: Vec<ServerSubsystemRef>,
}

impl ServerSubsystemDef {
    pub fn subsystem_id(&self) -> SubsystemId {
        SubsystemId::new(self.id.clone()).expect("validated subsystem id")
    }

    pub fn root_module_id(&self) -> ModuleId {
        ModuleId::new(self.root.clone()).expect("validated subsystem root module id")
    }
}

#[derive(Debug, Clone, PartialEq, FromEure)]
#[eure(crate = ::eure::document, rename_all = "kebab-case")]
pub struct ServerSubsystemRef {
    pub subsystem: String,
    pub replicas: u8,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, FromEure)]
#[eure(crate = ::eure::document, rename_all = "kebab-case")]
pub enum ServerMemoryScope {
    #[default]
    Global,
    Local,
}

#[derive(Debug, Clone)]
pub struct ExpandedSubsystem<'a> {
    pub scope: ScopeId,
    pub definition: &'a ServerSubsystemDef,
}

#[derive(Debug, Clone, PartialEq, FromEure)]
#[eure(crate = ::eure::document, rename_all = "kebab-case")]
pub struct ServerModuleSpec {
    pub id: RuntimeModule,
    pub replica_min: u8,
    pub replica_max: u8,
    #[eure(default = "default_replica_capacity")]
    pub replica_capacity: u8,
    pub bpm_min: f64,
    pub bpm_max: f64,
    pub initial_activation: f64,
    #[eure(default)]
    pub sessions: Vec<ServerModuleSessionSpec>,
    #[eure(default)]
    pub groups: Vec<ServerModuleGroup>,
    #[eure(default)]
    pub depends_on: Vec<RuntimeModule>,
    /// Module roles whose memo updates this module observes. Omitting the key
    /// keeps the default of every role; an explicit empty list subscribes to
    /// none. Every listed role must also be registered in this config.
    #[eure(default)]
    pub memo_sources: Option<Vec<RuntimeModule>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, FromEure)]
#[eure(crate = ::eure::document, rename_all = "kebab-case")]
pub enum ServerSessionTier {
    Cheap,
    #[default]
    Default,
    Premium,
    Image,
}

impl From<ServerSessionTier> for ModelTier {
    fn from(value: ServerSessionTier) -> Self {
        match value {
            ServerSessionTier::Cheap => Self::Cheap,
            ServerSessionTier::Default => Self::Default,
            ServerSessionTier::Premium => Self::Premium,
            ServerSessionTier::Image => Self::Image,
        }
    }
}

#[derive(Debug, Clone, PartialEq, FromEure)]
#[eure(crate = ::eure::document, rename_all = "kebab-case")]
pub struct ServerModuleSessionSpec {
    pub key: String,
    pub tier: ServerSessionTier,
}

#[derive(Debug, Clone, PartialEq, FromEure)]
#[eure(crate = ::eure::document, rename_all = "kebab-case")]
pub struct ServerActionSpec {
    pub name: String,
    #[eure(default)]
    pub label: Option<String>,
    pub description: String,
    #[eure(default)]
    pub use_when: Option<String>,
    #[eure(default)]
    pub effect: Option<String>,
    #[eure(rename = "json_schema")]
    json_schema: ServerActionJsonSchema,
}

#[derive(Debug, Clone, PartialEq)]
struct ServerActionJsonSchema(serde_json::Value);

impl eure::document::parse::FromEure<'_> for ServerActionJsonSchema {
    type Error = ParseError;

    fn parse(ctx: &ParseContext<'_>) -> Result<Self, Self::Error> {
        let doc: EureDocument = ctx.parse()?;
        let value =
            eure_json::document_to_value(&doc, &eure_json::Config::default()).map_err(|error| {
                ParseError {
                    node_id: ctx.node_id(),
                    kind: ParseErrorKind::InvalidPattern {
                        kind: "json_schema".to_owned(),
                        reason: error.to_string(),
                    },
                }
            })?;
        Ok(Self(value))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, FromEure)]
#[eure(crate = ::eure::document, rename_all = "kebab-case")]
pub enum ServerModuleGroup {
    Voluntary,
    SleepSuppressed,
    HomeostaticDrive,
    ActionTarget,
}

impl ServerModuleGroup {
    pub fn module_group_id(self) -> ModuleGroupId {
        let id = match self {
            Self::Voluntary => "voluntary",
            Self::SleepSuppressed => "sleep-suppressed",
            Self::HomeostaticDrive => "homeostatic-drive",
            Self::ActionTarget => "action-target",
        };
        ModuleGroupId::new(id).expect("server module group ids are valid")
    }
}

pub const DEFAULT_MODULES: &[RuntimeModule] = &[
    RuntimeModule::Sensory,
    RuntimeModule::CognitionGate,
    RuntimeModule::Allocation,
    RuntimeModule::Action,
    RuntimeModule::AttentionSchema,
    RuntimeModule::Interpreter,
    RuntimeModule::SelfModel,
    RuntimeModule::QueryMemory,
    RuntimeModule::Memory,
    RuntimeModule::MemoryCompaction,
    RuntimeModule::MemoryAssociation,
    RuntimeModule::Dreaming,
    RuntimeModule::Interoception,
    RuntimeModule::Homeostasis,
    RuntimeModule::Policy,
    RuntimeModule::PolicyCompaction,
    RuntimeModule::Reward,
    RuntimeModule::Predict,
    RuntimeModule::Surprise,
    RuntimeModule::Speak,
];

impl Default for ServerBootConfig {
    fn default() -> Self {
        Self {
            activation_table: default_activation_table_values(),
            modules: default_server_modules(),
            subsystem_definitions: Vec::new(),
            subsystems: Vec::new(),
            actions: Vec::new(),
        }
    }
}

impl RuntimeModule {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Sensory => "sensory",
            Self::CognitionGate => "cognition-gate",
            Self::Allocation => "allocation",
            Self::Action => "action",
            Self::AttentionSchema => "attention-schema",
            Self::Interpreter => "interpreter",
            Self::SelfModel => "self-model",
            Self::QueryMemory => "query-memory",
            Self::Memory => "memory",
            Self::MemoryCompaction => "memory-compaction",
            Self::MemoryAssociation => "memory-association",
            Self::Dreaming => "dreaming",
            Self::Interoception => "interoception",
            Self::Homeostasis => "homeostasis",
            Self::Policy => "policy",
            Self::PolicyCompaction => "policy-compaction",
            Self::Reward => "reward",
            Self::Predict => "predict",
            Self::Surprise => "surprise",
            Self::SubsystemGate => "subsystem-gate",
            Self::Speak => "speak",
        }
    }

    pub fn module_id(self) -> ModuleId {
        match self {
            Self::Sensory => builtin::sensory(),
            Self::CognitionGate => builtin::cognition_gate(),
            Self::Allocation => builtin::allocation(),
            Self::Action => builtin::action(),
            Self::AttentionSchema => builtin::attention_schema(),
            Self::Interpreter => builtin::interpreter(),
            Self::SelfModel => builtin::self_model(),
            Self::QueryMemory => builtin::query_memory(),
            Self::Memory => builtin::memory(),
            Self::MemoryCompaction => builtin::memory_compaction(),
            Self::MemoryAssociation => builtin::memory_association(),
            Self::Dreaming => builtin::dreaming(),
            Self::Interoception => builtin::interoception(),
            Self::Homeostasis => builtin::homeostasis(),
            Self::Policy => builtin::policy(),
            Self::PolicyCompaction => builtin::policy_compaction(),
            Self::Reward => builtin::reward(),
            Self::Predict => builtin::predict(),
            Self::Surprise => builtin::surprise(),
            Self::SubsystemGate => builtin::subsystem_gate(),
            Self::Speak => builtin::speak(),
        }
    }

    pub fn session_defaults(self) -> &'static [(&'static str, ModelTier)] {
        match self {
            Self::Sensory => &[
                ("one-shot", ModelTier::Cheap),
                ("ambient", ModelTier::Cheap),
            ],
            Self::CognitionGate => &[("main", ModelTier::Default)],
            Self::Allocation => &[("main", ModelTier::Default)],
            Self::Action => &[("main", ModelTier::Default)],
            Self::AttentionSchema => &[("main", ModelTier::Default)],
            Self::Interpreter => &[("main", ModelTier::Default)],
            Self::SelfModel => &[("main", ModelTier::Default)],
            Self::QueryMemory => &[("main", ModelTier::Cheap)],
            Self::Memory => &[("main", ModelTier::Cheap)],
            Self::MemoryCompaction => &[("main", ModelTier::Cheap), ("audit", ModelTier::Default)],
            Self::MemoryAssociation => &[("main", ModelTier::Cheap)],
            Self::Dreaming => &[("main", ModelTier::Cheap)],
            Self::Interoception => &[("main", ModelTier::Cheap)],
            Self::Homeostasis => &[],
            Self::Policy => &[("main", ModelTier::Default)],
            Self::PolicyCompaction => &[("main", ModelTier::Cheap)],
            Self::Reward => &[("main", ModelTier::Default)],
            Self::Predict => &[("main", ModelTier::Cheap)],
            Self::Surprise => &[("main", ModelTier::Default)],
            Self::SubsystemGate => &[],
            Self::Speak => &[("planning", ModelTier::Premium)],
        }
    }
}

impl ServerConfig {
    /// Starts a configuration builder backed entirely by in-memory values.
    pub fn builder(model_set: ModelSet) -> ServerConfigBuilder {
        ServerConfigBuilder::new(model_set)
    }

    /// Builds a minimal configuration without reading configuration files.
    pub fn from_memory(
        model_set: ModelSet,
        enabled_modules: impl IntoIterator<Item = RuntimeModule>,
        participants: impl IntoIterator<Item = String>,
        session_id: impl Into<String>,
    ) -> anyhow::Result<Self> {
        Self::builder(model_set)
            .enabled_modules(enabled_modules)
            .participants(participants)
            .session_id(session_id)
            .build()
    }

    pub fn active_modules(&self) -> Vec<RuntimeModule> {
        self.boot_config.active_modules()
    }
}

/// Builds [`ServerConfig`] without loading a model set or boot config from disk.
#[derive(Debug, Clone)]
pub struct ServerConfigBuilder {
    model_set: ModelSet,
    state_dir: PathBuf,
    agent_db_path: Option<PathBuf>,
    session_id: Option<String>,
    llm_log_root: Option<PathBuf>,
    boot_config: ServerBootConfig,
    enabled_modules: Option<HashSet<RuntimeModule>>,
    disabled_modules: Vec<RuntimeModule>,
    participants: Vec<String>,
    fresh_agent_db: bool,
    start_paused: bool,
}

impl ServerConfigBuilder {
    pub fn new(model_set: ModelSet) -> Self {
        Self {
            model_set,
            state_dir: PathBuf::from("."),
            agent_db_path: None,
            session_id: None,
            llm_log_root: None,
            boot_config: ServerBootConfig::default(),
            enabled_modules: None,
            disabled_modules: Vec::new(),
            participants: Vec::new(),
            fresh_agent_db: false,
            start_paused: false,
        }
    }

    pub fn state_dir(mut self, state_dir: impl Into<PathBuf>) -> Self {
        self.state_dir = state_dir.into();
        self
    }

    pub fn agent_db_path(mut self, agent_db_path: impl Into<PathBuf>) -> Self {
        self.agent_db_path = Some(agent_db_path.into());
        self
    }

    pub fn session_id(mut self, session_id: impl Into<String>) -> Self {
        self.session_id = Some(session_id.into());
        self
    }

    pub fn file_llm_trace_root(mut self, root: impl Into<PathBuf>) -> Self {
        self.llm_log_root = Some(root.into());
        self
    }

    pub fn disable_file_llm_trace(mut self) -> Self {
        self.llm_log_root = None;
        self
    }

    pub fn boot_config(mut self, boot_config: ServerBootConfig) -> Self {
        self.boot_config = boot_config;
        self
    }

    pub fn enabled_modules(mut self, modules: impl IntoIterator<Item = RuntimeModule>) -> Self {
        self.enabled_modules = Some(modules.into_iter().collect());
        self
    }

    pub fn disabled_modules(mut self, modules: impl IntoIterator<Item = RuntimeModule>) -> Self {
        self.disabled_modules = modules.into_iter().collect();
        self
    }

    pub fn participants(mut self, participants: impl IntoIterator<Item = String>) -> Self {
        self.participants = participants.into_iter().collect();
        self
    }

    pub fn fresh_agent_db(mut self, fresh: bool) -> Self {
        self.fresh_agent_db = fresh;
        self
    }

    /// Chooses whether the agent starts stopped instead of running immediately.
    ///
    /// A stopped agent resumes when the host invokes Run or publishes sensory input.
    pub fn start_paused(mut self, start_paused: bool) -> Self {
        self.start_paused = start_paused;
        self
    }

    pub fn build(mut self) -> anyhow::Result<ServerConfig> {
        if let Some(enabled) = self.enabled_modules {
            self.boot_config
                .modules
                .retain(|module| enabled.contains(&module.id));
        }
        self.boot_config.validate(Path::new("<memory>"))?;
        let backends = resolve_llm_backends(&self.model_set)?;
        let embedding_backend = resolve_embedding(&self.model_set.embedding)?;
        let agent_db_path = self
            .agent_db_path
            .unwrap_or_else(|| self.state_dir.join(AGENT_DB_FILE));

        Ok(ServerConfig {
            state_dir: self.state_dir,
            agent_db_path,
            session_id: self.session_id.unwrap_or_else(default_server_session_id),
            llm_log_root: self.llm_log_root,
            cheap_backend: backends.cheap,
            default_backend: backends.default,
            premium_backend: backends.premium,
            image_backend: backends.image,
            embedding_backend,
            boot_config: self.boot_config,
            disabled_modules: self.disabled_modules,
            participants: self.participants,
            fresh_agent_db: self.fresh_agent_db,
            start_paused: self.start_paused,
        })
    }
}

impl ServerBootConfig {
    pub fn expanded_subsystems(&self) -> Vec<ExpandedSubsystem<'_>> {
        let definitions = self
            .subsystem_definitions
            .iter()
            .map(|definition| (definition.id.as_str(), definition))
            .collect::<BTreeMap<_, _>>();
        let mut expanded = Vec::new();
        expand_subsystem_refs(
            &ScopeId::root(),
            &self.subsystems,
            &definitions,
            &mut expanded,
        );
        expanded
    }

    pub fn scope_labels(&self) -> ScopeLabels {
        let definitions = self
            .subsystem_definitions
            .iter()
            .map(|definition| (definition.id.as_str(), definition))
            .collect::<BTreeMap<_, _>>();
        let mut labels = Vec::new();
        expand_scope_labels(
            &ScopeId::root(),
            &self.subsystems,
            &definitions,
            &mut labels,
        );
        ScopeLabels::new(labels)
    }

    pub fn active_modules(&self) -> Vec<RuntimeModule> {
        self.modules.iter().map(|module| module.id).collect()
    }

    pub fn active_module_ids(&self) -> HashSet<ModuleId> {
        self.modules
            .iter()
            .map(ServerModuleSpec::module_id)
            .collect()
    }

    pub fn specs_in_group(&self, group: ServerModuleGroup) -> Vec<&ServerModuleSpec> {
        self.modules
            .iter()
            .filter(|module| module.groups.contains(&group))
            .collect()
    }

    pub fn action_affordances(&self) -> Vec<ActionAffordance> {
        self.actions
            .iter()
            .map(ServerActionSpec::affordance)
            .collect()
    }

    pub fn overlay_action_affordances(&self, base: Vec<ActionAffordance>) -> Vec<ActionAffordance> {
        let mut affordances = base
            .into_iter()
            .map(|affordance| (affordance.id.clone(), affordance))
            .collect::<BTreeMap<_, _>>();
        for affordance in self.action_affordances() {
            affordances.insert(affordance.id.clone(), affordance);
        }
        affordances.into_values().collect()
    }

    fn validate(&self, path: &Path) -> anyhow::Result<()> {
        validate_activation_table(&self.activation_table, path)?;
        let mut seen = HashSet::new();
        for module in &self.modules {
            if !seen.insert(module.id) {
                anyhow::bail!(
                    "server config {} declares module {} more than once",
                    path.display(),
                    module.id.as_str()
                );
            }
            module.validate(path)?;
        }
        for module in &self.modules {
            let Some(sources) = &module.memo_sources else {
                continue;
            };
            let mut seen_sources = HashSet::new();
            for source in sources {
                if !seen_sources.insert(*source) {
                    anyhow::bail!(
                        "server config {} declares memo source {} for module {} more than once",
                        path.display(),
                        source.as_str(),
                        module.id.as_str()
                    );
                }
                if !seen.contains(source) {
                    anyhow::bail!(
                        "server config {} declares unknown memo source {} for module {}",
                        path.display(),
                        source.as_str(),
                        module.id.as_str()
                    );
                }
            }
        }
        self.validate_subsystems(path)?;
        validate_config_actions(&self.action_affordances(), path)?;
        Ok(())
    }

    fn validate_subsystems(&self, path: &Path) -> anyhow::Result<()> {
        let mut definitions = BTreeMap::new();
        for definition in &self.subsystem_definitions {
            SubsystemId::new(definition.id.clone()).map_err(|error| {
                anyhow::anyhow!(
                    "server config {} has invalid subsystem id {:?}: {error}",
                    path.display(),
                    definition.id
                )
            })?;
            if let Some(label) = &definition.label
                && (label.trim().is_empty() || label.contains(['\n', '\r']))
            {
                anyhow::bail!(
                    "server config {} has invalid label for subsystem {}: labels must be non-empty single-line text",
                    path.display(),
                    definition.id
                );
            }
            if definitions
                .insert(definition.id.as_str(), definition)
                .is_some()
            {
                anyhow::bail!(
                    "server config {} declares subsystem {} more than once",
                    path.display(),
                    definition.id
                );
            }
            validate_module_set(&definition.modules, path, &definition.id)?;
            let module_ids = definition
                .modules
                .iter()
                .map(ServerModuleSpec::module_id)
                .collect::<HashSet<_>>();
            let root = ModuleId::new(definition.root.clone()).map_err(|error| {
                anyhow::anyhow!(
                    "server config {} has invalid root {:?} in subsystem {}: {error}",
                    path.display(),
                    definition.root,
                    definition.id
                )
            })?;
            if !module_ids.contains(&root) {
                anyhow::bail!(
                    "server config {} declares unknown root {} in subsystem {}",
                    path.display(),
                    root.as_str(),
                    definition.id
                );
            }
        }
        validate_subsystem_refs(&self.subsystems, "root", &definitions, path)?;
        for definition in &self.subsystem_definitions {
            validate_subsystem_refs(&definition.subsystems, &definition.id, &definitions, path)?;
        }
        let mut visiting = Vec::new();
        let mut visited = HashSet::new();
        // Validate every definition, including definitions that are not mounted
        // at the root yet. This keeps the reusable definition catalog valid on
        // its own and prevents a latent cycle from surfacing only when mounted.
        for definition in &self.subsystem_definitions {
            validate_subsystem_cycles(
                definition.id.as_str(),
                &definitions,
                &mut visiting,
                &mut visited,
                path,
            )?;
        }
        Ok(())
    }
}

fn validate_module_set(
    modules: &[ServerModuleSpec],
    path: &Path,
    owner: &str,
) -> anyhow::Result<()> {
    let mut seen = HashSet::new();
    for module in modules {
        if !seen.insert(module.id) {
            anyhow::bail!(
                "server config {} declares module {} more than once in subsystem {}",
                path.display(),
                module.id.as_str(),
                owner
            );
        }
        module.validate(path)?;
    }
    for module in modules {
        for dependency in &module.depends_on {
            if !seen.contains(dependency) {
                anyhow::bail!(
                    "server config {} declares unknown dependency {} for module {} in subsystem {}",
                    path.display(),
                    dependency.as_str(),
                    module.id.as_str(),
                    owner
                );
            }
        }
        if let Some(sources) = &module.memo_sources {
            let mut seen_sources = HashSet::new();
            for source in sources {
                if !seen_sources.insert(source) {
                    anyhow::bail!(
                        "server config {} declares memo source {} for module {} more than once in subsystem {}",
                        path.display(),
                        source.as_str(),
                        module.id.as_str(),
                        owner
                    );
                }
                if !seen.contains(source) {
                    anyhow::bail!(
                        "server config {} declares unknown memo source {} for module {} in subsystem {}",
                        path.display(),
                        source.as_str(),
                        module.id.as_str(),
                        owner
                    );
                }
            }
        }
    }
    Ok(())
}

fn validate_subsystem_refs<'a>(
    references: &[ServerSubsystemRef],
    owner: &str,
    definitions: &BTreeMap<&'a str, &'a ServerSubsystemDef>,
    path: &Path,
) -> anyhow::Result<()> {
    let mut seen = HashSet::new();
    for reference in references {
        if reference.replicas == 0 {
            anyhow::bail!(
                "server config {} sets zero replicas for subsystem {} under {}",
                path.display(),
                reference.subsystem,
                owner
            );
        }
        if !definitions.contains_key(reference.subsystem.as_str()) {
            anyhow::bail!(
                "server config {} references unknown subsystem {} under {}",
                path.display(),
                reference.subsystem,
                owner
            );
        }
        if !seen.insert(reference.subsystem.as_str()) {
            anyhow::bail!(
                "server config {} references subsystem {} more than once under {}",
                path.display(),
                reference.subsystem,
                owner
            );
        }
    }
    Ok(())
}

fn validate_subsystem_cycles<'a>(
    id: &'a str,
    definitions: &BTreeMap<&'a str, &'a ServerSubsystemDef>,
    visiting: &mut Vec<&'a str>,
    visited: &mut HashSet<&'a str>,
    path: &Path,
) -> anyhow::Result<()> {
    if let Some(start) = visiting.iter().position(|candidate| *candidate == id) {
        let mut cycle = visiting[start..].to_vec();
        cycle.push(id);
        anyhow::bail!(
            "server config {} has recursive subsystem reference: {}",
            path.display(),
            cycle.join(" -> ")
        );
    }
    if !visited.insert(id) {
        return Ok(());
    }
    visiting.push(id);
    let definition = definitions[id];
    for child in &definition.subsystems {
        validate_subsystem_cycles(
            child.subsystem.as_str(),
            definitions,
            visiting,
            visited,
            path,
        )?;
    }
    visiting.pop();
    Ok(())
}

fn expand_subsystem_refs<'a>(
    parent: &ScopeId,
    references: &[ServerSubsystemRef],
    definitions: &BTreeMap<&'a str, &'a ServerSubsystemDef>,
    expanded: &mut Vec<ExpandedSubsystem<'a>>,
) {
    for reference in references {
        let definition = definitions[reference.subsystem.as_str()];
        for replica in 0..reference.replicas {
            let scope = parent.child(SubsystemInstanceId::new(
                definition.subsystem_id(),
                ReplicaIndex::new(replica),
            ));
            expanded.push(ExpandedSubsystem {
                scope: scope.clone(),
                definition,
            });
            expand_subsystem_refs(&scope, &definition.subsystems, definitions, expanded);
        }
    }
}

fn expand_scope_labels<'a>(
    parent: &ScopeId,
    references: &[ServerSubsystemRef],
    definitions: &BTreeMap<&'a str, &'a ServerSubsystemDef>,
    labels: &mut Vec<(ScopeId, Arc<str>)>,
) {
    for reference in references {
        let definition = definitions[reference.subsystem.as_str()];
        let label = definition
            .label
            .as_deref()
            .unwrap_or(definition.id.as_str())
            .trim();
        for replica in 0..reference.replicas {
            let scope = parent.child(SubsystemInstanceId::new(
                definition.subsystem_id(),
                ReplicaIndex::new(replica),
            ));
            let segment: Arc<str> = if reference.replicas == 1 {
                Arc::from(label)
            } else {
                Arc::from(format!("{label} {}", u16::from(replica) + 1))
            };
            labels.push((scope.clone(), segment));
            expand_scope_labels(&scope, &definition.subsystems, definitions, labels);
        }
    }
}

impl ServerModuleSpec {
    pub fn module_id(&self) -> ModuleId {
        self.id.module_id()
    }

    pub fn session_tier(&self, key: &str) -> ModelTier {
        self.sessions
            .iter()
            .find(|session| session.key == key)
            .map(|session| session.tier.into())
            .or_else(|| {
                self.id
                    .session_defaults()
                    .iter()
                    .find(|(candidate, _)| *candidate == key)
                    .map(|(_, tier)| *tier)
            })
            .unwrap_or_else(|| {
                panic!(
                    "unknown session key {key:?} for module {}; config validation should reject this",
                    self.id.as_str()
                )
            })
    }

    pub fn replica_range(&self) -> ReplicaCapRange {
        ReplicaCapRange::new(self.replica_min, self.replica_max)
            .expect("server module spec should be validated before use")
    }

    fn validate(&self, path: &Path) -> anyhow::Result<()> {
        ReplicaCapRange::new(self.replica_min, self.replica_max).map_err(|error| {
            anyhow::anyhow!(
                "server config {} has invalid replica range for {}: {error}",
                path.display(),
                self.id.as_str()
            )
        })?;
        if self.replica_capacity > ReplicaCapRange::V1_MAX {
            anyhow::bail!(
                "server config {} sets replica-capacity={} for {}, above v1 max {}",
                path.display(),
                self.replica_capacity,
                self.id.as_str(),
                ReplicaCapRange::V1_MAX
            );
        }
        if self.replica_capacity < self.replica_max.max(1) {
            anyhow::bail!(
                "server config {} sets replica-capacity={} for {}, below policy max {}",
                path.display(),
                self.replica_capacity,
                self.id.as_str(),
                self.replica_max.max(1)
            );
        }
        validate_finite_ratio(self.initial_activation, "initial-activation", self.id, path)?;
        if !self.bpm_min.is_finite() || !self.bpm_max.is_finite() || self.bpm_min <= 0.0 {
            anyhow::bail!(
                "server config {} has invalid bpm range for {}: {}..={}",
                path.display(),
                self.id.as_str(),
                self.bpm_min,
                self.bpm_max
            );
        }
        if self.bpm_min > self.bpm_max {
            anyhow::bail!(
                "server config {} has bpm-min greater than bpm-max for {}",
                path.display(),
                self.id.as_str()
            );
        }
        let defaults = self.id.session_defaults();
        let mut seen_sessions = HashSet::new();
        for session in &self.sessions {
            if !seen_sessions.insert(session.key.as_str()) {
                anyhow::bail!(
                    "server config {} declares session {} for module {} more than once",
                    path.display(),
                    session.key,
                    self.id.as_str()
                );
            }
            if defaults
                .iter()
                .all(|(default_key, _)| *default_key != session.key.as_str())
            {
                let expected = defaults
                    .iter()
                    .map(|(key, _)| *key)
                    .collect::<Vec<_>>()
                    .join(", ");
                anyhow::bail!(
                    "server config {} declares unknown session {} for module {}; expected one of: {}",
                    path.display(),
                    session.key,
                    self.id.as_str(),
                    if expected.is_empty() {
                        "(none)"
                    } else {
                        &expected
                    }
                );
            }
        }
        Ok(())
    }
}

impl ServerActionSpec {
    fn affordance(&self) -> ActionAffordance {
        ActionAffordance {
            id: self.name.clone(),
            label: self.label.clone().unwrap_or_else(|| self.name.clone()),
            description: self.description.clone(),
            use_when: self.use_when.clone().unwrap_or_default(),
            effect: self.effect.clone().unwrap_or_default(),
            input_schema: self.json_schema.0.clone(),
        }
    }
}

fn validate_config_actions(affordances: &[ActionAffordance], path: &Path) -> anyhow::Result<()> {
    let mut seen = HashSet::new();
    for affordance in affordances {
        affordance.validate().map_err(|error| {
            anyhow::anyhow!(
                "server config {} has invalid action {}: {error}",
                path.display(),
                affordance.id
            )
        })?;
        if !seen.insert(affordance.id.as_str()) {
            anyhow::bail!(
                "server config {} declares action {} more than once",
                path.display(),
                affordance.id
            );
        }
    }
    Ok(())
}

pub fn load_server_boot_config(state_dir: &Path) -> anyhow::Result<ServerBootConfig> {
    let path = state_dir.join(SERVER_BOOT_CONFIG_FILE);
    match fs::read_to_string(&path) {
        Ok(content) => parse_server_boot_config_content(&content, &path),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(ServerBootConfig::default()),
        Err(error) => Err(anyhow::anyhow!(
            "failed to read server config {}: {error}",
            path.display()
        )),
    }
}

pub(crate) fn parse_server_boot_config_content(
    content: &str,
    path: &Path,
) -> anyhow::Result<ServerBootConfig> {
    let config: ServerBootConfig =
        eure::parse_content(content, path.to_path_buf()).map_err(|message| {
            anyhow::anyhow!(
                "failed to parse server config {}: {message}",
                path.display()
            )
        })?;
    config.validate(path)?;
    Ok(config)
}

fn validate_activation_table(values: &[f64], path: &Path) -> anyhow::Result<()> {
    for value in values {
        if value.is_finite() && (0.0..=1.0).contains(value) {
            continue;
        }
        anyhow::bail!(
            "server config {} has invalid activation-table value: {value}",
            path.display()
        );
    }
    Ok(())
}

fn validate_finite_ratio(
    value: f64,
    field: &str,
    module: RuntimeModule,
    path: &Path,
) -> anyhow::Result<()> {
    if value.is_finite() && (0.0..=1.0).contains(&value) {
        return Ok(());
    }
    anyhow::bail!(
        "server config {} has invalid {field} for {}: {value}",
        path.display(),
        module.as_str()
    )
}

fn default_activation_table_values() -> Vec<f64> {
    vec![1.0, 0.85, 0.7, 0.5, 0.3, 0.0]
}

fn default_replica_capacity() -> u8 {
    ReplicaCapRange::V1_MAX
}

fn default_server_modules() -> Vec<ServerModuleSpec> {
    use RuntimeModule as M;
    use ServerModuleGroup as G;

    vec![
        module_spec(M::Sensory, 1, 1, 3.0, 8.0, 1.0, [], []),
        module_spec(
            M::CognitionGate,
            1,
            1,
            6.0,
            12.0,
            1.0,
            [G::Voluntary, G::SleepSuppressed],
            [
                M::Sensory,
                M::QueryMemory,
                M::Policy,
                M::SelfModel,
                M::Surprise,
            ],
        ),
        module_spec(M::Allocation, 1, 1, 6.0, 6.0, 1.0, [], []),
        module_spec(
            M::Action,
            1,
            1,
            3.0,
            9.0,
            0.0,
            [G::Voluntary, G::SleepSuppressed],
            [
                M::QueryMemory,
                M::Interpreter,
                M::SelfModel,
                M::Surprise,
                M::CognitionGate,
            ],
        ),
        module_spec(
            M::AttentionSchema,
            0,
            1,
            3.0,
            6.0,
            0.0,
            [G::Voluntary, G::SleepSuppressed],
            [],
        ),
        module_spec(
            M::Interpreter,
            0,
            1,
            3.0,
            6.0,
            0.0,
            [G::Voluntary, G::SleepSuppressed],
            [],
        ),
        module_spec(
            M::SelfModel,
            0,
            1,
            3.0,
            6.0,
            0.0,
            [G::Voluntary, G::SleepSuppressed],
            [M::QueryMemory],
        ),
        module_spec(
            M::QueryMemory,
            1,
            1,
            12.0,
            30.0,
            0.0,
            [G::Voluntary, G::SleepSuppressed],
            [],
        ),
        module_spec(
            M::Memory,
            1,
            1,
            6.0,
            18.0,
            0.0,
            [G::Voluntary, G::SleepSuppressed],
            [],
        ),
        module_spec(
            M::MemoryCompaction,
            0,
            1,
            2.0,
            6.0,
            0.0,
            [G::HomeostaticDrive],
            [M::MemoryAssociation, M::Homeostasis],
        ),
        module_spec(
            M::MemoryAssociation,
            0,
            1,
            2.0,
            6.0,
            0.0,
            [G::HomeostaticDrive],
            [M::Homeostasis],
        ),
        module_spec(
            M::Dreaming,
            0,
            1,
            2.0,
            6.0,
            0.0,
            [G::Voluntary, G::HomeostaticDrive],
            [M::MemoryCompaction, M::Homeostasis],
        ),
        module_spec(M::Interoception, 1, 1, 1.0, 3.0, 1.0, [], []),
        module_spec(M::Homeostasis, 1, 1, 6.0, 20.0, 1.0, [], []),
        module_spec(
            M::Policy,
            1,
            1,
            2.0,
            6.0,
            0.0,
            [G::Voluntary, G::SleepSuppressed],
            [],
        ),
        module_spec(
            M::PolicyCompaction,
            0,
            1,
            2.0,
            6.0,
            0.0,
            [G::HomeostaticDrive],
            [M::Reward, M::Homeostasis],
        ),
        module_spec(
            M::Reward,
            1,
            1,
            1.0,
            2.0,
            0.0,
            [G::Voluntary, G::SleepSuppressed],
            [M::Policy],
        ),
        module_spec(
            M::Predict,
            1,
            1,
            1.0,
            6.0,
            0.0,
            [G::Voluntary, G::SleepSuppressed],
            [],
        ),
        module_spec(
            M::Surprise,
            1,
            1,
            1.0,
            3.0,
            0.0,
            [G::Voluntary, G::SleepSuppressed],
            [M::Predict],
        ),
        module_spec(
            M::Speak,
            0,
            1,
            6.0,
            18.0,
            0.0,
            [G::SleepSuppressed, G::ActionTarget],
            [
                M::QueryMemory,
                M::Interpreter,
                M::SelfModel,
                M::Surprise,
                M::CognitionGate,
            ],
        ),
    ]
}

fn module_spec<const G: usize, const D: usize>(
    id: RuntimeModule,
    replica_min: u8,
    replica_max: u8,
    bpm_min: f64,
    bpm_max: f64,
    initial_activation: f64,
    groups: [ServerModuleGroup; G],
    depends_on: [RuntimeModule; D],
) -> ServerModuleSpec {
    ServerModuleSpec {
        id,
        replica_min,
        replica_max,
        replica_capacity: default_replica_capacity(),
        bpm_min,
        bpm_max,
        initial_activation,
        sessions: Vec::new(),
        groups: groups.to_vec(),
        depends_on: depends_on.to_vec(),
        memo_sources: None,
    }
}

pub fn default_run_id() -> String {
    Utc::now().format("%Y%m%dT%H%M%SZ").to_string()
}

pub fn default_server_session_id() -> String {
    format!("server-{}-{}", default_run_id(), Uuid::now_v7())
}

pub fn load_server_config_from_options(options: ServerRunOptions) -> anyhow::Result<ServerConfig> {
    let model_set_path = resolve_model_set_path(&options.state_dir, options.model_set);
    let agent_db_path = resolve_agent_db_path(&options.state_dir, options.agent_db);
    let model_set = parse_model_set_file(&model_set_path)?;
    let backends = resolve_llm_backends(&model_set)?;
    let cheap_backend = backends.cheap;
    let default_backend = backends.default;
    let premium_backend = backends.premium;
    let image_backend = backends.image;
    let embedding_backend = resolve_embedding(&model_set.embedding)?;
    let session_id = resolve_session_id(options.session_id, options.run_id);
    let boot_config = load_server_boot_config(&options.state_dir)?;

    Ok(ServerConfig {
        state_dir: options.state_dir,
        agent_db_path,
        session_id,
        llm_log_root: Some(options.llm_log_root),
        cheap_backend,
        default_backend,
        premium_backend,
        image_backend,
        embedding_backend,
        boot_config,
        disabled_modules: options.disabled_modules,
        participants: options.participants,
        fresh_agent_db: options.fresh_agent_db,
        start_paused: false,
    })
}

pub fn resolve_session_id(session_id: Option<String>, run_id_alias: Option<String>) -> String {
    session_id
        .or(run_id_alias)
        .unwrap_or_else(default_server_session_id)
}

pub fn resolve_model_set_path(state_dir: &Path, model_set: Option<PathBuf>) -> PathBuf {
    model_set.unwrap_or_else(|| state_dir.join(STATE_MODEL_SET_FILE))
}

pub fn resolve_agent_db_path(state_dir: &Path, agent_db: Option<PathBuf>) -> PathBuf {
    agent_db.unwrap_or_else(|| state_dir.join(AGENT_DB_FILE))
}

pub fn resolve_embedding(role: &EmbeddingRole) -> anyhow::Result<EmbeddingBackendConfig> {
    let endpoint = role
        .endpoint()
        .unwrap_or(DEFAULT_OPENAI_EMBEDDING_ENDPOINT)
        .to_string();
    let token = resolve_token_fields(
        "embedding",
        role.token_env.as_deref(),
        role.token.as_deref(),
        None,
    )?;
    Ok(EmbeddingBackendConfig {
        endpoint,
        token,
        model: role.model.clone(),
        dimensions: role.dimensions as usize,
    })
}

pub fn install_lutum_trace_subscriber() -> anyhow::Result<()> {
    static INSTALL_RESULT: OnceLock<Result<(), String>> = OnceLock::new();
    let result = INSTALL_RESULT.get_or_init(|| {
        let subscriber = tracing_subscriber::registry().with(lutum_trace::layer());
        tracing::subscriber::set_global_default(subscriber).map_err(|error| error.to_string())
    });
    result
        .as_ref()
        .map(|_| ())
        .map_err(|message| anyhow::anyhow!("failed to install lutum trace subscriber: {message}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn in_memory_builder_uses_values_without_loading_config_files() {
        let model_set = crate::model_set::parse_model_set_str(
            r#"
models {
  text { token = "local" model = "text" }
}
cheap-model = "text"
default-model = "text"
premium-model = "text"
embedding {
  token = "local"
  model = "embed"
  dimensions = 8
}
"#,
            "local-storage:model-set",
        )
        .unwrap();

        let config = ServerConfig::from_memory(
            model_set,
            [RuntimeModule::Sensory],
            ["person".to_owned()],
            "browser-session",
        )
        .unwrap();

        assert_eq!(config.active_modules(), vec![RuntimeModule::Sensory]);
        assert_eq!(config.participants, vec!["person"]);
        assert_eq!(config.session_id, "browser-session");
        assert_eq!(config.agent_db_path, PathBuf::from("./agent.db"));
        assert_eq!(config.llm_log_root, None);
        assert!(!config.start_paused);
    }

    #[test]
    fn in_memory_builder_can_start_paused() {
        let model_set = crate::model_set::parse_model_set_str(
            r#"
models {
  text { token = "local" model = "text" }
}
cheap-model = "text"
default-model = "text"
premium-model = "text"
embedding {
  token = "local"
  model = "embed"
  dimensions = 8
}
"#,
            "local-storage:model-set",
        )
        .unwrap();

        let config = ServerConfig::builder(model_set)
            .start_paused(true)
            .build()
            .unwrap();

        assert!(config.start_paused);
    }

    #[test]
    fn load_server_boot_config_missing_file_uses_default_modules() {
        let missing = PathBuf::from(format!(".tmp/missing-server-config-{}", Uuid::now_v7()));
        let config = load_server_boot_config(&missing).unwrap();

        assert_eq!(config.active_modules(), DEFAULT_MODULES.to_vec());
        assert_eq!(config.activation_table, default_activation_table_values());
        let speak = config
            .modules
            .iter()
            .find(|module| module.id == RuntimeModule::Speak)
            .expect("default config includes speak");
        assert_eq!(speak.bpm_min, 6.0);
        assert_eq!(speak.bpm_max, 18.0);
        assert_eq!(speak.session_tier("planning"), ModelTier::Premium);
    }

    #[test]
    fn parse_server_boot_config_reads_module_specs() {
        let config = parse_server_boot_config_content(
            r#"
activation-table = [1.0, 0.5]

@ modules[] {
  id = "sensory"
  replica-min = 1
  replica-max = 1
  replica-capacity = 2
  bpm-min = 3.0
  bpm-max = 8.0
  initial-activation = 1.0
}

@ modules[] {
  id = "speak"
  replica-min = 0
  replica-max = 1
  replica-capacity = 2
  bpm-min = 3.0
  bpm-max = 6.0
  initial-activation = 0.0
  groups = ["voluntary", "sleep-suppressed"]
  depends-on = ["cognition-gate"]
  memo-sources = ["sensory"]

  @ sessions[] {
    key = "planning"
    tier = "premium"
  }

}
"#,
            Path::new(".tmp/server/config.eure"),
        )
        .unwrap();

        assert_eq!(config.activation_table, vec![1.0, 0.5]);
        assert_eq!(
            config.active_modules(),
            vec![RuntimeModule::Sensory, RuntimeModule::Speak]
        );
        assert_eq!(
            config.modules[1].session_tier("planning"),
            ModelTier::Premium
        );
        assert_eq!(
            config.modules[1].groups,
            vec![
                ServerModuleGroup::Voluntary,
                ServerModuleGroup::SleepSuppressed
            ]
        );
        assert_eq!(
            config.modules[1].depends_on,
            vec![RuntimeModule::CognitionGate]
        );
        assert_eq!(
            config.modules[1].memo_sources,
            Some(vec![RuntimeModule::Sensory])
        );
        assert_eq!(config.modules[0].memo_sources, None);
    }

    #[test]
    fn parse_server_boot_config_preserves_empty_memo_sources() {
        let config = parse_server_boot_config_content(
            r#"
@ modules[] {
  id = "self-model"
  replica-min = 0
  replica-max = 1
  bpm-min = 3.0
  bpm-max = 6.0
  initial-activation = 0.0
  memo-sources = []
}
"#,
            Path::new(".tmp/server/config.eure"),
        )
        .unwrap();

        assert_eq!(config.modules[0].memo_sources, Some(Vec::new()));
    }

    #[test]
    fn parse_server_boot_config_rejects_unknown_memo_source() {
        let error = parse_server_boot_config_content(
            r#"
@ modules[] {
  id = "self-model"
  replica-min = 0
  replica-max = 1
  bpm-min = 3.0
  bpm-max = 6.0
  initial-activation = 0.0
  memo-sources = ["query-memory"]
}
"#,
            Path::new(".tmp/server/config.eure"),
        )
        .unwrap_err()
        .to_string();

        assert!(
            error.contains("unknown memo source query-memory for module self-model"),
            "{error}"
        );
    }

    #[test]
    fn parse_server_boot_config_reads_action_specs() {
        let config = parse_server_boot_config_content(
            r#"
@ actions[] {
  name = "move"
  description = "Move through the scene."

  json_schema {
    type = "object"
    additionalProperties = false
    required = ["direction"]

    properties {
      direction {
        type = "string"
        description = "Direction to move."
      }
    }
  }
}

@ actions[] {
  name = "poet"
  label = "Poet"
  description = "Record a short poem."
  use-when = "when quiet writing is appropriate"
  effect = "a poem is recorded"

  json_schema {
    type = "object"
    additionalProperties = false
    required = ["poem"]

    properties {
      poem {
        type = "string"
        description = "The poem text."
      }
    }
  }
}
"#,
            Path::new(".tmp/server/config.eure"),
        )
        .unwrap();

        assert_eq!(
            config.action_affordances(),
            vec![
                ActionAffordance {
                    id: "move".to_owned(),
                    label: "move".to_owned(),
                    description: "Move through the scene.".to_owned(),
                    use_when: String::new(),
                    effect: String::new(),
                    input_schema: serde_json::json!({
                        "type": "object",
                        "additionalProperties": false,
                        "required": ["direction"],
                        "properties": {
                            "direction": {
                                "type": "string",
                                "description": "Direction to move."
                            }
                        }
                    }),
                },
                ActionAffordance {
                    id: "poet".to_owned(),
                    label: "Poet".to_owned(),
                    description: "Record a short poem.".to_owned(),
                    use_when: "when quiet writing is appropriate".to_owned(),
                    effect: "a poem is recorded".to_owned(),
                    input_schema: serde_json::json!({
                        "type": "object",
                        "additionalProperties": false,
                        "required": ["poem"],
                        "properties": {
                            "poem": {
                                "type": "string",
                                "description": "The poem text."
                            }
                        }
                    }),
                }
            ]
        );
    }

    #[test]
    fn config_actions_override_base_affordances() {
        let config = parse_server_boot_config_content(
            r#"
@ actions[] {
  name = "poet"
  description = "Config poet."

  json_schema {
    type = "object"
  }
}
"#,
            Path::new(".tmp/server/config.eure"),
        )
        .unwrap();

        let base = vec![ActionAffordance {
            id: "poet".to_owned(),
            label: "Persisted Poet".to_owned(),
            description: "Persisted poet.".to_owned(),
            use_when: "persisted".to_owned(),
            effect: "persisted".to_owned(),
            input_schema: serde_json::json!({"type": "object", "properties": {}}),
        }];

        assert_eq!(
            config.overlay_action_affordances(base),
            vec![ActionAffordance {
                id: "poet".to_owned(),
                label: "poet".to_owned(),
                description: "Config poet.".to_owned(),
                use_when: String::new(),
                effect: String::new(),
                input_schema: serde_json::json!({"type": "object"}),
            }]
        );
    }

    #[test]
    fn parse_server_boot_config_rejects_duplicate_action_names() {
        let error = parse_server_boot_config_content(
            r#"
@ actions[] {
  name = "move"
  description = "Move."
  json_schema {
    type = "object"
  }
}

@ actions[] {
  name = "move"
  description = "Move again."
  json_schema {
    type = "object"
  }
}
"#,
            Path::new(".tmp/server/config.eure"),
        )
        .unwrap_err()
        .to_string();

        assert!(error.contains("action move more than once"), "{error}");
    }

    #[test]
    fn parse_server_boot_config_rejects_reserved_action_names() {
        let error = parse_server_boot_config_content(
            r#"
@ actions[] {
  name = "sleep"
  description = "Sleep."
  json_schema {
    type = "object"
  }
}
"#,
            Path::new(".tmp/server/config.eure"),
        )
        .unwrap_err()
        .to_string();

        assert!(
            error.contains("reserved for a built-in action: sleep"),
            "{error}"
        );
    }

    #[test]
    fn parse_server_boot_config_rejects_non_object_action_schema() {
        let error = parse_server_boot_config_content(
            r#"
@ actions[] {
  name = "move"
  description = "Move."
  json_schema = true
}
"#,
            Path::new(".tmp/server/config.eure"),
        )
        .unwrap_err()
        .to_string();

        assert!(
            error.contains("action input schema must be a JSON object: move"),
            "{error}"
        );
    }

    #[test]
    fn runtime_module_parses_kebab_case_eure_names() {
        let config = parse_server_boot_config_content(
            r#"
@ modules[] {
  id = "query-memory"
  replica-min = 1
  replica-max = 1
  bpm-min = 12.0
  bpm-max = 30.0
  initial-activation = 0.0
}

@ modules[] {
  id = "policy-compaction"
  replica-min = 0
  replica-max = 1
  bpm-min = 2.0
  bpm-max = 6.0
  initial-activation = 0.0
}
"#,
            Path::new(".tmp/server/config.eure"),
        )
        .unwrap();

        assert_eq!(
            config.active_modules(),
            vec![RuntimeModule::QueryMemory, RuntimeModule::PolicyCompaction]
        );
    }

    #[test]
    fn parse_server_boot_config_rejects_duplicate_modules() {
        let error = parse_server_boot_config_content(
            r#"
@ modules[] {
  id = "sensory"
  replica-min = 1
  replica-max = 1
  bpm-min = 3.0
  bpm-max = 8.0
  initial-activation = 1.0
}

@ modules[] {
  id = "sensory"
  replica-min = 1
  replica-max = 1
  bpm-min = 3.0
  bpm-max = 8.0
  initial-activation = 1.0
}
"#,
            Path::new(".tmp/server/config.eure"),
        )
        .unwrap_err()
        .to_string();

        assert!(error.contains("sensory more than once"), "{error}");
    }

    #[test]
    fn parse_server_boot_config_rejects_unknown_session_key() {
        let error = parse_server_boot_config_content(
            r#"
@ modules[] {
  id = "speak"
  replica-min = 0
  replica-max = 1
  bpm-min = 3.0
  bpm-max = 6.0
  initial-activation = 0.0

  @ sessions[] {
    key = "draft"
    tier = "premium"
  }
}
"#,
            Path::new(".tmp/server/config.eure"),
        )
        .unwrap_err()
        .to_string();

        assert!(
            error.contains("unknown session draft for module speak"),
            "{error}"
        );
    }

    #[test]
    fn parse_server_boot_config_rejects_duplicate_session_keys() {
        let error = parse_server_boot_config_content(
            r#"
@ modules[] {
  id = "speak"
  replica-min = 0
  replica-max = 1
  bpm-min = 3.0
  bpm-max = 6.0
  initial-activation = 0.0

  @ sessions[] {
    key = "planning"
    tier = "premium"
  }

  @ sessions[] {
    key = "planning"
    tier = "default"
  }
}
"#,
            Path::new(".tmp/server/config.eure"),
        )
        .unwrap_err()
        .to_string();

        assert!(
            error.contains("session planning for module speak more than once"),
            "{error}"
        );
    }

    #[test]
    fn parse_server_boot_config_rejects_invalid_module_parameters() {
        let error = parse_server_boot_config_content(
            r#"
@ modules[] {
  id = "speak"
  replica-min = 1
  replica-max = 0
  bpm-min = 3.0
  bpm-max = 6.0
  initial-activation = 0.0
}
"#,
            Path::new(".tmp/server/config.eure"),
        )
        .unwrap_err()
        .to_string();

        assert!(error.contains("invalid replica range"), "{error}");
    }

    #[test]
    fn subsystem_topology_expands_finite_nested_replicas() {
        let config = parse_server_boot_config_content(
            r#"
@ subsystem-definitions[] {
  id = "finger"
  root: predict
  memory-scope = "local"

  @ modules[] {
    id = "predict"
    replica-min = 1
    replica-max = 1
    bpm-min = 1.0
    bpm-max = 2.0
    initial-activation = 1.0
  }
}

@ subsystem-definitions[] {
  id = "arm"
  root: cognition-gate

  @ modules[] {
    id = "cognition-gate"
    replica-min = 1
    replica-max = 1
    bpm-min = 1.0
    bpm-max = 2.0
    initial-activation = 1.0
  }

  @ subsystems[] {
    subsystem = "finger"
    replicas = 2
  }
}

@ subsystems[] {
  subsystem = "arm"
  replicas = 2
}
"#,
            Path::new(".tmp/server/config.eure"),
        )
        .unwrap();

        assert_eq!(
            config
                .expanded_subsystems()
                .into_iter()
                .map(|expanded| expanded.scope.to_string())
                .collect::<Vec<_>>(),
            vec![
                "/arm[0]",
                "/arm[0]/finger[0]",
                "/arm[0]/finger[1]",
                "/arm[1]",
                "/arm[1]/finger[0]",
                "/arm[1]/finger[1]",
            ]
        );
    }

    #[test]
    fn subsystem_topology_rejects_recursive_reference() {
        let error = parse_server_boot_config_content(
            r#"
@ subsystem-definitions[] {
  id = "alpha"
  root: predict
  @ modules[] {
    id: predict
    replica-min = 1
    replica-max = 1
    bpm-min = 1.0
    bpm-max = 1.0
    initial-activation = 1.0
  }
  @ subsystems[] { subsystem = "beta" replicas = 1 }
}
@ subsystem-definitions[] {
  id = "beta"
  root: predict
  @ modules[] {
    id: predict
    replica-min = 1
    replica-max = 1
    bpm-min = 1.0
    bpm-max = 1.0
    initial-activation = 1.0
  }
  @ subsystems[] { subsystem = "alpha" replicas = 1 }
}
@ subsystems[] { subsystem = "alpha" replicas = 1 }
"#,
            Path::new(".tmp/server/config.eure"),
        )
        .unwrap_err()
        .to_string();

        assert!(error.contains("alpha -> beta -> alpha"), "{error}");
    }

    #[test]
    fn subsystem_gate_can_root_scopes_with_action_and_speak_modules() {
        let config = parse_server_boot_config_content(
            r#"
@ subsystem-definitions[] {
  id: arm
  label: Arm
  root: subsystem-gate
  memory-scope: local

  @ modules[] {
    id: subsystem-gate
    replica-min = 1
    replica-max = 1
    bpm-min = 1.0
    bpm-max = 2.0
    initial-activation = 1.0
  }

  @ modules[] {
    id: action
    replica-min = 0
    replica-max = 1
    bpm-min = 1.0
    bpm-max = 2.0
    initial-activation = 0.0
  }

  @ modules[] {
    id: speak
    replica-min = 0
    replica-max = 1
    bpm-min = 1.0
    bpm-max = 2.0
    initial-activation = 0.0
  }
}

@ subsystems[] {
  subsystem: arm
  replicas = 2
}
"#,
            Path::new(".tmp/server/subsystem-gate-test.eure"),
        )
        .unwrap();

        assert_eq!(config.subsystem_definitions.len(), 1);
        assert_eq!(
            config.subsystem_definitions[0].root_module_id(),
            builtin::subsystem_gate()
        );
        assert!(
            config.subsystem_definitions[0]
                .modules
                .iter()
                .any(|module| module.id == RuntimeModule::Action)
        );
        assert!(
            config.subsystem_definitions[0]
                .modules
                .iter()
                .any(|module| module.id == RuntimeModule::Speak)
        );
        assert_eq!(config.expanded_subsystems().len(), 2);
        assert_eq!(
            config.scope_labels().relative_descendant_label(
                &ScopeId::root(),
                &config.expanded_subsystems()[0].scope,
            ),
            Some("Arm 1".to_owned())
        );
    }

    #[test]
    fn single_subsystem_replica_omits_display_number() {
        let config = parse_server_boot_config_content(
            r#"
@ subsystem-definitions[] {
  id: arm
  label: Arm
  root: predict
  @ modules[] {
    id: predict
    replica-min = 1
    replica-max = 1
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
            Path::new(".tmp/server/config.eure"),
        )
        .unwrap();
        let scope = &config.expanded_subsystems()[0].scope;

        assert_eq!(
            config
                .scope_labels()
                .relative_descendant_label(&ScopeId::root(), scope),
            Some("Arm".to_string())
        );
    }
}
