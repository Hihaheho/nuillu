use std::{
    collections::{BTreeMap, HashSet},
    fs, io,
    num::NonZeroUsize,
    path::{Path, PathBuf},
    sync::{Arc, OnceLock},
    time::Duration,
};

use chrono::Utc;
use eure::FromEure;
use eure::document::{
    EureDocument,
    parse::{ParseContext, ParseError, ParseErrorKind},
};
use nuillu_module::{ActionAffordance, ScopeLabels};
#[cfg(test)]
use nuillu_types::builtin;
use nuillu_types::{
    ModelTier, ModuleGroupId, ModuleGroupIdParseError, ModuleId, ModuleIdParseError,
    ReplicaCapRange, ReplicaIndex, ScopeId, SubsystemId, SubsystemInstanceId,
};
use tracing_subscriber::{EnvFilter, Layer as _, layer::SubscriberExt as _};
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
    pub disabled_modules: Vec<ModuleId>,
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
    pub disabled_modules: Vec<ModuleId>,
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

macro_rules! runtime_modules {
    ($($variant:ident => $name:literal),+ $(,)?) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
        pub enum RuntimeModule {
            $($variant,)+
        }

        impl RuntimeModule {
            pub const ALL: &'static [Self] = &[$(Self::$variant,)+];

            pub fn as_str(self) -> &'static str {
                match self {
                    $(Self::$variant => $name,)+
                }
            }

            pub fn from_module_id(id: &ModuleId) -> Option<Self> {
                match id.as_str() {
                    $($name => Some(Self::$variant),)+
                    _ => None,
                }
            }

            pub fn module_id(self) -> ModuleId {
                ModuleId::new(self.as_str()).expect("built-in module id is kebab-case")
            }
        }
    };
}

runtime_modules! {
    Sensory => "sensory",
    CognitionGate => "cognition-gate",
    Allocation => "allocation",
    Action => "action",
    AttentionSchema => "attention-schema",
    Interpreter => "interpreter",
    SelfModel => "self-model",
    QueryMemory => "query-memory",
    Memory => "memory",
    MemoryCompaction => "memory-compaction",
    MemoryAssociation => "memory-association",
    Dreaming => "dreaming",
    Interoception => "interoception",
    Homeostasis => "homeostasis",
    Policy => "policy",
    PolicyCompaction => "policy-compaction",
    Reward => "reward",
    Predict => "predict",
    Surprise => "surprise",
    SubsystemGate => "subsystem-gate",
    SubsystemAllocation => "subsystem-allocation",
    Speak => "speak",
}

/// A module id parsed and validated at the server configuration boundary.
///
/// This adapter keeps the generic `nuillu-types` crate independent of Eure.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ConfiguredModuleId(ModuleId);

impl ConfiguredModuleId {
    pub fn new(value: impl Into<String>) -> Result<Self, ModuleIdParseError> {
        ModuleId::new(value).map(Self)
    }

    pub fn as_module_id(&self) -> &ModuleId {
        &self.0
    }

    pub fn into_module_id(self) -> ModuleId {
        self.0
    }

    pub fn as_str(&self) -> &str {
        self.0.as_str()
    }
}

impl From<RuntimeModule> for ConfiguredModuleId {
    fn from(value: RuntimeModule) -> Self {
        Self(value.module_id())
    }
}

impl From<RuntimeModule> for ModuleId {
    fn from(value: RuntimeModule) -> Self {
        value.module_id()
    }
}

impl From<ConfiguredModuleId> for ModuleId {
    fn from(value: ConfiguredModuleId) -> Self {
        value.into_module_id()
    }
}

impl std::fmt::Display for ConfiguredModuleId {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(formatter)
    }
}

impl PartialEq<RuntimeModule> for ConfiguredModuleId {
    fn eq(&self, other: &RuntimeModule) -> bool {
        self.0 == other.module_id()
    }
}

impl eure::document::parse::FromEure<'_> for ConfiguredModuleId {
    type Error = ParseError;

    fn parse(ctx: &ParseContext<'_>) -> Result<Self, Self::Error> {
        let value: String = ctx.parse()?;
        Self::new(value.clone()).map_err(|error| ParseError {
            node_id: ctx.node_id(),
            kind: ParseErrorKind::InvalidPattern {
                kind: "module-id".to_owned(),
                reason: format!("invalid module id {value:?}: {error}"),
            },
        })
    }
}

/// A module group id parsed and validated at the server configuration boundary.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ConfiguredModuleGroupId(ModuleGroupId);

impl ConfiguredModuleGroupId {
    pub fn new(value: impl Into<String>) -> Result<Self, ModuleGroupIdParseError> {
        ModuleGroupId::new(value).map(Self)
    }

    pub fn as_module_group_id(&self) -> &ModuleGroupId {
        &self.0
    }

    pub fn into_module_group_id(self) -> ModuleGroupId {
        self.0
    }

    pub fn as_str(&self) -> &str {
        self.0.as_str()
    }
}

impl From<ServerModuleGroup> for ConfiguredModuleGroupId {
    fn from(value: ServerModuleGroup) -> Self {
        Self(value.module_group_id())
    }
}

impl From<ConfiguredModuleGroupId> for ModuleGroupId {
    fn from(value: ConfiguredModuleGroupId) -> Self {
        value.into_module_group_id()
    }
}

impl std::fmt::Display for ConfiguredModuleGroupId {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(formatter)
    }
}

impl PartialEq<ServerModuleGroup> for ConfiguredModuleGroupId {
    fn eq(&self, other: &ServerModuleGroup) -> bool {
        self.0 == other.module_group_id()
    }
}

impl eure::document::parse::FromEure<'_> for ConfiguredModuleGroupId {
    type Error = ParseError;

    fn parse(ctx: &ParseContext<'_>) -> Result<Self, Self::Error> {
        let value: String = ctx.parse()?;
        Self::new(value.clone()).map_err(|error| ParseError {
            node_id: ctx.node_id(),
            kind: ParseErrorKind::InvalidPattern {
                kind: "module-group-id".to_owned(),
                reason: format!("invalid module group id {value:?}: {error}"),
            },
        })
    }
}

pub const SERVER_BOOT_CONFIG_FILE: &str = "config.eure";

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
    pub allocation_description: String,
    #[eure(default)]
    pub modules: Vec<ServerModuleSpec>,
    #[eure(default)]
    pub memory_scope: ServerMemoryScope,
    #[eure(default)]
    pub subsystems: Vec<ServerSubsystemRef>,
}

impl ServerSubsystemDef {
    pub fn subsystem_id(&self) -> SubsystemId {
        SubsystemId::new(self.id.clone()).expect("validated subsystem id")
    }
}

#[derive(Debug, Clone, PartialEq, FromEure)]
#[eure(crate = ::eure::document, rename_all = "kebab-case")]
pub struct ServerSubsystemRef {
    pub subsystem: String,
    #[eure(default)]
    pub replicas: Option<u8>,
    #[eure(default)]
    pub replica_min: Option<u8>,
    #[eure(default)]
    pub replica_max: Option<u8>,
    #[eure(default)]
    pub replica_capacity: Option<u8>,
    #[eure(default)]
    pub initial_activation: Option<f64>,
    #[eure(default)]
    pub replica_curve: ServerProjectionCurve,
    #[eure(default)]
    pub replica_threshold: Option<f64>,
    #[eure(default = "default_activation_table_values")]
    pub activation_table: Vec<f64>,
}

impl ServerSubsystemRef {
    pub fn replica_min(&self) -> u8 {
        self.replica_min
            .unwrap_or_else(|| self.replicas.unwrap_or(1))
    }

    pub fn replica_max(&self) -> u8 {
        self.replica_max
            .unwrap_or_else(|| self.replicas.unwrap_or(1))
    }

    pub fn replica_capacity(&self) -> u8 {
        self.replica_capacity.unwrap_or_else(|| self.replica_max())
    }

    pub fn initial_activation(&self) -> f64 {
        self.initial_activation.unwrap_or(1.0)
    }

    pub fn replica_projection(&self) -> ServerProjectionSpec {
        ServerProjectionSpec {
            curve: self.replica_curve,
            threshold: self.replica_threshold,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, FromEure)]
#[eure(crate = ::eure::document, rename_all = "kebab-case")]
pub struct ServerProjectionSpec {
    #[eure(default)]
    pub curve: ServerProjectionCurve,
    #[eure(default)]
    pub threshold: Option<f64>,
}

impl Default for ServerProjectionSpec {
    fn default() -> Self {
        Self {
            curve: ServerProjectionCurve::Linear,
            threshold: None,
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, FromEure)]
#[eure(crate = ::eure::document, rename_all = "kebab-case")]
pub enum ServerProjectionCurve {
    #[default]
    Linear,
    Threshold,
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
    pub mount: &'a ServerSubsystemRef,
}

#[derive(Debug, Clone, PartialEq, FromEure)]
#[eure(crate = ::eure::document, rename_all = "kebab-case")]
pub struct ServerModuleSpec {
    pub id: ConfiguredModuleId,
    pub replica_min: u8,
    pub replica_max: u8,
    #[eure(default = "default_replica_capacity")]
    pub replica_capacity: u8,
    pub bpm_min: f64,
    pub bpm_max: f64,
    pub initial_activation: f64,
    #[eure(default)]
    pub replica_curve: ServerProjectionCurve,
    #[eure(default)]
    pub replica_threshold: Option<f64>,
    #[eure(default)]
    pub rate_curve: ServerProjectionCurve,
    #[eure(default)]
    pub rate_threshold: Option<f64>,
    #[eure(default)]
    pub model_slots: Vec<ServerModelSlotSpec>,
    #[eure(default)]
    pub groups: Vec<ConfiguredModuleGroupId>,
    #[eure(default)]
    pub depends_on: Vec<ConfiguredModuleId>,
    #[eure(default)]
    pub activation_barrier: Option<ServerActivationBarrierSpec>,
    /// Module roles whose memo updates this module observes. Omitting the key
    /// keeps the default of every role; an explicit empty list subscribes to
    /// none. Every listed role must also be registered in this config.
    #[eure(default)]
    pub memo_sources: Option<Vec<ConfiguredModuleId>>,
    /// Human-facing listener label to stable logical target path mappings
    /// available to Speak. The labels are shown to the model; paths are only
    /// written to the broadcast utterance target metadata.
    #[eure(default)]
    pub scope_targets: BTreeMap<String, String>,
}

#[derive(Debug, Clone, PartialEq, FromEure)]
#[eure(crate = ::eure::document, rename_all = "kebab-case")]
pub struct ServerActivationBarrierSpec {
    pub prerequisites: Vec<ConfiguredModuleId>,
    #[eure(default)]
    pub timeout_seconds: Option<f64>,
}

impl ServerActivationBarrierSpec {
    pub fn timeout(&self) -> Option<Duration> {
        self.timeout_seconds.map(|seconds| {
            Duration::try_from_secs_f64(seconds)
                .expect("validated activation-barrier timeout should fit Duration")
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, FromEure)]
#[eure(crate = ::eure::document, rename_all = "kebab-case")]
pub enum ServerModelTier {
    Cheap,
    #[default]
    Default,
    Premium,
    Image,
}

impl From<ServerModelTier> for ModelTier {
    fn from(value: ServerModelTier) -> Self {
        match value {
            ServerModelTier::Cheap => Self::Cheap,
            ServerModelTier::Default => Self::Default,
            ServerModelTier::Premium => Self::Premium,
            ServerModelTier::Image => Self::Image,
        }
    }
}

#[derive(Debug, Clone, PartialEq, FromEure)]
#[eure(crate = ::eure::document, rename_all = "kebab-case")]
pub struct ServerModelSlotSpec {
    pub key: String,
    pub tier: ServerModelTier,
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
    pub fn model_slot_defaults(self) -> &'static [(&'static str, ModelTier)] {
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
            Self::SubsystemAllocation => &[("main", ModelTier::Default)],
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
    pub fn from_memory<I>(
        model_set: ModelSet,
        enabled_modules: I,
        participants: impl IntoIterator<Item = String>,
        session_id: impl Into<String>,
    ) -> anyhow::Result<Self>
    where
        I: IntoIterator,
        I::Item: Into<ModuleId>,
    {
        Self::builder(model_set)
            .enabled_modules(enabled_modules)
            .participants(participants)
            .session_id(session_id)
            .build()
    }

    pub fn active_modules(&self) -> Vec<ModuleId> {
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
    enabled_modules: Option<HashSet<ModuleId>>,
    disabled_modules: Vec<ModuleId>,
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

    pub fn enabled_modules<I>(mut self, modules: I) -> Self
    where
        I: IntoIterator,
        I::Item: Into<ModuleId>,
    {
        self.enabled_modules = Some(modules.into_iter().map(Into::into).collect());
        self
    }

    pub fn disabled_modules<I>(mut self, modules: I) -> Self
    where
        I: IntoIterator,
        I::Item: Into<ModuleId>,
    {
        self.disabled_modules = modules.into_iter().map(Into::into).collect();
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
                .retain(|module| enabled.contains(module.id.as_module_id()));
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

    pub fn active_modules(&self) -> Vec<ModuleId> {
        self.modules
            .iter()
            .map(ServerModuleSpec::module_id)
            .collect()
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
            .filter(|module| module.groups.iter().any(|candidate| candidate == &group))
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
        validate_module_set(&self.modules, path, ModuleSetScope::Root)?;
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
            if definition.allocation_description.trim().is_empty() {
                anyhow::bail!(
                    "server config {} has an empty allocation-description for subsystem {}",
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
            validate_module_set(
                &definition.modules,
                path,
                ModuleSetScope::Subsystem(&definition.id),
            )?;
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

#[derive(Debug, Clone, Copy)]
enum ModuleSetScope<'a> {
    Root,
    Subsystem(&'a str),
}

impl std::fmt::Display for ModuleSetScope<'_> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Root => formatter.write_str("root"),
            Self::Subsystem(id) => write!(formatter, "subsystem {id}"),
        }
    }
}

fn validate_module_set(
    modules: &[ServerModuleSpec],
    path: &Path,
    scope: ModuleSetScope<'_>,
) -> anyhow::Result<()> {
    let mut seen = HashSet::new();
    for module in modules {
        if !seen.insert(module.id.clone()) {
            anyhow::bail!(
                "server config {} declares module {} more than once in {}",
                path.display(),
                module.id.as_str(),
                scope
            );
        }
        module.validate(path)?;
    }
    for module in modules {
        for dependency in &module.depends_on {
            if !seen.contains(dependency) {
                anyhow::bail!(
                    "server config {} declares unknown dependency {} for module {} in {}",
                    path.display(),
                    dependency.as_str(),
                    module.id.as_str(),
                    scope
                );
            }
        }
        if let Some(barrier) = &module.activation_barrier {
            for prerequisite in &barrier.prerequisites {
                if !seen.contains(prerequisite) {
                    anyhow::bail!(
                        "server config {} declares unknown activation-barrier prerequisite {} for module {} in {}",
                        path.display(),
                        prerequisite.as_str(),
                        module.id.as_str(),
                        scope
                    );
                }
            }
        }
        if let Some(sources) = &module.memo_sources {
            let mut seen_sources = HashSet::new();
            for source in sources {
                if !seen_sources.insert(source) {
                    anyhow::bail!(
                        "server config {} declares memo source {} for module {} more than once in {}",
                        path.display(),
                        source.as_str(),
                        module.id.as_str(),
                        scope
                    );
                }
                if !seen.contains(source) {
                    anyhow::bail!(
                        "server config {} declares unknown memo source {} for module {} in {}",
                        path.display(),
                        source.as_str(),
                        module.id.as_str(),
                        scope
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
        validate_activation_table(&reference.activation_table, path)?;
        if reference.activation_table.is_empty() {
            anyhow::bail!(
                "server config {} has an empty activation-table for subsystem {} under {}",
                path.display(),
                reference.subsystem,
                owner
            );
        }
        if reference.replica_capacity() == 0 {
            anyhow::bail!(
                "server config {} sets replica-capacity=0 for subsystem {} under {}; subsystem mounts must have persistent capacity",
                path.display(),
                reference.subsystem,
                owner
            );
        }
        let min = reference.replica_min();
        let max = reference.replica_max();
        if min > max {
            anyhow::bail!(
                "server config {} has invalid replica range for subsystem {} under {}: min {min} exceeds max {max}",
                path.display(),
                reference.subsystem,
                owner
            );
        }
        let capacity = reference.replica_capacity();
        if capacity < max {
            anyhow::bail!(
                "server config {} sets replica-capacity={} for subsystem {} under {}, below policy max {}",
                path.display(),
                capacity,
                reference.subsystem,
                owner,
                max
            );
        }
        validate_finite_subsystem_ratio(
            reference.initial_activation(),
            "initial-activation",
            &reference.subsystem,
            path,
        )?;
        reference
            .replica_projection()
            .validate(&format!("subsystem {} replicas", reference.subsystem), path)?;
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
    references: &'a [ServerSubsystemRef],
    definitions: &BTreeMap<&'a str, &'a ServerSubsystemDef>,
    expanded: &mut Vec<ExpandedSubsystem<'a>>,
) {
    for reference in references {
        let definition = definitions[reference.subsystem.as_str()];
        for replica in 0..reference.replica_capacity() {
            let scope = parent.child(SubsystemInstanceId::new(
                definition.subsystem_id(),
                ReplicaIndex::new(replica),
            ));
            expanded.push(ExpandedSubsystem {
                scope: scope.clone(),
                definition,
                mount: reference,
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
        for replica in 0..reference.replica_capacity() {
            let scope = parent.child(SubsystemInstanceId::new(
                definition.subsystem_id(),
                ReplicaIndex::new(replica),
            ));
            let segment: Arc<str> = if reference.replica_capacity() == 1 {
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
        self.id.as_module_id().clone()
    }

    pub fn model_tier(&self, key: &str) -> ModelTier {
        self.model_slots
            .iter()
            .find(|session| session.key == key)
            .map(|session| session.tier.into())
            .or_else(|| {
                RuntimeModule::from_module_id(self.id.as_module_id())
                    .and_then(|module| {
                        module
                            .model_slot_defaults()
                            .iter()
                            .find(|(candidate, _)| *candidate == key)
                            .map(|(_, tier)| *tier)
                    })
            })
            .unwrap_or_else(|| {
                panic!(
                    "unknown model slot {key:?} for module {}; catalog validation should reject this",
                    self.id.as_str()
                )
            })
    }

    pub fn speech_targets(&self) -> impl Iterator<Item = (&str, &str)> {
        self.scope_targets
            .iter()
            .map(|(label, path)| (label.as_str(), path.as_str()))
    }

    pub fn replica_range(&self) -> ReplicaCapRange {
        ReplicaCapRange::new(self.replica_min, self.replica_max)
            .expect("server module spec should be validated before use")
    }

    pub fn replica_projection(&self) -> ServerProjectionSpec {
        ServerProjectionSpec {
            curve: self.replica_curve,
            threshold: self.replica_threshold,
        }
    }

    pub fn rate_projection(&self) -> ServerProjectionSpec {
        ServerProjectionSpec {
            curve: self.rate_curve,
            threshold: self.rate_threshold,
        }
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
        validate_finite_ratio(
            self.initial_activation,
            "initial-activation",
            self.id.as_module_id(),
            path,
        )?;
        self.replica_projection()
            .validate(&format!("module {} replicas", self.id.as_str()), path)?;
        self.rate_projection()
            .validate(&format!("module {} rate", self.id.as_str()), path)?;
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
        let mut seen_slots = HashSet::new();
        for slot in &self.model_slots {
            if !seen_slots.insert(slot.key.as_str()) {
                anyhow::bail!(
                    "server config {} declares model slot {} for module {} more than once",
                    path.display(),
                    slot.key,
                    self.id.as_str()
                );
            }
        }
        if self.id != RuntimeModule::Speak && !self.scope_targets.is_empty() {
            anyhow::bail!(
                "server config {} declares scope-targets for module {}; only speak accepts them",
                path.display(),
                self.id.as_str()
            );
        }
        if let Some(barrier) = &self.activation_barrier {
            if barrier.prerequisites.is_empty() {
                anyhow::bail!(
                    "server config {} declares an empty activation-barrier for module {}",
                    path.display(),
                    self.id.as_str()
                );
            }
            let mut seen = HashSet::new();
            for prerequisite in &barrier.prerequisites {
                if prerequisite == &self.id {
                    anyhow::bail!(
                        "server config {} declares module {} as its own activation-barrier prerequisite",
                        path.display(),
                        self.id.as_str()
                    );
                }
                if !seen.insert(prerequisite) {
                    anyhow::bail!(
                        "server config {} declares activation-barrier prerequisite {} for module {} more than once",
                        path.display(),
                        prerequisite.as_str(),
                        self.id.as_str()
                    );
                }
            }
            if let Some(timeout) = barrier.timeout_seconds
                && (!timeout.is_finite()
                    || timeout <= 0.0
                    || Duration::try_from_secs_f64(timeout).is_err())
            {
                anyhow::bail!(
                    "server config {} has invalid activation-barrier timeout-seconds for {}: {}",
                    path.display(),
                    self.id.as_str(),
                    timeout
                );
            }
        }
        for (label, logical_path) in &self.scope_targets {
            if label.trim() != label
                || label.is_empty()
                || label.contains(['\n', '\r'])
                || matches!(label.as_str(), "everyone" | "self")
            {
                anyhow::bail!(
                    "server config {} has invalid speech target label {:?} for module {}: labels must be trimmed, non-empty, single-line text and cannot be self or everyone",
                    path.display(),
                    label,
                    self.id.as_str()
                );
            }
            validate_logical_speech_target_path(logical_path, self.id.as_module_id(), path)?;
        }
        Ok(())
    }
}

fn validate_logical_speech_target_path(
    logical_path: &str,
    module: &ModuleId,
    config_path: &Path,
) -> anyhow::Result<()> {
    let valid = logical_path.trim() == logical_path
        && !logical_path.is_empty()
        && !logical_path.contains(['\n', '\r'])
        && logical_path
            .split('/')
            .all(|segment| !segment.is_empty() && SubsystemId::new(segment.to_owned()).is_ok());
    if !valid {
        anyhow::bail!(
            "server config {} has invalid logical speech target path {:?} for module {}: expected slash-separated subsystem-style ids such as arm1/finger1",
            config_path.display(),
            logical_path,
            module.as_str()
        );
    }
    Ok(())
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
    module: &ModuleId,
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

fn validate_finite_subsystem_ratio(
    value: f64,
    field: &str,
    subsystem: &str,
    path: &Path,
) -> anyhow::Result<()> {
    if value.is_finite() && (0.0..=1.0).contains(&value) {
        return Ok(());
    }
    anyhow::bail!(
        "server config {} has invalid {field} for subsystem {subsystem}: {value}",
        path.display()
    )
}

impl ServerProjectionSpec {
    fn validate(&self, target: &str, path: &Path) -> anyhow::Result<()> {
        match self.curve {
            ServerProjectionCurve::Linear if self.threshold.is_some() => anyhow::bail!(
                "server config {} sets threshold for linear projection on {target}",
                path.display()
            ),
            ServerProjectionCurve::Threshold => {
                let Some(threshold) = self.threshold else {
                    anyhow::bail!(
                        "server config {} omits threshold for threshold projection on {target}",
                        path.display()
                    );
                };
                if !threshold.is_finite() || !(0.0..=1.0).contains(&threshold) {
                    anyhow::bail!(
                        "server config {} has invalid threshold for {target}: {threshold}",
                        path.display()
                    );
                }
            }
            ServerProjectionCurve::Linear => {}
        }
        Ok(())
    }
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

#[allow(clippy::too_many_arguments)]
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
        id: id.into(),
        replica_min,
        replica_max,
        replica_capacity: default_replica_capacity(),
        bpm_min,
        bpm_max,
        initial_activation,
        replica_curve: ServerProjectionCurve::Linear,
        replica_threshold: None,
        rate_curve: ServerProjectionCurve::Linear,
        rate_threshold: None,
        model_slots: Vec::new(),
        groups: groups.into_iter().map(Into::into).collect(),
        depends_on: depends_on.into_iter().map(Into::into).collect(),
        activation_barrier: None,
        memo_sources: None,
        scope_targets: BTreeMap::new(),
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
        let stderr_layer = tracing_subscriber::fmt::layer()
            .with_writer(io::stderr)
            .with_filter(
                EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
            );
        let subscriber = tracing_subscriber::registry()
            .with(lutum_trace::layer())
            .with(stderr_layer);
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

        assert_eq!(config.active_modules(), vec![builtin::sensory()]);
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

        assert_eq!(
            config.active_modules(),
            DEFAULT_MODULES
                .iter()
                .copied()
                .map(RuntimeModule::module_id)
                .collect::<Vec<_>>()
        );
        assert_eq!(config.activation_table, default_activation_table_values());
        let speak = config
            .modules
            .iter()
            .find(|module| module.id == RuntimeModule::Speak)
            .expect("default config includes speak");
        assert_eq!(speak.bpm_min, 6.0);
        assert_eq!(speak.bpm_max, 18.0);
        assert_eq!(speak.model_tier("planning"), ModelTier::Premium);
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
  depends-on = ["sensory"]
  memo-sources = ["sensory"]

  @ model-slots[] {
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
            vec![builtin::sensory(), builtin::speak()]
        );
        assert_eq!(config.modules[1].model_tier("planning"), ModelTier::Premium);
        assert_eq!(
            config.modules[1].groups,
            vec![
                ServerModuleGroup::Voluntary,
                ServerModuleGroup::SleepSuppressed
            ]
        );
        assert_eq!(config.modules[1].depends_on, vec![RuntimeModule::Sensory]);
        assert_eq!(
            config.modules[1].memo_sources,
            Some(vec![RuntimeModule::Sensory.into()])
        );
        assert_eq!(config.modules[0].memo_sources, None);
    }

    #[test]
    fn parse_server_boot_config_accepts_host_module_and_open_group_ids() {
        let config = parse_server_boot_config_content(
            r#"
@ modules[] {
  id: code
  replica-min = 1
  replica-max = 1
  replica-capacity = 1
  bpm-min = 6.0
  bpm-max = 18.0
  initial-activation = 0.0
  groups = ["voluntary", "workspace-tools"]

  @ model-slots[] {
    key: conflict
    tier: premium
  }
}
"#,
            Path::new(".tmp/server/config.eure"),
        )
        .unwrap();

        assert_eq!(config.modules[0].id.as_str(), "code");
        assert_eq!(
            config.modules[0]
                .groups
                .iter()
                .map(ConfiguredModuleGroupId::as_str)
                .collect::<Vec<_>>(),
            vec!["voluntary", "workspace-tools"]
        );
        assert_eq!(config.modules[0].model_slots[0].key, "conflict");
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
    fn parse_server_boot_config_reads_activation_barrier_with_optional_timeout() {
        let config = parse_server_boot_config_content(
            r#"
@ modules[] {
  id: sensory
  replica-min = 1
  replica-max = 1
  bpm-min = 3.0
  bpm-max = 6.0
  initial-activation = 1.0
}

@ modules[] {
  id: speak
  replica-min = 0
  replica-max = 1
  bpm-min = 3.0
  bpm-max = 6.0
  initial-activation = 0.0

  activation-barrier {
    prerequisites = ["sensory"]
    timeout-seconds = 2.5
  }
}
"#,
            Path::new(".tmp/server/config.eure"),
        )
        .unwrap();

        let barrier = config.modules[1].activation_barrier.as_ref().unwrap();
        assert_eq!(barrier.prerequisites, vec![RuntimeModule::Sensory]);
        assert_eq!(barrier.timeout(), Some(Duration::from_millis(2_500)));
    }

    #[test]
    fn parse_server_boot_config_rejects_invalid_activation_barrier() {
        let error = parse_server_boot_config_content(
            r#"
@ modules[] {
  id: sensory
  replica-min = 1
  replica-max = 1
  bpm-min = 3.0
  bpm-max = 6.0
  initial-activation = 1.0
  activation-barrier {
    prerequisites = ["sensory"]
    timeout-seconds = 0.0
  }
}
"#,
            Path::new(".tmp/server/config.eure"),
        )
        .unwrap_err()
        .to_string();

        assert!(
            error.contains("own activation-barrier prerequisite"),
            "{error}"
        );
    }

    #[test]
    fn activation_barrier_validation_rejects_empty_duplicate_unknown_and_invalid_timeout() {
        let path = Path::new(".tmp/server/config.eure");
        let speak = || {
            default_server_modules()
                .into_iter()
                .find(|module| module.id == RuntimeModule::Speak)
                .unwrap()
        };

        let mut empty = speak();
        empty.activation_barrier = Some(ServerActivationBarrierSpec {
            prerequisites: Vec::new(),
            timeout_seconds: None,
        });
        assert!(
            empty
                .validate(path)
                .unwrap_err()
                .to_string()
                .contains("empty")
        );

        let mut duplicate = speak();
        duplicate.activation_barrier = Some(ServerActivationBarrierSpec {
            prerequisites: vec![RuntimeModule::Sensory.into(), RuntimeModule::Sensory.into()],
            timeout_seconds: None,
        });
        assert!(
            duplicate
                .validate(path)
                .unwrap_err()
                .to_string()
                .contains("more than once")
        );

        let mut invalid_timeout = speak();
        invalid_timeout.activation_barrier = Some(ServerActivationBarrierSpec {
            prerequisites: vec![RuntimeModule::Sensory.into()],
            timeout_seconds: Some(0.0),
        });
        assert!(
            invalid_timeout
                .validate(path)
                .unwrap_err()
                .to_string()
                .contains("timeout-seconds")
        );

        let mut unknown = speak();
        unknown.depends_on.clear();
        unknown.activation_barrier = Some(ServerActivationBarrierSpec {
            prerequisites: vec![RuntimeModule::Sensory.into()],
            timeout_seconds: None,
        });
        assert!(
            validate_module_set(&[unknown], path, ModuleSetScope::Root)
                .unwrap_err()
                .to_string()
                .contains("unknown activation-barrier prerequisite sensory")
        );
    }

    #[test]
    fn parse_speak_scope_targets_preserves_labels_and_logical_paths() {
        let config = parse_server_boot_config_content(
            r#"
@ modules[] {
  id: speak
  replica-min = 0
  replica-max = 1
  bpm-min = 3.0
  bpm-max = 6.0
  initial-activation = 0.0
  scope-targets = {
    "Arm 1 の Finger 1" => "arm1/finger1"
    "Arm 2 の Finger 1" => "arm2/finger1"
  }
}
"#,
            Path::new(".tmp/server/speak-targets.eure"),
        )
        .unwrap();

        assert_eq!(
            config.modules[0].scope_targets,
            BTreeMap::from([
                ("Arm 1 の Finger 1".to_string(), "arm1/finger1".to_string()),
                ("Arm 2 の Finger 1".to_string(), "arm2/finger1".to_string()),
            ])
        );
    }

    #[test]
    fn parse_speak_scope_targets_rejects_runtime_replica_paths() {
        let error = parse_server_boot_config_content(
            r#"
@ modules[] {
  id: speak
  replica-min = 0
  replica-max = 1
  bpm-min = 3.0
  bpm-max = 6.0
  initial-activation = 0.0
  scope-targets = { "Finger" => "arm[0]/finger[1]" }
}
"#,
            Path::new(".tmp/server/speak-targets.eure"),
        )
        .unwrap_err()
        .to_string();

        assert!(
            error.contains("invalid logical speech target path"),
            "{error}"
        );
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
    fn runtime_module_ids_are_complete_unique_and_round_trip() {
        let ids = RuntimeModule::ALL
            .iter()
            .copied()
            .map(RuntimeModule::module_id)
            .collect::<HashSet<_>>();

        assert_eq!(RuntimeModule::ALL.len(), 22);
        assert_eq!(ids.len(), RuntimeModule::ALL.len());
        for module in RuntimeModule::ALL {
            assert_eq!(
                RuntimeModule::from_module_id(&module.module_id()),
                Some(*module)
            );
        }
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
    fn parse_server_boot_config_defers_unknown_model_slot_to_catalog_validation() {
        let config = parse_server_boot_config_content(
            r#"
@ modules[] {
  id = "speak"
  replica-min = 0
  replica-max = 1
  bpm-min = 3.0
  bpm-max = 6.0
  initial-activation = 0.0

  @ model-slots[] {
    key = "draft"
    tier = "premium"
  }
}
"#,
            Path::new(".tmp/server/config.eure"),
        )
        .unwrap();

        assert_eq!(config.modules[0].model_slots[0].key, "draft");
    }

    #[test]
    fn parse_server_boot_config_rejects_duplicate_model_slot_keys() {
        let error = parse_server_boot_config_content(
            r#"
@ modules[] {
  id = "speak"
  replica-min = 0
  replica-max = 1
  bpm-min = 3.0
  bpm-max = 6.0
  initial-activation = 0.0

  @ model-slots[] {
    key = "planning"
    tier = "premium"
  }

  @ model-slots[] {
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
            error.contains("model slot planning for module speak more than once"),
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
  allocation-description = "Test finger subsystem."
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
  allocation-description = "Test arm subsystem."

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
  allocation-description = "Test alpha subsystem."
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
  allocation-description = "Test beta subsystem."
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
    fn subsystem_gate_is_an_optional_ordinary_module_with_action_and_speak_peers() {
        let config = parse_server_boot_config_content(
            r#"
@ subsystem-definitions[] {
  id: arm
  label: Arm
  allocation-description = "Test arm subsystem."
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
  allocation-description = "Test arm subsystem."
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

    #[test]
    fn parses_standard_projection_curves_and_subsystem_capacity() {
        let config = parse_server_boot_config_content(
            r#"
@ modules[] {
  id: predict
  replica-min = 0
  replica-max = 2
  replica-capacity = 2
  bpm-min = 1.0
  bpm-max = 5.0
  initial-activation = 0.5
  replica-curve: threshold
  replica-threshold = 0.3
  rate-curve: linear
}
@ subsystem-definitions[] {
  id: arm
  allocation-description = "Test arm subsystem."
}
@ subsystems[] {
  subsystem: arm
  replica-min = 0
  replica-max = 4
  replica-capacity = 4
  initial-activation = 0.5
  replica-curve: threshold
  replica-threshold = 0.4
  activation-table = [1.0, 0.6, 0.2, 0.0]
}
"#,
            Path::new(".tmp/server/projection-test.eure"),
        )
        .unwrap();

        assert_eq!(
            config.modules[0].replica_curve,
            ServerProjectionCurve::Threshold
        );
        assert_eq!(config.subsystems[0].replica_min(), 0);
        assert_eq!(config.subsystems[0].replica_max(), 4);
        assert_eq!(
            config.subsystems[0].activation_table,
            vec![1.0, 0.6, 0.2, 0.0]
        );
        assert_eq!(
            config.subsystem_definitions[0].allocation_description,
            "Test arm subsystem."
        );
        assert_eq!(config.expanded_subsystems().len(), 4);
    }

    #[test]
    fn rejects_invalid_projection_parameters() {
        let error = parse_server_boot_config_content(
            r#"
@ modules[] {
  id: predict
  replica-min = 0
  replica-max = 1
  bpm-min = 1.0
  bpm-max = 5.0
  initial-activation = 0.5
  replica-curve: threshold
  replica-threshold = 1.2
}
"#,
            Path::new(".tmp/server/invalid-projection-test.eure"),
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("invalid threshold"), "{error}");
    }

    #[test]
    fn rejects_zero_capacity_subsystem_mount() {
        let error = parse_server_boot_config_content(
            r#"
@ subsystem-definitions[] {
  id: arm
  allocation-description = "Test arm subsystem."
}
@ subsystems[] {
  subsystem: arm
  replica-min = 0
  replica-max = 0
  replica-capacity = 0
}
"#,
            Path::new(".tmp/server/zero-capacity-subsystem-test.eure"),
        )
        .unwrap_err()
        .to_string();

        assert!(error.contains("replica-capacity=0"), "{error}");
    }

    #[test]
    fn sibling_subsystem_definitions_mount_distinct_single_replica_scopes() {
        let config = parse_server_boot_config_content(
            r#"
@ subsystem-definitions[] {
  id: left-leg
  allocation-description = "Test left leg subsystem."
  memory-scope: local
}
@ subsystem-definitions[] {
  id: center-leg
  allocation-description = "Test center leg subsystem."
  memory-scope: local
}
@ subsystem-definitions[] {
  id: right-leg
  allocation-description = "Test right leg subsystem."
  memory-scope: global
}
@ subsystems[] {
  subsystem: left-leg
  replica-min = 0
  replica-max = 1
  replica-capacity = 1
}
@ subsystems[] {
  subsystem: center-leg
  replica-min = 0
  replica-max = 1
  replica-capacity = 1
}
@ subsystems[] {
  subsystem: right-leg
  replica-min = 0
  replica-max = 1
  replica-capacity = 1
}
"#,
            Path::new(".tmp/server/sibling-subsystem-scopes-test.eure"),
        )
        .unwrap();

        assert_eq!(
            config
                .expanded_subsystems()
                .iter()
                .map(|expanded| (
                    expanded.scope.to_string(),
                    expanded.definition.id.as_str().to_owned(),
                    expanded.definition.memory_scope,
                ))
                .collect::<Vec<_>>(),
            vec![
                (
                    "/left-leg[0]".to_owned(),
                    "left-leg".to_owned(),
                    ServerMemoryScope::Local
                ),
                (
                    "/center-leg[0]".to_owned(),
                    "center-leg".to_owned(),
                    ServerMemoryScope::Local
                ),
                (
                    "/right-leg[0]".to_owned(),
                    "right-leg".to_owned(),
                    ServerMemoryScope::Global
                ),
            ]
        );
    }
}
