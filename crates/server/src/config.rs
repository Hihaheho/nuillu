use std::{
    collections::HashSet,
    fs, io,
    num::NonZeroUsize,
    path::{Path, PathBuf},
    sync::OnceLock,
};

use chrono::Utc;
use eure::FromEure;
use nuillu_types::{ModelTier, ModuleId, ReplicaCapRange, builtin};
use tracing_subscriber::layer::SubscriberExt as _;
use uuid::Uuid;

use crate::model_set::ReasoningEffort;

#[derive(Debug, Clone)]
pub struct ServerConfig {
    pub state_dir: PathBuf,
    pub session_id: String,
    pub llm_log_root: PathBuf,
    pub cheap_backend: LlmBackendConfig,
    pub default_backend: LlmBackendConfig,
    pub premium_backend: LlmBackendConfig,
    pub model_dir: PathBuf,
    pub embedding_backend: Option<EmbeddingBackendConfig>,
    pub boot_config: ServerBootConfig,
    pub disabled_modules: Vec<RuntimeModule>,
    pub participants: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct LlmBackendConfig {
    pub model_key: String,
    pub endpoint: String,
    pub token: String,
    pub model: String,
    pub reasoning_effort: Option<ReasoningEffort>,
    pub use_responses_api: bool,
    pub compaction_input_token_threshold: u64,
    pub max_concurrent_llm_calls: Option<NonZeroUsize>,
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
    AttentionSchema,
    SelfModel,
    QueryMemory,
    Memory,
    MemoryCompaction,
    MemoryAssociation,
    MemoryRecombination,
    Interoception,
    Homeostasis,
    Policy,
    PolicyCompaction,
    Reward,
    Predict,
    Surprise,
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
    pub tier: ServerModelTier,
    #[eure(default)]
    pub groups: Vec<ServerModuleGroup>,
    #[eure(default)]
    pub depends_on: Vec<RuntimeModule>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, FromEure)]
#[eure(crate = ::eure::document, rename_all = "kebab-case")]
pub enum ServerModelTier {
    Cheap,
    #[default]
    Default,
    Premium,
}

impl From<ServerModelTier> for ModelTier {
    fn from(value: ServerModelTier) -> Self {
        match value {
            ServerModelTier::Cheap => Self::Cheap,
            ServerModelTier::Default => Self::Default,
            ServerModelTier::Premium => Self::Premium,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, FromEure)]
#[eure(crate = ::eure::document, rename_all = "kebab-case")]
pub enum ServerModuleGroup {
    Voluntary,
    SleepSuppressed,
    HomeostaticDrive,
}

pub const DEFAULT_MODULES: &[RuntimeModule] = &[
    RuntimeModule::Sensory,
    RuntimeModule::CognitionGate,
    RuntimeModule::Allocation,
    RuntimeModule::AttentionSchema,
    RuntimeModule::SelfModel,
    RuntimeModule::QueryMemory,
    RuntimeModule::Memory,
    RuntimeModule::MemoryCompaction,
    RuntimeModule::MemoryAssociation,
    RuntimeModule::MemoryRecombination,
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
        }
    }
}

impl RuntimeModule {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Sensory => "sensory",
            Self::CognitionGate => "cognition-gate",
            Self::Allocation => "allocation",
            Self::AttentionSchema => "attention-schema",
            Self::SelfModel => "self-model",
            Self::QueryMemory => "query-memory",
            Self::Memory => "memory",
            Self::MemoryCompaction => "memory-compaction",
            Self::MemoryAssociation => "memory-association",
            Self::MemoryRecombination => "memory-recombination",
            Self::Interoception => "interoception",
            Self::Homeostasis => "homeostasis",
            Self::Policy => "policy",
            Self::PolicyCompaction => "policy-compaction",
            Self::Reward => "reward",
            Self::Predict => "predict",
            Self::Surprise => "surprise",
            Self::Speak => "speak",
        }
    }

    pub fn module_id(self) -> ModuleId {
        match self {
            Self::Sensory => builtin::sensory(),
            Self::CognitionGate => builtin::cognition_gate(),
            Self::Allocation => builtin::allocation(),
            Self::AttentionSchema => builtin::attention_schema(),
            Self::SelfModel => builtin::self_model(),
            Self::QueryMemory => builtin::query_memory(),
            Self::Memory => builtin::memory(),
            Self::MemoryCompaction => builtin::memory_compaction(),
            Self::MemoryAssociation => builtin::memory_association(),
            Self::MemoryRecombination => builtin::memory_recombination(),
            Self::Interoception => builtin::interoception(),
            Self::Homeostasis => builtin::homeostasis(),
            Self::Policy => builtin::policy(),
            Self::PolicyCompaction => builtin::policy_compaction(),
            Self::Reward => builtin::reward(),
            Self::Predict => builtin::predict(),
            Self::Surprise => builtin::surprise(),
            Self::Speak => builtin::speak(),
        }
    }
}

impl ServerConfig {
    pub fn active_modules(&self) -> Vec<RuntimeModule> {
        self.boot_config.active_modules()
    }
}

impl ServerBootConfig {
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
        Ok(())
    }
}

impl ServerModuleSpec {
    pub fn module_id(&self) -> ModuleId {
        self.id.module_id()
    }

    pub fn tier(&self) -> ModelTier {
        self.tier.into()
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
        Ok(())
    }
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

fn parse_server_boot_config_content(
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
    use ServerModelTier as T;
    use ServerModuleGroup as G;

    vec![
        module_spec(M::Sensory, 1, 1, 3.0, 8.0, 1.0, T::Cheap, [], []),
        module_spec(
            M::CognitionGate,
            1,
            1,
            6.0,
            12.0,
            1.0,
            T::Default,
            [G::Voluntary, G::SleepSuppressed],
            [
                M::Sensory,
                M::QueryMemory,
                M::Policy,
                M::SelfModel,
                M::Surprise,
            ],
        ),
        module_spec(M::Allocation, 1, 1, 6.0, 6.0, 1.0, T::Default, [], []),
        module_spec(
            M::AttentionSchema,
            0,
            1,
            3.0,
            6.0,
            0.0,
            T::Default,
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
            T::Default,
            [G::Voluntary, G::SleepSuppressed],
            [M::QueryMemory],
        ),
        module_spec(
            M::QueryMemory,
            1,
            1,
            6.0,
            15.0,
            0.0,
            T::Cheap,
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
            T::Cheap,
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
            T::Cheap,
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
            T::Cheap,
            [G::HomeostaticDrive],
            [M::Homeostasis],
        ),
        module_spec(
            M::MemoryRecombination,
            0,
            1,
            2.0,
            6.0,
            0.0,
            T::Cheap,
            [G::HomeostaticDrive],
            [M::MemoryCompaction, M::Homeostasis],
        ),
        module_spec(M::Interoception, 1, 1, 1.0, 3.0, 1.0, T::Cheap, [], []),
        module_spec(M::Homeostasis, 1, 1, 6.0, 20.0, 1.0, T::Cheap, [], []),
        module_spec(
            M::Policy,
            1,
            1,
            2.0,
            6.0,
            0.0,
            T::Default,
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
            T::Cheap,
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
            T::Default,
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
            T::Cheap,
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
            T::Default,
            [G::Voluntary, G::SleepSuppressed],
            [M::Predict],
        ),
        module_spec(
            M::Speak,
            0,
            1,
            3.0,
            6.0,
            0.0,
            T::Premium,
            [G::Voluntary, G::SleepSuppressed],
            [M::QueryMemory, M::SelfModel, M::Surprise, M::CognitionGate],
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
    tier: ServerModelTier,
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
        tier,
        groups: groups.to_vec(),
        depends_on: depends_on.to_vec(),
    }
}

pub fn default_run_id() -> String {
    Utc::now().format("%Y%m%dT%H%M%SZ").to_string()
}

pub fn default_server_session_id() -> String {
    format!("server-{}-{}", default_run_id(), Uuid::now_v7())
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
    fn load_server_boot_config_missing_file_uses_default_modules() {
        let missing = PathBuf::from(format!(".tmp/missing-server-config-{}", Uuid::now_v7()));
        let config = load_server_boot_config(&missing).unwrap();

        assert_eq!(config.active_modules(), DEFAULT_MODULES.to_vec());
        assert_eq!(config.activation_table, default_activation_table_values());
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
  tier = "cheap"
}

@ modules[] {
  id = "speak"
  replica-min = 0
  replica-max = 1
  replica-capacity = 2
  bpm-min = 3.0
  bpm-max = 6.0
  initial-activation = 0.0
  tier = "premium"
  groups = ["voluntary", "sleep-suppressed"]
  depends-on = ["cognition-gate"]
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
        assert_eq!(config.modules[1].tier(), ModelTier::Premium);
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
    }

    #[test]
    fn runtime_module_parses_kebab_case_eure_names() {
        let config = parse_server_boot_config_content(
            r#"
@ modules[] {
  id = "query-memory"
  replica-min = 1
  replica-max = 1
  bpm-min = 6.0
  bpm-max = 15.0
  initial-activation = 0.0
  tier = "cheap"
}

@ modules[] {
  id = "policy-compaction"
  replica-min = 0
  replica-max = 1
  bpm-min = 2.0
  bpm-max = 6.0
  initial-activation = 0.0
  tier = "cheap"
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
}
