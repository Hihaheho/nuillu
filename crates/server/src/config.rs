use std::{
    fs, io,
    num::NonZeroUsize,
    path::{Path, PathBuf},
    sync::OnceLock,
};

use chrono::Utc;
use eure::FromEure;
use nuillu_types::{ModuleId, builtin};
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
    pub deactivated_modules: Vec<RuntimeModule>,
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

#[derive(Debug, Clone, Default, FromEure)]
#[eure(crate = ::eure::document, rename_all = "kebab-case")]
pub struct ServerBootConfig {
    #[eure(default)]
    pub deactivate_modules: Vec<RuntimeModule>,
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
        active_server_modules(&self.deactivated_modules)
    }
}

pub fn active_server_modules(deactivated_modules: &[RuntimeModule]) -> Vec<RuntimeModule> {
    DEFAULT_MODULES
        .iter()
        .copied()
        .filter(|module| !deactivated_modules.contains(module))
        .collect()
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
    eure::parse_content(content, path.to_path_buf()).map_err(|message| {
        anyhow::anyhow!(
            "failed to parse server config {}: {message}",
            path.display()
        )
    })
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
    fn load_server_boot_config_missing_file_is_noop() {
        let missing = PathBuf::from(format!(".tmp/missing-server-config-{}", Uuid::now_v7()));
        let config = load_server_boot_config(&missing).unwrap();

        assert_eq!(config.deactivate_modules, Vec::new());
    }

    #[test]
    fn parse_server_boot_config_reads_deactivated_modules() {
        let config = parse_server_boot_config_content(
            r#"deactivate-modules = ["policy", "policy-compaction", "reward"]"#,
            Path::new(".tmp/server/config.eure"),
        )
        .unwrap();

        assert_eq!(
            config.deactivate_modules,
            vec![
                RuntimeModule::Policy,
                RuntimeModule::PolicyCompaction,
                RuntimeModule::Reward,
            ]
        );
    }

    #[test]
    fn parse_server_boot_config_reports_path_on_malformed_file() {
        let error = parse_server_boot_config_content(
            r#"deactivate-modules = ["not-a-module"]"#,
            Path::new(".tmp/server/config.eure"),
        )
        .unwrap_err()
        .to_string();

        assert!(error.contains(".tmp/server/config.eure"), "{error}");
    }

    #[test]
    fn runtime_module_parses_kebab_case_eure_names() {
        let config = parse_server_boot_config_content(
            r#"deactivate-modules = ["query-memory", "policy-compaction"]"#,
            Path::new(".tmp/server/config.eure"),
        )
        .unwrap();

        assert_eq!(
            config.deactivate_modules,
            vec![RuntimeModule::QueryMemory, RuntimeModule::PolicyCompaction]
        );
    }

    #[test]
    fn active_server_modules_excludes_deactivated_modules() {
        let modules = active_server_modules(&[
            RuntimeModule::Policy,
            RuntimeModule::PolicyCompaction,
            RuntimeModule::Reward,
        ]);

        assert!(!modules.contains(&RuntimeModule::Policy));
        assert!(!modules.contains(&RuntimeModule::PolicyCompaction));
        assert!(!modules.contains(&RuntimeModule::Reward));
        assert!(modules.contains(&RuntimeModule::Speak));
    }
}
