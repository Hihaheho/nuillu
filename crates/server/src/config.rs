use std::{num::NonZeroUsize, path::PathBuf, sync::OnceLock};

use chrono::Utc;
use nuillu_types::{ModuleId, builtin};
use tracing_subscriber::layer::SubscriberExt as _;

use crate::model_set::ReasoningEffort;

#[derive(Debug, Clone)]
pub struct ServerConfig {
    pub state_dir: PathBuf,
    pub run_id: String,
    pub cheap_backend: LlmBackendConfig,
    pub default_backend: LlmBackendConfig,
    pub premium_backend: LlmBackendConfig,
    pub model_dir: PathBuf,
    pub embedding_backend: Option<EmbeddingBackendConfig>,
    pub max_concurrent_llm_calls: Option<NonZeroUsize>,
    pub disabled_modules: Vec<RuntimeModule>,
    pub participants: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct LlmBackendConfig {
    pub endpoint: String,
    pub token: String,
    pub model: String,
    pub reasoning_effort: Option<ReasoningEffort>,
    pub use_responses_api: bool,
    pub compaction_input_token_threshold: u64,
}

#[derive(Debug, Clone)]
pub struct EmbeddingBackendConfig {
    pub endpoint: String,
    pub token: String,
    pub model: String,
    pub dimensions: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, clap::ValueEnum)]
pub enum RuntimeModule {
    Sensory,
    CognitionGate,
    AllocationController,
    AttentionSchema,
    SelfModel,
    QueryMemory,
    Memory,
    MemoryCompaction,
    MemoryAssociation,
    MemoryRecombination,
    Interoception,
    HomeostaticController,
    Policy,
    PolicyCompaction,
    Reward,
    Predict,
    Surprise,
    SpeakGate,
    Speak,
}

pub const DEFAULT_MODULES: &[RuntimeModule] = &[
    RuntimeModule::Sensory,
    RuntimeModule::CognitionGate,
    RuntimeModule::AllocationController,
    RuntimeModule::AttentionSchema,
    RuntimeModule::SelfModel,
    RuntimeModule::QueryMemory,
    RuntimeModule::Memory,
    RuntimeModule::MemoryCompaction,
    RuntimeModule::MemoryAssociation,
    RuntimeModule::MemoryRecombination,
    RuntimeModule::Interoception,
    RuntimeModule::HomeostaticController,
    RuntimeModule::Policy,
    RuntimeModule::PolicyCompaction,
    RuntimeModule::Reward,
    RuntimeModule::Predict,
    RuntimeModule::Surprise,
    RuntimeModule::SpeakGate,
    RuntimeModule::Speak,
];

impl RuntimeModule {
    pub fn module_id(self) -> ModuleId {
        match self {
            Self::Sensory => builtin::sensory(),
            Self::CognitionGate => builtin::cognition_gate(),
            Self::AllocationController => builtin::allocation_controller(),
            Self::AttentionSchema => builtin::attention_schema(),
            Self::SelfModel => builtin::self_model(),
            Self::QueryMemory => builtin::query_memory(),
            Self::Memory => builtin::memory(),
            Self::MemoryCompaction => builtin::memory_compaction(),
            Self::MemoryAssociation => builtin::memory_association(),
            Self::MemoryRecombination => builtin::memory_recombination(),
            Self::Interoception => builtin::interoception(),
            Self::HomeostaticController => builtin::homeostatic_controller(),
            Self::Policy => builtin::policy(),
            Self::PolicyCompaction => builtin::policy_compaction(),
            Self::Reward => builtin::reward(),
            Self::Predict => builtin::predict(),
            Self::Surprise => builtin::surprise(),
            Self::SpeakGate => builtin::speak_gate(),
            Self::Speak => builtin::speak(),
        }
    }
}

pub fn default_run_id() -> String {
    Utc::now().format("%Y%m%dT%H%M%SZ").to_string()
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
