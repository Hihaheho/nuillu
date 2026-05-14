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
    pub disabled_modules: Vec<ServerModule>,
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
pub enum ServerModule {
    Sensory,
    CognitionGate,
    AttentionController,
    AttentionSchema,
    SelfModel,
    QueryVector,
    QueryPolicy,
    Memory,
    MemoryCompaction,
    MemoryRecombination,
    Vital,
    HomeostaticController,
    Policy,
    ValueEstimator,
    Reward,
    Predict,
    Surprise,
    SpeakGate,
    Speak,
}

pub const DEFAULT_MODULES: &[ServerModule] = &[
    ServerModule::Sensory,
    ServerModule::CognitionGate,
    ServerModule::AttentionController,
    ServerModule::AttentionSchema,
    ServerModule::SelfModel,
    ServerModule::QueryVector,
    ServerModule::QueryPolicy,
    ServerModule::Memory,
    ServerModule::MemoryCompaction,
    ServerModule::MemoryRecombination,
    ServerModule::Vital,
    ServerModule::HomeostaticController,
    ServerModule::ValueEstimator,
    ServerModule::Reward,
    ServerModule::Policy,
    ServerModule::Predict,
    ServerModule::Surprise,
    ServerModule::SpeakGate,
    ServerModule::Speak,
];

impl ServerModule {
    pub fn module_id(self) -> ModuleId {
        match self {
            Self::Sensory => builtin::sensory(),
            Self::CognitionGate => builtin::cognition_gate(),
            Self::AttentionController => builtin::attention_controller(),
            Self::AttentionSchema => builtin::attention_schema(),
            Self::SelfModel => builtin::self_model(),
            Self::QueryVector => builtin::query_vector(),
            Self::QueryPolicy => builtin::query_policy(),
            Self::Memory => builtin::memory(),
            Self::MemoryCompaction => builtin::memory_compaction(),
            Self::MemoryRecombination => builtin::memory_recombination(),
            Self::Vital => builtin::vital(),
            Self::HomeostaticController => builtin::homeostatic_controller(),
            Self::Policy => builtin::policy(),
            Self::ValueEstimator => builtin::value_estimator(),
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
