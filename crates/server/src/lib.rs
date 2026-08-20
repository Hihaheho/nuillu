pub mod commands;
pub mod config;
pub mod environment;
pub mod gui;
pub mod history;
pub mod llm_db_trace;
pub mod llm_observer;
mod memory_seed;
pub mod model_set;
pub mod registry;
pub mod runtime;
mod runtime_event_log;
pub mod snapshot;
pub mod state;

pub const SERVER_TAB_ID: &str = "server";

pub use config::{
    DEFAULT_MODULES, EmbeddingBackendConfig, LlmBackendConfig, LlmGenerationConfig, RuntimeModule,
    ServerBootConfig, ServerConfig, ServerConfigBuilder, ServerModuleGroup,
    ServerModuleSessionSpec, ServerModuleSpec, ServerRunOptions, ServerSessionTier, default_run_id,
    default_server_session_id, install_lutum_trace_subscriber, load_server_boot_config,
    load_server_config_from_options,
};
pub use environment::{
    build_embedder, build_lutum, build_model_handle, build_tiers, server_llm_log_context,
};
pub use gui::{VisualizerEventSink, VisualizerHook};
pub use llm_observer::VisualizerLlmObserver;
pub use model_set::{
    EmbeddingRole, ModelDefinition, ModelSet, ModelSetError, ModelSetFile, ReasoningEffort,
    ResolvedLlmBackends, TierBinding, model_concurrency_from_backends, parse_model_set_file,
    parse_model_set_str, resolve_llm_backends, resolve_token_fields,
};
pub use nuillu_llm_trace_file::{FileLlmTraceSink, LlmLogContext};
pub use registry::ServerModuleRegistrar;
pub use runtime::{
    ServerAmbientSensorySnapshotRecord, ServerEvent, ServerExternalActionEventRecord,
    ServerExternalActionEventStatus, ServerLlmCall, ServerLlmCallSource,
    ServerOneShotSensoryInputRecord, ServerRuntimeHandle, ServerRuntimeStatus,
    ServerUtteranceEventKind, ServerUtteranceEventRecord, run_server, run_server_with_visualizer,
    spawn_server_runtime, spawn_server_runtime_with_module_registrars,
};
pub use snapshot::{
    duration_millis_u64, linked_memory_record_view, memory_metadata_views, memory_rank_name,
    memory_record_view, module_policy_views, zero_replica_window_view,
};
