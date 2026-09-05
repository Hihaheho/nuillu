pub mod commands;
pub mod config;
pub mod environment;
pub mod gui;
#[cfg(feature = "libsql")]
pub mod history;
pub mod llm_db_trace;
pub mod llm_observer;
mod memory_seed;
pub mod model_set;
pub mod ports;
pub mod registry;
pub mod runtime;
mod runtime_event_log;
pub mod snapshot;
pub mod state;

pub const SERVER_TAB_ID: &str = "server";

pub use config::{
    ConfiguredModuleGroupId, ConfiguredModuleId, DEFAULT_MODULES, EmbeddingBackendConfig,
    LlmBackendConfig, LlmGenerationConfig, RuntimeModule, ServerActivationBarrierSpec,
    ServerBootConfig, ServerConfig, ServerConfigBuilder, ServerModelSlotSpec, ServerModelTier,
    ServerModuleGroup, ServerModuleSpec, ServerRunOptions, default_run_id,
    default_server_session_id, install_lutum_trace_subscriber, load_server_boot_config,
    load_server_config_from_options, parse_server_boot_config_content,
};
pub use environment::{
    build_embedder, build_embedder_with_api_key, build_in_memory_host_ports, build_lutum,
    build_lutum_with_api_key, build_lutum_with_http_client, build_model_handle, build_tiers,
    server_llm_log_context,
};
pub use gui::{VisualizerEventSink, VisualizerHook, VisualizerServerMessageReceiverExt};
pub use llm_observer::VisualizerLlmObserver;
pub use memory_seed::FileMemorySeedPort;
pub use model_set::{
    EmbeddingRole, ModelDefinition, ModelSet, ModelSetError, ModelSetFile, ReasoningEffort,
    ResolvedLlmBackends, TierBinding, model_concurrency_from_backends, parse_model_set_file,
    parse_model_set_str, resolve_llm_backends, resolve_token_fields,
};
pub use nuillu_llm_trace_file::{FileLlmTraceSink, LlmLogContext};
pub use nuillu_types::ModuleId;
pub use ports::{
    MemorySeedPort, MemorySeedSummary, MemorySeedTarget, NoopMemorySeed, NoopRuntimeEventLog,
    RuntimeEventLogPort, ServerHostPorts, ServerStatePort,
};
pub use registry::{
    FilledServerModuleSlot, ResolvedServerModuleConfig, ServerModelSlotDescriptor,
    ServerModuleConfigError, ServerModuleDescriptor, ServerModuleFactory, ServerModuleFactoryError,
    ServerModuleFactoryFn, ServerModuleSlot, builtin_server_registry, server_initial_allocation,
    server_registry_with_factories,
};
pub use runtime::{
    Server, ServerAmbientSensorySnapshotRecord, ServerEvent, ServerExternalActionEventRecord,
    ServerExternalActionEventStatus, ServerHost, ServerLlmCall, ServerLlmCallSource,
    ServerOneShotSensoryInputRecord, ServerRuntimeHandle, ServerRuntimeStatus,
    ServerUtteranceEventKind, ServerUtteranceEventRecord,
};
pub use runtime_event_log::FileRuntimeEventLog;
pub use snapshot::{
    duration_millis_u64, linked_memory_record_view, memory_metadata_views, memory_rank_name,
    memory_record_view, module_policy_views, zero_replica_window_view,
};
pub use state::{FileServerStatePort, InMemoryServerStatePort};
