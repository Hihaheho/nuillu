pub mod commands;
pub mod config;
pub mod environment;
pub mod gui;
pub mod llm_observer;
pub mod model_set;
pub mod registry;
pub mod runtime;
pub mod snapshot;
pub mod state;

pub const SERVER_TAB_ID: &str = "server";

pub use config::{
    DEFAULT_MODULES, EmbeddingBackendConfig, LlmBackendConfig, RuntimeModule, ServerConfig,
    default_run_id, default_server_session_id, install_lutum_trace_subscriber,
};
pub use environment::{build_embedder, build_lutum, build_tiers, server_llm_log_context};
pub use gui::{
    VisualizerEventSink, VisualizerHook, accept_visualizer_connection, drain_child_stdio,
    spawn_visualizer_gui, wait_for_visualizer_exit_with_context,
};
pub use llm_observer::VisualizerLlmObserver;
pub use model_set::{
    EmbeddingRole, ModelSet, ModelSetError, ModelSetFile, ModelSetRole, ReasoningEffort,
    parse_model_set_file,
};
pub use nuillu_llm_trace_file::{FileLlmTraceSink, LlmLogContext};
pub use runtime::run_server_with_visualizer;
pub use snapshot::{
    bpm_from_cooldown, duration_millis_u64, linked_memory_record_view, list_all_memories,
    memory_metadata_views, memory_rank_name, memory_record_view, model_tier_name,
    module_policy_views, zero_replica_window_view,
};
