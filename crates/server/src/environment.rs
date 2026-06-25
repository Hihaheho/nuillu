use std::{fs, path::Path, rc::Rc, sync::Arc};

use anyhow::Context as _;
use async_trait::async_trait;
use chrono::Local;
use lutum::{
    FrequencyPenalty, Lutum, MaxOutputTokens, ModelName, PresencePenalty, RawTelemetryConfig,
    RequestExtensions, Seed, SharedPoolBudgetManager, SharedPoolBudgetOptions, StopSequences,
    Temperature, TopK, TopP,
};
use lutum_libsql_adapter::{
    EmbeddingProfile, LibsqlAgentStore, LibsqlAgentStoreConfig, LibsqlAmbientSensorySnapshotStore,
    LibsqlLlmTranscriptStore, LibsqlOneShotSensoryInputStore, LibsqlUtteranceEventStore,
    NewUtteranceEvent, UtteranceEventKind,
};
use lutum_openai::{FeatureFlags, OpenAiAdapter, OpenAiReasoningEffort};
use nuillu_blackboard::{AllocationLimits, Blackboard};
use nuillu_llm_trace_file::{FileLlmTraceSink, LlmLogContext};
use nuillu_memory::{MemoryCapabilities, MemoryStore};
use nuillu_module::ports::{Clock, Embedder, PortError, SystemClock};
use nuillu_module::{
    CapabilityProviderConfig, CapabilityProviderPorts, CapabilityProviderRuntime,
    CapabilityProviders, LlmConcurrencyPool, LlmTierHandle, LutumTiers, RuntimeEvent,
    RuntimeEventSink, RuntimePolicy, SessionCompactionPolicy,
};
use nuillu_openai_embedding_adapter::{OpenAiEmbedder, OpenAiEmbedderConfig};
use nuillu_reward::{PolicyCapabilities, PolicyStore};
use nuillu_speak::{Utterance, UtteranceAbort, UtteranceDelta, UtteranceSink};
use nuillu_visualizer_protocol::{VisualizerEvent, VisualizerTabId};

use super::SERVER_TAB_ID;
use super::config::{EmbeddingBackendConfig, LlmBackendConfig, LlmGenerationConfig, ServerConfig};
use super::gui::VisualizerEventSink;
use super::llm_db_trace::DbLlmTraceSink;
use super::llm_observer::VisualizerLlmObserver;
use super::memory_seed::seed_memory_from_state_dir;
use super::runtime_event_log::{
    RuntimeEventLogWriter, runtime_event_log_path, runtime_event_message,
};

const AGENT_DB_FILE: &str = "agent.db";

pub(super) struct ServerEnvironment {
    pub(super) server_session_id: String,
    pub(super) blackboard: Blackboard,
    pub(super) caps: CapabilityProviders,
    pub(super) memory: Rc<dyn MemoryStore>,
    pub(super) memory_caps: MemoryCapabilities,
    pub(super) policy_caps: PolicyCapabilities,
    pub(super) clock: Rc<dyn Clock>,
    pub(super) utterance_sink: Rc<dyn UtteranceSink>,
    pub(super) llm_transcript_store: LibsqlLlmTranscriptStore,
    pub(super) one_shot_sensory_input_store: LibsqlOneShotSensoryInputStore,
    pub(super) ambient_sensory_snapshot_store: LibsqlAmbientSensorySnapshotStore,
    pub(super) utterance_event_store: LibsqlUtteranceEventStore,
}

pub(super) async fn build_server_environment(
    config: &ServerConfig,
    allocation: nuillu_blackboard::ResourceAllocation,
    visualizer: VisualizerEventSink,
) -> anyhow::Result<ServerEnvironment> {
    let blackboard = Blackboard::with_allocation(allocation);
    let runtime_event_log_path = runtime_event_log_path(&config.state_dir, &config.session_id);
    let runtime_event_log = RuntimeEventLogWriter::open(
        runtime_event_log_path,
        config.session_id.clone(),
        SERVER_TAB_ID.to_string(),
    )
    .with_context(|| {
        format!(
            "open runtime event log under {}",
            config.state_dir.display()
        )
    })?;
    eprintln!(
        "nuillu-server runtime-event-log path={}",
        runtime_event_log.path().display()
    );
    let clock: Rc<dyn Clock> = Rc::new(SystemClock);
    let visualizer_for_events = visualizer.clone();
    let llm_observer = VisualizerLlmObserver::new(SERVER_TAB_ID.to_string(), visualizer.clone());
    let agent_store = connect_agent_store(config).await?;
    let one_shot_sensory_input_store = agent_store.one_shot_sensory_input_store();
    let ambient_sensory_snapshot_store = agent_store.ambient_sensory_snapshot_store();
    let utterance_event_store = agent_store.utterance_event_store();
    let utterance_sink = Rc::new(ServerUtteranceSink::new(
        SERVER_TAB_ID.to_string(),
        config.session_id.clone(),
        visualizer,
        utterance_event_store.clone(),
        clock.clone(),
    ));
    let memory: Rc<dyn MemoryStore> = Rc::new(agent_store.memory_store());
    let policy_store: Rc<dyn PolicyStore> = Rc::new(agent_store.policy_store());
    let session_store = Rc::new(agent_store.session_store());
    let allocation_store = Rc::new(agent_store.allocation_store());
    let memo_log_repository = Rc::new(agent_store.memo_log_repository());
    let cognition_log_repository = Rc::new(agent_store.cognition_log_repository());
    let llm_transcript_store = agent_store.llm_transcript_store();
    let db_trace_sink =
        DbLlmTraceSink::new(config.session_id.clone(), llm_transcript_store.clone());
    let event_sink = Rc::new(ServerRuntimeEventSink::new(
        SERVER_TAB_ID.to_string(),
        visualizer_for_events,
        runtime_event_log,
        llm_observer.clone(),
        db_trace_sink.clone(),
    ));
    let llm_concurrency_pool = LlmConcurrencyPool::default();
    let caps = CapabilityProviders::new(CapabilityProviderConfig {
        ports: CapabilityProviderPorts {
            blackboard: blackboard.clone(),
            cognition_log_port: cognition_log_repository,
            clock: clock.clone(),
            tiers: build_tiers(
                &config.cheap_backend,
                &config.default_backend,
                &config.premium_backend,
                &llm_concurrency_pool,
                Some(llm_observer),
                Some(server_llm_log_context(config)),
                Some(db_trace_sink),
            )?,
        },
        runtime: CapabilityProviderRuntime {
            event_sink,
            policy: server_runtime_policy(config),
            session_store,
            allocation_store,
            memo_log_repository,
        },
    });

    let memory_caps = MemoryCapabilities::new(
        blackboard.clone(),
        clock.clone(),
        memory.clone(),
        Vec::new(),
    );
    let seeded_memories = seed_memory_from_state_dir(&config.state_dir, &memory_caps)
        .await
        .context("seed startup memories")?;
    if seeded_memories > 0 {
        eprintln!("nuillu-server seeded memory entries count={seeded_memories}");
    }
    memory_caps
        .bootstrap_identity_memories()
        .await
        .map_err(|error| anyhow::anyhow!("failed to load identity memories: {error}"))?;

    let policy_caps =
        PolicyCapabilities::new(blackboard.clone(), clock.clone(), policy_store, Vec::new());
    policy_caps
        .bootstrap_core_policies()
        .await
        .map_err(|error| anyhow::anyhow!("failed to load core policies: {error}"))?;

    Ok(ServerEnvironment {
        server_session_id: config.session_id.clone(),
        blackboard,
        caps,
        memory,
        memory_caps,
        policy_caps,
        clock,
        utterance_sink,
        llm_transcript_store,
        one_shot_sensory_input_store,
        ambient_sensory_snapshot_store,
        utterance_event_store,
    })
}

fn server_runtime_policy(config: &ServerConfig) -> RuntimePolicy {
    RuntimePolicy {
        allocation_limits: server_allocation_limits(),
        memo_retained_per_owner: 8,
        cognition_log_retained_entries: 16,
        session_compaction: session_compaction_policy(config),
        ..RuntimePolicy::default()
    }
}

fn server_allocation_limits() -> AllocationLimits {
    AllocationLimits::unlimited()
}

fn session_compaction_policy(config: &ServerConfig) -> SessionCompactionPolicy {
    SessionCompactionPolicy::new(
        config.cheap_backend.compaction_input_token_threshold,
        config.default_backend.compaction_input_token_threshold,
        config.premium_backend.compaction_input_token_threshold,
    )
}

async fn connect_agent_store(config: &ServerConfig) -> anyhow::Result<LibsqlAgentStore> {
    let (memory_embedder, memory_profile, memory_dimensions) =
        build_embedder(&config.embedding_backend)?;
    let (policy_embedder, policy_profile, policy_dimensions) =
        build_embedder(&config.embedding_backend)?;
    if config.fresh_agent_db {
        if let Some(path) = backup_agent_db_with_timestamp(
            &config.state_dir,
            &Local::now().format("%Y%m%d%H%M").to_string(),
        )? {
            eprintln!(
                "nuillu-server backed up existing agent db to {}",
                path.display()
            );
        }
    }
    LibsqlAgentStore::connect(
        LibsqlAgentStoreConfig::local(
            config.state_dir.join(AGENT_DB_FILE),
            memory_dimensions,
            policy_dimensions,
        )
        .with_memory_active_profile(memory_profile)
        .with_policy_active_profile(policy_profile),
        memory_embedder,
        policy_embedder,
    )
    .await
    .context("connect libsql agent store")
}

#[derive(Debug, Clone, Copy)]
enum AgentDbBackupFile {
    Main,
    Wal,
    Shm,
}

impl AgentDbBackupFile {
    fn source_name(self) -> &'static str {
        match self {
            Self::Main => AGENT_DB_FILE,
            Self::Wal => "agent.db-wal",
            Self::Shm => "agent.db-shm",
        }
    }

    fn backup_name(self, stem: &str) -> String {
        match self {
            Self::Main => format!("{stem}.db"),
            Self::Wal => format!("{stem}.db-wal"),
            Self::Shm => format!("{stem}.db-shm"),
        }
    }
}

#[derive(Debug)]
struct AgentDbBackupSource {
    file: AgentDbBackupFile,
    path: std::path::PathBuf,
}

fn backup_agent_db_with_timestamp(
    state_dir: &Path,
    timestamp: &str,
) -> anyhow::Result<Option<std::path::PathBuf>> {
    let sources = agent_db_backup_sources(state_dir)
        .into_iter()
        .filter(|source| source.path.exists())
        .collect::<Vec<_>>();
    if sources.is_empty() {
        return Ok(None);
    }

    let suffix = next_agent_db_backup_suffix(state_dir, timestamp)?;
    for source in sources {
        let target = agent_db_backup_path(state_dir, timestamp, suffix, source.file);
        fs::rename(&source.path, &target).with_context(|| {
            format!("back up {} to {}", source.path.display(), target.display())
        })?;
    }

    Ok(Some(agent_db_backup_path(
        state_dir,
        timestamp,
        suffix,
        AgentDbBackupFile::Main,
    )))
}

fn agent_db_backup_sources(state_dir: &Path) -> Vec<AgentDbBackupSource> {
    [
        AgentDbBackupFile::Main,
        AgentDbBackupFile::Wal,
        AgentDbBackupFile::Shm,
    ]
    .into_iter()
    .map(|file| AgentDbBackupSource {
        file,
        path: state_dir.join(file.source_name()),
    })
    .collect()
}

fn next_agent_db_backup_suffix(state_dir: &Path, timestamp: &str) -> anyhow::Result<u32> {
    let mut suffix = 0_u32;
    loop {
        if !agent_db_backup_targets(state_dir, timestamp, suffix)
            .into_iter()
            .any(|path| path.exists())
        {
            return Ok(suffix);
        }
        suffix = suffix.checked_add(1).with_context(|| {
            format!(
                "find unused agent db backup name in {}",
                state_dir.display()
            )
        })?;
    }
}

fn agent_db_backup_targets(
    state_dir: &Path,
    timestamp: &str,
    suffix: u32,
) -> Vec<std::path::PathBuf> {
    [
        AgentDbBackupFile::Main,
        AgentDbBackupFile::Wal,
        AgentDbBackupFile::Shm,
    ]
    .into_iter()
    .map(|file| agent_db_backup_path(state_dir, timestamp, suffix, file))
    .collect()
}

fn agent_db_backup_path(
    state_dir: &Path,
    timestamp: &str,
    suffix: u32,
    file: AgentDbBackupFile,
) -> std::path::PathBuf {
    let stem = if suffix == 0 {
        format!("agent-bk{timestamp}")
    } else {
        format!("agent-bk{timestamp}-{suffix}")
    };
    state_dir.join(file.backup_name(&stem))
}

pub fn build_embedder(
    embedding: &EmbeddingBackendConfig,
) -> anyhow::Result<(Box<dyn Embedder>, EmbeddingProfile, usize)> {
    let embedder = OpenAiEmbedder::new(OpenAiEmbedderConfig {
        base_url: embedding.endpoint.clone(),
        api_key: embedding.token.clone(),
        model: embedding.model.clone(),
        target_dimensions: embedding.dimensions,
        request_timeout: None,
    })?;
    let profile = EmbeddingProfile::new(embedding.model.clone(), "openai", embedding.dimensions);
    Ok((Box::new(embedder), profile, embedding.dimensions))
}

pub fn build_tiers(
    cheap: &LlmBackendConfig,
    default: &LlmBackendConfig,
    premium: &LlmBackendConfig,
    pool: &LlmConcurrencyPool,
    llm_observer: Option<VisualizerLlmObserver>,
    llm_log_context: Option<LlmLogContext>,
    db_trace_sink: Option<DbLlmTraceSink>,
) -> anyhow::Result<LutumTiers> {
    let file_trace_sink = llm_log_context.map(FileLlmTraceSink::new);
    Ok(LutumTiers {
        cheap: build_tier_handle(
            cheap,
            pool,
            llm_observer.clone(),
            file_trace_sink.clone(),
            db_trace_sink.clone(),
        )?,
        default: build_tier_handle(
            default,
            pool,
            llm_observer.clone(),
            file_trace_sink.clone(),
            db_trace_sink.clone(),
        )?,
        premium: build_tier_handle(premium, pool, llm_observer, file_trace_sink, db_trace_sink)?,
    })
}

fn build_tier_handle(
    config: &LlmBackendConfig,
    pool: &LlmConcurrencyPool,
    llm_observer: Option<VisualizerLlmObserver>,
    file_trace_sink: Option<FileLlmTraceSink>,
    db_trace_sink: Option<DbLlmTraceSink>,
) -> anyhow::Result<LlmTierHandle> {
    let lutum = build_lutum_with_file_trace(config, llm_observer, file_trace_sink, db_trace_sink)?;
    let concurrency = pool.limiter_for(&config.model_key, config.max_concurrent_llm_calls);
    Ok(LlmTierHandle::new(
        lutum,
        concurrency,
        config.model_key.clone(),
        config.reasoning,
    ))
}

pub fn build_model_handle(
    config: &LlmBackendConfig,
    pool: &LlmConcurrencyPool,
    llm_observer: Option<VisualizerLlmObserver>,
    file_trace_sink: Option<FileLlmTraceSink>,
    db_trace_sink: Option<DbLlmTraceSink>,
) -> anyhow::Result<LlmTierHandle> {
    build_tier_handle(config, pool, llm_observer, file_trace_sink, db_trace_sink)
}

pub fn build_lutum(
    config: &LlmBackendConfig,
    llm_observer: Option<VisualizerLlmObserver>,
) -> anyhow::Result<Lutum> {
    build_lutum_with_file_trace(config, llm_observer, None, None)
}

pub fn build_lutum_with_file_trace(
    config: &LlmBackendConfig,
    llm_observer: Option<VisualizerLlmObserver>,
    file_trace_sink: Option<FileLlmTraceSink>,
    db_trace_sink: Option<DbLlmTraceSink>,
) -> anyhow::Result<Lutum> {
    let feature_flags = FeatureFlags::OPENAI
        .with_top_k(config.generation.top_k.is_some() && !config.use_responses_api);
    let adapter = OpenAiAdapter::new(config.token.clone())
        .with_base_url(config.endpoint.clone())
        .with_default_model(ModelName::new(&config.model)?)
        .with_feature_flags(feature_flags)
        .with_resolve_reasoning_effort(ConfiguredReasoningEffort);
    let adapter = if config.use_responses_api {
        adapter
    } else {
        adapter.with_chat_completions()
    };
    let mut lutum = Lutum::new(
        Arc::new(adapter),
        SharedPoolBudgetManager::new(SharedPoolBudgetOptions::default()),
    )
    .with_extension(RawTelemetryConfig::all());
    if let Some(sink) = file_trace_sink {
        lutum.extend_hooks(sink.hook_set());
    }
    if let Some(sink) = db_trace_sink {
        lutum.extend_hooks(sink.hook_set());
    }
    if let Some(observer) = llm_observer {
        lutum.extend_hooks(observer.hook_set());
    }
    let lutum = apply_generation_defaults(lutum, &config.generation)?;
    Ok(match config.reasoning_effort {
        Some(reasoning_effort) => {
            lutum.with_extension(ReasoningEffortConfig(reasoning_effort.into()))
        }
        None => lutum,
    })
}

fn apply_generation_defaults(
    mut lutum: Lutum,
    generation: &LlmGenerationConfig,
) -> anyhow::Result<Lutum> {
    if let Some(value) = generation.temperature {
        lutum = lutum.with_generation_param(
            Temperature::new(value as f32)
                .with_context(|| format!("invalid temperature {value}"))?,
        );
    }
    if let Some(value) = generation.top_p {
        lutum = lutum.with_generation_param(
            TopP::new(value as f32).with_context(|| format!("invalid top-p {value}"))?,
        );
    }
    if let Some(value) = generation.top_k {
        lutum = lutum.with_generation_param(
            TopK::new(value).with_context(|| format!("invalid top-k {value}"))?,
        );
    }
    if let Some(value) = generation.frequency_penalty {
        lutum = lutum.with_generation_param(
            FrequencyPenalty::new(value as f32)
                .with_context(|| format!("invalid frequency-penalty {value}"))?,
        );
    }
    if let Some(value) = generation.presence_penalty {
        lutum = lutum.with_generation_param(
            PresencePenalty::new(value as f32)
                .with_context(|| format!("invalid presence-penalty {value}"))?,
        );
    }
    if let Some(value) = generation.max_output_tokens {
        lutum = lutum.with_generation_param(MaxOutputTokens::new(value));
    }
    if let Some(value) = generation.seed {
        lutum = lutum.with_generation_param(Seed::new(value));
    }
    if let Some(value) = generation.stop_sequences.as_ref() {
        lutum = lutum.with_generation_param(StopSequences::new(value.clone()));
    }
    Ok(lutum)
}

pub fn server_llm_log_context(config: &ServerConfig) -> LlmLogContext {
    LlmLogContext::new(config.llm_log_root.clone(), vec![config.session_id.clone()])
}

#[derive(Clone, Copy)]
struct ReasoningEffortConfig(OpenAiReasoningEffort);

#[lutum::impl_hook(lutum_openai::ResolveReasoningEffort)]
async fn configured_reasoning_effort(
    extensions: &RequestExtensions,
) -> Option<OpenAiReasoningEffort> {
    extensions
        .get::<ReasoningEffortConfig>()
        .map(|value| value.0)
}

struct ServerRuntimeEventSink {
    tab_id: String,
    visualizer: VisualizerEventSink,
    runtime_event_log: RuntimeEventLogWriter,
    llm_observer: VisualizerLlmObserver,
    db_trace_sink: DbLlmTraceSink,
}

impl ServerRuntimeEventSink {
    fn new(
        tab_id: String,
        visualizer: VisualizerEventSink,
        runtime_event_log: RuntimeEventLogWriter,
        llm_observer: VisualizerLlmObserver,
        db_trace_sink: DbLlmTraceSink,
    ) -> Self {
        Self {
            tab_id,
            visualizer,
            runtime_event_log,
            llm_observer,
            db_trace_sink,
        }
    }
}

impl RuntimeEventSink for ServerRuntimeEventSink {
    fn on_event(&self, event: RuntimeEvent) -> Result<(), PortError> {
        if let RuntimeEvent::ModuleActivationAttemptFailed {
            owner,
            activation_attempt,
            message,
            ..
        } = &event
        {
            self.llm_observer.mark_activation_attempt_failed(
                owner,
                *activation_attempt,
                message.clone(),
            );
            let db_trace_sink = self.db_trace_sink.clone();
            let owner = owner.clone();
            let activation_attempt = *activation_attempt;
            let message = message.clone();
            tokio::task::spawn_local(async move {
                db_trace_sink
                    .mark_activation_attempt_failed(owner, activation_attempt, message)
                    .await;
            });
        }
        let message = runtime_event_message(&self.tab_id, &event);
        eprintln!("{message}");
        if let Err(error) = self.runtime_event_log.append(&message, &event) {
            tracing::warn!(
                path = %self.runtime_event_log.path().display(),
                ?error,
                "failed to append runtime event log"
            );
            eprintln!(
                "nuillu-server runtime-event-log-write-failed path={} error={}",
                self.runtime_event_log.path().display(),
                error
            );
        }
        self.visualizer.send(VisualizerEvent::RuntimeEvent {
            tab_id: VisualizerTabId::new(self.tab_id.clone()),
            event,
        });
        Ok(())
    }
}

struct ServerUtteranceSink {
    tab_id: String,
    server_session_id: String,
    visualizer: VisualizerEventSink,
    store: LibsqlUtteranceEventStore,
    clock: Rc<dyn Clock>,
}

impl ServerUtteranceSink {
    fn new(
        tab_id: String,
        server_session_id: String,
        visualizer: VisualizerEventSink,
        store: LibsqlUtteranceEventStore,
        clock: Rc<dyn Clock>,
    ) -> Self {
        Self {
            tab_id,
            server_session_id,
            visualizer,
            store,
            clock,
        }
    }

    async fn append_event(&self, event: NewUtteranceEvent) {
        match self.store.append(event).await {
            Ok(record) => self
                .visualizer
                .send(VisualizerEvent::UtteranceEventAppended {
                    tab_id: VisualizerTabId::new(self.tab_id.clone()),
                    row: super::commands::utterance_event_row_view(record),
                }),
            Err(error) => {
                tracing::warn!(error = ?error, "utterance event persistence failed");
                self.visualizer.send(VisualizerEvent::Log {
                    tab_id: VisualizerTabId::new(self.tab_id.clone()),
                    message: format!("failed to persist utterance event: {error}"),
                });
            }
        }
    }
}

#[async_trait(?Send)]
impl UtteranceSink for ServerUtteranceSink {
    async fn on_complete(&self, utterance: Utterance) -> Result<(), PortError> {
        self.append_event(NewUtteranceEvent {
            server_session_id: self.server_session_id.clone(),
            event_kind: UtteranceEventKind::Completed,
            sender: utterance.sender,
            target: utterance.target,
            generation_id: utterance.generation_id,
            sequence: 0,
            content: utterance.text,
            reason: None,
            occurred_at_ms: utterance.emitted_at.timestamp_millis(),
        })
        .await;
        Ok(())
    }

    async fn on_delta(&self, delta: UtteranceDelta) -> Result<(), PortError> {
        self.append_event(NewUtteranceEvent {
            server_session_id: self.server_session_id.clone(),
            event_kind: UtteranceEventKind::Delta,
            sender: delta.sender,
            target: delta.target,
            generation_id: delta.generation_id,
            sequence: delta.sequence,
            content: delta.delta,
            reason: None,
            occurred_at_ms: self.clock.now().timestamp_millis(),
        })
        .await;
        Ok(())
    }

    async fn on_abort(&self, abort: UtteranceAbort) -> Result<(), PortError> {
        self.append_event(NewUtteranceEvent {
            server_session_id: self.server_session_id.clone(),
            event_kind: UtteranceEventKind::Aborted,
            sender: abort.sender,
            target: abort.target,
            generation_id: abort.generation_id,
            sequence: abort.sequence,
            content: abort.partial_utterance,
            reason: Some(abort.reason),
            occurred_at_ms: abort.aborted_at.timestamp_millis(),
        })
        .await;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::{
        collections::HashMap,
        fs,
        path::PathBuf,
        sync::atomic::{AtomicU64, Ordering},
    };

    use nuillu_blackboard::{ActivationRatio, Bpm, ModulePolicy, linear_ratio_fn};
    use nuillu_types::{ModuleId, ReplicaCapRange, builtin};

    use super::*;
    use crate::config::{DEFAULT_MODULES, ServerBootConfig};
    use crate::registry::full_agent_allocation;

    static NEXT_TEST_DIR: AtomicU64 = AtomicU64::new(0);

    fn test_embedding_backend() -> EmbeddingBackendConfig {
        EmbeddingBackendConfig {
            endpoint: "http://localhost:11434/v1".to_string(),
            token: "local".to_string(),
            model: "embed".to_string(),
            dimensions: 8,
        }
    }

    fn test_backend_config() -> LlmBackendConfig {
        LlmBackendConfig {
            model_key: "model".to_string(),
            endpoint: "http://localhost:11434/v1".to_string(),
            token: "local".to_string(),
            model: "model".to_string(),
            reasoning: false,
            reasoning_effort: None,
            generation: LlmGenerationConfig::default(),
            use_responses_api: false,
            compaction_input_token_threshold: 16_000,
            max_concurrent_llm_calls: None,
        }
    }

    #[test]
    fn server_llm_log_context_uses_session_id_namespace() {
        let config = ServerConfig {
            state_dir: PathBuf::from(".tmp/server"),
            session_id: "session-1".to_string(),
            llm_log_root: PathBuf::from("llm-logs"),
            cheap_backend: test_backend_config(),
            default_backend: test_backend_config(),
            premium_backend: test_backend_config(),
            embedding_backend: test_embedding_backend(),
            boot_config: ServerBootConfig::default(),
            disabled_modules: Vec::new(),
            participants: Vec::new(),
            fresh_agent_db: false,
            visualizer_bin: None,
        };

        let context = server_llm_log_context(&config);

        assert_eq!(context.root, PathBuf::from("llm-logs"));
        assert_eq!(context.namespace, vec!["session-1"]);
    }

    #[test]
    fn build_lutum_attaches_generation_defaults() {
        let mut backend = test_backend_config();
        backend.generation = LlmGenerationConfig {
            temperature: Some(0.3),
            top_p: Some(0.8),
            top_k: Some(40),
            frequency_penalty: Some(0.1),
            presence_penalty: Some(-0.1),
            max_output_tokens: Some(256),
            seed: Some(7),
            stop_sequences: Some(vec!["END".to_string()]),
        };

        let lutum = build_lutum(&backend, None).unwrap();
        let extensions = lutum.default_extensions();

        assert!(matches!(
            extensions.get::<lutum::GenerationSetting<lutum::Temperature>>(),
            Some(lutum::GenerationSetting::Set(value)) if value.get() == 0.3
        ));
        assert!(matches!(
            extensions.get::<lutum::GenerationSetting<lutum::TopP>>(),
            Some(lutum::GenerationSetting::Set(value)) if value.get() == 0.8
        ));
        assert!(matches!(
            extensions.get::<lutum::GenerationSetting<lutum::TopK>>(),
            Some(lutum::GenerationSetting::Set(value)) if value.get() == 40
        ));
        assert!(matches!(
            extensions.get::<lutum::GenerationSetting<lutum::MaxOutputTokens>>(),
            Some(lutum::GenerationSetting::Set(value)) if value.get() == 256
        ));
        assert!(matches!(
            extensions.get::<lutum::GenerationSetting<lutum::Seed>>(),
            Some(lutum::GenerationSetting::Set(value)) if value.get() == 7
        ));
        assert!(matches!(
            extensions.get::<lutum::GenerationSetting<lutum::StopSequences>>(),
            Some(lutum::GenerationSetting::Set(value))
                if value.as_slice().len() == 1 && value.as_slice()[0] == "END"
        ));
    }

    #[test]
    fn server_allocation_limits_keep_fifth_positive_priority_slot_active() {
        let mut allocation = full_agent_allocation(&ServerBootConfig::default());
        let table = allocation.activation_table().to_vec();
        let priority_modules = [
            builtin::attention_schema(),
            builtin::self_model(),
            builtin::query_memory(),
            builtin::memory(),
            builtin::speak(),
        ];

        for (rank, module) in priority_modules.iter().cloned().enumerate() {
            allocation.set_activation(
                module,
                table.get(rank).copied().unwrap_or(ActivationRatio::ZERO),
            );
        }

        let policies = DEFAULT_MODULES
            .iter()
            .map(|module| {
                (
                    module.module_id(),
                    ModulePolicy::new(
                        ReplicaCapRange::new(0, 1).unwrap(),
                        Bpm::range(3.0, 6.0),
                        linear_ratio_fn,
                    ),
                )
            })
            .collect::<HashMap<ModuleId, ModulePolicy>>();
        let effective = allocation
            .derived(&policies)
            .limited(server_allocation_limits());

        assert_eq!(
            effective.activation_for(&builtin::speak()),
            table[4],
            "speak should retain the allocation's positive fifth priority ratio"
        );
        assert_eq!(
            effective.active_replicas(&builtin::speak()),
            1,
            "server should not clip speak solely because the default module set already has many active baseline roles"
        );
    }

    #[test]
    fn backup_agent_db_moves_existing_db_to_timestamped_name() {
        let state_dir = test_state_dir();
        fs::write(state_dir.join("agent.db"), "old db").unwrap();

        let backup =
            backup_agent_db_with_timestamp(&state_dir, "202606200745").expect("backup succeeds");

        let backup_path = state_dir.join("agent-bk202606200745.db");
        assert_eq!(backup, Some(backup_path.clone()));
        assert!(!state_dir.join("agent.db").exists());
        assert_eq!(fs::read_to_string(backup_path).unwrap(), "old db");
    }

    #[test]
    fn backup_agent_db_is_noop_without_db_or_sidecars() {
        let state_dir = test_state_dir();

        let backup =
            backup_agent_db_with_timestamp(&state_dir, "202606200745").expect("backup succeeds");

        assert_eq!(backup, None);
        assert!(!state_dir.join("agent-bk202606200745.db").exists());
    }

    #[test]
    fn backup_agent_db_uses_shared_suffix_when_backup_exists() {
        let state_dir = test_state_dir();
        fs::write(state_dir.join("agent.db"), "old db").unwrap();
        fs::write(state_dir.join("agent-bk202606200745.db"), "previous").unwrap();

        let backup =
            backup_agent_db_with_timestamp(&state_dir, "202606200745").expect("backup succeeds");

        let backup_path = state_dir.join("agent-bk202606200745-1.db");
        assert_eq!(backup, Some(backup_path.clone()));
        assert_eq!(
            fs::read_to_string(state_dir.join("agent-bk202606200745.db")).unwrap(),
            "previous"
        );
        assert_eq!(fs::read_to_string(backup_path).unwrap(), "old db");
    }

    #[test]
    fn backup_agent_db_moves_sidecars_with_same_stem() {
        let state_dir = test_state_dir();
        fs::write(state_dir.join("agent.db"), "main").unwrap();
        fs::write(state_dir.join("agent.db-wal"), "wal").unwrap();
        fs::write(state_dir.join("agent.db-shm"), "shm").unwrap();

        let backup =
            backup_agent_db_with_timestamp(&state_dir, "202606200745").expect("backup succeeds");

        assert_eq!(backup, Some(state_dir.join("agent-bk202606200745.db")));
        assert!(!state_dir.join("agent.db").exists());
        assert!(!state_dir.join("agent.db-wal").exists());
        assert!(!state_dir.join("agent.db-shm").exists());
        assert_eq!(
            fs::read_to_string(state_dir.join("agent-bk202606200745.db")).unwrap(),
            "main"
        );
        assert_eq!(
            fs::read_to_string(state_dir.join("agent-bk202606200745.db-wal")).unwrap(),
            "wal"
        );
        assert_eq!(
            fs::read_to_string(state_dir.join("agent-bk202606200745.db-shm")).unwrap(),
            "shm"
        );
    }

    fn test_state_dir() -> PathBuf {
        let root = std::env::current_dir()
            .unwrap()
            .join(".tmp")
            .join("server-environment-tests")
            .join(format!(
                "{}-{}",
                std::process::id(),
                NEXT_TEST_DIR.fetch_add(1, Ordering::Relaxed)
            ));
        fs::create_dir_all(&root).unwrap();
        root
    }
}
