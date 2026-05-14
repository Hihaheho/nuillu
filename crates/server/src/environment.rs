use std::sync::Arc;

use anyhow::Context as _;
use async_trait::async_trait;
use lutum::{
    Lutum, ModelName, RawTelemetryConfig, RequestExtensions, SharedPoolBudgetManager,
    SharedPoolBudgetOptions,
};
use lutum_in_memory_adapter::InMemoryCognitionLogRepository;
use lutum_libsql_adapter::{
    EmbeddingProfile, LibsqlMemoryStore, LibsqlMemoryStoreConfig, LibsqlPolicyStore,
    LibsqlPolicyStoreConfig,
};
use lutum_model2vec_adapter::PotionBase8MEmbedder;
use lutum_openai::{OpenAiAdapter, OpenAiReasoningEffort};
use nuillu_blackboard::Blackboard;
use nuillu_memory::{MemoryCapabilities, MemoryStore};
use nuillu_module::ports::{Clock, Embedder, PortError, SystemClock};
use nuillu_module::{
    CapabilityProviderConfig, CapabilityProviderPorts, CapabilityProviderRuntime,
    CapabilityProviders, LutumTiers, RuntimeEvent, RuntimeEventSink, RuntimePolicy,
};
use nuillu_openai_embedding_adapter::{OpenAiEmbedder, OpenAiEmbedderConfig};
use nuillu_query_agentic::{FileSearchProvider, NoopFileSearchProvider};
use nuillu_reward::{PolicyCapabilities, PolicyStore};
use nuillu_speak::{Utterance, UtteranceDelta, UtteranceSink};
use nuillu_visualizer_protocol::{
    UtteranceDeltaView, UtteranceView, VisualizerEvent, VisualizerTabId,
};

use super::SERVER_TAB_ID;
use super::config::{LlmBackendConfig, ServerConfig};
use super::gui::VisualizerEventSink;
use super::llm_observer::ServerLlmObserver;

pub(super) struct ServerEnvironment {
    pub(super) blackboard: Blackboard,
    pub(super) caps: CapabilityProviders,
    pub(super) memory: Arc<dyn MemoryStore>,
    pub(super) memory_caps: MemoryCapabilities,
    pub(super) policy_caps: PolicyCapabilities,
    pub(super) clock: Arc<dyn Clock>,
    pub(super) file_search: Arc<dyn FileSearchProvider>,
    pub(super) utterance_sink: Arc<dyn UtteranceSink>,
}

pub(super) async fn build_server_environment(
    config: &ServerConfig,
    allocation: nuillu_blackboard::ResourceAllocation,
    visualizer: VisualizerEventSink,
) -> anyhow::Result<ServerEnvironment> {
    let blackboard = Blackboard::with_allocation(allocation);
    let event_sink = Arc::new(ServerRuntimeEventSink::new(
        SERVER_TAB_ID.to_string(),
        visualizer.clone(),
    ));
    let llm_observer = ServerLlmObserver::new(SERVER_TAB_ID.to_string(), visualizer.clone());
    let utterance_sink = Arc::new(ServerUtteranceSink::new(
        SERVER_TAB_ID.to_string(),
        visualizer,
    ));
    let clock: Arc<dyn Clock> = Arc::new(SystemClock);
    let memory: Arc<dyn MemoryStore> = Arc::new(connect_memory_store(config).await?);
    let policy_store: Arc<dyn PolicyStore> = Arc::new(connect_policy_store(config).await?);
    let caps = CapabilityProviders::new(CapabilityProviderConfig {
        ports: CapabilityProviderPorts {
            blackboard: blackboard.clone(),
            cognition_log_port: Arc::new(InMemoryCognitionLogRepository::new()),
            clock: clock.clone(),
            tiers: build_tiers(config, Some(llm_observer))?,
        },
        runtime: CapabilityProviderRuntime {
            event_sink,
            policy: RuntimePolicy {
                memo_retained_per_owner: 256,
                max_concurrent_llm_calls: config.max_concurrent_llm_calls,
                ..RuntimePolicy::default()
            },
        },
    });

    let memory_caps = MemoryCapabilities::new(
        blackboard.clone(),
        clock.clone(),
        memory.clone(),
        Vec::new(),
    );
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
        blackboard,
        caps,
        memory,
        memory_caps,
        policy_caps,
        clock,
        file_search: Arc::new(NoopFileSearchProvider),
        utterance_sink,
    })
}

async fn connect_memory_store(config: &ServerConfig) -> anyhow::Result<LibsqlMemoryStore> {
    let (embedder, profile, dimensions) = build_embedder(config)?;
    LibsqlMemoryStore::connect(
        LibsqlMemoryStoreConfig::local(config.state_dir.join("memory.db"), dimensions)
            .with_active_profile(profile),
        embedder,
    )
    .await
    .context("connect libsql memory store")
}

async fn connect_policy_store(config: &ServerConfig) -> anyhow::Result<LibsqlPolicyStore> {
    let (embedder, profile, dimensions) = build_embedder(config)?;
    LibsqlPolicyStore::connect(
        LibsqlPolicyStoreConfig::local(config.state_dir.join("policy.db"), dimensions)
            .with_active_profile(profile),
        embedder,
    )
    .await
    .context("connect libsql policy store")
}

fn build_embedder(
    config: &ServerConfig,
) -> anyhow::Result<(Box<dyn Embedder>, EmbeddingProfile, usize)> {
    if let Some(embedding) = &config.embedding_backend {
        let embedder = OpenAiEmbedder::new(OpenAiEmbedderConfig {
            base_url: embedding.endpoint.clone(),
            api_key: embedding.token.clone(),
            model: embedding.model.clone(),
            target_dimensions: embedding.dimensions,
            request_timeout: None,
        })?;
        let profile =
            EmbeddingProfile::new(embedding.model.clone(), "openai", embedding.dimensions);
        Ok((Box::new(embedder), profile, embedding.dimensions))
    } else {
        let embedder = PotionBase8MEmbedder::from_local_dir(&config.model_dir)
            .with_context(|| format!("load model2vec model from {}", config.model_dir.display()))?;
        let dimensions = embedder.dimensions();
        Ok((
            Box::new(embedder),
            EmbeddingProfile::new("potion-base-8M", "local", dimensions),
            dimensions,
        ))
    }
}

fn build_tiers(
    config: &ServerConfig,
    llm_observer: Option<ServerLlmObserver>,
) -> anyhow::Result<LutumTiers> {
    Ok(LutumTiers {
        cheap: build_lutum(&config.cheap_backend, llm_observer.clone())?,
        default: build_lutum(&config.default_backend, llm_observer.clone())?,
        premium: build_lutum(&config.premium_backend, llm_observer)?,
    })
}

fn build_lutum(
    config: &LlmBackendConfig,
    llm_observer: Option<ServerLlmObserver>,
) -> anyhow::Result<Lutum> {
    let adapter = OpenAiAdapter::new(config.token.clone())
        .with_base_url(config.endpoint.clone())
        .with_default_model(ModelName::new(&config.model)?)
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
    if let Some(observer) = llm_observer {
        lutum.extend_hooks(observer.hook_set());
    }
    Ok(match config.reasoning_effort {
        Some(reasoning_effort) => {
            lutum.with_extension(ReasoningEffortConfig(reasoning_effort.into()))
        }
        None => lutum,
    })
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

#[derive(Debug)]
struct ServerRuntimeEventSink {
    tab_id: String,
    visualizer: VisualizerEventSink,
}

impl ServerRuntimeEventSink {
    fn new(tab_id: String, visualizer: VisualizerEventSink) -> Self {
        Self { tab_id, visualizer }
    }
}

#[async_trait(?Send)]
impl RuntimeEventSink for ServerRuntimeEventSink {
    async fn on_event(&self, event: RuntimeEvent) -> Result<(), PortError> {
        match &event {
            RuntimeEvent::LlmAccessed {
                call, owner, tier, ..
            } => eprintln!(
                "nuillu-server llm-accessed tab={} call={} owner={} tier={:?}",
                self.tab_id, call, owner, tier
            ),
            RuntimeEvent::MemoUpdated {
                owner, char_count, ..
            } => eprintln!(
                "nuillu-server memo-updated tab={} owner={} chars={}",
                self.tab_id, owner, char_count
            ),
            RuntimeEvent::RateLimitDelayed {
                owner,
                capability,
                delayed_for,
                ..
            } => eprintln!(
                "nuillu-server rate-limit-delayed tab={} owner={} capability={:?} delayed_ms={}",
                self.tab_id,
                owner,
                capability,
                delayed_for.as_millis()
            ),
            RuntimeEvent::ModuleBatchThrottled {
                owner, delayed_for, ..
            } => eprintln!(
                "nuillu-server module-batch-throttled tab={} owner={} delayed_ms={}",
                self.tab_id,
                owner,
                delayed_for.as_millis()
            ),
            RuntimeEvent::ModuleBatchReady {
                owner,
                batch_type,
                batch_debug,
                ..
            } => eprintln!(
                "nuillu-server module-batch-ready tab={} owner={} type={} chars={}",
                self.tab_id,
                owner,
                batch_type,
                batch_debug.chars().count()
            ),
        }
        self.visualizer.send(VisualizerEvent::RuntimeEvent {
            tab_id: VisualizerTabId::new(self.tab_id.clone()),
            event,
        });
        Ok(())
    }
}

#[derive(Debug)]
struct ServerUtteranceSink {
    tab_id: String,
    visualizer: VisualizerEventSink,
}

impl ServerUtteranceSink {
    fn new(tab_id: String, visualizer: VisualizerEventSink) -> Self {
        Self { tab_id, visualizer }
    }
}

#[async_trait(?Send)]
impl UtteranceSink for ServerUtteranceSink {
    async fn on_complete(&self, utterance: Utterance) -> Result<(), PortError> {
        self.visualizer.send(VisualizerEvent::UtteranceCompleted {
            tab_id: VisualizerTabId::new(self.tab_id.clone()),
            utterance: UtteranceView {
                sender: utterance.sender.to_string(),
                target: utterance.target,
                text: utterance.text,
                emitted_at: utterance.emitted_at,
            },
        });
        Ok(())
    }

    async fn on_delta(&self, delta: UtteranceDelta) -> Result<(), PortError> {
        self.visualizer.send(VisualizerEvent::UtteranceDelta {
            tab_id: VisualizerTabId::new(self.tab_id.clone()),
            utterance: UtteranceDeltaView {
                sender: delta.sender.to_string(),
                target: delta.target,
                generation_id: delta.generation_id,
                sequence: delta.sequence,
                delta: delta.delta,
            },
        });
        Ok(())
    }
}
