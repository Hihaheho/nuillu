use std::path::Path;
use std::rc::Rc;
use std::sync::Arc;

use anyhow::Context as _;
use async_trait::async_trait;
use lutum::{
    Lutum, ModelName, RawTelemetryConfig, RequestExtensions, SharedPoolBudgetManager,
    SharedPoolBudgetOptions,
};
use lutum_in_memory_adapter::InMemoryCognitionLogRepository;
use lutum_libsql_adapter::{EmbeddingProfile, LibsqlAgentStore, LibsqlAgentStoreConfig};
use lutum_model2vec_adapter::PotionBase8MEmbedder;
use lutum_openai::{OpenAiAdapter, OpenAiReasoningEffort};
use nuillu_blackboard::{AllocationLimits, Blackboard};
use nuillu_llm_trace_file::{FileLlmTraceSink, LlmLogContext};
use nuillu_memory::{MemoryCapabilities, MemoryStore};
use nuillu_module::ports::{Clock, Embedder, PortError, SystemClock};
use nuillu_module::{
    CapabilityProviderConfig, CapabilityProviderPorts, CapabilityProviderRuntime,
    CapabilityProviders, LutumTiers, RuntimeEvent, RuntimeEventSink, RuntimePolicy,
    SessionCompactionPolicy,
};
use nuillu_openai_embedding_adapter::{OpenAiEmbedder, OpenAiEmbedderConfig};
use nuillu_reward::{PolicyCapabilities, PolicyStore};
use nuillu_speak::{Utterance, UtteranceDelta, UtteranceSink};
use nuillu_visualizer_protocol::{
    UtteranceDeltaView, UtteranceView, VisualizerEvent, VisualizerTabId,
};

use super::SERVER_TAB_ID;
use super::config::{EmbeddingBackendConfig, LlmBackendConfig, ServerConfig};
use super::gui::VisualizerEventSink;
use super::llm_observer::VisualizerLlmObserver;

pub(super) struct ServerEnvironment {
    pub(super) blackboard: Blackboard,
    pub(super) caps: CapabilityProviders,
    pub(super) memory: Rc<dyn MemoryStore>,
    pub(super) memory_caps: MemoryCapabilities,
    pub(super) policy_caps: PolicyCapabilities,
    pub(super) clock: Rc<dyn Clock>,
    pub(super) utterance_sink: Rc<dyn UtteranceSink>,
}

pub(super) async fn build_server_environment(
    config: &ServerConfig,
    allocation: nuillu_blackboard::ResourceAllocation,
    visualizer: VisualizerEventSink,
) -> anyhow::Result<ServerEnvironment> {
    let blackboard = Blackboard::with_allocation(allocation);
    let event_sink = Rc::new(ServerRuntimeEventSink::new(
        SERVER_TAB_ID.to_string(),
        visualizer.clone(),
    ));
    let llm_observer = VisualizerLlmObserver::new(SERVER_TAB_ID.to_string(), visualizer.clone());
    let utterance_sink = Rc::new(ServerUtteranceSink::new(
        SERVER_TAB_ID.to_string(),
        visualizer,
    ));
    let clock: Rc<dyn Clock> = Rc::new(SystemClock);
    let agent_store = connect_agent_store(config).await?;
    let memory: Rc<dyn MemoryStore> = Rc::new(agent_store.memory_store());
    let policy_store: Rc<dyn PolicyStore> = Rc::new(agent_store.policy_store());
    let caps = CapabilityProviders::new(CapabilityProviderConfig {
        ports: CapabilityProviderPorts {
            blackboard: blackboard.clone(),
            cognition_log_port: Rc::new(InMemoryCognitionLogRepository::new()),
            clock: clock.clone(),
            tiers: build_tiers(
                &config.cheap_backend,
                &config.default_backend,
                &config.premium_backend,
                Some(llm_observer),
                Some(server_llm_log_context(config)),
            )?,
        },
        runtime: CapabilityProviderRuntime {
            event_sink,
            policy: server_runtime_policy(config),
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
        utterance_sink,
    })
}

fn server_runtime_policy(config: &ServerConfig) -> RuntimePolicy {
    RuntimePolicy {
        allocation_limits: server_allocation_limits(),
        memo_retained_per_owner: 8,
        cognition_log_retained_entries: 16,
        max_concurrent_llm_calls: config.max_concurrent_llm_calls,
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
        build_embedder(config.embedding_backend.as_ref(), &config.model_dir)?;
    let (policy_embedder, policy_profile, policy_dimensions) =
        build_embedder(config.embedding_backend.as_ref(), &config.model_dir)?;
    LibsqlAgentStore::connect(
        LibsqlAgentStoreConfig::local(
            config.state_dir.join("agent.db"),
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

pub fn build_embedder(
    embedding_backend: Option<&EmbeddingBackendConfig>,
    model_dir: &Path,
) -> anyhow::Result<(Box<dyn Embedder>, EmbeddingProfile, usize)> {
    if let Some(embedding) = embedding_backend {
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
        let embedder = PotionBase8MEmbedder::from_local_dir(model_dir)
            .with_context(|| format!("load model2vec model from {}", model_dir.display()))?;
        let dimensions = embedder.dimensions();
        Ok((
            Box::new(embedder),
            EmbeddingProfile::new("potion-base-8M", "local", dimensions),
            dimensions,
        ))
    }
}

pub fn build_tiers(
    cheap: &LlmBackendConfig,
    default: &LlmBackendConfig,
    premium: &LlmBackendConfig,
    llm_observer: Option<VisualizerLlmObserver>,
    llm_log_context: Option<LlmLogContext>,
) -> anyhow::Result<LutumTiers> {
    let file_trace_sink = llm_log_context.map(FileLlmTraceSink::new);
    Ok(LutumTiers {
        cheap: build_lutum_with_file_trace(cheap, llm_observer.clone(), file_trace_sink.clone())?,
        default: build_lutum_with_file_trace(
            default,
            llm_observer.clone(),
            file_trace_sink.clone(),
        )?,
        premium: build_lutum_with_file_trace(premium, llm_observer, file_trace_sink)?,
    })
}

pub fn build_lutum(
    config: &LlmBackendConfig,
    llm_observer: Option<VisualizerLlmObserver>,
) -> anyhow::Result<Lutum> {
    build_lutum_with_file_trace(config, llm_observer, None)
}

pub fn build_lutum_with_file_trace(
    config: &LlmBackendConfig,
    llm_observer: Option<VisualizerLlmObserver>,
    file_trace_sink: Option<FileLlmTraceSink>,
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
    if let Some(sink) = file_trace_sink {
        lutum.extend_hooks(sink.hook_set());
    }
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
            RuntimeEvent::ModuleTaskFailed {
                owner,
                phase,
                message,
                ..
            } => eprintln!(
                "nuillu-server module-task-failed tab={} owner={} phase={} error={}",
                self.tab_id, owner, phase, message
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

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::path::PathBuf;

    use nuillu_blackboard::{ActivationRatio, Bpm, ModuleConfig, ModulePolicy, linear_ratio_fn};
    use nuillu_types::{ModuleId, ReplicaCapRange, builtin};

    use super::*;
    use crate::config::DEFAULT_MODULES;
    use crate::registry::full_agent_allocation;

    fn test_backend_config() -> LlmBackendConfig {
        LlmBackendConfig {
            endpoint: "http://localhost:11434/v1".to_string(),
            token: "local".to_string(),
            model: "model".to_string(),
            reasoning_effort: None,
            use_responses_api: false,
            compaction_input_token_threshold: 16_000,
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
            model_dir: PathBuf::from("models/potion-base-8M"),
            embedding_backend: None,
            max_concurrent_llm_calls: None,
            disabled_modules: Vec::new(),
            participants: Vec::new(),
        };

        let context = server_llm_log_context(&config);

        assert_eq!(context.root, PathBuf::from("llm-logs"));
        assert_eq!(context.namespace, vec!["session-1"]);
    }

    #[test]
    fn server_allocation_limits_keep_fifth_positive_priority_slot_active() {
        let mut allocation = full_agent_allocation(DEFAULT_MODULES);
        let table = allocation.activation_table().to_vec();
        let priority_modules = [
            builtin::attention_schema(),
            builtin::self_model(),
            builtin::query_memory(),
            builtin::memory(),
            builtin::speak(),
        ];

        for (rank, module) in priority_modules.iter().cloned().enumerate() {
            allocation.set(
                module.clone(),
                ModuleConfig {
                    guidance: format!("priority {rank}"),
                },
            );
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
            "speak should retain the allocation-controller's positive fifth priority ratio"
        );
        assert_eq!(
            effective.active_replicas(&builtin::speak()),
            1,
            "server should not clip speak solely because the default module set already has many active baseline roles"
        );
    }
}
