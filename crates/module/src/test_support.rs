//! Test fixtures shared by `#[cfg(test)] mod tests` blocks in this crate.

use std::sync::Arc;

use lutum::{Lutum, MockLlmAdapter, SharedPoolBudgetManager, SharedPoolBudgetOptions};
use nuillu_blackboard::Blackboard;
use nuillu_types::{ModuleId, ModuleInstanceId, ReplicaIndex};

use crate::ports::{
    FileSearchProvider, MemoryStore, NoopCognitionLogRepository, NoopFileSearchProvider,
    NoopMemoryStore, NoopUtteranceSink, SystemClock,
};
use crate::{
    CapabilityProviderConfig, CapabilityProviderPorts, CapabilityProviderRuntime,
    CapabilityProviders, LutumTiers, ModuleCapabilityFactory, RuntimePolicy,
};

pub(crate) fn test_caps(blackboard: Blackboard) -> CapabilityProviders {
    test_caps_with_policy(blackboard, RuntimePolicy::default())
}

pub(crate) fn test_caps_with_policy(
    blackboard: Blackboard,
    policy: RuntimePolicy,
) -> CapabilityProviders {
    test_caps_with_stores_and_adapter(
        blackboard,
        Arc::new(NoopMemoryStore),
        Vec::new(),
        Arc::new(NoopFileSearchProvider),
        MockLlmAdapter::new(),
        policy,
    )
}

pub(crate) fn test_caps_with_stores(
    blackboard: Blackboard,
    primary_memory_store: Arc<dyn MemoryStore>,
    memory_replicas: Vec<Arc<dyn MemoryStore>>,
    file_search: Arc<dyn FileSearchProvider>,
) -> CapabilityProviders {
    test_caps_with_stores_and_adapter(
        blackboard,
        primary_memory_store,
        memory_replicas,
        file_search,
        MockLlmAdapter::new(),
        RuntimePolicy::default(),
    )
}

pub(crate) fn test_caps_with_stores_and_adapter(
    blackboard: Blackboard,
    primary_memory_store: Arc<dyn MemoryStore>,
    memory_replicas: Vec<Arc<dyn MemoryStore>>,
    file_search: Arc<dyn FileSearchProvider>,
    adapter: MockLlmAdapter,
    policy: RuntimePolicy,
) -> CapabilityProviders {
    let adapter = Arc::new(adapter);
    let budget = SharedPoolBudgetManager::new(SharedPoolBudgetOptions::default());
    let lutum = Lutum::new(adapter, budget);
    CapabilityProviders::new(CapabilityProviderConfig {
        ports: CapabilityProviderPorts {
            blackboard,
            cognition_log_port: Arc::new(NoopCognitionLogRepository),
            primary_memory_store,
            memory_replicas,
            file_search,
            utterance_sink: Arc::new(NoopUtteranceSink),
            clock: Arc::new(SystemClock),
            tiers: LutumTiers {
                cheap: lutum.clone(),
                default: lutum.clone(),
                premium: lutum,
            },
        },
        runtime: CapabilityProviderRuntime {
            policy,
            ..CapabilityProviderRuntime::default()
        },
    })
}

pub(crate) fn scoped(
    caps: &CapabilityProviders,
    module: ModuleId,
    replica: u8,
) -> ModuleCapabilityFactory {
    caps.scoped(ModuleInstanceId::new(module, ReplicaIndex::new(replica)))
}
