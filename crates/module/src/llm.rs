use std::collections::HashMap;
use std::future::Future;
use std::num::NonZeroUsize;
use std::ops::Deref;
use std::sync::{Arc, Mutex, OnceLock};

use lutum::Lutum;
use nuillu_types::{ModelTier, ModuleActivationId, ModuleInstanceId};
use serde::{Deserialize, Serialize};
use tokio::sync::{OwnedSemaphorePermit, Semaphore, TryAcquireError};
use tokio::task::Id as TaskId;

use crate::LutumTiers;
use crate::runtime_events::RuntimeEventEmitter;
use crate::r#trait::ModuleBatch;

const MAX_LLM_BATCH_DEBUG_CHARS: usize = 20_000;

/// Source of a Lutum request issued by a module-scoped runtime handle.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LlmRequestSource {
    ModuleTurn,
    SessionCompaction,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct LlmBatchDebug {
    pub batch_type: String,
    pub batch_debug: String,
}

impl LlmBatchDebug {
    pub fn from_batch(batch: &ModuleBatch) -> Self {
        Self {
            batch_type: batch.type_name().to_string(),
            batch_debug: truncated_batch_debug(batch.debug()),
        }
    }
}

/// Per-request metadata attached to [`lutum::RequestExtensions`].
///
/// The module crate only stamps this metadata. External observers such as eval
/// and visualizer hooks decide how to render or record it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct LlmRequestMetadata {
    pub owner: ModuleInstanceId,
    pub tier: ModelTier,
    pub source: LlmRequestSource,
    #[serde(default)]
    pub session_key: Option<String>,
    pub activation_id: ModuleActivationId,
    pub activation_attempt: u32,
    pub batch: LlmBatchDebug,
}

#[derive(Clone, Debug)]
struct ActivationLlmRequestMetadata {
    activation_id: ModuleActivationId,
    activation_attempt: u32,
    batch: LlmBatchDebug,
}

tokio::task_local! {
    static ACTIVATION_LLM_REQUEST_METADATA: ActivationLlmRequestMetadata;
}

pub async fn with_activation_llm_request_metadata<F, T>(
    activation_id: ModuleActivationId,
    activation_attempt: u32,
    batch: &ModuleBatch,
    future: F,
) -> T
where
    F: Future<Output = T>,
{
    ACTIVATION_LLM_REQUEST_METADATA
        .scope(
            ActivationLlmRequestMetadata {
                activation_id,
                activation_attempt,
                batch: LlmBatchDebug::from_batch(batch),
            },
            future,
        )
        .await
}

pub fn current_activation_llm_request_metadata() -> Option<(ModuleActivationId, u32, LlmBatchDebug)>
{
    ACTIVATION_LLM_REQUEST_METADATA
        .try_with(|metadata| {
            (
                metadata.activation_id,
                metadata.activation_attempt,
                metadata.batch.clone(),
            )
        })
        .ok()
}

type HeldConcurrencyKey = (TaskId, usize);

struct HeldConcurrencyPermit {
    count: usize,
    _permit: OwnedSemaphorePermit,
}

fn held_concurrency_permits() -> &'static Mutex<HashMap<HeldConcurrencyKey, HeldConcurrencyPermit>>
{
    static HELD: OnceLock<Mutex<HashMap<HeldConcurrencyKey, HeldConcurrencyPermit>>> =
        OnceLock::new();
    HELD.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Permit returned by [`LlmConcurrencyLimiter`].
///
/// Permits are reentrant within the current Tokio task. This lets a module
/// finish a model turn, keep the leased `Lutum` value in scope, and then run
/// session compaction through the same model limiter without self-deadlocking.
pub struct LlmConcurrencyPermit {
    held: LlmConcurrencyPermitHeld,
}

enum LlmConcurrencyPermitHeld {
    Task(HeldConcurrencyKey),
    Unscoped {
        _permit: Option<OwnedSemaphorePermit>,
    },
}

impl LlmConcurrencyPermit {
    fn task_scoped(key: HeldConcurrencyKey) -> Self {
        Self {
            held: LlmConcurrencyPermitHeld::Task(key),
        }
    }

    fn unscoped(permit: OwnedSemaphorePermit) -> Self {
        Self {
            held: LlmConcurrencyPermitHeld::Unscoped {
                _permit: Some(permit),
            },
        }
    }
}

impl Drop for LlmConcurrencyPermit {
    fn drop(&mut self) {
        let LlmConcurrencyPermitHeld::Task(key) = &self.held else {
            return;
        };
        let mut held = held_concurrency_permits()
            .lock()
            .expect("LLM concurrency permit map mutex poisoned");
        let Some(entry) = held.get_mut(key) else {
            debug_assert!(false, "LLM concurrency permit missing from task map");
            return;
        };
        if entry.count > 1 {
            entry.count -= 1;
        } else {
            held.remove(key);
        }
    }
}

/// Shared admission control for LLM turns scoped to one model definition.
#[derive(Clone)]
pub struct LlmConcurrencyLimiter {
    max_concurrent_calls: Option<NonZeroUsize>,
    semaphore: Option<Arc<Semaphore>>,
}

impl LlmConcurrencyLimiter {
    pub fn new(max_concurrent_calls: Option<NonZeroUsize>) -> Self {
        Self {
            max_concurrent_calls,
            semaphore: max_concurrent_calls.map(|max| Arc::new(Semaphore::new(max.get()))),
        }
    }

    pub fn max_concurrent_calls(&self) -> Option<NonZeroUsize> {
        self.max_concurrent_calls
    }

    pub async fn acquire(&self) -> Option<LlmConcurrencyPermit> {
        self.acquire_with_wait_observer(|| {}).await
    }

    async fn acquire_with_wait_observer(
        &self,
        on_wait: impl FnOnce(),
    ) -> Option<LlmConcurrencyPermit> {
        let semaphore = self.semaphore.as_ref()?;
        let task_key =
            tokio::task::try_id().map(|task_id| (task_id, Arc::as_ptr(semaphore) as usize));
        if let Some(key) = task_key {
            let mut held = held_concurrency_permits()
                .lock()
                .expect("LLM concurrency permit map mutex poisoned");
            if let Some(entry) = held.get_mut(&key) {
                entry.count += 1;
                return Some(LlmConcurrencyPermit::task_scoped(key));
            }
        }

        let permit = match semaphore.clone().try_acquire_owned() {
            Ok(permit) => permit,
            Err(TryAcquireError::NoPermits) => {
                on_wait();
                semaphore
                    .clone()
                    .acquire_owned()
                    .await
                    .expect("LLM concurrency semaphore is never closed")
            }
            Err(TryAcquireError::Closed) => {
                panic!("LLM concurrency semaphore is never closed");
            }
        };
        let Some(key) = task_key else {
            return Some(LlmConcurrencyPermit::unscoped(permit));
        };
        held_concurrency_permits()
            .lock()
            .expect("LLM concurrency permit map mutex poisoned")
            .insert(
                key,
                HeldConcurrencyPermit {
                    count: 1,
                    _permit: permit,
                },
            );
        Some(LlmConcurrencyPermit::task_scoped(key))
    }
}

/// A [`Lutum`] handle plus any runtime admission permit held for its use.
///
/// Dropping this value releases the concurrent-call slot and emits an
/// [`RuntimeEvent::LlmCompleted`] paired with the [`RuntimeEvent::LlmAccessed`]
/// that was issued when this lease was acquired. It dereferences to [`Lutum`]
/// so module code can bind the value and pass `&lutum`.
pub struct LlmLease {
    lutum: Lutum,
    _permit: Option<LlmConcurrencyPermit>,
    completion: Option<LlmLeaseCompletion>,
}

struct LlmLeaseCompletion {
    events: RuntimeEventEmitter,
    owner: ModuleInstanceId,
    tier: ModelTier,
    call: u64,
}

impl LlmLease {
    fn new(
        lutum: Lutum,
        permit: Option<LlmConcurrencyPermit>,
        completion: LlmLeaseCompletion,
    ) -> Self {
        Self {
            lutum,
            _permit: permit,
            completion: Some(completion),
        }
    }

    pub fn lutum(&self) -> &Lutum {
        &self.lutum
    }
}

impl Deref for LlmLease {
    type Target = Lutum;

    fn deref(&self) -> &Self::Target {
        &self.lutum
    }
}

impl Drop for LlmLease {
    fn drop(&mut self) {
        if let Some(completion) = self.completion.take() {
            completion
                .events
                .llm_completed(completion.owner, completion.tier, completion.call);
        }
    }
}

/// LLM-access capability.
///
/// Owner-stamped and key-stamped: `lutum()` returns a leased [`Lutum`] handle
/// bound to the tier chosen for this named session/LLM key at module
/// construction.
///
/// Modules consume the returned `Lutum` directly — typically by passing it to
/// `lutum::Session`'s turn builder collect step
/// (e.g. `session.text_turn().collect(&lutum)`) for a single-turn or agent-loop
/// activation. The capability deliberately stops at the `Lutum` boundary;
/// everything past it (Session shape, prompts, tools) is the module's own
/// concern.
#[derive(Clone)]
pub struct LlmAccess {
    owner: ModuleInstanceId,
    key: String,
    tier: ModelTier,
    tiers: LutumTiers,
    events: RuntimeEventEmitter,
}

impl LlmAccess {
    pub(crate) fn new(
        owner: ModuleInstanceId,
        key: impl Into<String>,
        tier: ModelTier,
        tiers: LutumTiers,
        events: RuntimeEventEmitter,
    ) -> Self {
        Self {
            owner,
            key: key.into(),
            tier,
            tiers,
            events,
        }
    }

    /// A leased [`Lutum`] for this handle's configured tier.
    pub async fn lutum(&self) -> LlmLease {
        self.lutum_with_metadata(LlmRequestSource::ModuleTurn).await
    }

    pub fn tier(&self) -> ModelTier {
        self.tier
    }

    pub fn key(&self) -> &str {
        &self.key
    }

    async fn lutum_with_metadata(&self, source: LlmRequestSource) -> LlmLease {
        let tier = self.tier;
        let handle = self.tiers.pick_handle(tier);
        let events = self.events.clone();
        let owner = self.owner.clone();
        let permit = handle
            .concurrency
            .acquire_with_wait_observer(|| {
                events.llm_semaphore_wait_started(owner, tier);
            })
            .await;
        let call = self.events.llm_accessed(self.owner.clone(), tier);
        let lutum = if let Some((activation_id, activation_attempt, batch)) =
            current_activation_llm_request_metadata()
        {
            handle.lutum.clone().with_extension(LlmRequestMetadata {
                owner: self.owner.clone(),
                tier,
                source,
                session_key: Some(self.key.clone()),
                activation_id,
                activation_attempt,
                batch,
            })
        } else {
            handle.lutum.clone()
        };
        LlmLease::new(
            lutum,
            permit,
            LlmLeaseCompletion {
                events: self.events.clone(),
                owner: self.owner.clone(),
                tier,
                call,
            },
        )
    }
}

fn truncated_batch_debug(debug: &str) -> String {
    let mut out = String::with_capacity(debug.len().min(MAX_LLM_BATCH_DEBUG_CHARS));
    for (index, ch) in debug.chars().enumerate() {
        if index == MAX_LLM_BATCH_DEBUG_CHARS {
            return out;
        }
        out.push(ch);
    }
    out
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroUsize;
    use std::rc::Rc;
    use std::sync::{Arc, Mutex};
    use std::time::Duration;

    use lutum::{
        FinishReason, Lutum, LutumHooksSet, MockLlmAdapter, MockTextScenario, ModelInput,
        ModelInputHookContext, OnModelInput, RawTextTurnEvent, SharedPoolBudgetManager,
        SharedPoolBudgetOptions, Usage,
    };
    use nuillu_types::{ModelTier, ModuleActivationId, ModuleInstanceId, ReplicaIndex, builtin};

    use crate::ports::PortError;
    use crate::runtime_events::{RuntimeEvent, RuntimeEventEmitter, RuntimeEventSink};
    use crate::{LlmTierHandle, LutumTiers};

    use super::{
        LlmAccess, LlmBatchDebug, LlmConcurrencyLimiter, LlmRequestMetadata, LlmRequestSource,
        with_activation_llm_request_metadata,
    };

    #[derive(Debug, Default)]
    struct RecordingSink {
        events: Mutex<Vec<RuntimeEvent>>,
    }

    impl RuntimeEventSink for RecordingSink {
        fn on_event(&self, event: RuntimeEvent) -> Result<(), PortError> {
            self.events.lock().expect("event lock poisoned").push(event);
            Ok(())
        }
    }

    #[derive(Clone)]
    struct RecordingModelInputHook {
        seen: Arc<Mutex<Vec<Option<LlmRequestMetadata>>>>,
    }

    impl OnModelInput for RecordingModelInputHook {
        async fn call(&self, cx: &ModelInputHookContext<'_>) {
            self.seen
                .lock()
                .expect("metadata lock poisoned")
                .push(cx.extensions().get::<LlmRequestMetadata>().cloned());
        }
    }

    #[tokio::test]
    async fn lutum_emits_one_runtime_event_per_acquisition() {
        let owner = ModuleInstanceId::new(builtin::cognition_gate(), ReplicaIndex::ZERO);
        let adapter = Arc::new(MockLlmAdapter::new());
        let budget = SharedPoolBudgetManager::new(SharedPoolBudgetOptions::default());
        let lutum = Lutum::new(adapter, budget);
        let tiers = crate::LutumTiers::from_shared_lutum(lutum);
        let sink = Rc::new(RecordingSink::default());
        let events = RuntimeEventEmitter::new(sink.clone());
        let access = LlmAccess::new(owner.clone(), "main", ModelTier::Premium, tiers, events);

        let _ = access.lutum().await;
        let _ = access.lutum().await;

        let actual = sink.events.lock().expect("event lock poisoned").clone();
        assert_eq!(
            actual,
            vec![
                RuntimeEvent::LlmAccessed {
                    sequence: 0,
                    call: 0,
                    owner: owner.clone(),
                    tier: ModelTier::Premium,
                },
                RuntimeEvent::LlmCompleted {
                    sequence: 1,
                    call: 0,
                    owner: owner.clone(),
                    tier: ModelTier::Premium,
                },
                RuntimeEvent::LlmAccessed {
                    sequence: 2,
                    call: 1,
                    owner: owner.clone(),
                    tier: ModelTier::Premium,
                },
                RuntimeEvent::LlmCompleted {
                    sequence: 3,
                    call: 1,
                    owner,
                    tier: ModelTier::Premium,
                },
            ]
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn lutum_emits_semaphore_wait_event_only_when_acquisition_blocks() {
        let local = tokio::task::LocalSet::new();
        local
            .run_until(async {
                let owner = ModuleInstanceId::new(builtin::cognition_gate(), ReplicaIndex::ZERO);
                let adapter = Arc::new(MockLlmAdapter::new());
                let budget = SharedPoolBudgetManager::new(SharedPoolBudgetOptions::default());
                let lutum = Lutum::new(adapter, budget);
                let limiter = LlmConcurrencyLimiter::new(NonZeroUsize::new(1));
                let handle = LlmTierHandle::new(lutum, limiter, "test", false);
                let tiers = LutumTiers {
                    cheap: handle.clone(),
                    default: handle.clone(),
                    premium: handle,
                };
                let sink = Rc::new(RecordingSink::default());
                let events = RuntimeEventEmitter::new(sink.clone());
                let access =
                    LlmAccess::new(owner.clone(), "main", ModelTier::Default, tiers, events);

                let first = access.lutum().await;
                assert_eq!(
                    sink.events.lock().expect("event lock poisoned").as_slice(),
                    &[RuntimeEvent::LlmAccessed {
                        sequence: 0,
                        call: 0,
                        owner: owner.clone(),
                        tier: ModelTier::Default,
                    }]
                );

                let second = tokio::task::spawn_local({
                    let access = access.clone();
                    async move { access.lutum().await }
                });
                tokio::task::yield_now().await;

                assert_eq!(
                    sink.events.lock().expect("event lock poisoned").as_slice(),
                    &[
                        RuntimeEvent::LlmAccessed {
                            sequence: 0,
                            call: 0,
                            owner: owner.clone(),
                            tier: ModelTier::Default,
                        },
                        RuntimeEvent::LlmSemaphoreWaitStarted {
                            sequence: 1,
                            owner: owner.clone(),
                            tier: ModelTier::Default,
                        },
                    ]
                );

                drop(first);
                let second = second.await.expect("second local task should complete");
                drop(second);

                assert_eq!(
                    sink.events.lock().expect("event lock poisoned").as_slice(),
                    &[
                        RuntimeEvent::LlmAccessed {
                            sequence: 0,
                            call: 0,
                            owner: owner.clone(),
                            tier: ModelTier::Default,
                        },
                        RuntimeEvent::LlmSemaphoreWaitStarted {
                            sequence: 1,
                            owner: owner.clone(),
                            tier: ModelTier::Default,
                        },
                        RuntimeEvent::LlmCompleted {
                            sequence: 2,
                            call: 0,
                            owner: owner.clone(),
                            tier: ModelTier::Default,
                        },
                        RuntimeEvent::LlmAccessed {
                            sequence: 3,
                            call: 1,
                            owner: owner.clone(),
                            tier: ModelTier::Default,
                        },
                        RuntimeEvent::LlmCompleted {
                            sequence: 4,
                            call: 1,
                            owner,
                            tier: ModelTier::Default,
                        },
                    ]
                );
            })
            .await;
    }

    #[tokio::test]
    async fn lutum_attaches_owner_metadata_to_request_extensions() {
        let owner = ModuleInstanceId::new(builtin::cognition_gate(), ReplicaIndex::ZERO);
        let seen = Arc::new(Mutex::new(Vec::new()));
        let adapter = Arc::new(
            MockLlmAdapter::new().with_text_scenario(MockTextScenario::events(vec![
                Ok(RawTextTurnEvent::Started {
                    request_id: Some("req-metadata".into()),
                    model: "mock".into(),
                }),
                Ok(RawTextTurnEvent::TextDelta { delta: "ok".into() }),
                Ok(RawTextTurnEvent::Completed {
                    request_id: Some("req-metadata".into()),
                    finish_reason: FinishReason::Stop,
                    usage: Usage {
                        total_tokens: 1,
                        ..Usage::zero()
                    },
                }),
            ])),
        );
        let budget = SharedPoolBudgetManager::new(SharedPoolBudgetOptions::default());
        let lutum = Lutum::with_hooks(
            adapter,
            budget,
            LutumHooksSet::new()
                .with_on_model_input(RecordingModelInputHook { seen: seen.clone() }),
        );
        let tiers = crate::LutumTiers::from_shared_lutum(lutum);
        let sink = Rc::new(RecordingSink::default());
        let events = RuntimeEventEmitter::new(sink);
        let access = LlmAccess::new(owner.clone(), "main", ModelTier::Premium, tiers, events);

        let batch = crate::ModuleBatch::new("metadata-batch");
        let expected_batch = LlmBatchDebug::from_batch(&batch);
        let lutum = with_activation_llm_request_metadata(
            ModuleActivationId::new(7),
            1,
            &batch,
            access.lutum(),
        )
        .await;
        let _ = lutum
            .text_turn(ModelInput::new().user("hello"))
            .collect()
            .await
            .expect("mock text turn should complete");

        assert_eq!(
            seen.lock().expect("metadata lock poisoned").as_slice(),
            &[Some(LlmRequestMetadata {
                owner,
                tier: ModelTier::Premium,
                source: LlmRequestSource::ModuleTurn,
                session_key: Some("main".to_owned()),
                activation_id: ModuleActivationId::new(7),
                activation_attempt: 1,
                batch: expected_batch,
            })]
        );
    }

    #[tokio::test]
    async fn lutum_uses_configured_tier_and_key() {
        let owner = ModuleInstanceId::new(builtin::memory_compaction(), ReplicaIndex::ZERO);
        let seen = Arc::new(Mutex::new(Vec::new()));
        let adapter = Arc::new(
            MockLlmAdapter::new().with_text_scenario(MockTextScenario::events(vec![
                Ok(RawTextTurnEvent::Started {
                    request_id: Some("req-fixed-tier".into()),
                    model: "mock".into(),
                }),
                Ok(RawTextTurnEvent::TextDelta { delta: "ok".into() }),
                Ok(RawTextTurnEvent::Completed {
                    request_id: Some("req-fixed-tier".into()),
                    finish_reason: FinishReason::Stop,
                    usage: Usage {
                        total_tokens: 1,
                        ..Usage::zero()
                    },
                }),
            ])),
        );
        let budget = SharedPoolBudgetManager::new(SharedPoolBudgetOptions::default());
        let lutum = Lutum::with_hooks(
            adapter,
            budget,
            LutumHooksSet::new()
                .with_on_model_input(RecordingModelInputHook { seen: seen.clone() }),
        );
        let tiers = crate::LutumTiers::from_shared_lutum(lutum);
        let sink = Rc::new(RecordingSink::default());
        let events = RuntimeEventEmitter::new(sink.clone());
        let access = LlmAccess::new(owner.clone(), "audit", ModelTier::Default, tiers, events);

        let batch = crate::ModuleBatch::new("fixed-tier-batch");
        let expected_batch = LlmBatchDebug::from_batch(&batch);
        let lutum = with_activation_llm_request_metadata(
            ModuleActivationId::new(8),
            1,
            &batch,
            access.lutum(),
        )
        .await;
        let _ = lutum
            .text_turn(ModelInput::new().user("hello"))
            .collect()
            .await
            .expect("mock text turn should complete");
        drop(lutum);

        assert_eq!(
            sink.events.lock().expect("event lock poisoned").as_slice(),
            &[
                RuntimeEvent::LlmAccessed {
                    sequence: 0,
                    call: 0,
                    owner: owner.clone(),
                    tier: ModelTier::Default,
                },
                RuntimeEvent::LlmCompleted {
                    sequence: 1,
                    call: 0,
                    owner: owner.clone(),
                    tier: ModelTier::Default,
                },
            ]
        );
        assert_eq!(
            seen.lock().expect("metadata lock poisoned").as_slice(),
            &[Some(LlmRequestMetadata {
                owner,
                tier: ModelTier::Default,
                source: LlmRequestSource::ModuleTurn,
                session_key: Some("audit".to_owned()),
                activation_id: ModuleActivationId::new(8),
                activation_attempt: 1,
                batch: expected_batch,
            })]
        );
    }

    #[tokio::test]
    async fn concurrency_limiter_is_reentrant_within_current_task() {
        let limiter = LlmConcurrencyLimiter::new(Some(std::num::NonZeroUsize::new(1).unwrap()));

        tokio::spawn(async move {
            let first = limiter.acquire().await;
            let second = tokio::time::timeout(Duration::from_millis(5), limiter.acquire()).await;
            assert!(second.is_ok());

            drop(second);
            drop(first);
        })
        .await
        .unwrap();
    }

    #[tokio::test]
    async fn concurrency_limiter_waits_for_permit_held_by_another_task() {
        let limiter = LlmConcurrencyLimiter::new(Some(std::num::NonZeroUsize::new(1).unwrap()));

        let first = limiter.acquire().await;
        let second = tokio::spawn({
            let limiter = limiter.clone();
            async move { limiter.acquire().await }
        });
        tokio::pin!(second);
        assert!(
            tokio::time::timeout(Duration::from_millis(5), &mut second)
                .await
                .is_err()
        );

        drop(first);
        let second = tokio::time::timeout(Duration::from_millis(50), &mut second).await;
        assert!(matches!(second, Ok(Ok(Some(_)))));
    }
}
