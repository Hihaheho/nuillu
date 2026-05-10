use std::time::Duration;

use anyhow::Result;
use nuillu_module::AttentionControlRequest;
use tokio::time::Instant;

use crate::AttentionControllerModule;

const DEFAULT_CONTROL_SILENT_WINDOW: Duration = Duration::from_millis(100);
const DEFAULT_CONTROL_BUDGET: Duration = Duration::from_secs(1);

#[derive(Debug, Default)]
pub struct NextBatch {
    pub(crate) requests: Vec<AttentionControlRequest>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct AttentionControlBatchConfig {
    silent_window: Duration,
    budget: Duration,
}

impl Default for AttentionControlBatchConfig {
    fn default() -> Self {
        Self {
            silent_window: DEFAULT_CONTROL_SILENT_WINDOW,
            budget: DEFAULT_CONTROL_BUDGET,
        }
    }
}

impl AttentionControllerModule {
    pub(crate) async fn next_batch(&mut self) -> Result<NextBatch> {
        let mut batch = self.await_first_batch().await?;
        let _ = self.collect_ready_events_into_batch(&mut batch)?;
        self.collect_attention_control_burst(&mut batch).await?;
        Ok(batch)
    }

    async fn await_first_batch(&mut self) -> Result<NextBatch> {
        let mut batch = NextBatch::default();
        tokio::select! {
            update = self.updates.next_item() => {
                let _ = update?;
            }
            request = self.requests.next_item() => {
                batch.requests.push(request?.body);
            }
        }
        Ok(batch)
    }

    async fn collect_attention_control_burst(&mut self, batch: &mut NextBatch) -> Result<()> {
        let mut waited = Duration::ZERO;
        while waited < self.batching.budget {
            let remaining = self.batching.budget.saturating_sub(waited);
            let wait_for = std::cmp::min(self.batching.silent_window, remaining);
            if wait_for.is_zero() {
                break;
            }

            let started = Instant::now();
            tokio::select! {
                update = self.updates.next_item() => {
                    let _ = update?;
                    waited += std::cmp::min(started.elapsed(), wait_for);
                    let _ = self.collect_ready_events_into_batch(batch)?;
                }
                request = self.requests.next_item() => {
                    batch.requests.push(request?.body);
                    waited += std::cmp::min(started.elapsed(), wait_for);
                    let _ = self.collect_ready_events_into_batch(batch)?;
                }
                _ = tokio::time::sleep(wait_for) => {
                    waited += wait_for;
                    let ready = self.collect_ready_events_into_batch(batch)?;
                    if ready.is_empty() {
                        break;
                    }
                }
            }
        }
        Ok(())
    }

    fn collect_ready_events_into_batch(&mut self, batch: &mut NextBatch) -> Result<ReadyCounts> {
        let memo_updates = self.updates.take_ready_items()?.items.len();
        let requests = self.requests.take_ready_items()?;
        let request_count = requests.items.len();
        batch
            .requests
            .extend(requests.items.into_iter().map(|envelope| envelope.body));
        Ok(ReadyCounts {
            memo_updates,
            requests: request_count,
        })
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct ReadyCounts {
    memo_updates: usize,
    requests: usize,
}

impl ReadyCounts {
    fn is_empty(self) -> bool {
        self.memo_updates == 0 && self.requests == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::cell::RefCell;
    use std::rc::Rc;
    use std::sync::Arc;

    use async_trait::async_trait;
    use lutum::{Lutum, MockLlmAdapter, SharedPoolBudgetManager, SharedPoolBudgetOptions};
    use nuillu_blackboard::{Blackboard, Bpm, linear_ratio_fn};
    use nuillu_module::ports::{
        NoopCognitionLogRepository, NoopFileSearchProvider, NoopMemoryStore, NoopUtteranceSink,
        SystemClock,
    };
    use nuillu_module::{
        CapabilityProviderPorts, CapabilityProviders, LutumTiers, MemoUpdated, Module,
        ModuleRegistry,
    };
    use nuillu_types::{ModuleInstanceId, ReplicaIndex, builtin};

    type BatchRecorder = Rc<RefCell<Vec<usize>>>;

    struct RecordingAttentionController {
        inner: AttentionControllerModule,
        recorder: BatchRecorder,
    }

    #[async_trait(?Send)]
    impl Module for RecordingAttentionController {
        type Batch = NextBatch;

        fn id() -> &'static str {
            AttentionControllerModule::id()
        }

        fn role_description() -> &'static str {
            AttentionControllerModule::role_description()
        }

        async fn next_batch(&mut self) -> Result<Self::Batch> {
            let batch = self.inner.next_batch().await?;
            self.recorder.borrow_mut().push(batch.requests.len());
            Ok(batch)
        }

        async fn activate(
            &mut self,
            _cx: &nuillu_module::ActivateCx<'_>,
            _batch: &Self::Batch,
        ) -> Result<()> {
            Ok(())
        }
    }

    fn test_caps() -> CapabilityProviders {
        let adapter = Arc::new(MockLlmAdapter::new());
        let budget = SharedPoolBudgetManager::new(SharedPoolBudgetOptions::default());
        let lutum = Lutum::new(adapter, budget);
        CapabilityProviders::new(CapabilityProviderPorts {
            blackboard: Blackboard::default(),
            cognition_log_port: Arc::new(NoopCognitionLogRepository),
            primary_memory_store: Arc::new(NoopMemoryStore),
            memory_replicas: Vec::new(),
            file_search: Arc::new(NoopFileSearchProvider),
            utterance_sink: Arc::new(NoopUtteranceSink),
            clock: Arc::new(SystemClock),
            tiers: LutumTiers {
                cheap: lutum.clone(),
                default: lutum.clone(),
                premium: lutum,
            },
        })
    }

    async fn build_recording_controller(
        caps: &CapabilityProviders,
        recorder: BatchRecorder,
        batching: AttentionControlBatchConfig,
    ) -> nuillu_module::AllocatedModule {
        let modules = ModuleRegistry::new()
            .register(
                1..=1,
                Bpm::from_f64(60_000.0)..=Bpm::from_f64(60_000.0),
                linear_ratio_fn,
                move |caps| RecordingAttentionController {
                    inner: AttentionControllerModule::new(
                        caps.memo_updated_inbox(),
                        caps.attention_control_inbox(),
                        caps.blackboard_reader(),
                        caps.cognition_log_reader(),
                        caps.allocation_reader(),
                        caps.allocation_writer(),
                        caps.memo(),
                        caps.llm_access(),
                    )
                    .with_batch_config(batching),
                    recorder: recorder.clone(),
                },
            )
            .unwrap()
            .build(caps)
            .await
            .unwrap();
        let (_, mut modules) = modules.into_parts();
        assert_eq!(modules.len(), 1);
        modules.remove(0)
    }

    #[test]
    fn default_batch_budget_is_one_second() {
        let config = AttentionControlBatchConfig::default();

        assert_eq!(config.silent_window, Duration::from_millis(100));
        assert_eq!(config.budget, Duration::from_secs(1));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn memo_update_waits_silent_window_for_attention_control_request() {
        let caps = test_caps();
        let recorder = Rc::new(RefCell::new(Vec::new()));
        let mut controller = build_recording_controller(
            &caps,
            recorder.clone(),
            AttentionControlBatchConfig {
                silent_window: Duration::from_millis(10),
                budget: Duration::from_millis(50),
            },
        )
        .await;
        let harness = caps.internal_harness_io();
        let memo_updates = harness.memo_updated_mailbox();
        let requests = harness.attention_control_mailbox();

        memo_updates
            .publish(MemoUpdated {
                owner: ModuleInstanceId::new(builtin::sensory(), ReplicaIndex::ZERO),
                index: 1,
            })
            .await
            .expect("controller memo subscriber exists");

        let delayed_request = async {
            tokio::time::sleep(Duration::from_millis(2)).await;
            requests
                .publish(AttentionControlRequest::query("Which route is safe?"))
                .await
                .expect("controller request subscriber exists");
        };
        let next_batch = controller.next_batch();

        let (batch_result, _) = tokio::join!(next_batch, delayed_request);
        batch_result.expect("controller next batch succeeds");

        assert_eq!(recorder.borrow().as_slice(), &[1]);
    }
}
