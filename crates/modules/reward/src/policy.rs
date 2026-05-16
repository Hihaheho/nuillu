use std::rc::Rc;
use std::time::Duration;

use anyhow::{Context, Result};
use async_trait::async_trait;
use lutum::{Session, StructuredTurnOutcome};
use nuillu_blackboard::{
    Blackboard, BlackboardCommand, CognitionLogEntryRecord, MemoLogRecord, PolicyMetaPatch,
};
use nuillu_module::ports::PortError;
use nuillu_module::{
    AllocationReader, AllocationUpdatedInbox, CognitionLogEvictedInbox, InteroceptiveReader,
    LlmAccess, MemoLogEvictedInbox, Module, format_current_attention_guidance,
    push_formatted_cognition_log_batch, push_formatted_memo_log_batch,
};
use nuillu_types::{ModuleId, PolicyIndex, PolicyRank, SignedUnitF32, UnitF32};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use tokio::time::Instant;

use crate::{IndexedPolicy, NewPolicy, PolicyStore, fanout_policy_put};

const SYSTEM_PROMPT: &str = r#"You are the policy module.
Inspect evicted memo-log and cognition-log evidence, then preserve successful or distinctive
behavior patterns as new tentative policies.
Only create policies when the trigger and behavior are concrete enough to reuse later.
Never rewrite existing policy records; refined behavior is a new policy."#;
const DEFAULT_POLICY_BATCH_SILENT_WINDOW: Duration = Duration::from_millis(100);
const DEFAULT_POLICY_BATCH_BUDGET: Duration = Duration::from_secs(1);

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct PolicyFormationDecision {
    pub candidates: Vec<PolicyCandidate>,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct PolicyCandidate {
    pub trigger: String,
    pub behavior: String,
    pub reason: String,
}

/// Inserts new tentative policies, mirrors metadata onto the blackboard, and
/// fans out indexed writes to replica stores.
#[derive(Clone)]
pub struct PolicyWriter {
    primary_store: Rc<dyn PolicyStore>,
    replicas: Vec<Rc<dyn PolicyStore>>,
    blackboard: Blackboard,
}

impl PolicyWriter {
    pub(crate) fn new(
        primary_store: Rc<dyn PolicyStore>,
        replicas: Vec<Rc<dyn PolicyStore>>,
        blackboard: Blackboard,
    ) -> Self {
        Self {
            primary_store,
            replicas,
            blackboard,
        }
    }

    pub async fn insert(
        &self,
        trigger: String,
        behavior: String,
        decay_secs: i64,
    ) -> Result<PolicyIndex, PortError> {
        let new = NewPolicy {
            trigger: trigger.clone(),
            behavior: behavior.clone(),
            rank: PolicyRank::Tentative,
            expected_reward: SignedUnitF32::ZERO,
            confidence: UnitF32::ZERO,
            value: SignedUnitF32::ZERO,
            reward_tokens: 0,
            decay_remaining_secs: decay_secs,
        };
        let index = self.primary_store.insert(new).await?;
        let indexed = IndexedPolicy {
            index: index.clone(),
            trigger,
            behavior,
            rank: PolicyRank::Tentative,
            expected_reward: SignedUnitF32::ZERO,
            confidence: UnitF32::ZERO,
            value: SignedUnitF32::ZERO,
            reward_tokens: 0,
            decay_remaining_secs: decay_secs,
        };
        fanout_policy_put(&self.replicas, indexed).await;
        self.blackboard
            .apply(BlackboardCommand::UpsertPolicyMetadata {
                index: index.clone(),
                rank_if_new: PolicyRank::Tentative,
                decay_if_new_secs: decay_secs,
                patch: PolicyMetaPatch {
                    rank: Some(PolicyRank::Tentative),
                    expected_reward: Some(SignedUnitF32::ZERO),
                    confidence: Some(UnitF32::ZERO),
                    value: Some(SignedUnitF32::ZERO),
                    reward_tokens: Some(0),
                    decay_remaining_secs: Some(decay_secs),
                    ..Default::default()
                },
            })
            .await;
        Ok(index)
    }
}

pub struct PolicyModule {
    owner: ModuleId,
    memo_evictions: MemoLogEvictedInbox,
    cognition_evictions: CognitionLogEvictedInbox,
    allocation_updates: AllocationUpdatedInbox,
    allocation: AllocationReader,
    interoception: InteroceptiveReader,
    writer: PolicyWriter,
    llm: LlmAccess,
    session: Session,
    batching: PolicyBatchConfig,
}

impl PolicyModule {
    pub fn new(
        memo_evictions: MemoLogEvictedInbox,
        cognition_evictions: CognitionLogEvictedInbox,
        allocation_updates: AllocationUpdatedInbox,
        allocation: AllocationReader,
        interoception: InteroceptiveReader,
        writer: PolicyWriter,
        llm: LlmAccess,
    ) -> Self {
        Self {
            owner: ModuleId::new(<Self as Module>::id()).expect("policy id is valid"),
            memo_evictions,
            cognition_evictions,
            allocation_updates,
            allocation,
            interoception,
            writer,
            llm,
            session: Session::new(),
            batching: PolicyBatchConfig::default(),
        }
    }

    #[cfg(test)]
    fn with_batch_config(mut self, batching: PolicyBatchConfig) -> Self {
        self.batching = batching;
        self
    }

    async fn activate(
        &mut self,
        cx: &nuillu_module::ActivateCx<'_>,
        batch: &PolicyBatch,
    ) -> Result<()> {
        push_formatted_memo_log_batch(&mut self.session, &batch.memo_logs, cx.now());
        push_formatted_cognition_log_batch(&mut self.session, &batch.cognition_log, cx.now());
        let allocation = self.allocation.snapshot().await;
        let interoception = self.interoception.snapshot().await;
        self.session.push_ephemeral_system(SYSTEM_PROMPT);
        if let Some(guidance) = format_current_attention_guidance(&allocation) {
            self.session.push_ephemeral_system(guidance);
        }
        self.session.push_ephemeral_system(format!(
            "Current interoception: affect_arousal={:.2}; valence={:.2}; emotion={}",
            interoception.affect_arousal,
            interoception.valence,
            if interoception.emotion.trim().is_empty() {
                "unknown"
            } else {
                interoception.emotion.trim()
            }
        ));
        self.session.push_ephemeral_user(format!(
            "Policy formation activation for {}:\nAllocation updated: {}\nEvicted memo-log entries: {}\nEvicted cognition-log entries: {}",
            self.owner,
            if batch.allocation_updated { "yes" } else { "no" },
            batch.memo_logs.len(),
            batch.cognition_log.len(),
        ));

        let lutum = self.llm.lutum().await;
        let result = self
            .session
            .structured_turn::<PolicyFormationDecision>(&lutum)
            .collect()
            .await
            .context("policy structured turn failed")?;
        let StructuredTurnOutcome::Structured(decision) = result.semantic else {
            return Ok(());
        };
        for candidate in decision.candidates {
            let trigger = candidate.trigger.trim();
            let behavior = candidate.behavior.trim();
            if trigger.is_empty() || behavior.is_empty() {
                continue;
            }
            self.writer
                .insert(trigger.to_owned(), behavior.to_owned(), 86_400)
                .await?;
        }
        Ok(())
    }

    async fn next_batch(&mut self) -> Result<PolicyBatch> {
        let mut batch = self.await_first_batch().await?;
        let _ = self.collect_ready_events_into_batch(&mut batch)?;
        self.collect_eviction_burst(&mut batch).await?;
        Ok(batch)
    }

    async fn await_first_batch(&mut self) -> Result<PolicyBatch> {
        let batch = tokio::select! {
            update = self.allocation_updates.next_item() => {
                let _ = update?;
                PolicyBatch::allocation_update()
            }
            evicted = self.memo_evictions.next_item() => {
                PolicyBatch::memo_eviction(evicted?.body)
            }
            evicted = self.cognition_evictions.next_item() => {
                PolicyBatch::cognition_log_eviction(evicted?.body)
            }
        };
        Ok(batch)
    }

    async fn collect_eviction_burst(&mut self, batch: &mut PolicyBatch) -> Result<()> {
        let mut waited = Duration::ZERO;
        while waited < self.batching.budget {
            let remaining = self.batching.budget.saturating_sub(waited);
            let wait_for = std::cmp::min(self.batching.silent_window, remaining);
            if wait_for.is_zero() {
                break;
            }

            let started = Instant::now();
            tokio::select! {
                update = self.allocation_updates.next_item() => {
                    let _ = update?;
                    waited += std::cmp::min(started.elapsed(), wait_for);
                    let _ = self.collect_ready_events_into_batch(batch)?;
                }
                evicted = self.memo_evictions.next_item() => {
                    batch.memo_logs.push(evicted?.body);
                    waited += std::cmp::min(started.elapsed(), wait_for);
                    let _ = self.collect_ready_events_into_batch(batch)?;
                }
                evicted = self.cognition_evictions.next_item() => {
                    batch.cognition_log.push(evicted?.body);
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

    fn collect_ready_events_into_batch(&mut self, batch: &mut PolicyBatch) -> Result<ReadyCounts> {
        let allocation_updates = self.allocation_updates.take_ready_items()?.items.len();
        if allocation_updates > 0 {
            batch.mark_allocation_updated();
        }

        let memo_evictions = self.memo_evictions.take_ready_items()?;
        let memo_count = memo_evictions.items.len();
        batch.memo_logs.extend(
            memo_evictions
                .items
                .into_iter()
                .map(|envelope| envelope.body),
        );

        let cognition_evictions = self.cognition_evictions.take_ready_items()?;
        let cognition_count = cognition_evictions.items.len();
        batch.cognition_log.extend(
            cognition_evictions
                .items
                .into_iter()
                .map(|envelope| envelope.body),
        );

        Ok(ReadyCounts {
            allocation_updates,
            memo_evictions: memo_count,
            cognition_evictions: cognition_count,
        })
    }
}

#[derive(Debug, Default)]
pub struct PolicyBatch {
    pub(crate) allocation_updated: bool,
    pub(crate) memo_logs: Vec<MemoLogRecord>,
    pub(crate) cognition_log: Vec<CognitionLogEntryRecord>,
}

impl PolicyBatch {
    fn allocation_update() -> Self {
        Self {
            allocation_updated: true,
            ..Self::default()
        }
    }

    fn memo_eviction(record: MemoLogRecord) -> Self {
        Self {
            memo_logs: vec![record],
            ..Self::default()
        }
    }

    fn cognition_log_eviction(record: CognitionLogEntryRecord) -> Self {
        Self {
            cognition_log: vec![record],
            ..Self::default()
        }
    }

    fn mark_allocation_updated(&mut self) {
        self.allocation_updated = true;
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PolicyBatchConfig {
    silent_window: Duration,
    budget: Duration,
}

impl Default for PolicyBatchConfig {
    fn default() -> Self {
        Self {
            silent_window: DEFAULT_POLICY_BATCH_SILENT_WINDOW,
            budget: DEFAULT_POLICY_BATCH_BUDGET,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct ReadyCounts {
    allocation_updates: usize,
    memo_evictions: usize,
    cognition_evictions: usize,
}

impl ReadyCounts {
    fn is_empty(self) -> bool {
        self.allocation_updates == 0 && self.memo_evictions == 0 && self.cognition_evictions == 0
    }
}

#[async_trait(?Send)]
impl Module for PolicyModule {
    type Batch = PolicyBatch;

    fn id() -> &'static str {
        "policy"
    }

    fn role_description() -> &'static str {
        "Creates new tentative trigger/behavior policies from successful or distinctive behavior patterns."
    }

    async fn next_batch(&mut self) -> Result<Self::Batch> {
        PolicyModule::next_batch(self).await
    }

    async fn activate(
        &mut self,
        cx: &nuillu_module::ActivateCx<'_>,
        batch: &Self::Batch,
    ) -> Result<()> {
        PolicyModule::activate(self, cx, batch).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::cell::RefCell;
    use std::rc::Rc;
    use std::sync::Arc;

    use chrono::{TimeZone, Utc};
    use lutum::{Lutum, MockLlmAdapter, SharedPoolBudgetManager, SharedPoolBudgetOptions};
    use nuillu_blackboard::{Bpm, ModulePolicy, linear_ratio_fn};
    use nuillu_module::ports::{NoopCognitionLogRepository, SystemClock};
    use nuillu_module::{CapabilityProviderPorts, CapabilityProviders, LutumTiers, ModuleRegistry};
    use nuillu_types::{ModuleInstanceId, ReplicaCapRange, ReplicaIndex, builtin};

    use crate::{NoopPolicyStore, PolicyCapabilities};

    type BatchRecorder = Rc<RefCell<Vec<(bool, usize, usize)>>>;

    struct RecordingPolicy {
        inner: PolicyModule,
        recorder: BatchRecorder,
    }

    #[async_trait(?Send)]
    impl Module for RecordingPolicy {
        type Batch = PolicyBatch;

        fn id() -> &'static str {
            PolicyModule::id()
        }

        fn role_description() -> &'static str {
            PolicyModule::role_description()
        }

        async fn next_batch(&mut self) -> Result<Self::Batch> {
            let batch = self.inner.next_batch().await?;
            self.recorder.borrow_mut().push((
                batch.allocation_updated,
                batch.memo_logs.len(),
                batch.cognition_log.len(),
            ));
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

    fn test_caps() -> (Blackboard, CapabilityProviders, PolicyCapabilities) {
        let blackboard = Blackboard::default();
        let adapter = Arc::new(MockLlmAdapter::new());
        let budget = SharedPoolBudgetManager::new(SharedPoolBudgetOptions::default());
        let lutum = Lutum::new(adapter, budget);
        let clock = Rc::new(SystemClock);
        let caps = CapabilityProviders::new(CapabilityProviderPorts {
            blackboard: blackboard.clone(),
            cognition_log_port: Rc::new(NoopCognitionLogRepository),
            clock: clock.clone(),
            tiers: LutumTiers {
                cheap: lutum.clone(),
                default: lutum.clone(),
                premium: lutum,
            },
        });
        let policy_caps = PolicyCapabilities::new(
            blackboard.clone(),
            clock,
            Rc::new(NoopPolicyStore),
            Vec::new(),
        );
        (blackboard, caps, policy_caps)
    }

    async fn build_recording_policy(
        caps: &CapabilityProviders,
        policy_caps: PolicyCapabilities,
        recorder: BatchRecorder,
        batching: PolicyBatchConfig,
    ) -> nuillu_module::AllocatedModule {
        let modules = ModuleRegistry::new()
            .register(
                ModulePolicy::new(
                    ReplicaCapRange::new(1, 1).unwrap(),
                    Bpm::from_f64(60_000.0)..=Bpm::from_f64(60_000.0),
                    linear_ratio_fn,
                ),
                move |caps| RecordingPolicy {
                    inner: PolicyModule::new(
                        caps.memo_log_evicted_inbox(),
                        caps.cognition_log_evicted_inbox(),
                        caps.allocation_updated_inbox(),
                        caps.allocation_reader(),
                        caps.interoception_reader(),
                        policy_caps.writer(),
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
        modules.remove(0)
    }

    fn memo_record(index: u64, content: &str) -> MemoLogRecord {
        MemoLogRecord {
            owner: ModuleInstanceId::new(builtin::sensory(), ReplicaIndex::ZERO),
            index,
            written_at: Utc.timestamp_opt(index as i64, 0).unwrap(),
            content: content.to_owned(),
        }
    }

    fn cognition_record(index: u64, content: &str) -> CognitionLogEntryRecord {
        CognitionLogEntryRecord {
            index,
            source: ModuleInstanceId::new(builtin::cognition_gate(), ReplicaIndex::ZERO),
            entry: nuillu_blackboard::CognitionLogEntry {
                at: Utc.timestamp_opt(index as i64, 0).unwrap(),
                text: content.to_owned(),
            },
        }
    }

    #[tokio::test]
    async fn next_batch_collects_memo_and_cognition_eviction_burst() {
        let (_blackboard, caps, policy_caps) = test_caps();
        let recorder = Rc::new(RefCell::new(Vec::new()));
        let mut module = build_recording_policy(
            &caps,
            policy_caps,
            recorder.clone(),
            PolicyBatchConfig {
                silent_window: Duration::from_millis(10),
                budget: Duration::from_millis(50),
            },
        )
        .await;
        let harness = caps.internal_harness_io();
        let memo_evictions = harness.memo_log_evicted_mailbox();
        let cognition_evictions = harness.cognition_log_evicted_mailbox();

        memo_evictions
            .publish(memo_record(0, "first memo"))
            .await
            .expect("policy memo-eviction subscriber exists");

        let delayed_evictions = async {
            tokio::time::sleep(Duration::from_millis(2)).await;
            memo_evictions
                .publish(memo_record(1, "second memo"))
                .await
                .expect("policy memo-eviction subscriber exists");
            cognition_evictions
                .publish(cognition_record(2, "cognition"))
                .await
                .expect("policy cognition-eviction subscriber exists");
        };
        let next_batch = module.next_batch();

        let (batch_result, _) = tokio::join!(next_batch, delayed_evictions);
        batch_result.expect("policy next batch succeeds");

        assert_eq!(recorder.borrow().as_slice(), &[(false, 2, 1)]);
    }
}
