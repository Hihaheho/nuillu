//! Foundation integration test: trivial modules driven by their own
//! inbox loops, verifying the capability layer end-to-end.

use std::cell::RefCell;
use std::ops::RangeInclusive;
use std::rc::Rc;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use lutum::{
    FinishReason, Lutum, MockLlmAdapter, MockTextScenario, RawTextTurnEvent,
    SharedPoolBudgetManager, SharedPoolBudgetOptions, Usage,
};
use nuillu_agent::{AgentEventLoop, run};
use nuillu_blackboard::{
    AttentionStreamEvent, Blackboard, BlackboardCommand, MemoryMetaPatch, ModuleConfig,
    ResourceAllocation,
};
use nuillu_memory::MemoryModule;
use nuillu_module::{
    AllocatedModules, CapabilityProviders, LutumTiers, Memo, MemoryImportance, MemoryRequest,
    Module, ModuleCapabilityProviders, ModuleRegistry, PeriodicInbox, QueryInbox, QueryMailbox,
    QueryRequest, SelfModelRequest,
    ports::{
        AttentionRepository, FileSearchHit, FileSearchProvider, FileSearchQuery, IndexedMemory,
        MemoryQuery, MemoryRecord, MemoryStore, NewMemory, PortError, SystemClock, Utterance,
        UtteranceSink,
    },
};
use nuillu_predict::PredictModule;
use nuillu_surprise::SurpriseModule;
use nuillu_types::{
    MemoryContent, MemoryIndex, MemoryRank, ModelTier, ModuleId, ModuleInstanceId, builtin,
};
use tokio::sync::{Mutex, oneshot};
use tokio::task::LocalSet;

const ECHO_ID: &str = "echo";
const TICKER_ID: &str = "ticker";

fn echo_id() -> ModuleId {
    ModuleId::new(ECHO_ID).unwrap()
}

fn ticker_id() -> ModuleId {
    ModuleId::new(TICKER_ID).unwrap()
}

// ---------- modules ----------

struct TickerModule {
    periodic: PeriodicInbox,
    memo: Memo,
    query_mailbox: QueryMailbox,
    target_ticks: u32,
    on_done: Option<oneshot::Sender<()>>,
}

impl TickerModule {
    fn new(
        periodic: PeriodicInbox,
        memo: Memo,
        query_mailbox: QueryMailbox,
        target_ticks: u32,
        on_done: oneshot::Sender<()>,
    ) -> Self {
        Self {
            periodic,
            memo,
            query_mailbox,
            target_ticks,
            on_done: Some(on_done),
        }
    }
}

#[async_trait(?Send)]
impl Module for TickerModule {
    async fn run(&mut self) {
        let mut counter: u32 = 0;
        while self.periodic.next_tick().await.is_ok() {
            counter += 1;
            self.memo.write(format!("sent {counter} pings")).await;
            let _ = self
                .query_mailbox
                .publish(QueryRequest::new(format!("ping {counter}")))
                .await;
            if counter >= self.target_ticks
                && let Some(tx) = self.on_done.take()
            {
                let _ = tx.send(());
            }
        }
    }
}

struct EchoModule {
    query_inbox: QueryInbox,
    memo: Memo,
}

impl EchoModule {
    fn new(query_inbox: QueryInbox, memo: Memo) -> Self {
        Self { query_inbox, memo }
    }
}

#[async_trait(?Send)]
impl Module for EchoModule {
    async fn run(&mut self) {
        loop {
            match self.query_inbox.next_item().await {
                Ok(env) => {
                    self.memo
                        .write(format!("echoed {}", env.body.question))
                        .await
                }
                Err(_) => break,
            }
        }
    }
}

struct SummarizeStub {
    periodic: PeriodicInbox,
    writer: nuillu_module::AttentionWriter,
    fired: Arc<Mutex<bool>>,
}

#[async_trait(?Send)]
impl Module for SummarizeStub {
    async fn run(&mut self) {
        while self.periodic.next_tick().await.is_ok() {
            let mut fired = self.fired.lock().await;
            if *fired {
                continue;
            }
            *fired = true;
            self.writer.append("novel-event").await;
        }
    }
}

struct PeriodicCounter {
    periodic: PeriodicInbox,
    count: Arc<Mutex<u32>>,
}

#[async_trait(?Send)]
impl Module for PeriodicCounter {
    async fn run(&mut self) {
        while self.periodic.next_tick().await.is_ok() {
            *self.count.lock().await += 1;
        }
    }
}

struct BatchingPeriodicCounter {
    periodic: PeriodicInbox,
    count: Arc<Mutex<u32>>,
}

#[async_trait(?Send)]
impl Module for BatchingPeriodicCounter {
    async fn run(&mut self) {
        while self.periodic.next_tick().await.is_ok() {
            if self.periodic.take_ready_ticks().is_err() {
                return;
            }
            *self.count.lock().await += 1;
        }
    }
}

struct CapturedCapsModule;

#[async_trait(?Send)]
impl Module for CapturedCapsModule {
    async fn run(&mut self) {}
}

fn capture_caps(
    registry: ModuleRegistry,
    module: ModuleId,
    cap_range: RangeInclusive<u8>,
) -> (ModuleRegistry, Rc<RefCell<Vec<ModuleCapabilityProviders>>>) {
    let captured = Rc::new(RefCell::new(Vec::new()));
    let captured_for_builder = Rc::clone(&captured);
    let registry = registry
        .register(module, cap_range, move |caps| {
            captured_for_builder.borrow_mut().push(caps);
            CapturedCapsModule
        })
        .unwrap();
    (registry, captured)
}

async fn module_caps(
    caps: &CapabilityProviders,
    module: ModuleId,
    cap_range: RangeInclusive<u8>,
) -> Vec<ModuleCapabilityProviders> {
    let (registry, captured) = capture_caps(ModuleRegistry::new(), module, cap_range);
    let _modules = registry.build(caps).await.unwrap();
    std::mem::take(&mut *captured.borrow_mut())
}

async fn module_cap(
    caps: &CapabilityProviders,
    module: ModuleId,
    cap_range: RangeInclusive<u8>,
) -> ModuleCapabilityProviders {
    let mut caps = module_caps(caps, module, cap_range).await;
    assert_eq!(caps.len(), 1);
    caps.pop().unwrap()
}

async fn memory_modules_with_publisher(
    caps: &CapabilityProviders,
) -> (AllocatedModules, ModuleCapabilityProviders) {
    let (registry, surprise_caps) = capture_caps(ModuleRegistry::new(), builtin::surprise(), 0..=1);
    let modules = registry
        .register(builtin::memory(), 0..=1, |caps| {
            MemoryModule::new(
                caps.periodic_inbox(),
                caps.memory_request_inbox(),
                caps.activation_gate(),
                caps.blackboard_reader(),
                caps.memory_writer(),
                caps.llm_access(),
            )
        })
        .unwrap()
        .build(caps)
        .await
        .unwrap();
    let mut surprise_caps = std::mem::take(&mut *surprise_caps.borrow_mut());
    assert_eq!(surprise_caps.len(), 1);
    (modules, surprise_caps.pop().unwrap())
}

// ---------- tests ----------

#[tokio::test(flavor = "current_thread", start_paused = false)]
async fn typed_query_fanout_and_periodic_capabilities_work_together() {
    let local = LocalSet::new();
    local
        .run_until(async {
            let mut alloc = ResourceAllocation::default();
            alloc.set(
                ticker_id(),
                ModuleConfig {
                    replicas: 1,
                    tier: ModelTier::Default,
                    period: Some(Duration::from_millis(20)),
                    ..Default::default()
                },
            );
            alloc.set(
                echo_id(),
                ModuleConfig {
                    replicas: 1,
                    tier: ModelTier::Default,
                    period: None,
                    ..Default::default()
                },
            );

            let blackboard = Blackboard::with_allocation(alloc);
            let caps = test_caps(blackboard.clone());
            let mut event_loop = AgentEventLoop::new(caps.periodic_activation());

            let (done_tx, done_rx) = oneshot::channel();
            let done_tx = Rc::new(RefCell::new(Some(done_tx)));
            let modules = ModuleRegistry::new()
                .register(ticker_id(), 0..=1, {
                    let done_tx = Rc::clone(&done_tx);
                    move |caps| {
                        let done_tx = done_tx
                            .borrow_mut()
                            .take()
                            .expect("ticker module should be built once");
                        TickerModule::new(
                            caps.periodic_inbox(),
                            caps.memo(),
                            caps.query_mailbox(),
                            3,
                            done_tx,
                        )
                    }
                })
                .unwrap()
                .register(echo_id(), 0..=1, |caps| {
                    EchoModule::new(caps.query_inbox(), caps.memo())
                })
                .unwrap()
                .build(&caps)
                .await
                .unwrap();

            run(modules, async move {
                for _ in 0..3 {
                    event_loop.tick(Duration::from_millis(20)).await;
                    tokio::task::yield_now().await;
                }
                let _ = done_rx.await;
                for _ in 0..4 {
                    tokio::task::yield_now().await;
                }
            })
            .await
            .expect("scheduler returned err");

            blackboard
                .read(|bb| {
                    let ticker_memo = bb.memo(&ticker_id()).expect("ticker memo missing");
                    let echo_memo = bb.memo(&echo_id()).expect("echo memo missing");
                    assert!(
                        ticker_memo.starts_with("sent "),
                        "unexpected ticker memo: {ticker_memo}",
                    );
                    assert!(
                        echo_memo.starts_with("echoed ping "),
                        "unexpected echo memo: {echo_memo}",
                    );
                    assert_eq!(
                        bb.attention_stream().len(),
                        0,
                        "neither test module holds AttentionWriter",
                    );
                })
                .await;
        })
        .await;
}

#[tokio::test(flavor = "current_thread", start_paused = false)]
async fn attention_writer_capability_appends_to_stream() {
    let local = LocalSet::new();
    local
        .run_until(async {
            let mut alloc = ResourceAllocation::default();
            alloc.set(
                builtin::summarize(),
                ModuleConfig {
                    replicas: 1,
                    tier: ModelTier::Default,
                    period: Some(Duration::from_millis(15)),
                    ..Default::default()
                },
            );

            let blackboard = Blackboard::with_allocation(alloc);
            let caps = test_caps(blackboard.clone());
            let mut event_loop = AgentEventLoop::new(caps.periodic_activation());

            let fired = Arc::new(Mutex::new(false));
            let modules = ModuleRegistry::new()
                .register(builtin::summarize(), 0..=1, {
                    let fired = fired.clone();
                    move |caps| SummarizeStub {
                        periodic: caps.periodic_inbox(),
                        writer: caps.attention_writer(),
                        fired: fired.clone(),
                    }
                })
                .unwrap()
                .build(&caps)
                .await
                .unwrap();

            run(modules, async move {
                event_loop.tick(Duration::from_millis(15)).await;
                for _ in 0..10 {
                    if *fired.lock().await {
                        break;
                    }
                    tokio::task::yield_now().await;
                }
            })
            .await
            .expect("scheduler returned err");

            blackboard
                .read(|bb| {
                    assert_eq!(bb.attention_stream().len(), 1);
                    assert_eq!(bb.attention_stream().entries()[0].text, "novel-event");
                })
                .await;
        })
        .await;
}

#[tokio::test(flavor = "current_thread", start_paused = false)]
async fn periodic_tick_accumulates_until_period() {
    let local = LocalSet::new();
    local
        .run_until(async {
            let (blackboard, modules, count, mut event_loop) =
                periodic_counter_setup(Duration::from_millis(10), true).await;

            run(modules, async move {
                event_loop.tick(Duration::from_millis(6)).await;
                tokio::task::yield_now().await;
                assert_eq!(*count.lock().await, 0);

                event_loop.tick(Duration::from_millis(4)).await;
                tokio::task::yield_now().await;
                assert_eq!(*count.lock().await, 1);
            })
            .await
            .expect("scheduler returned err");

            blackboard.read(|_| ()).await;
        })
        .await;
}

#[tokio::test(flavor = "current_thread", start_paused = false)]
async fn periodic_tick_emits_at_most_once_and_carries_remainder() {
    let local = LocalSet::new();
    local
        .run_until(async {
            let (_blackboard, modules, count, mut event_loop) =
                periodic_counter_setup(Duration::from_millis(10), true).await;

            run(modules, async move {
                event_loop.tick(Duration::from_millis(25)).await;
                tokio::task::yield_now().await;
                assert_eq!(*count.lock().await, 1);

                event_loop.tick(Duration::from_millis(5)).await;
                tokio::task::yield_now().await;
                assert_eq!(*count.lock().await, 2);
            })
            .await
            .expect("scheduler returned err");
        })
        .await;
}

#[tokio::test(flavor = "current_thread", start_paused = false)]
async fn ready_periodic_ticks_can_be_collapsed_by_module_prelude() {
    let local = LocalSet::new();
    local
        .run_until(async {
            let mut alloc = ResourceAllocation::default();
            alloc.set(
                ticker_id(),
                ModuleConfig {
                    replicas: 1,
                    tier: ModelTier::Default,
                    period: Some(Duration::from_millis(10)),
                    ..Default::default()
                },
            );

            let blackboard = Blackboard::with_allocation(alloc);
            let caps = test_caps(blackboard);
            let mut event_loop = AgentEventLoop::new(caps.periodic_activation());
            let count = Arc::new(Mutex::new(0));
            let modules = ModuleRegistry::new()
                .register(ticker_id(), 0..=1, {
                    let count = count.clone();
                    move |caps| BatchingPeriodicCounter {
                        periodic: caps.periodic_inbox(),
                        count: count.clone(),
                    }
                })
                .unwrap()
                .build(&caps)
                .await
                .unwrap();

            run(modules, async move {
                event_loop.tick(Duration::from_millis(10)).await;
                event_loop.tick(Duration::from_millis(10)).await;
                event_loop.tick(Duration::from_millis(10)).await;
                tokio::task::yield_now().await;
                assert_eq!(*count.lock().await, 1);
            })
            .await
            .expect("scheduler returned err");
        })
        .await;
}

#[tokio::test(flavor = "current_thread", start_paused = false)]
async fn periodic_tick_clears_elapsed_while_disabled() {
    let local = LocalSet::new();
    local
        .run_until(async {
            let (blackboard, modules, count, mut event_loop) =
                periodic_counter_setup(Duration::from_millis(10), false).await;

            run(modules, async move {
                event_loop.tick(Duration::from_millis(50)).await;
                tokio::task::yield_now().await;
                assert_eq!(*count.lock().await, 0);

                set_counter_allocation(&blackboard, Duration::from_millis(10), true).await;
                event_loop.tick(Duration::from_millis(9)).await;
                tokio::task::yield_now().await;
                assert_eq!(*count.lock().await, 0);

                event_loop.tick(Duration::from_millis(1)).await;
                tokio::task::yield_now().await;
                assert_eq!(*count.lock().await, 1);
            })
            .await
            .expect("scheduler returned err");
        })
        .await;
}

#[tokio::test(flavor = "current_thread", start_paused = false)]
async fn periodic_tick_uses_latest_allocation_period() {
    let local = LocalSet::new();
    local
        .run_until(async {
            let (blackboard, modules, count, mut event_loop) =
                periodic_counter_setup(Duration::from_millis(20), true).await;

            run(modules, async move {
                event_loop.tick(Duration::from_millis(15)).await;
                tokio::task::yield_now().await;
                assert_eq!(*count.lock().await, 0);

                set_counter_allocation(&blackboard, Duration::from_millis(10), true).await;
                event_loop.tick(Duration::ZERO).await;
                tokio::task::yield_now().await;
                assert_eq!(*count.lock().await, 1);
            })
            .await
            .expect("scheduler returned err");
        })
        .await;
}

#[tokio::test(flavor = "current_thread", start_paused = false)]
async fn periodic_tick_only_targets_modules_with_periodic_inbox() {
    let local = LocalSet::new();
    local
        .run_until(async {
            let mut alloc = ResourceAllocation::default();
            alloc.set(
                ticker_id(),
                ModuleConfig {
                    replicas: 1,
                    tier: ModelTier::Default,
                    period: Some(Duration::from_millis(10)),
                    ..Default::default()
                },
            );
            alloc.set(
                echo_id(),
                ModuleConfig {
                    replicas: 1,
                    tier: ModelTier::Default,
                    period: Some(Duration::from_millis(10)),
                    ..Default::default()
                },
            );

            let blackboard = Blackboard::with_allocation(alloc);
            let caps = test_caps(blackboard);
            let mut event_loop = AgentEventLoop::new(caps.periodic_activation());
            let count = Arc::new(Mutex::new(0));
            let modules = ModuleRegistry::new()
                .register(ticker_id(), 0..=1, {
                    let count = count.clone();
                    move |caps| PeriodicCounter {
                        periodic: caps.periodic_inbox(),
                        count: count.clone(),
                    }
                })
                .unwrap()
                .build(&caps)
                .await
                .unwrap();

            run(modules, async move {
                event_loop.tick(Duration::from_millis(10)).await;
                tokio::task::yield_now().await;
                assert_eq!(*count.lock().await, 1);
            })
            .await
            .expect("scheduler returned err");
        })
        .await;
}

#[tokio::test(flavor = "current_thread", start_paused = false)]
async fn capabilities_are_non_exclusive() {
    let local = LocalSet::new();
    local
        .run_until(async {
            let blackboard = Blackboard::default();
            let caps = test_caps(blackboard);
            let summarize_caps = module_cap(&caps, builtin::summarize(), 0..=1).await;
            let controller_caps = module_cap(&caps, builtin::attention_controller(), 0..=1).await;
            let _w1 = summarize_caps.attention_writer();
            let _w2 = summarize_caps.attention_writer();
            let _a1 = controller_caps.allocation_writer();
            let _a2 = controller_caps.allocation_writer();
        })
        .await;
}

#[tokio::test(flavor = "current_thread", start_paused = false)]
async fn query_mailbox_fans_out_to_multiple_subscribers_with_owner_stamp() {
    let caps = test_caps(Blackboard::default());
    let publisher_caps = module_cap(&caps, ticker_id(), 0..=1).await;
    let vector_caps = module_cap(&caps, builtin::query_vector(), 0..=1).await;
    let agentic_caps = module_cap(&caps, builtin::query_agentic(), 0..=1).await;
    let publisher = publisher_caps.query_mailbox();
    let mut vector = vector_caps.query_inbox();
    let mut agentic = agentic_caps.query_inbox();

    publisher
        .publish(QueryRequest::new("find memories about rust"))
        .await
        .expect("query topic should have subscribers");

    let vector_env = vector
        .next_item()
        .await
        .expect("vector subscriber receives query");
    let agentic_env = agentic
        .next_item()
        .await
        .expect("agentic subscriber receives query");

    assert_eq!(vector_env.sender, agentic_env.sender);
    assert_eq!(vector_env.sender.module, ticker_id());
    assert_eq!(vector_env.body.question, "find memories about rust");
    assert_eq!(agentic_env.body.question, "find memories about rust");
}

#[tokio::test(flavor = "current_thread", start_paused = false)]
async fn self_model_mailbox_is_a_separate_typed_topic() {
    let caps = test_caps(Blackboard::default());
    let ticker_caps = module_cap(&caps, ticker_id(), 0..=1).await;
    let echo_caps = module_cap(&caps, echo_id(), 0..=1).await;
    let query_caps = module_cap(&caps, builtin::query_vector(), 0..=1).await;
    let schema_caps = module_cap(&caps, builtin::attention_schema(), 0..=1).await;
    let query_publisher = ticker_caps.query_mailbox();
    let self_publisher = echo_caps.self_model_mailbox();
    let mut query_inbox = query_caps.query_inbox();
    let mut self_inbox = schema_caps.self_model_inbox();

    query_publisher
        .publish(QueryRequest::new("memory only"))
        .await
        .expect("query subscriber exists");
    self_publisher
        .publish(SelfModelRequest::new("what are you aware of?"))
        .await
        .expect("self-model subscriber exists");

    assert_eq!(
        query_inbox.next_item().await.unwrap().body.question,
        "memory only"
    );
    assert_eq!(
        self_inbox.next_item().await.unwrap().body.question,
        "what are you aware of?"
    );
    assert!(query_inbox.take_ready_items().unwrap().items.is_empty());
    assert!(self_inbox.take_ready_items().unwrap().items.is_empty());
}

#[tokio::test(flavor = "current_thread", start_paused = false)]
async fn memory_request_mailbox_fans_out_with_owner_stamp() {
    let caps = test_caps(Blackboard::default());
    let surprise_caps = module_cap(&caps, builtin::surprise(), 0..=1).await;
    let memory_caps = module_cap(&caps, builtin::memory(), 0..=1).await;
    let publisher = surprise_caps.memory_request_mailbox();
    let mut memory = memory_caps.memory_request_inbox();

    publisher
        .publish(MemoryRequest {
            content: "remember unexpected turn".into(),
            importance: MemoryImportance::High,
            reason: "prediction diverged".into(),
        })
        .await
        .expect("memory request subscriber exists");

    let envelope = memory
        .next_item()
        .await
        .expect("memory request subscriber receives request");

    assert_eq!(envelope.sender.module, builtin::surprise());
    assert_eq!(envelope.body.content, "remember unexpected turn");
    assert_eq!(envelope.body.importance, MemoryImportance::High);
    assert_eq!(envelope.body.reason, "prediction diverged");
}

#[tokio::test(flavor = "current_thread", start_paused = false)]
async fn role_topics_load_balance_across_active_replicas() {
    let mut alloc = ResourceAllocation::default();
    alloc.set(
        builtin::query_vector(),
        ModuleConfig {
            replicas: 2,
            ..Default::default()
        },
    );
    alloc.set(
        builtin::query_agentic(),
        ModuleConfig {
            replicas: 1,
            ..Default::default()
        },
    );
    let caps = test_caps(Blackboard::with_allocation(alloc));
    let publisher_caps = module_cap(&caps, ticker_id(), 0..=1).await;
    let vector_caps = module_caps(&caps, builtin::query_vector(), 0..=3).await;
    let agentic_caps = module_caps(&caps, builtin::query_agentic(), 0..=1).await;
    let publisher = publisher_caps.query_mailbox();
    let mut vector_0 = vector_caps[0].query_inbox();
    let mut vector_1 = vector_caps[1].query_inbox();
    let mut agentic_0 = agentic_caps[0].query_inbox();

    publisher.publish(QueryRequest::new("first")).await.unwrap();
    publisher
        .publish(QueryRequest::new("second"))
        .await
        .unwrap();

    assert_eq!(vector_0.next_item().await.unwrap().body.question, "first");
    assert_eq!(vector_1.next_item().await.unwrap().body.question, "second");
    let agentic = agentic_0
        .take_ready_items()
        .unwrap()
        .items
        .into_iter()
        .map(|item| item.body.question)
        .collect::<Vec<_>>();
    assert_eq!(agentic, vec!["first", "second"]);
}

#[tokio::test(flavor = "current_thread", start_paused = false)]
async fn role_topics_do_not_route_to_disabled_replicas() {
    let mut alloc = ResourceAllocation::default();
    alloc.set(
        builtin::query_vector(),
        ModuleConfig {
            replicas: 1,
            ..Default::default()
        },
    );
    let caps = test_caps(Blackboard::with_allocation(alloc));
    let publisher_caps = module_cap(&caps, ticker_id(), 0..=1).await;
    let vector_caps = module_caps(&caps, builtin::query_vector(), 0..=2).await;
    let publisher = publisher_caps.query_mailbox();
    let mut vector_0 = vector_caps[0].query_inbox();
    let mut vector_1 = vector_caps[1].query_inbox();

    publisher
        .publish(QueryRequest::new("active only"))
        .await
        .unwrap();

    assert_eq!(
        vector_0.next_item().await.unwrap().body.question,
        "active only"
    );
    assert!(vector_1.take_ready_items().unwrap().items.is_empty());
}

#[tokio::test(flavor = "current_thread", start_paused = false)]
async fn periodic_activation_targets_active_replicas_only() {
    let mut alloc = ResourceAllocation::default();
    alloc.set(
        ticker_id(),
        ModuleConfig {
            replicas: 2,
            period: Some(Duration::from_millis(10)),
            ..Default::default()
        },
    );
    let caps = test_caps(Blackboard::with_allocation(alloc));
    let ticker_caps = module_caps(&caps, ticker_id(), 0..=3).await;
    let mut replica_0 = ticker_caps[0].periodic_inbox();
    let mut replica_1 = ticker_caps[1].periodic_inbox();
    let mut replica_2 = ticker_caps[2].periodic_inbox();
    let mut event_loop = AgentEventLoop::new(caps.periodic_activation());

    event_loop.tick(Duration::from_millis(10)).await;

    assert_eq!(replica_0.take_ready_ticks().unwrap(), 1);
    assert_eq!(replica_1.take_ready_ticks().unwrap(), 1);
    assert_eq!(replica_2.take_ready_ticks().unwrap(), 0);
}

#[tokio::test(flavor = "current_thread", start_paused = false)]
async fn activation_gate_blocks_until_replica_is_enabled() {
    let local = LocalSet::new();
    local
        .run_until(async {
            let mut alloc = ResourceAllocation::default();
            alloc.set(
                ticker_id(),
                ModuleConfig {
                    replicas: 0,
                    ..Default::default()
                },
            );
            let blackboard = Blackboard::with_allocation(alloc);
            let caps = test_caps(blackboard.clone());
            let ticker_caps = module_cap(&caps, ticker_id(), 0..=1).await;
            let gate = ticker_caps.activation_gate();
            let passed = Arc::new(Mutex::new(false));
            let passed_for_task = passed.clone();

            let handle = tokio::task::spawn_local(async move {
                gate.block().await;
                *passed_for_task.lock().await = true;
            });

            for _ in 0..3 {
                tokio::task::yield_now().await;
            }
            assert!(!*passed.lock().await);

            set_counter_allocation(&blackboard, Duration::from_millis(10), true).await;
            for _ in 0..10 {
                if *passed.lock().await {
                    break;
                }
                tokio::task::yield_now().await;
            }

            assert!(*passed.lock().await);
            handle.await.unwrap();
        })
        .await;
}

#[tokio::test(flavor = "current_thread", start_paused = false)]
async fn providers_issue_split_search_capabilities() {
    let caps = test_caps(Blackboard::default());
    let _vector = caps.vector_memory_searcher();
    let _content = caps.memory_content_reader();
    let _file = caps.file_searcher();
}

#[tokio::test(flavor = "current_thread", start_paused = false)]
async fn file_searcher_exposes_only_rg_like_controls_and_clamps_bounds() {
    let file_search = RecordingFileSearchProvider::default();
    let caps = test_caps_with_stores(
        Blackboard::default(),
        Arc::new(NoopMemoryStore),
        Vec::new(),
        Arc::new(file_search.clone()),
    );

    caps.file_searcher()
        .search(FileSearchQuery {
            pattern: "fn main".into(),
            regex: true,
            invert_match: true,
            case_sensitive: false,
            context: 100,
            max_matches: 0,
        })
        .await
        .expect("file search succeeds");

    assert_eq!(
        file_search.last_query().await,
        Some(FileSearchQuery {
            pattern: "fn main".into(),
            regex: true,
            invert_match: true,
            case_sensitive: false,
            context: 8,
            max_matches: 1,
        })
    );
}

#[tokio::test(flavor = "current_thread", start_paused = false)]
async fn memory_writer_fans_out_and_writes_single_metadata_after_primary_success() {
    let blackboard = Blackboard::default();
    let index = MemoryIndex::new("fanout");
    let primary = RecordingMemoryStore {
        generated_index: index.clone(),
        ..Default::default()
    };
    let secondary = RecordingMemoryStore {
        fail_put: true,
        ..Default::default()
    };
    let caps = test_caps_with_stores(
        blackboard.clone(),
        Arc::new(primary.clone()),
        vec![Arc::new(secondary.clone())],
        Arc::new(NoopFileSearchProvider),
    );

    let written = caps
        .memory_writer()
        .insert("replicated memory".into(), MemoryRank::LongTerm, 30)
        .await
        .expect("primary insert succeeds");

    assert_eq!(written, index);
    assert_eq!(primary.inserted_indexes().await, vec![index.clone()]);
    assert!(secondary.put_indexes().await.is_empty());
    let metadata_len = blackboard
        .read(|bb| bb.memory_metadata().contains_key(&index))
        .await;
    assert!(metadata_len);
}

#[tokio::test(flavor = "current_thread", start_paused = false)]
async fn memory_writer_fans_out_to_secondary_on_happy_path() {
    let blackboard = Blackboard::default();
    let index = MemoryIndex::new("happy-fanout");
    let primary = RecordingMemoryStore {
        generated_index: index.clone(),
        ..Default::default()
    };
    let secondary = RecordingMemoryStore::default();
    let caps = test_caps_with_stores(
        blackboard.clone(),
        Arc::new(primary.clone()),
        vec![Arc::new(secondary.clone())],
        Arc::new(NoopFileSearchProvider),
    );

    let written = caps
        .memory_writer()
        .insert("replicated memory".into(), MemoryRank::LongTerm, 30)
        .await
        .expect("primary insert succeeds");

    assert_eq!(written, index.clone());
    assert_eq!(primary.inserted_indexes().await, vec![index.clone()]);
    assert_eq!(secondary.put_indexes().await, vec![index]);
}

#[tokio::test(flavor = "current_thread", start_paused = false)]
async fn memory_writer_primary_failure_prevents_metadata_and_secondary_writes() {
    let blackboard = Blackboard::default();
    let index = MemoryIndex::new("primary-fails");
    let primary = RecordingMemoryStore {
        generated_index: index.clone(),
        fail_insert: true,
        ..Default::default()
    };
    let secondary = RecordingMemoryStore::default();
    let caps = test_caps_with_stores(
        blackboard.clone(),
        Arc::new(primary.clone()),
        vec![Arc::new(secondary.clone())],
        Arc::new(NoopFileSearchProvider),
    );

    let result = caps
        .memory_writer()
        .insert("not durable".into(), MemoryRank::ShortTerm, 5)
        .await;

    assert!(result.is_err());
    assert!(secondary.put_indexes().await.is_empty());
    let has_metadata = blackboard
        .read(|bb| bb.memory_metadata().contains_key(&index))
        .await;
    assert!(!has_metadata);
}

#[tokio::test(flavor = "current_thread", start_paused = false)]
async fn vector_memory_searcher_records_access_for_hits_only() {
    let blackboard = Blackboard::default();
    let hit_index = MemoryIndex::new("hit");
    let miss_index = MemoryIndex::new("miss");
    blackboard
        .apply(BlackboardCommand::UpsertMemoryMetadata {
            index: miss_index.clone(),
            rank_if_new: MemoryRank::MidTerm,
            decay_if_new_secs: 0,
            now: Utc::now(),
            patch: MemoryMetaPatch::default(),
        })
        .await;
    let primary = RecordingMemoryStore {
        search_hits: vec![MemoryRecord {
            index: hit_index.clone(),
            content: MemoryContent::new("matched content"),
            rank: MemoryRank::MidTerm,
        }],
        ..Default::default()
    };
    let caps = test_caps_with_stores(
        blackboard.clone(),
        Arc::new(primary),
        Vec::new(),
        Arc::new(NoopFileSearchProvider),
    );

    let hits = caps
        .vector_memory_searcher()
        .search("matched", 10, None)
        .await
        .expect("search succeeds");

    assert_eq!(hits.len(), 1);
    let (hit_access_count, miss_access_count) = blackboard
        .read(|bb| {
            (
                bb.memory_metadata()
                    .get(&hit_index)
                    .map(|meta| meta.access_count),
                bb.memory_metadata()
                    .get(&miss_index)
                    .map(|meta| meta.access_count),
            )
        })
        .await;
    assert_eq!(hit_access_count, Some(1));
    assert_eq!(miss_access_count, Some(0));
}

#[tokio::test(flavor = "current_thread", start_paused = false)]
async fn memory_compactor_uses_atomic_compaction_and_updates_metadata() {
    let blackboard = Blackboard::default();
    let source = MemoryIndex::new("source");
    let merged = MemoryIndex::new("merged");
    let primary = RecordingMemoryStore {
        generated_index: merged.clone(),
        ..Default::default()
    };
    let secondary = RecordingMemoryStore::default();
    let caps = test_caps_with_stores(
        blackboard.clone(),
        Arc::new(primary.clone()),
        vec![Arc::new(secondary.clone())],
        Arc::new(NoopFileSearchProvider),
    );

    let result = caps
        .memory_compactor()
        .merge(
            std::slice::from_ref(&source),
            "merged content".into(),
            MemoryRank::LongTerm,
            30,
        )
        .await;

    assert_eq!(result.unwrap(), merged.clone());
    assert_eq!(primary.compacted_indexes().await, vec![merged.clone()]);
    assert_eq!(primary.compacted_sources().await, vec![source.clone()]);
    assert_eq!(secondary.compacted_indexes().await, vec![merged.clone()]);
    assert_eq!(secondary.compacted_sources().await, vec![source]);
    let has_metadata = blackboard
        .read(|bb| bb.memory_metadata().contains_key(&merged))
        .await;
    assert!(has_metadata);
}

#[tokio::test(flavor = "current_thread", start_paused = false)]
async fn memory_compactor_primary_compact_failure_prevents_metadata_and_replica_writes() {
    let blackboard = Blackboard::default();
    let source = MemoryIndex::new("source");
    let merged = MemoryIndex::new("merged");
    let primary = RecordingMemoryStore {
        generated_index: merged.clone(),
        fail_compact: true,
        ..Default::default()
    };
    let secondary = RecordingMemoryStore::default();
    let caps = test_caps_with_stores(
        blackboard.clone(),
        Arc::new(primary.clone()),
        vec![Arc::new(secondary.clone())],
        Arc::new(NoopFileSearchProvider),
    );

    let result = caps
        .memory_compactor()
        .merge(&[source], "merged content".into(), MemoryRank::LongTerm, 30)
        .await;

    assert!(result.is_err());
    assert!(secondary.compacted_indexes().await.is_empty());
    let has_metadata = blackboard
        .read(|bb| bb.memory_metadata().contains_key(&merged))
        .await;
    assert!(!has_metadata);
}

#[tokio::test(flavor = "current_thread", start_paused = false)]
async fn memory_request_does_not_insert_without_llm_tool_decision() {
    let local = LocalSet::new();
    local
        .run_until(async {
            let blackboard = Blackboard::default();
            let primary = RecordingMemoryStore::default();
            let caps = test_caps_with_stores(
                blackboard,
                Arc::new(primary.clone()),
                Vec::new(),
                Arc::new(NoopFileSearchProvider),
            );
            let (modules, surprise_caps) = memory_modules_with_publisher(&caps).await;
            let publisher = surprise_caps.memory_request_mailbox();

            run(modules, async {
                publisher
                    .publish(MemoryRequest {
                        content: "candidate memory".into(),
                        importance: MemoryImportance::High,
                        reason: "test request".into(),
                    })
                    .await
                    .expect("memory module subscribed");

                for _ in 0..10 {
                    tokio::task::yield_now().await;
                }
            })
            .await
            .expect("scheduler returned err");

            assert!(primary.inserted_indexes().await.is_empty());
        })
        .await;
}

#[tokio::test(flavor = "current_thread", start_paused = false)]
async fn memory_request_batch_allows_llm_to_consolidate_to_one_insert() {
    let local = LocalSet::new();
    local
        .run_until(async {
            let blackboard = Blackboard::default();
            let primary = RecordingMemoryStore::default();
            let adapter = MockLlmAdapter::new()
                .with_text_scenario(memory_insert_tool_scenario(
                    "Ryo wants inbox batching implemented.",
                ))
                .with_text_scenario(final_text_scenario("memory-complete"));
            let caps = test_caps_with_stores_and_adapter(
                blackboard,
                Arc::new(primary.clone()),
                Vec::new(),
                Arc::new(NoopFileSearchProvider),
                adapter,
            );
            let (modules, surprise_caps) = memory_modules_with_publisher(&caps).await;
            let publisher = surprise_caps.memory_request_mailbox();

            run(modules, async {
                publisher
                    .publish(MemoryRequest {
                        content: "Ryo wants inbox batching implemented".into(),
                        importance: MemoryImportance::Normal,
                        reason: "first signal".into(),
                    })
                    .await
                    .expect("memory module subscribed");
                publisher
                    .publish(MemoryRequest {
                        content: "Inbox batching implementation is important to Ryo".into(),
                        importance: MemoryImportance::High,
                        reason: "duplicate signal".into(),
                    })
                    .await
                    .expect("memory module subscribed");

                for _ in 0..20 {
                    if primary.inserted_indexes().await.len() == 1 {
                        break;
                    }
                    tokio::task::yield_now().await;
                }
            })
            .await
            .expect("scheduler returned err");

            let indexes = primary.inserted_indexes().await;
            assert_eq!(indexes.len(), 1);
        })
        .await;
}

#[tokio::test(flavor = "current_thread", start_paused = false)]
async fn predict_and_surprise_constructor_shapes_match_capabilities() {
    let caps = test_caps(Blackboard::default());
    let predict_caps = module_cap(&caps, builtin::predict(), 0..=1).await;
    let surprise_caps = module_cap(&caps, builtin::surprise(), 0..=1).await;

    let _predict = PredictModule::new(
        predict_caps.attention_stream_updated_inbox(),
        predict_caps.periodic_inbox(),
        predict_caps.activation_gate(),
        predict_caps.attention_reader(),
        predict_caps.blackboard_reader(),
        predict_caps.memo(),
        predict_caps.llm_access(),
    );

    let _surprise = SurpriseModule::new(
        surprise_caps.attention_stream_updated_inbox(),
        surprise_caps.activation_gate(),
        surprise_caps.attention_reader(),
        surprise_caps.blackboard_reader(),
        surprise_caps.memory_request_mailbox(),
        surprise_caps.memo(),
        surprise_caps.llm_access(),
    );
}

async fn periodic_counter_setup(
    period: Duration,
    enabled: bool,
) -> (
    Blackboard,
    AllocatedModules,
    Arc<Mutex<u32>>,
    AgentEventLoop,
) {
    let mut alloc = ResourceAllocation::default();
    alloc.set(
        ticker_id(),
        ModuleConfig {
            replicas: u8::from(enabled),
            tier: ModelTier::Default,
            period: Some(period),
            ..Default::default()
        },
    );

    let blackboard = Blackboard::with_allocation(alloc);
    let caps = test_caps(blackboard.clone());
    let event_loop = AgentEventLoop::new(caps.periodic_activation());
    let count = Arc::new(Mutex::new(0));
    let modules = ModuleRegistry::new()
        .register(ticker_id(), 0..=1, {
            let count = count.clone();
            move |caps| PeriodicCounter {
                periodic: caps.periodic_inbox(),
                count: count.clone(),
            }
        })
        .unwrap()
        .build(&caps)
        .await
        .unwrap();

    (blackboard, modules, count, event_loop)
}

async fn set_counter_allocation(blackboard: &Blackboard, period: Duration, enabled: bool) {
    let mut alloc = ResourceAllocation::default();
    alloc.set(
        ticker_id(),
        ModuleConfig {
            replicas: u8::from(enabled),
            tier: ModelTier::Default,
            period: Some(period),
            ..Default::default()
        },
    );
    blackboard
        .apply(BlackboardCommand::SetAllocation(alloc))
        .await;
}

// ---------- noop port ----------

struct NoopAttentionRepo;

#[async_trait(?Send)]
impl AttentionRepository for NoopAttentionRepo {
    async fn append(
        &self,
        _stream: ModuleInstanceId,
        _event: AttentionStreamEvent,
    ) -> Result<(), PortError> {
        Ok(())
    }
    async fn since(
        &self,
        _stream: &ModuleInstanceId,
        _from: DateTime<Utc>,
    ) -> Result<Vec<AttentionStreamEvent>, PortError> {
        Ok(vec![])
    }
}

struct NoopMemoryStore;

#[async_trait(?Send)]
impl MemoryStore for NoopMemoryStore {
    async fn insert(&self, _mem: NewMemory) -> Result<MemoryIndex, PortError> {
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let id = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Ok(MemoryIndex::new(format!("noop-memory-{id}")))
    }

    async fn put(&self, _mem: IndexedMemory) -> Result<(), PortError> {
        Ok(())
    }

    async fn compact(
        &self,
        _mem: NewMemory,
        _sources: &[MemoryIndex],
    ) -> Result<MemoryIndex, PortError> {
        self.insert(_mem).await
    }

    async fn put_compacted(
        &self,
        _mem: IndexedMemory,
        _sources: &[MemoryIndex],
    ) -> Result<(), PortError> {
        Ok(())
    }

    async fn get(&self, _index: &MemoryIndex) -> Result<Option<MemoryRecord>, PortError> {
        Ok(None)
    }

    async fn search(&self, _q: &MemoryQuery) -> Result<Vec<MemoryRecord>, PortError> {
        Ok(Vec::new())
    }

    async fn delete(&self, _index: &MemoryIndex) -> Result<(), PortError> {
        Ok(())
    }
}

#[derive(Clone)]
struct RecordingMemoryStore {
    state: Arc<Mutex<RecordingMemoryStoreState>>,
    generated_index: MemoryIndex,
    next_index: Arc<std::sync::atomic::AtomicU64>,
    search_hits: Vec<MemoryRecord>,
    fail_insert: bool,
    fail_put: bool,
    fail_compact: bool,
}

impl Default for RecordingMemoryStore {
    fn default() -> Self {
        Self {
            state: Arc::new(Mutex::new(RecordingMemoryStoreState::default())),
            generated_index: MemoryIndex::new("generated"),
            next_index: Arc::new(std::sync::atomic::AtomicU64::new(0)),
            search_hits: Vec::new(),
            fail_insert: false,
            fail_put: false,
            fail_compact: false,
        }
    }
}

#[derive(Default)]
struct RecordingMemoryStoreState {
    inserted_indexes: Vec<MemoryIndex>,
    put_indexes: Vec<MemoryIndex>,
    compacted_indexes: Vec<MemoryIndex>,
    compacted_sources: Vec<MemoryIndex>,
}

impl RecordingMemoryStore {
    async fn inserted_indexes(&self) -> Vec<MemoryIndex> {
        self.state.lock().await.inserted_indexes.clone()
    }

    async fn put_indexes(&self) -> Vec<MemoryIndex> {
        self.state.lock().await.put_indexes.clone()
    }

    async fn compacted_indexes(&self) -> Vec<MemoryIndex> {
        self.state.lock().await.compacted_indexes.clone()
    }

    async fn compacted_sources(&self) -> Vec<MemoryIndex> {
        self.state.lock().await.compacted_sources.clone()
    }

    fn next_generated_index(&self) -> MemoryIndex {
        if self.generated_index.as_str() != "generated" {
            return self.generated_index.clone();
        }
        let id = self
            .next_index
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        MemoryIndex::new(format!("generated-{id}"))
    }
}

#[async_trait(?Send)]
impl MemoryStore for RecordingMemoryStore {
    async fn insert(&self, _mem: NewMemory) -> Result<MemoryIndex, PortError> {
        if self.fail_insert {
            return Err(PortError::Backend("insert failed".into()));
        }
        let index = self.next_generated_index();
        self.state.lock().await.inserted_indexes.push(index.clone());
        Ok(index)
    }

    async fn put(&self, mem: IndexedMemory) -> Result<(), PortError> {
        if self.fail_put {
            return Err(PortError::Backend("put failed".into()));
        }
        self.state.lock().await.put_indexes.push(mem.index);
        Ok(())
    }

    async fn compact(
        &self,
        _mem: NewMemory,
        sources: &[MemoryIndex],
    ) -> Result<MemoryIndex, PortError> {
        if self.fail_compact {
            return Err(PortError::Backend("compact failed".into()));
        }
        let index = self.next_generated_index();
        let mut state = self.state.lock().await;
        state.compacted_indexes.push(index.clone());
        state.compacted_sources.extend_from_slice(sources);
        Ok(index)
    }

    async fn put_compacted(
        &self,
        mem: IndexedMemory,
        sources: &[MemoryIndex],
    ) -> Result<(), PortError> {
        if self.fail_put {
            return Err(PortError::Backend("put compacted failed".into()));
        }
        let mut state = self.state.lock().await;
        state.compacted_indexes.push(mem.index);
        state.compacted_sources.extend_from_slice(sources);
        Ok(())
    }

    async fn get(&self, index: &MemoryIndex) -> Result<Option<MemoryRecord>, PortError> {
        Ok(self
            .search_hits
            .iter()
            .find(|record| &record.index == index)
            .cloned())
    }

    async fn search(&self, _q: &MemoryQuery) -> Result<Vec<MemoryRecord>, PortError> {
        Ok(self.search_hits.clone())
    }

    async fn delete(&self, _index: &MemoryIndex) -> Result<(), PortError> {
        Ok(())
    }
}

struct NoopUtteranceSink;

#[async_trait(?Send)]
impl UtteranceSink for NoopUtteranceSink {
    async fn on_complete(&self, _: Utterance) -> Result<(), PortError> {
        Ok(())
    }
}

struct NoopFileSearchProvider;

#[async_trait(?Send)]
impl FileSearchProvider for NoopFileSearchProvider {
    async fn search(&self, _query: &FileSearchQuery) -> Result<Vec<FileSearchHit>, PortError> {
        Ok(Vec::new())
    }
}

#[derive(Clone, Default)]
struct RecordingFileSearchProvider {
    last_query: Arc<Mutex<Option<FileSearchQuery>>>,
}

impl RecordingFileSearchProvider {
    async fn last_query(&self) -> Option<FileSearchQuery> {
        self.last_query.lock().await.clone()
    }
}

#[async_trait(?Send)]
impl FileSearchProvider for RecordingFileSearchProvider {
    async fn search(&self, query: &FileSearchQuery) -> Result<Vec<FileSearchHit>, PortError> {
        *self.last_query.lock().await = Some(query.clone());
        Ok(Vec::new())
    }
}

fn memory_insert_tool_scenario(content: &str) -> MockTextScenario {
    MockTextScenario::events(vec![
        Ok(RawTextTurnEvent::Started {
            request_id: Some("memory-tool".into()),
            model: "mock".into(),
        }),
        Ok(RawTextTurnEvent::ToolCallChunk {
            id: "insert-1".into(),
            name: "insert_memory".into(),
            arguments_json_delta: format!(
                r#"{{"content":"{content}","rank":"ShortTerm","decay_secs":86400}}"#
            ),
        }),
        Ok(RawTextTurnEvent::Completed {
            request_id: Some("memory-tool".into()),
            finish_reason: FinishReason::ToolCall,
            usage: Usage::zero(),
        }),
    ])
}

fn final_text_scenario(text: &str) -> MockTextScenario {
    MockTextScenario::events(vec![
        Ok(RawTextTurnEvent::Started {
            request_id: Some("memory-final".into()),
            model: "mock".into(),
        }),
        Ok(RawTextTurnEvent::TextDelta { delta: text.into() }),
        Ok(RawTextTurnEvent::Completed {
            request_id: Some("memory-final".into()),
            finish_reason: FinishReason::Stop,
            usage: Usage::zero(),
        }),
    ])
}

fn test_caps(blackboard: Blackboard) -> CapabilityProviders {
    test_caps_with_stores(
        blackboard,
        Arc::new(NoopMemoryStore),
        Vec::new(),
        Arc::new(NoopFileSearchProvider),
    )
}

fn test_caps_with_stores(
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
    )
}

fn test_caps_with_stores_and_adapter(
    blackboard: Blackboard,
    primary_memory_store: Arc<dyn MemoryStore>,
    memory_replicas: Vec<Arc<dyn MemoryStore>>,
    file_search: Arc<dyn FileSearchProvider>,
    adapter: MockLlmAdapter,
) -> CapabilityProviders {
    let adapter = Arc::new(adapter);
    let budget = SharedPoolBudgetManager::new(SharedPoolBudgetOptions::default());
    let lutum = Lutum::new(adapter, budget);
    CapabilityProviders::new(
        blackboard,
        Arc::new(NoopAttentionRepo),
        primary_memory_store,
        memory_replicas,
        file_search,
        Arc::new(NoopUtteranceSink),
        Arc::new(SystemClock),
        LutumTiers {
            cheap: lutum.clone(),
            default: lutum.clone(),
            premium: lutum,
        },
    )
}
