use std::future::Future;
use std::pin::pin;

use futures::stream::{FuturesUnordered, StreamExt};
use nuillu_module::AllocatedModules;
use thiserror::Error;
use tokio::task::{JoinHandle, spawn_local};
use tracing::{Instrument as _, instrument::WithSubscriber as _};

#[derive(Debug, Error)]
pub enum SchedulerError {
    #[error("module task failed: {message}")]
    ModuleTaskFailed { message: String },
}

/// Run the agent event loop until `shutdown` resolves.
///
/// Each module is spawned as a persistent task (`spawn_local`) that runs
/// `Module::run` to completion. Modules drive their own loops via typed
/// inbox capabilities; the scheduler does not interpret module state and
/// does not handle triggers itself.
///
/// Awaited inside `tokio::task::LocalSet::run_until` (or `spawn_local`
/// on a `LocalSet`) so spawned tasks have an executor.
pub async fn run(
    modules: AllocatedModules,
    shutdown: impl Future<Output = ()>,
) -> Result<(), SchedulerError> {
    let mut handles: FuturesUnordered<JoinHandle<()>> = FuturesUnordered::new();
    let subscriber = tracing::dispatcher::get_default(Clone::clone);
    let parent = tracing::Span::current();

    for mut module in modules.into_modules() {
        handles.push(spawn_local(
            async move {
                module.run().await;
            }
            .instrument(parent.clone())
            .with_subscriber(subscriber.clone()),
        ));
    }

    let mut shutdown = pin!(shutdown);

    loop {
        tokio::select! {
            biased;
            _ = shutdown.as_mut() => {
                for handle in handles.iter() {
                    handle.abort();
                }
                while handles.next().await.is_some() {}
                return Ok(());
            },
            joined = handles.next() => {
                match joined {
                    Some(Ok(())) => {}
                    Some(Err(e)) => {
                        if !e.is_cancelled() {
                            let message = e.to_string();
                            tracing::error!(error = ?e, "module task failed");
                            return Err(SchedulerError::ModuleTaskFailed { message });
                        }
                    }
                    None => {
                        // No tasks left. Wait for shutdown.
                        shutdown.as_mut().await;
                        return Ok(());
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::rc::Rc;
    use std::sync::Arc;
    use std::time::Duration;

    use async_trait::async_trait;
    use nuillu_blackboard::{Blackboard, ModuleConfig, ResourceAllocation};
    use nuillu_module::{
        AttentionWriter, Memo, Module, ModuleRegistry, PeriodicInbox, QueryInbox, QueryMailbox,
        QueryRequest,
    };
    use nuillu_types::{ModelTier, ModuleId, builtin};
    use tokio::sync::{Mutex, oneshot};
    use tokio::task::LocalSet;

    use crate::AgentEventLoop;
    use crate::testing::test_caps;

    fn ticker_id() -> ModuleId {
        ModuleId::new("ticker").unwrap()
    }

    fn echo_id() -> ModuleId {
        ModuleId::new("echo").unwrap()
    }

    struct TickerModule {
        periodic: PeriodicInbox,
        memo: Memo,
        query_mailbox: QueryMailbox,
        target_ticks: u32,
        on_done: Option<oneshot::Sender<()>>,
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

    #[async_trait(?Send)]
    impl Module for EchoModule {
        async fn run(&mut self) {
            while let Ok(env) = self.query_inbox.next_item().await {
                self.memo
                    .write(format!("echoed {}", env.body.question))
                    .await;
            }
        }
    }

    struct SummarizeStub {
        periodic: PeriodicInbox,
        writer: AttentionWriter,
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
                            TickerModule {
                                periodic: caps.periodic_inbox(),
                                memo: caps.memo(),
                                query_mailbox: caps.query_mailbox(),
                                target_ticks: 3,
                                on_done: Some(done_tx),
                            }
                        }
                    })
                    .unwrap()
                    .register(echo_id(), 0..=1, |caps| EchoModule {
                        query_inbox: caps.query_inbox(),
                        memo: caps.memo(),
                    })
                    .unwrap()
                    .build(&caps)
                    .await
                    .unwrap();

                super::run(modules, async move {
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

                super::run(modules, async move {
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
}
