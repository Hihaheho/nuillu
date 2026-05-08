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

    use async_trait::async_trait;
    use nuillu_blackboard::{Blackboard, ModuleConfig, ResourceAllocation};
    use nuillu_module::{AttentionWriter, Memo, Module, ModuleRegistry, QueryInbox, QueryRequest};
    use nuillu_types::{ModelTier, ModuleId, builtin};
    use tokio::sync::oneshot;
    use tokio::task::LocalSet;

    use crate::testing::test_caps;

    fn echo_id() -> ModuleId {
        ModuleId::new("echo").unwrap()
    }

    struct EchoModule {
        query_inbox: QueryInbox,
        memo: Memo,
        on_done: Option<oneshot::Sender<()>>,
    }

    #[async_trait(?Send)]
    impl Module for EchoModule {
        async fn run(&mut self) {
            while let Ok(env) = self.query_inbox.next_item().await {
                self.memo
                    .write(format!("echoed {}", env.body.question))
                    .await;
                if let Some(tx) = self.on_done.take() {
                    let _ = tx.send(());
                }
            }
        }
    }

    struct SummarizeStub {
        writer: AttentionWriter,
        on_done: Option<oneshot::Sender<()>>,
    }

    #[async_trait(?Send)]
    impl Module for SummarizeStub {
        async fn run(&mut self) {
            self.writer.append("novel-event").await;
            if let Some(tx) = self.on_done.take() {
                let _ = tx.send(());
            }
        }
    }

    #[tokio::test(flavor = "current_thread", start_paused = false)]
    async fn typed_query_fanout_reaches_active_module() {
        let local = LocalSet::new();
        local
            .run_until(async {
                let mut alloc = ResourceAllocation::default();
                alloc.set(
                    echo_id(),
                    ModuleConfig {
                        activation_ratio: nuillu_blackboard::ActivationRatio::ONE,
                        tier: ModelTier::Default,
                        ..Default::default()
                    },
                );

                let blackboard = Blackboard::with_allocation(alloc);
                let caps = test_caps(blackboard.clone());
                let (done_tx, done_rx) = oneshot::channel();
                let done_tx = Rc::new(RefCell::new(Some(done_tx)));
                let modules = ModuleRegistry::new()
                    .register(echo_id(), 0..=1, {
                        let done_tx = Rc::clone(&done_tx);
                        move |caps| EchoModule {
                            query_inbox: caps.query_inbox(),
                            memo: caps.memo(),
                            on_done: done_tx.borrow_mut().take(),
                        }
                    })
                    .unwrap()
                    .build(&caps)
                    .await
                    .unwrap();
                let mailbox = caps.internal_harness_io().query_mailbox();

                super::run(modules, async move {
                    mailbox
                        .publish(QueryRequest::new("ping"))
                        .await
                        .expect("query should route to echo");
                    let _ = done_rx.await;
                    for _ in 0..4 {
                        tokio::task::yield_now().await;
                    }
                })
                .await
                .expect("scheduler returned err");

                blackboard
                    .read(|bb| {
                        let echo_memo = bb.memo(&echo_id()).expect("echo memo missing");
                        assert_eq!(echo_memo, "echoed ping");
                        assert_eq!(
                            bb.attention_stream().len(),
                            0,
                            "echo does not hold AttentionWriter",
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
                        activation_ratio: nuillu_blackboard::ActivationRatio::ONE,
                        tier: ModelTier::Default,
                        ..Default::default()
                    },
                );

                let blackboard = Blackboard::with_allocation(alloc);
                let caps = test_caps(blackboard.clone());
                let (done_tx, done_rx) = oneshot::channel();
                let done_tx = Rc::new(RefCell::new(Some(done_tx)));
                let modules = ModuleRegistry::new()
                    .register(builtin::summarize(), 0..=1, {
                        let done_tx = Rc::clone(&done_tx);
                        move |caps| SummarizeStub {
                            writer: caps.attention_writer(),
                            on_done: done_tx.borrow_mut().take(),
                        }
                    })
                    .unwrap()
                    .build(&caps)
                    .await
                    .unwrap();

                super::run(modules, async move {
                    let _ = done_rx.await;
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
