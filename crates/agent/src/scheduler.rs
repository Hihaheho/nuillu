use std::future::Future;
use std::pin::pin;

use futures::stream::{FuturesUnordered, StreamExt};
use nuillu_module::AllocatedModules;
use thiserror::Error;
use tokio::task::{JoinHandle, spawn_local};

#[derive(Debug, Error)]
pub enum SchedulerError {
    // No variants for now. Reserved for future fatal-policy decisions.
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

    for mut module in modules.into_modules() {
        handles.push(spawn_local(async move {
            module.run().await;
        }));
    }

    let mut shutdown = pin!(shutdown);

    loop {
        tokio::select! {
            biased;
            _ = shutdown.as_mut() => return Ok(()),
            joined = handles.next() => {
                match joined {
                    Some(Ok(())) => {}
                    Some(Err(e)) => {
                        if !e.is_cancelled() {
                            tracing::warn!(error = ?e, "module task panicked");
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
