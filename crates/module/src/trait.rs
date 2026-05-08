use std::any::Any;

use anyhow::{Result, anyhow};
use async_trait::async_trait;

/// A cognitive or functional module.
///
/// Capabilities (including typed inboxes that yield activations) are owned
/// by the module struct, passed in at construction. Modules do not own
/// their persistent run loop: the agent event loop awaits `next_batch`,
/// then invokes `activate` with a shared reference to the returned batch.
///
/// `?Send` is intentional: the runtime is single-threaded
/// (`spawn_local` / `LocalSet`) and capabilities may capture non-`Send`
/// state (e.g. `Rc`-bearing `lutum::Session`).
#[async_trait(?Send)]
pub trait Module {
    type Batch: 'static;

    async fn next_batch(&mut self) -> Result<Self::Batch>;
    async fn activate(&mut self, batch: &Self::Batch) -> Result<()>;
}

pub struct ModuleBatch {
    inner: Box<dyn Any>,
}

impl ModuleBatch {
    fn new(inner: Box<dyn Any>) -> Self {
        Self { inner }
    }

    fn as_any(&self) -> &dyn Any {
        self.inner.as_ref()
    }
}

#[async_trait(?Send)]
pub(crate) trait ErasedModule {
    async fn next_batch(&mut self) -> Result<ModuleBatch>;
    async fn activate(&mut self, batch: &ModuleBatch) -> Result<()>;
}

#[async_trait(?Send)]
impl<M> ErasedModule for M
where
    M: Module + 'static,
{
    async fn next_batch(&mut self) -> Result<ModuleBatch> {
        Module::next_batch(self)
            .await
            .map(|batch| ModuleBatch::new(Box::new(batch)))
    }

    async fn activate(&mut self, batch: &ModuleBatch) -> Result<()> {
        let batch = batch
            .as_any()
            .downcast_ref::<M::Batch>()
            .ok_or_else(|| anyhow!("module batch type mismatch"))?;
        Module::activate(self, batch).await
    }
}
