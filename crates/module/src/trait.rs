use async_trait::async_trait;

/// A cognitive or functional module.
///
/// Capabilities (including typed inboxes that yield activations) are owned
/// by the module struct, passed in at construction. `run` is the module's
/// *main loop* — it should repeatedly pull activations from its inbox
/// capabilities and act on them via its other capabilities. Returning
/// from `run` ends the module's lifetime.
///
/// `?Send` is intentional: the runtime is single-threaded
/// (`spawn_local` / `LocalSet`) and capabilities may capture non-`Send`
/// state (e.g. `Rc`-bearing `lutum::Session`).
#[async_trait(?Send)]
pub trait Module {
    async fn run(&mut self);
}
