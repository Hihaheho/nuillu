//! Kick channels for dependency-aware module activation.
//!
//! Kicks are scheduler-internal. When a module has collected a batch, the scheduler kicks its
//! active dependencies and waits for them to finish or noop before activating the dependent.

use futures::StreamExt as _;
use futures::channel::mpsc;
use nuillu_types::ModuleInstanceId;
use tokio::sync::oneshot;

/// One pending kick awaiting completion notification.
pub(crate) struct Kick {
    #[allow(dead_code)]
    pub(crate) sender: ModuleInstanceId,
    pub(crate) completion: oneshot::Sender<()>,
}

impl Kick {
    pub(crate) fn notify_finish(self) {
        let _ = self.completion.send(());
    }
}

/// Sender side held by the scheduler to deliver kicks to one module instance.
#[derive(Clone)]
pub(crate) struct KickHandle {
    sender: mpsc::UnboundedSender<Kick>,
}

impl KickHandle {
    /// Send a kick to the target. The returned receiver resolves when the target completes or
    /// noops the kick. If the target is gone, the receiver resolves with `Err`.
    pub(crate) fn send(&self, sender: ModuleInstanceId) -> oneshot::Receiver<()> {
        let (completion, receiver) = oneshot::channel();
        let _ = self.sender.unbounded_send(Kick { sender, completion });
        receiver
    }
}

/// Receiver side held by the scheduler for one module instance.
pub(crate) struct KickInbox {
    receiver: mpsc::UnboundedReceiver<Kick>,
}

impl KickInbox {
    pub(crate) fn new() -> (Self, KickHandle) {
        let (sender, receiver) = mpsc::unbounded();
        (Self { receiver }, KickHandle { sender })
    }

    pub(crate) async fn next(&mut self) -> Option<Kick> {
        self.receiver.next().await
    }

    pub(crate) fn drain_ready(&mut self) -> Vec<Kick> {
        let mut out = Vec::new();
        while let Ok(kick) = self.receiver.try_recv() {
            out.push(kick);
        }
        out
    }
}
