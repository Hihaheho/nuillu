use std::{
    sync::mpsc::{Receiver, Sender},
    time::Duration,
};

use futures::{StreamExt as _, channel::mpsc as async_mpsc};
use nuillu_module::ports::{Timer, TokioTimer, timeout as timer_timeout};
use nuillu_visualizer_protocol::{
    VisualizerAction, VisualizerClientMessage, VisualizerEvent, VisualizerServerMessage,
};

/// Non-blocking batch access to queued visualizer server messages.
///
/// This is implemented for both channel kinds accepted by [`VisualizerHook`], so native and
/// wasm hosts can forward one `Vec<VisualizerServerMessage>` per bridge call instead of crossing
/// the host boundary once per message.
pub trait VisualizerServerMessageReceiverExt {
    /// Removes at most `max` currently queued messages, preserving their send order.
    fn drain(&mut self, max: usize) -> Vec<VisualizerServerMessage>;
}

impl VisualizerServerMessageReceiverExt for Receiver<VisualizerServerMessage> {
    fn drain(&mut self, max: usize) -> Vec<VisualizerServerMessage> {
        self.try_iter().take(max).collect()
    }
}

impl VisualizerServerMessageReceiverExt for async_mpsc::UnboundedReceiver<VisualizerServerMessage> {
    fn drain(&mut self, max: usize) -> Vec<VisualizerServerMessage> {
        let mut messages = Vec::with_capacity(max.min(64));
        while messages.len() < max {
            match self.try_recv() {
                Ok(message) => messages.push(message),
                Err(async_mpsc::TryRecvError::Empty | async_mpsc::TryRecvError::Closed) => break,
            }
        }
        messages
    }
}

#[derive(Clone, Debug)]
pub struct VisualizerEventSink {
    events: EventSender,
}

impl VisualizerEventSink {
    pub fn new(events: Sender<VisualizerServerMessage>) -> Self {
        Self {
            events: EventSender::Sync(events),
        }
    }

    pub fn new_async(events: async_mpsc::UnboundedSender<VisualizerServerMessage>) -> Self {
        Self {
            events: EventSender::Async(events),
        }
    }

    pub fn send(&self, event: VisualizerEvent) {
        self.events.send(VisualizerServerMessage::event(event));
    }
}

pub struct VisualizerHook {
    events: EventSender,
    commands: CommandReceiver,
    shutdown_requested: bool,
}

#[derive(Clone, Debug)]
enum EventSender {
    Sync(Sender<VisualizerServerMessage>),
    Async(async_mpsc::UnboundedSender<VisualizerServerMessage>),
}

impl EventSender {
    fn send(&self, message: VisualizerServerMessage) {
        match self {
            Self::Sync(sender) => {
                let _ = sender.send(message);
            }
            Self::Async(sender) => {
                let _ = sender.unbounded_send(message);
            }
        }
    }
}

enum CommandReceiver {
    Sync(Receiver<VisualizerClientMessage>),
    Async(async_mpsc::UnboundedReceiver<VisualizerClientMessage>),
}

impl VisualizerHook {
    pub fn new(
        events: Sender<VisualizerServerMessage>,
        commands: Receiver<VisualizerClientMessage>,
    ) -> Self {
        Self {
            events: EventSender::Sync(events),
            commands: CommandReceiver::Sync(commands),
            shutdown_requested: false,
        }
    }

    /// Constructs a hook from caller-owned async channels, suitable for a wasm local executor.
    pub fn new_async(
        events: async_mpsc::UnboundedSender<VisualizerServerMessage>,
        commands: async_mpsc::UnboundedReceiver<VisualizerClientMessage>,
    ) -> Self {
        Self {
            events: EventSender::Async(events),
            commands: CommandReceiver::Async(commands),
            shutdown_requested: false,
        }
    }

    pub fn event_sender(&self) -> VisualizerEventSink {
        VisualizerEventSink {
            events: self.events.clone(),
        }
    }

    pub fn send_event(&self, event: VisualizerEvent) {
        self.events.send(VisualizerServerMessage::event(event));
    }

    pub fn send_server_message(&self, message: VisualizerServerMessage) {
        self.events.send(message);
    }

    pub fn offer_action(&self, action: VisualizerAction) {
        self.events
            .send(VisualizerServerMessage::OfferAction { action });
    }

    pub fn revoke_action(&self, action_id: String) {
        self.events
            .send(VisualizerServerMessage::RevokeAction { action_id });
    }

    pub fn request_shutdown(&mut self) {
        self.shutdown_requested = true;
    }

    pub fn shutdown_requested(&self) -> bool {
        self.shutdown_requested
    }

    pub fn try_recv_command(&mut self) -> Option<VisualizerClientMessage> {
        match &mut self.commands {
            CommandReceiver::Sync(receiver) => match receiver.try_recv() {
                Ok(message) => Some(message),
                Err(std::sync::mpsc::TryRecvError::Empty) => None,
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    self.shutdown_requested = true;
                    None
                }
            },
            CommandReceiver::Async(receiver) => match receiver.try_recv() {
                Ok(message) => Some(message),
                Err(async_mpsc::TryRecvError::Closed) => {
                    self.shutdown_requested = true;
                    None
                }
                Err(async_mpsc::TryRecvError::Empty) => None,
            },
        }
    }

    /// Waits for an async-channel command without polling, up to `timeout`.
    ///
    /// The legacy synchronous channel keeps its non-blocking behavior and waits only for the
    /// timeout before returning control to the server snapshot loop.
    pub async fn recv_command_timeout(
        &mut self,
        timeout: Duration,
    ) -> Option<VisualizerClientMessage> {
        self.recv_command_timeout_with_timer(&TokioTimer::new(), timeout)
            .await
    }

    /// Waits for a command using the host-provided timer.
    pub async fn recv_command_timeout_with_timer(
        &mut self,
        timer: &dyn Timer,
        timeout: Duration,
    ) -> Option<VisualizerClientMessage> {
        match &mut self.commands {
            CommandReceiver::Sync(receiver) => match receiver.try_recv() {
                Ok(message) => Some(message),
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    self.shutdown_requested = true;
                    None
                }
                Err(std::sync::mpsc::TryRecvError::Empty) => {
                    timer.sleep(timeout).await;
                    self.try_recv_command()
                }
            },
            CommandReceiver::Async(receiver) => {
                match timer_timeout(timer, timeout, receiver.next()).await {
                    Ok(Some(message)) => Some(message),
                    Ok(None) => {
                        self.shutdown_requested = true;
                        None
                    }
                    Err(_) => None,
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test(flavor = "current_thread")]
    async fn async_hook_receives_commands_and_emits_events() {
        let (event_tx, mut event_rx) = async_mpsc::unbounded();
        let (command_tx, command_rx) = async_mpsc::unbounded();
        let mut hook = VisualizerHook::new_async(event_tx, command_rx);
        let command = VisualizerClientMessage::hello();

        command_tx.unbounded_send(command.clone()).unwrap();
        hook.send_event(VisualizerEvent::Log {
            tab_id: nuillu_visualizer_protocol::VisualizerTabId::new("server"),
            message: "ready".to_owned(),
        });

        assert!(matches!(
            hook.recv_command_timeout(Duration::from_secs(1)).await,
            Some(VisualizerClientMessage::Hello { .. })
        ));
        assert!(event_rx.next().await.is_some());
    }

    #[test]
    fn sync_receiver_drains_up_to_max_in_send_order() {
        let (event_tx, mut event_rx) = std::sync::mpsc::channel();
        event_tx.send(VisualizerServerMessage::hello()).unwrap();
        event_tx.send(VisualizerServerMessage::hello()).unwrap();
        event_tx.send(VisualizerServerMessage::hello()).unwrap();

        assert_eq!(event_rx.drain(2).len(), 2);
        assert_eq!(event_rx.drain(2).len(), 1);
        assert!(event_rx.drain(2).is_empty());
    }

    #[test]
    fn async_receiver_drains_up_to_max_in_send_order() {
        let (event_tx, mut event_rx) = async_mpsc::unbounded();
        event_tx
            .unbounded_send(VisualizerServerMessage::hello())
            .unwrap();
        event_tx
            .unbounded_send(VisualizerServerMessage::hello())
            .unwrap();
        event_tx
            .unbounded_send(VisualizerServerMessage::hello())
            .unwrap();

        assert_eq!(event_rx.drain(2).len(), 2);
        assert_eq!(event_rx.drain(2).len(), 1);
        assert!(event_rx.drain(2).is_empty());
    }
}
