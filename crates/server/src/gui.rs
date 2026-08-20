use std::{
    sync::mpsc::{Receiver, Sender},
    time::Duration,
};

use futures::{StreamExt as _, channel::mpsc as async_mpsc};
use nuillu_visualizer_protocol::{
    VisualizerAction, VisualizerClientMessage, VisualizerEvent, VisualizerServerMessage,
};

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
        match &mut self.commands {
            CommandReceiver::Sync(receiver) => match receiver.try_recv() {
                Ok(message) => Some(message),
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    self.shutdown_requested = true;
                    None
                }
                Err(std::sync::mpsc::TryRecvError::Empty) => {
                    tokio::time::sleep(timeout).await;
                    self.try_recv_command()
                }
            },
            CommandReceiver::Async(receiver) => {
                match tokio::time::timeout(timeout, receiver.next()).await {
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
}
