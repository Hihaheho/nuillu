use std::sync::mpsc::{Receiver, Sender};

use nuillu_visualizer_protocol::{
    VisualizerAction, VisualizerClientMessage, VisualizerEvent, VisualizerServerMessage,
};

#[derive(Clone, Debug)]
pub struct VisualizerEventSink {
    events: Sender<VisualizerServerMessage>,
}

impl VisualizerEventSink {
    pub fn new(events: Sender<VisualizerServerMessage>) -> Self {
        Self { events }
    }

    pub fn send(&self, event: VisualizerEvent) {
        let _ = self.events.send(VisualizerServerMessage::event(event));
    }
}

pub struct VisualizerHook {
    events: Sender<VisualizerServerMessage>,
    commands: Receiver<VisualizerClientMessage>,
    shutdown_requested: bool,
}

impl VisualizerHook {
    pub fn new(
        events: Sender<VisualizerServerMessage>,
        commands: Receiver<VisualizerClientMessage>,
    ) -> Self {
        Self {
            events,
            commands,
            shutdown_requested: false,
        }
    }

    pub fn event_sender(&self) -> VisualizerEventSink {
        VisualizerEventSink::new(self.events.clone())
    }

    pub fn send_event(&self, event: VisualizerEvent) {
        let _ = self.events.send(VisualizerServerMessage::event(event));
    }

    pub fn offer_action(&self, action: VisualizerAction) {
        let _ = self
            .events
            .send(VisualizerServerMessage::OfferAction { action });
    }

    pub fn revoke_action(&self, action_id: String) {
        let _ = self
            .events
            .send(VisualizerServerMessage::RevokeAction { action_id });
    }

    pub fn request_shutdown(&mut self) {
        self.shutdown_requested = true;
    }

    pub fn shutdown_requested(&self) -> bool {
        self.shutdown_requested
    }

    pub fn try_recv_command(&mut self) -> Option<VisualizerClientMessage> {
        match self.commands.try_recv() {
            Ok(message) => Some(message),
            Err(std::sync::mpsc::TryRecvError::Empty) => None,
            Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                self.request_shutdown();
                None
            }
        }
    }
}
