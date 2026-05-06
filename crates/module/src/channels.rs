use std::sync::{Arc, Mutex};

use chrono::{DateTime, Utc};
use futures::StreamExt;
use futures::channel::mpsc;
use nuillu_types::ModuleId;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// Owner-stamped message delivered over a typed topic.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct Envelope<T> {
    pub sender: ModuleId,
    pub body: T,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum TopicRecvError {
    #[error("topic inbox closed")]
    Closed,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReadyItems<T> {
    pub items: Vec<Envelope<T>>,
}

#[derive(Clone)]
pub(crate) struct Topic<T: Clone> {
    subscribers: Arc<Mutex<Vec<mpsc::UnboundedSender<Envelope<T>>>>>,
}

impl<T: Clone> Topic<T> {
    pub(crate) fn new() -> Self {
        Self {
            subscribers: Arc::new(Mutex::new(Vec::new())),
        }
    }

    fn subscribe(&self) -> mpsc::UnboundedReceiver<Envelope<T>> {
        let (sender, receiver) = mpsc::unbounded();
        self.subscribers
            .lock()
            .expect("Topic subscribers poisoned")
            .push(sender);
        receiver
    }
}

/// Publish capability for one typed fanout topic.
#[derive(Clone)]
pub struct TopicMailbox<T: Clone> {
    owner: ModuleId,
    topic: Topic<T>,
}

impl<T: Clone> TopicMailbox<T> {
    pub(crate) fn new(owner: ModuleId, topic: Topic<T>) -> Self {
        Self { owner, topic }
    }

    pub fn owner(&self) -> &ModuleId {
        &self.owner
    }

    pub fn publish(&self, body: T) -> Result<usize, Envelope<T>> {
        let envelope = Envelope {
            sender: self.owner.clone(),
            body,
        };
        let mut delivered = 0;
        let mut subscribers = self
            .topic
            .subscribers
            .lock()
            .expect("Topic subscribers poisoned");

        subscribers.retain(|subscriber| {
            if subscriber.unbounded_send(envelope.clone()).is_ok() {
                delivered += 1;
                true
            } else {
                false
            }
        });

        if delivered == 0 {
            Err(envelope)
        } else {
            Ok(delivered)
        }
    }
}

/// Subscribe capability for one typed fanout topic.
pub struct TopicInbox<T: Clone> {
    owner: ModuleId,
    receiver: mpsc::UnboundedReceiver<Envelope<T>>,
}

impl<T: Clone> TopicInbox<T> {
    pub(crate) fn new(owner: ModuleId, topic: Topic<T>) -> Self {
        Self {
            owner,
            receiver: topic.subscribe(),
        }
    }

    pub fn owner(&self) -> &ModuleId {
        &self.owner
    }

    pub async fn next_item(&mut self) -> Result<Envelope<T>, TopicRecvError> {
        self.receiver.next().await.ok_or(TopicRecvError::Closed)
    }

    pub fn take_ready_items(&mut self) -> Result<ReadyItems<T>, TopicRecvError> {
        let mut items = Vec::new();
        loop {
            match self.receiver.try_recv() {
                Ok(envelope) => items.push(envelope),
                Err(mpsc::TryRecvError::Empty) => return Ok(ReadyItems { items }),
                Err(mpsc::TryRecvError::Closed) => return Err(TopicRecvError::Closed),
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct QueryRequest {
    pub question: String,
}

impl QueryRequest {
    pub fn new(question: impl Into<String>) -> Self {
        Self {
            question: question.into(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct SelfModelRequest {
    pub question: String,
}

impl SelfModelRequest {
    pub fn new(question: impl Into<String>) -> Self {
        Self {
            question: question.into(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub enum MemoryImportance {
    Normal,
    High,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct MemoryRequest {
    pub content: String,
    pub importance: MemoryImportance,
    pub reason: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct AttentionStreamUpdated;

pub type QueryMailbox = TopicMailbox<QueryRequest>;
pub type QueryInbox = TopicInbox<QueryRequest>;
pub type SelfModelMailbox = TopicMailbox<SelfModelRequest>;
pub type SelfModelInbox = TopicInbox<SelfModelRequest>;
pub type MemoryRequestMailbox = TopicMailbox<MemoryRequest>;
pub type MemoryRequestInbox = TopicInbox<MemoryRequest>;
pub type AttentionStreamUpdatedMailbox = TopicMailbox<AttentionStreamUpdated>;
pub type AttentionStreamUpdatedInbox = TopicInbox<AttentionStreamUpdated>;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub enum SensoryInput {
    Heard {
        direction: Option<String>,
        content: String,
        observed_at: DateTime<Utc>,
    },
    Seen {
        direction: Option<String>,
        appearance: String,
        observed_at: DateTime<Utc>,
    },
}

pub type SensoryInputMailbox = TopicMailbox<SensoryInput>;
pub type SensoryInputInbox = TopicInbox<SensoryInput>;
