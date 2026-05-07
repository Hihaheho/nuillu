use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use chrono::{DateTime, Utc};
use futures::StreamExt;
use futures::channel::mpsc;
use nuillu_blackboard::Blackboard;
use nuillu_types::{ModuleId, ModuleInstanceId};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// Owner-stamped message delivered over a typed topic.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct Envelope<T> {
    pub sender: ModuleInstanceId,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TopicPolicy {
    Fanout,
    RoleLoadBalanced,
}

#[derive(Clone)]
pub(crate) struct Topic<T: Clone> {
    inner: Arc<Mutex<TopicInner<T>>>,
    blackboard: Blackboard,
    policy: TopicPolicy,
}

impl<T: Clone> Topic<T> {
    pub(crate) fn new(blackboard: Blackboard, policy: TopicPolicy) -> Self {
        Self {
            inner: Arc::new(Mutex::new(TopicInner::default())),
            blackboard,
            policy,
        }
    }

    fn subscribe(&self, owner: ModuleInstanceId) -> mpsc::UnboundedReceiver<Envelope<T>> {
        let (sender, receiver) = mpsc::unbounded();
        self.inner
            .lock()
            .expect("Topic inner poisoned")
            .subscribers
            .push(TopicSubscriber { owner, sender });
        receiver
    }
}

struct TopicInner<T: Clone> {
    subscribers: Vec<TopicSubscriber<T>>,
    next_by_role: HashMap<ModuleId, usize>,
}

impl<T: Clone> Default for TopicInner<T> {
    fn default() -> Self {
        Self {
            subscribers: Vec::new(),
            next_by_role: HashMap::new(),
        }
    }
}

struct TopicSubscriber<T: Clone> {
    owner: ModuleInstanceId,
    sender: mpsc::UnboundedSender<Envelope<T>>,
}

/// Publish capability for one typed topic.
#[derive(Clone)]
pub struct TopicMailbox<T: Clone> {
    owner: ModuleInstanceId,
    topic: Topic<T>,
}

impl<T: Clone> TopicMailbox<T> {
    pub(crate) fn new(owner: ModuleInstanceId, topic: Topic<T>) -> Self {
        Self { owner, topic }
    }

    pub async fn publish(&self, body: T) -> Result<usize, Envelope<T>> {
        let envelope = Envelope {
            sender: self.owner.clone(),
            body,
        };
        let allocation = self
            .topic
            .blackboard
            .read(|bb| bb.allocation().clone())
            .await;
        let mut delivered = 0;
        let mut inner = self.topic.inner.lock().expect("Topic inner poisoned");

        inner
            .subscribers
            .retain(|subscriber| !subscriber.sender.is_closed());

        match self.topic.policy {
            TopicPolicy::Fanout => {
                for subscriber in inner
                    .subscribers
                    .iter()
                    .filter(|subscriber| allocation.is_replica_active(&subscriber.owner))
                {
                    if subscriber.sender.unbounded_send(envelope.clone()).is_ok() {
                        delivered += 1;
                    }
                }
            }
            TopicPolicy::RoleLoadBalanced => {
                let mut by_role = HashMap::<ModuleId, Vec<usize>>::new();
                for (idx, subscriber) in inner.subscribers.iter().enumerate() {
                    if allocation.is_replica_active(&subscriber.owner) {
                        by_role
                            .entry(subscriber.owner.module.clone())
                            .or_default()
                            .push(idx);
                    }
                }

                for (role, indexes) in by_role {
                    let next = inner.next_by_role.entry(role).or_default();
                    let chosen = indexes[*next % indexes.len()];
                    *next = next.wrapping_add(1);
                    if inner.subscribers[chosen]
                        .sender
                        .unbounded_send(envelope.clone())
                        .is_ok()
                    {
                        delivered += 1;
                    }
                }
            }
        }

        if delivered == 0 {
            Err(envelope)
        } else {
            Ok(delivered)
        }
    }
}

/// Subscribe capability for one typed topic.
pub struct TopicInbox<T: Clone> {
    receiver: mpsc::UnboundedReceiver<Envelope<T>>,
}

impl<T: Clone> TopicInbox<T> {
    pub(crate) fn new(owner: ModuleInstanceId, topic: Topic<T>) -> Self {
        Self {
            receiver: topic.subscribe(owner.clone()),
        }
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

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct AttentionStreamUpdated {
    pub stream: ModuleInstanceId,
}

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
