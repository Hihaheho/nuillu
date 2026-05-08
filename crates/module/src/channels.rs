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
    owner: ModuleInstanceId,
    receiver: mpsc::UnboundedReceiver<Envelope<T>>,
    exclude_self: bool,
}

impl<T: Clone> TopicInbox<T> {
    pub(crate) fn new(owner: ModuleInstanceId, topic: Topic<T>) -> Self {
        Self {
            owner: owner.clone(),
            receiver: topic.subscribe(owner.clone()),
            exclude_self: false,
        }
    }

    pub(crate) fn new_excluding_self(owner: ModuleInstanceId, topic: Topic<T>) -> Self {
        Self {
            owner: owner.clone(),
            receiver: topic.subscribe(owner),
            exclude_self: true,
        }
    }

    pub async fn next_item(&mut self) -> Result<Envelope<T>, TopicRecvError> {
        while let Some(envelope) = self.receiver.next().await {
            if self.accepts(&envelope) {
                return Ok(envelope);
            }
        }
        Err(TopicRecvError::Closed)
    }

    pub fn take_ready_items(&mut self) -> Result<ReadyItems<T>, TopicRecvError> {
        let mut items = Vec::new();
        loop {
            match self.receiver.try_recv() {
                Ok(envelope) => {
                    if self.accepts(&envelope) {
                        items.push(envelope);
                    }
                }
                Err(mpsc::TryRecvError::Empty) => return Ok(ReadyItems { items }),
                Err(mpsc::TryRecvError::Closed) => return Err(TopicRecvError::Closed),
            }
        }
    }

    fn accepts(&self, envelope: &Envelope<T>) -> bool {
        !self.exclude_self || envelope.sender != self.owner
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
#[serde(tag = "source", rename_all = "snake_case")]
pub enum AttentionStreamUpdated {
    StreamAppended { stream: ModuleInstanceId },
    AgenticDeadlockMarker,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct AllocationUpdated;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct MemoUpdated {
    pub owner: ModuleInstanceId,
}

pub type QueryMailbox = TopicMailbox<QueryRequest>;
pub type QueryInbox = TopicInbox<QueryRequest>;
pub type SelfModelMailbox = TopicMailbox<SelfModelRequest>;
pub type SelfModelInbox = TopicInbox<SelfModelRequest>;
pub type MemoryRequestMailbox = TopicMailbox<MemoryRequest>;
pub type MemoryRequestInbox = TopicInbox<MemoryRequest>;
pub type AttentionStreamUpdatedMailbox = TopicMailbox<AttentionStreamUpdated>;
pub type AttentionStreamUpdatedInbox = TopicInbox<AttentionStreamUpdated>;
pub type AllocationUpdatedMailbox = TopicMailbox<AllocationUpdated>;
pub type AllocationUpdatedInbox = TopicInbox<AllocationUpdated>;
pub type MemoUpdatedMailbox = TopicMailbox<MemoUpdated>;
pub type MemoUpdatedInbox = TopicInbox<MemoUpdated>;

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

#[cfg(test)]
mod tests {
    use super::*;

    use nuillu_blackboard::{ActivationRatio, ModuleConfig, ResourceAllocation};
    use nuillu_types::{ReplicaCapRange, builtin};

    use crate::test_support::{scoped, test_caps};

    fn ticker_id() -> ModuleId {
        ModuleId::new("ticker").unwrap()
    }

    fn echo_id() -> ModuleId {
        ModuleId::new("echo").unwrap()
    }

    #[tokio::test]
    async fn query_mailbox_fans_out_to_multiple_subscribers_with_owner_stamp() {
        let caps = test_caps(Blackboard::default());
        let publisher = scoped(&caps, ticker_id(), 0).query_mailbox();
        let mut vector = scoped(&caps, builtin::query_vector(), 0).query_inbox();
        let mut agentic = scoped(&caps, builtin::query_agentic(), 0).query_inbox();

        publisher
            .publish(QueryRequest::new("find memories about rust"))
            .await
            .expect("query topic should have subscribers");

        let vector_env = vector
            .next_item()
            .await
            .expect("vector subscriber receives query");
        let agentic_env = agentic
            .next_item()
            .await
            .expect("agentic subscriber receives query");

        assert_eq!(vector_env.sender, agentic_env.sender);
        assert_eq!(vector_env.sender.module, ticker_id());
        assert_eq!(vector_env.body.question, "find memories about rust");
        assert_eq!(agentic_env.body.question, "find memories about rust");
    }

    #[tokio::test]
    async fn self_model_mailbox_is_a_separate_typed_topic() {
        let caps = test_caps(Blackboard::default());
        let query_publisher = scoped(&caps, ticker_id(), 0).query_mailbox();
        let self_publisher = scoped(&caps, echo_id(), 0).self_model_mailbox();
        let mut query_inbox = scoped(&caps, builtin::query_vector(), 0).query_inbox();
        let mut self_inbox = scoped(&caps, builtin::attention_schema(), 0).self_model_inbox();

        query_publisher
            .publish(QueryRequest::new("memory only"))
            .await
            .expect("query subscriber exists");
        self_publisher
            .publish(SelfModelRequest::new("what are you aware of?"))
            .await
            .expect("self-model subscriber exists");

        assert_eq!(
            query_inbox.next_item().await.unwrap().body.question,
            "memory only"
        );
        assert_eq!(
            self_inbox.next_item().await.unwrap().body.question,
            "what are you aware of?"
        );
        assert!(query_inbox.take_ready_items().unwrap().items.is_empty());
        assert!(self_inbox.take_ready_items().unwrap().items.is_empty());
    }

    #[tokio::test]
    async fn role_topics_load_balance_across_active_replicas() {
        let mut alloc = ResourceAllocation::default();
        alloc.set(
            builtin::query_vector(),
            ModuleConfig {
                activation_ratio: ActivationRatio::ONE,
                ..Default::default()
            },
        );
        alloc.set(
            builtin::query_agentic(),
            ModuleConfig {
                activation_ratio: ActivationRatio::ONE,
                ..Default::default()
            },
        );
        let alloc = alloc.clamped(&std::collections::HashMap::from([
            (builtin::query_vector(), ReplicaCapRange { min: 0, max: 2 }),
            (builtin::query_agentic(), ReplicaCapRange { min: 0, max: 1 }),
        ]));
        let caps = test_caps(Blackboard::with_allocation(alloc));
        let publisher = scoped(&caps, ticker_id(), 0).query_mailbox();
        let mut vector_0 = scoped(&caps, builtin::query_vector(), 0).query_inbox();
        let mut vector_1 = scoped(&caps, builtin::query_vector(), 1).query_inbox();
        let mut agentic_0 = scoped(&caps, builtin::query_agentic(), 0).query_inbox();

        publisher.publish(QueryRequest::new("first")).await.unwrap();
        publisher
            .publish(QueryRequest::new("second"))
            .await
            .unwrap();

        assert_eq!(vector_0.next_item().await.unwrap().body.question, "first");
        assert_eq!(vector_1.next_item().await.unwrap().body.question, "second");
        let agentic = agentic_0
            .take_ready_items()
            .unwrap()
            .items
            .into_iter()
            .map(|item| item.body.question)
            .collect::<Vec<_>>();
        assert_eq!(agentic, vec!["first", "second"]);
    }

    #[tokio::test]
    async fn role_topics_do_not_route_to_disabled_replicas() {
        let mut alloc = ResourceAllocation::default();
        alloc.set(
            builtin::query_vector(),
            ModuleConfig {
                activation_ratio: ActivationRatio::from_f64(0.5),
                ..Default::default()
            },
        );
        let alloc = alloc.clamped(&std::collections::HashMap::from([(
            builtin::query_vector(),
            ReplicaCapRange { min: 0, max: 2 },
        )]));
        let caps = test_caps(Blackboard::with_allocation(alloc));
        let publisher = scoped(&caps, ticker_id(), 0).query_mailbox();
        let mut vector_0 = scoped(&caps, builtin::query_vector(), 0).query_inbox();
        let mut vector_1 = scoped(&caps, builtin::query_vector(), 1).query_inbox();

        publisher
            .publish(QueryRequest::new("active only"))
            .await
            .unwrap();

        assert_eq!(
            vector_0.next_item().await.unwrap().body.question,
            "active only"
        );
        assert!(vector_1.take_ready_items().unwrap().items.is_empty());
    }
}
