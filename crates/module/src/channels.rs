use std::borrow::Cow;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use chrono::{DateTime, Utc};
use futures::StreamExt;
use futures::channel::mpsc;
use nuillu_blackboard::{Blackboard, CognitionLogEntryRecord, MemoLogRecord};
use nuillu_types::{ModuleId, ModuleInstanceId};
use schemars::{JsonSchema, Schema, SchemaGenerator};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::rate_limit::{CapabilityKind, RateLimiter, TopicKind};
use crate::runtime_events::RuntimeEventEmitter;

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
    kind: TopicKind,
    rate_limiter: RateLimiter,
    events: RuntimeEventEmitter,
}

impl<T: Clone> Topic<T> {
    pub(crate) fn new(
        blackboard: Blackboard,
        policy: TopicPolicy,
        kind: TopicKind,
        rate_limiter: RateLimiter,
        events: RuntimeEventEmitter,
    ) -> Self {
        Self {
            inner: Arc::new(Mutex::new(TopicInner::default())),
            blackboard,
            policy,
            kind,
            rate_limiter,
            events,
        }
    }

    fn subscribe(
        &self,
        owner: ModuleInstanceId,
        exclude_self: bool,
    ) -> mpsc::UnboundedReceiver<Envelope<T>> {
        let (sender, receiver) = mpsc::unbounded();
        self.inner
            .lock()
            .expect("Topic inner poisoned")
            .subscribers
            .push(TopicSubscriber {
                owner,
                sender,
                exclude_self,
            });
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
    exclude_self: bool,
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
        let capability = CapabilityKind::ChannelPublish {
            topic: self.topic.kind,
        };
        let outcome = self
            .topic
            .rate_limiter
            .acquire(&self.owner, capability)
            .await;
        if outcome.was_delayed() {
            self.topic
                .events
                .rate_limit_delayed(self.owner.clone(), capability, outcome.delayed_for)
                .await;
        }

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
                let mut active_by_role = HashMap::<ModuleId, bool>::new();
                for subscriber in &inner.subscribers {
                    let active = allocation.is_replica_active(&subscriber.owner);
                    active_by_role
                        .entry(subscriber.owner.module.clone())
                        .and_modify(|any_active| *any_active |= active)
                        .or_insert(active);
                }
                for subscriber in inner.subscribers.iter().filter(|subscriber| {
                    allocation.is_replica_active(&subscriber.owner)
                        || (!active_by_role
                            .get(&subscriber.owner.module)
                            .copied()
                            .unwrap_or(false)
                            && subscriber.owner.replica == nuillu_types::ReplicaIndex::ZERO)
                }) {
                    if subscriber.exclude_self && subscriber.owner == envelope.sender {
                        continue;
                    }
                    if subscriber.sender.unbounded_send(envelope.clone()).is_ok() {
                        delivered += 1;
                    }
                }
            }
            TopicPolicy::RoleLoadBalanced => {
                let mut by_role = HashMap::<ModuleId, Vec<usize>>::new();
                let mut fallback_by_role = HashMap::<ModuleId, usize>::new();
                for (idx, subscriber) in inner.subscribers.iter().enumerate() {
                    if subscriber.exclude_self && subscriber.owner == envelope.sender {
                        continue;
                    }
                    if allocation.is_replica_active(&subscriber.owner) {
                        by_role
                            .entry(subscriber.owner.module.clone())
                            .or_default()
                            .push(idx);
                    } else if subscriber.owner.replica == nuillu_types::ReplicaIndex::ZERO {
                        fallback_by_role
                            .entry(subscriber.owner.module.clone())
                            .or_insert(idx);
                    }
                }
                for (role, idx) in fallback_by_role {
                    by_role.entry(role).or_insert_with(|| vec![idx]);
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
            receiver: topic.subscribe(owner, false),
            exclude_self: false,
        }
    }

    pub(crate) fn new_excluding_self(owner: ModuleInstanceId, topic: Topic<T>) -> Self {
        Self {
            owner: owner.clone(),
            receiver: topic.subscribe(owner, true),
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub enum MemoryImportance {
    Normal,
    High,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum AttentionControlRequest {
    Query {
        question: String,
        reason: Option<String>,
    },
    SelfModel {
        question: String,
        reason: Option<String>,
    },
    Memory {
        content: String,
        importance: MemoryImportance,
        reason: String,
    },
}

impl AttentionControlRequest {
    pub fn query(question: impl Into<String>) -> Self {
        Self::Query {
            question: question.into(),
            reason: None,
        }
    }

    pub fn query_with_reason(question: impl Into<String>, reason: impl Into<String>) -> Self {
        Self::Query {
            question: question.into(),
            reason: Some(reason.into()),
        }
    }

    pub fn self_model(question: impl Into<String>) -> Self {
        Self::SelfModel {
            question: question.into(),
            reason: None,
        }
    }

    pub fn self_model_with_reason(question: impl Into<String>, reason: impl Into<String>) -> Self {
        Self::SelfModel {
            question: question.into(),
            reason: Some(reason.into()),
        }
    }

    pub fn memory(
        content: impl Into<String>,
        importance: MemoryImportance,
        reason: impl Into<String>,
    ) -> Self {
        Self::Memory {
            content: content.into(),
            importance,
            reason: reason.into(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum CognitionLogUpdated {
    EntryAppended { source: ModuleInstanceId },
    AgenticDeadlockMarker,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct AllocationUpdated;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct InteroceptiveUpdated;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct MemoUpdated {
    pub owner: ModuleInstanceId,
    pub index: u64,
}

pub type AttentionControlRequestMailbox = TopicMailbox<AttentionControlRequest>;
pub type AttentionControlRequestInbox = TopicInbox<AttentionControlRequest>;
pub type CognitionLogUpdatedMailbox = TopicMailbox<CognitionLogUpdated>;
pub type CognitionLogUpdatedInbox = TopicInbox<CognitionLogUpdated>;
pub type AllocationUpdatedMailbox = TopicMailbox<AllocationUpdated>;
pub type AllocationUpdatedInbox = TopicInbox<AllocationUpdated>;
pub type InteroceptiveUpdatedMailbox = TopicMailbox<InteroceptiveUpdated>;
pub type InteroceptiveUpdatedInbox = TopicInbox<InteroceptiveUpdated>;
pub type MemoUpdatedMailbox = TopicMailbox<MemoUpdated>;
pub type MemoUpdatedInbox = TopicInbox<MemoUpdated>;
pub type MemoLogEvictedMailbox = TopicMailbox<MemoLogRecord>;
pub type MemoLogEvictedInbox = TopicInbox<MemoLogRecord>;
pub type CognitionLogEvictedMailbox = TopicMailbox<CognitionLogEntryRecord>;
pub type CognitionLogEvictedInbox = TopicInbox<CognitionLogEntryRecord>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SensoryModality {
    Vision,
    Audition,
    Smell,
    Taste,
    Touch,
    Proprioception,
    Interoception,
    Other(String),
}

impl SensoryModality {
    pub fn parse(value: impl AsRef<str>) -> Self {
        let value = value.as_ref().trim();
        match normalize_modality(value).as_str() {
            "vision" | "visual" | "seen" | "sight" => Self::Vision,
            "audition" | "audio" | "heard" | "hearing" | "sound" => Self::Audition,
            "smell" | "olfaction" | "scent" => Self::Smell,
            "taste" | "gustation" => Self::Taste,
            "touch" | "tactile" => Self::Touch,
            "proprioception" | "proprioceptive" => Self::Proprioception,
            "interoception" | "interoceptive" => Self::Interoception,
            "" => Self::Other("other".to_string()),
            normalized => Self::Other(normalized.to_string()),
        }
    }

    pub fn as_str(&self) -> &str {
        match self {
            Self::Vision => "vision",
            Self::Audition => "audition",
            Self::Smell => "smell",
            Self::Taste => "taste",
            Self::Touch => "touch",
            Self::Proprioception => "proprioception",
            Self::Interoception => "interoception",
            Self::Other(value) => value.as_str(),
        }
    }
}

impl Serialize for SensoryModality {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(self.as_str())
    }
}

impl<'de> Deserialize<'de> for SensoryModality {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        String::deserialize(deserializer).map(Self::parse)
    }
}

impl JsonSchema for SensoryModality {
    fn inline_schema() -> bool {
        true
    }

    fn schema_name() -> Cow<'static, str> {
        "SensoryModality".into()
    }

    fn json_schema(_generator: &mut SchemaGenerator) -> Schema {
        Schema::try_from(serde_json::json!({
            "type": "string",
            "description": "Sensory category/modality such as vision, audition, smell, taste, touch, proprioception, interoception, or a custom modality string."
        }))
        .expect("sensory modality schema must be a JSON object")
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct AmbientSensoryEntry {
    pub id: String,
    pub modality: SensoryModality,
    pub content: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SensoryInput {
    OneShot {
        modality: SensoryModality,
        direction: Option<String>,
        content: String,
        observed_at: DateTime<Utc>,
    },
    AmbientSnapshot {
        entries: Vec<AmbientSensoryEntry>,
        observed_at: DateTime<Utc>,
    },
}

pub type SensoryInputMailbox = TopicMailbox<SensoryInput>;
pub type SensoryInputInbox = TopicInbox<SensoryInput>;

fn normalize_modality(value: &str) -> String {
    value
        .chars()
        .flat_map(char::to_lowercase)
        .map(|ch| if ch.is_ascii_alphanumeric() { ch } else { '-' })
        .collect::<String>()
        .split('-')
        .filter(|part| !part.is_empty())
        .collect::<Vec<_>>()
        .join("-")
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::time::Duration;

    use nuillu_blackboard::{ActivationRatio, BlackboardCommand, ModuleConfig, ResourceAllocation};
    use nuillu_types::{ReplicaCapRange, builtin};
    use tokio::time::Instant;

    use crate::test_support::{scoped, test_caps, test_caps_with_policy};
    use crate::{CapabilityKind, RateLimitConfig, RateLimitPolicy, RuntimePolicy, TopicKind};

    fn ticker_id() -> ModuleId {
        ModuleId::new("ticker").unwrap()
    }

    #[test]
    fn multimodal_sensory_input_round_trips_as_strings() {
        let input = SensoryInput::AmbientSnapshot {
            entries: vec![AmbientSensoryEntry {
                id: "ambient-1".to_string(),
                modality: SensoryModality::Other("thermal".to_string()),
                content: "warm air near the door".to_string(),
            }],
            observed_at: DateTime::parse_from_rfc3339("2026-05-13T00:00:00Z")
                .unwrap()
                .with_timezone(&Utc),
        };

        let json = serde_json::to_value(&input).unwrap();
        assert_eq!(
            json,
            serde_json::json!({
                "kind": "ambient_snapshot",
                "entries": [
                    {
                        "id": "ambient-1",
                        "modality": "thermal",
                        "content": "warm air near the door",
                    }
                ],
                "observed_at": "2026-05-13T00:00:00Z",
            })
        );
        assert_eq!(serde_json::from_value::<SensoryInput>(json).unwrap(), input);

        let input = SensoryInput::OneShot {
            modality: SensoryModality::Audition,
            direction: Some("front".to_string()),
            content: "a bell rings".to_string(),
            observed_at: DateTime::parse_from_rfc3339("2026-05-13T00:00:00Z")
                .unwrap()
                .with_timezone(&Utc),
        };
        let json = serde_json::to_value(&input).unwrap();
        assert_eq!(
            json,
            serde_json::json!({
                "kind": "one_shot",
                "modality": "audition",
                "direction": "front",
                "content": "a bell rings",
                "observed_at": "2026-05-13T00:00:00Z",
            })
        );
        assert_eq!(serde_json::from_value::<SensoryInput>(json).unwrap(), input);
    }

    #[tokio::test]
    async fn attention_control_mailbox_delivers_to_controller_with_owner_stamp() {
        let caps = test_caps(Blackboard::default());
        let publisher = scoped(&caps, ticker_id(), 0).attention_control_mailbox();
        let mut controller =
            scoped(&caps, builtin::allocation_controller(), 0).attention_control_inbox();

        publisher
            .publish(AttentionControlRequest::query("find memories about rust"))
            .await
            .expect("attention-control topic should have subscribers");

        let envelope = controller
            .next_item()
            .await
            .expect("controller subscriber receives request");
        assert_eq!(envelope.sender.module, ticker_id());
        assert_eq!(
            envelope.body,
            AttentionControlRequest::query("find memories about rust")
        );
    }

    #[tokio::test]
    async fn attention_control_load_balances_across_active_controller_replicas() {
        let mut alloc = ResourceAllocation::default();
        alloc.set(builtin::allocation_controller(), ModuleConfig::default());
        alloc.set_activation(builtin::allocation_controller(), ActivationRatio::ONE);
        let blackboard = Blackboard::with_allocation(alloc);
        blackboard
            .apply(BlackboardCommand::SetModulePolicies {
                policies: vec![(
                    builtin::allocation_controller(),
                    nuillu_blackboard::ModulePolicy::new(
                        ReplicaCapRange::new(0, 2).unwrap(),
                        nuillu_blackboard::Bpm::from_f64(60.0)
                            ..=nuillu_blackboard::Bpm::from_f64(60.0),
                        nuillu_blackboard::linear_ratio_fn,
                    ),
                )],
            })
            .await;
        let caps = test_caps(blackboard);
        let publisher = scoped(&caps, ticker_id(), 0).attention_control_mailbox();
        let mut controller_0 =
            scoped(&caps, builtin::allocation_controller(), 0).attention_control_inbox();
        let mut controller_1 =
            scoped(&caps, builtin::allocation_controller(), 1).attention_control_inbox();

        publisher
            .publish(AttentionControlRequest::query("first"))
            .await
            .unwrap();
        publisher
            .publish(AttentionControlRequest::query("second"))
            .await
            .unwrap();

        assert_eq!(
            controller_0.next_item().await.unwrap().body,
            AttentionControlRequest::query("first")
        );
        assert_eq!(
            controller_1.next_item().await.unwrap().body,
            AttentionControlRequest::query("second")
        );
    }

    #[tokio::test]
    async fn attention_control_routes_to_replica_zero_when_controller_is_inactive() {
        let mut alloc = ResourceAllocation::default();
        alloc.set(builtin::allocation_controller(), ModuleConfig::default());
        alloc.set_activation(builtin::allocation_controller(), ActivationRatio::ZERO);
        let blackboard = Blackboard::with_allocation(alloc);
        blackboard
            .apply(BlackboardCommand::SetModulePolicies {
                policies: vec![(
                    builtin::allocation_controller(),
                    nuillu_blackboard::ModulePolicy::new(
                        ReplicaCapRange::new(0, 2).unwrap(),
                        nuillu_blackboard::Bpm::from_f64(60.0)
                            ..=nuillu_blackboard::Bpm::from_f64(60.0),
                        nuillu_blackboard::linear_ratio_fn,
                    ),
                )],
            })
            .await;
        let caps = test_caps(blackboard);
        let publisher = scoped(&caps, ticker_id(), 0).attention_control_mailbox();
        let mut controller_0 =
            scoped(&caps, builtin::allocation_controller(), 0).attention_control_inbox();
        let mut controller_1 =
            scoped(&caps, builtin::allocation_controller(), 1).attention_control_inbox();

        publisher
            .publish(AttentionControlRequest::query("active only"))
            .await
            .unwrap();

        assert_eq!(
            controller_0.next_item().await.unwrap().body,
            AttentionControlRequest::query("active only")
        );
        assert!(controller_1.take_ready_items().unwrap().items.is_empty());
    }

    #[tokio::test]
    async fn publish_waits_before_routing_when_rate_limited() {
        let publisher_id = ticker_id();
        let caps = test_caps_with_policy(
            Blackboard::default(),
            RuntimePolicy {
                rate_limits: RateLimitPolicy::for_module(
                    publisher_id.clone(),
                    CapabilityKind::ChannelPublish {
                        topic: TopicKind::AttentionControlRequest,
                    },
                    RateLimitConfig::new(Duration::from_millis(10), 100.0).unwrap(),
                )
                .unwrap(),
                ..RuntimePolicy::default()
            },
        );
        let publisher = scoped(&caps, publisher_id, 0).attention_control_mailbox();
        let mut controller =
            scoped(&caps, builtin::allocation_controller(), 0).attention_control_inbox();

        publisher
            .publish(AttentionControlRequest::query("first"))
            .await
            .unwrap();
        let started = Instant::now();
        publisher
            .publish(AttentionControlRequest::query("second"))
            .await
            .unwrap();

        assert!(started.elapsed() >= Duration::from_millis(8));
        assert_eq!(
            controller.next_item().await.unwrap().body,
            AttentionControlRequest::query("first")
        );
        assert_eq!(
            controller.next_item().await.unwrap().body,
            AttentionControlRequest::query("second")
        );
    }
}
