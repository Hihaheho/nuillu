use std::borrow::Cow;
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use chrono::{DateTime, Utc};
use futures::StreamExt;
use futures::channel::mpsc;
use nuillu_blackboard::{Blackboard, CognitionLogEntryRecord, MemoLogRecord};
use nuillu_types::{ModuleId, ModuleInstanceId, ScopeId, ScopedModuleId};

use crate::MemoSubscription;
use schemars::{JsonSchema, Schema, SchemaGenerator};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

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

/// Opaque claim for one module's currently delivered wake notifications.
///
/// This is a delivery cursor, not a blackboard state epoch: it only tracks
/// successful typed-topic deliveries to a module inbox.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WakeClaim {
    owner: ModuleInstanceId,
    delivered_through: u64,
}

#[derive(Clone, Default)]
pub(crate) struct WakeRegistry {
    inner: Arc<Mutex<WakeRegistryInner>>,
    notify: Arc<tokio::sync::Notify>,
}

#[derive(Default)]
struct WakeRegistryInner {
    delivered_by_owner: HashMap<ModuleInstanceId, u64>,
    completed_by_owner: HashMap<ModuleInstanceId, u64>,
    change_sequence: u64,
}

impl WakeRegistry {
    pub(crate) fn record_wake(&self, owner: &ModuleInstanceId) {
        {
            let mut inner = self.inner.lock().expect("wake registry poisoned");
            let next = inner
                .delivered_by_owner
                .get(owner)
                .copied()
                .unwrap_or_default()
                .saturating_add(1);
            inner.delivered_by_owner.insert(owner.clone(), next);
            inner.change_sequence = inner.change_sequence.saturating_add(1);
        }
        self.notify.notify_waiters();
    }

    pub(crate) fn claim_wake(&self, owner: &ModuleInstanceId) -> Option<WakeClaim> {
        let inner = self.inner.lock().expect("wake registry poisoned");
        let delivered = inner
            .delivered_by_owner
            .get(owner)
            .copied()
            .unwrap_or_default();
        let completed = inner
            .completed_by_owner
            .get(owner)
            .copied()
            .unwrap_or_default();
        (delivered > completed).then(|| WakeClaim {
            owner: owner.clone(),
            delivered_through: delivered,
        })
    }

    pub(crate) fn complete_wake_claim(&self, claim: WakeClaim) {
        let mut inner = self.inner.lock().expect("wake registry poisoned");
        let owner = claim.owner;
        let delivered = inner
            .delivered_by_owner
            .get(&owner)
            .copied()
            .unwrap_or_default();
        let completed = claim.delivered_through.min(delivered);
        inner
            .completed_by_owner
            .entry(owner)
            .and_modify(|current| *current = (*current).max(completed))
            .or_insert(completed);
    }

    pub(crate) fn has_pending_wake(&self, owner: &ModuleInstanceId) -> bool {
        let inner = self.inner.lock().expect("wake registry poisoned");
        let delivered = inner
            .delivered_by_owner
            .get(owner)
            .copied()
            .unwrap_or_default();
        let completed = inner
            .completed_by_owner
            .get(owner)
            .copied()
            .unwrap_or_default();
        delivered > completed
    }

    pub(crate) fn change_sequence(&self) -> u64 {
        self.inner
            .lock()
            .expect("wake registry poisoned")
            .change_sequence
    }

    pub(crate) async fn changed_since(&self, observed: u64) {
        loop {
            let notified = self.notify.notified();
            if self.change_sequence() > observed {
                return;
            }
            notified.await;
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TopicPolicy {
    Fanout,
    RoleLoadBalanced,
}

type RoundRobinHook = Rc<dyn Fn(&ModuleId)>;

#[derive(Clone)]
pub(crate) struct Topic<T: Clone> {
    inner: Arc<Mutex<TopicInner<T>>>,
    blackboard: Blackboard,
    wakes: WakeRegistry,
    default_policy: TopicPolicy,
}

impl<T: Clone> Topic<T> {
    pub(crate) fn new(blackboard: Blackboard, wakes: WakeRegistry, policy: TopicPolicy) -> Self {
        Self {
            inner: Arc::new(Mutex::new(TopicInner::default())),
            blackboard,
            wakes,
            default_policy: policy,
        }
    }

    fn subscribe(
        &self,
        owner: ModuleInstanceId,
        observed_scope: ScopeId,
        exclude_self: bool,
        source_filter: MemoSubscription,
    ) -> TopicSubscription<T> {
        let (sender, receiver) = mpsc::unbounded();
        let pending = Arc::new(AtomicBool::new(false));
        let mut inner = self.inner.lock().expect("Topic inner poisoned");
        let id = inner.next_subscription_id;
        inner.next_subscription_id = inner.next_subscription_id.wrapping_add(1);
        inner.subscribers.push(TopicSubscriber {
            id,
            owner,
            observed_scope,
            sender,
            exclude_self,
            policy: self.default_policy,
            coalesce: false,
            pending: pending.clone(),
            source_filter,
        });
        TopicSubscription {
            id,
            receiver,
            pending,
        }
    }

    fn subscriber_mut(inner: &mut TopicInner<T>, id: u64) -> &mut TopicSubscriber<T> {
        inner
            .subscribers
            .iter_mut()
            .find(|subscriber| subscriber.id == id)
            .expect("topic subscription disappeared while its inbox is alive")
    }

    fn configure_delivery(&self, id: u64, policy: TopicPolicy) {
        let mut inner = self.inner.lock().expect("Topic inner poisoned");
        Self::subscriber_mut(&mut inner, id).policy = policy;
    }

    fn configure_coalescing(&self, id: u64) {
        let mut inner = self.inner.lock().expect("Topic inner poisoned");
        Self::subscriber_mut(&mut inner, id).coalesce = true;
    }
}

struct TopicSubscription<T: Clone> {
    id: u64,
    receiver: mpsc::UnboundedReceiver<Envelope<T>>,
    pending: Arc<AtomicBool>,
}

struct TopicInner<T: Clone> {
    subscribers: Vec<TopicSubscriber<T>>,
    next_by_role: HashMap<ScopedModuleId, usize>,
    next_subscription_id: u64,
}

impl<T: Clone> Default for TopicInner<T> {
    fn default() -> Self {
        Self {
            subscribers: Vec::new(),
            next_by_role: HashMap::new(),
            next_subscription_id: 0,
        }
    }
}

struct TopicSubscriber<T: Clone> {
    id: u64,
    owner: ModuleInstanceId,
    observed_scope: ScopeId,
    sender: mpsc::UnboundedSender<Envelope<T>>,
    exclude_self: bool,
    policy: TopicPolicy,
    coalesce: bool,
    pending: Arc<AtomicBool>,
    source_filter: MemoSubscription,
}

/// Publish capability for one typed topic.
#[derive(Clone)]
pub struct TopicMailbox<T: Clone> {
    owner: ModuleInstanceId,
    delivery_scope: ScopeId,
    topic: Topic<T>,
}

impl<T: Clone> TopicMailbox<T> {
    pub(crate) fn new(owner: ModuleInstanceId, topic: Topic<T>) -> Self {
        let delivery_scope = owner.scope.clone();
        Self {
            owner,
            delivery_scope,
            topic,
        }
    }

    pub(crate) fn new_in_scope(
        owner: ModuleInstanceId,
        delivery_scope: ScopeId,
        topic: Topic<T>,
    ) -> Self {
        Self {
            owner,
            delivery_scope,
            topic,
        }
    }

    pub async fn publish(&self, body: T) -> Result<usize, Envelope<T>> {
        let envelope = Envelope {
            sender: self.owner.clone(),
            body,
        };
        let mut allocations = HashMap::new();
        for blackboard in self.topic.blackboard.all_scopes() {
            allocations.insert(
                blackboard.scope().clone(),
                blackboard.read(|bb| bb.allocation().clone()).await,
            );
        }
        let mut delivered = 0;
        let mut delivered_owners = Vec::new();
        let mut inner = self.topic.inner.lock().expect("Topic inner poisoned");

        inner
            .subscribers
            .retain(|subscriber| !subscriber.sender.is_closed());

        let mut active_by_role = HashMap::<ScopedModuleId, bool>::new();
        for subscriber in &inner.subscribers {
            if subscriber.observed_scope != self.delivery_scope {
                continue;
            }
            if subscriber.exclude_self && subscriber.owner == envelope.sender {
                continue;
            }
            if !subscriber.source_filter.accepts(&envelope.sender.module) {
                continue;
            }
            let active = allocations
                .get(&subscriber.owner.scope)
                .is_some_and(|allocation| allocation.is_replica_active(&subscriber.owner));
            active_by_role
                .entry(subscriber.owner.scoped_module())
                .and_modify(|any_active| *any_active |= active)
                .or_insert(active);
        }

        let mut chosen = Vec::new();
        let mut round_robin_by_role = HashMap::<ScopedModuleId, Vec<usize>>::new();
        let mut round_robin_fallback_by_role = HashMap::<ScopedModuleId, usize>::new();
        for (idx, subscriber) in inner.subscribers.iter().enumerate() {
            if subscriber.observed_scope != self.delivery_scope {
                continue;
            }
            if subscriber.exclude_self && subscriber.owner == envelope.sender {
                continue;
            }
            if !subscriber.source_filter.accepts(&envelope.sender.module) {
                continue;
            }
            let active = allocations
                .get(&subscriber.owner.scope)
                .is_some_and(|allocation| allocation.is_replica_active(&subscriber.owner));
            let fallback = !active_by_role
                .get(&subscriber.owner.scoped_module())
                .copied()
                .unwrap_or(false)
                && subscriber.owner.replica == nuillu_types::ReplicaIndex::ZERO;
            match subscriber.policy {
                TopicPolicy::Fanout if active || fallback => chosen.push(idx),
                TopicPolicy::Fanout => {}
                TopicPolicy::RoleLoadBalanced if active => {
                    round_robin_by_role
                        .entry(subscriber.owner.scoped_module())
                        .or_default()
                        .push(idx);
                }
                TopicPolicy::RoleLoadBalanced
                    if subscriber.owner.replica == nuillu_types::ReplicaIndex::ZERO =>
                {
                    round_robin_fallback_by_role
                        .entry(subscriber.owner.scoped_module())
                        .or_insert(idx);
                }
                TopicPolicy::RoleLoadBalanced => {}
            }
        }
        for (role, idx) in round_robin_fallback_by_role {
            round_robin_by_role.entry(role).or_insert_with(|| vec![idx]);
        }
        for (role, indexes) in round_robin_by_role {
            let next = inner.next_by_role.entry(role).or_default();
            chosen.push(indexes[*next % indexes.len()]);
            *next = next.wrapping_add(1);
        }

        for idx in chosen {
            let subscriber = &inner.subscribers[idx];
            let should_enqueue = !subscriber.coalesce
                || subscriber
                    .pending
                    .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
                    .is_ok();
            if !should_enqueue {
                delivered += 1;
                continue;
            }
            if subscriber.sender.unbounded_send(envelope.clone()).is_ok() {
                delivered += 1;
                delivered_owners.push(subscriber.owner.clone());
            } else if subscriber.coalesce {
                subscriber.pending.store(false, Ordering::Release);
            }
        }
        drop(inner);

        for owner in &delivered_owners {
            self.topic.wakes.record_wake(owner);
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
    topic: Topic<T>,
    subscription_id: u64,
    receiver: mpsc::UnboundedReceiver<Envelope<T>>,
    pending: Arc<AtomicBool>,
    exclude_self: bool,
    delivery_configured: bool,
    round_robin_hook: Option<RoundRobinHook>,
}

impl<T: Clone> TopicInbox<T> {
    pub(crate) fn new(owner: ModuleInstanceId, topic: Topic<T>) -> Self {
        let observed_scope = owner.scope.clone();
        let subscription =
            topic.subscribe(owner.clone(), observed_scope, false, MemoSubscription::All);
        Self {
            owner,
            topic,
            subscription_id: subscription.id,
            receiver: subscription.receiver,
            pending: subscription.pending,
            exclude_self: false,
            delivery_configured: false,
            round_robin_hook: None,
        }
    }

    pub(crate) fn new_excluding_self(owner: ModuleInstanceId, topic: Topic<T>) -> Self {
        Self::new_excluding_self_with_round_robin_hook(owner, topic, None)
    }

    pub(crate) fn new_excluding_self_with_round_robin_hook(
        owner: ModuleInstanceId,
        topic: Topic<T>,
        round_robin_hook: Option<RoundRobinHook>,
    ) -> Self {
        let observed_scope = owner.scope.clone();
        let subscription =
            topic.subscribe(owner.clone(), observed_scope, true, MemoSubscription::All);
        Self {
            owner,
            topic,
            subscription_id: subscription.id,
            receiver: subscription.receiver,
            pending: subscription.pending,
            exclude_self: true,
            delivery_configured: false,
            round_robin_hook,
        }
    }

    pub(crate) fn new_excluding_self_with_round_robin_hook_and_sources(
        owner: ModuleInstanceId,
        topic: Topic<T>,
        round_robin_hook: Option<RoundRobinHook>,
        source_filter: MemoSubscription,
    ) -> Self {
        let observed_scope = owner.scope.clone();
        let subscription = topic.subscribe(owner.clone(), observed_scope, true, source_filter);
        Self {
            owner,
            topic,
            subscription_id: subscription.id,
            receiver: subscription.receiver,
            pending: subscription.pending,
            exclude_self: true,
            delivery_configured: false,
            round_robin_hook,
        }
    }

    pub(crate) fn new_excluding_self_in_scope(
        owner: ModuleInstanceId,
        observed_scope: ScopeId,
        topic: Topic<T>,
        round_robin_hook: Option<RoundRobinHook>,
    ) -> Self {
        let subscription =
            topic.subscribe(owner.clone(), observed_scope, true, MemoSubscription::All);
        Self {
            owner,
            topic,
            subscription_id: subscription.id,
            receiver: subscription.receiver,
            pending: subscription.pending,
            exclude_self: true,
            delivery_configured: false,
            round_robin_hook,
        }
    }

    /// Deliver every message to this module replica. This is the default for
    /// update topics such as memo and cognition-log notifications.
    pub fn broadcast(mut self) -> Self {
        self.set_delivery(TopicPolicy::Fanout);
        self
    }

    /// Deliver each message to one active replica of this module role.
    pub fn round_robin(mut self) -> Self {
        self.set_delivery(TopicPolicy::RoleLoadBalanced);
        if let Some(hook) = &self.round_robin_hook {
            hook(&self.owner.module);
        }
        self
    }

    /// Keep at most one unread activation signal in this inbox.
    ///
    /// Durable state remains authoritative; additional notifications are
    /// accepted but do not allocate more queue entries until the pending one
    /// is consumed.
    pub fn coalesce(self) -> Self {
        self.topic.configure_coalescing(self.subscription_id);
        self
    }

    fn set_delivery(&mut self, policy: TopicPolicy) {
        assert!(
            !self.delivery_configured,
            "topic inbox delivery strategy may only be configured once"
        );
        self.topic.configure_delivery(self.subscription_id, policy);
        self.delivery_configured = true;
    }

    pub async fn next_item(&mut self) -> Result<Envelope<T>, TopicRecvError> {
        while let Some(envelope) = self.receiver.next().await {
            self.pending.store(false, Ordering::Release);
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
                    self.pending.store(false, Ordering::Release);
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
pub enum AttentionControlRequestKind {
    Activate,
    Inhibit,
}

#[derive(Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct AttentionControlRequest {
    kind: AttentionControlRequestKind,
    text: String,
}

impl AttentionControlRequest {
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            kind: AttentionControlRequestKind::Activate,
            text: text.into(),
        }
    }

    pub fn inhibit(text: impl Into<String>) -> Self {
        Self {
            kind: AttentionControlRequestKind::Inhibit,
            text: text.into(),
        }
    }

    pub fn kind(&self) -> AttentionControlRequestKind {
        self.kind
    }

    pub fn as_str(&self) -> &str {
        &self.text
    }

    pub fn into_inner(self) -> String {
        self.text
    }
}

impl From<String> for AttentionControlRequest {
    fn from(text: String) -> Self {
        Self::new(text)
    }
}

impl From<&str> for AttentionControlRequest {
    fn from(text: &str) -> Self {
        Self::new(text)
    }
}

impl std::fmt::Display for AttentionControlRequest {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.text)
    }
}

impl std::fmt::Debug for AttentionControlRequest {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.text.fmt(formatter)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum CognitionLogUpdated {
    EntryAppended { source: ModuleInstanceId },
    AgenticDeadlockMarker,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct InteroceptiveUpdated;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct ActionAffordancesUpdated {
    pub version: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct MemoUpdated {
    pub owner: ModuleInstanceId,
    pub index: u64,
}

pub type AttentionControlRequestMailbox = TopicMailbox<AttentionControlRequest>;
pub type AttentionControlRequestInbox = TopicInbox<AttentionControlRequest>;
pub type CognitionLogUpdatedMailbox = TopicMailbox<CognitionLogUpdated>;
pub type CognitionLogUpdatedInbox = TopicInbox<CognitionLogUpdated>;
pub type InteroceptiveUpdatedMailbox = TopicMailbox<InteroceptiveUpdated>;
pub type InteroceptiveUpdatedInbox = TopicInbox<InteroceptiveUpdated>;
pub type ActionAffordancesUpdatedMailbox = TopicMailbox<ActionAffordancesUpdated>;
pub type ActionAffordancesUpdatedInbox = TopicInbox<ActionAffordancesUpdated>;
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

    use nuillu_blackboard::{ActivationRatio, BlackboardCommand, ResourceAllocation};
    use nuillu_types::{ReplicaCapRange, builtin};

    use crate::test_support::{scoped, test_caps};

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
        let mut controller = scoped(&caps, builtin::allocation(), 0).attention_control_inbox();

        publisher
            .publish(AttentionControlRequest::new("find memories about rust"))
            .await
            .expect("attention-control topic should have subscribers");

        let envelope = controller
            .next_item()
            .await
            .expect("controller subscriber receives request");
        assert_eq!(envelope.sender.module, ticker_id());
        assert_eq!(
            envelope.body,
            AttentionControlRequest::new("find memories about rust")
        );
    }

    #[tokio::test]
    async fn attention_control_load_balances_across_active_controller_replicas() {
        let mut alloc = ResourceAllocation::default();
        alloc.set_activation(builtin::allocation(), ActivationRatio::ONE);
        let blackboard = Blackboard::with_allocation(alloc);
        blackboard
            .apply(BlackboardCommand::SetModulePolicies {
                policies: vec![(
                    builtin::allocation(),
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
        let mut controller_0 = scoped(&caps, builtin::allocation(), 0).attention_control_inbox();
        let mut controller_1 = scoped(&caps, builtin::allocation(), 1).attention_control_inbox();

        publisher
            .publish(AttentionControlRequest::new("first"))
            .await
            .unwrap();
        publisher
            .publish(AttentionControlRequest::new("second"))
            .await
            .unwrap();

        assert_eq!(
            controller_0.next_item().await.unwrap().body,
            AttentionControlRequest::new("first")
        );
        assert_eq!(
            controller_1.next_item().await.unwrap().body,
            AttentionControlRequest::new("second")
        );
    }

    #[tokio::test]
    async fn attention_control_routes_to_replica_zero_when_controller_is_inactive() {
        let mut alloc = ResourceAllocation::default();
        alloc.set_activation(builtin::allocation(), ActivationRatio::ZERO);
        let blackboard = Blackboard::with_allocation(alloc);
        blackboard
            .apply(BlackboardCommand::SetModulePolicies {
                policies: vec![(
                    builtin::allocation(),
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
        let mut controller_0 = scoped(&caps, builtin::allocation(), 0).attention_control_inbox();
        let mut controller_1 = scoped(&caps, builtin::allocation(), 1).attention_control_inbox();

        publisher
            .publish(AttentionControlRequest::new("active only"))
            .await
            .unwrap();

        assert_eq!(
            controller_0.next_item().await.unwrap().body,
            AttentionControlRequest::new("active only")
        );
        assert!(controller_1.take_ready_items().unwrap().items.is_empty());
    }
}
