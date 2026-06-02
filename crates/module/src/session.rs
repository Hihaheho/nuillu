use std::cell::{Cell, RefCell, RefMut};
use std::fmt;
use std::rc::Rc;
use std::sync::Arc;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use lutum::{
    AssistantInputItem, Image, InputMessageRole, ItemView, MessageContent, ModelInput,
    ModelInputItem, RawJson, Session, ToolCallId, ToolCallItemView, ToolName, ToolResult,
    ToolResultItemView, TurnRole, TurnView,
};
use nuillu_blackboard::IdentityMemoryRecord;
use nuillu_types::ModuleInstanceId;
use serde::{Deserialize, Deserializer, Serialize, Serializer, de::Error as _};

use crate::ports::PortError;
use crate::session_compaction::{SessionCompactionConfig, SessionCompactionProtectedPrefix};
use crate::{format_identity_memory_seed, seed_persistent_faculty_session};

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct SessionKey(String);

impl SessionKey {
    pub fn new(value: impl Into<String>) -> Result<Self, PortError> {
        let value = value.into();
        validate_session_key(&value)?;
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for SessionKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

fn validate_session_key(value: &str) -> Result<(), PortError> {
    let mut chars = value.chars();
    let Some(first) = chars.next() else {
        return Err(PortError::InvalidInput(
            "session key must not be empty".into(),
        ));
    };
    if !first.is_ascii_lowercase() {
        return Err(PortError::InvalidInput(format!(
            "session key must start with a lowercase ASCII letter: {value}"
        )));
    }
    let mut previous_dash = false;
    for ch in std::iter::once(first).chain(chars) {
        if ch == '-' {
            if previous_dash {
                return Err(PortError::InvalidInput(format!(
                    "session key must not contain consecutive dashes: {value}"
                )));
            }
            previous_dash = true;
            continue;
        }
        previous_dash = false;
        if !ch.is_ascii_lowercase() && !ch.is_ascii_digit() {
            return Err(PortError::InvalidInput(format!(
                "session key must be kebab-case ASCII: {value}"
            )));
        }
    }
    if value.ends_with('-') {
        return Err(PortError::InvalidInput(format!(
            "session key must not end with a dash: {value}"
        )));
    }
    Ok(())
}

#[derive(Clone)]
pub struct ModuleSession {
    inner: Rc<ModuleSessionInner>,
}

struct ModuleSessionInner {
    owner: ModuleInstanceId,
    key: SessionKey,
    session: RefCell<Session>,
    dirty: Cell<bool>,
    restored: Cell<bool>,
    seeded: Cell<bool>,
}

#[derive(Clone, Copy, Debug)]
pub struct SessionAutoCompaction {
    pub config: SessionCompactionConfig,
    pub protected_prefix: SessionCompactionProtectedPrefix,
    pub compacted_prefix: &'static str,
    pub compaction_prompt: &'static str,
}

impl SessionAutoCompaction {
    pub fn new(
        config: SessionCompactionConfig,
        protected_prefix: SessionCompactionProtectedPrefix,
        compacted_prefix: &'static str,
        compaction_prompt: &'static str,
    ) -> Self {
        Self {
            config,
            protected_prefix,
            compacted_prefix,
            compaction_prompt,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModuleSessionMetadata {
    pub owner: ModuleInstanceId,
    pub session_key: SessionKey,
}

#[derive(Clone, Debug)]
pub(crate) struct PersistentSessionMetadata {
    pub owner: ModuleInstanceId,
    pub key: SessionKey,
    pub auto_compaction: Option<SessionAutoCompaction>,
    pub restored: bool,
    pub seeded: bool,
}

#[derive(Debug, thiserror::Error)]
pub enum SessionCheckpointError {
    #[error("session is missing persistent capability metadata")]
    MissingMetadata,
}

impl ModuleSession {
    pub(crate) fn new(owner: ModuleInstanceId, key: SessionKey) -> Self {
        Self {
            inner: Rc::new(ModuleSessionInner {
                owner,
                key,
                session: RefCell::new(Session::new()),
                dirty: Cell::new(false),
                restored: Cell::new(false),
                seeded: Cell::new(false),
            }),
        }
    }

    pub fn owner(&self) -> &ModuleInstanceId {
        &self.inner.owner
    }

    pub fn key(&self) -> &SessionKey {
        &self.inner.key
    }

    pub fn session_mut<R>(&self, f: impl FnOnce(&mut Session) -> R) -> R {
        self.inner.dirty.set(true);
        f(&mut self.inner.session.borrow_mut())
    }

    pub fn borrow_mut(&self) -> RefMut<'_, Session> {
        self.inner.dirty.set(true);
        self.inner.session.borrow_mut()
    }

    pub fn ensure_seeded(
        &self,
        system_prompt: impl Into<String>,
        identity_memories: &[IdentityMemoryRecord],
        now: DateTime<Utc>,
    ) {
        if self.inner.seeded.get() || self.inner.restored.get() {
            self.inner.seeded.set(true);
            return;
        }
        self.session_mut(|session| {
            seed_persistent_faculty_session(session, system_prompt, identity_memories, now);
        });
        self.inner.seeded.set(true);
    }

    pub fn push_identity_seed_if_absent(
        &self,
        identity_memories: &[IdentityMemoryRecord],
        now: DateTime<Utc>,
    ) {
        if self.inner.restored.get() {
            return;
        }
        if let Some(seed) = format_identity_memory_seed(identity_memories, now) {
            self.session_mut(|session| session.push_assistant_text(seed));
        }
    }

    pub(crate) fn restore(&self, snapshot: PersistedSessionSnapshot) {
        *self.inner.session.borrow_mut() = snapshot.into_session();
        self.inner.restored.set(true);
        self.inner.seeded.set(true);
        self.inner.dirty.set(false);
    }

    pub(crate) fn snapshot_if_dirty(&self) -> Option<PersistedSessionSnapshot> {
        if !self.inner.dirty.get() {
            return None;
        }
        Some(PersistedSessionSnapshot::from_session(
            &self.inner.session.borrow(),
        ))
    }

    pub(crate) fn mark_clean(&self) {
        self.inner.dirty.set(false);
    }
}

impl fmt::Debug for ModuleSession {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ModuleSession")
            .field("owner", &self.inner.owner)
            .field("key", &self.inner.key)
            .field("dirty", &self.inner.dirty.get())
            .field("restored", &self.inner.restored.get())
            .field("seeded", &self.inner.seeded.get())
            .finish()
    }
}

pub(crate) fn attach_persistent_session_metadata(
    session: &mut Session,
    owner: ModuleInstanceId,
    key: SessionKey,
    auto_compaction: Option<SessionAutoCompaction>,
    restored: bool,
) {
    session.extensions_mut().insert(ModuleSessionMetadata {
        owner: owner.clone(),
        session_key: key.clone(),
    });
    session.extensions_mut().insert(PersistentSessionMetadata {
        owner,
        key,
        auto_compaction,
        restored,
        seeded: restored,
    });
}

pub(crate) fn persistent_session_metadata(session: &Session) -> Option<&PersistentSessionMetadata> {
    session.extensions().get::<PersistentSessionMetadata>()
}

pub(crate) fn persistent_session_metadata_mut(
    session: &mut Session,
) -> Option<&mut PersistentSessionMetadata> {
    session
        .extensions_mut()
        .get_mut::<PersistentSessionMetadata>()
}

pub fn ensure_persistent_session_seeded(
    session: &mut Session,
    system_prompt: impl Into<String>,
    identity_memories: &[IdentityMemoryRecord],
    now: DateTime<Utc>,
) {
    let should_seed = match persistent_session_metadata_mut(session) {
        Some(metadata) if metadata.seeded || metadata.restored => {
            metadata.seeded = true;
            false
        }
        Some(_) | None => true,
    };
    if !should_seed {
        return;
    }
    seed_persistent_faculty_session(session, system_prompt, identity_memories, now);
    if let Some(metadata) = persistent_session_metadata_mut(session) {
        metadata.seeded = true;
    }
}

pub fn push_persistent_identity_seed_if_absent(
    session: &mut Session,
    identity_memories: &[IdentityMemoryRecord],
    now: DateTime<Utc>,
) {
    if persistent_session_metadata(session)
        .map(|metadata| metadata.restored)
        .unwrap_or(false)
    {
        return;
    }
    if let Some(seed) = format_identity_memory_seed(identity_memories, now) {
        session.push_assistant_text(seed);
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PersistedSessionSnapshot {
    pub version: u32,
    pub items: Vec<PersistedModelInputItem>,
}

impl PersistedSessionSnapshot {
    pub fn from_session(session: &Session) -> Self {
        Self::from_input(session.input())
    }

    pub fn from_input(input: &ModelInput) -> Self {
        Self {
            version: 1,
            items: input
                .items()
                .iter()
                .filter_map(PersistedModelInputItem::from_model_input_item)
                .collect(),
        }
    }

    pub fn into_session(self) -> Session {
        Session::from_input(self.into_input())
    }

    pub fn into_input(self) -> ModelInput {
        ModelInput::from_items(
            self.items
                .into_iter()
                .map(PersistedModelInputItem::into_model_input_item)
                .collect(),
        )
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum PersistedModelInputItem {
    Message {
        role: InputMessageRole,
        content: Vec<MessageContent>,
    },
    Assistant {
        item: AssistantInputItem,
    },
    ToolResult {
        result: PersistedToolResult,
    },
    Turn {
        turn: PersistedCommittedTurn,
    },
}

impl PersistedModelInputItem {
    fn from_model_input_item(item: &ModelInputItem) -> Option<Self> {
        match item {
            ModelInputItem::Message { role, content } => Some(Self::Message {
                role: *role,
                content: content.as_slice().to_vec(),
            }),
            ModelInputItem::Assistant(item) => Some(Self::Assistant { item: item.clone() }),
            ModelInputItem::ToolResult(result) => Some(Self::ToolResult {
                result: PersistedToolResult::from_tool_result(result),
            }),
            ModelInputItem::Turn(turn) if turn.ephemeral() => None,
            ModelInputItem::Turn(turn) => Some(Self::Turn {
                turn: PersistedCommittedTurn::from_turn(turn.as_ref()),
            }),
        }
    }

    fn into_model_input_item(self) -> ModelInputItem {
        match self {
            Self::Message { role, content } => {
                let mut content = content.into_iter().collect::<Vec<_>>();
                if content.is_empty() {
                    content.push(MessageContent::Text(String::new()));
                }
                let non_empty = lutum::NonEmpty::try_from_vec(content)
                    .expect("empty content was populated above");
                ModelInputItem::message(role, non_empty)
            }
            Self::Assistant { item } => ModelInputItem::assistant(item),
            Self::ToolResult { result } => ModelInputItem::tool_result(result.into_tool_result()),
            Self::Turn { turn } => ModelInputItem::turn(Arc::new(turn)),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PersistedToolResult {
    id: ToolCallId,
    name: ToolName,
    arguments: PersistedRawJson,
    result: PersistedRawJson,
}

impl PersistedToolResult {
    fn from_tool_result(result: &ToolResult) -> Self {
        Self {
            id: result.id.clone(),
            name: result.name.clone(),
            arguments: PersistedRawJson::from_raw_json(&result.arguments),
            result: PersistedRawJson::from_raw_json(&result.result),
        }
    }

    fn into_tool_result(self) -> ToolResult {
        ToolResult {
            id: self.id,
            name: self.name,
            arguments: self.arguments.into_raw_json(),
            result: self.result.into_raw_json(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PersistedCommittedTurn {
    role: PersistedTurnRole,
    items: Vec<PersistedTurnItem>,
}

impl PersistedCommittedTurn {
    fn from_turn(turn: &dyn TurnView) -> Self {
        let mut items = Vec::new();
        for index in 0..turn.item_count() {
            let Some(item) = turn.item_at(index) else {
                continue;
            };
            items.push(PersistedTurnItem::from_item(item));
        }
        Self {
            role: PersistedTurnRole::from_turn_role(turn.role()),
            items,
        }
    }
}

impl TurnView for PersistedCommittedTurn {
    fn role(&self) -> TurnRole {
        self.role.into_turn_role()
    }

    fn item_count(&self) -> usize {
        self.items.len()
    }

    fn item_at(&self, index: usize) -> Option<&dyn ItemView> {
        self.items.get(index).map(|item| item as &dyn ItemView)
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum PersistedTurnRole {
    System,
    Developer,
    User,
    Assistant,
}

impl PersistedTurnRole {
    fn from_turn_role(role: TurnRole) -> Self {
        match role {
            TurnRole::System => Self::System,
            TurnRole::Developer => Self::Developer,
            TurnRole::User => Self::User,
            TurnRole::Assistant => Self::Assistant,
        }
    }

    fn into_turn_role(self) -> TurnRole {
        match self {
            Self::System => TurnRole::System,
            Self::Developer => TurnRole::Developer,
            Self::User => TurnRole::User,
            Self::Assistant => TurnRole::Assistant,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum PersistedTurnItem {
    Text {
        text: String,
    },
    Image {
        image: Image,
    },
    Reasoning {
        text: String,
    },
    Refusal {
        text: String,
    },
    ToolCall {
        id: ToolCallId,
        name: ToolName,
        arguments: PersistedRawJson,
    },
    ToolResult {
        id: ToolCallId,
        name: ToolName,
        arguments: PersistedRawJson,
        result: PersistedRawJson,
    },
    Unknown,
}

impl PersistedTurnItem {
    fn from_item(item: &dyn ItemView) -> Self {
        if let Some(text) = item.as_text() {
            return Self::Text {
                text: text.to_string(),
            };
        }
        if let Some(text) = item.as_reasoning() {
            return Self::Reasoning {
                text: text.to_string(),
            };
        }
        if let Some(text) = item.as_refusal() {
            return Self::Refusal {
                text: text.to_string(),
            };
        }
        if let Some(tool) = item.as_tool_call() {
            return Self::ToolCall {
                id: tool.id.clone(),
                name: tool.name.clone(),
                arguments: PersistedRawJson::from_raw_json(tool.arguments),
            };
        }
        if let Some(tool) = item.as_tool_result() {
            return Self::ToolResult {
                id: tool.id.clone(),
                name: tool.name.clone(),
                arguments: PersistedRawJson::from_raw_json(tool.arguments),
                result: PersistedRawJson::from_raw_json(tool.result),
            };
        }
        Self::Unknown
    }
}

impl ItemView for PersistedTurnItem {
    fn as_text(&self) -> Option<&str> {
        match self {
            Self::Text { text } => Some(text),
            _ => None,
        }
    }

    fn as_reasoning(&self) -> Option<&str> {
        match self {
            Self::Reasoning { text } => Some(text),
            _ => None,
        }
    }

    fn as_refusal(&self) -> Option<&str> {
        match self {
            Self::Refusal { text } => Some(text),
            _ => None,
        }
    }

    fn as_tool_call(&self) -> Option<ToolCallItemView<'_>> {
        match self {
            Self::ToolCall {
                id,
                name,
                arguments,
            } => Some(ToolCallItemView {
                id,
                name,
                arguments: arguments.as_raw_json(),
            }),
            _ => None,
        }
    }

    fn as_tool_result(&self) -> Option<ToolResultItemView<'_>> {
        match self {
            Self::ToolResult {
                id,
                name,
                arguments,
                result,
            } => Some(ToolResultItemView {
                id,
                name,
                arguments: arguments.as_raw_json(),
                result: result.as_raw_json(),
            }),
            _ => None,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct PersistedRawJson {
    json: String,
    raw: RawJson,
}

impl PersistedRawJson {
    fn from_raw_json(raw: &RawJson) -> Self {
        Self {
            json: raw.get().to_owned(),
            raw: raw.clone(),
        }
    }

    fn parse(json: String) -> Result<Self, serde_json::Error> {
        let raw = RawJson::parse(json.clone())?;
        Ok(Self { json, raw })
    }

    fn as_raw_json(&self) -> &RawJson {
        &self.raw
    }

    fn into_raw_json(self) -> RawJson {
        self.raw
    }
}

impl Serialize for PersistedRawJson {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        #[derive(Serialize)]
        struct RawJsonWire<'a> {
            format: &'static str,
            json: &'a str,
        }

        RawJsonWire {
            format: "raw-json-v1",
            json: &self.json,
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for PersistedRawJson {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct RawJsonWire {
            format: String,
            json: String,
        }

        let wire = RawJsonWire::deserialize(deserializer)?;
        if wire.format != "raw-json-v1" {
            return Err(D::Error::custom(format!(
                "unsupported raw JSON snapshot format: {}",
                wire.format
            )));
        }
        Self::parse(wire.json).map_err(D::Error::custom)
    }
}

#[async_trait(?Send)]
pub trait SessionStore {
    async fn load(
        &self,
        owner: &ModuleInstanceId,
        key: &SessionKey,
    ) -> Result<Option<PersistedSessionSnapshot>, PortError>;

    async fn save(
        &self,
        owner: &ModuleInstanceId,
        key: &SessionKey,
        snapshot: &PersistedSessionSnapshot,
    ) -> Result<(), PortError>;
}

#[derive(Debug, Default)]
pub struct NoopSessionStore;

#[async_trait(?Send)]
impl SessionStore for NoopSessionStore {
    async fn load(
        &self,
        _owner: &ModuleInstanceId,
        _key: &SessionKey,
    ) -> Result<Option<PersistedSessionSnapshot>, PortError> {
        Ok(None)
    }

    async fn save(
        &self,
        _owner: &ModuleInstanceId,
        _key: &SessionKey,
        _snapshot: &PersistedSessionSnapshot,
    ) -> Result<(), PortError> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lutum::{AssistantTurnItem, AssistantTurnView, RawJson};

    #[test]
    fn session_snapshot_round_trips_basic_items_and_turns() {
        let tool_arguments = RawJson::parse(r#"{"x":1}"#).unwrap();
        let tool_result = RawJson::parse(r#"{"ok":true}"#).unwrap();
        let mut session = Session::new();
        session.push_system("system");
        session.push_user("user");
        session.push_assistant_reasoning("thinking");
        session.input_mut().push(ModelInputItem::turn(Arc::new(
            AssistantTurnView::from_items(&[
                AssistantTurnItem::Text("answer".into()),
                AssistantTurnItem::ToolCall {
                    id: ToolCallId::new("call-1"),
                    name: ToolName::new("do_work"),
                    arguments: tool_arguments.clone(),
                },
            ]),
        )));
        session
            .input_mut()
            .push(ModelInputItem::tool_result(ToolResult {
                id: ToolCallId::new("call-1"),
                name: ToolName::new("do_work"),
                arguments: tool_arguments,
                result: tool_result,
            }));

        let restored = PersistedSessionSnapshot::from_session(&session).into_session();
        let original = PersistedSessionSnapshot::from_session(&session);
        let round_trip = PersistedSessionSnapshot::from_session(&restored);

        assert_eq!(
            serde_json::to_value(round_trip).unwrap(),
            serde_json::to_value(&original).unwrap()
        );

        let json = serde_json::to_string(&original).unwrap();
        let restored_from_json = serde_json::from_str::<PersistedSessionSnapshot>(&json).unwrap();
        assert_eq!(
            serde_json::to_value(restored_from_json).unwrap(),
            serde_json::to_value(original).unwrap()
        );
    }

    #[test]
    fn session_key_rejects_non_kebab_case() {
        assert!(SessionKey::new("main").is_ok());
        assert!(SessionKey::new("abort-judgement").is_ok());
        assert!(SessionKey::new("Main").is_err());
        assert!(SessionKey::new("main_session").is_err());
        assert!(SessionKey::new("main--session").is_err());
    }
}
