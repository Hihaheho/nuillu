use std::fmt;
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
use crate::{
    REASONING_SYSTEM_PROMPT, format_identity_memory_seed, format_persistent_system_seed,
    seed_persistent_faculty_session,
};

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

#[derive(Clone, Copy, Debug)]
pub struct SessionAutoCompaction {
    pub config: SessionCompactionConfig,
    pub protected_prefix: SessionCompactionProtectedPrefix,
    pub compacted_prefix: &'static str,
    pub compaction_focus: &'static str,
}

impl SessionAutoCompaction {
    pub fn new(
        config: SessionCompactionConfig,
        protected_prefix: SessionCompactionProtectedPrefix,
        compacted_prefix: &'static str,
        compaction_focus: &'static str,
    ) -> Self {
        Self {
            config,
            protected_prefix,
            compacted_prefix,
            compaction_focus,
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
    pub reasoning: bool,
}

#[derive(Debug, thiserror::Error)]
pub enum SessionCheckpointError {
    #[error("session is missing persistent capability metadata")]
    MissingMetadata,
}

pub(crate) fn attach_persistent_session_metadata(
    session: &mut Session,
    owner: ModuleInstanceId,
    key: SessionKey,
    auto_compaction: Option<SessionAutoCompaction>,
    restored: bool,
    reasoning: bool,
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
        reasoning,
    });
}

pub(crate) fn restore_persistent_session_metadata(
    session: &mut Session,
    metadata: &PersistentSessionMetadata,
) {
    session.extensions_mut().insert(ModuleSessionMetadata {
        owner: metadata.owner.clone(),
        session_key: metadata.key.clone(),
    });
    session.extensions_mut().insert(metadata.clone());
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
    let system_prompt = system_prompt.into();
    let reasoning = persistent_session_metadata(session)
        .map(|metadata| metadata.reasoning)
        .unwrap_or(false);
    let should_seed = match persistent_session_metadata_mut(session) {
        Some(metadata) if metadata.seeded || metadata.restored => {
            metadata.seeded = true;
            false
        }
        Some(_) | None => true,
    };
    if !should_seed {
        ensure_combined_system_seed(session, &system_prompt, reasoning, identity_memories, now);
        return;
    }
    seed_persistent_faculty_session(session, system_prompt, reasoning, identity_memories, now);
    if let Some(metadata) = persistent_session_metadata_mut(session) {
        metadata.seeded = true;
    }
}

fn ensure_combined_system_seed(
    session: &mut Session,
    system_prompt: &str,
    reasoning: bool,
    identity_memories: &[IdentityMemoryRecord],
    now: DateTime<Utc>,
) {
    let items = session.input_mut().items_mut();
    let has_leading_system = leading_system_text(items).is_some();
    let has_reasoning = leading_system_text(items)
        .map(|text| text.contains(REASONING_SYSTEM_PROMPT))
        .unwrap_or(false);
    let has_legacy_identity = legacy_identity_seed_at(items, 1);
    if has_leading_system && !has_legacy_identity && (!reasoning || has_reasoning) {
        return;
    }

    let seed = ModelInputItem::text(
        InputMessageRole::System,
        format_persistent_system_seed(system_prompt.to_owned(), reasoning, identity_memories, now),
    );
    if has_leading_system {
        items[0] = seed;
    } else {
        items.insert(0, seed);
    }
    if legacy_identity_seed_at(items, 1) {
        items.remove(1);
    }
}

fn leading_system_text(items: &[ModelInputItem]) -> Option<&str> {
    let Some(ModelInputItem::Message {
        role: InputMessageRole::System,
        content,
    }) = items.first()
    else {
        return None;
    };
    match content.as_slice() {
        [MessageContent::Text(text)] => Some(text),
        _ => None,
    }
}

fn legacy_identity_seed_at(items: &[ModelInputItem], index: usize) -> bool {
    matches!(
        items.get(index),
        Some(ModelInputItem::Assistant(AssistantInputItem::Text(text)))
            if text.starts_with("What I already remember about myself")
    )
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
    let Some(seed) = format_identity_memory_seed(identity_memories, now) else {
        return;
    };
    let items = session.input_mut().items_mut();
    if let Some(system) = leading_system_text(items).map(str::to_owned) {
        if !system.contains("What I already remember about myself") {
            items[0] =
                ModelInputItem::text(InputMessageRole::System, format!("{system}\n\n{seed}"));
        }
        return;
    }
    session.push_system(seed);
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

    async fn delete_owner(&self, owner: &ModuleInstanceId) -> Result<u64, PortError>;
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

    async fn delete_owner(&self, _owner: &ModuleInstanceId) -> Result<u64, PortError> {
        Ok(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone as _;
    use lutum::{AssistantTurnItem, AssistantTurnView, RawJson};
    use nuillu_types::{MemoryContent, MemoryIndex, ReplicaIndex, builtin};

    fn now() -> DateTime<Utc> {
        Utc.with_ymd_and_hms(2026, 5, 11, 6, 23, 0).unwrap()
    }

    fn identity_memory() -> IdentityMemoryRecord {
        IdentityMemoryRecord {
            index: MemoryIndex::new("identity-1"),
            content: MemoryContent::new("The agent is named Nuillu."),
            occurred_at: None,
        }
    }

    fn attach_test_metadata(session: &mut Session, restored: bool, reasoning: bool) {
        attach_persistent_session_metadata(
            session,
            ModuleInstanceId::new(builtin::sensory(), ReplicaIndex::ZERO),
            SessionKey::new("main").unwrap(),
            None,
            restored,
            reasoning,
        );
    }

    fn leading_system_text_for_test(session: &Session) -> &str {
        let ModelInputItem::Message {
            role: InputMessageRole::System,
            content,
        } = &session.input().items()[0]
        else {
            panic!("expected leading system message");
        };
        let [MessageContent::Text(text)] = content.as_slice() else {
            panic!("expected leading system text");
        };
        text
    }

    fn occurrences(haystack: &str, needle: &str) -> usize {
        haystack.match_indices(needle).count()
    }

    #[test]
    fn reasoning_session_seeds_combined_system_message() {
        let mut session = Session::new();
        attach_test_metadata(&mut session, false, true);

        ensure_persistent_session_seeded(&mut session, "SYSTEM", &[identity_memory()], now());

        let items = session.input().items();
        assert_eq!(items.len(), 1);
        let system = leading_system_text_for_test(&session);
        assert!(system.starts_with("SYSTEM\n\n"));
        assert!(system.contains(REASONING_SYSTEM_PROMPT));
        assert!(system.contains("What I already remember about myself"));
        assert!(system.contains("The agent is named Nuillu."));
    }

    #[test]
    fn non_reasoning_session_omits_reasoning_prompt() {
        let mut session = Session::new();
        attach_test_metadata(&mut session, false, false);

        ensure_persistent_session_seeded(&mut session, "SYSTEM", &[identity_memory()], now());

        let system = leading_system_text_for_test(&session);
        assert!(!system.contains(REASONING_SYSTEM_PROMPT));
        assert!(system.contains("What I already remember about myself"));
    }

    #[test]
    fn restored_reasoning_session_backfills_without_duplication() {
        let mut session = Session::new();
        session.push_system("SYSTEM");
        session.push_assistant_text(
            "What I already remember about myself at 2026-05-11T06:23:00Z:\n- old identity",
        );
        session.push_user("history");
        attach_test_metadata(&mut session, true, true);

        ensure_persistent_session_seeded(&mut session, "SYSTEM", &[identity_memory()], now());
        ensure_persistent_session_seeded(&mut session, "SYSTEM", &[identity_memory()], now());

        let items = session.input().items();
        let system = leading_system_text_for_test(&session);
        assert_eq!(occurrences(system, REASONING_SYSTEM_PROMPT), 1);
        assert!(system.contains("The agent is named Nuillu."));
        assert!(!matches!(
            items.get(1),
            Some(ModelInputItem::Assistant(AssistantInputItem::Text(text)))
                if text.starts_with("What I already remember about myself")
        ));
        assert!(matches!(
            items.get(1),
            Some(ModelInputItem::Message {
                role: InputMessageRole::User,
                ..
            })
        ));
    }

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
