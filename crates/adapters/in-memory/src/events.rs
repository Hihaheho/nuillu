use std::sync::Mutex;

use async_trait::async_trait;
use nuillu_module::ports::PortError;
use nuillu_storage::{
    AmbientSensorySnapshotRecord, AmbientSensorySnapshotStore, ExternalActionEventRecord,
    ExternalActionEventStatus, ExternalActionEventStore, LlmTranscriptStore,
    LlmTranscriptTurnRecord, NewAmbientSensorySnapshot, NewExternalActionEvent,
    NewLlmTranscriptTurn, NewOneShotSensoryInput, NewUtteranceEvent, OneShotSensoryInputRecord,
    OneShotSensoryInputStore, UtteranceEventRecord, UtteranceEventStore,
};

fn lock_error(name: &str) -> PortError {
    PortError::Backend(format!("{name} lock poisoned"))
}

fn now_ms() -> i64 {
    chrono::Utc::now().timestamp_millis()
}

fn recent_page<T: Clone>(records: &[T], offset: usize, limit: usize) -> Vec<T> {
    if limit == 0 {
        return Vec::new();
    }
    let mut page = records
        .iter()
        .rev()
        .skip(offset)
        .take(limit)
        .cloned()
        .collect::<Vec<_>>();
    page.reverse();
    page
}

#[derive(Debug, Default)]
pub struct InMemoryLlmTranscriptStore {
    state: Mutex<AppendState<LlmTranscriptTurnRecord>>,
}

impl InMemoryLlmTranscriptStore {
    pub fn new() -> Self {
        Self::default()
    }
}

#[async_trait]
impl LlmTranscriptStore for InMemoryLlmTranscriptStore {
    async fn insert_completed_turn(&self, turn: NewLlmTranscriptTurn) -> Result<(), PortError> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| lock_error("LLM transcript store"))?;
        state.records.retain(|record| {
            record.server_session_id != turn.server_session_id || record.turn_id != turn.turn_id
        });
        let record = LlmTranscriptTurnRecord {
            id: state.next_id(),
            server_session_id: turn.server_session_id,
            turn_id: turn.turn_id,
            owner: turn.owner,
            owner_module: turn.owner_module,
            owner_replica: turn.owner_replica,
            tier: turn.tier,
            source: turn.source,
            session_key: turn.session_key,
            operation: turn.operation,
            started_at_ms: turn.started_at_ms,
            completed_at_ms: turn.completed_at_ms,
            trace_json: turn.trace_json,
        };
        state.records.push(record);
        Ok(())
    }

    async fn completed_turns_page(
        &self,
        offset: usize,
        limit: usize,
    ) -> Result<Vec<LlmTranscriptTurnRecord>, PortError> {
        let state = self
            .state
            .lock()
            .map_err(|_| lock_error("LLM transcript store"))?;
        Ok(recent_page(&state.records, offset, limit))
    }

    async fn prune_completed_turns(&self, keep: usize) -> Result<(), PortError> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| lock_error("LLM transcript store"))?;
        let remove = state.records.len().saturating_sub(keep);
        state.records.drain(..remove);
        Ok(())
    }
}

#[derive(Debug)]
struct AppendState<T> {
    records: Vec<T>,
    next_id: i64,
}

impl<T> Default for AppendState<T> {
    fn default() -> Self {
        Self {
            records: Vec::new(),
            next_id: 0,
        }
    }
}

impl<T> AppendState<T> {
    fn next_id(&mut self) -> i64 {
        self.next_id = self.next_id.saturating_add(1);
        self.next_id
    }
}

#[derive(Debug, Default)]
pub struct InMemoryOneShotSensoryInputStore {
    state: Mutex<AppendState<OneShotSensoryInputRecord>>,
}

impl InMemoryOneShotSensoryInputStore {
    pub fn new() -> Self {
        Self::default()
    }
}

#[async_trait(?Send)]
impl OneShotSensoryInputStore for InMemoryOneShotSensoryInputStore {
    async fn append(
        &self,
        input: NewOneShotSensoryInput,
    ) -> Result<OneShotSensoryInputRecord, PortError> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| lock_error("one-shot sensory input store"))?;
        let record = OneShotSensoryInputRecord {
            id: state.next_id(),
            server_session_id: input.server_session_id,
            modality: input.modality,
            direction: input.direction,
            content: input.content,
            observed_at_ms: input.observed_at_ms,
            created_at_ms: now_ms(),
        };
        state.records.push(record.clone());
        Ok(record)
    }

    async fn recent_page(
        &self,
        offset: usize,
        limit: usize,
    ) -> Result<Vec<OneShotSensoryInputRecord>, PortError> {
        let state = self
            .state
            .lock()
            .map_err(|_| lock_error("one-shot sensory input store"))?;
        Ok(recent_page(&state.records, offset, limit))
    }
}

#[derive(Debug, Default)]
pub struct InMemoryAmbientSensorySnapshotStore {
    state: Mutex<AppendState<AmbientSensorySnapshotRecord>>,
}

impl InMemoryAmbientSensorySnapshotStore {
    pub fn new() -> Self {
        Self::default()
    }
}

#[async_trait(?Send)]
impl AmbientSensorySnapshotStore for InMemoryAmbientSensorySnapshotStore {
    async fn append(
        &self,
        snapshot: NewAmbientSensorySnapshot,
    ) -> Result<AmbientSensorySnapshotRecord, PortError> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| lock_error("ambient sensory snapshot store"))?;
        let record = AmbientSensorySnapshotRecord {
            id: state.next_id(),
            server_session_id: snapshot.server_session_id,
            entries: snapshot.entries,
            observed_at_ms: snapshot.observed_at_ms,
            created_at_ms: now_ms(),
        };
        state.records.push(record.clone());
        Ok(record)
    }

    async fn recent_page(
        &self,
        offset: usize,
        limit: usize,
    ) -> Result<Vec<AmbientSensorySnapshotRecord>, PortError> {
        let state = self
            .state
            .lock()
            .map_err(|_| lock_error("ambient sensory snapshot store"))?;
        Ok(recent_page(&state.records, offset, limit))
    }
}

#[derive(Debug, Default)]
pub struct InMemoryUtteranceEventStore {
    state: Mutex<AppendState<UtteranceEventRecord>>,
}

impl InMemoryUtteranceEventStore {
    pub fn new() -> Self {
        Self::default()
    }
}

#[async_trait(?Send)]
impl UtteranceEventStore for InMemoryUtteranceEventStore {
    async fn append(&self, event: NewUtteranceEvent) -> Result<UtteranceEventRecord, PortError> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| lock_error("utterance event store"))?;
        let record = UtteranceEventRecord {
            id: state.next_id(),
            server_session_id: event.server_session_id,
            event_kind: event.event_kind,
            sender: event.sender,
            target: event.target,
            generation_id: event.generation_id,
            sequence: event.sequence,
            content: event.content,
            reason: event.reason,
            occurred_at_ms: event.occurred_at_ms,
            created_at_ms: now_ms(),
        };
        state.records.push(record.clone());
        Ok(record)
    }

    async fn recent_page(
        &self,
        offset: usize,
        limit: usize,
    ) -> Result<Vec<UtteranceEventRecord>, PortError> {
        let state = self
            .state
            .lock()
            .map_err(|_| lock_error("utterance event store"))?;
        Ok(recent_page(&state.records, offset, limit))
    }
}

#[derive(Debug, Default)]
pub struct InMemoryExternalActionEventStore {
    state: Mutex<AppendState<ExternalActionEventRecord>>,
}

impl InMemoryExternalActionEventStore {
    pub fn new() -> Self {
        Self::default()
    }
}

#[async_trait(?Send)]
impl ExternalActionEventStore for InMemoryExternalActionEventStore {
    async fn append_pending(
        &self,
        event: NewExternalActionEvent,
    ) -> Result<ExternalActionEventRecord, PortError> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| lock_error("external action event store"))?;
        let created_at_ms = now_ms();
        let record = ExternalActionEventRecord {
            id: state.next_id(),
            server_session_id: event.server_session_id,
            invocation_id: event.invocation_id,
            invoked_by: event.invoked_by,
            action_id: event.action_id,
            arguments: event.arguments,
            status: ExternalActionEventStatus::Pending,
            accepted: None,
            message: None,
            requested_at_ms: event.requested_at_ms,
            completed_at_ms: None,
            created_at_ms,
            updated_at_ms: created_at_ms,
        };
        state.records.push(record.clone());
        Ok(record)
    }

    async fn complete(
        &self,
        id: i64,
        accepted: bool,
        message: String,
        completed_at_ms: i64,
    ) -> Result<ExternalActionEventRecord, PortError> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| lock_error("external action event store"))?;
        let record = state
            .records
            .iter_mut()
            .find(|record| record.id == id)
            .ok_or_else(|| PortError::NotFound(format!("external action event not found: {id}")))?;
        record.status = ExternalActionEventStatus::Completed;
        record.accepted = Some(accepted);
        record.message = Some(message);
        record.completed_at_ms = Some(completed_at_ms);
        record.updated_at_ms = now_ms();
        Ok(record.clone())
    }

    async fn recent_page(
        &self,
        offset: usize,
        limit: usize,
    ) -> Result<Vec<ExternalActionEventRecord>, PortError> {
        let state = self
            .state
            .lock()
            .map_err(|_| lock_error("external action event store"))?;
        Ok(recent_page(&state.records, offset, limit))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nuillu_module::AmbientSensoryEntry;
    use nuillu_module::SensoryModality;
    use nuillu_storage::{ExternalActionEventStatus, UtteranceEventKind};
    use nuillu_types::{ModuleInstanceId, ReplicaIndex, builtin};

    fn owner() -> ModuleInstanceId {
        ModuleInstanceId::new(builtin::speak(), ReplicaIndex::ZERO)
    }

    #[tokio::test(flavor = "current_thread")]
    async fn append_only_event_stores_page_from_the_newest_window() {
        let one_shot = InMemoryOneShotSensoryInputStore::new();
        for content in ["first", "second", "third"] {
            one_shot
                .append(NewOneShotSensoryInput {
                    server_session_id: "session".into(),
                    modality: "vision".into(),
                    direction: None,
                    content: content.into(),
                    observed_at_ms: 1,
                })
                .await
                .unwrap();
        }
        assert_eq!(
            one_shot
                .recent_page(1, 2)
                .await
                .unwrap()
                .iter()
                .map(|record| record.content.as_str())
                .collect::<Vec<_>>(),
            vec!["first", "second"]
        );

        let ambient = InMemoryAmbientSensorySnapshotStore::new();
        let ambient_record = ambient
            .append(NewAmbientSensorySnapshot {
                server_session_id: "session".into(),
                entries: vec![AmbientSensoryEntry {
                    id: "room".into(),
                    modality: SensoryModality::Vision,
                    content: "bright".into(),
                }],
                observed_at_ms: 2,
            })
            .await
            .unwrap();
        assert_eq!(ambient.recent(1).await.unwrap(), vec![ambient_record]);

        let utterances = InMemoryUtteranceEventStore::new();
        let utterance_record = utterances
            .append(NewUtteranceEvent {
                server_session_id: "session".into(),
                event_kind: UtteranceEventKind::Completed,
                sender: owner(),
                target: "user".into(),
                generation_id: 4,
                sequence: 2,
                content: "hello".into(),
                reason: None,
                occurred_at_ms: 3,
            })
            .await
            .unwrap();
        assert_eq!(utterances.recent(1).await.unwrap(), vec![utterance_record]);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn transcript_upsert_and_external_action_completion_match_store_contracts() {
        let transcripts = InMemoryLlmTranscriptStore::new();
        for trace in [
            serde_json::json!({"attempt": 1}),
            serde_json::json!({"attempt": 2}),
        ] {
            transcripts
                .insert_completed_turn(NewLlmTranscriptTurn {
                    server_session_id: "session".into(),
                    turn_id: "turn".into(),
                    owner: "owner".into(),
                    owner_module: "memory".into(),
                    owner_replica: 0,
                    tier: "default".into(),
                    source: "module".into(),
                    session_key: None,
                    operation: "completion".into(),
                    started_at_ms: 1,
                    completed_at_ms: 2,
                    trace_json: trace,
                })
                .await
                .unwrap();
        }
        let records = transcripts.recent_completed_turns(2).await.unwrap();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].trace_json, serde_json::json!({"attempt": 2}));

        let actions = InMemoryExternalActionEventStore::new();
        let pending = actions
            .append_pending(NewExternalActionEvent {
                server_session_id: "session".into(),
                invocation_id: "invoke-1".into(),
                invoked_by: owner(),
                action_id: "wave".into(),
                arguments: serde_json::json!({}),
                requested_at_ms: 3,
            })
            .await
            .unwrap();
        let completed = actions
            .complete(pending.id, true, "done".into(), 4)
            .await
            .unwrap();
        assert_eq!(completed.status, ExternalActionEventStatus::Completed);
        assert_eq!(completed.accepted, Some(true));
        assert_eq!(completed.completed_at_ms, Some(4));
    }
}
