use anyhow::{Context, Result};
use async_trait::async_trait;
use futures::StreamExt;
use lutum::{Session, StructuredTurnOutcome, TextTurnEvent};
use nuillu_module::{
    AllocationReader, AllocationUpdatedInbox, AttentionReader, AttentionStreamUpdatedInbox,
    LlmAccess, Memo, Module, UtteranceWriter,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

mod batch;

const DECISION_PROMPT: &str = r#"You are the speak module.
Read only the current cognitive attention-stream set and allocation guidance. Decide whether a
user-visible utterance is warranted right now. If guidance says summary/query work is still needed
or the attention stream lacks answer-ready content, wait silently and explain the wait in rationale.
Do not inspect memos, route work, write attention, or change allocation.
Return only raw JSON for the structured decision; do not wrap it in Markdown or code fences."#;

const GENERATION_PROMPT: &str = r#"You are the speak module.
Generate concise user-visible text from the current cognitive attention-stream set and allocation
guidance. Use only the provided attention context and the generation hint. If partial_utterance is present, continue that
utterance from exactly where it stopped; do not repeat, rewrite, or replace the already emitted
partial text. Do not mention hidden state or unavailable module results."#;

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
struct SpeakDecision {
    should_respond: bool,
    rationale: String,
    generation_hint: Option<String>,
}

struct GenerationDraft {
    generation_id: u64,
    sequence: u32,
    accumulated: String,
    generation_hint: Option<String>,
    rationale: String,
}

impl GenerationDraft {
    fn new(
        generation_id: u64,
        generation_hint: Option<String>,
        rationale: String,
    ) -> GenerationDraft {
        GenerationDraft {
            generation_id,
            sequence: 0,
            accumulated: String::new(),
            generation_hint,
            rationale,
        }
    }

    fn push_delta(&mut self, delta: &str) -> u32 {
        let sequence = self.sequence;
        self.accumulated.push_str(delta);
        self.sequence = self.sequence.wrapping_add(1);
        sequence
    }
}

fn generation_input(
    attention_json: serde_json::Value,
    allocation_json: serde_json::Value,
    draft: &GenerationDraft,
) -> serde_json::Value {
    let mut input = serde_json::json!({
        "attention_streams": attention_json,
        "allocation": allocation_json,
        "generation_hint": draft.generation_hint.as_deref(),
    });
    if !draft.accumulated.is_empty() {
        input["partial_utterance"] = serde_json::Value::String(draft.accumulated.clone());
    }
    input
}

pub struct SpeakModule {
    updates: AttentionStreamUpdatedInbox,
    allocation_updates: AllocationUpdatedInbox,
    attention: AttentionReader,
    allocation: AllocationReader,
    memo: Memo,
    utterance: UtteranceWriter,
    llm: LlmAccess,
}

impl SpeakModule {
    pub fn new(
        updates: AttentionStreamUpdatedInbox,
        allocation_updates: AllocationUpdatedInbox,
        attention: AttentionReader,
        allocation: AllocationReader,
        memo: Memo,
        utterance: UtteranceWriter,
        llm: LlmAccess,
    ) -> Self {
        Self {
            updates,
            allocation_updates,
            attention,
            allocation,
            memo,
            utterance,
            llm,
        }
    }

    #[tracing::instrument(skip_all, err(Debug, level = "warn"))]
    async fn activate(&mut self) -> Result<()> {
        let attention = self.attention.snapshot().await;
        let mut attention_json = attention.compact_json();
        let mut allocation_json = serde_json::to_value(self.allocation.snapshot().await)
            .context("serialize allocation for speak decision")?;

        let lutum = self.llm.lutum().await;
        let mut decision_session = Session::new(lutum.clone());
        decision_session.push_system(DECISION_PROMPT);
        decision_session.push_user(
            serde_json::json!({
                "attention_streams": attention_json,
                "allocation": allocation_json,
            })
            .to_string(),
        );

        let result = decision_session
            .structured_turn::<SpeakDecision>()
            .collect()
            .await
            .context("speak decision turn failed")?;

        let StructuredTurnOutcome::Structured(decision) = result.semantic else {
            anyhow::bail!("speak decision turn refused");
        };

        if !decision.should_respond {
            self.memo.write(decision.rationale).await;
            return Ok(());
        }

        let mut draft = GenerationDraft::new(
            self.utterance.next_generation_id(),
            decision.generation_hint,
            decision.rationale,
        );

        loop {
            let interrupted = self
                .stream_generation(attention_json, allocation_json, &mut draft)
                .await?;
            if !interrupted {
                return Ok(());
            }
            attention_json = self.attention.snapshot().await.compact_json();
            allocation_json = serde_json::to_value(self.allocation.snapshot().await)
                .context("serialize allocation for speak restart")?;
        }
    }

    async fn stream_generation(
        &mut self,
        attention_json: serde_json::Value,
        allocation_json: serde_json::Value,
        draft: &mut GenerationDraft,
    ) -> Result<bool> {
        let lutum = self.llm.lutum().await;
        let mut session = Session::new(lutum);
        session.push_system(GENERATION_PROMPT);
        session.push_user(generation_input(attention_json, allocation_json, draft).to_string());

        let mut stream = session
            .text_turn()
            .stream()
            .await
            .context("speak generation stream failed")?;

        loop {
            tokio::select! {
                event = stream.next() => {
                    match event {
                        Some(Ok(TextTurnEvent::TextDelta { delta })) => {
                            let sequence = draft.push_delta(&delta);
                            self.utterance
                                .emit_delta(draft.generation_id, sequence, delta)
                                .await;
                        }
                        Some(Ok(TextTurnEvent::WillRetry { .. })) => {
                            return Ok(true);
                        }
                        Some(Ok(TextTurnEvent::Completed { .. })) | None => {
                            let text = draft.accumulated.trim().to_owned();
                            self.memo
                                .write(serde_json::json!({
                                    "utterance": text,
                                    "rationale": draft.rationale.as_str(),
                                }).to_string())
                                .await;
                            if !text.is_empty() {
                                self.utterance.emit(text).await;
                            }
                            return Ok(false);
                        }
                        Some(Ok(_)) => {}
                        Some(Err(error)) => return Err(error).context("speak generation stream event failed"),
                    }
                }
                update = self.updates.next_item() => {
                    let _ = update?;
                    let _ = self.updates.take_ready_items()?;
                    let _ = self.allocation_updates.take_ready_items()?;
                    return Ok(true);
                }
                update = self.allocation_updates.next_item() => {
                    let _ = update?;
                    let _ = self.allocation_updates.take_ready_items()?;
                    let _ = self.updates.take_ready_items()?;
                    return Ok(true);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fresh_generation_omits_partial_utterance() {
        let draft = GenerationDraft::new(7, Some("be concise".into()), "respond".into());

        let input = generation_input(
            serde_json::json!({"streams": []}),
            serde_json::json!({"speak": {"guidance": "respond"}}),
            &draft,
        );

        assert_eq!(draft.generation_id, 7);
        assert_eq!(draft.sequence, 0);
        assert_eq!(input["generation_hint"], "be concise");
        assert!(input.get("partial_utterance").is_none());
    }

    #[test]
    fn resumed_generation_keeps_id_sequence_and_partial_utterance() {
        let mut draft = GenerationDraft::new(11, None, "respond".into());

        assert_eq!(draft.push_delta("hello "), 0);
        assert_eq!(draft.push_delta("world"), 1);
        let input = generation_input(
            serde_json::json!({"streams": []}),
            serde_json::json!({"speak": {"guidance": "respond"}}),
            &draft,
        );

        assert_eq!(draft.generation_id, 11);
        assert_eq!(draft.sequence, 2);
        assert_eq!(draft.accumulated, "hello world");
        assert_eq!(input["partial_utterance"], "hello world");
    }
}

#[async_trait(?Send)]
impl Module for SpeakModule {
    type Batch = ();

    async fn next_batch(&mut self) -> Result<Self::Batch> {
        SpeakModule::next_batch(self).await
    }

    async fn activate(&mut self, _batch: &Self::Batch) -> Result<()> {
        SpeakModule::activate(self).await
    }
}
