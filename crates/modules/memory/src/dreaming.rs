use anyhow::{Context, Result};
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use lutum::ModelInput;
use nuillu_blackboard::{CognitionLogEntryRecord, MemoLogRecord};
use nuillu_module::{
    AllocationReader, BlackboardReader, InteroceptiveUpdatedInbox, LlmAccess, Memo, Module,
    compact_llm_context_text, format_policy_system_prompt,
};
use nuillu_types::builtin;

const MAX_DREAM_MEMO_RECORDS: usize = 12;
const MAX_DREAM_COGNITION_RECORDS: usize = 8;
const DREAM_MEMO_CHARS: usize = 700;
const DREAM_COGNITION_CHARS: usize = 800;
const DREAM_CONTEXT_MAX_CHARS: usize = 7_000;
const DREAM_TURN_MAX_OUTPUT_TOKENS: u32 = 768;

const SYSTEM_PROMPT: &str = r#"You are the dreaming module.
You run a REM-like internal dream simulation. Use upstream faculty notes from query-memory,
predict, and self-model, plus recent non-dream cognition, as seeds for free associative thought.
Do not retrieve memory directly, ask other modules directly, or treat any seed as final truth.

Produce one short associative counterfactual, daydream, or scenario that may help future planning,
surprise, self-understanding, or memory reweighting. This is not a verified fact and not an outward
reply. Do not repeat the logs faithfully. Do not violate core policies. Keep the memo text under
200 Japanese characters or 120 English words.

Write only the cognitive memo text. Start useful output with "Internal dream, not a verified fact:".
If the surfaced seeds are not enough for a useful internal simulation, write nothing."#;

pub struct DreamingModule {
    owner: nuillu_types::ModuleId,
    interoception_updates: InteroceptiveUpdatedInbox,
    allocation: AllocationReader,
    blackboard: BlackboardReader,
    memo: Memo,
    llm: LlmAccess,
}

impl DreamingModule {
    pub fn new(
        interoception_updates: InteroceptiveUpdatedInbox,
        allocation: AllocationReader,
        blackboard: BlackboardReader,
        memo: Memo,
        llm: LlmAccess,
    ) -> Self {
        Self {
            owner: nuillu_types::ModuleId::new(<Self as Module>::id())
                .expect("dreaming id is valid"),
            interoception_updates,
            allocation,
            blackboard,
            memo,
            llm,
        }
    }

    #[tracing::instrument(skip_all, err(Debug, level = "warn"))]
    async fn activate(&mut self, cx: &nuillu_module::ActivateCx<'_>) -> Result<()> {
        let allocation = self.allocation.snapshot().await;
        if allocation.activation_for(&self.owner) == nuillu_blackboard::ActivationRatio::ZERO {
            return Ok(());
        }

        let (seed_memos, recent_cognition) = self
            .blackboard
            .read(|bb| {
                let mut seed_memos = bb
                    .recent_memo_logs()
                    .into_iter()
                    .filter(is_dream_seed_memo)
                    .collect::<Vec<_>>();
                keep_latest(&mut seed_memos, MAX_DREAM_MEMO_RECORDS);

                let mut recent_cognition = bb
                    .unread_cognition_log_entries(None)
                    .into_iter()
                    .filter(|record| !is_dreaming_cognition(record))
                    .collect::<Vec<_>>();
                recent_cognition.sort_by_key(|record| record.index);
                keep_latest(&mut recent_cognition, MAX_DREAM_COGNITION_RECORDS);

                (seed_memos, recent_cognition)
            })
            .await;
        if seed_memos.is_empty() && recent_cognition.is_empty() {
            return Ok(());
        }

        let input = ModelInput::new()
            .system(nuillu_module::format_system_seed(
                format_policy_system_prompt(SYSTEM_PROMPT, cx.core_policies()),
                false,
                cx.identity_memories(),
                cx.now(),
            ))
            .user(format_dreaming_context(
                &seed_memos,
                &recent_cognition,
                cx.now(),
            ));

        let lutum = self.llm.lutum().await;
        let result = lutum
            .text_turn(input)
            .max_output_tokens(DREAM_TURN_MAX_OUTPUT_TOKENS)
            .collect()
            .await
            .context("dreaming text turn failed")?;
        if let Some(memo) = prepare_dream_memo(result.assistant_text()) {
            self.memo.write_cognitive(memo).await;
        }
        Ok(())
    }

    async fn next_batch(&mut self) -> Result<()> {
        let _ = self.interoception_updates.next_item().await?;
        let _ = self.interoception_updates.take_ready_items()?;
        Ok(())
    }
}

fn is_dream_seed_memo(record: &MemoLogRecord) -> bool {
    record.cognitive
        && (record.owner.module == builtin::query_memory()
            || record.owner.module == builtin::predict()
            || record.owner.module == builtin::self_model())
}

fn is_dreaming_cognition(record: &CognitionLogEntryRecord) -> bool {
    record.source.module == builtin::dreaming()
        || record.entry.origin.owner.module == builtin::dreaming()
}

fn keep_latest<T>(records: &mut Vec<T>, max_records: usize) {
    if records.len() > max_records {
        let dropped = records.len() - max_records;
        records.drain(0..dropped);
    }
}

fn format_dreaming_context(
    seed_memos: &[MemoLogRecord],
    recent_cognition: &[CognitionLogEntryRecord],
    now: DateTime<Utc>,
) -> String {
    let mut out = String::from("Dreaming seed material.");
    out.push_str("\n\nUpstream faculty notes:");
    if seed_memos.is_empty() {
        out.push_str("\n- none surfaced");
    } else {
        for record in seed_memos {
            push_bounded_line(
                &mut out,
                DREAM_CONTEXT_MAX_CHARS,
                format!(
                    "- {} at {}: {}",
                    record.owner.module.as_str(),
                    record.written_at.to_rfc3339(),
                    compact_llm_context_text(&record.content, DREAM_MEMO_CHARS)
                ),
            );
        }
    }

    push_bounded_line(
        &mut out,
        DREAM_CONTEXT_MAX_CHARS,
        format!("\nRecent non-dream cognition at {}:", now.to_rfc3339()),
    );
    let non_dream_cognition = recent_cognition
        .iter()
        .filter(|record| !is_dreaming_cognition(record))
        .collect::<Vec<_>>();
    if non_dream_cognition.is_empty() {
        push_bounded_line(
            &mut out,
            DREAM_CONTEXT_MAX_CHARS,
            "- none surfaced".to_owned(),
        );
    } else {
        for record in non_dream_cognition {
            push_bounded_line(
                &mut out,
                DREAM_CONTEXT_MAX_CHARS,
                format!(
                    "- {}: {}",
                    record.source.module.as_str(),
                    compact_llm_context_text(&record.entry.text, DREAM_COGNITION_CHARS)
                ),
            );
        }
    }

    push_bounded_line(
        &mut out,
        DREAM_CONTEXT_MAX_CHARS,
        "\nUse these as associative seeds only. Do not state them as verified facts.".to_owned(),
    );
    out
}

fn push_bounded_line(out: &mut String, max_chars: usize, line: String) {
    if out.chars().count() >= max_chars {
        return;
    }
    let remaining = max_chars.saturating_sub(out.chars().count());
    out.push('\n');
    if line.chars().count() <= remaining {
        out.push_str(&line);
    } else {
        out.push_str(&compact_llm_context_text(&line, remaining));
    }
}

fn prepare_dream_memo(text: String) -> Option<String> {
    let text = text.trim();
    if text.is_empty() {
        return None;
    }
    Some(text.to_owned())
}

#[async_trait(?Send)]
impl Module for DreamingModule {
    type Batch = ();

    fn id() -> &'static str {
        "dreaming"
    }

    fn peer_context() -> Option<&'static str> {
        None
    }

    async fn next_batch(&mut self) -> Result<Self::Batch> {
        DreamingModule::next_batch(self).await
    }

    async fn activate(
        &mut self,
        cx: &nuillu_module::ActivateCx<'_>,
        _batch: &Self::Batch,
    ) -> Result<()> {
        DreamingModule::activate(self, cx).await
    }
}

#[cfg(test)]
mod tests {
    use chrono::TimeZone;
    use nuillu_blackboard::{CognitionLogEntry, CognitionLogOrigin};
    use nuillu_types::{ModuleInstanceId, ReplicaIndex};

    use super::*;

    fn now() -> DateTime<Utc> {
        Utc.with_ymd_and_hms(2026, 6, 14, 12, 0, 0).unwrap()
    }

    fn memo(module: nuillu_types::ModuleId, index: u64, content: &str) -> MemoLogRecord {
        MemoLogRecord {
            owner: ModuleInstanceId::new(module, ReplicaIndex::ZERO),
            index,
            written_at: now(),
            content: content.to_owned(),
            cognitive: true,
        }
    }

    fn cognition(
        module: nuillu_types::ModuleId,
        index: u64,
        text: &str,
    ) -> CognitionLogEntryRecord {
        let source = ModuleInstanceId::new(module, ReplicaIndex::ZERO);
        CognitionLogEntryRecord {
            index,
            source: source.clone(),
            entry: CognitionLogEntry {
                at: now(),
                text: text.to_owned(),
                origin: CognitionLogOrigin::direct(source),
            },
        }
    }

    #[test]
    fn dream_memo_uses_assistant_text_directly() {
        let memo = prepare_dream_memo(
            "  Internal dream, not a verified fact: associative scenario  ".to_owned(),
        );

        assert_eq!(
            memo,
            Some("Internal dream, not a verified fact: associative scenario".to_owned())
        );
        assert_eq!(prepare_dream_memo("  \n\t ".to_owned()), None);
    }

    #[test]
    fn dream_seed_memos_are_limited_to_upstream_cognitive_faculties() {
        let mut non_cognitive = memo(builtin::query_memory(), 0, "hidden retrieval");
        non_cognitive.cognitive = false;

        assert!(is_dream_seed_memo(&memo(
            builtin::query_memory(),
            1,
            "memory evidence"
        )));
        assert!(is_dream_seed_memo(&memo(
            builtin::predict(),
            2,
            "near future expectation"
        )));
        assert!(is_dream_seed_memo(&memo(
            builtin::self_model(),
            3,
            "self-state"
        )));
        assert!(!is_dream_seed_memo(&memo(
            builtin::sensory(),
            4,
            "direct sensory note"
        )));
        assert!(!is_dream_seed_memo(&non_cognitive));
    }

    #[test]
    fn dreaming_context_uses_upstream_memos_and_non_dream_cognition() {
        let context = format_dreaming_context(
            &[
                memo(builtin::query_memory(), 0, "Ryo once liked quiet rooms."),
                memo(builtin::predict(), 1, "The room may stay quiet."),
                memo(builtin::self_model(), 2, "I feel uncertain."),
            ],
            &[
                cognition(builtin::cognition_gate(), 0, "Ryo asked a question."),
                cognition(
                    builtin::dreaming(),
                    1,
                    "Internal dream, not a verified fact: old.",
                ),
            ],
            now(),
        );

        assert!(context.contains("Upstream faculty notes:"));
        assert!(context.contains("query-memory"));
        assert!(context.contains("predict"));
        assert!(context.contains("self-model"));
        assert!(context.contains("Ryo asked a question."));
        assert!(!context.contains("old."));
        assert!(context.contains("associative seeds only"));
    }
}
