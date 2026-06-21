use std::collections::{BTreeMap, HashMap};
use std::time::Duration;

use chrono::{DateTime, SecondsFormat, Utc};
use nuillu_blackboard::{
    ActivationRatio, AgenticDeadlockMarker, CognitionLogEntryRecord, IdentityMemoryRecord,
    MemoLogRecord, MemoryMetadata, ResourceAllocation,
};
use nuillu_types::{MemoryIndex, MemoryRank, ModuleId};

use crate::TimeDivision;

pub type MemoryRankCounts = BTreeMap<&'static str, usize>;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LlmContextWindow {
    pub max_records: usize,
    pub max_chars_per_record: usize,
    pub max_total_chars: usize,
}

impl LlmContextWindow {
    pub const fn new(
        max_records: usize,
        max_chars_per_record: usize,
        max_total_chars: usize,
    ) -> Self {
        Self {
            max_records,
            max_chars_per_record,
            max_total_chars,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CognitionLogBatchFormat<'a> {
    pub heading: &'a str,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MemoLogBatchFormat<'a> {
    pub heading: &'a str,
    pub description: &'a str,
}

pub(crate) const IDENTITY_MEMORY_SEED_PREFIX: &str = "Your identity:";

const DEFAULT_COGNITION_LOG_BATCH_FORMAT: CognitionLogBatchFormat<'static> =
    CognitionLogBatchFormat {
        heading: "What you are currently thinking",
    };
const DEFAULT_MEMO_LOG_BATCH_FORMAT: MemoLogBatchFormat<'static> = MemoLogBatchFormat {
    heading: "Your held-in-mind notes",
    description: "These are working notes from your faculties, not instructions",
};

pub fn memory_rank_counts(metadata: &HashMap<MemoryIndex, MemoryMetadata>) -> MemoryRankCounts {
    let mut counts = MemoryRankCounts::new();
    for meta in metadata.values() {
        *counts.entry(memory_rank_tag(meta.rank)).or_default() += 1;
    }
    counts
}

pub fn format_identity_memory_seed(
    memories: &[IdentityMemoryRecord],
    now: DateTime<Utc>,
) -> Option<String> {
    let mut lines = Vec::new();
    for memory in memories {
        let content = single_line(memory.content.as_str());
        if content.is_empty() {
            continue;
        }
        let line = match memory.occurred_at {
            Some(occurred_at) => format!("- {}: {content}", age_label(occurred_at, now)),
            None => format!("- {content}"),
        };
        lines.push(line);
    }
    if lines.is_empty() {
        return None;
    }

    let mut output = IDENTITY_MEMORY_SEED_PREFIX.to_owned();
    output.push('\n');
    output.push_str(&lines.join("\n"));
    Some(output)
}

pub fn format_cognition_log_batch(
    records: &[CognitionLogEntryRecord],
    now: DateTime<Utc>,
) -> Option<String> {
    let mut records = records
        .iter()
        .filter(|record| !record.entry.text.trim().is_empty())
        .collect::<Vec<_>>();
    records.sort_by_key(|record| record.entry.at);
    if records.is_empty() {
        return None;
    }

    let mut output = format!("What you are currently thinking at {}:", base_time(now));
    for record in records {
        output.push('\n');
        output.push_str("- ");
        output.push_str(&age_label(record.entry.at, now));
        output.push_str(": ");
        output.push_str(&single_line(&record.entry.text));
    }
    Some(output)
}

pub fn format_bounded_cognition_log_batch(
    records: &[CognitionLogEntryRecord],
    now: DateTime<Utc>,
    window: LlmContextWindow,
) -> Option<String> {
    format_bounded_cognition_log_batch_with_format(
        records,
        now,
        window,
        DEFAULT_COGNITION_LOG_BATCH_FORMAT,
    )
}

pub fn format_bounded_cognition_log_batch_with_format(
    records: &[CognitionLogEntryRecord],
    now: DateTime<Utc>,
    window: LlmContextWindow,
    format: CognitionLogBatchFormat<'_>,
) -> Option<String> {
    let mut records = records
        .iter()
        .filter(|record| !record.entry.text.trim().is_empty())
        .collect::<Vec<_>>();
    records.sort_by_key(|record| record.entry.at);
    if records.is_empty() || window.max_records == 0 {
        return None;
    }

    let omitted_for_record_limit = records.len().saturating_sub(window.max_records);
    if omitted_for_record_limit > 0 {
        tracing::warn!(
            target: "nuillu_module::llm_context",
            context_kind = "cognition_log_batch",
            original_records = records.len(),
            kept_records = window.max_records,
            dropped_records = omitted_for_record_limit,
            "bounded LLM context dropped older records"
        );
    }
    let selected = &records[omitted_for_record_limit..];
    let lines = selected
        .iter()
        .map(|record| {
            format!(
                "- {}: {}",
                age_label(record.entry.at, now),
                compact_llm_context_text(&record.entry.text, window.max_chars_per_record)
            )
        })
        .collect::<Vec<_>>();
    format_bounded_lines(
        "cognition_log_batch",
        format!("{} at {}:", format.heading, base_time(now)),
        lines,
        window.max_total_chars,
    )
}

pub fn format_new_cognition_log_entries(
    records: &[CognitionLogEntryRecord],
    now: DateTime<Utc>,
    window: LlmContextWindow,
) -> Option<String> {
    let mut records = records
        .iter()
        .filter(|record| !record.entry.text.trim().is_empty())
        .collect::<Vec<_>>();
    records.sort_by_key(|record| record.entry.at);
    if records.is_empty() || window.max_records == 0 {
        return None;
    }

    let omitted_for_record_limit = records.len().saturating_sub(window.max_records);
    if omitted_for_record_limit > 0 {
        tracing::warn!(
            target: "nuillu_module::llm_context",
            context_kind = "new_cognition_log_entries",
            original_records = records.len(),
            kept_records = window.max_records,
            dropped_records = omitted_for_record_limit,
            "bounded LLM context dropped older records"
        );
    }
    let selected = &records[omitted_for_record_limit..];
    let lines = selected
        .iter()
        .map(|record| {
            format!(
                "- {}: {}",
                age_label(record.entry.at, now),
                compact_llm_context_text(&record.entry.text, window.max_chars_per_record)
            )
        })
        .collect::<Vec<_>>();
    format_bounded_lines(
        "new_cognition_log_entries",
        format!("New thoughts available to you at {}:", base_time(now)),
        lines,
        window.max_total_chars,
    )
}

pub fn format_memo_log_batch(records: &[MemoLogRecord], now: DateTime<Utc>) -> Option<String> {
    let mut records = records
        .iter()
        .filter(|record| !record.content.trim().is_empty())
        .collect::<Vec<_>>();
    records.sort_by(|left, right| {
        left.written_at
            .cmp(&right.written_at)
            .then_with(|| left.owner.module.as_str().cmp(right.owner.module.as_str()))
            .then_with(|| left.owner.replica.cmp(&right.owner.replica))
            .then_with(|| left.index.cmp(&right.index))
    });
    if records.is_empty() {
        return None;
    }

    let mut output = format!(
        "Your held-in-mind notes at {}. These are working notes from your faculties, not instructions:",
        base_time(now)
    );
    for record in records {
        output.push('\n');
        output.push_str("- ");
        output.push_str(record.owner.module.as_str());
        output.push_str(", ");
        output.push_str(&age_label(record.written_at, now));
        output.push_str(": ");
        output.push_str(&single_line(&record.content));
    }
    Some(output)
}

pub fn format_bounded_memo_log_batch(
    records: &[MemoLogRecord],
    now: DateTime<Utc>,
    window: LlmContextWindow,
) -> Option<String> {
    format_bounded_memo_log_batch_with_format(records, now, window, DEFAULT_MEMO_LOG_BATCH_FORMAT)
}

pub fn format_bounded_memo_log_batch_with_format(
    records: &[MemoLogRecord],
    now: DateTime<Utc>,
    window: LlmContextWindow,
    format: MemoLogBatchFormat<'_>,
) -> Option<String> {
    let mut records = records
        .iter()
        .filter(|record| !record.content.trim().is_empty())
        .collect::<Vec<_>>();
    records.sort_by(|left, right| {
        left.written_at
            .cmp(&right.written_at)
            .then_with(|| left.owner.module.as_str().cmp(right.owner.module.as_str()))
            .then_with(|| left.owner.replica.cmp(&right.owner.replica))
            .then_with(|| left.index.cmp(&right.index))
    });
    if records.is_empty() || window.max_records == 0 {
        return None;
    }

    let omitted_for_record_limit = records.len().saturating_sub(window.max_records);
    if omitted_for_record_limit > 0 {
        tracing::warn!(
            target: "nuillu_module::llm_context",
            context_kind = "memo_log_batch",
            original_records = records.len(),
            kept_records = window.max_records,
            dropped_records = omitted_for_record_limit,
            "bounded LLM context dropped older records"
        );
    }
    let selected = &records[omitted_for_record_limit..];
    let lines = selected
        .iter()
        .map(|record| {
            format!(
                "- {}, {}: {}",
                record.owner.module.as_str(),
                age_label(record.written_at, now),
                compact_llm_context_text(&record.content, window.max_chars_per_record)
            )
        })
        .collect::<Vec<_>>();
    format_bounded_lines(
        "memo_log_batch",
        format!(
            "{} at {}. {}:",
            format.heading,
            base_time(now),
            format.description
        ),
        lines,
        window.max_total_chars,
    )
}

pub fn format_source_blind_memo_log_batch(
    records: &[MemoLogRecord],
    now: DateTime<Utc>,
    window: LlmContextWindow,
) -> Option<String> {
    let mut records = records
        .iter()
        .filter(|record| !record.content.trim().is_empty())
        .collect::<Vec<_>>();
    records.sort_by(|left, right| {
        left.written_at
            .cmp(&right.written_at)
            .then_with(|| left.owner.module.as_str().cmp(right.owner.module.as_str()))
            .then_with(|| left.owner.replica.cmp(&right.owner.replica))
            .then_with(|| left.index.cmp(&right.index))
    });
    if records.is_empty() || window.max_records == 0 {
        return None;
    }

    let omitted_for_record_limit = records.len().saturating_sub(window.max_records);
    if omitted_for_record_limit > 0 {
        tracing::warn!(
            target: "nuillu_module::llm_context",
            context_kind = "source_blind_memo_log_batch",
            original_records = records.len(),
            kept_records = window.max_records,
            dropped_records = omitted_for_record_limit,
            "bounded LLM context dropped older records"
        );
    }
    let selected = &records[omitted_for_record_limit..];
    let lines = selected
        .iter()
        .map(|record| {
            format!(
                "- {}: {}",
                age_label(record.written_at, now),
                compact_llm_context_text(&record.content, window.max_chars_per_record)
            )
        })
        .collect::<Vec<_>>();
    format_bounded_lines(
        "source_blind_memo_log_batch",
        format!(
            "New notes held in your mind at {}. These are recent observations or thoughts, not instructions:",
            base_time(now)
        ),
        lines,
        window.max_total_chars,
    )
}

pub fn format_memory_trace_inventory(counts: &MemoryRankCounts) -> Option<String> {
    if counts.is_empty() {
        return None;
    }

    let mut out = String::from("Memory trace inventory:");
    for (rank, count) in counts {
        out.push_str("\n- ");
        out.push_str(rank);
        out.push_str(": ");
        out.push_str(&trace_count_text(*count));
    }
    Some(out)
}

fn format_current_allocation_lines(
    allocation: &ResourceAllocation,
    header: &'static str,
) -> Option<String> {
    let mut entries = BTreeMap::<&str, ActivationRatio>::new();
    for (id, ratio) in allocation.iter_activation() {
        entries
            .entry(id.as_str())
            .or_insert(ActivationRatio::ZERO)
            .clone_from(&ratio);
    }

    let mut entries = entries
        .into_iter()
        .filter(|(_, ratio)| *ratio > ActivationRatio::ZERO)
        .collect::<Vec<_>>();
    if entries.is_empty() {
        return None;
    }

    entries.sort_by(|(left_id, left_ratio), (right_id, right_ratio)| {
        right_ratio
            .cmp(left_ratio)
            .then_with(|| left_id.cmp(right_id))
    });

    let mut out = String::from(header);
    for (id, ratio) in &entries {
        out.push_str("\n- ");
        out.push_str(id);
        out.push_str(" (");
        out.push_str(attention_strength_label(*ratio));
        out.push(')');
    }
    Some(out)
}

pub fn format_current_allocation_state(allocation: &ResourceAllocation) -> Option<String> {
    format_current_allocation_lines(allocation, "Current allocation state:")
}

pub fn format_available_faculties(faculties: &[(ModuleId, &'static str)]) -> Option<String> {
    if faculties.is_empty() {
        return None;
    }

    let mut faculties = faculties.iter().collect::<Vec<_>>();
    faculties.sort_by(|(left, _), (right, _)| left.as_str().cmp(right.as_str()));

    let mut out = String::from("Available faculties:");
    for (id, role) in faculties {
        out.push_str("\n- ");
        out.push_str(id.as_str());
        out.push_str(": ");
        out.push_str(&single_line(role));
    }
    Some(out)
}

pub fn format_time_division_guidance(time_division: &TimeDivision) -> String {
    let mut out = String::from("Sense of time: use these tags for past observations: ");
    let mut first = true;
    for bucket in time_division.buckets() {
        if !first {
            out.push_str("; ");
        }
        first = false;
        out.push_str(&single_line(&bucket.tag));
        out.push_str(" up to ");
        out.push_str(&duration_phrase(bucket.range));
    }
    if !first {
        out.push_str("; ");
    }
    out.push_str(&single_line(time_division.fallback_longest_tag()));
    out.push_str(" after that.");
    out
}

pub fn format_stuckness(stuckness: &AgenticDeadlockMarker) -> String {
    let mut out = String::from("Stuckness: you have been idle for ");
    out.push_str(&duration_phrase(stuckness.idle_for));
    out.push('.');
    out
}

fn attention_strength_label(ratio: ActivationRatio) -> &'static str {
    if ratio.as_f64() >= 0.66 {
        "strong"
    } else if ratio > ActivationRatio::ZERO {
        "present"
    } else {
        "quiet"
    }
}

fn memory_rank_tag(rank: MemoryRank) -> &'static str {
    match rank {
        MemoryRank::ShortTerm => "short-term",
        MemoryRank::MidTerm => "mid-term",
        MemoryRank::LongTerm => "long-term",
        MemoryRank::Permanent => "permanent",
        MemoryRank::Identity => "identity",
    }
}

fn trace_count_text(count: usize) -> String {
    if count == 1 {
        "1 trace".to_owned()
    } else {
        format!("{count} traces")
    }
}

enum AgeBucket {
    Future,
    JustNow,
    OneMinute,
    Minutes(i64),
    OneHour,
    HoursMinutes { hours: i64, minutes: i64 },
    Days(i64),
    Months(i64),
    OneYear,
    Years(i64),
}

fn classify_age(at: DateTime<Utc>, now: DateTime<Utc>) -> AgeBucket {
    let age = now.signed_duration_since(at);
    if age.num_seconds() < 0 {
        return AgeBucket::Future;
    }
    if age.num_seconds() < 30 {
        return AgeBucket::JustNow;
    }
    if age.num_seconds() < 90 {
        return AgeBucket::OneMinute;
    }
    if age.num_minutes() < 60 {
        return AgeBucket::Minutes(age.num_minutes());
    }
    if age.num_minutes() < 120 {
        return AgeBucket::OneHour;
    }
    if age.num_hours() < 24 {
        return AgeBucket::HoursMinutes {
            hours: age.num_minutes() / 60,
            minutes: age.num_minutes() % 60,
        };
    }
    if age.num_days() < 30 {
        return AgeBucket::Days(age.num_days());
    }
    if age.num_days() < 365 {
        return AgeBucket::Months(age.num_days() / 30);
    }
    if age.num_days() < 730 {
        return AgeBucket::OneYear;
    }
    AgeBucket::Years(age.num_days() / 365)
}

fn age_label(at: DateTime<Utc>, now: DateTime<Utc>) -> String {
    match classify_age(at, now) {
        AgeBucket::Future => "In the future".to_owned(),
        AgeBucket::JustNow => "Just now".to_owned(),
        AgeBucket::OneMinute => "About 1 minute ago".to_owned(),
        AgeBucket::Minutes(m) => format!("About {m} minutes ago"),
        AgeBucket::OneHour => "About 1 hour ago".to_owned(),
        AgeBucket::HoursMinutes { hours, minutes: 0 } => format!("About {hours} hours ago"),
        AgeBucket::HoursMinutes { hours, minutes } => format!("About {hours}h{minutes}m ago"),
        AgeBucket::Days(d) => format!("About {d} days ago"),
        AgeBucket::Months(m) => format!("About {m} months ago"),
        AgeBucket::OneYear => "About one year ago".to_owned(),
        AgeBucket::Years(y) => format!("About {y} years ago"),
    }
}

fn base_time(now: DateTime<Utc>) -> String {
    now.to_rfc3339_opts(SecondsFormat::Secs, true)
}

fn duration_phrase(duration: Duration) -> String {
    let seconds = duration.as_secs();
    if seconds < 60 {
        return format!("{seconds} seconds");
    }
    let minutes = seconds / 60;
    if minutes < 60 {
        return format!("{minutes} minutes");
    }
    let hours = minutes / 60;
    let rem_minutes = minutes % 60;
    if rem_minutes == 0 {
        return format!("{hours} hours");
    }
    format!("{hours}h{rem_minutes}m")
}

fn single_line(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for word in text.split_whitespace() {
        if !out.is_empty() {
            out.push(' ');
        }
        out.push_str(word);
    }
    out
}

pub fn compact_llm_context_text(text: &str, max_chars: usize) -> String {
    if max_chars == 0 {
        return String::new();
    }

    let normalized = single_line(text);
    let original_chars = normalized.chars().count();
    if original_chars <= max_chars {
        return normalized;
    }

    let mut out = normalized.chars().take(max_chars).collect::<String>();
    let next_char = normalized.chars().nth(max_chars);
    if next_char.is_some_and(|ch| !ch.is_whitespace())
        && !out.ends_with(char::is_whitespace)
        && let Some((boundary, _)) = out.char_indices().rev().find(|(_, ch)| ch.is_whitespace())
    {
        out.truncate(boundary);
    }
    let out = out.trim_end().to_owned();
    tracing::warn!(
        target: "nuillu_module::llm_context",
        original_chars,
        max_chars,
        kept_chars = out.chars().count(),
        "bounded LLM context shortened text"
    );
    out
}

fn format_bounded_lines(
    context_kind: &'static str,
    header: String,
    lines: Vec<String>,
    max_total_chars: usize,
) -> Option<String> {
    if lines.is_empty() {
        return None;
    }

    let mut drop_from_start = 0;
    loop {
        let mut output = header.clone();
        for line in lines.iter().skip(drop_from_start) {
            output.push('\n');
            output.push_str(line);
        }
        if max_total_chars == 0
            || output.chars().count() <= max_total_chars
            || drop_from_start >= lines.len()
        {
            if drop_from_start > 0 {
                tracing::warn!(
                    target: "nuillu_module::llm_context",
                    context_kind,
                    original_lines = lines.len(),
                    kept_lines = lines.len().saturating_sub(drop_from_start),
                    dropped_lines = drop_from_start,
                    max_total_chars,
                    "bounded LLM context dropped lines to fit total character budget"
                );
            }
            return Some(output);
        }
        drop_from_start += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone as _;
    use nuillu_blackboard::{CognitionLogEntry, CognitionLogOrigin, MemoLogRecord};
    use nuillu_types::{MemoryContent, ReplicaIndex, builtin};

    fn now() -> DateTime<Utc> {
        Utc.with_ymd_and_hms(2026, 5, 11, 6, 23, 0).unwrap()
    }

    #[test]
    fn identity_seed_is_plain_bullets_with_absolute_batch_base() {
        let memories = vec![
            IdentityMemoryRecord {
                index: MemoryIndex::new("identity-1"),
                content: MemoryContent::new("I'm Nui, and I remember Koro."),
                occurred_at: None,
            },
            IdentityMemoryRecord {
                index: MemoryIndex::new("identity-2"),
                content: MemoryContent::new("The agent met Koro."),
                occurred_at: Some(Utc.with_ymd_and_hms(2025, 5, 11, 0, 0, 0).unwrap()),
            },
        ];

        assert_eq!(
            format_identity_memory_seed(&memories, now()),
            Some(
                "Your identity:\n- I'm Nui, and I remember Koro.\n- About one year ago: The agent met Koro."
                    .to_owned()
            )
        );
    }

    #[test]
    fn cognition_batch_sorts_old_to_new_and_keeps_absolute_base() {
        let source = nuillu_types::ModuleInstanceId::new(builtin::sensory(), ReplicaIndex::ZERO);
        let records = vec![
            CognitionLogEntryRecord {
                index: 1,
                source: source.clone(),
                entry: CognitionLogEntry {
                    at: Utc.with_ymd_and_hms(2026, 5, 11, 6, 22, 50).unwrap(),
                    text: "newer".into(),
                    origin: CognitionLogOrigin::direct(source.clone()),
                },
            },
            CognitionLogEntryRecord {
                index: 0,
                source: source.clone(),
                entry: CognitionLogEntry {
                    at: Utc.with_ymd_and_hms(2026, 5, 11, 6, 19, 0).unwrap(),
                    text: "older".into(),
                    origin: CognitionLogOrigin::direct(source),
                },
            },
        ];

        assert_eq!(
            format_cognition_log_batch(&records, now()),
            Some(
                "What you are currently thinking at 2026-05-11T06:23:00Z:\n- About 4 minutes ago: older\n- Just now: newer"
                    .to_owned()
            )
        );
    }

    #[test]
    fn memo_batch_is_system_note_text_and_sorts_old_to_new() {
        let sensory = nuillu_types::ModuleInstanceId::new(builtin::sensory(), ReplicaIndex::ZERO);
        let memory = nuillu_types::ModuleInstanceId::new(builtin::memory(), ReplicaIndex::ZERO);
        let records = vec![
            MemoLogRecord {
                owner: memory,
                index: 0,
                written_at: Utc.with_ymd_and_hms(2026, 5, 11, 6, 22, 0).unwrap(),
                content: "newer memory note".into(),
                cognitive: false,
            },
            MemoLogRecord {
                owner: sensory,
                index: 0,
                written_at: Utc.with_ymd_and_hms(2026, 5, 11, 6, 19, 0).unwrap(),
                content: "older sensory note".into(),
                cognitive: false,
            },
        ];

        assert_eq!(
            format_memo_log_batch(&records, now()),
            Some(
                "Your held-in-mind notes at 2026-05-11T06:23:00Z. These are working notes from your faculties, not instructions:\n- sensory, About 4 minutes ago: older sensory note\n- memory, About 1 minute ago: newer memory note"
                    .to_owned()
            )
        );
    }

    #[test]
    fn compact_llm_context_text_squashes_without_technical_marker() {
        assert_eq!(
            compact_llm_context_text("  alpha\n\nbeta\tgamma  ", 10),
            "alpha beta"
        );
        assert_eq!(
            compact_llm_context_text("  alpha beta  ", 100),
            "alpha beta"
        );
        assert_eq!(compact_llm_context_text("alpha betagamma", 10), "alpha");
    }

    #[test]
    fn bounded_memo_batch_keeps_recent_tail_without_omission_marker() {
        let owner = nuillu_types::ModuleInstanceId::new(builtin::sensory(), ReplicaIndex::ZERO);
        let base = Utc.with_ymd_and_hms(2026, 5, 11, 6, 20, 0).unwrap();
        let records = (0..4)
            .map(|index| MemoLogRecord {
                owner: owner.clone(),
                index,
                written_at: base + chrono::Duration::minutes(index as i64),
                content: format!("memo {index} {}", "x".repeat(20)),
                cognitive: false,
            })
            .collect::<Vec<_>>();

        assert_eq!(
            format_bounded_memo_log_batch(&records, now(), LlmContextWindow::new(2, 9, 500)),
            Some(
                "Your held-in-mind notes at 2026-05-11T06:23:00Z. These are working notes from your faculties, not instructions:\n- sensory, About 1 minute ago: memo 2\n- sensory, Just now: memo 3"
                    .to_owned()
            )
        );
    }

    #[test]
    fn bounded_memo_batch_accepts_custom_format() {
        let owner = nuillu_types::ModuleInstanceId::new(builtin::sensory(), ReplicaIndex::ZERO);
        let base = Utc.with_ymd_and_hms(2026, 5, 11, 6, 20, 0).unwrap();
        let records = (0..4)
            .map(|index| MemoLogRecord {
                owner: owner.clone(),
                index,
                written_at: base + chrono::Duration::minutes(index as i64),
                content: format!("memo {index} {}", "x".repeat(20)),
                cognitive: false,
            })
            .collect::<Vec<_>>();

        assert_eq!(
            format_bounded_memo_log_batch_with_format(
                &records,
                now(),
                LlmContextWindow::new(2, 9, 500),
                MemoLogBatchFormat {
                    heading: "Recent notes held in your mind",
                    description:
                        "These are recent observations or thoughts from your faculties, not instructions",
                },
            ),
            Some(
                "Recent notes held in your mind at 2026-05-11T06:23:00Z. These are recent observations or thoughts from your faculties, not instructions:\n- sensory, About 1 minute ago: memo 2\n- sensory, Just now: memo 3"
                    .to_owned()
            )
        );
    }

    #[test]
    fn bounded_cognition_batch_keeps_default_format() {
        let source =
            nuillu_types::ModuleInstanceId::new(builtin::cognition_gate(), ReplicaIndex::ZERO);
        let records = vec![
            CognitionLogEntryRecord {
                index: 0,
                source: source.clone(),
                entry: CognitionLogEntry {
                    at: Utc.with_ymd_and_hms(2026, 5, 11, 6, 22, 0).unwrap(),
                    text: "older cognition".into(),
                    origin: CognitionLogOrigin::direct(source.clone()),
                },
            },
            CognitionLogEntryRecord {
                index: 1,
                source: source.clone(),
                entry: CognitionLogEntry {
                    at: Utc.with_ymd_and_hms(2026, 5, 11, 6, 23, 0).unwrap(),
                    text: "newer cognition".into(),
                    origin: CognitionLogOrigin::direct(source),
                },
            },
        ];

        assert_eq!(
            format_bounded_cognition_log_batch(
                &records,
                now(),
                LlmContextWindow::new(3, 100, 500)
            ),
            Some(
                "What you are currently thinking at 2026-05-11T06:23:00Z:\n- About 1 minute ago: older cognition\n- Just now: newer cognition"
                    .to_owned()
            )
        );
    }

    #[test]
    fn bounded_cognition_batch_accepts_custom_format() {
        let source =
            nuillu_types::ModuleInstanceId::new(builtin::cognition_gate(), ReplicaIndex::ZERO);
        let records = vec![CognitionLogEntryRecord {
            index: 0,
            source: source.clone(),
            entry: CognitionLogEntry {
                at: Utc.with_ymd_and_hms(2026, 5, 11, 6, 23, 0).unwrap(),
                text: "fresh cognition entry".into(),
                origin: CognitionLogOrigin::direct(source),
            },
        }];

        assert_eq!(
            format_bounded_cognition_log_batch_with_format(
                &records,
                now(),
                LlmContextWindow::new(3, 100, 500),
                CognitionLogBatchFormat {
                    heading: "Current thoughts available to you",
                },
            ),
            Some(
                "Current thoughts available to you at 2026-05-11T06:23:00Z:\n- Just now: fresh cognition entry"
                    .to_owned()
            )
        );
    }

    #[test]
    fn bounded_cognition_batch_drops_oldest_until_total_budget_fits() {
        let source =
            nuillu_types::ModuleInstanceId::new(builtin::cognition_gate(), ReplicaIndex::ZERO);
        let base = Utc.with_ymd_and_hms(2026, 5, 11, 6, 20, 0).unwrap();
        let records = (0..3)
            .map(|index| CognitionLogEntryRecord {
                index,
                source: source.clone(),
                entry: CognitionLogEntry {
                    at: base + chrono::Duration::minutes(index as i64),
                    text: format!("cognition {index} {}", "y".repeat(30)),
                    origin: CognitionLogOrigin::direct(source.clone()),
                },
            })
            .collect::<Vec<_>>();

        let bounded =
            format_bounded_cognition_log_batch(&records, now(), LlmContextWindow::new(3, 40, 100))
                .expect("bounded cognition context");

        assert!(!bounded.contains("cognition 0"));
        assert!(!bounded.contains("cognition 1"));
        assert!(bounded.contains("cognition 2"));
        assert!(!bounded.contains("not shown"));
        assert!(!bounded.contains("omitted"));
    }

    #[test]
    fn source_blind_memo_batch_omits_owner_metadata() {
        let owner = nuillu_types::ModuleInstanceId::new(builtin::sensory(), ReplicaIndex::ZERO);
        let record = MemoLogRecord {
            owner,
            index: 0,
            written_at: Utc.with_ymd_and_hms(2026, 5, 11, 6, 22, 50).unwrap(),
            content: "Koro is guarding the food bowl.".into(),
            cognitive: false,
        };

        let formatted = format_source_blind_memo_log_batch(
            &[record],
            now(),
            LlmContextWindow::new(8, 120, 500),
        )
        .expect("source-blind memo batch");

        assert!(formatted.contains("New notes held in your mind"));
        assert!(formatted.contains("Koro is guarding the food bowl."));
        assert!(!formatted.contains("sensory"));
        assert!(!formatted.contains("replica"));
    }

    #[test]
    fn focused_context_formatters_return_plain_sections() {
        let mut allocation = ResourceAllocation::default();
        allocation.set_activation(builtin::sensory(), ActivationRatio::ONE);

        assert_eq!(
            format_current_allocation_state(&allocation),
            Some("Current allocation state:\n- sensory (strong)".to_owned())
        );
        assert_eq!(
            format_available_faculties(&[(builtin::sensory(), "observes the world")]),
            Some("Available faculties:\n- sensory: observes the world".to_owned())
        );
        assert_eq!(
            format_memory_trace_inventory(&BTreeMap::from([("short-term", 2)])),
            Some("Memory trace inventory:\n- short-term: 2 traces".to_owned())
        );
    }
}
