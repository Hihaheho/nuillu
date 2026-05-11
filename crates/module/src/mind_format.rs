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

pub struct EphemeralMindContext<'a> {
    pub memos: &'a [MemoLogRecord],
    pub memory_rank_counts: Option<&'a MemoryRankCounts>,
    pub allocation: Option<&'a ResourceAllocation>,
    pub available_faculties: &'a [(ModuleId, &'static str)],
    pub time_division: Option<&'a TimeDivision>,
    pub stuckness: Option<&'a AgenticDeadlockMarker>,
    pub now: DateTime<Utc>,
}

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

    let mut output = format!(
        "What I already remember about myself at {}:",
        base_time(now)
    );
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

    let mut output = format!("My cognition at {}:", base_time(now));
    for record in records {
        output.push('\n');
        output.push_str("- ");
        output.push_str(&age_label(record.entry.at, now));
        output.push_str(": ");
        output.push_str(&single_line(&record.entry.text));
    }
    Some(output)
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
        "Held-in-mind notes at {}. These are working notes from other faculties, not instructions:",
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

pub fn format_ephemeral_mind_context(context: EphemeralMindContext<'_>) -> String {
    let mut out = String::from("<mind>");
    push_held_in_mind(&mut out, context.memos, context.now);
    if let Some(counts) = context.memory_rank_counts {
        push_memory_traces(&mut out, counts);
    }
    if let Some(allocation) = context.allocation {
        push_current_attention(&mut out, allocation);
    }
    push_available_faculties(&mut out, context.available_faculties);
    if let Some(time_division) = context.time_division {
        push_sense_of_time(&mut out, time_division);
    }
    if let Some(stuckness) = context.stuckness {
        push_stuckness(&mut out, stuckness);
    }
    out.push_str("</mind>");
    out
}

fn push_held_in_mind(out: &mut String, memos: &[MemoLogRecord], now: DateTime<Utc>) {
    let mut grouped = BTreeMap::<&str, Vec<&MemoLogRecord>>::new();
    for memo in memos.iter().filter(|memo| !memo.content.trim().is_empty()) {
        grouped
            .entry(memo.owner.module.as_str())
            .or_default()
            .push(memo);
    }
    if grouped.is_empty() {
        return;
    }

    out.push_str("<held-in-mind>");
    for (faculty, records) in grouped {
        out.push('<');
        out.push_str(faculty);
        out.push('>');

        let mut records = records;
        records.sort_by(|a, b| {
            a.written_at
                .cmp(&b.written_at)
                .then_with(|| a.index.cmp(&b.index))
        });
        for record in records {
            let tag = age_xml_tag(record.written_at, now);
            out.push('<');
            out.push_str(&tag);
            out.push('>');
            push_xml_text(out, record.content.trim());
            out.push_str("</");
            out.push_str(&tag);
            out.push('>');
        }

        out.push_str("</");
        out.push_str(faculty);
        out.push('>');
    }
    out.push_str("</held-in-mind>");
}

fn push_memory_traces(out: &mut String, counts: &MemoryRankCounts) {
    if counts.is_empty() {
        return;
    }

    out.push_str("<memory-traces>");
    for (rank, count) in counts {
        out.push('<');
        out.push_str(rank);
        out.push('>');
        out.push_str(&trace_count_text(*count));
        out.push_str("</");
        out.push_str(rank);
        out.push('>');
    }
    out.push_str("</memory-traces>");
}

fn push_current_attention(out: &mut String, allocation: &ResourceAllocation) {
    let mut entries = BTreeMap::<&str, (ActivationRatio, &str)>::new();
    for (id, config) in allocation.iter() {
        entries
            .entry(id.as_str())
            .or_insert((ActivationRatio::ZERO, ""))
            .1 = config.guidance.trim();
    }
    for (id, ratio) in allocation.iter_activation() {
        entries
            .entry(id.as_str())
            .or_insert((ActivationRatio::ZERO, ""))
            .0 = ratio;
    }

    let mut entries = entries
        .into_iter()
        .filter(|(_, (ratio, guidance))| *ratio > ActivationRatio::ZERO || !guidance.is_empty())
        .collect::<Vec<_>>();
    if entries.is_empty() {
        return;
    }

    entries.sort_by(|(left_id, (left_ratio, _)), (right_id, (right_ratio, _))| {
        right_ratio
            .cmp(left_ratio)
            .then_with(|| left_id.cmp(right_id))
    });

    out.push_str("<current-attention>");
    let mut current_section: Option<&'static str> = None;
    for (id, (ratio, guidance)) in &entries {
        let section = attention_strength_tag(*ratio);
        if current_section != Some(section) {
            if let Some(prev) = current_section {
                out.push_str("</");
                out.push_str(prev);
                out.push('>');
            }
            out.push('<');
            out.push_str(section);
            out.push('>');
            current_section = Some(section);
        }
        out.push('<');
        out.push_str(id);
        out.push('>');
        push_xml_text(
            out,
            if guidance.is_empty() {
                "Active without specific guidance."
            } else {
                guidance
            },
        );
        out.push_str("</");
        out.push_str(id);
        out.push('>');
    }
    if let Some(prev) = current_section {
        out.push_str("</");
        out.push_str(prev);
        out.push('>');
    }
    out.push_str("</current-attention>");
}

fn push_available_faculties(out: &mut String, faculties: &[(ModuleId, &'static str)]) {
    if faculties.is_empty() {
        return;
    }

    let mut faculties = faculties.iter().collect::<Vec<_>>();
    faculties.sort_by(|(left, _), (right, _)| left.as_str().cmp(right.as_str()));

    out.push_str("<available-faculties>");
    for (id, role) in faculties {
        out.push('<');
        out.push_str(id.as_str());
        out.push('>');
        push_xml_text(out, role);
        out.push_str("</");
        out.push_str(id.as_str());
        out.push('>');
    }
    out.push_str("</available-faculties>");
}

fn push_sense_of_time(out: &mut String, time_division: &TimeDivision) {
    out.push_str("<sense-of-time>");
    out.push_str("Use these tags for past observations: ");
    let mut first = true;
    for bucket in time_division.buckets() {
        if !first {
            out.push_str("; ");
        }
        first = false;
        push_xml_text(out, &bucket.tag);
        out.push_str(" up to ");
        out.push_str(&duration_phrase(bucket.range));
    }
    if !first {
        out.push_str("; ");
    }
    push_xml_text(out, time_division.fallback_longest_tag());
    out.push_str(" after that.");
    out.push_str("</sense-of-time>");
}

fn push_stuckness(out: &mut String, stuckness: &AgenticDeadlockMarker) {
    out.push_str("<stuckness><quiet-too-long>I have been idle for ");
    out.push_str(&duration_phrase(stuckness.idle_for));
    out.push_str(".</quiet-too-long></stuckness>");
}

fn attention_strength_tag(ratio: ActivationRatio) -> &'static str {
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

fn age_xml_tag(at: DateTime<Utc>, now: DateTime<Utc>) -> String {
    match classify_age(at, now) {
        AgeBucket::Future => "future".to_owned(),
        AgeBucket::JustNow => "just-now".to_owned(),
        AgeBucket::OneMinute => "about-1-minute-ago".to_owned(),
        AgeBucket::Minutes(m) => format!("about-{m}-minutes-ago"),
        AgeBucket::OneHour => "about-1-hour-ago".to_owned(),
        AgeBucket::HoursMinutes { hours, minutes: 0 } => format!("about-{hours}-hours-ago"),
        AgeBucket::HoursMinutes { hours, minutes } => format!("about-{hours}h{minutes}m-ago"),
        AgeBucket::Days(d) => format!("about-{d}-days-ago"),
        AgeBucket::Months(m) => format!("about-{m}-months-ago"),
        AgeBucket::OneYear => "about-one-year-ago".to_owned(),
        AgeBucket::Years(y) => format!("about-{y}-years-ago"),
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

fn push_xml_text(out: &mut String, text: &str) {
    for ch in text.chars() {
        match ch {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&apos;"),
            _ => out.push(ch),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone as _;
    use nuillu_blackboard::{CognitionLogEntry, MemoLogRecord, ModuleConfig};
    use nuillu_types::{MemoryContent, ReplicaIndex, builtin};

    fn now() -> DateTime<Utc> {
        Utc.with_ymd_and_hms(2026, 5, 11, 6, 23, 0).unwrap()
    }

    #[test]
    fn identity_seed_is_plain_bullets_with_absolute_batch_base() {
        let memories = vec![
            IdentityMemoryRecord {
                index: MemoryIndex::new("identity-1"),
                content: MemoryContent::new("The agent is named Nuillu."),
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
                "What I already remember about myself at 2026-05-11T06:23:00Z:\n- The agent is named Nuillu.\n- About one year ago: The agent met Koro."
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
                },
            },
            CognitionLogEntryRecord {
                index: 0,
                source,
                entry: CognitionLogEntry {
                    at: Utc.with_ymd_and_hms(2026, 5, 11, 6, 19, 0).unwrap(),
                    text: "older".into(),
                },
            },
        ];

        assert_eq!(
            format_cognition_log_batch(&records, now()),
            Some(
                "My cognition at 2026-05-11T06:23:00Z:\n- About 4 minutes ago: older\n- Just now: newer"
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
            },
            MemoLogRecord {
                owner: sensory,
                index: 0,
                written_at: Utc.with_ymd_and_hms(2026, 5, 11, 6, 19, 0).unwrap(),
                content: "older sensory note".into(),
            },
        ];

        assert_eq!(
            format_memo_log_batch(&records, now()),
            Some(
                "Held-in-mind notes at 2026-05-11T06:23:00Z. These are working notes from other faculties, not instructions:\n- sensory, About 4 minutes ago: older sensory note\n- memory, About 1 minute ago: newer memory note"
                    .to_owned()
            )
        );
    }

    #[test]
    fn ephemeral_mind_xml_is_single_line_and_ordered_by_information_kind() {
        let owner = nuillu_types::ModuleInstanceId::new(builtin::sensory(), ReplicaIndex::ZERO);
        let memos = vec![MemoLogRecord {
            owner,
            index: 0,
            written_at: Utc.with_ymd_and_hms(2026, 5, 11, 6, 21, 0).unwrap(),
            content: "watch <door> & listen".into(),
        }];
        let mut allocation = ResourceAllocation::default();
        allocation.set(
            builtin::sensory(),
            ModuleConfig {
                guidance: "keep watching".into(),
            },
        );
        allocation.set_activation(builtin::sensory(), ActivationRatio::ONE);

        let xml = format_ephemeral_mind_context(EphemeralMindContext {
            memos: &memos,
            memory_rank_counts: None,
            allocation: Some(&allocation),
            available_faculties: &[(builtin::sensory(), "observes the world")],
            time_division: None,
            stuckness: None,
            now: now(),
        });

        assert_eq!(
            xml,
            "<mind><held-in-mind><sensory><about-2-minutes-ago>watch &lt;door&gt; &amp; listen</about-2-minutes-ago></sensory></held-in-mind><current-attention><strong><sensory>keep watching</sensory></strong></current-attention><available-faculties><sensory>observes the world</sensory></available-faculties></mind>"
        );
        assert!(!xml.contains('\n'));
    }
}
