use chrono::{DateTime, SecondsFormat, Utc};

pub fn render_memory_for_llm(
    content: &str,
    occurred_at: Option<DateTime<Utc>>,
    now: DateTime<Utc>,
) -> String {
    let Some(occurred_at) = occurred_at else {
        return content.to_owned();
    };

    let tag = memory_age_tag(occurred_at, now);
    let timestamp = occurred_at.to_rfc3339_opts(SecondsFormat::Secs, true);
    format!("<{tag} occurred-at=\"{timestamp}\">\n{content}\n</{tag}>")
}

fn memory_age_tag(occurred_at: DateTime<Utc>, now: DateTime<Utc>) -> &'static str {
    let age = now.signed_duration_since(occurred_at);
    if age.num_seconds() < 0 {
        return "future-memory";
    }
    if age.num_minutes() < 5 {
        return "now";
    }
    if age.num_days() < 1 {
        return "today";
    }
    if age.num_days() < 30 {
        return "days-ago";
    }
    if age.num_days() < 365 {
        return "months-ago";
    }
    if age.num_days() < 730 {
        return "one-year-ago";
    }
    "years-ago"
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone as _;

    #[test]
    fn renders_untimed_memory_as_raw_content() {
        let now = Utc.with_ymd_and_hms(2026, 5, 10, 0, 0, 0).unwrap();
        assert_eq!(render_memory_for_llm("plain", None, now), "plain");
    }

    #[test]
    fn renders_one_year_old_memory_with_xml_like_tag() {
        let now = Utc.with_ymd_and_hms(2026, 5, 10, 0, 0, 0).unwrap();
        let occurred_at = Utc.with_ymd_and_hms(2025, 5, 10, 0, 0, 0).unwrap();
        assert_eq!(
            render_memory_for_llm("memory", Some(occurred_at), now),
            "<one-year-ago occurred-at=\"2025-05-10T00:00:00Z\">\nmemory\n</one-year-ago>"
        );
    }
}
