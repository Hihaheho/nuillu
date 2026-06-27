use chrono::{DateTime, FixedOffset, Utc};

const JST_OFFSET_SECONDS: i32 = 9 * 60 * 60;

fn jst_offset() -> FixedOffset {
    FixedOffset::east_opt(JST_OFFSET_SECONDS).expect("JST offset is valid")
}

pub(crate) fn format_jst_datetime(at: DateTime<Utc>) -> String {
    at.with_timezone(&jst_offset())
        .format("%Y-%m-%d %H:%M:%S")
        .to_string()
}

pub(crate) fn format_jst_time(at: DateTime<Utc>) -> String {
    at.with_timezone(&jst_offset())
        .format("%H:%M:%S")
        .to_string()
}

#[cfg(test)]
mod tests {
    use chrono::{TimeZone, Utc};

    use super::*;

    #[test]
    fn format_jst_datetime_converts_utc_to_jst_without_suffix() {
        let formatted = format_jst_datetime(Utc.with_ymd_and_hms(2026, 5, 13, 0, 0, 0).unwrap());

        assert_eq!(formatted, "2026-05-13 09:00:00");
        assert!(!formatted.contains('Z'));
        assert!(!formatted.contains("+09:00"));
        assert!(!formatted.contains("JST"));
    }

    #[test]
    fn format_jst_time_omits_milliseconds() {
        let formatted = format_jst_time(Utc.timestamp_opt(1_778_630_400, 123_000_000).unwrap());

        assert_eq!(formatted, "09:00:00");
        assert!(!formatted.contains('.'));
    }
}
