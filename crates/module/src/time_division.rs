use std::sync::Arc;
use std::time::Duration;

use eure::FromEure;
use serde::Serialize;
use thiserror::Error;

const DEFAULT_TIME_DIVISION_CONFIG: &str = include_str!("../../../configs/time-division.eure");

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct TimeDivisionBucket {
    pub tag: String,
    pub range: Duration,
}

#[derive(Debug, Clone)]
pub struct TimeDivision {
    fallback_longest_tag: String,
    buckets: Arc<[TimeDivisionBucket]>,
}

impl TimeDivision {
    pub fn new(
        fallback_longest_tag: impl Into<String>,
        buckets: impl Into<Vec<TimeDivisionBucket>>,
    ) -> Self {
        let mut buckets = buckets.into();
        buckets.sort_by_key(|bucket| bucket.range);
        Self {
            fallback_longest_tag: fallback_longest_tag.into(),
            buckets: buckets.into(),
        }
    }

    pub fn default_policy() -> Self {
        Self::from_eure_str(DEFAULT_TIME_DIVISION_CONFIG)
            .expect("bundled configs/time-division.eure must be valid")
    }

    pub fn from_eure_str(content: &str) -> Result<Self, TimeDivisionError> {
        let parsed: TimeDivisionConfig =
            eure::parse_content(content, "configs/time-division.eure".into())
                .map_err(|message| TimeDivisionError::Parse { message })?;
        Self::from_config(parsed)
    }

    pub fn tag_for_age(&self, age: Duration) -> &str {
        self.buckets
            .iter()
            .find(|bucket| age <= bucket.range)
            .map(|bucket| bucket.tag.as_str())
            .unwrap_or(&self.fallback_longest_tag)
    }

    pub fn fallback_longest_tag(&self) -> &str {
        &self.fallback_longest_tag
    }

    pub fn buckets(&self) -> &[TimeDivisionBucket] {
        &self.buckets
    }
}

impl Default for TimeDivision {
    fn default() -> Self {
        Self::default_policy()
    }
}

#[derive(Debug, Clone, FromEure)]
#[eure(crate = ::eure::document, rename_all = "kebab-case")]
struct TimeDivisionConfig {
    fallback_longest_tag: String,
    #[eure(default)]
    tags: Vec<TimeDivisionTagConfig>,
}

#[derive(Debug, Clone, FromEure)]
#[eure(crate = ::eure::document, rename_all = "kebab-case")]
struct TimeDivisionTagConfig {
    tag: String,
    range_sec: f64,
}

impl TimeDivision {
    fn from_config(config: TimeDivisionConfig) -> Result<Self, TimeDivisionError> {
        let fallback_longest_tag = config.fallback_longest_tag.trim();
        if fallback_longest_tag.is_empty() {
            return Err(TimeDivisionError::InvalidConfig {
                message: "fallback-longest-tag must not be empty".into(),
            });
        }
        if config.tags.is_empty() {
            return Err(TimeDivisionError::InvalidConfig {
                message: "at least one time-division tag is required".into(),
            });
        }

        let mut buckets = Vec::with_capacity(config.tags.len());
        for tag in config.tags {
            let tag_name = tag.tag.trim();
            if tag_name.is_empty() {
                return Err(TimeDivisionError::InvalidConfig {
                    message: "time-division tag must not be empty".into(),
                });
            }
            buckets.push(TimeDivisionBucket {
                tag: tag_name.to_owned(),
                range: duration_from_seconds(tag_name, tag.range_sec)?,
            });
        }

        Ok(Self::new(fallback_longest_tag.to_owned(), buckets))
    }
}

fn duration_from_seconds(tag: &str, seconds: f64) -> Result<Duration, TimeDivisionError> {
    if !seconds.is_finite() || seconds <= 0.0 {
        return Err(TimeDivisionError::InvalidConfig {
            message: format!("time-division tag {tag} must have a positive finite range-sec"),
        });
    }
    Duration::try_from_secs_f64(seconds).map_err(|_| TimeDivisionError::InvalidConfig {
        message: format!("time-division tag {tag} range-sec is outside Duration bounds"),
    })
}

#[derive(Debug, Error)]
pub enum TimeDivisionError {
    #[error("failed to parse time-division config: {message}")]
    Parse { message: String },
    #[error("invalid time-division config: {message}")]
    InvalidConfig { message: String },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tags_age_by_first_matching_bucket() {
        let policy = TimeDivision::default();

        assert_eq!(policy.tag_for_age(Duration::from_secs(2)), "now");
        assert_eq!(policy.tag_for_age(Duration::from_secs(30)), "last_30sec");
        assert_eq!(policy.tag_for_age(Duration::from_secs(90)), "last_2min");
        assert_eq!(
            policy.tag_for_age(Duration::from_secs(200_000)),
            "before_24hour"
        );
    }

    #[test]
    fn default_policy_loads_bundled_eure_config() {
        let policy = TimeDivision::default_policy();

        assert_eq!(policy.tag_for_age(Duration::from_secs(3)), "now");
        assert_eq!(policy.tag_for_age(Duration::from_secs(4)), "last_30sec");
        assert_eq!(policy.tag_for_age(Duration::from_secs(120)), "last_2min");
        assert_eq!(
            policy.tag_for_age(Duration::from_secs(3_601)),
            "last_24hour"
        );
    }

    #[test]
    fn rejects_empty_eure_bucket_set() {
        let err =
            TimeDivision::from_eure_str(r#"fallback-longest-tag = "before_all""#).unwrap_err();

        assert!(matches!(
            err,
            TimeDivisionError::InvalidConfig { message }
                if message.contains("at least one time-division tag")
        ));
    }
}
