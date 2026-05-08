use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{Context, Result};
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use lutum::{Session, StructuredTurnOutcome};
use nuillu_module::{
    ActivationGate, AllocationReader, LlmAccess, Memo, Module, SensoryInput, SensoryInputInbox,
    ports::Clock,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

const SYSTEM_PROMPT: &str = r#"You are the sensory module.
You receive raw observations from the environment and decide whether each is notable
enough to record.
Return a structured SensoryDecision object:
- notable: whether the observation should be recorded.
- normalized: the concise plain-text observation to write into your memo when notable is true.
Use the deterministic salience features as guidance; repeated low-change stimuli should usually
be ignored or folded into summary.
Do not write to the attention stream, memory, or emit utterances. Do not return a bare string;
the only textual observation belongs in the normalized field. Return only raw JSON for the
structured object; do not wrap it in Markdown or code fences."#;

const MAX_OBSERVATIONS: usize = 20;
const USER_DIRECTED_DIRECTIONS: &[&str] = &["user", "front"];

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
struct SensoryDecision {
    notable: bool,
    normalized: String,
}

struct ObservationRecord {
    kind: &'static str,
    direction: Option<String>,
    content: String,
    relative_age: String,
}

#[derive(Clone, Debug, Serialize)]
struct SalienceFeatures {
    signature: String,
    repeated: bool,
    repetition_count: u32,
    seconds_since_previous: Option<i64>,
    user_directed: bool,
    novelty_score: f32,
    salience_score: f32,
}

#[derive(Clone, Debug)]
struct StimulusState {
    last_observed_at: DateTime<Utc>,
    repetition_count: u32,
}

pub struct SensoryModule {
    inbox: SensoryInputInbox,
    gate: ActivationGate,
    allocation: AllocationReader,
    memo: Memo,
    clock: Arc<dyn Clock>,
    llm: LlmAccess,
    observations: Vec<ObservationRecord>,
    stimuli: HashMap<String, StimulusState>,
}

impl SensoryModule {
    pub fn new(
        inbox: SensoryInputInbox,
        gate: ActivationGate,
        allocation: AllocationReader,
        memo: Memo,
        clock: Arc<dyn Clock>,
        llm: LlmAccess,
    ) -> Self {
        Self {
            inbox,
            gate,
            allocation,
            memo,
            clock,
            llm,
            observations: Vec::new(),
            stimuli: HashMap::new(),
        }
    }

    fn format_age(now: DateTime<Utc>, observed_at: DateTime<Utc>) -> String {
        let secs = (now - observed_at).num_seconds().max(0) as u64;
        if secs < 60 {
            format!("{secs} seconds ago")
        } else if secs < 3600 {
            let mut m = secs / 60;
            let mut s = ((secs % 60) + 5) / 10 * 10;
            if s == 60 {
                m += 1;
                s = 0;
            }
            if m == 60 {
                return "1 hour 0 minutes ago".into();
            }
            format!("{m} minute{} {s} second{} ago", plural(m), plural(s))
        } else if secs < 86400 {
            let mut h = secs / 3600;
            let mut m = ((secs % 3600) / 60 + 5) / 10 * 10;
            if m == 60 {
                h += 1;
                m = 0;
            }
            format!("{h} hour{} {m} minute{} ago", plural(h), plural(m))
        } else {
            let d = secs / 86400;
            let h = (secs % 86400) / 3600;
            format!("{d} day{} {h} hour{} ago", plural(d), plural(h))
        }
    }

    #[tracing::instrument(skip_all, err(Debug, level = "warn"))]
    async fn handle(&mut self, input: SensoryInput) -> Result<()> {
        let now = self.clock.now();
        let (kind, direction, raw_content, observed_at) = match &input {
            SensoryInput::Heard {
                direction,
                content,
                observed_at,
            } => ("heard", direction.clone(), content.clone(), *observed_at),
            SensoryInput::Seen {
                direction,
                appearance,
                observed_at,
            } => ("seen", direction.clone(), appearance.clone(), *observed_at),
        };
        let relative_age = Self::format_age(now, observed_at);
        let salience =
            self.salience_features(kind, direction.as_deref(), &raw_content, observed_at);
        let allocation = self.allocation.snapshot().await;

        let prompt = serde_json::json!({
            "kind": kind,
            "direction": direction,
            "content": raw_content,
            "relative_age": relative_age,
            "salience": salience,
            "allocation": allocation,
        });

        let lutum = self.llm.lutum().await;
        let mut session = Session::new(lutum);
        session.push_system(SYSTEM_PROMPT);
        session.push_user(prompt.to_string());

        let result = session
            .structured_turn::<SensoryDecision>()
            .collect()
            .await
            .context("sensory structured turn failed")?;

        let StructuredTurnOutcome::Structured(decision) = result.semantic else {
            anyhow::bail!("sensory structured turn refused");
        };

        if !decision.notable {
            return Ok(());
        }

        self.observations.push(ObservationRecord {
            kind,
            direction,
            content: decision.normalized,
            relative_age,
        });
        if self.observations.len() > MAX_OBSERVATIONS {
            self.observations.remove(0);
        }

        self.memo.write(self.render_memo()).await;
        Ok(())
    }

    fn salience_features(
        &mut self,
        kind: &str,
        direction: Option<&str>,
        content: &str,
        observed_at: DateTime<Utc>,
    ) -> SalienceFeatures {
        let signature = format!(
            "{}:{}:{}",
            kind,
            direction.unwrap_or_default().to_ascii_lowercase(),
            content.trim().to_ascii_lowercase()
        );
        let previous = self.stimuli.get(&signature).cloned();
        let repetition_count = previous
            .as_ref()
            .map(|state| state.repetition_count.saturating_add(1))
            .unwrap_or(1);
        let seconds_since_previous = previous
            .as_ref()
            .map(|state| (observed_at - state.last_observed_at).num_seconds().max(0));
        let repeated = previous.is_some();
        let user_directed = direction.map(is_user_directed_direction).unwrap_or(false);
        let novelty_score = if repeated { 0.15 } else { 0.85 };
        let repetition_penalty = if repeated {
            (repetition_count.saturating_sub(1) as f32 * 0.15).min(0.45)
        } else {
            0.0
        };
        let salience_score = (novelty_score + if user_directed { 0.2 } else { 0.0 }
            - repetition_penalty)
            .clamp(0.0, 1.0);

        self.stimuli.insert(
            signature.clone(),
            StimulusState {
                last_observed_at: observed_at,
                repetition_count,
            },
        );

        SalienceFeatures {
            signature,
            repeated,
            repetition_count,
            seconds_since_previous,
            user_directed,
            novelty_score,
            salience_score,
        }
    }

    fn render_memo(&self) -> String {
        self.observations
            .iter()
            .map(|o| {
                let dir = o
                    .direction
                    .as_deref()
                    .map(|d| format!(" [{d}]"))
                    .unwrap_or_default();
                format!("[{}{}] {} ({})", o.kind, dir, o.content, o.relative_age)
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    async fn run_loop(&mut self) -> Result<()> {
        loop {
            for input in self.next_batch().await? {
                self.handle(input).await?;
            }
        }
    }

    async fn next_batch(&mut self) -> Result<Vec<SensoryInput>> {
        let first = self.inbox.next_item().await?.body;
        let mut batch = vec![first];
        batch.extend(
            self.inbox
                .take_ready_items()?
                .items
                .into_iter()
                .map(|envelope| envelope.body),
        );
        self.gate.block().await;
        batch.extend(
            self.inbox
                .take_ready_items()?
                .items
                .into_iter()
                .map(|envelope| envelope.body),
        );
        Ok(batch)
    }
}

fn plural(n: u64) -> &'static str {
    if n == 1 { "" } else { "s" }
}

fn is_user_directed_direction(direction: &str) -> bool {
    let direction = direction.to_ascii_lowercase();
    USER_DIRECTED_DIRECTIONS.contains(&direction.as_str())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reference_now() -> DateTime<Utc> {
        DateTime::parse_from_rfc3339("2026-05-07T12:00:00Z")
            .unwrap()
            .with_timezone(&Utc)
    }

    #[test]
    fn format_age_rounds_boundary_values() {
        let now = reference_now();

        assert_eq!(
            SensoryModule::format_age(now, now - chrono::Duration::seconds(59)),
            "59 seconds ago"
        );
        assert_eq!(
            SensoryModule::format_age(now, now - chrono::Duration::seconds(60)),
            "1 minute 0 seconds ago"
        );
        assert_eq!(
            SensoryModule::format_age(now, now - chrono::Duration::seconds(3599)),
            "1 hour 0 minutes ago"
        );
        assert_eq!(
            SensoryModule::format_age(now, now - chrono::Duration::seconds(3600)),
            "1 hour 0 minutes ago"
        );
    }

    #[test]
    fn user_directed_direction_is_generic_not_username_specific() {
        assert!(is_user_directed_direction("user"));
        assert!(is_user_directed_direction("front"));
        assert!(is_user_directed_direction("USER"));
        assert!(!is_user_directed_direction("ryo"));
        assert!(!is_user_directed_direction("left"));
    }
}

#[async_trait(?Send)]
impl Module for SensoryModule {
    async fn run(&mut self) {
        if let Err(error) = self.run_loop().await {
            panic!("sensory module failed: {error:#}");
        }
    }
}
