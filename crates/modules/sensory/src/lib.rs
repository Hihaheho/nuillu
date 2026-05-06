use std::sync::Arc;

use anyhow::{Context, Result};
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use lutum::{Session, StructuredTurnOutcome};
use nuillu_module::{LlmAccess, Memo, Module, SensoryInput, SensoryInputInbox, ports::Clock};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

const SYSTEM_PROMPT: &str = r#"You are the sensory module.
You receive raw observations from the environment and decide whether each is notable
enough to record. If notable, produce a concise normalized description in plain text.
Do not write to the attention stream, memory, or emit utterances — your only output
is the normalized observation text written into your memo."#;

const MAX_OBSERVATIONS: usize = 20;

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

pub struct SensoryModule {
    inbox: SensoryInputInbox,
    memo: Memo,
    clock: Arc<dyn Clock>,
    llm: LlmAccess,
    observations: Vec<ObservationRecord>,
}

impl SensoryModule {
    pub fn new(
        inbox: SensoryInputInbox,
        memo: Memo,
        clock: Arc<dyn Clock>,
        llm: LlmAccess,
    ) -> Self {
        Self {
            inbox,
            memo,
            clock,
            llm,
            observations: Vec::new(),
        }
    }

    fn format_age(now: DateTime<Utc>, observed_at: DateTime<Utc>) -> String {
        let secs = (now - observed_at).num_seconds().max(0) as u64;
        if secs < 60 {
            format!("{secs} seconds ago")
        } else if secs < 3600 {
            let m = secs / 60;
            let s = secs % 60;
            format!("{m} minute{} {s} second{} ago", plural(m), plural(s))
        } else if secs < 86400 {
            let h = secs / 3600;
            let m = (secs % 3600) / 60;
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

        let prompt = serde_json::json!({
            "kind": kind,
            "direction": direction,
            "content": raw_content,
            "relative_age": relative_age,
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
            kind: if kind == "heard" { "heard" } else { "seen" },
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
            let envelope = self.inbox.next_item().await?;
            let _ = self.handle(envelope.body).await;
        }
    }
}

fn plural(n: u64) -> &'static str {
    if n == 1 { "" } else { "s" }
}

#[async_trait(?Send)]
impl Module for SensoryModule {
    async fn run(&mut self) {
        if let Err(error) = self.run_loop().await {
            tracing::debug!(?error, "sensory module loop stopped");
        }
    }
}
