use anyhow::{Context, Result};
use lutum::{InputMessageRole, Lutum, ModelInput, ModelInputItem, Session};

pub const DEFAULT_SESSION_COMPACTION_INPUT_TOKEN_THRESHOLD: u64 = 16_000;
pub const DEFAULT_SESSION_COMPACTION_PREFIX_RATIO: f64 = 0.8;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SessionCompactionConfig {
    pub input_token_threshold: u64,
    pub prefix_ratio: f64,
}

impl Default for SessionCompactionConfig {
    fn default() -> Self {
        Self {
            input_token_threshold: DEFAULT_SESSION_COMPACTION_INPUT_TOKEN_THRESHOLD,
            prefix_ratio: DEFAULT_SESSION_COMPACTION_PREFIX_RATIO,
        }
    }
}

pub fn session_compaction_cutoff(item_count: usize, prefix_ratio: f64) -> Option<usize> {
    if item_count < 2 {
        return None;
    }
    let ratio = if prefix_ratio.is_finite() {
        prefix_ratio.clamp(f64::EPSILON, 1.0)
    } else {
        DEFAULT_SESSION_COMPACTION_PREFIX_RATIO
    };
    let cutoff = ((item_count as f64) * ratio).floor() as usize;
    Some(cutoff.clamp(1, item_count.saturating_sub(1)))
}

pub async fn compact_session_if_needed(
    session: &mut Session,
    input_tokens: u64,
    lutum: &Lutum,
    config: SessionCompactionConfig,
    module_name: &str,
    compacted_prefix: &str,
    compaction_prompt: &str,
) {
    if input_tokens <= config.input_token_threshold {
        return;
    }
    if let Err(error) =
        compact_session(session, lutum, config, compacted_prefix, compaction_prompt).await
    {
        tracing::warn!(
            module = module_name,
            input_tokens,
            threshold = config.input_token_threshold,
            error = ?error,
            "module session compaction failed"
        );
    }
}

pub async fn compact_session(
    session: &mut Session,
    lutum: &Lutum,
    config: SessionCompactionConfig,
    compacted_prefix: &str,
    compaction_prompt: &str,
) -> Result<()> {
    let items = session.input().items();
    let Some(cutoff) = session_compaction_cutoff(items.len(), config.prefix_ratio) else {
        return Ok(());
    };

    let prefix = items[..cutoff].to_vec();
    let suffix = items[cutoff..].to_vec();

    let mut summary_items = Vec::with_capacity(prefix.len().saturating_add(1));
    summary_items.push(ModelInputItem::text(
        InputMessageRole::System,
        compaction_prompt,
    ));
    summary_items.extend(prefix);
    let summary = lutum
        .text_turn(ModelInput::from_items(summary_items))
        .collect()
        .await
        .context("summarize session prefix")?
        .assistant_text();
    let summary = summary.trim();
    if summary.is_empty() {
        tracing::warn!("module session compaction produced an empty summary");
        return Ok(());
    }

    let mut compacted_items = Vec::with_capacity(suffix.len().saturating_add(1));
    compacted_items.push(ModelInputItem::text(
        InputMessageRole::System,
        format!("{compacted_prefix}\n{summary}"),
    ));
    compacted_items.extend(suffix);
    *session = Session::from_input(ModelInput::from_items(compacted_items));
    Ok(())
}
