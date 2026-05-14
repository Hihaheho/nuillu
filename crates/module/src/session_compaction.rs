use anyhow::{Context, Result};
use lutum::{AssistantInputItem, InputMessageRole, Lutum, ModelInput, ModelInputItem, Session};
use nuillu_types::ModelTier;

pub const DEFAULT_SESSION_COMPACTION_INPUT_TOKEN_THRESHOLD: u64 = 16_000;
pub const DEFAULT_SESSION_COMPACTION_PREFIX_RATIO: f64 = 0.8;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SessionCompactionConfig {
    pub prefix_ratio: f64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SessionCompactionPolicy {
    pub cheap_input_token_threshold: u64,
    pub default_input_token_threshold: u64,
    pub premium_input_token_threshold: u64,
}

#[derive(Clone)]
pub struct SessionCompactionRuntime {
    lutum: Lutum,
    module_tier: ModelTier,
    policy: SessionCompactionPolicy,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SessionCompactionProtectedPrefix {
    None,
    LeadingSystem,
    LeadingSystemAndIdentitySeed,
    Count(usize),
}

impl Default for SessionCompactionConfig {
    fn default() -> Self {
        Self {
            prefix_ratio: DEFAULT_SESSION_COMPACTION_PREFIX_RATIO,
        }
    }
}

impl SessionCompactionPolicy {
    pub const fn new(
        cheap_input_token_threshold: u64,
        default_input_token_threshold: u64,
        premium_input_token_threshold: u64,
    ) -> Self {
        Self {
            cheap_input_token_threshold,
            default_input_token_threshold,
            premium_input_token_threshold,
        }
    }

    pub fn threshold_for(&self, tier: ModelTier) -> u64 {
        match tier {
            ModelTier::Cheap => self.cheap_input_token_threshold,
            ModelTier::Default => self.default_input_token_threshold,
            ModelTier::Premium => self.premium_input_token_threshold,
        }
    }
}

impl Default for SessionCompactionPolicy {
    fn default() -> Self {
        Self::new(
            DEFAULT_SESSION_COMPACTION_INPUT_TOKEN_THRESHOLD,
            DEFAULT_SESSION_COMPACTION_INPUT_TOKEN_THRESHOLD,
            DEFAULT_SESSION_COMPACTION_INPUT_TOKEN_THRESHOLD,
        )
    }
}

impl SessionCompactionRuntime {
    pub fn new(lutum: Lutum, module_tier: ModelTier, policy: SessionCompactionPolicy) -> Self {
        Self {
            lutum,
            module_tier,
            policy,
        }
    }

    pub fn lutum(&self) -> &Lutum {
        &self.lutum
    }

    pub fn module_tier(&self) -> ModelTier {
        self.module_tier
    }

    pub fn policy(&self) -> SessionCompactionPolicy {
        self.policy
    }

    pub fn input_token_threshold(&self) -> u64 {
        self.policy.threshold_for(self.module_tier)
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
    runtime: &SessionCompactionRuntime,
    config: SessionCompactionConfig,
    protected_prefix: SessionCompactionProtectedPrefix,
    module_name: &str,
    compacted_prefix: &str,
    compaction_prompt: &str,
) {
    let threshold = runtime.input_token_threshold();
    if input_tokens <= threshold {
        return;
    }
    if let Err(error) = compact_session(
        session,
        runtime.lutum(),
        config,
        protected_prefix,
        compacted_prefix,
        compaction_prompt,
    )
    .await
    {
        tracing::warn!(
            module = module_name,
            input_tokens,
            threshold,
            module_tier = ?runtime.module_tier(),
            error = ?error,
            "module session compaction failed"
        );
    }
}

pub async fn compact_session(
    session: &mut Session,
    lutum: &Lutum,
    config: SessionCompactionConfig,
    protected_prefix: SessionCompactionProtectedPrefix,
    compacted_prefix: &str,
    compaction_prompt: &str,
) -> Result<()> {
    let items = session.input().items();
    let protected_prefix_len = protected_prefix_len(items, protected_prefix);
    let compactable_len = items.len().saturating_sub(protected_prefix_len);
    let Some(relative_cutoff) = session_compaction_cutoff(compactable_len, config.prefix_ratio)
    else {
        return Ok(());
    };
    let cutoff = protected_prefix_len + relative_cutoff;

    let protected_prefix = items[..protected_prefix_len].to_vec();
    let prefix = items[protected_prefix_len..cutoff].to_vec();
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

    let mut compacted_items = Vec::with_capacity(
        protected_prefix
            .len()
            .saturating_add(suffix.len())
            .saturating_add(1),
    );
    compacted_items.extend(protected_prefix);
    compacted_items.push(ModelInputItem::assistant_text(format!(
        "{compacted_prefix}\n{summary}"
    )));
    compacted_items.extend(suffix);
    *session = Session::from_input(ModelInput::from_items(compacted_items));
    Ok(())
}

fn protected_prefix_len(
    items: &[ModelInputItem],
    protected_prefix: SessionCompactionProtectedPrefix,
) -> usize {
    match protected_prefix {
        SessionCompactionProtectedPrefix::None => 0,
        SessionCompactionProtectedPrefix::LeadingSystem => leading_system_len(items),
        SessionCompactionProtectedPrefix::LeadingSystemAndIdentitySeed => {
            let mut len = leading_system_len(items);
            if matches!(
                items.get(len),
                Some(ModelInputItem::Assistant(AssistantInputItem::Text(text)))
                    if text.starts_with("What I already remember about myself")
            ) {
                len += 1;
            }
            len
        }
        SessionCompactionProtectedPrefix::Count(len) => len.min(items.len()),
    }
}

fn leading_system_len(items: &[ModelInputItem]) -> usize {
    match items.first() {
        Some(ModelInputItem::Message {
            role: InputMessageRole::System,
            ..
        }) => 1,
        _ => 0,
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use async_trait::async_trait;
    use lutum::{
        AdapterStructuredTurn, AdapterTextTurn, AgentError, ErasedStructuredTurnEventStream,
        ErasedTextTurnEventStream, FinishReason, MessageContent, MockLlmAdapter, MockTextScenario,
        RawTextTurnEvent, SharedPoolBudgetManager, SharedPoolBudgetOptions, TurnAdapter, Usage,
    };
    use nuillu_types::ModelTier;

    use super::*;

    #[derive(Clone)]
    struct CapturingAdapter {
        inner: MockLlmAdapter,
        text_inputs: Arc<Mutex<Vec<ModelInput>>>,
    }

    impl CapturingAdapter {
        fn new(inner: MockLlmAdapter) -> Self {
            Self {
                inner,
                text_inputs: Arc::new(Mutex::new(Vec::new())),
            }
        }

        fn text_inputs(&self) -> Vec<ModelInput> {
            self.text_inputs.lock().unwrap().clone()
        }
    }

    #[async_trait]
    impl TurnAdapter for CapturingAdapter {
        async fn text_turn(
            &self,
            input: ModelInput,
            turn: AdapterTextTurn,
        ) -> Result<ErasedTextTurnEventStream, AgentError> {
            self.text_inputs.lock().unwrap().push(input.clone());
            self.inner.text_turn(input, turn).await
        }

        async fn structured_turn(
            &self,
            input: ModelInput,
            turn: AdapterStructuredTurn,
        ) -> Result<ErasedStructuredTurnEventStream, AgentError> {
            self.inner.structured_turn(input, turn).await
        }
    }

    fn lutum_with_adapter(adapter: CapturingAdapter) -> (Lutum, CapturingAdapter) {
        let observed = adapter.clone();
        let budget = SharedPoolBudgetManager::new(SharedPoolBudgetOptions::default());
        let lutum = Lutum::new(Arc::new(adapter), budget);
        (lutum, observed)
    }

    fn compaction_runtime(
        lutum: Lutum,
        module_tier: ModelTier,
        threshold: u64,
    ) -> SessionCompactionRuntime {
        SessionCompactionRuntime::new(
            lutum,
            module_tier,
            SessionCompactionPolicy::new(threshold, threshold, threshold),
        )
    }

    fn summary_scenario(summary: &str) -> MockTextScenario {
        MockTextScenario::events(vec![
            Ok(RawTextTurnEvent::Started {
                request_id: Some("compact".into()),
                model: "mock".into(),
            }),
            Ok(RawTextTurnEvent::TextDelta {
                delta: summary.into(),
            }),
            Ok(RawTextTurnEvent::Completed {
                request_id: Some("compact".into()),
                finish_reason: FinishReason::Stop,
                usage: Usage::zero(),
            }),
        ])
    }

    fn text_of_user_messages(items: &[ModelInputItem]) -> Vec<&str> {
        items
            .iter()
            .filter_map(|item| match item {
                ModelInputItem::Message {
                    role: InputMessageRole::User,
                    content,
                } => {
                    let [MessageContent::Text(text)] = content.as_slice() else {
                        panic!("expected one text content item");
                    };
                    Some(text.as_str())
                }
                _ => None,
            })
            .collect()
    }

    fn assistant_text(item: &ModelInputItem) -> &str {
        let ModelInputItem::Assistant(AssistantInputItem::Text(text)) = item else {
            panic!("expected assistant text item");
        };
        text
    }

    #[test]
    fn policy_resolves_threshold_by_module_tier() {
        let policy = SessionCompactionPolicy::new(11, 22, 33);

        assert_eq!(policy.threshold_for(ModelTier::Cheap), 11);
        assert_eq!(policy.threshold_for(ModelTier::Default), 22);
        assert_eq!(policy.threshold_for(ModelTier::Premium), 33);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn below_runtime_threshold_does_not_call_compaction_llm() {
        let adapter = CapturingAdapter::new(
            MockLlmAdapter::new().with_text_scenario(summary_scenario("unexpected")),
        );
        let (lutum, observed) = lutum_with_adapter(adapter);
        let runtime = compaction_runtime(lutum, ModelTier::Default, 42);
        let mut session = Session::new();
        session.push_user("history-0");
        session.push_user("history-1");

        compact_session_if_needed(
            &mut session,
            42,
            &runtime,
            SessionCompactionConfig::default(),
            SessionCompactionProtectedPrefix::None,
            "test-module",
            "Compacted session:",
            "Summarize.",
        )
        .await;

        assert!(observed.text_inputs().is_empty());
        assert_eq!(
            text_of_user_messages(session.input().items()),
            vec!["history-0", "history-1"]
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn above_runtime_threshold_compacts() {
        let adapter = CapturingAdapter::new(
            MockLlmAdapter::new().with_text_scenario(summary_scenario("old history")),
        );
        let (lutum, observed) = lutum_with_adapter(adapter);
        let runtime = compaction_runtime(lutum, ModelTier::Default, 42);
        let mut session = Session::new();
        session.push_user("history-0");
        session.push_user("history-1");
        session.push_user("history-2");

        compact_session_if_needed(
            &mut session,
            43,
            &runtime,
            SessionCompactionConfig::default(),
            SessionCompactionProtectedPrefix::None,
            "test-module",
            "Compacted session:",
            "Summarize.",
        )
        .await;

        assert_eq!(observed.text_inputs().len(), 1);
        assert_eq!(
            assistant_text(&session.input().items()[0]),
            "Compacted session:\nold history"
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn protects_system_and_identity_and_summarizes_tail_only() {
        let adapter = CapturingAdapter::new(
            MockLlmAdapter::new().with_text_scenario(summary_scenario("history summarized")),
        );
        let (lutum, observed) = lutum_with_adapter(adapter);
        let mut session = Session::new();
        session.push_system("SYSTEM PROMPT");
        session.push_assistant_text(
            "What I already remember about myself at 2026-05-11T06:23:00Z:\n- identity",
        );
        for index in 0..5 {
            session.push_user(format!("history-{index}"));
        }

        compact_session(
            &mut session,
            &lutum,
            SessionCompactionConfig { prefix_ratio: 0.6 },
            SessionCompactionProtectedPrefix::LeadingSystemAndIdentitySeed,
            "Compacted session:",
            "Summarize.",
        )
        .await
        .unwrap();

        let items = session.input().items();
        let ModelInputItem::Message {
            role: InputMessageRole::System,
            content,
        } = &items[0]
        else {
            panic!("expected protected system prompt first");
        };
        let [MessageContent::Text(system)] = content.as_slice() else {
            panic!("expected system prompt text");
        };
        assert_eq!(system, "SYSTEM PROMPT");
        assert!(assistant_text(&items[1]).starts_with("What I already remember about myself"));
        assert_eq!(
            assistant_text(&items[2]),
            "Compacted session:\nhistory summarized"
        );
        assert_eq!(text_of_user_messages(items), vec!["history-3", "history-4"]);

        let captured = observed.text_inputs();
        assert_eq!(captured.len(), 1);
        let summary_items = captured[0].items();
        assert_eq!(summary_items.len(), 4);
        let summary_users = text_of_user_messages(summary_items);
        assert_eq!(summary_users, vec!["history-0", "history-1", "history-2"]);
        assert!(!summary_items.iter().any(|item| matches!(
            item,
            ModelInputItem::Message {
                role: InputMessageRole::System,
                content,
            } if matches!(content.as_slice(), [MessageContent::Text(text)] if text == "SYSTEM PROMPT")
        )));
        assert!(!summary_items.iter().any(|item| {
            matches!(
                item,
                ModelInputItem::Assistant(AssistantInputItem::Text(text))
                    if text.starts_with("What I already remember about myself")
            )
        }));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn none_compacts_from_start_and_inserts_assistant_summary_at_zero() {
        let adapter = CapturingAdapter::new(
            MockLlmAdapter::new().with_text_scenario(summary_scenario("all old history")),
        );
        let (lutum, _observed) = lutum_with_adapter(adapter);
        let mut session = Session::new();
        for index in 0..5 {
            session.push_user(format!("history-{index}"));
        }

        compact_session(
            &mut session,
            &lutum,
            SessionCompactionConfig { prefix_ratio: 0.6 },
            SessionCompactionProtectedPrefix::None,
            "Compacted session:",
            "Summarize.",
        )
        .await
        .unwrap();

        let items = session.input().items();
        assert_eq!(
            assistant_text(&items[0]),
            "Compacted session:\nall old history"
        );
        assert_eq!(text_of_user_messages(items), vec!["history-3", "history-4"]);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn count_prefix_is_clamped_to_session_len() {
        let adapter = CapturingAdapter::new(
            MockLlmAdapter::new().with_text_scenario(summary_scenario("unused")),
        );
        let (lutum, observed) = lutum_with_adapter(adapter);
        let mut session = Session::new();
        session.push_system("SYSTEM PROMPT");
        session.push_user("history-0");

        compact_session(
            &mut session,
            &lutum,
            SessionCompactionConfig { prefix_ratio: 0.8 },
            SessionCompactionProtectedPrefix::Count(99),
            "Compacted session:",
            "Summarize.",
        )
        .await
        .unwrap();

        assert!(observed.text_inputs().is_empty());
        let items = session.input().items();
        assert_eq!(items.len(), 2);
        assert!(matches!(
            &items[0],
            ModelInputItem::Message {
                role: InputMessageRole::System,
                ..
            }
        ));
        assert_eq!(text_of_user_messages(items), vec!["history-0"]);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn ratio_one_still_keeps_one_raw_suffix_item() {
        let adapter = CapturingAdapter::new(
            MockLlmAdapter::new().with_text_scenario(summary_scenario("nearly all history")),
        );
        let (lutum, _observed) = lutum_with_adapter(adapter);
        let mut session = Session::new();
        session.push_system("SYSTEM PROMPT");
        for index in 0..4 {
            session.push_user(format!("history-{index}"));
        }

        compact_session(
            &mut session,
            &lutum,
            SessionCompactionConfig { prefix_ratio: 1.0 },
            SessionCompactionProtectedPrefix::LeadingSystem,
            "Compacted session:",
            "Summarize.",
        )
        .await
        .unwrap();

        let items = session.input().items();
        assert_eq!(
            assistant_text(&items[1]),
            "Compacted session:\nnearly all history"
        );
        assert_eq!(text_of_user_messages(items), vec!["history-3"]);
    }
}
