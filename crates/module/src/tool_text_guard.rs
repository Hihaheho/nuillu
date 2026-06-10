use std::{error::Error, fmt};

use lutum::{
    AgentError, TextToolEventHandler, TextToolHandlerContext, TextToolHandlerDirective,
    TextToolHandlerResult, TextTurnEventWithTools, Toolset,
};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ToolNameInAssistantText {
    pub tool_name: String,
    pub text_suffix: String,
}

impl fmt::Display for ToolNameInAssistantText {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "assistant text contained available tool name `{}` near text suffix {:?}",
            self.tool_name, self.text_suffix
        )
    }
}

impl Error for ToolNameInAssistantText {}

#[derive(Debug, Default)]
pub struct AbortOnAvailableToolNameInText {
    text: String,
    deferred: Option<ToolNameInAssistantText>,
    saw_tool_call: bool,
}

impl AbortOnAvailableToolNameInText {
    pub fn new() -> Self {
        Self::default()
    }
}

#[cfg_attr(target_family = "wasm", async_trait::async_trait(?Send))]
#[cfg_attr(not(target_family = "wasm"), async_trait::async_trait)]
impl<T> TextToolEventHandler<T> for AbortOnAvailableToolNameInText
where
    T: Toolset,
{
    async fn on_event(
        &mut self,
        event: &TextTurnEventWithTools<T>,
        cx: &TextToolHandlerContext<T>,
    ) -> TextToolHandlerResult<T> {
        match event {
            TextTurnEventWithTools::Started { .. } | TextTurnEventWithTools::WillRetry { .. } => {
                self.text.clear();
                self.deferred = None;
                self.saw_tool_call = false;
            }
            TextTurnEventWithTools::TextDelta { delta } => {
                self.text.push_str(delta);

                let matched = cx
                    .tool_definitions()
                    .iter()
                    .filter_map(|definition| {
                        let name = definition.name.as_str();
                        classify_tool_name_mention(&self.text, name)
                            .map(|disposition| (name, disposition))
                    })
                    .max_by_key(|(name, disposition)| {
                        (
                            *disposition == ToolNameMentionDisposition::AbortImmediately,
                            name.len(),
                        )
                    });

                if let Some((tool_name, disposition)) = matched {
                    let text_suffix = tail_chars(&self.text, 160);
                    let violation = ToolNameInAssistantText {
                        tool_name: tool_name.to_owned(),
                        text_suffix,
                    };

                    if disposition == ToolNameMentionDisposition::DeferUntilTurnOutcome {
                        self.deferred = Some(violation);
                        return Ok(TextToolHandlerDirective::Continue);
                    }

                    tracing::warn!(
                        tool_name = tool_name,
                        text_suffix = %violation.text_suffix,
                        disposition = "immediate",
                        "assistant text contained available tool name"
                    );
                    return Err(AgentError::other(violation));
                }
            }
            TextTurnEventWithTools::ToolCallChunk { .. }
            | TextTurnEventWithTools::ToolCallReady(_) => {
                self.saw_tool_call = true;
            }
            TextTurnEventWithTools::Completed { .. } => {
                if !self.saw_tool_call
                    && let Some(violation) = self.deferred.take()
                {
                    tracing::warn!(
                        tool_name = %violation.tool_name,
                        text_suffix = %violation.text_suffix,
                        disposition = "deferred",
                        "assistant text contained available tool name without a native tool call"
                    );
                    return Err(AgentError::other(violation));
                }
            }
            _ => {}
        }

        Ok(TextToolHandlerDirective::Continue)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ToolNameMentionDisposition {
    AbortImmediately,
    DeferUntilTurnOutcome,
}

fn classify_tool_name_mention(text: &str, name: &str) -> Option<ToolNameMentionDisposition> {
    if name.is_empty() {
        return None;
    }

    let mut deferred = false;
    for (start, _) in text.match_indices(name) {
        let before = text[..start].chars().next_back();
        let after = text[start + name.len()..].chars().next();

        if before.is_some_and(is_tool_name_char) || after.is_some_and(is_tool_name_char) {
            continue;
        }

        if is_immediate_tool_call_text(text, start, name) {
            return Some(ToolNameMentionDisposition::AbortImmediately);
        }
        deferred = true;
    }

    deferred.then_some(ToolNameMentionDisposition::DeferUntilTurnOutcome)
}

fn is_immediate_tool_call_text(text: &str, start: usize, name: &str) -> bool {
    let line_start = text[..start]
        .rfind('\n')
        .map(|index| index + 1)
        .unwrap_or(0);
    let line_end = text[start + name.len()..]
        .find('\n')
        .map(|index| start + name.len() + index)
        .unwrap_or(text.len());
    let before_on_line = text[line_start..start].trim();
    let after_on_line = text[start + name.len()..line_end].trim_start();
    let line_is_closed = text[start + name.len()..].contains('\n');

    before_on_line.is_empty()
        && ((after_on_line.is_empty() && line_is_closed)
            || after_on_line.starts_with('(')
            || after_on_line.starts_with('{')
            || after_on_line.starts_with(':'))
}

fn is_tool_name_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '_' || c == '-'
}

fn tail_chars(s: &str, max_chars: usize) -> String {
    let mut chars = s.chars().rev().take(max_chars).collect::<Vec<_>>();
    chars.reverse();
    chars.into_iter().collect()
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use lutum::{
        FinishReason, Lutum, MockLlmAdapter, MockTextScenario, ModelInput, RawTextTurnEvent,
        SharedPoolBudgetManager, SharedPoolBudgetOptions, Usage,
    };
    use schemars::JsonSchema;
    use serde::{Deserialize, Serialize};

    #[lutum::tool_input(name = "search_memory", output = SearchMemoryOutput)]
    #[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
    struct SearchMemoryArgs {
        query: String,
    }

    #[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
    struct SearchMemoryOutput {
        found: bool,
    }

    #[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema, lutum::Toolset)]
    enum GuardTools {
        SearchMemory(SearchMemoryArgs),
    }

    fn contains_tool_name(text: &str, name: &str) -> bool {
        classify_tool_name_mention(text, name).is_some()
    }

    #[test]
    fn contains_tool_name_requires_token_boundaries() {
        assert!(contains_tool_name(
            "call promote_to_cognition with the current item",
            "promote_to_cognition"
        ));
        assert!(contains_tool_name(
            "try `leave-cognition-unchanged` instead",
            "leave-cognition-unchanged"
        ));
        assert!(!contains_tool_name(
            "call xpromote_to_cognition",
            "promote_to_cognition"
        ));
        assert!(!contains_tool_name(
            "call promote_to_cognition_now",
            "promote_to_cognition"
        ));
        assert!(!contains_tool_name(
            "call promote_to_cognition-now",
            "promote_to_cognition"
        ));
    }

    #[test]
    fn tail_chars_keeps_character_boundaries() {
        assert_eq!(tail_chars("abcあいう", 4), "cあいう");
    }

    #[tokio::test(flavor = "current_thread")]
    async fn collect_aborts_when_deferred_tool_name_finishes_without_native_call() {
        let adapter = Arc::new(
            MockLlmAdapter::new().with_text_scenario(MockTextScenario::events(vec![
                Ok(RawTextTurnEvent::Started {
                    request_id: Some("guard".into()),
                    model: "mock".into(),
                }),
                Ok(RawTextTurnEvent::TextDelta {
                    delta: "I will call search_".into(),
                }),
                Ok(RawTextTurnEvent::TextDelta {
                    delta: "memory next".into(),
                }),
                Ok(RawTextTurnEvent::Completed {
                    request_id: Some("guard".into()),
                    finish_reason: FinishReason::Stop,
                    usage: Usage::zero(),
                }),
            ])),
        );
        let lutum = Lutum::new(
            adapter,
            SharedPoolBudgetManager::new(SharedPoolBudgetOptions::default()),
        );

        let error = lutum
            .text_turn(ModelInput::new().user("find notes"))
            .tools::<GuardTools>()
            .available_tools([GuardToolsSelector::SearchMemory])
            .collect_controlled_with(AbortOnAvailableToolNameInText::new())
            .await
            .expect_err("assistant text containing an available tool name should abort");

        let lutum::CollectError::Handler { source, .. } = error else {
            panic!("expected handler error");
        };
        let source = source
            .downcast_other::<ToolNameInAssistantText>()
            .expect("handler error should preserve guard source");
        assert_eq!(source.tool_name, "search_memory");
        assert_eq!(source.text_suffix, "I will call search_memory next");
        assert!(
            source
                .to_string()
                .contains("I will call search_memory next")
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn collect_allows_deferred_tool_name_when_native_call_arrives() {
        let adapter = Arc::new(
            MockLlmAdapter::new().with_text_scenario(MockTextScenario::events(vec![
                Ok(RawTextTurnEvent::Started {
                    request_id: Some("guard".into()),
                    model: "mock".into(),
                }),
                Ok(RawTextTurnEvent::TextDelta {
                    delta: "I will call `search_memory` now.".into(),
                }),
                Ok(RawTextTurnEvent::ToolCallChunk {
                    id: "call-search".into(),
                    name: "search_memory".into(),
                    arguments_json_delta: serde_json::json!({ "query": "notes" }).to_string(),
                }),
                Ok(RawTextTurnEvent::Completed {
                    request_id: Some("guard".into()),
                    finish_reason: FinishReason::ToolCall,
                    usage: Usage::zero(),
                }),
            ])),
        );
        let lutum = Lutum::new(
            adapter,
            SharedPoolBudgetManager::new(SharedPoolBudgetOptions::default()),
        );

        lutum
            .text_turn(ModelInput::new().user("find notes"))
            .tools::<GuardTools>()
            .available_tools([GuardToolsSelector::SearchMemory])
            .collect_controlled_with(AbortOnAvailableToolNameInText::new())
            .await
            .expect("assistant text may mention a tool name when the turn includes a native call");
    }

    #[tokio::test(flavor = "current_thread")]
    async fn collect_aborts_immediately_on_function_shaped_tool_text() {
        let adapter = Arc::new(
            MockLlmAdapter::new().with_text_scenario(MockTextScenario::events(vec![
                Ok(RawTextTurnEvent::Started {
                    request_id: Some("guard".into()),
                    model: "mock".into(),
                }),
                Ok(RawTextTurnEvent::TextDelta {
                    delta: "search_memory(".into(),
                }),
                Ok(RawTextTurnEvent::ToolCallChunk {
                    id: "call-search".into(),
                    name: "search_memory".into(),
                    arguments_json_delta: serde_json::json!({ "query": "notes" }).to_string(),
                }),
                Ok(RawTextTurnEvent::Completed {
                    request_id: Some("guard".into()),
                    finish_reason: FinishReason::ToolCall,
                    usage: Usage::zero(),
                }),
            ])),
        );
        let lutum = Lutum::new(
            adapter,
            SharedPoolBudgetManager::new(SharedPoolBudgetOptions::default()),
        );

        let error = lutum
            .text_turn(ModelInput::new().user("find notes"))
            .tools::<GuardTools>()
            .available_tools([GuardToolsSelector::SearchMemory])
            .collect_controlled_with(AbortOnAvailableToolNameInText::new())
            .await
            .expect_err("function-shaped tool text should abort before later events");

        let lutum::CollectError::Handler { source, .. } = error else {
            panic!("expected handler error");
        };
        let source = source
            .downcast_other::<ToolNameInAssistantText>()
            .expect("handler error should preserve guard source");
        assert_eq!(source.tool_name, "search_memory");
        assert_eq!(source.text_suffix, "search_memory(");
    }
}
