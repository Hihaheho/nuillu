use async_trait::async_trait;
use lutum::{
    GenerationParams, Lutum, ModelInput, RequestBudget, StructuredTurnOutcome, Temperature,
};
use lutum_eval::{FieldValue, TraceSnapshot};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};
use thiserror::Error;

use crate::{
    artifact::CaseArtifact,
    cases::{RubricCriterion, RubricJudgeInput},
    evaluation::normalize_text_block,
};

const MAX_JSON_ARRAY_ITEMS: usize = 16;
const MAX_JSON_OBJECT_ENTRIES: usize = 32;
const MAX_JSON_STRING_CHARS: usize = 1000;
const MAX_RENDERED_SECTION_CHARS: usize = 6000;
const MAX_TRACE_SPANS: usize = 32;
const MAX_TRACE_EVENTS: usize = 32;
const MAX_TRACE_TEXT_CHARS: usize = 800;

#[derive(Debug, Clone)]
pub struct RubricJudgeRequest {
    pub prompt: String,
    pub context: Option<String>,
    pub rubric: String,
    pub criteria: Vec<RubricCriterion>,
    pub pass_score: f64,
    pub judge_inputs: Vec<RubricJudgeInput>,
    pub judge_max_output_tokens: u32,
    pub artifact: CaseArtifact,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct RubricJudgeVerdict {
    pub passed: bool,
    pub score: f64,
    pub summary: String,
    pub criteria: Vec<RubricJudgeVerdictCriterion>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct RubricJudgeVerdictCriterion {
    pub name: String,
    pub passed: bool,
    pub score: f64,
    pub reason: String,
    pub evidence: Option<String>,
}

#[async_trait(?Send)]
pub trait RubricJudge {
    async fn judge(
        &self,
        trace: &TraceSnapshot,
        request: RubricJudgeRequest,
    ) -> Result<RubricJudgeVerdict, RubricJudgeError>;
}

#[derive(Debug, Clone)]
pub struct JudgeOptions {
    pub temperature: Option<f32>,
    pub max_output_tokens: u32,
}

impl Default for JudgeOptions {
    fn default() -> Self {
        Self {
            temperature: None,
            max_output_tokens: 1200,
        }
    }
}

#[derive(Clone)]
pub struct LlmRubricJudge {
    llm: Lutum,
    concurrency: nuillu_module::LlmConcurrencyLimiter,
    options: JudgeOptions,
}

impl LlmRubricJudge {
    pub fn new(llm: Lutum) -> Self {
        Self::with_concurrency(llm, nuillu_module::LlmConcurrencyLimiter::new(None))
    }

    pub fn with_concurrency(llm: Lutum, concurrency: nuillu_module::LlmConcurrencyLimiter) -> Self {
        Self {
            llm,
            concurrency,
            options: JudgeOptions::default(),
        }
    }

    pub fn with_options(mut self, options: JudgeOptions) -> Self {
        self.options = options;
        self
    }
}

#[derive(Debug, Error)]
pub enum RubricJudgeError {
    #[error("invalid judge temperature {0}")]
    InvalidTemperature(f32),
    #[error("invalid rubric judge schema: {0}")]
    InvalidSchema(String),
    #[error("judge turn failed: {0}")]
    Turn(String),
    #[error("judge refused to grade: {reason}")]
    Refusal { reason: String },
}

#[async_trait(?Send)]
impl RubricJudge for LlmRubricJudge {
    async fn judge(
        &self,
        trace: &TraceSnapshot,
        request: RubricJudgeRequest,
    ) -> Result<RubricJudgeVerdict, RubricJudgeError> {
        let max_output_tokens = if request.judge_max_output_tokens == 0 {
            self.options.max_output_tokens
        } else {
            request.judge_max_output_tokens
        };
        let mut generation = GenerationParams::default();
        generation.max_output_tokens = Some(max_output_tokens);
        if let Some(temperature) = self.options.temperature {
            let temperature = Temperature::new(temperature)
                .map_err(|_| RubricJudgeError::InvalidTemperature(temperature))?;
            generation.temperature = Some(temperature);
        }

        let input = render_judge_model_input(trace, &request);
        let output_schema = rubric_judge_output_schema(&request)?;
        let _permit = self.concurrency.acquire().await;
        let result = self
            .llm
            .structured_turn::<Value>(input)
            .output_schema("rubric_judge_verdict", output_schema)
            .generation_config(generation)
            .budget(RequestBudget::unlimited())
            .collect()
            .await
            .map_err(|error| RubricJudgeError::Turn(error.to_string()))?;
        let raw_verdict = match result.semantic {
            StructuredTurnOutcome::Structured(value) => value,
            StructuredTurnOutcome::Refusal(reason) => {
                return Err(RubricJudgeError::Refusal { reason });
            }
        };
        let mut verdict = parse_dynamic_verdict(raw_verdict, &request.criteria)?;
        normalize_verdict(&mut verdict);
        Ok(verdict)
    }
}

fn rubric_judge_output_schema(request: &RubricJudgeRequest) -> Result<Value, RubricJudgeError> {
    let mut criteria_properties = Map::new();
    let mut required_criteria = Vec::with_capacity(request.criteria.len());
    for criterion in &request.criteria {
        if criterion.name.is_empty() {
            return Err(RubricJudgeError::InvalidSchema(
                "criterion name must not be empty".to_string(),
            ));
        }
        if criteria_properties
            .insert(criterion.name.clone(), criterion_output_schema())
            .is_some()
        {
            return Err(RubricJudgeError::InvalidSchema(format!(
                "duplicate criterion name '{}'",
                criterion.name
            )));
        }
        required_criteria.push(criterion.name.clone());
    }

    Ok(json!({
        "type": "object",
        "properties": {
            "passed": { "type": "boolean" },
            "score": { "type": "number", "minimum": 0.0, "maximum": 1.0 },
            "summary": { "type": "string" },
            "criteria": {
                "type": "object",
                "properties": criteria_properties,
                "required": required_criteria,
                "additionalProperties": false
            }
        },
        "required": ["passed", "score", "summary", "criteria"],
        "additionalProperties": false
    }))
}

fn criterion_output_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "passed": { "type": "boolean" },
            "score": { "type": "number", "minimum": 0.0, "maximum": 1.0 },
            "reason": { "type": "string" },
            "evidence": { "type": ["string", "null"] }
        },
        "required": ["passed", "score", "reason", "evidence"],
        "additionalProperties": false
    })
}

fn parse_dynamic_verdict(
    value: Value,
    criteria: &[RubricCriterion],
) -> Result<RubricJudgeVerdict, RubricJudgeError> {
    let object = expect_object(&value, "verdict")?;
    let criteria_value = required_value(object, "criteria", "verdict.criteria")?;
    let criteria_object = expect_object(criteria_value, "verdict.criteria")?;
    for actual_name in criteria_object.keys() {
        if !criteria
            .iter()
            .any(|expected| expected.name.as_str() == actual_name)
        {
            return Err(invalid_verdict(format!(
                "unexpected criterion '{actual_name}'"
            )));
        }
    }

    let mut verdict = RubricJudgeVerdict {
        passed: required_bool(object, "passed", "verdict.passed")?,
        score: required_f64(object, "score", "verdict.score")?,
        summary: required_string(object, "summary", "verdict.summary")?,
        criteria: Vec::with_capacity(criteria.len()),
    };

    for expected in criteria {
        let path = format!("verdict.criteria.{}", expected.name);
        let criterion_value = criteria_object.get(&expected.name).ok_or_else(|| {
            invalid_verdict(format!("missing required criterion '{}'", expected.name))
        })?;
        let criterion_object = expect_object(criterion_value, &path)?;
        verdict.criteria.push(RubricJudgeVerdictCriterion {
            name: expected.name.clone(),
            passed: required_bool(criterion_object, "passed", &format!("{path}.passed"))?,
            score: required_f64(criterion_object, "score", &format!("{path}.score"))?,
            reason: required_string(criterion_object, "reason", &format!("{path}.reason"))?,
            evidence: optional_string(criterion_object, "evidence", &format!("{path}.evidence"))?,
        });
    }

    Ok(verdict)
}

fn expect_object<'a>(
    value: &'a Value,
    path: &str,
) -> Result<&'a Map<String, Value>, RubricJudgeError> {
    value
        .as_object()
        .ok_or_else(|| invalid_verdict(format!("{path} must be an object")))
}

fn required_value<'a>(
    object: &'a Map<String, Value>,
    key: &str,
    path: &str,
) -> Result<&'a Value, RubricJudgeError> {
    object
        .get(key)
        .ok_or_else(|| invalid_verdict(format!("{path} is missing")))
}

fn required_bool(
    object: &Map<String, Value>,
    key: &str,
    path: &str,
) -> Result<bool, RubricJudgeError> {
    required_value(object, key, path)?
        .as_bool()
        .ok_or_else(|| invalid_verdict(format!("{path} must be a boolean")))
}

fn required_f64(
    object: &Map<String, Value>,
    key: &str,
    path: &str,
) -> Result<f64, RubricJudgeError> {
    required_value(object, key, path)?
        .as_f64()
        .ok_or_else(|| invalid_verdict(format!("{path} must be a number")))
}

fn required_string(
    object: &Map<String, Value>,
    key: &str,
    path: &str,
) -> Result<String, RubricJudgeError> {
    required_value(object, key, path)?
        .as_str()
        .map(ToOwned::to_owned)
        .ok_or_else(|| invalid_verdict(format!("{path} must be a string")))
}

fn optional_string(
    object: &Map<String, Value>,
    key: &str,
    path: &str,
) -> Result<Option<String>, RubricJudgeError> {
    let Some(value) = object.get(key) else {
        return Ok(None);
    };
    if value.is_null() {
        return Ok(None);
    }
    value
        .as_str()
        .map(|text| Some(text.to_string()))
        .ok_or_else(|| invalid_verdict(format!("{path} must be a string or null")))
}

fn invalid_verdict(message: impl Into<String>) -> RubricJudgeError {
    RubricJudgeError::Turn(format!(
        "judge returned invalid structured verdict: {}",
        message.into()
    ))
}

fn render_judge_model_input(trace: &TraceSnapshot, request: &RubricJudgeRequest) -> ModelInput {
    ModelInput::new()
        .system(
            "You are an eval judge for a capability-based agent runtime. \
Apply the rubric strictly to the selected evidence sections. Grade only observable behavior. \
Return structured output only. Scores are floats from 0.0 to 1.0. \
Return criteria as an object keyed exactly by the provided criterion names.",
        )
        .user(render_judge_input(trace, request))
}

pub fn render_judge_input(trace: &TraceSnapshot, request: &RubricJudgeRequest) -> String {
    let context = request.context.as_deref().unwrap_or("(none)");
    let criteria = if request.criteria.is_empty() {
        "(none; use the rubric as a single holistic criterion)".to_string()
    } else {
        request
            .criteria
            .iter()
            .map(|criterion| {
                format!(
                    "- {} (weight {}, pass >= {:.2}): {}",
                    criterion.name,
                    criterion.weight,
                    criterion.pass_score,
                    normalize_text_block(&criterion.description.content)
                )
            })
            .collect::<Vec<_>>()
            .join("\n")
    };
    let evidence = render_selected_judge_inputs(trace, request);

    format!(
        "Prompt:\n{}\n\nAdditional context:\n{}\n\nRubric:\n{}\n\nOverall pass score: {:.2}\n\nCriteria:\n{}\n\nSelected judge inputs:\n{}\n",
        request.prompt, context, request.rubric, request.pass_score, criteria, evidence
    )
}

fn render_selected_judge_inputs(trace: &TraceSnapshot, request: &RubricJudgeRequest) -> String {
    if request.judge_inputs.is_empty() {
        return "(none)".to_string();
    }
    request
        .judge_inputs
        .iter()
        .map(|input| render_judge_input_section(*input, trace, request))
        .collect::<Vec<_>>()
        .join("\n\n")
}

fn render_judge_input_section(
    input: RubricJudgeInput,
    trace: &TraceSnapshot,
    request: &RubricJudgeRequest,
) -> String {
    match input {
        RubricJudgeInput::Output => {
            section("Primary artifact output", render_artifact_output(request))
        }
        RubricJudgeInput::Utterance => section(
            "Recorded utterances JSON",
            render_observation_paths(
                &request.artifact,
                &[
                    ("last_state.utterances", &["last_state", "utterances"]),
                    ("agent.utterances", &["agent", "utterances"]),
                ],
            ),
        ),
        RubricJudgeInput::Failure => section(
            "Artifact failure",
            request
                .artifact
                .failure
                .clone()
                .unwrap_or_else(|| "(none)".to_string()),
        ),
        RubricJudgeInput::Trace => section("Trace summary", render_trace_summary(trace)),
        RubricJudgeInput::Memory => section(
            "Memory JSON",
            render_observation_paths(
                &request.artifact,
                &[
                    ("memory_diff", &["memory_diff"]),
                    ("last_state.memory", &["last_state", "memory"]),
                    ("agent.memory_metadata", &["agent", "memory_metadata"]),
                ],
            ),
        ),
        RubricJudgeInput::MemoryDiff => section(
            "Memory diff JSON",
            render_observation_paths(&request.artifact, &[("memory_diff", &["memory_diff"])]),
        ),
        RubricJudgeInput::MemoryMetadata => section(
            "Memory metadata JSON",
            render_observation_paths(
                &request.artifact,
                &[("agent.memory_metadata", &["agent", "memory_metadata"])],
            ),
        ),
        RubricJudgeInput::PolicyDiff => section(
            "Policy diff JSON",
            render_observation_paths(&request.artifact, &[("policy_diff", &["policy_diff"])]),
        ),
        RubricJudgeInput::Memos => section(
            "Memos JSON",
            render_observation_paths(
                &request.artifact,
                &[
                    (
                        "last_state.blackboard.memo_logs",
                        &["last_state", "blackboard", "memo_logs"],
                    ),
                    ("agent.memo_logs", &["agent", "memo_logs"]),
                ],
            ),
        ),
        RubricJudgeInput::MemoContents => {
            section("Memo contents", render_memo_contents(&request.artifact))
        }
        RubricJudgeInput::Cognition => section(
            "Cognition entries",
            render_cognition_entries(&request.artifact),
        ),
        RubricJudgeInput::CognitionEntries => section(
            "Cognition entries",
            render_cognition_entries(&request.artifact),
        ),
        RubricJudgeInput::ToolCalls => section("Trace tool calls", render_tool_calls(trace)),
        RubricJudgeInput::ToolResults => section("Trace tool results", render_tool_results(trace)),
    }
}

fn section(title: &str, body: String) -> String {
    format!("{title}:\n{body}")
}

fn render_artifact_output(request: &RubricJudgeRequest) -> String {
    if request.artifact.output.is_empty() {
        "(empty)".to_string()
    } else {
        request.artifact.output.clone()
    }
}

fn render_observation_paths(artifact: &CaseArtifact, paths: &[(&str, &[&str])]) -> String {
    render_named_json_values(paths.iter().filter_map(|(label, path)| {
        observation_path(artifact, path).map(|value| ((*label).to_string(), value.clone()))
    }))
}

fn render_named_json_values(
    values: impl IntoIterator<Item = (String, serde_json::Value)>,
) -> String {
    let rendered = values
        .into_iter()
        .map(|(label, value)| {
            let reduced = cap_json_value(&value);
            let rendered = pretty_json_value(&reduced);
            format!(
                "{label}:\n{}",
                truncate_text(&rendered, MAX_RENDERED_SECTION_CHARS)
            )
        })
        .collect::<Vec<_>>();
    if rendered.is_empty() {
        "(not present)".to_string()
    } else {
        rendered.join("\n\n")
    }
}

fn render_memo_contents(artifact: &CaseArtifact) -> String {
    let mut lines = Vec::new();
    for path in [
        &["last_state", "blackboard", "memo_logs"][..],
        &["agent", "memo_logs"][..],
    ] {
        if let Some(value) = observation_path(artifact, path) {
            collect_string_field_values(value, "content", &mut lines);
        }
    }
    render_text_lines(lines)
}

fn render_cognition_entries(artifact: &CaseArtifact) -> String {
    let mut lines = Vec::new();
    for path in [
        &["last_state", "blackboard", "cognition_logs"][..],
        &["agent", "cognition_logs"][..],
    ] {
        if let Some(value) = observation_path(artifact, path) {
            collect_string_field_values(value, "text", &mut lines);
        }
    }
    render_text_lines(lines)
}

fn collect_string_field_values(value: &serde_json::Value, field: &str, lines: &mut Vec<String>) {
    match value {
        serde_json::Value::Array(items) => {
            for item in items {
                collect_string_field_values(item, field, lines);
            }
        }
        serde_json::Value::Object(map) => {
            if let Some(text) = map.get(field).and_then(serde_json::Value::as_str) {
                lines.push(text.to_string());
            }
            for (key, item) in map {
                if key != "source" {
                    collect_string_field_values(item, field, lines);
                }
            }
        }
        _ => {}
    }
}

fn render_text_lines(lines: Vec<String>) -> String {
    let rendered = lines
        .into_iter()
        .enumerate()
        .map(|(index, line)| {
            format!(
                "{}. {}",
                index + 1,
                truncate_text(line.trim(), MAX_JSON_STRING_CHARS)
            )
        })
        .collect::<Vec<_>>();
    if rendered.is_empty() {
        "(not present)".to_string()
    } else {
        truncate_text(&rendered.join("\n"), MAX_RENDERED_SECTION_CHARS)
    }
}

fn observation_path<'a>(
    artifact: &'a CaseArtifact,
    path: &[&str],
) -> Option<&'a serde_json::Value> {
    let (first, rest) = path.split_first()?;
    let mut current = artifact.observations.get(*first)?;
    for part in rest {
        current = current.get(*part)?;
    }
    Some(current)
}

fn pretty_json_value(value: &serde_json::Value) -> String {
    serde_json::to_string_pretty(value)
        .unwrap_or_else(|error| format!("{{\"serialization_error\":\"{error}\"}}"))
}

fn cap_json_value(value: &serde_json::Value) -> serde_json::Value {
    match value {
        serde_json::Value::Array(items) => {
            let mut capped = items
                .iter()
                .take(MAX_JSON_ARRAY_ITEMS)
                .map(cap_json_value)
                .collect::<Vec<_>>();
            if items.len() > MAX_JSON_ARRAY_ITEMS {
                capped.push(serde_json::json!({
                    "_truncated": format!("omitted {} array items", items.len() - MAX_JSON_ARRAY_ITEMS),
                }));
            }
            serde_json::Value::Array(capped)
        }
        serde_json::Value::Object(map) => {
            let mut capped = serde_json::Map::new();
            for (key, item) in map.iter().take(MAX_JSON_OBJECT_ENTRIES) {
                capped.insert(key.clone(), cap_json_value(item));
            }
            if map.len() > MAX_JSON_OBJECT_ENTRIES {
                capped.insert(
                    "_truncated".to_string(),
                    serde_json::Value::String(format!(
                        "omitted {} object fields",
                        map.len() - MAX_JSON_OBJECT_ENTRIES
                    )),
                );
            }
            serde_json::Value::Object(capped)
        }
        serde_json::Value::String(text) => {
            serde_json::Value::String(truncate_text(text, MAX_JSON_STRING_CHARS))
        }
        other => other.clone(),
    }
}

fn render_tool_calls(trace: &TraceSnapshot) -> String {
    let mut lines = Vec::new();
    for event in &trace.root_events {
        push_tool_call_line(event, &mut lines);
    }
    for root in &trace.roots {
        collect_tool_call_lines(root, &mut lines);
    }
    if lines.is_empty() {
        "(none)".to_string()
    } else {
        truncate_text(&lines.join("\n"), MAX_RENDERED_SECTION_CHARS)
    }
}

fn collect_tool_call_lines(span: &lutum_eval::SpanNode, lines: &mut Vec<String>) {
    for event in &span.events {
        push_tool_call_line(event, lines);
    }
    for child in &span.children {
        collect_tool_call_lines(child, lines);
    }
}

fn push_tool_call_line(event: &lutum_eval::EventRecord, lines: &mut Vec<String>) {
    if event.message.as_deref() != Some("tool_call") {
        return;
    }
    let name = event
        .field("tool_name")
        .and_then(field_value_str)
        .unwrap_or("(unknown)");
    let args = event
        .field("args_json")
        .and_then(field_value_str)
        .unwrap_or("{}");
    lines.push(format!(
        "- {name} args={}",
        truncate_text(args.trim(), MAX_TRACE_TEXT_CHARS)
    ));
}

fn render_tool_results(trace: &TraceSnapshot) -> String {
    let mut lines = Vec::new();
    for event in &trace.root_events {
        push_tool_result_lines(event, &mut lines);
    }
    for root in &trace.roots {
        collect_tool_result_lines(root, &mut lines);
    }
    if lines.is_empty() {
        "(none)".to_string()
    } else {
        truncate_text(&lines.join("\n"), MAX_RENDERED_SECTION_CHARS)
    }
}

fn collect_tool_result_lines(span: &lutum_eval::SpanNode, lines: &mut Vec<String>) {
    for event in &span.events {
        push_tool_result_lines(event, lines);
    }
    for child in &span.children {
        collect_tool_result_lines(child, lines);
    }
}

fn push_tool_result_lines(event: &lutum_eval::EventRecord, lines: &mut Vec<String>) {
    if event.message.as_deref() != Some("llm_input_transcript") {
        return;
    }
    let Some(transcript) = event.field("transcript").and_then(field_value_str) else {
        return;
    };
    for result in extract_tool_results(transcript) {
        let line = format!(
            "- {} output={}",
            result.name,
            truncate_text(result.body.trim(), MAX_TRACE_TEXT_CHARS)
        );
        if !lines.contains(&line) {
            lines.push(line);
        }
    }
}

fn render_trace_summary(trace: &TraceSnapshot) -> String {
    let mut lines = Vec::new();
    let mut omitted_spans = 0_usize;
    let mut omitted_events = 0_usize;
    lines.push(format!(
        "Reduced trace summary; raw input transcripts and streaming chunks are omitted. roots={} root_events={}",
        trace.roots.len(),
        trace.root_events.len()
    ));
    for event in &trace.root_events {
        push_reduced_event(
            "root",
            event,
            &mut lines,
            &mut omitted_events,
            MAX_TRACE_EVENTS,
        );
    }
    for root in &trace.roots {
        render_span(root, 0, &mut lines, &mut omitted_spans, &mut omitted_events);
    }
    if omitted_spans > 0 {
        lines.push(format!("- omitted {omitted_spans} low-signal spans"));
    }
    if omitted_events > 0 {
        lines.push(format!(
            "- omitted {omitted_events} low-signal trace events"
        ));
    }
    if lines.len() == 1 && omitted_spans == 0 && omitted_events == 0 {
        "(empty trace)".to_string()
    } else {
        lines.join("\n")
    }
}

fn render_span(
    span: &lutum_eval::SpanNode,
    depth: usize,
    lines: &mut Vec<String>,
    omitted_spans: &mut usize,
    omitted_events: &mut usize,
) {
    let indent = "  ".repeat(depth);
    if should_render_span(span) && count_span_lines(lines) < MAX_TRACE_SPANS {
        lines.push(format!(
            "{indent}- span {} target={} fields={}",
            span.name,
            span.target,
            render_reduced_fields(&span.fields)
        ));
    } else {
        *omitted_spans += 1;
    }
    for event in &span.events {
        push_reduced_event(
            &format!("span {}", span.name),
            event,
            lines,
            omitted_events,
            MAX_TRACE_EVENTS,
        );
    }
    for child in &span.children {
        render_span(child, depth + 1, lines, omitted_spans, omitted_events);
    }
}

fn should_render_span(span: &lutum_eval::SpanNode) -> bool {
    if span.name == "lutum_hook" {
        return span
            .field("name")
            .and_then(field_value_str)
            .is_some_and(|name| name != "on_stream_event");
    }
    !is_noise_target(&span.target)
}

fn push_reduced_event(
    scope: &str,
    event: &lutum_eval::EventRecord,
    lines: &mut Vec<String>,
    omitted_events: &mut usize,
    max_events: usize,
) {
    let Some(message) = event.message.as_deref() else {
        *omitted_events += 1;
        return;
    };
    if lines.iter().filter(|line| line.contains("event:")).count() >= max_events {
        *omitted_events += 1;
        return;
    }
    match message {
        "llm_output" => {
            if let Some(output) = event.field("output").and_then(field_value_str) {
                lines.push(format!(
                    "- {scope} event: llm_output output={}",
                    truncate_text(output.trim(), MAX_TRACE_TEXT_CHARS)
                ));
            } else {
                lines.push(format!("- {scope} event: llm_output"));
            }
        }
        "llm_input_transcript" => {
            if let Some(transcript) = event.field("transcript").and_then(field_value_str) {
                let tool_results = extract_tool_results(transcript);
                if tool_results.is_empty() {
                    *omitted_events += 1;
                } else {
                    for result in tool_results {
                        lines.push(format!(
                            "- {scope} event: tool_result name={} output={}",
                            result.name,
                            truncate_text(result.body.trim(), MAX_TRACE_TEXT_CHARS)
                        ));
                    }
                }
            } else {
                *omitted_events += 1;
            }
        }
        _ if is_error_event(event) || !is_noise_target(&event.target) => {
            lines.push(format!(
                "- {scope} event: target={} level={} message={} fields={}",
                event.target,
                event.level,
                truncate_text(message, MAX_TRACE_TEXT_CHARS),
                render_reduced_fields(&event.fields)
            ));
        }
        _ => {
            *omitted_events += 1;
        }
    }
}

fn is_error_event(event: &lutum_eval::EventRecord) -> bool {
    event.level == "WARN"
        || event.level == "ERROR"
        || event.message.as_deref().is_some_and(|message| {
            let message = message.to_ascii_lowercase();
            message.contains("error") || message.contains("failed") || message.contains("refused")
        })
}

fn is_noise_target(target: &str) -> bool {
    target.starts_with("hyper")
        || target.starts_with("h2::")
        || target.starts_with("lutum::hooks")
        || target.starts_with("lutum_openai::")
        || target.starts_with("rustls")
        || target.starts_with("tokio")
        || target.starts_with("libsql::")
}

fn count_span_lines(lines: &[String]) -> usize {
    lines
        .iter()
        .filter(|line| line.trim_start().starts_with("- span "))
        .count()
}

struct ToolResultBlock {
    name: String,
    body: String,
}

fn extract_tool_results(transcript: &str) -> Vec<ToolResultBlock> {
    let mut results = Vec::new();
    let mut current_name: Option<String> = None;
    let mut current_body = Vec::new();
    for line in transcript.lines() {
        if let Some(name) = parse_tool_result_header(line) {
            flush_tool_result(&mut results, &mut current_name, &mut current_body);
            current_name = Some(name);
            continue;
        }
        if current_name.is_some() && line.starts_with('[') && line.ends_with(']') {
            flush_tool_result(&mut results, &mut current_name, &mut current_body);
            continue;
        }
        if current_name.is_some() {
            current_body.push(line.to_string());
        }
    }
    flush_tool_result(&mut results, &mut current_name, &mut current_body);
    results
}

fn parse_tool_result_header(line: &str) -> Option<String> {
    let prefix = "[tool_result name=";
    let rest = line.strip_prefix(prefix)?;
    let name = rest.strip_suffix(']')?;
    Some(name.to_string())
}

fn flush_tool_result(
    results: &mut Vec<ToolResultBlock>,
    current_name: &mut Option<String>,
    current_body: &mut Vec<String>,
) {
    let Some(name) = current_name.take() else {
        return;
    };
    let body = current_body.join("\n");
    current_body.clear();
    results.push(ToolResultBlock { name, body });
}

fn render_reduced_fields(fields: &[(String, FieldValue)]) -> String {
    if fields.is_empty() {
        return "{}".to_string();
    }
    let fields = fields
        .iter()
        .filter(|(name, _)| name != "transcript")
        .map(|(name, value)| format!("{name}={}", render_field_value(value)))
        .collect::<Vec<_>>();
    if fields.is_empty() {
        return "{omitted raw transcript}".to_string();
    }
    format!("{{{}}}", fields.join(", "))
}

fn render_field_value(value: &FieldValue) -> String {
    match value {
        FieldValue::Bool(value) => value.to_string(),
        FieldValue::I64(value) => value.to_string(),
        FieldValue::U64(value) => value.to_string(),
        FieldValue::I128(value) => value.to_string(),
        FieldValue::U128(value) => value.to_string(),
        FieldValue::F64(value) => value.to_string(),
        FieldValue::Str(value) => format!("{:?}", truncate_text(value, MAX_TRACE_TEXT_CHARS)),
    }
}

fn field_value_str(value: &FieldValue) -> Option<&str> {
    match value {
        FieldValue::Str(value) => Some(value),
        _ => None,
    }
}

fn truncate_text(text: &str, max_chars: usize) -> String {
    let mut end_byte = text.len();
    let mut count = 0_usize;
    for (index, _) in text.char_indices() {
        if count == max_chars {
            end_byte = index;
            break;
        }
        count += 1;
    }
    if count <= max_chars && end_byte == text.len() {
        return text.to_string();
    }
    let omitted = text.chars().count().saturating_sub(max_chars);
    format!("{} [truncated {omitted} chars]", &text[..end_byte])
}

fn normalize_verdict(verdict: &mut RubricJudgeVerdict) {
    verdict.score = verdict.score.clamp(0.0, 1.0);
    for criterion in &mut verdict.criteria {
        criterion.score = criterion.score.clamp(0.0, 1.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use eure::value::{Language, Text};

    fn criterion(name: &str) -> RubricCriterion {
        RubricCriterion {
            name: name.to_string(),
            description: Text::new("criterion description", Language::Plaintext),
            weight: 1,
            pass_score: 0.8,
        }
    }

    fn empty_request(judge_inputs: Vec<RubricJudgeInput>) -> RubricJudgeRequest {
        RubricJudgeRequest {
            prompt: "Prompt".to_string(),
            context: None,
            rubric: "Rubric".to_string(),
            criteria: Vec::new(),
            pass_score: 0.8,
            judge_inputs,
            judge_max_output_tokens: 1200,
            artifact: CaseArtifact::new("artifact output"),
        }
    }

    #[test]
    fn dynamic_rubric_schema_requires_exact_criteria_keys() {
        let request = RubricJudgeRequest {
            criteria: vec![
                criterion("grounded-in-attention"),
                criterion("simplified-first-person-model"),
            ],
            ..empty_request(vec![RubricJudgeInput::Output])
        };

        let schema = rubric_judge_output_schema(&request).unwrap();

        assert_eq!(
            schema.pointer("/properties/criteria/required").unwrap(),
            &serde_json::json!(["grounded-in-attention", "simplified-first-person-model"])
        );
        assert!(
            schema
                .pointer("/properties/criteria/properties/grounded-in-attention")
                .is_some()
        );
        assert!(
            schema
                .pointer("/properties/criteria/properties/simplified-first-person-model")
                .is_some()
        );
    }

    #[test]
    fn dynamic_verdict_uses_criteria_keys_as_names() {
        let criteria = vec![
            criterion("grounded-in-attention"),
            criterion("simplified-first-person-model"),
        ];
        let raw = serde_json::json!({
            "passed": true,
            "score": 1.0,
            "summary": "passes",
            "criteria": {
                "grounded-in-attention": {
                    "passed": true,
                    "score": 1.0,
                    "reason": "grounded",
                    "evidence": "Koro growling"
                },
                "simplified-first-person-model": {
                    "passed": true,
                    "score": 0.9,
                    "reason": "first person",
                    "evidence": null
                }
            }
        });

        let verdict = parse_dynamic_verdict(raw, &criteria).unwrap();

        assert_eq!(verdict.criteria.len(), 2);
        assert_eq!(verdict.criteria[0].name, "grounded-in-attention");
        assert_eq!(
            verdict.criteria[0].evidence.as_deref(),
            Some("Koro growling")
        );
        assert_eq!(verdict.criteria[1].name, "simplified-first-person-model");
        assert_eq!(verdict.criteria[1].evidence, None);
    }

    fn event(message: &str, fields: Vec<(String, FieldValue)>) -> lutum_eval::EventRecord {
        lutum_eval::EventRecord {
            target: "lutum".to_string(),
            level: "INFO".to_string(),
            message: Some(message.to_string()),
            fields,
        }
    }

    fn span(events: Vec<lutum_eval::EventRecord>) -> lutum_eval::SpanNode {
        lutum_eval::SpanNode {
            name: "model_turn".to_string(),
            target: "lutum".to_string(),
            level: "INFO".to_string(),
            fields: Vec::new(),
            events,
            children: Vec::new(),
        }
    }

    fn empty_trace() -> TraceSnapshot {
        TraceSnapshot {
            roots: Vec::new(),
            root_events: Vec::new(),
        }
    }

    #[test]
    fn trace_summary_keeps_tool_evidence_without_raw_input_transcript() {
        let trace = TraceSnapshot {
            roots: vec![span(vec![
                event(
                    "llm_input_transcript",
                    vec![(
                        "transcript".to_string(),
                        FieldValue::Str(
                            "[System]\nSECRET SYSTEM PROMPT\n\n[tool_result name=search_memory]\n{\"memories\":[{\"content\":\"Koro rule\"}]}\n\n[User]\nQuestion"
                                .to_string(),
                        ),
                    )],
                ),
                event(
                    "llm_output",
                    vec![(
                        "output".to_string(),
                        FieldValue::Str(
                            "<tool_call name=search_memory>{\"query\":\"Koro food\"}</tool_call>"
                                .to_string(),
                        ),
                    )],
                ),
            ])],
            root_events: Vec::new(),
        };
        let request = empty_request(vec![RubricJudgeInput::Trace]);

        let rendered = render_judge_input(&trace, &request);

        assert!(rendered.contains("<tool_call name=search_memory>"));
        assert!(rendered.contains("tool_result name=search_memory"));
        assert!(rendered.contains("Koro rule"));
        assert!(!rendered.contains("SECRET SYSTEM PROMPT"));
    }

    #[test]
    fn memory_judge_input_includes_diff_and_caps_large_arrays() {
        let entries = (0..30)
            .map(|index| {
                serde_json::json!({
                    "index": format!("memory-{index}"),
                    "kind": "Reflection",
                    "content": "summary",
                })
            })
            .collect::<Vec<_>>();
        let request = RubricJudgeRequest {
            artifact: CaseArtifact::new("artifact output").with_observation(
                "memory_diff",
                serde_json::json!({
                    "entries": entries,
                    "links": [],
                }),
            ),
            ..empty_request(vec![RubricJudgeInput::Memory])
        };

        let rendered = render_judge_input(&empty_trace(), &request);

        assert!(rendered.contains("memory_diff"));
        assert!(rendered.contains("memory-0"));
        assert!(rendered.contains("omitted 14 array items"));
    }

    #[test]
    fn cognition_judge_input_omits_source_metadata() {
        let request = RubricJudgeRequest {
            artifact: CaseArtifact::new("artifact output").with_observation(
                "agent",
                serde_json::json!({
                    "cognition_logs": [
                        {
                            "entries": [
                                {
                                    "at": "2026-05-25T00:00:00Z",
                                    "text": "Pibi asks how to approach Koro."
                                }
                            ],
                            "source": {
                                "module": "cognition-gate",
                                "replica": 0
                            }
                        }
                    ]
                }),
            ),
            ..empty_request(vec![RubricJudgeInput::Cognition])
        };

        let rendered = render_judge_input(&empty_trace(), &request);

        assert!(rendered.contains("Pibi asks how to approach Koro."));
        assert!(!rendered.contains("cognition-gate"));
        assert!(!rendered.contains("replica"));
    }

    #[test]
    fn tool_calls_judge_input_uses_structured_tool_events_only() {
        let trace = TraceSnapshot {
            roots: vec![span(vec![
                event(
                    "tool_call",
                    vec![
                        (
                            "tool_name".to_string(),
                            FieldValue::Str("search_memory".to_string()),
                        ),
                        (
                            "args_json".to_string(),
                            FieldValue::Str("{\"query\":\"Koro food\"}".to_string()),
                        ),
                    ],
                ),
                lutum_eval::EventRecord {
                    target: "lutum_openai::sse".to_string(),
                    level: "DEBUG".to_string(),
                    message: Some("openai sse payload".to_string()),
                    fields: vec![(
                        "payload".to_string(),
                        FieldValue::Str("stream chunk noise".to_string()),
                    )],
                },
            ])],
            root_events: Vec::new(),
        };
        let request = empty_request(vec![RubricJudgeInput::ToolCalls, RubricJudgeInput::Trace]);

        let rendered = render_judge_input(&trace, &request);

        assert!(rendered.contains("- search_memory args={\"query\":\"Koro food\"}"));
        assert!(rendered.contains("omitted"));
        assert!(!rendered.contains("stream chunk noise"));
    }
}
