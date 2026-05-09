use async_trait::async_trait;
use lutum::{Lutum, ModelInput, Temperature};
use lutum_eval::{Eval as _, FieldValue, JudgeEval, JudgeEvalError, TraceSnapshot};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{
    artifact::CaseArtifact,
    cases::{RubricCriterion, RubricJudgeInput},
    evaluation::normalize_text_block,
};

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
    options: JudgeOptions,
}

impl LlmRubricJudge {
    pub fn new(llm: Lutum) -> Self {
        Self {
            llm,
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
        let mut eval = JudgeEval::<
            RubricJudgeRequest,
            RubricJudgeVerdict,
            fn(&TraceSnapshot, &RubricJudgeRequest) -> ModelInput,
        >::new(render_judge_model_input)
        .max_output_tokens(max_output_tokens);
        if let Some(temperature) = self.options.temperature {
            let temperature = Temperature::new(temperature)
                .map_err(|_| RubricJudgeError::InvalidTemperature(temperature))?;
            eval = eval.temperature(temperature);
        }

        let mut verdict = eval
            .evaluate(&self.llm, trace, &request)
            .await
            .map_err(RubricJudgeError::from)?;
        normalize_verdict(&mut verdict);
        Ok(verdict)
    }
}

impl From<JudgeEvalError> for RubricJudgeError {
    fn from(error: JudgeEvalError) -> Self {
        match error {
            JudgeEvalError::Refusal { reason } => Self::Refusal { reason },
            other => Self::Turn(other.to_string()),
        }
    }
}

fn render_judge_model_input(trace: &TraceSnapshot, request: &RubricJudgeRequest) -> ModelInput {
    ModelInput::new()
        .system(
            "You are an eval judge for a capability-based agent runtime. \
Apply the rubric strictly to the selected evidence sections. Grade only observable behavior. \
Return structured output only. Scores are floats from 0.0 to 1.0.",
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
            "Utterance",
            render_artifact_output(request)
                + "\n\nRecorded utterances JSON:\n"
                + &render_observation_paths(
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
        RubricJudgeInput::Observations => section(
            "Artifact observations JSON",
            pretty_json_value(
                &serde_json::to_value(&request.artifact.observations).unwrap_or_else(
                    |error| serde_json::json!({ "serialization_error": error.to_string() }),
                ),
            ),
        ),
        RubricJudgeInput::Blackboard => section(
            "Blackboard JSON",
            render_blackboard_input(&request.artifact),
        ),
        RubricJudgeInput::Memory => section(
            "Memory JSON",
            render_observation_paths(
                &request.artifact,
                &[
                    ("last_state.memory", &["last_state", "memory"]),
                    ("agent.memory_metadata", &["agent", "memory_metadata"]),
                ],
            ),
        ),
        RubricJudgeInput::Memos => section(
            "Memos JSON",
            render_observation_paths(
                &request.artifact,
                &[
                    (
                        "last_state.blackboard.memos",
                        &["last_state", "blackboard", "memos"],
                    ),
                    (
                        "last_state.blackboard.memo_logs",
                        &["last_state", "blackboard", "memo_logs"],
                    ),
                    ("agent.memos", &["agent", "memos"]),
                    ("agent.memo_logs", &["agent", "memo_logs"]),
                ],
            ),
        ),
        RubricJudgeInput::Attention => section(
            "Attention JSON",
            render_observation_paths(
                &request.artifact,
                &[
                    (
                        "last_state.blackboard.attention_streams",
                        &["last_state", "blackboard", "attention_streams"],
                    ),
                    ("agent.attention_streams", &["agent", "attention_streams"]),
                ],
            ),
        ),
        RubricJudgeInput::Allocation => section(
            "Allocation JSON",
            render_allocation_input(&request.artifact),
        ),
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

fn render_blackboard_input(artifact: &CaseArtifact) -> String {
    let last_state_blackboard = observation_path(artifact, &["last_state", "blackboard"])
        .map(|value| ("last_state.blackboard".to_string(), value.clone()));
    let agent_blackboard =
        agent_blackboard_observation(artifact).map(|value| ("agent.blackboard".to_string(), value));
    render_named_json_values(
        [last_state_blackboard, agent_blackboard]
            .into_iter()
            .flatten(),
    )
}

fn render_allocation_input(artifact: &CaseArtifact) -> String {
    let last_state_allocation =
        observation_path(artifact, &["last_state", "blackboard"]).map(|bb| {
            let mut map = serde_json::Map::new();
            for key in [
                "base_allocation",
                "allocation",
                "allocation_proposals",
                "replica_caps",
            ] {
                if let Some(value) = bb.get(key) {
                    map.insert(key.to_string(), value.clone());
                }
            }
            (
                "last_state.blackboard.allocation".to_string(),
                serde_json::Value::Object(map),
            )
        });
    let agent_allocation = observation_path(artifact, &["agent"]).map(|agent| {
        let mut map = serde_json::Map::new();
        for key in ["allocation", "allocation_proposals", "replica_caps"] {
            if let Some(value) = agent.get(key) {
                map.insert(key.to_string(), value.clone());
            }
        }
        (
            "agent.allocation".to_string(),
            serde_json::Value::Object(map),
        )
    });
    render_named_json_values(
        [last_state_allocation, agent_allocation]
            .into_iter()
            .flatten(),
    )
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
        .map(|(label, value)| format!("{label}:\n{}", pretty_json_value(&value)))
        .collect::<Vec<_>>();
    if rendered.is_empty() {
        "(not present)".to_string()
    } else {
        rendered.join("\n\n")
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

fn agent_blackboard_observation(artifact: &CaseArtifact) -> Option<serde_json::Value> {
    let agent = observation_path(artifact, &["agent"])?.as_object()?;
    let mut map = serde_json::Map::new();
    for key in [
        "memos",
        "memo_logs",
        "attention_streams",
        "allocation",
        "allocation_proposals",
        "replica_caps",
        "memory_metadata",
    ] {
        if let Some(value) = agent.get(key) {
            map.insert(key.to_string(), value.clone());
        }
    }
    Some(serde_json::Value::Object(map))
}

fn pretty_json_value(value: &serde_json::Value) -> String {
    serde_json::to_string_pretty(value)
        .unwrap_or_else(|error| format!("{{\"serialization_error\":\"{error}\"}}"))
}

fn render_trace_summary(trace: &TraceSnapshot) -> String {
    let mut lines = Vec::new();
    for event in &trace.root_events {
        lines.push(format!("- root event: {}", render_event(event)));
    }
    for root in &trace.roots {
        render_span(root, 0, &mut lines);
    }
    if lines.is_empty() {
        "(empty trace)".to_string()
    } else {
        lines.join("\n")
    }
}

fn render_span(span: &lutum_eval::SpanNode, depth: usize, lines: &mut Vec<String>) {
    let indent = "  ".repeat(depth);
    lines.push(format!(
        "{indent}- span {} target={} fields={}",
        span.name,
        span.target,
        render_fields(&span.fields)
    ));
    for event in &span.events {
        lines.push(format!("{indent}  - event: {}", render_event(event)));
    }
    for child in &span.children {
        render_span(child, depth + 1, lines);
    }
}

fn render_event(event: &lutum_eval::EventRecord) -> String {
    format!(
        "target={} message={} fields={}",
        event.target,
        event.message.as_deref().unwrap_or("(none)"),
        render_fields(&event.fields)
    )
}

fn render_fields(fields: &[(String, FieldValue)]) -> String {
    if fields.is_empty() {
        return "{}".to_string();
    }
    let fields = fields
        .iter()
        .map(|(name, value)| format!("{name}={}", render_field_value(value)))
        .collect::<Vec<_>>();
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
        FieldValue::Str(value) => format!("{value:?}"),
    }
}

fn normalize_verdict(verdict: &mut RubricJudgeVerdict) {
    verdict.score = verdict.score.clamp(0.0, 1.0);
    for criterion in &mut verdict.criteria {
        criterion.score = criterion.score.clamp(0.0, 1.0);
    }
}
