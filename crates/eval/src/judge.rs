use async_trait::async_trait;
use lutum::{Lutum, ModelInput, Temperature};
use lutum_eval::{Eval as _, FieldValue, JudgeEval, JudgeEvalError, TraceSnapshot};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{artifact::CaseArtifact, cases::RubricCriterion, evaluation::normalize_text_block};

#[derive(Debug, Clone)]
pub struct RubricJudgeRequest {
    pub prompt: String,
    pub context: Option<String>,
    pub rubric: String,
    pub criteria: Vec<RubricCriterion>,
    pub pass_score: f64,
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
    pub temperature: f32,
    pub max_output_tokens: u32,
}

impl Default for JudgeOptions {
    fn default() -> Self {
        Self {
            temperature: 0.0,
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
        let temperature = Temperature::new(self.options.temperature)
            .map_err(|_| RubricJudgeError::InvalidTemperature(self.options.temperature))?;
        let eval = JudgeEval::<
            RubricJudgeRequest,
            RubricJudgeVerdict,
            fn(&TraceSnapshot, &RubricJudgeRequest) -> ModelInput,
        >::new(render_judge_model_input)
        .temperature(temperature)
        .max_output_tokens(max_output_tokens);

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
Apply the rubric strictly to the artifact and trace. Grade only observable behavior. \
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
    let artifact = serde_json::to_string_pretty(&request.artifact.as_json())
        .unwrap_or_else(|error| format!("{{\"serialization_error\":\"{error}\"}}"));

    format!(
        "Prompt:\n{}\n\nAdditional context:\n{}\n\nRubric:\n{}\n\nOverall pass score: {:.2}\n\nCriteria:\n{}\n\nArtifact JSON:\n{}\n\nTrace summary:\n{}\n",
        request.prompt,
        context,
        request.rubric,
        request.pass_score,
        criteria,
        artifact,
        render_trace_summary(trace)
    )
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
