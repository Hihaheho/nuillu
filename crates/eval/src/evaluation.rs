use std::collections::{BTreeMap, HashMap};
use std::convert::Infallible;
use std::time::Duration;

use lutum_eval::{EventRecord, FieldValue, Objective, PureEval, Score, TraceSnapshot};
use lutum_eval_runner::{mean_pass_at_k, mean_pass_hat_k};
use nuillu_module::RuntimeEvent;
use nuillu_types::ModuleInstanceId;
use serde::Serialize;
use serde_json::Value;

use crate::measure::MeasurementValue;
use crate::{
    artifact::CaseArtifact,
    cases::{ArtifactTextField, Assertion, EvalCase},
    judge::{RubricJudge, RubricJudgeRequest, RubricJudgeVerdict},
};

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct CaseTiming {
    pub elapsed_ms: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct MultiTrialTiming {
    pub min_ms: u64,
    pub max_ms: u64,
    pub mean_ms: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct SuiteTiming {
    pub elapsed_ms: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ModuleActivationRecord {
    pub sequence: u64,
    pub module: String,
    pub replica: u8,
    pub started_offset_ms: u64,
    pub duration_ms: u64,
    pub batch_type: Option<String>,
    pub succeeded: bool,
}

pub fn duration_millis_u64(duration: Duration) -> u64 {
    duration.as_millis().try_into().unwrap_or(u64::MAX)
}

pub fn build_activation_timeline(
    timed_events: &[(u64, RuntimeEvent)],
) -> Vec<ModuleActivationRecord> {
    let mut pending: HashMap<ModuleInstanceId, (u64, u64, String)> = HashMap::new();
    let mut records = Vec::new();

    for (offset_ms, event) in timed_events {
        match event {
            RuntimeEvent::ModuleBatchReady {
                sequence,
                owner,
                batch_type,
                ..
            } => {
                pending.insert(owner.clone(), (*sequence, *offset_ms, batch_type.clone()));
            }
            RuntimeEvent::ModuleActivationCompleted {
                sequence,
                owner,
                duration,
                succeeded,
                ..
            } => {
                let (started_sequence, started_offset_ms, batch_type) = pending
                    .remove(owner)
                    .unwrap_or((*sequence, *offset_ms, String::new()));
                records.push(ModuleActivationRecord {
                    sequence: started_sequence,
                    module: owner.module.as_str().to_string(),
                    replica: owner.replica.get(),
                    started_offset_ms,
                    duration_ms: duration_millis_u64(*duration),
                    batch_type: (!batch_type.is_empty()).then_some(batch_type),
                    succeeded: *succeeded,
                });
            }
            _ => {}
        }
    }

    records
}

pub fn aggregate_trial_timing(trials: &[CaseTrialSummary]) -> MultiTrialTiming {
    let elapsed_ms = trials
        .iter()
        .map(|trial| trial.timing.elapsed_ms)
        .collect::<Vec<_>>();
    let min_ms = elapsed_ms.iter().copied().min().unwrap_or(0);
    let max_ms = elapsed_ms.iter().copied().max().unwrap_or(0);
    let mean_ms = if elapsed_ms.is_empty() {
        0
    } else {
        elapsed_ms.iter().sum::<u64>() / elapsed_ms.len() as u64
    };
    MultiTrialTiming {
        min_ms,
        max_ms,
        mean_ms,
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct AssertionOutcome {
    pub name: String,
    pub kind: String,
    pub passed: bool,
    pub errored: bool,
    pub must_pass: bool,
    pub weight: i64,
    pub diagnostic: Option<String>,
    pub rubric: Option<RubricJudgeVerdict>,
}

#[derive(Debug, Clone, Serialize)]
pub struct CaseReport {
    pub runtime_failure: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub llm_log_directory: Option<String>,
    pub assertions: Vec<AssertionOutcome>,
    pub measurements: BTreeMap<String, MeasurementValue>,
    pub invalid: bool,
    pub must_pass_ok: bool,
    pub weighted_points_earned: u64,
    pub weighted_points_total: u64,
    pub score: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct CaseSummary {
    pub path: String,
    pub runtime_config: String,
    pub id: String,
    pub description: Option<String>,
    pub passed: bool,
    pub invalid: bool,
    pub score: f64,
    pub report: CaseReport,
    pub timing: CaseTiming,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub trial_timing: Option<MultiTrialTiming>,
    pub activations: Vec<ModuleActivationRecord>,
    pub measurement_statistics: BTreeMap<String, MeasurementStatistics>,
    pub trial_count: usize,
    pub passed_trials: usize,
    pub failed_trials: usize,
    pub invalid_trials: usize,
    pub trials: Vec<CaseTrialSummary>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct MeasurementStatistics {
    pub samples: usize,
    pub min: f64,
    pub max: f64,
    pub mean: f64,
    pub standard_deviation: f64,
    pub p50: f64,
    pub p95: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct CaseTrialSummary {
    pub trial: usize,
    pub output_dir: String,
    pub path: String,
    pub runtime_config: String,
    pub id: String,
    pub description: Option<String>,
    pub passed: bool,
    pub invalid: bool,
    pub score: f64,
    pub report: CaseReport,
    pub timing: CaseTiming,
}

#[derive(Debug, Clone, Serialize)]
pub struct SuiteRunReport {
    pub run_id: String,
    pub cases_root: String,
    pub output_dir: String,
    pub case_patterns: Vec<String>,
    pub runtime_config_override: Option<String>,
    pub failed_only: bool,
    pub failed_from: Option<String>,
    pub fail_fast: bool,
    pub model_concurrency: BTreeMap<String, Option<usize>>,
    pub trials: usize,
    pub case_concurrency: usize,
    pub planned_case_count: usize,
    pub models: SuiteModelNames,
}

#[derive(Debug, Clone, Serialize)]
pub struct SuiteModelNames {
    pub judge: String,
    pub cheap: String,
    pub default: String,
    pub premium: String,
    pub image: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct SuiteReport {
    pub run: SuiteRunReport,
    pub case_count: usize,
    pub passed_cases: usize,
    pub failed_cases: usize,
    pub invalid_cases: usize,
    pub mean_score: f64,
    pub metrics: SuiteMetrics,
    pub timing: SuiteTiming,
    pub cases: Vec<CaseSummary>,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct SuiteMetrics {
    pub pass_at: Vec<KMetricReport>,
    pub pass_hat: Vec<KMetricReport>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct KMetricReport {
    pub k: usize,
    pub value: f64,
}

impl SuiteMetrics {
    pub fn from_case_counts(cases: &[CaseSummary]) -> Self {
        let Some(max_k) = cases.iter().map(|case| case.trial_count).min() else {
            return Self::default();
        };
        if max_k == 0 {
            return Self::default();
        }

        let counts = cases
            .iter()
            .map(|case| (case.trial_count as u64, case.passed_trials as u64))
            .collect::<Vec<_>>();
        Self {
            pass_at: (1..=max_k)
                .map(|k| KMetricReport {
                    k,
                    value: mean_pass_at_k(&counts, k as u64),
                })
                .collect(),
            pass_hat: (1..=max_k)
                .map(|k| KMetricReport {
                    k,
                    value: mean_pass_hat_k(&counts, k as u64),
                })
                .collect(),
        }
    }
}

impl CaseReport {
    pub fn passed(&self) -> bool {
        self.runtime_failure.is_none() && !self.invalid && self.must_pass_ok
    }

    pub fn recompute(&mut self) {
        let (weighted_points_earned, weighted_points_total) = weighted_points(&self.assertions);
        self.weighted_points_earned = weighted_points_earned;
        self.weighted_points_total = weighted_points_total;
        self.invalid = self.assertions.iter().any(|outcome| outcome.errored);
        self.must_pass_ok = self
            .assertions
            .iter()
            .filter(|outcome| outcome.must_pass)
            .all(outcome_satisfies_requirement);
        self.score = score_report(self);
    }
}

pub struct CaseEval {
    assertions: Vec<Assertion>,
}

impl CaseEval {
    pub fn new(case: &EvalCase) -> Self {
        Self {
            assertions: case.assertions().to_vec(),
        }
    }
}

impl PureEval for CaseEval {
    type Artifact = CaseArtifact;
    type Report = CaseReport;
    type Error = Infallible;

    fn evaluate(
        &self,
        trace: &TraceSnapshot,
        artifact: &Self::Artifact,
    ) -> Result<Self::Report, Self::Error> {
        let mut assertions = Vec::with_capacity(self.assertions.len());
        for check in &self.assertions {
            if matches!(check, Assertion::Rubric { .. }) {
                continue;
            }
            assertions.push(evaluate_deterministic_check(check, trace, artifact));
        }

        let mut report = CaseReport {
            runtime_failure: artifact.failure.clone(),
            llm_log_directory: artifact
                .observations
                .get("llm_log_directory")
                .and_then(|value| value.as_str())
                .map(ToOwned::to_owned),
            assertions,
            measurements: artifact
                .observations
                .get("measurements")
                .cloned()
                .and_then(|value| serde_json::from_value(value).ok())
                .unwrap_or_default(),
            invalid: false,
            must_pass_ok: false,
            weighted_points_earned: 0,
            weighted_points_total: 0,
            score: 0.0,
        };
        report.recompute();
        Ok(report)
    }
}

pub struct CaseObjective;

impl Objective<CaseReport> for CaseObjective {
    type Error = Infallible;

    fn score(&self, report: &CaseReport) -> Result<Score, Self::Error> {
        Ok(Score::new_clamped(score_report(report) as f32))
    }
}

pub async fn evaluate_case(
    case: &EvalCase,
    trace: &TraceSnapshot,
    artifact: &CaseArtifact,
    judge: Option<&dyn RubricJudge>,
) -> CaseReport {
    evaluate_case_with_overrides(case, trace, artifact, judge, &BTreeMap::new()).await
}

pub async fn evaluate_case_with_overrides(
    case: &EvalCase,
    trace: &TraceSnapshot,
    artifact: &CaseArtifact,
    judge: Option<&dyn RubricJudge>,
    assertion_overrides: &BTreeMap<String, AssertionOutcome>,
) -> CaseReport {
    let mut report = CaseEval::new(case)
        .evaluate(trace, artifact)
        .unwrap_or_else(|never| match never {});

    for (index, check) in case.assertions().iter().enumerate() {
        let check_name = check.display_name();
        if let Some(outcome) = assertion_overrides.get(&check_name) {
            if let Some(existing) = report
                .assertions
                .iter_mut()
                .find(|existing| existing.name == check_name)
            {
                *existing = outcome.clone();
            } else {
                let insert_at = index.min(report.assertions.len());
                report.assertions.insert(insert_at, outcome.clone());
            }
            continue;
        }
        if !matches!(check, Assertion::Rubric { .. }) {
            continue;
        }

        let outcome = evaluate_assertion(case, trace, artifact, judge, check).await;
        let insert_at = index.min(report.assertions.len());
        report.assertions.insert(insert_at, outcome);
    }

    report.recompute();
    report.score = CaseObjective
        .score(&report)
        .unwrap_or_else(|never| match never {})
        .value() as f64;
    report
}

pub async fn evaluate_assertion(
    case: &EvalCase,
    trace: &TraceSnapshot,
    artifact: &CaseArtifact,
    judge: Option<&dyn RubricJudge>,
    check: &Assertion,
) -> AssertionOutcome {
    let Assertion::Rubric {
        rubric,
        pass_score,
        judge_inputs,
        criteria,
        ..
    } = check
    else {
        return evaluate_deterministic_check(check, trace, artifact);
    };

    let Some(judge) = judge else {
        return build_error_outcome(
            check,
            "rubric check requires a RubricJudge implementation".to_string(),
        );
    };
    let request = RubricJudgeRequest {
        prompt: normalize_text_block(&case.prompt_for_judge()),
        context: case
            .context_for_judge()
            .map(|text| normalize_text_block(&text)),
        rubric: normalize_text_block(&rubric.content),
        criteria: criteria.clone(),
        pass_score: *pass_score,
        judge_inputs: judge_inputs.clone(),
        judge_max_output_tokens: case.scoring().judge_max_output_tokens,
        artifact: artifact.clone(),
    };
    match judge.judge(trace, request).await {
        Ok(verdict) => build_rubric_outcome(check, *pass_score, verdict),
        Err(error) => build_error_outcome(check, error.to_string()),
    }
}

fn evaluate_deterministic_check(
    check: &Assertion,
    trace: &TraceSnapshot,
    artifact: &CaseArtifact,
) -> AssertionOutcome {
    match check {
        Assertion::ArtifactTextContains {
            field, contains, ..
        } => {
            let text = artifact_text(artifact, field.unwrap_or(ArtifactTextField::Output));
            build_outcome(
                check,
                text.contains(contains),
                (!text.contains(contains)).then(|| {
                    format!(
                        "expected {field_name} to contain {contains:?}",
                        field_name = field_label(field.unwrap_or(ArtifactTextField::Output))
                    )
                }),
            )
        }
        Assertion::ArtifactTextExact { field, exact, .. } => {
            let expected = normalize_text_block(&exact.content);
            let text = normalize_text_block(artifact_text(
                artifact,
                field.unwrap_or(ArtifactTextField::Output),
            ));
            build_outcome(
                check,
                text == expected,
                (text != expected).then(|| {
                    format!(
                        "expected {field_name} to equal {expected:?}, got {text:?}",
                        field_name = field_label(field.unwrap_or(ArtifactTextField::Output))
                    )
                }),
            )
        }
        Assertion::JsonPointerEquals {
            pointer, expected, ..
        } => {
            let json = artifact.as_json();
            let actual = pointer_text(&json, pointer);
            build_outcome(
                check,
                actual.as_deref() == Some(expected.as_str()),
                (actual.as_deref() != Some(expected.as_str())).then(|| match actual {
                    Some(actual) => format!(
                        "expected JSON pointer {pointer:?} to equal {expected:?}, got {actual:?}"
                    ),
                    None => format!("JSON pointer {pointer:?} did not match artifact"),
                }),
            )
        }
        Assertion::JsonPointerContains {
            pointer, contains, ..
        } => {
            let json = artifact.as_json();
            let actual = pointer_text(&json, pointer);
            build_outcome(
                check,
                actual
                    .as_deref()
                    .is_some_and(|text| text.contains(contains)),
                (!actual
                    .as_deref()
                    .is_some_and(|text| text.contains(contains)))
                .then(|| match actual {
                    Some(actual) => format!(
                        "expected JSON pointer {pointer:?} to contain {contains:?}, got {actual:?}"
                    ),
                    None => format!("JSON pointer {pointer:?} did not match artifact"),
                }),
            )
        }
        Assertion::JsonPointerNumericInRange {
            pointer, min, max, ..
        } => {
            let json = artifact.as_json();
            let actual = pointer_number(&json, pointer);
            let (passed, diagnostic) = numeric_range_outcome(pointer, actual, *min, *max);
            build_outcome(check, passed, diagnostic)
        }
        Assertion::TraceSpan { span_name, .. } => build_outcome(
            check,
            trace.span_exists(span_name),
            (!trace.span_exists(span_name)).then(|| format!("expected trace span {span_name:?}")),
        ),
        Assertion::TraceEvent {
            message_contains, ..
        } => {
            let passed =
                trace.events_matching(|event| trace_event_contains(event, message_contains));
            build_outcome(
                check,
                !passed.is_empty(),
                passed
                    .is_empty()
                    .then(|| format!("expected trace event containing {message_contains:?}")),
            )
        }
        Assertion::TraceToolCall {
            tool_name,
            args_json_contains,
            ..
        } => {
            let expected_args = match args_json_contains {
                Some(args_json_contains) => {
                    match serde_json::from_str::<Value>(&args_json_contains.content) {
                        Ok(value) => Some(value),
                        Err(error) => {
                            return build_error_outcome(
                                check,
                                format!("invalid args-json-contains JSON: {error}"),
                            );
                        }
                    }
                }
                None => None,
            };
            let calls = trace_tool_calls(trace);
            let passed = calls.iter().any(|call| {
                call.name == *tool_name
                    && expected_args
                        .as_ref()
                        .is_none_or(|expected| json_contains(&call.args, expected))
            });
            build_outcome(
                check,
                passed,
                (!passed).then(|| {
                    if expected_args.is_some() {
                        format!("expected trace tool call {tool_name:?} with matching arguments")
                    } else {
                        format!("expected trace tool call {tool_name:?}")
                    }
                }),
            )
        }
        Assertion::TraceSpansOrdered { names, .. } => {
            let refs = names.iter().map(String::as_str).collect::<Vec<_>>();
            let passed = trace.spans_ordered(&refs);
            build_outcome(
                check,
                passed,
                (!passed).then(|| format!("expected trace spans in order: {}", names.join(", "))),
            )
        }
        Assertion::Rubric { .. } => unreachable!("rubric assertions are evaluated asynchronously"),
    }
}

#[derive(Debug, Clone, PartialEq)]
struct TraceToolCallObserved {
    name: String,
    args: Value,
}

fn trace_tool_calls(trace: &TraceSnapshot) -> Vec<TraceToolCallObserved> {
    trace
        .all_events()
        .filter_map(trace_tool_call_from_event)
        .collect()
}

fn trace_tool_call_from_event(event: &EventRecord) -> Option<TraceToolCallObserved> {
    if event.message.as_deref() != Some("tool_call") {
        return None;
    }
    let name = event
        .field("tool_name")
        .and_then(trace_field_value_str)?
        .to_owned();
    let args = event
        .field("args_json")
        .and_then(trace_field_value_str)
        .and_then(|text| serde_json::from_str::<Value>(text).ok())?;
    Some(TraceToolCallObserved { name, args })
}

fn json_contains(actual: &Value, expected: &Value) -> bool {
    match (actual, expected) {
        (Value::Object(actual), Value::Object(expected)) => expected.iter().all(|(key, value)| {
            actual
                .get(key)
                .is_some_and(|actual_value| json_contains(actual_value, value))
        }),
        (Value::Array(actual), Value::Array(expected)) => expected.iter().all(|expected_value| {
            actual
                .iter()
                .any(|actual_value| json_contains(actual_value, expected_value))
        }),
        _ => actual == expected,
    }
}

fn trace_event_contains(event: &EventRecord, needle: &str) -> bool {
    if event
        .message()
        .is_some_and(|message| message.contains(needle))
    {
        return true;
    }

    match event.message.as_deref() {
        Some("llm_output") => event
            .field("output")
            .and_then(trace_field_value_str)
            .is_some_and(|output| output.contains(needle)),
        Some("llm_input_transcript") => event
            .field("transcript")
            .and_then(trace_field_value_str)
            .is_some_and(|transcript| transcript_contains_tool_evidence(transcript, needle)),
        _ => event.fields.iter().any(|(key, value)| {
            key != "transcript" && matches!(value, FieldValue::Str(text) if text.contains(needle))
        }),
    }
}

fn transcript_contains_tool_evidence(transcript: &str, needle: &str) -> bool {
    if transcript.contains(&format!("<tool_call name={needle}")) {
        return true;
    }

    transcript.lines().any(|line| {
        line.starts_with("[tool_result name=") && line.contains(&format!("name={needle}"))
    })
}

fn trace_field_value_str(value: &FieldValue) -> Option<&str> {
    match value {
        FieldValue::Str(text) => Some(text.as_str()),
        _ => None,
    }
}

fn build_rubric_outcome(
    check: &Assertion,
    pass_score: f64,
    verdict: RubricJudgeVerdict,
) -> AssertionOutcome {
    let score = rubric_verdict_score(check, &verdict);
    let criteria_failures = rubric_criteria_failures(check, &verdict);
    let passed = score >= pass_score && criteria_failures.is_empty();
    let diagnostic = (!passed).then(|| {
        let mut parts = Vec::new();
        if score < pass_score {
            parts.push(format!(
                "judge score {:.3} below threshold {:.3}: {}",
                score, pass_score, verdict.summary
            ));
        }
        if !criteria_failures.is_empty() {
            parts.push(format!("criteria failed: {}", criteria_failures.join(", ")));
        }
        parts.join("; ")
    });
    build_outcome_with_rubric(check, passed, diagnostic, false, Some(verdict))
}

fn rubric_verdict_score(check: &Assertion, verdict: &RubricJudgeVerdict) -> f64 {
    let Assertion::Rubric { criteria, .. } = check else {
        return 0.0;
    };

    rubric_verdict_score_for(criteria, verdict)
}

fn rubric_verdict_score_for(
    criteria: &[crate::cases::RubricCriterion],
    verdict: &RubricJudgeVerdict,
) -> f64 {
    let mut weighted_score = 0.0;
    let mut total_weight = 0.0;

    for expected in criteria {
        let Some(actual) = verdict
            .criteria
            .iter()
            .find(|actual| actual.name == expected.name)
        else {
            continue;
        };
        let weight = expected.weight.max(0) as f64;
        if weight == 0.0 {
            continue;
        }
        weighted_score += actual.score * weight;
        total_weight += weight;
    }

    if total_weight > 0.0 {
        weighted_score / total_weight
    } else if verdict.criteria.is_empty() {
        0.0
    } else {
        verdict
            .criteria
            .iter()
            .map(|criterion| criterion.score)
            .sum::<f64>()
            / verdict.criteria.len() as f64
    }
}

fn rubric_criteria_failures(check: &Assertion, verdict: &RubricJudgeVerdict) -> Vec<String> {
    let Assertion::Rubric { criteria, .. } = check else {
        return Vec::new();
    };

    rubric_criteria_failures_for(criteria, verdict)
}

fn rubric_criteria_failures_for(
    criteria: &[crate::cases::RubricCriterion],
    verdict: &RubricJudgeVerdict,
) -> Vec<String> {
    criteria
        .iter()
        .filter_map(|expected| {
            let Some(actual) = verdict
                .criteria
                .iter()
                .find(|actual| actual.name == expected.name)
            else {
                return Some(format!("{} missing", expected.name));
            };
            (actual.score < expected.pass_score).then(|| {
                format!(
                    "{} score {:.3} < {:.3}",
                    expected.name, actual.score, expected.pass_score
                )
            })
        })
        .collect()
}

fn build_outcome(
    check: &Assertion,
    raw_passed: bool,
    diagnostic: Option<String>,
) -> AssertionOutcome {
    build_outcome_with_rubric(check, raw_passed, diagnostic, false, None)
}

fn build_error_outcome(check: &Assertion, diagnostic: String) -> AssertionOutcome {
    build_outcome_with_rubric(check, false, Some(diagnostic), true, None)
}

fn build_outcome_with_rubric(
    check: &Assertion,
    raw_passed: bool,
    diagnostic: Option<String>,
    errored: bool,
    rubric: Option<RubricJudgeVerdict>,
) -> AssertionOutcome {
    let common = check.common();
    let (passed, diagnostic) = if common.weight < 0 && !errored {
        let passed = !raw_passed;
        let diagnostic = if passed {
            None
        } else {
            diagnostic.or_else(|| Some("forbidden condition matched".to_string()))
        };
        (passed, diagnostic)
    } else {
        (raw_passed, diagnostic)
    };

    AssertionOutcome {
        name: check.display_name(),
        kind: check.kind_name().to_string(),
        passed,
        errored,
        must_pass: common.must_pass,
        weight: common.weight,
        diagnostic,
        rubric,
    }
}

pub(crate) fn artifact_text(artifact: &CaseArtifact, field: ArtifactTextField) -> &str {
    match field {
        ArtifactTextField::Output => &artifact.output,
        ArtifactTextField::Failure => artifact.failure.as_deref().unwrap_or(""),
    }
}

pub(crate) fn field_label(field: ArtifactTextField) -> &'static str {
    match field {
        ArtifactTextField::Output => "output",
        ArtifactTextField::Failure => "failure",
    }
}

pub(crate) fn pointer_text(value: &serde_json::Value, pointer: &str) -> Option<String> {
    value.pointer(pointer).map(json_value_text)
}

fn json_value_text(value: &serde_json::Value) -> String {
    match value {
        serde_json::Value::String(value) => value.clone(),
        other => {
            serde_json::to_string(other).unwrap_or_else(|error| format!("<json error: {error}>"))
        }
    }
}

pub(crate) fn pointer_number(value: &serde_json::Value, pointer: &str) -> Option<f64> {
    value.pointer(pointer).and_then(|v| v.as_f64())
}

pub(crate) fn numeric_range_outcome(
    pointer: &str,
    actual: Option<f64>,
    min: Option<f64>,
    max: Option<f64>,
) -> (bool, Option<String>) {
    let Some(actual) = actual else {
        return (
            false,
            Some(format!(
                "JSON pointer {pointer:?} did not resolve to a number"
            )),
        );
    };
    let above_min = min.is_none_or(|m| actual >= m);
    let below_max = max.is_none_or(|m| actual <= m);
    if above_min && below_max {
        return (true, None);
    }
    let range_label = match (min, max) {
        (Some(min), Some(max)) => format!("[{min}, {max}]"),
        (Some(min), None) => format!(">= {min}"),
        (None, Some(max)) => format!("<= {max}"),
        (None, None) => "any".to_string(),
    };
    (
        false,
        Some(format!(
            "expected JSON pointer {pointer:?} to be in range {range_label}, got {actual}"
        )),
    )
}

fn outcome_satisfies_requirement(outcome: &AssertionOutcome) -> bool {
    outcome.passed && !outcome.errored
}

fn weighted_points(assertions: &[AssertionOutcome]) -> (u64, u64) {
    let mut earned = 0u64;
    let mut total = 0u64;

    for outcome in assertions {
        if outcome.errored || outcome.must_pass || outcome.weight == 0 {
            continue;
        }
        let magnitude = outcome.weight.unsigned_abs();
        total += magnitude;
        if outcome.passed {
            earned += magnitude;
        }
    }

    (earned, total)
}

fn score_report(report: &CaseReport) -> f64 {
    if report.runtime_failure.is_some() || !report.must_pass_ok {
        return 0.0;
    }
    if report.weighted_points_total == 0 {
        return f64::from(!report.invalid);
    }
    report.weighted_points_earned as f64 / report.weighted_points_total as f64
}

pub fn normalize_text_block(input: &str) -> String {
    let trimmed = input.trim_matches('\n');
    if trimmed.is_empty() {
        return String::new();
    }

    let lines = trimmed.lines().collect::<Vec<_>>();
    let indent = lines
        .iter()
        .filter(|line| !line.trim().is_empty())
        .map(|line| line.chars().take_while(|ch| ch.is_whitespace()).count())
        .min()
        .unwrap_or(0);

    lines
        .into_iter()
        .map(|line| {
            if line.trim().is_empty() {
                String::new()
            } else {
                line.chars().skip(indent).collect::<String>()
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
        .trim()
        .to_string()
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use lutum_eval::{EventRecord, FieldValue, SpanNode, TraceSnapshot};
    use nuillu_types::{ModuleActivationId, ModuleId, ModuleInstanceId, ReplicaIndex};

    use super::{
        CaseReport, CaseTiming, CaseTrialSummary, ModuleActivationRecord, aggregate_trial_timing,
        build_activation_timeline, evaluate_deterministic_check, json_contains,
        trace_event_contains,
    };
    use crate::artifact::CaseArtifact;
    use crate::cases::{Assertion, AssertionCommon};

    fn owner(module: &str) -> ModuleInstanceId {
        ModuleInstanceId::new(
            ModuleId::new(module).expect("valid module id"),
            ReplicaIndex::ZERO,
        )
    }

    fn common(name: &str) -> AssertionCommon {
        AssertionCommon {
            name: Some(name.to_string()),
            must_pass: true,
            weight: 1,
        }
    }

    fn trial(elapsed_ms: u64) -> CaseTrialSummary {
        CaseTrialSummary {
            trial: 1,
            output_dir: "out".to_string(),
            path: "case.eure".to_string(),
            runtime_config: "config.eure".to_string(),
            id: "case".to_string(),
            description: None,
            passed: true,
            invalid: false,
            score: 1.0,
            report: CaseReport {
                runtime_failure: None,
                llm_log_directory: None,
                assertions: Vec::new(),
                measurements: std::collections::BTreeMap::new(),
                invalid: false,
                must_pass_ok: true,
                weighted_points_earned: 0,
                weighted_points_total: 0,
                score: 1.0,
            },
            timing: CaseTiming { elapsed_ms },
        }
    }

    fn event(message: &str, fields: Vec<(String, FieldValue)>) -> EventRecord {
        EventRecord {
            target: "lutum".to_string(),
            level: "DEBUG".to_string(),
            message: Some(message.to_string()),
            fields,
        }
    }

    fn tool_call_event(tool_name: &str, args_json: &str, tool_call_id: &str) -> EventRecord {
        event(
            "tool_call",
            vec![
                (
                    "tool_name".to_string(),
                    FieldValue::Str(tool_name.to_string()),
                ),
                (
                    "args_json".to_string(),
                    FieldValue::Str(args_json.to_string()),
                ),
                (
                    "tool_call_id".to_string(),
                    FieldValue::Str(tool_call_id.to_string()),
                ),
            ],
        )
    }

    fn trace(events: Vec<EventRecord>) -> TraceSnapshot {
        TraceSnapshot {
            roots: vec![SpanNode {
                name: "llm_turn".to_string(),
                target: "lutum".to_string(),
                level: "INFO".to_string(),
                fields: Vec::new(),
                events,
                children: Vec::new(),
            }],
            root_events: Vec::new(),
        }
    }

    #[test]
    fn build_activation_timeline_pairs_batch_ready_with_completion() {
        let speak = owner("speak");
        let timed_events = vec![
            (
                100,
                nuillu_module::RuntimeEvent::ModuleBatchReady {
                    sequence: 1,
                    activation_id: ModuleActivationId::new(1),
                    activation_attempt: 1,
                    owner: speak.clone(),
                    batch_type: "cognition".to_string(),
                    batch_debug: String::new(),
                },
            ),
            (
                250,
                nuillu_module::RuntimeEvent::ModuleActivationCompleted {
                    sequence: 2,
                    activation_id: ModuleActivationId::new(1),
                    owner: speak.clone(),
                    duration: Duration::from_millis(150),
                    succeeded: true,
                },
            ),
        ];

        assert_eq!(
            build_activation_timeline(&timed_events),
            vec![ModuleActivationRecord {
                sequence: 1,
                module: "speak".to_string(),
                replica: 0,
                started_offset_ms: 100,
                duration_ms: 150,
                batch_type: Some("cognition".to_string()),
                succeeded: true,
            }]
        );
    }

    #[test]
    fn aggregate_trial_timing_computes_min_max_mean() {
        let timing = aggregate_trial_timing(&[trial(100), trial(300), trial(200)]);

        assert_eq!(timing.min_ms, 100);
        assert_eq!(timing.max_ms, 300);
        assert_eq!(timing.mean_ms, 200);
    }

    #[test]
    fn trace_event_check_matches_tool_calls_in_event_fields() {
        let tool_output = event(
            "llm_output",
            vec![(
                "output".to_string(),
                FieldValue::Str(
                    "<tool_call name=fetch_linked_memories>{\"memory_indexes\":[\"koro-approach-primary\"]}</tool_call>"
                        .to_string(),
                ),
            )],
        );
        assert!(trace_event_contains(&tool_output, "fetch_linked_memories"));

        let prompt_only = event(
            "llm_input_transcript",
            vec![(
                "transcript".to_string(),
                FieldValue::Str(
                    "[System]\nUse fetch_linked_memories only after a specific search hit.\n[User]\nQuestion"
                        .to_string(),
                ),
            )],
        );
        assert!(!trace_event_contains(&prompt_only, "fetch_linked_memories"));

        let check = Assertion::TraceEvent {
            common: common("calls-fetch-linked-memories"),
            message_contains: "fetch_linked_memories".to_string(),
        };
        let outcome =
            evaluate_deterministic_check(&check, &trace(vec![tool_output]), &CaseArtifact::new(""));
        assert!(outcome.passed);
    }

    #[test]
    fn json_contains_matches_object_subset_and_unordered_arrays() {
        let actual = serde_json::json!({
            "selected_indexes": [
                "koro-approach-primary",
                "koro-signal-drill-linked",
                "koro-food-fern-wait"
            ],
            "extra": true
        });
        let expected = serde_json::json!({
            "selected_indexes": [
                "koro-food-fern-wait",
                "koro-signal-drill-linked"
            ]
        });

        assert!(json_contains(&actual, &expected));
    }

    #[test]
    fn json_contains_rejects_wrong_scalar_value() {
        let actual = serde_json::json!({
            "selected_indexes": ["koro-signal-drill-linked"]
        });
        let expected = serde_json::json!({
            "selected_indexes": ["wrong-index"]
        });

        assert!(!json_contains(&actual, &expected));
    }

    #[test]
    fn trace_tool_call_check_uses_structured_tool_events_only() {
        let name_only = Assertion::TraceToolCall {
            common: common("calls-fetch-linked-memories"),
            tool_name: "fetch_linked_memories".to_string(),
            args_json_contains: None,
        };

        let prompt_only_trace = trace(vec![event(
            "llm_input_transcript",
            vec![(
                "transcript".to_string(),
                FieldValue::Str("Use fetch_linked_memories after linked retrieval.".to_string()),
            )],
        )]);
        assert!(
            !evaluate_deterministic_check(&name_only, &prompt_only_trace, &CaseArtifact::new(""))
                .passed
        );

        let trace = trace(vec![
            tool_call_event(
                "fetch_linked_memories",
                r#"{"memory_indexes":["koro-approach-primary"]}"#,
                "call-fetch",
            ),
            tool_call_event(
                "write_retrieval_memo",
                r#"{"selected_indexes":["koro-approach-primary","koro-signal-drill-linked"]}"#,
                "call-write",
            ),
        ]);

        assert!(evaluate_deterministic_check(&name_only, &trace, &CaseArtifact::new("")).passed);

        let matching_args = Assertion::TraceToolCall {
            common: common("memos-linked-signal-index"),
            tool_name: "write_retrieval_memo".to_string(),
            args_json_contains: Some(eure::value::Text::plaintext(
                r#"{"selected_indexes":["koro-signal-drill-linked"]}"#,
            )),
        };
        assert!(
            evaluate_deterministic_check(&matching_args, &trace, &CaseArtifact::new("")).passed
        );

        let wrong_tool = Assertion::TraceToolCall {
            common: common("missing-tool"),
            tool_name: "missing_tool".to_string(),
            args_json_contains: None,
        };
        assert!(!evaluate_deterministic_check(&wrong_tool, &trace, &CaseArtifact::new("")).passed);

        let wrong_arg = Assertion::TraceToolCall {
            common: common("wrong-arg"),
            tool_name: "write_retrieval_memo".to_string(),
            args_json_contains: Some(eure::value::Text::plaintext(
                r#"{"selected_indexes":["wrong-index"]}"#,
            )),
        };
        assert!(!evaluate_deterministic_check(&wrong_arg, &trace, &CaseArtifact::new("")).passed);
    }
}
