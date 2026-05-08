use std::convert::Infallible;

use lutum_eval::{Objective, PureEval, Score, TraceSnapshot};
use serde::Serialize;

use crate::{
    artifact::CaseArtifact,
    cases::{ArtifactTextField, Check, EvalCase},
    judge::{RubricJudge, RubricJudgeRequest, RubricJudgeVerdict},
};

#[derive(Debug, Clone, Serialize)]
pub struct CheckOutcome {
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
    pub checks: Vec<CheckOutcome>,
    pub invalid: bool,
    pub must_pass_ok: bool,
    pub weighted_points_earned: u64,
    pub weighted_points_total: u64,
    pub score: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct CaseSummary {
    pub path: String,
    pub id: String,
    pub description: Option<String>,
    pub passed: bool,
    pub invalid: bool,
    pub score: f64,
    pub report: CaseReport,
}

#[derive(Debug, Clone, Serialize)]
pub struct SuiteReport {
    pub case_count: usize,
    pub passed_cases: usize,
    pub failed_cases: usize,
    pub invalid_cases: usize,
    pub mean_score: f64,
    pub cases: Vec<CaseSummary>,
}

impl CaseReport {
    pub fn passed(&self) -> bool {
        self.runtime_failure.is_none() && !self.invalid && self.must_pass_ok
    }

    pub fn recompute(&mut self) {
        let (weighted_points_earned, weighted_points_total) = weighted_points(&self.checks);
        self.weighted_points_earned = weighted_points_earned;
        self.weighted_points_total = weighted_points_total;
        self.invalid = self.checks.iter().any(|outcome| outcome.errored);
        self.must_pass_ok = self
            .checks
            .iter()
            .filter(|outcome| outcome.must_pass)
            .all(outcome_satisfies_requirement);
        self.score = score_report(self);
    }
}

pub struct CaseEval {
    checks: Vec<Check>,
}

impl CaseEval {
    pub fn new(case: &EvalCase) -> Self {
        Self {
            checks: case.checks().to_vec(),
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
        let mut checks = Vec::with_capacity(self.checks.len());
        for check in &self.checks {
            if matches!(check, Check::Rubric { .. }) {
                continue;
            }
            checks.push(evaluate_deterministic_check(check, trace, artifact));
        }

        let mut report = CaseReport {
            runtime_failure: artifact.failure.clone(),
            checks,
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
    let mut report = CaseEval::new(case)
        .evaluate(trace, artifact)
        .unwrap_or_else(|never| match never {});

    for (index, check) in case.checks().iter().enumerate() {
        if !matches!(check, Check::Rubric { .. }) {
            continue;
        }

        let outcome = match check {
            Check::Rubric {
                rubric,
                pass_score,
                criteria,
                ..
            } => match judge {
                Some(judge) => {
                    let request = RubricJudgeRequest {
                        prompt: normalize_text_block(&case.prompt_for_judge()),
                        context: case
                            .context_for_judge()
                            .map(|text| normalize_text_block(&text)),
                        rubric: normalize_text_block(&rubric.content),
                        criteria: criteria.clone(),
                        pass_score: *pass_score,
                        judge_max_output_tokens: case.scoring().judge_max_output_tokens,
                        artifact: artifact.clone(),
                    };
                    match judge.judge(trace, request).await {
                        Ok(verdict) => build_rubric_outcome(check, *pass_score, verdict),
                        Err(error) => build_error_outcome(check, error.to_string()),
                    }
                }
                None => build_error_outcome(
                    check,
                    "rubric check requires a RubricJudge implementation".to_string(),
                ),
            },
            _ => unreachable!("guarded by rubric match"),
        };
        let insert_at = index.min(report.checks.len());
        report.checks.insert(insert_at, outcome);
    }

    report.recompute();
    report.score = CaseObjective
        .score(&report)
        .unwrap_or_else(|never| match never {})
        .value() as f64;
    report
}

fn evaluate_deterministic_check(
    check: &Check,
    trace: &TraceSnapshot,
    artifact: &CaseArtifact,
) -> CheckOutcome {
    match check {
        Check::ArtifactTextContains {
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
        Check::ArtifactTextExact { field, exact, .. } => {
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
        Check::JsonPointerEquals {
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
        Check::JsonPointerContains {
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
        Check::TraceSpan { span_name, .. } => build_outcome(
            check,
            trace.span_exists(span_name),
            (!trace.span_exists(span_name)).then(|| format!("expected trace span {span_name:?}")),
        ),
        Check::TraceEvent {
            message_contains, ..
        } => {
            let passed = trace.events_matching(|event| {
                event
                    .message()
                    .is_some_and(|message| message.contains(message_contains))
            });
            build_outcome(
                check,
                !passed.is_empty(),
                passed.is_empty().then(|| {
                    format!("expected trace event message containing {message_contains:?}")
                }),
            )
        }
        Check::TraceSpansOrdered { names, .. } => {
            let refs = names.iter().map(String::as_str).collect::<Vec<_>>();
            let passed = trace.spans_ordered(&refs);
            build_outcome(
                check,
                passed,
                (!passed).then(|| format!("expected trace spans in order: {}", names.join(", "))),
            )
        }
        Check::Rubric { .. } => unreachable!("rubric checks are evaluated asynchronously"),
    }
}

fn build_rubric_outcome(
    check: &Check,
    pass_score: f64,
    verdict: RubricJudgeVerdict,
) -> CheckOutcome {
    let criteria_failures = rubric_criteria_failures(check, &verdict);
    let passed = verdict.passed && verdict.score >= pass_score && criteria_failures.is_empty();
    let diagnostic = (!passed).then(|| {
        let mut parts = Vec::new();
        if !verdict.passed || verdict.score < pass_score {
            parts.push(format!(
                "judge score {:.3} below threshold {:.3} or failed verdict: {}",
                verdict.score, pass_score, verdict.summary
            ));
        }
        if !criteria_failures.is_empty() {
            parts.push(format!("criteria failed: {}", criteria_failures.join(", ")));
        }
        parts.join("; ")
    });
    build_outcome_with_rubric(check, passed, diagnostic, false, Some(verdict))
}

fn rubric_criteria_failures(check: &Check, verdict: &RubricJudgeVerdict) -> Vec<String> {
    let Check::Rubric { criteria, .. } = check else {
        return Vec::new();
    };

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
            (!actual.passed || actual.score < expected.pass_score).then(|| {
                format!(
                    "{} score {:.3} < {:.3}",
                    expected.name, actual.score, expected.pass_score
                )
            })
        })
        .collect()
}

fn build_outcome(check: &Check, raw_passed: bool, diagnostic: Option<String>) -> CheckOutcome {
    build_outcome_with_rubric(check, raw_passed, diagnostic, false, None)
}

fn build_error_outcome(check: &Check, diagnostic: String) -> CheckOutcome {
    build_outcome_with_rubric(check, false, Some(diagnostic), true, None)
}

fn build_outcome_with_rubric(
    check: &Check,
    raw_passed: bool,
    diagnostic: Option<String>,
    errored: bool,
    rubric: Option<RubricJudgeVerdict>,
) -> CheckOutcome {
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

    CheckOutcome {
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

fn artifact_text(artifact: &CaseArtifact, field: ArtifactTextField) -> &str {
    match field {
        ArtifactTextField::Output => &artifact.output,
        ArtifactTextField::Failure => artifact.failure.as_deref().unwrap_or(""),
    }
}

fn field_label(field: ArtifactTextField) -> &'static str {
    match field {
        ArtifactTextField::Output => "output",
        ArtifactTextField::Failure => "failure",
    }
}

fn pointer_text(value: &serde_json::Value, pointer: &str) -> Option<String> {
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

fn outcome_satisfies_requirement(outcome: &CheckOutcome) -> bool {
    outcome.passed && !outcome.errored
}

fn weighted_points(checks: &[CheckOutcome]) -> (u64, u64) {
    let mut earned = 0u64;
    let mut total = 0u64;

    for outcome in checks {
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
