use std::collections::HashMap;
use std::convert::Infallible;
use std::time::Duration;

use lutum_eval::{Objective, PureEval, Score, TraceSnapshot};
use lutum_eval_runner::{mean_pass_at_k, mean_pass_hat_k};
use nuillu_module::RuntimeEvent;
use nuillu_types::ModuleInstanceId;
use serde::Serialize;

use crate::{
    artifact::CaseArtifact,
    cases::{ArtifactTextField, Check, EvalCase, EvalModule, ModuleChecks, ModuleRubric},
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
    pub modules_checks: Vec<ModuleChecksReport>,
    pub invalid: bool,
    pub must_pass_ok: bool,
    pub weighted_points_earned: u64,
    pub weighted_points_total: u64,
    pub score: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct ModuleChecksReport {
    pub module: String,
    pub rubrics: Vec<ModuleRubricOutcome>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ModuleRubricOutcome {
    pub name: String,
    pub passed: bool,
    pub errored: bool,
    pub pass_score: f64,
    pub diagnostic: Option<String>,
    pub rubric: Option<RubricJudgeVerdict>,
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
    pub timing: CaseTiming,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub trial_timing: Option<MultiTrialTiming>,
    pub activations: Vec<ModuleActivationRecord>,
    pub trial_count: usize,
    pub passed_trials: usize,
    pub failed_trials: usize,
    pub invalid_trials: usize,
    pub trials: Vec<CaseTrialSummary>,
}

#[derive(Debug, Clone, Serialize)]
pub struct CaseTrialSummary {
    pub trial: usize,
    pub output_dir: String,
    pub path: String,
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
    pub failed_only: bool,
    pub failed_from: Option<String>,
    pub fail_fast: bool,
    pub max_concurrent_llm_calls: Option<usize>,
    pub trials: usize,
    pub planned_case_count: usize,
    pub models: SuiteModelNames,
    pub module_filters: Vec<String>,
    pub disabled_modules: Vec<String>,
    pub exclude_full_agent: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct SuiteModelNames {
    pub judge: String,
    pub cheap: String,
    pub default: String,
    pub premium: String,
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
            modules_checks: Vec::new(),
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
                judge_inputs,
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
                        judge_inputs: judge_inputs.clone(),
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

    if !case.modules_checks().is_empty() {
        report.modules_checks = evaluate_modules_checks(case, trace, artifact, judge).await;
    }

    report.recompute();
    report.score = CaseObjective
        .score(&report)
        .unwrap_or_else(|never| match never {})
        .value() as f64;
    report
}

async fn evaluate_modules_checks(
    case: &EvalCase,
    trace: &TraceSnapshot,
    artifact: &CaseArtifact,
    judge: Option<&dyn RubricJudge>,
) -> Vec<ModuleChecksReport> {
    let mut reports = Vec::new();
    for checks in case.modules_checks() {
        let scoped_artifact = module_scoped_artifact(checks.module, artifact);
        let mut rubrics = Vec::new();
        for rubric in &checks.rubrics {
            let outcome =
                evaluate_module_rubric(case, checks, rubric, trace, scoped_artifact.clone(), judge)
                    .await;
            rubrics.push(outcome);
        }
        reports.push(ModuleChecksReport {
            module: checks.module.as_str().to_string(),
            rubrics,
        });
    }
    reports
}

async fn evaluate_module_rubric(
    case: &EvalCase,
    checks: &ModuleChecks,
    rubric: &ModuleRubric,
    trace: &TraceSnapshot,
    artifact: CaseArtifact,
    judge: Option<&dyn RubricJudge>,
) -> ModuleRubricOutcome {
    let name = rubric.display_name();
    let Some(judge) = judge else {
        return build_module_rubric_error_outcome(
            &name,
            rubric.pass_score,
            "module rubric requires a RubricJudge implementation".to_string(),
        );
    };

    let request = RubricJudgeRequest {
        prompt: normalize_text_block(&case.prompt_for_judge()),
        context: Some(format!(
            "Module-scoped full-agent check for '{}'. Judge only the selected evidence for this module.",
            checks.module.as_str()
        )),
        rubric: normalize_text_block(&rubric.rubric.content),
        criteria: rubric.criteria.clone(),
        pass_score: rubric.pass_score,
        judge_inputs: rubric.judge_inputs.clone(),
        judge_max_output_tokens: case.scoring().judge_max_output_tokens,
        artifact,
    };

    match judge.judge(trace, request).await {
        Ok(verdict) => build_module_rubric_outcome(&name, rubric, verdict),
        Err(error) => {
            build_module_rubric_error_outcome(&name, rubric.pass_score, error.to_string())
        }
    }
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
        Check::JsonPointerNumericInRange {
            pointer, min, max, ..
        } => {
            let json = artifact.as_json();
            let actual = pointer_number(&json, pointer);
            let (passed, diagnostic) = numeric_range_outcome(pointer, actual, *min, *max);
            build_outcome(check, passed, diagnostic)
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

fn build_module_rubric_outcome(
    name: &str,
    rubric: &ModuleRubric,
    verdict: RubricJudgeVerdict,
) -> ModuleRubricOutcome {
    let criteria_failures = rubric_criteria_failures_for(&rubric.criteria, &verdict);
    let passed =
        verdict.passed && verdict.score >= rubric.pass_score && criteria_failures.is_empty();
    let diagnostic = (!passed).then(|| {
        let mut parts = Vec::new();
        if !verdict.passed || verdict.score < rubric.pass_score {
            parts.push(format!(
                "judge score {:.3} below threshold {:.3} or failed verdict: {}",
                verdict.score, rubric.pass_score, verdict.summary
            ));
        }
        if !criteria_failures.is_empty() {
            parts.push(format!("criteria failed: {}", criteria_failures.join(", ")));
        }
        parts.join("; ")
    });
    ModuleRubricOutcome {
        name: name.to_string(),
        passed,
        errored: false,
        pass_score: rubric.pass_score,
        diagnostic,
        rubric: Some(verdict),
    }
}

fn build_module_rubric_error_outcome(
    name: &str,
    pass_score: f64,
    diagnostic: String,
) -> ModuleRubricOutcome {
    ModuleRubricOutcome {
        name: name.to_string(),
        passed: false,
        errored: true,
        pass_score,
        diagnostic: Some(diagnostic),
        rubric: None,
    }
}

fn rubric_criteria_failures(check: &Check, verdict: &RubricJudgeVerdict) -> Vec<String> {
    let Check::Rubric { criteria, .. } = check else {
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

fn module_scoped_artifact(module: EvalModule, artifact: &CaseArtifact) -> CaseArtifact {
    let mut scoped = CaseArtifact::new(render_module_output(module, artifact));
    scoped.failure = artifact.failure.clone();
    scoped.observations.insert(
        "module".to_string(),
        serde_json::Value::String(module.as_str().to_string()),
    );
    for key in ["typed_memo_logs", "memory_diff"] {
        if let Some(value) = artifact.observations.get(key).cloned() {
            scoped.observations.insert(key.to_string(), value);
        }
    }
    if let Some(agent) = scoped_agent_observation(module, artifact) {
        scoped.observations.insert("agent".to_string(), agent);
    }
    if let Some(last_state) = scoped_last_state_observation(module, artifact) {
        scoped
            .observations
            .insert("last_state".to_string(), last_state);
    }
    scoped
}

fn render_module_output(module: EvalModule, artifact: &CaseArtifact) -> String {
    let logs = module_memo_log_values(module, artifact);
    if !logs.is_empty() {
        return logs
            .iter()
            .map(render_memo_log_value)
            .collect::<Vec<_>>()
            .join("\n\n");
    }

    format!("(no memo logs for module {})", module.as_str())
}

fn module_memo_log_values(module: EvalModule, artifact: &CaseArtifact) -> Vec<serde_json::Value> {
    if let Some(logs) = observation_path_value(
        &artifact.observations,
        &["agent", "memo_logs", module.as_str()],
    )
    .and_then(serde_json::Value::as_array)
    {
        return logs.clone();
    }

    observation_path_value(
        &artifact.observations,
        &["last_state", "blackboard", "memo_logs"],
    )
    .and_then(serde_json::Value::as_array)
    .map(|logs| {
        logs.iter()
            .filter(|log| value_field_text(log, "module").as_deref() == Some(module.as_str()))
            .cloned()
            .collect()
    })
    .unwrap_or_default()
}

fn render_memo_log_value(log: &serde_json::Value) -> String {
    let replica = value_field_text(log, "replica").unwrap_or_else(|| "?".to_string());
    let index = value_field_text(log, "index").unwrap_or_else(|| "?".to_string());
    let written_at = value_field_text(log, "written_at")
        .or_else(|| value_field_text(log, "written-at"))
        .unwrap_or_else(|| "?".to_string());
    let content = value_field_text(log, "content").unwrap_or_default();
    format!("replica={replica} index={index} written_at={written_at}\n{content}")
}

fn scoped_agent_observation(
    module: EvalModule,
    artifact: &CaseArtifact,
) -> Option<serde_json::Value> {
    let agent = observation_path_value(&artifact.observations, &["agent"])?.as_object()?;
    let mut scoped = serde_json::Map::new();
    scoped.insert(
        "module".to_string(),
        serde_json::Value::String(module.as_str().to_string()),
    );
    insert_if_some(
        &mut scoped,
        "memo_logs",
        filter_object_entry(agent.get("memo_logs"), module.as_str()),
    );
    insert_if_some(
        &mut scoped,
        "cognition_logs",
        filter_array_by_nested_module(agent.get("cognition_logs"), &["source", "module"], module),
    );
    insert_if_some(
        &mut scoped,
        "allocation",
        scoped_agent_allocation(agent.get("allocation"), module),
    );
    insert_if_some(
        &mut scoped,
        "allocation_proposals",
        scoped_agent_allocation_proposals(agent.get("allocation_proposals"), module),
    );
    insert_if_some(
        &mut scoped,
        "replica_caps",
        filter_object_entry(agent.get("replica_caps"), module.as_str()),
    );
    insert_if_some(
        &mut scoped,
        "memory_metadata",
        agent.get("memory_metadata").cloned(),
    );
    if module == EvalModule::Speak {
        insert_if_some(&mut scoped, "utterances", agent.get("utterances").cloned());
    }
    Some(serde_json::Value::Object(scoped))
}

fn scoped_last_state_observation(
    module: EvalModule,
    artifact: &CaseArtifact,
) -> Option<serde_json::Value> {
    let last_state =
        observation_path_value(&artifact.observations, &["last_state"])?.as_object()?;
    let mut scoped = serde_json::Map::new();
    insert_if_some(&mut scoped, "case", last_state.get("case").cloned());
    if let Some(blackboard) = last_state
        .get("blackboard")
        .and_then(serde_json::Value::as_object)
    {
        scoped.insert(
            "blackboard".to_string(),
            scoped_last_state_blackboard(blackboard, module),
        );
    }
    insert_if_some(&mut scoped, "memory", last_state.get("memory").cloned());
    if module == EvalModule::Speak {
        insert_if_some(
            &mut scoped,
            "utterances",
            last_state.get("utterances").cloned(),
        );
    }
    Some(serde_json::Value::Object(scoped))
}

fn scoped_last_state_blackboard(
    blackboard: &serde_json::Map<String, serde_json::Value>,
    module: EvalModule,
) -> serde_json::Value {
    let mut scoped = serde_json::Map::new();
    insert_if_some(
        &mut scoped,
        "memo_logs",
        filter_array_by_field(blackboard.get("memo_logs"), "module", module.as_str()),
    );
    insert_if_some(
        &mut scoped,
        "cognition_logs",
        filter_array_by_nested_module(
            blackboard.get("cognition_logs"),
            &["source", "module"],
            module,
        ),
    );
    insert_if_some(
        &mut scoped,
        "base_allocation",
        scoped_allocation_array(blackboard.get("base_allocation"), module),
    );
    insert_if_some(
        &mut scoped,
        "allocation",
        scoped_allocation_array(blackboard.get("allocation"), module),
    );
    insert_if_some(
        &mut scoped,
        "allocation_proposals",
        scoped_last_state_allocation_proposals(blackboard.get("allocation_proposals"), module),
    );
    insert_if_some(
        &mut scoped,
        "replica_caps",
        scoped_allocation_array(blackboard.get("replica_caps"), module),
    );
    serde_json::Value::Object(scoped)
}

fn insert_if_some(
    object: &mut serde_json::Map<String, serde_json::Value>,
    key: &str,
    value: Option<serde_json::Value>,
) {
    if let Some(value) = value {
        object.insert(key.to_string(), value);
    }
}

fn filter_object_entry(value: Option<&serde_json::Value>, key: &str) -> Option<serde_json::Value> {
    let object = value?.as_object()?;
    let entry = object.get(key)?;
    let mut filtered = serde_json::Map::new();
    filtered.insert(key.to_string(), entry.clone());
    Some(serde_json::Value::Object(filtered))
}

fn filter_array_by_field(
    value: Option<&serde_json::Value>,
    field: &str,
    expected: &str,
) -> Option<serde_json::Value> {
    let filtered = value?
        .as_array()?
        .iter()
        .filter(|entry| value_field_text(entry, field).as_deref() == Some(expected))
        .cloned()
        .collect::<Vec<_>>();
    Some(serde_json::Value::Array(filtered))
}

fn filter_array_by_nested_module(
    value: Option<&serde_json::Value>,
    path: &[&str],
    module: EvalModule,
) -> Option<serde_json::Value> {
    let filtered = value?
        .as_array()?
        .iter()
        .filter(|entry| nested_value_text(entry, path).as_deref() == Some(module.as_str()))
        .cloned()
        .collect::<Vec<_>>();
    Some(serde_json::Value::Array(filtered))
}

fn scoped_agent_allocation(
    value: Option<&serde_json::Value>,
    module: EvalModule,
) -> Option<serde_json::Value> {
    if module == EvalModule::AllocationController {
        return value.cloned();
    }
    filter_object_entry(value, module.as_str())
}

fn scoped_agent_allocation_proposals(
    value: Option<&serde_json::Value>,
    module: EvalModule,
) -> Option<serde_json::Value> {
    let proposals = value?.as_object()?;
    let mut scoped = serde_json::Map::new();
    for (owner, proposal) in proposals {
        if module == EvalModule::AllocationController {
            if owner_matches_module(owner, module) {
                scoped.insert(owner.clone(), proposal.clone());
            }
            continue;
        }
        let Some(filtered) = filter_object_entry(Some(proposal), module.as_str()) else {
            continue;
        };
        scoped.insert(owner.clone(), filtered);
    }
    Some(serde_json::Value::Object(scoped))
}

fn scoped_allocation_array(
    value: Option<&serde_json::Value>,
    module: EvalModule,
) -> Option<serde_json::Value> {
    if module == EvalModule::AllocationController {
        return value.cloned();
    }
    filter_array_by_field(value, "module", module.as_str())
}

fn scoped_last_state_allocation_proposals(
    value: Option<&serde_json::Value>,
    module: EvalModule,
) -> Option<serde_json::Value> {
    let proposals = value?.as_array()?;
    let mut scoped = Vec::new();
    for proposal in proposals {
        if module == EvalModule::AllocationController {
            if nested_value_text(proposal, &["controller", "module"]).as_deref()
                == Some(module.as_str())
            {
                scoped.push(proposal.clone());
            }
            continue;
        }
        let Some(modules) = proposal
            .get("modules")
            .and_then(serde_json::Value::as_array)
        else {
            continue;
        };
        let modules = modules
            .iter()
            .filter(|entry| value_field_text(entry, "module").as_deref() == Some(module.as_str()))
            .cloned()
            .collect::<Vec<_>>();
        if modules.is_empty() {
            continue;
        }
        let mut filtered = serde_json::Map::new();
        if let Some(controller) = proposal.get("controller") {
            filtered.insert("controller".to_string(), controller.clone());
        }
        filtered.insert("modules".to_string(), serde_json::Value::Array(modules));
        scoped.push(serde_json::Value::Object(filtered));
    }
    Some(serde_json::Value::Array(scoped))
}

fn owner_matches_module(owner: &str, module: EvalModule) -> bool {
    owner == module.as_str()
        || owner
            .strip_prefix(module.as_str())
            .is_some_and(|suffix| suffix.starts_with('['))
}

fn observation_path_value<'a>(
    observations: &'a std::collections::BTreeMap<String, serde_json::Value>,
    path: &[&str],
) -> Option<&'a serde_json::Value> {
    let (first, rest) = path.split_first()?;
    let mut current = observations.get(*first)?;
    for part in rest {
        current = current.get(*part)?;
    }
    Some(current)
}

fn nested_value_text(value: &serde_json::Value, path: &[&str]) -> Option<String> {
    let mut current = value;
    for part in path {
        current = current.get(*part)?;
    }
    Some(json_value_text(current))
}

fn value_field_text(value: &serde_json::Value, field: &str) -> Option<String> {
    value.get(field).map(json_value_text)
}

pub(crate) fn json_value_text(value: &serde_json::Value) -> String {
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

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use nuillu_types::{ModuleId, ModuleInstanceId, ReplicaIndex};

    use super::{
        CaseTiming, CaseTrialSummary, ModuleActivationRecord, aggregate_trial_timing,
        build_activation_timeline,
    };
    use crate::evaluation::CaseReport;

    fn test_report(passed: bool, invalid: bool, score: f64) -> CaseReport {
        CaseReport {
            runtime_failure: invalid.then(|| "invalid".to_string()),
            checks: Vec::new(),
            modules_checks: Vec::new(),
            invalid,
            must_pass_ok: passed,
            weighted_points_earned: 0,
            weighted_points_total: 0,
            score,
        }
    }

    fn owner(module: &str) -> ModuleInstanceId {
        ModuleInstanceId::new(
            ModuleId::new(module).expect("valid module id"),
            ReplicaIndex::ZERO,
        )
    }

    #[test]
    fn build_activation_timeline_pairs_batch_ready_with_completion() {
        let speak = owner("speak");
        let timed_events = vec![
            (
                100,
                nuillu_module::RuntimeEvent::ModuleBatchReady {
                    sequence: 1,
                    owner: speak.clone(),
                    batch_type: "cognition".to_string(),
                    batch_debug: String::new(),
                },
            ),
            (
                250,
                nuillu_module::RuntimeEvent::ModuleActivationCompleted {
                    sequence: 2,
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
        let trials = vec![
            CaseTrialSummary {
                trial: 1,
                output_dir: "a".to_string(),
                path: "a.eure".to_string(),
                id: "a".to_string(),
                description: None,
                passed: true,
                invalid: false,
                score: 1.0,
                report: test_report(true, false, 1.0),
                timing: CaseTiming { elapsed_ms: 100 },
            },
            CaseTrialSummary {
                trial: 2,
                output_dir: "b".to_string(),
                path: "b.eure".to_string(),
                id: "b".to_string(),
                description: None,
                passed: true,
                invalid: false,
                score: 1.0,
                report: test_report(true, false, 1.0),
                timing: CaseTiming { elapsed_ms: 300 },
            },
            CaseTrialSummary {
                trial: 3,
                output_dir: "c".to_string(),
                path: "c.eure".to_string(),
                id: "c".to_string(),
                description: None,
                passed: true,
                invalid: false,
                score: 1.0,
                report: test_report(true, false, 1.0),
                timing: CaseTiming { elapsed_ms: 200 },
            },
        ];

        let timing = aggregate_trial_timing(&trials);
        assert_eq!(timing.min_ms, 100);
        assert_eq!(timing.max_ms, 300);
        assert_eq!(timing.mean_ms, 200);
    }
}
