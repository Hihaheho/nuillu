use std::path::{Path, PathBuf};

use async_trait::async_trait;
use lutum_eval::{RawTraceSnapshot, TraceSnapshot};
use thiserror::Error;
use tracing::instrument::WithSubscriber as _;
use tracing_subscriber::layer::SubscriberExt as _;

use crate::{
    artifact::CaseArtifact,
    cases::{CaseFileError, EvalCase, discover_case_files, parse_case_file},
    evaluation::{CaseSummary, SuiteReport, evaluate_case, normalize_text_block},
    judge::RubricJudge,
};

pub type DriverError = Box<dyn std::error::Error>;

#[async_trait(?Send)]
pub trait CaseDriver {
    /// Execute one parsed case and return the normalized artifact to score.
    ///
    /// The runner wraps this future in `lutum_trace::capture_raw`. Drivers that
    /// call Lutum should enable raw telemetry on their `Lutum` value when they
    /// want provider request/stream details to appear in `CaseRunOutput::raw_trace`.
    async fn run_case(&mut self, case: &EvalCase) -> Result<CaseArtifact, DriverError>;
}

#[derive(Debug, Clone)]
pub struct RunOptions {
    pub fail_fast: bool,
}

#[derive(Debug, Clone)]
pub struct CaseRunOutput {
    pub case_path: PathBuf,
    pub summary: CaseSummary,
    pub artifact: CaseArtifact,
    pub trace: TraceSnapshot,
    pub raw_trace: RawTraceSnapshot,
}

impl Default for RunOptions {
    fn default() -> Self {
        Self { fail_fast: false }
    }
}

#[derive(Debug, Error)]
pub enum RunnerError {
    #[error(transparent)]
    Case(#[from] CaseFileError),
    #[error("case driver failed for {path}: {message}")]
    Driver { path: PathBuf, message: String },
    #[error("failed to discover eval cases under {path}: {source}")]
    DiscoverCases {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
}

pub async fn run_case(
    case_path: &Path,
    driver: &mut dyn CaseDriver,
    judge: Option<&dyn RubricJudge>,
) -> Result<CaseSummary, RunnerError> {
    run_case_detailed(case_path, driver, judge)
        .await
        .map(|output| output.summary)
}

pub async fn run_case_detailed(
    case_path: &Path,
    driver: &mut dyn CaseDriver,
    judge: Option<&dyn RubricJudge>,
) -> Result<CaseRunOutput, RunnerError> {
    let case = parse_case_file(case_path)?;
    let dispatch =
        tracing::Dispatch::new(tracing_subscriber::registry().with(lutum_trace::layer()));
    let collected = lutum_trace::capture_raw(driver.run_case(&case))
        .with_subscriber(dispatch)
        .await;
    let trace = collected.trace;
    let raw_trace = collected.raw;
    let artifact = collected.output.map_err(|error| RunnerError::Driver {
        path: case_path.to_path_buf(),
        message: error.to_string(),
    })?;
    let report = evaluate_case(&case, &trace, &artifact, judge).await;

    let summary = CaseSummary {
        path: case_path.display().to_string(),
        id: case_id(case_path, &case),
        description: case
            .description
            .as_ref()
            .map(|text| normalize_text_block(&text.content)),
        passed: report.passed(),
        invalid: report.invalid,
        score: report.score,
        report,
    };

    Ok(CaseRunOutput {
        case_path: case_path.to_path_buf(),
        summary,
        artifact,
        trace,
        raw_trace,
    })
}

pub async fn run_suite(
    cases_root: &Path,
    driver: &mut dyn CaseDriver,
    judge: Option<&dyn RubricJudge>,
    options: &RunOptions,
) -> Result<SuiteReport, RunnerError> {
    let case_paths =
        discover_case_files(cases_root).map_err(|source| RunnerError::DiscoverCases {
            path: cases_root.to_path_buf(),
            source,
        })?;
    let mut cases = Vec::with_capacity(case_paths.len());

    for path in case_paths {
        let output = run_case_detailed(&path, driver, judge).await?;
        let summary = output.summary;
        let failed = !summary.passed || summary.invalid;
        cases.push(summary);
        if failed && options.fail_fast {
            break;
        }
    }

    Ok(aggregate_suite(cases))
}

fn aggregate_suite(cases: Vec<CaseSummary>) -> SuiteReport {
    let case_count = cases.len();
    let passed_cases = cases.iter().filter(|case| case.passed).count();
    let invalid_cases = cases.iter().filter(|case| case.invalid).count();
    let failed_cases = case_count.saturating_sub(passed_cases + invalid_cases);
    let mean_score = if cases.is_empty() {
        0.0
    } else {
        cases.iter().map(|case| case.score).sum::<f64>() / cases.len() as f64
    };

    SuiteReport {
        case_count,
        passed_cases,
        failed_cases,
        invalid_cases,
        mean_score,
        cases,
    }
}

fn case_id(path: &Path, case: &EvalCase) -> String {
    case.id.clone().unwrap_or_else(|| {
        path.with_extension("")
            .file_name()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_else(|| "case".to_string())
    })
}
