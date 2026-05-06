//! Data-driven evaluation primitives for nuillu.
//!
//! Cases live as `.eure` files, normally under `eval-cases/**/*.eure`.
//! The crate deliberately separates case execution from scoring: callers
//! provide a [`CaseDriver`] that runs a case against whichever nuillu boot
//! wiring they want to evaluate, then this crate evaluates deterministic
//! checks and rubric-based LLM-as-judge checks over the returned artifact.

pub mod artifact;
pub mod cases;
pub mod evaluation;
pub mod judge;
pub mod runner;

pub use artifact::CaseArtifact;
pub use cases::{
    ArtifactTextField, CaseFileError, CaseScoring, CaseTarget, Check, CheckCommon, EvalCase,
    EvalCaseFile, MemorySeed, MemorySeedRank, RubricCriterion, discover_case_files,
    parse_case_file,
};
pub use evaluation::{
    CaseEval, CaseObjective, CaseReport, CaseSummary, CheckOutcome, SuiteReport, evaluate_case,
    normalize_text_block,
};
pub use judge::{
    JudgeOptions, LlmRubricJudge, RubricJudge, RubricJudgeError, RubricJudgeRequest,
    RubricJudgeVerdict, RubricJudgeVerdictCriterion, render_judge_input,
};
pub use runner::{
    CaseDriver, CaseRunOutput, RunOptions, RunnerError, run_case, run_case_detailed, run_suite,
};
