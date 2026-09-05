//! Data-driven evaluation runner for nuillu.
//!
//! Cases live as `.eure` files, normally under `eval-cases/**/*.eure`.
//! Public API surface is limited to case parsing, scoring primitives, and
//! the runner configuration used by the binary.

// Eval harness plumbing threads many independent knobs (fixtures, seeds, budgets, reporting
// sinks) through single call sites; bundling them into parameter structs buys nothing here.
#![allow(clippy::too_many_arguments)]

pub mod artifact;
pub mod cases;
pub mod evaluation;
pub mod gui;
pub mod judge;
pub mod measure;
pub mod query;
pub mod runner;
pub mod state_dump;
pub mod timeline;
pub mod trace_json;

pub use artifact::CaseArtifact;
pub use cases::{
    ArtifactTextField, Assertion, AssertionCommon, CaseFileError, CaseScoring, CognitionLogSeed,
    EvalCase, EvalLimits, EvalModule, EvalStep, EventSelectorSpec, Measurement, MemoSeed,
    MemoryLinkSeed, MemorySeed, MemorySeedRank, PolicySeed, PolicySeedRank, RubricCriterion,
    RubricJudgeInput, RuntimeCase, RuntimeCaseFile, Stimulus, WaitFor, discover_case_files,
    parse_case_file, parse_runtime_case_file,
};
pub use evaluation::{
    AssertionOutcome, CaseEval, CaseObjective, CaseReport, CaseSummary, CaseTiming,
    CaseTrialSummary, KMetricReport, MeasurementStatistics, ModuleActivationRecord,
    MultiTrialTiming, SuiteMetrics, SuiteModelNames, SuiteReport, SuiteRunReport, SuiteTiming,
    aggregate_trial_timing, build_activation_timeline, evaluate_assertion, evaluate_case,
    evaluate_case_with_overrides, normalize_text_block,
};
pub use judge::{
    JudgeOptions, LlmRubricJudge, RubricJudge, RubricJudgeError, RubricJudgeRequest,
    RubricJudgeVerdict, RubricJudgeVerdictCriterion, render_judge_input,
};
pub use nuillu_server::model_set::{
    EmbeddingRole, ModelDefinition, ModelSet, ModelSetError, ModelSetFile, ReasoningEffort,
    ResolvedLlmBackends, TierBinding, model_concurrency_from_backends, parse_model_set_file,
    resolve_llm_backends, resolve_token_fields,
};
pub use runner::{
    CaseRunOutput, EmbeddingBackendConfig, LiveOutput, LlmBackendConfig, RunnerConfig, RunnerError,
    RunnerHooks, VisualizerHook, default_run_id, install_lutum_trace_subscriber, run_case_detailed,
    run_suite, run_suite_with_hooks,
};
pub use state_dump::{
    AgenticDeadlockDump, AllocationModuleDump, AllocationProposalDump, BlackboardLastStateDump,
    CognitionEntryDump, CognitionLogDump, DumpText, MemoLogDump, MemoryEntryDump,
    MemoryLastStateDump, MemoryMetadataDump, ModuleInstanceDump, ReplicaCapDump,
    RuntimeLastStateCaseDump, RuntimeLastStateDump, StateDumpRenderError, UtteranceDump,
    render_runtime_last_state_eure,
};
pub use trace_json::{raw_trace_has_error, raw_trace_snapshot_json, trace_snapshot_json};
