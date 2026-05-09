//! Data-driven evaluation runner for nuillu.
//!
//! Cases live as `.eure` files, normally under `eval-cases/**/*.eure`.
//! Public API surface is limited to case parsing, scoring primitives, and
//! the runner configuration used by the binary.

pub mod artifact;
pub mod cases;
pub mod evaluation;
pub mod judge;
pub mod model_set;
pub mod runner;
pub mod state_dump;
pub mod trace_json;

pub use artifact::CaseArtifact;
pub use cases::{
    ArtifactTextField, AttentionSeed, CaseFileError, CaseScoring, Check, CheckCommon, EvalCase,
    EvalLimits, EvalModule, FileSeed, FullAgentCase, FullAgentCaseFile, FullAgentInput, MemorySeed,
    MemorySeedRank, ModuleCase, ModuleCaseFile, ModuleChecks, ModuleEvalTarget, ModuleRubric,
    RubricCriterion, RubricJudgeInput, discover_case_files, parse_case_file,
    parse_full_agent_case_file, parse_module_case_file,
};
pub use evaluation::{
    CaseEval, CaseObjective, CaseReport, CaseSummary, CheckOutcome, ModuleChecksReport,
    ModuleRubricOutcome, SuiteReport, evaluate_case, normalize_text_block,
};
pub use judge::{
    JudgeOptions, LlmRubricJudge, RubricJudge, RubricJudgeError, RubricJudgeRequest,
    RubricJudgeVerdict, RubricJudgeVerdictCriterion, render_judge_input,
};
pub use model_set::{
    ModelSet, ModelSetError, ModelSetFile, ModelSetRole, ReasoningEffort, parse_model_set_file,
};
pub use runner::{
    CaseRunOutput, LlmBackendConfig, RunnerConfig, RunnerError, default_run_id,
    install_lutum_trace_subscriber, run_case_detailed, run_suite,
};
pub use state_dump::{
    AgenticDeadlockDump, AllocationModuleDump, AllocationProposalDump, AttentionEntryDump,
    AttentionStreamDump, BlackboardLastStateDump, DumpText, FullAgentLastStateCaseDump,
    FullAgentLastStateDump, MemoDump, MemoLogDump, MemoryEntryDump, MemoryLastStateDump,
    MemoryMetadataDump, ModuleInstanceDump, ReplicaCapDump, StateDumpRenderError, UtteranceDump,
    render_full_agent_last_state_eure,
};
pub use trace_json::{raw_trace_has_error, raw_trace_snapshot_json, trace_snapshot_json};
