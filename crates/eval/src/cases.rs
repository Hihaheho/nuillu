use std::{
    collections::BTreeSet,
    fs, io,
    path::{Path, PathBuf},
};

use chrono::{DateTime, Datelike as _, FixedOffset, NaiveDate, TimeZone as _, Utc};
use eure::{FromEure, value::Text};
use nuillu_types::{MemoryRank, ModuleId, builtin};
use thiserror::Error;

fn default_weight() -> i64 {
    1
}

fn default_memory_rank() -> MemorySeedRank {
    MemorySeedRank::ShortTerm
}

fn default_memory_decay_secs() -> i64 {
    86_400
}

fn default_seed_seconds_ago() -> i64 {
    0
}

fn default_memo_replica() -> u8 {
    0
}

fn default_pass_score() -> f64 {
    0.8
}

fn default_judge_max_output_tokens() -> u32 {
    1200
}

fn default_judge_inputs() -> Vec<RubricJudgeInput> {
    vec![
        RubricJudgeInput::Output,
        RubricJudgeInput::Failure,
        RubricJudgeInput::Observations,
        RubricJudgeInput::Trace,
    ]
}

fn default_module_rubric_judge_inputs() -> Vec<RubricJudgeInput> {
    vec![RubricJudgeInput::Output, RubricJudgeInput::Observations]
}

fn default_max_llm_calls() -> Option<u64> {
    Some(10)
}

#[derive(Debug, Clone, FromEure)]
#[eure(crate = ::eure::document, rename_all = "kebab-case")]
pub struct FullAgentCaseFile {
    #[eure(flatten)]
    pub case: FullAgentCase,
}

#[derive(Debug, Clone, FromEure)]
#[eure(crate = ::eure::document, rename_all = "kebab-case")]
pub struct FullAgentCase {
    #[eure(default)]
    pub id: Option<String>,
    #[eure(default)]
    pub description: Option<Text>,
    #[eure(default)]
    pub now: Option<String>,
    #[eure(default)]
    pub modules: Option<Vec<EvalModule>>,
    #[eure(default)]
    pub inputs: Vec<FullAgentInput>,
    #[eure(default)]
    pub steps: Vec<EvalStep>,
    #[eure(default)]
    pub participants: Vec<String>,
    #[eure(default)]
    pub memories: Vec<MemorySeed>,
    #[eure(default)]
    pub memos: Vec<MemoSeed>,
    #[eure(default)]
    pub limits: EvalLimits,
    #[eure(default)]
    pub checks: Vec<Check>,
    #[eure(default)]
    pub modules_checks: Vec<ModuleChecks>,
    #[eure(default)]
    pub scoring: CaseScoring,
}

#[derive(Debug, Clone, FromEure)]
#[eure(
    crate = ::eure::document,
    rename_all = "kebab-case",
    rename_all_fields = "kebab-case"
)]
pub enum FullAgentInput {
    Heard {
        #[eure(default)]
        direction: Option<String>,
        content: Text,
    },
    Seen {
        #[eure(default)]
        direction: Option<String>,
        appearance: Text,
    },
    OneShot {
        modality: String,
        #[eure(default)]
        direction: Option<String>,
        content: Text,
    },
}

#[derive(Debug, Clone, FromEure)]
#[eure(crate = ::eure::document, rename_all = "kebab-case")]
pub struct EvalStep {
    #[eure(default)]
    pub description: Option<Text>,
    pub inputs: Vec<FullAgentInput>,
    #[eure(default)]
    pub wait_for: Option<WaitFor>,
    #[eure(default)]
    pub checks: Vec<Check>,
}

#[derive(Debug, Clone, FromEure)]
#[eure(
    crate = ::eure::document,
    rename_all = "kebab-case",
    rename_all_fields = "kebab-case"
)]
pub enum WaitFor {
    MemoFrom { module: EvalModule, timeout_ms: u64 },
}

#[derive(Debug, Clone, FromEure)]
#[eure(crate = ::eure::document, rename_all = "kebab-case")]
pub struct ModuleCaseFile {
    #[eure(flatten)]
    pub case: ModuleCase,
}

#[derive(Debug, Clone, FromEure)]
#[eure(crate = ::eure::document, rename_all = "kebab-case")]
pub struct ModuleCase {
    #[eure(default)]
    pub id: Option<String>,
    #[eure(default)]
    pub description: Option<Text>,
    #[eure(default)]
    pub now: Option<String>,
    #[eure(default)]
    pub modules: Option<Vec<EvalModule>>,
    pub prompt: Text,
    #[eure(default)]
    pub context: Option<Text>,
    #[eure(default)]
    pub allow_empty_output: bool,
    #[eure(default)]
    pub participants: Vec<String>,
    #[eure(default)]
    pub memories: Vec<MemorySeed>,
    #[eure(default)]
    pub memos: Vec<MemoSeed>,
    #[eure(default)]
    pub cognition_log: Vec<CognitionLogSeed>,
    #[eure(default)]
    pub limits: EvalLimits,
    #[eure(default)]
    pub checks: Vec<Check>,
    #[eure(default)]
    pub scoring: CaseScoring,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, clap::ValueEnum, FromEure)]
#[eure(crate = ::eure::document, rename_all = "kebab-case")]
pub enum EvalModule {
    Sensory,
    CognitionGate,
    AllocationController,
    AttentionSchema,
    SelfModel,
    QueryVector,
    QueryPolicy,
    Memory,
    MemoryCompaction,
    MemoryAssociation,
    MemoryRecombination,
    Vital,
    HomeostaticController,
    Policy,
    ValueEstimator,
    Reward,
    Predict,
    Surprise,
    SpeakGate,
    Speak,
}

pub const DEFAULT_FULL_AGENT_MODULES: &[EvalModule] = &[
    EvalModule::Sensory,
    EvalModule::CognitionGate,
    EvalModule::AllocationController,
    EvalModule::AttentionSchema,
    EvalModule::SelfModel,
    EvalModule::QueryVector,
    EvalModule::QueryPolicy,
    EvalModule::Memory,
    EvalModule::MemoryCompaction,
    EvalModule::MemoryAssociation,
    EvalModule::MemoryRecombination,
    EvalModule::Vital,
    EvalModule::HomeostaticController,
    EvalModule::ValueEstimator,
    EvalModule::Reward,
    EvalModule::Policy,
    EvalModule::Predict,
    EvalModule::Surprise,
    EvalModule::SpeakGate,
    EvalModule::Speak,
];

impl EvalModule {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Sensory => "sensory",
            Self::CognitionGate => "cognition-gate",
            Self::AllocationController => "allocation-controller",
            Self::AttentionSchema => "attention-schema",
            Self::SelfModel => "self-model",
            Self::QueryVector => "query-vector",
            Self::QueryPolicy => "query-policy",
            Self::Memory => "memory",
            Self::MemoryCompaction => "memory-compaction",
            Self::MemoryAssociation => "memory-association",
            Self::MemoryRecombination => "memory-recombination",
            Self::Vital => "vital",
            Self::HomeostaticController => "homeostatic-controller",
            Self::Policy => "policy",
            Self::ValueEstimator => "value-estimator",
            Self::Reward => "reward",
            Self::Predict => "predict",
            Self::Surprise => "surprise",
            Self::SpeakGate => "speak-gate",
            Self::Speak => "speak",
        }
    }

    pub fn module_id(self) -> ModuleId {
        match self {
            Self::Sensory => builtin::sensory(),
            Self::CognitionGate => builtin::cognition_gate(),
            Self::AllocationController => builtin::allocation_controller(),
            Self::AttentionSchema => builtin::attention_schema(),
            Self::SelfModel => builtin::self_model(),
            Self::QueryVector => builtin::query_vector(),
            Self::QueryPolicy => builtin::query_policy(),
            Self::Memory => builtin::memory(),
            Self::MemoryCompaction => builtin::memory_compaction(),
            Self::MemoryAssociation => builtin::memory_association(),
            Self::MemoryRecombination => builtin::memory_recombination(),
            Self::Vital => builtin::vital(),
            Self::HomeostaticController => builtin::homeostatic_controller(),
            Self::Policy => builtin::policy(),
            Self::ValueEstimator => builtin::value_estimator(),
            Self::Reward => builtin::reward(),
            Self::Predict => builtin::predict(),
            Self::Surprise => builtin::surprise(),
            Self::SpeakGate => builtin::speak_gate(),
            Self::Speak => builtin::speak(),
        }
    }

    pub fn is_action_module(self) -> bool {
        matches!(self, Self::Speak)
    }
}

impl FullAgentCase {
    pub fn effective_modules(&self) -> Vec<EvalModule> {
        self.modules
            .clone()
            .unwrap_or_else(|| DEFAULT_FULL_AGENT_MODULES.to_vec())
    }

    /// Inputs flattened across steps, or the legacy `inputs` list when
    /// `steps` is empty. Used for places that need to summarize the entire
    /// sensory feed of a case (e.g. judge prompt rendering).
    pub fn flat_inputs(&self) -> Vec<&FullAgentInput> {
        if !self.steps.is_empty() {
            self.steps
                .iter()
                .flat_map(|step| step.inputs.iter())
                .collect()
        } else {
            self.inputs.iter().collect()
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModuleEvalTarget {
    CognitionGate,
    QueryVector,
    AttentionSchema,
    SelfModel,
    Memory,
    MemoryCompaction,
    MemoryAssociation,
    MemoryRecombination,
    SpeakGate,
    Speak,
}

impl ModuleEvalTarget {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::CognitionGate => "cognition-gate",
            Self::QueryVector => "query-vector",
            Self::AttentionSchema => "attention-schema",
            Self::SelfModel => "self-model",
            Self::Memory => "memory",
            Self::MemoryCompaction => "memory-compaction",
            Self::MemoryAssociation => "memory-association",
            Self::MemoryRecombination => "memory-recombination",
            Self::SpeakGate => "speak-gate",
            Self::Speak => "speak",
        }
    }

    pub fn module(self) -> EvalModule {
        match self {
            Self::CognitionGate => EvalModule::CognitionGate,
            Self::QueryVector => EvalModule::QueryVector,
            Self::AttentionSchema => EvalModule::AttentionSchema,
            Self::SelfModel => EvalModule::SelfModel,
            Self::Memory => EvalModule::Memory,
            Self::MemoryCompaction => EvalModule::MemoryCompaction,
            Self::MemoryAssociation => EvalModule::MemoryAssociation,
            Self::MemoryRecombination => EvalModule::MemoryRecombination,
            Self::SpeakGate => EvalModule::SpeakGate,
            Self::Speak => EvalModule::Speak,
        }
    }

    fn from_path(path: &Path) -> Option<Self> {
        path.components()
            .filter_map(|component| component.as_os_str().to_str())
            .find_map(|part| match part {
                "cognition-gate" => Some(Self::CognitionGate),
                "query-vector" => Some(Self::QueryVector),
                "attention-schema" => Some(Self::AttentionSchema),
                "self-model" => Some(Self::SelfModel),
                "memory" => Some(Self::Memory),
                "memory-compaction" => Some(Self::MemoryCompaction),
                "memory-association" => Some(Self::MemoryAssociation),
                "memory-recombination" => Some(Self::MemoryRecombination),
                "speak-gate" => Some(Self::SpeakGate),
                "speak" => Some(Self::Speak),
                _ => None,
            })
    }
}

#[derive(Debug, Clone)]
pub enum EvalCase {
    FullAgent(FullAgentCase),
    Module {
        target: ModuleEvalTarget,
        case: ModuleCase,
    },
}

impl EvalCase {
    pub fn id(&self) -> Option<&str> {
        match self {
            Self::FullAgent(case) => case.id.as_deref(),
            Self::Module { case, .. } => case.id.as_deref(),
        }
    }

    pub fn description(&self) -> Option<&Text> {
        match self {
            Self::FullAgent(case) => case.description.as_ref(),
            Self::Module { case, .. } => case.description.as_ref(),
        }
    }

    pub fn memories(&self) -> &[MemorySeed] {
        match self {
            Self::FullAgent(case) => &case.memories,
            Self::Module { case, .. } => &case.memories,
        }
    }

    pub fn limits(&self) -> &EvalLimits {
        match self {
            Self::FullAgent(case) => &case.limits,
            Self::Module { case, .. } => &case.limits,
        }
    }

    pub fn checks(&self) -> &[Check] {
        match self {
            Self::FullAgent(case) => &case.checks,
            Self::Module { case, .. } => &case.checks,
        }
    }

    pub fn modules_checks(&self) -> &[ModuleChecks] {
        match self {
            Self::FullAgent(case) => &case.modules_checks,
            Self::Module { .. } => &[],
        }
    }

    pub fn scoring(&self) -> &CaseScoring {
        match self {
            Self::FullAgent(case) => &case.scoring,
            Self::Module { case, .. } => &case.scoring,
        }
    }

    pub fn prompt_for_judge(&self) -> String {
        match self {
            Self::FullAgent(case) => case
                .flat_inputs()
                .into_iter()
                .map(FullAgentInput::as_prompt_line)
                .collect::<Vec<_>>()
                .join("\n"),
            Self::Module { case, .. } => case.prompt.content.clone(),
        }
    }

    pub fn context_for_judge(&self) -> Option<String> {
        match self {
            Self::FullAgent(_) => None,
            Self::Module { case, .. } => case.context.as_ref().map(|text| text.content.clone()),
        }
    }
}

impl FullAgentInput {
    pub fn as_prompt_line(&self) -> String {
        match self {
            Self::Heard { direction, content } => {
                format!("heard{}: {}", direction_suffix(direction), content.content)
            }
            Self::Seen {
                direction,
                appearance,
            } => {
                format!(
                    "seen{}: {}",
                    direction_suffix(direction),
                    appearance.content
                )
            }
            Self::OneShot {
                modality,
                direction,
                content,
            } => {
                format!(
                    "one-shot:{modality}{}: {}",
                    direction_suffix(direction),
                    content.content
                )
            }
        }
    }
}

fn direction_suffix(direction: &Option<String>) -> String {
    direction
        .as_deref()
        .map(|direction| format!("[{direction}]"))
        .unwrap_or_default()
}

#[derive(Debug, Clone, FromEure)]
#[eure(crate = ::eure::document, rename_all = "kebab-case")]
pub struct EvalLimits {
    #[eure(default = "default_max_llm_calls")]
    pub max_llm_calls: Option<u64>,
}

impl Default for EvalLimits {
    fn default() -> Self {
        Self {
            max_llm_calls: default_max_llm_calls(),
        }
    }
}

#[derive(Debug, Clone, FromEure)]
#[eure(crate = ::eure::document, rename_all = "kebab-case")]
pub struct MemorySeed {
    #[eure(default = "default_memory_rank")]
    pub rank: MemorySeedRank,
    #[eure(default = "default_memory_decay_secs")]
    pub decay_secs: i64,
    #[eure(default)]
    pub datetime: Option<String>,
    #[eure(default)]
    pub seconds_ago: Option<i64>,
    pub content: Text,
}

#[derive(Debug, Clone, FromEure)]
#[eure(crate = ::eure::document, rename_all = "kebab-case")]
pub struct MemoSeed {
    pub module: String,
    #[eure(default = "default_memo_replica")]
    pub replica: u8,
    pub content: Text,
    #[eure(default = "default_seed_seconds_ago")]
    pub seconds_ago: i64,
}

#[derive(Debug, Clone, FromEure)]
#[eure(crate = ::eure::document, rename_all = "kebab-case")]
pub struct CognitionLogSeed {
    pub text: Text,
    #[eure(default = "default_seed_seconds_ago")]
    pub seconds_ago: i64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, FromEure)]
#[eure(crate = ::eure::document, rename_all = "kebab-case")]
pub enum MemorySeedRank {
    ShortTerm,
    MidTerm,
    LongTerm,
    Permanent,
    Identity,
}

impl From<MemorySeedRank> for MemoryRank {
    fn from(rank: MemorySeedRank) -> Self {
        match rank {
            MemorySeedRank::ShortTerm => Self::ShortTerm,
            MemorySeedRank::MidTerm => Self::MidTerm,
            MemorySeedRank::LongTerm => Self::LongTerm,
            MemorySeedRank::Permanent => Self::Permanent,
            MemorySeedRank::Identity => Self::Identity,
        }
    }
}

#[derive(Debug, Clone, Default, FromEure)]
#[eure(crate = ::eure::document, rename_all = "kebab-case")]
pub struct CheckCommon {
    #[eure(default)]
    pub name: Option<String>,
    #[eure(default)]
    pub must_pass: bool,
    #[eure(default = "default_weight")]
    pub weight: i64,
}

#[derive(Debug, Clone, FromEure)]
#[eure(crate = ::eure::document, rename_all = "kebab-case")]
pub struct ModuleChecks {
    pub module: EvalModule,
    #[eure(default)]
    pub rubrics: Vec<ModuleRubric>,
}

#[derive(Debug, Clone, FromEure)]
#[eure(crate = ::eure::document, rename_all = "kebab-case")]
pub struct ModuleRubric {
    #[eure(default)]
    pub name: Option<String>,
    pub rubric: Text,
    #[eure(default = "default_pass_score")]
    pub pass_score: f64,
    #[eure(default = "default_module_rubric_judge_inputs")]
    pub judge_inputs: Vec<RubricJudgeInput>,
    #[eure(default)]
    pub criteria: Vec<RubricCriterion>,
}

impl ModuleRubric {
    pub fn display_name(&self) -> String {
        self.name
            .clone()
            .unwrap_or_else(|| "module-rubric".to_string())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, FromEure)]
#[eure(crate = ::eure::document, rename_all = "kebab-case")]
pub enum ArtifactTextField {
    Output,
    Failure,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, FromEure)]
#[eure(crate = ::eure::document, rename_all = "kebab-case")]
pub enum RubricJudgeInput {
    Output,
    Utterance,
    Failure,
    Trace,
    Observations,
    Blackboard,
    Memory,
    Memos,
    Cognition,
    Allocation,
}

#[derive(Debug, Clone, FromEure)]
#[eure(
    crate = ::eure::document,
    rename_all = "kebab-case",
    rename_all_fields = "kebab-case"
)]
pub enum Check {
    ArtifactTextContains {
        #[eure(flatten)]
        common: CheckCommon,
        #[eure(default)]
        field: Option<ArtifactTextField>,
        contains: String,
    },
    ArtifactTextExact {
        #[eure(flatten)]
        common: CheckCommon,
        #[eure(default)]
        field: Option<ArtifactTextField>,
        exact: Text,
    },
    JsonPointerEquals {
        #[eure(flatten)]
        common: CheckCommon,
        pointer: String,
        expected: String,
    },
    JsonPointerContains {
        #[eure(flatten)]
        common: CheckCommon,
        pointer: String,
        contains: String,
    },
    Rubric {
        #[eure(flatten)]
        common: CheckCommon,
        rubric: Text,
        #[eure(default = "default_pass_score")]
        pass_score: f64,
        #[eure(default = "default_judge_inputs")]
        judge_inputs: Vec<RubricJudgeInput>,
        #[eure(default)]
        criteria: Vec<RubricCriterion>,
    },
    TraceSpan {
        #[eure(flatten)]
        common: CheckCommon,
        span_name: String,
    },
    TraceEvent {
        #[eure(flatten)]
        common: CheckCommon,
        message_contains: String,
    },
    TraceSpansOrdered {
        #[eure(flatten)]
        common: CheckCommon,
        names: Vec<String>,
    },
}

#[derive(Debug, Clone, FromEure)]
#[eure(crate = ::eure::document, rename_all = "kebab-case")]
pub struct RubricCriterion {
    pub name: String,
    pub description: Text,
    #[eure(default = "default_weight")]
    pub weight: i64,
    #[eure(default = "default_pass_score")]
    pub pass_score: f64,
}

#[derive(Debug, Clone, FromEure)]
#[eure(crate = ::eure::document, rename_all = "kebab-case")]
pub struct CaseScoring {
    #[eure(default = "default_judge_max_output_tokens")]
    pub judge_max_output_tokens: u32,
}

impl Default for CaseScoring {
    fn default() -> Self {
        Self {
            judge_max_output_tokens: default_judge_max_output_tokens(),
        }
    }
}

impl Check {
    pub fn common(&self) -> &CheckCommon {
        match self {
            Self::ArtifactTextContains { common, .. }
            | Self::ArtifactTextExact { common, .. }
            | Self::JsonPointerEquals { common, .. }
            | Self::JsonPointerContains { common, .. }
            | Self::Rubric { common, .. }
            | Self::TraceSpan { common, .. }
            | Self::TraceEvent { common, .. }
            | Self::TraceSpansOrdered { common, .. } => common,
        }
    }

    pub fn kind_name(&self) -> &'static str {
        match self {
            Self::ArtifactTextContains { .. } => "artifact-text-contains",
            Self::ArtifactTextExact { .. } => "artifact-text-exact",
            Self::JsonPointerEquals { .. } => "json-pointer-equals",
            Self::JsonPointerContains { .. } => "json-pointer-contains",
            Self::Rubric { .. } => "rubric",
            Self::TraceSpan { .. } => "trace-span",
            Self::TraceEvent { .. } => "trace-event",
            Self::TraceSpansOrdered { .. } => "trace-spans-ordered",
        }
    }

    pub fn display_name(&self) -> String {
        self.common()
            .name
            .clone()
            .unwrap_or_else(|| self.kind_name().to_string())
    }
}

#[derive(Debug, Error)]
pub enum CaseFileError {
    #[error("failed to read eval case {path}: {source}")]
    Read {
        path: PathBuf,
        #[source]
        source: io::Error,
    },
    #[error("failed to parse eval case {path}: {message}")]
    Parse { path: PathBuf, message: String },
    #[error("invalid eval case {path}: {message}")]
    Validation { path: PathBuf, message: String },
}

pub fn parse_case_file(path: &Path) -> Result<EvalCase, CaseFileError> {
    if is_full_agent_case_path(path) {
        return parse_full_agent_case_file(path).map(EvalCase::FullAgent);
    }
    let target = ModuleEvalTarget::from_path(path).ok_or_else(|| CaseFileError::Validation {
        path: path.to_path_buf(),
        message: "module eval case path must include a supported module directory".to_string(),
    })?;
    let case = parse_module_case_file(path)?;
    validate_module_case_target(path, target, &case)?;
    Ok(EvalCase::Module { target, case })
}

pub fn parse_full_agent_case_file(path: &Path) -> Result<FullAgentCase, CaseFileError> {
    let content = read_case(path)?;
    let file: FullAgentCaseFile =
        eure::parse_content(&content, path.to_path_buf()).map_err(|message| {
            CaseFileError::Parse {
                path: path.to_path_buf(),
                message,
            }
        })?;
    validate_full_agent_case(path, &file.case)?;
    Ok(file.case)
}

pub fn parse_module_case_file(path: &Path) -> Result<ModuleCase, CaseFileError> {
    let content = read_case(path)?;
    let file: ModuleCaseFile =
        eure::parse_content(&content, path.to_path_buf()).map_err(|message| {
            CaseFileError::Parse {
                path: path.to_path_buf(),
                message,
            }
        })?;
    validate_module_case(path, &file.case)?;
    Ok(file.case)
}

pub(crate) fn parse_case_now(now: Option<&str>) -> Result<Option<DateTime<FixedOffset>>, String> {
    now.map(|value| {
        DateTime::parse_from_rfc3339(value.trim())
            .map_err(|error| format!("now must be RFC3339 datetime: {error}"))
    })
    .transpose()
}

pub(crate) fn parse_memory_datetime(
    value: &str,
    case_now: Option<DateTime<FixedOffset>>,
) -> Result<DateTime<Utc>, String> {
    let value = value.trim();
    if let Ok(datetime) = DateTime::parse_from_rfc3339(value) {
        return Ok(datetime.with_timezone(&Utc));
    }

    let date = NaiveDate::parse_from_str(value, "%Y-%m-%d").map_err(|error| {
        format!("memory datetime must be RFC3339 datetime or YYYY-MM-DD: {error}")
    })?;
    let offset = case_now
        .map(|now| *now.offset())
        .unwrap_or_else(|| FixedOffset::east_opt(0).expect("zero offset is valid"));
    offset
        .with_ymd_and_hms(date.year(), date.month(), date.day(), 0, 0, 0)
        .single()
        .ok_or_else(|| format!("memory datetime date is not representable: {value}"))
        .map(|datetime| datetime.with_timezone(&Utc))
}

fn read_case(path: &Path) -> Result<String, CaseFileError> {
    fs::read_to_string(path).map_err(|source| CaseFileError::Read {
        path: path.to_path_buf(),
        source,
    })
}

pub fn discover_case_files(root: &Path) -> Result<Vec<PathBuf>, io::Error> {
    let mut files = Vec::new();
    discover_case_files_inner(root, &mut files)?;
    files.sort();
    Ok(files)
}

fn discover_case_files_inner(root: &Path, files: &mut Vec<PathBuf>) -> Result<(), io::Error> {
    if root.is_file() {
        if root.extension().is_some_and(|ext| ext == "eure") {
            files.push(root.to_path_buf());
        }
        return Ok(());
    }

    for entry in fs::read_dir(root)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            if should_skip_case_dir(&path) {
                continue;
            }
            discover_case_files_inner(&path, files)?;
        } else if path.extension().is_some_and(|ext| ext == "eure") && !should_skip_case_file(&path)
        {
            files.push(path);
        }
    }

    Ok(())
}

fn should_skip_case_dir(path: &Path) -> bool {
    path.file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| name == "results")
}

fn should_skip_case_file(path: &Path) -> bool {
    path.file_name()
        .and_then(|name| name.to_str())
        .is_some_and(is_persisted_eval_output_file_name)
}

fn is_persisted_eval_output_file_name(name: &str) -> bool {
    matches!(
        name,
        "result.eure" | "report.eure" | "artifact.eure" | "last-state.eure"
    )
}

pub(crate) fn is_full_agent_case_path(path: &Path) -> bool {
    path.components()
        .filter_map(|component| component.as_os_str().to_str())
        .any(|part| part == "full-agent")
}

fn validate_full_agent_case(path: &Path, case: &FullAgentCase) -> Result<(), CaseFileError> {
    if !case.inputs.is_empty() && !case.steps.is_empty() {
        return Err(CaseFileError::Validation {
            path: path.to_path_buf(),
            message: "full-agent case must use either `inputs` or `steps`, not both".to_string(),
        });
    }
    if case.inputs.is_empty() && case.steps.is_empty() {
        return Err(CaseFileError::Validation {
            path: path.to_path_buf(),
            message: "full-agent case must have at least one input or one step".to_string(),
        });
    }
    for (index, input) in case.inputs.iter().enumerate() {
        validate_full_agent_input(path, &format!("inputs[{index}]"), input)?;
    }
    for (step_index, step) in case.steps.iter().enumerate() {
        if step.inputs.is_empty() {
            return Err(CaseFileError::Validation {
                path: path.to_path_buf(),
                message: format!("steps[{step_index}].inputs must not be empty"),
            });
        }
        for (input_index, input) in step.inputs.iter().enumerate() {
            validate_full_agent_input(
                path,
                &format!("steps[{step_index}].inputs[{input_index}]"),
                input,
            )?;
        }
        if let Some(wait_for) = &step.wait_for {
            validate_wait_for(path, step_index, wait_for)?;
        }
        for check in &step.checks {
            validate_check(path, check)?;
            if !is_step_compatible_check(check) {
                return Err(CaseFileError::Validation {
                    path: path.to_path_buf(),
                    message: format!(
                        "steps[{step_index}].checks: {kind} cannot run mid-step (use the case-level checks instead)",
                        kind = check.kind_name()
                    ),
                });
            }
        }
    }
    validate_modules(path, case.modules.as_deref())?;
    validate_module_checks(path, case)?;
    validate_common(
        path,
        case.now.as_deref(),
        &case.memories,
        &case.memos,
        &case.limits,
        &case.checks,
    )
}

fn validate_full_agent_input(
    path: &Path,
    label: &str,
    input: &FullAgentInput,
) -> Result<(), CaseFileError> {
    match input {
        FullAgentInput::Heard { content, .. } if content.content.trim().is_empty() => {
            Err(CaseFileError::Validation {
                path: path.to_path_buf(),
                message: format!("{label}.content must not be empty"),
            })
        }
        FullAgentInput::Seen { appearance, .. } if appearance.content.trim().is_empty() => {
            Err(CaseFileError::Validation {
                path: path.to_path_buf(),
                message: format!("{label}.appearance must not be empty"),
            })
        }
        FullAgentInput::OneShot { modality, .. } if modality.trim().is_empty() => {
            Err(CaseFileError::Validation {
                path: path.to_path_buf(),
                message: format!("{label}.modality must not be empty"),
            })
        }
        FullAgentInput::OneShot { content, .. } if content.content.trim().is_empty() => {
            Err(CaseFileError::Validation {
                path: path.to_path_buf(),
                message: format!("{label}.content must not be empty"),
            })
        }
        _ => Ok(()),
    }
}

fn validate_wait_for(
    path: &Path,
    step_index: usize,
    wait_for: &WaitFor,
) -> Result<(), CaseFileError> {
    match wait_for {
        WaitFor::MemoFrom { timeout_ms, .. } => {
            if *timeout_ms == 0 {
                return Err(CaseFileError::Validation {
                    path: path.to_path_buf(),
                    message: format!(
                        "steps[{step_index}].wait-for.timeout-ms must be greater than zero"
                    ),
                });
            }
        }
    }
    Ok(())
}

/// Mid-step checks can only consult agent observations (JSON pointers / text in
/// the running artifact). Trace and rubric checks require a completed run.
fn is_step_compatible_check(check: &Check) -> bool {
    matches!(
        check,
        Check::JsonPointerEquals { .. }
            | Check::JsonPointerContains { .. }
            | Check::ArtifactTextContains { .. }
            | Check::ArtifactTextExact { .. }
    )
}

fn validate_module_case(path: &Path, case: &ModuleCase) -> Result<(), CaseFileError> {
    if case.prompt.content.trim().is_empty() {
        return Err(CaseFileError::Validation {
            path: path.to_path_buf(),
            message: "prompt must not be empty".to_string(),
        });
    }
    for (index, participant) in case.participants.iter().enumerate() {
        if participant.trim().is_empty() {
            return Err(CaseFileError::Validation {
                path: path.to_path_buf(),
                message: format!("participants[{index}] must not be empty"),
            });
        }
    }
    for (index, seed) in case.cognition_log.iter().enumerate() {
        if seed.text.content.trim().is_empty() {
            return Err(CaseFileError::Validation {
                path: path.to_path_buf(),
                message: format!("cognition-log[{index}].text must not be empty"),
            });
        }
        if seed.seconds_ago < 0 {
            return Err(CaseFileError::Validation {
                path: path.to_path_buf(),
                message: format!("cognition-log[{index}].seconds-ago must not be negative"),
            });
        }
    }
    validate_modules(path, case.modules.as_deref())?;
    validate_common(
        path,
        case.now.as_deref(),
        &case.memories,
        &case.memos,
        &case.limits,
        &case.checks,
    )
}

fn validate_module_case_target(
    path: &Path,
    target: ModuleEvalTarget,
    case: &ModuleCase,
) -> Result<(), CaseFileError> {
    let Some(modules) = case.modules.as_deref() else {
        return Ok(());
    };
    let target_module = target.module();
    if modules.contains(&target_module) {
        return Ok(());
    }
    Err(CaseFileError::Validation {
        path: path.to_path_buf(),
        message: format!(
            "modules must include target module '{}' for {} eval cases",
            target_module.as_str(),
            target.as_str(),
        ),
    })
}

fn validate_modules(path: &Path, modules: Option<&[EvalModule]>) -> Result<(), CaseFileError> {
    let Some(modules) = modules else {
        return Ok(());
    };
    if modules.is_empty() {
        return Err(CaseFileError::Validation {
            path: path.to_path_buf(),
            message: "modules must not be empty when present".to_string(),
        });
    }

    let mut seen = BTreeSet::new();
    for module in modules {
        if !seen.insert(*module) {
            return Err(CaseFileError::Validation {
                path: path.to_path_buf(),
                message: format!("modules contains duplicate module '{}'", module.as_str()),
            });
        }
    }
    Ok(())
}

fn validate_module_checks(path: &Path, case: &FullAgentCase) -> Result<(), CaseFileError> {
    let modules = case.effective_modules();
    let mut seen = BTreeSet::new();
    for (index, checks) in case.modules_checks.iter().enumerate() {
        if !modules.contains(&checks.module) {
            return Err(CaseFileError::Validation {
                path: path.to_path_buf(),
                message: format!(
                    "modules-checks[{index}].module '{}' must be included in full-agent modules",
                    checks.module.as_str()
                ),
            });
        }
        if !seen.insert(checks.module) {
            return Err(CaseFileError::Validation {
                path: path.to_path_buf(),
                message: format!(
                    "modules-checks contains duplicate module '{}'",
                    checks.module.as_str()
                ),
            });
        }
        if checks.rubrics.is_empty() {
            return Err(CaseFileError::Validation {
                path: path.to_path_buf(),
                message: format!(
                    "modules-checks[{index}] for '{}' must contain at least one rubric",
                    checks.module.as_str()
                ),
            });
        }
        for rubric in &checks.rubrics {
            let name = rubric.display_name();
            if rubric
                .name
                .as_deref()
                .is_some_and(|name| name.trim().is_empty())
            {
                return Err(CaseFileError::Validation {
                    path: path.to_path_buf(),
                    message: format!(
                        "module rubric for '{}' has an empty name",
                        checks.module.as_str()
                    ),
                });
            }
            validate_rubric_fields(
                path,
                &format!("module {} rubric '{}'", checks.module.as_str(), name),
                &rubric.rubric,
                rubric.pass_score,
                &rubric.judge_inputs,
                &rubric.criteria,
            )?;
        }
    }
    Ok(())
}

fn validate_common(
    path: &Path,
    now: Option<&str>,
    memories: &[MemorySeed],
    memos: &[MemoSeed],
    limits: &EvalLimits,
    checks: &[Check],
) -> Result<(), CaseFileError> {
    if matches!(limits.max_llm_calls, Some(0)) {
        return Err(CaseFileError::Validation {
            path: path.to_path_buf(),
            message: "limits.max-llm-calls must be greater than zero when present".to_string(),
        });
    }
    let case_now = parse_case_now(now).map_err(|message| CaseFileError::Validation {
        path: path.to_path_buf(),
        message,
    })?;

    for (index, memory) in memories.iter().enumerate() {
        if memory.content.content.trim().is_empty() {
            return Err(CaseFileError::Validation {
                path: path.to_path_buf(),
                message: format!("memories[{index}].content must not be empty"),
            });
        }
        if memory.decay_secs < 0 {
            return Err(CaseFileError::Validation {
                path: path.to_path_buf(),
                message: format!("memories[{index}].decay-secs must not be negative"),
            });
        }
        if memory.datetime.is_some() && memory.seconds_ago.is_some() {
            return Err(CaseFileError::Validation {
                path: path.to_path_buf(),
                message: format!(
                    "memories[{index}] must not specify both datetime and seconds-ago"
                ),
            });
        }
        if matches!(memory.seconds_ago, Some(seconds_ago) if seconds_ago < 0) {
            return Err(CaseFileError::Validation {
                path: path.to_path_buf(),
                message: format!("memories[{index}].seconds-ago must not be negative"),
            });
        }
        if let Some(datetime) = &memory.datetime {
            parse_memory_datetime(datetime, case_now).map_err(|message| {
                CaseFileError::Validation {
                    path: path.to_path_buf(),
                    message: format!("memories[{index}].datetime is invalid: {message}"),
                }
            })?;
        }
    }

    for (index, memo) in memos.iter().enumerate() {
        if memo.module.trim().is_empty() {
            return Err(CaseFileError::Validation {
                path: path.to_path_buf(),
                message: format!("memos[{index}].module must not be empty"),
            });
        }
        if let Err(error) = ModuleId::new(memo.module.clone()) {
            return Err(CaseFileError::Validation {
                path: path.to_path_buf(),
                message: format!("memos[{index}].module is invalid: {error}"),
            });
        }
        if memo.content.content.trim().is_empty() {
            return Err(CaseFileError::Validation {
                path: path.to_path_buf(),
                message: format!("memos[{index}].content must not be empty"),
            });
        }
        if memo.seconds_ago < 0 {
            return Err(CaseFileError::Validation {
                path: path.to_path_buf(),
                message: format!("memos[{index}].seconds-ago must not be negative"),
            });
        }
    }

    for check in checks {
        validate_check(path, check)?;
    }

    Ok(())
}

fn validate_check(path: &Path, check: &Check) -> Result<(), CaseFileError> {
    if let Check::JsonPointerEquals { pointer, .. } | Check::JsonPointerContains { pointer, .. } =
        check
    {
        validate_json_pointer(path, pointer)?;
    }

    if let Check::Rubric {
        rubric,
        pass_score,
        judge_inputs,
        criteria,
        ..
    } = check
    {
        validate_rubric_fields(
            path,
            &format!("rubric check '{}'", check.display_name()),
            rubric,
            *pass_score,
            judge_inputs,
            criteria,
        )?;
    }

    match check {
        Check::TraceSpan { span_name, .. } if span_name.trim().is_empty() => {
            return Err(CaseFileError::Validation {
                path: path.to_path_buf(),
                message: format!(
                    "trace check '{}' has an empty span name",
                    check.display_name()
                ),
            });
        }
        Check::TraceSpansOrdered { names, .. } if names.is_empty() => {
            return Err(CaseFileError::Validation {
                path: path.to_path_buf(),
                message: format!(
                    "trace check '{}' must list at least one span name",
                    check.display_name()
                ),
            });
        }
        Check::TraceSpansOrdered { names, .. }
            if names.iter().any(|name| name.trim().is_empty()) =>
        {
            return Err(CaseFileError::Validation {
                path: path.to_path_buf(),
                message: format!(
                    "trace check '{}' has an empty span name",
                    check.display_name()
                ),
            });
        }
        _ => {}
    }

    Ok(())
}

fn validate_rubric_fields(
    path: &Path,
    label: &str,
    rubric: &Text,
    pass_score: f64,
    judge_inputs: &[RubricJudgeInput],
    criteria: &[RubricCriterion],
) -> Result<(), CaseFileError> {
    if rubric.content.trim().is_empty() {
        return Err(CaseFileError::Validation {
            path: path.to_path_buf(),
            message: format!("{label} has an empty rubric"),
        });
    }
    validate_pass_score(path, pass_score, label)?;
    if judge_inputs.is_empty() {
        return Err(CaseFileError::Validation {
            path: path.to_path_buf(),
            message: format!("{label} has empty judge-inputs"),
        });
    }
    let mut names = BTreeSet::new();
    for criterion in criteria {
        if criterion.name.trim().is_empty() {
            return Err(CaseFileError::Validation {
                path: path.to_path_buf(),
                message: format!("{label} has an empty criterion name"),
            });
        }
        if !names.insert(criterion.name.clone()) {
            return Err(CaseFileError::Validation {
                path: path.to_path_buf(),
                message: format!("{label} has duplicate criterion '{}'", criterion.name),
            });
        }
        validate_pass_score(path, criterion.pass_score, &criterion.name)?;
    }
    Ok(())
}

fn validate_json_pointer(path: &Path, pointer: &str) -> Result<(), CaseFileError> {
    if pointer.is_empty() || pointer.starts_with('/') {
        return Ok(());
    }
    Err(CaseFileError::Validation {
        path: path.to_path_buf(),
        message: format!("json pointer '{pointer}' must be empty or start with '/'"),
    })
}

fn validate_pass_score(path: &Path, score: f64, name: &str) -> Result<(), CaseFileError> {
    if (0.0..=1.0).contains(&score) {
        return Ok(());
    }
    Err(CaseFileError::Validation {
        path: path.to_path_buf(),
        message: format!("pass score for '{name}' must be between 0.0 and 1.0"),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn date_only_memory_datetime_uses_case_now_offset() {
        let case_now = parse_case_now(Some("2026-05-10T08:21:00+09:00"))
            .unwrap()
            .unwrap();

        let occurred_at = parse_memory_datetime("2025-05-10", Some(case_now)).unwrap();

        assert_eq!(occurred_at.to_rfc3339(), "2025-05-09T15:00:00+00:00");
    }

    #[test]
    fn memory_datetime_accepts_rfc3339() {
        let occurred_at = parse_memory_datetime("2025-05-10T08:21:00+09:00", None).unwrap();

        assert_eq!(occurred_at.to_rfc3339(), "2025-05-09T23:21:00+00:00");
    }

    fn full_agent_case_with(inputs: Vec<FullAgentInput>, steps: Vec<EvalStep>) -> FullAgentCase {
        FullAgentCase {
            id: Some("case".to_string()),
            description: None,
            now: None,
            modules: None,
            inputs,
            steps,
            participants: Vec::new(),
            memories: Vec::new(),
            memos: Vec::new(),
            limits: EvalLimits::default(),
            checks: Vec::new(),
            modules_checks: Vec::new(),
            scoring: CaseScoring::default(),
        }
    }

    fn seen_input(appearance: &str) -> FullAgentInput {
        FullAgentInput::Seen {
            direction: None,
            appearance: Text::plaintext(appearance),
        }
    }

    #[test]
    fn validate_full_agent_case_rejects_both_inputs_and_steps() {
        let case = full_agent_case_with(
            vec![seen_input("input")],
            vec![EvalStep {
                description: None,
                inputs: vec![seen_input("step")],
                wait_for: None,
                checks: Vec::new(),
            }],
        );

        let error = validate_full_agent_case(Path::new("case.eure"), &case).unwrap_err();
        assert!(
            matches!(error, CaseFileError::Validation { message, .. } if message.contains("either `inputs` or `steps`"))
        );
    }

    #[test]
    fn validate_full_agent_case_rejects_empty_step_inputs() {
        let case = full_agent_case_with(
            Vec::new(),
            vec![EvalStep {
                description: None,
                inputs: Vec::new(),
                wait_for: None,
                checks: Vec::new(),
            }],
        );

        let error = validate_full_agent_case(Path::new("case.eure"), &case).unwrap_err();
        assert!(
            matches!(error, CaseFileError::Validation { message, .. } if message.contains("steps[0].inputs must not be empty"))
        );
    }

    #[test]
    fn validate_full_agent_case_rejects_zero_wait_timeout() {
        let case = full_agent_case_with(
            Vec::new(),
            vec![EvalStep {
                description: None,
                inputs: vec![seen_input("step")],
                wait_for: Some(WaitFor::MemoFrom {
                    module: EvalModule::Predict,
                    timeout_ms: 0,
                }),
                checks: Vec::new(),
            }],
        );

        let error = validate_full_agent_case(Path::new("case.eure"), &case).unwrap_err();
        assert!(
            matches!(error, CaseFileError::Validation { message, .. } if message.contains("timeout-ms must be greater than zero"))
        );
    }

    #[test]
    fn validate_full_agent_case_rejects_unsupported_step_check_kind() {
        let trace_check = Check::TraceSpan {
            common: CheckCommon::default(),
            span_name: "x".to_string(),
        };
        let case = full_agent_case_with(
            Vec::new(),
            vec![EvalStep {
                description: None,
                inputs: vec![seen_input("step")],
                wait_for: None,
                checks: vec![trace_check],
            }],
        );

        let error = validate_full_agent_case(Path::new("case.eure"), &case).unwrap_err();
        assert!(
            matches!(error, CaseFileError::Validation { message, .. } if message.contains("trace-span cannot run mid-step"))
        );
    }

    #[test]
    fn parses_surprise_on_prediction_violation_sample_case() {
        let path = Path::new("../../eval-cases/full-agent/surprise-on-prediction-violation.eure");
        let case = parse_full_agent_case_file(path).expect("sample case should parse");
        assert_eq!(case.steps.len(), 2);
        assert!(case.inputs.is_empty());
        assert!(matches!(
            case.steps[0].wait_for,
            Some(WaitFor::MemoFrom {
                module: EvalModule::Predict,
                ..
            })
        ));
        assert!(matches!(
            case.steps[1].wait_for,
            Some(WaitFor::MemoFrom {
                module: EvalModule::Surprise,
                ..
            })
        ));
        assert_eq!(case.steps[0].checks.len(), 1);
    }

    #[test]
    fn validate_full_agent_case_accepts_steps_with_memo_from_wait() {
        let case = full_agent_case_with(
            Vec::new(),
            vec![
                EvalStep {
                    description: Some(Text::plaintext("step 1")),
                    inputs: vec![seen_input("baseline")],
                    wait_for: Some(WaitFor::MemoFrom {
                        module: EvalModule::Predict,
                        timeout_ms: 5_000,
                    }),
                    checks: vec![Check::JsonPointerContains {
                        common: CheckCommon::default(),
                        pointer: "/observations/agent/memo_logs/predict/0/content".to_string(),
                        contains: "predict".to_string(),
                    }],
                },
                EvalStep {
                    description: Some(Text::plaintext("step 2")),
                    inputs: vec![seen_input("violation")],
                    wait_for: Some(WaitFor::MemoFrom {
                        module: EvalModule::Surprise,
                        timeout_ms: 5_000,
                    }),
                    checks: Vec::new(),
                },
            ],
        );

        validate_full_agent_case(Path::new("case.eure"), &case).unwrap();
    }

    #[test]
    fn memory_seed_rejects_datetime_and_seconds_ago_together() {
        let memory = MemorySeed {
            rank: MemorySeedRank::Permanent,
            decay_secs: 0,
            datetime: Some("2025-05-10".to_string()),
            seconds_ago: Some(60),
            content: Text::plaintext("memory"),
        };

        let error = validate_common(
            Path::new("case.eure"),
            Some("2026-05-10T08:21:00+09:00"),
            &[memory],
            &[],
            &EvalLimits::default(),
            &[],
        )
        .unwrap_err();

        assert!(
            matches!(error, CaseFileError::Validation { message, .. } if message.contains("must not specify both datetime and seconds-ago"))
        );
    }
}
