use std::{
    collections::BTreeSet,
    fs, io,
    path::{Path, PathBuf},
};

use chrono::{DateTime, Datelike as _, FixedOffset, NaiveDate, TimeZone as _, Utc};
use eure::{
    FromEure,
    document::parse::{ParseContext, ParseError, ParseErrorKind},
    value::Text,
};
use nuillu_types::{
    MemoryRank, ModuleId, PolicyRank, ReplicaIndex, ScopeId, SubsystemId, SubsystemInstanceId,
};
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

fn default_policy_rank() -> PolicySeedRank {
    PolicySeedRank::Established
}

fn default_policy_decay_secs() -> i64 {
    2_592_000
}

fn default_seed_seconds_ago() -> i64 {
    0
}

fn default_memo_replica() -> u8 {
    0
}

fn default_scope() -> String {
    "/".to_string()
}

fn default_cognition_module() -> String {
    "cognition-gate".to_string()
}

fn default_pass_score() -> f64 {
    0.8
}

fn default_judge_max_output_tokens() -> u32 {
    1200
}

fn default_max_llm_calls() -> Option<u64> {
    Some(10)
}

fn default_case_timeout_ms() -> u64 {
    60_000
}

fn default_wait_max_matches() -> usize {
    1
}

fn default_quiet_sleep_threshold_ms() -> u64 {
    30_000
}

fn default_arousal_change_multiplier() -> f64 {
    1.0
}

fn default_wake_arousal_at_least() -> f64 {
    -1.0
}

fn default_wake_arousal_at_most() -> f64 {
    2.0
}

#[derive(Debug, Clone, FromEure)]
#[eure(crate = ::eure::document, rename_all = "kebab-case")]
pub struct RuntimeCaseFile {
    #[eure(flatten)]
    pub case: RuntimeCase,
}

#[derive(Debug, Clone, FromEure)]
#[eure(crate = ::eure::document, rename_all = "kebab-case")]
pub struct RuntimeCase {
    #[eure(default)]
    pub id: Option<String>,
    #[eure(default)]
    pub description: Option<Text>,
    /// Runtime topology to evaluate, resolved relative to this case file.
    pub runtime_config: String,
    #[eure(default)]
    pub now: Option<String>,
    #[eure(default)]
    pub prompt: Option<Text>,
    #[eure(default)]
    pub context: Option<Text>,
    #[eure(default)]
    pub inputs: Vec<Stimulus>,
    #[eure(default)]
    pub steps: Vec<EvalStep>,
    #[eure(default)]
    pub participants: Vec<String>,
    #[eure(default)]
    pub allow_empty_output: bool,
    #[eure(default)]
    pub activate_allocation: Vec<ActivateAllocation>,
    #[eure(default)]
    pub memories: Vec<MemorySeed>,
    #[eure(default)]
    pub memory_links: Vec<MemoryLinkSeed>,
    #[eure(default)]
    pub policies: Vec<PolicySeed>,
    #[eure(default)]
    pub memos: Vec<MemoSeed>,
    #[eure(default)]
    pub cognition_log: Vec<CognitionLogSeed>,
    #[eure(default)]
    pub limits: EvalLimits,
    #[eure(default)]
    pub assertions: Vec<Assertion>,
    #[eure(default)]
    pub measurements: Vec<Measurement>,
    #[eure(default)]
    pub scoring: CaseScoring,
}

#[derive(Debug, Clone, FromEure)]
#[eure(crate = ::eure::document, rename_all = "kebab-case")]
pub struct ActivateAllocation {
    pub module: EvalModule,
    pub activation_ratio: f64,
}

#[derive(Debug, Clone, FromEure)]
#[eure(
    crate = ::eure::document,
    rename_all = "kebab-case",
    rename_all_fields = "kebab-case"
)]
pub enum Stimulus {
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
    AmbientSnapshot {
        #[eure(default)]
        entries: Vec<AmbientSensoryInputEntry>,
    },
}

#[derive(Debug, Clone, FromEure)]
#[eure(crate = ::eure::document, rename_all = "kebab-case")]
pub struct AmbientSensoryInputEntry {
    pub id: String,
    pub modality: String,
    pub content: Text,
}

#[derive(Debug, Clone, FromEure)]
#[eure(crate = ::eure::document, rename_all = "kebab-case")]
pub struct EvalStep {
    #[eure(default)]
    pub id: Option<String>,
    #[eure(default)]
    pub description: Option<Text>,
    #[eure(default)]
    pub terminal: bool,
    #[eure(default)]
    pub inputs: Vec<Stimulus>,
    #[eure(default)]
    pub memos: Vec<MemoSeed>,
    #[eure(default)]
    pub cognition_log: Vec<CognitionLogSeed>,
    #[eure(default)]
    pub wait_for: Option<WaitFor>,
    #[eure(default)]
    pub assertions: Vec<Assertion>,
}

#[derive(Debug, Clone, FromEure)]
#[eure(
    crate = ::eure::document,
    rename_all = "kebab-case",
    rename_all_fields = "kebab-case"
)]
pub enum WaitFor {
    MemoFrom {
        #[eure(default)]
        scope: Option<String>,
        module: EvalModule,
        timeout_ms: u64,
    },
    UtteranceFrom {
        #[eure(default)]
        scope: Option<String>,
        module: EvalModule,
        target: String,
        #[eure(default)]
        until_assertion: Option<String>,
        #[eure(default = "default_wait_max_matches")]
        max_matches: usize,
        timeout_ms: u64,
    },
    Interoception {
        timeout_ms: u64,
        #[eure(default)]
        mode: Option<EvalInteroceptiveMode>,
        #[eure(default = "default_wake_arousal_at_least")]
        wake_arousal_at_least: f64,
        #[eure(default = "default_wake_arousal_at_most")]
        wake_arousal_at_most: f64,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct EvalModule(ModuleId);

impl EvalModule {
    pub fn new(value: impl Into<String>) -> Result<Self, nuillu_types::ModuleIdParseError> {
        ModuleId::new(value).map(Self)
    }

    pub fn as_str(&self) -> &str {
        self.0.as_str()
    }

    pub fn module_id(&self) -> ModuleId {
        self.0.clone()
    }
}

impl std::str::FromStr for EvalModule {
    type Err = nuillu_types::ModuleIdParseError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        Self::new(value)
    }
}

impl eure::document::parse::FromEure<'_> for EvalModule {
    type Error = ParseError;

    fn parse(ctx: &ParseContext<'_>) -> Result<Self, Self::Error> {
        let value: String = ctx.parse()?;
        Self::new(value.clone()).map_err(|error| ParseError {
            node_id: ctx.node_id(),
            kind: ParseErrorKind::InvalidPattern {
                kind: "module-id".to_owned(),
                reason: format!("invalid module id {value:?}: {error}"),
            },
        })
    }
}

impl RuntimeCase {
    /// Inputs flattened across steps, or the legacy `inputs` list when
    /// `steps` is empty. Used for places that need to summarize the entire
    /// sensory feed of a case (e.g. judge prompt rendering).
    pub fn flat_inputs(&self) -> Vec<&Stimulus> {
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

#[derive(Debug, Clone)]
pub enum EvalCase {
    Runtime(RuntimeCase),
}

impl EvalCase {
    pub fn id(&self) -> Option<&str> {
        match self {
            Self::Runtime(case) => case.id.as_deref(),
        }
    }

    pub fn description(&self) -> Option<&Text> {
        match self {
            Self::Runtime(case) => case.description.as_ref(),
        }
    }

    pub fn memories(&self) -> &[MemorySeed] {
        match self {
            Self::Runtime(case) => &case.memories,
        }
    }

    pub fn memory_links(&self) -> &[MemoryLinkSeed] {
        match self {
            Self::Runtime(case) => &case.memory_links,
        }
    }

    pub fn policies(&self) -> &[PolicySeed] {
        match self {
            Self::Runtime(case) => &case.policies,
        }
    }

    pub fn limits(&self) -> &EvalLimits {
        match self {
            Self::Runtime(case) => &case.limits,
        }
    }

    pub fn assertions(&self) -> &[Assertion] {
        match self {
            Self::Runtime(case) => &case.assertions,
        }
    }

    pub fn measurements(&self) -> &[Measurement] {
        match self {
            Self::Runtime(case) => &case.measurements,
        }
    }

    pub fn scoring(&self) -> &CaseScoring {
        match self {
            Self::Runtime(case) => &case.scoring,
        }
    }

    pub fn prompt_for_judge(&self) -> String {
        match self {
            Self::Runtime(case) => case
                .prompt
                .as_ref()
                .map(|prompt| prompt.content.clone())
                .unwrap_or_else(|| {
                    case.flat_inputs()
                        .into_iter()
                        .map(Stimulus::as_prompt_line)
                        .collect::<Vec<_>>()
                        .join("\n")
                }),
        }
    }

    pub fn context_for_judge(&self) -> Option<String> {
        match self {
            Self::Runtime(case) => case.context.as_ref().map(|text| text.content.clone()),
        }
    }

    pub fn runtime(&self) -> &RuntimeCase {
        match self {
            Self::Runtime(case) => case,
        }
    }
}

impl Stimulus {
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
            Self::AmbientSnapshot { entries } => {
                let entries = entries
                    .iter()
                    .map(|entry| {
                        format!("{}:{}: {}", entry.id, entry.modality, entry.content.content)
                    })
                    .collect::<Vec<_>>()
                    .join("; ");
                format!("ambient-snapshot: {entries}")
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
    #[eure(default = "default_case_timeout_ms")]
    pub timeout_ms: u64,
    #[eure(default)]
    pub interoception: EvalInteroceptionLimits,
}

impl Default for EvalLimits {
    fn default() -> Self {
        Self {
            max_llm_calls: default_max_llm_calls(),
            timeout_ms: default_case_timeout_ms(),
            interoception: EvalInteroceptionLimits::default(),
        }
    }
}

#[derive(Debug, Clone, FromEure)]
#[eure(crate = ::eure::document, rename_all = "kebab-case")]
pub struct EvalInteroceptionLimits {
    #[eure(default = "default_quiet_sleep_threshold_ms")]
    pub quiet_sleep_threshold_ms: u64,
    #[eure(default = "default_arousal_change_multiplier")]
    pub wake_arousal_change_multiplier: f64,
    #[eure(default = "default_arousal_change_multiplier")]
    pub affect_arousal_change_multiplier: f64,
}

impl Default for EvalInteroceptionLimits {
    fn default() -> Self {
        Self {
            quiet_sleep_threshold_ms: default_quiet_sleep_threshold_ms(),
            wake_arousal_change_multiplier: default_arousal_change_multiplier(),
            affect_arousal_change_multiplier: default_arousal_change_multiplier(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, FromEure)]
#[eure(crate = ::eure::document, rename_all = "kebab-case")]
pub enum EvalInteroceptiveMode {
    Wake,
    NremPressure,
    RemPressure,
}

impl EvalInteroceptiveMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Wake => "wake",
            Self::NremPressure => "nrem-pressure",
            Self::RemPressure => "rem-pressure",
        }
    }
}

#[derive(Debug, Clone, FromEure)]
#[eure(crate = ::eure::document, rename_all = "kebab-case")]
pub struct MemorySeed {
    #[eure(default = "default_scope")]
    pub scope: String,
    #[eure(default)]
    pub index: Option<String>,
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
pub struct MemoryLinkSeed {
    pub from_memory: usize,
    pub to_memory: usize,
    pub relation: String,
}

#[derive(Debug, Clone, FromEure)]
#[eure(crate = ::eure::document, rename_all = "kebab-case")]
pub struct PolicySeed {
    pub index: String,
    #[eure(default = "default_policy_rank")]
    pub rank: PolicySeedRank,
    #[eure(default = "default_policy_decay_secs")]
    pub decay_secs: i64,
    pub trigger: Text,
    pub behavior: Text,
}

#[derive(Debug, Clone, FromEure)]
#[eure(crate = ::eure::document, rename_all = "kebab-case")]
pub struct MemoSeed {
    #[eure(default = "default_scope")]
    pub scope: String,
    pub module: String,
    #[eure(default = "default_memo_replica")]
    pub replica: u8,
    #[eure(default)]
    pub cognitive: bool,
    pub content: Text,
    #[eure(default = "default_seed_seconds_ago")]
    pub seconds_ago: i64,
}

#[derive(Debug, Clone, FromEure)]
#[eure(crate = ::eure::document, rename_all = "kebab-case")]
pub struct CognitionLogSeed {
    #[eure(default = "default_scope")]
    pub scope: String,
    #[eure(default = "default_cognition_module")]
    pub module: String,
    #[eure(default = "default_memo_replica")]
    pub replica: u8,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, FromEure)]
#[eure(crate = ::eure::document, rename_all = "kebab-case")]
pub enum PolicySeedRank {
    Tentative,
    Provisional,
    Established,
    Habit,
    Core,
}

impl From<PolicySeedRank> for PolicyRank {
    fn from(rank: PolicySeedRank) -> Self {
        match rank {
            PolicySeedRank::Tentative => Self::Tentative,
            PolicySeedRank::Provisional => Self::Provisional,
            PolicySeedRank::Established => Self::Established,
            PolicySeedRank::Habit => Self::Habit,
            PolicySeedRank::Core => Self::Core,
        }
    }
}

#[derive(Debug, Clone, Default, FromEure)]
#[eure(crate = ::eure::document, rename_all = "kebab-case")]
pub struct AssertionCommon {
    #[eure(default)]
    pub name: Option<String>,
    #[eure(default)]
    pub must_pass: bool,
    #[eure(default = "default_weight")]
    pub weight: i64,
}

#[derive(Debug, Clone, FromEure)]
#[eure(crate = ::eure::document, rename_all = "kebab-case")]
pub struct EventSelectorSpec {
    #[eure(default)]
    pub scopes: Vec<String>,
    #[eure(default)]
    pub origin_scopes: Vec<String>,
    #[eure(default)]
    pub modules: Vec<EvalModule>,
    #[eure(default)]
    pub variants: Vec<String>,
    #[eure(default)]
    pub steps: Vec<String>,
    #[eure(default)]
    pub replicas: Vec<u8>,
}

#[derive(Debug, Clone, FromEure)]
#[eure(
    crate = ::eure::document,
    rename_all = "kebab-case",
    rename_all_fields = "kebab-case"
)]
pub enum Measurement {
    Count {
        name: String,
        select: EventSelectorSpec,
    },
    FirstMatchLatency {
        name: String,
        select: EventSelectorSpec,
        #[eure(default)]
        group_by_scope: bool,
    },
    UniqueScopeCount {
        name: String,
        select: EventSelectorSpec,
    },
    ScopeCoverage {
        name: String,
        select: EventSelectorSpec,
        scopes: Vec<String>,
    },
    ScopeConvergenceLatency {
        name: String,
        select: EventSelectorSpec,
        scopes: Vec<String>,
    },
}

impl Measurement {
    pub fn name(&self) -> &str {
        match self {
            Self::Count { name, .. }
            | Self::FirstMatchLatency { name, .. }
            | Self::UniqueScopeCount { name, .. }
            | Self::ScopeCoverage { name, .. }
            | Self::ScopeConvergenceLatency { name, .. } => name,
        }
    }

    fn selector(&self) -> &EventSelectorSpec {
        match self {
            Self::Count { select, .. }
            | Self::FirstMatchLatency { select, .. }
            | Self::UniqueScopeCount { select, .. }
            | Self::ScopeCoverage { select, .. }
            | Self::ScopeConvergenceLatency { select, .. } => select,
        }
    }

    fn expected_scopes(&self) -> &[String] {
        match self {
            Self::ScopeCoverage { scopes, .. } | Self::ScopeConvergenceLatency { scopes, .. } => {
                scopes
            }
            _ => &[],
        }
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
    Timeline,
    Failure,
    Trace,
    Memory,
    MemoryDiff,
    MemoryMetadata,
    PolicyDiff,
    PolicyConsiderations,
    MemoContents,
    Cognition,
    CognitionEntries,
    ToolCalls,
    ToolResults,
}

#[derive(Debug, Clone, FromEure)]
#[eure(
    crate = ::eure::document,
    rename_all = "kebab-case",
    rename_all_fields = "kebab-case"
)]
pub enum Assertion {
    ArtifactTextContains {
        #[eure(flatten)]
        common: AssertionCommon,
        #[eure(default)]
        field: Option<ArtifactTextField>,
        contains: String,
    },
    ArtifactTextExact {
        #[eure(flatten)]
        common: AssertionCommon,
        #[eure(default)]
        field: Option<ArtifactTextField>,
        exact: Text,
    },
    JsonPointerEquals {
        #[eure(flatten)]
        common: AssertionCommon,
        pointer: String,
        expected: String,
    },
    JsonPointerContains {
        #[eure(flatten)]
        common: AssertionCommon,
        pointer: String,
        contains: String,
    },
    JsonPointerNumericInRange {
        #[eure(flatten)]
        common: AssertionCommon,
        pointer: String,
        #[eure(default)]
        min: Option<f64>,
        #[eure(default)]
        max: Option<f64>,
    },
    Rubric {
        #[eure(flatten)]
        common: AssertionCommon,
        rubric: Text,
        #[eure(default = "default_pass_score")]
        pass_score: f64,
        judge_inputs: Vec<RubricJudgeInput>,
        #[eure(default)]
        criteria: Vec<RubricCriterion>,
    },
    TraceSpan {
        #[eure(flatten)]
        common: AssertionCommon,
        span_name: String,
    },
    TraceEvent {
        #[eure(flatten)]
        common: AssertionCommon,
        message_contains: String,
    },
    TraceToolCall {
        #[eure(flatten)]
        common: AssertionCommon,
        tool_name: String,
        #[eure(default)]
        args_json_contains: Option<Text>,
    },
    TraceSpansOrdered {
        #[eure(flatten)]
        common: AssertionCommon,
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

impl Assertion {
    pub fn common(&self) -> &AssertionCommon {
        match self {
            Self::ArtifactTextContains { common, .. }
            | Self::ArtifactTextExact { common, .. }
            | Self::JsonPointerEquals { common, .. }
            | Self::JsonPointerContains { common, .. }
            | Self::JsonPointerNumericInRange { common, .. }
            | Self::Rubric { common, .. }
            | Self::TraceSpan { common, .. }
            | Self::TraceEvent { common, .. }
            | Self::TraceToolCall { common, .. }
            | Self::TraceSpansOrdered { common, .. } => common,
        }
    }

    pub fn kind_name(&self) -> &'static str {
        match self {
            Self::ArtifactTextContains { .. } => "artifact-text-contains",
            Self::ArtifactTextExact { .. } => "artifact-text-exact",
            Self::JsonPointerEquals { .. } => "json-pointer-equals",
            Self::JsonPointerContains { .. } => "json-pointer-contains",
            Self::JsonPointerNumericInRange { .. } => "json-pointer-numeric-in-range",
            Self::Rubric { .. } => "rubric",
            Self::TraceSpan { .. } => "trace-span",
            Self::TraceEvent { .. } => "trace-event",
            Self::TraceToolCall { .. } => "trace-tool-call",
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
    parse_runtime_case_file(path).map(EvalCase::Runtime)
}

pub fn parse_runtime_case_file(path: &Path) -> Result<RuntimeCase, CaseFileError> {
    let content = read_case(path)?;
    let mut file: RuntimeCaseFile =
        eure::parse_content(&content, path.to_path_buf()).map_err(|message| {
            CaseFileError::Parse {
                path: path.to_path_buf(),
                message,
            }
        })?;
    validate_runtime_case(path, &file.case)?;
    let config_path = path
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join(&file.case.runtime_config);
    let config_content =
        fs::read_to_string(&config_path).map_err(|source| CaseFileError::Read {
            path: config_path.clone(),
            source,
        })?;
    let boot_config =
        nuillu_server::parse_server_boot_config_content(&config_content, &config_path).map_err(
            |error| CaseFileError::Validation {
                path: path.to_path_buf(),
                message: format!(
                    "invalid runtime-config {}: {error:#}",
                    config_path.display()
                ),
            },
        )?;
    let configured_modules = boot_config
        .modules
        .iter()
        .map(nuillu_server::ServerModuleSpec::module_id)
        .chain(
            boot_config
                .expanded_subsystems()
                .into_iter()
                .flat_map(|expanded| {
                    expanded
                        .definition
                        .modules
                        .iter()
                        .map(nuillu_server::ServerModuleSpec::module_id)
                        .collect::<Vec<_>>()
                }),
        )
        .collect::<BTreeSet<_>>();
    let configured_scopes = std::iter::once("/".to_string())
        .chain(
            boot_config
                .expanded_subsystems()
                .into_iter()
                .map(|expanded| expanded.scope.to_string()),
        )
        .collect::<BTreeSet<_>>();
    for scope in file
        .case
        .memories
        .iter()
        .map(|seed| seed.scope.as_str())
        .chain(file.case.memos.iter().map(|seed| seed.scope.as_str()))
        .chain(
            file.case
                .cognition_log
                .iter()
                .map(|seed| seed.scope.as_str()),
        )
        .chain(
            file.case
                .steps
                .iter()
                .flat_map(|step| step.memos.iter().map(|seed| seed.scope.as_str())),
        )
        .chain(
            file.case
                .steps
                .iter()
                .flat_map(|step| step.cognition_log.iter().map(|seed| seed.scope.as_str())),
        )
        .chain(
            file.case
                .steps
                .iter()
                .filter_map(|step| match &step.wait_for {
                    Some(
                        WaitFor::MemoFrom {
                            scope: Some(scope), ..
                        }
                        | WaitFor::UtteranceFrom {
                            scope: Some(scope), ..
                        },
                    ) => Some(scope.as_str()),
                    _ => None,
                }),
        )
    {
        let normalized = parse_scope_id(scope)
            .expect("fixture scope validated before runtime config")
            .to_string();
        if !configured_scopes.contains(&normalized) {
            return Err(CaseFileError::Validation {
                path: path.to_path_buf(),
                message: format!(
                    "referenced scope {scope:?} is absent from runtime-config {}",
                    config_path.display()
                ),
            });
        }
    }
    for activation in &file.case.activate_allocation {
        if !configured_modules.contains(&activation.module.0) {
            return Err(CaseFileError::Validation {
                path: path.to_path_buf(),
                message: format!(
                    "activate-allocation module {:?} is not present in runtime-config {}",
                    activation.module.as_str(),
                    config_path.display()
                ),
            });
        }
    }
    for measurement in &file.case.measurements {
        let selector = measurement.selector();
        for module in &selector.modules {
            if !configured_modules.contains(&module.0) {
                return Err(CaseFileError::Validation {
                    path: path.to_path_buf(),
                    message: format!(
                        "measurement {:?} selects module {:?} absent from runtime-config",
                        measurement.name(),
                        module.as_str()
                    ),
                });
            }
        }
        for scope in selector
            .scopes
            .iter()
            .chain(&selector.origin_scopes)
            .chain(measurement.expected_scopes())
        {
            if !configured_scopes.contains(scope) {
                return Err(CaseFileError::Validation {
                    path: path.to_path_buf(),
                    message: format!(
                        "measurement {:?} selects scope {scope:?} absent from runtime-config",
                        measurement.name()
                    ),
                });
            }
        }
        for variant in &selector.variants {
            if !crate::timeline::EVENT_VARIANTS.contains(&variant.as_str()) {
                return Err(CaseFileError::Validation {
                    path: path.to_path_buf(),
                    message: format!(
                        "measurement {:?} selects unknown event variant {variant:?}",
                        measurement.name()
                    ),
                });
            }
        }
    }
    file.case.runtime_config = config_path.display().to_string();
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

fn validate_runtime_case(path: &Path, case: &RuntimeCase) -> Result<(), CaseFileError> {
    if case.runtime_config.trim().is_empty() {
        return Err(CaseFileError::Validation {
            path: path.to_path_buf(),
            message: "runtime-config must not be empty".to_string(),
        });
    }
    if !case.inputs.is_empty() && !case.steps.is_empty() {
        return Err(CaseFileError::Validation {
            path: path.to_path_buf(),
            message: "runtime case must use either `inputs` or `steps`, not both".to_string(),
        });
    }
    if case.inputs.is_empty()
        && case.steps.is_empty()
        && case.memos.is_empty()
        && case.cognition_log.is_empty()
        && case.policies.is_empty()
        && case.memories.is_empty()
    {
        return Err(CaseFileError::Validation {
            path: path.to_path_buf(),
            message: "eval case must have setup state, stimuli, or a scenario step".to_string(),
        });
    }
    for (index, input) in case.inputs.iter().enumerate() {
        validate_stimulus(path, &format!("inputs[{index}]"), input)?;
    }
    let mut step_ids = BTreeSet::new();
    for (step_index, step) in case.steps.iter().enumerate() {
        if step
            .id
            .as_deref()
            .is_some_and(|step_id| step_id.trim().is_empty())
        {
            return Err(CaseFileError::Validation {
                path: path.to_path_buf(),
                message: format!("steps[{step_index}].id must not be empty"),
            });
        }
        let effective_step_id = step
            .id
            .clone()
            .unwrap_or_else(|| format!("step-{}", step_index + 1));
        if !step_ids.insert(effective_step_id.clone()) {
            return Err(CaseFileError::Validation {
                path: path.to_path_buf(),
                message: format!("duplicate step id {effective_step_id:?}"),
            });
        }
        if step.terminal && step_index + 1 != case.steps.len() {
            return Err(CaseFileError::Validation {
                path: path.to_path_buf(),
                message: format!("steps[{step_index}]: a terminal step must be the final step"),
            });
        }
        if step.terminal && step.wait_for.is_none() {
            return Err(CaseFileError::Validation {
                path: path.to_path_buf(),
                message: format!("steps[{step_index}]: a terminal step must declare wait-for"),
            });
        }
        if step.inputs.is_empty()
            && step.memos.is_empty()
            && step.cognition_log.is_empty()
            && step.wait_for.is_none()
        {
            return Err(CaseFileError::Validation {
                path: path.to_path_buf(),
                message: format!(
                    "steps[{step_index}] must contain setup updates, stimuli, or wait-for"
                ),
            });
        }
        for (input_index, input) in step.inputs.iter().enumerate() {
            validate_stimulus(
                path,
                &format!("steps[{step_index}].inputs[{input_index}]"),
                input,
            )?;
        }
        validate_memo_seeds(path, &format!("steps[{step_index}].memos"), &step.memos)?;
        validate_cognition_log_seeds(
            path,
            &format!("steps[{step_index}].cognition-log"),
            &step.cognition_log,
        )?;
        if let Some(wait_for) = &step.wait_for {
            validate_wait_for(path, step_index, wait_for)?;
            if let WaitFor::UtteranceFrom {
                until_assertion: Some(assertion_name),
                ..
            } = wait_for
            {
                if !step.terminal {
                    return Err(CaseFileError::Validation {
                        path: path.to_path_buf(),
                        message: format!(
                            "steps[{step_index}].wait-for.until-assertion requires terminal = true"
                        ),
                    });
                }
                let matches = case
                    .assertions
                    .iter()
                    .filter(|assertion| assertion.display_name() == *assertion_name)
                    .collect::<Vec<_>>();
                if matches.len() != 1 {
                    return Err(CaseFileError::Validation {
                        path: path.to_path_buf(),
                        message: format!(
                            "steps[{step_index}].wait-for.until-assertion must name exactly one case assertion, found {} named {assertion_name:?}",
                            matches.len()
                        ),
                    });
                }
                if let Assertion::Rubric { judge_inputs, .. } = matches[0]
                    && judge_inputs.iter().any(|input| {
                        matches!(
                            input,
                            RubricJudgeInput::Trace
                                | RubricJudgeInput::ToolCalls
                                | RubricJudgeInput::ToolResults
                        )
                    })
                {
                    return Err(CaseFileError::Validation {
                        path: path.to_path_buf(),
                        message: format!(
                            "steps[{step_index}].wait-for.until-assertion cannot use trace or tool-call judge inputs during a live runtime"
                        ),
                    });
                }
            }
        }
        for check in &step.assertions {
            validate_check(path, check)?;
            if !is_step_compatible_check(check) {
                return Err(CaseFileError::Validation {
                    path: path.to_path_buf(),
                    message: format!(
                        "steps[{step_index}].assertions: {kind} cannot run mid-step (use the case-level assertions instead)",
                        kind = check.kind_name()
                    ),
                });
            }
        }
    }
    validate_cognition_log_seeds(path, "cognition-log", &case.cognition_log)?;
    validate_activate_allocation(path, case)?;
    let mut measurement_names = BTreeSet::new();
    for measurement in &case.measurements {
        let name = measurement.name();
        if name.trim().is_empty() {
            return Err(CaseFileError::Validation {
                path: path.to_path_buf(),
                message: "measurement name must not be empty".to_string(),
            });
        }
        if !measurement_names.insert(name) {
            return Err(CaseFileError::Validation {
                path: path.to_path_buf(),
                message: format!("duplicate measurement name {name:?}"),
            });
        }
        for step in &measurement.selector().steps {
            if step != "input" && !step_ids.contains(step) {
                return Err(CaseFileError::Validation {
                    path: path.to_path_buf(),
                    message: format!("measurement {name:?} selects unknown step {step:?}"),
                });
            }
        }
    }
    validate_common(
        path,
        case.now.as_deref(),
        &case.memories,
        &case.memory_links,
        &case.policies,
        &case.memos,
        &case.limits,
        &case.assertions,
    )
}

fn validate_stimulus(path: &Path, label: &str, input: &Stimulus) -> Result<(), CaseFileError> {
    match input {
        Stimulus::Heard { content, .. } if content.content.trim().is_empty() => {
            Err(CaseFileError::Validation {
                path: path.to_path_buf(),
                message: format!("{label}.content must not be empty"),
            })
        }
        Stimulus::Seen { appearance, .. } if appearance.content.trim().is_empty() => {
            Err(CaseFileError::Validation {
                path: path.to_path_buf(),
                message: format!("{label}.appearance must not be empty"),
            })
        }
        Stimulus::OneShot { modality, .. } if modality.trim().is_empty() => {
            Err(CaseFileError::Validation {
                path: path.to_path_buf(),
                message: format!("{label}.modality must not be empty"),
            })
        }
        Stimulus::OneShot { content, .. } if content.content.trim().is_empty() => {
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
    if let WaitFor::MemoFrom {
        scope: Some(scope), ..
    }
    | WaitFor::UtteranceFrom {
        scope: Some(scope), ..
    } = wait_for
    {
        parse_scope_id(scope).map_err(|error| CaseFileError::Validation {
            path: path.to_path_buf(),
            message: format!("steps[{step_index}].wait-for.scope is invalid: {error}"),
        })?;
    }
    match wait_for {
        WaitFor::MemoFrom { timeout_ms, .. }
        | WaitFor::UtteranceFrom { timeout_ms, .. }
        | WaitFor::Interoception { timeout_ms, .. } => {
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
    if let WaitFor::UtteranceFrom {
        target,
        until_assertion,
        max_matches,
        ..
    } = wait_for
    {
        if target.trim().is_empty() {
            return Err(CaseFileError::Validation {
                path: path.to_path_buf(),
                message: format!("steps[{step_index}].wait-for.target must not be empty"),
            });
        }
        if *max_matches == 0 {
            return Err(CaseFileError::Validation {
                path: path.to_path_buf(),
                message: format!(
                    "steps[{step_index}].wait-for.max-matches must be greater than zero"
                ),
            });
        }
        if until_assertion.is_none() && *max_matches != 1 {
            return Err(CaseFileError::Validation {
                path: path.to_path_buf(),
                message: format!(
                    "steps[{step_index}].wait-for.max-matches requires until-assertion"
                ),
            });
        }
    }
    if let WaitFor::Interoception {
        mode,
        wake_arousal_at_least,
        wake_arousal_at_most,
        ..
    } = wait_for
        && mode.is_none()
        && !wake_arousal_min_is_set(*wake_arousal_at_least)
        && !wake_arousal_max_is_set(*wake_arousal_at_most)
    {
        return Err(CaseFileError::Validation {
            path: path.to_path_buf(),
            message: format!(
                "steps[{step_index}].wait-for.interoception must set at least one condition"
            ),
        });
    }
    Ok(())
}

pub(crate) fn wake_arousal_min_is_set(value: f64) -> bool {
    value >= 0.0
}

pub(crate) fn wake_arousal_max_is_set(value: f64) -> bool {
    value <= 1.0
}

/// Mid-step assertions can only consult agent observations (JSON pointers / text in
/// the running artifact). Trace and rubric assertions require a completed run.
fn is_step_compatible_check(check: &Assertion) -> bool {
    matches!(
        check,
        Assertion::JsonPointerEquals { .. }
            | Assertion::JsonPointerContains { .. }
            | Assertion::JsonPointerNumericInRange { .. }
            | Assertion::ArtifactTextContains { .. }
            | Assertion::ArtifactTextExact { .. }
    )
}

fn validate_cognition_log_seeds(
    path: &Path,
    label: &str,
    seeds: &[CognitionLogSeed],
) -> Result<(), CaseFileError> {
    for (index, seed) in seeds.iter().enumerate() {
        validate_scope_path(path, &format!("{label}[{index}].scope"), &seed.scope)?;
        ModuleId::new(seed.module.clone()).map_err(|error| CaseFileError::Validation {
            path: path.to_path_buf(),
            message: format!("{label}[{index}].module is invalid: {error}"),
        })?;
        if seed.text.content.trim().is_empty() {
            return Err(CaseFileError::Validation {
                path: path.to_path_buf(),
                message: format!("{label}[{index}].text must not be empty"),
            });
        }
        if seed.seconds_ago < 0 {
            return Err(CaseFileError::Validation {
                path: path.to_path_buf(),
                message: format!("{label}[{index}].seconds-ago must not be negative"),
            });
        }
    }
    Ok(())
}

fn validate_activate_allocation(path: &Path, case: &RuntimeCase) -> Result<(), CaseFileError> {
    let mut seen = BTreeSet::new();
    for (index, activation) in case.activate_allocation.iter().enumerate() {
        if !seen.insert(activation.module.clone()) {
            return Err(CaseFileError::Validation {
                path: path.to_path_buf(),
                message: format!(
                    "activate-allocation contains duplicate module '{}'",
                    activation.module.as_str()
                ),
            });
        }
        if !activation.activation_ratio.is_finite()
            || !(0.0..=1.0).contains(&activation.activation_ratio)
        {
            return Err(CaseFileError::Validation {
                path: path.to_path_buf(),
                message: format!(
                    "activate-allocation[{index}].activation-ratio must be between 0.0 and 1.0"
                ),
            });
        }
    }
    Ok(())
}

fn validate_common(
    path: &Path,
    now: Option<&str>,
    memories: &[MemorySeed],
    memory_links: &[MemoryLinkSeed],
    policies: &[PolicySeed],
    memos: &[MemoSeed],
    limits: &EvalLimits,
    assertions: &[Assertion],
) -> Result<(), CaseFileError> {
    if matches!(limits.max_llm_calls, Some(0)) {
        return Err(CaseFileError::Validation {
            path: path.to_path_buf(),
            message: "limits.max-llm-calls must be greater than zero when present".to_string(),
        });
    }
    if limits.timeout_ms == 0 {
        return Err(CaseFileError::Validation {
            path: path.to_path_buf(),
            message: "limits.timeout-ms must be greater than zero".to_string(),
        });
    }
    let case_now = parse_case_now(now).map_err(|message| CaseFileError::Validation {
        path: path.to_path_buf(),
        message,
    })?;

    for (index, memory) in memories.iter().enumerate() {
        validate_scope_path(path, &format!("memories[{index}].scope"), &memory.scope)?;
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
        if let Some(explicit_index) = &memory.index
            && explicit_index.trim().is_empty()
        {
            return Err(CaseFileError::Validation {
                path: path.to_path_buf(),
                message: format!("memories[{index}].index must not be empty"),
            });
        }
    }

    for (index, link) in memory_links.iter().enumerate() {
        if link.from_memory >= memories.len() {
            return Err(CaseFileError::Validation {
                path: path.to_path_buf(),
                message: format!(
                    "memory-links[{index}].from-memory {} is out of range for {} seeded memories",
                    link.from_memory,
                    memories.len()
                ),
            });
        }
        if link.to_memory >= memories.len() {
            return Err(CaseFileError::Validation {
                path: path.to_path_buf(),
                message: format!(
                    "memory-links[{index}].to-memory {} is out of range for {} seeded memories",
                    link.to_memory,
                    memories.len()
                ),
            });
        }
        if !is_valid_memory_link_relation(&link.relation) {
            return Err(CaseFileError::Validation {
                path: path.to_path_buf(),
                message: format!(
                    "memory-links[{index}].relation must be one of related, supports, contradicts, updates, corrects, derived_from"
                ),
            });
        }
    }

    for (index, policy) in policies.iter().enumerate() {
        if policy.index.trim().is_empty() {
            return Err(CaseFileError::Validation {
                path: path.to_path_buf(),
                message: format!("policies[{index}].index must not be empty"),
            });
        }
        if policy.trigger.content.trim().is_empty() {
            return Err(CaseFileError::Validation {
                path: path.to_path_buf(),
                message: format!("policies[{index}].trigger must not be empty"),
            });
        }
        if policy.behavior.content.trim().is_empty() {
            return Err(CaseFileError::Validation {
                path: path.to_path_buf(),
                message: format!("policies[{index}].behavior must not be empty"),
            });
        }
        if policy.decay_secs < 0 {
            return Err(CaseFileError::Validation {
                path: path.to_path_buf(),
                message: format!("policies[{index}].decay-secs must not be negative"),
            });
        }
    }

    validate_memo_seeds(path, "memos", memos)?;

    for check in assertions {
        validate_check(path, check)?;
    }

    Ok(())
}

fn validate_memo_seeds(path: &Path, label: &str, memos: &[MemoSeed]) -> Result<(), CaseFileError> {
    for (index, memo) in memos.iter().enumerate() {
        validate_scope_path(path, &format!("{label}[{index}].scope"), &memo.scope)?;
        if memo.module.trim().is_empty() {
            return Err(CaseFileError::Validation {
                path: path.to_path_buf(),
                message: format!("{label}[{index}].module must not be empty"),
            });
        }
        if let Err(error) = ModuleId::new(memo.module.clone()) {
            return Err(CaseFileError::Validation {
                path: path.to_path_buf(),
                message: format!("{label}[{index}].module is invalid: {error}"),
            });
        }
        if memo.content.content.trim().is_empty() {
            return Err(CaseFileError::Validation {
                path: path.to_path_buf(),
                message: format!("{label}[{index}].content must not be empty"),
            });
        }
        if memo.seconds_ago < 0 {
            return Err(CaseFileError::Validation {
                path: path.to_path_buf(),
                message: format!("{label}[{index}].seconds-ago must not be negative"),
            });
        }
    }
    Ok(())
}

fn validate_scope_path(path: &Path, label: &str, scope: &str) -> Result<(), CaseFileError> {
    parse_scope_id(scope).map_err(|error| CaseFileError::Validation {
        path: path.to_path_buf(),
        message: format!("{label} is invalid: {error}"),
    })?;
    Ok(())
}

pub(crate) fn parse_scope_id(value: &str) -> anyhow::Result<ScopeId> {
    if value == "/" {
        return Ok(ScopeId::root());
    }
    anyhow::ensure!(value.starts_with('/'), "scope must start with '/'");
    let mut scope = ScopeId::root();
    for component in value[1..].split('/') {
        let (subsystem, replica) = component
            .strip_suffix(']')
            .and_then(|component| component.rsplit_once('['))
            .ok_or_else(|| {
                anyhow::anyhow!("scope component {component:?} must be name[replica]")
            })?;
        let subsystem = SubsystemId::new(subsystem.to_string())
            .map_err(|error| anyhow::anyhow!("invalid subsystem in {component:?}: {error}"))?;
        let replica = replica
            .parse::<u8>()
            .map_err(|error| anyhow::anyhow!("invalid replica in {component:?}: {error}"))?;
        scope = scope.child(SubsystemInstanceId::new(
            subsystem,
            ReplicaIndex::new(replica),
        ));
    }
    Ok(scope)
}

fn is_valid_memory_link_relation(value: &str) -> bool {
    matches!(
        value.trim().to_ascii_lowercase().as_str(),
        "related"
            | "supports"
            | "contradicts"
            | "updates"
            | "corrects"
            | "derived_from"
            | "derived-from"
    )
}

fn validate_check(path: &Path, check: &Assertion) -> Result<(), CaseFileError> {
    if let Assertion::JsonPointerEquals { pointer, .. }
    | Assertion::JsonPointerContains { pointer, .. }
    | Assertion::JsonPointerNumericInRange { pointer, .. } = check
    {
        validate_json_pointer(path, pointer)?;
    }

    if let Assertion::JsonPointerNumericInRange { min, max, .. } = check {
        if min.is_none() && max.is_none() {
            return Err(CaseFileError::Validation {
                path: path.to_path_buf(),
                message: format!(
                    "json-pointer-numeric-in-range check '{}' must set at least one of min or max",
                    check.display_name()
                ),
            });
        }
        if let (Some(min), Some(max)) = (min, max)
            && min > max
        {
            return Err(CaseFileError::Validation {
                path: path.to_path_buf(),
                message: format!(
                    "json-pointer-numeric-in-range check '{}' has min ({min}) greater than max ({max})",
                    check.display_name()
                ),
            });
        }
    }

    if let Assertion::Rubric {
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
        Assertion::TraceSpan { span_name, .. } if span_name.trim().is_empty() => {
            return Err(CaseFileError::Validation {
                path: path.to_path_buf(),
                message: format!(
                    "trace check '{}' has an empty span name",
                    check.display_name()
                ),
            });
        }
        Assertion::TraceToolCall { tool_name, .. } if tool_name.trim().is_empty() => {
            return Err(CaseFileError::Validation {
                path: path.to_path_buf(),
                message: format!(
                    "trace-tool-call check '{}' has an empty tool name",
                    check.display_name()
                ),
            });
        }
        Assertion::TraceToolCall {
            args_json_contains: Some(args_json_contains),
            ..
        } => {
            if let Err(error) =
                serde_json::from_str::<serde_json::Value>(&args_json_contains.content)
            {
                return Err(CaseFileError::Validation {
                    path: path.to_path_buf(),
                    message: format!(
                        "trace-tool-call check '{}' has invalid args-json-contains JSON: {error}",
                        check.display_name()
                    ),
                });
            }
        }
        Assertion::TraceSpansOrdered { names, .. } if names.is_empty() => {
            return Err(CaseFileError::Validation {
                path: path.to_path_buf(),
                message: format!(
                    "trace check '{}' must list at least one span name",
                    check.display_name()
                ),
            });
        }
        Assertion::TraceSpansOrdered { names, .. }
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
    if criteria.is_empty() {
        return Err(CaseFileError::Validation {
            path: path.to_path_buf(),
            message: format!("{label} has no criteria"),
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
    use super::{parse_case_file, parse_case_now, parse_memory_datetime};

    const RUNTIME_CONFIG: &str = r#"
activation-table = [1.0, 0.0]

@ modules[] {
  id: sensory
  replica-min = 1
  replica-max = 1
  replica-capacity = 1
  bpm-min = 10.0
  bpm-max = 10.0
  initial-activation = 1.0
}
"#;

    /// Writes a self-contained case plus its runtime config and returns the
    /// validation error the parser rejects it with.
    fn rejection(case_body: &str) -> String {
        let dir = tempfile::tempdir().expect("temp dir");
        std::fs::write(dir.path().join("runtime.eure"), RUNTIME_CONFIG).expect("write config");
        let case_path = dir.path().join("case.eure");
        std::fs::write(
            &case_path,
            format!("id: invalid-case\nruntime-config: ./runtime.eure\n{case_body}"),
        )
        .expect("write case");

        parse_case_file(&case_path)
            .expect_err("case must be rejected")
            .to_string()
    }

    #[test]
    fn rejects_negative_cognition_log_seed_seconds_ago() {
        let error = rejection(
            r#"
@ cognition-log[] {
  text = "Current attended item"
  seconds-ago = -1
}
"#,
        );

        assert!(
            error.contains("seconds-ago must not be negative"),
            "{error}"
        );
    }

    #[test]
    fn rejects_empty_rubric_judge_inputs() {
        let error = rejection(
            r#"
@ inputs[] {
  $variant: heard
  direction: Peer
  content: Hello
}

@ assertions[] {
  $variant: rubric
  name: bad-rubric
  judge-inputs = []
  rubric = "Judge the output."

  @ criteria[] {
    name: some-criterion
    description: A criterion.
  }
}
"#,
        );

        assert!(error.contains("has empty judge-inputs"), "{error}");
    }

    #[test]
    fn rejects_rubric_without_criteria() {
        let error = rejection(
            r#"
@ inputs[] {
  $variant: heard
  direction: Peer
  content: Hello
}

@ assertions[] {
  $variant: rubric
  name: holistic
  judge-inputs = ["output"]
  rubric = "Judge holistically."
}
"#,
        );

        assert!(error.contains("has no criteria"), "{error}");
    }

    #[test]
    fn rejects_duplicate_activate_allocation_modules() {
        let error = rejection(
            r#"
@ inputs[] {
  $variant: heard
  direction: Peer
  content: Hello
}

@ activate-allocation[] {
  module: sensory
  activation-ratio = 1.0
}

@ activate-allocation[] {
  module: sensory
  activation-ratio = 0.5
}
"#,
        );

        assert!(
            error.contains("activate-allocation contains duplicate module 'sensory'"),
            "{error}"
        );
    }

    #[test]
    fn rejects_activation_ratio_outside_unit_range() {
        let error = rejection(
            r#"
@ inputs[] {
  $variant: heard
  direction: Peer
  content: Hello
}

@ activate-allocation[] {
  module: sensory
  activation-ratio = 1.2
}
"#,
        );

        assert!(
            error.contains("activation-ratio must be between"),
            "{error}"
        );
    }

    #[test]
    fn rejects_both_inputs_and_steps() {
        let error = rejection(
            r#"
@ inputs[] {
  $variant: heard
  direction: Peer
  content: Hello
}

@ steps[] {
  description: A step.

  @ inputs[] {
    $variant: heard
    direction: Peer
    content: Hello again
  }
}
"#,
        );

        assert!(
            error.contains("must use either `inputs` or `steps`, not both"),
            "{error}"
        );
    }

    #[test]
    fn rejects_zero_wait_timeout() {
        let error = rejection(
            r#"
@ steps[] {
  description: A step that never waits.
  wait-for.$variant: some.memo-from
  wait-for.module: sensory
  wait-for.timeout-ms = 0

  @ inputs[] {
    $variant: heard
    direction: Peer
    content: Hello
  }
}
"#,
        );

        assert!(
            error.contains("timeout-ms must be greater than zero"),
            "{error}"
        );
    }

    #[test]
    fn rejects_interoception_wait_without_any_condition() {
        let error = rejection(
            r#"
@ steps[] {
  description: Wait on interoception without naming a condition.
  wait-for.$variant: some.interoception
  wait-for.timeout-ms = 5000
}
"#,
        );

        assert!(
            error.contains("wait-for.interoception must set at least one condition"),
            "{error}"
        );
    }

    #[test]
    fn rejects_trace_assertion_inside_a_step() {
        let error = rejection(
            r#"
@ steps[] {
  description: A step with a trace assertion.

  @ inputs[] {
    $variant: heard
    direction: Peer
    content: Hello
  }

  @ assertions[] {
    $variant: trace-span
    name: mid-step-trace
    span-name: llm_turn
  }
}
"#,
        );

        assert!(error.contains("cannot run mid-step"), "{error}");
    }

    #[test]
    fn rejects_invalid_trace_tool_call_args_json_contains() {
        let error = rejection(
            r#"
@ inputs[] {
  $variant: heard
  direction: Peer
  content: Hello
}

@ assertions[] {
  $variant: trace-tool-call
  name: bad-tool-args
  tool-name: write_retrieval_memo
  args-json-contains = `{not-json`
}
"#,
        );

        assert!(error.contains("invalid args-json-contains JSON"), "{error}");
    }

    #[test]
    fn rejects_memory_seed_with_both_datetime_and_seconds_ago() {
        let error = rejection(
            r#"
@ memories[] {
  rank: permanent
  datetime: 2025-05-10
  seconds-ago = 60
  content: Koro walks calmly to his bowl.
}
"#,
        );

        assert!(
            error.contains("must not specify both datetime and seconds-ago"),
            "{error}"
        );
    }

    #[test]
    fn date_only_memory_datetime_uses_case_now_offset() {
        let case_now = parse_case_now(Some("2026-05-10T08:21:00+09:00"))
            .expect("parse case now")
            .expect("case now is present");

        let occurred_at =
            parse_memory_datetime("2025-05-10", Some(case_now)).expect("parse date-only datetime");

        assert_eq!(occurred_at.to_rfc3339(), "2025-05-09T15:00:00+00:00");
    }

    #[test]
    fn memory_datetime_accepts_rfc3339() {
        let occurred_at =
            parse_memory_datetime("2025-05-10T08:21:00+09:00", None).expect("parse rfc3339");

        assert_eq!(occurred_at.to_rfc3339(), "2025-05-09T23:21:00+00:00");
    }
}
