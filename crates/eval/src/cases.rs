use std::{
    collections::BTreeSet,
    fs, io,
    path::{Path, PathBuf},
};

use chrono::{DateTime, Datelike as _, FixedOffset, NaiveDate, TimeZone as _, Utc};
use eure::{FromEure, value::Text};
use nuillu_types::{MemoryRank, ModuleId, PolicyRank, builtin};
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

fn default_pass_score() -> f64 {
    0.8
}

fn default_judge_max_output_tokens() -> u32 {
    1200
}

fn default_max_llm_calls() -> Option<u64> {
    Some(10)
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
    pub allow_empty_output: bool,
    #[eure(default)]
    pub activate_allocation: Vec<ActivateAllocation>,
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
    pub description: Option<Text>,
    #[eure(default)]
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
    MemoFrom {
        module: EvalModule,
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
    pub memory_links: Vec<MemoryLinkSeed>,
    #[eure(default)]
    pub policies: Vec<PolicySeed>,
    #[eure(default)]
    pub memos: Vec<MemoSeed>,
    #[eure(default)]
    pub cognition_log: Vec<CognitionLogSeed>,
    #[eure(default)]
    pub inputs: Vec<FullAgentInput>,
    #[eure(default)]
    pub steps: Vec<ModuleEvalStep>,
    #[eure(default)]
    pub limits: EvalLimits,
    #[eure(default)]
    pub checks: Vec<Check>,
    #[eure(default)]
    pub scoring: CaseScoring,
}

#[derive(Debug, Clone, FromEure)]
#[eure(crate = ::eure::document, rename_all = "kebab-case")]
pub struct ModuleEvalStep {
    #[eure(default)]
    pub description: Option<Text>,
    #[eure(default)]
    pub memos: Vec<MemoSeed>,
    #[eure(default)]
    pub cognition_log: Vec<CognitionLogSeed>,
    #[eure(default)]
    pub inputs: Vec<FullAgentInput>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, clap::ValueEnum, FromEure)]
#[eure(crate = ::eure::document, rename_all = "kebab-case")]
pub enum EvalModule {
    Sensory,
    CognitionGate,
    Allocation,
    Action,
    AttentionSchema,
    Interpreter,
    SelfModel,
    QueryMemory,
    Memory,
    MemoryCompaction,
    MemoryAssociation,
    Dreaming,
    Interoception,
    Homeostasis,
    Policy,
    PolicyCompaction,
    Reward,
    Predict,
    Surprise,
    Speak,
    Sleep,
    Poet,
}

pub const DEFAULT_FULL_AGENT_MODULES: &[EvalModule] = &[
    EvalModule::Sensory,
    EvalModule::CognitionGate,
    EvalModule::Allocation,
    EvalModule::Action,
    EvalModule::AttentionSchema,
    EvalModule::Interpreter,
    EvalModule::SelfModel,
    EvalModule::QueryMemory,
    EvalModule::Memory,
    EvalModule::MemoryCompaction,
    EvalModule::MemoryAssociation,
    EvalModule::Dreaming,
    EvalModule::Interoception,
    EvalModule::Homeostasis,
    EvalModule::Policy,
    EvalModule::PolicyCompaction,
    EvalModule::Reward,
    EvalModule::Predict,
    EvalModule::Surprise,
    EvalModule::Speak,
    EvalModule::Sleep,
    EvalModule::Poet,
];

impl EvalModule {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Sensory => "sensory",
            Self::CognitionGate => "cognition-gate",
            Self::Allocation => "allocation",
            Self::Action => "action",
            Self::AttentionSchema => "attention-schema",
            Self::Interpreter => "interpreter",
            Self::SelfModel => "self-model",
            Self::QueryMemory => "query-memory",
            Self::Memory => "memory",
            Self::MemoryCompaction => "memory-compaction",
            Self::MemoryAssociation => "memory-association",
            Self::Dreaming => "dreaming",
            Self::Interoception => "interoception",
            Self::Homeostasis => "homeostasis",
            Self::Policy => "policy",
            Self::PolicyCompaction => "policy-compaction",
            Self::Reward => "reward",
            Self::Predict => "predict",
            Self::Surprise => "surprise",
            Self::Speak => "speak",
            Self::Sleep => "sleep",
            Self::Poet => "poet",
        }
    }

    pub fn module_id(self) -> ModuleId {
        match self {
            Self::Sensory => builtin::sensory(),
            Self::CognitionGate => builtin::cognition_gate(),
            Self::Allocation => builtin::allocation(),
            Self::Action => builtin::action(),
            Self::AttentionSchema => builtin::attention_schema(),
            Self::Interpreter => builtin::interpreter(),
            Self::SelfModel => builtin::self_model(),
            Self::QueryMemory => builtin::query_memory(),
            Self::Memory => builtin::memory(),
            Self::MemoryCompaction => builtin::memory_compaction(),
            Self::MemoryAssociation => builtin::memory_association(),
            Self::Dreaming => builtin::dreaming(),
            Self::Interoception => builtin::interoception(),
            Self::Homeostasis => builtin::homeostasis(),
            Self::Policy => builtin::policy(),
            Self::PolicyCompaction => builtin::policy_compaction(),
            Self::Reward => builtin::reward(),
            Self::Predict => builtin::predict(),
            Self::Surprise => builtin::surprise(),
            Self::Speak => builtin::speak(),
            Self::Sleep => builtin::sleep(),
            Self::Poet => builtin::poet(),
        }
    }

    pub fn is_action_module(self) -> bool {
        matches!(self, Self::Speak)
    }

    pub fn is_action_target(self) -> bool {
        matches!(self, Self::Speak | Self::Sleep | Self::Poet)
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
    Sensory,
    CognitionGate,
    Action,
    QueryMemory,
    AttentionSchema,
    Interpreter,
    SelfModel,
    Memory,
    MemoryCompaction,
    MemoryAssociation,
    Dreaming,
    Policy,
    PolicyCompaction,
    Allocation,
    Predict,
    Surprise,
    Speak,
    Sleep,
    Poet,
}

impl ModuleEvalTarget {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Sensory => "sensory",
            Self::CognitionGate => "cognition-gate",
            Self::Action => "action",
            Self::QueryMemory => "query-memory",
            Self::AttentionSchema => "attention-schema",
            Self::Interpreter => "interpreter",
            Self::SelfModel => "self-model",
            Self::Memory => "memory",
            Self::MemoryCompaction => "memory-compaction",
            Self::MemoryAssociation => "memory-association",
            Self::Dreaming => "dreaming",
            Self::Policy => "policy",
            Self::PolicyCompaction => "policy-compaction",
            Self::Allocation => "allocation",
            Self::Predict => "predict",
            Self::Surprise => "surprise",
            Self::Speak => "speak",
            Self::Sleep => "sleep",
            Self::Poet => "poet",
        }
    }

    pub fn module(self) -> EvalModule {
        match self {
            Self::Sensory => EvalModule::Sensory,
            Self::CognitionGate => EvalModule::CognitionGate,
            Self::Action => EvalModule::Action,
            Self::QueryMemory => EvalModule::QueryMemory,
            Self::AttentionSchema => EvalModule::AttentionSchema,
            Self::Interpreter => EvalModule::Interpreter,
            Self::SelfModel => EvalModule::SelfModel,
            Self::Memory => EvalModule::Memory,
            Self::MemoryCompaction => EvalModule::MemoryCompaction,
            Self::MemoryAssociation => EvalModule::MemoryAssociation,
            Self::Dreaming => EvalModule::Dreaming,
            Self::Policy => EvalModule::Policy,
            Self::PolicyCompaction => EvalModule::PolicyCompaction,
            Self::Allocation => EvalModule::Allocation,
            Self::Predict => EvalModule::Predict,
            Self::Surprise => EvalModule::Surprise,
            Self::Speak => EvalModule::Speak,
            Self::Sleep => EvalModule::Sleep,
            Self::Poet => EvalModule::Poet,
        }
    }

    fn from_path(path: &Path) -> Option<Self> {
        path.components()
            .filter_map(|component| component.as_os_str().to_str())
            .find_map(|part| match part {
                "sensory" => Some(Self::Sensory),
                "cognition-gate" => Some(Self::CognitionGate),
                "action" => Some(Self::Action),
                "query-memory" => Some(Self::QueryMemory),
                "attention-schema" => Some(Self::AttentionSchema),
                "interpreter" => Some(Self::Interpreter),
                "self-model" => Some(Self::SelfModel),
                "memory" => Some(Self::Memory),
                "memory-compaction" => Some(Self::MemoryCompaction),
                "memory-association" => Some(Self::MemoryAssociation),
                "dreaming" => Some(Self::Dreaming),
                "policy" => Some(Self::Policy),
                "policy-compaction" => Some(Self::PolicyCompaction),
                "allocation" => Some(Self::Allocation),
                "predict" => Some(Self::Predict),
                "surprise" => Some(Self::Surprise),
                "speak" => Some(Self::Speak),
                "sleep" => Some(Self::Sleep),
                "poet" => Some(Self::Poet),
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

    pub fn memory_links(&self) -> &[MemoryLinkSeed] {
        match self {
            Self::FullAgent(_) => &[],
            Self::Module { case, .. } => &case.memory_links,
        }
    }

    pub fn policies(&self) -> &[PolicySeed] {
        match self {
            Self::FullAgent(_) => &[],
            Self::Module { case, .. } => &case.policies,
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
    #[eure(default)]
    pub interoception: EvalInteroceptionLimits,
}

impl Default for EvalLimits {
    fn default() -> Self {
        Self {
            max_llm_calls: default_max_llm_calls(),
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
    JsonPointerNumericInRange {
        #[eure(flatten)]
        common: CheckCommon,
        pointer: String,
        #[eure(default)]
        min: Option<f64>,
        #[eure(default)]
        max: Option<f64>,
    },
    Rubric {
        #[eure(flatten)]
        common: CheckCommon,
        rubric: Text,
        #[eure(default = "default_pass_score")]
        pass_score: f64,
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
    TraceToolCall {
        #[eure(flatten)]
        common: CheckCommon,
        tool_name: String,
        #[eure(default)]
        args_json_contains: Option<Text>,
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
        if step.inputs.is_empty() && step.wait_for.is_none() {
            return Err(CaseFileError::Validation {
                path: path.to_path_buf(),
                message: format!(
                    "steps[{step_index}].inputs must not be empty unless wait-for is set"
                ),
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
    validate_activate_allocation(path, case)?;
    validate_module_checks(path, case)?;
    validate_common(
        path,
        case.now.as_deref(),
        &case.memories,
        &[],
        &[],
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
        WaitFor::MemoFrom { timeout_ms, .. } | WaitFor::Interoception { timeout_ms, .. } => {
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

/// Mid-step checks can only consult agent observations (JSON pointers / text in
/// the running artifact). Trace and rubric checks require a completed run.
fn is_step_compatible_check(check: &Check) -> bool {
    matches!(
        check,
        Check::JsonPointerEquals { .. }
            | Check::JsonPointerContains { .. }
            | Check::JsonPointerNumericInRange { .. }
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
    if !case.steps.is_empty()
        && (!case.memos.is_empty() || !case.cognition_log.is_empty() || !case.inputs.is_empty())
    {
        return Err(CaseFileError::Validation {
            path: path.to_path_buf(),
            message:
                "module case must use either top-level memos/cognition-log/inputs or `steps`, not both"
                    .to_string(),
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
    validate_cognition_log_seeds(path, "cognition-log", &case.cognition_log)?;
    for (index, input) in case.inputs.iter().enumerate() {
        validate_full_agent_input(path, &format!("inputs[{index}]"), input)?;
    }
    for (step_index, step) in case.steps.iter().enumerate() {
        validate_memo_seeds(path, &format!("steps[{step_index}].memos"), &step.memos)?;
        validate_cognition_log_seeds(
            path,
            &format!("steps[{step_index}].cognition-log"),
            &step.cognition_log,
        )?;
        for (input_index, input) in step.inputs.iter().enumerate() {
            validate_full_agent_input(
                path,
                &format!("steps[{step_index}].inputs[{input_index}]"),
                input,
            )?;
        }
    }
    validate_modules(path, case.modules.as_deref())?;
    validate_common(
        path,
        case.now.as_deref(),
        &case.memories,
        &case.memory_links,
        &case.policies,
        &case.memos,
        &case.limits,
        &case.checks,
    )
}

fn validate_cognition_log_seeds(
    path: &Path,
    label: &str,
    seeds: &[CognitionLogSeed],
) -> Result<(), CaseFileError> {
    for (index, seed) in seeds.iter().enumerate() {
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

fn validate_module_case_target(
    path: &Path,
    target: ModuleEvalTarget,
    case: &ModuleCase,
) -> Result<(), CaseFileError> {
    let step_inputs_count = case
        .steps
        .iter()
        .map(|step| step.inputs.len())
        .sum::<usize>();
    let step_memos_count = case
        .steps
        .iter()
        .map(|step| step.memos.len())
        .sum::<usize>();
    let step_cognition_count = case
        .steps
        .iter()
        .map(|step| step.cognition_log.len())
        .sum::<usize>();

    if target == ModuleEvalTarget::Sensory && case.inputs.is_empty() && step_inputs_count == 0 {
        return Err(CaseFileError::Validation {
            path: path.to_path_buf(),
            message: "sensory module case must include at least one input".to_string(),
        });
    }
    if target == ModuleEvalTarget::Surprise {
        if case.cognition_log.is_empty() && step_cognition_count == 0 {
            return Err(CaseFileError::Validation {
                path: path.to_path_buf(),
                message: "surprise module case must include at least one cognition-log seed"
                    .to_string(),
            });
        }
        if !case
            .memos
            .iter()
            .any(|memo| memo.module.as_str() == "predict")
            && !case.steps.iter().any(|step| {
                step.memos
                    .iter()
                    .any(|memo| memo.module.as_str() == "predict")
            })
        {
            return Err(CaseFileError::Validation {
                path: path.to_path_buf(),
                message: "surprise module case must include at least one predict memo seed"
                    .to_string(),
            });
        }
    }
    if matches!(
        target,
        ModuleEvalTarget::Interpreter | ModuleEvalTarget::Predict
    ) && case.cognition_log.is_empty()
        && step_cognition_count == 0
    {
        return Err(CaseFileError::Validation {
            path: path.to_path_buf(),
            message: format!(
                "{} module case must include at least one cognition-log seed",
                target.as_str()
            ),
        });
    }
    if target == ModuleEvalTarget::Allocation && case.memos.is_empty() && step_memos_count == 0 {
        return Err(CaseFileError::Validation {
            path: path.to_path_buf(),
            message: "allocation module case must include at least one memo seed".to_string(),
        });
    }
    if target == ModuleEvalTarget::Policy
        && case.memos.is_empty()
        && step_memos_count == 0
        && case.cognition_log.is_empty()
        && step_cognition_count == 0
    {
        return Err(CaseFileError::Validation {
            path: path.to_path_buf(),
            message: "policy module case must include at least one memo or cognition-log seed"
                .to_string(),
        });
    }
    if target == ModuleEvalTarget::PolicyCompaction && case.policies.is_empty() {
        return Err(CaseFileError::Validation {
            path: path.to_path_buf(),
            message: "policy-compaction module case must include at least one policy seed"
                .to_string(),
        });
    }
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

fn validate_activate_allocation(path: &Path, case: &FullAgentCase) -> Result<(), CaseFileError> {
    let modules = case.effective_modules();
    let mut seen = BTreeSet::new();
    for (index, activation) in case.activate_allocation.iter().enumerate() {
        if !modules.contains(&activation.module) {
            return Err(CaseFileError::Validation {
                path: path.to_path_buf(),
                message: format!(
                    "activate-allocation[{index}].module '{}' must be included in full-agent modules",
                    activation.module.as_str()
                ),
            });
        }
        if !seen.insert(activation.module) {
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
    memory_links: &[MemoryLinkSeed],
    policies: &[PolicySeed],
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
        if let Some(explicit_index) = &memory.index {
            if explicit_index.trim().is_empty() {
                return Err(CaseFileError::Validation {
                    path: path.to_path_buf(),
                    message: format!("memories[{index}].index must not be empty"),
                });
            }
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

    for check in checks {
        validate_check(path, check)?;
    }

    Ok(())
}

fn validate_memo_seeds(path: &Path, label: &str, memos: &[MemoSeed]) -> Result<(), CaseFileError> {
    for (index, memo) in memos.iter().enumerate() {
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

fn validate_check(path: &Path, check: &Check) -> Result<(), CaseFileError> {
    if let Check::JsonPointerEquals { pointer, .. }
    | Check::JsonPointerContains { pointer, .. }
    | Check::JsonPointerNumericInRange { pointer, .. } = check
    {
        validate_json_pointer(path, pointer)?;
    }

    if let Check::JsonPointerNumericInRange { min, max, .. } = check {
        if min.is_none() && max.is_none() {
            return Err(CaseFileError::Validation {
                path: path.to_path_buf(),
                message: format!(
                    "json-pointer-numeric-in-range check '{}' must set at least one of min or max",
                    check.display_name()
                ),
            });
        }
        if let (Some(min), Some(max)) = (min, max) {
            if min > max {
                return Err(CaseFileError::Validation {
                    path: path.to_path_buf(),
                    message: format!(
                        "json-pointer-numeric-in-range check '{}' has min ({min}) greater than max ({max})",
                        check.display_name()
                    ),
                });
            }
        }
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
        Check::TraceToolCall { tool_name, .. } if tool_name.trim().is_empty() => {
            return Err(CaseFileError::Validation {
                path: path.to_path_buf(),
                message: format!(
                    "trace-tool-call check '{}' has an empty tool name",
                    check.display_name()
                ),
            });
        }
        Check::TraceToolCall {
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
            allow_empty_output: false,
            activate_allocation: Vec::new(),
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
    fn validate_full_agent_case_accepts_empty_step_inputs_with_interoception_wait() {
        let case = full_agent_case_with(
            Vec::new(),
            vec![EvalStep {
                description: None,
                inputs: Vec::new(),
                wait_for: Some(WaitFor::Interoception {
                    timeout_ms: 5_000,
                    mode: Some(EvalInteroceptiveMode::NremPressure),
                    wake_arousal_at_least: default_wake_arousal_at_least(),
                    wake_arousal_at_most: 0.25,
                }),
                checks: Vec::new(),
            }],
        );

        validate_full_agent_case(Path::new("case.eure"), &case).unwrap();
    }

    #[test]
    fn validate_full_agent_case_rejects_empty_interoception_wait_conditions() {
        let case = full_agent_case_with(
            Vec::new(),
            vec![EvalStep {
                description: None,
                inputs: Vec::new(),
                wait_for: Some(WaitFor::Interoception {
                    timeout_ms: 5_000,
                    mode: None,
                    wake_arousal_at_least: default_wake_arousal_at_least(),
                    wake_arousal_at_most: default_wake_arousal_at_most(),
                }),
                checks: Vec::new(),
            }],
        );

        let error = validate_full_agent_case(Path::new("case.eure"), &case).unwrap_err();
        assert!(
            matches!(error, CaseFileError::Validation { message, .. } if message.contains("wait-for.interoception must set at least one condition"))
        );
    }

    #[test]
    fn validate_full_agent_case_rejects_invalid_activate_allocation() {
        let mut case = full_agent_case_with(vec![seen_input("input")], Vec::new());
        case.modules = Some(vec![EvalModule::Sensory]);
        case.activate_allocation = vec![ActivateAllocation {
            module: EvalModule::Interoception,
            activation_ratio: 1.0,
        }];

        let error = validate_full_agent_case(Path::new("case.eure"), &case).unwrap_err();
        assert!(
            matches!(error, CaseFileError::Validation { message, .. } if message.contains("activate-allocation[0].module 'interoception' must be included"))
        );

        let mut case = full_agent_case_with(vec![seen_input("input")], Vec::new());
        case.activate_allocation = vec![ActivateAllocation {
            module: EvalModule::Sensory,
            activation_ratio: 1.2,
        }];
        let error = validate_full_agent_case(Path::new("case.eure"), &case).unwrap_err();
        assert!(
            matches!(error, CaseFileError::Validation { message, .. } if message.contains("activation-ratio must be between"))
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
    fn validate_check_rejects_invalid_trace_tool_call_args_json_contains() {
        let check = Check::TraceToolCall {
            common: CheckCommon {
                name: Some("bad-tool-args".to_string()),
                must_pass: true,
                weight: 1,
            },
            tool_name: "write_retrieval_memo".to_string(),
            args_json_contains: Some(Text::plaintext("{not-json")),
        };

        let error = validate_common(
            Path::new("case.eure"),
            None,
            &[],
            &[],
            &[],
            &[],
            &EvalLimits::default(),
            &[check],
        )
        .unwrap_err();

        assert!(
            matches!(error, CaseFileError::Validation { message, .. } if message.contains("invalid args-json-contains JSON"))
        );
    }

    #[test]
    fn parses_module_case_steps() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("module-steps.eure");
        std::fs::write(
            &path,
            r#"
id = "module-steps"
prompt = "Admit load-bearing facts."

@ steps[] {
  description = "First activation"

  @ memos[] {
    module = "sensory"
    content = "Pibi asked about Koro."
  }
}

@ steps[] {
  description = "Second activation"

  @ memos[] {
    module = "query-memory"
    content = "Koro lunged when I turned away."
  }
}
"#,
        )
        .unwrap();

        let case = parse_module_case_file(&path).unwrap();

        assert!(case.memos.is_empty());
        assert_eq!(case.steps.len(), 2);
        assert_eq!(
            case.steps[0]
                .description
                .as_ref()
                .map(|text| text.content.as_str()),
            Some("First activation")
        );
        assert_eq!(case.steps[0].memos[0].module, "sensory");
        assert_eq!(case.steps[1].memos[0].module, "query-memory");
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
        assert!(matches!(case.checks.as_slice(), [Check::Rubric { .. }]));
    }

    #[test]
    fn parses_sleep_wake_sample_cases() {
        let quiet = Path::new("../../eval-cases/full-agent/sleep-after-sensory-quiet.eure");
        let quiet = parse_full_agent_case_file(quiet).expect("sleep case should parse");
        assert!(quiet.allow_empty_output);
        assert_eq!(quiet.activate_allocation.len(), 4);
        assert_eq!(
            quiet.activate_allocation[2].module,
            EvalModule::Interoception
        );
        assert_eq!(quiet.activate_allocation[2].activation_ratio, 1.0);
        assert_eq!(quiet.limits.interoception.quiet_sleep_threshold_ms, 1_000);
        assert_eq!(
            quiet.limits.interoception.wake_arousal_change_multiplier,
            8.0
        );
        assert!(matches!(
            quiet.steps[0].wait_for,
            Some(WaitFor::Interoception {
                mode: Some(EvalInteroceptiveMode::NremPressure),
                wake_arousal_at_most: 0.25,
                ..
            })
        ));

        let wake = Path::new("../../eval-cases/full-agent/wake-from-sleep-on-salient-sensory.eure");
        let wake = parse_full_agent_case_file(wake).expect("wake case should parse");
        assert!(wake.allow_empty_output);
        let wake_modules = wake.modules.as_ref().expect("wake case lists modules");
        assert!(!wake_modules.contains(&EvalModule::MemoryCompaction));
        assert!(!wake_modules.contains(&EvalModule::MemoryAssociation));
        assert!(!wake_modules.contains(&EvalModule::Dreaming));
        assert!(!wake_modules.contains(&EvalModule::PolicyCompaction));
        assert_eq!(wake.steps.len(), 2);
        assert!(matches!(
            wake.steps[1].wait_for,
            Some(WaitFor::Interoception {
                mode: Some(EvalInteroceptiveMode::Wake),
                wake_arousal_at_least: 0.70,
                ..
            })
        ));
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
            index: None,
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
            &[],
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
