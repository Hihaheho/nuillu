use std::{
    collections::BTreeSet,
    fs, io,
    path::{Path, PathBuf},
};

use eure::{FromEure, value::Text};
use nuillu_types::MemoryRank;
use thiserror::Error;

fn default_weight() -> i64 {
    1
}

fn default_query_target() -> CaseTarget {
    CaseTarget::Query
}

fn default_memory_rank() -> MemorySeedRank {
    MemorySeedRank::ShortTerm
}

fn default_pass_score() -> f64 {
    0.8
}

fn default_judge_max_output_tokens() -> u32 {
    1200
}

#[derive(Debug, Clone, FromEure)]
#[eure(crate = ::eure::document, rename_all = "kebab-case")]
pub struct EvalCaseFile {
    #[eure(flatten)]
    pub case: EvalCase,
}

#[derive(Debug, Clone, FromEure)]
#[eure(crate = ::eure::document, rename_all = "kebab-case")]
pub struct EvalCase {
    #[eure(default)]
    pub id: Option<String>,
    #[eure(default)]
    pub description: Option<Text>,
    #[eure(default = "default_query_target")]
    pub target: CaseTarget,
    pub prompt: Text,
    #[eure(default)]
    pub context: Option<Text>,
    #[eure(default)]
    pub memories: Vec<MemorySeed>,
    #[eure(default)]
    pub checks: Vec<Check>,
    #[eure(default)]
    pub scoring: CaseScoring,
}

impl Default for EvalCase {
    fn default() -> Self {
        Self {
            id: None,
            description: None,
            target: default_query_target(),
            prompt: Text::plaintext(String::new()),
            context: None,
            memories: Vec::new(),
            checks: Vec::new(),
            scoring: CaseScoring::default(),
        }
    }
}

impl Default for EvalCaseFile {
    fn default() -> Self {
        Self {
            case: EvalCase::default(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, FromEure)]
#[eure(crate = ::eure::document, rename_all = "kebab-case")]
pub enum CaseTarget {
    Query,
    SelfModel,
    Conversation,
    Periodic,
    Custom,
}

#[derive(Debug, Clone, FromEure)]
#[eure(crate = ::eure::document, rename_all = "kebab-case")]
pub struct MemorySeed {
    #[eure(default)]
    pub key: Option<String>,
    #[eure(default = "default_memory_rank")]
    pub rank: MemorySeedRank,
    pub content: Text,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, FromEure)]
#[eure(crate = ::eure::document, rename_all = "kebab-case")]
pub enum MemorySeedRank {
    ShortTerm,
    MidTerm,
    LongTerm,
    Permanent,
}

impl From<MemorySeedRank> for MemoryRank {
    fn from(rank: MemorySeedRank) -> Self {
        match rank {
            MemorySeedRank::ShortTerm => Self::ShortTerm,
            MemorySeedRank::MidTerm => Self::MidTerm,
            MemorySeedRank::LongTerm => Self::LongTerm,
            MemorySeedRank::Permanent => Self::Permanent,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, FromEure)]
#[eure(crate = ::eure::document, rename_all = "kebab-case")]
pub enum ArtifactTextField {
    Output,
    Failure,
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
    let content = fs::read_to_string(path).map_err(|source| CaseFileError::Read {
        path: path.to_path_buf(),
        source,
    })?;
    let file: EvalCaseFile =
        eure::parse_content(&content, path.to_path_buf()).map_err(|message| {
            CaseFileError::Parse {
                path: path.to_path_buf(),
                message,
            }
        })?;
    validate_case(path, &file.case)?;
    Ok(file.case)
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
    matches!(name, "result.eure" | "report.eure" | "artifact.eure")
}

fn validate_case(path: &Path, case: &EvalCase) -> Result<(), CaseFileError> {
    if case.prompt.content.trim().is_empty() {
        return Err(CaseFileError::Validation {
            path: path.to_path_buf(),
            message: "prompt must not be empty".to_string(),
        });
    }

    for (index, memory) in case.memories.iter().enumerate() {
        if memory.key.as_deref().is_some_and(str::is_empty) {
            return Err(CaseFileError::Validation {
                path: path.to_path_buf(),
                message: format!("memories[{index}].key must not be empty when present"),
            });
        }
        if memory.content.content.trim().is_empty() {
            return Err(CaseFileError::Validation {
                path: path.to_path_buf(),
                message: format!("memories[{index}].content must not be empty"),
            });
        }
    }

    for check in &case.checks {
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
        criteria,
        ..
    } = check
    {
        if rubric.content.trim().is_empty() {
            return Err(CaseFileError::Validation {
                path: path.to_path_buf(),
                message: format!(
                    "rubric check '{}' has an empty rubric",
                    check.display_name()
                ),
            });
        }
        validate_pass_score(path, *pass_score, &check.display_name())?;
        let mut names = BTreeSet::new();
        for criterion in criteria {
            if criterion.name.trim().is_empty() {
                return Err(CaseFileError::Validation {
                    path: path.to_path_buf(),
                    message: format!(
                        "rubric check '{}' has an empty criterion name",
                        check.display_name()
                    ),
                });
            }
            if !names.insert(criterion.name.clone()) {
                return Err(CaseFileError::Validation {
                    path: path.to_path_buf(),
                    message: format!(
                        "rubric check '{}' has duplicate criterion '{}'",
                        check.display_name(),
                        criterion.name
                    ),
                });
            }
            validate_pass_score(path, criterion.pass_score, &criterion.name)?;
        }
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
