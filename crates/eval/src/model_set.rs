use std::{
    fs, io,
    path::{Path, PathBuf},
};

use eure::FromEure;
use lutum_openai::OpenAiReasoningEffort;
use serde::Serialize;
use thiserror::Error;

#[derive(Debug, Clone, FromEure)]
#[eure(crate = ::eure::document, rename_all = "kebab-case")]
pub struct ModelSetFile {
    #[eure(flatten)]
    pub model_set: ModelSet,
}

#[derive(Debug, Clone, FromEure)]
#[eure(crate = ::eure::document, rename_all = "kebab-case")]
pub struct ModelSet {
    #[eure(default)]
    pub name: Option<String>,
    #[eure(default)]
    pub judge: Option<ModelSetRole>,
    #[eure(default)]
    pub cheap: Option<ModelSetRole>,
    #[eure(default)]
    pub default: Option<ModelSetRole>,
    #[eure(default)]
    pub premium: Option<ModelSetRole>,
    #[eure(default)]
    pub embedding: Option<EmbeddingRole>,
}

#[derive(Debug, Clone, FromEure)]
#[eure(crate = ::eure::document, rename_all = "kebab-case")]
pub struct ModelSetRole {
    #[eure(default)]
    pub endpoint: Option<String>,
    #[eure(default)]
    pub base_url: Option<String>,
    #[eure(default)]
    pub token: Option<String>,
    #[eure(default)]
    pub token_env: Option<String>,
    #[eure(default)]
    pub model: Option<String>,
    #[eure(default)]
    pub reasoning_effort: Option<ReasoningEffort>,
    #[eure(default)]
    pub use_responses_api: Option<bool>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, clap::ValueEnum, FromEure)]
#[eure(crate = ::eure::document, rename_all = "kebab-case")]
#[serde(rename_all = "lowercase")]
pub enum ReasoningEffort {
    None,
    Minimal,
    Low,
    Medium,
    High,
    Xhigh,
}

impl From<ReasoningEffort> for OpenAiReasoningEffort {
    fn from(value: ReasoningEffort) -> Self {
        match value {
            ReasoningEffort::None => Self::None,
            ReasoningEffort::Minimal => Self::Minimal,
            ReasoningEffort::Low => Self::Low,
            ReasoningEffort::Medium => Self::Medium,
            ReasoningEffort::High => Self::High,
            ReasoningEffort::Xhigh => Self::Xhigh,
        }
    }
}

impl ModelSetRole {
    pub fn endpoint(&self) -> Option<&str> {
        self.endpoint.as_deref().or(self.base_url.as_deref())
    }
}

#[derive(Debug, Clone, FromEure)]
#[eure(crate = ::eure::document, rename_all = "kebab-case")]
pub struct EmbeddingRole {
    #[eure(default)]
    pub endpoint: Option<String>,
    #[eure(default)]
    pub base_url: Option<String>,
    #[eure(default)]
    pub token: Option<String>,
    #[eure(default)]
    pub token_env: Option<String>,
    #[eure(default)]
    pub model: Option<String>,
    #[eure(default)]
    pub dimensions: Option<u32>,
}

impl EmbeddingRole {
    pub fn endpoint(&self) -> Option<&str> {
        self.endpoint.as_deref().or(self.base_url.as_deref())
    }
}

#[derive(Debug, Error)]
pub enum ModelSetError {
    #[error("failed to read model set {path}: {source}")]
    Read {
        path: PathBuf,
        #[source]
        source: io::Error,
    },
    #[error("failed to parse model set {path}: {message}")]
    Parse { path: PathBuf, message: String },
    #[error("invalid model set {path}: {message}")]
    Validation { path: PathBuf, message: String },
}

pub fn parse_model_set_file(path: &Path) -> Result<ModelSet, ModelSetError> {
    let content = fs::read_to_string(path).map_err(|source| ModelSetError::Read {
        path: path.to_path_buf(),
        source,
    })?;
    let file: ModelSetFile =
        eure::parse_content(&content, path.to_path_buf()).map_err(|message| {
            ModelSetError::Parse {
                path: path.to_path_buf(),
                message,
            }
        })?;
    validate_model_set(path, &file.model_set)?;
    Ok(file.model_set)
}

fn validate_model_set(path: &Path, model_set: &ModelSet) -> Result<(), ModelSetError> {
    validate_optional_text(path, "name", model_set.name.as_deref())?;
    validate_role(path, "judge", model_set.judge.as_ref())?;
    validate_role(path, "cheap", model_set.cheap.as_ref())?;
    validate_role(path, "default", model_set.default.as_ref())?;
    validate_role(path, "premium", model_set.premium.as_ref())?;
    validate_embedding_role(path, model_set.embedding.as_ref())
}

fn validate_embedding_role(path: &Path, role: Option<&EmbeddingRole>) -> Result<(), ModelSetError> {
    let Some(role) = role else {
        return Ok(());
    };
    if role.endpoint.is_some() && role.base_url.is_some() {
        return Err(ModelSetError::Validation {
            path: path.to_path_buf(),
            message: "embedding.endpoint and embedding.base-url cannot both be set".to_string(),
        });
    }
    validate_optional_text(path, "embedding.endpoint", role.endpoint.as_deref())?;
    validate_optional_text(path, "embedding.base-url", role.base_url.as_deref())?;
    validate_optional_text(path, "embedding.token", role.token.as_deref())?;
    validate_optional_text(path, "embedding.token-env", role.token_env.as_deref())?;
    validate_optional_text(path, "embedding.model", role.model.as_deref())?;
    if role.dimensions == Some(0) {
        return Err(ModelSetError::Validation {
            path: path.to_path_buf(),
            message: "embedding.dimensions must be greater than zero when set".to_string(),
        });
    }
    Ok(())
}

fn validate_role(
    path: &Path,
    name: &str,
    role: Option<&ModelSetRole>,
) -> Result<(), ModelSetError> {
    let Some(role) = role else {
        return Ok(());
    };
    if role.endpoint.is_some() && role.base_url.is_some() {
        return Err(ModelSetError::Validation {
            path: path.to_path_buf(),
            message: format!("{name}.endpoint and {name}.base-url cannot both be set"),
        });
    }
    validate_optional_text(path, &format!("{name}.endpoint"), role.endpoint.as_deref())?;
    validate_optional_text(path, &format!("{name}.base-url"), role.base_url.as_deref())?;
    validate_optional_text(path, &format!("{name}.token"), role.token.as_deref())?;
    validate_optional_text(
        path,
        &format!("{name}.token-env"),
        role.token_env.as_deref(),
    )?;
    validate_optional_text(path, &format!("{name}.model"), role.model.as_deref())
}

fn validate_optional_text(
    path: &Path,
    field: &str,
    value: Option<&str>,
) -> Result<(), ModelSetError> {
    if value.is_some_and(|value| value.trim().is_empty()) {
        return Err(ModelSetError::Validation {
            path: path.to_path_buf(),
            message: format!("{field} must not be empty when present"),
        });
    }
    Ok(())
}
