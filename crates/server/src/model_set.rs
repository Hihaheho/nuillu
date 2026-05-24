use std::{
    collections::{BTreeMap, HashMap},
    fs, io,
    num::NonZeroUsize,
    path::{Path, PathBuf},
};

use anyhow::Context as _;
use eure::FromEure;
use lutum_openai::OpenAiReasoningEffort;
use nuillu_module::DEFAULT_SESSION_COMPACTION_INPUT_TOKEN_THRESHOLD;
use serde::Serialize;
use thiserror::Error;

use crate::config::LlmBackendConfig;

const DEFAULT_OPENAI_COMPAT_ENDPOINT: &str = "http://localhost:11434/v1";
const DEFAULT_OPENAI_COMPAT_TOKEN: &str = "local";

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
    pub compaction_input_token_threshold: Option<u64>,
    #[eure(default)]
    pub models: HashMap<String, ModelDefinition>,
    #[eure(default)]
    pub judge_model: Option<String>,
    #[eure(default)]
    pub cheap_model: Option<String>,
    #[eure(default)]
    pub default_model: Option<String>,
    #[eure(default)]
    pub premium_model: Option<String>,
    #[eure(default)]
    pub judge: Option<TierBinding>,
    #[eure(default)]
    pub cheap: Option<TierBinding>,
    #[eure(default)]
    pub default: Option<TierBinding>,
    #[eure(default)]
    pub premium: Option<TierBinding>,
    #[eure(default)]
    pub embedding: Option<EmbeddingRole>,
}

#[derive(Debug, Clone, FromEure)]
#[eure(crate = ::eure::document, rename_all = "kebab-case")]
pub struct ModelDefinition {
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
    #[eure(default)]
    pub max_concurrent_llm_calls: Option<u64>,
}

impl ModelDefinition {
    pub fn endpoint(&self) -> Option<&str> {
        self.endpoint.as_deref().or(self.base_url.as_deref())
    }

    pub fn max_concurrent_llm_calls(&self) -> Option<NonZeroUsize> {
        self.max_concurrent_llm_calls
            .and_then(|value| usize::try_from(value).ok())
            .and_then(NonZeroUsize::new)
    }
}

#[derive(Debug, Clone, FromEure)]
#[eure(crate = ::eure::document, rename_all = "kebab-case")]
pub struct TierBinding {
    pub model: String,
    #[eure(default)]
    pub reasoning_effort: Option<ReasoningEffort>,
    #[eure(default)]
    pub compaction_input_token_threshold: Option<u64>,
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

#[derive(Debug, Clone)]
pub struct ResolvedLlmBackends {
    pub judge: Option<LlmBackendConfig>,
    pub cheap: LlmBackendConfig,
    pub default: LlmBackendConfig,
    pub premium: LlmBackendConfig,
    pub model_concurrency: BTreeMap<String, Option<NonZeroUsize>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TierRole {
    Judge,
    Cheap,
    Default,
    Premium,
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

pub fn resolve_llm_backends(model_set: &ModelSet) -> anyhow::Result<ResolvedLlmBackends> {
    let global_compaction = model_set.compaction_input_token_threshold;
    let judge = resolve_tier(
        TierRole::Judge,
        model_set.judge.as_ref(),
        model_set.judge_model.as_deref(),
        &model_set.models,
        global_compaction,
    )?;
    let cheap = resolve_tier(
        TierRole::Cheap,
        model_set.cheap.as_ref(),
        model_set.cheap_model.as_deref(),
        &model_set.models,
        global_compaction,
    )?
    .ok_or_else(|| {
        anyhow::anyhow!("missing cheap tier: set cheap-model or cheap {{ model = ... }}")
    })?;
    let default = resolve_tier(
        TierRole::Default,
        model_set.default.as_ref(),
        model_set.default_model.as_deref(),
        &model_set.models,
        global_compaction,
    )?
    .ok_or_else(|| {
        anyhow::anyhow!("missing default tier: set default-model or default {{ model = ... }}")
    })?;
    let premium = resolve_tier(
        TierRole::Premium,
        model_set.premium.as_ref(),
        model_set.premium_model.as_deref(),
        &model_set.models,
        global_compaction,
    )?
    .ok_or_else(|| {
        anyhow::anyhow!("missing premium tier: set premium-model or premium {{ model = ... }}")
    })?;

    let mut model_concurrency = BTreeMap::new();
    for backend in [&cheap, &default, &premium]
        .into_iter()
        .chain(judge.as_ref())
    {
        model_concurrency
            .entry(backend.model_key.clone())
            .or_insert(backend.max_concurrent_llm_calls);
    }

    Ok(ResolvedLlmBackends {
        judge,
        cheap,
        default,
        premium,
        model_concurrency,
    })
}

pub fn model_concurrency_from_backends(
    backends: impl IntoIterator<Item = LlmBackendConfig>,
) -> BTreeMap<String, Option<NonZeroUsize>> {
    let mut model_concurrency = BTreeMap::new();
    for backend in backends {
        model_concurrency
            .entry(backend.model_key)
            .or_insert(backend.max_concurrent_llm_calls);
    }
    model_concurrency
}

fn resolve_tier(
    role: TierRole,
    binding: Option<&TierBinding>,
    model_ref: Option<&str>,
    models: &HashMap<String, ModelDefinition>,
    global_compaction: Option<u64>,
) -> anyhow::Result<Option<LlmBackendConfig>> {
    let label = tier_label(role);
    let binding = match (binding, model_ref) {
        (Some(_), Some(_)) => {
            anyhow::bail!("{label} cannot set both {label}-model and {label} {{ ... }}")
        }
        (Some(binding), None) => Some(binding.clone()),
        (None, Some(model_key)) => Some(TierBinding {
            model: model_key.to_string(),
            reasoning_effort: None,
            compaction_input_token_threshold: None,
        }),
        (None, None) => return Ok(None),
    };

    let binding = binding.expect("resolved above");
    let definition = models
        .get(&binding.model)
        .ok_or_else(|| anyhow::anyhow!("{label} references unknown model `{}`", binding.model))?;
    let endpoint = definition
        .endpoint()
        .unwrap_or(DEFAULT_OPENAI_COMPAT_ENDPOINT)
        .to_string();
    let token = resolve_token_fields(
        label,
        definition.token_env.as_deref(),
        definition.token.as_deref(),
        Some(DEFAULT_OPENAI_COMPAT_TOKEN),
    )?;
    let api_model = definition
        .model
        .as_deref()
        .ok_or_else(|| anyhow::anyhow!("models.{}.model must be set", binding.model))?
        .to_string();
    let reasoning_effort = binding.reasoning_effort.or(definition.reasoning_effort);
    let use_responses_api = definition.use_responses_api.unwrap_or(false);
    let compaction_input_token_threshold = binding
        .compaction_input_token_threshold
        .or(global_compaction)
        .unwrap_or(DEFAULT_SESSION_COMPACTION_INPUT_TOKEN_THRESHOLD);

    Ok(Some(LlmBackendConfig {
        model_key: binding.model,
        endpoint,
        token,
        model: api_model,
        reasoning_effort,
        use_responses_api,
        compaction_input_token_threshold,
        max_concurrent_llm_calls: definition.max_concurrent_llm_calls(),
    }))
}

pub fn resolve_token_fields(
    label: &str,
    token_env: Option<&str>,
    token: Option<&str>,
    default: Option<&str>,
) -> anyhow::Result<String> {
    if let Some(env_name) = token_env {
        match std::env::var(env_name) {
            Ok(value) if !value.trim().is_empty() => return Ok(value),
            Ok(_) => {
                if let Some(value) = token {
                    return Ok(value.to_string());
                }
                anyhow::bail!("{label}.token-env {env_name} is set but empty");
            }
            Err(source) => {
                if let Some(value) = token {
                    return Ok(value.to_string());
                }
                return Err(source)
                    .with_context(|| format!("read {label}.token-env {env_name} from model set"));
            }
        }
    }

    if let Some(value) = token {
        return Ok(value.to_string());
    }
    match default {
        Some(value) => Ok(value.to_string()),
        None => anyhow::bail!("{label}.token or {label}.token-env must be set"),
    }
}

fn tier_label(role: TierRole) -> &'static str {
    match role {
        TierRole::Judge => "judge",
        TierRole::Cheap => "cheap",
        TierRole::Default => "default",
        TierRole::Premium => "premium",
    }
}

fn validate_model_set(path: &Path, model_set: &ModelSet) -> Result<(), ModelSetError> {
    validate_optional_text(path, "name", model_set.name.as_deref())?;
    if model_set.compaction_input_token_threshold == Some(0) {
        return Err(ModelSetError::Validation {
            path: path.to_path_buf(),
            message: "compaction-input-token-threshold must be greater than zero when set"
                .to_string(),
        });
    }
    if model_set.models.is_empty() {
        return Err(ModelSetError::Validation {
            path: path.to_path_buf(),
            message: "models must contain at least one model definition".to_string(),
        });
    }
    for (name, definition) in &model_set.models {
        validate_model_definition(path, name, definition)?;
    }
    validate_tier_binding(
        path,
        TierRole::Judge,
        model_set.judge.as_ref(),
        model_set.judge_model.as_deref(),
    )?;
    validate_tier_binding(
        path,
        TierRole::Cheap,
        model_set.cheap.as_ref(),
        model_set.cheap_model.as_deref(),
    )?;
    validate_tier_binding(
        path,
        TierRole::Default,
        model_set.default.as_ref(),
        model_set.default_model.as_deref(),
    )?;
    validate_tier_binding(
        path,
        TierRole::Premium,
        model_set.premium.as_ref(),
        model_set.premium_model.as_deref(),
    )?;
    validate_embedding_role(path, model_set.embedding.as_ref())
}

fn validate_model_definition(
    path: &Path,
    name: &str,
    definition: &ModelDefinition,
) -> Result<(), ModelSetError> {
    let prefix = format!("models.{name}");
    if definition.endpoint.is_some() && definition.base_url.is_some() {
        return Err(ModelSetError::Validation {
            path: path.to_path_buf(),
            message: format!("{prefix}.endpoint and {prefix}.base-url cannot both be set"),
        });
    }
    validate_optional_text(
        path,
        &format!("{prefix}.endpoint"),
        definition.endpoint.as_deref(),
    )?;
    validate_optional_text(
        path,
        &format!("{prefix}.base-url"),
        definition.base_url.as_deref(),
    )?;
    validate_optional_text(
        path,
        &format!("{prefix}.token"),
        definition.token.as_deref(),
    )?;
    validate_optional_text(
        path,
        &format!("{prefix}.token-env"),
        definition.token_env.as_deref(),
    )?;
    validate_optional_text(
        path,
        &format!("{prefix}.model"),
        definition.model.as_deref(),
    )?;
    if let Some(value) = definition.max_concurrent_llm_calls {
        if value == 0 {
            return Err(ModelSetError::Validation {
                path: path.to_path_buf(),
                message: format!(
                    "{prefix}.max-concurrent-llm-calls must be greater than zero when set"
                ),
            });
        }
        if usize::try_from(value).is_err() {
            return Err(ModelSetError::Validation {
                path: path.to_path_buf(),
                message: format!("{prefix}.max-concurrent-llm-calls must fit in usize"),
            });
        }
    }
    Ok(())
}

fn validate_tier_binding(
    path: &Path,
    role: TierRole,
    binding: Option<&TierBinding>,
    model_ref: Option<&str>,
) -> Result<(), ModelSetError> {
    let label = tier_label(role);
    if binding.is_some() && model_ref.is_some() {
        return Err(ModelSetError::Validation {
            path: path.to_path_buf(),
            message: format!("{label} cannot set both {label}-model and {label} {{ ... }}"),
        });
    }
    let Some(binding) = binding else {
        validate_optional_text(path, &format!("{label}-model"), model_ref)?;
        return Ok(());
    };
    validate_optional_text(
        path,
        &format!("{label}.model"),
        Some(binding.model.as_str()),
    )?;
    if binding.compaction_input_token_threshold == Some(0) {
        return Err(ModelSetError::Validation {
            path: path.to_path_buf(),
            message: format!(
                "{label}.compaction-input-token-threshold must be greater than zero when set"
            ),
        });
    }
    Ok(())
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

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::*;

    fn parse_model_set(content: &str) -> Result<ModelSet, ModelSetError> {
        let path = Path::new("test-model-set.eure");
        let file: ModelSetFile =
            eure::parse_content(content, path.to_path_buf()).map_err(|message| {
                ModelSetError::Parse {
                    path: path.to_path_buf(),
                    message,
                }
            })?;
        validate_model_set(path, &file.model_set)?;
        Ok(file.model_set)
    }

    #[test]
    fn parses_models_and_tier_refs() {
        let model_set = parse_model_set(
            r#"
models {
  gemma4 {
    endpoint = "http://localhost:8080/v1"
    token = "local"
    model = "gemma4:e4b"
    max-concurrent-llm-calls = 4
    reasoning-effort = "none"
  }
}

cheap-model = "gemma4"
default-model = "gemma4"
premium-model = "gemma4"
"#,
        )
        .unwrap();

        assert_eq!(model_set.models.len(), 1);
        assert_eq!(model_set.cheap_model.as_deref(), Some("gemma4"));
        let resolved = resolve_llm_backends(&model_set).unwrap();
        assert_eq!(resolved.cheap.model_key, "gemma4");
        assert_eq!(
            resolved.cheap.max_concurrent_llm_calls,
            NonZeroUsize::new(4)
        );
        assert_eq!(
            resolved.model_concurrency.get("gemma4"),
            Some(&NonZeroUsize::new(4))
        );
    }

    #[test]
    fn parses_tier_overlay_with_reasoning_effort() {
        let model_set = parse_model_set(
            r#"
models {
  gpt5-nano {
    endpoint = "https://api.openai.com/v1"
    token = "local"
    model = "gpt-5.4-nano"
    use-responses-api = true
    max-concurrent-llm-calls = 2
  }
}

cheap {
  model = "gpt5-nano"
  reasoning-effort = "low"
}

default {
  model = "gpt5-nano"
  reasoning-effort = "medium"
}

premium {
  model = "gpt5-nano"
  reasoning-effort = "medium"
}
"#,
        )
        .unwrap();

        let resolved = resolve_llm_backends(&model_set).unwrap();
        assert_eq!(resolved.cheap.reasoning_effort, Some(ReasoningEffort::Low));
        assert_eq!(
            resolved.default.reasoning_effort,
            Some(ReasoningEffort::Medium)
        );
        assert_eq!(resolved.cheap.model_key, "gpt5-nano");
        assert_eq!(
            resolved.cheap.max_concurrent_llm_calls,
            NonZeroUsize::new(2)
        );
    }

    #[test]
    fn rejects_unknown_model_reference() {
        let model_set = parse_model_set(
            r#"
models {
  gemma4 {
    endpoint = "http://localhost:8080/v1"
    token = "local"
    model = "gemma4:e4b"
  }
}

cheap-model = "missing"
"#,
        )
        .unwrap();

        let error = resolve_llm_backends(&model_set).unwrap_err();
        assert!(error.to_string().contains("unknown model `missing`"));
    }

    #[test]
    fn rejects_zero_max_concurrent_llm_calls_on_model() {
        let error = parse_model_set(
            r#"
models {
  gemma4 {
    endpoint = "http://localhost:8080/v1"
    token = "local"
    model = "gemma4:e4b"
    max-concurrent-llm-calls = 0
  }
}
"#,
        )
        .unwrap_err();

        assert!(matches!(
            error,
            ModelSetError::Validation { message, .. }
                if message.contains("max-concurrent-llm-calls must be greater than zero")
        ));
    }

    #[test]
    fn rejects_conflicting_tier_ref_and_block() {
        let error = parse_model_set(
            r#"
models {
  gemma4 {
    endpoint = "http://localhost:8080/v1"
    token = "local"
    model = "gemma4:e4b"
  }
}

cheap-model = "gemma4"

cheap {
  model = "gemma4"
}
"#,
        )
        .unwrap_err();

        assert!(matches!(
            error,
            ModelSetError::Validation { message, .. }
                if message.contains("cannot set both cheap-model and cheap")
        ));
    }

    #[test]
    fn parses_global_compaction_input_token_threshold() {
        let model_set = parse_model_set(
            r#"
compaction-input-token-threshold = 8192

models {
  gemma4 {
    endpoint = "http://localhost:8080/v1"
    token = "local"
    model = "gemma4:e4b"
  }
}

cheap-model = "gemma4"
default-model = "gemma4"
premium-model = "gemma4"
"#,
        )
        .unwrap();

        assert_eq!(model_set.compaction_input_token_threshold, Some(8192));
        let resolved = resolve_llm_backends(&model_set).unwrap();
        assert_eq!(resolved.cheap.compaction_input_token_threshold, 8192);
    }

    #[test]
    fn tier_compaction_threshold_overrides_global() {
        let model_set = parse_model_set(
            r#"
compaction-input-token-threshold = 8192

models {
  gemma4 {
    endpoint = "http://localhost:8080/v1"
    token = "local"
    model = "gemma4:e4b"
  }
}

cheap {
  model = "gemma4"
  compaction-input-token-threshold = 4096
}

default {
  model = "gemma4"
}

premium {
  model = "gemma4"
}
"#,
        )
        .unwrap();

        let resolved = resolve_llm_backends(&model_set).unwrap();
        assert_eq!(resolved.cheap.compaction_input_token_threshold, 4096);
    }
}
