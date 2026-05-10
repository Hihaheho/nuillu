use std::fmt;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Identifier for a module instance.
///
/// String-backed so that user-defined modules can be added without changing
/// the type system. Module ids are kebab-case: lowercase ASCII words separated
/// by single hyphens. The well-known cognitive modules from the design have
/// constants in [`builtin`] for ergonomic equality checks and routing.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
pub struct ModuleId(String);

impl ModuleId {
    pub fn new(name: impl Into<String>) -> Result<Self, ModuleIdParseError> {
        let name = name.into();
        if name.is_empty() {
            return Err(ModuleIdParseError::Empty);
        }
        if !is_kebab_case(&name) {
            return Err(ModuleIdParseError::InvalidChar);
        }
        Ok(Self(name))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

fn is_kebab_case(value: &str) -> bool {
    let bytes = value.as_bytes();
    if !bytes.first().is_some_and(|b| b.is_ascii_lowercase()) {
        return false;
    }

    let mut prev_hyphen = false;
    for &b in bytes {
        let valid = b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'-';
        if !valid {
            return false;
        }
        if b == b'-' {
            if prev_hyphen {
                return false;
            }
            prev_hyphen = true;
        } else {
            prev_hyphen = false;
        }
    }
    !prev_hyphen
}

impl fmt::Display for ModuleId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// Zero-based index of one persistent replica for a module role.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize, JsonSchema,
)]
pub struct ReplicaIndex(u8);

impl ReplicaIndex {
    pub const ZERO: Self = Self(0);

    pub fn new(index: u8) -> Self {
        Self(index)
    }

    pub fn get(self) -> u8 {
        self.0
    }
}

impl fmt::Display for ReplicaIndex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

/// One persistent module loop. Owner-stamped capabilities carry this value.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
pub struct ModuleInstanceId {
    pub module: ModuleId,
    pub replica: ReplicaIndex,
}

impl ModuleInstanceId {
    pub fn new(module: ModuleId, replica: ReplicaIndex) -> Self {
        Self { module, replica }
    }
}

impl fmt::Display for ModuleInstanceId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.replica == ReplicaIndex::ZERO {
            self.module.fmt(f)
        } else {
            write!(f, "{}[{}]", self.module, self.replica)
        }
    }
}

/// Boot-time policy limiting the replicas a module role may run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct ReplicaCapRange {
    pub min: u8,
    pub max: u8,
}

impl ReplicaCapRange {
    pub const V1_MAX: u8 = 2;

    /// Construct a range of total active replicas. `new(0, 1)` means the
    /// module can be fully disabled or run one active replica.
    pub fn new(min: u8, max: u8) -> Result<Self, ReplicaCapRangeError> {
        if min > max {
            return Err(ReplicaCapRangeError::MinGreaterThanMax);
        }
        if max > Self::V1_MAX {
            return Err(ReplicaCapRangeError::AboveV1Max { max });
        }
        Ok(Self { min, max })
    }

    pub fn clamp(self, replicas: u8) -> u8 {
        replicas.clamp(self.min, self.max)
    }
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum ReplicaCapRangeError {
    #[error("replica cap range min must be <= max")]
    MinGreaterThanMax,
    #[error("replica cap range max {max} exceeds v1 limit")]
    AboveV1Max { max: u8 },
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum ModuleIdParseError {
    #[error("module id must not be empty")]
    Empty,
    #[error("module id must be kebab-case: [a-z][a-z0-9]*(?:-[a-z0-9]+)*")]
    InvalidChar,
}

/// Constructors for the cognitive modules defined in `attention-schema.md`.
///
/// These are *conventions*, not enum variants — the agent supports modules
/// outside this list. Use these so routing is consistent across the workspace.
pub mod builtin {
    use super::ModuleId;

    macro_rules! builtin {
        ($($fn_name:ident => $id:literal),* $(,)?) => {
            $(
                pub fn $fn_name() -> ModuleId {
                    ModuleId::new($id).expect("builtin id is valid")
                }
            )*
        };
    }

    builtin!(
        sensory               => "sensory",
        cognition_gate        => "cognition-gate",
        attention_controller  => "attention-controller",
        attention_schema      => "attention-schema",
        self_model            => "self-model",
        query_vector          => "query-vector",
        query_agentic         => "query-agentic",
        memory                => "memory",
        memory_compaction     => "memory-compaction",
        predict               => "predict",
        surprise              => "surprise",
        speak_gate            => "speak-gate",
        speak                 => "speak",
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_empty() {
        assert_eq!(ModuleId::new(""), Err(ModuleIdParseError::Empty));
    }

    #[test]
    fn rejects_uppercase() {
        assert_eq!(ModuleId::new("Foo"), Err(ModuleIdParseError::InvalidChar));
    }

    #[test]
    fn rejects_underscore() {
        assert_eq!(
            ModuleId::new("attention_schema"),
            Err(ModuleIdParseError::InvalidChar)
        );
    }

    #[test]
    fn rejects_bad_hyphens() {
        assert_eq!(
            ModuleId::new("-query"),
            Err(ModuleIdParseError::InvalidChar)
        );
        assert_eq!(
            ModuleId::new("query-"),
            Err(ModuleIdParseError::InvalidChar)
        );
        assert_eq!(
            ModuleId::new("query--memory"),
            Err(ModuleIdParseError::InvalidChar)
        );
    }

    #[test]
    fn accepts_kebab_case() {
        assert_eq!(ModuleId::new("query2").unwrap().as_str(), "query2");
        assert_eq!(
            ModuleId::new("attention-schema").unwrap().as_str(),
            "attention-schema"
        );
    }

    #[test]
    fn builtins_parse() {
        let _ = builtin::cognition_gate();
        let _ = builtin::attention_controller();
        let _ = builtin::attention_schema();
        let _ = builtin::self_model();
        let _ = builtin::query_vector();
        let _ = builtin::query_agentic();
    }

    #[test]
    fn replica_cap_range_validates_order_and_v1_limit() {
        assert_eq!(
            ReplicaCapRange::new(2, 1),
            Err(ReplicaCapRangeError::MinGreaterThanMax)
        );
        assert_eq!(
            ReplicaCapRange::new(0, 3),
            Err(ReplicaCapRangeError::AboveV1Max { max: 3 })
        );
        assert_eq!(
            ReplicaCapRange::new(0, 2).unwrap(),
            ReplicaCapRange { min: 0, max: 2 }
        );
        // Always-1-active is the typical default.
        assert_eq!(
            ReplicaCapRange::new(0, 0).unwrap(),
            ReplicaCapRange { min: 0, max: 0 }
        );
    }
}
