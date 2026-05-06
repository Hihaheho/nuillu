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
        summarize             => "summarize",
        attention_controller  => "attention-controller",
        attention_schema      => "attention-schema",
        query_vector          => "query-vector",
        query_agentic         => "query-agentic",
        memory                => "memory",
        memory_compaction     => "memory-compaction",
        predict               => "predict",
        surprise              => "surprise",
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
        let _ = builtin::summarize();
        let _ = builtin::attention_controller();
        let _ = builtin::attention_schema();
        let _ = builtin::query_vector();
        let _ = builtin::query_agentic();
    }
}
