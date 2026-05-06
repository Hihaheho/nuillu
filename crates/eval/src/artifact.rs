use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

/// A normalized artifact produced by a case driver.
///
/// Drivers should put the primary user-visible answer in `output`, and put
/// structured runtime state under `observations` so deterministic JSON-pointer
/// checks and rubric judges can inspect it.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct CaseArtifact {
    pub output: String,
    pub observations: BTreeMap<String, serde_json::Value>,
    pub failure: Option<String>,
}

impl CaseArtifact {
    pub fn new(output: impl Into<String>) -> Self {
        Self {
            output: output.into(),
            observations: BTreeMap::new(),
            failure: None,
        }
    }

    pub fn failed(message: impl Into<String>) -> Self {
        Self {
            output: String::new(),
            observations: BTreeMap::new(),
            failure: Some(message.into()),
        }
    }

    pub fn with_observation(
        mut self,
        key: impl Into<String>,
        value: impl Into<serde_json::Value>,
    ) -> Self {
        self.observations.insert(key.into(), value.into());
        self
    }

    pub fn as_json(&self) -> serde_json::Value {
        serde_json::to_value(self).unwrap_or_else(|error| {
            serde_json::json!({
                "output": self.output,
                "failure": self.failure,
                "observations": {},
                "serialization_error": error.to_string(),
            })
        })
    }
}
