use std::collections::HashMap;
use std::sync::Arc;

use nuillu_types::ScopeId;

/// Boot-time human-facing labels for expanded subsystem scopes.
///
/// Keys are stable scope IDs; values contain only the display segment for the
/// final subsystem instance in that scope (for example `Arm 1`).
#[derive(Clone, Debug, Default)]
pub struct ScopeLabels {
    segments: HashMap<ScopeId, Arc<str>>,
}

impl ScopeLabels {
    pub fn new(segments: impl IntoIterator<Item = (ScopeId, Arc<str>)>) -> Self {
        Self {
            segments: segments.into_iter().collect(),
        }
    }

    pub fn label(&self, scope: &ScopeId) -> Option<String> {
        self.relative_descendant_label(&ScopeId::root(), scope)
    }

    /// Formats a descendant scope relative to the activating module's scope.
    /// Returns `None` for the same scope, an outer scope, or an unknown scope.
    pub fn relative_descendant_label(&self, current: &ScopeId, origin: &ScopeId) -> Option<String> {
        let relative = origin.path().strip_prefix(current.path())?;
        if relative.is_empty() {
            return None;
        }

        let mut scope = current.clone();
        let mut labels = Vec::with_capacity(relative.len());
        for instance in relative {
            scope = scope.child(instance.clone());
            labels.push(self.segments.get(&scope)?.as_ref());
        }
        Some(labels.join(" / "))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nuillu_types::{ReplicaIndex, SubsystemId, SubsystemInstanceId};

    fn child(scope: &ScopeId, subsystem: &str, replica: u8) -> ScopeId {
        scope.child(SubsystemInstanceId::new(
            SubsystemId::new(subsystem).unwrap(),
            ReplicaIndex::new(replica),
        ))
    }

    #[test]
    fn relative_labels_include_only_descendant_segments() {
        let root = ScopeId::root();
        let arm = child(&root, "arm", 0);
        let finger = child(&arm, "finger", 1);
        let labels = ScopeLabels::new([
            (arm.clone(), Arc::from("Arm")),
            (finger.clone(), Arc::from("Finger 2")),
        ]);

        assert_eq!(
            labels.relative_descendant_label(&root, &finger),
            Some("Arm / Finger 2".to_string())
        );
        assert_eq!(labels.label(&finger), Some("Arm / Finger 2".to_string()));
        assert_eq!(
            labels.relative_descendant_label(&arm, &finger),
            Some("Finger 2".to_string())
        );
        assert_eq!(labels.relative_descendant_label(&finger, &finger), None);
        assert_eq!(labels.relative_descendant_label(&finger, &root), None);
    }
}
