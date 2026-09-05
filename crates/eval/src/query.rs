use std::collections::BTreeSet;

use nuillu_types::ScopeId;

use crate::timeline::EvalEvent;

/// Every field mirrors an `event-selector` field in `schemas/eval-case.schema.eure`;
/// keep the two in step rather than growing filters no case can express.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct EventSelector {
    pub scopes: ScopeSelector,
    pub origin_scopes: BTreeSet<String>,
    pub modules: BTreeSet<String>,
    pub replicas: BTreeSet<u8>,
    pub variants: BTreeSet<String>,
    pub steps: BTreeSet<String>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum ScopeSelector {
    #[default]
    Any,
    Exact(BTreeSet<String>),
}

impl EventSelector {
    pub fn select<'a>(&self, timeline: &'a [EvalEvent]) -> Vec<&'a EvalEvent> {
        timeline
            .iter()
            .filter(|event| self.matches(event))
            .collect()
    }

    pub fn matches(&self, event: &EvalEvent) -> bool {
        self.matches_scope(&event.scope)
            && (self.origin_scopes.is_empty()
                || event
                    .origin_scope()
                    .is_some_and(|scope| self.origin_scopes.contains(scope)))
            && (self.modules.is_empty() || self.modules.contains(event.module.as_str()))
            && (self.replicas.is_empty() || self.replicas.contains(&event.replica))
            && (self.variants.is_empty() || self.variants.contains(event.payload.variant()))
            && (self.steps.is_empty()
                || event
                    .step
                    .as_ref()
                    .is_some_and(|step| self.steps.contains(step)))
    }

    fn matches_scope(&self, scope: &ScopeId) -> bool {
        match &self.scopes {
            ScopeSelector::Any => true,
            ScopeSelector::Exact(scopes) => scopes.contains(&scope.to_string()),
        }
    }

    pub fn measurement_scope<'a>(&self, event: &'a EvalEvent) -> Option<&'a str> {
        if self.origin_scopes.is_empty() {
            None
        } else {
            event.origin_scope()
        }
    }
}
