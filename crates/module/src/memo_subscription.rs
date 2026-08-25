use std::collections::BTreeSet;
use std::sync::Arc;

use nuillu_types::ModuleId;

/// Immutable boot-time scope for memo update notifications and memo-log reads.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum MemoSubscription {
    /// Receive memos from every module role.
    #[default]
    All,
    /// Receive memos only from the listed module roles.
    Only(Arc<BTreeSet<ModuleId>>),
}

impl MemoSubscription {
    pub fn only(sources: impl IntoIterator<Item = ModuleId>) -> Self {
        Self::Only(Arc::new(sources.into_iter().collect()))
    }

    pub fn accepts(&self, source: &ModuleId) -> bool {
        match self {
            Self::All => true,
            Self::Only(sources) => sources.contains(source),
        }
    }

    pub fn sources(&self) -> Option<&BTreeSet<ModuleId>> {
        match self {
            Self::All => None,
            Self::Only(sources) => Some(sources),
        }
    }
}
