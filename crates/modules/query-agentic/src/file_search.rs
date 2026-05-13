//! `FileSearchProvider` port + the `FileSearcher` capability handle.
//!
//! Only the query-agentic module consumes file search, so the trait and the
//! capability that wraps it live in this crate. Adapters provide concrete
//! implementations of `FileSearchProvider` and the host injects them into the
//! query-agentic registration closure.

use std::sync::Arc;

use async_trait::async_trait;
use nuillu_module::ports::PortError;

#[async_trait(?Send)]
pub trait FileSearchProvider {
    async fn search(&self, query: &FileSearchQuery) -> Result<Vec<FileSearchHit>, PortError>;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FileSearchQuery {
    pub pattern: String,
    pub regex: bool,
    pub invert_match: bool,
    pub case_sensitive: bool,
    pub context: usize,
    pub max_matches: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FileSearchHit {
    pub path: String,
    pub line: usize,
    pub snippet: String,
}

/// File search provider that returns no hits.
#[derive(Debug, Default)]
pub struct NoopFileSearchProvider;

#[async_trait(?Send)]
impl FileSearchProvider for NoopFileSearchProvider {
    async fn search(&self, _query: &FileSearchQuery) -> Result<Vec<FileSearchHit>, PortError> {
        Ok(Vec::new())
    }
}

/// Read-only ripgrep-like file search capability.
#[derive(Clone)]
pub struct FileSearcher {
    search: Arc<dyn FileSearchProvider>,
}

impl FileSearcher {
    pub fn new(search: Arc<dyn FileSearchProvider>) -> Self {
        Self { search }
    }

    pub async fn search(
        &self,
        mut query: FileSearchQuery,
    ) -> Result<Vec<FileSearchHit>, PortError> {
        query.context = query.context.min(8);
        query.max_matches = query.max_matches.clamp(1, 256);
        self.search.search(&query).await
    }
}
