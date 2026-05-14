//! Outbound traits used by capability handles.
//!
//! Adapters provide concrete implementations of these traits and inject
//! them into the [`CapabilityProviders`] at boot; modules should prefer the
//! capability handles that wrap them.
//!
//! All async traits use `?Send` so the agent can run on a single-threaded
//! runtime (current-thread tokio / wasm32) without requiring `Send` bounds.
//!
//! Domain-specific ports live in their owning crates:
//! `MemoryStore` (and the memory capabilities) in `nuillu-memory`,
//! `PolicyStore` (and policy capabilities) in `nuillu-reward`, and
//! `UtteranceSink` in `nuillu-speak`.
//!
//! [`CapabilityProviders`]: crate::CapabilityProviders

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use nuillu_blackboard::CognitionLogEntry;
use nuillu_types::ModuleInstanceId;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum PortError {
    #[error("storage backend did not find requested resource: {0}")]
    NotFound(String),
    #[error("invalid port input: {0}")]
    InvalidInput(String),
    #[error("storage backend returned invalid data: {0}")]
    InvalidData(String),
    #[error("storage backend reported: {0}")]
    Backend(String),
}

/// Text embedding provider used by vector-capable memory adapters.
#[async_trait(?Send)]
pub trait Embedder {
    fn dimensions(&self) -> usize;
    async fn embed(&self, text: &str) -> Result<Vec<f32>, PortError>;
}

/// Append-only persistence for the cognition log.
#[async_trait(?Send)]
pub trait CognitionLogRepository {
    async fn append(
        &self,
        source: ModuleInstanceId,
        entry: CognitionLogEntry,
    ) -> Result<(), PortError>;
    async fn since(
        &self,
        source: &ModuleInstanceId,
        from: DateTime<Utc>,
    ) -> Result<Vec<CognitionLogEntry>, PortError>;
}

/// Time source plus sleep. Indirected so tests can fully inject time —
/// `sleep_until` lets the scheduler wait for a virtual deadline without
/// blocking real time in tests.
#[async_trait(?Send)]
pub trait Clock {
    fn now(&self) -> DateTime<Utc>;

    /// Sleep until the given absolute deadline. Implementations should return
    /// immediately if the deadline is already in the past.
    async fn sleep_until(&self, deadline: DateTime<Utc>);

    /// Sleep for the given duration. Default impl computes the deadline via
    /// `now()` and delegates to `sleep_until`.
    async fn sleep_for(&self, duration: std::time::Duration) {
        let deadline = self.now() + chrono::Duration::from_std(duration).unwrap_or_default();
        self.sleep_until(deadline).await;
    }
}

/// System clock: adequate default for non-test use.
#[derive(Debug, Clone, Copy, Default)]
pub struct SystemClock;

#[async_trait(?Send)]
impl Clock for SystemClock {
    fn now(&self) -> DateTime<Utc> {
        Utc::now()
    }

    async fn sleep_until(&self, deadline: DateTime<Utc>) {
        let remaining = deadline - Utc::now();
        let Ok(duration) = remaining.to_std() else {
            return;
        };
        if duration.is_zero() {
            return;
        }
        tokio::time::sleep(duration).await;
    }
}

/// Cognition-log repository that discards appends and reports no history.
#[derive(Debug, Default)]
pub struct NoopCognitionLogRepository;

#[async_trait(?Send)]
impl CognitionLogRepository for NoopCognitionLogRepository {
    async fn append(
        &self,
        _source: ModuleInstanceId,
        _entry: CognitionLogEntry,
    ) -> Result<(), PortError> {
        Ok(())
    }

    async fn since(
        &self,
        _source: &ModuleInstanceId,
        _from: DateTime<Utc>,
    ) -> Result<Vec<CognitionLogEntry>, PortError> {
        Ok(Vec::new())
    }
}
