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

use std::{future::Future, pin::Pin, time::Duration};

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
#[derive(Debug, Clone, PartialEq)]
pub struct PersistedCognitionLogEntry {
    pub source: ModuleInstanceId,
    pub entry: CognitionLogEntry,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PersistedCognitionLogPageEntry {
    /// Repository-local, monotonically increasing identity.
    pub id: i64,
    pub source: ModuleInstanceId,
    pub entry: CognitionLogEntry,
}

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
    async fn recent(&self, limit: usize) -> Result<Vec<PersistedCognitionLogEntry>, PortError>;
    /// Returns newest-first history for visualizer-style incremental loading.
    async fn page_desc(
        &self,
        offset: usize,
        limit: usize,
    ) -> Result<Vec<PersistedCognitionLogPageEntry>, PortError> {
        let requested = offset.saturating_add(limit);
        let mut entries = self.recent(requested).await?;
        entries.reverse();
        Ok(entries
            .into_iter()
            .skip(offset)
            .take(limit)
            .enumerate()
            .map(|(index, record)| PersistedCognitionLogPageEntry {
                id: -i64::try_from(offset.saturating_add(index).saturating_add(1))
                    .unwrap_or(i64::MAX),
                source: record.source,
                entry: record.entry,
            })
            .collect())
    }
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

/// Monotonic time and asynchronous sleeping supplied by the runtime host.
///
/// Keeping this separate from [`Clock`] lets wasm hosts drive waits from the
/// JavaScript event loop while retaining wall-clock timestamps for durable
/// records.
pub trait Timer {
    /// Sleeps for `duration`.
    fn sleep(&self, duration: Duration) -> Pin<Box<dyn Future<Output = ()> + 'static>>;

    /// Monotonic time since this timer was created.
    fn elapsed(&self) -> Duration;
}

/// Returned when [`timeout`] reaches its deadline before the wrapped future.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("deadline elapsed")]
pub struct Elapsed;

/// Awaits `future` until it completes or the timer reaches `duration`.
pub async fn timeout<F: Future>(
    timer: &dyn Timer,
    duration: Duration,
    future: F,
) -> Result<F::Output, Elapsed> {
    let future = Box::pin(future);
    let sleep = timer.sleep(duration);
    match futures::future::select(future, sleep).await {
        futures::future::Either::Left((output, _)) => Ok(output),
        futures::future::Either::Right(((), _)) => Err(Elapsed),
    }
}

/// Native timer backed by Tokio's time driver.
#[derive(Debug, Clone)]
pub struct TokioTimer {
    started_at: tokio::time::Instant,
}

impl TokioTimer {
    pub fn new() -> Self {
        Self {
            started_at: tokio::time::Instant::now(),
        }
    }
}

impl Default for TokioTimer {
    fn default() -> Self {
        Self::new()
    }
}

impl Timer for TokioTimer {
    fn sleep(&self, duration: Duration) -> Pin<Box<dyn Future<Output = ()> + 'static>> {
        Box::pin(tokio::time::sleep(duration))
    }

    fn elapsed(&self) -> Duration {
        self.started_at.elapsed()
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

/// Clock frozen at a fixed instant, with sleeps returning immediately. Use it
/// where an activation must observe a scripted timestamp instead of wall time.
#[derive(Debug, Clone, Copy)]
pub struct FixedClock(DateTime<Utc>);

impl FixedClock {
    pub fn new(now: DateTime<Utc>) -> Self {
        Self(now)
    }
}

#[async_trait(?Send)]
impl Clock for FixedClock {
    fn now(&self) -> DateTime<Utc> {
        self.0
    }

    async fn sleep_until(&self, _deadline: DateTime<Utc>) {}
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

    async fn recent(&self, _limit: usize) -> Result<Vec<PersistedCognitionLogEntry>, PortError> {
        Ok(Vec::new())
    }
}
