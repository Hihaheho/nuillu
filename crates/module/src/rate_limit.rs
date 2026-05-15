use std::collections::{HashMap, VecDeque};
use std::num::NonZeroUsize;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use nuillu_blackboard::AllocationLimits;
use nuillu_types::{ModuleId, ModuleInstanceId};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tokio::time::{Instant, sleep_until};

use crate::session_compaction::SessionCompactionPolicy;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TopicKind {
    AttentionControlRequest,
    SensoryInput,
    CognitionLogUpdated,
    AllocationUpdated,
    InteroceptiveUpdated,
    MemoUpdated,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(tag = "capability", rename_all = "snake_case")]
pub enum CapabilityKind {
    LlmCall,
    ChannelPublish { topic: TopicKind },
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RateLimitConfig {
    /// Time span used to derive burst capacity.
    ///
    /// This is not a hard sliding-window admission rule. The limiter uses a
    /// token bucket: `max_rate` is the steady-state refill rate, and `window`
    /// determines how many tokens can accumulate for short bursts.
    pub window: Duration,
    /// Steady-state events per second for this configured capability key.
    pub max_rate: f64,
}

impl RateLimitConfig {
    pub fn new(window: Duration, max_rate: f64) -> Result<Self, RateLimitPolicyError> {
        let config = Self { window, max_rate };
        config.validate()?;
        Ok(config)
    }

    fn validate(&self) -> Result<(), RateLimitPolicyError> {
        if self.window == Duration::ZERO {
            return Err(RateLimitPolicyError::ZeroWindow);
        }
        if !self.max_rate.is_finite() {
            return Err(RateLimitPolicyError::NonFiniteMaxRate);
        }
        if self.max_rate <= 0.0 {
            return Err(RateLimitPolicyError::NonPositiveMaxRate);
        }
        Ok(())
    }

    fn burst_capacity(&self) -> f64 {
        (self.max_rate * self.window.as_secs_f64()).max(1.0)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct RateLimitPolicy {
    configs: HashMap<(ModuleId, CapabilityKind), RateLimitConfig>,
}

impl RateLimitPolicy {
    pub fn disabled() -> Self {
        Self {
            configs: HashMap::new(),
        }
    }

    pub fn new(
        configs: HashMap<(ModuleId, CapabilityKind), RateLimitConfig>,
    ) -> Result<Self, RateLimitPolicyError> {
        for config in configs.values() {
            config.validate()?;
        }
        Ok(Self { configs })
    }

    pub fn for_module(
        module: ModuleId,
        kind: CapabilityKind,
        config: RateLimitConfig,
    ) -> Result<Self, RateLimitPolicyError> {
        let mut configs = HashMap::new();
        configs.insert((module, kind), config);
        Self::new(configs)
    }

    fn config_for(&self, module: &ModuleId, kind: CapabilityKind) -> Option<RateLimitConfig> {
        self.configs.get(&(module.clone(), kind)).copied()
    }
}

impl Default for RateLimitPolicy {
    fn default() -> Self {
        Self::disabled()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct RuntimePolicy {
    pub rate_limits: RateLimitPolicy,
    pub allocation_limits: AllocationLimits,
    pub memo_retained_per_owner: usize,
    pub max_concurrent_llm_calls: Option<NonZeroUsize>,
    pub session_compaction: SessionCompactionPolicy,
}

impl Default for RuntimePolicy {
    fn default() -> Self {
        Self {
            rate_limits: RateLimitPolicy::default(),
            allocation_limits: AllocationLimits::default(),
            memo_retained_per_owner: 8,
            max_concurrent_llm_calls: None,
            session_compaction: SessionCompactionPolicy::default(),
        }
    }
}

#[derive(Debug, Error, Clone, Copy, PartialEq, Eq)]
pub enum RateLimitPolicyError {
    #[error("rate-limit window must be positive")]
    ZeroWindow,
    #[error("rate-limit max_rate must be finite")]
    NonFiniteMaxRate,
    #[error("rate-limit max_rate must be positive")]
    NonPositiveMaxRate,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RateLimitOutcome {
    pub delayed_for: Duration,
}

impl RateLimitOutcome {
    pub fn immediate() -> Self {
        Self {
            delayed_for: Duration::ZERO,
        }
    }

    pub fn was_delayed(self) -> bool {
        self.delayed_for > Duration::ZERO
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ActivitySnapshot {
    pub window: Duration,
    pub max_rate: f64,
    pub event_count: usize,
    pub observed_rate: f64,
    pub available_burst: f64,
}

#[derive(Debug, Clone)]
pub struct RateLimiter {
    inner: Arc<Mutex<RateLimiterInner>>,
}

impl RateLimiter {
    pub fn disabled() -> Self {
        Self::new(RateLimitPolicy::disabled())
    }

    pub fn new(policy: RateLimitPolicy) -> Self {
        Self {
            inner: Arc::new(Mutex::new(RateLimiterInner {
                policy,
                states: HashMap::new(),
            })),
        }
    }

    pub async fn acquire(
        &self,
        owner: &ModuleInstanceId,
        kind: CapabilityKind,
    ) -> RateLimitOutcome {
        let requested_at = Instant::now();
        let mut waited = false;
        loop {
            let now = Instant::now();
            let wait_until = {
                let mut inner = self.inner.lock().expect("RateLimiter poisoned");
                let Some(config) = inner.policy.config_for(&owner.module, kind) else {
                    return RateLimitOutcome::immediate();
                };
                // Budgets are role-scoped: replicas of the same module share
                // one bucket while observations still retain the exact owner.
                let state = inner
                    .states
                    .entry((owner.module.clone(), kind))
                    .or_insert_with(|| RateLimitState::new(&config, now));
                state.refill(&config, now);

                if state.tokens >= 1.0 {
                    state.tokens -= 1.0;
                    state.events.push_back(now);
                    state.prune(&config, now);
                    return RateLimitOutcome {
                        delayed_for: if waited {
                            now.saturating_duration_since(requested_at)
                        } else {
                            Duration::ZERO
                        },
                    };
                }

                let wait = Duration::from_secs_f64((1.0 - state.tokens) / config.max_rate);
                now + wait
            };
            // A permit is not reserved while sleeping. If another task consumes
            // the refilled token first, the next loop computes a later wait.
            sleep_until(wait_until).await;
            waited = true;
        }
    }

    /// Snapshot current activity for a configured key.
    ///
    /// This is an observing API, but it normalizes internal state first by
    /// refilling the token bucket and pruning expired event timestamps. The
    /// stored timestamps are for observability only; admission is decided by
    /// the token bucket.
    pub fn snapshot(&self, module: &ModuleId, kind: CapabilityKind) -> Option<ActivitySnapshot> {
        let now = Instant::now();
        let mut inner = self.inner.lock().expect("RateLimiter poisoned");
        let config = inner.policy.config_for(module, kind)?;
        let state = inner
            .states
            .entry((module.clone(), kind))
            .or_insert_with(|| RateLimitState::new(&config, now));
        state.refill(&config, now);
        state.prune(&config, now);

        Some(ActivitySnapshot {
            window: config.window,
            max_rate: config.max_rate,
            event_count: state.events.len(),
            observed_rate: state.events.len() as f64 / config.window.as_secs_f64(),
            available_burst: state.tokens,
        })
    }
}

impl Default for RateLimiter {
    fn default() -> Self {
        Self::disabled()
    }
}

#[derive(Debug)]
struct RateLimiterInner {
    policy: RateLimitPolicy,
    states: HashMap<(ModuleId, CapabilityKind), RateLimitState>,
}

#[derive(Debug)]
struct RateLimitState {
    tokens: f64,
    last_refill: Instant,
    events: VecDeque<Instant>,
}

impl RateLimitState {
    fn new(config: &RateLimitConfig, now: Instant) -> Self {
        Self {
            tokens: config.burst_capacity(),
            last_refill: now,
            events: VecDeque::new(),
        }
    }

    fn refill(&mut self, config: &RateLimitConfig, now: Instant) {
        let elapsed = now.saturating_duration_since(self.last_refill);
        self.tokens =
            (self.tokens + elapsed.as_secs_f64() * config.max_rate).min(config.burst_capacity());
        self.last_refill = now;
    }

    fn prune(&mut self, config: &RateLimitConfig, now: Instant) {
        while let Some(&front) = self.events.front() {
            if now.saturating_duration_since(front) <= config.window {
                break;
            }
            self.events.pop_front();
        }
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use nuillu_types::{ModuleInstanceId, ReplicaIndex, builtin};
    use tokio::time::Instant;

    use super::{CapabilityKind, RateLimitConfig, RateLimitPolicy, RateLimiter, TopicKind};

    #[test]
    fn rejects_invalid_configs() {
        assert!(RateLimitConfig::new(Duration::ZERO, 1.0).is_err());
        assert!(RateLimitConfig::new(Duration::from_secs(1), 0.0).is_err());
        assert!(RateLimitConfig::new(Duration::from_secs(1), f64::NAN).is_err());
    }

    #[tokio::test]
    async fn disabled_policy_grants_immediately() {
        let limiter = RateLimiter::disabled();
        let owner = ModuleInstanceId::new(builtin::query_memory(), ReplicaIndex::ZERO);

        let outcome = limiter.acquire(&owner, CapabilityKind::LlmCall).await;

        assert_eq!(outcome.delayed_for, Duration::ZERO);
    }

    #[tokio::test]
    async fn delays_when_next_event_would_exceed_rate() {
        let owner = ModuleInstanceId::new(builtin::query_memory(), ReplicaIndex::ZERO);
        let policy = RateLimitPolicy::for_module(
            owner.module.clone(),
            CapabilityKind::LlmCall,
            RateLimitConfig::new(Duration::from_millis(10), 100.0).unwrap(),
        )
        .unwrap();
        let limiter = RateLimiter::new(policy);

        let first = limiter.acquire(&owner, CapabilityKind::LlmCall).await;
        let started = Instant::now();
        let second = limiter.acquire(&owner, CapabilityKind::LlmCall).await;

        assert_eq!(first.delayed_for, Duration::ZERO);
        assert!(second.was_delayed());
        assert!(started.elapsed() >= Duration::from_millis(8));
    }

    #[tokio::test]
    async fn shared_module_budget_counts_multiple_replicas() {
        let owner_0 = ModuleInstanceId::new(builtin::query_memory(), ReplicaIndex::ZERO);
        let owner_1 = ModuleInstanceId::new(builtin::query_memory(), ReplicaIndex::new(1));
        let policy = RateLimitPolicy::for_module(
            owner_0.module.clone(),
            CapabilityKind::ChannelPublish {
                topic: TopicKind::AttentionControlRequest,
            },
            RateLimitConfig::new(Duration::from_millis(10), 100.0).unwrap(),
        )
        .unwrap();
        let limiter = RateLimiter::new(policy);

        let first = limiter
            .acquire(
                &owner_0,
                CapabilityKind::ChannelPublish {
                    topic: TopicKind::AttentionControlRequest,
                },
            )
            .await;
        let second = limiter
            .acquire(
                &owner_1,
                CapabilityKind::ChannelPublish {
                    topic: TopicKind::AttentionControlRequest,
                },
            )
            .await;

        assert_eq!(first.delayed_for, Duration::ZERO);
        assert!(second.was_delayed());
    }

    #[tokio::test]
    async fn topic_keys_are_independent() {
        let owner = ModuleInstanceId::new(builtin::query_memory(), ReplicaIndex::ZERO);
        let policy = RateLimitPolicy::for_module(
            owner.module.clone(),
            CapabilityKind::ChannelPublish {
                topic: TopicKind::AttentionControlRequest,
            },
            RateLimitConfig::new(Duration::from_millis(10), 100.0).unwrap(),
        )
        .unwrap();
        let limiter = RateLimiter::new(policy);

        limiter
            .acquire(
                &owner,
                CapabilityKind::ChannelPublish {
                    topic: TopicKind::AttentionControlRequest,
                },
            )
            .await;
        let outcome = limiter
            .acquire(
                &owner,
                CapabilityKind::ChannelPublish {
                    topic: TopicKind::SensoryInput,
                },
            )
            .await;

        assert_eq!(outcome.delayed_for, Duration::ZERO);
    }
}
