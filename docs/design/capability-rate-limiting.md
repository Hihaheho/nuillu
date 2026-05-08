# Capability Rate Limiting

Date: 2026-05-08
Scope: Deterministic runtime guardrails for capability use in the current
`CapabilityProviders` / replica runtime.

This document describes desired architecture, not implementation progress.

---

## 0. Decisions

1. **Runtime guardrail, not a cognitive module** — Rate limiting lives in the
   capability/runtime layer. It is injected through `CapabilityProviders` and is
   not registered in `ModuleRegistry`.
2. **Allocation remains the adaptive policy** — The attention controller owns
   semantic resource allocation through `activation_ratio`, `guidance`, and
   `tier`. Rate limiting does not write allocation proposals and does not change
   `ResourceAllocation`.
3. **Deterministic intervention** — Rate limiting only delays capability
   acquisition or publish operations in v1. It does not ask an LLM for policy,
   does not deny LLM handles, and does not add module-local throttling logic.
4. **Capability-boundary only** — Intervention points are
   `LlmAccess::lutum().await` and `TopicMailbox::publish(...).await`.
   `next_batch()` / `activate()` signatures and module constructors do not
   receive rate-limit state.
5. **No periodic or boost semantics** — The current runtime has no
   `PeriodicActivation`, period, or scheduler tick API. Low-activity recovery is
   handled by attention-controller allocation guidance, not by a rate-limit
   `min_rate` or boost factor.
6. **Async delay, not background send** — `publish()` is already async, so a
   publish permit is awaited directly before routing. The design does not add an
   internal deferred-send task or hidden background queue.
7. **No-op is the baseline** — A disabled policy preserves current behavior and
   is the baseline for tests and ablation studies.

---

## 1. Purpose

The attention controller is an adaptive cognitive policy: it decides which
module roles should be active, what tier they should use, and what guidance they
should follow. That policy can still over-activate a role, react late to a burst,
or allow a module interaction loop to produce too many LLM acquisitions or
channel publishes.

Rate limiting is the deterministic safety layer underneath that policy. It
limits the rate at which capability side effects can enter the runtime, while
leaving module code and allocation semantics intact.

Primary risks addressed:

- runaway LLM acquisition by an over-active module role,
- publish storms that repeatedly wake downstream modules,
- controller lag during bursts, where delaying a capability use gives allocation
  changes time to catch up,
- eval and production observability gaps around throttling behavior.

Non-goals for v1:

- changing `ResourceAllocation` or adding allocation fields,
- cancelling or failing module activations,
- enforcing provider-request budgets inside `lutum` sessions or tool loops,
- introducing bounded mailbox queues or overflow/drop semantics,
- adding rate-limit logic to individual modules.

---

## 2. Core Abstraction

The intended implementation lives in `crates/module/src/rate_limit.rs`.

```rust
pub enum CapabilityKind {
    LlmCall,
    ChannelPublish(TopicKind),
}

pub enum TopicKind {
    Query,
    SelfModel,
    Speak,
    SensoryInput,
    SensoryDetailRequest,
    MemoryRequest,
    AttentionStreamUpdated,
    AllocationUpdated,
    MemoUpdated,
}

pub struct RateLimitConfig {
    pub window: Duration, // burst span, not a hard sliding-window rule
    pub max_rate: f64,    // steady-state events/sec
}

pub struct RateLimitPolicy {
    pub configs: HashMap<(ModuleId, CapabilityKind), RateLimitConfig>,
}

pub struct RateLimiter { /* token bucket state per configured key */ }

impl RateLimiter {
    pub fn disabled() -> Self;
    pub fn new(policy: RateLimitPolicy) -> Self;

    pub async fn acquire(
        &self,
        owner: &ModuleInstanceId,
        kind: CapabilityKind,
    ) -> RateLimitOutcome;

    pub fn snapshot(
        &self,
        module: &ModuleId,
        kind: CapabilityKind,
    ) -> Option<ActivitySnapshot>;
}
```

Rate limits are keyed by `ModuleId` by default, not by exact
`ModuleInstanceId`. Replicas of the same module role therefore share the same
role budget. The exact owner instance is still included in observations and
runtime events.

`RateLimiter` must not hold a mutable borrow or lock across `.await`. A typical
implementation computes the wait deadline while holding interior state, releases
that state, awaits `tokio::time::sleep_until(deadline)`, then records the
granted event.

Because the current runtime is `LocalSet` / `?Send` oriented, either `Rc`-based
or `Arc`-based sharing is acceptable if the chosen state container matches the
rest of the runtime. The implementation should avoid adding `Send` requirements
to modules or capabilities.

---

## 3. Intervention Points

### LLM acquisition

`LlmAccess::lutum().await` acquires the permit before resolving allocation tier
and before emitting `RuntimeEvent::LlmAccessed`.

```rust
pub async fn lutum(&self) -> Lutum {
    self.rate_limiter
        .acquire(&self.owner, CapabilityKind::LlmCall)
        .await;

    let cfg = self
        .blackboard
        .read(|bb| bb.allocation().for_module(&self.owner.module))
        .await;
    self.events.llm_accessed(self.owner.clone(), cfg.tier).await;
    self.tiers.pick(cfg.tier)
}
```

This preserves the existing meaning of `LlmAccessed`: it is emitted when the
handle is actually acquired, not when a module first asks and is delayed. It also
means tier changes during the delay are respected.

The LLM limit is an acquisition limit only. It does not count provider requests
inside a `lutum::Session`, streamed chunks, tool rounds, or adapter retries.
Those budgets remain owned by `lutum` and its budget managers.

### Topic publish

`TopicMailbox::publish(...).await` acquires the permit before reading allocation
and routing to active replicas.

```rust
pub async fn publish(&self, body: T) -> Result<usize, Envelope<T>> {
    self.rate_limiter
        .acquire(&self.owner, CapabilityKind::ChannelPublish(self.topic_kind))
        .await;

    // Build envelope, read current allocation, then route.
}
```

Routing must happen after the delay. If allocation changes while the publisher
waits, the publish is delivered according to the active replicas at actual send
time. This is intentional: one purpose of delay is to let controller allocation
catch up before more work is routed.

No background send queue is introduced in v1. The publishing task itself waits
for the permit, then performs normal topic routing.

---

## 4. Rate Calculation

For each configured `(ModuleId, CapabilityKind)` key, the limiter keeps a token
bucket:

```text
capacity = max(1.0, max_rate * window.as_secs_f64())
refill   = elapsed_secs * max_rate
grant    = consume 1 token
```

`max_rate` defines the steady-state rate. `window` defines the allowed burst
capacity at that rate. If no token is available, the limiter computes the
earliest monotonic deadline at which one token will be refilled and awaits that
deadline. A permit is not reserved while sleeping: under contention, another
task may consume the token first, and the waiter will compute a later deadline on
the next loop. This is acceptable for v1 because the limiter is a deterministic
safety guardrail, not a fairness scheduler.

The limiter may also keep recent granted event timestamps for snapshots and
observability:

```text
prune events older than now - window
observed_rate = granted_count / window.as_secs_f64()
```

Those timestamps do not decide admission. `snapshot()` may prune expired
timestamps and refill the token bucket before returning a view; this mutates only
limiter bookkeeping so the observation is current.

The monotonic clock for this calculation is `tokio::time::Instant`, matching the
scheduler's existing idle-deadline use. The injected `Clock` remains the source
for module-visible timestamps and human/semantic time; it is not used for rate
windows.

Configs must reject invalid values:

- `window == Duration::ZERO`,
- non-finite `max_rate`,
- `max_rate <= 0.0`.

An absent config is no-op for that key.

---

## 5. Runtime Policy Injection

`CapabilityProviders::new()` and `new_with_runtime_events()` should keep current
behavior by installing a disabled limiter.

An additional constructor or policy builder should accept rate-limit policy:

```rust
pub struct RuntimePolicy {
    pub rate_limits: RateLimitPolicy,
}

impl CapabilityProviders {
    pub fn new_with_runtime_policy(
        /* existing boot ports */,
        runtime_event_sink: Arc<dyn RuntimeEventSink>,
        policy: RuntimePolicy,
    ) -> Self;
}
```

`ModuleCapabilityFactory` then passes the shared limiter into owner-stamped
capabilities when issuing `LlmAccess` and `TopicMailbox`.

The rate limiter is a runtime policy knob, not a capability granted to modules.
Modules cannot inspect it, change it, or bypass it except by not holding the
capability in question.

---

## 6. Observability And Ablation

Rate limiting should be observable without changing module power. Extend runtime
events with delay observations:

```rust
pub enum RuntimeEvent {
    RateLimitDelayed {
        sequence: u64,
        owner: ModuleInstanceId,
        capability: CapabilityKind,
        delayed_for: Duration,
    },
}
```

The event is emitted after a delayed permit is granted. Immediate permits do not
need events in v1; snapshots and aggregate metrics can report normal activity.

Ablation studies should vary only injected runtime policy while keeping module
registration, allocation controller, prompts, and eval cases constant:

- no limiter,
- LLM acquisition only,
- channel publish only,
- LLM plus channel publish,
- topic-specific channel policy,
- loose versus strict windows and rates.

Useful metrics:

- total `LlmAccessed` events,
- total and per-owner delay count,
- cumulative delayed duration,
- first completed utterance latency,
- eval rubric score / success,
- allocation update count,
- agentic deadlock marker count.

---

## 7. Mailbox Backpressure Follow-Up

Current typed topics use unbounded per-subscriber queues. V1 rate limiting does
not change queue capacity or overflow behavior; it only delays before routing.

Bounded mailboxes are a separate runtime policy because overflow semantics differ
by topic:

- Wake-only topics (`AttentionStreamUpdated`, `MemoUpdated`,
  `AllocationUpdated`) may coalesce redundant queued wakes because durable state
  is the source of truth.
- Work-carrying topics (`QueryRequest`, `SelfModelRequest`, `SpeakRequest`,
  `SensoryInput`, `SensoryDetailRequest`, `MemoryRequest`) must not silently
  drop by default because the payload is the work.
- If a work-carrying topic is made lossy, that must be explicit per topic, traced
  as a runtime event, and justified by the caller's semantics.

This follow-up can satisfy bounded-capacity requirements without complicating
the v1 rate limiter with background queues or hidden task ownership.

---

## 8. Tests

Unit tests for `RateLimiter`:

- `disabled_policy_grants_immediately`,
- `missing_config_grants_immediately`,
- `rejects_zero_window`,
- `rejects_non_positive_max_rate`,
- `prunes_old_events_from_window`,
- `delays_when_next_event_would_exceed_rate`,
- `shared_module_budget_counts_multiple_replicas`,
- `topic_keys_are_independent`.

Capability integration tests:

- `llm_access_waits_before_runtime_event`,
- `llm_access_uses_tier_at_grant_time`,
- `publish_waits_before_allocation_read`,
- `publish_routes_to_active_replicas_after_delay`,
- `no_op_policy_preserves_existing_publish_behavior`.

Regression tests that must continue passing:

- role fanout and role load-balance routing,
- disabled replicas receive no newly routed messages,
- memo update inbox filters self writes,
- allocation writer publishes only effective changes,
- scheduler does not start inactive replicas,
- eval `max-llm-calls` still counts `LlmAccessed` acquisitions after delay.

---

## 9. Invariants

| Invariant | Enforcement |
|---|---|
| Rate limiting is not a module | installed through `CapabilityProviders`, not `ModuleRegistry` |
| Allocation remains semantic policy | limiter never writes `ResourceAllocation` or controller proposals |
| Module code stays unaware | limiter is held inside capability handles only |
| Capabilities remain non-exclusive | shared limiter does not enforce handle uniqueness |
| Owner identity cannot be forged | permits use the owner-stamped `ModuleInstanceId` already captured by the capability |
| Replica budgets are role-scoped by default | rate keys use `ModuleId`, while events retain `ModuleInstanceId` |
| No periodic activation is reintroduced | no `tick`, `period`, `min_rate`, or boost factor |
| LLM acquisition is delayed, not denied | `lutum()` still returns `Lutum` and emits `LlmAccessed` after permit |
| Publish is delayed, not backgrounded | `publish().await` waits before routing and owns the send path |
| Disabled policy is current behavior | default constructors install no-op policy |
| Module-visible time remains injected | rate windows use monotonic runtime time; `Clock` remains for timestamps |
