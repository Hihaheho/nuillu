# Implementation Design

Date: 2026-05-07
Scope: Runtime and capability implementation for `attention-schema.md` on top of `lutum`.

This document is the implementation source of truth. It describes the desired architecture, not work progress.

---

## 0. Decisions

1. **Runtime shape** — The scheduler/event loop starts `next_batch()` for active module replicas, runs `activate(&batch)` with configured retry, and waits for shutdown on a single-threaded `LocalSet`.
2. **Single-thread / WASM-compatible futures** — Module futures use `#[async_trait(?Send)]`; the runtime is current-thread / `LocalSet` oriented.
3. **Capability-based design** — A module's possible side effects are exactly the capability handles passed to its constructor. Without a capability, there is no API path to the operation.
4. **Owner-stamped operations** — Identity-bearing capabilities bake a hidden `ModuleInstanceId = (ModuleId, replica)` in at construction. Module constructors receive only a replica-scoped capability factory, so modules cannot claim to send, memo-write, append, or request as another module instance.
5. **Typed channels, not generic payloads** — Module communication uses typed channel capabilities. There is no central `MailboxPayload`, no `serde_json::Value` mailbox, and no request/response correlation protocol.
6. **Memo-authoritative results** — Query-module and self-model answers are written to the producing module's indexed `Memo` queue. Channel messages wake modules or submit work; they are not durable output.
7. **Kebab-case module ids** — `ModuleId(String)` accepts only `^[a-z][a-z0-9]*(?:-[a-z0-9]+)*$`. Builtins live in `nuillu_types::builtin::*`.
8. **Non-exclusive capabilities** — Root providers and replica-scoped capability factories issue handles without uniqueness checks; any capability may be granted to multiple module instances. Single-writer roles are upheld by boot-time wiring and replica-owned state, not by issuer enforcement.
9. **No periodic activation** — Modules wake from typed channels only. Controller guidance in `ResourceAllocation` replaces period/cadence fields; modules decide from allocation guidance whether an allocation wake should produce work or remain silent.
10. **Sensory/action boundaries** — The only app-facing external input is `SensoryInput`, consumed by the `sensory` module. Full-agent runs publish observations through that boundary and collect user-visible text from `speak` / `UtteranceSink`. Internal evidence requests enter through `AttentionControlRequest`, consumed only by allocation-controller.
11. **Injectable time** — All module-visible current time comes from an injected `Clock` capability. Production boot uses `SystemClock`; eval/sandbox boot can pass a fixed or scripted clock. Capabilities and modules must not call `Utc::now()` directly.
12. **Streaming for user-visible output, collect for internal work** — The `speak` module uses `.stream()` for LLM text generation so that `UtteranceWriter` can emit progressive deltas to `UtteranceSink` before the turn completes. All other modules use `.collect()` for control decisions, tool loops, and complete free-form notes written through `Memo`. Streaming does not remove the final memo write; it adds progressive forwarding at the utterance boundary only.
13. **Activation-gated speech** — Cognition-log updates wake Speak. Before Speak activates, the runtime sends the pending Speak batch to active `ActivationGate<SpeakModule>` holders such as SpeakGate and waits for allow/suppress votes.
14. **Local deterministic inbox batching** — Modules may batch transient inbox activations immediately before LLM work. Batching is module-local, bounded by boot-time module registration, and deterministic; it does not add a shared runtime batch type or ask an LLM to decide batch membership.
15. **Replica-capped persistent module instances** — Boot registers modules as `(module_id, cap_range, builder)`, creates module instances up to `cap_range.max`, and never destroys those instances when allocation lowers replica count. Allocation changes routing and event-loop scheduling.
16. **Controller proposals with deterministic effective allocation** — Allocation-controller replicas write allocation proposals. The runtime derives the effective `ResourceAllocation` by deterministic averaging of `activation_ratio`, `guidance`, and `tier`, computes active replicas from each module's boot-time cap range, then applies `RuntimePolicy` hard limits such as max total active replicas and max Premium replicas.
17. **Registry-derived controller schema** — The allocation-controller structured-output JSON Schema is generated from module registrations, enumerates module ids, and exposes only `activation_ratio`, `guidance`, and `tier`. Parsed ratios are still clamped to `0.0..=1.0` because LLM output is not a trust boundary.
18. **Free-form memo logs** — `Memo` appends durable module-output log entries. Each entry is plain free-form text, not JSON/YAML, a code-fenced format, or a structured data exchange protocol. Modules consume this surface through unread memo logs and keep any needed durable context in their own persistent `Session` plus compaction; there is no latest-memo snapshot read API. Structured output is reserved for runtime control decisions whose fields are read by code.
19. **Memory and learning are distinct reinforcement substrates** — Memory preserves *what happened* and is reinforced by **access**: rank rises when entries are read or queried, and this rule is owned by `MemoryStore` rather than by any module. Learning preserves *what worked* and is reinforced by **reward** through a TD-0 (temporal-difference, one-step lookbehind, no discount) credit-assignment loop split actor/critic style: `policy` proposes new behaviors as `(trigger, behavior)` pairs; `query-policy` retrieves candidates by vector search over the `trigger` field only; `value-estimator` predicts each retrieved policy's `expected_reward` in the current context (critic); `reward` aggregates the 6-source `ObservedReward`, computes `td_error = observed_reward − expected_reward`, and applies value, expected-reward, and confidence deltas through `PolicyValueUpdater::reinforce`. Rank crosses tiers only as a derived consequence of value × reward-tokens. Policies and memories never share a store, a rank enum, or a strengthening rule. Creating new policy entries belongs to `policy`; predicting value belongs to `value-estimator`; updating value, expected reward, confidence, rank, and reward tokens belongs to `reward`; retrieval belongs to `query-policy`. The retrieval module that surfaces memories is named `query-memory` (the role) rather than `query-vector` (the implementation); the underlying crate, tool ids, and `VectorMemorySearcher` capability keep their names for now and are renamed in a follow-up implementation pass.

---

## 1. Crate Layout

```
crates/
  types/                      # newtypes only (ModuleId, ReplicaIndex, MemoryRank, ModelTier, ...)
  blackboard/                 # Blackboard state + commands
  module/                     # Module trait, port traits, capability handles, typed channels, capability issuers
  modules/
    sensory/                  # observations -> deterministic salience + LLM-filtered memo logs
    cognition-gate/           # non-cognitive snapshot -> cognition log
    allocation-controller/     # cognition log -> resource allocation
    attention-schema/         # memo logs/allocation/cognition log -> first-person attention cognition-log entries
    self-model/               # attention-schema cognition log + memo logs -> self-model memo logs
    query-memory/             # blackboard/vector memory RAG -> query-memory memo logs (crate dir still query-vector/ in v1)
    query-policy/             # blackboard/policy store search -> query-policy memo logs
    memory/                   # blackboard snapshot -> memory inserts (access-reinforced; rank elevation owned by MemoryStore)
    memory-compaction/        # memory metadata/content -> merges
    policy/                   # blackboard snapshot -> tentative (trigger, behavior) policy inserts
    value-estimator/          # query-policy memo + cognition context -> expected_reward predictions (critic)
    reward/                   # ObservedReward + value-estimator memo -> TD-error -> policy value/rank/confidence updates
    predict/                  # cognition log + blackboard -> forward prediction memo logs
    surprise/                 # cognition log divergence/novelty -> surprise memo logs
    speak/                    # attended state + memo logs -> user-visible utterances
  agent/                      # scheduler
  adapters/                   # concrete persistence/search adapters
  app/                        # boot wiring
```

Modules each live in their own crate. Their constructor signatures are the role boundary: boot wiring grants only the capabilities a module is allowed to hold. Replica identity is not passed to constructors; it is captured by the replica-scoped capability factory used to issue owner-stamped handles.

---

## 2. Core Abstractions

### `Module`

```rust
#[async_trait(?Send)]
pub trait Module {
    type Batch: 'static;

    async fn next_batch(&mut self) -> anyhow::Result<Self::Batch>;
    async fn activate(&mut self, batch: &Self::Batch) -> anyhow::Result<()>;
}
```

Modules own inbox capabilities and decide how to form a deterministic module-local batch. They do not own a persistent run loop; the agent event loop awaits `next_batch`, keeps the batch, and invokes `activate(&batch)`.

### Module identity and registration

`ModuleId` identifies a role. `ModuleInstanceId` identifies one persistent replica of that role:

```rust
pub struct ReplicaIndex(u8);

pub struct ReplicaCapRange {
    pub min: u8,
    pub max: u8,
}

pub struct ModuleInstanceId {
    pub module: ModuleId,
    pub replica: ReplicaIndex,
}
```

`ReplicaCapRange` is boot-time policy. It must satisfy `min <= max`, and v1 caps should stay within `0..=3` unless a later runtime design explicitly raises the global limit. The runtime creates `max` module instances for the registration. Allocation chooses the effective active replica count and is clamped to `min..=max`.

Application boot registers modules as `(module_id, cap_range, builder)`:

```rust
let registry = ModuleRegistry::new().register(builtin::query_vector(), 0..=3, |caps| {
    QueryVectorModule::new(
        caps.allocation_updated_inbox(),
        caps.cognition_log_updated_inbox(),
        caps.allocation_reader(),
        caps.blackboard_reader(),
        caps.vector_memory_searcher(),
        caps.memo(),
        caps.llm_access(),
    )
})?;
```

The builder receives `ModuleCapabilityFactory`, a replica-scoped view over the root `CapabilityProviders`. It is already bound to the hidden `ModuleInstanceId`, so owner-stamped capability methods take no owner argument. Module constructors never receive `ModuleInstanceId`, `ReplicaIndex`, or `ModuleId` for self-stamping.

Root/shared capabilities such as `BlackboardReader`, `CognitionLogReader`, `AllocationReader`, memory search/write ports, file search, `Clock`, and `TimeDivision` may still be issued through the scoped factory. Owner-stamped capabilities such as `Memo`, `LlmAccess`, typed inboxes/mailboxes, `CognitionWriter`, and `UtteranceWriter` always stamp the hidden instance.

Replica identity is visible in runtime state and diagnostics, not in ordinary module code. Compact views preserve the current shape: if a module has exactly one relevant replica, prompt serialization and observations should not label it as `replica 0`.

### Resource allocation and proposals

`ResourceAllocation` is keyed by `ModuleId` and has exactly three public allocation axes: activation ratio, controller guidance, and model tier.

```rust
pub struct ModuleConfig {
    pub activation_ratio: ActivationRatio,
    pub guidance: String,
    pub tier: ModelTier,
}

pub struct ResourceAllocation {
    per_module: HashMap<ModuleId, ModuleConfig>,
}
```

`ActivationRatio` is an internal fixed-point value serialized as a JSON number in `0.0..=1.0`. Active replicas are derived from the effective ratio and boot-time cap range:

```rust
if cap_range.max == 0 {
    active = 0;
} else {
    requested = ceil(activation_ratio * cap_range.max);
    active = requested.clamp(cap_range.min, cap_range.max);
}
```

This means `cap_range.min` remains a boot-time guarantee. `activation_ratio = 0.0` disables a module only when its registered `cap_range.min == 0`.

Allocation-controller replicas do not directly replace the effective allocation. `AllocationWriter` records the holder's proposal. The runtime computes effective allocation from active controller proposals:

- `activation_ratio`: clamp each proposal to `0.0..=1.0`, then take the arithmetic mean.
- `guidance`: keep the single active controller's guidance as-is; for multiple active controller proposals, preserve every active guidance with owner labels in deterministic order.
- `tier`: ordinal mean over `Cheap < Default < Premium`, rounded to nearest tier, then mapped back to `ModelTier`.

If there are no active controller proposals for a module, the effective allocation falls back to the boot/base allocation for that module and is still clamped to its cap range. If there is only one active controller replica, effective allocation is equivalent to that replica's clamped proposal.

The allocation-controller structured-output schema is generated from the module registry at activation time. It enumerates registered module ids and exposes only:

- `activation_ratio`: JSON number in `0.0..=1.0`,
- `guidance`: natural-language controller guidance for that module,
- `tier`: `Cheap | Default | Premium`.

Runtime parsing still sanitizes output: unknown module ids are ignored, missing fields preserve the current proposal value, and out-of-range ratios are clamped because LLM output is not trusted. `replicas`, `period`, `context_budget`, `message_only`, and `period_ms` are not allocation fields.

### Typed channel capabilities

`crates/module` owns the builtin typed channel capabilities:

```rust
use chrono::{DateTime, Utc};

pub struct Envelope<T> {
    pub sender: ModuleInstanceId,
    pub body: T,
}

pub enum AttentionControlRequest {
    Query { question: String, reason: Option<String> },
    SelfModel { question: String, reason: Option<String> },
    Memory { content: String, importance: MemoryImportance, reason: String },
}
pub type AttentionControlRequestMailbox = TopicMailbox<AttentionControlRequest>;
pub type AttentionControlRequestInbox = TopicInbox<AttentionControlRequest>;

pub enum SensoryInput {
    OneShot {
        modality: SensoryModality,
        direction: Option<String>,
        content: String,
        observed_at: DateTime<Utc>,
    },
    AmbientSnapshot {
        entries: Vec<AmbientSensoryEntry>,
        observed_at: DateTime<Utc>,
    },
}
pub type SensoryInputMailbox = TopicMailbox<SensoryInput>;
pub type SensoryInputInbox = TopicInbox<SensoryInput>;

pub enum CognitionLogUpdated {
    EntryAppended { source: ModuleInstanceId },
    AgenticDeadlockMarker,
}
pub type CognitionLogUpdatedInbox = TopicInbox<CognitionLogUpdated>;

pub struct MemoUpdated {
    pub owner: ModuleInstanceId,
}
pub type MemoUpdatedInbox = TopicInbox<MemoUpdated>;

pub struct AllocationUpdated;
pub type AllocationUpdatedInbox = TopicInbox<AllocationUpdated>;

pub enum MemoryImportance {
    Normal,
    High,
}
```

Typed topics are backed by per-replica inbox queues and route only to currently active target replicas. Holding a publish capability permits sending on that typed topic; holding an inbox permits subscribing. Payloads are Rust types, not ad-hoc JSON.

Delivery policy is per topic:

- `CognitionLogUpdated` is fanout: every active subscriber replica receives the wake signal.
- `MemoUpdated` is fanout with self-filtering at the inbox handle: every active subscriber replica receives memo writes except its own writes.
- `AllocationUpdated` is fanout: every active subscriber replica receives the wake signal when effective allocation or guidance changes.
- `AttentionControlRequest` and `SensoryInput` are role fanout plus replica load-balance: each subscribed module role receives the message, and one active replica of that role is selected round-robin. Boot wiring grants `AttentionControlRequestInbox` only to allocation-controller.
- Disabled replicas receive no newly routed topic messages. Messages already in their local inbox remain there until the event loop starts that replica's next active batch.

`SensoryInput` is the only external stimulus type accepted by full-agent/app boot. It is not a durable answer. The built-in variants are:

- `OneShot { modality, direction, content, observed_at }` for discrete stimuli,
- `AmbientSnapshot { entries, observed_at }` for the complete active ambient sensory field keyed by row id.

`direction` is an optional host-provided egocentric/source label such as `"front"`, `"left"`, `"screen"`, or `"user"`. It is intentionally a string in the first design pass; a richer direction/source type can replace it once real app and eval use cases make the needed geometry clear.

`observed_at` is the host-observed datetime of the stimulus. The sensory module records a normalized observation in its memo with detailed relative timing, not a raw correlation id and not a time-division bucket. It computes `Clock::now() - observed_at` and formats the age as:

- exact seconds below one minute,
- minutes plus seconds rounded to 10-second increments below one hour,
- hours plus minutes rounded to 10-minute increments below one day,
- days plus hours at one day and beyond.

Examples: `Ryo said "..." 42 seconds ago`, `Ryo said "..." 1 minute 20 seconds ago`, `screen showed "..." 3 hours 40 minutes ago`, `Ryo said "..." 2 days 5 hours ago`. Future timestamps caused by host clock skew should be clamped to `0 seconds ago`.

The sensory module does not publish derived typed work requests. Internal work bids use `AttentionControlRequestMailbox` and are consumed by allocation-controller; app-facing/full-agent eval cases must not use them as external inputs.

Speech actions are app-facing port writes rather than channel messages:

```rust
pub struct Utterance {
    pub sender: ModuleInstanceId,
    pub target: String,
    pub text: String,
    pub emitted_at: DateTime<Utc>,
}

pub struct UtteranceDelta {
    pub sender: ModuleInstanceId,
    pub target: String,
    pub generation_id: u64, // increments on each new generation attempt (interruption or retry)
    pub sequence: u32,      // monotone within a generation
    pub delta: String,
}
```

`UtteranceWriter` owner-stamps the emitting module instance, stamps `emitted_at` from `Clock`, and sends utterances to an `UtteranceSink`. The writer is a side-effect capability like `CognitionWriter`; it is not a request/response path.

`UtteranceWriter` exposes three methods:

- `emit(target, text)` — stamps `emitted_at`, owner, target, and sends a complete `Utterance` to the sink. Used by any non-streaming utterance path.
- `emit_delta(target, generation_id, sequence, delta)` — sends a targeted `UtteranceDelta` chunk. Used by the speak module during streaming. After the stream completes, speak also calls `emit()` with the full assembled text so that sinks which only consume complete utterances receive a well-formed record.
- `record_progress(progress)` — owner-stamps the latest utterance progress on the blackboard so SpeakGate can inspect speech state while deciding future Speak activations.

`UtteranceDelta` carries no durability semantics. When generation retries, speak keeps the same `generation_id`, passes the already-emitted partial utterance back into the next generation request, and continues with the next `sequence`. Cognition-log updates that arrive during streaming remain queued in Speak's inbox and are considered after the current activation completes. Sinks append resumed retry chunks to their in-progress buffer. Eval harnesses (Section 7) ignore deltas entirely and score output only from complete `Utterance` records.

### Allocation updates

`AllocationWriter` records the holder controller replica's proposal. After each write, the blackboard recomputes effective allocation. If the effective allocation changes in any module's `activation_ratio`, `guidance`, `tier`, or derived active replica count, `AllocationUpdated` is published.

`AllocationUpdated` is not a request and does not carry work. It tells modules that controller guidance changed and that they should reread allocation as source of truth. Modules may use that wake to start guidance-driven work, update local state, or wait silently.

### Replica activation

Lowering `activation_ratio` never destroys module instances and never closes inboxes. The agent event loop owns activation gating: it starts `next_batch()` only for active replicas and calls `activate(&batch)` only while that replica is active. Disabled replicas keep their module state but do not receive newly started batch or activation work.

The activation prelude is:

```rust
let mut batch = self.await_first_batch().await?;
self.collect_ready_events_into_batch(&mut batch)?;
Ok(batch)
```

The awaited receive and ready drain are module-local. Active-replica gating happens outside the module in the event loop before `next_batch` starts and before `activate(&batch)` runs.

Wake-only activations such as `CognitionLogUpdated`, `MemoUpdated`, and `AllocationUpdated` may be collapsed into a single pending wake because the source of truth is durable state such as cognition logs, memo logs, or allocation. Work-carrying activations such as `SensoryInput` and `AttentionControlRequest` must not be silently discarded by the transport because the payload is the work. `AttentionControlRequest` is controller-only: allocation-controller may admit, defer, or reject it in its memo and allocation guidance; there is no durable pending queue after that judgement.

If a replica is disabled while it has no pending work, the event loop leaves it stored and does not start `next_batch`. If a batch is already available when the replica becomes inactive, the event loop holds that batch and defers activation until the replica is active again. Allocation changes do not preempt an already-running activation, including active Speak streams.

Boot policy should register `allocation-controller` with `cap_range.min >= 1`. If all controller replicas are disabled by host policy, only the host or explicit boot wiring can recover allocation.

### Inbox batching

Inbox batching is a per-module activation prelude. A module waits for an activation event, deterministically collects module-local transient events, and converts them into a private `NextBatch` before any LLM call or side effect. Most modules use the immediate ready-drain pattern:

```rust
async fn next_batch(&mut self) -> Result<NextBatch> {
    let mut batch = self.await_first_batch().await?;
    self.collect_ready_events_into_batch(&mut batch)?;
    Ok(batch)
}
```

Rules:
- By default, the first event is the only awaited receive. The ready collection uses `take_ready_items()` and does not wait for more work.
- Domain-specific burst batching is allowed only as module-local deterministic policy. Sensory and allocation-controller use a short silent window with a one-second total budget to coalesce bursts before activation.
- `TopicInbox::next_item()` and `TopicInbox::take_ready_items()` hide transport details from modules. Closed inboxes are application-shutdown signals and propagate as `TopicRecvError::Closed`.
- After the deterministic batch is collected, the event loop owns semantic processing. If the replica is inactive, the event loop performs no LLM call, memo write, or side effect and keeps the pending batch for later activation.
- v1 does not impose a shared `max_ready_events` limit. If a future module needs burst limits, that policy should be added as module-local domain logic rather than as a scheduler allocation field.
- There is no shared `ActivationBatch`, `Incoming`, or `NextBatch` type in `crates/module`. Each module defines the private event and batch shape that matches its capability set.
- Cross-inbox event ordering has no runtime-wide meaning. Each module defines whether `calculate_next_batch` preserves order, groups by source, or reduces wake signals into booleans.
- Batching is deterministic computation over already-received transient events. It must not call an LLM to decide batch membership. Module-local LLM work may still decide semantic output over the deterministic batch, such as memory candidate deduplication.
- Closed inboxes terminate the module loop by propagating a typed receive error to the module's `run_loop`.
- When attention-control requests and memo wakes arrive during allocation-controller's silent window, allocation-controller considers them in one allocation decision.

Module conventions:
- Allocation-update modules, such as cognition-gate, query, self-model, memory, and memory-compaction, collapse multiple ready allocation updates into one guidance activation and reread allocation as source of truth.
- Memo-update modules, such as allocation-controller, collapse multiple ready memo updates into one wake activation and reread the blackboard as source of truth. Allocation-controller also waits a bounded silent window so near-simultaneous memo wakes and attention-control requests share one allocation decision. `MemoUpdatedInbox` drops self-sent updates so a module cannot wake itself by writing its own memo.
- Cognition-log-update modules, such as attention-schema, predict, and surprise, collapse multiple ready cognition-log updates into one wake activation and reread the cognition log as source of truth. `CognitionLogUpdatedInbox` drops self-sent updates so a module cannot wake itself by appending to its own cognition log.
- Query and self-model modules no longer receive explicit request payloads. They wake from allocation guidance and write memo-authoritative results from the controller's guidance plus blackboard context.
- Memory wakes from cognition-log updates and allocation guidance, reads the current cognition log plus indexed memo logs, and writes only through `insert_memory` tool calls. Memo logs, cognition-log entries, and preservation guidance are candidate evidence, not durable write commands; the memory module may deduplicate, merge, normalize, or reject candidates.
- Speak batches ready `CognitionLogUpdated` wake signals, then activation gates decide whether that batch may run. Cognition-log updates received during a generation stream remain queued for the next Speak batch.
- Sensory coalesces raw sensory inputs with a bounded silent window before salience scoring. Allocation guidance is read during input processing, but allocation updates alone do not wake sensory.

### Replica-owned blackboard views

Module-owned durable state is keyed internally by `ModuleInstanceId`.

Memos:
- `Memo::write` replaces only the holder instance's memo.
- `BlackboardInner` stores `HashMap<ModuleInstanceId, String>`.
- Reader helpers provide exact-instance reads for capabilities and grouped module reads for prompts/eval.
- Singleton serialization keeps the old shape: a module with one active memo appears as `"query-memory": "<memo>"`, not `"query-memory[0]"`.
- Multi-replica serialization groups entries by module and includes `replica` only when there is more than one memo for that module.

Cognition log:
- `CognitionWriter` appends to the holder instance's cognition log.
- `BlackboardInner` stores cognition logs keyed by `ModuleInstanceId`.
- `CognitionLogReader` exposes a cognition-log set, not one global stream.
- Prompt/eval serialization keeps the old array shape for a single source, uses source records with `{ module, replica, entries }` when more than one source exists, and includes `agentic_deadlock_marker` only when the event loop has recorded one.
- `CognitionLogUpdated` carries either the updated source owner or an agentic-deadlock marker wake; self-filtering inboxes drop updates sent by their own owner, and consumers still reread the cognition-log set as source of truth.

Memory metadata remains keyed by `MemoryIndex`. Storage-level memory replicas are external persistence mirrors and are unrelated to module replicas.

### Other capability handles

All capabilities are non-exclusive: capability issuers do not enforce uniqueness on any handle. The "Owner-stamped" column indicates capabilities whose hidden `ModuleInstanceId` is baked in at construction so it cannot be forged by module code.

| Handle | Owner-stamped | Purpose |
|---|---:|---|
| `ModuleCapabilityFactory` | yes | replica-scoped issuer of owner-stamped handles |
| `SensoryInputMailbox` | yes | publish external observations into the agent boundary |
| `SensoryInputInbox` | yes | subscribe to external observations |
| `MemoUpdatedInbox` | yes | subscribe to memo-update wake signals, excluding the holder's own memo writes |
| `AllocationUpdatedInbox` | yes | subscribe to effective allocation/guidance changes |
| `AttentionControlRequestMailbox` | yes | publish internal attention-control bids to allocation-controller |
| `AttentionControlRequestInbox` | yes | subscribe to internal attention-control bids; boot wiring grants this only to allocation-controller |
| `Memo` | yes | append/read holder instance's own bounded indexed memo queue |
| `LlmAccess` | yes | get the current-tier `lutum::Lutum` for the holder's module allocation |
| `BlackboardReader` | no | read whole blackboard through compact grouped views |
| `CognitionLogReader` | no | read the cognition-log set only |
| `AllocationReader` | no | read effective allocation snapshot and, for controller prompts, registry cap metadata |
| `ModuleStatusReader` | no | read scheduler-owned module run status |
| `VectorMemorySearcher` | no | primary-store memory search + access metadata patch |
| `MemoryContentReader` | no | primary-store content lookup by memory index |
| `MemoryWriter` | no | primary-assigned insert + storage-replica fan-out + single metadata mirror |
| `MemoryCompactor` | no | store-atomic compaction + storage-replica fan-out + remember-token increment |
| `CognitionWriter` | yes | append holder instance's cognition log + persist + publish update |
| `AllocationWriter` | yes | record holder controller instance's allocation proposal |
| `UtteranceWriter` | yes | emit app-facing speech actions for one speak replica + persist/notify through an adapter |
| `Clock` | no | injected current time for timestamping and relative-time normalization |
| `TimeDivision` | no | load cognition-boundary age buckets from `configs/time-division.eure` |

### Runtime event observation

Capability providers may be constructed with a `RuntimeEventSink`. The default sink is a no-op, but eval and observability boot can inject a sink that receives owner-stamped runtime events without giving modules any new side-effect power.

The initial event set is:

```rust
pub enum RuntimeEvent {
    LlmAccessed {
        sequence: u64,
        call: u64,
        owner: ModuleInstanceId,
        tier: ModelTier,
    },
    MemoUpdated {
        sequence: u64,
        owner: ModuleInstanceId,
        char_count: usize,
    },
    RateLimitDelayed {
        sequence: u64,
        owner: ModuleInstanceId,
        capability: CapabilityKind,
        delayed_for: Duration,
    },
    ModuleBatchThrottled {
        sequence: u64,
        owner: ModuleInstanceId,
        delayed_for: Duration,
    },
}
```

`sequence` is global to the runtime event emitter and preserves event ordering across event kinds. `LlmAccess::lutum().await` emits exactly one `LlmAccessed` event after any rate-limit delay, after resolving the holder's current effective tier, and before returning the `lutum::Lutum` handle. Its `call` field is the LLM acquisition sequence only; it observes acquisition count, not provider request count. `Memo::write(...).await` emits `MemoUpdated` after the blackboard memo write completes; `char_count` is the memo's character count at write time. `RateLimitDelayed` is emitted after a configured capability permit waits and before the delayed capability operation continues. `ModuleBatchThrottled` is emitted after a scheduler-owned next-batch cooldown expires and before the module's next `next_batch()` future is started. Sinks must not deny acquisition, memo writes, delayed capability operations, or delayed scheduler operations; limit policies such as eval `max-llm-calls` request shutdown after observing `LlmAccessed`.

---

## 3. Scheduler And Event Loop

Boot expands registered modules into persistent replica instances before the scheduler starts:

```rust
let modules = ModuleRegistry::new()
    .register(builtin::query_vector(), 0..=3, |caps| {
        QueryVectorModule::new(
            caps.query_inbox(),
            caps.allocation_updated_inbox(),
            caps.allocation_reader(),
            caps.blackboard_reader(),
            caps.vector_memory_searcher(),
            caps.memo(),
            caps.llm_access(),
        )
    })?
    .build(&caps)
    .await?;
```

`ModuleRegistry::register` validates `cap_range.min <= cap_range.max` and the v1 global cap limit. `build` creates one `ModuleCapabilityFactory` per replica index in `0..cap_range.max`, calls the builder once per scoped factory, and records cap metadata for routing, event-loop activation, and allocation-controller schema generation.

The scheduler owns the loop over built modules:

```rust
pub async fn run(
    modules: AllocatedModules,
    config: AgentEventLoopConfig,
    shutdown: impl Future<Output = ()>,
) -> Result<(), SchedulerError> {
    // start next_batch for active replicas, run activate(&batch) with retry,
    // record agentic-deadlock markers, and wait for shutdown.
}
```

`AllocatedModules` is an opaque newtype returned by `ModuleRegistry::build`; public boot paths do not accept raw module vectors. This keeps scheduler startup tied to the registry step that records replica caps for allocation, routing, event-loop activation, and allocation-controller schema generation.

There is no runtime tick API and no periodic activation. Time is still injectable for modules that need timestamps; the event loop also uses injected `Clock` to timestamp agentic-deadlock markers while using `tokio::time::Instant` for monotonic idle-threshold measurement.

The boot layer injects time through `CapabilityProviders`:

```rust
let caps = CapabilityProviders::new(
    CapabilityProviderPorts {
        blackboard,
        cognition_log_port,
        primary_memory_store,
        memory_replicas,
        file_search,
        utterance_sink,
        clock: Arc::new(SystemClock),
        tiers,
    },
);

// Eval/sandbox boot uses the same field with a fixed clock:
// clock: Arc::new(FixedClock::new(observed_now)),
```

`CapabilityProviders` stores `Arc<dyn Clock>` and exposes `clock()` for modules that need current time. Capability handles that stamp time, such as `CognitionWriter` and `UtteranceWriter`, also receive the same clock. This keeps sandboxed eval deterministic and avoids direct `Utc::now()` calls in module code.

---

## 4. Module Responsibilities

Activation gating is event-loop owned. The role-specific capability lists below include only handles held by module structs.

### Sensory

Capabilities: `SensoryInputInbox`, `AllocationReader`, `Memo`, `Clock`, `LlmAccess`.

Receives external one-shot stimuli and ambient snapshots, computes deterministic salience features, then uses an LLM tool turn to decide whether to ignore the stimulus/diff or write a concise normalized observation to the sensory memo. Memo writes happen only through the sensory memo-writing tool. Memo text should use the observation datetime and detailed relative-age formatting, for example `Ryo said "..." 1 minute 20 seconds ago`. It does not read the blackboard, append cognition-log entries, write allocation, write memory, publish query/self-model requests, or emit utterances.

The sensory module is deliberately a pre-attentive filter, not the conversation owner and not a work router. It maintains local stimulus state keyed by a normalized signature such as source/direction plus content or appearance. Habituation and decay are calculated, not delegated to the LLM: repeated low-change stimuli lose salience, old stimuli decay, and novel/user-directed/intense/changed stimuli gain salience.

The LLM stage receives persisted assistant-side sensory ledger entries for one-shots and ambient add/update/remove diffs, plus ephemeral assistant context containing the full current ambient field. It also receives detailed relative age, normalized signature, repetition/change metrics, decay-adjusted salience, current allocation guidance, and any configured thresholds. Its durable shared output is constrained to tool calls: ignore the observation/diff or write a memo observation. The memo is the contract with the rest of the system. It should contain filtered observations and enough inspection detail to explain the computed salience and the LLM decision. Raw sensory events and full ambient snapshots are transient and should not be mirrored wholesale into durable shared state.

### Cognition Gate

Capabilities: `MemoUpdatedInbox`, `AllocationUpdatedInbox`, `BlackboardReader`, `AllocationReader`, `CognitionWriter`, `TimeDivision`, `LlmAccess`.

Reads the non-cognitive blackboard snapshot and current allocation guidance, then appends concise, novel, currently relevant events to this replica's cognition log. It has no `Memo` capability; its only durable output is the cognition log. When promoting sensory memo content, cognition-gate is the only place that applies time-division rounding: it converts the detailed memo age into the bucket/tag from `configs/time-division.eure` before writing the cognition log event.

The module may summarize content as part of its decision, but its architectural job is gating: selecting which non-cognitive memo/blackboard state becomes admitted cognitive evidence.

### Allocation Controller

Capabilities: `MemoUpdatedInbox`, `AttentionControlRequestInbox`, `BlackboardReader`, `CognitionLogReader`, `AllocationReader`, `AllocationWriter`, `Memo`, `LlmAccess`.

Wakes on memo updates, excluding its own memo writes, and internal `AttentionControlRequest` bids. It reads unread memo-log entries into a persistent `Session`, plus current attention-control requests, memory metadata, cognition-log set, effective allocation, and registry cap metadata. It writes this controller replica's allocation proposal; the runtime averages active proposals into the effective allocation.

The controller prompt receives a registry-derived JSON Schema. The schema enumerates known module ids and exposes exactly `activation_ratio`, `guidance`, and `tier` for each allocation entry. Runtime parsing clamps `activation_ratio` to `0.0..=1.0`; active replicas are derived later from the target module's `cap_range`.

When current memo logs and cognition logs suggest that the agent should gather evidence before speaking, the controller enters an evidence-gathering allocation phase: it keeps `speak` and `speak-gate` active, keeps or raises query and cognition-gate activation ratios/tier, and writes guidance for the modules that can gather or promote the missing evidence. Query modules continue to write memo-authoritative results as log entries; cognition-gate may later promote useful query memo-log content into cognition logs. Once cognition-log state and memo-log history indicate that enough evidence is available for a user-visible answer, SpeakGate can allow the pending Speak activation.

This is not a request/response wait and not a query-completion correlation protocol. The controller allocates work from memo logs, attention-control requests, cognition logs, and allocation state, while query results remain durable only through query-module memo logs. Attention-control requests are admitted, deferred, or rejected in the controller memo; deferred requests are not stored in a separate pending queue.

### Attention Schema

Capabilities: `MemoUpdatedInbox`, `AllocationUpdatedInbox`, `CognitionLogUpdatedInbox`, `BlackboardReader`, `AllocationReader`, `CognitionLogReader`, `CognitionWriter`, `LlmAccess`.

Reads unread memo-log entries into a persistent `Session`, plus allocation state and cognition-log set, to decide whether the current attention state should become admitted cognitive evidence. Its tool set offers a single append operation whose input payload contains plaintext; calling the tool appends a concise first-person attention experience to this replica's cognition log, while not calling the tool means no new attention experience should be admitted. It does not receive `Memo`, attention-control inbox, allocation-write, or memory-write capabilities.

### Self Model

Capabilities: `AllocationUpdatedInbox`, `AllocationReader`, `BlackboardReader` for memo-log/context reads, `CognitionLogReader`, `Memo`, `LlmAccess`.

Maintains a current self-description by integrating attention-schema cognition-log entries, relevant module memo logs, self-related knowledge that query modules surface from memory, and the allocation-controller guidance that requested self-model work. Output is appended to this replica's memo log, which is memo-authoritative for self-model answers.

Stable self-knowledge belongs in memory and is surfaced through query-module memo logs; the self-model module is the dynamic integration layer over that knowledge, current attention, active task context, uncertainty, and recent module outputs. v1 should avoid granting broad direct memory search to self-model unless a later narrow self-knowledge capability is introduced; if `BlackboardReader` proves too wide for this role, introduce a narrower memo-log/context reader before implementation. Object/world models remain distributed through sensory, query, predict, and surprise memo logs in v1; add a dedicated `world-model` only if those fragments need one owner.

### Query Memory

Capabilities: `AllocationUpdatedInbox`, `CognitionLogUpdatedInbox`, `BlackboardReader`, `AllocationReader`, `VectorMemorySearcher`, `Memo`, `LlmAccess`.

Handles memory retrieval (vector-memory/RAG backed in v1). Allocation updates wake it to act on controller guidance; cognition-log updates may also wake it to retrieve memory relevant to the current cognitive surface. It does not handle self-referential or self-model integration, and does not query the policy store. Output is appended to this replica's memo log, and those entries contain only retrieved memory content copied from query results. Reads count as memory access; rank elevation is applied by `MemoryStore` and is not module-visible. The crate directory remains `crates/modules/query-vector/` in this revision; the rename of the crate, eval-case directory, tool id, and `VectorMemorySearcher` capability is a follow-up implementation pass.

### Query Agentic

Capabilities: `AllocationUpdatedInbox`, `BlackboardReader`, `AllocationReader`, `FileSearcher`, `Memo`, `LlmAccess`.

Handles read-only file-search retrieval from controller guidance. Allocation updates wake it to act on guidance with the current blackboard context. Its file-search tool exposes only ripgrep-like controls: `pattern`, `regex`, `invert_match`, `case_sensitive`, `context`, and `max_matches`. It does not receive raw filesystem or shell access. Output is appended to this replica's memo log, and those entries contain only retrieved file content/snippets copied from query results.

### Memory

Capabilities: `CognitionLogUpdatedInbox`, `AllocationUpdatedInbox`, `BlackboardReader`, `AllocationReader`, `MemoryWriter`, `LlmAccess`.

Decides whether useful information should be preserved and inserts memory entries. Cognition-log updates and allocation guidance are wake paths. The memory module evaluates current cognition-log entries, indexed unread/recent memo logs, and preservation guidance as candidates; it may reject, normalize, deduplicate, or merge them, and only persists records that the LLM chooses to write through `insert_memory`.

### Memory Compaction

Capabilities: `AllocationUpdatedInbox`, `BlackboardReader`, `AllocationReader`, `MemoryCompactor`, `LlmAccess`.

Fetches related memory contents and merges redundant entries while accumulating remember tokens. Allocation updates wake it to consider compaction guidance.

### Policy

Capabilities: `CognitionLogUpdatedInbox`, `AllocationUpdatedInbox`, `BlackboardReader`, `AllocationReader`, `PolicyWriter`, `LlmAccess`.

Decides whether a successful or distinctive behavior pattern visible in recent cognition-log entries and module memo logs should be preserved as a tentative policy. Cognition-log updates and allocation guidance are wake paths. Candidates include speak completion memos, surprise-resolved sequences, and explicit controller policy-formation guidance. Each persisted record is a `(trigger, behavior)` pair: `trigger` is the situation description that will be embedded and matched by `query-policy`, and `behavior` is the action/pattern to apply when the trigger matches. Only persists records the LLM chooses to write through `insert_policy`; all new entries start at `PolicyRank::Tentative` with `value = 0.0`, `expected_reward = 0.0`, `confidence = 0.0`, and `reward_tokens = 0`. The module may reject, normalize, deduplicate, or merge candidates against existing policies and may rewrite an existing trigger to broaden or narrow its scope when it inserts a refined record. It does not modify existing policy `value`, `expected_reward`, `confidence`, `rank`, `reward_tokens`, or `decay` — those mutations belong to `reward`. It does not write memory, cognition-log entries, allocation, or memos.

### Value Estimator

Capabilities: `MemoUpdatedInbox`, `CognitionLogUpdatedInbox`, `AllocationUpdatedInbox`, `BlackboardReader`, `CognitionLogReader`, `AllocationReader`, `Memo`, `LlmAccess`.

Predicts the `expected_reward` for each policy currently surfaced by `query-policy` in the current cognitive context. It is the **critic** half of the actor/critic split: `query-policy` retrieves candidates, value-estimator scores them. Wakes primarily on `query-policy` memo updates (filtered through `MemoUpdatedInbox`); cognition-log updates and allocation guidance can also wake it when the context changes meaningfully between retrievals. Reads the retrieved policy entries (trigger, behavior, stored `expected_reward`, `confidence`) plus the recent cognition-log window, then writes a free-form memo containing a list of `(policy_index, predicted_expected_reward, rationale)` entries. The memo is the per-window prediction baseline that `reward` later compares against `ObservedReward` to compute `td_error`. Value-estimator does not modify policy state, write memory, write cognition-log entries, write allocation, or call `PolicyValueUpdater`. Its predictions can refine or override the stored `expected_reward` for the current window without persisting, leaving context-dependent valuation as a v2 schema extension.

### Reward

Capabilities: `CognitionLogUpdatedInbox`, `MemoUpdatedInbox`, `AllocationUpdatedInbox`, `BlackboardReader`, `CognitionLogReader`, `AllocationReader`, `PolicyValueUpdater`, `AttentionControlRequestMailbox`, `Memo`, `LlmAccess`.

Closes the TD loop. Aggregates the v1 6-source `ObservedReward = { external, task, social, cost, risk, novelty }` from surprise memos, speak completion memos, sensory memos (extrinsic signals), and controller guidance, then collapses the structured reward into a scalar `observed_scalar` (configured aggregation; v1 default is summed positives minus summed penalties). Pairs `observed_scalar` with the most recent value-estimator memo for the same policy retrieval window to compute `td_error = observed_scalar − expected_reward`. Applies the TD-0 update through `PolicyValueUpdater::reinforce(index, value_delta, reward_tokens_delta, expected_reward_delta, confidence_delta)`, where v1 deltas follow:

```
expected_reward_delta = α · td_error
value_delta           = β · clamp(td_error, -1.0, 1.0)
confidence_delta      = γ_c · max(0.0, 1.0 - |td_error| / scale)
reward_tokens_delta   = 1 if |td_error| < settle_band else 0
```

The coefficients `α`, `β`, `γ_c`, `scale`, and `settle_band` live in `configs/policy-reinforcement.eure`; v1 ships placeholders that are tunable without spec changes. Rank elevation/demotion remains a derived store-level consequence of value × reward-tokens thresholds; reward cannot bypass those thresholds. Wakes on cognition-log updates and memo updates (notably surprise, speak-completion, value-estimator, and `query-policy` memos). Writes a free-form reward-assessment memo recording the structured `ObservedReward`, the `expected_reward` it compared against, the resulting `td_error`, and the applied deltas, so the allocation-controller can observe learning pressure. May publish `AttentionControlRequest::Policy` to ask the controller to raise `policy` activation when a novel pattern deserves formation. Cannot create new policy entries; cannot predict `expected_reward` (that is value-estimator's role); cannot write memory, cognition-log entries, or allocation; cannot invent rank changes independent of value × token thresholds.

### Query Policy

Capabilities: `AllocationUpdatedInbox`, `CognitionLogUpdatedInbox`, `BlackboardReader`, `AllocationReader`, `PolicySearcher`, `Memo`, `LlmAccess`.

Retrieves applicable policies for the current situation. Allocation updates wake it to act on controller guidance; cognition-log updates may also wake it to surface policies relevant to a newly admitted situation. The vector-search input is constructed from the current situation and is matched against the `trigger` embedding of stored policies only — `behavior`, `expected_reward`, and `confidence` are returned alongside but never participate in similarity scoring. Output is appended to this replica's memo log and contains only retrieved policy entries (`trigger`, `behavior`, stored `expected_reward`, `confidence`, `value`) copied from search results; it does not synthesize advice, modify policy state, or describe itself. Retrieval is the credit-assignment substrate for `reward` and the prediction substrate for `value-estimator`: the memo entries record which policies were active in the window. Retrieval counts as access for diagnostic `usage_history` only — access does not elevate policy rank.

### Predict

Capabilities: `CognitionLogUpdatedInbox`, `CognitionLogReader`, `AllocationReader`, `BlackboardReader`, `Memo`, `LlmAccess`.

Activates on cognition-log updates. Uses an LLM to generate free-form predictions about the likely near-future states of current cognition-log targets, with allocation guidance in context. Subjects are inferred from all active cognition log entries and may include external entities, conversational trajectory, or the agent's own mental state when attention-schema cognition-log entries report self-directed attention. The memo should preserve the target subject, predicted state, estimated validity horizon, and rationale for each useful prediction without imposing a schema.

Predict does not detect divergence and does not write memory.

### Surprise

Capabilities: `CognitionLogUpdatedInbox`, `CognitionLogReader`, `AllocationReader`, `BlackboardReader`, `Memo`, `AttentionControlRequestMailbox`, `LlmAccess`.

Activates on each cognition-log update. Uses an LLM to assess whether the updated cognition log is expected or surprising, with allocation guidance in context. When predict memo logs are available in session context, the assessment is framed as divergence from pending predictions. When predict is absent, the assessment is framed as novelty from recent cognition-log history alone. Runtime reads the `significant` and `memory_request` decision fields for preservation side effects, then writes the same assessment information to this replica's free-form memo log.

Surprise does not generate forward predictions; its activation is cognition-log-driven. When a significant event should be preserved, it publishes an `AttentionControlRequest::Memory` bid for allocation-controller rather than writing memory directly.

The v1 surprise threshold is represented by the structured LLM field `significant`; there is no numeric global threshold yet.

### SpeakGate

Capabilities: `ActivationGate<SpeakModule>`, `CognitionLogReader`, `BlackboardReader`, `ModuleStatusReader`, `AttentionControlRequestMailbox`, `Memo`, `LlmAccess`.

Activates only on pending Speak activation-gate events. Reads unread memo-log entries into its persistent `Session`, plus the cognition-log set, scheduler-owned module status, and utterance progress, to decide whether the pending Speak batch is speech-ready. It can call evidence tools that publish memory query and self-model `AttentionControlRequest` bids. Those tools only report whether work was requested or duplicate; available memo evidence is already in SpeakGate's session from unread memo logs. After publishing missing evidence work, SpeakGate suppresses the current activation and waits for a later Speak batch to reconsider. If speech is ready, it returns `Allow`; otherwise it returns `Suppress` and writes the wait decision and any missing-evidence notes to its memo log. SpeakGate does not emit utterances, write cognition-log entries, write allocation, or write memory.

### Speak

Capabilities: `CognitionLogUpdatedInbox`, `CognitionLogReader`, `SceneReader`, `UtteranceWriter`, `Memo`, `Clock`, `LlmAccess`.

Emits user-visible text. The module is named `speak`, not `talk`, because it represents the concrete speech action capability rather than the whole conversational process.

Speak reads only the cognition-log set and the current scene target schema — it has no `BlackboardReader`, no `AllocationReader`, and does not inspect other modules' memo logs or allocation guidance. This keeps the utterance boundary narrow: speak distills the admitted cognitive surface after SpeakGate has allowed the activation.

Speak waits on `CognitionLogUpdatedInbox`; a ready batch reaches generation only after activation gates allow it. Speak first runs a short structured target-selection turn constrained by `SceneReader`, then streams text to that target. During `text_turn().stream()`, each `TextTurnEvent::TextDelta { delta }` chunk is forwarded immediately via `emit_delta()` and mirrored into utterance progress as the current partial utterance. Cognition-log updates do not cancel an active stream; they remain queued for a later Speak batch.

Speak does not publish query or self-model requests. If an external observation has entered the cognition log but supporting work has not completed, SpeakGate may publish explicit evidence requests during its gate decision, records its wait decision in a memo-log entry, and then waits for a later Speak batch rather than polling for completion. A completed utterance memo-log entry wakes the controller as ordinary output context, but speech start itself is not controlled through memo structure, memo JSON, or allocation guidance.

---

## 5. Lutum Integration

`LlmAccess::lutum()` returns a `lutum::Lutum` for the holder instance's module-level effective tier. Replica identity controls ownership and gating; tier and guidance are allocated per `ModuleId` and shared by all active replicas of that module. Modules build their own `Session` and choose the concrete turn shape (`structured_turn`, `text_turn().tools()`, etc.). The capability layer deliberately does not impose a shared session or agent-loop abstraction.

Tool loops are written directly by each module so tool availability, round limits, result commits, and memo-writing remain local to the module role.

### When to use `.collect()` vs `.stream()`

`.collect()` is the default for all internal cognitive modules. It is appropriate when:

- the turn produces a structured decision (`structured_turn::<T>().collect()`) — output is only useful once complete and schema-validated,
- the turn is part of a tool loop (`text_turn().tools::<T>().collect()`) — each round must complete before tool results can be committed,
- the result is written directly to durable internal state such as free-form `Memo` text or a cognition-log entry — there is no consumer of partial output.

cognition-gate, allocation-controller, attention-schema, self-model, query-memory, query-policy, memory, memory-compaction, policy, value-estimator, and reward use `.collect()` exclusively.

`.stream()` is appropriate only for the `speak` module's text generation step, where the response is user-facing and `UtteranceSink` can act on each chunk as it arrives. See Section 4 (Speak) for the full streaming + interruption pattern.

`StructuredOutputChunk { json_delta }` events and `ReasoningDelta` events are not forwarded to `UtteranceSink`; partial JSON and internal reasoning are not user-meaningful.

---

## 6. Storage Adapters

`module::ports` defines adapter boundaries:

- `MemoryStore`: replicated memory content plus adapter-owned search/indexing state. The primary store assigns `MemoryIndex` values; replica stores accept primary-assigned indexes. These storage replicas are persistence mirrors and are not module replicas. The store also owns memory's access-based rank elevation: when a read path records access (`record_access: true`), the store updates the in-window access counter and, when the configured threshold for the current rank is reached, promotes the entry to the next rank, resets the decay timer to that rank's default, and zeroes the in-window counter. Modules do not see this mechanism beyond the resulting metadata. v1 thresholds (`ShortTerm → MidTerm` at ≥ 3 accesses, `MidTerm → LongTerm` at ≥ 5, `LongTerm → Permanent` at ≥ 8; `Permanent` has no runtime promotion, `Identity` is boot-only) live in `configs/memory-reinforcement.eure` and are tunable without spec changes.
- `PolicyStore`: replicated policy content plus adapter-owned indexing state, parallel to `MemoryStore`. Each `PolicyRecord` holds `(rank, trigger, behavior, expected_reward, confidence, value, reward_tokens, decay, usage_history)`. `trigger` carries both the prose situation description and its embedding; `behavior` is the action/pattern text. The primary store assigns `PolicyIndex` values; replica stores accept primary-assigned indexes. Methods: `insert(NewPolicy) -> PolicyIndex`, `put(IndexedPolicy)`, `get(&PolicyIndex)`, `list_by_rank(PolicyRank)`, `search(&TriggerQuery)`, `reinforce(&PolicyIndex, value_delta, reward_tokens_delta, expected_reward_delta, confidence_delta) -> PolicyRecord`, and `delete(&PolicyIndex)`. `search()` performs vector similarity over the `trigger` embedding only; `behavior`, `value`, `expected_reward`, and `confidence` are returned with hits but never participate in scoring. Decay is applied by the store. Rank changes are driven only by `reinforce()` — when `value` crosses a tier threshold with sufficient `reward_tokens` — or by decay expiry; access never elevates policy rank. `Core` rank is boot-only / manually seeded in v1, parallel to `Identity` memory.
- `ObservedReward`: structured v1 reward payload `{ external, task, social, cost, risk, novelty }` (six `f32` channels). `external` carries explicit user/environment feedback, `task` covers task-completion signals, `social` covers conversation-quality signals, `cost` is a non-negative penalty for token/time/tool expense, `risk` is a non-negative penalty for uncertainty/danger, and `novelty` is an intrinsic exploration bonus. The `reward` module aggregates an `ObservedReward` from surprise, speak-completion, sensory, and controller memos, then collapses it to a scalar for TD comparison. Channels are not stored on `PolicyRecord` in v1 — only the aggregated `value` and `expected_reward` are persisted — but the reward memo records the full breakdown for audit and future context-aware extensions.
- `FileSearchProvider`: read-only ripgrep-like file search with pattern, regex/literal mode, invert match, case sensitivity, context lines, and maximum match count.
- `CognitionLogRepository`: append-only cognition log persistence with the emitting `ModuleInstanceId` / source owner.
- `UtteranceSink`: append-only app-facing output persistence/notification for user-visible speech actions with the emitting `ModuleInstanceId`.
- `Clock`: injectable time source.
- Time-division config: `configs/time-division.eure`, an ordered set of duration buckets used only when cognition-gate promotes sensory memo content into cognition log events. It converts observation age into tags such as `now`, `last_30sec`, `last_2min`, and `before_24hour`.

Desired Eure shape:

```eure
fallback-longest-tag = "before_24hour"

@ tags[] {
  tag = "now"
  range-sec = 3.0
}

@ tags[] {
  tag = "last_30sec"
  range-sec = 30.0
}

@ tags[] {
  tag = "last_2min"
  range-sec = 120.0
}
```

Memory content identity is owned by the primary `MemoryStore`. `MemoryWriter` inserts new content into the primary store first, then mirrors the primary-assigned `MemoryIndex` to replica stores with indexed writes. `MemoryCompactor` uses store-level atomic compaction: primary `compact(new, sources)` must assign the merged `MemoryIndex` and atomically create the merged record while removing all source records; replica `put_compacted(indexed, sources)` must atomically mirror that replacement for the primary-assigned id. Memory metadata is mirrored once on the blackboard after primary success. Primary write/compaction failure fails the operation before metadata changes. Secondary failures are logged. Concrete adapters may be native-only; WASM builds use adapter alternatives.

Policy content identity is owned by the primary `PolicyStore`, parallel to memory. `PolicyWriter` inserts new `(trigger, behavior)` records into the primary store first, then mirrors the primary-assigned `PolicyIndex` to replica stores with indexed writes; new entries start at `PolicyRank::Tentative` with `value = 0.0`, `expected_reward = 0.0`, `confidence = 0.0`, and `reward_tokens = 0`. `PolicyValueUpdater::reinforce` is atomic at the primary store: the primary applies `(value_delta, reward_tokens_delta, expected_reward_delta, confidence_delta)`, clamps the resulting `value`, `expected_reward`, and `confidence` to their respective ranges, checks tier-transition thresholds against the new `value` and `reward_tokens` count, and returns the updated `PolicyRecord` whose rank reflects any transition. The store does not contain the TD formula; learning rates and aggregation rules live in the `reward` module and `configs/policy-reinforcement.eure`. Replicas accept the returned record by primary-assigned id. Policy metadata is mirrored once on the blackboard after primary success. Primary reinforce/insert failure fails the operation before metadata changes; secondary failures are logged.

`UtteranceSink` is not a query response channel. It is an observable action log for host applications and eval harnesses. It exposes two notification surfaces:

- `on_complete(utterance: Utterance)` — a complete, timestamped utterance. Every compliant sink must implement this.
- `on_delta(delta: UtteranceDelta)` — a streaming chunk during generation. This method has a default no-op implementation; sinks that do not need progressive output ignore it. If generation retries, speak resumes the same utterance with the same `generation_id` and the next `sequence`. Sinks append resumed retry chunks to the partial text they already accepted.

Implementations may persist utterances, stream deltas to UI, or both. The `on_complete` call always follows the full `on_delta` sequence for the same `generation_id`, so a UI adapter that consumed deltas can use `on_complete` as a framing signal rather than re-rendering the text.

---

## 7. Eval Integration

`crates/eval` has two schema families because they exercise different boundaries.

Full-agent boundary eval cases live under `eval-cases/full-agent/**/*.eure`. They model app input and therefore support batched `inputs[]` whose `heard`, `seen`, and `one-shot` variants all publish `SensoryInput::OneShot`. They may seed memories, but the user-facing request still enters only through sensory input; memory-required full-agent cases exercise whether controller/query/cognition-gate/speak can surface stored context without direct harness messages. The runner publishes all inputs through `CapabilityProviders::host_io().sensory_input_mailbox()`, yields the current-thread runtime while module tasks react to channel updates, waits until the latest completed action has been silent for one second, max loop iterations, or runtime-event shutdown, and returns the latest complete `Utterance` as `CaseArtifact::output`.

Full-agent eval boot uses a minimal bootstrap allocation rather than waking every module. Sensory, allocation-controller, speak-gate, and speak start with positive activation ratios; cognition-gate starts low and is raised by controller guidance after sensory memo writes; lower-priority query (memory, policy), memory, memory-compaction, policy, value-estimator, reward, prediction, surprise, attention-schema, and self-model modules start at zero activation ratio until the allocation-controller proposes an effective allocation. This keeps full-agent evals testing the controller path instead of bypassing it with an all-on static schedule.

Module eval cases live under `eval-cases/modules/{query-vector,attention-schema,self-model}/**/*.eure`. They are explicit internal harnesses, not app-facing scenarios. The runner seeds the target module's allocation guidance with the module prompt, publishes `AllocationUpdated`, then scores the target module's memo-log entries as the artifact. Attention-schema module cases instead score attention-schema cognition-log entries as the artifact. Module cases may seed `cognition-log[]` entries for cognition-log consumers, and may seed `memos[]` entries as input syntax; those seeds append memo-log entries rather than latest snapshots. Query evals statically check that retrieved content reached the artifact, while rubrics can judge generated search/tool arguments by opting into `trace` as a rubric `judge-inputs[]` value.

Rubric checks choose their judge evidence with data-driven `judge-inputs[]` enum values in the `.eure` case: `output`, `utterance`, `failure`, `trace`, `observations`, `blackboard`, `memory`, `memos`, `cognition`, and `allocation`. The `memos` judge input renders `memo_logs`. The rubric text and criteria are always included; the selected judge inputs control only evidence sections. This keeps full-agent rubrics focused on utterance/memory/blackboard state and query rubrics focused on output plus trace without always passing the whole artifact observation payload.

Common eval behavior:

1. memory seeds are inserted through `MemoryWriter` using `{ content, rank, decay-secs }` so libSQL content and blackboard memory metadata stay aligned; `identity` rank seeds are loaded into the boot identity-memory snapshot when the registry builds modules,
2. the eval runner uses libSQL as the primary memory store, local `PotionBase8MEmbedder` for embeddings, and `lutum-openai` in Chat Completions mode for Ollama,
3. the eval binary installs a global tracing subscriber with `lutum_trace::layer()` before running cases, matching the Lutum trace setup contract; spawned module tasks inherit that global dispatcher,
4. each case writes `artifact.json`, `report.json`, `events.json`, and `trace.json` under `.tmp/eval/<run-id>/<case-id>/`; failed, invalid, runtime-error, and panic cases also write `raw-trace.json` for provider/protocol-level debugging, and the suite writes both `suite-report.json` and append-only `events.jsonl`,
5. live progress is always emitted to stderr for suite start/end, case start/end, full-agent allocation changes, completed utterances, `LlmAccessed` events, `RateLimitDelayed` events, `ModuleBatchThrottled` events, and stop requests,
6. limits include max loop iterations, optional loop sleep, and `max-llm-calls`,
7. `max-llm-calls` counts `LlmAccessed` runtime events and requests scheduler shutdown after the event that reaches the limit; the acquisition is observed, not denied.

This keeps realistic artifacts observable without adding request/response correlation to module channels. Full-agent artifacts are collected at the host boundary from `UtteranceSink`; module artifacts remain memo-authoritative.

---

## 8. Invariants

| Invariant | Enforcement |
|---|---|
| Capabilities are non-exclusive | root providers and scoped factories issue handles without uniqueness checks; any capability may be granted to multiple module instances |
| Module constructors cannot forge replica identity | constructors receive only `ModuleCapabilityFactory`; hidden `ModuleInstanceId` is captured inside owner-stamped handles |
| Replica caps are boot policy | `ModuleRegistry::register` validates `cap_range`; effective allocation derives active replicas from `activation_ratio` and clamps to that range |
| Lowering activation ratio never destroys module state | boot builds `cap_range.max` replicas and allocation only changes routing plus event-loop scheduling |
| Disabled replicas perform no semantic work | routed topics target active replicas only, and the event loop does not start `next_batch` or `activate` for inactive replicas |
| Controller schema matches registered caps | allocation-controller schema is generated from the registry and parsed output is clamped after decoding |
| Controller, attention schema, and self-model are separate modules | separate crates and separate constructor capabilities |
| Controller wakes only on memo updates | it receives `MemoUpdatedInbox`, not `CognitionLogUpdatedInbox`; memo inbox filters self writes |
| Sensory is the full-agent observation boundary | full-agent boot wiring grants `SensoryInputInbox` only to sensory |
| Full-agent external input is sensory-only | app/full-agent eval uses `HostIo` with only `SensoryInputMailbox`; `AttentionControlRequest` is internal and controller-only |
| Sensory cannot answer, route work, or alter cognition directly | it receives no `AttentionControlRequestMailbox`, `UtteranceWriter`, `CognitionWriter`, `AllocationWriter`, memory capabilities, or broad readers |
| Speak is the full-agent utterance boundary | full-agent boot wiring grants `UtteranceWriter` only to speak |
| Speak cannot route work or mutate cognition | it receives no query/self-model mailbox, `CognitionWriter`, `AllocationWriter`, or memory capabilities |
| Speak reads only cognition log and scene targets | it receives `CognitionLogReader`, `CognitionLogUpdatedInbox`, and `SceneReader`, not `BlackboardReader` or `AllocationReader` |
| Speak start is activation-gate-controlled | `ActivationGate<SpeakModule>` votes from SpeakGate decide whether a pending Speak batch activates; allocation guidance is not parsed by speak |
| Speak still does not route query work | evidence gathering is driven by SpeakGate, allocation-controller guidance, query memo logs, and cognition-log updates; speak receives no `AttentionControlRequestMailbox` |
| Speak does not interrupt active streams on cognition updates | cognition-log updates received during streaming remain queued for the next Speak batch |
| Cognition-gate is the only non-cognitive promotion path | boot-time wiring grants cognition-gate memo/allocation read paths plus `CognitionWriter`; attention-schema also has `CognitionWriter` but no `Memo` output or non-cognitive promotion role |
| Cognition-log inboxes filter self writes | `CognitionLogUpdatedInbox` is constructed with the same self-exclusion policy as `MemoUpdatedInbox` |
| Cognition-log writes cannot wake controller directly | cognition-log appends publish `CognitionLogUpdated`, which the controller does not receive |
| Only controller replicas write allocation proposals | boot-time wiring grants `AllocationWriter` only to allocation-controller registrations; runtime computes effective allocation |
| Attention schema models attention only | it receives memo, allocation, and cognition-log read/wake capabilities plus `CognitionWriter` and `LlmAccess`, not `Memo`, attention-control inbox, `AllocationWriter`, or memory capabilities |
| Self-model handles self-report | allocation-controller writes self-model guidance; self-model receives `AllocationUpdatedInbox` and writes self-model answers to its own memo |
| Self-model is not raw memory retrieval | stable self-knowledge is surfaced through query memo logs; self-model integrates that knowledge with attention-schema cognition-log entries and current memo-log context |
| Query memory is memory/RAG only | it receives `VectorMemorySearcher`, not policy or self-model capabilities |
| Query policy is policy-retrieval only | it receives `PolicySearcher`, not memory or self-model capabilities; its memo entries contain only retrieved policy records |
| Policy vector search is trigger-only | `PolicySearcher` performs similarity over the `trigger` embedding; `behavior`, `value`, `expected_reward`, and `confidence` are never used as search keys |
| Memory rank elevation is store-internal | no module holds a memory rank-elevation capability; `MemoryStore` applies access-threshold promotions on read paths that set `record_access: true` |
| Policy creation and policy update are separate roles | `policy` holds `PolicyWriter` but no mutation path on existing entries; `reward` holds `PolicyValueUpdater` but no insert path |
| Value prediction and value update are separate roles | `value-estimator` writes only its own memo with `expected_reward` predictions; it holds no `PolicyValueUpdater` and no `PolicyWriter` |
| Reward does not write allocation | `reward` may bid only through `AttentionControlRequest::Policy`; only allocation-controller holds `AllocationWriter` |
| Policy rank changes derive from reward, not access | `PolicyValueUpdater::reinforce` is the only path to tier transitions outside decay expiry; `PolicySearcher` records hits in `usage_history` for diagnostics only |
| Policies strengthen by reward, never by access | memories strengthen by access, never by reward; the two stores never share rank enums or strengthening rules |
| TD learning is TD-0 in v1 | `td_error = observed_scalar − expected_reward`; no γ discount, no `next_state` bootstrap, no eligibility traces; the reward module owns the formula and learning rates |
| `ObservedReward` channels are fixed at six in v1 | `external`, `task`, `social`, `cost`, `risk`, `novelty`; channels are recorded in reward memos and collapsed to a scalar for TD; per-policy per-channel storage is a v2 extension |
| `PolicySearcher` does not return demoted or expired policies | the store filters by current rank and decay before returning hits |
| Policy / value-estimator / reward / query-policy ablations are wiring-only | boot wiring may include any subset; ablating `value-estimator` collapses `expected_reward` to the stored value (still functional with degraded credit assignment); ablating `surprise` degrades reward to speak-completion plus sensory only |
| Results are durable via memo, not responses | module instances write their own `Memo`; channels are transient |
| Modules cannot impersonate each other | owner-stamped capabilities construct envelope senders, memo owners, cognition log owners, and utterance owners |
| No periodic activation | there is no `PeriodicInbox`, `PeriodicActivation`, `period`, `period_ms`, or scheduler tick path |
| Modules with `cap_range.min = 0` are detachable by allocation | derived active replica count `0` fully disables all instances without killing loops |
| Modules with `cap_range.min > 0` cannot be fully allocation-disabled | active replica derivation clamps requested replicas up to the registered minimum |
| Query ablations are wiring-only | boot wiring may include query-memory, query-policy, either, or neither |
| Predict and surprise are separate modules | separate crates and separate constructor capabilities |
| Surprise has no forward-modeling responsibility | it receives no direct memo path from predict; predict output arrives through unread memo-log entries on `BlackboardReader` |
| Predict and surprise ablations are wiring-only | boot wiring may include predict, surprise, both, or neither |
| Conversation artifacts are boundary observations | eval collects utterances from `UtteranceSink`, not channel responses |
| LLM call limits observe acquisitions | `LlmAccess::lutum()` emits `LlmAccessed`; eval requests shutdown after the limit event rather than denying the handle |

Mailbox backpressure policy remains a separate runtime policy decision. Current typed topics use unbounded per-subscriber queues; if bounded queues are introduced, overflow policy must be explicit per topic. The source-of-truth state is not carried in channel payloads.
