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
6. **Memo-authoritative results** — Query-module and self-model answers are written to the producing module's `Memo`. Channel messages wake modules or submit work; they are not durable output.
7. **Kebab-case module ids** — `ModuleId(String)` accepts only `^[a-z][a-z0-9]*(?:-[a-z0-9]+)*$`. Builtins live in `nuillu_types::builtin::*`.
8. **Non-exclusive capabilities** — Root providers and replica-scoped capability factories issue handles without uniqueness checks; any capability may be granted to multiple module instances. Single-writer roles are upheld by boot-time wiring and replica-owned state, not by issuer enforcement.
9. **No periodic activation** — Modules wake from typed channels only. Controller guidance in `ResourceAllocation` replaces period/cadence fields; modules decide from allocation guidance whether an allocation wake should produce work or remain silent.
10. **Sensory/action boundaries** — The only app-facing external input is `SensoryInput`, consumed by the `sensory` module. Full-agent runs publish observations through that boundary and collect user-visible text from `speak` / `UtteranceSink`. `QueryRequest` and `SelfModelRequest` are internal module messages; eval may publish them only from module-level harnesses.
11. **Injectable time** — All module-visible current time comes from an injected `Clock` capability. Production boot uses `SystemClock`; eval/sandbox boot can pass a fixed or scripted clock. Capabilities and modules must not call `Utc::now()` directly.
12. **Streaming for user-visible output, collect for internal decisions** — The `speak` module uses `.stream()` for LLM text generation so that `UtteranceWriter` can emit progressive deltas to `UtteranceSink` before the turn completes. All other modules (structured decision turns and tool loops) use `.collect()` because their output is memo-authoritative and has no value until complete and validated. Streaming does not remove the final memo write; it adds progressive forwarding at the utterance boundary only.
13. **Attention/allocation interruption of speak streaming** — If `AttentionStreamUpdated` or `AllocationUpdated` arrives while speak is generating, the current text stream is cancelled and generation restarts with fresh attention and allocation snapshots. This ensures speak's utterances reflect both the latest attended state and the latest controller guidance.
14. **Local deterministic inbox batching** — Modules may batch transient inbox activations immediately before LLM work. Batching is module-local, bounded by boot-time module registration, and deterministic; it does not add a shared runtime batch type or ask an LLM to decide batch membership.
15. **Replica-capped persistent module instances** — Boot registers modules as `(module_id, cap_range, builder)`, creates module instances up to `cap_range.max`, and never destroys those instances when allocation lowers replica count. Allocation changes routing and event-loop scheduling.
16. **Controller proposals with deterministic effective allocation** — Attention-controller replicas write allocation proposals. The runtime derives the effective `ResourceAllocation` by deterministic averaging of `activation_ratio`, `guidance`, and `tier`, then computes active replicas from each module's boot-time cap range.
17. **Registry-derived controller schema** — The attention-controller structured-output JSON Schema is generated from module registrations, enumerates module ids, and exposes only `activation_ratio`, `guidance`, and `tier`. Parsed ratios are still clamped to `0.0..=1.0` because LLM output is not a trust boundary.

---

## 1. Crate Layout

```
crates/
  types/                      # newtypes only (ModuleId, ReplicaIndex, MemoryRank, ModelTier, ...)
  blackboard/                 # Blackboard state + commands
  module/                     # Module trait, port traits, capability handles, typed channels, capability issuers
  modules/
    sensory/                  # observations -> deterministic salience + LLM-filtered memo
    summarize/                # non-cognitive snapshot -> attention stream
    attention-controller/     # attention stream -> resource allocation
    attention-schema/         # attention stream -> self-model memo
    query-vector/             # blackboard/vector memory RAG -> query-vector memo
    query-agentic/            # blackboard/file search -> query-agentic memo
    memory/                   # blackboard snapshot -> memory inserts
    memory-compaction/        # memory metadata/content -> merges
    predict/                  # attention stream + blackboard -> forward predictions memo
    surprise/                 # attention stream divergence/novelty -> memory requests
    speak/                    # attended state + memos -> user-visible utterances
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
        caps.query_inbox(),
        caps.allocation_updated_inbox(),
        caps.allocation_reader(),
        caps.blackboard_reader(),
        caps.vector_memory_searcher(),
        caps.memo(),
        caps.llm_access(),
    )
})?;
```

The builder receives `ModuleCapabilityFactory`, a replica-scoped view over the root `CapabilityProviders`. It is already bound to the hidden `ModuleInstanceId`, so owner-stamped capability methods take no owner argument. Module constructors never receive `ModuleInstanceId`, `ReplicaIndex`, or `ModuleId` for self-stamping.

Root/shared capabilities such as `BlackboardReader`, `AttentionReader`, `AllocationReader`, memory search/write ports, file search, `Clock`, and `TimeDivision` may still be issued through the scoped factory. Owner-stamped capabilities such as `Memo`, `LlmAccess`, typed inboxes/mailboxes, `AttentionWriter`, and `UtteranceWriter` always stamp the hidden instance.

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

Attention-controller replicas do not directly replace the effective allocation. `AllocationWriter` records the holder's proposal. The runtime computes effective allocation from active controller proposals:

- `activation_ratio`: clamp each proposal to `0.0..=1.0`, then take the arithmetic mean.
- `guidance`: keep the single active controller's guidance as-is; for multiple active controller proposals, preserve every active guidance with owner labels in deterministic order.
- `tier`: ordinal mean over `Cheap < Default < Premium`, rounded to nearest tier, then mapped back to `ModelTier`.

If there are no active controller proposals for a module, the effective allocation falls back to the boot/base allocation for that module and is still clamped to its cap range. If there is only one active controller replica, effective allocation is equivalent to that replica's clamped proposal.

The attention-controller structured-output schema is generated from the module registry at activation time. It enumerates registered module ids and exposes only:

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

pub struct QueryRequest { pub question: String }
pub type QueryMailbox = TopicMailbox<QueryRequest>;
pub type QueryInbox = TopicInbox<QueryRequest>;

pub struct SelfModelRequest { pub question: String }
pub type SelfModelMailbox = TopicMailbox<SelfModelRequest>;
pub type SelfModelInbox = TopicInbox<SelfModelRequest>;

pub enum SensoryInput {
    Heard {
        direction: Option<String>,
        content: String,
        observed_at: DateTime<Utc>,
    },
    Seen {
        direction: Option<String>,
        appearance: String,
        observed_at: DateTime<Utc>,
    },
}
pub type SensoryInputMailbox = TopicMailbox<SensoryInput>;
pub type SensoryInputInbox = TopicInbox<SensoryInput>;

pub enum AttentionStreamUpdated {
    StreamAppended { stream: ModuleInstanceId },
    AgenticDeadlockMarker,
}
pub type AttentionStreamUpdatedInbox = TopicInbox<AttentionStreamUpdated>;

pub struct MemoUpdated {
    pub owner: ModuleInstanceId,
}
pub type MemoUpdatedInbox = TopicInbox<MemoUpdated>;

pub struct AllocationUpdated;
pub type AllocationUpdatedInbox = TopicInbox<AllocationUpdated>;

pub struct MemoryRequest {
    pub content: String,
    pub importance: MemoryImportance,
    pub reason: String,
}
pub enum MemoryImportance { Normal, High }
pub type MemoryRequestMailbox = TopicMailbox<MemoryRequest>;
pub type MemoryRequestInbox   = TopicInbox<MemoryRequest>;
```

Typed topics are backed by per-replica inbox queues and route only to currently active target replicas. Holding a publish capability permits sending on that typed topic; holding an inbox permits subscribing. Payloads are Rust types, not serialized enums.

Delivery policy is per topic:

- `AttentionStreamUpdated` is fanout: every active subscriber replica receives the wake signal.
- `MemoUpdated` is fanout with self-filtering at the inbox handle: every active subscriber replica receives memo writes except its own writes.
- `AllocationUpdated` is fanout: every active subscriber replica receives the wake signal when effective allocation or guidance changes.
- `QueryRequest`, `SelfModelRequest`, `SensoryInput`, and `MemoryRequest` are role fanout plus replica load-balance: each subscribed module role receives the message, and one active replica of that role is selected round-robin.
- Disabled replicas receive no newly routed topic messages. Messages already in their local inbox remain there until the event loop starts that replica's next active batch.

`SensoryInput` is the only external stimulus type accepted by full-agent/app boot. It is not a durable answer. The initial built-in variants are:

- `Heard { direction, content, observed_at }` for linguistic/auditory input,
- `Seen { direction, appearance, observed_at }` for visual input.

`direction` is an optional host-provided egocentric/source label such as `"front"`, `"left"`, `"screen"`, or `"user"`. It is intentionally a string in the first design pass; a richer direction/source type can replace it once real app and eval use cases make the needed geometry clear.

`observed_at` is the host-observed datetime of the stimulus. The sensory module records a normalized observation in its memo with detailed relative timing, not a raw correlation id and not a time-division bucket. It computes `Clock::now() - observed_at` and formats the age as:

- exact seconds below one minute,
- minutes plus seconds rounded to 10-second increments below one hour,
- hours plus minutes rounded to 10-minute increments below one day,
- days plus hours at one day and beyond.

Examples: `Ryo said "..." 42 seconds ago`, `Ryo said "..." 1 minute 20 seconds ago`, `screen showed "..." 3 hours 40 minutes ago`, `Ryo said "..." 2 days 5 hours ago`. Future timestamps caused by host clock skew should be clamped to `0 seconds ago`.

The sensory module does not publish derived typed work requests. `QueryMailbox` and `SelfModelMailbox` are internal messaging surfaces; module-level eval harnesses may hold them to isolate query modules or attention-schema, but app-facing/full-agent eval cases must not use them as external inputs. A separate routing/planning module may own those mailboxes in a later design.

Speech actions are app-facing port writes rather than channel messages:

```rust
pub struct Utterance {
    pub sender: ModuleInstanceId,
    pub text: String,
    pub emitted_at: DateTime<Utc>,
}

pub struct UtteranceDelta {
    pub sender: ModuleInstanceId,
    pub generation_id: u64, // increments on each new generation attempt (interruption or retry)
    pub sequence: u32,      // monotone within a generation
    pub delta: String,
}
```

`UtteranceWriter` owner-stamps the emitting module instance, stamps `emitted_at` from `Clock`, and sends utterances to an `UtteranceSink`. The writer is a side-effect capability like `AttentionWriter`; it is not a request/response path.

`UtteranceWriter` exposes two methods:

- `emit(text)` — stamps `emitted_at`, owner, and sends a complete `Utterance` to the sink. Used by any non-streaming utterance path.
- `emit_delta(generation_id, sequence, delta)` — sends a `UtteranceDelta` chunk. Used by the speak module during streaming. After the stream completes, speak also calls `emit()` with the full assembled text so that sinks which only consume complete utterances receive a well-formed record.

`UtteranceDelta` carries no durability semantics. When generation is interrupted (attention-stream update or LLM retry), speak keeps the same `generation_id`, passes the already-emitted partial utterance back into the next generation request, and continues with the next `sequence`. Sinks append resumed chunks to their in-progress buffer. Eval harnesses (Section 7) ignore deltas entirely and score output only from complete `Utterance` records.

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

Wake-only activations such as `AttentionStreamUpdated`, `MemoUpdated`, and `AllocationUpdated` may be collapsed into a single pending wake because the source of truth is durable state such as attention streams, memos, or allocation. Work-carrying activations such as `QueryRequest`, `SelfModelRequest`, `MemoryRequest`, and `SensoryInput` must not be silently discarded because the payload is the work; modules retain them in their existing module-local batch/request shape until the event loop activates the replica.

If a replica is disabled while it has no pending work, the event loop leaves it stored and does not start `next_batch`. If a batch is already available when the replica becomes inactive, the event loop holds that batch and defers activation until the replica is active again. Allocation changes do not preempt an already-running activation unless that module has its own interruption rule; speak's attention/allocation-update interruption is the v1 preemption behavior.

Boot policy should register `attention-controller` with `cap_range.min >= 1`. If all controller replicas are disabled by host policy, only the host or explicit boot wiring can recover allocation.

### Inbox batching

Inbox batching is a per-module activation prelude. A module waits for exactly one activation event, collects already-ready events without awaiting, then deterministically converts those events into a module-local `NextBatch` before any LLM call or side effect:

```rust
async fn next_batch(&mut self) -> Result<NextBatch> {
    let mut batch = self.await_first_batch().await?;
    self.collect_ready_events_into_batch(&mut batch)?;
    Ok(batch)
}
```

Rules:
- The first event is the only awaited receive. The ready collection must use `take_ready_items()` and must not wait for more work.
- `TopicInbox::next_item()` and `TopicInbox::take_ready_items()` hide transport details from modules. Closed inboxes are application-shutdown signals and propagate as `TopicRecvError::Closed`.
- After the deterministic batch is collected, the event loop owns semantic processing. If the replica is inactive, the event loop performs no LLM call, memo write, or side effect and keeps the pending batch for later activation.
- v1 does not impose a shared `max_ready_events` limit. If a future module needs burst limits, that policy should be added as module-local domain logic rather than as a scheduler allocation field.
- There is no shared `ActivationBatch`, `Incoming`, or `NextBatch` type in `crates/module`. Each module defines the private event and batch shape that matches its capability set.
- Cross-inbox event ordering has no runtime-wide meaning. Each module defines whether `calculate_next_batch` preserves order, groups by source, or reduces wake signals into booleans.
- Batching is deterministic computation over already-received transient events. It must not call an LLM to decide batch membership. Module-local LLM work may still decide semantic output over the deterministic batch, such as memory candidate deduplication.
- Closed inboxes terminate the module loop by propagating a typed receive error to the module's `run_loop`.
- When explicit requests and allocation updates are ready together, explicit requests are handled first. `attention-schema` refreshes the self-model before answering if a self-model request and allocation wake are batched together.

Module conventions:
- Allocation-update modules, such as summarize and memory-compaction, collapse multiple ready allocation updates into one guidance activation and reread allocation as source of truth.
- Memo-update modules, such as attention-controller, collapse multiple ready memo updates into one wake activation and reread the blackboard as source of truth. `MemoUpdatedInbox` drops self-sent updates so a module cannot wake itself by writing its own memo.
- Attention-update modules, such as predict and surprise, collapse multiple ready attention updates into one wake activation and reread the attention stream as source of truth.
- Query and self-model modules collect ready explicit requests into a module-local batch and answer the batch in one LLM turn. The module decides answer granularity and memo write policy; channel responses and request/response correlation remain out of scope.
- Memory collects ready `MemoryRequest` values into a deterministic batch of memory candidates, passes them with the blackboard snapshot to its LLM deliberation, and writes only through `insert_memory` tool calls. A request is not a durable write command; the memory module may deduplicate, merge, normalize, or reject candidates.
- Speak may batch while waiting to start work, but attention and allocation updates received during a generation stream remain immediate interruption signals and are not delayed behind start-time batching.
- Sensory batching is deferred in v1 because its batching policy is tied to salience, habituation, and stimulus decay rather than generic activation collapse.

### Replica-owned blackboard views

Module-owned durable state is keyed internally by `ModuleInstanceId`.

Memos:
- `Memo::write` replaces only the holder instance's memo.
- `BlackboardInner` stores `HashMap<ModuleInstanceId, String>`.
- Reader helpers provide exact-instance reads for capabilities and grouped module reads for prompts/eval.
- Singleton serialization keeps the old shape: a module with one active memo appears as `"query-vector": "<memo>"`, not `"query-vector[0]"`.
- Multi-replica serialization groups entries by module and includes `replica` only when there is more than one memo for that module.

Attention:
- `AttentionWriter` appends to the holder instance's attention stream.
- `BlackboardInner` stores attention streams keyed by `ModuleInstanceId`.
- `AttentionReader` exposes an attention-stream set, not one global stream.
- Prompt/eval serialization keeps the old array shape for a single stream, uses stream records with `{ module, replica, entries }` when more than one stream exists, and includes `agentic_deadlock_marker` only when the event loop has recorded one.
- `AttentionStreamUpdated` carries either the updated stream owner or an agentic-deadlock marker wake; consumers still reread the attention-stream set as source of truth.

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
| `QueryMailbox` / `SelfModelMailbox` | yes | publish typed work requests |
| `QueryInbox` / `SelfModelInbox` | yes | subscribe to typed work requests |
| `Memo` | yes | read/write holder instance's own memo slot |
| `LlmAccess` | yes | get the current-tier `lutum::Lutum` for the holder's module allocation |
| `BlackboardReader` | no | read whole blackboard through compact grouped views |
| `AttentionReader` | no | read the attention-stream set only |
| `AllocationReader` | no | read effective allocation snapshot and, for controller prompts, registry cap metadata |
| `VectorMemorySearcher` | no | primary-store memory search + access metadata patch |
| `MemoryContentReader` | no | primary-store content lookup by memory index |
| `MemoryWriter` | no | primary-assigned insert + storage-replica fan-out + single metadata mirror |
| `MemoryCompactor` | no | store-atomic compaction + storage-replica fan-out + remember-token increment |
| `FileSearcher` | no | read-only ripgrep-like file search |
| `AttentionWriter` | yes | append holder instance's attention stream + persist + publish update |
| `AllocationWriter` | yes | record holder controller instance's allocation proposal |
| `MemoryRequestMailbox` | yes | publish explicit memory requests to the memory module |
| `MemoryRequestInbox` | yes | subscribe to explicit memory requests |
| `UtteranceWriter` | yes | emit app-facing speech actions for one speak replica + persist/notify through an adapter |
| `Clock` | no | injected current time for timestamping and relative-time normalization |
| `TimeDivision` | no | load attention-boundary age buckets from `configs/time-division.eure` |

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
}
```

`sequence` is global to the runtime event emitter and preserves event ordering across event kinds. `LlmAccess::lutum().await` emits exactly one `LlmAccessed` event after resolving the holder's current effective tier and before returning the `lutum::Lutum` handle. Its `call` field is the LLM acquisition sequence only; it observes acquisition count, not provider request count. `Memo::write(...).await` emits `MemoUpdated` after the blackboard memo write completes; `char_count` is the memo's character count at write time. Sinks must not deny acquisition or memo writes; limit policies such as eval `max-llm-calls` request shutdown after observing the event.

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

`ModuleRegistry::register` validates `cap_range.min <= cap_range.max` and the v1 global cap limit. `build` creates one `ModuleCapabilityFactory` per replica index in `0..cap_range.max`, calls the builder once per scoped factory, and records cap metadata for routing, event-loop activation, and attention-controller schema generation.

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

`AllocatedModules` is an opaque newtype returned by `ModuleRegistry::build`; public boot paths do not accept raw module vectors. This keeps scheduler startup tied to the registry step that records replica caps for allocation, routing, event-loop activation, and attention-controller schema generation.

There is no runtime tick API and no periodic activation. Time is still injectable for modules that need timestamps; the event loop also uses injected `Clock` to timestamp agentic-deadlock markers while using `tokio::time::Instant` for monotonic idle-threshold measurement.

The boot layer injects time through `CapabilityProviders`:

```rust
let caps = CapabilityProviders::new(
    blackboard,
    attention_port,
    primary_memory_store,
    memory_replicas,
    file_search,
    utterance_sink,
    Arc::new(SystemClock),
    tiers,
);

// Eval/sandbox boot may instead pass:
let caps = CapabilityProviders::new(..., Arc::new(FixedClock::new(observed_now)), tiers);
```

`CapabilityProviders` stores `Arc<dyn Clock>` and exposes `clock()` for modules that need current time. Capability handles that stamp time, such as `AttentionWriter` and `UtteranceWriter`, also receive the same clock. This keeps sandboxed eval deterministic and avoids direct `Utc::now()` calls in module code.

---

## 4. Module Responsibilities

Activation gating is event-loop owned. The role-specific capability lists below include only handles held by module structs.

### Sensory

Capabilities: `SensoryInputInbox`, `AllocationReader`, `Memo`, `Clock`, `LlmAccess`.

Receives external observations, computes deterministic salience features, then uses an LLM to decide whether to ignore the stimulus, fold it into a background summary, or write a concise normalized observation to the sensory memo. Memo text should use the observation datetime and detailed relative-age formatting, for example `Ryo said "..." 1 minute 20 seconds ago`. It does not read the blackboard, append attention, write allocation, write memory, publish query/self-model requests, or emit utterances.

The sensory module is deliberately a pre-attentive filter, not the conversation owner and not a work router. It maintains local stimulus state keyed by a normalized signature such as source/direction plus content or appearance. Habituation and decay are calculated, not delegated to the LLM: repeated low-change stimuli lose salience, old stimuli decay, and novel/user-directed/intense/changed stimuli gain salience.

The LLM stage receives the raw observation, detailed relative age, normalized signature, repetition/change metrics, decay-adjusted salience, current allocation guidance, and any configured thresholds. Its output is constrained to one of: ignore, update background summary, or write/update memo observation. The memo is the contract with the rest of the system. It should contain filtered observations and enough inspection detail to explain the computed salience and the LLM decision. Raw sensory events are transient and should not be mirrored wholesale into durable state.

### Summarize

Capabilities: `AllocationUpdatedInbox`, `BlackboardReader`, `AllocationReader`, `AttentionWriter`, `TimeDivision`, `LlmAccess`.

Reads the non-cognitive blackboard snapshot and current allocation guidance, then appends concise, novel, currently relevant events to this replica's attention stream. It has no `Memo` capability; its only durable output is the attention stream. When promoting sensory memo content, summarize is the only place that applies time-division rounding: it converts the detailed memo age into the bucket/tag from `configs/time-division.eure` before writing the attention stream event.

### Attention Controller

Capabilities: `MemoUpdatedInbox`, `BlackboardReader`, `AttentionReader`, `AllocationReader`, `AllocationWriter`, `Memo`, `LlmAccess`.

Wakes only on memo updates, excluding its own memo writes. It reads the blackboard memo set, memory metadata, attention-stream set, effective allocation, and registry cap metadata. It writes this controller replica's allocation proposal; the runtime averages active proposals into the effective allocation.

The controller prompt receives a registry-derived JSON Schema. The schema enumerates known module ids and exposes exactly `activation_ratio`, `guidance`, and `tier` for each allocation entry. Runtime parsing clamps `activation_ratio` to `0.0..=1.0`; active replicas are derived later from the target module's `cap_range`.

When current memos and attention suggest that the agent should gather evidence before speaking, the controller enters an evidence-gathering allocation phase: it lowers `speak.activation_ratio` when the speak registration permits `min = 0`, keeps or raises query and summarize activation ratios/tier, and writes explicit guidance telling Speak to wait silently. Query modules continue to write memo-authoritative results; summarize may later promote useful query memo content into attention streams. Once attended state and memos indicate that enough evidence is available for a user-visible answer, the controller raises Speak and updates its guidance.

This is not a request/response wait and not a query-completion correlation protocol. The controller judges readiness from memos, attention, and allocation guidance, while query results remain durable only through query-module memos.

### Attention Schema

Capabilities: `SelfModelInbox`, `AllocationUpdatedInbox`, `AttentionReader`, `AllocationReader`, `Memo`, `LlmAccess`.

Maintains a simplified first-person self-model from the attention-stream set and allocation guidance. Explicit self-model questions arrive on `SelfModelInbox`; allocation updates can refresh the model. Output is this replica's memo.

### Query Vector

Capabilities: `QueryInbox`, `AllocationUpdatedInbox`, `BlackboardReader`, `AllocationReader`, `VectorMemorySearcher`, `Memo`, `LlmAccess`.

Handles vector-memory/RAG queries only. Explicit query requests arrive on `QueryInbox`; topic routing load-balances requests across active query-vector replicas. Allocation updates may wake it to act on guidance with the current blackboard context. It does not handle self-referential or self-model questions. Output is this replica's memo, and that memo contains only retrieved memory content copied from query results.

### Query Agentic

Capabilities: `QueryInbox`, `AllocationUpdatedInbox`, `BlackboardReader`, `AllocationReader`, `FileSearcher`, `Memo`, `LlmAccess`.

Handles read-only file-search queries. Explicit query requests arrive on `QueryInbox`; topic routing load-balances requests across active query-agentic replicas. Allocation updates may wake it to act on guidance with the current blackboard context. Its file-search tool exposes only ripgrep-like controls: `pattern`, `regex`, `invert_match`, `case_sensitive`, `context`, and `max_matches`. It does not receive raw filesystem or shell access. Output is this replica's memo, and that memo contains only retrieved file content/snippets copied from query results.

### Memory

Capabilities: `AllocationUpdatedInbox`, `BlackboardReader`, `AllocationReader`, `MemoryWriter`, `MemoryRequestInbox`, `LlmAccess`.

Decides whether useful information should be preserved and inserts memory entries. Explicit `MemoryRequest` messages from other modules (primarily surprise) are preservation candidates, not write commands. Topic routing load-balances requests across active memory replicas. Allocation updates may also wake it to scan current guidance and blackboard state. The memory module evaluates candidates with the current blackboard snapshot, may deduplicate or merge related candidates, and only persists records that the LLM chooses to write through `insert_memory`.

For request-derived short-term memories, `Normal` and `High` requests provide default decay hints of 1 day and 7 days respectively. The final decision, normalized content, rank, and decay belong to the memory module.

### Memory Compaction

Capabilities: `AllocationUpdatedInbox`, `BlackboardReader`, `AllocationReader`, `MemoryCompactor`, `LlmAccess`.

Fetches related memory contents and merges redundant entries while accumulating remember tokens. Allocation updates wake it to consider compaction guidance.

### Predict

Capabilities: `AttentionStreamUpdatedInbox`, `AllocationUpdatedInbox`, `AttentionReader`, `AllocationReader`, `BlackboardReader`, `Memo`, `LlmAccess`.

Activates on attention updates and allocation updates. Uses an LLM to generate predictions about the likely near-future states of currently attended subjects, with allocation guidance in context. Subjects are inferred from all active attention stream entries and may include external entities, conversational trajectory, or the agent's own mental state when attention-schema memos report self-directed attention. Each prediction entry includes the attended subject, predicted state, and an estimated validity horizon. Writes all predictions to this replica's memo.

Predict does not detect divergence and does not send memory requests.

The v1 prediction memo schema is local to the predict module: a rationale plus entries containing subject, predicted state, validity horizon, and rationale.

### Surprise

Capabilities: `AttentionStreamUpdatedInbox`, `AttentionReader`, `AllocationReader`, `BlackboardReader`, `MemoryRequestMailbox`, `Memo`, `LlmAccess`.

Activates on each attention update. Uses an LLM to assess whether the updated attention stream is expected or surprising, with allocation guidance in context. When predict memos are available in the blackboard, the assessment is framed as divergence from pending predictions. When predict is absent, the assessment is framed as novelty from recent attention history alone. When surprise is judged significant, sends a `MemoryRequest` and writes the assessment to this replica's memo.

Surprise does not generate forward predictions; its activation is attention-driven.

The v1 surprise threshold is represented by the structured LLM field `significant`; there is no numeric global threshold yet.

### Speak

Capabilities: `AttentionStreamUpdatedInbox`, `AllocationUpdatedInbox`, `AttentionReader`, `AllocationReader`, `UtteranceWriter`, `Memo`, `Clock`, `LlmAccess`.

Emits user-visible text. The module is named `speak`, not `talk`, because it represents the concrete speech action capability rather than the whole conversational process.

Speak reads only the attention-stream set and allocation guidance — it has no `BlackboardReader` and does not inspect other modules' memos. This keeps the utterance boundary narrow: speak distills what is attended and follows controller guidance, not the full blackboard state.

Speak replicas are allocation-suppressed during evidence gathering. If a speak replica has a pending attention/allocation batch while inactive, the event loop preserves that pending wake without a decision turn, generation turn, memo write, delta, or complete utterance. When allocation later includes that replica, the pending wake runs once against fresh attention and allocation snapshots; it does not need another attention update just to start speaking.

Speak uses a two-stage LLM interaction:

1. **Decision turn** — `structured_turn::<SpeakDecision>().collect()`. Reads the current attention-stream set and allocation snapshots and determines whether a response is warranted (`should_respond`, `rationale`, optional generation hint). If `should_respond` is false, speak writes the decision to its memo and emits nothing.
2. **Generation turn** — `text_turn().stream()`. Each `TextTurnEvent::TextDelta { delta }` chunk is forwarded immediately via `emit_delta()`. Speak simultaneously listens on `AttentionStreamUpdatedInbox` and `AllocationUpdatedInbox` via `tokio::select!`. If either update arrives before the stream completes, the stream is cancelled and speak restarts from step 1 with fresh snapshots:

```rust
loop {
    let attention = attention_reader.snapshot().await;
    let allocation = allocation_reader.snapshot().await;
    // Stage 1: decide
    let decision = session.structured_turn::<SpeakDecision>()
        .collect().await?;
    if !decision.should_respond { break; }
    // Stage 2: stream generation
    let mut stream = session.text_turn().stream().await?;
    let mut accumulated = String::new();
    let mut gen_id = utterance_writer.next_generation_id();
    let mut seq = 0u32;
    let interrupted = loop {
        tokio::select! {
            event = stream.next() => match event {
                Some(Ok(TextTurnEvent::TextDelta { delta })) => {
                    accumulated.push_str(&delta);
                    utterance_writer.emit_delta(gen_id, seq, &delta).await;
                    seq += 1;
                }
                Some(Ok(TextTurnEvent::WillRetry { .. })) => {
                    accumulated.clear();
                    gen_id = utterance_writer.next_generation_id();
                    seq = 0;
                }
                Some(Ok(TextTurnEvent::Completed { .. })) | None => break false,
                _ => {}
            },
            _ = attention_inbox.recv() => break true,
            _ = allocation_inbox.recv() => break true,
        }
    };
    if interrupted { continue; } // restart with new context
    // Stage 3: commit
    memo.write(SpeakMemo { utterance: accumulated.clone(), rationale: decision.rationale }).await;
    utterance_writer.emit(accumulated).await;
    break;
}
```

Speak does not publish query or self-model requests. If an external observation has been attended but supporting work has not completed, the preferred control path is for attention-controller proposals to lower `speak.activation_ratio` and/or give Speak guidance to wait silently while query and summarization work progresses. A completed utterance memo wakes the controller so it can reclaim Speak resources afterward.

---

## 5. Lutum Integration

`LlmAccess::lutum()` returns a `lutum::Lutum` for the holder instance's module-level effective tier. Replica identity controls ownership and gating; tier and guidance are allocated per `ModuleId` and shared by all active replicas of that module. Modules build their own `Session` and choose the concrete turn shape (`structured_turn`, `text_turn().tools()`, etc.). The capability layer deliberately does not impose a shared session or agent-loop abstraction.

Tool loops are written directly by each module so tool availability, round limits, result commits, and memo-writing remain local to the module role.

### When to use `.collect()` vs `.stream()`

`.collect()` is the default for all internal cognitive modules. It is appropriate when:

- the turn produces a structured decision (`structured_turn::<T>().collect()`) — output is only useful once complete and schema-validated,
- the turn is part of a tool loop (`text_turn().tools::<T>().collect()`) — each round must complete before tool results can be committed,
- the result is written directly to a `Memo` slot — there is no consumer of partial output.

summarize, attention-controller, attention-schema, query-vector, query-agentic, memory, and memory-compaction use `.collect()` exclusively.

`.stream()` is appropriate only for the `speak` module's text generation step, where the response is user-facing and `UtteranceSink` can act on each chunk as it arrives. See Section 4 (Speak) for the full streaming + interruption pattern.

`StructuredOutputChunk { json_delta }` events and `ReasoningDelta` events are not forwarded to `UtteranceSink`; partial JSON and internal reasoning are not user-meaningful.

---

## 6. Storage Adapters

`module::ports` defines adapter boundaries:

- `MemoryStore`: replicated memory content plus adapter-owned search/indexing state. The primary store assigns `MemoryIndex` values; replica stores accept primary-assigned indexes. These storage replicas are persistence mirrors and are not module replicas.
- `FileSearchProvider`: read-only ripgrep-like file search with pattern, regex/literal mode, invert match, case sensitivity, context lines, and maximum match count.
- `AttentionRepository`: append-only attention stream persistence with the emitting `ModuleInstanceId` / stream owner.
- `UtteranceSink`: append-only app-facing output persistence/notification for user-visible speech actions with the emitting `ModuleInstanceId`.
- `Clock`: injectable time source.
- Time-division config: `configs/time-division.eure`, an ordered set of duration buckets used only when summarize promotes sensory memo content into attention stream events. It converts observation age into tags such as `now`, `last_30sec`, `last_2min`, and `before_24hour`.

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

`UtteranceSink` is not a query response channel. It is an observable action log for host applications and eval harnesses. It exposes two notification surfaces:

- `on_complete(utterance: Utterance)` — a complete, timestamped utterance. Every compliant sink must implement this.
- `on_delta(delta: UtteranceDelta)` — a streaming chunk during generation. This method has a default no-op implementation; sinks that do not need progressive output ignore it. If generation is interrupted by an attention-stream update or LLM retry, speak resumes the same utterance with the same `generation_id` and the next `sequence`; sinks append those resumed chunks to the partial text they already accepted.

Implementations may persist utterances, stream deltas to UI, or both. The `on_complete` call always follows the full `on_delta` sequence for the same `generation_id`, so a UI adapter that consumed deltas can use `on_complete` as a framing signal rather than re-rendering the text.

---

## 7. Eval Integration

`crates/eval` has two schema families because they exercise different boundaries.

Full-agent boundary eval cases live under `eval-cases/full-agent/**/*.eure`. They model app input and therefore support only batched `inputs[]` whose variants map to `SensoryInput::Heard` and `SensoryInput::Seen`. The runner publishes all inputs through `CapabilityProviders::host_io().sensory_input_mailbox()`, yields the current-thread runtime while module tasks react to channel updates, waits until the first completed utterance, max loop iterations, or runtime-event shutdown, and returns the first complete `Utterance` as `CaseArtifact::output`.

Full-agent eval boot uses a minimal bootstrap allocation rather than waking every module. Sensory, attention-controller, and speak start with positive activation ratios; summarize starts low and is raised by controller guidance after sensory memo writes; lower-priority query, memory, prediction, surprise, and attention-schema modules start at zero activation ratio until the attention-controller proposes an effective allocation. This keeps full-agent evals testing the controller path instead of bypassing it with an all-on static schedule.

Module eval cases live under `eval-cases/modules/{query-vector,query-agentic,attention-schema}/**/*.eure`. They are explicit internal harnesses, not app-facing scenarios. The runner may publish `QueryRequest` to query modules or `SelfModelRequest` to attention-schema through `CapabilityProviders::internal_harness_io()`, then score the target module memo as the artifact. Query evals statically check that retrieved content reached the artifact, while rubrics judge generated search/tool arguments from the trace.

Common eval behavior:

1. memory seeds are inserted through `MemoryWriter` using `{ content, rank, decay-secs }` so libSQL content and blackboard memory metadata stay aligned,
2. the eval runner uses libSQL as the primary memory store, local `PotionBase8MEmbedder` for embeddings, and `lutum-openai` in Chat Completions mode for Ollama,
3. the eval binary installs a global tracing subscriber with `lutum_trace::layer()` before running cases, matching the Lutum trace setup contract; spawned module tasks inherit that global dispatcher,
4. each case writes `artifact.json`, `report.json`, `events.json`, and `trace.json` under `.tmp/eval/<run-id>/<case-id>/`; failed, invalid, runtime-error, and panic cases also write `raw-trace.json` for provider/protocol-level debugging, and the suite writes both `suite-report.json` and append-only `events.jsonl`,
5. live progress is always emitted to stderr for suite start/end, case start/end, full-agent allocation changes, completed utterances, `LlmAccessed` events, and stop requests,
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
| Controller schema matches registered caps | attention-controller schema is generated from the registry and parsed output is clamped after decoding |
| Controller and schema are separate modules | separate crates and separate constructor capabilities |
| Controller wakes only on memo updates | it receives `MemoUpdatedInbox`, not `AttentionStreamUpdatedInbox`, and the inbox filters self writes |
| Sensory is the full-agent observation boundary | full-agent boot wiring grants `SensoryInputInbox` only to sensory |
| Full-agent external input is sensory-only | app/full-agent eval uses `HostIo` with only `SensoryInputMailbox`; `QueryRequest` and `SelfModelRequest` are internal messages |
| Sensory cannot answer, route work, or alter cognition directly | it receives no `QueryMailbox`, `SelfModelMailbox`, `UtteranceWriter`, `AttentionWriter`, `AllocationWriter`, memory capabilities, or readers |
| Speak is the full-agent utterance boundary | full-agent boot wiring grants `UtteranceWriter` only to speak |
| Speak cannot route work or mutate cognition | it receives no query/self-model mailbox, `AttentionWriter`, `AllocationWriter`, or memory capabilities |
| Speak reads only attention and allocation | it receives `AttentionReader` and `AllocationReader`, not `BlackboardReader` |
| Speak suppression is allocation-controlled | attention-controller proposals lower `speak.activation_ratio` or guidance; inactive speak replicas are not activated by the event loop and emit only when active |
| Speak still does not route query work | evidence gathering is driven by allocation, query memos, and attention updates; speak receives no `QueryMailbox` |
| Speak interrupts streaming on attention/allocation updates | speak holds `AttentionStreamUpdatedInbox` and `AllocationUpdatedInbox`; generation restarts from step 1 with fresh attention and allocation snapshots |
| Summarize replicas are the only path to attention stream append | boot-time wiring grants `AttentionWriter` only to summarize registrations; each replica writes its own stream |
| Summary cannot wake controller directly | summarize has no `Memo`; attention appends publish `AttentionStreamUpdated`, which the controller does not receive |
| Only controller replicas write allocation proposals | boot-time wiring grants `AllocationWriter` only to attention-controller registrations; runtime computes effective allocation |
| Query vector is memory/RAG only | it receives `VectorMemorySearcher`, not file or self-model capabilities |
| Query agentic is file-search only | it receives `FileSearcher`, not memory or self-model capabilities |
| Self-model queries go to attention-schema | callers need `SelfModelMailbox`; schema receives `SelfModelInbox` |
| Results are durable via memo, not responses | module instances write their own `Memo`; channels are transient |
| Modules cannot impersonate each other | owner-stamped capabilities construct envelope senders, memo owners, attention stream owners, and utterance owners |
| No periodic activation | there is no `PeriodicInbox`, `PeriodicActivation`, `period`, `period_ms`, or scheduler tick path |
| Modules with `cap_range.min = 0` are detachable by allocation | derived active replica count `0` fully disables all instances without killing loops |
| Modules with `cap_range.min > 0` cannot be fully allocation-disabled | active replica derivation clamps requested replicas up to the registered minimum |
| Query ablations are wiring-only | boot wiring may include query-vector, query-agentic, both, or neither |
| Predict and surprise are separate modules | separate crates and separate constructor capabilities |
| Predict has no memory-request path | it receives no `MemoryRequestMailbox` |
| Surprise has no forward-modeling responsibility | it receives no memo path from predict; reads grouped predict memos only via `BlackboardReader` |
| Surprise is the only module that sends `MemoryRequest` | boot-time wiring convention; capability issuers do not enforce uniqueness |
| Predict and surprise ablations are wiring-only | boot wiring may include predict, surprise, both, or neither |
| Conversation artifacts are boundary observations | eval collects utterances from `UtteranceSink`, not channel responses |
| LLM call limits observe acquisitions | `LlmAccess::lutum()` emits `LlmAccessed`; eval requests shutdown after the limit event rather than denying the handle |

Mailbox backpressure policy remains a separate runtime policy decision. Current typed topics use unbounded per-subscriber queues; if bounded queues are introduced, overflow policy must be explicit per topic. The source-of-truth state is not carried in channel payloads.
