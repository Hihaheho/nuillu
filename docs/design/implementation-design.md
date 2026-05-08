# Implementation Design

Date: 2026-05-07
Scope: Runtime and capability implementation for `attention-schema.md` on top of `lutum`.

This document is the implementation source of truth. It describes the desired architecture, not work progress.

---

## 0. Decisions

1. **Runtime shape** — Each module runs as a persistent task spawned via `tokio::task::spawn_local`. The scheduler waits for shutdown and does not interpret module state.
2. **Single-thread / WASM-compatible futures** — Module futures use `#[async_trait(?Send)]`; the runtime is current-thread / `LocalSet` oriented.
3. **Capability-based design** — A module's possible side effects are exactly the capability handles passed to its constructor. Without a capability, there is no API path to the operation.
4. **Owner-stamped operations** — Identity-bearing capabilities bake a hidden `ModuleInstanceId = (ModuleId, replica)` in at construction. Module constructors receive only a replica-scoped capability factory, so modules cannot claim to send, memo-write, append, or request as another module instance.
5. **Typed channels, not generic payloads** — Module communication uses typed channel capabilities. There is no central `MailboxPayload`, no `serde_json::Value` mailbox, and no request/response correlation protocol.
6. **Memo-authoritative results** — Query-module and self-model answers are written to the producing module's `Memo`. Channel messages wake modules or submit work; they are not durable output.
7. **Kebab-case module ids** — `ModuleId(String)` accepts only `^[a-z][a-z0-9]*(?:-[a-z0-9]+)*$`. Builtins live in `nuillu_types::builtin::*`.
8. **Non-exclusive capabilities** — Root providers and replica-scoped capability factories issue handles without uniqueness checks; any capability may be granted to multiple module instances. Single-writer roles are upheld by boot-time wiring and replica-owned state, not by issuer enforcement.
9. **Elapsed-tick periodic activation** — The app advances elapsed time explicitly. Periodic activation is allocation-aware, replica-aware, accumulator-based, and available only to instances whose registration builder requests `PeriodicInbox`.
10. **Sensory/action boundaries** — The only app-facing external input is `SensoryInput`, consumed by the `sensory` module. Full-agent runs publish observations through that boundary and collect user-visible text from `speak` / `UtteranceSink`. `QueryRequest` and `SelfModelRequest` are internal module messages; eval may publish them only from module-level harnesses.
11. **Injectable time** — All module-visible current time comes from an injected `Clock` capability. Production boot uses `SystemClock`; eval/sandbox boot can pass a fixed or scripted clock. Capabilities and modules must not call `Utc::now()` directly.
12. **Streaming for user-visible output, collect for internal decisions** — The `speak` module uses `.stream()` for LLM text generation so that `UtteranceWriter` can emit progressive deltas to `UtteranceSink` before the turn completes. All other modules (structured decision turns and tool loops) use `.collect()` because their output is memo-authoritative and has no value until complete and validated. Streaming does not remove the final memo write; it adds progressive forwarding at the utterance boundary only.
13. **Attention-stream interruption of speak streaming** — If `AttentionStreamUpdated` arrives while speak is generating, the current text stream is cancelled and generation restarts with a fresh attention snapshot. The in-progress delta sequence is abandoned; the sink detects abandonment via a `generation_id` change on `UtteranceDelta`. This ensures speak's utterances always reflect the latest attended state rather than a stale snapshot taken at stream start.
14. **Local deterministic inbox batching** — Modules may batch transient inbox activations immediately before LLM work. Batching is module-local, bounded by boot-time module registration, and deterministic; it does not add a shared runtime batch type or ask an LLM to decide batch membership.
15. **Replica-capped persistent loops** — Boot registers modules as `(module_id, cap_range, builder)`, creates persistent loops up to `cap_range.max`, and never kills those loops when allocation lowers replica count. Allocation only changes which replica inboxes are active.
16. **Controller proposals with deterministic effective allocation** — Attention-controller replicas write allocation proposals. The runtime derives the effective `ResourceAllocation` by deterministic averaging and clamps each module's requested `replicas` to its boot-time cap range.
17. **Registry-derived controller schema** — The attention-controller structured-output JSON Schema is generated from module registrations, enumerates module ids, and constrains each module's `replicas` to its `cap_range`. Parsed results are still clamped because LLM output is not a trust boundary.

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
  agent/                      # scheduler + manually ticked event-loop handle
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
    async fn run(&mut self);
}
```

`run` is the module's main loop. Modules own their inbox capabilities and decide how to combine them, typically with `tokio::select!`.

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

`ReplicaCapRange` is boot-time policy. It must satisfy `min <= max`, and v1 caps should stay within `0..=3` unless a later runtime design explicitly raises the global limit. The runtime creates `max` persistent loops for the registration. Allocation chooses the effective active replica count and is clamped to `min..=max`.

Application boot registers modules as `(module_id, cap_range, builder)`:

```rust
let registry = ModuleRegistry::new().register(builtin::query_vector(), 0..=3, |caps| {
    QueryVectorModule::new(
        caps.query_inbox(),
        caps.periodic_inbox(),
        caps.blackboard_reader(),
        caps.vector_memory_searcher(),
        caps.memo(),
        caps.llm_access(),
    )
})?;
```

The builder receives `ModuleCapabilityFactory`, a replica-scoped view over the root `CapabilityProviders`. It is already bound to the hidden `ModuleInstanceId`, so owner-stamped capability methods take no owner argument. Module constructors never receive `ModuleInstanceId`, `ReplicaIndex`, or `ModuleId` for self-stamping.

Root/shared capabilities such as `BlackboardReader`, `AttentionReader`, memory search/write ports, file search, `Clock`, and `TimeDivision` may still be issued through the scoped factory. Owner-stamped capabilities such as `Memo`, `LlmAccess`, `PeriodicInbox`, typed inboxes/mailboxes, `AttentionWriter`, and `UtteranceWriter` always stamp the hidden instance.

Replica identity is visible in runtime state and diagnostics, not in ordinary module code. Compact views preserve the current shape: if a module has exactly one relevant replica, prompt serialization and observations should not label it as `replica 0`.

### Resource allocation and proposals

`ResourceAllocation` is still keyed by `ModuleId`; replicas are a third allocation axis alongside tier and period:

```rust
pub struct ModuleConfig {
    pub replicas: u8,
    pub tier: ModelTier,
    pub period: Option<Duration>,
    pub context_budget: TokenBudget,
}

pub struct ResourceAllocation {
    per_module: HashMap<ModuleId, ModuleConfig>,
}
```

`replicas = 0` disables a module only when its registered `cap_range.min == 0`; otherwise the effective value is clamped up to `cap_range.min`. `enabled` is not a canonical allocation axis in this design. Migration code may accept legacy `enabled = false` as `replicas = 0` before clamping, but new controller prompts and schemas should use `replicas`.

Attention-controller replicas do not directly replace the effective allocation. `AllocationWriter` records the holder's proposal. The runtime computes effective allocation from active controller proposals:

- `replicas`: clamp each controller proposal to `cap_range.min..=cap_range.max`, then take the rounded arithmetic mean. The final effective value is clamped again at the trust boundary.
- `tier`: ordinal mean over `Cheap < Default < Premium`, rounded to nearest tier, then mapped back to `ModelTier`.
- `period`: if `None` is the majority, use `None`; otherwise average positive periods and round to milliseconds.
- `context_budget`: rounded arithmetic mean.

If there are no active controller proposals for a module, the effective allocation falls back to the boot/base allocation for that module and is still clamped to its cap range. If there is only one active controller replica, effective allocation is equivalent to that replica's clamped proposal.

The attention-controller structured-output schema is generated from the module registry at activation time. It must enumerate registered module ids and constrain each module branch's `replicas` field with that module's `cap_range.min` and `cap_range.max`. The parsed result is still sanitized: unknown module ids are ignored, missing fields preserve the current proposal value, and out-of-range `replicas` are clamped with `requested.clamp(min_cap, max_cap)` because LLM output is not trusted.

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

pub struct AttentionStreamUpdated {
    pub stream: ModuleInstanceId,
}
pub type AttentionStreamUpdatedInbox = TopicInbox<AttentionStreamUpdated>;

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
- `QueryRequest`, `SelfModelRequest`, `SensoryInput`, and `MemoryRequest` are role fanout plus replica load-balance: each subscribed module role receives the message, and one active replica of that role is selected round-robin.
- Disabled replicas receive no newly routed topic messages. Messages already in their local inbox remain there and are gated before semantic work.

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

### Periodic activation

`PeriodicInbox` is per-module-instance `mpsc`. Requesting it from a replica-scoped capability factory registers that instance for periodic activation:

```rust
let periodic = caps.periodic_inbox();
```

`PeriodicActivation::tick(elapsed)` reads the effective allocation and sends at most one tick per active registered instance when accumulated elapsed time reaches the module period.

Rules:
- An instance is active when its `replica < effective_config.replicas`.
- `period: None` or `period == 0` sends no tick and clears that instance's accumulator.
- Inactive replicas receive no ticks and have their accumulators cleared.
- Allocation changes take effect on the next `tick`.
- One `tick(elapsed)` emits at most one activation per active instance.
- Remainder elapsed carries forward per instance.
- If an active periodic inbox is full/closed, elapsed is retained for retry.

`PeriodicInbox` exposes `next_tick().await` for the first awaited activation and `take_ready_ticks()` for nonblocking batching after that first activation. Lowering replica count does not close the inbox; closed periodic inboxes are application-shutdown signals and propagate as `PeriodicRecvError::Closed`.

### Replica activation gate

Lowering `replicas` never kills module loops and never closes inboxes. Each persistent loop has an owner-stamped `ActivationGate` capability. The gate exposes only `block().await`, which returns immediately when the holder instance is active and waits when allocation currently excludes that replica. It does not expose the full allocation snapshot.

The activation prelude is:

```rust
let mut batch = self.await_first_batch().await?;
self.collect_ready_events_into_batch(&mut batch)?;
self.gate.block().await?;
self.collect_ready_events_into_batch(&mut batch)?;
Ok(batch)
```

This order is required. A module first takes one real activation and drains already-ready work, then parks before semantic work if its replica is disabled. After the gate opens, it drains once more so work that arrived before routing noticed the allocation change can be coalesced into the same deterministic batch.

Wake-only activations such as `AttentionStreamUpdated` and periodic ticks may be collapsed into a single pending wake because the source of truth is durable state such as attention streams or the blackboard. Work-carrying activations such as `QueryRequest`, `SelfModelRequest`, `MemoryRequest`, and `SensoryInput` must not be silently discarded because the payload is the work; modules retain them in their existing module-local batch/request shape until their gate opens.

If a disabled replica is waiting on an empty inbox, it simply remains idle; re-enabling preserves the loop and any local/session state, but does not invent domain work. If a disabled replica is parked at `gate.block()` with a pending batch, re-enabling wakes it and it resumes from the same suspended future. Allocation changes do not preempt an already-running activation unless that module has its own interruption rule; speak's attention-update interruption remains the only v1 preemption behavior.

Boot policy should register `attention-controller` with `cap_range.min >= 1`. If all controller replicas are disabled by host policy, only the host or explicit boot wiring can recover allocation.

### Inbox batching

Inbox batching is a per-module activation prelude, not a scheduler feature. A module waits for exactly one activation event, collects already-ready events without awaiting, blocks on its activation gate, then deterministically converts those events into a module-local `NextBatch` before any LLM call or side effect:

```rust
async fn next_batch(&mut self) -> Result<NextBatch> {
    let mut batch = self.await_first_batch().await?;
    self.collect_ready_events_into_batch(&mut batch)?;
    self.gate.block().await?;
    self.collect_ready_events_into_batch(&mut batch)?;
    Ok(batch)
}
```

Rules:
- The first event is the only awaited receive. The ready collection must use `take_ready_items()` / `take_ready_ticks()` and must not wait for more work.
- `TopicInbox::next_item()` and `TopicInbox::take_ready_items()` hide transport details from modules. Closed inboxes are application-shutdown signals and propagate as `TopicRecvError::Closed`.
- `PeriodicInbox::next_tick()` and `PeriodicInbox::take_ready_ticks()` do the same for periodic activation.
- After the deterministic batch is collected, the module calls `ActivationGate::block()` before semantic processing. If disabled, the module performs no LLM call, memo write, or side effect while the pending batch remains in the suspended `next_batch` future. After the gate opens, the module collects ready events once more and returns the batch.
- v1 does not impose a shared `max_ready_events` limit. If a future module needs burst limits, that policy should be added as module-local domain logic rather than as a scheduler allocation field.
- There is no shared `ActivationBatch`, `Incoming`, or `NextBatch` type in `crates/module`. Each module defines the private event and batch shape that matches its capability set.
- Cross-inbox event ordering has no runtime-wide meaning. Each module defines whether `calculate_next_batch` preserves order, groups by source, or reduces wake signals into booleans.
- Batching is deterministic computation over already-received transient events. It must not call an LLM to decide batch membership. Module-local LLM work may still decide semantic output over the deterministic batch, such as memory candidate deduplication.
- Closed inboxes terminate the module loop by propagating a typed receive error to the module's `run_loop`.
- When explicit requests and periodic ticks are ready together, explicit requests are handled first. `attention-schema` is the exception: if a self-model request and tick are batched together, it refreshes the self-model before answering.

Module conventions:
- Tick-only modules, such as memory-compaction, collapse multiple ready ticks into one periodic activation.
- Memo-update modules, such as summarize, collapse multiple ready memo updates into one wake activation and reread the blackboard as source-of-truth. `MemoUpdatedInbox` drops self-sent updates so a module cannot wake itself by writing its own memo.
- Attention-update modules, such as attention-controller and predict, collapse multiple ready attention updates into one wake activation and reread the attention stream as source-of-truth.
- Query and self-model modules collect ready explicit requests into a module-local batch and answer the batch in one LLM turn. The module decides answer granularity and memo write policy; channel responses and request/response correlation remain out of scope.
- Memory collects ready `MemoryRequest` values into a deterministic batch of memory candidates, passes them with the blackboard snapshot to its LLM deliberation, and writes only through `insert_memory` tool calls. A request is not a durable write command; the memory module may deduplicate, merge, normalize, or reject candidates.
- Speak may batch while waiting to start work, but attention updates received during a generation stream remain immediate interruption signals and are not delayed behind start-time batching.
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
- Prompt/eval serialization keeps the old array shape for a single stream and uses stream records with `{ module, replica, entries }` when more than one stream exists.
- `AttentionStreamUpdated` carries the updated stream owner, but consumers still reread the attention-stream set as source of truth.

Memory metadata remains keyed by `MemoryIndex`. Storage-level memory replicas are external persistence mirrors and are unrelated to module replicas.

### Other capability handles

All capabilities are non-exclusive: capability issuers do not enforce uniqueness on any handle. The "Owner-stamped" column indicates capabilities whose hidden `ModuleInstanceId` is baked in at construction so it cannot be forged by module code.

| Handle | Owner-stamped | Purpose |
|---|---:|---|
| `ModuleCapabilityFactory` | yes | replica-scoped issuer of owner-stamped handles |
| `PeriodicInbox` | yes | allocation-aware periodic activations for one module instance |
| `ActivationGate` | yes | owner-scoped block that returns only when the holder replica is active |
| `SensoryInputMailbox` | yes | publish external observations into the agent boundary |
| `SensoryInputInbox` | yes | subscribe to external observations |
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

Boot expands registered modules into persistent replica loops before the scheduler starts:

```rust
let modules = ModuleRegistry::new()
    .register(builtin::query_vector(), 0..=3, |caps| {
        QueryVectorModule::new(
            caps.query_inbox(),
            caps.periodic_inbox(),
            caps.blackboard_reader(),
            caps.vector_memory_searcher(),
            caps.memo(),
            caps.llm_access(),
        )
    })?
    .build(&caps)
    .await?;
```

`ModuleRegistry::register` validates `cap_range.min <= cap_range.max` and the v1 global cap limit. `build` creates one `ModuleCapabilityFactory` per replica index in `0..cap_range.max`, calls the builder once per scoped factory, and records cap metadata for routing, periodic activation, activation gates, and attention-controller schema generation.

The scheduler only spawns the built loops:

```rust
pub async fn run(
    modules: AllocatedModules,
    shutdown: impl Future<Output = ()>,
) -> Result<(), SchedulerError> {
    for mut module in modules.into_modules() {
        spawn_local(async move { module.run().await });
    }
    // wait for shutdown or task completion policy
}
```

`AllocatedModules` is an opaque newtype returned by `ModuleRegistry::build`; public boot paths do not accept a raw `Vec<Box<dyn Module>>`. This keeps scheduler startup tied to the registry step that records replica caps for allocation, routing, periodic activation, activation gates, and attention-controller schema generation.

The app owns an `AgentEventLoop` and advances time explicitly:

```rust
let mut loop_handle = AgentEventLoop::new(caps.periodic_activation());
loop_handle.tick(Duration::from_millis(20)).await;
```

Production convenience wrappers may call `tick(interval)` from a timer, but wall-clock sleep does not live inside module inboxes. Replica activation gates and topic routing consult the same effective allocation snapshot as periodic activation.

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

The role-specific capability lists below omit the ubiquitous `ActivationGate`. Every module that consumes activations receives its own owner-stamped gate and blocks on it after batching but before semantic work. The gate is not a side-effect capability and does not expose the full allocation snapshot.

### Sensory

Capabilities: `SensoryInputInbox`, `Memo`, `Clock`, `LlmAccess`.

Receives external observations, computes deterministic salience features, then uses an LLM to decide whether to ignore the stimulus, fold it into a background summary, or write a concise normalized observation to the sensory memo. Memo text should use the observation datetime and detailed relative-age formatting, for example `Ryo said "..." 1 minute 20 seconds ago`. It does not read the blackboard, append attention, write allocation, write memory, publish query/self-model requests, or emit utterances.

The sensory module is deliberately a pre-attentive filter, not the conversation owner and not a work router. It maintains local stimulus state keyed by a normalized signature such as source/direction plus content or appearance. Habituation and decay are calculated, not delegated to the LLM: repeated low-change stimuli lose salience, old stimuli decay, and novel/user-directed/intense/changed stimuli gain salience.

The LLM stage receives the raw observation, detailed relative age, normalized signature, repetition/change metrics, decay-adjusted salience, and any configured thresholds. Its output is constrained to one of: ignore, update background summary, or write/update memo observation. The memo is the contract with the rest of the system. It should contain filtered observations and enough inspection detail to explain the computed salience and LLM decision. Raw sensory events are transient and should not be mirrored wholesale into durable state.

### Summarize

Capabilities: `PeriodicInbox`, `BlackboardReader`, `AttentionWriter`, `Memo`, `TimeDivision`, `LlmAccess`.

Reads the non-cognitive blackboard snapshot and appends concise, novel, currently relevant events to this replica's attention stream. When promoting sensory memo content, summarize is the only place that applies time-division rounding: it converts the detailed memo age into the bucket/tag from `configs/time-division.eure` before writing the attention stream event.

### Attention Controller

Capabilities: `AttentionStreamUpdatedInbox`, optional `PeriodicInbox`, `AttentionReader`, `AllocationReader`, `AllocationWriter`, `Memo`, `LlmAccess`.

Reads only the attention-stream set, effective allocation, and registry cap metadata. Writes this controller replica's allocation proposal; the runtime averages active proposals into the effective allocation.

The controller prompt receives a registry-derived JSON Schema. The schema enumerates known module ids and constrains each module's `replicas` to its registered cap range. The controller may also patch tier, period, and context budget. Runtime parsing still clamps `replicas` to `cap_range.min..=cap_range.max`.

When the current attention suggests that the agent should gather evidence before speaking, the controller enters an evidence-gathering allocation phase: it proposes `speak.replicas = 0` when the speak registration permits `min = 0`, keeps or raises query-module replicas/cadence/tier, and records the rationale in its memo. Query modules continue to write memo-authoritative results; summarize may later promote useful query memo content into attention streams. Once the attention-stream set indicates that enough evidence is available for a user-visible answer, the controller proposes a positive speak replica count.

This is not a request/response wait and not a query-completion correlation protocol. The controller judges readiness from attended state and allocation, while query results remain durable only through query-module memos.

### Attention Schema

Capabilities: `SelfModelInbox`, `PeriodicInbox`, `AttentionReader`, `Memo`, `LlmAccess`.

Maintains a simplified first-person self-model from the attention-stream set. Explicit self-model questions arrive on `SelfModelInbox`; periodic activation refreshes the model. Output is this replica's memo.

### Query Vector

Capabilities: `QueryInbox`, `PeriodicInbox`, `BlackboardReader`, `VectorMemorySearcher`, `Memo`, `LlmAccess`.

Handles vector-memory/RAG queries only. Explicit query requests arrive on `QueryInbox`; topic routing load-balances requests across active query-vector replicas. Periodic activation may refresh memory-oriented context for each active replica. It does not handle self-referential or self-model questions. Output is this replica's memo.

### Query Agentic

Capabilities: `QueryInbox`, `PeriodicInbox`, `BlackboardReader`, `FileSearcher`, `Memo`, `LlmAccess`.

Handles read-only file-search queries. Explicit query requests arrive on `QueryInbox`; topic routing load-balances requests across active query-agentic replicas. Periodic activation may refresh file-oriented context for each active replica. Its file-search tool exposes only ripgrep-like controls: `pattern`, `regex`, `invert_match`, `case_sensitive`, `context`, and `max_matches`. It does not receive raw filesystem or shell access. Output is this replica's memo.

### Memory

Capabilities: `PeriodicInbox`, `BlackboardReader`, `MemoryWriter`, `MemoryRequestInbox`, `LlmAccess`.

Decides whether useful information should be preserved and inserts memory entries. Explicit `MemoryRequest` messages from other modules (primarily surprise) are preservation candidates, not write commands. Topic routing load-balances requests across active memory replicas. The memory module evaluates them with the current blackboard snapshot, may deduplicate or merge related candidates, and only persists records that the LLM chooses to write through `insert_memory`.

For request-derived short-term memories, `Normal` and `High` requests provide default decay hints of 1 day and 7 days respectively. The final decision, normalized content, rank, and decay belong to the memory module.

### Memory Compaction

Capabilities: `PeriodicInbox`, `BlackboardReader`, `MemoryCompactor`, `LlmAccess`.

Fetches related memory contents and merges redundant entries while accumulating remember tokens.

### Predict

Capabilities: `AttentionStreamUpdatedInbox`, `PeriodicInbox`, `AttentionReader`, `BlackboardReader`, `Memo`, `LlmAccess`.

Activates on attention updates and periodic ticks. Uses an LLM to generate predictions about the likely near-future states of currently attended subjects. Subjects are inferred from all active attention stream entries and may include external entities, conversational trajectory, or the agent's own mental state when attention-schema memos report self-directed attention. Each prediction entry includes the attended subject, predicted state, and an estimated validity horizon. Writes all predictions to this replica's memo.

Predict does not detect divergence, does not send memory requests, and does not read allocation.

The v1 prediction memo schema is local to the predict module: a rationale plus entries containing subject, predicted state, validity horizon, and rationale.

### Surprise

Capabilities: `AttentionStreamUpdatedInbox`, `AttentionReader`, `BlackboardReader`, `MemoryRequestMailbox`, `Memo`, `LlmAccess`.

Activates on each attention update. Uses an LLM to assess whether the updated attention stream is expected or surprising. When predict memos are available in the blackboard, the assessment is framed as divergence from pending predictions. When predict is absent, the assessment is framed as novelty from recent attention history alone. When surprise is judged significant, sends a `MemoryRequest` and writes the assessment to this replica's memo.

Surprise does not generate forward predictions and does not hold `PeriodicInbox` — its activation is fully attention-driven.

The v1 surprise threshold is represented by the structured LLM field `significant`; there is no numeric global threshold yet.

### Speak

Capabilities: `AttentionStreamUpdatedInbox`, optional `PeriodicInbox`, `AttentionReader`, `UtteranceWriter`, `Memo`, `Clock`, `LlmAccess`.

Emits user-visible text. The module is named `speak`, not `talk`, because it represents the concrete speech action capability rather than the whole conversational process.

Speak reads only the attention-stream set — it has no `BlackboardReader` and does not inspect other modules' memos. This keeps the utterance boundary narrow: speak distills what is attended, not the full blackboard state.

Speak replicas are allocation-suppressed during evidence gathering. If a speak replica receives an attention update or periodic wake and then finds itself inactive at `gate.block()`, it preserves the pending wake without a decision turn, generation turn, memo write, delta, or complete utterance. When allocation later includes that replica, the pending wake runs once against a fresh attention snapshot; it does not need another attention update just to start speaking.

Speak uses a two-stage LLM interaction:

1. **Decision turn** — `structured_turn::<SpeakDecision>().collect()`. Reads the current attention-stream set snapshot and determines whether a response is warranted (`should_respond`, `rationale`, optional generation hint). If `should_respond` is false, speak writes the decision to its memo and emits nothing.
2. **Generation turn** — `text_turn().stream()`. Each `TextTurnEvent::TextDelta { delta }` chunk is forwarded immediately via `emit_delta()`. Speak simultaneously listens on `AttentionStreamUpdatedInbox` via `tokio::select!`. If an update arrives before the stream completes, the stream is cancelled, the current `generation_id` is abandoned, and speak restarts from step 1 with a fresh attention snapshot:

```rust
loop {
    let attention = attention_reader.snapshot().await;
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
        }
    };
    if interrupted { continue; } // restart with new context
    // Stage 3: commit
    memo.write(SpeakMemo { utterance: accumulated.clone(), rationale: decision.rationale }).await;
    utterance_writer.emit(accumulated).await;
    break;
}
```

Speak does not publish query or self-model requests. If an external observation has been attended but supporting work has not completed, the preferred control path is for attention-controller proposals to lower speak replicas while query and summarization work progresses. If a speak replica is active and the attended state is still insufficient, it may remain silent or produce a bounded clarification/failure utterance according to boot-time policy.

---

## 5. Lutum Integration

`LlmAccess::lutum()` returns a `lutum::Lutum` for the holder instance's module-level effective tier. Replica identity controls ownership and gating; tier, period, and context budget are allocated per `ModuleId` and shared by all active replicas of that module. Modules build their own `Session` and choose the concrete turn shape (`structured_turn`, `text_turn().tools()`, etc.). The capability layer deliberately does not impose a shared session or agent-loop abstraction.

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

Full-agent boundary eval cases live under `eval-cases/full-agent/**/*.eure`. They model app input and therefore support only batched `inputs[]` whose variants map to `SensoryInput::Heard` and `SensoryInput::Seen`. The runner publishes all inputs through `CapabilityProviders::host_io().sensory_input_mailbox()` before ticking, advances `AgentEventLoop::tick(elapsed)` until the first completed utterance, max ticks, or runtime-event shutdown, and returns the first complete `Utterance` as `CaseArtifact::output`.

Full-agent eval boot uses a minimal bootstrap allocation rather than waking every module. Sensory and summarize start on cheap tier, attention-controller and speak start on premium tier, and lower-priority query, memory, prediction, surprise, and attention-schema modules start at zero replicas until the attention-controller proposes an effective allocation. This keeps full-agent evals testing the controller path instead of bypassing it with an all-on static schedule.

Module eval cases live under `eval-cases/modules/{query-vector,query-agentic,attention-schema}/**/*.eure`. They are explicit internal harnesses, not app-facing scenarios. The runner may publish `QueryRequest` to query modules or `SelfModelRequest` to attention-schema through `CapabilityProviders::internal_harness_io()`, then score the target module memo as the artifact.

Common eval behavior:

1. memory seeds are inserted through `MemoryWriter` using `{ content, rank, decay-secs }` so libSQL content and blackboard memory metadata stay aligned,
2. the eval runner uses libSQL as the primary memory store, local `PotionBase8MEmbedder` for embeddings, and `lutum-openai` in Chat Completions mode for Ollama,
3. the eval binary installs a global tracing subscriber with `lutum_trace::layer()` before running cases, matching the Lutum trace setup contract; spawned module tasks inherit that global dispatcher,
4. each case writes `artifact.json`, `report.json`, `events.json`, and `trace.json` under `.tmp/eval/<run-id>/<case-id>/`; failed, invalid, runtime-error, and panic cases also write `raw-trace.json` for provider/protocol-level debugging, and the suite writes both `suite-report.json` and append-only `events.jsonl`,
5. live progress is always emitted to stderr for suite start/end, case start/end, full-agent allocation changes, completed utterances, `LlmAccessed` events, and stop requests,
6. limits include `max-ticks`, `tick-ms`, and `max-llm-calls`,
7. `max-llm-calls` counts `LlmAccessed` runtime events and requests scheduler shutdown after the event that reaches the limit; the acquisition is observed, not denied.

This keeps realistic artifacts observable without adding request/response correlation to module channels. Full-agent artifacts are collected at the host boundary from `UtteranceSink`; module artifacts remain memo-authoritative.

---

## 8. Invariants

| Invariant | Enforcement |
|---|---|
| Capabilities are non-exclusive | root providers and scoped factories issue handles without uniqueness checks; any capability may be granted to multiple module instances |
| Module constructors cannot forge replica identity | constructors receive only `ModuleCapabilityFactory`; hidden `ModuleInstanceId` is captured inside owner-stamped handles |
| Replica caps are boot policy | `ModuleRegistry::register` validates `cap_range`; effective allocation clamps `replicas` to that range |
| Lowering replicas never kills loops | boot spawns `cap_range.max` persistent loops and allocation only changes routing, periodic eligibility, and `ActivationGate` state |
| Disabled replicas perform no semantic work | routed topics/periodic ticks target active replicas only, and already-received work blocks at `gate.block()` before LLM calls, memo writes, or side effects |
| Controller schema matches registered caps | attention-controller schema is generated from the registry and parsed output is clamped after decoding |
| Controller and schema are separate modules | separate crates and separate constructor capabilities |
| Controller cannot read non-cognitive blackboard | it receives no `BlackboardReader` |
| Sensory is the full-agent observation boundary | full-agent boot wiring grants `SensoryInputInbox` only to sensory |
| Full-agent external input is sensory-only | app/full-agent eval uses `HostIo` with only `SensoryInputMailbox`; `QueryRequest` and `SelfModelRequest` are internal messages |
| Sensory cannot answer, route work, or alter cognition directly | it receives no `QueryMailbox`, `SelfModelMailbox`, `UtteranceWriter`, `AttentionWriter`, `AllocationWriter`, memory capabilities, or readers |
| Speak is the full-agent utterance boundary | full-agent boot wiring grants `UtteranceWriter` only to speak |
| Speak cannot route work or mutate cognition | it receives no query/self-model mailbox, `AttentionWriter`, `AllocationWriter`, or memory capabilities |
| Speak reads only the attention-stream set | it receives `AttentionReader`, not `BlackboardReader` |
| Speak suppression is replica allocation-controlled | attention-controller proposals lower `speak.replicas`; inactive speak replicas defer pending wakes through `ActivationGate` before decision/generation and emit only when active |
| Speak still does not route query work | evidence gathering is driven by allocation, query memos, and attention updates; speak receives no `QueryMailbox` |
| Speak interrupts streaming on attention updates | speak holds `AttentionStreamUpdatedInbox` and cancels the generation stream on receipt; generation restarts from step 1 with a fresh attention snapshot |
| Summarize replicas are the only path to attention stream append | boot-time wiring grants `AttentionWriter` only to summarize registrations; each replica writes its own stream |
| Only controller replicas write allocation proposals | boot-time wiring grants `AllocationWriter` only to attention-controller registrations; runtime computes effective allocation |
| Query vector is memory/RAG only | it receives `VectorMemorySearcher`, not file or self-model capabilities |
| Query agentic is file-search only | it receives `FileSearcher`, not memory or self-model capabilities |
| Self-model queries go to attention-schema | callers need `SelfModelMailbox`; schema receives `SelfModelInbox` |
| Results are durable via memo, not responses | module instances write their own `Memo`; channels are transient |
| Modules cannot impersonate each other | owner-stamped capabilities construct envelope senders, memo owners, attention stream owners, and utterance owners |
| Periodic activation is opt-in | only `caps.periodic_inbox()` in a registration builder registers that module instance for ticks |
| Modules with `cap_range.min = 0` are detachable by allocation | effective `replicas = 0` fully disables all instances without killing loops |
| Modules with `cap_range.min > 0` cannot be fully allocation-disabled | effective allocation clamps requested `replicas` up to the registered minimum |
| Query ablations are wiring-only | boot wiring may include query-vector, query-agentic, both, or neither |
| Predict and surprise are separate modules | separate crates and separate constructor capabilities |
| Predict has no memory-request path | it receives no `MemoryRequestMailbox` |
| Surprise has no forward-modeling responsibility | it receives no memo path from predict; reads grouped predict memos only via `BlackboardReader` |
| Surprise is the only module that sends `MemoryRequest` | boot-time wiring convention; capability issuers do not enforce uniqueness |
| Predict and surprise ablations are wiring-only | boot wiring may include predict, surprise, both, or neither |
| Conversation artifacts are boundary observations | eval collects utterances from `UtteranceSink`, not channel responses |
| LLM call limits observe acquisitions | `LlmAccess::lutum()` emits `LlmAccessed`; eval requests shutdown after the limit event rather than denying the handle |

Mailbox backpressure policy remains a separate runtime policy decision. Current typed topics use unbounded per-subscriber queues; if bounded queues are introduced, overflow policy must be explicit per topic. The source-of-truth state is not carried in channel payloads.
