# Implementation Design

Date: 2026-05-05
Scope: Runtime and capability implementation for `attention-schema.md` on top of `lutum`.

This document is the implementation source of truth. It describes the desired architecture, not work progress.

---

## 0. Decisions

1. **Runtime shape** — Each module runs as a persistent task spawned via `tokio::task::spawn_local`. The scheduler waits for shutdown and does not interpret module state.
2. **Single-thread / WASM-compatible futures** — Module futures use `#[async_trait(?Send)]`; the runtime is current-thread / `LocalSet` oriented.
3. **Capability-based design** — A module's possible side effects are exactly the capability handles passed to its constructor. Without a capability, there is no API path to the operation.
4. **Owner-stamped operations** — Identity-bearing capabilities bake the owner's `ModuleId` in at construction. A module cannot claim to send, memo-write, append, or request as another module.
5. **Typed channels, not generic payloads** — Module communication uses typed channel capabilities. There is no central `MailboxPayload`, no `serde_json::Value` mailbox, and no request/response correlation protocol.
6. **Memo-authoritative results** — Query-module and self-model answers are written to the producing module's `Memo`. Channel messages wake modules or submit work; they are not durable output.
7. **Kebab-case module ids** — `ModuleId(String)` accepts only `^[a-z][a-z0-9]*(?:-[a-z0-9]+)*$`. Builtins live in `nuillu_types::builtin::*`.
8. **Non-exclusive capabilities** — `CapabilityFactory` issues every capability handle without uniqueness checks; any capability may be granted to multiple modules. Single-writer roles are upheld by boot-time wiring, not by factory enforcement.
9. **Elapsed-tick periodic activation** — The app advances elapsed time explicitly. Periodic activation is allocation-aware, accumulator-based, and available only to modules granted `PeriodicInbox`.
10. **Sensory/action boundaries** — Full-agent runs receive external observations through a `sensory` module and emit user-visible text through a `speak` module. Module-level eval targets may still drive `query-*` or `attention-schema` directly.
11. **Injectable time** — All module-visible current time comes from an injected `Clock` capability. Production boot uses `SystemClock`; eval/sandbox boot can pass a fixed or scripted clock. Capabilities and modules must not call `Utc::now()` directly.
12. **Streaming for user-visible output, collect for internal decisions** — The `speak` module uses `.stream()` for LLM text generation so that `UtteranceWriter` can emit progressive deltas to `UtteranceSink` before the turn completes. All other modules (structured decision turns and tool loops) use `.collect()` because their output is memo-authoritative and has no value until complete and validated. Streaming does not remove the final memo write; it adds progressive forwarding at the utterance boundary only.
13. **Attention-stream interruption of speak streaming** — If `AttentionStreamUpdated` arrives while speak is generating, the current text stream is cancelled and generation restarts with a fresh attention snapshot. The in-progress delta sequence is abandoned; the sink detects abandonment via a `generation_id` change on `UtteranceDelta`. This ensures speak's utterances always reflect the latest attended state rather than a stale snapshot taken at stream start.
14. **Local deterministic inbox batching** — Modules may batch transient inbox activations immediately before LLM work. Batching is module-local, bounded by boot-time module config, and deterministic; it does not add a shared runtime batch type or ask an LLM to decide batch membership.
15. **Full activation gating** — `ResourceAllocation.enabled` means the module is eligible to run any activation, not only periodic ticks. A disabled module may receive transient inbox messages, but it must defer or coalesce them without semantic LLM work, memo writes, or side effects until it is re-enabled.

---

## 1. Crate Layout

```
crates/
  types/                      # newtypes only (ModuleId, MemoryRank, ModelTier, ...)
  blackboard/                 # Blackboard state + commands
  module/                     # Module trait, port traits, capability handles, typed channels, factory
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

Modules each live in their own crate. Their constructor signatures are the role boundary: boot wiring grants only the capabilities a module is allowed to hold.

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

### Typed channel capabilities

`crates/module` owns the builtin typed channel capabilities:

```rust
use chrono::{DateTime, Utc};

pub struct Envelope<T> {
    pub sender: ModuleId,
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

pub struct AttentionStreamUpdated;
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

`SensoryInputMailbox`, `QueryMailbox`, and `SelfModelMailbox` are MPMC fanout topics backed by per-subscriber `futures::channel::mpsc` queues. Holding a publish capability permits sending on that typed topic; holding an inbox permits subscribing. Payloads are Rust types, not serialized enums.

`SensoryInput` is an external stimulus, not a durable answer. The initial built-in variants are:

- `Heard { direction, content, observed_at }` for linguistic/auditory input,
- `Seen { direction, appearance, observed_at }` for visual input.

`direction` is an optional host-provided egocentric/source label such as `"front"`, `"left"`, `"screen"`, or `"user"`. It is intentionally a string in the first design pass; a richer direction/source type can replace it once real app and eval use cases make the needed geometry clear.

`observed_at` is the host-observed datetime of the stimulus. The sensory module records a normalized observation in its memo with detailed relative timing, not a raw correlation id and not a time-division bucket. It computes `Clock::now() - observed_at` and formats the age as:

- exact seconds below one minute,
- minutes plus seconds rounded to 10-second increments below one hour,
- hours plus minutes rounded to 10-minute increments below one day,
- days plus hours at one day and beyond.

Examples: `Ryo said "..." 42 seconds ago`, `Ryo said "..." 1 minute 20 seconds ago`, `screen showed "..." 3 hours 40 minutes ago`, `Ryo said "..." 2 days 5 hours ago`. Future timestamps caused by host clock skew should be clamped to `0 seconds ago`.

The sensory module does not publish derived typed work requests. Direct app use of `QueryMailbox` and `SelfModelMailbox` remains useful for module-level tests, and a separate routing/planning module may own those mailboxes in a later design. Full-agent artifact runs should enter through `SensoryInputMailbox`.

Speech actions are app-facing port writes rather than channel messages:

```rust
pub struct Utterance {
    pub text: String,
    pub emitted_at: DateTime<Utc>,
}

pub struct UtteranceDelta {
    pub generation_id: u64, // increments on each new generation attempt (interruption or retry)
    pub sequence: u32,      // monotone within a generation
    pub delta: String,
}
```

`UtteranceWriter` owner-stamps the emitting module, stamps `emitted_at` from `Clock`, and sends utterances to an `UtteranceSink`. The writer is a side-effect capability like `AttentionWriter`; it is not a request/response path.

`UtteranceWriter` exposes two methods:

- `emit(text)` — stamps `emitted_at`, owner, and sends a complete `Utterance` to the sink. Used by any non-streaming utterance path.
- `emit_delta(generation_id, sequence, delta)` — sends a `UtteranceDelta` chunk. Used by the speak module during streaming. After the stream completes, speak also calls `emit()` with the full assembled text so that sinks which only consume complete utterances receive a well-formed record.

`UtteranceDelta` carries no durability semantics. When generation is interrupted (attention-stream update or LLM retry), `generation_id` is incremented and `sequence` resets to 0. Sinks use the `generation_id` change to detect abandonment and discard their in-progress buffer. Eval harnesses (Section 7) ignore deltas entirely and score output only from complete `Utterance` records.

### Periodic activation

`PeriodicInbox` is per-module `mpsc`. Requesting it from the factory registers that module for periodic activation:

```rust
let periodic = factory.periodic_inbox_for(builtin::query_vector());
```

`PeriodicActivation::tick(elapsed)` reads the current allocation and sends at most one tick per registered module when accumulated elapsed time reaches the module period.

Rules:
- `period: None` or `period == 0` sends no tick and clears that module's accumulator.
- `enabled == false` sends no tick and clears that module's accumulator as the periodic side of the full activation gate.
- Allocation changes take effect on the next `tick`.
- One `tick(elapsed)` emits at most one activation per module.
- Remainder elapsed carries forward.
- If the periodic inbox is full/closed, elapsed is retained for retry.

`PeriodicInbox` exposes `next_tick().await` for the first awaited activation and `take_ready_ticks()` for nonblocking batching after that first activation. Closed periodic inboxes are application-shutdown signals and propagate as `PeriodicRecvError::Closed`.

### Full activation gate

`ModuleConfig.enabled` is the allocation-controlled gate for all activation sources: periodic ticks, typed topic messages, attention-stream updates, and any future wake channel. This is an eligibility bit, not a durability mechanism.

Modules that consume activations receive an owner-stamped `ActivationGate` capability. The gate exposes only `block().await`, which returns immediately when the holder is enabled and otherwise waits until the holder is re-enabled. It does not expose the full allocation snapshot. This preserves module boundaries: `speak`, query modules, and other workers can block until they may run without gaining access to allocation policy. `LlmAccess::lutum()` continues to resolve only the holder's model tier; it does not enforce enabled state.

The activation prelude is:

1. await one activation,
2. collect already-ready transient activations into the module-local batch,
3. call `ActivationGate::block().await`,
4. collect ready activations once more to coalesce work that arrived while blocked,
5. run one normal module-specific semantic activation from the pending batch and a fresh source-of-truth read.

Wake-only activations such as `AttentionStreamUpdated` and periodic ticks may be collapsed into a single pending wake because the source of truth is durable state such as the attention stream or blackboard. Work-carrying activations such as `QueryRequest`, `SelfModelRequest`, `MemoryRequest`, and `SensoryInput` must not be silently discarded because the payload is the work; modules retain them in their existing module-local batch/request shape until enabled. If a future bounded queue policy is introduced, overflow behavior must be explicit per topic and must not be hidden inside `ActivationGate`.

Re-enabling a module with pending activation acts as a wake; the module must not require a new domain event just to notice that it can now process the latest durable state. Allocation changes do not preempt an already-running activation unless that module has its own interruption rule; speak's existing attention-update interruption remains the only v1 preemption behavior.

Boot policy should keep `attention-controller` enabled. If the controller is disabled, only the host or explicit boot wiring can recover allocation.

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
- Tick-only modules, such as summarize and memory-compaction, collapse multiple ready ticks into one periodic activation.
- Attention-update modules, such as attention-controller and predict, collapse multiple ready attention updates into one wake activation and reread the attention stream as source-of-truth.
- Query and self-model modules collect ready explicit requests into a module-local batch and answer the batch in one LLM turn. The module decides answer granularity and memo write policy; channel responses and request/response correlation remain out of scope.
- Memory collects ready `MemoryRequest` values into a deterministic batch of memory candidates, passes them with the blackboard snapshot to its LLM deliberation, and writes only through `insert_memory` tool calls. A request is not a durable write command; the memory module may deduplicate, merge, normalize, or reject candidates.
- Speak may batch while waiting to start work, but attention updates received during a generation stream remain immediate interruption signals and are not delayed behind start-time batching.
- Sensory batching is deferred in v1 because its batching policy is tied to salience, habituation, and stimulus decay rather than generic activation collapse.

### Other capability handles

All capabilities are non-exclusive: the factory does not enforce uniqueness on any handle. The "Owner-stamped" column indicates capabilities whose identity is baked in at construction so they cannot be forged.

| Handle | Owner-stamped | Purpose |
|---|---:|---|
| `PeriodicInbox` | yes | allocation-aware periodic activations |
| `ActivationGate` | yes | owner-scoped block that returns only when the holder may run an activation |
| `SensoryInputMailbox` | yes | publish external observations into the agent boundary |
| `SensoryInputInbox` | yes | subscribe to external observations |
| `QueryMailbox` / `SelfModelMailbox` | yes | publish typed work requests |
| `QueryInbox` / `SelfModelInbox` | yes | subscribe to typed work requests |
| `Memo` | yes | read/write holder's own memo slot |
| `LlmAccess` | yes | get the current-tier `lutum::Lutum` for the owner |
| `BlackboardReader` | no | read whole blackboard |
| `AttentionReader` | no | read attention stream only |
| `AllocationReader` | no | read allocation snapshot |
| `VectorMemorySearcher` | no | primary-store memory search + access metadata patch |
| `MemoryContentReader` | no | primary-store content lookup by memory index |
| `MemoryWriter` | no | primary-assigned insert + replica fan-out + single metadata mirror |
| `MemoryCompactor` | no | store-atomic compaction + remember-token increment |
| `FileSearcher` | no | read-only ripgrep-like file search |
| `AttentionWriter` | yes | append attention stream + persist + publish update |
| `AllocationWriter` | no | replace `ResourceAllocation` |
| `MemoryRequestMailbox` | yes | publish explicit memory requests to the memory module |
| `MemoryRequestInbox` | yes | subscribe to explicit memory requests |
| `UtteranceWriter` | yes | emit app-facing speech actions + persist/notify through an adapter |
| `Clock` | no | injected current time for timestamping and relative-time normalization |
| `TimeDivision` | no | load attention-boundary age buckets from `configs/time-division.eure` |

---

## 3. Scheduler And Event Loop

The scheduler only spawns modules:

```rust
pub async fn run(
    modules: Vec<Box<dyn Module>>,
    shutdown: impl Future<Output = ()>,
) -> Result<(), SchedulerError> {
    for mut module in modules {
        spawn_local(async move { module.run().await });
    }
    // wait for shutdown or task completion policy
}
```

The app owns an `AgentEventLoop` and advances time explicitly:

```rust
let mut loop_handle = AgentEventLoop::new(factory.periodic_activation());
loop_handle.tick(Duration::from_millis(20)).await;
```

Production convenience wrappers may call `tick(interval)` from a timer, but wall-clock sleep does not live inside module inboxes.

The boot layer injects time through `CapabilityFactory`:

```rust
let factory = CapabilityFactory::new(
    blackboard,
    attention_port,
    primary_memory_store,
    memory_replicas,
    file_search,
    tiers,
    Arc::new(SystemClock),
);

// Eval/sandbox boot may instead pass:
let factory = CapabilityFactory::new(..., Arc::new(FixedClock::new(observed_now)));
```

`CapabilityFactory` stores `Arc<dyn Clock>` and exposes `clock()` for modules that need current time. Capability handles that stamp time, such as `AttentionWriter` and `UtteranceWriter`, also receive the same clock. This keeps sandboxed eval deterministic and avoids direct `Utc::now()` calls in module code.

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

Reads the non-cognitive blackboard snapshot and appends concise, novel, currently relevant events to the attention stream. When promoting sensory memo content, summarize is the only place that applies time-division rounding: it converts the detailed memo age into the bucket/tag from `configs/time-division.eure` before writing the attention stream event.

### Attention Controller

Capabilities: `AttentionStreamUpdatedInbox`, optional `PeriodicInbox`, `AttentionReader`, `AllocationReader`, `AllocationWriter`, `Memo`, `LlmAccess`.

Reads only the attention stream and allocation. Writes the next allocation snapshot.

When the current attention suggests that the agent should gather evidence before speaking, the controller enters an evidence-gathering allocation phase: it sets `speak.enabled = false`, keeps or raises query-module eligibility/cadence/tier, and records the rationale in its memo. Query modules continue to write memo-authoritative results; summarize may later promote useful query memo content into the attention stream. Once the attention stream indicates that enough evidence is available for a user-visible answer, the controller writes `speak.enabled = true`.

This is not a request/response wait and not a query-completion correlation protocol. The controller judges readiness from attended state and allocation, while query results remain durable only through query-module memos.

### Attention Schema

Capabilities: `SelfModelInbox`, `PeriodicInbox`, `AttentionReader`, `Memo`, `LlmAccess`.

Maintains a simplified first-person self-model from the attention stream. Explicit self-model questions arrive on `SelfModelInbox`; periodic activation refreshes the model. Output is its memo.

### Query Vector

Capabilities: `QueryInbox`, `PeriodicInbox`, `BlackboardReader`, `VectorMemorySearcher`, `Memo`, `LlmAccess`.

Handles vector-memory/RAG queries only. Explicit query requests arrive on `QueryInbox`; periodic activation may refresh memory-oriented context. It does not handle self-referential or self-model questions. Output is its memo.

### Query Agentic

Capabilities: `QueryInbox`, `PeriodicInbox`, `BlackboardReader`, `FileSearcher`, `Memo`, `LlmAccess`.

Handles read-only file-search queries. Explicit query requests arrive on `QueryInbox`; periodic activation may refresh file-oriented context. Its file-search tool exposes only ripgrep-like controls: `pattern`, `regex`, `invert_match`, `case_sensitive`, `context`, and `max_matches`. It does not receive raw filesystem or shell access. Output is its memo.

### Memory

Capabilities: `PeriodicInbox`, `BlackboardReader`, `MemoryWriter`, `MemoryRequestInbox`, `LlmAccess`.

Decides whether useful information should be preserved and inserts memory entries. Explicit `MemoryRequest` messages from other modules (primarily surprise) are preservation candidates, not write commands. The memory module evaluates them with the current blackboard snapshot, may deduplicate or merge related candidates, and only persists records that the LLM chooses to write through `insert_memory`.

For request-derived short-term memories, `Normal` and `High` requests provide default decay hints of 1 day and 7 days respectively. The final decision, normalized content, rank, and decay belong to the memory module.

### Memory Compaction

Capabilities: `PeriodicInbox`, `BlackboardReader`, `MemoryCompactor`, `LlmAccess`.

Fetches related memory contents and merges redundant entries while accumulating remember tokens.

### Predict

Capabilities: `AttentionStreamUpdatedInbox`, `PeriodicInbox`, `AttentionReader`, `BlackboardReader`, `Memo`, `LlmAccess`.

Activates on attention updates and periodic ticks. Uses an LLM to generate predictions about the likely near-future states of currently attended subjects. Subjects are inferred from attention stream entries and may include external entities, conversational trajectory, or the agent's own mental state when attention-schema's memo reports self-directed attention. Each prediction entry includes the attended subject, predicted state, and an estimated validity horizon. Writes all predictions to its memo.

Predict does not detect divergence, does not send memory requests, and does not read allocation.

The v1 prediction memo schema is local to the predict module: a rationale plus entries containing subject, predicted state, validity horizon, and rationale.

### Surprise

Capabilities: `AttentionStreamUpdatedInbox`, `AttentionReader`, `BlackboardReader`, `MemoryRequestMailbox`, `Memo`, `LlmAccess`.

Activates on each attention update. Uses an LLM to assess whether the new attention entry is expected or surprising. When predict's memo is available in the blackboard, the assessment is framed as divergence from pending predictions. When predict is absent, the assessment is framed as novelty from recent attention history alone. When surprise is judged significant, sends a `MemoryRequest` and writes the assessment to its memo.

Surprise does not generate forward predictions and does not hold `PeriodicInbox` — its activation is fully attention-driven.

The v1 surprise threshold is represented by the structured LLM field `significant`; there is no numeric global threshold yet.

### Speak

Capabilities: `AttentionStreamUpdatedInbox`, optional `PeriodicInbox`, `AttentionReader`, `UtteranceWriter`, `Memo`, `Clock`, `LlmAccess`.

Emits user-visible text. The module is named `speak`, not `talk`, because it represents the concrete speech action capability rather than the whole conversational process.

Speak reads only the attention stream — it has no `BlackboardReader` and does not inspect other modules' memos. This keeps the utterance boundary narrow: speak distills what is attended, not the full blackboard state.

Speak is allocation-suppressed during evidence gathering. If it receives an attention update or periodic wake while `speak.enabled = false`, it records a pending speak wake without a decision turn, generation turn, memo write, delta, or complete utterance. When the controller later re-enables speak, that pending wake runs once against a fresh attention snapshot; it does not need another attention update just to start speaking.

Speak uses a two-stage LLM interaction:

1. **Decision turn** — `structured_turn::<SpeakDecision>().collect()`. Reads the current attention stream snapshot and determines whether a response is warranted (`should_respond`, `rationale`, optional generation hint). If `should_respond` is false, speak writes the decision to its memo and emits nothing.
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

Speak does not publish query or self-model requests. If an external observation has been attended but supporting work has not completed, the preferred control path is for attention-controller to suppress speak while query and summarization work progresses. If speak is enabled and the attended state is still insufficient, it may remain silent or produce a bounded clarification/failure utterance according to boot-time policy.

---

## 5. Lutum Integration

`LlmAccess::lutum()` returns a `lutum::Lutum` for the holder's currently allocated tier. Modules build their own `Session` and choose the concrete turn shape (`structured_turn`, `text_turn().tools()`, etc.). The capability layer deliberately does not impose a shared session or agent-loop abstraction.

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

- `MemoryStore`: replicated memory content plus adapter-owned search/indexing state. The primary store assigns `MemoryIndex` values; replica stores accept primary-assigned indexes.
- `FileSearchProvider`: read-only ripgrep-like file search with pattern, regex/literal mode, invert match, case sensitivity, context lines, and maximum match count.
- `AttentionRepository`: append-only attention stream persistence.
- `UtteranceSink`: append-only app-facing output persistence/notification for user-visible speech actions.
- `Clock`: injectable time source.
- Time-division config: `configs/time-division.eure`, an ordered set of duration buckets used only when summarize promotes sensory memo content into attention stream events. It converts observation age into tags such as `now`, `last_30sec`, `last_2min`, and `before_24hour`.

Desired Eure shape:

```eure
fallback-longest-tag = "before_24hour"

@ tags[] {
  tag = "now"
  range = { sec = 3 }
}

@ tags[] {
  tag = "last_30sec"
  range = { sec = 30 }
}

@ tags[] {
  tag = "last_2min"
  range = { min = 2.0 }
}
```

Memory content identity is owned by the primary `MemoryStore`. `MemoryWriter` inserts new content into the primary store first, then mirrors the primary-assigned `MemoryIndex` to replica stores with indexed writes. `MemoryCompactor` uses store-level atomic compaction: primary `compact(new, sources)` must assign the merged `MemoryIndex` and atomically create the merged record while removing all source records; replica `put_compacted(indexed, sources)` must atomically mirror that replacement for the primary-assigned id. Memory metadata is mirrored once on the blackboard after primary success. Primary write/compaction failure fails the operation before metadata changes. Secondary failures are logged. Concrete adapters may be native-only; WASM builds use adapter alternatives.

`UtteranceSink` is not a query response channel. It is an observable action log for host applications and eval harnesses. It exposes two notification surfaces:

- `on_complete(utterance: Utterance)` — a complete, timestamped utterance. Every compliant sink must implement this.
- `on_delta(delta: UtteranceDelta)` — a streaming chunk during generation. This method has a default no-op implementation; sinks that do not need progressive output ignore it. When `generation_id` changes between successive deltas, the sink discards its in-progress buffer and starts fresh — this is how interruption (attention-stream update or LLM retry) propagates to the display layer.

Implementations may persist utterances, stream deltas to UI, or both. The `on_complete` call always follows the full `on_delta` sequence for the same `generation_id`, so a UI adapter that consumed deltas can use `on_complete` as a framing signal rather than re-rendering the text.

---

## 7. Eval Integration

`crates/eval` keeps supporting module-level targets:

- `query` publishes `QueryRequest` and scores query-module memos/artifacts,
- `self-model` publishes `SelfModelRequest` and scores the attention-schema memo,
- `periodic` and `custom` remain low-level driver hooks.

Full-agent artifact evaluation should add a conversation-style target that:

1. boots the selected app wiring with `sensory`, `summarize`, query modules, `attention-schema`, `attention-controller`, memory modules, and `speak`,
2. publishes the case prompt as `SensoryInput::Heard { direction: Some("Ryo".into()), content, observed_at }`,
3. advances `AgentEventLoop::tick(elapsed)` until an utterance is emitted, the driver reaches a max tick budget, or shutdown is requested,
4. returns a `CaseArtifact` whose `output` is the collected user-visible utterance text (eval harnesses consume only the `on_complete` surface of `UtteranceSink`; streaming deltas are not collected and do not affect artifact output),
5. records observations such as sensory filtering decisions, memos, attention entries, allocation snapshots, and the utterance log under `CaseArtifact::observations`.

This keeps realistic artifacts observable without adding request/response correlation to module channels. The artifact is collected at the host boundary from `UtteranceSink`; intermediate reasoning remains memo/attention authoritative.

---

## 8. Invariants

| Invariant | Enforcement |
|---|---|
| Capabilities are non-exclusive | factory issues handles without uniqueness checks; any capability may be granted to multiple modules |
| Controller and schema are separate modules | separate crates and separate constructor capabilities |
| Controller cannot read non-cognitive blackboard | it receives no `BlackboardReader` |
| Sensory is the full-agent observation boundary | full-agent boot wiring grants `SensoryInputInbox` only to sensory |
| Sensory cannot answer, route work, or alter cognition directly | it receives no `QueryMailbox`, `SelfModelMailbox`, `UtteranceWriter`, `AttentionWriter`, `AllocationWriter`, memory capabilities, or readers |
| Speak is the full-agent utterance boundary | full-agent boot wiring grants `UtteranceWriter` only to speak |
| Speak cannot route work or mutate cognition | it receives no query/self-model mailbox, `AttentionWriter`, `AllocationWriter`, or memory capabilities |
| Speak reads only the attention stream | it receives `AttentionReader`, not `BlackboardReader` |
| Speak suppression is allocation-controlled | attention-controller writes `speak.enabled`; speak defers pending wakes through `ActivationGate` before decision/generation and emits only when enabled |
| Speak still does not route query work | evidence gathering is driven by allocation, query memos, and attention updates; speak receives no `QueryMailbox` |
| Speak interrupts streaming on attention updates | speak holds `AttentionStreamUpdatedInbox` and cancels the generation stream on receipt; generation restarts from step 1 with a fresh attention snapshot |
| `ResourceAllocation.enabled` gates all activations | periodic dispatch suppresses ticks, and modules defer or coalesce typed/attention activations after batching when their `ActivationGate` is disabled |
| Summarize is the only path to attention stream append | boot-time wiring (only summarize is granted `AttentionWriter`) |
| Only controller writes allocation | boot-time wiring (only attention-controller is granted `AllocationWriter`) |
| Query vector is memory/RAG only | it receives `VectorMemorySearcher`, not file or self-model capabilities |
| Query agentic is file-search only | it receives `FileSearcher`, not memory or self-model capabilities |
| Self-model queries go to attention-schema | callers need `SelfModelMailbox`; schema receives `SelfModelInbox` |
| Results are durable via memo, not responses | modules write their own `Memo`; channels are transient |
| Modules cannot impersonate each other | owner-stamped capabilities construct envelope senders and memo owners |
| Periodic activation is opt-in | only `periodic_inbox_for(module)` registers a module for ticks |
| All modules are detachable | boot wiring chooses which constructors and capabilities exist |
| Query ablations are wiring-only | boot wiring may include query-vector, query-agentic, both, or neither |
| Predict and surprise are separate modules | separate crates and separate constructor capabilities |
| Predict has no memory-request path | it receives no `MemoryRequestMailbox` |
| Surprise has no forward-modeling responsibility | it receives no memo path from predict; reads predict memo only via `BlackboardReader` |
| Surprise is the only module that sends `MemoryRequest` | boot-time wiring convention; factory does not enforce uniqueness |
| Predict and surprise ablations are wiring-only | boot wiring may include predict, surprise, both, or neither |
| Conversation artifacts are boundary observations | eval collects utterances from `UtteranceSink`, not channel responses |

Mailbox backpressure policy remains a separate runtime policy decision. Current typed topics use unbounded per-subscriber queues; if bounded queues are introduced, overflow policy must be explicit per topic. The source-of-truth state is not carried in channel payloads.
