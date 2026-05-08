# Attention Schema

An event loop runs multiple brain modules concurrently. Modules do not depend on each other directly; they cooperate through:

- a non-cognitive blackboard,
- a cognitive attention stream,
- typed transient channels,
- app-facing sensory/action ports,
- per-module memos.

A module's role is exactly the set of capabilities it is granted at construction. A module cannot perform an operation it was not granted, and cannot claim to act as another module: identity-bearing operations are owner-stamped by their capability handles.

The module set is open. The modules below are core roles, and additional domain modules can be added without changing the runtime.

## Agent States

### Attention Stream

The cognitive surface. Entries are appended only by the summarize module — the privileged holder of the attention-stream-write capability.

Stream entry: `(time, event)`.

### Blackboard

The non-cognitive blackboard holds:

- **per-module memos** — each module owns one slot and writes only its own slot,
- **memory metadata** — rank, decay, access counts, and remember tokens,
- **resource allocation** — enabled/period/model-tier/context-budget per module.

Module results that should be durable live in memos or other blackboard state. Channel messages are transient activation signals and are not persisted.

External input and output sit at the boundary of cognition:

- **sensory input** is the only external/app-facing input; it enters through a typed `SensoryInput` channel and is normalized by the sensory module into its memo,
- **speech output** leaves through a speak/action capability and is mirrored into the speak module's memo.

`QueryRequest` and `SelfModelRequest` are internal typed messages. Module-level eval harnesses may publish them to isolate query modules or attention-schema, but full-agent/app-facing cases must enter through `SensoryInput`.

Sensory input is multimodal. Initial built-in observations are `Heard { direction, content, observed_at }` for linguistic/auditory input and `Seen { direction, appearance, observed_at }` for visual input. The variant names are past-tense observations rather than actions: `Listen` would describe what the agent does, while `Heard` describes what arrived at the sensory boundary; `Visible` would describe a property, while `Seen` describes an observed event.

`observed_at` is the host-observed datetime of the stimulus. When the sensory module writes its memo, it compares the current clock time to `observed_at` and keeps a detailed human-readable relative age. The memo should preserve detail, for example `Ryo said "..." 1 minute 20 seconds ago`, not a time-division bucket and not a raw turn id.

Time-division rounding belongs at the attention boundary. When summarize decides that sensory memo content should enter the attention stream, it maps the observation age through `configs/time-division.eure` and writes the rounded bucket/tag into the attention stream event.

The attention stream remains the cognitive surface. Sensory input is not appended to attention directly; summarize notices the sensory memo through the blackboard and decides what should become attended.

### Memories

A memory is `(rank, content, decay, query_history, remember_tokens)`.

- **rank**: short-term, mid-term, long-term, permanent
- **query history**: recent access log
- **remember tokens**: accumulated by memory compaction when repeated memories merge
- **decay**: remaining time to stay in the current rank

Rank can rise through access or remember-token accumulation and fall when decay expires.

## Activations

A module can be activated by any inbox capability it holds:

- typed fanout topics, such as external `SensoryInputInbox` or internal `QueryInbox` / `SelfModelInbox`,
- `PeriodicInbox`, driven by elapsed application ticks and current allocation,
- `AttentionStreamUpdatedInbox`, published by attention-stream writes.

Periodic activation is opt-in. A module becomes periodic only when boot wiring grants it `PeriodicInbox`. The event loop is elapsed-time based; it does not sleep inside inboxes.

## Capabilities

| Module | Read blackboard | Read attention stream | Memo | Periodic | Clock | LLM | Special capabilities |
|---|---|---|---|---|---|---|---|
| sensory | — | — | ✓ | — | ✓ | ✓ | `SensoryInputInbox` |
| summarize | ✓ | — | ✓ | ✓ | — | ✓ | `AttentionWriter`, `TimeDivision` |
| attention-controller | — | ✓ | ✓ | optional | — | ✓ | `AttentionStreamUpdatedInbox`, `AllocationReader`, `AllocationWriter` |
| attention-schema | — | ✓ | ✓ | ✓ | — | ✓ | `SelfModelInbox` |
| query-vector | ✓ | — | ✓ | ✓ | — | ✓ | `QueryInbox`, `VectorMemorySearcher` |
| query-agentic | ✓ | — | ✓ | ✓ | — | ✓ | `QueryInbox`, `FileSearcher` |
| memory | ✓ | — | optional | ✓ | — | ✓ | `MemoryWriter`, `MemoryRequestInbox` |
| memory-compaction | ✓ | — | optional | ✓ | — | ✓ | `MemoryCompactor` |
| predict | ✓ | ✓ | ✓ | ✓ | — | ✓ | `AttentionStreamUpdatedInbox` |
| surprise | ✓ | ✓ | ✓ | — | — | ✓ | `AttentionStreamUpdatedInbox`, `MemoryRequestMailbox` |
| speak | — | ✓ | ✓ | optional | ✓ | ✓ | `AttentionStreamUpdatedInbox`, `UtteranceWriter` |

Notable absences:

- The attention controller cannot read the non-cognitive blackboard.
- The attention schema module cannot read allocation and does not drive resource policy.
- Query modules do not receive self-model requests and do not answer self-referential questions.
- The sensory module is the app-facing observation boundary; it cannot write attention, publish work requests, or emit utterances.
- The speak module is the app-facing output boundary; it cannot write attention, allocation, memory, or query/self-model requests.
- Only summarize can write the attention stream.
- Only the attention controller can write resource allocation.
- The memory module cannot increment remember tokens; that belongs to memory compaction.
- Predict does not send memory requests; that belongs to surprise.
- Surprise does not generate forward predictions; that belongs to predict.

## Modules

### Sensory

Receives external observations through `SensoryInputInbox`, normalizes them, and writes selected observations to the sensory memo. It is the only observation boundary for realistic eval runs and application integrations.

Sensory is a pre-attentive filter. Its role is to decide which stimuli should be made available to the rest of the system through its memo, not to decide which cognitive module should work next.

It has two stages:

1. deterministic salience scoring,
2. LLM-based ignore/summarize decision.

The deterministic stage maintains stimulus habituation and decay:

- repeated low-change stimuli lose salience,
- old stimuli decay unless refreshed,
- novel, user-directed, intense, or changed stimuli gain salience.

The LLM stage receives the raw observation plus deterministic salience features and decides whether to ignore it, fold it into a background summary, or write a concise normalized observation to the sensory memo.

The sensory memo is the only output. It should contain the filtered, normalized observations that other modules may inspect through the blackboard path, plus enough inspection detail to explain salience and the LLM decision. It should not mirror every raw input event. Sensory does not answer the user, append to attention, publish query/self-model requests, or mutate shared state beyond its memo.

### Summarize

Reads the non-cognitive blackboard snapshot and appends to the attention stream when something is novel, unresolved, or strongly changed. It is the only bridge from the non-cognitive blackboard to the cognitive surface.

When sensory memo content is promoted, summarize rounds the detailed memo age through `configs/time-division.eure` before appending the attention event. Detailed timing stays in the sensory memo; rounded timing belongs to the attention stream.

### Attention Controller

Reads the attention stream and current allocation, then writes the next resource allocation: enabled state, periodic cadence, model tier, and context budget. It is intentionally unaware of non-cognitive detail.

### Attention Schema

Maintains a simplified first-person model of what the agent is attending to: external input, thought, memory, goal, and the self's relation to those contents. It handles explicit self-model questions through `SelfModelInbox` and writes self-model / self-report output to its memo.

The self-model is a simplification. It may diverge from the controller's internal allocation state; that gap is part of the architecture.

### Query Vector

Handles vector-memory/RAG queries only. Explicit internal query requests arrive through `QueryInbox`; periodic activation can refresh memory-oriented context. Output is written to the query-vector module's memo.

### Query Agentic

Handles read-only file-search queries only. Explicit internal query requests arrive through `QueryInbox`; periodic activation can refresh file-oriented context. Output is written to the query-agentic module's memo.

### Memory

Preserves useful information by inserting memory entries based on the blackboard snapshot. Explicit
`MemoryRequest` messages are preservation candidates, not writes. The memory module may reject,
normalize, merge, or deduplicate candidates, and only persists records through its own
`insert_memory` tool decision.

### Memory Compaction

Fetches related memory contents and merges redundant memories, accumulating remember tokens.

### Predict

Maintains forward predictions about the current attention targets. On each attention stream update or periodic tick, reads the attention stream and blackboard context, then uses an LLM to generate predictions about the likely near-future states of whatever is currently attended — external subjects (people, objects, events), conversational trajectory, or the agent's own mental state when attention-schema reports that the agent is attending to its own mind. Each prediction entry includes the attended subject, predicted state, and an estimated validity horizon.

Predictions are written to the predict memo. Predict does not detect whether its predictions were correct and does not send memory requests — divergence detection belongs to surprise.

### Surprise

Detects unexpected attention events by comparing new attention stream entries against the predict memo (if present) or against recent attention history alone (if predict is absent). Uses an LLM to assess the degree of divergence or novelty. When surprise is significant, sends a `MemoryRequest` candidate to the memory module and records the surprise assessment in its memo.

Surprise-triggered requests carry `Normal` / `High` importance as memory-module hints, not storage commands. The memory module owns the final preservation decision.

Surprise does not generate predictions. When the predict memo is absent, surprise acts as a novelty detector over the attention stream — it judges unexpectedness from recent history rather than from explicit predictions.

### Speak

Emits user-visible utterances. The module is named `speak` rather than `talk` because its role is the action of producing an utterance, not owning the whole conversation.

Speak reads the attention stream, decides whether a user-facing response is warranted, writes the chosen utterance to its memo, and emits it through `UtteranceWriter` so the application or eval harness can collect the utterance as an artifact.

Speak is not a planner or router. It does not publish query or self-model work, and it does not make resource-allocation decisions. If required information is not yet available in attended state or memos, it should remain silent or emit a bounded clarification/failure utterance according to application policy.

## Invariants

These invariants are upheld by boot-time capability wiring and owner-stamped handles:

- The attention controller and attention schema module are separate modules.
- The attention controller cannot directly read the non-cognitive blackboard.
- The sensory module is the canonical app-facing path for external observations in full-agent runs.
- Query and self-model messages are internal; they are only driven directly by module-level eval harnesses or internal modules that hold those mailbox capabilities.
- The speak module is the canonical app-facing path for user-visible utterances in full-agent runs.
- The summarize module is the only path from the non-cognitive blackboard to the attention stream.
- Query modules and self-model questions are separated by typed channels.
- Durable answers are memo-authoritative, not mailbox responses.
- A module cannot impersonate another module.
- Periodic activation is granted explicitly.
- All modules are detachable.
- Query-vector and query-agentic are independently detachable for ablation.
- Predict and surprise are independently detachable for ablation.
- Ablating the attention schema module should degrade self-report specifically while task performance largely survives.
- Ablating sensory or speak should degrade end-to-end artifact evaluation while leaving lower-level query and self-model module evaluations possible.
- Ablating predict degrades surprise to attention-history novelty detection while leaving the memory-request path intact.
- Ablating surprise disables prediction-failure learning while leaving forward predictions available to other modules.
- Ablating both predict and surprise removes the prediction loop entirely without affecting other modules.
