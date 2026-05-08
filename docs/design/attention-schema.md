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

The cognitive surface. Entries are appended only by the attention-gate module — the privileged holder of the attention-stream-write capability.

Stream entry: `(time, event)`.

### Blackboard

The non-cognitive blackboard holds:

- **per-module memos** — each module owns one slot and writes only its own slot,
- **memory metadata** — rank, decay, access counts, and remember tokens,
- **resource allocation** — activation ratio, controller guidance, and model tier per module.

Module results that should be durable live in memos or other blackboard state. Channel messages are transient activation signals and are not persisted.

External input and output sit at the boundary of cognition:

- **sensory input** is the only external/app-facing input; it enters through a typed `SensoryInput` channel and is normalized by the sensory module into its memo,
- **speech output** leaves through a speak/action capability and is mirrored into the speak module's memo.

`QueryRequest` and `SelfModelRequest` are internal typed messages. Module-level eval harnesses may publish them to isolate query modules or the self-model module, and may seed attention-stream entries to exercise attention consumers. Full-agent/app-facing cases must enter through `SensoryInput`.

Sensory input is multimodal. Initial built-in observations are `Heard { direction, content, observed_at }` for linguistic/auditory input and `Seen { direction, appearance, observed_at }` for visual input. The variant names are past-tense observations rather than actions: `Listen` would describe what the agent does, while `Heard` describes what arrived at the sensory boundary; `Visible` would describe a property, while `Seen` describes an observed event.

`observed_at` is the host-observed datetime of the stimulus. When the sensory module writes its memo, it compares the current clock time to `observed_at` and keeps a detailed human-readable relative age. The memo should preserve detail, for example `Ryo said "..." 1 minute 20 seconds ago`, not a time-division bucket and not a raw turn id.

Time-division rounding belongs at the attention boundary. When attention-gate decides that sensory memo content should enter the attention stream, it maps the observation age through `configs/time-division.eure` and writes the rounded bucket/tag into the attention stream event.

The attention stream remains the cognitive surface. Sensory input is not appended to attention directly; attention-gate notices the sensory memo through the blackboard and decides what should become attended.

The design uses [Graziano's attention-schema framing](https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2017.00060/full) as an engineering analogy: object/world models, self-models, and the attention schema are distinct internal models. The attention schema models the act and dynamics of attention; the self-model integrates that attentional relation with stable and current facts about the agent.

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
- `MemoUpdatedInbox`, published by memo writes and filtered so a holder does not wake on its own writes,
- `AllocationUpdatedInbox`, published when effective allocation or guidance changes,
- `AttentionStreamUpdatedInbox`, published by attention-stream writes and by the event loop when it records an agentic-deadlock marker.

There is no periodic wake mechanism. Allocation guidance is the controller's durable control plane: modules read the current allocation snapshot and decide whether a wake should produce work, defer silently, or only update local state.

Guidance-based allocation flow:

```text
Sensory memo -> AttentionController -> AllocationUpdated -> AttentionGate -> AttentionStreamUpdated -> Speak
Query/Speak/etc memo -> AttentionController -> AllocationUpdated -> modules
```

Attention-gate does not write a memo. Attention appended by attention-gate wakes attention consumers such as attention-schema, Speak, Predict, and Surprise, but does not directly wake the controller. Speak writes a completion memo after a finished utterance; that memo wakes the controller so it can reduce `speak.activation_ratio` and reallocate resources.

## Capabilities

| Module | Read blackboard | Read attention stream | Read allocation | Memo | Clock | LLM | Special capabilities |
|---|---|---|---|---|---|---|---|
| sensory | — | — | ✓ | ✓ | ✓ | ✓ | `SensoryInputInbox` |
| attention-gate | ✓ | — | ✓ | — | — | ✓ | `AllocationUpdatedInbox`, `AttentionWriter`, `TimeDivision` |
| attention-controller | ✓ | ✓ | ✓ | ✓ | — | ✓ | `MemoUpdatedInbox`, `AllocationWriter` |
| attention-schema | — | ✓ | — | ✓ | — | ✓ | `AttentionStreamUpdatedInbox` |
| self-model | ✓ | — | — | ✓ | — | ✓ | `SelfModelInbox` |
| query-vector | ✓ | — | ✓ | ✓ | — | ✓ | `QueryInbox`, `AllocationUpdatedInbox`, `VectorMemorySearcher` |
| query-agentic | ✓ | — | ✓ | ✓ | — | ✓ | `QueryInbox`, `AllocationUpdatedInbox`, `FileSearcher` |
| memory | ✓ | — | ✓ | — | — | ✓ | `AllocationUpdatedInbox`, `MemoryWriter`, `MemoryRequestInbox` |
| memory-compaction | ✓ | — | ✓ | — | — | ✓ | `AllocationUpdatedInbox`, `MemoryCompactor` |
| predict | ✓ | ✓ | ✓ | ✓ | — | ✓ | `AttentionStreamUpdatedInbox`, `AllocationUpdatedInbox` |
| surprise | ✓ | ✓ | ✓ | ✓ | — | ✓ | `AttentionStreamUpdatedInbox`, `MemoryRequestMailbox` |
| speak | — | ✓ | ✓ | ✓ | ✓ | ✓ | `AttentionStreamUpdatedInbox`, `AllocationUpdatedInbox`, `UtteranceWriter` |

Notable absences:

- The attention schema module has no self-model inbox, allocation-write path, memory-write path, or attention-write path. It models attention only.
- The self-model module handles explicit self-model questions, but has no attention-write, allocation-write, or memory-write path.
- Query modules do not receive self-model requests and do not answer self-referential questions.
- The sensory module is the app-facing observation boundary; it cannot write attention, publish work requests, or emit utterances.
- The speak module is the app-facing output boundary; it cannot write attention, allocation, memory, or query/self-model requests.
- Only attention-gate can write the attention stream.
- Attention-gate cannot write memo, so attention appends cannot wake the controller directly.
- The attention controller wakes only on memo updates, excluding its own memo writes; it reads the attention stream but is not woken by `AttentionStreamUpdated`.
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
2. LLM-based ignore/background-summary decision.

The deterministic stage maintains stimulus habituation and decay:

- repeated low-change stimuli lose salience,
- old stimuli decay unless refreshed,
- novel, user-directed, intense, or changed stimuli gain salience.

The LLM stage receives the raw observation plus deterministic salience features and decides whether to ignore it, fold it into a background summary, or write a concise normalized observation to the sensory memo.

The sensory memo is the only output. It should contain the filtered, normalized observations that other modules may inspect through the blackboard path, plus enough inspection detail to explain salience and the LLM decision. It should not mirror every raw input event. Sensory does not answer the user, append to attention, publish query/self-model requests, or mutate shared state beyond its memo.

### Attention Gate

Reads the non-cognitive blackboard snapshot and current allocation guidance, then appends to the attention stream when something is novel, unresolved, strongly changed, or requested by controller guidance. It is the only bridge from the non-cognitive blackboard to the cognitive surface, and it has no memo capability. This role is the successor to the older `summarize` design name: selection and promotion to attention are the essence of the module, not generic summarization.

When sensory memo content is promoted, attention-gate rounds the detailed memo age through `configs/time-division.eure` before appending the attention event. Detailed timing stays in the sensory memo; rounded timing belongs to the attention stream.

### Attention Controller

Wakes only on memo updates from other modules. It reads the blackboard memo set, memory metadata, attention stream, and current allocation, then writes the next resource allocation for every registered module: `activation_ratio`, natural-language `guidance`, and `tier`.

The controller uses guidance rather than request/response correlation. For example, if Speak should wait for query or attention-gate work, the controller raises the relevant modules and gives Speak guidance to wait silently until attended evidence is sufficient.

### Attention Schema

Maintains a simplified model of the current attentional relation and process from the attention stream. It tracks what is attended, whether that attention is stable or contested, what competing targets are present, and what the current attention makes claimable as awareness.

The attention schema is not a self-model and does not answer self-model requests. Its memo is an attention-state model that other modules, especially self-model, may use. It may diverge from the controller's internal allocation state; that gap is part of the architecture.

### Self Model

Maintains a current self-description by integrating the attention-schema memo, relevant module memos, and self-related knowledge that query modules surface from memory. It handles explicit self-model questions through `SelfModelInbox` and writes self-model / self-report output to its memo.

Stable self-knowledge belongs in memory and can be retrieved by query modules, but a live self-model is not just memory retrieval. The self-model module turns long-lived self facts, current attention, active task context, uncertainty, and recent module outputs into a current first-person state. In v1, object/world models remain distributed through sensory, query, predict, and surprise memos; a dedicated `world-model` can be added later if those fragments need a single owner.

### Query Vector

Handles vector-memory/RAG queries only. Explicit internal query requests arrive through `QueryInbox`; allocation updates can also wake it to act on controller guidance. Its memo contains only retrieved memory content, copied from query results; it does not synthesize answers or describe itself.

### Query Agentic

Handles read-only file-search queries only. Explicit internal query requests arrive through `QueryInbox`; allocation updates can also wake it to act on controller guidance. Its memo contains only retrieved file content/snippets, copied from query results; it does not synthesize answers or describe itself.

### Memory

Preserves useful information by inserting memory entries based on the blackboard snapshot and allocation guidance. Explicit `MemoryRequest` messages are preservation candidates, not writes. The memory module may reject, normalize, merge, or deduplicate candidates, and only persists records through its own `insert_memory` tool decision.

### Memory Compaction

Fetches related memory contents and merges redundant memories, accumulating remember tokens. Allocation updates wake it to consider compaction guidance.

### Predict

Maintains forward predictions about the current attention targets. On each attention stream update or allocation update, reads the attention stream, blackboard context, and allocation guidance, then uses an LLM to generate predictions about the likely near-future states of whatever is currently attended — external subjects (people, objects, events), conversational trajectory, or the agent's own mental state when attention-schema reports that the agent is attending to its own mind. Each prediction entry includes the attended subject, predicted state, and an estimated validity horizon.

Predictions are written to the predict memo. Predict does not detect whether its predictions were correct and does not send memory requests — divergence detection belongs to surprise.

### Surprise

Detects unexpected attention events by comparing new attention stream entries against the predict memo (if present) or against recent attention history alone (if predict is absent). Uses an LLM to assess the degree of divergence or novelty with allocation guidance in context. When surprise is significant, sends a `MemoryRequest` candidate to the memory module and records the surprise assessment in its memo.

Surprise-triggered requests carry `Normal` / `High` importance as memory-module hints, not storage commands. The memory module owns the final preservation decision.

Surprise does not generate predictions. When the predict memo is absent, surprise acts as a novelty detector over the attention stream — it judges unexpectedness from recent history rather than from explicit predictions.

### Speak

Emits user-visible utterances. The module is named `speak` rather than `talk` because its role is the action of producing an utterance, not owning the whole conversation.

Speak reads the attention stream and allocation guidance, decides whether a user-facing response is warranted, writes the chosen utterance to its memo, and emits it through `UtteranceWriter` so the application or eval harness can collect the utterance as an artifact.

Speak is not a planner or router. It does not publish query or self-model work, and it does not make resource-allocation decisions. If allocation guidance indicates query or attention-gate work is still in progress, Speak may wait silently. Completed utterance memos wake the controller so resources can be reclaimed.

## Invariants

These invariants are upheld by boot-time capability wiring and owner-stamped handles:

- The attention controller, attention schema, and self-model modules are separate modules.
- The sensory module is the canonical app-facing path for external observations in full-agent runs.
- Query and self-model messages are internal; they are only driven directly by module-level eval harnesses or internal modules that hold those mailbox capabilities.
- The speak module is the canonical app-facing path for user-visible utterances in full-agent runs.
- The attention-gate module is the only path from the non-cognitive blackboard to the attention stream.
- Query modules and self-model questions are separated by typed channels.
- Durable answers are memo-authoritative, not mailbox responses.
- A module cannot impersonate another module.
- Attention-gate has no memo capability and cannot wake the controller by appending attention.
- The attention controller is not woken by attention stream updates.
- Allocation activation ratios respect boot-time `cap_range.min/max`; modules with `cap_range.min = 0` are detachable by allocation.
- Query-vector and query-agentic are independently detachable for ablation.
- Predict and surprise are independently detachable for ablation.
- Ablating the attention schema module should degrade attention-state modeling while task performance largely survives.
- Ablating the self-model module should degrade self-report specifically while task performance largely survives.
- Ablating sensory or speak should degrade end-to-end artifact evaluation while leaving lower-level query, attention-schema, and self-model module evaluations possible.
- Ablating predict degrades surprise to attention-history novelty detection while leaving the memory-request path intact.
- Ablating surprise disables prediction-failure learning while leaving forward predictions available to other modules.
- Ablating both predict and surprise removes the prediction loop entirely without affecting other modules.
