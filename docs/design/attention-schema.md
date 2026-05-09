# Attention Schema

An event loop runs multiple brain modules concurrently. Modules do not depend on each other directly; they cooperate through:

- a non-cognitive blackboard,
- a cognition log,
- typed transient channels,
- app-facing sensory/action ports,
- per-module memos.

A module's role is exactly the set of capabilities it is granted at construction. A module cannot perform an operation it was not granted, and cannot claim to act as another module: identity-bearing operations are owner-stamped by their capability handles.

The module set is open. The modules below are core roles, and additional domain modules can be added without changing the runtime.

## Agent States

### Cognition Log

The admitted cognitive surface. Cognition-gate promotes selected non-cognitive memo/blackboard state into this surface. Attention-schema may also append first-person attention experience entries when the current attention state itself should become cognitively available.

Log entry: `(time, event)`.

### Blackboard

The non-cognitive blackboard holds:

- **per-module memo queues** — each module owner writes only its own bounded indexed log,
- **memory metadata** — rank, decay, access counts, and remember tokens,
- **identity memory snapshot** — identity-ranked memories loaded once at agent startup,
- **resource allocation** — activation ratio, controller guidance, and model tier per module.

Module results that should be durable live in memo logs or other blackboard state. Each memo item has a per-owner monotonic index; reader handles track their own last-seen index so modules can distinguish unread output from recent context. Memo content is free-form prose on the shared workspace surface, not JSON, schema-shaped data, or a structured data exchange channel. Channel messages are transient activation signals and are not persisted.

External input and output sit at the boundary of cognition:

- **sensory input** is the only external/app-facing input; it enters through a typed `SensoryInput` channel and is normalized by the sensory module into its memo,
- **speech output** leaves through a speak/action capability and is mirrored into the speak module's memo.

`QueryRequest` and `SelfModelRequest` are internal typed messages. Module-level eval harnesses may publish them to isolate query modules or the self-model module, and may seed cognition-log entries to exercise cognition-log consumers. Full-agent/app-facing cases must enter through `SensoryInput`.

Sensory input is multimodal. Initial built-in observations are `Heard { direction, content, observed_at }` for linguistic/auditory input and `Seen { direction, appearance, observed_at }` for visual input. The variant names are past-tense observations rather than actions: `Listen` would describe what the agent does, while `Heard` describes what arrived at the sensory boundary; `Visible` would describe a property, while `Seen` describes an observed event.

`observed_at` is the host-observed datetime of the stimulus. When the sensory module writes its memo, it compares the current clock time to `observed_at` and keeps a detailed human-readable relative age. The memo should preserve detail, for example `Ryo said "..." 1 minute 20 seconds ago`, not a time-division bucket and not a raw turn id.

Time-division rounding belongs at the cognition boundary. When cognition-gate decides that sensory memo content should enter the cognition log, it maps the observation age through `configs/time-division.eure` and writes the rounded bucket/tag into the cognition log event.

The cognition log remains the admitted cognitive surface. Sensory input is not appended to the cognition log directly; cognition-gate notices the sensory memo through the blackboard and decides what should become relevant to cognitive processing. Attention-schema writes only concise first-person attention experience entries to the cognition log; it does not create a separate attention-state memo.

The design uses [Graziano's attention-schema framing](https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2017.00060/full) as an engineering analogy: object/world models, self-models, and the attention schema are distinct internal models. The attention schema models the act and dynamics of attention; the self-model integrates that attentional relation with stable and current facts about the agent.

### Memories

A memory is `(rank, content, decay, query_history, remember_tokens)`.

- **rank**: short-term, mid-term, long-term, permanent, identity
- **query history**: recent access log
- **remember tokens**: accumulated by memory compaction when repeated memories merge
- **decay**: remaining time to stay in the current rank

Rank can rise through access or remember-token accumulation and fall when decay expires.
Identity-ranked memories are startup identity facts: the agent loads all of them from the primary
memory store before module activation and injects that stable snapshot into LLM system prompts.

## Activations

A module can be activated by any inbox capability it holds:

- typed fanout topics, such as external `SensoryInputInbox` or internal `QueryInbox` / `SelfModelInbox`,
- `MemoUpdatedInbox`, published by memo writes and filtered so a holder does not wake on its own writes,
- `AllocationUpdatedInbox`, published when effective allocation or guidance changes,
- `CognitionLogUpdatedInbox`, published by cognition-log writes and by the event loop when it records an agentic-deadlock marker, and filtered so a holder does not wake on its own cognition-log writes.

There is no periodic wake mechanism. Allocation guidance is the controller's durable control plane: modules read the current allocation snapshot and decide whether a wake should produce work, defer silently, or only update local state.

Guidance-based allocation flow:

```text
Sensory memo -> AttentionController -> allocation proposal
Sensory memo -> CognitionGate -> CognitionLogUpdated -> SpeakGate
SpeakGate -> Query/SelfModel/SensoryDetail evidence requests
Surprise -> MemoryRequest -> Memory
SpeakGate -> SpeakRequest -> Speak
SpeakGate memo -> AttentionController -> allocation proposal
CognitionLogUpdated -> Query/Memory/Predict/Surprise/AttentionSchema
```

Cognition-gate does not write a memo. Cognition-log entries wake cognition-log consumers such as attention-schema, SpeakGate, Predict, and Surprise, but a module's own cognition-log write does not wake that same module, and cognition-log updates do not directly wake the controller or Speak. SpeakGate sends a typed one-shot `SpeakRequest` to Speak when the cognition log is speech-ready or when a current utterance should be replaced. When SpeakGate waits, its decision and missing-evidence notes are durable only in its memo; the memo update wakes the controller through the ordinary memo path. Speak writes a completion memo after a finished utterance.

## Capabilities

| Module | Read blackboard | Read cognition log | Read allocation | Memo | Clock | LLM | Special capabilities |
|---|---|---|---|---|---|---|---|
| sensory | — | — | ✓ | ✓ | ✓ | ✓ | `SensoryInputInbox`, `SensoryDetailRequestInbox` |
| cognition-gate | ✓ | — | ✓ | — | — | ✓ | `MemoUpdatedInbox`, `AllocationUpdatedInbox`, `CognitionWriter`, `TimeDivision` |
| attention-controller | ✓ | ✓ | ✓ | ✓ | — | ✓ | `MemoUpdatedInbox`, `AllocationWriter` |
| attention-schema | ✓ | ✓ | ✓ | — | — | ✓ | `MemoUpdatedInbox`, `AllocationUpdatedInbox`, `CognitionLogUpdatedInbox`, `CognitionWriter` |
| self-model | ✓ | ✓ | — | ✓ | — | ✓ | `SelfModelInbox` |
| query-vector | ✓ | — | ✓ | ✓ | — | ✓ | `QueryInbox`, `CognitionLogUpdatedInbox`, `VectorMemorySearcher` |
| query-agentic | ✓ | — | ✓ | ✓ | — | ✓ | `QueryInbox`, `AllocationUpdatedInbox`, `FileSearcher` |
| memory | ✓ | — | ✓ | — | — | ✓ | `CognitionLogUpdatedInbox`, `MemoryRequestInbox`, `MemoryWriter` |
| memory-compaction | ✓ | — | ✓ | — | — | ✓ | `AllocationUpdatedInbox`, `MemoryCompactor` |
| predict | ✓ | ✓ | ✓ | ✓ | — | ✓ | `CognitionLogUpdatedInbox` |
| surprise | ✓ | ✓ | ✓ | ✓ | — | ✓ | `CognitionLogUpdatedInbox`, `MemoryRequestMailbox` |
| speak-gate | ✓ | ✓ | — | ✓ | — | ✓ | `CognitionLogUpdatedInbox`, `ModuleStatusReader`, `QueryMailbox`, `SelfModelMailbox`, `SensoryDetailRequestMailbox`, `SpeakMailbox` |
| speak | — | ✓ | — | ✓ | ✓ | ✓ | `SpeakInbox`, `UtteranceWriter` |

Notable absences:

- The attention schema module has no memo, self-model inbox, allocation-write path, or memory-write path. Its durable output is limited to first-person attention experience entries in the cognition log.
- The self-model module handles explicit self-model questions, but has no cognition-log-write, allocation-write, or memory-write path.
- Query modules do not receive self-model requests and do not answer self-referential questions.
- The sensory module is the app-facing observation boundary; it cannot write cognition-log entries, publish work requests, or emit utterances.
- The speak-gate module can inspect memos and call evidence lookup tools to decide speech readiness, but cannot emit utterances, write cognition-log entries, change allocation, or write memory.
- The speak module is the app-facing output boundary; it cannot read blackboard memos or allocation guidance, and cannot write cognition-log entries, allocation, memory, or query/self-model requests.
- Only modules granted `CognitionWriter` can append cognition-log entries; current boot wiring grants it to cognition-gate and attention-schema.
- Cognition-log appends cannot wake the controller directly; the controller wakes on memo updates.
- The attention controller wakes on memo updates; it reads the cognition log but is not woken by `CognitionLogUpdated`.
- Only the attention controller can write resource allocation.
- The memory module cannot increment remember tokens; that belongs to memory compaction.
- Predict does not write memory; preservation belongs to surprise requests and the memory module.
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

The sensory memo is the only output. It should contain the filtered, normalized observations that other modules may inspect through the blackboard path, plus enough inspection detail to explain salience and the LLM decision. It should not mirror every raw input event. Sensory does not answer the user, append to the cognition log, publish query/self-model requests, or mutate shared state beyond its memo.

### Cognition Gate

Reads the non-cognitive blackboard snapshot and current allocation guidance, then appends to the cognition log when something is novel, unresolved, strongly changed, or requested by controller guidance. It is the only bridge from the non-cognitive blackboard to the cognitive surface, and it has no memo capability.

When sensory memo content is promoted, cognition-gate rounds the detailed memo age through `configs/time-division.eure` before appending the cognition-log entry. Detailed timing stays in the sensory memo; rounded timing belongs to the cognition log.

### Attention Controller

Wakes only on memo updates from other modules. It reads the blackboard memo set, memory metadata, cognition log, and current allocation, then writes the next resource allocation for every registered module: `activation_ratio`, natural-language `guidance`, and `tier`.

The controller uses guidance rather than request/response correlation. For example, if speech should wait for query or cognition-gate work, the controller raises the relevant modules and lets SpeakGate keep speech silent until attended evidence is sufficient. Speak itself does not parse allocation guidance.

### Attention Schema

Reads the aggregate memo surface, current allocation, and cognition log to decide whether the current attention state should become admitted cognitive evidence. It keeps a persistent LLM session so repeated activations share decision context. When a new, claimable, cognitively useful attention experience exists, it appends a concise first-person cognition-log entry through a plaintext tool call. When nothing new should be admitted, it calls no tool and writes nothing.

The attention schema is not a self-model and does not answer self-model requests. It assumes a non-physical experiencer that can direct attention to any target and freely control that attention, but its appended text should avoid mechanical internals and decision noise. It may diverge from the controller's internal allocation state; that gap is part of the architecture.

### Self Model

Maintains a current self-description by integrating attention-schema cognition-log entries, relevant module memos, and self-related knowledge that query modules surface from memory. It handles explicit self-model questions through `SelfModelInbox` and writes self-model / self-report output to its memo.

Stable self-knowledge belongs in memory and can be retrieved by query modules, but a live self-model is not just memory retrieval. The self-model module turns long-lived self facts, current attention, active task context, uncertainty, and recent module outputs into a current first-person state. In v1, object/world models remain distributed through sensory, query, predict, and surprise memos; a dedicated `world-model` can be added later if those fragments need a single owner.

### Query Vector

Handles vector-memory/RAG queries only. Explicit internal query requests arrive through `QueryInbox`; cognition-log updates can also wake it to retrieve memory relevant to the current cognitive surface. Its memo contains only retrieved memory content, copied from query results; it does not synthesize answers or describe itself.

### Query Agentic

Handles read-only file-search queries only. Explicit internal query requests arrive through `QueryInbox`; allocation updates can also wake it to act on controller guidance. Its memo contains only retrieved file content/snippets, copied from query results; it does not synthesize answers or describe itself.

### Memory

Preserves useful information by inserting memory entries after cognition-log updates or explicit surprise memory requests. It inspects the current cognition log plus indexed unread/recent memo logs as candidate evidence. Attention-schema entries are ordinary cognition-log evidence when the current attention state matters. `MemoryRequest` messages are preservation candidates, not writes. The memory module may reject, normalize, merge, or deduplicate candidates, and only persists records through its own `insert_memory` tool decision.

### Memory Compaction

Fetches related memory contents and merges redundant memories, accumulating remember tokens. Allocation updates wake it to consider compaction guidance.

### Predict

Maintains forward predictions about current cognition-log targets. On each cognition log update, reads the cognition log, blackboard context, and allocation guidance, then uses an LLM to generate predictions about the likely near-future states of the admitted cognitive targets — external subjects (people, objects, events), conversational trajectory, or the agent's own mental state when attention-schema cognition-log entries report self-directed attention. Each prediction entry includes the target subject, predicted state, and an estimated validity horizon.

Predictions are written to the predict memo. Predict does not detect whether its predictions were correct and does not write memory — divergence detection and explicit preservation requests belong to surprise.

### Surprise

Detects unexpected cognition-log entries by comparing new cognition log entries against the predict memo (if present) or against recent cognition-log history alone (if predict is absent). Uses an LLM to assess the degree of divergence or novelty with allocation guidance in context, sends a `MemoryRequest` when a significant event should be preserved, and records the surprise assessment in its memo.

Surprise does not generate predictions. When the predict memo is absent, surprise acts as a novelty detector over the cognition log — it judges unexpectedness from recent history rather than from explicit predictions.

### SpeakGate

Decides whether the cognition log is ready for speech. SpeakGate is triggered only by cognition-log
updates. When it wakes, it can read blackboard memos, the cognition log, scheduler-owned module
status, and utterance progress, so it can distinguish memo-only facts from attended facts and decide
whether a new cognition log should interrupt an in-progress utterance. If speech is ready or an
in-progress stream should be replaced, it sends a typed `SpeakRequest` with a mandatory target to
Speak. The target is the addressee for the utterance; self-directed speech uses `self`. If speech should
wait, it writes the wait decision and any missing-evidence notes to its memo. It does not emit
utterances, write cognition-log entries, change allocation, or write memory. It may call evidence tools for
memory, self-model, and sensory-detail lookup during its decision turn.

### Speak

Emits user-visible utterances. The module is named `speak` rather than `talk` because its role is the action of producing an utterance, not owning the whole conversation.

Speak reads the cognition log and typed `SpeakRequest`, records targeted utterance progress while
streaming, writes the completed targeted utterance to its memo, and emits through `UtteranceWriter` so the
application or eval harness can collect the utterance as an artifact. It does not read blackboard
memos, allocation guidance, or module status; readiness and interruption decisions are delegated to
SpeakGate.

Speak is not a planner or router. It does not publish query or self-model work, and it does not make resource-allocation decisions. It starts and interrupts streams only when SpeakGate sends a typed `SpeakRequest`. Completed utterance memos wake the controller.

## Invariants

These invariants are upheld by boot-time capability wiring and owner-stamped handles:

- The attention controller, attention schema, and self-model modules are separate modules.
- The sensory module is the canonical app-facing path for external observations in full-agent runs.
- Query and self-model messages are internal; they are only driven directly by module-level eval harnesses or internal modules that hold those mailbox capabilities.
- The speak-gate module may inspect memos to judge readiness, but speech generation itself receives only the cognition log plus a typed `SpeakRequest`.
- The speak module is the canonical app-facing path for user-visible utterances in full-agent runs.
- Cognition-gate is the only path from the non-cognitive blackboard to the cognition log; attention-schema writes only attention-experience entries derived from its attention-state integration.
- Query modules and self-model questions are separated by typed channels.
- Durable answers are memo-authoritative, not mailbox responses.
- A module cannot impersonate another module.
- Cognition-log appenders cannot wake the controller directly; the controller wakes on memo updates.
- The attention controller is not woken by cognition log updates.
- Allocation activation ratios respect boot-time `cap_range.min/max`; modules with `cap_range.min = 0` are detachable by allocation.
- Query-vector and query-agentic are independently detachable for ablation.
- Predict and surprise are independently detachable for ablation.
- Ablating the attention schema module should degrade attention-state modeling while task performance largely survives.
- Ablating the self-model module should degrade self-report specifically while task performance largely survives.
- Ablating sensory or speak should degrade end-to-end artifact evaluation while leaving lower-level query, attention-schema, and self-model module evaluations possible.
- Ablating predict degrades surprise to cognition-log-history novelty detection.
- Ablating surprise disables prediction-failure learning while leaving forward predictions available to other modules.
- Ablating both predict and surprise removes the prediction loop entirely without affecting other modules.
