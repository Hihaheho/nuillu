# Attention Schema

An event loop runs multiple brain modules concurrently. Modules do not depend on each other directly; they cooperate through:

- a non-cognitive blackboard,
- a cognition log,
- typed transient channels,
- app-facing sensory/action ports,
- per-module memo logs.

A module's role is exactly the set of capabilities it is granted at construction. A module cannot perform an operation it was not granted, and cannot claim to act as another module: identity-bearing operations are owner-stamped by their capability handles.

The module set is open. The modules below are core roles, and additional domain modules can be added without changing the runtime.

## Agent States

### Cognition Log

The admitted cognitive surface. Cognition-gate promotes selected non-cognitive memo-log/blackboard state into this surface. Attention-schema may also append first-person attention experience entries when the current attention state itself should become cognitively available.

Log entry: `(time, event)`.

### Blackboard

The non-cognitive blackboard holds:

- **per-module memo logs** — each module owner writes only its own bounded indexed log,
- **memory metadata** — rank, decay, access counts, and remember tokens,
- **identity memory snapshot** — identity-ranked memories loaded once at agent startup,
- **resource allocation** — activation ratio, controller guidance, and model tier per module.

Module results that should be durable live in memo logs or other blackboard state. Each memo item has a per-owner monotonic index; reader handles track their own last-seen index so modules can distinguish unread output from recent context. Memo content is free-form prose on the shared workspace surface, not JSON, schema-shaped data, or a structured data exchange channel. Channel messages are transient activation signals and are not persisted.

External input and output sit at the boundary of cognition:

- **sensory input** is the only external/app-facing input; it enters through a typed `SensoryInput` channel and is normalized by the sensory module into memo-log entries,
- **speech output** leaves through a speak/action capability and is mirrored into the speak module's memo log.

`AttentionControlRequest` is the internal typed attention-bid message. It is consumed only by allocation-controller, which admits, defers, or rejects the bid by writing allocation guidance and a controller memo. Module-level eval harnesses seed allocation guidance directly to isolate query modules or the self-model module. Full-agent/app-facing cases must enter through `SensoryInput`.

Sensory input is split into `OneShot { modality, direction, content, observed_at }` and `AmbientSnapshot { entries, observed_at }`. One-shot input is a discrete stimulus such as a heard utterance, visual event, or custom modality. Ambient snapshots are the complete current enabled background field keyed by row id; omitted rows mean they are no longer active ambient context, not proof that an external condition disappeared.

`observed_at` is the host-observed datetime of the stimulus. When the sensory module writes a memo-log entry, it compares the current clock time to `observed_at` and keeps a detailed human-readable relative age. The memo-log entry should preserve detail, for example `Ryo said "..." 1 minute 20 seconds ago`, not a time-division bucket and not a raw turn id.

Time-division rounding belongs at the cognition boundary. When cognition-gate decides that sensory memo-log content should enter the cognition log, it maps the observation age through `configs/time-division.eure` and writes the rounded bucket/tag into the cognition log event.

The cognition log remains the admitted cognitive surface. Sensory input is not appended to the cognition log directly; cognition-gate notices sensory memo-log entries through the blackboard and decides what should become relevant to cognitive processing. Attention-schema writes only concise first-person attention experience entries to the cognition log; it does not create a separate attention-state memo.

The design uses [Graziano's attention-schema framing](https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2017.00060/full) as an engineering analogy: object/world models, self-models, and the attention schema are distinct internal models. The attention schema models the act and dynamics of attention; the self-model integrates that attentional relation with stable and current facts about the agent.

### Memories

A memory is `(rank, content, decay, query_history, remember_tokens)`.

- **rank**: short-term, mid-term, long-term, permanent, identity
- **query history**: recent access log
- **remember tokens**: accumulated by memory compaction when repeated memories merge
- **decay**: remaining time to stay in the current rank

Rank can rise through access or remember-token accumulation and fall when decay expires.
Access-based rank elevation is owned by `MemoryStore`, not by any module: reads that record
access advance an in-window counter, and crossing a per-rank threshold promotes the entry and
resets the decay timer. Identity-ranked memories are startup identity facts: the agent loads all of
them from the primary memory store before module activation and injects that stable snapshot into
LLM system prompts.

### Policies

A policy is `(rank, trigger, behavior, expected_reward, confidence, value, reward_tokens, decay, usage_history)`.

- **rank**: tentative, provisional, established, habit, core
- **trigger**: situation description plus its embedding; the only field used for `query-policy` vector search
- **behavior**: the action/pattern to apply when the trigger matches
- **expected_reward**: scalar in `[-1, 1]`; TD-updated prediction of reward when the policy fires
- **confidence**: scalar in `[0, 1]`; rises when prediction accuracy is high (small `|td_error|`)
- **value**: scalar in `[-1, 1]`; cumulative commitment, updated by `reward`
- **reward tokens**: count of TD-settled reinforcement events that credit the policy
- **decay**: remaining time before rank reduction if unrewarded
- **usage history**: recent access log; diagnostic only — does NOT elevate rank

Rank rises when `value` crosses a tier threshold with sufficient reward tokens, and falls when
decay expires without re-reward. Core-ranked policies are loaded at startup like identity memories
and injected into LLM system prompts. Policies strengthen through a TD-0 loop driven by
`reward`, not by access; memories strengthen by access, never by reward. The two stores never
share rank enums or strengthening rules.

### Reward signals

`reward` consumes a structured v1 `ObservedReward = { external, task, social, cost, risk, novelty }`
aggregated from surprise, speak-completion, sensory, and controller memos. The reward module
collapses the six channels into a scalar `observed_scalar` and computes `td_error =
observed_scalar − expected_reward` against the latest `value-estimator` memo. The full channel
breakdown is preserved in the reward memo for audit; only the aggregated `value`, `expected_reward`,
and `confidence` are persisted on the policy record in v1. Per-policy per-channel value tracking is
a v2 extension.

## Activations

A module can be activated by any inbox capability it holds:

- typed fanout topics, such as external `SensoryInputInbox` or controller-only `AttentionControlRequestInbox`,
- `MemoUpdatedInbox`, published by memo writes and filtered so a holder does not wake on its own writes,
- `AllocationUpdatedInbox`, published when effective allocation or guidance changes,
- `CognitionLogUpdatedInbox`, published by cognition-log writes and by the event loop when it records an agentic-deadlock marker, and filtered so a holder does not wake on its own cognition-log writes.

There is no periodic wake mechanism. Allocation guidance is the controller's durable control plane: modules read the current allocation snapshot and decide whether a wake should produce work, defer silently, or only update local state.

Guidance-based allocation flow:

```text
Sensory memo -> AllocationController -> allocation proposal
Sensory memo -> CognitionGate -> CognitionLogUpdated -> Speak
Runtime Speak batch -> SpeakGate ActivationGate -> Speak activation
SpeakGate -> AttentionControlRequest evidence bids -> AllocationController
Surprise -> AttentionControlRequest::Memory -> AllocationController
SpeakGate memo -> AllocationController -> allocation proposal
CognitionLogUpdated -> QueryMemory/QueryPolicy/Memory/Policy/Reward/Predict/Surprise/AttentionSchema/ValueEstimator
QueryPolicy memo -> ValueEstimator -> expected_reward memo
ValueEstimator memo + Surprise memo + Speak memo + Sensory memo -> Reward -> PolicyValueUpdater::reinforce
Reward -> AttentionControlRequest::Policy -> AllocationController
QueryPolicy memo -> AllocationController / CognitionGate (memo-authoritative)
```

Cognition-gate does not write a memo. Cognition-log entries wake cognition-log consumers such as attention-schema, Speak, Predict, and Surprise, but a module's own cognition-log write does not wake that same module, and cognition-log updates do not directly wake the controller. Before Speak activates, the runtime sends the pending Speak batch to active `ActivationGate<Speak>` holders such as SpeakGate. SpeakGate returns allow or suppress; when it suppresses, its decision and missing-evidence notes are durable only in its memo, and that memo update wakes the controller through the ordinary memo path. Speak writes a completion memo after a finished utterance.

## Capabilities

| Module | Read blackboard | Read cognition log | Read allocation | Memo | Clock | LLM | Special capabilities |
|---|---|---|---|---|---|---|---|
| sensory | — | — | ✓ | ✓ | ✓ | ✓ | `SensoryInputInbox` |
| cognition-gate | ✓ | — | ✓ | — | — | ✓ | `MemoUpdatedInbox`, `AllocationUpdatedInbox`, `CognitionWriter`, `TimeDivision` |
| allocation-controller | ✓ | ✓ | ✓ | ✓ | — | ✓ | `MemoUpdatedInbox`, `AttentionControlRequestInbox`, `AllocationWriter` |
| attention-schema | ✓ | ✓ | ✓ | — | — | ✓ | `MemoUpdatedInbox`, `AllocationUpdatedInbox`, `CognitionLogUpdatedInbox`, `CognitionWriter` |
| self-model | ✓ | ✓ | ✓ | ✓ | — | ✓ | `AllocationUpdatedInbox` |
| query-memory | ✓ | — | ✓ | ✓ | — | ✓ | `AllocationUpdatedInbox`, `CognitionLogUpdatedInbox`, `VectorMemorySearcher` |
| query-policy | ✓ | — | ✓ | ✓ | — | ✓ | `AllocationUpdatedInbox`, `CognitionLogUpdatedInbox`, `PolicySearcher` |
| memory | ✓ | — | ✓ | — | — | ✓ | `CognitionLogUpdatedInbox`, `AllocationUpdatedInbox`, `MemoryWriter` |
| memory-compaction | ✓ | — | ✓ | — | — | ✓ | `AllocationUpdatedInbox`, `MemoryCompactor` |
| policy | ✓ | — | ✓ | — | — | ✓ | `CognitionLogUpdatedInbox`, `AllocationUpdatedInbox`, `PolicyWriter` |
| value-estimator | ✓ | ✓ | ✓ | ✓ | — | ✓ | `MemoUpdatedInbox`, `CognitionLogUpdatedInbox`, `AllocationUpdatedInbox` |
| reward | ✓ | ✓ | ✓ | ✓ | — | ✓ | `CognitionLogUpdatedInbox`, `MemoUpdatedInbox`, `AllocationUpdatedInbox`, `PolicyValueUpdater`, `AttentionControlRequestMailbox` |
| predict | ✓ | ✓ | ✓ | ✓ | — | ✓ | `CognitionLogUpdatedInbox` |
| surprise | ✓ | ✓ | ✓ | ✓ | — | ✓ | `CognitionLogUpdatedInbox`, `AttentionControlRequestMailbox` |
| speak-gate | ✓ | ✓ | — | ✓ | — | ✓ | `ActivationGate<SpeakModule>`, `ModuleStatusReader`, `AttentionControlRequestMailbox` |
| speak | — | ✓ | — | ✓ | ✓ | ✓ | `CognitionLogUpdatedInbox`, `SceneReader`, `UtteranceWriter` |

Notable absences:

- The attention schema module has no memo, self-model inbox, allocation-write path, or memory-write path. Its durable output is limited to first-person attention experience entries in the cognition log.
- The self-model module handles controller self-model guidance, but has no cognition-log-write, allocation-write, or memory-write path.
- Query modules receive controller guidance, not self-model requests, and do not perform self-model integration.
- The sensory module is the app-facing observation boundary; it cannot write cognition-log entries, publish work requests, or emit utterances.
- The speak-gate module can inspect memo-log history in its session and call evidence lookup tools to decide speech readiness, but cannot emit utterances, write cognition-log entries, change allocation, or write memory.
- The speak module is the app-facing output boundary; it cannot read blackboard memo logs or allocation guidance, and cannot write cognition-log entries, allocation, memory, or attention-control requests.
- Only modules granted `CognitionWriter` can append cognition-log entries; current boot wiring grants it to cognition-gate and attention-schema.
- Cognition-log appends cannot wake the controller directly; the controller wakes on memo updates.
- The attention controller wakes on memo updates; it reads the cognition log but is not woken by `CognitionLogUpdated`.
- Only the attention controller can write resource allocation.
- The memory module cannot increment remember tokens; that belongs to memory compaction.
- The memory module cannot elevate memory rank; that belongs to `MemoryStore` on read-path access.
- The policy module cannot mutate existing policies; value, expected-reward, confidence, rank, and reward-token changes belong to reward.
- The reward module cannot create new policies; insertion belongs to the policy module.
- The reward module cannot predict `expected_reward`; that is value-estimator's role.
- The value-estimator module cannot mutate policy state; its predictions live only in its memo and never persist to the policy record in v1.
- `PolicySearcher` is trigger-only; `behavior`, `value`, `expected_reward`, and `confidence` are returned with hits but never participate in similarity scoring.
- Reward cannot write allocation; it may only bid through `AttentionControlRequest::Policy`.
- Access does not strengthen policies; reward does not strengthen memories.
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

The LLM stage receives one-shot events and ambient diffs plus deterministic salience features and decides whether to ignore them, fold them into a background summary, or write a concise normalized observation to the sensory memo. The current full ambient field is passed as ephemeral assistant context on each sensory LLM turn; only one-shot ledger entries and ambient add/update/remove diffs are persisted in sensory's private LLM session.

The sensory memo is the only output. It should contain the filtered, normalized observations that other modules may inspect through the blackboard path, plus enough inspection detail to explain salience and the LLM decision. It should not mirror every raw input event. Sensory does not answer the user, append to the cognition log, publish query/self-model requests, or mutate shared state beyond its memo.

### Cognition Gate

Reads the non-cognitive blackboard snapshot and current allocation guidance, then appends to the cognition log when something is novel, unresolved, strongly changed, or requested by controller guidance. It is the only bridge from the non-cognitive blackboard to the cognitive surface, and it has no memo capability.

When sensory memo content is promoted, cognition-gate rounds the detailed memo age through `configs/time-division.eure` before appending the cognition-log entry. Detailed timing stays in the sensory memo; rounded timing belongs to the cognition log.

### Allocation Controller

Wakes on memo updates from other modules and controller-only attention-control requests. It reads unread memo-log entries into its persistent session, plus current attention-control bids, memory metadata, cognition log, and current allocation, then writes the next resource allocation for every registered module: `activation_ratio`, natural-language `guidance`, and `tier`.

The controller uses guidance rather than request/response correlation. For example, if speech should wait for query or cognition-gate work, the controller raises the relevant modules and lets SpeakGate keep speech silent until attended evidence is sufficient. Attention-control bids are admitted, deferred, or rejected in the controller memo; deferred bids are not stored in a separate pending queue. Speak itself does not parse allocation guidance.

### Attention Schema

Reads unread memo-log entries into a persistent LLM session, plus current allocation and cognition log, to decide whether the current attention state should become admitted cognitive evidence. When a new, claimable, cognitively useful attention experience exists, it appends a concise first-person cognition-log entry through a plaintext tool call. When nothing new should be admitted, it calls no tool and writes nothing.

The attention schema is not a self-model and does not answer self-model guidance. It assumes a non-physical experiencer that can direct attention to any target and freely control that attention, but its appended text should avoid mechanical internals and decision noise. It may diverge from the controller's internal allocation state; that gap is part of the architecture.

### Self Model

Maintains a current self-description by integrating attention-schema cognition-log entries, relevant module memo logs, self-related knowledge that query modules surface from memory, and self-model guidance from allocation-controller. It writes self-model / self-report output to its memo log.

Stable self-knowledge belongs in memory and can be retrieved by query modules, but a live self-model is not just memory retrieval. The self-model module turns long-lived self facts, current attention, active task context, uncertainty, and recent module outputs into a current first-person state. In v1, object/world models remain distributed through sensory, query, predict, and surprise memo logs; a dedicated `world-model` can be added later if those fragments need a single owner.

### Query Memory

Handles memory retrieval (vector-memory/RAG backed in v1). Allocation updates wake it to act on controller guidance; cognition-log updates can also wake it to retrieve memory relevant to the current cognitive surface. Its memo-log entries contain only retrieved memory content, copied from query results; it does not synthesize answers, describe itself, or query the policy store. Reads count as memory access; rank elevation is applied by `MemoryStore` and is not module-visible.

### Query Policy

Retrieves applicable policies from the policy store. Allocation updates wake it to act on controller guidance; cognition-log updates can also wake it to surface policies relevant to a newly admitted situation. Vector search is over the `trigger` embedding only — `behavior`, `value`, `expected_reward`, and `confidence` are returned with hits but never used as search keys. Memo-log entries contain only retrieved policy records (`trigger`, `behavior`, `expected_reward`, `confidence`, `value`) copied from search results; it does not synthesize advice, modify policy state, or describe itself. The retrieval memos are the substrate for both `value-estimator` (which predicts `expected_reward` for each hit) and `reward` (which uses the retrieval window for credit assignment). Retrieval counts as access for diagnostic `usage_history` only — access does not elevate policy rank.

### Memory

Preserves useful information by inserting memory entries after cognition-log updates or preservation guidance from allocation-controller. It inspects the current cognition log plus indexed unread/recent memo logs as candidate evidence. Attention-schema entries are ordinary cognition-log evidence when the current attention state matters. Preservation guidance is a candidate, not a write. The memory module may reject, normalize, merge, or deduplicate candidates, and only persists records through its own `insert_memory` tool decision. It does not elevate memory rank — access-based rank elevation belongs to `MemoryStore`.

### Memory Compaction

Fetches related memory contents and merges redundant memories, accumulating remember tokens. Allocation updates wake it to consider compaction guidance.

### Policy

Preserves successful or distinctive behavior patterns as tentative `(trigger, behavior)` policy records. Cognition-log updates and allocation guidance are wake paths; candidates include speak completion memos, surprise-resolved sequences, and explicit controller policy-formation guidance. `trigger` is the situation description that `query-policy` will embed and match against future situations; `behavior` is the action/pattern to apply when the trigger matches. The module may reject, normalize, deduplicate, or merge candidates against existing policies and may rewrite an existing trigger to broaden or narrow its scope. New entries start at `tentative` rank with `value = 0`, `expected_reward = 0`, `confidence = 0`, and `reward_tokens = 0`. It does not mutate existing policy entries — value, expected reward, confidence, rank, and reward-token changes belong to reward.

### Value Estimator

Predicts the `expected_reward` for each policy currently surfaced by `query-policy`. It is the critic half of the actor/critic split — `query-policy` retrieves candidates, value-estimator scores them. It wakes primarily on `query-policy` memo updates; cognition-log updates and allocation guidance can also wake it when the cognitive surface changes meaningfully between retrievals. It reads each hit's stored `expected_reward` and `confidence` plus the recent cognition log and writes a per-window prediction memo of `(policy_index, predicted_expected_reward, rationale)` entries. The memo is the prediction baseline that reward compares against `ObservedReward` to compute `td_error`. It does not modify policy state, write memory, write cognition-log entries, write allocation, or call `PolicyValueUpdater`; context-dependent valuation lives only in this memo, never on the persisted record in v1.

### Reward

Closes the TD-0 loop. It aggregates the v1 6-channel `ObservedReward = { external, task, social, cost, risk, novelty }` from surprise, speak-completion, sensory, and controller memos, collapses the channels to a scalar `observed_scalar`, and pairs that with the most recent value-estimator memo to compute `td_error = observed_scalar − expected_reward`. It applies the update through `PolicyValueUpdater::reinforce(index, value_delta, reward_tokens_delta, expected_reward_delta, confidence_delta)` with v1 deltas `α·td_error`, `β·clamp(td_error, ±1)`, and `γ_c·max(0, 1 − |td_error|/scale)` (coefficients in `configs/policy-reinforcement.eure`). It writes a reward-assessment memo recording the full channel breakdown, the compared `expected_reward`, the resulting `td_error`, and the applied deltas so the allocation-controller can observe learning pressure. It may publish `AttentionControlRequest::Policy` to ask the controller to raise policy-module activation when a novel pattern deserves formation. Rank elevation/demotion is a derived consequence of value crossing tier thresholds with sufficient reward tokens; reward cannot invent rank changes independent of those thresholds. It does not predict `expected_reward` (that is value-estimator's role), create new policy entries, write memory, write cognition-log entries, or write allocation.

### Predict

Maintains forward predictions about current cognition-log targets. On each cognition log update, reads the cognition log, blackboard context, and allocation guidance, then uses an LLM to generate predictions about the likely near-future states of the admitted cognitive targets — external subjects (people, objects, events), conversational trajectory, or the agent's own mental state when attention-schema cognition-log entries report self-directed attention. Each prediction entry includes the target subject, predicted state, and an estimated validity horizon.

Predictions are written to the predict memo. Predict does not detect whether its predictions were correct and does not write memory — divergence detection and explicit preservation requests belong to surprise.

### Surprise

Detects unexpected cognition-log entries by comparing new cognition log entries against the predict memo (if present) or against recent cognition-log history alone (if predict is absent). Uses an LLM to assess the degree of divergence or novelty with allocation guidance in context, sends an `AttentionControlRequest::Memory` bid when a significant event should be preserved, and records the surprise assessment in its memo.

Surprise does not generate predictions. When predict memo-log evidence is absent, surprise acts as a novelty detector over the cognition log: it judges unexpectedness from recent history rather than from explicit predictions.

### SpeakGate

Decides whether a pending Speak activation should run. SpeakGate is triggered by
`ActivationGate<SpeakModule>` events after Speak has formed a cognition-log update batch and before
Speak activates. When it wakes, it reads unread memo-log entries into its persistent session, plus
the cognition log, scheduler-owned module status, and utterance progress, so it can distinguish
memo-only facts from attended facts. If speech is ready, it allows the activation. If speech should
wait, it suppresses the activation and writes the wait decision and any missing-evidence notes to
its memo log. It does not emit utterances, write cognition-log entries, change allocation, or write
memory. It may call evidence tools that publish attention-control bids for memory and self-model lookup during its
decision turn.

### Speak

Emits user-visible utterances. The module is named `speak` rather than `talk` because its role is the action of producing an utterance, not owning the whole conversation.

Speak wakes on cognition-log updates, then activates only after runtime activation gates allow its
batch. It reads the cognition log, chooses a target from the current scene, records targeted
utterance progress while streaming, writes the completed targeted utterance to its memo log, and
emits through `UtteranceWriter` so the application or eval harness can collect the utterance as an
artifact. It does not read blackboard memo logs, allocation guidance, or module status; readiness
decisions are delegated to SpeakGate.

Speak is not a planner or router. It does not publish query or self-model work, and it does not make resource-allocation decisions. Completed utterance memo-log entries wake the controller.

## Invariants

These invariants are upheld by boot-time capability wiring and owner-stamped handles:

- The attention controller, attention schema, and self-model modules are separate modules.
- The sensory module is the canonical app-facing path for external observations in full-agent runs.
- Attention-control messages are internal and controller-only; module-level eval harnesses isolate query/self-model modules by seeding allocation guidance rather than by publishing target-specific request payloads.
- The speak-gate module may inspect memo-log history in its session to judge readiness, but speech generation itself receives only the cognition log plus the target selected by Speak.
- The speak module is the canonical app-facing path for user-visible utterances in full-agent runs.
- Cognition-gate is the only path from the non-cognitive blackboard to the cognition log; attention-schema writes only attention-experience entries derived from its attention-state integration.
- Query modules and self-model work are separated by controller guidance and module-specific capabilities.
- Durable answers are memo-log-authoritative, not mailbox responses.
- A module cannot impersonate another module.
- Cognition-log appenders cannot wake the controller directly; the controller wakes on memo updates.
- The attention controller is not woken by cognition log updates.
- Allocation activation ratios respect boot-time `cap_range.min/max`; modules with `cap_range.min = 0` are detachable by allocation.
- Query-memory and query-policy are independently detachable for ablation.
- Predict and surprise are independently detachable for ablation.
- Memory and learning are distinct reinforcement substrates: memory rank elevation is store-internal and access-driven; policy rank elevation is reward-driven and runs only through `PolicyValueUpdater::reinforce`.
- Policy creation (`policy` module), value prediction (`value-estimator` module), and policy update (`reward` module) are separate roles: `policy` cannot mutate existing entries, `value-estimator` cannot mutate policy state, and `reward` cannot insert new entries or predict expected reward.
- Policy vector search is trigger-only: `PolicySearcher` scores similarity against the `trigger` embedding; `behavior`, `value`, `expected_reward`, and `confidence` are never used as search keys.
- TD learning is TD-0 in v1: `td_error = observed_scalar − expected_reward`; no discount factor, no `next_state` bootstrap, no eligibility traces.
- `ObservedReward` is fixed at six channels in v1 (external, task, social, cost, risk, novelty); per-policy per-channel storage is a v2 extension.
- Reward cannot write allocation; it may only bid through `AttentionControlRequest::Policy`.
- `PolicySearcher` does not return demoted or expired policies.
- Policy, value-estimator, reward, and query-policy are independently detachable for ablation; ablating value-estimator collapses `expected_reward` to the stored value with degraded credit assignment but functional learning.
- Ablating the attention schema module should degrade attention-state modeling while task performance largely survives.
- Ablating the self-model module should degrade self-report specifically while task performance largely survives.
- Ablating sensory or speak should degrade end-to-end artifact evaluation while leaving lower-level query, attention-schema, and self-model module evaluations possible.
- Ablating predict degrades surprise to cognition-log-history novelty detection.
- Ablating surprise degrades reward to speak-completion only; learning continues but with a weaker credit signal.
- Ablating surprise disables prediction-failure learning while leaving forward predictions available to other modules.
- Ablating both predict and surprise removes the prediction loop entirely without affecting other modules.
