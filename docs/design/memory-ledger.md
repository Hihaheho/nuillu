# Memory Ledger

Date: 2026-05-13
Scope: v2 durable memory organization, recall, linking, compaction, and dream-like balancing.

This document describes desired architecture, not implementation progress. It extends
`attention-schema.md` and `implementation-design.md` without changing current module
wiring, migrations, prompts, or public boot APIs.

---

## 0. Decisions

1. **Memory is not a fact table** - Nuillu does not model durable memory around
   structured fact claims, current-fact projections, or source-authority records.
   It stores remembered evidence and lets recall, links, rank, time, compaction,
   and downstream interpretation shape behavior.
2. **Canonical memory is short natural text** - A memory entry is a concise,
   normalized natural-language memory, usually one to three sentences.
3. **Memory text is effectively immutable** - Changed situations, corrections,
   and edits create new memory entries and later links. They do not overwrite the
   older remembered text.
4. **No memory versions in v2** - The ledger does not require a separate
   `memory_version` layer. If a memory must be revised, the revision is another
   memory connected by links.
5. **No stored current-truth hint** - Retrieval returns remembered material with
   timestamps, rank, kind, concepts, tags, and links. It does not attach a
   "current" decision.
6. **No source authority field** - If source context matters, such as who said
   something or where it was seen, that context belongs in the memory sentence
   itself. The ledger does not treat `user`, `agent`, or `import` as truth
   authorities.
7. **Concepts and tags are sidecars** - Concepts support associative recall.
   Operational tags support handling, compaction, search, and UI. Neither is
   embedded into canonical text.
8. **Direct links are compaction-owned** - Ingest extracts kind, concepts, and
   operational tags. Memory-to-memory links are formed later by compaction and
   rebalancing.
9. **`MemoryRank` is not truth** - Existing memory ranks describe durability,
   salience, access reinforcement, and boot identity weight. Rank does not mean a
   memory is factual, current, or authoritative.
10. **Hard delete is allowed** - User deletion can remove a memory and all attached
    sidecars, search rows, embeddings, and links. Orphaned concepts become garbage
    collection candidates.
11. **Stored affect is write-time context** - `affect_arousal`, `valence`, and
    `emotion` are copied from the current interoceptive state when a memory is
    written. They are not truth labels, not current emotion, and not later
    reinterpretation of the memory.

---

## 1. Fit With The Current Runtime

The module boundary stays capability-based. Modules do not receive raw database
handles, arbitrary SQL access, or a new request/response protocol. Channel messages
remain transient wake signals, and durable module output remains memo-authoritative.

| Role | Memory-ledger responsibility |
|---|---|
| `memory` | Preserves useful cognitive evidence as short normalized memory text. Ingest always creates short-term memory and extracts only kind, concepts, and operational tags. |
| `query-memory` | Retrieves flat memory hits and may explicitly fetch linked memories through tools when useful. |
| `memory-compaction` | Performs NREM-like consolidation: creates summary memories, operational tags, and memory-to-memory links while preserving source memories. |
| `memory-recombination` | Performs REM-like associative simulation. Its outputs are dream or hypothesis material, not verified facts. |
| `interoception` / `homeostatic-controller` | `interoception` estimates internal state; `homeostatic-controller` regulates sleep-like memory balancing by raising compaction or recombination allocation from that state. |
| `allocation-controller` | Allocates memory work through ordinary guidance. It does not gain direct memory graph mutation or truth-projection power. |
| `self-model` | Integrates retrieved identity and self-related memories into current self-description. Identity memories are high-rank memories, not current-fact rows. |

The store owns ledger consistency, search indexes, embeddings, access accounting,
hard deletion, and link mutation. The blackboard remains a mirror of operational
metadata such as rank, decay, access counts, remember tokens, and boot identity
snapshots.

---

## 2. Canonical Model

The ledger stores memory entries plus sidecars. The memory text is the durable
evidence. Sidecars help recall and consolidation, but they do not replace the text.

```text
memory {
  id
  content
  kind
  rank
  occurred_at
  stored_at
  affect_arousal
  valence
  emotion
  deleted_at
}

concept {
  name
  loose_type?
}

concept_alias {
  concept_name
  alias
}

memory_concept {
  memory_id
  concept_name
  mention_text?
  confidence
}

tag {
  name
  namespace
}

memory_tag {
  memory_id
  tag_name
  confidence
}

memory_link {
  from_memory_id
  to_memory_id
  relation
  freeform_relation?
  strength
  confidence
  updated_at
}

memory_search {
  memory_id
  search_text
  concept_text
  tag_text
}

memory_embedding {
  memory_id
  embedding_profile
  embedding
}
```

`occurred_at` is when the remembered event or state happened when known.
`stored_at` is when Nuillu stored the memory. Both are exact timestamps. Prompt and
display surfaces may round time through existing time-division behavior, but the
ledger stores exact values.

`affect_arousal`, `valence`, and `emotion` are the interoceptive affect snapshot
at `stored_at`. `affect_arousal` is clamped to `0.0..=1.0`, `valence` to
`-1.0..=1.0`, and `emotion` remains untyped text. These fields describe the
agent's storage-time context. They do not assert that the remembered event was
objectively positive or negative, and they are not updated when the memory is
later recalled or reinterpreted.

The initial loose memory kinds are:

```text
episode
statement
reflection
hypothesis
dream
procedure
plan
```

Kind is a weak retrieval and display hint. It is not a hard store boundary.
For example, a `statement` can later support an `episode` summary, and a `dream`
can remain useful as an association without becoming verified evidence.

Concepts are loose symbolic nodes: people, places, objects, projects, topics, or
recurring ideas. They may carry optional loose types such as `person`, `place`,
`project`, `topic`, `object`, `self`, or `unknown`. These are hints, not a strict
ontology. There are no reserved concept IDs, including `self`; special self meaning
belongs in module prompts and self-model integration, not in the ledger schema.

Concept deduplication is conservative. Exact normalized labels and explicit aliases
can unify concepts. Ambiguous same-name concepts stay separate until later evidence
or manual correction justifies merging.

Tags are operational classification sidecars. They are not the main association
graph. Initial uses include tags such as `dream`, `hypothesis`, `follow-up`, and
`preference`, where a tag changes handling, compaction, search, or UI treatment.

---

## 3. Ingest

Ingest keeps the first write simple.

The memory module writes a short normalized memory sentence from cognition-log and
memo-log evidence. The sentence should be readable by a human and a small LLM. If
source context matters, it is part of the sentence:

```text
Ryo said he no longer lives in Tokyo and recently moved to Kyoto.
```

The sidecar extraction at ingest is limited to:

```text
kind
concepts
operational tags
```

Ingest does not choose memory rank, decay, or occurrence time. The
`insert_memory` tool has no rank, decay, or `occurred_at` arguments, and every
ordinary memory-module insert starts as `MemoryRank::ShortTerm` with runtime-stamped
decay and occurrence metadata. Higher ranks come only from store-owned
reinforcement, compaction/ledger maintenance paths, or boot/manual identity seeding.

Concepts and tags at ingest are name newtypes serialized as simple string arrays,
for example `concepts: ["Ryo", "super red apple"]` and
`tags: ["dialogue_flow"]`. Concept and tag names are the ledger keys. Ingest does
not create concept ids, tag ids, aliases, or canonicalization records; later
compaction or ledger-maintenance work may merge or relate names when there is
enough context.

Ingest does not create direct memory-to-memory links. It also does not infer
updates, corrections, contradictions, or truth. Those relationships require broader
context and belong to compaction or rebalancing.

This keeps small LLM tasks understandable:

- normalize a candidate into concise memory text,
- classify its loose kind,
- extract mentioned concepts,
- add lightweight operational tags only when useful.

The memory module may still reject, deduplicate, or normalize preservation
candidates. It does not receive authority to overwrite older memory text or decide
that one memory is now the current truth.

---

## 4. Links

Memory-to-memory links are sidecars. They are separate from memory text and are
mutable so later compaction can reshape associations without rewriting remembered
evidence.

Core relations:

```text
related
supports
contradicts
updates
corrects
derived_from
```

The relation vocabulary is `core + freeform`. Core relations are preferred because
they are searchable and testable. If compaction finds a useful relationship that
does not fit the core set, it may store a freeform relation label alongside a core
fallback such as `related`.

Links are basically directed:

- `A supports B` means A is remembered as supporting B.
- `A contradicts B` means A conflicts with B.
- `A updates B` means B may have been true at the time, but A remembers a later
  state change.
- `A corrects B` means B appears wrong or superseded by later remembered evidence.
- `A derived_from B` means A was produced from B, as with compaction summaries.
- `related` is treated as symmetric even if stored in one direction.

`strength` and `confidence` can change over time. Text does not. Inferred
`corrects` links are allowed, but should start with low confidence and become
stronger only through repeated support, later compaction, or explicit remembered
evidence.

For example:

```text
M1: Ryo used to live in Tokyo.
M2: Ryo said he recently moved to Kyoto.

M2 updates M1
```

The link does not hide M1. It makes M1 and M2 easier to co-recall as a changed
state. If M1 is recalled first, M2 can be fetched as related context; if M2 is
recalled first, M1 can still interfere as an older memory.

---

## 5. Retrieval

The memory search API returns flat hits. It does not return bundles, current facts,
or final answers.

Seed candidates may come from:

1. vector search,
2. full-text search when available,
3. concept matches,
4. tag matches.

`query-memory` may then explicitly fetch linked memories when useful. Link expansion
is a tool action, not a hidden behavior of the store's ordinary search call.

Returned memory evidence should include:

```text
content
kind
rank
occurred_at
stored_at
affect_arousal
valence
emotion
matched concepts
operational tags
direct link metadata when fetched
```

No currentness hint is attached. The retrieval layer does not say "this is the
current value" or "this old memory is outdated." It gives downstream modules enough
remembered material to behave naturally: sometimes emphasizing newer information,
sometimes surfacing uncertainty, and sometimes noticing an older conflicting memory.

`query-memory` may summarize evidence and conflict structure in its memo, but it
must not produce the user-facing answer or decide final truth. A valid evidence
summary is:

```text
Memory evidence about Ryo's residence is mixed. One older memory says he lived in
Tokyo. A later memory says he moved to Kyoto. The linked relation is an update,
not a deletion of the older memory.
```

That summary is still retrieval evidence. Speak and self-model integration remain
downstream of the normal cognition and memo surfaces.

Access reinforcement applies only to direct search hits. Memories fetched only by
explicit link expansion do not gain access reinforcement from that expansion alone.
This prevents a linked cluster from strengthening itself merely because one member
was recalled.

---

## 6. Compaction And Dreaming

Memory balancing is homeostatic and sleep-like.

NREM-like compaction is responsible for:

- creating summary memories from related source memories,
- adding `derived_from` links from summaries to sources,
- creating or updating `updates`, `corrects`, `contradicts`, `supports`, and
  `related` links,
- adding operational tags when they help later handling,
- preserving source memories unless hard deletion is requested.

Compaction output is `summary + links`. It does not collapse the system into a
single current fact. The source memories remain retrievable and can still interfere
with later recall.

Example compaction summary:

```text
Ryo's remembered residence changed from Tokyo to Kyoto across two memories.
```

This summary can link to both source memories using `derived_from`, and the later
source memory can link to the older one using `updates`.

REM-like recombination is different. `memory-recombination` produces associative
dream or hypothesis material. These entries may be useful for planning, surprise,
or later associations, but they are not verified evidence. They should use `dream`
or `hypothesis` kind and operational tags when stored.

Existing `interoception` and `homeostatic-controller` roles drive this balancing through
allocation. `interoception` owns the pressure and affect estimate; `homeostatic-controller`
only regulates compaction/recombination drive and action caps from that estimate. The
memory ledger does not add a new memory tick API or a background SQL job that bypasses
module capabilities.

---

## 7. Deletion And Garbage Collection

This document does not define a privacy model. It only defines deletion behavior for
the memory ledger.

Hard deleting a memory removes:

- the memory text,
- concept mentions attached to the memory,
- tag assignments attached to the memory,
- links where the memory is either endpoint,
- search rows,
- embeddings.

Concepts and tags are not automatically deleted just because one memory was
removed. If a concept or tag becomes orphaned, it becomes a garbage collection
candidate. Garbage collection must not infer that neighboring memories should be
deleted merely because a linked memory was deleted.

Compaction may archive or reduce operational weight in later implementation passes,
but archive behavior is separate from hard deletion. Hard deletion is the user
control path for removing remembered material and its sidecars.

---

## 8. Future Implementation Shape

The first implementation pass should keep current module-facing capabilities intact.
The exact Rust names can be chosen later, but the semantic boundaries are fixed:

- modules request preservation, retrieval, linked-memory lookup, or compaction,
- the store owns ledger consistency and indexing,
- compaction owns direct memory-to-memory link formation,
- blackboard mirrors metadata only,
- query output remains memo-authoritative,
- no module receives raw SQL or arbitrary table mutation power.

The search API should remain flat by default. A separate linked-memory lookup tool
can take memory ids and relation filters and return neighboring memories plus link
metadata. This keeps ordinary search predictable while allowing `query-memory` to
follow associations when its LLM session judges that useful.

The current libSQL adapter can continue as a simple content plus vector search
implementation until the ledger sidecars exist. This document does not require a
specific SQLite extension or a full SQL migration shape.

---

## 9. Acceptance Scenarios

Design-level scenarios for the implementation pass:

1. **Simple ingest** - Ingest one short memory and verify content, kind, concepts,
   operational tags, timestamps, search row, embedding row, and blackboard metadata
   are coherent.
2. **Changed situation** - Store a later changed-state memory and verify the older
   memory text is not overwritten.
3. **Compaction links** - Run compaction over older and newer related memories and
   verify it creates `updates`, `corrects`, or `contradicts` links when appropriate.
4. **Flat retrieval** - Ordinary search returns flat hits. Linked memories appear
   only when `query-memory` explicitly fetches them.
5. **Old memory remains searchable** - A memory linked by `updates` or `corrects`
   is still searchable and retrievable.
6. **Evidence summary boundary** - `query-memory` may summarize conflict evidence
   in its memo, but does not produce the user-facing answer or final truth decision.
7. **Compaction summary** - Compaction creates a summary memory and `derived_from`
   links while preserving source memories.
8. **Dream handling** - Dream or hypothesis memories are marked by kind/tag and are
   not treated as verified evidence.
9. **Access reinforcement** - Direct search hits record access; linked memories
   fetched only by expansion do not gain access reinforcement.
10. **Hard delete** - Deleting a memory removes its sidecars, links, search row, and
    embedding while leaving unrelated memories intact.

---

## 10. Invariants

| Invariant | Enforcement target |
|---|---|
| Memory is not a fact table | No claim/current projection is required or exposed. |
| Canonical text is effectively immutable | Revisions become new memories plus links. |
| Ingest stays simple | Ingest extracts kind, concepts, and operational tags only. |
| Direct links are compaction-owned | Ingest does not infer memory-to-memory corrections or contradictions. |
| Currentness is not stored | Retrieval returns evidence and metadata, not current truth. |
| Source is not authority | Any source context must live in natural memory text. |
| Rank is durability/salience | `MemoryRank` does not imply truth, type, or currentness. |
| Identity is high-rank memory | Identity-ranked memories are prompt seeds, not fact projections. |
| Query output remains memo-authoritative | `query-memory` writes retrieval evidence and summaries to its memo. |
| Capabilities remain the only module path | Modules hold memory capabilities, not raw database handles. |
| Blackboard remains non-canonical | Durable content, links, search rows, and embeddings live in the store. |
| Dream output is not verified evidence | `dream` and `hypothesis` entries require downstream interpretation. |
| Hard delete removes attached sidecars | Memory deletion removes concept mentions, tag assignments, links, search rows, and embeddings for that memory. |
