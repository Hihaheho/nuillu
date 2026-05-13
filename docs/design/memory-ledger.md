# Memory Ledger

Date: 2026-05-13
Scope: v2 durable memory storage, retrieval, extraction, and fact-update design.

This document describes desired architecture, not implementation progress. It extends
`attention-schema.md` and `implementation-design.md` without changing the current v1
runtime, module wiring, prompts, migrations, or public boot APIs.

---

## 0. Decisions

1. **One canonical ledger, typed operation** - Episodes, facts, reflections, plans,
   and procedures share one durable memory ledger. Retrieval, update, ranking, and
   response behavior use typed projections over that ledger.
2. **Text is canonical evidence** - Raw memory text is stored unchanged for the
   version that produced it. Meaning is not embedded into canonical text with XML,
   JSON snippets, or inline machine tags.
3. **Meaning lives in sidecars** - Entities, aliases, mentions, tags, claims, links,
   privacy flags, full-text index rows, and embeddings are stored as sidecar records
   keyed to a memory version.
4. **Claims are append-only** - Structured facts are inserted as `fact_claim` rows.
   A mutable `current_fact` projection chooses the current claim for ordinary
   response paths. Old claims are superseded, disputed, retracted, or archived; they
   are not hard-overwritten.
5. **`MemoryRank` is not memory type** - Existing memory ranks describe durability,
   salience, and access reinforcement. They do not mean episode/fact/procedure, and
   they do not imply truth. Logical kind and claim status are separate dimensions.
6. **Capabilities remain the boundary** - Modules continue to interact through
   memory capabilities such as `MemoryWriter`, `VectorMemorySearcher` or its future
   broader search replacement, and `MemoryCompactor`. Modules must not receive raw
   database handles or a new request/response mailbox protocol.
7. **Blackboard stays a mirror** - Durable memory content, extraction sidecars, and
   indexes live in the memory store adapter. The blackboard mirrors operational
   metadata needed by modules, such as rank, decay, access counts, remember tokens,
   and boot identity snapshots.
8. **Embeddings are sensitive data** - Embeddings are useful for recall, but they
   are not privacy neutral. They follow the same privacy, deletion, provider, and
   retention policy as the text they encode, unless a stricter policy is configured.

---

## 1. Fit With The Current Runtime

The cognitive surface does not change. Memory modules still observe cognition-log
updates, memo-log context, and attention-controller guidance. Channel messages stay
transient wake signals, and durable module output remains memo-authoritative.

The existing roles map onto this design as follows:

| Role | v2 memory-ledger responsibility |
|---|---|
| `memory` | Decides which cognitive evidence deserves durable preservation, then writes through the memory capability. The store performs canonical insert, extraction, indexing, and projection updates behind that capability. |
| `query-memory` | Retrieves memory evidence and writes copied hits to its memo. It does not synthesize answers, decide truth, or bypass cognition-gate admission. Existing `query-vector` crate/tool names may remain until a follow-up rename. |
| `memory-compaction` | Merges, summarizes, archives, or prunes redundant memory material through a compaction capability. It must preserve claim history and provenance links even when raw text or vectors are compacted. |
| `attention-controller` | Sees memory pressure only through ordinary blackboard metadata and module memos. It does not gain direct fact-update or retrieval power. |
| `self-model` | Integrates surfaced memory evidence into current first-person state. Stable self-knowledge comes from current identity-eligible memory/fact projections, not from superseded claims. |

This document does not introduce a new cognitive module. It refines the storage
adapter and capability behavior below the existing module boundary.

---

## 2. Canonical Model

The ledger separates durable evidence from operational views.

| Concept | Responsibility |
|---|---|
| `memory` | Logical durable record. Holds tenant/scope, origin, logical kind, creation time, and deletion marker. |
| `memory_version` | Versioned text/evidence body. Holds raw text, summary, language, salience, confidence, observed/learned/valid time fields, privacy level, extraction payload, current marker, and supersession link. |
| `entity` | Canonical entity identity within a tenant, such as a person, place, project, object, or organization. |
| `entity_alias` | Locale-aware names and spelling variants that can resolve to an entity. |
| `memory_mention` | Sidecar span linking a memory version to an entity, including mention text, optional character range, extraction method, and confidence. |
| `tag` | Canonical topic, domain, or classifier label within a namespace. |
| `memory_tag` | Sidecar assignment from a memory version to a tag, including source and confidence. |
| `fact_claim` | Append-only structured claim extracted from or directly inserted with a memory version. |
| `current_fact` | Mutable projection from `(tenant, subject, relation, scope)` to the claim currently used by ordinary response paths. |
| `memory_link` | Explicit edge between memory versions, such as `supports`, `contradicts`, `supersedes`, `derived_from`, `same_entity`, `same_tag`, or `similar_to`. |
| `memory_search` | Materialized search document combining current text, summary, resolved entity text, tag text, kind, status, and privacy filters. |
| `memory_vec` | Vector sidecar for semantic candidate generation, keyed to the memory version and privacy/search metadata. |

`memory_version` is the unit indexed for retrieval because edits, summaries,
archive transitions, and claim extraction all depend on a concrete text version.
If canonical text changes, the system creates a new version and ties old and new
versions with `supersedes` or `derived_from` links. Sidecar spans never need to
survive text mutation; they stay attached to the version whose text they describe.

Logical kind is an operational hint, not a hard store boundary:

- `episode` records an event, observation, interaction, or autobiographical moment.
- `fact` records a declarative memory item or imported knowledge record.
- `reflection` records derived interpretation or consolidation.
- `plan` records intended future work or commitments.
- `procedure` records reusable know-how.

Facts can be derived from episodes, imported directly, or stated explicitly by a
user. The store must preserve provenance either way.

---

## 3. Sidecar Extraction And Links

The write path stores raw text first, then derives sidecars. The extraction result
is a candidate interpretation, not a trust boundary.

The minimum extraction payload for a v2-capable store is:

- `entities[]` with labels, types, aliases, privacy hints, and optional spans,
- `tags[]` with namespaces, display labels, language, and confidence,
- `claims[]` with subject, relation, object, update policy, confidence, valid time,
  and source spans when available,
- `time_expressions[]` that can populate observed, learned, valid-from, and
  valid-to fields,
- `privacy_flags[]` that can constrain retrieval, embedding, export, or retention.

The extraction implementation may use schema-constrained LLM output, deterministic
rules, manual edits, or imports. The canonical persistence format is normalized
sidecar rows, not the LLM's raw JSON. The raw JSON may be retained in
`extraction_json` for audit and replay if its privacy level permits.

Human-facing edit surfaces may accept Obsidian-style `[[links]]`, but those links
are UI syntax only. On save, they are resolved to entity/tag ids and optional display
labels. Canonical memory text must not require those brackets to remain correct.

XML-like tags are allowed only as temporary debug/export renderings. They are not
canonical persistence because inline machine annotations make editing, span repair,
privacy redaction, and re-extraction brittle.

Embeddings are candidate generators. They can help retrieve multilingual,
paraphrased, or weakly related material, but they do not decide truth. Any response
that treats a fact as current must be grounded in `current_fact`, claim status,
source/provenance, or explicit retrieved evidence.

---

## 4. Retrieval

Retrieval combines independent candidate sources before reranking:

1. **Symbolic candidates** - Current facts, shared entities, shared tags, and
   explicit `memory_link` edges.
2. **Full-text candidates** - Keyword/phrase search over `memory_search.text`,
   `summary`, `tag_text`, and `entity_text`.
3. **Vector candidates** - Semantic nearest neighbors from `memory_vec`, filtered
   by tenant, currentness, logical kind when requested, and privacy level.

Candidate fusion should be deterministic, for example reciprocal-rank fusion or a
weighted sum of normalized source ranks. Reranking then applies typed policy:

- current memory versions outrank superseded or archived versions for ordinary
  questions,
- `current_fact` outranks old `fact_claim` rows for direct fact answers,
- explicit history, provenance, conflict, or correction questions may include
  superseded claims,
- `MemoryRank` and access metadata affect salience and durability but do not make a
  claim true,
- recency is a signal, not truth; source trust, confidence, relation policy, and
  corroboration can override it,
- privacy filters apply before material is returned to a module or embedded for an
  external provider.

`query-memory` still writes only retrieved evidence into its memo. Speak remains
downstream of cognition-gate and SpeakGate; it must not read hidden memory sidecars
directly.

Access-based rank elevation remains store-owned. A search path that records access
patches the blackboard metadata after successful retrieval. Modules see the result
as updated metadata, not as a separate rank-elevation capability.

---

## 5. Changed Facts

Changed facts are relation-policy problems, not memory-kind problems.

`fact_claim` rows use one of these default update policies:

| Policy | Use | Update behavior |
|---|---|---|
| `immutable` | Birth date, historical event occurrence, stable identity fact | Equivalent claims add support. Conflicting claims become disputed unless manually resolved. |
| `exclusive` | Current city, current employer, current preference when phrased as singular | A winning new claim supersedes prior active claims for the same scoped relation and updates `current_fact`. |
| `additive` | Skills, liked foods, known contacts, interests | Non-duplicate claims accumulate. Duplicates add support or aliases. |
| `event` | Trips, meetings, tasks performed | Claims append as events and normally do not supersede each other. |

When a user directly corrects an exclusive fact:

1. Insert a new `fact_claim` with the new source version, confidence, valid time, and
   transaction time.
2. Mark the prior current claim as `superseded`, set `valid_to` when known, and link
   it to the new claim.
3. Replace the `current_fact` projection for the relation scope.
4. Keep the old memory text, old claim, and provenance links for audit, explanation,
   and temporal queries.

Default conflict resolution orders evidence by:

1. explicit user correction or direct user statement,
2. trusted import or system source,
3. agent inference from recent evidence,
4. confidence/certainty,
5. corroboration count,
6. recency.

This ordering is a default, not a global truth oracle. High-stakes domains should
answer with "last confirmed" context or ask for confirmation even when a current
projection exists.

---

## 6. Compaction, Archiving, And Garbage Collection

Compaction reduces operational weight without destroying provenance.

Allowed compaction outcomes:

- create a summary/reflection memory version derived from source versions,
- archive low-salience raw text while keeping summaries, claims, and links,
- drop or regenerate vectors for archived, superseded, or low-value versions,
- merge duplicate tags or aliases after exact normalization or reviewed near-dedupe,
- remove search rows for material that is no longer retrievable under current
  privacy and archive policy.

Compaction must preserve:

- `fact_claim` history,
- `current_fact` correctness,
- `supersedes`, `supports`, `contradicts`, and `derived_from` links,
- enough source metadata to explain why a current fact is believed,
- `MemoryRank` and remember-token semantics for the resulting memory record.

The safest garbage-collection order is vector sidecars first, then search rows for
archived material, then raw text for low-salience episodes if a summary and claim
provenance remain. `fact_claim` rows and supersession chains are the last data to
delete and should normally be retained indefinitely unless privacy deletion requires
removal.

Full-text indexes over external content must be kept synchronized by adapter-owned
transactions or triggers. Any compaction or archive batch that mutates indexed text
must either update the index in the same operation or schedule a verified rebuild.

---

## 7. Implementation Shape For A Future Pass

The first implementation pass should keep the module-facing capability model intact
and broaden store behavior behind it.

Expected adapter changes:

- extend the primary memory store schema from single-row content plus vector search
  toward the ledger/projection model above,
- expose a memory search API that can return typed hits with provenance, status,
  privacy, rank, score breakdown, and optional current-fact metadata,
- keep simple vector-only compatibility during migration so existing evals can run,
- backfill existing v1 rows as active `memory_version` records with conservative
  logical kind and no extracted claims until extraction replay runs,
- load identity memories from identity-eligible current projections only, never from
  superseded claims.

Expected capability constraints:

- no module receives raw SQL, table names, or arbitrary projection mutation methods,
- no new mailbox carries durable memory answers,
- no module can stamp another module as the source of a memory write,
- claim projection updates happen inside the memory capability/store transaction,
  not by ad-hoc module code.

The exact Rust API names can be chosen in the implementation pass. The required
semantic boundary is that modules request preservation, retrieval, or compaction;
the store owns ledger consistency, extraction persistence, indexing, access
accounting, and current-fact projection.

---

## 8. Test Scenarios

Design-level acceptance scenarios for the implementation pass:

1. **Episode ingest** - Ingest one user episode and verify the raw text, active
   memory version, sidecar mentions, tags, search row, embedding row, and blackboard
   metadata are created coherently.
2. **Explicit correction** - Ingest a user correction for an exclusive relation and
   verify the old claim is superseded, `valid_to` is set when possible, and
   `current_fact` points to the new claim.
3. **Hybrid retrieval** - Query for a memory that requires symbolic, full-text, and
   vector candidates; verify fused results respect tenant, privacy, currentness,
   status, and rank/salience filters.
4. **Tag dedupe** - Insert exact duplicate tags after normalization and verify they
   collapse to one canonical tag; route near duplicates through alias/manual or LLM
   confirmation rather than automatic merge.
5. **Compaction provenance** - Compact redundant memories and verify
   `derived_from`/`supersedes` links, claim history, current projections, and
   access-rank behavior remain correct.
6. **Identity bootstrap** - Load startup identity memories from current
   identity-eligible records and verify superseded identity claims are not injected
   into module prompts as current facts.
7. **Privacy filtering** - Mark a memory version or embedding as above the query's
   allowed privacy level and verify it is excluded before module-visible retrieval
   results are formed.
8. **Temporal query** - Ask for what was believed at a prior valid or transaction
   time and verify superseded claims can be returned only for that temporal request.

---

## 9. Invariants

| Invariant | Enforcement target |
|---|---|
| Memory storage is unified | One ledger stores all logical memory kinds; projections provide typed behavior. |
| Memory rank is durability/salience only | Rank is separate from logical kind, claim status, and truth. |
| Canonical text is not machine-annotated | Extraction writes sidecar rows and optional audit JSON. |
| Facts are not hard-overwritten | New claims append; current projections move. |
| Current response paths are current-first | Ordinary retrieval favors active versions and `current_fact`; history appears only when requested or needed for conflict explanation. |
| Provenance survives compaction | Summary/archive operations preserve claim chains and memory links. |
| Capabilities remain the only module path | Modules hold memory capabilities, not raw database handles. |
| Blackboard remains non-canonical | Blackboard mirrors metadata and boot snapshots; the memory store owns durable content and indexes. |
| Query output remains memo-authoritative | Query-memory writes retrieved evidence to its memo and does not synthesize answers. |
| Embeddings are treated as sensitive | Vector rows obey privacy, provider, deletion, and retention policy. |
