# Agent Rules

This file provides guidance to coding agents when working with code in this repository.

## Toolchain

Pinned in `rust-toolchain.toml`: **nightly** with **edition 2024** and Cargo **resolver = "3"**. Workspace lints forbid `unsafe_code` and warn on `dbg_macro` and `unused_must_use`.

## Build / test / lint

```bash
cargo build                                  # workspace
cargo nextest run                                   # all tests
cargo nextest run -p nuillu-agent                   # one crate
cargo nextest run -p nuillu-agent --test two_module_loop   # one integration file
cargo nextest run -p nuillu-agent --test two_module_loop -- attention_writer_is_exclusive
cargo clippy --workspace --all-targets
cargo fmt
eure check # validate all eure files against the schemas
```

Async tests use `#[tokio::test(flavor = "current_thread")]` and a `tokio::task::LocalSet` because the runtime is `?Send` / `spawn_local`-based — keep that pattern when adding tests that exercise `nuillu_agent::run` or modules holding `LlmAccess`.

## Architecture (read this before editing module wiring)

This is a capability-based agent runtime (design source of truth: `docs/design/attention-schema.md` and `docs/design/implementation-design.md`). A module's *role is exactly the set of capability handles it holds at construction* — invariants below are upheld by type-level wiring, not runtime checks.

### Crate layering (do not invert)

```
types  →  blackboard  →  module  →  modules/*
                               └→ agent (scheduler + event loop)
```

- `crates/types`: newtypes only (`ModuleId`, `MemoryRank`, `ModelTier`, …). No business logic.
- `crates/blackboard`: passive shared state (memos, attention stream, memory metadata, `ResourceAllocation`). Mutates only via `apply(BlackboardCommand)` under a single write lock.
- `crates/module`: the `Module` trait, outbound port traits (`MemoryStore`, `AttentionRepository`, `Clock`), capability handles, typed channels, and the `CapabilityFactory` that dispenses them at boot.
- `crates/modules/*`: one crate per cognitive module (sensory, cognition-gate, allocation, attention-schema, self-model, memory, homeostasis, vital, reward, predict, surprise, speak). Additional builtin ids in `nuillu_types::builtin` (query-memory, query-policy, memory-compaction, memory-association, memory-recombination, policy, value-estimator, speak-gate) are reserved for legacy data or implemented inside sibling crates.
- `crates/agent`: the scheduler (`run`) and `AgentEventLoop` — the only places that spawn module tasks or advance periodic time.

### Runtime shape

- Single-threaded: `nuillu_agent::run` accepts only `AllocatedModules` returned by `ModuleRegistry::build`, calls `tokio::task::spawn_local` per module, and runs inside a `LocalSet`. Do not expose or pass raw `Vec<Box<dyn Module>>` through public boot APIs. The `Module` trait is `#[async_trait(?Send)]`. Capability handles may capture non-`Send` state (e.g. `Rc`-bearing `lutum::Session`).
- The scheduler does not sleep, time, or interpret module state — it only spawns and waits for shutdown.
- Periodic activation is **explicitly** advanced by the application calling `AgentEventLoop::tick(elapsed)`. There is no internal wall-clock sleep inside inboxes. A module is eligible for periodic ticks only if it was granted `PeriodicInbox` via `factory.periodic_inbox_for(id)` — that call also registers it.

### Capabilities (the load-bearing concept)

`CapabilityFactory` is the single point that issues handles. **All capabilities are non-exclusive** — every issuer call returns a fresh handle and the factory does not enforce uniqueness. Single-writer roles (cognition-gate → attention stream, allocation → allocation) are upheld by boot-time wiring, not by panics.

- `AttentionWriter` — appends to the cognitive attention stream (conventionally granted to cognition-gate).
- `AllocationWriter` — replaces `ResourceAllocation` (conventionally granted to allocation).

Owner-stamped handles (`Memo`, `LlmAccess`, `PeriodicInbox`, `AttentionWriter`, typed `*Mailbox`/`*Inbox`) bake the `ModuleId` in at construction so a module cannot impersonate another. Don't add an API that lets a holder override the stamped owner.

`LlmAccess::lutum().await` resolves the holder's tier from the *current* allocation snapshot per call — modules don't pick tiers themselves.

### Module communication

- Typed broadcast topics (`QueryMailbox`/`Inbox`, `SelfModelMailbox`/`Inbox`, `AttentionStreamUpdated*`) carry `Envelope<T>` with a stamped sender. There is no shared `MailboxPayload` enum or `serde_json::Value` mailbox — add a new typed channel rather than overloading an existing one.
- Channel messages are *transient activation signals*. Durable module output goes in the producing module's `Memo` (or other blackboard state). Don't introduce request/response correlation on channels — answers are memo-authoritative.

### Invariants to preserve when editing

When changing module wiring or adding a module, these must remain true (they're enforced by which capabilities a constructor accepts):

- **Capabilities are non-exclusive**: the factory does not enforce uniqueness on any handle, and any capability may be granted to multiple modules. Don't reintroduce `*_taken` / panic-on-second-issuance.
- Allocation controller and attention schema are separate modules; controller has no `BlackboardReader`; schema has no allocation-write path.
- Cognition-gate is conventionally the only holder of `AttentionWriter` (boot-time wiring; not factory-enforced).
- Allocation controller is conventionally the only holder of `AllocationWriter` (boot-time wiring; not factory-enforced).
- Query receives `QueryInbox` only; self-model questions go to attention-schema via `SelfModelInbox`.
- Periodic activation is opt-in: only modules constructed with `PeriodicInbox` are reachable by `PeriodicActivation::tick`.
- `ModuleId` is kebab-case (`^[a-z][a-z0-9]*(?:-[a-z0-9]+)*$`); use `nuillu_types::builtin::*` for the cognitive set.

## Conventions

- This is a public repository: never commit anything unfit for public view — secrets/credentials, private IPs or hosts, LLM endpoint URLs, host-specific or personal model identifiers, internal infra topology, or local-only config filenames (e.g. `*.local.eure`). This applies everywhere, including devlogs, comments, and captured probe/report tables: redact or omit such fields rather than pasting raw run output. Reference model sets by the documented convention name (`configs/modelsets/test.local.eure`), not by an ad-hoc local file.
- Use `<project root>/.tmp` for scratch files (the repo's `.gitignore` covers it). Don't write to `/tmp`.
- Don't use Python for project scripts; prefer shell or Rust.
- In tests for structured JSON/schema values, prefer one `assert_eq!(actual, expected)` against the whole value. Don't walk a `serde_json::Value` with nested indexing, `find`, and multiple partial assertions when the expected shape is static.
- In new eval cases, don't use `artifact-text-contains` for invariants that can be read structurally. Prefer `json-pointer-equals` / `json-pointer-contains` against typed observations or memory diffs; reserve rubrics for natural-language quality.
- For numeric blackboard/allocation values (ratios, cooldowns, arousal, etc.) in eval cases, use `json-pointer-numeric-in-range` with `min`/`max` bounds. Don't assert numeric values via `json-pointer-equals` / `json-pointer-contains` against the float's string form — that pattern is brittle to formatting and breaks when the value moves *further* in the intended direction (e.g. `0.0` failing a "heavily suppressed" check that asserted `contains "0.0009"`).
- Full-agent eval `activate-allocation` seeds activation ratios only; do not write module guidance from full-agent eval bootstrap because guidance is allocation output. Module eval may use explicit guidance fixtures only when allocation is absent and the target module normally consumes allocation guidance.
- The `lutum` LLM SDK is a git dependency. The capability layer stops at returning a `Lutum`; each module builds its own `Session` and tool loop — don't add a shared session/agent-loop abstraction in `crates/module`.
- Do not add runtime compatibility code that detects and patches stale local data, old prompt text, or old persisted LLM session history after an implementation change. This is a strong anti-pattern: it makes production paths unreadable, unmaintainable, and complex. Instead, tell the user which local data, database, or session history must be reset or regenerated.

## libSQL migrations

- The persistent agent database is `agent.db`; don't reintroduce separate `memory.db` / `policy.db` server or eval state.
- The initial released baseline is `v0.0` with no domain schema; the first schema is a normal `v0.1.<task-tag>` dev migration until it is folded by the release tool.
- libSQL schema migration happens only during the explicit startup initialization path (`LibsqlAgentStore::connect`). Never add lazy migration in store read/write methods or first-use code paths.
- Schema changes in `crates/adapters/libsql` must add an idempotent dev task migration file under `crates/adapters/libsql/migrations/current/dev/` and a manifest entry in `crates/adapters/libsql/migrations/current/dev.eure`.
- Runtime migration lists are generated at build time from the migration manifests; keep manifest order authoritative and don't duplicate migration registration in Rust source.
- Dev task migration names are `v<major>.<next-minor>.<task-tag>.sql`; task tags are lowercase ASCII letters/digits/hyphens and are append-only once applied. If an applied task needs correction, create a new task migration instead of editing the old one.
- Released migrations, the current snapshot, and the previous-major bridge migration are release artifacts. Generate or update them with the Rust release tool in `lutum-libsql-adapter`; don't hand-edit them during ordinary feature work.
- Migration SQL must be safe to run more than once: use `CREATE TABLE IF NOT EXISTS`, `CREATE INDEX IF NOT EXISTS`, guarded column additions/backfills, and data updates that tolerate already-migrated rows.
- A major release keeps exactly one bridge migration from the previous major's terminal version to `v<current-major>.0`. Older/non-terminal major upgrades should fail clearly rather than attempting best-effort repair.

## Evaluation

If the task is running eval run with `eval.local.eure` for the model set.

`cargo run -p nuillu-eval -- --model-set configs/modelsets/eval.local.eure <options>`

Do not add eval-case-specific prompt text, examples, tool descriptions, or production guidance to make a failing eval pass. Prompt/module changes motivated by eval failures must be justified as case-agnostic product behavior or context-structure fixes; do not encode fixture names, scenario-specific success criteria, or rubric phrasing into runtime prompts.
