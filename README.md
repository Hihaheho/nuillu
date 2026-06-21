# Nuillu

Build a **lightweight cognitive agent that runs locally**, inspired by [Attention Schema Theory](https://en.wikipedia.org/wiki/Attention_schema_theory) (AST).

**Nuillu** is pronounced like *nu-illusion*. In Japanese, ぬイリュージョン (*nu-iryūjon*). The name is a portmanteau of **nu** and **illusion**.

## Design

The agent is built as a small society of cooperating cognitive modules. The default runtime wiring currently includes sensory, cognition-gate, allocation, attention-schema, self-model, query-memory, memory, memory-compaction, memory-association, memory-recombination, interoception, homeostasis, policy, policy-compaction, reward, predict, surprise, and speak.

Modules cooperate through a non-cognitive blackboard, per-module memo logs, typed transient channels, and an admitted **cognition log**. The cognition-gate promotes selected memo/blackboard state into that cognitive surface; attention-schema may append concise first-person attention-experience entries; self-model integrates those attention entries with stable and current context in its own memo log. Durable module output is memo- or log-authoritative rather than request/response traffic.

Resource allocation is durable priority state rather than a wake path. Allocation wakes from memo updates and internal attention-control bids, then writes activation priorities for modules. Effective allocation controls active replicas and rate limiting. Homeostatic control can also drive or cap allocation from interoceptive state.

Two characteristics shape the implementation:

1. **Multi-agent decomposition over readily available LLMs.** Specialized cognitive roles are split across small, focused LLM calls instead of one large dedicated model.
2. **Deterministic logic where LLMs should not be in the loop.** Scheduling, ranking, decay, owner-stamped routing, and other state transitions are implemented as plain code, not LLM prompts.

## Why this design

Ideally, an agent of this shape would be a single large model with a purpose-built architecture and dedicated training. Developing and running that kind of model is not feasible for this project, so Nuillu approximates it under tighter constraints.

The multi-agent decomposition and the deterministic logic are therefore means, not ends. The end is a self-contained cognitive agent running on local hardware, and the project is an attempt to see what kinds of behavior, self-report, and adaptation fall out when such an agent has an explicit, shared model of its own attention instead of treating cognition as a single forward pass.


## Project status

A proof-of-concept runtime is currently being built. Core crates, server/runtime wiring, eval tooling, and many built-in modules exist, but the project is still pre-alpha and APIs and behavior remain in flux. The design notes in `docs/design/` are the source of truth for architectural intent. Nothing is published to crates.io.

## License

Licensed under the **Mozilla Public License, version 2.0** (MPL-2.0). See [LICENSE](./LICENSE) for the full text.
