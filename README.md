# Nuillu

Build a **lightweight cognitive agent that runs locally**, inspired by [Attention Schema Theory](https://en.wikipedia.org/wiki/Attention_schema_theory) (AST).

**Nuillu** is pronounced like *nu-illusion*. In Japanese, ぬイリュージョン (*nu-iryūjon*). The name is a portmanteau of **nu** and **illusion**.

## Design

The agent is built as a small society of cooperating cognitive modules: sensory, summarize, allocation-controller, attention-schema, query-vector, memory, memory-compaction, predict, surprise, and speak. The modules share an *attention stream*, the agent's running picture of what it is currently attending to. That stream drives resource allocation: the allocation-controller uses it to decide which modules are enabled, how often they run, and what model tier and context budget they receive. The agent also maintains a coarse internal model of its own attention through the **attention-schema** module rather than a complete description of its inner workings.

Two characteristics shape the implementation:

1. **Multi-agent decomposition over readily available LLMs.** Specialized cognitive roles are split across small, focused LLM calls instead of one large dedicated model.
2. **Deterministic logic where LLMs should not be in the loop.** Scheduling, ranking, decay, owner-stamped routing, and other state transitions are implemented as plain code, not LLM prompts.

## Why this design

Ideally, an agent of this shape would be a single large model with a purpose-built architecture and dedicated training. Developing and running that kind of model is not feasible for this project, so Nuillu approximates it under tighter constraints.

The multi-agent decomposition and the deterministic logic are therefore means, not ends. The end is a self-contained cognitive agent running on local hardware, and the project is an attempt to see what kinds of behavior, self-report, and adaptation fall out when such an agent has an explicit, shared model of its own attention instead of treating cognition as a single forward pass.


## Project status

A proof-of-concept is currently being built. It is not yet at a usable stage, let alone pre-alpha. The design (`docs/design/`) leads the implementation, and large parts of the system are still missing or in flux. Nothing is published to crates.io.

## License

Licensed under the **Mozilla Public License, version 2.0** (MPL-2.0). See [LICENSE](./LICENSE) for the full text.
