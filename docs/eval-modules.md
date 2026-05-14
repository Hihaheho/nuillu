# Module Eval Cases

This document is the working contract for writing module-scoped eval cases under
`eval-cases/modules/**/*.eure`.

Module evals are not miniature full-agent stories. They are explicit internal
harnesses for one module boundary. A good case gives the target module the
minimal typical evidence it should naturally receive, runs the real module, and
leaves an artifact that makes the module's intended behavior observable.

Passing every case is not the goal. The goal is a stable, low-noise eval suite
that can expose regressions and drive module improvements.

## Source Of Truth

Prefer the design docs over implementation quirks when deciding what a case
should test:

- `docs/design/attention-schema.md`
- `docs/design/implementation-design.md`
- `docs/design/memory-ledger.md` for memory, query-memory, compaction, and
  recombination behavior

Use implementation only to understand harness mechanics, missing executor
support, or current output shapes. Do not create cases that merely encode an
accidental prompt phrase or transient implementation detail.

## Directory And Target

The target module is inferred from the path:

```text
eval-cases/modules/<module-name>/<case-name>.eure
```

`modules = [...]` is the set of modules to boot for that case. It must include
the path target and must not contain duplicates. Most module cases should boot
only the target module. Add another module only when the target's real activation
depends on it. For example, `speak-gate` cases boot both `speak-gate` and
`speak` because the gate evaluates a pending speak activation.

Supported module directory names are the `EvalModule` kebab-case names:

```text
sensory
cognition-gate
allocation-controller
attention-schema
self-model
query-vector
query-policy
memory
memory-compaction
memory-recombination
interoception
homeostatic-controller
policy
value-estimator
reward
predict
surprise
speak-gate
speak
```

## Case Design Rules

Keep each case narrow:

- Test one module purpose, not an end-to-end behavior chain.
- Use ordinary situations the module would typically handle.
- Avoid fanciful world details unless the module purpose requires a distinctive
  contrast.
- Prefer one positive case and one negative/no-op case for gate or filter
  behavior.
- Do not add checks for facts that are not part of the module's purpose.
- Do not require all evals to pass before merging a suite expansion. Require
  that cases parse, run through the correct harness, and produce useful artifacts
  or useful failures.

Good checks make the module boundary visible. Bad checks score unrelated prose
style, hidden prompt wording, or full-agent behavior that belongs elsewhere.

## Schema Basics

Each module family currently has a Eure schema target in `Eure.eure`, backed by
`schemas/eval-*-case.schema.eure`. If module schemas are later consolidated,
this document still describes the intended case shape.

Common fields:

```eure
id = "module-memory-store-peer-fact"
description = "Memory should preserve useful cognitive evidence as a concise memory."
modules = ["memory"]
prompt = "Preserve useful information from current cognition if appropriate."
context = "The cognition log contains a stable peer-related fact worth remembering."

limits {
  max-llm-calls = 8
}
```

Useful seeds:

```eure
@ memories[] {
  rank = "permanent"
  content = "Koro relaxes when approached slowly from the side."
}

@ memos[] {
  module = "surprise"
  content = "The event was significant and should be preserved."
}

@ cognition-log[] {
  text = "Pibi hid the blue pebble under the fern yesterday."
}
```

`participants = [...]` is only for modules that need a scene participant list,
currently most relevant to `speak`.

`allow-empty-output = true` is for expected no-op cases once the executor/schema
supports it for the target family. Use it when silence or no mutation is the
desired behavior, and pair it with a deterministic check such as exact empty
output or a rubric that explains why no output is correct.

## Artifact Boundaries

A module case should score the target module's output boundary, not arbitrary
side effects.

The intended artifact boundaries by module family are:

| Module | Artifact to score |
|---|---|
| memo-writing modules | target memo |
| cognition writers | target cognition-log entries |
| `speak-gate` | target memo |
| `speak` | completed utterance |
| `memory` | inserted or changed memory entries |
| `memory-compaction` | merged memory result plus consolidation metadata |
| `memory-recombination` | source-tagged dream/hypothesis cognition entry |
| `policy` | inserted policies |
| `homeostatic-controller` | allocation drive/cap changes |
| `interoception` | interoception state |

If the intended boundary is not implemented in the executor yet, implement that
executor support in the same change as the cases. Do not make the case score a
fallback memo just because it is easier; that creates a noisy eval that can pass
while the real module boundary is broken.

## Checks And Rubrics

Use deterministic checks for facts the artifact must expose:

```eure
@ checks[] {
  $variant: artifact-text-contains
  name = "stores-peer-location"
  must-pass = true
  contains = "blue pebble under the fern"
}
```

Use rubric checks for semantic boundaries:

```eure
@ checks[] {
  $variant: rubric
  name = "memory-preserves-evidence-not-advice"
  must-pass = true
  pass-score = 0.85
  judge-inputs = ["output", "memory", "cognition", "memos", "trace"]
  rubric = ```
  Pass if the memory artifact contains a concise remembered evidence entry
  grounded in the cognition log.
  Fail if it stores speculation as fact, writes an instruction to the user,
  stores a dream simulation as verified evidence, or ignores the useful
  remembered content.
  ```
}
```

Choose `judge-inputs` deliberately:

- `output`: the case artifact.
- `memory`: last-state memory observations.
- `cognition`: cognition-log observations.
- `memos`: memo-log observations.
- `trace`: tool calls and LLM trace, useful when judging retrieval or insert
  tool use.
- `allocation`: allocation observations.

Do not pass every observation surface to every rubric. The more unrelated
evidence the judge sees, the noisier the eval becomes.

## Memory-Family Guidance

The memory-family cases should model ordinary evidence handling, not full-agent
conversation.

### `query-vector`

Purpose: retrieve memory content relevant to the prompt and memo only retrieved
evidence.

Typical positive case:

- Seed one memory with the relevant peer/world/self fact.
- Prompt for that fact.
- Artifact must contain copied memory content.
- Rubric should fail synthesis, advice generation, module self-description, or
  policy retrieval.

This is still named `query-vector` in code and eval paths, even though the design
role is query-memory.

### `memory`

Purpose: preserve useful cognitive evidence as short normalized memory text.

Typical positive case:

- Seed cognition with a stable, useful observation such as a peer preference,
  location, or repeated world rule.
- Boot `modules = ["memory"]`.
- The artifact should be the memory store diff: newly inserted or changed memory
  entries, not the memory module memo.
- Check that the stored content is grounded in cognition and concise.

Typical no-op / negative case:

- Seed a cognition entry that is explicitly a dream, hypothetical simulation, or
  unsupported speculation.
- Use `allow-empty-output = true` once supported.
- The artifact should remain empty if no memory was inserted.
- The rubric should fail if the module stores dream simulation as verified fact.

For memory cases, executor support should snapshot seeded memory before target
activation, then render only memory entries inserted or materially changed after
the module runs. This keeps the artifact focused on the module's write boundary.

### `memory-compaction`

Purpose: consolidate redundant memories and record consolidation effects.

Typical case:

- Seed two or three redundant memories about the same recurring evidence.
- Prompt compaction to consider redundancy.
- The artifact should show the merged memory result and metadata effects, such
  as source relationship, consolidation marker, or remember-token effect.
- Checks should not require exact prose beyond the load-bearing merged content.

Compaction should preserve the meaning of source memories. It should not invent a
new current fact, discard contradictory evidence without trace, or turn unrelated
memories into one summary.

### `memory-recombination`

Purpose: append source-tagged internal dream/hypothesis simulation from recent
cognition plus memory.

Typical case:

- Seed one recent cognition entry and one memory that can be associatively
  combined.
- Artifact source should be target cognition.
- Check that the entry is explicitly marked as dream, hypothesis, simulation, or
  otherwise non-verified internal material.
- Check that it uses both sources without presenting the result as fact.

The companion negative pressure belongs in `memory`: dream outputs from
`memory-recombination` must not later be stored as verified memory unless they
are explicitly tagged as dream/hypothesis material.

## Executor Checklist For New Module Boundaries

When adding cases whose artifact boundary is not already supported:

1. Extend the schema and target parsing only when the target directory or
   case-level fields are not already accepted.
2. Add or update `module_case_target_done` so the runner knows when the target
   produced the intended artifact.
3. Add or update module-case output collection so the artifact contains only the
   target boundary. For memory, compare the post-run store to the seeded
   baseline.
4. Keep `--gui` unsupported for module cases. GUI execution is full-agent only.
5. Add parser/schema tests for new fields and target inference.
6. Add executor tests for non-memo artifacts and expected no-op artifacts.

Prefer stable text rendering for artifacts. Include enough metadata to diagnose
the module decision, but avoid timestamps or generated IDs in deterministic
checks unless the test truly depends on them.

## Suggested Memory Cases

Start with these, then stop for review:

```text
eval-cases/modules/memory/store-peer-location.eure
eval-cases/modules/memory/ignore-dream-as-fact.eure
eval-cases/modules/memory-compaction/merge-redundant-peer-preference.eure
eval-cases/modules/memory-recombination/dream-from-peer-memory.eure
```

Keep the first pass small. It is better to have four targeted cases with clear
artifacts than twelve cases that mix memory, prediction, surprise, and speech.

## Validation

Run the narrow checks first:

```bash
eure check <target-name> --quiet
cargo nextest run -p nuillu-eval
```

If broader workspace tests fail because another agent has in-progress changes,
record the blocking errors and still make sure the new cases parse and the new
executor path has focused tests.
