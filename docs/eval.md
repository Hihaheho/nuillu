# Nuillu eval

`nuillu-eval` evaluates a configured Nuillu agent. Every case uses the same
runtime path: the case names a normal server runtime config, eval expands that
config, constructs its scopes and modules, applies fixtures, drives stimuli,
and evaluates the resulting artifact and normalized event timeline.

There is no separate module-case format or module execution backend. To focus
an eval on one module, use a runtime config containing one module. To evaluate
an agent, point at the same topology the agent uses in production.

## Case format

```eure
id: greeting
description: The configured agent answers a peer greeting.
runtime-config: ../configs/agent.eure
participants = ["Peer"]

@ inputs[] {
  $variant: heard
  direction: Peer
  content: Hello
}

@ assertions[] {
  $variant: artifact-text-contains
  name: answers-peer
  must-pass = true
  contains: Hello
}

@ measurements[] {
  $variant: count
  name: llm-calls
  select {
    variants = ["llm-completed"]
  }
}
```

`runtime-config` is resolved relative to the case file and uses the server
config schema. It is the sole source of topology: module ids, replicas,
subsystems, dependencies, activation barriers, rates, groups, and model slots
must not be repeated in the eval case.

## Built-in evaluation runtime variants

The configs under `configs/eval/` are ordinary server runtime configs and can
be selected either from a case or with the CLI `--runtime-config` override.

| Config | Intended comparison |
| --- | --- |
| `default.eure` | The normal conversational baseline: the user-maintained compact configuration. |
| `sleep-wake.eure` | Only the homeostatic sleep/wake path plus observable suppression and memory-drive targets. |
| `prediction-surprise.eure` | Sensory-to-prediction-to-surprise flow with memory preservation. |
| `cognitive-minimal.eure` | Only sensory, cognition-gate, allocation, action, and speak. |
| `no-attention-schema.eure` | The default baseline with attention-schema removed. |
| `no-memory.eure` | The default baseline with query-memory and memory removed. |
| `focused/*.eure` | A normal one-module runtime for focused module evaluation. |

For example, the same topology-neutral case can be run as an ablation
benchmark:

```bash
cargo run -p nuillu-eval -- \
  --model-set configs/modelsets/eval.local.eure \
  --cases eval-cases/scenarios/responds-to-peer-greeting.eure \
  --runtime-config configs/eval/no-attention-schema.eure \
  --trials 5
```

Cases may declare top-level fixtures (`memories`, `memory-links`, `policies`,
`memos`, and `cognition-log`), direct `inputs`, or ordered `steps`. Assertions
operate on the final artifact. Rubric assertions use the configured judge model;
JSON-pointer assertions are preferred for structural invariants.

The final step may set `terminal = true` when its explicit `wait-for` marks the
user-visible task as complete. After that wait succeeds, the runner pauses new
module activations before taking the step-assertion snapshot, then terminates
without waiting for global runtime silence if those assertions pass. This is
useful for agent topologies whose inter-scope feedback remains active after an
answer has been produced. Terminal steps must declare `wait-for` and must be
last; omit the flag when the benchmark is intended to measure eventual runtime
convergence.

`wait-for.memo-from` may set `scope` to wait for that exact module scope. When
omitted, memos from the named module in any configured scope satisfy the wait.
`wait-for.utterance-from` additionally requires a completed utterance to match
the named `target`; use it for terminal steps that must wait for user-visible
output rather than an internal peer-directed utterance.

An utterance wait on a terminal step may set `until-assertion` to a named
case-level assertion and `max-matches` to an attempt budget. Each matching
utterance produces an immutable live artifact snapshot containing the matching
utterance history through that attempt. Snapshot judges run concurrently while
the agent runtime and event collector continue. The earliest passing attempt
terminates the step; otherwise collection closes immediately at the maximum
number of matches, waits for the already-started judges, and fails the step.
Live rubric assertions may not request trace or tool-call judge inputs.

`limits.timeout-ms` sets the wall-clock budget for the complete case trial,
including fixture setup, runtime execution, and judging. It defaults to 60000.
Step-level `wait-for.timeout-ms` remains a separate budget and should be lower
than the case timeout.

Memory, memo, and cognition fixtures accept a `scope` such as `/research[0]`.
Cognition fixtures also accept `module` and `replica`; defaults are root scope,
`cognition-gate`, and replica zero. Fixture scopes must exist in the referenced
runtime topology. A scoped memory is inserted through that scope's memory
namespace, so local-memory isolation is exercised by the ordinary runtime.

## Timeline and measurements

Every run writes `timeline.jsonl` and also exposes the same typed events under
`/observations/timeline`. Events contain a stable sequence, elapsed time,
`scope`, open `module` id, replica, optional scenario `step`, and typed
`variant`. Besides runtime lifecycle events, it includes content-bearing
`stimulus-published`, `memo-written`, `cognition-appended`, and
`utterance-completed` events. Selectors can restrict `scopes`, `modules`,
`replicas`, `variants`, and `steps`. For `cognition-appended`, `origin-scopes`
filters by the cognition producer's scope rather than the scope of the log that
stores it. Scope-grouped measurements also group by origin when this filter is
present. A rubric can request `timeline` as one of its `judge-inputs` for
semantic sequence checks such as topic residue.

Declared measurements are written to `/observations/measurements` and the case
report. Supported measurements are `count`, `first-match-latency`,
`unique-scope-count`, `scope-coverage`, and `scope-convergence-latency`. A
single-step latency selector is anchored at that step's stimulus rather than at
process startup. Multi-trial case summaries retain each trial's value and
report sample count, min, max, mean, standard deviation, p50, and p95
(including per-scope statistics).

`eval-cases/octopus/` demonstrates multi-scope agent benchmarks: abrupt topic
switches, competition between local-memory priors and a new shared topic, and
integration of facts distributed across separate local memories.

## Running a benchmark

```bash
cargo run -p nuillu-eval -- \
  --model-set configs/modelsets/eval.local.eure \
  --cases eval-cases \
  --runtime-config configs/agents/my-agent/config.eure \
  --trials 5 \
  --concurrency 3
```

Omit `--runtime-config` to use each case's own config. Supplying it replaces the
runtime config for every selected case, which makes the same benchmark suite
reusable across agent topologies.

Results are written below `.tmp/eval/<run-id>/`. `suite-report.json` includes
case scores, pass@k/pass^k, trial timing, activation records, assertions, and
measurements. Each trial directory contains its artifact, traces, last state,
and normalized timeline.

Live terminal output is concise by default: case/trial boundaries, step
results, utterances, idle diagnostics, stops, and warnings or failures. The
complete event stream is still written to each run or trial's `events.jsonl`.
Use `--verbose` to mirror every runtime event to the terminal, or `--quiet` to
print only the final suite summary.

## User-defined modules

Library users can evaluate host-provided modules as well as built-ins. Put the
custom module id in the runtime config, construct `RunnerConfig` with the same
`Arc<dyn nuillu_server::ServerModuleFactory>` used by `ServerHost`, and call
`run_suite` or `run_case_detailed`. The eval runner passes those factories to
the server-owned topology validator and registry builder, so custom modules and
nested scopes use the same construction contract as the running agent.

The CLI has no way to dynamically load Rust implementations; it evaluates
built-ins compiled into the binary. Applications embedding Nuillu eval should
provide their factories through `RunnerConfig::module_factories`.
