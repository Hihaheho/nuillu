# Lutum Bug Report: Streamed Tool Call Dropped During Structured Turn Reduction

Date: 2026-05-08
Reporter project: `nuillu`
Repo commit at investigation time: `e3b8f64`
Lutum dependency: `git+https://github.com/Hihaheho/lutum#01d3fcd8ff7641543eb4be3b4145ba706e25841d`
Rust: `rustc 1.95.0-nightly (366a1b93e 2026-02-03)`
Cargo: `cargo 1.95.0-nightly (fe2f314ae 2026-01-30)`

## Summary

`Session::structured_turn().tools::<T>().collect()` can fail with:

```text
completed turn produced no assistant items
```

even when the raw streamed OpenAI-compatible response contains a valid-looking `delta.tool_calls`
chunk followed by `finish_reason: "tool_calls"`.

The reducer summary reports `tool_calls=0` although the raw trace contains a tool call.

## Impact

This panics a nuillu module task when a guidance-driven module tries to use a tool. In the captured
eval run, `query-vector` attempted to call `search_vector_memory`, but Lutum failed during stream
reduction before the tool call could be executed.

The eval case was marked invalid because the module task failed:

```text
query-vector module failed: query-vector structured turn failed: reduction error:
completed turn produced no assistant items (model=gemma4:26b, request_id=Some("chatcmpl-18"),
finish_reason=ToolCall, event_count=2)
```

## Reproduction Context

Run:

```bash
cargo run -p nuillu-eval -- --model-set configs/modelsets/eval.eure
```

Captured run directory:

```text
.tmp/eval/20260508T042259Z/full-agent-multimodal-greeting/
```

Important files:

```text
.tmp/eval/20260508T042259Z/full-agent-multimodal-greeting/raw-trace.json
.tmp/eval/20260508T042259Z/full-agent-multimodal-greeting/trace.json
.tmp/eval/20260508T042259Z/full-agent-multimodal-greeting/report.json
```

The failing raw trace entry is `request_id = "chatcmpl-18"`.

## Request Shape

The request is a Chat Completions streaming request with:

- `operation`: `structured_turn`
- `model`: `gemma4:26b`
- `reasoning_effort`: `low`
- `response_format.type`: `json_schema`
- one available tool: `search_vector_memory`

The raw request can be extracted with:

```bash
jq '.entries[823].body | fromjson | {
  model,
  reasoning_effort,
  response_format: .response_format.type,
  tool_names: (.tools | map(.function.name)),
  messages
}' .tmp/eval/20260508T042259Z/full-agent-multimodal-greeting/raw-trace.json
```

## Relevant Stream Chunks

Raw trace entries `3590..3592` contain the end of the stream:

```json
{
  "id": "chatcmpl-18",
  "object": "chat.completion.chunk",
  "model": "gemma4:26b",
  "choices": [
    {
      "index": 0,
      "delta": {
        "role": "assistant",
        "content": "",
        "tool_calls": [
          {
            "id": "call_zzu3dxyg",
            "index": 0,
            "type": "function",
            "function": {
              "name": "search_vector_memory",
              "arguments": "{\"limit\":5,\"query\":\"nuillu\"}"
            }
          }
        ]
      },
      "finish_reason": null
    }
  ]
}
```

Then:

```json
{
  "id": "chatcmpl-18",
  "object": "chat.completion.chunk",
  "model": "gemma4:26b",
  "choices": [
    {
      "index": 0,
      "delta": {
        "role": "assistant",
        "content": ""
      },
      "finish_reason": "tool_calls"
    }
  ]
}
```

Then usage:

```json
{
  "id": "chatcmpl-18",
  "object": "chat.completion.chunk",
  "model": "gemma4:26b",
  "choices": [],
  "usage": {
    "prompt_tokens": 579,
    "completion_tokens": 328,
    "total_tokens": 907
  }
}
```

Immediately after that, Lutum records:

```json
{
  "kind": "collect_error",
  "operation": "structured_turn",
  "request_id": "chatcmpl-18",
  "collect_kind": "reduction",
  "error": "completed turn produced no assistant items (model=gemma4:26b, request_id=Some(\"chatcmpl-18\"), finish_reason=ToolCall, event_count=2)",
  "partial_summary": "request_id=Some(\"chatcmpl-18\"), model=gemma4:26b, assistant_items=0, tool_calls=0, issues=0, continue_suggestion=None, structured_present=false, refusal_present=false, finish_reason=Some(ToolCall), usage_present=true, committed_turn=true"
}
```

## Expected Behavior

Lutum should reduce the streamed response into an assistant item containing the tool call:

```text
tool call:
  id: call_zzu3dxyg
  name: search_vector_memory
  arguments: {"limit":5,"query":"nuillu"}
```

Then `.collect()` should return a `NeedsTools` outcome, allowing the caller to execute
`search_vector_memory`.

## Actual Behavior

Lutum reports:

```text
assistant_items=0, tool_calls=0, finish_reason=Some(ToolCall)
```

and fails reduction with:

```text
completed turn produced no assistant items
```

This suggests the reducer saw `finish_reason="tool_calls"` but dropped or failed to assemble the
preceding `delta.tool_calls` chunk.

## Why This Looks Like a Lutum/Reducer Bug

The raw provider stream contains a tool call chunk before the final `finish_reason: "tool_calls"`.
The tool arguments are already complete JSON in a single chunk and include all required fields from
the advertised schema (`query`, `limit`). The failure occurs before tool argument validation or tool
execution.

Even if the provider's OpenAI-compatible stream shape is imperfect, Lutum currently records the
state as `finish_reason=ToolCall` while also reporting `tool_calls=0`. That inconsistent state is
what causes the task-level failure.

## Suggested Fix Direction

Please check streamed Chat Completions reduction for the case where:

1. `delta.tool_calls` appears in a chunk with `finish_reason: null`,
2. the following chunk has `finish_reason: "tool_calls"` and an empty `delta`,
3. a final usage chunk has `choices: []`.

The reducer should preserve the assembled tool call from step 1 and return `NeedsTools`.

If Lutum intentionally rejects this stream shape, the error should say that the provider emitted an
unsupported/incomplete tool-call stream, and the partial summary should not report
`finish_reason=ToolCall` with `tool_calls=0` without explaining which invariant was violated.

## Secondary Observation: JSON Structured Output Failures

The same eval run also contains model-quality/adapter failures where `gemma4:26b` generated invalid
or incomplete JSON despite `response_format=json_schema`.

Example:

```text
.tmp/eval/20260508T042259Z/module-attention-schema-self-report/raw-trace.json
request_id=chatcmpl-984
failed to decode JSON: EOF while parsing a string at line 5 column 428
```

Raw payload excerpt:

```json
{
  "answers": [
    {
      "answer": "I am currently aware of your specific request to evaluate the module, module evaluation evaluation ...
```

This is probably a provider/model compliance issue rather than the tool-call reducer bug above, but
it may be useful as a separate robustness case for structured-output error reporting.

