# Tool Calling in the Mila Inference Server

**Specification version:** 0.1
**Target model:** Llama 3.1 8B Instruct (FP8, Alpha.4)
**Scope:** Single tool call per turn. Parallel tool calls deferred.

---

## Overview

When a client such as Codex CLI connects to MIS and sends a request that includes tool definitions, the expected contract is identical to what the client would receive from a cloud provider like OpenAI. The cloud provider's serving layer intercepts raw model tool call output and converts it to a structured response before it leaves the server. MIS must do the same — Codex CLI has no awareness of Llama's internal `<|python_tag|>` format and expects to receive a clean, structured `function_call` output item.

The tool calling flow therefore lives entirely within MIS, across three layers:

```
Codex CLI
    │  POST /v1/responses  (input + tool definitions)
    ▼
MIS: parse_responses_request
    │  build_instruct_prompt (with tools serialized into system block)
    ▼
Mila C++ inference
    │  raw text output, possibly containing <|python_tag|>...
    ▼
MIS: ToolCallParser (via pybind11)
    │  structured ToolCall or None
    ▼
MIS: format_responses_response / _stream_responses
    │  function_call output item  OR  message output item
    ▼
Codex CLI
    │  executes tool, sends result back as function_call_output turn
    ▼
MIS: parse_responses_request (multi-turn)
    │  ipython role turn injected into prompt
    ▼
Mila C++ inference  →  final assistant response
```

---

## Design Decisions

### D1 — ToolCallParser lives in C++, exposed via pybind11

`Chat.ToolCallParser` already implements the full detection and parsing logic for Llama 3.x tool call output:

- `<|python_tag|>`-prefixed blocks (primary path for Llama 3.1)
- Pythonic call syntax `[tool_name(key="value")]`
- Bare JSON fallback `{"name": ..., "arguments": {...}}`

This logic must not be duplicated in Python. A pybind11 binding exposes `ToolCallParser::parse(response: str) -> ToolCall | None` to the MIS Python layer.

The binding returns a simple dataclass-equivalent:

```python
@dataclass
class ToolCall:
    id: str        # e.g. "call_1"
    name: str      # tool name
    arguments: str # JSON string of arguments dict
```

### D2 — Streaming is suppressed for tool call responses

Tool calls cannot be streamed meaningfully — the complete output text must be accumulated before `ToolCallParser::parse()` can run. When streaming is requested and the completed output contains a tool call, MIS emits the lifecycle preamble events (`response.created`, `response.output_item.added`, `response.content_part.added`) as normal, but does not emit any `response.output_text.delta` events. It proceeds directly to the done events with a `function_call` output item.

This is transparent to Codex CLI, which does not expect delta events for function call output items.

### D3 — Tool definitions are passed through for Llama 3.1 8B only

`parse_responses_request` currently sets `tools = None` unconditionally, because Llama 3.2 3B does not have reliable tool calling. Once Llama 3.1 8B is the active model this suppression is lifted. Tool definitions arriving from Codex CLI are forwarded to `build_instruct_prompt`, which serializes them into the system block.

A model capability flag in `config.py` gates this behaviour:

```python
# config.py
MILA_TOOL_CALLING_ENABLED=false   # set true when Llama 3.1 8B is active
```

### D4 — Parallel tool calls deferred

Llama 3.1 8B can emit multiple tool calls in one turn. Codex CLI's agentic loop typically issues one tool call per turn. Parallel tool call support is deferred until single tool call flow is validated end-to-end.

---

## Prompt Format

### Tool definitions (system block)

Tool definitions are serialized as JSON and appended to the system prompt. `build_instruct_prompt` already supports this via the `tools` parameter. No changes required.

```
<|begin_of_text|>
<|start_header_id|>system<|end_header_id|>

You are a helpful assistant.

You have access to the following tools:
[
  { "type": "function", "function": { "name": "...", ... } }
]
<|eot_id|>
<|start_header_id|>user<|end_header_id|>

{user message}
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>

```

### Tool result turn (ipython role)

Llama 3.1's convention for injecting a tool result back into the prompt is the `ipython` role. When a `function_call_output` item arrives in the next request's `input` array, `parse_responses_request` maps it to an `ipython` history turn, which `build_instruct_prompt` renders as:

```
<|start_header_id|>ipython<|end_header_id|>

{tool result text}
<|eot_id|>
```

`build_instruct_prompt` requires a small extension to pass `ipython` through as a role without filtering or mapping it.

---

## Request Parsing Changes

### parse_responses_request — tool call input handling

The `input` array in a Codex CLI follow-up request (after a tool call) contains a `function_call_output` item:

```json
{
  "type": "function_call_output",
  "call_id": "call_1",
  "output": "{\"temperature\": \"22C\", \"condition\": \"sunny\"}"
}
```

This is not a `message` item and has no `role` field. `parse_responses_request` must detect it and map it to an `ipython` history turn:

```python
elif msg.get("type") == "function_call_output":
    content = msg.get("output", "")
    merged.append({"role": "ipython", "content": content})
```

The `call_id` is not forwarded to the prompt — Llama's format does not use it.

---

## Response Formatting Changes

### Non-streaming — format_responses_response

When `ToolCallParser::parse()` returns a `ToolCall`, the response output item is a `function_call` block rather than a `message` block:

```python
def format_responses_tool_call(self, tool_call: ToolCall, response: InferenceResponse) -> dict:
    response_id = f"resp-{uuid.uuid4().hex}"
    return {
        "id": response_id,
        "object": "response",
        "created_at": int(time.time()),
        "model": _MODEL_NAME,
        "status": "completed",
        "output": [
            {
                "type": "function_call",
                "id": tool_call.id,
                "call_id": tool_call.id,
                "name": tool_call.name,
                "arguments": tool_call.arguments,
            }
        ],
        "usage": {
            "input_tokens": response.prompt_token_count,
            "output_tokens": response.completion_token_count,
            "total_tokens": response.prompt_token_count + response.completion_token_count,
        },
    }
```

### Streaming — _stream_responses

The streaming path accumulates all tokens into `full_text` (already done). After generation completes, the `finally` block checks for a tool call before deciding which done events to emit:

```python
tool_call = parse_tool_call(full_text)   # pybind11 call

if tool_call:
    yield adapter.format_responses_stream_function_call(response_id, tool_call)
else:
    yield adapter.format_responses_stream_chunk(full_text, done=True, response_id=response_id)
    yield adapter.format_responses_stream_content_part_done(response_id, full_text)
    yield adapter.format_responses_stream_output_item_done(response_id, item_id, full_text)

yield adapter.format_responses_stream_done(response_id, tool_call=tool_call, output_text=full_text)
```

`format_responses_stream_function_call` emits a single SSE event:

```python
def format_responses_stream_function_call(self, response_id: str, tool_call: ToolCall) -> str:
    data = {
        "type": "response.output_item.added",
        "response_id": response_id,
        "output_index": 0,
        "item": {
            "type": "function_call",
            "id": tool_call.id,
            "call_id": tool_call.id,
            "name": tool_call.name,
            "arguments": tool_call.arguments,
            "status": "completed",
        },
    }
    return f"event: response.output_item.added\ndata: {json.dumps(data)}\n\n"
```

`format_responses_stream_done` gains an optional `tool_call` parameter so it can include the `function_call` item in the `response.completed` output array when appropriate.

---

## pybind11 Binding

A new binding in the existing pybind11 module exposes `ToolCallParser::parse()`:

```cpp
py::class_<ToolCall>( m, "ToolCall" )
    .def_readwrite( "id", &ToolCall::id )
    .def_readwrite( "name", &ToolCall::name )
    .def_readwrite( "arguments", &ToolCall::arguments );

m.def( "parse_tool_call", []( const std::string& response ) -> std::optional<ToolCall>
{
    return Mila::ChatApp::ToolCallParser::parse( response );
} );
```

On the Python side:

```python
import mila

def parse_tool_call(text: str):
    return mila.parse_tool_call(text)   # returns mila.ToolCall or None
```

---

## Implementation Order

1. Add `ToolCall` dataclass and `parse_tool_call` pybind11 binding
2. Add `MILA_TOOL_CALLING_ENABLED` config flag
3. Extend `build_instruct_prompt` to pass `ipython` role through unchanged
4. Update `parse_responses_request` to handle `function_call_output` input items and reinstate tool definitions when flag is set
5. Add `format_responses_tool_call` and `format_responses_stream_function_call` to `OpenAIAdapter`
6. Update `_dispatch_responses` and `_stream_responses` to call `parse_tool_call` and branch on result
7. Validate end-to-end with Llama 3.1 8B Instruct: single tool call round-trip via Codex CLI

---

## Out of Scope

- Parallel tool calls (multiple calls per turn)
- Tool execution within MIS (Codex CLI executes tools; MIS only parses and formats)
- Vision / multimodal tool inputs
- Llama 3.2 3B tool calling (unreliable; gated by config flag)
