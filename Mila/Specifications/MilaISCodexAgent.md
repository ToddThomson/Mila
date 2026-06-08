# Mila Inference Server + Codex CLI: Local Agentic Loop

**Date:** 2026-05-15
**Status:** Proven / Experimental
**Component:** `Mila/Inference/Server`

## Overview

This document describes the design and implementation of a protocol translation layer that
enables [Codex CLI](https://github.com/openai/codex) to drive an agentic coding loop against
a locally-hosted Llama 3.2 3B Instruct model served by the Mila Inference Server (MIS).

The result is a fully local, self-hosted agentic coding assistant: no OpenAI API key, no cloud
dependency, no model modification required.

## Architecture

```
Codex CLI  <-->  MIS /v1/responses  <-->  Llama 3.2 3B Instruct
 (OpenAI         (protocol bridge)         (local GPU)
 Responses API)
```

Three parties interact; none required modification:

- **Codex CLI** speaks the standard OpenAI Responses API with full JSON tool schemas.
- **Llama 3.2 3B** uses its native pythonic tool-calling format it was fine-tuned on.
- **MIS** is the translation layer in the middle.

## The Core Problem

Codex CLI sends tool definitions as full OpenAI JSON schemas — `$defs`, `oneOf`, nested
`properties`, boolean and array parameters. A 3B model cannot reliably parse and act on this
structure.

Llama 3.2 3B was instead fine-tuned on a simplified pythonic format:

```
<|python_tag|>[func_name(param1="value1", param2="value2")]
```

MIS bridges the gap in both directions.

## Inbound Translation: Tool Schema Distillation

`protocols/openai/tool_bridge.py` — `build_tool_injection(tools)`

When Codex sends a request to `/v1/responses`, the full tool schema array is distilled into a
flat system prompt suffix the 3B model can act on:

```
You have access to tools to complete tasks. When you need to use a tool, you MUST output
ONLY the tool call on a single line and nothing else. Use EXACTLY this format:
<|python_tag|>[func_name(param1="value1", param2="value2")]

Examples:
<|python_tag|>[exec_command(cmd="ls -la")]
<|python_tag|>[exec_command(cmd="printf '// hello\nint main(){return 0;}' > hello.cpp")]

Available tools:
- exec_command: Runs a command in a PTY...
  cmd (string): Shell command to execute.
```

Tools with complex argument structures that cannot be expressed in flat `key=value` form are
excluded. For Codex CLI this means `apply_patch`, `update_plan`, `spawn_agent`, and others
are suppressed, leaving only `exec_command` and `write_stdin` which map cleanly.

## Outbound Translation: Tool Call Detection and SSE Emission

### Buffered Streaming

Because the model emits a tool call token-by-token and we cannot know it is a tool call until
generation completes, `_stream_responses` in `routes/factory.py` suppresses all per-token SSE
delta events and buffers the full output. The `finally` block inspects the accumulated text
and branches:

- **Tool call detected** — emit the `function_call` SSE event sequence Codex expects.
- **Plain text** — emit the standard `output_text` SSE event sequence.

### Parsing

`tool_bridge.py` — `parse_tool_call(text)`

Detection order mirrors `Chat.ToolCallParser` from the Mila Chat sample:

1. Search for `<|python_tag|>` prefix; slice past it if found.
2. `find("[")` for bracket open, `rfind("]")` for bracket close.
3. `find("(")` for paren open, `rfind(")")` for paren close — `rfind` is required to handle
   cmd strings that contain parentheses (e.g. `main()`).
4. Parse `key="value"` pairs using `rfind('"')` for the closing quote, preserving embedded
   quotes in shell commands (e.g. `<<"Hello\n"`).

Known tokenization artifacts (`eexec_command`, `ejec_command`) are corrected via a lookup
table before the result is returned.

### SSE Event Sequence for a Function Call

```
event: response.created
event: response.output_item.added        (type: function_call, arguments: "")
event: response.function_call_arguments.delta
event: response.function_call_arguments.done
event: response.output_item.done
event: response.completed
```

## Turn Continuation: Tool Result Handling

After Codex executes the tool, it sends a follow-up request to `/v1/responses` with the
conversation history containing `function_call` and `function_call_output` items.

`parse_responses_request` converts these into Llama role turns:

| Responses API item type  | Llama history role | Content                           |
|--------------------------|--------------------|-----------------------------------|
| `function_call`          | `assistant`        | `<|python_tag|>[name(args)]`      |
| `function_call_output`   | `tool`             | shell output or `"(no output)"`   |
| `message`                | `user`/`assistant` | message content                   |

When a `function_call_output` is present, tool injection is suppressed and the system prompt
instructs the model to summarize and conclude rather than emit another tool call, breaking the
loop cleanly.

## Limitations

- **Model output quality:** Llama 3.2 3B occasionally corrupts tokens at quote boundaries
  (e.g. `int` → `inl`). This is a model capacity ceiling, not a protocol issue. A larger
  model (Llama 3.1 8B, Qwen2.5-Coder-7B-Instruct) eliminates this.
- **Single tool call per turn:** One tool call per generation turn is handled. Multi-tool
  turns would require a dispatch loop in `_stream_responses`.
- **`apply_patch` not supported:** The Codex `apply_patch` tool uses a structured `command`
  array argument incompatible with flat key=value format. File writes use `exec_command` with
  `printf` redirection instead.
- **No streaming progress during generation:** Tokens are buffered silently. The user sees no
  streaming indicator until the full response is ready.

## Relevant Files

| File | Purpose |
|------|---------|
| `protocols/openai/tool_bridge.py` | Schema distillation, tool call parsing |
| `protocols/openai/responses.py` | Responses API adapter, turn reconstruction |
| `routes/factory.py` | Buffered streaming, tool call branch in `finally` |

## Verified Configuration

- **Model:** `llama-3.2-3b-instruct` served by MIS on local GPU
- **Client:** Codex CLI (openai/codex), default approval mode
- **Task:** "Please create a helloworld.cpp file in the current project"
- **Result:** File created on disk, Codex loop terminated cleanly with a one-sentence summary
