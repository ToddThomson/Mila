---
title: "Driving a Local C++ Inference Server from Claude Code and Codex CLI"
date: 2026-05-14
description: "Point a foreign agentic harness at your own OpenAI-compatible endpoint: wiring the Mila Inference Server to Claude Code and Codex CLI, end to end."
discussion: "https://github.com/ToddThomson/Mila/discussions/10"
---

*Mila Discussions — Inference Server Series*

---

## Introduction

One of the goals of the Mila Inference Server (MIS) is to make a locally-running Llama model feel, from the outside, indistinguishable from a cloud API. If MIS does its job correctly, you should be able to point Claude Code or Codex CLI at `http://localhost:8000` and have them work as if they were talking to Anthropic or OpenAI's servers — except your data never leaves your machine and your GPU is doing the work.

Getting there turned out to be more interesting than expected. This article is an honest account of what it took to connect both clients to MIS, the surprises we hit along the way, and the exact fixes that made it work. It covers:

- What MIS is and what it currently supports
- Connecting Claude Code — the setup, the Windows-specific pitfalls, and the working configuration
- Connecting Codex CLI — a longer story involving a different API entirely, a role nobody documents, and a token flood that produced spectacular garbage output
- What's coming next (tool calling)

At the time of writing, MIS runs **Llama 3.2 3B Instruct at CUDA BF16** on an RTX 4070. It is the only supported model for now. The Alpha.4 roadmap adds Llama 3.1 8B Instruct at FP8, which will be the first tool-calling capable model.

---

## What Is the Mila Inference Server?

MIS is a FastAPI-based inference server that sits in front of Mila's C++ inference engine via a pybind11 binding. It exposes three OpenAI-compatible endpoints:

- `POST /v1/chat/completions` — the standard Chat Completions API
- `POST /v1/completions` — the legacy Completions API
- `POST /v1/responses` — the newer OpenAI Responses API

The protocol is selected via the `MILA_PROTOCOL` environment variable in `.env`. For both Claude Code and Codex CLI, set:

```env
MILA_PROTOCOL=openai
```

The server does not execute tools. It parses requests, builds a correctly-formatted Llama 3.x instruct prompt, runs inference through the Mila C++ engine, and returns a structured response. Tool calling support is on the roadmap and is discussed at the end of this article.

A minimal `.env` for getting started. `MILA_MODEL` is the name of a model already installed in
the local Mila store — MIS resolves it there and never downloads:

```env
MILA_MODEL=Llama-3.2-3B-Instruct-fp4
MILA_PROTOCOL=openai
MILA_LOG_LEVEL=info
MILA_CONTEXT_LENGTH=16384
MILA_DEFAULT_MAX_NEW_TOKENS=1024
MILA_DEFAULT_TEMPERATURE=0.6
MILA_DEFAULT_TOP_K=40
MILA_DEFAULT_TOP_P=0.9
```

---

## Connecting Claude Code

Claude Code is Anthropic's terminal-based coding assistant. It connects to any OpenAI-compatible server via a base URL override and uses the standard `/v1/chat/completions` endpoint.

Getting it working took a couple of days of iteration — the details of early false starts are lost to history, but the working configuration is straightforward once you know the pitfalls.

### Working configuration

Point Claude Code at MIS with:

```bash
claude --api-url http://localhost:8000 --model llama-3.2-3b-instruct
```

Or set it persistently in Claude Code's config. It will use `/v1/chat/completions` and the standard chat message format — `system`, `user`, `assistant` roles, `content` as a plain string. MIS handles this cleanly.

### Windows-specific: DLL resolution

On Windows, Python cannot find the CUDA runtime DLLs unless you explicitly add the CUDA bin directory to the DLL search path before importing the pybind11 module. This must happen at the very top of your FastAPI application, before any Mila import:

```python
import os

cuda_path = os.environ.get("CUDA_PATH", r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2")
os.add_dll_directory(os.path.join(cuda_path, "bin"))

import mila  # pybind11 module — safe to import now
```

Without this, the import fails with a cryptic `DLL load failed` error that gives no hint about which DLL is missing or why.

### Mila initialization placement

`Mila::initialize()` must be called in the pybind11 module entry point, not lazily on first use. If you defer it, the first inference call will either crash or produce garbage. This is handled internally in the binding — just ensure the module is imported before your FastAPI app starts serving requests.

### Special token registration

During early development, the BPE tokenizer was only registering 2 special tokens instead of the 7 that Llama 3.2's chat template requires. The fix was to register `extended_special_tokens` before calling `buildSpecialTokenList()`. If you see malformed prompt boundaries or the model fails to respect the instruct format, check your special token count — it should be 7 for Llama 3.2.

### Result

With these issues resolved, Claude Code connected cleanly and produced coherent, high-quality output. The `/v1/chat/completions` path in MIS is straightforward — Claude Code sends well-structured requests with no surprises.

---

## Connecting Codex CLI

This is where things got interesting.

Codex CLI is OpenAI's terminal-based coding agent. It is open source and under active development. We were running a recent version (post-0.122.0) and assumed it would work the same way as Claude Code — point it at MIS, set the model name, get responses.

That assumption was wrong in almost every detail.

### Surprise 1: Codex CLI uses a different API entirely

Claude Code uses `/v1/chat/completions`. Codex CLI, in recent versions, uses `/v1/responses` — the OpenAI Responses API, which is a newer and structurally different endpoint.

The first sign of this was a debug log showing the request arriving at `parse_responses_request` instead of `parse_chat_request`. MIS already had a `/v1/responses` route implemented, so the request was being handled — but the output was garbage.

The Responses API differs from Chat Completions in several ways:

- The system prompt arrives in an `instructions` field at the top level, not as a `system` role message inside `messages`
- The conversation history arrives in an `input` field, not `messages`
- `input` can be a plain string, or an array of typed message objects
- Message content is not a plain string — it is an array of typed content blocks: `{"type": "input_text", "text": "..."}`
- The response format uses an `output` array with typed items rather than a `choices` array

If your server is built for Chat Completions and you try to handle Responses API requests with the same parsing logic, you will extract empty strings or raw Python list objects as the user message, feed them to `build_instruct_prompt`, and the model will generate nonsense.

The fix is a proper content extractor:

```python
def _extract_content(content: str | list) -> str:
    if isinstance(content, str):
        return content
    text_parts = [
        block.get("text", "")
        for block in content
        if isinstance(block, dict) and block.get("type") in ("input_text", "text")
    ]
    return "".join(text_parts)
```

### Surprise 2: The `developer` role

Codex CLI sends its first input item with `"role": "developer"` — not `"system"`. This role carries Codex's own sandbox permissions and safety instructions. It is functionally a system message and must be treated as one.

If your parser only checks for `role == "system"` to extract the system prompt, the `developer` block falls through as a history turn with an unrecognised role. Llama's chat template has no `developer` role. The resulting prompt is malformed and the model generates garbage.

The fix is to consume all leading `system` and `developer` role items into the system prompt block:

```python
while messages and messages[0].get("role") in ("system", "developer"):
    system_parts.append(_extract_content(messages[0].get("content", "")))
    messages = messages[1:]

if system_parts:
    instructions = "\n\n".join(system_parts)
```

This is not documented in the Responses API spec. We found it by logging the raw request body.

### Surprise 3: Multiple consecutive `user` turns

After consuming the `developer` block, Codex CLI sends two consecutive `user` turns before the actual user message:

1. An environment context block containing `<cwd>`, `<shell>`, `<current_date>`, `<timezone>`
2. The actual user message ("Hi", or whatever the user typed)

A naive parser that takes `messages[-1]` as the user message and everything before as history will produce a history containing a `user` turn immediately followed by another `user` turn — with no `assistant` turn in between. This is invalid for Llama's instruct template, which expects strictly alternating `user` / `assistant` pairs.

The fix is to collapse consecutive same-role turns by merging their content:

```python
merged: list[dict] = []
for msg in messages:
    role = msg.get("role", "user")
    content = _extract_content(msg.get("content", ""))
    if merged and merged[-1]["role"] == role:
        merged[-1]["content"] += "\n" + content
    else:
        merged.append({"role": role, "content": content})
```

With this in place, the environment context and the user message are merged into a single coherent user turn.

### Surprise 4: The tool definition token flood

After fixing the parsing issues, MIS was producing output — but it was bizarre schema-like text rather than a response to "Hi". The debug log showed 11,346 tokens for a simple greeting.

The cause: Codex CLI sends its full set of shell tool definitions in the `tools` field of every request. These include `container.exec`, `container.write_file`, and several others, each with detailed JSON schemas. MIS was faithfully serializing all of them into the system block via `build_instruct_prompt`.

11,000 tokens of JSON schema in the context, followed by a `user` turn of "Hi", is not a situation Llama 3.2 3B handles gracefully. With only 1,024 tokens of generation budget remaining, the model pattern-matched on the schema-heavy context and produced schema-like text. It was not a model failure — it was a context failure.

The fix is simple: MIS does not execute tools. Strip them entirely:

```python
tools = None  # Codex CLI sends shell tool definitions; Mila doesn't execute them
```

After this single change, the same "Hi" prompt produced:

> *Hello! I'm ready to assist you. It's nice to meet you on this day, May 14th, 2026. I see we're in the America/Vancouver timezone. What's on your mind?*

Coherent, contextually aware, correctly formatted. Four surprises, four fixes.

### The model metadata warning

After getting coherent output, Codex CLI displays a warning on startup:

```
⚠ Model metadata for `llama-3.2-3b-instruct` not found.
  Defaulting to fallback metadata; this can degrade performance and cause issues.
```

This is a Codex CLI client-side issue, not a MIS issue. Codex ships with a built-in table of context windows and capabilities for OpenAI's own models. Any non-OpenAI model slug triggers this warning. Everything still works correctly.

To silence it, add to `~/.codex/config.toml`:

```toml
model_context_window = 16384
model_max_output_tokens = 1024
```

Match these values to your `MILA_CONTEXT_LENGTH` and `MILA_DEFAULT_MAX_NEW_TOKENS`.

### Working Codex CLI configuration

```toml
# ~/.codex/config.toml
model = "llama-3.2-3b-instruct"
model_context_window = 16384
model_max_output_tokens = 1024

[providers.mila]
name = "Mila Inference Server"
base_url = "http://localhost:8000"
requires_openai_auth = false
```

---

## The Responses API Streaming Protocol

For completeness, the full SSE event sequence MIS emits for a streaming Responses API request is:

```
event: response.created
event: response.output_item.added
event: response.content_part.added
event: response.in_progress          ← keepalive, repeated until first token
event: response.output_text.delta    ← one per token
...
event: response.output_text.done
event: response.content_part.done
event: response.output_item.done
event: response.completed
```

Codex CLI expects this exact sequence. The `response.completed` event carries the full assembled response in its `output` array, which is how Codex CLI renders the final reply.

---

## What's Coming: Tool Calling

Codex CLI is a coding *agent* — it reads files, writes patches, and runs shell commands. None of that works without tool calling, because those capabilities are exposed as tools that the model must invoke.

MIS currently strips all tool definitions from Codex CLI requests. This is correct for Llama 3.2 3B, which does not have reliable tool calling support. The Alpha.4 roadmap changes this:

**Alpha.4 adds Llama 3.1 8B Instruct at FP8** — the first tool-calling capable model in Mila. With it, MIS will gain:

- Tool definition forwarding (serialized into the system block for Llama 3.1's format)
- `ToolCallParser` integration via pybind11 — detecting `<|python_tag|>`-prefixed tool calls in model output and converting them to structured `function_call` response items
- Multi-turn tool result handling — `function_call_output` input items mapped to Llama's `ipython` role
- A `MILA_TOOL_CALLING_ENABLED` config flag gating the behaviour

The full design is specified in `specifications/toolcalling.md`.

Once tool calling is working, Codex CLI will be able to use MIS as a genuine local coding agent backend — reading your codebase, proposing edits, and running commands, entirely on your own hardware.

---

## Summary

| | Claude Code | Codex CLI |
|---|---|---|
| Endpoint | `/v1/chat/completions` | `/v1/responses` |
| Content format | Plain string | Typed content blocks |
| System prompt | `system` role in `messages` | `instructions` field + `developer` role |
| Tool definitions | Not sent | Sent every request — strip them |
| Streaming format | `data: {...}` SSE chunks | Typed SSE event sequence |
| Tool calling | Not applicable | Requires Alpha.4 + Llama 3.1 8B |
| Status | Working | Working (chat only) |

Both clients are now connected and producing coherent output from a local Llama 3.2 3B Instruct model running on an RTX 4070 via Mila's CUDA BF16 inference engine. No data leaves the machine. No API key required.

---

*Mila is an open source C++23/CUDA LLM inference framework. This article documents real development experience and will be updated as MIS matures.*

