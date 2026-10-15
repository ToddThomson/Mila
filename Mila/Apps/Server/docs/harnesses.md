# Mila Inference Server (MIS)

The Mila Inference Server (MIS) is a FastAPI/Uvicorn HTTP server that exposes a locally
installed Mila model through OpenAI-compatible and Anthropic-compatible APIs. It is
designed to serve as a drop-in backend for AI coding agents such as the OpenAI Codex CLI,
the Anthropic Claude Code CLI and Hermes Agent.

**This page covers the harnesses.** Installing and running the server itself lives in
[the server's README](../README.md) — one copy, so the two cannot disagree.

The short version: `pip install mila-llm-server`, install a model into the local Mila store,
then

```bash
mila-server
```

The server listens on `http://0.0.0.0:8000` by default. The model is chosen by store name
(`MILA_MODEL`, default `gemma-4-12b-it-fp4`) and the protocol by `MILA_PROTOCOL`, from the
environment or a `.env` in the working directory.

---

## OpenAI Codex CLI

Codex CLI uses the OpenAI Responses API (`POST /v1/responses`).
Set `MILA_PROTOCOL=openai` in `.env`.

### Install

```bash
npm install -g @openai/codex
```

### Configure

Create or edit `~/.codex/config.toml`:

```toml
model = "gemma-4-12b-it-fp4"
model_provider = "mila"

[model_providers.mila]
name = "Mila Inference Server"
base_url = "http://localhost:8000/v1"
wire_api = "responses"
env_key = "MILA_API_KEY"
stream_idle_timeout_ms = 600000
```

`model` is the store name of the model MIS is serving. Set a placeholder API key (MIS does not
validate it):

```bash
# Linux / macOS
export MILA_API_KEY=mila

# Windows (PowerShell)
$env:MILA_API_KEY = "mila"
```

### Run

```bash
codex
```

---

## Anthropic Claude Code CLI

Claude Code uses the Anthropic Messages API (`POST /v1/messages`).
Set `MILA_PROTOCOL=anthropic` in `.env`.

### Install

```bash
npm install -g @anthropic-ai/claude-code
```

### Configure

Point Claude Code at MIS by setting environment variables:

```bash
# Linux / macOS
export ANTHROPIC_BASE_URL=http://localhost:8000
export ANTHROPIC_API_KEY=mila

# Windows (PowerShell)
$env:ANTHROPIC_BASE_URL = "http://localhost:8000"
$env:ANTHROPIC_API_KEY  = "mila"
```

Alternatively, add them to `~/.claude/.env` to persist across sessions.

Leave Claude Code's effort at `auto` for a model whose chat template has no effort setting,
such as Gemma 4 or Llama: MIS refuses an effort it cannot pass to the model. Set it with
`/effort auto`, `CLAUDE_CODE_EFFORT_LEVEL=auto`, or by removing `effortLevel` from
`~/.claude/settings.json`. Qwen 3.8 accepts `low`, `medium` and `xhigh`.

### Run

```bash
claude --model gemma-4-12b-it-fp4
```

---

## Hermes Agent

Hermes Agent uses the OpenAI Responses API (`POST /v1/responses`).
Set `MILA_PROTOCOL=openai` in `.env`.

### Install

```bash
curl -fsSL https://hermes-agent.nousresearch.com/install.sh | bash
```

### Configure

In `~/.hermes/config.yaml`:

```yaml
model:
  default: gemma-4-12b-it-fp4
  provider: custom
  base_url: http://localhost:8000/v1
  api_mode: codex_responses
```

### Run

```bash
hermes
```

---

## Notes

### Tested With

Run on Ubuntu 26.04 under WSL, against MIS on a 16 GB RTX 5060 Ti, each harness reading a file
with a tool and answering from it:

| Harness | Version | Gemma 4 12B | Gemma 4 26B-A4B | Qwen 3.8 27B |
|---|---|---|---|---|
| Codex CLI | 0.162.1 | yes | yes | yes |
| Claude Code | 2.1.300 | yes | yes | no: its first request is about 20,400 tokens, and a 16 GB card holds 11,264 for this model |
| Hermes Agent | 0.22.2 | yes | yes | yes |

Claude Code fills most of the 26B-A4B's 27,648-token context on a 16 GB card, so a long
session there runs out of room sooner than on the 12B.

### Context Window

MIS loads the model at the longest context the GPU can hold unless `MILA_CONTEXT_LENGTH`
fixes one, and `/v1/models` reports it as `context_window`. The harnesses send large
system prompts: 16,384 is the recommended minimum for Codex, and Claude Code's first
request is about 20,400 tokens before any work is done.

### Tool Calls

Each model calls tools in its own native format, on every API MIS serves: the model emits a
call, MIS returns it to the harness in that API's shape, and the harness executes it and
sends back the result. MIS never executes a tool itself.

### Model Quality

Response quality follows the model that is installed, not MIS. `gemma-4-12b-it-fp4` is the
default; the tool-calling paths are validated against the models in the table above. A 3B
answers concise tasks well and will struggle with multi-step coding work.
