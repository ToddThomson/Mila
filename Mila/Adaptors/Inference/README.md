# Mila Inference Server (MIS)

The Mila Inference Server (MIS) is a FastAPI/Uvicorn HTTP server that exposes a locally
installed Mila model through OpenAI-compatible and Anthropic-compatible APIs. It is
designed to serve as a drop-in backend for AI coding agents such as the OpenAI Codex CLI
and the Anthropic Claude Code CLI.

**This page covers the harnesses.** Installing and running the server itself lives in
[`Server/README.md`](Server/README.md) — one copy, so the two cannot disagree.

The short version: `pip install mila-llm`, install a model into the local Mila store, then

```bash
python main.py
```

from `Mila/Adaptors/Inference/Server`. The server listens on `http://0.0.0.0:8000` by
default. The model is chosen by store name (`MILA_MODEL`, default `gemma-4-12b-it-fp4`)
and the protocol by `MILA_PROTOCOL`.

---

## OpenAI Codex CLI

Codex CLI uses the OpenAI Responses API (`POST /v1/responses`).
Set `MILA_PROTOCOL=openai` in `.env`.

### Install

```bash
npm install -g @openai/codex
```

### Configure

Create or edit `~/.codex/config.yaml`:

```yaml
model: gemma-4-12b-it-fp4
provider: openai
providers:
  openai:
    name: Mila Inference Server
    baseURL: http://localhost:8000
    envKey: OPENAI_API_KEY
```

Set a placeholder API key (MIS does not validate it):

```bash
# Linux / macOS
export OPENAI_API_KEY=mila

# Windows (PowerShell)
$env:OPENAI_API_KEY = "mila"
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

### Run

```bash
claude
```

---

## Notes

### Context Window

MIS builds the model at `MILA_CONTEXT_LENGTH` rather than at the architecture's maximum,
because the KV cache is the dominant VRAM cost. Both harnesses send large system prompts:
16,384 is the recommended minimum for Codex, and a full Claude Code turn has been measured
at 35.7K tokens — see the `.env` in `Server/` for the figures behind the default.

### Tool Calls

Gemma 4's native tool-call grammar is wired through on the OpenAI Responses path and the
Anthropic Messages path: the model emits a call, MIS surfaces it to the harness, and the
harness executes it and returns the result. MIS never executes a tool itself. A model whose
vocabulary has no tool-call tokens gets the schemas stripped and answers in plain text.

### Model Quality

Response quality follows the model that is installed, not MIS. `gemma-4-12b-it-fp4` is the
default and the one the tool-calling paths are validated against; a 3B answers concise
tasks well and will struggle with multi-step coding work.
