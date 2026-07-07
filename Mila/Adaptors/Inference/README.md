# Mila Inference Server (MIS)

The Mila Inference Server (MIS) is a FastAPI/Uvicorn HTTP server that exposes a
local LLaMA 3.2 3B Instruct model through OpenAI-compatible and Anthropic-compatible
APIs. It is designed to serve as a drop-in backend for AI coding agents such as
the OpenAI Codex CLI and the Anthropic Claude Code CLI.

## Requirements

- Python 3.11+
- CUDA-capable GPU (SM 8.0 minimum; SM 8.9+ for FP8 quantization)
- Mila Python extension (`mila.pyd`) built and present on `PYTHONPATH`
- LLaMA 3.2 3B Instruct pretrained artifact in Mila binary format
- Mila BPE tokenizer binary

## Setup

```bash
cd Mila/Adaptors/Inference/Server
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and configure paths and options:

```ini
# === Paths (required)
MILA_MODEL_PATH=C:\path\to\llama32_3b_instruct_bf16.bin
MILA_TOKENIZER_PATH=C:\path\to\llama32_tokenizer.bin

# === Protocol: openai | anthropic
MILA_PROTOCOL=openai

# === Server
MILA_LOG_LEVEL=info

# === Context window
# Codex CLI requires at least 16384 to accommodate its system prompt.
MILA_CONTEXT_LENGTH=16384

# === Generation defaults
MILA_DEFAULT_MAX_NEW_TOKENS=1024
MILA_DEFAULT_TEMPERATURE=0.6
MILA_DEFAULT_TOP_K=40
MILA_DEFAULT_TOP_P=0.9
```

## Starting the Server

```bash
cd Mila/Adaptors/Inference/Server
python main.py
```

The server listens on `http://0.0.0.0:8000` by default.

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
model: llama-3.2-3b-instruct
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

LLaMA 3.2 3B Instruct was pretrained with a maximum sequence length of 131,072
tokens. MIS builds the model at `MILA_CONTEXT_LENGTH` to control GPU memory usage.
Codex CLI sends a large system prompt (~5,000 tokens) plus tool schemas; a context
length of 16,384 is the recommended minimum for Codex.

### Tool Calls

MIS accepts requests that include tool definitions (function schemas) but does not
execute them. Tool schemas are stripped from the prompt before tokenization because
LLaMA 3.2 3B was not trained on OpenAI-style function-calling JSON. The model
responds with plain text; the calling agent handles tool dispatch.

### Model Quality

LLaMA 3.2 3B is a small model. Response quality is best for concise tasks.
For complex multi-step coding work, keep the conversation history short to leave
sufficient generation budget within the context window.
