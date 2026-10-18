# Mila Inference Server (MIS)

A FastAPI-based HTTP inference server for [Mila](https://github.com/ToddT/Mila). It imports the
`mila` Python binding and serves a model over HTTP under a selectable API protocol (OpenAI or
Anthropic Messages).

The default model is **`gemma-4-12b-it-fp4`**; any installed Llama 3.x or Qwen 3.8 model is served
too. MIS loads by store name, so what it can serve is whatever is installed — not a list compiled
into it. The record's architecture chooses the session and the prompt template, so nothing has to
be configured to match it.

Every model's prompt template and tool grammar come from the runtime, through the `mila` binding's
`ChatProtocol` for the loaded model. MIS renders no template of its own, so it and the chat harness
cannot send the same model different prompts.

---

## Requirements

- Windows or Linux
- **Python 3.12, 3.13 or 3.14** — the `mila` binding is a version-locked CPython extension
  (`_mila.cp313-win_amd64.pyd`), and a wheel is published for each of those minors. Any other
  interpreter yields `ModuleNotFoundError: No module named 'mila'`.
- CUDA-capable GPU (tested on RTX 4070 12GB)
- **A model installed in the local Mila store.** MIS loads by name and never downloads — see
  [Getting a model](#getting-a-model).

---

## Installation

MIS is the `mila-llm-server` package and depends on the `mila` runtime like any other Python
package. Where that runtime comes from is the only choice: the published wheel, or the one your
own build produced.

### 1. Create and activate a virtual environment

**Do this from this directory** (`Mila/Apps/Server`). A virtual environment is
required, not optional: a bare `python`/`pip` on your PATH may resolve to an interpreter with no
matching binding wheel (a uv-, conda-, or Store-managed Python). The version-pinned launcher
guarantees which one you get.

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

```bash
python3.13 -m venv .venv
source .venv/bin/activate
```

(If PowerShell blocks the activate script, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`
once, then retry.)

### 2. Install MIS

**From a release** — this pulls the `mila` runtime with it, and the runtime wheel carries its own
CUDA libraries (`nvidia-cublas`, `nvidia-curand`), so no CUDA Toolkit install is involved:

```
pip install mila-llm-server
```

**From this checkout** — install MIS editable, and point the `mila` dependency at your own build
rather than PyPI. The Mila build stages the freshly built extension into the package tree
(`Mila/Bindings/CMakeLists.txt`), so a later rebuild of `MilaPy` is picked up with no reinstall.

```
pip install -e ../../Bindings/Package
pip install -e ".[dev]"
```

Build `MilaPy` first, or the tree has no extension in it yet. It is a normal target of the standard
presets — nothing MIS-specific.

The *version* is the one thing a rebuild does not refresh: both packages take theirs from
`Version.txt` through the CMake configure, and pip records it at install time. After a version bump,
reinstall `Bindings/Package` — otherwise the stale metadata can fall below MIS's `mila-llm` floor
even though the code is newer.

Confirm the chain resolves before continuing:

```
python -c "import mila; print('mila', mila.__version__, mila.cuda_library_directories)"
```

An empty directory list is the thing to look at when the import fails. Since Python 3.8, Windows
does **not** search `PATH` when resolving an extension module's DLL dependencies — only system
directories, the extension's own directory, and directories registered with `os.add_dll_directory`.
The package's `__init__` handles that for you (it loads the pinned CUDA libraries before the
extension, on both platforms); an empty list means it found neither NVIDIA's CUDA wheels nor a local
CUDA Toolkit.

### Getting a model

MIS serves a model **by name** out of the local Mila store, the same store Chat uses. Starting the
server never downloads: if the name is not installed, startup fails with a message listing what is.

```
mila-server models --online
mila-server install gemma-4-12b-it-fp4
mila-server models
```

`models --online` lists what is published, `install` downloads into the store, and `models` lists
what is installed and whether each one runs on the card `MILA_DEVICE_INDEX` names: `yes`, `no`, or
`yes, at fp8 or fp4` for unquantized weights that fit once quantized as they load.

---

## Starting the Server

```bash
mila-server
```

With no command it serves, configured through `MILA_*` environment variables or a `.env` in the
working directory; it takes no flags. `python -m mila_llm_server` is equivalent, and
`uvicorn mila_llm_server.app:app --host 0.0.0.0 --port 8000` runs the same app under your own
uvicorn invocation.

On successful startup you will see:

```
INFO:     Started server process [...]
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### Interactive API Docs

FastAPI serves auto-generated documentation at `http://localhost:8000/docs` — the easiest way to
explore endpoints without a client.

---

## Configuration

Configured via environment variables or a `.env` file **in the working directory**. All variables
use the `MILA_` prefix.

Every variable has a default, so a store holding `gemma-4-12b-it-fp4` needs no configuration at all
and no `.env` on disk. The `.env` committed in this directory is a development convenience, picked
up when you run MIS from the checkout.

| Variable | Default | Description |
|---|---|---|
| `MILA_MODEL` | `gemma-4-12b-it-fp4` | Name of an installed model in the local store |
| `MILA_PROTOCOL` | `openai` | API protocol to expose: `openai` or `anthropic` (one per launch) |
| `MILA_CONTEXT_LENGTH` | `auto` | Tokens the model holds, prompt and reply together. `auto` takes the longest the GPU can hold; a number that does not fit stops startup, naming what it needs and what is free |
| `MILA_KV_CACHE` | `bf16` | How the model holds its context: `bf16` or `fp8`. `fp8` takes half the memory, so `auto` fits a longer context |
| `MILA_DEVICE_INDEX` | `0` | CUDA device ordinal |
| `MILA_KEEPALIVE_INTERVAL` | `15.0` | Seconds between SSE keepalive pings during prefill |
| `MILA_DECODE_TIMEOUT` | `30.0` | Seconds to wait for each subsequent token during decode |
| `MILA_HOST` | `0.0.0.0` | Server bind address |
| `MILA_PORT` | `8000` | Server port |
| `MILA_LOG_LEVEL` | `info` | Log level (`debug`, `info`, `warning`, `error`) |

The identifier a client sees in API responses is the **store record's** name, not `MILA_MODEL`. The
store matches case-insensitively, so the two can differ; what is reported is what was loaded. The
same holds for the context: `/v1/models` reports the length the model loaded at as `context_window`,
and a prompt longer than it is refused.

Reply length has no setting. A request that names no length may reply until the context is full, and
`/v1/models` reports the context length as `max_output_tokens`. A model that reasons spends much of its reply
thinking, so a small length cuts it off before it answers.

Sampling has no setting. A request that leaves out `temperature`, `top_k`, `top_p` or `presence_penalty`
gets the model's own recommended value, which can differ with thinking on and off: Qwen 3.8, for one,
recommends a lower temperature and a presence penalty when it is not thinking. A request's own values always
win. A Chat Completions or Completions request that sends `seed` gets the same reply each time it is sent with
the same seed and settings.

There is no family, path or quantization setting. All three are properties of the weights, which
the store record already states — `gemma-4-12b-it-fp4` *is* FP4 weights, and a variable that said
otherwise could only ever be wrong.

### Example `.env` (Gemma 4 12B, Anthropic protocol)

```env
MILA_PROTOCOL=anthropic
MILA_MODEL=gemma-4-12b-it-fp4
MILA_CONTEXT_LENGTH=auto
MILA_DEVICE_INDEX=0
```

---

## Protocols and Endpoints

`MILA_PROTOCOL` selects a single adapter at launch; only that adapter's routes are registered.

| Protocol | Endpoints |
|---|---|
| `openai` | `/v1/chat/completions`, `/v1/completions`, `/v1/responses`, `/v1/models` |
| `anthropic` | `/v1/messages`, `/v1/models` |

On the `openai` protocol, `/v1/completions` takes a `prompt` as text or as token ids, one prompt per
request. Token ids reach the model as sent, which is how an evaluation harness gives two engines the
same input. `stop` cuts a reply at the first stop sequence, streamed or not, on both Completions and
Chat Completions; on `anthropic`, `stop_sequences` does the same.

Every reply says why it ended. A reply cut off at `max_tokens`, or at the end of the context, reports
`length` on `openai`, and `max_tokens` or `model_context_window_exceeded` on `anthropic`;
one cut at a stop sequence reports `stop`, or `stop_sequence` with the sequence. On `/v1/responses` a
cut-off reply is `incomplete`, with `max_output_tokens` as the reason.

`/v1/models` answers in each protocol's own shape — OpenAI's `{"object": "list", "data": [...]}`
against Anthropic's `{"data": [...], "has_more": false}` envelope — and both carry the same lineage
fields.

Native tool calling (Gemma and Qwen) is wired on the OpenAI Responses path and the Anthropic
Messages path. On `/v1/messages`, tool_use is currently supported for **non-streaming** requests;
streaming `tool_use` is a work-in-progress (see `BACKLOG.md`).

### Lineage and attribution

`/v1/models` carries `base_model`, `license` and `attribution` alongside OpenAI's fields. The first
two come from the store record; `attribution` is the text the license requires be displayed wherever
the model is presented, and is empty for licenses that require none.

For the Llama families this is not optional — section 1.b.i of the Llama 3.1 and 3.2 Community
Licenses requires "Built with Llama" on a user interface or product documentation. A server has no
interface of its own, so MIS discharges it in the two places it presents the model: the startup log,
and this endpoint. **A client that renders a model list should render the attribution with it.**

---

## Dev Notes

### Layout

MIS is an src-layout package: the code lives in `src/mila_llm_server/`, and running it means
installing it. Nothing MIS owns sits on `sys.path` ahead of site-packages, which is what stopped a
stale copy of the binding shadowing an installed `mila-llm`.

### Python Environments

MIS runs from its own `.venv`, and the `mila` binding is locked to that venv's interpreter. Keep it
isolated from any other Python environment on the machine — notably a uv- or conda-managed
interpreter, or a separate venv for Mila's HuggingFace conversion scripts. If a different
environment is active, it shadows the MIS venv and `import mila` fails with `ModuleNotFoundError`.
The fix is always the same: activate `.venv` (or run `.venv\Scripts\mila-server.exe` directly).

### Running Tests

```bash
pytest
```

Requires the `dev` extras. The suite needs no GPU and no store: it covers the Gemma grammar, the
prompt builder, and model resolution against a fake store.
