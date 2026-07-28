# Mila Inference Server (MIS)

A FastAPI-based HTTP inference server for [Mila](https://github.com/ToddT/Mila). MIS is the
*wire adaptor* over the Mila runtime: it imports the `mila` Python binding and exposes it over
HTTP under a selectable API protocol (Mila native, OpenAI, or Anthropic Messages).

The default target is **Gemma 4 12B Instruct (FP4)**; the Llama 3.x family (BF16) is also supported.

---

## Requirements

- Windows or Linux
- **Python 3.13** — the `mila` binding is a version-locked CPython extension (`mila.cp313-win_amd64.pyd`).
  It loads *only* under the exact CPython minor version it was built against. A different interpreter
  (3.11, 3.12, 3.14, or an unrelated venv) yields `ModuleNotFoundError: No module named 'mila'`.
- CUDA-capable GPU (tested on RTX 4070 12GB)
- A Mila pretrained model artifact (`.bin`)
- A Mila tokenizer binary (`.bin`)

---

## Installation

Installation has two independent halves: the **`mila` binding** (a compiled C++ extension, built by
the Mila build) and the **Python server dependencies** (FastAPI etc., installed into an isolated
virtual environment). Do the steps in order.

### 1. Build the `mila` binding

`import mila` does **not** come from pip. It is the `MilaPy` CMake target. Build it in Visual Studio
2026 (or via CMake). A post-build step copies the extension into this directory automatically
(`Mila/Bindings/CMakeLists.txt`):

```
Copying mila extension module to Server directory
-> Mila/Adaptors/Inference/Server/mila.cp313-win_amd64.pyd
```

The `.pyd` is gitignored, so it exists only after a successful build. If you cloned fresh, moved the
server, or changed the binding, (re)build `MilaPy` so the current `.pyd` sits beside `main.py`.

### 2. Create and activate a Python 3.13 virtual environment

**Do this from this directory** (`Mila/Adaptors/Inference/Server`). A virtual environment is
required, not optional: the binding is locked to CPython 3.13, but a bare `python`/`pip` on your PATH
may resolve to a different interpreter (a uv-, conda-, or Store-managed Python), which then cannot
import the `cp313` binding. Creating the venv with the version-pinned launcher guarantees 3.13
regardless of your PATH, and activation makes `python`/`pip` unambiguous for the rest of the steps.

**Windows — VS 2026 Developer Command Prompt (cmd.exe):**

```bat
py -3.13 -m venv .venv
.venv\Scripts\activate.bat
```

**Windows — PowerShell:**

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

(If PowerShell blocks the activate script, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`
once, then retry.)

**Linux:**

```bash
python3.13 -m venv .venv
source .venv/bin/activate
```

Your prompt now shows `(.venv)`, and `python --version` reports 3.13.x. Verify:

```
python --version        # -> Python 3.13.x
```

### 3. Install the server dependencies

With the venv **active**, from this directory:

```
pip install -e .
```

For development extras (pytest, httpx):

```
pip install -e ".[dev]"
```

Confirm the whole chain resolves (binding + CUDA DLLs) before continuing:

```
python -c "import cuda_runtime, mila; print('mila OK', cuda_runtime.CUDA_DLL_DIRECTORIES)"
```

`cuda_runtime` must come first, and this is not a formality. Since Python 3.8,
Windows does **not** search `PATH` when resolving an extension module's DLL
dependencies — only system directories, the extension's own directory, and
directories registered with `os.add_dll_directory`. The binding links `cublasLt`
and `curand` from the CUDA Toolkit, so a bare `import mila` fails with
`DLL load failed while importing mila` on a machine whose CUDA install is
perfectly good. Importing `cuda_runtime` registers the toolkit directories; every
MIS module that imports the binding does the same. If the import still fails, the
printed directory list is the first thing to look at — an empty list means no CUDA
Toolkit was found at `CUDA_PATH` or the default install root.

---

## Configuration

Configured via environment variables or a `.env` file in this directory. All variables use the
`MILA_` prefix.

### Required

| Variable | Description |
|---|---|
| `MILA_MODEL_PATH` | Absolute path to the Mila pretrained model artifact (`.bin`) |
| `MILA_TOKENIZER_PATH` | Absolute path to the Mila tokenizer binary (Llama BPE or Gemma SentencePiece) |

### Optional

| Variable | Default | Description |
|---|---|---|
| `MILA_PROTOCOL` | `openai` | API protocol to expose: `mila`, `openai`, or `anthropic` (one per launch) |
| `MILA_MODEL_FAMILY` | `gemma` | Model family: `gemma` (Gemma 4, FP4) or `llama` (Llama 3.x, BF16) |
| `MILA_MODEL_NAME` | `gemma-4-12b-it` | Model identifier returned in API responses |
| `MILA_CONTEXT_LENGTH` | `4096` | Maximum sequence length passed to `fromPretrained()` |
| `MILA_DEVICE_INDEX` | `0` | CUDA device ordinal |
| `MILA_DEFAULT_MAX_NEW_TOKENS` | `1024` | Default token budget for generation |
| `MILA_DEFAULT_TEMPERATURE` | `0.6` | Default sampling temperature |
| `MILA_DEFAULT_TOP_K` | `40` | Default top-k (0 = disabled) |
| `MILA_DEFAULT_TOP_P` | `0.9` | Default top-p (nucleus) |
| `MILA_KEEPALIVE_INTERVAL` | `15.0` | Seconds between SSE keepalive pings during prefill |
| `MILA_DECODE_TIMEOUT` | `30.0` | Seconds to wait for each subsequent token during decode |
| `MILA_HOST` | `0.0.0.0` | Server bind address |
| `MILA_PORT` | `8000` | Server port |
| `MILA_LOG_LEVEL` | `info` | Log level (`debug`, `info`, `warning`, `error`) |

### Example `.env` (Gemma 4 12B, Anthropic protocol)

The checkpoint is stored BF16 (`..._bf16.bin`); the Gemma family is quantized to FP4 at load time,
so the on-disk artifact is the BF16 file. `MILA_CONTEXT_LENGTH=16384` is validated on the 12 GB 4070.

```env
MILA_PROTOCOL=anthropic
MILA_MODEL_FAMILY=gemma
MILA_MODEL_PATH=D:\Repos\Mila\Data\Models\Gemma\gemma4_12b_it_bf16.bin
MILA_TOKENIZER_PATH=D:\Repos\Mila\Data\Models\Gemma\gemma_tokenizer.bin
MILA_CONTEXT_LENGTH=16384
MILA_DEVICE_INDEX=0
```

---

## Protocols and Endpoints

`MILA_PROTOCOL` selects a single adapter at launch; only that adapter's routes are registered.

| Protocol | Endpoints |
|---|---|
| `openai` | `/v1/chat/completions`, `/v1/completions`, `/v1/responses`, `/v1/models` |
| `anthropic` | `/v1/messages` |
| `mila` | `/v1/chat/completions`, `/v1/completions` |

Native tool calling (Gemma family) is wired on the OpenAI Responses path and the Anthropic Messages
path. On `/v1/messages`, tool_use is currently supported for **non-streaming** requests; streaming
`tool_use` is a work-in-progress (see `BACKLOG.md`).

---

## Starting the Server

With the venv **active** (step 2 above — every new terminal needs to activate it again), from this
directory:

```bash
python main.py
```

Or equivalently via uvicorn directly:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

If you prefer not to activate, invoke the venv's interpreter directly (no activation needed):

```bat
.venv\Scripts\python.exe main.py
```

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

## Dev Notes

### Python Environments

MIS runs from its own `.venv` (Python 3.13), created in step 2. The `mila` binding is locked to that
same 3.13 interpreter. Keep this venv isolated from any other Python environment on the machine —
notably a uv- or conda-managed interpreter, or a separate venv for Mila's HuggingFace conversion
scripts. If a different environment is active when you run `python main.py`, it shadows the MIS venv
and `import mila` fails with `ModuleNotFoundError`. The fix is always the same: activate `.venv`
(or run `.venv\Scripts\python.exe` directly).

### Running Tests

```bash
pytest
```

Requires the `dev` extras to be installed.
