# Mila Inference Server (MIS)

A FastAPI-based HTTP inference server for [Mila](https://github.com/ToddT/Mila). MIS is the
*wire adaptor* over the Mila runtime: it imports the `mila` Python binding and exposes it over
HTTP under a selectable API protocol (Mila native, OpenAI, or Anthropic Messages).

The default model is **`gemma-4-12b-it-fp4`**; any installed Llama 3.x model is served too. MIS
loads by store name, so what it can serve is whatever is installed — not a list compiled into it.

---

## Requirements

- Windows or Linux
- **Python 3.13** — the `mila` binding is a version-locked CPython extension
  (`_mila.cp313-win_amd64.pyd`). It loads *only* under the exact CPython minor version it was built
  against. A different interpreter (3.11, 3.12, 3.14, or an unrelated venv) yields
  `ModuleNotFoundError: No module named 'mila'`.
- CUDA-capable GPU (tested on RTX 4070 12GB)
- **A model installed in the local Mila store.** MIS loads by name and never downloads — see
  [Getting a model](#getting-a-model).

---

## Installation

MIS depends on the `mila` runtime like any other Python package. Where that package comes from is
the only choice: the published wheel, or the one your own build produced. Do the steps in order.

### 1. Create and activate a Python 3.13 virtual environment

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

### 2. Install the `mila` runtime

**From a release** — nothing else is needed; the wheel carries the extension and pulls its own CUDA
libraries (`nvidia-cublas`, `nvidia-curand`), so no CUDA Toolkit install is involved:

```
pip install mila-llm
```

**From your own build** — install the package tree in editable mode. The Mila build stages the
freshly built extension into it (`Mila/Bindings/CMakeLists.txt`), so this is done **once**: every
later rebuild of `MilaPy` is picked up with no reinstall.

```
pip install -e ../../../Bindings/Package
```

Build `MilaPy` first, or the tree has no extension in it yet. It is a normal target of the standard
presets — nothing MIS-specific.

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

### 3. Install the server dependencies

With the venv **active**, from this directory:

```
pip install -e .
```

For development extras (pytest, httpx, jinja2):

```
pip install -e ".[dev]"
```

### Getting a model

MIS loads a model **by name** out of the local Mila store, the same store the chat harness uses. It
never downloads: pull and load are separate verbs, and a server that fetched 6 GB because a name was
misspelled would be a worse failure than refusing to start. If the name is not installed, startup
fails with a message listing what is.

Install one with the chat harness (`/install gemma-4-12b-it-fp4`), with `ExportArtifact --install`
from a package you built, or from Python:

```python
import mila
store = mila.ModelStore()
store.pull("gemma-4-12b-it-fp4", mila.default_hub_owner())
print([model.name for model in store.list()])
```

`Mila/Samples/Python/store.py` does the same from the command line.

---

## Configuration

Configured via environment variables or a `.env` file in this directory. All variables use the
`MILA_` prefix.

Every variable has a default, so a store holding `gemma-4-12b-it-fp4` needs no configuration at all.

| Variable | Default | Description |
|---|---|---|
| `MILA_MODEL` | `gemma-4-12b-it-fp4` | Name of an installed model in the local store. Also the identifier returned in API responses |
| `MILA_PROTOCOL` | `openai` | API protocol to expose: `mila`, `openai`, or `anthropic` (one per launch) |
| `MILA_CONTEXT_LENGTH` | `4096` | Maximum sequence length passed to `from_store()` |
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

There is no family, path or quantization setting. All three are properties of the artifact, which
the store record already states — `gemma-4-12b-it-fp4` *is* FP4 weights, and a variable that said
otherwise could only ever be wrong.

### Example `.env` (Gemma 4 12B, Anthropic protocol)

`MILA_CONTEXT_LENGTH=16384` is validated on the 12 GB 4070.

```env
MILA_PROTOCOL=anthropic
MILA_MODEL=gemma-4-12b-it-fp4
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

With the venv **active** (step 1 above — every new terminal needs to activate it again), from this
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

MIS runs from its own `.venv` (Python 3.13), created in step 1. The `mila` binding is locked to that
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
