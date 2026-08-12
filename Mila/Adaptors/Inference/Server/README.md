# Mila Inference Server (MIS)

A FastAPI-based HTTP inference server for [Mila](https://github.com/ToddT/Mila). MIS is the
*wire adaptor* over the Mila runtime: it imports the `mila` Python binding and exposes it over
HTTP under a selectable API protocol (Mila native, OpenAI, or Anthropic Messages).

The default model is **`gemma-4-12b-it-fp4`**; any installed Llama 3.x model is served too. MIS
loads by store name, so what it can serve is whatever is installed — not a list compiled into it.

---

## Requirements

- Windows or Linux
- **Python 3.12 or 3.13** — the `mila` binding is a version-locked CPython extension
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

**Do this from this directory** (`Mila/Adaptors/Inference/Server`). A virtual environment is
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
pip install -e ../../../Bindings/Package
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

## Starting the Server

```bash
mila-server
```

The console script is the whole command line: serving is configured through `MILA_*` environment
variables, or a `.env` in the working directory. `python -m mila_llm_server` is equivalent, and
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

The identifier a client sees in API responses is the **store record's** name, not `MILA_MODEL`. The
store matches case-insensitively, so the two can differ; what is reported is what was loaded.

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
| `anthropic` | `/v1/messages`, `/v1/models` |
| `mila` | `/v1/chat/completions`, `/v1/completions` |

`/v1/models` answers in each protocol's own shape — OpenAI's `{"object": "list", "data": [...]}`
against Anthropic's `{"data": [...], "has_more": false}` envelope — and both carry the same lineage
fields.

Native tool calling (Gemma family) is wired on the OpenAI Responses path and the Anthropic Messages
path. On `/v1/messages`, tool_use is currently supported for **non-streaming** requests; streaming
`tool_use` is a work-in-progress (see `BACKLOG.md`).

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
