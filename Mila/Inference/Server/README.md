# Mila Inference Server

A FastAPI-based HTTP inference server for [Mila](https://github.com/ToddT/Mila), exposing Llama 3.2 3B Instruct running on a CUDA BF16 backend.

---

## Requirements

- Windows or Linux
- Python 3.11 or newer
- CUDA-capable GPU (tested on RTX 4070 12GB)
- A Mila pretrained model artifact (`.bin`)
- A Mila tokenizer binary (`.bin`)

---

## Installation

Clone the repository and navigate to the inference server directory:

```bash
cd Mila/Inference/Server
```

Install the package and its dependencies:

```bash
pip install -e .
```

For development dependencies (pytest, httpx):

```bash
pip install -e ".[dev]"
```

---

## Configuration

The server is configured via environment variables or a `.env` file placed in the `Mila/Inference/Server` directory. All variables use the `MILA_` prefix.

### Required

| Variable | Description |
|---|---|
| `MILA_MODEL_PATH` | Absolute path to the Mila pretrained model artifact (`.bin`) |
| `MILA_TOKENIZER_PATH` | Absolute path to the Mila tokenizer binary (`.bin`) |

### Optional

| Variable | Default | Description |
|---|---|---|
| `MILA_CONTEXT_LENGTH` | `8192` | Maximum sequence length passed to `fromPretrained()` |
| `MILA_DEVICE_INDEX` | `0` | CUDA device ordinal |
| `MILA_STRICT_LOAD` | `true` | Raise on unrecognised parameter names during model load |
| `MILA_DEFAULT_MAX_NEW_TOKENS` | `256` | Default token budget for generation |
| `MILA_DEFAULT_TEMPERATURE` | `1.0` | Default sampling temperature |
| `MILA_DEFAULT_TOP_K` | `0` | Default top-k (0 = disabled) |
| `MILA_HOST` | `0.0.0.0` | Server bind address |
| `MILA_PORT` | `8000` | Server port |
| `MILA_LOG_LEVEL` | `info` | Uvicorn log level (`debug`, `info`, `warning`, `error`) |

### Example `.env`

```env
MILA_MODEL_PATH=C:\Users\ToddT\Src\Repos\Mila\Data\Models\LLaMa\llama32_3b_bf16.bin
MILA_TOKENIZER_PATH=C:\Users\ToddT\Src\Repos\Mila\Data\Models\LLaMa\llama32_tokenizer.bin
MILA_CONTEXT_LENGTH=8192
MILA_DEVICE_INDEX=0
```

---

## Starting the Server

```bash
python main.py
```

Or equivalently via uvicorn directly:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

On successful startup you will see:

```
INFO:     Started server process [...]
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### Interactive API Docs

FastAPI serves auto-generated documentation at:

```
http://localhost:8000/docs
```

This is the easiest way to explore and test endpoints without a client.

---

## Dev Notes

### Models Available

| File | Precision | Size |
|---|---|---|
| `llama32_1b_fp32.bin` | FP32 | ~5.6 GB |
| `llama32_1b_bf16.bin` | BF16 | ~2.8 GB |
| `llama32_3b_bf16.bin` | BF16 | ~6.7 GB |

The server targets `llama32_3b_bf16.bin` by default via `MILA_MODEL_PATH`.

### Python Environments

If you maintain a separate virtual environment for Mila's HuggingFace model conversion scripts, keep it isolated from the inference server. The inference server installs into system Python. Activating the conversion venv before running `python main.py` will cause import failures.

### Running Tests

```bash
pytest
```

Requires the `dev` extras to be installed.