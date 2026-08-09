# Mila from Python

Two samples over `mila`, Mila's pybind11 extension module. Standard library only —
there is no `requirements.txt`, and that is the point: a local LLM in Python
without a framework underneath it.

| | |
|---|---|
| [`chat.py`](chat.py) | Streaming chat with Gemma 4. The instruct template, the token loop, the channel filter, and cooperative Ctrl-C — the whole thing, in one file. |
| [`generate.py`](generate.py) | Tokenizer round-trip and the sampling knobs (`temperature`, `top_k`, `top_p`), blocking generation, Gemma or Llama. |
| [`common.py`](common.py) | Finding the built extension and the weights. Not inference; it is here so the other two can open with the part worth reading. |

---

## Prerequisites

**A CUDA GPU.** The binding is CUDA-only. Gemma 4 12B FP4 wants ~12 GB of VRAM at
a 4096 context; Llama 3.2 3B BF16 is the smaller first run.

**Mila itself.** Once `mila-llm` is published, `pip install mila-llm` is all of it —
the samples import an installed wheel in preference to anything else. Until then,
a build of the `MilaPy` target. It is `ON` by default, but the checked-in build
directories may have it off:

```bash
cmake -S . -B out/build/x64-release -G Ninja -DCMAKE_BUILD_TYPE=Release -DMILA_ENABLE_PYTHON_BINDINGS=ON
```

The build publishes the `mila` package to `out/build/<preset>/python/`, which is
where the samples look. Run them with **the same Python the extension was built
against** — the ABI tag in the filename (`_mila.cp313-win_amd64.pyd`) has to match,
and `common.py` says so plainly if it does not. Override the search with
`MILA_PYD_DIR`.

The CUDA DLL directories are registered by `mila/__init__.py`, before the extension
loads. That is not optional politeness: since Python 3.8, `PATH` is **not** searched
when resolving an extension module's dependencies, so without it `import mila` fails
with `DLL load failed` on a machine whose CUDA install is perfectly fine. It lives in
the package rather than in the samples so that every consumer gets it for free.

**Weights.** Mila does not ship model weights, and these samples do not download
any. Convert a HuggingFace checkpoint with `Tools/Converters` (see
`Data/Models/README.md`); the samples default to:

| Family | Weights | Tokenizer |
|---|---|---|
| `gemma` | `Data/Models/Gemma/gemma4_12b_it_bf16.bin` | `Data/Models/Gemma/gemma_tokenizer.bin` |
| `llama` | `Data/Models/LLaMa/llama32_3b_instruct_bf16.bin` | `Data/Models/LLaMa/llama32_tokenizer.bin` |

Point elsewhere with `--weights` / `--tokenizer`, or `MILA_MODEL_PATH` /
`MILA_TOKENIZER_PATH`.

---

## Running

```bash
python Mila/Samples/Python/chat.py
```

```bash
python Mila/Samples/Python/generate.py --sweep
```

`chat.py`: Ctrl-C stops the current response and keeps the session, `/clear`
forgets the conversation, `/exit` quits, `--stats` prints time-to-first-token and
decode rate per turn.

`generate.py`: `--family llama --fp8` loads Llama with FP8 weights;
`--prompt "..."` sets the text; `--sweep` runs greedy / balanced / creative over
the same prompt so the knobs are visible side by side.

---

## What the binding exposes

| Symbol | Members |
|---|---|
| `mila.initialize` | `log_level` = `trace \| info \| warning \| error` |
| `mila.BpeTokenizer` | `load_llama32`, `load_gemma`, `encode`, `decode`, `token_to_string`, `is_valid_token`, `vocab_size`, `bos_token_id`, `eos_token_id`, `pad_token_id` |
| `mila.GemmaModel` | `from_store(name, context_length, device_index=0)`, `from_pretrained(path, context_length, device_index=0, quantization="fp4")`, `generate`, `generate_streaming`, `get_config`, `__repr__` |
| `mila.LlamaModel` | `from_store(name, context_length, device_index=0)`, `from_pretrained(path, context_length, device_index=0, quantization="bf16")`, `generate`, `generate_streaming`, `get_config`, `__repr__` |
| `mila.StopController` | `request_stop`, `stop_requested` |

The GIL is released around generation, so a streaming callback runs on a live
interpreter and `StopController` cancels a decode loop already in flight.

## What it does not

Stated here because the limits are documentation, not an omission from it.

- **No GPT-2.** `GptModel` exists in the C++ library and is not bound.
- **No precision choice for Gemma.** `GemmaModel` is FP4, always. That is the only
  configuration 12B fits a consumer card in, but the C++ API can express a choice
  the Python API cannot.
- **No wheel.** Reaching Mila from Python means building Mila first. A published
  binary wheel is the piece that would remove that, and it is post-v0.20 work.
- **No weight download.** Both samples expect a converted `.bin` on disk.
- **No training.** Inference sessions only.
- **No batching, and one request at a time.** A model instance is not thread-safe;
  serialize calls through a single worker thread (the Inference Server under
  `Mila/Adaptors/Inference/Server` is the worked example).
- **Text in, text out.** No embeddings, logits, or hidden-state access.

## Related

- `Mila/Adaptors/Inference/Server` — the OpenAI/Anthropic-protocol server built on
  this same binding: what the loop in `chat.py` grows into.
- `Mila/Specifications/PythonBinding.md` — why the binding is shaped this way and
  what is planned next.
