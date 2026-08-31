# Mila from Python

Samples over `mila`, Mila's pybind11 extension module. Standard library only — there is
no `requirements.txt`, and that is the point: a local LLM in Python without a framework
underneath it.

**Start with [`quickstart.py`](quickstart.py).** It is the whole thing in one screen, and
its C++ twin at [`../Cpp/main.cpp`](../Cpp/main.cpp) does the same job with the same model
and template, so the two read side by side.

```bash
python quickstart.py "Why is the sky blue?"
```

| | |
|---|---|
| [`quickstart.py`](quickstart.py) | One prompt in, tokens streamed out. Single-shot: no history, no REPL, thinking off. The smallest complete thing. |
| [`chat.py`](chat.py) | Streaming chat with Gemma 4. The instruct template, the token loop, the channel filter, and cooperative Ctrl-C — the whole thing, in one file. |
| [`generate.py`](generate.py) | Tokenizer round-trip and the sampling knobs (`temperature`, `top_k`, `top_p`), the reply collected rather than streamed, Gemma or Llama. |
| [`store.py`](store.py) | What the model store holds — listing, locating, and what each model costs on disk. No network: pull and load are separate verbs. |
| [`common.py`](common.py) | Finding the built extension and resolving a model. Not inference; it is here so the others can open with the part worth reading. |

---

## Prerequisites

**A CUDA GPU.** The binding is CUDA-only. Gemma 4 12B FP4 wants ~12 GB of VRAM at
a 4096 context; Llama 3.2 3B BF16 is the smaller first run.

**Mila itself.** `pip install mila-llm` is all of it — the samples import an
installed wheel in preference to anything else. Alternatively, build the `MilaPy`
target. It is `ON` by default, but the checked-in build directories may have it off:

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

**A model.** Mila does not ship weights, and none of these samples download one:
pull and load are separate verbs, so a multi-gigabyte transfer cannot begin inside a
sample. Fetch one deliberately — from Python, with no C++ build in sight:

```python
import mila
store = mila.ModelStore()
store.pull("gemma-4-12b-it-fp4", mila.default_hub_owner())
```

`/install <name>` in the chat harness does the same thing. Then name it:

```bash
python Mila/Samples/QuickStart/Python/chat.py --model gemma-4-12b-it-fp4
python Mila/Samples/QuickStart/Python/store.py          # what is already installed
```

`--model` defaults to `gemma-4-12b-it-fp4`. A store name is the whole key — the
record carries the architecture, the quantization, and which blobs are the weights
and the tokenizer, which is why `generate.py` needs no `--family` on this path.

**Loose `.bin` files are the fallback**, for a checkpoint converted locally from a
family Mila does not publish. Convert with `Tools/Converters` (see
`Data/Models/README.md`), then pass `--weights` / `--tokenizer` — or set
`MILA_MODEL_PATH` / `MILA_TOKENIZER_PATH` — and the defaults are:

| Family | Weights | Tokenizer |
|---|---|---|
| `gemma` | `Data/Models/Gemma/gemma4_12b_it_bf16.bin` | `Data/Models/Gemma/gemma_tokenizer.bin` |
| `llama` | `Data/Models/LLaMa/llama32_3b_instruct_bf16.bin` | `Data/Models/LLaMa/llama32_tokenizer.bin` |

On that path `--quantization` is a load-time choice, because the artifact is
unquantized; a published one already is, and its record says to what.

---

## Running

```bash
python Mila/Samples/QuickStart/Python/chat.py
```

```bash
python Mila/Samples/QuickStart/Python/generate.py --sweep
```

`chat.py`: Ctrl-C stops the current response and keeps the session, `/clear`
forgets the conversation, `/exit` quits, `--stats` prints time-to-first-token and
decode rate per turn.

```bash
python Mila/Samples/QuickStart/Python/generate.py --model Llama-3.2-3B-Instruct-fp4
```

`generate.py`: `--model` picks an installed model and its template follows from the
record; `--prompt "..."` sets the text; `--sweep` runs greedy / balanced / creative
over the same prompt so the knobs are visible side by side. On the `--weights` path,
`--family llama --quantization fp8` loads a converted Llama with FP8 weights.

---

## What the binding exposes

| Symbol | Members |
|---|---|
| `mila.initialize` | `log_level` = `trace \| info \| warning \| error` |
| `mila.BpeTokenizer` | `from_store(name)`, `load_llama32`, `load_gemma`, `encode`, `decode`, `token_to_string`, `is_valid_token`, `vocab_size`, `bos_token_id`, `eos_token_id`, `pad_token_id` |
| `mila.ModelStore` | `root`, `list`, `locate(name)`, `remove(name)`, `usage`, `install(package_directory, ...)`, `pull(name, owner, transport=None)`, `list_hub_models(owner, transport=None)` |
| `mila.http_transport` | The standard-library transport `pull` uses when none is passed; `mila.default_hub_owner()` names the owner Mila publishes under |
| `mila.GemmaModel` | `from_store(name, context_length, device_index=0)`, `from_pretrained(path, context_length, device_index=0, quantization="fp4")`, `generate(prompt_tokens, on_token, ...)`, `get_config`, `__repr__` |
| `mila.LlamaModel` | `from_store(name, context_length, device_index=0)`, `from_pretrained(path, context_length, device_index=0, quantization="bf16")`, `generate(prompt_tokens, on_token, ...)`, `get_config`, `__repr__` |
| `mila.StopController` | `request_stop`, `stop_requested` |

The GIL is released around generation, so a streaming callback runs on a live
interpreter and `StopController` cancels a decode loop already in flight.

## What it does not

Stated here because the limits are documentation, not an omission from it.

- **No GPT-2.** `GptModel` exists in the C++ library and is not bound.
- **Loading never downloads.** Pull and load are separate verbs by design, so a
  multi-gigabyte transfer can never begin inside an inference call. `ModelStore.pull`
  is how you fetch deliberately; none of these samples call it.
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
