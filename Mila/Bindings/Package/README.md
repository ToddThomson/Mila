# mila-llm

The LLM stack you can read and understand.

Mila is a C++23/CUDA runtime for open large language models — inference and
training, built from explicit neural-network components. Device and precision are
compile-time decisions, every forward pass is explicit, and there is no hidden
execution engine. This package is its Python projection.

```bash
pip install mila-llm
```

```python
import mila

mila.initialize("warning")

# Once: fetch a published model into the local store (~6.3 GB).
mila.ModelStore().pull("gemma-4-12b-it-fp4", mila.default_hub_owner())

tokenizer = mila.BpeTokenizer.from_store("gemma-4-12b-it-fp4")
model = mila.GemmaModel.from_store("gemma-4-12b-it-fp4", 4096)

reason = model.generate(tokenizer.encode(prompt), print)
```

A model is named, not pathed. `from_store` reads the local store's record, which is
what knows the weights are already FP4 — so nothing pairs a weights path with a
tokenizer path, and nothing has to be told what the bytes are. **Pull and load are
separate verbs**: a load never reaches the network, so an uninstalled name is an
error rather than a surprise download.

`generate` hands each token to the callback as it is produced and returns why it
stopped — `stop`, `length`, `context_limit` or `cancelled` — which the tokens
themselves cannot tell you.

The GIL is released around generation, so the callback runs on a live interpreter
and `StopController` cancels a decode loop already in flight.

## Requirements

An NVIDIA GPU. The CUDA runtime libraries arrive as dependencies
(`nvidia-cublas`, `nvidia-curand`, and `nvidia-cuda-runtime` on Linux, where the
extension links cudart dynamically) — **no CUDA Toolkit installation is required**.
An installed Toolkit is used as a fallback if those are absent.

`gemma-4-12b-it-fp4` wants roughly 12 GB of VRAM at a 4096 context; a Llama 3.2 3B
is the smaller first run.

## What it exposes

| Symbol | Members |
|---|---|
| `mila.initialize` | `log_level` = `trace \| info \| warning \| error` |
| `mila.BpeTokenizer` | `from_store(name)`, `load_llama32`, `load_gemma`, `load_qwen`, `encode`, `decode`, `token_to_string`, `is_valid_token`, `vocab_size`, `bos_token_id`, `eos_token_id`, `pad_token_id` |
| `mila.GemmaModel` | `from_store(name, context_length, device_index=0)`, `from_pretrained(path, context_length, device_index=0, quantization="fp4")`, `generate(prompt_tokens, on_token, ...)`, `get_config` |
| `mila.LlamaModel` | `from_store(name, context_length, device_index=0)`, `from_pretrained(path, context_length, device_index=0, quantization="bf16")`, `generate(prompt_tokens, on_token, ...)`, `get_config` |
| `mila.QwenModel` | `from_store(name, context_length, device_index=0)`, `from_pretrained(path, context_length, device_index=0, quantization="fp4")`, `generate(prompt_tokens, on_token, ...)`, `get_config` |
| `mila.qwen_format_prompt` | `(history, enable_thinking=False, reasoning_effort=3, tools_json="")` — the runtime's own Qwen 3.8 template |
| `mila.qwen_parse_tool_call` | `(response)` → `{call id, name, arguments}` or `None` |
| `mila.qwen_protocol_tokens` | Qwen's control tokens, for a caller that streams |
| `mila.ModelStore` | `root`, `list`, `locate`, `remove`, `usage`, `install`, `pull`, `list_hub_models` |
| `mila.StopController` | `request_stop`, `stop_requested` |

The store is shared with Mila's chat harness and inference server, so a model
installed by any of them is loadable by all of them.

## What it does not

Stated because the limits are documentation, not an omission from it.

- **No weights are bundled.** They are fetched on request into a local store, over a
  transport this package supplies from the standard library — the wheel carries no
  HTTP client of its own.
- **A load never downloads.** `pull` and `from_store` are separate calls on purpose.
- **No GPT-2.** It exists in the C++ library and is not bound.
- **A published model's quantization is fixed** — its bytes are already FP4 or
  FP8. Choosing a quantization applies only to unquantized weights loaded by path.
- **No training**, no batching, and a model instance is not thread-safe: serialize
  calls through a single worker thread.
- **Text in, text out.** No embeddings, logits, or hidden-state access.

## Links

- Documentation: <https://mila.toddt.me>
- Source: <https://github.com/toddthomson/Mila>
- Models: <https://huggingface.co/mila-llm> — what `pull` fetches from
