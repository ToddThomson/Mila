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

tokenizer = mila.BpeTokenizer.load_gemma("gemma_tokenizer.bin")
model = mila.GemmaModel.from_pretrained("gemma4_12b_it_bf16.bin", 4096)

model.generate_streaming(tokenizer.encode(prompt), print)
```

The GIL is released around generation, so a streaming callback runs on a live
interpreter and `StopController` cancels a decode loop already in flight.

## Requirements

An NVIDIA GPU. The CUDA runtime libraries arrive as dependencies
(`nvidia-cublas`, `nvidia-curand`) — **no CUDA Toolkit installation is required**.
An installed Toolkit is used as a fallback if those are absent.

Gemma 4 12B loads FP4 and wants roughly 12 GB of VRAM at a 4096 context; Llama 3.2
3B at BF16 is the smaller first run.

## What it exposes

| Symbol | Members |
|---|---|
| `mila.initialize` | `log_level` = `trace \| info \| warning \| error` |
| `mila.BpeTokenizer` | `load_llama32`, `load_gemma`, `encode`, `decode`, `token_to_string`, `is_valid_token`, `vocab_size`, `bos_token_id`, `eos_token_id`, `pad_token_id` |
| `mila.GemmaModel` | `from_pretrained(path, context_length, device_index=0)`, `generate`, `generate_streaming`, `get_config` |
| `mila.LlamaModel` | `from_pretrained(path, context_length, device_index=0, quantize_fp8=False)`, `generate`, `generate_streaming`, `get_config` |
| `mila.StopController` | `request_stop`, `stop_requested` |

## What it does not

Stated because the limits are documentation, not an omission from it.

- **Weights are not included and are not downloaded.** Models load from a Mila
  binary artifact produced by the converters in the source tree.
- **No GPT-2.** It exists in the C++ library and is not bound.
- **No precision choice for Gemma** — FP4, always.
- **No training**, no batching, and a model instance is not thread-safe: serialize
  calls through a single worker thread.
- **Text in, text out.** No embeddings, logits, or hidden-state access.

## Links

- Documentation: <https://mila.toddt.me>
- Source: <https://github.com/toddthomson/Mila>
