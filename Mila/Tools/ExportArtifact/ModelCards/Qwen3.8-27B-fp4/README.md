---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
tags:
  - mila
  - qwen
  - fp4
  - quantized
library_name: mila
---

# Qwen3.8 27B Instruct — FP4 for Mila

`Qwen/Qwen3.8-27B` quantized to FP4 for [Mila](https://github.com/ToddThomson/Mila). It reasons
before it answers, and it calls tools. (Qwen publishes the untuned model as `Qwen3.8-27B-Base`,
so the plain name is the instruction-tuned one.)

## What it needs

A **16 GB** card. The download is 15.1 GiB, down from 50.1 GiB at BF16.

For a 12 GB card, use [Qwen3.8-27B-cb2-3](https://huggingface.co/mila-llm/Qwen3.8-27B-cb2-3),
which fits in less memory at 13.9% higher perplexity.

## Use

```
/model install Qwen3.8-27B-fp4
/model load Qwen3.8-27B-fp4
```

`/model list` shows what you have installed, `/model list --online` what you can install.

Thinking is hidden by default. `/verbose thoughts` shows it, `/thinking off` turns it off.

## Quality

Perplexity on wikitext-2 test — lower is better:

| Context | 4096 | 8192 | 16384 |
|---|---|---|---|
| | 6.439 | 6.126 | 5.686 |

## Files

| File | Purpose |
|---|---|
| `qwen38_27b_fp4.safetensors` | The weights |
| `qwen38_tokenizer.bin` | Mila tokenizer |
| `mila.json` | Manifest: file digests, quantization, minimum Mila version |
| `LICENSE` | Apache 2.0, as published with the base model |

## Quantization

The transformer blocks' linear weights are FP4 E2M1, two values packed per byte, with FP32 absmax
scales per group of 128 along the input axis. Norms and embeddings stay BF16. Nothing was
fine-tuned, distilled or otherwise changed about what the model learned.

This is Mila's own format — not NVFP4 or MXFP4 — so `transformers` and vLLM cannot load it.

## License

Apache 2.0, inherited from `Qwen/Qwen3.8-27B`. The full text is in `LICENSE`.
