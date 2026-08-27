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

A pre-quantized [Mila](https://github.com/ToddThomson/Mila) artifact of `Qwen/Qwen3.8-27B`, in
safetensors format. The weights have been modified: they are quantized to FP4 E2M1 with
per-group FP32 scales.

`Qwen/Qwen3.8-27B` is the **instruction-tuned** checkpoint — Qwen marks the untuned variant
`-Base`, which is why neither name carries an `-it` or `-Instruct` suffix. It has a reasoning
channel and a trained tool-calling grammar, both of which Mila drives natively.

**15.1 GB**, down from 50.1 GB at BF16. The packing is done once here instead of on every load,
so a Mila session starts without first quantizing 27 billion parameters.

Weights occupy 12.71 GB on the device, so this build wants a 16 GB card. For a 12 GB card, use
[Qwen3.8-27B-cb2-3](https://huggingface.co/mila-llm/Qwen3.8-27B-cb2-3),
which trades 13.9% higher perplexity for a residency that fits. This build is the more accurate
of the two, and it is the oracle that figure is measured against.

## Files

| File | Purpose |
|---|---|
| `qwen38_27b_it_fp4.safetensors` | Weights: packed FP4 E2M1 with per-group FP32 scales |
| `qwen38_tokenizer.bin` | Mila tokenizer |
| `mila.json` | Manifest: file digests, quantization scheme, minimum Mila version |
| `LICENSE` | Apache 2.0, as published with the base model |

## Use

From the Mila chat harness:

```
/model install Qwen3.8-27B-fp4
/model Qwen3.8-27B-fp4
```

Installing is a deliberate step, and it is the only one that touches the network. It verifies
each file against the digest in `mila.json` and leaves it in a content-addressed local store;
every load afterwards reads the store and nothing else. `/model list --online` lists what is
published, and `/model list` lists what is already installed.

Qwen3.8 has a reasoning channel. Mila's chat harness opens it by default and shows the answer
alone; `/thinking off` closes it, and `/verbose thoughts` shows the reasoning as well.

## License

Apache 2.0, inherited from `Qwen/Qwen3.8-27B`. The full text is in `LICENSE`.
