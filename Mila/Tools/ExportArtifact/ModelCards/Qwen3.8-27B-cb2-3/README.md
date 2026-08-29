---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
tags:
  - mila
  - qwen
  - codebook
  - quantized
library_name: mila
---

# Qwen3.8 27B Instruct — mixed 2/3-bit codebook for Mila

`Qwen/Qwen3.8-27B`, pre-quantized for [Mila](https://github.com/ToddThomson/Mila) and published in
safetensors format. The weights have been modified: they are quantized, and the codebooks were
fitted offline against calibration data.

`Qwen/Qwen3.8-27B` is the **instruction-tuned** checkpoint — Qwen marks the untuned variant
`-Base`, which is why neither name carries an `-it` or `-Instruct` suffix. It has a reasoning
channel and a trained tool-calling grammar, both of which Mila drives natively.

Unlike a uniform format, these weights spend a different number of bits per role. The
feed-forward and mixer projections carry fitted 2- and 3-bit codebooks; the full-attention
layers stay at 4.125 bits, where the accuracy is worth the space. The quantized parameters
average **2.82 bits per weight**.

## What this buys, and where

**In the download and on the device both.** This file is 11.1 GiB against 15.1 GiB for the FP4
build of the same model, and it runs Qwen3.8 27B on a 12 GB card where the FP4 build's weights
alone are 12.71 GB and need a larger one.

Nothing is quantized when it loads. Every tensor a policy quantizes is packed here — the
feed-forward and mixer rows at 2 and 3 bits, the full-attention and head rows at 4.125 — and the
2.4 GiB that remains BF16 is the embedding table, the norms and the mixer's gating, which no
policy quantizes.

Choose this model for a 12 GB card. Choose
[Qwen3.8-27B-fp4](https://huggingface.co/mila-llm/Qwen3.8-27B-fp4) if you have 16 GB, since
it is the more accurate of the two.

## Quality

Measured against the FP4 build as the oracle, teacher-forced over ~31,650 positions of
wikitext-2 test, each cell its own deployment:

| Context | FP4, 4.125 bits | This build, 2.82 bits | Ratio |
|---|---|---|---|
| 4096 | 6.439 | 7.089 | 1.101 |
| 8192 | 6.126 | 6.704 | 1.094 |
| 16384 | 5.686 | 6.478 | 1.139 |

Perplexity is **13.9% higher** than FP4 at 16K context — the cost of the smaller residency. The
ratio is flat from 1K to 16K, so the gap does not widen as the context grows.

Codebooks were fitted on wikitext-2 train.

## Files

| File | Purpose |
|---|---|
| `qwen38_27b_cb2-3.safetensors` | Weights: packed codebook indices, packed FP4 nibbles, and their per-group scales |
| `qwen38_tokenizer.bin` | Mila tokenizer |
| `mila.json` | Manifest: file digests, quantization scheme, minimum Mila version |
| `LICENSE` | Apache 2.0, as published with the base model |

## Use

From the Mila chat harness:

```
/model install Qwen3.8-27B-cb2-3
/model Qwen3.8-27B-cb2-3
```

Installing is a deliberate step, and it is the only one that touches the network. It verifies
each file against the digest in `mila.json` and leaves it in a content-addressed local store;
every load afterwards reads the store and nothing else. `/model list --online` lists what is
published, and `/model list` lists what is already installed.

Mila builds the codebook path at compile time from the scheme the weights declare, so a load
refuses weights whose scheme is not the one this build carries. There is no quantize-on-load
path to this format and there cannot be one: a codebook is fitted against calibration data, so
the weights either carry the codes or they do not.

## License

Apache 2.0, inherited from `Qwen/Qwen3.8-27B`. The full text is in `LICENSE`.
