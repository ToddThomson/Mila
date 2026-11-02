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

Measured against the original `Qwen/Qwen3.8-27B` in BF16. Both builds read the same held-out
conversations, and at every position we check whether the build's most likely next token is the
original's:

| Build | Tool-calling conversations | Chat conversations |
|---|---|---|
| This model, 2.82 bits | 78.9% | 76.2% |
| [Qwen3.8-27B-fp4](https://huggingface.co/mila-llm/Qwen3.8-27B-fp4), 4.125 bits | 84.4% | 86.7% |

The FP4 build agrees with the original more often: by 5 points on tool calls and 10 on chat.

### How this was measured

8,184 positions per text, from conversations that played no part in fitting the codebooks. The
median KL divergence from the original is 0.018 nats on tool calls and 0.171 on chat (FP4: 0.006
and 0.042). The codebooks were fitted on tool-calling conversations, general chat and wikitext,
rendered through the model's own chat template.

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
each file against the digest in `mila.json` and leaves it in a local store;
every load afterwards reads the store and nothing else. `/model list --online` lists what is
published, and `/model list` lists what is already installed.

Mila builds the codebook path at compile time from the scheme the weights declare, so a load
refuses weights whose scheme is not the one this build carries. There is no quantize-on-load
path to this format and there cannot be one: a codebook is fitted against calibration data, so
the weights either carry the codes or they do not.

## License

Apache 2.0, inherited from `Qwen/Qwen3.8-27B`. The full text is in `LICENSE`.
