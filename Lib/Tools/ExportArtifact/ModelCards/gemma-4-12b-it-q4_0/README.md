---
license: apache-2.0
base_model: google/gemma-4-12B-it-qat-q4_0-unquantized
tags:
  - mila
  - gemma
  - q4_0
  - quantized
library_name: mila
---

# Gemma 4 12B Instruct — Q4_0 for Mila

Google's Gemma 4 12B Instruct in the 4-bit form Google trained it for, packaged for
[Mila](https://github.com/ToddThomson/Mila). It holds a conversation, thinks before it answers when asked to,
and calls tools.

**About 7 GB to download.** It runs on an NVIDIA GPU with 16 GB of memory and on a Mac with Apple silicon and
16 GB.

## Run it

In Mila Assistant, it is the model suggested on a first run when it suits the computer: choose *Install*.

From a terminal:

```
mila-store install gemma-4-12b-it-q4_0
mila-chat --model gemma-4-12b-it-q4_0
```

Inside `mila-chat`, `/model install gemma-4-12b-it-q4_0` installs it and `/model load gemma-4-12b-it-q4_0` loads
it. No account or token is needed.

## How well it holds a long conversation

Facts planted in a conversation and asked for later, 480 questions over 12 conversations at each length, on an
NVIDIA RTX 5060 Ti with 16 GB:

| Conversation length | Recalled exactly |
|---|---|
| 32,768 tokens | 479 of 480 |
| 65,536 tokens | 479 of 480 |
| 131,072 tokens | 473 of 480 |

Asked for a fact that was never in the conversation, it said so every time (96 of 96). Benchmark scores against
the original model are not yet measured.

## Files

| File | Purpose |
|---|---|
| `gemma4_12b_it_qat_q4_0.safetensors` | Weights |
| `gemma_tokenizer.bin` | Tokenizer |
| `mila.json` | What the model is and can do, and each file's checksum |
| `LICENSE` | Apache License 2.0 |

## Format

The weights are Q4_0: 4-bit values with one FP16 scale per group of 32, the grid Google's quantization-aware
training used, so the model runs in the format it was trained for. The token table, shared with the output head,
is stored as 6-bit values with one FP16 scale per group of 32. Norms stay BF16.

The file opens in any safetensors reader, but loading it as a model requires Mila; `transformers`, vLLM and
llama.cpp cannot load it.

## Modifications from the base model

Per the Apache 2.0 requirement to state changes:

- The quantization-aware checkpoint's weights stored as Q4_0, on the grid it was trained for
- The token table stored as 6-bit values with per-group scales, and shared with the tied output head
- Converted to Mila's safetensors layout
- No fine-tuning, distillation, or other change to what the model learned

## License and attribution

Gemma 4 is released by Google under the **Apache License 2.0**, and this derivative is distributed under the same
terms. See `LICENSE`.

Base model: `google/gemma-4-12B-it-qat-q4_0-unquantized`, © Google. This repository is not affiliated with or
endorsed by Google.
