---
license: llama3.2
base_model: meta-llama/Llama-3.2-3B-Instruct
tags:
  - mila
  - llama
  - fp4
  - quantized
library_name: mila
---

# Llama 3.2 3B Instruct — FP4 for Mila

**Built with Llama.**

`meta-llama/Llama-3.2-3B-Instruct`, pre-quantized for
[Mila](https://github.com/ToddThomson/Mila) and published in safetensors format.

**2.86 GiB**, down from 6.72 GiB at BF16. The FP4 packing is done once here instead of
on every load, so a Mila session starts near-instantly rather than quantizing 3 billion parameters
first. It is the smallest instruct model Mila publishes, and it leaves most of a 12 GB card free
for context.

## Files

| File | Purpose |
|---|---|
| `llama32_3b_instruct_fp4.safetensors` | Weights: packed FP4 E2M1 with per-group FP32 scales |
| `llama32_tokenizer.bin` | Mila tokenizer, shared across the Llama 3.x line |
| `mila.json` | Manifest: file digests, quantization, minimum Mila version |
| `LICENSE` | Llama 3.2 Community License Agreement |
| `NOTICE` | Attribution the license requires be retained |

## Use

From the Mila chat harness:

```
/model install Llama-3.2-3B-Instruct-fp4
/model Llama-3.2-3B-Instruct-fp4
```

Installing is a deliberate step, and it is the only one that touches the network. It verifies each
file against the digest in `mila.json` and leaves it in a content-addressed local store; every load
afterwards reads the store and nothing else. `/model list --online` shows what is published,
and `/model list` shows what is already installed.

From the library:

```cpp
ModelStore store;

// Pull once. This is the only thing here that touches the network.
const auto hub = makeDefaultModelHub();
ModelResolver resolver( store, *hub );
resolver.pull( "Llama-3.2-3B-Instruct-fp4", std::string( kDefaultHubOwner ) );

// Load from the store thereafter -- no network, and no manifest fetch.
const auto model = store.locate( "Llama-3.2-3B-Instruct-fp4" );
```

The quantization travels in the name rather than in a variant suffix: one name is one model.

No token is required — this repository is public and ungated. It is redistributed under the terms
below, which permit it; the base model on `meta-llama` is gated by Meta's own choice, not by the
license.

## Quantization

The transformer blocks' linear weights are FP4 E2M1, two values packed per byte, with FP32 absmax
scales per group of 128 along the input axis. Scales travel as sibling tensors
(`<name>.weight_scale`). Norms stay BF16.

**The token embedding and `lm_head` are not quantized, and not tied** — both stay BF16. On a 3B with
a 128k vocabulary that is a larger share of the file than it would be on a bigger model, which is
why this is roughly half the BF16 build rather than a quarter of it. Mila's Gemma chassis quantizes
and ties both; the Llama chassis does not yet, so the saving is available but unclaimed.

This is **Mila's own scheme, not a portable one.** It is deliberately not NVFP4 (group 16, FP8
scales) or MXFP4 (group 32, UE8M0 scales), so `transformers` and vLLM cannot consume it. The file
opens in any safetensors reader — shapes, dtypes and metadata are all inspectable — but loading it
as a model requires Mila.

## Modifications from the base model

- Weights quantized from BF16 to FP4 E2M1 with per-group scales
- Repacked from the HuggingFace checkpoint into Mila's tensor layout, then into safetensors
- Tokenizer converted to Mila's format
- No fine-tuning, distillation, or other change to what the model learned

Quantization is lossy. Mila publishes the quantized build because it is the one that fits the
hardware Mila targets; if you need the weights untouched, convert them yourself from
`meta-llama/Llama-3.2-3B-Instruct` with the converters in the Mila repository.

## License and attribution

Llama 3.2 is licensed under the **Llama 3.2 Community License**, Copyright © Meta Platforms, Inc.
All Rights Reserved. The full agreement is in `LICENSE` and the required attribution in `NOTICE`.

Use is subject to the [Llama 3.2 Acceptable Use Policy](https://www.llama.com/llama3_2/use-policy),
incorporated into the agreement by reference.

Base model: `meta-llama/Llama-3.2-3B-Instruct`, © Meta Platforms, Inc. This repository is not
affiliated with or endorsed by Meta.

Note for anyone deriving further: **Llama 3.1 and 3.2 are separate agreements** with different
attribution strings and different Acceptable Use Policies. A 3.1 derivative does not inherit this
one — take the license that matches its base model.
