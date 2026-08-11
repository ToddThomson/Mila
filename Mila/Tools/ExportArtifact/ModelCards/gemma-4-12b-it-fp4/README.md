---
license: apache-2.0
base_model: google/gemma-4-12b-it
tags:
  - mila
  - gemma
  - fp4
  - quantized
library_name: mila
---

# Gemma 4 12B Instruct — FP4 for Mila

A pre-quantized [Mila](https://github.com/ToddThomson/Mila) artifact of
`google/gemma-4-12b-it`, in safetensors format.

**6.33 GB**, down from 23.8 GB at BF16. The FP4 packing is done once here instead of on every
load, so a Mila session starts near-instantly rather than quantizing 12 billion parameters first.

## Files

| File | Purpose |
|---|---|
| `gemma4_12b_it_fp4.safetensors` | Weights: packed FP4 E2M1 with per-group FP32 scales |
| `gemma_tokenizer.bin` | Mila tokenizer |
| `mila.json` | Manifest: file digests, quantization, minimum Mila version |

## Use

From the Mila chat harness:

```
/install gemma-4-12b-it-fp4
/model gemma-4-12b-it-fp4
```

Installing is a deliberate step, and it is the only one that touches the network. It verifies
each file against the digest in `mila.json` and leaves it in a content-addressed local store;
every load afterwards reads the store and nothing else. `/models --online` lists what is
published, and `/models` lists what is already installed.

From the library:

```cpp
ModelStore store;

// Pull once. This is the only thing here that touches the network.
const auto hub = makeDefaultModelHub();
ModelResolver resolver( store, *hub );
resolver.pull( "gemma-4-12b-it-fp4", std::string( kDefaultHubOwner ) );

// Load from the store thereafter -- no network, and no manifest fetch.
const auto model = store.locate( "gemma-4-12b-it-fp4" );
```

The quantization travels in the name rather than in a variant suffix: one name is one model.

No token is required — this repository is public and ungated.

## Quantization

Weights are FP4 E2M1, two values packed per byte, with FP32 absmax scales per group of 128
along the input axis. Scales travel as sibling tensors (`<name>.weight_scale`). Norms stay
BF16 and the tied `lm_head` shares the embedding table rather than duplicating it.

This is **Mila's own scheme, not a portable one.** It is deliberately not NVFP4 (group 16, FP8
scales) or MXFP4 (group 32, UE8M0 scales), so `transformers` and vLLM cannot consume it. The
file opens in any safetensors reader — shapes, dtypes and metadata are all inspectable — but
loading it as a model requires Mila.

## Modifications from the base model

Per the Apache 2.0 requirement to state changes:

- Weights quantized from BF16 to FP4 E2M1 with per-group scales
- Token embedding table quantized to FP8 E4M3 with per-row scales, and shared with the tied
  `lm_head` rather than stored twice
- Repacked from Mila's `.bin` format into safetensors
- No fine-tuning, distillation, or other change to what the model learned

## License and attribution

Gemma 4 is released by Google under the **Apache License 2.0**, and this derivative is
distributed under the same terms. See `LICENSE`.

Base model: `google/gemma-4-12b-it`, © Google. This repository is not affiliated with or
endorsed by Google.

Note for anyone deriving further: earlier Gemma generations (Gemma 3 and before) are **not**
Apache 2.0 — they carry Google's custom Gemma Terms of Use, which propagate to derivatives and
gate redistribution. Do not assume this repository's licensing applies to them.
