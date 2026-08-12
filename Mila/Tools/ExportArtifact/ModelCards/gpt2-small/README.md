---
license: mit
base_model: openai-community/gpt2
tags:
  - mila
  - gpt2
  - reference
library_name: mila
---

# GPT-2 Small — FP32 for Mila

A [Mila](https://github.com/ToddThomson/Mila) artifact of `openai-community/gpt2`, in safetensors
format, at the precision the weights were released in.

**622 MiB.** It is the reference model in Mila's catalogue rather than a deployment target: the
smallest thing that exercises the whole path end to end, and the only published artifact that is
not quantized.

## What this is for

GPT-2 is a **base model** — it continues a prompt. It was not instruction-tuned, so it reads no
chat template, holds no turns, and calls no tools. Three consequences worth stating up front,
because each of them is a design decision rather than a defect:

- **The Mila chat harness refuses it**, keyed on the manifest's `instruct: false`. Chat renders
  turns and applies a template; a base model uses none of that.
- **The Mila Inference Server does not serve it.** MIS speaks the OpenAI and Anthropic chat
  protocols, which are instruct-shaped.
- **Completion is the mode it belongs to**, along with the training path — Mila's `Bard` sample
  trains a GPT-2 stack from scratch.

It is also the artifact to reach for when what you are testing is Mila rather than the model:
622 MiB resolves, pulls, verifies, adopts and loads in the time a 12B spends on its first file.

## Files

| File | Purpose |
|---|---|
| `gpt2_small_fp32.safetensors` | Weights: FP32, unquantized |
| `gpt2_tokenizer.bin` | Mila tokenizer (BPE, 50257 entries) |
| `mila.json` | Manifest: file digests, quantization, minimum Mila version |
| `LICENSE` | OpenAI's Modified MIT License for GPT-2 |

## Use

From the Mila chat harness — `/install` works, `/model` will refuse it, and that refusal is the
documented behaviour above:

```
/install gpt2-small
```

From the library:

```cpp
ModelStore store;

// Pull once. This is the only thing here that touches the network.
const auto hub = makeDefaultModelHub();
ModelResolver resolver( store, *hub );
resolver.pull( "gpt2-small", std::string( kDefaultHubOwner ) );

// Load from the store thereafter -- no network, and no manifest fetch.
const auto model = store.locate( "gpt2-small" );
```

Installing is a deliberate step, and it is the only one that touches the network. It verifies each
file against the digest in `mila.json` and leaves it in a content-addressed local store; every load
afterwards reads the store and nothing else.

No token is required — this repository is public and ungated.

## Precision

FP32, exactly as released. There is no quantized build here, and the name carries no precision
suffix for that reason: the other artifacts in this catalogue are pre-quantized deployment builds,
and this one is not.

FP32 is also what Mila's training path is validated against, which is the second reason to leave it
alone. A reader comparing Mila's arithmetic against a reference wants the reference unmodified.

## Modifications from the base model

- Repacked from the HuggingFace checkpoint into Mila's tensor layout, then into safetensors
- Tokenizer converted to Mila's format
- No quantization, fine-tuning, distillation, or other change to what the model learned

The file opens in any safetensors reader — shapes, dtypes and metadata are all inspectable — but
loading it as a model requires Mila.

## License and attribution

GPT-2 is released under OpenAI's **Modified MIT License**, Software Copyright © 2019 OpenAI. The
full text is in `LICENSE` and travels with every copy, as that license requires.

The modification from stock MIT is a carve-out in the model's favour: the copyright and permission
notices "need not be included with content created by the Software." OpenAI asks that GPT-2 be used
responsibly and that generated content be clearly indicated as such.

Base model: `openai-community/gpt2`, © OpenAI. This repository is not affiliated with or endorsed
by OpenAI.
