# Mila

Model artifacts for [Mila](https://github.com/ToddThomson/Mila) — a C++23 and CUDA reference
implementation of LLM inference. Mila has no execution engine: a model is ordinary C++ objects you
compose and read, not a graph fed a configuration file.

**These artifacts load only with Mila.** They are deliberately not NVFP4 (group 16, FP8 scales) or
MXFP4 (group 32, UE8M0 scales), so `transformers`, vLLM and llama.cpp cannot consume them. The files
are ordinary safetensors — shapes, dtypes and metadata open in any safetensors reader — but loading
one as a model requires Mila. If you came here looking for a quantized checkpoint for another
runtime, this is not it, and the upstream repository each artifact names is where to go.

## What an artifact is

A model that has already been quantized, so a session starts by mapping weights rather than by
converting billions of parameters first.

| File | What it is |
|---|---|
| `*.safetensors` | Weights: FP4 E2M1, two values per byte, per-group FP32 absmax scales as sibling `<name>.weight_scale` tensors. Norms stay BF16. |
| `*_tokenizer.bin` | The tokenizer, in Mila's format |
| `mila.json` | The manifest: every file's SHA-256 and size, the architecture, the quantization, and the minimum Mila version |
| `LICENSE`, `NOTICE` | The source model's terms, and any attribution its license requires be retained |

Every file is declared in `mila.json`, and installing verifies each one against its digest before it
enters a content-addressed local store. The license travels with the weights rather than staying
behind on this page.

## Using one

From the Mila chat harness:

```
/install Llama-3.2-3B-Instruct-fp4
/model Llama-3.2-3B-Instruct-fp4
```

Installing is a deliberate act and the only step that touches the network. Loading reads the local
store and nothing else — there is no implicit download, and Mila's inference server refuses to pull
at all.

From the library:

```cpp
ModelStore store;

const auto hub = makeDefaultModelHub();
ModelResolver resolver( store, *hub );
resolver.pull( "Llama-3.2-3B-Instruct-fp4", std::string( kDefaultHubOwner ) );

const auto model = store.locate( "Llama-3.2-3B-Instruct-fp4" );
```

`mila.json` declares `minimum_mila_version`. Mila is in public beta, and a build older than an
artifact's floor refuses it by name rather than failing somewhere inside the load.

## Reading a name

A name is the upstream model plus what Mila did to it:

```
Llama-3.2-3B-Instruct-fp4      gemma-4-12b-it-fp4
^-- as its publisher writes it  ^-- lowercase suffix, always Mila's
```

There is no house style, deliberately. The name mirrors each family's own convention — Meta's
capitalization for Llama, Google's for Gemma — because one imposed style would have to contradict
one upstream. The quantization suffix is Mila's and stays lowercase whatever the name does. One name
is one model: quantization is part of the identity, not a runtime option.

## Licensing

Per family, and not interchangeable.

**Gemma 4** is Apache 2.0 — public and ungated. Quantization is a modification, and the artifact
says so.

**Llama 3.1 and 3.2.** **Built with Llama.** Llama is licensed under the Llama Community License,
Copyright © Meta Platforms, Inc. All Rights Reserved. Each artifact ships the agreement that matches
its own base model — 3.1 and 3.2 are separate agreements, with different attribution and different
Acceptable Use Policies, and a 3.2 derivative does not inherit 3.1's. Use is subject to the Llama
Acceptable Use Policy, incorporated into the agreement by reference.

Quantization is lossy. These builds trade some accuracy for fitting a large model on a consumer
card; if you need the weights untouched, convert them yourself from the upstream repository with the
converters in the Mila repository.

None of these repositories are affiliated with or endorsed by Meta or Google.

---

[mila.toddt.me](https://mila.toddt.me) · [github.com/ToddThomson/Mila](https://github.com/ToddThomson/Mila)
