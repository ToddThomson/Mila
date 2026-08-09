# Mila/Tools

Developer tooling. None of it ships: the wheel excludes it, and a consumer building Mila via
FetchContent never configures it.

| Directory | Language | What it does |
|---|---|---|
| `Converters/` | Python | HuggingFace weights and tokenizers to Mila format, per family |
| `ExportArtifact/` | C++ | Artifact, package and local-store lifecycle |
| `Publishing/` | Python | Uploads a prepared model card directory to the HuggingFace Hub |
| `Tokenize/` | C++ | Trains, encodes and decodes vocabularies |

## Build

`CMakeLists.txt` here adds the two C++ tools; the Python ones are run in place from a virtual
environment and are not part of any build.

`Tokenize` always configures. `ExportArtifact` configures only under `MILA_ENABLE_CUDA` — it reads
quantized weights back off the device, which needs a built, weights-loaded model.

Both are behind `PROJECT_IS_TOP_LEVEL` in `Mila/CMakeLists.txt`, so they build when Mila is the top
level project and not when it is a subproject.

## Converters

`convert_weights.py` and `convert_tokenizer.py` under `Gpt2/`, `Llama/` and `Gemma/`, over a shared
`MilaWeightWriter` in `common.py`. Converters always write BF16; quantization happens later, at load
time. Requires PyTorch and Transformers — see `Converters/README.md` for the interpreter constraint.

## ExportArtifact

Nine modes selected by flag. `ExportArtifact` with no arguments prints the full surface.

Producing an artifact:

- default — load a `.bin` with a quantization policy and write safetensors (GPU, loads the model)
- `--transcode` — rewrite a model file as safetensors tensor for tensor, no GPU and no numeric change

Packaging and the local store:

- `--package` — assemble a directory with a `mila.json` carrying real digests
- `--validate` — read every declared byte and report whether a package agrees with its manifest
- `--install` — install a package into the local model store
- `--rename` — rename an installed model, rewriting one record without moving bytes

Diagnostics:

- `--compare` — diff two artifacts
- `--fingerprint` — print a logits fingerprint for a fixed prompt, to diff two files that should
  hold the same model
- `--fetch` — pull one URL through Mila's own HTTP client and report byte count and digest. Takes a
  URL rather than a coordinate: what it exercises is the transport, below the level at which a hub
  knows anything. `--resume` continues from whatever the destination already holds, replaying it
  into the digest and sending a `Range` header, so the resume protocol the store uses for every
  large transfer can be driven on a small file:

```
ExportArtifact --fetch <lfs-url> probe.bin              # full copy, note the digest
# truncate probe.bin to any size
ExportArtifact --fetch <lfs-url> probe.bin --resume     # digest must come back identical
```

There is no upload here. Publishing is `Publishing/publish_model.py`, and the library itself never
uploads.

## Publishing

`publish_model.py <directory> [--repo <owner>/<name>] [--dry-run]` validates before it uploads and
verifies after, and is safe to re-run — anything already correct on the Hub is skipped. It takes
either a package directory built by `ExportArtifact --package`, or a card directory carrying a
`publish.json` that maps Hub paths to large files kept outside the repository.

`Publishing/README.md` is the process end to end: convert, quantize, card, package, install, publish.

## Tokenize

`tokenize <train|encode|decode|help> [options]`, over char and BPE tokenizers. `tokenize help`
prints the full option set.
