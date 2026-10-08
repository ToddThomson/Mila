# Mila/Tools

Developer tooling. None of it ships: the wheel excludes it, and a consumer building Mila via
FetchContent never configures it.

| Directory | Language | What it does |
|---|---|---|
| `Cli/` | C++ | The `mila` command: the local model store and the server front door |
| `ContextProfile/` | C++ | What a model configuration is worth to an agent at each context length it fits |
| `Converters/` | Python | HuggingFace weights and tokenizers to Mila format, per family |
| `Drafting/` | C++ | Whether a draft model pays on its target: what a verify costs, and how many drafts are accepted |
| `Evaluation/` | Python | Standard benchmarks (IFEval, GSM8K, RULER, BFCL) on Mila, transformers and llama.cpp from the same token ids, compared by document; Hugging Face Hub benchmarks (MMLU-Pro) run to their own definition and written as the model repo's result |
| `ExportArtifact/` | C++ | Artifact, package and local-store lifecycle |
| `Measurement/` | C++ | Header-only harness: a network built from weights, scored teacher-forced; PG-19 books |
| `Publishing/` | Python | Uploads a model package to the HuggingFace Hub |
| `Quantization/` | Python | Fits, encodes and gates a sub-4-bit weight quantization scheme |
| `Tokenize/` | C++ | Trains, encodes and decodes vocabularies |

## Build

`CMakeLists.txt` here adds the C++ tools; the Python ones are run in place from a virtual
environment and are not part of any build.

`Tokenize` always configures. `ExportArtifact` configures only under `MILA_ENABLE_CUDA` — it reads
quantized weights back off the device, which needs a built, weights-loaded model.

Both are behind `PROJECT_IS_TOP_LEVEL` in `Mila/CMakeLists.txt`, so they build when Mila is the top
level project and not when it is a subproject.

`Measurement` is added from `Mila/CMakeLists.txt` directly, whenever the tools or the tests are, since
the tests link it and build without the tools. Include it after `import Mila;`.

## Converters

`convert_weights.py` and `convert_tokenizer.py` under `Gpt2/`, `Llama/` and `Gemma/`, over a shared
`MilaWeightWriter` in `common.py`. `Gemma/convert_drafter.py` converts Google's Gemma 4 draft model into a file of
its own, `drafter.*` (`Specifications/Gemma4Mtp.md`). Converters always write BF16; quantization is a separate offline
step — `Quantization/` for the sub-4-bit formats, `ExportArtifact` for FP8 and FP4. Requires PyTorch
and Transformers — see `Converters/README.md` for the interpreter constraint.
`Gemma/gemma_4_26b_moe/hf_gemma_router_reference.py` captures the HuggingFace router reference that
`Specifications/Notebooks/Gemma4MoE.md` Phase 5 gates against; it reads the router tensors by byte range and
needs no checkpoint download. Beside it, `hf_gemma_experts_reference.py` captures the Phase 6
expert-bank reference from `Gemma4TextExperts` on seeded synthetic weights, offline, and
`hf_gemma_moe_model_reference.py` captures the Phase 8 wiring reference: a tiny random Gemma 4 MoE
model, its logits, and its own checkpoint converted by `Gemma/convert_weights.py` at FP32 and BF16.
`hf_gemma_layer_stream.py` is the HuggingFace half of the 26B BF16 parity gate: it runs the real
checkpoint one decoder layer at a time and writes each layer's last-token hidden state and the
last-position logits. Run `--self-test` first — it proves the streamed driver bitwise against the whole
model, with two negative controls.

## Quantization

`quality_gate.py` drives four modules: `formats.py` (the level sets, grouping and codebook fitting),
`fit.py` (activation calibration and sequential GPTQ), `artifact.py` (Mila-named safetensors
emission) and `evaluate.py` (perplexity and greedy agreement against a BF16 reference).
`packing.py` is the packed-layout codec, held to `Src/Dnn/Quantization/Weight/CodebookPacking.ixx`
by a generated fixture the C++ oracle test bit-matches. Shares the `Converters/` virtual environment.

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

## ContextProfile

`ContextProfile list` names the configurations; `ContextProfile run <configuration>` profiles one on the card
it runs on and writes `<configuration>.json` and a Markdown rendering beside it. Phase 1 of
`Specifications/ContextProfile.md`: fit at each band (the planner, nothing allocated), loss by band (G2's
protocol on PG-19), and recall at depth (records planted in a conversation of tool results, each asked
for from the end of it). `--bands`, `--arms`, `--books` and `--conversations` narrow a run; with no
`--bands` it profiles 16K, 32K, 64K, 128K and the planner's own choice, every one that fits. CUDA-only.

## Drafting

`Drafting verify-cost` times what checking K drafted tokens costs Gemma 4 12B at Q4_0, against one decode, at
each `--depths` (default 8192 and 65536), for K from 1 to `--max-draft`. The decode is the replayed step a model
runs; the verify is the prefill path from the same rewound position, with the head on the last row and on every
row. `Drafting drafter-parity` runs Google's draft model from the 12B's state after a PG-19 prompt and dumps each
step's inputs, both caches and its logits; `Converters/Gemma/hf_gemma_drafter_reference.py --dump <dir>` runs
HuggingFace's drafter on the same inputs and gates the two. Stage 1 of `Specifications/Gemma4Mtp.md`, section 5.1.
CUDA-only.

## Evaluation

Paired benchmark evaluations: one engine against another on the same token ids, compared by document.
`run_arm.py hf|mila|llamacpp --output <run>` runs lm-eval tasks (IFEval, GSM8K, RULER with `--per-band`) on
HuggingFace transformers, MIS or `llama-server`; `run_bfcl.py` runs BFCL against a server, with `reference_server.py`
as its transformers arm. `compare_arms.py <run>/hf <run>/mila` pairs the two by document and reports the score
difference with its interval, the answers each arm alone got right, and how often the replies match exactly. Its
first use is Mila BF16 against transformers on Llama 3.2 3B, which is the noise floor a quantized build is read
against. `run_inspect.py` runs a benchmark registered on the Hugging Face Hub (MMLU-Pro) through inspect-ai, to the
benchmark's own `eval.yaml`, and `eval_results.py` writes the mila arm's score as the model repo's
`.eval_results/<task>.yaml`, which the Hub shows on the model page and in the benchmark's leaderboard.
`Evaluation/README.md` covers setup and the reading of a report; `Specifications/ModelEval.md` is the design.

## Publishing

`publish_model.py <package-dir> --repo <owner>/<name> [--dry-run]` validates before it uploads and
verifies after, and is safe to re-run — anything already correct on the Hub is skipped. It takes a
package directory built by `ExportArtifact --package`, and nothing else.

`Publishing/README.md` is the process end to end: convert, quantize, card, package, install, publish.

## Tokenize

`tokenize <train|encode|decode|help> [options]`, over char and BPE tokenizers. `tokenize help`
prints the full option set.
