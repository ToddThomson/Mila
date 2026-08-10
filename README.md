# Mila

**A C++23 library for open LLMs at the metal — inference and training, built from explicit neural-network components you can read and understand.**

*Mila is a **craft project** — built for working at the metal: understanding exactly what every forward pass, gradient, and kernel does, which is also how you make them fast. Mastery is what that craft leads to.*

Mila is built for researchers, engineers, and developers who find high-level frameworks too opaque—who want to understand exactly what happens in every forward pass, trace every gradient,
and write kernels that do precisely what they intend. No autograd engine. No runtime
dispatch magic. Just C++23, CUDA, and full control.

> *Currently in public beta (`0.20.0-beta.2`) — feature-frozen and hardening toward the v0.20 first
> production release. Pre-1.0: the API is not yet stable.*
> *Active development lands on the [`dev`](https://github.com/ToddThomson/Mila/tree/dev) branch; `master` tracks tagged releases.*
> *See the [Roadmap](https://github.com/ToddThomson/Mila/blob/dev/ROADMAP.md) for current status and trajectory.*

---

| Branch | Build | Test | Docs |
|--------|-------|------|------|
| master | ![Build](https://github.com/ToddThomson/Mila/actions/workflows/build-pipeline.yml/badge.svg?branch=master&job=build) | ![Test](https://github.com/ToddThomson/Mila/actions/workflows/build-pipeline.yml/badge.svg?branch=master&job=test) | ![Docs](https://github.com/ToddThomson/Mila/actions/workflows/build-pipeline.yml/badge.svg?branch=master&job=docs) |
| dev    | ![Build](https://github.com/ToddThomson/Mila/actions/workflows/build-pipeline.yml/badge.svg?branch=dev&job=build) | ![Test](https://github.com/ToddThomson/Mila/actions/workflows/build-pipeline.yml/badge.svg?branch=dev&job=test) | ![Docs](https://github.com/ToddThomson/Mila/actions/workflows/build-pipeline.yml/badge.svg?branch=dev&job=docs) |

---

## What Mila Is

Mila is a component-based neural-network library for open LLMs, crafted so that **device and precision are chosen at compile
time, every forward and backward pass is explicit, and every gradient is yours to inspect**.

There is no hidden execution engine. When you call `forward()`, you know exactly what runs.
When you call `backward()`, you know exactly what accumulates. The architecture is designed
to be read, understood, extended, and challenged.

That philosophy fixes Mila's product shape: an **inference runtime library plus a small family of
adaptors**, distinguished by who closes the generation loop. The **Chat** harness closes it
in-process with a human in the gate; the **Mila Inference Server (MIS)** exports it over an
OpenAI/Anthropic-compatible wire so best-in-class foreign harnesses (Codex, Claude Code) can drive
Mila — and double as a ruthless validation oracle; a future **Agentic** adaptor will close the
loop on itself, on-device. Mila is a **library, not a framework**: your application owns `main()`,
the loop, and the tools — Mila makes the model an ordinary C++ object inside them. The full
positioning lives in
[MilaProductFamily.md](https://github.com/ToddThomson/Mila/blob/dev/Mila/Specifications/MilaProductFamily.md).

Mila does not try to run everything the way the industry standards do — llama.cpp and vLLM already
do that. It aims instead to *crest the wave*: to bring up the current best open models as they arrive
and put them within reach on home and edge hardware — a 12 GB card on your desk, not a rack in a
datacenter. A short, curated set, each one raised from the metal and held to parity or better. Not a
model zoo, and not second-best on the models it runs.

**This makes Mila well-suited for:**
- Researchers implementing novel architectures who need full visibility into compute
- Engineers studying training dynamics, gradient flow, or numerical precision
- Developers building custom CUDA kernels who want structured, modern C++ around them
- Anyone who learns best by reading code that does not hide what it does

---

## Design Principles

**Explicit over implicit.** Forward and backward passes are implemented manually per
component. Gradient flow is auditable by design, not by accident.

**Type safety at compile time.** Device type and precision are template parameters.
A CPU tensor and a CUDA tensor are different types. Mixing them is a compile error,
not a runtime surprise.

**Ownership is clear.** Every component owns its parameters and gradients. Composition
is explicit. There is no shared global state.

**C++23 throughout.** Modules, deducing-this, std::format, concepts — Mila is written
in modern C++ and intends to stay there. No header soup. Fast incremental builds with Ninja.

**CUDA-native.** Matrix operations via cuBLASLt. Hand-written kernels where control
matters. Vectorized memory access throughout — float4 for FP32, uint4 for BF16.

**Precision is deliberate.** BF16 is the primary reduced-precision compute target — it
matches FP32's exponent range, avoiding overflow and underflow without loss scaling, with
native Tensor Core support on Ada Lovelace and newer. FP16 is not a Mila target; BF16
supersedes it for all current use cases. Weight quantization is applied at model load time
as a pure compile-time decision via a `TWeightQuant` policy on `Linear` — no runtime
dispatch, no quantized checkpoint format. FP8 (`PerChannelFp8<>`) enables 8B-class models
within a 12 GB VRAM budget via per-channel BF16→FP8_E4M3 quantization with cuBLASLt
mixed-precision GEMM. FP4 E2M1 (`PerGroupFp4<>`) halves weight storage again — packed
nibbles dequantized per-group inline at inference time, forward-compatible with Blackwell
native FP4 compute when it becomes available.

---

## Model Families

Mila's validated targets, in priority order — the current best open models that fit home and edge
hardware.

### Gemma 4 12B — the flagship

Gemma 4 12B Instruct is Mila's most capable inference target and the chat CLI default. It runs the
full Gemma 4 architecture — per-layer sliding-window local/global attention, dual local/global RoPE,
GeGLU, RMSNorm, and final logit softcap — validated **token-for-token against HuggingFace**.

- **Fits a 12 GB consumer card at FP4**, with a large context window: weight-tying reclaims ~2 GB and
  a bounded-KV sliding-window ring cache holds persistent KV growth to the global layers, so long
  chats stay inside the budget.
- **Tensor-core flash-attention prefill** — a warp-tiled flash-attention kernel on the sliding local
  layers. Prefill lands within 1.14x of llama.cpp at long context.
- **On-device sampling** — logits never leave the GPU; only the sampled token is read back, at
  49 tok/s FP4 decode at 32K context on a 12 GB card — within 1.03x of llama.cpp.
- **Native tool calling** — Gemma's trained tool grammar, reconciled byte-for-byte to Google's
  canonical chat template, driven end-to-end through MIS by foreign harnesses (Codex CLI, Claude Code)
  across plain-chat, single-tool, and tool-result-resume flows.

### Llama 3.x

Mila's primary validated inference lineage — Llama 3.2 1B, 3.2 3B, and 3.1 8B — built from RMSNorm,
SwiGLU, Grouped Query Attention, and RoPE, with SentencePiece tokenization and HuggingFace weight
conversion. Each is validated **token-for-token against HuggingFace**: 1B at FP32, 3B at BF16, and the
quantized paths (FP8 E4M3 per-channel, FP4 E2M1 per-group) against that baseline. Llama 3.1 8B at FP4
(~6 GB) is the small-footprint workhorse — it fits a 12 GB card with room to spare — with FP8 as the
finer-precision alternative. Tool calling is validated on Llama 3.2 3B Instruct.

### GPT-2 — where it started

GPT-2 is where Mila began — built from scratch in the spirit of Karpathy's [`llm.c`](https://github.com/karpathy/llm.c)
and its ethos of *understanding a model by building it*. Mila took that in its own direction — a
component-based C++23 library rather than a single file, but the same conviction that a language model
should be readable all the way to the metal. The full stack — BPE tokenizer, learned positional embeddings, multi-head attention,
MLP, and KV-cache — is validated **token-for-token against HuggingFace** (greedy and sampled). It
remains Mila's training reference — the Bard sample trains a GPT-2 from scratch — and the simplest
place to read one token's journey end to end.

---

## Current Status — Beta.2 (feature-frozen, hardening)

Mila is in public beta, hardening toward a craft-complete first release (v0.20). The alpha
phase built and validated the core architecture against known-good reference implementations; the
feature set is now **frozen**, and the remaining work is validation, packaging, documentation, and
recovering the full GPT-2 / training foundation — so the first release ships everything Mila has
built, inference and training, as one coherent, tested, documented package.

**Hardening through beta (Production Hardening)**
Feature-frozen: validation, packaging, and documentation only. **Model distribution is the one
deliberate carve-in**, landed in `beta.2` — a release nobody can get a model for is not an onboarding
story, and it was an alpha omission rather than a new idea. The v0.20 workstreams still in flight
are test-suite revival, training revival (the MNIST and Bard GPT-2 samples re-aligned to the current
API — Llama 3.1/3.2 training is not part of this release), API documentation, and production
hardening itself, from which the `beta.X` and `rc.X` tags are cut. See
[RELEASING.md](https://github.com/ToddThomson/Mila/blob/dev/RELEASING.md) for how stages and tags
relate.

See [ROADMAP.md](https://github.com/ToddThomson/Mila/blob/dev/ROADMAP.md) for the full roadmap,
including what comes after v0.20.

---

## Validated Capabilities

The complete validated surface — the model paths from **Model Families** above, plus the components,
tokenizers, and tooling beneath them.

| Capability | Status |
|---|---|
| Gemma 4 12B Instruct inference — greedy decode | Validated against HuggingFace (token-for-token) |
| Gemma 4 12B Instruct — FP4 E2M1 per-group quantization | Validated — chat CLI default; runs a large context window in 12 GB (weight-tying + bounded-KV ring) |
| Llama 3.1 8B inference — FP4 E2M1 per-group quantization | Validated — ~6 GB, ~57 tok/s decode, fits 12 GB |
| Llama 3.1 8B inference — FP8 E4M3 per-channel quantization | Validated — ~11.6 GB at ctx 8192 |
| Llama 3.2 3B inference — FP4 E2M1 per-group quantization | Validated — coherent generation, 44–48 tok/s decode |
| Llama 3.2 3B inference — FP8 E4M3 per-channel quantization | Validated — coherent generation, ~41 tok/s decode |
| Llama 3.2 3B inference — greedy decode at BF16 | Validated against HuggingFace |
| Llama 3.2 1B inference — greedy decode at FP32 | Validated against HuggingFace |
| GPT-2 inference — greedy and sampled | Validated against HuggingFace |
| Two-phase KV-cache — prefill + decode | Complete |
| HuggingFace Gemma weight converter | Complete |
| HuggingFace Llama weight converter | Complete |
| HuggingFace GPT-2 weight converter | Complete |
| Model store — install, list, remove; shared by Chat and MIS as separate processes | Complete |
| Model retrieval from HuggingFace — digest-verified pull | Validated — pulled independently on Windows (C++) and Linux (Python), byte-identical blobs |
| Published models — Gemma 4 12B FP4, Llama 3.1 8B FP4, Llama 3.2 3B FP4 (`mila-llm`) | Complete — ungated; licence and notice travel with the weights |
| Model packaging and publishing — manifest, package, publish | Complete |
| Instruction following — Llama 3.2 3B Instruct | Validated |
| Tool calling framework | Complete |
| Chat CLI | Complete |
| Mila Inference Server (MIS) — OpenAI/Anthropic wire | Validated — Codex and Claude Code CLI round-trips on Gemma 4 12B FP4 |
| MNIST training — ~97.9% test accuracy | Complete |
| AdamW optimizer | Complete |
| cuBLASLt Linear — forward + backward | Complete |
| LayerNorm, RMSNorm, GELU, SiLU, Softmax | Complete |
| SwiGLU MLP — forward + CUDA kernel | Complete |
| Multi-Head Attention — forward + backward | Complete |
| Grouped Query Attention — GQA with KV-cache | Complete |
| Sliding-window attention — per-layer local/global, dual RoPE (Gemma 4) | Complete |
| GeGLU FFN (Gemma 4) | Complete |
| Final logit softcap (Gemma 4) | Complete |
| RoPE — rotary positional encoding | Complete |
| BPE tokenizer | Complete |
| SentencePiece tokenizer | Complete |

---

## Adaptors

The runtime plus a small family of adaptors, each closing the generation loop for a
different consumer. See [Mila/Adaptors/README.md](https://github.com/ToddThomson/Mila/blob/dev/Mila/Adaptors/README.md) and the full
positioning in [MilaProductFamily.md](https://github.com/ToddThomson/Mila/blob/dev/Mila/Specifications/MilaProductFamily.md).

### Chat — a human at a prompt

```
You: In one sentence, what is a KV-cache?
Mila: It stores the key and value tensors from earlier tokens so each new token attends
      over them instead of recomputing the whole sequence each step.
```

Located under `Mila/Adaptors/Chat`. An instruction-following chat harness that closes the
loop in-process with a human in the gate — the default model is Gemma 4 12B Instruct at FP4,
loaded via the two-phase (prefill + decode) KV-cache pipeline, with model hot-switching
(`/model <name> [quant]`) and tool calling. Models come from the local store: `/models --online`
lists what Mila publishes, `/install <name>` downloads one, and `/models` shows what is installed and
what each costs in memory. On a 12 GB card, Gemma 4 12B FP4 runs a large context
window — its two memory-fit gates, weight-tying and the bounded-KV sliding-window ring cache, landed
in the alpha.6 line.

### Mila Inference Server (MIS) — a foreign harness over the wire

Located under `Mila/Adaptors/Inference`. Exports the generation loop over an
OpenAI/Anthropic-compatible wire (a pybind11 bridge plus a Python server) so a best-in-class
harness you did not write — Codex, Claude Code — can drive Mila from another process, and
double as a ruthless validation oracle.

---

## Samples

### MNIST Classifier

Located under `Mila/Samples/MNIST`. Trains a 3-layer MLP on MNIST to ~97.9% test accuracy.
Demonstrates the full training loop: data loading, forward pass, loss, backward pass, AdamW step.

### Bard — GPT-2 character-level training

Located under `Mila/Samples/Bard`. Trains a small GPT-2-style transformer on Tiny Shakespeare to
coherent, Shakespeare-structured text — the transformer counterpart to MNIST's MLP, revived to the
current API as part of v0.20 Training Revival.

### QuickStart — consume Mila via FetchContent

Located under `Mila/Samples/QuickStart`. A standalone downstream project showing how to depend on Mila
with `FetchContent` (the supported consumption path for a C++23 module library) and call its public
API. See [getting-started.md §7](https://github.com/ToddThomson/Mila/blob/dev/getting-started.md) and the sample's
[README](https://github.com/ToddThomson/Mila/blob/dev/Mila/Samples/QuickStart/README.md).

---

## Build

### Prerequisites

| Requirement | Version |
|---|---|
| C++ compiler | MSVC (Visual Studio 2026 18.6.2+) on Windows, or Clang 19+ / GCC 16 on Linux |
| CUDA Toolkit | 13.0+ on Windows; 13.3+ on Linux (Ubuntu 26.04 / glibc 2.43) |
| CMake | 4.0 or newer |
| Git | 2.x or newer (validated on 2.54.0) |
| GTest | 1.17.0 |
| Doxygen + Graphviz | latest (optional — docs only) |
| C++ Standard | C++23 |

Ninja is the recommended generator — significantly faster than MSBuild for
incremental C++23 module builds.

Mila is CI-tested on CUDA 13.0 and developed on 13.3; newer 13.x releases are expected
to work but are not exhaustively validated. On Linux (Ubuntu 26.04 / glibc 2.43), CUDA 13.3
is required — 13.0 fails to build there.

On Windows, use Visual Studio 2026 18.6.2 or newer — earlier 2026 builds have a regression
that breaks the C++23 module build. On Linux, the C++23 modules build with Clang 19+ or GCC 16;
on Ubuntu 26.04 install the `gcc-16` package (GCC 15.2 and earlier cannot compile the modules).

Git must be installed and on `PATH`: the first CMake configure fetches dependencies via CPM
(`git clone`), so it is needed beyond the initial repository clone. GitHub Desktop is an
optional convenience, not a requirement.

Building the API docs is optional — enable it with `-DMILA_ENABLE_DOCS=ON` (default
`OFF`), which requires Doxygen (and Graphviz for the call graphs). A normal
library/test build needs neither.

### Quick Start

```bash
git clone https://github.com/toddthomson/mila.git
cd mila
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DMILA_ENABLE_TESTING=ON
cmake --build build
ctest --test-dir build
```

Tests are opt-in (`MILA_ENABLE_TESTING` defaults to `OFF`); omit the flag for a
library-only build.

### Visual Studio

Open the repository folder — Visual Studio detects CMakeLists.txt automatically.
Select the Ninja generator and Release configuration. Build with F7.

### Linux (native / WSL)

On Linux — including WSL 2 — build with Clang against the bundled CMake presets. Requires
Clang 19+ (or GCC 16) and CUDA 13.3+:

```bash
cmake --preset linux-clang-release
cmake --build out/build/linux-clang-release
ctest --test-dir out/build/linux-clang-release
```

`linux-clang-debug` is the Debug equivalent. The `linux-clang-cpu-debug`/`-release` presets
build the CPU-only configuration (`MILA_ENABLE_CUDA=OFF`) — the same path the CI test ratchet
exercises, requiring no CUDA toolkit.

### Docker

A development container provides a reproducible Linux build toolchain (CUDA 13.3,
Clang 21 with a gcc-15 nvcc host, CMake 4.x, Ninja) — the simplest way to build Mila without
installing the toolchain locally, for example from WSL. It mounts the repo at `/mila` with GPU access.

```bash
# Build and start the dev container (requires the NVIDIA Container Toolkit for GPU access)
docker compose -f Docker/docker-compose.yml run --rm mila-dev

# Inside the container:
cmake -S . -B out/build/linux-release -G Ninja -DCMAKE_BUILD_TYPE=Release -DMILA_ENABLE_TESTING=ON
cmake --build out/build/linux-release
ctest --test-dir out/build/linux-release
```

VS Code users can instead **Reopen in Container** — see `.devcontainer/`.

Model weights are not included. The image sets `MILA_CACHE_DIR=/mila/Data/Models/Store`, which sits
on the repo bind mount, so a model installed with `/install` survives `run --rm` and is the same
store the host uses — install it once from either side.

> A slim, published runtime image — `docker run … mila` for users who only want to run
> inference without building — is planned for the v0.20 release. See [ROADMAP.md](https://github.com/ToddThomson/Mila/blob/dev/ROADMAP.md).

---

## Documentation

Site: https://mila.toddt.me — including the
[blog](https://mila.toddt.me/blog/) on the CUDA kernels and architecture work.

API reference: https://mila.toddt.me/api/

Both are rebuilt automatically on every push to `dev`, so the API reference tracks the code rather
than the last release.

---

## Contributing

Mila is in public beta (feature-frozen, hardening toward v0.20) and welcomes contributors who share its philosophy.
Good starting points are CPU reference ops, test coverage, and new encoding strategies
under /Components/Encodings/. Mila is GPU-first by design: the CUDA backend is the
validated inference path, and CPU op coverage beyond the GPT-2 lineage is intentionally
demand-driven — implementing a CPU op for Llama (RmsNorm, SwiGLU, RoPE, token embedding)
is a well-scoped, self-contained first contribution, not a gap to apologize for.

1. Fork the repository and create a branch from dev
2. Make changes with clear, focused commits
3. Ensure new components include forward and backward pass tests
4. Open a pull request targeting dev

New contributors: [getting-started.md](https://github.com/ToddThomson/Mila/blob/dev/getting-started.md) walks through a fresh clone,
build, running inference, and opening your first PR.
See CONTRIBUTING.md for coding standards and the pull request process.

---

## Attributions

For research and open-source acknowledgements, see [ATTRIBUTIONS.md](https://github.com/ToddThomson/Mila/blob/dev/ATTRIBUTIONS.md).

---

## License

MIT License — see [License.md](https://github.com/ToddThomson/Mila/blob/dev/License.md) for details.
