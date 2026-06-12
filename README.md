# Mila

**A C++23 module-based deep neural network library for those who want full control&mdash;to work at the metal.**

*Mila is a **craft project** — built for working at the metal: understanding exactly what every forward pass, gradient, and kernel does, which is also how you make them fast. Mastery is what that craft leads to.*

Mila is built for researchers, engineers, and developers who find high-level frameworks too opaque—who want to understand exactly what happens in every forward pass, trace every gradient,
and write kernels that do precisely what they intend. No autograd engine. No runtime
dispatch magic. Just C++23, CUDA, and full control.

> *Currently in active alpha development. API is not yet stable.*
> *Active development lands on the [`dev`](https://github.com/ToddThomson/Mila/tree/dev) branch; `master` tracks tagged releases.*
> *See the [Roadmap](https://github.com/ToddThomson/Mila/blob/dev/ROADMAP.md) for current status and trajectory.*

---

| Branch | Build | Test | Docs |
|--------|-------|------|------|
| master | ![Build](https://github.com/ToddThomson/Mila/actions/workflows/build-pipeline.yml/badge.svg?branch=master&job=build) | ![Test](https://github.com/ToddThomson/Mila/actions/workflows/build-pipeline.yml/badge.svg?branch=master&job=test) | ![Docs](https://github.com/ToddThomson/Mila/actions/workflows/build-pipeline.yml/badge.svg?branch=master&job=docs) |
| dev    | ![Build](https://github.com/ToddThomson/Mila/actions/workflows/build-pipeline.yml/badge.svg?branch=dev&job=build) | ![Test](https://github.com/ToddThomson/Mila/actions/workflows/build-pipeline.yml/badge.svg?branch=dev&job=test) | ![Docs](https://github.com/ToddThomson/Mila/actions/workflows/build-pipeline.yml/badge.svg?branch=dev&job=docs) |

---

## What Mila Is

Mila is a component-based DNN library, crafted so that **device and precision are chosen at compile
time, every forward and backward pass is explicit, and every gradient is yours to inspect**.

There is no hidden execution engine. When you call `forward()`, you know exactly what runs.
When you call `backward()`, you know exactly what accumulates. The architecture is designed
to be read, understood, extended, and challenged.

**This makes Mila well-suited for:**
- Researchers implementing novel architectures who need full visibility into compute
- Engineers studying training dynamics, gradient flow, or numerical precision
- Developers building custom CUDA kernels who want a structured C++ framework around them
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

## Current Status — Alpha.6

Mila is under active development toward a public, craft-complete first release (v0.20). The alpha
phase built and validated the core architecture against known-good reference implementations; the
current focus is consolidating that work and recovering the full GPT-2 / training foundation, so the
first release ships everything Mila has built — inference and training — as one coherent, tested,
documented package.

**Alpha.1 — Complete**
GPT-2 inference validated token-for-token against HuggingFace using greedy decoding.
The full GPT-2 stack — tokenizer, embeddings, attention, MLP, KV-cache — is implemented,
tested, and confirmed correct.

**Alpha.2 — Complete**
Llama architecture validated token-for-token against HuggingFace at FP32. RoPE, RMSNorm,
SwiGLU, and Grouped Query Attention are implemented and confirmed correct. The full
LlamaModel stack — including SentencePiece tokenization and HuggingFace weight conversion
— matches HuggingFace LlamaForCausalLM token-for-token on greedy decode.

**Alpha.3 — Complete**
BF16 compute backend validated token-for-token against HuggingFace. Greedy decode of
Llama 3.2 3B matches HuggingFace LlamaForCausalLM at BF16 using the same methodology
applied to FP32.

**Alpha.4 — Complete**
Instruction following and tool calling, validated on Llama 3.2 3B Instruct at BF16.
Delivers the structured message and tool calling infrastructure in the Chat application
layer. No model architecture changes required.

**Alpha.5 — Complete**
FP8 and FP4 E2M1 load-time weight quantization, validated on Llama 3.2 3B and Llama 3.1 8B
Instruct. Weights are quantized from BF16 at model load time inside `Linear` via a compile-time
`TWeightQuant` policy — no quantized checkpoint format required. FP8 uses per-channel
absmax scaling with cuBLASLt mixed-precision GEMM; FP4 E2M1 uses per-group absmax scaling
with a dedicated decode matvec kernel. Llama 3.1 8B at FP4 (~6 GB) is the production default,
fitting comfortably within a 12 GB VRAM budget; FP8 is the validated finer-precision
alternative. Both paths are validated against the existing BF16 baseline.

Alpha.5 also introduces compile-time operation dispatch via `OperationTraits<OperationType,
TDeviceType, TPrecision, TPolicy>`. `Linear` is the reference implementation — a missing
specialization is a compile error, not a registry miss. All remaining components migrate
to `OperationTraits` dispatch as part of this alpha. The component type system
(`ComponentType`, `OperationType`) has been audited for completeness and consistency
across all leaf components.

**Alpha.6 — In Progress (Consolidation)**
Feature freeze and debt burndown to earn the beta label, plus the architectural foundation for
recovering the parked training path. The first release (v0.20) then proceeds through Alpha.7
(test-suite revival), Alpha.8 (training revival — the GPT-2 / MLP samples MNIST and Bard re-aligned
to the current API; Llama 3.1/3.2 training is not part of this release), and Alpha.9 (API
documentation), before Beta.1 production hardening.

**vNext — Qwen 3**
Qwen 3 transformer architecture with thinking mode and model-agnostic tool calling, validated on
Qwen 3 8B Instruct at BF16 and FP8, with FP8 KV cache compression. Follows the v0.20 release.

See [ROADMAP.md](https://github.com/ToddThomson/Mila/blob/dev/ROADMAP.md) for the full milestone breakdown.

---

## Validated Capabilities

| Capability | Status |
|---|---|
| GPT-2 inference — greedy and sampled | Validated against HuggingFace |
| Llama 3.2 1B inference — greedy decode at FP32 | Validated against HuggingFace |
| Llama 3.2 3B inference — greedy decode at BF16 | Validated against HuggingFace |
| Llama 3.2 3B inference — FP8 E4M3 per-channel quantization | Validated — coherent generation, ~41 tok/s decode |
| Llama 3.2 3B inference — FP4 E2M1 per-group quantization | Validated — coherent generation, 44–48 tok/s decode |
| Llama 3.1 8B inference — FP8 E4M3 per-channel quantization | Validated — fits 12 GB VRAM budget, ~11.6 GB at ctx 8192 |
| Llama 3.1 8B inference — FP4 E2M1 per-group quantization | Validated — production default, ~6 GB, ~57 tok/s decode |
| Two-phase KV-cache — prefill + decode | Complete |
| HuggingFace GPT-2 weight converter | Complete |
| HuggingFace Llama weight converter | Complete |
| Instruction following — Llama 3.2 3B Instruct | Validated |
| Tool calling framework | Complete |
| Chat CLI | Complete |
| MNIST training — 97.5% test accuracy | Complete |
| AdamW optimizer | Complete |
| cuBLASLt Linear — forward + backward | Complete |
| LayerNorm, RMSNorm, GELU, SiLU, Softmax, CrossEntropy | Complete |
| SwiGLU MLP — forward + CUDA kernel | Complete |
| Multi-Head Attention — forward + backward | Complete |
| Grouped Query Attention — GQA with KV-cache | Complete |
| RoPE — rotary positional encoding | Complete |
| BPE tokenizer | Complete |
| SentencePiece tokenizer | Complete |

---

## Samples

### Chat CLI

```
You: In one sentence, what is a KV-cache?
Mila: It stores the key and value tensors from earlier tokens so each new token attends
      over them instead of recomputing the whole sequence each step.
```

Located under `Samples/Chat`. An instruction-following chat harness — the default model is
Llama 3.1 8B Instruct at FP4, loaded via the two-phase (prefill + decode) KV-cache pipeline,
with model hot-switching (`/model <alias> [quant]`) and tool calling.

### MNIST Classifier

Located under `Samples/Mnist`. Trains a 3-layer MLP on MNIST to 97.5% test accuracy.
Demonstrates the full training loop: data loading, forward pass, loss, backward pass, AdamW step.

---

## Build

### Prerequisites

| Requirement | Version |
|---|---|
| Visual Studio | 2026 18.6.2 or newer |
| Git | 2.x or newer (validated on 2.54.0) |
| CUDA Toolkit | 13.0 or newer |
| CMake | 4.0 or newer |
| GTest | 1.17.0 |
| Doxygen + Graphviz | latest (optional — docs only) |
| C++ Standard | C++23 |

Ninja is the recommended generator — significantly faster than MSBuild for
incremental C++23 module builds.

Mila is CI-tested on CUDA 13.0 and developed on 13.3; newer 13.x releases are expected
to work but are not exhaustively validated.

Use Visual Studio 2026 18.6.2 or newer — earlier 2026 builds have a regression that breaks
the C++23 module build.

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

### Docker

A development container provides a reproducible Linux build toolchain (CUDA 13.0,
Clang 19, CMake 4.x, Ninja) — the simplest way to build Mila without installing the
toolchain locally, for example from WSL. It mounts the repo at `/mila` with GPU access.

```bash
# Build and start the dev container (requires the NVIDIA Container Toolkit for GPU access)
docker compose -f Docker/docker-compose.yml run --rm mila-dev

# Inside the container:
cmake -S . -B out/build/linux-release -G Ninja -DCMAKE_BUILD_TYPE=Release -DMILA_ENABLE_TESTING=ON
cmake --build out/build/linux-release
ctest --test-dir out/build/linux-release
```

VS Code users can instead **Reopen in Container** — see `.devcontainer/`.

Model weights are not included; they are converted offline on the host (see
`Mila/Tools/Converters/`), and the repo bind mount makes the converted `.bin` files
available inside the container automatically.

> A slim, published runtime image — `docker run … mila` for users who only want to run
> inference without building — is planned for the beta release. See [ROADMAP.md](https://github.com/ToddThomson/Mila/blob/dev/ROADMAP.md).

---

## Documentation

API reference: https://toddthomson.github.io/Mila

Updated automatically on every push to master.

---

## Contributing

Mila is approaching a public beta and welcomes contributors who share its philosophy.
Good starting points are CPU reference ops, test coverage, and new encoding strategies
under /Components/Encodings/. Mila is GPU-first by design: the CUDA backend is the
validated inference path, and CPU op coverage beyond the GPT-2 lineage is intentionally
demand-driven — implementing a CPU op for Llama (RmsNorm, SwiGLU, RoPE, token embedding)
is a well-scoped, self-contained first contribution, not a gap to apologize for.

1. Fork the repository and create a branch from dev
2. Make changes with clear, focused commits
3. Ensure new components include forward and backward pass tests
4. Open a pull request targeting dev

New contributors: [getting-started.md](getting-started.md) walks through a fresh clone,
build, model weight conversion, running inference, and opening your first PR.
See CONTRIBUTING.md for coding standards and the pull request process.

---

## License

MIT License — see [License.md](License.md) for details.
