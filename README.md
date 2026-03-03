# Mila

**A C++23 module-based deep neural network library for those who want full control&mdash;to work at the metal.**

Mila is built for researchers, engineers, and developers who find high-level frameworks too opaque—who wants to understand exactly what happens in every forward pass, trace every gradient,
and write kernels that do precisely what they intend. No autograd engine. No runtime
dispatch magic. Just C++23, CUDA, and full control.

> *Currently in active alpha development. API is not yet stable.*
> *See the [Roadmap](ROADMAP.md) for current status and trajectory.*

---

| Branch | Build | Test | Docs |
|--------|-------|------|------|
| master | ![Build](https://github.com/ToddThomson/Mila/actions/workflows/build-pipeline.yml/badge.svg?branch=master&job=build) | ![Test](https://github.com/ToddThomson/Mila/actions/workflows/build-pipeline.yml/badge.svg?branch=master&job=test) | ![Docs](https://github.com/ToddThomson/Mila/actions/workflows/build-pipeline.yml/badge.svg?branch=master&job=docs) |
| dev    | ![Build](https://github.com/ToddThomson/Mila/actions/workflows/build-pipeline.yml/badge.svg?branch=dev&job=build) | ![Test](https://github.com/ToddThomson/Mila/actions/workflows/build-pipeline.yml/badge.svg?branch=dev&job=test) | ![Docs](https://github.com/ToddThomson/Mila/actions/workflows/build-pipeline.yml/badge.svg?branch=dev&job=docs) |

---

## What Mila Is

Mila is a component-based DNN library where **device and precision are chosen at compile time,
every forward and backward pass is explicit, and every gradient is yours to inspect**.

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
matters. float4 vectorized memory access in bandwidth-bound operations.

---

## Current Status — Alpha

Mila is under active development toward a public beta. The alpha phase focuses on
building and validating the core architecture against known-good reference implementations.

**Alpha.1 — Complete**
GPT-2 inference validated token-for-token against HuggingFace using greedy decoding.
The full GPT-2 stack — tokenizer, embeddings, attention, MLP, KV-cache — is implemented,
tested, and confirmed correct.

**Alpha.2 — In Progress**
Extending to the Llama architecture: RoPE, RMSNorm, SwiGLU, Grouped Query Attention.
Target: Llama 3.2 1B validated token-for-token against HuggingFace using the same
methodology applied to GPT-2.

See [ROADMAP.md](ROADMAP.md) for the full task breakdown.

---

## Validated Capabilities

| Capability | Status |
|---|---|
| GPT-2 inference — greedy and sampled | Validated against HuggingFace |
| Two-phase KV-cache — prefill + decode | Complete |
| HuggingFace GPT-2 weight converter | Complete |
| Chat CLI | Complete |
| MNIST training — 97.5% test accuracy | Complete |
| AdamW optimizer | Complete |
| cuBLASLt Linear — forward + backward | Complete |
| LayerNorm, RMSNorm, GELU, Softmax, CrossEntropy | Complete |
| Multi-Head Attention — forward + backward | Complete |
| BPE tokenizer | Complete |

---

## Samples

### Chat CLI

```
You: Once upon a time
Mila: , the world was a place of great beauty and great danger...
```

Located under `Samples/Chat`. Loads a converted HuggingFace GPT-2 checkpoint and
generates text using the two-phase KV-cache pipeline.

### MNIST Classifier

Located under `Samples/Mnist`. Trains a 3-layer MLP on MNIST to 97.5% test accuracy.
Demonstrates the full training loop: data loading, forward pass, loss, backward pass, AdamW step.

---

## Build

### Prerequisites

| Requirement | Version |
|---|---|
| Visual Studio | 2022 or newer |
| CUDA Toolkit | 13.0 |
| CMake | 4.0 or newer |
| GTest | 1.17.0 |
| C++ Standard | C++23 |

Ninja is the recommended generator — significantly faster than MSBuild for
incremental C++23 module builds.

### Quick Start

```bash
git clone https://github.com/toddthomson/mila.git
cd mila
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build
```

### Visual Studio

Open the repository folder — Visual Studio detects CMakeLists.txt automatically.
Select the Ninja generator and Release configuration. Build with F7.

### Docker

```bash
# GPU
docker run -it --rm --gpus all toddthomson/mila:latest

# CPU only
docker run -it --rm toddthomson/mila:latest

# Local build
git clone https://github.com/toddthomson/mila.git && cd mila
docker build -t mila:local .
docker run -it --rm --gpus all -v $(pwd):/mila/src mila:local
```

---

## Documentation

API reference: https://toddthomson.github.io/Mila

Updated automatically on every push to master.

---

## Contributing

Mila is approaching a public beta and welcomes contributors who share its philosophy.
Good starting points are CPU reference implementations, test coverage, and new
encoding strategies under /Components/Encodings/.

1. Fork the repository and create a branch from dev
2. Make changes with clear, focused commits
3. Ensure new components include forward and backward pass tests
4. Open a pull request targeting dev

See CONTRIBUTING.md for coding standards and the pull request process.

---

## License

Apache License 2.0 — see LICENSE for details.
