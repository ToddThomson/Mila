# Mila

**A C++23 module-based deep neural network library for those who want full control&mdash;to work at the metal.**

Mila is built for researchers, engineers, and developers who find high-level frameworks too opaque—who want to understand exactly what happens in every forward pass, trace every gradient,
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
matters. Vectorized memory access throughout — float4 for FP32, uint4 for BF16.

**Precision is deliberate.** BF16 is the primary reduced-precision compute target — it
matches FP32's exponent range, avoiding overflow and underflow without loss scaling, with
native Tensor Core support on Ada Lovelace and newer. FP16 is not a Mila target; BF16
supersedes it for all current use cases. FP8 quantization is applied at model load time
as a weight compression strategy: weights are quantized from BF16 to FP8_E4M3 inside the
`Linear` component via a compile-time `TWeightQuant` policy, enabling 8B-class models
to run within a 12 GB VRAM budget.

---

## Current Status — Alpha.5

Mila is under active development toward a public beta. The alpha phase focuses on
building and validating the core architecture against known-good reference implementations.

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

**Alpha.5 — In Progress**
FP8 load-time quantization pipeline, validated on Llama 3.2 3B Instruct. Weights are
quantized from BF16 to FP8_E4M3 inside `Linear` at model load time. The existing BF16
baseline provides the correctness reference for FP8 validation.

**Alpha.6 — Planned**
Qwen 3 transformer architecture with thinking mode and model-agnostic tool calling,
validated on Qwen 3 8B Instruct at BF16 and FP8. FP8 KV cache compression introduced
alongside weight quantization.

See [ROADMAP.md](ROADMAP.md) for the full task breakdown.

---

## Validated Capabilities

| Capability | Status |
|---|---|
| GPT-2 inference — greedy and sampled | Validated against HuggingFace |
| Llama 3.2 1B inference — greedy decode at FP32 | Validated against HuggingFace |
| Llama 3.2 3B inference — greedy decode at BF16 | Validated against HuggingFace |
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
