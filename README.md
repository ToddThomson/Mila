| Branch | Build | Test | Docs |
|--------|-------|------|------|
| master | ![Build](https://github.com/ToddThomson/Mila/actions/workflows/build-pipeline.yml/badge.svg?branch=master&job=build) | ![Test](https://github.com/ToddThomson/Mila/actions/workflows/build-pipeline.yml/badge.svg?branch=master&job=test) | ![Docs](https://github.com/ToddThomson/Mila/actions/workflows/build-pipeline.yml/badge.svg?branch=master&job=docs) |
| dev    | ![Build](https://github.com/ToddThomson/Mila/actions/workflows/build-pipeline.yml/badge.svg?branch=dev&job=build) | ![Test](https://github.com/ToddThomson/Mila/actions/workflows/build-pipeline.yml/badge.svg?branch=dev&job=test) | ![Docs](https://github.com/ToddThomson/Mila/actions/workflows/build-pipeline.yml/badge.svg?branch=dev&job=docs) |

# Mila
Mila Deep Neural Network Library

## Prerelease Notice

**Mila v0.10.1-alpha.2** — internal development build.

This release is not API-stable. Breaking changes may occur between alpha versions.
See [ROADMAP.md](ROADMAP.md) for the current development trajectory.

---

## Overview

Mila is a C++23 deep neural network library built on CUDA and cuBLASLt. It provides a
component-based architecture for composing, training, and running inference on neural
networks with explicit, manually-implemented forward and backward passes — no autograd engine.

**Design philosophy:**
- Device and precision are template parameters, not runtime choices — type safety is enforced at compile time
- Every component owns its parameters and gradients; composition is explicit
- The backward pass is implemented manually per component, making gradient flow auditable and predictable
- C++23 modules throughout — no header-soup, fast incremental builds with Ninja

---

## Current State — Alpha.1 Complete ✅

Alpha.1 closed with GPT-2 inference **validated token-for-token against HuggingFace** using
greedy decoding. The root cause of the original divergence — an inter-head stride bug in the
padded KV-cache `unpermute_output` kernel — was isolated and fixed through systematic
per-component comparison against the reference implementation.

| Capability | Status |
|---|---|
| GPT-2 inference (greedy + sampled) | ✅ Validated against HuggingFace |
| Two-phase KV-cache (prefill + decode) | ✅ |
| HuggingFace GPT-2 weight converter | ✅ |
| Chat CLI (`Mila.Chat`) | ✅ |
| MNIST training — 97.5% test accuracy | ✅ |
| AdamW optimizer | ✅ |
| cuBLASLt Linear (forward + backward) | ✅ |
| LayerNorm, GELU, Softmax, CrossEntropy | ✅ |
| Multi-Head Attention (forward + backward) | ✅ |
| BPE tokenizer (GPT-2) | ✅ |

---

## Alpha.2 Focus — Llama Compatibility 🚧

Alpha.2 applies the same validation methodology to the Llama architecture:
convert HuggingFace weights → run greedy decode → compare token-for-token.

| Component | Purpose |
|---|---|
| `RMSNorm` | Replaces `LayerNorm` in Llama |
| `RotaryEmbedding` (RoPE) | Replaces learned positional embeddings |
| `SiLU` + `SwiGLU` MLP | Replaces GELU MLP |
| `GroupedQueryAttention` | GQA with configurable `num_kv_heads` |
| `LlamaBlock` / `LlamaTransformer` / `LlamaModel` | Full decoder stack |
| SentencePiece / tiktoken tokenizer | Llama 3.x tokenization |

Target model: **Llama 3.2 1B** (smallest publicly accessible variant).

See [ROADMAP.md](ROADMAP.md) for the full task breakdown.

---

## Samples

### Chat CLI

A GPT-2 completion CLI under `Samples/Chat` loads a converted HuggingFace GPT-2
checkpoint and generates text using the two-phase KV-cache pipeline.

```
You: Once upon a time
Mila: , the world was a place of great beauty and great danger...
```

### MNIST

A feedforward classifier under `Samples/Mnist` trains a 3-layer MLP on MNIST,
reaching **97.5% test accuracy**. Demonstrates the full training loop:
data loading → forward pass → loss → backward pass → AdamW step.

---

## Top Features

- **GPT-2 inference validated** — token-for-token agreement with HuggingFace on greedy decode
- **CUDA-accelerated via cuBLASLt** — matrix operations, attention, layer normalization
- **Two-phase KV-cache** — prefill + autoregressive decode for efficient generation
- **Component-based architecture** — composable, device-templated building blocks
- **C++23 modules** — fast incremental builds, no leaking header dependencies
- **Manual backward pass** — explicit, auditable gradient flow with no autograd overhead
- **CPU and CUDA parity** — all components run on both devices

---

## Documentation

API reference hosted on GitHub Pages:
[https://toddthomson.github.io/Mila](https://toddthomson.github.io/Mila)

Updated automatically on every push to `master`.

---

## Build Instructions

### Prerequisites

| Requirement | Version |
|---|---|
| Visual Studio | 2022 or newer (2026 recommended) |
| CUDA Toolkit | 13.0 |
| CMake | 4.0 or newer |
| GTest | 1.17.0 |
| C++ standard | C++23 |

Ninja is the recommended generator — significantly faster than MSBuild for
incremental C++23 module builds.

### Visual Studio (Recommended)

1. Clone the repository:
   ```
   git clone https://github.com/toddthomson/mila.git
   cd mila
   ```
2. Open the folder in Visual Studio — it detects `CMakeLists.txt` automatically.
3. Select the **Ninja** generator and **Release** configuration in CMake Settings.
4. **Build All** via the Build menu or `F7`.
5. Run tests via **Test Explorer**.

### Command Line (Ninja)

```
git clone https://github.com/toddthomson/mila.git
cd mila
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build
```

### Visual Studio Code

1. Install the **CMake Tools** and **C/C++** extensions.
2. Open the repository folder — CMake Tools detects the project automatically.
3. Select the **Ninja** kit and **Release** variant.
4. Run **CMake: Build** (`Ctrl+Shift+P`) and **CMake: Run Tests**.

### Docker (Linux + GPU)

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

## License

Mila is licensed under the **Apache License 2.0**.

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the
License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
either express or implied. See the License for the specific language governing permissions
and limitations under the License.

---

## Contributing

1. Fork the repository and create a branch from `dev`.
2. Make changes with clear, focused commits.
3. Ensure new components include both forward and backward pass tests.
4. Open a pull request targeting `dev`.

Refer to [CONTRIBUTING.md](CONTRIBUTING.md) for coding standards and the pull request process.
