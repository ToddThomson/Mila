---
title: "Getting started"
description: "Build Mila from source, convert model weights, and run local LLM inference in C++ on a CUDA GPU - clone, cmake, ctest, and a first Gemma 4 or GPT-2 run."
layout: "page"
---

Mila is a source distribution. C++23 modules cannot ship as prebuilt binaries — the consumer compiles
the module graph in its own toolchain — so getting started means building it.

## 1. Check the toolchain

| | |
|---|---|
| **C++ compiler** | Visual Studio 2026 18.6.2+ on Windows, or Clang 19+ / GCC 16 on Linux |
| **CUDA Toolkit** | 13.0+ on Windows, 13.3+ on Linux |
| **CMake** | 4.0 or newer |
| **Generator** | Ninja — significantly faster for incremental module builds |
| **Git** | 2.x — dependencies are fetched at configure time |

Earlier Visual Studio 2026 builds have a C++23 module regression; 18.6.2 fixed it. GCC 15.2 and
earlier cannot compile the modules at all.

## 2. Build and test

```bash
git clone https://github.com/toddthomson/mila.git
cd mila
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DMILA_ENABLE_TESTING=ON
cmake --build build
ctest --test-dir build
```

Tests are opt-in — drop `-DMILA_ENABLE_TESTING=ON` for a library-only build. On Windows you can
instead open the folder in Visual Studio, which detects `CMakeLists.txt` and the bundled presets
automatically.

There is also a dev container mirroring the CI toolchain, if you would rather not install CUDA and a
compiler locally.

## 3. Get model weights

Weights are not included. A converter turns HuggingFace checkpoints into Mila's binary format, and
always writes BF16 — quantization to FP8 or FP4 happens at load time, so there is no separate
quantized checkpoint to manage.

**GPT-2 is the easiest first target**: it is ungated, small, and needs no HuggingFace account. Llama
and Gemma are gated and require auth.

## 4. Run something

The Chat CLI is the quickest way to see it work — an instruction-following harness with model
hot-switching and tool calling, defaulting to Gemma 4 12B at FP4.

## 5. Or consume it from your own project

CMake `FetchContent` is the supported path. Your project compiles Mila's modules in your toolchain,
then links against them like any other target.

---

**The full guide** — every step above with exact commands, converter setup, the gated-model auth
flow, `FetchContent` details, and how to open a first pull request — is
[getting-started.md](https://github.com/ToddThomson/Mila/blob/master/getting-started.md) in the
repository. It is the authoritative version; this page is the short road in.
