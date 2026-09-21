---
title: "Getting started"
description: "Build Mila from source and run local LLM inference in C++ on a CUDA GPU - clone, cmake, ctest, install a published model, and a first Llama 3.2 run."
layout: "page"
---

Mila is a source distribution. C++23 modules cannot ship as prebuilt binaries — the consumer compiles
the module graph in its own toolchain — so getting started means building it.

## 1. Check the toolchain

| | |
|---|---|
| **C++ compiler** | Visual Studio 2026 18.6.2+ on Windows, or Clang 19+ / GCC 16 on Linux |
| **CUDA Toolkit** | 13.3 |
| **CMake** | 4.0 or newer |
| **Generator** | Ninja — significantly faster for incremental module builds |
| **Git** | 2.x — dependencies are fetched at configure time |

Earlier Visual Studio 2026 builds have a C++23 module regression; 18.6.2 fixed it. GCC 15.2 and
earlier cannot compile the modules at all.

## 2. Build and test

```bash
git clone https://github.com/toddthomson/mila.git
cd mila
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build
```

A clone builds everything — the library, the tests, the samples, the chat app and the developer
tools. Each is a `MILA_ENABLE_*` option that is on when Mila is the top-level project and off when
another project consumes it, so a `FetchContent` consumer compiles the library alone. Set the ones
you do not want to `OFF` to trim a local build.

On Windows you can instead open the folder in Visual Studio, which detects `CMakeLists.txt` and the
bundled presets automatically.

There is also a dev container mirroring the CI toolchain, if you would rather not install CUDA and a
compiler locally.

## 3. Install a model

Models are not included, and you do not convert anything — Mila publishes models ready to run and
installs them by name:

```bash
mila install Llama-3.2-3B-Instruct-fp4
mila models --online
```

That first one is the smallest instruct model published and needs no HuggingFace account, so it is
the shortest path to a first answer. Larger ones — Gemma 4 12B, Llama 3.1 8B, Qwen 3.8 27B — install
the same way.

## 4. Run something

The chat app is the quickest way to see it work — an instruction-following console with model
hot-switching and tool calling:

```bash
mila-chat --model Llama-3.2-3B-Instruct-fp4
```

It opens on whatever your store holds; `/model list` shows that, and `/model install <name>` adds to
it without leaving the session.

## 5. Or consume it from your own project

CMake `FetchContent` is the supported path. Your project compiles Mila's modules in your toolchain,
then links against them like any other target.

---

**The full guide** — every step above with exact commands, the dev container, `FetchContent`
details, troubleshooting, and how to open a first pull request — is
[getting-started.md](https://github.com/ToddThomson/Mila/blob/master/getting-started.md) in the
repository. It is the authoritative version; this page is the short road in.
