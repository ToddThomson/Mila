# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## Project Overview

Mila is a C++23 module-based DNN library for CUDA/CPU inference. It is in active alpha development (currently **Alpha.5**). The design philosophy: device and precision are compile-time decisions, every forward pass is explicit, and there is no hidden execution engine. Breaking changes are acceptable — backward compatibility is not a goal.

Primary validated targets: Llama 3.2 3B Instruct (BF16, FP8, FP4) and Llama 3.1 8B Instruct (FP4 default, FP8 alternative). The chat CLI default is Llama 3.1 8B FP4.

---

## Build

**Toolchain:** Visual Studio 2022+ (user uses VS 2026), CUDA Toolkit 13.0, CMake 4.0+, Ninja (required for C++23 module incremental builds), GTest 1.17.0, C++23.

The user builds exclusively inside **Visual Studio 2026** — never run `git commit`, `git add`, or `git push` commands. Describe what would make a good commit message and let the user commit.

```bash
# CMake configure (Ninja, Debug)
cmake -S . -B out/build/x64-debug -G Ninja -DCMAKE_BUILD_TYPE=Debug

# Build
cmake --build out/build/x64-debug

# Run all tests
ctest --test-dir out/build/x64-debug

# Run a single test binary (example)
./out/build/x64-debug/Mila/Tests/Dnn/Components/Activations/Gelu/GeluTests
```

CMake presets are in `CMakePresets.json`: `x64-debug`, `x64-release`, `x86-debug`, `x86-release`. The output directory is always `out/build/<preset-name>`.

---

## Repository Layout

```
Mila/
  Src/Dnn/
    Components/       DNN building blocks (Linear, Attention, Normalization, Activations, etc.)
      Transformers/   GptBlock/GptTransformer, LlamaBlock/LlamaTransformer
      Linear/         Linear.ixx — the reference component for OperationTraits dispatch
      Attention/      GroupedQueryAttention, MultiHeadAttention
      Normalization/  RmsNorm, LayerNorm
      Activations/    Gelu, Swiglu (SiLU)
      Embeddings/     TokenEmbedding
      Encodings/      RoPE, LPE
      FFN/            MLP
    Compute/          Device abstractions, dispatch, execution context
      Devices/Cuda/   CudaExecutionContext, CudaDevice, CUDA operation impls
        Operations/   CudaLinearOp, CudaGqaOp, CudaMhaOp, CudaGeluOp, etc.
      Devices/Cpu/    CpuExecutionContext, CPU operation impls
      Operations/     OperationTraits.Template.ixx, OperationTraits.ixx (.Cuda/.Cpu partitions)
                      OperationType.ixx, OperationBase.ixx
    Models/           GptModel.ixx, LlamaModel.ixx, LlamaModelConfig.ixx
    Quantization/     Weight/Policies.ixx (NoWeightQuant, PerChannelFp8<>, PerGroupFp4<>)
                      KvCache/Policy.ixx (NoKvCompression, PerChannelKvFp8<>)
    Tensors/          Tensor<T, MR> and memory resources
    Serialization/    Model weight loading from binary blobs
  Tests/Dnn/          GTest unit tests — mirrors Src/Dnn tree
  Samples/
    Chat/Src/         Chat CLI harness (fair game to edit freely — see API boundary below)
    MNIST/            MNIST training loop sample
    Bard/             GPT-2 text generation sample
  Tools/
    Converters/       Python scripts: convert_llama_weights.py, etc. (HuggingFace → Mila binary)
    Tokenize/         Tokenizer tools
  Specifications/     Design documents (OperationDispatch.md, Quantization.V2.md, etc.)
Data/
  Models/             Binary weight files (llama32_3b_instruct_bf16.bin, llama31_8b_instruct_bf16.bin, etc.)
  Scripts/            Python dev/conversion scripts
Dev/Scripts/          Python virtual environment and dev utilities
```

---

## Architecture

### Compile-Time Type Axes

Every component and operation is templated on two independent axes:

- **`TDeviceType`** (`DeviceType::Cpu` or `DeviceType::Cuda`) — determines memory resource and kernel dispatch.
- **`TPrecision`** (`TensorDataType::FP32`, `BF16`, etc.) — activation input and compute type. BF16 is the primary reduced-precision target; FP16 is not used.
- **`TWeightQuant`** (on `Linear` only) — compile-time weight quantization policy. Defaults to `NoWeightQuant`. Setting `PerChannelFp8<>` or `PerGroupFp4<128>` quantizes weights at model load time — no quantized checkpoint format, no runtime config object.

### Operation Dispatch — OperationTraits

Components resolve their concrete operation type at compile time via:

```cpp
using OpType = typename Compute::OperationTraits<
    Compute::OperationType::LinearOp,
    TDeviceType, TComputePrecision, TWeightQuant>::type;
```

The `OperationTraits` primary template is in `Compute/Operations/OperationTraits.Template.ixx`. Specializations live in `OperationTraits.Cuda.ixx` (`:Cuda` partition) and `OperationTraits.Cpu.ixx` (`:Cpu` partition). A missing specialization is a **hard compile error**, not a runtime miss. The old `OperationRegistry` string-keyed runtime dispatch is being phased out — do not add new `*Registrar` classes.

`Linear` is the canonical reference implementation of the full dispatch pattern.

### Component Lifecycle

Components derive from `Component<TDeviceType, TPrecision>`. Operations derive from `Operation<TDeviceType, TPrecision>`. `UnaryOperation`, `BinaryOperation`, and `PairedOperation` intermediate base classes are being removed — new ops derive directly from `Operation`.

Each component owns its parameters (weights, biases) and gradients. Composition is explicit with no shared global state. `IExecutionContext*` is passed at construction.

### Model Entry Points

- `LlamaModel::fromPretrained(path, config)` — loads weights from binary blob, dispatches to `fromPretrainedImpl<TWeightQuant, TKvPolicy>` based on `ChatConfig::QuantizationMode`.
- `GptModel::fromPretrained(path, config)` — equivalent for GPT-2.
- Both use a two-phase KV-cache: prefill (full sequence) + decode (one token at a time, outer_size == 1).

### Quantization Pipeline (Alpha.5)

Quantization happens at `loadParameter()` time inside `Linear`, not at checkpoint creation:
1. Converter always writes BF16.
2. `Linear<Cuda, BF16, PerChannelFp8<>>` calls `operation_->quantize()` during `loadParameter()`.
3. For FP8: per-channel absmax scales computed host-side, FP8 weights + FP32 scales uploaded to device; cuBLASLt handles mixed-precision GEMM natively.
4. For FP4 E2M1: per-group absmax scales, weights packed as nibbles (2 per byte); dequantized inline during the W4A16 GEMM tile load via LUT.
5. Decode path (outer_size == 1): dedicated `matvec_decode_bf16_qfp8_kernel` / `matvec_decode_bf16_qfp4_kernel` — bypasses cuBLASLt entirely.

The `getDeviceScratchBuffer()` grow-on-demand shared scratch buffer in `ExecutionContext` is used for the FP8 2-phase dequant staging buffer — **fetch at `forward()` time, never cache the pointer across calls** (it may be reallocated on grow).

---

## Chat Harness (`Mila/Samples/Chat/Src/`)

**API Boundary:** Files under `Mila/Samples/Chat/Src/` can be edited freely. Any change to the core Mila library (`Mila/Src/`) requires explicit agreement first.

Key files:
- `Chat.ixx` — main chat loop, model hot-switching (`/model <alias> [quant]`), tool call dispatch
- `Chat.Config.ixx` — `ChatConfig` with `ModelType`, `ModelSize`, `ModelPrecision` (compute), `QuantizationMode` (none/fp8/fp4)
- `Chat.Renderer.ixx` — `ConsoleRenderer` (standalone non-exported module): braille spinner, solid-color response blocks, word-wrap with leading-indent preservation, Unicode welcome box, ANSI stats line
- `main.cpp` — entry point; default model is Llama 3.1 8B FP4

Model aliases: `gpt2`, `llama-1b`, `llama-3b`, `llama-8b`, and `-fp32` variants. `llama-8b` uses the `llama31` family prefix in filename construction; 1B/3B use `llama32`.

All responses are fully buffered before display — streaming has been removed from the hot path.

---

## C++ Module Conventions

Source files use `.ixx` for C++23 module interface units and module partitions. The module naming convention mirrors the directory structure (e.g., `Compute.OperationTraits`, `Dnn.Components.Linear`).

Module partition files (`:Cuda`, `:Cpu` suffixes) are used to separate backend specializations while keeping a single aggregator module. Example: `OperationTraits.ixx` re-exports `OperationTraits.Template.ixx` + `:Cuda` + `:Cpu`.

---

## Code Style

- **No abbreviations in identifiers.** All names must be spelled out in full: `Quantization` not `Quant`, `Parameter` not `Param`, `Context` not `Ctx`, `Index` not `Idx`, `Implementation` not `Impl`. Template parameters follow the same rule: `TWeightQuantization` not `TWeightQuant`. Exception: established acronyms like `Kv` (Key-Value), `Gqa`, `Mha`, `Mlp`, `Lpe`, `Bpe` are acceptable.
- No column alignment with extra spaces — single-space formatting throughout.
- Blank line before control flow blocks (`if`, `for`, `while`, `switch`).
- Blank line after closing brace of blocks.
- No blank line between `} else {` or `} catch {`.
- Blank line before final `return`. No blank line for early-return guard clauses.
- Comments explain WHY or state a non-obvious contract — never restate what the code does.
- ASCII only in code comments (no Unicode symbols, emojis).
- File-level Doxygen: one to three sentences maximum. Detail belongs on the symbol.

---

## Workflow Notes

- When the user ends a message with **"Your thoughts?"** — respond with analysis only. No code edits.
- User commits via **VS 2026 integrated git**. When a commit point is reached, suggest a commit message but do not run any git commands.

---

## Key Specifications

Design decisions are documented under `Mila/Specifications/`:
- `OperationDispatch.md` — the full `OperationTraits` design, migration checklist, file layout
- `Quantization.V2.md` — quantization policy design and scope table
- `PromptCaching.md`, `TokenSampling.md`, `ToolCalling.md` — planned features

Current progress is tracked in `ROADMAP.md` at the repo root.
