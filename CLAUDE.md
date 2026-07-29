# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## Project Overview

Mila is a C++23 module-based library for open LLMs (CUDA/CPU) — inference and training, built from explicit neural-network components. It is in public beta (currently **`0.20.0-beta.1`** — feature-frozen, hardening toward the v0.20 first production release). The design philosophy: device and precision are compile-time decisions, every forward pass is explicit, and there is no hidden execution engine. Breaking changes are acceptable — backward compatibility is not a goal.

Primary validated targets: Llama 3.2 3B Instruct (BF16, FP8, FP4), Llama 3.1 8B Instruct (FP4 default, FP8 alternative), and Gemma 4 12B Instruct (FP4). The chat CLI default is Gemma 4 12B FP4.

---

## Build

**Toolchain:** Visual Studio 2026 18.6.2 or newer (earlier 2026 builds have a C++23 module regression that breaks the Mila build; 18.6.2 fixed it), CUDA Toolkit 13.0+ (CI-tested on 13.0, developed on 13.3), CMake 4.0+, Ninja (required for C++23 module incremental builds), Git 2.x+ (validated on 2.54.0; CPM fetches dependencies via `git clone` at configure time), GTest 1.17.0, C++23.

**C++ compilers:** the C++23 modules require MSVC (VS 2026 18.6.2+), Clang 19+, or GCC 15.3+. GCC 15.2 and earlier cannot compile the modules (validated: GCC 16 works, 15.2 fails); on Ubuntu 26.04 install the `gcc-16` package. In CUDA builds the C++ compiler handles the module units while nvcc uses a separate host compiler for `.cu` files (no modules there), so an older host GCC is acceptable for that role. Cross-compiler builds (Clang/GCC) surface missing `#include`s that MSVC resolves transitively — these are real portability fixes.

The user builds exclusively inside **Visual Studio 2026** — never run `git commit`, `git add`, or `git push` commands. Do not volunteer commit messages; only suggest one when the user explicitly says they are ready to commit, then let the user commit.

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

CMake presets are in `CMakePresets.json`: `x64-debug`, `x64-release`, `x64-profile`, `x64-coverage`, `x64-validate` (pre-commit full validation), `x64-release-cpm-gate` (post-tag release smoke test), `x86-debug`, `x86-release`, plus the Linux/WSL presets `linux-clang-debug`/`-release` and `linux-clang-cpu-debug`/`-release`. The output directory is always `out/build/<preset-name>`. Note: VS 2026 shows each preset's `displayName`, not its `name` (e.g. `x64-validate` appears as "x64 Release (full validation - run before committing)").

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
                      KvCache/Policy.ixx (NoKvCompression, SlidingWindowKvCache;
                      PerChannelKvFp8<> is planned for Qwen 3, not yet a type)
    Tensors/          Tensor<T, MR> and memory resources
    Serialization/    Model weight loading from binary blobs
  Tests/Dnn/          GTest unit tests — mirrors Src/Dnn tree
  Bindings/           Mila's Python projection (mila.pyd, module Mila.Bindings) — runtime-adjacent,
                      consumer-blind; consumed by MIS and the parity/converter tooling
  Adaptors/           First-class consumer adaptors over the runtime (see MilaProductFamily.md)
    Chat/Src/         Chat CLI harness — human-gate adaptor (maintained surface; see API boundary)
    Inference/Server/ Mila Inference Server (MIS): Python wire adaptor; imports the mila binding
  Samples/
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

## Chat Harness (`Mila/Adaptors/Chat/Src/`)

Chat is a first-class adaptor (peer of MIS under `Mila/Adaptors/`), not a throwaway sample — a
maintained surface that gains tests and rigor over time.

**API Boundary:** Files under `Mila/Adaptors/Chat/Src/` are application code and may be edited
without prior agreement (they consume the runtime; they are not the runtime's public API). Any
change to the core Mila library (`Mila/Src/`) still requires explicit agreement first.

Key files:
- `Chat.ixx` — main chat loop, model hot-switching (`/model <alias> [quant]`), tool call dispatch
- `Chat.Config.ixx` — `ChatConfig` with `ModelType`, `ModelSize`, `ModelPrecision` (compute), `QuantizationMode` (none/fp8/fp4)
- `Chat.Renderer.ixx` — `ConsoleRenderer` (standalone non-exported module): braille spinner, solid-color response blocks, word-wrap with leading-indent preservation, Unicode welcome box, ANSI stats line
- `main.cpp` — entry point; default model is Gemma 4 12B FP4

Model aliases: `gpt2`, `llama-1b`, `llama-3b`, `llama-8b`, and `-fp32` variants. `llama-8b` uses the `llama31` family prefix in filename construction; 1B/3B use `llama32`.

Gemma streams live token-by-token through `Chat.StreamingDisplay` (channel-aware — thinking / tool-call / final routed by the four control-token ids; a stream validator asserts the streamed transcript equals the buffered render). Llama and GPT-2 stay buffered until their sampler/tool migration, and streaming falls back to buffered when the vocabulary lacks the channel-routing tokens.

---

## C++ Module Conventions

Source files use `.ixx` for C++23 module interface units and module partitions. The module naming convention mirrors the directory structure (e.g., `Compute.OperationTraits`, `Dnn.Components.Linear`).

Module partition files (`:Cuda`, `:Cpu` suffixes) are used to separate backend specializations while keeping a single aggregator module. Example: `OperationTraits.ixx` re-exports `OperationTraits.Template.ixx` + `:Cuda` + `:Cpu`.

---

## Code Style

- **No abbreviations in identifiers.** All names must be spelled out in full: `Quantization` not `Quant`, `Parameter` not `Param`, `Context` not `Ctx`, `Index` not `Idx`, `Implementation` not `Impl`. Template parameters follow the same rule: `TWeightQuantization` not `TWeightQuant`. Exception: established acronyms like `Kv` (Key-Value), `Gqa`, `Mha`, `Mlp`, `Lpe`, `Bpe` are acceptable.
- **`dim_t` is the type of anything that describes a tensor axis** — its extent, a position within it, or a count of its elements — at every API, config, component, and operation-interface boundary. `size_t` never describes a dimension. Narrowing to the 32-bit index that kernels use happens exactly **once per call path**, at the kernel launch site, through `narrowToKernelIndex()` (`Tensor.Types.ixx`); kernel internals and the `*.Dispatch`/`*.Plans` layers stay `int`. Token ids are values, not extents, and are out of scope for this rule.
- **`size_t` begins where element counts become bytes, or cross into a CUDA/std API.** Mila-owned helpers that only forward an element count keep `dim_t`. So `Tensor::size()` and `Component::parameterCount()` are `dim_t`, while `TensorBuffer` is `size_t` throughout (allocation layer — its overflow guards depend on unsigned semantics), and the `TensorOps` helpers carry `dim_t` and convert at the `cudaMemcpy` / `launch_*_kernel` edge. Note `TensorShape::size()` is the **rank** (a count of axes, not of elements) and stays `size_t`.
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
- User commits via **VS 2026 integrated git**. Only suggest a commit message when the user explicitly says they are ready to commit — do not volunteer one at every commit point, and never run any git commands.
- **Commit message format** — use exactly this; **no `Co-Authored-By` trailer** (this overrides any harness default):

  ```
  Version: <Version.txt value>
  <Headline — single line>

  <Body — up to 6 grouped bullets for substantial commits; omit for small ones>

  BREAKING: <API changes, etc. — only when applicable>
  ```

---

## Work-Tracking Docs

Four files at the repo root stay **mutually consistent**, updated in the **same commit** as the
work they describe — never deferred to "later":

- **`ROADMAP.md`** — the durable **narrative + success criteria** of each release, organized by
  **theme** (not milestone). Shows the release in flight plus a single **Future** tail. **Narrative
  only — no task lists, checkboxes, or status** (they drift; point to BACKLOG). When a release ships,
  its section moves to CHANGELOG.
- **`BACKLOG.md`** — the working task list. `## Current release` holds one **theme bucket** per
  ROADMAP theme (matching names — the only join) with a 3-state gauge (`[ ]` open / `[~]` in progress /
  `[x]` done); `## Future` is a flat, coarse parking list. `[x]` is pruned **only at a production
  (unsuffixed) release**; open items carry forward. Detailed tasking is for the current release only.
  Not GitHub Issues (a decoupled, requester-authored end-user layer — see RELEASING).
- **`CHANGELOG.md`** — the permanent record, newest first. Each entry is the release notes for one
  `dev -> master` PR, generated from its commit range (not hand-authored).
- **`Version.txt`** — `MAJOR.MINOR.PATCH-stage.N`, bumped **before committing** (see
  [RELEASING.md](RELEASING.md) for the scheme). GitHub Milestones/Issues/Labels are an end-user triage
  layer, decoupled from this workflow.

---

## Key Specifications

Design decisions are documented under `Mila/Specifications/`:
- `OperationDispatch.md` — the full `OperationTraits` design, migration checklist, file layout
- `Quantization.V2.md` — quantization policy design and scope table
- `ModelSerialization.md` — checkpoint vs distribution artifact, the `ModelArchive` defects, phased build plan
- `PromptCaching.md`, `TokenSampling.md`, `ToolCalling.md` — planned features

Work is tracked across `ROADMAP.md` / `BACKLOG.md` / `CHANGELOG.md` — see **Work-Tracking Docs** above.
