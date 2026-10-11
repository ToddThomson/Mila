# Retiring the CPU Compute Operations

**Status:** Decided 2026-10-11 (Todd): the CPU compute operations are retired. Retirement is deletion; git history
is the archive. **Blocked on coverage** (section 4): nothing is deleted until every behaviour only a CPU test proves
is proven on CUDA, and two of the operations are Qwen 4's only implementation today. **Admitted to v0.21.0**
2026-10-11 (Todd) as an internal fix (`BACKLOG.md`).

## 1. What is retired, and what is not

| | Fate |
|---|---|
| The CPU compute operations: the 15 `OperationTraits.Cpu` specializations and their kernels under `Compute/Devices/Cpu/Operations/` | **Retired** |
| The components' CPU instantiations, the GPT-2 model on CPU, `CpuAdamWOptimizer`, and their `.Cpu.cpp` tests | **Retired** with them |
| The CPU **device**: host memory resources, staging, host tensors and their `TensorOps` (fill, zero, copy, random, math) | **Kept** — it is the host |
| The **CUDA-off build** (`MILA_ENABLE_CUDA=OFF`) | **Kept** — it guards the device boundary and builds `mila-store` and the store tests without the toolkit |
| The `TDeviceType` axis | **Kept**, with one compute value until a second backend arrives |
| Device-independent tests that happen to be named `.Cpu.cpp` (Distribution, Quantization packing, Serialization, Apps) | **Kept**, untouched |

## 2. Why

No end user runs a CPU model, and none of the CPU-only needs looked at serves one:

- **Running models:** an edge case (Todd), and the real CPU-only edge user — home automation on a Raspberry Pi or
  mini PC — needs 4-bit weights read on the CPU, NEON/AVX2 kernels and threads. These FP32 operations are not a
  step toward that; it would be a new backend, started from a measured quantized Linear, and it is breadth.
- **Testing the families on CPU:** a CPU arm tests a second implementation of the math, not the CUDA kernels,
  formats, replay or planning a user runs. Cheaper routes cover what it would: device-independent protocol and
  store tests, a CUDA compile-only CI job, and a GPU runner.
- **Reference:** no CUDA test checks against a CPU operation (one GPT-2 test is the exception); each computes its
  own exact reference on the host.
- **Cost:** every change to the `Operation` interface — decode, the decode position, scratch — is carried into 15
  operations nothing ships.

Edge hardware that matters for language models is mostly CUDA (Jetson, NVIDIA's ARM desktop boxes), which is a
direction for the CUDA path, not a reason to keep this one.

## 3. Method

For each CPU test file that goes, every behaviour it proves is either proven by a CUDA test already, moved to a
CUDA test, or judged not worth keeping by name. Counts of `backward`/`gradient` references were the first pass
(section 4); the per-test reading is the next step and replaces them.

## 4. Coverage, first pass

| CPU test | Tests | Backward refs | CUDA counterpart | Backward refs | First reading |
|---|---|---|---|---|---|
| `Activation.Cpu` | 5 | 5 | `Activation.Cuda` | 6 | likely covered |
| `Gelu.Cpu` | 12 | 24 | `Gelu.Cuda` | 9 | backward thinner on CUDA |
| `MultiHeadAttention.Cpu` | 12 | 18 | `MultiHeadAttention.Cuda` | 3 | **backward gap** |
| `Residual.Cpu` | 9 | 22 | `Residual.Cuda` | 4 | **backward gap** |
| `Lpe.Cpu` | 11 | 11 | `Lpe.Cuda` | 7 | read |
| `MLP.Cpu` | 31 | 42 | `MLP.Cuda` | 18 | read |
| `Linear.Cpu` | 27 | 32 | `Linear.Cuda`, `Linear.Decode.Cuda` | 25 | likely covered |
| `Softmax.Cpu` | 9 | 8 | `Softmax.Cuda` | 3 | read |
| `LayerNorm.Cpu` | 16 | 24 | `LayerNorm.Cuda` | 6 | **backward gap** |
| `RmsNorm.Cpu` | 17 | 26 | `RmsNorm.Cuda` | 7 | **backward gap** |
| `GptBlock.Cpu` | 22 | 31 | `GptBlock.Cuda` | 20 | read |
| `GptTransformer.Cpu` | 22 | 13 | `GptTransformer.Cuda` | 9 | read |
| `GptModel.Cpu` | 3 | 0 | `GptModel.Cuda` | 0 | likely covered |
| `AdamW.Cpu` | 32 | 38 | `AdamW.Cuda`, `AdamW.MixedPrecision.Cuda` | 52 | likely covered |
| `MixtureOfExperts.Cpu`, `Router.Cpu` | 7, 8 | 1, 1 | both have CUDA tests | 1, 1 | likely covered |
| `CausalConv1d.Cpu` | 9 | 0 | `CausalConv1d.Cuda` | 0 | likely covered |
| **`Network.Cpu`** | 12 | 35 | **none** | — | **the training spine is proven only here** |
| **`Model.Cpu`** | 2 | 0 | **none** | — | read |
| **`GatedResidual.Cpu`**, **`NgramEmbedding.Cpu`** | 9, 8 | 0 | **none, and no CUDA operation exists** | — | **Qwen 4's only implementation** |
| `CrossEntropy.Cpu` | 0 | — | — | — | empty file |

Two findings change the order of work:

1. **The training spine is proven only on CPU.** `Network.Cpu` holds the network-level training tests — build for
   training, backward through the graph, optimizer registration, and a training loop that reduces a regression
   loss — and nothing on CUDA proves the same. Several components' backward passes (multi-head attention,
   residual, LayerNorm, RmsNorm) are also tested more heavily on CPU. With training returning, these move to CUDA
   before the CPU path goes.
2. **Qwen 4 is mid-flight on CPU.** `GatedResidual` and `NgramEmbedding` exist only as CPU operations; Qwen 4's
   Phase 2 gated them on CPU against the tiny reference (`Qwen4.md` section 9). Retiring them now deletes the only
   implementation. Their CUDA operations — which Qwen 4 needs anyway — come first, gated against the same reference.

## 5. Next

1. Read each "read" and "gap" row test by test, and write the CUDA test list this retirement needs.
2. The CUDA tests for the training spine, then the CUDA operations for `GatedResidual` and `NgramEmbedding`.
3. The retirement change: delete the operations, their `OperationTraits.Cpu` entries, the CPU instantiations and
   the retired tests; reword "CPU-practical" text in specs and comments; the CPU-only CI job then proves the build
   and the device-independent tests, and says so.

The CPU device's memory reading (`Untriaged.md`, 2026-10-11) goes at triage: with no CPU compute backend, the
planner never offers the CPU.
