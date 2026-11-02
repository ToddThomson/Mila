# Mila Mixed-Precision Compute Architecture Specification

## Overview

This document captures the design rationale and implementation decisions for Mila's
mixed-precision CUDA compute backend. The SwiGLU op is the canonical reference
implementation. All ops follow the same pattern.

---

## 1. Supported Precision Types

Mila supports the following `TensorDataType` values for CUDA compute:

| Abstract Type        | Native CUDA Type    | Forward Activations | Gradient Buffer | Notes                        |
|----------------------|---------------------|---------------------|-----------------|------------------------------|
| `TensorDataType::FP32`     | `float`             | FP32                | FP32            | Baseline, fully validated    |
| `TensorDataType::BF16`     | `__nv_bfloat16`     | BF16                | FP32            | Primary inference + training |
| `TensorDataType::FP16`     | `__half`            | FP16                | FP32            | Deferred — no current need   |
| `TensorDataType::FP8_E4M3` | `__nv_fp8_e4m3`     | FP8                 | FP32            | Future                       |
| `TensorDataType::FP8_E5M2` | `__nv_fp8_e5m2`     | FP8                 | FP32            | Future                       |

**BF16 is the primary reduced-precision target.** It has the same dynamic range as FP32
(same exponent width), which makes it numerically stable for both inference and training
without loss scaling. The RTX 4070 (Ada Lovelace) has native BF16 Tensor Core support.
FP16 is deferred — there is no current use case that BF16 does not serve better on the
target hardware.

---

## 2. Type Resolution Chain

The dispatch chain from abstract type to native kernel is fully compile-time:

```
TensorDataType (enum)
  └─► TensorDataTypeMap<TPrecision>::native_type       // abstract → native C++ type
        └─► cuda_op_impl<NativeType>                   // dispatch struct (per op)
              └─► cuda_op_forward_bf16(...)            // plain C kernel launcher
                    └─► op_bf16_forward_kernel<<<>>>   // __global__ kernel
```

### Key files per op

| File                        | Role                                                        |
|-----------------------------|-------------------------------------------------------------|
| `Op.ixx`                    | Hardware-agnostic component. Knows only `TensorDataType`.   |
| `CudaOp.ixx`                | CUDA op. Resolves `NativeType` via `TensorDataTypeMap`.     |
| `CudaOp.Dispatch.ixx`       | Module partition. `cuda_op_impl<NativeType>` structs.       |
| `CudaOp.Registrar.ixx`      | Runtime registry. One `registerUnaryOperation` per type.    |
| `Op.Fp32.cu`                | FP32 kernel + launcher.                                     |
| `Op.Bf16.cu`                | BF16 kernel + launcher.                                     |

### `TensorDataTypeMap` is the single source of truth

`CudaTensorDataType-Maps.ixx` maps every `TensorDataType` to its CUDA native type.
No op or kernel file duplicates this mapping. The dispatch struct constraint on the
primary template derives from this map — it is never hand-enumerated per op.

---

## 3. Dispatch Partition Contract

Each op provides a `CudaOp.Dispatch.ixx` module partition containing:

```cpp
namespace Detail
{
    // Primary template — gates to CUDA float native types only.
    // Constraint is derived from TensorDataTypeMap, never hand-enumerated.
    template <typename TNative>
        requires CudaFloatNativeType<TNative>
    struct cuda_op_impl;

    // One complete specialization per supported native type.
    // If a specialization exists, ALL methods are implemented — no stub throws.
    // If a type is not ready, there is no specialization. The missing
    // specialization is a compile error at CudaOp instantiation — the correct
    // failure mode.
    template <>
    struct cuda_op_impl<float> { ... };

    template <>
    struct cuda_op_impl<__nv_bfloat16> { ... };
}
```

### Rules

- **Complete or absent.** A specialization that throws at runtime for an unimplemented
  method violates the contract. The compile error from a missing specialization is the
  correct diagnostic.
- **No runtime type switching.** The dispatch struct is instantiated at compile time from
  `NativeType`. There are no `if/switch` on `TensorDataType` at runtime inside an op.
- **No state for elementwise ops.** For ops like SwiGLU where the impl carries no
  per-instance data, the struct is empty and compiles to nothing. The layer exists
  for consistency and to accommodate stateful ops (e.g. cuBLASLt plan holders in
  `CudaLinearOp`, `CudaGqaOp`).

---

## 4. Registrar Contract

Each op provides a `CudaOp.Registrar.ixx` containing a `CudaOpRegistrar` class with
a static `registerOperations()` method. One `registerUnaryOperation` (or equivalent)
call per supported `TensorDataType`.

### Rules

- **Registrar and dispatch specialization set must stay in sync.** Registering a type
  that has no dispatch specialization is a compile error. Providing a dispatch
  specialization without a registrar entry is a silent runtime omission — the op
  compiles but is unreachable via the registry.
- **When adding a new type**, both the dispatch specialization and the registrar entry
  must land together.

---

## 5. Memory Layout

Mila uses **contiguous halves** throughout, not interleaved per-token layout.

For SwiGLU with input `X` of size `2N` and output `Y` of size `N`:

```
X: [ gate_0, gate_1, ..., gate_N-1 | up_0, up_1, ..., up_N-1 ]
    └──────────── first half ───────┘└─────────── second half ─┘
Y: [ y_0, y_1, ..., y_N-1 ]
```

**Rationale:** Contiguous halves is easier to reason with, produces simpler vectorized
indexing (no per-element token/col arithmetic), and is consistent with how all other
Mila ops handle split buffers (QKV projection, etc.).

This differs from HuggingFace's interleaved layout which falls out of fused QKV
projections. Mila uses explicit separate projections so the contiguous layout is natural.

**Batch size:** Mila targets B=1 for decode (single-user local inference). Batch > 1
is not a current architectural requirement. The vectorized kernels are correct for B=1
by construction.

---

## 6. Memory Alignment

Every tensor buffer pointer is guaranteed aligned at allocation time by
`get_alignment<TDataType, MR>()`:

```cpp
// CUDA alignment = CUDA_WARP_SIZE (32) * sizeof(element)
FP32:  128 bytes  (32 floats)      — supports float4 loads
BF16:   64 bytes  (32 bfloat16)    — supports uint4 loads (8 BF16 per load)
FP16:   64 bytes  (32 halfs)       — supports uint4 loads
INT8:   32 bytes  (32 int8)        — supports int4 loads
```

**Consequence for kernels:** Vectorized loads are unconditional — no scalar prologue,
no scalar epilogue for alignment. The only remainder handling required is for `N` not
being a multiple of the vector width, which is enforced at the op level (see Section 7).

---

## 7. Vectorization

All elementwise CUDA kernels use vectorized loads and stores. The kernel is
unconditionally vectorized — no scalar fallback path.

### Vector widths per type

| Type   | Vector type  | Elements per thread | Bytes per load |
|--------|--------------|---------------------|----------------|
| FP32   | `float4`     | 4                   | 16             |
| BF16   | `uint4`      | 8                   | 16             |
| FP16   | `uint4`      | 8                   | 16             |

### Exported vector width constants

Each kernel file exports a `constexpr int kOpTypeVectorWidth` constant. The op's
`forward()` validates against this constant rather than a magic number:

```cpp
// Op.Fp32.cu
constexpr int kSwigluFp32VectorWidth = 4;

// Op.Bf16.cu
constexpr int kSwigluBf16VectorWidth = 8;
```

### Op-level validation

The op `forward()` enforces the vector width precondition before launching the kernel:

```cpp
// Example for BF16 SwiGLU — input size must be multiple of 2 * VectorWidth
// (gate half + up half, each must be a multiple of VectorWidth)
if ( input.size() % ( 2 * kSwigluBf16VectorWidth ) != 0 )
{
    throw std::invalid_argument(
        std::format( "CudaSwigluOp: input size must be a multiple of {} for vectorized BF16.",
            2 * kSwigluBf16VectorWidth )
    );
}
```

### Block size

All forward and backward kernels use **256 threads per block**. Grid size is computed
over the number of vector-width chunks, not scalar elements:

```cpp
int vec_N = N / kVectorWidth;
int grid_size = ( vec_N + 256 - 1 ) / 256;
```

---

## 8. BF16 Arithmetic: FP32 Promotion

BF16 kernels load and store in BF16 but compute in FP32. This applies to both
forward and backward passes.

**Rationale:** Training stability. BF16 has only 7 mantissa bits — insufficient
precision for intermediate values like sigmoid, exp, and gradient chain products.
Promoting to FP32 for arithmetic gives full precision where it matters while
preserving the memory bandwidth and VRAM benefits of BF16 storage. This is
consistent with PyTorch's internal BF16 kernel strategy.

### Pattern for paired BF16 arithmetic

```cuda
// Load 8 BF16 elements as uint4
uint4 packed = reinterpret_cast<const uint4*>(X)[i];

// Reinterpret as four __nv_bfloat162 pairs
__nv_bfloat162 ab = reinterpret_cast<const __nv_bfloat162*>(&packed)[0];
__nv_bfloat162 cd = reinterpret_cast<const __nv_bfloat162*>(&packed)[1];
// ... etc

// Promote to FP32 for arithmetic
float2 ab_f = __bfloat1622float2( ab );
// ... compute in float ...
// Demote back to BF16 for store
__nv_bfloat162 result = __float22bfloat162_rn( result_f );
```

---

## 9. Mixed-Precision Training: Forward and Backward Tensor Types

### Forward pass

| Input tensor  | Output tensor |
|---------------|---------------|
| BF16          | BF16          |

### Backward pass

| dY (upstream gradient) | X (saved activations) | dX (gradient output) |
|------------------------|-----------------------|----------------------|
| FP32                   | BF16                  | FP32                 |

**Rationale:** FP32 gradients are the canonical format at the optimizer boundary.
Both CUDA Adam (on-device) and CPU Adam (offloaded) consume FP32 gradients.
CPU Adam is the practical path for users who offload optimizer state to host RAM
to extend effective VRAM — a first-class use case for Mila.

The backward kernel signature for all BF16 ops follows this pattern:

```cuda
void cuda_op_backward_bf16(
    float* dX,                   // FP32 gradient output — optimizer boundary
    const __nv_bfloat16* X,      // BF16 saved forward activations
    const float* dY,             // FP32 upstream gradient
    int N,
    cudaStream_t stream )
```

---

## 10. Mixed-Precision Training: Weight Strategy

Mila targets the standard Micikevicius (2018) mixed-precision training recipe:

```
FP32 master weights
  → cast to BF16 for forward pass
  → BF16 activations through forward
  → FP32 gradients through backward
  → FP32 Adam optimizer step (CUDA or CPU)
  → update FP32 master weights
  → repeat
```

**Minimum hardware requirement for training:** 16GB VRAM.

**Rationale for FP32 master weights:**
- Adam's first and second moments require FP32 to accumulate small updates correctly
- Weight updates (`w -= lr * grad`) can be vanishingly small relative to `w` in BF16
- FP32 master weights make Mila's training results directly comparable to the literature
- At 16GB the standard recipe is viable for Llama 1B with gradient checkpointing

**CPU Adam offload:** Users with 16GB VRAM and sufficient system RAM can offload
FP32 master weights and Adam moments to CPU, extending effective training capacity
well beyond 1B parameters. This is a first-class supported configuration.

**`REVIEW:`** Stochastic rounding on BF16 weight updates is a future consideration
for training stability without FP32 master weights. Not a current kernel concern but
must not be accidentally designed around in the optimizer interface.

---

## 11. cuBLASLt and BF16

For ops using cuBLASLt (`CudaLinearOp`, `CudaGqaOp`):

- Data type: `CUDA_R_16BF`
- Compute type: `CUBLAS_COMPUTE_32F_FAST_16BF`

This is mixed-precision matmul: data moves in BF16, accumulation is FP32 internally.
`CudaDataTypeMap<__nv_bfloat16>::fp32_compute_type` holds this value.

**Important:** `CudaDataTypeMap<__nv_bfloat16>` has no `compute_type` member
(BF16-native accumulation does not exist in cuBLAS). Plan builders must always use
`fp32_compute_type` for BF16 — never `compute_type`. Any generic plan builder
that falls through to `compute_type` without type checking is a silent bug.

Pre-built cuBLASLt plans are cached per op instance. Plan selection for
mixed-precision BF16 matmuls has more heuristic space than FP32 — cuBLASLt
may select a different optimal algorithm. Caching is therefore more valuable
for BF16 than FP32.

---

## 12. Adding a New Type to an Existing Op — Checklist

When adding support for a new `TensorDataType` (e.g. BF16) to an existing op:

- [ ] `Op.Bf16.cu` — kernel + launcher, exports `kOpBf16VectorWidth`
- [ ] `CudaOp.Dispatch.ixx` — add complete `cuda_op_impl<__nv_bfloat16>` specialization
- [ ] `CudaOp.Registrar.ixx` — add `registerUnaryOperation<Cuda, BF16, BF16>` entry
- [ ] `CudaOp.ixx` `forward()` — add vector width validation against `kOpBf16VectorWidth`
- [ ] Validate kernel output against FP32 reference before enabling vectorization

All four must land together. Missing the registrar entry is a silent runtime omission.
Missing the dispatch specialization is a compile error.

---

*This document reflects design decisions made through April 2026.*
*Update when new types, ops, or training strategies are added.*