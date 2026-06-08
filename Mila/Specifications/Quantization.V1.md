# Mila — Quantization Design

> **Status:** Alpha.5 — In Progress (weight quantization); Alpha.6 — Planned (KV cache compression)
> **Scope:** `Linear` component weight quantization; `GroupedQueryAttention` KV cache compression

---

## Design Philosophy

Mila's core principle is **type safety at compile time**. Device type and precision are
template parameters — mixing them is a compile error, not a runtime surprise. This
principle applies fully to quantization and cache compression.

Weight precision and KV cache compression are **not** runtime configuration concerns.
They are compile-time deployment decisions, expressed as template parameters. The
framework enforces this statically. There is no runtime dispatch on quantization state.

---

## Module Layout

```
Src/Dnn/Quantization/
    Weight/
        Policies.ixx        — NoWeightQuant, PerChannelFp8<>; WeightQuantPolicy concept
        Quantizer.ixx       — IWeightQuantizer concept (future: pluggable host-side quantizers)
    KvCache/
        Policy.ixx          — KvCachePolicy concept; NoKvCompression identity struct
        QuantPolicy.ixx     — PerChannelKvFp8<>; satisfies KvCachePolicy
```

`Linear` imports from `WeightQuant/` only. `GroupedQueryAttention` imports from
`KvCache/` only. Neither subsystem knows the other exists. Shared primitive vocabulary
(dtype constants, scale granularity) lives in each policy struct's static members —
no shared `Primitives.ixx` is required at this scope.

---

## Part I — Linear Weight Quantization

### Strategy

FP8 quantization is a **load-time, one-way weight compression** strategy applied
exclusively to the `Linear` component. Weights are loaded from a BF16 pretrained
checkpoint and quantized to FP8_E4M3 during `initializeParameters()`. After
quantization, only the FP8 representation lives on device. The BF16 source is never
retained.

**FP8 format:** `E4M3` (`__nv_fp8_e4m3`). Higher precision (more mantissa bits) is
correct for stored weights. `E5M2` (wider dynamic range) is reserved for gradients and
is not a current Mila target.

**Scale granularity:** Per output channel. One `float32` scale per output feature,
derived from the maximum absolute value in that channel. This is standard for LLM weight
quantization and is natively supported by cuBLASLt.

**Dequantization strategy:** None at runtime. cuBLASLt native FP8 matmul is used —
weights remain FP8 through the entire operation. No transient BF16 weight copy per
forward pass.

### Policy Structs — `WeightQuant/Policies.ixx`

```cpp
namespace Mila::Dnn::Quant::Weight
{
    struct NoWeightQuant
    {
        static constexpr bool kIsQuantized            = false;
        static constexpr TensorDataType kStorageDtype = TensorDataType::kUndefined;
        static constexpr TensorDataType kScaleDtype   = TensorDataType::kUndefined;
    };

    template<TensorDataType TStorage = TensorDataType::kFp8E4M3>
    struct PerChannelFp8
    {
        static constexpr bool kIsQuantized            = true;
        static constexpr TensorDataType kStorageDtype = TStorage;
        static constexpr TensorDataType kScaleDtype   = TensorDataType::kFloat32;
        static constexpr bool kPerChannel             = true;
    };

    template<typename T>
    concept WeightQuantPolicy = requires
    {
        { T::kIsQuantized } -> std::convertible_to<bool>;
        { T::kStorageDtype } -> std::convertible_to<TensorDataType>;
        { T::kScaleDtype }   -> std::convertible_to<TensorDataType>;
    };
}
```

### Type System

Weight precision is encoded as a template parameter at every level of the stack.

#### Component

```cpp
export template<
    DeviceType          TDeviceType,
    TensorDataType      TComputePrecision,
    WeightQuantPolicy   TWeightQuant = NoWeightQuant>
    requires PrecisionSupportedOnDevice<TComputePrecision, TDeviceType>
class Linear : public Component<TDeviceType, TComputePrecision>
{
public:
    static constexpr bool kIsQuantized = TWeightQuant::kIsQuantized;

    static constexpr TensorDataType kWeightDtype = kIsQuantized
        ? TWeightQuant::kStorageDtype
        : TComputePrecision;

    using WeightTensorType = Tensor<kWeightDtype, MR>;
};
```

#### Operation

```cpp
export template<
    TensorDataType    TComputePrecision,
    WeightQuantPolicy TWeightQuant = NoWeightQuant>
    requires PrecisionSupportedOnDevice<TComputePrecision, DeviceType::Cuda>
class CudaLinearOp : public Operation<DeviceType::Cuda, TComputePrecision>
{
    static constexpr bool kIsQuantized = TWeightQuant::kIsQuantized;
};
```

#### Type Map

```cpp
// Unquantized — default for all compute precisions
template<>
struct LinearOpTypeMap<DeviceType::Cuda, TensorDataType::kBF16, NoWeightQuant>
{
    using op_type = Cuda::Linear::CudaLinearOp<TensorDataType::kBF16, NoWeightQuant>;
};

// FP8 per-channel — Alpha.5 quantized path
template<>
struct LinearOpTypeMap<DeviceType::Cuda, TensorDataType::kBF16, PerChannelFp8<>>
{
    using op_type = Cuda::Linear::CudaLinearOp<TensorDataType::kBF16, PerChannelFp8<>>;
};
```

#### Deployment Configurations

| Configuration | `TComputePrecision` | `TWeightQuant` | `kIsQuantized` |
|---|---|---|---|
| Reference / validation | `FP32` | `NoWeightQuant` | `false` |
| Standard inference | `BF16` | `NoWeightQuant` | `false` |
| Quantized inference | `BF16` | `PerChannelFp8<>` | `true` |

### Ownership Model

`Linear` owns all parameters. The operation receives pointers to component-owned tensors
and must not free them.

```cpp
std::unique_ptr<WeightTensorType> weight_{ nullptr };         // Tensor<kWeightDtype, MR>
std::unique_ptr<TensorType>       bias_{ nullptr };           // Tensor<TComputePrecision, MR> — optional
std::unique_ptr<TensorType>       weight_scales_{ nullptr };  // Tensor<Float32, MR> — kIsQuantized only
```

`weight_scales_` is allocated and populated only when `kIsQuantized` is true. It is
`nullptr` for all non-quantized specializations.

### Quantization Pipeline

```
fromPretrained()
    └── build( build_context )
            └── operation_ constructed (CudaLinearOp<BF16, PerChannelFp8<>>)
                    └── cuBLASLt FP8 plan built (types known statically)
    └── initializeParameters( reader )
            └── loadParameter( "weight", blob )
                    └── if constexpr ( kIsQuantized )
                            └── operation_->quantize( blob, *weight_, *weight_scales_, expected_shape )
                        else
                            └── loadParameterFromBlob( "weight", blob, *weight_, expected_shape )
    └── setParameters called after load
            └── operation_->setParameters( weight_.get(), bias_.get() )
            └── if constexpr ( kIsQuantized )
                    └── operation_->setWeightScales( weight_scales_.get() )
```

#### `loadParameter()` on `Linear`

```cpp
void loadParameter( const std::string& name, const ITensorBlob& blob ) override
{
    if ( name == "weight" )
    {
        const shape_t expected_shape{ config_.getOutputFeatures(), config_.getInputFeatures() };

        if constexpr ( kIsQuantized )
        {
            operation_->quantize( blob, *weight_, *weight_scales_, expected_shape );
        }
        else
        {
            this->loadParameterFromBlob( "weight", blob, *weight_, expected_shape );
        }
    }
    else if ( name == "bias" )
    {
        // bias path unchanged — always at compute precision
    }
}
```

#### `quantize()` on `CudaLinearOp`

`quantize()` is a concrete method on `CudaLinearOp`, not a virtual method on the
operation base class. It is only callable through the resolved `OperationType`, which
`Linear` holds directly. No casting is required.

Responsibilities:

1. Walk the BF16 blob on the host; compute per-channel `float32` scales (`max(abs(W[o,:])) / 448.0f`)
2. Upload scales to `weight_scales_` on device
3. Quantize BF16 values to FP8_E4M3 using the computed scales
4. Upload quantized weights to `weight_` on device

Scale computation is host-side. This is a one-time load-time cost. No reduction kernel
is required.

### Operation Base Class Contract

```cpp
// Universal contract — all operations
virtual void setParameters( ITensor* weight, ITensor* bias ) {}
virtual void setGradients( ITensor* weight_grad, ITensor* bias_grad ) {}

// CudaLinearOp only — not on the base class
void setWeightScales( ITensor* scales );    // kIsQuantized path only
void quantize( const ITensorBlob& blob,
               ITensor& weight_out,
               ITensor& scales_out,
               const shape_t& expected_shape );
```

`setWeightScales()` and `quantize()` are not part of the base contract. Non-quantized
operations are entirely unaware they exist.

### cuBLASLt FP8 Plan

`CudaLinearOp<BF16, PerChannelFp8<>>::build()` constructs the cuBLASLt plan for FP8
inputs at build time. All type information is statically known from template parameters.
The plan references scale descriptor pointers, populated by `setWeightScales()` after
`initializeParameters()` completes. No plan rebuild is required after weight loading.

The Ada Lovelace architecture (RTX 4070) has native FP8 tensor core support. cuBLASLt
operations with `CUDA_R_8F_E4M3` weight type and per-column float32 scales run without
dequantization overhead.

---

## Part II — KV Cache Compression

### Strategy

KV cache compression is a **runtime, per-token** operation applied to the K and V
tensors in `GroupedQueryAttention`. Unlike weight quantization — which is load-time and
permanent — cache compression happens on every prefill chunk write and every decode
append. The compressed representation is what is stored; dequantization happens at read
time before attention score computation.

The compression policy is symmetric: K and V use the same algorithm and storage dtype.
Asymmetric K/V compression is not a current target.

**FP8 format:** `E4M3`, consistent with weight quantization. The KV cache dtype and the
weight dtype are independent — both happen to use FP8_E4M3 for the Alpha.6 target.

**Scale granularity:** Per-head per-token. One `float32` scale per KV head per cached
token position (`scale[head, token] = max(abs(x[head, token, :])) / 448.0f`). This is
coarser than per-channel weight quantization but appropriate for the dynamic, growing
shape of the KV cache. Scale tensors have shape `[num_kv_heads, max_seq_len]`.

**Dequantization strategy:** At read time, immediately before use in attention score
and weighted-sum computation. The dequantized values are transient — they are never
written back to the cache.

### Policy Structs — `KvCache/Policy.ixx` and `KvCache/QuantPolicy.ixx`

```cpp
namespace Mila::Dnn::Quant::KvCache
{
    // Concept — all KV cache policies satisfy this
    template<typename T>
    concept KvCachePolicy = requires
    {
        { T::kIsActive } -> std::convertible_to<bool>;
    };

    // Identity — no compression; zero-cost; default for all GQA instances
    struct NoKvCompression
    {
        static constexpr bool kIsActive = false;
    };

    // Symmetric per-head per-token FP8 compression — Alpha.6 target
    template<TensorDataType TStorage = TensorDataType::kFp8E4M3>
    struct PerChannelKvFp8
    {
        static constexpr bool kIsActive               = true;
        static constexpr TensorDataType kStorageDtype = TStorage;
        static constexpr TensorDataType kScaleDtype   = TensorDataType::kFloat32;
        static constexpr bool kPerHeadPerToken        = true;
        static constexpr bool kSymmetric              = true;   // K and V use same policy
    };
}
```

`KvCachePolicy` is intentionally minimal. It does not require `kStorageDtype` or
`kScaleDtype` — those are constraints of the `QuantKvPolicy` refinement. A future
`SlidingWindowPolicy` satisfies `KvCachePolicy` without carrying dtype fields.

### Type System

#### Component

```cpp
export template<
    DeviceType      TDeviceType,
    TensorDataType  TComputePrecision,
    KvCachePolicy   TKvPolicy = NoKvCompression>
    requires PrecisionSupportedOnDevice<TComputePrecision, TDeviceType>
class GroupedQueryAttention : public Component<TDeviceType, TComputePrecision>
{
    static constexpr bool kKvCompressed = TKvPolicy::kIsActive;

    // Cache tensor dtype: NoKvCompression → TComputePrecision passthrough
    static constexpr TensorDataType kCacheDtype = kKvCompressed
        ? TKvPolicy::kStorageDtype
        : TComputePrecision;

    using KvCacheTensorType = Tensor<kCacheDtype, MR>;
};
```

#### Type Map

```cpp
// Unquantized — default
template<>
struct GroupedQueryAttentionOpTypeMap<DeviceType::Cuda, TensorDataType::kBF16, NoKvCompression>
{
    using op_type = Cuda::Gqa::CudaGqaOp<TensorDataType::kBF16, NoKvCompression>;
};

// FP8 KV cache — Alpha.6
template<>
struct GroupedQueryAttentionOpTypeMap<DeviceType::Cuda, TensorDataType::kBF16, PerChannelKvFp8<>>
{
    using op_type = Cuda::Gqa::CudaGqaOp<TensorDataType::kBF16, PerChannelKvFp8<>>;
};
```

#### Deployment Configurations

| Configuration | `TComputePrecision` | `TKvPolicy` | `kKvCompressed` | Est. KV VRAM |
|---|---|---|---|---|
| Reference / validation | `FP32` | `NoKvCompression` | `false` | baseline |
| Standard inference | `BF16` | `NoKvCompression` | `false` | baseline |
| KV-compressed inference | `BF16` | `PerChannelKvFp8<>` | `true` | ~50% of BF16 KV |

### Ownership Model

`GroupedQueryAttention` owns all cache and scale tensors. The operation receives pointers
and must not free them.

```cpp
// Existing — present for all configurations
std::unique_ptr<KvCacheTensorType> k_cache_{ nullptr };   // Tensor<kCacheDtype, MR>
std::unique_ptr<KvCacheTensorType> v_cache_{ nullptr };   // Tensor<kCacheDtype, MR>

// New — kKvCompressed only
std::unique_ptr<TensorType> k_scale_{ nullptr };   // Tensor<Float32, MR> [num_kv_heads, max_seq_len]
std::unique_ptr<TensorType> v_scale_{ nullptr };   // Tensor<Float32, MR> [num_kv_heads, max_seq_len]
```

`k_scale_` and `v_scale_` are allocated in `build()` only when `kKvCompressed` is true.
Their lifetime matches the KV cache tensors.

### Compression Pipeline — Prefill

On each prefill chunk write, K and V are quantized before being written to the cache:

```
prefill chunk → CudaGqaOp::prefill()
    └── if constexpr ( kKvCompressed )
            └── kv_write_quantize_kernel(
                    K_bf16, V_bf16,               // source: current chunk
                    k_cache_, v_cache_,            // dest: FP8 cache slices
                    k_scale_, v_scale_,            // dest: FP32 scale slices
                    chunk_len, num_kv_heads, head_dim )
        else
            └── kv_write_passthrough(...)          // existing path, unchanged
```

The quantize kernel computes `scale[head, token] = max(abs(x[head, token, :])) / 448.0f`
per head per token and writes both the FP8 values and the float32 scales to their
respective cache positions.

### Compression Pipeline — Decode

On each decode step, a single token's K and V are quantized and appended:

```
decode step → CudaGqaOp::decode()
    └── if constexpr ( kKvCompressed )
            └── kv_append_quantize_kernel(
                    K_bf16, V_bf16,               // source: current token
                    k_cache_, v_cache_,            // dest: FP8 cache at seq_pos
                    k_scale_, v_scale_,            // dest: scale at seq_pos
                    seq_pos, num_kv_heads, head_dim )
        else
            └── kv_append_passthrough(...)         // existing path, unchanged
```

### Dequantization at Read Time

Before attention score and weighted-sum computation, the compressed cache is
dequantized to BF16:

```
attention compute → CudaGqaOp::computeAttention()
    └── if constexpr ( kKvCompressed )
            └── kv_read_dequantize_kernel(
                    k_cache_, v_cache_,            // source: FP8 cache
                    k_scale_, v_scale_,            // source: FP32 scales
                    K_bf16_transient,              // dest: ephemeral BF16 buffers
                    V_bf16_transient,
                    seq_len, num_kv_heads, head_dim )
        else
            └── use k_cache_, v_cache_ directly    // existing path, unchanged
```

Dequantized `K_bf16_transient` and `V_bf16_transient` are ephemeral device buffers,
allocated in `build()` at `[num_kv_heads, max_seq_len, head_dim]` for BF16. They are
reused across decode steps. When `kKvCompressed` is false these buffers do not exist.

### `CudaGqaOp` Interface Extensions

```cpp
// Extended for kKvCompressed path — not on the base class
void setKvScales( ITensor* k_scale, ITensor* v_scale );   // called from build()
```

`setKvScales()` is a concrete method on `CudaGqaOp`, not virtual on the operation base
class. Non-compressed operations are entirely unaware it exists.

---

## What Was Removed / Superseded

### `TKvCache` bare `TensorDataType` on `GroupedQueryAttention`

The earlier `TKvCache = TComputePrecision` template parameter is replaced by
`TKvPolicy = NoKvCompression`. The bare dtype conflated storage type with compression
algorithm. `TKvPolicy` carries both, and extends naturally to non-quantization policies
(sliding window, low-rank projection) without a signature change on `GroupedQueryAttention`.

### `TWeight` bare `TensorDataType` on `Linear`

Similarly replaced by `TWeightQuant = NoWeightQuant`. `kIsQuantized` is now
`TWeightQuant::kIsQuantized` rather than `(TWeight != TComputePrecision)`. Semantics
are identical for the current targets; the policy form is unambiguous as additional
algorithms are added.

### `QuantizationConfig` from `BuildContext`

Removed. `Linear` knows at compile time whether it is quantized via `kIsQuantized`.
`BuildContext` was redundantly carrying a runtime value for a statically-known fact,
contradicting Mila's compile-time type safety principle. The template parameter
`TWeightQuant` is the sole source of truth for weight quantization state.

---

## Non-Goals

- **Runtime quantization toggling.** A `Linear` or `GroupedQueryAttention` instance is
  either quantized/compressed or it is not. This is fixed at compile time.
- **Activation quantization.** Only weights and KV cache values are quantized.
  Activations remain at compute precision (BF16) throughout.
- **Per-group weight scales.** Per-channel scales are the current target. Per-group
  (finer granularity) is not planned for Alpha.5.
- **Asymmetric K/V compression.** K and V use the same policy symmetrically. Per-tensor
  asymmetric policies are not a current target.
- **FP16 support.** BF16 supersedes FP16 for all Mila compute targets.
- **Training with quantized weights or compressed KV cache.** Both are inference
  optimizations. The training path is unaffected.
- **MLA / low-rank KV projection.** Architecturally compatible with the `KvCachePolicy`
  extension point but not a planned target before beta.
- **Sliding window attention / cache eviction.** Similarly deferred; the `KvCachePolicy`
  concept accommodates it when the time comes (Alpha.7 / Ministral).