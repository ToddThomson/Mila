# Mila — Quantization Design v2

> **Status:** Alpha.5 — In Progress (weight quantization); Alpha.6 — Planned (KV cache compression)
> **Scope:** `Linear` component weight quantization; `GroupedQueryAttention` KV cache compression
> **Supersedes:** Quantization Design v1

---

## Design Philosophy

Mila's core principle is **type safety at compile time**. Device type and precision are
template parameters — mixing them is a compile error, not a runtime surprise. This
principle applies fully to quantization and cache compression at the component and
operation level.

Weight quantization and KV cache compression are compile-time deployment decisions
at the `Linear` and `GroupedQueryAttention` level, expressed as template parameters.
The framework enforces this statically — there is no runtime dispatch on quantization
state in the hot path.

Above the component level, quantization is a **deployment configuration** expressed
via `ModelConfig`. The runtime→compile-time bridge is owned entirely by
`fromPretrained()` and is an implementation detail invisible to the caller.

---

## Architectural Boundary

```
Client Code
    XxxModelConfig                  — deployment configuration, runtime values

Mila Public API
    LanguageModel<TDevice, TPrecision>
    LlamaModel<TDevice, TPrecision>::fromPretrained( path, config, device )
                                    — runtime→compile-time bridge (implementation detail)

Mila Internal — Model Layer
    LlamaModel<TDevice, TPrecision, TWeightQuant, TKvPolicy>
    LlamaTransformer<TDevice, TPrecision, TWeightQuant, TKvPolicy>
    LlamaDecoderLayer<TDevice, TPrecision, TWeightQuant, TKvPolicy>

Mila Internal — Component Layer
    Linear<TDevice, TPrecision, TWeightQuant>
    GroupedQueryAttention<TDevice, TPrecision, TKvPolicy>

Mila Internal — Operation Layer
    CudaLinearOp<TPrecision, TWeightQuant>
    CudaGqaOp<TPrecision, TKvPolicy>
```

**Template parameters on model types** express what the model *is*:
- `TDeviceType` — target device
- `TComputePrecision` — numeric identity of the entire model; pervasive through every component and operation

**`ModelConfig` fields** express how the model is *deployed*:
- `WeightQuantization` — weight storage and matmul strategy for `Linear`
- `KvCacheCompression` — KV cache storage and compression strategy for `GroupedQueryAttention`
- `context_length` — maximum sequence length

`TComputePrecision` remains a template parameter on all model types. It flows through
every tensor type, operation plan, and cuBLASLt descriptor in the stack.
`LlamaModel<Cuda, BF16>` and `LlamaModel<Cuda, FP32>` are genuinely distinct types.
This is not a deployment option — it is the numeric identity of the model.

---

## Module Layout

```
Src/
    Models/
        Config/
            ModelConfig.ixx         — ModelConfig<TDerived> CRTP base;
                                      WeightQuantization enum;
                                      KvCacheCompression enum;
                                      all fluent base methods
            LlamaModelConfig.ixx    — LlamaModelConfig : ModelConfig<LlamaModelConfig>
            QwenModelConfig.ixx     — QwenModelConfig : ModelConfig<QwenModelConfig>
            MistralModelConfig.ixx  — MistralModelConfig : ModelConfig<MistralModelConfig>
            GemmaModelConfig.ixx    — GemmaModelConfig : ModelConfig<GemmaModelConfig>

    Dnn/Quantization/
        Weight/
            Policies.ixx            — NoWeightQuant, PerChannelFp8<>; WeightQuantPolicy concept
            Quantizer.ixx           — IWeightQuantizer concept (future: pluggable host-side quantizers)
        KvCache/
            Policy.ixx              — KvCachePolicy concept; NoKvCompression identity struct
            QuantPolicy.ixx         — PerChannelKvFp8<>; satisfies KvCachePolicy
```

---

## Part I — Model Configuration

### `ModelConfig<TDerived>` — CRTP Base

`ModelConfig<TDerived>` is the shared base for all concrete model configs. It owns
the quantization enums and all fluent methods. Derived types inherit and extend with
model-specific fields and methods, with fluent chains that return `TDerived&` throughout.

```cpp
export enum class WeightQuantization
{
    None,           // BF16 weights — default
    FP8,            // PerChannelFp8<> — Alpha.5
    FP4,            // future
};

export enum class KvCacheCompression
{
    None,           // no compression — default
    FP8,            // PerChannelKvFp8<> — Alpha.6
    // future: SlidingWindow, LowRank, TurboQuant, ...
};

export template<typename TDerived>
struct ModelConfig
{
    size_t             context_length{ 0 };
    WeightQuantization weight_quantization{ WeightQuantization::None };
    KvCacheCompression kv_cache_compression{ KvCacheCompression::None };

    // Fine-grained control
    TDerived& withContextLength( size_t length )
    {
        context_length = length;
        return static_cast<TDerived&>( *this );
    }

    TDerived& withWeightQuantization( WeightQuantization wq )
    {
        weight_quantization = wq;
        return static_cast<TDerived&>( *this );
    }

    TDerived& withKvCacheCompression( KvCacheCompression kv )
    {
        kv_cache_compression = kv;
        return static_cast<TDerived&>( *this );
    }

    // Convenience presets
    TDerived& withFullPrecision()
    {
        weight_quantization  = WeightQuantization::None;
        kv_cache_compression = KvCacheCompression::None;
        return static_cast<TDerived&>( *this );
    }

    TDerived& withFP8Quantization()
    {
        weight_quantization  = WeightQuantization::FP8;
        kv_cache_compression = KvCacheCompression::FP8;
        return static_cast<TDerived&>( *this );
    }

    TDerived& withFP4Quantization()
    {
        weight_quantization  = WeightQuantization::FP4;
        kv_cache_compression = KvCacheCompression::FP8;
        return static_cast<TDerived&>( *this );
    }
};
```

### Concrete Model Configs

Each concrete model config inherits `ModelConfig<TDerived>` and adds model-specific
fields. Fluent chains work across both base and derived methods without casting.

```cpp
// Llama — no model-specific fields currently
export struct LlamaModelConfig : ModelConfig<LlamaModelConfig>
{
    LlamaModelConfig() = default;
    explicit LlamaModelConfig( size_t ctx ) { context_length = ctx; }
};

// Qwen3 — thinking mode
export struct QwenModelConfig : ModelConfig<QwenModelConfig>
{
    QwenModelConfig() = default;
    explicit QwenModelConfig( size_t ctx ) { context_length = ctx; }

    QwenModelConfig& withThinkingMode()
    {
        enable_thinking = true;
        return *this;
    }

    bool enable_thinking{ false };
};

// Gemma / MoE models
export struct GemmaModelConfig : ModelConfig<GemmaModelConfig>
{
    GemmaModelConfig() = default;
    explicit GemmaModelConfig( size_t ctx ) { context_length = ctx; }

    GemmaModelConfig& withMoEConfig( size_t num_experts, size_t active_experts )
    {
        num_experts_    = num_experts;
        active_experts_ = active_experts;
        return *this;
    }

    size_t num_experts_{ 0 };
    size_t active_experts_{ 0 };
};
```

### Client Usage

```cpp
// Standard BF16 inference — no quantization
LlamaModelConfig config = LlamaModelConfig( context_length );

// FP8 weights + FP8 KV cache — convenience preset
LlamaModelConfig config = LlamaModelConfig( context_length )
    .withFP8Quantization();

// FP4 weights, no KV compression — fine-grained control
LlamaModelConfig config = LlamaModelConfig( context_length )
    .withWeightQuantization( WeightQuantization::FP4 )
    .withKvCacheCompression( KvCacheCompression::None );

// Qwen3 with thinking mode and FP8
QwenModelConfig config = QwenModelConfig( context_length )
    .withFP8Quantization()
    .withThinkingMode();
```

### `fromPretrained` — Runtime→Compile-Time Bridge

The mapping from `ModelConfig` runtime enums to template instantiations is owned
entirely by `fromPretrained()`. This is an implementation detail — the caller holds
only `unique_ptr<LanguageModel<TDevice, TPrecision>>` and is unaware of `TWeightQuant`
or `TKvPolicy`.

The internal dispatch pattern is a `fromPretrainedImpl<TWeightQuant, TKvPolicy>()`
private static, called from `fromPretrained()` after resolving the config enums.
See implementation notes.

### Deployment Configurations

| Preset | `WeightQuantization` | `KvCacheCompression` | `TWeightQuant` | `TKvPolicy` |
|---|---|---|---|---|
| Full precision | `None` | `None` | `NoWeightQuant` | `NoKvCompression` |
| FP8 | `FP8` | `FP8` | `PerChannelFp8<>` | `PerChannelKvFp8<>` |
| FP4 | `FP4` | `FP8` | `PerGroupFp4<>` (future) | `PerChannelKvFp8<>` |
| FP32 reference | `None` | `None` | `NoWeightQuant` | `NoKvCompression` |

---

## Part II — Linear Weight Quantization

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
template<>
struct LinearOpTypeMap<DeviceType::Cuda, TensorDataType::kBF16, NoWeightQuant>
{
    using op_type = Cuda::Linear::CudaLinearOp<TensorDataType::kBF16, NoWeightQuant>;
};

template<>
struct LinearOpTypeMap<DeviceType::Cuda, TensorDataType::kBF16, PerChannelFp8<>>
{
    using op_type = Cuda::Linear::CudaLinearOp<TensorDataType::kBF16, PerChannelFp8<>>;
};
```

### Ownership Model

```cpp
std::unique_ptr<WeightTensorType> weight_{ nullptr };         // Tensor<kWeightDtype, MR>
std::unique_ptr<TensorType>       bias_{ nullptr };           // Tensor<TComputePrecision, MR> — optional
std::unique_ptr<TensorType>       weight_scales_{ nullptr };  // Tensor<Float32, MR> — kIsQuantized only
```

`weight_scales_` is allocated and populated only when `kIsQuantized` is true.

### Quantization Pipeline

```
fromPretrained()
    └── build( build_context )
            └── CudaLinearOp<BF16, PerChannelFp8<>> constructed
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

### Operation Base Class Contract

```cpp
// Universal contract — all operations
virtual void setParameters( ITensor* weight, ITensor* bias ) {}
virtual void setGradients( ITensor* weight_grad, ITensor* bias_grad ) {}

// CudaLinearOp only — not on the base class
void setWeightScales( ITensor* scales );
void quantize( const ITensorBlob& blob,
               ITensor& weight_out,
               ITensor& scales_out,
               const shape_t& expected_shape );
```

`setWeightScales()` and `quantize()` are concrete methods on `CudaLinearOp` only.
Non-quantized operations are entirely unaware they exist.

---

## Part III — KV Cache Compression

### Strategy

KV cache compression is a **runtime, per-token** operation applied to the K and V
tensors in `GroupedQueryAttention`. Unlike weight quantization — which is load-time and
permanent — cache compression happens on every prefill chunk write and every decode
append. The compressed representation is what is stored; dequantization happens at read
time before attention score computation.

**FP8 format:** `E4M3`, consistent with weight quantization.

**Scale granularity:** Per-head per-token. Scale tensors have shape
`[num_kv_heads, max_seq_len]`.

**Dequantization strategy:** At read time, immediately before use in attention score
and weighted-sum computation. Dequantized values are transient — never written back.

### Policy Structs

```cpp
namespace Mila::Dnn::Quant::KvCache
{
    template<typename T>
    concept KvCachePolicy = requires
    {
        { T::kIsActive } -> std::convertible_to<bool>;
    };

    struct NoKvCompression
    {
        static constexpr bool kIsActive = false;
    };

    template<TensorDataType TStorage = TensorDataType::kFp8E4M3>
    struct PerChannelKvFp8
    {
        static constexpr bool kIsActive               = true;
        static constexpr TensorDataType kStorageDtype = TStorage;
        static constexpr TensorDataType kScaleDtype   = TensorDataType::kFloat32;
        static constexpr bool kPerHeadPerToken        = true;
        static constexpr bool kSymmetric              = true;
    };
}
```

`KvCachePolicy` is intentionally minimal — it does not require `kStorageDtype` or
`kScaleDtype`. A future `SlidingWindowPolicy` satisfies `KvCachePolicy` without
carrying dtype fields. New compression algorithms extend `KvCacheCompression` enum
and add a corresponding policy struct and `fromPretrained` branch — no other
changes required.

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

    static constexpr TensorDataType kCacheDtype = kKvCompressed
        ? TKvPolicy::kStorageDtype
        : TComputePrecision;

    using KvCacheTensorType = Tensor<kCacheDtype, MR>;
};
```

#### Type Map

```cpp
template<>
struct GroupedQueryAttentionOpTypeMap<DeviceType::Cuda, TensorDataType::kBF16, NoKvCompression>
{
    using op_type = Cuda::Gqa::CudaGqaOp<TensorDataType::kBF16, NoKvCompression>;
};

template<>
struct GroupedQueryAttentionOpTypeMap<DeviceType::Cuda, TensorDataType::kBF16, PerChannelKvFp8<>>
{
    using op_type = Cuda::Gqa::CudaGqaOp<TensorDataType::kBF16, PerChannelKvFp8<>>;
};
```

### Ownership Model

```cpp
std::unique_ptr<KvCacheTensorType> k_cache_{ nullptr };   // Tensor<kCacheDtype, MR>
std::unique_ptr<KvCacheTensorType> v_cache_{ nullptr };   // Tensor<kCacheDtype, MR>

// kKvCompressed only
std::unique_ptr<TensorType> k_scale_{ nullptr };   // Tensor<Float32, MR> [num_kv_heads, max_seq_len]
std::unique_ptr<TensorType> v_scale_{ nullptr };   // Tensor<Float32, MR> [num_kv_heads, max_seq_len]
```

### `CudaGqaOp` Interface Extension

```cpp
// Concrete method on CudaGqaOp — not virtual on the base class
void setKvScales( ITensor* k_scale, ITensor* v_scale );
```

---

## Part IV — Quantization Scope

Weight quantization and KV cache compression are the complete quantization scope for
Mila inference. The components that benefit are well understood:

| Component | Quantization | Rationale |
|---|---|---|
| `Linear` | `TWeightQuant` | Dominant VRAM consumer; cuBLASLt FP8 native support |
| `GroupedQueryAttention` | `TKvPolicy` | KV cache is the dominant memory pressure at long context |
| `TokenEmbedding` | None | Lookup operation — not a matmul; quantization gives negligible benefit |
| `RmsNorm` / `LayerNorm` | None | Small vectors; FP8 precision loss actively harmful |
| `SwiGLU` / activation layers | None | Element-wise on activations; activations remain at compute precision |

MoE gating matmuls, when added, are `Linear` under the hood and inherit `TWeightQuant`
naturally. No new quantization scope is anticipated before beta.

---

## What Was Removed / Superseded

### `WeightQuantMode` / `KvCacheMode` internal enums (v1 proposal)

Replaced by `WeightQuantization` and `KvCacheCompression` on `ModelConfig`. The
internal enum intermediary layer was unnecessary — `fromPretrained` maps `ModelConfig`
fields directly to template instantiations.

### `QuantizationPreset` flat enum (v1 proposal)

Replaced by independent `WeightQuantization` and `KvCacheCompression` axes on
`ModelConfig`, with convenience preset methods (`withFP8Quantization()` etc.) on
`ModelConfig<TDerived>`. The flat enum would have broken with any algorithm that
does not fit the "weight + KV as a bundle" model (e.g. `TurboQuant`, sliding window,
low-rank KV projection).

### `QuantizationConfig` from `BuildContext`

Removed. `Linear` knows at compile time whether it is quantized via `kIsQuantized`.
`BuildContext` was redundantly carrying a runtime value for a statically-known fact.

### `TKvCache` bare `TensorDataType` on `GroupedQueryAttention`

Replaced by `TKvPolicy = NoKvCompression`. The bare dtype conflated storage type with
compression algorithm.

### `TWeight` bare `TensorDataType` on `Linear`

Replaced by `TWeightQuant = NoWeightQuant`.

---

## Non-Goals

- **Runtime quantization toggling.** A `Linear` or `GroupedQueryAttention` instance is
  either quantized/compressed or it is not. Fixed at compile time.
- **Activation quantization.** Only weights and KV cache values are quantized.
  Activations remain at compute precision throughout.
- **Per-group weight scales.** Per-channel scales are the current target.
- **Asymmetric K/V compression.** K and V use the same policy symmetrically.
- **FP16 support.** BF16 supersedes FP16 for all Mila compute targets.
- **Training with quantized weights or compressed KV cache.** Both are inference
  optimizations.
- **MLA / low-rank KV projection.** Deferred; `KvCachePolicy` accommodates it.
- **Sliding window attention / cache eviction.** Deferred to Alpha.7 / Ministral;
  `KvCachePolicy` concept accommodates it without signature changes on
  `GroupedQueryAttention`.
- **FP4 KV cache.** FP4 weight quantization is a planned target; FP4 KV cache
  compression is not — the quality/complexity tradeoff is unfavorable at current
  context lengths.
