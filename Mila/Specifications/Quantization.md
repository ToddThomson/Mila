# Mila — Quantization Design

> **Status:** weight quantization shipped (FP8 per-channel, FP4 per-group, sub-4-bit codebook);
> KV cache compression designed, no policy type in the tree
> **Scope:** `Linear` component weight quantization; `GroupedQueryAttention` KV cache compression

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
            Policies.ixx            — NoWeightQuant, PerChannelFp8<>, PerGroupFp4<>,
                                      PerGroupCodebook2<>/3<>; WeightQuantPolicy concept
            CodebookPacking.ixx     — normative packed layout + CPU reference codec
            PrecisionPlan.ixx       — per-role policy table for a block (Qwen3.8)
        KvCache/
            Policy.ixx              — KvCachePolicy concept; NoKvCompression identity struct
            QuantPolicy.ixx         — PerChannelKvFp8<>; satisfies KvCachePolicy
```

The `Weight/` layout is the target shape as well as the current one: a policy, the codec
that states its bytes, and nothing that runs a forward pass. `Fp4Packing.ixx` and
`Fp8Packing.ixx` are the two files it is missing.

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

Weight quantization is **one-way and offline**: the packed weights and their scales are
decided when an artifact is built, and a load uploads bytes that are already what they
will be. Only the `Linear` component quantizes. The full-precision source is never
retained on device, and after the change described in *Fitting is offline, encoding is a
codec* below it is not required on the machine that runs the model at all.

Quantize-on-load is the shipped mechanism and the one this section's pipeline still
describes. It is a transitional path, not the design of record — see that section for
what replaces it and why.

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

### Load Pipeline

Two shapes reach the same device state. `Linear::loadParameter` picks between them on the
blob's dtype: storage dtype means the bytes are already packed and the scales arrive as
their own tensor; compute precision means a full-precision source that must be fitted here
(`Linear.ixx:601`). Re-quantizing packed bytes would read nibbles as BF16 and produce a
model that runs and is wrong, so the two must never be confused.

**Pre-quantized (the target shape).** No fitting, no staging buffer, no device pass:

```
fromPretrained()
    └── PretrainedModelReader — __metadata__["mila_quantization"] names the policy
    └── the model refuses an artifact whose policy is not the one this build compiled
    └── initializeParameters( reader )
            └── loadParameter( "weight",        blob )  -> direct upload, packed layout
            └── loadParameter( "weight_scale",  blob )  -> direct upload
            └── loadParameter( "weight_codebook"/"weight_high_plane", ... )  -- format permitting
                    └── operation_->onQuantizedWeightsLoaded()
```

**Quantize-on-load (transitional, FP8 and FP4 only).** Reads a full-precision blob and
fits it on device:

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

### Fitting is offline, encoding is a codec

Decided 2026-08-19. "Quantization" names two operations with nothing in common, and
conflating them is why the weight quantizer ended up bolted to an inference operation:

- **Fitting** — choosing the scales, the codebook, the assignment of each weight to a
  code. Data-dependent in general; data-free only for the absmax formats.
- **Encoding** — value to code, code to bytes: layout, bit order, scale dtype.
  Deterministic, bit-exact, and checkable without a device.

**Fitting is offline for every format.** FP8 and FP4 acquired a load-time fitter because
absmax is cheap enough to hide inside `loadParameter()`, not because that was where it
belonged. Two consequences follow that the load-time form cannot deliver:

1. **Provenance.** An artifact's bytes are fixed and hashed at package time, so the model
   card's claim about what the weights are describes something a third party can verify.
   Weights fitted in the user's process on their device are unreproducible by construction,
   and no manifest can describe them.
2. **The format stops being bounded by what a load-time kernel can do.** While fitting had
   to run data-free during a load, the format could only ever be absmax rounding. The Qwen3.8
   precision plan — per-tensor codebooks fitted with Hessian-diagonal importance and a
   compensated column walk — cannot exist under that constraint, which is why the sub-4-bit
   work went offline the moment it was real (`Qwen3.8.md` §8, *Converter quantization pipeline*).

The rule does not extend to the KV cache. That compression is genuinely runtime and
per-token; see Part III.

**Encoding is a normative codec, owned by `Quantization/Weight/`.** The model is
`CodebookPacking.ixx`, which states the packed layout once and resolves any disagreement
between the CUDA kernels and the Python packer in its own favor — a generated fixture holds
both to it in both directions. FP4 and FP8 have no such file: `cuda_quantize_fp4_per_group`
is the only place the nibble order and the `/6.0f` scale convention are written down, which
makes the wire format of two shipped artifacts unreadable from CPU-only CI and unstatable to
anyone reading the spec. That is the defect underneath "the quantizer lives on the operation",
and the codec files are the fix.

Once encoding is normative, the fitter is an implementation choice rather than a format
decision: the CUDA absmax path may stay as an optimization of the export tool, or move to
Python beside the codebook packer, without either changing a byte on disk.

### Where the tooling lives

Decided 2026-08-19. **The fitter is Python and stays Python**, for every family — not only for
the ones Mila cannot yet run. Three reasons, and the first is the weakest:

1. GPTQ accumulates Hessians from activations produced by a reference forward pass, so a C++
   fitter would require Mila to run an architecture before it could quantize it.
2. **The ordering forbids it even where the chassis exists.** A new architecture's artifact must
   exist before its kernels can be validated — `Qwen3.8.md` §8 designs the converter pipeline when
   it does precisely because Phase 1 needs its output as the oracle. C++ can never be at the head
   of that chain, so a C++ fitter is only ever available for the families that finished it, which
   are the families that need it least.
3. **Mila's forward must not be its own calibration oracle.** Calibration decides which channels
   matter by watching activations; taking them from the implementation about to be validated lets
   a bug in that forward shape the codebook it is then measured against. The same reason Gemma
   parity used HF's `output_hidden_states` rather than Mila's own numbers.

The Phase 0 research is the evidence: all of it ran on Llama 3.2 3B, a family Mila fully
supports, and it ran in Python regardless.

That machinery is general, not Qwen's, and lives in **`Tools/Quantization`**: `formats.py` (level
sets, grouping, codebook fitting), `fit.py` (calibration and sequential GPTQ), `artifact.py`
(Mila-named emission), `evaluate.py` (the harness that gates a scheme), `packing.py` (the codec),
and a command line that only orchestrates. The scheme tables are keyed on HuggingFace module
suffixes and are the one part that knows which family it is looking at.

`Tools/ExportArtifact` becomes **`mila-compress`** and narrows to the artifact: export,
fingerprint, transcode, package. The local-store verbs it accumulated — install, rename,
validate — move to `mila`, which owns the store. `ExportArtifact --install` and `mila install`
are today the same word for two different operations (adopt a local package; download a
published model), and the split resolves that.

The end state has one producer per stage: Python fits and encodes, `mila-compress` packages,
`mila` installs and serves. Whether `mila-compress` then merges into `mila` is deferred — it
is only a clean question once the fitter has left C++ and the two share a build gate.

### Where the quantizer lives — and where it does not

Two shapes were considered and rejected, recorded so they are not re-proposed:

- **A new `OperationType`.** An `Operation` in Mila has a `forward()`, resolves through
  `OperationTraits` on device x precision x policy, binds 1:1 to a component, and carries the
  build/`setParameters`/`setGradients` lifecycle. A weight quantizer has none of that: it runs
  exactly once, holds no per-call state, and has no component. Registering it would place a
  load-time producer permanently in the inference dispatch table — encoding "quantization is
  part of inference" in the type system at the moment that stopped being true — and would add
  an entry to a surface `OperationDispatch.md` is narrowing.
- **A tensor op.** `TensorOps` is elementwise and copy work over `Tensor<T, MR>`. This is not
  that: the input is a host `ITensorBlob` owned by a reader, and the output is two or three
  tensors in a policy-defined relationship — packed nibbles at halved physical columns, scales
  at `[out, in / group]`, and for a codebook a table and a high plane. Calling that a tensor op
  flattens the only part that matters, which is the layout contract.

The codec is neither. It is a peer of the policy that defines it, and it lives beside it.

### What stays on the operation

`onQuantizedWeightsLoaded()` (`CudaLinearOp.ixx:402`) is not quantization and does not move.
It derives `weight_fp8_scale_` from the group scales — a forward-path scalar that exists
because of how this kernel stages weights, and which nothing else computes. It is the general
hook every storage format needs: *the tensors have landed, derive what forward requires.* Its
name undersells that.

The `:Quantize` partition mostly survives as well. It is already a non-template NVCC bridge
with no dependence on `CudaLinearOp` state, so it is re-homed rather than rewritten — what it
gains is a layout file to be checked against.

**The end state for `Linear::loadParameter` is one shape for every policy**: refuse a
compute-precision blob, upload the packed bytes, bind, derive. The codebook path already has
it (`Linear.ixx:574`), and FP4/FP8 converge onto it. The dtype sniff at `:601` disappears, and
with it the class of defect it guards against.

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
- **Activation quantization as a policy axis.** Activations stay at compute precision as
  far as the type system is concerned. The FP8 prefill path quantizes activations inside
  one kernel as a private staging decision (`Fp8ActivationPrefill.md`); it is not a policy,
  and nothing outside that kernel can observe it.
- **Asymmetric K/V compression.** K and V use the same policy symmetrically.
- **Fitting weights at load time.** Superseded — see *Fitting is offline, encoding is a
  codec*. Quantize-on-load survives as a transitional FP8/FP4 path only.
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
