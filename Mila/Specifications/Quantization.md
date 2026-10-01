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
`load()` and is an implementation detail invisible to the caller.

---

## Architectural Boundary

```
Client Code
    XxxModelConfig                  — deployment configuration, runtime values

Mila Public API
    LanguageModel<TDevice, TPrecision>
    LlamaModel<TDevice, TPrecision>::load( path, config, device )
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
            WeightQuantPolicy.ixx   — the concept every policy satisfies
            NoWeightQuant.ixx, PerChannelFp8.ixx, PerGroupInt4.ixx, PerGroupFp4.ixx,
            PerGroupCodebook2.ixx, PerGroupCodebook3.ixx
                                    — one policy per file
            HasCodebookTable.ixx, HasHighBitPlane.ixx
                                    — companion-tensor concepts
            Policies.ixx            — re-exports all of the above
            CodebookPacking.ixx     — normative packed layout + CPU reference codec (codebook)
            Int4Packing.ixx         — normative packed layout + CPU reference codec (INT4 / Q4_0)
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

    // Convenience presets. The weight presets set the weight format only; the KV cache is
    // the caller's separate choice (decided 2026-10-01: a preset that also set FP8 KV made
    // FP8 every quantized load's default without anyone asking for it).
    TDerived& withFullPrecision()
    {
        weight_quantization  = WeightQuantization::None;
        kv_cache_compression = KvCacheCompression::None;
        return static_cast<TDerived&>( *this );
    }

    TDerived& withFP8Quantization()
    {
        weight_quantization  = WeightQuantization::FP8;
        return static_cast<TDerived&>( *this );
    }

    TDerived& withFP4Quantization()
    {
        weight_quantization  = WeightQuantization::FP4;
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

// FP8 weights, BF16 KV cache — convenience preset
LlamaModelConfig config = LlamaModelConfig( context_length )
    .withFP8Quantization();

// FP4 weights, FP8 KV cache — the cache asked for separately
LlamaModelConfig config = LlamaModelConfig( context_length )
    .withFP4Quantization()
    .withKvCacheCompression( KvCacheCompression::FP8 );

// Qwen3 with thinking mode and FP8
QwenModelConfig config = QwenModelConfig( context_length )
    .withFP8Quantization()
    .withThinkingMode();
```

### `load` — Runtime→Compile-Time Bridge

The mapping from `ModelConfig` runtime enums to template instantiations is owned
entirely by `load()`. This is an implementation detail — the caller holds
only `unique_ptr<LanguageModel<TDevice, TPrecision>>` and is unaware of `TWeightQuant`
or `TKvPolicy`.

The internal dispatch pattern is a `loadImpl<TWeightQuant, TKvPolicy>()`
private static, called from `load()` after resolving the config enums.
See implementation notes.

### Deployment Configurations

| Preset | `WeightQuantization` | `TWeightQuant` |
|---|---|---|
| Full precision | `None` (and KV `None`) | `NoWeightQuant` |
| FP8 | `FP8` | `PerChannelFp8<>` |
| FP4 | `FP4` | `PerGroupFp4<>` |
| Q4_0 | `Q4_0` | `PerGroupInt4<32>` |

`KvCacheCompression` is set only by `withKvCacheCompression`: `None` is `NoKvCompression`; `FP8` is
`PerTokenKvFp8<>` on every Llama layer and Gemma's global layers (`dispatchKvCacheCompression`), on CUDA BF16
builds only, and Qwen refuses it.

---

## Part II — Linear Weight Quantization

### Strategy

Weight quantization is **one-way and offline**: the packed weights and their scales are
decided when an artifact is built, and a load uploads bytes that are already what they
will be. Only the `Linear` component quantizes. The full-precision source is never
retained on device, and after the change described in *Fitting is offline, encoding is a
codec* below it is not required on the machine that runs the model at all.

Quantize-on-load is a supported path for FP8 and FP4, decided 2026-09-14. It is how a user
runs full-precision weights that nobody has exported, and it is the exporter's own engine.
Published models are fitted offline (*Fitting is offline, encoding is a codec*); a load from
full-precision weights fits them in the user's process, with the provenance limit that section
states.

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
load()
    └── WeightsReader — __metadata__["mila_quantization"] names the policy
    └── the model refuses an artifact whose policy is not the one this build compiled
    └── initializeParameters( reader )
            └── loadParameter( "weight",        blob )  -> direct upload, packed layout
            └── loadParameter( "weight_scale",  blob )  -> direct upload
            └── loadParameter( "weight_codebook"/"weight_high_plane", ... )  -- format permitting
                    └── operation_->onQuantizedWeightsLoaded()
```

**Quantize-on-load (FP8 and FP4).** Reads a full-precision blob and fits it on device:

```
load()
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

**Staging belongs to the load, decided 2026-09-14.** Each fit copies its BF16 source to the
device through a staging buffer. Today that buffer is the execution context's forward scratch
(`getDeviceScratchBuffer`), which grows and never shrinks, so load-time staging stays on the
device for the life of the process: up to 256 MiB from the FP4 `Linear`, expert-bank and
embedding sites, which stage in row blocks under that ceiling, and the whole tensor from the FP8
per-channel `Linear` site (`CudaLinearOp.ixx:475`), which has no ceiling.

The staging buffer is the load's instead: a buffer of its own on the CUDA execution context
(`getLoadStagingBuffer`), grown during the load to its largest request and freed by
`releaseLoadStaging()` when each transformer's `loadParameters` returns. A model loaded from full-precision weights
then settles at the same footprint as its exported form. The buffer is still alive on top of the
built model while the last tensor is fitted, because `build()` allocates before
`loadParameters()`; `MemoryFootprint.md` section 11.4 accounts for that load peak separately.
Its size is fixed, 256 MiB (`kLoadStagingLimitBytes`). A smaller buffer costs more copies, not more
memory, and loaded no slower down to 16 MiB (`MemoryFootprint.md` Phase 6 step 1 result).

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

**Fitting is offline for every published model, and for every format below 4 bits.** FP8 and
FP4 also have a load-time fitter, because absmax is cheap enough to run inside
`loadParameter()`; since 2026-09-14 that fitter is a supported path for weights nobody has
exported, not a transitional one. Two consequences follow that the load-time form cannot
deliver:

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
decision: the CUDA absmax path stays, because quantize-on-load runs it in the user's process,
and any second fitter beside the codebook packer must produce the same bytes.

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

**`Linear::loadParameter` keeps two shapes for FP4 and FP8.** The end state recorded here on
2026-08-19 -- refuse a compute-precision blob, so FP4 and FP8 converge on the codebook path's
single shape (`Linear.ixx:574`) -- was withdrawn on 2026-09-14, when quantize-on-load became a
supported path. The dtype branch at `:601` stays, and so does the defect it guards against:
packed bytes must never be fitted as BF16. Codebook formats keep the single shape, since they
have no load-time fitter.

### Q4_0 — `PerGroupInt4<32>`

Decided 2026-09-27 (`ModelFamilyParity.md` §9 item 14; `BACKLOG.md`, the Gemma 4 QAT entry). A
producer's quantization-aware weights run in the format they were trained for, and Google trained
Gemma 4 QAT for Q4_0. Five decisions, each with the alternative it closed:

1. **The policy replaces the old `PerGroupInt4`.** The GPTQ-shaped policy -- FP32 scales, optional
   zero points, groups 64 and 128 -- had no model, no quantizer and no test, and its fused kernel
   `cuda_w4a16_gemm` launches nothing for a group outside {64, 128}. `PerGroupInt4<kGroupSize>`
   becomes symmetric only, with an FP16 scale; Q4_0 is `PerGroupInt4<32>`, 4.5 bits per weight.
   The zero-point path and `cuda_w4a16_gemm` are deleted. An int4 import (`Vnext.md`,
   compressed-tensors `pack-quantized`) is symmetric too, and maps here at its own group.
2. **Mila's two-plane layout, not the GGUF 18-byte block.** Codes `[out, in/2]` in Mila's nibble
   order, FP16 scales `[out, in/32]`, as every per-group format here: Linear's weight and
   `weight_scale` contract is unchanged, loads stay 16-byte aligned, and adjacent elements feed
   `bf16x2`. Equality with Google's GGUF is in value -- every code and every scale bit -- not in
   bytes. The normative layout and reference rounding are `Quantization/Weight/Int4Packing.ixx`.
3. **Users see `q4_0`**, the format's own name and the one in Google's repository:
   `WeightQuantization::Q4_0`, scheme name `q4_0`, Chat mode and binding variant `q4_0`.
4. **Prefill multiplies in INT8.** Activations are quantized to INT8 with one FP32 scale per
   32-element block -- scale `a / 127`, code `round(x * (127 / a))` for the block's largest magnitude
   `a` -- and one kernel multiplies them against the packed codes on the INT8 tensor cores. A k32 MMA
   is exactly one Q4_0 block, so each block's dot product is exact in INT32 and its two scales apply
   once, in FP32. The per-token FP8 activations of the FP4 prefill are not used: one scale per row and
   three mantissa bits, against one per 32 elements and seven. Per element the rounding is at most
   `a / 254`, near BF16's own for the block's large values. Decode has its own matvec: code times BF16 activation summed in FP32 per
   block, then times `d`, exact. The GEMM's K tile is 128 deep wherever 128 divides the input width and 64
   deep otherwise, so a width must be a multiple of 64 (since `0.21.0-dev+24`, for the Gemma 4 26B-A4B's
   2112-wide dense `fc_down`; the 64-deep tile measured slower on 2026-09-28).
   *Why not BF16* (decided 2026-09-28, replacing "prefill stages BF16 first"): the staged BF16 path
   -- codes expanded to BF16, then a cuBLASLt BF16 GEMM -- already ran its GEMMs at the BF16
   tensor-core ceiling (about 62 TFLOPS on the RTX 4070 against a measured `mma.sync` peak of 58.9),
   so no BF16 kernel could close its gap to llama.cpp; only the 11% expansion was left to win. INT8
   `mma.sync` issues at 232.5 TFLOPS on the 4070 and 207.5 on the RTX 5060 Ti, 4x BF16 on both
   (`Profiling/Microbenchmarks/MmaInstructionPeak.cu`). The staged path measured 1,430 tokens/s at
   an 8K Gemma 4 12B prompt against 2,385 for llama.cpp, and its BF16 weight strip (225 MiB at
   `fc_gate_up`) cost the 262144 context its 1024-token chunk; the INT8 path's scratch is the
   activations alone (15 MiB at 1024 x 15360). Kernel, 1024 rows, 2026-09-28: 92-108 TFLOPS on the
   4070 and 84-94 on the 5060 Ti at Gemma 4 12B and Llama 3.1 8B shapes; llama.cpp's `mul_mat_q`
   runs about 84 on the 4070. **End to end, 2026-09-28** (`ProfileModel --phase prefill`, chunk
   1024): Gemma 4 12B on the 5060 Ti 2,688 tokens/s at 8K and 1,950 at 32K, against llama.cpp's
   2,385 and 1,960 on Google's GGUF; Llama 3.1 8B on the 4070 4,748 at 8K and 3,862 at 16K, against
   4,370 and 3,675 on bit-identical weights. The 12B's prefill scratch is 18 MiB, and a 262144
   context plans a 1024-token chunk again.
5. **Gemma dense and Llama first.** Gemma's routed 26B refuses `q4_0` until its expert bank has
   an int4 path (`Gemma4MoE.md` Phase 9; Google ships the 26B-A4B's experts as Q4_0,
   `Gemma.md` §10.4), and Qwen's own dispatcher refuses it. Llama reaches it through the
   shared dispatcher, by quantize-on-load from BF16 weights.

**Rounding.** One rule, Q4_0's reference rounding, in three places that must agree bit for bit: the
CPU codec, the CUDA quantizer behind quantize-on-load, and therefore `ExportArtifact`. Two details
decide individual codes: the value is multiplied by the FP32 reciprocal of `d` rather than divided
by `d`, and the product is rounded before 8.5 is added -- a fused multiply-add moves codes on a
boundary, so the CUDA quantizer uses `__fmul_rn` and `__fadd_rn`. `d` carries the sign of the
group's extreme and may be negative -- an all-zero group stores negative zero, `0x8000` -- so no
kernel may assume it positive. Measured 2026-09-27: this rule reproduces every code and scale bit of
Google's GGUF (`ModelFamilyParity.md` 8.2, G2 result).

**Gate.** The codec's unit tests; the CUDA quantizer equal to the codec; Linear decode against a
host reference of exact weights, and prefill against a host reference of the INT8 block arithmetic; every Q4_0 tensor of the exported 12B equal in code and scale bits to
`google/gemma-4-12B-it-qat-q4_0-gguf` (fused projections split by rows; the global layers have no
`attn_v`); then G2 re-run on the Q4_0 build.

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

### Decisions, 2026-09-28 (Todd)

These supersede the sections below where they differ; the code follows them, and the sections are rewritten
when it lands.

1. **One E4M3 scale per KV head per token, computed at the write.** No calibration: Mila has no path that
   calibrates activations. The scale is 4 bytes against 128 to 512 bytes of values. The policy is not per channel,
   so `PerChannelKvFp8` is renamed `PerTokenKvFp8`, in its own module.
2. **The flash kernels read FP8 directly.** The packed prefill kernel and the fused decode kernel load FP8 tiles and
   dequantize them on chip before the BF16 mma. No transient BF16 buffers: they would cost memory and bandwidth at
   the lengths the compression exists for. FP8 is the storage format, not the compute format -- an FP8 mma for
   QK^T would round the queries too, the projection Gemma's long-context loss traced to.
3. **Every read goes through the quantized cache**, the current chunk's own keys included, so prefill and decode
   see the same K and V. Decode-against-prefill agreement and prefix-difference scoring (`ModelFamilyParity.md`
   G2, L3) stay valid.
4. **Unbounded layers only, and only where flash serves them:** every Llama layer, Qwen's full-attention layers and
   Gemma's global layers (K = V, one tensor, one scale set). Gemma's sliding ring is bounded and stays BF16. A
   layer on the cuBLASLt path -- an FP32 build, or head size 64 (Llama 3.2 1B) -- refuses FP8 KV at planning.
5. **The request chooses it; the planner prices it and never chooses it** in 0.21. `KvCacheCompression::FP8` on
   the request, priced exactly by `getRequiredMemory` so that `PlanEqualsBuild` holds. Automatic choice waits for
   all three families' measurements.
6. **Gate, per family, written before the run:** `ModelFamilyParity.md`'s long-context protocol (G2 / L3) on the
   same weights, FP8 KV against BF16 KV, every band within 0.01 nats per token and the difference not growing with
   context; then the same protocol past the BF16 plan's context, to what the FP8 plan reaches, where the whole
   book must still predict no worse than 1024 tokens of it. Decode rate at 32K recorded. Before any family: the op
   against BF16 on synthetic K and V. **Beside it, a behavioral arm that reports and does not gate** until its
   noise is known: an instruction at position 0 (a format or refusal constraint), a PG-19 book filling the
   context to the plan's length, then a question -- does the instruction still hold, BF16 KV against FP8 KV. A
   loss averaged over book tokens cannot see whether an early instruction still governs the output; "The Pitfalls
   of KV Cache Compression" (Chen, Geh, Grover, Van den Broeck and Israel, arXiv 2510.00231) shows compression
   degrading some instructions far faster than others while aggregate scores hold. Their failure is eviction's,
   which rounding every entry alike does not share; the arm checks that assumption rather than making it.
7. **Order:** the shared GQA op and both kernels; Llama (L3's BF16-KV curve is the baseline arm); Qwen (the
   `BACKLOG.md` entry that admits the work); Gemma's global layers. `Deployment.md` §2's "not a knob" row and
   `QuantizationDispatch.ixx`'s refusal change in the same work.

### Where compression buys speed and where it buys capacity (2026-09-29)

A decode step reads every weight once and every live cache row once, so the two byte counts decide whether a
smaller cache makes it faster. The KV traffic of one step overtakes the weight traffic at

```
s* = weight bytes per step / ( 2 * attention layers * KV heads * head dim * bytes per KV value * batch )
```

(the factor 2 is K and V; a layer whose K is its V counts once). Below `s*` a step is weight-bound and compressing
the cache buys little speed; above it, a cache `c` times smaller approaches `c` times faster. The model and the
crossover are from "The KV Cache Is the New Memory Wall" (Singh, arXiv 2609.30854), a survey whose own figures are
derived for batched serving; its quality numbers are other papers' and are not relied on here.

At batch 1, on the published Q4_0 and FP4 packages:

| Model | Weight bytes a step | KV bytes a position (BF16) | `s*` BF16 / FP8 KV | Rate at 32K over depth 0: predicted / Mila / llama.cpp |
|---|---|---|---|---|
| Llama 3.1 8B Q4_0 | 4.97 GB (body + BF16 head) | 128 KiB (32 layers x 8 heads x 128) | ~38K / ~76K | 0.54 / 0.50 / 0.50 |
| Gemma 4 12B Q4_0 | 7.1 GB (body + tied FP8 head) | 8 KiB global (8 layers x 1 head x 512, K = V); sliding layers a constant 335 MB past 1024 | ~870K / beyond the context | 0.92 / 0.85 / 0.89 |
| Qwen 3.8 27B FP4 | 13.7-16 GB (embedding storage not separated) | 64 KiB (16 attention layers x 4 heads x 256) | ~210-250K / beyond the context | not measured |

Measured on the RTX 5060 Ti (`ModelFamilyParity.md` 8.4 L5; Mila after the RMSNorm change and, for Llama, L5). The
model predicts Llama in both engines to within the few percent both engines' decode attention reads below bandwidth,
which is what licenses using it to size the levers. So:

- **For Llama, FP8 KV is a speed lever at depth.** Halving 4.3 GB of cache reads at 32K predicts ~1.3x faster decode
  there (about 38 -> 49 tokens a second) and ~1.1x at 8K -- the cells where Llama trails llama.cpp. Decision 7's
  order already puts Llama first.
- **For Gemma and Qwen, FP8 KV is a capacity lever.** Gemma's global cache at 32K is 0.27 GB against 7.1 GB of
  weights: FP8 saves about 0.6 ms a token there. What it buys is a longer planned context, which is the reason the
  Qwen entry gives.
- **Gemma's depth loss is kernel efficiency, not bytes.** The bytes predict 0.92 at 32K; Mila measures 0.85, so
  ~7% of Gemma's 32K decode is the head-size-512 global decode attention reading below bandwidth. Bounded, and
  owned by `DecodeGraph.md` section 8's "what remains", not by this Part.
- **The decode recording and cache compression cover opposite ends.** The recording saves a fixed ~1 ms a step at
  every depth, which matters most below `s*`; compression grows with depth above it.
- **A cache compression gated only on the book bands is gated on an aggregate.** The survey's warning that
  aggregate scores hide failures of position-sensitive recall (from the eviction literature) is decision 6's
  behavioral arm; a format change passes that arm too.

**How the kernels read it (the mechanics of decision 2).** Every E4M3 value is exact in BF16, so a tile's FP8 codes
widen into the BF16 stage the packed flash kernel already feeds to its BF16 MMA, unscaled and without rounding. The
per-token scales apply in FP32 where they factor out of the sums: the K scale of each key multiplies that key's score
column after QK, and the V scale of each key multiplies its probability before P is packed for PV, while the softmax
normalizer sums the unscaled probabilities. No scale is ever rounded into a stored value -- the ordering the FP8
head's staged path lacked (`Untriaged.md`, "The FP8 head's batched path rounds every weight before it scales"). Tiles
arrive by `cp.async` as FP8, half the bytes of BF16, into a staging area beside the existing stages, with each
stage's key scales beside them. The write (`kvcache_write_kv`) quantizes each row -- one head, one token -- with its
own absmax over 448, one warp per row. The fused decode kernel reads the same way. Both the flash prefill and the
fused decode read the chunk's own keys back from the cache after writing them (`CudaGqaOp.ixx`, `prefill_optimized`,
`decode_optimized`), so decision 3 holds by construction.

**The op, gated 2026-09-28** (`CudaGqaOp<BF16, false, true>` from `PerTokenKvFp8<>` through `OperationTraits`;
`Tests/Dnn/Components/Attention/GQA/CudaGqaOp.Fp8Cache.Cuda.cpp`; RTX 5060 Ti and RTX 4070 alike). On K and V the FP8
cache holds exactly -- a power-of-two scale times E4M3 codes with one at 448 -- the FP8 op and the BF16 op agree **bit for
bit** through chunked prefill and decode at Llama's (128, group 4), Gemma's global (512, group 16, batch 2) and Qwen's
(256, group 6) geometries: every scale factors out of its products without rounding. On normal random K and V, each
op's mean error against exact double attention over the values its cache holds is 3.0e-4 to 3.5e-4 for FP8 and
2.6e-4 to 3.1e-4 for BF16 (bound 1.25x, set after the first run: a 1e-2 absolute bound written before it assumed
outputs below 1, and they reach 3 to 4); what the quantization itself moves the output, 4.7e-3 to 5.9e-3 in the mean,
is 16 to 20 times the kernels' own error. The footprint equals the build (`getRequiredStateMemorySize`: codes plus one
FP32 scale per row); a head size the fused kernels do not serve is refused at build, and prefill with flash off is
refused. The BF16 kernels, now templated on the cache type, pass every existing flash and fused-decode parity test.

**Llama 3.1 8B, decision 6's first arm, 2026-09-28** (`LlamaQualityCudaTests.DISABLED_AcrossContextLengths_Q4_0_Fp8Cache`
against `..._Q4_0`, the L3 protocol on the same Q4_0 weights and five PG-19 books at 69632, RTX 5060 Ti). FP8 cache
less BF16 cache, whole-book nats per token, bands 0-8K, 8K-16K, 16K-32K, 32K-64K, 64K-69632:

| Book | Difference |
|---|---|
| 30312 | +0.0014, +0.0024, +0.0028, +0.0038, +0.0050 |
| 3608 | +0.0026, +0.0021, +0.0044, +0.0037, +0.0052 |
| 10321 | +0.0051, +0.0046, +0.0041, +0.0050, +0.0030 |
| 10356 | +0.0016, +0.0036, +0.0040, +0.0054, +0.0033 |
| 10762 | +0.0036, +0.0047, +0.0044, +0.0047, +0.0047 |
| Pooled | +0.0028, +0.0034, +0.0039, +0.0045, +0.0042 |

**Every band is within the 0.01 bound; the largest is 0.0054.** The pooled difference rises by about 0.0017 from the first
band to the fourth and then levels; two books rise to about 0.005, three do not -- growth well inside the bound,
recorded rather than read as a failure of "not growing". Test 1 reads as with the BF16 cache on every book (book
10762's model-side result included, `ModelFamilyParity.md` section 9, item 17). The FP8 run prefilled at chunk 1024
where the BF16 cache left room only for 128 -- the planner's rule on the freed memory -- a difference L4 measured at
about 0.0007 nats per token, and it ran 2.5 times faster (268 s a book against 661).

**At Llama 3.1's full 131072, 2026-09-28** (`..._Q4_0_Fp8Cache_131072`, RTX 5060 Ti). The FP8 cache fits the 16 GB card
at 131072 with a 128-token prefill chunk, where the BF16 cache stops at 69632. A book must be 131072 tokens long, so the
run takes the next long-enough books in order: 30312, 3608, 28988, 30754 (10321, 10356 and 10762 are too short). Four
of five finished, about 716 s each; the run was stopped for a reboot. Whole-book / short-context nats per token by band,
0-8K, 8K-16K, 16K-32K, 32K-64K, 64K-131072:

| Book | Whole book | Short context | Test 1 |
|---|---|---|---|
| 30312 | 2.6550, 2.5948, 2.6221, 2.6592, 2.5287 | 2.6728, 2.6398, 2.6492, 2.6865, 2.6014 | pass |
| 3608 | 2.2193, 2.1952, 2.0888, 2.1202, 2.1173 | 2.2408, 2.2345, 2.1028, 2.1457, 2.1736 | pass |
| 28988 | 2.5699, 2.4123, 2.3833, 2.3895, 2.4034 | 2.6106, 2.4401, 2.3790, 2.3895, 2.4156 | fails 16K-32K by 0.004 |
| 30754 | 2.4255, 2.6503, 2.5777, 2.4182, 2.6510 | 2.4155, 2.6333, 2.6030, 2.4559, 2.6880 | fails 0-8K by 0.010, 8K-16K by 0.017 |

**The top band never fails**: in all four books the stretch past the BF16 cache's reach is scored better with the whole
book than with 1024 tokens of it. The four failures sit in the bands the BF16 cache also reaches, on books it has not
been run on; whether they are the model's, as 10762's were (`ModelFamilyParity.md` section 9, item 17), is unmeasured --
the BF16 cache at 69632 on 28988 and 30754 answers it. On 30312, the one book scored both ways at the same 128-token
chunk, FP8 less BF16 is +0.0011, +0.0016, +0.0045, +0.0024 over the four bands both reach.

**Gemma 4, decision 6's second arm, 2026-09-30** (`GemmaLogLikelihoodCudaTests.DISABLED_KvCache_*`, the G2 protocol
whole-book only, five PG-19 books per model, RTX 5060 Ti; working tree on `0.21.0-dev+24`). Gemma's caches were BF16
everywhere until then -- the FP8 setting reached no Gemma layer. Two FP8 arms: `PerTokenKvFp8` on the global layers
(the sliding ring BF16), and that plus `SlidingWindowKvFp8`, a new policy putting the sliding ring's rows in the same
format (operation gate: bit-identical to the BF16 ring on values FP8 holds exactly, at head sizes 256 and 128). FP8 arm
less BF16, nats per token, largest of any book and band / pooled per band:

| Arm | Gemma 4 12B Q4_0, 65536, chunk 1024 (bands 0-8K, -16K, -32K, -64K) | Gemma 4 26B-A4B Q4_0, 24576, chunk 64 (bands 0-8K, -16K, -24K) |
|---|---|---|
| FP8 global | 0.0064 / -0.0016, +0.0002, +0.0011, +0.0004 | 0.0079 / +0.0034, +0.0037, +0.0031 |
| FP8 global and ring | 0.0098 / +0.0038, +0.0063, +0.0026, +0.0017 | **0.0230** / -0.0046, -0.0005, +0.0047 |

**The FP8 global cache passes on both; the FP8 ring fails on the 26B-A4B** (0.0230 and 0.0192, of both signs) and
clears the 12B by 0.0002. The 26B's arms were first run at the chunks the planner gave each (64, 512, 1024) and differed
by up to 0.0145; held to one chunk they read as above.

**Past the BF16 caches' reach, 2026-10-01** (`..._KvCache_26B_Q4_0_Fp8Global_PastBf16Reach`, RTX 5060 Ti, working tree
on `0.21.0-dev+27`). On the 16 GB card the 26B-A4B's BF16 caches stop at 32768 and its FP8 global cache reaches 65536,
so there the arm is gate 1 alone: whole book against the 1024 book tokens before each block, five PG-19 books.
**Every band of every book passes.** No prefill chunk fit: the 64-row scoring window adds 230 MiB of head scratch, and
even the 64-row chunk was predicted 148 MiB over the 15,172 MiB free (at the default one-row window the same build fits
a 256-row chunk with 58 MiB spare). The harness built at 64 rows anyway; the scores do not depend on the fit, but the
timing below may include a spill to host memory. Pooled, nats per token, whole book / window only:

| Band | 0-8K | 8K-16K | 16K-32K | 32K-64K |
|---|---|---|---|---|
| Pooled, 5 books | 3.6224 / 3.7331 | 3.4225 / 3.6107 | 3.4491 / 3.6569 | 3.4573 / 3.6025 |

The band past the BF16 reach gains 0.145 nats per token from the whole book, about as much as the bands below it; the
narrowest margin of the twenty is 0.0315 (book 10762, 32K-64K). About 330 s a book.

**Decode at 32K, 2026-10-01** (ProfileModel `--phase decode --seq-len 32512 --tokens 256 --ignore-eos --context-length
32768 --kv-cache bf16|fp8`, Q4_0, RTX 5060 Ti, one warm-up and three measured runs, which agree within 0.1%; the
planner's chunk). Tokens a second, BF16 cache / FP8 cache, and the 32,512-token prefill each run starts with:

| Model | Decode | Prefill (chunk) |
|---|---|---|
| Llama 3.1 8B | 44.31 / 56.28, **1.27x** | 13.53 s / 14.92 s (the largest rung both) |
| Gemma 4 12B | 48.14 / 49.52, 1.03x | 14.05 s / 15.05 s (1024 both) |
| Gemma 4 26B-A4B | 101.07 / 109.26, 1.08x | 11.84 s (256) / 8.37 s (1024) |

Both predictions above hold: on Llama the FP8 cache is a speed lever at depth (1.3x predicted), on Gemma 12B a saving of
0.58 ms a token (about 0.6 predicted). On the 26B-A4B it is both at once on the 16 GB card: 0.74 ms a token in decode,
and its freed memory lets the planner take a 1024-row chunk where the BF16 caches leave room for 256, which makes the
prefill 1.41x faster. At equal chunk the FP8 cache's prefill was **7 to 10% slower** (Llama, Gemma 12B) -- unpredicted.

**The FP8 prefill penalty, 2026-10-01.** Prefill attention is compute-bound, so halving the bytes it reads saves
little, while every block that shares a KV head widens the same codes again; the penalty is all in the attention
kernel (Nsight Systems, RTX 5060 Ti, Q4_0, the 32,512-token prefill at the planner's chunk; every other kernel within
noise). Both prefill kernels issued each load only after
widening the tile before it, and widened two codes a thread at a time. With each load issued first and sixteen codes a
thread a step (`Gqa.Fp8Widen.cuh`, decode's own widen, now shared), measured old against new binaries in one sitting:

| FP8 attention over BF16, same chunk | Before | After | Whole prefill, after |
|---|---|---|---|
| Gemma 4 12B, head size 512 (global layers) | +1,141 ms, +28.6% | +238 ms, +6.0% | +0.7% |
| Llama 3.1 8B, packed kernel, head size 128 | +1,390 ms, +17.7% | +1,004 ms, +12.8% | +7.0% |

Two other suspects were measured and are not the cause: shared-memory bank conflicts (an eight-code widen cut them
from 36.7M to 11.8M a launch and moved nothing, and cost decode 0.3-0.5%, so it was not kept) and register spills
(the head-size-128 FP8 instantiation spilled 16 bytes, stored before the key loop and reloaded after it). What remained
on Llama was the packed kernel's single BF16 stage: each tile waited, barriered, widened with all eight warps, and
barriered again before its MMAs, where the BF16 cache has one barrier, and at one block a multiprocessor nothing hid
the widening (tensor pipe 70% busy against BF16's 80%, issue slots 23% busy in both).

**Widening ahead, `0.21.0-dev+29`.** Where two BF16 stages and one code stage fit in a block's 99 KB -- head size 128
at 87 KB, and head size 512 through a window -- the packed kernel widens the next tile into the other BF16 stage at
the end of this one, after its MMAs, so a tile costs one barrier as over the BF16 cache. No barrier guards the codes:
a thread widens only the chunks its own `cp.async` wrote, so its own pipeline wait suffices. Head size 256 (101 KB)
keeps the single stage. Llama 3.1 8B, same 32K prefill, `+28` and `+29` binaries in one sitting: FP8 attention over
BF16 **+1,000 ms (+12.7%) -> +549 ms (+7.0%)**, the whole prefill +6.6% -> **+3.5%**; at the last chunk the kernel
runs 13.12 ms against 13.84 and BF16's 12.22, in 202 registers without the spill, tensor pipe 74% busy. What remains
is the widening itself, repeated by every block that shares a KV head: the price of computing in BF16 (`Untriaged.md`,
"The FP8 cache's prefill widens every key to BF16").

**Wired, 2026-10-01.** A request's `KvCacheCompression::FP8` reaches `PerTokenKvFp8<>` on every Llama layer and
on Gemma's global layers through one dispatcher, `dispatchKvCacheCompression` (`QuantizationDispatch.ixx`), which the
plan, the load and the footprint all use. Gemma's ring stays BF16 whatever the request says. A build without the FP8
kernels (CPU, FP32) refuses it at planning, as does a head size they do not serve: `CudaGqaOp` refuses that when the
cache is priced as well as at build. The weight presets stopped setting FP8 KV in the same change, so the cache is
FP8 only when a caller asks.

**Decision 6's behavioral arm, 2026-10-01** (`GemmaInstructionRetentionCudaTests.DISABLED_*`,
`LlamaInstructionRetentionCudaTests.DISABLED_Bf16AgainstFp8Cache_Q4_0`, Q4_0 weights, RTX 5060 Ti, working tree on
`0.21.0-dev+27`). A system instruction at position 0, a PG-19 book (30312, 3608) in the user turn to a total length, a
question about the book, and a greedy reply of up to 256 tokens. Three instructions: reply in exactly three capital
words, begin with BANANA, end with OVER. A reply to the last that reaches 256 tokens is cut, not judged. Instructions
holding, BF16 cache / FP8 cache, over the lengths both reach (six prompts per instruction):

| Model, lengths | Three capital words | Begins BANANA | Ends OVER (holds, cut) | All |
|---|---|---|---|---|
| Gemma 4 26B-A4B, 2048, 16384, 31744 | 4 / 4 | 6 / 6 | 3, 3 / 3, 3 | 13 / 13 |
| Gemma 4 12B, 2048, 16384, 64512 | 4 / 5 | 6 / 6 | 2, 4 / 2, 4 | 12 / 13 |
| Llama 3.1 8B, 2048, 16384, 65536 | 1 / 2 | 4 / 4 | 2, 3 / 3, 1 | 7 / 9 |

**No instruction is lost with the FP8 cache that the BF16 cache keeps, beyond single cells in both directions.** On
the 26B-A4B every one of the 18 verdicts matches; on the 12B three differ, two of them FP8's way; on Llama the cells
differ both ways and FP8 holds two more. A first run with a 32-token budget cut every "ends OVER" reply mid-sentence;
the counts above are from the 256-token harness, where only an end-judged reply can be cut.

- **Past the BF16 cache's reach.** The 26B-A4B with the FP8 global cache at 64512 holds BANANA on both books and
  three words on 30312; on 3608 it answers in four words, as both caches do at 16384 and 31744. Llama with the FP8
  cache at 130048 holds nothing, and neither did the BF16 cache at 65536.
- **Llama loses the instruction by 64K with either cache.** At 65536 no reply in either arm begins with BANANA or is
  three words long; all twelve open "The main character of this book". The loss is not the cache's. Whether it is the
  model's or Mila's at length is unmeasured (`Untriaged.md`). Both Gemma models keep BANANA at every length measured.
- **What it cannot see.** Two books and one greedy reply per cell; both families answer at length, so "ends OVER" is
  cut in 1 to 4 of its 6 cells per arm. It reports and does not gate, as decision 6 says, until its noise is known.
  The arms' replies to the same prompt usually diverge in wording within a sentence, and a verdict can flip with
  them, which is the size of the differences above.

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
and add a corresponding policy struct and `load` branch — no other
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
internal enum intermediary layer was unnecessary — `load` maps `ModelConfig`
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
- **Fitting sub-4-bit formats at load time.** Codebook tables are fitted offline against
  calibration data. Only the absmax formats, FP8 and FP4, are fitted during a load.
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
