# Mila FP8 Weight Quantization Design

> **Status:** Alpha.5 — In Progress  
> **Scope:** `Linear` component and `CudaLinearOp` only  
> **Source of truth:** This document supersedes the `QuantizationConfig`-in-`BuildContext` design.

---

## Design Philosophy

Mila's core principle is **type safety at compile time**. Device type and precision are
template parameters — mixing them is a compile error, not a runtime surprise. This principle
applies fully to weight quantization.

Weight precision is **not** a runtime configuration concern. It is a compile-time deployment
decision, expressed as a template parameter. The framework enforces this statically. There is
no runtime dispatch on quantization state.

---

## Quantization Strategy

FP8 quantization in Mila is a **load-time, one-way weight compression** strategy applied
exclusively to the `Linear` component. Weights are loaded from a BF16 pretrained checkpoint
and quantized to FP8_E4M3 during `loadParameter()`. After quantization, only the FP8
representation lives on device. The BF16 source is never retained.

**FP8 format:** `E4M3` (`__nv_fp8_e4m3`). Higher precision (more mantissa bits) is correct
for stored weights. `E5M2` (wider dynamic range) is reserved for gradients and is not a
current Mila target.

**Scale granularity:** Per output channel. One `float32` scale per output feature, derived
from the maximum absolute value in that channel. This is the standard for LLM weight
quantization and is natively supported by cuBLASLt.

**Dequantization strategy:** None at runtime. cuBLASLt native FP8 matmul is used —
weights remain FP8 through the entire operation. No transient BF16 weight copy per forward
pass.

---

## Type System

Weight precision is encoded as a template parameter at every level of the stack.

### Component

```cpp
template<DeviceType TDeviceType, TensorDataType TComputePrecision, TensorDataType TWeight = TComputePrecision>
    requires PrecisionSupportedOnDevice<TComputePrecision, TDeviceType>
class Linear : public Component<TDeviceType, TComputePrecision>
{
    static constexpr bool kIsQuantized = (TWeight != TComputePrecision);
};
```

### Operation

```cpp
template<TensorDataType TComputePrecision, TensorDataType TWeightPrecision = TComputePrecision>
    requires PrecisionSupportedOnDevice<TComputePrecision, DeviceType::Cuda>
class CudaLinearOp : public Operation<DeviceType::Cuda, TComputePrecision>
{
    static constexpr bool kIsQuantized = (TWeightPrecision != TComputePrecision);
};
```

### Type Map

The `LinearOpTypeMap` resolves the correct concrete operation type from the component's
template parameters. The quantized specialization is:

```cpp
template<>
struct LinearOpTypeMap<DeviceType::Cuda, TensorDataType::BF16, TensorDataType::FP8_E4M3>
{
    using op_type = Cuda::Linear::CudaLinearOp<TensorDataType::BF16, TensorDataType::FP8_E4M3>;
};
```

The three meaningful deployment configurations are:

| Configuration | `TComputePrecision` | `TWeight` | `kIsQuantized` |
|---|---|---|---|
| Reference / validation | `FP32` | `FP32` | `false` |
| Standard inference | `BF16` | `BF16` | `false` |
| Quantized inference | `BF16` | `FP8_E4M3` | `true` |

---

## Ownership Model

`Linear` (component) owns all parameters. The operation receives pointers to
component-owned tensors and must not free them.

### Parameter Members on `Linear`

```cpp
std::unique_ptr<WeightTensorType> weight_{ nullptr };   // Tensor<TWeight, MR>
std::unique_ptr<TensorType>       bias_{ nullptr };     // Tensor<TComputePrecision, MR> — optional
std::unique_ptr<TensorType>       weight_scales_{ nullptr }; // Tensor<FP32, MR> — quantized path only
```

`weight_scales_` is allocated and populated only when `kIsQuantized` is true. It is
`nullptr` for all non-quantized specializations.

---

## Quantization Pipeline

### Sequence

```
fromPretrained()
    └── build( build_context )
            └── operation_ constructed (CudaLinearOp<BF16, FP8_E4M3>)
                    └── cuBLASLt FP8 plan built (types known statically)
    └── loadParameters( reader )
            └── loadParameter( "weight", blob )
                    └── if constexpr ( kIsQuantized )
                            └── operation_->quantize( blob, *weight_, *weight_scales_, expected_shape )
                        else
                            └── loadParameterFromBlob( "weight", blob, *weight_, expected_shape )
    └── (implicit) setParameters called after load
            └── operation_->setParameters( weight_.get(), bias_.get() )
            └── if constexpr ( kIsQuantized )
                    └── operation_->setWeightScales( weight_scales_.get() )
```

### `loadParameter()` on `Linear`

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

### `quantize()` on `CudaLinearOp`

`quantize()` is a concrete method on `CudaLinearOp`, not a virtual method on the
operation base class. It is only callable through the resolved `OperationType`, which
`Linear` holds directly. No casting is required.

Responsibilities:

1. Walk the BF16 blob on the host, compute per-channel `float32` scales (max abs per output channel)
2. Upload scales to `weight_scales_` on device
3. Quantize BF16 values to FP8_E4M3 using the computed scales
4. Upload quantized weights to `weight_` on device

Scale computation is host-side. This is a one-time load-time cost. No reduction kernel
is required.

---

## Operation Base Class Contract

The base `Operation` class defines the universal parameter binding contract:

```cpp
// Bind weight and bias — universal contract for all operations
virtual void setParameters( ITensor* weight, ITensor* bias ) {}

// Bind gradient tensors — mirrors setParameters semantics
virtual void setGradients( ITensor* weight_grad, ITensor* bias_grad ) {}
```

`setWeightScales()` is **not** on the base class. It is a concrete method on
`CudaLinearOp` only, called directly through `OperationType`:

```cpp
// On CudaLinearOp only — not part of the base contract
void setWeightScales( ITensor* scales );
```

This keeps the base class free of quantization-specific concerns. Non-quantized
operations are entirely unaware that `setWeightScales()` exists.

---

## What Was Removed

### `QuantizationConfig` from `BuildContext`

The earlier design carried a `QuantizationConfig` value in `BuildContext` to signal
quantization intent to `Linear::build()`. This was incorrect for two reasons:

1. `Linear` already knows at compile time whether it is quantized via `kIsQuantized`.
   `BuildContext` was redundantly carrying a runtime value for a statically-known fact.

2. It contradicted Mila's core design principle: type safety at compile time, not
   runtime dispatch magic.

`QuantizationConfig` is removed from `BuildContext`. The template parameter `TWeight`
is the sole source of truth for quantization state.

---

## cuBLASLt FP8 Plan

`CudaLinearOp<BF16, FP8_E4M3>::build()` constructs the cuBLASLt plan for FP8 inputs
at build time. All type information is statically known from the template parameters.
The plan references scale descriptor pointers, which are populated by `setWeightScales()`
after `loadParameters()` completes. No plan rebuild is required after weight loading.

The Ada Lovelace architecture (RTX 4070) has native FP8 tensor core support.
cuBLASLt operations with `CUDA_R_8F_E4M3` weight type and per-column float32 scales
run without dequantization overhead.

---

## Non-Goals

- **Runtime quantization toggling.** A `Linear` instance is either quantized or it is
  not. This is fixed at compile time.
- **Activation quantization.** Only weights are quantized. Activations remain at
  compute precision (BF16).
- **Per-group scales.** Per-channel scales are the current target. Per-group (finer
  granularity) is not planned for Alpha.5.
- **FP16 support.** BF16 supersedes FP16 for all Mila compute targets.
- **Training with quantized weights.** Load-time quantization is an inference
  optimization. The training path is unaffected.