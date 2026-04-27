# Mila Operation Dispatch Architecture Specification

## Overview

This document specifies the compile-time operation dispatch system that replaces
the `OperationRegistry` singleton. All components that call into a device-specific
compute operation follow the pattern described here. `Linear` is the canonical
reference implementation.

---

## 1. Motivation

The previous `OperationRegistry` approach used a runtime hash map keyed on
`TypeID{ DeviceType, TensorDataType, TensorDataType, ComputePrecision }`.
This design became structurally wrong as Mila evolved:

- `DeviceType` and `TPrecision` are **fully known at compile time** on every
  component — the runtime lookup emulated what the compiler already resolved.
- Adding quantization required a fourth key axis (`TWeight`), forcing a registry
  refactor each time a new weight format was introduced.
- Each distinct `forward`/`backward` arity (unary, binary, paired) required a new
  intermediate base class (`UnaryOperation`, `BinaryOperation`, `PairedOperation`),
  scaling the class hierarchy with arity rather than with semantic differences.
- String-keyed lookup (`"LinearOp"`) provided no compiler enforcement.
- The `*Registrar` classes introduced global init-order-dependent side effects via
  the singleton.

The traits-based dispatch system eliminates all of these issues without changing the
`Operation<TDeviceType, TPrecision>` lifecycle contract.

---

## 2. Type Axes

Every operation is parameterized by two independent type axes:

| Axis | Template parameter | Meaning |
|---|---|---|
| `TPrecision` | existing on `Component` and `Operation` | Activation input type and compute/accumulation precision. The op owns all internal precision decisions (dequantization, promotion, accumulation). |
| `TWeight` | new, defaulted to `TPrecision` | Weight storage type. The only axis that quantization varies. |

`TPrecision` is not split into separate activation and compute parameters.
Operations handle internal accumulation precision as an implementation detail —
this is unchanged from the previous design.

**Consequence:** every existing instantiation `Linear<Cpu, FP32>` and
`Linear<Cuda, FP32>` is unaffected. `TWeight` defaults to `TPrecision` and
is invisible to existing code.

---

## 3. Operation Base Class

`Operation<TDeviceType, TPrecision>` is the **only** base class for all compute
operations. It owns the lifecycle contract exclusively:

```cpp
template <DeviceType TDeviceType, TensorDataType TPrecision>
class Operation
{
public:
    static constexpr DeviceType device_type = TDeviceType;
    static constexpr TensorDataType data_type = TPrecision;

    virtual ~Operation() = default;

    virtual void build( const BuildContext& );
    virtual void setParameters( ITensor* weight, ITensor* bias );
    virtual void setGradients( ITensor* weight_grad, ITensor* bias_grad );
    virtual void clearGradients() noexcept;
    virtual void setTrainingMode( TrainingMode );
    virtual bool isBuilt() const;
    virtual bool isEvalMode() const;
    virtual OperationType getOperationType() const = 0;
    virtual DeviceType getDeviceType() const;
    virtual TensorDataType getDataType() const;
    virtual std::string getName() const = 0;
};
```

`UnaryOperation`, `BinaryOperation`, and `PairedOperation` are **removed**.
The `forward`/`backward` signature is a property of each concrete op, not of
a base class. The arity distinction that required three separate intermediate
classes disappears — the component holds the concrete type and calls it directly.

---

## 4. Concrete Operation Contract

Each concrete op:

- Derives from `Operation<TDeviceType, TPrecision>` only.
- Declares `forward` and `backward` as plain (non-virtual) methods with typed
  `Tensor<T, MR>&` parameters matching its own arity.
- Owns its `MR`, `TensorType`, and cast helper aliases locally — they are not
  inherited from a base.
- Accepts `IExecutionContext*` and the component config at construction.

```cpp
class CpuLinearOp : public Operation<DeviceType::Cpu, TensorDataType::FP32>
{
public:
    using MR = CpuMemoryResource;
    using TensorType = Tensor<TensorDataType::FP32, MR>;

    CpuLinearOp( IExecutionContext* context, const LinearConfig& config );

    void forward( const TensorType& input, TensorType& output ) const;
    void backward( const TensorType& input, const TensorType& output_grad,
                   TensorType& input_grad ) const;

    // lifecycle overrides
    OperationType getOperationType() const override;
    std::string getName() const override;
};
```

`CpuEncoderOp` previously used `TInput=INT32` on the intermediate base to vary
the input tensor element type. With concrete ownership this distinction moves
directly into the method signature — the type mismatch between `INT32` tokens
and `FP32` output is expressed in the method parameter types, not via a base
class template parameter.

---

## 5. Per-Component Traits

Each component defines a primary traits template and one specialization per
backend/precision combination. Traits files are co-located with the op they name.

### Primary template

Lives with the component, not with any backend:

```cpp
// Components/Linear/LinearOpTraits.ixx
export module Compute.LinearOpTraits;

namespace Mila::Dnn::Compute
{
    export template<DeviceType TDeviceType, TensorDataType TPrecision,
                    TensorDataType TWeight = TPrecision>
    struct LinearOpTraits
    {
        static_assert(
            sizeof(TDeviceType) == 0,
            "No LinearOp for this (DeviceType, TPrecision, TWeight) combination."
        );
    };
}
```

### Backend specialization

Lives with the concrete op:

```cpp
// Devices/Cpu/Operations/CpuLinearOpTraits.ixx
export module Compute.CpuLinearOpTraits;

import Compute.LinearOpTraits;
import Compute.CpuLinearOp;

namespace Mila::Dnn::Compute
{
    export template<>
    struct LinearOpTraits<DeviceType::Cpu, TensorDataType::FP32>
    {
        using type = CpuLinearOp;
    };
}
```

### Quantized weight specialization

Adding INT8 weight support is purely additive — no existing file is modified:

```cpp
// Devices/Cuda/Operations/CudaLinearOpInt8Traits.ixx
export template<>
struct LinearOpTraits<DeviceType::Cuda, TensorDataType::FP32, TensorDataType::INT8>
{
    using type = CudaLinearOpInt8;
};
```

---

## 6. Component Integration

The component adds a defaulted `TWeight` parameter and derives the op type from
traits. The `operation_` field and `createOperation()` change; everything else
is untouched.

```cpp
export template<DeviceType TDeviceType, TensorDataType TPrecision,
                TensorDataType TWeight = TPrecision>
    requires PrecisionSupportedOnDevice<TPrecision, TDeviceType>
class Linear : public Component<TDeviceType, TPrecision>
{
private:
    using OpType = typename LinearOpTraits<TDeviceType, TPrecision, TWeight>::type;
    std::shared_ptr<OpType> operation_{ nullptr };

    void createOperation()
    {
        operation_ = std::make_shared<OpType>(
            this->getExecutionContext(), config_ );
    }
};
```

`import Compute.OperationRegistry` is removed.
`import Compute.CpuLinearOpTraits` (or the CUDA variant) replaces it, selected
by the build configuration.

---

## 7. Signature Contract Enforcement

Without a pure virtual base, the `forward`/`backward` contract is enforced by a
C++20 concept at the traits specialization site:

```cpp
template<typename TOp, typename TTensor>
concept LinearOpConcept = requires( const TOp& op, const TTensor& in, TTensor& out )
{
    op.forward( in, out );
    op.backward( in, out, out );
};

// Validated in the component:
static_assert( LinearOpConcept<OpType, TensorType> );
```

An unsupported `(DeviceType, TPrecision, TWeight)` combination is a hard compile
error with a `static_assert` message, not a runtime exception thrown from a hash
map lookup.

---

## 8. What Is Eliminated

| Removed | Replaced by |
|---|---|
| `OperationRegistry` singleton and three hash maps | Per-component traits + `std::make_shared<ConcreteOp>` |
| `TypeID` hashing and string key lookups | Compile-time traits specialization lookup |
| `shared_ptr<void>` type erasure + `static_pointer_cast` | `shared_ptr<ConcreteOp>` — direct type |
| `*Registrar` classes and global init side-effects | Removed entirely |
| `UnaryOperation`, `BinaryOperation`, `PairedOperation` base classes | Concrete op owns its signature |
| Stringly-typed `"LinearOp"` key | No equivalent — compiler resolves the type |

`OperationRegistry`, `UnaryOperation`, `BinaryOperation`, and `PairedOperation`
are deleted once all components are migrated.
`Operation<TDeviceType, TPrecision>` is retained unchanged.

---

## 9. File Layout

```
Components/Linear/
  Linear.ixx                       component — imports LinearOpTraits
  LinearConfig.ixx
  LinearOpTraits.ixx               primary template + static_assert

Devices/Cpu/Operations/
  CpuLinearOp.ixx                  concrete op
  CpuLinearOpTraits.ixx            specialization for <Cpu, FP32>

Devices/Cuda/Operations/
  CudaLinearOp.ixx
  CudaLinearOpTraits.ixx           specialization for <Cuda, FP32>
  CudaLinearOpInt8.ixx             quantized weight variant
  CudaLinearOpInt8Traits.ixx       specialization for <Cuda, FP32, INT8>
```

Each file has a single clear responsibility. Adding a new backend or weight format
is self-contained — no shared file is modified.

---

## 10. Adding a New Op — Checklist

When adding a new component with its own compute operation:

- [ ] `ComponentOpTraits.ixx` — primary template with `static_assert` diagnostic
- [ ] `BackendComponentOp.ixx` — concrete op deriving from `Operation<D, P>` only,
      non-virtual `forward`/`backward` with typed tensor parameters
- [ ] `BackendComponentOpTraits.ixx` — specialization naming the concrete type
- [ ] `Component.ixx` — `using OpType = typename ComponentOpTraits<...>::type;`
      field and `createOperation()` using `std::make_shared<OpType>`
- [ ] `static_assert( ComponentOpConcept<OpType, TensorType> )` in component

---

## 11. Adding a Quantized Weight Variant — Checklist

When adding quantized weight support to an existing op (e.g. INT8 weights for Linear):

- [ ] `BackendOpQuantized.ixx` — concrete op for the quantized path,
      derives from `Operation<D, P>`, owns dequantization and scale application
- [ ] `BackendOpQuantizedTraits.ixx` — specialization for `<Device, TPrecision, TWeight>`
- [ ] No existing file is modified — the default `TWeight = TPrecision` path is untouched
- [ ] `QuantizationConfig` nested in the component config carries scale tensors,
      zero-points, and granularity (per-tensor vs per-channel) for dynamic quantization;
      the op reads these at construction

---

*This document reflects design decisions made through April 2026.*
*Update when new operations, weight formats, or backend devices are added.*
