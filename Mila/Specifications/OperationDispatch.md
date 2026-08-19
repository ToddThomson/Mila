# Mila Operation Dispatch Architecture Specification

## Overview

This document specifies the compile-time operation dispatch system that replaces
the `OperationRegistry` singleton. All components that call into a device-specific
compute operation follow the pattern described here. `Linear` is the canonical
reference implementation — fully migrated and building as of May 2026.

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

## 5. Unified Operation Traits

Rather than one traits primary template per component, Mila uses a **single**
`OperationTraits` template keyed on the `OperationType` enum. All backend
specializations live in two partition files — one per device family.

### Primary template

```cpp
// Compute/Operations/OperationTraits.Template.ixx
export module Compute.OperationTraits.Template;

export import Compute.DeviceType;
export import Compute.OperationType;
export import Dnn.TensorDataType;

namespace Mila::Dnn::Compute
{
    /// Primary template — no default definition.
    /// A missing specialization is a hard compile error on ::type.
    export template<OperationType TOp,
                    DeviceType   TDeviceType,
                    TensorDataType TPrecision,
                    typename     TPolicy = void>
    struct OperationTraits;
}
```

`TPolicy` carries the quantization or cache policy for ops that need it
(`WeightQuantPolicy` for Linear, `KvCachePolicy` for GQA, `void` for all others).
No op needs both simultaneously — the scope table in `Quantization.md` bounds this.

### CUDA specializations — OperationTraits.Cuda.ixx

All CUDA mappings live in the `:Cuda` partition of `Compute.OperationTraits`.
Adding a new backend or weight format is purely additive — one new specialization
block, no existing file is modified:

```cpp
export module Compute.OperationTraits:Cuda;

import Compute.OperationTraits.Template;
import Compute.CudaLinearOp;
import Dnn.Quantization.Weight.Policies;

namespace Mila::Dnn::Compute
{
    using namespace Mila::Dnn::Quant::Weight;
    using namespace Mila::Dnn::Compute::Cuda::Linear;

    // LinearOp
    export template<>
    struct OperationTraits<OperationType::LinearOp, DeviceType::Cuda,
                           TensorDataType::BF16, NoWeightQuant>
    { using type = CudaLinearOp<TensorDataType::BF16, NoWeightQuant>; };

    export template<>
    struct OperationTraits<OperationType::LinearOp, DeviceType::Cuda,
                           TensorDataType::BF16, PerChannelFp8<>>
    { using type = CudaLinearOp<TensorDataType::BF16, PerChannelFp8<>>; };

    // GQA, SamplingOp, and policy-free ops added here as components migrate.
}
```

### Component integration

```cpp
// In Linear<TDeviceType, TComputePrecision, TWeightQuant>:
using OpType = typename Compute::OperationTraits<
    Compute::OperationType::LinearOp,
    TDeviceType, TComputePrecision, TWeightQuant>::type;

std::shared_ptr<OpType> operation_{ nullptr };

void createOperation() {
    operation_ = std::make_shared<OpType>(
        this->getExecutionContext(), config_ );
}
```

### Why not static_assert(Concept) in the class body?

`LinearOpConcept<OpType, TensorType>` is defined for documentation and test use,
but is **not** placed as a `static_assert` inside the `Linear` class template.
MSVC eagerly instantiates all member function bodies when a concept is evaluated
inside a class template body, forcing `CudaLinearOp::build()` to compile before the
full `ExecutionContext<Cuda>` definition is in scope. A missing `OperationTraits`
specialization already produces a hard error on `::type` — that is the practical
compile-time guard.

---

## 6. Component Integration

The component imports `Compute.OperationTraits` (which re-exports the template and
all active backend partitions) and derives the op type directly. The `operation_`
field and `createOperation()` change; everything else is untouched.

```cpp
export template<DeviceType TDeviceType, TensorDataType TComputePrecision,
                WeightQuantPolicy TWeightQuant = NoWeightQuant>
    requires PrecisionSupportedOnDevice<TComputePrecision, TDeviceType>
class Linear : public Component<TDeviceType, TComputePrecision>
{
public:
    using OpType = typename Compute::OperationTraits<
        Compute::OperationType::LinearOp,
        TDeviceType, TComputePrecision, TWeightQuant>::type;

private:
    std::shared_ptr<OpType> operation_{ nullptr };

    void createOperation()
    {
        operation_ = std::make_shared<OpType>(
            this->getExecutionContext(), config_ );
    }
};
```

`import Compute.OperationRegistry` is removed.
`import Compute.OperationTraits` replaces it — no per-backend import selection,
no build-system guards.

---

## 7. Signature Contract Enforcement

Without a pure virtual base, the `forward`/`backward` contract is documented by a
C++20 concept defined in `OperationTraits.Template.ixx`:

```cpp
template<typename TOp, typename TTensor>
concept LinearOpConcept = requires( TOp& op, const TTensor& in, TTensor& out )
{
    op.forward( in, out );
    op.backward( in, out, out );
};
```

This concept is available for use in tests and for documentation purposes.
It is **not** placed as a `static_assert` inside the component class template —
see §5 for the MSVC eager-instantiation constraint that prevents this.

An unsupported `(OperationType, DeviceType, TPrecision, TPolicy)` combination
produces a hard compile error at the `::type` access site — no runtime exception,
no hash map lookup. The *legibility* of that error is not automatic, however: two
distinct failure modes produce very different diagnostics, and one is actively
misleading. See §12.

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
Compute/Operations/
  OperationTraits.ixx              aggregator — export import Template + :Cuda + :Cpu
  OperationTraits.Template.ixx     unified primary template; concepts
  OperationTraits.Cuda.ixx         :Cuda partition — all CUDA specializations
  OperationTraits.Cpu.ixx          :Cpu partition — all CPU specializations (pending)

Components/Linear/
  Linear.ixx                       component — import Compute.OperationTraits
  LinearConfig.ixx

Devices/Cuda/Operations/Linear/
  CudaLinearOp.ixx                 concrete op — derives from Operation<Cuda, P> only
```

Adding a new weight format or backend: one new specialization block in
`OperationTraits.Cuda.ixx` or `OperationTraits.Cpu.ixx`. No shared file is
modified. No registrar file is created.

---

## 10. Adding a New Op — Checklist

When adding a new component with its own compute operation:

- [ ] `BackendComponentOp.ixx` — concrete op deriving from `Operation<D, P>` only;
      non-virtual `forward`/`backward` with typed tensor parameters
- [ ] `OperationTraits.Cuda.ixx` — add specialization block for the new `OperationType`
      enum value; one specialization per (precision, policy) combination
- [ ] `OperationType.ixx` — add the new enum entry and `OperationNames::` string constant
- [ ] `Component.ixx` — `using OpType = typename Compute::OperationTraits<OperationType::NewOp, ...>::type;`
      and `createOperation()` using `std::make_shared<OpType>`
- [ ] `ComponentOpConcept` defined in `OperationTraits.Template.ixx` for documentation

---

## 11. Adding a Quantized Variant — Checklist

When adding a new policy dimension to an existing op (e.g. FP4 weights for Linear):

- [ ] New policy struct satisfying the relevant concept (`WeightQuantPolicy` or `KvCachePolicy`)
- [ ] `OperationTraits.Cuda.ixx` — one new specialization: `<OperationType, Cuda, Precision, NewPolicy>`
- [ ] Concrete op handles the new policy path via `if constexpr (TPolicy::kIsActive)` or
      a separate op class if the implementation diverges significantly
- [ ] No existing specialization is modified — all default paths are untouched

---

## 12. Diagnostics — making unsupported combinations fail legibly

Compile-time dispatch means the compiler error *is* the user interface for an
unsupported combination. Today that interface is poor, and §7's hard error is
legible in only one of two failure modes:

1. **Missing specialization** — `OperationTraits<...>::type` on an unspecialized
   primary yields an "incomplete type" / use-of-undefined-template error. It does
   not name *which* axis is unsupported, only that a type is incomplete.

2. **Present-but-broken specialization** — a specialization that exists but maps
   to a concrete op whose own constraints fail. The diagnostic is a multi-level
   constraint cascade deep inside the kernel and never names the real cause. This
   is *worse* than a missing specialization: the dispatch table advertises a
   capability the backend does not have.

   **Motivating example:** `OperationTraits<GeluOp, Cuda, BF16>` maps to
   `CudaGeluOp<BF16>`, but `cuda_gelu_impl` is constrained to `float || half` (an
   FP16-era kernel never migrated to BF16). Instantiating `Gelu<Cuda, BF16>`
   produces an opaque `C7602` constraint failure on MSVC. The row lied, and
   nothing caught it until a test instantiated it.

### Principle: fail high, fail in words, from one source of truth

**A. Friendly primary-template assert.** The unspecialized `OperationTraits`
primary should carry `static_assert( always_false<...>, "No operation registered
for this OperationType / DeviceType / TensorDataType / Policy. See
OperationDispatch.md." )` via the dependent-false idiom, converting failure mode 1
from an incomplete-type puzzle into a sentence. A `static_assert` message prints
verbatim, outside the template backtrace, so it is compiler-agnostic and
side-steps MSVC's `C7602` opacity. Removing a bogus specialization (failure mode
2) routes it back into this friendly path.

**B. A single authoritative capability predicate.** The root cause of the BF16 lie
is that "is `<Op, Device, Precision, Policy>` real?" is asserted in two places that
can drift — the traits table and the kernel's `requires`-clause. Make it one: a
pure boolean trait `OperationSupported<TOp, TDeviceType, TPrecision, TPolicy>`,
specialized `true` only for combinations the backend actually implements. The
kernel constraint, the traits specialization, and a component-level assert all
reference it, so the table cannot advertise what the kernel rejects.

This predicate is **safe to `static_assert` inside the component class body**,
unlike the member-probing `LinearOpConcept` of §5. The §5 hazard is specific to
concepts that name op *members* (`op.forward(...)`), which forces MSVC to eagerly
instantiate those member bodies. A pure capability predicate over the
`(Op, Device, Precision, Policy)` tuple touches no op member, so it does not
trigger eager instantiation — it surfaces the error at `Gelu<Cuda, BF16>`, the
line the user wrote, naming the component.

**C. Name the kernel concepts.** Prefer a named concept
(`concept CudaGeluNative = ...;`) over a raw `requires std::is_same_v<...> || ...`
so the diagnostic at least names the failed concept. Marginal, but free as each op
is touched.

### Workflow

When a specialization error is opaque, reproduce the offending translation unit
under **Clang** (the WSL build). Clang's template diagnostics are markedly more
legible than MSVC's `C7602` form.

### Adoption

This is `Src` work spanning the dispatch core and every op's constraint, so it is
adopted incrementally — pair it with the FP16 removal, when each op's
supported-precision set is made explicit anyway, rather than a big-bang refactor.
The capability predicate (B) is the design target; the friendly primary assert (A)
is the high-ROI first step. Tracked in BACKLOG.

---

*This document reflects design decisions made through June 2026.*
*Update when new operations, weight formats, or backend devices are added.*
