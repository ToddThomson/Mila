# Mila Component Test Methodology

A repeatable structure for unit-testing Mila components. Mila is a component
library built from a few repeatable patterns; its test suite follows the same
discipline. `Component.cpp` is the canonical reference for the base-contract
archetype the way `Linear.ixx` is the reference for `OperationTraits` dispatch.

The goal is **100% coverage of each module's API surface** — every public and
protected member invoked, and every documented contract (each `@throws`, each
state transition) asserted. That is contract coverage: stronger and more durable
than line coverage, and it doubles as an executable spec.

---

## 0. File and namespace organization

Complete public-API coverage means **one test file per public module**. The test
tree mirrors `Mila/Src` exactly:

```
Src/Dnn/Core/Component.ixx                       -> Tests/Dnn/Core/Component.cpp
Src/Dnn/Components/Activations/Gelu/Gelu.ixx     -> Tests/Dnn/Components/Activations/Gelu/Gelu.Cpu.cpp
```

The **test namespace mirrors the physical path** under `Tests/`, rooted at
`Mila::Tests`:

```
Tests/Dnn/Core/Component.cpp                      -> namespace Mila::Tests::Dnn::Core
Tests/Dnn/Components/Activations/Gelu/Gelu.Cpu.cpp -> namespace Mila::Tests::Dnn::Components::Activations::Gelu
```

This makes location self-describing and gives every file's helper types a unique
enclosing namespace. Note two distinct uniqueness rules:

- **C++ helper types** (harnesses, mock configs) are isolated by the file's
  namespace *and* placed in an inner anonymous namespace — belt and braces
  against ODR collisions in the single `MilaTests` binary.
- **GTest suite names** (the fixture class name in `TEST_F`) are **not** scoped by
  C++ namespace — GTest registers them globally. So fixture names must be unique
  across the whole binary regardless of namespace (`ComponentTests`,
  `ComponentConfigTests`, `GeluCpuTests`, ...).

---

## 1. Four archetypes — test inherited machinery exactly once

| Archetype | Files | Tests | Does NOT test |
|---|---|---|---|
| **Base contract** | `Core/Component.cpp`, `Core/ComponentConfig.cpp` | the inherited lifecycle / state machine, via a spy-capable harness | anything component-specific |
| **Config** | `*Config.cpp` (e.g. `GeluConfig.cpp`) | fluent setters, `validate()`, metadata round-trip | forward / backward |
| **Concrete component** | `*.Cpu.cpp` / `*.Cuda.cpp` (e.g. `Gelu.Cpu.cpp`) | the *delta*: construction, build, forward/backward numerics, component-specific accessors, parameter load | the base machinery (already guaranteed by `Component.cpp`) |
| **Operation** | `<Op>.Cuda.cpp` under `Tests/Dnn/Compute/.../Operations/` (e.g. `CudaLinearOp.Cuda.cpp`) | ONLY the residue **unreachable** through the public component API — in practice the weight-quantization white-box (per-channel/per-group absmax scales, FP4 nibble packing, exactly-representable round-trip). See the reachability rule below | kernel forward/backward numerics and the prefill/decode path split — both reachable through the component, so re-checking them at the op layer is a redundant mirror; all component orchestration |
| **Value type / god-module** | `<Subject>.<Area>.cpp` / `.Cuda.cpp` (e.g. `Tensor.Constructors.cpp`) | a large non-component module (no Component/Operation lifecycle) split by API *area*, each area file exhaustive against the source surface; throws as negative tests; dtype as a typed sweep only where behavior varies by dtype | the component/operation machinery (there is none) |

The payoff is leverage: because `Component.cpp` proves the base contract once,
every concrete component test is short — it asserts only what is new. A new
component test is "copy the skeleton, fill in the numeric reference and the
component-specific accessors."

### The Operation archetype — the reachability boundary

Backend operations are **not** part of the public `import Mila;` surface — they are
implementation detail behind `OperationTraits` dispatch. The default is therefore
the standard rule: **test observable behavior through the public component, not the
implementation.** A component drives its op on every forward/backward, so op
correctness is already exercised transitively; a separate op test that re-checks
the same numerics is a redundant *mirror* that couples the suite to internal
structure and breaks on a kernel refactor that changed nothing a user can see.

So an op test is justified by exactly one question: **is this behavior observable
and assertable through the public component API?**

- **Reachable -> test it only at the component.** Forward/backward numerics, the
  prefill->decode shape regime (section D / §5), mode gating, `loadParameter`
  routing. No op mirror.
- **Genuinely unreachable -> this is the only legitimate op-test residue.**

In this codebase the residue is essentially **one thing: weight quantization.**
`Linear::loadParameter` just forwards a BF16 blob to `operation_->quantize()`, so
the per-channel/per-group absmax scales, the FP4 E2M1 nibble packing, and the
exactly-representable round-trip are invisible at the component surface. The
component can prove only the *black-box* wiring (build the quantized component, let
`loadParameter` drive `quantize()`, assert `forward()` ~= the BF16 reference within
a format-appropriate tolerance, assert `backward()` throws); the *white-box* checks
(scales == host absmax, nibble packing) can only live at the op layer. This residue
is **net-new** coverage — drought-era code with no authored test — tracked as the
quantization backfill item in BACKLOG, *not* a revival of the old op tests.

Consequences of the boundary:

- **CPU op tests have no residue.** CPU has no quantization and its kernels are
  plain reference loops fully exercised by the component, so the authored
  `Cpu*OpTests` are pure mirrors — **delete, do not revive.**
- **CUDA op tests that re-check GEMM / attention numerics** (`CudaMatMulBiasOpTests`,
  `CudaMultHeadAttentionOpTests`, ...) are reachable via `Linear.Cuda` /
  `MultiHeadAttention.Cuda` — **delete.** The only CUDA op test that should *exist*
  is the quantization white-box for `CudaLinearOp`, written fresh against the
  current surface rather than revived.
- **A component whose behavior cannot be reached is a component bug, not an op-test
  mandate.** `GroupedQueryAttention` numerics look "op-only" solely because its
  standalone `forward()` is a no-op stub (a filed bug); the fix is to repair the
  component so the numerics become reachable and test them there — not to enshrine
  the stub by testing the op around it.

Conventions for the quantization op test that does belong here:

- **Instantiate the op via `OperationTraits<...>::type`**, not by naming the
  concrete class. This exercises the dispatch table itself and catches a poisoned
  specialization (the class of bug the Gelu BF16 typed test surfaced — an entry
  that advertises a type whose kernel does not support that precision).
- The op harness **owns what the component's `onBuilding` normally does**: allocate
  and bind the parameter / scale / gradient tensors via `setParameters()` /
  `setWeightScales()` / `setGradients()`, then `build()`. A single op is driven
  standalone, without a parent component. It is GPU-local, like the CUDA component
  tests.

### The value-type / god-module archetype

Some subjects are neither components nor operations: `Tensor`, and the
infrastructure types (`Device`, `ExecutionContext`, the registries). They have no
Component/Operation lifecycle, and the largest — `Tensor.ixx` — is a single module
far too big for the "one test file per module" rule of §0. For these:

- **Split by API area, not by module.** One file per cohesive area
  (`Tensor.Constructors.cpp`, `Tensor.MemoryProperties.cpp`,
  `Tensor.ShapeTransform.cpp`, ...), and the split must be *exhaustive* — every
  public member maps to exactly one area file, audited against a per-subject
  coverage matrix (e.g. `Specifications/Testing.Tensors.md`).
- **The device axis is still a file split.** `<Area>.cpp` (CPU, always compiled)
  plus `<Area>.Cuda.cpp` (device instantiations), never an inline `#ifdef
  MILA_HAS_CUDA`. The host-only contract — accessors constrained by `requires
  is_host_accessible` — is asserted in the `.Cuda.cpp` companion with a
  `requires`-expression proving they are *not callable* on a device-only resource.
- **Sweep dtype only where behavior varies.** `elementSize` / `getStorageSize` /
  `item()` / `data()` host-type mapping vary by data type; shape/stride/view/name/uid
  do not. Sweep the former, single-instantiate the latter — unlike a concrete
  component, where the precision sweep is the blanket default.

### The gradient-check archetype (finite-difference backward verification)

The authored suite was **forward-only** — inference validated forward passes
against HuggingFace, so every `backward()` the training samples drive had zero
coverage. Section F is where that gap is closed, and the reusable verifier lives
in `Tests/Common/GradientCheck.h` so no component re-derives an analytic
reference.

The check is **black-box**. For a fixed upstream gradient `g`, form the scalar
loss `L = sum_j output[j] * g[j]`; then the analytic gradient `backward()`
produces for any buffer (the input, or a parameter tensor) must equal `dL/dbuffer`
measured by central differences of the component's own `forward()`, because
`dL/dx_i = sum_j g[j] * d output[j]/d x_i = (J^T g)_i` — exactly what `backward()`
computes from the same `g`. The verifier therefore carries no per-component math.

Two free functions:

- `centralDifferenceGradient(perturbable, size, upstream_gradient, output_size,
  evaluateOutput, epsilon)` — perturbs each element of `perturbable` by `+/-eps`,
  re-evaluates `L`, and returns the numeric gradient. `evaluateOutput` is a
  `() -> const float*` lambda that runs the component's forward against the
  *current* buffer contents — this is the single seam that absorbs the
  non-uniform forward/backward signatures (e.g. `forward(input)->output&` vs
  `forward(input, output)`), so the helper itself is signature-agnostic.
- `expectGradientsClose(analytic, numeric, abs_tol, rel_tol, label)` — elementwise
  assert with a combined absolute/relative tolerance (relative is required because
  finite-difference magnitudes span orders of magnitude across one layer).

Conventions for a `Backward_MatchesNumericGradient` case:

- **Snapshot the analytic gradient** (`backward()`'s `input_grad`, and each
  `getGradients()` entry) into a `std::vector<float>` *before* probing — the probe
  re-runs `forward()`, which may reuse component-internal buffers.
- **One helper call per differentiated buffer.** A leaf with parameters checks the
  input gradient and each parameter gradient with the same `evaluate` lambda
  (`Linear`/`LayerNorm`: dX, dW, dB).
- **FP32 defaults**: `epsilon = 1e-2`, tolerance `abs ~1e-2 / rel ~1e-2` (stateless
  elementwise leaves like `Gelu` tighten `abs` to `1e-3`). `1e-2` half-steps keep
  cancellation roundoff below the `O(eps^2)` truncation error.
- This is the **precondition for Training Revival's convergence oracle** — a
  per-component numeric gradient check is what makes "the sample converges" a test
  result rather than an eyeball. Reference applications: `Gelu.Cpu.cpp` (stateless
  input grad) and `LayerNorm.Cpu.cpp` (input + parameter grads).

The earlier ad-hoc analytic backward cases (`LayerNorm`, `Softmax`, `Lpe`) stay as
a second, independent oracle — an analytic reference and a finite-difference probe
catch different mistakes (a wrong derivation vs a wrong implementation), so keep
both where they already exist rather than deleting one.

---

## 2. The shared section taxonomy

Every test file uses the same section banners, in the same order. Empty sections
are kept (as a one-line comment) so a missing section is a visible gap, not a
silent one. Coverage becomes auditable by eye before the tool confirms it.

```
A. Construction & Validation       F. Backward (numeric vs analytic gradient)
B. Build Lifecycle                 G. Parameters & Gradients
C. Execution Context / Device      H. Serialization (save_ / round-trip)
D. Runtime + Training Mode         I. Diagnostics (toString, operator<<)
E. Forward (numeric vs reference)  J. Static traits (getDeviceType/getPrecision)
```

- **Base contract** (`Component.cpp`): A–D, G, I, J are live. E/F are absent by
  design — the base has no `forward`/`backward`; those are a concrete-component
  concern. That absence is itself a clarifying boundary.
- **Concrete leaf without parameters** (`Gelu.Cpu.cpp`): A–F, I, J light up; G is
  the "no parameters" contract (counts are zero, vectors empty); H is minimal.
- **Concrete leaf with parameters** (`Linear`): adds the substance of G and H.

---

## 3. The harness (base-contract archetype)

Base classes are abstract, so the *mock is the surface*. The base-contract test
defines a `Component` subclass that:

1. implements the pure-virtual surface trivially,
2. **counts hook invocations** (`onBuilding`, `onExecutionContextSet`,
   `onTrainingModeChanging`) so "the hook fired at the right moment" is assertable,
3. can be told to **throw from a hook** (to drive the failure-restore paths),
4. **surfaces the protected members** through `expose*` forwarders so negative
   paths can be driven without a parent `Network`/`CompositeComponent`.

Harness types live in an **anonymous namespace** (internal linkage) so multiple
test translation units compiled into the single `MilaTests` binary cannot collide.

Concrete component tests reuse the *spy idea* for their own hooks, but do not
re-test the base machinery.

---

## 4. Every documented `@throws` is one negative test

The source Doxygen is the contract. Each `@throws` becomes a `Region_ThrowsWhen…`
test. This binds the suite to the documentation and surfaces drift — writing
these tests is what flushes out stale comments for the Alpha.9 documentation pass.

---

## 5. The testing axes — type axes vs the runtime axis — no `#ifdef`

A component's behavior varies along four axes, and they fall into two kinds that
are handled differently:

- **Type axes** — compile-time template parameters: `TDeviceType`, `TPrecision`,
  and (on quantizable components) `TWeightQuantization`. These drive the *file
  structure* — a file split (device) or a typed sweep (precision, quantization).
- **The runtime axis** — `BuildContext` values: `RuntimeMode`,
  `initialize_parameters`, and the runtime input shape. This is a *value* axis,
  not a type axis; it cannot be a file split or a typed sweep, so it is exercised
  *inside* test bodies (section D + the build-lifecycle shape behavior).

The axes are **not** symmetric, and the asymmetry decides the structure.

### Device is a physical file split (forced by the CI gate)

Under `MILA_ENABLE_CUDA=OFF`, a `<Component><Cuda, ...>` instantiation must never
be compiled — it pulls the CUDA operation and `OperationTraits<..., Cuda, ...>`,
which do not compile without CUDA. So the device axis *cannot* live inside one
translation unit; it has to be the file boundary:

- `Component.Cpu.cpp` (and device-agnostic base logic) goes in the always-compiled
  `add_executable` list so it rides the `MILA_ENABLE_CUDA=OFF` CI gate.
- `Component.Cuda.cpp` goes in the `if(MILA_ENABLE_CUDA)` `target_sources(...)`
  block in `Tests/CMakeLists.txt`. New `*.Cuda.cpp` / `Cuda*.cpp` files belong in
  that block, never in the always-compiled list.

This is the structural reason the ratchet stays green: a CPU-only build compiles
and runs the CPU set without ever instantiating the CUDA path.

### Precision lives *inside* the device file

A given device's supported precisions all compile together (no gate involved), so
precision is **not** a file split and **not** a namespace segment — it is a type
parameter within the device file, driven by a `TYPED_TEST` sweep over a
per-precision tag list (`::testing::Types<...>`).

**The sweep scaffold is the default for every concrete-component test, even one
that currently supports a single precision.** Any component *could* be templated on
a precision it does not have a kernel for yet — single-precision-ness is almost
always *contingent* (nobody has written the BF16 kernel) rather than *fundamental*
(the operation is mathematically FP32-only). Attention proves the point: MHA is
FP32 today, but `GroupedQueryAttention` runs BF16, so a BF16 MHA kernel is a
plausible future, not an impossibility. Collapsing a single-precision test to a
plain `TEST_F` bakes in the contingent assumption and turns "add a precision" into
a full re-scaffold; keeping the sweep makes it a one-line edit.

So:

- **Single supported precision** — a one-entry list, e.g.
  `using MhaPrecisions = ::testing::Types<Fp32Precision>;`, with a comment naming
  why the others are absent (no kernel yet) and how to add one. The bodies are
  already precision-general.
- **Multiple supported precisions** — list them all, e.g. `Gelu.Cuda.cpp` over
  `Types<Fp32Precision, Bf16Precision>` for `Gelu<Cuda, FP32>` / `Gelu<Cuda, BF16>`.

The supported-precision list is the **single point of change**: the day a device
gains a precision, add one tag and the suite re-runs every body for it — no
structural edit.

(A handful of early revival tests — e.g. `Gelu.Cpu.cpp`, `LayerNorm.Cuda.cpp` —
were written as plain single-precision `TEST_F`s before this rule settled. They are
correct as-is; migrate them to the one-entry sweep opportunistically when next
touched, not as a dedicated pass. The base-contract and config archetypes are not
precision-swept — this rule is about concrete-component forward/backward tests.)

Numeric references are precision-independent — compute them once in `float`. Only
two things vary per precision, so they go in a small per-precision traits struct:
the **tolerance** (FP32 ~`1e-4`, BF16 ~`1e-2`) and a **read-as-float accessor**
for comparing a reduced-precision tensor element against the float reference.

### Weight quantization is a type axis — but an op-layer one

`TWeightQuantization` (on `Linear`, e.g. `PerChannelFp8<>`, `PerGroupFp4<128>`) is
a compile-time type like precision, but it is **not** a symmetric precision-style
sweep, for three reasons: it is **CUDA-only** (no CPU specialization),
**inference-only** (`backward` throws `std::logic_error`), and it **changes the
testable surface** — the weight tensor is a packed reduced-precision dtype (FP4 /
INT4 store two nibbles per byte, so the column count is `input_features/2`), a
`weight_scales_` tensor appears, and loading runs `quantize()` rather than a plain
copy. Same bodies do not fit.

So the quantization sweep lives at the **operation layer** (see §1, the Operation
archetype), where the surface (`forward` + `quantize`) *is* uniform — a
`TYPED_TEST` over `Types<Fp8, Fp4<128>, Fp4<64>, ...>` in the op test. The
component only proves the wiring black-box. Two reference strategies:

- **White-box quantize round-trip** (op test): use weights **exactly
  representable** in the target format (small values / powers of two) so the
  dequantized result matches losslessly and the scale/packing assertion is tight
  and deterministic.
- **Realistic forward accuracy** (op and component): compare against the **BF16
  forward of the same weights** with a format-appropriate tolerance — FP8 is a few
  percent; FP4 E2M1 is coarse and needs a generous budget.

### The build context is the runtime axis — section D, not a split

`BuildContext` is a runtime **value** axis, so it is never a file split or a typed
sweep — it is driven by building / forwarding with different contexts inside test
bodies, under section **D (Runtime + Training Mode)** plus the build-lifecycle
shape behavior (section B). It has three dimensions:

1. **`RuntimeMode` Inference vs Training** — gradient buffers exist only for
   Training; `backward` throws on an Inference-built component; `setTrainingMode`
   (`Normal`<->`Eval`) is legal only on a Training-built one, and the component's
   `onTrainingModeChanging` effect (clearing / rebinding gradients) is part of the
   delta.
2. **Prefill -> decode runtime shapes** — build for a prefill shape `{B, T, C}`,
   then `forward` a decode shape `{B, 1, C}` (`outer_size == 1`). This is the
   inference hot path. For a quantized op, `outer_size == 1` selects the
   `matvec_decode_*` kernels — so this dimension and the quant decode-path coverage
   are the **same test seen from two axes**. Assert output shape and values; do
   **not** assert buffer-reuse internals (an implementation detail).
3. **`initialize_parameters` true / false** — `false` (pretrained load; values
   filled later by `loadParameter`) is what inference uses — test it now. The
   `true` (train-from-scratch) assertions are **deferred to Alpha.8**, because the
   active `xavier` is currently a no-op stub; asserting init=true now would only
   codify a known gap.

### Naming recap

- **File:** `<Component>.<Device>.cpp` — device is the only discriminator in the
  name; **precision never appears in the filename**.
- **GTest suite:** `<Component><Device>Tests` (`GeluCpuTests`, `GeluCudaTests`).
  Typed-test instances suffix the precision via a type-name generator
  (`GeluCudaTests/Fp32`, `GeluCudaTests/Bf16`) rather than `/0`, `/1`.

---

## 6. Conventions

- **One concept per `TEST_F`.** Name `Region_Behavior`:
  `Build_ThrowsWhenAlreadyBuilt`, `TrainingMode_ThrowsOnInferenceBuilt`.
- **Fixture and helper names are unique per file**; helpers go in an anonymous
  namespace. The fixture class name is the GTest suite name and must be unique
  across the whole `MilaTests` binary.
- **Numeric tests carry a host reference function and an explicit tolerance.**
  The reference is the spec (e.g. the GELU tanh approximation), computed
  independently of the component under test.
- **Compile-time contracts use `static_assert`**, not only runtime checks
  (e.g. `getDeviceType()` / `getPrecision()` are `static constexpr`).
- **ASCII only**; match the surrounding `Mila/Src` formatting (single-space, blank
  line before control-flow blocks).

---

## 7. Coverage workflow

Coverage is measured with the Visual Studio 2026 built-in tool, by file, after
each addition. There is **no CI coverage gate yet** — the bar is met *by
construction*: the section-per-region structure makes gaps visible on inspection,
and the tool confirms. An automated coverage gate is a later CI-ratchet task; it
is deliberately not a prerequisite for the revival.

---

## 8. Adding a test for a new component (the checklist)

1. Start from `Gelu.Cpu.cpp` (leaf without parameters) or `Linear.*.cpp` (leaf
   with parameters), whichever the new component resembles.
2. Keep the section banners; delete only the sections that genuinely do not apply.
3. Replace the numeric reference (E/F) with the component's spec.
4. Replace the component-specific accessor assertions (e.g.
   `getApproximationMethod()` for Gelu, `getInputFeatures()` for Linear).
5. Do **not** re-test the inherited base contract — it is covered by
   `Component.cpp`.
6. Cover the **runtime axis** in section D: Inference vs Training (gradients,
   backward legality, `setTrainingMode` transitions), the prefill->decode shape
   regime, and the `initialize_parameters=false` path.
7. If the component has a quantization axis or non-trivial kernels, add the
   **operation** test (`<Op>.Cuda.cpp` / `.Cpu.cpp`) for the internal surface — the
   quantization sweep and the prefill/decode path split — keeping the
   component/operation division of labor from §1.
8. Put CUDA-instantiation tests in the `*.Cuda.cpp` companion.
9. Run VS 2026 coverage on the module; close any section gap.
