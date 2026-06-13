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

## 1. Three archetypes — test inherited machinery exactly once

| Archetype | Files | Tests | Does NOT test |
|---|---|---|---|
| **Base contract** | `Core/Component.cpp`, `Core/ComponentConfig.cpp` | the inherited lifecycle / state machine, via a spy-capable harness | anything component-specific |
| **Config** | `*Config.cpp` (e.g. `GeluConfig.cpp`) | fluent setters, `validate()`, metadata round-trip | forward / backward |
| **Concrete component** | `*.Cpu.cpp` / `*.Cuda.cpp` (e.g. `Gelu.Cpu.cpp`) | the *delta*: construction, build, forward/backward numerics, component-specific accessors, parameter load | the base machinery (already guaranteed by `Component.cpp`) |

The payoff is leverage: because `Component.cpp` proves the base contract once,
every concrete component test is short — it asserts only what is new. A new
component test is "copy the skeleton, fill in the numeric reference and the
component-specific accessors."

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

## 5. Device and precision axes — no `#ifdef`

Concrete components are templated on two independent axes, `TDeviceType` and
`TPrecision`. They are **not** symmetric, and the asymmetry decides the file
structure.

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
parameter within the device file:

- When a device supports a **single** precision, test it explicitly — no
  machinery. (CPU `Gelu` is FP32-only, so `Gelu.Cpu.cpp` is a plain `Gelu<Cpu,
  FP32>` test.) **CPU tests are pragmatic this way by default.**
- When a device supports **multiple** precisions, parameterize the device file
  with a `TYPED_TEST` over that device's supported set. (`Gelu.Cuda.cpp` runs the
  same bodies for `Gelu<Cuda, FP32>` and `Gelu<Cuda, BF16>` over
  `::testing::Types<...>`.)

The supported-precision list is the **single point of change**: the day a device
gains a precision, add one entry and the suite re-runs for it.

Numeric references are precision-independent — compute them once in `float`. Only
two things vary per precision, so they go in a small per-precision traits struct:
the **tolerance** (FP32 ~`1e-4`, BF16 ~`1e-2`) and a **read-as-float accessor**
for comparing a reduced-precision tensor element against the float reference.

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
6. Put CUDA-instantiation tests in the `*.Cuda.cpp` companion.
7. Run VS 2026 coverage on the module; close any section gap.
