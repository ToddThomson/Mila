# Tensor Suite Coverage Matrix

A per-suite coverage checklist for the `Tensors/` test revival (Alpha.7). This is
the worklist and the definition of "done": every exported member of the target
modules maps to a row here, and the suite is complete when every row is `covered`.

Companion to [Testing.md](Testing.md) — the general methodology. This file is the
concrete instance for the Tensor cluster, and the proving ground for the
**value-type / god-module archetype** Testing.md does not yet document (see
"Methodology decisions" below).

**Scope of this pass: `Tensor.ixx` core only** — the `Tensor<TDataType,
TMemoryResource>` class, its free operators, and the device aliases. The wider
`Tensors/` tree (`TensorBuffer`, the four `TensorDataType*` maps, `ITensor`,
`Tensor.{Types,Helpers,Partitioning,Initializers,Serialization}`, and
`Operations/{Fill,Math,Zero,Transfer,Structural,Random}`) is a follow-on slice and
is intentionally *not* enumerated here yet.

---

## Progress (Alpha.7)

**Core `Tensor.ixx` complete (methodology-aligned, awaiting VS2026 build).** All eight area
files: `Constructors`, `DataAccess`, `DataPointers`, `Identity`, `Io`, `MemoryProperties`,
`Properties`, `ShapeTransform` — each on the `Mila::Tests::Dnn::Tensors` namespace, with the
device axis as a file split (every inline `#ifdef MILA_HAS_CUDA` removed) and `.Cuda.cpp`
companions for `Constructors`, `DataAccess`, `MemoryProperties`, `ShapeTransform`. New coverage
this slice: `elementSize`/`getStorageSize`, the whole shape-transform area (previously no file),
`item()` + scalar/non-scalar negatives, and the device-tensor host-only SFINAE contract
(`requires is_host_accessible` proven non-callable). The value-type / god-module archetype is now
documented in [Testing.md](Testing.md) §1.

**Next (wider tree, follow-on slice):** `TensorBuffer`, the `TensorDataType*` maps, `ITensor`,
`Tensor.{Partitioning,Initializers,Serialization}`, and the `Transfer` device-split.
`Operations/{Fill,Math,Zero,Structural}` are done; `Random` stays deferred behind the CUDA
FP32-only `fill_normal` bug.

**Methodology note added 2026-07-28 (from the `Structural` slice):** the "CPU file plus `.Cuda.cpp`
companion" rule assumes every op has a CPU backend. `split` does not, so it is a `.Cuda.cpp`-only
area. Check for the CPU implementation before assuming the pair — an area with no CPU op gets one
file, not an empty one.

---

## File decomposition (the god-module exception)

`Tensor.ixx` is a single module far too large for the Testing.md "one test file per
module" rule, so the Tensor suite splits by **API area** instead. The split must be
exhaustive: every public member belongs to exactly one area file, and every area
file has a CPU unit (always compiled, rides the `MILA_ENABLE_CUDA=OFF` gate) plus a
`.Cuda.cpp` companion for device-tensor instantiations (no inline `#ifdef
MILA_HAS_CUDA` — the device axis is a file split).

| Area file | API area | Status today |
|---|---|---|
| `Tensor.Constructors.cpp` / `.Cuda.cpp` | construction, move, deleted copy, validateDeviceId | active (pre-methodology); **no `.Cuda.cpp`** |
| `Tensor.MemoryProperties.cpp` / `.Cuda.cpp` | type info + interface compliance, accessibility statics, getMemoryResource | active (pre-methodology); **no `.Cuda.cpp`** |
| `Tensor.DataAccess.cpp` / `.Cuda.cpp` | scalar `item()`, multi-dim `operator[]` | active (pre-methodology); **no `.Cuda.cpp`** |
| `Tensor.DataPointers.cpp` / `.Cuda.cpp` | `data()`, `rawData()` | active (pre-methodology); **no `.Cuda.cpp`** |
| `Tensor.Properties.cpp` | shape, strides, size, rank, empty, isValid, getDeviceId | **dark** (commented) |
| `Tensor.ShapeTransform.cpp` / `.Cuda.cpp` | `view`, `isView`, `reshape`, `flatten` | **missing** (no file) |
| `Tensor.Identity.cpp` | uid, name, setName | **dark** (commented) |
| `Tensor.Io.cpp` | `toString`, `operator<<` | **dark** (commented) |

`Tensor.Initializers.cpp` (commented) targets `Tensor.Initializers.ixx`, a separate
module — it belongs to the wider-tree slice, not core.

---

## Axis treatment

- **Memory resource = file split.** CPU instantiation in `Tensor.<Area>.cpp`; device
  instantiation (`DeviceTensor`/`PinnedTensor`/`UniversalTensor`) in
  `Tensor.<Area>.Cuda.cpp`. Delete every inline `#ifdef MILA_HAS_CUDA` from the
  existing files.
- **Data type = `TYPED_TEST` sweep, only where behavior varies by dtype.** Sweep the
  dtype-dependent members; test the structural members once. Marked per row below.
- **Negative test per `@throws`** (Testing.md §4) — enumerated in the throws list.

dtype-swept members: `elementSize`, `getStorageSize` (incl. the sub-byte FP4
nibble-packed path), `getDataType`, `getDataTypeName`, `item()`, `data()`,
`host_value_t`. Everything else is dtype-independent (single instantiation).

---

## Coverage matrix — `Tensor.ixx`

Status legend: `[x]` covered & methodology-aligned · `[~]` covered but
pre-methodology (re-green) · `[d]` dark (commented file) · `[ ]` missing.

### Construction, Assignment, Destruction -> `Tensor.Constructors.cpp` (+`.Cuda.cpp`)

- [~] `Tensor(DeviceId, shape, name={})` — CPU; scalar `{}`, empty (zero in shape), named
- [ ] `Tensor(DeviceId, shape, name={})` — device (Cuda/Pinned/Universal) `.Cuda.cpp`
- [~] move constructor — leaves source in invalid state
- [~] move assignment — self-assignment safe, source invalidated
- [x] deleted copy ctor / copy assign — `static_assert(!is_copy_constructible)` etc.
- [ ] `~Tensor` RAII / buffer release (shared_ptr refcount via a view)
- [~] `validateDeviceId` device-type mismatch throw (currently inline `#ifdef`)

### Type Information & Interface -> `Tensor.MemoryProperties.cpp` (+`.Cuda.cpp`)

- [~] `getDeviceType()` — CPU; [ ] device in `.Cuda.cpp`
- [~] `elementSize()` — **sweep** (dtype size; REVIEW: sub-byte returns bytes)
- [~] `getStorageSize()` — **sweep**; 0 when no buffer; **FP4 sub-byte packed path**
- [~] `getDataType()` / `getDataTypeName()` — **sweep**
- [x] `is_host_accessible()` / `is_device_accessible()` — `static_assert`, CPU + device
- [~] `getMemoryResource()` — non-null with buffer, null when empty

### Scalar & Multi-dim Access -> `Tensor.DataAccess.cpp` (+`.Cuda.cpp`)

- [~] `isScalar()` — true for `{}`, false otherwise
- [~] `item()` non-const / const — **sweep**; read/write round-trip (host only)
- [~] `operator[](index_t)` non-const / const — **sweep**; read/write
- [~] `operator[](Indices...)` variadic non-const / const
- [ ] device tensor: `item()`/`operator[]`/`data()` unavailable — `requires
  is_host_accessible` SFINAE contract asserted in `.Cuda.cpp` (not callable)

### Data Pointers -> `Tensor.DataPointers.cpp` (+`.Cuda.cpp`)

- [~] `data()` non-const / const — **sweep**; nullptr when no buffer; view offset applied
- [~] `rawData()` non-const / const — byte offset = `view_offset_ * elementSize`
- [ ] device `rawData()` non-null (`.Cuda.cpp`); `data()` host-only contract

### Properties & Introspection -> `Tensor.Properties.cpp` (dark)

- [d] `getDeviceId()`, `shape()`, `strides()`, `size()`, `empty()`, `rank()`
- [d] `empty()` vs scalar (scalar size==1 is NOT empty; zero-in-shape IS empty)
- [d] `isValid()` — note FIXME (moved-from state undefined); assert current contract

### Shape Transforms -> `Tensor.ShapeTransform.cpp` (missing) (+`.Cuda.cpp`)

- [ ] `view(new_shape, offset)` — shares buffer (refcount), accumulates offset, `name+".view"`
- [ ] `isView()` — true on view, false on owner
- [ ] `reshape(new_shape)` — preserves count; empty->grow allocates
- [ ] `flatten()` — no-op rank<=1; collapses to 2D `{prod(0..n-1), last}`

### Identity & Metadata -> `Tensor.Identity.cpp` (dark)

- [d] `getUId()` — unique across instances (atomic generator)
- [d] `getName()` / `setName()` — round-trip; empty-name throw

### String Representation -> `Tensor.Io.cpp` (dark)

- [d] `toString(showBuffer=false)` — uid/name/shape/dtype/device fields present
- [d] `toString(true)` — buffer content (host); "not host-accessible" on device
- [d] `operator<<` free function — delegates to `toString`

### Aliases (exported)

- [x] `HostTensor` (CPU) — exercised throughout the CPU files
- [ ] `DeviceTensor` / `PinnedTensor` / `UniversalTensor` — exercised in `.Cuda.cpp` files

---

## Negative tests (every documented `@throws` -> one test)

| Throw site | Exception | Target file |
|---|---|---|
| `validateDeviceId` device/MR mismatch | `runtime_error` | Constructors |
| `item()` on non-scalar | `runtime_error` | DataAccess |
| `operator[]` on scalar | `runtime_error` | DataAccess |
| `validateIndices` rank mismatch | `runtime_error` | DataAccess |
| `validateIndices` out of range | `out_of_range` | DataAccess |
| `reshape` size mismatch (non-empty) | `runtime_error` | ShapeTransform |
| `view` on tensor with no buffer | `runtime_error` | ShapeTransform |
| `view` offset+size exceeds buffer | `invalid_argument` | ShapeTransform |
| `setName("")` empty | `invalid_argument` | Identity |
| `detail::getStorageSize` overflow | `overflow_error` | MemoryProperties |

---

## Source markers surfaced (-> Alpha.9 doc queue, not fixed in the test pass)

Writing the contract tests pins these `Tensor.ixx` markers as decisions, not silent
debt: `getStorageSize` duplication REVIEW (line ~83); `elementSize` sub-byte REVIEW
(~263); `isValid()` moved-from FIXME (~474); `computeSize` accumulate-overflow
REVIEW/TODO (~798); the protected-vs-public `rawData`/`getMemoryResource` TJT review
(~632). Record each in the test as "asserts current contract" so a later doc/refactor
pass has a green oracle.

---

## TensorOps coverage (Operations/ subtree)

Free-function ops dispatched by `TMemoryResource::device_type` to `TensorOps<Device>`.
Each op module is its own area; CPU file (`<Op>.Cpu.cpp`, gate-riding) + `<Op>.Cuda.cpp`.
Behavior is dtype-independent for most, so a float + integer case suffices.

| Op module | Functions | Status |
|---|---|---|
| `:Zero` | `zero` | `[x]` **done** — `Zero.Cpu.cpp` + `Zero.Cuda.cpp` (methodology) |
| `:Fill` | `fill` (scalar + vector) | `[x]` **re-greened** — namespace + header; CPU/CUDA already split, promoted to Section 1 |
| `:Math` | `add` `subtract` `multiply` `divide` `sum` + operators `+ - * /` | `[x]` **re-greened** — namespace + header; CPU/CUDA already split, promoted to Section 1 |
| `:Transfer` | `copy` `toHost` (+ device variants) | `[~]` namespace + header re-greened (Section 2); **device-split pending** — inherently cross-device (shared fixture builds both contexts), so lifting the `#ifdef MILA_HAS_CUDA` cases into a `TensorOps.Transfer.Cuda.cpp` with its own fixture is a dedicated follow-up |
| `:Structural` | `split` (2 overloads) | `[x]` **done** — `Structural.Cuda.cpp`. **No `.Cpu.cpp`: `split` is CUDA-only** (no `CpuTensorOps::split`), the one area where the CPU-file-plus-companion rule does not apply. Covers both overloads, the null-context default-stream path, and every precondition throw. Writing it found an out-of-bounds write: the alignment precondition was a flat `D % 4` while the BF16 kernel moves 8 elements per 16-byte `uint4`, so a BF16 width of 4 passed validation and then stored 8 elements into a 4-element row (see BACKLOG) |
| `:Random` | `fill_normal` `fill_uniform` `xavier` | `[ ]` **deferred -> Alpha.8** — training-init; CUDA path has the known FP32-only bug parked in Training Revival. Do not write CUDA numeric tests against the broken path |

---

## Methodology decisions (feed back into Testing.md)

1. **God-module / value-type archetype** — Tensor is the first non-component, non-op
   subject. The area-split (this file's table) plus the "sweep only where dtype
   matters" rule is the archetype; promote it into Testing.md as a fifth archetype
   once validated here, so Device/ExecutionContext/Registry reuse it.
2. **`static_assert` contracts** — copy-deletion, the host-accessibility `requires`
   clauses, and the accessibility statics are compile-time; assert them with
   `static_assert`, not only runtime checks (Testing.md §6).
