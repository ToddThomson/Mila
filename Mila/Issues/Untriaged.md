# Untriaged

Captured, not yet judged. Writing here needs no judgement, which is the whole point — finding
something mid-task leaves seconds, and a format that asks for more is a format that goes unused.
Facts may grow; judgement may not.

Entries are **deleted unexamined at the release tag**, and anything user-reported is a pointer to
its GitHub issue rather than a copy. Triage flow, categories and the entry format are in
[README.md](README.md); entries here carry an anchor where the judged files carry tags.

---

## The packaging fixtures are an unread detector for source-tree pollution

`Mila/Tests/Packaging/{fetchcontent,cpm}_consumer/Mila/Adaptors/Inference/Server/mila.cp313-win_amd64.pyd`
@ `a87e8315`

Two 16 MB fossils dated 2026-07-21, left by the third binding destination that
`Mila/Bindings/CMakeLists.txt:112-118` records as removed. They landed there because the copy was
source-relative and a subproject build makes the consumer's root `CMAKE_SOURCE_DIR` — the same
defect class as the `tokenize` and wheel-VERSION items, both closed at `+40`. Nothing looks in
these directories, so the evidence sat for six weeks. Found widening the FetchContent gate. The
open question is whether the gate should assert the fixture directories are clean afterwards,
which is what would have caught this in a day rather than six weeks.

## MSVC fails to complete a reachable EXPLICIT SPECIALIZATION, and there is a fix that publishes nothing

`Mila/Src/Dnn/Compute/Devices/Cuda/CudaExecutionContext.ixx:41` @ `a0070f64`

Narrowed with an identical plain-import-only consumer chain, varying only the form of the
dereferenced type: an explicit specialization of a declared-only primary FAILS, a defined primary
class template PASSES, a plain non-template class PASSES. So the trigger is explicit
specialization, not visibility — which is why operations complete through
`OperationTraits.Cuda.ixx`'s plain imports while `ExecutionContext<Cuda>` does not, and why Mila
already carries three instances of the same workaround (`Mila.ixx:207` for the `OperationTraits`
specializations, `TensorOps.ixx:6` for `TensorOps<Cuda>`, and the operation change at `+41`).
A fix was validated: make the dereferenced type a plain class and reduce the specialization to a
one-line traits map alone in its own module, `export import`ed. That module exports no entities at
all — a specialization cannot carry `export` — yet it satisfies MSVC, and the consumer still gets
`C2039` when naming the type. An alias template on its own does not work; it relocates the
completeness demand onto the traits (`C2794`). For Mila that means `CudaExecutionContext` becomes a
real plain class, `ExecutionContext<TDeviceType>` an alias template over the map, every existing
spelling unchanged; the same shape lets `TensorOps.ixx:6` stop publishing the CUDA backend, and it
would supersede the `+41` workaround.

## Superseded first diagnosis: MSVC completes a reachable class only when it is also visible

`Mila/Src/Dnn/Compute/Devices/Cuda/CudaExecutionContext.ixx:41` @ `a0070f64`

Under `[module.reach]` a class is complete where its definition is *reachable*, and interface
dependency is transitive and indifferent to `export`. MSVC 14.51 demands visibility instead, so a
consumer instantiating any CUDA component gets `C2027: use of undefined type
ExecutionContext<Cuda>` pointing into a module it never named. Clang 21.1.8 compiles the identical
construct clean; a negative control (specialization declared, never defined) makes Clang reject it
at the same line, so the pass is not a skipped instantiation. Isolated to one syntactic act: a 2x2
holding everything else fixed shows `auto*` and an explicit pointer type both fail *with* a
dereference and both pass without one — deduction of a pointer needs only a declaration.
`export` on the specialization is not an option, that is `C7760`. Reproduced in nine files with no
CUDA and no Mila (a primary template exported from one module, its specialization defined
unexported in a partition, a template that dereferences it in a module the umbrella reaches only
through plain imports); the repro was built in a session scratchpad and is **not preserved in
tree**. Not filed with Microsoft. The operation layer is worked around at `+41` by exporting the
import in the 19 modules that dereference, which publishes nothing because nothing re-exports
them; `CudaTensorOps.*` and `TensorOps.ixx:6` are the same defect where that is not available.

## `Dnn.TensorOps` publishes the entire CUDA backend to work around the same defect

`Mila/Src/Dnn/Tensors/Operations/TensorOps.ixx:6` @ `a0070f64`

`export import Compute.CudaTensorOps;` is load-bearing: the device-neutral `copy()` dispatches
through `TensorOps<device>::copy(...)`, and `TensorOps<Cuda>` is itself a specialization defined
unexported at `CudaTensorOps.ixx:18` — the same construct as `ExecutionContext<Cuda>`. So a
consumer needs it complete, and the export is what supplies that. The cost is that `import Mila;`
hands every consumer `ZeroOps`, `FillOps`, `MathOps`, `TransferOps`, `StructuralOps` and
`RandomOps` as a side effect of wanting `copy()`. Nothing outside `Mila/Src` names any of them.
Demoting the line reintroduces the C2027 on `TensorOps<Cuda>`, so this and the six
`CudaTensorOps.*` partitions are one decision, not two.

## Fossils of earlier attempts at the same defect, still in the tree

`Mila/Src/Dnn/Compute/Devices/Cuda/Tensors/Operations/CudaTensorOps.Zero.ixx:36` @ `a0070f64`

Three layers of abandoned attempt. `//import Compute.CudaExecutionContext;` at `Fill.ixx:40` and
`Math.ixx:41` names a module that has never existed — the file is `CudaExecutionContext.ixx` but it
declares the partition `Compute.ExecutionContext:Cuda`. `class CudaExecutionContext;` at
`Zero.ixx:36` and `Structural.ixx:35` does not forward-declare the real alias at
`CudaExecutionContext.ixx:479` — an alias cannot be forward-declared, so it introduces a new,
never-defined class in the nested `::Cuda` namespace, which then shadows the real one for
unqualified lookup and would yield the identical "undefined type" error from a different cause.
Nothing names it, so the shipped code is correct; `cast_context_` is a template in `::Compute`
reached through the using-directive and never spells the alias. Separately, all six device-neutral
partitions (`TensorOps.{Zero,Fill,Math,Transfer,Structural,Random}.ixx`) import
`Compute.ExecutionContext` while never naming anything from it.

## Public Doxygen on `Dnn.TensorOps` shows an example that cannot compile

`Mila/Src/Dnn/Tensors/Operations/TensorOps.Transfer.ixx:76` @ `a0070f64`

`auto ctx = std::make_unique<CudaExecutionContext>( 0 );` appears in `@code` blocks at
`TensorOps.Fill.ixx:84` and `:126`, `TensorOps.Transfer.ixx:76`, `TensorOps.Math.ixx:74`, and the
`CudaTensorOps.*` mirrors at `Transfer:104`, `Fill:101`/`:196`, `Math:96`. It fails twice: the
constructor takes a `DeviceId`, whose only constructor is `( DeviceType, int )`
(`DeviceId.ixx:61`) and which is not constructible from `0`; and a consumer cannot name
`CudaExecutionContext` at all, since `Compute.ExecutionContext` is never exported from the
umbrella. `Dnn.TensorOps` is public API, so this is what a consumer reads.

## The Metal and ROCm execution contexts cannot compile, and one specializes on Vulkan

`Mila/Src/Dnn/Compute/Devices/Rocm/RocmExecutionContext.ixx:28` @ `a0070f64`

Both write `export template<>` on an explicit specialization, which is `C7760` — illegal C++, not a
compiler quirk. Neither file is ever compiled (`MILA_HAS_METAL` and `MILA_HAS_ROCM` are never
defined), so the error is latent: anyone enabling either flag gets a compile failure on the first
file. The Cpu and Cuda siblings write `template<>` correctly, which is why they build. Separately,
`RocmExecutionContext.ixx:29` specializes `ExecutionContext<DeviceType::Vulkan>` in a file named
Rocm. Found while grepping for alias-template risk, not while working on either backend.

## One alias declared twenty-two times, and four no-op casts

`Mila/Src/Dnn/Compute/Devices/Cuda/CudaExecutionContext.ixx:479` @ `a0070f64`

`export using CudaExecutionContext = ExecutionContext<DeviceType::Cuda>;` exists at namespace scope
(`:479`), and thirteen operation and optimizer files redeclare the identical alias at class scope.
The CPU side repeats the pattern: `CpuExecutionContext.ixx:71` declares the alias and seven CPU
operation files redeclare it. Separately, `CudaSwigluOp.ixx:75` and `:105`, `CudaGeluOp.ixx:142`
and `CudaGegluOp.ixx:75` do `static_cast<CudaExecutionContext*>( context_ )` where `context_` is
already declared `CudaExecutionContext*`.
