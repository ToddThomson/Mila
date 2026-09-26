---
title: "C++23 Modules at Scale: 338 Modules, MSVC, Clang, and CUDA"
date: 2026-09-22
draft: true
description: "What building an LLM library entirely in C++23 modules taught me - 338 module units, seventeen months on MSVC alone, then Clang 21 - and the one problem still open."
---

Andrej Karpathy's [llm.c](https://github.com/karpathy/llm.c) is GPT-2 in plain C, in one file. Mila
started from the same model and the same idea - understand it by building it - and went the other
way. Today it is a C++23 library that runs models up to Qwen 3.8 27B, and every line of it is
modules: 338 module units and about 95,000 lines of C++ in the library, beside 24,000 lines of CUDA.

There are not many modules codebases this size in public, and most of what I learned building one
is written down nowhere else. This is that list.

## How it got here

The repository is older than the library. In January 2022 an earlier project in it - a recurrent
network library on cuDNN - was converted to C++20 modules, 48 of them, on MSVC. Then the commits
stop for three years.

What happened in between was the pandemic, and Andrej Karpathy. I'm retired, I was stuck inside the
house, and I started watching his lectures on neural networks and transformers. I was hooked. Then
came llm.c.

The Mila that exists now started in January 2025 as a port of llm.c's GPT-2, and it was modules
from the first line. It grew steadily: 45 module files that month, 120 by May, 200 by November,
309 by May 2026. Llama arrived in January 2026, Gemma 4 in June, Qwen 3.8 in August.

For the first seventeen months it built on exactly one compiler, MSVC. Clang 21 built it for the
first time in June 2026. That split matters to everything below: most of
the lessons come from the MSVC-only period, and the portability lessons come from the day that
ended.

## How the code is laid out

Three rules do most of the organizing:

- **One type per module file, and the module name follows the path.**
  `Src/Distribution/ModelPackage.ixx` is `Distribution.ModelPackage`. The directory is the grouping
  and the file is the unit, so navigating the modules and navigating the tree are the same thing.
- **Backends are partitions.** CUDA and CPU implementations are `:Cuda` and `:Cpu` partitions of one
  module, so the device-neutral code imports one name.
- **`import Mila;` is the public API.** One umbrella module re-exports everything a consumer may
  use, and its export list is the specification of the public surface.

The type system does the dispatch. Device, precision and weight quantization are template
parameters, and each component resolves its concrete operation at compile time:

```cpp
using QuantizedProjection =
    Linear<DeviceType::Cuda, TensorDataType::BF16, PerGroupFp4<128>>;
```

CUDA cannot join the module graph - nvcc does not consume modules - so the kernels live in ordinary
`.cu` files that export plain launcher functions through a `.cuh` header. A `:Dispatch` partition
includes that header in its global module fragment and maps each element type to its launcher:

```cpp
module;
#include <cublasLt.h>
#include <cuda_bf16.h>
#include "Kernels/Residual.cuh"

export module Compute.CudaResidualOp:Dispatch;
```

The global module fragment is the seam. Everything above it is modules, everything below it is
nvcc, and the two meet in one small, named place per operation. nvcc also uses its own host
compiler for the `.cu` files, so the module compiler and the CUDA host compiler are separate
choices: on Linux, Clang 21 compiles the modules while GCC 15 hosts nvcc.

## Lesson 1: reachable is not visible

In July 2026 I tried to narrow the public API by dropping a handful of internal-looking modules
from the umbrella - the operation base class, the traits that select an operation, the tensor's
storage buffer. No consumer named any of them. A search of the codebase confirmed it.

The result was hundreds of compile errors, and a full revert.

Under MSVC, a type named in a public template's interface - a member or a base class - must be
*visible* in the consumer's translation unit when the template is instantiated, not merely
*reachable*. `Tensor<>` holds a `std::shared_ptr<TensorBuffer<...>>`. Every component derives from
`Operation<>` and names `OperationTraits` in a member alias. A consumer that instantiates
`Tensor<>` or `Linear<>` needs those modules imported, even though it never writes their names.

Two consequences I did not expect:

- **Searching the code proves nothing.** The dependency exists only at instantiation, so the only
  reliable check is a full rebuild.
- **It fails asymmetrically.** `Linear<Cuda, BF16>` compiled through the umbrella, because a
  *default* template argument only needs to be reachable. `Linear<Cuda, BF16, PerChannelFp8<>>` -
  the spelling the documentation uses - did not, until the quantization policies were exported too.

The umbrella now carries a comment explaining why its export list is as wide as it is, so the next
person who tries to tidy it does not have to rediscover this.

## Lesson 2: a dependent use of a reachable specialization

This was the hardest one to isolate.

`ExecutionContext<Cuda>` was an explicit specialization, and nineteen CUDA operations held a pointer
to one. They failed to compile in consumers with `C2027: use of undefined type` - the
specialization was reachable but MSVC would not complete it. The workaround was an `export import`
in all nineteen modules, which published the entire CUDA backend through the public API.

What finally explained it was one file, one module and one consumer: a `static` member reading a
parameter compiled, and a member function reading `context_` did not. **The trigger was dependence,
not visibility.** When the dereference is non-dependent, MSVC binds it inside the module that
imported the type. When it is dependent - a member of the enclosing class template - instantiation
is deferred to the consumer's translation unit, and there the specialization is only reachable.

The fix has become a pattern across the codebase:

```cpp
// The concrete type is a plain class in its own module.
// The specialization shrinks to a one-line traits map, alone in a module:
template<>
struct ExecutionContextTraits<DeviceType::Cuda>
{
    using type = CudaExecutionContext;
};

// The public name is an alias template over the map.
export template<DeviceType TDeviceType>
using ExecutionContext = typename ExecutionContextTraits<TDeviceType>::type;
```

The aggregating module `export import`s the traits map and imports the concrete class *plainly*.
An explicit specialization cannot carry `export`, so re-exporting its module publishes nothing - and
a plain class completes fine when it is merely reachable. The CUDA backend left the public API, and
all nineteen workarounds went with it.

## Lesson 3: `<execution>` in a global module fragment

For about a month, element-wise tensor math on the CPU was switched off because importing one partition
crashed MSVC with `C1116: unrecoverable error importing module`. The error named
`Compute.MemoryResource`, and that is where the blame sat.

It was wrong. The partition included `<execution>` in its global module fragment for a parallel
fast path, and MSVC's parallel algorithms pull in `<stop_token>` - the header actually in the
backtrace. Dropping `<execution>` for serial loops fixed it. The module the error named had nothing
to do with it.

Now no module in Mila includes `<execution>`. The broader habit is to distrust the module an import
error names until the backtrace agrees.

## Lesson 4: `import Mila;` still costs a consumer part of the standard library

This one is still open. It is now diagnosed and reported to Microsoft, and the fix is known.

A translation unit outside the library that does `import Mila;` - on MSVC 14.51 - loses parts of the
standard library:

- **Stream input fails.** `std::getline` and `cin.getline` both stop compiling with
  `'_Ok' uses undefined class 'std::basic_istream<char>::sentry'`. Output is fine.
- **Instantiating a model needs `<sstream>` included first.** `Component::toString()` is virtual,
  so each model's `toString()` compiles into the consumer through the vtable, and it uses
  `std::ostringstream`.
- **`import Mila;` must come after the `#include`s.** Import first and MSVC stops with `C1116`.

The C++ quick start carries all three workarounds, with comments saying they are workarounds. What
I know about the cause: 94 modules include `<sstream>` in their global module fragments, and adding
more includes on the library's side changes nothing - entities from a global module fragment are
reachable in the importer, not visible, and MSVC will not instantiate a class template whose
definition is only reachable.

A [standalone repro](https://github.com/ToddThomson/msvc-module-std-repro) with no Mila code in it
reproduces all three on MSVC 19.51 and none on Clang 21. The stream failure needs *two* modules that
each read from a stream in their interface; either one alone is fine. The fix is `import std;`, but
it has to be in the library: converting only the consumer clears the `<sstream>` failure and leaves
`std::getline` broken. Converting the library clears all three, including for consumers that still
`#include`. It is [reported to Microsoft](https://developercommunity.visualstudio.com/t/MSVC-modules-trigger-C2079-after-global-/11157342),
and converting Mila is tracked in [#30](https://github.com/ToddThomson/Mila/issues/30).

It hid for a long time because every consumer I had - the chat app, the samples, the tests - built
inside the library's own tree. It took a consumer outside the tree to find it.

## Lesson 5: Clang is the portability check

When Clang 21 first built Mila, it found five kinds of error that MSVC had accepted for seventeen
months:

1. **Missing transitive imports.** Sixteen operations used a type from `Dnn.Component` without
   importing it; MSVC resolved it through other imports.
2. **Template parameter shadowing.** A member template re-declared its class's `TPrecision`.
3. **Internal linkage inside `export namespace`.** Namespace-scope `const` and `constexpr` needed
   `inline`.
4. **Two-phase lookup.** Clang checks uninstantiated template members that MSVC never looked at,
   which exposed dead code nobody had called in months.
5. **A missing include.** `cudaStream_t` needs `<cuda_runtime_api.h>`, which MSVC had been getting
   by accident.

Every one was a real bug. None of them was Clang being difficult. Clang now builds everything Mila
ships for Linux - CI, the Python wheel and the container images - and when a missing import slips
through on MSVC, the Linux build names it.

## Lesson 6: the toolchain is the price of admission

- **GCC cannot build Mila yet.** In 2025, GCC 14.2 and GCC 15 both failed on the modules. Clang 19
  was easier to get, it worked, and I have not been back. GCC 16 is untried.
- **Early Visual Studio 2026 builds had a modules regression** that broke Mila; 18.6.2 fixed it.
- **Mila builds with Ninja.** Incremental module builds depend on it.
- **Incremental builds can lie.** Adding a virtual function to a root base class once left the
  build green and one unrelated FP4 test producing NaN. A clean rebuild fixed it, with no code
  change. I have not isolated the cause, but the rule I follow now is simple: a change to a root
  class's vtable gets a clean rebuild before anything else is debugged.

And the question every modules post gets: how long does it take to build? A clean Release build on
a 14-core Intel i5-13600K with 32 GB, using MSVC 19.51 and CUDA 13.4, takes **90 seconds for the
library** and **under four minutes for every target** - library, tests, chat app, tools and Python
bindings.

What surprised me is where the library's time goes. Its 315 C++ module objects take 490 CPU-seconds
between them. Its 67 CUDA files, compiled for five GPU architectures, take 1,013. The modules are the
smaller half of the build.

## Lesson 7: keep the preprocessor in the fragment

The global module fragment is the one place the preprocessor belongs. A `#ifdef` inside a module's
purview makes its interface depend on the preprocessor state it was compiled with, and it leaks
build configuration into every importer.

Mila does not fully meet that rule yet: 39 of its module files still have a conditional in the
purview, almost all of them the gate that removes CUDA from a CPU-only build. The direction is
clear, though - platform differences go into implementation units that CMake selects, and a
backend difference goes into a partition.

## Was it worth it?

Yes, with the costs stated plainly.

**What modules bought:**

- **Containment.** cuBLASLt, the CUDA runtime headers and their macros live in global module
  fragments and stop there. A consumer of `import Mila;` does not inherit them.
- **Structure you can navigate.** With one type per file and names that follow the path, the
  module graph is the architecture diagram.
- **Honest interfaces.** The export list is the API. What is not exported cannot be depended on,
  which is exactly what made Lesson 2 fixable.

**What they cost:**

- **The toolchain.** Visual Studio 2026 18.6.2+ or Clang 19+, CUDA 13, CMake 4 and Ninja. For many
  would-be contributors that list is where it ends.
- **No prebuilt distribution.** Module interfaces are compiler-specific, so a C++ consumer builds
  Mila from source through CMake `FetchContent`. (The Python wheel is the exception, because the
  modules stay inside it.)
- **Diagnostics.** Several of the lessons above started with an error that named the wrong thing.

Would I do it again? Yes. But I would add a second compiler on the first day, not the five
hundredth.

## The code

Everything above is in the repository:
[github.com/ToddThomson/Mila](https://github.com/ToddThomson/Mila). `Linear` (in
`Mila/Src/Dnn/Components/Linear/`) is the reference for the compile-time dispatch; the traits-map
pattern from Lesson 2 is in `Mila/Src/Dnn/Compute/ExecutionContext.ixx`; and the umbrella with its
notes on visibility is `Mila/Src/Mila.ixx`.

If you have hit any of these differently, or you know why Lesson 4 happens, I would like to hear it.
