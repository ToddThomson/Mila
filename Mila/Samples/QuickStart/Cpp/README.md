# Quick Start — C++

One prompt in, generated tokens streamed out — the smallest complete program that runs a local
LLM with Mila.

```bash
./build/mila_quickstart "Why is the sky blue?"
```

```
Mila 0.20.0-beta.3
Loading gemma-4-12b-it-fp4 ...

Sunlight contains every colour, but the short blue wavelengths scatter far more...

[stop]
```

This is a **standalone project**: it is not part of the Mila build tree, and it stands in for a
downstream app that pulls Mila in as a dependency. Copy it and it is your project — which makes
its `CMakeLists.txt` the worked example of how to depend on Mila, referenced from
`getting-started.md` §7.

**Single-shot by design.** No conversation history, no REPL — neither teaches anything about
Mila, and both are what make a chat harness large. [`Mila/Adaptors/Chat`](../../../Adaptors/Chat)
is where multi-turn, channel routing and tool calls live. The
[Python quick start](../Python/quickstart.py) does the same thing with the same model, the same
template and the same defaults, so the two read side by side.

## What it needs

A **CUDA GPU** and **`gemma-4-12b-it-fp4` installed** (~6.3 GB). Loading never downloads, so an
uninstalled name is an error rather than a surprise transfer — see
[the Quick Start index](../README.md#getting-a-model) for how to install one.

## The four things it does

Everything else in `main.cpp` is argument handling and error messages.

| | |
|---|---|
| **Locate** | `ModelStore::locate(name)` — the store is the only source; nothing consults a hub or takes a path. |
| **Load** | `GemmaModel<Cuda, BF16>::fromPretrained(...)` with `withFP4Quantization()`. Device and precision are template arguments, so the type *is* the configuration. |
| **Encode** | The Gemma instruct template, applied to one turn. Thinking is off, which takes two things — see the comment on `buildGemmaPrompt`. |
| **Generate** | `model->generate(tokens, on_token, params)` — the model owns the decode loop and pushes each token to your callback on your thread. |

The version prints before any of it, so a failure is separable from a failure to build and link
Mila at all.

## Why FetchContent (and not `find_package`)

A C++23 module library is a **source distribution**: module BMIs are not portable, so *any*
consumer recompiles Mila's module interfaces in its own toolchain. That voids `find_package`'s
prebuilt-binary benefit while adding an install-layout apparatus and an ABI split between the
prebuilt archive and the recompiled modules. FetchContent compiles Mila in your project's
toolchain — no install step, no ABI coupling — and is the same mechanism Mila uses for its own
dependencies (googletest, CUTLASS, nlohmann). `find_package(Mila)` was removed in 0.20.0-beta.3;
FetchContent is the one supported path.

**Budget for the first build.** The trade FetchContent makes is that your project builds *all*
of Mila — every CUDA kernel, not just the module interfaces — so expect the first configure and
build to take a while. Set `-DCMAKE_CUDA_ARCHITECTURES` to your own GPU's arch (`89` for Ada,
`120` for Blackwell); the default multi-arch list compiles every kernel several times over.
Rebuilds are incremental.

## Consuming Mila

```cmake
cmake_minimum_required(VERSION 4.0)
project(MyApp LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_SCAN_FOR_MODULES ON)   # your toolchain recompiles Mila's module units

include(FetchContent)
FetchContent_Declare(
    Mila
    GIT_REPOSITORY https://github.com/ToddThomson/Mila.git
    GIT_TAG        v0.20.0-beta.2    # pin to a published release tag
    # or, for a local working tree:  SOURCE_DIR /path/to/Mila
    # or, for a release archive:     URL https://github.com/ToddThomson/Mila/archive/refs/tags/v0.20.0-beta.2.zip
)
FetchContent_MakeAvailable(Mila)

add_executable(my_app main.cpp)
target_link_libraries(my_app PRIVATE Mila::Mila)

# Clang consumers only (MSVC auto-configures module consumption):
# target_compile_options(my_app PRIVATE -fno-implicit-modules -fno-implicit-module-maps)
```

Pin an immutable tag, not a branch. Mila is pre-1.0 and breaking changes are expected, so a
floating ref puts your build on Mila's release schedule instead of your own. The published tags
are on the [Releases page](https://github.com/ToddThomson/Mila/releases).

```cpp
import Mila;

int main()
{
    Mila::initialize();
    // ... use Mila ...
    Mila::shutdown();
}
```

The project and target names in this sample's own `CMakeLists.txt` are ours; in your project
call them whatever you like. Only the `FetchContent` block and the `Mila::Mila` link matter.

## Building and running it

```bash
cmake -S . -B build -G Ninja -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build
./build/mila_quickstart "Why is the sky blue?"
```

Run it with no arguments and it prompts on stdin, so it also works under a pipe:

```bash
echo "Explain KV caching in two sentences." | ./build/mila_quickstart
```

`FetchContent_MakeAvailable(Mila)` fetches Mila and `add_subdirectory()`'s it, so Mila builds
as a subproject in your toolchain. Because module BMIs are not portable, your toolchain
recompiles Mila's module units from source during this build — that is inherent to consuming a
module library, not specific to Mila.

## Automated gate

`Tests/Packaging`'s `packaging_fetchcontent_consumer` gate automates exactly this against the
local working tree (network-free) and fails CI if Mila stops being subproject-consumable. The
opt-in `packaging_cpm_consumer` additionally proves a published tag is fetchable over the network.
