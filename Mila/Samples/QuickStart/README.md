# Mila QuickStart

The minimal example of consuming Mila from your own CMake project via **FetchContent** —
the supported way to depend on Mila. It is referenced from `getting-started.md`.

Unlike the other entries under `Samples/`, this is a **standalone project**: it is not part
of the Mila build tree. It stands in for a downstream app that pulls Mila in as a dependency.

## Why FetchContent (and not `find_package`)

A C++23 module library is a **source distribution**: module BMIs are not portable, so *any*
consumer recompiles Mila's module units in its own toolchain. That voids `find_package`'s
prebuilt-binary benefit while adding an install-layout apparatus and an ABI split between the
prebuilt archive and the recompiled modules. FetchContent compiles Mila **once**, in your
project's toolchain — no install step, no ABI coupling — and is the same mechanism Mila uses
for its own dependencies (googletest, CUTLASS, nlohmann). `find_package` is parked (see the
repo's Packaging notes).

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
    GIT_TAG        v0.20.0           # pin to a published release tag
    # or, for a local working tree:  SOURCE_DIR /path/to/Mila
    # or, for a release archive:     URL https://github.com/ToddThomson/Mila/archive/refs/tags/v0.20.0.zip
)
FetchContent_MakeAvailable(Mila)

add_executable(my_app main.cpp)
target_link_libraries(my_app PRIVATE Mila::Mila)

# Clang consumers only (MSVC auto-configures module consumption):
# target_compile_options(my_app PRIVATE -fno-implicit-modules -fno-implicit-module-maps)
```

```cpp
import Mila;

int main()
{
    Mila::initialize();
    // ... use Mila ...
    Mila::shutdown();
}
```

## Building it

```bash
cmake -S . -B build -G Ninja
cmake --build build
```

`FetchContent_MakeAvailable(Mila)` fetches Mila and `add_subdirectory()`'s it, so Mila builds
as a subproject in your toolchain. Because module BMIs are not portable, your toolchain
recompiles Mila's module units from source during this build — that is inherent to consuming a
module library, not specific to Mila.

## Automated gate

`Tests/Packaging`'s `packaging_fetchcontent_consumer` gate automates exactly this against the
local working tree (network-free) and fails CI if Mila stops being subproject-consumable. The
opt-in `packaging_cpm_consumer` additionally proves a published tag is fetchable over the network.
