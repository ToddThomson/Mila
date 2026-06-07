# Mila QuickStart

The minimal example of consuming **an installed Mila** from your own CMake project
via `find_package(Mila)`. It is referenced from `getting-started.md`.

Unlike the other entries under `Samples/`, this is a **standalone project**: it is
not part of the Mila build tree. It is configured on its own against an *install
prefix*, which is the only way to exercise `find_package(Mila)` against the real
installed package (the in-tree samples link the `Mila` target directly and so never
touch the install layout).

## Consuming Mila

```cmake
cmake_minimum_required(VERSION 4.0)
project(MyApp LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_SCAN_FOR_MODULES ON)   # consumer rebuilds Mila's module units

find_package(Mila CONFIG REQUIRED)

add_executable(my_app main.cpp)
target_link_libraries(my_app PRIVATE Mila::Mila)
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

First install Mila somewhere (a throwaway prefix is fine):

```bash
cmake --install <your-mila-build-dir> --prefix /tmp/mila-install
```

Then configure and build this project against that prefix:

```bash
cmake -S . -B build -G Ninja -DCMAKE_PREFIX_PATH=/tmp/mila-install
cmake --build build
```

C++23 module BMIs are not portable, so the consumer toolchain recompiles Mila's
installed module units from source during this build. That recompile is the point:
it resolves the module units' file-relative kernel-header includes against the
install tree, which is what the packaging layout has to get right.

## Automated gate

`Tests/Packaging` automates install + configure + build of this project and fails
CI if the installed package is not consumable. Enable it with
`-DMILA_ENABLE_PACKAGING_TEST=ON` (alongside `-DMILA_ENABLE_TESTING=ON`).
