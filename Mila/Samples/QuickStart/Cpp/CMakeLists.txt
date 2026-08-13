# Mila C++ quick start -- one prompt in, generated tokens streamed out.
#
# This is a STANDALONE project (intentionally NOT add_subdirectory()'d by the Mila build):
# it stands in for a downstream app that pulls Mila in as a dependency, so this file is also
# the worked example of how to depend on Mila. FetchContent fetches Mila and
# add_subdirectory()'s it, compiling Mila once in THIS project's toolchain -- the only
# supported way to consume a C++23 module library. (Module BMIs are not portable, so a
# consumer recompiles Mila's module units regardless; FetchContent does it in one coherent
# toolchain, with no install step and no prebuilt/recompiled ABI split. find_package was
# removed in 0.20.0-beta.3 -- see Mila/Tests/Packaging/CMakeLists.txt for why.)
#
# The target and project names below are ours; in your own project call them whatever you
# like. Only the FetchContent block and the Mila::Mila link are load-bearing.
#
# Build it standalone:
#   cmake -S . -B build -G Ninja -DCMAKE_CUDA_ARCHITECTURES=89
#   cmake --build build
#   ./build/mila_quickstart "Why is the sky blue?"

cmake_minimum_required(VERSION 4.0)

project(MilaQuickStart LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Required so this project's toolchain scans main.cpp for `import Mila;` and recompiles
# Mila's C++23 module units (BMIs are not portable across toolchains).
set(CMAKE_CXX_SCAN_FOR_MODULES ON)

include(FetchContent)
FetchContent_Declare(
    Mila
    GIT_REPOSITORY https://github.com/ToddThomson/Mila.git
    GIT_TAG        v0.20.0-beta.2 # pin to a published release tag (see the GitHub Releases page)
    # Local alternative for developing against a working tree (no network):
    #   SOURCE_DIR /path/to/Mila
)
FetchContent_MakeAvailable(Mila)

add_executable(mila_quickstart main.cpp)
target_link_libraries(mila_quickstart PRIVATE Mila::Mila)

# Clang needs explicit module flags to consume the module units; MSVC configures this
# automatically.
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options(mila_quickstart PRIVATE -fno-implicit-modules -fno-implicit-module-maps)
endif()
