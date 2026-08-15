# Your project, inside the Mila development image.
#
# The only difference from Mila/Samples/QuickStart/Cpp/CMakeLists.txt is where Mila comes
# from. That one fetches a published tag over the network, which is what you would do on
# your own machine. Here the release is already on disk at /src -- the same tree this image
# was built from -- so it is added directly. Nothing is hidden: change the path below and
# you change what this links against.
#
# Everything else is what any consumer of Mila needs, and is copied verbatim from the
# sample. Only the add_subdirectory line and the Mila::Mila link are load-bearing.
#
# Build it:
#   cmake --build build
#   ./build/myapp "Why is the sky blue?"
#
# The first build compiles Mila itself and takes roughly fifteen minutes: C++23 module BMIs
# are not portable between toolchains, so every consumer recompiles Mila's module units and
# the copy already built at /opt/mila/build cannot be linked here. Builds after that one are
# seconds, because only main.cpp changed.

cmake_minimum_required(VERSION 4.0)

project(MyApp LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Required so this project's toolchain scans main.cpp for `import Mila;` and recompiles
# Mila's C++23 module units (BMIs are not portable across toolchains).
set(CMAKE_CXX_SCAN_FOR_MODULES ON)

# The Mila release this image ships. On your own machine this would be a FetchContent block
# pinning a tag -- see the sample named above for that form.
add_subdirectory(/src mila)

add_executable(myapp main.cpp)
target_link_libraries(myapp PRIVATE Mila::Mila)

# Clang needs explicit module flags to consume the module units; MSVC configures this
# automatically.
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options(myapp PRIVATE -fno-implicit-modules -fno-implicit-module-maps)
endif()
