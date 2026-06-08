# FetchContent / add_subdirectory consumer fixture for the packaging gate.
#
# Unlike Samples/QuickStart (which consumes an INSTALLED Mila via find_package),
# this project embeds Mila as a subproject: FetchContent_MakeAvailable runs
# add_subdirectory on the Mila source tree and links the in-tree Mila::Mila
# target. It exists to prove Mila is subproject-friendly -- that nothing in
# Mila's CMake assumes it is the top-level project (CMAKE_SOURCE_DIR vs
# PROJECT_SOURCE_DIR, PROJECT_IS_TOP_LEVEL gating, install-rule leakage).
#
# The Mila source tree is provided locally via -DMILA_SOURCE_DIR so the gate
# validates the CURRENT working tree rather than a published tag (network-free).

cmake_minimum_required(VERSION 4.0)

project(MilaFetchContentConsumer LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Required for the consumer to scan main.cpp for `import Mila;` and to compile
# Mila's module units (built in-tree as a subproject).
set(CMAKE_CXX_SCAN_FOR_MODULES ON)

if(NOT DEFINED MILA_SOURCE_DIR)
    message(FATAL_ERROR "MILA_SOURCE_DIR must point at the Mila source tree.")
endif()

include(FetchContent)
FetchContent_Declare(Mila SOURCE_DIR "${MILA_SOURCE_DIR}")
FetchContent_MakeAvailable(Mila)

add_executable(mila_fetchcontent_consumer main.cpp)
target_link_libraries(mila_fetchcontent_consumer PRIVATE Mila::Mila)

# Clang needs explicit module flags to consume the module units; MSVC configures
# this automatically.
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options(mila_fetchcontent_consumer PRIVATE -fno-implicit-modules -fno-implicit-module-maps)
endif()
