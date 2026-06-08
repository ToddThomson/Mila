# CMakeList.txt : Root CMake project for the Mila Project

cmake_minimum_required (VERSION 4.0)

# Single source of truth for the version. Version.txt feeds project(VERSION ...)
# (numeric triple) plus MILA_PRERELEASE_TAG for the label; see cmake/MilaVersion.cmake.
# CMAKE_CURRENT_SOURCE_DIR (not CMAKE_SOURCE_DIR) keeps this correct when Mila is
# add_subdirectory'd by a consumer.
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/MilaVersion.cmake")
mila_read_version("${CMAKE_CURRENT_SOURCE_DIR}/Version.txt" MILA_VERSION_NUMERIC MILA_PRERELEASE_TAG)

# CUDA is enabled on demand via enable_language(CUDA) in the backend-detection
# block below, gated on MILA_ENABLE_CUDA + a found toolkit. Declaring it here would
# force the CUDA toolkit as a hard requirement and break a CPU-only configure.
project(MilaProject VERSION ${MILA_VERSION_NUMERIC} LANGUAGES CXX)

# PROJECT_IS_TOP_LEVEL (set by project() above) is FALSE when Mila is consumed
# via add_subdirectory / FetchContent. Developer-facing extras (samples,
# profiling) and install rules default ON only when Mila is the top project, so
# an embedding consumer gets just the library and pays for nothing else.

# Set global C++ and CUDA standard
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# On for a top-level build (like SAMPLES/PROFILING/INSTALL), off when Mila is embedded
# via add_subdirectory/FetchContent/CPM -- so a consumer never builds Mila's tests or
# pulls googletest. The genuinely heavy gates (full install + subproject rebuild) are a
# separate opt-in: MILA_ENABLE_PACKAGING_TEST, off by default, enabled by x64-validate.
option(MILA_ENABLE_TESTING "Enable Mila Library tests." ${PROJECT_IS_TOP_LEVEL})
option(MILA_ENABLE_PYTHON_BINDINGS "Build the MilaPy pybind11 extension module." OFF)
option(MILA_ENABLE_BENCHMARKS "Enable Mila Library benchmarks." OFF)
option(MILA_ENABLE_PROFILING "Enable Mila Library profiling." ${PROJECT_IS_TOP_LEVEL})
option(MILA_ENABLE_SAMPLES "Enable Mila Library samples." ${PROJECT_IS_TOP_LEVEL})
option(MILA_ENABLE_DOCS "Build Doxygen API documentation. Requires Doxygen and Graphviz." OFF)
option(MILA_ENABLE_OPENMP "Enable OpenMP support." OFF)
option(MILA_INSTALL "Generate Mila install/export rules (find_package consumption)." ${PROJECT_IS_TOP_LEVEL})
option(MILA_ENABLE_CUDA "Enable CUDA compute backend" ON)

# The Metal and ROCm backends are currently for internal testing only. Please leave them disabled.
option(MILA_ENABLE_METAL "Enable Metal compute backend (macOS/iOS only)" OFF)
option(MILA_ENABLE_ROCM "Enable ROCm compute backend" OFF)

# =============================================================================
# Compute backend detection
# =============================================================================
message(STATUS "Detecting Mila Compute Backends...")

# CPU backend (always available - no external dependencies)
set(MILA_HAS_CPU TRUE)
message(STATUS "CPU backend: enabled")

if(MILA_ENABLE_CUDA)
    find_package(CUDAToolkit QUIET)
    if(CUDAToolkit_FOUND)
        set(MILA_HAS_CUDA TRUE)
        enable_language(CUDA)
        message(STATUS "CUDA backend: enabled")
    else()
        set(MILA_HAS_CUDA FALSE)
        message(STATUS "CUDA backend: disabled (CUDA toolkit not found)")
    endif()
else()
    set(MILA_HAS_CUDA FALSE)
    message(STATUS "CUDA backend: disabled (explicitly disabled)")
endif()

if(MILA_ENABLE_METAL)
    if(NOT APPLE)
        message(WARNING "Metal backend requested but not on Apple platform - disabling")
        set(MILA_HAS_METAL FALSE)
    else()
        find_library(METAL_FRAMEWORK Metal)
        find_library(METALPERFORMANCESHADERS_FRAMEWORK MetalPerformanceShaders)
        find_library(FOUNDATION_FRAMEWORK Foundation)
        
        if(METAL_FRAMEWORK AND METALPERFORMANCESHADERS_FRAMEWORK AND FOUNDATION_FRAMEWORK)
            set(MILA_HAS_METAL TRUE)
            enable_language(OBJCXX)
            message(STATUS "Metal backend: enabled")
            message(STATUS " Note: Internal testing only (FUTURE)")
        else()
            set(MILA_HAS_METAL FALSE)
            message(STATUS "Metal backend: disabled (required frameworks not found)")
        endif()
    endif()
else()
    set(MILA_HAS_METAL FALSE)
    if(APPLE)
        message(STATUS "Metal backend: disabled (explicitly disabled)")
    else()
        message(STATUS "Metal backend: not available (not on Apple platform)")
    endif()
endif()

if(UNIX AND NOT APPLE)
    option(MILA_ENABLE_ROCM "Enable ROCm compute backend (stubbed)" OFF)
else()
    set(MILA_ENABLE_ROCM OFF)
endif()

if(MILA_ENABLE_ROCM)
    if(NOT (UNIX AND NOT APPLE))
        message(WARNING "ROCm backend requested but not on Linux platform - disabling")
        set(MILA_HAS_ROCM FALSE)
    else()
        find_package(hip QUIET)
        find_package(rocblas QUIET)
        
        if(hip_FOUND AND rocblas_FOUND)
            set(MILA_HAS_ROCM TRUE)
            enable_language(HIP)
            message(STATUS "ROCm backend: enabled (Internal testing only)")
        else()
            set(MILA_HAS_ROCM FALSE)
            message(STATUS "ROCm backend: disabled (ROCm not found)")
        endif()
    endif()
else()
    set(MILA_HAS_ROCM FALSE)
    if(UNIX AND NOT APPLE)
        message(STATUS "ROCm backend: disabled (explicitly disabled via MILA_ENABLE_ROCM=OFF)")
    else()
        message(STATUS "ROCm backend: not available (not on Linux platform)")
    endif()
endif()

# Summary of available backends
message(STATUS "Mila Compute Backend Summary:")
message(STATUS "===========================================")
message(STATUS "  CPU:     ${MILA_HAS_CPU}")
message(STATUS "  CUDA:    ${MILA_HAS_CUDA}")
message(STATUS "  Metal:   ${MILA_HAS_METAL}  (NOTE: Internal testing only)")
message(STATUS "  ROCm:    ${MILA_HAS_ROCM}  (NOTE: Internal testing only)")

# A Mila build with no compute backend cannot construct an execution context or run
# any operation -- it would be a non-functional artifact. CPU is the guaranteed
# baseline today (MILA_HAS_CPU is unconditionally TRUE), so this cannot fire yet; it
# is the load-bearing invariant for the day CPU becomes optional.
if(NOT (MILA_HAS_CPU OR MILA_HAS_CUDA OR MILA_HAS_METAL OR MILA_HAS_ROCM))
    message(FATAL_ERROR "Mila requires at least one compute backend, but all are disabled.")
endif()

# Use CPM to manage dependencies. If the embedding consumer already provides CPM
# (CPMAddPackage defined), reuse it; otherwise bootstrap into Mila's own tree
# (PROJECT_SOURCE_DIR, not CMAKE_SOURCE_DIR) so we never write into a consumer's
# source root.
if(NOT COMMAND CPMAddPackage)
    file(MAKE_DIRECTORY "${PROJECT_SOURCE_DIR}/cmake")
    set(CPM_PATH "${PROJECT_SOURCE_DIR}/cmake/CPM.cmake")
    if(NOT EXISTS "${CPM_PATH}")
        message(STATUS "Downloading CPM.cmake to ${CPM_PATH}")
        file(DOWNLOAD
            https://github.com/cpm-cmake/CPM.cmake/releases/latest/download/CPM.cmake
            "${CPM_PATH}"
            TLS_VERIFY ON
        )
    endif()
    include(${CPM_PATH})
endif()

CPMAddPackage(
    NAME miniz
    GITHUB_REPOSITORY richgel999/miniz
    GIT_TAG master
)

CPMAddPackage(
    NAME nlohmann_json
    VERSION 3.12.0
    GITHUB_REPOSITORY nlohmann/json
    OPTIONS
        "JSON_BuildTests OFF"
        "JSON_Install OFF"
        "JSON_MultipleHeaders OFF"
)

# TODO: Remove or Keep?
# SentencePiece - Llama-compatible BPE tokenizer
# Built without protobuf and shared libs to keep the dependency footprint minimal.
#CPMAddPackage(
#    NAME sentencepiece
#    GITHUB_REPOSITORY google/sentencepiece
#    GIT_TAG v0.2.1
#    OPTIONS
#        "SPM_ENABLE_SHARED OFF"
#        "SPM_ENABLE_TCMALLOC OFF"
#        "SPM_USE_BUILTIN_PROTOBUF ON"
#        "SPM_BUILD_TEST OFF"
#)

if(MILA_ENABLE_PYTHON_BINDINGS)
    CPMAddPackage(
        NAME pybind11
        GITHUB_REPOSITORY pybind/pybind11
        GIT_TAG v3.0.4
        OPTIONS
            "PYBIND11_INSTALL OFF"
            "PYBIND11_TEST OFF"
    )
endif()

if( MILA_HAS_CUDA )
    CPMAddPackage(
        NAME cutlass
        GITHUB_REPOSITORY NVIDIA/cutlass
        GIT_TAG v4.5.1
        OPTIONS
            "CUTLASS_ENABLE_TESTS OFF"
            "CUTLASS_ENABLE_EXAMPLES OFF"
            "CUTLASS_ENABLE_LIBRARY OFF"
            "CUTLASS_ENABLE_CUBLAS OFF"
    )
endif()

# Gated on MILA_HAS_CUDA: a CPU-only configure has no toolkit and must not require
# one. When CUDA is on, find_package re-establishes the CUDA::* imported targets the
# library and consumers link against.
if(MILA_HAS_CUDA)
    find_package(CUDAToolkit REQUIRED)

    # Default to building for the GPU on the current system. Guarded by NOT DEFINED
    # so an embedding consumer's architecture choice is respected.
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES native 86 CACHE STRING "CUDA architectures" )
    endif()
endif()

# miniz is provided by CPMAddPackage above (the `miniz` target plus miniz_SOURCE_DIR/
# miniz_BINARY_DIR); requiring find_package(miniz) added nothing the build uses and
# forced consumers/CI to system-install miniz just to satisfy it.
find_package(nlohmann_json REQUIRED)

# CTest is enabled once, here -- and only when Mila is the top-level project, since an
# embedding consumer owns its own testing setup. enable_testing() propagates to every
# subdirectory, so neither Mila/ nor Tests/ calls it again.
if(PROJECT_IS_TOP_LEVEL)
    enable_testing()
endif()

# The Mila library and Doxygen docs
add_subdirectory( Mila )
