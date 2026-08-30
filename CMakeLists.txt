# CMakeList.txt : Root CMake project for the Mila Project

cmake_minimum_required (VERSION 4.0)

# Single source of truth for the version. Version.txt feeds project(VERSION ...)
# (numeric triple) plus MILA_PRERELEASE_TAG for the label; see cmake/MilaVersion.cmake.
# CMAKE_CURRENT_SOURCE_DIR (not CMAKE_SOURCE_DIR) keeps this correct when Mila is
# add_subdirectory'd by a consumer.
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/MilaVersion.cmake")
mila_read_version("${CMAKE_CURRENT_SOURCE_DIR}/Version.txt" MILA_VERSION_NUMERIC MILA_PRERELEASE_TAG)
mila_pep440_version("${MILA_VERSION_NUMERIC}" "${MILA_PRERELEASE_TAG}" MILA_PEP440_VERSION)

# Version.txt is read at configure time, which creates no dependency on it by itself --
# so a version bump would otherwise leave the generated wheel version stale until
# something else forced a reconfigure. That is the exact failure this derivation exists
# to end, so make the bump itself trigger one.
set_property(DIRECTORY APPEND PROPERTY
    CMAKE_CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/Version.txt")

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

# Compile job pool -- bound peak build RAM by capping concurrent compile jobs.
#
# The heaviest module interface units (Compute/Operations/OperationTraits.ixx and the
# CUDA op .ixx units) each instantiate the full device x precision x quantization
# dispatch table and cost several GB of RAM to compile. When an IDE (VS 2026 or VSCode
# CMake Tools) drives Ninja at full core-count parallelism inside a memory-constrained
# container or WSL VM, several land at once, exhaust the VM's RAM, and the build appears
# to HANG at OperationTraits.ixx -- it is swap-thrashing, not stuck. A Ninja compile job
# pool caps concurrent compiles regardless of how the build is launched, unlike the
# mila-build-* scripts' `-j` which an IDE-driven build never sees. Default 0 = unlimited
# (no pool), so RAM-rich native builds are unaffected; the build container seeds a safe
# value via the MILA_COMPILE_JOB_POOL_SIZE env var (Docker/Dockerfile). Ninja-only;
# other generators ignore JOB_POOLS.
if(DEFINED ENV{MILA_COMPILE_JOB_POOL_SIZE})
    set(_mila_default_compile_pool $ENV{MILA_COMPILE_JOB_POOL_SIZE})
else()
    set(_mila_default_compile_pool 0)
endif()
set(MILA_COMPILE_JOB_POOL_SIZE "${_mila_default_compile_pool}" CACHE STRING
    "Max concurrent compile jobs (0 = unlimited); bounds peak build RAM on constrained VMs")
if(MILA_COMPILE_JOB_POOL_SIZE GREATER 0)
    set_property(GLOBAL PROPERTY JOB_POOLS mila_compile=${MILA_COMPILE_JOB_POOL_SIZE})
    set(CMAKE_JOB_POOL_COMPILE mila_compile)
    message(STATUS "Mila: compile job pool capped at ${MILA_COMPILE_JOB_POOL_SIZE} concurrent jobs")
endif()

# On for a top-level build (like SAMPLES/PROFILING/INSTALL), off when Mila is embedded
# via add_subdirectory/FetchContent/CPM -- so a consumer never builds Mila's tests or
# pulls googletest. The genuinely heavy gates (full install + subproject rebuild) are a
# separate opt-in: MILA_ENABLE_PACKAGING_TEST, off by default, enabled by x64-validate.
option(MILA_ENABLE_TESTING "Enable Mila Library tests." ${PROJECT_IS_TOP_LEVEL})
# Same rule, and it is load-bearing rather than tidiness: MilaPy is a MODULE library, so an
# embedded build links the vendored static libcurl into a shared object. Those objects are not
# built PIC, which ELF rejects outright -- the FetchContent gate fails at the MilaPy link while
# the consumer's own target has already linked cleanly. Windows has no equivalent constraint, so
# this reproduces only on Linux.
option(MILA_ENABLE_PYTHON_BINDINGS "Build the MilaPy pybind11 extension module." ${PROJECT_IS_TOP_LEVEL})
option(MILA_ENABLE_PROFILING "Enable Mila Library profiling." ${PROJECT_IS_TOP_LEVEL})
option(MILA_ENABLE_SAMPLES "Enable Mila Library samples." ${PROJECT_IS_TOP_LEVEL})
option(MILA_ENABLE_ADAPTORS "Build the first-class Mila adaptors (Chat CLI)." ${PROJECT_IS_TOP_LEVEL})
# Developer tooling: tokenize, the mila CLI, ExportArtifact. None of it ships -- the wheel
# excludes it and an embedding consumer just wants the library -- so this follows the same
# rule as the extras above rather than the bare PROJECT_IS_TOP_LEVEL it used to carry, which
# left no way to turn it off in a top-level build (the wheel presets configure one and got
# tokenize and ExportArtifact whether they wanted them or not).
option(MILA_ENABLE_TOOLS "Build Mila's developer tools (tokenize, mila CLI, ExportArtifact)." ${PROJECT_IS_TOP_LEVEL})
# The docs target's Doxyfile paths are Mila's own, so it is meaningless in a consumer's build
# and used to be offered there anyway. The docs CI job runs Doxygen directly against
# Mila/Docs/Doxyfile and never configures CMake, so nothing outside a top-level build needs it.
option(MILA_ENABLE_DOCS "Build Doxygen API documentation. Requires Doxygen and Graphviz." ${PROJECT_IS_TOP_LEVEL})
option(MILA_ENABLE_OPENMP "Enable OpenMP support." OFF)

# Mila's own HTTP transport. OFF drops libcurl and nothing else: the store, packaging,
# install, pull, and every HuggingFace URL, token rule and status meaning are still compiled,
# so such a build knows everything about the hub except how to reach it -- and a caller that
# supplies its own transport loses nothing at all.
#
# The reason is TLS, not the library: manylinux permits a wheel to vendor libcurl statically,
# but curl links the system OpenSSL, which is not whitelisted. Vendoring OpenSSL instead means
# owning its CVEs and shipping a CA bundle, losing the system trust store. So the Linux wheel
# builds with this OFF and supplies the transport from Python, which already has one.
# See Specifications/ModelDistribution.md.
option(MILA_ENABLE_LIBCURL "Build Mila's libcurl HTTP transport." ON)

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

if(MILA_ENABLE_LIBCURL)
    # Vendored and pinned on both platforms. Windows has no linkable OS libcurl, so it is
    # built there regardless; building it on Linux too buys one known version everywhere
    # rather than whatever the distribution shipped.
    #
    # TLS comes from the OS on both sides -- Schannel on Windows, system OpenSSL on Linux --
    # so there is no CA bundle to ship or keep current, and a corporate proxy with an
    # injected root just works.
    #
    # Protocols are cut to HTTP/HTTPS. No nghttp2, brotli or zstd either: HuggingFace serves
    # LFS blobs uncompressed over HTTP/1.1, so each would be a dependency that buys nothing.
    #
    # REVIEW: 8.21 is the latest GH release
    CPMAddPackage(
        NAME curl
        GITHUB_REPOSITORY curl/curl
        GIT_TAG curl-8_11_1
        OPTIONS
            "BUILD_SHARED_LIBS OFF"
            "BUILD_STATIC_LIBS ON"
            "BUILD_CURL_EXE OFF"
            "BUILD_TESTING OFF"
            "BUILD_EXAMPLES OFF"
            "CURL_DISABLE_INSTALL ON"
            "CURL_USE_SCHANNEL ${WIN32}"
            "CURL_USE_LIBSSH2 OFF"
            "CURL_USE_LIBPSL OFF"
            "CURL_ZLIB OFF"
            "CURL_BROTLI OFF"
            "CURL_ZSTD OFF"
            "USE_NGHTTP2 OFF"
            "USE_LIBIDN2 OFF"
            "ENABLE_UNIX_SOCKETS OFF"
            "CURL_DISABLE_LDAP ON"
            "CURL_DISABLE_LDAPS ON"
            "CURL_DISABLE_TELNET ON"
            "CURL_DISABLE_DICT ON"
            "CURL_DISABLE_GOPHER ON"
            "CURL_DISABLE_SMTP ON"
            "CURL_DISABLE_POP3 ON"
            "CURL_DISABLE_IMAP ON"
            "CURL_DISABLE_RTSP ON"
            "CURL_DISABLE_TFTP ON"
            "CURL_DISABLE_MQTT ON"
            "CURL_DISABLE_SMB ON"
            "CURL_DISABLE_FTP ON"
            "CURL_DISABLE_FILE ON"
    )
endif()

if(MILA_ENABLE_PYTHON_BINDINGS)
    CPMAddPackage(
        NAME pybind11
        GITHUB_REPOSITORY pybind/pybind11
        GIT_TAG v3.1.0
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
