# CMakeList.txt : Root CMake project for the Mila Project

cmake_minimum_required (VERSION 3.31)  
 
project(MilaProject LANGUAGES CXX CUDA)  
 
# Set global C++ and CUDA standard 
set(CMAKE_CXX_STANDARD 23)   
set(CMAKE_CXX_STANDARD_REQUIRED ON)  
set(CMAKE_CXX_EXTENSIONS OFF)  
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
 
option(MILA_ENABLE_TESTING "Enable Mila Library tests." ON)
option(MILA_ENABLE_PERFTESTS "Enable performance tests." ON)
option(MILA_ENABLE_COVERAGE "Enable Mila Library code coverage." ON)
option(MILA_ENABLE_BENCHMARKS "Enable Mila Library benchmarks." ON)
option(MILA_ENABLE_GPT2_TESTING "Enable GPT2 tests." OFF)  
option(MILA_ENABLE_SAMPLES "Enable Mila Library samples." ON)
option(MILA_ENABLE_OPENMP "Enable OpenMP support." OFF)
option(MILA_ENABLE_COPY_MODEL_DATA "Copy data directory for samples and tests." ON)

option(MILA_ENABLE_CUDA "Enable CUDA compute backend" ON)
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

# Use CPM to manage dependencies
file(MAKE_DIRECTORY "${CMAKE_SOURCE_DIR}/cmake")
set(CPM_PATH "${CMAKE_SOURCE_DIR}/cmake/CPM.cmake")
if(NOT EXISTS "${CPM_PATH}")
    message(STATUS "Downloading CPM.cmake to ${CPM_PATH}")
    file(DOWNLOAD
        https://github.com/cpm-cmake/CPM.cmake/releases/latest/download/CPM.cmake
        "${CPM_PATH}"
        TLS_VERIFY ON
    )
endif()
include(${CPM_PATH})

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

find_package(CUDAToolkit REQUIRED)  
find_package(miniz REQUIRED)
find_package(nlohmann_json REQUIRED)

add_custom_target(MilaUpdateVersion
    COMMAND ${CMAKE_COMMAND} 
        -D CMAKE_SOURCE_DIR=${CMAKE_SOURCE_DIR}
        -P ${CMAKE_SOURCE_DIR}/cmake/IncrementVersion.cmake
    COMMENT "Incrementing build version number"
    VERBATIM
)

# Default to building for the GPU on the current system  
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)  
   set(CMAKE_CUDA_ARCHITECTURES native 86 CACHE STRING "CUDA architectures" )  
endif()

enable_testing()

# The Mila library and Doxygen docs
add_subdirectory( Mila )
