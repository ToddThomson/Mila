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
option(MILA_ENABLE_COVERAGE "Enable Mila Library code coverage." ON)
option(MILA_ENABLE_BENCHMARKS "Enable Mila Library benchmarks." OFF)
option(MILA_ENABLE_GPT2_TESTING "Enable GPT2 tests." OFF)  
option(MILA_ENABLE_SAMPLES "Enable Mila Library samples." OFF)
option(MILA_ENABLE_OPENMP "Enable OpenMP support." OFF)
option(MILA_ENABLE_COPY_MODEL_DATA "Copy data directory for samples and tests." ON)

# Optional compute backend detection (FUTURE)
option(MILA_ENABLE_METAL "Enable Metal compute backend (macOS/iOS only)" OFF)
option(MILA_ENABLE_OPENCL "Enable OpenCL compute backend" OFF)
option(MILA_ENABLE_VULKAN "Enable Vulkan compute backend" OFF)

# Metal detection (Apple platforms only)
if(MILA_ENABLE_METAL AND APPLE)
    find_library(METAL_FRAMEWORK Metal REQUIRED)
    find_library(METALPERFORMANCESHADERS_FRAMEWORK MetalPerformanceShaders REQUIRED)
    if(METAL_FRAMEWORK AND METALPERFORMANCESHADERS_FRAMEWORK)
        set(METAL_FOUND TRUE)
        message(STATUS "Metal support enabled")
    endif()
endif()

# OpenCL detection
if(MILA_ENABLE_OPENCL)
    find_package(OpenCL QUIET)
    if(OpenCL_FOUND)
        message(STATUS "OpenCL support enabled - Version: ${OpenCL_VERSION_STRING}")
        message(STATUS "OpenCL libraries: ${OpenCL_LIBRARIES}")
        message(STATUS "OpenCL include dirs: ${OpenCL_INCLUDE_DIRS}")
    else()
        message(STATUS "OpenCL not found - OpenCL backend disabled")
    endif()
endif()

# Vulkan detection
if(MILA_ENABLE_VULKAN)
    find_package(Vulkan QUIET)
    if(Vulkan_FOUND)
        message(STATUS "Vulkan support enabled - Version: ${Vulkan_VERSION}")
        message(STATUS "Vulkan libraries: ${Vulkan_LIBRARIES}")
        message(STATUS "Vulkan include dirs: ${Vulkan_INCLUDE_DIRS}")
    else()
        message(STATUS "Vulkan not found - Vulkan backend disabled")
    endif()
endif()

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
        "JSON_MultipleHeaders ON"
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
   set(CMAKE_CUDA_ARCHITECTURES native)  
endif()

enable_testing()

# The Mila library and Doxygen docs
add_subdirectory( Mila )
