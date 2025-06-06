﻿# CMakeList.txt : Root CMake project for the Mila Project

cmake_minimum_required (VERSION 3.31)  
 
project(MilaProject LANGUAGES CXX CUDA)  
 
# Set global C++ and CUDA standard 
set(CMAKE_CXX_STANDARD 23)   
set(CMAKE_CXX_STANDARD_REQUIRED ON)  
set(CMAKE_CXX_EXTENSIONS OFF)  
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
 
option(MILA_ENABLE_TESTING "Enable Mila Library tests." ON)
option(MILA_ENABLE_BENCHMARKS "Enable Mila Library benchmarks." ON)
option(MILA_ENABLE_GPT2_TESTING "Enable GPT2 tests." OFF)  
option(MILA_ENABLE_SAMPLES "Enable Mila Library samples." ON)
option(MILA_ENABLE_OPENMP "Enable OpenMP support." OFF)
option(MILA_ENABLE_COPY_MODEL_DATA "Copy data directory for samples and tests." ON)

# Use CPM to manage dependencies
include(Cmake/CPM.cmake)

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
