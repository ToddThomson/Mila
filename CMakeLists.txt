﻿# CMakeList.txt : Root CMake project for the Mila library  
#  
cmake_minimum_required (VERSION 3.31)  
 
project(MilaProject LANGUAGES CXX CUDA)  
 
# Set global C++ and CUDA standard 
set(CMAKE_CXX_STANDARD 23)   
set(CMAKE_CXX_STANDARD_REQUIRED ON)  
set(CMAKE_CXX_EXTENSIONS OFF)  
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
 
option(MILA_ENABLE_TESTING "Enable Mila Library tests." ON)  
option(MILA_ENABLE_GPT2_TESTING "Enable GPT2 tests." OFF )  
option(MILA_ENABLE_SAMPLES "Enable Mila Library samples." OFF )
option(MILA_ENABLE_NVBENCH "Enable NvBench." OFF )
option(MILA_ENABLE_OPENMP "Enable OpenMP support." ON )

# Include CPM.cmake
include(Cmake/CPM.cmake)

# Use CPM to download miniz
CPMAddPackage(
    NAME miniz
    GITHUB_REPOSITORY richgel999/miniz
    GIT_TAG 3.0.2
)

find_package(CUDAToolkit)  
find_package(miniz REQUIRED)  

if(MILA_ENABLE_OPENMP)
   find_package(OpenMP)
   if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
		add_compile_options( -openmp:llvm -DUSE_OMP )
		add_link_options( -openmp:llvm )
	else()
		add_compile_options( ${OpenMP_CXX_FLAGS})
		add_link_options( OpenMP::OpendMP_CXX)
	endif()
endif()  
 
# Default to building for the GPU on the current system  
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)  
   set(CMAKE_CUDA_ARCHITECTURES native)  
endif()  

# The Mila library and Doxygen docs
add_subdirectory( Mila )

if (MILA_ENABLE_NVBENCH )
#add_subdirectory( Benches )
endif()
