# CMakeList.txt : Root CMake project for the Mila library  
#  
cmake_minimum_required (VERSION 3.31)  
 
project(MilaProject LANGUAGES CXX CUDA)  
 
# Set global C++ and CUDA standard 
set(CMAKE_CXX_STANDARD 23)  
set(CMAKE_CXX_STANDARD_REQUIRED ON)  
set(CMAKE_CXX_EXTENSIONS OFF)  
set(CMAKE_CUDA_STANDARD 20)  
 
option(MILA_ENABLE_TESTING "Enable Mila Library tests." "ON")  
option(MILA_ENABLE_GPT2_TESTING "Enable GPT2 tests." "OFF")  
option(MILA_ENABLE_SAMPLES "Enable Mila Library samples." "ON")
option(MILA_ENABLE_NVBENCH "Enable NvBench." "OFF")
option(MILA_ENABLE_OPENMP "Enable OpenMP support." "OFF")  
 
find_package(CUDAToolkit)  
find_package(Thrust REQUIRED)  

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

# The Mila library  
add_subdirectory( Mila )  
 
enable_testing()  
 
# Add MilaTest directory  
if (MILA_ENABLE_TESTING)  
add_subdirectory(Mila/Tests)  
endif()  
 
if (MILA_ENABLE_SAMPLES)  
add_subdirectory(Mila/Samples)  
endif()

if (MILA_ENABLE_NVBENCH )
add_subdirectory( Benches )
endif()
