# CMakeList.txt : Top-level CMake project file
#
cmake_minimum_required (VERSION 3.20)
message(STATUS "CMake version: ${CMAKE_VERSION}")

project(
    DeviceQuery VERSION 1.0
    DESCRIPTION "CUDA Device Query"
    LANGUAGES CXX CUDA)

find_package( CUDA REQUIRED )

message( STATUS "Found lib CXX: ${CMAKE_CXX_STANDARD_LIBRARIES}" )
message( STATUS "Found CXX compiler: ${CMAKE_CXX_COMPILER}" )

# Specify binary name & source file to build it from
add_executable(
    ${PROJECT_NAME}
    DeviceQuery.cpp )

# Add CUDA Toolkit include directories
include_directories( cuda_helpers ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} )

#target_compile_features( DeviceQuery PUBLIC cxx_std_17 )

# Specify target & include directories
target_include_directories(
    ${PROJECT_NAME}
    PRIVATE
    "C:/ProgramData/NVIDIA Corporation/CUDA Samples/v11.4/common/inc" )

# Linking ...
target_link_libraries(
    ${PROJECT_NAME}
    PRIVATE
    ${CUDA_LIBRARIES} )