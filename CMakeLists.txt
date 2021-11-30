# CMakeList.txt : Mila top-level CMake project file
#
cmake_minimum_required( VERSION 3.20 )

project(
    Mila VERSION 0.9.0
    DESCRIPTION "Mila Deep Neural Net library"
    LANGUAGES CXX CUDA )

set( CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake; ${CMAKE_MODULE_PATH}" )

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True )

find_package( CUDA 11.5 REQUIRED )
find_package( CuDNN REQUIRED )

set( MILA_ROOT ${PROJECT_SOURCE_DIR} )
set( CUDA_VERSION "${CMAKE_CUDA_COMPILER_VERSION}" )

option( BUILD_TESTS "Build unit tests" ON )
option( BUILD_SAMPLES "Build samples" ON )

# Include sub-projects.
add_subdirectory( "Source")

if ( BUILD_SAMPLES )
    add_subdirectory( "Samples" )
endif()

if ( BUILD_TESTS )
    enable_testing()
    add_subdirectory( "Tests" )
endif()
