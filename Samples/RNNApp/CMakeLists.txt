# CMakeList.txt : MilaApp source project
#
cmake_minimum_required (VERSION 3.20)

# Add Cuda and CuDNN include directories
include_directories( cuda_helpers ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} )

# Add source to this project's executable.
add_executable( SampleRnnApp 
    "SampleRnnApp.cpp" 
    "SampleRnnApp.h"  
    "SampleRnnModel.h"  )

target_link_libraries( SampleRnnApp MilaLib ) #${CUDNN_LIBRARY} )

# add the binary tree to the search path for include files
# so that we will find Mila.h
target_include_directories(
    MilaLib PUBLIC
    "${PROJECT_BINARY_DIR}" )
