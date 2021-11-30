# CMakeList.txt : Mila Testing subproject
#
cmake_minimum_required (VERSION 3.20)

enable_testing()

include( FetchContent )
FetchContent_Declare(
    googletest 
    URL https://github.com/google/googletest/archive/refs/tags/release-1.11.0.zip
)

# For Windows: Prevent overriding the parent project's compiler/linker settings
set( gtest_force_shared_crt ON CACHE BOOL "" FORCE )

FetchContent_MakeAvailable( googletest )
include( GoogleTest )

add_executable( 
    MilaTest
    "main.cpp"
    "Cuda/Device.cpp"
    "dnn/Context.cpp"
    "dnn/Utils.cpp"
    "dnn/RNN/RnnDataSet.cpp"
    "dnn/RNN/RnnOp.cpp"
    "Dnn/RNN/RnnModel.cpp"
    "dnn/Tensor.cpp"
    "dnn/Common.h"
    "dnn/Dropout.cpp"  
     )
    
# Cuda and CuDNN includes
target_include_directories(
    MilaTest
    PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} )

target_link_libraries( MilaTest MilaLib gtest )
    
gtest_discover_tests( MilaTest )