# CMake project for testing the Mila Library

enable_testing()

include(FetchContent)
   FetchContent_Declare(
     googletest
     URL https://github.com/google/googletest/archive/refs/tags/v1.17.0.zip
)

# For Windows: Prevent overriding the parent project's compiler/linker settings
set( gtest_force_shared_crt ON CACHE BOOL "" FORCE )

FetchContent_MakeAvailable( googletest )
include( GoogleTest )

# Set global C++ and CUDA standard 
set(CMAKE_CXX_STANDARD 23)  
set(CMAKE_CXX_STANDARD_REQUIRED ON)  
set(CMAKE_CXX_EXTENSIONS OFF)  
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)

# Default to building for the GPU on the current system
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()


add_executable( MilaTests
    "main.cpp"
    "Dnn/Core/Tensor.cpp"
    "Dnn/Core/TensorBuffer.cpp"
    "Dnn/Core/TensorHelpers.cpp"
     
    "Dnn/Compute/DeviceHelpers.cpp" 
    "Dnn/Compute/DeviceContext.cpp"

    #"Dnn/Model.cpp"
    
    "Dnn/Modules/Linear.cpp"
    "Dnn/Modules/Gelu.cpp" 
    #"Dnn/Modules/Attention.cpp" 
    #"Dnn/Modules/Layernorm.cpp"
    #"Dnn/Modules/Residual.cpp"
    #"Dnn/Modules/Softmax.cpp"
    #"Dnn/Modules/Encoder.cpp"

    "Dnn/Modules/Blocks/MLP.cpp" 
    #"Dnn/Modules/Blocks/TransformerBlock.cpp"

    #"Dnn/Data/DataLoader.cpp"
    #"Dnn/Models/Gpt2/DatasetReader.cpp"

    # Cpu Operations...    
    #"Dnn/Compute/Operations/Cpu/CpuGeluOpTests.cpp"
    #"Dnn/Compute/Operations/Cpu/CpuFullyConnectedOpTests.cpp"
    #"Dnn/Compute/Operations/Cpu/CpuEncoderOpTests.cpp"
    #"Dnn/Compute/Operations/Cpu/CpuLayerNormTests.cpp"
    #"Dnn/Compute/Operations/Cpu/CpuResidualOpTests.cpp"
    #"Dnn/Compute/Operations/Cpu/CpuSoftmaxOpTests.cpp"
    # Cuda Operations...
    #"Dnn/Compute/Operations/Cuda/CudaMatMulBiasOpTests.cpp"
    #"Dnn/Compute/Operations/Cuda/CudaEncoderOpTests.cpp"
    #"Dnn/Compute/Operations/Cuda/CudaGeluOpTests.cpp"
    #"Dnn/Compute/Operations/Cuda/CudaLayerNormOpTests.cpp"
    #"Dnn/Compute/Operations/Cuda/CudaMultHeadAttentionOpTests.cpp"
    #"Dnn/Compute/Operations/Cuda/CudaResidualOpTests.cpp"
    #"Dnn/Compute/Operations/Cuda/CudaSoftmaxOpTests.cpp"
     
    
)

# TODO: Add globally 
target_compile_features( MilaTests PUBLIC cxx_std_23 )

if(MILA_ENABLE_OPENMP)  
	if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
		target_compile_options( MilaTests PRIVATE -openmp:llvm -DUSE_OMP )
		target_link_options( MilaTests PRIVATE -openmp:llvm )
	else()
		target_compile_options( MilaTests PRIVATE ${OpenMP_CXX_FLAGS})
		target_link_options( MilaTests PRIVATE OpenMP::OpendMP_CXX)
	endif()
endif()  

target_link_libraries( MilaTests PRIVATE Mila gtest CUDA::cudart )

add_custom_command(
    TARGET MilaTests POST_BUILD
	COMMAND ${CMAKE_COMMAND} -E copy_directory_if_different 
        "${CMAKE_SOURCE_DIR}/data"
        "${CMAKE_CURRENT_BINARY_DIR}/data"
    COMMENT "Copying data directory to build output directory"
)

gtest_discover_tests(MilaTests)