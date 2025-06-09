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
    
    "Dnn/Tensors/Tensor.cpp"
    "Dnn/Tensors/TensorBuffer.cpp"
    "Dnn/Tensors/TensorHelpers.cpp"
     
    "Dnn/Compute/DeviceHelpers.cpp" 
    "Dnn/Compute/DeviceContext.cpp"

    #"Dnn/Model.cpp"

    "Dnn/Modules/Module.cpp"
    "Dnn/Modules/ModuleConfig.cpp"
    
    #"Dnn/Modules/Layers/Encoder.cpp"
    #"Dnn/Modules/Layers/EncoderConfig.cpp"
    #"Dnn/Modules/Layers/Gelu.cpp" 
    #"Dnn/Modules/Layers/GeluConfig.cpp"
    "Dnn/Modules/Layers/Linear.cpp"
    "Dnn/Modules/Layers/LinearConfig.cpp"
    #"Dnn/Modules/Attention.cpp" 
    #"Dnn/Modules/Layers/Layernorm.cpp"
    #"Dnn/Modules/Layers/Residual.cpp"
    #"Dnn/Modules/Layers/Softmax.cpp"
    #"Dnn/Modules/Blocks/MLP.cpp" 
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

target_compile_features( MilaTests PUBLIC cxx_std_23 )

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  target_compile_options(MilaTests PRIVATE 
    -fno-implicit-modules
    -fno-implicit-module-maps
    -fmodules
  )
endif()

target_link_libraries( MilaTests PRIVATE Mila gtest CUDA::cudart )

gtest_discover_tests(MilaTests)