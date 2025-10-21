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

    # --------------------------------------------------------------
    # Dnn::Compute::Devices Tests
    # --------------------------------------------------------------
    "Dnn/Compute/Devices/DeviceHelpers.cpp"
    "Dnn/Compute/Devices/DeviceTypes.cpp"
    "Dnn/Compute/Devices/Cpu/CpuDevicePlugin.cpp"
    "Dnn/Compute/Devices/Cuda/CudaDevicePlugin.cpp"

    "Dnn/Compute/Devices/Registry/DeviceRegistry.cpp"

    # --------------------------------------------------------------
    # Dnn::Compute::Execution Context Tests
    # --------------------------------------------------------------
    "Dnn/Compute/ExecutionContext/ExecutionContext.cpp"
    "Dnn/Compute/ExecutionContext/CudaExecutionContext.cpp"
    "Dnn/Compute/ExecutionContext/CpuExecutionContext.cpp"

    # --------------------------------------------------------------
    # Tensor Tests
    # --------------------------------------------------------------
    "Dnn/Tensors/TensorBuffer.cpp"
    "Dnn/Tensors/Tensor.Constructors.cpp"
    "Dnn/Tensors/Tensor.MemoryProperties.cpp"
    "Dnn/Tensors/Tensor.Properties.cpp"
    "Dnn/Tensors/Tensor.DataAccess.cpp"
    "Dnn/Tensors/Tensor.Identity.cpp"
    "Dnn/Tensors/Tensor.Io.cpp"
    "Dnn/Tensors/Tensor.DataPointers.cpp"

    # TensorOps
    "Dnn/Tensors/TensorOps/Fill.Cpu.cpp"
    "Dnn/Tensors/TensorOps/Fill.Cuda.cpp"
    "Dnn/Tensors/TensorOps/Math.Cpu.cpp"
    "Dnn/Tensors/TensorOps/Math.Cuda.cpp"
    "Dnn/Tensors/TensorOps/TensorOps.Transfer.cpp"

    # Tensor Helpers
    "Dnn/Tensors/Tensor.Initializers.cpp"

    # --------------------------------------------------------------
    # Compute::Device::Cpu Operations Tests
    # --------------------------------------------------------------
    #"Dnn/Compute/Devices/Cpu/Operations/CpuEncoderOpTests.cpp"
    #"Dnn/Compute/Devices/Cpu/Operations/CpuLinearOpTests.cpp"
    #"Dnn/Compute/Devices/Cpu/Operations/CpuGeluOpTests.cpp"
    #"Dnn/Compute/Devices/Cpu/Operations/CpuLayerNormTests.cpp"
    #"Dnn/Compute/Devices/Cpu/Operations/CpuResidualOpTests.cpp"
    #"Dnn/Compute/Devices/Cpu/Operations/CpuSoftmaxOpTests.cpp"
    
    # --------------------------------------------------------------
    # Compute::Device::Cuda Operations Tests
    # --------------------------------------------------------------
    #"Dnn/Compute/Operations/Cuda/CudaMatMulBiasOpTests.cpp"
    #"Dnn/Compute/Operations/Cuda/CudaEncoderOpTests.cpp"
    #"Dnn/Compute/Operations/Cuda/CudaGeluOpTests.cpp"
    #"Dnn/Compute/Operations/Cuda/CudaLayerNormOpTests.cpp"
    #"Dnn/Compute/Operations/Cuda/CudaMultHeadAttentionOpTests.cpp"
    #"Dnn/Compute/Operations/Cuda/CudaResidualOpTests.cpp"
    #"Dnn/Compute/Operations/Cuda/CudaSoftmaxOpTests.cpp"

    # --------------------------------------------------------------
    # Dnn::Models Tests
    # --------------------------------------------------------------
    #"Dnn/Model.cpp"

    # --------------------------------------------------------------
    # Dnn::Modules Tests
    # --------------------------------------------------------------

    "Dnn/Modules/Module.cpp"
    "Dnn/Modules/ModuleConfig.cpp"
    
    #"Dnn/Modules/Layers/Encoder.cpp"
    #"Dnn/Modules/Layers/EncoderConfig.cpp"
    #"Dnn/Modules/Attention.cpp"
    "Dnn/Modules/Activations/Gelu.Cpu.cpp" 
    "Dnn/Modules/Activations/GeluConfig.cpp"
    #"Dnn/Modules/Layers/Linear.cpp"
    #"Dnn/Modules/Layers/LinearConfig.cpp"
    #"Dnn/Modules/Normalization/LayerNorm.cpp"
    #"Dnn/Modules/Normalization/LayerNormConfig.cpp"
    #"Dnn/Modules/Layers/Residual.cpp"
    #"Dnn/Modules/Layers/Softmax.cpp"
    
    # --------------------------------------------------------------
    # Modules::Blocks Tests
    # --------------------------------------------------------------
    #"Dnn/Modules/Blocks/MLP.cpp" 
    #"Dnn/Modules/Blocks/TransformerBlock.cpp"

    #"Dnn/Data/DataLoader.cpp"
    #"Dnn/Models/Gpt2/DatasetReader.cpp"

    # Cuda Operations...
    #"Dnn/Compute/Operations/Cuda/CudaMatMulBiasOpTests.cpp"
    #"Dnn/Compute/Operations/Cuda/CudaEncoderOpTests.cpp"
    #"Dnn/Compute/Operations/Cuda/CudaGeluOpTests.cpp"
    #"Dnn/Compute/Operations/Cuda/CudaLayerNormOpTests.cpp"
    #"Dnn/Compute/Operations/Cuda/CudaMultHeadAttentionOpTests.cpp"
    #"Dnn/Compute/Operations/Cuda/CudaResidualOpTests.cpp"
    #"Dnn/Compute/Operations/Cuda/CudaSoftmaxOpTests.cpp"
    
    #"Dnn/Modules/Blocks/MLPConfig.cpp" 
    #"Dnn/Seriaalization/ZipSerializer.cpp"
    #"Dnn/Tensors/TensorBuffer.Tracking.cpp"

    
    
    "Dnn/Compute/Operations/OperationBase.cpp"
    "Dnn/Compute/Operations/UnaryOperation.cpp"
    #"Dnn/Compute/Operations/BinaryOperation.cpp"
    "Dnn/Modules/Activations/Gelu.Cuda.cpp")

target_compile_features( MilaTests PUBLIC cxx_std_23 )

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  target_compile_options(MilaTests PRIVATE 
    -fno-implicit-modules
    -fno-implicit-module-maps
    -fmodules
  )
endif()

target_link_libraries( MilaTests PUBLIC Mila gtest CUDA::cudart )

gtest_discover_tests(MilaTests)