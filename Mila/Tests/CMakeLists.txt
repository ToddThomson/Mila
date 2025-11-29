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
    "Dnn/Compute/Devices/Cpu/Operations/CpuGeluOpTests.cpp"
    #"Dnn/Compute/Devices/Cpu/Operations/CpuLinearOpTests.cpp"
    #"Dnn/Compute/Devices/Cpu/Operations/CpuLayerNormOpTests.cpp"
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
    # Dnn::Components Tests
    # --------------------------------------------------------------
    "Dnn/Components/Component.cpp"
    "Dnn/Components/ComponentConfig.cpp"

    "Dnn/Components/Layers/Attention.Cpu.cpp"
    
    "Dnn/Components/Layers/Encoder.Cpu.cpp"
    "Dnn/Components/Layers/Encoder.Cuda.cpp"
    "Dnn/Components/Layers/EncoderConfig.cpp"
    
    "Dnn/Components/Connections/Residual.Cpu.cpp"
    "Dnn/Components/Connections/Residual.Cuda.cpp"
    "Dnn/Components/Connections/Residual.Config.cpp"

    "Dnn/Components/Activations/Gelu.Cpu.cpp"
    "Dnn/Components/Activations/Gelu.Cuda.cpp"
    "Dnn/Components/Activations/GeluConfig.cpp"

    "Dnn/Components/Normalization/LayerNorm.Cpu.cpp"
    "Dnn/Components/Normalization/LayerNorm.Cuda.cpp"
    "Dnn/Components/Normalization/LayerNormConfig.cpp"

    "Dnn/Components/Layers/Linear.Cpu.cpp"
    "Dnn/Components/Layers/Linear.Cuda.cpp"
    "Dnn/Components/Layers/LinearConfig.cpp"

    "Dnn/Components/Connections/Residual.Config.cpp"
    "Dnn/Components/Connections/Residual.Cuda.cpp"
    "Dnn/Components/Connections/Residual.Cpu.cpp"

    "Dnn/Components/Layers/Softmax.Cpu.cpp"
    "Dnn/Components/Layers/Softmax.Cuda.cpp"    
    "Dnn/Components/Layers/SoftmaxConfig.cpp"

    # --------------------------------------------------------------
    # Modules::Blocks Tests
    # --------------------------------------------------------------
    "Dnn/Components/Blocks/MLP.Cpu.cpp"
    "Dnn/Components/Blocks/MLP.Cuda.cpp"
    "Dnn/Components/Blocks/MLPConfig.cpp"

    "Dnn/Components/Blocks/Transformer.Cpu.cpp"
    
    # --------------------------------------------------------------
    # Data Tests
    # --------------------------------------------------------------
    "Dnn/Data/DataLoader.cpp"
    
    # Cuda Operations...
    #"Dnn/Compute/Operations/Cuda/CudaMatMulBiasOpTests.cpp"
    #"Dnn/Compute/Operations/Cuda/CudaEncoderOpTests.cpp"
    #"Dnn/Compute/Operations/Cuda/CudaGeluOpTests.cpp"
    #"Dnn/Compute/Operations/Cuda/CudaLayerNormOpTests.cpp"
    #"Dnn/Compute/Operations/Cuda/CudaMultHeadAttentionOpTests.cpp"
    #"Dnn/Compute/Operations/Cuda/CudaResidualOpTests.cpp"
    #"Dnn/Compute/Operations/Cuda/CudaSoftmaxOpTests.cpp"
    
    # --------------------------------------------------------------
    # Serialization Tests
    # --------------------------------------------------------------
    "Dnn/Serialization/ZipSerializer.cpp"

    # --------------------------------------------------------------
    # Modeling Tests
    # --------------------------------------------------------------
    #"Dnn/Modeling/Model.Cpu.cpp"

    # --------------------------------------------------------------
    #"Dnn/Tensors/TensorBuffer.Tracking.cpp"
    
    "Dnn/Compute/Operations/OperationBase.cpp"
    "Dnn/Compute/Operations/UnaryOperation.cpp"
    #"Dnn/Compute/Operations/BinaryOperation.cpp"

    # TJT: These are op tests. We should be testing Components instead.
    #"Dnn/Optimizers/AdamW.Cuda.cpp"
    #"Dnn/Optimizers/AdamW.Cpu.cpp"
    
    "Dnn/Components/Losses/CrossEntropy.Cpu.cpp"
    "Dnn/Components/Losses/SoftmaxCrossEntropy.Cuda.cpp"
    "Dnn/Components/Losses/CrossEntropy.Config.cpp"
    "Dnn/Components/Layers/Attention.Cuda.cpp"
    "Dnn/Components/Layers/Attention.Config.cpp"
    "Dnn/Components/Blocks/Transformer.Cuda.cpp"
    "Dnn/Components/Blocks/Transformer.Config.cpp"

    "Dnn/Serialization/ModelArchive.cpp" 
    "Dnn/Modeling/Network.Cpu.cpp"
)

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
