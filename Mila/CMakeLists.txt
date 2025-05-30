set(CMAKE_CXX_SCAN_FOR_MODULES ON)

add_library( Mila STATIC
	"Src/Dnn/Compute/Operations/Cuda/Kernels/CudaUtils.h"
	"Src/Dnn/Compute/Operations/Cuda/Kernels/CudaOps.h"

	"Src/Dnn/Compute/Operations/Cuda/Kernels/CudaEncoder.cu"
	"Src/Dnn/Compute/Operations/Cuda/Kernels/CudaLayerNorm.cu"

	"Src/Dnn/Compute/Operations/Cuda/Kernels/MatMul/CudaMatMulFp32.cu"
	#"Src/Dnn/Compute/Operations/Cuda/Kernels/MatMul/CudaMatMulFp16.cu"
	"Src/Dnn/Compute/Operations/Cuda/Kernels/MatMul/FusedMatMulBiasGelu.cu"
	#"Src/Dnn/Compute/Operations/Cuda/Kernels/CudaMHA.cu"
	"Src/Dnn/Compute/Operations/Cuda/Kernels/CudaGelu.cu"
	"Src/Dnn/Compute/Operations/Cuda/Kernels/CudaResidual.cu"
	"Src/Dnn/Compute/Operations/Cuda/Kernels/CudaSoftmax.cu"
 )

 set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

set_target_properties( Mila PROPERTIES
	CXX_STANDARD 23
	CXX_STANDARD_REQUIRED ON
	CXX_EXTENSIONS OFF
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES 75 # For Turing GPUs (T4, V100, A100, H100, etc.)
)

target_compile_features( Mila PUBLIC cxx_std_23 )

find_package(miniz REQUIRED)  
find_package(nlohmann_json REQUIRED)

target_include_directories( Mila PRIVATE ${miniz_SOURCE_DIR})

if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
  target_compile_options(Mila PRIVATE 
    $<$<COMPILE_LANGUAGE:CXX>:/std:c++latest /permissive- /Zc:preprocessor /W4>
  )
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  target_compile_options(Mila PRIVATE 
    $<$<COMPILE_LANGUAGE:CXX>:-fmodules-ts -std=c++23>
  )
elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  target_compile_options(Mila PRIVATE 
    $<$<COMPILE_LANGUAGE:CXX>:-std=c++23 -fno-implicit-modules -fno-implicit-module-maps>
  )
  set(CMAKE_CXX_COMPILER_CLANG_SCAN_DEPS "/usr/bin/clang-scan-deps-19")
endif()

if(MILA_ENABLE_OPENMP)
  find_package(OpenMP REQUIRED)
  if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    target_compile_options(Mila PRIVATE 
      $<$<COMPILE_LANGUAGE:CXX>:-openmp:llvm -DUSE_OMP>
    )
    target_link_options(Mila PRIVATE -openmp:llvm)
  else()
    target_compile_options(Mila PRIVATE 
      $<$<COMPILE_LANGUAGE:CXX>:${OpenMP_CXX_FLAGS}>
    )
    target_link_libraries(Mila PRIVATE OpenMP::OpenMP_CXX)
  endif()
endif()

target_sources( Mila
PUBLIC
    FILE_SET module_files
       TYPE CXX_MODULES FILES
        #---------------------------------------------------------------
        # Core and Version
        #---------------------------------------------------------------
        "Src/Version.ixx"
        "Src/Mila.ixx"
        
        #---------------------------------------------------------------
        # Common Utilities
        #---------------------------------------------------------------
        "Src/Misc/Glob.ixx"
        "Src/Helpers/Memory.ixx"
        "Src/Utils/Logger.ixx"
        "Src/Utils/DefaultLogger.ixx"
        "Src/Utils/TrainingLogger.ixx"
        "Src/Utils/RandomGenerator.ixx"
        
        #---------------------------------------------------------------
        # Tensor System
        #---------------------------------------------------------------
        "Src/Dnn/Tensor/Tensor.ixx"
        "Src/Dnn/Tensor/TensorBuffer.ixx"
        "Src/Dnn/Tensor/TensorType.ixx" 
        "Src/Dnn/Tensor/TensorTraits.ixx" 
        "Src/Dnn/Tensor/TensorHelpers.ixx"
        "Src/Dnn/Tensor/TensorPtr.ixx"
        
        #---------------------------------------------------------------
        # Common Components
        #---------------------------------------------------------------
        "Src/Dnn/Common/ActivationType.ixx"
        "Src/Dnn/Common/PrecisionConfig.ixx"
        
        #---------------------------------------------------------------
        # Device Management
        #---------------------------------------------------------------
        "Src/Dnn/Compute/ComputeDevice.ixx"
        "Src/Dnn/Compute/Devices/DeviceType.ixx"
        "Src/Dnn/Compute/Devices/DeviceRegistry.ixx"
        "Src/Dnn/Compute/Devices/DeviceRegistrar.ixx"
        "Src/Dnn/Compute/Devices/DeviceHelpers.ixx"
        "Src/Dnn/Compute/Devices/DeviceContext.ixx"
        "Src/Dnn/Compute/Devices/DeviceContextHelpers.ixx"
        
        #---------------------------------------------------------------
        # Memory Resources
        #---------------------------------------------------------------
        "Src/Dnn/Compute/MemoryResource.ixx"
        "Src/Dnn/Compute/ComputeResource.ixx"
        "Src/Dnn/Compute/DynamicMemoryResource.ixx"
        "Src/Dnn/Compute/MemoryResourceProperties.ixx"
        "Src/Dnn/Compute/MemoryResourceTracker.ixx"
        "Src/Dnn/Compute/ComputeResourceHelpers.ixx"
        "Src/Dnn/Compute/ComputePrecision.ixx"
        
        #---------------------------------------------------------------
        # Operation Framework
        #---------------------------------------------------------------
        "Src/Dnn/Compute/Operations/OperationBase.ixx" 
        "Src/Dnn/Compute/Operations/OperationType.ixx"
        "Src/Dnn/Compute/Operations/OperationAttributes.ixx"
        "Src/Dnn/Compute/Operations/UnaryOperation.ixx" 
        "Src/Dnn/Compute/Operations/BinaryOperation.ixx"
        "Src/Dnn/Compute/Operations/OperationRegistry.ixx"
        "Src/Dnn/Compute/Operations/OperationsRegistrar.ixx"
        
        #---------------------------------------------------------------
        # CPU Implementation
        #---------------------------------------------------------------
        "Src/Dnn/Compute/Devices/Cpu/CpuDevice.ixx"
        "Src/Dnn/Compute/Devices/Cpu/CpuMemoryResource.ixx"
        "Src/Dnn/Compute/Devices/Cpu/CpuComputeResource.ixx"
        
        # CPU Operations
        "Src/Dnn/Compute/Operations/Cpu/CpuOperations.ixx"
        "Src/Dnn/Compute/Operations/Cpu/CpuEncoderOp.ixx" 
        "Src/Dnn/Compute/Operations/Cpu/CpuLayerNormOp.ixx" 
        "Src/Dnn/Compute/Operations/Cpu/CpuMutiHeadAttentionOp.ixx" 
        "Src/Dnn/Compute/Operations/Cpu/CpuResidualOp.ixx" 
        "Src/Dnn/Compute/Operations/Cpu/CpuLinearOp.ixx"
        "Src/Dnn/Compute/Operations/Cpu/CpuSoftmaxOp.ixx"
        "Src/Dnn/Compute/Operations/Cpu/CpuGeluOp.ixx"
        "Src/Dnn/Compute/Operations/Cpu/CpuCrossEntropyOp.ixx"
        
        #---------------------------------------------------------------
        # CUDA Implementation
        #---------------------------------------------------------------
        # CUDA Core
        "Src/Dnn/Compute/Cuda/CudaError.ixx"
        "Src/Dnn/Compute/Cuda/CudaHelpers.ixx"
        "Src/Dnn/Compute/Cuda/CudaComputeResource.ixx"
        "Src/Dnn/Compute/Cuda/CudaBadAlloc.ixx"
        "Src/Dnn/Compute/Cuda/CublasLtError.ixx"
        
        # CUDA Devices
        "Src/Dnn/Compute/Devices/Cuda/CudaDevice.ixx"
        "Src/Dnn/Compute/Devices/Cuda/CudaDeviceProps.ixx"
        "Src/Dnn/Compute/Devices/Cuda/CudaMemoryResource.ixx"
        "Src/Dnn/Compute/Devices/Cuda/CudaManagedMemoryResource.ixx"
        "Src/Dnn/Compute/Devices/Cuda/CudaPinnedMemoryResource.ixx"
        
        # CUDA Operations
        "Src/Dnn/Compute/Operations/Cuda/CudaOperations.ixx"
        "Src/Dnn/Compute/Operations/Cuda/CudaDataTypeTraits.ixx"
        "Src/Dnn/Compute/Operations/Cuda/CublasLtMatMulBias.ixx"
        "Src/Dnn/Compute/Operations/Cuda/MatMulBiasActivation.ixx"
        "Src/Dnn/Compute/Operations/Cuda/CudaLinearOp.ixx"
        "Src/Dnn/Compute/Operations/Cuda/CudaResidualOp.ixx"
        "Src/Dnn/Compute/Operations/Cuda/CudaGeluOp.ixx" 
        "Src/Dnn/Compute/Operations/Cuda/CudaSoftmaxOp.ixx" 
        "Src/Dnn/Compute/Operations/Cuda/CudaLayerNormOp.ixx"
        "Src/Dnn/Compute/Operations/Cuda/CudaMultiHeadAttentionOp.ixx" 
        "Src/Dnn/Compute/Operations/Cuda/CudaEncoderOp.ixx"
        
        #---------------------------------------------------------------
        # Module Framework
        #---------------------------------------------------------------
        "Src/Dnn/Modules/Module.ixx"
        "Src/Dnn/Common/ComponentConfig.ixx"
        "Src/Dnn/Modules/CompositeModule.ixx"
        
        #---------------------------------------------------------------
        # Module Implementations - Grouped by Category
        #---------------------------------------------------------------
        # Activations
        "Src/Dnn/Modules/Activations/Gelu.ixx"
        "Src/Dnn/Modules/Activations/GeluConfig.ixx"
        
        # Layers
        "Src/Dnn/Modules/Layers/Linear.ixx"
        "Src/Dnn/Modules/Layers/LinearConfig.ixx"
        "Src/Dnn/Modules/Layers/Encoder.ixx"
        "Src/Dnn/Modules/Layers/EncoderConfig.ixx"
        "Src/Dnn/Modules/Layers/MultiHeadAttention.ixx"
        "Src/Dnn/Modules/Layers/MultiHeadAttentionConfig.ixx"
        "Src/Dnn/Modules/Layers/Residual.ixx"
        "Src/Dnn/Modules/Layers/ResidualConfig.ixx"
        "Src/Dnn/Modules/Layers/Softmax.ixx"
        "Src/Dnn/Modules/Layers/SoftmaxConfig.ixx"
        
        # Normalization
        "Src/Dnn/Modules/Normalization/LayerNorm.ixx"
        "Src/Dnn/Modules/Normalization/LayerNormConfig.ixx"
        
        # Losses
        "Src/Dnn/Modules/Losses/CrossEntropy.ixx"
        "Src/Dnn/Modules/Losses/CrossEntropyConfig.ixx"
        
        # Regularization
        "Src/Dnn/Modules/Regularization/Dropout.ixx"
        "Src/Dnn/Modules/Regularization/DroputConfig.ixx"
        
        # Blocks
        "Src/Dnn/Modules/Blocks/MLP.ixx"
        "Src/Dnn/Modules/Blocks/MLPConfig.ixx" 
        "Src/Dnn/Modules/Blocks/TransformerBlock.ixx"
        "Src/Dnn/Modules/Blocks/TransformerBlockConfig.ixx"
        
        #---------------------------------------------------------------
        # Models
        #---------------------------------------------------------------
        "Src/Dnn/Models/Model.ixx"
        "Src/Dnn/Models/Graph.ixx"
        "Src/Dnn/Models/ModelCallback.ixx"
        "Src/Dnn/Models/Gpt2/Gpt2DatasetReader.ixx"
        
        #---------------------------------------------------------------
        # Data Handling
        #---------------------------------------------------------------
        "Src/Dnn/Data/DataLoader.ixx"

        #---------------------------------------------------------------
        # Serialization
        #---------------------------------------------------------------
        "Src/Dnn/Serialization/ModuleSerializer.ixx"
        "Src/Dnn/Serialization/ZipFileSerializer.ixx"
        "Src/Dnn/Serialization/ModelArchive.ixx"

        #---------------------------------------------------------------
        # WIP
        #---------------------------------------------------------------
        "Src/Dnn/Common/Component.ixx"
)

target_link_libraries(Mila PRIVATE 
    miniz
    nlohmann_json::nlohmann_json
    CUDA::nvml 
    CUDA::cublas 
    CUDA::cublasLt 
    CUDA::cudart
)

# Parse the version from Version.txt
file(READ "${CMAKE_SOURCE_DIR}/Version.txt" VERSION_CONTENT)

# Parse the version components using regular expressions
string(REGEX MATCH "([0-9]+)\\.([0-9]+)\\.([0-9]+)-([a-zA-Z]+)\\.([0-9]+)" VERSION_MATCH ${VERSION_CONTENT})

if(VERSION_MATCH)
  set(MILA_VERSION_MAJOR ${CMAKE_MATCH_1})
  set(MILA_VERSION_MINOR ${CMAKE_MATCH_2})
  set(MILA_VERSION_PATCH ${CMAKE_MATCH_3})
  set(MILA_VERSION_PRERELEASE_TAG ${CMAKE_MATCH_4})
  set(MILA_VERSION_PRERELEASE ${CMAKE_MATCH_5})
else()
  message(FATAL_ERROR "Failed to parse version from Version.txt")
endif()

# Configure the Version.h file from Version.h.in template
configure_file(
  "${CMAKE_SOURCE_DIR}/Mila/Src/Version.h.in"
  "${CMAKE_BINARY_DIR}/Mila/Src/Version.h"
  @ONLY
)

include_directories(${CMAKE_BINARY_DIR}/Mila/Src)
target_include_directories(Mila PRIVATE ${CMAKE_BINARY_DIR}/Mila/Src)

install(TARGETS Mila
    EXPORT MilaTargets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    FILE_SET module_files DESTINATION include
)

add_subdirectory( Docs )

if (MILA_ENABLE_TESTING)  
enable_testing()
add_subdirectory(Tests) 
endif()

if (MILA_ENABLE_BENCHMARKS)
#add_subdirectory(Benchmarks)
endif()
 
if (MILA_ENABLE_SAMPLES)  
#add_subdirectory(Samples)  
endif()
