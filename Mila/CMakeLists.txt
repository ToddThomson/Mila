set(CMAKE_CXX_SCAN_FOR_MODULES ON)

add_library( Mila STATIC
    "Src/Dnn/Compute/Devices/Cuda/Helpers/CudaUtils.h"
    "Src/Dnn/Compute/Devices/Cuda/Operations/Kernels/CudaOps.h"

    #--------------------------------------------------------------------------
    # Cuda Operations Kernels
    #--------------------------------------------------------------------------
    "Src/Dnn/Compute/Devices/Cuda/Operations/Kernels/CudaEncoder.cu"
    "Src/Dnn/Compute/Devices/Cuda/Operations/Kernels/CudaLayerNorm.cu"

    "Src/Dnn/Compute/Devices/Cuda/Operations/Kernels/MatMul/CudaReduction.cu"

    "Src/Dnn/Compute/Devices/Cuda/Operations/Kernels/MatMul/CudaMatMulFp32.cu"
    "Src/Dnn/Compute/Devices/Cuda/Operations/Kernels/MatMul/CudaMatMulFp16.cu"
    "Src/Dnn/Compute/Devices/Cuda/Operations/Kernels/MatMul/CudaLinearGelu.cu"

    # TJT: Deprecated "Src/Dnn/Compute/Devices/Cuda/Operations/Kernels/CudaMHA.cu"
    "Src/Dnn/Compute/Devices/Cuda/Operations/Kernels/CudaGelu.cu"
    "Src/Dnn/Compute/Devices/Cuda/Operations/Kernels/CudaResidual.cu"
    "Src/Dnn/Compute/Devices/Cuda/Operations/Kernels/CudaSoftmax.cu"
    "Src/Dnn/Compute/Devices/Cuda/Operations/Kernels/CudaSoftmaxCrossEntropy.cu"
    
    #--------------------------------------------------------------------------
    # Cuda TensorOps Kernels
    #--------------------------------------------------------------------------
    "Src/Dnn/Compute/Devices/Cuda/Tensors/Operations/Kernels/Transfer.Copy.h"
    "Src/Dnn/Compute/Devices/Cuda/Tensors/Operations/Kernels/TensorCopy.cu"
    "Src/Dnn/Compute/Devices/Cuda/Tensors/Operations/Kernels/TensorOps.Fill.h"
    "Src/Dnn/Compute/Devices/Cuda/Tensors/Operations/Kernels/TensorOps.Fill.cu"
    "Src/Dnn/Compute/Devices/Cuda/Tensors/Operations/Kernels/CudaTensorOps.h"
    "Src/Dnn/Compute/Devices/Cuda/Tensors/Operations/Kernels/Math.Elementwise.h"
    "Src/Dnn/Compute/Devices/Cuda/Tensors/Operations/Kernels/Math.Elementwise.cu"

    #--------------------------------------------------------------------------
    # Cuda Optimizer Kernels
    #--------------------------------------------------------------------------
    "Src/Dnn/Compute/Devices/Cuda/Optimizers/Kernels/CudaOptimizers.h"
    "Src/Dnn/Compute/Devices/Cuda/Optimizers/Kernels/CudaAdamW.cu"
    "Src/Dnn/Visualization/Components/Block.Visualizer.ixx"
    "Src/Dnn/Compute/Devices/Cuda/Tensors/Operations/Kernels/Math.Reduction.h"
)

 # FIXME: add_dependencies(Mila MilaUpdateVersion)

 set_target_properties( Mila PROPERTIES
	CXX_STANDARD 23
	CXX_STANDARD_REQUIRED ON
	CXX_EXTENSIONS OFF
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
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
  set(CMAKE_CUDA_HOST_COMPILER "${CMAKE_CXX_COMPILER}")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -ccbin ${CMAKE_CXX_COMPILER}")
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
        # Version and Mila 
        #---------------------------------------------------------------
        "Src/Version.ixx"
        "Src/Mila.ixx"

        #----------------------------------------------------------------------
        # Core
        #----------------------------------------------------------------------
        "Src/Dnn/Core/Component.ixx"
        "Src/Dnn/Core/CompositeComponent.ixx"
        "Src/Dnn/Core/ComponentConfig.ixx"
        "Src/Dnn/Core/ComponentFactory.ixx"

        #---------------------------------------------------------------
        # Top-level Utilities
        #---------------------------------------------------------------
        "Src/Utils/Logger.ixx"
        "Src/Utils/DefaultLogger.ixx"
        "Src/Utils/TrainingLogger.ixx"
        "Src/Utils/RandomGenerator.ixx"
        "Src/Utils/json.ixx"

        #---------------------------------------------------------------
        # Dnn / Tensors (top-level)
        #---------------------------------------------------------------
        "Src/Dnn/Tensors/Tensor.ixx"
        "Src/Dnn/Tensors/TensorBuffer.ixx"
        "Src/Dnn/Tensors/ITensor.ixx"
        "Src/Dnn/Tensors/TensorDataType.ixx"
        "Src/Dnn/Tensors/Tensor.Initializers.ixx"
        "Src/Dnn/Tensors/Tensor.Serialization.ixx"

        "Src/Dnn/Tensors/TensorDataTypeMap.ixx"
        "Src/Dnn/Tensors/TensorDataTypeTraits.ixx"
        "Src/Dnn/Tensors/TensorHostTypeMap.ixx"

        # Dnn / Tensors / Operations
        "Src/Dnn/Tensors/Operations/TensorOps.ixx"
        "Src/Dnn/Tensors/Operations/TensorOps-Base.ixx"
        "Src/Dnn/Tensors/Operations/TensorOps.Math.ixx"
        "Src/Dnn/Tensors/Operations/TensorOps.Fill.ixx"
        "Src/Dnn/Tensors/Operations/TensorOps.Transfer.ixx"
                
        #---------------------------------------------------------------
        # Dnn / Common Components
        #---------------------------------------------------------------
        "Src/Dnn/Components/Activations/ActivationType.ixx"

        #---------------------------------------------------------------
        # Dnn / Compute (root)
        #---------------------------------------------------------------
        "Src/Dnn/Compute/ComputeDevice.ixx"
        "Src/Dnn/Compute/ComputeDeviceTraits.ixx"
        "Src/Dnn/Compute/DeviceType.ixx"
        "Src/Dnn/Compute/DeviceTypeTraits.ixx"

        "Src/Dnn/Compute/Devices/Cuda/CudaDeviceTypeTraits.ixx"
        "Src/Dnn/Compute/Devices/Cpu/CpuDeviceTypeTraits.ixx"
        
        "Src/Dnn/Compute/ComputePrecision.ixx"
        
        "Src/Dnn/Compute/ExecutionContext.ixx"
        "Src/Dnn/Compute/IExecutionContext.ixx"

        # Dnn / Compute / Memory Resources
        "Src/Dnn/Compute/MemoryResource.ixx"
        "Src/Dnn/Compute/MemoryResourceTraits.ixx"
        "Src/Dnn/Compute/MemoryResourceProperties.ixx"
        "Src/Dnn/Compute/MemoryResourceTracker.ixx"

        #----------------------------------------------------------------------
        # Dnn / Compute / Optimizers
        #----------------------------------------------------------------------
        "Src/Dnn/Compute/Optimizers/OptimizerBase.ixx"

        #---------------------------------------------------------------
        # Dnn / Compute / Operations
        #---------------------------------------------------------------
        "Src/Dnn/Compute/Operations/OperationBase.ixx"
        "Src/Dnn/Compute/Operations/OperationType.ixx"
        
        "Src/Dnn/Compute/Operations/UnaryOperation.ixx"
        "Src/Dnn/Compute/Operations/BinaryOperation.ixx"
        
        "Src/Dnn/Compute/Operations/OperationRegistry.ixx"
        "Src/Dnn/Compute/Operations/OperationsRegistrar.ixx"

        #---------------------------------------------------------------
        # Dnn / Compute / Devices (general)
        #---------------------------------------------------------------
        "Src/Dnn/Compute/Registry/DeviceRegistry.ixx"
        "Src/Dnn/Compute/Registry/DeviceRegistrar.ixx"
        "Src/Dnn/Compute/Registry/DeviceRegistryHelpers.ixx"
        
        "Src/Dnn/Compute/ExecutionContextHelpers.ixx"
        
        "Src/Dnn/Compute/Registry/DevicePlugin.ixx"

        #---------------------------------------------------------------
        # Dnn / Compute / Devices / Cpu
        #---------------------------------------------------------------
        "Src/Dnn/Compute/Devices/Cpu/CpuDevice.ixx"
        
        "Src/Dnn/Compute/Devices/Cpu/CpuDevicePlugin.ixx"
        "Src/Dnn/Compute/Devices/Cpu/CpuMemoryResource.ixx"
        
        "Src/Dnn/Compute/Devices/Cpu/CpuExecutionContext.ixx"
        "Src/Dnn/Compute/Devices/Cpu/CpuDeviceTraits.ixx"

        "Src/Dnn/Compute/Devices/Cpu/Tensors/CpuTensorDataTypeTraits.ixx"
        "Src/Dnn/Compute/Devices/Cpu/Tensors/Operations/CpuTensorOps.ixx"
        "Src/Dnn/Compute/Devices/Cpu/Tensors/Operations/CpuTensorOps.Fill.ixx"
        "Src/Dnn/Compute/Devices/Cpu/Tensors/Operations/CpuTensorOps.Math.ixx"
        "Src/Dnn/Compute/Devices/Cpu/Tensors/Operations/CpuTensorOps.Transfer.ixx"
        
        "Src/Dnn/Compute/Devices/Cpu/Operations/CpuOperations.ixx"
        
        "Src/Dnn/Compute/Devices/Cpu/Operations/CpuEncoderOp.ixx"
        "Src/Dnn/Compute/Devices/Cpu/Operations/CpuGeluOp.ixx"
        "Src/Dnn/Compute/Devices/Cpu/Operations/CpuLayerNormOp.ixx"
        "Src/Dnn/Compute/Devices/Cpu/Operations/CpuLinearOp.ixx"
        "Src/Dnn/Compute/Devices/Cpu/Operations/CpuAttentionOp.ixx"
        "Src/Dnn/Compute/Devices/Cpu/Operations/CpuResidualOp.ixx"
        "Src/Dnn/Compute/Devices/Cpu/Operations/CpuSoftmaxCrossEntropyOp.ixx"
        "Src/Dnn/Compute/Devices/Cpu/Operations/CpuSoftmaxOp.ixx"

        #---------------------------------------------------------------
        # Dnn / Compute / Devices / Cuda
        #---------------------------------------------------------------
        # CUDA Helpers
        "Src/Dnn/Compute/Devices/Cuda/Helpers/CudaError.ixx"
        "Src/Dnn/Compute/Devices/Cuda/Helpers/CudaHelpers.ixx"
        "Src/Dnn/Compute/Devices/Cuda/Helpers/CudaBadAlloc.ixx"
        "Src/Dnn/Compute/Devices/Cuda/Helpers/CublasLtError.ixx"
        

        # CUDA Devices
        "Src/Dnn/Compute/Devices/Cuda/CudaDevice.ixx"
        
        "Src/Dnn/Compute/Devices/Cuda/CudaDeviceProps.ixx"
        "Src/Dnn/Compute/Devices/Cuda/CudaDeviceMemoryResource.ixx"
        "Src/Dnn/Compute/Devices/Cuda/CudaManagedMemoryResource.ixx"
        "Src/Dnn/Compute/Devices/Cuda/CudaPinnedMemoryResource.ixx"
        
        "Src/Dnn/Compute/Devices/Cuda/CudaDevicePlugin.ixx"
        "Src/Dnn/Compute/Devices/Cuda/CudaDeviceTraits.ixx"
        "Src/Dnn/Compute/Devices/Cuda/CudaExecutionContext.ixx"

        # CUDA / Tensors
        "Src/Dnn/Compute/Devices/Cuda/Tensors/CudaTensorDataType.ixx"
        "Src/Dnn/Compute/Devices/Cuda/Tensors/CudaTensorDataType-Specializations.ixx"
        "Src/Dnn/Compute/Devices/Cuda/Tensors/CudaTensorDataType-Maps.ixx"
        
        "Src/Dnn/Compute/Devices/Cuda/Tensors/Operations/CudaTensorOps.ixx"
        "Src/Dnn/Compute/Devices/Cuda/Tensors/Operations/CudaTensorOps.Math.ixx"
        "Src/Dnn/Compute/Devices/Cuda/Tensors/Operations/CudaTensorOps.Fill.ixx"
        "Src/Dnn/Compute/Devices/Cuda/Tensors/Operations/CudaTensorOps.Transfer.ixx"

        # CUDA / Operations
        "Src/Dnn/Compute/Devices/Cuda/Operations/CudaOperations.ixx"
        "Src/Dnn/Compute/Devices/Cuda/Operations/CudaAttentionOp.ixx"
        "Src/Dnn/Compute/Devices/Cuda/Operations/CudaEncoderOp.ixx"
        #"Src/Dnn/Compute/Operations/Cuda/CudaDataTypeTraits.ixx"
        "Src/Dnn/Compute/Devices/Cuda/Operations/CublasLtMatMulBias.ixx"
        #"Src/Dnn/Compute/Operations/Cuda/MatMulBiasActivation.ixx"
        "Src/Dnn/Compute/Devices/Cuda/Operations/CudaGeluOp.ixx"
        "Src/Dnn/Compute/Devices/Cuda/Operations/CudaLayerNormOp.ixx"
        "Src/Dnn/Compute/Devices/Cuda/Operations/CudaLinearOp.ixx"
        "Src/Dnn/Compute/Devices/Cuda/Operations/CudaResidualOp.ixx"
        "Src/Dnn/Compute/Devices/Cuda/Operations/CudaSoftmaxOp.ixx"
        "Src/Dnn/Compute/Devices/Cuda/Operations/CudaSoftmaxCrossEntropyOp.ixx"

        # CUDA / Optimizers
        "Src/Dnn/Compute/Devices/Cuda/Optimizers/CudaAdamWOptimizer.ixx"

        #---------------------------------------------------------------
        # Dnn / Compute / Devices / Metal
        #---------------------------------------------------------------
        "Src/Dnn/Compute/Devices/Metal/MetalDevice.ixx"
        "Src/Dnn/Compute/Devices/Metal/MetalExecutionContext.ixx"
        "Src/Dnn/Compute/Devices/Metal/MetalDevicePlugin.ixx"
        "Src/Dnn/Compute/Devices/Metal/MetalMemoryResource.ixx"
        "Src/Dnn/Compute/Devices/Metal/Tensors/MetalTensorTraits.ixx"

        #---------------------------------------------------------------
        # Dnn / Compute / Devices / OpenCL
        #---------------------------------------------------------------
        "Src/Dnn/Compute/Devices/OpenCL/OpenCLExecutionContext.ixx"
        "Src/Dnn/Compute/Devices/OpenCL/OpenCLMemoryResource.ixx"
        "Src/Dnn/Compute/Devices/OpenCL/Tensors/OpenCLTensorTraits.ixx"

        #---------------------------------------------------------------
        # Dnn / Compute / Devices / Vulkan
        #---------------------------------------------------------------
        "Src/Dnn/Compute/Devices/Vulkan/VulkanExecutionContext.ixx"
        "Src/Dnn/Compute/Devices/Vulkan/VulkanMemoryResource.ixx"
        "Src/Dnn/Compute/Devices/Vulkan/Tensors/VulkanTensorTraits.ixx"

        # Components - Activations
        "Src/Dnn/Components/Activations/Gelu.ixx"
        "Src/Dnn/Components/Activations/GeluConfig.ixx"
        
        # Components - Connections
        "Src/Dnn/Components/Connections/Residual.ixx"
        "Src/Dnn/Components/Connections/ResidualConfig.ixx"

        # Components - Layers
        "Src/Dnn/Components/Layers/Attention.ixx"
        "Src/Dnn/Components/Layers/AttentionConfig.ixx"
        "Src/Dnn/Components/Layers/Linear.ixx"
        "Src/Dnn/Components/Layers/LinearConfig.ixx"
        "Src/Dnn/Components/Layers/Encoder.ixx"
        "Src/Dnn/Components/Layers/EncoderConfig.ixx"
        "Src/Dnn/Components/Losses/Softmax.ixx"
        "Src/Dnn/Components/Losses/SoftmaxConfig.ixx"

        # Normalization / Losses / Regularization / Blocks
        "Src/Dnn/Components/Normalization/LayerNorm.ixx"
        "Src/Dnn/Components/Normalization/LayerNormConfig.ixx"
        
        "Src/Dnn/Components/Losses/SoftmaxCrossEntropy.ixx"
        "Src/Dnn/Components/Losses/CrossEntropyConfig.ixx"
        
        #"Src/Dnn/Components/Regularization/Dropout.ixx"
        #"Src/Dnn/Components/Regularization/DropoutConfig.ixx"
        
        # --- Blocks ---
        "Src/Dnn/Components/Blocks/MLP.ixx"
        "Src/Dnn/Components/Blocks/MLPConfig.ixx"
        
        "Src/Dnn/Components/Blocks/Transformer.ixx"
        "Src/Dnn/Components/Blocks/Transformer.Config.ixx"
        
        #----------------------------------------------------------------------
        # Dnn / Data
        #----------------------------------------------------------------------
        "Src/Dnn/Data/DatasetLoader.ixx"

        #---------------------------------------------------------------
        # Dnn / Serialization
        #---------------------------------------------------------------
        "Src/Dnn/Serialization/SerializationMode.ixx"
        "Src/Dnn/Serialization/ModelArchive.ixx"
        "Src/Dnn/Serialization/ModelSerializer.ixx"
        "Src/Dnn/Serialization/ZipSerializer.ixx"

        #---------------------------------------------------------------
        # Dnn / Visualization
        #---------------------------------------------------------------
        "Src/Dnn/Visualization/Core/ComponentVisualizer.ixx"
        "Src/Dnn/Visualization/Components/LayerNorm.Visualizer.ixx"
        "Src/Dnn/Visualization/Core/VisualizerContext.ixx"
        "Src/Dnn/Visualization/Rendering/FrameBuffer.ixx"
        "Src/Dnn/Visualization/Rendering/ColorLUT.ixx"
        "Src/Dnn/Visualization/Rendering/HeatMapRenderer.ixx"
        "Src/Dnn/Visualization/Components/Block.Visualizer.ixx"
        "Src/Dnn/Visualization/Components/MLP.Visualizer.ixx"
        
        
        "Src/Dnn/Compute/Operations/OperationRegistryHelpers.ixx"
        "Src/Dnn/Compute/Devices/Vulkan/VulkanDevice.ixx"
        "Src/Dnn/Compute/Devices/OpenCL/OpenCLDevice.ixx"
        "Src/Dnn/Compute/DeviceDescriptor.ixx" 
        "Src/Dnn/Compute/Devices/Cuda/CudaDeviceResources.ixx"
        "Src/Dnn/Tensors/Tensor.Partitioning.ixx"
        "Src/Dnn/Tensors/Tensor.Types.ixx"
        "Src/Dnn/Compute/Devices/Cpu/Optimizers/CpuAdamWOptimizer.ixx" 
        "Src/Dnn/Optimizers/AdamW.ixx"
        "Src/Dnn/Optimizers/AdamWConfig.ixx"
        "Src/Dnn/Components/Connections/ConnectionType.ixx"
        #"Src/Dnn/Compute/Devices/Cuda/Operations/Kernels/CublasLtAttention.ixx"

        #----------------------------------------------------------------------
        # Network
        #---------------------------------------------------------------
        "Src/Dnn/Network/Network.ixx"
        "Src/Dnn/Network/NetworkFactory.ixx"

        #----------------------------------------------------------------------
        # Dnn / Modeling
        #----------------------------------------------------------------------
        "Src/Dnn/Modeling/Model.ixx"
        "Src/Dnn/Modeling/ModelConfig.ixx"
        "Src/Dnn/Modeling/ModelMetadata.ixx"

        "Src/Dnn/Modeling/Checkpoints/CheckpointManager.ixx"
        "Src/Dnn/Modeling/Checkpoints/CheckpointMetaData.ixx"
        
        "Src/Dnn/Modeling/TrainingHistory.ixx"
        #"Src/Dnn/Modeling/ModelExporter.ixx"

        #----------------------------------------------------------------------
        # Dnn Runtime Engine
        #----------------------------------------------------------------------
        #"Src/Dnn/Engine/InferenceEngine.ixx"
        "Src/Dnn/Serialization/OpenMode.ixx"
        "Src/Dnn/Extensibility/PluginManager.ixx"
        "Src/Dnn/Extensibility/IModulePlugin.ixx"
        "Src/Dnn/Extensibility/PluginInfo.ixx"
        "Src/Dnn/Components/MilaComponents.ixx"
        "Src/Dnn/Core/Loss.ixx"
 "Src/Dnn/Core/ComponentType.ixx")

target_link_libraries(Mila PRIVATE 
    miniz
    nlohmann_json::nlohmann_json
    # CUDA::nvml -> not used in the current code, but can be added if needed
    CUDA::cublas 
    CUDA::cublasLt 
    CUDA::cudart
)

# Parse the version from Version.txt
file(READ "${CMAKE_SOURCE_DIR}/Version.txt" VERSION_CONTENT)

# Parse the version components using regular expressions
string(REGEX MATCH "([0-9]+)\\.([0-9]+)\\.([0-9]+)-([a-zA-Z]+)" VERSION_MATCH ${VERSION_CONTENT})

if(VERSION_MATCH)
  set(MILA_VERSION_MAJOR ${CMAKE_MATCH_1})
  set(MILA_VERSION_MINOR ${CMAKE_MATCH_2})
  set(MILA_VERSION_PATCH ${CMAKE_MATCH_3})
  set(MILA_VERSION_PRERELEASE_TAG ${CMAKE_MATCH_4})
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

# Guard the copy so CI / checkouts without a `data` directory do not fail the build.
if(EXISTS "${CMAKE_SOURCE_DIR}/data")
    add_custom_target(MilaCopyData 
        COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_BINARY_DIR}/data"
        COMMAND ${CMAKE_COMMAND} -E copy_directory_if_different 
            "${CMAKE_SOURCE_DIR}/data"
            "${CMAKE_BINARY_DIR}/data"
        COMMENT "Copying model data to build directory"
        VERBATIM
    )

    if( MILA_ENABLE_COPY_MODEL_DATA AND NOT DEFINED ENV{CI} )
        set_target_properties(MilaCopyData PROPERTIES EXCLUDE_FROM_ALL FALSE)
    else()
        set_target_properties(MilaCopyData PROPERTIES EXCLUDE_FROM_ALL TRUE)
    endif()
else()
    message(STATUS "No 'data' directory found in source tree; skipping MilaCopyData target")
    # harmless placeholder so references to MilaCopyData won't break other targets
    add_custom_target(MilaCopyData
        COMMENT "No model data to copy"
        VERBATIM
    )
    set_target_properties(MilaCopyData PROPERTIES EXCLUDE_FROM_ALL TRUE)
endif()

add_subdirectory( Docs )

 # Configure code coverage for MSVC
 if( MILA_ENABLE_COVERAGE AND MSVC)
    add_compile_options(/Zi /Od)
    add_link_options(/DEBUG:FULL /PROFILE)
    
    message(STATUS "Code coverage enabled for MSVC")
endif()

if(MILA_ENABLE_TESTING)  
    enable_testing()
    add_subdirectory(Tests) 
endif()

if(MILA_ENABLE_BENCHMARKS)
    add_subdirectory(Benchmarks)
endif()
 
if(MILA_ENABLE_SAMPLES)
     message(STATUS "MILA_ENABLE_SAMPLES is ON - adding Samples subdirectory")
     add_subdirectory(Samples)
else()
    message(STATUS "MILA_ENABLE_SAMPLES is OFF - skipping Samples")
endif()
