add_library( Mila STATIC
	"Src/Dnn/Compute/Operations/Cuda/Kernels/Cuda.Utils.h"
	"Src/Dnn/Compute/Operations/Cuda/Kernels/Cuda.Ops.h"
	"Src/Dnn/Compute/Operations/Cuda/Kernels/Cuda.Encoder.cu"
	"Src/Dnn/Compute/Operations/Cuda/Kernels/CudaLayerNorm.cu"
	"Src/Dnn/Compute/Operations/Cuda/Kernels/Cuda.MatMul.cu"
	#"Src/Dnn/Compute/Operations/Cuda/Kernels/CudaMHA.cu"
	"Src/Dnn/Compute/Operations/Cuda/Kernels/Cuda.Gelu.cu"
	"Src/Dnn/Compute/Operations/Cuda/Kernels/CudaResidual.cu"
	"Src/Dnn/Compute/Operations/Cuda/Kernels/CudaSoftmax.cu"
	"Src/Dnn/Compute/Operations/Cuda/Kernels/FusedMatMulBiasGelu.cu"
 )

set_target_properties( Mila PROPERTIES
	CXX_STANDARD 23
	CXX_STANDARD_REQUIRED ON
	CXX_EXTENSIONS OFF )

target_compile_features( Mila PUBLIC cxx_std_23 )

find_package(miniz REQUIRED)  

target_include_directories( Mila INTERFACE ${miniz_SOURCE_DIR})
target_link_libraries( Mila PRIVATE miniz)

# Add explicit compiler-specific module flags
if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
  add_compile_options(/std:c++latest)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  add_compile_options(-fmodules-ts)
  add_compile_options(-std=c++23)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-fmodules)
endif()

if(MILA_ENABLE_OPENMP)
	find_package(OpenMP REQUIRED)
	if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
		target_compile_options( Mila PRIVATE -openmp:llvm -DUSE_OMP )
		target_link_options( Mila PRIVATE -openmp:llvm )
	else()
		target_compile_options( Mila PRIVATE ${OpenMP_CXX_FLAGS})
		target_link_options( Mila PRIVATE OpenMP::OpenMP_CXX)
	endif()
endif()

target_sources( Mila
PUBLIC
	FILE_SET module_files
       TYPE CXX_MODULES FILES
		# Version
		"Src/Version.ixx"
		"Src/Mila.ixx"
		"Src/Misc/Glob.ixx"
		
		# Helpers
		"Src/Helpers/Memory.ixx"
		"Src/Utils/TrainingLogger.ixx"
		
		# Tensors
		"Src/Dnn/Core/Tensor.ixx"
		"Src/Dnn/Core/TensorBuffer.ixx"
		"Src/Dnn/Core/TensorType.ixx" 
		"Src/Dnn/Core/TensorTraits.ixx" 
		"Src/Dnn/Core/TensorHelpers.ixx"
		# Modules
		"Src/Dnn/Model.ixx" 
		"Src/Dnn/Modules/Module.ixx" 
		"Src/Dnn/Modules/Layers/LayerNorm.ixx"
		"Src/Dnn/Modules/Layers/FullyConnected.ixx"
		"Src/Dnn/Modules/Layers/Gelu.ixx"
		"Src/Dnn/Modules/Layers/MultiHeadAttention.ixx"
		"Src/Dnn/Modules/Layers/Residual.ixx"
		"Src/Dnn/Modules/Layers/Softmax.ixx"
		"Src/Dnn/Modules/Layers/Encoder.ixx"
		"Src/Dnn/Modules/Blocks/MLP.ixx"
		"Src/Dnn/Modules/Blocks/TransformerBlock.ixx"
		#Dnn Compute
		"Src/Dnn/Compute/DeviceType.ixx"
		"Src/Dnn/Compute/Devices/DeviceRegistry.ixx"
		"Src/Dnn/Compute/ComputeDevice.ixx"  
		"Src/Dnn/Compute/Devices/DeviceHelpers.ixx" 
		"Src/Dnn/Compute/Operations/Base/OperationBase.ixx" 
		"Src/Dnn/Compute/Operations/OperationType.ixx"
		"Src/Dnn/Compute/Operations/OperationAttributes.ixx"
		"Src/Dnn/Compute/Operations/Base/UnaryOperation.ixx" 
		"Src/Dnn/Compute/Operations/Base/BinaryOperation.ixx"
		"Src/Dnn/Compute/DeviceContext.ixx"
		"Src/Dnn/Compute/Operations/OperationRegistry.ixx"
		"Src/Dnn/Compute/Operations/OperationsRegistrar.ixx"  
		"Src/Dnn/Compute/ComputeResource.ixx"
		"Src/Dnn/Compute/MemoryResource.ixx"
		"Src/Dnn/Compute/MemoryResourceProperties.ixx"
		
		# Dnn Compute CPU
		"Src/Dnn/Compute/Devices/Cpu/CpuDevice.ixx" 
		"Src/Dnn/Compute/Operations/Cpu/CpuEncoderOp.ixx" 
		"Src/Dnn/Compute/Operations/Cpu/CpuLayerNormOp.ixx" 
		"Src/Dnn/Compute/Operations/Cpu/CpuMutiHeadAttentionOp.ixx" 
		"Src/Dnn/Compute/Operations/Cpu/CpuResidualOp.ixx" 
		"Src/Dnn/Compute/Operations/Cpu/CpuFullyConnectedOp.ixx"
		"Src/Dnn/Compute/Operations/Cpu/CpuSoftmaxOp.ixx"
		"Src/Dnn/Compute/Operations/Cpu/CpuGeluOp.ixx"
		"Src/Dnn/Compute/Operations/Cpu/CpuCrossEntropyOp.ixx"  
		"Src/Dnn/Compute/Operations/Cpu/CpuOperations.ixx"
		"Src/Dnn/Compute/Devices/Cpu/CpuMemoryResource.ixx"
		
		# Dnn Compute CUDA
		"Src/Dnn/Compute/Cuda/CudaError.ixx"
		"Src/Dnn/Compute/Cuda/CudaHelpers.ixx"
		"Src/Dnn/Compute/Devices/Cuda/CudaDevice.ixx"
		"Src/Dnn/Compute/Operations/Cuda/CudaOperations.ixx"
		"Src/Dnn/Compute/Devices/Cuda/CudaDeviceProps.ixx"
		"Src/Dnn/Compute/Devices/Cuda/CudaMemoryResource.ixx"
		"Src/Dnn/Compute/Devices/Cuda/CudaManagedMemoryResource.ixx"
		"Src/Dnn/Compute/Devices/Cuda/CudaPinnedMemoryResource.ixx"
		# CUDA Ops
		"Src/Dnn/Compute/Operations/Cuda/CudaFullyConnectedOp.ixx"
		"Src/Dnn/Compute/Operations/Cuda/CudaResidualOp.ixx"
		"Src/Dnn/Compute/Operations/Cuda/CudaGeluOp.ixx" 
		"Src/Dnn/Compute/Operations/Cuda/CudaSoftmaxOp.ixx" 
		"Src/Dnn/Compute/Operations/Cuda/CudaLayerNormOp.ixx"
		"Src/Dnn/Compute/Operations/Cuda/CudaMultiHeadAttentionOp.ixx" 
		"Src/Dnn/Compute/Operations/Cuda/CudaEncoderOp.ixx" 
		# Data
		"Src/Data/DataLoader.ixx"
		"Src/Dnn/Models/Gpt2/Gpt2DatasetReader.ixx"

		"Src/Dnn/Compute/Devices/Cpu/CpuComputeResource.ixx"
		"Src/Dnn/Graph.ixx"
		"Src/Dnn/Compute/ComputeResourceHelpers.ixx"
		"Src/Dnn/Compute/Cuda/CudaComputeResource.ixx"
		 
		
		# Utils
		"Src/Utils/Logger.ixx"
		"Src/Utils/DefaultLogger.ixx"

		"Src/Dnn/ModelCallback.ixx"
		"Src/Dnn/Compute/Cuda/CudaBadAlloc.ixx"
		"Src/Dnn/Compute/MemoryResourceTracker.ixx"
		"Src/Dnn/Core/TensorPtr.ixx"
		#"Src/Dnn/FusedModule.ixx"
		"Src/Dnn/Compute/Operations/Cuda/MatMulBiasActivation.ixx"
		"Src/Dnn/Compute/Devices/DeviceRegistrar.ixx" 
		"Src/Dnn/Compute/DynamicMemoryResource.ixx"
		"Src/Dnn/Compute/DeviceContextHelpers.ixx"
		"Src/Dnn/Modules/Block.ixx"
 "Src/Core/RandomGenerator.ixx")

target_link_libraries( Mila PRIVATE CUDA::nvml CUDA::cublas CUDA::cublasLt CUDA::cudart )

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

add_subdirectory( Docs )

if (MILA_ENABLE_TESTING)  
enable_testing()
add_subdirectory( Tests) 
endif()  
 
if (MILA_ENABLE_SAMPLES)  
#add_subdirectory(Mila/Samples)  
endif()
