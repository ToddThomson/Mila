# CMake project for Mila Library

add_library( Mila STATIC
	"Src/Dnn/Compute/Cuda/Kernels/Cuda.Utils.h"	
	"Src/Dnn/Compute/Cuda/Kernels/Cuda.Ops.h"
	"Src/Dnn/Compute/Cuda/Kernels/Cuda.Encoder.cu"
	"Src/Dnn/Compute/Cuda/Kernels/CudaLayerNorm.cu"
	"Src/Dnn/Compute/Cuda/Kernels/Cuda.MatMul.cu"
	#"Src/Dnn/Compute/Cuda/Kernels/CudaMHA.cu"
	"Src/Dnn/Compute/Cuda/Kernels/Cuda.Gelu.cu"
	"Src/Dnn/Compute/Cuda/Kernels/CudaResidual.cu"
	#"Src/Dnn/Compute/Cuda/Kernels/CudaSoftmax.cu"
)

set_target_properties( Mila PROPERTIES
	CXX_STANDARD_REQUIRED ON
	CXX_EXTENSIONS OFF )

# TODO: Add globally 
target_compile_features( Mila PUBLIC cxx_std_23 )

find_package(miniz REQUIRED)  

target_include_directories( Mila INTERFACE ${miniz_SOURCE_DIR})
target_link_libraries( Mila PRIVATE miniz)

if(MILA_ENABLE_OPENMP)  
	if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
		target_compile_options( Mila PRIVATE -openmp:llvm -DUSE_OMP )
		target_link_options( Mila PRIVATE -openmp:llvm )
	else()
		target_compile_options( Mila PRIVATE ${OpenMP_CXX_FLAGS})
		target_link_options( Mila PRIVATE OpenMP::OpendMP_CXX)
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
		"Src/Utils/Logger.ixx"
		
		# Dnn
		"Src/Dnn/Tensor.ixx"
		"Src/Dnn/TensorBuffer.ixx"
		"Src/Dnn/TensorType.ixx" 
		"Src/Dnn/TensorTraits.ixx" 
		"Src/Dnn/TensorHelpers.ixx"
		"Src/Dnn/Model.ixx" 
		"Src/Dnn/Module.ixx" 
		"Src/Dnn/Modules/LayerNorm.ixx"
		"Src/Dnn/Modules/FullyConnected.ixx" 
		 
		"Src/Dnn/Compute/DeviceType.ixx"
		"Src/Dnn/Compute/DeviceRegistry.ixx"
		"Src/Dnn/Compute/ComputeDevice.ixx"  
		"Src/Dnn/Compute/DeviceHelpers.ixx" 
		"Src/Dnn/Compute/OperationBase.ixx" 
		"Src/Dnn/Compute/OperationType.ixx"
		"Src/Dnn/Compute/UnaryOperation.ixx" 
		"Src/Dnn/Compute/BinaryOperation.ixx"
		"Src/Dnn/Compute/DeviceContext.ixx"
		"Src/Dnn/Compute/OperationRegistry.ixx"
		"Src/Dnn/Compute/Operations.ixx"  
		"Src/Dnn/Compute/ComputeResource.ixx"
		"Src/Dnn/Compute/MemoryResource.ixx"
		"Src/Dnn/Compute/MemoryResourceProperties.ixx"
		
		# Dnn Compute CPU
		"Src/Dnn/Compute/CPU/CpuDevice.ixx" 
		"Src/Dnn/Compute/CPU/Ops/CpuEncoderOp.ixx" 
		"Src/Dnn/Compute/CPU/Ops/CpuLayerNormOp.ixx" 
		"Src/Dnn/Compute/CPU/Ops/CpuAttentionOp.ixx" 
		"Src/Dnn/Compute/CPU/Ops/CpuResidualOp.ixx" 
		"Src/Dnn/Compute/CPU/Ops/CpuFullyConnectedOp.ixx"
		"Src/Dnn/Compute/CPU/Ops/CpuSoftmaxOp.ixx"
		"Src/Dnn/Compute/CPU/Ops/CpuGeluOp.ixx"
		"Src/Dnn/Compute/CPU/Ops/CpuCrossEntropyOp.ixx"  
		"Src/Dnn/Compute/CPU/Ops/CpuOperations.ixx"
		"Src/Dnn/Compute/CPU/CpuMemoryResource.ixx"
		
		# Dnn Compute CUDA
		"Src/Dnn/Compute/Cuda/CudaError.ixx"
		"Src/Dnn/Compute/Cuda/CudaHelpers.ixx"
		"Src/Dnn/Compute/Cuda/CudaDevice.ixx"
		"Src/Dnn/Compute/Cuda/CudaFullyConnectedOp.ixx"
		"Src/Dnn/Compute/Cuda/CudaOperations.ixx"
		"Src/Dnn/Compute/Cuda/CudaDeviceProps.ixx"
		"Src/Dnn/Compute/Cuda/CudaMemoryResource.ixx"
		"Src/Dnn/Compute/Cuda/CudaManagedMemoryResource.ixx"
		"Src/Dnn/Compute/Cuda/CudaPinnedMemoryResource.ixx"
		
		# Data
		"Src/Data/DataLoaderBase.ixx" 
		"Src/Dnn/Compute/CPU/CpuComputeResource.ixx"
		"Src/Dnn/Modules/Gelu.ixx"
		"Src/Dnn/Modules/MultiHeadAttention.ixx"
		"Src/Dnn/Modules/Residual.ixx"
		"Src/Dnn/Modules/Softmax.ixx"
		"Src/Dnn/Modules/Encoder.ixx"
		"Src/Dnn/Modules/Blocks/MLP.ixx"
		"Src/Dnn/Modules/Blocks/TransformerBlock.ixx"
		
		"Src/Dnn/Graph.ixx"
		"Src/Dnn/Compute/Backend.ixx"
		"Src/Dnn/Compute/Cuda/CudaComputeResource.ixx"
		"Src/Dnn/Compute/Cuda/CudaGeluOp.ixx" 
		"Src/Dnn/Compute/Cuda/CudaSoftmaxOp.ixx" 
		"Src/Dnn/Compute/Cuda/CudaLayerNormOp.ixx" 
		"Src/Dnn/Compute/Cuda/CudaEncoderOp.ixx" 
		"Src/Dnn/Compute/Cuda/CudaMultiHeadAttentionOp.ixx" 
		"Src/Dnn/ModelBase.ixx" 
		"Src/Dnn/Compute/OperationAttributes'.ixx"
		"Src/Dnn/Compute/Cuda/CudaResidualOp.ixx"
)

target_link_libraries( Mila PRIVATE CUDA::nvml CUDA::cublas CUDA::cublasLt CUDA::cudart )

# Get the current date and time
string(TIMESTAMP BUILD_DATE "%Y-%m-%d")
string(TIMESTAMP BUILD_TIME "%H:%M:%S")

# Write the version file
configure_file(
    ${CMAKE_SOURCE_DIR}/Mila/Src/Version.h.in
    ${CMAKE_BINARY_DIR}/Mila/Version.h
    @ONLY
)

# Include the generated version file
include_directories(${CMAKE_BINARY_DIR})