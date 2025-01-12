# CMake project for Mila Library

add_library( Mila STATIC )

set_target_properties( Mila PROPERTIES
	CXX_STANDARD_REQUIRED ON
	#CXX_MODULE_STD ON
	CXX_EXTENSIONS OFF )

# TODO: Add globally 
target_compile_features( Mila PUBLIC cxx_std_23 )

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
		"Src/Version.ixx"
		"Src/Mila.ixx"
		"Src/Misc/Glob.ixx"
		"Src/Dnn/Tensor.ixx"
		"Src/Dnn/TensorType.ixx" 
		"Src/Dnn/TensorBuffer.ixx"  
		"Src/Dnn/TensorHelpers.ixx"
		"Src/Dnn/Compute/Operations.ixx"
		"Src/Dnn/Compute/DeviceFactory.ixx"
		"Src/Dnn/Compute/DeviceRegistry.ixx"
		#"Src/Dnn/Session.ixx" 
		"Src/Dnn/Compute/DeviceInterface.ixx"  
		"Src/Dnn/Compute/Cuda/CudaDevice.ixx" 
		"Src/Dnn/Compute/CPU/CpuDevice.ixx" 
		"Src/Dnn/Compute/CPU/Ops/Encoder.ixx" 
		"Src/Dnn/Compute/CPU/Ops/LayerNorm.ixx" 
		"Src/Dnn/Compute/CPU/Ops/Attention.ixx" 
		"Src/Dnn/Compute/CPU/Ops/Residual.ixx" 
		"Src/Dnn/Compute/CPU/Ops/MatMul.ixx"
		"Src/Dnn/Compute/CPU/Ops/Softmax.ixx"
		"Src/Dnn/Compute/CPU/Ops/Gelu.ixx"
		"Src/Dnn/Compute/CPU/Ops/CrossEntropy.ixx"  
		"Src/Helpers/Memory.ixx" 
		"Src/Dnn/TensorOp.ixx" 
		"Src/Utils/Logger.ixx"
		"Src/Dnn/Model.ixx" 
		"Src/Data/DataLoaderBase.ixx" 
		"Src/Dnn/TensorTraits.ixx" 
		"Src/Dnn/Module.ixx" 
		#"Src/Dnn/Modules/LayerNorm.ixx"
)

target_link_libraries( Mila PRIVATE Thrust::Thrust CUDA::nvml CUDA::cublas CUDA::cublasLt CUDA::cudart )
