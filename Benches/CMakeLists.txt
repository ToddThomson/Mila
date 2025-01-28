# CMakeList.txt : CMake project for Mila examples

include( ../cmake/CPM.cmake)
# CPMAddPackage("gh:NVIDIA/cccl#main")
CPMAddPackage("gh:NVIDIA/nvbench#main")

# Add executable
add_executable(MilaBench layernorm_forward.cu )

target_link_libraries(MilaBench Thrust::Thrust CUDA::cublas CUDA::cublasLt nvbench::main)
# target_compile_options(MilaBench PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math --extended-lambda>)
