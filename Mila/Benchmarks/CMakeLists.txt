# CMake configuration file for building the Mila benchmark executable.

# Find the nlohmann/json package
include(FetchContent)
FetchContent_Declare(
    json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.12.0
)
FetchContent_MakeAvailable(json)

add_executable(mila_benchmark 
    "Src/main.cpp"
    #"C:/Users/ToddT/Src/Repos/Mila/Mila/Src/Dnn/Compute/Operations/Cuda/Kernels/CudaGelu.cu"
)

target_sources(mila_benchmark
     PUBLIC
       FILE_SET module_files
       TYPE CXX_MODULES FILES
            "Src/Benchmark.ixx"
            "Src/ModuleBenchmark.ixx"
            "Src/OperationBenchmark.ixx"
            "Src/KernelBenchmark.ixx"
            "Src/BlockModuleBenchmark.ixx"
            "Src/BenchmarkManager.ixx"
            "Src/BinaryModuleBenchmark.ixx"
            "Src/BenchmarkDefinitions.ixx"
)

target_include_directories(mila_benchmark PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/../Src
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(mila_benchmark PRIVATE 
    Mila
    CUDA::nvml CUDA::cublas CUDA::cublasLt CUDA::cudart
    nlohmann_json::nlohmann_json
)

set_target_properties(mila_benchmark PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES}
)

install(TARGETS mila_benchmark
    RUNTIME DESTINATION bin
)
