include(FetchContent)

# Google Benchmark
FetchContent_Declare(
    benchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG v1.9.4
)

# Disable building tests and examples for benchmark
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "")
FetchContent_MakeAvailable(benchmark)

add_executable( MilaBenchmarks
 "gelu_forward_fp32.cpp"
 "main.cpp" 
 "softmax_forward_fp32.cpp"
 "linear_forward_fp32.cpp")

target_compile_features( MilaBenchmarks PUBLIC cxx_std_23)

target_include_directories( MilaBenchmarks PRIVATE
 ${CMAKE_CURRENT_SOURCE_DIR}/../Src
)

target_link_libraries( MilaBenchmarks PRIVATE
 Mila
 benchmark::benchmark
 CUDA::cudart
 CUDA::cublas
 CUDA::cublasLt
)

set_target_properties( MilaBenchmarks PROPERTIES
 RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

install(TARGETS MilaBenchmarks RUNTIME DESTINATION bin)
