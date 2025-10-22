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

add_executable(mila_gbench
 "gelu_forward_fp32.cpp"
 "main.cpp")

target_compile_features(mila_gbench PUBLIC cxx_std_23)

target_include_directories(mila_gbench PRIVATE
 ${CMAKE_CURRENT_SOURCE_DIR}/../Src
)

target_link_libraries(mila_gbench PRIVATE
 Mila
 benchmark::benchmark
 CUDA::cudart
 CUDA::cublas
 CUDA::cublasLt
)

set_target_properties(mila_gbench PROPERTIES
 RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

install(TARGETS mila_gbench RUNTIME DESTINATION bin)
