#pragma once
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

// convenience function for calculating grid/block dimensions for kernels
constexpr int ceil_div(int M, int N) {
    return (M + N - 1) / N;
}

// CUDA error checking
inline void cudaCheck( cudaError_t error, const char* file, int line ) {
    if ( error != cudaSuccess ) {
        printf( "[CUDA ERROR] at file %s:%d:\n%s\n", file, line,
            cudaGetErrorString( error ) );
        exit( EXIT_FAILURE );
    }
}

inline void cudaCheck( cudaError_t error ) {
    cudaCheck(error, __FILE__, __LINE__);
}