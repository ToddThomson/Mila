#pragma once
#include <cuda_runtime.h>

void cuda_gelu_forward(
    float* out,
    const float* inp,
    int N ); //, cudaStream_t stream );
