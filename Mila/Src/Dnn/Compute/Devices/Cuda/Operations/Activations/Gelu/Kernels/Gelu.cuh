
#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace Mila::Dnn::Compute::Cuda::Gelu
{    
    void cuda_gelu_forward_fp32(
        float* Y,
        const float* X,
        int N,
        cudaStream_t stream );

    void cuda_gelu_backward_fp32(
        float* dX,
        const float* X,
        const float* dY,
        const int N,
        cudaStream_t stream );

    void cuda_gelu_forward_fp16(
        half* Y,
        const half* X,
        int N,
        cudaStream_t stream );

    void cuda_gelu_backward_fp16(
        half* dX,
        const half* X,
        const half* dY,
        const int N,
        cudaStream_t stream );
}