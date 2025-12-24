#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace Mila::Dnn::Compute::Cuda::LayerNorm
{
    void cuda_layernorm_forward_fp32(
        float* Y,
        float* mean, float* rstd,
        const float* X,
        const float* weight, const float* bias,
        int B, int T, int C, float epsilon,
        cudaStream_t stream );

    void cuda_layernorm_forward_fp16(
        half* Y,
        half* mean, half* rstd,
        const half* X,
        const half* weight, const half* bias,
        int B, int T, int C, float epsilon,
        cudaStream_t stream );

    // Softmax functions
    template <typename TPrecision>
    void cuda_softmax_forward(
        TPrecision* Y,
        const TPrecision* X,
        int N,
        int C,
        cudaStream_t stream );

    template <typename TPrecision>
    void cuda_softmax_forward_general(
        TPrecision* Y,
        const TPrecision* X,
        int outer_size,
        int dim_size,
        int inner_size,
        cudaStream_t stream );
}