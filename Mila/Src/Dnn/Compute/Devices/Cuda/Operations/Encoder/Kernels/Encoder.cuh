#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace Mila::Dnn::Compute::Cuda::Encoder
{
    void cuda_encoder_forward_fp32(
        float* Y, const int* X,
        const float* wte, const float* wpe,
        int B, int T, int C,
        cudaStream_t stream );

    void cuda_encoder_backward_fp32(
        float* wte_grad, float* wpe_grad,
        const float* dY,
        const int* X,
        int B, int T, int C,
        cudaStream_t stream );

    void cuda_encoder_forward_fp16(
        half* Y, const int* X,
        const half* wte, const half* wpe,
        int B, int T, int C,
        cudaStream_t stream );

    void cuda_encoder_backward_fp16(
        half* wte_grad, half* wpe_grad,
        const half* dY,
        const int* X,
        int B, int T, int C,
        cudaStream_t stream );
}