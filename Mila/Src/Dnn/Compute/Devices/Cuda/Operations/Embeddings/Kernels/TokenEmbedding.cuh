#pragma once
#include <cuda_runtime.h>
#include <cuda_bf16.h>

namespace Mila::Dnn::Compute::Cuda::TokenEmbedding
{
    // ========================================================================
    // FP32
    // ========================================================================
    void cuda_token_embedding_forward_fp32(
        float* Y, const int* X, const float* wte,
        int B, int T, int C, cudaStream_t stream );

    void cuda_token_embedding_backward_fp32(
        float* dwte, const float* dY, const int* X,
        int B, int T, int C, cudaStream_t stream );

    void cuda_token_embedding_decode_fp32(
        float* Y, const int* X, const float* wte,
        int B, int C, cudaStream_t stream );

    // ========================================================================
    // BF16
    // ========================================================================
    void cuda_token_embedding_forward_bf16(
        __nv_bfloat16* Y, const int* X, const __nv_bfloat16* wte,
        int B, int T, int C, cudaStream_t stream );

    void cuda_token_embedding_backward_bf16(
        __nv_bfloat16* dwte, const __nv_bfloat16* dY, const int* X,
        int B, int T, int C, cudaStream_t stream );

    void cuda_token_embedding_decode_bf16(
        __nv_bfloat16* Y, const int* X, const __nv_bfloat16* wte,
        int B, int C, cudaStream_t stream );
}