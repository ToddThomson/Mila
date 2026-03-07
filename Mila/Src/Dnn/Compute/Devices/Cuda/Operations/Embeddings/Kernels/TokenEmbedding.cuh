#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace Mila::Dnn::Compute::Cuda::TokenEmbedding
{
    void cuda_token_embedding_forward_fp32(
        float* Y, const int* X, const float* wte,
        int B, int T, int C, cudaStream_t stream );

    void cuda_token_embedding_backward_fp32(
        float* dwte, const float* dY, const int* X,
        int B, int T, int C, cudaStream_t stream );

    void cuda_token_embedding_decode_fp32(
        float* Y, const int* X, const float* wte,
        int B, int C, cudaStream_t stream );

    void cuda_token_embedding_forward_fp16(
        half* Y, const int* X, const half* wte,
        int B, int T, int C, cudaStream_t stream );

    void cuda_token_embedding_backward_fp16(
        half* dwte, const half* dY, const int* X,
        int B, int T, int C, cudaStream_t stream );

    void cuda_token_embedding_decode_fp16(
        half* Y, const int* X, const half* wte,
        int B, int C, cudaStream_t stream );
}