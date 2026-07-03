/**
 * @file CudaTokenEmbeddingOp.Dispatch.ixx
 * @brief CUDA kernel dispatch helpers for the TokenEmbedding operation.
 *
 * Internal to the Compute.CudaTokenEmbeddingOp module.
 */

module;
#include <cuda_fp16.h>
#include <type_traits>
#include "Kernels/TokenEmbedding.cuh"

export module Compute.CudaTokenEmbeddingOp:Dispatch;

namespace Mila::Dnn::Compute::Cuda::TokenEmbedding::Detail
{
    template <typename TNative>
        requires std::is_same_v<TNative, float> || std::is_same_v<TNative, __nv_bfloat16>
    struct cuda_token_embedding_impl;

    // ========================================================================
    // FP32
    // ========================================================================

    template <>
    struct cuda_token_embedding_impl<float>
    {
        static void forward(
            float* Y, const int* X, const float* wte,
            int B, int T, int C, cudaStream_t stream )
        {
            cuda_token_embedding_forward_fp32( Y, X, wte, B, T, C, stream );
        }

        static void backward(
            float* dwte, const float* dY, const int* X,
            int B, int T, int C, cudaStream_t stream )
        {
            cuda_token_embedding_backward_fp32( dwte, dY, X, B, T, C, stream );
        }

        static void decode(
            float* Y, const int* X, const float* wte,
            int B, int C, cudaStream_t stream )
        {
            cuda_token_embedding_decode_fp32( Y, X, wte, B, C, stream );
        }
    };

    // ========================================================================
    // BF16
    // ========================================================================

    template <>
    struct cuda_token_embedding_impl<__nv_bfloat16>
    {
        static void forward(
            __nv_bfloat16* Y, const int* X, const __nv_bfloat16* wte,
            int B, int T, int C, cudaStream_t stream )
        {
            cuda_token_embedding_forward_bf16( Y, X, wte, B, T, C, stream );
        }

        static void backward(
            __nv_bfloat16* dwte, const __nv_bfloat16* dY, const int* X,
            int B, int T, int C, cudaStream_t stream )
        {
            cuda_token_embedding_backward_bf16( dwte, dY, X, B, T, C, stream );
        }

        static void decode(
            __nv_bfloat16* Y, const int* X, const __nv_bfloat16* wte,
            int B, int C, cudaStream_t stream )
        {
            cuda_token_embedding_decode_bf16( Y, X, wte, B, C, stream );
        }
    };
}