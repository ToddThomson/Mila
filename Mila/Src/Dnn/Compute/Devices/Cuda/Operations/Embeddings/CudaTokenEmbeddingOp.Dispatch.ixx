/**
 * @file TokenEmbedding.Dispatch.ixx
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
        requires std::is_same_v<TNative, float> || std::is_same_v<TNative, half>
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
    // FP16 (stubs)
    // ========================================================================

    template <>
    struct cuda_token_embedding_impl<half>
    {
        static void forward(
            half* Y, const int* X, const half* wte,
            int B, int T, int C, cudaStream_t stream )
        {
            // TODO: cuda_token_embedding_forward_fp16(...)
        }

        static void backward(
            half* dwte, const half* dY, const int* X,
            int B, int T, int C, cudaStream_t stream )
        {
            // TODO: cuda_token_embedding_backward_fp16(...)
        }

        static void decode(
            half* Y, const int* X, const half* wte,
            int B, int C, cudaStream_t stream )
        {
            // TODO: cuda_token_embedding_decode_fp16(...)
        }
    };
}