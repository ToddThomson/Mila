/**
 * @file Lpe.Dispatch.ixx
 * @brief CUDA kernel dispatch helpers for the Lpe (token + positional embedding) operation.
 *
 * Internal to the Compute.CudaLpeOp module. Not visible to external importers.
 */

module;
#include <cuda_fp16.h>
#include <type_traits>
#include "Kernels/Lpe.cuh"

export module Compute.CudaLpeOp:Dispatch;

namespace Mila::Dnn::Compute::Cuda::Lpe::Detail
{
    /**
     * @brief CUDA kernel dispatcher for Lpe forward, backward, and positional decode.
     *
     * Primary template is constrained to float and half; no other specializations
     * are defined.
     *
     * @tparam TNative CUDA native type: float (FP32) or half (FP16).
     */
    template <typename TNative>
        requires std::is_same_v<TNative, float> || std::is_same_v<TNative, half>
    struct cuda_lpe_impl;

    /**
     * @brief FP32 specialization of the Lpe CUDA kernel dispatcher.
     */
    template <>
    struct cuda_lpe_impl<float>
    {
        /**
         * @brief Full-sequence forward pass: output[b,t,:] = wte[X[b,t],:] + wpe[t,:].
         *
         * @param Y   Output embeddings [B, T, C].
         * @param X   Input token indices [B, T] (INT32).
         * @param wte Token embedding table [vocab_size, C].
         * @param wpe Positional embedding table [max_seq_len, C].
         * @param B   Batch size.
         * @param T   Sequence length.
         * @param C   Embedding dimension.
         */
        static void forward(
            float* Y, const int32_t* X,
            const float* wte, const float* wpe,
            int B, int T, int C,
            cudaStream_t stream )
        {
            cuda_encoder_forward_fp32( Y, X, wte, wpe, B, T, C, stream );
        }

        /**
         * @brief Backward pass accumulating gradients into wte and wpe.
         *
         * @param dwte Gradient accumulation buffer for wte [vocab_size, C].
         * @param dwpe Gradient accumulation buffer for wpe [max_seq_len, C].
         * @param X    Input token indices used in the forward pass [B, T].
         * @param dY   Upstream gradient [B, T, C].
         * @param B    Batch size.
         * @param T    Sequence length.
         * @param C    Embedding dimension.
         */
        static void backward(
            float* dwte, float* dwpe,
            const int32_t* X, const float* dY,
            int B, int T, int C,
            cudaStream_t stream )
        {
            cuda_encoder_backward_fp32( dwte, dwpe, dY, X, B, T, C, stream );
        }

        /**
         * @brief Single-token decode at an explicit sequence position.
         *
         * Computes output[b,:] = wte[X[b],:] + wpe[position,:] for each batch
         * element. Delegates to cuda_encoder_decode_fp32 which reads only the
         * single wpe row at `position` — no sequence iteration overhead.
         *
         * @param Y        Output embeddings [B, C].
         * @param X        Input token indices [B] (INT32).
         * @param wte      Token embedding table [vocab_size, C].
         * @param wpe      Positional embedding table [max_seq_len, C].
         * @param B        Batch size.
         * @param position Absolute sequence position for the wpe row lookup.
         * @param C        Embedding dimension.
         */
        static void decode(
            float* Y, const int32_t* X,
            const float* wte, const float* wpe,
            int B, int position, int C,
            cudaStream_t stream )
        {
            cuda_encoder_decode_fp32( Y, X, wte, wpe, B, position, C, stream );
        }
    };

    /**
     * @brief FP16 specialization of the Lpe CUDA kernel dispatcher.
     */
    template <>
    struct cuda_lpe_impl<half>
    {
        /**
         * @brief Full-sequence forward pass: output[b,t,:] = wte[X[b,t],:] + wpe[t,:].
         *
         * @param Y   Output embeddings [B, T, C].
         * @param X   Input token indices [B, T] (INT32).
         * @param wte Token embedding table [vocab_size, C].
         * @param wpe Positional embedding table [max_seq_len, C].
         * @param B   Batch size.
         * @param T   Sequence length.
         * @param C   Embedding dimension.
         */
        static void forward(
            half* Y, const int32_t* X,
            const half* wte, const half* wpe,
            int B, int T, int C,
            cudaStream_t stream )
        {
            cuda_encoder_forward_fp16( Y, X, wte, wpe, B, T, C, stream );
        }

        /**
         * @brief Backward pass accumulating gradients into wte and wpe.
         *
         * @param dwte Gradient accumulation buffer for wte [vocab_size, C].
         * @param dwpe Gradient accumulation buffer for wpe [max_seq_len, C].
         * @param X    Input token indices used in the forward pass [B, T].
         * @param dY   Upstream gradient [B, T, C].
         * @param B    Batch size.
         * @param T    Sequence length.
         * @param C    Embedding dimension.
         */
        static void backward(
            half* dwte, half* dwpe,
            const int32_t* X, const half* dY,
            int B, int T, int C,
            cudaStream_t stream )
        {
            // TODO: cuda_encoder_backward_fp16( dwte, dwpe, dY, X, B, T, C, stream );
        }

        /**
         * @brief Single-token decode at an explicit sequence position.
         *
         * @param Y        Output embeddings [B, C].
         * @param X        Input token indices [B] (INT32).
         * @param wte      Token embedding table [vocab_size, C].
         * @param wpe      Positional embedding table [max_seq_len, C].
         * @param B        Batch size.
         * @param position Absolute sequence position for the wpe row lookup.
         * @param C        Embedding dimension.
         */
        static void decode(
            half* Y, const int32_t* X,
            const half* wte, const half* wpe,
            int B, int position, int C,
            cudaStream_t stream )
        {
            // TODO: cuda_encoder_decode_fp16( Y, X, wte, wpe, B, position, C, stream );
        }
    };
}