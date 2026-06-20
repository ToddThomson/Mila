/**
 * @file Rope.Dispatch.ixx
 * @brief CUDA kernel dispatch helpers for the Rope (rotary positional embedding) operation.
 *
 * Internal to the Compute.CudaRopeOp module. Not visible to external importers.
 */

module;
#include <cuda_bf16.h>
#include <type_traits>
#include "Kernels/Rope.cuh"

export module Compute.CudaRopeOp:Dispatch;

namespace Mila::Dnn::Compute::Cuda::Rope::Detail
{
    /**
     * @brief CUDA kernel dispatcher for RoPE forward, backward, cache build,
     *        and positional decode.
     *
     * Primary template constrained to float and half. Only the float
     * specialization is fully implemented; the half specialization follows
     * the same pattern with TODOs for FP16 kernel stubs.
     *
     * @tparam TNative CUDA native type: float (FP32) or half (FP16).
     */
    template <typename TNative>
        requires std::is_same_v<TNative, float> || std::is_same_v<TNative, __nv_bfloat16>
    struct cuda_rope_impl;

    // ========================================================================
    // FP32 specialization
    // ========================================================================

    template <>
    struct cuda_rope_impl<float>
    {
        /**
         * @brief Build the cos/sin frequency cache on the device (called once in build()).
         */
        static void build_cache(
            float* cos_cache,
            float* sin_cache,
            int    max_seq_len,
            int    head_dim,
            float  base,
            int    rotary_dim,
            cudaStream_t stream )
        {
            cuda_rope_build_cache_fp32(
                cos_cache, sin_cache,
                max_seq_len, head_dim, base, rotary_dim, stream );
        }

        /**
         * @brief Full-sequence forward: apply RoPE to Q and K with position offset.
         *
         * @param position_offset Absolute position of first token in this chunk.
         *                        Pass 0 for standard training forward passes.
         */
        static void forward(
            float* Q_out, float* K_out,
            const float* Q_in, const float* K_in,
            const float* cos_cache, const float* sin_cache,
            int B, int T,
            int n_heads, int n_kv_heads, int head_dim,
            int position_offset,
            cudaStream_t stream )
        {
            cuda_rope_forward_fp32(
                Q_out, K_out, Q_in, K_in,
                cos_cache, sin_cache,
                B, T, n_heads, n_kv_heads, head_dim, position_offset, stream );
        }

        /**
         * @brief Full-sequence backward: inverse rotation on upstream gradients.
         */
        static void backward(
            float* dQ_in, float* dK_in,
            const float* dQ_out, const float* dK_out,
            const float* cos_cache, const float* sin_cache,
            int B, int T,
            int n_heads, int n_kv_heads, int head_dim,
            cudaStream_t stream )
        {
            cuda_rope_backward_fp32(
                dQ_in, dK_in, dQ_out, dK_out,
                cos_cache, sin_cache,
                B, T, n_heads, n_kv_heads, head_dim, stream );
        }

        /**
         * @brief Single-token decode at an explicit sequence position.
         */
        static void decode(
            float* Q_out, float* K_out,
            const float* Q_in, const float* K_in,
            const float* cos_cache, const float* sin_cache,
            int B, int position,
            int n_heads, int n_kv_heads, int head_dim,
            cudaStream_t stream )
        {
            cuda_rope_decode_fp32(
                Q_out, K_out, Q_in, K_in,
                cos_cache, sin_cache,
                B, position, n_heads, n_kv_heads, head_dim, stream );
        }
    };

    // ========================================================================
    // BF16 specialization
    // ========================================================================

    template <>
    struct cuda_rope_impl<__nv_bfloat16>
    {
        // Cache is always FP32 — delegate directly to the FP32 launcher.
        static void build_cache(
            float* cos_cache,
            float* sin_cache,
            int   max_seq_len,
            int   head_dim,
            float base,
            int   rotary_dim,
            cudaStream_t stream )
        {
            // REVIEW: Why is cache building going through the dispatcher. Call directly.
            cuda_rope_build_cache_fp32(
                cos_cache, sin_cache,
                max_seq_len, head_dim, base, rotary_dim, stream );
        }

        static void forward(
            __nv_bfloat16* Q_out, __nv_bfloat16* K_out,
            const __nv_bfloat16* Q_in, const __nv_bfloat16* K_in,
            const float* cos_cache, const float* sin_cache,
            int B, int T,
            int n_heads, int n_kv_heads, int head_dim,
            int position_offset,
            cudaStream_t stream )
        {
            cuda_rope_forward_bf16(
                Q_out, K_out, Q_in, K_in,
                cos_cache, sin_cache,
                B, T, n_heads, n_kv_heads, head_dim, position_offset, stream );
        }

        static void backward(
            __nv_bfloat16* dQ_in, __nv_bfloat16* dK_in,
            const __nv_bfloat16* dQ_out, const __nv_bfloat16* dK_out,
            const float* cos_cache, const float* sin_cache,
            int B, int T,
            int n_heads, int n_kv_heads, int head_dim,
            cudaStream_t stream )
        {
            cuda_rope_backward_bf16(
                dQ_in, dK_in, dQ_out, dK_out,
                cos_cache, sin_cache,
                B, T, n_heads, n_kv_heads, head_dim, stream );
        }

        static void decode(
            __nv_bfloat16* Q_out, __nv_bfloat16* K_out,
            const __nv_bfloat16* Q_in, const __nv_bfloat16* K_in,
            const float* cos_cache, const float* sin_cache,
            int B, int position,
            int n_heads, int n_kv_heads, int head_dim,
            cudaStream_t stream )
        {
            cuda_rope_decode_bf16(
                Q_out, K_out, Q_in, K_in,
                cos_cache, sin_cache,
                B, position, n_heads, n_kv_heads, head_dim, stream );
        }
    };
}