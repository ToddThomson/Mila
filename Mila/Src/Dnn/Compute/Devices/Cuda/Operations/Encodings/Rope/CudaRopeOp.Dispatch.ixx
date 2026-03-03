/**
 * @file Rope.Dispatch.ixx
 * @brief CUDA kernel dispatch helpers for the Rope (rotary positional embedding) operation.
 *
 * Internal to the Compute.CudaRopeOp module. Not visible to external importers.
 */

module;
#include <cuda_fp16.h>
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
     * the same pattern as LPE with TODOs for FP16 kernel stubs.
     *
     * @tparam TNative CUDA native type: float (FP32) or half (FP16).
     */
    template <typename TNative>
        requires std::is_same_v<TNative, float> || std::is_same_v<TNative, half>
    struct cuda_rope_impl;

    // ========================================================================
    // FP32 specialization
    // ========================================================================

    template <>
    struct cuda_rope_impl<float>
    {
        /**
         * @brief Build the cos/sin frequency cache on the device (called once in build()).
         *
         * @param cos_cache  Device buffer [max_seq_len, head_dim/2].
         * @param sin_cache  Device buffer [max_seq_len, head_dim/2].
         * @param max_seq_len  Maximum sequence length.
         * @param head_dim   Per-head embedding dimension.
         * @param base       Frequency base (default 10000.0f).
         * @param stream     CUDA stream.
         */
        static void build_cache(
            float* cos_cache,
            float* sin_cache,
            int    max_seq_len,
            int    head_dim,
            float  base,
            cudaStream_t stream )
        {
            cuda_rope_build_cache_fp32(
                cos_cache, sin_cache,
                max_seq_len, head_dim, base, stream );
        }

        /**
         * @brief Full-sequence forward: apply RoPE to Q and K.
         *
         * @param Q_out      Rotated Q  [B, T, n_heads,    head_dim].
         * @param K_out      Rotated K  [B, T, n_kv_heads, head_dim].
         * @param Q_in       Input Q    [B, T, n_heads,    head_dim].
         * @param K_in       Input K    [B, T, n_kv_heads, head_dim].
         * @param cos_cache  [max_seq_len, head_dim/2].
         * @param sin_cache  [max_seq_len, head_dim/2].
         */
        static void forward(
            float* Q_out, float* K_out,
            const float* Q_in, const float* K_in,
            const float* cos_cache, const float* sin_cache,
            int B, int T,
            int n_heads, int n_kv_heads, int head_dim,
            cudaStream_t stream )
        {
            cuda_rope_forward_fp32(
                Q_out, K_out, Q_in, K_in,
                cos_cache, sin_cache,
                B, T, n_heads, n_kv_heads, head_dim, stream );
        }

        /**
         * @brief Full-sequence backward: inverse rotation on upstream gradients.
         *
         * @param dQ_in   Gradient w.r.t. Q input  [B, T, n_heads,    head_dim].
         * @param dK_in   Gradient w.r.t. K input  [B, T, n_kv_heads, head_dim].
         * @param dQ_out  Upstream Q gradient       [B, T, n_heads,    head_dim].
         * @param dK_out  Upstream K gradient       [B, T, n_kv_heads, head_dim].
         * @param cos_cache  [max_seq_len, head_dim/2].
         * @param sin_cache  [max_seq_len, head_dim/2].
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
         *
         * @param Q_out      Rotated Q  [B, 1, n_heads,    head_dim].
         * @param K_out      Rotated K  [B, 1, n_kv_heads, head_dim].
         * @param Q_in       Input Q    [B, 1, n_heads,    head_dim].
         * @param K_in       Input K    [B, 1, n_kv_heads, head_dim].
         * @param cos_cache  [max_seq_len, head_dim/2].
         * @param sin_cache  [max_seq_len, head_dim/2].
         * @param position   Absolute sequence position for the cache row lookup.
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
    // FP16 specialization (stubs — mirrors LPE pattern)
    // ========================================================================

    template <>
    struct cuda_rope_impl<half>
    {
        static void build_cache(
            half* cos_cache,
            half* sin_cache,
            int    max_seq_len,
            int    head_dim,
            float  base,
            cudaStream_t stream )
        {
            // TODO: cuda_rope_build_cache_fp16(...)
        }

        static void forward(
            half* Q_out, half* K_out,
            const half* Q_in, const half* K_in,
            const half* cos_cache, const half* sin_cache,
            int B, int T,
            int n_heads, int n_kv_heads, int head_dim,
            cudaStream_t stream )
        {
            // TODO: cuda_rope_forward_fp16(...)
        }

        static void backward(
            half* dQ_in, half* dK_in,
            const half* dQ_out, const half* dK_out,
            const half* cos_cache, const half* sin_cache,
            int B, int T,
            int n_heads, int n_kv_heads, int head_dim,
            cudaStream_t stream )
        {
            // TODO: cuda_rope_backward_fp16(...)
        }

        static void decode(
            half* Q_out, half* K_out,
            const half* Q_in, const half* K_in,
            const half* cos_cache, const half* sin_cache,
            int B, int position,
            int n_heads, int n_kv_heads, int head_dim,
            cudaStream_t stream )
        {
            // TODO: cuda_rope_decode_fp16(...)
        }
    };
}
