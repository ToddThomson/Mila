/**
 * @file CudaRopeOp.Dispatch.ixx
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
    /// The angle parameters, named inside the module so the op can hold them.
    using AngleParameters = Mila::Dnn::Compute::Cuda::Rope::RopeAngleParameters;

    inline AngleParameters angle_parameters(
        int head_dim, int rotary_dim, int rotary_layout, float base,
        float scaling_factor, float scaling_low_frequency_factor, float scaling_high_frequency_factor,
        int scaling_original_context_length )
    {
        return makeRopeAngleParameters( head_dim, rotary_dim, rotary_layout, base,
            scaling_factor, scaling_low_frequency_factor, scaling_high_frequency_factor, scaling_original_context_length );
    }

    /**
     * @brief CUDA kernel dispatcher for RoPE forward, backward and positional decode.
     *
     * @tparam TNative CUDA native type: float (FP32) or __nv_bfloat16 (BF16).
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
         * @brief Full-sequence forward: apply RoPE to Q and K with position offset.
         *
         * @param position_offset Absolute position of first token in this chunk.
         *                        Pass 0 for standard training forward passes.
         */
        static void forward(
            float* Q_out, float* K_out,
            const float* Q_in, const float* K_in,
            const RopeAngleParameters& angles,
            int B, int T,
            int n_heads, int n_kv_heads, int head_dim,
            int rotary_dim, int rotary_layout,
            int position_offset,
            cudaStream_t stream )
        {
            cuda_rope_forward_fp32(
                Q_out, K_out, Q_in, K_in, angles,
                B, T, n_heads, n_kv_heads, head_dim, rotary_dim, rotary_layout, position_offset, stream );
        }

        /**
         * @brief Full-sequence backward: inverse rotation on upstream gradients.
         */
        static void backward(
            float* dQ_in, float* dK_in,
            const float* dQ_out, const float* dK_out,
            const RopeAngleParameters& angles,
            int B, int T,
            int n_heads, int n_kv_heads, int head_dim,
            int rotary_dim, int rotary_layout,
            cudaStream_t stream )
        {
            cuda_rope_backward_fp32(
                dQ_in, dK_in, dQ_out, dK_out, angles,
                B, T, n_heads, n_kv_heads, head_dim, rotary_dim, rotary_layout, stream );
        }

        /**
         * @brief Single-token decode at an explicit sequence position.
         */
        static void decode(
            float* Q_out, float* K_out,
            const float* Q_in, const float* K_in,
            const RopeAngleParameters& angles,
            int B, const int* position,
            int n_heads, int n_kv_heads, int head_dim,
            int rotary_dim, int rotary_layout,
            cudaStream_t stream )
        {
            cuda_rope_decode_fp32(
                Q_out, K_out, Q_in, K_in, angles,
                B, position, n_heads, n_kv_heads, head_dim, rotary_dim, rotary_layout, stream );
        }
    };

    // ========================================================================
    // BF16 specialization
    // ========================================================================

    template <>
    struct cuda_rope_impl<__nv_bfloat16>
    {
        static void forward(
            __nv_bfloat16* Q_out, __nv_bfloat16* K_out,
            const __nv_bfloat16* Q_in, const __nv_bfloat16* K_in,
            const RopeAngleParameters& angles,
            int B, int T,
            int n_heads, int n_kv_heads, int head_dim,
            int rotary_dim, int rotary_layout,
            int position_offset,
            cudaStream_t stream )
        {
            cuda_rope_forward_bf16(
                Q_out, K_out, Q_in, K_in, angles,
                B, T, n_heads, n_kv_heads, head_dim, rotary_dim, rotary_layout, position_offset, stream );
        }

        static void backward(
            __nv_bfloat16* dQ_in, __nv_bfloat16* dK_in,
            const __nv_bfloat16* dQ_out, const __nv_bfloat16* dK_out,
            const RopeAngleParameters& angles,
            int B, int T,
            int n_heads, int n_kv_heads, int head_dim,
            int rotary_dim, int rotary_layout,
            cudaStream_t stream )
        {
            cuda_rope_backward_bf16(
                dQ_in, dK_in, dQ_out, dK_out, angles,
                B, T, n_heads, n_kv_heads, head_dim, rotary_dim, rotary_layout, stream );
        }

        static void decode(
            __nv_bfloat16* Q_out, __nv_bfloat16* K_out,
            const __nv_bfloat16* Q_in, const __nv_bfloat16* K_in,
            const RopeAngleParameters& angles,
            int B, const int* position,
            int n_heads, int n_kv_heads, int head_dim,
            int rotary_dim, int rotary_layout,
            cudaStream_t stream )
        {
            cuda_rope_decode_bf16(
                Q_out, K_out, Q_in, K_in, angles,
                B, position, n_heads, n_kv_heads, head_dim, rotary_dim, rotary_layout, stream );
        }
    };
}
