/**
 * @file CudaGqa.cuh
 * @brief CUDA kernel declarations for Grouped-Query Attention operations.
 *
 * Declares the GQA-specific kernel host functions implemented in
 * CudaGqa.Permute.cu.  Softmax and unpermute functions shared with MHA are
 * declared separately in Common/Kernels/CudaAttentionCommon.cuh.
 *
 * Function naming convention
 * ──────────────────────────
 *   cuda_gqa_<operation>_fp32 / _fp16
 *
 * All functions are in namespace:
 *   Mila::Dnn::Compute::Cuda::GroupedQueryAttention
 *
 * Template wrapper
 * ────────────────
 * The cuda_gqa_kernels<T> struct provides a type-dispatched interface so that
 * CudaGqaOp.ixx can call kernels without fp32/fp16 conditional branches:
 *
 *   Detail::cuda_gqa_kernels<NativeType>::permute_qkv( ... )
 *
 * Layout conventions
 * ──────────────────
 *   Input X        : [B, T,    (NH + 2*NKV) * HS]
 *   Q output       : [B, NH,   T, HS]
 *   K, V (compact) : [B, NKV,  T, HS]
 *   k_exp, v_exp   : [B, NH,   T, HS]   (after expand_kv)
 */

#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace Mila::Dnn::Compute::Cuda::GroupedQueryAttention
{
    // ========================================================================
    // QKV permute — FP32
    // ========================================================================

    /**
     * @brief Split packed QKV [B,T,(NH+2*NKV)*HS] into Q[B,NH,T,HS],
     *        K[B,NKV,T,HS], V[B,NKV,T,HS] (FP32, training).
     */
    void cuda_gqa_permute_qkv_fp32(
        float* Q, float* K, float* V,
        const float* X,
        int B, int T, int NH, int NKV, int HS,
        cudaStream_t stream );

    /**
     * @brief Padded variant: pads output to output_T, zeroing positions
     *        >= input_T (FP32, prefill).
     */
    void cuda_gqa_permute_qkv_padded_fp32(
        float* Q, float* K, float* V,
        const float* X,
        int B, int input_T, int output_T, int NH, int NKV, int HS,
        cudaStream_t stream );

    /**
     * @brief Single-token decode: writes Q at position and appends K/V to
     *        cache at the same position (FP32).
     */
    void cuda_gqa_permute_qkv_decode_fp32(
        float* Q, float* K, float* V,
        const float* X,
        int B, int position, int cache_T, int NH, int NKV, int HS,
        cudaStream_t stream );

    // ========================================================================
    // QKV permute — FP16
    // ========================================================================

    /// @copydoc cuda_gqa_permute_qkv_fp32
    void cuda_gqa_permute_qkv_fp16(
        half* Q, half* K, half* V,
        const half* X,
        int B, int T, int NH, int NKV, int HS,
        cudaStream_t stream );

    /// @copydoc cuda_gqa_permute_qkv_padded_fp32
    void cuda_gqa_permute_qkv_padded_fp16(
        half* Q, half* K, half* V,
        const half* X,
        int B, int input_T, int output_T, int NH, int NKV, int HS,
        cudaStream_t stream );

    /// @copydoc cuda_gqa_permute_qkv_decode_fp32
    void cuda_gqa_permute_qkv_decode_fp16(
        half* Q, half* K, half* V,
        const half* X,
        int B, int position, int cache_T, int NH, int NKV, int HS,
        cudaStream_t stream );

    // ========================================================================
    // KV expansion — FP32 / FP16
    // ========================================================================

    /**
     * @brief Broadcast K/V from compact [B,NKV,T,HS] to expanded [B,NH,T,HS].
     *
     * Each Q-head nh reads from KV-head (nh / (NH/NKV)).
     * Must be called before any cuBLASLt attention matmul.
     *
     * @param k_exp     Output expanded K buffer [B, NH,  T, HS].
     * @param v_exp     Output expanded V buffer [B, NH,  T, HS].
     * @param k_compact Input compact  K buffer [B, NKV, T, HS].
     * @param v_compact Input compact  V buffer [B, NKV, T, HS].
     * @param B         Batch size.
     * @param T         Sequence length (or actual_len during decode).
     * @param NH        Total number of Q heads.
     * @param NKV       Number of KV heads (must divide NH).
     * @param HS        Head size.
     * @param stream    CUDA stream.
     */
    void cuda_gqa_expand_kv_fp32(
        float* k_exp, float* v_exp,
        const float* k_compact, const float* v_compact,
        int B, int T, int NH, int NKV, int HS,
        cudaStream_t stream );

    /// @copydoc cuda_gqa_expand_kv_fp32
    void cuda_gqa_expand_kv_fp16(
        half* k_exp, half* v_exp,
        const half* k_compact, const half* v_compact,
        int B, int T, int NH, int NKV, int HS,
        cudaStream_t stream );

    // ========================================================================
    // KV gradient reduction — FP32 / FP16
    // ========================================================================

    /**
     * @brief Reduce expanded KV gradients [B,NH,T,HS] → compact [B,NKV,T,HS].
     *
     * Sums the GS = NH/NKV gradient contributions from each Q-head group
     * back into the corresponding KV-head gradient.  Called during backward
     * after the cuBLASLt dK/dV matmuls produce the expanded-layout gradients.
     *
     * @param dk_compact  Output dK [B, NKV, T, HS].
     * @param dv_compact  Output dV [B, NKV, T, HS].
     * @param dk_exp      Input  dK [B, NH,  T, HS] (expanded).
     * @param dv_exp      Input  dV [B, NH,  T, HS] (expanded).
     * @param B           Batch size.
     * @param T           Sequence length.
     * @param NH          Total number of Q heads.
     * @param NKV         Number of KV heads.
     * @param HS          Head size.
     * @param stream      CUDA stream.
     */
    void cuda_gqa_reduce_kv_grad_fp32(
        float* dk_compact, float* dv_compact,
        const float* dk_exp, const float* dv_exp,
        int B, int T, int NH, int NKV, int HS,
        cudaStream_t stream );

    /// @copydoc cuda_gqa_reduce_kv_grad_fp32
    void cuda_gqa_reduce_kv_grad_fp16(
        half* dk_compact, half* dv_compact,
        const half* dk_exp, const half* dv_exp,
        int B, int T, int NH, int NKV, int HS,
        cudaStream_t stream );

    // ========================================================================
    // Permute backward — FP32 / FP16
    // ========================================================================

    /**
     * @brief Pack per-head gradients back into concatenated dX (FP32).
     *
     * Writes:
     *   dQ [B, NH,  T, HS] → dX[b, t, nh*HS + hs]
     *   dK [B, NKV, T, HS] → dX[b, t, (NH+nkv)*HS + hs]
     *   dV [B, NKV, T, HS] → dX[b, t, (NH+NKV+nkv)*HS + hs]
     *
     * @param dX  Output gradient [B, T, (NH+2*NKV)*HS].
     * @param dQ  Q gradient      [B, NH,  T, HS].
     * @param dK  K gradient      [B, NKV, T, HS].
     * @param dV  V gradient      [B, NKV, T, HS].
     */
    void cuda_gqa_permute_backward_fp32(
        float* dX,
        const float* dQ, const float* dK, const float* dV,
        int B, int T, int NH, int NKV, int HS,
        cudaStream_t stream );

    /// @copydoc cuda_gqa_permute_backward_fp32
    void cuda_gqa_permute_backward_fp16(
        half* dX,
        const half* dQ, const half* dK, const half* dV,
        int B, int T, int NH, int NKV, int HS,
        cudaStream_t stream );
}
