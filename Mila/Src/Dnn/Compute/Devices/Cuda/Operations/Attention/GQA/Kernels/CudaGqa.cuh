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
 *   Mila::Dnn::Compute::Cuda::Gqa
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

namespace Mila::Dnn::Compute::Cuda::Gqa
{
    // ========================================================================
    // GQA Prefill — BF16
    // ========================================================================

    void cuda_gqa_prefill_softmax_bf16(
        __nv_bfloat16* att, const __nv_bfloat16* preatt,
        int B, int NH, int T_stride, int attended_len, int chunk_stride,
        int chunk_len, int position_offset, int window,
        cudaStream_t stream );

    /**
     * @brief Bounded sliding-window ring prefill softmax (BF16).
     *
     * Variant of cuda_gqa_prefill_softmax_bf16 for a KV cache stored as a ring of
     * `capacity` rows. preatt/att column j is RING SLOT j; the kernel reconstructs
     * each slot's absolute position (from end = position_offset + chunk_len - 1) to
     * apply the window + causal mask. See SlidingWindowKvCache.md D6.
     */
    void cuda_gqa_prefill_softmax_ring_bf16(
        __nv_bfloat16* att, const __nv_bfloat16* preatt,
        int B, int NH, int capacity,
        int chunk_len, int position_offset, int window,
        cudaStream_t stream );

    void cuda_gqa_prefill_unpermute_output_padded_bf16(
        const __nv_bfloat16* vaccum, __nv_bfloat16* out,
        int B, int actual_T, int padded_T,
        int NQH, int HS,
        cudaStream_t stream );

    /**
     * @brief Fused FlashAttention prefill over the compact BF16 KV cache (Iteration 1).
     *
     * Streaming, causal, online-softmax attention that replaces the cuBLASLt
     * QK -> prefill_softmax -> AV pipeline with a single kernel and zero transient
     * attention workspace. Reads Q from the projection output [B, chunk_len, NH*HS]
     * and K/V from the compact cache [B, NKV, cache_capacity, HS], writing the
     * attention output directly to Y [B, chunk_len, NH*HS] -- no Q permute, no
     * unpermute, no preatt/att/v_out materialization.
     *
     * Unbounded / global causal path (the caller routes kBounded == false here; the
     * bounded ring routes to cuda_gqa_flash_prefill_ring_bf16 below). HS must be a
     * multiple of 32 and <= 512 (Gemma 4 global_head_dim is 512, the head dim this path
     * actually runs on; Llama 128 also qualifies). `scale` is the config-derived attention
     * scale (1/sqrt(HS) for Llama, 1.0 for Gemma) -- never recomputed here. See
     * GqaFlashAttention.md.
     */
    void cuda_gqa_flash_prefill_bf16(
        const __nv_bfloat16* Q, const __nv_bfloat16* K, const __nv_bfloat16* V,
        __nv_bfloat16* Y,
        int B, int chunk_len, int NH, int NKV, int HS, int cache_capacity,
        int position_offset, int window, float scale,
        cudaStream_t stream );

    /**
     * @brief Fused FlashAttention prefill over the bounded sliding-window ring cache (BF16).
     *
     * Ring variant of cuda_gqa_flash_prefill_bf16 for the kBounded op: K/V physical row =
     * absolute position % cache_capacity (capacity = window + prefill_chunk - 1), and the
     * key-tile loop starts at the sliding band (~window keys per query tile, constant per
     * block) instead of running causal-triangular from zero. Replaces the cuBLASLt QK ->
     * prefill_softmax_ring -> AV pipeline on Gemma's local sliding layers. Requires
     * window > 0. See GqaFlashAttention.md.
     */
    void cuda_gqa_flash_prefill_ring_bf16(
        const __nv_bfloat16* Q, const __nv_bfloat16* K, const __nv_bfloat16* V,
        __nv_bfloat16* Y,
        int B, int chunk_len, int NH, int NKV, int HS, int cache_capacity,
        int position_offset, int window, float scale,
        cudaStream_t stream );

    // ========================================================================
    // GQA Decode — fused attention (BF16)
    // ========================================================================

    /// True when the fused decode kernel covers this geometry (head_size in
    /// {128, 256, 512}, group_size 1..16); the caller falls back to the
    /// cuBLASLt decode pipeline otherwise.
    bool cuda_gqa_decode_attention_supported( int head_size, int group_size );

    /// Device scratch bytes the fused decode kernel needs for its split-K
    /// partials at the worst-case split count. Fetch via
    /// ExecutionContext::getDeviceScratchBuffer at decode time.
    size_t cuda_gqa_decode_attention_scratch_bytes( int B, int NH, int HS );

    /**
     * @brief Fused single-token decode attention over the compact BF16 KV cache.
     *
     * Replaces the cuBLASLt QK -> softmax_decode -> AV pipeline (and its
     * T=1 identity Q-permute/unpermute copies) with one streaming
     * online-softmax kernel that reads only the live attention band:
     * absolute positions [max(0, actual_len - window), actual_len), physical
     * cache row = position % cache_capacity (identity when unbounded). Serves
     * both the unbounded and bounded-ring ops. Q is read from the projection
     * output [B, 1, NH*HS]; Y is written as [B, 1, NH*HS]. Long bands are
     * split-K parallelized across blocks with a fixup merge launch;
     * split_scratch must hold cuda_gqa_decode_attention_scratch_bytes.
     * `scale` is the config-derived attention scale, applied to the QK dots.
     */
    void cuda_gqa_decode_attention_bf16(
        const __nv_bfloat16* Q, const __nv_bfloat16* K, const __nv_bfloat16* V,
        __nv_bfloat16* Y, float* split_scratch,
        int B, int NH, int NKV, int HS, int cache_capacity,
        int actual_len, int window, float scale,
        cudaStream_t stream );

    // ========================================================================
    // GQA KV Cache FP32
    // ========================================================================

    /**
     * @brief Permute Q from [B, chunk_len, NH*HS] to [B, NH, T_max, HS].
     *
     * @param Q           Output Q buffer [B * NH * T_max * HS], device memory.
     * @param X           Input Q buffer  [B * chunk_len * NH * HS], device memory.
     * @param batch       Batch size.
     * @param chunk_len   Number of tokens in this prefill chunk.
     * @param NH          Number of query heads.
     * @param HS          Head dimension.
     * @param start_pos   Absolute token position of the first chunk token.
     * @param max_seq_len KV cache capacity.
     * @param stream      CUDA stream for kernel scheduling.
     */
    void cuda_gqa_kvcache_write_q_fp32(
        float* Q,
        const float* X,
        int batch, int chunk_len,
        int NH, int HS,
        int start_pos, int max_seq_len,
        cudaStream_t stream );

    /**
     * @brief Permute K and V from [B, chunk_len, NKV*HS] to [B, NKV, T_max, HS].
     *
     * K and V are permuted in a single kernel launch since they share identical
     * stride arithmetic.
     *
     * @param K           Output K buffer [B * NKV * T_max * HS], device memory.
     * @param V           Output V buffer [B * NKV * T_max * HS], device memory.
     * @param Xk          Input K buffer  [B * chunk_len * NKV * HS], device memory.
     * @param Xv          Input V buffer  [B * chunk_len * NKV * HS], device memory.
     * @param batch       Batch size.
     * @param chunk_len   Number of tokens in this prefill chunk.
     * @param NKV         Number of key/value heads.
     * @param HS          Head dimension.
     * @param start_pos   Absolute token position of the first chunk token.
     * @param max_seq_len KV cache capacity.
     * @param stream      CUDA stream for kernel scheduling.
     */
    void cuda_gqa_kvcache_write_kv_fp32(
        float* K, float* V,
        const float* Xk, const float* Xv,
        int batch, int chunk_len,
        int NKV, int HS,
        int start_pos, int max_seq_len,
        cudaStream_t stream );

    // ========================================================================
    // GQA KV Cache BF16
    // ========================================================================

    /**
     * @brief Permute Q from [B, chunk_len, NH*HS] to [B, NH, T_max, HS] (BF16).
     *
     * @param Q           Output Q buffer [B * NH * T_max * HS], device memory.
     * @param X           Input Q buffer  [B * chunk_len * NH * HS], device memory.
     * @param batch       Batch size.
     * @param chunk_len   Number of tokens in this prefill chunk.
     * @param NH          Number of query heads.
     * @param HS          Head dimension.
     * @param start_pos   Absolute token position of the first chunk token.
     * @param max_seq_len KV cache capacity.
     * @param stream      CUDA stream for kernel scheduling.
     */
    void cuda_gqa_kvcache_write_q_bf16(
        __nv_bfloat16* Q,
        const __nv_bfloat16* X,
        int batch, int chunk_len,
        int NH, int HS,
        int start_pos, int max_seq_len,
        cudaStream_t stream );

    /**
     * @brief Permute K and V from [B, chunk_len, NKV*HS] to [B, NKV, T_max, HS] (BF16).
     *
     * K and V are permuted in a single kernel launch since they share identical
     * stride arithmetic.
     *
     * @param K           Output K buffer [B * NKV * T_max * HS], device memory.
     * @param V           Output V buffer [B * NKV * T_max * HS], device memory.
     * @param Xk          Input K buffer  [B * chunk_len * NKV * HS], device memory.
     * @param Xv          Input V buffer  [B * chunk_len * NKV * HS], device memory.
     * @param batch       Batch size.
     * @param chunk_len   Number of tokens in this prefill chunk.
     * @param NKV         Number of key/value heads.
     * @param HS          Head dimension.
     * @param start_pos   Absolute token position of the first chunk token.
     * @param max_seq_len KV cache capacity.
     * @param stream      CUDA stream for kernel scheduling.
     */
    void cuda_gqa_kvcache_write_kv_bf16(
        __nv_bfloat16* K, __nv_bfloat16* V,
        const __nv_bfloat16* Xk, const __nv_bfloat16* Xv,
        int batch, int chunk_len,
        int NKV, int HS,
        int start_pos, int max_seq_len,
        cudaStream_t stream );

    void cuda_gqa_kvcache_expand_kv_bf16(
        __nv_bfloat16* k_exp, __nv_bfloat16* v_exp,
        const __nv_bfloat16* k_compact, const __nv_bfloat16* v_compact,
        int B, int chunk_len, int T_stride, int NH, int NKV, int HS,
        int position_offset,
        cudaStream_t stream );

    void cuda_gqa_kvcache_expand_kv_fp32(
        float* k_exp, float* v_exp,
        const float* k_compact, const float* v_compact,
        int B, int chunk_len, int T, int NH, int NKV, int HS, int position_offset,
        cudaStream_t stream );

    // ========================================================================
    // TODO Reorg GQA Kernels
    // ========================================================================

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

    void cuda_gqa_prefill_expand_kv_fp16(
        half* k_exp, half* v_exp,
        const half* k_compact, const half* v_compact,
        int B, int chunk_len, int T, int NH, int NKV, int HS, int position_offset,
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

    // ========================================================================
    // GQA Prefill — FP32 / FP16
    // ========================================================================

    void cuda_gqa_prefill_softmax_fp32(
        float* att, const float* preatt,
        int B, int NH, int T_stride, int attended_len, int chunk_stride,
        int chunk_len, int position_offset, int window,
        cudaStream_t stream );

    /// @copydoc cuda_gqa_prefill_softmax_ring_bf16
    void cuda_gqa_prefill_softmax_ring_fp32(
        float* att, const float* preatt,
        int B, int NH, int capacity,
        int chunk_len, int position_offset, int window,
        cudaStream_t stream );

    void cuda_gqa_prefill_softmax_fp16(
        half* att, const half* preatt,
        int B_NH, int T_stride, int chunk_stride, int chunk_len, int position_offset,
        cudaStream_t stream );


    void cuda_gqa_prefill_permute_q_fp32(
        float* Q,
        const float* X,
        int batch, int chunk_len,
        int NH, int HS,
        int start_pos, int max_seq_len,
        cudaStream_t stream );

    void cuda_gqa_prefill_permute_kv_fp32(
        float* K, float* V,
        const float* Xk, const float* Xv,
        int batch, int chunk_len,
        int NKV, int HS,
        int start_pos, int max_seq_len,
        cudaStream_t stream );

    void cuda_gqa_prefill_permute_qkv_fp16(
        half* Q, half* K, half* V,
        const float* X,
        int batch, int chunk_len,
        int num_heads, int num_kv_heads, int head_size,
        int start_pos,
        int max_seq_len,
        cudaStream_t stream );

    void cuda_gqa_prefill_expand_kv_fp32(
        float* k_exp, float* v_exp,
        const float* k_compact, const float* v_compact,
        int B,
        int chunk_len,
        int T_stride,
        int NH,
        int NKV,
        int HS,
        int position_offset,
        cudaStream_t stream );

    void cuda_gqa_prefill_expand_kv_fp16(
        half* k_exp, half* v_exp,
        const half* k_compact, const half* v_compact,
        int B,
        int chunk_len,
        int T_stride,
        int NH,
        int NKV,
        int HS,
        int position_offset,
        cudaStream_t stream );

    void cuda_gqa_prefill_unpermute_output_padded_fp32(
        const float* vaccum, float* out,
        int B, int actual_T, int padded_T,
        int NQH, int HS,
        cudaStream_t stream );

    void cuda_gqa_prefill_unpermute_output_padded_fp16(
        const half* vaccum, half* out,
        int B, int actual_T, int padded_T,
        int NQH, int HS,
        cudaStream_t stream );

    // ========================================================================
    // Optimized path — compact Q permute (Phase 1)
    //
    // Writes Q from [B, chunk_len, NH*HS] into a compact [B, NH, chunk_len, HS]
    // buffer with stride chunk_len*HS between heads. Unlike kvcache_write_q,
    // the output is NOT indexed into a T_max-row persistent cache — it writes
    // tight-packed rows starting at row 0, matching the strideA expected by
    // the NKV-layout cuBLASLt plans.
    //
    // TEMP: Remove with legacy path once the optimized path is validated.
    // ========================================================================

    /**
     * @brief Permute Q from [B, chunk_len, NH*HS] to compact [B, NH, chunk_len, HS] (FP32).
     *
     * Output stride between heads is chunk_len*HS, not T_max*HS.
     * Used exclusively by the optimized NKV-layout prefill path.
     *
     * @param Q         Output buffer [B * NH * chunk_len * HS], device memory.
     * @param X         Input Q buffer [B * chunk_len * NH * HS], device memory.
     * @param B         Batch size.
     * @param chunk_len Number of tokens in this prefill chunk (or 1 for decode).
     * @param NH        Number of query heads.
     * @param HS        Head dimension.
     * @param stream    CUDA stream.
     */
    void cuda_gqa_permute_q_compact_fp32(
        float* Q,
        const float* X,
        int B, int chunk_len,
        int NH, int HS,
        cudaStream_t stream );

    /**
     * @brief Permute Q from [B, chunk_len, NH*HS] to compact [B, NH, chunk_len, HS] (BF16).
     *
     * Output stride between heads is chunk_len*HS, not T_max*HS.
     * Used exclusively by the optimized NKV-layout prefill path.
     *
     * @param Q         Output buffer [B * NH * chunk_len * HS], device memory.
     * @param X         Input Q buffer [B * chunk_len * NH * HS], device memory.
     * @param B         Batch size.
     * @param chunk_len Number of tokens in this prefill chunk (or 1 for decode).
     * @param NH        Number of query heads.
     * @param HS        Head dimension.
     * @param stream    CUDA stream.
     */
    void cuda_gqa_permute_q_compact_bf16(
        __nv_bfloat16* Q,
        const __nv_bfloat16* X,
        int B, int chunk_len,
        int NH, int HS,
        cudaStream_t stream );
}
