/**
 * @file CudaAttentionCommon.cuh
 * @brief Shared CUDA kernel declarations for attention operations.
 *
 * Contains the subset of attention kernels whose signatures and semantics are
 * identical across Multi-Head Attention (MHA) and Grouped-Query Attention (GQA):
 *
 *   Softmax family
 *     cuda_attention_softmax_forward          — causal softmax, training
 *     cuda_attention_softmax_padded_forward   — causal softmax, prefill (padded batch)
 *     cuda_attention_softmax_decode_forward   — non-causal softmax, single decode step
 *     cuda_attention_softmax_backward         — softmax backward, training
 *
 *   Unpermute family
 *     cuda_attention_unpermute_output         — [B,NH,T,HS] → [B,T,C]
 *     cuda_attention_unpermute_output_padded  — [B,NH,padT,HS] → [B,actT,C]
 *     cuda_attention_unpermute_backward       — [B,T,C] → [B,NH,T,HS]  (grad scatter)
 *
 * All functions live in:
 *   Mila::Dnn::Compute::Cuda::Attention::Common
 *
 * MHA- and GQA-specific permute kernels (which differ in their QKV trailing
 * dimension and head counts) are declared in their respective op headers
 * (CudaMha.cuh / CudaGqa.cuh) and are NOT included here.
 *
 * Layout convention (row-major throughout)
 * ─────────────────────────────────────────
 *   Per-head tensors : [B, NH, T, HS]   — innermost dim is HS
 *   Output / grad   : [B, T,  C]        — C = NH * HS
 *   Attention matrix: [B*NH, T, T]      — row = query position
 */

#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace Mila::Dnn::Compute::Cuda::Attention::Common
{
    // ========================================================================
    // Softmax — FP32
    // ========================================================================

    /**
     * @brief Causal scaled softmax forward (FP32, training).
     *
     * Each thread handles one attention row (query position t).
     * Position t attends to positions 0..t; future positions are zeroed.
     *
     * @param att     Output attention weights  [B*NH, T, T].
     * @param scale   Scalar multiplier applied before exp (typically 1.0f).
     * @param preatt  Input pre-softmax scores  [B*NH, T, T].
     * @param B       Batch size.
     * @param NH      Number of attention heads.
     * @param T       Sequence length.
     * @param stream  CUDA stream.
     */
    void cuda_attention_softmax_forward_fp32(
        float* att, float scale, const float* preatt,
        int B, int NH, int T,
        cudaStream_t stream );

    /**
     * @brief Causal scaled softmax forward with padding (FP32, prefill).
     *
     * Identical to the plain variant but masks out rows at positions
     * >= actual_T (padding rows are zeroed, not skipped, so that the output
     * buffer is fully initialised).
     *
     * @param att      Output attention weights  [B*NH, max_T, max_T].
     * @param scale    Scalar multiplier.
     * @param preatt   Input pre-softmax scores  [B*NH, max_T, max_T].
     * @param B        Batch size.
     * @param NH       Number of attention heads.
     * @param max_T    Padded sequence length (buffer size).
     * @param actual_T Valid token count.
     * @param stream   CUDA stream.
     */
    void cuda_attention_softmax_padded_forward_fp32(
        float* att, float scale, const float* preatt,
        int B, int NH, int max_T, int actual_T,
        cudaStream_t stream );

    /**
     * @brief Non-causal softmax for single-token decode (FP32).
     *
     * One thread per (b, nh) pair; attends over all actual_len cached
     * positions without causal masking (the query is the final position).
     *
     * @param att        Output attention weights  [B*NH, 1, max_len].
     * @param scale      Scalar multiplier.
     * @param preatt     Input pre-softmax scores  [B*NH, 1, max_len].
     * @param B          Batch size.
     * @param NH         Number of attention heads.
     * @param max_len    Allocated cache length.
     * @param actual_len Number of valid cached tokens.
     * @param stream     CUDA stream.
     */
    void cuda_attention_softmax_decode_forward_fp32(
        float* att, float scale, const float* preatt,
        int B, int NH, int max_len, int actual_len,
        cudaStream_t stream );

    /**
     * @brief Softmax backward pass (FP32).
     *
     * Computes dPreatt from dAtt and the saved Att weights.
     * Only the causal (lower-triangular) region is written; upper triangle
     * is zeroed.
     *
     * @param dpreatt  Output gradient  [B*NH, T, T].
     * @param datt     Input gradient   [B*NH, T, T].
     * @param att      Saved forward attention weights [B*NH, T, T].
     * @param scale    Same scale factor used in the forward pass.
     * @param B        Batch size.
     * @param NH       Number of attention heads.
     * @param T        Sequence length.
     * @param stream   CUDA stream.
     */
    void cuda_attention_softmax_backward_fp32(
        float* dpreatt, const float* datt, const float* att,
        float scale,
        int B, int NH, int T,
        cudaStream_t stream );

    // ========================================================================
    // Softmax — FP16
    // ========================================================================

    /// @copydoc cuda_attention_softmax_forward_fp32
    void cuda_attention_softmax_forward_fp16(
        half* att, float scale, const half* preatt,
        int B, int NH, int T,
        cudaStream_t stream );

    /// @copydoc cuda_attention_softmax_padded_forward_fp32
    void cuda_attention_softmax_padded_forward_fp16(
        half* att, float scale, const half* preatt,
        int B, int NH, int max_T, int actual_T,
        cudaStream_t stream );

    /// @copydoc cuda_attention_softmax_decode_forward_fp32
    void cuda_attention_softmax_decode_forward_fp16(
        half* att, float scale, const half* preatt,
        int B, int NH, int max_len, int actual_len,
        cudaStream_t stream );

    /// @copydoc cuda_attention_softmax_backward_fp32
    void cuda_attention_softmax_backward_fp16(
        half* dpreatt, const half* datt, const half* att,
        float scale,
        int B, int NH, int T,
        cudaStream_t stream );

    // ========================================================================
    // Unpermute — FP32
    // ========================================================================

    /**
     * @brief Reorder per-head accumulator into concatenated output (FP32).
     *
     * Reads vaccum [B, NH, T, HS] and writes out [B, T, C] where C = NH * HS.
     *
     * @param vaccum  Input  [B, NH, T, HS].
     * @param out     Output [B, T, C].
     * @param B       Batch size.
     * @param T       Sequence length.
     * @param NH      Number of attention heads.
     * @param HS      Head size.
     * @param stream  CUDA stream.
     */
    void cuda_attention_unpermute_output_fp32(
        const float* vaccum, float* out,
        int B, int T, int NH, int HS,
        cudaStream_t stream );

    /**
     * @brief Reorder per-head accumulator into compact output, discarding padding (FP32).
     *
     * Source buffer vaccum is sized at [B, NH, padded_T, HS]; only the first
     * actual_T token rows are written to out [B, actual_T, C].
     *
     * @param vaccum    Input  [B, NH, padded_T, HS].
     * @param out       Output [B, actual_T, C].
     * @param B         Batch size.
     * @param actual_T  Valid token count to emit.
     * @param padded_T  Padded sequence length (stride in vaccum).
     * @param NH        Number of attention heads.
     * @param HS        Head size.
     * @param stream    CUDA stream.
     */
    void cuda_attention_unpermute_output_padded_fp32(
        const float* vaccum, float* out,
        int B, int actual_T, int padded_T, int NH, int HS,
        cudaStream_t stream );

    /**
     * @brief Scatter output gradient dout into per-head dvaccum (FP32, backward).
     *
     * Inverse of unpermute_output: reads dout [B, T, C] and writes
     * dvaccum [B, NH, T, HS].
     *
     * @param dvaccum  Output gradient [B, NH, T, HS].
     * @param dout     Input gradient  [B, T, C].
     * @param B        Batch size.
     * @param T        Sequence length.
     * @param NH       Number of attention heads.
     * @param HS       Head size.
     * @param stream   CUDA stream.
     */
    void cuda_attention_unpermute_backward_fp32(
        float* dvaccum, const float* dout,
        int B, int T, int NH, int HS,
        cudaStream_t stream );

    // ========================================================================
    // Unpermute — FP16
    // ========================================================================

    /// @copydoc cuda_attention_unpermute_output_fp32
    void cuda_attention_unpermute_output_fp16(
        const half* vaccum, half* out,
        int B, int T, int NH, int HS,
        cudaStream_t stream );

    /// @copydoc cuda_attention_unpermute_output_padded_fp32
    void cuda_attention_unpermute_output_padded_fp16(
        const half* vaccum, half* out,
        int B, int actual_T, int padded_T, int NH, int HS,
        cudaStream_t stream );

    /// @copydoc cuda_attention_unpermute_backward_fp32
    void cuda_attention_unpermute_backward_fp16(
        half* dvaccum, const half* dout,
        int B, int T, int NH, int HS,
        cudaStream_t stream );

}
