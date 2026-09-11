#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace Mila::Dnn::Compute::Cuda::Rope
{
    // ========================================================================
    // Cache construction
    // ========================================================================

    /**
     * @brief Build the cos/sin frequency cache on the device.
     *
     * Fills cos_cache[pos, i] = cos(pos * theta_i) and
     *       sin_cache[pos, i] = sin(pos * theta_i)
     * for pos in [0, max_seq_len) and i in [0, head_dim/2).
     *
     * @param cos_cache  Device buffer [max_seq_len, head_dim/2].
     * @param sin_cache  Device buffer [max_seq_len, head_dim/2].
     * @param max_seq_len Maximum sequence length.
     * @param head_dim   Per-head embedding dimension (must be even).
     * @param base       Frequency base (default 10000.0f).
     * @param rotary_dim Number of dimensions to rotate; 0 (or >= head_dim) = full
     *                   rotation (default). A positive value < head_dim rotates only
     *                   the first rotary_dim dims (proportional partial-rotary); the
     *                   remainder get zero frequency (identity / pass-through).
     * @param stream     CUDA stream.
     */
    void cuda_rope_build_cache_fp32(
        float* cos_cache,
        float* sin_cache,
        int    max_seq_len,
        int    head_dim,
        float  base,
        int    rotary_dim,
        int    rotary_layout,
        cudaStream_t stream );

    // ========================================================================
    // Forward — full sequence with position offset
    // ========================================================================

    /**
     * @brief Apply RoPE to Q and K for a (possibly offset) sequence chunk.
     *
     * Each token at chunk-local position t is rotated using the cache row
     * at absolute position (t + position_offset). This enables chunked prefill
     * where successive chunks use increasing offsets.
     *
     * For standard training/forward passes, pass position_offset = 0.
     *
     * @param Q_out           Output Q [B, T, n_heads,    head_dim].
     * @param K_out           Output K [B, T, n_kv_heads, head_dim].
     * @param Q_in            Input  Q [B, T, n_heads,    head_dim].
     * @param K_in            Input  K [B, T, n_kv_heads, head_dim].
     * @param cos_cache       Precomputed cosines [max_seq_len, head_dim/2].
     * @param sin_cache       Precomputed sines   [max_seq_len, head_dim/2].
     * @param B               Batch size.
     * @param T               Sequence length of this chunk.
     * @param n_heads         Number of query heads.
     * @param n_kv_heads      Number of key/value heads (GQA: n_kv_heads <= n_heads).
     * @param head_dim        Per-head dimension (must be divisible by 2).
     * @param position_offset Absolute position of the first token in this chunk.
     * @param stream          CUDA stream.
     */
    void cuda_rope_forward_fp32(
        float* Q_out,
        float* K_out,
        const float* Q_in,
        const float* K_in,
        const float* cos_cache,
        const float* sin_cache,
        int B, int T,
        int n_heads, int n_kv_heads, int head_dim,
        int rotary_dim, int rotary_layout,
        int position_offset,
        cudaStream_t stream );

    // ========================================================================
    // Backward — full sequence
    // ========================================================================

    /**
     * @brief Backward pass for RoPE (full sequence).
     *
     * RoPE is an orthogonal rotation, so the backward pass is the inverse
     * rotation: negate the sin terms (rotate by -theta). Position offset is
     * always 0 because backward is only used during training.
     *
     * @param dQ_in      Output gradient w.r.t. Q input  [B, T, n_heads,    head_dim].
     * @param dK_in      Output gradient w.r.t. K input  [B, T, n_kv_heads, head_dim].
     * @param dQ_out     Upstream gradient for Q output  [B, T, n_heads,    head_dim].
     * @param dK_out     Upstream gradient for K output  [B, T, n_kv_heads, head_dim].
     * @param cos_cache  Precomputed cosines [max_seq_len, head_dim/2].
     * @param sin_cache  Precomputed sines   [max_seq_len, head_dim/2].
     * @param B          Batch size.
     * @param T          Sequence length.
     * @param n_heads    Number of query heads.
     * @param n_kv_heads Number of key/value heads.
     * @param head_dim   Per-head dimension (must be divisible by 2).
     * @param stream     CUDA stream.
     */
    void cuda_rope_backward_fp32(
        float* dQ_in,
        float* dK_in,
        const float* dQ_out,
        const float* dK_out,
        const float* cos_cache,
        const float* sin_cache,
        int B, int T,
        int n_heads, int n_kv_heads, int head_dim,
        int rotary_dim, int rotary_layout,
        cudaStream_t stream );

    // ========================================================================
    // Decode — single token, explicit position
    // ========================================================================

    /**
     * @brief Apply RoPE for a single decode step at an explicit sequence position.
     *
     * Reads only the single cache row at `position`. Intended for KV-cache
     * autoregressive generation where T=1.
     *
     * @param Q_out      Output Q [B, 1, n_heads,    head_dim].
     * @param K_out      Output K [B, 1, n_kv_heads, head_dim].
     * @param Q_in       Input  Q [B, 1, n_heads,    head_dim].
     * @param K_in       Input  K [B, 1, n_kv_heads, head_dim].
     * @param cos_cache  Precomputed cosines [max_seq_len, head_dim/2].
     * @param sin_cache  Precomputed sines   [max_seq_len, head_dim/2].
     * @param B          Batch size.
     * @param position   Absolute sequence position (selects cache row).
     * @param n_heads    Number of query heads.
     * @param n_kv_heads Number of key/value heads.
     * @param head_dim   Per-head dimension (must be divisible by 2).
     * @param stream     CUDA stream.
     */
    void cuda_rope_decode_fp32(
        float* Q_out,
        float* K_out,
        const float* Q_in,
        const float* K_in,
        const float* cos_cache,
        const float* sin_cache,
        int B, int position,
        int n_heads, int n_kv_heads, int head_dim,
        int rotary_dim, int rotary_layout,
        cudaStream_t stream );

    // ========================================================================
    // BF16
    // ========================================================================

    void cuda_rope_forward_bf16(
        __nv_bfloat16* Q_out,
        __nv_bfloat16* K_out,
        const __nv_bfloat16* Q_in,
        const __nv_bfloat16* K_in,
        const float* cos_cache,
        const float* sin_cache,
        int B, int T,
        int n_heads, int n_kv_heads, int head_dim,
        int rotary_dim, int rotary_layout,
        int position_offset,
        cudaStream_t stream );

    void cuda_rope_backward_bf16(
        __nv_bfloat16* dQ_in,
        __nv_bfloat16* dK_in,
        const __nv_bfloat16* dQ_out,
        const __nv_bfloat16* dK_out,
        const float* cos_cache,
        const float* sin_cache,
        int B, int T,
        int n_heads, int n_kv_heads, int head_dim,
        int rotary_dim, int rotary_layout,
        cudaStream_t stream );

    void cuda_rope_decode_bf16(
        __nv_bfloat16* Q_out,
        __nv_bfloat16* K_out,
        const __nv_bfloat16* Q_in,
        const __nv_bfloat16* K_in,
        const float* cos_cache,
        const float* sin_cache,
        int B, int position,
        int n_heads, int n_kv_heads, int head_dim,
        int rotary_dim, int rotary_layout,
        cudaStream_t stream );
}