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
     * Fills cos_cache[pos, i] = cos(pos * θ_i) and
     *       sin_cache[pos, i] = sin(pos * θ_i)
     * for pos in [0, max_seq_len) and i in [0, head_dim/2).
     * θ_i = base^(-2i / head_dim), base = 10000.
     *
     * @param cos_cache  Device buffer [max_seq_len, head_dim/2].
     * @param sin_cache  Device buffer [max_seq_len, head_dim/2].
     * @param max_seq_len Maximum sequence length.
     * @param head_dim   Per-head embedding dimension (must be even).
     * @param base       Frequency base (default 10000.0f).
     * @param stream     CUDA stream.
     */
    void cuda_rope_build_cache_fp32(
        float* cos_cache,
        float* sin_cache,
        int    max_seq_len,
        int    head_dim,
        float  base,
        cudaStream_t stream );

    // ========================================================================
    // Forward — full sequence
    // ========================================================================

    /**
     * @brief Apply RoPE to Q and K for a full sequence (training / prefill).
     *
     * Input layout:  [B, T, n_heads,   head_dim] for Q
     *                [B, T, n_kv_heads, head_dim] for K
     * Output layout: same shape, rotated in-place or to separate buffers.
     *
     * @param Q_out      Output Q [B, T, n_heads,    head_dim].
     * @param K_out      Output K [B, T, n_kv_heads, head_dim].
     * @param Q_in       Input  Q [B, T, n_heads,    head_dim].
     * @param K_in       Input  K [B, T, n_kv_heads, head_dim].
     * @param cos_cache  Precomputed cosines [max_seq_len, head_dim/2].
     * @param sin_cache  Precomputed sines   [max_seq_len, head_dim/2].
     * @param B          Batch size.
     * @param T          Sequence length.
     * @param n_heads    Number of query heads.
     * @param n_kv_heads Number of key/value heads (GQA: n_kv_heads <= n_heads).
     * @param head_dim   Per-head dimension (must be divisible by 2).
     * @param stream     CUDA stream.
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
        cudaStream_t stream );

    // ========================================================================
    // Backward — full sequence
    // ========================================================================

    /**
     * @brief Backward pass for RoPE (full sequence).
     *
     * RoPE is an orthogonal rotation, so the backward pass is the inverse
     * rotation: negate the sin terms (rotate by -θ).
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
        cudaStream_t stream );
}
