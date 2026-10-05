#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

namespace Mila::Dnn::Compute::Cuda::Rope
{
    // ========================================================================
    // Angles
    // ========================================================================

    /**
     * @brief Everything the angle of (position, pair) depends on besides those two.
     *
     * Every rotation calculates its cos and sin from these through one device function (Rope.Angle.cuh).
     */
    struct RopeAngleParameters
    {
        float base;
        int rope_pairs;             ///< pairs that rotate; the rest carry zero frequency (identity)
        int frequency_denominator;  ///< the width the frequency spectrum spans: head_dim, or rotary_dim for RotaryPrefix
        float scaling_factor;
        float scaling_low_frequency_factor;
        float scaling_high_frequency_factor;
        int scaling_original_context_length;  ///< 0 = no frequency scaling
    };

    /**
     * @brief The angle parameters of one RoPE configuration.
     *
     * rotary_dim 0 (or >= head_dim) rotates every pair. A positive rotary_dim < head_dim rotates the first
     * rotary_dim/2 pairs. WholeHead (layout 0) spreads the spectrum across the whole head and keeps its first
     * rope_pairs frequencies -- the proportional form, Gemma's global layers. RotaryPrefix (layout 1) compresses
     * the same spectrum into rotary_dim, as `compute_default_rope_parameters` does for Qwen; at rotary_dim 64 of
     * head_dim 256 the two differ by ~29000x at the last rotated pair, so the choice is not a rounding matter.
     */
    inline RopeAngleParameters makeRopeAngleParameters(
        int head_dim, int rotary_dim, int rotary_layout, float base,
        float scaling_factor, float scaling_low_frequency_factor, float scaling_high_frequency_factor,
        int scaling_original_context_length )
    {
        const bool partial = rotary_dim > 0 && rotary_dim < head_dim;

        return RopeAngleParameters{
            base,
            partial ? rotary_dim / 2 : head_dim / 2,
            ( rotary_layout == 1 && partial ) ? rotary_dim : head_dim,
            scaling_factor,
            scaling_low_frequency_factor,
            scaling_high_frequency_factor,
            scaling_original_context_length };
    }

    // ========================================================================
    // Forward — full sequence with position offset
    // ========================================================================

    /**
     * @brief Apply RoPE to Q and K for a (possibly offset) sequence chunk.
     *
     * Each token at chunk-local position t is rotated at absolute position t + position_offset, so successive
     * prefill chunks pass increasing offsets. For standard training/forward passes, pass position_offset = 0.
     *
     * @param Q_out           Output Q [B, T, n_heads,    head_dim].
     * @param K_out           Output K [B, T, n_kv_heads, head_dim].
     * @param Q_in            Input  Q [B, T, n_heads,    head_dim].
     * @param K_in            Input  K [B, T, n_kv_heads, head_dim].
     * @param angles          The configuration's angle parameters (makeRopeAngleParameters).
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
        const RopeAngleParameters& angles,
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
     * @param angles     The configuration's angle parameters.
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
        const RopeAngleParameters& angles,
        int B, int T,
        int n_heads, int n_kv_heads, int head_dim,
        int rotary_dim, int rotary_layout,
        cudaStream_t stream );

    // ========================================================================
    // Decode — single token, explicit position
    // ========================================================================

    /**
     * @brief Apply RoPE for a decode step of T tokens in a row from an explicit sequence position.
     *
     * The position is read on the device, so a recorded decode step replays at every position; token t rotates at
     * *position + t. One token is the ordinary decode step; a few are a multi-token decode (Gemma4Mtp.md 4.7).
     *
     * @param Q_out      Output Q [B, T, n_heads,    head_dim].
     * @param K_out      Output K [B, T, n_kv_heads, head_dim].
     * @param Q_in       Input  Q [B, T, n_heads,    head_dim].
     * @param K_in       Input  K [B, T, n_kv_heads, head_dim].
     * @param angles     The configuration's angle parameters.
     * @param B          Batch size.
     * @param T          Tokens per sequence.
     * @param position   Device int holding the absolute sequence position of the first token.
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
        const RopeAngleParameters& angles,
        int B, int T, const int* position,
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
        const RopeAngleParameters& angles,
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
        const RopeAngleParameters& angles,
        int B, int T,
        int n_heads, int n_kv_heads, int head_dim,
        int rotary_dim, int rotary_layout,
        cudaStream_t stream );

    void cuda_rope_decode_bf16(
        __nv_bfloat16* Q_out,
        __nv_bfloat16* K_out,
        const __nv_bfloat16* Q_in,
        const __nv_bfloat16* K_in,
        const RopeAngleParameters& angles,
        int B, int T, const int* position,
        int n_heads, int n_kv_heads, int head_dim,
        int rotary_dim, int rotary_layout,
        cudaStream_t stream );
}
