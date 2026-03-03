#define _USE_MATH_DEFINES
#include <math.h>
#include <cassert>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "device_launch_parameters.h"
#include "CudaUtils.h"
#include "Rope.cuh"

namespace Mila::Dnn::Compute::Cuda::Rope
{
    // ========================================================================
    // Cache construction kernel
    // ========================================================================

    /**
     * @brief Builds the cos/sin frequency cache.
     *
     * Each thread handles one (position, freq_pair) cell.
     * Grid: [max_seq_len, head_dim/2] threads via 2D launch.
     *
     * θ_i = 1 / (base ^ (2i / head_dim))
     *
     * @param cos_out  [max_seq_len, head_dim/2]
     * @param sin_out  [max_seq_len, head_dim/2]
     * @param half_dim head_dim / 2
     * @param base     Frequency base (10000.0f standard)
     */
    __global__ void rope_build_cache_kernel(
        float* __restrict__ cos_out,
        float* __restrict__ sin_out,
        int half_dim,
        int max_seq_len,
        float base )
    {
        int pos = blockIdx.x * blockDim.x + threadIdx.x;  // position index
        int i = blockIdx.y * blockDim.y + threadIdx.y;  // frequency pair index

        if ( pos >= max_seq_len || i >= half_dim ) return;

        // θ_i = base^(-2i / head_dim) = 1 / base^(2i / head_dim)
        float theta = __powf( base, -2.0f * static_cast<float>(i) / static_cast<float>(half_dim * 2) );
        float angle = static_cast<float>(pos) * theta;

        int idx = pos * half_dim + i;
        cos_out[ idx ] = cosf( angle );
        sin_out[ idx ] = sinf( angle );
    }

    // ========================================================================
    // Core rotation helper — used by forward, backward, and decode kernels
    // ========================================================================

    /**
     * @brief Rotate a single float2 pair by (cos_val, sin_val).
     *
     * Forward:  x' = (x0*c - x1*s,  x0*s + x1*c)
     * Backward: x' = (x0*c + x1*s, -x0*s + x1*c)  (negate_sin = true)
     *
     * @tparam negate_sin Set true for the backward (inverse) rotation.
     */
    template <bool negate_sin = false>
    __device__ __forceinline__ float2 rotate_pair( float2 x, float cos_val, float sin_val )
    {
        if constexpr ( negate_sin )
        {
            // Inverse rotation: multiply by R^T = R(-θ)
            return make_float2(
                x.x * cos_val + x.y * sin_val,
                -x.x * sin_val + x.y * cos_val );
        }
        else
        {
            return make_float2(
                x.x * cos_val - x.y * sin_val,
                x.x * sin_val + x.y * cos_val );
        }
    }

    // ========================================================================
    // Forward / backward kernel  (shared template, backward = negate_sin)
    // ========================================================================

    /**
     * @brief Full-sequence RoPE rotation kernel.
     *
     * One thread per (b, t, h, i) where i is a frequency-pair index in [0, head_dim/2).
     * Grid flattens (B * T * n_heads) onto blockIdx.x for Q, and a separate
     * launch handles K with n_kv_heads.
     *
     * Layout: input[b, t, head, 2i .. 2i+1]
     * Strides (row-major): [T*n_heads*head_dim, n_heads*head_dim, head_dim, 1]
     *
     * @tparam negate_sin  false → forward rotation, true → backward (inverse) rotation.
     *
     * @param out       Output tensor (same shape as in).
     * @param in        Input tensor.
     * @param cos_cache [max_seq_len, head_dim/2].
     * @param sin_cache [max_seq_len, head_dim/2].
     * @param total_heads  B * T * n_heads  (or B * T * n_kv_heads for K).
     * @param half_dim  head_dim / 2.
     * @param T         Sequence length (needed to recover t from linear index).
     * @param n_heads   Number of heads for this tensor (Q or K).
     */
    template <bool negate_sin>
    __global__ void rope_rotate_kernel(
        float* __restrict__       out,
        const float* __restrict__ in,
        const float* __restrict__ cos_cache,
        const float* __restrict__ sin_cache,
        int total_heads,
        int half_dim,
        int T,
        int n_heads )
    {
        // Each thread handles one (b*T*h, i) pair
        int bth = blockIdx.x * blockDim.x + threadIdx.x;   // flattened batch/time/head
        int i = blockIdx.y * blockDim.y + threadIdx.y;   // freq pair index [0, half_dim)

        if ( bth >= total_heads || i >= half_dim ) return;

        // Recover sequence position t from flattened index
        // bth = b * T * n_heads + t * n_heads + h
        int t = (bth / n_heads) % T;

        // Load precomputed cos/sin for this (position, frequency pair)
        float c = cos_cache[ t * half_dim + i ];
        float s = sin_cache[ t * half_dim + i ];

        // Address the two elements that form this rotation pair
        int base_idx = bth * (half_dim * 2) + i * 2;

        float2 x = make_float2( in[ base_idx ], in[ base_idx + 1 ] );
        float2 y = rotate_pair<negate_sin>( x, c, s );

        out[ base_idx ] = y.x;
        out[ base_idx + 1 ] = y.y;
    }

    /**
     * @brief Single-token decode kernel.
     *
     * Identical to rope_rotate_kernel but position is fixed (not derived from
     * the index), so the cache lookup always hits the same row. T=1 is implicit.
     *
     * @tparam negate_sin  Forward or backward rotation (decode is always forward).
     */
    template <bool negate_sin>
    __global__ void rope_decode_kernel(
        float* __restrict__       out,
        const float* __restrict__ in,
        const float* __restrict__ cos_cache,
        const float* __restrict__ sin_cache,
        int total_heads,   // B * n_heads  (or B * n_kv_heads)
        int half_dim,
        int position,
        int n_heads )
    {
        int bh = blockIdx.x * blockDim.x + threadIdx.x;   // flattened batch/head
        int i = blockIdx.y * blockDim.y + threadIdx.y;   // freq pair index

        if ( bh >= total_heads || i >= half_dim ) return;

        // All threads share the same position row — L1 broadcasts this load
        float c = cos_cache[ position * half_dim + i ];
        float s = sin_cache[ position * half_dim + i ];

        int base_idx = bh * (half_dim * 2) + i * 2;

        float2 x = make_float2( in[ base_idx ], in[ base_idx + 1 ] );
        float2 y = rotate_pair<negate_sin>( x, c, s );

        out[ base_idx ] = y.x;
        out[ base_idx + 1 ] = y.y;
    }

    // ========================================================================
    // Launch helpers
    // ========================================================================

    /**
     * @brief Shared kernel launcher for both Q and K with (potentially) different
     *        head counts.  Used by forward, backward, and decode host functions.
     *
     * @tparam negate_sin  Forward or backward rotation.
     */
    template <bool negate_sin>
    static void launch_rotate_full(
        float* out_Q,
        float* out_K,
        const float* in_Q,
        const float* in_K,
        const float* cos_cache,
        const float* sin_cache,
        int B, int T,
        int n_heads, int n_kv_heads, int head_dim,
        cudaStream_t stream )
    {
        assert( head_dim % 2 == 0 );
        const int half_dim = head_dim / 2;

        // 2D grid: x = flattened (b,t,h) tiles, y = freq pair tiles
        constexpr int TX = 32;
        constexpr int TY = 16;

        // --- Q ---
        {
            int total = B * T * n_heads;
            dim3 block( TX, TY );
            dim3 grid(
                (total + TX - 1) / TX,
                (half_dim + TY - 1) / TY );

            rope_rotate_kernel<negate_sin> << <grid, block, 0, stream >> > (
                out_Q, in_Q, cos_cache, sin_cache,
                total, half_dim, T, n_heads);
        }

        // --- K ---
        {
            int total = B * T * n_kv_heads;
            dim3 block( TX, TY );
            dim3 grid(
                (total + TX - 1) / TX,
                (half_dim + TY - 1) / TY );

            rope_rotate_kernel<negate_sin> << <grid, block, 0, stream >> > (
                out_K, in_K, cos_cache, sin_cache,
                total, half_dim, T, n_kv_heads);
        }

        cudaCheck( cudaGetLastError() );
    }

    template <bool negate_sin>
    static void launch_rotate_decode(
        float* out_Q,
        float* out_K,
        const float* in_Q,
        const float* in_K,
        const float* cos_cache,
        const float* sin_cache,
        int B, int position,
        int n_heads, int n_kv_heads, int head_dim,
        cudaStream_t stream )
    {
        assert( head_dim % 2 == 0 );
        const int half_dim = head_dim / 2;

        constexpr int TX = 32;
        constexpr int TY = 16;

        // --- Q ---
        {
            int total = B * n_heads;
            dim3 block( TX, TY );
            dim3 grid(
                (total + TX - 1) / TX,
                (half_dim + TY - 1) / TY );

            rope_decode_kernel<negate_sin> << <grid, block, 0, stream >> > (
                out_Q, in_Q, cos_cache, sin_cache,
                total, half_dim, position, n_heads);
        }

        // --- K ---
        {
            int total = B * n_kv_heads;
            dim3 block( TX, TY );
            dim3 grid(
                (total + TX - 1) / TX,
                (half_dim + TY - 1) / TY );

            rope_decode_kernel<negate_sin> << <grid, block, 0, stream >> > (
                out_K, in_K, cos_cache, sin_cache,
                total, half_dim, position, n_kv_heads);
        }

        cudaCheck( cudaGetLastError() );
    }

    // ========================================================================
    // Public host launchers
    // ========================================================================

    void cuda_rope_build_cache_fp32(
        float* cos_cache,
        float* sin_cache,
        int    max_seq_len,
        int    head_dim,
        float  base,
        cudaStream_t stream )
    {
        assert( head_dim % 2 == 0 );
        const int half_dim = head_dim / 2;

        constexpr int TX = 32;
        constexpr int TY = 16;

        dim3 block( TX, TY );
        dim3 grid(
            (max_seq_len + TX - 1) / TX,
            (half_dim + TY - 1) / TY );

        rope_build_cache_kernel << <grid, block, 0, stream >> > (
            cos_cache, sin_cache, half_dim, max_seq_len, base);

        cudaCheck( cudaGetLastError() );
    }

    void cuda_rope_forward_fp32(
        float* Q_out,
        float* K_out,
        const float* Q_in,
        const float* K_in,
        const float* cos_cache,
        const float* sin_cache,
        int B, int T,
        int n_heads, int n_kv_heads, int head_dim,
        cudaStream_t stream )
    {
        launch_rotate_full<false>(
            Q_out, K_out, Q_in, K_in,
            cos_cache, sin_cache,
            B, T, n_heads, n_kv_heads, head_dim, stream );
    }

    void cuda_rope_backward_fp32(
        float* dQ_in,
        float* dK_in,
        const float* dQ_out,
        const float* dK_out,
        const float* cos_cache,
        const float* sin_cache,
        int B, int T,
        int n_heads, int n_kv_heads, int head_dim,
        cudaStream_t stream )
    {
        // Backward through an orthogonal rotation is just the transpose (inverse),
        // which means negating the sin terms: rotate by -θ.
        launch_rotate_full<true>(
            dQ_in, dK_in, dQ_out, dK_out,
            cos_cache, sin_cache,
            B, T, n_heads, n_kv_heads, head_dim, stream );
    }

    void cuda_rope_decode_fp32(
        float* Q_out,
        float* K_out,
        const float* Q_in,
        const float* K_in,
        const float* cos_cache,
        const float* sin_cache,
        int B, int position,
        int n_heads, int n_kv_heads, int head_dim,
        cudaStream_t stream )
    {
        launch_rotate_decode<false>(
            Q_out, K_out, Q_in, K_in,
            cos_cache, sin_cache,
            B, position, n_heads, n_kv_heads, head_dim, stream );
    }
}
