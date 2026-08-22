#define _USE_MATH_DEFINES
#include <math.h>
#include <cassert>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include "device_launch_parameters.h"
#include "CudaUtils.h"
#include "Rope.cuh"

namespace Mila::Dnn::Compute::Cuda::Rope
{
    // ========================================================================
    // Forward / backward kernel
    // ========================================================================

    /**
     * @brief Full-sequence RoPE rotation kernel (BF16).
     *
     * Q/K loaded and stored as BF16; cos/sin read from the shared FP32 cache.
     * All rotation arithmetic executes in FP32.
     *
     * RoPE rotation matrix:
     *   forward  R:    [c, -s;  s, c]
     *   backward R^T:  [c,  s; -s, c]  (exact inverse, R is orthogonal)
     *
     * @tparam negate_sin  false -> forward rotation, true -> backward (inverse) rotation.
     */
    template <bool negate_sin>
    __global__ void rope_rotate_bf16_kernel(
        __nv_bfloat16* __restrict__       out,
        const __nv_bfloat16* __restrict__ in,
        const float* __restrict__         cos_cache,
        const float* __restrict__         sin_cache,
        int total_heads,
        int pair_half,
        int cache_stride,
        int head_stride,
        int T,
        int n_heads,
        int position_offset )
    {
        int bth = blockIdx.x * blockDim.x + threadIdx.x;
        int i = blockIdx.y * blockDim.y + threadIdx.y;

        if ( bth >= total_heads || i >= pair_half ) return;

        int t = (bth / n_heads) % T;
        int abs_pos = t + position_offset;

        float c = cos_cache[ abs_pos * cache_stride + i ];
        float s = sin_cache[ abs_pos * cache_stride + i ];

        int base_idx = bth * head_stride;
        float x0 = __bfloat162float( in[ base_idx + i ] );
        float x1 = __bfloat162float( in[ base_idx + i + pair_half ] );

        float r0, r1;

        if constexpr ( negate_sin )
        {
            r0 = x0 * c + x1 * s;
            r1 = -x0 * s + x1 * c;
        }
        else
        {
            r0 = x0 * c - x1 * s;
            r1 = x0 * s + x1 * c;
        }

        out[ base_idx + i ] = __float2bfloat16( r0 );
        out[ base_idx + i + pair_half ] = __float2bfloat16( r1 );
    }

    /**
     * @brief Single-token decode RoPE rotation kernel (BF16).
     *
     * Q/K loaded and stored as BF16; cos/sin read from the shared FP32 cache.
     * All rotation arithmetic executes in FP32.
     *
     * @tparam negate_sin  false -> forward rotation, true -> backward (inverse) rotation.
     */
    template <bool negate_sin>
    __global__ void rope_decode_bf16_kernel(
        __nv_bfloat16* __restrict__       out,
        const __nv_bfloat16* __restrict__ in,
        const float* __restrict__         cos_cache,
        const float* __restrict__         sin_cache,
        int total_heads,
        int pair_half,
        int cache_stride,
        int head_stride,
        int position,
        int n_heads )
    {
        int bh = blockIdx.x * blockDim.x + threadIdx.x;
        int i = blockIdx.y * blockDim.y + threadIdx.y;

        if ( bh >= total_heads || i >= pair_half ) return;

        float c = cos_cache[ position * cache_stride + i ];
        float s = sin_cache[ position * cache_stride + i ];

        int base_idx = bh * head_stride;
        float x0 = __bfloat162float( in[ base_idx + i ] );
        float x1 = __bfloat162float( in[ base_idx + i + pair_half ] );

        float r0, r1;

        if constexpr ( negate_sin )
        {
            r0 = x0 * c + x1 * s;
            r1 = -x0 * s + x1 * c;
        }
        else
        {
            r0 = x0 * c - x1 * s;
            r1 = x0 * s + x1 * c;
        }

        out[ base_idx + i ] = __float2bfloat16( r0 );
        out[ base_idx + i + pair_half ] = __float2bfloat16( r1 );
    }

    // ========================================================================
    // Launch helpers
    // ========================================================================

    template <bool negate_sin>
    static void launch_rotate_full_bf16(
        __nv_bfloat16* out_Q,
        __nv_bfloat16* out_K,
        const __nv_bfloat16* in_Q,
        const __nv_bfloat16* in_K,
        const float* cos_cache,
        const float* sin_cache,
        int B, int T,
        int n_heads, int n_kv_heads, int head_dim,
        int rotary_dim, int rotary_layout,
        int position_offset,
        cudaStream_t stream )
    {
        assert( head_dim % 2 == 0 );

        // Which channel pairs actually rotate. WholeHead spans the head and lets the cache
        // carry identity beyond rotary_dim (Gemma); RotaryPrefix confines the rotation to the
        // leading rotary_dim and pairs inside it (Qwen). The cache layout is head_dim/2 wide
        // in both cases, so only the pairing offset and the bound change.
        const int cache_stride = head_dim / 2;
        const int head_stride = head_dim;
        const int pair_half = ( rotary_layout == 1 && rotary_dim > 0 && rotary_dim < head_dim )
            ? ( rotary_dim / 2 )
            : cache_stride;

        constexpr int TX = 32;
        constexpr int TY = 16;

        // --- Q ---
        {
            int total = B * T * n_heads;
            dim3 block( TX, TY );
            dim3 grid(
                (total + TX - 1) / TX,
                (pair_half + TY - 1) / TY );

            rope_rotate_bf16_kernel<negate_sin> << <grid, block, 0, stream >> > (
                out_Q, in_Q, cos_cache, sin_cache,
                total, pair_half, cache_stride, head_stride, T, n_heads, position_offset);
        }

        // --- K ---
        {
            int total = B * T * n_kv_heads;
            dim3 block( TX, TY );
            dim3 grid(
                (total + TX - 1) / TX,
                (pair_half + TY - 1) / TY );

            rope_rotate_bf16_kernel<negate_sin> << <grid, block, 0, stream >> > (
                out_K, in_K, cos_cache, sin_cache,
                total, pair_half, cache_stride, head_stride, T, n_kv_heads, position_offset);
        }

        cudaCheck( cudaGetLastError() );
    }

    template <bool negate_sin>
    static void launch_rotate_decode_bf16(
        __nv_bfloat16* out_Q,
        __nv_bfloat16* out_K,
        const __nv_bfloat16* in_Q,
        const __nv_bfloat16* in_K,
        const float* cos_cache,
        const float* sin_cache,
        int B, int position,
        int n_heads, int n_kv_heads, int head_dim,
        int rotary_dim, int rotary_layout,
        cudaStream_t stream )
    {
        assert( head_dim % 2 == 0 );

        // Which channel pairs actually rotate. WholeHead spans the head and lets the cache
        // carry identity beyond rotary_dim (Gemma); RotaryPrefix confines the rotation to the
        // leading rotary_dim and pairs inside it (Qwen). The cache layout is head_dim/2 wide
        // in both cases, so only the pairing offset and the bound change.
        const int cache_stride = head_dim / 2;
        const int head_stride = head_dim;
        const int pair_half = ( rotary_layout == 1 && rotary_dim > 0 && rotary_dim < head_dim )
            ? ( rotary_dim / 2 )
            : cache_stride;

        constexpr int TX = 32;
        constexpr int TY = 16;

        // --- Q ---
        {
            int total = B * n_heads;
            dim3 block( TX, TY );
            dim3 grid(
                (total + TX - 1) / TX,
                (pair_half + TY - 1) / TY );

            rope_decode_bf16_kernel<negate_sin> << <grid, block, 0, stream >> > (
                out_Q, in_Q, cos_cache, sin_cache,
                total, pair_half, cache_stride, head_stride, position, n_heads);
        }

        // --- K ---
        {
            int total = B * n_kv_heads;
            dim3 block( TX, TY );
            dim3 grid(
                (total + TX - 1) / TX,
                (pair_half + TY - 1) / TY );

            rope_decode_bf16_kernel<negate_sin> << <grid, block, 0, stream >> > (
                out_K, in_K, cos_cache, sin_cache,
                total, pair_half, cache_stride, head_stride, position, n_kv_heads);
        }

        cudaCheck( cudaGetLastError() );
    }

    // ========================================================================
    // Public host launchers — BF16
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
        cudaStream_t stream )
    {
        launch_rotate_full_bf16<false>(
            Q_out, K_out, Q_in, K_in,
            cos_cache, sin_cache,
            B, T, n_heads, n_kv_heads, head_dim, rotary_dim, rotary_layout, position_offset, stream );
    }

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
        cudaStream_t stream )
    {
        launch_rotate_full_bf16<true>(
            dQ_in, dK_in, dQ_out, dK_out,
            cos_cache, sin_cache,
            B, T, n_heads, n_kv_heads, head_dim, rotary_dim, rotary_layout, 0, stream );
    }

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
        cudaStream_t stream )
    {
        launch_rotate_decode_bf16<false>(
            Q_out, K_out, Q_in, K_in,
            cos_cache, sin_cache,
            B, position, n_heads, n_kv_heads, head_dim, rotary_dim, rotary_layout, stream );
    }
}