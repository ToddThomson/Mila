#pragma once
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include "CudaUtils.h"
#include "Rope.cuh"
#include "Rope.Angle.cuh"

namespace Mila::Dnn::Compute::Cuda::Rope
{
    __device__ __forceinline__ float rope_load( float value )
    {
        return value;
    }

    __device__ __forceinline__ float rope_load( __nv_bfloat16 value )
    {
        return __bfloat162float( value );
    }

    template <typename TElement>
    __device__ __forceinline__ TElement rope_store( float value );

    template <>
    __device__ __forceinline__ float rope_store<float>( float value )
    {
        return value;
    }

    template <>
    __device__ __forceinline__ __nv_bfloat16 rope_store<__nv_bfloat16>( float value )
    {
        return __float2bfloat16( value );
    }

    /// Pairs one block covers along x; its rows (y) share those pairs' angles across the token's heads.
    constexpr int kRotationPairsPerBlock = 32;
    constexpr int kRotationHeadRows = 8;

    /**
     * @brief RoPE with each angle calculated once per (token, pair) and applied to every Q and K head of the token.
     *
     * The angle depends on the position and the pair, never on the head, so the block's first row calculates the
     * cos and sin of its pairs into shared memory and every row then rotates heads with them: the trigonometry is
     * done once per (token, pair) rather than once per head. Each block is one token and one span of pairs; its rows
     * stride over the token's Q heads and then its K heads, so Q and K take one launch.
     *
     * Input and output may alias (every caller rotates in place), so neither is declared restrict.
     *
     * @tparam negate_sin  false -> forward rotation, true -> backward (inverse) rotation.
     * @param position     Device position for a decode step (every token at it), or null for a prefill chunk,
     *                     whose token t sits at t mod T + position_offset.
     */
    template <bool negate_sin, typename TElement>
    __global__ void rope_rotate_kernel(
        TElement* out_q,
        const TElement* in_q,
        TElement* out_k,
        const TElement* in_k,
        RopeAngleParameters angles,
        int T,
        int n_heads,
        int n_kv_heads,
        int head_dim,
        int pair_half,
        int position_offset,
        const int* __restrict__ position )
    {
        __shared__ float shared_cos[ kRotationPairsPerBlock ];
        __shared__ float shared_sin[ kRotationPairsPerBlock ];

        const int token = blockIdx.x;
        const int i = blockIdx.y * blockDim.x + threadIdx.x;

        if ( threadIdx.y == 0 && i < pair_half )
        {
            const int absolute_position = position != nullptr ? *position : token % T + position_offset;

            rope_cos_sin( absolute_position, i, angles, shared_cos[ threadIdx.x ], shared_sin[ threadIdx.x ] );
        }

        __syncthreads();

        if ( i >= pair_half )
            return;

        const float c = shared_cos[ threadIdx.x ];
        const float s = shared_sin[ threadIdx.x ];

        for ( int h = threadIdx.y; h < n_heads + n_kv_heads; h += blockDim.y )
        {
            const bool is_query = h < n_heads;
            TElement* out = is_query ? out_q : out_k;
            const TElement* in = is_query ? in_q : in_k;
            const int base_idx = is_query
                ? ( token * n_heads + h ) * head_dim
                : ( token * n_kv_heads + ( h - n_heads ) ) * head_dim;

            const float x0 = rope_load( in[ base_idx + i ] );
            const float x1 = rope_load( in[ base_idx + i + pair_half ] );

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

            out[ base_idx + i ] = rope_store<TElement>( r0 );
            out[ base_idx + i + pair_half ] = rope_store<TElement>( r1 );
        }
    }

    /**
     * @brief Launch the rotation over `tokens` tokens of Q and K, in one launch.
     *
     * @param position Device decode position, or null for a prefill chunk at position_offset.
     */
    template <bool negate_sin, typename TElement>
    void launch_rope_rotation(
        TElement* out_q, TElement* out_k,
        const TElement* in_q, const TElement* in_k,
        const RopeAngleParameters& angles,
        int tokens, int T,
        int n_heads, int n_kv_heads, int head_dim,
        int rotary_dim, int rotary_layout,
        int position_offset, const int* position,
        cudaStream_t stream )
    {
        // Which channel pairs rotate. RotaryPrefix (Qwen) pairs inside the leading rotary_dim; WholeHead pairs
        // across the head, and the angle function gives identity past the rotated pairs (Gemma's global layers).
        const int pair_half = ( rotary_layout == 1 && rotary_dim > 0 && rotary_dim < head_dim )
            ? ( rotary_dim / 2 )
            : head_dim / 2;

        const dim3 block( kRotationPairsPerBlock, kRotationHeadRows );
        const dim3 grid( tokens, ( pair_half + kRotationPairsPerBlock - 1 ) / kRotationPairsPerBlock );

        rope_rotate_kernel<negate_sin, TElement> << <grid, block, 0, stream >> > (
            out_q, in_q, out_k, in_k, angles, T, n_heads, n_kv_heads, head_dim, pair_half, position_offset, position );

        cudaCheck( cudaGetLastError() );
    }
}
