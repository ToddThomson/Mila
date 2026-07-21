/**
 * @file Gqa.Decode.Bf16.cu
 * @brief Fused single-token decode attention over the compact BF16 KV cache.
 *
 * Replaces the cuBLASLt QK -> softmax_decode -> AV decode pipeline (plus its
 * identity Q-permute/unpermute copies) with one streaming online-softmax kernel
 * that reads only the live attention band. See the GQA decode item in BACKLOG
 * and GqaFlashAttention.md for the prefill sibling.
 */

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_pipeline.h>
#include <device_launch_parameters.h>
#include <math_constants.h>
#include <algorithm>
#include <cassert>
#include "CudaUtils.h"
#include "CudaGqa.cuh"

namespace Mila::Dnn::Compute::Cuda::Gqa
{
    namespace
    {
        // Split-K partial rows are [head_size + 2] floats: unnormalized O, then m, l.
        constexpr int kMaxDecodeSplits = 128;

        // Block = group_size warps (one warp per Q row of the KV-head group).
        // 16 covers every production geometry: Gemma global MQA GS=16, Gemma
        // local GS=2, Llama GS=4. __launch_bounds__ below assumes this cap.
        constexpr int kMaxDecodeGroupSize = 16;

        // Grid-fill target for the split-K axis. The Gemma global layers are MQA
        // (NKV=1): without position splits a decode layer would run ONE block and
        // idle the other 45 SMs.
        constexpr int kDecodeTargetBlocks = 128;
        constexpr int kDecodeMinPositionsPerSplit = 64;

        /**
         * @brief Fused decode attention: one (kv_head, split, batch) block.
         *
         * Walks ABSOLUTE positions p in [window_start, actual_len) with physical
         * cache row = p % capacity -- identical rows to the ring decode softmax's
         * slot->position reconstruction (the KV write wraps by capacity), and the
         * identity mapping for the unbounded cache (p < capacity always). Only the
         * live band is ever read; the allocated capacity beyond it is not touched.
         *
         * Each warp owns one Q row of the group (register Q/O fragments, HS/64
         * float2 pairs per lane) and carries scalar online-softmax state (m, l)
         * replicated across its lanes via the shfl reduction. K/V are staged to
         * shared memory in position tiles loaded cooperatively by the whole block;
         * the per-lane pair index lane + i*32 keeps both the global loads and the
         * shared reads conflict-free.
         *
         * With num_splits == 1 the block normalizes and writes Y directly. With
         * splits, each block writes an unnormalized (O, m, l) partial and the
         * fixup kernel below merges them -- the standard flash-decode split-K.
         *
         * Q is read straight from the projection output [B, 1, NH*HS] and Y is
         * written as [B, 1, NH*HS]: for a single token both have exactly the
         * per-(batch, head) row layout used here, which is what made the old
         * permute_q_compact / unpermute_output launches identity copies.
         */
        template<int kHeadSize>
        __global__ void __launch_bounds__( kMaxDecodeGroupSize * 32 )
            gqa_decode_attention_bf16_kernel(
                const __nv_bfloat16* __restrict__ q,
                const __nv_bfloat16* __restrict__ k_cache,
                const __nv_bfloat16* __restrict__ v_cache,
                __nv_bfloat16* __restrict__ y,
                float* __restrict__ split_partials,
                int num_kv_heads,
                int group_size,
                int capacity,
                int actual_len,
                int window,
                int num_splits,
                float scale )
        {
            static_assert( kHeadSize % 64 == 0, "per-lane float2 fragments need HS divisible by 64" );

            // Tile size chosen so the double-buffered K + V stage is 32 KB for
            // every head size (2 stages x 2 tensors x 8 KB).
            constexpr int kTilePositions = 4096 / kHeadSize;
            constexpr int kPairsPerLane = kHeadSize / 64;
            constexpr int kInt4PerRow = kHeadSize / 8;
            constexpr int kTileElements = kTilePositions * kHeadSize;

            __shared__ __nv_bfloat16 s_k[ 2 ][ kTileElements ];
            __shared__ __nv_bfloat16 s_v[ 2 ][ kTileElements ];

            const int lane = threadIdx.x & 31;
            const int g = threadIdx.x >> 5;
            const int kv = blockIdx.x;
            const int split = blockIdx.y;
            const int batch = blockIdx.z;
            const int h = kv * group_size + g;
            const int num_heads = num_kv_heads * group_size;

            // Live band and this block's split chunk.
            const int window_start = ( window > 0 ) ? max( 0, actual_len - window ) : 0;
            const int band_len = actual_len - window_start;
            const int chunk = ( band_len + num_splits - 1 ) / num_splits;
            const int chunk_begin = window_start + split * chunk;
            const int chunk_end = min( chunk_begin + chunk, actual_len );

            // Register Q fragment for this warp's row.
            const __nv_bfloat162* q2 = reinterpret_cast<const __nv_bfloat162*>(
                q + ( static_cast<size_t>( batch ) * num_heads + h ) * kHeadSize );

            float2 q_frag[ kPairsPerLane ];

#pragma unroll
            for ( int i = 0; i < kPairsPerLane; ++i )
                q_frag[ i ] = __bfloat1622float2( q2[ lane + i * 32 ] );

            // Online-softmax state and unnormalized output accumulator. An empty
            // chunk (splits past the band end) flows through as (m=-inf, l=0, O=0),
            // which the fixup weights to exactly zero.
            float m = -CUDART_INF_F;
            float l = 0.0f;
            float2 o_frag[ kPairsPerLane ] = {};

            const size_t kv_row_base = ( static_cast<size_t>( batch ) * num_kv_heads + kv ) * capacity;

            // cp.async double-buffer: tile i+1 is prefetched into the idle stage
            // while tile i computes, so the K/V stream overlaps the softmax math
            // (single-buffering measured 227 GB/s at the full global band; the
            // load-then-compute serialization was the limiter). Exactly one
            // async group is in flight at each wait, so wait_prior(0) suffices;
            // the top-of-loop barrier both publishes the waited stage and proves
            // the stage the next prefetch overwrites is fully consumed.
            const int num_tiles = ( chunk_end > chunk_begin )
                ? ( chunk_end - chunk_begin + kTilePositions - 1 ) / kTilePositions
                : 0;

            const auto prefetch_tile = [&]( int tile_index, int stage )
            {
                const int tile_begin = chunk_begin + tile_index * kTilePositions;
                const int tile_rows = min( kTilePositions, chunk_end - tile_begin );

                for ( int i = threadIdx.x; i < tile_rows * kInt4PerRow; i += blockDim.x )
                {
                    const int t = i / kInt4PerRow;
                    const int j = i - t * kInt4PerRow;
                    const int slot = ( tile_begin + t ) % capacity;
                    const size_t row = ( kv_row_base + slot ) * kHeadSize;

                    __pipeline_memcpy_async(
                        reinterpret_cast<int4*>( s_k[ stage ] ) + t * kInt4PerRow + j,
                        reinterpret_cast<const int4*>( k_cache + row ) + j, sizeof( int4 ) );
                    __pipeline_memcpy_async(
                        reinterpret_cast<int4*>( s_v[ stage ] ) + t * kInt4PerRow + j,
                        reinterpret_cast<const int4*>( v_cache + row ) + j, sizeof( int4 ) );
                }

                // Zero the ragged tail rows (last tile only) so the compute loop's
                // fixed-trip unroll reads defined values; the -inf score mask turns
                // their probability weights into exact zeros.
                for ( int i = threadIdx.x + tile_rows * kInt4PerRow;
                    i < kTilePositions * kInt4PerRow; i += blockDim.x )
                {
                    reinterpret_cast<int4*>( s_k[ stage ] )[ i ] = int4{ 0, 0, 0, 0 };
                    reinterpret_cast<int4*>( s_v[ stage ] )[ i ] = int4{ 0, 0, 0, 0 };
                }

                __pipeline_commit();
            };

            if ( num_tiles > 0 )
                prefetch_tile( 0, 0 );

            for ( int tile = 0; tile < num_tiles; ++tile )
            {
                const int stage = tile & 1;
                const int tile_begin = chunk_begin + tile * kTilePositions;
                const int tile_rows = min( kTilePositions, chunk_end - tile_begin );

                __pipeline_wait_prior( 0 );
                __syncthreads();

                if ( tile + 1 < num_tiles )
                    prefetch_tile( tile + 1, stage ^ 1 );

                // Tile-granular online softmax (the per-position form serialized the
                // warp on its shfl reduce + scalar m/l chain and measured 229 GB/s at
                // the full global band). Phase 1 computes every tile score with
                // independent, interleavable reduce chains; phase 2 applies ONE
                // rescale for the whole tile. Ragged tail positions carry a -inf
                // score, so their weights are exact zeros against the zero-filled
                // smem rows.
                float score[ kTilePositions ];

#pragma unroll
                for ( int t = 0; t < kTilePositions; ++t )
                {
                    const __nv_bfloat162* k2 = reinterpret_cast<const __nv_bfloat162*>( s_k[ stage ] + t * kHeadSize );

                    float dot = 0.0f;

#pragma unroll
                    for ( int i = 0; i < kPairsPerLane; ++i )
                    {
                        const float2 kf = __bfloat1622float2( k2[ lane + i * 32 ] );
                        dot += q_frag[ i ].x * kf.x + q_frag[ i ].y * kf.y;
                    }

#pragma unroll
                    for ( int offset = 16; offset > 0; offset >>= 1 )
                        dot += __shfl_xor_sync( 0xffffffffu, dot, offset );

                    score[ t ] = ( t < tile_rows ) ? dot * scale : -CUDART_INF_F;
                }

                float tile_max = -CUDART_INF_F;

#pragma unroll
                for ( int t = 0; t < kTilePositions; ++t )
                    tile_max = fmaxf( tile_max, score[ t ] );

                const float m_new = fmaxf( m, tile_max );
                const float alpha = expf( m - m_new );

                float p[ kTilePositions ];
                float p_sum = 0.0f;

#pragma unroll
                for ( int t = 0; t < kTilePositions; ++t )
                {
                    p[ t ] = expf( score[ t ] - m_new );
                    p_sum += p[ t ];
                }

                l = l * alpha + p_sum;

#pragma unroll
                for ( int i = 0; i < kPairsPerLane; ++i )
                {
                    float2 acc;
                    acc.x = o_frag[ i ].x * alpha;
                    acc.y = o_frag[ i ].y * alpha;

#pragma unroll
                    for ( int t = 0; t < kTilePositions; ++t )
                    {
                        const float2 vf = __bfloat1622float2(
                            reinterpret_cast<const __nv_bfloat162*>( s_v[ stage ] + t * kHeadSize )[ lane + i * 32 ] );
                        acc.x += p[ t ] * vf.x;
                        acc.y += p[ t ] * vf.y;
                    }

                    o_frag[ i ] = acc;
                }

                m = m_new;
            }

            if ( num_splits == 1 )
            {
                const float inv_l = 1.0f / l;
                __nv_bfloat162* y2 = reinterpret_cast<__nv_bfloat162*>(
                    y + ( static_cast<size_t>( batch ) * num_heads + h ) * kHeadSize );

#pragma unroll
                for ( int i = 0; i < kPairsPerLane; ++i )
                    y2[ lane + i * 32 ] = __floats2bfloat162_rn( o_frag[ i ].x * inv_l, o_frag[ i ].y * inv_l );
            }
            else
            {
                // Partial row layout: [O[0..HS)] fp32, then m, l. (HS+2)*4 bytes is
                // 8-byte aligned for every supported HS, so float2 stores are legal.
                float* partial = split_partials +
                    ( ( ( static_cast<size_t>( batch ) * num_kv_heads + kv ) * num_splits + split )
                        * group_size + g ) * ( kHeadSize + 2 );
                float2* p2 = reinterpret_cast<float2*>( partial );

#pragma unroll
                for ( int i = 0; i < kPairsPerLane; ++i )
                    p2[ lane + i * 32 ] = o_frag[ i ];

                if ( lane == 0 )
                {
                    partial[ kHeadSize ] = m;
                    partial[ kHeadSize + 1 ] = l;
                }
            }
        }

        /**
         * @brief Split-K merge: combines per-split (O, m, l) partials into Y.
         *
         * One block per (head, batch). The per-split softmax weights
         * exp(m_s - m_max) are staged in shared memory once, then each thread
         * merges a strided subset of the head dims. Empty splits carry
         * (m=-inf, l=0, O=0) and contribute exactly zero.
         */
        __global__ void gqa_decode_attention_fixup_bf16_kernel(
            __nv_bfloat16* __restrict__ y,
            const float* __restrict__ split_partials,
            int num_kv_heads,
            int group_size,
            int head_size,
            int num_splits )
        {
            __shared__ float s_weight[ kMaxDecodeSplits ];
            __shared__ float s_inv_l;

            const int h = blockIdx.x;
            const int batch = blockIdx.y;
            const int kv = h / group_size;
            const int g = h - kv * group_size;
            const int num_heads = gridDim.x;

            const size_t partial_stride = static_cast<size_t>( head_size + 2 );
            const size_t split_stride = static_cast<size_t>( group_size ) * partial_stride;
            const float* base = split_partials +
                ( ( static_cast<size_t>( batch ) * num_kv_heads + kv ) * num_splits * group_size + g )
                * partial_stride;

            if ( threadIdx.x == 0 )
            {
                float m_max = -CUDART_INF_F;

                for ( int s = 0; s < num_splits; ++s )
                    m_max = fmaxf( m_max, base[ s * split_stride + head_size ] );

                float l_total = 0.0f;

                for ( int s = 0; s < num_splits; ++s )
                {
                    const float weight = expf( base[ s * split_stride + head_size ] - m_max );
                    s_weight[ s ] = weight;
                    l_total += base[ s * split_stride + head_size + 1 ] * weight;
                }

                s_inv_l = 1.0f / l_total;
            }

            __syncthreads();

            for ( int dim = threadIdx.x; dim < head_size; dim += blockDim.x )
            {
                float acc = 0.0f;

                for ( int s = 0; s < num_splits; ++s )
                    acc += base[ s * split_stride + dim ] * s_weight[ s ];

                y[ ( static_cast<size_t>( batch ) * num_heads + h ) * head_size + dim ] =
                    __float2bfloat16( acc * s_inv_l );
            }
        }
    } // anonymous namespace

    bool cuda_gqa_decode_attention_supported( int head_size, int group_size )
    {
        return ( head_size == 128 || head_size == 256 || head_size == 512 )
            && group_size >= 1 && group_size <= kMaxDecodeGroupSize;
    }

    size_t cuda_gqa_decode_attention_scratch_bytes( int B, int NH, int HS )
    {
        return static_cast<size_t>( B ) * NH * kMaxDecodeSplits
            * ( static_cast<size_t>( HS ) + 2 ) * sizeof( float );
    }

    void cuda_gqa_decode_attention_bf16(
        const __nv_bfloat16* Q, const __nv_bfloat16* K, const __nv_bfloat16* V,
        __nv_bfloat16* Y, float* split_scratch,
        int B, int NH, int NKV, int HS, int cache_capacity,
        int actual_len, int window, float scale,
        cudaStream_t stream )
    {
        const int group_size = NH / NKV;

        assert( NH % NKV == 0 );
        assert( cuda_gqa_decode_attention_supported( HS, group_size ) );
        assert( actual_len >= 1 );

        // Split policy: fill the device (kDecodeTargetBlocks) without dropping a
        // split's chunk below kDecodeMinPositionsPerSplit positions.
        const int window_start = ( window > 0 ) ? std::max( 0, actual_len - window ) : 0;
        const int band_len = actual_len - window_start;
        const int splits_by_target = ( kDecodeTargetBlocks + NKV - 1 ) / NKV;
        const int splits_by_band =
            ( band_len + kDecodeMinPositionsPerSplit - 1 ) / kDecodeMinPositionsPerSplit;
        const int num_splits =
            std::max( 1, std::min( { splits_by_target, splits_by_band, kMaxDecodeSplits } ) );

        const dim3 grid( NKV, num_splits, B );
        const dim3 block( group_size * 32 );

        switch ( HS )
        {
            case 128:
                gqa_decode_attention_bf16_kernel<128><<<grid, block, 0, stream>>>(
                    Q, K, V, Y, split_scratch, NKV, group_size, cache_capacity,
                    actual_len, window, num_splits, scale );
                break;

            case 256:
                gqa_decode_attention_bf16_kernel<256><<<grid, block, 0, stream>>>(
                    Q, K, V, Y, split_scratch, NKV, group_size, cache_capacity,
                    actual_len, window, num_splits, scale );
                break;

            case 512:
                gqa_decode_attention_bf16_kernel<512><<<grid, block, 0, stream>>>(
                    Q, K, V, Y, split_scratch, NKV, group_size, cache_capacity,
                    actual_len, window, num_splits, scale );
                break;

            default:
                assert( false && "cuda_gqa_decode_attention_bf16: unsupported head size" );
                return;
        }

        cudaCheck( cudaGetLastError() );

        if ( num_splits > 1 )
        {
            const dim3 fixup_grid( NH, B );

            gqa_decode_attention_fixup_bf16_kernel<<<fixup_grid, 128, 0, stream>>>(
                Y, split_scratch, NKV, group_size, HS, num_splits );

            cudaCheck( cudaGetLastError() );
        }
    }
}
