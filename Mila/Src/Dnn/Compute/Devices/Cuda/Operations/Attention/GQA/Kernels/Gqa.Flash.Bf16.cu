// Gqa.Flash.Bf16.cu
//
// RETIRED 2026-07-10 -- out of the CMake build (see Mila/CMakeLists.txt). This scalar
// warp-per-row kernel was profiled a non-fix: it is LSU / shared-memory-instruction
// bound (one load per FMA), so the shared-memory tiling that cut global traffic ~16x
// bought only ~1.2x (GqaFlashAttention.md 5.2.1). It is superseded by the tensor-core
// WMMA path (Gqa.Flash.Wmma.cu), which owns the cuda_gqa_flash_prefill_bf16 symbol.
// Kept in-tree as the reference scalar implementation and the tiling scaffold whose
// block/tile/two-pass/online-softmax structure the WMMA kernel reuses. Do not re-add to
// the build alongside the WMMA file -- both define cuda_gqa_flash_prefill_bf16.
//
// ----------------------------------------------------------------------------------
//
// FlashAttention prefill for the compact BF16 GQA KV cache (Iteration 2).
//
// Fused, streaming attention over the production [B, NKV, cache_capacity, HS] BF16
// cache. Replaces the cuBLASLt QK -> softmax -> AV pipeline (and its permute /
// unpermute bookends) with a single kernel that reads Q from the projection output,
// K/V from the cache, and writes the attention output directly -- no preatt / att /
// v_out materialization. This is the "where-we-compute, not what-we-compute" change:
// numerically it reproduces prefill_softmax_bf16_kernel + the AV GEMM within atol,
// while transient attention workspace scales to zero. See GqaFlashAttention.md.
//
// Iteration 2 adds the shared-memory K/V tiling that Iteration 1 skipped (5.0/5.2). A
// block owns Br = kWarpsPerBlock query rows (one warp per row, HS striped across the
// 32 lanes exactly as Iteration 1) for a fixed (batch, query-head), and streams Bc-key
// tiles of K, then V, through a single reused shared-memory buffer. K/V are therefore
// read from global memory ONCE per query tile instead of once per query row -- the ~Br
// traffic cut that turns Iteration 1's memory-bound regression into a win. The
// online-softmax state (m, l, o) accumulates across key tiles per row with one rescale
// per tile (FlashAttention-2), not one per key.
//
// Iteration 2 scope: unbounded / global causal path only (kBounded == false, window
// == 0 for Gemma global layers, or full causal for Llama). The bounded sliding-window
// ring is a later iteration (5.4). Head dim HS must be a multiple of 32 and <= 512.
// This ceiling is load-bearing: the Gemma 4 global layers (the only layers this path
// serves) use global_head_dim = 512, NOT the sliding head_dim of 256 -- the sliding
// layers are bounded and take the cuBLASLt path. Llama's 128 also qualifies.

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <device_launch_parameters.h>
#include <math_constants.h>
#include <stdexcept>
#include "CudaUtils.h"
#include "CudaGqa.cuh"

namespace Mila::Dnn::Compute::Cuda::Gqa
{
    // Tiling shape (Iteration 2). One thread block owns Br = kWarpsPerBlock query rows
    // (one warp each, HS striped over the 32 lanes: lane L holds dims { L, L+32, ... }).
    // A single shared-memory tile of kFlashTileKeys key rows is loaded from K, then
    // reused for V (the two-pass K-then-V load), so each K/V row is fetched from global
    // memory once per query TILE rather than once per query row (~Br traffic reduction).
    //
    // Br = 16 is the sweet spot for this warp-per-row (HS-in-registers) layout: bigger Br
    // cuts more traffic, but the per-lane register arrays (q_reg[HS/32] + accum[HS/32] +
    // p_tile[Bc]) make Br = 32 overflow the register file. Bc = 32 keeps the reused tile
    // at 32 KB (32 x 512 x 2), under the 48 KB default limit -- no >48 KB opt-in
    // (cudaFuncAttributeMaxDynamicSharedMemorySize) is needed here; that is 5.3's problem,
    // where the FP32 accumulator tile forces it. Both are tuning knobs for the profiler.
    constexpr int kFlashWarpSize = 32;
    constexpr int kWarpsPerBlock = 16;   // Br: query rows per block
    constexpr int kFlashTileKeys = 32;   // Bc: key positions per shared-memory tile
    constexpr int kMaxDimsPerLane = 16;  // HS / 32 <= 16  (HS <= 512)

    __device__ __forceinline__ float flashWarpReduceSum( float value )
    {
        // Butterfly reduction: every lane ends with the full warp sum (all lanes need
        // the complete QK score to update their own output stripe). Called warp-uniformly
        // (the whole warp is active or the whole warp is idle), so the full mask is valid.
        for ( int offset = kFlashWarpSize / 2; offset > 0; offset >>= 1 )
            value += __shfl_xor_sync( 0xffffffffu, value, offset );

        return value;
    }

    // Dynamic shared memory: a single [kFlashTileKeys * HS] BF16 tile, reused for the K
    // pass and then the V pass of each key tile. Sized at launch to kFlashTileKeys * HS.
    __global__ void gqa_flash_prefill_bf16_kernel(
        const __nv_bfloat16* __restrict__ Q,   // [B, chunk_len, NH * HS] (projection output)
        const __nv_bfloat16* __restrict__ K,   // [B, NKV, cache_capacity, HS]
        const __nv_bfloat16* __restrict__ V,   // [B, NKV, cache_capacity, HS]
        __nv_bfloat16* __restrict__ Y,         // [B, chunk_len, NH * HS] (attention output)
        int B, int chunk_len, int NH, int NKV, int HS, int cache_capacity,
        int position_offset, int window, float scale )
    {
        extern __shared__ __nv_bfloat16 kv_tile[];   // [kFlashTileKeys * HS]

        const int lane = threadIdx.x % kFlashWarpSize;
        const int warp_in_block = threadIdx.x / kFlashWarpSize;

        // Block -> (batch, query head, query tile). Every warp in the block shares the
        // same (b, nh) -- and thus the same KV head and the same K/V tile in shared
        // memory -- and owns a distinct query token within the tile.
        const int num_query_tiles = ( chunk_len + kWarpsPerBlock - 1 ) / kWarpsPerBlock;
        const int query_tile = blockIdx.x % num_query_tiles;
        int tmp = blockIdx.x / num_query_tiles;
        const int nh = tmp % NH;
        const int b = tmp / NH;

        const int t = query_tile * kWarpsPerBlock + warp_in_block;

        // Warps past the (possibly partial) final tile carry no query row, but must still
        // participate in the cooperative tile loads and __syncthreads below -- never
        // early-return them, or the block deadlocks. Only the per-row compute is guarded.
        const bool active = ( t < chunk_len );

        const int group_size = NH / NKV;
        const int head_kv = nh / group_size;
        const int dims_per_lane = HS / kFlashWarpSize;   // HS multiple of 32

        // Base of this row's KV head in the compact cache. Physical position == absolute
        // position for the unbounded cache (no ring wrap). Block-uniform.
        const size_t kv_head_base =
            static_cast<size_t>( b * NKV + head_kv ) * cache_capacity * HS;

        const int abs_t = position_offset + t;

        // Sliding-window lower bound. window <= 0 is global causal (window_start = 0).
        // Iteration 2 only routes the unbounded cache here, where window == 0; the
        // expression is kept general so a future windowed-but-unbounded case is correct.
        const int window_start = ( window > 0 ) ? max( 0, abs_t - window + 1 ) : 0;

        // Load this row's Q stripe into registers once.
        const size_t row_base = ( static_cast<size_t>( b * chunk_len + t ) * NH + nh ) * HS;

        float q_reg[ kMaxDimsPerLane ];
        float accum[ kMaxDimsPerLane ];
        if ( active )
        {
            const __nv_bfloat16* q_row = Q + row_base;
            for ( int i = 0; i < dims_per_lane; ++i )
            {
                q_reg[ i ] = __bfloat162float( q_row[ i * kFlashWarpSize + lane ] );
                accum[ i ] = 0.0f;
            }
        }

        float m_i = -CUDART_INF_F;
        float l_i = 0.0f;

        // Scores, then per-key probabilities, for the tile currently in shared memory.
        float p_tile[ kFlashTileKeys ];

        // Causal upper bound: the block never attends past its own last row's absolute
        // position, so key tiles beyond it are skipped entirely (the triangular-global
        // work saving). Rows with smaller abs_t mask the tail per key below.
        const int block_last_t = min( chunk_len - 1, query_tile * kWarpsPerBlock + kWarpsPerBlock - 1 );
        const int block_max_key = position_offset + block_last_t;

        for ( int tile_start = 0; tile_start <= block_max_key; tile_start += kFlashTileKeys )
        {
            const int tile_len = min( kFlashTileKeys, block_max_key - tile_start + 1 );

            // --- Pass 1: stage the K tile in shared memory, cooperatively over the block.
            __syncthreads();   // prior tile's V pass finished reading kv_tile
            for ( int idx = threadIdx.x; idx < tile_len * HS; idx += blockDim.x )
            {
                const int j = idx / HS;
                const int d = idx % HS;
                kv_tile[ idx ] = K[ kv_head_base + static_cast<size_t>( tile_start + j ) * HS + d ];
            }
            __syncthreads();

            if ( active )
            {
                // Score every key in the tile; track the masked tile max.
                float m_tile = -CUDART_INF_F;
                for ( int j = 0; j < tile_len; ++j )
                {
                    const __nv_bfloat16* k_s = kv_tile + static_cast<size_t>( j ) * HS;

                    float partial = 0.0f;
                    for ( int i = 0; i < dims_per_lane; ++i )
                        partial += q_reg[ i ] * __bfloat162float( k_s[ i * kFlashWarpSize + lane ] );

                    float score = flashWarpReduceSum( partial ) * scale;

                    const int p_key = tile_start + j;
                    if ( p_key > abs_t || p_key < window_start )
                        score = -CUDART_INF_F;

                    p_tile[ j ] = score;
                    m_tile = fmaxf( m_tile, score );
                }

                if ( m_tile > -CUDART_INF_F )
                {
                    // Merge this tile into the running online-softmax state: one rescale of
                    // the accumulator and denominator for the whole tile (FlashAttention-2),
                    // not one per key. alpha is 0 on the first contributing tile (m_i == -inf),
                    // which avoids the -inf - (-inf) NaN and correctly discards the empty state.
                    const float m_new = fmaxf( m_i, m_tile );
                    const float alpha = ( m_i == -CUDART_INF_F ) ? 0.0f : expf( m_i - m_new );

                    for ( int i = 0; i < dims_per_lane; ++i )
                        accum[ i ] *= alpha;

                    l_i *= alpha;

                    for ( int j = 0; j < tile_len; ++j )
                    {
                        const float p_exp = ( p_tile[ j ] == -CUDART_INF_F ) ? 0.0f : expf( p_tile[ j ] - m_new );
                        p_tile[ j ] = p_exp;
                        l_i += p_exp;
                    }

                    m_i = m_new;
                }
                else
                {
                    // Fully masked tile (all keys future / below window): contributes nothing.
                    for ( int j = 0; j < tile_len; ++j )
                        p_tile[ j ] = 0.0f;
                }
            }

            // --- Pass 2: overwrite the shared tile with V, accumulate the weighted sum.
            __syncthreads();   // all warps done reading K from kv_tile
            for ( int idx = threadIdx.x; idx < tile_len * HS; idx += blockDim.x )
            {
                const int j = idx / HS;
                const int d = idx % HS;
                kv_tile[ idx ] = V[ kv_head_base + static_cast<size_t>( tile_start + j ) * HS + d ];
            }
            __syncthreads();

            if ( active )
            {
                for ( int j = 0; j < tile_len; ++j )
                {
                    const float p_exp = p_tile[ j ];
                    if ( p_exp == 0.0f )
                        continue;

                    const __nv_bfloat16* v_s = kv_tile + static_cast<size_t>( j ) * HS;
                    for ( int i = 0; i < dims_per_lane; ++i )
                        accum[ i ] += p_exp * __bfloat162float( v_s[ i * kFlashWarpSize + lane ] );
                }
            }
        }

        if ( !active )
            return;

        // Causal attention always attends at least its own key, so l_i > 0.
        const float inv_sum = 1.0f / l_i;

        __nv_bfloat16* y_row = Y + row_base;
        for ( int i = 0; i < dims_per_lane; ++i )
            y_row[ i * kFlashWarpSize + lane ] = __float2bfloat16( accum[ i ] * inv_sum );
    }

    void cuda_gqa_flash_prefill_bf16(
        const __nv_bfloat16* Q, const __nv_bfloat16* K, const __nv_bfloat16* V,
        __nv_bfloat16* Y,
        int B, int chunk_len, int NH, int NKV, int HS, int cache_capacity,
        int position_offset, int window, float scale,
        cudaStream_t stream )
    {
        // Head-dim contract: striped across 32 lanes into fixed-size register arrays.
        // A violation here is silent stack corruption, not a wrong answer -- fail loud.
        if ( HS % kFlashWarpSize != 0 || HS / kFlashWarpSize > kMaxDimsPerLane )
            throw std::runtime_error(
                "cuda_gqa_flash_prefill_bf16: head_size must be a multiple of 32 and <= 512" );

        const int num_query_tiles = ceil_div( chunk_len, kWarpsPerBlock );
        const int grid_size = B * NH * num_query_tiles;
        const int block_threads = kWarpsPerBlock * kFlashWarpSize;
        const size_t smem_bytes = static_cast<size_t>( kFlashTileKeys ) * HS * sizeof( __nv_bfloat16 );

        gqa_flash_prefill_bf16_kernel <<< grid_size, block_threads, smem_bytes, stream >>> (
            Q, K, V, Y,
            B, chunk_len, NH, NKV, HS, cache_capacity,
            position_offset, window, scale );

        cudaCheck( cudaGetLastError() );
    }
}
