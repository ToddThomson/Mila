// Gqa.Flash.Bf16.cu
//
// FlashAttention prefill for the compact BF16 GQA KV cache (Iteration 1).
//
// Fused, streaming attention over the production [B, NKV, cache_capacity, HS] BF16
// cache. Replaces the cuBLASLt QK -> softmax -> AV pipeline (and its permute /
// unpermute bookends) with a single kernel that reads Q from the projection output,
// K/V from the cache, and writes the attention output directly -- no preatt / att /
// v_out materialization. This is the "where-we-compute, not what-we-compute" change:
// numerically it reproduces prefill_softmax_bf16_kernel + the AV GEMM within atol,
// while transient attention workspace scales to zero. See GqaFlashAttention.md.
//
// Iteration 1 scope: unbounded / global causal path only (kBounded == false, window
// == 0 for Gemma global layers, or full causal for Llama). The bounded sliding-window
// ring is a later iteration (its column-j-is-a-ring-slot masking differs). Head dim
// HS must be a multiple of 32 and <= 512. This ceiling is load-bearing: the Gemma 4
// global layers (the only layers this path serves) use global_head_dim = 512, NOT the
// sliding head_dim of 256 -- the sliding layers are bounded and take the cuBLASLt path.
// Llama's 128 also qualifies.

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <device_launch_parameters.h>
#include <math_constants.h>
#include <stdexcept>
#include "CudaUtils.h"
#include "CudaGqa.cuh"

namespace Mila::Dnn::Compute::Cuda::Gqa
{
    // One warp owns one (batch, query-head, query-token) row. The head dimension is
    // striped across the 32 lanes: lane L holds dims { L, L+32, L+64, ... }, so at each
    // stripe step all 32 lanes touch a contiguous 32-element span of the row (coalesced
    // K/V loads). kMaxDimsPerLane bounds the per-lane register arrays: HS <= 512 -> 16,
    // sized for the Gemma 4 global head dim (512), which is what this path runs on.
    constexpr int kFlashWarpSize = 32;
    constexpr int kFlashWarpsPerBlock = 4;
    constexpr int kMaxDimsPerLane = 16;

    __device__ __forceinline__ float flashWarpReduceSum( float value )
    {
        // Butterfly reduction: every lane ends with the full warp sum (all lanes need
        // the complete QK score to update their own output stripe).
        for ( int offset = kFlashWarpSize / 2; offset > 0; offset >>= 1 )
            value += __shfl_xor_sync( 0xffffffffu, value, offset );

        return value;
    }

    __global__ void gqa_flash_prefill_bf16_kernel(
        const __nv_bfloat16* __restrict__ Q,   // [B, chunk_len, NH * HS] (projection output)
        const __nv_bfloat16* __restrict__ K,   // [B, NKV, cache_capacity, HS]
        const __nv_bfloat16* __restrict__ V,   // [B, NKV, cache_capacity, HS]
        __nv_bfloat16* __restrict__ Y,         // [B, chunk_len, NH * HS] (attention output)
        int B, int chunk_len, int NH, int NKV, int HS, int cache_capacity,
        int position_offset, int window, float scale )
    {
        const int lane = threadIdx.x % kFlashWarpSize;
        const int warp_in_block = threadIdx.x / kFlashWarpSize;
        const int warp_global = blockIdx.x * kFlashWarpsPerBlock + warp_in_block;

        const int total_rows = B * NH * chunk_len;

        if ( warp_global >= total_rows )
            return;

        // Decompose the flat warp index into (batch, query head, query token).
        const int t = warp_global % chunk_len;
        int tmp = warp_global / chunk_len;
        const int nh = tmp % NH;
        const int b = tmp / NH;

        const int group_size = NH / NKV;
        const int head_kv = nh / group_size;

        const int dims_per_lane = HS / kFlashWarpSize;   // HS multiple of 32

        // Load this row's Q stripe into registers once.
        const size_t row_base = ( static_cast<size_t>( b * chunk_len + t ) * NH + nh ) * HS;
        const __nv_bfloat16* q_row = Q + row_base;

        float q_reg[ kMaxDimsPerLane ];
        float accum[ kMaxDimsPerLane ];
        for ( int i = 0; i < dims_per_lane; ++i )
        {
            q_reg[ i ] = __bfloat162float( q_row[ i * kFlashWarpSize + lane ] );
            accum[ i ] = 0.0f;
        }

        const int abs_t = position_offset + t;

        // Sliding-window lower bound. window <= 0 is global causal (window_start = 0).
        // Iteration 1 only routes the unbounded cache here, where window == 0; the
        // expression is kept general so a future windowed-but-unbounded case is correct.
        const int window_start = ( window > 0 ) ? max( 0, abs_t - window + 1 ) : 0;

        // Base of this row's KV head in the compact cache. Physical position == absolute
        // position for the unbounded cache (no ring wrap).
        const size_t kv_head_base =
            static_cast<size_t>( b * NKV + head_kv ) * cache_capacity * HS;

        float m_i = -CUDART_INF_F;
        float l_i = 0.0f;

        for ( int p = window_start; p <= abs_t; ++p )
        {
            const __nv_bfloat16* k_row = K + kv_head_base + static_cast<size_t>( p ) * HS;

            float partial = 0.0f;
            for ( int i = 0; i < dims_per_lane; ++i )
                partial += q_reg[ i ] * __bfloat162float( k_row[ i * kFlashWarpSize + lane ] );

            const float score = flashWarpReduceSum( partial ) * scale;

            const float m_next = fmaxf( m_i, score );
            const float alpha = expf( m_i - m_next );
            const float p_exp = expf( score - m_next );

            const __nv_bfloat16* v_row = V + kv_head_base + static_cast<size_t>( p ) * HS;
            for ( int i = 0; i < dims_per_lane; ++i )
                accum[ i ] = accum[ i ] * alpha + p_exp * __bfloat162float( v_row[ i * kFlashWarpSize + lane ] );

            l_i = l_i * alpha + p_exp;
            m_i = m_next;
        }

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

        const int total_rows = B * NH * chunk_len;
        const int block_threads = kFlashWarpsPerBlock * kFlashWarpSize;
        const int grid_size = ceil_div( total_rows, kFlashWarpsPerBlock );

        gqa_flash_prefill_bf16_kernel <<< grid_size, block_threads, 0, stream >>> (
            Q, K, V, Y,
            B, chunk_len, NH, NKV, HS, cache_capacity,
            position_offset, window, scale );

        cudaCheck( cudaGetLastError() );
    }
}
