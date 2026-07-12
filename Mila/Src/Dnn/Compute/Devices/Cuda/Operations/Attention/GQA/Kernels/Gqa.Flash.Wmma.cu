// Gqa.Flash.Wmma.cu
//
// FlashAttention prefill over the compact BF16 GQA KV cache -- tensor-core path,
// Iteration 3 / Stage 2c (cp.async double-buffered software pipeline). See
// GqaFlashAttention.md 5.3 / 5.3.1.
//
// LADDER RECAP. Scalar warp-per-row (Gqa.Flash.Bf16.cu) was LSU-bound. Stage 1 (single-
// warp wmma, O in smem) proved correctness. Stage 2a (multi-warp HS-split, O in smem)
// engaged tensor cores but was occupancy/latency-bound (O tile capped 1 block/SM). Stage
// 2b moved O_w into mma.sync accumulator REGISTERS (per-row alpha on the documented
// fragment layout), freeing smem to ~42 KB / 2 blocks/SM and halving per-instance time --
// but ncu showed the dominant stall is now LONG SCOREBOARD (~35%): the synchronous
// global->smem K/V loads block the MMAs.
//
// STAGE 2c fixes that with a cp.async double-buffered pipeline: the next key tile's K and
// V are prefetched from global (cp.async) while the current tile's QK + softmax + PV run,
// so load latency overlaps compute instead of stalling it. The QK split-K (wmma), online
// softmax, and mma.sync PV math are IDENTICAL to 2b -- only the staging changes, so the
// CudaGqaFlashPrefillParity oracle stays exact.
//
// OCCUPANCY TRADE (HS=512). Double-buffering separate K and V tiles costs ~90 KB smem ->
// 1 block/SM (2b was ~42 KB / 2 blocks). The pipeline must hide enough latency to beat the
// halved occupancy; unlike 2a/2b this is not a near-certain win and must be profiled.
// OOB key rows (p >= cache_capacity, only in a partial final tile) are causally masked
// anyway, so their cp.async source is clamped to the last valid row rather than zero-filled
// -- QK score masked to -inf, PV weight P=0, identical result to 2b's zero-fill.
//
// m16n8k16 .f32.bf16.bf16.f32 fragment layout (PTX ISA), group = lane/4, tig = lane%4:
//   A[16x16] a0..a3 (row-major): a0={A[g][2t],A[g][2t+1]} a1={A[g+8][2t],..}
//                                a2={A[g][2t+8],..}       a3={A[g+8][2t+8],..}
//   B[16x8]  b0,b1  (col-major): b0={B[2t][g],B[2t+1][g]} b1={B[2t+8][g],B[2t+9][g]}
//   C[16x8]  c0..c3 (row-major): c0=C[g][2t] c1=C[g][2t+1] c2=C[g+8][2t] c3=C[g+8][2t+1]
//
// Unbounded / global causal path only (kBounded == false, window == 0 for Gemma global;
// full causal for Llama). HS a multiple of 16, HSt = HS/W a multiple of 16, HSt <=
// kMaxNTiles*8. Physical cache row == absolute position.

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <mma.h>
#include <math_constants.h>
#include <cstdint>
#include <stdexcept>
#include <cuda_pipeline.h>
#include "CudaUtils.h"
#include "CudaGqa.cuh"

namespace Mila::Dnn::Compute::Cuda::Gqa
{
    using namespace nvcuda;

    namespace
    {
        constexpr int kWmmaM = 16;   // query rows per tile (WMMA/MMA M) == Br
        constexpr int kWmmaN = 16;   // key positions per tile (WMMA N, MMA K) == Bc
        constexpr int kWmmaK = 16;   // contraction step (WMMA K)
        constexpr int kRowsPerTile = kWmmaM;   // Br
        constexpr int kKeysPerTile = kWmmaN;   // Bc
        constexpr int kMaxWarps = 8;           // W ceiling (register/occupancy budget)
        constexpr int kMmaN = 8;               // mma.sync N tile (m16n8k16)
        constexpr int kMaxNTiles = 8;          // HSt / kMmaN ceiling -> HSt <= 64
        constexpr int kCopyElems = 8;          // bf16 elements per 16-byte cp.async copy

        __host__ __device__ inline int flash_warp_count( int HS )
        {
            int W = kMaxWarps;

            while ( W > 1 && ( HS % W != 0 || ( HS / W ) % kWmmaK != 0 ) )
                W /= 2;

            return W;
        }

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
        __device__ __forceinline__ void mma_m16n8k16_bf16(
            float& c0, float& c1, float& c2, float& c3,
            uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
            uint32_t b0, uint32_t b1 )
        {
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                : "+f"( c0 ), "+f"( c1 ), "+f"( c2 ), "+f"( c3 )
                : "r"( a0 ), "r"( a1 ), "r"( a2 ), "r"( a3 ), "r"( b0 ), "r"( b1 ) );
        }

        // Prefetch one key tile's K and V ([Bc x HS] BF16 each) from global into the given
        // smem stage via 16-byte cp.async copies. OOB key rows are clamped to the last valid
        // cache row (their contribution is masked out downstream). Caller commits + waits.
        __device__ __forceinline__ void cp_async_kv_tile(
            __nv_bfloat16* s_K_stage, __nv_bfloat16* s_V_stage,
            const __nv_bfloat16* K, const __nv_bfloat16* V,
            size_t kv_head_base, int tile_start, int HS, int cache_capacity,
            int tid, int block_threads )
        {
            const int chunks_per_row = HS / kCopyElems;
            const int num_chunks = kKeysPerTile * chunks_per_row;

            for ( int c = tid; c < num_chunks; c += block_threads )
            {
                const int p_local = c / chunks_per_row;
                const int d0 = ( c % chunks_per_row ) * kCopyElems;
                const int p = tile_start + p_local;
                const int p_src = ( p < cache_capacity ) ? p : ( cache_capacity - 1 );

                const size_t g_off = kv_head_base + static_cast<size_t>( p_src ) * HS + d0;
                const int s_off = p_local * HS + d0;

                __pipeline_memcpy_async( s_K_stage + s_off, K + g_off, 16 );
                __pipeline_memcpy_async( s_V_stage + s_off, V + g_off, 16 );
            }
        }
#endif
    }

    // Stage 2c: a block of W warps owns (batch, query-head, 16-row query tile) and splits
    // the head dimension (warp w owns output columns [w*HSt, (w+1)*HSt)). K and V key tiles
    // are DOUBLE-BUFFERED in smem and prefetched with cp.async one tile ahead. Q is resident
    // for the whole key loop; O_w is register-resident (mma.sync). Shared memory does NOT
    // hold O.
    __global__ void gqa_flash_prefill_wmma_bf16_kernel(
        const __nv_bfloat16* __restrict__ Q,   // [B, chunk_len, NH * HS]
        const __nv_bfloat16* __restrict__ K,   // [B, NKV, cache_capacity, HS]
        const __nv_bfloat16* __restrict__ V,   // [B, NKV, cache_capacity, HS]
        __nv_bfloat16* __restrict__ Y,         // [B, chunk_len, NH * HS]
        int B, int chunk_len, int NH, int NKV, int HS, int cache_capacity,
        int position_offset, int window, float scale )
    {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
        const int tid = threadIdx.x;
        const int warp_id = tid >> 5;
        const int lane = tid & 31;
        const int warps = blockDim.x >> 5;
        const int block_threads = blockDim.x;

        const int g = lane >> 2;           // mma accumulator group (owns rows g, g+8)
        const int tg = lane & 3;           // mma thread-in-group   (owns cols 2*tg, +1)

        const int hs_tile = HS / warps;
        const int col0 = warp_id * hs_tile;
        const int num_ntiles = hs_tile / kMmaN;

        const int query_tile = blockIdx.x;
        const int nh = blockIdx.y;
        const int b = blockIdx.z;

        const int qrow0 = query_tile * kRowsPerTile;

        const int group_size = NH / NKV;
        const int head_kv = nh / group_size;
        const size_t kv_head_base =
            static_cast<size_t>( b * NKV + head_kv ) * cache_capacity * HS;

        // --- shared-memory layout (must match the launcher's size computation) ---
        extern __shared__ char smem_raw[];
        float* s_S_partial = reinterpret_cast<float*>( smem_raw );        // [W * Br * Bc]
        float* s_S = s_S_partial + warps * kRowsPerTile * kKeysPerTile;   // [Br * Bc]
        float* s_m = s_S + kRowsPerTile * kKeysPerTile;                   // [Br]
        float* s_l = s_m + kRowsPerTile;                                  // [Br]
        float* s_alpha = s_l + kRowsPerTile;                             // [Br]
        __nv_bfloat16* s_Q =
            reinterpret_cast<__nv_bfloat16*>( s_alpha + kRowsPerTile );   // [Br * HS]
        __nv_bfloat16* s_K = s_Q + kRowsPerTile * HS;                     // [2][Bc * HS]
        __nv_bfloat16* s_V = s_K + 2 * kKeysPerTile * HS;                 // [2][Bc * HS]
        __nv_bfloat16* s_P = s_V + 2 * kKeysPerTile * HS;                 // [Br * Bc]

        float* s_S_w = s_S_partial + warp_id * kRowsPerTile * kKeysPerTile;

        const int stage_elems = kKeysPerTile * HS;   // bf16 per K (or V) buffer stage

        float o_acc[ kMaxNTiles ][ 4 ];
#pragma unroll
        for ( int nt = 0; nt < kMaxNTiles; ++nt )
            o_acc[ nt ][ 0 ] = o_acc[ nt ][ 1 ] = o_acc[ nt ][ 2 ] = o_acc[ nt ][ 3 ] = 0.0f;

        // --- load this tile's Q rows into smem once; zero rows past chunk_len ---
        for ( int idx = tid; idx < kRowsPerTile * HS; idx += block_threads )
        {
            const int m = idx / HS;
            const int d = idx % HS;
            const int t = qrow0 + m;

            if ( t < chunk_len )
            {
                const size_t row_base = ( static_cast<size_t>( b * chunk_len + t ) * NH + nh ) * HS;
                s_Q[ idx ] = Q[ row_base + d ];
            }
            else
                s_Q[ idx ] = __float2bfloat16( 0.0f );
        }

        if ( tid < kRowsPerTile )
        {
            s_m[ tid ] = -CUDART_INF_F;
            s_l[ tid ] = 0.0f;
        }

        const int last_row = min( chunk_len - 1, qrow0 + kRowsPerTile - 1 );
        const int block_max_key = position_offset + last_row;
        const int num_tiles = block_max_key / kKeysPerTile + 1;

        // --- prologue: prefetch key tile 0 into stage 0 ---
        cp_async_kv_tile( s_K, s_V, K, V, kv_head_base, 0, HS, cache_capacity, tid, block_threads );
        __pipeline_commit();

        __syncthreads();   // Q resident + softmax state initialised before the loop

        wmma::fragment<wmma::matrix_a, kWmmaM, kWmmaN, kWmmaK, __nv_bfloat16, wmma::row_major> q_frag;
        wmma::fragment<wmma::matrix_b, kWmmaM, kWmmaN, kWmmaK, __nv_bfloat16, wmma::col_major> k_frag;
        wmma::fragment<wmma::accumulator, kWmmaM, kWmmaN, kWmmaK, float> s_frag;

        for ( int t = 0; t < num_tiles; ++t )
        {
            const int tile_start = t * kKeysPerTile;
            const int stage = t & 1;
            __nv_bfloat16* Kc = s_K + stage * stage_elems;
            __nv_bfloat16* Vc = s_V + stage * stage_elems;

            // Prefetch the next tile into the other stage, then wait for THIS tile. The
            // wait target differs on the final tile (no next prefetch in flight).
            if ( t + 1 < num_tiles )
            {
                const int next_stage = ( t + 1 ) & 1;
                cp_async_kv_tile( s_K + next_stage * stage_elems, s_V + next_stage * stage_elems,
                    K, V, kv_head_base, ( t + 1 ) * kKeysPerTile, HS, cache_capacity, tid, block_threads );
                __pipeline_commit();
                __pipeline_wait_prior( 1 );   // two groups in flight -> wait for the older (this tile)
            }
            else
                __pipeline_wait_prior( 0 );

            __syncthreads();   // this tile's K/V visible to all threads

            // --- QK^T split-K (wmma): warp w accumulates a partial S_w over its HS slice ---
            wmma::fill_fragment( s_frag, 0.0f );
            for ( int k_step = col0; k_step < col0 + hs_tile; k_step += kWmmaK )
            {
                wmma::load_matrix_sync( q_frag, s_Q + k_step, HS );
                wmma::load_matrix_sync( k_frag, Kc + k_step, HS );   // col_major -> K^T
                wmma::mma_sync( s_frag, q_frag, k_frag, s_frag );
            }
            wmma::store_matrix_sync( s_S_w, s_frag, kWmmaN, wmma::mem_row_major );

            __syncthreads();

            // --- reduce the W split-K partials into the full score tile ---
            for ( int idx = tid; idx < kRowsPerTile * kKeysPerTile; idx += block_threads )
            {
                float acc = 0.0f;
                for ( int w = 0; w < warps; ++w )
                    acc += s_S_partial[ w * kRowsPerTile * kKeysPerTile + idx ];

                s_S[ idx ] = acc;
            }

            __syncthreads();

            // --- online softmax (one lane per query row) -> P tile + per-row alpha ---
            if ( tid < kRowsPerTile )
            {
                const int r = tid;
                const int t_row = qrow0 + r;

                if ( t_row < chunk_len )
                {
                    const int abs_t = position_offset + t_row;
                    const int window_start = ( window > 0 ) ? max( 0, abs_t - window + 1 ) : 0;

                    float row_scores[ kKeysPerTile ];
                    float m_tile = -CUDART_INF_F;
                    for ( int c = 0; c < kKeysPerTile; ++c )
                    {
                        const int key = tile_start + c;
                        float s = s_S[ r * kWmmaN + c ] * scale;

                        if ( key > abs_t || key < window_start )
                            s = -CUDART_INF_F;

                        row_scores[ c ] = s;
                        m_tile = fmaxf( m_tile, s );
                    }

                    if ( m_tile > -CUDART_INF_F )
                    {
                        const float m_old = s_m[ r ];
                        const float m_new = fmaxf( m_old, m_tile );
                        const float alpha = ( m_old == -CUDART_INF_F ) ? 0.0f : expf( m_old - m_new );

                        float row_sum = 0.0f;
                        for ( int c = 0; c < kKeysPerTile; ++c )
                        {
                            const float pj = ( row_scores[ c ] == -CUDART_INF_F ) ? 0.0f : expf( row_scores[ c ] - m_new );
                            s_P[ r * kWmmaN + c ] = __float2bfloat16( pj );
                            row_sum += pj;
                        }

                        s_l[ r ] = s_l[ r ] * alpha + row_sum;
                        s_m[ r ] = m_new;
                        s_alpha[ r ] = alpha;
                    }
                    else
                    {
                        for ( int c = 0; c < kWmmaN; ++c )
                            s_P[ r * kWmmaN + c ] = __float2bfloat16( 0.0f );

                        s_alpha[ r ] = 1.0f;
                    }
                }
                else
                {
                    for ( int c = 0; c < kWmmaN; ++c )
                        s_P[ r * kWmmaN + c ] = __float2bfloat16( 0.0f );

                    s_alpha[ r ] = 1.0f;
                }
            }

            __syncthreads();

            // --- load A = P into this thread's mma fragment (same P for every warp) ---
            const uint32_t a0 = *reinterpret_cast<const uint32_t*>( &s_P[ ( g     ) * kWmmaN + 2 * tg     ] );
            const uint32_t a1 = *reinterpret_cast<const uint32_t*>( &s_P[ ( g + 8 ) * kWmmaN + 2 * tg     ] );
            const uint32_t a2 = *reinterpret_cast<const uint32_t*>( &s_P[ ( g     ) * kWmmaN + 2 * tg + 8 ] );
            const uint32_t a3 = *reinterpret_cast<const uint32_t*>( &s_P[ ( g + 8 ) * kWmmaN + 2 * tg + 8 ] );

            // --- rescale the persistent O_w registers by the per-row alpha (FA-2) ---
            const float alpha_lo = s_alpha[ g ];
            const float alpha_hi = s_alpha[ g + 8 ];
#pragma unroll
            for ( int nt = 0; nt < kMaxNTiles; ++nt )
            {
                o_acc[ nt ][ 0 ] *= alpha_lo;
                o_acc[ nt ][ 1 ] *= alpha_lo;
                o_acc[ nt ][ 2 ] *= alpha_hi;
                o_acc[ nt ][ 3 ] *= alpha_hi;
            }

            // --- PV (mma.sync): O_w[nt] += P . V[:, n-tile], accumulated in registers ---
#pragma unroll
            for ( int nt = 0; nt < kMaxNTiles; ++nt )
            {
                if ( nt >= num_ntiles )
                    continue;

                const int ncol = col0 + nt * kMmaN + g;
                const uint16_t b0lo = *reinterpret_cast<const uint16_t*>( &Vc[ ( 2 * tg     ) * HS + ncol ] );
                const uint16_t b0hi = *reinterpret_cast<const uint16_t*>( &Vc[ ( 2 * tg + 1 ) * HS + ncol ] );
                const uint16_t b1lo = *reinterpret_cast<const uint16_t*>( &Vc[ ( 2 * tg + 8 ) * HS + ncol ] );
                const uint16_t b1hi = *reinterpret_cast<const uint16_t*>( &Vc[ ( 2 * tg + 9 ) * HS + ncol ] );
                const uint32_t b0 = static_cast<uint32_t>( b0lo ) | ( static_cast<uint32_t>( b0hi ) << 16 );
                const uint32_t b1 = static_cast<uint32_t>( b1lo ) | ( static_cast<uint32_t>( b1hi ) << 16 );

                mma_m16n8k16_bf16(
                    o_acc[ nt ][ 0 ], o_acc[ nt ][ 1 ], o_acc[ nt ][ 2 ], o_acc[ nt ][ 3 ],
                    a0, a1, a2, a3, b0, b1 );
            }

            __syncthreads();   // before this stage buffer is reused by tile t+2
        }

        // --- normalize the register accumulator by 1/l and write out ---
        const int t_lo = qrow0 + g;
        const int t_hi = qrow0 + g + 8;
        const float inv_lo = ( t_lo < chunk_len ) ? ( 1.0f / s_l[ g ] ) : 0.0f;
        const float inv_hi = ( t_hi < chunk_len ) ? ( 1.0f / s_l[ g + 8 ] ) : 0.0f;

#pragma unroll
        for ( int nt = 0; nt < kMaxNTiles; ++nt )
        {
            if ( nt >= num_ntiles )
                continue;

            const int c_base = col0 + nt * kMmaN + 2 * tg;

            if ( t_lo < chunk_len )
            {
                const size_t rb = ( static_cast<size_t>( b * chunk_len + t_lo ) * NH + nh ) * HS;
                Y[ rb + c_base     ] = __float2bfloat16( o_acc[ nt ][ 0 ] * inv_lo );
                Y[ rb + c_base + 1 ] = __float2bfloat16( o_acc[ nt ][ 1 ] * inv_lo );
            }

            if ( t_hi < chunk_len )
            {
                const size_t rb = ( static_cast<size_t>( b * chunk_len + t_hi ) * NH + nh ) * HS;
                Y[ rb + c_base     ] = __float2bfloat16( o_acc[ nt ][ 2 ] * inv_hi );
                Y[ rb + c_base + 1 ] = __float2bfloat16( o_acc[ nt ][ 3 ] * inv_hi );
            }
        }
#endif // __CUDA_ARCH__ >= 800
    }

    // Shared-memory footprint for one block, matching the kernel's carve exactly. Double-
    // buffered K and V (2 stages each); no O accumulator (register-resident).
    static size_t wmma_stage2c_smem_bytes( int HS, int warps )
    {
        const size_t f32_elems =
            static_cast<size_t>( warps * kRowsPerTile * kKeysPerTile )   // S_partial
            + static_cast<size_t>( kRowsPerTile * kKeysPerTile )         // S
            + static_cast<size_t>( 3 * kRowsPerTile );                   // m, l, alpha
        const size_t bf16_elems =
            static_cast<size_t>( kRowsPerTile * HS )                     // Q
            + static_cast<size_t>( 2 * kKeysPerTile * HS )               // K (2 stages)
            + static_cast<size_t>( 2 * kKeysPerTile * HS )               // V (2 stages)
            + static_cast<size_t>( kRowsPerTile * kKeysPerTile );        // P
        return f32_elems * sizeof( float ) + bf16_elems * sizeof( __nv_bfloat16 );
    }

    void cuda_gqa_flash_prefill_bf16(
        const __nv_bfloat16* Q, const __nv_bfloat16* K, const __nv_bfloat16* V,
        __nv_bfloat16* Y,
        int B, int chunk_len, int NH, int NKV, int HS, int cache_capacity,
        int position_offset, int window, float scale,
        cudaStream_t stream )
    {
        if ( HS % kWmmaK != 0 )
            throw std::runtime_error(
                "cuda_gqa_flash_prefill_bf16 (WMMA): head_size must be a multiple of 16" );

        const int warps = flash_warp_count( HS );
        const int hs_tile = HS / warps;

        if ( hs_tile % kWmmaK != 0 )
            throw std::runtime_error(
                "cuda_gqa_flash_prefill_bf16 (WMMA): head_size not splittable into 16-wide HS tiles" );

        if ( hs_tile > kMaxNTiles * kMmaN )
            throw std::runtime_error(
                "cuda_gqa_flash_prefill_bf16 (WMMA): HS/W exceeds the register accumulator bound" );

        // cp.async 16-byte copies require HS to be a multiple of 8 (one copy stays in a row).
        if ( HS % kCopyElems != 0 )
            throw std::runtime_error(
                "cuda_gqa_flash_prefill_bf16 (WMMA): head_size must be a multiple of 8 for cp.async" );

        int device = 0;
        int sm_major = 0;
        cudaCheck( cudaGetDevice( &device ) );
        cudaCheck( cudaDeviceGetAttribute( &sm_major, cudaDevAttrComputeCapabilityMajor, device ) );

        if ( sm_major < 8 )
            throw std::runtime_error(
                "cuda_gqa_flash_prefill_bf16 (WMMA): requires compute capability >= 8.0" );

        const size_t smem_bytes = wmma_stage2c_smem_bytes( HS, warps );

        cudaCheck( cudaFuncSetAttribute(
            gqa_flash_prefill_wmma_bf16_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>( smem_bytes ) ) );

        const int num_query_tiles = ceil_div( chunk_len, kRowsPerTile );
        dim3 grid( num_query_tiles, NH, B );
        dim3 block( warps * 32, 1, 1 );

        gqa_flash_prefill_wmma_bf16_kernel <<< grid, block, smem_bytes, stream >>> (
            Q, K, V, Y,
            B, chunk_len, NH, NKV, HS, cache_capacity,
            position_offset, window, scale );

        cudaCheck( cudaGetLastError() );
    }
}
