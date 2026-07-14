// Gqa.Flash.Wmma.cu
//
// FlashAttention prefill over the compact BF16 GQA KV cache -- tensor-core path,
// Iteration 3 / Stage 2d (raw mma.sync throughout, restructured barrier pipeline).
// See GqaFlashAttention.md 5.3 / 5.3.1.
//
// LADDER RECAP. Scalar warp-per-row (Gqa.Flash.Bf16.cu) was LSU-bound. Stage 1 proved
// WMMA correctness. Stage 2a engaged tensor cores (multi-warp HS-split, O in smem).
// Stage 2b moved O into mma.sync accumulator registers. Stage 2c added the cp.async
// double-buffered K/V pipeline. Profiling 2c showed the kernel LATENCY-bound at 16.67%
// occupancy (1 block/SM, smem-limited): bank-conflict padding (34x fewer conflicts) and
// W=16 (2x occupancy) both failed to move the wall, because every key tile crosses five
// block-wide __syncthreads and all warps live in the one block -- the barrier chain IS
// the critical path.
//
// STAGE 2d attacks that chain directly:
//   - QK is raw mma.sync with MANUAL smem loads (the 2b PV pattern), replacing the
//     nvcuda::wmma fragments. The split-K partial S_w lives in registers and is stored
//     ONCE per tile with a padded row stride (kPartialStride), eliminating the
//     store_matrix_sync round-trip and its measured shared-store bank conflicts.
//   - The softmax is DISTRIBUTED across every warp (v1 put it, plus the fused split-K
//     reduction, on 16 lanes of warp 0 and measured a 57% regression -- the single-warp
//     softmax was the critical path all along). Each warp owns Br/W query rows; each
//     row's lane group sums the W partials for its own columns and reduces the row
//     max/sum with intra-group shuffles. The block-wide reduce step, its smem
//     round-trip, and one barrier are still deleted -- but the work now lands on all
//     256 threads instead of 16.
//   - The next tile's cp.async prefetch is issued AFTER the top-of-loop visibility
//     barrier, which then also proves the prior iteration fully drained -- so the
//     end-of-loop stage-protection barrier is deleted.
//   => 3 barriers per key tile (was 5), 2 smem trips for the score tile (was 3), and
//      no nvcuda::wmma anywhere. Math, masking, and the online-softmax state are
//      IDENTICAL to 2c, so the CudaGqaFlashPrefillParity oracle gates the rewrite.
//
// LADDER RUNG (post-2d-v2, fa-5090 V4): every smem->register fragment load (Q, K, P,
// and V) is a warp-collective ldmatrix.x4 -- ~68 scalar loads per thread per key tile
// (including the 2-byte strided V gathers) collapse to ~13 instructions on the
// measured-busiest L1/TEX pipe. The kSmemSkew row padding makes each ldmatrix tile's 8
// row addresses span all 32 banks, so the two changes compose. fa-5090's single-V-
// buffer rung (their V5) is deliberately NOT taken: its payoff was doubling the key
// tile with the freed smem, and at HS = 512 no Bc = 32 configuration fits the Ada smem
// cap in any buffering scheme.
//
// m16n8k16 .f32.bf16.bf16.f32 fragment layout (PTX ISA), g = lane/4, tg = lane%4:
//   A[16x16] a0..a3 (row-major): a0={A[g][2t],A[g][2t+1]} a1={A[g+8][2t],..}
//                                a2={A[g][2t+8],..}       a3={A[g+8][2t+8],..}
//   B[16x8]  b0,b1  (col-major): b0={B[2t][g],B[2t+1][g]} b1={B[2t+8][g],B[2t+9][g]}
//   C[16x8]  c0..c3 (row-major): c0=C[g][2t] c1=C[g][2t+1] c2=C[g+8][2t] c3=C[g+8][2t+1]
//
// For QK (S = Q . K^T) the B operand is K^T, so B[k][n] = K[key n][dim k]: b0 reads two
// CONSECUTIVE dims of key row g (one uint32), b1 the same row 8 dims later -- the manual
// loads stay 4-byte aligned and stride the skew-padded smem row.
//
// TWO CACHE VARIANTS, one skeleton (kBoundedRing template axis, if constexpr deltas only):
//   - Unbounded (kBoundedRing == false): physical cache row == absolute position, OOB key
//     rows clamp to the last valid row (masked downstream). window == 0 for Gemma global /
//     Llama full causal; window > 0 is honored by the masking either way.
//   - Bounded sliding-window ring (kBoundedRing == true): the KV cache holds
//     cache_capacity = window + prefill_chunk - 1 rows and physical row = abs_pos %
//     cache_capacity. The kernel iterates ABSOLUTE key positions (the ring mapping lives
//     only in the cp.async source math), so the score/mask/softmax logic is byte-identical
//     to the unbounded variant. Ring slots aliased by newer or older absolute positions
//     land exclusively on columns the per-row causal + window mask already forces to -inf:
//     stale-below-band columns only occur under the block-wide band minimum (masked for
//     every row), and future-position columns are causally masked. The band-start loop
//     bound makes the key loop CONSTANT (~window/Bc + 1 tiles) instead of triangular.
//
// HS a multiple of 16, HSt = HS/W a multiple of 16, HSt <= kMaxNTiles*8.

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <math_constants.h>
#include <cstdint>
#include <stdexcept>
#include <cuda_pipeline.h>
#include "CudaUtils.h"
#include "CudaGqa.cuh"

namespace Mila::Dnn::Compute::Cuda::Gqa
{
    namespace
    {
        constexpr int kRowsPerTile = 16;       // Br: query rows per block tile (MMA M)
        constexpr int kKeysPerTile = 16;       // Bc: key positions per tile
        constexpr int kMmaK = 16;              // contraction step (m16n8k16 K)
        constexpr int kMmaN = 8;               // mma.sync N tile (m16n8k16)
        constexpr int kMaxNTiles = 8;          // HSt / kMmaN ceiling -> HSt <= 64
        constexpr int kCopyElems = 8;          // bf16 elements per 16-byte cp.async copy
        constexpr int kKeyNTiles = kKeysPerTile / kMmaN;   // n-tiles per QK score tile

        // The QK inner loop loads both key n-tiles with a single ldmatrix.x4 and issues
        // exactly two mma calls; a different Bc would need that block generalized.
        static_assert( kKeyNTiles == 2, "QK ldmatrix.x4 layout assumes Bc == 16" );

        // Per-row smem padding (bf16 elements) for the Q/K/V tiles. At HS = 512 an
        // unpadded row strides 1024 B -- an exact multiple of the 128-B shared-memory
        // bank cycle, so every tile row maps onto the same 32 banks (ncu measured 944M
        // shared-load bank conflicts). Padding by 8 bf16 (16 B) shifts consecutive rows
        // 4 banks apart while keeping every row start 16-B aligned for cp.async.
        // Storage stride only: global reads and the math are unchanged. 0 restores the
        // exact unpadded layout (the A/B kill switch).
        constexpr int kSmemSkew = 8;

        static_assert( kSmemSkew % kCopyElems == 0,
            "kSmemSkew must be a multiple of the cp.async copy width" );

        // Row stride of the per-warp split-K score partials in smem. +2 keeps the
        // stride EVEN so the QK result stores can be 8-byte float2 (c0,c1 and c2,c3 are
        // column-adjacent), while 18*r mod 32 is distinct for all 16 rows -- the
        // softmax lanes' column reads stay bank-conflict-free (an unpadded 16-float
        // stride puts every other row on the same banks).
        constexpr int kPartialStride = kKeysPerTile + 2;

        // W ceiling. 16 was MEASURED WORSE (2026-07-13, heavy instance 22.8 -> 24.8-25.5
        // ms despite occupancy doubling 16.67 -> 33.3%): every warp lives in the ONE
        // smem-limited block and the kernel barriers block-wide per tile, so extra warps
        // do not hide latency across __syncthreads -- they halve per-warp work per
        // barrier interval and double the split-K reduction traffic. Raising this only
        // pays if the per-tile barrier count drops further still.
        constexpr int kMaxWarps = 8;

        // W ceiling for the bounded-ring variant. At its production geometry (Gemma local
        // sliding, HS = 256) W = 4 keeps the block under half the Ada per-SM shared memory
        // (47552 B + the 1 KB per-block reservation, vs 102400 B/SM), so TWO independent
        // blocks co-reside per SM -- the occupancy regime the HS = 512 global variant cannot
        // reach with any double-buffer. The per-warp tile shape (hs_tile 64, 8 n-tiles) is
        // identical to the proven HS = 512 / W = 8 shape. W = 8 at HS = 256 would cost
        // 52160 B -> 1 block/SM: 8 warps sharing every block-wide barrier, the axis the
        // W = 16 experiment measured as a regression.
        constexpr int kMaxWarpsBounded = 4;

        __host__ __device__ inline int flash_warp_count( int HS, int max_warps )
        {
            int W = max_warps;

            while ( W > 1 && ( HS % W != 0 || ( HS / W ) % kMmaK != 0 ) )
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

        // Generic -> shared-state-space conversion via explicit PTX, NOT the
        // __cvta_generic_to_shared intrinsic: this build shipped ldmatrix the raw
        // GENERIC address (compute-sanitizer showed faults at generic-window base
        // 0x5c000000 + the expected smem offset), which is an illegal shared access at
        // scale -- and, insidiously, wraps to the CORRECT data at small offsets, so the
        // parity oracle stayed green over the bug. CUTLASS's cast_smem_ptr_to_uint uses
        // this same raw-PTX form for exactly this reason.
        __device__ __forceinline__ uint32_t shared_address( const void* pointer )
        {
            uint32_t address;
            asm volatile(
                "{ .reg .u64 address64; cvta.to.shared.u64 address64, %1; cvt.u32.u64 %0, address64; }\n"
                : "=r"( address ) : "l"( pointer ) );

            return address;
        }

        // ldmatrix loads four 8x8 bf16 tiles in ONE warp-collective instruction: lanes
        // 0-7 / 8-15 / 16-23 / 24-31 each supply one tile's row addresses (16-byte
        // aligned), and every returned register lands pre-distributed in the mma
        // fragment layout. Replaces the per-fragment scalar loads (the 2-byte strided
        // V gathers especially), cutting shared-load instruction count ~5x on the
        // measured-busiest L1TEX pipe. The .trans variant transposes each tile at load
        // (row-major V tile -> col-major B fragment).
        // The .shared qualifier is LOAD-BEARING: without it PTX uses generic
        // addressing, and the cvta-converted shared OFFSET below gets dereferenced as
        // a generic address -- illegal reads that happen to alias correct data at
        // small offsets (green oracle) and crash at scale. cvta + .shared together are
        // the canonical CUTLASS pattern.
        __device__ __forceinline__ void ldmatrix_x4(
            uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, uint32_t address )
        {
            asm volatile( "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                : "=r"( r0 ), "=r"( r1 ), "=r"( r2 ), "=r"( r3 ) : "r"( address ) );
        }

        __device__ __forceinline__ void ldmatrix_x4_trans(
            uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, uint32_t address )
        {
            asm volatile( "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                : "=r"( r0 ), "=r"( r1 ), "=r"( r2 ), "=r"( r3 ) : "r"( address ) );
        }

        // Prefetch one key tile's K and V ([Bc x HS] BF16 each) from global into the given
        // smem stage via 16-byte cp.async copies. The tile is addressed by ABSOLUTE key
        // position; the cache-row mapping is the single point where the two variants differ.
        // Unbounded: OOB key rows clamp to the last valid cache row. Bounded ring: row =
        // abs_pos % cache_capacity -- always in-bounds, and any slot whose resident absolute
        // position differs from the requested one is a column the causal/window mask forces
        // to -inf (see the file header). Caller commits + waits. Global rows stride HS;
        // smem rows stride the padded hs_pad (kSmemSkew).
        template<bool kBoundedRing>
        __device__ __forceinline__ void cp_async_kv_tile(
            __nv_bfloat16* s_K_stage, __nv_bfloat16* s_V_stage,
            const __nv_bfloat16* K, const __nv_bfloat16* V,
            size_t kv_head_base, int tile_start, int HS, int hs_pad, int cache_capacity,
            int tid, int block_threads )
        {
            const int chunks_per_row = HS / kCopyElems;
            const int num_chunks = kKeysPerTile * chunks_per_row;

            for ( int c = tid; c < num_chunks; c += block_threads )
            {
                const int p_local = c / chunks_per_row;
                const int d0 = ( c % chunks_per_row ) * kCopyElems;
                const int p = tile_start + p_local;
                int p_src;

                if constexpr ( kBoundedRing )
                    p_src = p % cache_capacity;
                else
                    p_src = ( p < cache_capacity ) ? p : ( cache_capacity - 1 );

                const size_t g_off = kv_head_base + static_cast<size_t>( p_src ) * HS + d0;
                const int s_off = p_local * hs_pad + d0;

                __pipeline_memcpy_async( s_K_stage + s_off, K + g_off, 16 );
                __pipeline_memcpy_async( s_V_stage + s_off, V + g_off, 16 );
            }
        }
#endif
    }

    // Stage 2d: a block of W warps owns (batch, query-head, 16-row query tile) and splits
    // the head dimension (warp w owns output columns [w*HSt, (w+1)*HSt)). K and V key tiles
    // are DOUBLE-BUFFERED in smem and prefetched with cp.async one tile ahead. Q is resident
    // for the whole key loop; O_w and the split-K QK partials are register-resident
    // (mma.sync). Three block-wide barriers per key tile. kBoundedRing selects the
    // sliding-window ring cache-row mapping (see the file header); the math is otherwise
    // identical.
    template<bool kBoundedRing>
    __global__ void gqa_flash_prefill_mma_bf16_kernel(
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
        const int hs_pad = HS + kSmemSkew;   // padded smem row stride (bank de-conflict)

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
        float* s_S_partial = reinterpret_cast<float*>( smem_raw );   // [W][Br * kPartialStride]
        float* s_m = s_S_partial + warps * kRowsPerTile * kPartialStride;   // [Br]
        float* s_l = s_m + kRowsPerTile;                                    // [Br]
        float* s_alpha = s_l + kRowsPerTile;                                // [Br]
        __nv_bfloat16* s_Q =
            reinterpret_cast<__nv_bfloat16*>( s_alpha + kRowsPerTile );   // [Br * hs_pad]
        __nv_bfloat16* s_K = s_Q + kRowsPerTile * hs_pad;                 // [2][Bc * hs_pad]
        __nv_bfloat16* s_V = s_K + 2 * kKeysPerTile * hs_pad;             // [2][Bc * hs_pad]
        __nv_bfloat16* s_P = s_V + 2 * kKeysPerTile * hs_pad;             // [Br * Bc]

        float* s_S_w = s_S_partial + warp_id * kRowsPerTile * kPartialStride;

        const int stage_elems = kKeysPerTile * hs_pad;   // bf16 per K (or V) buffer stage

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
                s_Q[ m * hs_pad + d ] = Q[ row_base + d ];
            }
            else
                s_Q[ m * hs_pad + d ] = __float2bfloat16( 0.0f );
        }

        if ( tid < kRowsPerTile )
        {
            s_m[ tid ] = -CUDART_INF_F;
            s_l[ tid ] = 0.0f;
        }

        const int last_row = min( chunk_len - 1, qrow0 + kRowsPerTile - 1 );
        const int block_max_key = position_offset + last_row;
        const int num_tiles = block_max_key / kKeysPerTile + 1;

        // First key tile of the attended band. window_start grows with the row index, so
        // the block-wide minimum is the FIRST row's -- every column in an earlier tile sits
        // below every row's window and would be masked to -inf; skipping those tiles makes
        // the sliding-window key loop CONSTANT (~window/Bc + 1 tiles) instead of causal-
        // triangular. window == 0 (global) keeps first_tile == 0 and the loop unchanged.
        const int band_start =
            ( window > 0 ) ? max( 0, position_offset + qrow0 - window + 1 ) : 0;
        const int first_tile = band_start / kKeysPerTile;

        // --- prologue: prefetch the first attended key tile into its parity stage ---
        cp_async_kv_tile<kBoundedRing>( s_K + ( first_tile & 1 ) * stage_elems,
            s_V + ( first_tile & 1 ) * stage_elems,
            K, V, kv_head_base, first_tile * kKeysPerTile, HS, hs_pad, cache_capacity,
            tid, block_threads );
        __pipeline_commit();

        for ( int t = first_tile; t < num_tiles; ++t )
        {
            const int tile_start = t * kKeysPerTile;
            const int stage = t & 1;
            __nv_bfloat16* Kc = s_K + stage * stage_elems;
            __nv_bfloat16* Vc = s_V + stage * stage_elems;

            // Exactly one copy group (this tile's) is in flight at the top of every
            // iteration: the next prefetch is issued only after the barrier below.
            __pipeline_wait_prior( 0 );

            // Barrier 1 of 3: this tile's K/V (and, on the first iteration, s_Q and the
            // softmax state) visible to all threads. Because every thread has fully finished
            // iteration t-1 to get here, it ALSO proves the stage buffer the next
            // prefetch writes (tile t+1 -> stage of t-1) has no remaining readers --
            // the former end-of-loop barrier is subsumed.
            __syncthreads();

            if ( t + 1 < num_tiles )
            {
                const int next_stage = ( t + 1 ) & 1;
                cp_async_kv_tile<kBoundedRing>( s_K + next_stage * stage_elems, s_V + next_stage * stage_elems,
                    K, V, kv_head_base, ( t + 1 ) * kKeysPerTile, HS, hs_pad, cache_capacity, tid, block_threads );
                __pipeline_commit();
            }

            // --- QK^T split-K (mma.sync): warp w accumulates a partial S_w over its HS
            // slice in registers, then stores it once with the padded stride ---
            float s_acc[ kKeyNTiles ][ 4 ];
#pragma unroll
            for ( int nt = 0; nt < kKeyNTiles; ++nt )
                s_acc[ nt ][ 0 ] = s_acc[ nt ][ 1 ] = s_acc[ nt ][ 2 ] = s_acc[ nt ][ 3 ] = 0.0f;

            for ( int k_step = col0; k_step < col0 + hs_tile; k_step += kMmaK )
            {
                // A = Q[16 x 16]: lanes 0-15 address rows 0-15 at the low dim half,
                // lanes 16-31 the high half -> r0..r3 land as a0..a3.
                uint32_t a0, a1, a2, a3;
                ldmatrix_x4( a0, a1, a2, a3, shared_address(
                    &s_Q[ ( lane & 15 ) * hs_pad + k_step + ( lane >> 4 ) * kMmaN ] ) );

                // B = K^T for BOTH key n-tiles in one x4: lane groups supply (keys 0-7,
                // dim-lo), (keys 0-7, dim-hi), (keys 8-15, dim-lo), (keys 8-15, dim-hi)
                // -> {b0,b1} of n-tile 0 then n-tile 1.
                uint32_t b00, b01, b10, b11;
                {
                    const int key_row = ( lane >> 4 ) * kMmaN + ( lane & 7 );
                    const int dim_half = ( ( lane >> 3 ) & 1 ) * kMmaN;
                    ldmatrix_x4( b00, b01, b10, b11, shared_address(
                        &Kc[ key_row * hs_pad + k_step + dim_half ] ) );
                }

                mma_m16n8k16_bf16(
                    s_acc[ 0 ][ 0 ], s_acc[ 0 ][ 1 ], s_acc[ 0 ][ 2 ], s_acc[ 0 ][ 3 ],
                    a0, a1, a2, a3, b00, b01 );
                mma_m16n8k16_bf16(
                    s_acc[ 1 ][ 0 ], s_acc[ 1 ][ 1 ], s_acc[ 1 ][ 2 ], s_acc[ 1 ][ 3 ],
                    a0, a1, a2, a3, b10, b11 );
            }

            // c0,c1 (and c2,c3) are column-adjacent -> one 8-byte store each; the even
            // kPartialStride and even column base keep them float2-aligned.
#pragma unroll
            for ( int nt = 0; nt < kKeyNTiles; ++nt )
            {
                const int c_base = nt * kMmaN + 2 * tg;
                *reinterpret_cast<float2*>( &s_S_w[ ( g     ) * kPartialStride + c_base ] ) =
                    make_float2( s_acc[ nt ][ 0 ], s_acc[ nt ][ 1 ] );
                *reinterpret_cast<float2*>( &s_S_w[ ( g + 8 ) * kPartialStride + c_base ] ) =
                    make_float2( s_acc[ nt ][ 2 ], s_acc[ nt ][ 3 ] );
            }

            // Barrier 2 of 3: all warps' score partials stored.
            __syncthreads();

            // --- online softmax, DISTRIBUTED across every warp (2d-v1 lesson: putting
            // the reduction + softmax on 16 lanes of warp 0 serialized ~1.5k cycles per
            // tile while 7.5 warps idled -- the single-warp softmax IS the critical
            // path). Each warp owns kRowsPerTile/W query rows; each row's lane group
            // sums the W split-K partials for its own columns (the fused cross-warp
            // reduction, now 8/W loads per lane in parallel), then reduces the row max
            // and sum with intra-group shuffles. The shuffles sit OUTSIDE all branches
            // (row groups within a warp may diverge on masking), so the full-warp mask
            // stays legal; rows past chunk_len flow through the same path with all
            // scores forced to -inf. ---
            {
                const int rows_per_warp = kRowsPerTile / warps;
                const int lanes_per_row = 32 / rows_per_warp;
                const int cols_per_lane = kKeysPerTile / lanes_per_row;
                const int r = warp_id * rows_per_warp + lane / lanes_per_row;
                const int lane_in_row = lane % lanes_per_row;
                const int t_row = qrow0 + r;

                const int abs_t = position_offset + t_row;
                const int window_start = ( window > 0 ) ? max( 0, abs_t - window + 1 ) : 0;
                const bool row_active = ( t_row < chunk_len );

                float col_scores[ kKeysPerTile / 2 ];   // cols_per_lane <= Bc/2 (W >= 1)
                float m_tile = -CUDART_INF_F;
                for ( int cc = 0; cc < cols_per_lane; ++cc )
                {
                    const int c = lane_in_row * cols_per_lane + cc;

                    float acc = 0.0f;
                    for ( int w = 0; w < warps; ++w )
                        acc += s_S_partial[ ( w * kRowsPerTile + r ) * kPartialStride + c ];

                    float s = acc * scale;

                    const int key = tile_start + c;
                    if ( !row_active || key > abs_t || key < window_start )
                        s = -CUDART_INF_F;

                    col_scores[ cc ] = s;
                    m_tile = fmaxf( m_tile, s );
                }

                for ( int off = lanes_per_row >> 1; off >= 1; off >>= 1 )
                    m_tile = fmaxf( m_tile, __shfl_xor_sync( 0xffffffffu, m_tile, off ) );

                // m_new stays m_old on a fully-masked tile so the exp guards below see
                // -inf scores and produce exact zeros (never -inf minus -inf).
                const bool has_scores = ( m_tile > -CUDART_INF_F );
                const float m_old = s_m[ r ];
                const float m_new = has_scores ? fmaxf( m_old, m_tile ) : m_old;

                float lane_sum = 0.0f;
                for ( int cc = 0; cc < cols_per_lane; ++cc )
                {
                    const int c = lane_in_row * cols_per_lane + cc;
                    // __expf (MUFU.EX2 fast path) over expf: the softmax section is the
                    // measured critical path and P is rounded to bf16 anyway; fa-5090
                    // ships the same choice.
                    const float pj = ( col_scores[ cc ] == -CUDART_INF_F )
                        ? 0.0f : __expf( col_scores[ cc ] - m_new );
                    s_P[ r * kKeysPerTile + c ] = __float2bfloat16( pj );
                    lane_sum += pj;
                }

                for ( int off = lanes_per_row >> 1; off >= 1; off >>= 1 )
                    lane_sum += __shfl_xor_sync( 0xffffffffu, lane_sum, off );

                if ( lane_in_row == 0 )
                {
                    if ( has_scores )
                    {
                        const float alpha = ( m_old == -CUDART_INF_F ) ? 0.0f : __expf( m_old - m_new );
                        s_l[ r ] = s_l[ r ] * alpha + lane_sum;
                        s_m[ r ] = m_new;
                        s_alpha[ r ] = alpha;
                    }
                    else
                        s_alpha[ r ] = 1.0f;
                }
            }

            // Barrier 3 of 3: P tile and per-row alpha ready for every warp's PV.
            __syncthreads();

            // --- load A = P into this thread's mma fragment (same P for every warp) ---
            uint32_t a0, a1, a2, a3;
            ldmatrix_x4( a0, a1, a2, a3, shared_address(
                &s_P[ ( lane & 15 ) * kKeysPerTile + ( lane >> 4 ) * kMmaN ] ) );

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

            // --- PV (mma.sync): O_w[nt] += P . V[:, n-tile], accumulated in registers.
            // One ldmatrix.x4.trans loads V's B fragments for TWO n-tiles (lanes 0-15
            // address key rows 0-15 at the first n-tile's columns, lanes 16-31 at the
            // second's; .trans turns the row-major V tile into the col-major fragment),
            // replacing 32 strided 2-byte gathers per pair. num_ntiles is always even
            // (HSt is a multiple of 16), so pairs never split. ---
#pragma unroll
            for ( int nt = 0; nt < kMaxNTiles; nt += 2 )
            {
                if ( nt >= num_ntiles )
                    continue;

                uint32_t b00, b01, b10, b11;
                ldmatrix_x4_trans( b00, b01, b10, b11, shared_address(
                    &Vc[ ( lane & 15 ) * hs_pad + col0 + nt * kMmaN + ( lane >> 4 ) * kMmaN ] ) );

                mma_m16n8k16_bf16(
                    o_acc[ nt ][ 0 ], o_acc[ nt ][ 1 ], o_acc[ nt ][ 2 ], o_acc[ nt ][ 3 ],
                    a0, a1, a2, a3, b00, b01 );
                mma_m16n8k16_bf16(
                    o_acc[ nt + 1 ][ 0 ], o_acc[ nt + 1 ][ 1 ], o_acc[ nt + 1 ][ 2 ], o_acc[ nt + 1 ][ 3 ],
                    a0, a1, a2, a3, b10, b11 );
            }

            // No end-of-loop barrier: the next iteration's top barrier subsumes it.
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
    // buffered K and V (2 stages each); no O accumulator and no reduced score tile (both
    // register-resident / fused). Q/K/V rows stride the skew-padded width.
    static size_t flash_stage2d_smem_bytes( int HS, int warps )
    {
        const size_t hs_pad = static_cast<size_t>( HS + kSmemSkew );
        const size_t f32_elems =
            static_cast<size_t>( warps * kRowsPerTile * kPartialStride )   // S_partial
            + static_cast<size_t>( 3 * kRowsPerTile );                     // m, l, alpha
        const size_t bf16_elems =
            static_cast<size_t>( kRowsPerTile ) * hs_pad                   // Q
            + static_cast<size_t>( 2 * kKeysPerTile ) * hs_pad             // K (2 stages)
            + static_cast<size_t>( 2 * kKeysPerTile ) * hs_pad             // V (2 stages)
            + static_cast<size_t>( kRowsPerTile * kKeysPerTile );          // P
        return f32_elems * sizeof( float ) + bf16_elems * sizeof( __nv_bfloat16 );
    }

    template<bool kBoundedRing>
    static void launch_flash_prefill_bf16(
        const __nv_bfloat16* Q, const __nv_bfloat16* K, const __nv_bfloat16* V,
        __nv_bfloat16* Y,
        int B, int chunk_len, int NH, int NKV, int HS, int cache_capacity,
        int position_offset, int window, float scale,
        cudaStream_t stream )
    {
        if ( HS % kMmaK != 0 )
            throw std::runtime_error(
                "cuda_gqa_flash_prefill_bf16 (MMA): head_size must be a multiple of 16" );

        const int warps =
            flash_warp_count( HS, kBoundedRing ? kMaxWarpsBounded : kMaxWarps );
        const int hs_tile = HS / warps;

        if ( hs_tile % kMmaK != 0 )
            throw std::runtime_error(
                "cuda_gqa_flash_prefill_bf16 (MMA): head_size not splittable into 16-wide HS tiles" );

        if ( hs_tile > kMaxNTiles * kMmaN )
            throw std::runtime_error(
                "cuda_gqa_flash_prefill_bf16 (MMA): HS/W exceeds the register accumulator bound" );

        // cp.async 16-byte copies require HS to be a multiple of 8 (one copy stays in a row).
        if ( HS % kCopyElems != 0 )
            throw std::runtime_error(
                "cuda_gqa_flash_prefill_bf16 (MMA): head_size must be a multiple of 8 for cp.async" );

        if ( kBoundedRing && window <= 0 )
            throw std::runtime_error(
                "cuda_gqa_flash_prefill_ring_bf16 (MMA): bounded ring requires a positive window" );

        int device = 0;
        int sm_major = 0;
        cudaCheck( cudaGetDevice( &device ) );
        cudaCheck( cudaDeviceGetAttribute( &sm_major, cudaDevAttrComputeCapabilityMajor, device ) );

        if ( sm_major < 8 )
            throw std::runtime_error(
                "cuda_gqa_flash_prefill_bf16 (MMA): requires compute capability >= 8.0" );

        const size_t smem_bytes = flash_stage2d_smem_bytes( HS, warps );

        cudaCheck( cudaFuncSetAttribute(
            gqa_flash_prefill_mma_bf16_kernel<kBoundedRing>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>( smem_bytes ) ) );

        const int num_query_tiles = ceil_div( chunk_len, kRowsPerTile );
        dim3 grid( num_query_tiles, NH, B );
        dim3 block( warps * 32, 1, 1 );

        gqa_flash_prefill_mma_bf16_kernel<kBoundedRing> <<< grid, block, smem_bytes, stream >>> (
            Q, K, V, Y,
            B, chunk_len, NH, NKV, HS, cache_capacity,
            position_offset, window, scale );

        cudaCheck( cudaGetLastError() );
    }

    void cuda_gqa_flash_prefill_bf16(
        const __nv_bfloat16* Q, const __nv_bfloat16* K, const __nv_bfloat16* V,
        __nv_bfloat16* Y,
        int B, int chunk_len, int NH, int NKV, int HS, int cache_capacity,
        int position_offset, int window, float scale,
        cudaStream_t stream )
    {
        launch_flash_prefill_bf16<false>(
            Q, K, V, Y, B, chunk_len, NH, NKV, HS, cache_capacity,
            position_offset, window, scale, stream );
    }

    // cuda_gqa_flash_prefill_ring_bf16 (the bounded sliding-window local layers) now lives
    // in Gqa.Flash.Fa2.cu: at HS = 256 the local layers use a row-split FA-2 kernel
    // (register-resident O + P, one barrier per tile) that the HS = 512 global variant
    // here cannot -- see that file's header. This unit owns only the unbounded global
    // symbol, so launch_flash_prefill_bf16<true> is no longer instantiated.
}
